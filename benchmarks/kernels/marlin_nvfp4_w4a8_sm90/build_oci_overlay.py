#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build the NVFP4 hybrid operator against an installed vLLM package."""

import argparse
import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, load

import vllm


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()

source = args.source.resolve()
output = args.output.resolve()
assert not output.exists(), output

installed = Path(vllm.__file__).resolve().parent
overlay = output / "overlay" / "vllm"
build = output / "converter"
shutil.copytree(installed, overlay)
shutil.copytree(source / "vllm", overlay, dirs_exist_ok=True)
build.mkdir(parents=True)

cutlass = installed / "third_party" / "fmha_sm100" / "cutlass"
cutlass_include = cutlass / "include"
cutlass_tools_include = cutlass / "tools" / "util" / "include"
assert cutlass_include.is_dir(), cutlass_include
assert cutlass_tools_include.is_dir(), cutlass_tools_include
assert CUDA_HOME is not None
cuda_home = Path(CUDA_HOME)
cuda_cccl_include = cuda_home / "include" / "cccl"
cuda_stubs = cuda_home / "lib64" / "stubs"
assert cuda_cccl_include.is_dir(), cuda_cccl_include
assert cuda_stubs.is_dir(), cuda_stubs

marlin = source / "csrc" / "libtorch_stable" / "quantization" / "marlin"
scaled_mm = source / "csrc" / "libtorch_stable" / "quantization" / "w8a8" / "cutlass"
subprocess.run(
    [sys.executable, str(marlin / "generate_kernels.py"), "9.0a"], check=True
)
marlin_kernels = sorted(marlin.glob("sm80_kernel_*.cu"))
assert len(marlin_kernels) == 14, marlin_kernels
sources = [
    marlin / "marlin_nvfp4_to_fp8.cu",
    source
    / "csrc"
    / "libtorch_stable"
    / "quantization"
    / "w8a8"
    / "fp8"
    / "per_token_group_quant.cu",
    marlin / "marlin.cu",
    *marlin_kernels,
    source / "csrc" / "libtorch_stable" / "cutlass_extensions" / "common.cpp",
    scaled_mm / "scaled_mm_entry.cu",
    scaled_mm / "scaled_mm_c3x_sm90.cu",
    scaled_mm / "c3x" / "scaled_mm_sm90_fp8.cu",
    scaled_mm / "c3x" / "scaled_mm_sm90_int8.cu",
    scaled_mm / "c3x" / "scaled_mm_azp_sm90_int8.cu",
    scaled_mm / "c3x" / "scaled_mm_blockwise_sm90_fp8.cu",
]
assert len(sources) == 24, sources

definitions = [
    "-DTORCH_TARGET_VERSION=0x020B000000000000ULL",
    "-DUSE_CUDA",
    "-DENABLE_SCALED_MM_SM90=1",
    "-DCUTLASS_ENABLE_DIRECT_CUDA_DRIVER_CALL=1",
    "-DVLLM_MARLIN_NO_DISPATCH_REGISTRATION=1",
]

base_dso = overlay / "_C_stable_libtorch.abi3.so"
try:
    ctypes.CDLL(str(base_dso), mode=ctypes.RTLD_GLOBAL)
except OSError as error:
    cuda_stub = Path("/usr/local/cuda/lib64/stubs/libcuda.so")
    missing_libcuda = "libcuda.so.1: cannot open shared object file" in str(error)
    if not missing_libcuda or not cuda_stub.is_file():
        raise
    ctypes.CDLL(str(cuda_stub), mode=ctypes.RTLD_GLOBAL)
    ctypes.CDLL(str(base_dso), mode=ctypes.RTLD_GLOBAL)
assert not hasattr(torch.ops._C, "marlin_nvfp4_to_fp8")

os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
load(
    name="marlin_nvfp4_to_fp8_sm90a",
    sources=[str(path) for path in sources],
    extra_include_paths=[
        str(source / "csrc"),
        str(marlin),
        str(cutlass_include),
        str(cutlass_tools_include),
        str(cuda_cccl_include),
    ],
    extra_cflags=["-O3", "-std=c++20", *definitions],
    extra_cuda_cflags=[
        "-O3",
        "--expt-relaxed-constexpr",
        "-std=c++20",
        "-static-global-template-stub=false",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        *definitions,
    ],
    extra_ldflags=[f"-L{cuda_stubs}", "-lcuda"],
    build_directory=str(build),
    with_cuda=True,
    is_python_module=False,
    verbose=True,
)

extension = build / "marlin_nvfp4_to_fp8_sm90a.so"
assert extension.is_file()
assert hasattr(torch.ops._C, "marlin_nvfp4_to_fp8")
assert hasattr(torch.ops._C, "marlin_nvfp4_hybrid_linear")

manifest = {
    "source_revision": os.environ["VLLM_SOURCE_REVISION"],
    "torch": torch.__version__,
    "vllm": vllm.__version__,
    "base_dso": {"path": str(base_dso), "sha256": sha256(base_dso)},
    "extension": {"path": str(extension), "sha256": sha256(extension)},
}
(output / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n"
)
print(json.dumps(manifest, sort_keys=True), flush=True)
