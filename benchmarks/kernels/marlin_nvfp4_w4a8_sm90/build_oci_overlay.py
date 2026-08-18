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
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

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
    sources=[
        str(source / "csrc/libtorch_stable/quantization/marlin/marlin_nvfp4_to_fp8.cu")
    ],
    extra_include_paths=[
        str(source / "csrc"),
        str(source / "csrc/libtorch_stable/quantization/marlin"),
    ],
    extra_cflags=["-std=c++20"],
    extra_cuda_cflags=[
        "-O3",
        "--expt-relaxed-constexpr",
        "-DUSE_CUDA",
        "-std=c++20",
    ],
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
