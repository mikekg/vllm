# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build DeepGEMM's `_C` pybind11 extension for <TARGET_PY>.

Driven from cmake/external_projects/deepgemm.cmake. The driver runs against
the build interpreter's torch; <TARGET_PY> is only consulted for INCLUDEPY
and SOABI, so target venvs don't need torch installed.

Usage: python build_deepgemm_C.py <DEEPGEMM_SRC_DIR> <OUTPUT_DIR> <TARGET_PY>
"""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from cuda.pathfinder import find_nvidia_header_directory
from torch.utils import cpp_extension

if len(sys.argv) != 4:
    sys.exit(f"usage: {sys.argv[0]} <SRC> <OUT> <TARGET_PY>")

src = Path(sys.argv[1]).resolve()
out = Path(sys.argv[2]).resolve()
target_py = sys.argv[3]
vllm_root = Path(__file__).resolve().parent.parent
out.mkdir(parents=True, exist_ok=True)

patched_csrc = out / "deepgemm-csrc"
shutil.copytree(src / "csrc", patched_csrc, dirs_exist_ok=True)
compiler = patched_csrc / "jit" / "compiler.hpp"
compiler_source = compiler.read_text()
# NVRTC omits device-runtime declarations that NVCC pre-includes.
nvrtc_flags = ' --device-int128",'
assert compiler_source.count(nvrtc_flags) == 1
compiler_source = compiler_source.replace(
    nvrtc_flags,
    ' --device-int128 --pre-include=cuda_device_runtime_api.h",',
)
compile_log = '                printf("NVRTC log: %s\\n", compilation_log.c_str());\n'
compile_log += "            }\n        }\n"
assert compiler_source.count(compile_log) == 1
compiler_source = compiler_source.replace(
    compile_log,
    compile_log + "        DG_NVRTC_CHECK(compile_result);\n",
)
compiler.write_text(compiler_source)

info = json.loads(
    subprocess.check_output(
        [
            target_py,
            "-c",
            "import sysconfig, json; "
            "print(json.dumps({k: sysconfig.get_config_var(k) "
            "for k in ('EXT_SUFFIX', 'INCLUDEPY')}))",
        ]
    ).decode()
)

cuda_home = cpp_extension.CUDA_HOME
if cuda_home is None:
    sys.exit("CUDA_HOME not found; cannot build DeepGEMM _C")
cusparse_include = find_nvidia_header_directory("cusparse")
if cusparse_include is None:
    sys.exit("cuSPARSE headers not found; cannot build DeepGEMM _C")
# CCCL lives outside the standard CUDAToolkit search (mirrors DeepGEMM's setup.py).
includes = [
    info["INCLUDEPY"],
    f"{cuda_home}/include",
    f"{cuda_home}/include/cccl",
    cusparse_include,
    str(vllm_root / "csrc"),
    str(patched_csrc),
    str(src / "deep_gemm/include"),
    str(src / "third-party/cutlass/include"),
    str(src / "third-party/cutlass/tools/util/include"),
    str(src / "third-party/fmt/include"),
    *cpp_extension.include_paths(device_type="cuda"),
]

cmd = [
    os.environ.get("CXX", "g++"),
    "-shared",
    "-fPIC",
    "-std=c++20",
    "-O3",
    "-g0",
    "-Wno-psabi",
    "-Wno-deprecated-declarations",
    "-DTORCH_API_INCLUDE_EXTENSION_H",
    "-DTORCH_EXTENSION_NAME=_C",
    f"-D_GLIBCXX_USE_CXX11_ABI={int(torch.compiled_with_cxx11_abi())}",
    *(f"-I{p}" for p in includes),
    str(vllm_root / "csrc/deepgemm_torch_bindings.cpp"),
    *(f"-L{p}" for p in cpp_extension.library_paths(device_type="cuda")),
    f"-L{cuda_home}/lib64",
    "-ltorch",
    "-ltorch_python",
    "-ltorch_cpu",
    "-ltorch_cuda",
    "-lc10",
    "-lc10_cuda",
    "-lcudart",
    "-lnvrtc",
    "-o",
    str(out / f"_C{info['EXT_SUFFIX']}"),
]
print("[build_deepgemm_C] " + " ".join(cmd), flush=True)
subprocess.check_call(cmd)
