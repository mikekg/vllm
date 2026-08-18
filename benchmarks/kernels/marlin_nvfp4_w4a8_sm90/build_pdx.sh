#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
set -euo pipefail

root=${S41_ROOT:?set S41_ROOT to the immutable PDX artifact root}
archive=$root/source/vllm-w4a8-hopper-b13ca6b020.tar
archive_sha=bcce6c800d5380c0bc05887076a43240c8ead01170268cf1bd0c13764234dc13
source=$root/source/tree
artifact=$root/artifacts/full-package
build_source=$artifact/source
venv=$artifact/venv
tooling=$root/tooling

actual_sha=$(sha256sum "$archive" | awk '{print $1}')
[[ $actual_sha == "$archive_sha" ]]
[[ -d $source/.deps/cutlass-src ]]
[[ ! -e $artifact ]]

mkdir -p "$artifact"
cp -a --reflink=auto "$source" "$build_source"

export PATH="$tooling/cmake/data/bin:$tooling:$PATH"
export CUDA_HOME=${CUDA_HOME:-/usr/local/cuda}
export CUDACXX=$CUDA_HOME/bin/nvcc
export TORCH_CUDA_ARCH_LIST=9.0a
export MAX_JOBS=${MAX_JOBS:-4}
export UV_LINK_MODE=copy
export UV_NO_PROGRESS=1
export UV_PYTHON_DOWNLOADS=never
export VLLM_VERSION_OVERRIDE=0.0.0+w4a8hopper.b13ca6b020
export CMAKE_ARGS=-DFETCHCONTENT_FULLY_DISCONNECTED=ON

"$tooling/uv" venv --python 3.12 --system-site-packages "$venv"
"$tooling/uv" pip install --python "$venv/bin/python" --no-deps \
  --no-build-isolation "$build_source"

"$venv/bin/python" - <<'PY'
import hashlib
import json
from pathlib import Path

import torch
import vllm

package = Path(vllm.__file__).parent
libraries = sorted(package.glob("*.so"))
assert libraries
print(
    json.dumps(
        {
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "libraries": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in libraries
            },
        },
        sort_keys=True,
    )
)
PY
