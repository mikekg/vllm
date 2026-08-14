#!/usr/bin/env bash
set -euo pipefail

python=${VLLM_PYTHON:-.venv/bin/python}

"$python" -m pytest tests/kernels/quantization/test_marlin_nvfp4_to_fp8.py -v
"$python" -m pytest tests/quantization/test_modelopt.py -k nvfp4_bycopy -v
"$python" -m pytest tests/kernels/core/test_fused_silu_mul_block_quant.py -v
"$python" -m pytest tests/kernels/moe/test_nvfp4_bycopy_moe.py -v
"$python" -m pytest tests/compile/passes/test_functionalization.py -v
"$python" -m pytest tests/v1/worker/test_workspace.py tests/kernels/moe/test_shared_experts.py -v
