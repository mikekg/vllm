# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SwiGLU-clamp test for the Marlin NVFP4 MoE kernel.

Verifies that ``fused_marlin_moe`` applies the SwiGLU activation clamp
(``swiglu_limit`` / ``gemm1_clamp_limit``, value 7.0) via the in-kernel
``swiglu_limit_func`` path, comparing against a torch reference that uses
``SiluAndMulWithClamp(7.0)``. This mirrors the ``silu_clamp`` case of
``tests/kernels/moe/test_trtllm_nvfp4_moe.py`` but for the Marlin backend.

The Marlin backend requires Marlin-repacked weights, so the raw NVFP4
weights produced here are loaded onto a synthetic layer in the exact
attribute layout the ModelOpt loader uses, then repacked via
``prepare_moe_fp4_layer_for_marlin`` before calling ``fused_marlin_moe``
directly (the modular ``MarlinExperts`` path couples scales through the
quant_config and is awkward to drive standalone).
"""

import pytest
import torch

from tests.kernels.moe.utils import make_dummy_moe_config
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.custom_op import CustomOp, op_registry
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    fused_marlin_moe,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_moe_fp4_layer_for_marlin,
)
from vllm.platforms import current_platform
from vllm.scalar_type import scalar_types
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability(75):
    pytest.skip("Marlin requires compute capability 7.5+", allow_module_level=True)

# (m, n, k) = (tokens, intermediate_size_per_partition, hidden_dim).
# Both shapes keep hidden_dim % 128 == 0 and intermediate_size 64-aligned so
# the (non-padding) in-place prepare_moe_fp4_layer_for_marlin accepts them.
MNK_FACTORS = [
    (64, 1024, 1024),
    (64, 2048, 1536),
]

_SWIGLU_LIMIT = 7.0
_QUANT_BLOCKSIZE = 16
_CLAMP_OP_NAME = "test_marlin_silu_and_mul_with_clamp"

# Test-only fixed-limit clamp. Setting ``custom_op_name`` makes the class
# itself a valid ``activation=`` argument to ``torch_moe`` (which only looks
# up ``activation.custom_op_name`` in ``op_registry`` and then instantiates
# it with no arguments).
if _CLAMP_OP_NAME not in op_registry:

    @CustomOp.register(_CLAMP_OP_NAME)
    class _SiluAndMulWithClampTest(SiluAndMulWithClamp):
        custom_op_name = _CLAMP_OP_NAME

        def __init__(self, *, compile_native: bool = False) -> None:
            super().__init__(_SWIGLU_LIMIT, compile_native=compile_native)


SILU_WITH_CLAMP = op_registry[_CLAMP_OP_NAME]


def _quantize_expert_nvfp4(
    w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize a single 2D weight to NVFP4.

    Returns:
        fp4: packed fp4 bytes, shape ``(rows, cols // 2)`` uint8.
        scale_linear: per-block scales, shape ``(rows, cols // 16)``,
            ``float8_e4m3fn`` (linear layout, matching the ModelOpt
            ``w*_weight_scale`` checkpoint contract).
        scale_swizzled: per-block scales in the 128x4-swizzled layout used
            by ``dequantize_nvfp4_to_dtype`` for the reference path.
        global_scale: per-tensor global scale (scalar float32).
    """
    global_scale = (FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / w.abs().max().to(torch.float32)
    # The packed fp4 bytes are identical for both scale layouts; only the
    # scale storage differs, so the layer (linear) and the reference
    # (swizzled) share the same packed weights and global scale.
    fp4, scale_linear = ops.scaled_fp4_quant(
        w, global_scale, is_sf_swizzled_layout=False
    )
    _, scale_swizzled = ops.scaled_fp4_quant(
        w, global_scale, is_sf_swizzled_layout=True
    )
    return fp4, scale_linear.view(torch.float8_e4m3fn), scale_swizzled, global_scale


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [8])
@pytest.mark.parametrize("topk", [2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_marlin_nvfp4_moe_swiglu_clamp(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
    workspace_init,
):
    set_random_seed(7)
    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        # Scale activations well past the clamp limit (7.0) so the clamp is
        # exercised on a meaningful fraction of elements.
        a = torch.randn((m, k), device="cuda", dtype=dtype) * _SWIGLU_LIMIT * 3

        # Raw bf16 expert weights.
        w13_16 = torch.randn((e, 2 * n, k), device="cuda", dtype=dtype) / 15
        w2_16 = torch.randn((e, k, n), device="cuda", dtype=dtype) / 15

        # NVFP4-quantize each expert; collect packed weights, linear scales
        # (for the Marlin layer), and swizzled scales + global scales (for
        # the torch reference).
        w13_q = torch.empty((e, 2 * n, k // 2), device="cuda", dtype=torch.uint8)
        w2_q = torch.empty((e, k, n // 2), device="cuda", dtype=torch.uint8)
        w13_scale = torch.empty(
            (e, 2 * n, k // _QUANT_BLOCKSIZE),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        w2_scale = torch.empty(
            (e, k, n // _QUANT_BLOCKSIZE),
            device="cuda",
            dtype=torch.float8_e4m3fn,
        )
        w13_blockscale_sw: list[torch.Tensor | None] = [None] * e
        w2_blockscale_sw: list[torch.Tensor | None] = [None] * e
        w13_gs = torch.empty((e,), device="cuda", dtype=torch.float32)
        w2_gs = torch.empty((e,), device="cuda", dtype=torch.float32)

        for i in range(e):
            (
                w13_q[i],
                w13_scale[i],
                w13_blockscale_sw[i],
                w13_gs[i],
            ) = _quantize_expert_nvfp4(w13_16[i])
            (
                w2_q[i],
                w2_scale[i],
                w2_blockscale_sw[i],
                w2_gs[i],
            ) = _quantize_expert_nvfp4(w2_16[i])

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        # Build a synthetic layer in the exact attribute layout
        # prepare_moe_fp4_layer_for_marlin reads, then repack in place.
        layer = torch.nn.Module()
        layer.w13_weight = torch.nn.Parameter(w13_q, requires_grad=False)
        layer.w2_weight = torch.nn.Parameter(w2_q, requires_grad=False)
        layer.w13_weight_scale = torch.nn.Parameter(w13_scale, requires_grad=False)
        layer.w2_weight_scale = torch.nn.Parameter(w2_scale, requires_grad=False)
        layer.w13_weight_scale_2 = torch.nn.Parameter(w13_gs, requires_grad=False)
        layer.w2_weight_scale_2 = torch.nn.Parameter(w2_gs, requires_grad=False)
        layer.params_dtype = dtype
        layer.moe_config = make_dummy_moe_config(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
            in_dtype=dtype,
        )

        prepare_moe_fp4_layer_for_marlin(layer)

        # Direct fused_marlin_moe call with the repacked weights, processed
        # scales, processed global scales, and the SwiGLU clamp limit. This
        # mirrors MarlinExperts.apply for the NVFP4 backend (global_scale*
        # = processed w*_weight_scale_2, clamp_limit = gemm1_clamp_limit).
        marlin_output = fused_marlin_moe(
            hidden_states=a,
            w1=layer.w13_weight,
            w2=layer.w2_weight,
            bias1=None,
            bias2=None,
            w1_scale=layer.w13_weight_scale,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            quant_type_id=scalar_types.float4_e2m1f.id,
            global_num_experts=e,
            expert_map=None,
            global_scale1=layer.w13_weight_scale_2,
            global_scale2=layer.w2_weight_scale_2,
            workspace=layer.workspace,
            activation=MoEActivation.SILU,
            input_dtype=None,
            clamp_limit=_SWIGLU_LIMIT,
        )

        # Reference: round-trip activations and weights through NVFP4
        # quant/dequant so the comparison isolates the clamp/activation
        # behavior from quantization error.
        a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / a.abs().max()).to(
            torch.float32
        )
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        a_in_dtype = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=a.dtype,
            device=a.device,
            block_size=_QUANT_BLOCKSIZE,
        )

        w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)
        for i in range(e):
            w1_d[i] = dequantize_nvfp4_to_dtype(
                w13_q[i],
                w13_blockscale_sw[i],
                w13_gs[i],
                dtype=dtype,
                device=w13_q.device,
                block_size=_QUANT_BLOCKSIZE,
            )
            w2_d[i] = dequantize_nvfp4_to_dtype(
                w2_q[i],
                w2_blockscale_sw[i],
                w2_gs[i],
                dtype=dtype,
                device=w2_q.device,
                block_size=_QUANT_BLOCKSIZE,
            )

        ref_output = torch_moe(
            a_in_dtype, w1_d, w2_d, score, topk, activation=SILU_WITH_CLAMP
        )

        # Loose tolerance: NVFP4 weights + Marlin dequant carry real
        # quantization error (matches tests/kernels/moe/
        # test_marlin_vs_trtllm_mxint4.py).
        torch.testing.assert_close(marlin_output, ref_output, atol=0.3, rtol=1.0)


if __name__ == "__main__":
    test_marlin_nvfp4_moe_swiglu_clamp(64, 1024, 1024, 8, 2, torch.bfloat16, None)
