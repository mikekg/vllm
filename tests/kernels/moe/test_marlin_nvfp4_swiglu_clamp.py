# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test that the Marlin NVFP4 MoE backend correctly applies the SwiGLU clamp
(gemm1_clamp_limit / swiglu_limit) when it is present in the quant config.

Mirrors the silu_clamp case from test_trtllm_nvfp4_moe.py but targets the
Marlin backend (SM75+) instead of TRTLLM (SM100-only).  The swiglu_limit is
needed for models such as DeepSeek V4 Flash whose quantization config carries a
per-layer activation clamp value to prevent quantization spikes when activation
magnitudes exceed the fp4 calibration range.
"""

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config, make_test_weights
from tests.kernels.quantization.nvfp4_utils import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    dequantize_nvfp4_to_dtype,
)
from tests.kernels.utils import torch_moe
from vllm import _custom_ops as ops
from vllm.config import ParallelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.activation import SiluAndMulWithClamp
from vllm.model_executor.layers.fused_moe import fused_topk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import nvfp4_moe_quant_config
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
    MarlinExpertsBase,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import (
    make_moe_prepare_and_finalize_no_dp_ep,
)
from vllm.platforms import current_platform
from vllm.utils.torch_utils import set_random_seed

if not current_platform.has_device_capability((7, 5)):
    pytest.skip("Marlin requires SM75+", allow_module_level=True)

MNK_FACTORS = [
    (2, 1024, 1024),
    (64, 2048, 1536),
]

_SWIGLU_LIMIT = 7.0


@pytest.mark.parametrize("m,n,k", MNK_FACTORS)
@pytest.mark.parametrize("e", [8, 64])
@pytest.mark.parametrize("topk", [1, 2])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
@torch.inference_mode()
def test_marlin_nvfp4_moe_swiglu_clamp(
    m: int,
    n: int,
    k: int,
    e: int,
    topk: int,
    dtype: torch.dtype,
):
    """Marlin NVFP4 MoE with gemm1_clamp_limit matches torch_moe reference."""
    set_random_seed(7)
    quant_blocksize = 16

    with set_current_vllm_config(
        VllmConfig(parallel_config=ParallelConfig(pipeline_parallel_size=1))
    ):
        a = torch.randn((m, k), device="cuda", dtype=dtype) * _SWIGLU_LIMIT * 3

        (_, w1_q, w1_blockscale, w1_gs), (_, w2_q, w2_blockscale, w2_gs) = (
            make_test_weights(
                e,
                n,
                k,
                in_dtype=dtype,
                quant_dtype="nvfp4",
                block_shape=None,
                per_out_ch_quant=False,
            )
        )

        assert w1_gs is not None and w2_gs is not None
        assert w1_blockscale is not None and w2_blockscale is not None

        a1_gs = torch.ones((e,), device="cuda", dtype=torch.float32)
        a2_gs = torch.ones((e,), device="cuda", dtype=torch.float32)

        quant_config = nvfp4_moe_quant_config(
            g1_alphas=(1 / w1_gs),
            g2_alphas=(1 / w2_gs),
            a1_gscale=a1_gs,
            a2_gscale=a2_gs,
            w1_scale=w1_blockscale,
            w2_scale=w2_blockscale,
            gemm1_clamp_limit=_SWIGLU_LIMIT,
        )

        score = torch.randn((m, e), device="cuda", dtype=dtype)
        topk_weights, topk_ids, _ = fused_topk(a, score, topk, renormalize=False)

        moe_config = make_dummy_moe_config(
            num_experts=e,
            experts_per_token=topk,
            hidden_dim=k,
            intermediate_size=n,
        )

        marlin_kernel = mk.FusedMoEKernel(
            make_moe_prepare_and_finalize_no_dp_ep(use_monolithic=False),
            MarlinExpertsBase(moe_config=moe_config, quant_config=quant_config),
        )

        marlin_output = marlin_kernel.apply(
            hidden_states=a,
            w1=w1_q,
            w2=w2_q,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            activation=MoEActivation.SILU,
            global_num_experts=e,
            expert_map=None,
            apply_router_weight_on_input=False,
        )

        a_global_scale = ((FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX) / a.abs().max()).to(
            torch.float32
        )
        a_fp4, a_scale_interleaved = ops.scaled_fp4_quant(a, a_global_scale)
        a_dq = dequantize_nvfp4_to_dtype(
            a_fp4,
            a_scale_interleaved,
            a_global_scale,
            dtype=dtype,
            device=a.device,
            block_size=quant_blocksize,
        )

        w1_d = torch.empty((e, 2 * n, k), device="cuda", dtype=dtype)
        w2_d = torch.empty((e, k, n), device="cuda", dtype=dtype)
        for i in range(e):
            w1_d[i] = dequantize_nvfp4_to_dtype(
                w1_q[i],
                w1_blockscale[i],
                w1_gs[i],
                dtype=dtype,
                device=w1_q.device,
                block_size=quant_blocksize,
            )
            w2_d[i] = dequantize_nvfp4_to_dtype(
                w2_q[i],
                w2_blockscale[i],
                w2_gs[i],
                dtype=dtype,
                device=w2_q.device,
                block_size=quant_blocksize,
            )

        ref_output = torch_moe(
            a_dq,
            w1_d,
            w2_d,
            topk_weights,
            topk_ids,
            activation=SiluAndMulWithClamp(_SWIGLU_LIMIT, compile_native=False),
        )

        torch.testing.assert_close(marlin_output, ref_output, atol=0.05, rtol=0.05)
