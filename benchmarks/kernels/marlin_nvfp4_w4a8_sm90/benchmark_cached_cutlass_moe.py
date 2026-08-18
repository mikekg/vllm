# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""H100 NVFP4 Marlin versus cached block-FP8 MoE A/B."""

import argparse
import json
import statistics
from functools import partial
from types import SimpleNamespace

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_make_prepare_finalize,
)
from vllm.model_executor.layers.fused_moe.config import (
    fp8_w8a8_moe_quant_config,
    nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts.deep_gemm_moe import (
    DeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
    FlashInferExperts,
)
from vllm.model_executor.layers.fused_moe.experts.nvfp4_bycopy_moe import (
    NvFp4ByCopyExperts,
)
from vllm.model_executor.layers.fused_moe.experts.triton_deep_gemm_moe import (
    TritonOrDeepGemmExperts,
)
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repacked_nk,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_nvfp4_moe_layer_for_marlin,
)
from vllm.v1.worker.workspace import init_workspace_manager

SHAPES = {
    "q3m": (128, 768, 2048, 8),
    "q36m": (256, 512, 2048, 8),
    "deepseek_v4_flash": (256, 2048, 4096, 6),
    "deepseek_v4_pro": (384, 3072, 7168, 6),
}
DEFAULT_MS = (1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
ROUTINGS = {"balanced": 1, "half": 2, "quarter": 4}


def make_weights(e: int, n: int, k: int):
    generator = torch.Generator(device="cuda").manual_seed(7)
    w13 = torch.randint(
        256,
        (e, 2 * n, k // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        256,
        (e, k, n // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w13_scale = torch.full(
        (e, 2 * n, k // 16),
        2**-4,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    w2_scale = torch.full(
        (e, k, n // 16),
        2**-4,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    w13_global = torch.full((e,), 2**-4, device="cuda")
    w2_global = torch.full((e,), 2**-4, device="cuda")
    layer = SimpleNamespace(
        num_experts=e,
        hidden_size=k,
        intermediate_size_per_partition=n,
        params_dtype=torch.bfloat16,
    )
    values = prepare_nvfp4_moe_layer_for_marlin(
        layer,
        w13,
        w13_scale,
        w13_global,
        w2,
        w2_scale,
        w2_global,
        is_act_and_mul=True,
    )
    (
        layer.w13_weight,
        w13_scale,
        w13_global,
        layer.w2_weight,
        w2_scale,
        w2_global,
    ) = values
    quant = nvfp4_w4a16_moe_quant_config(w13_global, w2_global, w13_scale, w2_scale)
    return layer, quant


def make_kernel(config, quant, experts):
    prepare_finalize = maybe_make_prepare_finalize(
        moe=config,
        quant_config=quant,
        allow_new_interface=True,
        use_monolithic=False,
    )
    return mk.FusedMoEKernel(prepare_finalize, experts)


def make_block_fp8_weights(layer, quant):
    e = layer.w13_weight.size(0)
    w13_n, w13_k = marlin_repacked_nk(layer.w13_weight[0], 4)
    w2_n, w2_k = marlin_repacked_nk(layer.w2_weight[0], 4)
    fp8 = torch.float8_e4m3fn
    w13 = torch.empty((e, w13_n, w13_k), dtype=fp8, device="cuda")
    w2 = torch.empty((e, w2_n, w2_k), dtype=fp8, device="cuda")
    w13_scale = torch.empty((e, w13_n // 128, w13_k // 128), device="cuda")
    w2_scale = torch.empty((e, w2_n // 128, w2_k // 128), device="cuda")
    for expert in range(e):
        ops.marlin_nvfp4_to_fp8(
            w13[expert],
            w13_scale[expert],
            layer.w13_weight[expert],
            quant.w1_scale[expert],
            quant.g1_alphas[expert : expert + 1],
            layer.w13_fp8_scale_divisor_code[expert],
            torch.bfloat16,
        )
        ops.marlin_nvfp4_to_fp8(
            w2[expert],
            w2_scale[expert],
            layer.w2_weight[expert],
            quant.w2_scale[expert],
            quant.g2_alphas[expert : expert + 1],
            layer.w2_fp8_scale_divisor_code[expert],
            torch.bfloat16,
        )
    block_quant = fp8_w8a8_moe_quant_config(w13_scale, w2_scale, block_shape=[128, 128])
    return w13, w2, block_quant


def graph_us(fn, calls: int = 10, samples: int = 30) -> float:
    for _ in range(3):
        fn()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    for _ in range(5):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(samples):
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return statistics.median(times)


def run_shape(
    name: str,
    ms: list[int],
    routings: list[str],
    production_only: bool = False,
) -> None:
    e, n, k, topk = SHAPES[name]
    config = make_dummy_moe_config(
        num_experts=e,
        num_local_experts=e,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size=n,
        in_dtype=torch.bfloat16,
        max_num_tokens=max(ms),
    )
    layer, quant = make_weights(e, n, k)
    hybrid = NvFp4ByCopyExperts(config, quant)
    hybrid.process_weights_after_loading(layer)
    if not production_only:
        block_w1, block_w2, block_quant = make_block_fp8_weights(layer, quant)
    hybrid.m_knee = 1
    marlin = make_kernel(config, quant, hybrid.fallback_experts)
    cutlass = make_kernel(config, hybrid.quant_config, hybrid)
    if not production_only:
        block = make_kernel(
            config, block_quant, TritonOrDeepGemmExperts(config, block_quant)
        )
        deepgemm = make_kernel(
            config, block_quant, DeepGemmExperts(config, block_quant)
        )
        triton = make_kernel(config, block_quant, TritonExperts(config, block_quant))
        flashinfer = make_kernel(
            config, block_quant, FlashInferExperts(config, block_quant)
        )

    for m in ms:
        generator = torch.Generator(device="cuda").manual_seed(m)
        hidden = (
            torch.randn(
                (m, k), dtype=torch.bfloat16, device="cuda", generator=generator
            )
            / k**0.5
        )
        topk_weights = torch.full(
            (m, topk), 1 / topk, dtype=torch.float32, device="cuda"
        )
        for routing in routings:
            active_experts = e // ROUTINGS[routing]
            assert active_experts >= topk
            topk_ids = (
                torch.arange(m * topk, device="cuda").view(m, topk) % active_experts
            )

            def apply(
                kernel,
                w1=layer.w13_weight,
                w2=layer.w2_weight,
                hidden=hidden,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
            ):
                return kernel.apply(
                    hidden,
                    w1,
                    w2,
                    topk_weights,
                    topk_ids,
                    MoEActivation.SILU,
                    e,
                    None,
                    False,
                )

            run_marlin = partial(apply, marlin)
            run_cutlass = partial(apply, cutlass)
            reference = run_marlin().clone()
            candidate = run_cutlass().clone()
            if production_only:
                error = (candidate.float() - reference.float()).abs()
                ref_norm = torch.linalg.vector_norm(reference.float())
                cosine = torch.nn.functional.cosine_similarity(
                    reference.float().flatten(), candidate.float().flatten(), dim=0
                )
                marlin_us = graph_us(run_marlin)
                cutlass_us = graph_us(run_cutlass)
                print(
                    json.dumps(
                        {
                            "shape": name,
                            "routing": routing,
                            "active_experts": active_experts,
                            "mean_routed_rows": m * topk / active_experts,
                            "E": e,
                            "M": m,
                            "N": n,
                            "K": k,
                            "topk": topk,
                            "marlin_us": marlin_us,
                            "hybrid_us": cutlass_us,
                            "cutlass_us": cutlass_us,
                            "speedup": marlin_us / cutlass_us,
                            "max_abs": error.max().item(),
                            "relative_l2": (
                                torch.linalg.vector_norm(error) / ref_norm
                            ).item(),
                            "cosine": cosine.item(),
                        }
                    ),
                    flush=True,
                )
                continue

            run_block = partial(apply, block, block_w1, block_w2)
            run_triton = partial(apply, triton, block_w1, block_w2)
            run_flashinfer = partial(apply, flashinfer, block_w1, block_w2)

            def run_staged_deepgemm():
                ops.marlin_nvfp4_to_fp8(
                    block_w1,
                    block_quant.w1_scale,
                    layer.w13_weight,
                    quant.w1_scale,
                    quant.g1_alphas,
                    layer.w13_fp8_scale_divisor_code,
                    torch.bfloat16,
                )
                ops.marlin_nvfp4_to_fp8(
                    block_w2,
                    block_quant.w2_scale,
                    layer.w2_weight,
                    quant.w2_scale,
                    quant.g2_alphas,
                    layer.w2_fp8_scale_divisor_code,
                    torch.bfloat16,
                )
                return apply(deepgemm, block_w1, block_w2)

            block_candidate = run_block().clone()
            triton_candidate = run_triton().clone()
            flashinfer_candidate = run_flashinfer().clone()
            staged_deepgemm_candidate = run_staged_deepgemm().clone()
            error = (candidate.float() - reference.float()).abs()
            ref_norm = torch.linalg.vector_norm(reference.float())
            cosine = torch.nn.functional.cosine_similarity(
                reference.float().flatten(), candidate.float().flatten(), dim=0
            )
            marlin_us = graph_us(run_marlin)
            cutlass_us = graph_us(run_cutlass)
            block_us = graph_us(run_block)
            triton_us = graph_us(run_triton)
            flashinfer_us = graph_us(run_flashinfer)
            staged_deepgemm_us = graph_us(run_staged_deepgemm)
            block_error = (block_candidate.float() - reference.float()).abs()
            triton_error = (triton_candidate.float() - reference.float()).abs()
            flashinfer_error = (flashinfer_candidate.float() - reference.float()).abs()
            staged_deepgemm_error = (
                staged_deepgemm_candidate.float() - reference.float()
            ).abs()
            print(
                json.dumps(
                    {
                        "shape": name,
                        "routing": routing,
                        "active_experts": active_experts,
                        "mean_routed_rows": m * topk / active_experts,
                        "E": e,
                        "M": m,
                        "N": n,
                        "K": k,
                        "topk": topk,
                        "marlin_us": marlin_us,
                        "cutlass_us": cutlass_us,
                        "block_us": block_us,
                        "speedup": marlin_us / cutlass_us,
                        "block_speedup": marlin_us / block_us,
                        "triton_us": triton_us,
                        "triton_speedup": marlin_us / triton_us,
                        "flashinfer_us": flashinfer_us,
                        "flashinfer_speedup": marlin_us / flashinfer_us,
                        "staged_deepgemm_us": staged_deepgemm_us,
                        "staged_deepgemm_speedup": marlin_us / staged_deepgemm_us,
                        "max_abs": error.max().item(),
                        "relative_l2": (
                            torch.linalg.vector_norm(error) / ref_norm
                        ).item(),
                        "cosine": cosine.item(),
                        "block_relative_l2": (
                            torch.linalg.vector_norm(block_error) / ref_norm
                        ).item(),
                        "block_cosine": torch.nn.functional.cosine_similarity(
                            reference.float().flatten(),
                            block_candidate.float().flatten(),
                            dim=0,
                        ).item(),
                        "triton_relative_l2": (
                            torch.linalg.vector_norm(triton_error) / ref_norm
                        ).item(),
                        "triton_cosine": torch.nn.functional.cosine_similarity(
                            reference.float().flatten(),
                            triton_candidate.float().flatten(),
                            dim=0,
                        ).item(),
                        "flashinfer_relative_l2": (
                            torch.linalg.vector_norm(flashinfer_error) / ref_norm
                        ).item(),
                        "flashinfer_cosine": torch.nn.functional.cosine_similarity(
                            reference.float().flatten(),
                            flashinfer_candidate.float().flatten(),
                            dim=0,
                        ).item(),
                        "staged_deepgemm_relative_l2": (
                            torch.linalg.vector_norm(staged_deepgemm_error) / ref_norm
                        ).item(),
                        "staged_deepgemm_cosine": torch.nn.functional.cosine_similarity(
                            reference.float().flatten(),
                            staged_deepgemm_candidate.float().flatten(),
                            dim=0,
                        ).item(),
                    }
                ),
                flush=True,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shapes", nargs="+", choices=SHAPES, default=("q3m", "q36m"))
    parser.add_argument("--m", nargs="+", type=int, default=DEFAULT_MS)
    parser.add_argument("--routing", nargs="+", choices=ROUTINGS, default=ROUTINGS)
    parser.add_argument("--production-only", action="store_true")
    args = parser.parse_args()
    init_workspace_manager(torch.accelerator.current_device_index())
    for shape in args.shapes:
        run_shape(shape, args.m, args.routing, args.production_only)


if __name__ == "__main__":
    main()
