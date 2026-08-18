# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compile and dump the two dense NVFP4 execution bodies independently."""

import argparse
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("VLLM_DISABLE_COMPILE_CACHE", "1")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "1")

import torch

from vllm import _custom_ops as ops
from vllm.compilation.backends import VllmBackend
from vllm.compilation.monitor import monitor_torch_compile
from vllm.config import (
    CompilationConfig,
    SchedulerConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.config.compilation import CompilationMode, CUDAGraphMode
from vllm.model_executor.layers.quantization.utils.fp8_utils import (
    per_token_group_quant_fp8,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repacked_nk,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
)

LLAMA_SHAPES = {
    "qkv": (6144, 4096),
    "o": (4096, 4096),
    "gate_up": (28672, 4096),
    "down": (4096, 14336),
}


def make_layer(n: int, k: int) -> SimpleNamespace:
    codes = (
        torch.arange(k, device="cuda", dtype=torch.int32)
        .remainder_(16)
        .to(torch.uint8)
        .expand(n, k)
    )
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    layer = SimpleNamespace(
        output_size_per_partition=n,
        input_size_per_partition=k,
        params_dtype=torch.bfloat16,
        weight=torch.nn.Parameter(packed, requires_grad=False),
        weight_scale=torch.nn.Parameter(
            torch.ones((n, k // 16), device="cuda", dtype=torch.float8_e4m3fn),
            requires_grad=False,
        ),
        weight_global_scale=torch.nn.Parameter(
            torch.tensor(0.25, device="cuda", dtype=torch.float32),
            requires_grad=False,
        ),
    )
    prepare_fp4_layer_for_marlin(layer)
    return layer


class MarlinBody(torch.nn.Module):
    def __init__(self, layer: SimpleNamespace, n: int, k: int):
        super().__init__()
        self.n = n
        self.k = k
        self.padded_n, self.padded_k = marlin_repacked_nk(layer.weight, 4)
        for name in ("weight", "weight_scale", "weight_global_scale", "workspace"):
            self.register_buffer(name, getattr(layer, name), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_fp4_marlin_linear(
            x,
            self.weight,
            self.weight_scale,
            self.weight_global_scale,
            self.workspace,
            self.n,
            self.k,
        )


class HybridBody(torch.nn.Module):
    def __init__(self, layer: SimpleNamespace, n: int, k: int):
        super().__init__()
        self.n = n
        self.k = k
        self.padded_n, resident_k = marlin_repacked_nk(layer.weight, 4)
        self.padded_k = (resident_k + 127) // 128 * 128
        for name in (
            "weight",
            "weight_scale",
            "weight_global_scale",
            "weight_fp8_scale_divisor_code",
        ):
            self.register_buffer(name, getattr(layer, name), persistent=False)
        self.register_buffer(
            "fp8_weight",
            torch.empty(
                (self.padded_n, self.padded_k),
                device="cuda",
                dtype=torch.float8_e4m3fn,
            ),
            persistent=False,
        )
        self.register_buffer(
            "fp8_weight_scale",
            torch.empty(
                ((self.padded_n + 127) // 128, self.padded_k // 128),
                device="cuda",
                dtype=torch.float32,
            ),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q, x_scale = per_token_group_quant_fp8(
            x,
            group_size=128,
            column_major_scales=True,
            dtype=torch.float8_e4m3fn,
            use_ue8m0=False,
        )
        ops.marlin_nvfp4_to_fp8(
            self.fp8_weight,
            self.fp8_weight_scale,
            self.weight,
            self.weight_scale,
            self.weight_global_scale,
            self.weight_fp8_scale_divisor_code,
            x.dtype,
        )
        output = ops.cutlass_scaled_mm(
            x_q,
            self.fp8_weight.T,
            scale_a=x_scale,
            scale_b=self.fp8_weight_scale.T,
            out_dtype=x.dtype,
        )
        return output[:, : self.n].contiguous()


def make_config(m: int, dump: Path) -> VllmConfig:
    return VllmConfig(
        scheduler_config=SchedulerConfig(
            max_model_len=m,
            max_num_batched_tokens=m,
            max_num_seqs=m,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            backend="inductor",
            custom_ops=["all"],
            splitting_ops=[],
            compile_ranges_endpoints=[m],
            cudagraph_mode=CUDAGraphMode.NONE,
            debug_dump_path=dump,
            inductor_compile_config={
                "force_disable_caches": True,
                "trace.enabled": True,
            },
        ),
    )


def run_one(
    branch: str,
    shape_name: str,
    n: int,
    k: int,
    m: int,
    root: Path,
) -> dict:
    dump = root / shape_name / f"m{m}" / branch
    dump.mkdir(parents=True, exist_ok=True)
    os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(dump / "torch_compile_debug")
    os.environ["TORCH_TRACE"] = str(dump / "torch_trace")
    torch._dynamo.config.debug_dir_root = str(dump / "torch_compile_debug")
    torch._dynamo.reset()
    layer = make_layer(n, k)
    model_cls = MarlinBody if branch == "marlin" else HybridBody
    model = model_cls(layer, n, k).eval()
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    production = MarlinBody(layer, n, k).eval()
    expected = production(x)
    eager = model(x)
    torch.accelerator.synchronize()
    if branch == "marlin":
        torch.testing.assert_close(eager, expected, rtol=0, atol=0)
        normalized_mae = 0.0
    else:
        error_mean = (eager.float() - expected.float()).abs().mean().item()
        reference_mean = expected.float().abs().mean().item()
        normalized_mae = error_mean / reference_mean
        if normalized_mae >= 0.08:
            raise AssertionError(f"hybrid normalized_mae={normalized_mae}")
    config = make_config(m, dump / "vllm")
    backend = VllmBackend(config)
    with set_current_vllm_config(config):
        compiled = torch.compile(model, backend=backend, fullgraph=True)
        with monitor_torch_compile(config):
            actual = compiled(x)
        torch.accelerator.synchronize()
        torch.testing.assert_close(actual, eager, rtol=0.01, atol=0.1)
    (dump / "dynamo_fx.py").write_text(backend.graph.print_readable(print_output=False))
    (dump / "split_fx.py").write_text(
        backend.split_gm.print_readable(print_output=False)
    )

    summary = {
        "branch": branch,
        "shape": shape_name,
        "m": m,
        "n": n,
        "k": k,
        "normalized_mae_vs_production_marlin": normalized_mae,
        "allocated_bytes": torch.accelerator.memory_allocated(),
        "reserved_bytes": torch.accelerator.memory_reserved(),
    }
    (dump / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    del compiled, backend, model, production, layer, x, expected, eager, actual
    torch._dynamo.reset()
    torch.accelerator.empty_cache()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("hybrid_compile_dump"))
    parser.add_argument("--shapes", nargs="*", default=list(LLAMA_SHAPES))
    parser.add_argument("--m", type=int, default=32768)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for shape_name in args.shapes:
        n, k = LLAMA_SHAPES[shape_name]
        for branch in ("marlin", "hybrid"):
            results.append(run_one(branch, shape_name, n, k, args.m, args.output))
    (args.output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    archive = shutil.make_archive(str(args.output), "gztar", args.output)
    print(f"artifact={archive}", flush=True)


if __name__ == "__main__":
    main()
