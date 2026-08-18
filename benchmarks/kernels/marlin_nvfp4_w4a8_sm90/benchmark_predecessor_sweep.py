# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import itertools
import json
from statistics import median
from types import SimpleNamespace

DEFAULT_AXIS = (512, 1024, 2048, 4096)
L2_BYTES = 50 * 1024 * 1024


def shape_pool(n_axis, k_axis) -> tuple[tuple[int, int], ...]:
    return tuple(itertools.product(dict.fromkeys(n_axis), dict.fromkeys(k_axis)))


def source_copies(shapes: tuple[tuple[int, int], ...]) -> int:
    bytes_per_set = sum(n * k // 2 + n * (k // 16) * 2 for n, k in shapes)
    cache_copies = (4 * L2_BYTES + bytes_per_set - 1) // bytes_per_set
    return max(4 * len(shapes), cache_copies)


def check() -> None:
    shapes = shape_pool(DEFAULT_AXIS, DEFAULT_AXIS)
    assert len(shapes) == 16
    assert len(shapes) ** 2 == 256
    assert source_copies(shapes) >= 2
    print(json.dumps({"record": "check", "transitions_per_m_backend": 256}))


def make_source(
    n: int,
    k: int,
    seed: int,
    torch,
    prepare_fp4_layer_for_marlin,
    ops=None,
):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    packed = torch.randint(
        0, 256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=generator
    )
    scale = torch.empty((n, k // 16), dtype=torch.float8_e4m3fn, device="cuda")
    scale.fill_((0.5, 1.0, 2.0, 4.0)[seed % 4])
    layer = SimpleNamespace(
        output_size_per_partition=n,
        input_size_per_partition=k,
        params_dtype=torch.bfloat16,
        weight=torch.nn.Parameter(packed, requires_grad=False),
        weight_scale=torch.nn.Parameter(scale, requires_grad=False),
        weight_global_scale=torch.nn.Parameter(
            torch.tensor(0.37, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        ),
    )
    prepare_fp4_layer_for_marlin(layer)
    if ops is not None:
        layer.cached_fp8_weight = torch.empty(
            (n, k), dtype=torch.float8_e4m3fn, device="cuda"
        )
        layer.cached_fp8_scale = torch.empty(
            (n // 128, k // 128), dtype=torch.float32, device="cuda"
        )
        ops.marlin_nvfp4_to_fp8(
            layer.cached_fp8_weight,
            layer.cached_fp8_scale,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.weight_fp8_scale_divisor_code,
            torch.bfloat16,
        )
    return layer


def source_bytes(layer) -> int:
    tensors = (
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.weight_fp8_scale_divisor_code,
    )
    return sum(t.numel() * t.element_size() for t in tensors) + sum(
        t.numel() * t.element_size()
        for t in (
            getattr(layer, "cached_fp8_weight", None),
            getattr(layer, "cached_fp8_scale", None),
        )
        if t is not None
    )


def run_sweep(args) -> None:
    import torch

    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.nvfp4 import (
        marlin_fp8 as _marlin_fp8,  # noqa: F401
    )
    from vllm.model_executor.layers.quantization.utils.fp8_utils import (
        per_token_group_quant_fp8,
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        apply_fp4_marlin_linear,
        prepare_fp4_layer_for_marlin,
    )

    assert torch.cuda.get_device_capability() == (9, 0)
    shapes = shape_pool(args.n, args.k)
    current_indices = tuple(dict.fromkeys(args.current_index or range(len(shapes))))
    if (
        any(m <= 0 for m in args.m)
        or any(n <= 0 or k <= 0 or n % 128 or k % 128 for n, k in shapes)
        or any(index < 0 or index >= len(shapes) for index in current_indices)
    ):
        raise ValueError(
            "M must be positive; N and K must be positive multiples of 128; "
            "current indices must address the shape pool"
        )
    copies = args.source_copies or source_copies(shapes)
    if copies < 4 * len(shapes):
        raise ValueError(
            f"need at least {4 * len(shapes)} source copies to avoid reuse "
            "across consecutive backend sequences"
        )
    sources = {
        shape: [
            make_source(
                *shape,
                10000 * index + copy,
                torch,
                prepare_fp4_layer_for_marlin,
                ops if args.cached_weight else None,
            )
            for copy in range(copies)
        ]
        for index, shape in enumerate(shapes)
    }
    corpus_bytes = sum(source_bytes(s) for group in sources.values() for s in group)
    if corpus_bytes < 4 * L2_BYTES:
        raise ValueError(
            f"source corpus is only {corpus_bytes} bytes; need >= {4 * L2_BYTES}"
        )

    max_weight = max(n * k for n, k in shapes)
    max_scale = max((n // 128) * (k // 128) for n, k in shapes)
    if args.target_mode == "rotating" and args.target_slots < 64:
        raise ValueError("rotating mode requires at least 64 target slots")
    target_count = 1 if args.target_mode == "fixed" else args.target_slots
    fp8_scratch = None
    fp32_scratch = None
    if not args.cached_weight:
        fp8_scratch = torch.empty(
            (target_count, max_weight), dtype=torch.float8_e4m3fn, device="cuda"
        )
        fp32_scratch = torch.empty(
            (target_count, max_scale), dtype=torch.float32, device="cuda"
        )
    hidden = {
        (m, k): torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
        for m in args.m
        for k in {k for _, k in shapes}
    }
    transitions = tuple(itertools.product(range(len(shapes)), current_indices))
    print(
        json.dumps(
            {
                "record": "metadata",
                "pool": "global",
                "n_axis": list(dict.fromkeys(args.n)),
                "k_axis": list(dict.fromkeys(args.k)),
                "shapes": [
                    {
                        "n": n,
                        "k": k,
                    }
                    for n, k in shapes
                ],
                "current_shapes": [
                    {"n": shapes[index][0], "k": shapes[index][1]}
                    for index in current_indices
                ],
                "transition_count": len(transitions),
                "source_copies": copies,
                "source_corpus_bytes": corpus_bytes,
                "source_rotation": (
                    "distinct address for every predecessor/current launch; "
                    "no reuse across consecutive backend sequences"
                ),
                "target_mode": args.target_mode,
                "cached_weight": args.cached_weight,
                "target_slots": target_count,
                "target_corpus_bytes": target_count * (max_weight + max_scale * 4),
                "scratch": (
                    "one fixed max-sized FP8 buffer and one FP32 buffer"
                    if args.target_mode == "fixed"
                    else f"{target_count} rotating max-sized FP8 and FP32 buffers"
                ),
                "dispatch_policy": "unmeasured M,N,K must fall back to W4A16",
            },
            sort_keys=True,
        ),
        flush=True,
    )

    target_cursor = [0]

    def run(backend: str, shape, layer, x):
        n, k = shape
        if backend == "w4a16":
            return apply_fp4_marlin_linear(
                x,
                layer.weight,
                layer.weight_scale,
                layer.weight_global_scale,
                layer.workspace,
                n,
                k,
            )
        x_fp8, x_scale = per_token_group_quant_fp8(
            x,
            128,
            dtype=torch.float8_e4m3fn,
            column_major_scales=True,
            use_ue8m0=False,
        )
        if args.cached_weight:
            return ops.cutlass_scaled_mm(
                x_fp8,
                layer.cached_fp8_weight.T,
                scale_a=x_scale,
                scale_b=layer.cached_fp8_scale.T,
                out_dtype=torch.bfloat16,
            )
        target = 0 if args.target_mode == "fixed" else target_cursor[0]
        target_cursor[0] = (target + 1) % target_count
        weight = fp8_scratch[target, : n * k].view(n, k)
        weight_scale = fp32_scratch[target, : (n // 128) * (k // 128)].view(
            n // 128, k // 128
        )
        ops.marlin_nvfp4_to_fp8(
            weight,
            weight_scale,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.weight_fp8_scale_divisor_code,
            torch.bfloat16,
        )
        return ops.cutlass_scaled_mm(
            x_fp8,
            weight.T,
            scale_a=x_scale,
            scale_b=weight_scale.T,
            out_dtype=torch.bfloat16,
        )

    source_cursor = dict.fromkeys(shapes, 0)

    def next_source(shape):
        cursor = source_cursor[shape]
        source_cursor[shape] = (cursor + 1) % copies
        return sources[shape][cursor]

    for m in args.m:
        samples = {backend: [[] for _ in transitions] for backend in ("w4a16", "w4a8")}
        totals = {backend: [] for backend in ("w4a16", "w4a8")}
        for repetition in range(-args.warmup, args.repetitions):
            for backend in ("w4a16", "w4a8") if repetition % 2 else ("w4a8", "w4a16"):
                total_begin = torch.cuda.Event(enable_timing=True)
                total_end = torch.cuda.Event(enable_timing=True)
                events = []
                total_begin.record()
                for position, (pred_index, current_index) in enumerate(transitions):
                    pred_shape, current_shape = (
                        shapes[pred_index],
                        shapes[current_index],
                    )
                    run(
                        backend,
                        pred_shape,
                        next_source(pred_shape),
                        hidden[(m, pred_shape[1])],
                    )
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
                    run(
                        backend,
                        current_shape,
                        next_source(current_shape),
                        hidden[(m, current_shape[1])],
                    )
                    end.record()
                    events.append((begin, end))
                total_end.record()
                torch.accelerator.synchronize()
                if repetition >= 0:
                    totals[backend].append(total_begin.elapsed_time(total_end) * 1000)
                    for position, (begin, end) in enumerate(events):
                        samples[backend][position].append(
                            begin.elapsed_time(end) * 1000
                        )

        for position, (pred_index, current_index) in enumerate(transitions):
            pred_n, pred_k = shapes[pred_index]
            current_n, current_k = shapes[current_index]
            w4a16_us = median(samples["w4a16"][position])
            w4a8_us = median(samples["w4a8"][position])
            print(
                json.dumps(
                    {
                        "record": "transition",
                        "pool": "global",
                        "target_mode": args.target_mode,
                        "m": m,
                        "position": position,
                        "predecessor": {"n": pred_n, "k": pred_k},
                        "current": {"n": current_n, "k": current_k},
                        "w4a16_us_samples": samples["w4a16"][position],
                        "w4a8_us_samples": samples["w4a8"][position],
                        "w4a16_us": w4a16_us,
                        "w4a8_us": w4a8_us,
                        "delta_us": w4a8_us - w4a16_us,
                        "w4a8_over_w4a16": w4a8_us / w4a16_us,
                        "w4a16_over_w4a8_speedup": w4a16_us / w4a8_us,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        for current_index in current_indices:
            current_n, current_k = shapes[current_index]
            positions = [
                position
                for position, (_, index) in enumerate(transitions)
                if index == current_index
            ]
            w4a16_values = [median(samples["w4a16"][p]) for p in positions]
            w4a8_values = [median(samples["w4a8"][p]) for p in positions]
            deltas = [b - a for a, b in zip(w4a16_values, w4a8_values)]
            ratios = [b / a for a, b in zip(w4a16_values, w4a8_values)]
            speedups = [a / b for a, b in zip(w4a16_values, w4a8_values)]

            def stats(values):
                return {
                    "min": min(values),
                    "median": median(values),
                    "max": max(values),
                }

            print(
                json.dumps(
                    {
                        "record": "current_summary",
                        "pool": "global",
                        "target_mode": args.target_mode,
                        "m": m,
                        "current": {"n": current_n, "k": current_k},
                        "aggregation": "unweighted_across_predecessors",
                        "predecessor_count": len(positions),
                        "w4a16_us": stats(w4a16_values),
                        "w4a8_us": stats(w4a8_values),
                        "delta_us": stats(deltas),
                        "w4a8_over_w4a16": stats(ratios),
                        "w4a16_over_w4a8_speedup": stats(speedups),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        transition_w4a16_sum = sum(
            median(samples["w4a16"][position]) for position in range(len(transitions))
        )
        transition_w4a8_sum = sum(
            median(samples["w4a8"][position]) for position in range(len(transitions))
        )
        print(
            json.dumps(
                {
                    "record": "unweighted_sum",
                    "pool": "global",
                    "target_mode": args.target_mode,
                    "m": m,
                    "aggregation": "unweighted_sum_of_transition_medians",
                    "transition_count": len(transitions),
                    "w4a16_us": transition_w4a16_sum,
                    "w4a8_us": transition_w4a8_sum,
                    "delta_us": transition_w4a8_sum - transition_w4a16_sum,
                    "w4a8_over_w4a16": transition_w4a8_sum / transition_w4a16_sum,
                    "w4a16_over_w4a8_speedup": transition_w4a16_sum
                    / transition_w4a8_sum,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        w4a16_total = median(totals["w4a16"])
        w4a8_total = median(totals["w4a8"])
        print(
            json.dumps(
                {
                    "record": "sequence",
                    "pool": "global",
                    "target_mode": args.target_mode,
                    "m": m,
                    "aggregation": "measured_ordered_sequence_including_predecessors",
                    "transition_count": len(transitions),
                    "w4a16_us_samples": totals["w4a16"],
                    "w4a8_us_samples": totals["w4a8"],
                    "w4a16_us": w4a16_total,
                    "w4a8_us": w4a8_total,
                    "delta_us": w4a8_total - w4a16_total,
                    "w4a8_over_w4a16": w4a8_total / w4a16_total,
                    "w4a16_over_w4a8_speedup": w4a16_total / w4a8_total,
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", nargs="+", type=int, default=list(DEFAULT_AXIS))
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_AXIS))
    parser.add_argument(
        "--m",
        nargs="+",
        type=int,
        default=[
            1,
            64,
            128,
            256,
            384,
            512,
            768,
            1024,
            1536,
            2048,
            3072,
            4096,
        ],
    )
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--source-copies", type=int, default=0)
    parser.add_argument("--current-index", nargs="+", type=int)
    parser.add_argument("--target-mode", choices=("fixed", "rotating"), default="fixed")
    parser.add_argument("--target-slots", type=int, default=64)
    parser.add_argument("--cached-weight", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        check()
        return
    run_sweep(args)


if __name__ == "__main__":
    main()
