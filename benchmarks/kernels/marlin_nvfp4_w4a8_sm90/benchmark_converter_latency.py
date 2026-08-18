# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from statistics import median
from types import SimpleNamespace

TRAIN_SHAPES = tuple(
    (n, k)
    for n in (256, 512, 1024, 2048, 4096)
    for k in (512, 1024, 2048, 4096, 8192, 16384)
)
HOLDOUT_SHAPES = (
    (384, 768),
    (384, 20480),
    (640, 1536),
    (768, 3072),
    (1280, 6144),
    (1536, 10240),
    (2560, 768),
    (3072, 5120),
    (3584, 12288),
)
TIMED_LAUNCHES = 7 * 64


def make_source(n, k, torch, prepare_fp4_layer_for_marlin):
    packed = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    scale = torch.ones((n, k // 16), dtype=torch.float8_e4m3fn, device="cuda")
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
    return layer


def tensor_bytes(tensor):
    return tensor.numel() * tensor.element_size()


def main():
    import torch

    from vllm import _custom_ops as ops
    from vllm.model_executor.kernels.linear.nvfp4 import (
        marlin_fp8 as _marlin_fp8,  # noqa: F401
    )
    from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
        prepare_fp4_layer_for_marlin,
    )

    assert torch.cuda.get_device_capability() == (9, 0)
    l2_bytes = torch.cuda.get_device_properties(0).L2_cache_size
    print(
        json.dumps(
            {
                "record": "metadata",
                "kernel": "production_double_buffer",
                "l2_bytes": l2_bytes,
                "timed_launches": TIMED_LAUNCHES,
                "source_policy": "distinct address for every timed launch",
                "target_policy": "fixed",
                "train_shapes": TRAIN_SHAPES,
                "holdout_shapes": HOLDOUT_SHAPES,
            }
        ),
        flush=True,
    )

    for split, shapes in (("train", TRAIN_SHAPES), ("holdout", HOLDOUT_SHAPES)):
        for n, k in shapes:
            layer = make_source(n, k, torch, prepare_fp4_layer_for_marlin)
            bytes_per_source = tensor_bytes(layer.weight) + tensor_bytes(
                layer.weight_scale
            )
            source_count = max(
                TIMED_LAUNCHES,
                (4 * l2_bytes + bytes_per_source - 1) // bytes_per_source,
            )
            packed = (
                layer.weight.unsqueeze(0)
                .expand(source_count, *layer.weight.shape)
                .clone()
            )
            scales = (
                layer.weight_scale.unsqueeze(0)
                .expand(source_count, *layer.weight_scale.shape)
                .clone()
            )
            source_corpus_bytes = tensor_bytes(packed) + tensor_bytes(scales)
            if source_corpus_bytes < 4 * l2_bytes:
                raise RuntimeError(
                    f"source corpus {source_corpus_bytes} is smaller than 4x L2"
                )
            output = torch.empty((n, k), dtype=torch.float8_e4m3fn, device="cuda")
            output_scale = torch.empty(
                (n // 128, k // 128), dtype=torch.float32, device="cuda"
            )

            for _ in range(16):
                ops.marlin_nvfp4_to_fp8(
                    output,
                    output_scale,
                    layer.weight,
                    layer.weight_scale,
                    layer.weight_global_scale,
                    layer.weight_fp8_scale_divisor_code,
                    torch.bfloat16,
                )
            torch.accelerator.synchronize()

            samples = []
            source = 0
            for _ in range(7):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(64):
                    ops.marlin_nvfp4_to_fp8(
                        output,
                        output_scale,
                        packed[source],
                        scales[source],
                        layer.weight_global_scale,
                        layer.weight_fp8_scale_divisor_code,
                        torch.bfloat16,
                    )
                    source += 1
                end.record()
                end.synchronize()
                samples.append(begin.elapsed_time(end) * 1000 / 64)

            print(
                json.dumps(
                    {
                        "record": "latency",
                        "split": split,
                        "n": n,
                        "k": k,
                        "nk": n * k,
                        "samples_us": samples,
                        "median_us": median(samples),
                        "source_corpus_bytes": source_corpus_bytes,
                        "source_count": source_count,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del packed, scales, output, output_scale, layer
            torch.accelerator.empty_cache()


if __name__ == "__main__":
    main()
