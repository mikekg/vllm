# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from math import prod
from types import SimpleNamespace

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_repacked_nk,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    apply_fp4_marlin_linear,
    prepare_fp4_layer_for_marlin,
    prepare_nvfp4_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform


def _native_converter_available() -> bool:
    return (
        current_platform.is_cuda()
        and (
            current_platform.is_device_capability(89)
            or current_platform.is_device_capability(90)
        )
        and hasattr(torch.ops._C, "marlin_nvfp4_to_fp8")
    )


pytestmark = pytest.mark.skipif(
    not _native_converter_available(),
    reason="The native Marlin NVFP4-to-FP8 converter requires SM89 or SM90.",
)

_E2M1 = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
)


def _pack_e2m1(codes: torch.Tensor) -> torch.Tensor:
    return (codes[..., 0::2] | (codes[..., 1::2] << 4)).to(torch.uint8)


def _canonical_weight(
    codes: torch.Tensor,
    block_scales: torch.Tensor,
    global_scale: torch.Tensor,
) -> torch.Tensor:
    values = torch.tensor(_E2M1, dtype=torch.float32, device=codes.device)[codes.long()]
    globals_ = global_scale.float().reshape(
        (global_scale.numel(),) + (1,) * (codes.dim() - 1)
    )
    return values * block_scales.float().repeat_interleave(16, dim=-1) * globals_


def _satfinite_e4m3_bytes(values: torch.Tensor) -> torch.Tensor:
    """Encode FP16 with an independent E4M3FN RNE/SATFINITE oracle."""
    positive = []
    for code in range(127):
        exponent, mantissa = (code >> 3) & 15, code & 7
        positive.append(
            mantissa * 2.0**-9
            if exponent == 0
            else (1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7)
        )
    grid = torch.tensor(positive, dtype=torch.float32, device=values.device)
    raw = values.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    negative = (raw & 0x8000) != 0
    magnitude = values.abs().float().clamp(max=448.0)
    upper = torch.searchsorted(grid, magnitude).clamp(max=126)
    lower = (upper - 1).clamp(min=0)
    lower_distance = magnitude - grid[lower]
    upper_distance = grid[upper] - magnitude
    choose_upper = (upper_distance < lower_distance) | (
        (upper_distance == lower_distance) & ((upper & 1) == 0)
    )
    code = torch.where(choose_upper, upper, lower)
    return (code | (negative.to(torch.int64) << 7)).to(torch.uint8)


def _expected_weight(
    codes: torch.Tensor,
    processed_scales: torch.Tensor,
    tile_scale_divisor_codes: torch.Tensor,
) -> torch.Tensor:
    n, k = codes.shape
    rows = torch.arange(n, device=codes.device)[:, None]
    groups = torch.arange(k // 16, device=codes.device)[None, :]
    lane = rows & 63
    transposed = 8 * (lane & 7) + (lane >> 3)
    reordered = (transposed & ~3) | ((transposed & 1) << 1) | ((transposed & 2) >> 1)
    scale_indices = groups * n + 64 * (rows >> 6) + reordered
    scale_bytes = processed_scales.view(torch.uint8).flatten()[scale_indices]
    half_bits = (scale_bytes.to(torch.int32) << 7).to(torch.int16)
    local_scales = half_bits.view(torch.float16).repeat_interleave(16, 1)
    tile_scale_reciprocals = (
        _decode_s0e5m3(tile_scale_divisor_codes)
        .float()
        .reciprocal()
        .repeat_interleave(128, dim=-2)
        .repeat_interleave(128, dim=-1)[:n, :k]
    )
    normalized_scales = (
        (local_scales.float() * tile_scale_reciprocals)
        .clamp(max=65504)
        .to(torch.float16)
    )
    values = torch.tensor(_E2M1, dtype=torch.float16, device=codes.device)[codes.long()]
    return _satfinite_e4m3_bytes((values * normalized_scales).to(torch.float16))


def _decode_s0e5m3(codes: torch.Tensor) -> torch.Tensor:
    bits = (codes.to(torch.int32) << 7).to(torch.int16)
    return bits.view(torch.float16)


def _expand_tile_scales(tile_scales: torch.Tensor, n: int, k: int) -> torch.Tensor:
    return tile_scales.repeat_interleave(128, dim=-2).repeat_interleave(128, dim=-1)[
        ..., :n, :k
    ]


def _expected_global_scale(
    processed_global_scale: torch.Tensor,
    resident_dtype: torch.dtype,
    tile_scale_divisor_codes: torch.Tensor,
) -> torch.Tensor:
    exponent_bias = 14 if resident_dtype == torch.float16 else 126
    processed_global_compensation = processed_global_scale.float() * 2.0**-exponent_bias
    processed_global_compensation = processed_global_compensation.view(
        (processed_global_scale.numel(),) + (1,) * (tile_scale_divisor_codes.dim() - 1)
    )
    if tile_scale_divisor_codes.dim() == 2:
        processed_global_compensation = processed_global_compensation[0]
    return (
        processed_global_compensation * _decode_s0e5m3(tile_scale_divisor_codes).float()
    )


def _guarded_output(
    shape: tuple[int, ...], dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    byte_count = prod(shape) * dtype.itemsize
    storage = torch.full((byte_count + 32,), 0xA5, dtype=torch.uint8, device="cuda")
    return storage, storage[16 : 16 + byte_count].view(dtype).view(shape)


def _assert_canaries(storage: torch.Tensor) -> None:
    expected = torch.full((16,), 0xA5, dtype=torch.uint8, device="cuda")
    torch.testing.assert_close(storage[:16], expected)
    torch.testing.assert_close(storage[-16:], expected)


def _run_converter(
    packed_weight: torch.Tensor,
    processed_scales: torch.Tensor,
    processed_global_scale: torch.Tensor,
    tile_scale_divisor_codes: torch.Tensor,
    resident_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    packed = packed_weight[0] if packed_weight.dim() == 3 else packed_weight
    n, resident_k = marlin_repacked_nk(packed, 4)
    scratch_k = (resident_k + 127) // 128 * 128
    shape = (
        (packed_weight.size(0), n, scratch_k)
        if packed_weight.dim() == 3
        else (n, scratch_k)
    )
    weight_storage, output = _guarded_output(shape, torch.float8_e4m3fn)
    scale_storage, output_scale = _guarded_output(
        tuple(tile_scale_divisor_codes.shape), torch.float32
    )
    ops.marlin_nvfp4_to_fp8(
        output,
        output_scale,
        packed_weight,
        processed_scales,
        processed_global_scale,
        tile_scale_divisor_codes,
        resident_dtype,
    )
    return output, output_scale, weight_storage, scale_storage


def _dense_marlin_inputs(
    codes: torch.Tensor,
    resident_dtype: torch.dtype,
    scales: torch.Tensor | None = None,
    global_scale: float = 0.37,
) -> SimpleNamespace:
    n, k = codes.shape
    if scales is None:
        scales = torch.linspace(
            0.25,
            1.0,
            n * (k // 16),
            dtype=torch.float16,
            device="cuda",
        ).view(n, k // 16)
    layer = SimpleNamespace(
        output_size_per_partition=n,
        input_size_per_partition=k,
        params_dtype=resident_dtype,
        weight=torch.nn.Parameter(_pack_e2m1(codes), requires_grad=False),
        weight_scale=torch.nn.Parameter(
            scales.to(torch.float8_e4m3fn), requires_grad=False
        ),
        weight_global_scale=torch.nn.Parameter(
            torch.tensor(global_scale, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        ),
    )
    prepare_fp4_layer_for_marlin(layer)
    return layer


def _exact_dense_checkpoint(
    n: int, k: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(11)
    codes = (
        torch.randperm(n * k, device="cuda", generator=generator)
        .remainder(16)
        .view(n, k)
        .to(torch.uint8)
    )
    scale_levels = torch.tensor(
        [0.0234375, 0.046875, 0.09375, 0.1875, 0.375, 0.75, 1.5, 3, 6, 12, 24],
        dtype=torch.float16,
        device="cuda",
    )
    scale_indices = (
        torch.randperm(n * (k // 16), device="cuda", generator=generator)
        .remainder(scale_levels.numel())
        .view(n, k // 16)
    )
    block_scales = scale_levels[scale_indices].to(torch.float8_e4m3fn)
    global_scale = torch.tensor(0.37, dtype=torch.float32, device="cuda")
    return codes, block_scales, global_scale


@pytest.mark.parametrize("resident_dtype", [torch.float16, torch.bfloat16])
def test_dense_real_marlin_pack_matches_independent_oracle(
    resident_dtype: torch.dtype,
) -> None:
    n, k = 256, 256
    codes = (
        torch.arange(n * k, dtype=torch.int32, device="cuda")
        .view(n, k)
        .remainder_(16)
        .to(torch.uint8)
    )
    scales = torch.empty((n, k // 16), dtype=torch.float16, device="cuda")
    scales[:128, :8] = 0.25
    scales[:128, 8:] = 0.5
    scales[128:, :8] = 1.0
    scales[128:, 8:] = 2.0
    layer = _dense_marlin_inputs(codes, resident_dtype, scales)
    divisor_codes = layer.weight_fp8_scale_divisor_code
    assert divisor_codes.shape == (2, 2)
    assert divisor_codes.unique().numel() == 4

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        output, output_scale, weight_storage, scale_storage = _run_converter(
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            divisor_codes,
            resident_dtype,
        )
    torch.cuda.current_stream().wait_stream(stream)

    expected = _expected_weight(codes, layer.weight_scale, divisor_codes)
    torch.testing.assert_close(output.view(torch.uint8), expected)
    torch.testing.assert_close(
        output_scale,
        _expected_global_scale(
            layer.weight_global_scale, resident_dtype, divisor_codes
        ),
        rtol=0,
        atol=0,
    )
    _assert_canaries(weight_storage)
    _assert_canaries(scale_storage)


@pytest.mark.skipif(
    not current_platform.is_device_capability(90),
    reason="The specialized converter dispatch requires SM90.",
)
@pytest.mark.parametrize("resident_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    ("n_tiles", "k_tiles"),
    [(8, 8), (5, 13)],
    ids=["sparse-max", "paired-min"],
)
def test_dense_sm90_dispatch_boundaries_match_independent_oracle(
    n_tiles: int,
    k_tiles: int,
    resident_dtype: torch.dtype,
) -> None:
    n, k = 128 * n_tiles, 128 * k_tiles
    codes = (
        torch.arange(k, dtype=torch.int32, device="cuda")
        .remainder_(16)
        .to(torch.uint8)
        .expand(n, k)
    )
    layer = _dense_marlin_inputs(codes, resident_dtype)

    output, output_scale, weight_storage, scale_storage = _run_converter(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.weight_fp8_scale_divisor_code,
        resident_dtype,
    )

    torch.testing.assert_close(
        output.view(torch.uint8),
        _expected_weight(
            codes, layer.weight_scale, layer.weight_fp8_scale_divisor_code
        ),
    )
    torch.testing.assert_close(
        output_scale,
        _expected_global_scale(
            layer.weight_global_scale,
            resident_dtype,
            layer.weight_fp8_scale_divisor_code,
        ),
        rtol=0,
        atol=0,
    )
    _assert_canaries(weight_storage)
    _assert_canaries(scale_storage)


@pytest.mark.parametrize("resident_dtype", [torch.float16, torch.bfloat16])
def test_dense_real_marlin_round_trip_reconstructs_checkpoint(
    resident_dtype: torch.dtype,
) -> None:
    n, k = 128, 128
    codes, block_scales, global_scale = _exact_dense_checkpoint(n, k)
    layer = _dense_marlin_inputs(
        codes,
        resident_dtype,
        block_scales,
        global_scale.item(),
    )

    output, output_scale, _, _ = _run_converter(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.weight_fp8_scale_divisor_code,
        resident_dtype,
    )

    reconstructed = output.float() * _expand_tile_scales(output_scale, n, k)
    canonical = _canonical_weight(codes, block_scales, global_scale)
    assert torch.equal(reconstructed == 0, canonical == 0)
    relative_error = ((reconstructed - canonical) / canonical).abs()[canonical != 0]
    assert relative_error.max() < 0.08


@pytest.mark.parametrize("k", [64, 192])
def test_dense_partial_k_uses_zero_padded_fp8_scratch(k: int) -> None:
    n = 128
    codes, block_scales, global_scale = _exact_dense_checkpoint(n, k)
    layer = _dense_marlin_inputs(
        codes,
        torch.bfloat16,
        block_scales,
        global_scale.item(),
    )

    output, output_scale, weight_storage, scale_storage = _run_converter(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.weight_fp8_scale_divisor_code,
        torch.bfloat16,
    )

    scratch_k = (k + 127) // 128 * 128
    assert output.shape == (n, scratch_k)
    torch.testing.assert_close(
        output[:, :k].view(torch.uint8),
        _expected_weight(
            codes, layer.weight_scale, layer.weight_fp8_scale_divisor_code
        ),
    )
    assert not output[:, k:].view(torch.uint8).any()
    torch.testing.assert_close(
        output_scale,
        _expected_global_scale(
            layer.weight_global_scale,
            torch.bfloat16,
            layer.weight_fp8_scale_divisor_code,
        ),
        rtol=0,
        atol=0,
    )
    _assert_canaries(weight_storage)
    _assert_canaries(scale_storage)


@pytest.mark.usefixtures("default_vllm_config")
@pytest.mark.skipif(
    not current_platform.is_device_capability(90),
    reason="The block-scaled CUDA CUTLASS consumer requires SM90.",
)
@torch.inference_mode()
def test_dense_real_marlin_gemm_matches_quant_fp8_cutlass_consumer() -> None:
    n, k, m = 256, 256, 32
    codes, block_scales, global_scale = _exact_dense_checkpoint(n, k)
    layer = _dense_marlin_inputs(
        codes,
        torch.bfloat16,
        block_scales,
        global_scale.item(),
    )
    hidden = torch.randn((m, k + 8), dtype=torch.bfloat16, device="cuda")[:, :k]

    marlin = apply_fp4_marlin_linear(
        hidden,
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.workspace,
        n,
        k,
    )
    fp8_weight, weight_scale, _, _ = _run_converter(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.weight_fp8_scale_divisor_code,
        torch.bfloat16,
    )
    fp8_input, input_scale = QuantFP8(
        static=False,
        group_shape=GroupShape(1, 128),
        column_major_scales=True,
        use_ue8m0=False,
    )(hidden)
    cutlass = ops.cutlass_scaled_mm(
        fp8_input,
        fp8_weight.T,
        scale_a=input_scale,
        scale_b=weight_scale.T,
        out_dtype=torch.bfloat16,
    )

    canonical_weight = _canonical_weight(codes, block_scales, global_scale)
    converted_weight = fp8_weight.float() * _expand_tile_scales(weight_scale, n, k)
    converted_input = (
        fp8_input.float() * input_scale.float().repeat_interleave(128, dim=-1)[..., :k]
    )
    converted_reference = (converted_input @ converted_weight.T).to(torch.bfloat16)
    cutlass_error = (cutlass.float() - converted_reference.float()).abs().mean()
    assert cutlass_error / converted_reference.float().abs().mean() < 0.01

    canonical_reference = (hidden.float() @ canonical_weight.T).to(torch.bfloat16)
    torch.testing.assert_close(marlin, canonical_reference, rtol=0.01, atol=0.1)
    conversion_error = (cutlass.float() - marlin.float()).abs().mean()
    assert conversion_error / marlin.float().abs().mean() < 0.08


@pytest.mark.skipif(
    not current_platform.is_device_capability(90)
    or not hasattr(torch.ops._C, "marlin_nvfp4_hybrid_linear"),
    reason="The native hybrid operator requires SM90.",
)
def test_native_hybrid_is_functional() -> None:
    schema = torch.ops._C.marlin_nvfp4_hybrid_linear.default._schema
    assert all(argument.alias_info is None for argument in schema.arguments)


@pytest.mark.skipif(
    not current_platform.is_device_capability(90)
    or not hasattr(torch.ops._C, "marlin_nvfp4_hybrid_linear"),
    reason="The native hybrid operator requires SM90.",
)
@pytest.mark.parametrize(("m", "m_knee"), [(1, 2), (3, 1)])
@torch.inference_mode()
def test_native_hybrid_selects_marlin_or_padded_fp8(m: int, m_knee: int) -> None:
    n = k = 128
    codes, block_scales, global_scale = _exact_dense_checkpoint(n, k)
    layer = _dense_marlin_inputs(
        codes, torch.bfloat16, block_scales, global_scale.item()
    )
    hidden = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    reference = apply_fp4_marlin_linear(
        hidden,
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.workspace,
        n,
        k,
    )
    actual = ops.marlin_nvfp4_hybrid_linear(
        hidden,
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.weight_fp8_scale_divisor_code,
        layer.workspace,
        m_knee,
        False,
        False,
    )

    if m < m_knee:
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    else:
        error = (actual.float() - reference.float()).abs().mean()
        assert error / reference.float().abs().mean() < 0.08


def test_zero_tile_uses_identity_divisor_without_a_special_runtime_path() -> None:
    n = k = 128
    codes = torch.zeros((n, k), dtype=torch.uint8, device="cuda")
    layer = _dense_marlin_inputs(codes, torch.float16)
    divisor_codes = layer.weight_fp8_scale_divisor_code
    assert divisor_codes.item() == 0x78

    output, output_scale, _, _ = _run_converter(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        divisor_codes,
        torch.float16,
    )
    assert not output.view(torch.uint8).any()
    torch.testing.assert_close(
        output_scale,
        _expected_global_scale(layer.weight_global_scale, torch.float16, divisor_codes),
        rtol=0,
        atol=0,
    )


def test_max_tile_uses_representable_divisor_without_overflow() -> None:
    n = k = 128
    codes = torch.full((n, k), 7, dtype=torch.uint8, device="cuda")
    scales = torch.full((n, k // 16), 448, dtype=torch.float16, device="cuda")
    layer = _dense_marlin_inputs(codes, torch.float16, scales)
    divisor_codes = layer.weight_fp8_scale_divisor_code
    assert _decode_s0e5m3(divisor_codes).item() == 768

    output, _, _, _ = _run_converter(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        divisor_codes,
        torch.float16,
    )
    expected = _expected_weight(codes, layer.weight_scale, divisor_codes)
    torch.testing.assert_close(output.view(torch.uint8), expected)
    assert torch.all(expected == 0x7E)


def test_zero_group_with_large_scale_stays_zero() -> None:
    n = k = 128
    codes = torch.zeros((n, k), dtype=torch.uint8, device="cuda")
    codes[0, :16] = 1
    scales = torch.full((n, k // 16), 448, dtype=torch.float16, device="cuda")
    scales[0, 0] = 1 / 64
    layer = _dense_marlin_inputs(codes, torch.float16, scales)
    divisor_codes = layer.weight_fp8_scale_divisor_code
    assert divisor_codes.item() == 0x34

    output, _, _, _ = _run_converter(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        divisor_codes,
        torch.float16,
    )
    output_bytes = output.view(torch.uint8)
    assert not torch.isnan(output.float()).any()
    assert not output_bytes[codes == 0].any()
    torch.testing.assert_close(
        output_bytes,
        _expected_weight(codes, layer.weight_scale, divisor_codes),
    )


@pytest.mark.parametrize("resident_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(("experts", "n"), [(3, 64), (2, 128)])
def test_moe_real_marlin_round_trip_matches_rank3_oracle(
    resident_dtype: torch.dtype,
    experts: int,
    n: int,
) -> None:
    k = 128
    generator = torch.Generator(device="cuda").manual_seed(23)
    expert_ids = torch.arange(experts, device="cuda")[:, None, None]
    w13_code_base = (
        torch.randperm(2 * n * k, device="cuda", generator=generator)
        .remainder(16)
        .view(1, 2 * n, k)
    )
    w2_code_base = (
        torch.randperm(k * n, device="cuda", generator=generator)
        .remainder(16)
        .view(1, k, n)
    )
    w13_codes = (w13_code_base + expert_ids).remainder(16).to(torch.uint8)
    w2_codes = (w2_code_base + 3 * expert_ids).remainder(16).to(torch.uint8)
    scale_levels = torch.tensor(
        [3.0, 6.0, 12.0, 24.0], dtype=torch.float16, device="cuda"
    )
    w13_scale_base = (
        torch.randperm(2 * n * (k // 16), device="cuda", generator=generator)
        .remainder(scale_levels.numel())
        .view(1, 2 * n, k // 16)
    )
    w2_scale_base = (
        torch.randperm(k * (n // 16), device="cuda", generator=generator)
        .remainder(scale_levels.numel())
        .view(1, k, n // 16)
    )
    w13_scale_indices = (w13_scale_base + expert_ids).remainder(scale_levels.numel())
    w2_scale_indices = (w2_scale_base + 3 * expert_ids).remainder(scale_levels.numel())
    w13_scales = scale_levels[w13_scale_indices].to(torch.float8_e4m3fn)
    w2_scales = scale_levels[w2_scale_indices].to(torch.float8_e4m3fn)
    w13_global = torch.linspace(0.25, 0.75, experts, dtype=torch.float32, device="cuda")
    w2_global = torch.linspace(0.5, 1.0, experts, dtype=torch.float32, device="cuda")
    layer = SimpleNamespace(
        num_experts=experts,
        hidden_size=k,
        intermediate_size_per_partition=n,
        params_dtype=resident_dtype,
    )
    canonical = (
        (w13_codes, w13_scales, w13_global),
        (w2_codes, w2_scales, w2_global),
    )
    prepared = prepare_nvfp4_moe_layer_for_marlin(
        layer,
        _pack_e2m1(w13_codes),
        w13_scales,
        w13_global,
        _pack_e2m1(w2_codes),
        w2_scales,
        w2_global,
        is_act_and_mul=True,
    )

    divisor_codes = (
        layer.w13_fp8_scale_divisor_code,
        layer.w2_fp8_scale_divisor_code,
    )
    for (
        (codes, block_scales, global_scale),
        (weight, scales, processed_global),
        tile_divisor_codes,
    ) in zip(
        canonical,
        (prepared[:3], prepared[3:]),
        divisor_codes,
    ):
        output, output_scale, weight_storage, scale_storage = _run_converter(
            weight, scales, processed_global, tile_divisor_codes, resident_dtype
        )
        expected = torch.stack(
            [
                _expected_weight(
                    codes[expert], scales[expert], tile_divisor_codes[expert]
                )
                for expert in range(experts)
            ]
        )
        logical_output = output[..., : codes.size(-1)]
        torch.testing.assert_close(logical_output.view(torch.uint8), expected)
        assert not output[..., codes.size(-1) :].view(torch.uint8).any()
        torch.testing.assert_close(
            output_scale,
            _expected_global_scale(
                processed_global, resident_dtype, tile_divisor_codes
            ),
            rtol=0,
            atol=0,
        )
        reconstructed = logical_output.float() * _expand_tile_scales(
            output_scale, codes.size(-2), codes.size(-1)
        )
        canonical_weight = _canonical_weight(codes, block_scales, global_scale)
        assert torch.equal(reconstructed == 0, canonical_weight == 0)
        relative_error = ((reconstructed - canonical_weight) / canonical_weight).abs()
        assert relative_error[canonical_weight != 0].max() < 0.08
        _assert_canaries(weight_storage)
        _assert_canaries(scale_storage)


def test_converter_replays_in_cuda_graph_with_stable_outputs() -> None:
    n, k = 64, 128
    codes = torch.randint(0, 16, (n, k), dtype=torch.uint8, device="cuda")
    layer = _dense_marlin_inputs(codes, torch.float16)
    output, output_scale, _, _ = _run_converter(
        layer.weight,
        layer.weight_scale,
        layer.weight_global_scale,
        layer.weight_fp8_scale_divisor_code,
        torch.float16,
    )
    weight_pointer, scale_pointer = output.data_ptr(), output_scale.data_ptr()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        ops.marlin_nvfp4_to_fp8(
            output,
            output_scale,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.weight_fp8_scale_divisor_code,
            torch.float16,
        )
    output.zero_()
    output_scale.zero_()
    graph.replay()
    graph.replay()

    assert output.data_ptr() == weight_pointer
    assert output_scale.data_ptr() == scale_pointer
    torch.testing.assert_close(
        output.view(torch.uint8),
        _expected_weight(
            codes, layer.weight_scale, layer.weight_fp8_scale_divisor_code
        ),
    )


def test_converter_rejects_output_aliasing_resident_input() -> None:
    n, k = 64, 128
    codes = torch.zeros((n, k), dtype=torch.uint8, device="cuda")
    layer = _dense_marlin_inputs(codes, torch.float16)
    output = torch.empty((n, k), dtype=torch.float8_e4m3fn, device="cuda")

    with pytest.raises(RuntimeError, match="must not overlap resident inputs"):
        ops.marlin_nvfp4_to_fp8(
            output,
            layer.weight_global_scale.view(1, 1),
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.weight_fp8_scale_divisor_code,
            torch.float16,
        )
