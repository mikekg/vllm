# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from tests.kernels.moe.utils import make_dummy_moe_config
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    nvfp4_w4a16_moe_quant_config,
)
from vllm.model_executor.layers.fused_moe.experts import nvfp4_bycopy_moe
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.experts.nvfp4_bycopy_moe import (
    NvFp4ByCopyExperts,
    NvFp4ToFp8Experts,
    NvFp4ToFp8TritonExperts,
    _lookup_moe_m_knee,
    _moe_shape,
)
from vllm.model_executor.layers.fused_moe.oracle import nvfp4 as nvfp4_oracle
from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    prepare_nvfp4_moe_layer_for_marlin,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import kNvfp4Static
from vllm.platforms import current_platform


def _config(
    *,
    e: int = 128,
    p: int = 768,
    k: int = 2048,
    topk: int = 8,
    dtype: torch.dtype = torch.bfloat16,
):
    return make_dummy_moe_config(
        num_experts=e,
        num_local_experts=e,
        experts_per_token=topk,
        hidden_dim=k,
        intermediate_size=p,
        in_dtype=dtype,
        max_num_tokens=8192,
    )


def test_universal_knee_and_post_dispatch_selection():
    assert (
        _lookup_moe_m_knee("unknown GPU", torch.float16, (128, 704, 2048, 8, True))
        == 4097
    )
    assert (
        _lookup_moe_m_knee("unknown GPU", torch.float16, (256, 512, 2048, 8, True))
        == 8193
    )
    assert (
        _lookup_moe_m_knee("unknown GPU", torch.float16, (16, 1024, 2048, 4, True))
        == 1025
    )
    ep_config = replace(_config(), num_local_experts=64)
    assert (
        _lookup_moe_m_knee("unknown GPU", torch.float16, _moe_shape(ep_config)) == 4097
    )

    experts = object.__new__(NvFp4ByCopyExperts)
    experts.m_knee = 4097
    assert not experts._use_bycopy(4096)
    assert experts._use_bycopy(4097)


def test_deepgemm_selector_uses_runtime_m_and_static_layer_shape(monkeypatch):
    experts = object.__new__(NvFp4ToFp8Experts)
    experts.moe_config = _config(p=704, k=2112)
    experts.experts = MagicMock()
    experts.fallback_experts = MagicMock()
    supported = MagicMock(return_value=True)
    monkeypatch.setattr(nvfp4_bycopy_moe, "_deepgemm_shape_supported", supported)

    selected = experts._select_experts_impl(
        torch.empty((6144, 2048), device="meta"),
        torch.empty((64, 1, 1), device="meta"),
        torch.empty((64, 48, 1), device="meta"),
    )

    assert selected is experts.experts
    supported.assert_called_once_with(6144, 768, 2048, MoEActivation.SILU)


@pytest.mark.parametrize(("n", "uses_deepgemm"), [(496, False), (512, True)])
def test_deepgemm_selector_n_boundary(monkeypatch, n, uses_deepgemm):
    experts = object.__new__(NvFp4ToFp8Experts)
    experts.moe_config = _config(p=704)
    experts.experts = MagicMock()
    experts.fallback_experts = MagicMock()
    monkeypatch.setattr(nvfp4_bycopy_moe, "is_deep_gemm_supported", lambda: True)
    monkeypatch.setattr(nvfp4_bycopy_moe, "_valid_deep_gemm_shape", lambda *_: True)

    selected = experts._select_experts_impl(
        torch.empty((8192, 2048), device="meta"),
        torch.empty((64, 1, 1), device="meta"),
        torch.empty((64, n // 16, 1), device="meta"),
    )

    assert (selected is experts.experts) is uses_deepgemm


def test_workspace_includes_padded_silu_and_one_overlaid_weight():
    experts = object.__new__(NvFp4ToFp8TritonExperts)
    experts.moe_config = _config()
    m, n, k, topk, e = 4096, 704, 2048, 8, 128

    gated, workspace2, output = experts.workspace_shapes(
        m, n, k, topk, e, e, None, MoEActivation.SILU
    )

    assert workspace2 == (m, topk, k)
    assert output == (m, k)
    n_fp8 = 768
    prefix, weight, scale, _ = experts._workspace_layout(
        m, n, k, topk, e, MoEActivation.SILU
    )
    assert prefix >= m * topk * (n_fp8 + n_fp8 // 128 * 4)
    assert weight >= e * k * n_fp8
    assert gated[0] * torch.bfloat16.itemsize == prefix + weight + scale


def test_disabled_path_does_not_bind_or_reserve_workspace():
    experts = object.__new__(NvFp4ByCopyExperts)
    experts.m_knee = None
    experts.bycopy_experts = MagicMock()
    experts.fallback_experts = MagicMock()
    experts.bycopy_experts.w13_fp8_scale_divisor_code = None
    experts.bycopy_experts.w2_fp8_scale_divisor_code = None

    experts.process_weights_after_loading(MagicMock())

    assert experts.bycopy_experts.w13_fp8_scale_divisor_code is None
    assert experts.bycopy_experts.w2_fp8_scale_divisor_code is None
    experts.bycopy_experts.workspace_shapes.assert_not_called()
    experts.fallback_experts.workspace_shapes.assert_not_called()


def test_enabled_path_binds_codes_and_reserves_all_backends(monkeypatch):
    experts = object.__new__(NvFp4ByCopyExperts)
    experts.m_knee = 1
    experts.moe_config = _config(e=2, p=128, k=128, topk=1)
    experts.bycopy_experts = MagicMock()
    experts.bycopy_experts.experts = MagicMock()
    experts.bycopy_experts.fallback_experts = MagicMock()
    experts.fallback_experts = MagicMock()
    implementations = (
        experts.bycopy_experts.experts,
        experts.bycopy_experts.fallback_experts,
        experts.fallback_experts,
    )
    implementations[0].workspace_shapes.return_value = ((1,), (1,), (1,))
    implementations[1].workspace_shapes.return_value = ((400,), (1,), (1,))
    implementations[2].workspace_shapes.return_value = ((1,), (300,), (1,))
    w13_codes = torch.ones((2, 2, 1), dtype=torch.uint8)
    w2_codes = torch.ones((2, 1, 1), dtype=torch.uint8)
    layer = SimpleNamespace(
        w13_weight=torch.empty((2, 1)),
        w2_weight=torch.empty((2, 1)),
        w13_fp8_scale_divisor_code=w13_codes,
        w2_fp8_scale_divisor_code=w2_codes,
        workspace=torch.empty(1, dtype=torch.int32),
    )
    reserve = MagicMock()
    monkeypatch.setattr(
        nvfp4_bycopy_moe, "marlin_moe_intermediate_size", lambda *args: 128
    )
    monkeypatch.setattr(nvfp4_bycopy_moe, "reserve_workspace_for_all_ubatches", reserve)

    experts.process_weights_after_loading(layer)

    for implementation in implementations[:2]:
        assert implementation.w13_fp8_scale_divisor_code is w13_codes
        assert implementation.w2_fp8_scale_divisor_code is w2_codes
    for implementation in implementations:
        implementation.workspace_shapes.assert_called_once()
    reserve.assert_called_once_with(1280)


def test_native_outer_op_receives_both_resident_branches(monkeypatch):
    e, m, n, k, topk = 2, 4, 512, 128, 1
    config = _config(e=e, p=n, k=k, topk=topk)
    w13_scale = torch.empty((e, k // 16, 2 * n), dtype=torch.bfloat16)
    w2_scale = torch.empty((e, n // 16, k), dtype=torch.bfloat16)
    w13_global = torch.ones(e)
    w2_global = torch.ones(e)
    quant_config = nvfp4_w4a16_moe_quant_config(
        w13_global, w2_global, w13_scale, w2_scale
    )
    experts = NvFp4ByCopyExperts(config, quant_config)
    experts.m_knee = 8
    experts.marlin_workspace = torch.empty(4, dtype=torch.int32)
    experts.w13_fp8_scale_divisor_code = torch.ones(
        (e, 2 * n // 128, k // 128), dtype=torch.uint8
    )
    experts.w2_fp8_scale_divisor_code = torch.ones(
        (e, k // 128, n // 128), dtype=torch.uint8
    )
    native = MagicMock()
    monkeypatch.setattr(torch.ops._C, "marlin_nvfp4_hybrid_moe", native, raising=False)
    monkeypatch.setattr(
        nvfp4_bycopy_moe, "_deepgemm_shape_supported", lambda *args: True
    )

    workspace13_shape, workspace2_shape, output_shape = experts.workspace_shapes(
        m, n, k, topk, e, e, None, MoEActivation.SILU
    )
    hidden = torch.empty((m, k), dtype=torch.bfloat16)
    w13 = torch.empty((e, k // 16, 4 * n), dtype=torch.int32)
    w2 = torch.empty((e, n // 16, 2 * k), dtype=torch.int32)
    topk_weights = torch.ones((m, topk), dtype=torch.float32)
    topk_ids = torch.zeros((m, topk), dtype=torch.int64)
    output = torch.empty(output_shape, dtype=torch.bfloat16)
    workspace13 = torch.empty(workspace13_shape, dtype=torch.bfloat16)

    experts.apply(
        output,
        hidden,
        w13,
        w2,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        e,
        None,
        None,
        None,
        workspace13,
        torch.empty(workspace2_shape, dtype=torch.bfloat16),
        None,
        False,
    )

    native.assert_called_once()
    args = native.call_args.args
    assert args[:3] == (hidden, w13, w2)
    assert args[3:9] == (
        w13_scale,
        w2_scale,
        w13_global,
        w2_global,
        experts.w13_fp8_scale_divisor_code,
        experts.w2_fp8_scale_divisor_code,
    )
    assert args[12] is experts.marlin_workspace
    assert args[13] is output
    assert args[14] is workspace13
    assert args[15:] == (e, experts.m_knee, False)
    assert workspace2_shape == (0,)


def test_support_accepts_tp_and_pure_ep_but_rejects_dp(monkeypatch):
    config = _config()
    monkeypatch.setattr(
        NvFp4ByCopyExperts,
        "_supports_current_device",
        staticmethod(lambda: True),
    )
    monkeypatch.setattr(
        nvfp4_bycopy_moe.current_platform,
        "get_device_name",
        lambda *args: "NVIDIA H100 80GB HBM3",
    )

    def supported(candidate):
        return NvFp4ByCopyExperts.is_supported_config(
            NvFp4ByCopyExperts,
            candidate,
            kNvfp4Static,
            None,
            mk.FusedMoEActivationFormat.Standard,
        )[0]

    assert supported(config)
    assert supported(replace(config, in_dtype=torch.float16))
    assert not supported(replace(config, has_bias=True))
    assert not supported(replace(config, is_lora_enabled=True))
    assert supported(_config(p=704))
    assert supported(replace(_config(p=704), activation=MoEActivation.GELU))
    assert supported(_config(p=96))
    assert supported(_config(k=2112))
    assert not supported(_config(p=100))
    assert not supported(_config(k=2100))
    tp = replace(config.moe_parallel_config, tp_size=2, tp_rank=0)
    assert supported(replace(config, moe_parallel_config=tp))
    ep = replace(
        config.moe_parallel_config,
        tp_size=1,
        tp_rank=0,
        ep_size=2,
        ep_rank=0,
        use_ep=True,
    )
    assert supported(replace(config, moe_parallel_config=ep))
    parallel = replace(config.moe_parallel_config, dp_size=2)
    assert not supported(replace(config, moe_parallel_config=parallel))
    eplb = replace(config.moe_parallel_config, enable_eplb=True)
    assert not supported(replace(config, moe_parallel_config=eplb))


def test_internal_oracle_backend_precedes_marlin(monkeypatch):
    monkeypatch.delenv("VLLM_DISABLED_KERNELS", raising=False)

    class Support:
        @staticmethod
        def is_supported_config(*args, **kwargs):
            return True, None

    class Reject:
        @staticmethod
        def is_supported_config(*args, **kwargs):
            return False, "test rejection"

    def classes(backend):
        if backend in (
            nvfp4_oracle.NvFp4MoeBackend.MARLIN_FP8_BYCOPY,
            nvfp4_oracle.NvFp4MoeBackend.MARLIN,
        ):
            return [Support]
        return [Reject]

    monkeypatch.setattr(nvfp4_oracle, "backend_to_kernel_cls", classes)
    backend, _ = nvfp4_oracle.select_nvfp4_moe_backend(_config(), kNvfp4Static, None)
    assert backend == nvfp4_oracle.NvFp4MoeBackend.MARLIN_FP8_BYCOPY
    monkeypatch.setenv("VLLM_DISABLED_KERNELS", "NvFp4ByCopyExperts")
    backend, _ = nvfp4_oracle.select_nvfp4_moe_backend(_config(), kNvfp4Static, None)
    assert backend == nvfp4_oracle.NvFp4MoeBackend.MARLIN
    assert (
        nvfp4_oracle.map_nvfp4_backend("marlin") == nvfp4_oracle.NvFp4MoeBackend.MARLIN
    )
    with pytest.raises(ValueError):
        nvfp4_oracle.map_nvfp4_backend("marlin_fp8_bycopy")


@pytest.mark.parametrize(
    ("dtype", "compute_type"),
    [
        (torch.bfloat16, nvfp4_bycopy_moe.tl.bfloat16),
        (torch.float16, nvfp4_bycopy_moe.tl.float16),
    ],
)
@pytest.mark.parametrize(
    ("activation", "n", "k"),
    [
        (MoEActivation.SILU, 192, 192),
        (MoEActivation.GELU, 192, 128),
    ],
)
def test_staged_apply_pads_tails_and_reuses_weight_scratch(
    monkeypatch, dtype, compute_type, activation, n, k
):
    e, m, topk = 2, 1, 1
    k_fp8 = (k + 127) // 128 * 128
    n_fp8 = (n + 127) // 128 * 128
    config = _config(e=e, p=n, k=k, topk=topk, dtype=dtype)
    scales1 = torch.empty((e, k // 16, 2 * n), dtype=dtype)
    scales2 = torch.empty((e, n // 16, k), dtype=dtype)
    globals1 = torch.ones(e)
    globals2 = torch.ones(e)
    quant_config = nvfp4_w4a16_moe_quant_config(globals1, globals2, scales1, scales2)
    experts = object.__new__(NvFp4ToFp8TritonExperts)
    experts.moe_config = config
    experts.quant_config = quant_config
    experts.w13_fp8_scale_divisor_code = torch.ones(
        (e, 3, k_fp8 // 128), dtype=torch.uint8
    )
    experts.w2_fp8_scale_divisor_code = torch.ones(
        (e, (k + 127) // 128, n_fp8 // 128), dtype=torch.uint8
    )
    experts.activation = MagicMock(side_effect=lambda _, output, __: output.zero_())
    experts.moe_sum = MagicMock()

    w1 = torch.empty((e, k // 16, 4 * n), dtype=torch.int32)
    w2 = torch.empty((e, n // 16, 2 * k), dtype=torch.int32)
    hidden = torch.randn((m, k), dtype=dtype)
    topk_ids = torch.zeros((m, topk), dtype=torch.int64)
    topk_weights = torch.ones((m, topk), dtype=torch.float32)
    output = torch.empty_like(hidden)
    ws13_shape, ws2_shape, _ = experts.workspace_shapes(
        m, n, k, topk, e, e, None, activation
    )
    workspace13 = torch.empty(ws13_shape, dtype=dtype)
    workspace2 = torch.ones(ws2_shape, dtype=dtype)

    assignment = (
        None,
        torch.zeros(1, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
    )
    events = []
    materialize = MagicMock(
        side_effect=lambda *args, **kwargs: events.append("materialize")
    )
    quantize = MagicMock(
        side_effect=lambda value, *_: (
            torch.empty_like(value, dtype=torch.float8_e4m3fn),
            torch.ones((value.size(0), value.size(1) // 128)),
        )
    )
    fused_quant = MagicMock(
        side_effect=lambda *args, **kwargs: events.append("fused_quant")
    )
    assignment_mock = MagicMock(return_value=assignment)

    def run_gemm(*args, **kwargs):
        if events.count("gemm"):
            assert torch.count_nonzero(args[2]) == 0
        events.append("gemm")

    gemm = MagicMock(side_effect=run_gemm)
    config_mock = MagicMock(return_value={"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64})
    monkeypatch.setattr(nvfp4_bycopy_moe.ops, "marlin_nvfp4_to_fp8", materialize)
    monkeypatch.setattr(nvfp4_bycopy_moe, "moe_kernel_quantize_input", quantize)
    monkeypatch.setattr(nvfp4_bycopy_moe, "_prepare_expert_assignment", assignment_mock)
    monkeypatch.setattr(nvfp4_bycopy_moe, "invoke_fused_moe_triton_kernel", gemm)
    monkeypatch.setattr(torch.ops._C, "silu_and_mul_per_block_quant", fused_quant)
    monkeypatch.setattr(
        nvfp4_bycopy_moe,
        "try_get_optimal_moe_config",
        config_mock,
    )
    monkeypatch.setattr(
        nvfp4_bycopy_moe.current_platform,
        "fp8_dtype",
        lambda: torch.float8_e4m3fn,
    )

    experts.apply(
        output,
        hidden,
        w1,
        w2,
        topk_weights,
        topk_ids,
        activation,
        e,
        torch.tensor([0, -1], dtype=torch.int32),
        None,
        None,
        workspace13,
        workspace2,
        None,
        False,
    )

    assert assignment_mock.call_count == 1
    assert materialize.call_count == 2
    assert gemm.call_count == 2
    first, second = materialize.call_args_list
    assert first.args[0].shape == (e, 2 * n, k_fp8)
    assert second.args[0].shape == (e, k, n_fp8)
    assert first.args[0].data_ptr() == second.args[0].data_ptr()
    assert first.args[1].shape == (e, 3, k_fp8 // 128)
    assert second.args[1].shape == (
        e,
        (k + 127) // 128,
        n_fp8 // 128,
    )
    assert first.args[1].data_ptr() == second.args[1].data_ptr()
    assert first.args[5] is experts.w13_fp8_scale_divisor_code
    assert second.args[5] is experts.w2_fp8_scale_divisor_code
    assert first.args[6] == second.args[6] == dtype
    assert config_mock.call_args.args[:2] == (
        (e, 2 * n, k_fp8),
        (e, k, n_fp8),
    )
    assert config_mock.call_args.kwargs["block_shape"] == [128, 128]
    assert assignment_mock.call_args.kwargs["block_shape"] == [128, 128]
    padded_hidden = quantize.call_args_list[0].args[0]
    assert padded_hidden.shape == (m, k_fp8)
    torch.testing.assert_close(padded_hidden[:, :k], hidden)
    assert torch.count_nonzero(padded_hidden[:, k:]) == 0
    if activation == MoEActivation.SILU:
        assert quantize.call_count == 1
        fused_args = fused_quant.call_args.args
        assert fused_args[0].shape == (m * topk, n_fp8)
        assert fused_args[1].shape == (m * topk, 2 * n)
        assert fused_args[2].shape == (m * topk, n_fp8 // 128)
        assert fused_args[3:] == (128, None, False)
        q_end = fused_args[0].data_ptr() + fused_args[0].numel()
        scale_end = fused_args[2].data_ptr() + fused_args[2].numel() * 4
        assert fused_args[0].data_ptr() == workspace13.data_ptr()
        assert fused_args[2].data_ptr() == q_end
        assert second.args[0].data_ptr() >= scale_end
        assert events == [
            "materialize",
            "gemm",
            "fused_quant",
            "materialize",
            "gemm",
        ]
        experts.activation.assert_not_called()
    else:
        assert quantize.call_count == 2
        padded_intermediate = quantize.call_args_list[1].args[0]
        assert padded_intermediate.shape == (m * topk, n_fp8)
        assert torch.count_nonzero(padded_intermediate[:, n:]) == 0
        assert events == ["materialize", "gemm", "materialize", "gemm"]
        experts.activation.assert_called_once()
        fused_quant.assert_not_called()
    assert gemm.call_args_list[0].args[4].shape == first.args[1].shape
    assert gemm.call_args_list[1].args[4].shape == second.args[1].shape
    for call in gemm.call_args_list:
        assert call.kwargs["block_shape"] == [128, 128]
        assert call.kwargs["compute_type"] == compute_type


@pytest.mark.skipif(
    not (
        current_platform.is_cuda()
        and current_platform.is_device_capability(90)
        and hasattr(torch.ops._C, "marlin_nvfp4_to_fp8")
    ),
    reason="The staged NVFP4-to-FP8 MoE path requires SM90.",
)
@pytest.mark.usefixtures("default_vllm_config")
def test_real_staged_apply_matches_marlin(monkeypatch):
    e, m, n, k, topk = 2, 8, 128, 128, 2
    dtype = torch.bfloat16
    config = _config(e=e, p=n, k=k, topk=topk, dtype=dtype)
    generator = torch.Generator(device="cuda").manual_seed(7)
    w13 = torch.randint(
        0,
        256,
        (e, 2 * n, k // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w2 = torch.randint(
        0,
        256,
        (e, k, n // 2),
        dtype=torch.uint8,
        device="cuda",
        generator=generator,
    )
    w13_scale = torch.ones(
        (e, 2 * n, k // 16), dtype=torch.float8_e4m3fn, device="cuda"
    )
    w2_scale = torch.ones((e, k, n // 16), dtype=torch.float8_e4m3fn, device="cuda")
    w13_global = torch.tensor([2**-4, 2**-3], device="cuda")
    w2_global = torch.tensor([2**-3, 2**-4], device="cuda")
    layer = SimpleNamespace(
        num_experts=e,
        hidden_size=k,
        intermediate_size_per_partition=n,
        params_dtype=dtype,
    )
    w13, w13_scale, w13_global, w2, w2_scale, w2_global = (
        prepare_nvfp4_moe_layer_for_marlin(
            layer,
            w13,
            w13_scale,
            w13_global,
            w2,
            w2_scale,
            w2_global,
            is_act_and_mul=True,
        )
    )
    quant_config = nvfp4_w4a16_moe_quant_config(
        w13_global, w2_global, w13_scale, w2_scale
    )

    levels = torch.tensor(
        [-0.5, -0.25, -0.125, 0.0, 0.125, 0.25, 0.5],
        dtype=dtype,
        device="cuda",
    )
    hidden = levels[torch.arange(m * k, device="cuda").remainder(levels.numel())].view(
        m, k
    )
    topk_ids = torch.tensor([[0, 1], [1, 0]], device="cuda").repeat(m // 2, 1)
    topk_weights = torch.tensor(
        [[0.75, 0.25], [0.25, 0.75]], dtype=torch.float32, device="cuda"
    ).repeat(m // 2, 1)

    reference_experts = MarlinExperts(config, quant_config)
    reference_ws13_shape, reference_ws2_shape, _ = reference_experts.workspace_shapes(
        m, n, k, topk, e, e, None, MoEActivation.SILU
    )
    reference = torch.empty_like(hidden)
    reference_experts.apply(
        reference,
        hidden,
        w13,
        w2,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        e,
        None,
        None,
        None,
        torch.empty(reference_ws13_shape, dtype=dtype, device="cuda"),
        torch.empty(reference_ws2_shape, dtype=dtype, device="cuda"),
        None,
        False,
    )

    staged_experts = NvFp4ToFp8TritonExperts(config, quant_config)
    staged_experts.w13_fp8_scale_divisor_code = layer.w13_fp8_scale_divisor_code
    staged_experts.w2_fp8_scale_divisor_code = layer.w2_fp8_scale_divisor_code
    staged_ws13_shape, staged_ws2_shape, _ = staged_experts.workspace_shapes(
        m, n, k, topk, e, e, None, MoEActivation.SILU
    )
    staged = torch.empty_like(hidden)
    scratch = []
    native_converter = nvfp4_bycopy_moe.ops.marlin_nvfp4_to_fp8

    def record_scratch(fp8_out, scale_out, *args, **kwargs):
        scratch.append(
            (fp8_out.data_ptr(), scale_out.data_ptr(), tuple(scale_out.shape))
        )
        return native_converter(fp8_out, scale_out, *args, **kwargs)

    monkeypatch.setattr(nvfp4_bycopy_moe.ops, "marlin_nvfp4_to_fp8", record_scratch)
    staged_experts.apply(
        staged,
        hidden,
        w13,
        w2,
        topk_weights,
        topk_ids,
        MoEActivation.SILU,
        e,
        None,
        None,
        None,
        torch.empty(staged_ws13_shape, dtype=dtype, device="cuda"),
        torch.empty(staged_ws2_shape, dtype=dtype, device="cuda"),
        None,
        False,
    )

    assert len(scratch) == 2
    assert scratch[0][:2] == scratch[1][:2]
    assert scratch[0][2] == (e, 2, 1)
    assert scratch[1][2] == (e, 1, 1)
    assert reference.abs().max() > 0.1
    torch.testing.assert_close(staged, reference, rtol=0.1, atol=4e-2)
