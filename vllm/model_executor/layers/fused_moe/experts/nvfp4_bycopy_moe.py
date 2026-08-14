# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Runtime NVFP4-to-FP8 materialization for Triton MoE experts."""

from math import prod

import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm import _custom_ops as ops
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.experts.fallback import FallbackExperts
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import MarlinExperts
from vllm.model_executor.layers.fused_moe.experts.triton_moe import TritonExperts
from vllm.model_executor.layers.fused_moe.fused_moe import (
    _prepare_expert_assignment,
    invoke_fused_moe_triton_kernel,
    try_get_optimal_moe_config,
)
from vllm.model_executor.layers.fused_moe.utils import (
    _resize_cache,
    moe_kernel_quantize_input,
)
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_moe_intermediate_size,
    marlin_pad_dim,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    _lookup_nvfp4_bycopy_m_knee,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kNvfp4Static,
)
from vllm.platforms import current_platform
from vllm.triton_utils import tl
from vllm.utils.math_utils import round_up
from vllm.v1.worker.workspace import reserve_workspace_for_all_ubatches

MoeShape = tuple[int, int, int, int, bool]
_FP8_BLOCK_SHAPE = [128, 128]


def _moe_shape(moe_config: FusedMoEConfig) -> MoeShape:
    return (
        moe_config.num_local_experts,
        moe_config.intermediate_size_per_partition,
        moe_config.hidden_dim,
        moe_config.experts_per_token,
        moe_config.activation.is_gated,
    )


def _lookup_moe_m_knee(
    device_name: str,
    dtype: torch.dtype,
    shape: MoeShape,
) -> int:
    return _lookup_nvfp4_bycopy_m_knee(device_name, dtype, shape)


def _marlin_problem_size(
    a1: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
) -> tuple[int, int, int, int, int]:
    assert w1.dim() == 3 and w2.dim() == 3
    e = w1.size(0)
    k = a1.size(-1)
    n = marlin_moe_intermediate_size(w1, w2)
    if a1.dim() == 2:
        assert topk_ids.size(0) == a1.size(0)
        m = a1.size(0)
    else:
        assert a1.dim() == 3 and a1.size(0) == e
        m = a1.size(1)
    assert topk_ids.dim() == 2
    return e, m, n, k, topk_ids.size(1)


class NvFp4ToFp8TritonExperts(TritonExperts):
    """Stage Marlin NVFP4 weights through the existing Triton FP8 MoE."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        super().__init__(moe_config, quant_config)
        self.w13_fp8_scale_divisor_code: torch.Tensor | None = None
        self.w2_fp8_scale_divisor_code: torch.Tensor | None = None

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        return _marlin_problem_size(a1, w1, w2, topk_ids)

    def _workspace_layout(
        self,
        m: int,
        n: int,
        k: int,
        topk: int,
        local_num_experts: int,
        activation: MoEActivation,
    ) -> tuple[int, int, int, int]:
        itemsize = self.moe_config.in_dtype.itemsize
        w13_n = (2 if activation.is_gated else 1) * n
        n_fp8 = round_up(n, _FP8_BLOCK_SHAPE[1])
        k_fp8 = round_up(k, _FP8_BLOCK_SHAPE[1])
        if activation == MoEActivation.SILU:
            activation_bytes = (
                m
                * topk
                * (n_fp8 + n_fp8 // _FP8_BLOCK_SHAPE[1] * torch.float32.itemsize)
            )
        else:
            activation_bytes = m * topk * n * itemsize
        prefix_bytes = round_up(max(activation_bytes, m * k_fp8 * itemsize), 256)
        weight_bytes = round_up(
            max(
                local_num_experts * w13_n * k_fp8,
                local_num_experts * k * n_fp8,
            ),
            256,
        )
        block_n, block_k = _FP8_BLOCK_SHAPE
        scale_bytes = round_up(
            local_num_experts
            * max(
                ((w13_n + block_n - 1) // block_n) * (k_fp8 // block_k),
                ((k + block_n - 1) // block_n) * (n_fp8 // block_k),
            )
            * torch.float32.itemsize,
            256,
        )
        return prefix_bytes, weight_bytes, scale_bytes, w13_n

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        del global_num_experts, expert_tokens_meta
        prefix, weight, scale, w13_n = self._workspace_layout(
            M, N, K, topk, local_num_experts, activation
        )
        itemsize = self.moe_config.in_dtype.itemsize
        return (
            ((prefix + weight + scale) // itemsize,),
            (M, topk, max(w13_n, K)),
            (M, K),
        )

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ) -> None:
        del a1q_scale, a2_scale, expert_tokens_meta
        if hidden_states.size(0) == 0:
            return
        assert hidden_states.dim() == 2 and hidden_states.is_contiguous()
        assert hidden_states.dtype == self.moe_config.in_dtype
        assert self.w1_scale is not None and self.w2_scale is not None
        assert self.g1_alphas is not None and self.g2_alphas is not None
        assert self.w13_fp8_scale_divisor_code is not None
        assert self.w2_fp8_scale_divisor_code is not None
        assert self.w1_bias is None and self.w2_bias is None

        e, m, n, k, topk = self.moe_problem_size(hidden_states, w1, w2, topk_ids)
        n_fp8 = round_up(n, _FP8_BLOCK_SHAPE[1])
        k_fp8 = round_up(k, _FP8_BLOCK_SHAPE[1])
        if global_num_experts == -1:
            global_num_experts = e
        prefix_bytes, weight_bytes, scale_bytes, w13_n = self._workspace_layout(
            m, n, k, topk, e, activation
        )
        fp8_dtype = current_platform.fp8_dtype()
        assert fp8_dtype == torch.float8_e4m3fn
        compute_type = (
            tl.bfloat16 if self.moe_config.in_dtype == torch.bfloat16 else tl.float16
        )
        config = try_get_optimal_moe_config(
            (e, w13_n, k_fp8),
            (e, k, n_fp8),
            topk,
            "fp8_w8a8",
            m,
            block_shape=_FP8_BLOCK_SHAPE,
        )
        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            _prepare_expert_assignment(
                topk_ids,
                config,
                m,
                topk,
                global_num_experts,
                expert_map,
                use_int8_w8a16=False,
                use_int4_w4a16=False,
                block_shape=_FP8_BLOCK_SHAPE,
            )
        )

        workspace_bytes = workspace13.view(torch.uint8)
        fp8_weight = workspace_bytes.narrow(0, prefix_bytes, weight_bytes).view(
            fp8_dtype
        )
        fp8_scale = workspace_bytes.narrow(
            0, prefix_bytes + weight_bytes, scale_bytes
        ).view(torch.float32)
        w13_fp8 = fp8_weight[: e * w13_n * k_fp8].view(e, w13_n, k_fp8)
        w2_fp8 = fp8_weight[: e * k * n_fp8].view(e, k, n_fp8)
        block_n, block_k = _FP8_BLOCK_SHAPE
        w13_scale_shape = (
            e,
            (w13_n + block_n - 1) // block_n,
            k_fp8 // block_k,
        )
        w2_scale_shape = (
            e,
            (k + block_n - 1) // block_n,
            n_fp8 // block_k,
        )
        w13_fp8_scale = fp8_scale[: prod(w13_scale_shape)].view(w13_scale_shape)
        w2_fp8_scale = fp8_scale[: prod(w2_scale_shape)].view(w2_scale_shape)
        intermediate1 = _resize_cache(workspace2, (m, topk, w13_n))
        intermediate3 = _resize_cache(workspace2, (m, topk, k))

        ops.marlin_nvfp4_to_fp8(
            w13_fp8,
            w13_fp8_scale,
            w1,
            self.w1_scale,
            self.g1_alphas,
            self.w13_fp8_scale_divisor_code,
            self.moe_config.in_dtype,
        )
        # ponytail: Fuse K64/non-SiLU padding if tail throughput matters.
        padded_hidden_states = marlin_pad_dim(hidden_states, k, k_fp8)
        qhidden_states, input_scale = moe_kernel_quantize_input(
            padded_hidden_states,
            None,
            fp8_dtype,
            False,
            _FP8_BLOCK_SHAPE,
        )
        invoke_fused_moe_triton_kernel(
            qhidden_states,
            w13_fp8,
            intermediate1,
            input_scale,
            w13_fp8_scale,
            None,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            False,
            topk,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=_FP8_BLOCK_SHAPE,
            B_bias=None,
        )
        if activation == MoEActivation.SILU:
            rows = m * topk
            q_bytes = rows * n_fp8
            scale_bytes = rows * (n_fp8 // block_k) * torch.float32.itemsize
            qintermediate2 = (
                workspace_bytes.narrow(0, 0, q_bytes).view(fp8_dtype).view(rows, n_fp8)
            )
            intermediate2_scale = (
                workspace_bytes.narrow(0, q_bytes, scale_bytes)
                .view(torch.float32)
                .view(rows, n_fp8 // block_k)
            )
            torch.ops._C.silu_and_mul_per_block_quant(
                qintermediate2,
                intermediate1.view(-1, w13_n),
                intermediate2_scale,
                block_k,
                None,
                False,
            )
        else:
            intermediate2 = _resize_cache(workspace13, (m * topk, n))
            self.activation(activation, intermediate2, intermediate1.view(-1, w13_n))
            padded_intermediate2 = marlin_pad_dim(intermediate2, n, n_fp8)
            qintermediate2, intermediate2_scale = moe_kernel_quantize_input(
                padded_intermediate2,
                None,
                fp8_dtype,
                False,
                _FP8_BLOCK_SHAPE,
            )

        ops.marlin_nvfp4_to_fp8(
            w2_fp8,
            w2_fp8_scale,
            w2,
            self.w2_scale,
            self.g2_alphas,
            self.w2_fp8_scale_divisor_code,
            self.moe_config.in_dtype,
        )
        invoke_fused_moe_triton_kernel(
            qintermediate2,
            w2_fp8,
            intermediate3,
            intermediate2_scale,
            w2_fp8_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            config,
            compute_type=compute_type,
            use_fp8_w8a8=True,
            use_int8_w8a8=False,
            use_int8_w8a16=False,
            use_int4_w4a16=False,
            per_channel_quant=False,
            block_shape=_FP8_BLOCK_SHAPE,
            B_bias=None,
        )
        self.moe_sum(intermediate3, output)


class NvFp4ByCopyExperts(FallbackExperts):
    """Select Marlin or staged FP8 experts from post-dispatch M."""

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
    ) -> None:
        experts = NvFp4ToFp8TritonExperts(moe_config, quant_config)
        fallback = MarlinExperts(moe_config, quant_config)
        super().__init__(experts=experts, fallback_experts=fallback)
        self.bycopy_experts = experts
        self.m_knee = _lookup_moe_m_knee(
            current_platform.get_device_name(),
            moe_config.in_dtype,
            _moe_shape(moe_config),
        )

    @staticmethod
    def get_clses() -> tuple[
        type[mk.FusedMoEExpertsModular],
        type[mk.FusedMoEExpertsModular],
    ]:
        return NvFp4ToFp8TritonExperts, MarlinExperts

    @staticmethod
    def _supports_current_device() -> bool:
        return (
            current_platform.is_cuda()
            and current_platform.is_device_capability(90)
            and hasattr(torch.ops, "_C")
            and hasattr(torch.ops._C, "marlin_nvfp4_to_fp8")
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) == (kNvfp4Static, None)

    @classmethod
    def _supports_parallel_config(
        cls,
        moe_parallel_config: FusedMoEParallelConfig,
    ) -> bool:
        return (
            moe_parallel_config.tp_size == 1
            and moe_parallel_config.pcp_size == 1
            and moe_parallel_config.dp_size == 1
            and moe_parallel_config.ep_size == 1
            and moe_parallel_config.sp_size == 1
            and not moe_parallel_config.use_ep
            and not moe_parallel_config.enable_eplb
            and super()._supports_parallel_config(moe_parallel_config)
        )

    @staticmethod
    def is_supported_config(
        cls: type[mk.FusedMoEExperts],
        moe_config: FusedMoEConfig,
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
        activation_format: mk.FusedMoEActivationFormat,
    ) -> tuple[bool, str | None]:
        supported, reason = mk.FusedMoEExperts.is_supported_config(
            cls,
            moe_config,
            weight_key,
            activation_key,
            activation_format,
        )
        if not supported:
            return supported, reason
        if moe_config.has_bias:
            return False, "kernel does not support bias"
        if moe_config.hidden_dim % 64:
            return False, "kernel requires K divisible by 64"
        if moe_config.intermediate_size_per_partition % 16:
            return False, "kernel requires N divisible by 16"
        if (moe_config.num_experts + 31) // 32 * 32 >= 1024:
            return False, "kernel requires fewer than 1024 padded experts"
        return True, None

    def moe_problem_size(
        self,
        a1: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> tuple[int, int, int, int, int]:
        return self.fallback_experts.moe_problem_size(a1, w1, w2, topk_ids)

    def _use_bycopy(self, m: int) -> bool:
        return type(m) is int and self.m_knee is not None and m >= self.m_knee

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        experts = self.experts if self._use_bycopy(M) else self.fallback_experts
        return experts.workspace_shapes(
            M,
            N,
            K,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            activation,
        )

    def _select_experts_impl(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
    ) -> mk.FusedMoEExpertsModular:
        del w1, w2
        return (
            self.experts
            if self._use_bycopy(hidden_states.shape[0])
            else self.fallback_experts
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if self.m_knee is None:
            return
        self.bycopy_experts.w13_fp8_scale_divisor_code = (
            layer.w13_fp8_scale_divisor_code
        )
        self.bycopy_experts.w2_fp8_scale_divisor_code = layer.w2_fp8_scale_divisor_code

        m = self.moe_config.max_num_tokens
        n = marlin_moe_intermediate_size(layer.w13_weight, layer.w2_weight)
        k = self.moe_config.hidden_dim
        topk = self.moe_config.experts_per_token
        global_experts = self.moe_config.num_logical_experts
        local_experts = layer.w13_weight.size(0)
        dtype = self.moe_config.in_dtype

        def required_bytes(experts: mk.FusedMoEExpertsModular) -> int:
            workspace13, workspace2, output = experts.workspace_shapes(
                m,
                n,
                k,
                topk,
                global_experts,
                local_experts,
                None,
                self.moe_config.activation,
            )
            common = round_up(
                max(prod(workspace13), prod(output)) * dtype.itemsize, 256
            )
            second = round_up(prod(workspace2) * dtype.itemsize, 256)
            return common + second

        reserve_workspace_for_all_ubatches(
            max(
                required_bytes(self.bycopy_experts),
                required_bytes(self.fallback_experts),
            )
        )
