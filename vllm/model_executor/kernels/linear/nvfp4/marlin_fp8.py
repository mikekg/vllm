# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    marlin_pad_dim,
    marlin_permute_bias,
    marlin_repacked_nk,
    marlin_unpad_output,
)
from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
    _lookup_nvfp4_bycopy_m_knee,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.utils.torch_utils import direct_register_custom_op
from vllm.v1.worker.workspace import (
    current_workspace_manager,
    reserve_workspace_for_all_ubatches,
)

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig
from .marlin import MarlinNvFp4LinearKernel


def _lookup_dense_m_knee(
    device_name: str,
    dtype: torch.dtype,
    shape: tuple[int, int],
) -> int:
    return _lookup_nvfp4_bycopy_m_knee(device_name, dtype, shape)


def _is_nvfp4_bycopy_layer(layer: torch.nn.Module) -> bool:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    return not isinstance(layer, VocabParallelEmbedding)


def _marlin_nvfp4_to_fp8_block_scaled_mm(
    fp8_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    processed_block_scales: torch.Tensor,
    processed_global_scale: torch.Tensor,
    tile_scale_divisor_codes: torch.Tensor,
    resident_dtype: torch.dtype,
    x: torch.Tensor,
    activation_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    ops.marlin_nvfp4_to_fp8(
        fp8_weight,
        weight_scale,
        packed_weight,
        processed_block_scales,
        processed_global_scale,
        tile_scale_divisor_codes,
        resident_dtype,
    )
    return torch.ops.vllm.w8a8_triton_block_scaled_mm_func(
        x,
        fp8_weight,
        activation_scale,
        weight_scale,
        [128, 128],
        output_dtype,
    )


def _marlin_nvfp4_to_fp8_block_scaled_mm_fake(
    fp8_weight: torch.Tensor,
    weight_scale: torch.Tensor,
    packed_weight: torch.Tensor,
    processed_block_scales: torch.Tensor,
    processed_global_scale: torch.Tensor,
    tile_scale_divisor_codes: torch.Tensor,
    resident_dtype: torch.dtype,
    x: torch.Tensor,
    activation_scale: torch.Tensor,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(
        (x.size(0), fp8_weight.size(0)), dtype=output_dtype, device=x.device
    )


direct_register_custom_op(
    "marlin_nvfp4_to_fp8_block_scaled_mm",
    _marlin_nvfp4_to_fp8_block_scaled_mm,
    mutates_args=["fp8_weight", "weight_scale"],
    fake_impl=_marlin_nvfp4_to_fp8_block_scaled_mm_fake,
)


class MarlinNvFp4ToFp8LinearKernel(NvFp4LinearKernel):
    """Select Marlin or materialization followed by the existing FP8 GEMM."""

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.marlin = MarlinNvFp4LinearKernel(config)
        self.quant_fp8 = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            column_major_scales=True,
            use_ue8m0=False,
        )
        self.m_knee: int | None = None
        self.logical_n = 0
        self.logical_k = 0
        self.padded_n = 0
        self.resident_k = 0
        self.padded_k = 0

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if compute_capability is None:
            capability = current_platform.get_device_capability()
            compute_capability = (
                None if capability is None else capability[0] * 10 + capability[1]
            )
        if compute_capability != 90:
            return False, "Marlin NVFP4-to-FP8 requires Hopper"
        if not (
            hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "marlin_nvfp4_to_fp8")
        ):
            return False, "Marlin NVFP4-to-FP8 operator is unavailable"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        logical_bias = getattr(layer, "bias", None)
        self.marlin.process_weights_after_loading(layer)
        if logical_bias is not None:
            layer._marlin_nvfp4_bias = layer.bias.detach()
            layer.bias = logical_bias
        self.logical_n = layer.output_size_per_partition
        self.logical_k = layer.input_size_per_partition
        self.padded_n, self.resident_k = marlin_repacked_nk(layer.weight, 4)
        self.padded_k = round_up(self.resident_k, 128)
        self.m_knee = None
        if _is_nvfp4_bycopy_layer(layer):
            self.m_knee = _lookup_dense_m_knee(
                current_platform.get_device_name(layer.weight.device.index or 0),
                layer.params_dtype,
                (self.logical_n, self.logical_k),
            )
        if self.m_knee is not None:
            weight_bytes = round_up(self.padded_n * self.padded_k, 256)
            scale_bytes = round_up(
                round_up(self.padded_n, 128) // 128 * (self.padded_k // 128) * 4,
                256,
            )
            reserve_workspace_for_all_ubatches(weight_bytes + scale_bytes)

    def requires_single_stream_workspace(self, m: int) -> bool:
        return self.m_knee is not None and m >= self.m_knee

    def _marlin_bias(
        self, layer: torch.nn.Module, bias: torch.Tensor | None
    ) -> torch.Tensor | None:
        if bias is None:
            return None
        if bias is getattr(layer, "bias", None):
            cached_bias = getattr(layer, "_marlin_nvfp4_bias", None)
            if cached_bias is not None:
                return cached_bias
        return marlin_permute_bias(marlin_pad_dim(bias, self.logical_n, self.padded_n))

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        m = x.numel() // x.shape[-1]
        if type(m) is not int or not self.requires_single_stream_workspace(m):
            return self.marlin.apply_weights(layer, x, self._marlin_bias(layer, bias))

        fp8_weight, weight_scale = current_workspace_manager().get_simultaneous(
            ((self.padded_n, self.padded_k), torch.float8_e4m3fn),
            (
                (round_up(self.padded_n, 128) // 128, self.padded_k // 128),
                torch.float32,
            ),
        )
        x_2d = marlin_pad_dim(x.reshape(-1, x.shape[-1]), self.logical_k, self.padded_k)
        x_2d, activation_scale = self.quant_fp8(x_2d)
        output = torch.ops.vllm.marlin_nvfp4_to_fp8_block_scaled_mm(
            fp8_weight,
            weight_scale,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.weight_fp8_scale_divisor_code,
            layer.params_dtype,
            x_2d,
            activation_scale,
            x.dtype,
        )
        output = marlin_unpad_output(output, self.logical_n, self.padded_n)
        output = output.reshape(*x.shape[:-1], self.logical_n)
        if bias is not None:
            output = output + bias
        return output
