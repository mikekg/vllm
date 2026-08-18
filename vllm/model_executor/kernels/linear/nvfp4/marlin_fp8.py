# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm import _custom_ops as ops
from vllm.config import get_current_vllm_config_or_none
from vllm.model_executor.layers.quantization.utils.marlin_utils import (
    USE_FP32_REDUCE_DEFAULT,
    marlin_pad_dim,
    marlin_permute_bias,
    marlin_repacked_nk,
    should_use_atomic_add_reduce,
)
from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig
from .marlin import MarlinNvFp4LinearKernel


def _is_nvfp4_bycopy_layer(layer: torch.nn.Module) -> bool:
    from vllm.model_executor.layers.vocab_parallel_embedding import (
        VocabParallelEmbedding,
    )

    return not isinstance(layer, VocabParallelEmbedding)


class MarlinNvFp4ToFp8LinearKernel(NvFp4LinearKernel):
    """Select Marlin or materialization followed by the existing FP8 GEMM."""

    def __init__(self, config: NvFp4LinearLayerConfig) -> None:
        super().__init__(config)
        self.marlin = MarlinNvFp4LinearKernel(config)
        self.m_knee: int | None = None
        self.logical_n = 0
        self.logical_k = 0
        self.padded_n = 0
        self.resident_k = 0
        self.use_atomic_add = False
        self.use_fp32_reduce = USE_FP32_REDUCE_DEFAULT

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
        if not hasattr(torch.ops._C, "marlin_nvfp4_hybrid_linear"):
            return False, "Marlin NVFP4 hybrid linear operator is unavailable"
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
        aligned = (
            self.padded_n == self.logical_n
            and self.resident_k == self.logical_k
            and self.padded_n % 128 == 0
            and self.resident_k % 128 == 0
        )
        self.m_knee = 1 if _is_nvfp4_bycopy_layer(layer) and aligned else None

        vllm_config = get_current_vllm_config_or_none()
        if self.m_knee is not None:
            self.use_atomic_add = should_use_atomic_add_reduce(
                m=1,
                n=self.padded_n,
                k=self.resident_k,
                device=layer.weight.device,
                dtype=layer.params_dtype,
            )
            if vllm_config is not None:
                self.m_knee = max(
                    self.m_knee,
                    (vllm_config.compilation_config.max_cudagraph_capture_size or 0)
                    + 1,
                )

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
        if self.m_knee is None or bias is not None:
            return self.marlin.apply_weights(layer, x, self._marlin_bias(layer, bias))

        x_2d = x.reshape(-1, self.logical_k)
        output = ops.marlin_nvfp4_hybrid_linear(
            x_2d,
            layer.weight,
            layer.weight_scale,
            layer.weight_global_scale,
            layer.weight_fp8_scale_divisor_code,
            layer.workspace,
            self.m_knee,
            self.use_atomic_add,
            self.use_fp32_reduce,
        )
        return output.reshape(*x.shape[:-1], self.logical_n)
