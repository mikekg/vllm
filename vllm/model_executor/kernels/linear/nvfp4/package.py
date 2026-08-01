# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

from .base import NvFp4LinearKernel, NvFp4LinearLayerConfig


class NvFp4PackageLinearKernel(NvFp4LinearKernel):
    """NVFP4 W4A16 GEMM provided by the optional nvfp4 package."""

    @classmethod
    def is_supported(
        cls, compute_capability: int | None = None
    ) -> tuple[bool, str | None]:
        if not current_platform.is_cuda():
            return False, "nvfp4 package only supports CUDA"
        if compute_capability is not None:
            if compute_capability < 80:
                return False, "nvfp4 package requires SM80+"
        elif not current_platform.has_device_capability(80):
            return False, "nvfp4 package requires SM80+"

        try:
            from nvfp4 import nvfp4_weight_prep  # noqa: F401
        except Exception as exc:  # noqa: BLE001
            return False, f"nvfp4 package is unavailable: {exc}"
        return True, None

    @classmethod
    def can_implement(cls, config: NvFp4LinearLayerConfig) -> tuple[bool, str | None]:
        return True, None

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        from nvfp4 import nvfp4_weight_prep

        weight, weight_scale, global_scale, layout, logical_shape = nvfp4_weight_prep(
            (
                layer.weight.data,
                layer.weight_scale.data,
                layer.weight_global_scale.data.reciprocal(),
            ),
            activation_dtype_hint=layer.params_dtype,
        )
        layer.weight = torch.nn.Parameter(weight, requires_grad=False)
        layer.weight_scale = torch.nn.Parameter(weight_scale, requires_grad=False)
        layer.weight_global_scale = torch.nn.Parameter(
            global_scale, requires_grad=False
        )
        layer.nvfp4_layout = layout
        layer.nvfp4_logical_shape = logical_shape

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logical_n, logical_k = layer.nvfp4_logical_shape
        output_shape = (*x.shape[:-1], logical_n)
        x = x.reshape(-1, logical_k)
        padded_k = layer.weight.shape[1] * 2
        if logical_k != padded_k:
            x = F.pad(x, (0, padded_k - logical_k))

        output = torch.ops.nvfp4.nvfp4_linear_w4a16_forward(
            x,
            layer.weight,
            layer.weight_global_scale,
            layer.weight_scale,
            layer.nvfp4_layout,
            None,
            0.0,
            False,
        )[:, :logical_n]
        if bias is not None:
            output = output + bias
        return output.reshape(output_shape)
