# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.quantization.utils.marlin_utils_fp4 import (
    nvfp4_marlin_process_global_scale,
    prepare_fp4_layer_for_marlin,
)
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
        from nvfp4 import entry102_scales_are_lossless, nvfp4_weight_prep

        raw_weight = layer.weight.data
        raw_scales = layer.weight_scale.data
        raw_global = layer.weight_global_scale.data
        blob_bytes = torch.ops.nvfp4.sm100_entry102_workspace_size(
            raw_weight, 8, layer.params_dtype
        )

        sidecar = None
        sidecar_valid = False
        if blob_bytes:
            sidecar = SimpleNamespace(
                weight=raw_weight,
                weight_scale=raw_scales,
                weight_global_scale=raw_global,
                output_size_per_partition=layer.output_size_per_partition,
                input_size_per_partition=layer.input_size_per_partition,
                params_dtype=layer.params_dtype,
            )
            prepare_fp4_layer_for_marlin(sidecar)
            unadjusted_alpha = nvfp4_marlin_process_global_scale(
                raw_global.to(torch.float32), layer.params_dtype
            )
            scale_factor = float(
                (unadjusted_alpha / sidecar.weight_global_scale).item()
            )
            sidecar_valid = entry102_scales_are_lossless(
                raw_scales, scale_factor
            )

        weight, weight_scale, global_scale, layout, logical_shape = nvfp4_weight_prep(
            (
                raw_weight,
                raw_scales,
                raw_global.reciprocal(),
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

        workspace = None
        compact_scales = None
        compact_alpha = None
        if sidecar_valid and sidecar is not None:
            packed_b = sidecar.weight.detach().view(torch.uint8).reshape(-1)
            n = raw_weight.shape[0]
            k = raw_weight.shape[1] * 2
            sidecar_valid = (
                sidecar.weight.dtype == torch.int32
                and sidecar.weight.is_contiguous()
                and tuple(sidecar.weight.shape) == (k // 16, 2 * n)
                and packed_b.numel() == raw_weight.numel()
                and sidecar.weight_scale.dtype == torch.float8_e4m3fn
                and sidecar.weight_scale.is_contiguous()
                and tuple(sidecar.weight_scale.shape) == (k // 16, n)
                and sidecar.weight_global_scale.dtype == torch.float32
                and sidecar.weight_global_scale.numel() == 1
                and blob_bytes == 136_315_392
            )

        if sidecar_valid:
            weight_bytes = packed_b.numel()
            counter_bytes = 512
            workspace = torch.empty(
                blob_bytes, dtype=torch.uint8, device=raw_weight.device
            )
            workspace[:weight_bytes].copy_(packed_b)
            workspace[weight_bytes : weight_bytes + counter_bytes].zero_()
            compact_scales = sidecar.weight_scale.detach()
            compact_alpha = sidecar.weight_global_scale.detach().reshape(())

        layer.register_buffer(
            "_nvfp4_entry102_workspace", workspace, persistent=False
        )
        layer.register_buffer(
            "_nvfp4_entry102_scales", compact_scales, persistent=False
        )
        layer.register_buffer(
            "_nvfp4_entry102_alpha", compact_alpha, persistent=False
        )
        layer._nvfp4_entry102_scales_valid = workspace is not None

    def apply_weights(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logical_n, logical_k = layer.nvfp4_logical_shape
        output_shape = (*x.shape[:-1], logical_n)
        padded_k = layer.weight.shape[1] * 2
        aligned_2d = (
            x.ndim == 2
            and x.shape[1] == logical_k == padded_k
            and logical_n == layer.weight.shape[0]
        )
        if not aligned_2d:
            x = x.reshape(-1, logical_k)
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
            None,
            layer._nvfp4_entry102_workspace,
            layer._nvfp4_entry102_scales,
            layer._nvfp4_entry102_alpha,
            layer._nvfp4_entry102_scales_valid,
        )
        if not aligned_2d:
            output = output[:, :logical_n]
        if bias is not None:
            output = output + bias
        return output if aligned_2d else output.reshape(output_shape)
