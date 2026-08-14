// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "../../torch_utils.h"

#include "../../dispatch_utils.h"
#include "quant_conversions.cuh"

#include <limits>

namespace vllm {

// Logic: one thread block per (token, group) pair

template <typename scalar_t, typename scalar_out_t, bool is_scale_transposed,
          int32_t group_size>
__global__ void silu_and_mul_per_block_quant_kernel(
    scalar_out_t* __restrict__ out,  // Output: [num_tokens, hidden_size] in
                                     // FP8/INT8
    float* __restrict__ scales,      // Output: [num_tokens, hidden_size /
                                 // group_size] or [hidden_size / group_size,
                                 // num_tokens]
    scalar_t const* __restrict__ input,  // Input: [num_tokens, hidden_size * 2]
    float const* scale_ub,               // Optional scale upper bound
    int32_t const logical_hidden_size, int32_t const output_hidden_size) {
  static_assert((group_size & (group_size - 1)) == 0,
                "group_size must be a power of 2 for correct reduction");

  // Grid: (num_tokens, num_groups)
  int64_t const token_idx = blockIdx.x;
  int const group_idx = blockIdx.y;
  int const tid = threadIdx.x;  // tid in [0, group_size)
  int const num_tokens = gridDim.x;

  // Input layout: [gate || up] concatenated along last dimension
  int const input_stride = logical_hidden_size * 2;
  int const group_start = group_idx * group_size;
  int const element_idx = group_start + tid;

  // Pointers to this token's data
  scalar_t const* token_input_gate =
      input + token_idx * input_stride + group_start;
  scalar_t const* token_input_up = token_input_gate + logical_hidden_size;
  scalar_out_t* token_output =
      out + token_idx * output_hidden_size + group_start;

  // Scale pointer for this group
  int const num_groups = gridDim.y;
  float* group_scale_ptr = is_scale_transposed
                               ? scales + group_idx * num_tokens + token_idx
                               : scales + token_idx * num_groups + group_idx;

  // Shared memory for reduction (compile-time sized)
  __shared__ float shared_max[group_size];

  // Step 1: Each thread loads one element, computes SiLU, stores in register
  float gate = 0.0f;
  float up = 0.0f;
  if (element_idx < logical_hidden_size) {
    gate = static_cast<float>(token_input_gate[tid]);
    up = static_cast<float>(token_input_up[tid]);
  }

  // Compute SiLU(gate) * up
  float sigmoid_gate = 1.0f / (1.0f + expf(-gate));
  float silu_gate = gate * sigmoid_gate;
  float result = silu_gate * up;  // Keep in register

  // Step 2: Reduce to find group max
  shared_max[tid] = fabsf(result);
  __syncthreads();

// Power-of-2 reduction (group_size guaranteed to be power of 2)
#pragma unroll
  for (int stride = group_size / 2; stride > 0; stride >>= 1) {
    if (tid < stride) {
      shared_max[tid] = fmaxf(shared_max[tid], shared_max[tid + stride]);
    }
    __syncthreads();
  }

  // Step 3: Compute scale (thread 0), broadcast via shared memory
  if (tid == 0) {
    float group_max = shared_max[0];

    float const quant_range = quant_type_max_v<scalar_out_t>;
    float group_scale = group_max / quant_range;

    // Apply scale upper bound if provided
    if (scale_ub != nullptr) {
      group_scale = fminf(group_scale, *scale_ub);
    }

    // Use minimum safe scaling factor
    group_scale = fmaxf(group_scale, min_scaling_factor<scalar_out_t>::val());

    // Store scale to global memory
    *group_scale_ptr = group_scale;

    // Reuse shared_max[0] to broadcast scale
    shared_max[0] = group_scale;
  }
  __syncthreads();

  float group_scale = shared_max[0];

  // Step 4: Quantize and write output
  token_output[tid] =
      vllm::ScaledQuant<scalar_out_t, false>::quant_fn(result, group_scale);
}

}  // namespace vllm

void silu_and_mul_per_block_quant(torch::stable::Tensor& out,
                                  torch::stable::Tensor const& input,
                                  torch::stable::Tensor& scales,
                                  int64_t group_size,
                                  std::optional<torch::stable::Tensor> scale_ub,
                                  bool is_scale_transposed) {
  static torch::headeronly::ScalarType kFp8Type =
      is_fp8_ocp() ? torch::headeronly::ScalarType::Float8_e4m3fn
                   : torch::headeronly::ScalarType::Float8_e4m3fnuz;

  STD_TORCH_CHECK(out.scalar_type() == kFp8Type ||
                  out.scalar_type() == torch::headeronly::ScalarType::Char);
  STD_TORCH_CHECK(out.is_contiguous() && input.is_contiguous());
  STD_TORCH_CHECK(
      input.scalar_type() == torch::headeronly::ScalarType::Half ||
          input.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "Input must be FP16 or BF16");
  STD_TORCH_CHECK(scales.scalar_type() == torch::headeronly::ScalarType::Float);
  STD_TORCH_CHECK(group_size == 128 || group_size == 64,
                  "Unsupported group size: ", group_size);

  if (scale_ub.has_value()) {
    STD_TORCH_CHECK(out.scalar_type() == kFp8Type);
  }

  STD_TORCH_CHECK(out.dim() == 2 && input.dim() == 2 && scales.dim() == 2,
                  "out, input, and scales must be rank 2");
  STD_TORCH_CHECK(input.size(-1) % 2 == 0, "input last dim must be even");
  int64_t logical_hidden_size_64 = input.size(-1) / 2;
  int64_t output_hidden_size_64 = out.size(-1);
  int64_t expected_output_size =
      (logical_hidden_size_64 + group_size - 1) / group_size * group_size;
  STD_TORCH_CHECK(
      logical_hidden_size_64 > 0 &&
          logical_hidden_size_64 <= std::numeric_limits<int32_t>::max() &&
          output_hidden_size_64 <= std::numeric_limits<int32_t>::max(),
      "hidden dimensions must fit int32");
  STD_TORCH_CHECK(output_hidden_size_64 == expected_output_size,
                  "output hidden_size must round input hidden_size up to "
                  "group_size");
  STD_TORCH_CHECK(out.size(0) == input.size(0),
                  "out and input token counts must match");
  int32_t logical_hidden_size = logical_hidden_size_64;
  int32_t output_hidden_size = output_hidden_size_64;
  auto num_tokens = input.size(0);
  int32_t num_groups = output_hidden_size / group_size;
  if (is_scale_transposed) {
    STD_TORCH_CHECK(scales.numel() == num_tokens * num_groups,
                    "transposed scales must contain num_tokens * num_groups "
                    "values");
  } else {
    STD_TORCH_CHECK(
        scales.size(0) == num_tokens && scales.size(1) == num_groups,
        "scales must have shape [num_tokens, num_groups]");
  }
  STD_TORCH_CHECK(out.get_device_index() == input.get_device_index() &&
                      scales.get_device_index() == input.get_device_index(),
                  "out, input, and scales must be on the same device");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(input.get_device_index());

  dim3 grid(num_tokens, num_groups);
  dim3 block(group_size);

  VLLM_STABLE_DISPATCH_FLOATING_TYPES(
      input.scalar_type(), "silu_and_mul_per_block_quant", [&] {
        using scalar_in_t = scalar_t;

        VLLM_STABLE_DISPATCH_QUANT_TYPES(
            out.scalar_type(), "silu_and_mul_per_block_quant", [&] {
              using scalar_out_t = scalar_t;

              VLLM_STABLE_DISPATCH_GROUP_SIZE(group_size, gs, [&] {
                VLLM_STABLE_DISPATCH_BOOL(
                    is_scale_transposed, transpose_scale, [&] {
                      vllm::silu_and_mul_per_block_quant_kernel<
                          scalar_in_t, scalar_out_t, transpose_scale, gs>
                          <<<grid, block, 0, stream>>>(
                              out.mutable_data_ptr<scalar_out_t>(),
                              scales.mutable_data_ptr<float>(),
                              input.const_data_ptr<scalar_in_t>(),
                              scale_ub.has_value()
                                  ? scale_ub->const_data_ptr<float>()
                                  : nullptr,
                              logical_hidden_size, output_hidden_size);
                    });
              });
            });
      });
}
