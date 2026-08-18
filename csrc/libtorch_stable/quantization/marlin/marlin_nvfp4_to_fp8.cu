#include "dequant.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "core/scalar_type.hpp"
#include "libtorch_stable/ops.h"
#include "libtorch_stable/torch_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>

torch::stable::Tensor marlin_gemm(
    torch::stable::Tensor& a, std::optional<torch::stable::Tensor> c_or_none,
    torch::stable::Tensor& b_q_weight,
    std::optional<torch::stable::Tensor> const& b_bias_or_none,
    torch::stable::Tensor& b_scales,
    std::optional<torch::stable::Tensor> const& a_scales_or_none,
    std::optional<torch::stable::Tensor> const& global_scale_or_none,
    std::optional<torch::stable::Tensor> const& b_zeros_or_none,
    std::optional<torch::stable::Tensor> const& g_idx_or_none,
    std::optional<torch::stable::Tensor> const& perm_or_none,
    torch::stable::Tensor& workspace, vllm::ScalarTypeId const& b_type_id,
    int64_t size_m, int64_t size_n, int64_t size_k, bool is_k_full,
    bool use_atomic_add, bool use_fp32_reduce, bool is_zp_float);

namespace {

using ScalarType = torch::headeronly::ScalarType;
using Tensor = torch::stable::Tensor;

__device__ __forceinline__ int processed_scale_row_index(int n) {
  const int lane = n & 63;
  const int transposed = 8 * (lane & 7) + (lane >> 3);
  return (transposed & ~3) | ((transposed & 1) << 1) | ((transposed & 2) >> 1);
}

__device__ __forceinline__ int64_t processed_scale_index(int n, int group,
                                                         int n_padded) {
  return static_cast<int64_t>(group) * n_padded + 64 * (n >> 6) +
         processed_scale_row_index(n);
}

__device__ __forceinline__ uint16_t convert_pair(half2 values, half scale) {
  const half2 scaled = __hmul2(values, __half2half2(scale));
  return __nv_cvt_halfraw2_to_fp8x2(static_cast<__half2_raw>(scaled),
                                    __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ uint32_t pack_pairs(uint16_t low, uint16_t high) {
  return static_cast<uint32_t>(low) | (static_cast<uint32_t>(high) << 16);
}

__device__ __forceinline__ uint32_t convert_four(half2 low, half2 high,
                                                 half scale) {
  const half2 scale2 = __half2half2(scale);
  const half2 scaled_low = __hmul2(low, scale2);
  const half2 scaled_high = __hmul2(high, scale2);
  uint32_t result;
  asm volatile(
      "{\n"
      ".reg .b16 low_bytes;\n"
      ".reg .b16 high_bytes;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 low_bytes, %1;\n"
      "cvt.rn.satfinite.e4m3x2.f16x2 high_bytes, %2;\n"
      "mov.b32 %0, {low_bytes, high_bytes};\n"
      "}"
      : "=r"(result)
      : "r"(*reinterpret_cast<const uint32_t*>(&scaled_low)),
        "r"(*reinterpret_cast<const uint32_t*>(&scaled_high)));
  return result;
}

__device__ __forceinline__ uint32_t component(uint4 value, int index) {
  return index == 0   ? value.x
         : index == 1 ? value.y
         : index == 2 ? value.z
                      : value.w;
}

__device__ __forceinline__ void copy_async(uint32_t* destination,
                                           const uint32_t* source) {
  const uint32_t shared =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
               :
               : "r"(shared), "l"(source));
}

__device__ __forceinline__ int packed_tile_index(int n_subtile, int k_group,
                                                 int column, int pair,
                                                 int marlin_warp) {
  const int bank = 4 * ((column ^ k_group) & 7) + pair;
  const int high = (4 * n_subtile + marlin_warp) * 8 + k_group;
  return 32 * high + bank;
}

template <bool MultipleExperts, bool SyncPaired = false>
__global__ __launch_bounds__(256) void marlin_nvfp4_to_fp8_tiled_kernel(
    uint8_t* __restrict__ output, float* __restrict__ output_scale,
    const int32_t* __restrict__ packed,
    const uint8_t* __restrict__ processed_scales,
    const float* __restrict__ processed_global_scale,
    const uint8_t* __restrict__ tile_scale_divisor_codes, int n_padded,
    int resident_k, int scratch_k, bool resident_bfloat16) {
  __shared__ __align__(16) uint32_t packed_tile[2 * 8 * 8 * 4 * 4];
  __shared__ uint32_t scale_tile[8][32];

  const int expert = MultipleExperts ? blockIdx.z : 0;
  const int n_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int n_tiles = n_padded / 64;
  const int k_blocks = scratch_k / 128;
  const int64_t scale_block =
      static_cast<int64_t>(n_block) * k_blocks + k_block;
  const int64_t blocks_per_expert =
      static_cast<int64_t>(n_padded / 128) * k_blocks;
  const int64_t packed_stride =
      static_cast<int64_t>(resident_k / 16) * 2 * n_padded;
  const int64_t scale_stride = static_cast<int64_t>(resident_k / 16) * n_padded;
  const int32_t* expert_packed = packed + expert * packed_stride;
  const uint8_t* expert_scales = processed_scales + expert * scale_stride;
  uint8_t* expert_output =
      output + static_cast<int64_t>(expert) * n_padded * scratch_k;
  float* expert_output_scale =
      output_scale + static_cast<int64_t>(expert) * blocks_per_expert;
  const uint8_t* expert_divisors =
      tile_scale_divisor_codes +
      static_cast<int64_t>(expert) * blocks_per_expert;

  float reciprocal_lane = 0.0f;
  if ((threadIdx.x & 31) == 0) {
    const half divisor = __ushort_as_half(
        static_cast<uint16_t>(expert_divisors[scale_block]) << 7);
    reciprocal_lane = __fdividef(1.0f, __half2float(divisor));
    if (threadIdx.x == 0) {
      const float exponent_compensation =
          resident_bfloat16 ? 0x1p-126f : 0x1p-14f;
      expert_output_scale[scale_block] = processed_global_scale[expert] *
                                         exponent_compensation *
                                         __half2float(divisor);
    }
  }
  const float tile_scale_reciprocal =
      __shfl_sync(0xffffffffu, reciprocal_lane, 0);

  const int scale_k_group = threadIdx.x >> 5;
  const int scale_word = threadIdx.x & 31;
  const int global_scale_k_group = 8 * k_block + scale_k_group;
  const int64_t scale_offset =
      static_cast<int64_t>(global_scale_k_group) * n_padded + 128 * n_block;
  copy_async(&scale_tile[scale_k_group][scale_word],
             reinterpret_cast<const uint32_t*>(expert_scales + scale_offset) +
                 scale_word);

  if constexpr (SyncPaired) {
    asm volatile("cp.async.commit_group;");
  }

#pragma unroll
  for (int input_pass = 0; input_pass < 2; ++input_pass) {
    const int input = threadIdx.x + 256 * input_pass;
    const int input_segment = input >> 5;
    const int input_chunk = input & 31;
    const int n_subtile = input_segment >> 3;
    const int k_group = input_segment & 7;
    const int n_tile = 2 * n_block + n_subtile;
    const int global_k_group = 8 * k_block + k_group;
    const int64_t marlin_tile =
        static_cast<int64_t>(global_k_group) * n_tiles + n_tile;
    const int column = input_chunk >> 2;
    const int pair = input_chunk & 3;
    if constexpr (SyncPaired) {
      const uint4 words = reinterpret_cast<const uint4*>(
          expert_packed + 128 * marlin_tile)[input_chunk];
#pragma unroll
      for (int marlin_warp = 0; marlin_warp < 4; ++marlin_warp) {
        packed_tile[packed_tile_index(n_subtile, k_group, column, pair,
                                      marlin_warp)] =
            component(words, marlin_warp);
      }
    } else {
      const uint32_t* source =
          reinterpret_cast<const uint32_t*>(expert_packed + 128 * marlin_tile) +
          4 * input_chunk;
#pragma unroll
      for (int marlin_warp = 0; marlin_warp < 4; ++marlin_warp) {
        copy_async(&packed_tile[packed_tile_index(n_subtile, k_group, column,
                                                  pair, marlin_warp)],
                   source + marlin_warp);
      }
    }
  }
  if constexpr (!SyncPaired) {
    asm volatile("cp.async.commit_group;");
  }
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();

#pragma unroll
  for (int output_pass = 0; output_pass < 2; ++output_pass) {
    const int lane = threadIdx.x & 31;
    const int row_pair_global =
        4 * (threadIdx.x >> 5) + (lane >> 3) + 32 * output_pass;
    const int row_pair = row_pair_global & 31;
    const int k_group = lane & 7;
    const int n_subtile = row_pair_global >> 5;
    const int marlin_warp = row_pair >> 3;
    const int column = row_pair & 7;
    const int n0 = 128 * n_block + 64 * n_subtile + 16 * marlin_warp + column;
    const int n1 = n0 + 8;
    const int global_k_group = 8 * k_block + k_group;
    const int row0_in_block = 64 * n_subtile + 16 * marlin_warp + column;
    const int row1_in_block = row0_in_block + 8;
    const int scale_physical0 =
        64 * n_subtile + processed_scale_row_index(row0_in_block);
    const int scale_physical1 =
        64 * n_subtile + processed_scale_row_index(row1_in_block);
    const uint32_t scale_word_value = scale_tile[k_group][scale_physical0 >> 2];
    const half raw_scale0 = __ushort_as_half(
        static_cast<uint16_t>(
            (scale_word_value >> (8 * (scale_physical0 & 3))) & 0xff)
        << 7);
    const half raw_scale1 = __ushort_as_half(
        static_cast<uint16_t>(
            (scale_word_value >> (8 * (scale_physical1 & 3))) & 0xff)
        << 7);
    const half scale0 = __float2half_rn(
        fminf(__half2float(raw_scale0) * tile_scale_reciprocal, 65504.0f));
    const half scale1 = __float2half_rn(
        fminf(__half2float(raw_scale1) * tile_scale_reciprocal, 65504.0f));
    const uint4 words =
        reinterpret_cast<const uint4*>(&packed_tile[packed_tile_index(
            n_subtile, k_group, column, 0, marlin_warp)])[0];
    uint16_t n0_first[4];
    uint16_t n0_second[4];
    uint16_t n1_first[4];
    uint16_t n1_second[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const uint32_t word = component(words, pair);
      half2 values0[2];
      half2 values1[2];
      marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
          static_cast<int>(word << 8), values0);
      marlin::dequant<half2, vllm::kFE2M1f.id(), false>(static_cast<int>(word),
                                                        values1);
      n0_first[pair] = convert_pair(values0[0], scale0);
      n0_second[pair] = convert_pair(values0[1], scale0);
      n1_first[pair] = convert_pair(values1[0], scale1);
      n1_second[pair] = convert_pair(values1[1], scale1);
    }
    const int64_t output_offset0 =
        static_cast<int64_t>(n0) * scratch_k + 16 * global_k_group;
    const int64_t output_offset1 =
        static_cast<int64_t>(n1) * scratch_k + 16 * global_k_group;
    reinterpret_cast<uint4*>(expert_output + output_offset0)[0] =
        make_uint4(pack_pairs(n0_first[0], n0_first[1]),
                   pack_pairs(n0_first[2], n0_first[3]),
                   pack_pairs(n0_second[0], n0_second[1]),
                   pack_pairs(n0_second[2], n0_second[3]));
    reinterpret_cast<uint4*>(expert_output + output_offset1)[0] =
        make_uint4(pack_pairs(n1_first[0], n1_first[1]),
                   pack_pairs(n1_first[2], n1_first[3]),
                   pack_pairs(n1_second[0], n1_second[1]),
                   pack_pairs(n1_second[2], n1_second[3]));
  }
}

__global__ __launch_bounds__(64) void marlin_nvfp4_to_fp8_sparse64_kernel(
    uint8_t* __restrict__ output, float* __restrict__ output_scale,
    const int32_t* __restrict__ packed,
    const uint8_t* __restrict__ processed_scales,
    const float* __restrict__ processed_global_scale,
    const uint8_t* __restrict__ tile_scale_divisor_codes, int n_padded,
    int resident_k, int scratch_k, bool resident_bfloat16) {
  const int n_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int k_blocks = scratch_k / 128;
  const int n_tiles = n_padded / 64;
  const int64_t scale_block =
      static_cast<int64_t>(n_block) * k_blocks + k_block;
  const half divisor = __ushort_as_half(
      static_cast<uint16_t>(tile_scale_divisor_codes[scale_block]) << 7);
  const float reciprocal_lane =
      (threadIdx.x & 31) == 0 ? __frcp_rn(__half2float(divisor)) : 0.0f;
  const float reciprocal = __shfl_sync(0xffffffffu, reciprocal_lane, 0);
  if (threadIdx.x == 0 && blockIdx.z == 0) {
    const float compensation = resident_bfloat16 ? 0x1p-126f : 0x1p-14f;
    output_scale[scale_block] =
        processed_global_scale[0] * compensation * __half2float(divisor);
  }

  const int unit = threadIdx.x + 64 * blockIdx.z;
  const int lane = unit & 31;
  const int row_pair_global = 4 * (unit >> 5) + (lane >> 3);
  const int row_pair = row_pair_global & 31;
  const int k_group = lane & 7;
  const int n_subtile = row_pair_global >> 5;
  const int marlin_warp = row_pair >> 3;
  const int column = row_pair & 7;
  const int global_k_group = 8 * k_block + k_group;
  const int global_n0 =
      128 * n_block + 64 * n_subtile + 16 * marlin_warp + column;
  const int global_n1 = global_n0 + 8;
  const int64_t output_offset0 =
      static_cast<int64_t>(global_n0) * scratch_k + 16 * global_k_group;
  const int64_t output_offset1 =
      static_cast<int64_t>(global_n1) * scratch_k + 16 * global_k_group;
  if (16 * global_k_group >= resident_k) {
    reinterpret_cast<uint4*>(output + output_offset0)[0] =
        make_uint4(0, 0, 0, 0);
    reinterpret_cast<uint4*>(output + output_offset1)[0] =
        make_uint4(0, 0, 0, 0);
    return;
  }

  const uint8_t scale_byte0 = processed_scales[processed_scale_index(
      global_n0, global_k_group, n_padded)];
  const uint8_t scale_byte1 = processed_scales[processed_scale_index(
      global_n1, global_k_group, n_padded)];
  const half raw_scale0 =
      __ushort_as_half(static_cast<uint16_t>(scale_byte0) << 7);
  const half raw_scale1 =
      __ushort_as_half(static_cast<uint16_t>(scale_byte1) << 7);
  const half scale0 =
      __float2half_rn(fminf(__half2float(raw_scale0) * reciprocal, 65504.0f));
  const half scale1 =
      __float2half_rn(fminf(__half2float(raw_scale1) * reciprocal, 65504.0f));
  const int n_tile = 2 * n_block + n_subtile;
  const int64_t marlin_tile =
      static_cast<int64_t>(global_k_group) * n_tiles + n_tile;
  const int32_t* words = packed + 128 * marlin_tile;
  uint16_t n0_first[4], n0_second[4], n1_first[4], n1_second[4];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const uint32_t word =
        static_cast<uint32_t>(words[16 * column + 4 * pair + marlin_warp]);
    half2 values0[2], values1[2];
    marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
        static_cast<int>(word << 8), values0);
    marlin::dequant<half2, vllm::kFE2M1f.id(), false>(static_cast<int>(word),
                                                      values1);
    n0_first[pair] = convert_pair(values0[0], scale0);
    n0_second[pair] = convert_pair(values0[1], scale0);
    n1_first[pair] = convert_pair(values1[0], scale1);
    n1_second[pair] = convert_pair(values1[1], scale1);
  }
  reinterpret_cast<uint4*>(output + output_offset0)[0] =
      make_uint4(pack_pairs(n0_first[0], n0_first[1]),
                 pack_pairs(n0_first[2], n0_first[3]),
                 pack_pairs(n0_second[0], n0_second[1]),
                 pack_pairs(n0_second[2], n0_second[3]));
  reinterpret_cast<uint4*>(output + output_offset1)[0] =
      make_uint4(pack_pairs(n1_first[0], n1_first[1]),
                 pack_pairs(n1_first[2], n1_first[3]),
                 pack_pairs(n1_second[0], n1_second[1]),
                 pack_pairs(n1_second[2], n1_second[3]));
}

__device__ __forceinline__ void stage_dense_double_buffer_tile(
    uint32_t* packed_tile, uint32_t scale_tile[][32], const int32_t* packed,
    const uint8_t* scales, int n_padded, int n_tiles, int n_block,
    int k_block) {
  const int scale_k_group = threadIdx.x >> 5;
  const int scale_word = threadIdx.x & 31;
  const int64_t scale_offset =
      static_cast<int64_t>(8 * k_block + scale_k_group) * n_padded +
      128 * n_block;
  const int scale_storage_word =
      scale_word ^ (8 * (scale_k_group >> 1) + (scale_k_group & 1));
  copy_async(
      &scale_tile[scale_k_group][scale_storage_word],
      reinterpret_cast<const uint32_t*>(scales + scale_offset) + scale_word);
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int input = threadIdx.x + 256 * pass;
    const int segment = input >> 5;
    const int chunk = input & 31;
    const int n_subtile = segment >> 3;
    const int k_group = segment & 7;
    const int64_t marlin_tile =
        static_cast<int64_t>(8 * k_block + k_group) * n_tiles + 2 * n_block +
        n_subtile;
    const uint32_t* source =
        reinterpret_cast<const uint32_t*>(packed + 128 * marlin_tile) +
        4 * chunk;
#pragma unroll
    for (int marlin_warp = 0; marlin_warp < 4; ++marlin_warp) {
      copy_async(&packed_tile[packed_tile_index(n_subtile, k_group, chunk >> 2,
                                                chunk & 3, marlin_warp)],
                 source + marlin_warp);
    }
  }
  asm volatile("cp.async.commit_group;");
}

__device__ __forceinline__ void consume_dense_double_buffer_tile(
    uint8_t* output, const uint32_t* packed_tile,
    const uint32_t scale_tile[][32], float reciprocal, int n_block, int k_block,
    int scratch_k) {
#pragma unroll
  for (int pass = 0; pass < 2; ++pass) {
    const int lane = threadIdx.x & 31;
    const int row_pair_global =
        4 * (threadIdx.x >> 5) + (lane >> 3) + 32 * pass;
    const int row_pair = row_pair_global & 31;
    const int k_group = lane & 7;
    const int n_subtile = row_pair_global >> 5;
    const int marlin_warp = row_pair >> 3;
    const int column = row_pair & 7;
    const int row0 = 64 * n_subtile + 16 * marlin_warp + column;
    const int n0 = 128 * n_block + row0;
    const int n1 = n0 + 8;
    const int global_k_group = 8 * k_block + k_group;
    const int scale_byte = marlin_warp & 1;
    const int logical_scale_word =
        16 * n_subtile + 2 * column + (marlin_warp >> 1);
    const int scale_storage_word =
        logical_scale_word ^ (8 * (k_group >> 1) + (k_group & 1));
    const uint32_t scale_word = scale_tile[k_group][scale_storage_word];
    const uint32_t scale_pair = scale_word >> (8 * scale_byte);
    const half raw_scale0 =
        __ushort_as_half(static_cast<uint16_t>(scale_pair & 0xff) << 7);
    const half raw_scale1 =
        __ushort_as_half(static_cast<uint16_t>((scale_pair >> 16) & 0xff) << 7);
    const half scale0 =
        __float2half_rn(fminf(__half2float(raw_scale0) * reciprocal, 65504.0f));
    const half scale1 =
        __float2half_rn(fminf(__half2float(raw_scale1) * reciprocal, 65504.0f));
    const uint4 words =
        reinterpret_cast<const uint4*>(&packed_tile[packed_tile_index(
            n_subtile, k_group, column, 0, marlin_warp)])[0];
    uint32_t n0_values[4], n1_values[4];
#pragma unroll
    for (int pair_group = 0; pair_group < 2; ++pair_group) {
      const uint32_t word0 = component(words, 2 * pair_group);
      const uint32_t word1 = component(words, 2 * pair_group + 1);
      half2 word0_n0[2], word0_n1[2], word1_n0[2], word1_n1[2];
      marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
          static_cast<int>(word0 << 8), word0_n0);
      marlin::dequant<half2, vllm::kFE2M1f.id(), false>(static_cast<int>(word0),
                                                        word0_n1);
      marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
          static_cast<int>(word1 << 8), word1_n0);
      marlin::dequant<half2, vllm::kFE2M1f.id(), false>(static_cast<int>(word1),
                                                        word1_n1);
      n0_values[pair_group] = convert_four(word0_n0[0], word1_n0[0], scale0);
      n0_values[pair_group + 2] =
          convert_four(word0_n0[1], word1_n0[1], scale0);
      n1_values[pair_group] = convert_four(word0_n1[0], word1_n1[0], scale1);
      n1_values[pair_group + 2] =
          convert_four(word0_n1[1], word1_n1[1], scale1);
    }
    const int64_t offset0 =
        static_cast<int64_t>(n0) * scratch_k + 16 * global_k_group;
    const int64_t offset1 =
        static_cast<int64_t>(n1) * scratch_k + 16 * global_k_group;
    reinterpret_cast<uint4*>(output + offset0)[0] =
        make_uint4(n0_values[0], n0_values[1], n0_values[2], n0_values[3]);
    reinterpret_cast<uint4*>(output + offset1)[0] =
        make_uint4(n1_values[0], n1_values[1], n1_values[2], n1_values[3]);
  }
}

__global__ __launch_bounds__(256) void marlin_nvfp4_to_fp8_double_buffer_kernel(
    uint8_t* __restrict__ output, float* __restrict__ output_scale,
    const int32_t* __restrict__ packed, const uint8_t* __restrict__ scales,
    const float* __restrict__ global_scale,
    const uint8_t* __restrict__ divisors, int n_padded, int scratch_k,
    bool resident_bfloat16) {
  __shared__ __align__(16) uint32_t packed_tile[2][2 * 8 * 8 * 4 * 4];
  __shared__ uint32_t scale_tile[2][8][32];
  const int n_block = blockIdx.x;
  const int k_blocks = scratch_k / 128;
  const int k_block[2] = {static_cast<int>(blockIdx.y),
                          static_cast<int>(blockIdx.y) + k_blocks / 2};
  const int n_tiles = n_padded / 64;
  float reciprocal[2];
#pragma unroll
  for (int stage = 0; stage < 2; ++stage) {
    const int64_t scale_block =
        static_cast<int64_t>(n_block) * k_blocks + k_block[stage];
    float reciprocal_lane = 0.0f;
    if ((threadIdx.x & 31) == 0) {
      const half divisor =
          __ushort_as_half(static_cast<uint16_t>(divisors[scale_block]) << 7);
      reciprocal_lane = __fdividef(1.0f, __half2float(divisor));
      if (threadIdx.x == 0) {
        const float compensation = resident_bfloat16 ? 0x1p-126f : 0x1p-14f;
        output_scale[scale_block] =
            global_scale[0] * compensation * __half2float(divisor);
      }
    }
    reciprocal[stage] = __shfl_sync(0xffffffffu, reciprocal_lane, 0);
  }
  stage_dense_double_buffer_tile(packed_tile[0], scale_tile[0], packed, scales,
                                 n_padded, n_tiles, n_block, k_block[0]);
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  stage_dense_double_buffer_tile(packed_tile[1], scale_tile[1], packed, scales,
                                 n_padded, n_tiles, n_block, k_block[1]);
  consume_dense_double_buffer_tile(output, packed_tile[0], scale_tile[0],
                                   reciprocal[0], n_block, k_block[0],
                                   scratch_k);
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  consume_dense_double_buffer_tile(output, packed_tile[1], scale_tile[1],
                                   reciprocal[1], n_block, k_block[1],
                                   scratch_k);
}

__global__ void marlin_nvfp4_to_fp8_kernel(
    uint8_t* output, float* output_scale, const int32_t* packed,
    const uint8_t* processed_scales, const float* processed_global_scale,
    const uint8_t* tile_scale_divisor_codes, int experts, int n_padded,
    int resident_k, int scratch_k, bool resident_bfloat16) {
  __shared__ __align__(16) uint16_t converted_tile[64 * 16];
  __shared__ half tile_scales[2 * 64];
  __shared__ float tile_scale_reciprocal;

  const int n_tiles = n_padded / 64;
  const int k_tile_groups = scratch_k / 32;
  const int n_scale_blocks = (n_padded + 127) / 128;
  const int k_scale_blocks = (scratch_k + 127) / 128;
  const int64_t tiles_per_expert =
      static_cast<int64_t>(n_tiles) * k_tile_groups;
  const int64_t total_tiles = tiles_per_expert * experts;
  const int64_t packed_stride =
      static_cast<int64_t>(resident_k / 16) * 2 * n_padded;
  const int64_t scale_stride = static_cast<int64_t>(resident_k / 16) * n_padded;

  for (int64_t tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int expert = static_cast<int>(tile / tiles_per_expert);
    const int64_t expert_tile =
        tile - static_cast<int64_t>(expert) * tiles_per_expert;
    const int k_tile_group = static_cast<int>(expert_tile / n_tiles);
    const int n_tile = static_cast<int>(expert_tile % n_tiles);
    const int64_t scale_block =
        (static_cast<int64_t>(expert) * n_scale_blocks + (n_tile >> 1)) *
            k_scale_blocks +
        (k_tile_group >> 2);
    const half tile_scale_divisor = __ushort_as_half(
        static_cast<uint16_t>(tile_scale_divisor_codes[scale_block]) << 7);
    if ((n_tile & 1) == 0 && (k_tile_group & 3) == 0 && threadIdx.x == 0) {
      const float exponent_compensation =
          resident_bfloat16 ? 0x1p-126f : 0x1p-14f;
      const float processed_global_compensation =
          processed_global_scale[expert] * exponent_compensation;
      output_scale[scale_block] =
          processed_global_compensation * __half2float(tile_scale_divisor);
    }
    if (32 * k_tile_group >= resident_k) {
      if (threadIdx.x < 128) {
        const int row = threadIdx.x >> 1;
        const int64_t output_offset =
            static_cast<int64_t>(expert) * n_padded * scratch_k +
            static_cast<int64_t>(64 * n_tile + row) * scratch_k +
            32 * k_tile_group + 16 * (threadIdx.x & 1);
        uint4* destination = reinterpret_cast<uint4*>(output + output_offset);
        destination[0] = make_uint4(0, 0, 0, 0);
      }
      continue;
    }
    if (threadIdx.x == 0) {
      tile_scale_reciprocal = __frcp_rn(__half2float(tile_scale_divisor));
    }
    __syncthreads();
    if (threadIdx.x < 128) {
      const int scale_subtile = threadIdx.x >> 6;
      const int scale_row = threadIdx.x & 63;
      const int global_row = 64 * n_tile + scale_row;
      const int global_k_tile = 2 * k_tile_group + scale_subtile;
      const uint8_t scale_byte =
          processed_scales[static_cast<int64_t>(expert) * scale_stride +
                           processed_scale_index(global_row, global_k_tile,
                                                 n_padded)];
      const half scale =
          __ushort_as_half(static_cast<uint16_t>(scale_byte) << 7);
      const float normalized_scale =
          fminf(__half2float(scale) * tile_scale_reciprocal, 65504.0f);
      tile_scales[threadIdx.x] = __float2half_rn(normalized_scale);
    }
    __syncthreads();

    const int k_subtile = threadIdx.x >> 7;
    const int word_in_tile = threadIdx.x & 127;
    const int k_tile = 2 * k_tile_group + k_subtile;
    const int64_t marlin_tile = static_cast<int64_t>(k_tile) * n_tiles + n_tile;
    const uint32_t word = static_cast<uint32_t>(
        packed[static_cast<int64_t>(expert) * packed_stride +
               128 * marlin_tile + word_in_tile]);
    const int warp = word_in_tile & 3;
    const int marlin_thread = word_in_tile >> 2;
    const int pair = marlin_thread & 3;
    const int column = marlin_thread >> 2;
    const int n0 = 16 * warp + column;
    const int n1 = n0 + 8;
    const int pair_base = k_subtile * 8;
    const half scale0 = tile_scales[k_subtile * 64 + n0];
    const half scale1 = tile_scales[k_subtile * 64 + n1];
    half2 values0[2];
    half2 values1[2];
    marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
        static_cast<int>(word << 8), values0);
    marlin::dequant<half2, vllm::kFE2M1f.id(), false>(static_cast<int>(word),
                                                      values1);
    converted_tile[n0 * 16 + pair_base + pair] =
        convert_pair(values0[0], scale0);
    converted_tile[n0 * 16 + pair_base + 4 + pair] =
        convert_pair(values0[1], scale0);
    converted_tile[n1 * 16 + pair_base + pair] =
        convert_pair(values1[0], scale1);
    converted_tile[n1 * 16 + pair_base + 4 + pair] =
        convert_pair(values1[1], scale1);
    __syncthreads();

    if (threadIdx.x < 128) {
      const int row = threadIdx.x >> 1;
      const int64_t output_offset =
          static_cast<int64_t>(expert) * n_padded * scratch_k +
          static_cast<int64_t>(64 * n_tile + row) * scratch_k +
          32 * k_tile_group + 16 * (threadIdx.x & 1);
      const uint4* source = reinterpret_cast<const uint4*>(converted_tile);
      uint4* destination = reinterpret_cast<uint4*>(output + output_offset);
      destination[0] = source[threadIdx.x];
    }
  }
}

struct MemoryRange {
  uintptr_t begin;
  uintptr_t end;
};

MemoryRange memory_range(const Tensor& tensor) {
  const uintptr_t begin = reinterpret_cast<uintptr_t>(tensor.const_data_ptr());
  const uint64_t numel = static_cast<uint64_t>(tensor.numel());
  const size_t element_size = tensor.element_size();
  STD_TORCH_CHECK(
      numel <= (std::numeric_limits<uintptr_t>::max() - begin) / element_size,
      "tensor byte range overflows host pointer width");
  return {begin, begin + numel * element_size};
}

bool overlaps(const MemoryRange& a, const MemoryRange& b) {
  return a.begin < b.end && b.begin < a.end;
}

void check_cuda_contiguous(const Tensor& tensor, const char* name) {
  STD_TORCH_CHECK(tensor.is_cuda(), name, " must be CUDA");
  STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
}

void marlin_nvfp4_to_fp8(Tensor& fp8_out, Tensor& scale_out,
                         const Tensor& packed_weight,
                         const Tensor& processed_block_scales,
                         const Tensor& processed_global_scale,
                         const Tensor& tile_scale_divisor_codes,
                         ScalarType resident_dtype) {
  check_cuda_contiguous(fp8_out, "fp8_out");
  check_cuda_contiguous(scale_out, "scale_out");
  check_cuda_contiguous(packed_weight, "packed_weight");
  check_cuda_contiguous(processed_block_scales, "processed_block_scales");
  check_cuda_contiguous(processed_global_scale, "processed_global_scale");
  check_cuda_contiguous(tile_scale_divisor_codes, "tile_scale_divisor_codes");

  STD_TORCH_CHECK(fp8_out.scalar_type() == ScalarType::Float8_e4m3fn,
                  "fp8_out must be float8_e4m3fn");
  STD_TORCH_CHECK(scale_out.scalar_type() == ScalarType::Float,
                  "scale_out must be float32");
  STD_TORCH_CHECK(packed_weight.scalar_type() == ScalarType::Int,
                  "packed_weight must be int32 Marlin storage");
  STD_TORCH_CHECK(
      processed_block_scales.scalar_type() == ScalarType::Float8_e4m3fn,
      "processed_block_scales must use float8_e4m3fn byte storage");
  STD_TORCH_CHECK(processed_global_scale.scalar_type() == ScalarType::Float,
                  "processed_global_scale must be float32");
  STD_TORCH_CHECK(tile_scale_divisor_codes.scalar_type() == ScalarType::Byte,
                  "tile_scale_divisor_codes must be uint8");
  STD_TORCH_CHECK(resident_dtype == ScalarType::Half ||
                      resident_dtype == ScalarType::BFloat16,
                  "resident_dtype must be float16 or bfloat16");

  STD_TORCH_CHECK(fp8_out.dim() == 2 || fp8_out.dim() == 3,
                  "fp8_out must be rank 2 or rank 3");
  const int64_t experts = fp8_out.dim() == 3 ? fp8_out.size(0) : 1;
  const int64_t n_padded = fp8_out.size(-2);
  const int64_t scratch_k = fp8_out.size(-1);
  STD_TORCH_CHECK(experts > 0 && n_padded > 0 && scratch_k > 0,
                  "output dimensions must be positive");
  STD_TORCH_CHECK(n_padded % 64 == 0 && scratch_k % 128 == 0,
                  "Marlin output requires N and K divisible by 64");
  STD_TORCH_CHECK(
      reinterpret_cast<uintptr_t>(fp8_out.const_data_ptr()) % alignof(uint4) ==
          0,
      "fp8_out must be 16-byte aligned");
  STD_TORCH_CHECK(n_padded <= std::numeric_limits<int>::max() &&
                      scratch_k <= std::numeric_limits<int>::max() &&
                      experts <= std::numeric_limits<int>::max(),
                  "dimensions exceed 32-bit kernel indexing");

  STD_TORCH_CHECK(packed_weight.dim() == fp8_out.dim(),
                  "packed_weight rank must match fp8_out");
  STD_TORCH_CHECK(processed_block_scales.dim() == fp8_out.dim(),
                  "processed_block_scales rank must match fp8_out");
  if (fp8_out.dim() == 3) {
    STD_TORCH_CHECK(packed_weight.size(0) == experts &&
                        processed_block_scales.size(0) == experts,
                    "expert dimensions must match");
  }
  const int64_t resident_k = packed_weight.size(-2) * 16;
  STD_TORCH_CHECK(resident_k > 0 && resident_k % 64 == 0 &&
                      scratch_k == ((resident_k + 127) / 128) * 128,
                  "fp8_out K must round resident Marlin K up to 128");
  STD_TORCH_CHECK(packed_weight.size(-1) == 2 * n_padded,
                  "packed_weight has incompatible Marlin dimensions");
  STD_TORCH_CHECK(processed_block_scales.size(-2) == resident_k / 16 &&
                      processed_block_scales.size(-1) == n_padded,
                  "processed_block_scales has incompatible Marlin dimensions");
  const int64_t n_scale_blocks = (n_padded + 127) / 128;
  const int64_t k_scale_blocks = scratch_k / 128;
  STD_TORCH_CHECK(scale_out.dim() == fp8_out.dim(),
                  "scale_out rank must match fp8_out");
  STD_TORCH_CHECK(tile_scale_divisor_codes.dim() == fp8_out.dim(),
                  "tile_scale_divisor_codes rank must match fp8_out");
  if (fp8_out.dim() == 3) {
    STD_TORCH_CHECK(scale_out.size(0) == experts &&
                        tile_scale_divisor_codes.size(0) == experts,
                    "scale expert dimensions must match fp8_out");
  }
  STD_TORCH_CHECK(scale_out.size(-2) == n_scale_blocks &&
                      scale_out.size(-1) == k_scale_blocks,
                  "scale_out must contain one value per 128x128 tile");
  STD_TORCH_CHECK(tile_scale_divisor_codes.size(-2) == n_scale_blocks &&
                      tile_scale_divisor_codes.size(-1) == k_scale_blocks,
                  "tile_scale_divisor_codes must contain one byte per "
                  "128x128 tile");
  STD_TORCH_CHECK(processed_global_scale.numel() == experts,
                  "processed_global_scale must contain one value per expert");

  const int32_t device = fp8_out.get_device_index();
  STD_TORCH_CHECK(scale_out.get_device_index() == device &&
                      packed_weight.get_device_index() == device &&
                      processed_block_scales.get_device_index() == device &&
                      processed_global_scale.get_device_index() == device &&
                      tile_scale_divisor_codes.get_device_index() == device,
                  "all tensors must be on the same CUDA device");

  const MemoryRange fp8_range = memory_range(fp8_out);
  const MemoryRange scale_range = memory_range(scale_out);
  const MemoryRange packed_range = memory_range(packed_weight);
  const MemoryRange block_scales_range = memory_range(processed_block_scales);
  const MemoryRange global_scale_range = memory_range(processed_global_scale);
  const MemoryRange divisor_range = memory_range(tile_scale_divisor_codes);
  STD_TORCH_CHECK(!overlaps(fp8_range, scale_range),
                  "mutable outputs must not overlap");
  STD_TORCH_CHECK(!overlaps(fp8_range, packed_range) &&
                      !overlaps(scale_range, packed_range) &&
                      !overlaps(fp8_range, block_scales_range) &&
                      !overlaps(scale_range, block_scales_range) &&
                      !overlaps(fp8_range, global_scale_range) &&
                      !overlaps(scale_range, global_scale_range) &&
                      !overlaps(fp8_range, divisor_range) &&
                      !overlaps(scale_range, divisor_range),
                  "mutable outputs must not overlap resident inputs");

  const torch::stable::accelerator::DeviceGuard device_guard(device);
  const cudaDeviceProp* device_properties = get_device_prop();
  STD_TORCH_CHECK(
      (device_properties->major == 8 && device_properties->minor == 9) ||
          (device_properties->major == 9 && device_properties->minor == 0),
      "marlin_nvfp4_to_fp8 requires SM89 or SM90");

  const cudaStream_t stream = get_current_cuda_stream(device);
  constexpr int threads = 256;
  const uintptr_t packed_address =
      reinterpret_cast<uintptr_t>(packed_weight.const_data_ptr());
  const uintptr_t scales_address =
      reinterpret_cast<uintptr_t>(processed_block_scales.const_data_ptr());
  const bool tiled_inputs_aligned = packed_address % alignof(uint4) == 0 &&
                                    scales_address % alignof(uint32_t) == 0;
  if (n_padded % 128 == 0 && resident_k == scratch_k && tiled_inputs_aligned &&
      experts <= 65535 && scratch_k / 128 <= 65535) {
    const dim3 blocks(static_cast<uint32_t>(n_padded / 128),
                      static_cast<uint32_t>(scratch_k / 128),
                      static_cast<uint32_t>(experts));
    const int64_t tile_count = (n_padded / 128) * (scratch_k / 128);
    const bool sm90_single = experts == 1 && device_properties->major == 9;
    if (sm90_single && tile_count <= 64) {
      const dim3 sparse_blocks(static_cast<uint32_t>(n_padded / 128),
                               static_cast<uint32_t>(scratch_k / 128), 8);
      marlin_nvfp4_to_fp8_sparse64_kernel<<<sparse_blocks, 64, 0, stream>>>(
          reinterpret_cast<uint8_t*>(fp8_out.mutable_data_ptr()),
          scale_out.mutable_data_ptr<float>(),
          packed_weight.const_data_ptr<int32_t>(),
          reinterpret_cast<const uint8_t*>(
              processed_block_scales.const_data_ptr()),
          processed_global_scale.const_data_ptr<float>(),
          tile_scale_divisor_codes.const_data_ptr<uint8_t>(),
          static_cast<int>(n_padded), static_cast<int>(resident_k),
          static_cast<int>(scratch_k), resident_dtype == ScalarType::BFloat16);
    } else if (sm90_single) {
      marlin_nvfp4_to_fp8_tiled_kernel<false, true>
          <<<blocks, threads, 0, stream>>>(
              reinterpret_cast<uint8_t*>(fp8_out.mutable_data_ptr()),
              scale_out.mutable_data_ptr<float>(),
              packed_weight.const_data_ptr<int32_t>(),
              reinterpret_cast<const uint8_t*>(
                  processed_block_scales.const_data_ptr()),
              processed_global_scale.const_data_ptr<float>(),
              tile_scale_divisor_codes.const_data_ptr<uint8_t>(),
              static_cast<int>(n_padded), static_cast<int>(resident_k),
              static_cast<int>(scratch_k),
              resident_dtype == ScalarType::BFloat16);
    } else if (experts == 1 && (scratch_k / 128) % 2 == 0) {
      const dim3 double_buffer_blocks(static_cast<uint32_t>(n_padded / 128),
                                      static_cast<uint32_t>(scratch_k / 256));
      marlin_nvfp4_to_fp8_double_buffer_kernel<<<double_buffer_blocks, threads,
                                                 0, stream>>>(
          reinterpret_cast<uint8_t*>(fp8_out.mutable_data_ptr()),
          scale_out.mutable_data_ptr<float>(),
          packed_weight.const_data_ptr<int32_t>(),
          reinterpret_cast<const uint8_t*>(
              processed_block_scales.const_data_ptr()),
          processed_global_scale.const_data_ptr<float>(),
          tile_scale_divisor_codes.const_data_ptr<uint8_t>(),
          static_cast<int>(n_padded), static_cast<int>(scratch_k),
          resident_dtype == ScalarType::BFloat16);
    } else if (experts == 1) {
      marlin_nvfp4_to_fp8_tiled_kernel<false><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<uint8_t*>(fp8_out.mutable_data_ptr()),
          scale_out.mutable_data_ptr<float>(),
          packed_weight.const_data_ptr<int32_t>(),
          reinterpret_cast<const uint8_t*>(
              processed_block_scales.const_data_ptr()),
          processed_global_scale.const_data_ptr<float>(),
          tile_scale_divisor_codes.const_data_ptr<uint8_t>(),
          static_cast<int>(n_padded), static_cast<int>(resident_k),
          static_cast<int>(scratch_k), resident_dtype == ScalarType::BFloat16);
    } else {
      marlin_nvfp4_to_fp8_tiled_kernel<true><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<uint8_t*>(fp8_out.mutable_data_ptr()),
          scale_out.mutable_data_ptr<float>(),
          packed_weight.const_data_ptr<int32_t>(),
          reinterpret_cast<const uint8_t*>(
              processed_block_scales.const_data_ptr()),
          processed_global_scale.const_data_ptr<float>(),
          tile_scale_divisor_codes.const_data_ptr<uint8_t>(),
          static_cast<int>(n_padded), static_cast<int>(resident_k),
          static_cast<int>(scratch_k), resident_dtype == ScalarType::BFloat16);
    }
  } else {
    const int64_t tiles = experts * (n_padded / 64) * (scratch_k / 32);
    const int blocks = static_cast<int>(std::min<int64_t>(tiles, 65535));
    marlin_nvfp4_to_fp8_kernel<<<blocks, threads, 0, stream>>>(
        reinterpret_cast<uint8_t*>(fp8_out.mutable_data_ptr()),
        scale_out.mutable_data_ptr<float>(),
        packed_weight.const_data_ptr<int32_t>(),
        reinterpret_cast<const uint8_t*>(
            processed_block_scales.const_data_ptr()),
        processed_global_scale.const_data_ptr<float>(),
        tile_scale_divisor_codes.const_data_ptr<uint8_t>(),
        static_cast<int>(experts), static_cast<int>(n_padded),
        static_cast<int>(resident_k), static_cast<int>(scratch_k),
        resident_dtype == ScalarType::BFloat16);
  }
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(
      error == cudaSuccess,
      "marlin_nvfp4_to_fp8 kernel launch failed: ", cudaGetErrorString(error));
}

Tensor marlin_nvfp4_hybrid_linear(Tensor& input, Tensor& packed_weight,
                                  Tensor& processed_block_scales,
                                  Tensor& processed_global_scale,
                                  Tensor& tile_scale_divisor_codes,
                                  Tensor& marlin_workspace, int64_t m_knee,
                                  bool use_atomic_add, bool use_fp32_reduce) {
  STD_TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
  check_cuda_contiguous(packed_weight, "packed_weight");
  check_cuda_contiguous(processed_block_scales, "processed_block_scales");
  check_cuda_contiguous(processed_global_scale, "processed_global_scale");
  check_cuda_contiguous(tile_scale_divisor_codes, "tile_scale_divisor_codes");
  check_cuda_contiguous(marlin_workspace, "marlin_workspace");
  STD_TORCH_CHECK(input.dim() == 2, "input must be rank 2");
  STD_TORCH_CHECK(processed_block_scales.dim() == 2,
                  "processed_block_scales must be rank 2");
  const int64_t size_n = processed_block_scales.size(1);
  const int64_t size_k = input.size(1);
  STD_TORCH_CHECK(
      size_n > 0 && size_k > 0 && size_n % 128 == 0 && size_k % 128 == 0,
      "size_n and size_k must be positive multiples of 128");
  const int64_t size_m = input.size(0);
  STD_TORCH_CHECK(m_knee > 0, "m_knee must be positive");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::Half ||
                      input.scalar_type() == ScalarType::BFloat16,
                  "input must be float16 or bfloat16");
  STD_TORCH_CHECK(packed_weight.dim() == 2 &&
                      packed_weight.size(0) == size_k / 16 &&
                      packed_weight.size(1) == 2 * size_n &&
                      packed_weight.scalar_type() == ScalarType::Int,
                  "packed_weight has incompatible int32 Marlin dimensions");
  STD_TORCH_CHECK(
      processed_block_scales.dim() == 2 &&
          processed_block_scales.size(0) == size_k / 16 &&
          processed_block_scales.size(1) == size_n &&
          processed_block_scales.scalar_type() == ScalarType::Float8_e4m3fn,
      "processed_block_scales has incompatible dimensions");
  STD_TORCH_CHECK(processed_global_scale.numel() == 1 &&
                      processed_global_scale.scalar_type() == ScalarType::Float,
                  "processed_global_scale must contain one float32 value");
  STD_TORCH_CHECK(
      tile_scale_divisor_codes.dim() == 2 &&
          tile_scale_divisor_codes.size(0) == size_n / 128 &&
          tile_scale_divisor_codes.size(1) == size_k / 128 &&
          tile_scale_divisor_codes.scalar_type() == ScalarType::Byte,
      "tile_scale_divisor_codes must be uint8 [N/128, K/128]");
  STD_TORCH_CHECK(marlin_workspace.scalar_type() == ScalarType::Int,
                  "marlin_workspace must be int32");
  const int32_t device = input.get_device_index();
  STD_TORCH_CHECK(packed_weight.get_device_index() == device &&
                      processed_block_scales.get_device_index() == device &&
                      processed_global_scale.get_device_index() == device &&
                      tile_scale_divisor_codes.get_device_index() == device &&
                      marlin_workspace.get_device_index() == device,
                  "all tensors must be on the same CUDA device");
  if (size_m < m_knee) {
    const std::optional<Tensor> no_tensor = std::nullopt;
    const std::optional<Tensor> global_scale{processed_global_scale};
    return ::marlin_gemm(input, no_tensor, packed_weight, no_tensor,
                         processed_block_scales, no_tensor, global_scale,
                         no_tensor, no_tensor, no_tensor, marlin_workspace,
                         vllm::kFE2M1f.id(), size_m, size_n, size_k, true,
                         use_atomic_add, use_fp32_reduce, false);
  }
  Tensor quant_input = torch::stable::contiguous(input);
  const int64_t padded_m = (size_m + 3) / 4 * 4;
  Tensor output = torch::stable::new_empty(quant_input, {padded_m, size_n},
                                           input.scalar_type());
  Tensor activation_q = torch::stable::new_empty(
      quant_input, {padded_m, size_k}, ScalarType::Float8_e4m3fn);
  Tensor activation_scale_backing = torch::stable::new_empty(
      quant_input, {size_k / 128, padded_m}, ScalarType::Float);
  Tensor fp8_weight = torch::stable::new_empty(quant_input, {size_n, size_k},
                                               ScalarType::Float8_e4m3fn);
  Tensor fp8_weight_scale = torch::stable::new_empty(
      quant_input, {size_n / 128, size_k / 128}, ScalarType::Float);
  Tensor activation_scale =
      torch::stable::transpose(activation_scale_backing, 0, 1);
  Tensor logical_activation_q =
      torch::stable::narrow(activation_q, 0, 0, size_m);
  Tensor logical_activation_scale =
      torch::stable::narrow(activation_scale, 0, 0, size_m);
  per_token_group_quant_fp8(quant_input, logical_activation_q,
                            logical_activation_scale, 128, 1.0e-10, -448.0,
                            448.0, false, true, false);
  marlin_nvfp4_to_fp8(fp8_weight, fp8_weight_scale, packed_weight,
                      processed_block_scales, processed_global_scale,
                      tile_scale_divisor_codes, input.scalar_type());
  Tensor weight_t = torch::stable::transpose(fp8_weight, 0, 1);
  Tensor weight_scale_t = torch::stable::transpose(fp8_weight_scale, 0, 1);
  cutlass_scaled_mm(output, activation_q, weight_t, activation_scale,
                    weight_scale_t, std::nullopt);
  return torch::stable::narrow(output, 0, 0, size_m);
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, marlin_nvfp4_to_fp8_schema) {
  marlin_nvfp4_to_fp8_schema.def(
      "marlin_nvfp4_to_fp8(Tensor(a!) fp8_out, Tensor(b!) scale_out, "
      "Tensor packed_weight, Tensor processed_block_scales, "
      "Tensor processed_global_scale, Tensor tile_scale_divisor_codes, "
      "ScalarType resident_dtype) -> ()");
  marlin_nvfp4_to_fp8_schema.def(
      "marlin_nvfp4_hybrid_linear(Tensor input, Tensor packed_weight, "
      "Tensor processed_block_scales, "
      "Tensor processed_global_scale, Tensor tile_scale_divisor_codes, "
      "Tensor marlin_workspace, int m_knee, bool use_atomic_add, "
      "bool use_fp32_reduce) -> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, marlin_nvfp4_to_fp8_impl) {
  marlin_nvfp4_to_fp8_impl.impl("marlin_nvfp4_to_fp8",
                                TORCH_BOX(&marlin_nvfp4_to_fp8));
  marlin_nvfp4_to_fp8_impl.impl("marlin_nvfp4_hybrid_linear",
                                TORCH_BOX(&marlin_nvfp4_hybrid_linear));
}
