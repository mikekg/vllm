#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "dequant.h"

#define CUDA_CHECK(expr)                                      \
  do {                                                        \
    const cudaError_t error = (expr);                         \
    if (error != cudaSuccess) {                               \
      std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(error));                \
      std::exit(1);                                           \
    }                                                         \
  } while (0)

namespace {

__device__ __forceinline__ int64_t processed_scale_index(int n, int group,
                                                         int n_padded) {
  const int lane = n & 63;
  const int transposed = 8 * (lane & 7) + (lane >> 3);
  const int reordered =
      (transposed & ~3) | ((transposed & 1) << 1) | ((transposed & 2) >> 1);
  return static_cast<int64_t>(group) * n_padded + 64 * (n >> 6) + reordered;
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

__device__ __forceinline__ void copy_async(uint4* destination,
                                           const uint4* source) {
  const uint32_t shared =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared), "l"(source));
}

__device__ __forceinline__ void copy_async(uint32_t* destination,
                                           const uint32_t* source) {
  const uint32_t shared =
      static_cast<uint32_t>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;"
               :
               : "r"(shared), "l"(source));
}

__device__ __forceinline__ uint16_t e2m1_half(uint16_t biased) {
  const uint16_t magnitude = (biased >> 9) & 7;
  const uint16_t magnitude_bits =
      magnitude == 0
          ? 0
          : static_cast<uint16_t>(
                (27 + magnitude + static_cast<int>(magnitude > 1)) << 9);
  return (biased & 0x8000) | magnitude_bits;
}

__device__ __forceinline__ uint32_t e2m1_half2(uint32_t biased) {
  return e2m1_half(biased) |
         (static_cast<uint32_t>(e2m1_half(biased >> 16)) << 16);
}

__device__ __forceinline__ void dequant_e2m1(int q, half2* values) {
  constexpr uint32_t sign = 0x80008000;
  constexpr uint32_t magnitude = 0x70007000;
  uint32_t first = (q & sign) | ((q & magnitude) >> 3);
  q <<= 4;
  uint32_t second = (q & sign) | ((q & magnitude) >> 3);
  first = e2m1_half2(first);
  second = e2m1_half2(second);
  values[1] = *reinterpret_cast<half2*>(&first);
  values[0] = *reinterpret_cast<half2*>(&second);
}

template <bool DirectDecode, bool SkipBias>
__device__ __forceinline__ void decode_e2m1(int q, half2* values) {
  if constexpr (DirectDecode) {
    static_assert(!SkipBias);
    dequant_e2m1(q, values);
  } else {
    marlin::dequant<half2, vllm::kFE2M1f.id(), SkipBias>(q, values);
  }
}

template <bool Prefetch>
__device__ __forceinline__ uint32_t load_packed(const int32_t* address) {
  if constexpr (Prefetch) {
    uint32_t value;
    asm volatile("ld.global.L2::128B.u32 %0, [%1];"
                 : "=r"(value)
                 : "l"(address));
    return value;
  } else {
    return static_cast<uint32_t>(*address);
  }
}

__device__ __forceinline__ uint32_t load_scale_byte(const uint8_t* address) {
  uint32_t value;
  asm volatile("ld.global.ca.u8 %0, [%1];" : "=r"(value) : "l"(address));
  return value;
}

__device__ __forceinline__ int packed_tile_index(int n_subtile, int k_group,
                                                 int column, int pair,
                                                 int marlin_warp) {
  const int bank = 4 * ((column ^ k_group) & 7) + pair;
  const int high = (4 * n_subtile + marlin_warp) * 8 + k_group;
  return 32 * high + bank;
}

__global__ void reference_kernel(uint8_t* output, float* output_scale,
                                 const int32_t* packed,
                                 const uint8_t* processed_scales,
                                 const float* processed_global_scale,
                                 const uint8_t* tile_scale_divisor_codes,
                                 int n_padded, int resident_k, int scratch_k) {
  __shared__ __align__(16) uint16_t converted_tile[64 * 16];
  __shared__ half tile_scales[2 * 64];
  __shared__ float tile_scale_reciprocal;

  const int n_tiles = n_padded / 64;
  const int k_tile_groups = scratch_k / 32;
  const int n_scale_blocks = (n_padded + 127) / 128;
  const int k_scale_blocks = (scratch_k + 127) / 128;
  const int total_tiles = n_tiles * k_tile_groups;

  for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
    const int k_tile_group = tile / n_tiles;
    const int n_tile = tile % n_tiles;
    const int scale_block =
        (n_tile >> 1) * k_scale_blocks + (k_tile_group >> 2);
    const half tile_scale_divisor = __ushort_as_half(
        static_cast<uint16_t>(tile_scale_divisor_codes[scale_block]) << 7);
    if ((n_tile & 1) == 0 && (k_tile_group & 3) == 0 && threadIdx.x == 0) {
      output_scale[scale_block] = processed_global_scale[0] * 0x1p-14f *
                                  __half2float(tile_scale_divisor);
    }
    if (32 * k_tile_group >= resident_k) {
      if (threadIdx.x < 128) {
        const int row = threadIdx.x >> 1;
        const int64_t offset =
            static_cast<int64_t>(64 * n_tile + row) * scratch_k +
            32 * k_tile_group + 16 * (threadIdx.x & 1);
        reinterpret_cast<uint4*>(output + offset)[0] = make_uint4(0, 0, 0, 0);
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
      const uint8_t scale_byte = processed_scales[processed_scale_index(
          global_row, global_k_tile, n_padded)];
      const half scale =
          __ushort_as_half(static_cast<uint16_t>(scale_byte) << 7);
      const float normalized =
          fminf(__half2float(scale) * tile_scale_reciprocal, 65504.0f);
      tile_scales[threadIdx.x] = __float2half_rn(normalized);
    }
    __syncthreads();

    const int k_subtile = threadIdx.x >> 7;
    const int word_in_tile = threadIdx.x & 127;
    const int k_tile = 2 * k_tile_group + k_subtile;
    const int marlin_tile = k_tile * n_tiles + n_tile;
    const uint32_t word =
        static_cast<uint32_t>(packed[128 * marlin_tile + word_in_tile]);
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
      const int64_t offset =
          static_cast<int64_t>(64 * n_tile + row) * scratch_k +
          32 * k_tile_group + 16 * (threadIdx.x & 1);
      const uint4* source = reinterpret_cast<const uint4*>(converted_tile);
      reinterpret_cast<uint4*>(output + offset)[0] = source[threadIdx.x];
    }
  }
}

template <int Threads, bool Paired = false, bool PrefetchScale = false,
          bool AsyncScatter = false, bool ScaleSwizzle = false>
__global__ __launch_bounds__(Threads) void tiled_kernel(
    uint8_t* __restrict__ output, float* __restrict__ output_scale,
    const int32_t* __restrict__ packed,
    const uint8_t* __restrict__ processed_scales,
    const float* __restrict__ processed_global_scale,
    const uint8_t* __restrict__ tile_scale_divisor_codes, int n_padded,
    int resident_k, int scratch_k) {
  __shared__ uint32_t packed_tile[2 * 8 * 8 * 4 * 4];
  __shared__ uint32_t scale_tile[8][32];

  const int k_blocks = scratch_k / 128;
  const int n_tiles = n_padded / 64;
  const int n_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int scale_block = n_block * k_blocks + k_block;

  uint32_t prefetched_scales[1024 / Threads];
  if constexpr (PrefetchScale) {
#pragma unroll
    for (int output_pass = 0; output_pass < 1024 / Threads; ++output_pass) {
      const int output_segment = threadIdx.x + Threads * output_pass;
      const int row_in_block = output_segment >> 3;
      const int output_k_group = output_segment & 7;
      const int global_row = 128 * n_block + row_in_block;
      const int global_output_k_group = 8 * k_block + output_k_group;
      prefetched_scales[output_pass] = load_scale_byte(
          processed_scales +
          processed_scale_index(global_row, global_output_k_group, n_padded));
    }
  }

  float reciprocal_lane = 0.0f;
  if ((threadIdx.x & 31) == 0) {
    const half divisor = __ushort_as_half(
        static_cast<uint16_t>(tile_scale_divisor_codes[scale_block]) << 7);
    reciprocal_lane = __fdividef(1.0f, __half2float(divisor));
    if (threadIdx.x == 0) {
      output_scale[scale_block] =
          processed_global_scale[0] * 0x1p-14f * __half2float(divisor);
    }
  }
  const float tile_scale_reciprocal =
      __shfl_sync(0xffffffffu, reciprocal_lane, 0);

  static_assert(!(PrefetchScale && AsyncScatter));
  static_assert(!ScaleSwizzle || AsyncScatter);
  if constexpr (!PrefetchScale) {
    if (threadIdx.x < 256) {
      const int scale_k_group = threadIdx.x >> 5;
      const int scale_word = threadIdx.x & 31;
      const int global_scale_k_group = 8 * k_block + scale_k_group;
      const int64_t scale_offset =
          static_cast<int64_t>(global_scale_k_group) * n_padded + 128 * n_block;
      const int scale_storage_word =
          ScaleSwizzle
              ? scale_word ^ (8 * (scale_k_group >> 1) + (scale_k_group & 1))
              : scale_word;
      copy_async(
          &scale_tile[scale_k_group][scale_storage_word],
          reinterpret_cast<const uint32_t*>(processed_scales + scale_offset) +
              scale_word);
    }
    if constexpr (!AsyncScatter) {
      asm volatile("cp.async.commit_group;");
    }
  }

#pragma unroll
  for (int input_pass = 0; input_pass < 512 / Threads; ++input_pass) {
    const int input = threadIdx.x + Threads * input_pass;
    const int input_segment = input >> 5;
    const int input_chunk = input & 31;
    const int n_subtile = input_segment >> 3;
    const int k_group = input_segment & 7;
    const int n_tile = 2 * n_block + n_subtile;
    const int global_k_group = 8 * k_block + k_group;
    const int input_n_subtile = input_segment >> 3;
    const int input_k_group = input_segment & 7;
    const int input_column = input_chunk >> 2;
    const int input_pair = input_chunk & 3;
    const bool valid = n_tile < n_tiles && 16 * global_k_group < resident_k;
    if constexpr (AsyncScatter) {
      if (valid) {
        const int marlin_tile = global_k_group * n_tiles + n_tile;
        const uint32_t* source =
            reinterpret_cast<const uint32_t*>(packed + 128 * marlin_tile) +
            4 * input_chunk;
#pragma unroll
        for (int marlin_warp = 0; marlin_warp < 4; ++marlin_warp) {
          copy_async(&packed_tile[packed_tile_index(input_n_subtile,
                                                    input_k_group, input_column,
                                                    input_pair, marlin_warp)],
                     source + marlin_warp);
        }
      } else {
#pragma unroll
        for (int marlin_warp = 0; marlin_warp < 4; ++marlin_warp) {
          packed_tile[packed_tile_index(input_n_subtile, input_k_group,
                                        input_column, input_pair,
                                        marlin_warp)] = 0;
        }
      }
    } else {
      uint4 packed_value = make_uint4(0, 0, 0, 0);
      if (valid) {
        const int marlin_tile = global_k_group * n_tiles + n_tile;
        const uint4* source =
            reinterpret_cast<const uint4*>(packed + 128 * marlin_tile);
        packed_value = source[input_chunk];
      }
#pragma unroll
      for (int marlin_warp = 0; marlin_warp < 4; ++marlin_warp) {
        packed_tile[packed_tile_index(input_n_subtile, input_k_group,
                                      input_column, input_pair, marlin_warp)] =
            component(packed_value, marlin_warp);
      }
    }
  }
  if constexpr (AsyncScatter) {
    asm volatile("cp.async.commit_group;");
  }
  if constexpr (!PrefetchScale || AsyncScatter) {
    asm volatile("cp.async.wait_group 0;");
  }
  __syncthreads();

  if constexpr (Paired) {
    static_assert(Threads == 256 || Threads == 512);
    static_assert(!PrefetchScale);
#pragma unroll
    for (int output_pass = 0; output_pass < 512 / Threads; ++output_pass) {
      const int lane = threadIdx.x & 31;
      const int row_pair_global =
          4 * (threadIdx.x >> 5) + (lane >> 3) + 32 * output_pass;
      const int row_pair = row_pair_global & 31;
      const int output_k_group = lane & 7;
      const int output_n_subtile = row_pair_global >> 5;
      const int warp = row_pair >> 3;
      const int column = row_pair & 7;
      const int global_n0 =
          128 * n_block + 64 * output_n_subtile + 16 * warp + column;
      const int global_n1 = global_n0 + 8;
      const int global_output_k_group = 8 * k_block + output_k_group;
      const int64_t output_offset0 =
          static_cast<int64_t>(global_n0) * scratch_k +
          16 * global_output_k_group;
      const int64_t output_offset1 =
          static_cast<int64_t>(global_n1) * scratch_k +
          16 * global_output_k_group;
      if (16 * global_output_k_group >= resident_k) {
        reinterpret_cast<uint4*>(output + output_offset0)[0] =
            make_uint4(0, 0, 0, 0);
        reinterpret_cast<uint4*>(output + output_offset1)[0] =
            make_uint4(0, 0, 0, 0);
      } else {
        const int row0_in_block = 64 * output_n_subtile + 16 * warp + column;
        const int row1_in_block = row0_in_block + 8;
        const int scale_lane0 = row0_in_block & 63;
        const int scale_lane1 = row1_in_block & 63;
        const int scale_transposed0 =
            8 * (scale_lane0 & 7) + (scale_lane0 >> 3);
        const int scale_transposed1 =
            8 * (scale_lane1 & 7) + (scale_lane1 >> 3);
        const int scale_reordered0 = (scale_transposed0 & ~3) |
                                     ((scale_transposed0 & 1) << 1) |
                                     ((scale_transposed0 & 2) >> 1);
        const int scale_reordered1 = (scale_transposed1 & ~3) |
                                     ((scale_transposed1 & 1) << 1) |
                                     ((scale_transposed1 & 2) >> 1);
        const int scale_physical0 = 64 * output_n_subtile + scale_reordered0;
        const int scale_physical1 = 64 * output_n_subtile + scale_reordered1;
        const int scale_storage_word =
            ScaleSwizzle
                ? (scale_physical0 >> 2) ^
                      (8 * (output_k_group >> 1) + (output_k_group & 1))
                : scale_physical0 >> 2;
        const uint32_t scale_word =
            scale_tile[output_k_group][scale_storage_word];
        const half raw_scale0 = __ushort_as_half(
            static_cast<uint16_t>((scale_word >> (8 * (scale_physical0 & 3))) &
                                  0xff)
            << 7);
        const half raw_scale1 = __ushort_as_half(
            static_cast<uint16_t>((scale_word >> (8 * (scale_physical1 & 3))) &
                                  0xff)
            << 7);
        const half scale0 = __float2half_rn(
            fminf(__half2float(raw_scale0) * tile_scale_reciprocal, 65504.0f));
        const half scale1 = __float2half_rn(
            fminf(__half2float(raw_scale1) * tile_scale_reciprocal, 65504.0f));
        const uint4 words =
            reinterpret_cast<const uint4*>(&packed_tile[packed_tile_index(
                output_n_subtile, output_k_group, column, 0, warp)])[0];
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
          marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
              static_cast<int>(word), values1);
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
    }
  } else {
#pragma unroll
    for (int output_pass = 0; output_pass < 1024 / Threads; ++output_pass) {
      const int output_segment = threadIdx.x + Threads * output_pass;
      const int row_in_block = output_segment >> 3;
      const int output_k_group = output_segment & 7;
      const int global_row = 128 * n_block + row_in_block;
      const int global_output_k_group = 8 * k_block + output_k_group;
      if (global_row >= n_padded) {
        continue;
      }
      const int64_t output_offset =
          static_cast<int64_t>(global_row) * scratch_k +
          16 * global_output_k_group;
      if (16 * global_output_k_group >= resident_k) {
        reinterpret_cast<uint4*>(output + output_offset)[0] =
            make_uint4(0, 0, 0, 0);
        continue;
      }

      uint8_t scale_byte;
      if constexpr (PrefetchScale) {
        scale_byte = prefetched_scales[output_pass];
      } else {
        const int scale_lane = row_in_block & 63;
        const int scale_transposed = 8 * (scale_lane & 7) + (scale_lane >> 3);
        const int scale_reordered = (scale_transposed & ~3) |
                                    ((scale_transposed & 1) << 1) |
                                    ((scale_transposed & 2) >> 1);
        const int scale_physical = 64 * (row_in_block >> 6) + scale_reordered;
        scale_byte = reinterpret_cast<uint8_t*>(
            scale_tile)[128 * output_k_group + scale_physical];
      }
      const half raw_scale =
          __ushort_as_half(static_cast<uint16_t>(scale_byte) << 7);
      const half scale = __float2half_rn(
          fminf(__half2float(raw_scale) * tile_scale_reciprocal, 65504.0f));
      const int row_in_subtile = row_in_block & 63;
      const int output_n_subtile = row_in_block >> 6;
      const int warp = row_in_subtile >> 4;
      const int column = row_in_subtile & 7;
      const bool upper_row = (row_in_subtile & 8) != 0;
      const uint4 words =
          reinterpret_cast<const uint4*>(&packed_tile[packed_tile_index(
              output_n_subtile, output_k_group, column, 0, warp)])[0];
      uint16_t first[4];
      uint16_t second[4];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const uint32_t word = component(words, pair);
        half2 values[2];
        marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
            static_cast<int>(upper_row ? word : word << 8), values);
        first[pair] = convert_pair(values[0], scale);
        second[pair] = convert_pair(values[1], scale);
      }
      const uint4 result = make_uint4(
          pack_pairs(first[0], first[1]), pack_pairs(first[2], first[3]),
          pack_pairs(second[0], second[1]), pack_pairs(second[2], second[3]));
      reinterpret_cast<uint4*>(output + output_offset)[0] = result;
    }
  }
}

__device__ __forceinline__ void stage_double_buffer_tile(
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
    const int marlin_tile =
        (8 * k_block + k_group) * n_tiles + 2 * n_block + n_subtile;
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

__device__ __forceinline__ void consume_double_buffer_tile(
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
    const int warp = row_pair >> 3;
    const int column = row_pair & 7;
    const int row0 = 64 * n_subtile + 16 * warp + column;
    const int global_n0 = 128 * n_block + row0;
    const int global_n1 = global_n0 + 8;
    const int global_k_group = 8 * k_block + k_group;
    const int64_t output_offset0 =
        static_cast<int64_t>(global_n0) * scratch_k + 16 * global_k_group;
    const int64_t output_offset1 =
        static_cast<int64_t>(global_n1) * scratch_k + 16 * global_k_group;
    const int scale_byte = warp & 1;
    const int logical_scale_word = 16 * n_subtile + 2 * column + (warp >> 1);
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
            n_subtile, k_group, column, 0, warp)])[0];
    uint32_t n0[4], n1[4];
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
      n0[pair_group] = convert_four(word0_n0[0], word1_n0[0], scale0);
      n0[pair_group + 2] = convert_four(word0_n0[1], word1_n0[1], scale0);
      n1[pair_group] = convert_four(word0_n1[0], word1_n1[0], scale1);
      n1[pair_group + 2] = convert_four(word0_n1[1], word1_n1[1], scale1);
    }
    reinterpret_cast<uint4*>(output + output_offset0)[0] =
        make_uint4(n0[0], n0[1], n0[2], n0[3]);
    reinterpret_cast<uint4*>(output + output_offset1)[0] =
        make_uint4(n1[0], n1[1], n1[2], n1[3]);
  }
}

__global__ __launch_bounds__(256) void double_buffer_kernel(
    uint8_t* output, float* output_scale, const int32_t* packed,
    const uint8_t* scales, const float* global_scale, const uint8_t* divisors,
    int n_padded, int resident_k, int scratch_k) {
  __shared__ uint32_t packed_tile[2][2 * 8 * 8 * 4 * 4];
  __shared__ uint32_t scale_tile[2][8][32];
  const int n_block = blockIdx.x;
  const int k_block0 = blockIdx.y;
  const int k_blocks = scratch_k / 128;
  const int k_block1 = k_block0 + k_blocks / 2;
  const int n_tiles = n_padded / 64;
  float reciprocal[2];
#pragma unroll
  for (int stage = 0; stage < 2; ++stage) {
    const int k_block = stage ? k_block1 : k_block0;
    const int scale_block = n_block * k_blocks + k_block;
    float reciprocal_lane = 0.0f;
    if ((threadIdx.x & 31) == 0) {
      const half divisor =
          __ushort_as_half(static_cast<uint16_t>(divisors[scale_block]) << 7);
      reciprocal_lane = __fdividef(1.0f, __half2float(divisor));
      if (threadIdx.x == 0) {
        output_scale[scale_block] =
            global_scale[0] * 0x1p-14f * __half2float(divisor);
      }
    }
    reciprocal[stage] = __shfl_sync(0xffffffffu, reciprocal_lane, 0);
  }
  stage_double_buffer_tile(packed_tile[0], scale_tile[0], packed, scales,
                           n_padded, n_tiles, n_block, k_block0);
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  stage_double_buffer_tile(packed_tile[1], scale_tile[1], packed, scales,
                           n_padded, n_tiles, n_block, k_block1);
  consume_double_buffer_tile(output, packed_tile[0], scale_tile[0],
                             reciprocal[0], n_block, k_block0, scratch_k);
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();
  consume_double_buffer_tile(output, packed_tile[1], scale_tile[1],
                             reciprocal[1], n_block, k_block1, scratch_k);
}

__global__ __launch_bounds__(512, 2) void super_tiled_kernel(
    uint8_t* output, float* output_scale, const int32_t* packed,
    const uint8_t* processed_scales, const float* processed_global_scale,
    const uint8_t* tile_scale_divisor_codes, int n_padded, int resident_k,
    int scratch_k) {
  __shared__ uint32_t packed_tile[4][2 * 8 * 8 * 4 * 4];
  __shared__ float tile_scale_reciprocal[4];

  const int n_blocks = n_padded / 128;
  const int k_blocks = scratch_k / 128;
  const int n_tiles = n_padded / 64;
  const int super_n_blocks = n_blocks / 2;
  const int super_n = blockIdx.x % super_n_blocks;
  const int super_k = blockIdx.x / super_n_blocks;

  if (threadIdx.x < 4) {
    const int sub = threadIdx.x;
    const int n_block = 2 * super_n + (sub & 1);
    const int k_block = 2 * super_k + (sub >> 1);
    const int scale_block = n_block * k_blocks + k_block;
    const half divisor = __ushort_as_half(
        static_cast<uint16_t>(tile_scale_divisor_codes[scale_block]) << 7);
    tile_scale_reciprocal[sub] = __frcp_rn(__half2float(divisor));
    output_scale[scale_block] =
        processed_global_scale[0] * 0x1p-14f * __half2float(divisor);
  }

#pragma unroll 1
  for (int input_pass = 0; input_pass < 4; ++input_pass) {
    const int input = threadIdx.x + 512 * input_pass;
    const int sub = input >> 9;
    const int local = input & 511;
    const int input_segment = local >> 5;
    const int input_chunk = local & 31;
    const int input_n_subtile = input_segment >> 3;
    const int input_k_group = input_segment & 7;
    const int n_block = 2 * super_n + (sub & 1);
    const int k_block = 2 * super_k + (sub >> 1);
    const int n_tile = 2 * n_block + input_n_subtile;
    const int global_k_group = 8 * k_block + input_k_group;
    uint4 packed_value = make_uint4(0, 0, 0, 0);
    if (n_tile < n_tiles && 16 * global_k_group < resident_k) {
      const int marlin_tile = global_k_group * n_tiles + n_tile;
      const uint4* source =
          reinterpret_cast<const uint4*>(packed + 128 * marlin_tile);
      packed_value = source[input_chunk];
    }
    const int input_column = input_chunk >> 2;
    const int input_pair = input_chunk & 3;
#pragma unroll
    for (int marlin_warp = 0; marlin_warp < 4; ++marlin_warp) {
      packed_tile[sub][packed_tile_index(
          input_n_subtile, input_k_group, input_column, input_pair,
          marlin_warp)] = component(packed_value, marlin_warp);
    }
  }
  __syncthreads();

#pragma unroll 1
  for (int output_pass = 0; output_pass < 8; ++output_pass) {
    const int output_segment = threadIdx.x + 512 * output_pass;
    const int sub = output_segment >> 10;
    const int local = output_segment & 1023;
    const int row_in_block = local >> 3;
    const int output_k_group = local & 7;
    const int n_block = 2 * super_n + (sub & 1);
    const int k_block = 2 * super_k + (sub >> 1);
    const int global_row = 128 * n_block + row_in_block;
    const int global_output_k_group = 8 * k_block + output_k_group;
    const int64_t output_offset = static_cast<int64_t>(global_row) * scratch_k +
                                  16 * global_output_k_group;
    if (16 * global_output_k_group >= resident_k) {
      reinterpret_cast<uint4*>(output + output_offset)[0] =
          make_uint4(0, 0, 0, 0);
      continue;
    }

    const uint8_t scale_byte = processed_scales[processed_scale_index(
        global_row, global_output_k_group, n_padded)];
    const half raw_scale =
        __ushort_as_half(static_cast<uint16_t>(scale_byte) << 7);
    const half scale = __float2half_rn(
        fminf(__half2float(raw_scale) * tile_scale_reciprocal[sub], 65504.0f));
    const int row_in_subtile = row_in_block & 63;
    const int output_n_subtile = row_in_block >> 6;
    const int warp = row_in_subtile >> 4;
    const int column = row_in_subtile & 7;
    const bool upper_row = (row_in_subtile & 8) != 0;
    uint16_t first[4];
    uint16_t second[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const uint32_t word = packed_tile[sub][packed_tile_index(
          output_n_subtile, output_k_group, column, pair, warp)];
      half2 values[2];
      marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
          static_cast<int>(upper_row ? word : word << 8), values);
      first[pair] = convert_pair(values[0], scale);
      second[pair] = convert_pair(values[1], scale);
    }
    reinterpret_cast<uint4*>(output + output_offset)[0] = make_uint4(
        pack_pairs(first[0], first[1]), pack_pairs(first[2], first[3]),
        pack_pairs(second[0], second[1]), pack_pairs(second[2], second[3]));
  }
}

__global__ __launch_bounds__(256) void async_transpose_kernel(
    uint8_t* __restrict__ output, float* __restrict__ output_scale,
    const int32_t* __restrict__ packed,
    const uint8_t* __restrict__ processed_scales,
    const float* __restrict__ processed_global_scale,
    const uint8_t* __restrict__ tile_scale_divisor_codes, int n_padded,
    int resident_k, int scratch_k) {
  __shared__ __align__(16) uint4 raw_tile[2][8][33];
  __shared__ uint32_t scale_tile[8][34];

  const int n_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int k_blocks = scratch_k / 128;
  const int n_tiles = n_padded / 64;
  const int scale_block = n_block * k_blocks + k_block;
  float reciprocal_lane = 0.0f;
  if ((threadIdx.x & 31) == 0) {
    const half divisor = __ushort_as_half(
        static_cast<uint16_t>(tile_scale_divisor_codes[scale_block]) << 7);
    reciprocal_lane = __frcp_rn(__half2float(divisor));
    if (threadIdx.x == 0) {
      output_scale[scale_block] =
          processed_global_scale[0] * 0x1p-14f * __half2float(divisor);
    }
  }
  const float reciprocal = __shfl_sync(0xffffffffu, reciprocal_lane, 0);

  const int producer_k_group = threadIdx.x >> 5;
  const int producer_chunk = threadIdx.x & 31;
  const int global_producer_k_group = 8 * k_block + producer_k_group;
  const int64_t scale_offset =
      static_cast<int64_t>(global_producer_k_group) * n_padded + 128 * n_block;
  copy_async(
      &scale_tile[producer_k_group][producer_chunk],
      reinterpret_cast<const uint32_t*>(processed_scales + scale_offset) +
          producer_chunk);

#pragma unroll
  for (int n_subtile = 0; n_subtile < 2; ++n_subtile) {
    const int n_tile = 2 * n_block + n_subtile;
    const int marlin_tile = global_producer_k_group * n_tiles + n_tile;
    const uint4* source =
        reinterpret_cast<const uint4*>(packed + 128 * marlin_tile);
    copy_async(&raw_tile[n_subtile][producer_k_group][producer_chunk],
               &source[producer_chunk]);
  }
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncthreads();

  const int column = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int k_group = lane & 7;
  const int marlin_warp = lane >> 3;
  const int global_k_group = 8 * k_block + k_group;
#pragma unroll
  for (int n_subtile = 0; n_subtile < 2; ++n_subtile) {
    const uint4 values = raw_tile[n_subtile][k_group][4 * column + marlin_warp];
    uint32_t r0 = values.x;
    uint32_t r1 = values.y;
    uint32_t r2 = values.z;
    uint32_t r3 = values.w;
    const uint32_t t0 = __shfl_xor_sync(0xffffffffu, r1, 8);
    const uint32_t t1 = __shfl_xor_sync(0xffffffffu, r0, 8);
    const uint32_t t2 = __shfl_xor_sync(0xffffffffu, r3, 8);
    const uint32_t t3 = __shfl_xor_sync(0xffffffffu, r2, 8);
    if ((marlin_warp & 1) != 0) {
      r0 = t0;
      r2 = t2;
    } else {
      r1 = t1;
      r3 = t3;
    }
    const uint32_t u0 = __shfl_xor_sync(0xffffffffu, r2, 16);
    const uint32_t u1 = __shfl_xor_sync(0xffffffffu, r3, 16);
    const uint32_t u2 = __shfl_xor_sync(0xffffffffu, r0, 16);
    const uint32_t u3 = __shfl_xor_sync(0xffffffffu, r1, 16);
    if ((marlin_warp & 2) != 0) {
      r0 = u0;
      r1 = u1;
    } else {
      r2 = u2;
      r3 = u3;
    }

    const int row0_in_block = 64 * n_subtile + 16 * marlin_warp + column;
    const int row1_in_block = row0_in_block + 8;
    const int scale_physical0 = processed_scale_index(row0_in_block, 0, 128);
    const int scale_physical1 = processed_scale_index(row1_in_block, 0, 128);
    const uint32_t scale_word = scale_tile[k_group][scale_physical0 >> 2];
    const half raw_scale0 =
        __ushort_as_half(static_cast<uint16_t>(
                             (scale_word >> (8 * (scale_physical0 & 3))) & 0xff)
                         << 7);
    const half raw_scale1 =
        __ushort_as_half(static_cast<uint16_t>(
                             (scale_word >> (8 * (scale_physical1 & 3))) & 0xff)
                         << 7);
    const half scale0 =
        __float2half_rn(fminf(__half2float(raw_scale0) * reciprocal, 65504.0f));
    const half scale1 =
        __float2half_rn(fminf(__half2float(raw_scale1) * reciprocal, 65504.0f));

    const uint32_t words[4] = {r0, r1, r2, r3};
    uint16_t n0_first[4];
    uint16_t n0_second[4];
    uint16_t n1_first[4];
    uint16_t n1_second[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      half2 values0[2];
      half2 values1[2];
      marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
          static_cast<int>(words[pair] << 8), values0);
      marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
          static_cast<int>(words[pair]), values1);
      n0_first[pair] = convert_pair(values0[0], scale0);
      n0_second[pair] = convert_pair(values0[1], scale0);
      n1_first[pair] = convert_pair(values1[0], scale1);
      n1_second[pair] = convert_pair(values1[1], scale1);
    }
    const int n0 = 128 * n_block + row0_in_block;
    const int n1 = n0 + 8;
    const int64_t output_offset0 =
        static_cast<int64_t>(n0) * scratch_k + 16 * global_k_group;
    const int64_t output_offset1 =
        static_cast<int64_t>(n1) * scratch_k + 16 * global_k_group;
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
}

__global__ __launch_bounds__(512) void warp_staged_kernel(
    uint8_t* __restrict__ output, float* __restrict__ output_scale,
    const int32_t* __restrict__ packed,
    const uint8_t* __restrict__ processed_scales,
    const float* __restrict__ processed_global_scale,
    const uint8_t* __restrict__ tile_scale_divisor_codes, int n_padded,
    int resident_k, int scratch_k) {
  __shared__ __align__(16) uint32_t packed_tile[16][128];

  const int n_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int k_blocks = scratch_k / 128;
  const int n_tiles = n_padded / 64;
  const int scale_block = n_block * k_blocks + k_block;
  const half divisor = __ushort_as_half(
      static_cast<uint16_t>(tile_scale_divisor_codes[scale_block]) << 7);
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float reciprocal_lane =
      lane == 0 ? __frcp_rn(__half2float(divisor)) : 0.0f;
  const float reciprocal = __shfl_sync(0xffffffffu, reciprocal_lane, 0);
  if (threadIdx.x == 0) {
    output_scale[scale_block] =
        processed_global_scale[0] * 0x1p-14f * __half2float(divisor);
  }

  const int n_subtile = warp >> 3;
  const int column = warp & 7;
  const int n_tile = 2 * n_block + n_subtile;
  const int load_k_group = lane >> 2;
  const int load_pair = lane & 3;
  const int global_load_k_group = 8 * k_block + load_k_group;
  const int marlin_tile = global_load_k_group * n_tiles + n_tile;
  const uint4* source = reinterpret_cast<const uint4*>(
      packed + 128 * marlin_tile + 16 * column + 4 * load_pair);
  copy_async(reinterpret_cast<uint4*>(
                 &packed_tile[warp][4 * (8 * load_pair + load_k_group)]),
             source);
  asm volatile("cp.async.commit_group;");
  asm volatile("cp.async.wait_group 0;");
  __syncwarp();

  const int marlin_warp = lane >> 3;
  const int k_group = lane & 7;
  const int global_k_group = 8 * k_block + k_group;
  const int global_n0 =
      128 * n_block + 64 * n_subtile + 16 * marlin_warp + column;
  const int global_n1 = global_n0 + 8;
  const int64_t output_offset0 =
      static_cast<int64_t>(global_n0) * scratch_k + 16 * global_k_group;
  const int64_t output_offset1 =
      static_cast<int64_t>(global_n1) * scratch_k + 16 * global_k_group;

  uint64_t scale_chunk_lane = 0;
  if (marlin_warp == 0) {
    const int64_t scale_chunk_offset =
        static_cast<int64_t>(global_k_group) * n_padded + 64 * n_tile +
        8 * column;
    scale_chunk_lane = *reinterpret_cast<const uint64_t*>(processed_scales +
                                                          scale_chunk_offset);
  }
  const uint64_t scale_chunk =
      __shfl_sync(0xffffffffu, static_cast<uint32_t>(scale_chunk_lane),
                  k_group) |
      (static_cast<uint64_t>(__shfl_sync(
           0xffffffffu, static_cast<uint32_t>(scale_chunk_lane >> 32), k_group))
       << 32);
  const int scale_slot0 = 2 * (marlin_warp & 2) + (marlin_warp & 1);
  const uint8_t scale_byte0 = scale_chunk >> (8 * scale_slot0);
  const uint8_t scale_byte1 = scale_chunk >> (8 * (scale_slot0 + 2));
  const half raw_scale0 =
      __ushort_as_half(static_cast<uint16_t>(scale_byte0) << 7);
  const half raw_scale1 =
      __ushort_as_half(static_cast<uint16_t>(scale_byte1) << 7);
  const half scale0 =
      __float2half_rn(fminf(__half2float(raw_scale0) * reciprocal, 65504.0f));
  const half scale1 =
      __float2half_rn(fminf(__half2float(raw_scale1) * reciprocal, 65504.0f));

  uint16_t n0_first[4];
  uint16_t n0_second[4];
  uint16_t n1_first[4];
  uint16_t n1_second[4];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    const uint32_t word =
        packed_tile[warp][4 * (8 * pair + k_group) + marlin_warp];
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

__global__ __launch_bounds__(256) void warp_kernel(
    uint8_t* output, float* output_scale, const int32_t* packed,
    const uint8_t* processed_scales, const float* processed_global_scale,
    const uint8_t* tile_scale_divisor_codes, int n_padded, int resident_k,
    int scratch_k) {
  const int n_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int n_blocks = (n_padded + 127) / 128;
  const int k_blocks = scratch_k / 128;
  const int n_tiles = n_padded / 64;
  const int scale_block = n_block * k_blocks + k_block;
  const uint8_t divisor_code = tile_scale_divisor_codes[scale_block];
  const half divisor =
      __ushort_as_half(static_cast<uint16_t>(divisor_code) << 7);
  const float reciprocal_lane =
      (threadIdx.x & 31) == 0 ? __frcp_rn(__half2float(divisor)) : 0.0f;
  const float tile_scale_reciprocal =
      __shfl_sync(0xffffffffu, reciprocal_lane, 0);

  if (threadIdx.x == 0) {
    output_scale[scale_block] =
        processed_global_scale[0] * 0x1p-14f * __half2float(divisor);
  }

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n_subtile = warp_id >> 2;
  const int column_pair = warp_id & 3;
  const int n_tile = 2 * n_block + n_subtile;
  if (n_tile >= n_tiles) {
    return;
  }

  const int column_in_pair = lane >> 4;
  const int pair = (lane >> 2) & 3;
  const int marlin_warp = lane & 3;
  const int column = 2 * column_pair + column_in_pair;
  const int n0_in_subtile = 16 * marlin_warp + column;
  const int n1_in_subtile = n0_in_subtile + 8;
  const int global_n0 = 64 * n_tile + n0_in_subtile;
  const int global_n1 = 64 * n_tile + n1_in_subtile;

#pragma unroll 1
  for (int k_group = 0; k_group < 8; ++k_group) {
    const int global_k_group = 8 * k_block + k_group;
    if (16 * global_k_group >= resident_k) {
      if (pair < 2) {
        const int global_row = pair == 0 ? global_n0 : global_n1;
        const int64_t output_offset =
            static_cast<int64_t>(global_row) * scratch_k + 16 * global_k_group;
        reinterpret_cast<uint4*>(output + output_offset)[0] =
            make_uint4(0, 0, 0, 0);
      }
      continue;
    }

    half scale0 = __float2half(0.0f);
    half scale1 = __float2half(0.0f);
    if (pair == 0) {
      const uint8_t scale_byte0 = processed_scales[processed_scale_index(
          global_n0, global_k_group, n_padded)];
      const uint8_t scale_byte1 = processed_scales[processed_scale_index(
          global_n1, global_k_group, n_padded)];
      const half raw_scale0 =
          __ushort_as_half(static_cast<uint16_t>(scale_byte0) << 7);
      const half raw_scale1 =
          __ushort_as_half(static_cast<uint16_t>(scale_byte1) << 7);
      scale0 = __float2half_rn(
          fminf(__half2float(raw_scale0) * tile_scale_reciprocal, 65504.0f));
      scale1 = __float2half_rn(
          fminf(__half2float(raw_scale1) * tile_scale_reciprocal, 65504.0f));
    }
    const int scale_source_lane = (column_in_pair << 4) | marlin_warp;
    const uint32_t scale0_bits = __shfl_sync(
        0xffffffffu, static_cast<uint32_t>(__half_as_ushort(scale0)),
        scale_source_lane);
    const uint32_t scale1_bits = __shfl_sync(
        0xffffffffu, static_cast<uint32_t>(__half_as_ushort(scale1)),
        scale_source_lane);
    scale0 = __ushort_as_half(static_cast<uint16_t>(scale0_bits));
    scale1 = __ushort_as_half(static_cast<uint16_t>(scale1_bits));

    const int marlin_tile = global_k_group * n_tiles + n_tile;
    const int word_in_tile = 32 * column_pair + lane;
    const uint32_t word =
        static_cast<uint32_t>(packed[128 * marlin_tile + word_in_tile]);
    half2 values0[2];
    half2 values1[2];
    marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
        static_cast<int>(word << 8), values0);
    marlin::dequant<half2, vllm::kFE2M1f.id(), false>(static_cast<int>(word),
                                                      values1);
    const uint16_t n0_first = convert_pair(values0[0], scale0);
    const uint16_t n0_second = convert_pair(values0[1], scale0);
    const uint16_t n1_first = convert_pair(values1[0], scale1);
    const uint16_t n1_second = convert_pair(values1[1], scale1);

    const uint32_t first_rows = pack_pairs(n0_first, n1_first);
    const uint32_t second_rows = pack_pairs(n0_second, n1_second);
    uint32_t first[4];
    uint32_t second[4];
#pragma unroll
    for (int source_pair = 0; source_pair < 4; ++source_pair) {
      const int source_lane =
          (column_in_pair << 4) | (source_pair << 2) | marlin_warp;
      first[source_pair] = __shfl_sync(0xffffffffu, first_rows, source_lane);
      second[source_pair] = __shfl_sync(0xffffffffu, second_rows, source_lane);
    }
    if (pair < 2) {
      const int shift = 16 * pair;
      const uint4 result =
          make_uint4(pack_pairs(first[0] >> shift, first[1] >> shift),
                     pack_pairs(first[2] >> shift, first[3] >> shift),
                     pack_pairs(second[0] >> shift, second[1] >> shift),
                     pack_pairs(second[2] >> shift, second[3] >> shift));
      const bool upper_row = pair == 1;
      const int global_row = upper_row ? global_n1 : global_n0;
      const int64_t output_offset =
          static_cast<int64_t>(global_row) * scratch_k + 16 * global_k_group;
      reinterpret_cast<uint4*>(output + output_offset)[0] = result;
    }
  }
}

__global__ __launch_bounds__(128) void vector_output_kernel(
    uint8_t* __restrict__ output, float* __restrict__ output_scale,
    const int32_t* __restrict__ packed,
    const uint8_t* __restrict__ processed_scales,
    const float* __restrict__ processed_global_scale,
    const uint8_t* __restrict__ tile_scale_divisor_codes, int n_padded,
    int resident_k, int scratch_k) {
  const int n_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int k_blocks = scratch_k / 128;
  const int n_tiles = n_padded / 64;
  const int scale_block = n_block * k_blocks + k_block;
  const half divisor = __ushort_as_half(
      static_cast<uint16_t>(tile_scale_divisor_codes[scale_block]) << 7);
  const float reciprocal_lane =
      (threadIdx.x & 31) == 0 ? __frcp_rn(__half2float(divisor)) : 0.0f;
  const float reciprocal = __shfl_sync(0xffffffffu, reciprocal_lane, 0);
  if (threadIdx.x == 0) {
    output_scale[scale_block] =
        processed_global_scale[0] * 0x1p-14f * __half2float(divisor);
  }

  const int warp_id = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int n_subtile = warp_id >> 1;
  const int column = 4 * (warp_id & 1) + (lane >> 3);
  const int k_group = lane & 7;
  const int n_tile = 2 * n_block + n_subtile;
  const int global_k_group = 8 * k_block + k_group;
  if (16 * global_k_group >= resident_k) {
#pragma unroll
    for (int marlin_warp = 0; marlin_warp < 4; ++marlin_warp) {
      const int global_n0 = 64 * n_tile + 16 * marlin_warp + column;
      const int64_t output_offset0 =
          static_cast<int64_t>(global_n0) * scratch_k + 16 * global_k_group;
      reinterpret_cast<uint4*>(output + output_offset0)[0] =
          make_uint4(0, 0, 0, 0);
      reinterpret_cast<uint4*>(output + output_offset0 + 8 * scratch_k)[0] =
          make_uint4(0, 0, 0, 0);
    }
    return;
  }

  const int marlin_tile = global_k_group * n_tiles + n_tile;
  const int32_t* words = packed + 128 * marlin_tile + 16 * column;
  uint4 packed_pairs[4];
#pragma unroll
  for (int pair = 0; pair < 4; ++pair) {
    packed_pairs[pair] = reinterpret_cast<const uint4*>(words + 4 * pair)[0];
  }
  const int64_t scale_chunk_offset =
      static_cast<int64_t>(global_k_group) * n_padded + 64 * n_tile +
      8 * column;
  const uint64_t scale_chunk =
      *reinterpret_cast<const uint64_t*>(processed_scales + scale_chunk_offset);

#pragma unroll
  for (int marlin_warp = 0; marlin_warp < 4; ++marlin_warp) {
    const int global_n0 = 64 * n_tile + 16 * marlin_warp + column;
    const int global_n1 = global_n0 + 8;
    const int scale_slot0 = 2 * (marlin_warp & 2) + (marlin_warp & 1);
    const uint8_t scale_byte0 = scale_chunk >> (8 * scale_slot0);
    const uint8_t scale_byte1 = scale_chunk >> (8 * (scale_slot0 + 2));
    const half raw_scale0 =
        __ushort_as_half(static_cast<uint16_t>(scale_byte0) << 7);
    const half raw_scale1 =
        __ushort_as_half(static_cast<uint16_t>(scale_byte1) << 7);
    const half scale0 =
        __float2half_rn(fminf(__half2float(raw_scale0) * reciprocal, 65504.0f));
    const half scale1 =
        __float2half_rn(fminf(__half2float(raw_scale1) * reciprocal, 65504.0f));
    uint16_t n0_first[4];
    uint16_t n0_second[4];
    uint16_t n1_first[4];
    uint16_t n1_second[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const uint32_t word = component(packed_pairs[pair], marlin_warp);
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
        static_cast<int64_t>(global_n0) * scratch_k + 16 * global_k_group;
    const int64_t output_offset1 =
        static_cast<int64_t>(global_n1) * scratch_k + 16 * global_k_group;
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
}

template <int Threads>
__global__ __launch_bounds__(Threads) void output_owned_kernel(
    uint8_t* output, float* output_scale, const int32_t* packed,
    const uint8_t* processed_scales, const float* processed_global_scale,
    const uint8_t* tile_scale_divisor_codes, int n_padded, int resident_k,
    int scratch_k) {
  const int n_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int k_blocks = scratch_k / 128;
  const int n_tiles = n_padded / 64;
  const int scale_block = n_block * k_blocks + k_block;
  const half divisor = __ushort_as_half(
      static_cast<uint16_t>(tile_scale_divisor_codes[scale_block]) << 7);
  const float reciprocal_lane =
      (threadIdx.x & 31) == 0 ? __frcp_rn(__half2float(divisor)) : 0.0f;
  const float tile_scale_reciprocal =
      __shfl_sync(0xffffffffu, reciprocal_lane, 0);
  if (threadIdx.x == 0) {
    output_scale[scale_block] =
        processed_global_scale[0] * 0x1p-14f * __half2float(divisor);
  }

#pragma unroll
  for (int pass = 0; pass < 1024 / Threads; ++pass) {
    const int output_segment = threadIdx.x + Threads * pass;
    const int row_in_block = output_segment >> 3;
    const int k_group = output_segment & 7;
    const int global_row = 128 * n_block + row_in_block;
    const int global_k_group = 8 * k_block + k_group;
    if (global_row >= n_padded) {
      continue;
    }
    const int64_t output_offset =
        static_cast<int64_t>(global_row) * scratch_k + 16 * global_k_group;
    if (16 * global_k_group >= resident_k) {
      reinterpret_cast<uint4*>(output + output_offset)[0] =
          make_uint4(0, 0, 0, 0);
      continue;
    }

    const uint8_t scale_byte = processed_scales[processed_scale_index(
        global_row, global_k_group, n_padded)];
    const half raw_scale =
        __ushort_as_half(static_cast<uint16_t>(scale_byte) << 7);
    const half scale = __float2half_rn(
        fminf(__half2float(raw_scale) * tile_scale_reciprocal, 65504.0f));
    const int n_subtile = row_in_block >> 6;
    const int row_in_subtile = row_in_block & 63;
    const int marlin_warp = row_in_subtile >> 4;
    const int column = row_in_subtile & 7;
    const bool upper_row = (row_in_subtile & 8) != 0;
    const int n_tile = 2 * n_block + n_subtile;
    const int marlin_tile = global_k_group * n_tiles + n_tile;
    const int32_t* words = packed + 128 * marlin_tile;
    uint16_t first[4];
    uint16_t second[4];
#pragma unroll
    for (int pair = 0; pair < 4; ++pair) {
      const uint32_t word =
          static_cast<uint32_t>(words[16 * column + 4 * pair + marlin_warp]);
      half2 values[2];
      marlin::dequant<half2, vllm::kFE2M1f.id(), false>(
          static_cast<int>(upper_row ? word : word << 8), values);
      first[pair] = convert_pair(values[0], scale);
      second[pair] = convert_pair(values[1], scale);
    }
    reinterpret_cast<uint4*>(output + output_offset)[0] = make_uint4(
        pack_pairs(first[0], first[1]), pack_pairs(first[2], first[3]),
        pack_pairs(second[0], second[1]), pack_pairs(second[2], second[3]));
  }
}

template <int Threads, bool FuseBias = false, bool FusedPack = false,
          bool DirectDecode = false, bool Prefetch = false>
__global__ __launch_bounds__(Threads) void paired_output_kernel(
    uint8_t* __restrict__ output, float* __restrict__ output_scale,
    const int32_t* __restrict__ packed,
    const uint8_t* __restrict__ processed_scales,
    const float* __restrict__ processed_global_scale,
    const uint8_t* __restrict__ tile_scale_divisor_codes, int n_padded,
    int resident_k, int scratch_k) {
  const int n_block = blockIdx.x;
  const int k_block = blockIdx.y;
  const int k_blocks = scratch_k / 128;
  const int n_tiles = n_padded / 64;
  const int scale_block = n_block * k_blocks + k_block;
  const half divisor = __ushort_as_half(
      static_cast<uint16_t>(tile_scale_divisor_codes[scale_block]) << 7);
  const float reciprocal_lane =
      (threadIdx.x & 31) == 0 ? __frcp_rn(__half2float(divisor)) : 0.0f;
  const float tile_scale_reciprocal =
      __shfl_sync(0xffffffffu, reciprocal_lane, 0);
  if (threadIdx.x == 0) {
    output_scale[scale_block] =
        processed_global_scale[0] * 0x1p-14f * __half2float(divisor);
  }

#pragma unroll
  for (int pass = 0; pass < 512 / Threads; ++pass) {
    const int unit = threadIdx.x + Threads * pass;
    const int lane_in_unit = unit & 31;
    const int row_pair_global = 4 * (unit >> 5) + (lane_in_unit >> 3);
    const int row_pair = row_pair_global & 31;
    const int k_group = lane_in_unit & 7;
    const int n_subtile = row_pair_global >> 5;
    const int marlin_warp = row_pair >> 3;
    const int column = row_pair & 7;
    const int n_tile = 2 * n_block + n_subtile;
    if (n_tile >= n_tiles) {
      continue;
    }
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
      continue;
    }

    const uint8_t scale_byte0 = processed_scales[processed_scale_index(
        global_n0, global_k_group, n_padded)];
    const uint8_t scale_byte1 = processed_scales[processed_scale_index(
        global_n1, global_k_group, n_padded)];
    const half raw_scale0 =
        __ushort_as_half(static_cast<uint16_t>(scale_byte0) << 7);
    const half raw_scale1 =
        __ushort_as_half(static_cast<uint16_t>(scale_byte1) << 7);
    half scale0 = __float2half_rn(
        fminf(__half2float(raw_scale0) * tile_scale_reciprocal, 65504.0f));
    half scale1 = __float2half_rn(
        fminf(__half2float(raw_scale1) * tile_scale_reciprocal, 65504.0f));
    if constexpr (FuseBias) {
      constexpr float fp4_bias = 16384.0f;
      scale0 = __hmul(scale0, __float2half(fp4_bias));
      scale1 = __hmul(scale1, __float2half(fp4_bias));
    }
    const int marlin_tile = global_k_group * n_tiles + n_tile;
    const int32_t* words = packed + 128 * marlin_tile;
    uint4 result0;
    uint4 result1;
    if constexpr (FusedPack) {
      uint32_t n0[4];
      uint32_t n1[4];
#pragma unroll
      for (int pair_group = 0; pair_group < 2; ++pair_group) {
        const int pair = 2 * pair_group;
        const uint32_t word0 =
            load_packed<Prefetch>(words + 16 * column + 4 * pair + marlin_warp);
        const uint32_t word1 = load_packed<Prefetch>(
            words + 16 * column + 4 * (pair + 1) + marlin_warp);
        half2 word0_n0[2];
        half2 word0_n1[2];
        half2 word1_n0[2];
        half2 word1_n1[2];
        decode_e2m1<DirectDecode, FuseBias>(static_cast<int>(word0 << 8),
                                            word0_n0);
        decode_e2m1<DirectDecode, FuseBias>(static_cast<int>(word0), word0_n1);
        decode_e2m1<DirectDecode, FuseBias>(static_cast<int>(word1 << 8),
                                            word1_n0);
        decode_e2m1<DirectDecode, FuseBias>(static_cast<int>(word1), word1_n1);
        n0[pair_group] = convert_four(word0_n0[0], word1_n0[0], scale0);
        n0[pair_group + 2] = convert_four(word0_n0[1], word1_n0[1], scale0);
        n1[pair_group] = convert_four(word0_n1[0], word1_n1[0], scale1);
        n1[pair_group + 2] = convert_four(word0_n1[1], word1_n1[1], scale1);
      }
      result0 = make_uint4(n0[0], n0[1], n0[2], n0[3]);
      result1 = make_uint4(n1[0], n1[1], n1[2], n1[3]);
    } else {
      uint16_t n0_first[4];
      uint16_t n0_second[4];
      uint16_t n1_first[4];
      uint16_t n1_second[4];
#pragma unroll
      for (int pair = 0; pair < 4; ++pair) {
        const uint32_t word =
            load_packed<Prefetch>(words + 16 * column + 4 * pair + marlin_warp);
        half2 values0[2];
        half2 values1[2];
        decode_e2m1<DirectDecode, FuseBias>(static_cast<int>(word << 8),
                                            values0);
        decode_e2m1<DirectDecode, FuseBias>(static_cast<int>(word), values1);
        n0_first[pair] = convert_pair(values0[0], scale0);
        n0_second[pair] = convert_pair(values0[1], scale0);
        n1_first[pair] = convert_pair(values1[0], scale1);
        n1_second[pair] = convert_pair(values1[1], scale1);
      }
      result0 = make_uint4(pack_pairs(n0_first[0], n0_first[1]),
                           pack_pairs(n0_first[2], n0_first[3]),
                           pack_pairs(n0_second[0], n0_second[1]),
                           pack_pairs(n0_second[2], n0_second[3]));
      result1 = make_uint4(pack_pairs(n1_first[0], n1_first[1]),
                           pack_pairs(n1_first[2], n1_first[3]),
                           pack_pairs(n1_second[0], n1_second[1]),
                           pack_pairs(n1_second[2], n1_second[3]));
    }
    reinterpret_cast<uint4*>(output + output_offset0)[0] = result0;
    reinterpret_cast<uint4*>(output + output_offset1)[0] = result1;
  }
}

template <typename Launch>
float benchmark(Launch launch, int warmups, int iterations) {
  for (int i = 0; i < warmups; ++i) {
    launch();
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin;
  cudaEvent_t end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(begin));
  for (int i = 0; i < iterations; ++i) {
    launch();
  }
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  return milliseconds * 1000.0f / iterations;
}

template <typename Launch>
float benchmark_cold(Launch launch, int source_count,
                     std::vector<float>& raw_us) {
  for (int source = 0; source < source_count; ++source) {
    launch(source);
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  cudaEvent_t begin;
  cudaEvent_t end;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  int source = 0;
  for (int run = 0; run < 7; ++run) {
    CUDA_CHECK(cudaEventRecord(begin));
    for (int iteration = 0; iteration < 64; ++iteration) {
      launch(source);
      source = (source + 1) % source_count;
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float milliseconds = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
    raw_us.push_back(milliseconds * 1000.0f / 64);
  }
  CUDA_CHECK(cudaEventDestroy(begin));
  CUDA_CHECK(cudaEventDestroy(end));
  std::vector<float> sorted = raw_us;
  std::sort(sorted.begin(), sorted.end());
  return sorted[sorted.size() / 2];
}

}  // namespace

int main() {
  constexpr int n = 4096;
  constexpr int resident_k = 4096;
  constexpr int scratch_k = 4096;
  constexpr size_t packed_words = static_cast<size_t>(resident_k / 16) * 2 * n;
  constexpr size_t scale_bytes = static_cast<size_t>(resident_k / 16) * n;
  constexpr size_t output_bytes = static_cast<size_t>(n) * scratch_k;
  constexpr size_t tile_count =
      static_cast<size_t>(n / 128) * (scratch_k / 128);
  int l2_bytes = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0));
  constexpr size_t source_bytes = packed_words * sizeof(int32_t) + scale_bytes;
  const int source_count =
      std::max<int>(2, (4LL * l2_bytes + source_bytes - 1) / source_bytes);

  std::mt19937 generator(11);
  std::vector<int32_t> host_packed(packed_words);
  for (int32_t& value : host_packed) {
    value = static_cast<int32_t>(generator());
  }
  std::uniform_int_distribution<int> scale_distribution(0, 0x7e);
  std::uniform_int_distribution<int> divisor_distribution(1, 0x7e);
  std::vector<uint8_t> host_scales(scale_bytes);
  std::vector<uint8_t> host_divisors(tile_count);
  std::generate(host_scales.begin(), host_scales.end(),
                [&] { return scale_distribution(generator); });
  std::generate(host_divisors.begin(), host_divisors.end(),
                [&] { return divisor_distribution(generator); });
  const float host_global_scale = 1.0f;

  int32_t* packed = nullptr;
  uint8_t* scales = nullptr;
  uint8_t* divisors = nullptr;
  float* global_scale = nullptr;
  float* reference_scales = nullptr;
  float* candidate_scales = nullptr;
  uint8_t* reference = nullptr;
  uint8_t* candidate = nullptr;
  CUDA_CHECK(
      cudaMalloc(&packed, source_count * packed_words * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&scales, source_count * scale_bytes));
  CUDA_CHECK(cudaMalloc(&divisors, tile_count));
  CUDA_CHECK(cudaMalloc(&global_scale, sizeof(float)));
  CUDA_CHECK(cudaMalloc(&reference_scales, tile_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&candidate_scales, tile_count * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&reference, output_bytes));
  CUDA_CHECK(cudaMalloc(&candidate, output_bytes));
  for (int source = 0; source < source_count; ++source) {
    CUDA_CHECK(cudaMemcpy(packed + source * packed_words, host_packed.data(),
                          packed_words * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(scales + source * scale_bytes, host_scales.data(),
                          scale_bytes, cudaMemcpyHostToDevice));
  }
  CUDA_CHECK(cudaMemcpy(divisors, host_divisors.data(), tile_count,
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(global_scale, &host_global_scale, sizeof(float),
                        cudaMemcpyHostToDevice));

  const auto launch_reference = [&] {
    reference_kernel<<<8192, 256>>>(reference, reference_scales, packed, scales,
                                    global_scale, divisors, n, resident_k,
                                    scratch_k);
  };
  const auto launch_candidate_128 = [&] {
    output_owned_kernel<128><<<dim3(32, 32), 128>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_candidate = [&] {
    output_owned_kernel<256><<<dim3(32, 32), 256>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_candidate_512 = [&] {
    output_owned_kernel<512><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_candidate_1024 = [&] {
    output_owned_kernel<1024><<<dim3(32, 32), 1024>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_shared_tile_256 = [&] {
    tiled_kernel<256><<<dim3(32, 32), 256>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_shared_tile = [&] {
    tiled_kernel<512><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_paired_tile = [&] {
    tiled_kernel<512, true><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_paired_tile_256 = [&] {
    tiled_kernel<256, true><<<dim3(32, 32), 256>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_async_scatter = [&] {
    tiled_kernel<256, true, false, true><<<dim3(32, 32), 256>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_double_buffer = [&] {
    double_buffer_kernel<<<dim3(32, 16), 256>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_scale_swizzle = [&] {
    tiled_kernel<256, true, false, true, true><<<dim3(32, 32), 256>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_prefetched_scale_tile = [&] {
    tiled_kernel<512, false, true><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_super_tile = [&] {
    super_tiled_kernel<<<256, 512>>>(candidate, candidate_scales, packed,
                                     scales, global_scale, divisors, n,
                                     resident_k, scratch_k);
  };
  const auto launch_async_transpose = [&] {
    async_transpose_kernel<<<dim3(32, 32), 256>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_warp = [&] {
    warp_kernel<<<dim3(32, 32), 256>>>(candidate, candidate_scales, packed,
                                       scales, global_scale, divisors, n,
                                       resident_k, scratch_k);
  };
  const auto launch_vector = [&] {
    vector_output_kernel<<<dim3(32, 32), 128>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_paired_64 = [&] {
    paired_output_kernel<64><<<dim3(32, 32), 64>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_paired_128 = [&] {
    paired_output_kernel<128><<<dim3(32, 32), 128>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_paired_256 = [&] {
    paired_output_kernel<256><<<dim3(32, 32), 256>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_paired_512 = [&] {
    paired_output_kernel<512><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_paired_fused_512 = [&] {
    paired_output_kernel<512, true><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_paired_packed_512 = [&] {
    paired_output_kernel<512, false, true><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_paired_direct_decode_512 = [&] {
    paired_output_kernel<512, false, false, true><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };
  const auto launch_paired_prefetch_512 = [&] {
    paired_output_kernel<512, false, false, false, true><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed, scales, global_scale, divisors, n,
        resident_k, scratch_k);
  };

  launch_reference();
  launch_double_buffer();
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<uint8_t> host_reference(output_bytes);
  std::vector<uint8_t> host_candidate(output_bytes);
  CUDA_CHECK(cudaMemcpy(host_reference.data(), reference, output_bytes,
                        cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_candidate.data(), candidate, output_bytes,
                        cudaMemcpyDeviceToHost));
  size_t mismatches = 0;
  size_t first_mismatch = output_bytes;
  for (size_t index = 0; index < output_bytes; ++index) {
    if (host_reference[index] != host_candidate[index]) {
      if (first_mismatch == output_bytes) {
        first_mismatch = index;
      }
      ++mismatches;
    }
  }
  std::vector<float> host_reference_scales(tile_count);
  std::vector<float> host_candidate_scales(tile_count);
  CUDA_CHECK(cudaMemcpy(host_reference_scales.data(), reference_scales,
                        tile_count * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(host_candidate_scales.data(), candidate_scales,
                        tile_count * sizeof(float), cudaMemcpyDeviceToHost));
  size_t scale_mismatches = 0;
  for (size_t index = 0; index < tile_count; ++index) {
    scale_mismatches +=
        host_reference_scales[index] != host_candidate_scales[index];
  }
  std::printf(
      "gate=double_buffer correct=%s mismatches=%zu first=%zu "
      "scale_mismatches=%zu\n",
      mismatches == 0 && scale_mismatches == 0 ? "true" : "false", mismatches,
      first_mismatch, scale_mismatches);
  bool all_correct = mismatches == 0 && scale_mismatches == 0;
  const auto gate = [&](const char* name, auto launch) {
    launch();
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(host_candidate.data(), candidate, output_bytes,
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(host_candidate_scales.data(), candidate_scales,
                          tile_count * sizeof(float), cudaMemcpyDeviceToHost));
    size_t byte_errors = 0;
    size_t scale_errors = 0;
    for (size_t index = 0; index < output_bytes; ++index) {
      byte_errors += host_reference[index] != host_candidate[index];
    }
    for (size_t index = 0; index < tile_count; ++index) {
      scale_errors +=
          host_reference_scales[index] != host_candidate_scales[index];
    }
    const bool correct = byte_errors == 0 && scale_errors == 0;
    all_correct &= correct;
    std::printf("gate=%s correct=%s mismatches=%zu scale_mismatches=%zu\n",
                name, correct ? "true" : "false", byte_errors, scale_errors);
  };
  gate("sync_output512", launch_shared_tile);
  gate("sync_paired256", launch_paired_tile_256);
  gate("async_scatter", launch_async_scatter);
  gate("scale_prefetch", launch_prefetched_scale_tile);

  const float reference_us = benchmark(launch_reference, 100, 500);
  const float candidate_128_us = benchmark(launch_candidate_128, 100, 500);
  const float candidate_us = benchmark(launch_candidate, 100, 500);
  const float candidate_512_us = benchmark(launch_candidate_512, 100, 500);
  const float candidate_1024_us = benchmark(launch_candidate_1024, 100, 500);
  const float shared_tile_us = benchmark(launch_shared_tile, 100, 500);
  const float shared_tile_256_us = benchmark(launch_shared_tile_256, 100, 500);
  const float super_tile_us = benchmark(launch_super_tile, 100, 500);
  const float paired_tile_us = benchmark(launch_paired_tile, 100, 500);
  const float paired_tile_256_us = benchmark(launch_paired_tile_256, 100, 500);
  const float async_scatter_us = benchmark(launch_async_scatter, 100, 500);
  const float double_buffer_us = benchmark(launch_double_buffer, 100, 500);
  const float scale_swizzle_us = benchmark(launch_scale_swizzle, 100, 500);
  const float prefetched_scale_tile_us =
      benchmark(launch_prefetched_scale_tile, 100, 500);
  const float async_transpose_us = benchmark(launch_async_transpose, 100, 500);
  const float warp_us = benchmark(launch_warp, 100, 500);
  const float vector_us = benchmark(launch_vector, 100, 500);
  const float paired_64_us = benchmark(launch_paired_64, 100, 500);
  const float paired_128_us = benchmark(launch_paired_128, 100, 500);
  const float paired_256_us = benchmark(launch_paired_256, 100, 500);
  const float paired_512_us = benchmark(launch_paired_512, 100, 500);
  const float paired_fused_512_us =
      benchmark(launch_paired_fused_512, 100, 500);
  const float paired_packed_512_us =
      benchmark(launch_paired_packed_512, 100, 500);
  const float paired_direct_decode_512_us =
      benchmark(launch_paired_direct_decode_512, 100, 500);
  const float paired_prefetch_512_us =
      benchmark(launch_paired_prefetch_512, 100, 500);
  std::printf(
      "reference_us=%.3f output128_us=%.3f output256_us=%.3f "
      "output512_us=%.3f output1024_us=%.3f shared256_us=%.3f "
      "shared512_us=%.3f scale_prefetch_us=%.3f paired_tile_us=%.3f "
      "paired_tile256_us=%.3f "
      "async_scatter_us=%.3f double_buffer_us=%.3f scale_swizzle_us=%.3f "
      "super256x256_us=%.3f "
      "warp_us=%.3f "
      "async_transpose_us=%.3f vector_us=%.3f "
      "paired64_us=%.3f paired128_us=%.3f paired256_us=%.3f "
      "paired512_us=%.3f "
      "paired_fused512_us=%.3f paired_packed512_us=%.3f "
      "paired_direct_decode512_us=%.3f "
      "paired_prefetch512_us=%.3f "
      "best_speedup=%.4fx\n",
      reference_us, candidate_128_us, candidate_us, candidate_512_us,
      candidate_1024_us, shared_tile_256_us, shared_tile_us,
      prefetched_scale_tile_us, paired_tile_us, paired_tile_256_us,
      async_scatter_us, double_buffer_us, scale_swizzle_us, super_tile_us,
      warp_us, async_transpose_us, vector_us, paired_64_us, paired_128_us,
      paired_256_us, paired_512_us, paired_fused_512_us, paired_packed_512_us,
      paired_direct_decode_512_us, paired_prefetch_512_us,
      reference_us /
          std::min(
              std::min(
                  paired_fused_512_us,
                  std::min(paired_64_us,
                           std::min(paired_128_us,
                                    std::min(paired_256_us, paired_512_us)))),
              std::min(
                  std::min(
                      std::min(
                          std::min(shared_tile_256_us, shared_tile_us),
                          std::min(prefetched_scale_tile_us,
                                   std::min(paired_tile_us, super_tile_us))),
                      std::min(warp_us, async_transpose_us)),
                  std::min(std::min(candidate_128_us, candidate_us),
                           std::min(candidate_512_us, candidate_1024_us)))));

  std::printf(
      "cold_config l2_bytes=%d source_count=%d "
      "source_bytes=%zu corpus_bytes=%zu destination_bytes=%zu\n",
      l2_bytes, source_count, source_bytes, source_count * source_bytes,
      output_bytes);
  const auto report_cold = [&](const char* name, auto launch) {
    std::vector<float> raw_us;
    const float median_us = benchmark_cold(launch, source_count, raw_us);
    std::printf("cold method=%s median_us=%.3f raw_us=", name, median_us);
    for (size_t run = 0; run < raw_us.size(); ++run) {
      std::printf("%s%.3f", run ? "," : "", raw_us[run]);
    }
    std::printf("\n");
  };
  report_cold("reference", [&](int source) {
    reference_kernel<<<8192, 256>>>(candidate, candidate_scales,
                                    packed + source * packed_words,
                                    scales + source * scale_bytes, global_scale,
                                    divisors, n, resident_k, scratch_k);
  });
  report_cold("sync_output512", [&](int source) {
    tiled_kernel<512><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed + source * packed_words,
        scales + source * scale_bytes, global_scale, divisors, n, resident_k,
        scratch_k);
  });
  report_cold("sync_paired256", [&](int source) {
    tiled_kernel<256, true><<<dim3(32, 32), 256>>>(
        candidate, candidate_scales, packed + source * packed_words,
        scales + source * scale_bytes, global_scale, divisors, n, resident_k,
        scratch_k);
  });
  report_cold("async_scatter", [&](int source) {
    tiled_kernel<256, true, false, true><<<dim3(32, 32), 256>>>(
        candidate, candidate_scales, packed + source * packed_words,
        scales + source * scale_bytes, global_scale, divisors, n, resident_k,
        scratch_k);
  });
  report_cold("scale_prefetch", [&](int source) {
    tiled_kernel<512, false, true><<<dim3(32, 32), 512>>>(
        candidate, candidate_scales, packed + source * packed_words,
        scales + source * scale_bytes, global_scale, divisors, n, resident_k,
        scratch_k);
  });
  report_cold("double_buffer", [&](int source) {
    double_buffer_kernel<<<dim3(32, 16), 256>>>(
        candidate, candidate_scales, packed + source * packed_words,
        scales + source * scale_bytes, global_scale, divisors, n, resident_k,
        scratch_k);
  });

  CUDA_CHECK(cudaFree(candidate));
  CUDA_CHECK(cudaFree(reference));
  CUDA_CHECK(cudaFree(candidate_scales));
  CUDA_CHECK(cudaFree(reference_scales));
  CUDA_CHECK(cudaFree(global_scale));
  CUDA_CHECK(cudaFree(divisors));
  CUDA_CHECK(cudaFree(scales));
  CUDA_CHECK(cudaFree(packed));
  return all_correct ? 0 : 2;
}
