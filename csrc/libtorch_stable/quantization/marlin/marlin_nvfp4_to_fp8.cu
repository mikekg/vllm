#include "dequant.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "libtorch_stable/torch_utils.h"

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

using ScalarType = torch::headeronly::ScalarType;
using Tensor = torch::stable::Tensor;

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

bool overlaps(const Tensor& a, const Tensor& b) {
  const MemoryRange a_range = memory_range(a);
  const MemoryRange b_range = memory_range(b);
  return a_range.begin < b_range.end && b_range.begin < a_range.end;
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

  STD_TORCH_CHECK(!overlaps(fp8_out, scale_out),
                  "mutable outputs must not overlap");
  STD_TORCH_CHECK(!overlaps(fp8_out, packed_weight) &&
                      !overlaps(scale_out, packed_weight) &&
                      !overlaps(fp8_out, processed_block_scales) &&
                      !overlaps(scale_out, processed_block_scales) &&
                      !overlaps(fp8_out, processed_global_scale) &&
                      !overlaps(scale_out, processed_global_scale) &&
                      !overlaps(fp8_out, tile_scale_divisor_codes) &&
                      !overlaps(scale_out, tile_scale_divisor_codes),
                  "mutable outputs must not overlap resident inputs");

  const torch::stable::accelerator::DeviceGuard device_guard(device);
  int major = 0;
  int minor = 0;
  cudaError_t error =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  STD_TORCH_CHECK(error == cudaSuccess,
                  "failed to query CUDA major: ", cudaGetErrorString(error));
  error =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  STD_TORCH_CHECK(error == cudaSuccess,
                  "failed to query CUDA minor: ", cudaGetErrorString(error));
  STD_TORCH_CHECK((major == 8 && minor == 9) || (major == 9 && minor == 0),
                  "marlin_nvfp4_to_fp8 requires SM89 or SM90");

  const cudaStream_t stream = get_current_cuda_stream(device);
  constexpr int threads = 256;
  const int64_t tiles = experts * (n_padded / 64) * (scratch_k / 32);
  const int blocks = static_cast<int>(std::min<int64_t>(tiles, 65535));
  marlin_nvfp4_to_fp8_kernel<<<blocks, threads, 0, stream>>>(
      reinterpret_cast<uint8_t*>(fp8_out.mutable_data_ptr()),
      scale_out.mutable_data_ptr<float>(),
      packed_weight.const_data_ptr<int32_t>(),
      reinterpret_cast<const uint8_t*>(processed_block_scales.const_data_ptr()),
      processed_global_scale.const_data_ptr<float>(),
      tile_scale_divisor_codes.const_data_ptr<uint8_t>(),
      static_cast<int>(experts), static_cast<int>(n_padded),
      static_cast<int>(resident_k), static_cast<int>(scratch_k),
      resident_dtype == ScalarType::BFloat16);
  error = cudaGetLastError();
  STD_TORCH_CHECK(
      error == cudaSuccess,
      "marlin_nvfp4_to_fp8 kernel launch failed: ", cudaGetErrorString(error));
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, marlin_nvfp4_to_fp8_schema) {
  marlin_nvfp4_to_fp8_schema.def(
      "marlin_nvfp4_to_fp8(Tensor(a!) fp8_out, Tensor(b!) scale_out, "
      "Tensor packed_weight, Tensor processed_block_scales, "
      "Tensor processed_global_scale, Tensor tile_scale_divisor_codes, "
      "ScalarType resident_dtype) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, marlin_nvfp4_to_fp8_impl) {
  marlin_nvfp4_to_fp8_impl.impl("marlin_nvfp4_to_fp8",
                                TORCH_BOX(&marlin_nvfp4_to_fp8));
}
