#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include "core/registration.h"
#include "libtorch_stable/moe/permute_unpermute_kernels/moe_permute_unpermute_kernel.h"
#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/library.h>

// moe_permute kernels require at least CUDA 12.0
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)

namespace {

int64_t product_integers(torch::headeronly::IntHeaderOnlyArrayRef sizes) {
  int64_t numel = 1;
  for (int64_t s : sizes) {
    numel *= s;
  }
  return numel;
}

torch::stable::Tensor maybe_allocate_tensor(
    const std::optional<torch::stable::Tensor>& maybe_tensor,
    torch::headeronly::IntHeaderOnlyArrayRef expected_sizes,
    torch::headeronly::ScalarType dtype, torch::stable::Device device,
    char const* name) {
  auto expected_numel = product_integers(expected_sizes);
  if (maybe_tensor.has_value()) {
    auto tensor = maybe_tensor.value();
    STD_TORCH_CHECK(tensor.device() == device, name,
                    " must be on the same device");
    STD_TORCH_CHECK(tensor.scalar_type() == dtype, name,
                    " has incorrect dtype");
    STD_TORCH_CHECK(tensor.is_contiguous(), name, " must be contiguous");
    STD_TORCH_CHECK(tensor.numel() >= expected_numel, name,
                    " is too small for the requested shape");
    auto flat_tensor = torch::stable::view(tensor, {tensor.numel()});
    return torch::stable::view(
        torch::stable::narrow(flat_tensor, 0, 0, expected_numel),
        expected_sizes);
  }
  return torch::stable::empty(expected_sizes, dtype, std::nullopt, device);
}

}  // namespace

int64_t moe_permute_sort_workspace_size(int64_t num_expanded_rows,
                                        int64_t n_expert) {
  return static_cast<int64_t>(
      CubKeyValueSorter::getWorkspaceSize(num_expanded_rows, n_expert));
}

void moe_permute_impl(
    const torch::stable::Tensor& input,                 // [n_token, hidden]
    const torch::stable::Tensor& topk_ids,              // [n_token, topk]
    const torch::stable::Tensor& token_expert_indices,  // [n_token, topk]
    const std::optional<torch::stable::Tensor>& expert_map,  // [n_expert]
    int64_t n_expert, int64_t n_local_expert, int64_t topk,
    torch::stable::Tensor& permuted_input,  // [permuted_size, hidden]
    torch::stable::Tensor& expert_first_token_offset,  // [n_local_expert + 1]
    torch::stable::Tensor& inv_permuted_idx,           // [n_token, topk]
    torch::stable::Tensor& permuted_idx,               // [permute_size]
    const std::optional<torch::stable::Tensor>& maybe_sort_workspace,
    const std::optional<torch::stable::Tensor>& maybe_permuted_experts_id,
    const std::optional<torch::stable::Tensor>& maybe_sorted_row_idx,
    const std::optional<torch::stable::Tensor>& maybe_topk_ids_for_sort) {
  STD_TORCH_CHECK(expert_first_token_offset.scalar_type() ==
                      torch::headeronly::ScalarType::Long,
                  "expert_first_token_offset must be int64");
  STD_TORCH_CHECK(topk_ids.scalar_type() == torch::headeronly::ScalarType::Int,
                  "topk_ids must be int32");
  STD_TORCH_CHECK(
      token_expert_indices.scalar_type() == torch::headeronly::ScalarType::Int,
      "token_expert_indices must be int32");
  STD_TORCH_CHECK(
      inv_permuted_idx.scalar_type() == torch::headeronly::ScalarType::Int,
      "inv_permuted_idx must be int32");
  STD_TORCH_CHECK(expert_first_token_offset.size(0) == n_local_expert + 1,
                  "expert_first_token_offset shape != n_local_expert+1");
  STD_TORCH_CHECK(
      inv_permuted_idx.sizes().equals(token_expert_indices.sizes()),
      "token_expert_indices shape must be same as inv_permuted_idx");

  const torch::stable::accelerator::DeviceGuard device_guard(
      input.get_device_index());
  auto device = input.device();
  auto n_token = input.sizes()[0];
  auto n_hidden = input.sizes()[1];
  auto expanded_rows = n_token * topk;
  auto stream = get_current_cuda_stream(input.get_device_index());

  auto sorter_size = moe_permute_sort_workspace_size(expanded_rows, n_expert);
  auto sort_workspace = maybe_allocate_tensor(
      maybe_sort_workspace, {sorter_size}, torch::headeronly::ScalarType::Char,
      device, "sort_workspace");
  auto permuted_experts_id = maybe_allocate_tensor(
      maybe_permuted_experts_id, topk_ids.sizes(),
      torch::headeronly::ScalarType::Int, device, "permuted_experts_id");
  auto sorted_row_idx = maybe_allocate_tensor(
      maybe_sorted_row_idx, inv_permuted_idx.sizes(),
      torch::headeronly::ScalarType::Int, device, "sorted_row_idx");

  CubKeyValueSorter sorter{};
  int64_t* valid_num_ptr = nullptr;
  torch::stable::Tensor topk_ids_for_sort = topk_ids;

  if (expert_map.has_value()) {
    const int* expert_map_ptr = get_ptr<int>(expert_map.value());
    valid_num_ptr =
        get_ptr<int64_t>(expert_first_token_offset) + n_local_expert;
    topk_ids_for_sort = maybe_allocate_tensor(
        maybe_topk_ids_for_sort, topk_ids.sizes(),
        torch::headeronly::ScalarType::Int, device, "topk_ids_for_sort");
    torch::stable::copy_(topk_ids_for_sort, topk_ids);
    preprocessTopkIdLauncher(get_ptr<int>(topk_ids_for_sort), n_token * topk,
                             expert_map_ptr, n_expert, stream);
  }

  sortAndScanExpert(
      get_ptr<const int>(topk_ids_for_sort), get_ptr<int>(token_expert_indices),
      get_ptr<int>(permuted_experts_id), get_ptr<int>(sorted_row_idx),
      get_ptr<int64_t>(expert_first_token_offset), n_token, n_expert,
      n_local_expert, topk, sorter, get_ptr<int>(sort_workspace), stream);

  MOE_DISPATCH(input.scalar_type(), [&] {
    expandInputRowsKernelLauncher<scalar_t>(
        get_ptr<scalar_t>(input), get_ptr<scalar_t>(permuted_input),
        get_ptr<int>(sorted_row_idx), get_ptr<int>(inv_permuted_idx),
        get_ptr<int>(permuted_idx), get_ptr<int64_t>(expert_first_token_offset),
        n_token, valid_num_ptr, n_hidden, topk, n_local_expert, stream);
  });
}

void moe_permute(
    const torch::stable::Tensor& input,                 // [n_token, hidden]
    const torch::stable::Tensor& topk_ids,              // [n_token, topk]
    const torch::stable::Tensor& token_expert_indices,  // [n_token, topk]
    const std::optional<torch::stable::Tensor>& expert_map,  // [n_expert]
    int64_t n_expert, int64_t n_local_expert, int64_t topk,
    torch::stable::Tensor& permuted_input,  // [permuted_size, hidden]
    torch::stable::Tensor& expert_first_token_offset,  // [n_local_expert + 1]
    torch::stable::Tensor& inv_permuted_idx,           // [n_token, topk]
    torch::stable::Tensor& permuted_idx) {             // [permute_size]
  moe_permute_impl(input, topk_ids, token_expert_indices, expert_map, n_expert,
                   n_local_expert, topk, permuted_input,
                   expert_first_token_offset, inv_permuted_idx, permuted_idx,
                   std::nullopt, std::nullopt, std::nullopt, std::nullopt);
}

void moe_permute_with_scratch(
    const torch::stable::Tensor& input, const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& token_expert_indices,
    const std::optional<torch::stable::Tensor>& expert_map, int64_t n_expert,
    int64_t n_local_expert, int64_t topk, torch::stable::Tensor& permuted_input,
    torch::stable::Tensor& expert_first_token_offset,
    torch::stable::Tensor& inv_permuted_idx,
    torch::stable::Tensor& permuted_idx, torch::stable::Tensor& sort_workspace,
    torch::stable::Tensor& permuted_experts_id,
    torch::stable::Tensor& sorted_row_idx,
    torch::stable::Tensor& topk_ids_for_sort) {
  moe_permute_impl(input, topk_ids, token_expert_indices, expert_map, n_expert,
                   n_local_expert, topk, permuted_input,
                   expert_first_token_offset, inv_permuted_idx, permuted_idx,
                   sort_workspace, permuted_experts_id, sorted_row_idx,
                   topk_ids_for_sort);
}

template <typename IdType>
__device__ int local_expert_for_route(const IdType* topk_ids, int64_t route,
                                      const int32_t* expert_map,
                                      int64_t expert_map_size,
                                      int32_t num_experts) {
  auto expert = static_cast<int64_t>(topk_ids[route]);
  if (expert < 0) {
    return -1;
  }
  if (expert_map != nullptr) {
    int local = expert < expert_map_size ? expert_map[expert] : -1;
    return local >= 0 && local < num_experts ? local : -1;
  }
  return expert < num_experts ? static_cast<int>(expert) : -1;
}

template <typename IdType>
__global__ void countDeepGemmExpertRowsKernel(
    const IdType* topk_ids, int64_t num_routes, const int32_t* expert_map,
    int64_t expert_map_size, int32_t num_experts, int32_t* expert_offsets) {
  for (int64_t route = blockIdx.x * blockDim.x + threadIdx.x;
       route < num_routes; route += blockDim.x * gridDim.x) {
    int expert = local_expert_for_route(topk_ids, route, expert_map,
                                        expert_map_size, num_experts);
    if (expert >= 0) {
      atomicAdd(expert_offsets + expert, 1);
    }
  }
}

__global__ void alignDeepGemmExpertOffsetsKernel(int32_t* expert_offsets,
                                                 int32_t num_experts,
                                                 int32_t align_m) {
  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }
  int32_t offset = 0;
  for (int expert = 0; expert < num_experts; ++expert) {
    int32_t count = expert_offsets[expert];
    expert_offsets[expert] = offset;
    offset += ((count + align_m - 1) / align_m) * align_m;
  }
}

template <typename IdType>
__global__ void scatterDeepGemmRowsKernel(
    const uint4* input, const float* input_scale, const IdType* topk_ids,
    const int32_t* expert_map, int64_t expert_map_size, int64_t num_routes,
    int32_t num_experts, int32_t topk, int32_t hidden_u4, int32_t scale_cols,
    uint4* output, float* output_scale, int32_t* expert_ids, int32_t* inv_perm,
    int32_t* expert_offsets) {
  int64_t route = blockIdx.x;
  if (route >= num_routes) {
    return;
  }

  __shared__ int32_t destination;
  __shared__ int32_t local_expert;
  if (threadIdx.x == 0) {
    local_expert = local_expert_for_route(topk_ids, route, expert_map,
                                          expert_map_size, num_experts);
    destination =
        local_expert < 0 ? -1 : atomicAdd(expert_offsets + local_expert, 1);
    inv_perm[route] = destination;
    if (destination >= 0) {
      expert_ids[destination] = local_expert;
    }
  }
  __syncthreads();
  if (destination < 0) {
    return;
  }

  int64_t token = route / topk;
  for (int col = threadIdx.x; col < hidden_u4; col += blockDim.x) {
    output[static_cast<int64_t>(destination) * hidden_u4 + col] =
        input[token * hidden_u4 + col];
  }
  for (int col = threadIdx.x; col < scale_cols; col += blockDim.x) {
    output_scale[static_cast<int64_t>(destination) * scale_cols + col] =
        input_scale[token * scale_cols + col];
  }
}

template <typename IdType>
void launch_deepgemm_moe_permute(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& topk_ids,
    const std::optional<torch::stable::Tensor>& expert_map,
    torch::stable::Tensor& permuted_input,
    torch::stable::Tensor& permuted_scale, torch::stable::Tensor& expert_ids,
    torch::stable::Tensor& inv_perm, torch::stable::Tensor& expert_offsets,
    cudaStream_t stream) {
  int64_t num_routes = topk_ids.numel();
  int32_t num_experts = expert_offsets.numel();
  const int32_t* expert_map_ptr =
      expert_map ? get_ptr<int32_t>(*expert_map) : nullptr;
  int64_t expert_map_size = expert_map ? expert_map->numel() : 0;
  int blocks = std::min<int64_t>((num_routes + 255) / 256, 4096);
  countDeepGemmExpertRowsKernel<IdType><<<blocks, 256, 0, stream>>>(
      get_ptr<IdType>(topk_ids), num_routes, expert_map_ptr, expert_map_size,
      num_experts, get_ptr<int32_t>(expert_offsets));
  alignDeepGemmExpertOffsetsKernel<<<1, 1, 0, stream>>>(
      get_ptr<int32_t>(expert_offsets), num_experts, 128);

  scatterDeepGemmRowsKernel<IdType><<<num_routes, 256, 0, stream>>>(
      reinterpret_cast<const uint4*>(input.const_data_ptr()),
      get_ptr<float>(input_scale), get_ptr<IdType>(topk_ids), expert_map_ptr,
      expert_map_size, num_routes, num_experts, topk_ids.size(1),
      input.size(1) / 16, input_scale.size(1),
      reinterpret_cast<uint4*>(permuted_input.mutable_data_ptr()),
      get_ptr<float>(permuted_scale), get_ptr<int32_t>(expert_ids),
      get_ptr<int32_t>(inv_perm), get_ptr<int32_t>(expert_offsets));
}

void deepgemm_moe_permute(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& topk_ids,
    const std::optional<torch::stable::Tensor>& expert_map, int64_t align_m,
    torch::stable::Tensor& permuted_input,
    torch::stable::Tensor& permuted_scale, torch::stable::Tensor& expert_ids,
    torch::stable::Tensor& inv_perm, torch::stable::Tensor& expert_offsets) {
  STD_TORCH_CHECK(input.dim() == 2 && input.is_contiguous(),
                  "input must be contiguous and rank 2");
  STD_TORCH_CHECK(input.element_size() == 1 && input.size(1) % 16 == 0,
                  "input must contain 1-byte elements with K divisible by 16");
  STD_TORCH_CHECK(
      input_scale.dim() == 2 && input_scale.is_contiguous() &&
          input_scale.scalar_type() == torch::headeronly::ScalarType::Float,
      "input_scale must be contiguous FP32 and rank 2");
  STD_TORCH_CHECK(input_scale.size(0) == input.size(0),
                  "input and input_scale must have the same row count");
  STD_TORCH_CHECK(input_scale.size(1) * 128 == input.size(1),
                  "input_scale must contain one value per 128 columns");
  STD_TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.is_contiguous(),
                  "topk_ids must be contiguous and rank 2");
  STD_TORCH_CHECK(topk_ids.size(0) == input.size(0) && topk_ids.numel() > 0,
                  "topk_ids must have one non-empty row per input row");
  STD_TORCH_CHECK(align_m == 128, "DeepGEMM requires align_m=128");
  STD_TORCH_CHECK(
      permuted_input.is_contiguous() &&
          permuted_input.scalar_type() == input.scalar_type() &&
          permuted_input.size(1) == input.size(1) &&
          permuted_input.size(0) % align_m == 0 &&
          permuted_input.size(0) >=
              topk_ids.numel() + expert_offsets.numel() * (align_m - 1),
      "permuted_input has an incompatible layout");
  STD_TORCH_CHECK(permuted_scale.is_contiguous() &&
                      permuted_scale.scalar_type() ==
                          torch::headeronly::ScalarType::Float &&
                      permuted_scale.size(1) == input_scale.size(1) &&
                      permuted_scale.size(0) == permuted_input.size(0),
                  "permuted_scale has an incompatible layout");
  STD_TORCH_CHECK(
      expert_ids.scalar_type() == torch::headeronly::ScalarType::Int &&
          inv_perm.scalar_type() == torch::headeronly::ScalarType::Int &&
          expert_offsets.scalar_type() == torch::headeronly::ScalarType::Int,
      "index outputs must be int32");
  STD_TORCH_CHECK(expert_ids.numel() == permuted_input.size(0) &&
                      expert_offsets.dim() == 1 && expert_offsets.numel() > 0,
                  "expert index outputs have incompatible shapes");
  STD_TORCH_CHECK(inv_perm.sizes().equals(topk_ids.sizes()),
                  "inv_perm must match topk_ids");
  if (expert_map) {
    STD_TORCH_CHECK(
        expert_map->scalar_type() == torch::headeronly::ScalarType::Int &&
            expert_map->is_contiguous(),
        "expert_map must be contiguous int32");
  }

  const auto device = input.get_device_index();
  STD_TORCH_CHECK(input_scale.get_device_index() == device &&
                      topk_ids.get_device_index() == device &&
                      permuted_input.get_device_index() == device &&
                      permuted_scale.get_device_index() == device &&
                      expert_ids.get_device_index() == device &&
                      inv_perm.get_device_index() == device &&
                      expert_offsets.get_device_index() == device &&
                      (!expert_map || expert_map->get_device_index() == device),
                  "all inputs and outputs must be on the same device");

  const torch::stable::accelerator::DeviceGuard device_guard(device);
  auto stream = get_current_cuda_stream(device);
  cudaMemsetAsync(expert_offsets.mutable_data_ptr(), 0,
                  expert_offsets.numel() * expert_offsets.element_size(),
                  stream);
  cudaMemsetAsync(expert_ids.mutable_data_ptr(), 0xff,
                  expert_ids.numel() * expert_ids.element_size(), stream);
  cudaMemsetAsync(inv_perm.mutable_data_ptr(), 0xff,
                  inv_perm.numel() * inv_perm.element_size(), stream);
  cudaMemsetAsync(permuted_scale.mutable_data_ptr(), 0,
                  permuted_scale.numel() * permuted_scale.element_size(),
                  stream);

  if (topk_ids.scalar_type() == torch::headeronly::ScalarType::Int) {
    launch_deepgemm_moe_permute<int32_t>(
        input, input_scale, topk_ids, expert_map, permuted_input,
        permuted_scale, expert_ids, inv_perm, expert_offsets, stream);
  } else {
    STD_TORCH_CHECK(
        topk_ids.scalar_type() == torch::headeronly::ScalarType::Long,
        "topk_ids must be int32 or int64");
    launch_deepgemm_moe_permute<int64_t>(
        input, input_scale, topk_ids, expert_map, permuted_input,
        permuted_scale, expert_ids, inv_perm, expert_offsets, stream);
  }
}

void moe_unpermute(
    const torch::stable::Tensor&
        permuted_hidden_states,                     // [n_token * topk, hidden]
    const torch::stable::Tensor& topk_weights,      // [n_token, topk]
    const torch::stable::Tensor& inv_permuted_idx,  // [n_token, topk]
    const std::optional<torch::stable::Tensor>&
        expert_first_token_offset,  // [n_local_expert+1]
    int64_t topk,
    torch::stable::Tensor& hidden_states) {  // [n_token, hidden]
  STD_TORCH_CHECK(
      permuted_hidden_states.scalar_type() == hidden_states.scalar_type(),
      "permuted_hidden_states dtype must be same as hidden_states");

  const torch::stable::accelerator::DeviceGuard device_guard(
      hidden_states.get_device_index());
  auto n_token = hidden_states.size(0);
  auto n_hidden = hidden_states.size(1);
  auto stream = get_current_cuda_stream(hidden_states.get_device_index());

  int64_t const* valid_ptr = nullptr;
  if (expert_first_token_offset.has_value()) {
    int n_local_expert = expert_first_token_offset.value().size(0) - 1;
    valid_ptr =
        get_ptr<int64_t>(expert_first_token_offset.value()) + n_local_expert;
  }

  MOE_DISPATCH(hidden_states.scalar_type(), [&] {
    finalizeMoeRoutingKernelLauncher<scalar_t, scalar_t>(
        get_ptr<scalar_t>(permuted_hidden_states),
        get_ptr<scalar_t>(hidden_states), get_ptr<float>(topk_weights),
        get_ptr<int>(inv_permuted_idx), n_token, n_hidden, topk, valid_ptr,
        stream);
  });
}

template <typename T>
__global__ void shuffleInputRowsKernel(const T* input,
                                       const int32_t* dst2src_map, T* output,
                                       int64_t num_src_rows,
                                       int64_t num_dst_rows, int64_t num_cols) {
  int64_t dest_row_idx = blockIdx.x;
  if (blockIdx.x < num_dst_rows) {
    int64_t const source_row_idx = dst2src_map[dest_row_idx];
    // Load 128-bits per thread
    constexpr int64_t ELEM_PER_THREAD = 128 / sizeof(T) / 8;
    using DataElem = cutlass::Array<T, ELEM_PER_THREAD>;

    auto* dest_row_ptr =
        reinterpret_cast<DataElem*>(output + dest_row_idx * num_cols);
    int64_t const start_offset = threadIdx.x;
    int64_t const stride = blockDim.x;
    int64_t const num_elems_in_col = num_cols / ELEM_PER_THREAD;

    if (source_row_idx < 0 || source_row_idx >= num_src_rows) {
      DataElem const zero{};
      for (int elem_index = start_offset; elem_index < num_elems_in_col;
           elem_index += stride) {
        dest_row_ptr[elem_index] = zero;
      }
      return;
    }

    // Duplicate and permute rows
    auto const* source_row_ptr =
        reinterpret_cast<DataElem const*>(input + source_row_idx * num_cols);
    for (int elem_index = start_offset; elem_index < num_elems_in_col;
         elem_index += stride) {
      dest_row_ptr[elem_index] = source_row_ptr[elem_index];
    }
  }
}

void shuffle_rows(const torch::stable::Tensor& input_tensor,
                  const torch::stable::Tensor& dst2src_map,
                  torch::stable::Tensor& output_tensor) {
  STD_TORCH_CHECK(input_tensor.scalar_type() == output_tensor.scalar_type(),
                  "Input and output tensors must have the same data type");

  const torch::stable::accelerator::DeviceGuard device_guard(
      output_tensor.get_device_index());
  auto stream = get_current_cuda_stream(output_tensor.get_device_index());
  const int64_t blocks = output_tensor.size(0);
  const int64_t threads = 256;
  const int64_t num_dest_rows = output_tensor.size(0);
  const int64_t num_src_rows = input_tensor.size(0);
  const int64_t num_cols = input_tensor.size(1);

  STD_TORCH_CHECK(!(num_cols % (128 / input_tensor.element_size() / 8)),
                  "num_cols must be divisible by 128 / "
                  "input_tensor.element_size() / 8");

  MOE_DISPATCH(input_tensor.scalar_type(), [&] {
    shuffleInputRowsKernel<scalar_t><<<blocks, threads, 0, stream>>>(
        reinterpret_cast<const scalar_t*>(input_tensor.const_data_ptr()),
        reinterpret_cast<const int32_t*>(dst2src_map.const_data_ptr()),
        reinterpret_cast<scalar_t*>(output_tensor.mutable_data_ptr()),
        num_src_rows, num_dest_rows, num_cols);
  });
}

#else

int64_t moe_permute_sort_workspace_size(int64_t num_expanded_rows,
                                        int64_t n_expert) {
  STD_TORCH_CHECK(
      false, "moe_permute_sort_workspace_size is not supported on CUDA < 12.0");
}

void moe_permute(const torch::stable::Tensor& input,
                 const torch::stable::Tensor& topk_ids,
                 const torch::stable::Tensor& token_expert_indices,
                 const std::optional<torch::stable::Tensor>& expert_map,
                 int64_t n_expert, int64_t n_local_expert, int64_t topk,
                 torch::stable::Tensor& permuted_input,
                 torch::stable::Tensor& expert_first_token_offset,
                 torch::stable::Tensor& inv_permuted_idx,
                 torch::stable::Tensor& permuted_idx) {
  STD_TORCH_CHECK(false, "moe_permute is not supported on CUDA < 12.0");
}

void moe_permute_with_scratch(
    const torch::stable::Tensor& input, const torch::stable::Tensor& topk_ids,
    const torch::stable::Tensor& token_expert_indices,
    const std::optional<torch::stable::Tensor>& expert_map, int64_t n_expert,
    int64_t n_local_expert, int64_t topk, torch::stable::Tensor& permuted_input,
    torch::stable::Tensor& expert_first_token_offset,
    torch::stable::Tensor& inv_permuted_idx,
    torch::stable::Tensor& permuted_idx, torch::stable::Tensor& sort_workspace,
    torch::stable::Tensor& permuted_experts_id,
    torch::stable::Tensor& sorted_row_idx,
    torch::stable::Tensor& topk_ids_for_sort) {
  STD_TORCH_CHECK(false,
                  "moe_permute_with_scratch is not supported on CUDA < 12.0");
}

void deepgemm_moe_permute(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& input_scale,
    const torch::stable::Tensor& topk_ids,
    const std::optional<torch::stable::Tensor>& expert_map, int64_t align_m,
    torch::stable::Tensor& permuted_input,
    torch::stable::Tensor& permuted_scale, torch::stable::Tensor& expert_ids,
    torch::stable::Tensor& inv_perm, torch::stable::Tensor& expert_offsets) {
  STD_TORCH_CHECK(false,
                  "deepgemm_moe_permute is not supported on CUDA < 12.0");
}

void moe_unpermute(
    const torch::stable::Tensor& permuted_hidden_states,
    const torch::stable::Tensor& topk_weights,
    const torch::stable::Tensor& inv_permuted_idx,
    const std::optional<torch::stable::Tensor>& expert_first_token_offset,
    int64_t topk, torch::stable::Tensor& hidden_states) {
  STD_TORCH_CHECK(false, "moe_unpermute is not supported on CUDA < 12.0");
}

#endif

bool moe_permute_unpermute_supported() {
#if defined(CUDA_VERSION) && (CUDA_VERSION >= 12000)
  return true;
#else
  return false;
#endif
}

STABLE_TORCH_LIBRARY_IMPL(_moe_C, CUDA, m) {
  m.impl("moe_permute", TORCH_BOX(&moe_permute));
  m.impl("moe_permute_with_scratch", TORCH_BOX(&moe_permute_with_scratch));
  m.impl("deepgemm_moe_permute", TORCH_BOX(&deepgemm_moe_permute));
  m.impl("moe_unpermute", TORCH_BOX(&moe_unpermute));
}
