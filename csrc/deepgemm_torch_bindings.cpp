#include "python_api.cpp"

#include "core/scalar_type.hpp"

#include <ATen/core/dispatch/Dispatcher.h>
#include <torch/library.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>

#if DG_FP8_COMPATIBLE && DG_TENSORMAP_COMPATIBLE
namespace {

using torch::Tensor;

int64_t round_up(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

struct RegisteredOps {
  c10::OperatorHandle convert = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_C::marlin_nvfp4_to_fp8", "");
  c10::OperatorHandle quant = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_C::per_token_group_fp8_quant", "");
  c10::OperatorHandle silu =
      c10::Dispatcher::singleton().findSchemaOrThrow("_C::silu_and_mul", "");
  c10::OperatorHandle align = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_moe_C::moe_align_block_size", "");
  c10::OperatorHandle marlin = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_moe_C::moe_wna16_marlin_gemm", "");
  c10::OperatorHandle sum =
      c10::Dispatcher::singleton().findSchemaOrThrow("_moe_C::moe_sum", "");
  c10::OperatorHandle permute = c10::Dispatcher::singleton().findSchemaOrThrow(
      "_moe_C::deepgemm_moe_permute", "");
  c10::OperatorHandle unpermute =
      c10::Dispatcher::singleton().findSchemaOrThrow("_moe_C::moe_unpermute",
                                                     "");
};

const RegisteredOps& registered_ops() {
  static const RegisteredOps ops;
  return ops;
}

c10::IValue optional_tensor(const std::optional<Tensor>& tensor) {
  return tensor ? c10::IValue(*tensor) : c10::IValue();
}

void call_boxed(const c10::OperatorHandle& op,
                std::initializer_list<c10::IValue> arguments) {
  c10::Stack stack(arguments);
  op.callBoxed(&stack);
}

void deep_gemm_m_grouped_fp8_gemm_nt_contiguous(
    const Tensor& a, const Tensor& a_scale, const Tensor& weight,
    const Tensor& weight_scale, const Tensor& out, const Tensor& expert_ids) {
  deep_gemm::gemm::m_grouped_fp8_fp4_gemm_nt_contiguous(
      {a, a_scale}, {weight, weight_scale}, out, expert_ids, std::nullopt,
      std::nullopt, std::nullopt, "nk", true, false, true, std::nullopt);
}

void call_converter(const Tensor& fp8_weight, const Tensor& fp8_scale,
                    const Tensor& packed_weight,
                    const Tensor& processed_block_scales,
                    const Tensor& processed_global_scale,
                    const Tensor& divisor_codes,
                    at::ScalarType resident_dtype) {
  call_boxed(registered_ops().convert,
             {fp8_weight, fp8_scale, packed_weight, processed_block_scales,
              processed_global_scale, divisor_codes, resident_dtype});
}

void call_marlin(const Tensor& input, const Tensor& output,
                 const Tensor& packed_weight, const Tensor& scales,
                 const Tensor& global_scale, const Tensor& marlin_workspace,
                 const Tensor& sorted_token_ids, const Tensor& expert_ids,
                 const Tensor& num_tokens_post_pad, const Tensor& topk_weights,
                 int64_t block_size, int64_t topk, bool multiply_topk_weights,
                 int64_t m, int64_t n, int64_t k) {
  const c10::IValue none;
  call_boxed(registered_ops().marlin, {input,
                                       output,
                                       packed_weight,
                                       none,
                                       scales,
                                       none,
                                       global_scale,
                                       none,
                                       none,
                                       none,
                                       marlin_workspace,
                                       sorted_token_ids,
                                       expert_ids,
                                       num_tokens_post_pad,
                                       topk_weights,
                                       block_size,
                                       topk,
                                       multiply_topk_weights,
                                       vllm::kFE2M1f.id(),
                                       m,
                                       n,
                                       k,
                                       true,
                                       false,
                                       true,
                                       false,
                                       -1,
                                       -1,
                                       -1});
}

void marlin_nvfp4_hybrid_moe(
    const Tensor& hidden_states, const Tensor& w13, const Tensor& w2,
    const Tensor& w13_scales, const Tensor& w2_scales,
    const Tensor& w13_global_scale, const Tensor& w2_global_scale,
    const Tensor& w13_divisor_codes, const Tensor& w2_divisor_codes,
    const Tensor& topk_weights, const Tensor& topk_ids,
    const std::optional<Tensor>& expert_map, const Tensor& marlin_workspace,
    const Tensor& output, const Tensor& arena, int64_t global_num_experts,
    int64_t m_knee, bool apply_router_weight_on_input) {
  TORCH_CHECK(hidden_states.is_cuda() && hidden_states.is_contiguous() &&
                  hidden_states.dim() == 2,
              "hidden_states must be a contiguous CUDA matrix");
  TORCH_CHECK(hidden_states.scalar_type() == at::kHalf ||
                  hidden_states.scalar_type() == at::kBFloat16,
              "hidden_states must be float16 or bfloat16");
  TORCH_CHECK(w13.dim() == 3 && w2.dim() == 3 && w13.is_contiguous() &&
                  w2.is_contiguous(),
              "Marlin expert weights must be contiguous rank-3 tensors");
  TORCH_CHECK(w13.scalar_type() == at::kInt && w2.scalar_type() == at::kInt,
              "Marlin expert weights must use int32 storage");
  TORCH_CHECK(topk_ids.dim() == 2 && topk_ids.is_contiguous() &&
                  topk_weights.sizes() == topk_ids.sizes() &&
                  topk_weights.scalar_type() == at::kFloat,
              "routing tensors have incompatible layouts");
  TORCH_CHECK(output.sizes() == hidden_states.sizes() &&
                  output.scalar_type() == hidden_states.scalar_type() &&
                  output.is_contiguous(),
              "output must match hidden_states");
  TORCH_CHECK(arena.is_contiguous() && marlin_workspace.is_contiguous() &&
                  marlin_workspace.scalar_type() == at::kInt,
              "workspaces must be contiguous and Marlin workspace int32");
  TORCH_CHECK(m_knee > 0, "m_knee must be positive");

  const int64_t e = w13.size(0);
  const int64_t m = hidden_states.size(0);
  const int64_t k = hidden_states.size(1);
  const int64_t n = w2.size(1) * 16;
  const int64_t topk = topk_ids.size(1);
  const int64_t routes = m * topk;
  if (m == 0) return;
  if (global_num_experts == -1) global_num_experts = e;
  TORCH_CHECK(e > 0 && n > 0 && k > 0 && topk > 0 && global_num_experts >= e,
              "MoE dimensions and global expert count must be positive");
  TORCH_CHECK(w2.size(0) == e && w13.size(1) == k / 16 &&
                  w13.size(2) == 4 * n && w2.size(2) == 2 * k,
              "Marlin expert weights have incompatible dimensions");
  if (expert_map) {
    TORCH_CHECK(expert_map->scalar_type() == at::kInt &&
                    expert_map->is_contiguous() &&
                    expert_map->numel() == global_num_experts,
                "expert_map must be contiguous int32 with global-E entries");
  }

  Tensor arena_bytes = arena.view(at::kByte).flatten();
  int64_t offset = 0;
  auto take_bytes = [&](int64_t bytes) {
    offset = round_up(offset, 256);
    TORCH_CHECK(bytes >= 0 && offset <= arena_bytes.numel() - bytes,
                "transient MoE arena is too small");
    Tensor result = arena_bytes.narrow(0, offset, bytes);
    offset += bytes;
    return result;
  };
  const int64_t itemsize = hidden_states.element_size();

  if (m < m_knee) {
    int64_t effective_m = (m * e + global_num_experts - 1) / global_num_experts;
    int64_t block_size = 64;
    for (int64_t candidate : std::array<int64_t, 5>{8, 16, 32, 48, 64}) {
      block_size = candidate;
      if (10 * effective_m * topk < 9 * e * candidate) break;
    }
    int64_t padded_routes = routes + global_num_experts * (block_size - 1);
    if (routes < global_num_experts) {
      padded_routes = std::min(routes * block_size, padded_routes);
    }
    const int64_t first_bytes =
        std::max(output.numel() * output.element_size(), routes * n * itemsize);
    Tensor cache2 = take_bytes(first_bytes)
                        .narrow(0, 0, routes * n * itemsize)
                        .view(hidden_states.scalar_type())
                        .view({routes, n});
    Tensor cache13 = take_bytes(routes * std::max(2 * n, k) * itemsize)
                         .view(hidden_states.scalar_type());
    Tensor sorted_token_ids = take_bytes(padded_routes * sizeof(int32_t))
                                  .view(at::kInt)
                                  .view({padded_routes});
    Tensor expert_ids =
        take_bytes(((padded_routes + block_size - 1) / block_size) *
                   sizeof(int32_t))
            .view(at::kInt)
            .view({(padded_routes + block_size - 1) / block_size});
    Tensor num_tokens_post_pad = take_bytes(sizeof(int32_t)).view(at::kInt);

    call_boxed(registered_ops().align,
               {topk_ids, global_num_experts, block_size, sorted_token_ids,
                expert_ids, num_tokens_post_pad, optional_tensor(expert_map)});
    Tensor mm1 = cache13.narrow(0, 0, routes * 2 * n).view({routes, 2 * n});
    call_marlin(hidden_states, mm1, w13, w13_scales, w13_global_scale,
                marlin_workspace, sorted_token_ids, expert_ids,
                num_tokens_post_pad, topk_weights, block_size, topk, false, m,
                2 * n, k);
    call_boxed(registered_ops().silu, {cache2, mm1});
    Tensor mm2 = cache13.narrow(0, 0, routes * k).view({routes, k});
    call_marlin(cache2, mm2, w2, w2_scales, w2_global_scale, marlin_workspace,
                sorted_token_ids, expert_ids, num_tokens_post_pad, topk_weights,
                block_size, 1, !apply_router_weight_on_input, routes, k, n);
    call_boxed(registered_ops().sum, {mm2.view({m, topk, k}), output, topk_ids,
                                      optional_tensor(expert_map)});
    return;
  }

  TORCH_CHECK(n >= 512 && n % 128 == 0 && k % 128 == 0,
              "DeepGEMM branch requires N >= 512 and N,K divisible by 128");
  constexpr int64_t block = 128;
  const int64_t m_sum = round_up(routes + e * (block - 1), block);
  const int64_t first_bytes =
      std::max({output.numel() * output.element_size(), m * k, m_sum * n});
  Tensor quant_scratch = take_bytes(first_bytes);
  const int64_t scale_scratch_values =
      std::max({m * (k / block), (n / block) * m_sum,
                apply_router_weight_on_input ? routes : int64_t{0}});
  Tensor scale_scratch =
      take_bytes(scale_scratch_values * sizeof(float)).view(at::kFloat);
  Tensor permuted_input =
      take_bytes(m_sum * k).view(at::kFloat8_e4m3fn).view({m_sum, k});
  Tensor permuted_scale = take_bytes(m_sum * (k / block) * sizeof(float))
                              .view(at::kFloat)
                              .view({m_sum, k / block});
  Tensor gemm_output = take_bytes(m_sum * std::max(2 * n, k) * itemsize)
                           .view(hidden_states.scalar_type());
  const int64_t weight_values = std::max(e * 2 * n * k, m_sum * n * itemsize);
  Tensor weight_scratch = take_bytes(weight_values).view(at::kFloat8_e4m3fn);
  const int64_t weight_scale_values = std::max(
      e * (2 * n / block) * (k / block), e * (k / block) * (n / block));
  Tensor weight_scale_scratch =
      take_bytes(weight_scale_values * sizeof(float)).view(at::kFloat);
  Tensor expert_ids =
      take_bytes(m_sum * sizeof(int32_t)).view(at::kInt).view({m_sum});
  Tensor inv_perm =
      take_bytes(routes * sizeof(int32_t)).view(at::kInt).view({m, topk});
  Tensor expert_offsets =
      take_bytes(e * sizeof(int32_t)).view(at::kInt).view({e});

  Tensor w13_fp8 =
      weight_scratch.narrow(0, 0, e * 2 * n * k).view({e, 2 * n, k});
  Tensor w13_fp8_scale =
      weight_scale_scratch.narrow(0, 0, e * (2 * n / block) * (k / block))
          .view({e, 2 * n / block, k / block});
  call_converter(w13_fp8, w13_fp8_scale, w13, w13_scales, w13_global_scale,
                 w13_divisor_codes, hidden_states.scalar_type());

  Tensor a1q =
      quant_scratch.narrow(0, 0, m * k).view(at::kFloat8_e4m3fn).view({m, k});
  Tensor a1_scale =
      scale_scratch.narrow(0, 0, m * (k / block)).view({m, k / block});
  call_boxed(registered_ops().quant,
             {hidden_states, a1q, a1_scale, block, 1.0e-10, -448.0, 448.0,
              false, false, false});
  call_boxed(
      registered_ops().permute,
      {a1q, a1_scale, topk_ids, optional_tensor(expert_map), block,
       permuted_input, permuted_scale, expert_ids, inv_perm, expert_offsets});

  Tensor mm1 = gemm_output.narrow(0, 0, m_sum * 2 * n).view({m_sum, 2 * n});
  deep_gemm_m_grouped_fp8_gemm_nt_contiguous(
      permuted_input, permuted_scale, w13_fp8, w13_fp8_scale, mm1, expert_ids);

  Tensor a2q = quant_scratch.narrow(0, 0, m_sum * n)
                   .view(at::kFloat8_e4m3fn)
                   .view({m_sum, n});
  Tensor a2_scale = scale_scratch.narrow(0, 0, (n / block) * m_sum)
                        .view({n / block, m_sum})
                        .transpose(0, 1);
  Tensor a2 = weight_scratch.narrow(0, 0, m_sum * n * itemsize)
                  .view(hidden_states.scalar_type())
                  .view({m_sum, n});
  call_boxed(registered_ops().silu, {a2, mm1});
  call_boxed(registered_ops().quant, {a2, a2q, a2_scale, block, 1.0e-10, -448.0,
                                      448.0, false, false, false});

  Tensor w2_fp8 = weight_scratch.narrow(0, 0, e * k * n).view({e, k, n});
  Tensor w2_fp8_scale =
      weight_scale_scratch.narrow(0, 0, e * (k / block) * (n / block))
          .view({e, k / block, n / block});
  call_converter(w2_fp8, w2_fp8_scale, w2, w2_scales, w2_global_scale,
                 w2_divisor_codes, hidden_states.scalar_type());
  Tensor mm2 = gemm_output.narrow(0, 0, m_sum * k).view({m_sum, k});
  deep_gemm_m_grouped_fp8_gemm_nt_contiguous(a2q, a2_scale, w2_fp8,
                                             w2_fp8_scale, mm2, expert_ids);

  Tensor reduction_weights = topk_weights;
  if (apply_router_weight_on_input) {
    reduction_weights = scale_scratch.narrow(0, 0, routes).view({m, topk});
    reduction_weights.fill_(1.0);
  }
  call_boxed(registered_ops().unpermute,
             {mm2, reduction_weights, inv_perm, c10::IValue(), topk, output});
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, deep_gemm_dispatch) {
  deep_gemm_dispatch.def(
      "deep_gemm_m_grouped_fp8_gemm_nt_contiguous("
      "Tensor a, Tensor a_scale, Tensor weight, Tensor weight_scale, "
      "Tensor(a!) out, Tensor expert_ids) -> Tensor(a!)");
  deep_gemm_dispatch.impl(
      "deep_gemm_m_grouped_fp8_gemm_nt_contiguous", torch::kCUDA,
      [](const Tensor& a, const Tensor& a_scale, const Tensor& weight,
         const Tensor& weight_scale, const Tensor& out,
         const Tensor& expert_ids) {
        deep_gemm_m_grouped_fp8_gemm_nt_contiguous(
            a, a_scale, weight, weight_scale, out, expert_ids);
        return out;
      });
  deep_gemm_dispatch.def(
      "marlin_nvfp4_hybrid_moe("
      "Tensor hidden_states, Tensor w13, Tensor w2, Tensor w13_scales, "
      "Tensor w2_scales, Tensor w13_global_scale, Tensor w2_global_scale, "
      "Tensor w13_divisor_codes, Tensor w2_divisor_codes, "
      "Tensor topk_weights, Tensor topk_ids, Tensor? expert_map, "
      "Tensor marlin_workspace, Tensor(a!) output, Tensor arena, "
      "int global_num_experts, int m_knee, "
      "bool apply_router_weight_on_input) -> ()");
  deep_gemm_dispatch.impl("marlin_nvfp4_hybrid_moe", torch::kCUDA,
                          TORCH_FN(marlin_nvfp4_hybrid_moe));
}
#endif
