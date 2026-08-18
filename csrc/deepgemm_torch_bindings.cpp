#include "python_api.cpp"

#include <torch/library.h>

#if DG_FP8_COMPATIBLE && DG_TENSORMAP_COMPATIBLE
namespace {

torch::Tensor deep_gemm_m_grouped_fp8_gemm_nt_contiguous(
    const torch::Tensor& a, const torch::Tensor& a_scale,
    const torch::Tensor& weight, const torch::Tensor& weight_scale,
    const torch::Tensor& out, const torch::Tensor& expert_ids) {
  // FP8 A[M,K], FP32 A-scale, FP8 W[G,N,K], FP32 W-scale, BF16 out[M,N],
  // and INT32 expert_ids[M]. DeepGEMM validates the exact scale layouts.
  deep_gemm::gemm::m_grouped_fp8_fp4_gemm_nt_contiguous(
      {a, a_scale}, {weight, weight_scale}, out, expert_ids, std::nullopt,
      std::nullopt, std::nullopt, "nk", true, false, std::nullopt);
  return out;
}

}  // namespace

TORCH_LIBRARY_FRAGMENT(_C, deep_gemm_dispatch) {
  deep_gemm_dispatch.def(
      "deep_gemm_m_grouped_fp8_gemm_nt_contiguous("
      "Tensor a, Tensor a_scale, Tensor weight, Tensor weight_scale, "
      "Tensor(a!) out, Tensor expert_ids) -> Tensor(a!)");
  deep_gemm_dispatch.impl("deep_gemm_m_grouped_fp8_gemm_nt_contiguous",
                          torch::kCUDA,
                          TORCH_FN(deep_gemm_m_grouped_fp8_gemm_nt_contiguous));
}
#endif
