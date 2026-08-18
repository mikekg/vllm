#define main converter_lab_main
#include "converter_lab.cu"
#undef main

int main() {
  struct Shape {
    int n;
    int k;
    const char* split;
  };
  constexpr Shape shapes[] = {
      {128, 256, "train"},     {128, 2048, "train"},    {128, 8192, "train"},
      {128, 20480, "train"},   {512, 512, "train"},     {512, 4096, "train"},
      {512, 16384, "train"},   {1024, 1024, "train"},   {1024, 8192, "train"},
      {2048, 512, "train"},    {2048, 4096, "train"},   {2048, 16384, "train"},
      {4096, 1024, "train"},   {4096, 8192, "train"},   {8192, 2048, "train"},
      {8192, 8192, "train"},   {256, 16384, "holdout"}, {384, 20480, "holdout"},
      {768, 3072, "holdout"},  {1536, 6144, "holdout"}, {3072, 1280, "holdout"},
      {6144, 2560, "holdout"},
  };
  int l2_bytes = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&l2_bytes, cudaDevAttrL2CacheSize, 0));
  std::printf(
      "CONFIG l2_bytes=%d target=fixed batches=7 launches_per_batch=64\n",
      l2_bytes);
  for (const Shape shape : shapes) {
    const size_t packed_words = static_cast<size_t>(shape.k / 16) * 2 * shape.n;
    const size_t scale_bytes = static_cast<size_t>(shape.k / 16) * shape.n;
    const size_t source_bytes = packed_words * sizeof(int32_t) + scale_bytes;
    const int source_count =
        std::max<int>(64, (4LL * l2_bytes + source_bytes - 1) / source_bytes);
    const size_t output_bytes = static_cast<size_t>(shape.n) * shape.k;
    const size_t tile_count =
        static_cast<size_t>(shape.n / 128) * (shape.k / 128);
    int32_t* packed = nullptr;
    uint8_t* scales = nullptr;
    uint8_t* divisors = nullptr;
    float* global_scale = nullptr;
    float* output_scales = nullptr;
    uint8_t* output = nullptr;
    CUDA_CHECK(
        cudaMalloc(&packed, source_count * packed_words * sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&scales, source_count * scale_bytes));
    CUDA_CHECK(cudaMalloc(&divisors, tile_count));
    CUDA_CHECK(cudaMalloc(&global_scale, sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output_scales, tile_count * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&output, output_bytes));
    CUDA_CHECK(cudaMemset(packed, 0x55,
                          source_count * packed_words * sizeof(int32_t)));
    CUDA_CHECK(cudaMemset(scales, 0x38, source_count * scale_bytes));
    CUDA_CHECK(cudaMemset(divisors, 0x38, tile_count));
    const float one = 1.0f;
    CUDA_CHECK(
        cudaMemcpy(global_scale, &one, sizeof(float), cudaMemcpyHostToDevice));
    const auto launch = [&](int source) {
      double_buffer_kernel<<<dim3(shape.n / 128, shape.k / 256), 256>>>(
          output, output_scales, packed + source * packed_words,
          scales + source * scale_bytes, global_scale, divisors, shape.n,
          shape.k, shape.k);
    };
    launch(source_count - 1);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> raw_us;
    float median_us = benchmark_cold(launch, source_count, raw_us);
    std::printf(
        "RESULT split=%s n=%d k=%d median_us=%.6f source_count=%d "
        "source_bytes=%zu corpus_bytes=%zu raw_us=",
        shape.split, shape.n, shape.k, median_us, source_count, source_bytes,
        source_count * source_bytes);
    for (size_t run = 0; run < raw_us.size(); ++run) {
      std::printf("%s%.6f", run ? "," : "", raw_us[run]);
    }
    std::printf("\n");
    CUDA_CHECK(cudaFree(output));
    CUDA_CHECK(cudaFree(output_scales));
    CUDA_CHECK(cudaFree(global_scale));
    CUDA_CHECK(cudaFree(divisors));
    CUDA_CHECK(cudaFree(scales));
    CUDA_CHECK(cudaFree(packed));
  }
  return 0;
}
