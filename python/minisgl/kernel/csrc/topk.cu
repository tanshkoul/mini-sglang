#include <c10/cuda/CUDAStream.h>
#include <c10/util/Exception.h>
#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

namespace {

constexpr int TopK = 2048;
constexpr int ThreadsPerBlock = 1024;

struct FastTopKParams {
  const float *input; // [B, max_seq_len]
  int32_t *indices;   // [B, TopK]
  int32_t *lengths;   // [B]
  size_t max_seq_len;
};

__global__ auto slow_topk(const FastTopKParams params) -> void {
  const auto &[input, indices, lengths, max_seq_len] = params;
  const auto bid = blockIdx.x;
  const auto tid = threadIdx.x;
  const auto length = lengths[bid];
  const auto indice = indices + bid * TopK;
  if (length <= TopK) {
    for (auto i = tid; i < TopK; i += ThreadsPerBlock) {
      indice[i] = (i < length) ? i : -1;
    }
  } else {
    // not implemented yet
    __trap();
  }
}

auto fast_topk_interface(at::Tensor score, at::Tensor indices,
                         at::Tensor lengths) -> void {
  const auto B = score.size(0);
  const auto max_seq_len = score.size(1);
  TORCH_CHECK(score.dim() == 2);
  TORCH_CHECK(indices.dim() == 2);
  TORCH_CHECK(lengths.dim() == 1);
  TORCH_CHECK(indices.size(0) == B);
  TORCH_CHECK(indices.size(1) == TopK);
  TORCH_CHECK(lengths.size(0) == B);
  const FastTopKParams params{
      score.data_ptr<float>(), indices.data_ptr<int32_t>(),
      lengths.data_ptr<int32_t>(), static_cast<size_t>(max_seq_len)};
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  slow_topk<<<B, ThreadsPerBlock, 0, stream>>>(params);
}

} // namespace

PYBIND11_MODULE(topk_kernel, m) {
  m.def("fast_topk", &fast_topk_interface, "fast_topk");
}
