#include <ATen/core/TensorBase.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

namespace {

constexpr int TopK = 2048;
constexpr int kThreadsPerBlock = 1024;
constexpr size_t kSmem = 32 * 1024 * sizeof(uint32_t); // 128KB

struct FastTopKParams {
  const float *__restrict__ input; // [B, max_seq_len]
  int32_t *__restrict__ indices;   // [B, TopK]
  int32_t *__restrict__ lengths;   // [B]
  size_t max_seq_len;
};

// when length <= TopK, we can directly write the indices
__device__ void naive_topk_cuda(const float *__restrict__ score,
                                int32_t *__restrict__ indice, int32_t length) {
  const auto tid = threadIdx.x;
  for (int i = tid; i < TopK; i += kThreadsPerBlock) {
    indice[i] = (i < length) ? i : -1;
  }
}

// keep the first `length` entries, set others to -1
__device__ void
naive_topk_transform(const float *__restrict__ score, int32_t length,
                     int32_t *__restrict__ dst_page_table,
                     const int32_t *__restrict__ src_page_table) {
  const auto tid = threadIdx.x;
  for (auto i = length; i < TopK; i += kThreadsPerBlock) {
    dst_page_table[i] = (i < length) ? src_page_table[i] : -1;
  }
}

__device__ __forceinline__ uint8_t convert_to_uint8(float x) {
  __half h = __float2half_rn(x);
  uint16_t bits = __half_as_ushort(h);
  uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits & 0xFFFF)
                                 : static_cast<uint16_t>(bits | 0x8000);
  return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t convert_to_uint32(float x) {
  uint32_t bits = __float_as_uint(x);
  return (bits & 0x80000000u) ? (~bits & 0xFFFFFFFFu) : (bits | 0x80000000u);
}

template <bool Is_Epilogue = false, typename Indexer, typename Loader,
          int LENGTH, int MAX_REMAIN>
__device__ __forceinline__ auto
radix_topk(Indexer indexer, Loader loader, uint32_t length, int topk,
           int *__restrict__ index, int &__restrict__ s_counter,
           int (&__restrict__ s_histogram)[LENGTH],
           int &__restrict__ s_remain_cnt,
           int (&__restrict__ s_remain_idx)[MAX_REMAIN]) -> int {
  constexpr auto RADIX = LENGTH - 1;
  static_assert(RADIX > 1 && (RADIX & (RADIX - 1)) == 0,
                "RADIX must be power of 2");
  static_assert(RADIX <= kThreadsPerBlock);
  __shared__ uint32_t s_threshold_bin_id;

  const auto tx = threadIdx.x;
  if (tx < RADIX + 1)
    s_histogram[tx] = 0;
  __syncthreads();

  /// NOTE: Use uint32_t as the index
  for (auto i = tx; i < length; i += kThreadsPerBlock) {
    const auto idx = indexer(i);
    const auto bin = loader(idx);
    ::atomicAdd(&s_histogram[bin], 1);
  }
  __syncthreads();

  // cumsum (descending)
  if (tx == 0) {
    s_histogram[RADIX] = 0;
    s_remain_cnt = 0;
    for (int i = RADIX - 2; i >= 0; --i) {
      s_histogram[i] += s_histogram[i + 1];
    }
    // threshold bin
    for (int i = 0; i < RADIX; i++) {
      if (s_histogram[i] >= topk && s_histogram[i + 1] < topk) {
        s_threshold_bin_id = i;
        break;
      }
    }
  }
  __syncthreads();

  const auto threshold_bin = s_threshold_bin_id;
  const auto new_topk = topk - s_histogram[threshold_bin + 1];

  for (auto i = tx; i < length; i += kThreadsPerBlock) {
    const auto idx = indexer(i);
    const auto bin_id = static_cast<uint32_t>(loader(idx));
    if (bin_id > threshold_bin) {
      index[::atomicAdd(&s_counter, 1)] = idx;
    } else if (bin_id == threshold_bin && new_topk > 0) {
      if constexpr (Is_Epilogue) {
        index[::atomicAdd(&s_counter, 1)] = idx;
      } else {
        if (const auto cnt = ::atomicAdd(&s_remain_cnt, 1);
            C10_LIKELY(cnt < MAX_REMAIN)) {
          s_remain_idx[cnt] = idx;
        }
      }
    }
  }
  __syncthreads();

  return new_topk;
}

__device__ void fast_topk_cuda(const float *__restrict__ input,
                               int *__restrict__ index, int length,
                               int topk = TopK) {
  constexpr auto RADIX = 256;
  constexpr auto SMEM_INPUT_SIZE = kSmem / (2 * sizeof(int));

  __shared__ int s_histogram[RADIX + 1];
  __shared__ int s_num_input[2];
  __shared__ int s_counter;

  // allocate for two rounds
  extern __shared__ int s_input_idx[][SMEM_INPUT_SIZE];
  s_counter = 0;

  // collect candidates
  const auto indexer = [](int idx) { return idx; };
  const auto loader = [&input](int idx) {
    return convert_to_uint8(input[idx]);
  };
  int new_topk = radix_topk(indexer, loader, length, topk, index, s_counter,
                            s_histogram, s_num_input[0], s_input_idx[0]);
  if (new_topk <= 0)
    return;

  // round 0
  const auto indexer_0 = [](int idx) { return s_input_idx[0][idx]; };
  const auto loader_0 = [&input](int idx) {
    return (convert_to_uint32(input[idx]) >> 24) & 0xFF;
  };
  new_topk = radix_topk(indexer_0, loader_0, s_num_input[0], new_topk, index,
                        s_counter, s_histogram, s_num_input[1], s_input_idx[1]);
  if (new_topk <= 0)
    return;

  // round 1
  const auto indexer_1 = [](int idx) { return s_input_idx[1][idx]; };
  const auto loader_1 = [&input](int idx) {
    return (convert_to_uint32(input[idx]) >> 16) & 0xFF;
  };
  new_topk = radix_topk(indexer_1, loader_1, s_num_input[1], new_topk, index,
                        s_counter, s_histogram, s_num_input[0], s_input_idx[0]);
  if (new_topk <= 0)
    return;

  // round 2
  const auto loader_2 = [&input](int idx) {
    return (convert_to_uint32(input[idx]) >> 8) & 0xFF;
  };
  new_topk = radix_topk(indexer_0, loader_2, s_num_input[0], new_topk, index,
                        s_counter, s_histogram, s_num_input[1], s_input_idx[1]);
  if (new_topk <= 0)
    return;

  // round 3
  const auto loader_3 = [&input](int idx) {
    return convert_to_uint32(input[idx]) & 0xFF;
  };
  // epilogue
  radix_topk<true>(indexer_1, loader_3, s_num_input[1], new_topk, index,
                   s_counter, s_histogram, s_num_input[0], s_input_idx[0]);
}

__global__ void topk_kernel(const FastTopKParams params) {
  const auto &[input, indices, lengths, max_seq_len] = params;
  const auto bid = blockIdx.x;
  const auto length = *(lengths + bid);
  const auto indice = indices + bid * TopK;
  const auto score = input + bid * max_seq_len;
  if (length <= TopK) {
    return naive_topk_cuda(score, indice, length);
  } else {
    return fast_topk_cuda(score, indice, length);
  }
}

__global__ void topk_kernel_transform_decode( // decode
    const FastTopKParams params, int32_t *__restrict__ dst_page_table,
    const int32_t *__restrict__ src_page_table) {
  const auto &[input, _, lengths, max_seq_len] = params;
  const auto bid = blockIdx.x;
  const auto tid = threadIdx.x;
  const auto length = *(lengths + bid);
  const auto src_page_entry = src_page_table + bid * TopK;
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * max_seq_len;
  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    __shared__ int s_indices[TopK];
    fast_topk_cuda(score, s_indices, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

__global__ void topk_kernel_transform_prefill( // prefill
    const FastTopKParams params, int32_t *__restrict__ dst_page_table,
    const int32_t *__restrict__ src_page_table,
    const int32_t *__restrict__ cu_seqlens, const int32_t prefill_bs) {
  const auto &[input, _, lengths, max_seq_len] = params;
  const auto bid = blockIdx.x;
  const auto tid = threadIdx.x;
  const auto length = *(lengths + bid);
  const auto dst_page_entry = dst_page_table + bid * TopK;
  const auto score = input + bid * max_seq_len;

  /// NOTE: prefill bs is usually small, we can just use a simple loop here
  /// We ensure that last cu_seqlens is equal to number of blocks launched
  assert(gridDim.x == cu_seqlens[prefill_bs] &&
         "Invalid cu_seqlens in topk-transform-prefill");
  const int32_t *src_page_entry;
  for (int32_t offset = 0; offset < prefill_bs; ++offset) {
    if (bid < cu_seqlens[offset + 1]) {
      src_page_entry = src_page_table + offset * TopK;
      break;
    }
  }

  if (length <= TopK) {
    return naive_topk_transform(score, length, dst_page_entry, src_page_entry);
  } else {
    const auto src_page_entry = src_page_table + bid * TopK;
    __shared__ int s_indices[TopK];
    fast_topk_cuda(score, s_indices, length);
    // copy src[s_indices] to dst, we manually unroll here
    static_assert(TopK % kThreadsPerBlock == 0);
    static_assert(TopK / kThreadsPerBlock == 2);
    const auto idx_0 = tid;
    const auto pos_0 = s_indices[idx_0];
    dst_page_entry[idx_0] = src_page_entry[pos_0];
    const auto idx_1 = tid + kThreadsPerBlock;
    const auto pos_1 = s_indices[idx_1];
    dst_page_entry[idx_1] = src_page_entry[pos_1];
  }
}

auto get_params(at::Tensor score, at::Tensor indices, at::Tensor lengths)
    -> FastTopKParams {
  const auto B = score.size(0);
  const auto max_seq_len = score.size(1);
  TORCH_CHECK(score.dim() == 2 && score.is_contiguous());
  TORCH_CHECK(indices.dim() == 2 && indices.is_contiguous());
  TORCH_CHECK(lengths.dim() == 1 && lengths.is_contiguous());
  TORCH_CHECK(indices.size(0) == B);
  TORCH_CHECK(indices.size(1) == TopK);
  TORCH_CHECK(lengths.size(0) == B);
  return FastTopKParams{
      .input = score.data_ptr<float>(),
      .indices = indices.data_ptr<int32_t>(),
      .lengths = lengths.data_ptr<int32_t>(),
      .max_seq_len = static_cast<size_t>(max_seq_len),
  };
}

template <auto *f, size_t max_dynamic_smem>
auto setup_kernel_smem_once() -> void {
  [[maybe_unused]]
  static const auto result = [] {
    return ::cudaFuncSetAttribute(
        f, ::cudaFuncAttributeMaxDynamicSharedMemorySize, max_dynamic_smem);
  }();
  TORCH_CHECK(result == cudaSuccess,
              "set_up_kernel_once failed:", ::cudaGetErrorString(result));
}

auto fast_topk_interface(at::Tensor score, at::Tensor indices,
                         at::Tensor lengths) -> void {
  // launch kernel
  const auto params = get_params(score, indices, lengths);
  const auto B = score.size(0);
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};
  setup_kernel_smem_once<topk_kernel, kSmem>();
  topk_kernel<<<grid, block, kSmem, stream>>>(params);
  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess,
              "topk kernel failed:", ::cudaGetErrorString(result));
}

auto fast_topk_transform_interface(at::Tensor score, at::Tensor indices,
                                   at::Tensor lengths,
                                   at::Tensor dst_page_table,
                                   at::Tensor src_page_table,
                                   at::Tensor cu_seqlens) -> void {
  const auto params = get_params(score, indices, lengths);
  const auto B = score.size(0);
  const auto max_seq_len = params.max_seq_len;
  TORCH_CHECK(dst_page_table.dim() == 2 && dst_page_table.is_contiguous());
  TORCH_CHECK(src_page_table.dim() == 2 && src_page_table.is_contiguous());
  TORCH_CHECK(cu_seqlens.dim() == 1 && cu_seqlens.is_contiguous());
  const auto prefill_bs = cu_seqlens.size(0) - 1;
  TORCH_CHECK(dst_page_table.size(0) == B);
  TORCH_CHECK(dst_page_table.size(1) == TopK);
  TORCH_CHECK(src_page_table.size(0) == prefill_bs);
  TORCH_CHECK(src_page_table.size(1) == TopK);
  TORCH_CHECK(prefill_bs <= B); // prefill_bs should be smaller than expanded bs

  // launch kernel
  const auto stream = at::cuda::getCurrentCUDAStream().stream();
  const auto grid = dim3{static_cast<uint32_t>(B)};
  const auto block = dim3{kThreadsPerBlock};

  // dispatch to decode or prefill
  if (const auto is_decode = (prefill_bs == B); is_decode) {
    setup_kernel_smem_once<topk_kernel_transform_decode, kSmem>();
    topk_kernel_transform_decode<<<grid, block, kSmem, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>());
  } else {
    setup_kernel_smem_once<topk_kernel_transform_prefill, kSmem>();
    topk_kernel_transform_prefill<<<grid, block, kSmem, stream>>>(
        params, dst_page_table.data_ptr<int32_t>(),
        src_page_table.data_ptr<int32_t>(), cu_seqlens.data_ptr<int32_t>(),
        prefill_bs);
  }

  const auto result = cudaGetLastError();
  TORCH_CHECK(result == cudaSuccess,
              "topk kernel failed:", ::cudaGetErrorString(result));
}

} // namespace

PYBIND11_MODULE(topk_kernel, m) {
  m.def("fast_topk", &fast_topk_interface);
  m.def("fast_topk_transform", &fast_topk_transform_interface);
}
