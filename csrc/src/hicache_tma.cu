#include <cstdint>
#include <cstdio>
#include <cuda/barrier>
#include <cuda/ptx>

template <typename = void>
__device__ inline void mbarrier_inval(::std::uint64_t *__addr) {
  asm("mbarrier.inval.shared.b64 [%0];"
      :
      : "r"(cuda::ptx::__as_ptr_smem(__addr))
      : "memory");
}

constexpr auto kNumBuf = 512;
constexpr auto kItemSize = 256;
constexpr auto kBlockSize = 1024;
constexpr auto kRatio = 2;
constexpr auto kNumBarrier = kNumBuf / kRatio;

__global__ void tma_memcpy_kernel(const char *__restrict__ src, const int n) {
  namespace ptx = cuda::ptx;
  alignas(128) extern __shared__ char buffer[][kItemSize];
  __shared__ uint64_t mbarrier[kNumBarrier];

  if (threadIdx.x < kNumBarrier) {
    const auto bar = mbarrier + threadIdx.x;
    ptx::mbarrier_init(bar, kRatio);
  }
  __syncthreads();

  static_assert(kBlockSize % 32 == 0);
  static_assert(kBlockSize / 16 <= kNumBuf);
  static_assert(32 % kRatio == 0);
  static_assert(kBlockSize % (kRatio * 32) == 0);
  if (threadIdx.x % 32 < kRatio) {
    const auto start = (threadIdx.x / 32) * kRatio + threadIdx.x % 32;
    for (int i = start; i < n; i += (kBlockSize / 32) * kRatio) {
      const auto j = i / kRatio;
      const auto bar = mbarrier + j % kNumBarrier;
      const auto buf = buffer[i % kNumBuf];
      if (i >= kNumBuf) {
        const bool parity_bit = (j / kNumBarrier) % 2;
        while (ptx::mbarrier_test_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                              bar, parity_bit))
          ;
      }
      ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, buf,
                         src + i * kItemSize, kItemSize, bar);
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                     ptx::space_shared, bar, kItemSize);
    }
  }
}

constexpr auto kNumBuf_v2 = 64;

constexpr auto kProducers = 32 * 24;
constexpr auto kConsumers = 32 * 8;
constexpr auto kWorkingThreads = 8;
constexpr auto kBlockSize_v2 = kProducers + kConsumers;
constexpr auto kProducerWarps = kProducers / 32;
constexpr auto kConsumerWarps = kConsumers / 32;
static_assert(kProducerWarps <= kNumBuf_v2);
static_assert(kConsumerWarps <= kNumBuf_v2);

__global__ void tma_memcpy_kernel_v2(const char *__restrict__ src_,
                                     const int n) {
  namespace ptx = cuda::ptx;
  alignas(128) extern __shared__ char buf_2[][kWorkingThreads * kItemSize];
  __shared__ uint64_t _mbarrier[kNumBuf_v2 * 2];
  const auto mbarrier_producer = _mbarrier;
  const auto mbarrier_consumer = _mbarrier + kNumBuf_v2;

  if (threadIdx.x < kNumBuf_v2 * 2) {
    const auto bar = _mbarrier + threadIdx.x;
    ptx::mbarrier_init(bar, kWorkingThreads);
  }
  __syncthreads();

  const auto m = n / kWorkingThreads;
  const auto lane_id = threadIdx.x % 32;
  const auto warp_id = threadIdx.x / 32;

  if (lane_id >= kWorkingThreads)
    return;

  if (warp_id < kProducerWarps) {
    // asssume n / 32 == 0
    for (int i = warp_id; i < m; i += kProducerWarps) {
      // which buffer are we using
      const auto j = i % kNumBuf_v2;

      // wait for consumer to release buffer
      {
        const auto parity_bit = (i / kNumBuf_v2) & 1;
        const auto bar = mbarrier_consumer + j;
        while (ptx::mbarrier_test_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                              bar, parity_bit))
          ;
      }
      // produce data
      const auto bar = mbarrier_producer + j;
      const auto dst = buf_2[j] + lane_id * kItemSize;
      const auto src = src_ + (i * kWorkingThreads + lane_id) * kItemSize;
      ptx::cp_async_bulk(ptx::space_shared, ptx::space_global, dst, src,
                         kItemSize, bar);
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                     ptx::space_shared, bar, kItemSize);
    }
  } else {
    for (int i = warp_id - kProducerWarps; i < m; i += kConsumerWarps) {
      // which buffer are we using
      const auto j = i % kNumBuf_v2;

      // wait for producer to produce data
      {
        const auto parity_bit = ((i / kNumBuf_v2) & 1) ^ 1;
        const auto bar = mbarrier_producer + j;
        while (ptx::mbarrier_test_wait_parity(ptx::sem_acquire, ptx::scope_cta,
                                              bar, parity_bit))
          ;
      }

      // TODO: consume data, currently no-transaction
      const auto bar = mbarrier_consumer + j;
      ptx::mbarrier_arrive_expect_tx(ptx::sem_release, ptx::scope_cta,
                                     ptx::space_shared, bar, 0);
    }
  }
}

#define CHECK_CUDA()                                                           \
  do {                                                                         \
    cudaError_t err = cudaGetLastError();                                      \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error: %s at %d\n", cudaGetErrorString(err), __LINE__);     \
      return -1;                                                               \
    }                                                                          \
  } while (0)

int main() {
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  CHECK_CUDA();
  cudaEvent_t tic, toc;
  cudaEventCreate(&tic);
  CHECK_CUDA();
  cudaEventCreate(&toc);
  CHECK_CUDA();
  const auto smem_bytes = kNumBuf * kItemSize + 128;
  const auto smem_bytes_v2 = kNumBuf_v2 * kWorkingThreads * kItemSize + 128;
  // set dynamic shared memory size
  cudaFuncSetAttribute(tma_memcpy_kernel,
                       cudaFuncAttributeMaxDynamicSharedMemorySize, smem_bytes);
  cudaFuncSetAttribute(tma_memcpy_kernel_v2,
                       cudaFuncAttributeMaxDynamicSharedMemorySize,
                       smem_bytes_v2);
  for (const auto N : {1024, 2048, 4096, 8192, 16384, 32768}) {
    char *d_src;
    std::printf("N = %d\n", N);
    const auto nbytes = N * kItemSize;
    cudaMallocHost(&d_src, nbytes);
    // set to arange
    for (int i = 0; i < N; ++i) {
      std::fill_n(d_src + i * kItemSize, kItemSize, static_cast<char>(i));
    }
    CHECK_CUDA();

    // tma_memcpy_kernel<<<1, kBlockSize, smem_bytes, stream>>>(d_src, N);
    tma_memcpy_kernel_v2<<<1, kBlockSize_v2, smem_bytes_v2, stream>>>(d_src, N);
    cudaStreamSynchronize(stream);
    CHECK_CUDA();

    // tma_memcpy_kernel<<<1, kBlockSize, smem_bytes, stream>>>(d_src, N);
    tma_memcpy_kernel_v2<<<1, kBlockSize_v2, smem_bytes_v2, stream>>>(d_src, N);
    CHECK_CUDA();

    cudaEventRecord(tic, stream);
    // tma_memcpy_kernel<<<1, kBlockSize, smem_bytes, stream>>>(d_src, N);
    tma_memcpy_kernel_v2<<<1, kBlockSize_v2, smem_bytes_v2, stream>>>(d_src, N);
    cudaEventRecord(toc, stream);

    cudaEventSynchronize(toc);
    CHECK_CUDA();

    float milliseconds = 0;

    cudaEventElapsedTime(&milliseconds, tic, toc);
    CHECK_CUDA();

    printf("Elapsed time: %f ms\n", milliseconds);
    float bandwidth = (nbytes * 1e-9f) / (milliseconds * 1e-3f);
    printf("Effective Bandwidth: %f GB/s\n", bandwidth);
    cudaFreeHost(d_src);
  }
  cudaStreamDestroy(stream);
  return 0;
}
