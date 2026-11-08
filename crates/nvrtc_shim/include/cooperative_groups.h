// Cooperative groups subset: thread_block, thread_block_tile<N>, tiled_partition, reduce helpers.
#ifndef NVRTC_SHIM_COOPERATIVE_GROUPS_H
#define NVRTC_SHIM_COOPERATIVE_GROUPS_H
#include "nvrtc_shim_builtins.h"
namespace cooperative_groups {
struct thread_group {
  __device__ unsigned size() const { return 0; }
};
struct thread_block {
  __device__ void sync() const { __syncthreads(); }
  __device__ static void barrier_arrive() {}
  __device__ unsigned thread_rank() const { return (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x; }
  __device__ unsigned size() const { return blockDim.x * blockDim.y * blockDim.z; }
  __device__ unsigned num_threads() const { return size(); }
  __device__ dim3 group_index() const { return dim3(blockIdx.x, blockIdx.y, blockIdx.z); }
  __device__ dim3 thread_index() const { return dim3(threadIdx.x, threadIdx.y, threadIdx.z); }
  __device__ dim3 dim_threads() const { return blockDim; }
  __device__ dim3 group_dim() const { return blockDim; }
};
__device__ __forceinline__ thread_block this_thread_block() { return thread_block(); }
template <class G> __device__ __forceinline__ void sync(const G& g) { g.sync(); }
template <unsigned N> struct thread_block_tile {
  __device__ unsigned thread_rank() const { return ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) % N; }
  __device__ unsigned meta_group_rank() const { return ((threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x) / N; }
  __device__ unsigned meta_group_size() const { return (blockDim.x * blockDim.y * blockDim.z + N - 1) / N; }
  __device__ static constexpr unsigned size() { return N; }
  __device__ static constexpr unsigned num_threads() { return N; }
  __device__ unsigned __mask() const { unsigned lane = __lane_id() & ~(N - 1); return (N == 32 ? 0xffffffffu : ((1u << N) - 1u)) << lane; }
  __device__ void sync() const { __syncwarp(__mask()); }
  template <class T> __device__ T shfl(T v, int src) const { return __shfl_idx_sync(__mask(), v, src, N); }
  template <class T> __device__ T shfl_down(T v, unsigned d) const { return __shfl_down_sync(__mask(), v, d, N); }
  template <class T> __device__ T shfl_up(T v, unsigned d) const { return __shfl_up_sync(__mask(), v, d, N); }
  template <class T> __device__ T shfl_xor(T v, unsigned d) const { return __shfl_xor_sync(__mask(), v, d, N); }
  __device__ int any(int p) const { return __any_sync(__mask(), p); }
  __device__ int all(int p) const { return __all_sync(__mask(), p); }
  __device__ unsigned ballot(int p) const { return __ballot_sync(__mask(), p) >> (__lane_id() & ~(N - 1)) & (N == 32 ? 0xffffffffu : ((1u << N) - 1u)); }
};
template <unsigned N, class P> __device__ __forceinline__ thread_block_tile<N> tiled_partition(const P&) { return thread_block_tile<N>(); }
template <unsigned N> __device__ __forceinline__ thread_block_tile<N> tiled_partition(const thread_block&) { return thread_block_tile<N>(); }
template <class G> __device__ __forceinline__ unsigned meta_group_rank(const G& g) { return g.meta_group_rank(); }
}  // namespace cooperative_groups
namespace cg = cooperative_groups;
#endif
