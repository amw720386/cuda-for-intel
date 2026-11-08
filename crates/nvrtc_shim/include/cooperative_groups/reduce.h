#ifndef NVRTC_SHIM_CG_REDUCE_H
#define NVRTC_SHIM_CG_REDUCE_H
#include "../cooperative_groups.h"
namespace cooperative_groups {
template <class T> struct plus { __device__ T operator()(T a, T b) const { return a + b; } };
template <class T> struct less { __device__ T operator()(T a, T b) const { return a < b ? a : b; } };
template <class T> struct greater { __device__ T operator()(T a, T b) const { return a > b ? a : b; } };
template <unsigned N, class T, class Op>
__device__ __forceinline__ T reduce(const thread_block_tile<N>& g, T v, Op op) {
  for (unsigned o = N / 2; o > 0; o >>= 1) v = op(v, g.shfl_xor(v, o));
  return v;
}
}
#endif
