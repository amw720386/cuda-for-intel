#include <cuda_fp16.h>
// y = fma(x, w, b) in half; also exp in float from half input.
extern "C" __global__ void half_ops(const __half* x, const __half* w, const __half* b, __half* y, float* e, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    y[i] = __hfma(x[i], w[i], b[i]);
    e[i] = expf(__half2float(x[i]));
  }
}
