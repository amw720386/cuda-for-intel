template <typename T>
__global__ void axpy(T a, const T* x, T* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = a * x[i] + y[i];
}
namespace ns {
template <int K> __global__ void scale(float* p, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] *= (float)K;
}
}
