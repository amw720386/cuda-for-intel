extern "C" __global__ void atomics(const int* keys, const float* vals, unsigned* hist, float* total, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    atomicAdd(&hist[keys[i] & 7], 1u);
    atomicAdd(total, vals[i]);
  }
}
