extern "C" __global__ void mathfn(const float* x, float* o, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    float v = x[i];
    o[5 * i + 0] = sqrtf(v);
    o[5 * i + 1] = expf(-v);
    o[5 * i + 2] = logf(v);
    o[5 * i + 3] = sinf(v);
    o[5 * i + 4] = tanhf(v);
  }
}
