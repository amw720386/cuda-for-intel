// Block sum with shared memory tree reduction; one partial per block.
extern "C" __global__ void block_sum(const float* in, float* out, int n) {
  __shared__ float s[128];
  unsigned t = threadIdx.x;
  unsigned i = blockIdx.x * blockDim.x + t;
  s[t] = i < (unsigned)n ? in[i] : 0.0f;
  __syncthreads();
  for (unsigned o = blockDim.x / 2; o > 0; o >>= 1) {
    if (t < o) s[t] += s[t + o];
    __syncthreads();
  }
  if (t == 0) out[blockIdx.x] = s[0];
}
