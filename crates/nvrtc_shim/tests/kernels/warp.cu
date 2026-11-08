// Warp reduction with __shfl_down_sync, butterfly max with __shfl_xor_sync, ballot popcount.
extern "C" __global__ void warp_ops(const float* in, float* sum, float* mx, unsigned* votes) {
  unsigned t = blockIdx.x * blockDim.x + threadIdx.x;
  float v = in[t];
  float s = v;
  for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
  float m = v;
  for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
  unsigned b = __ballot_sync(0xffffffffu, v > 0.0f);
  if ((threadIdx.x & 31) == 0) {
    unsigned w = t / 32;
    sum[w] = s;
    mx[w] = m;
    votes[w] = __popc(b);
  }
}
