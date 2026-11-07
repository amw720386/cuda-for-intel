// atomics histogram, atomicAdd float/int, atomicCAS max, warp shuffles, ballot, cooperative warp reduce
#include "common.h"
__global__ void hist(const unsigned char* d, int n, unsigned* h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) atomicAdd(&h[d[i]], 1u); }
__global__ void hist_shared(const unsigned char* d, int n, unsigned* h) {
  __shared__ unsigned s[256]; for (int i = threadIdx.x; i < 256; i += blockDim.x) s[i] = 0; __syncthreads();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) atomicAdd(&s[d[i]], 1u);
  __syncthreads(); for (int i = threadIdx.x; i < 256; i += blockDim.x) atomicAdd(&h[i], s[i]); }
__global__ void fsum(const float* in, float* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) atomicAdd(out, in[i]); }
__global__ void imax(const int* in, int* out, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) atomicMax(out, in[i]); }
__global__ void cas_inc(unsigned* c, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
  unsigned old = *c, assumed; do { assumed = old; old = atomicCAS(c, assumed, assumed + 1); } while (old != assumed); }
__global__ void warp_sum(const float* in, float* out) {
  float v = in[blockIdx.x * 32 + threadIdx.x];
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  if (threadIdx.x == 0) out[blockIdx.x] = v; }
__global__ void warp_bcast_xor(int* out) { int l = threadIdx.x;
  out[l] = __shfl_sync(0xffffffffu, l * 3, 5); out[32 + l] = __shfl_xor_sync(0xffffffffu, l, 1);
  out[64 + l] = __shfl_up_sync(0xffffffffu, l, 2); }
__global__ void ballot_k(unsigned* out, unsigned* any, unsigned* all) { int l = threadIdx.x;
  out[0] = __ballot_sync(0xffffffffu, l % 3 == 0); any[0] = __any_sync(0xffffffffu, l == 7); all[0] = __all_sync(0xffffffffu, l < 32); }
int main() {
  int n = 1 << 16; std::vector<unsigned char> h(n); std::vector<unsigned> ref(256, 0);
  for (int i = 0; i < n; i++) { h[i] = (unsigned char)((i * 2654435761u) >> 24); ref[h[i]]++; }
  unsigned char* d; CK(cudaMalloc(&d, n)); CK(cudaMemcpy(d, h.data(), n, cudaMemcpyHostToDevice));
  unsigned *dh; CK(cudaMalloc(&dh, 1024)); std::vector<unsigned> o(256);
  CK(cudaMemset(dh, 0, 1024)); hist<<<n / 256, 256>>>(d, n, dh); CK(cudaMemcpy(o.data(), dh, 1024, cudaMemcpyDeviceToHost));
  EXPECT(o == ref, "histogram_global_atomics");
  CK(cudaMemset(dh, 0, 1024)); hist_shared<<<32, 256>>>(d, n, dh); CK(cudaMemcpy(o.data(), dh, 1024, cudaMemcpyDeviceToHost));
  EXPECT(o == ref, "histogram_shared_atomics");
  { std::vector<float> f(4096, 0.5f); float *df, *ds; CK(cudaMalloc(&df, 4096 * 4)); CK(cudaMalloc(&ds, 4));
    CK(cudaMemcpy(df, f.data(), 4096 * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(ds, 0, 4));
    fsum<<<16, 256>>>(df, ds, 4096); float r; CK(cudaMemcpy(&r, ds, 4, cudaMemcpyDeviceToHost)); EXPECT(r == 2048.f, "atomicAdd_float"); }
  { std::vector<int> v(1024); for (int i = 0; i < 1024; i++) v[i] = (i * 37) % 1000; int *dv, *dm; CK(cudaMalloc(&dv, 4096)); CK(cudaMalloc(&dm, 4));
    CK(cudaMemcpy(dv, v.data(), 4096, cudaMemcpyHostToDevice)); CK(cudaMemset(dm, 0, 4)); imax<<<4, 256>>>(dv, dm, 1024);
    int r, e = 0; for (int x : v) if (x > e) e = x; CK(cudaMemcpy(&r, dm, 4, cudaMemcpyDeviceToHost)); EXPECT(r == e, "atomicMax_int"); }
  { unsigned *dc; CK(cudaMalloc(&dc, 4)); CK(cudaMemset(dc, 0, 4)); cas_inc<<<2, 64>>>(dc, 128);
    unsigned r; CK(cudaMemcpy(&r, dc, 4, cudaMemcpyDeviceToHost)); EXPECT(r == 128u, "atomicCAS_loop"); }
  { std::vector<float> in(64); for (int i = 0; i < 64; i++) in[i] = (float)i; float *di, *dout; CK(cudaMalloc(&di, 256)); CK(cudaMalloc(&dout, 8));
    CK(cudaMemcpy(di, in.data(), 256, cudaMemcpyHostToDevice)); warp_sum<<<2, 32>>>(di, dout); float r[2]; CK(cudaMemcpy(r, dout, 8, cudaMemcpyDeviceToHost));
    EXPECT(r[0] == 496.f && r[1] == 1520.f, "warp_shfl_down_reduce"); }
  { int* dd; CK(cudaMalloc(&dd, 96 * 4)); warp_bcast_xor<<<1, 32>>>(dd); int r[96]; CK(cudaMemcpy(r, dd, 384, cudaMemcpyDeviceToHost)); bool ok = true;
    for (int l = 0; l < 32; l++) { ok &= r[l] == 15; ok &= r[32 + l] == (l ^ 1); ok &= r[64 + l] == (l >= 2 ? l - 2 : l); }
    EXPECT(ok, "warp_shfl_idx_xor_up"); }
  { unsigned* d3; CK(cudaMalloc(&d3, 12)); ballot_k<<<1, 32>>>(d3, d3 + 1, d3 + 2); unsigned r[3]; CK(cudaMemcpy(r, d3, 12, cudaMemcpyDeviceToHost));
    unsigned e = 0; for (int l = 0; l < 32; l += 3) e |= 1u << l; EXPECT(r[0] == e && r[1] == 1 && r[2] == 1, "warp_ballot_any_all"); }
  CK(cudaDeviceSynchronize()); printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; }
