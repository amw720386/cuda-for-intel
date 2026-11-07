// cub-like block/warp reduce + scan, bitonic sort, softmax, layernorm vs CPU
#include "common.h"
__inline__ __device__ float warp_reduce(float v) { for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o); return v; }
__inline__ __device__ float block_reduce(float v) {
  __shared__ float s[32]; int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
  v = warp_reduce(v); if (lane == 0) s[w] = v; __syncthreads();
  v = (threadIdx.x < (blockDim.x >> 5)) ? s[lane] : 0.f; if (w == 0) v = warp_reduce(v); return v; }
__global__ void k_block_sum(const float* in, float* out, int n) { float v = 0; for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) v += in[i];
  v = block_reduce(v); if (threadIdx.x == 0) out[blockIdx.x] = v; }
__global__ void k_block_max(const int* in, int* out, int n) { int v = INT_MIN; for (int i = threadIdx.x; i < n; i += blockDim.x) v = max(v, in[i]);
  for (int o = 16; o; o >>= 1) v = max(v, __shfl_down_sync(0xffffffffu, v, o)); __shared__ int s[32]; if ((threadIdx.x & 31) == 0) s[threadIdx.x >> 5] = v; __syncthreads();
  if (threadIdx.x == 0) { int m = s[0]; for (int i = 1; i < blockDim.x / 32; i++) m = max(m, s[i]); *out = m; } }
// Hillis-Steele inclusive scan in one block (shared double buffer) + warp-shuffle scan
__global__ void k_scan_shared(const int* in, int* out, int n) { extern __shared__ int sm[]; int t = threadIdx.x; int* a = sm; int* b = sm + n;
  a[t] = in[t]; __syncthreads(); for (int o = 1; o < n; o <<= 1) { b[t] = a[t] + (t >= o ? a[t - o] : 0); __syncthreads(); int* x = a; a = b; b = x; } out[t] = a[t]; }
__global__ void k_scan_warp(const int* in, int* out) { int l = threadIdx.x; int v = in[l];
  for (int o = 1; o < 32; o <<= 1) { int n = __shfl_up_sync(0xffffffffu, v, o); if (l >= o) v += n; } out[l] = v; }
__global__ void k_bitonic(float* d, int n) { extern __shared__ float s[]; int t = threadIdx.x; s[t] = d[t]; __syncthreads();
  for (int k = 2; k <= n; k <<= 1) for (int j = k >> 1; j > 0; j >>= 1) { int ixj = t ^ j;
    if (ixj > t) { bool up = ((t & k) == 0); float a = s[t], b = s[ixj]; if ((a > b) == up) { s[t] = b; s[ixj] = a; } } __syncthreads(); }
  d[t] = s[t]; }
__global__ void k_softmax(const float* in, float* out, int cols) { int r = blockIdx.x; const float* x = in + r * cols; float* y = out + r * cols;
  __shared__ float red[32]; float m = -INFINITY; for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, x[i]);
  for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o)); if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m; __syncthreads();
  float mm = red[0]; for (int i = 1; i < blockDim.x / 32; i++) mm = fmaxf(mm, red[i]); __syncthreads();
  float s = 0; for (int i = threadIdx.x; i < cols; i += blockDim.x) s += __expf(x[i] - mm); s = block_reduce(s); __shared__ float tot; if (threadIdx.x == 0) tot = s; __syncthreads();
  for (int i = threadIdx.x; i < cols; i += blockDim.x) y[i] = __expf(x[i] - mm) / tot; }
__global__ void k_layernorm(const float* in, const float* g, const float* b, float* out, int cols, float eps) { int r = blockIdx.x; const float* x = in + r * cols;
  float s = 0; for (int i = threadIdx.x; i < cols; i += blockDim.x) s += x[i]; s = block_reduce(s); __shared__ float mean, rstd; if (threadIdx.x == 0) mean = s / cols; __syncthreads();
  float v = 0; for (int i = threadIdx.x; i < cols; i += blockDim.x) { float d = x[i] - mean; v += d * d; } v = block_reduce(v); if (threadIdx.x == 0) rstd = rsqrtf(v / cols + eps); __syncthreads();
  for (int i = threadIdx.x; i < cols; i += blockDim.x) out[r * cols + i] = (x[i] - mean) * rstd * g[i] + b[i]; }
static bool t_block_sum() { int n = 1 << 16; std::vector<float> h(n); double ref = 0; for (int i = 0; i < n; i++) { h[i] = (i % 13) * 0.25f; ref += h[i]; }
  float* d = upload(h); float* o; CKB(cudaMalloc(&o, 64 * 4)); k_block_sum<<<64, 256>>>(d, o, n); CKB(cudaGetLastError()); auto r = download(o, 64); double s = 0; for (float v : r) s += v; return fabs(s - ref) < 1e-3 * ref; }
static bool t_block_max() { std::vector<int> h(1000); for (int i = 0; i < 1000; i++) h[i] = (i * 7919) % 10007 - 5000; int* d = upload(h); int* o; CKB(cudaMalloc(&o, 4)); k_block_max<<<1, 128>>>(d, o, 1000);
  int r; CKB(cudaMemcpy(&r, o, 4, cudaMemcpyDeviceToHost)); int e = INT_MIN; for (int v : h) e = std::max(e, v); return r == e; }
static bool t_scan() { std::vector<int> h(128), e(128); int a = 0; for (int i = 0; i < 128; i++) { h[i] = (i % 5) + 1; a += h[i]; e[i] = a; } int* d = upload(h); int* o; CKB(cudaMalloc(&o, 512));
  k_scan_shared<<<1, 128, 2 * 128 * 4>>>(d, o, 128); CKB(cudaGetLastError()); if (download(o, 128) != e) return false;
  std::vector<int> w(32), we(32); a = 0; for (int i = 0; i < 32; i++) { w[i] = i + 1; a += w[i]; we[i] = a; } int* dw = upload(w); k_scan_warp<<<1, 32>>>(dw, o); return download(o, 32) == we; }
static bool t_bitonic() { int n = 256; std::vector<float> h(n); for (int i = 0; i < n; i++) h[i] = (float)((i * 2654435761u) % 1000) - 500.f; float* d = upload(h);
  k_bitonic<<<1, n, n * 4>>>(d, n); CKB(cudaGetLastError()); auto r = download(d, n); std::sort(h.begin(), h.end()); return r == h; }
static bool t_softmax() { int R = 16, C = 200; std::vector<float> h(R * C); for (int i = 0; i < R * C; i++) h[i] = ((i * 37) % 101) / 10.f - 5.f; float* d = upload(h); float* o; CKB(cudaMalloc(&o, R * C * 4));
  k_softmax<<<R, 128>>>(d, o, C); CKB(cudaGetLastError()); auto r = download(o, R * C); for (int i = 0; i < R; i++) { double m = -1e30, s = 0; for (int j = 0; j < C; j++) m = std::max(m, (double)h[i * C + j]);
    for (int j = 0; j < C; j++) s += exp(h[i * C + j] - m); for (int j = 0; j < C; j++) if (fabs(r[i * C + j] - exp(h[i * C + j] - m) / s) > 1e-5) return false; } return true; }
static bool t_layernorm() { int R = 8, C = 96; std::vector<float> h(R * C), g(C), b(C); for (int i = 0; i < R * C; i++) h[i] = ((i * 17) % 53) / 7.f - 3.f; for (int i = 0; i < C; i++) { g[i] = 1.f + i * 0.01f; b[i] = i * 0.02f; }
  float *d = upload(h), *dg = upload(g), *db = upload(b), *o; CKB(cudaMalloc(&o, R * C * 4)); k_layernorm<<<R, 64>>>(d, dg, db, o, C, 1e-5f); CKB(cudaGetLastError()); auto r = download(o, R * C);
  for (int i = 0; i < R; i++) { double m = 0, v = 0; for (int j = 0; j < C; j++) m += h[i * C + j]; m /= C; for (int j = 0; j < C; j++) v += (h[i * C + j] - m) * (h[i * C + j] - m); v /= C;
    for (int j = 0; j < C; j++) if (fabs(r[i * C + j] - ((h[i * C + j] - m) / sqrt(v + 1e-5) * g[j] + b[j])) > 1e-4) return false; } return true; }
int main() { RUN(t_block_sum); RUN(t_block_max); RUN(t_scan); RUN(t_bitonic); RUN(t_softmax); RUN(t_layernorm);
  CK(cudaDeviceSynchronize()); printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; }
