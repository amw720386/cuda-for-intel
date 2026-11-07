// vector add, block reduction, naive + shared-tiled transpose, tiled matmul
#include "common.h"
__global__ void vadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) c[i] = a[i] + b[i]; }
__global__ void reduce_sum(const float* in, float* out, int n) {
  extern __shared__ float s[]; int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
  s[t] = i < n ? in[i] : 0.f; __syncthreads();
  for (int o = blockDim.x / 2; o > 0; o >>= 1) { if (t < o) s[t] += s[t + o]; __syncthreads(); }
  if (t == 0) out[blockIdx.x] = s[0]; }
__global__ void transpose_naive(const float* in, float* out, int w, int h) {
  int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
  if (x < w && y < h) out[x * h + y] = in[y * w + x]; }
__global__ void transpose_tiled(const float* in, float* out, int w, int h) {
  __shared__ float tile[32][33];
  int x = blockIdx.x * 32 + threadIdx.x, y = blockIdx.y * 32 + threadIdx.y;
  if (x < w && y < h) tile[threadIdx.y][threadIdx.x] = in[y * w + x];
  __syncthreads();
  x = blockIdx.y * 32 + threadIdx.x; y = blockIdx.x * 32 + threadIdx.y;
  if (x < h && y < w) out[y * h + x] = tile[threadIdx.x][threadIdx.y]; }
__global__ void matmul_tiled(const float* A, const float* B, float* C, int N) {
  __shared__ float As[16][16], Bs[16][16];
  int r = blockIdx.y * 16 + threadIdx.y, c = blockIdx.x * 16 + threadIdx.x; float acc = 0.f;
  for (int k = 0; k < N; k += 16) {
    As[threadIdx.y][threadIdx.x] = A[r * N + k + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[(k + threadIdx.y) * N + c];
    __syncthreads();
    for (int j = 0; j < 16; j++) acc += As[threadIdx.y][j] * Bs[j][threadIdx.x];
    __syncthreads(); }
  C[r * N + c] = acc; }
template <class T> T* dev(const std::vector<T>& h) { T* d; CK(cudaMalloc(&d, h.size() * sizeof(T)));
  CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice)); return d; }
int main() {
  int cnt = 0; CK(cudaGetDeviceCount(&cnt)); printf("devices=%d\n", cnt);
  { int n = 1 << 20; std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; i++) { a[i] = i; b[i] = 2.f * i; }
    float *da = dev(a), *db = dev(b), *dc; CK(cudaMalloc(&dc, n * 4));
    vadd<<<(n + 255) / 256, 256>>>(da, db, dc, n); CK(cudaGetLastError());
    CK(cudaMemcpy(c.data(), dc, n * 4, cudaMemcpyDeviceToHost));
    bool ok = true; for (int i = 0; i < n; i++) ok &= (c[i] == 3.f * i);
    EXPECT(ok, "vector_add"); cudaFree(da); cudaFree(db); cudaFree(dc); }
  { int n = 1 << 18, B = 256, G = (n + B - 1) / B; std::vector<float> a(n, 1.f);
    float* da = dev(a); float *dp, *dq; CK(cudaMalloc(&dp, G * 4)); CK(cudaMalloc(&dq, 64));
    reduce_sum<<<G, B, B * 4>>>(da, dp, n);
    reduce_sum<<<(G + B - 1) / B, B, B * 4>>>(dp, dq, G); std::vector<float> part((G + B - 1) / B);
    CK(cudaMemcpy(part.data(), dq, part.size() * 4, cudaMemcpyDeviceToHost));
    float tot = 0; for (float v : part) tot += v;
    EXPECT(tot == (float)n, "reduce_sum"); }
  { int w = 96, h = 64; std::vector<float> in(w * h), o1(w * h), o2(w * h);
    for (int i = 0; i < w * h; i++) in[i] = (float)i;
    float* di = dev(in); float *d1, *d2; CK(cudaMalloc(&d1, w * h * 4)); CK(cudaMalloc(&d2, w * h * 4));
    dim3 blk(32, 32), grd((w + 31) / 32, (h + 31) / 32);
    transpose_naive<<<grd, blk>>>(di, d1, w, h); transpose_tiled<<<grd, blk>>>(di, d2, w, h);
    CK(cudaMemcpy(o1.data(), d1, w * h * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(o2.data(), d2, w * h * 4, cudaMemcpyDeviceToHost));
    bool a1 = true, a2 = true;
    for (int y = 0; y < h; y++) for (int x = 0; x < w; x++) { a1 &= o1[x * h + y] == in[y * w + x]; a2 &= o2[x * h + y] == in[y * w + x]; }
    EXPECT(a1, "transpose_naive"); EXPECT(a2, "transpose_shared_tiled"); }
  { int N = 128; std::vector<float> A(N * N), B(N * N), C(N * N);
    for (int i = 0; i < N * N; i++) { A[i] = (i % 7) * 0.5f; B[i] = (i % 5) - 2.f; }
    float *dA = dev(A), *dB = dev(B), *dC; CK(cudaMalloc(&dC, N * N * 4));
    matmul_tiled<<<dim3(N / 16, N / 16), dim3(16, 16)>>>(dA, dB, dC, N);
    CK(cudaMemcpy(C.data(), dC, N * N * 4, cudaMemcpyDeviceToHost)); bool ok = true;
    for (int r = 0; r < N; r += 9) for (int c = 0; c < N; c += 11) { float e = 0; for (int k = 0; k < N; k++) e += A[r * N + k] * B[k * N + c]; ok &= close_f(C[r * N + c], e); }
    EXPECT(ok, "matmul_shared_tiled"); }
  CK(cudaDeviceSynchronize()); printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; }
