// Small pre-LN transformer block forward: cublasSgemm projections + custom LN/attention/GELU kernels vs CPU
#include "common.h"
#include <cublas_v2.h>
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { printf("FAIL %s:%d %s status=%d\n", __FILE__, __LINE__, #x, (int)s_); exit(1);} } while (0)
static const int T = 8, D = 32, H = 2, DH = 16, F = 4 * D;
__global__ void k_ln(const float* x, const float* g, const float* b, float* o) { int r = blockIdx.x, t = threadIdx.x; __shared__ float m, v; if (t == 0) { float s = 0; for (int i = 0; i < D; i++) s += x[r * D + i]; m = s / D; float q = 0; for (int i = 0; i < D; i++) { float d = x[r * D + i] - m; q += d * d; } v = rsqrtf(q / D + 1e-5f); }
  __syncthreads(); o[r * D + t] = (x[r * D + t] - m) * v * g[t] + b[t]; }
// qkv: [T, 3D] row-major; out: [T, D]; one block per (head, query), causal
__global__ void k_attn(const float* qkv, float* out) { int h = blockIdx.x, q = blockIdx.y, t = threadIdx.x; __shared__ float p[T]; __shared__ float mx, sm;
  if (t < T) { float s = -1e30f; if (t <= q) { s = 0; for (int i = 0; i < DH; i++) s += qkv[q * 3 * D + h * DH + i] * qkv[t * 3 * D + D + h * DH + i]; s *= rsqrtf((float)DH); } p[t] = s; } __syncthreads();
  if (t == 0) { float m = -1e30f; for (int i = 0; i < T; i++) m = fmaxf(m, p[i]); float z = 0; for (int i = 0; i < T; i++) { p[i] = expf(p[i] - m); z += p[i]; } mx = m; sm = z; } __syncthreads();
  if (t < DH) { float a = 0; for (int i = 0; i <= q; i++) a += p[i] / sm * qkv[i * 3 * D + 2 * D + h * DH + t]; out[q * D + h * DH + t] = a; } }
__global__ void k_gelu(float* x, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float v = x[i]; x[i] = 0.5f * v * (1.f + tanhf(0.79788456f * (v + 0.044715f * v * v * v))); } }
__global__ void k_add_bias_res(float* y, const float* bias, const float* res, int rows, int cols) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < rows * cols) y[i] += bias[i % cols] + (res ? res[i] : 0.f); }
// row-major C[M,N] = A[M,K] * B[K,N] using column-major cublas
static void mm(cublasHandle_t h, const float* A, const float* B, float* C, int M, int K, int N) { float one = 1, zero = 0; CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &one, B, N, A, K, &zero, C, N)); }
static std::vector<float> rnd(int n, float sc, int seed) { std::vector<float> v(n); unsigned s = seed; for (auto& x : v) { s = s * 1664525u + 1013904223u; x = ((s >> 8) & 0xffff) / 65536.f * 2 * sc - sc; } return v; }
static void cpu_mm(const float* A, const float* B, float* C, int M, int K, int N) { for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) { double s = 0; for (int k = 0; k < K; k++) s += (double)A[i * K + k] * B[k * N + j]; C[i * N + j] = (float)s; } }
static void cpu_ln(const float* x, const float* g, const float* b, float* o) { for (int r = 0; r < T; r++) { double m = 0, v = 0; for (int i = 0; i < D; i++) m += x[r * D + i]; m /= D; for (int i = 0; i < D; i++) v += (x[r * D + i] - m) * (x[r * D + i] - m); v /= D; for (int i = 0; i < D; i++) o[r * D + i] = (float)((x[r * D + i] - m) / sqrt(v + 1e-5) * g[i] + b[i]); } }
int main() { CK(cudaFree(0)); cublasHandle_t cb; CB(cublasCreate(&cb));
  auto x = rnd(T * D, 1, 1), g1 = rnd(D, 0.2f, 2), b1 = rnd(D, 0.1f, 3), g2 = rnd(D, 0.2f, 4), b2 = rnd(D, 0.1f, 5), Wqkv = rnd(D * 3 * D, 0.15f, 6), Wo = rnd(D * D, 0.15f, 7), bo = rnd(D, 0.1f, 8), W1 = rnd(D * F, 0.15f, 9), bb1 = rnd(F, 0.1f, 10), W2 = rnd(F * D, 0.1f, 11), bb2 = rnd(D, 0.1f, 12);
  for (auto& v : g1) v += 1; for (auto& v : g2) v += 1;
  // CPU reference
  std::vector<float> n1(T * D), qkv(T * 3 * D), att(T * D, 0.f), ao(T * D), x1(T * D), n2(T * D), hid(T * F), mo(T * D), y(T * D);
  cpu_ln(x.data(), g1.data(), b1.data(), n1.data()); cpu_mm(n1.data(), Wqkv.data(), qkv.data(), T, D, 3 * D);
  for (int h = 0; h < H; h++) for (int q = 0; q < T; q++) { std::vector<double> p(T); double m = -1e30; for (int t = 0; t <= q; t++) { double s = 0; for (int i = 0; i < DH; i++) s += (double)qkv[q * 3 * D + h * DH + i] * qkv[t * 3 * D + D + h * DH + i]; p[t] = s / sqrt((double)DH); m = std::max(m, p[t]); }
    double z = 0; for (int t = 0; t <= q; t++) { p[t] = exp(p[t] - m); z += p[t]; } for (int i = 0; i < DH; i++) { double a = 0; for (int t = 0; t <= q; t++) a += p[t] / z * qkv[t * 3 * D + 2 * D + h * DH + i]; att[q * D + h * DH + i] = (float)a; } }
  cpu_mm(att.data(), Wo.data(), ao.data(), T, D, D); for (int i = 0; i < T * D; i++) x1[i] = x[i] + ao[i] + bo[i % D];
  cpu_ln(x1.data(), g2.data(), b2.data(), n2.data()); cpu_mm(n2.data(), W1.data(), hid.data(), T, D, F);
  for (int i = 0; i < T * F; i++) { double v = hid[i] + bb1[i % F]; hid[i] = (float)(0.5 * v * (1 + tanh(0.7978845608 * (v + 0.044715 * v * v * v)))); }
  cpu_mm(hid.data(), W2.data(), mo.data(), T, F, D); for (int i = 0; i < T * D; i++) y[i] = x1[i] + mo[i] + bb2[i % D];
  // GPU
  float *dx = upload(x), *dg1 = upload(g1), *db1 = upload(b1), *dg2 = upload(g2), *db2 = upload(b2), *dWqkv = upload(Wqkv), *dWo = upload(Wo), *dbo = upload(bo), *dW1 = upload(W1), *dbb1 = upload(bb1), *dW2 = upload(W2), *dbb2 = upload(bb2);
  float *dn1, *dqkv, *datt, *dao, *dx1, *dn2, *dhid, *dmo; CK(cudaMalloc(&dn1, T*D*4)); CK(cudaMalloc(&dqkv, T*3*D*4)); CK(cudaMalloc(&datt, T*D*4)); CK(cudaMalloc(&dao, T*D*4)); CK(cudaMalloc(&dx1, T*D*4)); CK(cudaMalloc(&dn2, T*D*4)); CK(cudaMalloc(&dhid, T*F*4)); CK(cudaMalloc(&dmo, T*D*4));
  k_ln<<<T, D>>>(dx, dg1, db1, dn1); mm(cb, dn1, dWqkv, dqkv, T, D, 3 * D); k_attn<<<dim3(H, T), 32>>>(dqkv, datt); mm(cb, datt, dWo, dao, T, D, D);
  CK(cudaMemcpy(dx1, dao, T*D*4, cudaMemcpyDeviceToDevice)); k_add_bias_res<<<1, 256>>>(dx1, dbo, dx, T, D);
  k_ln<<<T, D>>>(dx1, dg2, db2, dn2); mm(cb, dn2, dW1, dhid, T, D, F); k_add_bias_res<<<(T*F+255)/256, 256>>>(dhid, dbb1, nullptr, T, F); k_gelu<<<(T*F+255)/256, 256>>>(dhid, T*F);
  mm(cb, dhid, dW2, dmo, T, F, D); k_add_bias_res<<<1, 256>>>(dmo, dbb2, dx1, T, D); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  auto r = download(dmo, T * D); double maxerr = 0; for (int i = 0; i < T * D; i++) maxerr = std::max(maxerr, (double)fabsf(r[i] - y[i]));
  printf("  max abs err vs CPU = %.3g\n", maxerr); EXPECT(maxerr < 2e-3, "transformer_block_forward");
  printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; }
