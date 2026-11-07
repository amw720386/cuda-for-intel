// __half / __half2 / __nv_bfloat16 kernels and conversions vs CPU
#include "common.h"
#include <cuda_fp16.h>
#include <cuda_bf16.h>
__global__ void k_half_axpy(const __half* x, const __half* y, __half* o, __half a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = __hfma(a, x[i], y[i]); }
__global__ void k_half2(const __half2* x, const __half2* y, __half2* o, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = __hadd2(__hmul2(x[i], y[i]), x[i]); }
__global__ void k_half_cvt(const float* in, __half* h, float* back, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { h[i] = __float2half(in[i]); back[i] = __half2float(h[i]); } }
__global__ void k_half_dot(const __half* a, const __half* b, float* out, int n) { float s = 0; for (int i = threadIdx.x; i < n; i += blockDim.x) s += __half2float(a[i]) * __half2float(b[i]);
  for (int o = 16; o; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o); if (threadIdx.x == 0) *out = s; }
__global__ void k_bf16(const float* in, __nv_bfloat16* b, float* back, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { b[i] = __float2bfloat16(in[i]); back[i] = __bfloat162float(b[i]); } }
__global__ void k_bf16_mul(const __nv_bfloat16* a, const __nv_bfloat16* b, float* o, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) o[i] = __bfloat162float(a[i]) * __bfloat162float(b[i]); }
static float h2f(float f) { return __half2float(__float2half(f)); }
static float b2f(float f) { return __bfloat162float(__float2bfloat16(f)); }
static bool t_half_axpy() { int n = 1000; std::vector<__half> x(n), y(n); std::vector<float> fx(n), fy(n); for (int i = 0; i < n; i++) { fx[i] = h2f((i % 17) * 0.125f); fy[i] = h2f((i % 5) - 2.f); x[i] = __float2half(fx[i]); y[i] = __float2half(fy[i]); }
  __half *dx = upload(x), *dy = upload(y), *o; CKB(cudaMalloc(&o, n * 2)); k_half_axpy<<<(n + 255) / 256, 256>>>(dx, dy, o, __float2half(0.5f), n); CKB(cudaGetLastError());
  auto r = download(o, n); for (int i = 0; i < n; i++) if (fabsf(__half2float(r[i]) - h2f(0.5f * fx[i] + fy[i])) > 2e-3f) return false; return true; }
static bool t_half2() { int n = 256; std::vector<__half2> x(n), y(n); std::vector<float> f(4 * n); for (int i = 0; i < n; i++) { float a = h2f(i * 0.03125f), b = h2f(1.f - i * 0.0078125f), c = h2f(0.5f + (i % 7)), d = h2f(-0.25f * (i % 3));
    x[i] = __floats2half2_rn(a, b); y[i] = __floats2half2_rn(c, d); f[4*i]=a; f[4*i+1]=b; f[4*i+2]=c; f[4*i+3]=d; }
  __half2 *dx = upload(x), *dy = upload(y), *o; CKB(cudaMalloc(&o, n * 4)); k_half2<<<1, 256>>>(dx, dy, o, n); CKB(cudaGetLastError()); auto r = download(o, n);
  for (int i = 0; i < n; i++) { float e0 = f[4*i]*f[4*i+2] + f[4*i], e1 = f[4*i+1]*f[4*i+3] + f[4*i+1]; if (fabsf(__low2float(r[i]) - e0) > 0.02f + 2e-3f*fabsf(e0) || fabsf(__high2float(r[i]) - e1) > 0.02f + 2e-3f*fabsf(e1)) return false; } return true; }
static bool t_half_cvt() { int n = 512; std::vector<float> f(n); for (int i = 0; i < n; i++) f[i] = (i - 256) * 0.37f; float* d = upload(f); __half* h; float* b; CKB(cudaMalloc(&h, n * 2)); CKB(cudaMalloc(&b, n * 4));
  k_half_cvt<<<2, 256>>>(d, h, b, n); CKB(cudaGetLastError()); auto r = download(b, n); auto hh = download(h, n); for (int i = 0; i < n; i++) { if (r[i] != h2f(f[i])) return false; if (__half_as_ushort(hh[i]) != __half_as_ushort(__float2half(f[i]))) return false; } return true; }
static bool t_half_dot() { int n = 1024; std::vector<__half> a(n), b(n); double e = 0; for (int i = 0; i < n; i++) { float x = h2f((i % 9) * 0.25f), y = h2f(((i % 4) - 1.5f)); a[i] = __float2half(x); b[i] = __float2half(y); e += x * y; }
  __half *da = upload(a), *db = upload(b); float* o; CKB(cudaMalloc(&o, 4)); k_half_dot<<<1, 32>>>(da, db, o, n); float r; CKB(cudaMemcpy(&r, o, 4, cudaMemcpyDeviceToHost)); return fabs(r - e) < 1e-2 * (1 + fabs(e)); }
static bool t_bf16() { int n = 512; std::vector<float> f(n); for (int i = 0; i < n; i++) f[i] = (i - 100) * 0.0731f; float* d = upload(f); __nv_bfloat16* b; float* o; CKB(cudaMalloc(&b, n * 2)); CKB(cudaMalloc(&o, n * 4));
  k_bf16<<<2, 256>>>(d, b, o, n); CKB(cudaGetLastError()); auto r = download(o, n); for (int i = 0; i < n; i++) if (r[i] != b2f(f[i])) return false;
  k_bf16_mul<<<2, 256>>>(b, b, o, n); r = download(o, n); for (int i = 0; i < n; i++) if (fabsf(r[i] - b2f(f[i]) * b2f(f[i])) > 1e-3f * fabsf(r[i]) + 1e-6f) return false; return true; }
int main() { RUN(t_half_axpy); RUN(t_half2); RUN(t_half_cvt); RUN(t_half_dot); RUN(t_bf16);
  CK(cudaDeviceSynchronize()); printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; }
