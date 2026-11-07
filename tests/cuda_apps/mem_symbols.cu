// constant memory, __device__ globals (cuModuleGetGlobal path), dynamic shared memory, cooperative groups
#include "common.h"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__constant__ float c_coef[16];
__constant__ int c_n = 0;
__device__ int g_counter = 0;
__device__ float g_table[64];
__device__ float g_init = 3.5f;
__global__ void k_const(float* out, const float* in, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float s = 0; for (int k = 0; k < c_n; k++) s += c_coef[k] * in[(i + k) % n]; out[i] = s; } }
__global__ void k_gbl_inc(int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { atomicAdd(&g_counter, 1); atomicAdd(&g_table[i % 64], 1.f); } }
__global__ void k_gbl_read(float* o) { o[0] = g_init; o[1] = (float)g_counter; }
__global__ void k_dynsmem(const float* in, float* out, int n) { extern __shared__ float sh[]; int t = threadIdx.x; sh[t] = in[t]; __syncthreads(); out[t] = sh[n - 1 - t]; }
__global__ void k_dynsmem2(float* out) { extern __shared__ unsigned char raw[]; float* f = (float*)raw; int* iv = (int*)(raw + 64 * sizeof(float)); int t = threadIdx.x;
  f[t] = t * 0.5f; iv[t] = t * 3; __syncthreads(); out[t] = f[(t + 1) % 64] + iv[(t + 2) % 64]; }
__global__ void k_cg(const float* in, float* out) { cg::thread_block blk = cg::this_thread_block(); cg::thread_block_tile<32> tile = cg::tiled_partition<32>(blk);
  float v = in[blk.thread_rank()]; for (int o = tile.size() / 2; o > 0; o >>= 1) v += tile.shfl_down(v, o);
  if (tile.thread_rank() == 0) out[tile.meta_group_rank()] = v; blk.sync(); if (blk.thread_rank() == 0) out[8] = (float)blk.size(); }
static bool t_const() { float c[16]; for (int i = 0; i < 16; i++) c[i] = 0.5f + i; int k = 5; CKB(cudaMemcpyToSymbol(c_coef, c, sizeof c)); CKB(cudaMemcpyToSymbol(c_n, &k, sizeof k));
  int n = 100; std::vector<float> h(n); for (int i = 0; i < n; i++) h[i] = i * 0.1f; float* d = upload(h); float* o; CKB(cudaMalloc(&o, n * 4)); k_const<<<1, 128>>>(o, d, n); CKB(cudaGetLastError()); auto r = download(o, n);
  for (int i = 0; i < n; i++) { float s = 0; for (int j = 0; j < k; j++) s += c[j] * h[(i + j) % n]; if (fabsf(r[i] - s) > 1e-3f) return false; } return true; }
static bool t_globals() { int zero = 0; std::vector<float> z(64, 0.f); CKB(cudaMemcpyToSymbol(g_counter, &zero, 4)); CKB(cudaMemcpyToSymbol(g_table, z.data(), 256)); k_gbl_inc<<<4, 128>>>(512); CKB(cudaGetLastError());
  int c; CKB(cudaMemcpyFromSymbol(&c, g_counter, 4)); std::vector<float> t(64); CKB(cudaMemcpyFromSymbol(t.data(), g_table, 256)); if (c != 512) return false; for (float v : t) if (v != 8.f) return false;
  float* o; CKB(cudaMalloc(&o, 8)); k_gbl_read<<<1, 1>>>(o); float r[2]; CKB(cudaMemcpy(r, o, 8, cudaMemcpyDeviceToHost)); void* p = nullptr; CKB(cudaGetSymbolAddress(&p, g_counter)); size_t sz = 0; CKB(cudaGetSymbolSize(&sz, g_table));
  return r[0] == 3.5f && r[1] == 512.f && p && sz == 256; }
static bool t_dynsmem() { int n = 64; std::vector<float> h(n); for (int i = 0; i < n; i++) h[i] = (float)i; float* d = upload(h); float* o; CKB(cudaMalloc(&o, 256)); k_dynsmem<<<1, 64, 64 * 4>>>(d, o, n); CKB(cudaGetLastError());
  auto r = download(o, n); for (int i = 0; i < n; i++) if (r[i] != h[n - 1 - i]) return false;
  k_dynsmem2<<<1, 64, 64 * 4 + 64 * 4>>>(o); CKB(cudaGetLastError()); r = download(o, n); for (int t = 0; t < 64; t++) if (r[t] != ((t + 1) % 64) * 0.5f + ((t + 2) % 64) * 3) return false; return true; }
static bool t_cg() { std::vector<float> h(128); for (int i = 0; i < 128; i++) h[i] = (float)(i + 1); float* d = upload(h); float* o; CKB(cudaMalloc(&o, 36)); k_cg<<<1, 128>>>(d, o); CKB(cudaGetLastError()); auto r = download(o, 9);
  for (int w = 0; w < 4; w++) { float s = 0; for (int i = 0; i < 32; i++) s += h[w * 32 + i]; if (r[w] != s) return false; } return r[8] == 128.f; }
int main() { RUN(t_const); RUN(t_globals); RUN(t_dynsmem); RUN(t_cg);
  CK(cudaDeviceSynchronize()); printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; }
