// 2D/3D memcpy, cudaMallocAsync + pools, pinned copies, multi-stream overlap, peer copy, event timing
#include "common.h"
__global__ void k_inc(float* p, int n, float c) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) p[i] += c; }
__global__ void k_spin(float* p, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) { float v = p[i]; for (int k = 0; k < 2000; k++) v = v * 1.0000001f + 1e-7f; p[i] = v; } }
static bool t_memcpy2d() { int W = 50, H = 20; std::vector<float> h(W * H); for (int i = 0; i < W * H; i++) h[i] = (float)i; float* d; size_t pitch;
  CKB(cudaMallocPitch((void**)&d, &pitch, W * 4, H)); CKB(cudaMemcpy2D(d, pitch, h.data(), W * 4, W * 4, H, cudaMemcpyHostToDevice)); std::vector<float> r(W * H, -1.f);
  CKB(cudaMemcpy2D(r.data(), W * 4, d, pitch, W * 4, H, cudaMemcpyDeviceToHost)); if (r != h) return false;
  float* d2; size_t p2; CKB(cudaMallocPitch((void**)&d2, &p2, W * 4, H)); CKB(cudaMemcpy2D(d2, p2, d, pitch, W * 4, H, cudaMemcpyDeviceToDevice)); std::vector<float> r2(W * H);
  CKB(cudaMemcpy2D(r2.data(), W * 4, d2, p2, W * 4, H, cudaMemcpyDeviceToHost)); return r2 == h && pitch >= (size_t)W * 4; }
static bool t_memcpy3d() { int W = 8, H = 6, D = 4; std::vector<float> h(W * H * D); for (size_t i = 0; i < h.size(); i++) h[i] = (float)i;
  cudaExtent ext = make_cudaExtent(W * 4, H, D); cudaPitchedPtr dp; CKB(cudaMalloc3D(&dp, ext)); cudaMemcpy3DParms up = {}; up.srcPtr = make_cudaPitchedPtr(h.data(), W * 4, W, H); up.dstPtr = dp; up.extent = ext; up.kind = cudaMemcpyHostToDevice; CKB(cudaMemcpy3D(&up));
  std::vector<float> r(h.size(), -1.f); cudaMemcpy3DParms dn = {}; dn.srcPtr = dp; dn.dstPtr = make_cudaPitchedPtr(r.data(), W * 4, W, H); dn.extent = ext; dn.kind = cudaMemcpyDeviceToHost; CKB(cudaMemcpy3D(&dn)); return r == h; }
static bool t_malloc_async() { cudaStream_t s; CKB(cudaStreamCreate(&s)); int n = 4096; float* d; CKB(cudaMallocAsync((void**)&d, n * 4, s)); CKB(cudaMemsetAsync(d, 0, n * 4, s));
  k_inc<<<n / 256, 256, 0, s>>>(d, n, 2.f); std::vector<float> r(n); CKB(cudaMemcpyAsync(r.data(), d, n * 4, cudaMemcpyDeviceToHost, s)); CKB(cudaFreeAsync(d, s)); CKB(cudaStreamSynchronize(s));
  for (float v : r) if (v != 2.f) return false; return true; }
static bool t_pool() { cudaMemPool_t pool; CKB(cudaDeviceGetDefaultMemPool(&pool, 0)); unsigned long long thr = ~0ULL; CKB(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
  cudaStream_t s; CKB(cudaStreamCreate(&s)); for (int it = 0; it < 4; it++) { float* d; CKB(cudaMallocFromPoolAsync((void**)&d, 1 << 16, pool, s)); CKB(cudaMemsetAsync(d, 0, 1 << 16, s)); CKB(cudaFreeAsync(d, s)); } CKB(cudaStreamSynchronize(s)); return true; }
static bool t_pinned() { int n = 1 << 16; float *hp, *hq; CKB(cudaHostAlloc((void**)&hp, n * 4, cudaHostAllocDefault)); CKB(cudaMallocHost((void**)&hq, n * 4)); for (int i = 0; i < n; i++) hp[i] = (float)i;
  float* d; CKB(cudaMalloc(&d, n * 4)); cudaStream_t s; CKB(cudaStreamCreate(&s)); CKB(cudaMemcpyAsync(d, hp, n * 4, cudaMemcpyHostToDevice, s)); k_inc<<<n / 256, 256, 0, s>>>(d, n, 1.f);
  CKB(cudaMemcpyAsync(hq, d, n * 4, cudaMemcpyDeviceToHost, s)); CKB(cudaStreamSynchronize(s)); for (int i = 0; i < n; i++) if (hq[i] != i + 1.f) return false;
  void* dp = nullptr; unsigned fl = 0; (void)dp; (void)fl; CKB(cudaFreeHost(hp)); CKB(cudaFreeHost(hq)); return true; }
static bool t_multistream() { const int S = 4, n = 1 << 14; cudaStream_t st[S]; float* d[S]; float* h[S]; for (int i = 0; i < S; i++) { CKB(cudaStreamCreateWithFlags(&st[i], cudaStreamNonBlocking)); CKB(cudaMalloc(&d[i], n * 4)); CKB(cudaMallocHost((void**)&h[i], n * 4)); for (int j = 0; j < n; j++) h[i][j] = (float)i; }
  cudaEvent_t a, b; CKB(cudaEventCreate(&a)); CKB(cudaEventCreate(&b)); CKB(cudaEventRecord(a, 0));
  for (int i = 0; i < S; i++) { CKB(cudaMemcpyAsync(d[i], h[i], n * 4, cudaMemcpyHostToDevice, st[i])); k_spin<<<n / 256, 256, 0, st[i]>>>(d[i], n); k_inc<<<n / 256, 256, 0, st[i]>>>(d[i], n, 10.f); CKB(cudaMemcpyAsync(h[i], d[i], n * 4, cudaMemcpyDeviceToHost, st[i])); }
  CKB(cudaDeviceSynchronize()); CKB(cudaEventRecord(b, 0)); CKB(cudaEventSynchronize(b)); float ms; CKB(cudaEventElapsedTime(&ms, a, b)); printf("  multistream elapsed %.3f ms\n", ms);
  for (int i = 0; i < S; i++) for (int j = 0; j < n; j += 97) if (fabsf(h[i][j] - (i + 10.f)) > 0.05f) return false; return ms >= 0.f; }
static bool t_timing() { float* d; int n = 1 << 18; CKB(cudaMalloc(&d, n * 4)); CKB(cudaMemset(d, 0, n * 4)); cudaEvent_t a, b; CKB(cudaEventCreate(&a)); CKB(cudaEventCreate(&b));
  CKB(cudaEventRecord(a)); for (int i = 0; i < 20; i++) k_spin<<<n / 256, 256>>>(d, n); CKB(cudaEventRecord(b)); CKB(cudaEventSynchronize(b)); float ms = -1; CKB(cudaEventElapsedTime(&ms, a, b)); printf("  20 kernels %.3f ms\n", ms); return ms > 0.001f && ms < 60000.f; }
static bool t_peer() { int cnt = 0; CKB(cudaGetDeviceCount(&cnt)); float *a, *b; int n = 1024; std::vector<float> h(n); for (int i = 0; i < n; i++) h[i] = (float)i; CKB(cudaSetDevice(0)); CKB(cudaMalloc(&a, n * 4)); CKB(cudaMalloc(&b, n * 4));
  CKB(cudaMemcpy(a, h.data(), n * 4, cudaMemcpyHostToDevice)); CKB(cudaMemcpyPeer(b, 0, a, 0, n * 4)); auto r = download(b, n); int can = -1; if (cnt > 1) CKB(cudaDeviceCanAccessPeer(&can, 0, 1)); printf("  devices=%d (peer copy device0->device0)\n", cnt); return r == h; }
int main() { RUN(t_memcpy2d); RUN(t_memcpy3d); RUN(t_malloc_async); RUN(t_pool); RUN(t_pinned); RUN(t_multistream); RUN(t_timing); RUN(t_peer);
  CK(cudaDeviceSynchronize()); printf(g_fail ? "RESULT FAIL\n" : "RESULT OK\n"); return g_fail; }
