#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <climits>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1);} } while (0)
static int g_fail = 0;
#define EXPECT(c, name) do { if (c) printf("PASS %s\n", name); else { printf("FAIL %s\n", name); g_fail++; } } while (0)
static inline bool close_f(float a, float b, float rt = 1e-3f) { return fabsf(a - b) <= rt * (1.f + fabsf(b)); }
#ifdef _WIN32
// Newer cudart loads nvcuda.dll with a restricted search path (System32 only), so map the shim's
// nvcuda.dll from the exe directory first; later LoadLibrary("nvcuda.dll") by name reuses it.
#include <windows.h>
static struct PreloadShim { PreloadShim() { char p[MAX_PATH]; DWORD n = GetModuleFileNameA(NULL, p, MAX_PATH);
  while (n && p[n - 1] != '\\') n--; strcpy(p + n, "nvcuda.dll"); if (!LoadLibraryA(p)) printf("preload nvcuda.dll failed (%lu)\n", GetLastError()); } } g_preload_shim;
#endif
// Non-fatal check for use inside bool-returning test functions.
#define CKB(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("  api error %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); return false; } } while (0)
#define RUN(fn) do { bool ok_ = fn(); EXPECT(ok_, #fn); } while (0)
template <class T> static inline T* upload(const std::vector<T>& h) { T* d = nullptr; if (cudaMalloc(&d, h.size() * sizeof(T)) != cudaSuccess) return nullptr;
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice); return d; }
template <class T> static inline std::vector<T> download(const T* d, size_t n) { std::vector<T> h(n); cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost); return h; }
