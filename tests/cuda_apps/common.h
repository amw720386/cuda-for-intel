#pragma once
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s:%d %s -> %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1);} } while (0)
static int g_fail = 0;
#define EXPECT(c, name) do { if (c) printf("PASS %s\n", name); else { printf("FAIL %s\n", name); g_fail++; } } while (0)
static inline bool close_f(float a, float b, float rt = 1e-3f) { return fabsf(a - b) <= rt * (1.f + fabsf(b)); }
