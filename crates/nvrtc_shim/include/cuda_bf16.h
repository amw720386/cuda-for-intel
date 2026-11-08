// Minimal cuda_bf16.h: __nv_bfloat16 / __nv_bfloat162 using integer bit tricks (no sm_80 cvt required).
#ifndef NVRTC_SHIM_CUDA_BF16_H
#define NVRTC_SHIM_CUDA_BF16_H
#include "nvrtc_shim_builtins.h"

struct __nv_bfloat16_raw { unsigned short x; };
__NV_HD__ __forceinline__ unsigned short __nvs_f2bf(float f) {
  unsigned u = __builtin_bit_cast(unsigned, f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return (unsigned short)((u >> 16) | 0x40);  // quiet NaN
  u += 0x7fffu + ((u >> 16) & 1u);
  return (unsigned short)(u >> 16);
}
struct __align__(2) __nv_bfloat16 {
  unsigned short __x;
  __NV_HD__ __nv_bfloat16() = default;
  __NV_HD__ __nv_bfloat16(const __nv_bfloat16_raw& r) : __x(r.x) {}
  __NV_HD__ __nv_bfloat16(float f) : __x(__nvs_f2bf(f)) {}
  __NV_HD__ __nv_bfloat16(double f) : __x(__nvs_f2bf((float)f)) {}
  __NV_HD__ __nv_bfloat16(int i) : __x(__nvs_f2bf((float)i)) {}
  __NV_HD__ __nv_bfloat16(unsigned i) : __x(__nvs_f2bf((float)i)) {}
  __NV_HD__ operator float() const { return __builtin_bit_cast(float, (unsigned)__x << 16); }
  __NV_HD__ __nv_bfloat16& operator+=(const __nv_bfloat16& o) { *this = __nv_bfloat16((float)*this + (float)o); return *this; }
  __NV_HD__ __nv_bfloat16& operator*=(const __nv_bfloat16& o) { *this = __nv_bfloat16((float)*this * (float)o); return *this; }
};
struct __align__(4) __nv_bfloat162 {
  __nv_bfloat16 x, y;
  __NV_HD__ __nv_bfloat162() = default;
  __NV_HD__ __nv_bfloat162(const __nv_bfloat16& a, const __nv_bfloat16& b) : x(a), y(b) {}
};
typedef __nv_bfloat16 nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

__NV_HD__ __forceinline__ float __bfloat162float(__nv_bfloat16 h) { return (float)h; }
__NV_HD__ __forceinline__ __nv_bfloat16 __float2bfloat16(float f) { return __nv_bfloat16(f); }
__NV_HD__ __forceinline__ __nv_bfloat16 __float2bfloat16_rn(float f) { return __nv_bfloat16(f); }
__NV_HD__ __forceinline__ __nv_bfloat16 __ushort_as_bfloat16(unsigned short s) { __nv_bfloat16 h; h.__x = s; return h; }
__NV_HD__ __forceinline__ unsigned short __bfloat16_as_ushort(__nv_bfloat16 h) { return h.__x; }
__NV_HD__ __forceinline__ __nv_bfloat16 __hadd(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat16((float)a + (float)b); }
__NV_HD__ __forceinline__ __nv_bfloat16 __hsub(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat16((float)a - (float)b); }
__NV_HD__ __forceinline__ __nv_bfloat16 __hmul(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat16((float)a * (float)b); }
__NV_HD__ __forceinline__ __nv_bfloat16 __hdiv(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat16((float)a / (float)b); }
__NV_HD__ __forceinline__ __nv_bfloat16 __hfma(__nv_bfloat16 a, __nv_bfloat16 b, __nv_bfloat16 c) { return __nv_bfloat16(__builtin_fmaf((float)a, (float)b, (float)c)); }
__NV_HD__ __forceinline__ __nv_bfloat16 operator+(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hadd(a, b); }
__NV_HD__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hsub(a, b); }
__NV_HD__ __forceinline__ __nv_bfloat16 operator*(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hmul(a, b); }
__NV_HD__ __forceinline__ __nv_bfloat16 operator/(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hdiv(a, b); }
__NV_HD__ __forceinline__ __nv_bfloat16 operator-(const __nv_bfloat16& a) { __nv_bfloat16 h; h.__x = a.__x ^ 0x8000; return h; }
__NV_HD__ __forceinline__ bool operator<(const __nv_bfloat16& a, const __nv_bfloat16& b) { return (float)a < (float)b; }
__NV_HD__ __forceinline__ bool operator>(const __nv_bfloat16& a, const __nv_bfloat16& b) { return (float)a > (float)b; }
__NV_HD__ __forceinline__ bool operator==(const __nv_bfloat16& a, const __nv_bfloat16& b) { return (float)a == (float)b; }
__NV_HD__ __forceinline__ __nv_bfloat162 __halves2bfloat162(__nv_bfloat16 a, __nv_bfloat16 b) { return __nv_bfloat162(a, b); }
__NV_HD__ __forceinline__ __nv_bfloat16 __low2bfloat16(__nv_bfloat162 a) { return a.x; }
__NV_HD__ __forceinline__ __nv_bfloat16 __high2bfloat16(__nv_bfloat162 a) { return a.y; }
__NV_HD__ __forceinline__ float2 __bfloat1622float2(__nv_bfloat162 a) { return make_float2((float)a.x, (float)a.y); }
__NV_HD__ __forceinline__ __nv_bfloat162 __floats2bfloat162_rn(float a, float b) { return __nv_bfloat162(__nv_bfloat16(a), __nv_bfloat16(b)); }
__NV_HD__ __forceinline__ __nv_bfloat162 __float22bfloat162_rn(float2 a) { return __nv_bfloat162(__nv_bfloat16(a.x), __nv_bfloat16(a.y)); }
__NV_HD__ __forceinline__ __nv_bfloat162 __hadd2(__nv_bfloat162 a, __nv_bfloat162 b) { return __nv_bfloat162(__hadd(a.x, b.x), __hadd(a.y, b.y)); }
__NV_HD__ __forceinline__ __nv_bfloat162 __hmul2(__nv_bfloat162 a, __nv_bfloat162 b) { return __nv_bfloat162(__hmul(a.x, b.x), __hmul(a.y, b.y)); }
__NV_HD__ __forceinline__ __nv_bfloat162 __hfma2(__nv_bfloat162 a, __nv_bfloat162 b, __nv_bfloat162 c) { return __nv_bfloat162(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)); }
__device__ __forceinline__ __nv_bfloat16 hexp(__nv_bfloat16 a) { return __nv_bfloat16(expf((float)a)); }
__device__ __forceinline__ __nv_bfloat16 hsqrt(__nv_bfloat16 a) { return __nv_bfloat16(__builtin_sqrtf((float)a)); }
__device__ __forceinline__ __nv_bfloat16 hrsqrt(__nv_bfloat16 a) { return __nv_bfloat16(1.0f / __builtin_sqrtf((float)a)); }
__device__ __forceinline__ __nv_bfloat16 atomicAdd(__nv_bfloat16* p, __nv_bfloat16 v) {
  unsigned long long addr = (unsigned long long)p;
  unsigned* w = (unsigned*)(addr & ~3ull);
  unsigned sh = (unsigned)(addr & 2ull) * 8;
  unsigned old = *w, assumed;
  do {
    assumed = old;
    __nv_bfloat16 cur; cur.__x = (unsigned short)(assumed >> sh);
    __nv_bfloat16 nv = __hadd(cur, v);
    unsigned upd = (assumed & ~(0xffffu << sh)) | ((unsigned)nv.__x << sh);
    old = atomicCAS(w, assumed, upd);
  } while (old != assumed);
  __nv_bfloat16 r; r.__x = (unsigned short)(old >> sh); return r;
}
#endif
