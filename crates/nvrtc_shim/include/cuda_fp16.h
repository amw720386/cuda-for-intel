// Minimal cuda_fp16.h: __half / __half2 with float-based arithmetic (cvt.f32.f16 / cvt.rn.f16.f32 only).
#ifndef NVRTC_SHIM_CUDA_FP16_H
#define NVRTC_SHIM_CUDA_FP16_H
#include "nvrtc_shim_builtins.h"

struct __half_raw { unsigned short x; };
struct __half2_raw { unsigned short x, y; };

struct __align__(2) __half {
  unsigned short __x;
  __NV_HD__ __half() = default;
  __NV_HD__ __half(const __half_raw& r) : __x(r.x) {}
  __NV_HD__ __half(float f) { _Float16 h = (_Float16)f; __x = __builtin_bit_cast(unsigned short, h); }
  __NV_HD__ __half(double f) { _Float16 h = (_Float16)(float)f; __x = __builtin_bit_cast(unsigned short, h); }
  __NV_HD__ __half(int i) { _Float16 h = (_Float16)(float)i; __x = __builtin_bit_cast(unsigned short, h); }
  __NV_HD__ __half(unsigned i) { _Float16 h = (_Float16)(float)i; __x = __builtin_bit_cast(unsigned short, h); }
  __NV_HD__ __half(short i) { _Float16 h = (_Float16)(float)i; __x = __builtin_bit_cast(unsigned short, h); }
  __NV_HD__ __half(long long i) { _Float16 h = (_Float16)(float)i; __x = __builtin_bit_cast(unsigned short, h); }
  __NV_HD__ operator float() const { return (float)__builtin_bit_cast(_Float16, __x); }
  __NV_HD__ operator __half_raw() const { __half_raw r; r.x = __x; return r; }
  __NV_HD__ __half& operator=(const __half_raw& r) { __x = r.x; return *this; }
  __NV_HD__ __half& operator=(float f) { *this = __half(f); return *this; }
  __NV_HD__ __half& operator+=(const __half& o) { *this = __half((float)*this + (float)o); return *this; }
  __NV_HD__ __half& operator-=(const __half& o) { *this = __half((float)*this - (float)o); return *this; }
  __NV_HD__ __half& operator*=(const __half& o) { *this = __half((float)*this * (float)o); return *this; }
  __NV_HD__ __half& operator/=(const __half& o) { *this = __half((float)*this / (float)o); return *this; }
};
struct __align__(4) __half2 {
  __half x, y;
  __NV_HD__ __half2() = default;
  __NV_HD__ __half2(const __half& a, const __half& b) : x(a), y(b) {}
  __NV_HD__ __half2(const __half2_raw& r) { x.__x = r.x; y.__x = r.y; }
};
typedef __half half;
typedef __half2 half2;

#define __NVS_H1(name, expr) __NV_HD__ __forceinline__ __half name(__half a) { float x = (float)a; return __half((float)(expr)); }
__NV_HD__ __forceinline__ float __half2float(__half h) { return (float)h; }
__NV_HD__ __forceinline__ __half __float2half(float f) { return __half(f); }
__NV_HD__ __forceinline__ __half __float2half_rn(float f) { return __half(f); }
__NV_HD__ __forceinline__ __half __double2half(double f) { return __half((float)f); }
__NV_HD__ __forceinline__ __half __int2half_rn(int i) { return __half((float)i); }
__NV_HD__ __forceinline__ __half __uint2half_rn(unsigned i) { return __half((float)i); }
__NV_HD__ __forceinline__ __half __short2half_rn(short i) { return __half((float)i); }
__NV_HD__ __forceinline__ int __half2int_rn(__half h) { return (int)__builtin_rintf((float)h); }
__NV_HD__ __forceinline__ int __half2int_rz(__half h) { return (int)(float)h; }
__NV_HD__ __forceinline__ short __half2short_rn(__half h) { return (short)__builtin_rintf((float)h); }
__NV_HD__ __forceinline__ __half __ushort_as_half(unsigned short s) { __half h; h.__x = s; return h; }
__NV_HD__ __forceinline__ __half __short_as_half(short s) { __half h; h.__x = (unsigned short)s; return h; }
__NV_HD__ __forceinline__ unsigned short __half_as_ushort(__half h) { return h.__x; }
__NV_HD__ __forceinline__ short __half_as_short(__half h) { return (short)h.__x; }
__NV_HD__ __forceinline__ __half __hadd(__half a, __half b) { return __half((float)a + (float)b); }
__NV_HD__ __forceinline__ __half __hsub(__half a, __half b) { return __half((float)a - (float)b); }
__NV_HD__ __forceinline__ __half __hmul(__half a, __half b) { return __half((float)a * (float)b); }
__NV_HD__ __forceinline__ __half __hdiv(__half a, __half b) { return __half((float)a / (float)b); }
__NV_HD__ __forceinline__ __half __hfma(__half a, __half b, __half c) { return __half(__builtin_fmaf((float)a, (float)b, (float)c)); }
__NV_HD__ __forceinline__ __half __hneg(__half a) { __half h; h.__x = a.__x ^ 0x8000; return h; }
__NV_HD__ __forceinline__ __half __habs(__half a) { __half h; h.__x = a.__x & 0x7fff; return h; }
__NV_HD__ __forceinline__ __half operator+(const __half& a, const __half& b) { return __hadd(a, b); }
__NV_HD__ __forceinline__ __half operator-(const __half& a, const __half& b) { return __hsub(a, b); }
__NV_HD__ __forceinline__ __half operator*(const __half& a, const __half& b) { return __hmul(a, b); }
__NV_HD__ __forceinline__ __half operator/(const __half& a, const __half& b) { return __hdiv(a, b); }
__NV_HD__ __forceinline__ __half operator-(const __half& a) { return __hneg(a); }
__NV_HD__ __forceinline__ __half operator+(const __half& a) { return a; }
__NV_HD__ __forceinline__ bool __heq(__half a, __half b) { return (float)a == (float)b; }
__NV_HD__ __forceinline__ bool __hne(__half a, __half b) { return (float)a != (float)b; }
__NV_HD__ __forceinline__ bool __hlt(__half a, __half b) { return (float)a < (float)b; }
__NV_HD__ __forceinline__ bool __hle(__half a, __half b) { return (float)a <= (float)b; }
__NV_HD__ __forceinline__ bool __hgt(__half a, __half b) { return (float)a > (float)b; }
__NV_HD__ __forceinline__ bool __hge(__half a, __half b) { return (float)a >= (float)b; }
__NV_HD__ __forceinline__ bool operator==(const __half& a, const __half& b) { return __heq(a, b); }
__NV_HD__ __forceinline__ bool operator!=(const __half& a, const __half& b) { return __hne(a, b); }
__NV_HD__ __forceinline__ bool operator<(const __half& a, const __half& b) { return __hlt(a, b); }
__NV_HD__ __forceinline__ bool operator<=(const __half& a, const __half& b) { return __hle(a, b); }
__NV_HD__ __forceinline__ bool operator>(const __half& a, const __half& b) { return __hgt(a, b); }
__NV_HD__ __forceinline__ bool operator>=(const __half& a, const __half& b) { return __hge(a, b); }
__NV_HD__ __forceinline__ bool __hisnan(__half a) { return (a.__x & 0x7fff) > 0x7c00; }
__NV_HD__ __forceinline__ bool __hisinf(__half a) { return (a.__x & 0x7fff) == 0x7c00; }
__NV_HD__ __forceinline__ __half __hmax(__half a, __half b) { return (float)a > (float)b ? a : b; }
__NV_HD__ __forceinline__ __half __hmin(__half a, __half b) { return (float)a < (float)b ? a : b; }
__NV_HD__ __forceinline__ __half __hmax_nan(__half a, __half b) { return __hisnan(a) ? a : (__hisnan(b) ? b : __hmax(a, b)); }
__NV_HD__ __forceinline__ __half __hmin_nan(__half a, __half b) { return __hisnan(a) ? a : (__hisnan(b) ? b : __hmin(a, b)); }
__NV_HD__ __forceinline__ __half __hsat(__half a) { float f = (float)a; return __half(f < 0.f ? 0.f : (f > 1.f ? 1.f : f)); }
__device__ __forceinline__ __half hexp(__half a) { return __half(expf((float)a)); }
__device__ __forceinline__ __half hexp2(__half a) { return __half(exp2f((float)a)); }
__device__ __forceinline__ __half hlog(__half a) { return __half(logf((float)a)); }
__device__ __forceinline__ __half hlog2(__half a) { return __half(log2f((float)a)); }
__device__ __forceinline__ __half hsin(__half a) { return __half(sinf((float)a)); }
__device__ __forceinline__ __half hcos(__half a) { return __half(cosf((float)a)); }
__device__ __forceinline__ __half hsqrt(__half a) { return __half(__builtin_sqrtf((float)a)); }
__device__ __forceinline__ __half hrsqrt(__half a) { return __half(1.0f / __builtin_sqrtf((float)a)); }
__device__ __forceinline__ __half hrcp(__half a) { return __half(1.0f / (float)a); }
__device__ __forceinline__ __half hfloor(__half a) { return __half(__builtin_floorf((float)a)); }
__device__ __forceinline__ __half hceil(__half a) { return __half(__builtin_ceilf((float)a)); }
__device__ __forceinline__ __half htrunc(__half a) { return __half(__builtin_truncf((float)a)); }
__device__ __forceinline__ __half hrint(__half a) { return __half(__builtin_rintf((float)a)); }

// half2
__NV_HD__ __forceinline__ __half2 __halves2half2(__half a, __half b) { return __half2(a, b); }
__NV_HD__ __forceinline__ __half2 __half2half2(__half a) { return __half2(a, a); }
__NV_HD__ __forceinline__ __half2 make_half2(__half a, __half b) { return __half2(a, b); }
__NV_HD__ __forceinline__ __half __low2half(__half2 a) { return a.x; }
__NV_HD__ __forceinline__ __half __high2half(__half2 a) { return a.y; }
__NV_HD__ __forceinline__ __half2 __lows2half2(__half2 a, __half2 b) { return __half2(a.x, b.x); }
__NV_HD__ __forceinline__ __half2 __highs2half2(__half2 a, __half2 b) { return __half2(a.y, b.y); }
__NV_HD__ __forceinline__ __half2 __lowhigh2highlow(__half2 a) { return __half2(a.y, a.x); }
__NV_HD__ __forceinline__ float __low2float(__half2 a) { return (float)a.x; }
__NV_HD__ __forceinline__ float __high2float(__half2 a) { return (float)a.y; }
__NV_HD__ __forceinline__ float2 __half22float2(__half2 a) { return make_float2((float)a.x, (float)a.y); }
__NV_HD__ __forceinline__ __half2 __float22half2_rn(float2 a) { return __half2(__half(a.x), __half(a.y)); }
__NV_HD__ __forceinline__ __half2 __floats2half2_rn(float a, float b) { return __half2(__half(a), __half(b)); }
__NV_HD__ __forceinline__ __half2 __float2half2_rn(float a) { return __half2(__half(a), __half(a)); }
__NV_HD__ __forceinline__ unsigned __half2_as_uint(__half2 a) { return (unsigned)a.x.__x | ((unsigned)a.y.__x << 16); }
__NV_HD__ __forceinline__ __half2 __uint_as_half2(unsigned u) { __half2_raw r; r.x = (unsigned short)u; r.y = (unsigned short)(u >> 16); return __half2(r); }
__NV_HD__ __forceinline__ __half2 __hadd2(__half2 a, __half2 b) { return __half2(__hadd(a.x, b.x), __hadd(a.y, b.y)); }
__NV_HD__ __forceinline__ __half2 __hsub2(__half2 a, __half2 b) { return __half2(__hsub(a.x, b.x), __hsub(a.y, b.y)); }
__NV_HD__ __forceinline__ __half2 __hmul2(__half2 a, __half2 b) { return __half2(__hmul(a.x, b.x), __hmul(a.y, b.y)); }
__NV_HD__ __forceinline__ __half2 __h2div(__half2 a, __half2 b) { return __half2(__hdiv(a.x, b.x), __hdiv(a.y, b.y)); }
__NV_HD__ __forceinline__ __half2 __hfma2(__half2 a, __half2 b, __half2 c) { return __half2(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)); }
__NV_HD__ __forceinline__ __half2 __hneg2(__half2 a) { return __half2(__hneg(a.x), __hneg(a.y)); }
__NV_HD__ __forceinline__ __half2 operator+(const __half2& a, const __half2& b) { return __hadd2(a, b); }
__NV_HD__ __forceinline__ __half2 operator-(const __half2& a, const __half2& b) { return __hsub2(a, b); }
__NV_HD__ __forceinline__ __half2 operator*(const __half2& a, const __half2& b) { return __hmul2(a, b); }
__NV_HD__ __forceinline__ __half2 operator/(const __half2& a, const __half2& b) { return __h2div(a, b); }
__NV_HD__ __forceinline__ __half2& operator+=(__half2& a, const __half2& b) { a = __hadd2(a, b); return a; }
__NV_HD__ __forceinline__ __half2& operator*=(__half2& a, const __half2& b) { a = __hmul2(a, b); return a; }
__NV_HD__ __forceinline__ __half2 __hmax2(__half2 a, __half2 b) { return __half2(__hmax(a.x, b.x), __hmax(a.y, b.y)); }
__NV_HD__ __forceinline__ __half2 __hmin2(__half2 a, __half2 b) { return __half2(__hmin(a.x, b.x), __hmin(a.y, b.y)); }
__device__ __forceinline__ __half2 h2exp(__half2 a) { return __half2(hexp(a.x), hexp(a.y)); }
__device__ __forceinline__ __half2 h2sqrt(__half2 a) { return __half2(hsqrt(a.x), hsqrt(a.y)); }
__device__ __forceinline__ __half2 h2rsqrt(__half2 a) { return __half2(hrsqrt(a.x), hrsqrt(a.y)); }
__device__ __forceinline__ __half2 h2rcp(__half2 a) { return __half2(hrcp(a.x), hrcp(a.y)); }

// shuffles and atomics
__device__ __forceinline__ __half __shfl_idx_sync(unsigned m, __half v, int a, int w = 32) { __half r; r.__x = (unsigned short)__nvs_shfl_idx(m, (int)v.__x, a, w); return r; }
__device__ __forceinline__ __half __shfl_down_sync(unsigned m, __half v, int a, int w = 32) { __half r; r.__x = (unsigned short)__nvs_shfl_down(m, (int)v.__x, a, w); return r; }
__device__ __forceinline__ __half __shfl_up_sync(unsigned m, __half v, int a, int w = 32) { __half r; r.__x = (unsigned short)__nvs_shfl_up(m, (int)v.__x, a, w); return r; }
__device__ __forceinline__ __half __shfl_xor_sync(unsigned m, __half v, int a, int w = 32) { __half r; r.__x = (unsigned short)__nvs_shfl_xor(m, (int)v.__x, a, w); return r; }
__device__ __forceinline__ __half2 __shfl_idx_sync(unsigned m, __half2 v, int a, int w = 32) { return __uint_as_half2((unsigned)__nvs_shfl_idx(m, (int)__half2_as_uint(v), a, w)); }
__device__ __forceinline__ __half2 __shfl_down_sync(unsigned m, __half2 v, int a, int w = 32) { return __uint_as_half2((unsigned)__nvs_shfl_down(m, (int)__half2_as_uint(v), a, w)); }
__device__ __forceinline__ __half2 __shfl_up_sync(unsigned m, __half2 v, int a, int w = 32) { return __uint_as_half2((unsigned)__nvs_shfl_up(m, (int)__half2_as_uint(v), a, w)); }
__device__ __forceinline__ __half2 __shfl_xor_sync(unsigned m, __half2 v, int a, int w = 32) { return __uint_as_half2((unsigned)__nvs_shfl_xor(m, (int)__half2_as_uint(v), a, w)); }

// atomicAdd on 16-bit halves: CAS loop on the containing 32-bit word.
__device__ __forceinline__ __half atomicAdd(__half* p, __half v) {
  unsigned long long addr = (unsigned long long)p;
  unsigned* w = (unsigned*)(addr & ~3ull);
  unsigned sh = (unsigned)(addr & 2ull) * 8;
  unsigned old = *w, assumed;
  do {
    assumed = old;
    __half cur; cur.__x = (unsigned short)(assumed >> sh);
    __half nv = __hadd(cur, v);
    unsigned upd = (assumed & ~(0xffffu << sh)) | ((unsigned)nv.__x << sh);
    old = atomicCAS(w, assumed, upd);
  } while (old != assumed);
  __half r; r.__x = (unsigned short)(old >> sh); return r;
}
__device__ __forceinline__ __half2 atomicAdd(__half2* p, __half2 v) {
  unsigned* w = (unsigned*)p;
  unsigned old = *w, assumed;
  do { assumed = old; old = atomicCAS(w, assumed, __half2_as_uint(__hadd2(__uint_as_half2(assumed), v))); } while (old != assumed);
  return __uint_as_half2(old);
}
#endif
