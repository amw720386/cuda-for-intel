// Core CUDA runtime subset force-included into every NVRTC-shim compile.
// Replaces the headers real NVRTC embeds; written for clang's NVPTX backend (no libdevice).
#ifndef NVRTC_SHIM_BUILTINS_H
#define NVRTC_SHIM_BUILTINS_H

#define __CUDACC_RTC__ 1
#define __global__ __attribute__((global))
#define __device__ __attribute__((device))
#define __host__ __attribute__((host))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((device))
#define __forceinline__ inline __attribute__((always_inline))
#define __inline_hint__ inline
#define __noinline__ __attribute__((noinline))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __align__(n) __attribute__((aligned(n)))
#define __builtin_align__(n) __attribute__((aligned(n)))
#define __cluster_dims__(...)
#define __grid_constant__ __attribute__((grid_constant))
#define __restrict__ __restrict
#define __CUDA_ALIGN__(n) __attribute__((aligned(n)))
#define CUDART_VERSION 12040
#define __CUDACC_VER_MAJOR__ 12
#define __CUDACC_VER_MINOR__ 4
#define __NVCC_SHIM_NVRTC__ 1

#define __NV_HD__ __host__ __device__

typedef __SIZE_TYPE__ size_t;
typedef __PTRDIFF_TYPE__ ptrdiff_t;

// ---------------------------------------------------------------- vector types
struct uint3;
#define __NVS_V1(T, N) struct N##1 { T x; }; \
  static __NV_HD__ __forceinline__ N##1 make_##N##1(T x) { N##1 r; r.x = x; return r; }
#define __NVS_V2(T, N, A) struct __attribute__((aligned(A))) N##2 { T x, y; }; \
  static __NV_HD__ __forceinline__ N##2 make_##N##2(T x, T y) { N##2 r; r.x = x; r.y = y; return r; }
#define __NVS_V3(T, N) struct N##3 { T x, y, z; }; \
  static __NV_HD__ __forceinline__ N##3 make_##N##3(T x, T y, T z) { N##3 r; r.x = x; r.y = y; r.z = z; return r; }
#define __NVS_V4(T, N, A) struct __attribute__((aligned(A))) N##4 { T x, y, z, w; }; \
  static __NV_HD__ __forceinline__ N##4 make_##N##4(T x, T y, T z, T w) { N##4 r; r.x = x; r.y = y; r.z = z; r.w = w; return r; }
#define __NVS_VEC(T, N, A2, A4) __NVS_V1(T, N) __NVS_V2(T, N, A2) __NVS_V3(T, N) __NVS_V4(T, N, A4)
__NVS_VEC(signed char, char, 2, 4)
__NVS_VEC(unsigned char, uchar, 2, 4)
__NVS_VEC(short, short, 4, 8)
__NVS_VEC(unsigned short, ushort, 4, 8)
__NVS_VEC(int, int, 8, 16)
__NVS_VEC(unsigned int, uint, 8, 16)
__NVS_VEC(long, long, 16, 16)
__NVS_VEC(unsigned long, ulong, 16, 16)
__NVS_VEC(long long, longlong, 16, 16)
__NVS_VEC(unsigned long long, ulonglong, 16, 16)
__NVS_VEC(float, float, 8, 16)
__NVS_VEC(double, double, 16, 16)

struct dim3 {
  unsigned int x, y, z;
  __NV_HD__ constexpr dim3(unsigned int vx = 1, unsigned int vy = 1, unsigned int vz = 1) : x(vx), y(vy), z(vz) {}
  __NV_HD__ constexpr dim3(uint3 v);
  __NV_HD__ constexpr operator uint3() const;
};
struct uint3_ctor_helper;
__device__ __forceinline__ uint3 __nvs_make_uint3(unsigned x, unsigned y, unsigned z) { uint3 r; r.x = x; r.y = y; r.z = z; return r; }
__NV_HD__ constexpr dim3::dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
__NV_HD__ constexpr dim3::operator uint3() const { uint3 r{}; r.x = x; r.y = y; r.z = z; return r; }

// ---------------------------------------------------------------- builtin variables
__device__ __forceinline__ uint3 __nvs_threadIdx() { return __nvs_make_uint3(__nvvm_read_ptx_sreg_tid_x(), __nvvm_read_ptx_sreg_tid_y(), __nvvm_read_ptx_sreg_tid_z()); }
__device__ __forceinline__ uint3 __nvs_blockIdx() { return __nvs_make_uint3(__nvvm_read_ptx_sreg_ctaid_x(), __nvvm_read_ptx_sreg_ctaid_y(), __nvvm_read_ptx_sreg_ctaid_z()); }
__device__ __forceinline__ dim3 __nvs_blockDim() { return dim3(__nvvm_read_ptx_sreg_ntid_x(), __nvvm_read_ptx_sreg_ntid_y(), __nvvm_read_ptx_sreg_ntid_z()); }
__device__ __forceinline__ dim3 __nvs_gridDim() { return dim3(__nvvm_read_ptx_sreg_nctaid_x(), __nvvm_read_ptx_sreg_nctaid_y(), __nvvm_read_ptx_sreg_nctaid_z()); }
#define threadIdx (__nvs_threadIdx())
#define blockIdx (__nvs_blockIdx())
#define blockDim (__nvs_blockDim())
#define gridDim (__nvs_gridDim())
#define warpSize 32

// ---------------------------------------------------------------- synchronization / memory
__device__ __forceinline__ void __syncwarp(unsigned mask = 0xffffffffu) { __nvvm_bar_warp_sync(mask); }
__device__ __forceinline__ int __syncthreads_count(int p) { int r; asm volatile("{ .reg .pred q; setp.ne.s32 q, %1, 0; bar.red.popc.u32 %0, 0, q; }" : "=r"(r) : "r"(p)); return r; }
__device__ __forceinline__ int __syncthreads_and(int p) { int r; asm volatile("{ .reg .pred q, o; setp.ne.s32 q, %1, 0; bar.red.and.pred o, 0, q; selp.s32 %0, 1, 0, o; }" : "=r"(r) : "r"(p)); return r; }
__device__ __forceinline__ int __syncthreads_or(int p) { int r; asm volatile("{ .reg .pred q, o; setp.ne.s32 q, %1, 0; bar.red.or.pred o, 0, q; selp.s32 %0, 1, 0, o; }" : "=r"(r) : "r"(p)); return r; }
__device__ __forceinline__ void __threadfence() { __nvvm_membar_gl(); }
__device__ __forceinline__ void __threadfence_block() { __nvvm_membar_cta(); }
__device__ __forceinline__ void __threadfence_system() { __nvvm_membar_sys(); }
__device__ __forceinline__ void __trap() { __builtin_trap(); }
__device__ __forceinline__ unsigned __activemask() { unsigned m; asm volatile("activemask.b32 %0;" : "=r"(m)); return m; }
__device__ __forceinline__ unsigned __lane_id() { return __nvvm_read_ptx_sreg_laneid(); }
__device__ __forceinline__ long long clock64() { return __nvvm_read_ptx_sreg_clock64(); }
__device__ __forceinline__ int clock() { return __nvvm_read_ptx_sreg_clock(); }
#define assert(c) ((void)((c) ? 0 : (__builtin_trap(), 0)))
#define NULL __null
#define offsetof(t, m) __builtin_offsetof(t, m)

template <class T> __device__ __forceinline__ T __ldg(const T* p) { return *p; }
template <class T> __device__ __forceinline__ T __ldcs(const T* p) { return *p; }
template <class T> __device__ __forceinline__ T __ldca(const T* p) { return *p; }
template <class T> __device__ __forceinline__ T __ldcg(const T* p) { return *p; }
template <class T> __device__ __forceinline__ void __stcs(T* p, T v) { *p = v; }
template <class T> __device__ __forceinline__ void __stwb(T* p, T v) { *p = v; }
template <class T> __device__ __forceinline__ void __stcg(T* p, T v) { *p = v; }

// ---------------------------------------------------------------- warp shuffle / vote
__device__ __forceinline__ int __nvs_shfl_idx(unsigned m, int v, int a, int w) { return __nvvm_shfl_sync_idx_i32(m, v, a, ((32 - w) << 8) | 0x1f); }
__device__ __forceinline__ int __nvs_shfl_up(unsigned m, int v, int a, int w) { return __nvvm_shfl_sync_up_i32(m, v, a, (32 - w) << 8); }
__device__ __forceinline__ int __nvs_shfl_down(unsigned m, int v, int a, int w) { return __nvvm_shfl_sync_down_i32(m, v, a, ((32 - w) << 8) | 0x1f); }
__device__ __forceinline__ int __nvs_shfl_xor(unsigned m, int v, int a, int w) { return __nvvm_shfl_sync_bfly_i32(m, v, a, ((32 - w) << 8) | 0x1f); }
#define __NVS_SHFL_FAMILY(OP) \
  __device__ __forceinline__ int __shfl_##OP##_sync(unsigned m, int v, int a, int w = 32) { return __nvs_shfl_##OP(m, v, a, w); } \
  __device__ __forceinline__ unsigned __shfl_##OP##_sync(unsigned m, unsigned v, int a, int w = 32) { return (unsigned)__nvs_shfl_##OP(m, (int)v, a, w); } \
  __device__ __forceinline__ float __shfl_##OP##_sync(unsigned m, float v, int a, int w = 32) { return __builtin_bit_cast(float, __nvs_shfl_##OP(m, __builtin_bit_cast(int, v), a, w)); } \
  __device__ __forceinline__ long long __shfl_##OP##_sync(unsigned m, long long v, int a, int w = 32) { \
    int lo = __nvs_shfl_##OP(m, (int)v, a, w); int hi = __nvs_shfl_##OP(m, (int)(v >> 32), a, w); \
    return ((long long)hi << 32) | (unsigned)lo; } \
  __device__ __forceinline__ unsigned long long __shfl_##OP##_sync(unsigned m, unsigned long long v, int a, int w = 32) { return (unsigned long long)__shfl_##OP##_sync(m, (long long)v, a, w); } \
  __device__ __forceinline__ long __shfl_##OP##_sync(unsigned m, long v, int a, int w = 32) { return (long)__shfl_##OP##_sync(m, (long long)v, a, w); } \
  __device__ __forceinline__ unsigned long __shfl_##OP##_sync(unsigned m, unsigned long v, int a, int w = 32) { return (unsigned long)__shfl_##OP##_sync(m, (long long)v, a, w); } \
  __device__ __forceinline__ double __shfl_##OP##_sync(unsigned m, double v, int a, int w = 32) { return __builtin_bit_cast(double, __shfl_##OP##_sync(m, __builtin_bit_cast(long long, v), a, w)); } \
  __device__ __forceinline__ short __shfl_##OP##_sync(unsigned m, short v, int a, int w = 32) { return (short)__nvs_shfl_##OP(m, (int)v, a, w); } \
  __device__ __forceinline__ unsigned short __shfl_##OP##_sync(unsigned m, unsigned short v, int a, int w = 32) { return (unsigned short)__nvs_shfl_##OP(m, (int)v, a, w); }
__NVS_SHFL_FAMILY(idx)
__NVS_SHFL_FAMILY(up)
__NVS_SHFL_FAMILY(down)
__NVS_SHFL_FAMILY(xor)
#define __shfl_sync(m, v, a, ...) __shfl_idx_sync(m, v, a, ##__VA_ARGS__)

__device__ __forceinline__ unsigned __ballot_sync(unsigned m, int p) { return __nvvm_vote_ballot_sync(m, p); }
__device__ __forceinline__ int __all_sync(unsigned m, int p) { return __nvvm_vote_all_sync(m, p); }
__device__ __forceinline__ int __any_sync(unsigned m, int p) { return __nvvm_vote_any_sync(m, p); }
__device__ __forceinline__ int __uni_sync(unsigned m, int p) { return __nvvm_vote_uni_sync(m, p); }
__device__ __forceinline__ unsigned __match_any_sync(unsigned m, unsigned v) {
  unsigned r = 0; for (int i = 0; i < 32; ++i) { unsigned o = __nvs_shfl_idx(0xffffffffu, (int)v, i, 32); r |= (o == v ? 1u : 0u) << i; } (void)m; return r; }

// ---------------------------------------------------------------- integer intrinsics
__device__ __forceinline__ int __popc(unsigned x) { return __builtin_popcount(x); }
__device__ __forceinline__ int __popcll(unsigned long long x) { return __builtin_popcountll(x); }
__device__ __forceinline__ int __clz(int x) { return x ? __builtin_clz((unsigned)x) : 32; }
__device__ __forceinline__ int __clzll(long long x) { return x ? __builtin_clzll((unsigned long long)x) : 64; }
__device__ __forceinline__ int __ffs(int x) { return __builtin_ffs(x); }
__device__ __forceinline__ int __ffsll(long long x) { return __builtin_ffsll(x); }
__device__ __forceinline__ unsigned __brev(unsigned x) { return __builtin_bitreverse32(x); }
__device__ __forceinline__ unsigned long long __brevll(unsigned long long x) { return __builtin_bitreverse64(x); }
__device__ __forceinline__ int __mul24(int a, int b) { return (((a << 8) >> 8) * ((b << 8) >> 8)); }
__device__ __forceinline__ unsigned __umul24(unsigned a, unsigned b) { return (a & 0xffffffu) * (b & 0xffffffu); }
__device__ __forceinline__ int __mulhi(int a, int b) { return (int)(((long long)a * (long long)b) >> 32); }
__device__ __forceinline__ unsigned __umulhi(unsigned a, unsigned b) { return (unsigned)(((unsigned long long)a * b) >> 32); }
__device__ __forceinline__ unsigned __byte_perm(unsigned x, unsigned y, unsigned s) { return __nvvm_prmt(x, y, s); }
__device__ __forceinline__ unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned s) { s &= 31; return s ? (hi << s) | (lo >> (32 - s)) : hi; }
__device__ __forceinline__ unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned s) { s &= 31; return s ? (lo >> s) | (hi << (32 - s)) : lo; }
__device__ __forceinline__ int __dp4a(int a, int b, int c) { int r; asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__device__ __forceinline__ unsigned __dp4a(unsigned a, unsigned b, unsigned c) { unsigned r; asm("dp4a.u32.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c)); return r; }
__device__ __forceinline__ int __vsubss4(int a, int b) { return a - b; }
__device__ __forceinline__ unsigned __vcmpeq4(unsigned a, unsigned b) {
  unsigned r = 0; for (int i = 0; i < 4; ++i) r |= (((a >> 8 * i) & 0xff) == ((b >> 8 * i) & 0xff) ? 0xffu : 0u) << 8 * i; return r; }
__device__ __forceinline__ unsigned __vcmpne4(unsigned a, unsigned b) { return ~__vcmpeq4(a, b); }

// bit casts
__device__ __forceinline__ int __float_as_int(float x) { return __builtin_bit_cast(int, x); }
__device__ __forceinline__ unsigned __float_as_uint(float x) { return __builtin_bit_cast(unsigned, x); }
__device__ __forceinline__ float __int_as_float(int x) { return __builtin_bit_cast(float, x); }
__device__ __forceinline__ float __uint_as_float(unsigned x) { return __builtin_bit_cast(float, x); }
__device__ __forceinline__ long long __double_as_longlong(double x) { return __builtin_bit_cast(long long, x); }
__device__ __forceinline__ double __longlong_as_double(long long x) { return __builtin_bit_cast(double, x); }
__device__ __forceinline__ int __double2hiint(double x) { return (int)(__builtin_bit_cast(long long, x) >> 32); }
__device__ __forceinline__ int __double2loint(double x) { return (int)__builtin_bit_cast(long long, x); }
__device__ __forceinline__ double __hiloint2double(int hi, int lo) { return __builtin_bit_cast(double, ((long long)hi << 32) | (unsigned)lo); }

// ---------------------------------------------------------------- atomics (device scope)
__device__ __forceinline__ int atomicAdd(int* p, int v) { return __nvvm_atom_add_gen_i(p, v); }
__device__ __forceinline__ unsigned atomicAdd(unsigned* p, unsigned v) { return __nvvm_atom_add_gen_i((int*)p, (int)v); }
__device__ __forceinline__ unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) { return __nvvm_atom_add_gen_ll((long long*)p, (long long)v); }
__device__ __forceinline__ float atomicAdd(float* p, float v) { return __nvvm_atom_add_gen_f(p, v); }
__device__ __forceinline__ double atomicAdd(double* p, double v) { return __nvvm_atom_add_gen_d(p, v); }
__device__ __forceinline__ int atomicSub(int* p, int v) { return __nvvm_atom_add_gen_i(p, -v); }
__device__ __forceinline__ unsigned atomicSub(unsigned* p, unsigned v) { return (unsigned)__nvvm_atom_add_gen_i((int*)p, -(int)v); }
__device__ __forceinline__ int atomicExch(int* p, int v) { return __nvvm_atom_xchg_gen_i(p, v); }
__device__ __forceinline__ unsigned atomicExch(unsigned* p, unsigned v) { return (unsigned)__nvvm_atom_xchg_gen_i((int*)p, (int)v); }
__device__ __forceinline__ unsigned long long atomicExch(unsigned long long* p, unsigned long long v) { return (unsigned long long)__nvvm_atom_xchg_gen_ll((long long*)p, (long long)v); }
__device__ __forceinline__ float atomicExch(float* p, float v) { return __int_as_float(__nvvm_atom_xchg_gen_i((int*)p, __float_as_int(v))); }
__device__ __forceinline__ int atomicMin(int* p, int v) { return __nvvm_atom_min_gen_i(p, v); }
__device__ __forceinline__ unsigned atomicMin(unsigned* p, unsigned v) { return __nvvm_atom_min_gen_ui(p, v); }
__device__ __forceinline__ int atomicMax(int* p, int v) { return __nvvm_atom_max_gen_i(p, v); }
__device__ __forceinline__ unsigned atomicMax(unsigned* p, unsigned v) { return __nvvm_atom_max_gen_ui(p, v); }
__device__ __forceinline__ unsigned atomicInc(unsigned* p, unsigned v) { return __nvvm_atom_inc_gen_ui(p, v); }
__device__ __forceinline__ unsigned atomicDec(unsigned* p, unsigned v) { return __nvvm_atom_dec_gen_ui(p, v); }
__device__ __forceinline__ int atomicAnd(int* p, int v) { return __nvvm_atom_and_gen_i(p, v); }
__device__ __forceinline__ unsigned atomicAnd(unsigned* p, unsigned v) { return (unsigned)__nvvm_atom_and_gen_i((int*)p, (int)v); }
__device__ __forceinline__ int atomicOr(int* p, int v) { return __nvvm_atom_or_gen_i(p, v); }
__device__ __forceinline__ unsigned atomicOr(unsigned* p, unsigned v) { return (unsigned)__nvvm_atom_or_gen_i((int*)p, (int)v); }
__device__ __forceinline__ int atomicXor(int* p, int v) { return __nvvm_atom_xor_gen_i(p, v); }
__device__ __forceinline__ unsigned atomicXor(unsigned* p, unsigned v) { return (unsigned)__nvvm_atom_xor_gen_i((int*)p, (int)v); }
__device__ __forceinline__ int atomicCAS(int* p, int c, int v) { return __nvvm_atom_cas_gen_i(p, c, v); }
__device__ __forceinline__ unsigned atomicCAS(unsigned* p, unsigned c, unsigned v) { return (unsigned)__nvvm_atom_cas_gen_i((int*)p, (int)c, (int)v); }
__device__ __forceinline__ unsigned long long atomicCAS(unsigned long long* p, unsigned long long c, unsigned long long v) { return (unsigned long long)__nvvm_atom_cas_gen_ll((long long*)p, (long long)c, (long long)v); }
__device__ __forceinline__ int atomicAdd_block(int* p, int v) { return atomicAdd(p, v); }
__device__ __forceinline__ float atomicAdd_block(float* p, float v) { return atomicAdd(p, v); }
__device__ __forceinline__ float atomicMax(float* p, float v) {
  int old = __float_as_int(*p);
  while (__int_as_float(old) < v) { int a = atomicCAS((int*)p, old, __float_as_int(v)); if (a == old) break; old = a; }
  return __int_as_float(old);
}
__device__ __forceinline__ float atomicMin(float* p, float v) {
  int old = __float_as_int(*p);
  while (__int_as_float(old) > v) { int a = atomicCAS((int*)p, old, __float_as_int(v)); if (a == old) break; old = a; }
  return __int_as_float(old);
}

// ---------------------------------------------------------------- math (no libdevice: intrinsics + short sequences)
#define __NVS_PI_F 3.14159265358979323846f
__device__ __forceinline__ float fabsf(float x) { return __builtin_fabsf(x); }
__device__ __forceinline__ double fabs(double x) { return __builtin_fabs(x); }
__device__ __forceinline__ float fabs(float x) { return __builtin_fabsf(x); }
__device__ __forceinline__ float sqrtf(float x) { return __builtin_sqrtf(x); }
__device__ __forceinline__ double sqrt(double x) { return __builtin_sqrt(x); }
__device__ __forceinline__ float sqrt(float x) { return __builtin_sqrtf(x); }
__device__ __forceinline__ float rsqrtf(float x) { return 1.0f / __builtin_sqrtf(x); }
__device__ __forceinline__ double rsqrt(double x) { return 1.0 / __builtin_sqrt(x); }
__device__ __forceinline__ float floorf(float x) { return __builtin_floorf(x); }
__device__ __forceinline__ double floor(double x) { return __builtin_floor(x); }
__device__ __forceinline__ float floor(float x) { return __builtin_floorf(x); }
__device__ __forceinline__ float ceilf(float x) { return __builtin_ceilf(x); }
__device__ __forceinline__ double ceil(double x) { return __builtin_ceil(x); }
__device__ __forceinline__ float ceil(float x) { return __builtin_ceilf(x); }
__device__ __forceinline__ float truncf(float x) { return __builtin_truncf(x); }
__device__ __forceinline__ double trunc(double x) { return __builtin_trunc(x); }
__device__ __forceinline__ float rintf(float x) { return __builtin_rintf(x); }
__device__ __forceinline__ float nearbyintf(float x) { return __builtin_rintf(x); }
__device__ __forceinline__ double rint(double x) { return __builtin_rint(x); }
__device__ __forceinline__ float roundf(float x) { return __builtin_roundf(x); }
__device__ __forceinline__ double round(double x) { return __builtin_round(x); }
__device__ __forceinline__ float copysignf(float x, float y) { return __builtin_copysignf(x, y); }
__device__ __forceinline__ double copysign(double x, double y) { return __builtin_copysign(x, y); }
__device__ __forceinline__ float fmaf(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
__device__ __forceinline__ double fma(double a, double b, double c) { return __builtin_fma(a, b, c); }
__device__ __forceinline__ float fma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
__device__ __forceinline__ float fminf(float a, float b) { return __builtin_fminf(a, b); }
__device__ __forceinline__ float fmaxf(float a, float b) { return __builtin_fmaxf(a, b); }
__device__ __forceinline__ double fmin(double a, double b) { return __builtin_fmin(a, b); }
__device__ __forceinline__ double fmax(double a, double b) { return __builtin_fmax(a, b); }
__device__ __forceinline__ float fmin(float a, float b) { return __builtin_fminf(a, b); }
__device__ __forceinline__ float fmax(float a, float b) { return __builtin_fmaxf(a, b); }
__device__ __forceinline__ int min(int a, int b) { return a < b ? a : b; }
__device__ __forceinline__ int max(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ unsigned min(unsigned a, unsigned b) { return a < b ? a : b; }
__device__ __forceinline__ unsigned max(unsigned a, unsigned b) { return a > b ? a : b; }
__device__ __forceinline__ long long min(long long a, long long b) { return a < b ? a : b; }
__device__ __forceinline__ long long max(long long a, long long b) { return a > b ? a : b; }
__device__ __forceinline__ unsigned long long min(unsigned long long a, unsigned long long b) { return a < b ? a : b; }
__device__ __forceinline__ unsigned long long max(unsigned long long a, unsigned long long b) { return a > b ? a : b; }
__device__ __forceinline__ float min(float a, float b) { return __builtin_fminf(a, b); }
__device__ __forceinline__ float max(float a, float b) { return __builtin_fmaxf(a, b); }
__device__ __forceinline__ double min(double a, double b) { return __builtin_fmin(a, b); }
__device__ __forceinline__ double max(double a, double b) { return __builtin_fmax(a, b); }
__device__ __forceinline__ int abs(int a) { return a < 0 ? -a : a; }
__device__ __forceinline__ long long llabs(long long a) { return a < 0 ? -a : a; }
__device__ __forceinline__ long long abs(long long a) { return a < 0 ? -a : a; }

__device__ __forceinline__ bool isnan(float x) { return x != x; }
__device__ __forceinline__ bool isnan(double x) { return x != x; }
__device__ __forceinline__ bool isinf(float x) { return __builtin_fabsf(x) == __builtin_inff(); }
__device__ __forceinline__ bool isinf(double x) { return __builtin_fabs(x) == __builtin_inf(); }
__device__ __forceinline__ bool isfinite(float x) { return __builtin_fabsf(x) < __builtin_inff(); }
__device__ __forceinline__ bool isfinite(double x) { return __builtin_fabs(x) < __builtin_inf(); }
__device__ __forceinline__ int signbit(float x) { return __float_as_int(x) < 0; }

// fast approximate transcendental intrinsics (map straight to PTX *.approx)
__device__ __forceinline__ float __frcp_rn(float x) { return __nvvm_rcp_rn_f(x); }
__device__ __forceinline__ float __fdividef(float x, float y) { return x * __nvvm_rcp_approx_ftz_f(y); }
__device__ __forceinline__ float __fdiv_rn(float x, float y) { return x / y; }
__device__ __forceinline__ float __fsqrt_rn(float x) { return __builtin_sqrtf(x); }
__device__ __forceinline__ float __frsqrt_rn(float x) { return __nvvm_rsqrt_approx_f(x); }
__device__ __forceinline__ float __fadd_rn(float a, float b) { return __nvvm_add_rn_f(a, b); }
__device__ __forceinline__ float __fmul_rn(float a, float b) { return __nvvm_mul_rn_f(a, b); }
__device__ __forceinline__ float __fmaf_rn(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
__device__ __forceinline__ float __saturatef(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
__device__ __forceinline__ float __expf(float x) { return __nvvm_ex2_approx_f(x * 1.4426950408889634f); }
__device__ __forceinline__ float __exp2f(float x) { return __nvvm_ex2_approx_f(x); }
__device__ __forceinline__ float __log2f(float x) { return __nvvm_lg2_approx_f(x); }
__device__ __forceinline__ float __logf(float x) { return __nvvm_lg2_approx_f(x) * 0.6931471805599453f; }
__device__ __forceinline__ float __log10f(float x) { return __nvvm_lg2_approx_f(x) * 0.30102999566398120f; }
__device__ __forceinline__ float __sinf(float x) { return __nvvm_sin_approx_f(x); }
__device__ __forceinline__ float __cosf(float x) { return __nvvm_cos_approx_f(x); }
__device__ __forceinline__ float __tanf(float x) { return __nvvm_sin_approx_f(x) * __nvvm_rcp_approx_ftz_f(__nvvm_cos_approx_f(x)); }
__device__ __forceinline__ float __powf(float x, float y) { return __nvvm_ex2_approx_f(y * __nvvm_lg2_approx_f(x)); }

// accurate-ish versions: argument reduction around the approx hardware ops
__device__ __forceinline__ float exp2f(float x) { return __nvvm_ex2_approx_f(x); }
__device__ __forceinline__ float expf(float x) {
#ifdef __USE_FAST_MATH__
  return __expf(x);
#else
  float j = __builtin_rintf(x * 1.4426950408889634f);
  float f = __builtin_fmaf(j, -0.693145751953125f, x);
  f = __builtin_fmaf(j, -1.428606765330187e-06f, f);
  float h = j * 0.5f;                       // split 2^j to keep ex2 in range up to x ~ 88.72
  float hi = __builtin_rintf(h), lo = j - hi;
  return __nvvm_ex2_approx_f(f * 1.4426950408889634f) * __nvvm_ex2_approx_f(hi) * __nvvm_ex2_approx_f(lo);
#endif
}
__device__ __forceinline__ float expm1f(float x) { return expf(x) - 1.0f; }
__device__ __forceinline__ float log2f(float x) { return __nvvm_lg2_approx_f(x); }
__device__ __forceinline__ float logf(float x) { return __nvvm_lg2_approx_f(x) * 0.6931471805599453f; }
__device__ __forceinline__ float log10f(float x) { return __nvvm_lg2_approx_f(x) * 0.30102999566398120f; }
__device__ __forceinline__ float log1pf(float x) { return logf(1.0f + x); }
__device__ __forceinline__ float __nvs_reduce2pi(float x) {
  float k = __builtin_rintf(x * 0.15915494309189535f);
  float r = __builtin_fmaf(k, -6.28125f, x);
  return __builtin_fmaf(k, -1.9353071795864769e-03f, r);
}
__device__ __forceinline__ float sinf(float x) { return __nvvm_sin_approx_f(__nvs_reduce2pi(x)); }
__device__ __forceinline__ float cosf(float x) { return __nvvm_cos_approx_f(__nvs_reduce2pi(x)); }
__device__ __forceinline__ float tanf(float x) { float r = __nvs_reduce2pi(x); return __nvvm_sin_approx_f(r) / __nvvm_cos_approx_f(r); }
__device__ __forceinline__ void sincosf(float x, float* s, float* c) { float r = __nvs_reduce2pi(x); *s = __nvvm_sin_approx_f(r); *c = __nvvm_cos_approx_f(r); }
__device__ __forceinline__ float tanhf(float x) {
  float a = __builtin_fabsf(x);
  if (a > 9.0f) return __builtin_copysignf(1.0f, x);
  float e = expf(2.0f * a);
  return __builtin_copysignf((e - 1.0f) / (e + 1.0f), x);
}
__device__ __forceinline__ float sinhf(float x) { float e = expf(x); return 0.5f * (e - 1.0f / e); }
__device__ __forceinline__ float coshf(float x) { float e = expf(x); return 0.5f * (e + 1.0f / e); }
__device__ __forceinline__ float powf(float x, float y) {
  if (y == 0.0f) return 1.0f;
  if (y == 1.0f) return x;
  if (y == 2.0f) return x * x;
  float ay = __builtin_fabsf(y);
  float r = __nvvm_ex2_approx_f(y * __nvvm_lg2_approx_f(__builtin_fabsf(x)));
  if (x < 0.0f) {
    if (ay != __builtin_rintf(ay)) return __builtin_nanf("");
    if (__builtin_fmodf(ay, 2.0f) == 1.0f) return -r;
  }
  return r;
}
__device__ __forceinline__ float erff(float x) {
  float a = __builtin_fabsf(x), t = 1.0f / (1.0f + 0.3275911f * a);
  float p = ((((1.061405429f * t - 1.453152027f) * t) + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f;
  float y = 1.0f - p * t * expf(-a * a);
  return __builtin_copysignf(y, x);
}
__device__ __forceinline__ float erfcf(float x) { return 1.0f - erff(x); }
__device__ __forceinline__ float atanf(float x) {
  float a = __builtin_fabsf(x); bool inv = a > 1.0f; float z = inv ? 1.0f / a : a;
  float z2 = z * z;
  float r = z * (0.99997726f + z2 * (-0.33262347f + z2 * (0.19354346f + z2 * (-0.11643287f + z2 * (0.05265332f + z2 * -0.01172120f)))));
  if (inv) r = 1.57079632679f - r;
  return __builtin_copysignf(r, x);
}
__device__ __forceinline__ float atan2f(float y, float x) {
  float a = atanf(y / x);
  if (x < 0.0f) a += (y >= 0.0f ? __NVS_PI_F : -__NVS_PI_F);
  return a;
}
__device__ __forceinline__ float asinf(float x) { return atan2f(x, __builtin_sqrtf((1.0f - x) * (1.0f + x))); }
__device__ __forceinline__ float acosf(float x) { return atan2f(__builtin_sqrtf((1.0f - x) * (1.0f + x)), x); }
__device__ __forceinline__ float fmodf(float x, float y) { return __builtin_fmodf(x, y); }
__device__ __forceinline__ float hypotf(float x, float y) { return __builtin_sqrtf(x * x + y * y); }
__device__ __forceinline__ float cbrtf(float x) { return __builtin_copysignf(powf(__builtin_fabsf(x), 1.0f / 3.0f), x); }
__device__ __forceinline__ float fdimf(float a, float b) { return a > b ? a - b : 0.f; }
__device__ __forceinline__ float ldexpf(float x, int e) { return x * __nvvm_ex2_approx_f((float)e); }
__device__ __forceinline__ float sigmoidf_nvs(float x) { return 1.0f / (1.0f + expf(-x)); }

// overloads for unsuffixed names; double transcendentals evaluate in single precision (documented limitation)
__device__ __forceinline__ float exp(float x) { return expf(x); }
__device__ __forceinline__ float log(float x) { return logf(x); }
__device__ __forceinline__ float sin(float x) { return sinf(x); }
__device__ __forceinline__ float cos(float x) { return cosf(x); }
__device__ __forceinline__ float tan(float x) { return tanf(x); }
__device__ __forceinline__ float tanh(float x) { return tanhf(x); }
__device__ __forceinline__ float pow(float x, float y) { return powf(x, y); }
__device__ __forceinline__ float erf(float x) { return erff(x); }
__device__ __forceinline__ float log2(float x) { return log2f(x); }
__device__ __forceinline__ float exp2(float x) { return exp2f(x); }
__device__ __forceinline__ double exp(double x) { return (double)expf((float)x); }
__device__ __forceinline__ double log(double x) { return (double)logf((float)x); }
__device__ __forceinline__ double sin(double x) { return (double)sinf((float)x); }
__device__ __forceinline__ double cos(double x) { return (double)cosf((float)x); }
__device__ __forceinline__ double tanh(double x) { return (double)tanhf((float)x); }
__device__ __forceinline__ double pow(double x, double y) { return (double)powf((float)x, (float)y); }
__device__ __forceinline__ double erf(double x) { return (double)erff((float)x); }
__device__ __forceinline__ double fmod(double x, double y) { return __builtin_fmod(x, y); }
__device__ __forceinline__ float fmod(float x, float y) { return __builtin_fmodf(x, y); }

// ---- remaining C math names (jiterator and most user code do `using ::name;`) ----
// Double versions evaluate in single precision (documented limitation); sm_75-class hardware
// has no fast fp64 anyway.
__device__ __forceinline__ float sinhf_nvs(float x) { float e = expf(x); return 0.5f * (e - 1.0f / e); }
__device__ __forceinline__ float asinhf(float x) { float a = __builtin_fabsf(x); return __builtin_copysignf(logf(a + __builtin_sqrtf(a * a + 1.0f)), x); }
__device__ __forceinline__ float acoshf(float x) { return logf(x + __builtin_sqrtf((x - 1.0f) * (x + 1.0f))); }
__device__ __forceinline__ float atanhf(float x) { return 0.5f * logf((1.0f + x) / (1.0f - x)); }
__device__ __forceinline__ float frexpf(float x, int* e) {
  if (x == 0.0f || !(x == x) || __builtin_fabsf(x) == __builtin_inff()) { *e = 0; return x; }
  unsigned u = __float_as_uint(x); int ex = (int)((u >> 23) & 0xff);
  if (ex == 0) { x *= 16777216.0f; u = __float_as_uint(x); ex = (int)((u >> 23) & 0xff) - 24; }
  *e = ex - 126; return __uint_as_float((u & 0x807fffffu) | 0x3f000000u);
}
__device__ __forceinline__ float modff(float x, float* ip) { float t = truncf(x); *ip = t; return x - t; }
__device__ __forceinline__ float remainderf(float x, float y) { float q = rintf(x / y); return x - q * y; }
__device__ __forceinline__ float lgammaf(float x) {
  if (x < 0.5f) return logf(__NVS_PI_F / __builtin_fabsf(sinf(__NVS_PI_F * x))) - lgammaf(1.0f - x);
  x -= 1.0f;
  float a = 0.99999999999980993f;
  const float g[8] = {676.5203681218851f, -1259.1392167224028f, 771.32342877765313f, -176.61502916214059f,
                      12.507343278686905f, -0.13857109526572012f, 9.9843695780195716e-6f, 1.5056327351493116e-7f};
  for (int i = 0; i < 8; i++) a += g[i] / (x + (float)(i + 1));
  float t = x + 7.5f;
  return 0.91893853320467274f + (x + 0.5f) * logf(t) - t + logf(a);
}
__device__ __forceinline__ float tgammaf(float x) {
  if (x < 0.5f) return __NVS_PI_F / (sinf(__NVS_PI_F * x) * tgammaf(1.0f - x));
  return expf(lgammaf(x));
}
__device__ __forceinline__ float acos(float x) { return acosf(x); }
__device__ __forceinline__ float asin(float x) { return asinf(x); }
__device__ __forceinline__ float atan(float x) { return atanf(x); }
__device__ __forceinline__ float atan2(float y, float x) { return atan2f(y, x); }
__device__ __forceinline__ float cosh(float x) { return coshf(x); }
__device__ __forceinline__ float sinh(float x) { return sinhf(x); }
__device__ __forceinline__ float log10(float x) { return log10f(x); }
__device__ __forceinline__ float log1p(float x) { return log1pf(x); }
__device__ __forceinline__ float expm1(float x) { return expm1f(x); }
__device__ __forceinline__ float cbrt(float x) { return cbrtf(x); }
__device__ __forceinline__ float hypot(float x, float y) { return hypotf(x, y); }
__device__ __forceinline__ float asinh(float x) { return asinhf(x); }
__device__ __forceinline__ float acosh(float x) { return acoshf(x); }
__device__ __forceinline__ float atanh(float x) { return atanhf(x); }
__device__ __forceinline__ float erfc(float x) { return erfcf(x); }
__device__ __forceinline__ float lgamma(float x) { return lgammaf(x); }
__device__ __forceinline__ float tgamma(float x) { return tgammaf(x); }
__device__ __forceinline__ float ldexp(float x, int e) { return ldexpf(x, e); }
__device__ __forceinline__ float frexp(float x, int* e) { return frexpf(x, e); }
__device__ __forceinline__ float modf(float x, float* ip) { return modff(x, ip); }
__device__ __forceinline__ float remainder(float x, float y) { return remainderf(x, y); }
__device__ __forceinline__ double acos(double x) { return (double)acosf((float)x); }
__device__ __forceinline__ double asin(double x) { return (double)asinf((float)x); }
__device__ __forceinline__ double atan(double x) { return (double)atanf((float)x); }
__device__ __forceinline__ double atan2(double y, double x) { return (double)atan2f((float)y, (float)x); }
__device__ __forceinline__ double cosh(double x) { return (double)coshf((float)x); }
__device__ __forceinline__ double sinh(double x) { return (double)sinhf((float)x); }
__device__ __forceinline__ double tan(double x) { return (double)tanf((float)x); }
__device__ __forceinline__ double log10(double x) { return (double)log10f((float)x); }
__device__ __forceinline__ double log2(double x) { return (double)log2f((float)x); }
__device__ __forceinline__ double exp2(double x) { return (double)exp2f((float)x); }
__device__ __forceinline__ double log1p(double x) { return (double)log1pf((float)x); }
__device__ __forceinline__ double expm1(double x) { return (double)expm1f((float)x); }
__device__ __forceinline__ double cbrt(double x) { return (double)cbrtf((float)x); }
__device__ __forceinline__ double hypot(double x, double y) { return (double)hypotf((float)x, (float)y); }
__device__ __forceinline__ double asinh(double x) { return (double)asinhf((float)x); }
__device__ __forceinline__ double acosh(double x) { return (double)acoshf((float)x); }
__device__ __forceinline__ double atanh(double x) { return (double)atanhf((float)x); }
__device__ __forceinline__ double erfc(double x) { return (double)erfcf((float)x); }
__device__ __forceinline__ double lgamma(double x) { return (double)lgammaf((float)x); }
__device__ __forceinline__ double tgamma(double x) { return (double)tgammaf((float)x); }
__device__ __forceinline__ double ldexp(double x, int e) { return x * (double)__nvvm_ex2_approx_f((float)e); }
__device__ __forceinline__ double frexp(double x, int* e) { return (double)frexpf((float)x, e); }
__device__ __forceinline__ double modf(double x, double* ip) { double t = trunc(x); *ip = t; return x - t; }
__device__ __forceinline__ double remainder(double x, double y) { return (double)remainderf((float)x, (float)y); }

#ifdef __USE_FAST_MATH__
#endif

#endif  // NVRTC_SHIM_BUILTINS_H
