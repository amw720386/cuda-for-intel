# NVRTC shim (`crates/nvrtc_shim`)

Apps that JIT-compile CUDA C++ at run time (PyTorch jiterator, CuPy, Numba, cuda-python,
llama.cpp-style runtime kernels) call NVRTC. This crate exports that C API
(`nvrtcCreateProgram`, `nvrtcCompileProgram`, `nvrtcGetPTX(Size)`, `nvrtcGetCUBIN(Size)`,
`nvrtcGetProgramLog(Size)`, `nvrtcAddNameExpression`, `nvrtcGetLoweredName`,
`nvrtcDestroyProgram`, `nvrtcVersion`, `nvrtcGetErrorString`, `nvrtcGetSupportedArchs`,
`nvrtcGetNumSupportedArchs`, plus link-compatible stubs for PCH and flow-callback calls).
The PTX it returns is loaded by this project's `cuModuleLoadData`, translated to SPIR-V and run
through Level Zero like any other PTX.

Build: `cargo build -p nvrtc_shim --release`. The library is named `nvrtc64_120_0`
(`nvrtc64_120_0.dll` on Windows). On Linux copy it to `libnvrtc.so.12` (and `libnvrtc.so`).

## Strategy and feasibility

| Option | Status | Evidence |
| --- | --- | --- |
| (A) external clang, NVPTX backend | implemented | Homebrew LLVM 21.1.8 (`clang --print-targets` lists `nvptx64`). 8 real kernels (vector add, shared-memory reduction, warp shuffle/vote, atomics, templates, half math, cooperative groups, math functions) compile in 0.05 to 0.5 s each and run through the translator and the simulator with correct results (`cargo test --lib nvrtc_tests`). |
| (B) embedded subset compiler | out of scope | |
| (C) PTX pass-through | implemented | Source whose first token is `.version` is returned unchanged. |
| (D) result cache | implemented | In-memory map plus optional disk store (`NVRTC_SHIM_CACHE_DIR`), keyed by source, program name, headers, options, name expressions, built-in header contents, clang identity and shim version. Only successes are cached. |

A stock LLVM distribution is enough: `clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_75
-nocudainc -nocudalib -nostdinc -nostdinc++ -O2 -S`. No CUDA SDK, libdevice or host libc headers
are needed. Clang is not vendored or committed; it is found at run time:

1. `NVRTC_SHIM_CLANG` (path to a clang binary; must list `nvptx64` in `--print-targets`),
2. `clang` / `clang-NN` on `PATH`,
3. standard install locations (Homebrew `llvm`, `/usr/lib/llvm-NN`, `C:\Program Files\LLVM`).

Windows: install LLVM from the official installer (`winget install LLVM.LLVM`) or point
`NVRTC_SHIM_CLANG` at an existing `clang.exe`. If nothing usable is found `nvrtcCompileProgram`
returns `NVRTC_ERROR_BUILTIN_OPERATION_FAILURE` and the program log explains how to fix it.

### PTX version and architecture

Clang's default PTX ISA for sm_75 is 6.3, which the translator rejects (it accepts 7.x to 9.x).
The shim therefore always passes `-Xclang -target-feature -Xclang +ptx80` (`+ptx83` for sm_9x,
`+ptx86` for sm_100+), so modules start with `.version 8.0`.

The translator implements sm_75-era instructions only, so the effective target is
`min(requested, 75)` (raise with `NVRTC_SHIM_MAX_ARCH=80`). `__CUDA_ARCH__` therefore reflects
the capped value and `#if __CUDA_ARCH__ >= 800` fast paths are not selected. `nvrtcGetSupportedArchs`
reports 50..90 so apps accept their usual arch strings.

## Behaviour

* **Log**: `nvrtcGetProgramLog` returns clang's diagnostics (warnings on success too). The
  program name replaces the temp file name; staging paths are removed. Size includes the NUL.
* **Errors**: bad options give `NVRTC_ERROR_INVALID_OPTION`, diagnostics give
  `NVRTC_ERROR_COMPILATION`, a missing toolchain gives `NVRTC_ERROR_BUILTIN_OPERATION_FAILURE`.
  `nvrtcGetPTX` after a failed compile gives `NVRTC_ERROR_INVALID_INPUT`.
* **Options** (also accepted as `-opt=value` or `--opt value`): `--gpu-architecture`/`-arch`
  (`compute_XX`, `sm_XX`, `lto_XX`, suffixes like `90a` ignored), `--std=c++11/14/17/20`
  (`c++03` maps to `c++11`; default `c++17`), `-D`, `-U`, `-I`, `-include`, `-default-device`
  (clang's `force_cuda_host_device` pragma), `--use_fast_math` (defines `__USE_FAST_MATH__`, FTZ),
  `--ftz`, `--fmad` (`-ffp-contract`), `-w`. Accepted and ignored: `-rdc`, `-dc`, `-G`,
  `-lineinfo`, `-maxrregcount`, `-Xptxas`, `--prec-div`, `--prec-sqrt`, `-dlto`, diagnostics
  flags and the other documented NVRTC options. Unknown options fail with `INVALID_OPTION`.
* **Headers**: `includeNames`/`headers` are written to a per-compile temp dir (names with `..`
  or absolute paths are rejected) and searched with `-I`.
* **Name expressions**: each expression `E` appends `__device__ void* __nvrtc_ne_i = (void*)&E;`.
  Clang mangles it, the symbol is read from the PTX initializer, and the probe variable and the
  forward declaration it causes are stripped from the PTX. Works for kernel templates,
  namespaces and `__device__` variables; `extern "C"` names lower to themselves.
* **CUBIN**: no SASS assembler exists. `nvrtcGetCUBINSize` returns 0 for `compute_XX` targets
  (as real NVRTC) and the PTX text for `sm_XX` targets, which `cuModuleLoadData` here accepts.
  `NVRTC_SHIM_CUBIN_IS_PTX=1` forces that for all targets. LTO-IR is unsupported.
* **Version**: `nvrtcVersion` reports 12.4.

## Built-in headers (`crates/nvrtc_shim/include`)

Written once to `<tmp>/nvrtc_shim_inc_<hash>` and force-included, replacing NVRTC's embedded set:

* `nvrtc_shim_builtins.h` (always included): `__global__`/`__device__`/`__shared__`/
  `__launch_bounds__` and friends, `threadIdx`/`blockIdx`/`blockDim`/`gridDim`/`warpSize`,
  `dim3` and the full `char1..double4` vector family with `make_*`, `__syncthreads` (clang
  built-in) and `__syncthreads_count/and/or`, `__syncwarp`, fences, `__shfl_{,up_,down_,xor_}sync`
  (int, unsigned, float, 64-bit, double, short), `__ballot_sync`/`__all_sync`/`__any_sync`,
  `__activemask`, `__popc/__clz/__ffs/__brev/__byte_perm/__dp4a/...`, bit casts, `atomicAdd`
  (int, unsigned, ull, float, double), `atomicSub/Exch/Min/Max/Inc/Dec/And/Or/Xor/CAS`,
  `__ldg`, and a libdevice-free math library (see below).
* `cuda_fp16.h`: `__half`/`__half2`, conversions, arithmetic, comparisons, `hexp`/`hsqrt`/...,
  shuffles, `atomicAdd` on half and half2 (CAS loop on the containing word). Arithmetic goes
  through `cvt.f32.f16` / `cvt.rn.f16.f32`; `__hfma` rounds twice (f32 fma, then to half), so it
  can differ from hardware by 1 ulp on rare inputs.
* `cuda_bf16.h`: `__nv_bfloat16`/`__nv_bfloat162` using integer round-to-nearest-even
  (no sm_80 conversion instructions needed), atomicAdd.
* `cooperative_groups.h`, `cooperative_groups/reduce.h`: `thread_block`, `this_thread_block()`,
  `thread_block_tile<N>` with `shfl*`, `any/all/ballot`, `tiled_partition<N>`, `reduce` with
  `plus/less/greater`. Grid and cluster groups are not provided.
* Small virtual libc/libc++ headers (`stdint.h`, `limits.h`, `float.h`, `math.h`, `stdio.h`,
  `stddef.h`, `cstdint`, `cmath`, ...) because compilation runs with `-nostdinc`.
  Real NVRTC also has no libc, so apps already carry their own typedefs.

### Math accuracy (no libdevice)

Clang's CUDA math wrappers call libdevice, which is not available. The header maps to PTX
instructions instead: `sqrtf`, `floor`, `fma`, ... are exact; `expf` uses argument reduction plus
`ex2.approx`; `sinf`/`cosf` reduce by 2*pi then use `sin.approx`; `logf` is `lg2.approx * ln2`
(absolute error about 1e-7 near results of 0); `powf`, `tanhf`, `erff`, `atanf` are short
sequences. Typical error is a few ulp, comparable to `--use_fast_math`. Double-precision
`sqrt/fma/floor/...` are exact, but double `exp/log/sin/cos/tanh/pow/erf` are evaluated in
single precision. Do not use this path for numerics that need correctly rounded results.

## Tests

* `cargo test -p nvrtc_shim` (compile, log, options, headers, name expressions, cache, C ABI;
  skips when no clang).
* `cargo test --lib nvrtc_tests` (macOS-safe): compile each kernel in
  `crates/nvrtc_shim/tests/kernels`, translate the PTX and execute it in `src/ptx/sim.rs`
  against Rust references. The simulator gained `AtomicFAddEXT` and the OpenCL.std
  `native_exp2/log2/sin/cos/rsqrt/recip/sqrt/divide`, `log`, `sin`, `cos`, `rsqrt` operations.
* Hardware (Windows, Intel GPU, built `nvcuda.dll`): one dense ignored test runs all kernels via
  the Driver API and prints PASS/FAIL per kernel:

```
set NVRTC_SHIM_CLANG=C:\Program Files\LLVM\bin\clang.exe
cargo test --test gpu hw_nvrtc -- --ignored --nocapture
```

## Environment variables

`NVRTC_SHIM_CLANG`, `NVRTC_SHIM_MAX_ARCH`, `NVRTC_SHIM_CACHE_DIR`, `NVRTC_SHIM_NO_CACHE`,
`NVRTC_SHIM_CUBIN_IS_PTX`.

## Known gaps

No device `printf` (the call is emitted but the translator has no `vprintf`), no `cuda::std`/
libcu++, no `mma.h`/WMMA, no cooperative grid sync, no `-rdc` linking (use `nvrtcGetPTX` per
program), no LTO-IR/OptiX-IR outputs, no real CUBIN.
