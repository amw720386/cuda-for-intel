# Architecture

CUDA for Intel is a set of Windows DLLs that stand in for NVIDIA's driver and math libraries. An unmodified CUDA program (or the
CUDA Runtime inside it) calls the Driver API; this project executes the request on an Intel GPU through Level Zero.

```text
 CUDA application / PyTorch / ggml-cuda / stock Ollama
   |  CUDA Runtime (genuine NVIDIA cudart, static or shared)  <-- private export tables, via nvcudart_hybrid64.dll
   v
 nvcuda.dll  (crate cuda-for-intel, src/cuda*)       cublas64_12.dll  (crates/cublas_shim)
   handles, contexts, streams, events, graphs          cuBLAS + cuBLASLt entry points
   modules, libraries, fatbin registration             tiled PTX GEMM, native XMX (DPAS) GEMM
   |                                                     |  (loads its kernels through nvcuda.dll)
   v                                                     v
 PTX pipeline (src/ptx*): normalize -> inline -> parse -> flow analysis -> SPIR-V emit
   |
   v
 Level Zero backend (src/level_zero*): module build, command lists, USM, events
   v
 Intel compute runtime and graphics compiler  ->  Intel GPU
```

## Layers

| Layer | Where | Role |
| --- | --- | --- |
| Driver API frontend | `src/cuda.rs`, `src/cuda/` | Exported `cu*` entry points; CUDA handles; per-thread current context; primary contexts (`primary.rs`); allocation tables and pointer validation; error translation (`CuResult`); `cuGetProcAddress` versioned resolution (`discovery.rs`). |
| Runtime support | `src/cuda/export_table.rs`, `integrity.rs`, `library.rs`, `crates/fatbin` | The private tables the genuine CUDA Runtime requests (`HYBRID_CUDART`, tooling, context checks, integrity digest); the companion DLL is loaded from `CUDA_FOR_INTEL_HYBRID_DIR`. `cuLibrary*`/`cuKernel*` for CUDA 12+ runtimes; fatbin reader extracting PTX (zstd/LZ4 frames). |
| Streams, events, graphs | `src/cuda/streams.rs`, `streams/`, `graphs.rs`, `graph_api.rs` | Stream ordering (legacy default stream semantics), event generations, capture and replay, explicit graph API, host callbacks, a bounded in-flight work-group window (`CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS`). |
| PTX translator | `src/ptx.rs`, `src/ptx/` | PTX text to SPIR-V for the OpenCL kernel environment (see below). |
| Level Zero backend | `src/level_zero.rs`, `src/level_zero/` | Dynamic loader for the Level Zero API (no link-time dependency), device and queue-group selection, device/host USM allocation, module build with the driver compiler, immediate command lists (default) and regular lists with batching (feature `experimental-batching`), native binary cache, device-lost detection, a fake backend and fault injection for hardware-independent tests. |
| cuBLAS / cuBLASLt shim | `crates/cublas_shim` | `cublas64_12.dll`. Generates tiled PTX GEMMs (`ptxgen.rs`), plans cuBLASLt matmuls with fused epilogues (`ltplan.rs`, `lt.rs`), and runs a native XMX path (`xmx*.rs`). Exports `NOT_SUPPORTED` stubs for the rest of the surface (`stubs.rs`). |
| Minimal CUDA Runtime | `crates/cudart_shim` | `cudart64_12.dll`, the project's own small runtime used by controlled builds. |

Hardware-independent code (translator, fatbin reader, XMX generator, fake backend tests) builds and tests on macOS and Linux;
the Level Zero loader targets Windows and Linux 64-bit.

## PTX translator pipeline

`ptx::translate` takes PTX text and returns SPIR-V words plus entry metadata (parameter layout, work-group limits, shared-memory size).

1. **Normalise** (`normalize.rs`, `tier3*.rs`, `rewrites_cu128.rs`). Text-level canonicalisation: drops debug sections, merges tokens, rewrites
   cache-hint and memory-order qualifiers, expands `cp.async`, legacy warp intrinsics, carry chains, video SIMD and partial-mask collectives
   into already-supported instructions, extracts the dynamic `extern .shared` block. Unknown spellings pass through untouched.
2. **Inline** (`inline.rs`). `.func` bodies are spliced into call sites (registers and labels renamed per site; bounded depth and statement budget).
3. **Parse** (`parse.rs`, `param_space*.rs`, `globals.rs`). Entries become basic blocks with typed virtual registers; module-scope `.global`/`.const`
   variables become SPIR-V module variables resolved with `zeModuleGetGlobalPointer`.
4. **Flow analysis** (`flow.rs`). Definite assignment on every incoming path; barrier convergence: a `bar.sync` that thread-varying control flow could
   bypass is rejected (`CFI_STRICT_BARRIERS` tightens the loop heuristics).
5. **Emit** (`spirv.rs`, `ext*.rs`, `rounded.rs`, `tier3_mma.rs`). SPIR-V via `rspirv`: integer arithmetic, f32/f16/bf16 conversion, exact-rounded
   divide and square root by integer significand arithmetic, warp shuffles/votes as subgroup operations, `mma.sync` handled by `tier3_mma.rs`,
   atomics, dynamic shared memory (fixed window, `CUDA_FOR_INTEL_DYN_SHARED`), optional `dp4a` lowering to SPIR-V integer dot product.
6. **Pass-through** (`passthrough.rs`). Pre-built SPIR-V inside an ASCII container is accepted by `cuModuleLoadData`; the cuBLAS shim uses this for XMX,
   which PTX cannot express.

Unsupported inputs return an error naming the first failing construct; `docs/ptx-gap-report.md` measures coverage on the corpus.

## Caches

| Cache | Key | Location |
| --- | --- | --- |
| Translation cache (`src/cuda/xcache.rs`) | hash of prepared PTX text, entry name, crate and translator versions, process-wide translator switches (`CFI_*`, dp4a, dynamic shared window) | `CFI_CACHE_DIR`, default `%LOCALAPPDATA%\cuda-for-intel\spvcache`; `CFI_NO_CACHE=1` disables |
| Native binary cache (`src/level_zero/native_cache.rs`) | the driver's compiled module binary for a SPIR-V module | `CUDA_FOR_INTEL_CACHE_DIR`, default `%LOCALAPPDATA%\cuda-for-intel\cache\v1`; capped by `CUDA_FOR_INTEL_CACHE_MAX_MB` (default 4096); `CUDA_FOR_INTEL_CACHE=0` disables |

The translation cache skips PTX to SPIR-V work; the native cache skips the Intel compiler, which dominates first-run time for large kernels.
Library kernels (`cuLibraryGetKernel`) are translated lazily, one entry per module, so a library with thousands of entries does not stall the first call
(`CUDA_FOR_INTEL_LAZY=0` restores whole-library modules).

## Kernel substitution layer

A separate branch (`kernel-subst`, not part of this tree) adds hand-written native SPIR-V replacements for selected ggml kernels (mmvq, quantize_q8_1,
Q8_0 mul_mat_q), recognised by mangled name and parameter layout, verified against a host reference on first use, with fallback to the translated PTX.
Its switch is `CUDA_FOR_INTEL_SUBST=none|auto|all|<list>` (default `auto`; `-mmq`-style items switch a family off). See `docs/kernel-subst.md`.

## XMX GEMM

On Xe2/Xe3 GPUs that expose `SPV_INTEL_subgroup_matrix_multiply_accumulate`, the cuBLAS shim emits SIMD16 SPIR-V kernels using DPAS directly
(`crates/cublas_shim/src/xmx.rs`). cuBLAS matrices are column-major, so a kernel computes `C^T = op(B)^T op(A)^T`, each subgroup owning a tile of C
(default 2x4 DPAS tiles, `CFI_XMX_TILE`). Operands are f16 or bf16 with f32 accumulation; f32 inputs, accumulation in 16 bits and mixed A/B types use the
tiled PTX kernels. Selection is by `CFI_XMX`: `0` off, default *Auto* (only when `min(m,n) >= 128` and `m*n*k >= 2^24`, tunable with
`CFI_XMX_MIN_DIM` and `CFI_XMX_MIN_WORK`), `2` force for every qualifying call. A probe at first use decides which operand-mask spelling the driver accepts, and a failure
falls back to the PTX kernels. Offline, `xmx_interp.rs` interprets the emitted SPIR-V against a CPU reference. Hardware test: `tests/gpu/hw_xmx_gemm.rs`.
The commit that enabled XMX by default records a validation run on Arc B390 (574-case table and 144-case epilogue test exact).

## Failure model

Level Zero errors map to CUDA codes in `src/cuda.rs`: device lost or reset requires reset -> `CUDA_ERROR_LAUNCH_FAILED` (719); host or device out of memory ->
`CUDA_ERROR_OUT_OF_MEMORY` (2); unsupported feature -> `CUDA_ERROR_NOT_SUPPORTED` (801); invalid arguments -> 1; invalid handles -> 400.
A fake Level Zero backend (`src/level_zero/fake.rs`) and a fault injector (`fault.rs`) let the frontend's ownership, ordering and cleanup be tested on any OS.
