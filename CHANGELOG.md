# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer, 0.x policy](docs/community/VERSIONING.md).
Entries state current behaviour only; measured results live in `docs/hw-logs/`.

## [Unreleased]

### Added
- CUDA arrays, texture objects and surface objects, emulated on linear device memory (no image support in the SPIR-V target): `cuArrayCreate_v2`, `cuArray3DCreate_v2`, `cuArrayDestroy`, `cuArray{,3D}GetDescriptor_v2`, `cuMemcpy{HtoA,AtoH,DtoA,AtoD,AtoA}_v2`, `cuMemcpy{HtoA,AtoH}Async_v2`, `cuMemcpy2DUnaligned_v2` and array operands in `cuMemcpy2D/3D*`; `cuTexObject{Create,Destroy,GetResourceDesc,GetTextureDesc,GetResourceViewDesc}` over array, linear and pitch-2D resources; `cuSurfObject{Create,Destroy,GetResourceDesc}` over arrays. The translator lowers `tex[.level].{1d,2d,3d}`, `txq`, `suld.b`, `sust.b` and `suq` on object handles (all address modes, nearest and linear filtering, normalised coordinates, u8-f32 channel formats). Texture references, `tld4`, layered/cube/mipmapped textures, `sust.p` and `sured` still fail with a precise error. See `docs/textures.md`.
- `cfi-precompile` (`crates/cfi_precompile`, `scripts\cfi.ps1 precompile`): fills the SPIR-V cache and the native-binary cache ahead of time from the fatbins of DLLs/executables, through the installed `nvcuda.dll` (export `cfi_precompile_main`). Native builds can be limited to the ggml tensor types of GGUF models (`--gguf`, `--ggml-types`) and by name (`--native-skip`); substitute kernels and common cuBLAS XMX kernels are built too. See `docs/precompile.md`.
- f64 and complex cuBLAS in `cublas_shim` (also served as `cublas64_11.dll` by `cfi-run`): `{D,C,Z}gemm`, `gemmStridedBatched`, `gemmBatched`, `Cgemm3m*`/`Zgemm3m`, `CgemmEx`, and `cublasGemmEx`/`GemmStridedBatchedEx`/`GemmBatchedEx` on `CUDA_R_64F`, `CUDA_C_32F` and `CUDA_C_64F`; level 2 `gemv`, `trsv`, `trmv`, `symv`/`hemv`, `ger`/`geru`/`gerc`; level 3 `trsm`, `trsmBatched`, `trmm`, `syrk`/`herk`, `syrkx`/`herkx`, `syr2k`/`her2k`, `symm`/`hemm`, `geam`, `dgmm`; level 1 `axpy`, `scal` (incl. `Csscal`/`Zdscal`), `nrm2`, `dot`/`dotu`/`dotc`, `cublasDotEx`/`DotcEx` (also f16/bf16), `copy`, `swap`, `i?amax`/`i?amin`; `getrfBatched`/`getrsBatched`; `cublasSet/GetMatrix(Async)`, `cublasSet/GetVector(Async)`, `cublasSet/GetSmCountTarget`, `cublasGetCudartVersion`, `cublasXerbla`. All S/D/C/Z where cuBLAS defines them, as generated PTX; host or device scalars and results. The f16/bf16/f32 GEMM and XMX paths are unchanged.
- cuSPARSE shim (`cusparse64_12.dll`, also served as `cusparse64_11.dll` by `cfi-run`): handles, stream and pointer mode, the legacy matrix descriptor, generic CSR/CSC/COO and dense descriptors (const variants, getters, setters, strided batches, attributes), `cusparseSpMV`, `cusparseSpMM`, `cusparseSDDMM` (+ bufferSize and preprocess), `cusparse{S,D}bsr{mv,mm}`, `cusparseXcoo2csr`, `cusparseXcsr2coo` and `cusparseCreateIdentityPermutation`, as generated PTX (scalar and vector CSR kernels, atomic scatter for transposed and COO operands). f32 and f64; other value types return `CUSPARSE_STATUS_NOT_SUPPORTED`. Covers ZLUDA's 16 cuSPARSE entry points.
- cuFFT shim (`cufft64_11.dll`, also served as `cufft64_10.dll` by `cfi-run`): plans (`cufftPlan*`, `cufftMakePlan*`, `cufftXtMakePlanMany`), size queries, work area, stream, and `cufftExec{C2C,R2C,C2R,Z2Z,D2Z,Z2D}`/`cufftXtExec` as mixed-radix Stockham kernels in generated PTX. Covers ZLUDA's 32 cuFFT entry points and torch's `cufftXt*` path.
- PTX translator: fp8 pair conversions (`cvt.rn.satfinite[.relu].{e4m3x2,e5m2x2}.{f32,f16x2}`, `cvt.rn[.relu].f16x2.{e4m3x2,e5m2x2}`), `.reg .v2/.v4` vector registers, `mov.vN` and `ld/st .v8`.
- PTX translator: function pointers and indirect calls (a switch over the address-taken `.func`s matching the call prototype), `.reg` parameters and return values of `.func`, and function-scope `.shared`/`.global`/`.const` variables. Calls that cannot be inlined (into another module, recursive) fail with `unsupported PTX call: <reason>`.

### Fixed
- The translation-cache key no longer includes `CFI_REDIRECT_*` and other launcher, dump and tracing variables, so `cfi-run` and a run without it share cache entries.
- PTX translator: a branch on a `shfl.sync.idx` broadcast (lane and control independent of the thread), `vote` or `redux` result around `bar.sync` is no longer rejected as divergent. cuSOLVER's `laed4_par` (the divide-and-conquer merge of `syevd`/`heevd`) failed to translate, cuSOLVER ignored the failed `cuModuleGetFunction`, and `torch.linalg.eigh` returned the unmerged leaf eigenvalues.
- PTX translator: `ld.volatile` and `ld.relaxed.{gpu,sys}` loads from global or generic memory are followed by a fence, so spin-waits on another block's flag (cuSOLVER `sytrd4_gpu`, `eigh` for n > 32 in float64) see the update instead of hanging the GPU. Translation cache version 7.
- `nvcuda.dll` reports a kernel that fails lazy translation once on stderr, since some libraries continue without it.
- `cublas?geam` accepts a null `beta` as zero and a null `B` (not read, `ldb` not checked), as NVIDIA's cuBLAS does; cuSOLVER's `gesvdj` transposes wide matrices this way. `CFI_CUBLAS_TRACE` also logs the call site of a failing status.
- `cuMemcpy` and `cuMemcpyAsync` resolve through `cuGetProcAddress` from CUDA 4.0 (ABI 4000), where cudart 11.8 looks them up.
- `cublasSdot_v2` supports device-pointer results (as PyTorch calls it) and any increments; `cublasSgemv_v2` supports non-unit and negative increments.
- `nvcuda.dll` no longer aborts the process when a context call (`cuCtxSetCurrent`, push/pop, create/destroy) arrives after thread-local storage teardown, as cuSOLVER's handle destructor does at exit.

## [0.1.0] - TBD (first release, not yet tagged)

Experimental first public release: a CUDA Driver API compatibility runtime for Intel Arc GPUs via Level Zero, developed and hardware-tested on Windows with an Intel Arc B390.

### Added
- CUDA Driver API runtime (`nvcuda`) over Level Zero: contexts, streams, events, memory, modules, graphs capture, multi-device enumeration (untested with two GPUs; peer access is not implemented).
- PTX to SPIR-V translator with per-entry lazy translation, flow and block-uniformity analysis, inlining of library callees, and a translation cache.
- `cfi-fatbin` fatbin PTX extraction (zstd, lz4, zlib).
- cuBLAS / cuBLASLt shim with XMX matrix kernels.
- cudart shim and NVRTC shim (CUDA C++ to PTX through clang's NVPTX target).
- Kernel substitution layer for ggml quantised matmul kernels (mmvq, mmq), on by default in `auto` mode, with `CUDA_FOR_INTEL_SUBST_CHECK` shadow verification.
- Support work for Ollama (Windows setup, launcher and install-check scripts) and PyTorch bring-up (architecture-guard handling, reported compute capability).
- PTX corpus regression harness (`tests/ptx_corpus`) and hardware test suite (`tests/gpu`).

### Changed
- Background kernel prefetch is off by default; enable with `CUDA_FOR_INTEL_PREFETCH=1` or `CUDA_FOR_INTEL_WARMUP=1`.

### Known issues
- Experimental software; the Linux path is not hardware-verified. See `docs/compatibility.md` for the verified list.

[Unreleased]: https://github.com/amw720386/cuda-for-intel/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/amw720386/cuda-for-intel/releases/tag/v0.1.0
