# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer, 0.x policy](docs/community/VERSIONING.md).
Entries state current behaviour only; measured results live in `docs/hw-logs/`.

## [Unreleased]

### Added
- CUDA arrays, texture objects and surface objects, emulated on linear device memory (no image support in the SPIR-V target): `cuArrayCreate_v2`, `cuArray3DCreate_v2`, `cuArrayDestroy`, `cuArray{,3D}GetDescriptor_v2`, `cuMemcpy{HtoA,AtoH,DtoA,AtoD,AtoA}_v2`, `cuMemcpy{HtoA,AtoH}Async_v2`, `cuMemcpy2DUnaligned_v2` and array operands in `cuMemcpy2D/3D*`; `cuTexObject{Create,Destroy,GetResourceDesc,GetTextureDesc,GetResourceViewDesc}` over array, linear and pitch-2D resources; `cuSurfObject{Create,Destroy,GetResourceDesc}` over arrays. The translator lowers `tex[.level].{1d,2d,3d}`, `txq`, `suld.b`, `sust.b` and `suq` on object handles (all address modes, nearest and linear filtering, normalised coordinates, u8-f32 channel formats). Texture references, `tld4`, layered/cube/mipmapped textures, `sust.p` and `sured` still fail with a precise error. See `docs/textures.md`.
- `cfi-precompile` (`crates/cfi_precompile`, `scripts\cfi.ps1 precompile`): fills the SPIR-V cache and the native-binary cache ahead of time from the fatbins of DLLs/executables, through the installed `nvcuda.dll` (export `cfi_precompile_main`). Native builds can be limited to the ggml tensor types of GGUF models (`--gguf`, `--ggml-types`) and by name (`--native-skip`); substitute kernels and common cuBLAS XMX kernels are built too. See `docs/precompile.md`.
- cuBLAS `cublas{S,D,C,Z}geqrfBatched` (batched Householder QR, LAPACK `geqr2` convention) and `cublas{S,D,C,Z}gelsBatched` (batched least squares, op N, m >= n, `devInfoArray` for a zero diagonal of R), one work-group per matrix. All 55 cuBLAS names `torch_cuda.dll` imports are now implemented; torch's batched `linalg.qr`, `geqrf` and `linalg.lstsq` run on them.
- `tests/gpu/hw_libs_perf.rs`: throughput of the f64/complex BLAS kernels and cuFFT against a device-copy bandwidth roofline (timing only).
- f64 and complex cuBLAS in `cublas_shim` (also served as `cublas64_11.dll` by `cfi-run`): `{D,C,Z}gemm`, `gemmStridedBatched`, `gemmBatched`, `Cgemm3m*`/`Zgemm3m`, `CgemmEx`, and `cublasGemmEx`/`GemmStridedBatchedEx`/`GemmBatchedEx` on `CUDA_R_64F`, `CUDA_C_32F` and `CUDA_C_64F`; level 2 `gemv`, `trsv`, `trmv`, `symv`/`hemv`, `ger`/`geru`/`gerc`; level 3 `trsm`, `trsmBatched`, `trmm`, `syrk`/`herk`, `syrkx`/`herkx`, `syr2k`/`her2k`, `symm`/`hemm`, `geam`, `dgmm`; level 1 `axpy`, `scal` (incl. `Csscal`/`Zdscal`), `nrm2`, `dot`/`dotu`/`dotc`, `cublasDotEx`/`DotcEx` (also f16/bf16), `copy`, `swap`, `i?amax`/`i?amin`; `getrfBatched`/`getrsBatched`; `cublasSet/GetMatrix(Async)`, `cublasSet/GetVector(Async)`, `cublasSet/GetSmCountTarget`, `cublasGetCudartVersion`, `cublasXerbla`. All S/D/C/Z where cuBLAS defines them, as generated PTX; host or device scalars and results. The f16/bf16/f32 GEMM and XMX paths are unchanged.
- cuSPARSE shim (`cusparse64_12.dll`, also served as `cusparse64_11.dll` by `cfi-run`): handles, stream and pointer mode, the legacy matrix descriptor, generic CSR/CSC/COO and dense descriptors (const variants, getters, setters, strided batches, attributes), `cusparseSpMV`, `cusparseSpMM`, `cusparseSDDMM` (+ bufferSize and preprocess), `cusparse{S,D}bsr{mv,mm}`, `cusparseXcoo2csr`, `cusparseXcsr2coo` and `cusparseCreateIdentityPermutation`, as generated PTX (scalar and vector CSR kernels, atomic scatter for transposed and COO operands). f32 and f64; other value types return `CUSPARSE_STATUS_NOT_SUPPORTED`. Covers ZLUDA's 16 cuSPARSE entry points.
- cuFFT shim (`cufft64_11.dll`, also served as `cufft64_10.dll` by `cfi-run`): plans (`cufftPlan*`, `cufftMakePlan*`, `cufftXtMakePlanMany`), size queries, work area, stream, and `cufftExec{C2C,R2C,C2R,Z2Z,D2Z,Z2D}`/`cufftXtExec` as mixed-radix Stockham kernels in generated PTX. Covers ZLUDA's 32 cuFFT entry points and torch's `cufftXt*` path.
- PTX translator: fp8 pair conversions (`cvt.rn.satfinite[.relu].{e4m3x2,e5m2x2}.{f32,f16x2}`, `cvt.rn[.relu].f16x2.{e4m3x2,e5m2x2}`), `.reg .v2/.v4` vector registers, `mov.vN` and `ld/st .v8`.
- PTX translator: function pointers and indirect calls (a switch over the address-taken `.func`s matching the call prototype), `.reg` parameters and return values of `.func`, and function-scope `.shared`/`.global`/`.const` variables. Calls that cannot be inlined (into another module, recursive) fail with `unsupported PTX call: <reason>`.

- `tests/ptx_tests`: ZLUDA's vosen/ptx_tests suite on the SPIR-V simulator (`run.sh`, `simcuda`, CI job `ptx-tests-sim`) and on the GPU (`scripts/hwvalidate/ptx_tests.ps1`), compared with `expected_status.tsv`.
- PTX translator: `CFI_EXACT_FTZ=1` lowers `.ftz` exactly (subnormal f32 sources and results, and f16/f16x2 lanes of the half forms, become signed zeros). Off by default, where `.ftz` is still ignored.
- PTX translator: `set.CMP.{and,or,xor}` on f32 sources, `{add,sub,mul,fma}.{rz,rm,rp}.sat.f32`, `{div,sqrt,rcp}.{rz,rm,rp}.f32`, `mad.hi.{u16,s16}`, `sad.{u16,s16}`, `mad.hi.sat.s32`, `cvt.u64.s32`, `cvt.s64.u32` and `cvt.rn.ftz.f32.<int>`.
### Changed
- f64/complex gemm (and the syrk/symm/trmm family built on it) uses a register-blocked kernel (4 x 4 outputs per thread for D and C, 2 x 2 for Z) when C holds enough of its tiles (128 for D, 32 for Z, 16 for C); `CFI_CUBLAS_GEMM_TILE=small|big` forces a kernel.
- `dot`, `nrm2` and `i?amax`/`i?amin` reduce over up to 256 blocks into per-handle scratch, then one block combines the partials (`nrm2` keeps its scaled two-pass form as the fallback for sums that could over- or underflow).
- `trsm` on one matrix of order > 128 solves 64-row diagonal blocks and updates the remaining rows with the gemm kernel; `getrfBatched` uses one work-group per matrix above order 16.

### Fixed
- PTX translator: register names without `%` (valid PTX, used by ptx_tests and ZLUDA's fixtures) are accepted.
- PTX translator: `cvt.sat.{u32,u64}.{s8,s16}` clamps negative values to 0; `cvt.{rni,rzi,rmi,rpi}` to a float destination rounds; `testp.normal` is true for zero; `div.approx.f32` returns 0 for 2^126 < |b|; round-down add/sub of an exact zero sum gives -0 (also +0 - +0); adding two addresses gives an integer instead of an error.
- The translation-cache key no longer includes `CFI_REDIRECT_*` and other launcher, dump and tracing variables, so `cfi-run` and a run without it share cache entries.
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
