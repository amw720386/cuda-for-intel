# Changelog

All notable changes are documented here. Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Versioning: [SemVer, 0.x policy](docs/community/VERSIONING.md).
Entries state current behaviour only; measured results live in `docs/hw-logs/`.

## [Unreleased]

### Added
- cuFFT shim (`cufft64_11.dll`, also served as `cufft64_10.dll` by `cfi-run`): plans (`cufftPlan*`, `cufftMakePlan*`, `cufftXtMakePlanMany`), size queries, work area, stream, and `cufftExec{C2C,R2C,C2R,Z2Z,D2Z,Z2D}`/`cufftXtExec` as mixed-radix Stockham kernels in generated PTX. Covers ZLUDA's 32 cuFFT entry points and torch's `cufftXt*` path.
- PTX translator: fp8 pair conversions (`cvt.rn.satfinite[.relu].{e4m3x2,e5m2x2}.{f32,f16x2}`, `cvt.rn[.relu].f16x2.{e4m3x2,e5m2x2}`), `.reg .v2/.v4` vector registers, `mov.vN` and `ld/st .v8`.
- PTX translator: function pointers and indirect calls (a switch over the address-taken `.func`s matching the call prototype), `.reg` parameters and return values of `.func`, and function-scope `.shared`/`.global`/`.const` variables. Calls that cannot be inlined (into another module, recursive) fail with `unsupported PTX call: <reason>`.

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
