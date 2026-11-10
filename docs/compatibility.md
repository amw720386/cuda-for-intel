# Compatibility matrix

Experimental software. Every "works" below was run and has a log; "untested" means nobody has run it.
Report your own results with the compatibility issue template (it asks for `scripts\cfi-doctor.ps1` output); how reports are verified and added here is described in [community/COMPAT-PROCESS.md](community/COMPAT-PROCESS.md). Only reports a maintainer reproduced or backed with a log under `hw-logs/` count as verified.

## GPUs and drivers

| GPU | Driver | Status |
| --- | --- | --- |
| Intel Arc B390 (Xe3) | 32.0.101.9033 | **Tested.** All results in this repository come from this part. |
| Other Arc B-series, Arc A-series, Lunar Lake, Iris Xe, data-center parts | any | Untested by the current release. Older notes for an Iris Xe laptop are in [milestone-12.md](milestone-12.md); do not assume the current features work there. |
| Machines with two Intel GPUs | any | Untested. |
| Non-Intel GPUs | n/a | Not a goal. |

Operating system: **Windows x64 only.** There is no Linux hardware test box, so Linux is out of scope even
though the Level Zero loader code compiles there. macOS builds and runs library unit tests only.

## Applications

| Application | Status | Notes and recipe |
| --- | --- | --- |
| Ollama v0.12.6 (stock `ollama.exe` and stock `cuda_v13\ggml-cuda.dll`) | Runs the models listed in [hw-logs/model-matrix.md](hw-logs/model-matrix.md) (coherent output; several diverge from the CPU backend's greedy text) | Needs NVIDIA's signed companion DLL that you obtain yourself: [ollama.md](ollama.md). Other Ollama versions are untested. Slower than the CPU backend for small models. |
| llama.cpp (build for `75-virtual`) | Works on the logged runs | [llama-cpp.md](llama-cpp.md); flash attention off |
| PyTorch | In progress | Built-up results: [pytorch.md](pytorch.md). Per-workload matrix: [pytorch-status.md](pytorch-status.md). Wheels cu121 and newer carry SASS only and cannot run. |
| CUDA C programs with embedded PTX | Works for the bundled samples | [cuda-programs.md](cuda-programs.md) |
| cuDNN | Opt-in, experimental | Not part of the default install. Convolutions use PyTorch's native kernels with `torch.backends.cudnn.enabled = False`. |
| cuFFT | Implemented (38 entry points, ZLUDA's 32 plus torch's `cufftXt*` path) | C2C/R2C/C2R in f32 and f64, 1D-3D, batched, advanced layouts. `torch.fft` on cu118 matches CPU ([hw-logs/cufft-2026-11-09.md](hw-logs/cufft-2026-11-09.md)). No callbacks, multi-GPU or half precision; not performance-tuned |
| cuSPARSE | Implemented (81 entry points, ZLUDA's 16 plus 36 of torch's 98 imports) | Generic API: CSR/CSC/COO and dense descriptors, `cusparseSpMV`, `cusparseSpMM`, `cusparseSDDMM` (op N/T, both dense orders, 32/64-bit indices, strided batches, host or device scalars); legacy `cusparse{S,D}bsr{mv,mm}`, `cusparseXcoo2csr`/`Xcsr2coo`. f32 and f64 only. torch sparse matmul on cu118 matches CPU ([hw-logs/cusparse-2026-11-09.md](hw-logs/cusparse-2026-11-09.md)). No SpGEMM, `csrgeam2`, triangular solves or sorts (`CUSPARSE_STATUS_NOT_SUPPORTED`); not performance-tuned |
| cuBLAS f64 / complex | Implemented (161 cuBLAS entry points in all, 47 of torch's 55 imports, all 90 that NVIDIA's cuSOLVER imports) | `D/C/Z` gemm (plain, strided, pointer-array batches, `GemmEx` on R_64F/C_32F/C_64F), gemv, dot/dotc, nrm2, axpy, scal, trsm(Batched), trsv, trmm, trmv, syrk/herk, syr2k/her2k, symm/hemm, symv/hemv, ger, geam, dgmm, copy, swap, i?amax/amin, `getrfBatched`/`getrsBatched`, Set/Get Matrix/Vector; host or device scalars. torch float64/complex matmul, bmm, mv, dot and `linalg.solve_triangular`/batched solve/inv/det on cu118 match CPU ([hw-logs/cublas64-2026-11-09.md](hw-logs/cublas64-2026-11-09.md)). No `geqrfBatched`/`gelsBatched`, no `_64` ILP64 names; not performance-tuned |
| NCCL, cuSOLVER | Not implemented | |

## Feature status

| Feature | Status |
| --- | --- |
| PTX to SPIR-V translation | Supported; coverage and gaps in [ptx-gap-report.md](ptx-gap-report.md) |
| SASS / cubin only binaries | **Unsupported.** The program must embed PTX |
| cuBLAS / cuBLASLt | Entry points for the names exercised by the tested apps |
| XMX (DPAS) GEMM | Present, on by default only for large fp16/bf16 shapes. Kernel level up to 15x over the tiled path on large Qwen shapes; end to end no clear gain (Qwen2.5-0.5B prefill 7213 vs 6634 tok/s in one run, 6764 vs 6944 in another, i.e. within run-to-run noise); two small shapes are slower with it on ([hw-logs/xmx-2026-11-09.txt](hw-logs/xmx-2026-11-09.txt)). Switch off with `CFI_XMX=0` to compare |
| Quantised ggml kernels substitution | `CUDA_FOR_INTEL_SUBST=auto` by default; `none` turns it off |
| Texture and surface objects, CUDA arrays | Emulated on linear memory: 1D/2D/3D arrays and copies, texture objects over arrays, linear and pitched memory (every address mode, nearest/linear filtering, normalised coordinates), surface loads/stores. No texture references, gather, layered/cube or mipmapped textures; no hardware sampler speed ([textures.md](textures.md)) |
| VMM, multi-GPU peer access, conditional graph nodes | Not implemented |

## Known issues (honest list)

- Experimental. Do not use for anything where wrong output matters.
- Windows only. SASS-only binaries cannot run.
- XMX gives no demonstrated end-to-end speed-up; the GPU path is currently slower than the CPU backend for small models.
- First run of a model compiles kernels, taking minutes; caches make later runs fast.
- Greedy output can differ from the CPU backend at near-ties; K-quant models diverge earlier (expected).
- About 6 percent of probed PyTorch 2.13 kernels fail to translate.
- Not all `flash_attn_ext_*` kernels translate: disable flash attention.
- Architecture-guard stub kernels in PTX built for another architecture report error 209.
- Only one GPU model has ever been tested.
