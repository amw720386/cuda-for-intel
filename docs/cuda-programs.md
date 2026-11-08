# Running your own CUDA C programs

Build the device code as **PTX only**; the runtime cannot execute SASS/cubin.

```text
nvcc -arch=compute_75 -code=compute_75 -o app.exe app.cu
```

(`-gencode arch=compute_75,code=compute_75` is the spelling used by `tests/cuda_apps/build.bat`.) Inspect with
`cuobjdump -lptx app.exe` / `-lelf`: there must be PTX and no ELF.

## Runtime staging

Put these next to the executable:

- the built `nvcuda.dll`,
- the genuine NVIDIA `cudart64_*.dll` that the program links (shared or static cudart both worked),
- for cuBLAS programs, the built `cublas64_12.dll`, copied also as `cublasLt64_12.dll`,
- the signed `nvcudart_hybrid64.dll` in a directory named by `CUDA_FOR_INTEL_HYBRID_DIR` (absolute path; see `docs/ollama.md`).

Newer cudart versions look for the driver in System32 only. Preload the shim at the start of the program, before the first runtime call
(this is what `tests/cuda_apps/common.h` does):

```cpp
#ifdef _WIN32
#include <windows.h>
static struct Preload { Preload() { char p[MAX_PATH]; DWORD n = GetModuleFileNameA(NULL, p, MAX_PATH);
  while (n && p[n-1] != '\\') n--; strcpy(p + n, "nvcuda.dll"); LoadLibraryA(p); } } g_preload;
#endif
```

Programs using cuBLAS should call `cudaFree(0)` before `cublasCreate`: the shim needs a current context.

## Verified programs (Arc B390, `docs/hw-logs/cuda-apps-round1.md`, round 2, cudart 13.5 shared and static)

| Program | Result |
| --- | --- |
| `basic_kernels` (vector add, reduction, transposes, shared tiling, tiled matmul) | 5 of 5 |
| `atomics_warp` (histograms, atomics, shuffles, ballot) | 8 of 8 |
| `cublas_gemm` (Sgemm, GemmEx, batched; built against 12.x headers) | 6 of 6 |
| `cublaslt_bias` (BIAS and RELU_BIAS epilogues) | 2 of 2 |
| `streams_graphs` | 4 of 4, plus 2 skips (`cudaGraphGetNodes` / explicit node API, reported unimplemented at the time) |

`algos.cu`, `half_bf16.cu`, `mem_copies.cu`, `mem_symbols.cu` and `transformer_block.cu` also live in `tests/cuda_apps` but no committed log
records their results: unverified here.

Toolchain note from the log: nvcc 12.6 crashed in `cudafe++` against MSVC 14.51 headers, so kernel programs were built with a CUDA 13.x nvcc.

Build and run helpers: `tests/cuda_apps/build.bat` (set `CUDA` to the toolkit) and `tests/cuda_apps/run.bat`
(runs under the GPU lock, see CONTRIBUTING.md). No binaries are committed.
