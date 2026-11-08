# Running llama.cpp / GGML on an Intel GPU

llama.cpp's CUDA backend runs when the CUDA code is shipped as **PTX**, because this project translates
PTX (never SASS/cubin) to SPIR-V. Build it for a virtual architecture only.

## Build

Needs a CUDA toolkit (nvcc) on the build machine, Visual Studio C++ tools and CMake. Verified toolchains:
nvcc 13.5 for the `test-backend-ops` runs in `docs/hw-logs/ggml-test-backend-ops.md` (llama.cpp b12190),
nvcc 12.9 with `80-virtual` for the controlled `llama-completion` build in `docs/agent-handoff.md` (commit `52e53a7`).

```powershell
cmake -S . -B build -G Ninja -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75-virtual `
      -DGGML_CUDA_NO_VMM=ON -DGGML_CUDA_GRAPHS=OFF
cmake --build build --config Release -j 2
```

- `75-virtual` embeds compute_75 PTX and no SASS. A plain `75` or `native` would embed SASS that cannot run here.
- `GGML_CUDA_NO_VMM=ON`: the driver reports no virtual-memory-management capability, and GGML then uses its ordinary allocation pool.
  The graphs switch is taken from the controlled build in the handoff notes (which disabled VMM, graphs and flash attention);
  the `test-backend-ops` runs did not record their cache values, so treat `GRAPHS=OFF` as conservative rather than required.
- Run with flash attention off (`-fa off`): the flash-attention kernels are the main family that does not fully translate yet.

## Stage and run

Put next to the executable (or on `PATH`): the built `nvcuda.dll`, plus either
- the genuine NVIDIA `cudart64_12.dll` / `cudart64_13.dll` that your build links against and the signed
  `nvcudart_hybrid64.dll` in `CUDA_FOR_INTEL_HYBRID_DIR` (see `docs/ollama.md` for why and for how to obtain it), or
- the project's own `cudart64_12.dll` (build `-p cudart_shim`); the handoff notes list a `cudart64_12.dll` among the controlled build's DLLs. Which of the two you need depends on how your llama.cpp links cudart (static or shared); this is not yet documented by a log.

and `cublas64_12.dll` from this project (copy it twice, as `cublas64_12.dll` and `cublasLt64_12.dll`).

```powershell
. C:\cfi\cfi-env.ps1          # from scripts\setup-windows.ps1; sets CUDA_FOR_INTEL_HYBRID_DIR and PATH
.\llama-completion.exe -m model.gguf -p "Say hi." -n 8 -ngl 99 -fa off --temp 0 -t 2 -tb 2 -no-cnv
```

Always pass `-fa off` (flash attention). The first run compiles kernels for minutes; set
`CUDA_FOR_INTEL_CACHE_DIR` to keep them (the test-backend-ops log: about 5 s per case once cached).

## What was verified (Arc B390, Windows)

- `llama-completion`, Qwen2.5-0.5B Q8_0, `-ngl 99` and `-ngl 0` printed the same 8 tokens (handoff notes).
- `test-backend-ops` on Qwen2.5-0.5B-shaped `MUL_MAT` (f32, f16, q4_0, q8_0, q4_K): 157 of the 161 cases in the first table passed
  their `max_nmse` limits; the 4 crashing cases are batched f16 GQA shapes
  (`CUBLAS_STATUS_EXECUTION_FAILED`). A later test (`hw_cublas_batched`, commit `4e29111`) passes these shapes through the shim;
  `test-backend-ops` was not re-run, so the original log still lists them as failing.
- Perplexity (wikitext-2, 20 chunks, `-c 512`, `-fa off`): Q8_0 CPU 16.8323, GPU 16.8142 (-0.11 %), a small unexplained
  systematic offset recorded in the log. Q4_K_M GPU run was queued and never recorded.
- `llama-bench` Q8_0: pp512 303.8 +/- 124.9 tok/s, tg128 18.3 +/- 4.2 tok/s, measured while the GPU was shared and including JIT.

Not verified: other models, other quantisations in end-to-end runs, multi-GPU, any GPU other than Arc B390.
