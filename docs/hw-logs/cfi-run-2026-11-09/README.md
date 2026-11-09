# cfi-run on the Arc B390, 2026-11-09 (win-redirect, roadmap items 3 and 4)

Build: main + translator-next + win-redirect (integration revision 31f1c98), driver 32.0.101.9033. Install directory (`wr_install.txt`):
`nvcuda.dll`, `cublas64_12.dll`, `cublasLt64_12.dll` (the cuBLAS shim), `cfi_redirect.dll`, `cfi-run.exe`. `CFI_REDIRECT_LOG=1` for the logs.
`cfi-run.exe` was called directly (`scripts\cfi.ps1 run` was not exercised).

## Stock Ollama v0.12.6 without copying our DLLs into its tree

`scripts/profile/ollama_profile.ps1 -NoProfiled -Fa 0 -Launcher cfi-run.exe` with `CFI_REDIRECT_DIR` set, against the untouched stock tree
(stock `cublas64_13.dll`/`cublasLt64_13.dll`, no `nvcuda.dll`), compared with the same build staged into a copy of the tree.

| model | request | staged DLLs (prefill / decode tok/s) | cfi-run, pristine tree | text |
|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | 624-token prefill | 6779 | 6688 | identical |
| | 64-token decode | 38.5 | 42.9 | identical |
| Llama-3.2-1B Q4_K_M | 623-token prefill | 5234 | 5213 | identical |
| | 64-token decode | 45.2 | 39.3 | identical |

All 8 responses (warm-up, short, long, decode for both models) are token-identical; no server errors. Ollama lists CUDA0 = Intel Arc B390.
The redirect log shows the server and every runner child injected, `nvcuda.dll` mapped to the install directory, and `ggml-cuda.dll`'s
static import `cublas64_13.dll` preloaded from `<install>\aliases\cublas64_13.dll`. The stock tree's files are byte-identical before and
after (`pristine_tree_before.txt`, `pristine_tree_after.txt`). Speed differences are within run-to-run noise (decode varies about 10 %).

## PyTorch 2.2.2+cu118 ladder L1-L3 under cfi-run (roadmap item 4)

A copy of the cu118 venv with torch's own stock `cublas64_11.dll` / `cublasLt64_11.dll` restored (`torch_pristine_libs.txt`), no
`CFI_NVCUDA` preload, started as `cfi-run.exe --install-dir <bin> -- <venv>\Scripts\python.exe progressive.py --levels L1,L2,L3`.

| level | replaced-DLL venv (baseline) | cfi-run, stock torch\lib |
|---|---|---|
| L1 tensor create/copy/elementwise | pass, 11.3 s | pass, 15.1 s |
| L2 matmul f32/f16 vs CPU | pass, 1.6 s | pass, 1.6 s |
| L3 softmax/layernorm/reductions | pass, 12.3 s | pass, 12.3 s |

Whether torch's own libraries are redirected (`modcheck_out.txt`, `modcheck_redirect_log.txt`; a matmul in f32 and f16 under cfi-run, then
`GetModuleFileNameW` of each module):
- `nvcuda.dll` -> install directory (**redirected**; the machine has no NVIDIA driver).
- `torch\lib\cublas64_11.dll` and `torch\lib\cublasLt64_11.dll`, requested by full path -> `<install>\aliases\cublas64_11.dll` /
  `cublasLt64_11.dll` (**redirected**), also when pulled in as static imports of `cusolver64_11.dll`, `cusolverMg64_11.dll` and `shm.dll`.
- `cudnn64_8.dll` and `cudart64_110.dll` stay torch's own (the install directory has no cuDNN, and torch's cudart is a host runtime that
  then loads `nvcuda.dll`, which is redirected).
- `torch\lib` is unchanged afterwards (`torch_pristine_libs_after.txt`).

Not covered: the conda cuda128 torch, ladder levels L4 and L5 under cfi-run, and the item 5 check that the SASS-only diagnostic names
torch's stock cuBLAS (no SASS-only cuBLAS was loaded: the redirect replaced it).
