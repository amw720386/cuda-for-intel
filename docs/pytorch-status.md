# PyTorch on the shim: compatibility matrix

Host: Intel Arc B390, Windows. Sources: `docs/torch-bringup.md`, `docs/hw-logs/torch-round1.md`.
Columns: works = equals CPU within tolerance on hardware. "not re-run" means the item was not run
on this branch (the GPU gate was reserved for the release check); treat as unverified here.

| Workload | torch 2.2.2+cu118 (compute_37 PTX) | conda torch 2.13 cuda128 (compute_120) | Why / notes |
| --- | --- | --- | --- |
| tensor create / copy / pinned / events / streams | works | works | |
| matmul f32/f16, bmm, Linear fwd+bwd | works | works | cuBLAS replaced by `cublas_shim`; XMX on by default for large fp16 |
| elementwise, relu/gelu, sum | works | works | |
| softmax, layer_norm, batch_norm | works | works | |
| embedding, index_select, cat, dropout/randn (Philox) | works | works | |
| conv2d fwd/bwd, max_pool | works | works | native ATen kernels with `cudnn.enabled=False` |
| optimizers (SGD, Adam, AdamW, foreach) | works | works | |
| TransformerEncoderLayer fwd + grads | works | works | |
| SDPA default (mem-efficient / flash) | fails 209 | works at cc 8.0/12.0 | cu118 PTX is an arch-guard stub for `fmha_cutlassF`; use math SDPA or `CFI_REPORT_CC=3.7` |
| ResNet-18 forward | not re-run | works | |
| Hugging Face greedy generation vs CPU (GPT-2 / Qwen2.5-0.5B) | not re-run | not re-run | next step; expected to need math SDPA |
| cuDNN-backed conv | n/a | n/a | optional; `cudnn-fast` branch stand-in not merged here |
| torch.fft (fft/ifft/rfft/irfft/fftn/stft, f32 and f64) | works | not re-run | `cufft_shim` under `cfi-run`; 196 checks in `scripts/torch/fft_check.py` |
| torch.sparse matmul (CSR/COO/BSR @ dense, `mv`, `addmm`, `sparse.mm`, COO `bmm`, `sampled_addmm`; f32 and f64) | works | not re-run | `cusparse_shim` under `cfi-run`; 129 checks in `scripts/torch/sparse_check.py`. Sparse @ sparse, sparse + sparse, sparse triangular solves, complex and f16 sparse raise `CUSPARSE_STATUS_NOT_SUPPORTED` |
| float64 / complex64 / complex128 matmul, addmm, bmm, baddbmm, mv, addmv, dot/vdot/inner, `linalg.solve_triangular`, batched `linalg.solve`/`inv`/`det`/`lu_factor`/`lu_solve`, `multi_dot`, matmul backward | works | not re-run | f64/complex cuBLAS in `cublas_shim` under `cfi-run`; 527 checks in `scripts/torch/blas64_check.py` |
| single-matrix `linalg.solve`/`inv`/`det`/`cholesky`/`qr`/`lstsq` (float64) | works | not re-run | NVIDIA's `cusolver64_11.dll` from `torch\lib` on the shim's cuBLAS and driver |
| `linalg.eigh`, `linalg.svd`, batched `linalg.qr` | fails | not re-run | eigh returns wrong values and svd raises `CUSOLVER_STATUS_EXECUTION_FAILED`, both inside cuSOLVER's own kernels with no cuBLAS call involved; batched qr needs `cublas?geqrfBatched` (`CUBLAS_STATUS_NOT_SUPPORTED`) |
| NCCL | fails | fails | no stand-in |
| cu126/cu128/cu121+ Windows wheels, Linux cu126/cu128 wheels | fails | n/a | SASS only; now reported by a one-time error naming the SASS architectures and pointing at `docs/sass-support.md` (CUDA_ERROR_NO_BINARY_FOR_GPU) |
| PyTorch built with `TORCH_CUDA_ARCH_LIST=8.0+PTX` | untested | n/a | recipe in `docs/pytorch-build-ptx.md`; no build box yet |

Translator gaps listed in `docs/torch-bringup.md` still apply (2 complex<double> foreach kernels,
some very large cuda128 modules).
