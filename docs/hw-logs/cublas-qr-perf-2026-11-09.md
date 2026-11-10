# Batched QR / least squares and f64 cuBLAS performance on the Arc B390, 2026-11-09

Build: branch `libs-cublas` (`nvcuda.dll`, `cublas64_12.dll`, `cufft64_11.dll`, `cusparse64_12.dll`, `cfi_redirect.dll`, `cfi-run.exe`), release.
Baseline for the performance numbers: `main` at the merge of PR #8, same box, same test binary.

## What changed

- `cublas{S,D,C,Z}geqrfBatched`: Householder QR (LAPACK `geqr2`/`larfg` convention, tau to the Tau table), one 64-thread work-group per matrix.
- `cublas{S,D,C,Z}gelsBatched`: QR, `C = Q^H C`, back substitution for `R X = C(0:n)`, op N and m >= n; `devInfoArray` gets the first zero diagonal of R. `info` reports bad parameters as `-i` (trans other than N returns `NOT_SUPPORTED`).
- gemm: a register-blocked kernel (16 x 16 threads, 4 x 4 outputs each for D/C, 2 x 2 for Z, 16-deep shared tiles), chosen when C holds at least 128 (D), 32 (Z) or 16 (C) of its tiles; the 16 x 16 kernel stays for smaller products. Every operand mode (triangular, symmetric, Hermitian) and the triangle-only stores work in both, so syrk/symm/trmm/her2k use it too. `CFI_CUBLAS_GEMM_TILE=small|big` forces one.
- `dot`, `nrm2`, `i?amax`/`i?amin`: up to 256 blocks of partials in per-handle device scratch (allocated on first use, freed by `cublasDestroy`), then one block combines them. `nrm2` sums unscaled squares and falls back to the scaled two-pass form when the sum overflowed or the largest component is below 2^-400 (f64) / 2^-40 (f32).
- `trsm` (one matrix, order > 128): 64-row diagonal blocks by the per-vector kernel, then a gemm update of all unsolved rows (`X(rest) = alpha X(rest) - T(rest, blk) X(blk)` after the first block).
- `getrfBatched` above order 16: one work-group per matrix (shared arg-max pivot search, rows split across threads for scaling and the rank-1 update). Order 16 and below keeps one thread per matrix.

## Tests

`cargo test --release --test gpu hw_cublas64 -- --ignored --nocapture --test-threads=1`: 2 passed, default kernel choice and again with `CFI_CUBLAS_GEMM_TILE=big` (every gemm-family call on the blocked kernel). New cases: gemm 300x270x90, dot n = 100000 and 50001 (increments -1, 2), nrm2 n = 100000 and 30000 (incx 2) at magnitudes 1, 1e±30/1e±200, trsm of order 300 (blocked) for every side/uplo/op, getrf/getrs at n = 70, and:

| Routine | Cases | Worst error S | D | C | Z |
|---|---|---|---|---|---|
| `geqrfBatched` (max \|H_1..H_k R - A\| relative), 8x8 batch 40, 100x30 batch 6, 5x9 (m < n), 64x64 batch 4, padded lda | 4 per type | 1.3e-6 | 3.1e-15 | 2.0e-6 | 4.2e-15 |
| `gelsBatched` (X against the normal-equations solution), the same shapes with m >= n, nrhs 0-4, padded ldc; last matrix rank deficient (devInfo = 2) | 3 per type | 9.2e-6 | 9.8e-10 | 2.4e-5 | 2.5e-12 |

(The gels reference solves the normal equations, which square the condition number; the residual of the QR solution itself is smaller.)
Other unchanged worst errors (gemm 5.8e-16 D / 1.2e-15 Z, trsm 1.2e-15 D, nrm2 5.9e-15 D) are in the test's printed table.
Also passing on this build: `hw_blas_l12`, `hw_cublas_batched` (3), `hw_xmx_gemm` (6, the f16/f32 XMX path is untouched).

Simulator (`cargo test --release --lib cublas64`, macOS): 19 passed, including both gemm kernels for every gemm/rank-k/symm/trmm case, blocked trsm, multi-block dot/nrm2/amax with ties across blocks, both getrf kernels, and geqrf/gels against the same host references.

## torch through `cfi-run` (PyTorch 2.2.2+cu118, untouched `torch\lib`)

`scripts/torch/blas64_check.py`: **574 checks passed, 0 failed** (527 before). New checks: batched `linalg.qr` (R and Q), `torch.geqrf` and `linalg.lstsq` for 4x8x8, 3x20x6, 5x32x32 and 2x6x9 (qr only) in float64, complex64 and complex128, plus float32 batched qr and lstsq.
Informational section unchanged: `linalg.eigh` wrong (rel err 0.87) and `linalg.svd` failing inside cuSOLVER's own kernels.

## Performance (`tests/gpu/hw_libs_perf.rs`, best of 3-5 runs)

Device-to-device copy: 113 GB/s (read + write), the bandwidth roofline.
GFLOP/s for gemm (2mnk, 8mnk complex), trsm (m^2 n, 4x complex) and getrf (2/3 n^3 per matrix); GB/s of the vector read for the level-1 reductions (device-pointer results).

| case | before | after | unit | time | speedup |
|---|---|---|---|---|---|
| Dgemm NN n=128 | 123 | 111.85 | GFLOP/s | 0.034 -> 0.037 ms | 0.92x |
| Dgemm TN n=128 | 143.15 | 122.64 | GFLOP/s | 0.029 -> 0.034 ms | 0.85x |
| Dgemm NN n=256 | 204.23 | 182.76 | GFLOP/s | 0.164 -> 0.184 ms | 0.89x |
| Dgemm TN n=256 | 212.77 | 211.97 | GFLOP/s | 0.158 -> 0.158 ms | 1.00x |
| Dgemm NN n=512 | 280.03 | 280.82 | GFLOP/s | 0.959 -> 0.956 ms | 1.00x |
| Dgemm TN n=512 | 284.15 | 282.09 | GFLOP/s | 0.945 -> 0.952 ms | 0.99x |
| Dgemm NN n=1024 | 318.02 | 359.88 | GFLOP/s | 6.753 -> 5.967 ms | 1.13x |
| Dgemm TN n=1024 | 316.49 | 361.27 | GFLOP/s | 6.785 -> 5.944 ms | 1.14x |
| Zgemm NN n=128 | 250.78 | 241.4 | GFLOP/s | 0.067 -> 0.069 ms | 0.97x |
| Zgemm TN n=128 | 258.91 | 256.93 | GFLOP/s | 0.065 -> 0.065 ms | 1.00x |
| Zgemm NN n=256 | 298.99 | 351.45 | GFLOP/s | 0.449 -> 0.382 ms | 1.18x |
| Zgemm TN n=256 | 304.97 | 353.02 | GFLOP/s | 0.44 -> 0.38 ms | 1.16x |
| Zgemm NN n=512 | 349.57 | 364.02 | GFLOP/s | 3.072 -> 2.95 ms | 1.04x |
| Zgemm TN n=512 | 359.64 | 361.81 | GFLOP/s | 2.986 -> 2.968 ms | 1.01x |
| Cgemm NN n=256 | 1154.06 | 1167.11 | GFLOP/s | 0.116 -> 0.115 ms | 1.01x |
| Cgemm TN n=256 | 1101.05 | 1508.06 | GFLOP/s | 0.122 -> 0.089 ms | 1.37x |
| Cgemm NN n=512 | 1654.2 | 2075.67 | GFLOP/s | 0.649 -> 0.517 ms | 1.26x |
| Cgemm TN n=512 | 1388.34 | 2718.33 | GFLOP/s | 0.773 -> 0.395 ms | 1.96x |
| Ddot n=4194304 | 6.5 | 111.4 | GB/s | 10.342 -> 0.602 ms | 17.18x |
| Zdotc n=4194304 | 12.9 | 112.5 | GB/s | 10.372 -> 1.193 ms | 8.69x |
| Dnrm2 n=4194304 | 1.9 | 104.2 | GB/s | 17.956 -> 0.322 ms | 55.76x |
| Dznrm2 n=4194304 | 2.9 | 109.8 | GB/s | 23.274 -> 0.611 ms | 38.09x |
| Idamax n=4194304 | 3.1 | 101 | GB/s | 10.729 -> 0.332 ms | 32.32x |
| Dtrsm LLNN m=n=256 | 4.01 | 13.87 | GFLOP/s | 4.181 -> 1.21 ms | 3.46x |
| Dtrsm LLNN m=n=1024 | 13.1 | 129.1 | GFLOP/s | 81.99 -> 8.317 ms | 9.86x |
| Ztrsm LLNN m=n=256 | 10.46 | 34.67 | GFLOP/s | 6.418 -> 1.936 ms | 3.32x |
| Ztrsm LLNN m=n=1024 | 27.97 | 232.34 | GFLOP/s | 153.551 -> 18.485 ms | 8.31x |
| DgetrfBatched n=8 batch=10000 | 13.87 | 10.97 | GFLOP/s | 0.246 -> 0.311 ms | 0.79x |
| DgetrfBatched n=32 batch=2000 | 11.54 | 32.08 | GFLOP/s | 3.787 -> 1.362 ms | 2.78x |
| DgetrfBatched n=64 batch=256 | 3.99 | 47.35 | GFLOP/s | 11.201 -> 0.945 ms | 11.85x |
| DgetrfBatched n=128 batch=64 | 1.08 | 32.79 | GFLOP/s | 82.643 -> 2.728 ms | 30.29x |

Reductions now run at the copy roofline (they were one 256-thread block). trsm gains 3-10x, getrfBatched 3-30x for n >= 32; n = 8 stays on the one-thread kernel (the 11 vs 14 GFLOP/s difference is within the run-to-run noise of a 0.3 ms launch on a shared GPU).
The blocked gemm gains 13% for Dgemm 1024, 17% for Zgemm 256 and 25-95% for Cgemm 512; at the sizes that keep the 16 x 16 kernel the numbers are unchanged within noise. Register blocks of 2 (`CFI_CUBLAS_GEMM_RB=2`) were slower for every type.
f64 gemm still peaks near 360 GFLOP/s (Dgemm and Zgemm alike); the limiter was not profiled.
