# cuSPARSE SpGEMM and csrgeam2 on the Arc B390, 2026-11-09

Build: branch `libs-cusparse` (`nvcuda.dll`, `cusparse64_12.dll`, `cublas64_12.dll`, `cufft64_11.dll`, `cfi_redirect.dll`, `cfi-run.exe`), release.

## What changed

- `cusparseSpGEMM_createDescr`/`destroyDescr`/`workEstimation`/`compute`/`copy`/`getNumProducts`: CSR x CSR, op N, beta 0, f32 and f64, 32- or 64-bit offsets and indices per operand, index base 0 or 1. One thread per row of C on the product kernels (`kernels/ptxgen.rs`): `rowbound` counts products per row (buffer 1 = m + 1 offsets, `bufferSize1 = (m + 1) * 8`), `expand` inserts each row's products `alpha a b` into a sorted, duplicate-free list in its segment of buffer 2 (`bufferSize2` = products x (8 + value size) + 8 m), `compact` copies the lists into C once `cusparseCsrSetPointers` has set C's arrays. The two host scans synchronize the stream (C's nnz is a host result in cuSPARSE too).
- `cusparse{S,D}csrgeam2_bufferSizeExt` (m x 8 bytes), `cusparseXcsrgeam2Nnz` (host or device nnz per pointer mode) and `cusparse{S,D}csrgeam2`: `C = alpha A + beta B` for 32-bit CSR with sorted rows, a merge per row.
- Complex csrgeam2, SpGEMM with beta != 0 or op T, and `SpGEMMreuse_*` still return `CUSPARSE_STATUS_NOT_SUPPORTED`.

## `tests/gpu/hw_cusparse.rs`

1 passed, **341 cases** (329 before), 0.7 s. New, against a host f64 dense product or sum:

| case | f32 | f64 |
|---|---|---|
| SpGEMM 200x150x120 (idx32, base 0), 64x300x80 (idx64, base 1, long rows), 1000x1000x1000 (idx64), 37x29x41 (idx32, base 1); C rows checked sorted and unique, buffer 1 size checked | 7.1e-8 to 2.2e-7 | 0 |
| csrgeam2 300x200 (base 0), 50x70 (base 1), alpha 0.75, beta -1.5; buffer size checked | 5.6e-8 to 6.3e-8 | 2.0e-16 |

Simulator (`cargo test --release --lib cusparse`, macOS): 6 passed, including the product kernels for both value types, mixed index widths and bases, the csrgeam2 union pattern row by row.

## torch sparse through `cfi-run` (PyTorch 2.2.2+cu118, untouched `torch\lib`)

`scripts/torch/sparse_check.py`: **161 checks passed, 0 failed** (129 before). New, f32 and f64 at 30x30x30, 200x150x120, 64x300x80 and 1000x1000x1000: `csr @ csr`, `torch.sparse.mm` on CSR, `csr + csr`, `torch.add(csr, csr, alpha=-0.5)`.

Still failing (informational section):
- COO @ COO: torch's cu118 build refuses it before calling cuSPARSE (`sparse_mm: CUDA Float16 requires compute capability >= 53`, against the 3.7 the shim reports for the PTX arch guard).
- sparse `triangular_solve` (`cusparseSpSM_createDescr`), complex64 CSR @ dense (`cusparseSpMM_bufferSize`), float16 sparse (refused by torch).
