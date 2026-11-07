# cublasLt shim: ABI values to verify against the real library

Code: `crates/cublas_shim/src/{lt,ltplan}.rs` (exports), `ptxgen.rs` (`generate_ex`, fused epilogue, `generate_trsm`).
Install the built DLL under both `cublas64_12` and `cublasLt64_12` names (it exports both sets).
All values below were written from memory of `cublasLt.h`; trace the real `libcublasLt` (CUDA 12.x) on hardware and compare.

## Values to verify (priority order)

1. `cublasLtMatmulDescAttributes_t` ids. Used: COMPUTE_TYPE 0, SCALE_TYPE 1, POINTER_MODE 2, TRANSA 3, TRANSB 4, TRANSC 5, FILL_MODE 6, EPILOGUE 7, BIAS_POINTER 8, BIAS_BATCH_STRIDE 10 (9 believed unused), EPILOGUE_AUX_POINTER 11, AUX_LD 12, AUX_BATCH_STRIDE 13, ALPHA_VECTOR_BATCH_STRIDE 14, SM_COUNT_TARGET 15, A/B/C/D_SCALE_POINTER 17-20, AMAX_D_POINTER 21, EPILOGUE_AUX_DATA_TYPE 22, AUX_SCALE_POINTER 23, AUX_AMAX_POINTER 24, FAST_ACCUM 25, BIAS_DATA_TYPE 26. Ids 27..63 are stored opaquely.
2. Attribute byte sizes (Set must be exact): enums/ints 4, pointers and strides 8, FAST_ACCUM 1. Real library may accept e.g. BIAS_BATCH_STRIDE as int64 (assumed) vs int32.
3. `cublasLtEpilogue_t`: DEFAULT 1, RELU 2, BIAS 4, RELU_BIAS 6, GELU 32, GELU_BIAS 36 (aux/grad variants 128-bit flag combos, rejected as NOT_SUPPORTED).
4. `cublasLtMatrixLayoutAttribute_t`: TYPE 0, ORDER 1, ROWS 2, COLS 3, LD 4, BATCH_COUNT 5, STRIDED_BATCH_OFFSET 6, PLANE_OFFSET 7. `cublasLtOrder_t`: COL 0, ROW 1 (COL32 2 etc rejected). ROWS/COLS u64, LD i64, BATCH_COUNT i32, OFFSET i64.
5. `cublasLtMatmulPreferenceAttributes_t`: SEARCH_MODE 0, MAX_WORKSPACE_BYTES 1, REDUCTION_SCHEME_MASK 3, MIN_ALIGNMENT_A..D_BYTES 5-8, MAX_WAVES_COUNT 9, IMPL_MASK 12. Sizes: workspace u64, others u32/float.
6. `cublasLtPointerMode_t`: HOST 0, DEVICE 1; vector modes 2-4 rejected.
7. `cublasLtMatmulHeuristicResult_t`: `{algo[8 x u64]; size_t workspaceSize; int state; float wavesCount; int reserved[4]}` = 96 bytes (unit-tested for offsets, but confirm the real size, e.g. via `sizeof` in a C probe).
8. `cublasLtGetVersion` returns 120900 (12.9.0); real value encodes major*10000+minor*100+patch (assumed). `cublasLtGetProperty` ids MAJOR 0, MINOR 1, PATCH 2.
9. Descriptor defaults: TRANSA/B/C = N, EPILOGUE = DEFAULT(1), POINTER_MODE = HOST, BATCH_COUNT = 1.

## Behaviour to verify

- Epilogue order: shim computes `act(alpha*A*B + beta*C + bias)`; confirm bias is added after beta*C.
- GELU is the tanh approximation (`0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))`); confirm real `EPILOGUE_GELU` matches PyTorch's use (`approximate='tanh'`).
- Bias vector length is rows of D (M), dtype defaults to D's dtype when BIAS_DATA_TYPE is unset.
- Heuristic with an unsupported combination returns SUCCESS with `returnAlgoCount = 0` here; the real library may return `CUBLAS_STATUS_NOT_SUPPORTED`. Layout mismatches return INVALID_VALUE.
- Set-time validation: the shim rejects enum values out of range at SetAttribute; the real library may defer to matmul.
- Real `GetAttribute` with `sizeInBytes == 0` and non-null `sizeWritten` is treated as a size query.
- NULL `C` pointer with beta == 0 is accepted; `C != D` with beta != 0 returns NOT_SUPPORTED (kernel reads C through D).

## Known limitations (NOT_SUPPORTED)

Row-major D with a bias epilogue; vector/batched bias; aux/gradient epilogues; COL32/tile orders; FP8; TRANSC != N; mixed A/B dtypes; C dtype != D dtype; A/B batch counts that differ. Workspace is ignored (heuristic reports 0 bytes). Compute types TF32/FAST_16* run as plain f32 accumulation.

## Other changes in this branch needing hardware confirmation

- `CUBLAS_COMPUTE_16F` GemmEx now rounds the accumulator and alpha/beta terms to f16 after every operation (kernel suffix `_h16`). Compare with real cuBLAS half accumulation (which may use fused f16 FMA, differing in the last bit).
- `cublasStrsmBatched`: all side/uplo/trans/diag combinations, one thread per right-hand-side vector (`cfi_strsm_batched`). Check it on ggml `solve_tri` shapes and that the PTX translator accepts the loop with `div.rn.f32` on Intel (the main crate's translator test `translates_generated_cublaslt_epilogue_and_trsm_kernels` could not be run on macOS).
- Run ignored tests: `cargo test -p cublas_shim -- --ignored` (`gpu_gemm_matches_cpu_reference`, `gpu_cublaslt_and_strsm_match_cpu_reference`), and `cargo test --lib translates_generated_cublas` in the main crate.
