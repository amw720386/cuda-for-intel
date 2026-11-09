## Kernels by time (serialised host-side timing; rank, do not read absolute speed)

1588 launches, 345.01 ms total, submit 7.89 ms, wait 337.13 ms

| # | kernel | launches | total ms | % | avg us | submit % |
|---|---|---|---|---|---|---|
| 1 | `_Z9mul_mat_qIL9ggml_type8ELi128ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 165 | 192.47 | 55.8 | 1166.5 | 1 |
| 2 | `_Z9mul_mat_qIL9ggml_type8ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 93 | 33.08 | 9.6 | 355.8 | 1 |
| 3 | `_Z12soft_max_f32ILb1ELi512ELi512EfEvPKfPKT2_S1_Pf15soft_max_params` | 24 | 21.48 | 6.2 | 895.1 | 3 |
| 4 | `cfi_gemm_f16f16f32p` | 37 | 19.71 | 5.7 | 532.7 | 1 |
| 5 | `cfi_gemm_f16f16f16p_h16` | 38 | 16.61 | 4.8 | 437.2 | 1 |
| 6 | `_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0EEvPKfPKiPvxxxxxii` | 261 | 12.60 | 3.7 | 48.3 | 11 |
| 7 | `_Z21unary_gated_op_kernelIXadL_ZN38_INTERNAL_5f3b6e47_8_unary_cu_eb...` | 37 | 10.93 | 3.2 | 295.4 | 3 |
| 8 | `_Z11k_bin_bcastIXadL_ZN42_INTERNAL_bf62842d_11_binbcast_cu_6840010b...` | 189 | 7.44 | 2.2 | 39.4 | 18 |
| 9 | `_Z13convert_unaryIf6__halfEvPKvPT0_xxxxxx` | 75 | 7.09 | 2.1 | 94.5 | 6 |
| 10 | `_Z12rms_norm_f32ILi256ELb1ELb0EEvPKfPfixxxfS1_xxx5uint3S3_S3_S3_S1_...` | 76 | 4.84 | 1.4 | 63.7 | 11 |
| 11 | `_Z13mul_mat_vec_qIL9ggml_type8ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj` | 4 | 4.60 | 1.3 | 1149.3 | 0 |
| 12 | `_Z12soft_max_f32ILb1ELi0ELi0EfEvPKfPKT2_S1_Pf15soft_max_params` | 13 | 3.54 | 1.0 | 272.1 | 3 |
| 13 | `_Z9rope_neoxILb1ELb0EfEvPKT1_PS0_iiiiiPKifff14rope_corr_dimsfPKf` | 77 | 3.40 | 1.0 | 44.2 | 10 |
| 14 | `_Z7cpy_fltIXadL_ZN36_INTERNAL_5b1be78d_6_cpy_cu_843ac19f9cpy_1_fltI...` | 38 | 3.09 | 0.9 | 81.3 | 5 |
| 15 | `_Z10k_set_rowsIfx6__halfEvPKT_PKT0_PT1_xxxxxxxxxxxxxxxxx` | 76 | 1.65 | 0.5 | 21.7 | 16 |

## Driver API calls by host time

| api | calls | total ms |
|---|---|---|
| cuLaunchKernel | 1588 | 350.52 |
| cuMemcpyHtoDAsync_v2 | 11 | 47.64 |
| cuStreamSynchronize | 102 | 1.43 |
| cuCtxGetDevice_v2 | 1853 | 0.14 |
| cuStreamIsCapturing | 75 | 0.11 |
| cuMemcpyDtoHAsync_v2 | 2 | 0.02 |
| cuCtxGetCurrent | 76 | 0.01 |
| cuCtxSetCurrent | 1 | 0.00 |
| cuCtxGetApiVersion | 1 | 0.00 |

## Memcpy

| kind | calls | MiB |
|---|---|---|
| HtoD | 11 | 4.06 |
| DtoH | 2 | 0.58 |
