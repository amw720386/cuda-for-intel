## Kernels by time (serialised host-side timing; rank, do not read absolute speed)

1713 launches, 342.13 ms total, submit 7.51 ms, wait 334.62 ms

| # | kernel | launches | total ms | % | avg us | submit % |
|---|---|---|---|---|---|---|
| 1 | `_Z9mul_mat_qIL9ggml_type8ELi128ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 165 | 196.87 | 57.5 | 1193.2 | 1 |
| 2 | `_Z9mul_mat_qIL9ggml_type8ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 112 | 39.68 | 11.6 | 354.3 | 1 |
| 3 | `_Z12soft_max_f32ILb1ELi512ELi512EfEvPKfPKT2_S1_Pf15soft_max_params` | 24 | 21.62 | 6.3 | 900.8 | 1 |
| 4 | `_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0EEvPKfPKiPvxxxxxii` | 282 | 13.87 | 4.1 | 49.2 | 9 |
| 5 | `_Z21unary_gated_op_kernelIXadL_ZN38_INTERNAL_5f3b6e47_8_unary_cu_eb...` | 41 | 11.30 | 3.3 | 275.7 | 2 |
| 6 | `cfi_xmx_f16f32p_2x4_b0` | 24 | 7.82 | 2.3 | 325.9 | 1 |
| 7 | `_Z11k_bin_bcastIXadL_ZN42_INTERNAL_bf62842d_11_binbcast_cu_6840010b...` | 203 | 7.61 | 2.2 | 37.5 | 18 |
| 8 | `_Z13convert_unaryIf6__halfEvPKvPT0_xxxxxx` | 81 | 7.39 | 2.2 | 91.2 | 6 |
| 9 | `_Z12rms_norm_f32ILi256ELb1ELb0EEvPKfPfixxxfS1_xxx5uint3S3_S3_S3_S1_...` | 82 | 4.87 | 1.4 | 59.4 | 7 |
| 10 | `_Z13mul_mat_vec_qIL9ggml_type8ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj` | 4 | 4.62 | 1.4 | 1155.6 | 0 |
| 11 | `cfi_xmx_f16f16p_2x4_b0` | 24 | 4.52 | 1.3 | 188.3 | 3 |
| 12 | `_Z12soft_max_f32ILb1ELi0ELi0EfEvPKfPKT2_S1_Pf15soft_max_params` | 16 | 4.31 | 1.3 | 269.2 | 2 |
| 13 | `_Z9rope_neoxILb1ELb0EfEvPKT1_PS0_iiiiiPKifff14rope_corr_dimsfPKf` | 82 | 3.45 | 1.0 | 42.1 | 8 |
| 14 | `_Z7cpy_fltIXadL_ZN36_INTERNAL_5b1be78d_6_cpy_cu_843ac19f9cpy_1_fltI...` | 41 | 3.19 | 0.9 | 77.8 | 6 |
| 15 | `cfi_gemm_f16f16f16p_h16` | 17 | 2.91 | 0.9 | 171.2 | 1 |

## Driver API calls by host time

| api | calls | total ms |
|---|---|---|
| cuLaunchKernel | 1713 | 346.78 |
| cuMemcpyHtoDAsync_v2 | 11 | 47.98 |
| cuStreamSynchronize | 108 | 1.97 |
| cuCtxGetDevice_v2 | 2000 | 0.13 |
| cuStreamIsCapturing | 81 | 0.02 |
| cuMemcpyDtoHAsync_v2 | 2 | 0.02 |
| cuCtxGetCurrent | 81 | 0.01 |

## Memcpy

| kind | calls | MiB |
|---|---|---|
| HtoD | 11 | 4.06 |
| DtoH | 2 | 0.58 |
