## Kernels by time (serialised host-side timing; rank, do not read absolute speed)

46579 launches, 1477.89 ms total, submit 115.71 ms, wait 1362.18 ms

| # | kernel | launches | total ms | % | avg us | submit % |
|---|---|---|---|---|---|---|
| 1 | `_Z13mul_mat_vec_qIL9ggml_type8ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj` | 10535 | 1066.82 | 72.2 | 101.3 | 3 |
| 2 | `_Z13quantize_q8_1PKfPvxxxxxj5uint3` | 10535 | 74.26 | 5.0 | 7.0 | 28 |
| 3 | `_Z11k_bin_bcastIXadL_ZN42_INTERNAL_bf62842d_11_binbcast_cu_6840010b...` | 7649 | 66.38 | 4.5 | 8.7 | 34 |
| 4 | `_Z9mul_mat_qIL9ggml_type8ELi16ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiii...` | 165 | 45.28 | 3.1 | 274.4 | 1 |
| 5 | `_Z12rms_norm_f32ILi256ELb1ELb0EEvPKfPfixxxfS1_xxx5uint3S3_S3_S3_S1_...` | 3124 | 37.34 | 2.5 | 12.0 | 28 |
| 6 | `_Z10k_set_rowsIfx6__halfEvPKT_PKT0_PT1_xxxxxxxxxxxxxxxxx` | 3060 | 30.63 | 2.1 | 10.0 | 25 |
| 7 | `_Z9rope_neoxILb1ELb0EfEvPKT1_PS0_iiiiiPKifff14rope_corr_dimsfPKf` | 3059 | 26.13 | 1.8 | 8.5 | 25 |
| 8 | `_Z9mul_mat_qIL9ggml_type8ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 72 | 25.38 | 1.7 | 352.5 | 2 |
| 9 | `_Z13mul_mat_vec_fI6__halfS0_Li1ELi64EEvPKT_PKfPKiPfiiiiiiiiiiiii` | 319 | 17.77 | 1.2 | 55.7 | 4 |
| 10 | `_Z7cpy_fltIXadL_ZN36_INTERNAL_5b1be78d_6_cpy_cu_843ac19f9cpy_1_fltI...` | 1530 | 14.96 | 1.0 | 9.8 | 24 |
| 11 | `_Z13mul_mat_vec_fI6__halffLi1ELi32EEvPKT_PKfPKiPfiiiiiiiiiiiii` | 1495 | 14.83 | 1.0 | 9.9 | 23 |
| 12 | `_Z21unary_gated_op_kernelIXadL_ZN38_INTERNAL_5f3b6e47_8_unary_cu_eb...` | 1530 | 14.10 | 1.0 | 9.2 | 21 |
| 13 | `_Z13mul_mat_vec_fI6__halfS0_Li1ELi32EEvPKT_PKfPKiPfiiiiiiiiiiiii` | 1176 | 12.24 | 0.8 | 10.4 | 22 |
| 14 | `_Z12soft_max_f32ILb1ELi64ELi64EfEvPKfPKT2_S1_Pf15soft_max_params` | 768 | 7.48 | 0.5 | 9.7 | 32 |
| 15 | `_Z12soft_max_f32ILb1ELi0ELi0EfEvPKfPKT2_S1_Pf15soft_max_params` | 330 | 5.99 | 0.4 | 18.1 | 17 |

## Driver API calls by host time

| api | calls | total ms |
|---|---|---|
| cuLaunchKernel | 46579 | 1705.01 |
| cuKernelGetFunction | 2 | 66.00 |
| cuMemcpyHtoDAsync_v2 | 384 | 50.16 |
| cuStreamSynchronize | 1006 | 26.92 |
| cuCtxGetDevice_v2 | 36641 | 1.69 |
| cuMemcpyDtoHAsync_v2 | 128 | 0.53 |
| cuStreamIsCapturing | 46 | 0.01 |
| cuLibraryGetKernel | 4 | 0.01 |
| cuKernelSetAttribute | 1 | 0.00 |
| cuCtxGetCurrent | 46 | 0.00 |
| cuFuncSetAttribute | 1 | 0.00 |

## Memcpy

| kind | calls | MiB |
|---|---|---|
| DtoH | 128 | 37.31 |
| HtoD | 384 | 1.31 |
