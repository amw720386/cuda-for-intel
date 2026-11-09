## Kernels by time (serialised host-side timing; rank, do not read absolute speed)

44978 launches, 1418.55 ms total, submit 112.67 ms, wait 1305.88 ms

| # | kernel | launches | total ms | % | avg us | submit % |
|---|---|---|---|---|---|---|
| 1 | `_Z13mul_mat_vec_qIL9ggml_type8ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj` | 10189 | 1032.09 | 72.8 | 101.3 | 3 |
| 2 | `_Z13quantize_q8_1PKfPvxxxxxj5uint3` | 10189 | 71.09 | 5.0 | 7.0 | 29 |
| 3 | `_Z11k_bin_bcastIXadL_ZN42_INTERNAL_bf62842d_11_binbcast_cu_6840010b...` | 7390 | 64.21 | 4.5 | 8.7 | 34 |
| 4 | `_Z9mul_mat_qIL9ggml_type8ELi16ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiii...` | 165 | 42.94 | 3.0 | 260.2 | 1 |
| 5 | `_Z12rms_norm_f32ILi256ELb1ELb0EEvPKfPfixxxfS1_xxx5uint3S3_S3_S3_S1_...` | 3018 | 36.10 | 2.5 | 12.0 | 28 |
| 6 | `_Z10k_set_rowsIfx6__halfEvPKT_PKT0_PT1_xxxxxxxxxxxxxxxxx` | 2956 | 29.83 | 2.1 | 10.1 | 25 |
| 7 | `_Z9rope_neoxILb1ELb0EfEvPKT1_PS0_iiiiiPKifff14rope_corr_dimsfPKf` | 2956 | 26.53 | 1.9 | 9.0 | 25 |
| 8 | `_Z9mul_mat_qIL9ggml_type8ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 53 | 17.96 | 1.3 | 338.9 | 1 |
| 9 | `_Z13mul_mat_vec_fI6__halfS0_Li1ELi64EEvPKT_PKfPKiPfiiiiiiiiiiiii` | 270 | 15.11 | 1.1 | 55.9 | 5 |
| 10 | `_Z7cpy_fltIXadL_ZN36_INTERNAL_5b1be78d_6_cpy_cu_843ac19f9cpy_1_fltI...` | 1478 | 14.69 | 1.0 | 9.9 | 24 |
| 11 | `_Z13mul_mat_vec_fI6__halffLi1ELi32EEvPKT_PKfPKiPfiiiiiiiiiiiii` | 1446 | 14.44 | 1.0 | 10.0 | 23 |
| 12 | `_Z21unary_gated_op_kernelIXadL_ZN38_INTERNAL_5f3b6e47_8_unary_cu_eb...` | 1477 | 13.38 | 0.9 | 9.1 | 22 |
| 13 | `_Z13mul_mat_vec_fI6__halfS0_Li1ELi32EEvPKT_PKfPKiPfiiiiiiiiiiiii` | 1176 | 12.33 | 0.9 | 10.5 | 22 |
| 14 | `_Z12soft_max_f32ILb1ELi64ELi64EfEvPKfPKT2_S1_Pf15soft_max_params` | 768 | 7.31 | 0.5 | 9.5 | 33 |
| 15 | `_Z12soft_max_f32ILb1ELi0ELi0EfEvPKfPKT2_S1_Pf15soft_max_params` | 278 | 4.68 | 0.3 | 16.8 | 18 |

## Driver API calls by host time

| api | calls | total ms |
|---|---|---|
| cuLaunchKernel | 44978 | 1646.92 |
| cuKernelGetFunction | 2 | 66.11 |
| cuMemcpyHtoDAsync_v2 | 372 | 50.67 |
| cuStreamSynchronize | 970 | 25.52 |
| cuCtxGetDevice_v2 | 35359 | 1.71 |
| cuMemcpyDtoHAsync_v2 | 124 | 0.58 |
| cuLibraryGetKernel | 4 | 0.01 |
| cuStreamIsCapturing | 40 | 0.01 |
| cuKernelSetAttribute | 1 | 0.00 |
| cuCtxGetCurrent | 41 | 0.00 |
| cuCtxSetCurrent | 1 | 0.00 |
| cuFuncSetAttribute | 1 | 0.00 |
| cuCtxGetApiVersion | 1 | 0.00 |

## Memcpy

| kind | calls | MiB |
|---|---|---|
| DtoH | 124 | 36.15 |
| HtoD | 372 | 1.26 |
