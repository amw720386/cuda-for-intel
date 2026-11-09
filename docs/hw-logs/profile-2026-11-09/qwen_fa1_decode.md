## Kernels by time (serialised host-side timing; rank, do not read absolute speed)

43443 launches, 1493.15 ms total, submit 100.44 ms, wait 1392.71 ms

| # | kernel | launches | total ms | % | avg us | submit % |
|---|---|---|---|---|---|---|
| 1 | `_Z13mul_mat_vec_qIL9ggml_type8ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj` | 10486 | 1055.60 | 70.7 | 100.7 | 2 |
| 2 | `_Z13quantize_q8_1PKfPvxxxxxj5uint3` | 10486 | 72.20 | 4.8 | 6.9 | 26 |
| 3 | `_Z11k_bin_bcastIXadL_ZN42_INTERNAL_bf62842d_11_binbcast_cu_6840010b...` | 7641 | 66.99 | 4.5 | 8.8 | 31 |
| 4 | `_Z18flash_attn_ext_f16ILi64ELi64ELi32ELi1ELi4ELi2ELb0ELb0EEvPKcS1_S...` | 16 | 43.22 | 2.9 | 2701.4 | 0 |
| 5 | `_Z9mul_mat_qIL9ggml_type8ELi16ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiii...` | 165 | 40.93 | 2.7 | 248.1 | 1 |
| 6 | `_Z9mul_mat_qIL9ggml_type8ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 109 | 38.97 | 2.6 | 357.5 | 1 |
| 7 | `_Z12rms_norm_f32ILi256ELb1ELb0EEvPKfPfixxxfS1_xxx5uint3S3_S3_S3_S1_...` | 3120 | 36.55 | 2.4 | 11.7 | 27 |
| 8 | `_Z18flash_attn_ext_vecILi64ELi1EL9ggml_type1ELS0_1ELb0EEvPKcS2_S2_S...` | 1488 | 36.24 | 2.4 | 24.4 | 13 |
| 9 | `_Z10k_set_rowsIfx6__halfEvPKT_PKT0_PT1_xxxxxxxxxxxxxxxxx` | 3056 | 27.81 | 1.9 | 9.1 | 26 |
| 10 | `_Z9rope_neoxILb1ELb0EfEvPKT1_PS0_iiiiiPKifff14rope_corr_dimsfPKf` | 3056 | 25.63 | 1.7 | 8.4 | 24 |
| 11 | `_Z21unary_gated_op_kernelIXadL_ZN38_INTERNAL_5f3b6e47_8_unary_cu_eb...` | 1528 | 14.88 | 1.0 | 9.7 | 18 |
| 12 | `_Z18flash_attn_ext_f16ILi64ELi64ELi16ELi1ELi4ELi2ELb0ELb0EEvPKcS1_S...` | 24 | 14.51 | 1.0 | 604.7 | 1 |
| 13 | `_Z26flash_attn_combine_resultsILi64EEvPKfPK6float2Pfi` | 1488 | 11.33 | 0.8 | 7.6 | 21 |
| 14 | `_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0EEvPKfPKiPvxxxxxii` | 274 | 4.99 | 0.3 | 18.2 | 10 |
| 15 | `_Z7cpy_fltIXadL_ZN36_INTERNAL_5b1be78d_6_cpy_cu_843ac19f9cpy_1_fltI...` | 63 | 0.93 | 0.1 | 14.8 | 19 |

## Driver API calls by host time

| api | calls | total ms |
|---|---|---|
| cuLaunchKernel | 43444 | 1524.65 |
| cuMemcpyHtoDAsync_v2 | 378 | 50.50 |
| cuStreamSynchronize | 948 | 26.34 |
| cuCtxGetDevice_v2 | 29209 | 1.36 |
| cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags | 1528 | 0.48 |
| cuMemcpyDtoHAsync_v2 | 128 | 0.41 |

## Memcpy

| kind | calls | MiB |
|---|---|---|
| DtoH | 128 | 37.31 |
| HtoD | 378 | 4.20 |
