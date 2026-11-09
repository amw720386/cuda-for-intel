## Kernels by time (serialised host-side timing; rank, do not read absolute speed)

1174 launches, 943.45 ms total, submit 3.37 ms, wait 940.08 ms

| # | kernel | launches | total ms | % | avg us | submit % |
|---|---|---|---|---|---|---|
| 1 | `_Z18flash_attn_ext_f16ILi64ELi64ELi8ELi4ELi4ELi2ELb0ELb0EEvPKcS1_S1...` | 42 | 425.66 | 45.1 | 10134.9 | 0 |
| 2 | `_Z9mul_mat_qIL9ggml_type12ELi128ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiii...` | 94 | 276.15 | 29.3 | 2937.8 | 0 |
| 3 | `_Z9mul_mat_qIL9ggml_type14ELi128ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiii...` | 15 | 71.74 | 7.6 | 4782.6 | 0 |
| 4 | `_Z9mul_mat_qIL9ggml_type12ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiii...` | 57 | 45.80 | 4.9 | 803.4 | 0 |
| 5 | `_Z9mul_mat_qIL9ggml_type12ELi16ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 92 | 35.97 | 3.8 | 391.0 | 1 |
| 6 | `_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout1EEvPKfPKiPvxxxxxii` | 243 | 16.26 | 1.7 | 66.9 | 6 |
| 7 | `_Z9mul_mat_qIL9ggml_type14ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiii...` | 11 | 14.23 | 1.5 | 1293.2 | 0 |
| 8 | `_Z21unary_gated_op_kernelIXadL_ZN38_INTERNAL_5f3b6e47_8_unary_cu_eb...` | 40 | 11.99 | 1.3 | 299.6 | 1 |
| 9 | `_Z9mul_mat_qIL9ggml_type14ELi16ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 14 | 10.20 | 1.1 | 728.9 | 0 |
| 10 | `_Z12rms_norm_f32ILi1024ELb1ELb0EEvPKfPfixxxfS1_xxx5uint3S3_S3_S3_S1...` | 83 | 7.98 | 0.8 | 96.1 | 4 |
| 11 | `_Z11k_bin_bcastIXadL_ZN42_INTERNAL_bf62842d_11_binbcast_cu_6840010b...` | 81 | 7.64 | 0.8 | 94.4 | 4 |
| 12 | `_Z9rope_normILb1ELb1EfEvPKT1_PS0_iiiiiPKifff14rope_corr_dimsfPKf` | 82 | 5.65 | 0.6 | 68.9 | 3 |
| 13 | `_Z13mul_mat_vec_qIL9ggml_type14ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6...` | 2 | 4.61 | 0.5 | 2303.2 | 0 |
| 14 | `_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0EEvPKfPKiPvxxxxxii` | 40 | 4.59 | 0.5 | 114.7 | 2 |
| 15 | `_Z10k_set_rowsIfx6__halfEvPKT_PKT0_PT1_xxxxxxxxxxxxxxxxx` | 84 | 3.10 | 0.3 | 36.9 | 10 |

## Driver API calls by host time

| api | calls | total ms |
|---|---|---|
| cuLaunchKernel | 1174 | 1113.03 |
| cuMemcpyHtoDAsync_v2 | 11 | 48.23 |
| cuStreamSynchronize | 27 | 1.78 |
| cuCtxGetDevice_v2 | 1929 | 0.09 |
| cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags | 42 | 0.02 |
| cuMemcpyDtoHAsync_v2 | 2 | 0.01 |

## Memcpy

| kind | calls | MiB |
|---|---|---|
| HtoD | 11 | 6.75 |
| DtoH | 2 | 0.50 |
