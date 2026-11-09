## Kernels by time (serialised host-side timing; rank, do not read absolute speed)

1350 launches, 459.94 ms total, submit 3.24 ms, wait 456.70 ms

| # | kernel | launches | total ms | % | avg us | submit % |
|---|---|---|---|---|---|---|
| 1 | `_Z9mul_mat_qIL9ggml_type8ELi128ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 165 | 189.56 | 41.2 | 1148.9 | 0 |
| 2 | `_Z18flash_attn_ext_f16ILi64ELi64ELi32ELi1ELi4ELi2ELb0ELb0EEvPKcS1_S...` | 32 | 189.37 | 41.2 | 5917.8 | 0 |
| 3 | `_Z9mul_mat_qIL9ggml_type8ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 56 | 20.72 | 4.5 | 370.0 | 1 |
| 4 | `_Z17quantize_mmq_q8_1IL18mmq_q8_1_ds_layout0EEvPKfPKiPvxxxxxii` | 271 | 16.00 | 3.5 | 59.0 | 6 |
| 5 | `_Z21unary_gated_op_kernelIXadL_ZN38_INTERNAL_5f3b6e47_8_unary_cu_eb...` | 39 | 11.49 | 2.5 | 294.7 | 1 |
| 6 | `_Z11k_bin_bcastIXadL_ZN42_INTERNAL_bf62842d_11_binbcast_cu_6840010b...` | 194 | 8.27 | 1.8 | 42.6 | 7 |
| 7 | `_Z9mul_mat_qIL9ggml_type8ELi16ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiiii...` | 50 | 7.18 | 1.6 | 143.7 | 2 |
| 8 | `_Z12rms_norm_f32ILi256ELb1ELb0EEvPKfPfixxxfS1_xxx5uint3S3_S3_S3_S1_...` | 79 | 4.78 | 1.0 | 60.5 | 6 |
| 9 | `_Z13mul_mat_vec_qIL9ggml_type8ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6_jjj` | 4 | 4.48 | 1.0 | 1120.6 | 0 |
| 10 | `_Z9rope_neoxILb1ELb0EfEvPKT1_PS0_iiiiiPKifff14rope_corr_dimsfPKf` | 78 | 3.23 | 0.7 | 41.4 | 5 |
| 11 | `_Z18flash_attn_ext_f16ILi64ELi64ELi16ELi1ELi4ELi2ELb0ELb0EEvPKcS1_S...` | 8 | 2.58 | 0.6 | 322.2 | 1 |
| 12 | `_Z10k_set_rowsIfx6__halfEvPKT_PKT0_PT1_xxxxxxxxxxxxxxxxx` | 80 | 1.51 | 0.3 | 18.9 | 14 |
| 13 | `_Z25flash_attn_stream_k_fixupILi64ELi32ELi1EEvPfPK6float2iiii` | 8 | 0.40 | 0.1 | 49.5 | 4 |
| 14 | `_Z7cpy_fltIXadL_ZN36_INTERNAL_5b1be78d_6_cpy_cu_843ac19f9cpy_1_fltI...` | 2 | 0.15 | 0.0 | 75.8 | 6 |
| 15 | `_Z25flash_attn_stream_k_fixupILi64ELi16ELi1EEvPfPK6float2iiii` | 8 | 0.13 | 0.0 | 15.9 | 14 |

## Driver API calls by host time

| api | calls | total ms |
|---|---|---|
| cuLaunchKernel | 1350 | 462.95 |
| cuMemcpyHtoDAsync_v2 | 11 | 47.67 |
| cuStreamSynchronize | 27 | 1.68 |
| cuCtxGetDevice_v2 | 1847 | 0.08 |
| cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags | 40 | 0.02 |
| cuMemcpyDtoHAsync_v2 | 2 | 0.02 |

## Memcpy

| kind | calls | MiB |
|---|---|---|
| HtoD | 11 | 3.52 |
| DtoH | 2 | 0.58 |
