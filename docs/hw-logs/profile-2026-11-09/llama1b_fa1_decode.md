## Kernels by time (serialised host-side timing; rank, do not read absolute speed)

25893 launches, 1340.30 ms total, submit 60.85 ms, wait 1279.45 ms

| # | kernel | launches | total ms | % | avg us | submit % |
|---|---|---|---|---|---|---|
| 1 | `_Z13mul_mat_vec_qIL9ggml_type12ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6...` | 5965 | 454.57 | 33.9 | 76.2 | 3 |
| 2 | `_Z13mul_mat_vec_qIL9ggml_type14ELi1EEvPKvS2_PKiPfj5uint3jjjS6_jjjS6...` | 1061 | 420.13 | 31.3 | 396.0 | 1 |
| 3 | `_Z18flash_attn_ext_f16ILi64ELi64ELi2ELi4ELi4ELi1ELb0ELb0EEvPKcS1_S1...` | 994 | 175.50 | 13.1 | 176.6 | 2 |
| 4 | `_Z18flash_attn_ext_f16ILi64ELi64ELi8ELi4ELi4ELi2ELb0ELb0EEvPKcS1_S1...` | 22 | 49.19 | 3.7 | 2236.1 | 0 |
| 5 | `_Z13quantize_q8_1PKfPvxxxxxj5uint3` | 7026 | 48.04 | 3.6 | 6.8 | 28 |
| 6 | `_Z9mul_mat_qIL9ggml_type12ELi16ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 94 | 38.54 | 2.9 | 410.0 | 1 |
| 7 | `_Z9mul_mat_qIL9ggml_type12ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiii...` | 37 | 29.69 | 2.2 | 802.5 | 0 |
| 8 | `_Z12rms_norm_f32ILi1024ELb1ELb0EEvPKfPfixxxfS1_xxx5uint3S3_S3_S3_S1...` | 2096 | 24.78 | 1.8 | 11.8 | 28 |
| 9 | `_Z9rope_normILb1ELb1EfEvPKT1_PS0_iiiiiPKifff14rope_corr_dimsfPKf` | 2032 | 18.92 | 1.4 | 9.3 | 23 |
| 10 | `_Z10k_set_rowsIfx6__halfEvPKT_PKT0_PT1_xxxxxxxxxxxxxxxxx` | 2032 | 18.55 | 1.4 | 9.1 | 27 |
| 11 | `_Z11k_bin_bcastIXadL_ZN42_INTERNAL_bf62842d_11_binbcast_cu_6840010b...` | 2032 | 18.39 | 1.4 | 9.0 | 32 |
| 12 | `_Z9mul_mat_qIL9ggml_type14ELi16ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiiii...` | 15 | 11.66 | 0.9 | 777.0 | 0 |
| 13 | `_Z25flash_attn_stream_k_fixupILi64ELi2ELi4EEvPfPK6float2iiii` | 994 | 10.59 | 0.8 | 10.7 | 18 |
| 14 | `_Z21unary_gated_op_kernelIXadL_ZN38_INTERNAL_5f3b6e47_8_unary_cu_eb...` | 1016 | 10.05 | 0.8 | 9.9 | 19 |
| 15 | `_Z9mul_mat_qIL9ggml_type14ELi112ELb0EEvPKcPKiS4_S4_PfS5_iiiiiiiiiii...` | 4 | 5.64 | 0.4 | 1409.0 | 0 |

## Driver API calls by host time

| api | calls | total ms |
|---|---|---|
| cuLaunchKernel | 25893 | 1369.82 |
| cuMemcpyHtoDAsync_v2 | 384 | 50.38 |
| cuStreamSynchronize | 960 | 23.47 |
| cuCtxGetDevice_v2 | 20480 | 0.93 |
| cuMemcpyDtoHAsync_v2 | 128 | 0.57 |
| cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags | 1016 | 0.32 |
| cuCtxSetCurrent | 1 | 0.00 |
| cuCtxGetApiVersion | 1 | 0.00 |
| cuCtxGetCurrent | 1 | 0.00 |

## Memcpy

| kind | calls | MiB |
|---|---|---|
| DtoH | 128 | 31.81 |
| HtoD | 384 | 4.61 |
