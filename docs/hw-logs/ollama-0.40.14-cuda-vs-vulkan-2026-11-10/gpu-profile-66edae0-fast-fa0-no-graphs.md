decode wall (eval) 32.28 ms/token = 31.0 tok/s
GPU kernel time per token: 15710.6 us over 486.0 launches (0 untimed launches)
| # | kernel | launches per token | GPU us per token | % | avg us |
|---|---|---|---|---|---|
| 1 | `mul_mat_vec_q<8,1,1,1>` | 119.0 | 8933.2 | 56.9 | 75.07 |
| 2 | `mul_mat_vec_q<8,1,1,0>` | 24.0 | 2703.4 | 17.2 | 112.64 |
| 3 | `mul_mat_f<32,1,1,0>` | 24.0 | 1497.0 | 9.5 | 62.38 |
| 4 | `mul_mat_vec_q<8,1,0,1>` | 2.0 | 1356.8 | 8.6 | 678.40 |
| 5 | `mul_mat_vec_f<1,192,0,0>` | 24.0 | 469.7 | 3.0 | 19.57 |
| 6 | `quantize_q8_1` | 145.0 | 211.6 | 1.3 | 1.46 |
| 7 | `rms_norm_f32<256,1,1,1>` | 49.0 | 152.2 | 1.0 | 3.11 |
| 8 | `soft_max_f32<1,0,0>` | 24.0 | 120.0 | 0.8 | 5.00 |
| 9 | `rope_neox<1,0>` | 24.0 | 94.0 | 0.6 | 3.92 |
| 10 | `rope_neox<1,0>` | 24.0 | 90.4 | 0.6 | 3.77 |
| 11 | `k_set_rows` | 24.0 | 79.9 | 0.5 | 3.33 |
| 12 | `k_get_rows_float` | 2.0 | 4.0 | 0.0 | 2.00 |
| 13 | `k_bin_bcast` | 1.0 | 1.8 | 0.0 | 1.83 |
| 14 | `mul_mat_q<8,128,0,30>` | 0.0 | 1.7 | 0.0 | 213.02 |
| 15 | `convert_unary_cont_vec4` | 0.0 | 0.1 | 0.0 | 7.92 |
| 16 | `cpy_scalar` | 0.0 | -0.0 | -0.0 | -1.35 |
| 17 | `k_compute_batched_ptrs` | 0.0 | -0.0 | -0.0 | -2.40 |
| 18 | `unary_gated_op_kernel` | 0.0 | -0.0 | -0.0 | -4.58 |

