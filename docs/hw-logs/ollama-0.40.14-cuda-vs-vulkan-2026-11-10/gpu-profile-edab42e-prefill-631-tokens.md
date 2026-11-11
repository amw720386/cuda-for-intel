Request B (631-token prompt, 1 generated token): GPU kernel time 683.3 ms over 1554 timed launches (284 untimed); server prompt eval 756-759 ms (831-835 tok/s)
| # | kernel | launches | GPU ms | % | avg ms | max ms (whole run) |
|---|---|---|---|---|---|---|
| 1 | `mul_mat_q<8,128,0,30>` | 23 | 483.99 | 70.8 | 21.043 | 47.06 |
| 2 | `mul_mat_q<8,128,0,30>` | 284 | 129.32 | 18.9 | 0.455 | 2.63 |
| 3 | `cfi_gemm_f32f32f32p` | 48 | 22.51 | 3.3 | 0.469 | 1.92 |
| 4 | `soft_max_f32<1,512,512>` | 24 | 11.08 | 1.6 | 0.462 | 1.30 |
| 5 | `quantize_mmq_q8_1<8,0>` | 307 | 8.38 | 1.2 | 0.027 | 0.19 |
| 6 | `soft_max_f32<1,0,0>` | 24 | 5.85 | 0.9 | 0.244 | 0.25 |
| 7 | `cfi_xmx_f16f16p_2x4_b0` | 48 | 5.70 | 0.8 | 0.119 | 0.37 |
| 8 | `k_bin_bcast` | 237 | 3.44 | 0.5 | 0.015 | 0.09 |
| 9 | `cpy_scalar` | 48 | 2.15 | 0.3 | 0.045 | 0.19 |
| 10 | `convert_unary_cont_vec4` | 48 | 1.99 | 0.3 | 0.041 | 0.18 |
| 11 | `rope_neox<1,0>` | 48 | 1.81 | 0.3 | 0.038 | 0.16 |
| 12 | `mul_mat_vec_q<8,1,0,1>` | 1 | 1.79 | 0.3 | 1.787 | 2.21 |
| 13 | `unary_gated_op_kernel` | 23 | 1.68 | 0.2 | 0.073 | 0.11 |
| 14 | `rms_norm_f32<256,1,1,1>` | 96 | 1.44 | 0.2 | 0.015 | 0.06 |
| 15 | `convert_unary_cont_vec4` | 96 | 0.70 | 0.1 | 0.007 | 0.03 |
