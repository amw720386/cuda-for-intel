# cuDNN shim conv benchmark: explicit-GEMM engine vs ATen vs old direct kernels

Arc B390, torch 2.2.2+cu118, batch 8, ms per call as fwd/fwd+bwd. aten = cudnn disabled (im2col+cuBLAS), cudnn = this branch. Old direct-kernel numbers (r1): 14-25 ms fwd, 35-87 ms fwd+bwd on the 3x3/1x1/transposed shapes (5-7x slower than aten); nmse columns are y/dx/dw vs CPU fp32 for the cudnn column.

```
name                                              r2_aten     r2_cudnn | fwdbwd ms; nmse_y of last
r18_conv1_7x7s2|fp32|nchw                             err     4.4/15.9 | nmse 6.9e-14 8.4e-14 7.2e-12
r18_conv1_7x7s2|fp32|nhwc                        4.5/17.4     4.4/16.0 | nmse 0.0e+00 8.4e-14 7.2e-12
r18_conv1_7x7s2|fp16|nchw                        2.6/11.8     3.2/10.5 | nmse 4.3e-08 1.3e-07 8.7e-08
r18_conv1_7x7s2|fp16|nhwc                        2.6/11.9     3.5/11.5 | nmse 4.3e-08 1.3e-07 8.7e-08
r18_3x3_64_56|fp32|nchw                          3.4/16.1     3.4/15.3 | nmse 1.3e-13 4.7e-14 0.0e+00
r18_3x3_64_56|fp32|nhwc                          3.5/16.6     4.1/12.4 | nmse 2.8e-13 4.7e-14 0.0e+00
r18_3x3_64_56|fp16|nchw                           2.5/9.3      2.7/8.7 | nmse 4.3e-08 1.3e-07 8.5e-08
r18_3x3_64_56|fp16|nhwc                           2.6/9.7      3.1/8.5 | nmse 4.3e-08 1.3e-07 8.5e-08
r18_3x3s2_64to128|fp32|nchw                       1.7/8.0      1.4/6.1 | nmse 1.3e-13 1.0e-13 0.0e+00
r18_3x3s2_64to128|fp32|nhwc                       1.9/8.0      2.0/5.4 | nmse 2.8e-13 1.0e-13 0.0e+00
r18_3x3s2_64to128|fp16|nchw                       1.4/5.0      0.6/2.1 | nmse 4.3e-08 1.3e-07 8.6e-08
r18_3x3s2_64to128|fp16|nhwc                       1.5/5.2      0.5/2.0 | nmse 4.3e-08 1.3e-07 8.6e-08
r18_3x3_128_28|fp32|nchw                         3.0/12.1     2.7/11.0 | nmse 1.9e-13 8.6e-14 0.0e+00
r18_3x3_128_28|fp32|nhwc                         3.2/12.3     3.7/10.0 | nmse 5.6e-13 8.6e-14 0.0e+00
r18_3x3_128_28|fp16|nchw                          2.5/6.1      1.1/3.0 | nmse 4.3e-08 1.3e-07 8.7e-08
r18_3x3_128_28|fp16|nhwc                          2.4/5.8      1.0/3.0 | nmse 4.3e-08 1.3e-07 8.7e-08
r18_1x1s2_down|fp32|nchw                          0.5/2.2      0.3/1.1 | nmse 0.0e+00 0.0e+00 5.5e-13
r18_1x1s2_down|fp32|nhwc                          0.6/2.6      0.3/1.0 | nmse 0.0e+00 0.0e+00 5.5e-13
r18_1x1s2_down|fp16|nchw                          0.4/2.2      0.2/0.9 | nmse 4.3e-08 8.6e-08 8.8e-08
r18_1x1s2_down|fp16|nhwc                          0.5/2.4      0.1/0.9 | nmse 4.3e-08 8.6e-08 8.8e-08
r18_3x3_256_14|fp32|nchw                         3.0/11.5     2.7/10.1 | nmse 2.6e-13 1.6e-13 0.0e+00
r18_3x3_256_14|fp32|nhwc                         3.0/11.7      3.7/9.5 | nmse 1.1e-12 1.6e-13 0.0e+00
r18_3x3_256_14|fp16|nchw                          4.0/6.4      0.9/1.9 | nmse 4.3e-08 1.3e-07 8.6e-08
r18_3x3_256_14|fp16|nhwc                          4.0/6.4      0.7/1.8 | nmse 4.3e-08 1.3e-07 8.6e-08
r18_3x3_512_7|fp32|nchw                          3.9/14.1     3.0/10.9 | nmse 3.4e-13 2.4e-13 0.0e+00
r18_3x3_512_7|fp32|nhwc                          4.2/14.9      4.0/9.7 | nmse 2.0e-12 2.4e-13 0.0e+00
r18_3x3_512_7|fp16|nchw                           3.5/9.1      2.3/6.0 | nmse 4.2e-08 1.3e-07 8.7e-08
r18_3x3_512_7|fp16|nhwc                           3.8/9.7      0.7/1.8 | nmse 4.2e-08 1.3e-07 8.7e-08
r50_1x1_256to64_56|fp32|nchw                      1.2/9.5      1.6/6.2 | nmse 7.5e-14 0.0e+00 2.1e-12
r50_1x1_256to64_56|fp32|nhwc                     1.9/11.1      1.7/5.8 | nmse 7.5e-14 0.0e+00 2.1e-12
r50_1x1_256to64_56|fp16|nchw                      1.0/7.7      1.3/3.9 | nmse 4.3e-08 8.6e-08 8.6e-08
r50_1x1_256to64_56|fp16|nhwc                      1.4/8.5      1.4/4.3 | nmse 4.3e-08 8.6e-08 8.6e-08
r50_1x1_256to1024_14|fp32|nchw                    1.3/5.1      1.2/4.3 | nmse 7.5e-14 3.6e-13 1.3e-13
r50_1x1_256to1024_14|fp32|nhwc                    1.3/5.2      1.4/3.9 | nmse 7.5e-14 3.6e-13 1.3e-13
r50_1x1_256to1024_14|fp16|nchw                    0.8/3.2      0.3/0.9 | nmse 4.3e-08 8.6e-08 8.6e-08
r50_1x1_256to1024_14|fp16|nhwc                    0.9/3.6      0.2/0.8 | nmse 4.3e-08 8.6e-08 8.6e-08
dw3x3_64_56|fp32|nchw                             0.4/1.9      0.4/1.9 | nmse 0.0e+00 3.7e-15 1.2e-12
dw3x3_64_56|fp32|nhwc                             0.5/2.2      0.6/2.1 | nmse 0.0e+00 3.7e-15 1.2e-12
dw3x3_64_56|fp16|nchw                             0.1/1.1      0.1/1.1 | nmse 4.3e-08 8.6e-08 8.8e-08
dw3x3_64_56|fp16|nhwc                             0.2/1.3      0.3/1.2 | nmse 4.3e-08 8.6e-08 8.8e-08
dw3x3s2_128_28|fp32|nchw                          0.1/0.5      0.1/0.4 | nmse 0.0e+00 1.5e-15 1.4e-13
dw3x3s2_128_28|fp32|nhwc                          0.1/0.6      0.1/0.5 | nmse 0.0e+00 1.5e-15 0.0e+00
dw3x3s2_128_28|fp16|nchw                          0.0/0.3      0.0/0.3 | nmse 4.3e-08 8.7e-08 9.0e-08
dw3x3s2_128_28|fp16|nhwc                          0.1/0.4      0.0/0.3 | nmse 4.3e-08 8.7e-08 9.0e-08
group4_3x3_128_28|fp32|nchw                       2.3/7.5      1.3/4.9 | nmse 8.6e-14 0.0e+00 0.0e+00
group4_3x3_128_28|fp32|nhwc                       1.9/7.7      1.6/5.2 | nmse 1.3e-13 0.0e+00 0.0e+00
group4_3x3_128_28|fp16|nchw                       1.5/6.4      1.2/3.9 | nmse 4.3e-08 1.3e-07 8.6e-08
group4_3x3_128_28|fp16|nhwc                       1.5/6.8      1.3/4.2 | nmse 4.3e-08 1.3e-07 8.6e-08
convT_k4s2_64to32_32|fp32|nchw                    2.1/4.8      1.2/3.9 | nmse 3.9e-14 1.6e-13 0.0e+00
convT_k4s2_64to32_32|fp32|nhwc                    2.2/4.9      1.1/3.7 | nmse 3.9e-14 2.4e-13 0.0e+00
convT_k4s2_64to32_32|fp16|nchw                    1.3/3.6      0.5/2.7 | nmse 8.7e-08 8.6e-08 8.5e-08
convT_k4s2_64to32_32|fp16|nhwc                    1.4/3.7      0.5/2.7 | nmse 8.7e-08 8.6e-08 8.5e-08
convT_k2s2_128to64_28|fp32|nchw                   2.0/4.5      0.9/3.0 | nmse 0.0e+00 5.5e-14 0.0e+00
convT_k2s2_128to64_28|fp32|nhwc                   2.1/4.6      0.8/2.6 | nmse 0.0e+00 1.2e-13 0.0e+00
convT_k2s2_128to64_28|fp16|nchw                   1.5/3.9      0.3/1.4 | nmse 4.3e-08 8.6e-08 8.6e-08
convT_k2s2_128to64_28|fp16|nhwc                   1.6/4.1      0.3/1.3 | nmse 4.3e-08 8.6e-08 8.6e-08
```

ResNet18 112px batch 8, one fwd+bwd step (s) and worst NMSE vs CPU:
```
{"nchw_fp32": {"nmse_logits": 1.0377307119169032e-11, "nmse_grad_worst": 7.62808104885074e-05, "step_s": 0.08809209999162704}, "nchw_fp16": {"nmse_logits": 4.371040477312457e-06, "nmse_grad_worst": 0.04068307461785768, "step_s": 0.06599050003569573}, "nhwc_fp32": {"nmse_logits": 1.0377307119169032e-11, "nmse_grad_worst": 7.62808104885074e-05, "step_s": 0.08845849998760968}, "nhwc_fp16": {"nmse_logits": 4.371040477312457e-06, "nmse_grad_worst": 0.04068307461785768, "step_s": 0.06758780003292486}, "mode": "aten", "N": 8, "size": 112}
{"nchw_fp32": {"nmse_logits": 1.1302110852255338e-11, "nmse_grad_worst": 0.00011759526320010704, "step_s": 0.06506039999658242}, "nchw_fp16": {"nmse_logits": 4.974202643737195e-06, "nmse_grad_worst": 0.043718200666907195, "step_s": 0.044702500046696514}, "nhwc_fp32": {"nmse_logits": 9.970569523489973e-12, "nmse_grad_worst": 7.1734568742983e-05, "step_s": 0.06094180000945926}, "nhwc_fp16": {"nmse_logits": 4.6534177639298655e-06, "nmse_grad_worst": 0.04422669356389313, "step_s": 0.044268300000112504}, "mode": "cudnn", "N": 8, "size": 112}```

Old direct baseline (r1, 3x3/1x1/convT rows):
```
name                                              r1_aten    r1_direct | fwdbwd ms; nmse_y of last
r18_conv1_7x7s2|fp32|nchw                             err          err | ERR RuntimeError('GET was unable to find an 
r18_conv1_7x7s2|fp32|nhwc                        3.8/17.5          err | ERR RuntimeError('GET was unable to find an 
r18_conv1_7x7s2|fp16|nchw                        2.6/11.9          err | ERR RuntimeError('GET was unable to find an 
r18_conv1_7x7s2|fp16|nhwc                        2.6/12.2          err | ERR RuntimeError('GET was unable to find an 
r18_3x3_64_56|fp32|nchw                          3.4/16.2          err | ERR RuntimeError('GET was unable to find an 
r18_3x3_64_56|fp32|nhwc                          3.5/16.6          err | ERR RuntimeError('GET was unable to find an 
r18_3x3_64_56|fp16|nchw                           2.7/9.4          err | ERR RuntimeError('GET was unable to find an 
r18_3x3_64_56|fp16|nhwc                           2.5/9.6          err | ERR RuntimeError('GET was unable to find an 
r18_3x3s2_64to128|fp32|nchw                       1.7/8.3          err | ERR RuntimeError('GET was unable to find an 
r18_3x3s2_64to128|fp32|nhwc                       1.8/8.1          err | ERR RuntimeError('GET was unable to find an 
r18_3x3s2_64to128|fp16|nchw                       1.4/5.0          err | ERR RuntimeError('GET was unable to find an 
r18_3x3s2_64to128|fp16|nhwc                       1.4/4.9          err | ERR RuntimeError('GET was unable to find an 
r18_3x3_128_28|fp32|nchw                         3.0/12.1    23.2/83.4 | nmse 1.9e-13 3.9e-13 0.0e+00
r18_3x3_128_28|fp32|nhwc                         3.1/12.3    23.2/83.2 | nmse 1.9e-13 3.9e-13 0.0e+00
r18_3x3_128_28|fp16|nchw                          2.5/6.0     7.6/35.2 | nmse 4.3e-08 8.7e-08 8.7e-08
r18_3x3_128_28|fp16|nhwc                          2.5/6.1     7.8/35.5 | nmse 4.3e-08 8.7e-08 8.7e-08
r18_1x1s2_down|fp32|nchw                          0.5/2.2     1.8/11.7 | nmse 0.0e+00 0.0e+00 5.5e-13
r18_1x1s2_down|fp32|nhwc                          0.7/2.6     1.8/12.1 | nmse 0.0e+00 0.0e+00 5.5e-13
r18_1x1s2_down|fp16|nchw                          0.4/2.2      0.9/7.9 | nmse 4.3e-08 8.6e-08 8.8e-08
r18_1x1s2_down|fp16|nhwc                          0.5/2.3      0.9/8.5 | nmse 4.3e-08 8.6e-08 8.8e-08
r18_3x3_256_14|fp32|nchw                         3.0/11.8    24.1/85.6 | nmse 2.6e-13 7.6e-13 0.0e+00
r18_3x3_256_14|fp32|nhwc                         3.1/11.8    24.1/85.7 | nmse 2.6e-13 7.6e-13 0.0e+00
r18_3x3_256_14|fp16|nchw                          4.1/6.5     7.8/35.9 | nmse 4.3e-08 8.6e-08 8.6e-08
r18_3x3_256_14|fp16|nhwc                          4.0/6.5     7.9/36.4 | nmse 4.3e-08 8.6e-08 8.6e-08
r18_3x3_512_7|fp32|nchw                          4.0/14.2    25.0/87.1 | nmse 3.4e-13 1.4e-12 0.0e+00
r18_3x3_512_7|fp32|nhwc                          4.4/14.8    24.6/85.8 | nmse 3.4e-13 1.4e-12 0.0e+00
r18_3x3_512_7|fp16|nchw                           3.4/9.1     8.0/36.4 | nmse 4.2e-08 8.7e-08 8.7e-08
r18_3x3_512_7|fp16|nhwc                           3.6/9.3     8.0/36.8 | nmse 4.2e-08 8.7e-08 8.7e-08
r50_1x1_256to64_56|fp32|nchw                      1.2/9.4    14.2/47.8 | nmse 7.5e-14 0.0e+00 2.1e-12
r50_1x1_256to64_56|fp32|nhwc                     1.8/11.0    16.9/54.3 | nmse 7.5e-14 0.0e+00 2.1e-12
r50_1x1_256to64_56|fp16|nchw                      1.0/7.6     6.9/26.5 | nmse 4.3e-08 8.6e-08 8.6e-08
r50_1x1_256to64_56|fp16|nhwc                      1.4/8.5    13.9/41.3 | nmse 4.3e-08 8.6e-08 8.6e-08
r50_1x1_256to1024_14|fp32|nchw                    1.3/5.1    14.2/48.0 | nmse 7.5e-14 3.6e-13 1.3e-13
r50_1x1_256to1024_14|fp32|nhwc                    1.3/5.2    14.2/48.0 | nmse 7.5e-14 3.6e-13 1.3e-13
r50_1x1_256to1024_14|fp16|nchw                    0.8/3.3     6.9/26.1 | nmse 4.3e-08 8.6e-08 8.6e-08
r50_1x1_256to1024_14|fp16|nhwc                    0.8/3.0     6.9/25.6 | nmse 4.3e-08 8.6e-08 8.6e-08
dw3x3_64_56|fp32|nchw                             0.4/1.9      0.4/1.9 | nmse 0.0e+00 3.7e-15 1.2e-12
dw3x3_64_56|fp32|nhwc                             0.5/2.2     0.6/13.1 | nmse 0.0e+00 3.7e-15 2.2e-12
dw3x3_64_56|fp16|nchw                             0.1/1.1      0.1/1.1 | nmse 4.3e-08 8.6e-08 8.8e-08
dw3x3_64_56|fp16|nhwc                             0.2/1.3      0.3/5.4 | nmse 4.3e-08 8.6e-08 8.8e-08
dw3x3s2_128_28|fp32|nchw                          0.1/0.4      0.1/0.4 | nmse 0.0e+00 1.5e-15 1.4e-13
dw3x3s2_128_28|fp32|nhwc                          0.1/0.6      0.1/0.8 | nmse 0.0e+00 1.5e-15 0.0e+00
dw3x3s2_128_28|fp16|nchw                          0.0/0.3      0.0/0.3 | nmse 4.3e-08 8.7e-08 9.0e-08
dw3x3s2_128_28|fp16|nhwc                          0.1/0.4      0.0/0.6 | nmse 4.3e-08 8.7e-08 9.0e-08
group4_3x3_128_28|fp32|nchw                       2.2/7.7     5.8/23.3 | nmse 8.6e-14 1.0e-13 0.0e+00
group4_3x3_128_28|fp32|nhwc                       1.9/7.7     5.8/24.6 | nmse 8.6e-14 1.0e-13 0.0e+00
group4_3x3_128_28|fp16|nchw                       1.5/6.7     1.9/10.1 | nmse 4.3e-08 8.6e-08 8.6e-08
```
