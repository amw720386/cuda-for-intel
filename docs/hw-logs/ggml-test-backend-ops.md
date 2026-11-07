# GGML test-backend-ops on the shim (Arc B390, CUDA backend, PTX-only)

Build: llama.cpp b12190, CUDA 13.5 nvcc, `-DCMAKE_CUDA_ARCHITECTURES=75-virtual` (compute_75 PTX only, no SASS), shim from branch `integration-ggml-ops` (integration + perf-jit lazy modules and persistent compile cache + the fixes listed below). GGUFs: Qwen2.5-0.5B-Instruct Q8_0 / Q4_K_M. Scrubbed of paths and hosts.

Method: per-case isolation (`test-backend-ops test -b CUDA0 -o '<full case string>'`, one process per case) because a CUDA error aborts the whole run; case list taken from the CPU reference run; queue slices of at most 4 minutes under the shared GPU queue. NMSE comes from a local patch to test-backend-ops (stderr line per case, max over non-input nodes; limit shown is the test's own max_nmse). `TBO_QWEN=1` adds a Qwen2.5-0.5B shaped MUL_MAT set (m/k from 896, 128, 4864, 151936; n = 1,2,8,32,128,512; K-quants only where k is a multiple of 256; plus f16 K/Q attention shapes with GQA bs=[2,1], nr=[7,1]).

## Qwen-shaped MUL_MAT (NMSE per chunk)

| chunk | pass | fail | crash | max NMSE | NMSE limit |
|---|---:|---:|---:|---:|---:|
| qwen_attn_kq | 14 | 1 | 0 | 1.42e-06 | 5e-4 |
| qwen_attn_kqv | 12 | 3 | 0 | 1.10e-05 | 5e-4 |
| qwen_f16_m128 | 11 | 0 | 0 | 1.94e-05 | 5e-4 |
| qwen_f16_m151936 | 6 | 0 | 0 | 1.93e-05 | 5e-4 |
| qwen_f16_m4864 | 6 | 0 | 0 | 1.93e-05 | 5e-4 |
| qwen_f16_m896 | 12 | 0 | 0 | 1.04e-04 | 5e-4 |
| qwen_f32_m128 | 6 | 0 | 0 | 3.03e-13 | 5e-4 |
| qwen_f32_m151936 | 6 | 0 | 0 | 3.05e-13 | 5e-4 |
| qwen_f32_m4864 | 6 | 0 | 0 | 3.05e-13 | 5e-4 |
| qwen_f32_m896 | 12 | 0 | 0 | 1.63e-12 | 5e-4 |
| qwen_q4_0_m128 | 6 | 0 | 0 | 2.88e-07 | 5e-4 |
| qwen_q4_0_m151936 | 6 | 0 | 0 | 2.73e-07 | 5e-4 |
| qwen_q4_0_m4864 | 6 | 0 | 0 | 2.57e-07 | 5e-4 |
| qwen_q4_0_m896 | 12 | 0 | 0 | 2.80e-07 | 5e-4 |
| qwen_q4_K_m896 | 6 | 0 | 0 | 2.53e-05 | 5e-4 |
| qwen_q8_0_m128 | 6 | 0 | 0 | 3.55e-08 | 5e-4 |
| qwen_q8_0_m151936 | 6 | 0 | 0 | 3.57e-08 | 5e-4 |
| qwen_q8_0_m4864 | 6 | 0 | 0 | 3.64e-08 | 5e-4 |
| qwen_q8_0_m896 | 12 | 0 | 0 | 3.49e-08 | 5e-4 |
| soft_max_all | 0 | 0 | 0 | 7.63e-15 | 5e-4 |

## Failing or crashing cases

| chunk | case | status |
|---|---|---|
| qwen_attn_kq | `f16,type_b=f32,m=512,n=512,k=64 bs=[2,1],nr=[7,1]` | CRASH(exit=-1073740791) CUDA error: CUBLAS_STATUS_EXECUTION_FAILED / ggml-cuda.cu:111: CUDA error |
| qwen_attn_kqv | `f16,type_b=f32,m=64,n=32,k=128 bs=[2,1],nr=[7,1]` | CRASH(exit=-1073740791) CUDA error: CUBLAS_STATUS_EXECUTION_FAILED / ggml-cuda.cu:111: CUDA error |
| qwen_attn_kqv | `f16,type_b=f32,m=64,n=32,k=512 bs=[2,1],nr=[7,1]` | CRASH(exit=-1073740791) CUDA error: CUBLAS_STATUS_EXECUTION_FAILED / ggml-cuda.cu:111: CUDA error |
| qwen_attn_kqv | `f16,type_b=f32,m=64,n=512,k=512 bs=[2,1],nr=[7,1]` | CRASH(exit=-1073740791) CUDA error: CUBLAS_STATUS_EXECUTION_FAILED / ggml-cuda.cu:111: CUDA error |

## Perplexity, GPU vs CPU (wikitext-2 test, first 20 chunks, -c 512, flash attention off, f16 KV)

Q8_0 (running PPL after N chunks):

| chunk | CPU | GPU | rel. diff |
|---:|---:|---:|---:|
| 1 | 9.9685 | 9.8903 | -0.78% |
| 2 | 13.6278 | 13.5554 | -0.53% |
| 3 | 13.7812 | 13.7009 | -0.58% |
| 4 | 13.6389 | 13.5866 | -0.38% |
| 5 | 13.7623 | 13.7354 | -0.20% |
| 6 | 14.1998 | 14.1853 | -0.10% |
| 7 | 14.5252 | 14.5169 | -0.06% |
| 8 | 15.6358 | 15.6252 | -0.07% |
| 9 | 16.6222 | 16.6177 | -0.03% |
| 10 | 17.1937 | 17.1877 | -0.03% |
| 11 | 17.4675 | 17.4499 | -0.10% |
| 12 | 17.5570 | 17.5398 | -0.10% |
| 13 | 18.3663 | 18.3458 | -0.11% |
| 14 | 17.8293 | 17.8054 | -0.13% |
| 15 | 17.7245 | 17.7006 | -0.13% |
| 16 | 18.2006 | 18.1764 | -0.13% |
| 17 | 16.9095 | 16.8881 | -0.13% |
| 18 | 17.0310 | 17.0159 | -0.09% |
| 19 | 16.8337 | 16.8182 | -0.09% |
| 20 | 16.8323 | 16.8142 | -0.11% |

Final PPL: CPU 16.8323, GPU 16.8142 (-0.11%). The GPU value is below the CPU value at every chunk, so this is a small systematic offset, not noise. It is about 10x larger than the usual CUDA-vs-CPU difference for Q8_0, and above the requested 1e-3 relative on the first chunks (0.78% at chunk 1, 0.13% at chunk 5). Not yet explained. Candidates not yet isolated: f16 KV conversion (SET_ROWS/CPY rounding), SOFT_MAX, ROPE, RMS_NORM, the cuBLAS fp16 path for n>=32. Those ops are queued in the same suite. CPU reference with flash attention auto differs from flash attention off by 0.26% at chunk 1, so a CPU-side spread of that order already exists. Q4_K_M: CPU reference done (final [20] = 17.3797); the GPU run is queued.

## llama-bench (flash attention off, -ngl 99, -r 2)

| model | pp512 t/s | tg128 t/s |
|---|---:|---:|
| Qwen2.5-0.5B Q8_0 | 303.79 +/- 124.94 | 18.27 +/- 4.21 |
| Qwen2.5-0.5B Q4_K_M | queued | queued |

The large spread comes from sharing the GPU with other testers and first-run JIT.

## Findings and fixes (branch integration-ggml-ops)

1. `ld.global.nc.v2/v4.u8` unsupported (GET_ROWS iq3_xxs / iq3_s, ADD with those types): added LoadVecU8.
2. Dynamic shared window was fixed at 48 KiB: mmq_x=128 (Q8_0/Q4_0, n>=128) requests 57856 B, launch returned invalid argument and aborted every MUL_MAT with n>=128. The window now fills 64 KiB minus the entry's static shared arrays.
3. `ld.global.s32/s16/s8` into a 64-bit register (k_get_rows_float_vec) was rejected: now sign-extends.
4. Open, in the cuBLAS shim: batched f16 GEMM for GQA attention intermittently fails with stream sync err=600/719 (CUBLAS_STATUS_EXECUTION_FAILED): kq m=512,n=512,k=64; kqv m=64 with (n=32,k=128), (n=32,k=512), (n=512,k=512). Other shapes pass.
5. Flash attention (llama-perplexity default) fails translation (known, being fixed elsewhere), so all PPL numbers use `-fa off`.
6. PTX JIT of the mul_mat_q templates costs minutes per fresh process without the persistent cache (driver module build dominates); with `CUDA_FOR_INTEL_CACHE_DIR` set, later runs take about 5 s per case.

## Not done yet

SOFT_MAX, ROPE, RMS_NORM, MUL_MAT_ID, FLASH_ATTN_EXT and the rest of the op list are queued behind a single runner; this file will be updated when they finish. The generic 3118-case MUL_MAT list was dropped (each distinct template instance costs minutes of JIT).
