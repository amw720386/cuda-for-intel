# Dynamic shared window sized to the launch, Arc B390, 2026-11-10

Arc B390 (8086:B080, driver 32.0.101.9033, Core Ultra X7 358H), stock Ollama v0.12.6 on the shim, Qwen2.5-0.5B-Instruct Q8_0 GGUF
`ca59ca7f…`, SUBST auto, MMVQ fast path off (main's defaults). No other GPU job ran during these measurements.

## Problem

`normalize` gives every kernel that uses `extern __shared__` a dynamic window of `min(device max, CUDA_FOR_INTEL_DYN_SHARED, 64 KiB)`
(65536 B here; the device reports 131072 B per Xe-core), fixed at translation time, and a workgroup reserves its whole shared array.
ggml's decode attention launches these kernels with tiny requests, logged per launch with `CUDA_FOR_INTEL_LOG=1`:

| kernel | `sharedMemBytes` | window it ran with |
|---|---|---|
| `mul_mat_f<half2,32,1,1,false>` (KQ, FA off) | 2304 | 65536 |
| `mul_mat_vec_f<half,…,1,192>` (KQV, FA off) | 128 | 65536 |
| `soft_max_f32` | 2176 to 8320 | 65536 |
| `mul_mat_q<q8_0,128>` (prefill) | 57856 | 65536 (unchanged by this fix) |

The reservation alone caps occupancy. `hw_slm_reservation_occupancy` (new) runs one 32-thread kernel, 16 strided loads per thread,
with a workgroup `.shared` array of the given size; the results are bit-identical across sizes:

| workgroups | 65536 B | 49152 B | 16384 B | 4096 B |
|---|---|---|---|---|
| 448 | 18.27 us | 18.29 us | 5.38 us | **4.13 us** |
| 4096 | 224.25 us | 224.25 us | 63.37 us | **31.08 us** |

## Where decode time goes (`CFI_GPU_PROFILE`, kernel timestamps, launches not serialised)

Decode-only per token (two requests with the same 631-token prompt length, 8 and 136 generated tokens, difference / 128), FA off:

| kernel | launches/token | main: GPU us/token (us/launch) | this change |
|---|---|---|---|
| `mul_mat_vec_q` (stock-order mmvq) | 169 | 18024 (107) | 16793 (99) |
| `mul_mat_f` | 24 | **6681 (278)** | **1045 (44)** |
| `mul_mat_vec_f` | 23.6 | **1620 (69)** | **499 (21)** |
| all other kernels (~508 launches) | | ~1370 | ~1300 |
| total GPU kernel time | 725 | 27.7 ms | 19.6 ms |

With the profiler on, decode went from 26.5 to 36.9 tok/s. The profiler adds a near-constant cost per kernel, so use the unprofiled
numbers below for speed. Host launch cost is not the bottleneck: `hw_launch_host_overhead_by_param_count` (new) measures 2.5 to 2.9 us
of host time per `cuLaunchKernel` at ggml's 22 to 24 parameters, about 1.9 ms per token for 725 launches, and with #18's fast mmvq the Windows
GPU-engine counters showed the compute engine busy for 76 % of decode wall time.

## Unprofiled end to end (`scripts/bench-decode.ps1` from #18, 631-token prompt, 48 greedy tokens)

3 fresh servers x 5 measured runs per cell (n = 15); main is b1046ca (`nvcuda.dll` `afdc789a…`), this change `4ca5f9da…` (built
from this branch plus the separate, dormant `CFI_GPU_PROFILE` change, which is off unless set); every
server log shows `library=CUDA`, 25/25 layers on CUDA0 and the requested `flash_attn`; the counting text equals the CPU backend in
all 12 servers and the free-text story prompt equals it too (`story_diff` -1 everywhere).

| FA | build | prefill p50 | decode p50 | decode p5-p95 | per-server decode p50 |
|---|---|---|---|---|---|
| off | main | 7015 | 31.4 | 30.2-37.4 | 31.0 / 30.9 / 36.9 |
| off | this change | 6967 | **37.4 (+19 %)** | 35.0-40.1 | 36.3 / 36.6 / 38.7 |
| on | main | 4496 | 38.8 | 36.9-50.0 | 37.5 / 49.9 / 38.4 |
| on | this change | **6190 (+38 %)** | 39.9 (+3 %, noise) | 37.2-42.1 | 40.3 / 40.4 / 37.5 |

FA on, the variants are `flash_attn_ext_f16` (the prefill tile kernel) and `flash_attn_combine_results`, hence the prefill gain;
the FA decode kernels are not affected.

Decode on this iGPU is bimodal per server (about 31 or 39 tok/s FA off, 38 or 50 FA on, sometimes switching within one server,
see main's third and second servers), most likely clock or power state shared with the CPU; compare pooled medians over at least
3 servers, not single runs.

## Caveats

- `cfi-precompile` still translates every entry with the full window, so the first launch of each (kernel, window) pair translates
  and builds its variant at run time: 65 to 570 ms per variant here, 11 variants for this model with FA off and 2 with FA on. They are then in the
  translation cache.
- Not measured on a second Intel architecture yet: the Arc 140V box (#17) has no working GPU driver. Xe2 has a different shared
  memory budget per Xe-core, so the size of the gain there is open.
- `CUDA_FOR_INTEL_DYN_SHARED_SPECIALIZE=0` restores the old behaviour.

## Reproduce

- `cargo test --release --test gpu hw_slm_reservation_occupancy hw_launch_host -- --ignored --nocapture --test-threads=1`
- End to end: `scripts/bench-decode.ps1 -Tree <stock v0.12.6 tree with the shim> -Models <qwen2.5-0.5b-instruct-q8_0.gguf> -Out <dir>
  -Configs 'a=|b=|c=' -Fa '0,1' -Runs 5` once per build (from #18's branch until it is merged).
