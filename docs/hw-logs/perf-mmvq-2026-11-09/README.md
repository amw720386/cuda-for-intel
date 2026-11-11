# Decode mmvq and FlashAttention on the Arc B390, 2026-11-09

Arc B390 (driver 32.0.101.9033, LPDDR5X-8533), stock Ollama v0.12.6 on the shim, SUBST auto. Other agents shared the box
(GPU runs were serialised by the queue, but their CPU work shares the memory bus), so single decode numbers move by about +/-10 %.

## Ceilings (`hw_perf_probe`)

| what | result |
|---|---|
| global read bandwidth, random data, 512 MiB | **114 GB/s** (16-byte and 8-byte loads; 4-byte loads with 4 in flight also 114; 4-byte single load 100-110) |
| same, memset-filled buffer | 540 GB/s (the constant fill is served compressed; never benchmark on memset data) |
| back-to-back trivial launches, 1 block | 2.9 us host, 3.9 us to completion per launch |
| back-to-back trivial launches, 896 / 4864 blocks | 4.7 / 8.8 us per launch (GPU-bound; about 1 ns per block) |
| launch + cuStreamSynchronize round trip | 7-12 us |

At about 690 launches per decoded token (Qwen2.5-0.5B), the per-launch floor is about 2.8 ms of GPU time per token.

## Why mmvq ran at about 40 GB/s

The stock-order native mmvq reproduces ggml's thread split: one 128-thread block per row with a shared-memory reduction, and 4 to 32
lanes per quantized block, each loading 2 to 8 bytes as 16-bit halves (blocks are only 2-byte aligned). Too few bytes per load and too
few loads in flight: 31-53 GB/s at decode shapes (`mmvq_decode_shapes_ggml_layout.txt`). One subgroup per row with more accumulators
(same per-lane split) reaches 45-58 GB/s; giving each lane a whole 32-value group read as aligned dwords (`w` variants) reaches
106-109 GB/s on the output matrices and 55-75 GB/s on mid-size ones (`mmvq_decode_shapes_wide.txt`).

us per launch / GB/s (weights + activations), from `mmvq_decode_shapes_wide.txt`:

| shape | type | translated | native (stock order) | fast 4x1x2w (default) |
|---|---|---|---|---|
| Qwen 0.5B gate/up 4864x896 | Q8_0 | 175 / 26 | 154 / 30 | 81 / 58 |
| Qwen 0.5B lm_head 151936x896 | Q8_0 | 4746 / 31 | 4027 / 36 | 1345 / 108 |
| Llama 1B gate/up 8192x2048 | Q4_K | 208 / 46 | 155 / 61 | 132 / 72 |
| Llama 1B down 2048x8192 | Q6_K | 930 / 15 | 498 / 28 | 238 / 58 |
| Llama 1B lm_head 128256x2048 | Q6_K | 6285 / 34 | 4090 / 53 | 2025 / 106 |
| Qwen 1.5B lm_head 151936x1536 | Q6_K | 6659 / 29 | 4467 / 43 | 1806 / 106 |

(`mmvq_decode_shapes_wide_q8.txt`: its `f4x1x4w` column reused the non-wide kernel through a cache-key collision fixed before the other runs.)

## End to end (`bench_b2_fast_fattn.json`, `scripts/bench-decode.ps1`; `bench_b1_mmvq_fast.json` is an earlier 2-run pass)

631-token prompt, 48 greedy tokens, warm (the warm-up prompt differs from the measured ones, so the KV prefix cache cannot skip the
prefill), best of 3 runs. `base` = previous defaults (`CUDA_FOR_INTEL_MMVQ_FAST=0`, translated FlashAttention); `new` = this branch's
defaults (decode-fast mmvq and native FlashAttention). tok/s, prefill / decode:

| model | FA | base | new | prefill | decode |
|---|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | off | 7072 / 32.0 | 7351 / 45.7 | +4 % | **+43 %** |
| Qwen2.5-0.5B Q8_0 | on | 4504 / 38.0 | **8158** / 57.8 | **+81 %** | **+52 %** |
| Llama-3.2-1B Q4_K_M | off | 5229 / 32.7 | 5263 / 36.6 | +1 % | +12 % |
| Llama-3.2-1B Q4_K_M | on | 3020 / 51.9 | **5639** / 52.4 | **+87 %** | +1 % (noisy, see below) |
| Qwen2.5-1.5B Q4_K_M | off | 4038 / 25.2 | 4102 / 30.6 | +2 % | +21 % (best of `fast`: 34.4) |
| Qwen2.5-1.5B Q4_K_M | on | 2111 / 24.6 | **3967** / 34.6 | **+88 %** | **+41 %** |

`fast` alone (mmvq only) is in the JSON as well. Decode on this shared box moves by up to 20 % between runs of one server (Llama FA on,
base: 42.4 / 51.5 / 51.9), so treat the Q4_K_M decode rows as "no regression, modest gain": at kernel level every Q4_K/Q6_K shape is as
fast or faster (table above). FA-on prefill is now faster than FA-off for Qwen2.5-0.5B and Llama-3.2-1B.

Text: the counting prompt's 48 greedy tokens equal the CPU backend in every cell. A free-text story prompt (`story_diff`, first differing
character) already diverges from CPU under the old defaults for Llama (char 11) and Qwen2.5-1.5B (char 47); with the new defaults it
diverges at char 11 / 174 / 27-108 (Llama / Qwen 1.5B / Qwen 0.5B FA on). All outputs are coherent continuations; the divergence is the
usual near-tie flip from a different float add order (mmvq) and f32 softmax probabilities (stock rounds P to f16).

## Native FlashAttention kernel (`hw_subst_fattn_native_vs_reference`)

nmse against exact f64 attention 3.8e-14 to 3.9e-14. ms per launch (631 queries, 768 keys, causal + padding mask):

| shape | translated stock (CFI_PROFILE, serialised) | native |
|---|---|---|
| Qwen 0.5B, D 64, 14/2 heads | 5.9 | 4.1 |
| Llama 1B, D 64, 32/8 heads | 10.1 | 2.7 |
| Qwen 1.5B, D 128, 12/2 heads | n/a | 5.8 |

Variants tried and dropped: 2 keys per lane (more shared memory, lower occupancy: 4.3 -> 5.6 ms), f32 K/V in shared memory (packed f16
pairs halve it: Llama 5.9 -> 2.7 ms). Decode (fewer than 16 queries, Llama's GQA mma path) keeps the translated kernel: one tile per head
left the GPU idle (287 us native vs 176 us translated).
