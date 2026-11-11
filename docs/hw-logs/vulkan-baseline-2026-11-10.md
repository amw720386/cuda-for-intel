# Stock Ollama Vulkan and CPU baselines on the Arc B390, 2026-11-10

Reference numbers for comparing the CUDA-for-Intel decode path against the stock Vulkan backend under the same
request conditions as `scripts/bench-decode.ps1`, and the same Q8_0 model file. No code from this repo is involved.

## Hardware and software (measured)

| item | value |
|---|---|
| GPU | Intel Arc B390 integrated GPU, PCI 8086:B080 rev 04, Vulkan `PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` |
| driver | 32.0.101.9033 (2026-09-23), Vulkan API 1.4.356 |
| GPU memory | shared, one 36.19 GiB device-local heap; `VK_KHR_cooperative_matrix` supported; 48 KiB shared memory, 1024 invocations per workgroup |
| CPU, RAM | Intel Core Ultra X7 358H (16 cores, 16 threads), 63.5 GB |
| OS | Windows 11 Pro 10.0.26300 |
| Ollama | 0.40.14 stock (its own Vulkan and CPU backends) |
| model | Qwen2.5-0.5B-Instruct Q8_0 GGUF, SHA-256 `ca59ca7f13d0e15a...` (the file used in `ollama-stock-v0.12.6.md` and PR #18), imported with a one-line `FROM <gguf>` Modelfile |

The box was otherwise idle for these runs (no other GPU job).

## Method

`scripts/bench-decode-vulkan.py` sends the same request as `scripts/bench-decode.ps1`: raw prompt
`"<variant> " + "lorem" x 607 + "\nCount from 1 to 300 separated by commas: 1, 2, 3,"` (631 prompt tokens), 48 generated tokens,
`temperature 0, seed 0, top_k 1, repeat_penalty 1.0, num_ctx 2048, num_batch 512, num_gpu 99`. One 8-token warm-up on variant 2, then
5 measured requests on variants 3 to 7, which the KV prefix cache has never seen. Per FlashAttention setting, 3 fresh `ollama serve`
processes (env `OLLAMA_VULKAN=1 OLLAMA_IGPU_ENABLE=1 OLLAMA_NUM_PARALLEL=1 OLLAMA_FLASH_ATTENTION=0|1`), so n = 15 per row.

Placement was checked for every server: the log shows `offloaded 25/25 layers to GPU`, a `Vulkan0 model buffer` of 500.79 MiB, and
`flash_attn = disabled|enabled` as requested. `/api/ps` reports `size_vram == size` after each session. Raw per-run data: `vulkan-baseline-2026-11-10/`.

## Results (tok/s)

| backend | FA | prefill p50 | decode p50 | decode p95 | decode best | per-server decode p50 |
|---|---|---|---|---|---|---|
| Vulkan | off | 7188 | 157.4 | 158.5 | 159.0 | 156.3 / 158.2 / 157.6 |
| Vulkan | on | 8030 | 156.2 | 156.6 | 156.8 | 155.9 / 156.4 / 156.4 |

Regenerate the rows from the raw JSON with `python -I scripts/bench-decode-vulkan.py --pool "FA off=<dir>/fa0_s1.json,<dir>/fa0_s2.json,<dir>/fa0_s3.json" "FA on=<dir>/fa1_s1.json,..."` (`<dir>` = `docs/hw-logs/vulkan-baseline-2026-11-10`); every measured request in the JSON has 631 prompt tokens.

Output for variant 3 is the counting sequence `" 4, 5, 6, 7, 8, ..."` and is identical with FA off and on.

For context, PR #18 reports the CUDA path on the same box and model (best of 3, warm, v0.12.6 on the shim) at 7351 / 45.7 (FA off) and
8158 / 57.8 (FA on) prefill / decode. Prefill is already at Vulkan parity; decode is 2.7 to 3.4x behind Vulkan, so the remaining gap is
decode-specific (per-token work: about 22 ms on the CUDA path vs 6.4 ms on Vulkan).

### Q4_K_M, `scripts/benchmark-ollama.ps1` (different request shape, not comparable to the table above)

`qwen2.5:0.5b` (Q4_K_M, registry digest `a8b0c515...`), 800-character prompt with a unique first word, 48 tokens, 2 warm-ups + 9 samples,
3 repetitions each with a fresh model load:

| backend | prefill p50 | decode p50 | TTFT p50 |
|---|---|---|---|
| Vulkan | 7136 to 7175 | 196.9 to 199.2 | 54 ms |
| CPU (`num_gpu 0`) | 436 to 441 | 140.3 to 145.4 | 591 to 596 ms |

## Pitfalls found while measuring

- Ollama 0.40.14 drops integrated GPUs unless `OLLAMA_IGPU_ENABLE=1` is set (`dropping integrated GPU` in the log) and silently runs
  on the CPU; `/api/ps` then shows `size_vram 0`. Without the variable, every "Vulkan" number is a CPU number.
- Repeating an identical prompt reuses the KV prefix cache: prefill read about 11,000 tok/s on both CPU and GPU. Every measured prompt
  must be new.
- `benchmark-ollama.ps1` checks that exactly one model is resident; unload other models first (`keep_alive 0`) or it fails its
  placement check.
- Stock Ollama v0.12.6 for Windows ships no Vulkan backend (only CPU and CUDA), so the Vulkan reference has to come from a newer
  Ollama; the request shape and the model file are kept identical instead.
