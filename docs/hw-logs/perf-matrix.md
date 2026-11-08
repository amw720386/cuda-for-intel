# Qwen2.5-0.5B Q8_0 on the stock Ollama v0.12.6 path: speed measurements

Arc B390, shared machine (other jobs ran on the GPU and CPU at the same time), raw prompts, temperature 0,
`num_batch` 32, `num_ctx` 512, logging and tracing env vars unset. 12 prompts, 24 generated tokens each,
f16 KV cache, flash attention off. tok/s is total tokens over total time per phase.

| Run | Decode tok/s | Prefill tok/s | Notes |
| --- | --- | --- | --- |
| CPU backend (`num_gpu 0`) | 119.3 | 270.4 | |
| GPU, API trace logging ON (older build) | 5.1 | 0.9 | logging reopened its file on every launch; do not use |
| GPU, logging off, cold kernel cache | 46.4 | 8.0 | prefill includes first-use kernel compiles |
| GPU, logging off, warm kernel cache | 29.9 | 19.2 | noisy: other GPU jobs shared the machine |

The GPU is not currently faster than the CPU backend for this model. Output agreement with the CPU for these
runs: see `stock-scan.md` (10 of 12 prompts identical over 24 tokens).
