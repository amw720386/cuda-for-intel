# Stock Ollama v0.12.6 model matrix on the Arc B390

Raw prompts (no chat template), temperature 0, f16 KV cache, flash attention off, `num_batch` 32, logging off,
shared machine (other jobs on the GPU and CPU). Speeds are noisy and include first-use kernel compiles.
"Diverges" compares greedy output with the CPU backend (8 prompts, 64 tokens each): K-quant models diverge
earlier because the CPU quantises activations differently (q8_K 256-blocks vs q8_1 32-blocks), so divergence is
expected there and is not a correctness verdict; for those, coherence is reported instead.

| Model | Quant | Check | Result | GPU decode tok/s |
| --- | --- | --- | --- | --- |
| Qwen2.5-0.5B-Instruct | Q8_0 | 12 prompts x 24 tokens vs CPU | 10 of 12 identical | 30 to 46 (CPU 119) |
| Llama-3.2-1B-Instruct | Q8_0 | 8 prompts x 64 tokens vs CPU | 5 of 8 diverge, first at chars 42 to 299 | 22.1 (CPU 36.9) |
| Llama-3.2-1B-Instruct | Q4_K_M | same | 7 of 8 diverge, first at chars 1 to 319 | 38.2 (CPU 67.7) |
| Qwen2.5-1.5B-Instruct | Q4_K_M | same | 7 of 8 diverge, first at chars 55 to 182 | 27.3 to 30.9 (CPU 53.6) |
| gemma-3-1b-it | Q4_K_M | coherent story text | coherent | about 12 |
| Qwen3-0.6B | Q8_0 | 2 prompts x 48 tokens | on topic, repetitive on a short Q&A prompt | 20 to 40 |
| Qwen3-1.7B | Q8_0 | same | coherent | 15 to 16 |
| Phi-3.5-mini-instruct | Q4_K_M | same | coherent | 18.7 to 18.8 |
| Llama-3.2-3B-Instruct | Q4_K_M | same | coherent | 18.7 to 19.9 |
| Qwen2.5-3B-Instruct | Q4_K_M | same | coherent | 17.3 to 20.1 |
| Llama-3.1-8B-Instruct | Q4_K_M | same | coherent | 8.8 |
| nomic-embed-text-v1.5 | Q8_0 | embedding vs CPU | cosine 1.0 (768 dims) | n/a |

Not measured here: perplexity for these models (see `ggml-test-backend-ops.md` for llama.cpp numbers), Vulkan baselines,
flash attention on, KV cache quantisation.
