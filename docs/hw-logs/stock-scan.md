# Stock ggml-cuda.dll translation scan (Ollama v0.12.6, cuda_v13)

`cargo test --release --lib scan_stock_dll_translation -- --ignored --nocapture` with
CFI_STOCK_GGML_DLL set translates every PTX entry of every fatbin blob (117 blobs, 3744 entries).

| state | entries failing |
| --- | --- |
| integration at 1a42040 (before this round) | 670 |
| after the generalised early-exit barrier rule | 145 |
| after pointer/integer register coercion | 87 |

Remaining 87: 66 `flash_attn_ext_f16` (barrier divergence in the mma path), 20 `flash_attn_ext_vec`
(register read before any assignment on some path), 1 `add_id_kernel` (declaration mismatch).
All are flash-attention kernels except add_id; routine runs with OLLAMA_FLASH_ATTENTION=0 do not need them.

Qwen2.5-0.5B Q8_0 greedy, GPU vs CPU: "The capital of France is" 32 tokens identical.
A longer story prompt diverged at the first token (possible near-tie or a numerical difference in the
batched MMQ path); under investigation.
