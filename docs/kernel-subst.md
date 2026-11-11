# Kernel substitution

Opt-in layer that replaces recognised ggml-cuda kernels with hand-written Intel-native SPIR-V.

* Switch: `CUDA_FOR_INTEL_SUBST=none|auto|all|<list>` (default `auto`: mmvq at every ncols and type, quantize_q8_1, mmq for Q8_0, Q4_0, Q5_0, Q4_K, Q5_K, Q6_K with their stream-k fixups; `none` disables; `-mmq`, `-mmvq`, `-quantize_q8_1` or `-family:type` remove a family from auto/all, e.g. `auto,-mmq`). Measured on the Arc: mmvq 1.0-3.7x, mmq 5.3x at 4096x4096x512; 330 of 330 shadow checks of mmq/mmvq launches in Qwen2.5-0.5B prefill were bit-identical to the translated kernel; native mmq differs from stock stream-k only in float add order, so greedy output can differ in near-ties (perplexity within +-0.13% over 5 chunks).
  `CUDA_FOR_INTEL_SUBST_NWARPS=<pow2>` (mmvq experiment, changes only the cross-warp add order), `CUDA_FOR_INTEL_SUBST_PROBE=0` skips the first-use check.
* Recognition (`src/subst/mod.rs`): Itanium name (template args included) plus exact parameter layout. Any mismatch keeps the PTX and logs why.
  Launch shape must also match (mmvq block `32 x stock nwarps`, quantize block 256).
* First use (`src/cuda/subst_hook.rs`): build the module, run a small case on the device against a host reference
  (`src/subst/probe.rs`), then use it; any failure (build, probe, launch) falls back to the translated kernel. Graph replay goes through the same hook.
* Native kernels: `src/subst/{mmvq,quantize}.rs` on the builder in `spv.rs`. mmvq keeps ggml's thread decomposition, accumulation and
  reduction order and the fma structure nvcc emitted, so results are bit-identical to the translated stock PTX (verified on the interpreter,
  `CFI_STOCK_PTX_DIR=<dir with extracted blobN.ptx> cargo test --lib subst`). The only deliberate difference is the opt-in nwarps override.
* `mul_mat_q` Q8_0 (`mmq`, `src/subst/mmq.rs`): tiled SDot kernel (64x32 tile, 16x16 threads), own grid, non-MoE only (`ids_dst`/`expert_bounds` null, else PTX).
  Sequential-k accumulation `fma(dA*(float)sumi, dB, sum)`: bit-identical to the translated stock MMA kernel on non-split tiles (interpreter test);
  stream-k-split tiles differ in rounding only. The paired `mul_mat_q_stream_k_fixup` is skipped when it follows a native launch (matched by dst/tmp pointers).
  K-quants and legacy types (`mmq.rs` is parameterised by type): Q4_0, Q5_0 (float scale per 32), Q4_K, Q5_K (half2 `(d*sc, -dmin*m)` per 32, y in DS4 layout),
  Q6_K (int8 scale per 16, tmp accumulated per 128 values), each following the stock MMA float structure. Single-tile results are bit-identical to the translated stock kernel on the
  interpreter (`translated_stock_mmq_all_types_match_native_on_the_interpreter`) and on the Arc (`hw_subst_mmq_types`: 0 bitwise differences over Llama-3.2-1B/Qwen2.5-1.5B shapes).
  Measured mmq speedup at 4096x4096x512 (one CTA per tile): Q8_0 2.97x, Q4_0 3.21x, Q5_0 2.17x, Q4_K 4.82x, Q5_K 4.80x, Q6_K 3.58x; all in `auto` (`auto,-mmq:q4_k` etc. to drop one).
  Q4_1/Q5_1 are not covered. Not done: DPAS (int8 XMX) variant.
* Hardware: `tests/gpu/hw_subst.rs` (see its header for the command).
* Debug: `CUDA_FOR_INTEL_SUBST_CHECK=<n>` reruns the translated kernel after the first n native mmvq/mmq launches of each variant into a scratch copy, logs `subst-check:` lines (nmse, shapes) to cuda-for-intel-export.log, and on mismatch restores the translated result and falls back to PTX for that variant. It runs only for stream launches: kernels recorded into a CUDA graph's command list (`CUDA_FOR_INTEL_GRAPH_RECORD=1`) are substituted the same way but not shadow-checked.

## K-quant mmq end to end (stock Ollama v0.12.6, Arc, flash attention off, 631-token prompt, greedy, 48 tokens)

| model | SUBST | prefill tok/s | decode tok/s |
|---|---|---|---|
| Llama-3.2-1B Q4_K_M | none | 805 | 26.3 |
| Llama-3.2-1B Q4_K_M | auto | 4843 (6.0x) | 32.2 (+22%) |
| Qwen2.5-1.5B Q4_K_M | none | 643 | 21.2 |
| Qwen2.5-1.5B Q4_K_M | auto | 3736 (5.8x) | 24.9 (+18%) |

Shadow check (`CUDA_FOR_INTEL_SUBST_CHECK=30`, Llama-3.2-1B, mmq Q4_K and Q6_K plus mmvq): 120 of 120 launches bit-identical to the translated kernels.
Both modes produce coherent output; greedy text diverges from `none` after about 20 tokens at near-ties (e.g. "fashion, cuisine," versus "fashion,"), the stream-k
add-order effect described above. Reproduce with `scripts/bench-subst-ollama.ps1`; kernel-level log in `docs/hw-logs/hw-subst-mmq-types.log`.
