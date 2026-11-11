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

## Decode-fast mmvq (`CUDA_FOR_INTEL_MMVQ_FAST`, default on)

* Why the stock-order kernel was slow: one 128-thread block per row with a shared-memory reduction, and ggml's per-thread split of
  each quantized block (4 to 32 lanes per block, 2 to 8 bytes per lane, weights read as 16-bit halves). On the B390 that reaches
  31-53 GB/s at decode shapes; the device read ceiling is 114 GB/s (`hw_perf_probe`: random data, 512 MiB; a memset-filled buffer
  reads at 540 GB/s because it is served compressed, so never benchmark on memset data).
* Variant (`src/subst/mmvq.rs`, `build_fast`): ncols_dst 1 only, one 32-lane subgroup per row, 4 rows per workgroup, no shared
  memory or barrier; `wide` (Q8_0, Q4_K, Q6_K) gives each lane one 32-value group read as whole dwords from the 2-byte-aligned block
  (aligned loads plus a funnel shift, never past the block); 2 accumulators per lane. The driver appends `nrows` to the stock 23
  words. Integer sums are exact; only the float add order differs from stock (nmse vs the stock order < 1e-10 on the interpreter; the
  host model `refs::mmvq_fast` is bit-exact).
* Kernel numbers (`hw_subst_mmvq_decode_shapes`, weights rotated over enough copies to defeat the cache): output matrices 106-109 GB/s
  (stock-order native 36-53), mid-size matrices 55-75 GB/s, small ones bounded by launch cost (4-25 us). Tables and end-to-end runs:
  `docs/hw-logs/perf-mmvq-2026-11-09/`.
* Default on: an unset or empty `CUDA_FOR_INTEL_MMVQ_FAST` (or `1`/`on`/`true`/`yes`/`auto`) selects the per-type default (4 rows, 1 warp,
  unroll 2, `wide` for Q8_0 / Q4_K / Q6_K). `0`/`off`/`false`/`no`/`none` restores the stock-order kernel; `<rows>x<warps>x<unroll>[w]` picks
  another shape; an unparsable or out-of-range shape also keeps the stock-order kernel and logs
  `subst: CUDA_FOR_INTEL_MMVQ_FAST=... not understood` once to the export log. Only `ncols_dst` 1 is affected, and only where
  `CUDA_FOR_INTEL_SUBST` selects the native mmvq (`auto` does). `CUDA_FOR_INTEL_SUBST_NWARPS` applies to the stock-order kernel only,
  so set `CUDA_FOR_INTEL_MMVQ_FAST=0` with it to vary decode warps.
* Caches: the per-context module key (`mmvq:t<ty>:c1:fast<r>x<w>x<u>[w]`) and the entry name (`cfi_mmvq_fast_t<ty>_r_w_u[_wide]`)
  carry every shape knob, so fast, wide and stock-order variants never share a module; the driver-side native compile cache is keyed
  by the SPIR-V words. `cfi-precompile` builds the same plan as the launch path (`subst::decide`), so it warms the fast variant by
  default and the stock-order one under `CUDA_FOR_INTEL_MMVQ_FAST=0`. The PTX translation cache (`TRANSLATOR_VERSION`) is unaffected:
  native kernels are generated, not translated. The kernels use no dynamic shared memory, so the dynamic-shared-window
  specialisation (`CUDA_FOR_INTEL_DYN_SHARED_SPECIALIZE`) does not apply to them.
* B390 end to end with the default (Qwen2.5-0.5B-Instruct Q8_0, pooled decode p50, 3 servers x 5 runs, `scripts/bench-decode.ps1`, owner run on PR #18): FA off 32.3 -> 44.2 tok/s,
  FA on 37.7 -> 56.5 tok/s, the 48-token counting-prompt greedy text equal to CPU (the free-text story prompt diverges from CPU at near-tie logits, as it
  already did for some models before; `docs/hw-logs/perf-mmvq-2026-11-09/README.md`, "Text"); log:
  `docs/hw-logs/perf-mmvq-2026-11-09/owner-validation-2026-11-10.md`.

## Native FlashAttention (`fattn`, default on)

* In `auto` (switch off with `CUDA_FOR_INTEL_SUBST=auto,-fattn`), `flash_attn_ext_f16<D, D, ..., false, false>` (D 64 / 128, at
  least 16 queries) runs a SIMT tile kernel (`src/subst/fattn.rs`): 16 queries per workgroup, K/V chunks in shared memory as packed f16,
  online softmax per query, chunks whose mask is -inf for the whole tile skipped, f32 arithmetic. The paired `flash_attn_stream_k_fixup`
  is skipped. Launches with sinks, ALiBi, softcap, fewer than 16 queries (decode) or unaligned K/V keep the translated kernel.
* Accuracy: nmse 4e-14 against exact f64 attention on the device (`hw_subst_fattn_native_vs_reference`); the stock kernel rounds P to
  f16, so native and translated differ at about 1e-3 relative. End to end, FA-on prefill +81-88 % on the three test models.
* `CUDA_FOR_INTEL_FATTN_NW=8` (8 subgroups, 32 queries per workgroup) is an experiment switch.
