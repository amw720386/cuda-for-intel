# Kernel substitution

Opt-in layer that replaces recognised ggml-cuda kernels with hand-written Intel-native SPIR-V.

* Switch: `CUDA_FOR_INTEL_SUBST=none|auto|all|<list>` (default `auto`: mmvq at every ncols and type, quantize_q8_1, mmq Q8_0 with its stream-k fixup; `none` disables; `-mmq`, `-mmvq`, `-quantize_q8_1` or `-family:type` remove a family from auto/all, e.g. `auto,-mmq`). Measured on the Arc: mmvq 1.0-3.7x, mmq 5.3x at 4096x4096x512; 330 of 330 shadow checks of mmq/mmvq launches in Qwen2.5-0.5B prefill were bit-identical to the translated kernel; native mmq differs from stock stream-k only in float add order, so greedy output can differ in near-ties (perplexity within +-0.13% over 5 chunks).
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
  Not done: Q4_K/Q6_K mmq and DPAS (int8 XMX) variant.
* Hardware: `tests/gpu/hw_subst.rs` (see its header for the command).
