# Kernel substitution

Opt-in layer that replaces recognised ggml-cuda kernels with hand-written Intel-native SPIR-V.

* Switch: `CUDA_FOR_INTEL_SUBST=none|all|<list>` (default `none`). List items: `mmvq`, `mmvq:q8_0|q4_0|q4_k|q6_k`, `quantize_q8_1`.
  `CUDA_FOR_INTEL_SUBST_NWARPS=<pow2>` (mmvq experiment, changes only the cross-warp add order), `CUDA_FOR_INTEL_SUBST_PROBE=0` skips the first-use check.
* Recognition (`src/subst/mod.rs`): Itanium name (template args included) plus exact parameter layout. Any mismatch keeps the PTX and logs why.
  Launch shape must also match (mmvq block `32 x stock nwarps`, quantize block 256).
* First use (`src/cuda/subst_hook.rs`): build the module, run a small case on the device against a host reference
  (`src/subst/probe.rs`), then use it; any failure (build, probe, launch) falls back to the translated kernel. Graph replay goes through the same hook.
* Native kernels: `src/subst/{mmvq,quantize}.rs` on the builder in `spv.rs`. mmvq keeps ggml's thread decomposition, accumulation and
  reduction order and the fma structure nvcc emitted, so results are bit-identical to the translated stock PTX (verified on the interpreter,
  `CFI_STOCK_PTX_DIR=<dir with extracted blobN.ptx> cargo test --lib subst`). The only deliberate difference is the opt-in nwarps override.
* Not done: `mul_mat_q` (Q8_0/Q4_K, DPAS). The stock PTX uses the Turing MMA path with stream-k + a separate fixup kernel; a native version must
  consume `block_q8_1_mmq` (from `quantize_mmq_q8_1`), compute its own grid, skip the paired `mul_mat_q_stream_k_fixup`, and needs the
  interpreter to model the Intel DPAS op. Float order would differ from stock only on stream-k-split tiles (sequential k per output otherwise).
* Hardware: `tests/gpu/hw_subst.rs` (see its header for the command).
