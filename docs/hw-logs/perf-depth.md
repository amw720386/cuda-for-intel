# perf-depth: decode and launch-overhead work (branch perf-depth)

Status: Mac-side only so far. The Arc GPU is reserved for the release gate, so no profile
or before/after numbers exist yet; nothing below claims a measured speedup.

## Done (verified without the GPU)

* `CFI_OPT_LD16` load-pair merge (`src/ptx/normalize.rs`, `merge_u16_load_pairs`):
  review found one real soundness gap. The second 16-bit load is hoisted up to the first,
  and the straight-line check allowed stores, atomics, reductions and fences between them,
  so a write to the same dword would have been missed. The merge now refuses such pairs
  (test `leaves_pairs_alone_when_a_store_sits_between_the_loads`).
* New SPIR-V-simulator test `merged_loads_equal_the_original_on_the_simulator`: the original
  and rewritten kernels return identical bytes at every 2-byte alignment, plain and `.nc`
  loads, both operand orders. The shift/funnel arithmetic is correct.
* The earlier "batched-GEMM LaunchFailed" was a masked test panic (see a39f2b2), so the
  merge itself has no known failing case. It stays opt-in until the hardware matrix
  (batched GEMM test, 3 models, text equality over 48 greedy tokens) is run with
  `CFI_OPT_LD16=1`; flipping the default is a one-line change in `load16_merge_enabled`.
* `TRANSLATOR_VERSION` bumped to 4 (xcache already hashes the LD16 switch).

## Not yet done (needs the GPU)

1. Per-kernel GPU time on the 3 models (Level Zero timestamps or shim timing), launch and sync overhead.
2. Native kernels for the remaining decode ops, selected by the profile.
3. Hardware matrix: 631-token prefill and 48-token decode, SUBST auto, FA on/off, text equality.
