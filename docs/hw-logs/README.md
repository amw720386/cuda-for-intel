# Hardware test logs (Arc B390, driver 32.0.101.9033, Windows)

Round 1 on the integration branch (all feature branches merged). Paths scrubbed.

- build.log: `cargo build --locked -j 2 --workspace`
- test-unit.log: `cargo test --locked --workspace` (129 lib tests pass)
- test-gpu.log: `cargo test --test gpu -- --ignored --test-threads=1` (round 1: 40/40; final tree: 79/79 incl. graph_capture, hw kernels, module globals, event timing)
- test-lib-ignored.log: ignored lib tests except `gated_forge_copy_queue_timestamps`; 24 pass,
  `ptx::tests::validate_opencl_spirv` fails (emitter now produces SPIR-V 1.3, validator targets opencl2.2 = 1.2)
- test-lib-ignored-hang.log: `gated_forge_copy_queue_timestamps` spins at full CPU without finishing; killed after ~8 min

Final tree (all feature branches, tier2/tier3 translator, copy-queue fix): 215 lib tests pass, GPU suite 79/79
(the 3-minute-plus `hw_stock_ggml_kernels_compile` is skipped in routine runs), and
`gated_forge_copy_queue_timestamps` now passes in 0.5 s (test-lib-copyqueue.log).
See ollama-stock-v0.12.6.md for the stock Ollama first-token run.

See vulkan-baseline-2026-11-10.md for the stock Ollama Vulkan and CPU reference numbers (same request shape and Q8_0 file as `scripts/bench-decode.ps1`).

- translate-precompile-2026-11-10.md: CPU-only PTX -> SPIR-V translation cost of stock ggml-cuda (Core Ultra 7 268V host),
  `cfi-precompile --translate-only` before/after the concurrent module prepare.
