# Owner validation of PR #18 on the Arc B390 (2026-11-10)

Copied from the owner's review comment on PR #18
(https://github.com/amw720386/cuda-for-intel/pull/18#issuecomment-6781931809) so the README figures trace to a committed log.

Setup: Arc B390 (8086:B080, driver 32.0.101.9033, Core Ultra X7 358H), stock Ollama v0.12.6 on the shim, Qwen2.5-0.5B-Instruct
Q8_0 GGUF `ca59ca7f…`, `scripts/bench-decode.ps1` from PR #18 (4a82153), 5 measured runs per server, 624-token prompt
(`-NoCpu` skips the calibration to 631), 48 greedy tokens. Every cell's log shows `library=CUDA`, 25/25 layers on CUDA0 and
`flash_attn` as requested; the 48 tokens of the counting prompt equal the CPU backend's text in every cell; `nvcuda.dll` SHA-256
checked before each run. With the variable unset, PR #18 logged the stock-order key `native mmvq:t8:c1:w0`; with `=1`,
`native mmvq:t8:c1:fast4x1x2w` (the key the default selects since `perf/mmvq-fast-default`).

| build | FA | `CUDA_FOR_INTEL_MMVQ_FAST` | servers x runs | prefill p50 | decode p50 | decode p5-p95 | per-server decode p50 |
|---|---|---|---|---|---|---|---|
| main 11b70b5 | off | unset | 1 x 5 | 6724 | 31.1 | 30.2-31.9 | 31.1 |
| PR #18 4a82153 | off | unset | 3 x 5 | 6802 | 32.3 | 30.9-39.4 | 32.3 / 39.3 / 31.1 |
| PR #18 4a82153 | off | `1` | 3 x 5 | 6878 | 44.2 | 41.8-48.5 | 46.9 / 44.2 / 43.8 |
| main 11b70b5 | on | unset | 1 x 5 | 4437 | 37.9 | 37.7-40.1 | 37.9 |
| PR #18 4a82153 | on | unset | 3 x 5 | 7846 | 37.7 | 36.5-39.0 | 39.0 / 37.6 / 37.9 |
| PR #18 4a82153 | on | `1` | 3 x 5 | 8021 | 56.5 | 50.3-59.0 | 56.1 / 56.5 / 56.8 |

Notes from the same run:
* Decode on this iGPU is bimodal per server (about 31 or 39 tok/s FA off), sometimes switching within one server; the `=1`
  cells sit above every unset run regardless.
* No `CUDA error`, `device lost` or `LaunchFailed` lines and no subst rejections in any server log.
* The counting prompt is meaningful for about 30 tokens: past "18" the 0.5B model degenerates on CPU and GPU alike, so
  long-output comparisons beyond 48 tokens diverge at near-tie logits for both mmvq kernels. The free-text story prompt is not
  covered by this table; it diverges from CPU at near-ties (`README.md` in this directory, "Text").
* Stock Vulkan, same box, model and request shape (#19): decode p50 157.4 (FA off) / 156.2 (FA on), prefill 7188 / 8030.
