# Hardware validation runbook (`hw-validate`)

The branch `hw-validate` is `main` plus the five held, hardware-untested branches, merged in this order:
`translator-hardening`, `fix-divergence`, `api-coverage`, `xmx-speed` (includes `xmx-gqa-fix`), `cudnn-final`.
`TRANSLATOR_VERSION` is 5 (one bump covers every translator-output change); `cfi_build_info()` reports `translator=5` and
the unit test `build_info_matches_translator_version` keeps the two in step.

`scripts/hwvalidate/run_all.ps1` validates all of it in ONE GPU-queue slot and leaves a machine-readable `results.json` and a
human `summary.md`. The test result is the only thing that decides what merges to `main`.

## Preconditions

- The Arc box is up and the ssh alias for it is in `~/.ssh/config` on the Mac. The alias is passed as `CFI_BOX` and is never
  written into the repo.
- On the box (all under the user's `cfi-work` directory): `gpu-queue.ps1`, a Rust toolchain on PATH, the staged stock Ollama
  tree, three GGUF models, the hybrid dir (`nvcudart_hybrid64.dll`), the stock `ggml-cuda.dll`. Optional: an f16 GGUF, a PTX
  corpus subset, the torch venv python.
- Copy `scripts/hwvalidate/config.example.ps1` to `cfi-work\hwval\config.ps1` on the box once and edit the paths. Everything
  else is derived from the work dir.
- At least 30 GB free on the system drive (stage 0 checks). Nobody else holds the GPU lock (the run waits its turn in the
  queue otherwise).
- The working tree is committed and pushed: `sync.sh` ships `git archive HEAD`, so the box always runs a known revision.

## Second box (Lunar Lake / Arc 140V, Xe2, older driver)

A second Windows box is meant to be the Xe2 and older-driver target. It is set up like the first: ssh alias in `~/.ssh/config`,
everything under the user's `cfi-work`, the same `gpu-queue.ps1`, and the same toolchain (Rust stable MSVC 1.99.1,
Python 3.12.10, Git, LLVM 20.1.8 for `NVRTC_SHIM_CLANG`), plus VS Build Tools 2022 instead of VS 18. Stock Ollama v0.12.6
is extracted portably under `cfi-work\stock\ollama-v0.12.6`. Differences from the first box:
- The execution policy is Restricted, so run scripts with `powershell -NoProfile -ExecutionPolicy Bypass -File ...`. Child
  `powershell` processes inherit the process-scope policy, including the gpu-queue calls inside the repo scripts.
- The source tree is at `cfi-work\main\src` and the target dir is `cfi-work\main\target`. `build.ps1` and `gpuhw.ps1`
  (in `cfi-work`) build the workspace and run `gpu-*.exe --ignored --test-threads=1` in one queue slot.
- **As of 2026-11-10 its GPU driver does not load** (code 39, kernel code integrity rejects `igdkmdn64.sys` 32.0.101.5564),
  so `zeInit` finds no driver and every GPU test fails at `cuInit` = 100. Run `scripts\cfi-doctor.ps1` first: the
  `gpu-device-status` row must say `working`. See `hw-logs/second-box-bringup-2026-11-10.md`.

## Commands

```
# Mac: one transfer (scp + one extract command on one ssh connection), then start the run in the foreground
CFI_BOX=<alias> scripts/hwvalidate/sync.sh --run

# a dropped link does not lose work: rerun the same line, finished stages are skipped
CFI_BOX=<alias> scripts/hwvalidate/sync.sh --run

# Mac: fetch results.json, summary.md, stage*.json and small logs with one scp
CFI_BOX=<alias> scripts/hwvalidate/collect.sh
```

On the box directly:

```
powershell -NoProfile -File scripts\hwvalidate\run_all.ps1                 # everything
powershell -NoProfile -File scripts\hwvalidate\run_all.ps1 -Stages 3,6     # selected stages
powershell -NoProfile -File scripts\hwvalidate\run_all.ps1 -Force          # rerun finished stages
powershell -NoProfile -File scripts\hwvalidate\run_all.ps1 -RerunFailed    # rerun only non-green stages
```

Results land in `cfi-work\hwval\results\` (`stage<N>.json` is also the resume marker, raw logs in `logs\`, `collect.tgz` is what
`collect.sh` pulls). Strings are scrubbed of the user name, host name, profile path and IPv4 addresses before they are written.

Stage 0 (build, parse checks) runs before the queue slot is taken because it needs no GPU. Stages 1-9 run inside the slot,
with a heartbeat job refreshing the lock and a `finally` that always releases it. No stage kills a process: every child has a
timeout and the hardware tests carry `CFI_HW_WATCHDOG_S=600` so a hang aborts the test process itself. If a child is still
alive 120 s after its timeout the run is marked poisoned, the remaining stages are skipped and the lock is released; the
operator then deals with the stuck process by hand. The one exception to "nothing is terminated" is the Ollama server that
stages 2, 3, 6 and 8 start themselves: Ollama has no shutdown command, so after asking it to unload the model the script ends
the process tree it started (.NET `Process.Kill()` on its own children, never `taskkill`/`Stop-Process`).

## Stages, expected runtime, what a failure means

Runtimes are estimates (first hardware run, shader caches cold); the per-stage budget in `run_all.ps1` is what actually cuts
a stage off.

| # | Stage | Est. | Content | A failure means / blame |
|---|---|---|---|---|
| 0 | environment + build | 20-60 min | disk, stale processes, gpu-queue status, parse-check of every `scripts/**/*.ps1`, `cargo build --release --workspace`, `cargo test --no-run --test gpu`, DLL hashes | compile or script error: the merge itself (a conflict resolution) or the branch that owns the file; nothing else runs |
| 1 | baseline gate | 30-60 min | `hw_subst*`, `hw_nvrtc`, `hw_fa_vec`, rest of the GPU suite, with `CUDA_FOR_INTEL_FAST_DIV=1` and `CFI_XMX_BLK=0` (main behaviour) | `translator-hardening` or the merge: the new features are switched off here, so a red baseline cannot be fix-divergence or xmx-speed |
| 2 | FA Ollama text | 15-30 min | 3 models, CPU vs shim FA off vs FA on, short and 600-token prompt, texts stored | `review` = texts differ from CPU (read them: coherent text is fp drift, gibberish is a regression); `fail` = request error. Suspect `fix-divergence`, `xmx-speed`, `translator-hardening` |
| 3 | divergence | 20-60 min | `hw_div_sqrt_rn_f32_correctly_rounded` must PASS by default (with `CUDA_FOR_INTEL_FAST_DIV=1` it is informational: the B390 already rounds translated `div.rn` correctly); `hw_subst_quantize_fuzz`; auto-vs-none greedy scan and logits via `scripts/divergence/run_on_box.ps1 -NoQueue` | `fix-divergence`. Default test failing = the driver ignores the correct-rounding flag; the flag's effect shows in the auto-vs-none scan (12 of 20 prompts diverge with `FAST_DIV=1`, 7 of 20 without, 64 tokens) |
| 4 | translator edge | 10-25 min | `hw_translator_edge`, translator unit tests on Windows, PTX corpus subset (`corpus_scan`, every 50th entry through spirv-val) | `translator-hardening`; no runtime switch exists, bisect by reverting the branch |
| 5 | api coverage | 10-25 min | `api_coverage::*`, `hw_sgemv_sdot_numerics`, unit tests of the shim crates | `api-coverage` (cudart/driver additions, nvml/cufft/cusparse/cusolver shims, cublas gemv/dot) |
| 6 | XMX | 30-60 min | `hw_xmx_blk`, `hw_xmx_shapes` with `CFI_XMX_BENCH_MODES=0,1,2`, `hw_xmx_probe` with `CFI_XMX_TRACE=1`, remaining `hw_xmx_gemm`, Ollama f16 A/B (`CFI_XMX=0`, default, `CFI_XMX_BLK=0`, trace) | `xmx-speed` (or `xmx-gqa-fix` inside it). Isolate with `CFI_XMX_BLK=0` (block loads off) then `CFI_XMX=0` (XMX off) |
| 7 | cuDNN | 20-90 min | `hw_cudnn`, then `scripts/cudnn/gate_final.ps1 -NoQueue -SkipSuite` (callers, resnet18, conv bench, no-workspace sweep) against a v8-flavoured cudnn DLL | `cudnn-final`. Skipped when no torch python is configured |
| 8 | perf matrix | 60-150 min | per model: CPU reference, then SUBST auto x FA off/on x `CFI_OPT_LD16` 0/1; 631-token prefill, 48-token decode, text equality vs CPU | a new `LaunchFailed`/error line = LD16 (`perf-depth`) or the batched-GEMM path; slow = compare against the same cell on `main` |
| 9 | installer | 5-20 min | `cfi.ps1 install -DryRun`, real install into a scratch dir, `doctor`, `selftest`, grep of all output for user name / host / IPv4, `uninstall` round trip; then `torch_stage.ps1` (the `hw-prep` PyTorch ladder L0-L7, `scripts/torch/progressive.py`, per-level timeouts, no queue/kill of its own) when a torch python is configured | `main` packaging scripts (not one of the five branches) unless the leak grep trips on a branch-added message; a torch ladder failure names the failing level (L0 import, L2 matmul = cublas shim, L4 conv = `cudnn-final`, L6 generation = FA/XMX/divergence) |

Status values: `pass`, `fail`, `timeout`, `review` (needs a human read, not automatically red), `info`, `skip`.
`summary.md` lists blame and the isolation switch next to every non-green item.

## Isolation matrix (no rebuild needed)

| Feature | Branch | Disable / switch | Where it shows |
|---|---|---|---|
| correctly rounded `div.rn.f32` / `sqrt.rn.f32` | fix-divergence | `CUDA_FOR_INTEL_FAST_DIV=1` | stage 3, perf cells |
| XMX 2D block loads | xmx-speed | `CFI_XMX_BLK=0` | stage 6 |
| XMX GEMM entirely | xmx-speed / earlier | `CFI_XMX=0` (`CFI_XMX_BENCH_MODES` picks modes for the shape bench, `CFI_XMX_TRACE=1` logs selection) | stages 6, 8 |
| load16 merge | perf-depth (on main, opt-in) | `CFI_OPT_LD16=0/1` | stage 8 |
| flash attention | existing | `OLLAMA_FLASH_ATTENTION=0/1` | stages 2, 8 |
| kernel substitution | existing | `CUDA_FOR_INTEL_SUBST=none` | stage 3 |
| cuDNN shim | cudnn-final | no env switch: leave `cudnn64_*.dll` out of the run directory | stage 7 |
| translator hardening, api-coverage shims | translator-hardening, api-coverage | no runtime switch; revert the merge and rebuild (the `TRANSLATOR_VERSION` bump already invalidates caches) | stages 4, 5 |

Stage 1 runs with `CUDA_FOR_INTEL_FAST_DIV=1 CFI_XMX_BLK=0`, i.e. the main behaviour for the two switchable features, so it
doubles as a regression check of the merge. To bisect a failure of a switchable feature, rerun the failing item by hand with the
switch set; every `run_all.ps1` item names its test, and the raw output is in `logs\<item>.out/.err`.

## Rollback: which branch to drop

1. Read `summary.md`; the blame line names the branch.
2. Recreate the integration without that branch: `git checkout -b hw-validate-2 origin/main`, merge the remaining branches in the
   same order, set `TRANSLATOR_VERSION` and `translator=` in `src/cuda/xcache.rs` to 5, `cargo test --lib`, rerun with
   `-Stages` for the affected stages and stage 1 (`-Force`).
3. Order of dependence: `translator-hardening` and `fix-divergence` both touch the translator output (one version bump); dropping
   either keeps 5. `xmx-speed` and `api-coverage` are independent of each other. `cudnn-final` shares `Cargo.toml`/`Cargo.lock`
   members with `api-coverage`: if only one is dropped, keep the other crates in the workspace member list.
4. Never push to `main` from this branch; `hw-validate` is a test vehicle. After a clean run, merge the surviving branches to
   `main` individually, each with its own passing stage as the evidence.

## Mac-side checks that were run before handing over

`cargo test --lib` (379 passed, 16 ignored) and `cargo check --workspace --tests --target x86_64-pc-windows-gnu` are clean.
`bash -n` passes on `sync.sh` and `collect.sh`. PowerShell is not installed on the Mac (installing it needs sudo), so the
`.ps1` files were only checked for balanced braces and quotes; stage 0 of the first box run parse-checks every script with the
real PowerShell parser and reports any that does not parse (if `run_all.ps1` itself does not parse, PowerShell says so immediately).
