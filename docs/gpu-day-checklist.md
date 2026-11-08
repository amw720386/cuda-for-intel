# GPU-day checklist

One page per known gap, to run the moment the Arc test box is back. Nothing here needs new code.
Conventions: every GPU job holds the shared GPU queue slot (`gpu-queue.ps1`, see `scripts/torch/run_progressive.ps1` for the
acquire / heartbeat / release-in-finally pattern); results go under `docs/hw-logs/` with paths scrubbed (no hosts, IPs, user paths);
commit and push after every result. `<Work>` is the queue directory, `<Out>` a results directory, `<Py>` the cu118 venv python.
The full validation driver is `scripts/hwvalidate/run_all.ps1 (on branch hw-validate)`; use it instead of the individual commands
below when you want everything in one pass, and use this page to interpret a single failing item.

Build once (release, 2 jobs): `cargo build --release --locked -j 2 -p cuda-for-intel -p cublas_shim`.
Set `CUDA_FOR_INTEL_HYBRID_DIR` to the directory with `nvcudart_hybrid64.dll` (not bundled).

## 0. Order and budget

| Order | Item | Minutes |
| --- | --- | --- |
| 1 | 7 Perf profiling (cheap, informs the rest) | 15 |
| 2 | 4 SASS diagnostic | 5 |
| 3 | 1 Qwen-0.5B divergence | 40 |
| 4 | 2 XMX speed | 30 |
| 5 | 3 cuDNN default-on | 30 |
| 6 | 5 API coverage | 25 |
| 7 | 6 Translator edge | 40 |
| 8 | 8 LD16 default | 45 |

PyTorch ladder (items 1, 2, 3, 7, 8 reuse it): `scripts/torch/run_progressive.ps1`, about 15 to 25 min cold (first use of large ATen
modules translates for minutes; the translation cache makes the second run about 16 s).

## 1. Qwen-0.5B divergence (SUBST auto vs none, and shim vs CPU)

- Background: `docs/qwen05b-divergence.md`. Open item: the NWARPS control (`CUDA_FOR_INTEL_SUBST_NWARPS` 1/2/8, variants `mmvq_nw*`) and
  a native `quantize_q8_1` vs translated bit comparison.
- Commands (box):
  `scripts\divergence\run_on_box.ps1 -Work <Work>\div -OllamaRun <stock ollama tree> -Model <qwen2.5-0.5b-q8_0.gguf> -Hybrid <hybrid dir> -VariantList auto,none,mmvq,mmq,quantize_q8_1,mmvq_nw1,mmvq_nw2,mmvq_nw8`
  then on the Mac `python3 scripts/divergence/analyze_divergence.py --results <results> --gguf <model.gguf>`.
  PyTorch side: `CFI_HF_MODEL=<local Qwen2.5-0.5B dir>` with `scripts/torch/run_progressive.ps1 -Levels L6 -Model <dir>`.
- Pass: auto and none each within the CPU-q8 noise floor (RMS dlogit vs fp32 reference <= 0.15, top-1 flips only at reference margins
  < 0.05); NWARPS variants no worse than none; torch L6 either token-identical to CPU for 32 tokens or the first divergent position
  coincides with a CPU top-2 margin below 0.05 (it prints `first_divergent_position` and the logits check).
- Output: `<Work>\div\results\*`, `<Out>\L6.json`. Copy to `docs/hw-logs/qwen05b-divergence/` (scrubbed) and update the doc.
- Time: 25 to 40 min. On failure: a flip at a large margin (> 0.2) is a real bug. Bisect with `CUDA_FOR_INTEL_SUBST=none`, then
  one family at a time (`auto,-mmq`, `auto,-mmvq`, `auto,-quantize_q8_1`); then `CFI_OPT_LD16=0`, `CFI_XMX=0`, `CFI_DP4A` unset; run
  with `CUDA_FOR_INTEL_SYNC_LAUNCH=1` to localise faults. File the smallest differing kernel via `CFI_PTX_DUMP_DIR`.

## 2. XMX speed

- Command: `cargo test --release --test gpu hw_xmx -- --ignored --nocapture --test-threads=1 2>&1 | Tee-Object <Out>\xmx.txt`
  (probe, GEMM table, and the Qwen shapes), then the torch fp16 matmul A/B:
  `scripts/torch/run_progressive.ps1 -Levels L2` once with `CFI_XMX=0` and once with it unset; compare the L2 seconds, plus the table in `docs/pytorch.md` (fp16 2048^3: 14.7 ms off, 4.8 ms on).
- Pass: XMX result equals the tiled PTX GEMM within fp16 tolerance on every table row; fp16 2048^3 at most 6 ms; no row where
  auto mode is more than 10 % slower than `CFI_XMX=0`.
- Output: `<Out>\xmx.txt`; commit as `docs/hw-logs/xmx-<date>.txt`.
- Time: 30 min. On failure: wrong numbers: `CFI_XMX_TRACE=1`, retry `CFI_XMX_TILE=1x1`; slow rows: raise `CFI_XMX_MIN_DIM` /
  `CFI_XMX_MIN_WORK`, record the crossover, change defaults only with the table as evidence.

## 3. cuDNN default-on decision

- Precondition: the cuDNN stand-in exists only on the `cudnn-fast` branch (not in this tree). Merge or check it out first; if it is
  absent the answer today is "stay off" (`torch.backends.cudnn.enabled = False`).
- Commands: the ladder (`-Levels L4,L5`) with the stand-in installed, once as is (cudnn disabled by the harness) and once after
  changing the one `torch.backends.cudnn.enabled = False` line in `setup_torch()` of `scripts/torch/progressive.py` to `True` (local edit, do not commit);
  on that branch also run `<Py> scripts/cudnn/test_callers.py` with cudnn enabled and disabled.
- Pass (default-on): conv fwd/bwd, batch norm, pooling equal CPU within L4 tolerances, no 209/device loss across three consecutive runs,
  and conv at least as fast as the native kernels on a ResNet-18-sized forward.
- Output: `<Out>\L4.json`, `<Out>\L5.json` for each setting. Time: 30 min.
- On failure: keep default-off, record the failing op and shape in `docs/pytorch-status.md` (cuDNN row).

## 4. SASS diagnostic on a SASS-only wheel

- What to expect: a one-time error naming the SASS architectures and pointing at `docs/sass-support.md`, then
  `CUDA_ERROR_NO_BINARY_FOR_GPU` (209); torch prints `no kernel image is available for execution on the device`.
- Test input (no big download): `python3 scripts/ptxscan.py <file>` classifies a binary (exit 0 all fatbins PTX, 1 some SASS-only,
  2 no PTX, 3 no device code). For an end-to-end test use any CUDA program or wheel DLL known to be SASS-only
  (Windows torch cu126/cu128 `torch_cuda.dll`, see `docs/sass-support.md`), or build a tiny one: compile `tests/cuda_apps/basic_kernels.cu`
  with `nvcc -gencode arch=compute_86,code=sm_86` (SASS only, no `compute_` PTX) on a machine that has the toolkit.
- Command: with the shim preloaded, `<Py> -c "import ctypes;ctypes.WinDLL(r'<nvcuda.dll>');import torch;torch.zeros(4,device='cuda')" 2> <Out>\sass.txt`
  (cu126+ wheel venv), then `python3 scripts/ptxscan.py <torch_cuda.dll>; echo $?` (expect 1 or 2).
- Pass: stderr contains the SASS architecture list and `docs/sass-support.md`, exactly once per module, process exits with a Python
  RuntimeError (not a hang or device loss), and a following PTX-bearing run in a new process still works.
- Time: 5 min. On failure: message missing means the fatbin path in `src/cuda.rs::load_module_from_fatbin` / `src/cuda/library.rs`
  is bypassed (cudart 11.x uses CUDART_INTERFACE slots 1/6/8); trace with `CUDA_FOR_INTEL_TRACE=1`.

## 5. API coverage

- Commands: `cargo test --release --test gpu -- --ignored --test-threads=1 --skip hw_nvrtc 2>&1 | Tee-Object <Out>\gpu.txt`, and
  `tests\cuda_apps\build.bat` then `tests\cuda_apps\run.bat` (needs the CUDA toolkit to build; run holds the queue slot).
  For lists of unimplemented driver entry points use `CUDA_FOR_INTEL_PROFILE=1` (per-API counts in the export log) on a real workload.
- Pass: GPU suite all green (last full run 79/79 plus later additions; the 3-minute `hw_stock_ggml_kernels_compile` is optional),
  every cuda_apps program prints PASS; no `CUDA_ERROR_NOT_SUPPORTED` in the workloads' stderr beyond those listed in `docs/compatibility.md`.
- Output: `<Out>\gpu.txt` to `docs/hw-logs/test-gpu.log`. Time: 25 min.
- On failure: re-run the single test with `--nocapture`, set `CUDA_FOR_INTEL_SYNC_LAUNCH=1`; classify as translator, runtime or driver.

## 6. Translator edge

- Mac side first (free): `tests/ptx_corpus/run.sh` must exit 0 (expected_status.tsv unchanged).
- Box: dump the big modules and the remaining failures: `CFI_PTX_DUMP_DIR=<Out>\ptx` while running the ladder with the cuda128 conda torch,
  then `ptx-corpus-harness --probe <Out>\ptx\module_N.ptx` (builds on macOS, run it there). Targets: the 2 `complex<double>` foreach
  kernels, the 61 oversized cuda128 modules (use `CUDA_FOR_INTEL_BIG_FLAGS`), the ~6 % failing entries (`docs/torch-bringup.md`).
- Pass: no regression of the failing-entry count recorded in `docs/torch-bringup.md`; no new device loss in the ladder.
- Output: `<Out>\ptx\` summary lines (group counts) into `docs/ptx-gap-report.md`. Time: 40 min.
- On failure: minimise the PTX with the harness, add it to `tests/ptx_corpus` with a `--bless`ed status line in the same commit as the fix.

## 7. Perf profiling (decode / prefill)

- New this branch: `CFI_PROFILE=1` and `CFI_PROFILE_OUT=<file>`. Serialises every launch (ranks kernels, does not give absolute speed;
  run a normal unprofiled pass for absolute numbers).
- Commands: `$env:CFI_PROFILE='1'; $env:CFI_PROFILE_OUT='<Out>\prof-decode.tsv'` then run the workload (stock Ollama decode of the
  Qwen 0.5B Q8_0 GGUF: `scripts/benchmark-ollama.ps1`, or `scripts/torch/run_progressive.ps1 -Profile` for torch), repeat with a long prompt for
  prefill into `prof-prefill.tsv`. Summarise on the Mac:
  `python3 scripts/profile/summarize.py <Out>\prof-decode.tsv --top 25 --out <Out>\prof-decode.md`.
- Pass: the process exits normally, the file exists and starts with `CFIPROF`, the top-10 kernels cover more than 60 % of kernel time, and the
  launch count per decoded token is stable between runs (compare to `docs/hw-logs/perf-depth.md`).
- Output: `docs/hw-logs/profile-decode.md`, `profile-prefill.md` (tables from summarize.py). Time: 15 min.
- On failure: file empty means the process was killed before `atexit`; the file is also rewritten on context destroy, so stop the workload gracefully.
  High `submit %` for small kernels means launch overhead dominates: look at the API table (`cuLaunchKernel`, `cuStreamSynchronize` host time).

## 8. LD16 default decision (`CFI_OPT_LD16`)

- Background: +35 % decode on Qwen2.5-0.5B Q8_0, opt-in because of an unexplained batched-GEMM launch failure seen earlier (README).
- Commands: for each of `CFI_OPT_LD16=0` and `=1`, three times each (cache cleared with `CFI_CACHE_DIR` per setting because the cache key includes the switch):
  `scripts\benchmark-ollama.ps1` for decode/prefill (`scripts/bench-subst-ollama.ps1` for the SUBST matrix), then the torch ladder `-Levels L2,L5` and the
  cublas batched test `cargo test --release --test gpu hw_cublas_batched -- --ignored --nocapture --test-threads=1` with `CFI_OPT_LD16=1`.
- Pass (make default-on): decode gain >= 10 % reproducibly; outputs identical to LD16=0 for the 12-prompt scan (`docs/hw-logs/stock-scan.md`);
  ladder and the batched GEMM test green 3 of 3 with LD16=1.
- Output: `docs/hw-logs/ld16-<date>.md` with the A/B table. Time: 45 min.
- On failure (any launch failure or token mismatch): leave opt-in, capture the failing kernel with `CUDA_FOR_INTEL_SYNC_LAUNCH=1`, dump its PTX
  (`CFI_PTX_DUMP_DIR`) and compare the merged-load output with the unmerged one in the harness.
