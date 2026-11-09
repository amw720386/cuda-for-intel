# Divergence controls, 2026-11-09 (Arc B390, driver 32.0.101.9033)

Shim: hw-validate build (main + fix-divergence + hw-validate scripts, `TRANSLATOR_VERSION` 5), nvcuda.dll sha256 prefix 135A890F (`provenance.txt`).
Model: Qwen2.5-0.5B-Instruct Q8_0 (same GGUF as the earlier analysis). Ollama v0.12.6 stock, `raw`, greedy. All runs held the GPU queue slot.

## FAST_DIV control (is `-cl-fp32-correctly-rounded-divide-sqrt` worth it?)

Greedy SUBST=auto vs SUBST=none over the same 20 prompts; a prompt "diverges" when the two texts differ anywhere.

| build flags | tokens | prompts diverging | file |
|---|---|---|---|
| default (correctly rounded div/sqrt) | 64 | **7 of 20** | `scan_64tok_default64.txt` |
| default, earlier run (run_all run 2, same build) | 64 | 7 of 20, same rows and same offsets | run_all log |
| `CUDA_FOR_INTEL_FAST_DIV=1` (flag off, = main before the fix) | 64 | **12 of 20** | `scan_64tok_fastdiv.txt` |
| default | 96 | 10 of 20 | `scan_96tok_default.txt` |
| main 78ff0b6, no flag (earlier analysis) | 96 | 16 of 20 | `../scan_summary.txt` |

The SUBST=none texts are byte-identical with and without the flag (same lengths in every row): the translated kernels' `div.rn.f32`
is already exact on this driver (`hw_div_sqrt_rn_f32_correctly_rounded` passes with `FAST_DIV=1` too, 0 of 262144 wrong).
The flag changes only the native substitute kernels (built from OpenCL C, where `/` is not correctly rounded without the flag; quantize_q8_1 computes `x / d`), and moves them to
agree with the translated stock kernels on 5 more prompts. The scan is deterministic (two runs of the default build: identical rows).

## NWARPS control and family isolation (logits, `cfi_logits.exe`, teacher-forced on the SUBST=none tokens, 96 steps)

Each row is one run with only that family substituted; compared with the SUBST=none free run (same token inputs). `nwcmp` table: `nwarps_vs_none.md`.

| variant | SUBST | mean RMS dlogit vs none | max RMS | top-1 flips vs none (none's margin at the flip) |
|---|---|---|---|---|
| auto | all families | 0.0781 | 0.114 | 1 (0.126) |
| only mmq | mmq | 0.0781 | 0.114 | 1 (0.126) |
| only mmvq | mmvq (w0) | 0 (bit-identical) | 0 | 0 |
| only quantize_q8_1 | quantize_q8_1 | **0 (bit-identical)**; before the fix it differed from step 9 at RMS 0.08 | 0 | 0 |
| mmvq NWARPS=1 | mmvq, cross-warp add order changed | 0.0754 | 0.123 | 1 (0.013) |
| mmvq NWARPS=2 | same | 0.0772 | 0.124 | 2 (0.114, 0.034) |
| mmvq NWARPS=8 | same | 0.0730 | 0.140 | 4 (0.114, 0.126, 0.085, 0.034) |

Reading:
- With the flag, native quantize_q8_1 is now bit-identical to the translated kernel: the division was the whole quantize difference.
- The remaining auto-vs-none gap is entirely mmq (only-mmq equals auto exactly), and it is the same size as a pure change of the
  summation order (mmvq NWARPS 1/2/8: RMS 0.073 to 0.077, up to 4 flips). This is the activation-quantisation amplification of
  add-order noise described in `docs/qwen05b-divergence.md`, not a bug.
- Not done here: the fp32 reference (`analyze_divergence.py` needs torch and transformers, not installed on the Mac this run), so the
  checklist's "RMS vs fp32 <= 0.15" criterion is carried over from the earlier analysis (auto 0.112, none 0.112), not re-measured.
