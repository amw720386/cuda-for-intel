# hw-validate results

revision: eebe988 hw-validate  overall: **fail**  generated: 2026-11-09T04:00:11

| stage | title | status | secs |
|---|---|---|---|
| 0 | environment + build | pass | 19 |
| 1 | baseline gate (FAST_DIV=1, XMX_BLK=0 = main behaviour) | pass | 198 |
| 2 | FA Ollama text check (CPU vs shim FA off vs FA on) | review | 1526 |
| 3 | divergence (fix-divergence) | fail | 129 |
| 4 | translator edge (translator-hardening) | pass | 23 |
| 5 | api coverage (api-coverage) | pass | 24 |
| 6 | XMX 2D block loads (xmx-speed) | pass | 709 |
| 7 | cuDNN (cudnn-final) | pass | 173 |
| 8 | perf matrix (631 prefill, 48 decode) | pass | 1934 |
| 9 | installer self-test (+ torch) | pass | 34 |

## stage 0: environment + build - pass
- [pass] disk-free-gb: 1605.4 GB free (need 30)
- [info] stale-processes: present (not touched, may belong to others): ollama:14164 ollama app:6284
- [info] gpu-queue: lock free
- [pass] ps1-parse-check: all scripts parse
- [pass] cargo-build-release: rc=0
- [pass] cargo-test-build: rc=0
- [info] dll-sha256: nvcuda.dll=135A890F9A7C73E0 cublas64_12.dll=8FAD39FF0586211B
- [info] translator-version: source TRANSLATOR_VERSION=5 (expected 5 on hw-validate)

## stage 1: baseline gate (FAST_DIV=1, XMX_BLK=0 = main behaviour) - pass
- [pass] baseline-hw_subst: passed=3 failed=0
- [pass] baseline-hw_nvrtc: passed=1 failed=0
- [pass] baseline-hw_fa_vec: passed=2 failed=0
- [pass] baseline-rest-of-suite: passed=86 failed=0

## stage 2: FA Ollama text check (CPU vs shim FA off vs FA on) - review
- [pass] fa-qwen-short: fa1==fa0:False fa1==cpu:True first_diff_vs_cpu=-1 text=' Paris is a city of contrasts, with a rich history and a vibrant culture that has captivat'
- [pass] fa-qwen-long: fa1==fa0:True fa1==cpu:True first_diff_vs_cpu=-1 text=' The capital of France is Paris. It is a city located in the south of France, known for it'
- [pass] fa-llama32-1b-q4km-short: fa1==fa0:False fa1==cpu:True first_diff_vs_cpu=-1 text=' Paris is a city of great historical significance, known for its stunning architecture, vi'
- [review] fa-llama32-1b-q4km-long: fa1==fa0:False fa1==cpu:False first_diff_vs_cpu=149 text=' The capital of France is Paris. Paris is a major city in France, known for its historic a'
  - blame: read the text: coherent diverging text is expected fp drift; gibberish = FA/translator regression
- [review] fa-qwen25-15b-q4km-short: fa1==fa0:True fa1==cpu:False first_diff_vs_cpu=109 text=' Paris is the capital and most populous city of France. It is located on the Seine River i'
  - blame: read the text: coherent diverging text is expected fp drift; gibberish = FA/translator regression
- [review] fa-qwen25-15b-q4km-long: fa1==fa0:True fa1==cpu:False first_diff_vs_cpu=96 text=' The capital of France is Paris. Paris is a city located on the Seine River in the northwe'
  - blame: read the text: coherent diverging text is expected fp drift; gibberish = FA/translator regression

## stage 3: divergence (fix-divergence) - fail
- [fail] div_sqrt-FAST_DIV=1-must-fail: expected the test to FAIL with this switch, but it did not (passed=1 failed=0)
  - blame: fix-divergence (opt-out switch not effective, or test cannot detect inexact div)
  - log: logs\div_sqrt-FAST_DIV_1-must-fail.out/.err
- [pass] div_sqrt-default-must-pass: passed=1 failed=0
- [pass] quantize_fuzz-default: passed=1 failed=0
- [info] auto-vs-none-logits: prompts=20 diverging_auto_vs_none=7 tool_logits=False (fewer diverging prompts than before the fix = improvement; see logs\divergence_scan_summary.txt)

## stage 4: translator edge (translator-hardening) - pass
- [pass] hw_translator_edge: passed=6 failed=0
- [pass] translator-unit-tests: test result: ok. 223 passed; 0 failed
- [skip] corpus-scan-subset: CFI_CORPUS_DIR not set

## stage 5: api coverage (api-coverage) - pass
- [pass] api_coverage: passed=3 failed=0
- [pass] sgemv-sdot-numerics: passed=1 failed=0
- [pass] shim-crates-unit-tests: rc=0

## stage 6: XMX 2D block loads (xmx-speed) - pass
- [pass] hw_xmx_blk: passed=1 failed=0
- [pass] hw_xmx_shapes: passed=1 failed=0
- [pass] hw_xmx_probe-trace: passed=1 failed=0
- [info] xmx-trace-lines: 10 lines mention XMX in the trace output
- [pass] hw_xmx_gemm-other: passed=3 failed=0
- [info] ollama-qwen-xmx0: prefill_tps=6633.7 decode_tps=28.1 tokens=631+48
- [info] ollama-qwen-default: prefill_tps=7212.8 decode_tps=29.5 tokens=631+48
- [info] ollama-qwen-blk0: prefill_tps=7079.8 decode_tps=28.1 tokens=631+48
- [info] ollama-qwen-trace: prefill_tps=7199 decode_tps=29 tokens=631+48 xmx_trace_lines=292
- [pass] text-xmx0-vs-default: first_diff_char=-1 (f16 accumulation order differs with XMX; coherent text = fine)
- [pass] text-xmx0-vs-blk0: first_diff_char=-1 (f16 accumulation order differs with XMX; coherent text = fine)
- [pass] text-xmx0-vs-trace: first_diff_char=-1 (f16 accumulation order differs with XMX; coherent text = fine)

## stage 7: cuDNN (cudnn-final) - pass
- [pass] hw_cudnn: passed=1 failed=0
- [pass] cudnn-gate-final: steps=10 bad=0 

## stage 8: perf matrix (631 prefill, 48 decode) - pass
- [info] perf-qwen-cpu-ref: calibrated words=607
- [pass] perf-qwen-fa0_ld0: prefill_tps=7067.7 decode_tps=30.5 tokens=631+48 text_eq_cpu=True first_diff=-1
- [pass] perf-qwen-fa0_ld1: prefill_tps=7093.5 decode_tps=32 tokens=631+48 text_eq_cpu=True first_diff=-1
- [pass] perf-qwen-fa1_ld0: prefill_tps=4468.5 decode_tps=51.4 tokens=631+48 text_eq_cpu=True first_diff=-1
- [pass] perf-qwen-fa1_ld1: prefill_tps=4470.1 decode_tps=48.8 tokens=631+48 text_eq_cpu=True first_diff=-1
- [info] perf-llama32-1b-q4km-cpu-ref: calibrated words=608
- [pass] perf-llama32-1b-q4km-fa0_ld0: prefill_tps=5130.7 decode_tps=35.4 tokens=631+48 text_eq_cpu=True first_diff=-1
- [pass] perf-llama32-1b-q4km-fa0_ld1: prefill_tps=5152.8 decode_tps=35.1 tokens=631+48 text_eq_cpu=True first_diff=-1
- [pass] perf-llama32-1b-q4km-fa1_ld0: prefill_tps=3007.3 decode_tps=41.1 tokens=631+48 text_eq_cpu=True first_diff=-1
- [pass] perf-llama32-1b-q4km-fa1_ld1: prefill_tps=2990.8 decode_tps=51.8 tokens=631+48 text_eq_cpu=True first_diff=-1
- [info] perf-qwen25-15b-q4km-cpu-ref: calibrated words=607
- [pass] perf-qwen25-15b-q4km-fa0_ld0: prefill_tps=3983 decode_tps=29.2 tokens=631+48 text_eq_cpu=True first_diff=-1
- [pass] perf-qwen25-15b-q4km-fa0_ld1: prefill_tps=3998.1 decode_tps=24.7 tokens=631+48 text_eq_cpu=True first_diff=-1
- [pass] perf-qwen25-15b-q4km-fa1_ld0: prefill_tps=2123.7 decode_tps=27.1 tokens=631+48 text_eq_cpu=True first_diff=-1
- [pass] perf-qwen25-15b-q4km-fa1_ld1: prefill_tps=2120.8 decode_tps=25 tokens=631+48 text_eq_cpu=True first_diff=-1

## stage 9: installer self-test (+ torch) - pass
- [pass] cfi-install-dryrun: rc=0
- [pass] cfi-install-real: rc=0
- [pass] cfi-doctor: rc=0
- [pass] cfi-selftest: rc=0
- [pass] installer-output-leak-grep: no username/host/IP in install, doctor or selftest output
- [pass] cfi-uninstall-roundtrip: rc=0 manifest_removed=True
- [pass] torch-stage: rc=0 (details in its own logs)

