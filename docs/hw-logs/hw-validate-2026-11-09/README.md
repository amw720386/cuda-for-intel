# hw-validate on the Arc B390, 2026-11-09

Driver 32.0.101.9033, Windows x64, stock Ollama v0.12.6. `scripts/hwvalidate/run_all.ps1`, all stages, one GPU queue slot.
Full per-item output of the last run: `run_all-run3.md` (revision eebe988 of the hw-validate branch = main + fix-divergence +
the hw-validate scripts; `TRANSLATOR_VERSION` 5).

## Stages (run 3)

| stage | what | result | notes |
|---|---|---|---|
| 0 | build, script parse | pass | |
| 1 | baseline gate (`FAST_DIV=1`, `XMX_BLK=0`) | pass | hw_subst 3/3, hw_nvrtc 1/1, hw_fa_vec 2/2, rest of the suite 86/86 |
| 2 | FA text check, 3 models x short/long, CPU vs FA off vs FA on | review (no failure) | 3 of 6 equal CPU; the other 3 coherent and diverge at char 96 to 149 (fp drift) |
| 3 | divergence | "fail" = expectation wrong | `div_sqrt` with `FAST_DIV=1` passes: translated `div.rn` is exact on this driver without the flag. Default passes, quantize fuzz passes, auto-vs-none 7 of 20 prompts at 64 tokens. The check is informational from c9b96be on; see `../qwen05b-divergence/2026-11-09/` |
| 4 | translator edge | pass | hw_translator_edge 6/6, translator unit tests 223/223 |
| 5 | api coverage | pass | api_coverage 3/3, sgemv/sdot 1/1, shim crates |
| 6 | XMX 2D block loads | pass | blk/shapes/probe/gemm all pass; Qwen 0.5B 631-token prefill 6634 (`CFI_XMX=0`) vs 7213 (default) tok/s, text identical |
| 7 | cuDNN | pass | hw_cudnn 1/1, cudnn gate 10 steps 0 bad |
| 8 | perf matrix (631-token prefill, 48-token decode, SUBST auto) | pass | every cell's text equals the CPU backend; table below |
| 9 | installer self-test (+ torch) | pass | `SELFTEST: PASS (Intel(R) Arc(TM) B390 GPU; 1024 elements correct)`, install/doctor/uninstall round trip, no user/host/IP in the output |

Stage 8, tok/s (prefill / decode), single runs:

| model | FA off, LD16 0 | FA off, LD16 1 | FA on, LD16 0 | FA on, LD16 1 |
|---|---|---|---|---|
| Qwen2.5-0.5B Q8_0 | 7068 / 30.5 | 7094 / 32.0 | 4469 / 51.4 | 4470 / 48.8 |
| Llama-3.2-1B Q4_K_M | 5131 / 35.4 | 5153 / 35.1 | 3007 / 41.1 | 2991 / 51.8 |
| Qwen2.5-1.5B Q4_K_M | 3983 / 29.2 | 3998 / 24.7 | 2124 / 27.1 | 2121 / 25.0 |

## Runs 1 and 2 (superseded, do not quote)
- Run 1 (70a387d): script defects, not shim regressions. hw_nvrtc failed because `NVRTC_SHIM_CLANG` was unset; hw_barrier_with_exited_threads, one
  translator unit test and one shim-crate test failed on test defects fixed before run 2 and now on main (0076a01, 6591249, 7fa1d3a); stages 6 and 8 hit an Ollama tree locked by a runner the
  divergence stage had not stopped (fixed in 8034977 and the locked-tree fallback).
- Run 2 (96f60bc): the stage 8 prefill numbers (19 000 to 34 000 tok/s) are **invalid**. The measured prompt equalled the warm-up prompt,
  so Ollama served it from its KV prefix cache and only the last few tokens were evaluated. Fixed in 7e134b3 / babb2a1 (measured prompts
  differ from the warm-up); run 3 is the corrected measurement. Stage 6 failed on an Int32 overflow in the tok/s arithmetic (04ec670).
