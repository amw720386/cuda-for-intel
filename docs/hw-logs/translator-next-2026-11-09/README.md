# translator-next (TRANSLATOR_VERSION 6) on the Arc B390, 2026-11-09

Build: main + translator-next + win-redirect (integration revision 31f1c98; translator-next is the only translator change in it),
`TRANSLATOR_VERSION` 6, driver 32.0.101.9033. Last good run for comparison: run_all run 3 of the hw-validate branch
(`../hw-validate-2026-11-09/`, `TRANSLATOR_VERSION` 5).

| check | result |
|---|---|
| build, script parse (run_all stage 0) | pass |
| stock Ollama token comparison (run_all stage 2: 3 models x short/long prompt, CPU vs FA off vs FA on, 48 tokens) | **identical to run 3** in all 6 rows (same pass/review verdicts, same first-difference offsets 149/109/96, same texts); `run_all_stage0_2.md` |
| full GPU suite (`--ignored`, without `hw_stock_ggml_kernels_compile`) | 108 of 110, see below; `gpu_suite.txt` |
| `hw_tier2` rerun with the corrected test | 9 of 9 pass (`hw_tier2_rerun.txt`) |
| translator-edge PTX probe (17 cu128 modules, 4410 entries, Mac harness) | 4410 of 4410 emit, same as `TRANSLATOR_VERSION` 5 (`../checklist-2026-11-09/raw/tedge_probe_tv6.txt`) |
| PTX corpus (macOS, `tests/ptx_corpus/run.sh`) | 555 pass, 57 unsupported, 15 error; matches `expected_status.tsv` |

The two suite failures:
- `hw_f64_conversions_match_cpu`: `cvt.rzi.s32.f64` of NaN returned `0x80000000`, the test expected 0. translator-next deliberately implements
  the PTX ISA 9.0 rule (NaN gives 0 only for a non-`.f64` source with a destination of 32 bits or less, otherwise `1 << (bits - 1)`; LLVM's
  NVPTX backend made the same correction for `d2i`). The test expectation was wrong and is fixed in this branch; rerun passes.
- `hw_subst_native_vs_translated`: the same shared-export-log count failure as on main (`../checklist-2026-11-09/`); its numeric checks pass,
  and `hw_subst` passes 3 of 3 on its own.
