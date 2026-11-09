# GPU-day checklist results, 2026-11-09 (Arc B390, driver 32.0.101.9033)

One detached job held one GPU queue slot for all items (`docs/gpu-day-checklist.md` has the commands and pass criteria).
Build: hw-validate (main + fix-divergence + hw-validate scripts, `TRANSLATOR_VERSION` 5). Torch: cu118 venv (torch 2.2.2) unless noted.

| # | item | verdict | numbers |
|---|---|---|---|
| 7 | perf profile | **pass** (rerun) | first attempt produced no data (`ollama_profile.ps1` flattened its one-mode list and read a null exit code; fixed in 61b9912). Rerun: `../profile-2026-11-09/`. Files start with `CFIPROF`, processes exit normally; top 10 kernels cover 90 % or more of kernel time in every phase; decode about 160 mat-vec launches per token in each run |
| 4 | SASS diagnostic | **pass** | cu128 (torch 2.11) and cu130 (torch 2.14): the SASS-only message (architecture list + `docs/sass-support.md`) printed exactly once, then Python `AcceleratorError: no kernel image is available` (no hang, no device loss); `ptxscan.py` exit 2 (404 and 375 fatbins, none with PTX); a following cu118 run in a new process prints `ok 4.0` |
| 1 | Qwen-0.5B divergence (NWARPS, FAST_DIV) | **pass** (relative criteria) | `../qwen05b-divergence/2026-11-09/`: FAST_DIV=1 12 of 20 prompts diverge vs 7 of 20 with the default; NWARPS 1/2/8 RMS 0.073-0.077 vs none, auto 0.078 (no worse than an add-order change); native quantize_q8_1 now bit-identical. fp32-reference RMS not re-measured |
| 2 | XMX speed | **partial**: correctness and 2048^3 pass, "no row >10 % slower" fails | `../xmx-2026-11-09.txt`: 574 table cases bit-equal to tiled PTX, fp16 2048^3 1.98 ms (<= 6 ms); 896x512x896 f32 0.50x and 896x631x896 f16 0.79x with XMX. Torch L2 1.7 s (`CFI_XMX=0`) vs 1.6 s. End to end, Qwen 0.5B prefill: +9 % (run_all stage 6) and -3 % (profile run): no clear gain |
| 3 | cuDNN default-on | **stay off** | L4/L5 pass 3 of 3 with `cudnn.enabled=True` and the stand-in preloaded, no 209/device loss; but L4 is slower (9.9 / 9.9 / 10.4 s on vs 8.5 s off) and no ResNet-18 conv timing was taken |
| 5 | API coverage (full GPU suite) | **pass with one harness failure** | 109 of 110; `hw_subst_native_vs_translated` failed only its "a native kernel was installed" check, which counts lines in the shared `cuda-for-intel-export.log`; every numeric comparison in it passed (worst NMSE 3.9e-17). Rerun alone with fresh caches: `hw_subst` 3 of 3 pass, 34 native kernels ready, worst NMSE 3.9e-17, so the full-suite failure is the shared-log count, not the shim. tests/cuda_apps was not part of this job |
| 6 | translator edge (cu128 conda torch, PTX dump) | **pass** | L1/L2/L3 pass; L5 aborts in torch's cuDNN v9 loader (`Could not locate cudnn_graph64_9.dll`, environment, not the shim). 17 dumped modules, 4410 entries: all emit with TRANSLATOR_VERSION 5 and with translator-next (6), 0 failures (the 6 % in `docs/torch-bringup.md` came from larger modules this ladder does not load) |
| 8 | LD16 default | **stay opt-in** | batched GEMM test 3/3 and ladder L2+L5 3/3 green with LD16=1, but the decode gain is not reproducible: Qwen +10 % median (noisy), Llama-3.2-1B -18 %. Table below |

## LD16 A/B (item 8)

Stock Ollama, FA off, SUBST auto, fresh kernel caches per run, 3 runs per setting, unprofiled (`ollama_profile.ps1 -NoProfiled`; warm-up prompt differs from the measured one).

| model | CFI_OPT_LD16 | runs | prefill tok/s (624-token prompt) | decode tok/s (64 tokens) | server errors |
|---|---|---|---|---|---|
| llama32-1b-q4km | 0 | 3 | 5079, 5203, 5227 (median 5203) | 48.5, 44.6, 48.9 (median 48.5) | 0 |
| llama32-1b-q4km | 1 | 3 | 5286, 5268, 5286 (median 5286) | 39.6, 41.1, 38.3 (median 39.6) | 0 |
| qwen | 0 | 3 | 6865, 6761, 6721 (median 6761) | 32.8, 32.3, 34.6 (median 32.8) | 0 |
| qwen | 1 | 3 | 6987, 6882, 6936 (median 6936) | 32.8, 42.6, 36.1 (median 36.1) | 0 |

llama32-1b-q4km: decode median LD16=1 vs 0: -18.4 %; prefill +1.6 %

qwen: decode median LD16=1 vs 0: +10.1 %; prefill +2.6 %

Token identity: these runs predate the response-text field; run_all run 3 stage 8 has LD16=0 and LD16=1 both equal to the CPU backend's 48-token text on all three models. `hw_cublas_batched` with LD16=1: 3 passed, 3 of 3 runs (no launch failure). Ladder L2+L5 with LD16=1: pass 3 of 3 (L5 13.4 to 13.7 s).

Verdict: the >= 10 % reproducible decode gain needed to make LD16 default-on is not there (Llama is slower), so it stays opt-in. The README's earlier "+35 % decode" claim (claims-audit row 10) stays removed.
