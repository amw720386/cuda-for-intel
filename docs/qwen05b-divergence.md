# Qwen2.5-0.5B Q8_0: why `SUBST=auto` and `SUBST=none` diverge

Question: is the greedy divergence between kernel substitution `auto` and `none` (stock Ollama v0.12.6 through the shim, Arc B390) a bug or numeric noise?

## Method
* Model: Qwen2.5-0.5B-Instruct Q8_0 GGUF (sha256 in `hw-logs/qwen05b-divergence/provenance.txt`). Shim built from main 78ff0b6 (mmq, mmvq, quantize_q8_1 all active in `auto`).
* Stock Ollama v0.12.6 has no logprobs, so logits come from `scripts/divergence/cfi_logits.cpp` (llama.cpp, same shim, same `CUDA_FOR_INTEL_SUBST` switch): the prompt is one prefill batch, every later token a batch of one, teacher-forced or free-run, full-vocab fp32 logits per step.
* Validation of the harness: Ollama (`raw`, `top_k=1`, `repeat_penalty=1`) and the harness produce the identical 96-token text in both modes for the chosen prompt, and diverge at the same point (generated token 38, Ollama char 166).
* Reference: the same GGUF dequantised to fp32, run in transformers on CPU (`analyze_divergence.py`), fed the same token sequence.
* Prompt: `The quick brown fox jumps over the lazy dog. The quick brown fox` (14 tokens), 96 greedy tokens. In Ollama 16 of 20 scanned prompts differ between auto and none (`scan_summary.txt`).

## Results (96 steps, teacher-forced on the none-generated sequence; the auto-generated sequence gives the same picture, see `analysis-fp32.md`)
| | auto | none | CPU q8 pipeline |
|---|---|---|---|
| max abs dlogit vs fp32 ref (vocab / ref top-20) | 1.066 / 0.531 | 0.986 / 0.514 | 0.978 / 0.579 |
| RMS dlogit vs fp32 ref (mean over steps) | 0.112 | 0.112 | 0.114 |
| top-1 != fp32 top-1 (of 96) | 2 | 1 | 2 |
| RMS dlogit vs none | 0.095 | 0 | 0.105 |
| max abs dlogit vs none | 0.880 | 0 | 1.455 |

Top-1 flips auto vs none: exactly one in 96 steps, step 38 (the observed divergence). fp32 reference top-2 margin there is 0.030 (none: 0.0028, auto: 0.079), whereas the auto-none logit difference at that step is 0.32. The next smallest reference margins (0.049, 0.065) did not flip. A flip needs a margin smaller than the step's noise; no flip had a larger margin.

Isolation (`SUBST` value, each row a separate run): `mmvq` only is bit-identical to none; `quantize_q8_1` only differs from step 9 (RMS 0.08); `mmq` only differs from step 0 (RMS 0.097). Runs are deterministic (same config twice: bit-identical).

## Conclusion
No bug signature: auto is as close to the fp32 reference as none (RMS 0.112 vs 0.112, max 1.07 vs 0.99, top-20 0.53 vs 0.51), and the single flip sits at a near-tie. Caveat: the auto-none difference is NOT small float reordering (that would be about 1e-5). It is 0.1 RMS, the same size as the difference between none and the llama.cpp CPU backend and as each backend's distance to fp32. The consistent reading is that any perturbation (native add order in mmq, rounding in the native quantize_q8_1) is amplified by the discrete q8 activation rounding through 24 layers to this activation-quantisation noise floor. The planned control (mmvq `CUDA_FOR_INTEL_SUBST_NWARPS` 1/2/8, pure add-order changes, variants `mmvq_nw*`) was queued but not completed, so the amplification claim is supported by the CPU-vs-none comparison only. The native quantize_q8_1 is not bit-identical to the translated kernel (probably division/rounding); worth a separate check.

## Reproduce
Box: `scripts/divergence/run_on_box.ps1` (Ollama scan + `cfi_logits.exe` runs, holds the GPU queue slot; build the tool with `build_cfi_logits.cmd`). Mac: `python3 scripts/divergence/analyze_divergence.py --results <dir> --gguf <model.gguf>` (needs torch, transformers, gguf). Raw logits (60 MB per run) are not committed.
