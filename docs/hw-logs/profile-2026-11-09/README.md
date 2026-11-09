# CFI_PROFILE of stock Ollama, prefill and decode (checklist item 7), 2026-11-09

Arc B390, driver 32.0.101.9033, stock Ollama v0.12.6, SUBST auto, LD16 off. Build: main with fix-divergence (`TRANSLATOR_VERSION` 5).
Driver: `scripts/profile/ollama_profile.ps1` (one GPU queue slot). Per server: warm-up (a different 624-token prompt, 8 tokens), short
(15 tokens, 1 token), long (624 tokens, 1 token), decode (another 15-token prompt, 64 tokens). Prefill table = `long` snapshot minus `short`;
decode table = `decode` minus `long` (includes that 15-token prompt). `CFI_PROFILE` serialises every launch: use the tables to rank kernels, the
unprofiled ("plain") server for speed. Llama-3.2-1B FA off is missing (a loop-variable bug in the driver, fixed in 378747b).

## Unprofiled speed (tok/s; prefill of the 624-token prompt after warm-up, decode of 64 tokens)

| build / setting | Qwen 0.5B Q8_0 FA off | Qwen FA on | Llama-3.2-1B Q4_K_M FA on |
|---|---|---|---|
| this build | 6764 / 34.7 | 4458 / 43.2 | 3038 / 45.2 |
| this build, `CFI_XMX=0` | 6944 / 40.8 | | |
| gate4 release-candidate build (steady state) | 6759 / 35.9 | 4497 / 39.6 | |

The gate4 build's first-request (cold) numbers in its FA check (2 to 17 tok/s prefill) include first-use kernel translation; warm, it matches this
build. The 15-token `short` step is also cold (new kernel shapes, about 2 tok/s) and is not a steady-state number. Decode varies run to run (three identical
LD16=0 runs: Qwen 32.3 to 34.6, Llama 44.6 to 48.9 tok/s, `../checklist-2026-11-09/`), so treat single decode numbers as about +/-10 %.

## What dominates (share of serialised kernel time; `families.txt`)

| phase | FA off (Qwen) | FA on (Qwen) | FA on (Llama 1B) |
|---|---|---|---|
| prefill | mmq 69 %, soft_max 8 %, KQ/KQV GEMMs (XMX) 5 %, quantize 4 %, rest 12 % | mmq 47 %, **flash_attn_ext_f16 42 %**, quantize 4 %, rest 6 % | mmq 48 %, **flash_attn_ext_f16 45 %** |
| decode | **mmvq 73 %**, quantize 5 %, f16 mat-vec 3 %, rest 13 % | **mmvq 71 %**, FA (tile + vec) 7 %, quantize 5 % | **mmvq 65 %**, flash_attn_ext_f16 17 % |

Top-kernel tables: `qwen_fa0_prefill.md`, `qwen_fa1_prefill.md`, `qwen_fa0_decode.md`, `qwen_fa1_decode.md`, `llama1b_fa1_*.md`, and `*_xmx0.md`
(`CFI_XMX=0`: the KQ/KQV GEMMs move to the tiled path, about 36 ms instead of 15 ms of 342 ms; mmq unchanged).

Reading:
- FA off vs FA on, prefill: the unfused attention (soft_max + two cuBLAS GEMMs) costs about 45 ms for Qwen's 624-token prompt; the translated
  `flash_attn_ext_f16` tile kernel (`<64,64,32,1,4,2>`, 32 launches) costs 192 ms for the same work, about 4x. That is why FA-on prefill is
  about 35 % slower end to end (4458 vs 6764 tok/s). FA-on decode is faster for Qwen (43 vs 35 tok/s) because the vec kernel replaces several small launches.
- Decode is matrix-vector bound in every setting: the native `mul_mat_vec_q` (mmvq) substitute runs about 160 launches per token at about 100 us
  each, about 16 ms per token of the 29 ms per token measured unprofiled.

## Fixable hotspots (described, not implemented)

1. **`flash_attn_ext_f16` tile kernel, FA-on prefill: 42 % (Qwen), 45 % (Llama) of prefill kernel time.** It is the stock CUDA `mma.sync` tile
   kernel translated to SPIR-V; tensor-core `mma` becomes scalar or subgroup code, about 4x slower than the unfused path for the same attention.
   Proposed fix: a native substitute in the existing SUBST framework (`flash_attn_ext_f16` for D=64/128, ncols 16-64) that does QK^T and PV
   with the XMX DPAS path already used by `cfi_xmx_*`, with online softmax in registers; a cheaper interim fix is to report FA prefill as
   unsupported for ncols >= 16 so ggml falls back to the unfused path, which is already faster. Expected gain: FA-on prefill to at least FA-off
   speed (+50 % FA-on prefill), and the Llama FA-on decode tile launches (17 %).
2. **`mul_mat_vec_q` (Q8_0/Q4_K/Q6_K), decode: 65-73 % of decode kernel time.** About 100 us per launch for Qwen 0.5B's 896-wide rows suggests
   the kernel is far from memory bandwidth on this part (the Q8_0 GGUF is about 0.64 GB including the tied embedding/output matrix, read in about 16 ms of mat-vec per token, roughly 40 GB/s).
   Proposed fix: wider per-thread loads (16-byte, the LD16 idea, but inside the native mmvq rather than the translator), several rows per
   subgroup with a single subgroup reduction, and merging `quantize_q8_1` (5 %, 7 us per launch, one per mat-vec) into the mat-vec prologue.
   Check against the NWARPS variants in `../qwen05b-divergence/2026-11-09/` (the add order changes; text-identity tests must be re-blessed).
