# Moving the CUDA path to stock Ollama 0.40

Goal: compare against Ollama's Vulkan backend on the same release. The Vulkan baseline
([hw-logs/vulkan-baseline-2026-11-10.md](hw-logs/vulkan-baseline-2026-11-10.md)) uses 0.40.14,
because 0.12.6 for Windows ships no Vulkan backend.

## What 0.40.14's CUDA 13 plug-in needs (measured offline, no GPU)

Source: `ollama-windows-amd64.zip` of v0.40.14 (SHA-256 `65d68cad...7e29`),
`lib/ollama/cuda_v13/ggml-cuda.dll` (SHA-256 `2be789bc...d498`).

| Item | 0.40.14 | Status |
| --- | --- | --- |
| Driver API | no static `nvcuda` imports; everything goes through the runtime's `cuGetProcAddress` | served |
| cuBLAS | 13 imports (`Sgemm`, `GemmEx`, `Gemm{Strided,}BatchedEx`, `Sgemm{Strided,}Batched`, `StrsmBatched`, handle/stream/workspace/math mode, `GetStatusString`) | all implemented in `cublas_shim` |
| PTX translation | `scan_stock_dll_translation`: 146 modules, **7,221 of 7,223 entries translate**. The two failures are `ssm_conv_f32` (Mamba models only, `expected Local address`) | ready; FlashAttention kernels all translate (0.12.6 had 87 failures) |
| Stock `mul_mat_vec_q` numerics | `ggml_conformance::mmvq_*` on the 0.40 PTX: all 12 quant types, 1 to 8 columns, match the reference | correct when translated |
| NVRTC | the plug-in ships `nvrtc64_130_0.dll` and nvJitLink | not exercised yet |

## Native kernel substitution

The decode and prefill speedups come from native substitutes chosen by kernel name and exact
parameter layout (`src/subst`). Three of them changed signature in 0.40:

| Kernel | 0.12.6 | 0.40 | Status |
| --- | --- | --- | --- |
| `quantize_q8_1` | `PKfPvxxxxxj5uint3` | unchanged | substituted |
| `mul_mat_vec_q` (decode) | `<type, ncols>`, 23 words | `<type, ncols, has_fusion, small_k, halve_iters>`, 80-byte fusion struct after `ids`, trailing `ids_stride`: 44 words | **ported** |
| `mul_mat_vec_q` with fusion | none | gate dot product + GLU, row biases, shared expert | **ported for bias-only and SwiGLU epilogues without MoE** (Qwen/Llama q/k/v and FFN gate/up); GEGLU, SWIGLU_OAI/CLAMP, NVFP4 scales, MoE ids and shared experts stay on the translated kernel, decided per launch |
| `mul_mat_q` + `stream_k_fixup` (prefill) | `<type, mmq_x, need_check>` | `<type, mmq_x, need_check, ggml_prec, fused_glu>`, gate and GLU arguments | translated stock kernel; native port open |
| `flash_attn_ext_f16` (#18) | 0.12.6 template | changed | translated stock kernel |

`mul_mat_vec_q` details for 0.40, compiled for compute_75 (what a reported 7.5 selects, the
`MMVQ_PARAMETERS_TURING` table):

- Warps per block: one column 2 for K-quants (Q2_K to Q6_K), 4 otherwise; 2 to 4 columns 4;
  5 to 8 columns 2. Checked against the `.maxntid` of every instantiation.
- `small_k` (chosen when K has fewer blocks than one pass of the block covers: K < 1024 for
  Q8_0, so most of Qwen2.5-0.5B's decode products) gives each warp its own row: rows per block
  = warps per block. `halve_iters` only affects the GB10 table.
- The launch hook hands the native kernels the legacy 23 words (drops the fusion words and
  `ids_stride`), so the native kernels and their probes are unchanged apart from rows per block.
  For `has_fusion` kernels it appends `x_bias`, `gate`, `gate_bias` and `glu_op` from the
  struct; the native kernel adds the bias, and with a gate accumulates the gate dot product in
  the same loop and multiplies by `silu(gate + gate_bias)` (ggml's `x / (1 + expf(-x))`).
- In `has_fusion` kernels the translator types `vx` and `dst` as u64 (the shared-expert branch
  reassigns them); the registry treats u64 and pointer slots as the same 8 bytes.
- Q6_K: ggml raised `VDR_Q6_K_Q8_1_MMVQ` from 1 to 2, so the stock 0.40 kernel splits a block
  differently and rounds differently (NMSE about 1e-14 against the v0.12.6-order native kernel).
  The other types are bit for bit.

Tests (no GPU): `cargo test --lib -- fused_abi small_k_native fused_native fusion_launch`; with
the 0.40 PTX extracted locally (never committed) as `blob40.ptx` in a directory `D`:
`CFI_STOCK_PTX_DIR_040=D CFI_GGML_PTX_DIR=D cargo test --release --lib -- stock_0_40 mmvq_fused`
(the registry accepts all 52 supported instantiations, 12 of them fused; stock 0.40 output equals
the reference bit for bit without fusion except Q6_K, and to NMSE < 1e-10 with the fused epilogue,
where the stock kernel's expf is NVIDIA's ex2.approx sequence; the native kernels equal the same
reference).

## Next

1. Hardware (B390, then GPU A): stock 0.40.14 with the shim: `/info` reports CUDA, Qwen2.5-0.5B
   Q8_0 loads with `library=CUDA` and no Vulkan/CPU fallback, the `subst:` log lines show the
   native `mul_mat_vec_q` (including `:fused` keys) taken, tokens match the CPU backend, then
   `scripts/bench-decode.ps1` against the Vulkan rows of the baseline with the same request, with
   fusion on (default) and with `GGML_CUDA_DISABLE_FUSION=1` (more launches per token).
2. Native `mul_mat_q` for the 0.40 signature (prefill), then FlashAttention.
3. Q6_K in 0.40's `vdr` 2 order, for bit-identical output with the stock kernel.
