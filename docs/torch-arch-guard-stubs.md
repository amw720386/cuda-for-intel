# Architecture-guard stubs in PyTorch PTX, and the compute capability we report

Official PyTorch Windows cu118 wheels (2.0 to 2.7) carry PTX only for compute_37. Kernels
that are gated by `#if __CUDA_ARCH__ >= 750` (the CUTLASS `fmha_cutlassF_*_sm75` /
`sm80` memory-efficient attention variants, and similar) compile for compute_37 to an
empty body that only calls `vprintf` ("attempting to use an architecture-specific kernel
on the wrong arch") and returns. Translated and run, such a kernel computes nothing, so
`scaled_dot_product_attention`, `nn.MultiheadAttention` and `TransformerEncoderLayer`
silently returned wrong results while the shim reported compute capability 7.5.

## What the shim does now

1. Loud failure. A lazily translated kernel that has parameters but whose body never
   loads, stores (global, shared, generic), uses atomics, barriers, cp.async, mma or
   texture/surface ops (local-memory printf scratch and `ld/st.param` do not count) is
   rejected with CUDA_ERROR_NO_BINARY_FOR_GPU (209) from `cuModuleGetFunction` /
   `cuKernelGetFunction`, with a stderr line naming the kernel.
   `CFI_ALLOW_EMPTY_KERNELS=1` restores the old permissive behaviour.
   PyTorch then raises `CUDA error: no kernel image is available for execution on the device`.
2. Device-level answer. `CFI_REPORT_CC=<major>.<minor>` overrides the compute capability
   attributes (default 7.5). Measured on the Arc B390 with torch 2.2.2+cu118 (default
   SDPA, MultiheadAttention, TransformerEncoderLayer fwd vs CPU):

   | reported cc | result |
   |---|---|
   | 7.5 (default) | loud 209 failure (previously wrong results) |
   | 7.0 | same stub kernel chosen, loud failure |
   | 5.0 | same stub kernel chosen, loud failure |
   | 3.7 | correct: mem-efficient SDPA needs sm50+, PyTorch uses the math path |

   With `CFI_REPORT_CC=3.7`: embedding, index_select, cat, half cast, foreach SGD/Adam,
   transformer forward and all gradients match CPU. fp16 matmul goes through the legacy
   `cublasSgemmEx` path (PyTorch uses it below sm50); the shim now implements it as
   GemmEx with a 32-bit accumulator.
   Alternative that keeps 7.5: `torch.backends.cuda.enable_mem_efficient_sdp(False)` and
   `enable_flash_sdp(False)` in the application.
