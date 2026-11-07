# Stock Ollama v0.12.6 on Arc B390 through CUDA for Intel

Result: unchanged `ollama.exe` and unchanged `cuda_v13/ggml-cuda.dll` (SHA-256 4328F98A...38E1A)
loaded Qwen2.5-0.5B-Instruct Q8_0 (model SHA-256 CA59CA7F...6844E) fully on the GPU through this
driver and generated 8 tokens. Prompt "Say hi." (raw, no template), temperature 0, num_batch 1,
num_ctx 128, num_gpu 99, flash attention off:

    " I'm a 17 year old"   (token ids 358 2776 264 220 16 22 1042 2310)

The identical request with num_gpu 0 (CPU backend) produced the identical text. About 10,000
successful `cuLaunchKernel` calls were traced in the GPU run. Timing: 12.1 s load (includes
JIT of the large Q8 MMQ kernel by the Intel compiler, minutes on first use), 8 tokens in 1.06 s.

## Staging
- Binaries: ollama-windows-amd64.zip v0.12.6; signed nvcudart_hybrid64.dll companion (SHA-256
  BF430D65...22B3) from ZLUDA commit f4df215; Qwen GGUF from Hugging Face. None are in the repo.
- `cuda_v13/cublas64_13.dll` replaced by the built cublas shim, `nvcuda.dll` placed beside the plug-in,
  `cublasLt64_13.dll` removed, original cublas renamed.
- Env: CUDA_FOR_INTEL_HYBRID_DIR, OLLAMA_LIBRARY_PATH, OLLAMA_VULKAN=0, OLLAMA_FLASH_ATTENTION=0,
  OLLAMA_NUM_PARALLEL=1, OLLAMA_LOAD_TIMEOUT=60m, CUDA_FOR_INTEL_TRACE=1 (traces are opt-in now),
  CFI_DUMP_PTX_DIR to keep rejected PTX. Served with `ollama serve`, model created from the GGUF,
  `/api/generate` with the options above.

## Blockers hit and fixed (in order)
1. ldmatrix without `.shared` (generic address) rejected.
2. `bar.sync` divergence check treated every load as thread-varying; uniform-address loads are now uniform.
3. `.extern .shared` symbols (`data_mul_mat_q`, `ids_dst_shared`) must alias the dynamic block after static arrays.
4. Per-entry register cap 4096 < 7000 registers of the MMQ kernel; raised.
5. `cudaFuncSetAttribute` on CUkernel calls `cuKernelSetAttribute` (not implemented); added Get/Set attribute and cache config.
6. One unsupported kernel in a fatbin library module blocked every kernel; libraries now translate leniently per entry.
7. cublas shim synchronised a capturing stream (error 900 in GemmBatchedEx during graph reservation).
8. Test-harness only: trsm PTX generator reused one register as raw u64 and global pointer.

## Known remaining gaps
`k_set_rows` for bf16, q5_0 and iq4_nl (`setp.b16`, `shl.b16`, float literal as integer operand) fail
to translate and are dropped; not needed by this model. Multi-prompt, longer generation and numerical
comparison beyond one 8-token greedy check are not done. First MMQ JIT takes minutes.
