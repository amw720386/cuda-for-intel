# PyTorch on the shim, hardware round 1 (Arc B390, Windows)

Wheel torch 2.14.1+cu126 (CPython 3.12), stand-in nvcuda.dll preloaded with
ctypes, cublas/cublasLt shim installed under both names in torch/lib.

- `import torch` needs every cublas/cublasLt name cusolver/torch_cuda import:
  fixed with generated NOT_SUPPORTED stubs (scripts/torch/gen_cublas_stubs.py).
- cudart 12.6 integrity check reports version 12060: digest scheme is chosen by
  version % 10 (ZLUDA dark_api), so the fixed digest is returned.
- `torch.cuda.is_available()` -> True; `torch._C._cuda_init()` passes once
  cuModuleGetLoadingMode reports lazy loading (otherwise cudart loads every
  registered fatbin at context creation).
- `torch.zeros(4, device='cuda')` FAILS: cuLibraryLoadData on the fill-kernel
  fatbin (1.7 MB). The fatbin has SASS records only (kind=2, arch 50..90), no PTX.
- Scan of torch_cuda.dll in the cu126 (2.14.1), cu128 (2.11.0) and cu130 wheels:
  every record is SASS (kind 2), zero PTX records. Stock Windows wheels cannot run
  through a PTX translator.
- cublasLt branch checks: `translates_generated_cublas*` pass after fixing an
  in-place cvta in the trsm generator; `gpu_gemm_matches_cpu_reference` and
  `gpu_cublaslt_and_strsm_match_cpu_reference` pass on hardware.

## Round 2: torch 2.2.2+cu118 (PTX sm_37, ISA 7.8), Arc B390

Setup: new venv, wheel from download.pytorch.org/whl/cu118, our nvcuda.dll preloaded
with ctypes, the shim DLL installed as cublas64_11.dll and cublasLt64_11.dll,
CUDA_FOR_INTEL_HYBRID_DIR set (cudart 11.8 integrity version 11080 passes via
version % 10). cudart 11.x loads fatbins through CUDART_INTERFACE slots 1/6/8, not
cuLibrary*; these are now served and register a lazy module (one kernel translated
per cuModuleGetFunction).

Working on hardware, results equal to CPU:
- is_available, torch.zeros, x+y
- x@y fp32 in all four transpose layouts, bmm (cublas sgemm shim)
- nn.Linear forward and backward (dW, db, dx), relu, gelu, softmax, layer_norm, sum
- conv2d forward (cudnn disabled, native kernel), max_pool2d
- 3-step SGD training of Linear-ReLU-Linear with cross_entropy (foreach=False)

Required runtime fixes beyond translator work: a kernel's first touch of a never
copied Level Zero device allocation loses the device (DEVICE_LOST), so cuMemAlloc
now zero-fills (CFI_NO_ALLOC_TOUCH=1 disables).

Open gaps (translator): foreach optimizer kernels index kernel-parameter space with
a computed address (ld.param.u8 [%rd+1536]); index_select / embedding
(undefined %rd12 on an incoming path), CatArrayBatchedCopy (invalid PTX parameter
offset), a few unrolled copy kernels (expected B32 for %rd95, got B64),
fp16 matmul/copy variants (cvt.u16.u64, cvt.s64.s16, setp.eq.b64), register-count
shifts (only immediate left shifts below 64), divergent bar.sync in softmax_block.
Reproduce per-kernel status with the harness: dump modules via CFI_PTX_DUMP_DIR and run
`ptx-corpus-harness --probe module_N.ptx`.

## Round 3 (torch 2.2.2+cu118 with the integration-torch translator fixes, and conda-forge torch 2.13 cuda128)

cu118, Arc B390, all equal to CPU: embedding, index_select, torch.cat, fp16 cast
roundtrip, autocast fp16 matmul, bmm, conv2d backward, default (foreach) SGD and Adam,
TransformerEncoderLayer forward and all 12 gradients, fp16 transformer forward,
rand/randn/dropout (ATen Philox, no curand shim needed), Events, Streams,
pinned non_blocking copies, memory_stats.

Caveat: scaled_dot_product_attention default picks fmha_cutlassF_*_sm75, whose
compute_37 PTX is an arch-guard stub (empty body), giving wrong attention output.
Use torch.backends.cuda.enable_mem_efficient_sdp(False) (and flash off).

cuda128 (conda-forge torch 2.13.0, PTX sm_120, ISA 8.7, cudart 12.8): is_available,
torch.zeros and x+y run. cudart 12.8 loads through cuLibrary*; library kernels are
now translated lazily per entry (translating the whole module on first use did not
finish in two minutes). Shutdown logs cuLibraryUnload 999 (zeKernelDestroy after
Level Zero teardown), harmless.
