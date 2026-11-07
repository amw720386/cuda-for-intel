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
