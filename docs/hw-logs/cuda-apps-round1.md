# CUDA app smoke tests, round 1 (Arc B390, Windows)

Source: tests/cuda_apps (PTX-only, compute_75). Source torch build abandoned (PTX-bearing
wheels found instead); toolchain pieces kept on the box under the torchbuild work dir.

- nvcc 12.6 + MSVC 14.51 (VS 18): cudafe++ dies with ACCESS_VIOLATION on every .cu. VS cannot be
  modified (not admin). Used nvcc 13.5 redist for kernel programs; cublas programs are host-only,
  built with cl against 12.6 headers. All 5 programs build; cuobjdump shows PTX only, no ELF.
- Run under gpu.lock with the shim nvcuda.dll (+cublas64_12.dll as both cublas names) next to each exe:
  - basic_kernels / atomics_warp / streams_graphs (cudart 13): first runtime call fails
    (cudaGetDeviceCount / cudaMalloc / cudaMallocHost return an error with null string). cudart 13
    init path not handled by the shim yet (round1 notes only cover cudart 12.6 integrity check and
    cuModuleGetLoadingMode).
  - cublas_gemm: cublasCreate status=1 (NOT_INITIALIZED); cublaslt_bias: cudaMalloc (cudart 12.6)
    -> operation not supported.
  Zero tests reached PASS/FAIL checks; failures are at cudart/shim init, not in kernels.
