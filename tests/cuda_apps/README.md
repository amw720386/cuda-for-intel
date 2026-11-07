# CUDA app smoke tests (PTX-only)

Small CUDA programs that exercise the shim through cudart like real apps:
basic_kernels (vector add, reduction, transposes, shared tiling, tiled matmul),
atomics_warp (histograms, atomics, shuffles, ballot), streams_graphs (streams,
events, stream capture, explicit graphs), cublas_gemm (Sgemm, GemmEx, batched),
cublaslt_bias (cublasLtMatmul with BIAS / RELU_BIAS epilogues).

Kernel programs are built with `-gencode arch=compute_75,code=compute_75` (PTX only).
nvcc 12.6's cudafe++ crashes on MSVC 14.51 headers, so kernel programs use a CUDA 13.x
nvcc (cudart64_13.dll); the host-only cuBLAS programs use CUDA 12.x headers via cl.
`build.bat` builds (set CUDA=<toolkit>), `run.bat` runs under the gpu.lock protocol.
No binaries are committed.
