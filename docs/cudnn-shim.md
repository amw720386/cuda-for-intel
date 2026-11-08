
## Convolution engine (branch cudnn-fast)

Forward, backward data and backward filter run an explicit-GEMM algorithm when a workspace is
supplied and the layouts allow it: a PTX `im2col` kernel (any input strides, groups, dilation,
filters `[k][c][r][s]` or `[k][r][s][c]`) fills a column buffer in the workspace, then
`cublasGemmEx` / `cublasGemmStridedBatchedEx` of the loaded cuBLAS shim (so XMX/DPAS applies)
produces the output; backward data uses a gather `col2im` kernel with the alpha/beta epilogue. Large
batches are processed in chunks that fit the workspace (default 256 MB, `CFI_CUDNN_WS_MB`).
Planner: `gemmconv.rs` (pure, simulator-tested in `src/ptx/cudnn_tests.rs`).

Fallback to the direct thread-per-output kernels: filter flip, tiny groups (depthwise), unusual
output strides, no workspace, or `CFI_CUDNN_DIRECT=1`. Weight gradients with few outputs (depthwise)
use a work-group-per-element reduction kernel. Workspace sizes are reported by the v7
`Get*WorkspaceSize` calls and by the backend plan workspace attribute.

Benchmarks and checks: `scripts/cudnn/bench_conv.py` (modes `aten`, `direct`, `cudnn`),
`scripts/cudnn/test_resnet18.py`, `scripts/cudnn/run_bench.ps1`.
