
## cuDNN shim

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

## Support matrix and errors

| Case | Behaviour |
| --- | --- |
| 2-D convolution, forward / backward data / backward filter, fp32, fp16, bf16; NCHW and NHWC (any strides); stride, dilation, groups, padding; cross-correlation and true convolution | Runs (GEMM engine or direct kernels). Descriptors may be set with the 4-D or the Nd entry points. |
| 3-D convolution, fp64, int8, mixed element types across x / w / y, `NCHW_VECT_C` | `CUDNN_STATUS_NOT_SUPPORTED` from the conv calls, the `*Algorithm_v7` / `Find*` queries (count 0) and the workspace-size queries. |
| Inconsistent descriptors (output extent, channel mismatch, unset descriptors, null handle or operand) | `CUDNN_STATUS_BAD_PARAM`, never a crash. |
| RNN, dropout states, CTC, softmax, pooling and the remaining cuDNN API | `CUDNN_STATUS_NOT_SUPPORTED` stubs. |

PyTorch raises on NOT_SUPPORTED instead of falling back, so `scripts/torch/cfi_torch.py` (`cfi_torch.install()`) wraps the
functional convolution entry points and the RNN/CTC modules and runs the unsupported families with cuDNN disabled, forward and backward.
`CFI_CUDNN=0` disables the shim path altogether.

Workspace: the `Get*WorkspaceSize` calls advertise the column-buffer size (capped by `CFI_CUDNN_WS_MB`, default 256 MB); callers that
pass less, or none, get chunked execution or an internal scratch buffer, never an error.

Tests: `tests/gpu/hw_cudnn.rs` (descriptor variants, kernel sizes 1/3/5/7, bf16, Nd entry points, NOT_SUPPORTED and BAD_PARAM probes),
`scripts/cudnn/test_callers.py` (PyTorch-level checks against CPU fp64), `scripts/cudnn/gate_final.ps1` (the gate in one queue slot).
