# CUDA for Intel

Run unmodified CUDA programs on Intel GPUs under Windows. This project is an independent Rust implementation of the CUDA Driver API
(`nvcuda.dll`) plus cuBLAS / cuBLASLt stand-ins. It translates the PTX that CUDA programs carry into SPIR-V and runs it through Level Zero.
It is experimental: read the compatibility matrix and the limitations before relying on it.

```text
CUDA app / PyTorch / llama.cpp / Ollama
        |  CUDA Runtime (genuine NVIDIA cudart)
        v
 nvcuda.dll ----------------- cublas64_12.dll (cuBLAS + cuBLASLt, tiled PTX GEMM, native XMX GEMM)
   Driver API, contexts, streams, graphs
        |
        v
 PTX parser/normaliser -> SPIR-V (OpenCL environment) -> Level Zero -> Intel graphics compiler -> GPU
```

Architecture details: [docs/architecture.md](docs/architecture.md).

## Status and compatibility

Everything in this table was run, and the evidence is linked. "Untested" means nobody has run it, not that it is expected to fail.

### Hardware and OS

| Target | Status | Evidence |
| --- | --- | --- |
| Windows x64, Intel Arc B390 (Xe3, PCI `8086:b080`, driver 32.0.101.9033) | Primary target. Everything below was verified here. | [docs/hw-logs/](docs/hw-logs/README.md) |
| Windows x64, Intel Iris Xe (Alder Lake, PCI `8086:46a8`, driver 32.0.101.7085) | Driver API tests, vector add, reduction, RMSNorm, ggml residual add, genuine CUDA Runtime 13.0.88 kernel; stock Ollama enumerates it as CUDA0. Nothing newer (PyTorch, stock model generation, XMX) has been run on it. | [docs/milestone-12.md](docs/milestone-12.md), [docs/milestone-13.md](docs/milestone-13.md), [docs/agent-handoff.md](docs/agent-handoff.md) |
| Other Intel GPUs (Arc A-series, other Arc B-series, Lunar Lake, Data Center) | Untested. | |
| XMX (DPAS) GEMM path | Needs Xe2/Xe3 with `SPV_INTEL_subgroup_matrix_multiply_accumulate`; verified on Arc B390 only. Other GPUs fall back to the tiled PTX GEMM. | commit `57b9fa5`, `tests/gpu/hw_xmx_gemm.rs` |
| Linux | The Level Zero loader path is included for portability. Untested on hardware. | |
| macOS | Build and library unit tests only (no GPU). | `cargo test --lib`: 226 passed |

### Software

| Software | Status on Arc B390 | Recipe | Evidence |
| --- | --- | --- | --- |
| Stock Ollama v0.12.6 (unchanged `ollama.exe` and `cuda_v13/ggml-cuda.dll`) | Qwen2.5-0.5B-Instruct Q8_0 fully on the GPU, 8 greedy tokens equal to the CPU backend; 10 of 12 prompts identical over 24 tokens. Other models untested. | [docs/ollama.md](docs/ollama.md) | [ollama-stock-v0.12.6.md](docs/hw-logs/ollama-stock-v0.12.6.md), [stock-scan.md](docs/hw-logs/stock-scan.md) |
| llama.cpp (ggml CUDA backend built for `75-virtual`) | `llama-completion` GPU equals CPU on 8 tokens; Qwen-shaped `MUL_MAT` passes (157 of 161 cases); perplexity within 0.11 % of CPU. Flash attention off. | [docs/llama-cpp.md](docs/llama-cpp.md) | [ggml-test-backend-ops.md](docs/hw-logs/ggml-test-backend-ops.md) |
| PyTorch `2.2.2+cu118` Windows wheel (PTX compute_37) | Works: tensor ops, Linear/conv/transformer layer forward and backward, optimizers, autocast fp16, streams and events, equal to CPU. | [docs/pytorch.md](docs/pytorch.md) | [torch-round1.md](docs/hw-logs/torch-round1.md), [torch-bringup.md](docs/torch-bringup.md) |
| PyTorch 2.13.0 conda-forge `cuda128` build (PTX compute_120) | Works: matmul, Linear, softmax, layer_norm, gelu, conv2d forward and backward, SDPA math, autocast fp16, SGD loop. About 6 % of probed kernels still fail to translate. | [docs/pytorch.md](docs/pytorch.md) | [torch-bringup.md](docs/torch-bringup.md), [torch-arch-guard-stubs.md](docs/torch-arch-guard-stubs.md) |
| PyTorch Windows wheels cu121 to cu130 | Cannot run: they contain SASS only, no PTX. | [docs/pytorch.md](docs/pytorch.md) | [torch-round1.md](docs/hw-logs/torch-round1.md) |
| CUDA C sample programs (PTX only, cudart 13.5) | `basic_kernels` 5/5, `atomics_warp` 8/8, `cublas_gemm` 6/6, `cublaslt_bias` 2/2, `streams_graphs` 4/4 (+2 skips). | [docs/cuda-programs.md](docs/cuda-programs.md) | [cuda-apps-round1.md](docs/hw-logs/cuda-apps-round1.md) |
| Rust test suites | 215 library tests; GPU suite 79 of 79 on the final tree of the round-1 log. | [CONTRIBUTING.md](CONTRIBUTING.md) | [test-gpu.log](docs/hw-logs/test-gpu.log), [test-unit.log](docs/hw-logs/test-unit.log) |

What we tried: PyTorch Windows wheels newer than cu118 carry no PTX, so a translator cannot run them; the cu118 wheel and the conda CUDA 12.8 build do.

## Known limitations

- **PTX only.** Programs and libraries must embed PTX. SASS/cubin images cannot run. Build with `-arch=compute_75 -code=compute_75`.
- **Third-party companion DLL.** Programs that use the genuine NVIDIA CUDA Runtime (stock Ollama, CUDA 12/13 samples, PyTorch) need NVIDIA's signed
  `nvcudart_hybrid64.dll` in `CUDA_FOR_INTEL_HYBRID_DIR`. It is proprietary, not distributed here, and you obtain it yourself ([docs/ollama.md](docs/ollama.md)).
- **First run is slow.** Large kernels (quantised matrix multiplication in ggml, big ATen modules) take minutes to compile; later runs use the caches.
- **Speed.** Correctness first: the GPU path is currently slower than the CPU backend for the Qwen2.5-0.5B decode measured in the logs
  (logging off: 30 to 46 tok/s decode on the GPU vs 119 tok/s on the CPU backend, see [docs/hw-logs/perf-matrix.md](docs/hw-logs/perf-matrix.md); llama-bench tg128 18.3 tok/s with large variance). Do not expect native-Vulkan speed.
- **Flash attention** kernels (`flash_attn_ext_*`) do not all translate (87 entries in the stock Ollama scan); disable flash attention.
- **No cuDNN, NCCL, cuFFT, cuSOLVER.** Convolutions use PyTorch's native kernels (`torch.backends.cudnn.enabled = False`).
- **Not implemented:** SASS execution, VMM (`cuMemCreate` family reports no capability), multi-GPU peer access, conditional and memory-allocation graph nodes,
  texture/surface instructions.
- **Numerics.** GPU and CPU outputs can differ at near-ties (2 of 12 greedy prompts diverged); Q8_0 perplexity is 0.11 % below CPU in one run, unexplained.
- **Architecture-guard stubs.** PTX built for another architecture can contain empty stub kernels; the runtime reports error 209 instead of computing nothing
  (see Troubleshooting).
- Systems with two Intel GPUs are untested (both logged hosts had one root GPU).

## Build

Requirements: Rust stable (edition 2024, cargo >= 1.85) with the MSVC x64 toolchain on Windows. Cargo builds and the hardware-independent tests need no CUDA toolkit and no GPU.
Running on a GPU needs the current Intel graphics driver (it ships the Level Zero loader).

```powershell
cargo build --release --locked -p cuda-for-intel -p cublas_shim   # add -p cudart_shim for cudart64_12.dll
cargo test --locked                                              # hardware-independent tests
pwsh -File scripts\check-install.ps1                             # device query + vector add, PASS/FAIL
```

Outputs in `target\release`: `nvcuda.dll` (Driver API), `cublas64_12.dll` (cuBLAS and cuBLASLt entry points in one DLL; copy it under the name the
application imports, for example also as `cublasLt64_12.dll`), `cudart64_12.dll` (optional minimal runtime).

Guided setup (builds, stages, verifies hashes of the third-party files you supply, writes an environment script):

```powershell
pwsh -File scripts\setup-windows.ps1 -TargetDir C:\cfi -HybridDll <path\to\nvcudart_hybrid64.dll> [-OllamaDir <extracted ollama v0.12.6>]
pwsh -File scripts\run-ollama.ps1 -TargetDir C:\cfi
```

## Environment variables

None is required except `CUDA_FOR_INTEL_HYBRID_DIR` for programs using the genuine NVIDIA CUDA Runtime. Boolean switches are on when set to `1`
unless the description says otherwise. Defaults are taken from the source at this revision.

### Runtime and diagnostics

| Variable | Default | Meaning |
| --- | --- | --- |
| `CUDA_FOR_INTEL_HYBRID_DIR` | unset | Absolute directory containing `nvcudart_hybrid64.dll`. Needed by the genuine CUDA Runtime. |
| `CUDA_MODULE_LOADING` | (NVIDIA's) | Set `LAZY` for PyTorch builds. |
| `CUDA_FOR_INTEL_LOG` | off | `1`: print driver diagnostics (translation and Level Zero errors, kernel private/spill/SLM sizes) to stderr and write `cuda-for-intel-export.log` in the working directory. |
| `CUDA_FOR_INTEL_TRACE` | off | `1`: log every API result, resolver request and Level Zero failure to stderr. |
| `CUDA_FOR_INTEL_TRACE_ENTER` | off | Set (with `TRACE=1`): also log each API entry. |
| `CUDA_FOR_INTEL_EXPORT_LOG` | off | `1`: write the bring-up log file without the other diagnostics. |
| `CUDA_FOR_INTEL_PROFILE` | off | `1`: per-API call counts and wall time, appended to the export log about every two seconds. |
| `CUDA_FOR_INTEL_PROFILE_KERNELS` | off | `1`: per-kernel launch counts and time. |
| `CUDA_FOR_INTEL_SYNC_LAUNCH` | off | Set (not `0`): wait after every launch so a device fault is attributed to the kernel that caused it. Slow; debugging only. |
| `CUDA_FOR_INTEL_DUMP_LAUNCH` | off | Set: dump launch arguments (bring-up). |
| `CFI_LOG` | off | Non-empty and not `0`: enables log lines of the cudart and cuBLAS shim DLLs. |
| `CUDA_FOR_INTEL_DUMP_PTX` | off | Set: `cudart64_12.dll` shim writes each resolved PTX to `cfi_dump_<entry>.ptx`. |
| `CFI_PTX_DUMP_DIR` | unset | Directory receiving `raw_<hash>.ptx` (as supplied), `module_<handle>.ptx` (prepared), failing PTX and per-kernel `.spv`. |
| `CFI_DUMP_PTX_DIR` | unset | Directory receiving every rejected PTX module (`reject-<hash>.ptx`) and the SPIR-V of modules the driver rejects. |
| `CFI_DUMP_ENTRY` | unset | Substring; with `CFI_DUMP_PTX_DIR`, keeps the normalised PTX of matching library kernels. |

### Behaviour and compatibility

| Variable | Default | Meaning |
| --- | --- | --- |
| `CFI_REPORT_CC` | per process: 3.7 for torch cu118, 12.0 for torch CUDA 12 builds, else 7.5 | `major.minor` compute capability reported to applications; always wins over the automatic profile. |
| `CFI_ALLOW_EMPTY_KERNELS` | off | Load architecture-guard stub kernels instead of failing with error 209. Silently wrong results; do not use for real work. |
| `CFI_NO_ALLOC_TOUCH` | off | Skip the zero-fill of fresh `cuMemAlloc` buffers. Without the fill, a kernel's first touch of an untouched Level Zero allocation can lose the device on Windows. |
| `CFI_STRICT_BARRIERS` | off | Apply the barrier-convergence check also inside loops (by default tid-dependent loop tests are trusted). |
| `CUDA_FOR_INTEL_DYN_SHARED` | 65536 | Bytes reserved for the dynamic shared window of each kernel, clamped to the device maximum (at least 48 KiB where the device allows). |
| `CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS` | 65536 | Work-groups queued on a stream before the host waits for it to drain; `0` disables the window. |
| `CUDA_FOR_INTEL_LAZY` | on | `0`: translate whole library modules instead of one kernel per module. |
| `CUDA_FOR_INTEL_NO_INDIRECT` | off | Set: do not declare indirect USM access on kernels (by default every kernel declares it, which batched GEMM pointer tables need). |
| `CUDA_FOR_INTEL_BUILD_FLAGS` | empty | Extra options for the Intel compiler, for every module. |
| `CUDA_FOR_INTEL_BIG_FLAGS`, `CUDA_FOR_INTEL_BIG_WORDS` | empty, 400000 | Extra compiler options for modules larger than `BIG_WORDS` SPIR-V words. |
| `CUDA_FOR_INTEL_BATCH_CAP`, `_BATCH_US`, `_BATCH_TIMER` | off, 250, condvar | Only with `--features experimental-batching`: regular command lists of 1 to 1024 commands, flush deadline in microseconds, `high-resolution` for a Windows waitable timer. |
| `CUDA_FOR_INTEL_SUBST` | `none` | Native kernel substitution (`none`, `all` or a list). Exists on the unmerged `kernel-subst` branch only; no effect in this tree. |

### Translator optimisations

| Variable | Default | Meaning |
| --- | --- | --- |
| `CUDA_FOR_INTEL_DP4A` | off | `1`: lower `dp4a` to the SPIR-V integer dot product (Xe DP4A) after the driver accepts a probe module. Opt-in until a Q6_K mismatch is root-caused. |
| `CFI_DP4A_MODE` | `packed` | Dot-product encoding for bring-up: `packed`, `signed` or `vec`. |
| `CFI_DP4A_CONST` | off | `1`: also use the dot product when an operand is a compile-time constant (the Intel compiler miscompiled that case). |
| `CFI_OPT_LD16` | off | `1`: merge pairs of 16-bit loads into 32-bit loads (log: +35 % decode on Qwen2.5-0.5B Q8_0; opt-in because of an unexplained batched-GEMM launch failure seen earlier). |
| `CFI_OPT_SHFL` | on | `0`: use the general computed-lane lowering for constant butterfly shuffles. |
| `CFI_FLOW_DEBUG` | off | Set: print why a register was classed thread-dependent (barrier analysis). |

### Caches

| Variable | Default | Meaning |
| --- | --- | --- |
| `CFI_NO_CACHE` | off | Set: disable the PTX-to-SPIR-V translation cache. |
| `CFI_CACHE_DIR` | `%LOCALAPPDATA%\cuda-for-intel\spvcache` (Linux: `$XDG_CACHE_HOME` or `~/.cache`) | Location of the translation cache. Keyed by PTX hash, entry, versions and the `CFI_*` switches. |
| `CUDA_FOR_INTEL_CACHE` | on | `0`: disable the native (compiled module) binary cache. |
| `CUDA_FOR_INTEL_CACHE_DIR` | `%LOCALAPPDATA%\cuda-for-intel\cache\v1` (Linux: `$XDG_CACHE_HOME` or `~/.cache`) | Location of the native binary cache. |
| `CUDA_FOR_INTEL_CACHE_MAX_MB` | 4096 | Size cap of the native binary cache. |

### cuBLAS / cuBLASLt shim

| Variable | Default | Meaning |
| --- | --- | --- |
| `CFI_XMX` | auto | `0`: never use the native XMX GEMM. Unset: use it only for large fp16/bf16 GEMMs (see below). `2`: force it for every qualifying call. |
| `CFI_XMX_MIN_DIM` | 128 | Auto mode: minimum of `min(m, n)`. |
| `CFI_XMX_MIN_WORK` | 16777216 | Auto mode: minimum `m*n*k` per matrix. |
| `CFI_XMX_TILE` | `2x4` | DPAS tiles per subgroup (`IxJ`, I 1 to 4, J 1 to 8). |
| `CFI_XMX_TRACE` | off | Trace XMX qualification and launches. |
| `CFI_CUBLAS_ASYNC` | off | Do not synchronise the stream after each GEMM. |
| `CFI_CUBLAS_ONEMKL` | unset | Path to an optional library exporting `cfi_onemkl_gemm`, offered each GEMM first. |
| `CFI_CUBLAS_TRACE` | off | Trace cuBLASLt plan failures. |

### Tests only

`CUDA_FOR_INTEL_DEVICE` (Level Zero ordinal for GPU tests), `CUDA_FOR_INTEL_BENCH_*`, `CFI_STOCK_GGML_DLL` (Ollama `ggml-cuda.dll` for the scan and compile tests),
`CFI_SCAN_DUMP_DIR`, `CFI_PTX_FILE`, `CFI_PTX_ENTRY`, `CFI_PTX_PREPARED_OUT`, `CFI_STRESS_LAUNCHES`, `CFI_MIN_AVAILABLE_MB`. See the header of each test file.

## Troubleshooting

Set `CUDA_FOR_INTEL_TRACE=1` and `CUDA_FOR_INTEL_LOG=1` first; most failures name the call and the Level Zero status.

| Symptom | Cause and fix |
| --- | --- |
| `CUDA_ERROR_NO_BINARY_FOR_GPU` (209), "no kernel image is available for execution on the device" | The kernel is an architecture-guard stub (empty body compiled for another architecture), or the binary has only SASS. Rebuild with PTX (`-arch=compute_75 -code=compute_75`). For PyTorch set `CFI_REPORT_CC` (3.7 for cu118, 12.0 for cuda128) or disable the memory-efficient and flash SDP backends. |
| `CUDA_ERROR_INVALID_PTX` (218) or a "PTX translation rejected" line | The translator does not support a construct in that kernel. The message names it; dump with `CFI_DUMP_PTX_DIR`, check [docs/ptx-gap-report.md](docs/ptx-gap-report.md), open an issue with the PTX. |
| `CUDA_ERROR_FILE_NOT_FOUND` (301) at start-up of a runtime-using program | `CUDA_FOR_INTEL_HYBRID_DIR` is not set. Error 1 there means the path is not absolute or the file name is wrong. |
| Crash `0xc0000005` in a CUDA plug-in during registration | Wrong or missing companion DLL; verify the SHA-256 (`docs/ollama.md`). |
| `CUDA_ERROR_NOT_INITIALIZED` (3) / `NO_DEVICE` (100) | No Level Zero device or loader. Install the Intel graphics driver, check Device Manager, run `cargo run --release --example devices`. Remote desktop sessions and VMs without GPU passthrough expose no GPU. |
| `CUDA_ERROR_LAUNCH_FAILED` (719), "device lost", error 600/719 on synchronisation | The Level Zero device was lost or needs a reset (the driver reports device-lost and requires-reset as 719). On Windows this is usually a TDR: a single command took too long. Close other GPU-heavy programs; set `CUDA_FOR_INTEL_SYNC_LAUNCH=1` to find the kernel; reduce work per launch; `CUDA_FOR_INTEL_MAX_INFLIGHT_GROUPS` bounds queued work (default 65536 work-groups). If the first touch of fresh memory kills the device, make sure `CFI_NO_ALLOC_TOUCH` is not set. |
| `CUDA_ERROR_OUT_OF_MEMORY` (2) | Device or host allocation failed. Integrated GPUs share system RAM: `cuMemGetInfo` reports capacity minus this driver's allocations, bounded by available system memory minus a reserve. Use a smaller model or context, close other programs. |
| `CUDA_ERROR_NOT_SUPPORTED` (801) | A feature the Intel driver or this project lacks (for example VMM or an unimplemented cuBLAS name). |
| `CUBLAS_STATUS_NOT_INITIALIZED` | Call `cudaFree(0)` (or create a context) before `cublasCreate`; the shim needs a current context. |
| `CUBLAS_STATUS_EXECUTION_FAILED` on batched fp16 GQA GEMM | Seen in an early llama.cpp run; the shapes pass in `hw_cublas_batched`. Update to a current build. |
| Hangs for minutes on the first request | Compilation of large kernels by the Intel compiler. Wait, keep the caches (`CUDA_FOR_INTEL_CACHE_DIR`). |
| `import torch` fails with an unresolved import | The wheel imports cuBLAS names the shim lacks; regenerate stubs with `scripts/torch/gen_cublas_stubs.py` ([docs/pytorch.md](docs/pytorch.md)). |
| `cuLibraryUnload 999` at exit | Harmless: Level Zero is already torn down. |

## Performance guidance

- Warm the caches: the first run of any new model or program pays kernel compilation. Keep `CUDA_FOR_INTEL_CACHE_DIR` and `CFI_CACHE_DIR` on a persistent disk.
- XMX GEMM is on by default but engages only for large fp16/bf16 GEMMs (`min(m,n) >= 128`, `m*n*k >= 2^24`); small or skinny shapes were slower with XMX.
  Tune with `CFI_XMX_MIN_DIM`, `CFI_XMX_MIN_WORK`, `CFI_XMX_TILE`, or switch off with `CFI_XMX=0` to compare.
- `CFI_OPT_LD16=1` merged 16-bit load pairs and measured +35 % decode on Qwen2.5-0.5B Q8_0, but is opt-in; test your workload before enabling it.
- Leave `CUDA_FOR_INTEL_LOG`, `_TRACE`, `_PROFILE` and `_SYNC_LAUNCH` off for measurements; they add syscalls per call or serialise launches.
- `CFI_CUBLAS_ASYNC=1` removes the stream synchronisation after each GEMM; verify your results before relying on it.
- Quantised ggml models: Q8_0 is the format verified end to end; Q4_0 and Q4_K matrix multiplications pass `test-backend-ops` (the Q4_K_M perplexity run was not recorded).
- Measure against the CPU backend of the same program: for small models the CPU is currently faster.

## Documentation

| Topic | File |
| --- | --- |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Stock Ollama | [docs/ollama.md](docs/ollama.md) |
| llama.cpp | [docs/llama-cpp.md](docs/llama-cpp.md) |
| PyTorch | [docs/pytorch.md](docs/pytorch.md) |
| CUDA C programs | [docs/cuda-programs.md](docs/cuda-programs.md) |
| Contributing, tests, GPU lock | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Hardware logs | [docs/hw-logs/](docs/hw-logs/README.md) |
| Development history | [milestone 12](docs/milestone-12.md), [13](docs/milestone-13.md), [14](docs/milestone-14.md), [agent handoff](docs/agent-handoff.md), [PTX gap report](docs/ptx-gap-report.md) |

## References and license

MIT licensed. The implementation draws architectural inspiration from
[Intel-era ZLUDA](https://github.com/em-winterschon/ZLUDA-CUDA-on-Intel-GPU),
its [upstream v1 snapshot](https://github.com/vosen/ZLUDA/tree/v1),
[modern ZLUDA](https://github.com/vosen/ZLUDA), and
[CUDA for AMD on Windows](https://github.com/Speedstu/CUDA-for-AMD-Windows).
The Level Zero ABI declarations are attributed in [LICENSE](LICENSE).

It is built on the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/),
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/),
[OpenCL SPIR-V environment](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_Env.html),
and the [Level Zero core specification](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html).
[Intel Compute Runtime](https://github.com/intel/compute-runtime) and
[Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler)
describe the driver and compiler stack.
NVIDIA, CUDA, cuBLAS and cuDNN are trademarks of NVIDIA Corporation; this project is not affiliated with or endorsed by NVIDIA or Intel.
