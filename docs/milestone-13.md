# Milestone 13 — Ollama integration and CUDA kernel execution

## Executive summary

A specialization of GGML's CUDA `k_bin_bcast<op_add, float, float, float>` ran through this project's `nvcuda.dll`, PTX-to-SPIR-V translator, and Intel Level Zero on Iris Xe and Arc B390. Both immediate and opt-in batched command lists passed the FP32 reference and guard checks. This fixture covers the residual-add operation in isolation.

Ollama v0.40.15 generated Qwen2.5 0.5B tokens on the CPU and through the native Vulkan backend on both hosts. The stock CUDA inference attempt on Arc B390 reached CUDA plugin initialization and fell back to the CPU. No model layer executed through this compatibility runtime.

## Source and distribution pins

| Component | Revision or artifact |
| --- | --- |
| Ollama | v0.40.15, `02480ca2a47153e1f9c31cbe1e05536da97ce1ba` |
| llama.cpp and GGML | Ollama's `LLAMA_CPP_VERSION` specifies b12111, commit `52e53a76df0298d2c316ae7e704eeec03b0bf478`. GGML is bundled in that tree as `ggml/` at tree object `aba4ed214b4b573b4d25afd454666f80c626cee1`. |
| Windows x64 release | `ollama-windows-amd64.zip`, SHA-256 `9ca74a7ab66ec86870b1eecac7df160df08dc4e3a291ab5d2d59bcf9310221c6` |
| Local model | `qwen2.5:0.5b`, GGUF Q4_K_M, 494.03M parameters. Manifest digest `a8b0c51577010a279d933d14c2a8ab4b268079d44c5c8830c0a93900f1827c67`. |

Ollama's CMake fetches that llama.cpp pin, applies its GGUF/model compatibility overlay, and builds the external `llama-server` and GGML modules. Ollama's `discover` package enumerates runner directories, launches the server with `--list-devices` and combines its device and compiled-architecture output with a native probe. The Windows Go CUDA probe searches System32 explicitly and uses the CUDA driver version, PCI ID and device attributes. The runner filters candidates against compiled CUDA architectures and configured visibility before scheduling, then launches `llama-server` with the selected backend directories. GGML's `ggml_backend_load_all_from_path` loads plugins and resolves the backend registration symbol. `ggml_backend_cuda_reg` leads to `ggml_cuda_init`, which uses the CUDA Runtime to count and characterize devices before buffer allocation or graph execution. [Ollama dependency pin](https://github.com/ollama/ollama/blob/02480ca2a47153e1f9c31cbe1e05536da97ce1ba/LLAMA_CPP_VERSION) · [Runner build](https://github.com/ollama/ollama/blob/02480ca2a47153e1f9c31cbe1e05536da97ce1ba/llama/server/CMakeLists.txt) · [Windows native discovery](https://github.com/ollama/ollama/blob/02480ca2a47153e1f9c31cbe1e05536da97ce1ba/discover/native_probe_windows.go) · [Runner discovery and filtering](https://github.com/ollama/ollama/blob/02480ca2a47153e1f9c31cbe1e05536da97ce1ba/discover/llama_server.go) · [GGML loader](https://github.com/ggml-org/llama.cpp/blob/52e53a76df0298d2c316ae7e704eeec03b0bf478/ggml/src/ggml-backend-reg.cpp)

### Windows CUDA build and code objects

The Windows build runs separate CUDA 12 and CUDA 13 presets and ships `ggml-cuda.dll` with matching dependencies under `lib/ollama/cuda_v12` and `cuda_v13`. CUDA 12 presets specify virtual-only 50, 52, 60, and 61 plus real and virtual 70, 75, 80, 86, 89, 90, 90a, and 120. CUDA 13 specifies virtual-only 75, 80, 86, 89, 100 and 120. The shipped CUDA 13 DLL's PE imports include `cublas64_13.dll` and omit an ordinary cudart DLL import. Its embedded fatbin and the CUDA calls in GGML still require CUDA Runtime behavior. CMake supports static or shared cudart depending on `GGML_STATIC` and the platform. [Ollama Windows build](https://github.com/ollama/ollama/blob/02480ca2a47153e1f9c31cbe1e05536da97ce1ba/scripts/build_windows.ps1) · [CUDA presets](https://github.com/ollama/ollama/blob/02480ca2a47153e1f9c31cbe1e05536da97ce1ba/llama/server/CMakePresets.json) · [GGML CUDA link rules](https://github.com/ggml-org/llama.cpp/blob/52e53a76df0298d2c316ae7e704eeec03b0bf478/ggml/src/ggml-cuda/CMakeLists.txt)

NVIDIA `cuobjdump` 13.2 enumerated 1,764 PTX entries and 1,176 ELF cubins in the CUDA 12 DLL. Its inspected targets match the preset families, with `sm_120a` as the emitted target for the 120 family. The same tool enumerated 882 PTX entries in the shipped CUDA 13 DLL, with targets `sm_75`, `sm_80`, `sm_86`, `sm_89`, `sm_100`, and `sm_120a`. It found zero ELF cubins with `-all -lelf`. The extracted first `sm_75` program uses PTX 9.0 and 64-bit addressing. That PTX is embedded in fatbins inside the PE's `.nv_fatb` section. The runtime currently loads one entry of plain PTX 4.1 per module. CUDA 13's virtual-only release therefore still requires fatbin registration, code selection, and support for newer PTX and multi-entry modules. The fixture uses `sm_52` to get PTX 4.1 out of Clang. That code-generation target describes the fixture's input format and supplies no claim about an Intel GPU's NVIDIA compute capability. [Ollama release](https://github.com/ollama/ollama/releases/tag/v0.40.15) · [NVIDIA CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html)

## Compatibility matrix

The Driver API rows distinguish calls in the source from CUDA Runtime internals. Internal NVIDIA dispatch behavior was not enumerated from proprietary implementation code.

| Dependency and source | Status | Implemented behavior and remaining work |
| --- | --- | --- |
| Driver enumeration in Ollama `probeCUDADriverWindows` | Supported | `cuInit`, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem_v2`. These enumerate the selected Intel Level Zero devices. |
| Driver identity and loading in the same probe | Partially supported | `cuDeviceGetAttribute` covers launch and shared-memory limits. `cuDriverGetVersion`, `cuDeviceGetPCIBusId`, compute capability and integrated attributes are missing. The source's System32 search does not select a private project DLL. |
| Runtime bootstrap in `ggml_cuda_init`, `ggml_cuda_set_device` | Unsupported | Missing CUDA Runtime ABI for `cudaGetDeviceCount`, `cudaGetDeviceProperties`, `cudaGetDevice`, `cudaSetDevice`, `cudaDeviceGetAttribute`, `cudaDeviceGetPCIBusId`, `cudaMemGetInfo`, and error APIs. No primary-context management or `cuGetProcAddress` dispatch for an unmodified runtime. |
| Allocation, transfer and buffer setup in `ggml-cuda.cu` | Partially supported | Driver equivalents exist for basic device and pinned allocation, HtoD and DtoH and their pinned asynchronous forms. GGML uses `cudaMalloc`, `cudaFree`, `cudaMallocHost` / `cudaFreeHost`, host register/map, `cudaMemset[Async]`, `cudaMemcpyAsync` including device copies, and `cudaMemcpy2DAsync`. Missing Runtime entry points, general device-to-device copies, 2D copies, device fills, registration, and mapped-host semantics. Managed memory/advice are optional paths. |
| CUDA VMM in `ggml_cuda_pool_vmm` | Unsupported | With VMM enabled, direct driver calls include `cuDeviceGet`, `cuDeviceGetAttribute`, `cuGetErrorString`, `cuMemGetAllocationGranularity`, `cuMemAddressReserve/Free`, `cuMemCreate/Release`, `cuMemMap/Unmap`, and `cuMemSetAccess`. The runtime lacks virtual reservations, mappings and access descriptors. GGML can build with `GGML_CUDA_NO_VMM` and use its ordinary allocation pool. |
| Streams and events in `ggml-cuda.cu` and `common.cuh` | Partially supported | Driver streams, event generations and waits exist. Missing Runtime stream/event entry points, host callbacks and capture. Event timing and extra flags are outside the Driver API subset. |
| Runtime launch and code registration in CUDA-compiled GGML | Unsupported | Normal kernel launches, `cudaLaunchKernelEx`, function attributes, occupancy queries and CUDA's generated fatbin/function registration need runtime handling. `cuModuleLoadData` accepts plain single-entry PTX 4.1 and `cuLaunchKernel` accepts pointer-array arguments with static shared memory. Packed parameters, code-object selection, and dynamic shared memory are missing. |
| Graph capture and programmatic dependent launch | Unsupported | GGML uses `cudaStreamBeginCapture/EndCapture`, `cudaGraphInstantiate`, `cudaGraphExecUpdate`, `cudaGraphLaunch`, destruction and capture-state queries when enabled. Newer CUDA paths use programmatic launch completion and dependency synchronization. Disable these optional paths in an initial compatibility build. |
| Dense and batched BLAS in `ggml-cuda.cu` and `common.cuh` | Unsupported | No cuBLAS ABI, handle lifetime, stream binding, math mode, workspace, `cublasSgemm`, `cublasGemmEx`, or the batched entry points `cublasGemmStridedBatchedEx`, `cublasGemmBatchedEx`, `cublasSgemmStridedBatched`, and `cublasSgemmBatched`. `solve_tri.cu` uses `cublasStrsmBatched`, outside the initial Qwen path. |
| `binbcast.cu` residual add | Partially supported | The extracted FP32 two-dimensional, nonbroadcast specialization is validated. Full `uint3` aggregate arguments, fast division/modulo, general broadcasts, casts, nullable input and fused variants remain. |
| `norm.cu` RMSNorm and `common.cuh` reductions | Partially supported | Scalar FP32 arithmetic, static shared memory and workgroup barriers exist. Actual GGML RMSNorm uses vector loads, a warp shuffle reduction, and reciprocal square root. No native GGML RMSNorm has been validated here. |
| `convert.cu`, `mmvf.cu`, `mmvq.cu`, `mmq.cu`, attention and RoPE | Unsupported | FP16/BF16 conversion and arithmetic, packed/vector accesses, CUDA warp collectives, quantized bit operations, integer dot products and architecture-dependent MMA remain unvalidated or unsupported. These matter even for Q4_K_M, whose quantized weights still require dequantization, projection, normalization and attention. |
| Multi-GPU split buffers and peer access | Not applicable | Initial target is a single Intel GPU. GGML's peer capability, enablement and `cudaMemcpyPeerAsync` paths remain outside the scope. |
| Ollama Vulkan plugin | Not applicable | Separate native inference backend, measured as a baseline. |

[CUDA initialization, allocation, graphs, and dispatch](https://github.com/ggml-org/llama.cpp/blob/52e53a76df0298d2c316ae7e704eeec03b0bf478/ggml/src/ggml-cuda/ggml-cuda.cu) · [CUDA helpers](https://github.com/ggml-org/llama.cpp/blob/52e53a76df0298d2c316ae7e704eeec03b0bf478/ggml/src/ggml-cuda/common.cuh) · [GGML kernel sources](https://github.com/ggml-org/llama.cpp/tree/52e53a76df0298d2c316ae7e704eeec03b0bf478/ggml/src/ggml-cuda)

The following Runtime and cuBLAS call inventory includes conditional paths in the CUDA source tree. Actual calls on a supported GPU depend on the build and dispatch decisions.

| Area | Additional literal calls and grouped variants |
| --- | --- |
| Error and device control | `cudaGetErrorString`, `cudaGetLastError`, `cudaDeviceSynchronize`, `cudaSetDeviceFlags`, `cudaDeviceCanAccessPeer`, `cudaDeviceEnablePeerAccess` |
| Allocation and transfers | `cudaMallocManaged`, `cudaMemAdvise`, `cudaHostAlloc`, `cudaHostGetDevicePointer`, `cudaHostRegister`, `cudaHostUnregister`, `cudaMemcpyPeerAsync` |
| Streams, events and graphs | `cudaStreamCreateWithFlags`, `cudaStreamDestroy`, `cudaStreamSynchronize`, `cudaStreamWaitEvent`, `cudaStreamIsCapturing`, `cudaEventCreateWithFlags`, `cudaEventDestroy`, `cudaEventRecord`, `cudaEventSynchronize`, `cudaGraphDestroy`, `cudaGraphExecDestroy` |
| Launch, occupancy and dependencies | `cudaFuncGetAttributes`, `cudaFuncSetAttribute`, `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, `cudaLaunchCooperativeKernel`, `cudaLaunchHostFunc`, `cudaTriggerProgrammaticLaunchCompletion`, `cudaGridDependencySynchronize` |
| cuBLAS host control | `cublasCreate`, `cublasDestroy`, `cublasGetStatusString`, `cublasLoggerConfigure`, `cublasSetMathMode`, `cublasSetStream`, `cublasSetWorkspace` |

## Implemented CUDA kernel and verification

[ggml_add_f32.cu](../examples/ggml_add_f32.cu) specializes the pinned `binbcast.cu` kernel. It retains `op_add`, the grid-stride loop, early bounds and overflow checks, and 64-bit row offset calculations with independent strides. The specialization assumes non-null inputs, unit column strides, one channel/sample, equal extents, no fused extra operands, and one work-item in the block's Y and Z dimensions. Under these conditions the broadcast coordinate is the destination coordinate and the channel/sample offsets vanish. The source removes the production host dispatch, CUDA headers, identity casts and architecture-dependent programmatic launch hooks. Clang's NVVM register intrinsics stand in for CUDA built-in indexing. No postprocessing changes the generated PTX. [Pinned upstream kernel](https://github.com/ggml-org/llama.cpp/blob/52e53a76df0298d2c316ae7e704eeec03b0bf478/ggml/src/ggml-cuda/binbcast.cu) · [GGML MIT license](../examples/ggml-LICENSE)

The fixture required unsigned `setp.le.u32`, `and.pred` and `or.pred`, `%nctaid.{x,y,z}` as SPIR-V `NumWorkgroups`, and integer `shl.b64` with an immediate shift below 64. Predicate operands go through existing definite-assignment and divergence analysis. The shift rejects pointer operands and out-of-range counts. A hardware-independent test checks the generated opcodes, ABI and rejected inputs. Existing FP32 and Level Zero capability checks still apply. Immediate lists remain the default.

The DLL test uses six row × column shapes, `1×1`, `3×63`, `2×64`, `5×65`, `7×257`, and `64×896`. The 65-, 257-, and 896-column cases require multiple grid-stride iterations. An extra Y workgroup checks the row guard. Each row has different input and output pitches with padding and outer output guards. The reference computes one independent Rust FP32 addition per element. The finite result, infinities, and signed zero must match bit for bit because a single round-to-nearest FP32 addition has no accumulation error. NaNs must be NaNs, with payload and sign unconstrained. Cases cover subnormal preservation, ties, overflow, invalid infinity combinations, cancellation, and both signed and ordinary zeros.

Each shape runs 72 launches across five warmups and 31 pairs of measured samples. Every downloaded result is checked, including the row padding and both guards. A second context must reject the first context's function handle before the first context resumes execution. The test frees all device allocations, unloads the module and destroys both contexts. The profiling test also verifies its output and tears down the timestamp events and context. This exercises actual native lifecycle paths. The existing fault-ledger tests separately check ownership and leaks without executing GPU arithmetic.

The validation runner checks the selected device's FP32, copy, grid, workgroup, argument and allocation limits before selecting the GGML tests. The profiling case also requires kernel timestamp support. Recompiling the CUDA fixture with the documented Clang invocation reproduced the checked-in PTX byte for byte.

## Native Ollama baselines

Each row uses two warmups and nine timed streaming requests. The model context is 2,048 tokens, temperature zero, seed 42, and the token cap is 48. Prompts vary their leading record ID to limit prefix reuse. Prompt rates use the API's evaluated token counts and durations, so they reflect the work after any prefix reuse. First-token latency is measured at the HTTP client from request start to the first nonempty streamed response. The table uses nearest-rank median / p95. With nine observations p95 is the observed maximum. The memory columns are Ollama's resident model estimates from `/api/ps`, including model buffers and runtime allocations that Ollama counts. They are separate from process working set and physical GPU-only memory on these integrated devices.

| Host and backend | Model bytes / GPU bytes | Prompt tok/s | Generation tok/s | First token ms |
| --- | ---: | ---: | ---: | ---: |
| Iris Xe CPU | 458,804,427 / 0 | 138.1 / 160.8 | 50.8 / 58.0 | 1881.8 / 2103.3 |
| Iris Xe Vulkan | 456,361,245 / 456,361,245 | 1109.0 / 1179.1 | 60.1 / 61.5 | 275.3 / 312.4 |
| Arc B390 CPU | 458,804,427 / 0 | 432.7 / 437.2 | 149.0 / 150.5 | 600.7 / 604.6 |
| Arc B390 Vulkan | 456,361,245 / 456,361,245 | 7663.8 / 7689.1 | 204.6 / 213.4 | 52.2 / 53.2 |

All 44 requests, including warmups, completed and contained “Paris.” This is a bounded factual and stability check on the generated text. By default Ollama filtered both Vulkan devices as integrated. `OLLAMA_VULKAN=1` and `OLLAMA_IGPU_ENABLE=1` selected the Intel adapter on each host. Both logs recorded 25/25 layers offloaded and Vulkan buffers of 373.71 MiB for weights, 24.00 MiB for KV and 37.51 MiB for compute. The CPU requests used `num_gpu=0` and reported zero GPU bytes. In a separate generation observation, Iris Xe WDDM GPU engine samples attributed up to 97% utilization to the Vulkan `llama-server.exe` PID's 3D engine. The Iris Xe runner's resident working set was 750,776,320 B after that observation. Arc B390's runner reported the same layer offload and Vulkan buffer placement and used a 795,811,840 B working set in a separate generation. An independent Arc B390 WDDM engine counter sample was inconclusive.

## CUDA fixture performance

Each completed or timestamped case uses 31 timed samples after five warmups. Times are median / p95 in µs. Launch + wait measures the Driver API launch through completed context synchronization. Transfer-inclusive time includes three blocking HtoD copies (two inputs and the guarded destination), launch and blocking DtoH. CPU comparison is outside the timed region. Kernel-event time comes from Level Zero's kernel timestamps in a separate instrumented test on a tightly packed `64×896` shape, whereas the DLL test has different row pitches and guard regions. Translation-only time measures PTX parsing and SPIR-V emission in process. Module load includes translation and Level Zero module/kernel creation, with the driver's cache available. The first load is a single observation and repeat loads have 11 observations. The test does not purge the driver cache.

| Workload | Iris Xe immediate | Iris Xe batched | Arc B390 immediate | Arc B390 batched |
| --- | ---: | ---: | ---: | ---: |
| 7×257 launch + wait | 82.9 / 142.9 | 84.3 / 115.6 | 7.4 / 10.5 | 10.8 / 11.3 |
| 7×257 transfers included | 824.8 / 1740.1 | 920.8 / 1260.3 | 43.4 / 68.5 | 50.3 / 57.9 |
| 64×896 launch + wait | 136.4 / 239.0 | 104.8 / 145.7 | 18.0 / 29.4 | 22.7 / 28.4 |
| 64×896 transfers included | 1277.2 / 2314.1 | 1144.2 / 1347.9 | 144.6 / 392.0 | 124.8 / 177.3 |
| 64×896 kernel event | 9.69 / 11.46 | 8.33 / 8.96 | 2.19 / 2.29 | 2.50 / 2.60 |
| Translation only | 97.3 / 119.4 | 160.7 / 233.9 | 58.4 / 89.0 | 65.9 / 82.7 |
| First module load | 1486.6 | 896.6 | 486.3 | 439.1 |
| Repeat module loads | 396.6 / 530.5 | 469.7 / 629.6 | 190.4 / 268.3 | 217.5 / 384.2 |

Batching used a 32-command cap and a 250 µs high-resolution deadline. These isolated launch and transfer tests force frequent synchronization, which bounds batching opportunities. The baseline package exposes native Vulkan end-to-end inference timings but no matching standalone add benchmark. No kernel-level comparison was made to native Vulkan, SYCL, or NVIDIA CUDA.

## Cross-GPU regression and end-to-end attempt

Both Intel devices ran the CUDA add fixture and kernel timestamp case in immediate and batched modes. Formatting, debug and release locked builds, Clippy with warnings denied, and 28 ungated tests passed for both configurations. SPIR-V Tools v2025.1 on Iris Xe and v2026.4 on Arc B390 independently validated the output against the OpenCL 2.2 environment in both configurations. Gated DLL regressions were run with the Forge workload excluded. Iris Xe passed all 12 selected DLL tests in each mode, with the physical multi-device case reporting its one-device capability skip. On Arc B390 the initial immediate run passed 11 and failed the existing pinned-copy test. The rerun of the other 11 selected cases passed in both modes, with the same physical-device skip. The failed case read stale host data after retiring a stream and freeing the pinned source. It does not load a CUDA module. Isolated release reruns in immediate mode failed 3/24 times on the Milestone 12 source tree `a181a08b818ec878bcbc578970b6690ca704ddc0` and 2/24 on this tree. Panther Lake SSH closed after the run and subsequent connections closed before authentication.

On Arc B390 the full Windows x64 Ollama package was staged in a private directory. The compatibility DLL was placed beside `llama-server.exe`, with `GGML_BACKEND_PATH` pointing to the stock CUDA 13 module and Vulkan disabled. No system DLL, SSH configuration or hardware identity was changed. `llama-server --list-devices` reported no devices. The service log showed the CUDA 13 plugin loading, then `ggml_cuda_init: failed to initialize CUDA: (null)` and a native-discovery error for the missing `cuDriverGetVersion` procedure. CUDA 12 discovery separately reported failure to load `nvcuda.dll`. A Qwen generate request with 99 requested GPU layers returned eight valid tokens on the CPU. The service selected CPU inference and `/api/ps` reported `size_vram=0`. There was no CUDA compatibility inference throughput to measure.

## Path to full CUDA inference

| Approach | Assessment |
| --- | --- |
| Substitute a broader `nvcuda.dll` under stock cudart and cuBLAS | Highest fidelity to the shipped host ABI. Requires discovery identity policy, versioned driver dispatch and contexts, fatbins and JIT/link support, CUDA memory and stream semantics, and sufficient coverage for proprietary cuBLAS. A few discovery exports cannot establish execution. |
| Replace CUDA Runtime and cuBLAS at their ABIs | Preserves host call sites but must match versioned ABI, registration and hidden/static entry points. Compatibility would be fragile across binaries and toolkit releases. |
| Build a compatibility variant of GGML CUDA with a narrow adapter | Keeps CUDA device source and GGML graph dispatch, and can explicitly register supported standalone PTX and route to this driver. Use a capability-specific backend registration with no false NVIDIA compute capability. Disable graphs, VMM, PDL and multi-GPU features initially. Reject unsupported graph operations at load or planning time. |
| Implement selected GEMM/GEMV library operations | Needed wherever the chosen workload actually dispatches to cuBLAS. Prefer a scoped implementation with verified CUDA layout, transpose, scale, accumulation precision and stream ordering, or cover the graph through GGML's own supported CUDA kernels. |

For Milestone 14, first trace the pinned Qwen graph's op types and shapes at decode and prompt sizes. Add an explicitly selected GGML CUDA compatibility build and a small graph acceptance test. Extend device coverage in order of that trace. Likely early work includes scoped FP16/BF16 conversions, bounds-checked vector loads, typed signed and bitwise operations, and shuffle semantics that preserve CUDA's 32-lane masks on varying Intel subgroup widths. Then validate actual GGML RMSNorm and a Q4_K GEMV path. Add device-to-device copies and fills before model graph execution. Measure dense and quantized projection paths separately before taking on a cuBLAS ABI surface. Keep stock discovery and binary loading as a separate investigation.

## Reproduce

Run from the repository root with Windows x64 Rust, an Intel Level Zero driver, and `spirv-val` on PATH or selected by `SPIRV_VAL`. The fixture does not require CUDA SDK headers. Clang 23.1.1's NVPTX backend produced the checked-in file.

```powershell
clang++ --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -O2 -fno-unroll-loops -S examples/ggml_add_f32.cu -o examples/ggml_add_f32.ptx
cargo fmt --all -- --check
cargo build --locked
cargo build --release --locked
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo test --locked --lib validate_opencl_spirv -- --ignored --nocapture
cargo run --locked --example devices
$env:CUDA_FOR_INTEL_DEVICE = '0'
cargo test --release --locked --test gpu ggml_cuda_add_f32 -- --ignored --nocapture --test-threads=1
cargo test --release --locked --lib gated_ggml_add_device_time -- --ignored --nocapture --test-threads=1
```

Repeat the build, Clippy and tests with `--features experimental-batching` on each host. Set `CUDA_FOR_INTEL_BATCH_CAP=32`, `CUDA_FOR_INTEL_BATCH_US=250`, and `CUDA_FOR_INTEL_BATCH_TIMER=high-resolution` for the opt-in tests. Clear those three variables for immediate timing. Use `CUDA_FOR_INTEL_LOG=1` to diagnose translation or Level Zero errors.

For native baseline runs, unpack the pinned official release in a private directory. Run the following server and client commands in separate shells. For the Vulkan pass restart the same server with `OLLAMA_VULKAN=1` and `OLLAMA_IGPU_ENABLE=1`, and change the benchmark flag to `-Backend Vulkan`. Verify the log identifies the Vulkan adapter and records layer offload. Sample `Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine` instances with the matching runner PID while a separate generation is active.

```powershell
# Server shell from the private Ollama installation
$env:OLLAMA_HOST = '127.0.0.1:11513'
$env:OLLAMA_MODELS = Join-Path $PWD 'm13-models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_DEBUG = '1'
$env:OLLAMA_VULKAN = '0'
.\ollama.exe serve

# Client shell
$env:OLLAMA_HOST = '127.0.0.1:11513'
& 'C:/path/to/private/ollama.exe' pull qwen2.5:0.5b
pwsh -File scripts/benchmark-ollama.ps1 -Backend CPU
```

For the stock CUDA initialization attempt, use a fresh copy of the full private Ollama package. Run from its root, with `$shim` set to this repository's absolute release DLL path. The Windows process environment is sufficient for the test.

```powershell
$base = Join-Path $PWD 'lib/ollama'
$cuda = Join-Path $base 'cuda_v13'
$destination = Join-Path $base 'nvcuda.dll'
if (Test-Path $destination) { throw 'Use a fresh private release directory' }
Copy-Item -LiteralPath $shim -Destination $destination
$env:PATH = "$cuda;$env:PATH"
$env:GGML_BACKEND_PATH = Join-Path $cuda 'ggml-cuda.dll'
$env:OLLAMA_VULKAN = '0'
& (Join-Path $base 'llama-server.exe') --list-devices
```

Start the private server with the same environment, as above. Request a short `/api/generate` completion with `options.num_gpu=99`, then check the startup log, selected backend, and `/api/ps`. For the binary audit, run `cuobjdump -lptx` and `cuobjdump -all -lelf` against each variant's `ggml-cuda.dll`. Inspect the PE imports with `llvm-objdump -p`.
