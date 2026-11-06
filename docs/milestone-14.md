# Milestone 14 — CUDA compatibility foundation

## Executive summary

The runtime can negotiate a supported Driver API ABI, report Level Zero PCI identity, retain a device's primary context, and launch multiple entries from a single PTX 4.1 module. Immediate command lists remain the production default. The stock Ollama v0.40.15 CUDA 13 runner now reaches the project DLL's versioned resolver. Its first observed lookup is `cuGetExportTable`, which has no implementation here. CUDA 12's plugin has eight unresolved static CUDA VMM imports. Neither plugin enumerated a CUDA device or executed a model layer.

The starting revision was pushed `main` at `09bbcaf17523a84a37390a918e25b9ed9226883f`. The comparison uses the public ZLUDA head `f4df215edbc3b6698a1c61ef7a8688fdff3872be`. Ollama and the bundled GGML source are pinned in the Milestone 13 report. [Starting revision](https://github.com/amw720386/cuda-for-intel/commit/09bbcaf17523a84a37390a918e25b9ed9226883f) · [ZLUDA source pin](https://github.com/vosen/ZLUDA/tree/f4df215edbc3b6698a1c61ef7a8688fdff3872be) · [Ollama v0.40.15](https://github.com/ollama/ollama/releases/tag/v0.40.15) · [Milestone 13](milestone-13.md)

## Source-level architectural audit

| Subsystem | This runtime and modern ZLUDA | Application requirement and implementation boundary |
| --- | --- | --- |
| Driver ABI | `src/cuda/discovery.rs` has a supported-symbol table with ABI ranges. ZLUDA's `zluda/src/impl/driver.rs::get_proc_address_v2` consumes the generated `zluda_bindgen/src/process_table.rs`; its exports route to Rust implementations and HIP. | Pinned GGML CUDA 12 imports Driver API VMM functions directly. The CUDA 13 runner requests version 13000 through the resolver. Reuse the generated-table approach and use the public CUDA typedefs to verify boundaries. Own the Level Zero operations and error mapping. Returning a pointer to a different ABI or claiming PTDS semantics would make callers unsafe. [ZLUDA driver](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda/src/impl/driver.rs) · [ZLUDA resolver table](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda_bindgen/src/process_table.rs) |
| CUDA Runtime bootstrap | `cuInit` loads Intel Level Zero. ZLUDA's `get_export_table` and `CudaDarkApiGlobalTable` expose UUID-selected tables with specific callbacks, including primary-context and module hooks. The `zluda_trace` layer can wrap these interfaces. | CUDA 13's static cudart looks up the function that returns these tables. First measure the UUIDs, table slots, and required callbacks with a known CUDA reference. Reuse compatible, audited interface definitions where possible. Implement the callbacks against this project's ownership model. Table layout, lifetime, and failure behavior become externally observable ABI. Windows Ollama's outer `probeCUDADriverWindows` also searches System32, independently of the private runner test. [ZLUDA export tables](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/dark_api/src/lib.rs) · [Ollama Windows native probe](https://github.com/ollama/ollama/blob/02480ca2a47153e1f9c31cbe1e05536da97ce1ba/discover/native_probe_windows.go) |
| PTX parser and compiler | `src/ptx/parse.rs` parses a narrow PTX 4.1 dialect. `flow.rs` checks control flow and `spirv.rs` emits OpenCL SPIR-V. ZLUDA `ptx_parser::parse_module_checked` builds an AST. `ptx::to_llvm_module` normalizes predicates and basic blocks, expands operands, inserts implicit conversions and hoists globals before LLVM emission. | CUDA 13's inspected payload starts at PTX 9.0 and GGML requires aggregate/vector parameters, mixed precision, masked shuffles and newer memory/atomic forms. Reuse the parser and evaluate individual normalization passes. Own the mapping to SPIR-V, including Intel address spaces and execution scopes. Keep CUDA execution and FP semantics explicit in the lowering boundary. [ZLUDA parser](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/ptx_parser/src/lib.rs) · [Pass pipeline](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/ptx/src/pass/mod.rs) |
| Module and fatbin loading | `cuda.rs::load_ptx` validates all entries and owns one Level Zero module per CUDA module. ZLUDA's `dark_api::fatbin` iterates wrappers and images with decompression. `module.rs::get_best_ptx_and_compile` and `compile_and_cache` select PTX, compile for AMD, cache the binary, and load via HIP. `get_global_v2` exposes module globals. | Shipped GGML contains many compressed images and kernel registrations, with device symbols and code selection downstream of bootstrap. Evaluate reuse of fatbin format/decompression logic with a checked slice interface. Own selection policy, globals, Intel module lifetime and cache invalidation. Never key compatibility on a fake physical SM or use an unbounded pointer-derived size as a trusted container length. [Fatbin reader](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/dark_api/src/fatbin.rs) · [ZLUDA module implementation](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda/src/impl/module.rs) |
| Contexts and devices | `State::create_context` and `destroy_context` own Level Zero contexts and their resources, with thread-local selection and a per-device primary record. ZLUDA `context.rs::ContextState::reset` and `device.rs::primary_context_retain` share a stable HIP-backed primary object; `context.rs` manages the current stack. | Runtime and libraries need one primary context per device and shared ownership across threads. Reuse lifecycle tests as behavioral references. Own Level Zero context creation, draining, retirement, and stale-handle checks. Validate reset/retention against NVIDIA before promising a stable primary handle across generations or full CUDA context-stack behavior. [ZLUDA contexts](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda/src/impl/context.rs) · [ZLUDA devices](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda/src/impl/device.rs) |
| Registration and launch | `cuModuleGetFunction` lazily creates a Level Zero kernel. `cuLaunchKernel` validates per-entry arguments and enqueues work through `Streams::launch`. ZLUDA `function.rs::launch_kernel` dispatches through HIP, including an alternate packed argument buffer path. Its export table supplies runtime module callbacks. | GGML's compiled CUDA host code needs fatbin and function registration, aggregate/packed parameters, dynamic shared memory, attributes and occupancy. Graph capture and dependent launch are conditional in GGML. Own argument metadata and launch validation; use published CUDA buffer/argument ABI definitions. Keep registration module ownership independent of stream work and avoid tying names to global process-wide kernel handles. [ZLUDA launches](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda/src/impl/function.rs) · [Pinned GGML CUDA source](https://github.com/ggml-org/llama.cpp/blob/52e53a76df0298d2c316ae7e704eeec03b0bf478/ggml/src/ggml-cuda/ggml-cuda.cu) |
| cuBLAS | The project exports no cuBLAS DLL. ZLUDA's `zluda_blas/src/impl.rs` owns handles and forwards functions such as `sgemm_v2`, `gemm_ex` and strided batched GEMM to rocBLAS. | The stock plugins import cuBLAS. GGML's dense and batched paths require handle, stream, pointer mode, math mode, workspace, and GEMM behavior. Reuse the adapter pattern. Evaluate Intel oneMKL or oneMath backends for actual GEMM; own the CUDA layout, data type, accumulation and asynchronous ordering adapter. Decide these semantics before binding a vendor library's handles to CUDA streams. [ZLUDA BLAS implementation](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda_blas/src/impl.rs) · [Intel oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) |
| Errors and tracing | `cuda.rs::backend_error` maps Level Zero status codes. `trace.rs` logs exported API results and sanitized lookup details. ZLUDA `zluda_trace/src/lib.rs::logged_call` and `log.rs::Writer` wrap calls and private table entries with structured diagnostics. | Runtime bootstrap needs exact result values and resolver outcomes. Reuse the separation of resolution and invocation logging. Own a narrow logging policy and capture explicit outcomes at native failure and lookup boundaries. Treat optional missing queries separately from a demonstrated initialization blocker. [ZLUDA tracer](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda_trace/src/lib.rs) · [ZLUDA log writer](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda_trace/src/log.rs) |

## Implemented functionality and compatibility matrix

| Surface | Milestone 14 behavior |
| --- | --- |
| Discovery | `cuDriverGetVersion` returns 13000 before initialization. This is an entry-point negotiation ceiling. It carries no guarantee of CUDA 13 feature or PTX coverage. `cuDeviceGetPCIBusId` and attributes 33, 34 and 50 use optional `zeDevicePciGetPropertiesExt`. Attribute 18 is the physical Level Zero integrated flag. Attributes 1–8 retain their launch and local-memory limits. |
| NVIDIA architecture identity | Compute capability major/minor and warp-size queries return unsupported. The `sm_*` directive on accepted PTX is an input target label. Supported features are determined by the parser, lowerer, and Level Zero capabilities. |
| Resolution | `cuGetProcAddress` and `cuGetProcAddress_v2` accept base names and choose ABI versions from a whitelist. The four-argument resolver returns 500 for unresolved entries. The five-argument resolver returns a null pointer and a query result of 1 for unavailable entries or 2 for incompatible ABI versions. A request above 13000 or invalid flags returns 1. Default and legacy stream modes resolve legacy functions. Explicit PTDS requests fail for stream-sensitive functions with unimplemented PTDS variants. For example, `cuCtxCreate` resolves only the supported v2 range and `cuCtxGetDevice` does not return the one-argument ABI at version 13000. |
| Error introspection | `cuGetErrorName` and `cuGetErrorString` accept integer status codes before initialization, including all codes returned by the runtime. Unknown values return `CUDA_ERROR_INVALID_VALUE` and a null output. |
| Primary contexts | `Retain` shares a context per device without changing thread selection. `GetState` reports automatic scheduling and activity. `SetFlags` and `SetFlags_v2` support zero flags. The legacy setter rejects active contexts. `Release` and `Release_v2` drain and retire resources on final release. `Reset` and `Reset_v2` invalidate that generation's resources while preserving outstanding retain obligations. A later retain can create a new generation. A previously selected handle remains on its thread's stack until explicitly cleared or replaced and rejects work after retirement. `cuCtxDestroy_v2` rejects primary handles. Teardown with active host waiters returns `CUDA_ERROR_NOT_READY` and partial native teardown can be retried. |
| Modules and launches | Plain PTX 4.1 can contain multiple entry points with unique names. The compiled SPIR-V holds all kernels. Each entry has separate argument metadata, launch limits, and native kernel handle. Unload drains outstanding work, retires all function handles, and cleans up native kernels and the module in order. Partial destruction leaves a cleanup-only module for retry. Existing legacy stream and event ordering applies. |
| Trace | Set `CUDA_FOR_INTEL_TRACE=1` before the first API call. Stderr includes API completion codes, resolver names/versions/flags/status, device attributes and initialization, module and launch calls, and mapped Level Zero errors. It omits pointers, buffers, kernel arguments, PTX bodies and native build logs. The separate `CUDA_FOR_INTEL_LOG` retains its original verbose translation diagnostics. |
| Remaining unsupported | Private export tables, complete CUDA Runtime bootstrap, CUDA VMM, arbitrary context scheduling policies, per-thread default streams, fatbin loading/registration, device globals, JIT options and application caches, cubin execution, dynamic shared memory, packed launch arguments, and CUDA libraries. |

[NVIDIA entry-point API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html) · [Legacy entry-point API](https://docs.nvidia.com/cuda/archive/11.8.0/cuda-driver-api/group__CUDA__DRIVER__ENTRY__POINT.html) · [NVIDIA primary-context API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__PRIMARY__CTX.html) · [Level Zero PCI properties](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/EXT_PCIProperties.html)

## Compiler architecture evaluation

| Dimension | ZLUDA parser and AST adapter → current OpenCL SPIR-V emitter | PTX → LLVM IR → SPIR-V |
| --- | --- | --- |
| Complexity and coverage | Checked parser already handles much broader grammar. Adapter still must reject instructions without supported lowering and represent types, scopes, address spaces and per-entry metadata. Selected normalization passes may be portable after separating AMD assumptions. | Needs the same PTX frontend plus a separate SPIR/OpenCL LLVM code generator. Khronos `llvm-spirv` translates LLVM IR, so grammar and PTX semantics stay upstream of it. LLVM's own SPIR-V target is a separate candidate. |
| CUDA warp and memory semantics | Keep a logical 32-lane warp, participation mask, convergence and PTX ordering explicit. Gate direct subgroup lowering on a validated subgroup width and control-flow contract. Wider or narrower physical groups need semantics-preserving lowering or explicit rejection. | Same contract must survive LLVM optimizations and translation. ZLUDA's `emit.rs::kernel_call_convention` emits AMDGPU kernel ABI and `emit_method` sets wavefront features. AMD intrinsics, address-space conventions and linked helpers require replacement before SPIR-V translation. |
| FP correctness | Retain the current explicit FMA, contraction controls and rounded divide/sqrt paths until differential tests validate replacements. | LLVM can optimize SSA and memory accesses, but contraction, NaNs, subnormals, signed zeros and constrained rounding need end-to-end tests through the optimizer, translator and Intel driver. |
| Optimization | Current direct emitter relies on the Level Zero compiler for target optimization. A typed IR would support shared simplifications and reduce duplicated logic. | LLVM offers stronger optimization and tooling. Evaluate only after IR expresses CUDA-specific constraints and the emitted OpenCL environment passes independent validation. |
| Windows deployment, size and latency | Parser probe compiled as a standalone Windows x64 Rust executable with no LLVM or HIP dependency. No project dependency or production cold-load cost was added for the probe. | Bundling a matching LLVM/translator pair adds build and deployment work. Binary size and cold/warm compilation latency for an equivalent implementation remain unmeasured. Khronos documents matching LLVM release branches and SPIR-V version restrictions. |
| Maintenance and license | Pin a small parser integration and track upstream grammar. ZLUDA publishes MIT and Apache-2.0 license texts. Retain applicable notices and check any reused component's dependencies before integration. | Pin and test the compiler, headers and translator together. LLVM uses Apache-2.0 with LLVM exceptions. Khronos' translator carries its own license notice. Verify notices for the chosen build and linked dependencies. |

The proof of concept built the unmodified `ptx_parser` crate in a separate Windows x64 Rust executable. `parse_module_checked` accepted all seven checked-in example PTX files and the new two-entry fixture. That measures syntax acceptance. The production compiler ran its independent OpenCL SPIR-V validation and numerical GPU suite separately. A complete alternate LLVM lowering and benchmark were outside this experiment. The Windows Clang executable used for the fixture work exposes NVPTX and has no SPIR-V codegen target.

Use ZLUDA's parser behind a small adapter to a typed, target-neutral representation, and retain the current validated SPIR-V backend through a differential migration. First port the existing corpus to the adapter, then compare ABI metadata, diagnostics, independent SPIR-V validation and bit-pattern results on both devices. Keep an LLVM route experimental until it lowers the same fixture suite, including barriers, atomics and the FP32 rounding tests, and compare artifact size and cold/warm compilation before deploying it. [Khronos translator](https://github.com/KhronosGroup/SPIRV-LLVM-Translator) · [Translator license](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/blob/main/LICENSE.TXT) · [LLVM SPIR-V target](https://llvm.org/docs/SPIRVUsage.html) · [LLVM license](https://llvm.org/LICENSE.txt) · [ZLUDA AMDGPU emitter](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/ptx/src/pass/llvm/emit.rs) · [ZLUDA MIT license](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/LICENSE-MIT) · [ZLUDA Apache license](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/LICENSE-APACHE)

## Module and fatbin plan

1. Start with a size-bounded raw fatbin reader and structured image descriptors for kind, PTX version, target and compression. Validate all offsets, sizes, decompression bounds and alignment before allocating. Treat the pointer-only CUDA ABI and wrapper memory as a separate acquisition layer. Test truncated, overlapping, oversized, mixed-image and compression cases. Cubin records can be recognized and skipped without decoding SASS.
2. Select supported PTX by explicit compiler target/version and capability policy. Try only candidates whose required semantics are implemented. Distinguish a malformed container, unsupported PTX version, and no compatible image. Verify selection against the pinned CUDA 12 and 13 payloads.
3. Extend entry metadata to preserve aligned aggregate parameters, extern/dynamic shared memory, linked functions, and device-global/constant symbols. Keep pointer validation tied to the owning context. Expose symbol lookup and sizes only once address lifetime and initialization match CUDA.
4. Keep CUDA-facing handles per context and attach a generation to lazy native compilation and registration. Preserve the existing draining unload until command completions own references to every native object they use. Test launch/unload, reset, cache reuse, and concurrent teardown against both submission backends.
5. If caching is needed, key artifacts by selected PTX bytes, translator build, lowering and numerical policy, device/compiler identity, and relevant options. Validate cached metadata and artifacts before loading; serialize cache publication atomically.

## Stock Ollama initialization

The runner was the unmodified Windows x64 release (`ollama-windows-amd64.zip` SHA-256 `9ca74a7ab66ec86870b1eecac7df160df08dc4e3a291ab5d2d59bcf9310221c6`). `llama-server.exe --list-devices` ran with the chosen stock CUDA plugin, Vulkan disabled and the project DLL staged privately beside the runner. The operation exercises plugin initialization without a model or inference fallback.

| Variant | Pushed baseline | Milestone 14 |
| --- | --- | --- |
| CUDA 13 | No CUDA device. The DLL has no `cuGetProcAddress` export. | No CUDA device. The plugin makes two legacy resolver calls for `cuGetExportTable` at version 13000, and both return 500. No subsequent `cuInit`, context retain, module load or launch is observed in this path. |
| CUDA 12 | No CUDA device. Nine of eleven `nvcuda.dll` static imports are absent. | No CUDA device. `cuGetErrorString` is now present; eight required VMM imports remain unresolved before plugin initialization. |

The CUDA 13 trace below is the first failure at the shim boundary. `cuGetExportTable` is a documented function returning tables whose UUIDs and slot semantics are outside the public Driver API contract. Resolving the name alone cannot safely satisfy the runtime. The separate CUDA 12 blocker is supported by the PE import/export comparison. Its missing imports are `cuMemAddressFree`, `cuMemAddressReserve`, `cuMemCreate`, `cuMemGetAllocationGranularity`, `cuMemMap`, `cuMemRelease`, `cuMemSetAccess` and `cuMemUnmap`. [NVIDIA CUDA binary utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/index.html) · [ZLUDA export-table implementation](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda/src/impl/driver.rs#L598)

```text
CUDA for Intel trace resolve cuGetExportTable version=13000 flags=0 status=1
CUDA for Intel trace cuGetProcAddress -> 500
CUDA for Intel trace resolve cuGetExportTable version=13000 flags=0 status=1
CUDA for Intel trace cuGetProcAddress -> 500
Available devices:
  (none)
```

In a separate Iris Xe service startup, the baseline reproduced `ggml_cuda_init: failed to initialize CUDA: (null)` and the native discovery error for missing `cuDriverGetVersion`. The same service with the Milestone 14 DLL reported zero GPU candidates and `total_vram="0 B"`. Both selected the CPU backend. The direct CUDA 12 plugin load with the private DLL preloaded also returned Win32 error 127 (`ERROR_PROC_NOT_FOUND`). No inference request ran in these probes.

A separate diagnostic-only Iris Xe DLL exposed the `cuGetExportTable` signature, logged the requested 16-byte ID, cleared the output, and returned `CUDA_ERROR_NOT_SUPPORTED`. Both invocations requested `DD9E98A4-F410-4844-99B0-E2AEB7B432B5`, and the stock plugin logged `ggml_cuda_init: failed to initialize CUDA: (null)`. Current ZLUDA names that interface `HYBRID_CUDART`. Its Windows callback `hybrid_runtime_load_get_proc_address` allowlists `nvcudart_hybrid64.dll` and `nvcudart_hybrid64a.dll`, then resolves `__cudaGetProcAddress`. Neither hybrid DLL name appeared in the pinned Ollama ZIP. The initial Windows handoff requires a separate compatibility and dependency investigation. The diagnostic exposed no usable table and was separate from the validated production DLL. [ZLUDA table layout](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/dark_api/src/lib.rs#L342) · [ZLUDA hybrid loader](https://github.com/vosen/ZLUDA/blob/f4df215edbc3b6698a1c61ef7a8688fdff3872be/zluda/src/impl/driver.rs#L426)

## Known limitations and remaining blockers

The tested resolver and context behavior cover the documented subset on Windows x64 with Intel Level Zero. Old and modern NVIDIA entry-point semantics beyond that subset, primary handle identity across reset on NVIDIA, and mixed Driver/Runtime ownership need differential tests on a reference NVIDIA driver. The stock CUDA 13 package's PTX 9.0, image registration, actual launch ABI and cuBLAS paths remain behind the hybrid-runtime handoff. The pinned-copy issue still limits application-level validation on Arc.

## Validation results

Both requested hosts used Rust 1.99.0 x86-64 MSVC and SPIRV-Tools v2026.4. Each host passed formatting, locked debug and release builds, and Clippy for all targets with warnings denied in both feature configurations. The cross-GPU runner ran with one release comparison pass and the normal immediate and C250 batch correctness configurations.

| Check | Iris Xe 32.0.101.7085 | Arc B390 32.0.101.9033 |
| --- | --- | --- |
| Hardware-independent unit tests | 32 passed per feature configuration | 32 passed per feature configuration |
| Independent OpenCL 2.2 SPIR-V validation | 1 test validating 9 modules per configuration | 1 test validating 9 modules per configuration |
| DLL regressions, immediate / batching | 15 / 15 passed | 15 / 15 passed |
| Gated library tests, immediate / batching | 16 / 16 passed | 16 / 16 passed |
| Additional release cases | 4 comparison cases, 1 host timer, 1 pinned transfer benchmark passed | 4 comparison cases, 1 host timer, 1 pinned transfer benchmark, 1 dedicated copy-queue test passed |
| Physical multi-device isolation | Skipped, one root GPU | Skipped, one root GPU |
| Dedicated copy queue | Skipped, no copy-only queue | Forge correctness and device timestamp brackets passed. Mapped timestamps failed host-clock validation. |
| PCI ID in the new DLL test | `0000:00:02.0`, attributes matched | `0000:00:02.0`, attributes matched |

The new DLL test calls a resolved `cuInit` and `cuDevicePrimaryCtxRetain`, launches two entries with different argument lists through the shared context, reads back `0x12345678`, unloads with submitted work, and verifies that the retired function handle fails. The fake native ledger tests cover both submission policies. They inject native context create/destroy failures, a second kernel create failure and partial multi-kernel destruction, then verify retry, release balance and zero native ownership violations. The two-device fake model also ran in both configurations.

In the debug DLL runner's `64×896` GGML residual-add fixture, median / p95 completed launch time in µs was 157.40 / 253.90 (Iris Xe immediate), 139.00 / 189.30 (Iris Xe batch), 22.10 / 412.70 (Arc immediate) and 29.50 / 666.60 (Arc batch). The separate Level Zero kernel event medians were 7.19, 7.92, 2.19 and 2.50 µs respectively. Each distribution used 31 measured samples. These are current test observations with different fixture layouts and timing boundaries across the two tests. They do not establish a change relative to Milestone 13 release measurements.

## Arc B390 pinned-copy investigation

The standard debug DLL suite passed once per backend on Arc. The isolated immediate release test reproduced the existing failure in 4/40 processes using the retained Milestone 12 artifact (`a181a08b818ec878bcbc578970b6690ca704ddc0`) and 6/40 using Milestone 14. The test still fails on any mismatch.

The failing branch issues a 37-byte HtoD and DtoH pair on the same blocking stream, destroys the stream and frees the pinned source. `cuStreamDestroy_v2` checks immediate-list completion with a zero-time poll or retires the list for later collection. `cuMemFreeHost` drains all remaining lists before freeing the allocation. Every failure retained the exact destination contents of the prior partial DtoH. In all six Milestone 14 failures, a separate blocking DtoH after capturing the mismatch returned CUDA success and recovered the full expected input from the device allocation. That evidence narrows the failure to completion or host visibility of the second copy around immediate-list retirement. The diagnostic read occurs later, so it cannot establish when the upload completed. The fake native ledger checks object ownership and queue completion accounting but cannot model WDDM visibility or this driver's immediate copy behavior. [Existing reproduction](milestone-13.md#cross-gpu-regression-and-end-to-end-attempt) · [Level Zero immediate-list synchronization](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/api/apis/command_list-2.html#zecommandlisthostsynchronize)

A second experiment ran 80 fresh release test processes per arm in interleaved order against the same Arc DLL. The control mode was read before any CUDA work. The only intentional difference was an optional wait after enqueuing the full DtoH and before `cuStreamDestroy_v2`. All arms retained the original assertion and subsequent lifetime checks.

| Completion control | Failures |
| --- | ---: |
| Original retirement and host-free path | 6/80 |
| Explicit `cuStreamSynchronize(up)` before retirement | 6/80 |
| Record an event on `up` and wait with `cuEventSynchronize` before retirement | 0/80 |

An earlier control batch that read the mode between copy and retirement had 0/80 failures in every arm. That batch could not distinguish the controls. The revised experiment shows the failure can persist with an explicit immediate-list wait. The event control adds a Level Zero barrier with a host-scope signal and waits on that event, so the result does not isolate the barrier, event scope, and elapsed time. There is no production mitigation from this experiment. The next control should reproduce the copy pair directly in Level Zero and compare a host-scope event attached to the DtoH, an appended barrier, and immediate-list wait while holding timing and allocation reuse constant.

## Prioritized Milestone 15 work

1. Record the minimum callback sequence for the observed CUDA 13 hybrid-runtime table on a reference NVIDIA driver. Use a separately instrumented process and compare observable return codes, thread selection and cleanup. Implement each necessary table entry with an identified ABI and real backend semantics.
2. Isolate the Arc pinned-copy failure in a native Level Zero control. Identify whether upload, download, or host visibility fails under the same submission and retirement order. Require reproducible correctness before changing synchronization or choosing another queue.
3. Integrate the tested PTX parser through a differential adapter and add bounded fatbin image selection. Start with multi-entry PTX fixtures and one real registration payload before broadening instruction coverage.
4. For the CUDA 12 variant, map the eight VMM imports to actual Level Zero virtual reservation, mapping, access and lifetime behavior. Test alignment, reuse, fault handling and device capability gating before exposing symbols. CUDA 13 is the more direct stock bootstrap probe in this milestone.
5. Add the memory operations, kernel launch ABI and library paths justified by the first successful runtime trace. Validate a small unmodified GPU workload before a full GGML graph. Scope the cuBLAS adapter to verified dispatches and establish layout, precision and stream ordering against a reference.

## Reproduction

Run in PowerShell 7 from the repository root on each host with Windows x64 MSVC and `spirv-val` on PATH or selected by `SPIRV_VAL`.

```powershell
cargo fmt --all -- --check
foreach ($backend in 'immediate', 'batch') {
    $features = if ($backend -eq 'batch') { @('--features', 'experimental-batching') } else { @() }
    cargo build --locked @features
    cargo build --release --locked @features
    cargo clippy --locked --all-targets @features -- -D warnings
    cargo test --locked @features
    cargo test --locked --lib validate_opencl_spirv @features -- --ignored --nocapture
}
pwsh -File scripts/validate-gpu.ps1 -PerformancePasses 1 -LogPath "$env:TEMP/m14-gpu.log"
$env:CUDA_FOR_INTEL_DEVICE = '0'
cargo test --release --locked --lib gated_m10_host_timer -- --ignored --nocapture --test-threads=1
cargo test --release --locked --lib pinned_transfer_benchmark --features experimental-batching -- --ignored --nocapture --test-threads=1
# Run the next test only if the inventory lists a usable copy-only queue.
cargo test --release --locked --lib gated_forge_copy_queue_timestamps -- --ignored --nocapture --test-threads=1

cargo build --release --locked
cargo test --release --locked --test gpu --no-run
$test = Get-ChildItem target/release/deps/gpu-*.exe | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$failures = 0
1..40 | ForEach-Object {
    "iteration=$_"
    & $test.FullName pinned_async_transfers --ignored --nocapture --test-threads=1
    if ($LASTEXITCODE) { $failures++ }
}
"failures=$failures/40"
```

For the Arc completion controls, use a temporary copy of `tests/gpu/pinned.rs`. Read the mode before CUDA calls. After the full `cuMemcpyDtoHAsync_v2(dst, device, SIZE, up)` in the retirement block and before `cuStreamDestroy_v2(up)`, add either no call, `cuStreamSynchronize(up)`, or `cuEventRecord(event, up)` followed by `cuEventSynchronize(event)`. Check every call and keep the assertion. Rebuild the release test and interleave the three modes for 80 fresh processes each against the same immediate DLL.

Run stock initialization from a fresh private extraction of the pinned package. Build the pushed baseline in a separate checkout and repeat the commands with the baseline and Milestone 14 DLLs, saving each output separately. Use the CUDA 12 plugin in a separate process for the static import comparison. Keep the system DLL search paths intact.

```powershell
$shim = 'C:/path/to/target/release/nvcuda.dll'
$base = Join-Path $PWD 'lib/ollama'
$originalPath = $env:Path
Copy-Item -LiteralPath $shim -Destination (Join-Path $base 'nvcuda.dll')
$env:CUDA_FOR_INTEL_TRACE = '1'
$env:OLLAMA_VULKAN = '0'
foreach ($variant in 'cuda_v12', 'cuda_v13') {
    $cuda = Join-Path $base $variant
    $env:Path = "$cuda;$originalPath"
    $env:GGML_BACKEND_PATH = Join-Path $cuda 'ggml-cuda.dll'
    & (Join-Path $base 'llama-server.exe') --list-devices 2>&1
}
llvm-objdump -p (Join-Path $base 'cuda_v12/ggml-cuda.dll')
llvm-objdump -p $shim
$env:Path = $originalPath
Remove-Item Env:CUDA_FOR_INTEL_TRACE, Env:GGML_BACKEND_PATH, Env:OLLAMA_VULKAN
```

For the parser probe, create a standalone Cargo binary with a path dependency on `ptx_parser` in the pinned ZLUDA checkout. Read each `examples/*.ptx` file and `tests/gpu/multi_entry.ptx`, call `ptx_parser::parse_module_checked(&source)`, and require `Ok` for each input.
