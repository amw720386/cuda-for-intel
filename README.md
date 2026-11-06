# CUDA for Intel

An independent Rust implementation of a small CUDA Driver API subset for Intel GPUs. The first milestone runs precompiled CUDA PTX on Intel Iris Xe (Alder Lake) under Windows x64.

```text
CUDA Driver API → PTX parser and SPIR-V emitter → Level Zero → Intel GPU
```

The frontend owns CUDA handles and per-thread current context state. The translator emits SPIR-V 1.0 for the OpenCL kernel environment and has no GPU model dependencies. The Level Zero backend queries the installed driver for GPU capabilities and passes SPIR-V to the driver's compiler.

## Build and verify

Install a Rust toolchain with a Windows x64 linker. GPU execution requires the Intel graphics driver and the system Level Zero loader. The build emits `target/debug/nvcuda.dll`. Cargo builds and hardware-independent tests need no CUDA toolkit or Intel GPU.

```powershell
cargo build --locked
cargo test --locked
cargo run --locked --example devices
cargo test --locked --test gpu -- --ignored --nocapture
```

The opt-in GPU tests load the built DLL by absolute path. They verify a device-memory round trip and run the same precompiled PTX with two block sizes, comparing 1,003 FP32 sums and checking that 21 guard elements remain untouched. This was verified on the Intel Core i5-1235U's Iris Xe (PCI device `8086:46a8`, Windows graphics driver `32.0.101.7085`). The Linux loader path is included for portability. GPU validation currently covers Windows x64 on that adapter.

Clang 23.1.1 with the NVPTX backend generated the checked-in PTX from
[vector_add.cu](examples/vector_add.cu). Regenerate it with the following command.

```powershell
clang --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -O2 -S examples/vector_add.cu -o examples/vector_add.ptx
```

## Supported subset

| Area | Supported entry points and behavior |
| --- | --- |
| Discovery | `cuInit` with zero flags, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem_v2`, `cuDeviceGetAttribute` for workgroup and grid dimensions, threads per block, and shared local memory limit |
| Contexts | `cuCtxCreate_v2` with zero flags, `cuCtxDestroy_v2`, `cuCtxGetCurrent`, `cuCtxSetCurrent`, `cuCtxGetDevice`, `cuCtxSynchronize` |
| Memory | `cuMemAlloc_v2`, `cuMemFree_v2`, blocking `cuMemcpyHtoD_v2` and `cuMemcpyDtoH_v2` within a single allocation owned by the current context |
| Modules | `cuModuleLoad` from a UTF-8 path, `cuModuleLoadData` from NUL-terminated PTX text, `cuModuleGetFunction`, `cuModuleUnload` |
| Launch | `cuLaunchKernel` with the null stream, an array of pointers to argument values, zero dynamic shared memory, and a null `extra` pointer. Launches complete synchronously. |

The translator accepts one PTX 4.1 entry with 64-bit addressing and scalar
32- or 64-bit parameters. Pointer parameters must use `.ptr` and point into a
device allocation in the current context. Supported instructions cover parameter
loads, `cvta.to.global.u64`, reads of `%ctaid`, `%ntid`, and `%tid`, low 32-bit
multiply-add, unsigned wide 32-bit multiply, 64-bit add, unsigned 32-bit
greater-or-equal comparison, one predicated branch to the final return, 32-bit
global loads and stores, and round-to-nearest FP32 add. Callers must provide
correctly aligned buffers large enough for the kernel's accesses.

Cubins, fatbins, arbitrary PTX control flow, warp operations, shared-memory
operations, streams, events, primary contexts, dynamic symbol lookup via
`cuGetProcAddress`, the CUDA Runtime API and CUDA libraries are outside this
milestone. Set `CUDA_FOR_INTEL_LOG=1` to print translation diagnostics and
Level Zero failures to stderr.

## References and license

MIT licensed. The implementation draws architectural inspiration from
[Intel-era ZLUDA](https://github.com/em-winterschon/ZLUDA-CUDA-on-Intel-GPU),
its [upstream v1 snapshot](https://github.com/vosen/ZLUDA/tree/v1),
[modern ZLUDA](https://github.com/vosen/ZLUDA), and
[CUDA for AMD on Windows](https://github.com/Speedstu/CUDA-for-AMD-Windows).
The Level Zero ABI declarations are attributed in [LICENSE](LICENSE).

The implementation uses the [CUDA Driver API](https://docs.nvidia.com/cuda/cuda-driver-api/),
[PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/),
[OpenCL SPIR-V environment](https://registry.khronos.org/OpenCL/specs/unified/html/OpenCL_Env.html),
and [Level Zero core specification](https://oneapi-src.github.io/level-zero-spec/level-zero/latest/core/PROG.html).
[Intel Compute Runtime](https://github.com/intel/compute-runtime) and
[Intel Graphics Compiler](https://github.com/intel/intel-graphics-compiler)
describe the driver and compiler stack.
