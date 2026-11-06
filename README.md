# CUDA for Intel

An independent Rust implementation of a small CUDA Driver API subset for Intel GPUs. Precompiled CUDA vector addition, shared-memory block reduction, and FP32 RMSNorm run on Intel Iris Xe (Alder Lake) under Windows x64.

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
cargo test --locked --lib gated_stream_stays_pending -- --ignored
cargo test --locked --lib gated_event_generations -- --ignored
```

The opt-in GPU tests load the built DLL by absolute path. They verify a memory round trip, vector addition, a 1,003-element block reduction at six block widths, and atomic contention across 23 workgroups. The reduction includes odd widths and partial final blocks. It checks bit-exact dyadic inputs, an FP32 tree-depth error bound for rounded inputs, and guard regions. GPU validation covers the Intel Core i5-1235U's Iris Xe (PCI device `8086:46a8`, Windows graphics driver `32.0.101.7085`). The Linux loader path is included for portability.

The RMSNorm regression compares four rows against an FP64 CPU reference at
hidden sizes 32, 96, 256, 896, and 1024, using block widths 32, 63, and 128.
It checks two epsilon values, guards, and determinism. A separate GPU test
checks the rounding of individual FP32 instructions.
The Forge-style test queues RMSNorm, residual addition, and row reduction on
one stream and on three event-linked streams. It compares both modes with an
FP64 CPU reference and checks guard regions after completion.
The pinned-memory pipeline also compares blocking, single-stream asynchronous,
and event-linked upload/compute/download with one or two reusable buffer sets.
The `devices` example reports Level Zero queue groups and timestamp capabilities.
On the tested Iris Xe driver, one queue group serves both compute and copies.
The runtime submits CUDA streams to that group.

With Khronos `spirv-val` on PATH or selected by `SPIRV_VAL`, validate the emitted OpenCL SPIR-V independently.

```powershell
cargo test --locked --lib validate_opencl_spirv -- --ignored
```

Clang 23.1.1 with the NVPTX backend generated the checked-in PTX fixtures.

```powershell
clang --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -O2 -S examples/vector_add.cu -o examples/vector_add.ptx
clang --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -O2 -fno-unroll-loops -S examples/block_reduce.cu -o examples/block_reduce.ptx
clang --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -O2 -S examples/atomic_tickets.cu -o examples/atomic_tickets.ptx
clang --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -O2 -fno-unroll-loops -ffp-contract=off -S examples/rmsnorm.cu -o examples/rmsnorm.ptx
clang --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -O2 -fno-unroll-loops -ffp-contract=off -S examples/forge_residual.cu -o examples/forge_residual.ptx
clang --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -O2 -fno-unroll-loops -ffp-contract=off -S examples/forge_reduce.cu -o examples/forge_reduce.ptx
```

## Supported subset

| Area | Supported entry points and behavior |
| --- | --- |
| Discovery | `cuInit` with zero flags, `cuDeviceGetCount`, `cuDeviceGet`, `cuDeviceGetName`, `cuDeviceTotalMem_v2`, `cuDeviceGetAttribute` for workgroup and grid dimensions, threads per block, and shared local memory limit |
| Contexts | `cuCtxCreate_v2` with zero flags, `cuCtxDestroy_v2`, `cuCtxGetCurrent`, `cuCtxSetCurrent`, `cuCtxGetDevice`, `cuCtxSynchronize` |
| Memory | `cuMemAlloc_v2`, `cuMemFree_v2`, blocking `cuMemcpyHtoD_v2` and `cuMemcpyDtoH_v2` within a single allocation owned by the current context |
| Pinned transfers | `cuMemHostAlloc` with default flags, `cuMemFreeHost`, `cuMemcpyHtoDAsync_v2`, and `cuMemcpyDtoHAsync_v2` |
| Modules | `cuModuleLoad` from a UTF-8 path, `cuModuleLoadData` from NUL-terminated PTX text, `cuModuleGetFunction`, `cuModuleUnload` |
| Streams | `cuStreamCreate` with default or nonblocking flags, `cuStreamDestroy_v2`, `cuStreamSynchronize`, and `cuStreamQuery` |
| Events | `cuEventCreate`, `cuEventRecord`, `cuEventQuery`, `cuEventSynchronize`, `cuEventDestroy_v2`, and `cuStreamWaitEvent` |
| Launch | Asynchronous `cuLaunchKernel` with a supported stream, an array of pointers to argument values, zero dynamic shared memory, and a null `extra` pointer |

By default, each stream submits in order to an asynchronous Level Zero immediate list.
On Windows, regular command-list batching is available for experiments. Build with
`--features experimental-batching` and set `CUDA_FOR_INTEL_BATCH_CAP` to a list
capacity from 1 to 1024. `CUDA_FOR_INTEL_BATCH_US` sets the worker's deadline in
microseconds and defaults to 250. Set `CUDA_FOR_INTEL_BATCH_TIMER=high-resolution`
to use a Windows high-resolution waitable timer (Windows 10 version 1803 or later).
The batching worker uses a condition variable by default. Both timer paths
flush on size, synchronization and producer dependencies as well as deadlines.
Actual wake times depend on the Windows scheduler.
The null stream and `CU_STREAM_LEGACY` order work against blocking streams
in the same context. Nonblocking streams are excluded from that ordering.
Wait explicitly before reading results from nonblocking streams. Blocking
copies follow legacy ordering and complete before returning. Context
synchronization waits for every stream, including destroyed streams with
queued work. Module unload and device free conservatively drain the context.
GPU overlap depends on the Level Zero driver and hardware scheduler.

Async copies require both ranges to fit inside live allocations of the
appropriate type in the current context. Allocate the host side with
`cuMemHostAlloc` and keep its contents available to the GPU until completion.
Zero-byte copies validate the context and stream and enqueue nothing. Host
allocations are 4 KiB aligned and require a positive size. Free them by base
pointer in their owning context. Host free drains that context, and context
destruction reclaims any remaining pinned allocations after queued work finishes.
Portable, mapped and write-combined host allocations, host registration, and
memory IPC are outside this subset. Use the blocking API for pageable buffers.

Events and their streams must belong to the current context. Creation accepts
`CU_EVENT_DEFAULT` and `CU_EVENT_DISABLE_TIMING`. Both provide synchronization
only, and stream waits accept zero flags. Query, synchronization, and stream
wait succeed immediately for an unrecorded event. Each stream wait captures
the latest recording at enqueue time and remains valid after re-recording or
destroying the event handle. Context destruction returns `CUDA_ERROR_NOT_READY`
while an explicit stream or event host wait is active.

The translator accepts one PTX 4.1 entry with 64-bit addressing and scalar
32- or 64-bit parameters. Pointer parameters must use `.ptr` and point into a
device allocation in the current context. The instruction subset covers parameter
loads, `cvta.to.global.u64`, reads of `%ctaid`, `%ntid`, and `%tid`, selected integer
arithmetic and comparisons, and 32-bit global loads and stores. FP32 support
includes scalar parameters, round-to-nearest add, multiply, fused multiply-add,
unsigned-to-float conversion, divide, and square root. Divide and square root
use integer significand arithmetic on the GPU. Explicit FMA remains fused and
separate multiply/add operations keep their own rounding steps. The compiler
handles direct and predicated branches with reconvergence,
including loops, and requires a definition on every incoming path for each read.
Each virtual register retains its scalar or address-space type.

An entry may declare one static `.shared .align 4 .b8` array and access it
with 32-bit shared loads and stores. The runtime checks its size against Level
Zero's local-memory limit and enforces `.maxntid` as a total threads-per-block
limit. `bar.sync 0` synchronizes all threads in a workgroup and orders their
shared and global accesses. Conditional paths that may bypass a barrier within
the workgroup are rejected. Callers provide aligned buffers with valid bounds.

The atomic subset is `atom.global.add.u32` on device allocations with
device-scope relaxed ordering. The runtime requires Level Zero to report
atomic access support for device allocations.

Cubins, fatbins, indirect branches, general predicated instructions, warp
operations, dynamic shared memory, approximate FP32 math, alternate rounding
modes, flush-to-zero modifiers, other barrier and atomic variants, per-thread
default streams, stream priorities, event timing, blocking-sync and IPC event
flags, cross-context event waits,
primary contexts, dynamic symbol lookup via
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
