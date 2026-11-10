# CUDA virtual memory management (VMM)

`src/cuda/vmm.rs` implements the CUDA Driver VMM family on Level Zero virtual memory
(`src/level_zero/vmm.rs`):

| CUDA | Level Zero | Notes |
| --- | --- | --- |
| `cuMemGetAllocationGranularity` | `zeVirtualMemQueryPageSize` | Reports max(2 MiB, driver page size) for both MINIMUM and RECOMMENDED, so code that hard-codes NVIDIA's 2 MiB stays valid |
| `cuMemCreate` / `cuMemRelease` | `zePhysicalMemCreate` / `zePhysicalMemDestroy` | Pinned device memory on the current context's device. Size must be a multiple of the granularity. Released handles live until their last mapping is unmapped |
| `cuMemAddressReserve` / `cuMemAddressFree` | `zeVirtualMemReserve` / `zeVirtualMemFree` | The hint is passed through; an alignment above the granularity is met by re-reserving with padding |
| `cuMemMap` / `cuMemUnmap` | `zeVirtualMemMap` / `zeVirtualMemUnmap` | Base, size and offset granularity-aligned, inside one reservation, no overlap. Unmap must cover whole mappings and waits for the context's streams first |
| `cuMemSetAccess` / `cuMemGetAccess` | `zeVirtualMemSetAccessAttribute` | READWRITE, READ or NONE for the context's own device; the range must be tiled exactly by mappings |
| `cuMemRetainAllocationHandle`, `cuMemGetAllocationPropertiesFromHandle` | (bookkeeping) | Retain adds a reference that needs its own `cuMemRelease` |

How the rest of the driver sees VMM memory: accessible mappings are entered into the
context's allocation table as coalesced runs. A contiguous range built from several chunks
(GGML's growing pool) is one device range for copies, fills, graph nodes, kernel pointer
checks and `cuMemGetAddressRange`/`cuPointerGetAttribute`. Mapped-but-not-yet-accessible
ranges are rejected with `CUDA_ERROR_INVALID_VALUE`, like on NVIDIA. `cuMemFree` on a VMM
address returns `CUDA_ERROR_INVALID_VALUE`. Context destruction unmaps, destroys and
unreserves whatever the application left.

The first `cuMemSetAccess` on fresh physical pages zero-fills them with the copy engine,
the same residency workaround `cuMemAlloc` uses (a kernel's first touch of a fresh Level
Zero allocation can hang the GPU on Windows). Remapping pages that were already touched
keeps their contents. `CFI_NO_ALLOC_TOUCH=1` disables both.

## Capability gating

`CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` (102) reports **0 unless
`CUDA_FOR_INTEL_VMM=1`** (and the loader exports all eight Level Zero entry points). GGML
(stock Ollama, llama.cpp) and PyTorch's expandable segments pick their VMM allocator from
this attribute, so default behaviour does not change until the hardware gate below passes on
both GPUs. The entry points work regardless, for applications that call them without
checking the attribute. If the loader lacks the Level Zero functions, every VMM call returns
`CUDA_ERROR_NOT_SUPPORTED` as before.

## Not supported

Shareable handles (`requestedHandleTypes` other than none, `cuMemExportToShareableHandle`,
`cuMemImportFromShareableHandle`, fabric), host or NUMA locations, compression, peer access
through `cuMemSetAccess` for another device, `cuMemMapArrayAsync` (sparse arrays), and using a
handle or reservation from a context other than the one that created it.

## Tests

- Fake backend (any OS): `cargo test --lib vmm_tests` (5 tests: granularity and property
  validation, the GGML pool-growth pattern across chunk boundaries, misuse rejections,
  first-access fill and aliasing, alignment, teardown cleanup, attribute opt-in). The fake
  driver enforces Level Zero's alignment, containment and lifetime rules and fails the test
  on any violation.
- Hardware: `cargo test --release --test gpu vmm_pool_growth -- --ignored --nocapture`
  ([tests/gpu/vmm.rs](../tests/gpu/vmm.rs)); procedure and pass criteria are item 9 of
  [gpu-day-checklist.md](gpu-day-checklist.md). No hardware result has been recorded yet.
