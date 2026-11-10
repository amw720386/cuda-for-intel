//! CUDA VMM (`cuMemCreate` / `cuMemAddressReserve` / `cuMemMap` / `cuMemSetAccess`) on the
//! fake Level Zero backend, which checks the driver's alignment and lifetime preconditions.
use super::fake_tests::{Fixture, assert_baseline, context};
use crate::cuda::gaps::cuMemGetAddressRange_v2;
use crate::cuda::vmm::*;
use crate::cuda::*;
use crate::level_zero::fake::{self, Kind};
use std::ptr::null_mut;

const MIB: usize = 1 << 20;
const GRAN: usize = 2 * MIB;

fn prop(device: i32) -> CUmemAllocationProp {
    CUmemAllocationProp {
        kind: 1,
        requested_handle_types: 0,
        location: CUmemLocation {
            kind: 1,
            id: device,
        },
        win32_handle_meta_data: null_mut(),
        compression_type: 0,
        gpu_direct_rdma_capable: 0,
        usage: 0,
        reserved: [0; 4],
    }
}

fn rw(device: i32) -> CUmemAccessDesc {
    CUmemAccessDesc {
        location: CUmemLocation {
            kind: 1,
            id: device,
        },
        flags: 3,
    }
}

fn create(size: usize) -> u64 {
    let mut handle = 0;
    assert_eq!(
        unsafe { cuMemCreate(&mut handle, size, &prop(0), 0) },
        Success
    );
    assert_ne!(handle, 0);
    handle
}

fn reserve(size: usize, alignment: usize) -> u64 {
    let mut base = 0;
    assert_eq!(
        unsafe { cuMemAddressReserve(&mut base, size, alignment, 0, 0) },
        Success
    );
    base
}

fn range_of(pointer: u64) -> Option<(u64, usize)> {
    let (mut base, mut size) = (0, 0);
    (unsafe { cuMemGetAddressRange_v2(&mut base, &mut size, pointer) } == Success)
        .then_some((base, size))
}

fn assert_clean() {
    let snapshot = fake::vmm_snapshot();
    assert_eq!(snapshot, fake::VmmSnapshot::default(), "{snapshot:?}");
    assert!(
        fake::snapshot().ledger.violations.is_empty(),
        "{:?}",
        fake::snapshot()
    );
}

#[test]
fn granularity_is_cuda_minimum_and_rejects_unsupported_properties() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let mut value = 0;
    assert_eq!(
        unsafe { cuMemGetAllocationGranularity(&mut value, &prop(0), 0) },
        Success
    );
    assert_eq!(value, GRAN);
    assert_eq!(
        unsafe { cuMemGetAllocationGranularity(&mut value, &prop(0), 1) },
        Success
    );
    assert_eq!(value, GRAN);
    assert_eq!(
        unsafe { cuMemGetAllocationGranularity(&mut value, &prop(0), 2) },
        InvalidValue
    );
    let mut shareable = prop(0);
    shareable.requested_handle_types = 1; // POSIX file descriptor
    assert_eq!(
        unsafe { cuMemGetAllocationGranularity(&mut value, &shareable, 0) },
        NotSupported
    );
    let mut handle = 0;
    assert_eq!(
        unsafe { cuMemCreate(&mut handle, GRAN, &shareable, 0) },
        NotSupported
    );
    // The current context is on device 0; physical memory for device 1 needs its context.
    assert_eq!(
        unsafe { cuMemCreate(&mut handle, GRAN, &prop(1), 0) },
        InvalidDevice
    );
    assert_eq!(
        unsafe { cuMemCreate(&mut handle, GRAN + 4096, &prop(0), 0) },
        InvalidValue
    );
    let mut host = prop(0);
    host.location.kind = 2; // CU_MEM_LOCATION_TYPE_HOST
    assert_eq!(
        unsafe { cuMemCreate(&mut handle, GRAN, &host, 0) },
        NotSupported
    );
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

/// GGML's VMM pool: one large reservation, physical chunks mapped back to back as the pool
/// grows, access granted per chunk, then everything unmapped and released.
#[test]
fn ggml_pool_growth_forms_one_contiguous_device_range() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let base = reserve(64 * MIB, 0);
    assert_eq!(base as usize % GRAN, 0);
    let first = create(4 * MIB);
    let second = create(2 * MIB);
    assert_eq!(unsafe { cuMemMap(base, 4 * MIB, 0, first, 0) }, Success);
    // Mapped but not yet accessible: CUDA rejects device use.
    assert_eq!(cuMemsetD8_v2(base, 1, 16), InvalidValue);
    assert_eq!(unsafe { cuMemSetAccess(base, 4 * MIB, &rw(0), 1) }, Success);
    assert_eq!(cuMemsetD8_v2(base, 1, 4 * MIB), Success);
    assert_eq!(cuMemsetD8_v2(base, 1, 4 * MIB + 1), InvalidValue);

    let grown = base + 4 * MIB as u64;
    assert_eq!(unsafe { cuMemMap(grown, 2 * MIB, 0, second, 0) }, Success);
    assert_eq!(
        unsafe { cuMemSetAccess(grown, 2 * MIB, &rw(0), 1) },
        Success
    );
    // A copy or fill may cross the chunk boundary.
    assert_eq!(cuMemsetD8_v2(base + MIB as u64, 7, 5 * MIB), Success);
    assert_eq!(range_of(grown + 16), Some((base, 6 * MIB)));
    // VMM ranges are not cuMemAlloc allocations.
    assert_eq!(cuMemFree_v2(base), InvalidValue);

    // The handles can be released while mapped; the memory lives until unmapped.
    assert_eq!(unsafe { cuMemRelease(first) }, Success);
    assert_eq!(unsafe { cuMemRelease(second) }, Success);
    assert_eq!(unsafe { cuMemRelease(second) }, InvalidValue);
    assert_eq!(fake::vmm_snapshot().physical.len(), 2);
    assert_eq!(
        unsafe { cuMemAddressFree(base, 64 * MIB) },
        InvalidValue,
        "still mapped"
    );
    // Unmapping must cover whole mappings.
    assert_eq!(unsafe { cuMemUnmap(base, 2 * MIB) }, InvalidValue);
    assert_eq!(unsafe { cuMemUnmap(base, 6 * MIB) }, Success);
    assert_eq!(cuMemsetD8_v2(base, 0, 1), InvalidValue);
    assert_eq!(range_of(base), None);
    assert!(fake::vmm_snapshot().physical.is_empty());
    assert_eq!(unsafe { cuMemAddressFree(base, 64 * MIB) }, Success);
    assert_eq!(unsafe { cuMemAddressFree(base, 64 * MIB) }, InvalidValue);
    assert_clean();
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn map_rejects_misuse_and_first_access_fills_once() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let base = reserve(8 * MIB, 0);
    let handle = create(4 * MIB);
    let fills = || {
        fake::copy_log()
            .iter()
            .filter(|&&(_, dst, src, _)| dst == base as usize && src == 0)
            .count()
    };
    assert_eq!(
        unsafe { cuMemMap(base + 4096, GRAN, 0, handle, 0) },
        InvalidValue,
        "unaligned"
    );
    assert_eq!(
        unsafe { cuMemMap(base, GRAN, 4096, handle, 0) },
        InvalidValue,
        "unaligned offset"
    );
    assert_eq!(
        unsafe { cuMemMap(base, 6 * MIB, 0, handle, 0) },
        InvalidValue,
        "beyond handle"
    );
    assert_eq!(
        unsafe { cuMemMap(base + 8 * MIB as u64, GRAN, 0, handle, 0) },
        InvalidValue
    );
    assert_eq!(
        unsafe { cuMemMap(base, GRAN, 0, handle, 1) },
        InvalidValue,
        "flags"
    );
    assert_eq!(
        unsafe { cuMemMap(base, GRAN, 0, 12345, 0) },
        InvalidValue,
        "unknown handle"
    );
    // The same physical pages mapped twice (aliasing) at different addresses.
    assert_eq!(unsafe { cuMemMap(base, GRAN, 0, handle, 0) }, Success);
    assert_eq!(
        unsafe { cuMemMap(base, GRAN, 0, handle, 0) },
        InvalidValue,
        "overlap"
    );
    assert_eq!(
        unsafe { cuMemSetAccess(base, 2 * GRAN, &rw(0), 1) },
        InvalidValue,
        "gap"
    );
    assert_eq!(
        unsafe { cuMemSetAccess(base, GRAN, &rw(1), 1) },
        NotSupported,
        "peer"
    );
    assert_eq!(unsafe { cuMemSetAccess(base, GRAN, &rw(0), 1) }, Success);
    assert_eq!(fills(), 1, "first access makes fresh pages resident");
    let alias = base + 4 * MIB as u64;
    assert_eq!(unsafe { cuMemMap(alias, GRAN, 0, handle, 0) }, Success);
    assert_eq!(unsafe { cuMemSetAccess(alias, GRAN, &rw(0), 1) }, Success);
    assert!(
        !fake::copy_log()
            .iter()
            .any(|&(_, dst, src, _)| dst == alias as usize && src == 0),
        "remapped pages keep their contents"
    );
    // Revoking access removes the range from device use; restoring it does not refill.
    let none = CUmemAccessDesc { flags: 0, ..rw(0) };
    assert_eq!(unsafe { cuMemSetAccess(base, GRAN, &none, 1) }, Success);
    assert_eq!(cuMemsetD8_v2(base, 0, 1), InvalidValue);
    assert_eq!(cuMemsetD8_v2(alias, 0, 1), Success);
    let mut flags = 9;
    let location = CUmemLocation { kind: 1, id: 0 };
    assert_eq!(
        unsafe { cuMemGetAccess(&mut flags, &location, base) },
        Success
    );
    assert_eq!(flags, 0);
    assert_eq!(unsafe { cuMemSetAccess(base, GRAN, &rw(0), 1) }, Success);
    assert_eq!(
        unsafe { cuMemGetAccess(&mut flags, &location, base) },
        Success
    );
    assert_eq!(flags, 3);
    assert_eq!(fills(), 1);
    // The Level Zero mapping access follows cuMemSetAccess.
    let accesses: Vec<u32> = fake::vmm_snapshot().mappings.iter().map(|m| m.4).collect();
    assert_eq!(accesses, vec![1, 1]);

    // Device memory in use counts the physical allocation once, not once per mapping.
    let used = || {
        let guard = RUNTIME.lock().unwrap();
        used_device_bytes(&guard.as_ref().unwrap().contexts, 0)
    };
    assert_eq!(used(), 4 * MIB);
    let spare = create(GRAN);
    assert_eq!(
        used(),
        4 * MIB + GRAN,
        "unmapped physical memory is in use too"
    );
    assert_eq!(unsafe { cuMemRelease(spare) }, Success);
    assert_eq!(used(), 4 * MIB);

    // Retain returns the handle behind an address and needs its own release.
    let mut retained = 0;
    assert_eq!(
        unsafe { cuMemRetainAllocationHandle(&mut retained, (alias + 8) as _) },
        Success
    );
    assert_eq!(retained, handle);
    let mut props = prop(1);
    assert_eq!(
        unsafe { cuMemGetAllocationPropertiesFromHandle(&mut props, handle) },
        Success
    );
    assert_eq!(
        (props.kind, props.location),
        (1, CUmemLocation { kind: 1, id: 0 })
    );
    assert_eq!(unsafe { cuMemRelease(handle) }, Success);
    assert_eq!(unsafe { cuMemRelease(retained) }, Success);
    assert_eq!(unsafe { cuMemUnmap(base, 8 * MIB) }, Success);
    assert!(fake::vmm_snapshot().physical.is_empty());
    assert_eq!(unsafe { cuMemAddressFree(base, 8 * MIB) }, Success);
    assert_clean();
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn reservations_honour_alignment_and_context_teardown_releases_everything() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    // The fake leaves a guard page after each reservation, so the second one is
    // misaligned until re-reserved with padding.
    let first = reserve(GRAN, 0);
    let second = reserve(GRAN, 4 * GRAN);
    assert_eq!(second as usize % (4 * GRAN), 0);
    assert_ne!(first, second);
    let mut out = 0;
    assert_eq!(
        unsafe { cuMemAddressReserve(&mut out, GRAN, 3 * MIB, 0, 0) },
        InvalidValue
    );
    assert_eq!(
        unsafe { cuMemAddressReserve(&mut out, GRAN + 1, 0, 0, 0) },
        InvalidValue
    );
    let handle = create(GRAN);
    assert_eq!(unsafe { cuMemMap(second, GRAN, 0, handle, 0) }, Success);
    assert_eq!(unsafe { cuMemSetAccess(second, GRAN, &rw(0), 1) }, Success);
    let leaked = create(GRAN);
    assert_eq!(fake::snapshot().live.get(&Kind::PhysicalMemory), Some(&2));
    // No unmap, release or free: context destruction cleans up in driver order.
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_ne!(leaked, handle);
    assert_clean();
    assert_baseline();
}

#[test]
fn vmm_attribute_is_opt_in() {
    let _fixture = Fixture::new();
    let mut value = -1;
    unsafe { std::env::remove_var("CUDA_FOR_INTEL_VMM") };
    assert_eq!(unsafe { cuDeviceGetAttribute(&mut value, 102, 0) }, Success);
    assert_eq!(value, 0);
    unsafe { std::env::set_var("CUDA_FOR_INTEL_VMM", "1") };
    assert_eq!(unsafe { cuDeviceGetAttribute(&mut value, 102, 0) }, Success);
    unsafe { std::env::remove_var("CUDA_FOR_INTEL_VMM") };
    assert_eq!(value, 1);
}

/// The exact call sequence of `ggml_cuda_pool_vmm` in Ollama v0.12.6 (ggml-cuda.cu lines
/// 506-600): physical memory created before the first reservation, a 32 GiB reservation,
/// each handle released right after cuMemMap, access set per growth, and teardown unmapping
/// the whole grown pool in one call. Release, reserve and unmap failures abort GGML
/// (`CU_CHECK`), so each must succeed here.
#[test]
fn ggml_v0_12_6_vmm_pool_sequence() {
    const MAX: usize = 1 << 35;
    let _fixture = Fixture::new();
    let ctx = context(None);
    let mut granularity = 0;
    assert_eq!(
        unsafe { cuMemGetAllocationGranularity(&mut granularity, &prop(0), 1) },
        Success
    );
    let (mut pool_addr, mut pool_size) = (0u64, 0usize);
    // Three growths of 1, 3 and 2 granules, as successive graph evaluations need more.
    for granules in [1, 3, 2] {
        let reserve_size = granules * granularity;
        let handle = create(reserve_size);
        if pool_addr == 0 {
            assert_eq!(
                unsafe { cuMemAddressReserve(&mut pool_addr, MAX, 0, 0, 0) },
                Success
            );
        }
        let start = pool_addr + pool_size as u64;
        assert_eq!(
            unsafe { cuMemMap(start, reserve_size, 0, handle, 0) },
            Success
        );
        assert_eq!(unsafe { cuMemRelease(handle) }, Success);
        assert_eq!(
            unsafe { cuMemSetAccess(start, reserve_size, &rw(0), 1) },
            Success
        );
        pool_size += reserve_size;
        // Tensors are carved from the pool at 128-byte alignment and may straddle growths.
        assert_eq!(cuMemsetD8_v2(pool_addr + 128, 3, pool_size - 128), Success);
        assert_eq!(
            range_of(pool_addr + pool_size as u64 - 1),
            Some((pool_addr, pool_size))
        );
    }
    assert_eq!(
        fake::vmm_snapshot().physical.len(),
        3,
        "released handles stay alive while mapped"
    );
    // The failure path GGML takes when cuMemSetAccess fails: unmap the released chunk.
    let handle = create(granularity);
    let start = pool_addr + pool_size as u64;
    assert_eq!(
        unsafe { cuMemMap(start, granularity, 0, handle, 0) },
        Success
    );
    assert_eq!(unsafe { cuMemRelease(handle) }, Success);
    assert_eq!(unsafe { cuMemUnmap(start, granularity) }, Success);
    assert_eq!(fake::vmm_snapshot().physical.len(), 3);
    // ~ggml_cuda_pool_vmm
    assert_eq!(unsafe { cuMemUnmap(pool_addr, pool_size) }, Success);
    assert_eq!(unsafe { cuMemAddressFree(pool_addr, MAX) }, Success);
    assert_clean();
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}
