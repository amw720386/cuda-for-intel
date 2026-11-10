//! CUDA Driver Virtual Memory Management on Level Zero virtual memory.
//!
//! `cuMemCreate` makes Level Zero physical device memory, `cuMemAddressReserve` reserves
//! virtual addresses, `cuMemMap` maps one into the other and `cuMemSetAccess` makes the range
//! usable. Accessible ranges are entered into the context's allocation table as coalesced
//! runs, so copies, fills, kernel pointer checks and pointer queries treat a contiguous VMM
//! range (for example GGML's growing pool, several chunks mapped back to back) like one
//! `cuMemAlloc` allocation. `cuMemFree` rejects them.
//!
//! Objects belong to the context that is current when they are created (normally the
//! primary context). Mapping a handle into a reservation of another context returns
//! `CUDA_ERROR_INVALID_VALUE`. Shareable (IPC/fabric) handles, host-located memory,
//! compression and peer access through `cuMemSetAccess` are not supported.
//!
//! `CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED` reports 1 only when the loader
//! exports the Level Zero entry points and `CUDA_FOR_INTEL_VMM=1`: applications that probe
//! the attribute (GGML, PyTorch expandable segments) keep their ordinary allocators until the
//! path has been validated on hardware. The entry points themselves always work when the
//! loader supports them.
use super::{
    CuResult::{self, *},
    CudaContext, backend_error, graphs, status, streams, trace, with_context,
};
use crate::level_zero::{Api, Context, Handle, vmm::Access};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::ffi::{c_int, c_ulonglong};
use std::sync::atomic::{AtomicU64, Ordering};

type CUdeviceptr = u64;
type CUmemGenericAllocationHandle = u64;

/// NVIDIA's minimum VMM granularity. Reported unless the driver's page is larger, so
/// applications that hard-code 2 MiB rounding stay valid.
const CUDA_GRANULARITY: usize = 2 << 20;

/// `CUmemAllocationType`
const ALLOCATION_TYPE_PINNED: c_int = 1;
/// `CUmemLocationType`
const LOCATION_DEVICE: c_int = 1;
/// `CUmemAccess_flags`
const ACCESS_NONE: c_int = 0;
const ACCESS_READ: c_int = 1;
const ACCESS_READWRITE: c_int = 3;

/// `CUmemLocation`
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CUmemLocation {
    pub kind: c_int,
    pub id: c_int,
}

/// `CUmemAllocationProp` (32 bytes on 64-bit targets).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUmemAllocationProp {
    pub kind: c_int,
    pub requested_handle_types: c_int,
    pub location: CUmemLocation,
    pub win32_handle_meta_data: *mut std::ffi::c_void,
    pub compression_type: u8,
    pub gpu_direct_rdma_capable: u8,
    pub usage: u16,
    pub reserved: [u8; 4],
}

/// `CUmemAccessDesc` (12 bytes: arrays of it are packed at that stride).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct CUmemAccessDesc {
    pub location: CUmemLocation,
    pub flags: c_int,
}

const _: () = assert!(size_of::<CUmemAllocationProp>() == 32);
const _: () = assert!(size_of::<CUmemAccessDesc>() == 12);

static NEXT_HANDLE: AtomicU64 = AtomicU64::new(0x5600_0000_0001);

struct Physical {
    memory: Handle,
    size: usize,
    /// cuMemCreate and each cuMemRetainAllocationHandle add one, cuMemRelease removes one.
    refs: usize,
    /// Live mappings; the Level Zero memory is destroyed when both counts reach zero.
    maps: usize,
    /// (offset, size) ranges already made resident by a copy-engine fill.
    touched: Vec<(usize, usize)>,
}

struct Reservation {
    /// What Level Zero reserved (larger than `size` when re-reserved for alignment).
    raw_base: usize,
    raw_size: usize,
    size: usize,
}

#[derive(Clone, Copy)]
struct Mapping {
    size: usize,
    handle: CUmemGenericAllocationHandle,
    offset: usize,
    access: c_int,
}

#[derive(Default)]
pub(crate) struct State {
    physical: HashMap<CUmemGenericAllocationHandle, Physical>,
    reservations: BTreeMap<usize, Reservation>,
    mappings: BTreeMap<usize, Mapping>,
    /// Allocation-table entries that are coalesced accessible VMM runs.
    pub(crate) runs: BTreeSet<usize>,
}

impl State {
    fn reservation_of(&self, base: usize, size: usize) -> Option<(usize, usize)> {
        let (&start, r) = self.reservations.range(..=base).next_back()?;
        let offset = base - start;
        (offset < r.size && size <= r.size - offset).then_some((start, r.size))
    }

    fn overlapping(&self, base: usize, size: usize) -> Vec<(usize, Mapping)> {
        let end = base.saturating_add(size);
        self.mappings
            .range(..end)
            .filter(|&(&start, m)| start + m.size > base)
            .map(|(&start, &m)| (start, m))
            .collect()
    }

    /// The mappings that tile [base, base + size) exactly, or `None` on a gap or a
    /// mapping that sticks out of the range.
    fn tiling(&self, base: usize, size: usize) -> Option<Vec<(usize, Mapping)>> {
        let hits = self.overlapping(base, size);
        let mut cursor = base;
        for &(start, m) in &hits {
            if start != cursor {
                return None;
            }
            cursor = start + m.size;
        }
        (cursor == base + size).then_some(hits)
    }
}

fn granularity(api: &Api, backend: &Context) -> Result<usize, CuResult> {
    let small = backend.vmm_page_size(api, 1).map_err(backend_error)?;
    let large = backend
        .vmm_page_size(api, CUDA_GRANULARITY)
        .map_err(backend_error)?;
    Ok(CUDA_GRANULARITY.max(small).max(large))
}

fn require_vmm(api: &Api) -> Result<(), CuResult> {
    api.vmm.map(|_| ()).ok_or(NotSupported)
}

/// True when the device attribute should advertise VMM (see the module comment).
pub(crate) fn advertised(api: &Api) -> bool {
    api.vmm.is_some() && std::env::var("CUDA_FOR_INTEL_VMM").is_ok_and(|v| v == "1")
}

/// Rebuild the allocation-table runs inside the reservation that starts at `start`.
fn refresh_runs(context: &mut CudaContext, start: usize) {
    let Some(size) = context.vmm.reservations.get(&start).map(|r| r.size) else {
        return;
    };
    let end = start + size;
    let stale: Vec<usize> = context.vmm.runs.range(start..end).copied().collect();
    for base in stale {
        context.vmm.runs.remove(&base);
        context.allocations.remove(&base);
    }
    let mut run: Option<(usize, usize)> = None;
    let mut runs = Vec::new();
    for (&base, m) in context.vmm.mappings.range(start..end) {
        if m.access == ACCESS_NONE {
            runs.extend(run.take());
            continue;
        }
        run = match run {
            Some((b, s)) if b + s == base => Some((b, s + m.size)),
            other => {
                runs.extend(other);
                Some((base, m.size))
            }
        };
    }
    runs.extend(run);
    for (base, size) in runs {
        context.allocations.insert(base, size);
        context.vmm.runs.insert(base);
    }
}

fn destroy_if_unused(
    api: &Api,
    context: &mut CudaContext,
    handle: CUmemGenericAllocationHandle,
) -> Result<(), CuResult> {
    let Some(p) = context.vmm.physical.get(&handle) else {
        return Ok(());
    };
    if p.refs > 0 || p.maps > 0 {
        return Ok(());
    }
    context
        .backend
        .physical_destroy(api, p.memory)
        .inspect_err(|e| context.backend.health.observe(e))
        .map_err(backend_error)?;
    context.vmm.physical.remove(&handle);
    Ok(())
}

impl CudaContext {
    /// Unmap, destroy and unreserve everything at context teardown.
    pub(super) fn destroy_vmm(&mut self, api: &Api) -> Result<(), CuResult> {
        while let Some((&base, &m)) = self.vmm.mappings.first_key_value() {
            self.backend
                .vmm_unmap(api, base, m.size)
                .map_err(backend_error)?;
            self.vmm.mappings.remove(&base);
            if let Some(p) = self.vmm.physical.get_mut(&m.handle) {
                p.maps -= 1;
            }
        }
        for base in std::mem::take(&mut self.vmm.runs) {
            self.allocations.remove(&base);
        }
        let handles: Vec<_> = self.vmm.physical.keys().copied().collect();
        for handle in handles {
            if let Some(p) = self.vmm.physical.get_mut(&handle) {
                p.refs = 0;
            }
            destroy_if_unused(api, self, handle)?;
        }
        while let Some((&base, r)) = self.vmm.reservations.first_key_value() {
            let (raw_base, raw_size) = (r.raw_base, r.raw_size);
            self.backend
                .vmm_free(api, raw_base, raw_size)
                .map_err(backend_error)?;
            self.vmm.reservations.remove(&base);
        }
        Ok(())
    }
}

fn read_prop(prop: *const CUmemAllocationProp) -> Result<CUmemAllocationProp, CuResult> {
    if prop.is_null() {
        return Err(InvalidValue);
    }
    Ok(unsafe { prop.read_unaligned() })
}

/// Validates the properties this implementation can honour for `context`.
fn check_prop(context: &CudaContext, prop: &CUmemAllocationProp) -> Result<(), CuResult> {
    if prop.kind != ALLOCATION_TYPE_PINNED || prop.location.kind != LOCATION_DEVICE {
        return Err(if prop.kind == ALLOCATION_TYPE_PINNED {
            NotSupported
        } else {
            InvalidValue
        });
    }
    if prop.location.id != context.device {
        return Err(InvalidDevice);
    }
    // Shareable handles (POSIX fd, Win32, fabric) and compression need driver interop
    // this backend does not have.
    if prop.requested_handle_types != 0 || prop.compression_type != 0 {
        return Err(NotSupported);
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemGetAllocationGranularity(
    out: *mut usize,
    prop: *const CUmemAllocationProp,
    option: c_int,
) -> CuResult {
    trace::call("cuMemGetAllocationGranularity", || {
        status(with_context(|runtime, context| {
            require_vmm(&runtime.api)?;
            if out.is_null() || !matches!(option, 0 | 1) {
                return Err(InvalidValue);
            }
            check_prop(context, &read_prop(prop)?)?;
            let value = granularity(&runtime.api, &context.backend)?;
            unsafe { out.write(value) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemCreate(
    handle: *mut CUmemGenericAllocationHandle,
    size: usize,
    prop: *const CUmemAllocationProp,
    flags: c_ulonglong,
) -> CuResult {
    trace::call("cuMemCreate", || {
        if let Err(error) = graphs::unsafe_call() {
            return error;
        }
        status(with_context(|runtime, context| {
            require_vmm(&runtime.api)?;
            if handle.is_null() || flags != 0 || size == 0 {
                return Err(InvalidValue);
            }
            check_prop(context, &read_prop(prop)?)?;
            let page = granularity(&runtime.api, &context.backend)?;
            if !size.is_multiple_of(page) {
                return Err(InvalidValue);
            }
            let memory = context
                .backend
                .physical_create(&runtime.api, size)
                .inspect_err(|e| context.backend.health.observe(e))
                .map_err(backend_error)?;
            let id = NEXT_HANDLE.fetch_add(1, Ordering::Relaxed);
            context.vmm.physical.insert(
                id,
                Physical {
                    memory,
                    size,
                    refs: 1,
                    maps: 0,
                    touched: Vec::new(),
                },
            );
            unsafe { handle.write(id) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemRelease(handle: CUmemGenericAllocationHandle) -> CuResult {
    trace::call("cuMemRelease", || {
        status(with_context(|runtime, context| {
            let p = context.vmm.physical.get_mut(&handle).ok_or(InvalidValue)?;
            if p.refs == 0 {
                return Err(InvalidValue);
            }
            p.refs -= 1;
            destroy_if_unused(&runtime.api, context, handle)
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemAddressReserve(
    ptr: *mut CUdeviceptr,
    size: usize,
    alignment: usize,
    addr: CUdeviceptr,
    flags: c_ulonglong,
) -> CuResult {
    trace::call("cuMemAddressReserve", || {
        status(with_context(|runtime, context| {
            require_vmm(&runtime.api)?;
            if ptr.is_null() || size == 0 || flags != 0 {
                return Err(InvalidValue);
            }
            if alignment != 0 && !alignment.is_power_of_two() {
                return Err(InvalidValue);
            }
            let page = granularity(&runtime.api, &context.backend)?;
            if !size.is_multiple_of(page) {
                return Err(InvalidValue);
            }
            let align = alignment.max(page);
            let api = &runtime.api;
            let backend = &context.backend;
            let hint = if (addr as usize).is_multiple_of(align) {
                addr as usize
            } else {
                0
            };
            let raw = backend
                .vmm_reserve(api, hint, size)
                .map_err(backend_error)?;
            let (raw_base, raw_size, base) = if raw.is_multiple_of(align) {
                (raw, size, raw)
            } else {
                backend.vmm_free(api, raw, size).map_err(backend_error)?;
                let padded = size.checked_add(align).ok_or(OutOfMemory)?;
                let raw = backend.vmm_reserve(api, 0, padded).map_err(backend_error)?;
                (raw, padded, raw.next_multiple_of(align))
            };
            context.vmm.reservations.insert(
                base,
                Reservation {
                    raw_base,
                    raw_size,
                    size,
                },
            );
            unsafe { ptr.write(base as u64) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemAddressFree(ptr: CUdeviceptr, size: usize) -> CuResult {
    trace::call("cuMemAddressFree", || {
        status(with_context(|runtime, context| {
            let base = ptr as usize;
            let r = context.vmm.reservations.get(&base).ok_or(InvalidValue)?;
            if r.size != size || !context.vmm.overlapping(base, size).is_empty() {
                return Err(InvalidValue);
            }
            let (raw_base, raw_size) = (r.raw_base, r.raw_size);
            context
                .backend
                .vmm_free(&runtime.api, raw_base, raw_size)
                .map_err(backend_error)?;
            context.vmm.reservations.remove(&base);
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemMap(
    ptr: CUdeviceptr,
    size: usize,
    offset: usize,
    handle: CUmemGenericAllocationHandle,
    flags: c_ulonglong,
) -> CuResult {
    trace::call("cuMemMap", || {
        status(with_context(|runtime, context| {
            require_vmm(&runtime.api)?;
            let base = ptr as usize;
            if size == 0 || flags != 0 {
                return Err(InvalidValue);
            }
            let page = granularity(&runtime.api, &context.backend)?;
            if !base.is_multiple_of(page)
                || !size.is_multiple_of(page)
                || !offset.is_multiple_of(page)
            {
                return Err(InvalidValue);
            }
            let p = context.vmm.physical.get(&handle).ok_or(InvalidValue)?;
            if p.refs == 0 || offset > p.size || size > p.size - offset {
                return Err(InvalidValue);
            }
            let memory = p.memory;
            let (start, _) = context.vmm.reservation_of(base, size).ok_or(InvalidValue)?;
            if !context.vmm.overlapping(base, size).is_empty() {
                return Err(InvalidValue);
            }
            // Level Zero applies the page protection at map time; CUDA access stays off
            // until cuMemSetAccess, which the allocation table enforces.
            context
                .backend
                .vmm_map(&runtime.api, base, size, memory, offset, Access::ReadWrite)
                .inspect_err(|e| context.backend.health.observe(e))
                .map_err(backend_error)?;
            context.vmm.mappings.insert(
                base,
                Mapping {
                    size,
                    handle,
                    offset,
                    access: ACCESS_NONE,
                },
            );
            if let Some(p) = context.vmm.physical.get_mut(&handle) {
                p.maps += 1;
            }
            refresh_runs(context, start);
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemUnmap(ptr: CUdeviceptr, size: usize) -> CuResult {
    trace::call("cuMemUnmap", || {
        if let Err(error) = graphs::unsafe_call() {
            return error;
        }
        streams::drain_unlocked(None);
        status(with_context(|runtime, context| {
            let base = ptr as usize;
            let (start, _) = context.vmm.reservation_of(base, size).ok_or(InvalidValue)?;
            let hits = context.vmm.overlapping(base, size);
            if hits.is_empty()
                || hits
                    .iter()
                    .any(|&(s, m)| s < base || s + m.size > base + size)
            {
                return Err(InvalidValue);
            }
            context.streams.wait_all(&runtime.api)?;
            for (s, m) in hits {
                context
                    .backend
                    .vmm_unmap(&runtime.api, s, m.size)
                    .inspect_err(|e| context.backend.health.observe(e))
                    .map_err(backend_error)?;
                context.vmm.mappings.remove(&s);
                if let Some(p) = context.vmm.physical.get_mut(&m.handle) {
                    p.maps -= 1;
                }
                destroy_if_unused(&runtime.api, context, m.handle)?;
            }
            refresh_runs(context, start);
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemSetAccess(
    ptr: CUdeviceptr,
    size: usize,
    desc: *const CUmemAccessDesc,
    count: usize,
) -> CuResult {
    trace::call("cuMemSetAccess", || {
        let fills = with_context(|runtime, context| {
            require_vmm(&runtime.api)?;
            if desc.is_null() || count == 0 {
                return Err(InvalidValue);
            }
            let descs = unsafe { std::slice::from_raw_parts(desc, count) };
            let mut access = None;
            for d in descs {
                if d.location.kind != LOCATION_DEVICE
                    || !matches!(d.flags, ACCESS_NONE | ACCESS_READ | ACCESS_READWRITE)
                {
                    return Err(InvalidValue);
                }
                if d.location.id < 0 || d.location.id as usize >= runtime.devices.len() {
                    return Err(InvalidDevice);
                }
                if d.location.id != context.device {
                    // Peer mappings need a Level Zero context spanning both devices.
                    return Err(NotSupported);
                }
                access = Some(d.flags);
            }
            let access = access.ok_or(InvalidValue)?;
            let base = ptr as usize;
            let (start, _) = context.vmm.reservation_of(base, size).ok_or(InvalidValue)?;
            let tiles = context.vmm.tiling(base, size).ok_or(InvalidValue)?;
            let native = match access {
                ACCESS_READWRITE => Access::ReadWrite,
                ACCESS_READ => Access::ReadOnly,
                _ => Access::None,
            };
            let mut fills = Vec::new();
            for (s, m) in tiles {
                if let Some(entry) = context.vmm.mappings.get_mut(&s) {
                    entry.access = access;
                }
                // First access to physical pages: make them resident with a copy-engine
                // fill, as cuMemAlloc does (a kernel's first touch of fresh Level Zero
                // device memory hangs Windows GPUs). Remapped pages keep their contents.
                if access == ACCESS_NONE || std::env::var_os("CFI_NO_ALLOC_TOUCH").is_some() {
                    continue;
                }
                if let Some(p) = context.vmm.physical.get_mut(&m.handle) {
                    let covered = p
                        .touched
                        .iter()
                        .any(|&(o, l)| o <= m.offset && m.offset + m.size <= o + l);
                    if !covered {
                        p.touched.push((m.offset, m.size));
                        fills.push((s, m.size));
                    }
                }
            }
            // A read-only range is filled while its pages are still writable (as mapped),
            // and protected afterwards.
            if fills.is_empty() || native == Access::ReadWrite {
                set_native_access(&runtime.api, context, base, size, native)?;
            }
            refresh_runs(context, start);
            Ok((fills, native))
        });
        let (fills, native) = match fills {
            Ok(fills) => fills,
            Err(error) => return error,
        };
        if fills.is_empty() {
            return Success;
        }
        for &(base, size) in &fills {
            let result = super::memset_bytes(base as u64, 0, size, std::ptr::null_mut());
            if result != Success {
                return result;
            }
        }
        streams::drain_unlocked(None);
        if native == Access::ReadWrite {
            return Success;
        }
        status(with_context(|runtime, context| {
            set_native_access(&runtime.api, context, ptr as usize, size, native)
        }))
    })
}

fn set_native_access(
    api: &Api,
    context: &CudaContext,
    base: usize,
    size: usize,
    access: Access,
) -> Result<(), CuResult> {
    context
        .backend
        .vmm_set_access(api, base, size, access)
        .inspect_err(|e| context.backend.health.observe(e))
        .map_err(backend_error)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemGetAccess(
    flags: *mut c_ulonglong,
    location: *const CUmemLocation,
    ptr: CUdeviceptr,
) -> CuResult {
    trace::call("cuMemGetAccess", || {
        status(with_context(|_, context| {
            if flags.is_null() || location.is_null() {
                return Err(InvalidValue);
            }
            let location = unsafe { location.read_unaligned() };
            if location.kind != LOCATION_DEVICE {
                return Err(InvalidValue);
            }
            let hits = context.vmm.overlapping(ptr as usize, 1);
            let (_, m) = hits.first().ok_or(InvalidValue)?;
            let value = if location.id == context.device {
                m.access
            } else {
                ACCESS_NONE
            };
            unsafe { flags.write(value as c_ulonglong) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemGetAllocationPropertiesFromHandle(
    prop: *mut CUmemAllocationProp,
    handle: CUmemGenericAllocationHandle,
) -> CuResult {
    trace::call("cuMemGetAllocationPropertiesFromHandle", || {
        status(with_context(|_, context| {
            if prop.is_null() || !context.vmm.physical.contains_key(&handle) {
                return Err(InvalidValue);
            }
            let value = CUmemAllocationProp {
                kind: ALLOCATION_TYPE_PINNED,
                requested_handle_types: 0,
                location: CUmemLocation {
                    kind: LOCATION_DEVICE,
                    id: context.device,
                },
                win32_handle_meta_data: std::ptr::null_mut(),
                compression_type: 0,
                gpu_direct_rdma_capable: 0,
                usage: 0,
                reserved: [0; 4],
            };
            unsafe { prop.write_unaligned(value) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemRetainAllocationHandle(
    handle: *mut CUmemGenericAllocationHandle,
    addr: *mut std::ffi::c_void,
) -> CuResult {
    trace::call("cuMemRetainAllocationHandle", || {
        status(with_context(|_, context| {
            if handle.is_null() {
                return Err(InvalidValue);
            }
            let hits = context.vmm.overlapping(addr.addr(), 1);
            let (_, m) = hits.first().ok_or(InvalidValue)?;
            let p = context
                .vmm
                .physical
                .get_mut(&m.handle)
                .ok_or(InvalidValue)?;
            p.refs += 1;
            unsafe { handle.write(m.handle) };
            Ok(())
        }))
    })
}

/// `cuMemExportToShareableHandle` / `cuMemImportFromShareableHandle`: no interop handles.
#[unsafe(no_mangle)]
pub extern "system" fn cuMemExportToShareableHandle(
    _out: *mut std::ffi::c_void,
    _handle: CUmemGenericAllocationHandle,
    _kind: c_int,
    _flags: c_ulonglong,
) -> CuResult {
    NotSupported
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemImportFromShareableHandle(
    _handle: *mut CUmemGenericAllocationHandle,
    _os: *mut std::ffi::c_void,
    _kind: c_int,
) -> CuResult {
    NotSupported
}
