//! Driver API entry points that stock CUDA 13 clients call beyond the core
//! allocation, module, stream and event set. Pure decisions live in `gap_math`.
//!
//! Honest limits: there is no managed memory, peer access, cluster launch,
//! cooperative launch, CUDA arrays or user memory pools, so those requests
//! return the error a CUDA driver without the feature returns.
use super::gap_math::{self as math, Kind, MemoryType, PointerFacts, Value};
use super::*;
use std::collections::BTreeMap;

// ---------------------------------------------------------------------------
// Functions: attributes, occupancy
// ---------------------------------------------------------------------------

fn limits(info: &DeviceInfo, facts: math::FuncFacts) -> math::Limits {
    let get = |attr, fallback: u32| attribute(info, attr).map_or(fallback, |value| value as u32);
    math::Limits {
        max_threads_per_block: facts.max_threads,
        max_threads_per_mp: get(39, info.max_workgroup_size),
        max_blocks_per_mp: get(106, 16),
        shared_per_mp: get(81, info.max_shared_local_memory),
        shared_per_block: info.max_shared_local_memory,
        warp_size: 32,
        multiprocessors: multiprocessor_count(info),
    }
}

fn function_facts(info: &DeviceInfo, entry: &CudaEntry) -> math::FuncFacts {
    math::FuncFacts {
        max_threads: entry
            .max_threads
            .unwrap_or(info.max_workgroup_size)
            .min(info.max_workgroup_size),
        max_dynamic_shared: info.max_shared_local_memory,
        device_max_shared: info.max_shared_local_memory,
    }
}

/// Resolve a CUfunction (or library CUkernel) in the current context.
fn with_function<T>(
    function: Opaque,
    action: impl FnOnce(&DeviceInfo, &CudaEntry) -> Result<T, CuResult>,
) -> Result<T, CuResult> {
    state(|state| {
        let mut function = function.addr();
        if state.libraries.contains_kernel(function) {
            function = library::kernel_function(state, function)?;
        }
        let key = current().ok_or(InvalidContext)?;
        let context = state.contexts.get(&key).ok_or(ContextIsDestroyed)?;
        context.check_live()?;
        let (module, name) = context.functions.get(&function).ok_or(InvalidHandle)?;
        let module = context.modules.get(module).ok_or(InvalidHandle)?;
        let entry = module.entries.get(name).ok_or(InvalidHandle)?;
        action(&state.runtime.devices[context.device as usize].info, entry)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuFuncGetAttribute(
    out: *mut c_int,
    attr: c_int,
    function: Opaque,
) -> CuResult {
    trace::call("cuFuncGetAttribute", || {
        status(with_function(function, |info, entry| {
            if out.is_null() {
                return Err(InvalidValue);
            }
            let value =
                math::function_attribute(attr, function_facts(info, entry)).ok_or(InvalidValue)?;
            unsafe { out.write(value) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuFuncSetAttribute(function: Opaque, attr: c_int, value: c_int) -> CuResult {
    trace::call("cuFuncSetAttribute", || {
        status(with_function(
            function,
            |info, entry| match math::set_function_attribute(
                attr,
                value,
                function_facts(info, entry),
            ) {
                math::SetAttribute::Accepted => Ok(()),
                math::SetAttribute::Invalid => Err(InvalidValue),
                math::SetAttribute::Unsupported => Err(NotSupported),
            },
        ))
    })
}

/// Cache preference is a hint; the backend has no configurable L1/shared split.
#[unsafe(no_mangle)]
pub extern "system" fn cuFuncSetCacheConfig(function: Opaque, config: c_int) -> CuResult {
    trace::call("cuFuncSetCacheConfig", || {
        status(with_function(function, |_, _| {
            if (0..=3).contains(&config) {
                Ok(())
            } else {
                Err(InvalidValue)
            }
        }))
    })
}

fn blocks_per_mp(
    info: &DeviceInfo,
    entry: &CudaEntry,
    block: c_int,
    shared: usize,
) -> Result<c_int, CuResult> {
    if block <= 0 {
        return Err(InvalidValue);
    }
    let limits = limits(info, function_facts(info, entry));
    Ok(math::active_blocks(limits, block as u32, shared as u64) as c_int)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuOccupancyMaxActiveBlocksPerMultiprocessor(
    out: *mut c_int,
    function: Opaque,
    block: c_int,
    shared: usize,
) -> CuResult {
    trace::call("cuOccupancyMaxActiveBlocksPerMultiprocessor", || unsafe {
        occupancy_blocks(out, function, block, shared, 0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    out: *mut c_int,
    function: Opaque,
    block: c_int,
    shared: usize,
    flags: c_uint,
) -> CuResult {
    trace::call(
        "cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
        || unsafe { occupancy_blocks(out, function, block, shared, flags) },
    )
}

unsafe fn occupancy_blocks(
    out: *mut c_int,
    function: Opaque,
    block: c_int,
    shared: usize,
    flags: c_uint,
) -> CuResult {
    status(with_function(function, |info, entry| {
        // 1 = CU_OCCUPANCY_DISABLE_CACHING_OVERRIDE; caching is not modelled.
        if out.is_null() || flags > 1 {
            return Err(InvalidValue);
        }
        unsafe { out.write(blocks_per_mp(info, entry, block, shared)?) };
        Ok(())
    }))
}

/// CUoccupancyB2DSize: dynamic shared bytes for a block size.
type BlockToShared = Option<extern "system" fn(c_int) -> usize>;

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuOccupancyMaxPotentialBlockSize(
    min_grid: *mut c_int,
    block: *mut c_int,
    function: Opaque,
    callback: BlockToShared,
    shared: usize,
    limit: c_int,
) -> CuResult {
    trace::call("cuOccupancyMaxPotentialBlockSize", || unsafe {
        potential_block_size(min_grid, block, function, callback, shared, limit, 0)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuOccupancyMaxPotentialBlockSizeWithFlags(
    min_grid: *mut c_int,
    block: *mut c_int,
    function: Opaque,
    callback: BlockToShared,
    shared: usize,
    limit: c_int,
    flags: c_uint,
) -> CuResult {
    trace::call("cuOccupancyMaxPotentialBlockSizeWithFlags", || unsafe {
        potential_block_size(min_grid, block, function, callback, shared, limit, flags)
    })
}

unsafe fn potential_block_size(
    min_grid: *mut c_int,
    block: *mut c_int,
    function: Opaque,
    callback: BlockToShared,
    shared: usize,
    limit: c_int,
    flags: c_uint,
) -> CuResult {
    status(with_function(function, |info, entry| {
        if min_grid.is_null() || block.is_null() || flags > 1 || limit < 0 {
            return Err(InvalidValue);
        }
        let limits = limits(info, function_facts(info, entry));
        let (grid, size) =
            math::potential_block_size(limits, limit as u32, |size| match callback {
                Some(callback) => callback(size as c_int),
                None => shared,
            } as u64);
        unsafe {
            min_grid.write(grid as c_int);
            block.write(size as c_int);
        }
        Ok(())
    }))
}

// ---------------------------------------------------------------------------
// Extended launch
// ---------------------------------------------------------------------------

#[repr(C)]
pub(crate) struct LaunchConfig {
    grid: [c_uint; 3],
    block: [c_uint; 3],
    shared: c_uint,
    stream: Opaque,
    attributes: *const LaunchAttribute,
    count: c_uint,
}

/// CUlaunchAttribute: a 4-byte id, padded to 8, then a 64-byte value union.
#[repr(C)]
pub(crate) struct LaunchAttribute {
    id: c_uint,
    _pad: [u8; 4],
    value: [c_uint; 16],
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLaunchKernelEx(
    config: *const LaunchConfig,
    function: Opaque,
    params: *mut Opaque,
    extra: *mut Opaque,
) -> CuResult {
    trace::call("cuLaunchKernelEx", || unsafe {
        let Some(config) = config.as_ref() else {
            return InvalidValue;
        };
        if config.count != 0 && config.attributes.is_null() {
            return InvalidValue;
        }
        let attributes = if config.count == 0 {
            &[]
        } else {
            std::slice::from_raw_parts(config.attributes, config.count as usize)
        };
        for attribute in attributes {
            let value = [attribute.value[0], attribute.value[1], attribute.value[2]];
            match math::launch_attribute(attribute.id, value) {
                math::LaunchAttribute::Ignore => {}
                math::LaunchAttribute::Unsupported => return NotSupported,
            }
        }
        cuLaunchKernel(
            function,
            config.grid[0],
            config.grid[1],
            config.grid[2],
            config.block[0],
            config.block[1],
            config.block[2],
            config.shared,
            config.stream,
            params,
            extra,
        )
    })
}

// ---------------------------------------------------------------------------
// Memory: address ranges, pointer attributes, registration, memset widths
// ---------------------------------------------------------------------------

fn pointer_facts(pointer: u64) -> Result<PointerFacts, CuResult> {
    state(|state| {
        let address = pointer as usize;
        let unknown = PointerFacts {
            kind: Kind::Unknown,
            pointer: address,
            base: 0,
            size: 0,
            context: 0,
            ordinal: 0,
        };
        for (&key, context) in &state.contexts {
            for (kind, map) in [
                (Kind::Device, &context.allocations),
                (Kind::Host, &context.host_allocations),
                (Kind::Host, &context.host_registered),
            ] {
                if let Some((base, size)) = math::range_of(map, address) {
                    return Ok(PointerFacts {
                        kind,
                        pointer: address,
                        base,
                        size,
                        context: key,
                        ordinal: context.device,
                    });
                }
            }
        }
        Ok(unknown)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemGetAddressRange_v2(
    base: *mut u64,
    size: *mut usize,
    pointer: u64,
) -> CuResult {
    trace::call("cuMemGetAddressRange_v2", || {
        status(pointer_facts(pointer).and_then(|facts| {
            if facts.kind != Kind::Device {
                return Err(NotFound);
            }
            unsafe {
                if !base.is_null() {
                    base.write(facts.base as u64);
                }
                if !size.is_null() {
                    size.write(facts.size);
                }
            }
            Ok(())
        }))
    })
}

unsafe fn write_value(data: *mut c_void, value: Value) {
    unsafe {
        match value {
            Value::U32(value) => data.cast::<u32>().write_unaligned(value),
            Value::U64(value) => data.cast::<u64>().write_unaligned(value),
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuPointerGetAttribute(
    data: *mut c_void,
    attr: c_int,
    pointer: u64,
) -> CuResult {
    trace::call("cuPointerGetAttribute", || {
        if data.is_null() {
            return InvalidValue;
        }
        status(pointer_facts(pointer).and_then(|facts| {
            match math::pointer_attribute(attr, &facts) {
                Ok(value) => {
                    unsafe { write_value(data, value) };
                    Ok(())
                }
                Err(math::PointerError::Invalid) => Err(InvalidValue),
                Err(math::PointerError::Unsupported) => Err(NotSupported),
            }
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuPointerGetAttributes(
    count: c_uint,
    attributes: *const c_int,
    data: *mut *mut c_void,
    pointer: u64,
) -> CuResult {
    trace::call("cuPointerGetAttributes", || {
        if count == 0 || attributes.is_null() || data.is_null() {
            return InvalidValue;
        }
        status(pointer_facts(pointer).and_then(|facts| {
            let mut outcome = Ok(());
            for index in 0..count as usize {
                let (attr, slot) =
                    unsafe { (attributes.add(index).read(), data.add(index).read()) };
                if slot.is_null() {
                    return Err(InvalidValue);
                }
                // Unlike cuPointerGetAttribute, an unknown pointer is not an
                // error here: the attributes take their default values.
                let value = match math::pointer_attribute(attr, &facts) {
                    Err(_) if facts.kind == Kind::Unknown => {
                        math::pointer_default(attr).ok_or(InvalidValue)
                    }
                    Ok(value) => Ok(value),
                    Err(math::PointerError::Invalid) => Err(InvalidValue),
                    Err(math::PointerError::Unsupported) => Err(NotSupported),
                };
                match value {
                    Ok(value) => unsafe { write_value(slot, value) },
                    Err(error) => outcome = outcome.and(Err(error)),
                }
            }
            outcome
        }))
    })
}

/// Registration is bookkeeping only: transfers treat registered memory as
/// pageable, which is always correct, so no pinning is claimed.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemHostRegister_v2(
    host: Opaque,
    size: usize,
    flags: c_uint,
) -> CuResult {
    trace::call("cuMemHostRegister_v2", || {
        // 1 PORTABLE, 2 DEVICEMAP, 4 IOMEMORY, 8 READ_ONLY.
        if host.is_null() || size == 0 || flags & !0xf != 0 {
            return InvalidValue;
        }
        if flags & 6 != 0 {
            return NotSupported;
        }
        status(with_context(|_, context| {
            let start = host.addr();
            if math::overlaps(&context.host_allocations, start, size)
                || math::overlaps(&context.host_registered, start, size)
            {
                return Err(HostMemoryAlreadyRegistered);
            }
            context.host_registered.insert(start, size);
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemHostUnregister(host: Opaque) -> CuResult {
    trace::call("cuMemHostUnregister", || {
        status(with_context(|runtime, context| {
            if !context.host_registered.contains_key(&host.addr()) {
                return Err(HostMemoryNotRegistered);
            }
            // Pending transfers may still read the range.
            context.streams.wait_all(&runtime.api)?;
            context.host_registered.remove(&host.addr());
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemAllocHost_v2(out: *mut Opaque, size: usize) -> CuResult {
    trace::call("cuMemAllocHost_v2", || unsafe {
        cuMemHostAlloc(out, size, 0)
    })
}

unsafe fn memset_wide(
    dst: u64,
    value: u32,
    width: usize,
    count: usize,
    stream: Opaque,
) -> CuResult {
    if dst as usize % width != 0 {
        return InvalidValue;
    }
    let Some(size) = count.checked_mul(width) else {
        return InvalidValue;
    };
    if let Some(byte) = math::uniform_byte(value, width) {
        return memset_bytes(dst, byte, size, stream);
    }
    // Non-uniform pattern: stage it on the host and copy it down.
    if size == 0 {
        return memset_bytes(dst, 0, 0, stream);
    }
    let pattern = math::expand_pattern(value, width, count);
    unsafe { copy_async(pattern.as_ptr().cast_mut().cast(), dst, size, stream, true) }
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD16_v2(dst: u64, value: u16, count: usize) -> CuResult {
    trace::call("cuMemsetD16_v2", || unsafe {
        memset_wide(dst, value.into(), 2, count, std::ptr::null_mut())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD16Async(
    dst: u64,
    value: u16,
    count: usize,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemsetD16Async", || unsafe {
        memset_wide(dst, value.into(), 2, count, stream)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD32_v2(dst: u64, value: u32, count: usize) -> CuResult {
    trace::call("cuMemsetD32_v2", || unsafe {
        memset_wide(dst, value, 4, count, std::ptr::null_mut())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD32Async(
    dst: u64,
    value: u32,
    count: usize,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemsetD32Async", || unsafe {
        memset_wide(dst, value, 4, count, stream)
    })
}

// ---------------------------------------------------------------------------
// Generic and strided copies
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Eq, PartialEq)]
enum Place {
    Host,
    Device,
}

fn place_of(memory: MemoryType, address: usize, size: usize) -> Result<Place, CuResult> {
    match memory {
        MemoryType::Host => Ok(Place::Host),
        MemoryType::Device => Ok(Place::Device),
        MemoryType::Array => Err(NotSupported),
        MemoryType::Unified => with_context(|_, context| {
            Ok(if contains_range(&context.allocations, address, size) {
                Place::Device
            } else {
                Place::Host
            })
        }),
    }
}

/// Copy between two places. `stream` is `None` for a synchronous API.
unsafe fn move_bytes(
    dst: (Place, usize),
    src: (Place, usize),
    size: usize,
    stream: Option<Opaque>,
) -> CuResult {
    if size == 0 {
        return Success;
    }
    let (dst_address, src_address) = (dst.1, src.1);
    let host = |address: usize| std::ptr::with_exposed_provenance_mut::<c_void>(address);
    unsafe {
        match (dst.0, src.0) {
            (Place::Device, Place::Host) => match stream {
                Some(stream) => {
                    copy_async(host(src_address), dst_address as u64, size, stream, true)
                }
                None => copy(host(src_address), dst_address as u64, size, true),
            },
            (Place::Host, Place::Device) => match stream {
                Some(stream) => {
                    copy_async(host(dst_address), src_address as u64, size, stream, false)
                }
                None => copy(host(dst_address), src_address as u64, size, false),
            },
            (Place::Device, Place::Device) => match stream {
                Some(stream) => {
                    cuMemcpyDtoDAsync_v2(dst_address as u64, src_address as u64, size, stream)
                }
                None => cuMemcpyDtoD_v2(dst_address as u64, src_address as u64, size),
            },
            (Place::Host, Place::Host) => status(with_context(|runtime, context| {
                // Order after prior work on the stream, as a pageable copy does.
                let key = context
                    .streams
                    .launch_key(stream.unwrap_or(std::ptr::null_mut()))?;
                context.streams.wait_stream(&runtime.api, key, u64::MAX)?;
                if host(dst_address).is_null() || host(src_address).is_null() {
                    return Err(InvalidValue);
                }
                std::ptr::copy(
                    host(src_address).cast::<u8>(),
                    host(dst_address).cast::<u8>(),
                    size,
                );
                Ok(())
            })),
        }
    }
}

/// cuMemcpy: pointers are classified by the allocations the shim knows.
unsafe fn copy_generic(dst: u64, src: u64, size: usize, stream: Option<Opaque>) -> CuResult {
    if size == 0 {
        return Success;
    }
    let classify = |address: u64| place_of(MemoryType::Unified, address as usize, size);
    let places = classify(dst).and_then(|d| classify(src).map(|s| (d, s)));
    match places {
        Ok((d, s)) => unsafe { move_bytes((d, dst as usize), (s, src as usize), size, stream) },
        Err(error) => error,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpy(dst: u64, src: u64, size: usize) -> CuResult {
    trace::call("cuMemcpy", || unsafe { copy_generic(dst, src, size, None) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyAsync(
    dst: u64,
    src: u64,
    size: usize,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemcpyAsync", || unsafe {
        copy_generic(dst, src, size, Some(stream))
    })
}

#[repr(C)]
pub(crate) struct Memcpy2D {
    src_x: usize,
    src_y: usize,
    src_type: c_uint,
    src_host: *const c_void,
    src_device: u64,
    src_array: Opaque,
    src_pitch: usize,
    dst_x: usize,
    dst_y: usize,
    dst_type: c_uint,
    dst_host: *mut c_void,
    dst_device: u64,
    dst_array: Opaque,
    dst_pitch: usize,
    width: usize,
    height: usize,
}

#[repr(C)]
pub(crate) struct Memcpy3D {
    src_x: usize,
    src_y: usize,
    src_z: usize,
    src_lod: usize,
    src_type: c_uint,
    src_host: *const c_void,
    src_device: u64,
    src_array: Opaque,
    reserved0: Opaque,
    src_pitch: usize,
    src_height: usize,
    dst_x: usize,
    dst_y: usize,
    dst_z: usize,
    dst_lod: usize,
    dst_type: c_uint,
    dst_host: *mut c_void,
    dst_device: u64,
    dst_array: Opaque,
    reserved1: Opaque,
    dst_pitch: usize,
    dst_height: usize,
    width: usize,
    height: usize,
    depth: usize,
}

/// Base address of one side: the host pointer, or the device/unified address.
fn side_base(memory: c_uint, host: usize, device: u64) -> Result<(MemoryType, usize), CuResult> {
    let memory = MemoryType::from_raw(memory).ok_or(InvalidValue)?;
    Ok((
        memory,
        if memory == MemoryType::Host {
            host
        } else {
            device as usize
        },
    ))
}

/// Run a validated strided copy one contiguous row at a time.
unsafe fn run_plan(
    plan: math::Plan,
    dst: (MemoryType, usize),
    src: (MemoryType, usize),
    stream: Option<Opaque>,
) -> CuResult {
    if plan.is_empty() {
        return Success;
    }
    let (Some((dst_start, dst_len)), Some((src_start, src_len))) =
        (plan.span(false), plan.span(true))
    else {
        return InvalidValue;
    };
    let (Some(dst_first), Some(src_first)) =
        (dst.1.checked_add(dst_start), src.1.checked_add(src_start))
    else {
        return InvalidValue;
    };
    let places = place_of(dst.0, dst_first, dst_len)
        .and_then(|d| place_of(src.0, src_first, src_len).map(|s| (d, s)));
    let (dst_place, src_place) = match places {
        Ok(places) => places,
        Err(error) => return error,
    };
    for plane in 0..plan.planes {
        for row in 0..plan.rows {
            let Some((src_offset, dst_offset)) = plan.row(row, plane) else {
                return InvalidValue;
            };
            let (Some(from), Some(to)) =
                (src.1.checked_add(src_offset), dst.1.checked_add(dst_offset))
            else {
                return InvalidValue;
            };
            let result =
                unsafe { move_bytes((dst_place, to), (src_place, from), plan.width, stream) };
            if result != Success {
                return result;
            }
        }
    }
    Success
}

unsafe fn copy_2d(copy: *const Memcpy2D, stream: Option<Opaque>) -> CuResult {
    let Some(copy) = (unsafe { copy.as_ref() }) else {
        return InvalidValue;
    };
    let sides = side_base(copy.src_type, copy.src_host.addr(), copy.src_device).and_then(|src| {
        side_base(copy.dst_type, copy.dst_host.addr(), copy.dst_device).map(|dst| (src, dst))
    });
    let (src, dst) = match sides {
        Ok(sides) => sides,
        Err(error) => return error,
    };
    let side = |x_bytes, y, pitch| math::Side {
        x_bytes,
        y,
        z: 0,
        pitch,
        height: 0,
    };
    let plan = math::Plan::new(
        side(copy.src_x, copy.src_y, copy.src_pitch),
        side(copy.dst_x, copy.dst_y, copy.dst_pitch),
        copy.width,
        copy.height,
        1,
    );
    match plan {
        Some(plan) => unsafe { run_plan(plan, dst, src, stream) },
        None => InvalidValue,
    }
}

unsafe fn copy_3d(copy: *const Memcpy3D, stream: Option<Opaque>) -> CuResult {
    let Some(copy) = (unsafe { copy.as_ref() }) else {
        return InvalidValue;
    };
    let sides = side_base(copy.src_type, copy.src_host.addr(), copy.src_device).and_then(|src| {
        side_base(copy.dst_type, copy.dst_host.addr(), copy.dst_device).map(|dst| (src, dst))
    });
    let (src, dst) = match sides {
        Ok(sides) => sides,
        Err(error) => return error,
    };
    if copy.src_lod != 0 || copy.dst_lod != 0 {
        return InvalidValue;
    }
    let plan = math::Plan::new(
        math::Side {
            x_bytes: copy.src_x,
            y: copy.src_y,
            z: copy.src_z,
            pitch: copy.src_pitch,
            height: copy.src_height,
        },
        math::Side {
            x_bytes: copy.dst_x,
            y: copy.dst_y,
            z: copy.dst_z,
            pitch: copy.dst_pitch,
            height: copy.dst_height,
        },
        copy.width,
        copy.height,
        copy.depth,
    );
    match plan {
        Some(plan) => unsafe { run_plan(plan, dst, src, stream) },
        None => InvalidValue,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpy2D_v2(copy: *const Memcpy2D) -> CuResult {
    trace::call("cuMemcpy2D_v2", || unsafe { copy_2d(copy, None) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpy2DAsync_v2(
    copy: *const Memcpy2D,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemcpy2DAsync_v2", || unsafe {
        copy_2d(copy, Some(stream))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpy3D_v2(copy: *const Memcpy3D) -> CuResult {
    trace::call("cuMemcpy3D_v2", || unsafe { copy_3d(copy, None) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpy3DAsync_v2(
    copy: *const Memcpy3D,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemcpy3DAsync_v2", || unsafe {
        copy_3d(copy, Some(stream))
    })
}

// ---------------------------------------------------------------------------
// Peers
// ---------------------------------------------------------------------------

/// Copies between contexts complete before returning, which satisfies the
/// stream ordering of the Async form.
fn peer_copy(
    dst: u64,
    dst_context: Opaque,
    src: u64,
    src_context: Opaque,
    size: usize,
) -> CuResult {
    status(state(|state| {
        let (dst_key, src_key) = (dst_context.addr(), src_context.addr());
        for key in [dst_key, src_key] {
            state
                .contexts
                .get(&key)
                .ok_or(InvalidContext)?
                .check_live()?;
        }
        if size == 0 {
            return Ok(());
        }
        let api = &state.runtime.api;
        let source = state.contexts.get_mut(&src_key).ok_or(InvalidContext)?;
        if !contains_range(&source.allocations, src as usize, size) {
            return Err(InvalidValue);
        }
        source.streams.wait_all(api)?;
        let target = state.contexts.get_mut(&dst_key).ok_or(InvalidContext)?;
        if !contains_range(&target.allocations, dst as usize, size) {
            return Err(InvalidValue);
        }
        if dst_key != src_key {
            target.streams.wait_all(api)?;
        }
        let dst_ptr = std::ptr::with_exposed_provenance_mut(dst as usize);
        let src_ptr = std::ptr::with_exposed_provenance(src as usize);
        unsafe { target.backend.copy(api, dst_ptr, src_ptr, size) }
            .inspect_err(|error| target.backend.health.observe(error))
            .map_err(backend_error)
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyPeer(
    dst: u64,
    dst_context: Opaque,
    src: u64,
    src_context: Opaque,
    size: usize,
) -> CuResult {
    trace::call("cuMemcpyPeer", || {
        peer_copy(dst, dst_context, src, src_context, size)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemcpyPeerAsync(
    dst: u64,
    dst_context: Opaque,
    src: u64,
    src_context: Opaque,
    size: usize,
    _stream: Opaque,
) -> CuResult {
    trace::call("cuMemcpyPeerAsync", || {
        peer_copy(dst, dst_context, src, src_context, size)
    })
}

fn valid_device(ordinal: c_int) -> Result<(), CuResult> {
    runtime(|runtime| {
        usize::try_from(ordinal)
            .ok()
            .filter(|&ordinal| ordinal < runtime.devices.len())
            .map(|_| ())
            .ok_or(InvalidDevice)
    })
}

/// Level Zero devices are not linked for direct access by this shim.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceCanAccessPeer(
    out: *mut c_int,
    device: c_int,
    peer: c_int,
) -> CuResult {
    trace::call("cuDeviceCanAccessPeer", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(
            valid_device(device)
                .and_then(|()| valid_device(peer))
                .map(|()| unsafe { out.write(0) }),
        )
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetP2PAttribute(
    out: *mut c_int,
    attr: c_int,
    src: c_int,
    dst: c_int,
) -> CuResult {
    trace::call("cuDeviceGetP2PAttribute", || {
        // 1 PERFORMANCE_RANK, 2 ACCESS_SUPPORTED, 3 NATIVE_ATOMIC_SUPPORTED,
        // 4 CUDA_ARRAY_ACCESS_SUPPORTED, 5 ONLY_PARTIAL_NATIVE_ATOMIC_SUPPORTED.
        if out.is_null() || !(1..=5).contains(&attr) {
            return InvalidValue;
        }
        status(
            valid_device(src)
                .and_then(|()| valid_device(dst))
                .and_then(|()| {
                    if src == dst {
                        return Err(InvalidDevice);
                    }
                    unsafe { out.write(0) };
                    Ok(())
                }),
        )
    })
}

fn peer_context(peer: Opaque) -> Result<(), CuResult> {
    state(|state| {
        let current = state
            .contexts
            .get(&current().ok_or(InvalidContext)?)
            .ok_or(ContextIsDestroyed)?;
        let peer = state.contexts.get(&peer.addr()).ok_or(InvalidContext)?;
        peer.check_live()?;
        if peer.device == current.device {
            return Err(InvalidDevice);
        }
        Ok(())
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxEnablePeerAccess(peer: Opaque, flags: c_uint) -> CuResult {
    trace::call("cuCtxEnablePeerAccess", || {
        if flags != 0 {
            return InvalidValue;
        }
        status(peer_context(peer).and_then(|()| Err(PeerAccessUnsupported)))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxDisablePeerAccess(peer: Opaque) -> CuResult {
    trace::call("cuCtxDisablePeerAccess", || {
        status(peer_context(peer).and_then(|()| Err(PeerAccessNotEnabled)))
    })
}

// ---------------------------------------------------------------------------
// Context stack and streams
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxPushCurrent_v2(handle: Opaque) -> CuResult {
    trace::call("cuCtxPushCurrent_v2", || {
        status(state(|state| {
            state
                .contexts
                .get(&handle.addr())
                .ok_or(InvalidContext)?
                .check_live()?;
            CURRENT.with_borrow_mut(|stack| stack.push(handle.addr()));
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxPopCurrent_v2(out: *mut Opaque) -> CuResult {
    trace::call("cuCtxPopCurrent_v2", || {
        status(state(|_| {
            let key = CURRENT
                .with_borrow_mut(|stack| stack.pop())
                .ok_or(InvalidContext)?;
            if !out.is_null() {
                unsafe { out.write(std::ptr::without_provenance_mut(key)) };
            }
            Ok(())
        }))
    })
}

/// The device reports no stream priority support (attribute 78), so every
/// priority maps to the single level 0.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetStreamPriorityRange(
    least: *mut c_int,
    greatest: *mut c_int,
) -> CuResult {
    trace::call("cuCtxGetStreamPriorityRange", || {
        status(with_context(|_, _| {
            unsafe {
                if !least.is_null() {
                    least.write(0);
                }
                if !greatest.is_null() {
                    greatest.write(0);
                }
            }
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamCreateWithPriority(
    out: *mut Opaque,
    flags: c_uint,
    _priority: c_int,
) -> CuResult {
    trace::call("cuStreamCreateWithPriority", || unsafe {
        super::streams::cuStreamCreate(out, flags)
    })
}

// ---------------------------------------------------------------------------
// Stream-ordered allocation, pools, managed-memory hints
// ---------------------------------------------------------------------------

static POOL_ATTRIBUTES: Mutex<BTreeMap<(usize, c_int), u64>> = Mutex::new(BTreeMap::new());

fn pool_device(pool: Opaque) -> Result<usize, CuResult> {
    let ordinal = math::pool_ordinal(pool.addr()).ok_or(InvalidValue)?;
    valid_device(ordinal as c_int).map_err(|_| InvalidValue)?;
    Ok(ordinal)
}

/// The shim's allocator is already stream-safe: allocation is immediate and
/// free waits for outstanding work, which is a valid stream-ordered schedule.
unsafe fn alloc_async(out: *mut u64, size: usize, stream: Opaque) -> CuResult {
    let valid = with_context(|_, context| context.streams.launch_key(stream).map(|_| ()));
    match valid {
        Ok(()) => unsafe { cuMemAlloc_v2(out, size) },
        Err(error) => error,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemAllocAsync(
    out: *mut u64,
    size: usize,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemAllocAsync", || unsafe {
        alloc_async(out, size, stream)
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemAllocFromPoolAsync(
    out: *mut u64,
    size: usize,
    pool: Opaque,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemAllocFromPoolAsync", || {
        let device = with_context(|_, context| Ok(context.device as usize));
        match (pool_device(pool), device) {
            (Ok(pool), Ok(device)) if pool == device => unsafe { alloc_async(out, size, stream) },
            (Ok(_), Ok(_)) => InvalidValue,
            (Err(error), _) | (_, Err(error)) => error,
        }
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemFreeAsync(address: u64, stream: Opaque) -> CuResult {
    trace::call("cuMemFreeAsync", || {
        if address == 0 {
            return InvalidValue;
        }
        let valid = with_context(|_, context| context.streams.launch_key(stream).map(|_| ()));
        match valid {
            Ok(()) => cuMemFree_v2(address),
            Err(error) => error,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetDefaultMemPool(
    pool: *mut Opaque,
    device: c_int,
) -> CuResult {
    trace::call("cuDeviceGetDefaultMemPool", || {
        if pool.is_null() {
            return InvalidValue;
        }
        status(valid_device(device).map(|()| unsafe {
            pool.write(std::ptr::without_provenance_mut(math::pool_handle(
                device as usize,
            )))
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetMemPool(pool: *mut Opaque, device: c_int) -> CuResult {
    trace::call("cuDeviceGetMemPool", || unsafe {
        cuDeviceGetDefaultMemPool(pool, device)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuDeviceSetMemPool(device: c_int, pool: Opaque) -> CuResult {
    trace::call("cuDeviceSetMemPool", || {
        status(valid_device(device).and_then(|()| {
            // Only the default pool exists.
            if pool_device(pool)? == device as usize {
                Ok(())
            } else {
                Err(InvalidValue)
            }
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemPoolCreate(
    _pool: *mut Opaque,
    _props: *const c_void,
) -> CuResult {
    trace::call("cuMemPoolCreate", || NotSupported)
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemPoolDestroy(pool: Opaque) -> CuResult {
    trace::call("cuMemPoolDestroy", || {
        // The default pools are not destroyable.
        status(pool_device(pool).and(Err(InvalidValue)))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemPoolSetAttribute(
    pool: Opaque,
    attr: c_int,
    value: *mut c_void,
) -> CuResult {
    trace::call("cuMemPoolSetAttribute", || {
        if value.is_null() {
            return InvalidValue;
        }
        let ordinal = match pool_device(pool) {
            Ok(ordinal) => ordinal,
            Err(error) => return error,
        };
        let stored = match attr {
            // REUSE_* are int flags.
            1..=3 => u64::from(unsafe { value.cast::<c_int>().read_unaligned() } != 0),
            4 => unsafe { value.cast::<u64>().read_unaligned() },
            // The high-water marks may only be reset to zero.
            6 | 8 if unsafe { value.cast::<u64>().read_unaligned() } == 0 => return Success,
            _ => return InvalidValue,
        };
        match POOL_ATTRIBUTES.lock() {
            Ok(mut attributes) => {
                attributes.insert((ordinal, attr), stored);
                Success
            }
            Err(_) => Unknown,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuMemPoolGetAttribute(
    pool: Opaque,
    attr: c_int,
    value: *mut c_void,
) -> CuResult {
    trace::call("cuMemPoolGetAttribute", || {
        if value.is_null() {
            return InvalidValue;
        }
        let ordinal = match pool_device(pool) {
            Ok(ordinal) => ordinal,
            Err(error) => return error,
        };
        let Some(default) = math::pool_default(attr) else {
            // Usage statistics are not tracked.
            return if (5..=8).contains(&attr) {
                NotSupported
            } else {
                InvalidValue
            };
        };
        let Ok(attributes) = POOL_ATTRIBUTES.lock() else {
            return Unknown;
        };
        let stored = attributes.get(&(ordinal, attr)).copied().unwrap_or(default);
        unsafe {
            if attr == 4 {
                value.cast::<u64>().write_unaligned(stored);
            } else {
                value.cast::<c_int>().write_unaligned(stored as c_int);
            }
        }
        Success
    })
}

/// Freed memory is returned to the driver immediately, so there is nothing to trim.
#[unsafe(no_mangle)]
pub extern "system" fn cuMemPoolTrimTo(pool: Opaque, _keep: usize) -> CuResult {
    trace::call("cuMemPoolTrimTo", || status(pool_device(pool).map(|_| ())))
}

/// The range must be device memory the shim allocated. Device memory is
/// always resident on its device, so migration hints have nothing to do.
fn device_hint(pointer: u64, count: usize, stream: Option<Opaque>) -> CuResult {
    status(with_context(|_, context| {
        if let Some(stream) = stream {
            context.streams.launch_key(stream)?;
        }
        if count == 0 || !contains_range(&context.allocations, pointer as usize, count) {
            return Err(InvalidValue);
        }
        Ok(())
    }))
}

/// CUmemLocation: type 1 DEVICE, 2 HOST, 3 HOST_NUMA, 4 HOST_NUMA_CURRENT.
#[repr(C)]
#[derive(Clone, Copy)]
pub(crate) struct MemLocation {
    kind: c_int,
    id: c_int,
}

fn location_valid(location: MemLocation) -> bool {
    match location.kind {
        1 => valid_device(location.id).is_ok(),
        2 | 4 => true,
        3 => location.id >= 0,
        _ => false,
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemPrefetchAsync(
    pointer: u64,
    count: usize,
    device: c_int,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemPrefetchAsync", || {
        // The legacy API names the destination by ordinal; -1 is the CPU.
        if device != -1 && valid_device(device).is_err() {
            return InvalidDevice;
        }
        device_hint(pointer, count, Some(stream))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemPrefetchAsync_v2(
    pointer: u64,
    count: usize,
    location: MemLocation,
    flags: c_uint,
    stream: Opaque,
) -> CuResult {
    trace::call("cuMemPrefetchAsync_v2", || {
        if flags != 0 || !location_valid(location) {
            return InvalidValue;
        }
        device_hint(pointer, count, Some(stream))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemAdvise(
    pointer: u64,
    count: usize,
    advice: c_int,
    device: c_int,
) -> CuResult {
    trace::call("cuMemAdvise", || {
        if !(1..=6).contains(&advice) || (device != -1 && valid_device(device).is_err()) {
            return InvalidValue;
        }
        device_hint(pointer, count, None)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemAdvise_v2(
    pointer: u64,
    count: usize,
    advice: c_int,
    location: MemLocation,
) -> CuResult {
    trace::call("cuMemAdvise_v2", || {
        if !(1..=6).contains(&advice) || !location_valid(location) {
            return InvalidValue;
        }
        device_hint(pointer, count, None)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::null_mut;

    #[test]
    fn argument_validation_precedes_driver_state() {
        let _serial = TEST_RUNTIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        unsafe {
            assert_eq!(cuDeviceCanAccessPeer(null_mut(), 0, 1), InvalidValue);
            assert_eq!(cuDeviceGetP2PAttribute(null_mut(), 2, 0, 1), InvalidValue);
            assert_eq!(cuMemHostRegister_v2(null_mut(), 8, 0), InvalidValue);
            assert_eq!(cuMemHostRegister_v2(8 as Opaque, 8, 0x10), InvalidValue);
            assert_eq!(cuMemHostRegister_v2(8 as Opaque, 8, 2), NotSupported);
            assert_eq!(cuPointerGetAttribute(null_mut(), 2, 0), InvalidValue);
            assert_eq!(
                cuPointerGetAttributes(0, null_mut(), null_mut(), 0),
                InvalidValue
            );
            assert_eq!(
                cuLaunchKernelEx(std::ptr::null(), null_mut(), null_mut(), null_mut()),
                InvalidValue
            );
            assert_eq!(
                cuMemPoolSetAttribute(null_mut(), 4, null_mut()),
                InvalidValue
            );
            assert_eq!(cuMemPoolCreate(null_mut(), std::ptr::null()), NotSupported);
        }
        assert_eq!(cuCtxEnablePeerAccess(null_mut(), 1), InvalidValue);
    }

    #[test]
    fn calls_that_need_a_runtime_report_not_initialized() {
        let _serial = TEST_RUNTIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(RUNTIME.lock().unwrap().is_none());
        let mut out = 0;
        unsafe {
            assert_eq!(cuFuncGetAttribute(&mut out, 0, null_mut()), NotInitialized);
            assert_eq!(
                cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut out, null_mut(), 128, 0),
                NotInitialized
            );
            assert_eq!(
                cuMemGetAddressRange_v2(null_mut(), null_mut(), 16),
                NotInitialized
            );
            let mut value = 0u32;
            assert_eq!(
                cuPointerGetAttribute((&raw mut value).cast(), 2, 16),
                NotInitialized
            );
        }
        assert_eq!(cuMemFreeAsync(16, null_mut()), NotInitialized);
        assert_eq!(cuDeviceSetMemPool(0, null_mut()), NotInitialized);
    }

    #[test]
    fn pool_handles_are_validated_before_use() {
        assert_eq!(cuMemPoolTrimTo(null_mut(), 0), InvalidValue);
        assert_eq!(cuMemPoolDestroy(null_mut()), InvalidValue);
    }

    /// Needs an Intel GPU: `cargo test -- --ignored` on a machine with Level Zero.
    #[test]
    #[ignore = "requires an Intel GPU"]
    fn gpu_memory_and_pointer_apis() {
        let _serial = TEST_RUNTIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert_eq!(cuInit(0), Success);
        let mut context = null_mut();
        assert_eq!(unsafe { cuCtxCreate_v2(&mut context, 0, 0) }, Success);
        let mut device = 0u64;
        assert_eq!(unsafe { cuMemAlloc_v2(&mut device, 256) }, Success);

        // Pointer and range queries.
        let (mut base, mut size) = (0u64, 0usize);
        assert_eq!(
            unsafe { cuMemGetAddressRange_v2(&mut base, &mut size, device + 16) },
            Success
        );
        assert_eq!((base, size), (device, 256));
        let mut kind = 0u32;
        assert_eq!(
            unsafe { cuPointerGetAttribute((&raw mut kind).cast(), 2, device) },
            Success
        );
        assert_eq!(kind, 2);

        // Non-uniform 32-bit memset, then read back.
        assert_eq!(cuMemsetD32_v2(device, 0x0403_0201, 64), Success);
        let mut host = [0u8; 256];
        assert_eq!(
            unsafe { cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), device, 256) },
            Success
        );
        assert_eq!(&host[..8], [1, 2, 3, 4, 1, 2, 3, 4]);

        // 2D device-to-host with a 64-byte pitch picks 8 bytes from each row.
        let mut out = [0u8; 32];
        let copy = Memcpy2D {
            src_x: 4,
            src_y: 0,
            src_type: 2,
            src_host: std::ptr::null(),
            src_device: device,
            src_array: null_mut(),
            src_pitch: 64,
            dst_x: 0,
            dst_y: 0,
            dst_type: 1,
            dst_host: out.as_mut_ptr().cast(),
            dst_device: 0,
            dst_array: null_mut(),
            dst_pitch: 8,
            width: 8,
            height: 4,
        };
        assert_eq!(unsafe { cuMemcpy2D_v2(&copy) }, Success);
        assert_eq!(&out[..8], [1, 2, 3, 4, 1, 2, 3, 4]);

        assert_eq!(cuMemFreeAsync(device, null_mut()), Success);
        assert_eq!(cuCtxDestroy_v2(context), Success);
    }
}
