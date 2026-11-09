//! Driver API entry points that real applications call beyond the core set: context
//! limits and cache configuration, stream queries, profiler no-ops, `cuModuleLoadDataEx`,
//! `cuStreamAddCallback`, 2-D memset and cooperative launch.
//!
//! Limits of the backend are reported honestly: there is one stream priority level, no
//! configurable L1/shared split, and cooperative launches are only accepted when every
//! block can be resident at once (otherwise `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE`).
use super::stream_owner;
use crate::cuda::CuResult::*;
use crate::cuda::gaps::{
    cuMemsetD16_v2, cuMemsetD16Async, cuMemsetD32_v2, cuMemsetD32Async,
    cuOccupancyMaxActiveBlocksPerMultiprocessor,
};
use crate::cuda::*;
use std::collections::HashMap;
use std::sync::Mutex;

const LIMIT_COUNT: usize = 7;

/// Defaults a CUDA 12 driver reports for `cuCtxGetLimit` (limit enum 0..=6).
pub(crate) fn default_limit(limit: c_int) -> Option<usize> {
    Some(match limit {
        0 => 1024,        // CU_LIMIT_STACK_SIZE
        1 => 1 << 20,     // PRINTF_FIFO_SIZE
        2 => 8 << 20,     // MALLOC_HEAP_SIZE
        3 => 2,           // DEV_RUNTIME_SYNC_DEPTH
        4 => 2048,        // DEV_RUNTIME_PENDING_LAUNCH_COUNT
        5 => 0,           // MAX_L2_FETCH_GRANULARITY
        6 => 0,           // PERSISTING_L2_CACHE_SIZE
        _ => return None,
    })
}

pub(crate) fn valid_cache_config(config: c_int) -> bool {
    (0..=3).contains(&config)
}

pub(crate) fn valid_shared_mem_config(config: c_int) -> bool {
    (0..=2).contains(&config)
}

#[derive(Clone, Copy)]
struct ContextConfig {
    limits: [usize; LIMIT_COUNT],
    cache: c_int,
    shared_mem: c_int,
}

impl ContextConfig {
    fn new() -> Self {
        let mut limits = [0; LIMIT_COUNT];
        for (index, slot) in limits.iter_mut().enumerate() {
            *slot = default_limit(index as c_int).unwrap_or(0);
        }
        Self { limits, cache: 0, shared_mem: 0 }
    }
}

static CONFIGS: Mutex<Option<HashMap<usize, ContextConfig>>> = Mutex::new(None);
static STREAM_FLAGS: Mutex<Option<HashMap<usize, u32>>> = Mutex::new(None);

fn with_config<T>(action: impl FnOnce(&mut ContextConfig) -> Result<T, CuResult>) -> Result<T, CuResult> {
    let key = state(|state| {
        let key = current().ok_or(InvalidContext)?;
        state.contexts.get(&key).ok_or(ContextIsDestroyed)?.check_live()?;
        Ok(key)
    })?;
    let mut guard = CONFIGS.lock().map_err(|_| Unknown)?;
    action(guard.get_or_insert_with(HashMap::new).entry(key).or_insert_with(ContextConfig::new))
}

/// Called by `cuStreamCreate`; handles are never reused, so entries cannot alias.
pub(crate) fn note_stream(key: usize, flags: u32) {
    if let Ok(mut guard) = STREAM_FLAGS.lock() {
        guard.get_or_insert_with(HashMap::new).insert(key, flags);
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetLimit(out: *mut usize, limit: c_int) -> CuResult {
    trace::call("cuCtxGetLimit", || {
        if out.is_null() || default_limit(limit).is_none() {
            return InvalidValue;
        }
        status(with_config(|config| {
            unsafe { out.write(config.limits[limit as usize]) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxSetLimit(limit: c_int, value: usize) -> CuResult {
    trace::call("cuCtxSetLimit", || {
        if default_limit(limit).is_none() {
            return InvalidValue;
        }
        status(with_config(|config| {
            config.limits[limit as usize] = value;
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetCacheConfig(out: *mut c_int) -> CuResult {
    trace::call("cuCtxGetCacheConfig", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(with_config(|config| {
            unsafe { out.write(config.cache) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxSetCacheConfig(value: c_int) -> CuResult {
    trace::call("cuCtxSetCacheConfig", || {
        if !valid_cache_config(value) {
            return InvalidValue;
        }
        status(with_config(|config| {
            config.cache = value;
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetSharedMemConfig(out: *mut c_int) -> CuResult {
    trace::call("cuCtxGetSharedMemConfig", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(with_config(|config| {
            unsafe { out.write(config.shared_mem) };
            Ok(())
        }))
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuCtxSetSharedMemConfig(value: c_int) -> CuResult {
    trace::call("cuCtxSetSharedMemConfig", || {
        if !valid_shared_mem_config(value) {
            return InvalidValue;
        }
        status(with_config(|config| {
            config.shared_mem = value;
            Ok(())
        }))
    })
}

/// A context's id is its (never reused) handle value; `0` means the current context.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuCtxGetId(ctx: Opaque, out: *mut u64) -> CuResult {
    trace::call("cuCtxGetId", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            let key = if ctx.is_null() { current().ok_or(InvalidContext)? } else { ctx.addr() };
            state.contexts.get(&key).ok_or(InvalidContext)?.check_live()?;
            unsafe { out.write(key as u64) };
            Ok(())
        }))
    })
}

/// Scheduling flags have no effect on this backend; they are accepted and ignored.
#[unsafe(no_mangle)]
pub extern "system" fn cuCtxSetFlags(_flags: c_uint) -> CuResult {
    trace::call("cuCtxSetFlags", || status(with_config(|_| Ok(()))))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceComputeCapability(
    major: *mut c_int,
    minor: *mut c_int,
    device: c_int,
) -> CuResult {
    trace::call("cuDeviceComputeCapability", || {
        if major.is_null() || minor.is_null() {
            return InvalidValue;
        }
        let (mut high, mut low) = (0, 0);
        let result = unsafe { cuDeviceGetAttribute(&mut high, 75, device) };
        if result != Success {
            return result;
        }
        let result = unsafe { cuDeviceGetAttribute(&mut low, 76, device) };
        if result == Success {
            unsafe {
                major.write(high);
                minor.write(low);
            }
        }
        result
    })
}

/// `CUdevprop` is 21 ints; filled from the attribute table.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuDeviceGetProperties(prop: *mut c_int, device: c_int) -> CuResult {
    trace::call("cuDeviceGetProperties", || {
        if prop.is_null() {
            return InvalidValue;
        }
        // maxThreadsPerBlock, maxThreadsDim[3], maxGridSize[3], sharedMemPerBlock,
        // totalConstantMemory, SIMDWidth, memPitch, regsPerBlock, clockRate, textureAlign.
        const ATTRS: [c_int; 14] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14];
        let mut values = [0 as c_int; 14];
        for (slot, attr) in values.iter_mut().zip(ATTRS) {
            let result = unsafe { cuDeviceGetAttribute(slot, attr, device) };
            if result != Success {
                return result;
            }
        }
        unsafe { std::ptr::copy_nonoverlapping(values.as_ptr(), prop, values.len()) };
        Success
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuProfilerStart() -> CuResult {
    Success
}

#[unsafe(no_mangle)]
pub extern "system" fn cuProfilerStop() -> CuResult {
    Success
}

#[unsafe(no_mangle)]
pub extern "system" fn cuProfilerInitialize(
    _config: *const c_char,
    _output: *const c_char,
    _mode: c_int,
) -> CuResult {
    Success
}

/// Options (JIT log buffers, optimization level) have no effect on the PTX translator.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleLoadDataEx(
    out: *mut Opaque,
    image: *const c_void,
    _num_options: c_uint,
    _options: *mut c_int,
    _values: *mut Opaque,
) -> CuResult {
    unsafe { cuModuleLoadData(out, image) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuModuleLoadFatBinary(out: *mut Opaque, image: *const c_void) -> CuResult {
    unsafe { cuModuleLoadData(out, image) }
}

fn check_stream(stream: Opaque) -> Result<(usize, usize), CuResult> {
    state(|state| stream_owner(state, stream))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamGetFlags(stream: Opaque, out: *mut c_uint) -> CuResult {
    trace::call("cuStreamGetFlags", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(check_stream(stream).and_then(|(_, key)| {
            let flags = STREAM_FLAGS
                .lock()
                .ok()
                .and_then(|guard| guard.as_ref().and_then(|map| map.get(&key).copied()))
                .unwrap_or(0);
            unsafe { out.write(flags) };
            Ok(())
        }))
    })
}

/// Only one priority level exists (see `cuCtxGetStreamPriorityRange`).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamGetPriority(stream: Opaque, out: *mut c_int) -> CuResult {
    trace::call("cuStreamGetPriority", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(check_stream(stream).map(|_| unsafe { out.write(0) }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamGetId(stream: Opaque, out: *mut u64) -> CuResult {
    trace::call("cuStreamGetId", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(check_stream(stream).map(|(_, key)| unsafe { out.write(key as u64) }))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamGetDevice(stream: Opaque, out: *mut c_int) -> CuResult {
    trace::call("cuStreamGetDevice", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(state(|state| {
            let (owner, _) = stream_owner(state, stream)?;
            let device = state.contexts.get(&owner).ok_or(ContextIsDestroyed)?.device;
            unsafe { out.write(device) };
            Ok(())
        }))
    })
}

/// The green-context out parameter is always null (green contexts are not supported).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamGetCtx_v2(
    stream: Opaque,
    ctx: *mut Opaque,
    green: *mut Opaque,
) -> CuResult {
    trace::call("cuStreamGetCtx_v2", || {
        if ctx.is_null() {
            return InvalidValue;
        }
        status(check_stream(stream).map(|(owner, _)| unsafe {
            ctx.write(std::ptr::without_provenance_mut(owner));
            if !green.is_null() {
                green.write(std::ptr::null_mut());
            }
        }))
    })
}

type StreamCallback = unsafe extern "system" fn(Opaque, c_int, Opaque);

struct Pending {
    callback: StreamCallback,
    stream: usize,
    data: usize,
}

unsafe extern "system" fn run_callback(data: Opaque) {
    let pending = unsafe { Box::from_raw(data.cast::<Pending>()) };
    unsafe {
        (pending.callback)(std::ptr::without_provenance_mut(pending.stream), 0, std::ptr::without_provenance_mut(pending.data))
    };
}

/// Implemented on top of `cuLaunchHostFunc`; the callback always sees `CUDA_SUCCESS`
/// because failed work surfaces on the stream calls themselves.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamAddCallback(
    stream: Opaque,
    callback: Option<StreamCallback>,
    data: Opaque,
    flags: c_uint,
) -> CuResult {
    trace::call("cuStreamAddCallback", || {
        let Some(callback) = callback else { return InvalidValue };
        if flags != 0 {
            return InvalidValue;
        }
        if let Err(error) = check_stream(stream) {
            return error;
        }
        let pending = Box::into_raw(Box::new(Pending {
            callback,
            stream: stream.addr(),
            data: data.addr(),
        }));
        let result = unsafe {
            crate::cuda::graph_api::cuLaunchHostFunc(stream, run_callback as *const () as Opaque, pending.cast())
        };
        if result != Success {
            drop(unsafe { Box::from_raw(pending) });
        }
        result
    })
}

/// Row-wise 2-D memset; `pitch` is in bytes, `width`/`height` in elements/rows.
fn memset_2d(dst: u64, pitch: usize, width: usize, height: usize, elem: usize, fill: impl Fn(u64, usize) -> CuResult) -> CuResult {
    if width.checked_mul(elem).is_none_or(|bytes| bytes > pitch && height > 1) {
        return InvalidValue;
    }
    for row in 0..height {
        let Some(offset) = pitch.checked_mul(row) else { return InvalidValue };
        let result = fill(dst.wrapping_add(offset as u64), width);
        if result != Success {
            return result;
        }
    }
    Success
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD2D8_v2(dst: u64, pitch: usize, value: u8, width: usize, height: usize) -> CuResult {
    trace::call("cuMemsetD2D8_v2", || memset_2d(dst, pitch, width, height, 1, |at, n| cuMemsetD8_v2(at, value, n)))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD2D16_v2(dst: u64, pitch: usize, value: u16, width: usize, height: usize) -> CuResult {
    trace::call("cuMemsetD2D16_v2", || memset_2d(dst, pitch, width, height, 2, |at, n| cuMemsetD16_v2(at, value, n)))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD2D32_v2(dst: u64, pitch: usize, value: u32, width: usize, height: usize) -> CuResult {
    trace::call("cuMemsetD2D32_v2", || memset_2d(dst, pitch, width, height, 4, |at, n| cuMemsetD32_v2(at, value, n)))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD2D8Async(dst: u64, pitch: usize, value: u8, width: usize, height: usize, stream: Opaque) -> CuResult {
    trace::call("cuMemsetD2D8Async", || memset_2d(dst, pitch, width, height, 1, |at, n| cuMemsetD8Async(at, value, n, stream)))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD2D16Async(dst: u64, pitch: usize, value: u16, width: usize, height: usize, stream: Opaque) -> CuResult {
    trace::call("cuMemsetD2D16Async", || memset_2d(dst, pitch, width, height, 2, |at, n| cuMemsetD16Async(at, value, n, stream)))
}

#[unsafe(no_mangle)]
pub extern "system" fn cuMemsetD2D32Async(dst: u64, pitch: usize, value: u32, width: usize, height: usize, stream: Opaque) -> CuResult {
    trace::call("cuMemsetD2D32Async", || memset_2d(dst, pitch, width, height, 4, |at, n| cuMemsetD32Async(at, value, n, stream)))
}

/// Total blocks a cooperative grid may have: resident blocks per multiprocessor times
/// the multiprocessor count. `None` means the grid does not fit (or overflows).
pub(crate) fn cooperative_fits(grid: [u32; 3], blocks_per_mp: u32, multiprocessors: u32) -> bool {
    let blocks = grid.iter().try_fold(1u64, |total, &dim| total.checked_mul(dim as u64));
    match blocks {
        Some(blocks) => blocks > 0 && blocks <= blocks_per_mp as u64 * multiprocessors as u64,
        None => false,
    }
}

/// All blocks must be co-resident for grid-wide synchronization to make progress, so a
/// larger grid is refused with `CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE` (720). Grids that
/// fit are launched like an ordinary kernel.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuLaunchCooperativeKernel(
    function: Opaque,
    grid_x: c_uint,
    grid_y: c_uint,
    grid_z: c_uint,
    block_x: c_uint,
    block_y: c_uint,
    block_z: c_uint,
    shared: c_uint,
    stream: Opaque,
    params: *mut Opaque,
) -> CuResult {
    trace::call("cuLaunchCooperativeKernel", || {
        let Some(block) = block_x.checked_mul(block_y).and_then(|n| n.checked_mul(block_z)) else {
            return InvalidValue;
        };
        if block == 0 || block > c_int::MAX as u32 || grid_x == 0 || grid_y == 0 || grid_z == 0 {
            return InvalidValue;
        }
        let mut per_mp = 0;
        let result = unsafe {
            cuOccupancyMaxActiveBlocksPerMultiprocessor(&mut per_mp, function, block as c_int, shared as usize)
        };
        if result != Success {
            return result;
        }
        let mut device = 0;
        let mut count = 0;
        let result = status(with_context(|_, context| {
            device = context.device;
            Ok(())
        }));
        if result != Success {
            return result;
        }
        let result = unsafe { cuDeviceGetAttribute(&mut count, 16, device) };
        if result != Success {
            return result;
        }
        if !cooperative_fits([grid_x, grid_y, grid_z], per_mp.max(0) as u32, count.max(0) as u32) {
            return CooperativeLaunchTooLarge;
        }
        unsafe {
            cuLaunchKernel(function, grid_x, grid_y, grid_z, block_x, block_y, block_z, shared, stream, params, std::ptr::null_mut())
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cuda::streams::cuStreamDestroy_v2;
    use crate::cuda::streams::fake_tests::{Fixture, context, stream};
    use std::ptr::null_mut;

    #[test]
    fn limit_defaults_and_validation() {
        assert_eq!(default_limit(0), Some(1024));
        assert_eq!(default_limit(2), Some(8 << 20));
        assert_eq!(default_limit(7), None);
        assert_eq!(default_limit(-1), None);
        assert!(valid_cache_config(3) && !valid_cache_config(4) && !valid_cache_config(-1));
        assert!(valid_shared_mem_config(2) && !valid_shared_mem_config(3));
    }

    #[test]
    fn cooperative_grid_must_fit_resident_capacity() {
        assert!(cooperative_fits([4, 1, 1], 2, 2));
        assert!(!cooperative_fits([5, 1, 1], 2, 2));
        assert!(!cooperative_fits([0, 1, 1], 2, 2));
        assert!(!cooperative_fits([u32::MAX, u32::MAX, u32::MAX], u32::MAX, u32::MAX));
        assert!(!cooperative_fits([1, 1, 1], 0, 8));
    }

    #[test]
    fn memset_2d_walks_rows_and_rejects_bad_geometry() {
        let rows = std::cell::RefCell::new(Vec::new());
        let record = |at: u64, n: usize| {
            rows.borrow_mut().push((at, n));
            Success
        };
        assert_eq!(memset_2d(1000, 64, 16, 3, 2, record), Success);
        assert_eq!(*rows.borrow(), vec![(1000, 16), (1064, 16), (1128, 16)]);
        // width in bytes (16 * 4) exceeds the pitch with more than one row
        assert_eq!(memset_2d(0, 32, 16, 2, 4, |_, _| Success), InvalidValue);
        assert_eq!(memset_2d(0, 32, usize::MAX, 1, 4, |_, _| Success), InvalidValue);
        // failure of one row stops the walk
        assert_eq!(memset_2d(0, 64, 4, 3, 1, |_, _| OutOfMemory), OutOfMemory);
    }

    #[test]
    fn context_limits_cache_and_ids() {
        let _fixture = Fixture::new();
        let mut value = 0usize;
        assert_eq!(unsafe { cuCtxGetLimit(&mut value, 0) }, InvalidContext);
        let ctx = context(None);
        assert_eq!(unsafe { cuCtxGetLimit(&mut value, 0) }, Success);
        assert_eq!(value, 1024);
        assert_eq!(cuCtxSetLimit(0, 4096), Success);
        assert_eq!(unsafe { cuCtxGetLimit(&mut value, 0) }, Success);
        assert_eq!(value, 4096);
        assert_eq!(cuCtxSetLimit(99, 1), InvalidValue);
        assert_eq!(unsafe { cuCtxGetLimit(&mut value, 99) }, InvalidValue);
        assert_eq!(unsafe { cuCtxGetLimit(null_mut(), 0) }, InvalidValue);

        let mut cache = -1;
        assert_eq!(cuCtxSetCacheConfig(2), Success);
        assert_eq!(cuCtxSetCacheConfig(9), InvalidValue);
        assert_eq!(unsafe { cuCtxGetCacheConfig(&mut cache) }, Success);
        assert_eq!(cache, 2);
        assert_eq!(cuCtxSetSharedMemConfig(1), Success);
        assert_eq!(cuCtxSetSharedMemConfig(5), InvalidValue);
        assert_eq!(cuCtxSetFlags(0), Success);

        let (mut a, mut b) = (0u64, 0u64);
        assert_eq!(unsafe { cuCtxGetId(null_mut(), &mut a) }, Success);
        assert_eq!(unsafe { cuCtxGetId(ctx, &mut b) }, Success);
        assert_eq!(a, b);
        assert_eq!(unsafe { cuCtxGetId(std::ptr::without_provenance_mut(0xdead), &mut b) }, InvalidContext);
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
    }

    #[test]
    fn stream_queries() {
        let _fixture = Fixture::new();
        let ctx = context(None);
        let s = stream();
        let (mut flags, mut priority, mut id, mut device) = (9u32, 9, 0u64, -1);
        assert_eq!(unsafe { cuStreamGetFlags(s, &mut flags) }, Success);
        assert_eq!(flags, 1);
        assert_eq!(unsafe { cuStreamGetFlags(null_mut(), &mut flags) }, Success);
        assert_eq!(flags, 0);
        assert_eq!(unsafe { cuStreamGetPriority(s, &mut priority) }, Success);
        assert_eq!(priority, 0);
        assert_eq!(unsafe { cuStreamGetId(s, &mut id) }, Success);
        assert_eq!(unsafe { cuStreamGetDevice(s, &mut device) }, Success);
        assert_eq!(device, 0);
        let mut owner = null_mut();
        let mut green = std::ptr::without_provenance_mut(1);
        assert_eq!(unsafe { cuStreamGetCtx_v2(s, &mut owner, &mut green) }, Success);
        assert_eq!(owner, ctx);
        assert!(green.is_null());
        let bogus = std::ptr::without_provenance_mut(0xbeef);
        assert_eq!(unsafe { cuStreamGetFlags(bogus, &mut flags) }, InvalidHandle);
        assert_eq!(unsafe { cuStreamGetFlags(s, null_mut()) }, InvalidValue);
        assert_eq!(cuStreamDestroy_v2(s), Success);
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
    }

    #[test]
    fn profiler_calls_succeed_and_device_properties_validate() {
        assert_eq!(cuProfilerStart(), Success);
        assert_eq!(cuProfilerStop(), Success);
        assert_eq!(cuProfilerInitialize(std::ptr::null(), std::ptr::null(), 0), Success);
        assert_eq!(unsafe { cuDeviceGetProperties(null_mut(), 0) }, InvalidValue);
        assert_eq!(unsafe { cuDeviceComputeCapability(null_mut(), null_mut(), 0) }, InvalidValue);
    }

    #[test]
    fn stream_callback_rejects_bad_arguments() {
        let _fixture = Fixture::new();
        let ctx = context(None);
        assert_eq!(unsafe { cuStreamAddCallback(null_mut(), None, null_mut(), 0) }, InvalidValue);
        unsafe extern "system" fn cb(_: Opaque, _: c_int, _: Opaque) {}
        assert_eq!(unsafe { cuStreamAddCallback(null_mut(), Some(cb), null_mut(), 1) }, InvalidValue);
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
    }

    #[test]
    fn new_entry_points_are_discoverable() {
        use crate::cuda::discovery::cuGetProcAddress_v2;
        for name in [
            c"cuCtxGetLimit", c"cuCtxSetLimit", c"cuCtxGetCacheConfig", c"cuCtxGetId",
            c"cuStreamGetFlags", c"cuStreamGetPriority", c"cuStreamAddCallback",
            c"cuModuleLoadDataEx", c"cuMemsetD2D8", c"cuMemsetD2D32Async",
            c"cuLaunchCooperativeKernel", c"cuProfilerStart", c"cuDeviceGetProperties",
        ] {
            let mut pointer = null_mut();
            let mut query = -1;
            let result = unsafe { cuGetProcAddress_v2(name.as_ptr(), &mut pointer, 13000, 0, &mut query) };
            assert_eq!((result, query), (Success, 0), "{name:?}");
            assert!(!pointer.is_null(), "{name:?}");
        }
    }
}
