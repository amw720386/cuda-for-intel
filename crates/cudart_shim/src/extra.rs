//! Runtime API entry points beyond the ones ggml-cuda needs: events and streams,
//! synchronous and 2-D copies, stream-ordered allocation, pointer attributes, device
//! limits, driver entry points and error strings. Pure helpers are unit tested without a
//! driver; every entry point returns a CUDA error code (never panics) when the driver
//! cannot be loaded.
use crate::errors::error_info;
use crate::*;

type DriverStatus = c_int;

/// Look up an exported driver function by name.
macro_rules! drv_fn {
    ($drv:expr, $name:literal, $ty:ty) => {
        match unsafe { $drv._lib.get::<$ty>(concat!($name, "\0").as_bytes()) } {
            Ok(symbol) => *symbol,
            Err(_) => return set_error(NOT_SUPPORTED),
        }
    };
}

/// Run `body` with a loaded driver and the device's primary context current.
fn with_driver(body: impl FnOnce(&'static Driver) -> c_int) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    set_error(body(drv))
}

// ---------------------------------------------------------------------------
// Error strings
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGetErrorName(error: c_int) -> *const c_char {
    error_info(error).map_or(c"cudaErrorUnknown".as_ptr(), |(name, _)| name.as_ptr())
}

pub(crate) fn error_text(error: c_int) -> *const c_char {
    error_info(error).map_or(c"unknown error".as_ptr(), |(_, text)| text.as_ptr())
}

// ---------------------------------------------------------------------------
// Events
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaEventCreate(event: *mut CUevent) -> c_int {
    unsafe { cudaEventCreateWithFlags(event, 0) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaEventQuery(event: CUevent) -> c_int {
    // cudaErrorNotReady is a status, not a sticky error.
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let query = drv_fn!(drv, "cuEventQuery", unsafe extern "system" fn(CUevent) -> DriverStatus);
    match unsafe { query(event) } {
        600 => 600,
        other => set_error(other),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaEventElapsedTime(ms: *mut f32, start: CUevent, end: CUevent) -> c_int {
    if ms.is_null() {
        return set_error(INVALID_VALUE);
    }
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let elapsed = drv_fn!(drv, "cuEventElapsedTime", unsafe extern "system" fn(*mut f32, CUevent, CUevent) -> DriverStatus);
    set_error(unsafe { elapsed(ms, start, end) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaEventElapsedTime_v2(ms: *mut f32, start: CUevent, end: CUevent) -> c_int {
    unsafe { cudaEventElapsedTime(ms, start, end) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaEventRecordWithFlags(event: CUevent, stream: CUstream, flags: c_uint) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let record = drv_fn!(drv, "cuEventRecordWithFlags", unsafe extern "system" fn(CUevent, CUstream, c_uint) -> DriverStatus);
    set_error(unsafe { record(event, stream, flags) })
}

// ---------------------------------------------------------------------------
// Streams
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamCreate(stream: *mut CUstream) -> c_int {
    unsafe { cudaStreamCreateWithFlags(stream, 0) }
}

/// Only one priority level exists, so the priority is validated and clamped to 0.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamCreateWithPriority(stream: *mut CUstream, flags: c_uint, _priority: c_int) -> c_int {
    unsafe { cudaStreamCreateWithFlags(stream, flags) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> c_int {
    unsafe {
        if !least.is_null() {
            least.write(0);
        }
        if !greatest.is_null() {
            greatest.write(0);
        }
    }
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamQuery(stream: CUstream) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let query = drv_fn!(drv, "cuStreamQuery", unsafe extern "system" fn(CUstream) -> DriverStatus);
    match unsafe { query(stream) } {
        600 => 600,
        other => set_error(other),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamGetFlags(stream: CUstream, flags: *mut c_uint) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let get = drv_fn!(drv, "cuStreamGetFlags", unsafe extern "system" fn(CUstream, *mut c_uint) -> DriverStatus);
    set_error(unsafe { get(stream, flags) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamGetPriority(stream: CUstream, priority: *mut c_int) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let get = drv_fn!(drv, "cuStreamGetPriority", unsafe extern "system" fn(CUstream, *mut c_int) -> DriverStatus);
    set_error(unsafe { get(stream, priority) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamGetId(stream: CUstream, id: *mut u64) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let get = drv_fn!(drv, "cuStreamGetId", unsafe extern "system" fn(CUstream, *mut u64) -> DriverStatus);
    set_error(unsafe { get(stream, id) })
}

type StreamCallback = Option<unsafe extern "system" fn(CUstream, c_int, *mut c_void)>;

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamAddCallback(stream: CUstream, callback: StreamCallback, data: *mut c_void, flags: c_uint) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let add = drv_fn!(drv, "cuStreamAddCallback", unsafe extern "system" fn(CUstream, StreamCallback, *mut c_void, c_uint) -> DriverStatus);
    set_error(unsafe { add(stream, callback, data, flags) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaLaunchHostFunc(stream: CUstream, function: *mut c_void, data: *mut c_void) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let launch = drv_fn!(drv, "cuLaunchHostFunc", unsafe extern "system" fn(CUstream, *mut c_void, *mut c_void) -> DriverStatus);
    set_error(unsafe { launch(stream, function, data) })
}

// ---------------------------------------------------------------------------
// Synchronous and 2-D copies, memsets
// ---------------------------------------------------------------------------

/// `cudaMemcpy`: ordered after prior work on the legacy stream, complete on return.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemcpy(dst: *mut c_void, src: *const c_void, count: usize, kind: c_int) -> c_int {
    let err = unsafe { cudaMemcpyAsync(dst, src, count, kind, std::ptr::null_mut()) };
    if err != SUCCESS {
        return err;
    }
    unsafe { cudaStreamSynchronize(std::ptr::null_mut()) }
}

/// Validate 2-D copy geometry: rows must not overlap the pitch.
pub(crate) fn valid_2d(dpitch: usize, spitch: usize, width: usize, height: usize) -> bool {
    height == 0 || width == 0 || (width <= dpitch && width <= spitch)
}

pub(crate) unsafe fn copy_2d_rows(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: c_int,
    stream: CUstream,
) -> c_int {
    if !valid_2d(dpitch, spitch, width, height) {
        return set_error(INVALID_VALUE);
    }
    if width == 0 || height == 0 {
        return set_error(SUCCESS);
    }
    if dst.is_null() || src.is_null() {
        return set_error(INVALID_VALUE);
    }
    for row in 0..height {
        let d = unsafe { dst.cast::<u8>().add(row * dpitch) }.cast::<c_void>();
        let s = unsafe { src.cast::<u8>().add(row * spitch) }.cast::<c_void>();
        let err = unsafe { cudaMemcpyAsync(d, s, width, kind, stream) };
        if err != SUCCESS {
            return err;
        }
    }
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemcpy2D(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, width: usize, height: usize, kind: c_int) -> c_int {
    let err = unsafe { copy_2d_rows(dst, dpitch, src, spitch, width, height, kind, std::ptr::null_mut()) };
    if err != SUCCESS {
        return err;
    }
    unsafe { cudaStreamSynchronize(std::ptr::null_mut()) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemset2DAsync(dev_ptr: *mut c_void, pitch: usize, value: c_int, width: usize, height: usize, stream: CUstream) -> c_int {
    if !valid_2d(pitch, pitch, width, height) {
        return set_error(INVALID_VALUE);
    }
    for row in 0..height {
        let at = unsafe { dev_ptr.cast::<u8>().add(row * pitch) }.cast::<c_void>();
        let err = unsafe { cudaMemsetAsync(at, value, width, stream) };
        if err != SUCCESS {
            return err;
        }
    }
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemset2D(dev_ptr: *mut c_void, pitch: usize, value: c_int, width: usize, height: usize) -> c_int {
    let err = unsafe { cudaMemset2DAsync(dev_ptr, pitch, value, width, height, std::ptr::null_mut()) };
    if err != SUCCESS {
        return err;
    }
    unsafe { cudaStreamSynchronize(std::ptr::null_mut()) }
}

/// Device-to-device copy between devices: only the single-device case is available.
pub(crate) fn peer_copy_supported(dst_device: c_int, src_device: c_int) -> bool {
    dst_device == src_device
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemcpyPeer(dst: *mut c_void, dst_device: c_int, src: *const c_void, src_device: c_int, count: usize) -> c_int {
    if !peer_copy_supported(dst_device, src_device) {
        return set_error(NOT_SUPPORTED);
    }
    unsafe { cudaMemcpy(dst, src, count, 3) }
}

/// Round a row width up to the pitch alignment (512 bytes, as on NVIDIA devices).
pub(crate) fn pitch_for(width: usize) -> Option<usize> {
    const ALIGN: usize = 512;
    width.max(1).checked_add(ALIGN - 1).map(|n| n & !(ALIGN - 1))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMallocPitch(dev_ptr: *mut *mut c_void, pitch: *mut usize, width: usize, height: usize) -> c_int {
    if dev_ptr.is_null() || pitch.is_null() {
        return set_error(INVALID_VALUE);
    }
    let Some(row) = pitch_for(width) else {
        return set_error(INVALID_VALUE);
    };
    let Some(total) = row.checked_mul(height) else {
        return set_error(INVALID_VALUE);
    };
    let err = unsafe { cudaMalloc(dev_ptr, total) };
    if err == SUCCESS {
        unsafe { pitch.write(row) };
    }
    err
}

/// Stream-ordered allocation: allocation is immediate (the driver allocator is already
/// safe to use on any stream) and free first drains the stream.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMallocAsync(dev_ptr: *mut *mut c_void, size: usize, _stream: CUstream) -> c_int {
    unsafe { cudaMalloc(dev_ptr, size) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaFreeAsync(dev_ptr: *mut c_void, stream: CUstream) -> c_int {
    if dev_ptr.is_null() {
        return set_error(SUCCESS);
    }
    let err = unsafe { cudaStreamSynchronize(stream) };
    if err != SUCCESS {
        return err;
    }
    unsafe { cudaFree(dev_ptr) }
}

/// `struct cudaPointerAttributes`: type, device, devicePointer, hostPointer.
#[repr(C)]
struct PointerAttributes {
    kind: c_int,
    device: c_int,
    device_pointer: *mut c_void,
    host_pointer: *mut c_void,
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaPointerGetAttributes(attributes: *mut c_void, ptr: *const c_void) -> c_int {
    if attributes.is_null() {
        return set_error(INVALID_VALUE);
    }
    let device = is_device_pointer(ptr as usize);
    let value = PointerAttributes {
        kind: if device { 2 } else { 0 },
        device: if device { runtime().lock().map(|rt| rt.device).unwrap_or(0) } else { -2 },
        device_pointer: if device { ptr.cast_mut() } else { std::ptr::null_mut() },
        host_pointer: std::ptr::null_mut(),
    };
    unsafe { attributes.cast::<PointerAttributes>().write(value) };
    set_error(SUCCESS)
}

// ---------------------------------------------------------------------------
// Device queries and configuration
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGetDeviceProperties(prop: *mut c_void, device: c_int) -> c_int {
    unsafe { cudaGetDeviceProperties_v2(prop.cast(), device) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaChooseDevice(device: *mut c_int, _prop: *const c_void) -> c_int {
    if device.is_null() {
        return set_error(INVALID_VALUE);
    }
    let mut count = 0;
    let err = unsafe { cudaGetDeviceCount(&mut count) };
    if err != SUCCESS {
        return err;
    }
    if count == 0 {
        return set_error(100);
    }
    unsafe { device.write(0) };
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaInitDevice(device: c_int, _flags: c_uint, _device_flags: c_uint) -> c_int {
    unsafe { cudaSetDevice(device) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGetDeviceFlags(flags: *mut c_uint) -> c_int {
    if flags.is_null() {
        return set_error(INVALID_VALUE);
    }
    unsafe { flags.write(0) };
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaThreadSynchronize() -> c_int {
    unsafe { cudaDeviceSynchronize() }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceGetByPCIBusId(device: *mut c_int, pci_bus_id: *const c_char) -> c_int {
    if device.is_null() || pci_bus_id.is_null() {
        return set_error(INVALID_VALUE);
    }
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let get = drv_fn!(drv, "cuDeviceGetByPCIBusId", unsafe extern "system" fn(*mut c_int, *const c_char) -> DriverStatus);
    set_error(unsafe { get(device, pci_bus_id) })
}

/// `value` is always 0: peer access, native atomics and array access are unsupported.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceGetP2PAttribute(value: *mut c_int, _attr: c_int, _src: c_int, _dst: c_int) -> c_int {
    if value.is_null() {
        return set_error(INVALID_VALUE);
    }
    unsafe { value.write(0) };
    set_error(SUCCESS)
}

/// cudaLimit values match CUlimit for 0..=6.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceGetLimit(value: *mut usize, limit: c_int) -> c_int {
    if value.is_null() {
        return set_error(INVALID_VALUE);
    }
    with_driver(|drv| {
        let get = drv_fn_raw::<unsafe extern "system" fn(*mut usize, c_int) -> DriverStatus>(drv, b"cuCtxGetLimit\0");
        match get {
            Some(get) => unsafe { get(value, limit) },
            None => NOT_SUPPORTED,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceSetLimit(limit: c_int, value: usize) -> c_int {
    with_driver(|drv| match drv_fn_raw::<unsafe extern "system" fn(c_int, usize) -> DriverStatus>(drv, b"cuCtxSetLimit\0") {
        Some(set) => unsafe { set(limit, value) },
        None => NOT_SUPPORTED,
    })
}

fn drv_fn_raw<T: Copy>(drv: &Driver, name: &[u8]) -> Option<T> {
    unsafe { drv._lib.get::<T>(name) }.ok().map(|symbol| *symbol)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceGetCacheConfig(config: *mut c_int) -> c_int {
    if config.is_null() {
        return set_error(INVALID_VALUE);
    }
    with_driver(|drv| match drv_fn_raw::<unsafe extern "system" fn(*mut c_int) -> DriverStatus>(drv, b"cuCtxGetCacheConfig\0") {
        Some(get) => unsafe { get(config) },
        None => NOT_SUPPORTED,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceSetCacheConfig(config: c_int) -> c_int {
    with_driver(|drv| match drv_fn_raw::<unsafe extern "system" fn(c_int) -> DriverStatus>(drv, b"cuCtxSetCacheConfig\0") {
        Some(set) => unsafe { set(config) },
        None => NOT_SUPPORTED,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceGetSharedMemConfig(config: *mut c_int) -> c_int {
    if config.is_null() {
        return set_error(INVALID_VALUE);
    }
    with_driver(|drv| match drv_fn_raw::<unsafe extern "system" fn(*mut c_int) -> DriverStatus>(drv, b"cuCtxGetSharedMemConfig\0") {
        Some(get) => unsafe { get(config) },
        None => NOT_SUPPORTED,
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceSetSharedMemConfig(config: c_int) -> c_int {
    with_driver(|drv| match drv_fn_raw::<unsafe extern "system" fn(c_int) -> DriverStatus>(drv, b"cuCtxSetSharedMemConfig\0") {
        Some(set) => unsafe { set(config) },
        None => NOT_SUPPORTED,
    })
}

// ---------------------------------------------------------------------------
// Driver entry points
// ---------------------------------------------------------------------------

type GetProcAddress = unsafe extern "system" fn(*const c_char, *mut *mut c_void, c_int, u64, *mut c_int) -> DriverStatus;

fn entry_point(symbol: *const c_char, func: *mut *mut c_void, version: c_int, flags: u64, status: *mut c_int) -> c_int {
    if symbol.is_null() || func.is_null() {
        return set_error(INVALID_VALUE);
    }
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let Some(resolve) = drv_fn_raw::<GetProcAddress>(drv, b"cuGetProcAddress_v2\0") else {
        return set_error(NOT_SUPPORTED);
    };
    let mut outcome = 1;
    let result = unsafe { resolve(symbol, func, version, flags, &mut outcome) };
    if !status.is_null() {
        unsafe { status.write(outcome) };
    }
    set_error(result)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGetDriverEntryPoint(symbol: *const c_char, func: *mut *mut c_void, flags: u64, status: *mut c_int) -> c_int {
    entry_point(symbol, func, 12090, flags, status)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGetDriverEntryPointByVersion(symbol: *const c_char, func: *mut *mut c_void, version: c_uint, flags: u64, status: *mut c_int) -> c_int {
    entry_point(symbol, func, version as c_int, flags, status)
}

// ---------------------------------------------------------------------------
// Occupancy and cooperative launch
// ---------------------------------------------------------------------------

fn resolved_function(drv: &Driver, host: *const c_void) -> Result<usize, c_int> {
    let key = host as usize;
    let cached = runtime().lock().ok().and_then(|rt| rt.functions.get(&key).copied());
    if let Some(function) = cached {
        return Ok(function);
    }
    resolve_function(drv, key)?;
    runtime().lock().ok().and_then(|rt| rt.functions.get(&key).copied()).ok_or(INVALID_VALUE)
}

/// Driver-backed occupancy of a registered host kernel stub.
pub(crate) fn driver_occupancy(drv: &Driver, func: *const c_void, block: c_int, shared: usize, flags: c_uint) -> Option<c_int> {
    let function = resolved_function(drv, func).ok()?;
    let query = drv_fn_raw::<unsafe extern "system" fn(*mut c_int, CUfunction, c_int, usize, c_uint) -> DriverStatus>(drv, b"cuOccupancyMaxActiveBlocksPerMultiprocessorWithFlags\0")?;
    let mut blocks = 0;
    (unsafe { query(&mut blocks, function as CUfunction, block, shared, flags) } == SUCCESS).then_some(blocks.max(1))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(num_blocks: *mut c_int, func: *const c_void, block_size: c_int, dynamic_smem: usize) -> c_int {
    unsafe { cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(num_blocks, func, block_size, dynamic_smem, 0) }
}

/// Resident blocks for one cooperative grid: `blocks_per_mp * multiprocessors`.
pub(crate) fn cooperative_capacity(blocks_per_mp: c_int, multiprocessors: c_int) -> u64 {
    blocks_per_mp.max(0) as u64 * multiprocessors.max(0) as u64
}

pub(crate) fn cooperative_fits(grid: [c_uint; 3], capacity: u64) -> bool {
    let blocks = grid.iter().try_fold(1u64, |total, &dim| total.checked_mul(dim as u64));
    matches!(blocks, Some(blocks) if blocks > 0 && blocks <= capacity)
}

/// Returns the co-resident block capacity of `func` at this block shape, if known.
pub(crate) unsafe fn cooperative_limit(func: *const c_void, block_threads: c_int, shared: usize) -> Option<u64> {
    let drv = driver().ok()?;
    if ensure_context(drv) != SUCCESS {
        return None;
    }
    let function = resolved_function(drv, func).ok()?;
    let occupancy = drv_fn_raw::<unsafe extern "system" fn(*mut c_int, CUfunction, c_int, usize) -> DriverStatus>(drv, b"cuOccupancyMaxActiveBlocksPerMultiprocessor\0")?;
    let mut per_mp = 0;
    if unsafe { occupancy(&mut per_mp, function as CUfunction, block_threads, shared) } != SUCCESS {
        return None;
    }
    let attribute = drv.cuDeviceGetAttribute;
    let device = runtime().lock().map(|rt| rt.device).unwrap_or(0);
    let mut count = 0;
    if unsafe { attribute(&mut count, 16, device) } != SUCCESS {
        return None;
    }
    Some(cooperative_capacity(per_mp, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::ptr::{null, null_mut};

    fn text(ptr: *const c_char) -> String {
        unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned()
    }

    #[test]
    fn error_names_and_strings() {
        unsafe {
            assert_eq!(text(cudaGetErrorName(0)), "cudaSuccess");
            assert_eq!(text(cudaGetErrorName(1)), "cudaErrorInvalidValue");
            assert_eq!(text(cudaGetErrorName(801)), "cudaErrorNotSupported");
            assert_eq!(text(cudaGetErrorName(720)), "cudaErrorCooperativeLaunchTooLarge");
            assert_eq!(text(cudaGetErrorName(31337)), "cudaErrorUnknown");
            assert_eq!(text(cudaGetErrorString(0)), "no error");
            assert_eq!(text(cudaGetErrorString(2)), "out of memory");
            assert_eq!(text(cudaGetErrorString(209)), "no kernel image is available for execution on the device");
            assert_eq!(text(cudaGetErrorString(-5)), "unknown error");
        }
    }

    #[test]
    fn copy_geometry_and_pitch() {
        assert!(valid_2d(64, 64, 64, 4));
        assert!(!valid_2d(32, 64, 33, 4));
        assert!(!valid_2d(64, 16, 17, 2));
        assert!(valid_2d(0, 0, 0, 3));
        assert_eq!(pitch_for(1), Some(512));
        assert_eq!(pitch_for(512), Some(512));
        assert_eq!(pitch_for(513), Some(1024));
        assert_eq!(pitch_for(usize::MAX), None);
        assert!(peer_copy_supported(0, 0));
        assert!(!peer_copy_supported(0, 1));
    }

    #[test]
    fn cooperative_capacity_checks() {
        assert_eq!(cooperative_capacity(4, 8), 32);
        assert_eq!(cooperative_capacity(-1, 8), 0);
        assert!(cooperative_fits([4, 4, 2], 32));
        assert!(!cooperative_fits([33, 1, 1], 32));
        assert!(!cooperative_fits([0, 1, 1], 32));
        assert!(!cooperative_fits([u32::MAX; 3], u64::MAX));
    }

    #[test]
    fn stream_priority_range_is_single_level() {
        let (mut least, mut greatest) = (9, 9);
        assert_eq!(unsafe { cudaDeviceGetStreamPriorityRange(&mut least, &mut greatest) }, 0);
        assert_eq!((least, greatest), (0, 0));
        assert_eq!(unsafe { cudaDeviceGetStreamPriorityRange(null_mut(), null_mut()) }, 0);
    }

    #[test]
    fn pointer_attributes_of_unregistered_host_pointer() {
        let mut attrs = [0u64; 3];
        let host = 5u32;
        assert_eq!(unsafe { cudaPointerGetAttributes(attrs.as_mut_ptr().cast(), (&raw const host).cast()) }, 0);
        assert_eq!(attrs[0] as u32, 0, "cudaMemoryTypeUnregistered");
        assert_eq!(unsafe { cudaPointerGetAttributes(null_mut(), null()) }, 1);
    }

    #[test]
    fn null_arguments_are_rejected_without_a_driver() {
        unsafe {
            assert_eq!(cudaDeviceGetLimit(null_mut(), 0), 1);
            assert_eq!(cudaDeviceGetCacheConfig(null_mut()), 1);
            assert_eq!(cudaDeviceGetSharedMemConfig(null_mut()), 1);
            assert_eq!(cudaGetDeviceFlags(null_mut()), 1);
            assert_eq!(cudaEventElapsedTime(null_mut(), null_mut(), null_mut()), 1);
            assert_eq!(cudaMallocPitch(null_mut(), null_mut(), 8, 8), 1);
            assert_eq!(cudaDeviceGetP2PAttribute(null_mut(), 1, 0, 0), 1);
            assert_eq!(cudaGetDriverEntryPoint(null(), null_mut(), 0, null_mut()), 1);
            assert_eq!(copy_2d_rows(null_mut(), 8, null(), 8, 16, 1, 0, null_mut()), 1);
            assert_eq!(copy_2d_rows(null_mut(), 8, null(), 8, 0, 4, 0, null_mut()), 0);
        }
    }

    #[test]
    fn calls_without_a_driver_fail_with_codes_instead_of_panicking() {
        // No nvcuda is loadable in the unit-test environment: every call must return a
        // non-zero CUDA error and leave the process alive.
        unsafe {
            let mut event = null_mut();
            let r = cudaEventCreate(&mut event);
            assert_ne!(r, 0);
            assert_ne!(cudaEventQuery(null_mut()), 0);
            assert_ne!(cudaStreamQuery(null_mut()), 0);
            assert_ne!(cudaStreamGetFlags(null_mut(), null_mut()), 0);
            assert_ne!(cudaMemcpy(null_mut(), null(), 0, 0), 99999);
            assert_ne!(cudaStreamAddCallback(null_mut(), None, null_mut(), 0), 0);
            assert_ne!(cudaLaunchHostFunc(null_mut(), null_mut(), null_mut()), 0);
            assert_ne!(cudaDeviceSetLimit(0, 1), 0);
            assert_ne!(cudaMemcpyPeer(null_mut(), 0, null(), 1, 4), 0);
        }
    }
}
