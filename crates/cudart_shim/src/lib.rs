//! Minimal CUDA Runtime shim for ggml-cuda.dll on cuda-for-intel.
//!
//! Stock NVIDIA cudart64_12 probes the WDDM LUID then expects an NVIDIA kernel
//! driver (cudaErrorInsufficientDriver). This DLL implements the Runtime entry
//! points ggml imports by forwarding to our Driver API `nvcuda.dll`.
#![allow(non_snake_case)]

/// Verbose per-call diagnostics, enabled with `CFI_LOG=1` (read once).
#[allow(unused_macros)]
macro_rules! info_log {
    ($($arg:tt)*) => {
        if $crate::log_enabled() {
            eprintln!($($arg)*);
        }
    };
}

pub(crate) fn log_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var_os("CFI_LOG").is_some_and(|v| !v.is_empty() && v != "0")
    })
}

mod errors;
mod extra;
mod stubs;

use libloading::Library;
use std::collections::{BTreeMap, HashMap};
use std::ffi::{c_char, c_int, c_uint, c_ulonglong, c_void, CStr};
use std::sync::{Mutex, OnceLock};

const SUCCESS: c_int = 0;
const INVALID_VALUE: c_int = 1;
const NOT_INITIALIZED: c_int = 3;
const INVALID_DEVICE: c_int = 101;
const NOT_SUPPORTED: c_int = 801;
const UNKNOWN: c_int = 999;

type CuResult = c_int;
type CUdevice = c_int;
type CUcontext = *mut c_void;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUevent = *mut c_void;
type CUdeviceptr = u64;

struct Driver {
    _lib: Library,
    cuInit: unsafe extern "system" fn(c_uint) -> CuResult,
    cuDeviceGetCount: unsafe extern "system" fn(*mut c_int) -> CuResult,
    cuDeviceGet: unsafe extern "system" fn(*mut CUdevice, c_int) -> CuResult,
    cuDeviceGetName: unsafe extern "system" fn(*mut c_char, c_int, CUdevice) -> CuResult,
    cuDeviceTotalMem: unsafe extern "system" fn(*mut usize, CUdevice) -> CuResult,
    cuDeviceGetAttribute: unsafe extern "system" fn(*mut c_int, c_int, CUdevice) -> CuResult,
    cuDevicePrimaryCtxRetain: unsafe extern "system" fn(*mut CUcontext, CUdevice) -> CuResult,
    cuCtxSetCurrent: unsafe extern "system" fn(CUcontext) -> CuResult,
    cuCtxGetCurrent: unsafe extern "system" fn(*mut CUcontext) -> CuResult,
    cuCtxSynchronize: unsafe extern "system" fn() -> CuResult,
    cuMemAlloc: unsafe extern "system" fn(*mut CUdeviceptr, usize) -> CuResult,
    cuMemFree: unsafe extern "system" fn(CUdeviceptr) -> CuResult,
    cuMemHostAlloc: unsafe extern "system" fn(*mut *mut c_void, usize, c_uint) -> CuResult,
    cuMemFreeHost: unsafe extern "system" fn(*mut c_void) -> CuResult,
    cuMemcpyHtoD: unsafe extern "system" fn(CUdeviceptr, *const c_void, usize) -> CuResult,
    cuMemcpyDtoH: unsafe extern "system" fn(*mut c_void, CUdeviceptr, usize) -> CuResult,
    cuMemcpyHtoDAsync:
        unsafe extern "system" fn(CUdeviceptr, *const c_void, usize, CUstream) -> CuResult,
    cuMemcpyDtoHAsync:
        unsafe extern "system" fn(*mut c_void, CUdeviceptr, usize, CUstream) -> CuResult,
    cuMemcpyDtoDAsync:
        unsafe extern "system" fn(CUdeviceptr, CUdeviceptr, usize, CUstream) -> CuResult,
    cuMemcpyDtoD: unsafe extern "system" fn(CUdeviceptr, CUdeviceptr, usize) -> CuResult,
    cuMemsetD8: unsafe extern "system" fn(CUdeviceptr, u8, usize) -> CuResult,
    cuMemsetD8Async: unsafe extern "system" fn(CUdeviceptr, u8, usize, CUstream) -> CuResult,
    cuModuleLoadData: unsafe extern "system" fn(*mut CUmodule, *const c_void) -> CuResult,
    cuModuleGetFunction:
        unsafe extern "system" fn(*mut CUfunction, CUmodule, *const c_char) -> CuResult,
    cuLaunchKernel: unsafe extern "system" fn(
        CUfunction,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        c_uint,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    cuStreamCreate: unsafe extern "system" fn(*mut CUstream, c_uint) -> CuResult,
    cuStreamDestroy: unsafe extern "system" fn(CUstream) -> CuResult,
    cuStreamSynchronize: unsafe extern "system" fn(CUstream) -> CuResult,
    cuEventCreate: unsafe extern "system" fn(*mut CUevent, c_uint) -> CuResult,
    cuEventDestroy: unsafe extern "system" fn(CUevent) -> CuResult,
    cuEventRecord: unsafe extern "system" fn(CUevent, CUstream) -> CuResult,
    cuEventSynchronize: unsafe extern "system" fn(CUevent) -> CuResult,
    cuStreamWaitEvent: unsafe extern "system" fn(CUstream, CUevent, c_uint) -> CuResult,
    cuMemGetInfo: unsafe extern "system" fn(*mut usize, *mut usize) -> CuResult,
    cuGetErrorString: unsafe extern "system" fn(CuResult, *mut *const c_char) -> CuResult,
}

fn driver() -> Result<&'static Driver, c_int> {
    static DRIVER: OnceLock<Result<Driver, c_int>> = OnceLock::new();
    match DRIVER.get_or_init(load_driver) {
        Ok(driver) => Ok(driver),
        Err(code) => Err(*code),
    }
}

fn load_driver() -> Result<Driver, c_int> {
    unsafe {
        let lib = Library::new("nvcuda.dll").map_err(|_| NOT_INITIALIZED)?;
        macro_rules! sym {
            ($name:literal) => {
                *lib.get(concat!($name, "\0").as_bytes())
                    .map_err(|_| NOT_INITIALIZED)?
            };
        }
        let cuInit: unsafe extern "system" fn(c_uint) -> CuResult = sym!("cuInit");
        let err = cuInit(0);
        if err != SUCCESS {
            return Err(err);
        }
        Ok(Driver {
            cuInit,
            cuDeviceGetCount: sym!("cuDeviceGetCount"),
            cuDeviceGet: sym!("cuDeviceGet"),
            cuDeviceGetName: sym!("cuDeviceGetName"),
            cuDeviceTotalMem: sym!("cuDeviceTotalMem_v2"),
            cuDeviceGetAttribute: sym!("cuDeviceGetAttribute"),
            cuDevicePrimaryCtxRetain: sym!("cuDevicePrimaryCtxRetain"),
            cuCtxSetCurrent: sym!("cuCtxSetCurrent"),
            cuCtxGetCurrent: sym!("cuCtxGetCurrent"),
            cuCtxSynchronize: sym!("cuCtxSynchronize"),
            cuMemAlloc: sym!("cuMemAlloc_v2"),
            cuMemFree: sym!("cuMemFree_v2"),
            cuMemHostAlloc: sym!("cuMemHostAlloc"),
            cuMemFreeHost: sym!("cuMemFreeHost"),
            cuMemcpyHtoD: sym!("cuMemcpyHtoD_v2"),
            cuMemcpyDtoH: sym!("cuMemcpyDtoH_v2"),
            cuMemcpyHtoDAsync: sym!("cuMemcpyHtoDAsync_v2"),
            cuMemcpyDtoHAsync: sym!("cuMemcpyDtoHAsync_v2"),
            cuMemcpyDtoDAsync: sym!("cuMemcpyDtoDAsync_v2"),
            cuMemcpyDtoD: sym!("cuMemcpyDtoD_v2"),
            cuMemsetD8: sym!("cuMemsetD8_v2"),
            cuMemsetD8Async: sym!("cuMemsetD8Async"),
            cuModuleLoadData: sym!("cuModuleLoadData"),
            cuModuleGetFunction: sym!("cuModuleGetFunction"),
            cuLaunchKernel: sym!("cuLaunchKernel"),
            cuStreamCreate: sym!("cuStreamCreate"),
            cuStreamDestroy: sym!("cuStreamDestroy_v2"),
            cuStreamSynchronize: sym!("cuStreamSynchronize"),
            cuEventCreate: sym!("cuEventCreate"),
            cuEventDestroy: sym!("cuEventDestroy_v2"),
            cuEventRecord: sym!("cuEventRecord"),
            cuEventSynchronize: sym!("cuEventSynchronize"),
            cuStreamWaitEvent: sym!("cuStreamWaitEvent"),
            cuMemGetInfo: sym!("cuMemGetInfo_v2"),
            cuGetErrorString: sym!("cuGetErrorString"),
            _lib: lib,
        })
    }
}

#[derive(Default)]
struct Runtime {
    device: c_int,
    last_error: c_int,
    /// Device ordinal -> primary context handle. Retained once per device.
    contexts: HashMap<c_int, usize>,
    /// hostFun pointer → device function handle address
    functions: HashMap<usize, usize>,
    /// fatbin handle → loaded module handle address
    modules: HashMap<usize, usize>,
    /// fatbin handle → original fatbin image pointer (deferred load)
    pending_fatbins: HashMap<usize, usize>,
    /// hostFun → (fatbin handle, device name bytes)
    pending_functions: HashMap<usize, (usize, Vec<u8>)>,
    next_fatbin: usize,
}

fn runtime() -> &'static Mutex<Runtime> {
    static RUNTIME: OnceLock<Mutex<Runtime>> = OnceLock::new();
    RUNTIME.get_or_init(|| Mutex::new(Runtime::default()))
}

fn set_error(code: c_int) -> c_int {
    if let Ok(mut rt) = runtime().lock() {
        rt.last_error = code;
    }
    code
}

fn ensure_context(drv: &Driver) -> c_int {
    let device = runtime().lock().map(|rt| rt.device).unwrap_or(0);
    // Retain each primary context once: every retain needs a matching release, and
    // this runs on nearly every Runtime call.
    let cached = runtime()
        .lock()
        .ok()
        .and_then(|rt| rt.contexts.get(&device).copied());
    let ctx = match cached {
        Some(ctx) => ctx as CUcontext,
        None => {
            let mut ctx = std::ptr::null_mut();
            let err = unsafe { (drv.cuDevicePrimaryCtxRetain)(&mut ctx, device) };
            if err != SUCCESS {
                return err;
            }
            if let Ok(mut rt) = runtime().lock() {
                if let Some(&existing) = rt.contexts.get(&device) {
                    // A racing thread retained first; keep one reference per device.
                    ctx = existing as CUcontext;
                } else {
                    rt.contexts.insert(device, ctx as usize);
                }
            }
            ctx
        }
    };
    unsafe { (drv.cuCtxSetCurrent)(ctx) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDriverGetVersion(version: *mut c_int) -> c_int {
    if version.is_null() {
        return set_error(INVALID_VALUE);
    }
    // Match the Driver ABI advertised by nvcuda.
    unsafe { version.write(13000) };
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaRuntimeGetVersion(version: *mut c_int) -> c_int {
    if version.is_null() {
        return set_error(INVALID_VALUE);
    }
    unsafe { version.write(12090) };
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGetDeviceCount(count: *mut c_int) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    if count.is_null() {
        return set_error(INVALID_VALUE);
    }
    let err = unsafe { (drv.cuDeviceGetCount)(count) };
    set_error(err)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGetDevice(device: *mut c_int) -> c_int {
    if device.is_null() {
        return set_error(INVALID_VALUE);
    }
    let id = runtime().lock().map(|rt| rt.device).unwrap_or(0);
    unsafe { device.write(id) };
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaSetDevice(device: c_int) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let mut count = 0;
    let err = unsafe { (drv.cuDeviceGetCount)(&mut count) };
    if err != SUCCESS {
        return set_error(err);
    }
    if device < 0 || device >= count {
        return set_error(INVALID_DEVICE);
    }
    if let Ok(mut rt) = runtime().lock() {
        rt.device = device;
    }
    set_error(ensure_context(drv))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaSetDeviceFlags(_flags: c_uint) -> c_int {
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceSynchronize() -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    set_error(unsafe { (drv.cuCtxSynchronize)() })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceGetAttribute(
    value: *mut c_int,
    attr: c_int,
    device: c_int,
) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    set_error(unsafe { (drv.cuDeviceGetAttribute)(value, attr, device) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceGetPCIBusId(
    pci_bus_id: *mut c_char,
    len: c_int,
    device: c_int,
) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    unsafe {
        match drv._lib.get::<unsafe extern "system" fn(*mut c_char, c_int, CUdevice) -> CuResult>(
            b"cuDeviceGetPCIBusId\0",
        ) {
            Ok(f) => set_error(f(pci_bus_id, len, device)),
            Err(_) => set_error(NOT_SUPPORTED),
        }
    }
}

/// CUDA 12 `cudaDeviceProp` storage. Layout matches MSVC x64 `driver_types.h`.
/// Named early fields cover the common bring-up path; later fields (notably
/// `sharedMemPerBlockOptin` at byte 696) are written through `bytes` so GGML
/// soft_max can select the shared-memory kernels instead of the Lb0 fallback.
#[repr(C)]
pub struct CudaDeviceProp {
    pub name: [c_char; 256],
    pub uuid: [u8; 16],
    pub luid: [c_char; 8],
    pub luid_device_node_mask: c_uint,
    pub total_global_mem: usize,
    pub shared_mem_per_block: usize,
    pub regs_per_block: c_int,
    pub warp_size: c_int,
    pub mem_pitch: usize,
    pub max_threads_per_block: c_int,
    pub max_threads_dim: [c_int; 3],
    pub max_grid_size: [c_int; 3],
    pub clock_rate: c_int,
    pub total_const_mem: usize,
    pub major: c_int,
    pub minor: c_int,
    /// Bytes from offset 368 through the end of a CUDA 12 `cudaDeviceProp`
    /// (~offset 1024 with reserved[]). Zero-initialized then selectively filled.
    pub _pad: [u8; 656],
}

const PROP_TEXTURE_ALIGNMENT: usize = 368;
const PROP_MULTIPROCESSOR_COUNT: usize = 388;
const PROP_SHARED_MEM_PER_MULTIPROCESSOR: usize = 640;
const PROP_SHARED_MEM_PER_BLOCK_OPTIN: usize = 696;
const PROP_MAX_THREADS_PER_MULTIPROCESSOR: usize = 624;

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGetDeviceProperties_v2(
    prop: *mut CudaDeviceProp,
    device: c_int,
) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    if prop.is_null() {
        return set_error(INVALID_VALUE);
    }
    unsafe {
        prop.write(std::mem::zeroed());
        let p = &mut *prop;
        let err = (drv.cuDeviceGetName)(p.name.as_mut_ptr(), 256, device);
        if err != SUCCESS {
            return set_error(err);
        }
        let mut total = 0usize;
        let err = (drv.cuDeviceTotalMem)(&mut total, device);
        if err != SUCCESS {
            return set_error(err);
        }
        p.total_global_mem = total;
        let mut attr = |code: c_int, out: &mut c_int| {
            let mut v = 0;
            let e = (drv.cuDeviceGetAttribute)(&mut v, code, device);
            if e == SUCCESS {
                *out = v;
            }
            e
        };
        let _ = attr(1, &mut p.max_threads_per_block);
        let _ = attr(2, &mut p.max_threads_dim[0]);
        let _ = attr(3, &mut p.max_threads_dim[1]);
        let _ = attr(4, &mut p.max_threads_dim[2]);
        let _ = attr(5, &mut p.max_grid_size[0]);
        let _ = attr(6, &mut p.max_grid_size[1]);
        let _ = attr(7, &mut p.max_grid_size[2]);
        let mut shared = 0;
        if attr(8, &mut shared) == SUCCESS {
            p.shared_mem_per_block = shared as usize;
        }
        let _ = attr(10, &mut p.warp_size);
        let _ = attr(12, &mut p.regs_per_block);
        let _ = attr(13, &mut p.clock_rate);
        let mut constant = 0;
        if attr(9, &mut constant) == SUCCESS {
            p.total_const_mem = constant as usize;
        }
        let _ = attr(75, &mut p.major);
        let _ = attr(76, &mut p.minor);
        let mut pitch = 0;
        if attr(11, &mut pitch) == SUCCESS {
            p.mem_pitch = pitch as usize;
        }

        // Fill ABI offsets past `minor` that GGML / cudart consumers read.
        let base = prop as *mut u8;
        let write_usize = |offset: usize, value: usize| {
            base.add(offset).cast::<usize>().write_unaligned(value);
        };
        let write_i32 = |offset: usize, value: c_int| {
            base.add(offset).cast::<c_int>().write_unaligned(value);
        };
        write_usize(PROP_TEXTURE_ALIGNMENT, 512);
        write_usize(PROP_TEXTURE_ALIGNMENT + 8, 32); // texturePitchAlignment
        let mut mp = 0;
        if attr(16, &mut mp) == SUCCESS {
            write_i32(PROP_MULTIPROCESSOR_COUNT, mp);
        }
        let mut shared_mp = 0;
        if attr(81, &mut shared_mp) == SUCCESS {
            write_usize(PROP_SHARED_MEM_PER_MULTIPROCESSOR, shared_mp as usize);
        }
        let mut optin = 0;
        if attr(97, &mut optin) == SUCCESS {
            write_usize(PROP_SHARED_MEM_PER_BLOCK_OPTIN, optin as usize);
        } else {
            // Fall back to per-block shared so soft_max can still opt into SLM.
            write_usize(PROP_SHARED_MEM_PER_BLOCK_OPTIN, p.shared_mem_per_block);
        }
        let mut max_threads_mp = 0;
        if attr(39, &mut max_threads_mp) == SUCCESS {
            write_i32(PROP_MAX_THREADS_PER_MULTIPROCESSOR, max_threads_mp);
        }
        // Integrated / canMapHostMemory — common cudart bring-up checks.
        let mut integrated = 0;
        if attr(18, &mut integrated) == SUCCESS {
            write_i32(396, integrated);
        }
        write_i32(400, 1); // canMapHostMemory
    }
    set_error(SUCCESS)
}

/// Device allocations made through this shim: base address -> size. Everything else
/// (pageable or pinned host memory) is treated as host memory.
fn device_ranges() -> &'static Mutex<BTreeMap<usize, usize>> {
    static RANGES: OnceLock<Mutex<BTreeMap<usize, usize>>> = OnceLock::new();
    RANGES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn is_device_pointer(ptr: usize) -> bool {
    let Ok(ranges) = device_ranges().lock() else {
        return false;
    };
    ranges
        .range(..=ptr)
        .next_back()
        .is_some_and(|(&base, &size)| ptr - base < size)
}

/// Resolve cudaMemcpyDefault (4) into a concrete kind from where each side lives.
fn infer_kind(dst_device: bool, src_device: bool) -> c_int {
    match (dst_device, src_device) {
        (false, false) => 0,
        (true, false) => 1,
        (false, true) => 2,
        (true, true) => 3,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    if size == 0 {
        // cudaMalloc(0) succeeds and returns a null pointer.
        if !dev_ptr.is_null() {
            unsafe { dev_ptr.write(std::ptr::null_mut()) };
        }
        return set_error(SUCCESS);
    }
    let mut ptr = 0u64;
    let err = unsafe { (drv.cuMemAlloc)(&mut ptr, size) };
    if err == SUCCESS && !dev_ptr.is_null() {
        unsafe { dev_ptr.write(ptr as *mut c_void) };
        if let Ok(mut ranges) = device_ranges().lock() {
            ranges.insert(ptr as usize, size.max(1));
        }
    }
    set_error(err)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaFree(dev_ptr: *mut c_void) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    if dev_ptr.is_null() {
        return set_error(SUCCESS);
    }
    let err = unsafe { (drv.cuMemFree)(dev_ptr as u64) };
    if err == SUCCESS {
        if let Ok(mut ranges) = device_ranges().lock() {
            ranges.remove(&(dev_ptr as usize));
        }
    }
    set_error(err)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> c_int {
    unsafe { cudaHostAlloc(ptr, size, 0) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, flags: c_uint) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    if ptr.is_null() {
        return set_error(INVALID_VALUE);
    }
    if size == 0 {
        unsafe { ptr.write(std::ptr::null_mut()) };
        return set_error(SUCCESS);
    }
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    set_error(unsafe { (drv.cuMemHostAlloc)(ptr, size, flags) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaFreeHost(ptr: *mut c_void) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    if ptr.is_null() {
        return set_error(SUCCESS);
    }
    set_error(unsafe { (drv.cuMemFreeHost)(ptr) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMallocManaged(
    _ptr: *mut *mut c_void,
    _size: usize,
    _flags: c_uint,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemcpyAsync(
    dst: *mut c_void,
    src: *const c_void,
    count: usize,
    kind: c_int,
    stream: CUstream,
) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    // cudaMemcpyKind: HostToHost=0, HtoD=1, DtoH=2, DtoD=3, Default=4
    let kind = if kind == 4 {
        infer_kind(is_device_pointer(dst as usize), is_device_pointer(src as usize))
    } else {
        kind
    };
    let err = match kind {
        1 => unsafe { (drv.cuMemcpyHtoDAsync)(dst as u64, src, count, stream) },
        2 => unsafe { (drv.cuMemcpyDtoHAsync)(dst, src as u64, count, stream) },
        3 => unsafe { (drv.cuMemcpyDtoDAsync)(dst as u64, src as u64, count, stream) },
        0 => {
            if count == 0 || dst.is_null() || src.is_null() {
                SUCCESS
            } else {
                unsafe {
                    std::ptr::copy_nonoverlapping(src.cast::<u8>(), dst.cast::<u8>(), count);
                }
                SUCCESS
            }
        }
        _ => {
            eprintln!("CUDA for Intel cudaMemcpyAsync: unsupported kind={kind}");
            NOT_SUPPORTED
        }
    };
    set_error(err)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemcpy2DAsync(
    dst: *mut c_void,
    dpitch: usize,
    src: *const c_void,
    spitch: usize,
    width: usize,
    height: usize,
    kind: c_int,
    stream: CUstream,
) -> c_int {
    unsafe { extra::copy_2d_rows(dst, dpitch, src, spitch, width, height, kind, stream) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemcpyPeerAsync(
    dst: *mut c_void,
    dst_device: c_int,
    src: *const c_void,
    src_device: c_int,
    count: usize,
    stream: CUstream,
) -> c_int {
    if !extra::peer_copy_supported(dst_device, src_device) {
        return set_error(NOT_SUPPORTED);
    }
    unsafe { cudaMemcpyAsync(dst, src, count, 3, stream) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemset(dev_ptr: *mut c_void, value: c_int, count: usize) -> c_int {
    if count == 0 {
        return set_error(SUCCESS);
    }
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    set_error(unsafe { (drv.cuMemsetD8)(dev_ptr as u64, value as u8, count) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemsetAsync(
    dev_ptr: *mut c_void,
    value: c_int,
    count: usize,
    stream: CUstream,
) -> c_int {
    if count == 0 {
        return set_error(SUCCESS);
    }
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    // Fill on the caller's stream so it stays ordered after earlier work there.
    set_error(unsafe { (drv.cuMemsetD8Async)(dev_ptr as u64, value as u8, count, stream) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    set_error(unsafe { (drv.cuMemGetInfo)(free, total) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamCreateWithFlags(stream: *mut CUstream, flags: c_uint) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    set_error(unsafe { (drv.cuStreamCreate)(stream, flags) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamDestroy(stream: CUstream) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    set_error(unsafe { (drv.cuStreamDestroy)(stream) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamSynchronize(stream: CUstream) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    set_error(unsafe { (drv.cuStreamSynchronize)(stream) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamWaitEvent(
    stream: CUstream,
    event: CUevent,
    flags: c_uint,
) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    set_error(unsafe { (drv.cuStreamWaitEvent)(stream, event, flags) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaEventCreateWithFlags(event: *mut CUevent, flags: c_uint) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    set_error(unsafe { (drv.cuEventCreate)(event, flags) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaEventDestroy(event: CUevent) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    set_error(unsafe { (drv.cuEventDestroy)(event) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaEventRecord(event: CUevent, stream: CUstream) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    set_error(unsafe { (drv.cuEventRecord)(event, stream) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaEventSynchronize(event: CUevent) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    set_error(unsafe { (drv.cuEventSynchronize)(event) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGetLastError() -> c_int {
    let mut code = SUCCESS;
    if let Ok(mut rt) = runtime().lock() {
        code = rt.last_error;
        rt.last_error = SUCCESS;
    }
    code
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaPeekAtLastError() -> c_int {
    runtime()
        .lock()
        .map(|rt| rt.last_error)
        .unwrap_or(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGetErrorString(error: c_int) -> *const c_char {
    extra::error_text(error)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceCanAccessPeer(
    can: *mut c_int,
    _device: c_int,
    _peer: c_int,
) -> c_int {
    if !can.is_null() {
        unsafe { can.write(0) };
    }
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceEnablePeerAccess(_peer: c_int, _flags: c_uint) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceDisablePeerAccess(_peer: c_int) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaDeviceReset() -> c_int {
    // Best-effort: synchronize then clear the last error. Full context teardown
    // is not required for stock ggml-cuda LoadLibrary / init probes.
    let _ = unsafe { cudaDeviceSynchronize() };
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemcpy3DPeerAsync(
    _p: *const c_void,
    _stream: CUstream,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamBeginCapture(
    _stream: CUstream,
    _mode: c_uint,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamEndCapture(
    _stream: CUstream,
    _p_graph: *mut *mut c_void,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaStreamIsCapturing(
    _stream: CUstream,
    status: *mut c_uint,
) -> c_int {
    // cudaStreamCaptureStatusNone = 0
    if !status.is_null() {
        unsafe { status.write(0) };
    }
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGraphDestroy(_graph: *mut c_void) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGraphExecDestroy(_graph_exec: *mut c_void) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGraphExecUpdate(
    _graph_exec: *mut c_void,
    _graph: *mut c_void,
    _error_node: *mut *mut c_void,
    _update_result: *mut c_uint,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGraphInstantiate(
    _p_graph_exec: *mut *mut c_void,
    _graph: *mut c_void,
    _flags: c_ulonglong,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaGraphLaunch(
    _graph_exec: *mut c_void,
    _stream: CUstream,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaHostGetDevicePointer(
    _pdevice: *mut *mut c_void,
    _phost: *mut c_void,
    _flags: c_uint,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaHostRegister(
    _ptr: *mut c_void,
    _size: usize,
    _flags: c_uint,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaHostUnregister(_ptr: *mut c_void) -> c_int {
    set_error(NOT_SUPPORTED)
}

/// Layout matches CUDA 12 `cudaFuncAttributes` in driver_types.h (leading fields
/// used by ggml; trailing reserved[] zeroed).
#[repr(C)]
struct CudaFuncAttributes {
    shared_size_bytes: usize,
    const_size_bytes: usize,
    local_size_bytes: usize,
    max_threads_per_block: c_int,
    num_regs: c_int,
    ptx_version: c_int,
    binary_version: c_int,
    cache_mode_ca: c_int,
    max_dynamic_shared_size_bytes: c_int,
    preferred_shmem_carveout: c_int,
    cluster_dim_must_be_set: c_int,
    required_cluster_width: c_int,
    required_cluster_height: c_int,
    required_cluster_depth: c_int,
    cluster_scheduling_policy_preference: c_int,
    non_portable_cluster_size_allowed: c_int,
    reserved: [c_int; 16],
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaFuncGetAttributes(
    attr: *mut c_void,
    func: *const c_void,
) -> c_int {
    if attr.is_null() || func.is_null() {
        return set_error(INVALID_VALUE);
    }
    // Ensure the host stub is registered so later launches can resolve it.
    if let Ok(drv) = driver() {
        let _ = resolve_function(drv, func as usize);
    }
    // ggml uses ptxVersion >= 90 to enable Programmatic Dependent Launch.
    // Our fatbins are PTX 8.x / sm_80-virtual — report 80 so PDL stays off.
    let filled = CudaFuncAttributes {
        shared_size_bytes: 0,
        const_size_bytes: 0,
        local_size_bytes: 0,
        max_threads_per_block: 1024,
        num_regs: 32,
        ptx_version: 80,
        binary_version: 80,
        cache_mode_ca: 0,
        max_dynamic_shared_size_bytes: 48 * 1024,
        preferred_shmem_carveout: -1,
        cluster_dim_must_be_set: 0,
        required_cluster_width: 0,
        required_cluster_height: 0,
        required_cluster_depth: 0,
        cluster_scheduling_policy_preference: 0,
        non_portable_cluster_size_allowed: 0,
        reserved: [0; 16],
    };
    unsafe {
        std::ptr::write(attr.cast::<CudaFuncAttributes>(), filled);
    }
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaFuncSetAttribute(
    _func: *const c_void,
    _attr: c_int,
    _value: c_int,
) -> c_int {
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
    num_blocks: *mut c_int,
    func: *const c_void,
    block_size: c_int,
    dynamic_smem: usize,
    flags: c_uint,
) -> c_int {
    if num_blocks.is_null() || block_size <= 0 || flags > 1 {
        return set_error(INVALID_VALUE);
    }
    // Ask the driver; keep the historical answer of 1 when the kernel cannot be resolved.
    let blocks = driver()
        .ok()
        .filter(|drv| ensure_context(drv) == SUCCESS)
        .and_then(|drv| extra::driver_occupancy(drv, func, block_size, dynamic_smem, flags))
        .unwrap_or(1);
    unsafe { num_blocks.write(blocks) };
    set_error(SUCCESS)
}

/// CUDA `dim3` — 12 bytes; on Win64 x64 this is passed by hidden pointer.
#[repr(C)]
#[derive(Clone, Copy)]
struct Dim3 {
    x: c_uint,
    y: c_uint,
    z: c_uint,
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaLaunchKernel(
    func: *const c_void,
    grid: Dim3,
    block: Dim3,
    args: *mut *mut c_void,
    shared_mem: usize,
    stream: CUstream,
) -> c_int {
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    let key = func as usize;
    // Fast path: one lock acquisition when the kernel is already resolved.
    let cached = runtime().lock().ok().and_then(|rt| rt.functions.get(&key).copied());
    if cached.is_none()
        && let Err(err) = resolve_function(drv, key)
    {
        eprintln!(
            "CUDA for Intel cudaLaunchKernel: resolve failed err={err} host_fun={key:#x}"
        );
        return set_error(err);
    }
    let function = cached.or_else(|| match runtime().lock() {
        Ok(rt) => rt.functions.get(&key).copied(),
        Err(_) => None,
    });
    let Some(function) = function else {
        eprintln!("CUDA for Intel cudaLaunchKernel: no device fn for host={key:#x}");
        return set_error(INVALID_VALUE);
    };
    info_log!(
        "CUDA for Intel cudaLaunchKernel: grid=({},{},{}) block=({},{},{}) shmem={shared_mem} stream={stream:?}",
        grid.x, grid.y, grid.z, block.x, block.y, block.z
    );
    set_error(unsafe {
        (drv.cuLaunchKernel)(
            function as CUfunction,
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            shared_mem as c_uint,
            stream,
            args,
            std::ptr::null_mut(),
        )
    })
}

/// Keep header + a single `.entry` so sibling unsupported kernels in the same
/// nvcc fatbin do not poison `cuModuleLoadData`.
fn filter_ptx_entry(ptx: &[u8], entry_name: &[u8]) -> Option<Vec<u8>> {
    let text = std::str::from_utf8(ptx).ok()?;
    let header_end = text.find(".entry")?;
    let header = &text[..header_end];
    let needle = format!(".entry {}", String::from_utf8_lossy(entry_name));
    let start = text.find(&needle)?;
    let body = &text[start..];
    let brace = body.find('{')?;
    let mut depth = 0i32;
    let mut end = None;
    for (i, ch) in body[brace..].char_indices() {
        match ch {
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    end = Some(brace + i + 1);
                    break;
                }
            }
            _ => {}
        }
    }
    let end = end?;
    // Rebuild a minimal module header. nvcc inserts multi-line `.extern .func`
    // (e.g. vprintf) prototypes; line-wise skips leave orphaned `( .param ) ;`
    // tokens that poison ModuleLoad. Only the ISA triple is required.
    let mut out = String::new();
    out.push_str(&minimal_ptx_header(header));
    // Preserve a leading `.visible` if present immediately before `.entry`.
    let entry_prefix = text[..start].rsplit_once('\n').map(|(_, tail)| tail).unwrap_or("");
    if entry_prefix.trim() == ".visible" {
        out.push_str(".visible\n");
    }
    out.push_str(&body[..end]);
    out.push('\n');
    Some(out.into_bytes())
}

fn minimal_ptx_header(header: &str) -> String {
    let mut version = String::from("8.8");
    let mut target = String::from("sm_80");
    let mut address_size = String::from("64");
    for line in header.lines() {
        let t = line.trim_start();
        if let Some(rest) = t.strip_prefix(".version") {
            version = rest.trim().to_string();
        } else if let Some(rest) = t.strip_prefix(".target") {
            // Take the first sm_XX token; ignore trailing comma options.
            if let Some(sm) = rest
                .split(|c: char| c == ',' || c.is_whitespace())
                .find(|tok| tok.starts_with("sm_"))
            {
                target = sm.to_string();
            }
        } else if let Some(rest) = t.strip_prefix(".address_size") {
            address_size = rest.trim().to_string();
        }
    }
    format!(".version {version}\n.target {target}\n.address_size {address_size}\n\n")
}

fn resolve_function(drv: &Driver, host_fun: usize) -> Result<(), c_int> {
    let (fat_key, name) = {
        let rt = runtime().lock().map_err(|_| UNKNOWN)?;
        if rt.functions.contains_key(&host_fun) {
            return Ok(());
        }
        rt.pending_functions
            .get(&host_fun)
            .cloned()
            .ok_or(INVALID_VALUE)?
    };
    let image = {
        let rt = runtime().lock().map_err(|_| UNKNOWN)?;
        rt.pending_fatbins.get(&fat_key).copied()
    };
    let Some(image) = image else {
        return Err(INVALID_VALUE);
    };
    let Some(raw) = extract_ptx(image as *mut c_void) else {
        eprintln!(
            "CUDA for Intel resolve: extract_ptx failed fat={fat_key:#x} image={image:#x}"
        );
        return Err(INVALID_VALUE);
    };
    let Some(mut filtered) = filter_ptx_entry(&raw, &name) else {
        eprintln!(
            "CUDA for Intel resolve: entry filter failed name={}",
            String::from_utf8_lossy(&name)
        );
        return Err(INVALID_VALUE);
    };
    info_log!(
        "CUDA for Intel resolve: entry {} filtered {} -> {} bytes",
        String::from_utf8_lossy(&name),
        raw.len(),
        filtered.len()
    );
    if std::env::var_os("CUDA_FOR_INTEL_DUMP_PTX").is_some() {
        let dump_name = format!(
            "cfi_dump_{}.ptx",
            String::from_utf8_lossy(&name)
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
                .take(80)
                .collect::<String>()
        );
        let _ = std::fs::write(&dump_name, &filtered);
        eprintln!("CUDA for Intel resolve: dumped filtered PTX to {dump_name}");
    }
    let err = ensure_context(drv);
    if err != SUCCESS {
        eprintln!("CUDA for Intel resolve: ensure_context -> {err}");
        return Err(err);
    }
    let mut module = std::ptr::null_mut();
    filtered.push(0);
    let err = unsafe { (drv.cuModuleLoadData)(&mut module, filtered.as_ptr().cast()) };
    eprintln!("CUDA for Intel resolve: cuModuleLoadData -> {err}");
    if err != SUCCESS {
        return Err(err);
    }
    let mut function = std::ptr::null_mut();
    let c_name = std::ffi::CString::new(name.clone()).map_err(|_| INVALID_VALUE)?;
    let err = unsafe { (drv.cuModuleGetFunction)(&mut function, module, c_name.as_ptr()) };
    if err != SUCCESS {
        eprintln!(
            "CUDA for Intel resolve: cuModuleGetFunction({}) -> {err}",
            String::from_utf8_lossy(&name)
        );
        return Err(err);
    }
    if let Ok(mut rt) = runtime().lock() {
        // Keep every per-entry module alive; do not drop pending fatbin images
        // so sibling kernels from the same registration can load later.
        rt.modules.insert(host_fun, module as usize);
        rt.functions.insert(host_fun, function as usize);
        rt.pending_functions.remove(&host_fun);
    }
    Ok(())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaLaunchKernelExC(
    _config: *const c_void,
    _func: *const c_void,
    _args: *mut *mut c_void,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaLaunchCooperativeKernel(
    func: *const c_void,
    grid: Dim3,
    block: Dim3,
    args: *mut *mut c_void,
    shared: usize,
    stream: CUstream,
) -> c_int {
    // Grid-wide synchronization needs every block resident at once: refuse grids that
    // cannot be (cudaErrorCooperativeLaunchTooLarge), otherwise launch normally.
    let threads = block.x.saturating_mul(block.y).saturating_mul(block.z);
    if threads == 0 || threads > c_int::MAX as u32 {
        return set_error(INVALID_VALUE);
    }
    if let Some(capacity) = unsafe { extra::cooperative_limit(func, threads as c_int, shared) }
        && !extra::cooperative_fits([grid.x, grid.y, grid.z], capacity)
    {
        return set_error(720);
    }
    unsafe { cudaLaunchKernel(func, grid, block, args, shared, stream) }
}

thread_local! {
    static LAUNCH_CONFIG: std::cell::Cell<Option<(Dim3, Dim3, usize, CUstream)>> =
        const { std::cell::Cell::new(None) };
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaPushCallConfiguration(
    grid: Dim3,
    block: Dim3,
    shared: usize,
    stream: CUstream,
) -> c_int {
    LAUNCH_CONFIG.with(|slot| slot.set(Some((grid, block, shared, stream))));
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaPopCallConfiguration(
    grid: *mut Dim3,
    block: *mut Dim3,
    shared: *mut usize,
    stream: *mut CUstream,
) -> c_int {
    let Some((g, b, sh, st)) = LAUNCH_CONFIG.with(|slot| slot.take()) else {
        return INVALID_VALUE;
    };
    if !grid.is_null() {
        unsafe { grid.write(g) };
    }
    if !block.is_null() {
        unsafe { block.write(b) };
    }
    if !shared.is_null() {
        unsafe { shared.write(sh) };
    }
    if !stream.is_null() {
        unsafe { stream.write(st) };
    }
    SUCCESS
}

/// Fatbin registration — extract PTX when present and load via Driver API.
/// Opaque fatbin handle — must be a real heap pointer; callers dereference it.
struct FatbinHandle {
    id: usize,
    image: usize,
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaRegisterFatBinary(fatbin: *mut c_void) -> *mut c_void {
    // CUDA Runtime ABI: return a void** slot whose pointee is our FatbinHandle.
    // Returning the handle pointer directly made callers read FatbinHandle.id
    // (1,2,3…) as a nested pointer.
    let id = {
        let mut rt = runtime().lock().unwrap();
        rt.next_fatbin = rt.next_fatbin.saturating_add(1);
        rt.next_fatbin
    };
    let handle = Box::into_raw(Box::new(FatbinHandle {
        id,
        image: fatbin as usize,
    }));
    let slot = Box::into_raw(Box::new(handle));
    if let Ok(mut rt) = runtime().lock() {
        rt.pending_fatbins.insert(handle as usize, fatbin as usize);
    }
    slot.cast()
}

fn fatbin_handle_from_slot(fatbin: *mut c_void) -> Option<*mut FatbinHandle> {
    if fatbin.is_null() {
        return None;
    }
    let handle = unsafe { *(fatbin as *mut *mut FatbinHandle) };
    if handle.is_null() {
        None
    } else {
        Some(handle)
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaRegisterFatBinaryEnd(fatbin: *mut c_void) {
    // Do not ModuleLoadData here: CUDA invokes this during CRT/static init for every
    // fatbin (147 in ggml-cuda). Eager PTX→SPIR-V under the loader is too heavy and
    // unsafe. Keep the image in pending_fatbins; resolve_function loads on first launch.
    let Some(handle_ptr) = fatbin_handle_from_slot(fatbin) else {
        eprintln!("CUDA for Intel fatbin end: bad slot={fatbin:?}");
        return;
    };
    let id = unsafe { (*handle_ptr).id };
    let image = unsafe { (*handle_ptr).image };
    // One-shot extract smoke check on the first fatbin only (diagnostics).
    if id == 1 {
        match extract_ptx(image as *mut c_void) {
            Some(ptx) => eprintln!(
                "CUDA for Intel fatbin end: id=1 extract ok ({} PTX bytes); defer ModuleLoad",
                ptx.len()
            ),
            None => eprintln!(
                "CUDA for Intel fatbin end: id=1 extract_ptx failed image={image:#x}"
            ),
        }
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaUnregisterFatBinary(fatbin: *mut c_void) {
    let Some(handle_ptr) = fatbin_handle_from_slot(fatbin) else {
        return;
    };
    let key = handle_ptr as usize;
    if let Ok(mut rt) = runtime().lock() {
        rt.modules.remove(&key);
        rt.pending_fatbins.remove(&key);
    }
    unsafe { drop(Box::from_raw(handle_ptr)) };
    unsafe { *(fatbin as *mut *mut FatbinHandle) = std::ptr::null_mut() };
    unsafe { drop(Box::from_raw(fatbin as *mut *mut FatbinHandle)) };
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaRegisterFunction(
    fatbin: *mut *mut c_void,
    host_fun: *const c_void,
    _device_fun: *mut c_char,
    device_name: *const c_char,
    _thread_limit: c_int,
    _tid: *mut c_void,
    _bid: *mut c_void,
    _bdim: *mut c_void,
    _gdim: *mut c_void,
    _wsize: *mut c_int,
) {
    // `fatbin` is a void** holding our FatbinHandle*.
    let handle_ptr = unsafe { (*fatbin) as *mut FatbinHandle };
    let fat_key = if handle_ptr.is_null() {
        0
    } else {
        handle_ptr as usize
    };
    let name = if device_name.is_null() {
        Vec::new()
    } else {
        unsafe { CStr::from_ptr(device_name) }.to_bytes().to_vec()
    };
    if let Ok(mut rt) = runtime().lock() {
        rt.pending_functions
            .insert(host_fun as usize, (fat_key, name));
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaRegisterVar(
    _fatbin: *mut *mut c_void,
    _host_var: *mut c_void,
    _device_addr: *mut c_char,
    _device_name: *const c_char,
    _ext: c_int,
    _size: usize,
    _constant: c_int,
    _global: c_int,
) {
}

fn extract_ptx(fatbin: *mut c_void) -> Option<Vec<u8>> {
    unsafe { cfi_fatbin::extract_ptx(fatbin.cast()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_kind_follows_pointer_ownership() {
        device_ranges().lock().unwrap().insert(0x1000, 0x100);
        assert!(is_device_pointer(0x1000));
        assert!(is_device_pointer(0x10ff));
        assert!(!is_device_pointer(0x1100));
        assert!(!is_device_pointer(0xfff));
        assert_eq!(infer_kind(true, false), 1);
        assert_eq!(infer_kind(false, true), 2);
        assert_eq!(infer_kind(true, true), 3);
        assert_eq!(infer_kind(false, false), 0);
    }
}
