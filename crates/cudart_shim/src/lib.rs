//! Minimal CUDA Runtime shim for ggml-cuda.dll on cuda-for-intel.
//!
//! Stock NVIDIA cudart64_12 probes the WDDM LUID then expects an NVIDIA kernel
//! driver (cudaErrorInsufficientDriver). This DLL implements the Runtime entry
//! points ggml imports by forwarding to our Driver API `nvcuda.dll`.
#![allow(non_snake_case)]

use libloading::Library;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_uint, c_void, CStr};
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
    let mut ctx = std::ptr::null_mut();
    let err = unsafe { (drv.cuDevicePrimaryCtxRetain)(&mut ctx, device) };
    if err != SUCCESS {
        return err;
    }
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
    // Remaining fields zeroed — ggml reads the early ones during init.
    pub _pad: [u8; 512],
}

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
    }
    set_error(SUCCESS)
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
    let mut ptr = 0u64;
    let err = unsafe { (drv.cuMemAlloc)(&mut ptr, size) };
    if err == SUCCESS && !dev_ptr.is_null() {
        unsafe { dev_ptr.write(ptr as *mut c_void) };
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
    set_error(unsafe { (drv.cuMemFree)(dev_ptr as u64) })
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
    // kind: 1 HtoD, 2 DtoH, 3 DtoD (cudaMemcpyKind)
    let err = match kind {
        1 => unsafe { (drv.cuMemcpyHtoDAsync)(dst as u64, src, count, stream) },
        2 => unsafe { (drv.cuMemcpyDtoHAsync)(dst, src as u64, count, stream) },
        _ => NOT_SUPPORTED,
    };
    set_error(err)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemcpy2DAsync(
    _dst: *mut c_void,
    _dpitch: usize,
    _src: *const c_void,
    _spitch: usize,
    _width: usize,
    _height: usize,
    _kind: c_int,
    _stream: CUstream,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemcpyPeerAsync(
    _dst: *mut c_void,
    _dst_device: c_int,
    _src: *const c_void,
    _src_device: c_int,
    _count: usize,
    _stream: CUstream,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemset(dev_ptr: *mut c_void, value: c_int, count: usize) -> c_int {
    // Host-side fill via DtoH/HtoD is expensive; allocate staging.
    if count == 0 {
        return set_error(SUCCESS);
    }
    let mut host = vec![value as u8; count];
    let Ok(drv) = driver() else {
        return set_error(NOT_INITIALIZED);
    };
    let err = ensure_context(drv);
    if err != SUCCESS {
        return set_error(err);
    }
    set_error(unsafe { (drv.cuMemcpyHtoD)(dev_ptr as u64, host.as_ptr().cast(), count) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaMemsetAsync(
    dev_ptr: *mut c_void,
    value: c_int,
    count: usize,
    stream: CUstream,
) -> c_int {
    let _ = stream;
    unsafe { cudaMemset(dev_ptr, value, count) }
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
    match error {
        0 => c"no error".as_ptr(),
        1 => c"invalid argument".as_ptr(),
        3 => c"initialization error".as_ptr(),
        35 => c"CUDA driver version is insufficient for CUDA runtime version".as_ptr(),
        101 => c"invalid device ordinal".as_ptr(),
        801 => c"operation not supported".as_ptr(),
        _ => c"unknown error".as_ptr(),
    }
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

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaFuncGetAttributes(
    _attr: *mut c_void,
    _func: *const c_void,
) -> c_int {
    set_error(NOT_SUPPORTED)
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
    _func: *const c_void,
    _block_size: c_int,
    _dynamic_smem: usize,
    _flags: c_uint,
) -> c_int {
    if !num_blocks.is_null() {
        unsafe { num_blocks.write(1) };
    }
    set_error(SUCCESS)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cudaLaunchKernel(
    func: *const c_void,
    grid: [c_uint; 3],
    block: [c_uint; 3],
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
    if let Err(err) = resolve_function(drv, key) {
        return set_error(err);
    }
    let function = match runtime().lock() {
        Ok(rt) => rt.functions.get(&key).copied(),
        Err(_) => None,
    };
    let Some(function) = function else {
        return set_error(INVALID_VALUE);
    };
    set_error(unsafe {
        (drv.cuLaunchKernel)(
            function as CUfunction,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            shared_mem as c_uint,
            stream,
            args,
            std::ptr::null_mut(),
        )
    })
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
    if let Some(image) = image {
        if let Some(ptx) = extract_ptx(image as *mut c_void) {
            let err = ensure_context(drv);
            if err != SUCCESS {
                return Err(err);
            }
            let mut module = std::ptr::null_mut();
            let mut bytes = ptx;
            bytes.push(0);
            let err = unsafe { (drv.cuModuleLoadData)(&mut module, bytes.as_ptr().cast()) };
            if err == SUCCESS {
                if let Ok(mut rt) = runtime().lock() {
                    rt.modules.insert(fat_key, module as usize);
                    rt.pending_fatbins.remove(&fat_key);
                }
            } else {
                return Err(err);
            }
        }
    }
    let module = runtime()
        .lock()
        .map_err(|_| UNKNOWN)?
        .modules
        .get(&fat_key)
        .copied()
        .ok_or(INVALID_VALUE)?;
    let mut function = std::ptr::null_mut();
    let c_name = std::ffi::CString::new(name).map_err(|_| INVALID_VALUE)?;
    let err = unsafe { (drv.cuModuleGetFunction)(&mut function, module as CUmodule, c_name.as_ptr()) };
    if err != SUCCESS {
        return Err(err);
    }
    if let Ok(mut rt) = runtime().lock() {
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
    _func: *const c_void,
    _grid: [c_uint; 3],
    _block: [c_uint; 3],
    _args: *mut *mut c_void,
    _shared: usize,
    _stream: CUstream,
) -> c_int {
    set_error(NOT_SUPPORTED)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaPushCallConfiguration(
    _grid: [c_uint; 3],
    _block: [c_uint; 3],
    _shared: usize,
    _stream: CUstream,
) -> c_int {
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaPopCallConfiguration(
    _grid: *mut [c_uint; 3],
    _block: *mut [c_uint; 3],
    _shared: *mut usize,
    _stream: *mut CUstream,
) -> c_int {
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
    // Defer module load until first launch: loading during CRT/DllMain init of
    // ggml-cuda.dll can turn translator failures into ERROR_DLL_INIT_FAILED.
    let id = {
        let mut rt = runtime().lock().unwrap();
        rt.next_fatbin = rt.next_fatbin.saturating_add(1);
        rt.next_fatbin
    };
    let handle = Box::new(FatbinHandle {
        id,
        image: fatbin as usize,
    });
    let ptr = Box::into_raw(handle);
    if let Ok(mut rt) = runtime().lock() {
        rt.pending_fatbins.insert(ptr as usize, fatbin as usize);
    }
    ptr.cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaRegisterFatBinaryEnd(_fatbin: *mut c_void) {}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn __cudaUnregisterFatBinary(fatbin: *mut c_void) {
    // Called during process teardown; be liberal about already-freed state.
    if fatbin.is_null() {
        return;
    }
    let handle_ptr = unsafe { *(fatbin as *mut *mut FatbinHandle) };
    if handle_ptr.is_null() {
        return;
    }
    let key = handle_ptr as usize;
    if let Ok(mut rt) = runtime().lock() {
        rt.modules.remove(&key);
        rt.pending_fatbins.remove(&key);
    }
    unsafe { drop(Box::from_raw(handle_ptr)) };
    unsafe { *(fatbin as *mut *mut FatbinHandle) = std::ptr::null_mut() };
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

/// Best-effort PTX extraction from a CUDA fatbin / fatbinc wrapper.
///
/// Must never scan unbounded memory: the registration path runs during
/// `DllMain`/`CRT` init of ggml-cuda.dll, and a fault becomes ERROR_DLL_INIT_FAILED.
fn extract_ptx(fatbin: *mut c_void) -> Option<Vec<u8>> {
    if fatbin.is_null() {
        return None;
    }
    unsafe {
        let wrapper = fatbin.cast::<u8>();
        // fatbinc wrapper: magic 0x466243B1 at offset 0; on x64 `data*` is at +8.
        let magic = read_u32(wrapper)?;
        let data_ptr = if magic == 0x4662_43B1 {
            let addr = read_usize(wrapper.add(8))?;
            if addr == 0 {
                return None;
            }
            addr as *const u8
        } else {
            wrapper
        };
        let header_magic = read_u32(data_ptr)?;
        if header_magic != 0xBA55_ED50 {
            // Only accept clearly textual PTX with a bounded probe.
            let probe = std::slice::from_raw_parts(data_ptr, 16);
            if probe.starts_with(b".version") || probe.starts_with(b"//") {
                let mut out = Vec::new();
                for i in 0..16 * 1024 * 1024 {
                    let b = data_ptr.add(i).read();
                    if b == 0 {
                        break;
                    }
                    out.push(b);
                }
                return Some(out);
            }
            return None;
        }
        // FatbinHeader: magic u32, version u16, header_size u16, size u64
        let header_size = read_u16(data_ptr.add(6))? as usize;
        let total_size = read_u64(data_ptr.add(8))? as usize;
        if header_size < 16 || total_size <= header_size || total_size > 64 * 1024 * 1024 {
            return None;
        }
        let mut offset = header_size;
        while offset + 16 <= total_size {
            let entry = data_ptr.add(offset);
            let kind = entry.read();
            let entry_size = read_u64(entry.add(8))? as usize;
            if entry_size < 16 || offset + entry_size > total_size {
                break;
            }
            let payload_len = entry_size - 16;
            let payload = std::slice::from_raw_parts(entry.add(16), payload_len);
            if kind == 1 || payload.starts_with(b".version") {
                let end = payload
                    .iter()
                    .position(|&b| b == 0)
                    .unwrap_or(payload.len());
                return Some(payload[..end].to_vec());
            }
            offset += entry_size;
        }
        None
    }
}

unsafe fn read_u16(ptr: *const u8) -> Option<u16> {
    Some(u16::from_le_bytes([
        unsafe { ptr.add(0).read() },
        unsafe { ptr.add(1).read() },
    ]))
}

unsafe fn read_u32(ptr: *const u8) -> Option<u32> {
    Some(u32::from_le_bytes([
        unsafe { ptr.add(0).read() },
        unsafe { ptr.add(1).read() },
        unsafe { ptr.add(2).read() },
        unsafe { ptr.add(3).read() },
    ]))
}

unsafe fn read_u64(ptr: *const u8) -> Option<u64> {
    let mut bytes = [0u8; 8];
    for (i, slot) in bytes.iter_mut().enumerate() {
        *slot = unsafe { ptr.add(i).read() };
    }
    Some(u64::from_le_bytes(bytes))
}

unsafe fn read_usize(ptr: *const u8) -> Option<usize> {
    let mut bytes = [0u8; size_of::<usize>()];
    for (i, slot) in bytes.iter_mut().enumerate() {
        *slot = unsafe { ptr.add(i).read() };
    }
    Some(usize::from_le_bytes(bytes))
}
