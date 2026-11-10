//! cuSPARSE shim over cuda-for-intel's Driver API (`nvcuda.dll`).
//!
//! Same structure as the cuBLAS and cuFFT shims: the work runs as generated PTX that our
//! Driver API translates to SPIR-V (no oneAPI dependency):
//! * `kernels::plan`   - pure logic: enums, descriptors, and the launch lists for SpMV,
//!   SpMM, SDDMM and block-CSR products.
//! * `kernels::ptxgen` - the kernels (scalar and vector CSR gather, atomic scatter for
//!   transposed and COO operands, scale, SDDMM, BSR, and the COO/CSR index helpers).
//! * this file - the exported C ABI, handles and descriptors, kernel cache and launcher.
//!
//! Implemented: handles (`cusparseCreate`/`Destroy`, stream and pointer mode), the legacy
//! matrix descriptor, the generic API descriptors (CSR, CSC, COO, dense vector and matrix,
//! const variants, getters, setters, strided batches, attributes), `cusparseSpMV`,
//! `cusparseSpMM`, `cusparseSDDMM` (+ bufferSize and preprocess), `cusparse{S,D}bsrmv`,
//! `cusparse{S,D}bsrmm`, `cusparseXcoo2csr`, `cusparseXcsr2coo`,
//! `cusparseCreateIdentityPermutation`, and the version and error-string queries.
//! Values must be f32 or f64 (`CUDA_R_32F`/`CUDA_R_64F`), with the compute type equal to the
//! value type; anything else returns `CUSPARSE_STATUS_NOT_SUPPORTED`, as does every entry
//! point in `stubs.rs` (SpGEMM reuse, SpSV/SpSM, complex csrgeam2, sorts, BSR solves,
//! complex types). `cusparseSpGEMM` (CSR x CSR, op N, beta 0, f32/f64, 32- or 64-bit indices)
//! and `cusparse{S,D}csrgeam2` (32-bit CSR with sorted rows) run on the product kernels.
//!
//! Execution is asynchronous on the handle's stream, like cuSPARSE, except where cuSPARSE
//! itself returns host-side results (SpGEMM's C size, csrgeam2's nnz), which synchronize.
//! Only SpGEMM and csrgeam2 use external buffers; every other bufferSize query returns 0. `CFI_CUSPARSE_TRACE=1` logs each operation;
//! `CFI_CUSPARSE_KERNEL=scalar|vector` forces the CSR kernel (default: vector when rows hold
//! 32 or more nonzeros on average).
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

pub mod kernels;
mod stubs;

use kernels::plan::{self, Choice, DnMat, Launch, Scalars, SpMat};
use kernels::ptxgen::{self, Kernel, Variant};
use libloading::Library;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void};
use std::sync::{Mutex, OnceLock};

pub use plan::{INVALID_VALUE as CUSPARSE_STATUS_INVALID_VALUE, NOT_SUPPORTED as CUSPARSE_STATUS_NOT_SUPPORTED, SUCCESS as CUSPARSE_STATUS_SUCCESS};
use plan::{EXECUTION_FAILED, INTERNAL_ERROR, INVALID_VALUE, MATRIX_TYPE_NOT_SUPPORTED, NOT_INITIALIZED, NOT_SUPPORTED, SUCCESS};

/// Reported cuSPARSE version (12.5.9 -> 12509).
pub const VERSION: i32 = 12509;

pub fn status_name(status: i32) -> &'static std::ffi::CStr {
    match status {
        0 => c"CUSPARSE_STATUS_SUCCESS",
        1 => c"CUSPARSE_STATUS_NOT_INITIALIZED",
        2 => c"CUSPARSE_STATUS_ALLOC_FAILED",
        3 => c"CUSPARSE_STATUS_INVALID_VALUE",
        4 => c"CUSPARSE_STATUS_ARCH_MISMATCH",
        5 => c"CUSPARSE_STATUS_MAPPING_ERROR",
        6 => c"CUSPARSE_STATUS_EXECUTION_FAILED",
        7 => c"CUSPARSE_STATUS_INTERNAL_ERROR",
        8 => c"CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED",
        9 => c"CUSPARSE_STATUS_ZERO_PIVOT",
        10 => c"CUSPARSE_STATUS_NOT_SUPPORTED",
        11 => c"CUSPARSE_STATUS_INSUFFICIENT_RESOURCES",
        _ => c"unrecognized status code",
    }
}

pub fn status_message(status: i32) -> &'static std::ffi::CStr {
    match status {
        0 => c"success",
        1 => c"the library was not initialized",
        2 => c"resource allocation failed",
        3 => c"an invalid value was passed",
        4 => c"the device architecture is not supported",
        5 => c"mapping error",
        6 => c"the kernel failed to execute",
        7 => c"internal error",
        8 => c"the matrix type is not supported",
        9 => c"zero pivot",
        10 => c"the operation is not supported",
        11 => c"insufficient resources",
        _ => c"unrecognized status code",
    }
}

fn trace_enabled() -> bool {
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| std::env::var("CFI_CUSPARSE_TRACE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

macro_rules! trace {
    ($($arg:tt)*) => {
        if trace_enabled() {
            eprintln!("CUDA for Intel cusparse: {}", format!($($arg)*));
        }
    };
}

fn choice() -> Choice {
    static C: OnceLock<Choice> = OnceLock::new();
    *C.get_or_init(|| match std::env::var("CFI_CUSPARSE_KERNEL").as_deref() {
        Ok("scalar") => Choice::Scalar,
        Ok("vector") => Choice::Vector,
        _ => Choice::Auto,
    })
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

type CuResult = c_int;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUcontext = *mut c_void;

// Load the driver with the DLL on Windows, as the cuBLAS shim does.
#[cfg(windows)]
#[link(name = "nvcuda", kind = "raw-dylib")]
unsafe extern "system" {
    fn cuCtxGetCurrent(context: *mut CUcontext) -> CuResult;
}

struct Driver {
    _lib: Library,
    cuCtxGetCurrent: unsafe extern "system" fn(*mut CUcontext) -> CuResult,
    cuInit: unsafe extern "system" fn(u32) -> CuResult,
    cuDeviceGet: unsafe extern "system" fn(*mut c_int, c_int) -> CuResult,
    cuDevicePrimaryCtxRetain: unsafe extern "system" fn(*mut CUcontext, c_int) -> CuResult,
    cuCtxSetCurrent: unsafe extern "system" fn(CUcontext) -> CuResult,
    cuModuleLoadData: unsafe extern "system" fn(*mut CUmodule, *const c_void) -> CuResult,
    cuModuleGetFunction: unsafe extern "system" fn(*mut CUfunction, CUmodule, *const c_char) -> CuResult,
    cuLaunchKernel: unsafe extern "system" fn(CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream, *mut *mut c_void, *mut *mut c_void) -> CuResult,
    cuMemcpyDtoH_v2: unsafe extern "system" fn(*mut c_void, u64, usize) -> CuResult,
    cuMemcpyHtoD_v2: unsafe extern "system" fn(u64, *const c_void, usize) -> CuResult,
    cuStreamSynchronize: unsafe extern "system" fn(CUstream) -> CuResult,
}

// Function pointers into a library that is never unloaded.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

fn driver() -> Result<&'static Driver, i32> {
    static DRIVER: OnceLock<Result<Driver, i32>> = OnceLock::new();
    DRIVER.get_or_init(load_driver).as_ref().map_err(|e| *e)
}

fn load_driver() -> Result<Driver, i32> {
    unsafe {
        #[cfg(windows)]
        let lib = Library::new("nvcuda.dll").map_err(|_| NOT_INITIALIZED)?;
        #[cfg(not(windows))]
        let lib = Library::new("libcuda.so.1").map_err(|_| NOT_INITIALIZED)?;
        macro_rules! sym {
            ($name:literal) => {
                *lib.get(concat!($name, "\0").as_bytes()).map_err(|_| NOT_INITIALIZED)?
            };
        }
        Ok(Driver {
            #[cfg(windows)]
            cuCtxGetCurrent,
            #[cfg(not(windows))]
            cuCtxGetCurrent: sym!("cuCtxGetCurrent"),
            cuInit: sym!("cuInit"),
            cuDeviceGet: sym!("cuDeviceGet"),
            cuDevicePrimaryCtxRetain: sym!("cuDevicePrimaryCtxRetain"),
            cuCtxSetCurrent: sym!("cuCtxSetCurrent"),
            cuModuleLoadData: sym!("cuModuleLoadData"),
            cuModuleGetFunction: sym!("cuModuleGetFunction"),
            cuLaunchKernel: sym!("cuLaunchKernel"),
            cuMemcpyDtoH_v2: sym!("cuMemcpyDtoH_v2"),
            cuMemcpyHtoD_v2: sym!("cuMemcpyHtoD_v2"),
            cuStreamSynchronize: sym!("cuStreamSynchronize"),
            _lib: lib,
        })
    }
}

/// The current context, or device 0's primary context made current (cuSPARSE, being a
/// runtime-API library, works on the primary context when none is current).
fn context(drv: &Driver) -> Result<usize, i32> {
    let mut ctx: CUcontext = std::ptr::null_mut();
    if unsafe { (drv.cuCtxGetCurrent)(&mut ctx) } == 0 && !ctx.is_null() {
        return Ok(ctx as usize);
    }
    let mut dev = 0;
    unsafe {
        if (drv.cuInit)(0) != 0 || (drv.cuDeviceGet)(&mut dev, 0) != 0 {
            return Err(NOT_INITIALIZED);
        }
        if (drv.cuDevicePrimaryCtxRetain)(&mut ctx, dev) != 0 || (drv.cuCtxSetCurrent)(ctx) != 0 {
            return Err(NOT_INITIALIZED);
        }
    }
    Ok(ctx as usize)
}

/// Loads `src` and resolves `names`; f64 modules report NOT_SUPPORTED when the device
/// cannot build them.
fn load_module(drv: &Driver, src: &str, names: &[String], double: bool) -> Result<Vec<usize>, i32> {
    let src = format!("{src}\0");
    let mut module: CUmodule = std::ptr::null_mut();
    let err = unsafe { (drv.cuModuleLoadData)(&mut module, src.as_ptr().cast()) };
    if err != 0 {
        eprintln!("CUDA for Intel cusparse: loading the sparse kernels failed (err={err})");
        return Err(if double { NOT_SUPPORTED } else { INTERNAL_ERROR });
    }
    names
        .iter()
        .map(|name| {
            let c = std::ffi::CString::new(name.as_str()).unwrap();
            let mut f: CUfunction = std::ptr::null_mut();
            let err = unsafe { (drv.cuModuleGetFunction)(&mut f, module, c.as_ptr()) };
            if err != 0 || f.is_null() { Err(INTERNAL_ERROR) } else { Ok(f as usize) }
        })
        .collect()
}

/// Kernels of one variant module, cached per context.
fn function(drv: &Driver, ctx: usize, kernel: Kernel, v: Variant) -> Result<usize, i32> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, Variant), Vec<usize>>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    let pos = ptxgen::KERNELS.iter().position(|k| *k == kernel).unwrap();
    if let Some(f) = cache.get(&(ctx, v)) {
        return Ok(f[pos]);
    }
    let names: Vec<String> = ptxgen::KERNELS.iter().map(|k| ptxgen::entry_name(*k, v)).collect();
    let f = load_module(drv, &ptxgen::module(v), &names, v.double)?;
    let r = f[pos];
    cache.insert((ctx, v), f);
    Ok(r)
}

fn index_function(drv: &Driver, ctx: usize, name: &str) -> Result<usize, i32> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Vec<usize>>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    let names = [ptxgen::COO2CSR, ptxgen::CSR2COO, ptxgen::IDENTITY];
    let pos = names.iter().position(|n| *n == name).unwrap();
    if let Some(f) = cache.get(&ctx) {
        return Ok(f[pos]);
    }
    let owned: Vec<String> = names.iter().map(|s| s.to_string()).collect();
    let f = load_module(drv, &ptxgen::index_module(), &owned, false)?;
    let r = f[pos];
    cache.insert(ctx, f);
    Ok(r)
}

fn launch_raw(drv: &Driver, f: usize, blocks: u64, batch: u32, block: u32, stream: usize, args: &[u64]) -> Result<(), i32> {
    // Every parameter slot is 8 bytes.
    let mut slots: Vec<u64> = args.to_vec();
    let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|v| (v as *mut u64).cast()).collect();
    let err = unsafe { (drv.cuLaunchKernel)(f as CUfunction, blocks as u32, batch.max(1), 1, block, 1, 1, 0, stream as CUstream, ptrs.as_mut_ptr(), std::ptr::null_mut()) };
    if err != 0 {
        eprintln!("CUDA for Intel cusparse: launch failed (err={err})");
        return Err(EXECUTION_FAILED);
    }
    Ok(())
}

fn run(h: &Handle, launches: &[Launch]) -> Result<(), i32> {
    if launches.is_empty() {
        return Ok(());
    }
    let drv = driver()?;
    let ctx = context(drv)?;
    for l in launches {
        let f = function(drv, ctx, l.kernel, l.variant)?;
        launch_raw(drv, f, l.blocks, l.batch, l.block, h.stream, &l.args)?;
    }
    Ok(())
}

fn run_index(h: &Handle, name: &str, threads: u64, args: &[u64]) -> Result<(), i32> {
    if threads == 0 {
        return Ok(());
    }
    let drv = driver()?;
    let ctx = context(drv)?;
    let f = index_function(drv, ctx, name)?;
    launch_raw(drv, f, threads.div_ceil(u64::from(plan::BLOCK)), 1, plan::BLOCK, h.stream, args)
}

// ---------------------------------------------------------------------------
// Handles and descriptors
// ---------------------------------------------------------------------------

pub struct Handle {
    stream: usize,
    pointer_mode: i32,
}

pub struct MatDescr {
    ty: i32,
    fill: i32,
    diag: i32,
    base: i32,
}

pub struct DnVec {
    size: i64,
    val: u64,
    ty: i32,
}

fn status(r: Result<(), i32>) -> i32 {
    match r {
        Ok(()) => SUCCESS,
        Err(e) => e,
    }
}

unsafe fn handle<'a>(h: *mut c_void) -> Result<&'a mut Handle, i32> {
    if h.is_null() { Err(NOT_INITIALIZED) } else { Ok(unsafe { &mut *h.cast::<Handle>() }) }
}

unsafe fn obj<'a, T>(p: *const c_void) -> Result<&'a mut T, i32> {
    if p.is_null() { Err(INVALID_VALUE) } else { Ok(unsafe { &mut *(p as *mut T) }) }
}

unsafe fn put<T>(out: *mut T, v: T) -> Result<(), i32> {
    if out.is_null() {
        return Err(INVALID_VALUE);
    }
    unsafe { out.write(v) };
    Ok(())
}

unsafe fn create<T>(out: *mut *mut c_void, v: Result<T, i32>) -> i32 {
    if out.is_null() {
        return INVALID_VALUE;
    }
    match v {
        Ok(v) => {
            unsafe { out.write(Box::into_raw(Box::new(v)).cast()) };
            SUCCESS
        }
        Err(e) => e,
    }
}

unsafe fn destroy<T>(p: *mut c_void) -> i32 {
    if !p.is_null() {
        drop(unsafe { Box::from_raw(p.cast::<T>()) });
    }
    SUCCESS
}

/// Reads alpha/beta per the handle's pointer mode and the value type.
unsafe fn scalars(h: &Handle, double: bool, alpha: *const c_void, beta: *const c_void) -> Result<Scalars, i32> {
    if alpha.is_null() || beta.is_null() {
        return Err(INVALID_VALUE);
    }
    if h.pointer_mode == 1 {
        return Ok(Scalars { alpha: 0, beta: 0, alpha_ptr: alpha as u64, beta_ptr: beta as u64 });
    }
    let read = |p: *const c_void| unsafe { if double { p.cast::<u64>().read_unaligned() } else { u64::from(p.cast::<u32>().read_unaligned()) } };
    Ok(Scalars { alpha: read(alpha), beta: read(beta), alpha_ptr: 0, beta_ptr: 0 })
}

#[unsafe(no_mangle)]
pub extern "system" fn cusparseGetErrorName(status: i32) -> *const c_char {
    status_name(status).as_ptr()
}

#[unsafe(no_mangle)]
pub extern "system" fn cusparseGetErrorString(status: i32) -> *const c_char {
    status_message(status).as_ptr()
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseGetVersion(_handle: *mut c_void, version: *mut i32) -> i32 {
    status(unsafe { put(version, VERSION) })
}

/// `libraryPropertyType`: MAJOR_VERSION 0, MINOR_VERSION 1, PATCH_LEVEL 2.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseGetProperty(kind: i32, value: *mut i32) -> i32 {
    let field = match kind {
        0 => VERSION / 1000,
        1 => (VERSION / 100) % 10,
        2 => VERSION % 100,
        _ => return INVALID_VALUE,
    };
    status(unsafe { put(value, field) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreate(out: *mut *mut c_void) -> i32 {
    unsafe { create(out, Ok(Handle { stream: 0, pointer_mode: 0 })) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDestroy(h: *mut c_void) -> i32 {
    if h.is_null() {
        return NOT_INITIALIZED;
    }
    unsafe { destroy::<Handle>(h) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSetStream(h: *mut c_void, stream: *mut c_void) -> i32 {
    status(unsafe { handle(h) }.map(|h| h.stream = stream as usize))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseGetStream(h: *mut c_void, stream: *mut *mut c_void) -> i32 {
    status(unsafe { handle(h) }.and_then(|h| unsafe { put(stream, h.stream as *mut c_void) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSetPointerMode(h: *mut c_void, mode: i32) -> i32 {
    status(unsafe { handle(h) }.and_then(|h| {
        if !matches!(mode, 0 | 1) {
            return Err(INVALID_VALUE);
        }
        h.pointer_mode = mode;
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseGetPointerMode(h: *mut c_void, mode: *mut i32) -> i32 {
    status(unsafe { handle(h) }.and_then(|h| unsafe { put(mode, h.pointer_mode) }))
}

// --- legacy matrix descriptor

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateMatDescr(out: *mut *mut c_void) -> i32 {
    unsafe { create(out, Ok(MatDescr { ty: 0, fill: 0, diag: 0, base: 0 })) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDestroyMatDescr(d: *mut c_void) -> i32 {
    unsafe { destroy::<MatDescr>(d) }
}

unsafe fn set_descr(d: *mut c_void, v: i32, max: i32, field: impl FnOnce(&mut MatDescr) -> &mut i32) -> i32 {
    status(unsafe { obj::<MatDescr>(d) }.and_then(|d| {
        if !(0..=max).contains(&v) {
            return Err(INVALID_VALUE);
        }
        *field(d) = v;
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSetMatType(d: *mut c_void, v: i32) -> i32 {
    unsafe { set_descr(d, v, 3, |d| &mut d.ty) }
}

/// Returns the field by value (0 for a null descriptor), as cuSPARSE does.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseGetMatType(d: *const c_void) -> i32 {
    unsafe { obj::<MatDescr>(d) }.map(|d| d.ty).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSetMatFillMode(d: *mut c_void, v: i32) -> i32 {
    unsafe { set_descr(d, v, 1, |d| &mut d.fill) }
}

/// Returns the field by value (0 for a null descriptor), as cuSPARSE does.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseGetMatFillMode(d: *const c_void) -> i32 {
    unsafe { obj::<MatDescr>(d) }.map(|d| d.fill).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSetMatDiagType(d: *mut c_void, v: i32) -> i32 {
    unsafe { set_descr(d, v, 1, |d| &mut d.diag) }
}

/// Returns the field by value (0 for a null descriptor), as cuSPARSE does.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseGetMatDiagType(d: *const c_void) -> i32 {
    unsafe { obj::<MatDescr>(d) }.map(|d| d.diag).unwrap_or(0)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSetMatIndexBase(d: *mut c_void, v: i32) -> i32 {
    unsafe { set_descr(d, v, 1, |d| &mut d.base) }
}

/// Returns the field by value (0 for a null descriptor), as cuSPARSE does.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseGetMatIndexBase(d: *const c_void) -> i32 {
    unsafe { obj::<MatDescr>(d) }.map(|d| d.base).unwrap_or(0)
}

// --- generic API: sparse matrices

#[allow(clippy::too_many_arguments)]
unsafe fn create_sp(out: *mut *mut c_void, format: i32, rows: i64, cols: i64, nnz: i64, ptr: *const c_void, idx: *const c_void, val: *const c_void, ptr_ty: i32, idx_ty: i32, base: i32, val_ty: i32) -> i32 {
    let m = SpMat::new(format, rows, cols, nnz, ptr as u64, idx as u64, val as u64, ptr_ty, idx_ty, base, val_ty);
    if let Ok(m) = &m {
        trace!("create {} {}x{} nnz={nnz} ptr_ty={ptr_ty} idx_ty={idx_ty} base={base} val_ty={val_ty}", match format { 1 => "csr", 2 => "csc", _ => "coo" }, m.rows, m.cols);
    }
    unsafe { create(out, m) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateCsr(out: *mut *mut c_void, rows: i64, cols: i64, nnz: i64, offsets: *mut c_void, columns: *mut c_void, values: *mut c_void, off_ty: i32, col_ty: i32, base: i32, val_ty: i32) -> i32 {
    unsafe { create_sp(out, plan::FORMAT_CSR, rows, cols, nnz, offsets, columns, values, off_ty, col_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateConstCsr(out: *mut *mut c_void, rows: i64, cols: i64, nnz: i64, offsets: *const c_void, columns: *const c_void, values: *const c_void, off_ty: i32, col_ty: i32, base: i32, val_ty: i32) -> i32 {
    unsafe { create_sp(out, plan::FORMAT_CSR, rows, cols, nnz, offsets, columns, values, off_ty, col_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateCsc(out: *mut *mut c_void, rows: i64, cols: i64, nnz: i64, offsets: *mut c_void, row_ind: *mut c_void, values: *mut c_void, off_ty: i32, row_ty: i32, base: i32, val_ty: i32) -> i32 {
    unsafe { create_sp(out, plan::FORMAT_CSC, rows, cols, nnz, offsets, row_ind, values, off_ty, row_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateConstCsc(out: *mut *mut c_void, rows: i64, cols: i64, nnz: i64, offsets: *const c_void, row_ind: *const c_void, values: *const c_void, off_ty: i32, row_ty: i32, base: i32, val_ty: i32) -> i32 {
    unsafe { create_sp(out, plan::FORMAT_CSC, rows, cols, nnz, offsets, row_ind, values, off_ty, row_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateCoo(out: *mut *mut c_void, rows: i64, cols: i64, nnz: i64, row_ind: *mut c_void, col_ind: *mut c_void, values: *mut c_void, idx_ty: i32, base: i32, val_ty: i32) -> i32 {
    unsafe { create_sp(out, plan::FORMAT_COO, rows, cols, nnz, row_ind, col_ind, values, idx_ty, idx_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateConstCoo(out: *mut *mut c_void, rows: i64, cols: i64, nnz: i64, row_ind: *const c_void, col_ind: *const c_void, values: *const c_void, idx_ty: i32, base: i32, val_ty: i32) -> i32 {
    unsafe { create_sp(out, plan::FORMAT_COO, rows, cols, nnz, row_ind, col_ind, values, idx_ty, idx_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDestroySpMat(d: *mut c_void) -> i32 {
    unsafe { destroy::<SpMat>(d) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMatGetSize(d: *const c_void, rows: *mut i64, cols: *mut i64, nnz: *mut i64) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| unsafe {
        put(rows, m.rows)?;
        put(cols, m.cols)?;
        put(nnz, m.nnz)
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMatGetFormat(d: *const c_void, format: *mut i32) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| unsafe { put(format, m.format) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMatGetIndexBase(d: *const c_void, base: *mut i32) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| unsafe { put(base, m.base) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMatGetValues(d: *mut c_void, values: *mut *mut c_void) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| unsafe { put(values, m.val as *mut c_void) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseConstSpMatGetValues(d: *const c_void, values: *mut *const c_void) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| unsafe { put(values, m.val as *const c_void) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMatSetValues(d: *mut c_void, values: *mut c_void) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.map(|m| m.val = values as u64))
}

/// `cusparseSpMatAttribute_t`: FILL_MODE 0, DIAG_TYPE 1, each an `int`-sized enum.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMatGetAttribute(d: *const c_void, attr: i32, data: *mut c_void, size: usize) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| {
        if size != 4 {
            return Err(INVALID_VALUE);
        }
        let v = match attr {
            0 => m.fill,
            1 => m.diag,
            _ => return Err(INVALID_VALUE),
        };
        unsafe { put(data.cast::<i32>(), v) }
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMatSetAttribute(d: *mut c_void, attr: i32, data: *const c_void, size: usize) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| {
        if size != 4 || data.is_null() {
            return Err(INVALID_VALUE);
        }
        let v = unsafe { data.cast::<i32>().read_unaligned() };
        if !matches!(v, 0 | 1) {
            return Err(INVALID_VALUE);
        }
        match attr {
            0 => m.fill = v,
            1 => m.diag = v,
            _ => return Err(INVALID_VALUE),
        }
        Ok(())
    }))
}

unsafe fn set_pointers(d: *mut c_void, format: i32, ptr: *mut c_void, idx: *mut c_void, val: *mut c_void) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| {
        if m.format != format {
            return Err(INVALID_VALUE);
        }
        (m.ptr, m.idx, m.val) = (ptr as u64, idx as u64, val as u64);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCsrSetPointers(d: *mut c_void, offsets: *mut c_void, columns: *mut c_void, values: *mut c_void) -> i32 {
    unsafe { set_pointers(d, plan::FORMAT_CSR, offsets, columns, values) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCscSetPointers(d: *mut c_void, offsets: *mut c_void, rows: *mut c_void, values: *mut c_void) -> i32 {
    unsafe { set_pointers(d, plan::FORMAT_CSC, offsets, rows, values) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCooSetPointers(d: *mut c_void, rows: *mut c_void, cols: *mut c_void, values: *mut c_void) -> i32 {
    unsafe { set_pointers(d, plan::FORMAT_COO, rows, cols, values) }
}

#[allow(clippy::too_many_arguments)]
unsafe fn compressed_get(d: *const c_void, format: i32, rows: *mut i64, cols: *mut i64, nnz: *mut i64, ptr: *mut *mut c_void, idx: *mut *mut c_void, val: *mut *mut c_void, ptr_ty: *mut i32, idx_ty: *mut i32, base: *mut i32, val_ty: *mut i32) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| unsafe {
        if m.format != format {
            return Err(INVALID_VALUE);
        }
        put(rows, m.rows)?;
        put(cols, m.cols)?;
        put(nnz, m.nnz)?;
        put(ptr, m.ptr as *mut c_void)?;
        put(idx, m.idx as *mut c_void)?;
        put(val, m.val as *mut c_void)?;
        if format != plan::FORMAT_COO {
            put(ptr_ty, m.ptr_ty)?;
        }
        put(idx_ty, m.idx_ty)?;
        put(base, m.base)?;
        put(val_ty, m.val_ty)
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCsrGet(d: *mut c_void, rows: *mut i64, cols: *mut i64, nnz: *mut i64, offsets: *mut *mut c_void, columns: *mut *mut c_void, values: *mut *mut c_void, off_ty: *mut i32, col_ty: *mut i32, base: *mut i32, val_ty: *mut i32) -> i32 {
    unsafe { compressed_get(d, plan::FORMAT_CSR, rows, cols, nnz, offsets, columns, values, off_ty, col_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseConstCsrGet(d: *const c_void, rows: *mut i64, cols: *mut i64, nnz: *mut i64, offsets: *mut *mut c_void, columns: *mut *mut c_void, values: *mut *mut c_void, off_ty: *mut i32, col_ty: *mut i32, base: *mut i32, val_ty: *mut i32) -> i32 {
    unsafe { compressed_get(d, plan::FORMAT_CSR, rows, cols, nnz, offsets, columns, values, off_ty, col_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCscGet(d: *mut c_void, rows: *mut i64, cols: *mut i64, nnz: *mut i64, offsets: *mut *mut c_void, row_ind: *mut *mut c_void, values: *mut *mut c_void, off_ty: *mut i32, row_ty: *mut i32, base: *mut i32, val_ty: *mut i32) -> i32 {
    unsafe { compressed_get(d, plan::FORMAT_CSC, rows, cols, nnz, offsets, row_ind, values, off_ty, row_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseConstCscGet(d: *const c_void, rows: *mut i64, cols: *mut i64, nnz: *mut i64, offsets: *mut *mut c_void, row_ind: *mut *mut c_void, values: *mut *mut c_void, off_ty: *mut i32, row_ty: *mut i32, base: *mut i32, val_ty: *mut i32) -> i32 {
    unsafe { compressed_get(d, plan::FORMAT_CSC, rows, cols, nnz, offsets, row_ind, values, off_ty, row_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCooGet(d: *mut c_void, rows: *mut i64, cols: *mut i64, nnz: *mut i64, row_ind: *mut *mut c_void, col_ind: *mut *mut c_void, values: *mut *mut c_void, idx_ty: *mut i32, base: *mut i32, val_ty: *mut i32) -> i32 {
    unsafe { compressed_get(d, plan::FORMAT_COO, rows, cols, nnz, row_ind, col_ind, values, std::ptr::null_mut(), idx_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseConstCooGet(d: *const c_void, rows: *mut i64, cols: *mut i64, nnz: *mut i64, row_ind: *mut *mut c_void, col_ind: *mut *mut c_void, values: *mut *mut c_void, idx_ty: *mut i32, base: *mut i32, val_ty: *mut i32) -> i32 {
    unsafe { compressed_get(d, plan::FORMAT_COO, rows, cols, nnz, row_ind, col_ind, values, std::ptr::null_mut(), idx_ty, base, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCsrSetStridedBatch(d: *mut c_void, count: c_int, offsets_stride: i64, values_stride: i64) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| {
        if count < 1 || offsets_stride < 0 || values_stride < 0 || m.format == plan::FORMAT_COO {
            return Err(INVALID_VALUE);
        }
        (m.batch, m.ptr_bstride, m.idx_bstride) = (count, offsets_stride, values_stride);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCooSetStridedBatch(d: *mut c_void, count: c_int, stride: i64) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| {
        if count < 1 || stride < 0 || m.format != plan::FORMAT_COO {
            return Err(INVALID_VALUE);
        }
        (m.batch, m.ptr_bstride, m.idx_bstride) = (count, stride, stride);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMatGetStridedBatch(d: *const c_void, count: *mut c_int) -> i32 {
    status(unsafe { obj::<SpMat>(d) }.and_then(|m| unsafe { put(count, m.batch) }))
}

// --- generic API: dense vectors and matrices

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateDnVec(out: *mut *mut c_void, size: i64, values: *mut c_void, val_ty: i32) -> i32 {
    unsafe { create(out, if size < 0 { Err(INVALID_VALUE) } else { Ok(DnVec { size, val: values as u64, ty: val_ty }) }) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateConstDnVec(out: *mut *mut c_void, size: i64, values: *const c_void, val_ty: i32) -> i32 {
    unsafe { cusparseCreateDnVec(out, size, values as *mut c_void, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDestroyDnVec(d: *mut c_void) -> i32 {
    unsafe { destroy::<DnVec>(d) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDnVecGet(d: *mut c_void, size: *mut i64, values: *mut *mut c_void, val_ty: *mut i32) -> i32 {
    status(unsafe { obj::<DnVec>(d) }.and_then(|v| unsafe {
        put(size, v.size)?;
        put(values, v.val as *mut c_void)?;
        put(val_ty, v.ty)
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseConstDnVecGet(d: *const c_void, size: *mut i64, values: *mut *mut c_void, val_ty: *mut i32) -> i32 {
    unsafe { cusparseDnVecGet(d as *mut c_void, size, values, val_ty) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDnVecGetValues(d: *mut c_void, values: *mut *mut c_void) -> i32 {
    status(unsafe { obj::<DnVec>(d) }.and_then(|v| unsafe { put(values, v.val as *mut c_void) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseConstDnVecGetValues(d: *const c_void, values: *mut *mut c_void) -> i32 {
    unsafe { cusparseDnVecGetValues(d as *mut c_void, values) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDnVecSetValues(d: *mut c_void, values: *mut c_void) -> i32 {
    status(unsafe { obj::<DnVec>(d) }.map(|v| v.val = values as u64))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateDnMat(out: *mut *mut c_void, rows: i64, cols: i64, ld: i64, values: *mut c_void, val_ty: i32, order: i32) -> i32 {
    unsafe { create(out, DnMat::new(rows, cols, ld, values as u64, val_ty, order)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateConstDnMat(out: *mut *mut c_void, rows: i64, cols: i64, ld: i64, values: *const c_void, val_ty: i32, order: i32) -> i32 {
    unsafe { create(out, DnMat::new(rows, cols, ld, values as u64, val_ty, order)) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDestroyDnMat(d: *mut c_void) -> i32 {
    unsafe { destroy::<DnMat>(d) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDnMatGet(d: *mut c_void, rows: *mut i64, cols: *mut i64, ld: *mut i64, values: *mut *mut c_void, val_ty: *mut i32, order: *mut i32) -> i32 {
    status(unsafe { obj::<DnMat>(d) }.and_then(|m| unsafe {
        put(rows, m.rows)?;
        put(cols, m.cols)?;
        put(ld, m.ld)?;
        put(values, m.val as *mut c_void)?;
        put(val_ty, m.val_ty)?;
        put(order, m.order)
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseConstDnMatGet(d: *const c_void, rows: *mut i64, cols: *mut i64, ld: *mut i64, values: *mut *mut c_void, val_ty: *mut i32, order: *mut i32) -> i32 {
    unsafe { cusparseDnMatGet(d as *mut c_void, rows, cols, ld, values, val_ty, order) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDnMatGetValues(d: *mut c_void, values: *mut *mut c_void) -> i32 {
    status(unsafe { obj::<DnMat>(d) }.and_then(|m| unsafe { put(values, m.val as *mut c_void) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseConstDnMatGetValues(d: *const c_void, values: *mut *mut c_void) -> i32 {
    unsafe { cusparseDnMatGetValues(d as *mut c_void, values) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDnMatSetValues(d: *mut c_void, values: *mut c_void) -> i32 {
    status(unsafe { obj::<DnMat>(d) }.map(|m| m.val = values as u64))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDnMatSetStridedBatch(d: *mut c_void, count: c_int, stride: i64) -> i32 {
    status(unsafe { obj::<DnMat>(d) }.and_then(|m| {
        if count < 1 || stride < 0 {
            return Err(INVALID_VALUE);
        }
        (m.batch, m.bstride) = (count, stride);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDnMatGetStridedBatch(d: *const c_void, count: *mut c_int, stride: *mut i64) -> i32 {
    status(unsafe { obj::<DnMat>(d) }.and_then(|m| unsafe {
        put(count, m.batch)?;
        put(stride, m.bstride)
    }))
}

// ---------------------------------------------------------------------------
// Operations
// ---------------------------------------------------------------------------

/// Validates an SpMV call and returns its launch list.
#[allow(clippy::too_many_arguments)]
unsafe fn spmv_plan(h: *mut c_void, op: i32, alpha: *const c_void, a: *const c_void, x: *const c_void, beta: *const c_void, y: *const c_void, compute: i32) -> Result<(&'static Handle, Vec<Launch>), i32> {
    let h = unsafe { handle(h) }?;
    let (a, x, y) = unsafe { (obj::<SpMat>(a)?, obj::<DnVec>(x)?, obj::<DnVec>(y)?) };
    let double = plan::val_double(a.val_ty)?;
    let s = unsafe { scalars(h, double, alpha, beta) }?;
    let xd = DnMat::vector(x.size, x.val, x.ty)?;
    let yd = DnMat::vector(y.size, y.val, y.ty)?;
    let l = plan::spmv(op, a, &xd, &yd, compute, s, choice())?;
    trace!("spmv op={op} {}x{} nnz={} fmt={} {:?}", a.rows, a.cols, a.nnz, a.format, l.iter().map(|l| l.kernel).collect::<Vec<_>>());
    Ok((h, l))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMV(h: *mut c_void, op: i32, alpha: *const c_void, a: *const c_void, x: *const c_void, beta: *const c_void, y: *mut c_void, compute: i32, _alg: i32, _buffer: *mut c_void) -> i32 {
    status(unsafe { spmv_plan(h, op, alpha, a, x, beta, y, compute) }.and_then(|(h, l)| run(h, &l)))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMV_bufferSize(h: *mut c_void, op: i32, alpha: *const c_void, a: *const c_void, x: *const c_void, beta: *const c_void, y: *mut c_void, compute: i32, _alg: i32, size: *mut usize) -> i32 {
    status(unsafe { spmv_plan(h, op, alpha, a, x, beta, y, compute) }.and_then(|_| unsafe { put(size, 0) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMV_preprocess(h: *mut c_void, op: i32, alpha: *const c_void, a: *const c_void, x: *const c_void, beta: *const c_void, y: *mut c_void, compute: i32, _alg: i32, _buffer: *mut c_void) -> i32 {
    status(unsafe { spmv_plan(h, op, alpha, a, x, beta, y, compute) }.map(|_| ()))
}

#[allow(clippy::too_many_arguments)]
unsafe fn spmm_plan(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *const c_void, compute: i32) -> Result<(&'static Handle, Vec<Launch>), i32> {
    let h = unsafe { handle(h) }?;
    let (a, b, c) = unsafe { (obj::<SpMat>(a)?, obj::<DnMat>(b)?, obj::<DnMat>(c)?) };
    let double = plan::val_double(a.val_ty)?;
    let s = unsafe { scalars(h, double, alpha, beta) }?;
    let l = plan::spmm(op_a, op_b, a, b, c, compute, s, choice())?;
    trace!("spmm op_a={op_a} op_b={op_b} A {}x{} nnz={} fmt={} B {}x{} order={} C {}x{} batch={} {:?}", a.rows, a.cols, a.nnz, a.format, b.rows, b.cols, b.order, c.rows, c.cols, c.batch, l.iter().map(|l| l.kernel).collect::<Vec<_>>());
    Ok((h, l))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMM(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *mut c_void, compute: i32, _alg: i32, _buffer: *mut c_void) -> i32 {
    status(unsafe { spmm_plan(h, op_a, op_b, alpha, a, b, beta, c, compute) }.and_then(|(h, l)| run(h, &l)))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMM_bufferSize(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *mut c_void, compute: i32, _alg: i32, size: *mut usize) -> i32 {
    status(unsafe { spmm_plan(h, op_a, op_b, alpha, a, b, beta, c, compute) }.and_then(|_| unsafe { put(size, 0) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpMM_preprocess(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *mut c_void, compute: i32, _alg: i32, _buffer: *mut c_void) -> i32 {
    status(unsafe { spmm_plan(h, op_a, op_b, alpha, a, b, beta, c, compute) }.map(|_| ()))
}

#[allow(clippy::too_many_arguments)]
unsafe fn sddmm_plan(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *const c_void, compute: i32) -> Result<(&'static Handle, Vec<Launch>), i32> {
    let h = unsafe { handle(h) }?;
    let (a, b, c) = unsafe { (obj::<DnMat>(a)?, obj::<DnMat>(b)?, obj::<SpMat>(c)?) };
    let double = plan::val_double(c.val_ty)?;
    let s = unsafe { scalars(h, double, alpha, beta) }?;
    let l = plan::sddmm(op_a, op_b, a, b, c, compute, s)?;
    trace!("sddmm op_a={op_a} op_b={op_b} C {}x{} nnz={} batch={}", c.rows, c.cols, c.nnz, c.batch);
    Ok((h, l))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSDDMM(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *mut c_void, compute: i32, _alg: i32, _buffer: *mut c_void) -> i32 {
    status(unsafe { sddmm_plan(h, op_a, op_b, alpha, a, b, beta, c, compute) }.and_then(|(h, l)| run(h, &l)))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSDDMM_bufferSize(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *mut c_void, compute: i32, _alg: i32, size: *mut usize) -> i32 {
    status(unsafe { sddmm_plan(h, op_a, op_b, alpha, a, b, beta, c, compute) }.and_then(|_| unsafe { put(size, 0) }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSDDMM_preprocess(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *mut c_void, compute: i32, _alg: i32, _buffer: *mut c_void) -> i32 {
    status(unsafe { sddmm_plan(h, op_a, op_b, alpha, a, b, beta, c, compute) }.map(|_| ()))
}

// --- legacy block-CSR products (32-bit indices, column-major dense operands)

#[allow(clippy::too_many_arguments)]
unsafe fn bsrmm(double: bool, h: *mut c_void, dir: i32, op_a: i32, op_b: i32, mb: c_int, n: c_int, kb: c_int, nnzb: c_int, alpha: *const c_void, descr: *const c_void, val: *const c_void, ptr: *const c_void, col: *const c_void, bd: c_int, b: *const c_void, ldb: c_int, beta: *const c_void, c: *mut c_void, ldc: c_int) -> i32 {
    status((|| {
        let h = unsafe { handle(h) }?;
        let d = unsafe { obj::<MatDescr>(descr) }?;
        if d.ty != 0 {
            return Err(MATRIX_TYPE_NOT_SUPPORTED);
        }
        if op_a != plan::OP_N {
            return Err(NOT_SUPPORTED);
        }
        let s = unsafe { scalars(h, double, alpha, beta) }?;
        let l = plan::bsrmm(double, dir, mb.into(), n.into(), kb.into(), nnzb.into(), bd.into(), d.base, op_b, ptr as u64, col as u64, val as u64, b as u64, ldb.into(), c as u64, ldc.into(), s)?;
        trace!("bsrmm double={double} dir={dir} mb={mb} kb={kb} n={n} nnzb={nnzb} bd={bd} op_b={op_b}");
        run(h, &l)
    })())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSbsrmm(h: *mut c_void, dir: i32, op_a: i32, op_b: i32, mb: c_int, n: c_int, kb: c_int, nnzb: c_int, alpha: *const f32, descr: *const c_void, val: *const f32, ptr: *const c_int, col: *const c_int, bd: c_int, b: *const f32, ldb: c_int, beta: *const f32, c: *mut f32, ldc: c_int) -> i32 {
    unsafe { bsrmm(false, h, dir, op_a, op_b, mb, n, kb, nnzb, alpha.cast(), descr, val.cast(), ptr.cast(), col.cast(), bd, b.cast(), ldb, beta.cast(), c.cast(), ldc) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDbsrmm(h: *mut c_void, dir: i32, op_a: i32, op_b: i32, mb: c_int, n: c_int, kb: c_int, nnzb: c_int, alpha: *const f64, descr: *const c_void, val: *const f64, ptr: *const c_int, col: *const c_int, bd: c_int, b: *const f64, ldb: c_int, beta: *const f64, c: *mut f64, ldc: c_int) -> i32 {
    unsafe { bsrmm(true, h, dir, op_a, op_b, mb, n, kb, nnzb, alpha.cast(), descr, val.cast(), ptr.cast(), col.cast(), bd, b.cast(), ldb, beta.cast(), c.cast(), ldc) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSbsrmv(h: *mut c_void, dir: i32, op_a: i32, mb: c_int, nb: c_int, nnzb: c_int, alpha: *const f32, descr: *const c_void, val: *const f32, ptr: *const c_int, col: *const c_int, bd: c_int, x: *const f32, beta: *const f32, y: *mut f32) -> i32 {
    let (ldb, ldc) = (nb.saturating_mul(bd).max(1), mb.saturating_mul(bd).max(1));
    unsafe { bsrmm(false, h, dir, op_a, plan::OP_N, mb, 1, nb, nnzb, alpha.cast(), descr, val.cast(), ptr.cast(), col.cast(), bd, x.cast(), ldb, beta.cast(), y.cast(), ldc) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDbsrmv(h: *mut c_void, dir: i32, op_a: i32, mb: c_int, nb: c_int, nnzb: c_int, alpha: *const f64, descr: *const c_void, val: *const f64, ptr: *const c_int, col: *const c_int, bd: c_int, x: *const f64, beta: *const f64, y: *mut f64) -> i32 {
    let (ldb, ldc) = (nb.saturating_mul(bd).max(1), mb.saturating_mul(bd).max(1));
    unsafe { bsrmm(true, h, dir, op_a, plan::OP_N, mb, 1, nb, nnzb, alpha.cast(), descr, val.cast(), ptr.cast(), col.cast(), bd, x.cast(), ldb, beta.cast(), y.cast(), ldc) }
}

// --- index conversions (32-bit)

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseXcoo2csr(h: *mut c_void, coo_rows: *const c_int, nnz: c_int, m: c_int, csr_ptr: *mut c_int, base: i32) -> i32 {
    status((|| {
        let h = unsafe { handle(h) }?;
        if nnz < 0 || m < 0 || !matches!(base, 0 | 1) || csr_ptr.is_null() || (nnz > 0 && coo_rows.is_null()) {
            return Err(INVALID_VALUE);
        }
        run_index(h, ptxgen::COO2CSR, m as u64 + 1, &[coo_rows as u64, nnz as u64, m as u64, csr_ptr as u64, base as u64])
    })())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseXcsr2coo(h: *mut c_void, csr_ptr: *const c_int, nnz: c_int, m: c_int, coo_rows: *mut c_int, base: i32) -> i32 {
    status((|| {
        let h = unsafe { handle(h) }?;
        if nnz < 0 || m < 0 || !matches!(base, 0 | 1) || csr_ptr.is_null() || (nnz > 0 && coo_rows.is_null()) {
            return Err(INVALID_VALUE);
        }
        run_index(h, ptxgen::CSR2COO, m as u64, &[csr_ptr as u64, nnz as u64, m as u64, coo_rows as u64, base as u64])
    })())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseCreateIdentityPermutation(h: *mut c_void, n: c_int, p: *mut c_int) -> i32 {
    status((|| {
        let h = unsafe { handle(h) }?;
        if n < 0 || (n > 0 && p.is_null()) {
            return Err(INVALID_VALUE);
        }
        run_index(h, ptxgen::IDENTITY, n as u64, &[p as u64, n as u64, 0, 0, 0])
    })())
}

// ---------------------------------------------------------------------------
// Sparse products: SpGEMM (CSR x CSR) and csrgeam2 (CSR + CSR)
// ---------------------------------------------------------------------------

fn product_function(drv: &Driver, ctx: usize, stem: &str, double: bool) -> Result<usize, i32> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, bool), Vec<usize>>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner());
    let pos = ptxgen::PRODUCT_KERNELS.iter().position(|(n, _)| *n == stem).unwrap();
    if let Some(f) = cache.get(&(ctx, double)) {
        return Ok(f[pos]);
    }
    let names: Vec<String> = ptxgen::PRODUCT_KERNELS.iter().map(|(n, _)| ptxgen::product_name(n, double)).collect();
    let f = load_module(drv, &ptxgen::product_module(double), &names, double)?;
    let r = f[pos];
    cache.insert((ctx, double), f);
    Ok(r)
}

/// One thread per row of `rows` rows.
fn run_product(h: &Handle, stem: &str, double: bool, rows: u64, args: &[u64]) -> Result<(), i32> {
    if rows == 0 {
        return Ok(());
    }
    let drv = driver()?;
    let ctx = context(drv)?;
    let f = product_function(drv, ctx, stem, double)?;
    launch_raw(drv, f, rows.div_ceil(u64::from(plan::BLOCK)), 1, plan::BLOCK, h.stream, args)
}

fn sync(h: &Handle) -> Result<&'static Driver, i32> {
    let drv = driver()?;
    if unsafe { (drv.cuStreamSynchronize)(h.stream as CUstream) } != 0 {
        return Err(EXECUTION_FAILED);
    }
    Ok(drv)
}

fn download_u64(h: &Handle, src: u64, n: usize) -> Result<Vec<u64>, i32> {
    let drv = sync(h)?;
    let mut v = vec![0u64; n];
    if n > 0 && unsafe { (drv.cuMemcpyDtoH_v2)(v.as_mut_ptr().cast(), src, n * 8) } != 0 {
        return Err(EXECUTION_FAILED);
    }
    Ok(v)
}

fn upload(dst: u64, bytes: &[u8]) -> Result<(), i32> {
    let drv = driver()?;
    if !bytes.is_empty() && unsafe { (drv.cuMemcpyHtoD_v2)(dst, bytes.as_ptr().cast(), bytes.len()) } != 0 {
        return Err(EXECUTION_FAILED);
    }
    Ok(())
}

/// `base + exclusive prefix sums` of `counts` (m + 1 entries).
fn offsets(counts: &[u64], base: u64) -> Vec<u64> {
    let mut out = Vec::with_capacity(counts.len() + 1);
    let mut acc = 0u64;
    out.push(base);
    for &c in counts {
        acc += c;
        out.push(acc + base);
    }
    out
}

fn index_bytes(v: &[u64], wide: bool) -> Vec<u8> {
    if wide { v.iter().flat_map(|x| x.to_le_bytes()).collect() } else { v.iter().flat_map(|&x| (x as u32).to_le_bytes()).collect() }
}

/// SpGEMM state between `workEstimation`, `compute` and `copy`.
#[derive(Default)]
pub struct SpGemm {
    /// Device row offsets of the expanded products (buffer 1), and their total.
    off: u64,
    products: u64,
    /// Buffer 2: sorted product lists (columns, values) and per-row lengths.
    tcol: u64,
    tval: u64,
    count: u64,
    /// C's row offsets (with C's index base) once `compute` has run.
    c_ptr: Vec<u64>,
}

fn align8(x: u64) -> u64 {
    x.div_ceil(8) * 8
}

/// Checks a SpGEMM call: CSR operands, op N, matching f32/f64 values and compute type, and
/// beta = 0. Returns (A, B, C, double).
unsafe fn spgemm_args(h: &Handle, op_a: i32, op_b: i32, beta: *const c_void, a: *const c_void, b: *const c_void, c: *mut c_void, compute: i32) -> Result<(&'static SpMat, &'static SpMat, &'static mut SpMat, bool), i32> {
    let a = unsafe { obj::<SpMat>(a) }?;
    let b = unsafe { obj::<SpMat>(b) }?;
    let c = unsafe { obj::<SpMat>(c) }?;
    if op_a != plan::OP_N || op_b != plan::OP_N {
        return Err(NOT_SUPPORTED);
    }
    if [a.format, b.format, c.format].iter().any(|&f| f != plan::FORMAT_CSR) || a.batch > 1 || b.batch > 1 {
        return Err(NOT_SUPPORTED);
    }
    let double = match compute {
        plan::R_32F => false,
        plan::R_64F => true,
        _ => return Err(NOT_SUPPORTED),
    };
    let vt = if double { plan::R_64F } else { plan::R_32F };
    if a.val_ty != vt || b.val_ty != vt || c.val_ty != vt {
        return Err(NOT_SUPPORTED);
    }
    if a.cols != b.rows || c.rows != a.rows || c.cols != b.cols {
        return Err(INVALID_VALUE);
    }
    // beta * C is not supported (cuSPARSE's own SpGEMM requires beta = 0 as well)
    if beta.is_null() {
        return Err(INVALID_VALUE);
    }
    if h.pointer_mode == 0 {
        let zero = unsafe { if double { beta.cast::<f64>().read_unaligned() == 0.0 } else { beta.cast::<f32>().read_unaligned() == 0.0 } };
        if !zero {
            return Err(NOT_SUPPORTED);
        }
    }
    Ok((a, b, c, double))
}

fn wide(ty: i32) -> bool {
    ty == plan::INDEX_64I
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpGEMM_createDescr(out: *mut *mut c_void) -> i32 {
    unsafe { create(out, Ok(SpGemm::default())) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpGEMM_destroyDescr(d: *mut c_void) -> i32 {
    unsafe { destroy::<SpGemm>(d) }
}

/// First phase: products per row of C (buffer 1 = m + 1 u64 offsets).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpGEMM_workEstimation(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *mut c_void, compute: i32, _alg: i32, d: *mut c_void, size1: *mut usize, buffer1: *mut c_void) -> i32 {
    status((|| {
        let h = unsafe { handle(h) }?;
        let (a, b, _c, double) = unsafe { spgemm_args(h, op_a, op_b, beta, a, b, c, compute) }?;
        let g = unsafe { obj::<SpGemm>(d) }?;
        if size1.is_null() || alpha.is_null() {
            return Err(INVALID_VALUE);
        }
        let need = (a.rows as usize + 1) * 8;
        if buffer1.is_null() {
            return unsafe { put(size1, need) };
        }
        if unsafe { *size1 } < need {
            return Err(INVALID_VALUE);
        }
        let m = a.rows as u64;
        let flags = u64::from(wide(a.ptr_ty)) | u64::from(wide(a.idx_ty)) << 1 | u64::from(wide(b.ptr_ty)) << 2;
        let buf = buffer1 as u64;
        run_product(h, ptxgen::ROWBOUND, double, m, &[a.ptr, a.idx, b.ptr, buf, m, flags, a.base as u64, b.base as u64])?;
        let bound = download_u64(h, buf, m as usize)?;
        let off = offsets(&bound, 0);
        upload(buf, &index_bytes(&off, true))?;
        g.off = buf;
        g.products = off[m as usize];
        trace!("SpGEMM workEstimation {}x{} * {}x{}: {} products", a.rows, a.cols, b.rows, b.cols, g.products);
        Ok(())
    })())
}

/// Second phase: each row's sorted, merged product list in buffer 2; C's size.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpGEMM_compute(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *mut c_void, compute: i32, _alg: i32, d: *mut c_void, size2: *mut usize, buffer2: *mut c_void) -> i32 {
    status((|| {
        let h = unsafe { handle(h) }?;
        let (a, b, c, double) = unsafe { spgemm_args(h, op_a, op_b, beta, a, b, c, compute) }?;
        let g = unsafe { obj::<SpGemm>(d) }?;
        if size2.is_null() || alpha.is_null() || g.off == 0 {
            return Err(INVALID_VALUE);
        }
        let es = if double { 8 } else { 4 };
        let m = a.rows as u64;
        let tval = g.products * 8;
        let count = align8(tval + g.products * es);
        let need = (count + m * 8).max(8) as usize;
        if buffer2.is_null() {
            return unsafe { put(size2, need) };
        }
        if unsafe { *size2 } < need {
            return Err(INVALID_VALUE);
        }
        let buf = buffer2 as u64;
        (g.tcol, g.tval, g.count) = (buf, buf + tval, buf + count);
        let sc = unsafe { scalars(h, double, alpha, alpha) }?;
        let flags = u64::from(wide(a.ptr_ty)) | u64::from(wide(a.idx_ty)) << 1 | u64::from(wide(b.ptr_ty)) << 2 | u64::from(wide(b.idx_ty)) << 3;
        run_product(h, ptxgen::EXPAND, double, m, &[a.ptr, a.idx, a.val, b.ptr, b.idx, b.val, g.off, g.tcol, g.tval, g.count, m, flags, a.base as u64, b.base as u64, sc.alpha, sc.alpha_ptr])?;
        let counts = download_u64(h, g.count, m as usize)?;
        g.c_ptr = offsets(&counts, c.base as u64);
        c.nnz = (g.c_ptr[m as usize] - c.base as u64) as i64;
        trace!("SpGEMM compute: C {}x{} nnz={}", c.rows, c.cols, c.nnz);
        Ok(())
    })())
}

/// Third phase: C's row offsets, columns and values (after `cusparseCsrSetPointers`).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpGEMM_copy(h: *mut c_void, op_a: i32, op_b: i32, alpha: *const c_void, a: *const c_void, b: *const c_void, beta: *const c_void, c: *mut c_void, compute: i32, _alg: i32, d: *mut c_void) -> i32 {
    status((|| {
        let h = unsafe { handle(h) }?;
        let (a, _b, c, double) = unsafe { spgemm_args(h, op_a, op_b, beta, a, b, c, compute) }?;
        let g = unsafe { obj::<SpGemm>(d) }?;
        if alpha.is_null() || g.c_ptr.len() != a.rows as usize + 1 {
            return Err(INVALID_VALUE);
        }
        if c.ptr == 0 || (c.nnz > 0 && (c.idx == 0 || c.val == 0)) {
            return Err(INVALID_VALUE);
        }
        upload(c.ptr, &index_bytes(&g.c_ptr, wide(c.ptr_ty)))?;
        let m = a.rows as u64;
        let flags = u64::from(wide(c.ptr_ty)) | u64::from(wide(c.idx_ty)) << 1;
        run_product(h, ptxgen::COMPACT, double, m, &[g.off, g.tcol, g.tval, g.count, c.ptr, c.idx, c.val, m, flags, c.base as u64])
    })())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseSpGEMM_getNumProducts(d: *mut c_void, products: *mut i64) -> i32 {
    status(unsafe { obj::<SpGemm>(d) }.and_then(|g| unsafe { put(products, g.products as i64) }))
}

/// Legacy-descriptor index base (general matrices only).
unsafe fn descr_base(d: *const c_void) -> Result<u64, i32> {
    let m = unsafe { obj::<MatDescr>(d) }?;
    if m.ty != 0 {
        return Err(MATRIX_TYPE_NOT_SUPPORTED);
    }
    Ok(m.base as u64)
}

/// Bytes of csrgeam2's buffer: one u64 count per row.
fn geam_buffer(m: c_int) -> usize {
    (m.max(0) as usize).max(1) * 8
}

unsafe fn geam_buffer_size(m: c_int, n: c_int, size: *mut usize) -> i32 {
    if m < 0 || n < 0 {
        return INVALID_VALUE;
    }
    status(unsafe { put(size, geam_buffer(m)) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseScsrgeam2_bufferSizeExt(_h: *mut c_void, m: c_int, n: c_int, _alpha: *const f32, _da: *const c_void, _nnza: c_int, _va: *const f32, _pa: *const c_int, _ia: *const c_int, _beta: *const f32, _db: *const c_void, _nnzb: c_int, _vb: *const f32, _pb: *const c_int, _ib: *const c_int, _dc: *const c_void, _vc: *const f32, _pc: *const c_int, _ic: *const c_int, size: *mut usize) -> i32 {
    unsafe { geam_buffer_size(m, n, size) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDcsrgeam2_bufferSizeExt(_h: *mut c_void, m: c_int, n: c_int, _alpha: *const f64, _da: *const c_void, _nnza: c_int, _va: *const f64, _pa: *const c_int, _ia: *const c_int, _beta: *const f64, _db: *const c_void, _nnzb: c_int, _vb: *const f64, _pb: *const c_int, _ib: *const c_int, _dc: *const c_void, _vc: *const f64, _pc: *const c_int, _ic: *const c_int, size: *mut usize) -> i32 {
    unsafe { geam_buffer_size(m, n, size) }
}

/// C's row offsets for `alpha A + beta B` and its nnz (`*nnz` on the host or the device per
/// the pointer mode). Synchronizes the stream.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseXcsrgeam2Nnz(h: *mut c_void, m: c_int, n: c_int, da: *const c_void, _nnza: c_int, pa: *const c_int, ia: *const c_int, db: *const c_void, _nnzb: c_int, pb: *const c_int, ib: *const c_int, dc: *const c_void, pc: *mut c_int, nnz: *mut c_int, buffer: *mut c_void) -> i32 {
    status((|| {
        let h = unsafe { handle(h) }?;
        if m < 0 || n < 0 || pc.is_null() || nnz.is_null() {
            return Err(INVALID_VALUE);
        }
        let (ba, bb, bc) = unsafe { (descr_base(da)?, descr_base(db)?, descr_base(dc)?) };
        if m > 0 && (pa.is_null() || pb.is_null() || buffer.is_null()) {
            return Err(INVALID_VALUE);
        }
        let mu = m as u64;
        run_product(h, ptxgen::GEAMCOUNT, false, mu, &[pa as u64, ia as u64, pb as u64, ib as u64, buffer as u64, mu, ba, bb])?;
        let counts = download_u64(h, buffer as u64, m as usize)?;
        let off = offsets(&counts, bc);
        let total = off[m as usize] - bc;
        if total > i32::MAX as u64 {
            return Err(NOT_SUPPORTED);
        }
        upload(pc as u64, &index_bytes(&off, false))?;
        if h.pointer_mode == 1 {
            upload(nnz as u64, &(total as i32).to_le_bytes())?;
        } else {
            unsafe { nnz.write(total as i32) };
        }
        trace!("csrgeam2Nnz m={m} n={n}: nnz={total}");
        Ok(())
    })())
}

#[allow(clippy::too_many_arguments)]
unsafe fn csrgeam2(double: bool, h: *mut c_void, m: c_int, n: c_int, alpha: *const c_void, da: *const c_void, pa: *const c_int, ia: *const c_int, va: *const c_void, beta: *const c_void, db: *const c_void, pb: *const c_int, ib: *const c_int, vb: *const c_void, dc: *const c_void, pc: *const c_int, ic: *mut c_int, vc: *mut c_void) -> i32 {
    status((|| {
        let h = unsafe { handle(h) }?;
        if m < 0 || n < 0 {
            return Err(INVALID_VALUE);
        }
        let (ba, bb, bc) = unsafe { (descr_base(da)?, descr_base(db)?, descr_base(dc)?) };
        let sc = unsafe { scalars(h, double, alpha, beta) }?;
        if m > 0 && (pa.is_null() || pb.is_null() || pc.is_null()) {
            return Err(INVALID_VALUE);
        }
        let mu = m as u64;
        trace!("{}csrgeam2 m={m} n={n}", if double { "D" } else { "S" });
        run_product(h, ptxgen::GEAM, double, mu, &[pa as u64, ia as u64, va as u64, pb as u64, ib as u64, vb as u64, pc as u64, ic as u64, vc as u64, mu, ba, bb, bc, sc.alpha, sc.alpha_ptr, sc.beta, sc.beta_ptr])
    })())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseScsrgeam2(h: *mut c_void, m: c_int, n: c_int, alpha: *const f32, da: *const c_void, _nnza: c_int, va: *const f32, pa: *const c_int, ia: *const c_int, beta: *const f32, db: *const c_void, _nnzb: c_int, vb: *const f32, pb: *const c_int, ib: *const c_int, dc: *const c_void, vc: *mut f32, pc: *mut c_int, ic: *mut c_int, _buffer: *mut c_void) -> i32 {
    unsafe { csrgeam2(false, h, m, n, alpha.cast(), da, pa, ia, va.cast(), beta.cast(), db, pb, ib, vb.cast(), dc, pc, ic, vc.cast()) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cusparseDcsrgeam2(h: *mut c_void, m: c_int, n: c_int, alpha: *const f64, da: *const c_void, _nnza: c_int, va: *const f64, pa: *const c_int, ia: *const c_int, beta: *const f64, db: *const c_void, _nnzb: c_int, vb: *const f64, pb: *const c_int, ib: *const c_int, dc: *const c_void, vc: *mut f64, pc: *mut c_int, ic: *mut c_int, _buffer: *mut c_void) -> i32 {
    unsafe { csrgeam2(true, h, m, n, alpha.cast(), da, pa, ia, va.cast(), beta.cast(), db, pb, ib, vb.cast(), dc, pc, ic, vc.cast()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CStr;
    use std::ptr::{null, null_mut};

    #[test]
    fn names_and_strings() {
        let name = |s| unsafe { CStr::from_ptr(cusparseGetErrorName(s)) }.to_str().unwrap().to_owned();
        assert_eq!(name(0), "CUSPARSE_STATUS_SUCCESS");
        assert_eq!(name(10), "CUSPARSE_STATUS_NOT_SUPPORTED");
        assert_eq!(name(99), "unrecognized status code");
        assert!(!cusparseGetErrorString(7).is_null());
    }

    #[test]
    fn versions() {
        let mut v = 0;
        assert_eq!(unsafe { cusparseGetVersion(null_mut(), &mut v) }, 0);
        assert_eq!(v, 12509);
        assert_eq!(unsafe { cusparseGetVersion(null_mut(), null_mut()) }, CUSPARSE_STATUS_INVALID_VALUE);
        let (mut major, mut minor, mut patch) = (0, 0, 0);
        unsafe {
            cusparseGetProperty(0, &mut major);
            cusparseGetProperty(1, &mut minor);
            cusparseGetProperty(2, &mut patch);
        }
        assert_eq!((major, minor, patch), (12, 5, 9));
    }

    /// Handles, descriptors, getters and every error reported before the driver is entered.
    #[test]
    fn handles_descriptors_and_errors_without_a_device() {
        let mut h = null_mut();
        assert_eq!(unsafe { cusparseCreate(&mut h) }, 0);
        let s = 0x50 as *mut c_void;
        let mut got = null_mut();
        assert_eq!(unsafe { cusparseSetStream(h, s) }, 0);
        assert_eq!(unsafe { cusparseGetStream(h, &mut got) }, 0);
        assert_eq!(got, s);
        let mut mode = 9;
        assert_eq!(unsafe { cusparseGetPointerMode(h, &mut mode) }, 0);
        assert_eq!(mode, 0);
        assert_eq!(unsafe { cusparseSetPointerMode(h, 2) }, INVALID_VALUE);
        assert_eq!(unsafe { cusparseSetStream(null_mut(), s) }, NOT_INITIALIZED);

        // legacy descriptor
        let mut md = null_mut();
        assert_eq!(unsafe { cusparseCreateMatDescr(&mut md) }, 0);
        assert_eq!(unsafe { cusparseSetMatType(md, 3) }, 0);
        assert_eq!(unsafe { cusparseSetMatFillMode(md, 1) }, 0);
        assert_eq!(unsafe { cusparseSetMatDiagType(md, 1) }, 0);
        assert_eq!(unsafe { cusparseSetMatIndexBase(md, 1) }, 0);
        assert_eq!(unsafe { cusparseSetMatIndexBase(md, 2) }, INVALID_VALUE);
        assert_eq!(unsafe { (cusparseGetMatType(md), cusparseGetMatFillMode(md), cusparseGetMatDiagType(md), cusparseGetMatIndexBase(md)) }, (3, 1, 1, 1));

        // CSR descriptor with getters, setters, attributes and batches
        let (p, c, v) = (0x1000 as *mut c_void, 0x2000 as *mut c_void, 0x3000 as *mut c_void);
        let mut a = null_mut();
        assert_eq!(unsafe { cusparseCreateCsr(&mut a, 4, 5, 7, p, c, v, plan::INDEX_64I, plan::INDEX_32I, 0, plan::R_32F) }, 0);
        let (mut r, mut cc, mut nz) = (0, 0, 0);
        assert_eq!(unsafe { cusparseSpMatGetSize(a, &mut r, &mut cc, &mut nz) }, 0);
        assert_eq!((r, cc, nz), (4, 5, 7));
        let mut f = 0;
        assert_eq!(unsafe { cusparseSpMatGetFormat(a, &mut f) }, 0);
        assert_eq!(f, plan::FORMAT_CSR);
        let fill = 1i32;
        assert_eq!(unsafe { cusparseSpMatSetAttribute(a, 0, (&fill as *const i32).cast(), 4) }, 0);
        let mut back = 0i32;
        assert_eq!(unsafe { cusparseSpMatGetAttribute(a, 0, (&mut back as *mut i32).cast(), 4) }, 0);
        assert_eq!(back, 1);
        assert_eq!(unsafe { cusparseSpMatSetAttribute(a, 5, (&fill as *const i32).cast(), 4) }, INVALID_VALUE);
        assert_eq!(unsafe { cusparseCsrSetStridedBatch(a, 3, 5, 7) }, 0);
        let mut bc = 0;
        assert_eq!(unsafe { cusparseSpMatGetStridedBatch(a, &mut bc) }, 0);
        assert_eq!(bc, 3);
        assert_eq!(unsafe { cusparseCsrSetStridedBatch(a, 1, 5, 7) }, 0);
        assert_eq!(unsafe { cusparseCsrSetPointers(a, c, p, v) }, 0);
        let (mut pp, mut ci, mut vv) = (null_mut(), null_mut(), null_mut());
        let (mut pt, mut it, mut base, mut vt) = (0, 0, 9, 9);
        assert_eq!(unsafe { cusparseCsrGet(a, &mut r, &mut cc, &mut nz, &mut pp, &mut ci, &mut vv, &mut pt, &mut it, &mut base, &mut vt) }, 0);
        assert_eq!((pp, ci, vv, pt, it, base, vt), (c, p, v, plan::INDEX_64I, plan::INDEX_32I, 0, plan::R_32F));
        assert_eq!(unsafe { cusparseCooSetPointers(a, c, p, v) }, INVALID_VALUE, "format mismatch");
        assert_eq!(unsafe { cusparseCreateCsr(&mut got, -1, 5, 7, p, c, v, 2, 2, 0, 0) }, INVALID_VALUE);
        assert_eq!(unsafe { cusparseCreateCsr(&mut got, 1, 5, 7, p, c, v, 9, 2, 0, 0) }, INVALID_VALUE);

        // dense descriptors
        let (mut x, mut y, mut bm, mut cm) = (null_mut(), null_mut(), null_mut(), null_mut());
        assert_eq!(unsafe { cusparseCreateDnVec(&mut x, 5, v, plan::R_32F) }, 0);
        assert_eq!(unsafe { cusparseCreateConstDnVec(&mut y, 4, v, plan::R_32F) }, 0);
        assert_eq!(unsafe { cusparseCreateDnMat(&mut bm, 5, 3, 5, v, plan::R_32F, plan::ORDER_COL) }, 0);
        assert_eq!(unsafe { cusparseCreateConstDnMat(&mut cm, 4, 3, 3, v, plan::R_32F, plan::ORDER_ROW) }, 0);
        assert_eq!(unsafe { cusparseCreateDnMat(&mut got, 5, 3, 4, v, plan::R_32F, plan::ORDER_COL) }, INVALID_VALUE, "ld < rows");
        assert_eq!(unsafe { cusparseDnMatSetStridedBatch(cm, 3, 12) }, 0);
        let (mut n, mut st) = (0, 0i64);
        assert_eq!(unsafe { cusparseDnMatGetStridedBatch(cm, &mut n, &mut st) }, 0);
        assert_eq!((n, st), (3, 12));
        let (mut sz, mut vals, mut ty) = (0i64, null_mut(), 9);
        assert_eq!(unsafe { cusparseDnVecGet(x, &mut sz, &mut vals, &mut ty) }, 0);
        assert_eq!((sz, vals, ty), (5, v, 0));

        // buffer sizes validate shapes and types without a device
        let one = 1.0f32;
        let alpha: *const c_void = (&one as *const f32).cast();
        let mut ws = 99usize;
        assert_eq!(unsafe { cusparseSpMV_bufferSize(h, 0, alpha, a, x, alpha, y, plan::R_32F, 0, &mut ws) }, 0);
        assert_eq!(ws, 0);
        assert_eq!(unsafe { cusparseSpMV_bufferSize(h, 1, alpha, a, x, alpha, y, plan::R_32F, 0, &mut ws) }, INVALID_VALUE, "op T needs x of 4");
        assert_eq!(unsafe { cusparseSpMV_bufferSize(h, 0, alpha, a, x, alpha, y, plan::R_64F, 0, &mut ws) }, NOT_SUPPORTED, "mixed compute type");
        assert_eq!(unsafe { cusparseSpMM_bufferSize(h, 0, 0, alpha, a, bm, alpha, cm, plan::R_32F, 0, &mut ws) }, 0);
        assert_eq!(unsafe { cusparseSpMM_preprocess(h, 0, 0, alpha, a, bm, alpha, cm, plan::R_32F, 0, null_mut()) }, 0);
        assert_eq!(unsafe { cusparseSpMM_bufferSize(h, 0, 1, alpha, a, bm, alpha, cm, plan::R_32F, 0, &mut ws) }, INVALID_VALUE);
        assert_eq!(unsafe { cusparseSpMM_bufferSize(h, 0, 0, null(), a, bm, alpha, cm, plan::R_32F, 0, &mut ws) }, INVALID_VALUE);
        assert_eq!(unsafe { cusparseSpMM_bufferSize(null_mut(), 0, 0, alpha, a, bm, alpha, cm, plan::R_32F, 0, &mut ws) }, NOT_INITIALIZED);
        // half precision and complex values: not supported
        let mut hm = null_mut();
        assert_eq!(unsafe { cusparseCreateCoo(&mut hm, 4, 5, 7, p, c, v, plan::INDEX_32I, 0, 2) }, 0);
        assert_eq!(unsafe { cusparseSpMM_bufferSize(h, 0, 0, alpha, hm, bm, alpha, cm, 2, 0, &mut ws) }, NOT_SUPPORTED);
        assert_eq!(unsafe { cusparseDestroySpMat(hm) }, 0);
        // SDDMM shapes: C (CSR 4x5) = A (4x5) * B... use A 4x3 row, B 3x5
        let (mut da, mut db) = (null_mut(), null_mut());
        assert_eq!(unsafe { cusparseCreateDnMat(&mut da, 4, 3, 3, v, plan::R_32F, plan::ORDER_ROW) }, 0);
        assert_eq!(unsafe { cusparseCreateDnMat(&mut db, 3, 5, 3, v, plan::R_32F, plan::ORDER_COL) }, 0);
        assert_eq!(unsafe { cusparseSDDMM_bufferSize(h, 0, 0, alpha, da, db, alpha, a, plan::R_32F, 0, &mut ws) }, 0);
        assert_eq!(unsafe { cusparseSDDMM_bufferSize(h, 1, 0, alpha, da, db, alpha, a, plan::R_32F, 0, &mut ws) }, INVALID_VALUE);

        // a batched A needs a C with the same batch count
        assert_eq!(unsafe { cusparseCsrSetStridedBatch(a, 2, 5, 7) }, 0);
        assert_eq!(unsafe { cusparseSpMV_bufferSize(h, 0, alpha, a, x, alpha, y, plan::R_32F, 0, &mut ws) }, INVALID_VALUE);
        for d in [x, y] {
            assert_eq!(unsafe { cusparseDestroyDnVec(d) }, 0);
        }
        for d in [bm, cm, da, db] {
            assert_eq!(unsafe { cusparseDestroyDnMat(d) }, 0);
        }
        assert_eq!(unsafe { cusparseDestroySpMat(a) }, 0);
        assert_eq!(unsafe { cusparseDestroyMatDescr(md) }, 0);
        assert_eq!(unsafe { cusparseXcoo2csr(null_mut(), null(), 0, 0, null_mut(), 0) }, NOT_INITIALIZED);
        assert_eq!(unsafe { cusparseXcoo2csr(h, null(), -1, 0, null_mut(), 0) }, INVALID_VALUE);
        assert_eq!(unsafe { cusparseCreateIdentityPermutation(h, 0, null_mut()) }, 0, "empty: nothing launched");
        assert_eq!(unsafe { cusparseDestroy(h) }, 0);
        assert_eq!(unsafe { cusparseDestroy(null_mut()) }, NOT_INITIALIZED);
        assert_eq!(unsafe { cusparseCreate(null_mut()) }, INVALID_VALUE);
    }

    #[test]
    fn everything_else_reports_not_supported() {
        assert_eq!(stubs::cusparseSpGEMMreuse_compute(), NOT_SUPPORTED);
        assert_eq!(stubs::cusparseCcsrgeam2(), NOT_SUPPORTED);
        assert_eq!(stubs::cusparseCbsrmm(), NOT_SUPPORTED);
    }

    #[test]
    fn exported_surface_covers_zluda_and_the_generic_core() {
        let src = include_str!("lib.rs");
        // ZLUDA's 16 implemented cuSPARSE entry points (zluda_sparse/src/lib.rs at f4df215)
        let zluda = [
            "cusparseGetErrorName", "cusparseGetErrorString", "cusparseGetMatDiagType", "cusparseGetMatFillMode",
            "cusparseGetMatIndexBase", "cusparseGetMatType", "cusparseCreate", "cusparseCreateConstCoo",
            "cusparseCreateConstDnMat", "cusparseCreateDnMat", "cusparseDestroy", "cusparseDestroyDnMat",
            "cusparseDestroySpMat", "cusparseSpMM_bufferSize", "cusparseSpMM_preprocess", "cusparseSpMM",
        ];
        let core = [
            "cusparseSetStream", "cusparseCreateCsr", "cusparseCreateCoo", "cusparseCreateDnVec", "cusparseDestroyDnVec",
            "cusparseSpMV", "cusparseSpMV_bufferSize", "cusparseSetPointerMode", "cusparseCreateMatDescr",
            "cusparseXcoo2csr", "cusparseSDDMM", "cusparseSbsrmm", "cusparseDbsrmv",
        ];
        for name in zluda.iter().chain(core.iter()) {
            assert!(src.contains(&format!("extern \"system\" fn {name}(")), "{name}");
        }
    }
}
