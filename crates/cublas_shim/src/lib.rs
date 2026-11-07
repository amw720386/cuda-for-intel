//! Minimal cuBLAS shim for ggml-cuda.dll on cuda-for-intel.
//!
//! NVIDIA's `cublas64_12.dll` refuses to initialize without an NVIDIA kernel
//! driver. This DLL implements the subset ggml imports, backing SGEMM with a
//! real GPU kernel launched through our Driver API `nvcuda.dll`.
#![allow(non_snake_case)]

use libloading::Library;
use std::ffi::{c_char, c_float, c_int, c_void};
use std::sync::OnceLock;

const SUCCESS: c_int = 0;
const NOT_INITIALIZED: c_int = 1;
const ALLOC_FAILED: c_int = 3;
const INVALID_VALUE: c_int = 7;
const NOT_SUPPORTED: c_int = 14;
const STATUS_MAPPING_ERROR: c_int = 11;

type CuResult = c_int;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;

const CUBLAS_OP_N: c_int = 0;
const CUBLAS_OP_T: c_int = 1;
const CUBLAS_OP_C: c_int = 2;

struct Driver {
    _lib: Library,
    cuModuleLoadData: unsafe extern "system" fn(*mut CUmodule, *const c_void) -> CuResult,
    cuModuleGetFunction:
        unsafe extern "system" fn(*mut CUfunction, CUmodule, *const c_char) -> CuResult,
    cuLaunchKernel: unsafe extern "system" fn(
        CUfunction,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        u32,
        CUstream,
        *mut *mut c_void,
        *mut *mut c_void,
    ) -> CuResult,
    cuStreamSynchronize: unsafe extern "system" fn(CUstream) -> CuResult,
}

fn driver() -> Result<&'static Driver, c_int> {
    static DRIVER: OnceLock<Result<Driver, c_int>> = OnceLock::new();
    match DRIVER.get_or_init(load_driver) {
        Ok(d) => Ok(d),
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
        Ok(Driver {
            cuModuleLoadData: sym!("cuModuleLoadData"),
            cuModuleGetFunction: sym!("cuModuleGetFunction"),
            cuLaunchKernel: sym!("cuLaunchKernel"),
            cuStreamSynchronize: sym!("cuStreamSynchronize"),
            _lib: lib,
        })
    }
}

struct SgemmModule {
    _module: CUmodule,
    tn: CUfunction,
}

// CUDA driver handles are process-wide opaque pointers; guarded by OnceLock init.
unsafe impl Send for SgemmModule {}
unsafe impl Sync for SgemmModule {}

fn sgemm_module() -> Result<&'static SgemmModule, c_int> {
    static MODULE: OnceLock<Result<SgemmModule, c_int>> = OnceLock::new();
    match MODULE.get_or_init(load_sgemm) {
        Ok(m) => Ok(m),
        Err(code) => Err(*code),
    }
}

fn load_sgemm() -> Result<SgemmModule, c_int> {
    let drv = driver()?;
    let ptx = concat!(include_str!("sgemm_tn.ptx"), "\0");
    let mut module: CUmodule = std::ptr::null_mut();
    let err = unsafe { (drv.cuModuleLoadData)(&mut module, ptx.as_ptr().cast()) };
    if err != SUCCESS {
        eprintln!("CUDA for Intel cublas: ModuleLoadData failed err={err}");
        return Err(STATUS_MAPPING_ERROR);
    }
    let mut tn: CUfunction = std::ptr::null_mut();
    let err = unsafe {
        (drv.cuModuleGetFunction)(&mut tn, module, c"cfi_sgemm_tn".as_ptr())
    };
    if err != SUCCESS || tn.is_null() {
        eprintln!("CUDA for Intel cublas: GetFunction cfi_sgemm_tn failed err={err}");
        return Err(STATUS_MAPPING_ERROR);
    }
    Ok(SgemmModule {
        _module: module,
        tn,
    })
}

struct Handle {
    stream: CUstream,
}

fn handle_mut(handle: *mut c_void) -> Result<&'static mut Handle, c_int> {
    if handle.is_null() {
        return Err(INVALID_VALUE);
    }
    Ok(unsafe { &mut *(handle as *mut Handle) })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCreate_v2(handle: *mut *mut c_void) -> c_int {
    if handle.is_null() {
        return INVALID_VALUE;
    }
    let Ok(_) = driver() else {
        return NOT_INITIALIZED;
    };
    let ptr = Box::into_raw(Box::new(Handle {
        stream: std::ptr::null_mut(),
    })) as *mut c_void;
    unsafe { handle.write(ptr) };
    eprintln!("CUDA for Intel cublasCreate_v2 ok");
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDestroy_v2(handle: *mut c_void) -> c_int {
    if handle.is_null() {
        return SUCCESS;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut Handle));
    }
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetStream_v2(handle: *mut c_void, stream: CUstream) -> c_int {
    let Ok(h) = handle_mut(handle) else {
        return INVALID_VALUE;
    };
    h.stream = stream;
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetMathMode(_handle: *mut c_void, _mode: c_int) -> c_int {
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetWorkspace_v2(
    _handle: *mut c_void,
    _workspace: *mut c_void,
    _size: usize,
) -> c_int {
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetStatusString(status: c_int) -> *const c_char {
    let s: &'static [u8] = match status {
        SUCCESS => b"CUBLAS_STATUS_SUCCESS\0",
        NOT_INITIALIZED => b"CUBLAS_STATUS_NOT_INITIALIZED\0",
        ALLOC_FAILED => b"CUBLAS_STATUS_ALLOC_FAILED\0",
        INVALID_VALUE => b"CUBLAS_STATUS_INVALID_VALUE\0",
        NOT_SUPPORTED => b"CUBLAS_STATUS_NOT_SUPPORTED\0",
        _ => b"CUBLAS_STATUS_INTERNAL_ERROR\0",
    };
    s.as_ptr().cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSgemm_v2(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: c_int,
    b: *const c_float,
    ldb: c_int,
    beta: *const c_float,
    c: *mut c_float,
    ldc: c_int,
) -> c_int {
    if handle.is_null() || alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null()
    {
        return INVALID_VALUE;
    }
    if m < 0 || n < 0 || k < 0 || lda < 1 || ldb < 1 || ldc < 1 {
        return INVALID_VALUE;
    }
    // GGML F32 mul_mat uses OP_T x OP_N exclusively on the hot path.
    if transa != CUBLAS_OP_T || transb != CUBLAS_OP_N {
        eprintln!(
            "CUDA for Intel cublasSgemm: unsupported transa={transa} transb={transb} (need T,N)"
        );
        return NOT_SUPPORTED;
    }
    let Ok(h) = handle_mut(handle) else {
        return INVALID_VALUE;
    };
    let Ok(module) = sgemm_module() else {
        return STATUS_MAPPING_ERROR;
    };
    let Ok(drv) = driver() else {
        return NOT_INITIALIZED;
    };

    let alpha_v = unsafe { alpha.read() };
    let beta_v = unsafe { beta.read() };
    let mut a_ptr = a as u64;
    let mut b_ptr = b as u64;
    let mut c_ptr = c as u64;
    let mut alpha_mut = alpha_v;
    let mut beta_mut = beta_v;
    let mut m_u = m as u32;
    let mut n_u = n as u32;
    let mut k_u = k as u32;
    let mut lda_u = lda as u32;
    let mut ldb_u = ldb as u32;
    let mut ldc_u = ldc as u32;

    let mut args: [*mut c_void; 11] = [
        (&mut a_ptr as *mut u64).cast(),
        (&mut b_ptr as *mut u64).cast(),
        (&mut c_ptr as *mut u64).cast(),
        (&mut alpha_mut as *mut f32).cast(),
        (&mut beta_mut as *mut f32).cast(),
        (&mut m_u as *mut u32).cast(),
        (&mut n_u as *mut u32).cast(),
        (&mut k_u as *mut u32).cast(),
        (&mut lda_u as *mut u32).cast(),
        (&mut ldb_u as *mut u32).cast(),
        (&mut ldc_u as *mut u32).cast(),
    ];

    const TILE: u32 = 16;
    let grid_x = m_u.div_ceil(TILE).max(1);
    let grid_y = n_u.div_ceil(TILE).max(1);
    eprintln!(
        "CUDA for Intel cublasSgemm_v2 TN m={m} n={n} k={k} lda={lda} ldb={ldb} ldc={ldc} alpha={alpha_v} beta={beta_v} grid=({grid_x},{grid_y})"
    );
    let err = unsafe {
        (drv.cuLaunchKernel)(
            module.tn,
            grid_x,
            grid_y,
            1,
            TILE,
            TILE,
            1,
            0,
            h.stream,
            args.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    };
    if err != SUCCESS {
        eprintln!("CUDA for Intel cublasSgemm launch failed err={err}");
        return STATUS_MAPPING_ERROR;
    }
    // Match NVIDIA cuBLAS synchronous-enough behavior for the default stream
    // when ggml follows with DtoH; explicit sync keeps probes deterministic.
    let sync_stream = if h.stream.is_null() {
        // Legacy default stream — cuStreamSynchronize(null) is accepted by our driver.
        std::ptr::null_mut()
    } else {
        h.stream
    };
    let err = unsafe { (drv.cuStreamSynchronize)(sync_stream) };
    if err != SUCCESS {
        eprintln!("CUDA for Intel cublasSgemm sync failed err={err}");
        return STATUS_MAPPING_ERROR;
    }
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGemmEx(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    a_type: c_int,
    lda: c_int,
    b: *const c_void,
    b_type: c_int,
    ldb: c_int,
    beta: *const c_void,
    c: *mut c_void,
    c_type: c_int,
    ldc: c_int,
    compute_type: c_int,
    _algo: c_int,
) -> c_int {
    // CUDA_R_32F == 0
    const CUDA_R_32F: c_int = 0;
    if a_type == CUDA_R_32F
        && b_type == CUDA_R_32F
        && c_type == CUDA_R_32F
        && (compute_type == CUDA_R_32F || compute_type == 68 || compute_type == 67)
    {
        return unsafe {
            cublasSgemm_v2(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha.cast(),
                a.cast(),
                lda,
                b.cast(),
                ldb,
                beta.cast(),
                c.cast(),
                ldc,
            )
        };
    }
    eprintln!(
        "CUDA for Intel cublasGemmEx unsupported types a={a_type} b={b_type} c={c_type} compute={compute_type}"
    );
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGemmStridedBatchedEx(
    _handle: *mut c_void,
    _transa: c_int,
    _transb: c_int,
    _m: c_int,
    _n: c_int,
    _k: c_int,
    _alpha: *const c_void,
    _a: *const c_void,
    _a_type: c_int,
    _lda: c_int,
    _stride_a: i64,
    _b: *const c_void,
    _b_type: c_int,
    _ldb: c_int,
    _stride_b: i64,
    _beta: *const c_void,
    _c: *mut c_void,
    _c_type: c_int,
    _ldc: c_int,
    _stride_c: i64,
    _batch: c_int,
    _compute_type: c_int,
    _algo: c_int,
) -> c_int {
    eprintln!("CUDA for Intel cublasGemmStridedBatchedEx not implemented yet");
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGemmBatchedEx(
    _handle: *mut c_void,
    _transa: c_int,
    _transb: c_int,
    _m: c_int,
    _n: c_int,
    _k: c_int,
    _alpha: *const c_void,
    _a: *const *const c_void,
    _a_type: c_int,
    _lda: c_int,
    _b: *const *const c_void,
    _b_type: c_int,
    _ldb: c_int,
    _beta: *const c_void,
    _c: *const *mut c_void,
    _c_type: c_int,
    _ldc: c_int,
    _batch: c_int,
    _compute_type: c_int,
    _algo: c_int,
) -> c_int {
    eprintln!("CUDA for Intel cublasGemmBatchedEx not implemented yet");
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSgemmBatched(
    _handle: *mut c_void,
    _transa: c_int,
    _transb: c_int,
    _m: c_int,
    _n: c_int,
    _k: c_int,
    _alpha: *const c_float,
    _a: *const *const c_float,
    _lda: c_int,
    _b: *const *const c_float,
    _ldb: c_int,
    _beta: *const c_float,
    _c: *const *mut c_float,
    _ldc: c_int,
    _batch: c_int,
) -> c_int {
    eprintln!("CUDA for Intel cublasSgemmBatched not implemented yet");
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSgemmStridedBatched(
    _handle: *mut c_void,
    _transa: c_int,
    _transb: c_int,
    _m: c_int,
    _n: c_int,
    _k: c_int,
    _alpha: *const c_float,
    _a: *const c_float,
    _lda: c_int,
    _stride_a: i64,
    _b: *const c_float,
    _ldb: c_int,
    _stride_b: i64,
    _beta: *const c_float,
    _c: *mut c_float,
    _ldc: c_int,
    _stride_c: i64,
    _batch: c_int,
) -> c_int {
    eprintln!("CUDA for Intel cublasSgemmStridedBatched not implemented yet");
    NOT_SUPPORTED
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasStrsmBatched(
    _handle: *mut c_void,
    _side: c_int,
    _uplo: c_int,
    _trans: c_int,
    _diag: c_int,
    _m: c_int,
    _n: c_int,
    _alpha: *const c_float,
    _a: *const *const c_float,
    _lda: c_int,
    _b: *const *mut c_float,
    _ldb: c_int,
    _batch: c_int,
) -> c_int {
    eprintln!("CUDA for Intel cublasStrsmBatched not implemented yet");
    NOT_SUPPORTED
}

// Silence unused op constants for documentation of the ABI surface.
#[allow(dead_code)]
fn _op_docs() {
    let _ = (CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C);
}
