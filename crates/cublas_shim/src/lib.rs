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
    cuMemcpyDtoH_v2: unsafe extern "system" fn(*mut c_void, u64, usize) -> CuResult,
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
            cuMemcpyDtoH_v2: sym!("cuMemcpyDtoH_v2"),
            _lib: lib,
        })
    }
}

struct GemmModule {
    _module: CUmodule,
    tn: CUfunction,
}

// CUDA driver handles are process-wide opaque pointers; guarded by OnceLock init.
unsafe impl Send for GemmModule {}
unsafe impl Sync for GemmModule {}

fn sgemm_module() -> Result<&'static GemmModule, c_int> {
    static MODULE: OnceLock<Result<GemmModule, c_int>> = OnceLock::new();
    match MODULE.get_or_init(|| load_gemm_module("sgemm_tn.ptx", include_str!("sgemm_tn.ptx"), c"cfi_sgemm_tn")) {
        Ok(m) => Ok(m),
        Err(code) => Err(*code),
    }
}

fn hgemm_module() -> Result<&'static GemmModule, c_int> {
    static MODULE: OnceLock<Result<GemmModule, c_int>> = OnceLock::new();
    match MODULE.get_or_init(|| load_gemm_module("hgemm_tn.ptx", include_str!("hgemm_tn.ptx"), c"cfi_hgemm_tn")) {
        Ok(m) => Ok(m),
        Err(code) => Err(*code),
    }
}

fn load_gemm_module(
    label: &str,
    ptx_src: &'static str,
    entry: &'static std::ffi::CStr,
) -> Result<GemmModule, c_int> {
    let drv = driver()?;
    let ptx = format!("{ptx_src}\0");
    let mut module: CUmodule = std::ptr::null_mut();
    let err = unsafe { (drv.cuModuleLoadData)(&mut module, ptx.as_ptr().cast()) };
    if err != SUCCESS {
        eprintln!("CUDA for Intel cublas: ModuleLoadData {label} failed err={err}");
        return Err(STATUS_MAPPING_ERROR);
    }
    let mut tn: CUfunction = std::ptr::null_mut();
    let err = unsafe { (drv.cuModuleGetFunction)(&mut tn, module, entry.as_ptr()) };
    if err != SUCCESS || tn.is_null() {
        eprintln!(
            "CUDA for Intel cublas: GetFunction {} failed err={err}",
            entry.to_string_lossy()
        );
        return Err(STATUS_MAPPING_ERROR);
    }
    Ok(GemmModule {
        _module: module,
        tn,
    })
}

/// IEEE754 binary16 → f32 (host-side alpha/beta for CUBLAS_COMPUTE_16F).
fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            // subnormal
            let mut m = mant;
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
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
    // cudaDataType_t / cublasComputeType_t
    const CUDA_R_32F: c_int = 0;
    const CUDA_R_16F: c_int = 2;
    const CUBLAS_COMPUTE_16F: c_int = 64;
    const CUBLAS_COMPUTE_32F: c_int = 68;
    const CUBLAS_COMPUTE_32F_FAST_16F: c_int = 74;
    if a_type == CUDA_R_32F
        && b_type == CUDA_R_32F
        && c_type == CUDA_R_32F
        && (compute_type == CUDA_R_32F || compute_type == CUBLAS_COMPUTE_32F || compute_type == 67)
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
    if a_type == CUDA_R_16F
        && b_type == CUDA_R_16F
        && c_type == CUDA_R_16F
        && (compute_type == CUBLAS_COMPUTE_16F
            || compute_type == CUBLAS_COMPUTE_32F
            || compute_type == CUBLAS_COMPUTE_32F_FAST_16F
            || compute_type == CUDA_R_16F)
    {
        return unsafe { cublasHgemm_tn(handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc, compute_type) };
    }
    eprintln!(
        "CUDA for Intel cublasGemmEx unsupported types a={a_type} b={b_type} c={c_type} compute={compute_type}"
    );
    NOT_SUPPORTED
}

unsafe fn cublasHgemm_tn(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const c_void,
    lda: c_int,
    b: *const c_void,
    ldb: c_int,
    beta: *const c_void,
    c: *mut c_void,
    ldc: c_int,
    compute_type: c_int,
) -> c_int {
    if handle.is_null() || alpha.is_null() || beta.is_null() || a.is_null() || b.is_null() || c.is_null()
    {
        return INVALID_VALUE;
    }
    if m < 0 || n < 0 || k < 0 || lda < 1 || ldb < 1 || ldc < 1 {
        return INVALID_VALUE;
    }
    if transa != CUBLAS_OP_T || transb != CUBLAS_OP_N {
        eprintln!(
            "CUDA for Intel cublasHgemm: unsupported transa={transa} transb={transb} (need T,N)"
        );
        return NOT_SUPPORTED;
    }
    let Ok(h) = handle_mut(handle) else {
        return INVALID_VALUE;
    };
    let Ok(module) = hgemm_module() else {
        return STATUS_MAPPING_ERROR;
    };
    let Ok(drv) = driver() else {
        return NOT_INITIALIZED;
    };

    // CUBLAS_COMPUTE_16F passes half alpha/beta; 32F compute modes pass float.
    const CUBLAS_COMPUTE_16F: c_int = 64;
    let (alpha_v, beta_v) = if compute_type == CUBLAS_COMPUTE_16F || compute_type == 2 {
        let a_bits = unsafe { alpha.cast::<u16>().read() };
        let b_bits = unsafe { beta.cast::<u16>().read() };
        (f16_bits_to_f32(a_bits), f16_bits_to_f32(b_bits))
    } else {
        (unsafe { alpha.cast::<f32>().read() }, unsafe { beta.cast::<f32>().read() })
    };

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
        "CUDA for Intel cublasHgemm_tn m={m} n={n} k={k} lda={lda} ldb={ldb} ldc={ldc} alpha={alpha_v} beta={beta_v} grid=({grid_x},{grid_y})"
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
        eprintln!("CUDA for Intel cublasHgemm launch failed err={err}");
        return STATUS_MAPPING_ERROR;
    }
    let sync_stream = if h.stream.is_null() {
        std::ptr::null_mut()
    } else {
        h.stream
    };
    let err = unsafe { (drv.cuStreamSynchronize)(sync_stream) };
    if err != SUCCESS {
        eprintln!("CUDA for Intel cublasHgemm sync failed err={err}");
        return STATUS_MAPPING_ERROR;
    }
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGemmStridedBatchedEx(
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
    stride_a: i64,
    b: *const c_void,
    b_type: c_int,
    ldb: c_int,
    stride_b: i64,
    beta: *const c_void,
    c: *mut c_void,
    c_type: c_int,
    ldc: c_int,
    stride_c: i64,
    batch: c_int,
    compute_type: c_int,
    algo: c_int,
) -> c_int {
    // Soft-emulate via GemmEx loop. Strides are in *elements* of the matrix type.
    const CUDA_R_32F: c_int = 0;
    const CUDA_R_16F: c_int = 2;
    if batch <= 0 {
        return INVALID_VALUE;
    }
    let a_elem = match a_type {
        CUDA_R_32F => 4usize,
        CUDA_R_16F => 2usize,
        _ => {
            eprintln!(
                "CUDA for Intel cublasGemmStridedBatchedEx unsupported a_type={a_type}"
            );
            return NOT_SUPPORTED;
        }
    };
    let b_elem = match b_type {
        CUDA_R_32F => 4usize,
        CUDA_R_16F => 2usize,
        _ => return NOT_SUPPORTED,
    };
    let c_elem = match c_type {
        CUDA_R_32F => 4usize,
        CUDA_R_16F => 2usize,
        _ => return NOT_SUPPORTED,
    };
    eprintln!(
        "CUDA for Intel cublasGemmStridedBatchedEx batch={batch} m={m} n={n} k={k} lda={lda} ldb={ldb} ldc={ldc} strideA={stride_a} strideB={stride_b} strideC={stride_c} a_type={a_type} compute={compute_type}"
    );
    for i in 0..batch as i64 {
        let a_i = unsafe { (a as *const u8).add((i * stride_a) as usize * a_elem) as *const c_void };
        let b_i = unsafe { (b as *const u8).add((i * stride_b) as usize * b_elem) as *const c_void };
        let c_i = unsafe { (c as *mut u8).add((i * stride_c) as usize * c_elem) as *mut c_void };
        let status = unsafe {
            cublasGemmEx(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                a_i,
                a_type,
                lda,
                b_i,
                b_type,
                ldb,
                beta,
                c_i,
                c_type,
                ldc,
                compute_type,
                algo,
            )
        };
        if status != SUCCESS {
            eprintln!(
                "CUDA for Intel cublasGemmStridedBatchedEx batch {i}/{batch} failed status={status}"
            );
            return status;
        }
    }
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGemmBatchedEx(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: *const *const c_void,
    a_type: c_int,
    lda: c_int,
    b: *const *const c_void,
    b_type: c_int,
    ldb: c_int,
    beta: *const c_void,
    c: *const *mut c_void,
    c_type: c_int,
    ldc: c_int,
    batch: c_int,
    compute_type: c_int,
    _algo: c_int,
) -> c_int {
    // GGML builds device-side pointer tables (`k_compute_batched_ptrs`) on the
    // cuBLAS stream, then passes the *device* address of those tables cast to
    // `const void**`. Pull them to the host and loop over GemmEx.
    //
    // Critical: sync the handle stream *before* the host DtoH. Synchronous
    // `cuMemcpyDtoH_v2` only waits for legacy/blocking streams; ggml often uses
    // a non-null (possibly non-blocking) stream, so without an explicit
    // `cuStreamSynchronize` we can read stale/zero pointer tables and GEMM the
    // wrong buffers — which produces fluent-looking garbage tokens.
    if batch <= 0 {
        return INVALID_VALUE;
    }
    let Ok(drv) = driver() else {
        return NOT_INITIALIZED;
    };
    let Ok(h) = handle_mut(handle) else {
        return INVALID_VALUE;
    };
    let sync_stream = if h.stream.is_null() {
        std::ptr::null_mut()
    } else {
        h.stream
    };
    let err = unsafe { (drv.cuStreamSynchronize)(sync_stream) };
    if err != SUCCESS {
        eprintln!(
            "CUDA for Intel cublasGemmBatchedEx pre-DtoH stream sync failed err={err}"
        );
        return STATUS_MAPPING_ERROR;
    }
    let batch_n = batch as usize;
    let bytes = batch_n * std::mem::size_of::<u64>();
    let mut a_ptrs = vec![0u64; batch_n];
    let mut b_ptrs = vec![0u64; batch_n];
    let mut c_ptrs = vec![0u64; batch_n];
    let err = unsafe {
        (drv.cuMemcpyDtoH_v2)(a_ptrs.as_mut_ptr().cast(), a as u64, bytes)
    };
    if err != SUCCESS {
        eprintln!("CUDA for Intel cublasGemmBatchedEx DtoH A failed err={err}");
        return STATUS_MAPPING_ERROR;
    }
    let err = unsafe {
        (drv.cuMemcpyDtoH_v2)(b_ptrs.as_mut_ptr().cast(), b as u64, bytes)
    };
    if err != SUCCESS {
        eprintln!("CUDA for Intel cublasGemmBatchedEx DtoH B failed err={err}");
        return STATUS_MAPPING_ERROR;
    }
    let err = unsafe {
        (drv.cuMemcpyDtoH_v2)(c_ptrs.as_mut_ptr().cast(), c as u64, bytes)
    };
    if err != SUCCESS {
        eprintln!("CUDA for Intel cublasGemmBatchedEx DtoH C failed err={err}");
        return STATUS_MAPPING_ERROR;
    }
    // Cheap sanity: GGML never passes null matrix pointers into BatchedEx.
    if a_ptrs.iter().any(|&p| p == 0) || b_ptrs.iter().any(|&p| p == 0) || c_ptrs.iter().any(|&p| p == 0)
    {
        eprintln!(
            "CUDA for Intel cublasGemmBatchedEx null ptr in tables A={a_ptrs:?} B={b_ptrs:?} C={c_ptrs:?}"
        );
        return INVALID_VALUE;
    }
    eprintln!(
        "CUDA for Intel cublasGemmBatchedEx batch={batch} m={m} n={n} k={k} lda={lda} ldb={ldb} ldc={ldc} transa={transa} transb={transb} a_type={a_type} c_type={c_type} compute={compute_type} a0={:#x} b0={:#x} c0={:#x}",
        a_ptrs[0], b_ptrs[0], c_ptrs[0]
    );
    for i in 0..batch_n {
        let status = unsafe {
            cublasGemmEx(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha,
                a_ptrs[i] as *const c_void,
                a_type,
                lda,
                b_ptrs[i] as *const c_void,
                b_type,
                ldb,
                beta,
                c_ptrs[i] as *mut c_void,
                c_type,
                ldc,
                compute_type,
                0,
            )
        };
        if status != SUCCESS {
            eprintln!("CUDA for Intel cublasGemmBatchedEx batch {i}/{batch} failed status={status}");
            return status;
        }
    }
    SUCCESS
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
