//! cuBLAS shim for ggml-cuda (stock Ollama v0.12.6 / llama.cpp) on cuda-for-intel.
//!
//! NVIDIA's `cublas64_*.dll` refuses to initialize without an NVIDIA kernel
//! driver. This library exports the cuBLAS entry points ggml-cuda imports and
//! runs every GEMM on the GPU through our Driver API (`nvcuda.dll`):
//!
//! * `plan`   - pure parameter handling: enum decoding, leading-dimension
//!   validation, op/stride derivation, batch chunking, CPU reference GEMM.
//! * `ptxgen` - generator for tiled shared-memory GEMM kernels in PTX, which
//!   the Driver API translates to SPIR-V. One op-agnostic kernel per
//!   (A, B, C) element type in {f32, f16, bf16} and per batching mode
//!   (strided or device pointer arrays), f32 accumulation.
//! * this file - the exported C ABI, handle state, kernel cache and launcher,
//!   plus an optional oneMKL/oneMath hook (see [`onemkl`]).
//!
//! Entry points ggml-cuda actually calls (llama.cpp `ggml-cuda.cu`,
//! `mmvq`-free dense paths): `cublasCreate_v2`, `cublasDestroy_v2`,
//! `cublasSetStream_v2`, `cublasSetMathMode`, `cublasSetWorkspace_v2`,
//! `cublasGetStatusString`, `cublasSgemm_v2` (f32 mul_mat), `cublasGemmEx`
//! (f16/bf16 inputs with f16/bf16/f32 output, compute 32F or 16F),
//! `cublasGemmStridedBatchedEx` (contiguous batches, attention) and
//! `cublasGemmBatchedEx` (broadcast batches via device pointer tables),
//! `cublasStrsmBatched` (solve_tri; a GPU kernel). cublasLt is not used by ggml-cuda but the
//! `lt` module exports the cublasLt surface PyTorch's `nn.Linear` needs; build this
//! crate's DLL and also install it under the name `cublasLt64_12` (see docs/cublaslt-abi-todo.md).
//!
//! Stream ordering: kernels are enqueued on the handle's stream. By default the
//! shim then synchronizes that stream (the behaviour verified so far); set
//! `CFI_CUBLAS_ASYNC=1` for genuine cuBLAS-style asynchronous enqueue.
//! `CFI_CUBLAS_TRACE=1` logs each call.
#![allow(non_snake_case)]

#[allow(dead_code)]
mod plan;
#[allow(dead_code)]
mod ptxgen;
#[allow(dead_code)]
mod ltplan;
mod lt;
#[allow(dead_code)]
mod xmx;
#[allow(dead_code)]
mod xmx_host;
#[cfg(test)]
mod xmx_interp;
mod stubs;

use libloading::Library;
use plan::*;
use std::collections::HashMap;
use std::ffi::{c_char, c_float, c_int, c_void};
use std::sync::{Mutex, OnceLock};

const INTERNAL_ERROR: c_int = 14;

type CuResult = c_int;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUcontext = *mut c_void;

fn env_flag(name: &str) -> bool {
    std::env::var(name).map(|v| !v.is_empty() && v != "0").unwrap_or(false)
}

fn trace_enabled() -> bool {
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| env_flag("CFI_CUBLAS_TRACE"))
}

macro_rules! trace {
    ($($arg:tt)*) => {
        if trace_enabled() {
            eprintln!("CUDA for Intel cublas: {}", format!($($arg)*));
        }
    };
}

// Load the shim's driver dependency with the DLL on Windows. The NVIDIA
// Runtime can then find the same driver during static CUDA registration.
#[cfg(windows)]
#[link(name = "nvcuda", kind = "raw-dylib")]
unsafe extern "system" {
    fn cuCtxGetCurrent(context: *mut CUcontext) -> CuResult;
}

struct Driver {
    _lib: Library,
    cuCtxGetCurrent: unsafe extern "system" fn(*mut CUcontext) -> CuResult,
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
    cuStreamIsCapturing: unsafe extern "system" fn(CUstream, *mut c_int) -> CuResult,
    cuMemcpyDtoH_v2: unsafe extern "system" fn(*mut c_void, u64, usize) -> CuResult,
    cuMemcpyHtoD_v2: unsafe extern "system" fn(u64, *const c_void, usize) -> CuResult,
    cuMemAlloc_v2: unsafe extern "system" fn(*mut u64, usize) -> CuResult,
    cuMemFree_v2: unsafe extern "system" fn(u64) -> CuResult,
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
        #[cfg(windows)]
        let lib = Library::new("nvcuda.dll").map_err(|_| NOT_INITIALIZED)?;
        #[cfg(not(windows))]
        let lib = Library::new("libcuda.so.1").map_err(|_| NOT_INITIALIZED)?;
        macro_rules! sym {
            ($name:literal) => {
                *lib.get(concat!($name, "\0").as_bytes())
                    .map_err(|_| NOT_INITIALIZED)?
            };
        }
        Ok(Driver {
            #[cfg(windows)]
            cuCtxGetCurrent,
            #[cfg(not(windows))]
            cuCtxGetCurrent: sym!("cuCtxGetCurrent"),
            cuModuleLoadData: sym!("cuModuleLoadData"),
            cuModuleGetFunction: sym!("cuModuleGetFunction"),
            cuLaunchKernel: sym!("cuLaunchKernel"),
            cuStreamSynchronize: sym!("cuStreamSynchronize"),
            cuStreamIsCapturing: sym!("cuStreamIsCapturing"),
            cuMemcpyDtoH_v2: sym!("cuMemcpyDtoH_v2"),
            cuMemcpyHtoD_v2: sym!("cuMemcpyHtoD_v2"),
            cuMemAlloc_v2: sym!("cuMemAlloc_v2"),
            cuMemFree_v2: sym!("cuMemFree_v2"),
            _lib: lib,
        })
    }
}

// ---------------------------------------------------------------------------
// Kernel cache: one module per (context, variant), generated on first use.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Kernel(CUfunction);
// CUfunction handles are opaque driver objects valid across threads of a context.
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

fn elem(d: DType) -> ptxgen::Elem {
    match d {
        DType::F32 => ptxgen::Elem::F32,
        DType::F16 => ptxgen::Elem::F16,
        DType::Bf16 => ptxgen::Elem::Bf16,
    }
}

fn current_context(drv: &Driver) -> Result<CUcontext, c_int> {
    let mut ctx: CUcontext = std::ptr::null_mut();
    if unsafe { (drv.cuCtxGetCurrent)(&mut ctx) } != SUCCESS || ctx.is_null() {
        return Err(NOT_INITIALIZED);
    }
    Ok(ctx)
}

fn kernel_for(
    drv: &Driver,
    d: &GemmDesc,
    pointer_arrays: bool,
    opts: &ptxgen::Opts,
) -> Result<Kernel, c_int> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, String), Kernel>>> = OnceLock::new();
    let ctx = current_context(drv)?;
    let name = ptxgen::entry_name_ex(elem(d.a), elem(d.b), elem(d.c), pointer_arrays, opts);
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(k) = cache.get(&(ctx as usize, name.clone())) {
        return Ok(*k);
    }
    let ptx = format!(
        "{}\0",
        ptxgen::generate_ex(elem(d.a), elem(d.b), elem(d.c), pointer_arrays, opts)
    );
    let mut module: CUmodule = std::ptr::null_mut();
    let err = unsafe { (drv.cuModuleLoadData)(&mut module, ptx.as_ptr().cast()) };
    if err != SUCCESS {
        eprintln!("CUDA for Intel cublas: ModuleLoadData {name} failed err={err}");
        return Err(INTERNAL_ERROR);
    }
    let cname = std::ffi::CString::new(name.clone()).unwrap();
    let mut func: CUfunction = std::ptr::null_mut();
    let err = unsafe { (drv.cuModuleGetFunction)(&mut func, module, cname.as_ptr()) };
    if err != SUCCESS || func.is_null() {
        eprintln!("CUDA for Intel cublas: GetFunction {name} failed err={err}");
        return Err(INTERNAL_ERROR);
    }
    let k = Kernel(func);
    cache.insert((ctx as usize, name), k);
    Ok(k)
}

// ---------------------------------------------------------------------------
// Optional oneMKL / oneMath hook
// ---------------------------------------------------------------------------

/// Optional vendor-GEMM hook. When the environment variable `CFI_CUBLAS_ONEMKL`
/// names a shared library exporting
///
/// ```c
/// int cfi_onemkl_gemm(const CfiGemmCall *call);   // 0 = handled, else decline
/// ```
///
/// every launch is offered to it first (this is where an oneMKL/oneMath SYCL
/// `gemm_batch` binding would live; it needs the Level Zero / SYCL queue that
/// backs `call->stream`). Declining (any nonzero return) falls through to the
/// built-in PTX kernels, so the hook can accelerate a subset of shapes and
/// types. Without the variable nothing is loaded and behaviour is unchanged.
pub mod onemkl {
    use super::*;

    /// ABI of one (possibly batched) GEMM offered to the hook. All pointers are
    /// device addresses; strides are in elements; `pointer_arrays != 0` means
    /// `a`/`b`/`c` point at device tables of `batch` pointers.
    #[repr(C)]
    #[derive(Clone, Copy, Debug, PartialEq)]
    pub struct CfiGemmCall {
        pub struct_size: u32,
        pub trans_a: u32,
        pub trans_b: u32,
        pub m: u32,
        pub n: u32,
        pub k: u32,
        pub lda: u32,
        pub ldb: u32,
        pub ldc: u32,
        /// cudaDataType codes (0 = f32, 2 = f16, 14 = bf16).
        pub a_type: i32,
        pub b_type: i32,
        pub c_type: i32,
        pub alpha: f32,
        pub beta: f32,
        pub batch: u32,
        pub pointer_arrays: u32,
        pub a: u64,
        pub b: u64,
        pub c: u64,
        pub stride_a: i64,
        pub stride_b: i64,
        pub stride_c: i64,
        pub stream: u64,
    }

    type HookFn = unsafe extern "C" fn(*const CfiGemmCall) -> c_int;

    pub struct Hook {
        _lib: Library,
        f: HookFn,
    }

    pub fn hook() -> Option<&'static Hook> {
        static HOOK: OnceLock<Option<Hook>> = OnceLock::new();
        HOOK.get_or_init(|| {
            let path = std::env::var("CFI_CUBLAS_ONEMKL").ok().filter(|p| !p.is_empty())?;
            unsafe {
                let lib = Library::new(&path).ok()?;
                let f: HookFn = *lib.get(b"cfi_onemkl_gemm\0").ok()?;
                Some(Hook { _lib: lib, f })
            }
        })
        .as_ref()
    }

    impl Hook {
        /// True when the hook executed the GEMM.
        pub fn try_gemm(&self, call: &CfiGemmCall) -> bool {
            unsafe { (self.f)(call) == 0 }
        }
    }

    pub fn dtype_code(d: DType) -> i32 {
        match d {
            DType::F32 => CUDA_R_32F,
            DType::F16 => CUDA_R_16F,
            DType::Bf16 => CUDA_R_16BF,
        }
    }
}

// ---------------------------------------------------------------------------
// Handle
// ---------------------------------------------------------------------------

pub(crate) struct Handle {
    stream: CUstream,
    math_mode: c_int,
    pointer_mode: c_int,
    workspace: (usize, usize),
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
    let Ok(driver) = driver() else {
        return NOT_INITIALIZED;
    };
    if current_context(driver).is_err() {
        return NOT_INITIALIZED;
    }
    let ptr = Box::into_raw(Box::new(Handle {
        stream: std::ptr::null_mut(),
        math_mode: 0,
        pointer_mode: POINTER_MODE_HOST,
        workspace: (0, 0),
    })) as *mut c_void;
    unsafe { handle.write(ptr) };
    trace!("cublasCreate_v2 ok");
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDestroy_v2(handle: *mut c_void) -> c_int {
    if handle.is_null() {
        return NOT_INITIALIZED;
    }
    unsafe {
        drop(Box::from_raw(handle as *mut Handle));
    }
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetStream_v2(handle: *mut c_void, stream: CUstream) -> c_int {
    match handle_mut(handle) {
        Ok(h) => {
            h.stream = stream;
            SUCCESS
        }
        Err(_) => NOT_INITIALIZED,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetStream_v2(
    handle: *mut c_void,
    stream: *mut CUstream,
) -> c_int {
    match (handle_mut(handle), stream.is_null()) {
        (Ok(h), false) => {
            unsafe { stream.write(h.stream) };
            SUCCESS
        }
        (Err(_), _) => NOT_INITIALIZED,
        _ => INVALID_VALUE,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetMathMode(handle: *mut c_void, mode: c_int) -> c_int {
    // Accumulation is always f32; tensor-op / TF32 / reduced-precision-reduction
    // flags are accepted and recorded but do not change results.
    match handle_mut(handle) {
        Ok(h) => {
            h.math_mode = mode;
            SUCCESS
        }
        Err(_) => NOT_INITIALIZED,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetMathMode(handle: *mut c_void, mode: *mut c_int) -> c_int {
    match (handle_mut(handle), mode.is_null()) {
        (Ok(h), false) => {
            unsafe { mode.write(h.math_mode) };
            SUCCESS
        }
        (Err(_), _) => NOT_INITIALIZED,
        _ => INVALID_VALUE,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetPointerMode_v2(handle: *mut c_void, mode: c_int) -> c_int {
    if mode != POINTER_MODE_HOST && mode != POINTER_MODE_DEVICE {
        return INVALID_VALUE;
    }
    match handle_mut(handle) {
        Ok(h) => {
            h.pointer_mode = mode;
            SUCCESS
        }
        Err(_) => NOT_INITIALIZED,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetPointerMode_v2(
    handle: *mut c_void,
    mode: *mut c_int,
) -> c_int {
    match (handle_mut(handle), mode.is_null()) {
        (Ok(h), false) => {
            unsafe { mode.write(h.pointer_mode) };
            SUCCESS
        }
        (Err(_), _) => NOT_INITIALIZED,
        _ => INVALID_VALUE,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetWorkspace_v2(
    handle: *mut c_void,
    workspace: *mut c_void,
    size: usize,
) -> c_int {
    // The tiled kernels need no workspace; remember it for completeness.
    match handle_mut(handle) {
        Ok(h) => {
            h.workspace = (workspace as usize, size);
            SUCCESS
        }
        Err(_) => NOT_INITIALIZED,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetVersion_v2(_handle: *mut c_void, version: *mut c_int) -> c_int {
    if version.is_null() {
        return INVALID_VALUE;
    }
    unsafe { version.write(130000) };
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLoggerConfigure(
    _log_is_on: c_int,
    _log_to_std_out: c_int,
    _log_to_std_err: c_int,
    _log_file_name: *const c_char,
) -> c_int {
    SUCCESS
}

fn status_name(status: c_int) -> &'static [u8] {
    match status {
        SUCCESS => b"CUBLAS_STATUS_SUCCESS\0",
        NOT_INITIALIZED => b"CUBLAS_STATUS_NOT_INITIALIZED\0",
        ALLOC_FAILED => b"CUBLAS_STATUS_ALLOC_FAILED\0",
        INVALID_VALUE => b"CUBLAS_STATUS_INVALID_VALUE\0",
        8 => b"CUBLAS_STATUS_ARCH_MISMATCH\0",
        MAPPING_ERROR => b"CUBLAS_STATUS_MAPPING_ERROR\0",
        EXECUTION_FAILED => b"CUBLAS_STATUS_EXECUTION_FAILED\0",
        NOT_SUPPORTED => b"CUBLAS_STATUS_NOT_SUPPORTED\0",
        _ => b"CUBLAS_STATUS_INTERNAL_ERROR\0",
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetStatusString(status: c_int) -> *const c_char {
    status_name(status).as_ptr().cast()
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetStatusName(status: c_int) -> *const c_char {
    status_name(status).as_ptr().cast()
}

// ---------------------------------------------------------------------------
// Launcher
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
pub(crate) enum Batching {
    /// Base pointers plus per-matrix element strides (batch == 1 for plain GEMM).
    Strided { a: u64, b: u64, c: u64, sa: i64, sb: i64, sc: i64 },
    /// Device-resident tables of `batch` pointers each.
    Pointers { a: u64, b: u64, c: u64 },
}

fn sync_stream(drv: &Driver, stream: CUstream) -> c_int {
    // A stream that is being captured only records work; synchronising it would
    // invalidate the capture (GGML reserves its CUDA graph this way). Same-stream
    // ordering already covers the follow-up work.
    let mut capturing: c_int = 0;
    if unsafe { (drv.cuStreamIsCapturing)(stream, &mut capturing) } == SUCCESS && capturing != 0 {
        return SUCCESS;
    }
    let err = unsafe { (drv.cuStreamSynchronize)(stream) };
    if err != SUCCESS {
        eprintln!("CUDA for Intel cublas: stream sync failed err={err}");
        return EXECUTION_FAILED;
    }
    SUCCESS
}

/// Read alpha or beta. In host pointer mode the value is read directly; in
/// device pointer mode the stream is drained first so a producer kernel's
/// result is visible, then the scalar is copied to the host.
unsafe fn read_scalar(
    drv: &Driver,
    h: &Handle,
    ptr: *const c_void,
    kind: ScalarKind,
) -> Result<f32, c_int> {
    if ptr.is_null() {
        return Err(INVALID_VALUE);
    }
    let size = match kind {
        ScalarKind::F16 => 2,
        ScalarKind::F32 => 4,
    };
    let mut raw = [0u8; 4];
    if h.pointer_mode == POINTER_MODE_DEVICE {
        let st = sync_stream(drv, h.stream);
        if st != SUCCESS {
            return Err(st);
        }
        let err = unsafe { (drv.cuMemcpyDtoH_v2)(raw.as_mut_ptr().cast(), ptr as u64, size) };
        if err != SUCCESS {
            return Err(EXECUTION_FAILED);
        }
    } else {
        unsafe { std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), raw.as_mut_ptr(), size) };
    }
    Ok(match kind {
        ScalarKind::F16 => f16_bits_to_f32(u16::from_le_bytes([raw[0], raw[1]])),
        ScalarKind::F32 => f32::from_le_bytes(raw),
    })
}

fn offset_ptr(base: u64, elems: i64, esize: usize) -> u64 {
    base.wrapping_add((elems as u64).wrapping_mul(esize as u64))
}

/// Validate-then-launch. `batch` >= 1.
unsafe fn run_gemm(
    h: &mut Handle,
    d: &GemmDesc,
    alpha: *const c_void,
    beta: *const c_void,
    batch: c_int,
    batching: Batching,
) -> c_int {
    // `CUBLAS_COMPUTE_16F` accumulates in half precision: honour it by rounding
    // the accumulator to f16 in the kernel instead of silently using f32.
    let opts = ptxgen::Opts { acc16: d.scalar == ScalarKind::F16, ..Default::default() };
    unsafe { run_gemm_ex(h, d, alpha, beta, batch, batching, &opts, 0) }
}

/// [`run_gemm`] with kernel options (fused epilogue, f16 accumulation) and the
/// device pointer of the bias vector (required iff `opts.epilogue.has_bias()`).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn run_gemm_ex(
    h: &mut Handle,
    d: &GemmDesc,
    alpha: *const c_void,
    beta: *const c_void,
    batch: c_int,
    batching: Batching,
    opts: &ptxgen::Opts,
    bias: u64,
) -> c_int {
    if opts.epilogue.has_bias() && bias == 0 {
        return INVALID_VALUE;
    }
    if batch < 0 {
        return INVALID_VALUE;
    }
    let Ok(drv) = driver() else {
        return NOT_INITIALIZED;
    };
    if d.m == 0 || d.n == 0 || batch == 0 {
        return SUCCESS; // nothing to write
    }
    let null_operand = match batching {
        Batching::Strided { a, b, c, .. } | Batching::Pointers { a, b, c } => {
            c == 0 || (d.k > 0 && (a == 0 || b == 0))
        }
    };
    if null_operand {
        return INVALID_VALUE;
    }
    let (alpha_v, beta_v) = match unsafe {
        read_scalar(drv, h, alpha, d.scalar).and_then(|a| read_scalar(drv, h, beta, d.scalar).map(|b| (a, b)))
    } {
        Ok(v) => v,
        Err(e) => return e,
    };
    let Some((grid_x, grid_y)) = grid_xy(d.m, d.n) else {
        eprintln!("CUDA for Intel cublas: n={} exceeds grid limit", d.n);
        return NOT_SUPPORTED;
    };
    let pointer_arrays = matches!(batching, Batching::Pointers { .. });
    trace!(
        "gemm {} m={} n={} k={} lda={} ldb={} ldc={} ta={} tb={} batch={batch} alpha={alpha_v} beta={beta_v}",
        d.kernel_name(pointer_arrays), d.m, d.n, d.k, d.lda, d.ldb, d.ldc, d.trans_a, d.trans_b
    );

    // Optional vendor backend first.
    let plain = *opts == ptxgen::Opts::default();
    if let Some(hook) = onemkl::hook().filter(|_| plain) {
        let (a, b, c, sa, sb, sc) = match batching {
            Batching::Strided { a, b, c, sa, sb, sc } => (a, b, c, sa, sb, sc),
            Batching::Pointers { a, b, c } => (a, b, c, 0, 0, 0),
        };
        let call = onemkl::CfiGemmCall {
            struct_size: std::mem::size_of::<onemkl::CfiGemmCall>() as u32,
            trans_a: d.trans_a as u32,
            trans_b: d.trans_b as u32,
            m: d.m,
            n: d.n,
            k: d.k,
            lda: d.lda,
            ldb: d.ldb,
            ldc: d.ldc,
            a_type: onemkl::dtype_code(d.a),
            b_type: onemkl::dtype_code(d.b),
            c_type: onemkl::dtype_code(d.c),
            alpha: alpha_v,
            beta: beta_v,
            batch: batch as u32,
            pointer_arrays: pointer_arrays as u32,
            a,
            b,
            c,
            stride_a: sa,
            stride_b: sb,
            stride_c: sc,
            stream: h.stream as u64,
        };
        if hook.try_gemm(&call) {
            return finish(drv, h);
        }
    }

    // Native XMX (DPAS) fast path: CFI_XMX=1 (off by default until validated on hardware).
    if let Some(status) = unsafe {
        xmx_host::try_gemm(drv, h.stream, d, alpha_v, beta_v, batch, batching, opts, bias)
    } {
        if status != SUCCESS {
            return status;
        }
        return finish(drv, h);
    }

    let kernel = match kernel_for(drv, d, pointer_arrays, opts) {
        Ok(k) => k,
        Err(e) => return e,
    };
    let s = d.strides();
    for (start, count) in batch_chunks(batch as u32) {
        let (mut a_ptr, mut b_ptr, mut c_ptr, mut sa, mut sb, mut sc) = match batching {
            Batching::Strided { a, b, c, sa, sb, sc } => {
                let first = start as i64;
                (
                    offset_ptr(a, first.wrapping_mul(sa), d.a.size()),
                    offset_ptr(b, first.wrapping_mul(sb), d.b.size()),
                    offset_ptr(c, first.wrapping_mul(sc), d.c.size()),
                    sa as u64,
                    sb as u64,
                    sc as u64,
                )
            }
            Batching::Pointers { a, b, c } => {
                let off = start as u64 * 8;
                (a + off, b + off, c + off, 0, 0, 0)
            }
        };
        let (mut alpha_k, mut beta_k) = (alpha_v, beta_v);
        let (mut m, mut n, mut k) = (d.m, d.n, d.k);
        let (mut a_si, mut a_sl, mut b_sl, mut b_sj, mut ldc) =
            (s.a_si, s.a_sl, s.b_sl, s.b_sj, d.ldc);
        let mut bias_k = bias;
        let mut args: Vec<*mut c_void> = vec![
            (&mut a_ptr as *mut u64).cast(),
            (&mut b_ptr as *mut u64).cast(),
            (&mut c_ptr as *mut u64).cast(),
            (&mut alpha_k as *mut f32).cast(),
            (&mut beta_k as *mut f32).cast(),
            (&mut m as *mut u32).cast(),
            (&mut n as *mut u32).cast(),
            (&mut k as *mut u32).cast(),
            (&mut a_si as *mut u32).cast(),
            (&mut a_sl as *mut u32).cast(),
            (&mut b_sl as *mut u32).cast(),
            (&mut b_sj as *mut u32).cast(),
            (&mut ldc as *mut u32).cast(),
            (&mut sa as *mut u64).cast(),
            (&mut sb as *mut u64).cast(),
            (&mut sc as *mut u64).cast(),
        ];
        if opts.epilogue.has_bias() {
            args.push((&mut bias_k as *mut u64).cast());
        }
        let err = unsafe {
            (drv.cuLaunchKernel)(
                kernel.0,
                grid_x,
                grid_y,
                count,
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
            eprintln!("CUDA for Intel cublas: launch failed err={err}");
            return EXECUTION_FAILED;
        }
    }
    finish(drv, h)
}

fn finish(drv: &Driver, h: &Handle) -> c_int {
    if env_flag("CFI_CUBLAS_ASYNC") {
        return SUCCESS;
    }
    sync_stream(drv, h.stream)
}

#[allow(clippy::too_many_arguments)]
unsafe fn gemm_common(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a_type: c_int,
    lda: c_int,
    b_type: c_int,
    ldb: c_int,
    beta: *const c_void,
    c_type: c_int,
    ldc: c_int,
    compute_type: c_int,
    batch: c_int,
    batching: Batching,
) -> c_int {
    let h = match handle_mut(handle) {
        Ok(h) => h,
        Err(_) => return NOT_INITIALIZED,
    };
    if alpha.is_null() || beta.is_null() {
        return INVALID_VALUE;
    }
    let d = match validate(
        transa, transb, m, n, k, lda, ldb, ldc, a_type, b_type, c_type, compute_type,
    ) {
        Ok(d) => d,
        Err(e) => {
            eprintln!(
                "CUDA for Intel cublas: rejected gemm status={e} ta={transa} tb={transb} m={m} n={n} k={k} lda={lda} ldb={ldb} ldc={ldc} types=({a_type},{b_type},{c_type}) compute={compute_type}"
            );
            return e;
        }
    };
    unsafe { run_gemm(h, &d, alpha, beta, batch, batching) }
}

// ---------------------------------------------------------------------------
// GEMM entry points
// ---------------------------------------------------------------------------

/// Legacy mixed-precision entry (used by PyTorch below compute capability 5.0): GemmEx
/// with a CUBLAS_COMPUTE_32F accumulator and the default algorithm.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSgemmEx(
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
) -> c_int {
    unsafe {
        cublasGemmEx(
            handle, transa, transb, m, n, k, alpha, a, a_type, lda, b, b_type, ldb, beta, c, c_type, ldc, 68, -1,
        )
    }
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
    unsafe {
        gemm_common(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a_type,
            lda,
            b_type,
            ldb,
            beta,
            c_type,
            ldc,
            compute_type,
            1,
            Batching::Strided { a: a as u64, b: b as u64, c: c as u64, sa: 0, sb: 0, sc: 0 },
        )
    }
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
    _algo: c_int,
) -> c_int {
    unsafe {
        gemm_common(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a_type,
            lda,
            b_type,
            ldb,
            beta,
            c_type,
            ldc,
            compute_type,
            batch,
            Batching::Strided {
                a: a as u64,
                b: b as u64,
                c: c as u64,
                sa: stride_a,
                sb: stride_b,
                sc: stride_c,
            },
        )
    }
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
    // ggml builds the pointer tables on the cuBLAS stream (k_compute_batched_ptrs)
    // and passes their *device* addresses. The pointer-array kernel reads them
    // on the device, in stream order, so no host round trip or sync is needed.
    unsafe {
        gemm_common(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha,
            a_type,
            lda,
            b_type,
            ldb,
            beta,
            c_type,
            ldc,
            compute_type,
            batch,
            Batching::Pointers { a: a as u64, b: b as u64, c: c as u64 },
        )
    }
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
    unsafe {
        cublasGemmEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            a.cast(),
            CUDA_R_32F,
            lda,
            b.cast(),
            CUDA_R_32F,
            ldb,
            beta.cast(),
            c.cast(),
            CUDA_R_32F,
            ldc,
            COMPUTE_32F,
            0,
        )
    }
}

/// `alpha`/`beta` are `__half` (binary16) pointers.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasHgemm(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const u16,
    a: *const u16,
    lda: c_int,
    b: *const u16,
    ldb: c_int,
    beta: *const u16,
    c: *mut u16,
    ldc: c_int,
) -> c_int {
    unsafe {
        cublasGemmEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            a.cast(),
            CUDA_R_16F,
            lda,
            b.cast(),
            CUDA_R_16F,
            ldb,
            beta.cast(),
            c.cast(),
            CUDA_R_16F,
            ldc,
            COMPUTE_16F,
            0,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSgemmBatched(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_float,
    a: *const *const c_float,
    lda: c_int,
    b: *const *const c_float,
    ldb: c_int,
    beta: *const c_float,
    c: *const *mut c_float,
    ldc: c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        cublasGemmBatchedEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            a.cast(),
            CUDA_R_32F,
            lda,
            b.cast(),
            CUDA_R_32F,
            ldb,
            beta.cast(),
            c.cast(),
            CUDA_R_32F,
            ldc,
            batch,
            COMPUTE_32F,
            0,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSgemmStridedBatched(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_float,
    a: *const c_float,
    lda: c_int,
    stride_a: i64,
    b: *const c_float,
    ldb: c_int,
    stride_b: i64,
    beta: *const c_float,
    c: *mut c_float,
    ldc: c_int,
    stride_c: i64,
    batch: c_int,
) -> c_int {
    unsafe {
        cublasGemmStridedBatchedEx(
            handle,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            a.cast(),
            CUDA_R_32F,
            lda,
            stride_a,
            b.cast(),
            CUDA_R_32F,
            ldb,
            stride_b,
            beta.cast(),
            c.cast(),
            CUDA_R_32F,
            ldc,
            stride_c,
            batch,
            COMPUTE_32F,
            0,
        )
    }
}

fn trsm_kernel(drv: &Driver) -> Result<Kernel, c_int> {
    static CACHE: OnceLock<Mutex<HashMap<usize, Kernel>>> = OnceLock::new();
    let ctx = current_context(drv)?;
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(k) = cache.get(&(ctx as usize)) {
        return Ok(*k);
    }
    let ptx = format!("{}\0", ptxgen::generate_trsm());
    let mut module: CUmodule = std::ptr::null_mut();
    if unsafe { (drv.cuModuleLoadData)(&mut module, ptx.as_ptr().cast()) } != SUCCESS {
        eprintln!("CUDA for Intel cublas: ModuleLoadData strsm failed");
        return Err(INTERNAL_ERROR);
    }
    let cname = std::ffi::CString::new(ptxgen::TRSM_ENTRY).unwrap();
    let mut func: CUfunction = std::ptr::null_mut();
    if unsafe { (drv.cuModuleGetFunction)(&mut func, module, cname.as_ptr()) } != SUCCESS
        || func.is_null()
    {
        eprintln!("CUDA for Intel cublas: GetFunction strsm failed");
        return Err(INTERNAL_ERROR);
    }
    let k = Kernel(func);
    cache.insert(ctx as usize, k);
    Ok(k)
}

/// Batched triangular solve (ggml `solve_tri`): `op(A) X = alpha B` or
/// `X op(A) = alpha B`, B overwritten, A/B given as device pointer arrays.
/// One GPU thread per right-hand-side vector, f32 arithmetic.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasStrsmBatched(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    alpha: *const c_float,
    a: *const *const c_float,
    lda: c_int,
    b: *const *mut c_float,
    ldb: c_int,
    batch: c_int,
) -> c_int {
    let h = match handle_mut(handle) {
        Ok(h) => h,
        Err(_) => return NOT_INITIALIZED,
    };
    let plan = match trsm_plan(side, uplo, trans, diag, m, n, lda, ldb) {
        Ok(p) => p,
        Err(e) => return e,
    };
    if batch < 0 || alpha.is_null() {
        return INVALID_VALUE;
    }
    if m == 0 || n == 0 || batch == 0 {
        return SUCCESS;
    }
    if a.is_null() || b.is_null() {
        return INVALID_VALUE;
    }
    let Ok(drv) = driver() else {
        return NOT_INITIALIZED;
    };
    let alpha_v = match unsafe { read_scalar(drv, h, alpha.cast(), ScalarKind::F32) } {
        Ok(v) => v,
        Err(e) => return e,
    };
    let kernel = match trsm_kernel(drv) {
        Ok(k) => k,
        Err(e) => return e,
    };
    trace!("strsm {plan:?} alpha={alpha_v} batch={batch}");
    for (start, count) in batch_chunks(batch as u32) {
        let off = start as u64 * 8;
        let (mut at, mut bt) = (a as u64 + off, b as u64 + off);
        let mut alpha_k = alpha_v;
        let (mut len, mut nvec, mut a_rs, mut a_cs, mut b_vs, mut b_es) =
            (plan.len, plan.nvec, plan.a_rs, plan.a_cs, plan.b_vs, plan.b_es);
        let (mut lower, mut unit) = (plan.lower as u32, plan.unit as u32);
        let mut args: [*mut c_void; 11] = [
            (&mut at as *mut u64).cast(),
            (&mut bt as *mut u64).cast(),
            (&mut alpha_k as *mut f32).cast(),
            (&mut len as *mut u32).cast(),
            (&mut nvec as *mut u32).cast(),
            (&mut a_rs as *mut u32).cast(),
            (&mut a_cs as *mut u32).cast(),
            (&mut b_vs as *mut u32).cast(),
            (&mut b_es as *mut u32).cast(),
            (&mut lower as *mut u32).cast(),
            (&mut unit as *mut u32).cast(),
        ];
        let err = unsafe {
            (drv.cuLaunchKernel)(
                kernel.0,
                nvec.div_ceil(64),
                count,
                1,
                64,
                1,
                1,
                0,
                h.stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if err != SUCCESS {
            eprintln!("CUDA for Intel cublas: strsm launch failed err={err}");
            return EXECUTION_FAILED;
        }
    }
    finish(drv, h)
}

// ---------------------------------------------------------------------------
// Tests (host only; no driver, no GPU)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};

    #[test]
    fn status_names_use_real_cublas_codes() {
        let name = |s| unsafe { std::ffi::CStr::from_ptr(cublasGetStatusName(s)) }.to_str().unwrap();
        assert_eq!(name(0), "CUBLAS_STATUS_SUCCESS");
        assert_eq!(name(7), "CUBLAS_STATUS_INVALID_VALUE");
        assert_eq!(name(13), "CUBLAS_STATUS_EXECUTION_FAILED");
        assert_eq!(name(15), "CUBLAS_STATUS_NOT_SUPPORTED");
        assert_eq!(name(99), "CUBLAS_STATUS_INTERNAL_ERROR");
    }

    fn fake_handle() -> *mut c_void {
        Box::into_raw(Box::new(Handle {
            stream: null_mut(),
            math_mode: 0,
            pointer_mode: POINTER_MODE_HOST,
            workspace: (0, 0),
        })) as *mut c_void
    }

    #[test]
    fn handle_state_roundtrips() {
        let h = fake_handle();
        unsafe {
            let mut s: CUstream = 1 as _;
            assert_eq!(cublasSetStream_v2(h, 0x1234 as _), SUCCESS);
            assert_eq!(cublasGetStream_v2(h, &mut s), SUCCESS);
            assert_eq!(s as usize, 0x1234);
            assert_eq!(cublasSetMathMode(h, 3), SUCCESS);
            let mut mode = 0;
            assert_eq!(cublasGetMathMode(h, &mut mode), SUCCESS);
            assert_eq!(mode, 3);
            assert_eq!(cublasSetPointerMode_v2(h, 1), SUCCESS);
            assert_eq!(cublasSetPointerMode_v2(h, 5), INVALID_VALUE);
            assert_eq!(cublasSetWorkspace_v2(h, 0x10 as _, 64), SUCCESS);
            assert_eq!(cublasGetStream_v2(h, null_mut()), INVALID_VALUE);
            assert_eq!(cublasSetStream_v2(null_mut(), null_mut()), NOT_INITIALIZED);
            assert_eq!(cublasDestroy_v2(h), SUCCESS);
            assert_eq!(cublasDestroy_v2(null_mut()), NOT_INITIALIZED);
        }
    }

    #[test]
    fn parameter_errors_precede_any_driver_use() {
        let h = fake_handle();
        let one = 1.0f32;
        let p = &one as *const f32 as *const c_void;
        let dummy = 0x1000 as *const c_void;
        unsafe {
            // lda too small for op(A)=N (needs >= m).
            assert_eq!(
                cublasGemmEx(h, OP_N, OP_N, 8, 2, 4, p, dummy, CUDA_R_32F, 7, dummy, CUDA_R_32F, 4, p, dummy as _, CUDA_R_32F, 8, COMPUTE_32F, 0),
                INVALID_VALUE
            );
            // Unknown data type.
            assert_eq!(
                cublasGemmEx(h, OP_N, OP_N, 2, 2, 2, p, dummy, 9, 2, dummy, 9, 2, p, dummy as _, 9, 2, COMPUTE_32F, 0),
                NOT_SUPPORTED
            );
            // Mixed A/B types.
            assert_eq!(
                cublasGemmEx(h, OP_N, OP_N, 2, 2, 2, p, dummy, CUDA_R_16F, 2, dummy, CUDA_R_32F, 2, p, dummy as _, CUDA_R_32F, 2, COMPUTE_32F, 0),
                NOT_SUPPORTED
            );
            // Null alpha.
            assert_eq!(
                cublasGemmEx(h, OP_N, OP_N, 2, 2, 2, null(), dummy, CUDA_R_32F, 2, dummy, CUDA_R_32F, 2, p, dummy as _, CUDA_R_32F, 2, COMPUTE_32F, 0),
                INVALID_VALUE
            );
            // Null handle.
            assert_eq!(
                cublasSgemm_v2(null_mut(), OP_N, OP_N, 2, 2, 2, &one, null(), 2, null(), 2, &one, null_mut(), 2),
                NOT_INITIALIZED
            );
            assert_eq!(cublasDestroy_v2(h), SUCCESS);
        }
    }

    #[test]
    fn hook_is_absent_without_env_and_abi_is_stable() {
        // The environment variable is not set in unit tests.
        if std::env::var("CFI_CUBLAS_ONEMKL").is_err() {
            assert!(onemkl::hook().is_none());
        }
        // Layout is part of the plug-in ABI: 15 x 4-byte, then 8-aligned tail.
        assert_eq!(std::mem::size_of::<onemkl::CfiGemmCall>(), 15 * 4 + 4 + 7 * 8);
        assert_eq!(std::mem::align_of::<onemkl::CfiGemmCall>(), 8);
        assert_eq!(onemkl::dtype_code(DType::Bf16), 14);
    }

    #[test]
    fn pointer_offsets_wrap_for_negative_strides() {
        assert_eq!(offset_ptr(1000, 3, 4), 1012);
        assert_eq!(offset_ptr(1000, -2, 2), 996);
    }

    /// Needs a built nvcuda.dll, an Intel GPU and a current context; run on the
    /// GPU host with `cargo test -p cublas_shim -- --ignored`. Compares all
    /// op/type combinations against `plan::reference_gemm`.
    #[test]
    #[ignore = "requires nvcuda with an Intel GPU and a current CUDA context"]
    fn gpu_gemm_matches_cpu_reference() {
        gpu_tests::run_all();
    }

    /// GPU: cublasLt matmul (Linear shapes, all epilogues/dtypes, row-major D, batch)
    /// and cublasStrsmBatched against the CPU references.
    #[test]
    #[ignore = "requires nvcuda with an Intel GPU and a current CUDA context"]
    fn gpu_cublaslt_and_strsm_match_cpu_reference() {
        gpu_lt_tests::run_all();
    }

    mod gpu_lt_tests {
        use super::*;
        use crate::lt::*;
        use crate::ltplan::*;

        type MemAlloc = unsafe extern "system" fn(*mut u64, usize) -> c_int;
        type MemcpyHtoD = unsafe extern "system" fn(u64, *const c_void, usize) -> c_int;
        type MemcpyDtoH = unsafe extern "system" fn(*mut c_void, u64, usize) -> c_int;
        type CtxCreate = unsafe extern "system" fn(*mut CUcontext, u32, c_int) -> c_int;
        type Init = unsafe extern "system" fn(u32) -> c_int;

        fn fill(dtype: DType, elems: usize, seed: &mut u64) -> Vec<u8> {
            let mut bytes = vec![0u8; elems * dtype.size()];
            for i in 0..elems {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = (((*seed >> 33) % 2001) as f32 - 1000.0) / 500.0;
                encode(dtype, &mut bytes, i, v);
            }
            bytes
        }

        pub fn run_all() {
            let drv = driver().expect("driver");
            let lib = &drv._lib;
            unsafe {
                let init: libloading::Symbol<Init> = lib.get(b"cuInit\0").unwrap();
                let create: libloading::Symbol<CtxCreate> = lib.get(b"cuCtxCreate_v2\0").unwrap();
                let alloc: libloading::Symbol<MemAlloc> = lib.get(b"cuMemAlloc_v2\0").unwrap();
                let htod: libloading::Symbol<MemcpyHtoD> = lib.get(b"cuMemcpyHtoD_v2\0").unwrap();
                let dtoh: libloading::Symbol<MemcpyDtoH> = lib.get(b"cuMemcpyDtoH_v2\0").unwrap();
                assert_eq!(init(0), 0);
                let mut ctx = null_mut();
                assert_eq!(create(&mut ctx, 0, 0), 0);
                let up = |host: &[u8]| -> u64 {
                    let mut p = 0u64;
                    assert_eq!(alloc(&mut p, host.len().max(4)), 0);
                    assert_eq!(htod(p, host.as_ptr().cast(), host.len()), 0);
                    p
                };
                let mut lt: *mut c_void = null_mut();
                assert_eq!(cublasLtCreate(&mut lt), SUCCESS);
                let mut seed = 5u64;
                let code = |d: DType| match d {
                    DType::F32 => CUDA_R_32F,
                    DType::F16 => CUDA_R_16F,
                    DType::Bf16 => CUDA_R_16BF,
                };
                for &(dt, compute) in &[
                    (DType::F32, COMPUTE_32F),
                    (DType::F16, COMPUTE_32F),
                    (DType::Bf16, COMPUTE_32F),
                    (DType::F16, COMPUTE_16F),
                ] {
                    for epi in [EPI_DEFAULT, EPI_BIAS, EPI_RELU_BIAS, EPI_GELU_BIAS] {
                        for &(batch, row_d) in &[(1i32, false), (3, false), (1, true)] {
                            if row_d && epi != EPI_DEFAULT {
                                continue;
                            }
                            // Linear: out = W(out x in) x^T ; Lt call m=out, n=rows, k=in, transa=T.
                            let (m, n, k) = (37u64, 19u64, 45u64);
                            let scale = if compute == COMPUTE_16F { CUDA_R_16F } else { CUDA_R_32F };
                            let mut desc = new_desc(compute, scale);
                            desc.set(D_TRANSA, &1i32.to_le_bytes());
                            desc.set(D_EPILOGUE, &(epi as i32).to_le_bytes());
                            let mk = |r: u64, c: u64, row: bool| {
                                let ld = if row { c } else { r };
                                let mut l = new_layout(code(dt), r, c, ld as i64);
                                if row {
                                    l.set(L_ORDER, &1i32.to_le_bytes());
                                }
                                if batch > 1 {
                                    l.set(L_BATCH_COUNT, &batch.to_le_bytes());
                                    l.set(L_STRIDED_BATCH_OFFSET, &((r * c) as i64).to_le_bytes());
                                }
                                l
                            };
                            let (la, lb, ld_) = (mk(k, m, false), mk(k, n, false), mk(m, n, row_d));
                            let lc = ld_.clone();
                            let a = fill(dt, (k * m) as usize * batch as usize, &mut seed);
                            let b = fill(dt, (k * n) as usize * batch as usize, &mut seed);
                            let bias = fill(dt, m as usize, &mut seed);
                            let c0 = fill(dt, (m * n) as usize * batch as usize, &mut seed);
                            let mut want = c0.clone();
                            let has_bias = epi & 4 != 0;
                            reference_matmul(&desc, &la, &lb, &lc, &ld_, 1.0, 0.0, &a, &b, &c0, &mut want, has_bias.then_some((&bias[..], dt)));
                            // Device objects through the C ABI.
                            let (mut pd, mut pa, mut pb, mut pc): (*mut c_void, *mut c_void, *mut c_void, *mut c_void) =
                                (null_mut(), null_mut(), null_mut(), null_mut());
                            assert_eq!(cublasLtMatmulDescCreate(&mut pd, compute, scale), SUCCESS);
                            let one: c_int = 1;
                            let e = epi as c_int;
                            assert_eq!(cublasLtMatmulDescSetAttribute(pd, D_TRANSA, &one as *const _ as _, 4), SUCCESS);
                            assert_eq!(cublasLtMatmulDescSetAttribute(pd, D_EPILOGUE, &e as *const _ as _, 4), SUCCESS);
                            let (dev_a, dev_b, dev_d, dev_bias) = (up(&a), up(&b), up(&c0), up(&bias));
                            if has_bias {
                                let bp = dev_bias;
                                assert_eq!(cublasLtMatmulDescSetAttribute(pd, D_BIAS_POINTER, &bp as *const _ as _, 8), SUCCESS);
                            }
                            for (out, l) in [(&mut pa, &la), (&mut pb, &lb), (&mut pc, &ld_)] {
                                assert_eq!(cublasLtMatrixLayoutCreate(out, code(dt), l.u64_of(L_ROWS), l.u64_of(L_COLS), l.i64_of(L_LD)), SUCCESS);
                                let ord = l.i32_of(L_ORDER);
                                assert_eq!(cublasLtMatrixLayoutSetAttribute(*out, L_ORDER, &ord as *const _ as _, 4), SUCCESS);
                                let bc = l.i32_of(L_BATCH_COUNT);
                                assert_eq!(cublasLtMatrixLayoutSetAttribute(*out, L_BATCH_COUNT, &bc as *const _ as _, 4), SUCCESS);
                                let bo = l.i64_of(L_STRIDED_BATCH_OFFSET);
                                assert_eq!(cublasLtMatrixLayoutSetAttribute(*out, L_STRIDED_BATCH_OFFSET, &bo as *const _ as _, 8), SUCCESS);
                            }
                            let mut pref = null_mut();
                            assert_eq!(cublasLtMatmulPreferenceCreate(&mut pref), SUCCESS);
                            let mut res = std::mem::zeroed::<HeuristicResult>();
                            let mut cnt = 0;
                            assert_eq!(cublasLtMatmulAlgoGetHeuristic(lt, pd, pa, pb, pc, pc, pref, 1, &mut res, &mut cnt), SUCCESS);
                            assert_eq!(cnt, 1);
                            let (alpha32, beta32) = (1.0f32, 0.0f32);
                            let (alpha16, beta16) = (f32_to_f16_bits(1.0), 0u16);
                            let (al, be): (*const c_void, *const c_void) = if compute == COMPUTE_16F {
                                (&alpha16 as *const u16 as _, &beta16 as *const u16 as _)
                            } else {
                                (&alpha32 as *const f32 as _, &beta32 as *const f32 as _)
                            };
                            let st = cublasLtMatmul(lt, pd, al, dev_a as _, pa, dev_b as _, pb, be, dev_d as _, pc, dev_d as _, pc, &res.algo, null_mut(), 0, null_mut());
                            assert_eq!(st, SUCCESS, "dt={dt:?} epi={epi} batch={batch} rowD={row_d}");
                            let mut got = vec![0u8; c0.len()];
                            assert_eq!(dtoh(got.as_mut_ptr().cast(), dev_d, got.len()), 0);
                            for i in 0..want.len() / dt.size() {
                                let (g, w) = (decode(dt, &got, i), decode(dt, &want, i));
                                let tol = if compute == COMPUTE_16F { 5e-2 } else { 1e-2 } * (1.0 + w.abs());
                                assert!((g - w).abs() <= tol, "elem {i}: got {g} want {w} dt={dt:?} epi={epi} batch={batch} rowD={row_d}");
                            }
                            assert_eq!(cublasLtMatmulPreferenceDestroy(pref), SUCCESS);
                            for p in [pa, pb, pc] {
                                assert_eq!(cublasLtMatrixLayoutDestroy(p), SUCCESS);
                            }
                            assert_eq!(cublasLtMatmulDescDestroy(pd), SUCCESS);
                        }
                    }
                }
                assert_eq!(cublasLtDestroy(lt), SUCCESS);

                // ---- cublasStrsmBatched vs reference_trsm ----
                let mut handle = null_mut();
                assert_eq!(cublasCreate_v2(&mut handle), SUCCESS);
                for side in [SIDE_LEFT, SIDE_RIGHT] {
                    for uplo in [FILL_LOWER, FILL_UPPER] {
                        for trans in [OP_N, OP_T] {
                            for diag in [DIAG_NON_UNIT, DIAG_UNIT] {
                                let (m, n, batch) = (9usize, 6usize, 3usize);
                                let order = if side == SIDE_LEFT { m } else { n };
                                let (lda, ldb) = (order + 1, m + 2);
                                let mut af: Vec<Vec<f32>> = Vec::new();
                                let mut bf: Vec<Vec<f32>> = Vec::new();
                                let mut want: Vec<Vec<f32>> = Vec::new();
                                let (mut ap, mut bp) = (Vec::new(), Vec::new());
                                for _ in 0..batch {
                                    let ab = fill(DType::F32, lda * order, &mut seed);
                                    let mut a: Vec<f32> = (0..lda * order).map(|i| decode(DType::F32, &ab, i) * 0.25).collect();
                                    for d in 0..order {
                                        a[d + d * lda] = 2.0 + 0.1 * d as f32;
                                    }
                                    let bb = fill(DType::F32, ldb * n, &mut seed);
                                    let b: Vec<f32> = (0..ldb * n).map(|i| decode(DType::F32, &bb, i)).collect();
                                    let mut w = b.clone();
                                    reference_trsm(side, uplo, trans, diag, m, n, lda, ldb, 0.5, &a, &mut w);
                                    let bytes = |v: &[f32]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
                                    ap.push(up(&bytes(&a)));
                                    bp.push(up(&bytes(&b)));
                                    af.push(a);
                                    bf.push(b);
                                    want.push(w);
                                }
                                let tab = |v: &[u64]| up(&v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
                                let (ta, tb) = (tab(&ap), tab(&bp));
                                let alpha = 0.5f32;
                                let st = cublasStrsmBatched(
                                    handle, side, uplo, trans, diag, m as c_int, n as c_int, &alpha, ta as _, lda as c_int, tb as _, ldb as c_int, batch as c_int,
                                );
                                assert_eq!(st, SUCCESS);
                                for z in 0..batch {
                                    let mut got = vec![0u8; ldb * n * 4];
                                    assert_eq!(dtoh(got.as_mut_ptr().cast(), bp[z], got.len()), 0);
                                    for j in 0..n {
                                        for i in 0..m {
                                            let g = decode(DType::F32, &got, i + j * ldb);
                                            let w = want[z][i + j * ldb];
                                            assert!((g - w).abs() <= 2e-3 * (1.0 + w.abs()), "trsm side={side} uplo={uplo} trans={trans} diag={diag} z={z} ({i},{j}): {g} vs {w}");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                assert_eq!(cublasDestroy_v2(handle), SUCCESS);
            }
        }
    }

    mod gpu_tests {
        use super::*;

        type MemAlloc = unsafe extern "system" fn(*mut u64, usize) -> c_int;
        type MemcpyHtoD = unsafe extern "system" fn(u64, *const c_void, usize) -> c_int;
        type MemcpyDtoH = unsafe extern "system" fn(*mut c_void, u64, usize) -> c_int;
        type CtxCreate = unsafe extern "system" fn(*mut CUcontext, u32, c_int) -> c_int;
        type Init = unsafe extern "system" fn(u32) -> c_int;

        fn fill(dtype: DType, elems: usize, seed: &mut u64) -> Vec<u8> {
            let mut bytes = vec![0u8; elems * dtype.size()];
            for i in 0..elems {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = (((*seed >> 33) % 2001) as f32 - 1000.0) / 250.0;
                encode(dtype, &mut bytes, i, v);
            }
            bytes
        }

        pub fn run_all() {
            let drv = driver().expect("driver");
            let lib = &drv._lib;
            unsafe {
                let init: libloading::Symbol<Init> = lib.get(b"cuInit\0").unwrap();
                let create: libloading::Symbol<CtxCreate> = lib.get(b"cuCtxCreate_v2\0").unwrap();
                let alloc: libloading::Symbol<MemAlloc> = lib.get(b"cuMemAlloc_v2\0").unwrap();
                let htod: libloading::Symbol<MemcpyHtoD> = lib.get(b"cuMemcpyHtoD_v2\0").unwrap();
                let dtoh: libloading::Symbol<MemcpyDtoH> = lib.get(b"cuMemcpyDtoH_v2\0").unwrap();
                assert_eq!(init(0), 0);
                let mut ctx = null_mut();
                assert_eq!(create(&mut ctx, 0, 0), 0);
                let mut handle = null_mut();
                assert_eq!(cublasCreate_v2(&mut handle), SUCCESS);

                let mut seed = 99u64;
                let type_sets = [
                    (CUDA_R_32F, CUDA_R_32F, COMPUTE_32F),
                    (CUDA_R_16F, CUDA_R_16F, COMPUTE_32F),
                    (CUDA_R_16F, CUDA_R_32F, COMPUTE_32F),
                    (CUDA_R_16F, CUDA_R_16F, COMPUTE_16F),
                    (CUDA_R_16BF, CUDA_R_16BF, COMPUTE_32F),
                    (CUDA_R_16BF, CUDA_R_32F, COMPUTE_32F),
                ];
                for &(ab, cty, compute) in &type_sets {
                    for ta in [OP_N, OP_T] {
                        for tb in [OP_N, OP_T] {
                            for &(m, n, k) in &[(1, 1, 1), (5, 3, 7), (33, 17, 40)] {
                                let pad = 2;
                                let (ra, ca) = if ta == OP_N { (m, k) } else { (k, m) };
                                let (rb, cb) = if tb == OP_N { (k, n) } else { (n, k) };
                                let (lda, ldb, ldc) = (ra + pad, rb + pad, m + pad);
                                let d = validate(ta, tb, m, n, k, lda, ldb, ldc, ab, ab, cty, compute).unwrap();
                                let a = fill(d.a, (lda * ca) as usize, &mut seed);
                                let b = fill(d.b, (ldb * cb) as usize, &mut seed);
                                let c0 = fill(d.c, (ldc * n) as usize, &mut seed);
                                let mut dev = [0u64; 3];
                                for (slot, host) in dev.iter_mut().zip([&a, &b, &c0]) {
                                    assert_eq!(alloc(slot, host.len()), 0);
                                    assert_eq!(htod(*slot, host.as_ptr().cast(), host.len()), 0);
                                }
                                let (alpha, beta) = (0.5f32, 0.25f32);
                                let (ah, bh);
                                let (pa, pb): (*const c_void, *const c_void) = if compute == COMPUTE_16F {
                                    ah = f32_to_f16_bits(alpha);
                                    bh = f32_to_f16_bits(beta);
                                    (&ah as *const u16 as _, &bh as *const u16 as _)
                                } else {
                                    (&alpha as *const f32 as _, &beta as *const f32 as _)
                                };
                                let st = cublasGemmEx(
                                    handle, ta, tb, m, n, k, pa, dev[0] as _, ab, lda, dev[1] as _, ab, ldb, pb, dev[2] as _, cty, ldc, compute, 0,
                                );
                                assert_eq!(st, SUCCESS, "ta={ta} tb={tb} {m}x{n}x{k} ab={ab} c={cty}");
                                let mut got = vec![0u8; c0.len()];
                                assert_eq!(dtoh(got.as_mut_ptr().cast(), dev[2], got.len()), 0);
                                let mut want = c0.clone();
                                let (alpha_r, beta_r) = if compute == COMPUTE_16F {
                                    (f16_bits_to_f32(f32_to_f16_bits(alpha)), f16_bits_to_f32(f32_to_f16_bits(beta)))
                                } else {
                                    (alpha, beta)
                                };
                                reference_gemm(&d, alpha_r, beta_r, &a, &b, &mut want);
                                if compute == COMPUTE_16F {
                                    // Honest half-precision accumulation (rounded after every step).
                                    let r = ltplan::round_f16;
                                    let s = d.strides();
                                    want = c0.clone();
                                    for j in 0..n as usize {
                                        for i in 0..m as usize {
                                            let mut acc = 0.0f32;
                                            for l in 0..k as usize {
                                                let av = decode(d.a, &a, i * s.a_si as usize + l * s.a_sl as usize);
                                                let bv = decode(d.b, &b, l * s.b_sl as usize + j * s.b_sj as usize);
                                                acc = r(acc + av * bv);
                                            }
                                            let ci = i + j * ldc as usize;
                                            let v = r(r(acc * alpha_r) + beta_r * decode(d.c, &c0, ci));
                                            encode(d.c, &mut want, ci, v);
                                        }
                                    }
                                }
                                for idx in 0..(ldc * n) as usize {
                                    let (g, w) = (decode(d.c, &got, idx), decode(d.c, &want, idx));
                                    // Padding rows must be untouched; others within f32 reassociation error.
                                    let tol = 1e-2 * (1.0 + w.abs());
                                    assert!((g - w).abs() <= tol, "idx={idx} got={g} want={w} ta={ta} tb={tb} {m}x{n}x{k} ab={ab} c={cty}");
                                }
                            }
                        }
                    }
                }
                assert_eq!(cublasDestroy_v2(handle), SUCCESS);
            }
        }
    }
}
