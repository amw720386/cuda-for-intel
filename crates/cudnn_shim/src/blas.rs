//! Dynamic access to the cuBLAS (shim) GEMM entry points already loaded by the application.
//! The convolution engine uses `cublasGemmEx` / `cublasGemmStridedBatchedEx` so that the XMX (DPAS)
//! path and every tuning of the cuBLAS shim apply to convolutions too.
use libloading::Library;
use std::ffi::{c_int, c_void};
use std::sync::{Mutex, OnceLock};

type Create = unsafe extern "system" fn(*mut *mut c_void) -> c_int;
type SetStream = unsafe extern "system" fn(*mut c_void, *mut c_void) -> c_int;
type GemmEx = unsafe extern "system" fn(
    *mut c_void, c_int, c_int, c_int, c_int, c_int, *const c_void, *const c_void, c_int, c_int, *const c_void, c_int, c_int,
    *const c_void, *mut c_void, c_int, c_int, c_int, c_int,
) -> c_int;
type GemmSbEx = unsafe extern "system" fn(
    *mut c_void, c_int, c_int, c_int, c_int, c_int, *const c_void, *const c_void, c_int, c_int, i64, *const c_void, c_int, c_int, i64,
    *const c_void, *mut c_void, c_int, c_int, i64, c_int, c_int, c_int,
) -> c_int;

pub struct Blas {
    _lib: Library,
    handle: *mut c_void,
    set_stream: SetStream,
    gemm_ex: GemmEx,
    gemm_sb_ex: GemmSbEx,
}
unsafe impl Send for Blas {}

pub const R_32F: c_int = 0;
pub const R_16F: c_int = 2;
pub const R_16BF: c_int = 14;
pub const COMPUTE_32F: c_int = 68;

fn open() -> Option<Library> {
    let names: &[&str] = if cfg!(windows) {
        &["cublas64_11.dll", "cublas64_12.dll", "cublas64_13.dll"]
    } else {
        &["libcublas.so.11", "libcublas.so.12", "libcublas.so.13", "libcublas.so"]
    };
    // A name the application already loaded resolves to that module; prefer the one in the process.
    for n in names {
        if let Ok(l) = unsafe { Library::new(n) } {
            return Some(l);
        }
    }
    None
}

fn load() -> Option<Blas> {
    unsafe {
        let lib = open()?;
        let create: Create = *lib.get::<Create>(b"cublasCreate_v2\0").ok()?;
        let set_stream: SetStream = *lib.get::<SetStream>(b"cublasSetStream_v2\0").ok()?;
        let gemm_ex: GemmEx = *lib.get::<GemmEx>(b"cublasGemmEx\0").ok()?;
        let gemm_sb_ex: GemmSbEx = *lib.get::<GemmSbEx>(b"cublasGemmStridedBatchedEx\0").ok()?;
        let mut h = std::ptr::null_mut();
        if create(&mut h) != 0 || h.is_null() {
            return None;
        }
        Some(Blas { _lib: lib, handle: h, set_stream, gemm_ex, gemm_sb_ex })
    }
}

/// The process-wide handle; `None` when no cuBLAS is loadable (the direct kernels are used then).
/// A failed creation (for instance before the first context exists) is retried on the next call.
pub fn global() -> Option<&'static Mutex<Blas>> {
    static B: OnceLock<Mutex<Blas>> = OnceLock::new();
    static INIT: Mutex<()> = Mutex::new(());
    if let Some(b) = B.get() {
        return Some(b);
    }
    let _g = INIT.lock().unwrap_or_else(|e| e.into_inner());
    if B.get().is_none() {
        if let Some(l) = load() {
            let _ = B.set(Mutex::new(l));
        }
    }
    B.get()
}

impl Blas {
    pub fn set_stream(&mut self, s: *mut c_void) -> c_int {
        unsafe { (self.set_stream)(self.handle, s) }
    }
    /// Column-major GEMM with f32 accumulation. `dt` is a CUDA data type id; pointers are device addresses.
    #[allow(clippy::too_many_arguments)]
    pub fn gemm(&mut self, ta: bool, tb: bool, m: i32, n: i32, k: i32, alpha: f32, a: u64, lda: i32, b: u64, ldb: i32, beta: f32, c: u64, ldc: i32, dt: c_int, batch: i32, sa: i64, sb: i64, sc: i64) -> c_int {
        let (ta, tb) = (ta as c_int, tb as c_int);
        unsafe {
            if batch <= 1 {
                (self.gemm_ex)(
                    self.handle, ta, tb, m, n, k, (&alpha as *const f32).cast(), a as *const c_void, dt, lda, b as *const c_void, dt, ldb,
                    (&beta as *const f32).cast(), c as *mut c_void, dt, ldc, COMPUTE_32F, -1,
                )
            } else {
                (self.gemm_sb_ex)(
                    self.handle, ta, tb, m, n, k, (&alpha as *const f32).cast(), a as *const c_void, dt, lda, sa, b as *const c_void, dt, ldb, sb,
                    (&beta as *const f32).cast(), c as *mut c_void, dt, ldc, sc, batch, COMPUTE_32F, -1,
                )
            }
        }
    }
}
