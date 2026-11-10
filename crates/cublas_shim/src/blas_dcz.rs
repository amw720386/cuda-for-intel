//! f64 and complex cuBLAS (`D`, `C`, `Z`), plus the strided f32 level-1/2 routines, on the
//! generated kernels in `kernels::ptxgen` (launch lists from `kernels::plan`):
//!
//! * level 3: `{D,C,Z}gemm`, `{D,C,Z}gemmStridedBatched`, `{D,C,Z}gemmBatched`, the `C/Z`
//!   `gemm3m` aliases, `cublasCgemmEx`, and `cublasGemmEx`/`GemmStridedBatchedEx`/
//!   `GemmBatchedEx` with `CUDA_R_64F`, `CUDA_C_32F` or `CUDA_C_64F` operands (routed here
//!   from `gemm_common`; f16/bf16/f32 keep their existing kernels and XMX path);
//!   `{S,D,C,Z}trsm` and `{D,C,Z}trsmBatched`.
//! * level 2: `{D,C,Z}gemv` (and `Sgemv` with non-unit increments).
//! * level 1: `{S,D,C,Z}axpy`, `{S,D,C,Z,Cs,Zd}scal`, `{S,D,Sc,Dz}nrm2`, `{S,D}dot`,
//!   `{C,Z}dot{u,c}`, `cublasDotEx`/`cublasDotcEx` (also f16/bf16 with f32 accumulation).
//!
//! Scalars follow the handle's pointer mode: host scalars are passed by value, device
//! scalars are read by the kernel; dot/nrm2 results go straight to device memory in device
//! mode, or through a scratch word that is copied back in host mode.
use super::kernels::plan::{self, Batch, Launch, Scalar};
use super::kernels::ptxgen::{self, Ty};
use super::*;

/// Kernels of the module of `ty`, compiled once per context.
fn functions(drv: &Driver, ty: Ty) -> Result<Vec<CUfunction>, c_int> {
    struct Fns(Vec<CUfunction>);
    // CUfunction handles are opaque driver objects valid across threads of a context.
    unsafe impl Send for Fns {}
    static CACHE: OnceLock<Mutex<HashMap<(usize, Ty), Fns>>> = OnceLock::new();
    let ctx = current_context(drv)?;
    let mut cache = CACHE
        .get_or_init(Default::default)
        .lock()
        .unwrap_or_else(|e| e.into_inner());
    if let Some(f) = cache.get(&(ctx as usize, ty)) {
        return Ok(f.0.clone());
    }
    let src = format!("{}\0", ptxgen::module(ty));
    let mut module: CUmodule = std::ptr::null_mut();
    let err = unsafe { (drv.cuModuleLoadData)(&mut module, src.as_ptr().cast()) };
    if err != SUCCESS {
        eprintln!(
            "CUDA for Intel cublas: loading the {} BLAS kernels failed (err={err})",
            ty.tag()
        );
        // A device without fp64 cannot build the D/Z modules.
        return Err(if ty.double() {
            NOT_SUPPORTED
        } else {
            INTERNAL_ERROR
        });
    }
    let mut fns = Vec::new();
    for &k in ptxgen::kernels(ty) {
        let name = std::ffi::CString::new(ptxgen::entry_name(k, ty)).unwrap();
        let mut f: CUfunction = std::ptr::null_mut();
        if unsafe { (drv.cuModuleGetFunction)(&mut f, module, name.as_ptr()) } != SUCCESS
            || f.is_null()
        {
            eprintln!("CUDA for Intel cublas: GetFunction {name:?} failed");
            return Err(INTERNAL_ERROR);
        }
        fns.push(f);
    }
    cache.insert((ctx as usize, ty), Fns(fns.clone()));
    Ok(fns)
}

/// Enqueue `launches` on the handle's stream (no completion wait).
pub(crate) fn enqueue(drv: &Driver, h: &Handle, launches: &[Launch]) -> c_int {
    for l in launches {
        let fns = match functions(drv, l.ty) {
            Ok(f) => f,
            Err(e) => return e,
        };
        let pos = ptxgen::kernels(l.ty)
            .iter()
            .position(|k| *k == l.kernel)
            .unwrap();
        trace!(
            "{} grid={:?} args={:x?}",
            ptxgen::entry_name(l.kernel, l.ty),
            l.grid,
            l.args
        );
        let mut slots = l.args.clone();
        let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|v| (v as *mut u64).cast()).collect();
        let [gx, gy, gz] = l.grid;
        let [bx, by, bz] = l.block;
        let err = unsafe {
            (drv.cuLaunchKernel)(
                fns[pos],
                gx,
                gy,
                gz,
                bx,
                by,
                bz,
                0,
                h.stream,
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        };
        if err != SUCCESS {
            eprintln!("CUDA for Intel cublas: launch failed err={err}");
            return EXECUTION_FAILED;
        }
    }
    SUCCESS
}

pub(crate) fn run(h: &Handle, launches: Result<Vec<Launch>, c_int>) -> c_int {
    let launches = match launches {
        Ok(l) => l,
        Err(e) => return e,
    };
    if launches.is_empty() {
        return SUCCESS;
    }
    let Ok(drv) = driver() else {
        return NOT_INITIALIZED;
    };
    let st = enqueue(drv, h, &launches);
    if st != SUCCESS {
        return st;
    }
    finish(drv, h)
}

/// alpha/beta of type `ty` per the pointer mode.
pub(crate) unsafe fn scalar(h: &Handle, ptr: *const c_void, ty: Ty) -> Result<Scalar, c_int> {
    if ptr.is_null() {
        return Err(INVALID_VALUE);
    }
    if h.pointer_mode == POINTER_MODE_DEVICE {
        return Ok(Scalar::device(ptr as u64));
    }
    let bytes = unsafe { std::slice::from_raw_parts(ptr.cast::<u8>(), ty.size() as usize) };
    Ok(Scalar::from_bytes(ty, bytes))
}

pub(crate) fn handle(handle: *mut c_void) -> Result<&'static mut Handle, c_int> {
    handle_mut(handle).map_err(|_| NOT_INITIALIZED)
}

macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => {
                trace!("{}:{} returns status {e}: {}", file!(), line!(), stringify!($e).split_whitespace().collect::<Vec<_>>().join(" "));
                return e;
            }
        }
    };
}

// ---------------------------------------------------------------------------
// Level 3
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm(
    hd: *mut c_void,
    ty: Ty,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    lda: c_int,
    ldb: c_int,
    beta: *const c_void,
    ldc: c_int,
    batch: c_int,
    layout: Batch,
) -> c_int {
    let h = tri!(handle(hd));
    let a = tri!(unsafe { scalar(h, alpha, ty) });
    let b = tri!(unsafe { scalar(h, beta, ty) });
    let launches = plan::gemm(
        ty, transa, transb, m, n, k, lda, ldb, ldc, a, b, batch, layout,
    );
    if let Ok(l) = &launches {
        let null = match layout {
            Batch::Strided { a, b, c, .. } | Batch::Tables { a, b, c } => {
                c == 0 || (k > 0 && (a == 0 || b == 0))
            }
        };
        if !l.is_empty() && null {
            return INVALID_VALUE;
        }
    }
    trace!(
        "{}gemm ta={transa} tb={transb} m={m} n={n} k={k} lda={lda} ldb={ldb} ldc={ldc} batch={batch}",
        ty.tag()
    );
    run(h, launches)
}

/// `cublasGemmEx`-family entry for f64/complex operand types: `None` when `a_type` is not
/// one of them (the caller keeps its f16/bf16/f32 path).
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemm_ex(
    hd: *mut c_void,
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
    compute: c_int,
    batch: c_int,
    layout: Batch,
) -> Option<c_int> {
    let ty = plan::wide_type(a_type)?;
    if b_type != a_type || c_type != a_type || !plan::compute_ok(ty, compute) {
        eprintln!(
            "CUDA for Intel cublas: rejected gemmEx types=({a_type},{b_type},{c_type}) compute={compute}"
        );
        return Some(NOT_SUPPORTED);
    }
    Some(unsafe {
        gemm(
            hd, ty, transa, transb, m, n, k, alpha, lda, ldb, beta, ldc, batch, layout,
        )
    })
}

/// `cuComplex` / `cuDoubleComplex` as stored.
pub type Complex32 = [f32; 2];
pub type Complex64 = [f64; 2];

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDgemm_v2(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    let layout = Batch::Strided {
        a: a as u64,
        b: b as u64,
        c: c as u64,
        sa: 0,
        sb: 0,
        sc: 0,
    };
    unsafe {
        gemm(
            handle,
            Ty::D,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            1,
            layout,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDgemmStridedBatched(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    stride_a: i64,
    b: *const f64,
    ldb: c_int,
    stride_b: i64,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
    stride_c: i64,
    batch: c_int,
) -> c_int {
    let layout = Batch::Strided {
        a: a as u64,
        b: b as u64,
        c: c as u64,
        sa: stride_a,
        sb: stride_b,
        sc: stride_c,
    };
    unsafe {
        gemm(
            handle,
            Ty::D,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            batch,
            layout,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDgemmBatched(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const *const f64,
    lda: c_int,
    b: *const *const f64,
    ldb: c_int,
    beta: *const f64,
    c: *const *mut f64,
    ldc: c_int,
    batch: c_int,
) -> c_int {
    let layout = Batch::Tables {
        a: a as u64,
        b: b as u64,
        c: c as u64,
    };
    unsafe {
        gemm(
            handle,
            Ty::D,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            batch,
            layout,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgemm_v2(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    b: *const Complex32,
    ldb: c_int,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    let layout = Batch::Strided {
        a: a as u64,
        b: b as u64,
        c: c as u64,
        sa: 0,
        sb: 0,
        sc: 0,
    };
    unsafe {
        gemm(
            handle,
            Ty::C,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            1,
            layout,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgemmStridedBatched(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    stride_a: i64,
    b: *const Complex32,
    ldb: c_int,
    stride_b: i64,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: c_int,
    stride_c: i64,
    batch: c_int,
) -> c_int {
    let layout = Batch::Strided {
        a: a as u64,
        b: b as u64,
        c: c as u64,
        sa: stride_a,
        sb: stride_b,
        sc: stride_c,
    };
    unsafe {
        gemm(
            handle,
            Ty::C,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            batch,
            layout,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgemmBatched(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex32,
    a: *const *const Complex32,
    lda: c_int,
    b: *const *const Complex32,
    ldb: c_int,
    beta: *const Complex32,
    c: *const *mut Complex32,
    ldc: c_int,
    batch: c_int,
) -> c_int {
    let layout = Batch::Tables {
        a: a as u64,
        b: b as u64,
        c: c as u64,
    };
    unsafe {
        gemm(
            handle,
            Ty::C,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            batch,
            layout,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZgemm_v2(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: c_int,
    b: *const Complex64,
    ldb: c_int,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: c_int,
) -> c_int {
    let layout = Batch::Strided {
        a: a as u64,
        b: b as u64,
        c: c as u64,
        sa: 0,
        sb: 0,
        sc: 0,
    };
    unsafe {
        gemm(
            handle,
            Ty::Z,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            1,
            layout,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZgemmStridedBatched(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: c_int,
    stride_a: i64,
    b: *const Complex64,
    ldb: c_int,
    stride_b: i64,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: c_int,
    stride_c: i64,
    batch: c_int,
) -> c_int {
    let layout = Batch::Strided {
        a: a as u64,
        b: b as u64,
        c: c as u64,
        sa: stride_a,
        sb: stride_b,
        sc: stride_c,
    };
    unsafe {
        gemm(
            handle,
            Ty::Z,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            batch,
            layout,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZgemmBatched(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex64,
    a: *const *const Complex64,
    lda: c_int,
    b: *const *const Complex64,
    ldb: c_int,
    beta: *const Complex64,
    c: *const *mut Complex64,
    ldc: c_int,
    batch: c_int,
) -> c_int {
    let layout = Batch::Tables {
        a: a as u64,
        b: b as u64,
        c: c as u64,
    };
    unsafe {
        gemm(
            handle,
            Ty::Z,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            batch,
            layout,
        )
    }
}
// The Gauss "3m" variants compute the same product; ours uses the plain algorithm.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgemm3m(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    b: *const Complex32,
    ldb: c_int,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    let layout = Batch::Strided {
        a: a as u64,
        b: b as u64,
        c: c as u64,
        sa: 0,
        sb: 0,
        sc: 0,
    };
    unsafe {
        gemm(
            handle,
            Ty::C,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            1,
            layout,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgemm3mStridedBatched(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    stride_a: i64,
    b: *const Complex32,
    ldb: c_int,
    stride_b: i64,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: c_int,
    stride_c: i64,
    batch: c_int,
) -> c_int {
    let layout = Batch::Strided {
        a: a as u64,
        b: b as u64,
        c: c as u64,
        sa: stride_a,
        sb: stride_b,
        sc: stride_c,
    };
    unsafe {
        gemm(
            handle,
            Ty::C,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            batch,
            layout,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgemm3mBatched(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex32,
    a: *const *const Complex32,
    lda: c_int,
    b: *const *const Complex32,
    ldb: c_int,
    beta: *const Complex32,
    c: *const *mut Complex32,
    ldc: c_int,
    batch: c_int,
) -> c_int {
    let layout = Batch::Tables {
        a: a as u64,
        b: b as u64,
        c: c as u64,
    };
    unsafe {
        gemm(
            handle,
            Ty::C,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            batch,
            layout,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZgemm3m(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: c_int,
    b: *const Complex64,
    ldb: c_int,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: c_int,
) -> c_int {
    unsafe {
        cublasZgemm_v2(
            handle, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc,
        )
    }
}

/// Complex single-precision GemmEx (`CUDA_C_32F` operands only; `CUDA_C_8I` is not
/// supported).
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgemmEx(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex32,
    a: *const c_void,
    a_type: c_int,
    lda: c_int,
    b: *const c_void,
    b_type: c_int,
    ldb: c_int,
    beta: *const Complex32,
    c: *mut c_void,
    c_type: c_int,
    ldc: c_int,
) -> c_int {
    if a_type != plan::C_32F || b_type != plan::C_32F || c_type != plan::C_32F {
        return NOT_SUPPORTED;
    }
    let layout = Batch::Strided {
        a: a as u64,
        b: b as u64,
        c: c as u64,
        sa: 0,
        sb: 0,
        sc: 0,
    };
    unsafe {
        gemm(
            handle,
            Ty::C,
            transa,
            transb,
            m,
            n,
            k,
            alpha.cast(),
            lda,
            ldb,
            beta.cast(),
            ldc,
            1,
            layout,
        )
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn trsm(
    hd: *mut c_void,
    ty: Ty,
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    alpha: *const c_void,
    a: u64,
    lda: c_int,
    b: u64,
    ldb: c_int,
    batch: c_int,
    tables: bool,
) -> c_int {
    let h = tri!(handle(hd));
    let p = tri!(plan::tri(ty, side, uplo, trans, diag, m, n, lda, ldb));
    let al = tri!(unsafe { scalar(h, alpha, ty) });
    let launches = tri!(plan::trsm(ty, &p, al, a, b, batch, tables));
    if !launches.is_empty() && (a == 0 || b == 0) {
        return INVALID_VALUE;
    }
    trace!(
        "{}trsm{} {p:?} batch={batch}",
        ty.tag(),
        if tables { "Batched" } else { "" }
    );
    run(h, Ok(launches))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasStrsm_v2(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    b: *mut f32,
    ldb: c_int,
) -> c_int {
    unsafe {
        trsm(
            handle,
            Ty::S,
            side,
            uplo,
            trans,
            diag,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            1,
            false,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDtrsm_v2(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    b: *mut f64,
    ldb: c_int,
) -> c_int {
    unsafe {
        trsm(
            handle,
            Ty::D,
            side,
            uplo,
            trans,
            diag,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            1,
            false,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCtrsm_v2(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    b: *mut Complex32,
    ldb: c_int,
) -> c_int {
    unsafe {
        trsm(
            handle,
            Ty::C,
            side,
            uplo,
            trans,
            diag,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            1,
            false,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZtrsm_v2(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: c_int,
    b: *mut Complex64,
    ldb: c_int,
) -> c_int {
    unsafe {
        trsm(
            handle,
            Ty::Z,
            side,
            uplo,
            trans,
            diag,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            1,
            false,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDtrsmBatched(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: *const *const f64,
    lda: c_int,
    b: *const *mut f64,
    ldb: c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        trsm(
            handle,
            Ty::D,
            side,
            uplo,
            trans,
            diag,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            batch,
            true,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCtrsmBatched(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex32,
    a: *const *const Complex32,
    lda: c_int,
    b: *const *mut Complex32,
    ldb: c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        trsm(
            handle,
            Ty::C,
            side,
            uplo,
            trans,
            diag,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            batch,
            true,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZtrsmBatched(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex64,
    a: *const *const Complex64,
    lda: c_int,
    b: *const *mut Complex64,
    ldb: c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        trsm(
            handle,
            Ty::Z,
            side,
            uplo,
            trans,
            diag,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            batch,
            true,
        )
    }
}

// ---------------------------------------------------------------------------
// Level 2
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn gemv(
    hd: *mut c_void,
    ty: Ty,
    trans: c_int,
    m: c_int,
    n: c_int,
    alpha: *const c_void,
    a: u64,
    lda: c_int,
    x: u64,
    incx: c_int,
    beta: *const c_void,
    y: u64,
    incy: c_int,
) -> c_int {
    let h = tri!(handle(hd));
    let al = tri!(unsafe { scalar(h, alpha, ty) });
    let be = tri!(unsafe { scalar(h, beta, ty) });
    let launches = tri!(plan::gemv(
        ty, trans, m, n, a, lda, x, incx, y, incy, al, be
    ));
    if !launches.is_empty() && (a == 0 || x == 0 || y == 0) {
        return INVALID_VALUE;
    }
    trace!(
        "{}gemv trans={trans} m={m} n={n} lda={lda} incx={incx} incy={incy}",
        ty.tag()
    );
    run(h, Ok(launches))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDgemv_v2(
    handle: *mut c_void,
    trans: c_int,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    beta: *const f64,
    y: *mut f64,
    incy: c_int,
) -> c_int {
    unsafe {
        gemv(
            handle,
            Ty::D,
            trans,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            x as u64,
            incx,
            beta.cast(),
            y as u64,
            incy,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgemv_v2(
    handle: *mut c_void,
    trans: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    x: *const Complex32,
    incx: c_int,
    beta: *const Complex32,
    y: *mut Complex32,
    incy: c_int,
) -> c_int {
    unsafe {
        gemv(
            handle,
            Ty::C,
            trans,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            x as u64,
            incx,
            beta.cast(),
            y as u64,
            incy,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZgemv_v2(
    handle: *mut c_void,
    trans: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: c_int,
    x: *const Complex64,
    incx: c_int,
    beta: *const Complex64,
    y: *mut Complex64,
    incy: c_int,
) -> c_int {
    unsafe {
        gemv(
            handle,
            Ty::Z,
            trans,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            x as u64,
            incx,
            beta.cast(),
            y as u64,
            incy,
        )
    }
}

// ---------------------------------------------------------------------------
// Level 1
// ---------------------------------------------------------------------------

/// Run a reduction whose kernel writes `bytes` at a device address: straight into `result`
/// in device pointer mode, or through scratch memory copied to the host pointer.
pub(crate) unsafe fn reduce(
    hd: *mut c_void,
    result: *mut c_void,
    bytes: usize,
    launch: impl FnOnce(u64) -> Launch,
    n: c_int,
) -> c_int {
    if result.is_null() {
        return INVALID_VALUE;
    }
    let h = tri!(handle(hd));
    if h.pointer_mode != POINTER_MODE_DEVICE && n <= 0 {
        unsafe { std::ptr::write_bytes(result.cast::<u8>(), 0, bytes) };
        return SUCCESS;
    }
    let Ok(drv) = driver() else {
        return NOT_INITIALIZED;
    };
    if h.pointer_mode == POINTER_MODE_DEVICE {
        let st = enqueue(drv, h, &[launch(result as u64)]);
        return if st == SUCCESS { finish(drv, h) } else { st };
    }
    let mut scratch = 0u64;
    if unsafe { (drv.cuMemAlloc_v2)(&mut scratch, bytes.max(16)) } != SUCCESS {
        return ALLOC_FAILED;
    }
    let mut st = enqueue(drv, h, &[launch(scratch)]);
    if st == SUCCESS {
        st = sync_stream(drv, h.stream);
    }
    if st == SUCCESS && unsafe { (drv.cuMemcpyDtoH_v2)(result, scratch, bytes) } != SUCCESS {
        st = EXECUTION_FAILED;
    }
    unsafe { (drv.cuMemFree_v2)(scratch) };
    st
}

#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn dot(
    hd: *mut c_void,
    ty: Ty,
    n: c_int,
    x: u64,
    incx: c_int,
    y: u64,
    incy: c_int,
    result: *mut c_void,
    conj: bool,
) -> c_int {
    if result.is_null() {
        return INVALID_VALUE;
    }
    if handle_mut(hd).is_err() {
        return NOT_INITIALIZED;
    }
    if n > 0 && (x == 0 || y == 0) {
        return INVALID_VALUE;
    }
    trace!(
        "{}dot{} n={n} incx={incx} incy={incy}",
        ty.tag(),
        if conj { "c" } else { "" }
    );
    unsafe {
        reduce(
            hd,
            result,
            ty.size() as usize,
            |out| plan::dot(ty, n, x, incx, y, incy, out, conj),
            n,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDdot_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const f64,
    incx: c_int,
    y: *const f64,
    incy: c_int,
    result: *mut f64,
) -> c_int {
    unsafe {
        dot(
            handle,
            Ty::D,
            n,
            x as u64,
            incx,
            y as u64,
            incy,
            result.cast(),
            false,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCdotu_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex32,
    incx: c_int,
    y: *const Complex32,
    incy: c_int,
    result: *mut Complex32,
) -> c_int {
    unsafe {
        dot(
            handle,
            Ty::C,
            n,
            x as u64,
            incx,
            y as u64,
            incy,
            result.cast(),
            false,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCdotc_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex32,
    incx: c_int,
    y: *const Complex32,
    incy: c_int,
    result: *mut Complex32,
) -> c_int {
    unsafe {
        dot(
            handle,
            Ty::C,
            n,
            x as u64,
            incx,
            y as u64,
            incy,
            result.cast(),
            true,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZdotu_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex64,
    incx: c_int,
    y: *const Complex64,
    incy: c_int,
    result: *mut Complex64,
) -> c_int {
    unsafe {
        dot(
            handle,
            Ty::Z,
            n,
            x as u64,
            incx,
            y as u64,
            incy,
            result.cast(),
            false,
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZdotc_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex64,
    incx: c_int,
    y: *const Complex64,
    incy: c_int,
    result: *mut Complex64,
) -> c_int {
    unsafe {
        dot(
            handle,
            Ty::Z,
            n,
            x as u64,
            incx,
            y as u64,
            incy,
            result.cast(),
            true,
        )
    }
}

/// Execution type cuBLAS pairs with each stored type in `cublasDotEx`.
fn dot_ex_ok(ty: Ty, exec: c_int) -> bool {
    match ty {
        Ty::H | Ty::B | Ty::S => exec == plan::R_32F,
        Ty::D => exec == plan::R_64F,
        Ty::C => exec == plan::C_32F,
        Ty::Z => exec == plan::C_64F,
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn dot_ex(
    hd: *mut c_void,
    n: c_int,
    x: *const c_void,
    x_type: c_int,
    incx: c_int,
    y: *const c_void,
    y_type: c_int,
    incy: c_int,
    result: *mut c_void,
    result_type: c_int,
    exec: c_int,
    conj: bool,
) -> c_int {
    let Some(ty) = plan::data_type(x_type) else {
        return NOT_SUPPORTED;
    };
    if y_type != x_type || result_type != x_type || !dot_ex_ok(ty, exec) {
        return NOT_SUPPORTED;
    }
    unsafe { dot(hd, ty, n, x as u64, incx, y as u64, incy, result, conj) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDotEx(
    handle: *mut c_void,
    n: c_int,
    x: *const c_void,
    x_type: c_int,
    incx: c_int,
    y: *const c_void,
    y_type: c_int,
    incy: c_int,
    result: *mut c_void,
    result_type: c_int,
    exec: c_int,
) -> c_int {
    unsafe {
        dot_ex(
            handle,
            n,
            x,
            x_type,
            incx,
            y,
            y_type,
            incy,
            result,
            result_type,
            exec,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDotcEx(
    handle: *mut c_void,
    n: c_int,
    x: *const c_void,
    x_type: c_int,
    incx: c_int,
    y: *const c_void,
    y_type: c_int,
    incy: c_int,
    result: *mut c_void,
    result_type: c_int,
    exec: c_int,
) -> c_int {
    unsafe {
        dot_ex(
            handle,
            n,
            x,
            x_type,
            incx,
            y,
            y_type,
            incy,
            result,
            result_type,
            exec,
            true,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSnrm2_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const f32,
    incx: c_int,
    result: *mut f32,
) -> c_int {
    if n > 0 && incx > 0 && x.is_null() {
        return INVALID_VALUE;
    }
    let ty = Ty::S;
    trace!("{}nrm2 n={n} incx={incx}", ty.tag());
    unsafe {
        reduce(
            handle,
            result.cast(),
            ty.comp() as usize,
            |out| plan::nrm2(ty, n, x as u64, incx, out),
            if incx > 0 { n } else { 0 },
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDnrm2_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const f64,
    incx: c_int,
    result: *mut f64,
) -> c_int {
    if n > 0 && incx > 0 && x.is_null() {
        return INVALID_VALUE;
    }
    let ty = Ty::D;
    trace!("{}nrm2 n={n} incx={incx}", ty.tag());
    unsafe {
        reduce(
            handle,
            result.cast(),
            ty.comp() as usize,
            |out| plan::nrm2(ty, n, x as u64, incx, out),
            if incx > 0 { n } else { 0 },
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasScnrm2_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex32,
    incx: c_int,
    result: *mut f32,
) -> c_int {
    if n > 0 && incx > 0 && x.is_null() {
        return INVALID_VALUE;
    }
    let ty = Ty::C;
    trace!("{}nrm2 n={n} incx={incx}", ty.tag());
    unsafe {
        reduce(
            handle,
            result.cast(),
            ty.comp() as usize,
            |out| plan::nrm2(ty, n, x as u64, incx, out),
            if incx > 0 { n } else { 0 },
        )
    }
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDznrm2_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex64,
    incx: c_int,
    result: *mut f64,
) -> c_int {
    if n > 0 && incx > 0 && x.is_null() {
        return INVALID_VALUE;
    }
    let ty = Ty::Z;
    trace!("{}nrm2 n={n} incx={incx}", ty.tag());
    unsafe {
        reduce(
            handle,
            result.cast(),
            ty.comp() as usize,
            |out| plan::nrm2(ty, n, x as u64, incx, out),
            if incx > 0 { n } else { 0 },
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSaxpy_v2(
    handle: *mut c_void,
    n: c_int,
    alpha: *const f32,
    x: *const f32,
    incx: c_int,
    y: *mut f32,
    incy: c_int,
) -> c_int {
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    let al = tri!(unsafe { scalar(h, alpha.cast(), Ty::S) });
    let launches = tri!(plan::axpy(Ty::S, n, al, x as u64, incx, y as u64, incy));
    if !launches.is_empty() && (x.is_null() || y.is_null()) {
        return INVALID_VALUE;
    }
    run(h, Ok(launches))
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDaxpy_v2(
    handle: *mut c_void,
    n: c_int,
    alpha: *const f64,
    x: *const f64,
    incx: c_int,
    y: *mut f64,
    incy: c_int,
) -> c_int {
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    let al = tri!(unsafe { scalar(h, alpha.cast(), Ty::D) });
    let launches = tri!(plan::axpy(Ty::D, n, al, x as u64, incx, y as u64, incy));
    if !launches.is_empty() && (x.is_null() || y.is_null()) {
        return INVALID_VALUE;
    }
    run(h, Ok(launches))
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCaxpy_v2(
    handle: *mut c_void,
    n: c_int,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: c_int,
    y: *mut Complex32,
    incy: c_int,
) -> c_int {
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    let al = tri!(unsafe { scalar(h, alpha.cast(), Ty::C) });
    let launches = tri!(plan::axpy(Ty::C, n, al, x as u64, incx, y as u64, incy));
    if !launches.is_empty() && (x.is_null() || y.is_null()) {
        return INVALID_VALUE;
    }
    run(h, Ok(launches))
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZaxpy_v2(
    handle: *mut c_void,
    n: c_int,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: c_int,
    y: *mut Complex64,
    incy: c_int,
) -> c_int {
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    let al = tri!(unsafe { scalar(h, alpha.cast(), Ty::Z) });
    let launches = tri!(plan::axpy(Ty::Z, n, al, x as u64, incx, y as u64, incy));
    if !launches.is_empty() && (x.is_null() || y.is_null()) {
        return INVALID_VALUE;
    }
    run(h, Ok(launches))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSscal_v2(
    handle: *mut c_void,
    n: c_int,
    alpha: *const f32,
    x: *mut f32,
    incx: c_int,
) -> c_int {
    let (ty, sty): (Ty, Ty) = (Ty::S, Ty::S);
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    // A real alpha for complex x: read as the real type, imaginary part zero.
    let mut al = tri!(unsafe { scalar(h, alpha.cast(), sty) });
    let real = ty.complex() && !sty.complex();
    if real {
        al.im = 0;
    }
    let launches = tri!(plan::scal(ty, n, al, x as u64, incx, real));
    if !launches.is_empty() && x.is_null() {
        return INVALID_VALUE;
    }
    run(h, Ok(launches))
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDscal_v2(
    handle: *mut c_void,
    n: c_int,
    alpha: *const f64,
    x: *mut f64,
    incx: c_int,
) -> c_int {
    let (ty, sty): (Ty, Ty) = (Ty::D, Ty::D);
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    // A real alpha for complex x: read as the real type, imaginary part zero.
    let mut al = tri!(unsafe { scalar(h, alpha.cast(), sty) });
    let real = ty.complex() && !sty.complex();
    if real {
        al.im = 0;
    }
    let launches = tri!(plan::scal(ty, n, al, x as u64, incx, real));
    if !launches.is_empty() && x.is_null() {
        return INVALID_VALUE;
    }
    run(h, Ok(launches))
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCscal_v2(
    handle: *mut c_void,
    n: c_int,
    alpha: *const Complex32,
    x: *mut Complex32,
    incx: c_int,
) -> c_int {
    let (ty, sty): (Ty, Ty) = (Ty::C, Ty::C);
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    // A real alpha for complex x: read as the real type, imaginary part zero.
    let mut al = tri!(unsafe { scalar(h, alpha.cast(), sty) });
    let real = ty.complex() && !sty.complex();
    if real {
        al.im = 0;
    }
    let launches = tri!(plan::scal(ty, n, al, x as u64, incx, real));
    if !launches.is_empty() && x.is_null() {
        return INVALID_VALUE;
    }
    run(h, Ok(launches))
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZscal_v2(
    handle: *mut c_void,
    n: c_int,
    alpha: *const Complex64,
    x: *mut Complex64,
    incx: c_int,
) -> c_int {
    let (ty, sty): (Ty, Ty) = (Ty::Z, Ty::Z);
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    // A real alpha for complex x: read as the real type, imaginary part zero.
    let mut al = tri!(unsafe { scalar(h, alpha.cast(), sty) });
    let real = ty.complex() && !sty.complex();
    if real {
        al.im = 0;
    }
    let launches = tri!(plan::scal(ty, n, al, x as u64, incx, real));
    if !launches.is_empty() && x.is_null() {
        return INVALID_VALUE;
    }
    run(h, Ok(launches))
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCsscal_v2(
    handle: *mut c_void,
    n: c_int,
    alpha: *const f32,
    x: *mut Complex32,
    incx: c_int,
) -> c_int {
    let (ty, sty): (Ty, Ty) = (Ty::C, Ty::S);
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    // A real alpha for complex x: read as the real type, imaginary part zero.
    let mut al = tri!(unsafe { scalar(h, alpha.cast(), sty) });
    let real = ty.complex() && !sty.complex();
    if real {
        al.im = 0;
    }
    let launches = tri!(plan::scal(ty, n, al, x as u64, incx, real));
    if !launches.is_empty() && x.is_null() {
        return INVALID_VALUE;
    }
    run(h, Ok(launches))
}
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZdscal_v2(
    handle: *mut c_void,
    n: c_int,
    alpha: *const f64,
    x: *mut Complex64,
    incx: c_int,
) -> c_int {
    let (ty, sty): (Ty, Ty) = (Ty::Z, Ty::D);
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    // A real alpha for complex x: read as the real type, imaginary part zero.
    let mut al = tri!(unsafe { scalar(h, alpha.cast(), sty) });
    let real = ty.complex() && !sty.complex();
    if real {
        al.im = 0;
    }
    let launches = tri!(plan::scal(ty, n, al, x as u64, incx, real));
    if !launches.is_empty() && x.is_null() {
        return INVALID_VALUE;
    }
    run(h, Ok(launches))
}

/// The shim runs on every EU it is given; the target is recorded and reported back.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetSmCountTarget(handle: *mut c_void, count: c_int) -> c_int {
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    if count < 0 {
        return INVALID_VALUE;
    }
    h.sm_count_target = count;
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetSmCountTarget(
    handle: *mut c_void,
    count: *mut c_int,
) -> c_int {
    let h = tri!(handle_mut(handle).map_err(|_| NOT_INITIALIZED));
    if count.is_null() {
        return INVALID_VALUE;
    }
    unsafe { count.write(h.sm_count_target) };
    SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};

    fn host_handle() -> *mut c_void {
        Box::into_raw(Box::new(Handle {
            stream: null_mut(),
            math_mode: 0,
            pointer_mode: POINTER_MODE_HOST,
            workspace: (0, 0),
            sm_count_target: 0,
        }))
        .cast()
    }

    #[test]
    fn statuses_without_a_driver() {
        let h = host_handle();
        let one = [1.0f64, 0.0];
        unsafe {
            assert_eq!(
                cublasZgemm_v2(
                    null_mut(),
                    0,
                    0,
                    1,
                    1,
                    1,
                    one.as_ptr().cast(),
                    null(),
                    1,
                    null(),
                    1,
                    one.as_ptr().cast(),
                    null_mut(),
                    1
                ),
                NOT_INITIALIZED
            );
            assert_eq!(
                cublasZgemm_v2(
                    h,
                    5,
                    0,
                    1,
                    1,
                    1,
                    one.as_ptr().cast(),
                    null(),
                    1,
                    null(),
                    1,
                    one.as_ptr().cast(),
                    null_mut(),
                    1
                ),
                INVALID_VALUE
            );
            assert_eq!(
                cublasDgemm_v2(
                    h,
                    0,
                    0,
                    1,
                    1,
                    1,
                    null(),
                    null(),
                    1,
                    null(),
                    1,
                    one.as_ptr(),
                    null_mut(),
                    1
                ),
                INVALID_VALUE
            );
            // empty problems succeed without touching a device
            assert_eq!(
                cublasDgemm_v2(
                    h,
                    0,
                    0,
                    0,
                    4,
                    4,
                    one.as_ptr(),
                    null(),
                    1,
                    null(),
                    4,
                    one.as_ptr(),
                    null_mut(),
                    1
                ),
                SUCCESS
            );
            assert_eq!(
                cublasZgemmStridedBatched(
                    h,
                    0,
                    0,
                    2,
                    2,
                    2,
                    one.as_ptr().cast(),
                    null(),
                    2,
                    4,
                    null(),
                    2,
                    4,
                    one.as_ptr().cast(),
                    null_mut(),
                    2,
                    4,
                    0
                ),
                SUCCESS
            );
            // a non-empty product with null operands is rejected
            assert_eq!(
                cublasDgemm_v2(
                    h,
                    0,
                    0,
                    2,
                    2,
                    2,
                    one.as_ptr(),
                    null(),
                    2,
                    null(),
                    2,
                    one.as_ptr(),
                    null_mut(),
                    2
                ),
                INVALID_VALUE
            );
            assert_eq!(
                cublasDgemv_v2(
                    h,
                    0,
                    2,
                    2,
                    one.as_ptr(),
                    null(),
                    2,
                    null(),
                    0,
                    one.as_ptr(),
                    null_mut(),
                    1
                ),
                INVALID_VALUE
            );
            assert_eq!(
                cublasZtrsm_v2(
                    h,
                    0,
                    0,
                    3,
                    0,
                    2,
                    2,
                    one.as_ptr().cast(),
                    null(),
                    2,
                    null_mut(),
                    2
                ),
                INVALID_VALUE
            );
            assert_eq!(
                cublasDtrsmBatched(h, 0, 0, 0, 0, 2, 2, one.as_ptr(), null(), 2, null(), 2, 0),
                SUCCESS
            );
            let mut r = [7.0f64, 7.0];
            assert_eq!(
                cublasZdotc_v2(h, 0, null(), 1, null(), 1, r.as_mut_ptr().cast()),
                SUCCESS
            );
            assert_eq!(r, [0.0, 0.0]);
            assert_eq!(
                cublasDdot_v2(h, 3, null(), 1, null(), 1, r.as_mut_ptr()),
                INVALID_VALUE
            );
            assert_eq!(cublasDznrm2_v2(h, 4, null(), -1, r.as_mut_ptr()), SUCCESS);
            assert_eq!(r[0], 0.0);
            assert_eq!(cublasZdscal_v2(h, 0, one.as_ptr(), null_mut(), 1), SUCCESS);
            assert_eq!(
                cublasCaxpy_v2(h, 2, null(), null(), 1, null_mut(), 1),
                INVALID_VALUE
            );
            // DotEx type checks
            assert_eq!(
                cublasDotEx(
                    h,
                    1,
                    null(),
                    plan::R_16F,
                    1,
                    null(),
                    plan::R_16F,
                    1,
                    r.as_mut_ptr().cast(),
                    plan::R_16F,
                    plan::R_16F
                ),
                NOT_SUPPORTED
            );
            assert_eq!(
                cublasDotEx(
                    h,
                    1,
                    null(),
                    plan::R_64F,
                    1,
                    null(),
                    plan::R_32F,
                    1,
                    r.as_mut_ptr().cast(),
                    plan::R_64F,
                    plan::R_64F
                ),
                NOT_SUPPORTED
            );
            assert_eq!(
                cublasCgemmEx(
                    h,
                    0,
                    0,
                    1,
                    1,
                    1,
                    null(),
                    null(),
                    4,
                    1,
                    null(),
                    4,
                    1,
                    null(),
                    null_mut(),
                    8,
                    1
                ),
                NOT_SUPPORTED
            );
            let mut c = 9;
            assert_eq!(cublasSetSmCountTarget(h, 12), SUCCESS);
            assert_eq!(cublasGetSmCountTarget(h, &mut c), SUCCESS);
            assert_eq!(c, 12);
            drop(Box::from_raw(h.cast::<Handle>()));
        }
    }

    #[test]
    fn gemm_ex_routes_only_wide_types() {
        let h = host_handle();
        let one = [1.0f64, 0.0];
        let z = Batch::Strided {
            a: 0,
            b: 0,
            c: 0,
            sa: 0,
            sb: 0,
            sc: 0,
        };
        unsafe {
            assert_eq!(
                gemm_ex(
                    h,
                    0,
                    0,
                    1,
                    1,
                    1,
                    one.as_ptr().cast(),
                    plan::R_32F,
                    1,
                    plan::R_32F,
                    1,
                    one.as_ptr().cast(),
                    plan::R_32F,
                    1,
                    68,
                    1,
                    z
                ),
                None
            );
            assert_eq!(
                gemm_ex(
                    h,
                    0,
                    0,
                    1,
                    1,
                    1,
                    one.as_ptr().cast(),
                    plan::R_64F,
                    1,
                    plan::R_64F,
                    1,
                    one.as_ptr().cast(),
                    plan::R_64F,
                    1,
                    68,
                    1,
                    z
                ),
                Some(NOT_SUPPORTED)
            );
            assert_eq!(
                gemm_ex(
                    h,
                    0,
                    0,
                    1,
                    1,
                    1,
                    one.as_ptr().cast(),
                    plan::C_64F,
                    1,
                    plan::C_32F,
                    1,
                    one.as_ptr().cast(),
                    plan::C_64F,
                    1,
                    70,
                    1,
                    z
                ),
                Some(NOT_SUPPORTED)
            );
            assert_eq!(
                gemm_ex(
                    h,
                    0,
                    0,
                    0,
                    1,
                    1,
                    one.as_ptr().cast(),
                    plan::C_64F,
                    1,
                    plan::C_64F,
                    1,
                    one.as_ptr().cast(),
                    plan::C_64F,
                    1,
                    70,
                    1,
                    z
                ),
                Some(SUCCESS)
            );
            drop(Box::from_raw(h.cast::<Handle>()));
        }
    }
}
