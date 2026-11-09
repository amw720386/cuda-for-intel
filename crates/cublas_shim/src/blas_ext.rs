//! cuBLAS routines that NVIDIA's cuSOLVER (`cusolver64_11.dll`, loaded from `torch\lib`
//! under `cfi-run`) and torch's MAGMA call besides gemm and trsm: `copy`, `swap`,
//! `i?amax`/`i?amin`, `syrk`/`herk`, `syrkx`/`herkx`, `syr2k`/`her2k`, `symm`/`hemm`,
//! `trmm`, `trmv`, `trsv`, `symv`/`hemv`, `ger`/`geru`/`gerc`, `geam`, `dgmm`,
//! `getrfBatched`/`getrsBatched` (small batched LU, which torch uses for `linalg.solve`,
//! `inv`, `det` and `lu_factor`), and the host/device matrix and vector copies.
//!
//! All run on the kernels in `kernels::ptxgen`: gemm with triangular, symmetric or
//! Hermitian operand modes and triangle-only stores, plus copy, amax, geam, getrf and laswp.
use super::blas_dcz::{Complex32, Complex64, enqueue, handle, reduce, run, scalar};
use super::kernels::plan::{self as kp, Launch, Scalar};
use super::kernels::ptxgen::Ty;
use super::*;

macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return e,
        }
    };
}

/// A scalar as f64 values: read from the host, or (device pointer mode) after draining the
/// stream. `real` reads one component.
unsafe fn host_value(
    h: &Handle,
    ptr: *const c_void,
    ty: Ty,
    real: bool,
) -> Result<(f64, f64), c_int> {
    if ptr.is_null() {
        return Err(INVALID_VALUE);
    }
    let bytes = if real { ty.comp() } else { ty.size() } as usize;
    let mut buf = [0u8; 16];
    if h.pointer_mode == POINTER_MODE_DEVICE {
        let drv = driver().map_err(|_| NOT_INITIALIZED)?;
        let st = sync_stream(drv, h.stream);
        if st != SUCCESS {
            return Err(st);
        }
        if unsafe { (drv.cuMemcpyDtoH_v2)(buf.as_mut_ptr().cast(), ptr as u64, bytes) } != SUCCESS {
            return Err(EXECUTION_FAILED);
        }
    } else {
        unsafe { std::ptr::copy_nonoverlapping(ptr.cast::<u8>(), buf.as_mut_ptr(), bytes) };
    }
    let c = ty.comp() as usize;
    let comp = |o: usize| {
        if c == 8 {
            f64::from_le_bytes(buf[o..o + 8].try_into().unwrap())
        } else {
            f64::from(f32::from_le_bytes(buf[o..o + 4].try_into().unwrap()))
        }
    };
    Ok((comp(0), if real || !ty.complex() { 0.0 } else { comp(c) }))
}

/// Enqueue, wait, then free the scratch allocations.
fn run_with_scratch(h: &Handle, launches: Vec<Launch>, scratch: &[u64]) -> c_int {
    let Ok(drv) = driver() else {
        return NOT_INITIALIZED;
    };
    let mut st = enqueue(drv, h, &launches);
    let sync = sync_stream(drv, h.stream);
    if st == SUCCESS {
        st = sync;
    }
    for &s in scratch {
        unsafe { (drv.cuMemFree_v2)(s) };
    }
    st
}

fn alloc(bytes: usize) -> Result<u64, c_int> {
    let drv = driver().map_err(|_| NOT_INITIALIZED)?;
    let mut d = 0u64;
    if unsafe { (drv.cuMemAlloc_v2)(&mut d, bytes.max(16)) } != SUCCESS {
        return Err(ALLOC_FAILED);
    }
    Ok(d)
}

#[allow(clippy::too_many_arguments)]
unsafe fn copy(
    hd: *mut c_void,
    ty: Ty,
    n: c_int,
    x: u64,
    incx: c_int,
    y: u64,
    incy: c_int,
    swap: bool,
) -> c_int {
    let h = tri!(handle(hd));
    let l = tri!(kp::copy(ty, n, x, incx, y, incy, swap));
    if !l.is_empty() && (x == 0 || y == 0) {
        return INVALID_VALUE;
    }
    run(h, Ok(l))
}

#[allow(clippy::too_many_arguments)]
unsafe fn amax(
    hd: *mut c_void,
    ty: Ty,
    n: c_int,
    x: u64,
    incx: c_int,
    result: *mut c_int,
    min: bool,
) -> c_int {
    if result.is_null() {
        return INVALID_VALUE;
    }
    if handle(hd).is_err() {
        return NOT_INITIALIZED;
    }
    let live = if incx > 0 { n } else { 0 };
    if live > 0 && x == 0 {
        return INVALID_VALUE;
    }
    unsafe {
        reduce(
            hd,
            result.cast(),
            4,
            |out| kp::amax(ty, n, x, incx, out, min),
            live,
        )
    }
}

/// How a rank-k routine takes its scalars.
#[derive(Clone, Copy, PartialEq)]
enum Herm {
    /// `syrk`/`syrkx`: alpha and beta of the element type.
    No,
    /// `herk`: real alpha and beta.
    RealAlpha,
    /// `herkx`: complex alpha, real beta.
    ComplexAlpha,
}

#[allow(clippy::too_many_arguments)]
unsafe fn rank_k(
    hd: *mut c_void,
    ty: Ty,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: u64,
    lda: c_int,
    b: u64,
    ldb: c_int,
    beta: *const c_void,
    c: u64,
    ldc: c_int,
    herm: Herm,
) -> c_int {
    let h = tri!(handle(hd));
    let (al, be) = if herm == Herm::No {
        (
            tri!(unsafe { scalar(h, alpha, ty) }),
            tri!(unsafe { scalar(h, beta, ty) }),
        )
    } else {
        let (ar, ai) = tri!(unsafe { host_value(h, alpha, ty, herm == Herm::RealAlpha) });
        let (br, _) = tri!(unsafe { host_value(h, beta, ty, true) });
        (Scalar::host(ty, ar, ai), Scalar::host(ty, br, 0.0))
    };
    let l = tri!(kp::rank_k(
        ty,
        uplo,
        trans,
        n,
        k,
        a,
        lda,
        b,
        ldb,
        c,
        ldc,
        al,
        be,
        herm != Herm::No
    ));
    if !l.is_empty() && (c == 0 || (k > 0 && (a == 0 || b == 0))) {
        return INVALID_VALUE;
    }
    trace!("{}rank-k uplo={uplo} trans={trans} n={n} k={k}", ty.tag());
    run(h, Ok(l))
}

#[allow(clippy::too_many_arguments)]
unsafe fn rank_2k(
    hd: *mut c_void,
    ty: Ty,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const c_void,
    a: u64,
    lda: c_int,
    b: u64,
    ldb: c_int,
    beta: *const c_void,
    c: u64,
    ldc: c_int,
    herm: bool,
) -> c_int {
    let h = tri!(handle(hd));
    let (al, al2, be) = if herm {
        let (ar, ai) = tri!(unsafe { host_value(h, alpha, ty, false) });
        let (br, _) = tri!(unsafe { host_value(h, beta, ty, true) });
        (
            Scalar::host(ty, ar, ai),
            Scalar::host(ty, ar, -ai),
            Scalar::host(ty, br, 0.0),
        )
    } else {
        let al = tri!(unsafe { scalar(h, alpha, ty) });
        (al, al, tri!(unsafe { scalar(h, beta, ty) }))
    };
    let l = tri!(kp::rank_2k(
        ty, uplo, trans, n, k, a, lda, b, ldb, c, ldc, al, al2, be, herm
    ));
    if !l.is_empty() && (c == 0 || (k > 0 && (a == 0 || b == 0))) {
        return INVALID_VALUE;
    }
    trace!("{}rank-2k uplo={uplo} trans={trans} n={n} k={k}", ty.tag());
    run(h, Ok(l))
}

#[allow(clippy::too_many_arguments)]
unsafe fn symm(
    hd: *mut c_void,
    ty: Ty,
    side: c_int,
    uplo: c_int,
    m: c_int,
    n: c_int,
    alpha: *const c_void,
    a: u64,
    lda: c_int,
    b: u64,
    ldb: c_int,
    beta: *const c_void,
    c: u64,
    ldc: c_int,
    herm: bool,
) -> c_int {
    let h = tri!(handle(hd));
    let al = tri!(unsafe { scalar(h, alpha, ty) });
    let be = tri!(unsafe { scalar(h, beta, ty) });
    let l = tri!(kp::symm(
        ty, side, uplo, m, n, a, lda, b, ldb, c, ldc, al, be, herm
    ));
    if !l.is_empty() && (a == 0 || b == 0 || c == 0) {
        return INVALID_VALUE;
    }
    run(h, Ok(l))
}

#[allow(clippy::too_many_arguments)]
unsafe fn trmm(
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
    c: u64,
    ldc: c_int,
) -> c_int {
    let h = tri!(handle(hd));
    let al = tri!(unsafe { scalar(h, alpha, ty) });
    // validate before touching memory
    let l = tri!(kp::trmm(
        ty, side, uplo, trans, diag, m, n, a, lda, b, ldb, c, ldc, al
    ));
    if l.is_empty() {
        return SUCCESS;
    }
    if a == 0 || b == 0 || c == 0 {
        return INVALID_VALUE;
    }
    let es = u64::from(ty.size());
    let (bm, cm) = (
        (n as u64 - 1) * ldb as u64 + m as u64,
        (n as u64 - 1) * ldc as u64 + m as u64,
    );
    let overlap = b < c + cm * es && c < b + bm * es;
    if !overlap {
        return run(h, Ok(l));
    }
    // in place (MAGMA and cuSOLVER pass C == B): multiply from a copy of B
    let tmp = tri!(alloc(m as usize * n as usize * es as usize));
    let one = Scalar::host(ty, 1.0, 0.0);
    let mut all = tri!(kp::geam(
        ty,
        kp::OP_N,
        kp::OP_N,
        m,
        n,
        b,
        ldb,
        0,
        m.max(1),
        tmp,
        m.max(1),
        one,
        Scalar::host(ty, 0.0, 0.0)
    ));
    all.extend(tri!(kp::trmm(
        ty,
        side,
        uplo,
        trans,
        diag,
        m,
        n,
        a,
        lda,
        tmp,
        m.max(1),
        c,
        ldc,
        al
    )));
    run_with_scratch(h, all, &[tmp])
}

#[allow(clippy::too_many_arguments)]
unsafe fn trmv(
    hd: *mut c_void,
    ty: Ty,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: u64,
    lda: c_int,
    x: u64,
    incx: c_int,
) -> c_int {
    let h = tri!(handle(hd));
    // validate (with a dummy copy address) before allocating
    let l = tri!(kp::trmv(ty, uplo, trans, diag, n, a, lda, 0, x, incx));
    if l.is_empty() {
        return SUCCESS;
    }
    if a == 0 || x == 0 {
        return INVALID_VALUE;
    }
    let tmp = tri!(alloc(n as usize * ty.size() as usize));
    let mut all = tri!(kp::copy(ty, n, x, incx, tmp, 1, false));
    all.extend(tri!(kp::trmv(
        ty, uplo, trans, diag, n, a, lda, tmp, x, incx
    )));
    run_with_scratch(h, all, &[tmp])
}

#[allow(clippy::too_many_arguments)]
unsafe fn trsv(
    hd: *mut c_void,
    ty: Ty,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: u64,
    lda: c_int,
    x: u64,
    incx: c_int,
) -> c_int {
    let h = tri!(handle(hd));
    if incx == 0 {
        return INVALID_VALUE;
    }
    let mut p = tri!(kp::tri(
        ty,
        kp::SIDE_LEFT,
        uplo,
        trans,
        diag,
        n,
        1,
        lda,
        n.max(1)
    ));
    (p.b_vs, p.b_es) = (0, i64::from(incx) as u64);
    let base = kp::vec_base(x, i64::from(n), i64::from(incx), ty);
    let l = tri!(kp::trsm(
        ty,
        &p,
        Scalar::host(ty, 1.0, 0.0),
        a,
        base,
        1,
        false
    ));
    if !l.is_empty() && (a == 0 || x == 0) {
        return INVALID_VALUE;
    }
    run(h, Ok(l))
}

#[allow(clippy::too_many_arguments)]
unsafe fn symv(
    hd: *mut c_void,
    ty: Ty,
    uplo: c_int,
    n: c_int,
    alpha: *const c_void,
    a: u64,
    lda: c_int,
    x: u64,
    incx: c_int,
    beta: *const c_void,
    y: u64,
    incy: c_int,
    herm: bool,
) -> c_int {
    let h = tri!(handle(hd));
    let al = tri!(unsafe { scalar(h, alpha, ty) });
    let be = tri!(unsafe { scalar(h, beta, ty) });
    let l = tri!(kp::symv(
        ty, uplo, n, a, lda, x, incx, y, incy, al, be, herm
    ));
    if !l.is_empty() && (a == 0 || x == 0 || y == 0) {
        return INVALID_VALUE;
    }
    run(h, Ok(l))
}

#[allow(clippy::too_many_arguments)]
unsafe fn ger(
    hd: *mut c_void,
    ty: Ty,
    m: c_int,
    n: c_int,
    alpha: *const c_void,
    x: u64,
    incx: c_int,
    y: u64,
    incy: c_int,
    a: u64,
    lda: c_int,
    conj: bool,
) -> c_int {
    let h = tri!(handle(hd));
    let al = tri!(unsafe { scalar(h, alpha, ty) });
    let l = tri!(kp::ger(ty, m, n, x, incx, y, incy, a, lda, al, conj));
    if !l.is_empty() && (a == 0 || x == 0 || y == 0) {
        return INVALID_VALUE;
    }
    run(h, Ok(l))
}

#[allow(clippy::too_many_arguments)]
unsafe fn geam(
    hd: *mut c_void,
    ty: Ty,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    alpha: *const c_void,
    a: u64,
    lda: c_int,
    beta: *const c_void,
    b: u64,
    ldb: c_int,
    c: u64,
    ldc: c_int,
) -> c_int {
    let h = tri!(handle(hd));
    let al = tri!(unsafe { scalar(h, alpha, ty) });
    let be = tri!(unsafe { scalar(h, beta, ty) });
    let l = tri!(kp::geam(
        ty, transa, transb, m, n, a, lda, b, ldb, c, ldc, al, be
    ));
    if !l.is_empty() && c == 0 {
        return INVALID_VALUE;
    }
    run(h, Ok(l))
}

#[allow(clippy::too_many_arguments)]
unsafe fn dgmm(
    hd: *mut c_void,
    ty: Ty,
    side: c_int,
    m: c_int,
    n: c_int,
    a: u64,
    lda: c_int,
    x: u64,
    incx: c_int,
    c: u64,
    ldc: c_int,
) -> c_int {
    let h = tri!(handle(hd));
    let l = tri!(kp::dgmm(ty, side, m, n, a, lda, x, incx, c, ldc));
    if !l.is_empty() && (a == 0 || x == 0 || c == 0) {
        return INVALID_VALUE;
    }
    run(h, Ok(l))
}

#[allow(clippy::too_many_arguments)]
unsafe fn getrf_batched(
    hd: *mut c_void,
    ty: Ty,
    n: c_int,
    a: u64,
    lda: c_int,
    p: u64,
    info: u64,
    batch: c_int,
) -> c_int {
    let h = tri!(handle(hd));
    let l = tri!(kp::getrf(ty, n, a, lda, p, info, batch));
    if !l.is_empty() && (a == 0 || info == 0) {
        return INVALID_VALUE;
    }
    trace!(
        "{}getrfBatched n={n} lda={lda} pivot={} batch={batch}",
        ty.tag(),
        p != 0
    );
    run(h, Ok(l))
}

#[allow(clippy::too_many_arguments)]
unsafe fn getrs_batched(
    hd: *mut c_void,
    ty: Ty,
    trans: c_int,
    n: c_int,
    nrhs: c_int,
    a: u64,
    lda: c_int,
    p: u64,
    b: u64,
    ldb: c_int,
    info: *mut c_int,
    batch: c_int,
) -> c_int {
    let h = tri!(handle(hd));
    if info.is_null() {
        return INVALID_VALUE;
    }
    let l = match kp::getrs(ty, trans, n, nrhs, a, lda, p, b, ldb, batch) {
        Ok(l) => l,
        Err(i) => {
            unsafe { info.write(i) };
            return INVALID_VALUE;
        }
    };
    unsafe { info.write(0) };
    if !l.is_empty() && (a == 0 || b == 0) {
        return INVALID_VALUE;
    }
    trace!(
        "{}getrsBatched trans={trans} n={n} nrhs={nrhs} batch={batch}",
        ty.tag()
    );
    run(h, Ok(l))
}

#[derive(Clone, Copy, PartialEq)]
enum Dir {
    ToDevice,
    ToHost,
}

/// Column-major host <-> device copy of a rows x cols matrix (`cublasSetMatrix` and
/// friends). Strided device sides move their whole span in one transfer (read, patch and
/// write back for uploads). The Async forms drain their stream first and then copy.
#[allow(clippy::too_many_arguments)]
unsafe fn transfer(
    dir: Dir,
    rows: c_int,
    cols: c_int,
    es: c_int,
    src: usize,
    lds: c_int,
    dst: usize,
    ldd: c_int,
    stream: Option<CUstream>,
) -> c_int {
    if rows < 0 || cols < 0 || es <= 0 || lds < rows.max(1) || ldd < rows.max(1) {
        return INVALID_VALUE;
    }
    if rows == 0 || cols == 0 {
        return SUCCESS;
    }
    if src == 0 || dst == 0 {
        return INVALID_VALUE;
    }
    let Ok(drv) = driver() else {
        return NOT_INITIALIZED;
    };
    if let Some(s) = stream {
        let st = sync_stream(drv, s);
        if st != SUCCESS {
            return st;
        }
    }
    let (rows, cols, es) = (rows as usize, cols as usize, es as usize);
    let (lds, ldd) = (lds as usize, ldd as usize);
    let col = rows * es;
    let (host, hld, dev, dld) = if dir == Dir::ToDevice {
        (src, lds, dst as u64, ldd)
    } else {
        (dst, ldd, src as u64, lds)
    };
    let span = ((cols - 1) * dld + rows) * es;
    let mut buf = vec![0u8; span];
    let ok = |e: CuResult| e == SUCCESS;
    unsafe {
        if dir == Dir::ToDevice {
            if dld != rows && !ok((drv.cuMemcpyDtoH_v2)(buf.as_mut_ptr().cast(), dev, span)) {
                return MAPPING_ERROR;
            }
            for j in 0..cols {
                std::ptr::copy_nonoverlapping(
                    (host as *const u8).add(j * hld * es),
                    buf.as_mut_ptr().add(j * dld * es),
                    col,
                );
            }
            if !ok((drv.cuMemcpyHtoD_v2)(dev, buf.as_ptr().cast(), span)) {
                return MAPPING_ERROR;
            }
        } else {
            if !ok((drv.cuMemcpyDtoH_v2)(buf.as_mut_ptr().cast(), dev, span)) {
                return MAPPING_ERROR;
            }
            for j in 0..cols {
                std::ptr::copy_nonoverlapping(
                    buf.as_ptr().add(j * dld * es),
                    (host as *mut u8).add(j * hld * es),
                    col,
                );
            }
        }
    }
    SUCCESS
}

/// Strided vector copy: a 1 x n matrix whose leading dimensions are the increments.
#[allow(clippy::too_many_arguments)]
unsafe fn transfer_vec(
    dir: Dir,
    n: c_int,
    es: c_int,
    x: usize,
    incx: c_int,
    y: usize,
    incy: c_int,
    stream: Option<CUstream>,
) -> c_int {
    if incx <= 0 || incy <= 0 {
        return INVALID_VALUE;
    }
    unsafe { transfer(dir, 1, n, es, x, incx, y, incy, stream) }
}

// ---------------------------------------------------------------------------
// Exports (one function per cuBLAS name)
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasScopy_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const f32,
    incx: c_int,
    y: *mut f32,
    incy: c_int,
) -> c_int {
    unsafe { copy(handle, Ty::S, n, x as u64, incx, y as u64, incy, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSswap_v2(
    handle: *mut c_void,
    n: c_int,
    x: *mut f32,
    incx: c_int,
    y: *mut f32,
    incy: c_int,
) -> c_int {
    unsafe { copy(handle, Ty::S, n, x as u64, incx, y as u64, incy, true) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDcopy_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const f64,
    incx: c_int,
    y: *mut f64,
    incy: c_int,
) -> c_int {
    unsafe { copy(handle, Ty::D, n, x as u64, incx, y as u64, incy, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDswap_v2(
    handle: *mut c_void,
    n: c_int,
    x: *mut f64,
    incx: c_int,
    y: *mut f64,
    incy: c_int,
) -> c_int {
    unsafe { copy(handle, Ty::D, n, x as u64, incx, y as u64, incy, true) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCcopy_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex32,
    incx: c_int,
    y: *mut Complex32,
    incy: c_int,
) -> c_int {
    unsafe { copy(handle, Ty::C, n, x as u64, incx, y as u64, incy, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCswap_v2(
    handle: *mut c_void,
    n: c_int,
    x: *mut Complex32,
    incx: c_int,
    y: *mut Complex32,
    incy: c_int,
) -> c_int {
    unsafe { copy(handle, Ty::C, n, x as u64, incx, y as u64, incy, true) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZcopy_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex64,
    incx: c_int,
    y: *mut Complex64,
    incy: c_int,
) -> c_int {
    unsafe { copy(handle, Ty::Z, n, x as u64, incx, y as u64, incy, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZswap_v2(
    handle: *mut c_void,
    n: c_int,
    x: *mut Complex64,
    incx: c_int,
    y: *mut Complex64,
    incy: c_int,
) -> c_int {
    unsafe { copy(handle, Ty::Z, n, x as u64, incx, y as u64, incy, true) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasIsamax_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const f32,
    incx: c_int,
    result: *mut c_int,
) -> c_int {
    unsafe { amax(handle, Ty::S, n, x as u64, incx, result, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasIsamin_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const f32,
    incx: c_int,
    result: *mut c_int,
) -> c_int {
    unsafe { amax(handle, Ty::S, n, x as u64, incx, result, true) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasIdamax_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const f64,
    incx: c_int,
    result: *mut c_int,
) -> c_int {
    unsafe { amax(handle, Ty::D, n, x as u64, incx, result, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasIdamin_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const f64,
    incx: c_int,
    result: *mut c_int,
) -> c_int {
    unsafe { amax(handle, Ty::D, n, x as u64, incx, result, true) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasIcamax_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex32,
    incx: c_int,
    result: *mut c_int,
) -> c_int {
    unsafe { amax(handle, Ty::C, n, x as u64, incx, result, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasIcamin_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex32,
    incx: c_int,
    result: *mut c_int,
) -> c_int {
    unsafe { amax(handle, Ty::C, n, x as u64, incx, result, true) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasIzamax_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex64,
    incx: c_int,
    result: *mut c_int,
) -> c_int {
    unsafe { amax(handle, Ty::Z, n, x as u64, incx, result, false) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasIzamin_v2(
    handle: *mut c_void,
    n: c_int,
    x: *const Complex64,
    incx: c_int,
    result: *mut c_int,
) -> c_int {
    unsafe { amax(handle, Ty::Z, n, x as u64, incx, result, true) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSsyrk_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_k(
            handle,
            Ty::S,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            a as u64,
            lda,
            beta.cast(),
            c as u64,
            ldc,
            Herm::No,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSsyrkx(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_k(
            handle,
            Ty::S,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            Herm::No,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSsyr2k_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_2k(
            handle,
            Ty::S,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDsyrk_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_k(
            handle,
            Ty::D,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            a as u64,
            lda,
            beta.cast(),
            c as u64,
            ldc,
            Herm::No,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDsyrkx(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
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
    unsafe {
        rank_k(
            handle,
            Ty::D,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            Herm::No,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDsyr2k_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
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
    unsafe {
        rank_2k(
            handle,
            Ty::D,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCsyrk_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_k(
            handle,
            Ty::C,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            a as u64,
            lda,
            beta.cast(),
            c as u64,
            ldc,
            Herm::No,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCsyrkx(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
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
    unsafe {
        rank_k(
            handle,
            Ty::C,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            Herm::No,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCsyr2k_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
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
    unsafe {
        rank_2k(
            handle,
            Ty::C,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZsyrk_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: c_int,
    beta: *const Complex64,
    c: *mut Complex64,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_k(
            handle,
            Ty::Z,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            a as u64,
            lda,
            beta.cast(),
            c as u64,
            ldc,
            Herm::No,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZsyrkx(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
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
        rank_k(
            handle,
            Ty::Z,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            Herm::No,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZsyr2k_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
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
        rank_2k(
            handle,
            Ty::Z,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCherk_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f32,
    a: *const Complex32,
    lda: c_int,
    beta: *const f32,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_k(
            handle,
            Ty::C,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            a as u64,
            lda,
            beta.cast(),
            c as u64,
            ldc,
            Herm::RealAlpha,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCherkx(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    b: *const Complex32,
    ldb: c_int,
    beta: *const f32,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_k(
            handle,
            Ty::C,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            Herm::ComplexAlpha,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCher2k_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    b: *const Complex32,
    ldb: c_int,
    beta: *const f32,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_2k(
            handle,
            Ty::C,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            true,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZherk_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const f64,
    a: *const Complex64,
    lda: c_int,
    beta: *const f64,
    c: *mut Complex64,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_k(
            handle,
            Ty::Z,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            a as u64,
            lda,
            beta.cast(),
            c as u64,
            ldc,
            Herm::RealAlpha,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZherkx(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: c_int,
    b: *const Complex64,
    ldb: c_int,
    beta: *const f64,
    c: *mut Complex64,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_k(
            handle,
            Ty::Z,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            Herm::ComplexAlpha,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZher2k_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    n: c_int,
    k: c_int,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: c_int,
    b: *const Complex64,
    ldb: c_int,
    beta: *const f64,
    c: *mut Complex64,
    ldc: c_int,
) -> c_int {
    unsafe {
        rank_2k(
            handle,
            Ty::Z,
            uplo,
            trans,
            n,
            k,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            true,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSsymm_v2(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    b: *const f32,
    ldb: c_int,
    beta: *const f32,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    unsafe {
        symm(
            handle,
            Ty::S,
            side,
            uplo,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDsymm_v2(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    b: *const f64,
    ldb: c_int,
    beta: *const f64,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    unsafe {
        symm(
            handle,
            Ty::D,
            side,
            uplo,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCsymm_v2(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    b: *const Complex32,
    ldb: c_int,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    unsafe {
        symm(
            handle,
            Ty::C,
            side,
            uplo,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZsymm_v2(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    m: c_int,
    n: c_int,
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
        symm(
            handle,
            Ty::Z,
            side,
            uplo,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasChemm_v2(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    b: *const Complex32,
    ldb: c_int,
    beta: *const Complex32,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    unsafe {
        symm(
            handle,
            Ty::C,
            side,
            uplo,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            true,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZhemm_v2(
    handle: *mut c_void,
    side: c_int,
    uplo: c_int,
    m: c_int,
    n: c_int,
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
        symm(
            handle,
            Ty::Z,
            side,
            uplo,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            b as u64,
            ldb,
            beta.cast(),
            c as u64,
            ldc,
            true,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasStrmm_v2(
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
    b: *const f32,
    ldb: c_int,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    unsafe {
        trmm(
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
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasStrmv_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
) -> c_int {
    unsafe {
        trmv(
            handle,
            Ty::S,
            uplo,
            trans,
            diag,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasStrsv_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
) -> c_int {
    unsafe {
        trsv(
            handle,
            Ty::S,
            uplo,
            trans,
            diag,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDtrmm_v2(
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
    b: *const f64,
    ldb: c_int,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    unsafe {
        trmm(
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
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDtrmv_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
) -> c_int {
    unsafe {
        trmv(
            handle,
            Ty::D,
            uplo,
            trans,
            diag,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDtrsv_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
) -> c_int {
    unsafe {
        trsv(
            handle,
            Ty::D,
            uplo,
            trans,
            diag,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCtrmm_v2(
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
    b: *const Complex32,
    ldb: c_int,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    unsafe {
        trmm(
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
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCtrmv_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: *const Complex32,
    lda: c_int,
    x: *mut Complex32,
    incx: c_int,
) -> c_int {
    unsafe {
        trmv(
            handle,
            Ty::C,
            uplo,
            trans,
            diag,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCtrsv_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: *const Complex32,
    lda: c_int,
    x: *mut Complex32,
    incx: c_int,
) -> c_int {
    unsafe {
        trsv(
            handle,
            Ty::C,
            uplo,
            trans,
            diag,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZtrmm_v2(
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
    b: *const Complex64,
    ldb: c_int,
    c: *mut Complex64,
    ldc: c_int,
) -> c_int {
    unsafe {
        trmm(
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
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZtrmv_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: *const Complex64,
    lda: c_int,
    x: *mut Complex64,
    incx: c_int,
) -> c_int {
    unsafe {
        trmv(
            handle,
            Ty::Z,
            uplo,
            trans,
            diag,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZtrsv_v2(
    handle: *mut c_void,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    n: c_int,
    a: *const Complex64,
    lda: c_int,
    x: *mut Complex64,
    incx: c_int,
) -> c_int {
    unsafe {
        trsv(
            handle,
            Ty::Z,
            uplo,
            trans,
            diag,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSsymv_v2(
    handle: *mut c_void,
    uplo: c_int,
    n: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    beta: *const f32,
    y: *mut f32,
    incy: c_int,
) -> c_int {
    unsafe {
        symv(
            handle,
            Ty::S,
            uplo,
            n,
            alpha.cast(),
            a as u64,
            lda,
            x as u64,
            incx,
            beta.cast(),
            y as u64,
            incy,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDsymv_v2(
    handle: *mut c_void,
    uplo: c_int,
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
        symv(
            handle,
            Ty::D,
            uplo,
            n,
            alpha.cast(),
            a as u64,
            lda,
            x as u64,
            incx,
            beta.cast(),
            y as u64,
            incy,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCsymv_v2(
    handle: *mut c_void,
    uplo: c_int,
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
        symv(
            handle,
            Ty::C,
            uplo,
            n,
            alpha.cast(),
            a as u64,
            lda,
            x as u64,
            incx,
            beta.cast(),
            y as u64,
            incy,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZsymv_v2(
    handle: *mut c_void,
    uplo: c_int,
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
        symv(
            handle,
            Ty::Z,
            uplo,
            n,
            alpha.cast(),
            a as u64,
            lda,
            x as u64,
            incx,
            beta.cast(),
            y as u64,
            incy,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasChemv_v2(
    handle: *mut c_void,
    uplo: c_int,
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
        symv(
            handle,
            Ty::C,
            uplo,
            n,
            alpha.cast(),
            a as u64,
            lda,
            x as u64,
            incx,
            beta.cast(),
            y as u64,
            incy,
            true,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZhemv_v2(
    handle: *mut c_void,
    uplo: c_int,
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
        symv(
            handle,
            Ty::Z,
            uplo,
            n,
            alpha.cast(),
            a as u64,
            lda,
            x as u64,
            incx,
            beta.cast(),
            y as u64,
            incy,
            true,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSger_v2(
    handle: *mut c_void,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    x: *const f32,
    incx: c_int,
    y: *const f32,
    incy: c_int,
    a: *mut f32,
    lda: c_int,
) -> c_int {
    unsafe {
        ger(
            handle,
            Ty::S,
            m,
            n,
            alpha.cast(),
            x as u64,
            incx,
            y as u64,
            incy,
            a as u64,
            lda,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDger_v2(
    handle: *mut c_void,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    x: *const f64,
    incx: c_int,
    y: *const f64,
    incy: c_int,
    a: *mut f64,
    lda: c_int,
) -> c_int {
    unsafe {
        ger(
            handle,
            Ty::D,
            m,
            n,
            alpha.cast(),
            x as u64,
            incx,
            y as u64,
            incy,
            a as u64,
            lda,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgeru_v2(
    handle: *mut c_void,
    m: c_int,
    n: c_int,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: c_int,
    y: *const Complex32,
    incy: c_int,
    a: *mut Complex32,
    lda: c_int,
) -> c_int {
    unsafe {
        ger(
            handle,
            Ty::C,
            m,
            n,
            alpha.cast(),
            x as u64,
            incx,
            y as u64,
            incy,
            a as u64,
            lda,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgerc_v2(
    handle: *mut c_void,
    m: c_int,
    n: c_int,
    alpha: *const Complex32,
    x: *const Complex32,
    incx: c_int,
    y: *const Complex32,
    incy: c_int,
    a: *mut Complex32,
    lda: c_int,
) -> c_int {
    unsafe {
        ger(
            handle,
            Ty::C,
            m,
            n,
            alpha.cast(),
            x as u64,
            incx,
            y as u64,
            incy,
            a as u64,
            lda,
            true,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZgeru_v2(
    handle: *mut c_void,
    m: c_int,
    n: c_int,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: c_int,
    y: *const Complex64,
    incy: c_int,
    a: *mut Complex64,
    lda: c_int,
) -> c_int {
    unsafe {
        ger(
            handle,
            Ty::Z,
            m,
            n,
            alpha.cast(),
            x as u64,
            incx,
            y as u64,
            incy,
            a as u64,
            lda,
            false,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZgerc_v2(
    handle: *mut c_void,
    m: c_int,
    n: c_int,
    alpha: *const Complex64,
    x: *const Complex64,
    incx: c_int,
    y: *const Complex64,
    incy: c_int,
    a: *mut Complex64,
    lda: c_int,
) -> c_int {
    unsafe {
        ger(
            handle,
            Ty::Z,
            m,
            n,
            alpha.cast(),
            x as u64,
            incx,
            y as u64,
            incy,
            a as u64,
            lda,
            true,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSgeam(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    alpha: *const f32,
    a: *const f32,
    lda: c_int,
    beta: *const f32,
    b: *const f32,
    ldb: c_int,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    unsafe {
        geam(
            handle,
            Ty::S,
            transa,
            transb,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            beta.cast(),
            b as u64,
            ldb,
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSdgmm(
    handle: *mut c_void,
    side: c_int,
    m: c_int,
    n: c_int,
    a: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    c: *mut f32,
    ldc: c_int,
) -> c_int {
    unsafe {
        dgmm(
            handle,
            Ty::S,
            side,
            m,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDgeam(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    alpha: *const f64,
    a: *const f64,
    lda: c_int,
    beta: *const f64,
    b: *const f64,
    ldb: c_int,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    unsafe {
        geam(
            handle,
            Ty::D,
            transa,
            transb,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            beta.cast(),
            b as u64,
            ldb,
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDdgmm(
    handle: *mut c_void,
    side: c_int,
    m: c_int,
    n: c_int,
    a: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    c: *mut f64,
    ldc: c_int,
) -> c_int {
    unsafe {
        dgmm(
            handle,
            Ty::D,
            side,
            m,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgeam(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex32,
    a: *const Complex32,
    lda: c_int,
    beta: *const Complex32,
    b: *const Complex32,
    ldb: c_int,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    unsafe {
        geam(
            handle,
            Ty::C,
            transa,
            transb,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            beta.cast(),
            b as u64,
            ldb,
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCdgmm(
    handle: *mut c_void,
    side: c_int,
    m: c_int,
    n: c_int,
    a: *const Complex32,
    lda: c_int,
    x: *const Complex32,
    incx: c_int,
    c: *mut Complex32,
    ldc: c_int,
) -> c_int {
    unsafe {
        dgmm(
            handle,
            Ty::C,
            side,
            m,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZgeam(
    handle: *mut c_void,
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    alpha: *const Complex64,
    a: *const Complex64,
    lda: c_int,
    beta: *const Complex64,
    b: *const Complex64,
    ldb: c_int,
    c: *mut Complex64,
    ldc: c_int,
) -> c_int {
    unsafe {
        geam(
            handle,
            Ty::Z,
            transa,
            transb,
            m,
            n,
            alpha.cast(),
            a as u64,
            lda,
            beta.cast(),
            b as u64,
            ldb,
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZdgmm(
    handle: *mut c_void,
    side: c_int,
    m: c_int,
    n: c_int,
    a: *const Complex64,
    lda: c_int,
    x: *const Complex64,
    incx: c_int,
    c: *mut Complex64,
    ldc: c_int,
) -> c_int {
    unsafe {
        dgmm(
            handle,
            Ty::Z,
            side,
            m,
            n,
            a as u64,
            lda,
            x as u64,
            incx,
            c as u64,
            ldc,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetMatrix(
    rows: c_int,
    cols: c_int,
    elem_size: c_int,
    a: *const c_void,
    lda: c_int,
    b: *mut c_void,
    ldb: c_int,
) -> c_int {
    unsafe {
        transfer(
            Dir::ToDevice,
            rows,
            cols,
            elem_size,
            a as usize,
            lda,
            b as usize,
            ldb,
            None,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetMatrix(
    rows: c_int,
    cols: c_int,
    elem_size: c_int,
    a: *const c_void,
    lda: c_int,
    b: *mut c_void,
    ldb: c_int,
) -> c_int {
    unsafe {
        transfer(
            Dir::ToHost,
            rows,
            cols,
            elem_size,
            a as usize,
            lda,
            b as usize,
            ldb,
            None,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetMatrixAsync(
    rows: c_int,
    cols: c_int,
    elem_size: c_int,
    a: *const c_void,
    lda: c_int,
    b: *mut c_void,
    ldb: c_int,
    stream: CUstream,
) -> c_int {
    unsafe {
        transfer(
            Dir::ToDevice,
            rows,
            cols,
            elem_size,
            a as usize,
            lda,
            b as usize,
            ldb,
            Some(stream),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetMatrixAsync(
    rows: c_int,
    cols: c_int,
    elem_size: c_int,
    a: *const c_void,
    lda: c_int,
    b: *mut c_void,
    ldb: c_int,
    stream: CUstream,
) -> c_int {
    unsafe {
        transfer(
            Dir::ToHost,
            rows,
            cols,
            elem_size,
            a as usize,
            lda,
            b as usize,
            ldb,
            Some(stream),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetVector(
    n: c_int,
    elem_size: c_int,
    x: *const c_void,
    incx: c_int,
    y: *mut c_void,
    incy: c_int,
) -> c_int {
    unsafe {
        transfer_vec(
            Dir::ToDevice,
            n,
            elem_size,
            x as usize,
            incx,
            y as usize,
            incy,
            None,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetVector(
    n: c_int,
    elem_size: c_int,
    x: *const c_void,
    incx: c_int,
    y: *mut c_void,
    incy: c_int,
) -> c_int {
    unsafe {
        transfer_vec(
            Dir::ToHost,
            n,
            elem_size,
            x as usize,
            incx,
            y as usize,
            incy,
            None,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSetVectorAsync(
    n: c_int,
    elem_size: c_int,
    x: *const c_void,
    incx: c_int,
    y: *mut c_void,
    incy: c_int,
    stream: CUstream,
) -> c_int {
    unsafe {
        transfer_vec(
            Dir::ToDevice,
            n,
            elem_size,
            x as usize,
            incx,
            y as usize,
            incy,
            Some(stream),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasGetVectorAsync(
    n: c_int,
    elem_size: c_int,
    x: *const c_void,
    incx: c_int,
    y: *mut c_void,
    incy: c_int,
    stream: CUstream,
) -> c_int {
    unsafe {
        transfer_vec(
            Dir::ToHost,
            n,
            elem_size,
            x as usize,
            incx,
            y as usize,
            incy,
            Some(stream),
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSgetrfBatched(
    handle: *mut c_void,
    n: c_int,
    a: *const *mut f32,
    lda: c_int,
    p: *mut c_int,
    info: *mut c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        getrf_batched(
            handle,
            Ty::S,
            n,
            a as u64,
            lda,
            p as u64,
            info as u64,
            batch,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasSgetrsBatched(
    handle: *mut c_void,
    trans: c_int,
    n: c_int,
    nrhs: c_int,
    a: *const *const f32,
    lda: c_int,
    p: *const c_int,
    b: *const *mut f32,
    ldb: c_int,
    info: *mut c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        getrs_batched(
            handle,
            Ty::S,
            trans,
            n,
            nrhs,
            a as u64,
            lda,
            p as u64,
            b as u64,
            ldb,
            info,
            batch,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDgetrfBatched(
    handle: *mut c_void,
    n: c_int,
    a: *const *mut f64,
    lda: c_int,
    p: *mut c_int,
    info: *mut c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        getrf_batched(
            handle,
            Ty::D,
            n,
            a as u64,
            lda,
            p as u64,
            info as u64,
            batch,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasDgetrsBatched(
    handle: *mut c_void,
    trans: c_int,
    n: c_int,
    nrhs: c_int,
    a: *const *const f64,
    lda: c_int,
    p: *const c_int,
    b: *const *mut f64,
    ldb: c_int,
    info: *mut c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        getrs_batched(
            handle,
            Ty::D,
            trans,
            n,
            nrhs,
            a as u64,
            lda,
            p as u64,
            b as u64,
            ldb,
            info,
            batch,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgetrfBatched(
    handle: *mut c_void,
    n: c_int,
    a: *const *mut Complex32,
    lda: c_int,
    p: *mut c_int,
    info: *mut c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        getrf_batched(
            handle,
            Ty::C,
            n,
            a as u64,
            lda,
            p as u64,
            info as u64,
            batch,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasCgetrsBatched(
    handle: *mut c_void,
    trans: c_int,
    n: c_int,
    nrhs: c_int,
    a: *const *const Complex32,
    lda: c_int,
    p: *const c_int,
    b: *const *mut Complex32,
    ldb: c_int,
    info: *mut c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        getrs_batched(
            handle,
            Ty::C,
            trans,
            n,
            nrhs,
            a as u64,
            lda,
            p as u64,
            b as u64,
            ldb,
            info,
            batch,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZgetrfBatched(
    handle: *mut c_void,
    n: c_int,
    a: *const *mut Complex64,
    lda: c_int,
    p: *mut c_int,
    info: *mut c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        getrf_batched(
            handle,
            Ty::Z,
            n,
            a as u64,
            lda,
            p as u64,
            info as u64,
            batch,
        )
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasZgetrsBatched(
    handle: *mut c_void,
    trans: c_int,
    n: c_int,
    nrhs: c_int,
    a: *const *const Complex64,
    lda: c_int,
    p: *const c_int,
    b: *const *mut Complex64,
    ldb: c_int,
    info: *mut c_int,
    batch: c_int,
) -> c_int {
    unsafe {
        getrs_batched(
            handle,
            Ty::Z,
            trans,
            n,
            nrhs,
            a as u64,
            lda,
            p as u64,
            b as u64,
            ldb,
            info,
            batch,
        )
    }
}
