//! Pure cuBLAS argument handling for the f64/complex kernels in `ptxgen`: validation with
//! cuBLAS's statuses, op() and increment reduction to element strides, batch chunking, and
//! launch lists. No driver calls, so `src/ptx/cublas64_tests.rs` runs the same launch lists
//! in the SPIR-V simulator. Also the host reference used by the simulator and GPU tests.
use super::ptxgen::{BLOCK, Kernel, Mode, TILE, TRSM_BLOCK, Ty, gemm_rb};
use std::sync::atomic::{AtomicU8, Ordering};

pub const SUCCESS: i32 = 0;
pub const INVALID_VALUE: i32 = 7;
pub const NOT_SUPPORTED: i32 = 15;

pub const OP_N: i32 = 0;
pub const OP_T: i32 = 1;
pub const OP_C: i32 = 2;
/// `CUBLAS_OP_CONJG`: conjugate without transposing.
pub const OP_CONJG: i32 = 3;

pub const SIDE_LEFT: i32 = 0;
pub const SIDE_RIGHT: i32 = 1;
pub const FILL_LOWER: i32 = 0;
pub const FILL_UPPER: i32 = 1;
pub const DIAG_NON_UNIT: i32 = 0;
pub const DIAG_UNIT: i32 = 1;

// cudaDataType_t
pub const R_16F: i32 = 2;
pub const R_32F: i32 = 0;
pub const R_64F: i32 = 1;
pub const C_32F: i32 = 4;
pub const C_64F: i32 = 5;
pub const R_16BF: i32 = 14;
// cublasComputeType_t
pub const COMPUTE_32F: i32 = 68;
pub const COMPUTE_32F_PEDANTIC: i32 = 69;
pub const COMPUTE_64F: i32 = 70;
pub const COMPUTE_64F_PEDANTIC: i32 = 71;
pub const COMPUTE_32F_FAST_16F: i32 = 74;
pub const COMPUTE_32F_FAST_16BF: i32 = 75;
pub const COMPUTE_32F_FAST_TF32: i32 = 77;

pub const MAX_GRID_YZ: u32 = 65535;
pub const MAX_GRID_X: u64 = (1 << 31) - 1;

/// GemmEx-style element type of the f64/complex path (`None`: the f16/f32 path's business).
pub fn wide_type(code: i32) -> Option<Ty> {
    match code {
        R_64F => Some(Ty::D),
        C_32F => Some(Ty::C),
        C_64F => Some(Ty::Z),
        _ => None,
    }
}

/// cudaDataType of a stored element (`DotEx`).
pub fn data_type(code: i32) -> Option<Ty> {
    match code {
        R_32F => Some(Ty::S),
        R_64F => Some(Ty::D),
        C_32F => Some(Ty::C),
        C_64F => Some(Ty::Z),
        R_16F => Some(Ty::H),
        R_16BF => Some(Ty::B),
        _ => None,
    }
}

/// Compute types cuBLAS accepts for a GemmEx with A/B/C of `ty` (plus the legacy form that
/// passes the cudaDataType as the compute type).
pub fn compute_ok(ty: Ty, compute: i32) -> bool {
    match ty {
        Ty::D | Ty::Z => {
            matches!(compute, COMPUTE_64F | COMPUTE_64F_PEDANTIC)
                || compute == if ty == Ty::D { R_64F } else { C_64F }
        }
        Ty::C => matches!(
            compute,
            COMPUTE_32F
                | COMPUTE_32F_PEDANTIC
                | COMPUTE_32F_FAST_16F
                | COMPUTE_32F_FAST_16BF
                | COMPUTE_32F_FAST_TF32
                | C_32F
        ),
        _ => false,
    }
}

/// Scalar argument: bit patterns of a host value (f32 values in the low word), or a device
/// address (`ptr != 0`, `CUBLAS_POINTER_MODE_DEVICE`).
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct Scalar {
    pub re: u64,
    pub im: u64,
    pub ptr: u64,
}

impl Scalar {
    pub fn host(ty: Ty, re: f64, im: f64) -> Scalar {
        let bits = |v: f64| {
            if ty.double() {
                v.to_bits()
            } else {
                u64::from((v as f32).to_bits())
            }
        };
        Scalar {
            re: bits(re),
            im: if ty.complex() { bits(im) } else { 0 },
            ptr: 0,
        }
    }
    /// Decode a host scalar from its in-memory bytes (`ty.size()` of them for BLAS types).
    pub fn from_bytes(ty: Ty, b: &[u8]) -> Scalar {
        let c = ty.comp() as usize;
        let word = |o: usize| -> u64 {
            if c == 8 {
                u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
            } else {
                u64::from(u32::from_le_bytes(b[o..o + 4].try_into().unwrap()))
            }
        };
        Scalar {
            re: word(0),
            im: if ty.complex() { word(c) } else { 0 },
            ptr: 0,
        }
    }
    pub fn device(ptr: u64) -> Scalar {
        Scalar { re: 0, im: 0, ptr }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub kernel: Kernel,
    pub ty: Ty,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub args: Vec<u64>,
}

/// op() as (transposed, conjugated); conjugation is dropped for real types.
pub fn op(ty: Ty, code: i32) -> Result<(bool, bool), i32> {
    let (t, c) = match code {
        OP_N => (false, false),
        OP_T => (true, false),
        OP_C => (true, true),
        OP_CONJG => (false, true),
        _ => return Err(INVALID_VALUE),
    };
    Ok((t, c && ty.complex()))
}

/// Batch layout: element strides between matrices, or device tables of matrix pointers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Batch {
    Strided {
        a: u64,
        b: u64,
        c: u64,
        sa: i64,
        sb: i64,
        sc: i64,
    },
    Tables {
        a: u64,
        b: u64,
        c: u64,
    },
}

fn chunks(batch: u32) -> impl Iterator<Item = (u32, u32)> {
    (0..batch.div_ceil(MAX_GRID_YZ)).map(move |i| {
        let start = i * MAX_GRID_YZ;
        (start, (batch - start).min(MAX_GRID_YZ))
    })
}

fn offset(base: u64, elems: i64, ty: Ty) -> u64 {
    base.wrapping_add((elems as u64).wrapping_mul(u64::from(ty.size())))
}

/// Base of a vector of `len` elements with increment `inc`: BLAS starts a negative
/// increment at the far end.
pub fn vec_base(base: u64, len: i64, inc: i64, ty: Ty) -> u64 {
    if inc < 0 && len > 0 {
        offset(base, (len - 1) * -inc, ty)
    } else {
        base
    }
}

fn grid_1d(threads: u64, block: u32) -> Result<u32, i32> {
    let b = threads.div_ceil(u64::from(block)).max(1);
    if b > MAX_GRID_X {
        Err(NOT_SUPPORTED)
    } else {
        Ok(b as u32)
    }
}

/// One gemm-kernel problem, already reduced to element strides (see `ptxgen::gemm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Mm {
    pub m: u64,
    pub n: u64,
    pub k: u64,
    pub a_si: u64,
    pub a_sl: u64,
    pub b_sl: u64,
    pub b_sj: u64,
    pub ldc: u64,
    pub c_si: u64,
    pub conj_a: bool,
    pub conj_b: bool,
    /// 0: store all of C, 1: only `i >= j`, 2: only `i <= j`.
    pub tri_c: u64,
    /// Force a real diagonal (herk, her2k).
    pub herm_c: bool,
    pub mode_a: Mode,
    pub unit_a: bool,
    pub tr_a: bool,
    pub mode_b: Mode,
    pub unit_b: bool,
    pub tr_b: bool,
}

impl Mm {
    /// Plain product of `op(A)` (`ta`: stored transposed, leading dimension `lda`) and
    /// `op(B)` into a column-major C.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: u64,
        n: u64,
        k: u64,
        ta: bool,
        lda: u64,
        ca: bool,
        tb: bool,
        ldb: u64,
        cb: bool,
        ldc: u64,
    ) -> Mm {
        let (a_si, a_sl) = if ta { (lda, 1) } else { (1, lda) };
        let (b_sl, b_sj) = if tb { (ldb, 1) } else { (1, ldb) };
        Mm {
            m,
            n,
            k,
            a_si,
            a_sl,
            b_sl,
            b_sj,
            ldc,
            c_si: 1,
            conj_a: ca,
            conj_b: cb,
            tri_c: 0,
            herm_c: false,
            mode_a: Mode::Plain,
            unit_a: false,
            tr_a: ta,
            mode_b: Mode::Plain,
            unit_b: false,
            tr_b: tb,
        }
    }
}

/// Gemm kernel choice: 0 automatic, 1 always the 16 x 16 one-output-per-thread kernel,
/// 2 always the register-blocked one (`CFI_CUBLAS_GEMM_TILE=small|big` in the shim).
pub static GEMM_TILE: AtomicU8 = AtomicU8::new(0);

thread_local! {
    /// Per-thread override of [`GEMM_TILE`] (tests run both kernels).
    pub static GEMM_TILE_LOCAL: std::cell::Cell<u8> = const { std::cell::Cell::new(0) };
}

/// Whether an m x n product takes the register-blocked kernel: when C holds at least
/// [`gemm_blocked_tiles`] of its tiles (smaller products keep more work-groups with the 16 x 16
/// kernel).
pub fn gemm_blocked(ty: Ty, m: u64, n: u64) -> bool {
    let local = GEMM_TILE_LOCAL.with(|c| c.get());
    let choice = if local != 0 { local } else { GEMM_TILE.load(Ordering::Relaxed) };
    match choice {
        1 => false,
        2 => true,
        _ => {
            let bm = u64::from(TILE * gemm_rb(ty));
            m.div_ceil(bm) * n.div_ceil(bm) >= gemm_blocked_tiles(ty) && m >= bm / 2 && n >= bm / 2
        }
    }
}

/// Fewest blocked tiles of C for the blocked kernel (measured on the B390: f64 gains from
/// about 1024^2, complex double from 256^2, complex single from 256^2).
pub fn gemm_blocked_tiles(ty: Ty) -> u64 {
    match ty {
        Ty::D | Ty::S => 128,
        Ty::Z => 32,
        _ => 16,
    }
}

/// Launches of the gemm kernel for `mm` over `batch` matrices.
pub fn mm(
    ty: Ty,
    mm: &Mm,
    alpha: Scalar,
    beta: Scalar,
    batch: i32,
    layout: Batch,
) -> Result<Vec<Launch>, i32> {
    if batch < 0 {
        return Err(INVALID_VALUE);
    }
    if mm.m == 0 || mm.n == 0 || batch == 0 {
        return Ok(Vec::new());
    }
    let big = gemm_blocked(ty, mm.m, mm.n);
    let edge = u64::from(if big { TILE * gemm_rb(ty) } else { TILE });
    let gx = mm.m.div_ceil(edge);
    let gy = mm.n.div_ceil(edge);
    if gx > MAX_GRID_X || gy > u64::from(MAX_GRID_YZ) {
        return Err(NOT_SUPPORTED);
    }
    let tables = matches!(layout, Batch::Tables { .. });
    let kernel = match (big, tables) {
        (false, false) => Kernel::GemmS,
        (false, true) => Kernel::GemmP,
        (true, false) => Kernel::GemmBS,
        (true, true) => Kernel::GemmBP,
    };
    let mut out = Vec::new();
    for (start, count) in chunks(batch as u32) {
        let (a, b, c, sa, sb, sc) = match layout {
            Batch::Strided {
                a,
                b,
                c,
                sa,
                sb,
                sc,
            } => {
                let s = i64::from(start);
                (
                    offset(a, s.wrapping_mul(sa), ty),
                    offset(b, s.wrapping_mul(sb), ty),
                    offset(c, s.wrapping_mul(sc), ty),
                    sa as u64,
                    sb as u64,
                    sc as u64,
                )
            }
            Batch::Tables { a, b, c } => {
                let o = u64::from(start) * 8;
                (a + o, b + o, c + o, 0, 0, 0)
            }
        };
        out.push(Launch {
            kernel,
            ty,
            grid: [gx as u32, gy as u32, count],
            block: kernel.block(),
            args: vec![
                a,
                b,
                c,
                alpha.re,
                alpha.im,
                beta.re,
                beta.im,
                alpha.ptr,
                beta.ptr,
                mm.m,
                mm.n,
                mm.k,
                mm.a_si,
                mm.a_sl,
                mm.b_sl,
                mm.b_sj,
                mm.ldc,
                sa,
                sb,
                sc,
                mm.conj_a as u64,
                mm.conj_b as u64,
                mm.c_si,
                mm.tri_c,
                mm.herm_c as u64,
                mm.mode_a as u64,
                mm.unit_a as u64,
                mm.tr_a as u64,
                mm.mode_b as u64,
                mm.unit_b as u64,
                mm.tr_b as u64,
            ],
        });
    }
    Ok(out)
}

/// `C = alpha op(A) op(B) + beta C` for `batch` matrices (cuBLAS `?gemm*`).
#[allow(clippy::too_many_arguments)]
pub fn gemm(
    ty: Ty,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    k: i32,
    lda: i32,
    ldb: i32,
    ldc: i32,
    alpha: Scalar,
    beta: Scalar,
    batch: i32,
    layout: Batch,
) -> Result<Vec<Launch>, i32> {
    let (ta, ca) = op(ty, transa)?;
    let (tb, cb) = op(ty, transb)?;
    if m < 0 || n < 0 || k < 0 || batch < 0 {
        return Err(INVALID_VALUE);
    }
    let rows_a = if ta { k } else { m };
    let rows_b = if tb { n } else { k };
    if lda < rows_a.max(1) || ldb < rows_b.max(1) || ldc < m.max(1) {
        return Err(INVALID_VALUE);
    }
    let p = Mm::new(
        m as u64, n as u64, k as u64, ta, lda as u64, ca, tb, ldb as u64, cb, ldc as u64,
    );
    mm(ty, &p, alpha, beta, batch, layout)
}

fn uplo_lower(uplo: i32) -> Result<bool, i32> {
    match uplo {
        FILL_LOWER => Ok(true),
        FILL_UPPER => Ok(false),
        _ => Err(INVALID_VALUE),
    }
}

/// Rank-k updates of the `uplo` triangle of the n x n matrix C:
/// `C = alpha op(A) op(B)^T + beta C` (`syrkx`; `syrk` is `B = A`), with `^H` and a real
/// diagonal when `herm` (`herkx`, `herk`). op is N or T (C for `herm`); A and B are n x k.
#[allow(clippy::too_many_arguments)]
pub fn rank_k(
    ty: Ty,
    uplo: i32,
    trans: i32,
    n: i32,
    k: i32,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
    c: u64,
    ldc: i32,
    alpha: Scalar,
    beta: Scalar,
    herm: bool,
) -> Result<Vec<Launch>, i32> {
    let lower = uplo_lower(uplo)?;
    let t = match trans {
        OP_N => false,
        OP_T if !herm || !ty.complex() => true,
        OP_C if herm || !ty.complex() => true,
        _ => return Err(INVALID_VALUE),
    };
    if n < 0 || k < 0 {
        return Err(INVALID_VALUE);
    }
    let rows = if t { k } else { n };
    if lda < rows.max(1) || ldb < rows.max(1) || ldc < n.max(1) {
        return Err(INVALID_VALUE);
    }
    let cx = herm && ty.complex();
    // op(A) as stored (N: A, T/C: A^T or A^H); the second factor is op(B)^T (^H).
    let mut p = Mm::new(
        n as u64,
        n as u64,
        k as u64,
        t,
        lda as u64,
        cx && t,
        !t,
        ldb as u64,
        cx && !t,
        ldc as u64,
    );
    p.tri_c = if lower { 1 } else { 2 };
    p.herm_c = cx;
    mm(
        ty,
        &p,
        alpha,
        beta,
        1,
        Batch::Strided {
            a,
            b,
            c,
            sa: 0,
            sb: 0,
            sc: 0,
        },
    )
}

/// Rank-2k update (`syr2k`, `her2k`): `alpha op(A) op(B)^T + alpha' op(B) op(A)^T + beta C`
/// on the `uplo` triangle, with `alpha' = conj(alpha)` for `her2k`. Two launches; the
/// second accumulates (beta 1).
#[allow(clippy::too_many_arguments)]
pub fn rank_2k(
    ty: Ty,
    uplo: i32,
    trans: i32,
    n: i32,
    k: i32,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
    c: u64,
    ldc: i32,
    alpha: Scalar,
    alpha2: Scalar,
    beta: Scalar,
    herm: bool,
) -> Result<Vec<Launch>, i32> {
    let mut l = rank_k(
        ty, uplo, trans, n, k, a, lda, b, ldb, c, ldc, alpha, beta, herm,
    )?;
    l.extend(rank_k(
        ty,
        uplo,
        trans,
        n,
        k,
        b,
        ldb,
        a,
        lda,
        c,
        ldc,
        alpha2,
        Scalar::host(ty, 1.0, 0.0),
        herm,
    )?);
    Ok(l)
}

fn sym_mode(lower: bool, herm: bool) -> Mode {
    match (lower, herm) {
        (true, false) => Mode::SymLower,
        (false, false) => Mode::SymUpper,
        (true, true) => Mode::HermLower,
        (false, true) => Mode::HermUpper,
    }
}

/// `C = alpha A B + beta C` (left) or `alpha B A + beta C` (right) with A symmetric
/// (`symm`) or Hermitian (`hemm`), stored in its `uplo` triangle; C is m x n.
#[allow(clippy::too_many_arguments)]
pub fn symm(
    ty: Ty,
    side: i32,
    uplo: i32,
    m: i32,
    n: i32,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
    c: u64,
    ldc: i32,
    alpha: Scalar,
    beta: Scalar,
    herm: bool,
) -> Result<Vec<Launch>, i32> {
    let lower = uplo_lower(uplo)?;
    let left = match side {
        SIDE_LEFT => true,
        SIDE_RIGHT => false,
        _ => return Err(INVALID_VALUE),
    };
    if m < 0 || n < 0 {
        return Err(INVALID_VALUE);
    }
    let order = if left { m } else { n };
    if lda < order.max(1) || ldb < m.max(1) || ldc < m.max(1) {
        return Err(INVALID_VALUE);
    }
    let mode = sym_mode(lower, herm && ty.complex());
    let (m, n) = (m as u64, n as u64);
    let p = if left {
        let mut p = Mm::new(
            m, n, m, false, lda as u64, false, false, ldb as u64, false, ldc as u64,
        );
        p.mode_a = mode;
        p
    } else {
        let mut p = Mm::new(
            m, n, n, false, ldb as u64, false, false, lda as u64, false, ldc as u64,
        );
        p.mode_b = mode;
        p
    };
    let (x, y) = if left { (a, b) } else { (b, a) };
    mm(
        ty,
        &p,
        alpha,
        beta,
        1,
        Batch::Strided {
            a: x,
            b: y,
            c,
            sa: 0,
            sb: 0,
            sc: 0,
        },
    )
}

/// `C = alpha op(A) B` (left) or `alpha B op(A)` (right), A triangular (`trmm`); `b` must
/// not overlap `c` (the caller copies an in-place B first). C is m x n with `ldc`.
#[allow(clippy::too_many_arguments)]
pub fn trmm(
    ty: Ty,
    side: i32,
    uplo: i32,
    trans: i32,
    diag: i32,
    m: i32,
    n: i32,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
    c: u64,
    ldc: i32,
    alpha: Scalar,
) -> Result<Vec<Launch>, i32> {
    let lower = uplo_lower(uplo)?;
    let left = match side {
        SIDE_LEFT => true,
        SIDE_RIGHT => false,
        _ => return Err(INVALID_VALUE),
    };
    if trans == OP_CONJG {
        return Err(INVALID_VALUE);
    }
    let (t, cj) = op(ty, trans)?;
    let unit = match diag {
        DIAG_NON_UNIT => false,
        DIAG_UNIT => true,
        _ => return Err(INVALID_VALUE),
    };
    if m < 0 || n < 0 {
        return Err(INVALID_VALUE);
    }
    let order = if left { m } else { n };
    if lda < order.max(1) || ldb < m.max(1) || ldc < m.max(1) {
        return Err(INVALID_VALUE);
    }
    let mode = if lower {
        Mode::TriLower
    } else {
        Mode::TriUpper
    };
    let (m, n) = (m as u64, n as u64);
    let zero = Scalar::host(ty, 0.0, 0.0);
    let p = if left {
        let mut p = Mm::new(
            m, n, m, t, lda as u64, cj, false, ldb as u64, false, ldc as u64,
        );
        (p.mode_a, p.unit_a) = (mode, unit);
        p
    } else {
        let mut p = Mm::new(
            m, n, n, false, ldb as u64, false, t, lda as u64, cj, ldc as u64,
        );
        (p.mode_b, p.unit_b) = (mode, unit);
        p
    };
    let (x, y) = if left { (a, b) } else { (b, a) };
    mm(
        ty,
        &p,
        alpha,
        zero,
        1,
        Batch::Strided {
            a: x,
            b: y,
            c,
            sa: 0,
            sb: 0,
            sc: 0,
        },
    )
}

/// `y = alpha A x + beta y` with A symmetric (`symv`) or Hermitian (`hemv`) in `uplo`.
#[allow(clippy::too_many_arguments)]
pub fn symv(
    ty: Ty,
    uplo: i32,
    n: i32,
    a: u64,
    lda: i32,
    x: u64,
    incx: i32,
    y: u64,
    incy: i32,
    alpha: Scalar,
    beta: Scalar,
    herm: bool,
) -> Result<Vec<Launch>, i32> {
    let lower = uplo_lower(uplo)?;
    if n < 0 || lda < n.max(1) || incx == 0 || incy == 0 {
        return Err(INVALID_VALUE);
    }
    let (nn, ix, iy) = (n as u64, i64::from(incx), i64::from(incy));
    let mut p = Mm::new(nn, 1, nn, false, lda as u64, false, false, 1, false, 1);
    p.mode_a = sym_mode(lower, herm && ty.complex());
    (p.b_sl, p.c_si) = (ix as u64, iy as u64);
    let (xb, yb) = (
        vec_base(x, i64::from(n), ix, ty),
        vec_base(y, i64::from(n), iy, ty),
    );
    mm(
        ty,
        &p,
        alpha,
        beta,
        1,
        Batch::Strided {
            a,
            b: xb,
            c: yb,
            sa: 0,
            sb: 0,
            sc: 0,
        },
    )
}

/// `A += alpha x y^T` (`ger`, `geru`) or `alpha x y^H` (`gerc`), A m x n.
#[allow(clippy::too_many_arguments)]
pub fn ger(
    ty: Ty,
    m: i32,
    n: i32,
    x: u64,
    incx: i32,
    y: u64,
    incy: i32,
    a: u64,
    lda: i32,
    alpha: Scalar,
    conj: bool,
) -> Result<Vec<Launch>, i32> {
    if m < 0 || n < 0 || incx == 0 || incy == 0 || lda < m.max(1) {
        return Err(INVALID_VALUE);
    }
    let (ix, iy) = (i64::from(incx), i64::from(incy));
    let mut p = Mm::new(
        m as u64,
        n as u64,
        1,
        false,
        1,
        false,
        false,
        1,
        conj && ty.complex(),
        lda as u64,
    );
    (p.a_si, p.a_sl, p.b_sl, p.b_sj) = (ix as u64, 0, 0, iy as u64);
    let (xb, yb) = (
        vec_base(x, i64::from(m), ix, ty),
        vec_base(y, i64::from(n), iy, ty),
    );
    mm(
        ty,
        &p,
        alpha,
        Scalar::host(ty, 1.0, 0.0),
        1,
        Batch::Strided {
            a: xb,
            b: yb,
            c: a,
            sa: 0,
            sb: 0,
            sc: 0,
        },
    )
}

/// `x = op(A) x0` (`trmv`), where `x0` is a contiguous copy of x at `x0` (the caller makes
/// it) and x has increment `incx`.
#[allow(clippy::too_many_arguments)]
pub fn trmv(
    ty: Ty,
    uplo: i32,
    trans: i32,
    diag: i32,
    n: i32,
    a: u64,
    lda: i32,
    x0: u64,
    x: u64,
    incx: i32,
) -> Result<Vec<Launch>, i32> {
    let lower = uplo_lower(uplo)?;
    if trans == OP_CONJG {
        return Err(INVALID_VALUE);
    }
    let (t, cj) = op(ty, trans)?;
    let unit = match diag {
        DIAG_NON_UNIT => false,
        DIAG_UNIT => true,
        _ => return Err(INVALID_VALUE),
    };
    if n < 0 || lda < n.max(1) || incx == 0 {
        return Err(INVALID_VALUE);
    }
    let nn = n as u64;
    let mut p = Mm::new(nn, 1, nn, t, lda as u64, cj, false, nn.max(1), false, 1);
    (p.mode_a, p.unit_a) = (
        if lower {
            Mode::TriLower
        } else {
            Mode::TriUpper
        },
        unit,
    );
    p.c_si = i64::from(incx) as u64;
    let xb = vec_base(x, i64::from(n), i64::from(incx), ty);
    mm(
        ty,
        &p,
        Scalar::host(ty, 1.0, 0.0),
        Scalar::host(ty, 0.0, 0.0),
        1,
        Batch::Strided {
            a,
            b: x0,
            c: xb,
            sa: 0,
            sb: 0,
            sc: 0,
        },
    )
}

/// `y = x` or swap (`?copy`, `?swap`).
pub fn copy(
    ty: Ty,
    n: i32,
    x: u64,
    incx: i32,
    y: u64,
    incy: i32,
    swap: bool,
) -> Result<Vec<Launch>, i32> {
    if n <= 0 {
        return Ok(Vec::new());
    }
    let (n, ix, iy) = (i64::from(n), i64::from(incx), i64::from(incy));
    Ok(vec![Launch {
        kernel: Kernel::Copy,
        ty,
        grid: [grid_1d(n as u64, BLOCK)?, 1, 1],
        block: Kernel::Copy.block(),
        args: vec![
            vec_base(x, n, ix, ty),
            vec_base(y, n, iy, ty),
            n as u64,
            ix as u64,
            iy as u64,
            swap as u64,
        ],
    }])
}

/// Most blocks a multi-block reduction (`dot`, `nrm2`, `amax`) uses; its partials need
/// [`PART_BYTES`] of device scratch.
pub const PARTS: u32 = 256;
/// Scratch bytes of the partials of a multi-block reduction (16 bytes per block).
pub const PART_BYTES: u64 = PARTS as u64 * 16;

/// Blocks of a reduction over `n` elements: one per 2048 elements, at most [`PARTS`];
/// 1 (a single launch, no scratch) without scratch or for short vectors.
fn parts(n: i64, scratch: u64) -> u32 {
    if scratch == 0 {
        return 1;
    }
    (n / (i64::from(BLOCK) * 8)).clamp(1, i64::from(PARTS)) as u32
}

/// 1-based index of the largest (`min`: smallest) `|re| + |im|` into the device i32 `out`
/// (`i?amax`, `i?amin`); `n <= 0` or `incx <= 0` gives 0. `scratch` ([`PART_BYTES`], or 0)
/// allows a multi-block first pass.
pub fn amax(ty: Ty, n: i32, x: u64, incx: i32, out: u64, min: bool, scratch: u64) -> Vec<Launch> {
    let n = if incx <= 0 { 0 } else { n.max(0) };
    let g = parts(i64::from(n), scratch);
    let one = |grid: u32, args: Vec<u64>| Launch {
        kernel: Kernel::Amax,
        ty,
        grid: [grid, 1, 1],
        block: Kernel::Amax.block(),
        args,
    };
    let inc = i64::from(incx) as u64;
    if g == 1 {
        return vec![one(1, vec![x, out, n as u64, inc, min as u64, 0])];
    }
    vec![
        one(g, vec![x, scratch, n as u64, inc, min as u64, 1]),
        one(1, vec![scratch, out, u64::from(g), 0, min as u64, 2]),
    ]
}

/// Elementwise kernel launch over an m x n C (`geam` and `dgmm`).
#[allow(clippy::too_many_arguments)]
fn elementwise(
    ty: Ty,
    m: u64,
    n: u64,
    a: u64,
    b: u64,
    c: u64,
    alpha: Scalar,
    beta: Scalar,
    a_s: (u64, u64),
    b_s: (u64, u64),
    ldc: u64,
    conj: (bool, bool),
    mul: bool,
) -> Result<Vec<Launch>, i32> {
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    Ok(vec![Launch {
        kernel: Kernel::Geam,
        ty,
        grid: [grid_1d(m, BLOCK)?, n.min(u64::from(MAX_GRID_YZ)) as u32, 1],
        block: Kernel::Geam.block(),
        args: vec![
            a,
            b,
            c,
            alpha.re,
            alpha.im,
            beta.re,
            beta.im,
            alpha.ptr,
            beta.ptr,
            m,
            n,
            a_s.0,
            a_s.1,
            b_s.0,
            b_s.1,
            ldc,
            conj.0 as u64,
            conj.1 as u64,
            mul as u64,
            1,
            0,
        ],
    }])
}

/// `C = alpha op(A) + beta op(B)` (cuBLAS `?geam`), C m x n.
#[allow(clippy::too_many_arguments)]
pub fn geam(
    ty: Ty,
    transa: i32,
    transb: i32,
    m: i32,
    n: i32,
    a: u64,
    lda: i32,
    b: u64,
    ldb: i32,
    c: u64,
    ldc: i32,
    alpha: Scalar,
    beta: Scalar,
) -> Result<Vec<Launch>, i32> {
    if transa == OP_CONJG || transb == OP_CONJG {
        return Err(INVALID_VALUE);
    }
    let (ta, ca) = op(ty, transa)?;
    let (tb, cb) = op(ty, transb)?;
    if m < 0 || n < 0 {
        return Err(INVALID_VALUE);
    }
    let (ra, rb) = (if ta { n } else { m }, if tb { n } else { m });
    if lda < ra.max(1) || ldb < rb.max(1) || ldc < m.max(1) {
        return Err(INVALID_VALUE);
    }
    let s = |t: bool, ld: i32| if t { (ld as u64, 1) } else { (1, ld as u64) };
    elementwise(
        ty,
        m as u64,
        n as u64,
        a,
        b,
        c,
        alpha,
        beta,
        s(ta, lda),
        s(tb, ldb),
        ldc as u64,
        (ca, cb),
        false,
    )
}

/// `C = A diag(x)` (right) or `diag(x) A` (left) (cuBLAS `?dgmm`).
#[allow(clippy::too_many_arguments)]
pub fn dgmm(
    ty: Ty,
    side: i32,
    m: i32,
    n: i32,
    a: u64,
    lda: i32,
    x: u64,
    incx: i32,
    c: u64,
    ldc: i32,
) -> Result<Vec<Launch>, i32> {
    let left = match side {
        SIDE_LEFT => true,
        SIDE_RIGHT => false,
        _ => return Err(INVALID_VALUE),
    };
    if m < 0 || n < 0 || lda < m.max(1) || ldc < m.max(1) {
        return Err(INVALID_VALUE);
    }
    let ix = i64::from(incx);
    let len = if left { m } else { n };
    let xb = vec_base(x, i64::from(len), ix, ty);
    let xs = if left { (ix as u64, 0) } else { (0, ix as u64) };
    let one = Scalar::host(ty, 1.0, 0.0);
    elementwise(
        ty,
        m as u64,
        n as u64,
        a,
        xb,
        c,
        one,
        one,
        (1, lda as u64),
        xs,
        ldc as u64,
        (false, false),
        true,
    )
}

/// `y = alpha op(A) x + beta y` (cuBLAS `?gemv`), `A` stored `m x n`.
#[allow(clippy::too_many_arguments)]
pub fn gemv(
    ty: Ty,
    trans: i32,
    m: i32,
    n: i32,
    a: u64,
    lda: i32,
    x: u64,
    incx: i32,
    y: u64,
    incy: i32,
    alpha: Scalar,
    beta: Scalar,
) -> Result<Vec<Launch>, i32> {
    let (t, c) = op(ty, trans)?;
    if trans == OP_CONJG || m < 0 || n < 0 || lda < m.max(1) || incx == 0 || incy == 0 {
        return Err(INVALID_VALUE);
    }
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let (rows, cols) = if t { (n, m) } else { (m, n) };
    let (a_si, a_sl) = if t { (lda as u64, 1) } else { (1, lda as u64) };
    let (incx, incy) = (i64::from(incx), i64::from(incy));
    let xb = vec_base(x, i64::from(cols), incx, ty);
    let yb = vec_base(y, i64::from(rows), incy, ty);
    Ok(vec![Launch {
        kernel: Kernel::Gemv,
        ty,
        grid: [grid_1d(rows as u64, BLOCK)?, 1, 1],
        block: Kernel::Gemv.block(),
        args: vec![
            a,
            xb,
            yb,
            alpha.re,
            alpha.im,
            beta.re,
            beta.im,
            alpha.ptr,
            beta.ptr,
            rows as u64,
            cols as u64,
            a_si,
            a_sl,
            incx as u64,
            incy as u64,
            c as u64,
        ],
    }])
}

/// `*out = sum conj?(x) y` into the device address `out` (`n <= 0` writes zero). With
/// `scratch` ([`PART_BYTES`], or 0) long vectors take a multi-block pass of partial sums
/// (not for f16/bf16, whose partials would be rounded) and a one-block sum of them.
#[allow(clippy::too_many_arguments)]
pub fn dot(
    ty: Ty,
    n: i32,
    x: u64,
    incx: i32,
    y: u64,
    incy: i32,
    out: u64,
    conj: bool,
    scratch: u64,
) -> Vec<Launch> {
    let n = i64::from(n.max(0));
    let (incx, incy) = (i64::from(incx), i64::from(incy));
    let g = if matches!(ty, Ty::H | Ty::B) { 1 } else { parts(n, scratch) };
    let first = Launch {
        kernel: Kernel::Dot,
        ty,
        grid: [g, 1, 1],
        block: Kernel::Dot.block(),
        args: vec![
            vec_base(x, n, incx, ty),
            vec_base(y, n, incy, ty),
            if g == 1 { out } else { scratch },
            n as u64,
            incx as u64,
            incy as u64,
            (conj && ty.complex()) as u64,
        ],
    };
    if g == 1 {
        return vec![first];
    }
    vec![
        first,
        Launch {
            kernel: Kernel::Dot,
            ty,
            grid: [1, 1, 1],
            block: Kernel::Dot.block(),
            args: vec![scratch, 0, out, u64::from(g), 1, 0, 0],
        },
    ]
}

/// `*out = ||x||_2` (real) into the device address `out`; `n <= 0` or `incx <= 0` gives 0.
/// With `scratch` ([`PART_BYTES`], or 0) long vectors take a multi-block pass of (max,
/// sum of squares) partials; the one-block final launch falls back to the scaled two-pass
/// form when the unscaled sum could have overflowed or underflowed.
pub fn nrm2(ty: Ty, n: i32, x: u64, incx: i32, out: u64, scratch: u64) -> Vec<Launch> {
    let n = if incx <= 0 { 0 } else { n.max(0) };
    let g = parts(i64::from(n), scratch);
    let inc = i64::from(incx) as u64;
    let fin = |part: u64, nparts: u32| Launch {
        kernel: Kernel::Nrm2,
        ty,
        grid: [1, 1, 1],
        block: Kernel::Nrm2.block(),
        args: vec![x, out, n as u64, inc, part, u64::from(nparts)],
    };
    if g == 1 {
        return vec![fin(0, 0)];
    }
    vec![
        Launch {
            kernel: Kernel::Nrm2P,
            ty,
            grid: [g, 1, 1],
            block: Kernel::Nrm2P.block(),
            args: vec![x, scratch, n as u64, inc],
        },
        fin(scratch, g),
    ]
}

/// `y += alpha x` (cuBLAS `?axpy`).
pub fn axpy(
    ty: Ty,
    n: i32,
    alpha: Scalar,
    x: u64,
    incx: i32,
    y: u64,
    incy: i32,
) -> Result<Vec<Launch>, i32> {
    if n <= 0 {
        return Ok(Vec::new());
    }
    let (n, incx, incy) = (i64::from(n), i64::from(incx), i64::from(incy));
    Ok(vec![Launch {
        kernel: Kernel::Axpy,
        ty,
        grid: [grid_1d(n as u64, BLOCK)?, 1, 1],
        block: Kernel::Axpy.block(),
        args: vec![
            vec_base(x, n, incx, ty),
            vec_base(y, n, incy, ty),
            alpha.re,
            alpha.im,
            alpha.ptr,
            n as u64,
            incx as u64,
            incy as u64,
        ],
    }])
}

/// `x *= alpha` (cuBLAS `?scal`; `real_alpha` for `Csscal`/`Zdscal`, where a device scalar
/// has one component). `n <= 0` or `incx <= 0` does nothing.
pub fn scal(
    ty: Ty,
    n: i32,
    alpha: Scalar,
    x: u64,
    incx: i32,
    real_alpha: bool,
) -> Result<Vec<Launch>, i32> {
    if n <= 0 || incx <= 0 {
        return Ok(Vec::new());
    }
    Ok(vec![Launch {
        kernel: Kernel::Scal,
        ty,
        grid: [grid_1d(n as u64, BLOCK)?, 1, 1],
        block: Kernel::Scal.block(),
        args: vec![
            x,
            alpha.re,
            alpha.im,
            alpha.ptr,
            n as u64,
            incx as u64,
            real_alpha as u64,
        ],
    }])
}

/// Kernel-level form of a TRSM: solve `T x = alpha b` per vector (see `ptxgen::trsm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tri {
    pub len: u32,
    pub nvec: u32,
    pub a_rs: u64,
    pub a_cs: u64,
    pub b_vs: u64,
    pub b_es: u64,
    pub lower: bool,
    pub unit: bool,
    pub conj: bool,
}

/// Map cuBLAS side/uplo/trans/diag onto [`Tri`], validating as cuBLAS does.
#[allow(clippy::too_many_arguments)]
pub fn tri(
    ty: Ty,
    side: i32,
    uplo: i32,
    trans: i32,
    diag: i32,
    m: i32,
    n: i32,
    lda: i32,
    ldb: i32,
) -> Result<Tri, i32> {
    let right = match side {
        SIDE_LEFT => false,
        SIDE_RIGHT => true,
        _ => return Err(INVALID_VALUE),
    };
    let lower_stored = match uplo {
        FILL_LOWER => true,
        FILL_UPPER => false,
        _ => return Err(INVALID_VALUE),
    };
    if trans == OP_CONJG {
        return Err(INVALID_VALUE);
    }
    let (mut t, conj) = op(ty, trans)?;
    let unit = match diag {
        DIAG_NON_UNIT => false,
        DIAG_UNIT => true,
        _ => return Err(INVALID_VALUE),
    };
    if m < 0 || n < 0 {
        return Err(INVALID_VALUE);
    }
    let order = if right { n } else { m };
    if lda < order.max(1) || ldb < m.max(1) {
        return Err(INVALID_VALUE);
    }
    // X op(A) = B  <=>  op(A)^T x = b for every row x of X; (A^H)^T = conj(A).
    if right {
        t = !t;
    }
    let (a_rs, a_cs) = if t { (lda as u64, 1) } else { (1, lda as u64) };
    let (b_vs, b_es) = if right {
        (1, ldb as u64)
    } else {
        (ldb as u64, 1)
    };
    Ok(Tri {
        len: order as u32,
        nvec: if right { m } else { n } as u32,
        a_rs,
        a_cs,
        b_vs,
        b_es,
        lower: lower_stored != t,
        unit,
        conj,
    })
}

/// In-place `op(A) X = alpha B` / `X op(A) = alpha B` for `batch` pairs (cuBLAS `?trsm`,
/// `?trsmBatched`). `tables`: `a`/`b` are device pointer tables; otherwise single matrices.
#[allow(clippy::too_many_arguments)]
pub fn trsm(
    ty: Ty,
    p: &Tri,
    alpha: Scalar,
    a: u64,
    b: u64,
    batch: i32,
    tables: bool,
) -> Result<Vec<Launch>, i32> {
    if batch < 0 {
        return Err(INVALID_VALUE);
    }
    if p.len == 0 || p.nvec == 0 || batch == 0 {
        return Ok(Vec::new());
    }
    if !tables && batch == 1 && p.len > 2 * TRSM_NB {
        return trsm_blocked(ty, p, alpha, a, b);
    }
    trsm_vectors(ty, p, alpha, a, b, batch, tables)
}

/// Order of the diagonal blocks of [`trsm_blocked`].
pub const TRSM_NB: u32 = 64;

/// Blocked single-matrix trsm: the diagonal blocks of order [`TRSM_NB`] are solved in turn
/// by the per-vector kernel, and after each one the unsolved rows of every vector are
/// updated at once by a gemm, `X(rest) = beta X(rest) - T(rest, blk) X(blk)` (beta = alpha
/// after the first block, which applies alpha, else 1). Nearly all flops run in the gemm
/// kernel.
fn trsm_blocked(ty: Ty, p: &Tri, alpha: Scalar, a: u64, b: u64) -> Result<Vec<Launch>, i32> {
    let (nb, len) = (TRSM_NB, p.len);
    let one = Scalar::host(ty, 1.0, 0.0);
    let minus = Scalar::host(ty, -1.0, 0.0);
    let at = |base: u64, elems: u64| offset(base, elems as i64, ty);
    let mut out = Vec::new();
    for q in 0..len.div_ceil(nb) {
        let (r0, r1) = if p.lower {
            (q * nb, ((q + 1) * nb).min(len))
        } else {
            (len.saturating_sub((q + 1) * nb), len - q * nb)
        };
        let (r0l, r1l) = (u64::from(r0), u64::from(r1));
        let first = if q == 0 { alpha } else { one };
        let diag = Tri { len: r1 - r0, ..*p };
        let blk = at(b, r0l * p.b_es);
        out.extend(trsm_vectors(
            ty,
            &diag,
            first,
            at(a, r0l * (p.a_rs + p.a_cs)),
            blk,
            1,
            false,
        )?);
        let (s0, s1) = if p.lower { (r1l, u64::from(len)) } else { (0, r0l) };
        if s1 > s0 {
            let pm = Mm {
                m: s1 - s0,
                n: u64::from(p.nvec),
                k: r1l - r0l,
                a_si: p.a_rs,
                a_sl: p.a_cs,
                b_sl: p.b_es,
                b_sj: p.b_vs,
                ldc: p.b_vs,
                c_si: p.b_es,
                conj_a: p.conj,
                conj_b: false,
                tri_c: 0,
                herm_c: false,
                mode_a: Mode::Plain,
                unit_a: false,
                tr_a: false,
                mode_b: Mode::Plain,
                unit_b: false,
                tr_b: false,
            };
            out.extend(mm(
                ty,
                &pm,
                minus,
                first,
                1,
                Batch::Strided {
                    a: at(a, s0 * p.a_rs + r0l * p.a_cs),
                    b: blk,
                    c: at(b, s0 * p.b_es),
                    sa: 0,
                    sb: 0,
                    sc: 0,
                },
            )?);
        }
    }
    Ok(out)
}

/// The per-vector trsm kernel: one thread per right-hand-side vector.
fn trsm_vectors(
    ty: Ty,
    p: &Tri,
    alpha: Scalar,
    a: u64,
    b: u64,
    batch: i32,
    tables: bool,
) -> Result<Vec<Launch>, i32> {
    let kernel = if tables { Kernel::TrsmP } else { Kernel::TrsmS };
    let gx = grid_1d(u64::from(p.nvec), TRSM_BLOCK)?;
    let mut out = Vec::new();
    for (start, count) in chunks(batch as u32) {
        let o = if tables { u64::from(start) * 8 } else { 0 };
        out.push(Launch {
            kernel,
            ty,
            grid: [gx, count, 1],
            block: kernel.block(),
            args: vec![
                a + o,
                b + o,
                alpha.re,
                alpha.im,
                alpha.ptr,
                u64::from(p.len),
                u64::from(p.nvec),
                p.a_rs,
                p.a_cs,
                p.b_vs,
                p.b_es,
                p.lower as u64,
                p.unit as u64,
                p.conj as u64,
                0,
                0,
            ],
        });
    }
    Ok(out)
}

/// Batched LU with partial pivoting (`getrfBatched`); `a` is a device table of matrix
/// pointers, `p` the pivot output (`n` i32 per matrix, 0 = no pivoting), `info` one i32 per
/// matrix.
#[allow(clippy::too_many_arguments)]
pub fn getrf(
    ty: Ty,
    n: i32,
    a: u64,
    lda: i32,
    p: u64,
    info: u64,
    batch: i32,
) -> Result<Vec<Launch>, i32> {
    if n < 0 || lda < n.max(1) || batch < 0 {
        return Err(INVALID_VALUE);
    }
    if batch == 0 {
        return Ok(Vec::new());
    }
    if batch as u64 > MAX_GRID_X {
        return Err(NOT_SUPPORTED);
    }
    // small orders: one thread per matrix; larger: one work-group per matrix
    let (kernel, grid) = if n <= GETRF_THREAD_MAX {
        (Kernel::GetrfT, grid_1d(batch as u64, TRSM_BLOCK)?)
    } else {
        (Kernel::Getrf, batch as u32)
    };
    Ok(vec![Launch {
        kernel,
        ty,
        grid: [grid, 1, 1],
        block: kernel.block(),
        args: vec![
            a,
            p,
            info,
            n as u64,
            lda as u64,
            batch as u64,
            (p != 0) as u64,
        ],
    }])
}

/// Largest order `getrfBatched` factors with one thread per matrix.
pub const GETRF_THREAD_MAX: i32 = 16;

/// Batched Householder QR (`geqrfBatched`): `a` and `tau` are device tables of matrix and
/// tau-vector pointers (min(m, n) elements each). `Err(-i)`: parameter `i` (cuBLAS
/// numbering, handle = 1) is invalid.
pub fn geqrf(ty: Ty, m: i32, n: i32, a: u64, lda: i32, tau: u64, batch: i32) -> Result<Vec<Launch>, i32> {
    if m < 0 {
        return Err(-2);
    }
    if n < 0 {
        return Err(-3);
    }
    if lda < m.max(1) {
        return Err(-5);
    }
    if batch < 0 {
        return Err(-8);
    }
    if m == 0 || n == 0 || batch == 0 {
        return Ok(Vec::new());
    }
    qr_launch(ty, m, n, 0, a, lda, tau, 0, 1, 0, batch)
}

/// Batched least squares `min ||A X - C||` for full-rank m x n A with m >= n (`gelsBatched`,
/// op N only): QR of A in place, `C = Q^H C`, then `R X = C(0:n, :)` in place. `info` is a
/// device i32 per matrix (first zero diagonal of R, 1-based). `Err(-i)` as in [`geqrf`].
#[allow(clippy::too_many_arguments)]
pub fn gels(
    ty: Ty,
    trans: i32,
    m: i32,
    n: i32,
    nrhs: i32,
    a: u64,
    lda: i32,
    c: u64,
    ldc: i32,
    info: u64,
    batch: i32,
) -> Result<Vec<Launch>, i32> {
    if trans != OP_N {
        return Err(-2);
    }
    if m < 0 {
        return Err(-3);
    }
    if n < 0 || n > m {
        return Err(-4);
    }
    if nrhs < 0 {
        return Err(-5);
    }
    if lda < m.max(1) {
        return Err(-7);
    }
    if ldc < m.max(1) {
        return Err(-9);
    }
    if batch < 0 {
        return Err(-12);
    }
    if batch == 0 || (m == 0 && n == 0) {
        return Ok(Vec::new());
    }
    qr_launch(ty, m, n, nrhs, a, lda, 0, c, ldc, info, batch)
}

#[allow(clippy::too_many_arguments)]
fn qr_launch(
    ty: Ty,
    m: i32,
    n: i32,
    nrhs: i32,
    a: u64,
    lda: i32,
    tau: u64,
    c: u64,
    ldc: i32,
    info: u64,
    batch: i32,
) -> Result<Vec<Launch>, i32> {
    if batch as u64 > MAX_GRID_X {
        return Err(NOT_SUPPORTED);
    }
    Ok(vec![Launch {
        kernel: Kernel::Geqrf,
        ty,
        grid: [batch as u32, 1, 1],
        block: Kernel::Geqrf.block(),
        args: vec![
            a,
            tau,
            c,
            info,
            m as u64,
            n as u64,
            nrhs as u64,
            lda as u64,
            ldc.max(1) as u64,
            batch as u64,
        ],
    }])
}

/// Row interchanges of `getrf` applied to the n x nrhs matrices of table `b`.
#[allow(clippy::too_many_arguments)]
pub fn laswp(
    ty: Ty,
    n: i32,
    nrhs: i32,
    b: u64,
    ldb: i32,
    p: u64,
    batch: i32,
    reverse: bool,
) -> Result<Vec<Launch>, i32> {
    if n <= 0 || nrhs <= 0 || batch <= 0 || p == 0 {
        return Ok(Vec::new());
    }
    let gx = grid_1d(nrhs as u64, TRSM_BLOCK)?;
    Ok(chunks(batch as u32)
        .map(|(start, count)| Launch {
            kernel: Kernel::Laswp,
            ty,
            grid: [gx, count, 1],
            block: Kernel::Laswp.block(),
            args: vec![
                b + u64::from(start) * 8,
                p + u64::from(start) * n as u64 * 4,
                n as u64,
                nrhs as u64,
                ldb as u64,
                reverse as u64,
            ],
        })
        .collect())
}

/// `op(A) X = B` from `getrf` factors (`getrsBatched`): pivots, unit-lower L and upper U,
/// in the order op() requires. `Err(-i)`: parameter `i` (cuBLAS numbering, handle = 1)
/// is invalid.
#[allow(clippy::too_many_arguments)]
pub fn getrs(
    ty: Ty,
    trans: i32,
    n: i32,
    nrhs: i32,
    a: u64,
    lda: i32,
    p: u64,
    b: u64,
    ldb: i32,
    batch: i32,
) -> Result<Vec<Launch>, i32> {
    if !(OP_N..=OP_C).contains(&trans) {
        return Err(-2);
    }
    if n < 0 {
        return Err(-3);
    }
    if nrhs < 0 {
        return Err(-4);
    }
    if lda < n.max(1) {
        return Err(-6);
    }
    if ldb < n.max(1) {
        return Err(-9);
    }
    if batch < 0 {
        return Err(-11);
    }
    if n == 0 || nrhs == 0 || batch == 0 {
        return Ok(Vec::new());
    }
    let one = Scalar::host(ty, 1.0, 0.0);
    let lower = tri(
        ty, SIDE_LEFT, FILL_LOWER, trans, DIAG_UNIT, n, nrhs, lda, ldb,
    )
    .map_err(|_| -2)?;
    let upper = tri(
        ty,
        SIDE_LEFT,
        FILL_UPPER,
        trans,
        DIAG_NON_UNIT,
        n,
        nrhs,
        lda,
        ldb,
    )
    .map_err(|_| -2)?;
    let mut out = Vec::new();
    if trans == OP_N {
        out.extend(laswp(ty, n, nrhs, b, ldb, p, batch, false)?);
        out.extend(trsm(ty, &lower, one, a, b, batch, true)?);
        out.extend(trsm(ty, &upper, one, a, b, batch, true)?);
    } else {
        out.extend(trsm(ty, &upper, one, a, b, batch, true)?);
        out.extend(trsm(ty, &lower, one, a, b, batch, true)?);
        out.extend(laswp(ty, n, nrhs, b, ldb, p, batch, true)?);
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Host reference (f64 complex arithmetic), independent of the stride mapping above
// ---------------------------------------------------------------------------

pub type Cx = (f64, f64);

pub fn cmul(a: Cx, b: Cx) -> Cx {
    (a.0 * b.0 - a.1 * b.1, a.0 * b.1 + a.1 * b.0)
}

pub fn cadd(a: Cx, b: Cx) -> Cx {
    (a.0 + b.0, a.1 + b.1)
}

pub fn cdiv(a: Cx, b: Cx) -> Cx {
    let d = b.0 * b.0 + b.1 * b.1;
    ((a.0 * b.0 + a.1 * b.1) / d, (a.1 * b.0 - a.0 * b.1) / d)
}

/// Literal op(): element (r, c) of op(M) for column-major `M` with leading dimension `ld`.
pub fn op_at(m: &[Cx], ld: usize, code: i32, r: usize, c: usize) -> Cx {
    let v = match code {
        OP_T | OP_C => m[c + r * ld],
        _ => m[r + c * ld],
    };
    if code == OP_C || code == OP_CONJG {
        (v.0, -v.1)
    } else {
        v
    }
}

/// Column-major reference GEMM on one matrix triple (`beta == 0` ignores C).
#[allow(clippy::too_many_arguments)]
pub fn ref_gemm(
    ta: i32,
    tb: i32,
    m: usize,
    n: usize,
    k: usize,
    alpha: Cx,
    a: &[Cx],
    lda: usize,
    b: &[Cx],
    ldb: usize,
    beta: Cx,
    c: &mut [Cx],
    ldc: usize,
) {
    for j in 0..n {
        for i in 0..m {
            let mut acc = (0.0, 0.0);
            for l in 0..k {
                acc = cadd(acc, cmul(op_at(a, lda, ta, i, l), op_at(b, ldb, tb, l, j)));
            }
            let mut v = if alpha == (0.0, 0.0) {
                (0.0, 0.0)
            } else {
                cmul(alpha, acc)
            };
            if beta != (0.0, 0.0) {
                v = cadd(v, cmul(beta, c[i + j * ldc]));
            }
            c[i + j * ldc] = v;
        }
    }
}

/// Literal TRSM by substitution (entries outside the stored triangle are ignored).
#[allow(clippy::too_many_arguments)]
pub fn ref_trsm(
    side: i32,
    uplo: i32,
    trans: i32,
    diag: i32,
    m: usize,
    n: usize,
    alpha: Cx,
    a: &[Cx],
    lda: usize,
    b: &mut [Cx],
    ldb: usize,
) {
    let order = if side == SIDE_LEFT { m } else { n };
    // dense op(A) with the unreferenced triangle zeroed
    let mut t = vec![(0.0, 0.0); order * order];
    for r in 0..order {
        for c in 0..order {
            let stored_lower = uplo == FILL_LOWER;
            let (sr, sc) = if trans == OP_N { (r, c) } else { (c, r) };
            let inside = if stored_lower { sr >= sc } else { sr <= sc };
            if inside {
                let v = a[sr + sc * lda];
                t[r * order + c] = if trans == OP_C { (v.0, -v.1) } else { v };
            }
            if r == c && diag == DIAG_UNIT {
                t[r * order + c] = (1.0, 0.0);
            }
        }
    }
    let lower = (uplo == FILL_LOWER) == (trans == OP_N);
    if side == SIDE_LEFT {
        for j in 0..n {
            let rhs: Vec<Cx> = (0..m).map(|i| cmul(alpha, b[i + j * ldb])).collect();
            let mut x = vec![(0.0, 0.0); m];
            let order_iter: Vec<usize> = if lower {
                (0..m).collect()
            } else {
                (0..m).rev().collect()
            };
            for &i in &order_iter {
                let mut s = rhs[i];
                for &l in order_iter.iter().take_while(|&&l| l != i) {
                    s = cadd(s, cmul((-t[i * m + l].0, -t[i * m + l].1), x[l]));
                }
                x[i] = cdiv(s, t[i * m + i]);
            }
            for i in 0..m {
                b[i + j * ldb] = x[i];
            }
        }
    } else {
        // X T = alpha B, solved row by row: x T = r  <=>  for each column c of T
        for i in 0..m {
            let rhs: Vec<Cx> = (0..n).map(|j| cmul(alpha, b[i + j * ldb])).collect();
            let mut x = vec![(0.0, 0.0); n];
            // x_c T(c,c) = r_c - sum_{l != c} x_l T(l,c); lower T: rows l > c contribute
            let cols: Vec<usize> = if lower {
                (0..n).rev().collect()
            } else {
                (0..n).collect()
            };
            for &c in &cols {
                let mut s = rhs[c];
                for &l in cols.iter().take_while(|&&l| l != c) {
                    s = cadd(s, cmul((-x[l].0, -x[l].1), t[l * n + c]));
                }
                x[c] = cdiv(s, t[c * n + c]);
            }
            for j in 0..n {
                b[i + j * ldb] = x[j];
            }
        }
    }
}

fn cconj(v: Cx) -> Cx {
    (v.0, -v.1)
}

/// `H_0 H_1 ... H_{k-1} R` from the packed `geqrf` output (`H_j = I - tau_j v_j v_j^H`,
/// v_j(j) = 1, v_j below the diagonal), m x n column-major with leading dimension `ld`.
pub fn qr_product(m: usize, n: usize, qr: &[Cx], ld: usize, tau: &[Cx]) -> Vec<Cx> {
    let mut x = vec![(0.0, 0.0); m * n];
    for j in 0..n {
        for i in 0..=j.min(m - 1) {
            x[i + j * m] = qr[i + j * ld];
        }
    }
    for j in (0..m.min(n)).rev() {
        let v = |i: usize| if i == j { (1.0, 0.0) } else { qr[i + j * ld] };
        for c in 0..n {
            let mut w = (0.0, 0.0);
            for i in j..m {
                w = cadd(w, cmul(cconj(v(i)), x[i + c * m]));
            }
            let tw = cmul(tau[j], w);
            for i in j..m {
                x[i + c * m] = cadd(x[i + c * m], cmul((-v(i).0, -v(i).1), tw));
            }
        }
    }
    x
}

/// Least-squares solution of `A x = b` (m x n, full column rank) by the normal equations
/// in f64 with partial pivoting.
pub fn lstsq(m: usize, n: usize, a: &[Cx], lda: usize, b: &[Cx]) -> Vec<Cx> {
    let mut g = vec![(0.0, 0.0); n * (n + 1)];
    for r in 0..n {
        for c in 0..n {
            let mut acc = (0.0, 0.0);
            for i in 0..m {
                acc = cadd(acc, cmul(cconj(a[i + r * lda]), a[i + c * lda]));
            }
            g[r * (n + 1) + c] = acc;
        }
        let mut acc = (0.0, 0.0);
        for i in 0..m {
            acc = cadd(acc, cmul(cconj(a[i + r * lda]), b[i]));
        }
        g[r * (n + 1) + n] = acc;
    }
    let abs = |v: Cx| v.0.hypot(v.1);
    for k in 0..n {
        let p = (k..n).max_by(|&x, &y| abs(g[x * (n + 1) + k]).total_cmp(&abs(g[y * (n + 1) + k]))).unwrap();
        for c in 0..=n {
            g.swap(k * (n + 1) + c, p * (n + 1) + c);
        }
        let d = g[k * (n + 1) + k];
        let dn = d.0 * d.0 + d.1 * d.1;
        let inv = (d.0 / dn, -d.1 / dn);
        for r in 0..n {
            if r == k {
                continue;
            }
            let f = cmul(g[r * (n + 1) + k], inv);
            for c in 0..=n {
                let t = cmul(f, g[k * (n + 1) + c]);
                g[r * (n + 1) + c] = (g[r * (n + 1) + c].0 - t.0, g[r * (n + 1) + c].1 - t.1);
            }
        }
    }
    (0..n).map(|r| { let d = g[r * (n + 1) + r]; let dn = d.0 * d.0 + d.1 * d.1; cmul(g[r * (n + 1) + n], (d.0 / dn, -d.1 / dn)) }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s() -> Scalar {
        Scalar::host(Ty::Z, 1.0, 0.0)
    }

    #[test]
    fn gemm_validation_follows_cublas() {
        let z = Batch::Strided {
            a: 8,
            b: 8,
            c: 8,
            sa: 0,
            sb: 0,
            sc: 0,
        };
        assert_eq!(
            gemm(Ty::Z, 4, 0, 2, 2, 2, 2, 2, 2, s(), s(), 1, z),
            Err(INVALID_VALUE)
        );
        assert_eq!(
            gemm(Ty::Z, 0, 0, -1, 2, 2, 2, 2, 2, s(), s(), 1, z),
            Err(INVALID_VALUE)
        );
        assert_eq!(
            gemm(Ty::Z, 0, 0, 4, 2, 3, 3, 3, 4, s(), s(), 1, z),
            Err(INVALID_VALUE)
        ); // lda < m
        assert_eq!(
            gemm(Ty::Z, 1, 0, 4, 2, 3, 2, 3, 4, s(), s(), 1, z),
            Err(INVALID_VALUE)
        ); // lda < k for T
        assert_eq!(
            gemm(Ty::Z, 0, 2, 4, 2, 3, 4, 1, 4, s(), s(), 1, z),
            Err(INVALID_VALUE)
        ); // ldb < n for C
        assert_eq!(
            gemm(Ty::Z, 0, 0, 4, 2, 3, 4, 3, 3, s(), s(), 1, z),
            Err(INVALID_VALUE)
        ); // ldc < m
        assert_eq!(
            gemm(Ty::Z, 0, 0, 0, 2, 3, 1, 3, 1, s(), s(), 1, z),
            Ok(vec![])
        );
        assert_eq!(
            gemm(Ty::Z, 0, 0, 2, 2, 3, 2, 3, 2, s(), s(), 0, z),
            Ok(vec![])
        );
        let l = gemm(Ty::Z, 2, 3, 40, 17, 0, 1, 17, 40, s(), s(), 1, z).unwrap();
        assert_eq!(l[0].grid, [3, 2, 1]);
        assert_eq!(&l[0].args[12..16], &[1, 1, 1, 17]); // A^H: a_si = lda; CONJG B: not transposed
        assert_eq!(&l[0].args[20..22], &[1, 1]);
        // reals drop the conjugate flag
        let l = gemm(Ty::D, 2, 2, 4, 4, 4, 4, 4, 4, s(), s(), 1, z).unwrap();
        assert_eq!(&l[0].args[20..22], &[0, 0]);
    }

    #[test]
    fn gemm_batches_are_chunked() {
        let l = gemm(
            Ty::C,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            s(),
            s(),
            70000,
            Batch::Strided {
                a: 0x1000,
                b: 0x2000,
                c: 0x3000,
                sa: 1,
                sb: 2,
                sc: 3,
            },
        )
        .unwrap();
        assert_eq!(l.len(), 2);
        assert_eq!(l[1].grid[2], 70000 - 65535);
        assert_eq!(l[1].args[0], 0x1000 + 65535 * 8);
        assert_eq!(l[1].args[2], 0x3000 + 65535 * 3 * 8);
        let l = gemm(
            Ty::Z,
            0,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            s(),
            s(),
            65536,
            Batch::Tables {
                a: 0x1000,
                b: 0x2000,
                c: 0x3000,
            },
        )
        .unwrap();
        assert_eq!(l[1].kernel, Kernel::GemmP);
        assert_eq!(l[1].args[1], 0x2000 + 65535 * 8);
    }

    #[test]
    fn negative_increments_start_at_the_far_end() {
        assert_eq!(vec_base(1000, 5, -2, Ty::Z), 1000 + 4 * 2 * 16);
        assert_eq!(vec_base(1000, 5, 3, Ty::Z), 1000);
        let l = gemv(Ty::D, 1, 3, 5, 0, 3, 800, -1, 1600, -2, s(), s()).unwrap();
        // op T: rows = 5 (y), cols = 3 (x)
        assert_eq!(l[0].args[1], 800 + 2 * 8);
        assert_eq!(l[0].args[2], 1600 + 4 * 2 * 8);
        assert_eq!(l[0].args[13] as i64, -1);
        assert_eq!(
            gemv(Ty::D, 0, 3, 5, 0, 3, 0, 0, 0, 1, s(), s()),
            Err(INVALID_VALUE)
        );
        assert_eq!(gemv(Ty::D, 0, 3, 0, 0, 3, 0, 1, 0, 1, s(), s()), Ok(vec![]));
        assert_eq!(nrm2(Ty::D, 5, 0, -1, 0, 0)[0].args[2], 0);
        assert_eq!(nrm2(Ty::D, 1 << 20, 0, 1, 0, 64).len(), 2);
        assert_eq!(nrm2(Ty::D, 1 << 20, 0, 1, 0, 64)[0].grid[0], PARTS);
        assert_eq!(scal(Ty::Z, 5, s(), 0, 0, false), Ok(vec![]));
        assert_eq!(axpy(Ty::Z, 0, s(), 0, 1, 0, 1), Ok(vec![]));
    }

    #[test]
    fn tri_maps_side_and_conjugation() {
        let p = tri(
            Ty::Z,
            SIDE_RIGHT,
            FILL_LOWER,
            OP_C,
            DIAG_NON_UNIT,
            3,
            4,
            4,
            3,
        )
        .unwrap();
        assert_eq!((p.len, p.nvec, p.conj, p.lower), (4, 3, true, true));
        assert_eq!((p.a_rs, p.a_cs, p.b_vs, p.b_es), (1, 4, 1, 3));
        let p = tri(Ty::D, SIDE_LEFT, FILL_UPPER, OP_C, DIAG_UNIT, 3, 4, 3, 5).unwrap();
        assert_eq!((p.conj, p.lower, p.unit), (false, true, true));
        assert_eq!(
            tri(
                Ty::Z,
                SIDE_LEFT,
                FILL_UPPER,
                OP_CONJG,
                DIAG_UNIT,
                3,
                4,
                3,
                5
            ),
            Err(INVALID_VALUE)
        );
        assert_eq!(tri(Ty::Z, 2, 0, 0, 0, 3, 4, 3, 5), Err(INVALID_VALUE));
        assert_eq!(
            tri(Ty::Z, SIDE_RIGHT, 0, 0, 0, 3, 4, 3, 5),
            Err(INVALID_VALUE)
        ); // lda < n
    }

    #[test]
    fn compute_types() {
        assert!(compute_ok(Ty::D, COMPUTE_64F) && compute_ok(Ty::D, R_64F));
        assert!(compute_ok(Ty::Z, COMPUTE_64F_PEDANTIC) && compute_ok(Ty::Z, C_64F));
        assert!(!compute_ok(Ty::Z, COMPUTE_32F) && !compute_ok(Ty::D, C_64F));
        assert!(compute_ok(Ty::C, COMPUTE_32F_FAST_TF32) && compute_ok(Ty::C, C_32F));
        assert!(!compute_ok(Ty::C, COMPUTE_64F));
        assert_eq!(wide_type(R_32F), None);
        assert_eq!(
            Scalar::from_bytes(Ty::C, &[0, 0, 128, 63, 0, 0, 0, 64]),
            Scalar {
                re: 0x3f80_0000,
                im: 0x4000_0000,
                ptr: 0
            }
        );
    }

    #[test]
    fn reference_trsm_inverts_reference_gemm() {
        // B = op(A) X (left) / X op(A) (right); solving must give X back.
        let m = 4;
        let n = 3;
        for side in [SIDE_LEFT, SIDE_RIGHT] {
            for uplo in [FILL_LOWER, FILL_UPPER] {
                for trans in [OP_N, OP_T, OP_C] {
                    let order = if side == SIDE_LEFT { m } else { n };
                    let mut a = vec![(0.0, 0.0); order * order];
                    for r in 0..order {
                        for c in 0..order {
                            let inside = if uplo == FILL_LOWER { r >= c } else { r <= c };
                            a[r + c * order] = if inside {
                                (
                                    0.3 + (r * 7 + c * 3) as f64 * 0.1
                                        + if r == c { 2.0 } else { 0.0 },
                                    0.2 * c as f64 - 0.1 * r as f64,
                                )
                            } else {
                                (9.0, 9.0)
                            };
                        }
                    }
                    let mut tri_a = a.clone();
                    for r in 0..order {
                        for c in 0..order {
                            let inside = if uplo == FILL_LOWER { r >= c } else { r <= c };
                            if !inside {
                                tri_a[r + c * order] = (0.0, 0.0);
                            }
                        }
                    }
                    let x: Vec<Cx> = (0..m * n)
                        .map(|i| (i as f64 * 0.25 - 1.0, 0.5 - i as f64 * 0.125))
                        .collect();
                    let mut b = vec![(0.0, 0.0); m * n];
                    if side == SIDE_LEFT {
                        ref_gemm(
                            trans,
                            OP_N,
                            m,
                            n,
                            m,
                            (1.0, 0.0),
                            &tri_a,
                            m,
                            &x,
                            m,
                            (0.0, 0.0),
                            &mut b,
                            m,
                        );
                    } else {
                        ref_gemm(
                            OP_N,
                            trans,
                            m,
                            n,
                            n,
                            (1.0, 0.0),
                            &x,
                            m,
                            &tri_a,
                            n,
                            (0.0, 0.0),
                            &mut b,
                            m,
                        );
                    }
                    ref_trsm(
                        side,
                        uplo,
                        trans,
                        DIAG_NON_UNIT,
                        m,
                        n,
                        (1.0, 0.0),
                        &a,
                        order,
                        &mut b,
                        m,
                    );
                    for (g, w) in b.iter().zip(&x) {
                        assert!(
                            (g.0 - w.0).abs() < 1e-12 && (g.1 - w.1).abs() < 1e-12,
                            "side {side} uplo {uplo} trans {trans}"
                        );
                    }
                }
            }
        }
    }
}
