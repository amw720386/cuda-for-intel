//! f64 and complex cuBLAS (`cublas64_12.dll`) on the GPU against a host f64 reference:
//! D/C/Z gemm (every op pair including conjugate transposes, alpha/beta with NaN in the
//! unread operand, padded leading dimensions, k = 0), strided and pointer-table batches,
//! GemmEx/GemmStridedBatchedEx/GemmBatchedEx on R_64F/C_32F/C_64F, gemv with signed
//! increments, dot/dotc (host and device results, DotEx f16/bf16), nrm2 with extreme
//! magnitudes, axpy, scal, and trsm/trsmBatched over every side/uplo/trans/diag.
use super::cublas64_src::plan::{self, Cx};
use super::cublas64_src::ptxgen::Ty;
use super::{Handle, library, ok, symbols, test_device};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::ptr::null_mut;

type P = *mut c_void;
type C = *const c_void;

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn val(&mut self) -> f64 {
        (self.next() % 2001) as f64 / 1000.0 - 1.0
    }
    fn cx(&mut self, ty: Ty) -> Cx {
        let im = if ty.complex() { self.val() } else { 0.0 };
        round(ty, (self.val(), im))
    }
    fn vec(&mut self, ty: Ty, n: usize) -> Vec<Cx> {
        (0..n).map(|_| self.cx(ty)).collect()
    }
}

fn round(ty: Ty, v: Cx) -> Cx {
    match ty {
        Ty::D | Ty::Z => v,
        Ty::S | Ty::C => (f64::from(v.0 as f32), f64::from(v.1 as f32)),
        // 1/256 steps are exact in f16 and bf16 for |v| <= 1
        Ty::H | Ty::B => ((v.0 * 256.0).round() / 256.0, 0.0),
    }
}

fn enc(ty: Ty, v: &[Cx]) -> Vec<u8> {
    let mut out = Vec::new();
    for &(re, im) in v {
        let comps: &[f64] = if ty.complex() { &[re, im] } else { &[re] };
        for &c in comps {
            match ty {
                Ty::D | Ty::Z => out.extend(c.to_le_bytes()),
                Ty::S | Ty::C => out.extend((c as f32).to_le_bytes()),
                Ty::H => out.extend(f16_bits(c as f32).to_le_bytes()),
                Ty::B => out.extend((((c as f32).to_bits() >> 16) as u16).to_le_bytes()),
            }
        }
    }
    out
}

/// Exact for the 1/256-grid values the tests use.
fn f16_bits(v: f32) -> u16 {
    if v == 0.0 {
        return 0;
    }
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    sign | ((exp as u16) << 10) | ((b & 0x7f_ffff) >> 13) as u16
}

fn f16_value(h: u16) -> f64 {
    let e = i32::from((h >> 10) & 0x1f);
    let m = f64::from(h & 0x3ff);
    let v = if e == 0 {
        m * 2f64.powi(-24)
    } else {
        (1.0 + m / 1024.0) * 2f64.powi(e - 15)
    };
    if h & 0x8000 != 0 { -v } else { v }
}

fn dec(ty: Ty, b: &[u8]) -> Vec<Cx> {
    let c = ty.comp() as usize;
    let comp = |o: usize| -> f64 {
        match ty {
            Ty::H => f16_value(u16::from_le_bytes([b[o], b[o + 1]])),
            Ty::B => f64::from(f32::from_bits(
                u32::from(u16::from_le_bytes([b[o], b[o + 1]])) << 16,
            )),
            _ if c == 8 => f64::from_le_bytes(b[o..o + 8].try_into().unwrap()),
            _ => f64::from(f32::from_le_bytes(b[o..o + 4].try_into().unwrap())),
        }
    };
    let sz = ty.size() as usize;
    (0..b.len() / sz)
        .map(|i| {
            (
                comp(i * sz),
                if ty.complex() { comp(i * sz + c) } else { 0.0 },
            )
        })
        .collect()
}

fn max_err(got: &[Cx], want: &[Cx]) -> f64 {
    let scale = want
        .iter()
        .fold(1.0f64, |m, v| m.max(v.0.abs()).max(v.1.abs()));
    got.iter()
        .zip(want)
        .map(|(g, w)| {
            let same_nan = (g.0.is_nan() && w.0.is_nan()) || (g.1.is_nan() && w.1.is_nan());
            if same_nan {
                0.0
            } else if g.0.is_nan() || g.1.is_nan() {
                f64::INFINITY
            } else {
                (g.0 - w.0).abs().max((g.1 - w.1).abs())
            }
        })
        .fold(0.0, f64::max)
        / scale
}

fn tol(ty: Ty) -> f64 {
    if ty.double() { 1e-12 } else { 5e-5 }
}

fn api_type(ty: Ty) -> i32 {
    match ty {
        Ty::S => plan::R_32F,
        Ty::D => plan::R_64F,
        Ty::C => plan::C_32F,
        Ty::Z => plan::C_64F,
        Ty::H => plan::R_16F,
        Ty::B => plan::R_16BF,
    }
}

fn compute(ty: Ty) -> i32 {
    if ty.double() {
        plan::COMPUTE_64F
    } else {
        plan::COMPUTE_32F
    }
}

type Gemm =
    unsafe extern "system" fn(P, i32, i32, i32, i32, i32, C, C, i32, C, i32, C, P, i32) -> i32;
type GemmStrided = unsafe extern "system" fn(
    P,
    i32,
    i32,
    i32,
    i32,
    i32,
    C,
    C,
    i32,
    i64,
    C,
    i32,
    i64,
    C,
    P,
    i32,
    i64,
    i32,
) -> i32;
type GemmBatched =
    unsafe extern "system" fn(P, i32, i32, i32, i32, i32, C, C, i32, C, i32, C, P, i32, i32) -> i32;
type GemmEx = unsafe extern "system" fn(
    P,
    i32,
    i32,
    i32,
    i32,
    i32,
    C,
    C,
    i32,
    i32,
    C,
    i32,
    i32,
    C,
    P,
    i32,
    i32,
    i32,
    i32,
) -> i32;
type GemmStridedEx = unsafe extern "system" fn(
    P,
    i32,
    i32,
    i32,
    i32,
    i32,
    C,
    C,
    i32,
    i32,
    i64,
    C,
    i32,
    i32,
    i64,
    C,
    P,
    i32,
    i32,
    i64,
    i32,
    i32,
    i32,
) -> i32;
type GemmBatchedEx = unsafe extern "system" fn(
    P,
    i32,
    i32,
    i32,
    i32,
    i32,
    C,
    C,
    i32,
    i32,
    C,
    i32,
    i32,
    C,
    P,
    i32,
    i32,
    i32,
    i32,
    i32,
) -> i32;
type Gemv = unsafe extern "system" fn(P, i32, i32, i32, C, C, i32, C, i32, C, P, i32) -> i32;
type Dot = unsafe extern "system" fn(P, i32, C, i32, C, i32, P) -> i32;
type DotEx = unsafe extern "system" fn(P, i32, C, i32, i32, C, i32, i32, P, i32, i32) -> i32;
type Nrm2 = unsafe extern "system" fn(P, i32, C, i32, P) -> i32;
type Axpy = unsafe extern "system" fn(P, i32, C, C, i32, P, i32) -> i32;
type Scal = unsafe extern "system" fn(P, i32, C, P, i32) -> i32;
type Trsm = unsafe extern "system" fn(P, i32, i32, i32, i32, i32, i32, C, C, i32, P, i32) -> i32;
type TrsmBatched =
    unsafe extern "system" fn(P, i32, i32, i32, i32, i32, i32, C, C, i32, C, i32, i32) -> i32;

#[test]
#[ignore = "requires nvcuda.dll, cublas64_12.dll and an Intel Level Zero GPU with fp64"]
#[allow(non_snake_case, unused_unsafe)]
fn hw_cublas64_matches_host_reference() {
    let dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let lib =
        unsafe { libloading::Library::new(dir.join("cublas64_12.dll")) }.expect("cublas64_12.dll");
    let f = |name: &str| -> *const c_void {
        *unsafe { lib.get::<*const c_void>(format!("{name}\0").as_bytes()) }
            .unwrap_or_else(|_| panic!("{name}"))
    };
    macro_rules! sym {
        ($t:ty, $name:expr) => {
            unsafe { std::mem::transmute::<*const c_void, $t>(f(&$name)) }
        };
    }
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuMemFree_v2(u64),
        cuCtxSynchronize(),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let allocs = std::cell::RefCell::new(Vec::<u64>::new());
    let up = |bytes: &[u8]| -> u64 {
        let mut d = 0u64;
        ok!(cuMemAlloc_v2(&mut d, bytes.len().max(16)));
        if !bytes.is_empty() {
            ok!(cuMemcpyHtoD_v2(d, bytes.as_ptr().cast(), bytes.len()));
        }
        allocs.borrow_mut().push(d);
        d
    };
    let down = |d: u64, len: usize| -> Vec<u8> {
        ok!(cuCtxSynchronize());
        let mut b = vec![0u8; len];
        ok!(cuMemcpyDtoH_v2(b.as_mut_ptr().cast(), d, len));
        b
    };
    let create: unsafe extern "system" fn(*mut P) -> i32 = sym!(_, "cublasCreate_v2");
    let destroy: unsafe extern "system" fn(P) -> i32 = sym!(_, "cublasDestroy_v2");
    let set_stream: unsafe extern "system" fn(P, Handle) -> i32 = sym!(_, "cublasSetStream_v2");
    let set_mode: unsafe extern "system" fn(P, i32) -> i32 = sym!(_, "cublasSetPointerMode_v2");
    let mut h: P = null_mut();
    assert_eq!(unsafe { create(&mut h) }, 0);
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 0));
    assert_eq!(unsafe { set_stream(h, stream) }, 0);

    let mut rng = Rng(0x5eed_f00d_1234_5678);
    let mut worst: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let mut record = |key: String, err: f64, limit: f64| {
        assert!(err <= limit, "{key}: error {err:e} > {limit:e}");
        let e = worst.entry(key).or_insert((0.0, 0));
        e.0 = e.0.max(err);
        e.1 += 1;
    };
    let ops = |ty: Ty| -> &'static [i32] {
        if ty.complex() {
            &[plan::OP_N, plan::OP_T, plan::OP_C]
        } else {
            &[plan::OP_N, plan::OP_T]
        }
    };
    let scalars = [
        ((1.0, 0.0), (0.0, 0.0)),
        ((0.75, -0.5), (-0.5, 0.25)),
        ((0.0, 0.0), (1.5, -1.0)),
    ];

    // ---- gemm: plain, StridedBatched, Batched, and the Ex forms -------------------------
    for ty in [Ty::D, Ty::C, Ty::Z] {
        let t = ty.tag().to_uppercase();
        let gemm: Gemm = sym!(Gemm, format!("cublas{t}gemm_v2"));
        let strided: GemmStrided = sym!(GemmStrided, format!("cublas{t}gemmStridedBatched"));
        let batched: GemmBatched = sym!(GemmBatched, format!("cublas{t}gemmBatched"));
        let gemm_ex: GemmEx = sym!(GemmEx, "cublasGemmEx");
        let strided_ex: GemmStridedEx = sym!(GemmStridedEx, "cublasGemmStridedBatchedEx");
        let batched_ex: GemmBatchedEx = sym!(GemmBatchedEx, "cublasGemmBatchedEx");
        let sizes = [
            (1usize, 1usize, 1usize),
            (37, 29, 41),
            (64, 64, 64),
            (200, 130, 70),
            (5, 300, 3),
            (33, 17, 0),
        ];
        for (si, &(m, n, k)) in sizes.iter().enumerate() {
            for &ta in ops(ty) {
                for &tb in ops(ty) {
                    let last = *ops(ty).last().unwrap();
                    if si >= 3
                        && (ta, tb) != (plan::OP_N, plan::OP_N)
                        && (ta, tb) != (last, plan::OP_T)
                    {
                        continue;
                    }
                    let (alpha, beta) = scalars[(si + ta as usize + tb as usize) % 3];
                    let (alpha, beta) = (
                        round(ty, (alpha.0, if ty.complex() { alpha.1 } else { 0.0 })),
                        round(ty, (beta.0, if ty.complex() { beta.1 } else { 0.0 })),
                    );
                    let (ra, ca) = if ta == plan::OP_N { (m, k) } else { (k, m) };
                    let (rb, cb) = if tb == plan::OP_N { (k, n) } else { (n, k) };
                    let (lda, ldb, ldc) = (ra.max(1) + 3, rb.max(1) + 1, m + 2);
                    let batch = 3usize;
                    let (sa, sb, sc) = (lda * ca.max(1) + 7, ldb * cb.max(1), ldc * n + 5);
                    let mut a = rng.vec(ty, sa * batch);
                    let b = rng.vec(ty, sb * batch);
                    let mut c = rng.vec(ty, sc * batch);
                    if beta == (0.0, 0.0) {
                        c[0] = (f64::NAN, f64::NAN);
                    }
                    if alpha == (0.0, 0.0) {
                        a[0] = (f64::NAN, 0.0);
                    }
                    let mut want = c.clone();
                    for z in 0..batch {
                        plan::ref_gemm(
                            ta,
                            tb,
                            m,
                            n,
                            k,
                            alpha,
                            &a[z * sa..],
                            lda,
                            &b[z * sb..],
                            ldb,
                            beta,
                            &mut want[z * sc..],
                            ldc,
                        );
                    }
                    let (da, db) = (up(&enc(ty, &a)), up(&enc(ty, &b)));
                    let (al, be) = (enc(ty, &[alpha]), enc(ty, &[beta]));
                    let (al, be) = (al.as_ptr().cast::<c_void>(), be.as_ptr().cast::<c_void>());
                    let (mi, ni, ki, ldai, ldbi, ldci) = (
                        m as i32, n as i32, k as i32, lda as i32, ldb as i32, ldc as i32,
                    );
                    let esz = ty.size() as u64;
                    let tables = |base: u64, s: usize| {
                        up(&(0..batch)
                            .flat_map(|z| (base + (z * s) as u64 * esz).to_le_bytes())
                            .collect::<Vec<u8>>())
                    };
                    let at = api_type(ty);
                    for form in 0..6 {
                        let dc = up(&enc(ty, &c));
                        let (st, nb) = unsafe {
                            match form {
                                0 => (
                                    gemm(
                                        h, ta, tb, mi, ni, ki, al, da as C, ldai, db as C, ldbi,
                                        be, dc as P, ldci,
                                    ),
                                    1,
                                ),
                                1 => (
                                    strided(
                                        h,
                                        ta,
                                        tb,
                                        mi,
                                        ni,
                                        ki,
                                        al,
                                        da as C,
                                        ldai,
                                        sa as i64,
                                        db as C,
                                        ldbi,
                                        sb as i64,
                                        be,
                                        dc as P,
                                        ldci,
                                        sc as i64,
                                        batch as i32,
                                    ),
                                    batch,
                                ),
                                2 => (
                                    batched(
                                        h,
                                        ta,
                                        tb,
                                        mi,
                                        ni,
                                        ki,
                                        al,
                                        tables(da, sa) as C,
                                        ldai,
                                        tables(db, sb) as C,
                                        ldbi,
                                        be,
                                        tables(dc, sc) as P,
                                        ldci,
                                        batch as i32,
                                    ),
                                    batch,
                                ),
                                3 => (
                                    gemm_ex(
                                        h,
                                        ta,
                                        tb,
                                        mi,
                                        ni,
                                        ki,
                                        al,
                                        da as C,
                                        at,
                                        ldai,
                                        db as C,
                                        at,
                                        ldbi,
                                        be,
                                        dc as P,
                                        at,
                                        ldci,
                                        compute(ty),
                                        -1,
                                    ),
                                    1,
                                ),
                                4 => (
                                    strided_ex(
                                        h,
                                        ta,
                                        tb,
                                        mi,
                                        ni,
                                        ki,
                                        al,
                                        da as C,
                                        at,
                                        ldai,
                                        sa as i64,
                                        db as C,
                                        at,
                                        ldbi,
                                        sb as i64,
                                        be,
                                        dc as P,
                                        at,
                                        ldci,
                                        sc as i64,
                                        batch as i32,
                                        compute(ty),
                                        -1,
                                    ),
                                    batch,
                                ),
                                _ => (
                                    batched_ex(
                                        h,
                                        ta,
                                        tb,
                                        mi,
                                        ni,
                                        ki,
                                        al,
                                        tables(da, sa) as C,
                                        at,
                                        ldai,
                                        tables(db, sb) as C,
                                        at,
                                        ldbi,
                                        be,
                                        tables(dc, sc) as P,
                                        at,
                                        ldci,
                                        batch as i32,
                                        compute(ty),
                                        -1,
                                    ),
                                    batch,
                                ),
                            }
                        };
                        assert_eq!(st, 0, "{t}gemm form {form} status");
                        let got = dec(ty, &down(dc, c.len() * esz as usize));
                        // matrices beyond `nb` and the padding rows must be untouched
                        let mut w = c.clone();
                        for z in 0..nb {
                            for j in 0..n {
                                for i in 0..m {
                                    w[z * sc + i + j * ldc] = want[z * sc + i + j * ldc];
                                }
                            }
                        }
                        let name = [
                            "gemm",
                            "gemmStridedBatched",
                            "gemmBatched",
                            "GemmEx",
                            "GemmStridedBatchedEx",
                            "GemmBatchedEx",
                        ][form];
                        record(
                            format!("{t} {name}"),
                            max_err(&got, &w),
                            tol(ty) * (k.max(1) as f64).sqrt(),
                        );
                    }
                }
            }
        }
        // device pointer mode scalars, one strided batch
        assert_eq!(unsafe { set_mode(h, 1) }, 0);
        let (m, n, k, batch) = (19usize, 23usize, 31usize, 2usize);
        let (alpha, beta) = (rng.cx(ty), rng.cx(ty));
        let a = rng.vec(ty, m * k * batch);
        let b = rng.vec(ty, k * n * batch);
        let c = rng.vec(ty, m * n * batch);
        let mut want = c.clone();
        for z in 0..batch {
            plan::ref_gemm(
                plan::OP_N,
                plan::OP_N,
                m,
                n,
                k,
                alpha,
                &a[z * m * k..],
                m,
                &b[z * k * n..],
                k,
                beta,
                &mut want[z * m * n..],
                m,
            );
        }
        let (da, db, dc) = (up(&enc(ty, &a)), up(&enc(ty, &b)), up(&enc(ty, &c)));
        let (dal, dbe) = (up(&enc(ty, &[alpha])), up(&enc(ty, &[beta])));
        let st = unsafe {
            strided(
                h,
                0,
                0,
                m as i32,
                n as i32,
                k as i32,
                dal as C,
                da as C,
                m as i32,
                (m * k) as i64,
                db as C,
                k as i32,
                (k * n) as i64,
                dbe as C,
                dc as P,
                m as i32,
                (m * n) as i64,
                batch as i32,
            )
        };
        assert_eq!(st, 0);
        record(
            format!("{t} gemm device scalars"),
            max_err(&dec(ty, &down(dc, c.len() * ty.size() as usize)), &want),
            tol(ty) * 8.0,
        );
        assert_eq!(unsafe { set_mode(h, 0) }, 0);
    }

    // ---- gemv ---------------------------------------------------------------------------
    for ty in Ty::BLAS {
        let t = ty.tag().to_uppercase();
        let gemv: Gemv = sym!(Gemv, format!("cublas{t}gemv_v2"));
        for &trans in ops(ty) {
            for (m, n, incx, incy) in [
                (1i32, 1i32, 1i32, 1i32),
                (37, 29, 1, 1),
                (300, 211, 2, -1),
                (5, 700, -3, 2),
                (1000, 64, 1, 1),
            ] {
                let (alpha, beta) = (rng.cx(ty), if m == 37 { (0.0, 0.0) } else { rng.cx(ty) });
                let lda = m as usize + 4;
                let a = rng.vec(ty, lda * n as usize);
                let (xl, yl) = if trans == plan::OP_N {
                    (n as usize, m as usize)
                } else {
                    (m as usize, n as usize)
                };
                let x = rng.vec(ty, xl * incx.unsigned_abs() as usize);
                let mut y = rng.vec(ty, yl * incy.unsigned_abs() as usize);
                if beta == (0.0, 0.0) {
                    y.iter_mut().for_each(|v| *v = (f64::NAN, 0.0));
                }
                let pos = |len: usize, inc: i32, i: usize| {
                    if inc > 0 {
                        i * inc as usize
                    } else {
                        (len - 1 - i) * (-inc) as usize
                    }
                };
                let mut want = y.clone();
                for r in 0..yl {
                    let mut acc = (0.0, 0.0);
                    for l in 0..xl {
                        acc = plan::cadd(
                            acc,
                            plan::cmul(plan::op_at(&a, lda, trans, r, l), x[pos(xl, incx, l)]),
                        );
                    }
                    let yi = pos(yl, incy, r);
                    let mut v = plan::cmul(alpha, acc);
                    if beta != (0.0, 0.0) {
                        v = plan::cadd(v, plan::cmul(beta, y[yi]));
                    }
                    want[yi] = v;
                }
                let (da, dx, dy) = (up(&enc(ty, &a)), up(&enc(ty, &x)), up(&enc(ty, &y)));
                let (al, be) = (enc(ty, &[alpha]), enc(ty, &[beta]));
                let st = unsafe {
                    gemv(
                        h,
                        trans,
                        m,
                        n,
                        al.as_ptr().cast(),
                        da as C,
                        lda as i32,
                        dx as C,
                        incx,
                        be.as_ptr().cast(),
                        dy as P,
                        incy,
                    )
                };
                assert_eq!(st, 0, "{t}gemv status");
                record(
                    format!("{t} gemv"),
                    max_err(&dec(ty, &down(dy, y.len() * ty.size() as usize)), &want),
                    tol(ty) * (xl as f64).sqrt(),
                );
            }
        }
    }

    // ---- dot / dotc / DotEx, host and device results ------------------------------------
    let dot_ex: DotEx = sym!(DotEx, "cublasDotEx");
    for ty in Ty::ALL {
        let t = ty.tag().to_uppercase();
        let variants: Vec<(String, bool)> = match ty {
            Ty::S | Ty::D => vec![(format!("cublas{t}dot_v2"), false)],
            Ty::C | Ty::Z => vec![
                (format!("cublas{t}dotu_v2"), false),
                (format!("cublas{t}dotc_v2"), true),
            ],
            _ => vec![("cublasDotEx".into(), false)],
        };
        for (name, conj) in variants {
            for (n, incx, incy) in [
                (0i32, 1i32, 1i32),
                (1, 1, 1),
                (1000, 1, 1),
                (4097, 1, 1),
                (333, -2, 3),
            ] {
                let len = |inc: i32| (n.max(1) as usize) * inc.unsigned_abs() as usize;
                let (x, y) = (rng.vec(ty, len(incx)), rng.vec(ty, len(incy)));
                let pos = |inc: i32, i: usize| {
                    if inc > 0 {
                        i * inc as usize
                    } else {
                        (n as usize - 1 - i) * (-inc) as usize
                    }
                };
                let mut want = (0.0, 0.0);
                for i in 0..n as usize {
                    let xv = x[pos(incx, i)];
                    want = plan::cadd(
                        want,
                        plan::cmul(if conj { (xv.0, -xv.1) } else { xv }, y[pos(incy, i)]),
                    );
                }
                let (dx, dy) = (up(&enc(ty, &x)), up(&enc(ty, &y)));
                for device in [false, true] {
                    assert_eq!(unsafe { set_mode(h, device as i32) }, 0);
                    let dres = up(&[0xAAu8; 16]);
                    let mut host = [0u8; 16];
                    let res: P = if device {
                        dres as P
                    } else {
                        host.as_mut_ptr().cast()
                    };
                    let st = unsafe {
                        if matches!(ty, Ty::H | Ty::B) {
                            dot_ex(
                                h,
                                n,
                                dx as C,
                                api_type(ty),
                                incx,
                                dy as C,
                                api_type(ty),
                                incy,
                                res,
                                api_type(ty),
                                plan::R_32F,
                            )
                        } else {
                            let d: Dot = sym!(Dot, name);
                            d(h, n, dx as C, incx, dy as C, incy, res)
                        }
                    };
                    assert_eq!(st, 0, "{name} status");
                    let bytes = if device {
                        down(dres, 16)
                    } else {
                        host.to_vec()
                    };
                    let got = dec(ty, &bytes[..ty.size() as usize]);
                    let limit = match ty {
                        Ty::H => 2e-3 * f64::from(n.max(1)).sqrt() + 1e-3,
                        Ty::B => 2e-2,
                        _ => tol(ty) * f64::from(n.max(1)).sqrt(),
                    };
                    let scale = want.0.abs().max(want.1.abs()).max(1.0);
                    let lim = if matches!(ty, Ty::H | Ty::B) {
                        limit * scale.max(1.0)
                    } else {
                        limit
                    };
                    record(
                        format!(
                            "{t} {}{}",
                            name.trim_start_matches("cublas"),
                            if device { " (device)" } else { "" }
                        ),
                        max_err(&got, &[want]),
                        lim,
                    );
                }
                assert_eq!(unsafe { set_mode(h, 0) }, 0);
            }
        }
    }

    // ---- nrm2 ---------------------------------------------------------------------------
    for (ty, name) in [
        (Ty::S, "cublasSnrm2_v2"),
        (Ty::D, "cublasDnrm2_v2"),
        (Ty::C, "cublasScnrm2_v2"),
        (Ty::Z, "cublasDznrm2_v2"),
    ] {
        let nrm2: Nrm2 = sym!(Nrm2, name);
        for (n, incx) in [(1i32, 1i32), (1000, 1), (4097, 3)] {
            for scale in [
                1.0,
                if ty.double() { 1e200 } else { 1e30 },
                if ty.double() { 1e-200 } else { 1e-30 },
            ] {
                let x: Vec<Cx> = rng
                    .vec(ty, n as usize * incx as usize)
                    .iter()
                    .map(|v| round(ty, (v.0 * scale, v.1 * scale)))
                    .collect();
                let want = (0..n as usize)
                    .map(|i| x[i * incx as usize])
                    .map(|v| (v.0 / scale).powi(2) + (v.1 / scale).powi(2))
                    .sum::<f64>()
                    .sqrt()
                    * scale;
                let dx = up(&enc(ty, &x));
                let mut r = [0u8; 8];
                assert_eq!(
                    unsafe { nrm2(h, n, dx as C, incx, r.as_mut_ptr().cast()) },
                    0
                );
                let got = if ty.double() {
                    f64::from_le_bytes(r)
                } else {
                    f64::from(f32::from_le_bytes(r[..4].try_into().unwrap()))
                };
                record(
                    format!("{name} (rel)"),
                    (got - want).abs() / want,
                    tol(ty) * 4.0,
                );
            }
        }
    }

    // ---- axpy / scal --------------------------------------------------------------------
    for ty in Ty::BLAS {
        let t = ty.tag().to_uppercase();
        let axpy: Axpy = sym!(Axpy, format!("cublas{t}axpy_v2"));
        let n = 5000i32;
        for (incx, incy) in [(1i32, 1i32), (-2, 3)] {
            let alpha = rng.cx(ty);
            let len = |inc: i32| n as usize * inc.unsigned_abs() as usize;
            let (x, y) = (rng.vec(ty, len(incx)), rng.vec(ty, len(incy)));
            let pos = |inc: i32, i: usize| {
                if inc > 0 {
                    i * inc as usize
                } else {
                    (n as usize - 1 - i) * (-inc) as usize
                }
            };
            let mut want = y.clone();
            for i in 0..n as usize {
                want[pos(incy, i)] =
                    plan::cadd(want[pos(incy, i)], plan::cmul(alpha, x[pos(incx, i)]));
            }
            let (dx, dy) = (up(&enc(ty, &x)), up(&enc(ty, &y)));
            let al = enc(ty, &[alpha]);
            assert_eq!(
                unsafe { axpy(h, n, al.as_ptr().cast(), dx as C, incx, dy as P, incy) },
                0
            );
            record(
                format!("{t} axpy"),
                max_err(&dec(ty, &down(dy, y.len() * ty.size() as usize)), &want),
                tol(ty),
            );
        }
        let mut scals = vec![(format!("cublas{t}scal_v2"), ty)];
        if ty == Ty::C {
            scals.push(("cublasCsscal_v2".into(), Ty::S));
        }
        if ty == Ty::Z {
            scals.push(("cublasZdscal_v2".into(), Ty::D));
        }
        for (name, aty) in scals {
            let scal: Scal = sym!(Scal, name);
            let alpha = rng.cx(aty);
            let x = rng.vec(ty, n as usize * 2);
            let mut want = x.clone();
            for i in 0..n as usize {
                want[i * 2] = plan::cmul(alpha, x[i * 2]);
            }
            let dx = up(&enc(ty, &x));
            let al = enc(aty, &[alpha]);
            assert_eq!(unsafe { scal(h, n, al.as_ptr().cast(), dx as P, 2) }, 0);
            record(
                name.trim_start_matches("cublas").to_string(),
                max_err(&dec(ty, &down(dx, x.len() * ty.size() as usize)), &want),
                tol(ty),
            );
        }
    }

    // ---- trsm / trsmBatched -------------------------------------------------------------
    for ty in Ty::BLAS {
        let t = ty.tag().to_uppercase();
        let trsm: Trsm = sym!(Trsm, format!("cublas{t}trsm_v2"));
        let batched: Option<TrsmBatched> =
            (ty != Ty::S).then(|| sym!(TrsmBatched, format!("cublas{t}trsmBatched")));
        for side in [plan::SIDE_LEFT, plan::SIDE_RIGHT] {
            for uplo in [plan::FILL_LOWER, plan::FILL_UPPER] {
                for &trans in ops(ty) {
                    for diag in [plan::DIAG_NON_UNIT, plan::DIAG_UNIT] {
                        let (m, n) = (33usize, 17usize);
                        let order = if side == plan::SIDE_LEFT { m } else { n };
                        let (lda, ldb, batch) = (order + 3, m + 2, 4usize);
                        let alpha = rng.cx(ty);
                        let mut a: Vec<Cx> = rng
                            .vec(ty, lda * order * batch)
                            .iter()
                            .map(|v| round(ty, (v.0 * 0.2, v.1 * 0.2)))
                            .collect();
                        for z in 0..batch {
                            for d in 0..order {
                                let v = &mut a[z * lda * order + d + d * lda];
                                *v = round(ty, (v.0 + 1.5, v.1));
                            }
                        }
                        let b = rng.vec(ty, ldb * n * batch);
                        let mut want = b.clone();
                        for z in 0..batch {
                            plan::ref_trsm(
                                side,
                                uplo,
                                trans,
                                diag,
                                m,
                                n,
                                alpha,
                                &a[z * lda * order..],
                                lda,
                                &mut want[z * ldb * n..],
                                ldb,
                            );
                        }
                        let al = enc(ty, &[alpha]);
                        let esz = ty.size() as u64;
                        // non-batched on matrix 0
                        let (da, db) = (up(&enc(ty, &a)), up(&enc(ty, &b)));
                        let st = unsafe {
                            trsm(
                                h,
                                side,
                                uplo,
                                trans,
                                diag,
                                m as i32,
                                n as i32,
                                al.as_ptr().cast(),
                                da as C,
                                lda as i32,
                                db as P,
                                ldb as i32,
                            )
                        };
                        assert_eq!(st, 0, "{t}trsm status");
                        let got = dec(ty, &down(db, ldb * n * esz as usize));
                        record(
                            format!("{t} trsm"),
                            max_err(&got, &want[..ldb * n]),
                            tol(ty) * 50.0,
                        );
                        if let Some(bt) = batched {
                            let db = up(&enc(ty, &b));
                            let ta = up(&(0..batch)
                                .flat_map(|z| (da + (z * lda * order) as u64 * esz).to_le_bytes())
                                .collect::<Vec<u8>>());
                            let tb = up(&(0..batch)
                                .flat_map(|z| (db + (z * ldb * n) as u64 * esz).to_le_bytes())
                                .collect::<Vec<u8>>());
                            let st = unsafe {
                                bt(
                                    h,
                                    side,
                                    uplo,
                                    trans,
                                    diag,
                                    m as i32,
                                    n as i32,
                                    al.as_ptr().cast(),
                                    ta as C,
                                    lda as i32,
                                    tb as C,
                                    ldb as i32,
                                    batch as i32,
                                )
                            };
                            assert_eq!(st, 0, "{t}trsmBatched status");
                            let got = dec(ty, &down(db, b.len() * esz as usize));
                            record(
                                format!("{t} trsmBatched"),
                                max_err(&got, &want),
                                tol(ty) * 50.0,
                            );
                        }
                    }
                }
            }
        }
    }

    println!("| routine | cases | worst error |\n|---|---|---|");
    for (k, (e, n)) in &worst {
        println!("| {k} | {n} | {e:.1e} |");
    }
    assert_eq!(unsafe { destroy(h) }, 0);
    ok!(cuStreamDestroy_v2(stream));
    for d in allocs.into_inner() {
        ok!(cuMemFree_v2(d));
    }
    ok!(cuCtxDestroy_v2(ctx));
}

fn conj(v: Cx) -> Cx {
    (v.0, -v.1)
}

/// Routines cuSOLVER and MAGMA call besides gemm/trsm, through `cublas64_12.dll`:
/// syrk/herk, syrkx/herkx, syr2k/her2k, symm/hemm, trmm (out of place and in place), trmv,
/// trsv, symv/hemv, ger/geru/gerc, geam, dgmm, copy, swap, i?amax/i?amin and the
/// Set/Get Matrix/Vector copies, against a host f64 reference.
#[test]
#[ignore = "requires nvcuda.dll, cublas64_12.dll and an Intel Level Zero GPU with fp64"]
#[allow(non_snake_case, unused_unsafe, clippy::type_complexity)]
fn hw_cublas64_solver_helpers() {
    let dir = std::env::current_exe()
        .unwrap()
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf();
    let lib =
        unsafe { libloading::Library::new(dir.join("cublas64_12.dll")) }.expect("cublas64_12.dll");
    let f = |name: &str| -> *const c_void {
        *unsafe { lib.get::<*const c_void>(format!("{name}\0").as_bytes()) }
            .unwrap_or_else(|_| panic!("{name}"))
    };
    macro_rules! sym {
        ($t:ty, $name:expr) => {
            unsafe { std::mem::transmute::<*const c_void, $t>(f(&$name)) }
        };
    }
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuMemFree_v2(u64),
        cuCtxSynchronize(),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let allocs = std::cell::RefCell::new(Vec::<u64>::new());
    let up = |bytes: &[u8]| -> u64 {
        let mut d = 0u64;
        ok!(cuMemAlloc_v2(&mut d, bytes.len().max(16)));
        if !bytes.is_empty() {
            ok!(cuMemcpyHtoD_v2(d, bytes.as_ptr().cast(), bytes.len()));
        }
        allocs.borrow_mut().push(d);
        d
    };
    let down = |d: u64, len: usize| -> Vec<u8> {
        ok!(cuCtxSynchronize());
        let mut b = vec![0u8; len];
        ok!(cuMemcpyDtoH_v2(b.as_mut_ptr().cast(), d, len));
        b
    };
    let create: unsafe extern "system" fn(*mut P) -> i32 = sym!(_, "cublasCreate_v2");
    let destroy: unsafe extern "system" fn(P) -> i32 = sym!(_, "cublasDestroy_v2");
    let mut h: P = null_mut();
    assert_eq!(unsafe { create(&mut h) }, 0);
    let mut rng = Rng(0xabad_cafe_0042);
    let mut worst: BTreeMap<String, (f64, usize)> = BTreeMap::new();
    let mut record = |key: String, err: f64, limit: f64| {
        assert!(err <= limit, "{key}: error {err:e} > {limit:e}");
        let e = worst.entry(key).or_insert((0.0, 0));
        e.0 = e.0.max(err);
        e.1 += 1;
    };
    let at = |m: &[Cx], ld: usize, r: usize, c: usize| m[r + c * ld];
    for ty in Ty::BLAS {
        let t = ty.tag().to_uppercase();
        let cx = ty.complex();
        let es = ty.size() as usize;
        let rt = if ty.double() { Ty::D } else { Ty::S };
        let lim = tol(ty) * 10.0;
        // ---- rank-k family ----
        type RankK = unsafe extern "system" fn(P, i32, i32, i32, i32, C, C, i32, C, P, i32) -> i32;
        type RankKx =
            unsafe extern "system" fn(P, i32, i32, i32, i32, C, C, i32, C, i32, C, P, i32) -> i32;
        let mut fams: Vec<(String, bool, u8)> = vec![
            (format!("cublas{t}syrk_v2"), false, 0),
            (format!("cublas{t}syrkx"), false, 1),
            (format!("cublas{t}syr2k_v2"), false, 2),
        ];
        if cx {
            fams.extend([
                (format!("cublas{t}herk_v2"), true, 0),
                (format!("cublas{t}herkx"), true, 1),
                (format!("cublas{t}her2k_v2"), true, 2),
            ]);
        }
        for (name, herm, kind) in fams {
            let transes: &[i32] = if herm {
                &[plan::OP_N, plan::OP_C]
            } else {
                &[plan::OP_N, plan::OP_T]
            };
            for &trans in transes {
                for uplo in [plan::FILL_LOWER, plan::FILL_UPPER] {
                    let (n, k) = (70usize, 45usize);
                    let (ra, ca) = if trans == plan::OP_N { (n, k) } else { (k, n) };
                    let (lda, ldc) = (ra + 3, n + 2);
                    let (a, b, c0) = (
                        rng.vec(ty, lda * ca),
                        rng.vec(ty, lda * ca),
                        rng.vec(ty, ldc * n),
                    );
                    let alpha = if herm && kind == 0 {
                        (rng.val(), 0.0)
                    } else {
                        rng.cx(ty)
                    };
                    let beta = if herm { (rng.val(), 0.0) } else { rng.cx(ty) };
                    let fop = |x: &[Cx], i: usize, l: usize| plan::op_at(x, lda, trans, i, l);
                    let g = |y: &[Cx], l: usize, j: usize| {
                        let v = plan::op_at(y, lda, trans, j, l);
                        if herm { conj(v) } else { v }
                    };
                    let second = if herm { conj(alpha) } else { alpha };
                    let pairs: Vec<(&[Cx], &[Cx], Cx)> = match kind {
                        0 => vec![(&a, &a, alpha)],
                        1 => vec![(&a, &b, alpha)],
                        _ => vec![(&a, &b, alpha), (&b, &a, second)],
                    };
                    let mut want = c0.clone();
                    for j in 0..n {
                        for i in 0..n {
                            if (uplo == plan::FILL_LOWER && i < j)
                                || (uplo == plan::FILL_UPPER && i > j)
                            {
                                continue;
                            }
                            let mut v = plan::cmul(beta, c0[i + j * ldc]);
                            for &(x, y, al) in &pairs {
                                let mut acc = (0.0, 0.0);
                                for l in 0..k {
                                    acc = plan::cadd(acc, plan::cmul(fop(x, i, l), g(y, l, j)));
                                }
                                v = plan::cadd(v, plan::cmul(al, acc));
                            }
                            if herm && i == j {
                                v.1 = 0.0;
                            }
                            want[i + j * ldc] = v;
                        }
                    }
                    let (da, db, dc) = (up(&enc(ty, &a)), up(&enc(ty, &b)), up(&enc(ty, &c0)));
                    let sty = |real: bool| if real { rt } else { ty };
                    let al = enc(sty(herm && kind == 0), &[alpha]);
                    let be = enc(sty(herm), &[beta]);
                    let st = unsafe {
                        if kind == 0 {
                            let fnc: RankK = sym!(RankK, name);
                            fnc(
                                h,
                                uplo,
                                trans,
                                n as i32,
                                k as i32,
                                al.as_ptr().cast(),
                                da as C,
                                lda as i32,
                                be.as_ptr().cast(),
                                dc as P,
                                ldc as i32,
                            )
                        } else {
                            let fnc: RankKx = sym!(RankKx, name);
                            fnc(
                                h,
                                uplo,
                                trans,
                                n as i32,
                                k as i32,
                                al.as_ptr().cast(),
                                da as C,
                                lda as i32,
                                db as C,
                                lda as i32,
                                be.as_ptr().cast(),
                                dc as P,
                                ldc as i32,
                            )
                        }
                    };
                    assert_eq!(st, 0, "{name} status");
                    record(
                        name.trim_start_matches("cublas").to_string(),
                        max_err(&dec(ty, &down(dc, c0.len() * es)), &want),
                        lim,
                    );
                }
            }
        }
        // ---- symm / hemm, symv / hemv ----
        type Symm =
            unsafe extern "system" fn(P, i32, i32, i32, i32, C, C, i32, C, i32, C, P, i32) -> i32;
        type Symv = unsafe extern "system" fn(P, i32, i32, C, C, i32, C, i32, C, P, i32) -> i32;
        for herm in [false, true] {
            if herm && !cx {
                continue;
            }
            let (mname, vname) = if herm {
                (format!("cublas{t}hemm_v2"), format!("cublas{t}hemv_v2"))
            } else {
                (format!("cublas{t}symm_v2"), format!("cublas{t}symv_v2"))
            };
            let symm: Symm = sym!(Symm, mname);
            let symv: Symv = sym!(Symv, vname);
            for side in [plan::SIDE_LEFT, plan::SIDE_RIGHT] {
                for uplo in [plan::FILL_LOWER, plan::FILL_UPPER] {
                    let (m, n) = (53usize, 38usize);
                    let ord = if side == plan::SIDE_LEFT { m } else { n };
                    let (lda, ldb, ldc) = (ord + 1, m + 2, m + 1);
                    let a = rng.vec(ty, lda * ord);
                    let full = |r: usize, c: usize| -> Cx {
                        let inside = if uplo == plan::FILL_LOWER {
                            r >= c
                        } else {
                            r <= c
                        };
                        let v = if inside {
                            at(&a, lda, r, c)
                        } else {
                            at(&a, lda, c, r)
                        };
                        let v = if herm && !inside { conj(v) } else { v };
                        if herm && r == c { (v.0, 0.0) } else { v }
                    };
                    let (b, c0) = (rng.vec(ty, ldb * n), rng.vec(ty, ldc * n));
                    let (alpha, beta) = (rng.cx(ty), rng.cx(ty));
                    let mut want = c0.clone();
                    for j in 0..n {
                        for i in 0..m {
                            let mut acc = (0.0, 0.0);
                            for l in 0..ord {
                                acc = plan::cadd(
                                    acc,
                                    if side == plan::SIDE_LEFT {
                                        plan::cmul(full(i, l), at(&b, ldb, l, j))
                                    } else {
                                        plan::cmul(at(&b, ldb, i, l), full(l, j))
                                    },
                                );
                            }
                            want[i + j * ldc] = plan::cadd(
                                plan::cmul(alpha, acc),
                                plan::cmul(beta, c0[i + j * ldc]),
                            );
                        }
                    }
                    let (da, db, dc) = (up(&enc(ty, &a)), up(&enc(ty, &b)), up(&enc(ty, &c0)));
                    let (al, be) = (enc(ty, &[alpha]), enc(ty, &[beta]));
                    assert_eq!(
                        unsafe {
                            symm(
                                h,
                                side,
                                uplo,
                                m as i32,
                                n as i32,
                                al.as_ptr().cast(),
                                da as C,
                                lda as i32,
                                db as C,
                                ldb as i32,
                                be.as_ptr().cast(),
                                dc as P,
                                ldc as i32,
                            )
                        },
                        0
                    );
                    record(
                        mname.trim_start_matches("cublas").to_string(),
                        max_err(&dec(ty, &down(dc, c0.len() * es)), &want),
                        lim,
                    );
                    if side == plan::SIDE_LEFT {
                        let (x, y0) = (rng.vec(ty, 2 * m), rng.vec(ty, 3 * m));
                        let mut want = y0.clone();
                        for i in 0..m {
                            let mut acc = (0.0, 0.0);
                            for l in 0..m {
                                acc = plan::cadd(acc, plan::cmul(full(i, l), x[(m - 1 - l) * 2]));
                            }
                            want[3 * i] =
                                plan::cadd(plan::cmul(alpha, acc), plan::cmul(beta, y0[3 * i]));
                        }
                        let (dx, dy) = (up(&enc(ty, &x)), up(&enc(ty, &y0)));
                        assert_eq!(
                            unsafe {
                                symv(
                                    h,
                                    uplo,
                                    m as i32,
                                    al.as_ptr().cast(),
                                    da as C,
                                    lda as i32,
                                    dx as C,
                                    -2,
                                    be.as_ptr().cast(),
                                    dy as P,
                                    3,
                                )
                            },
                            0
                        );
                        record(
                            vname.trim_start_matches("cublas").to_string(),
                            max_err(&dec(ty, &down(dy, y0.len() * es)), &want),
                            lim,
                        );
                    }
                }
            }
        }
        // ---- trmm (out of place and in place), trmv, trsv ----
        type Trmm = unsafe extern "system" fn(
            P,
            i32,
            i32,
            i32,
            i32,
            i32,
            i32,
            C,
            C,
            i32,
            C,
            i32,
            P,
            i32,
        ) -> i32;
        type Trmv = unsafe extern "system" fn(P, i32, i32, i32, i32, C, i32, P, i32) -> i32;
        let trmm: Trmm = sym!(Trmm, format!("cublas{t}trmm_v2"));
        let trmv: Trmv = sym!(Trmv, format!("cublas{t}trmv_v2"));
        let trsv: Trmv = sym!(Trmv, format!("cublas{t}trsv_v2"));
        let transes: &[i32] = if cx {
            &[plan::OP_N, plan::OP_T, plan::OP_C]
        } else {
            &[plan::OP_N, plan::OP_T]
        };
        for side in [plan::SIDE_LEFT, plan::SIDE_RIGHT] {
            for uplo in [plan::FILL_LOWER, plan::FILL_UPPER] {
                for &trans in transes {
                    for diag in [plan::DIAG_NON_UNIT, plan::DIAG_UNIT] {
                        let (m, n) = (41usize, 29usize);
                        let ord = if side == plan::SIDE_LEFT { m } else { n };
                        let (lda, ldb) = (ord + 2, m + 1);
                        let mut a: Vec<Cx> = rng
                            .vec(ty, lda * ord)
                            .iter()
                            .map(|v| round(ty, (v.0 * 0.2, v.1 * 0.2)))
                            .collect();
                        for d in 0..ord {
                            a[d + d * lda] = round(ty, plan::cadd(a[d + d * lda], (1.5, 0.0)));
                        }
                        let tri = |r: usize, c: usize| -> Cx {
                            let (sr, sc) = if trans == plan::OP_N { (r, c) } else { (c, r) };
                            if sr == sc && diag == plan::DIAG_UNIT {
                                return (1.0, 0.0);
                            }
                            let inside = if uplo == plan::FILL_LOWER {
                                sr >= sc
                            } else {
                                sr <= sc
                            };
                            if inside {
                                plan::op_at(&a, lda, trans, r, c)
                            } else {
                                (0.0, 0.0)
                            }
                        };
                        let b = rng.vec(ty, ldb * n);
                        let alpha = rng.cx(ty);
                        let mut want = b.clone();
                        for j in 0..n {
                            for i in 0..m {
                                let mut acc = (0.0, 0.0);
                                for l in 0..ord {
                                    acc = plan::cadd(
                                        acc,
                                        if side == plan::SIDE_LEFT {
                                            plan::cmul(tri(i, l), at(&b, ldb, l, j))
                                        } else {
                                            plan::cmul(at(&b, ldb, i, l), tri(l, j))
                                        },
                                    );
                                }
                                want[i + j * ldb] = plan::cmul(alpha, acc);
                            }
                        }
                        let al = enc(ty, &[alpha]);
                        let da = up(&enc(ty, &a));
                        // out of place into a separate C with the same layout
                        let (db, dc) = (up(&enc(ty, &b)), up(&enc(ty, &b)));
                        assert_eq!(
                            unsafe {
                                trmm(
                                    h,
                                    side,
                                    uplo,
                                    trans,
                                    diag,
                                    m as i32,
                                    n as i32,
                                    al.as_ptr().cast(),
                                    da as C,
                                    lda as i32,
                                    db as C,
                                    ldb as i32,
                                    dc as P,
                                    ldb as i32,
                                )
                            },
                            0
                        );
                        record(
                            format!("{t}trmm"),
                            max_err(&dec(ty, &down(dc, b.len() * es)), &want),
                            lim,
                        );
                        // in place, as MAGMA and cuSOLVER call it
                        assert_eq!(
                            unsafe {
                                trmm(
                                    h,
                                    side,
                                    uplo,
                                    trans,
                                    diag,
                                    m as i32,
                                    n as i32,
                                    al.as_ptr().cast(),
                                    da as C,
                                    lda as i32,
                                    db as C,
                                    ldb as i32,
                                    db as P,
                                    ldb as i32,
                                )
                            },
                            0
                        );
                        record(
                            format!("{t}trmm in place"),
                            max_err(&dec(ty, &down(db, b.len() * es)), &want),
                            lim,
                        );
                        if side == plan::SIDE_LEFT {
                            let x = rng.vec(ty, 2 * m);
                            let mut want = x.clone();
                            for i in 0..m {
                                let mut acc = (0.0, 0.0);
                                for l in 0..m {
                                    acc = plan::cadd(acc, plan::cmul(tri(i, l), x[2 * l]));
                                }
                                want[2 * i] = acc;
                            }
                            let dx = up(&enc(ty, &x));
                            assert_eq!(
                                unsafe {
                                    trmv(
                                        h, uplo, trans, diag, m as i32, da as C, lda as i32,
                                        dx as P, 2,
                                    )
                                },
                                0
                            );
                            let got = dec(ty, &down(dx, x.len() * es));
                            record(format!("{t}trmv"), max_err(&got, &want), lim);
                            // trsv undoes it
                            assert_eq!(
                                unsafe {
                                    trsv(
                                        h, uplo, trans, diag, m as i32, da as C, lda as i32,
                                        dx as P, 2,
                                    )
                                },
                                0
                            );
                            record(
                                format!("{t}trsv"),
                                max_err(&dec(ty, &down(dx, x.len() * es)), &x),
                                lim * 10.0,
                            );
                        }
                    }
                }
            }
        }
        // ---- ger, geam, dgmm, copy, swap, i?amax ----
        type Ger = unsafe extern "system" fn(P, i32, i32, C, C, i32, C, i32, P, i32) -> i32;
        let gers: Vec<(String, bool)> = if cx {
            vec![
                (format!("cublas{t}geru_v2"), false),
                (format!("cublas{t}gerc_v2"), true),
            ]
        } else {
            vec![(format!("cublas{t}ger_v2"), false)]
        };
        let (m, n) = (300usize, 77usize);
        let lda = m + 5;
        let a0 = rng.vec(ty, lda * n);
        for (name, cj) in gers {
            let ger: Ger = sym!(Ger, name);
            let (x, y, alpha) = (rng.vec(ty, 3 * m), rng.vec(ty, 2 * n), rng.cx(ty));
            let mut want = a0.clone();
            for j in 0..n {
                for i in 0..m {
                    let yv = y[2 * (n - 1 - j)];
                    want[i + j * lda] = plan::cadd(
                        a0[i + j * lda],
                        plan::cmul(alpha, plan::cmul(x[3 * i], if cj { conj(yv) } else { yv })),
                    );
                }
            }
            let (da, dx, dy) = (up(&enc(ty, &a0)), up(&enc(ty, &x)), up(&enc(ty, &y)));
            let al = enc(ty, &[alpha]);
            assert_eq!(
                unsafe {
                    ger(
                        h,
                        m as i32,
                        n as i32,
                        al.as_ptr().cast(),
                        dx as C,
                        3,
                        dy as C,
                        -2,
                        da as P,
                        lda as i32,
                    )
                },
                0
            );
            record(
                name.trim_start_matches("cublas").to_string(),
                max_err(&dec(ty, &down(da, a0.len() * es)), &want),
                lim,
            );
        }
        type Geam =
            unsafe extern "system" fn(P, i32, i32, i32, i32, C, C, i32, C, C, i32, P, i32) -> i32;
        let geam: Geam = sym!(Geam, format!("cublas{t}geam"));
        for (ta, tb) in [
            (plan::OP_N, plan::OP_N),
            (plan::OP_T, if cx { plan::OP_C } else { plan::OP_T }),
        ] {
            let ld = m.max(n) + 1;
            let (a, b) = (rng.vec(ty, ld * m.max(n)), rng.vec(ty, ld * m.max(n)));
            let (alpha, beta) = (rng.cx(ty), rng.cx(ty));
            let mut want = vec![(0.0, 0.0); lda * n];
            for j in 0..n {
                for i in 0..m {
                    want[i + j * lda] = plan::cadd(
                        plan::cmul(alpha, plan::op_at(&a, ld, ta, i, j)),
                        plan::cmul(beta, plan::op_at(&b, ld, tb, i, j)),
                    );
                }
            }
            let (da, db, dc) = (up(&enc(ty, &a)), up(&enc(ty, &b)), up(&enc(ty, &want)));
            let (al, be) = (enc(ty, &[alpha]), enc(ty, &[beta]));
            assert_eq!(
                unsafe {
                    geam(
                        h,
                        ta,
                        tb,
                        m as i32,
                        n as i32,
                        al.as_ptr().cast(),
                        da as C,
                        ld as i32,
                        be.as_ptr().cast(),
                        db as C,
                        ld as i32,
                        dc as P,
                        lda as i32,
                    )
                },
                0
            );
            record(
                format!("{t}geam"),
                max_err(&dec(ty, &down(dc, want.len() * es)), &want),
                lim,
            );
        }
        type Dgmm = unsafe extern "system" fn(P, i32, i32, i32, C, i32, C, i32, P, i32) -> i32;
        let dgmm: Dgmm = sym!(Dgmm, format!("cublas{t}dgmm"));
        for side in [plan::SIDE_LEFT, plan::SIDE_RIGHT] {
            let len = if side == plan::SIDE_LEFT { m } else { n };
            let x = rng.vec(ty, len);
            let mut want = a0.clone();
            for j in 0..n {
                for i in 0..m {
                    want[i + j * lda] = plan::cmul(
                        a0[i + j * lda],
                        x[if side == plan::SIDE_LEFT { i } else { j }],
                    );
                }
            }
            let (da, dx, dc) = (up(&enc(ty, &a0)), up(&enc(ty, &x)), up(&enc(ty, &a0)));
            assert_eq!(
                unsafe {
                    dgmm(
                        h, side, m as i32, n as i32, da as C, lda as i32, dx as C, 1, dc as P,
                        lda as i32,
                    )
                },
                0
            );
            record(
                format!("{t}dgmm"),
                max_err(&dec(ty, &down(dc, a0.len() * es)), &want),
                lim,
            );
        }
        type Copy = unsafe extern "system" fn(P, i32, C, i32, P, i32) -> i32;
        type Amax = unsafe extern "system" fn(P, i32, C, i32, *mut i32) -> i32;
        let copy: Copy = sym!(Copy, format!("cublas{t}copy_v2"));
        let swap: Copy = sym!(Copy, format!("cublas{t}swap_v2"));
        let nn = 5000usize;
        let mut v = rng.vec(ty, 2 * nn);
        for e in v.iter_mut().filter(|e| e.0 == 0.0 && e.1 == 0.0) {
            *e = (0.5, 0.0); // a single exact zero below
        }
        v[2 * 1234] = (4.0, 0.0);
        v[2 * 4000] = (-4.0, 0.0);
        v[2 * 777] = (0.0, 0.0);
        let (dv, dw) = (up(&enc(ty, &v)), up(&enc(ty, &vec![(0.0, 0.0); nn])));
        assert_eq!(unsafe { copy(h, nn as i32, dv as C, 2, dw as P, -1) }, 0);
        let want: Vec<Cx> = (0..nn).map(|i| v[2 * (nn - 1 - i)]).collect();
        record(
            format!("{t}copy"),
            max_err(&dec(ty, &down(dw, nn * es)), &want),
            0.0,
        );
        let li = ty.tag();
        for (mm, expect) in [("amax", 1235), ("amin", 778)] {
            let fa: Amax = sym!(Amax, format!("cublasI{li}{mm}_v2"));
            let mut r = 0i32;
            assert_eq!(unsafe { fa(h, nn as i32, dv as C, 2, &mut r) }, 0);
            assert_eq!(r, expect, "I{li}{mm}");
            record(format!("I{li}{mm}"), 0.0, 0.0);
        }
        assert_eq!(unsafe { swap(h, nn as i32, dv as P, 2, dw as P, 1) }, 0);
        record(
            format!("{t}swap"),
            max_err(
                &dec(ty, &down(dw, nn * es)),
                &(0..nn).map(|i| v[2 * i]).collect::<Vec<_>>(),
            ),
            0.0,
        );
    }
    // ---- getrfBatched / getrsBatched: solve op(A) X = B, with and without pivoting ----
    type Getrf = unsafe extern "system" fn(P, i32, C, i32, *mut i32, *mut i32, i32) -> i32;
    type Getrs = unsafe extern "system" fn(
        P,
        i32,
        i32,
        i32,
        C,
        i32,
        *const i32,
        C,
        i32,
        *mut i32,
        i32,
    ) -> i32;
    for ty in Ty::BLAS {
        let t = ty.tag().to_uppercase();
        let es = ty.size() as usize;
        let getrf: Getrf = sym!(Getrf, format!("cublas{t}getrfBatched"));
        let getrs: Getrs = sym!(Getrs, format!("cublas{t}getrsBatched"));
        let ops: &[i32] = if ty.complex() {
            &[plan::OP_N, plan::OP_T, plan::OP_C]
        } else {
            &[plan::OP_N, plan::OP_T]
        };
        for pivot in [true, false] {
            for &op in ops {
                let (n, nrhs, lda, ldb, batch) = (24usize, 5usize, 26usize, 25usize, 50usize);
                let mut a = rng.vec(ty, lda * n * batch);
                if !pivot {
                    for z in 0..batch {
                        for d in 0..n {
                            a[z * lda * n + d + d * lda] = round(
                                ty,
                                plan::cadd(a[z * lda * n + d + d * lda], (n as f64, 0.0)),
                            );
                        }
                    }
                }
                let b = rng.vec(ty, ldb * nrhs * batch);
                let (da, db) = (up(&enc(ty, &a)), up(&enc(ty, &b)));
                let tab = |base: u64, s: usize| {
                    up(&(0..batch)
                        .flat_map(|z| (base + (z * s * es) as u64).to_le_bytes())
                        .collect::<Vec<u8>>())
                };
                let (ta, tb) = (tab(da, lda * n), tab(db, ldb * nrhs));
                let (dp, di) = (up(&vec![0u8; n * batch * 4]), up(&vec![0xFFu8; batch * 4]));
                let pp: *mut i32 = if pivot { dp as *mut i32 } else { null_mut() };
                assert_eq!(
                    unsafe {
                        getrf(
                            h,
                            n as i32,
                            ta as C,
                            lda as i32,
                            pp,
                            di as *mut i32,
                            batch as i32,
                        )
                    },
                    0
                );
                let infos = down(di, batch * 4);
                assert!(infos.iter().all(|&x| x == 0), "{t}getrfBatched info");
                let mut info = 9;
                assert_eq!(
                    unsafe {
                        getrs(
                            h,
                            op,
                            n as i32,
                            nrhs as i32,
                            ta as C,
                            lda as i32,
                            pp,
                            tb as C,
                            ldb as i32,
                            &mut info,
                            batch as i32,
                        )
                    },
                    0
                );
                assert_eq!(info, 0);
                let x = dec(ty, &down(db, b.len() * es));
                let mut worst_b = 0.0f64;
                for z in 0..batch {
                    let (ab, bb) = (z * lda * n, z * ldb * nrhs);
                    let mut back = vec![(0.0, 0.0); n * nrhs];
                    let mut want = vec![(0.0, 0.0); n * nrhs];
                    for c in 0..nrhs {
                        for i in 0..n {
                            let mut acc = (0.0, 0.0);
                            for l in 0..n {
                                acc = plan::cadd(
                                    acc,
                                    plan::cmul(
                                        plan::op_at(&a[ab..], lda, op, i, l),
                                        x[bb + l + c * ldb],
                                    ),
                                );
                            }
                            back[i + c * n] = acc;
                            want[i + c * n] = b[bb + i + c * ldb];
                        }
                    }
                    worst_b = worst_b.max(max_err(&back, &want));
                }
                record(
                    format!(
                        "{t} getrf+getrsBatched{} (residual)",
                        if pivot { "" } else { " no pivot" }
                    ),
                    worst_b,
                    tol(ty) * 1000.0,
                );
            }
        }
        let mut info = 0;
        assert_eq!(
            unsafe {
                getrs(
                    h,
                    7,
                    2,
                    1,
                    null_mut(),
                    2,
                    std::ptr::null(),
                    null_mut(),
                    2,
                    &mut info,
                    1,
                )
            },
            7
        );
        assert_eq!(info, -2);
    }

    // ---- host <-> device copies ----
    type SetMat = unsafe extern "system" fn(i32, i32, i32, C, i32, P, i32) -> i32;
    type SetMatAsync = unsafe extern "system" fn(i32, i32, i32, C, i32, P, i32, Handle) -> i32;
    type SetVec = unsafe extern "system" fn(i32, i32, C, i32, P, i32) -> i32;
    let set_m: SetMat = sym!(SetMat, "cublasSetMatrix");
    let get_m: SetMat = sym!(SetMat, "cublasGetMatrix");
    let set_ma: SetMatAsync = sym!(SetMatAsync, "cublasSetMatrixAsync");
    let get_ma: SetMatAsync = sym!(SetMatAsync, "cublasGetMatrixAsync");
    let set_v: SetVec = sym!(SetVec, "cublasSetVector");
    let get_v: SetVec = sym!(SetVec, "cublasGetVector");
    let (rows, cols, es) = (37usize, 11usize, 16usize);
    let host: Vec<u8> = (0..(rows + 4) * cols * es)
        .map(|i| (i * 7 % 251) as u8)
        .collect();
    let dev = up(&vec![0xEEu8; (rows + 9) * cols * es]);
    assert_eq!(
        unsafe {
            set_m(
                rows as i32,
                cols as i32,
                es as i32,
                host.as_ptr().cast(),
                (rows + 4) as i32,
                dev as P,
                (rows + 9) as i32,
            )
        },
        0
    );
    let mut back = vec![0u8; rows * cols * es];
    assert_eq!(
        unsafe {
            get_m(
                rows as i32,
                cols as i32,
                es as i32,
                dev as C,
                (rows + 9) as i32,
                back.as_mut_ptr().cast(),
                rows as i32,
            )
        },
        0
    );
    for j in 0..cols {
        assert_eq!(
            &back[j * rows * es..(j + 1) * rows * es],
            &host[j * (rows + 4) * es..j * (rows + 4) * es + rows * es],
            "matrix column {j}"
        );
    }
    let raw = down(dev, (rows + 9) * cols * es);
    assert!(
        raw[rows * es..(rows + 9) * es].iter().all(|&b| b == 0xEE),
        "SetMatrix wrote the padding"
    );
    let mut back2 = vec![0u8; rows * cols * es];
    assert_eq!(
        unsafe {
            set_ma(
                rows as i32,
                cols as i32,
                es as i32,
                back.as_ptr().cast(),
                rows as i32,
                dev as P,
                rows as i32,
                null_mut(),
            )
        },
        0
    );
    assert_eq!(
        unsafe {
            get_ma(
                rows as i32,
                cols as i32,
                es as i32,
                dev as C,
                rows as i32,
                back2.as_mut_ptr().cast(),
                rows as i32,
                null_mut(),
            )
        },
        0
    );
    assert_eq!(back, back2);
    let xs: Vec<u8> = (0..300 * 8).map(|i| (i % 253) as u8).collect();
    let dv = up(&vec![0u8; 100 * 3 * 8]);
    assert_eq!(
        unsafe { set_v(100, 8, xs.as_ptr().cast(), 3, dv as P, 3) },
        0
    );
    let mut ys = vec![0u8; 100 * 8];
    assert_eq!(
        unsafe { get_v(100, 8, dv as C, 3, ys.as_mut_ptr().cast(), 1) },
        0
    );
    for i in 0..100 {
        assert_eq!(&ys[i * 8..i * 8 + 8], &xs[i * 24..i * 24 + 8]);
    }
    assert_eq!(unsafe { set_v(4, 8, xs.as_ptr().cast(), 0, dv as P, 1) }, 7);
    record("Set/Get Matrix/Vector(Async)".into(), 0.0, 0.0);

    println!("| routine | cases | worst error |\n|---|---|---|");
    for (k, (e, n)) in &worst {
        println!("| {k} | {n} | {e:.1e} |");
    }
    assert_eq!(unsafe { destroy(h) }, 0);
    for d in allocs.into_inner() {
        ok!(cuMemFree_v2(d));
    }
    ok!(cuCtxDestroy_v2(ctx));
}
