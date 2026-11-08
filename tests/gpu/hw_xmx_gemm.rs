//! Native XMX (DPAS) GEMM fast path in the cuBLAS shim, one dense hardware run.
//!
//! Run (needs the built `nvcuda.dll`, `cublas64_12.dll` and an Intel GPU with
//! `SPV_INTEL_subgroup_matrix_multiply_accumulate`, i.e. Xe2/Xe3):
//!
//! ```text
//! cargo build --release -p cuda-for-intel -p cublas_shim
//! cargo test --release --test gpu hw_xmx -- --ignored --nocapture --test-threads=1
//! ```
//!
//! Everything runs in one process and one context:
//! 1. `hw_xmx_probe`: forces `CFI_XMX=2` on a tiny GEMM and reports the probe
//!    outcome (which operand-mask spelling worked, or that XMX is unusable).
//! 2. `hw_xmx_gemm_table`: a table of shapes (m, n, k in {1, 7, 16, 17, 31, 32,
//!    64, ...}, all transposes, f16/bf16 inputs, f32/f16/bf16 output, padded
//!    leading dimensions, alpha/beta, strided batches, pointer tables with A
//!    broadcast) through `cublasGemmEx`, `cublasGemmStridedBatchedEx` and
//!    `cublasGemmBatchedEx`, each against a CPU reference, once with the existing
//!    PTX kernels (`CFI_XMX=0`) and once through XMX (`CFI_XMX=2`). The values are
//!    multiples of 1/8 so every product and partial sum is exact in f32: any
//!    deviation is a real bug, not rounding. The XMX launch counter must advance.
//! 3. `hw_xmx_perf`: GFLOP/s for 512^3 and 2048^3 (plus a decode-like skinny
//!    shape) with the PTX kernels and with XMX for several subgroup tiles.
use super::hw_common::*;
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::c_void;
use std::ptr::null_mut;
use std::time::Instant;

type Create = unsafe extern "system" fn(*mut *mut c_void) -> i32;
type GemmEx = unsafe extern "system" fn(
    *mut c_void, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, i32, *const c_void,
    i32, i32, *const c_void, *mut c_void, i32, i32, i32, i32,
) -> i32;
type Strided = unsafe extern "system" fn(
    *mut c_void, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, i32, i64,
    *const c_void, i32, i32, i64, *const c_void, *mut c_void, i32, i32, i64, i32, i32, i32,
) -> i32;
type Batched = unsafe extern "system" fn(
    *mut c_void, i32, i32, i32, i32, i32, *const c_void, *const *const c_void, i32, i32,
    *const *const c_void, i32, i32, *const c_void, *const *mut c_void, i32, i32, i32, i32, i32,
) -> i32;
type Count = unsafe extern "system" fn() -> u64;
type Status = unsafe extern "system" fn() -> u32;

const R_32F: i32 = 0;
const R_16F: i32 = 2;
const R_16BF: i32 = 14;
const COMPUTE_32F: i32 = 68;
const COMPUTE_16F: i32 = 64;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Dt {
    F16,
    Bf16,
    F32,
}

impl Dt {
    fn code(self) -> i32 {
        match self {
            Dt::F16 => R_16F,
            Dt::Bf16 => R_16BF,
            Dt::F32 => R_32F,
        }
    }
    fn size(self) -> usize {
        if self == Dt::F32 { 4 } else { 2 }
    }
}

fn bf16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7fc0;
    }
    let x = v.to_bits();
    (x.wrapping_add(0x7fff + ((x >> 16) & 1)) >> 16) as u16
}

fn enc(dt: Dt, v: f32, out: &mut Vec<u8>) {
    match dt {
        Dt::F16 => out.extend_from_slice(&f32_to_f16(v).to_le_bytes()),
        Dt::Bf16 => out.extend_from_slice(&bf16_bits(v).to_le_bytes()),
        Dt::F32 => out.extend_from_slice(&v.to_le_bytes()),
    }
}

fn dec(dt: Dt, b: &[u8], i: usize) -> f32 {
    match dt {
        Dt::F16 => f16_to_f32(u16::from_le_bytes([b[2 * i], b[2 * i + 1]])),
        Dt::Bf16 => f32::from_bits((u16::from_le_bytes([b[2 * i], b[2 * i + 1]]) as u32) << 16),
        Dt::F32 => f32::from_le_bytes(b[4 * i..4 * i + 4].try_into().unwrap()),
    }
}

/// Multiples of 1/8 in [-2, 2]: exact in f16 and bf16, and every GEMM sum below
/// stays exact in f32 (|sum| <= k * 4 < 2^13, resolution 2^-6).
fn val(rng: &mut Rng) -> f32 {
    rng.range(-16, 16) as f32 / 8.0
}

#[derive(Clone, Copy, Debug)]
enum Kind {
    Plain,
    Strided(usize),
    /// Strided batch whose A has stride 0 (one A for every batch entry).
    Bcast(usize),
    /// Pointer tables; A is shared by `ratio` consecutive batch entries (GQA broadcast).
    Ptrs { batch: usize, ratio: usize },
}

#[derive(Clone, Copy, Debug)]
struct Case {
    m: usize,
    n: usize,
    k: usize,
    ta: bool,
    tb: bool,
    inp: Dt,
    out: Dt,
    kind: Kind,
    pad: usize,
    alpha: f32,
    beta: f32,
    /// CUBLAS_COMPUTE_16F with f16 alpha/beta (ggml KQV); f16 in/out only.
    c16: bool,
}

struct Ctx {
    lib: libloading::Library,
    handle: *mut c_void,
    alloc: unsafe extern "system" fn(*mut u64, usize) -> i32,
    free: unsafe extern "system" fn(u64) -> i32,
    htod: unsafe extern "system" fn(u64, *const c_void, usize) -> i32,
    dtoh: unsafe extern "system" fn(*mut c_void, u64, usize) -> i32,
    sync: unsafe extern "system" fn() -> i32,
}

impl Ctx {
    fn up(&self, b: &[u8]) -> u64 {
        let mut p = 0u64;
        assert_eq!(unsafe { (self.alloc)(&mut p, b.len().max(8)) }, 0, "cuMemAlloc");
        if !b.is_empty() {
            assert_eq!(unsafe { (self.htod)(p, b.as_ptr().cast(), b.len()) }, 0, "HtoD");
        }
        p
    }
    fn down(&self, p: u64, len: usize) -> Vec<u8> {
        let mut v = vec![0u8; len];
        assert_eq!(unsafe { (self.dtoh)(v.as_mut_ptr().cast(), p, len) }, 0, "DtoH");
        v
    }
    fn xmx_launches(&self) -> u64 {
        let f: libloading::Symbol<Count> = unsafe { self.shim().get(b"cfiXmxLaunchCount\0") }.expect("cfiXmxLaunchCount export");
        unsafe { f() }
    }
    fn probe_status(&self) -> u32 {
        let f: libloading::Symbol<Status> = unsafe { self.shim().get(b"cfiXmxProbeStatus\0") }.expect("cfiXmxProbeStatus export");
        unsafe { f() }
    }
    fn shim(&self) -> &libloading::Library {
        &self.lib
    }
}

#[allow(non_snake_case)]
fn setup() -> Ctx {
    let dll = library();
    // The library must outlive the context; leak it (process-lifetime driver).
    let dll: &'static libloading::Library = Box::leak(Box::new(dll));
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    macro_rules! get {
        ($name:literal, $t:ty) => {
            *unsafe { dll.get::<$t>(concat!($name, "\0").as_bytes()) }.expect($name)
        };
    }
    let lib = unsafe {
        libloading::Library::new(
            std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("cublas64_12.dll"),
        )
    }
    .expect("cublas64_12.dll");
    let create: libloading::Symbol<Create> = unsafe { lib.get(b"cublasCreate_v2\0") }.unwrap();
    let mut handle = null_mut();
    assert_eq!(unsafe { create(&mut handle) }, 0, "cublasCreate_v2");
    Ctx {
        handle,
        alloc: get!("cuMemAlloc_v2", unsafe extern "system" fn(*mut u64, usize) -> i32),
        free: get!("cuMemFree_v2", unsafe extern "system" fn(u64) -> i32),
        htod: get!("cuMemcpyHtoD_v2", unsafe extern "system" fn(u64, *const c_void, usize) -> i32),
        dtoh: get!("cuMemcpyDtoH_v2", unsafe extern "system" fn(*mut c_void, u64, usize) -> i32),
        sync: get!("cuCtxSynchronize", unsafe extern "system" fn() -> i32),
        lib,
    }
}

fn set_mode(mode: &str) {
    unsafe { std::env::set_var("CFI_XMX", mode) };
}

fn p<T>(v: &T) -> *const c_void {
    std::ptr::from_ref(v).cast()
}

struct Data {
    a: Vec<u8>,
    b: Vec<u8>,
    c0: Vec<u8>,
    lda: usize,
    ldb: usize,
    ldc: usize,
    ea: usize,
    eb: usize,
    ec: usize,
    batch: usize,
    a_mats: usize,
}

fn make_data(c: &Case, rng: &mut Rng) -> Data {
    let (ra, ca) = if c.ta { (c.k, c.m) } else { (c.m, c.k) };
    let (rb, cb) = if c.tb { (c.n, c.k) } else { (c.k, c.n) };
    let (lda, ldb, ldc) = (ra + c.pad, rb + c.pad, c.m + c.pad);
    let (ea, eb, ec) = (lda * ca, ldb * cb, ldc * c.n);
    let (batch, a_mats) = match c.kind {
        Kind::Plain => (1, 1),
        Kind::Strided(b) => (b, b),
        Kind::Bcast(b) => (b, 1),
        Kind::Ptrs { batch, ratio } => (batch, batch.div_ceil(ratio)),
    };
    let (mut a, mut b, mut c0) = (Vec::new(), Vec::new(), Vec::new());
    for _ in 0..ea * a_mats {
        enc(c.inp, val(rng), &mut a);
    }
    for _ in 0..eb * batch {
        enc(c.inp, val(rng), &mut b);
    }
    for _ in 0..ec * batch {
        enc(c.out, val(rng), &mut c0);
    }
    Data { a, b, c0, lda, ldb, ldc, ea, eb, ec, batch, a_mats }
}

fn reference(c: &Case, d: &Data) -> Vec<u8> {
    let mut want = d.c0.clone();
    let ratio = match c.kind {
        Kind::Ptrs { ratio, .. } => ratio,
        Kind::Bcast(b) => b,
        _ => 1,
    };
    let _ = d.a_mats;
    for z in 0..d.batch {
        let ao = (z / ratio) * d.ea;
        for j in 0..c.n {
            for i in 0..c.m {
                let mut acc = 0f32;
                for l in 0..c.k {
                    let ai = if c.ta { l + i * d.lda } else { i + l * d.lda };
                    let bi = if c.tb { j + l * d.ldb } else { l + j * d.ldb };
                    acc += dec(c.inp, &d.a, ao + ai) * dec(c.inp, &d.b, z * d.eb + bi);
                }
                let ci = z * d.ec + i + j * d.ldc;
                let mut x = c.alpha * acc;
                if c.beta != 0.0 {
                    x += c.beta * dec(c.out, &d.c0, ci);
                }
                let mut one = Vec::new();
                enc(c.out, x, &mut one);
                want[ci * c.out.size()..(ci + 1) * c.out.size()].copy_from_slice(&one);
            }
        }
    }
    want
}

/// Run one case through the shim in the current `CFI_XMX` mode; returns C bytes.
fn run(ctx: &Ctx, c: &Case, d: &Data) -> Vec<u8> {
    let (da, db, dc) = (ctx.up(&d.a), ctx.up(&d.b), ctx.up(&d.c0));
    let (ta, tb) = (c.ta as i32, c.tb as i32);
    let (m, n, k) = (c.m as i32, c.n as i32, c.k as i32);
    let (alpha, beta) = (c.alpha, c.beta);
    let (al16, be16) = (f32_to_f16(alpha), f32_to_f16(beta));
    let ct = if c.c16 { COMPUTE_16F } else { COMPUTE_32F };
    let (ap, bp) = if c.c16 { (p(&al16), p(&be16)) } else { (p(&alpha), p(&beta)) };
    let status = match c.kind {
        Kind::Plain => {
            let f: libloading::Symbol<GemmEx> = unsafe { ctx.lib.get(b"cublasGemmEx\0") }.unwrap();
            unsafe {
                f(ctx.handle, ta, tb, m, n, k, ap, da as *const c_void, c.inp.code(), d.lda as i32,
                  db as *const c_void, c.inp.code(), d.ldb as i32, bp, dc as *mut c_void, c.out.code(),
                  d.ldc as i32, ct, -1)
            }
        }
        Kind::Strided(batch) => {
            let f: libloading::Symbol<Strided> = unsafe { ctx.lib.get(b"cublasGemmStridedBatchedEx\0") }.unwrap();
            unsafe {
                f(ctx.handle, ta, tb, m, n, k, ap, da as *const c_void, c.inp.code(), d.lda as i32,
                  d.ea as i64, db as *const c_void, c.inp.code(), d.ldb as i32, d.eb as i64, bp,
                  dc as *mut c_void, c.out.code(), d.ldc as i32, d.ec as i64, batch as i32, ct, -1)
            }
        }
        Kind::Bcast(batch) => {
            let f: libloading::Symbol<Strided> = unsafe { ctx.lib.get(b"cublasGemmStridedBatchedEx\0") }.unwrap();
            unsafe {
                f(ctx.handle, ta, tb, m, n, k, ap, da as *const c_void, c.inp.code(), d.lda as i32,
                  0, db as *const c_void, c.inp.code(), d.ldb as i32, d.eb as i64, bp,
                  dc as *mut c_void, c.out.code(), d.ldc as i32, d.ec as i64, batch as i32, ct, -1)
            }
        }
        Kind::Ptrs { batch, ratio } => {
            let table = |base: u64, per: usize, div: usize| -> u64 {
                let v: Vec<u8> = (0..batch).flat_map(|z| (base + ((z / div) * per) as u64).to_le_bytes()).collect();
                ctx.up(&v)
            };
            let ta_ = table(da, d.ea * c.inp.size(), ratio);
            let tb_ = table(db, d.eb * c.inp.size(), 1);
            let tc_ = table(dc, d.ec * c.out.size(), 1);
            let f: libloading::Symbol<Batched> = unsafe { ctx.lib.get(b"cublasGemmBatchedEx\0") }.unwrap();
            let s = unsafe {
                f(ctx.handle, ta, tb, m, n, k, ap, ta_ as *const *const c_void, c.inp.code(), d.lda as i32,
                  tb_ as *const *const c_void, c.inp.code(), d.ldb as i32, bp, tc_ as *const *mut c_void,
                  c.out.code(), d.ldc as i32, batch as i32, ct, -1)
            };
            for t in [ta_, tb_, tc_] {
                unsafe { (ctx.free)(t) };
            }
            s
        }
    };
    let sync = unsafe { (ctx.sync)() };
    assert_eq!((status, sync), (0, 0), "{c:?}: gemm status / sync");
    let out = ctx.down(dc, d.c0.len());
    for ptr in [da, db, dc] {
        unsafe { (ctx.free)(ptr) };
    }
    out
}

/// Compare; returns (max abs error, number of inexact entries). Panics beyond tolerance.
fn compare(c: &Case, d: &Data, got: &[u8], want: &[u8], who: &str) -> (f64, usize) {
    let sz = c.out.size();
    let (mut worst, mut inexact) = (0f64, 0usize);
    for z in 0..d.batch {
        for j in 0..c.n {
            for i in 0..d.ldc {
                let ci = z * d.ec + i + j * d.ldc;
                if i >= c.m {
                    assert_eq!(&got[ci * sz..(ci + 1) * sz], &d.c0[ci * sz..(ci + 1) * sz], "{who} {c:?}: padding row {i} clobbered (z={z} j={j})");
                    continue;
                }
                let (g, w) = (dec(c.out, got, ci) as f64, dec(c.out, want, ci) as f64);
                if got[ci * sz..(ci + 1) * sz] != want[ci * sz..(ci + 1) * sz] {
                    inexact += 1;
                }
                let tol = match c.out {
                    Dt::F32 => 1e-4 * (1.0 + w.abs()),
                    Dt::F16 => 2e-3 * w.abs() + 2e-3,
                    Dt::Bf16 => 1.6e-2 * w.abs() + 2e-2,
                };
                worst = worst.max((g - w).abs());
                assert!((g - w).abs() <= tol, "{who} {c:?}: z={z} i={i} j={j} got {g} want {w}");
            }
        }
    }
    (worst, inexact)
}

fn table() -> Vec<Case> {
    let mut rng = Rng(0x786d_7867_656d_6d31);
    let mut cases = Vec::new();
    let small = [1usize, 7, 16, 17, 31, 32, 64];
    // 1. Every (m, n, k) triple of the small sizes once, cycling transposes / dtypes /
    //    output types / padding so each combination is hit many times.
    let mut idx = 0usize;
    for &m in &small {
        for &n in &small {
            for &k in &small {
                let out = [Dt::F32, Dt::F16, Dt::Bf16][idx % 3];
                let inp = if out == Dt::Bf16 || idx % 2 == 1 { Dt::Bf16 } else { Dt::F16 };
                // cuBLAS pairs f16 inputs with f16/f32 and bf16 inputs with bf16/f32 only.
                let out = match (inp, out) {
                    (Dt::F16, Dt::Bf16) | (Dt::Bf16, Dt::F16) => Dt::F32,
                    _ => out,
                };
                cases.push(Case {
                    m, n, k,
                    ta: idx % 4 >= 2,
                    tb: (idx / 4) % 2 == 1,
                    inp, out,
                    kind: Kind::Plain,
                    pad: [0, 0, 1, 3][idx % 4],
                    alpha: if idx % 5 == 0 { 0.5 } else { 1.0 },
                    beta: if idx % 7 == 0 { 0.25 } else { 0.0 },
                    c16: false,
                });
                idx += 1;
            }
        }
    }
    // 2. All transposes x input types x output types on mid-size shapes (aligned fast paths: pad 0).
    for &(m, n, k) in &[(128usize, 128usize, 128usize), (512, 64, 512), (33, 257, 96), (100, 40, 130)] {
        for ta in [false, true] {
            for tb in [false, true] {
                for (inp, out) in [(Dt::F16, Dt::F32), (Dt::F16, Dt::F16), (Dt::Bf16, Dt::F32), (Dt::Bf16, Dt::Bf16)] {
                    for pad in [0usize, 2] {
                        cases.push(Case { m, n, k, ta, tb, inp, out, kind: Kind::Plain, pad, alpha: 1.0, beta: if pad == 2 { 0.5 } else { 0.0 }, c16: false });
                    }
                }
            }
        }
    }
    // 3. Larger exact checks (full CPU reference).
    for &(m, n, k) in &[(512usize, 512usize, 512usize), (1024, 256, 2048), (2048, 16, 256)] {
        for (ta, tb) in [(true, false), (false, false)] {
            cases.push(Case { m, n, k, ta, tb, inp: Dt::F16, out: Dt::F32, kind: Kind::Plain, pad: 0, alpha: 1.0, beta: 0.0, c16: false });
        }
    }
    // 4. Batched: strided and pointer tables (attention shapes, GQA broadcast of A).
    for &(m, n, k) in &[(64usize, 32usize, 128usize), (512, 512, 64), (33, 17, 40), (16, 8, 16)] {
        for (ta, tb) in [(true, false), (false, false), (true, true)] {
            for inp in [Dt::F16, Dt::Bf16] {
                for out in [Dt::F32, if inp == Dt::F16 { Dt::F16 } else { Dt::Bf16 }] {
                    cases.push(Case { m, n, k, ta, tb, inp, out, kind: Kind::Strided(6), pad: 1, alpha: 1.0, beta: 0.0, c16: false });
                    cases.push(Case { m, n, k, ta, tb, inp, out, kind: Kind::Ptrs { batch: 14, ratio: 7 }, pad: 0, alpha: 1.0, beta: 0.0, c16: false });
                }
            }
        }
    }
    cases.push(Case { m: 70, n: 50, k: 90, ta: true, tb: false, inp: Dt::F16, out: Dt::F32, kind: Kind::Ptrs { batch: 9, ratio: 3 }, pad: 2, alpha: 0.5, beta: 0.25, c16: false });
    let _ = &mut rng;
    cases
}

/// llama.cpp's per-layer attention GEMMs for Qwen2.5-0.5B (14 query heads, 2 kv heads: ratio 7)
/// and Llama-3.2-1B (32 / 8: ratio 4), head dim 64. QK: A = K^T view (op T), f16 in, f32 out;
/// KQV: A = V, f16 in/out with `CUBLAS_COMPUTE_16F`. Both are pointer-table batched.
fn ggml_cases() -> Vec<Case> {
    let mut v = Vec::new();
    for &(heads, ratio) in &[(14usize, 7usize), (32, 4)] {
        for &(n_kv, n_tok) in &[(512usize, 512usize), (512, 33), (256, 256), (1024, 128)] {
            v.push(Case { m: n_kv, n: n_tok, k: 64, ta: true, tb: false, inp: Dt::F16, out: Dt::F32,
                kind: Kind::Ptrs { batch: heads, ratio }, pad: 0, alpha: 1.0, beta: 0.0, c16: false });
            v.push(Case { m: 64, n: n_tok, k: n_kv, ta: false, tb: false, inp: Dt::F16, out: Dt::F16,
                kind: Kind::Ptrs { batch: heads, ratio }, pad: 0, alpha: 1.0, beta: 0.0, c16: true });
        }
        // Broadcast through stride 0 instead of a table (single kv head) and a strided KQV.
        v.push(Case { m: 64, n: 256, k: 256, ta: false, tb: false, inp: Dt::F16, out: Dt::F16,
            kind: Kind::Bcast(heads), pad: 0, alpha: 1.0, beta: 0.0, c16: true });
        v.push(Case { m: 64, n: 256, k: 256, ta: false, tb: false, inp: Dt::F16, out: Dt::F16,
            kind: Kind::Strided(heads), pad: 0, alpha: 1.0, beta: 0.0, c16: true });
    }
    v
}

/// Mirrors `xmx_host::qualify` in Auto mode (`CFI_XMX` unset or `1`).
fn expect_auto_xmx(c: &Case) -> bool {
    let batch = match c.kind {
        Kind::Plain => 1,
        Kind::Strided(b) | Kind::Bcast(b) => b,
        Kind::Ptrs { batch, .. } => batch,
    };
    let (m, n, k) = (c.m as u64, c.n as u64, c.k as u64);
    m >= 16 && n >= 8 && k >= 16 && m.min(n) >= 16 && m * n * k * batch as u64 >= 1 << 26
}

#[test]
#[ignore = "requires nvcuda.dll, cublas64_12.dll and an Intel GPU with SPV_INTEL_subgroup_matrix_multiply_accumulate"]
fn hw_xmx_ggml_attention() {
    let ctx = setup();
    let mut rng = Rng(0x6767_6d6c);
    let (mut engaged, mut skipped) = (0, 0);
    for c in ggml_cases() {
        let d = make_data(&c, &mut rng);
        let want = reference(&c, &d);
        if !c.c16 {
            set_mode("0");
            let ptx = run(&ctx, &c, &d);
            compare(&c, &d, &ptx, &want, "ptx");
        }
        // Default (Auto) mode: the shipped decision, no forcing.
        set_mode("1");
        let before = ctx.xmx_launches();
        let got = run(&ctx, &c, &d);
        let launched = ctx.xmx_launches() > before;
        // A 16F call below the Auto threshold runs on the PTX kernel, which rounds its accumulator
        // to f16 after every operation (~0.5-3% off the single-rounded reference): loose check.
        let (worst, inexact) = if c.c16 && !launched {
            let sz = 2;
            let mut worst = 0f64;
            for i in 0..got.len() / sz {
                let (g, w) = (dec(Dt::F16, &got, i) as f64, dec(Dt::F16, &want, i) as f64);
                worst = worst.max((g - w).abs());
                assert!((g - w).abs() <= 0.03 * w.abs() + 0.05, "{c:?}: PTX 16F entry {i}: got {g} want {w}");
            }
            (worst, usize::MAX)
        } else {
            compare(&c, &d, &got, &want, "xmx")
        };
        println!("GGML {c:?}: xmx launched={launched} max abs err {worst:.3e} inexact {inexact}");
        assert_eq!(launched, expect_auto_xmx(&c), "{c:?}: XMX engagement differs from the threshold rule");
        if launched { engaged += 1 } else { skipped += 1 }
        if c.c16 && launched {
            // f32 accumulators, one rounding: bit-exact against the reference (exact-valued data).
            assert_eq!(inexact, 0, "{c:?}: 16F result is not the single-rounded f32 sum");
            // Opting out restores the PTX 16F path (which rounds the accumulator per operation).
            unsafe { std::env::set_var("CFI_XMX_ACC16", "0") };
            let b2 = ctx.xmx_launches();
            // The PTX per-op-f16 result is not compared against the single-rounded reference.
            let _ = run(&ctx, &c, &d);
            assert_eq!(ctx.xmx_launches(), b2, "CFI_XMX_ACC16=0 must decline");
            unsafe { std::env::remove_var("CFI_XMX_ACC16") };
        }
    }
    println!("GGML summary: {engaged} calls on XMX, {skipped} below the threshold");
    assert!(engaged >= 8);
    set_mode("0");
}

#[test]
#[ignore = "requires nvcuda.dll, cublas64_12.dll and an Intel GPU with SPV_INTEL_subgroup_matrix_multiply_accumulate"]
fn hw_xmx_probe() {
    let ctx = setup();
    unsafe { std::env::set_var("CFI_XMX_TRACE", "1") };
    set_mode("2");
    let mut rng = Rng(7);
    for inp in [Dt::F16, Dt::Bf16] {
        let c = Case { m: 33, n: 24, k: 40, ta: true, tb: true, inp, out: Dt::F32, kind: Kind::Plain, pad: 0, alpha: 1.0, beta: 0.0, c16: false };
        let d = make_data(&c, &mut rng);
        let before = ctx.xmx_launches();
        let got = run(&ctx, &c, &d);
        let want = reference(&c, &d);
        let (worst, inexact) = compare(&c, &d, &got, &want, "xmx");
        let status = ctx.probe_status();
        println!(
            "PROBE {inp:?}: status bits {status:#06b} (per input: 0 unprobed, 1 packed flags ok, 2 bare flags ok, 3 FAILED), xmx launches +{}, max abs err {worst}, inexact {inexact}",
            ctx.xmx_launches() - before
        );
        assert!(ctx.xmx_launches() > before, "{inp:?}: the XMX path was not taken (probe status {status:#06b}); see CFI_XMX_TRACE output");
    }
}

#[test]
#[ignore = "requires nvcuda.dll, cublas64_12.dll and an Intel GPU with SPV_INTEL_subgroup_matrix_multiply_accumulate"]
fn hw_xmx_gemm_table() {
    let ctx = setup();
    let cases = table();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let (mut worst_ptx, mut worst_xmx, mut inexact_ptx, mut inexact_xmx) = (0f64, 0f64, 0usize, 0usize);
    let started = Instant::now();
    let l0 = ctx.xmx_launches();
    let mut failures = Vec::new();
    for (idx, c) in cases.iter().enumerate() {
        let d = make_data(c, &mut rng);
        let want = reference(c, &d);
        set_mode("0");
        let ptx = run(&ctx, c, &d);
        let (w, i) = compare(c, &d, &ptx, &want, "ptx");
        worst_ptx = worst_ptx.max(w);
        inexact_ptx += i;
        set_mode("2");
        let before = ctx.xmx_launches();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let x = run(&ctx, c, &d);
            assert!(ctx.xmx_launches() > before, "case {idx} {c:?}: XMX path not taken");
            compare(c, &d, &x, &want, "xmx")
        }));
        match result {
            Ok((w, i)) => {
                worst_xmx = worst_xmx.max(w);
                inexact_xmx += i;
            }
            Err(e) => {
                let msg = e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                failures.push(format!("#{idx}: {msg}"));
            }
        }
    }
    set_mode("0");
    println!(
        "TABLE {} cases in {:.1}s; xmx launches +{}; max abs err ptx {worst_ptx:.3e} xmx {worst_xmx:.3e}; inexact entries ptx {inexact_ptx} xmx {inexact_xmx}",
        cases.len(), started.elapsed().as_secs_f32(), ctx.xmx_launches() - l0
    );
    for f in failures.iter().take(25) {
        println!("FAIL {f}");
    }
    assert!(failures.is_empty(), "{} of {} cases failed", failures.len(), cases.len());
}

fn gflops(ctx: &Ctx, m: usize, n: usize, k: usize, ta: bool, tb: bool, reps: usize) -> f64 {
    let c = Case { m, n, k, ta, tb, inp: Dt::F16, out: Dt::F32, kind: Kind::Plain, pad: 0, alpha: 1.0, beta: 0.0, c16: false };
    let mut rng = Rng(99);
    let d = make_data(&c, &mut rng);
    let (da, db, dc) = (ctx.up(&d.a), ctx.up(&d.b), ctx.up(&d.c0));
    let f: libloading::Symbol<GemmEx> = unsafe { ctx.lib.get(b"cublasGemmEx\0") }.unwrap();
    let (alpha, beta) = (1.0f32, 0.0f32);
    let call = || unsafe {
        f(ctx.handle, ta as i32, tb as i32, m as i32, n as i32, k as i32, p(&alpha), da as *const c_void, R_16F,
          d.lda as i32, db as *const c_void, R_16F, d.ldb as i32, p(&beta), dc as *mut c_void, R_32F, d.ldc as i32,
          COMPUTE_32F, -1)
    };
    unsafe { std::env::set_var("CFI_CUBLAS_ASYNC", "1") };
    for _ in 0..3 {
        assert_eq!(call(), 0);
    }
    assert_eq!(unsafe { (ctx.sync)() }, 0);
    let t = Instant::now();
    for _ in 0..reps {
        assert_eq!(call(), 0);
    }
    assert_eq!(unsafe { (ctx.sync)() }, 0);
    let secs = t.elapsed().as_secs_f64() / reps as f64;
    unsafe { std::env::remove_var("CFI_CUBLAS_ASYNC") };
    for ptr in [da, db, dc] {
        unsafe { (ctx.free)(ptr) };
    }
    2.0 * m as f64 * n as f64 * k as f64 / secs / 1e9
}

#[test]
#[ignore = "requires nvcuda.dll, cublas64_12.dll and an Intel GPU with SPV_INTEL_subgroup_matrix_multiply_accumulate"]
fn hw_xmx_perf() {
    let ctx = setup();
    for &(m, n, k, reps) in &[(512usize, 512usize, 512usize, 50usize), (2048, 2048, 2048, 5), (4096, 32, 4096, 20), (4096, 512, 4096, 5)] {
        for (ta, tb) in [(true, false), (false, false), (true, true)] {
            set_mode("0");
            let base = gflops(&ctx, m, n, k, ta, tb, reps);
            let mut line = format!("PERF {m}x{n}x{k} ta={} tb={} f16->f32: ptx {base:8.1} GFLOP/s", ta as u8, tb as u8);
            set_mode("2");
            for tile in ["2x4", "4x2", "2x2", "4x4", "1x4", "2x8"] {
                unsafe { std::env::set_var("CFI_XMX_TILE", tile) };
                let g = gflops(&ctx, m, n, k, ta, tb, reps);
                line.push_str(&format!(" | xmx {tile} {g:8.1} ({:.2}x)", g / base));
            }
            unsafe { std::env::remove_var("CFI_XMX_TILE") };
            println!("{line}");
        }
    }
    set_mode("0");
    assert!(ctx.xmx_launches() > 0, "XMX never launched");
}
