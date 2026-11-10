//! f64 / complex cuBLAS kernels (`crates/cublas_shim/src/kernels`): translation, and the
//! launch lists `plan` builds for gemm (N/T/C/CONJG, padding, alpha/beta, strided and
//! pointer-table batches, device scalars), gemv, dot/dotc, nrm2, axpy, scal and trsm, run in
//! the SPIR-V simulator against a host f64 complex reference.
use super::cublas64_src::plan::{self, Batch, Cx, Scalar};
use super::cublas64_src::ptxgen::{self as pg, Kernel, Ty};
use super::sim::{Gpu, Launch, Sim};
use super::*;
use std::collections::HashMap;

#[test]
fn cublas64_kernels_translate() {
    let entries = pg::all_entries();
    assert_eq!(entries.len(), 58);
    for (name, src, n) in entries {
        let t = translate(&src).unwrap_or_else(|e| panic!("{name}: {e}\n{src}"));
        assert_eq!(t.kernels[0].name, name);
        assert_eq!(t.kernels[0].params.len(), n, "{name}");
    }
    for ty in Ty::ALL {
        assert_eq!(
            translate(&pg::module(ty)).unwrap().kernels.len(),
            pg::kernels(ty).len()
        );
    }
}

#[derive(Default)]
struct Sims(HashMap<(Kernel, Ty), Sim>);

impl Sims {
    fn run(&mut self, gpu: &mut Gpu, launches: &[plan::Launch]) {
        for l in launches {
            let name = pg::entry_name(l.kernel, l.ty);
            let sim = self.0.entry((l.kernel, l.ty)).or_insert_with(|| {
                let t = translate(&pg::single(l.kernel, l.ty)).unwrap();
                Sim::new(&rspirv::dr::load_words(&t.words).unwrap())
            });
            sim.run(gpu, &Launch::new(l.grid, l.block, l.args.clone()))
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(
                gpu.violations.borrow().is_empty(),
                "{name}: {:?}",
                gpu.violations.borrow()
            );
        }
    }
}

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
        let re = self.val();
        let im = if ty.complex() { self.val() } else { 0.0 };
        round(ty, (re, im))
    }
    fn vec(&mut self, ty: Ty, n: usize) -> Vec<Cx> {
        (0..n).map(|_| self.cx(ty)).collect()
    }
}

fn round(ty: Ty, v: Cx) -> Cx {
    if ty.double() {
        v
    } else {
        (f64::from(v.0 as f32), f64::from(v.1 as f32))
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
                Ty::H => out.extend(f32_to_f16(c as f32).to_le_bytes()),
                Ty::B => out.extend(
                    (((c as f32).to_bits() + 0x7fff + (((c as f32).to_bits() >> 16) & 1)) >> 16)
                        .to_le_bytes()[..2]
                        .to_vec(),
                ),
            }
        }
    }
    out
}

/// f32 to f16, round to nearest even (normal and subnormal results; no overflow here).
fn f32_to_f16(v: f32) -> u16 {
    if v == 0.0 {
        return if v.is_sign_negative() { 0x8000 } else { 0 };
    }
    let b = v.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if exp <= 0 {
        let m = (man | 0x80_0000) >> (14 - exp);
        return sign | m as u16;
    }
    let mut h = (sign as u32) | ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    h as u16
}

fn dec(ty: Ty, b: &[u8]) -> Vec<Cx> {
    let c = ty.comp() as usize;
    let comp = |o: usize| -> f64 {
        match ty {
            Ty::H => f64::from(sim::f16_to_f32(u32::from(u16::from_le_bytes([
                b[o],
                b[o + 1],
            ])))),
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
            if (g.0.is_nan() && w.0.is_nan()) || (g.1.is_nan() && w.1.is_nan()) {
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
    if ty.double() { 1e-12 } else { 2e-5 }
}

fn scalar_of(ty: Ty, v: Cx) -> Scalar {
    Scalar::host(ty, v.0, v.1)
}

const OPS: [i32; 4] = [plan::OP_N, plan::OP_T, plan::OP_C, plan::OP_CONJG];

#[test]
fn sim_gemm_ops_padding_scalars_and_batches() {
    let mut sims = Sims::default();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut cases = 0;
    for ty in Ty::BLAS {
        let ops: &[i32] = if ty.complex() { &OPS } else { &OPS[..2] };
        for (ci, &(m, n, k)) in [
            (17usize, 5usize, 19usize),
            (1, 1, 1),
            (3, 18, 0),
            (16, 16, 33),
        ]
        .iter()
        .enumerate()
        {
            for &ta in ops {
                for &tb in ops {
                    if ci > 0
                        && (ta, tb) != (plan::OP_N, plan::OP_N)
                        && (ta, tb) != (plan::OP_C, plan::OP_T)
                    {
                        continue;
                    }
                    let (alpha, beta) = match cases % 4 {
                        0 => ((1.0, 0.0), (0.0, 0.0)),
                        1 => ((0.75, -0.5), (-0.5, 0.25)),
                        2 => ((0.0, 0.0), (1.5, 0.0)),
                        _ => ((-1.25, 0.5), (1.0, 0.0)),
                    };
                    let (alpha, beta) = (
                        round(ty, (alpha.0, if ty.complex() { alpha.1 } else { 0.0 })),
                        round(ty, (beta.0, if ty.complex() { beta.1 } else { 0.0 })),
                    );
                    let batch = if ci == 0 { 3 } else { 1 };
                    let (ra, ca_) = if ta == plan::OP_T || ta == plan::OP_C {
                        (k, m)
                    } else {
                        (m, k)
                    };
                    let (rb, cb_) = if tb == plan::OP_T || tb == plan::OP_C {
                        (n, k)
                    } else {
                        (k, n)
                    };
                    let (lda, ldb, ldc) = (ra.max(1) + 2, rb.max(1) + 1, m + 3);
                    let (sa, sb, sc) = (lda * ca_.max(1) + 5, ldb * cb_.max(1) + 2, ldc * n + 1);
                    let mut a = rng.vec(ty, sa * batch);
                    let b = rng.vec(ty, sb * batch);
                    let mut c = rng.vec(ty, sc * batch);
                    if beta == (0.0, 0.0) {
                        c[0] = (f64::NAN, f64::NAN); // beta 0 must not read C
                    }
                    if alpha == (0.0, 0.0) && !a.is_empty() {
                        a[0] = (f64::NAN, 0.0); // alpha 0 must not multiply A
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
                    for tables in [false, true] {
                        let mut gpu = Gpu::default();
                        let (da, db, dc) = (
                            gpu.upload(&enc(ty, &a)),
                            gpu.upload(&enc(ty, &b)),
                            gpu.upload(&enc(ty, &c)),
                        );
                        let layout = if tables {
                            let tab = |base: u64, s: usize, gpu: &mut Gpu| {
                                let ptrs: Vec<u8> = (0..batch)
                                    .flat_map(|z| {
                                        (base + (z * s) as u64 * u64::from(ty.size())).to_le_bytes()
                                    })
                                    .collect();
                                gpu.upload(&ptrs)
                            };
                            Batch::Tables {
                                a: tab(da, sa, &mut gpu),
                                b: tab(db, sb, &mut gpu),
                                c: tab(dc, sc, &mut gpu),
                            }
                        } else {
                            Batch::Strided {
                                a: da,
                                b: db,
                                c: dc,
                                sa: sa as i64,
                                sb: sb as i64,
                                sc: sc as i64,
                            }
                        };
                        // device-resident scalars on every other case
                        let (al, be) = if cases % 2 == 1 {
                            (
                                Scalar::device(gpu.upload(&enc(ty, &[alpha]))),
                                Scalar::device(gpu.upload(&enc(ty, &[beta]))),
                            )
                        } else {
                            (scalar_of(ty, alpha), scalar_of(ty, beta))
                        };
                        let launches = plan::gemm(
                            ty,
                            ta,
                            tb,
                            m as i32,
                            n as i32,
                            k as i32,
                            lda as i32,
                            ldb as i32,
                            ldc as i32,
                            al,
                            be,
                            batch as i32,
                            layout,
                        )
                        .unwrap();
                        sims.run(&mut gpu, &launches);
                        let got = dec(ty, &gpu.download(dc, c.len() * ty.size() as usize));
                        // padding (rows i >= m and the gaps between batches) must be untouched
                        for (idx, (g, w)) in got.iter().zip(&want).enumerate() {
                            let (z, off) = (idx / sc, idx % sc);
                            if z >= batch || off >= ldc * n || off % ldc >= m {
                                assert!(
                                    g == w || (g.0.is_nan() && w.0.is_nan()),
                                    "{ty:?} padding {idx} changed"
                                );
                            }
                        }
                        let err = max_err(&got, &want);
                        assert!(
                            err < tol(ty),
                            "{ty:?} ta={ta} tb={tb} {m}x{n}x{k} tables={tables} err {err:e}"
                        );
                        cases += 1;
                    }
                }
            }
        }
    }
    assert!(cases >= 100, "{cases}");
}

#[test]
fn sim_gemv_with_increments() {
    let mut sims = Sims::default();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for ty in Ty::BLAS {
        for trans in [plan::OP_N, plan::OP_T, plan::OP_C] {
            for (m, n, incx, incy, dev) in [
                (13i32, 7i32, 1i32, 1i32, false),
                (9, 20, -2, 3, true),
                (300, 2, 2, -1, false),
            ] {
                let (alpha, beta) = (rng.cx(ty), if dev { (0.0, 0.0) } else { rng.cx(ty) });
                let lda = m as usize + 3;
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
                // reference with BLAS negative-increment indexing
                let at = |len: usize, inc: i32, i: usize| {
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
                            plan::cmul(plan::op_at(&a, lda, trans, r, l), x[at(xl, incx, l)]),
                        );
                    }
                    let yi = at(yl, incy, r);
                    let mut v = plan::cmul(alpha, acc);
                    if beta != (0.0, 0.0) {
                        v = plan::cadd(v, plan::cmul(beta, y[yi]));
                    }
                    want[yi] = v;
                }
                let mut gpu = Gpu::default();
                let (da, dx, dy) = (
                    gpu.upload(&enc(ty, &a)),
                    gpu.upload(&enc(ty, &x)),
                    gpu.upload(&enc(ty, &y)),
                );
                let (al, be) = if dev {
                    (
                        Scalar::device(gpu.upload(&enc(ty, &[alpha]))),
                        Scalar::device(gpu.upload(&enc(ty, &[beta]))),
                    )
                } else {
                    (scalar_of(ty, alpha), scalar_of(ty, beta))
                };
                let l = plan::gemv(ty, trans, m, n, da, lda as i32, dx, incx, dy, incy, al, be)
                    .unwrap();
                sims.run(&mut gpu, &l);
                let got = dec(ty, &gpu.download(dy, y.len() * ty.size() as usize));
                let err = max_err(&got, &want);
                assert!(
                    err < tol(ty),
                    "{ty:?} gemv trans={trans} {m}x{n} inc {incx},{incy}: {err:e}"
                );
            }
        }
    }
}

#[test]
fn sim_level1_dot_nrm2_axpy_scal() {
    let mut sims = Sims::default();
    let mut rng = Rng(0x0dd_ba11_cafe);
    for ty in Ty::ALL {
        for (n, incx, incy) in [
            (0i32, 1i32, 1i32),
            (1, 1, 1),
            (700, 1, 1),
            (257, -2, 3),
            (33, 3, -1),
        ] {
            let (lx, ly) = (
                (n.max(1) as usize) * incx.unsigned_abs() as usize,
                (n.max(1) as usize) * incy.unsigned_abs() as usize,
            );
            let (x, y) = (rng.vec(ty, lx), rng.vec(ty, ly));
            let at = |inc: i32, i: usize| {
                if inc > 0 {
                    i * inc as usize
                } else {
                    (n as usize - 1 - i) * (-inc) as usize
                }
            };
            for conj in [false, true] {
                if conj && !ty.complex() {
                    continue;
                }
                let mut want = (0.0, 0.0);
                for i in 0..n.max(0) as usize {
                    let xv = x[at(incx, i)];
                    let xv = if conj { (xv.0, -xv.1) } else { xv };
                    want = plan::cadd(want, plan::cmul(xv, y[at(incy, i)]));
                }
                let mut gpu = Gpu::default();
                let (dx, dy, out) = (
                    gpu.upload(&enc(ty, &x)),
                    gpu.upload(&enc(ty, &y)),
                    gpu.upload(&enc(ty, &[(9.0, 9.0)])),
                );
                sims.run(&mut gpu, &[plan::dot(ty, n, dx, incx, dy, incy, out, conj)]);
                let got = dec(ty, &gpu.download(out, ty.size() as usize));
                let t = match ty {
                    Ty::H => 2e-3 * f64::from(n.max(1)),
                    Ty::B => 1e-2 * f64::from(n.max(1)),
                    _ => tol(ty) * f64::from(n.max(1)).sqrt(),
                };
                assert!(
                    max_err(&got, &[want]) < t,
                    "{ty:?} dot conj={conj} n={n}: {got:?} vs {want:?}"
                );
            }
            if !Ty::BLAS.contains(&ty) {
                continue;
            }
            // nrm2, also with huge and tiny magnitudes
            for scale in [
                1.0,
                if ty.double() { 1e200 } else { 1e30 },
                if ty.double() { 1e-200 } else { 1e-30 },
            ] {
                let xs: Vec<Cx> = x
                    .iter()
                    .map(|v| round(ty, (v.0 * scale, v.1 * scale)))
                    .collect();
                let pos = |i: usize| i * incx.max(1) as usize;
                let want = if incx <= 0 {
                    0.0
                } else {
                    (0..n.max(0) as usize)
                        .map(|i| xs[pos(i)])
                        .map(|v| (v.0 / scale).powi(2) + (v.1 / scale).powi(2))
                        .sum::<f64>()
                        .sqrt()
                        * scale
                };
                let mut gpu = Gpu::default();
                let (dx, out) = (gpu.upload(&enc(ty, &xs)), gpu.alloc(16));
                sims.run(&mut gpu, &[plan::nrm2(ty, n, dx, incx, out)]);
                let rt = if ty.double() { Ty::D } else { Ty::S };
                let got = dec(rt, &gpu.download(out, ty.comp() as usize))[0].0;
                assert!(
                    (got - want).abs() <= tol(ty) * 4.0 * want.max(f64::MIN_POSITIVE),
                    "{ty:?} nrm2 n={n} inc={incx} scale={scale:e}: {got:e} vs {want:e}"
                );
            }
            // axpy: y += alpha x (alpha from device memory once)
            for dev in [false, true] {
                let alpha = rng.cx(ty);
                let mut want = y.clone();
                for i in 0..n.max(0) as usize {
                    let j = at(incy, i);
                    want[j] = plan::cadd(want[j], plan::cmul(alpha, x[at(incx, i)]));
                }
                let mut gpu = Gpu::default();
                let (dx, dy) = (gpu.upload(&enc(ty, &x)), gpu.upload(&enc(ty, &y)));
                let al = if dev {
                    Scalar::device(gpu.upload(&enc(ty, &[alpha])))
                } else {
                    scalar_of(ty, alpha)
                };
                sims.run(
                    &mut gpu,
                    &plan::axpy(ty, n, al, dx, incx, dy, incy).unwrap(),
                );
                let got = dec(ty, &gpu.download(dy, y.len() * ty.size() as usize));
                assert!(max_err(&got, &want) < tol(ty), "{ty:?} axpy n={n}");
            }
            // scal (complex alpha, and a real alpha from device memory for Csscal/Zdscal)
            for real in [false, true] {
                if real && !ty.complex() {
                    continue;
                }
                let alpha = if real { (rng.val(), 0.0) } else { rng.cx(ty) };
                let alpha = round(ty, alpha);
                let mut want = x.clone();
                if incx > 0 {
                    for i in 0..n.max(0) as usize {
                        want[i * incx as usize] = plan::cmul(alpha, x[i * incx as usize]);
                    }
                }
                let mut gpu = Gpu::default();
                let dx = gpu.upload(&enc(ty, &x));
                let rty = if ty.double() { Ty::D } else { Ty::S };
                let al = if real {
                    Scalar::device(gpu.upload(&enc(rty, &[alpha])))
                } else {
                    scalar_of(ty, alpha)
                };
                sims.run(&mut gpu, &plan::scal(ty, n, al, dx, incx, real).unwrap());
                let got = dec(ty, &gpu.download(dx, x.len() * ty.size() as usize));
                assert!(
                    max_err(&got, &want) < tol(ty),
                    "{ty:?} scal real={real} n={n}"
                );
            }
        }
    }
}

#[test]
fn sim_trsm_every_side_uplo_trans_diag() {
    let mut sims = Sims::default();
    let mut rng = Rng(0x7357_7357_1);
    let mut cases = 0;
    for ty in Ty::BLAS {
        let transes: &[i32] = if ty.complex() {
            &[plan::OP_N, plan::OP_T, plan::OP_C]
        } else {
            &[plan::OP_N, plan::OP_T]
        };
        for side in [plan::SIDE_LEFT, plan::SIDE_RIGHT] {
            for uplo in [plan::FILL_LOWER, plan::FILL_UPPER] {
                for &trans in transes {
                    for diag in [plan::DIAG_NON_UNIT, plan::DIAG_UNIT] {
                        let (m, n) = (7usize, 5usize);
                        let order = if side == plan::SIDE_LEFT { m } else { n };
                        let (lda, ldb) = (order + 2, m + 1);
                        let batch = 2;
                        let alpha = rng.cx(ty);
                        let mut a = rng.vec(ty, lda * order * batch);
                        for z in 0..batch {
                            for d in 0..order {
                                let v = &mut a[z * lda * order + d + d * lda];
                                *v = round(ty, (v.0 + 3.0, v.1)); // well conditioned
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
                        let p = plan::tri(
                            ty, side, uplo, trans, diag, m as i32, n as i32, lda as i32, ldb as i32,
                        )
                        .unwrap();
                        for tables in [false, true] {
                            let mut gpu = Gpu::default();
                            let (da, db) = (gpu.upload(&enc(ty, &a)), gpu.upload(&enc(ty, &b)));
                            let launches = if tables {
                                let tab = |base: u64, s: usize, gpu: &mut Gpu| {
                                    gpu.upload(
                                        &(0..batch)
                                            .flat_map(|z| {
                                                (base + (z * s) as u64 * u64::from(ty.size()))
                                                    .to_le_bytes()
                                            })
                                            .collect::<Vec<u8>>(),
                                    )
                                };
                                let (ta, tb) =
                                    (tab(da, lda * order, &mut gpu), tab(db, ldb * n, &mut gpu));
                                plan::trsm(ty, &p, scalar_of(ty, alpha), ta, tb, batch as i32, true)
                                    .unwrap()
                            } else {
                                // the non-batched form, once per matrix
                                let mut v = Vec::new();
                                for z in 0..batch {
                                    let off = |base: u64, s: usize| {
                                        base + (z * s) as u64 * u64::from(ty.size())
                                    };
                                    v.extend(
                                        plan::trsm(
                                            ty,
                                            &p,
                                            scalar_of(ty, alpha),
                                            off(da, lda * order),
                                            off(db, ldb * n),
                                            1,
                                            false,
                                        )
                                        .unwrap(),
                                    );
                                }
                                v
                            };
                            sims.run(&mut gpu, &launches);
                            let got = dec(ty, &gpu.download(db, b.len() * ty.size() as usize));
                            let err = max_err(&got, &want);
                            assert!(
                                err < tol(ty) * 10.0,
                                "{ty:?} trsm side={side} uplo={uplo} trans={trans} diag={diag} tables={tables}: {err:e}"
                            );
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 2 * (2 * 2 * 2 * 2) * 2 + 2 * (2 * 2 * 3 * 2) * 2);
}

/// Column-major `rows x cols` matrix with leading dimension `ld`.
fn at(m: &[Cx], ld: usize, r: usize, c: usize) -> Cx {
    m[r + c * ld]
}

fn conj(v: Cx) -> Cx {
    (v.0, -v.1)
}

/// op(X)(r, c) for op N/T/C of a stored column-major matrix.
fn opx(x: &[Cx], ld: usize, op: i32, r: usize, c: usize) -> Cx {
    plan::op_at(x, ld, op, r, c)
}

#[test]
fn sim_rank_k_symm_trmm_ger_and_friends() {
    let mut sims = Sims::default();
    let mut rng = Rng(0xfeed_beef_1234);
    let one = (1.0, 0.0);
    for ty in Ty::BLAS {
        let cx = ty.complex();
        let es = ty.size() as usize;
        let mut gpu = Gpu::default();
        let mut run = |gpu: &mut Gpu, l: Vec<plan::Launch>| sims.run(gpu, &l);
        // ---- syrk / herk / syrkx / syr2k / her2k ----
        for herm in [false, true] {
            if herm && !cx {
                continue;
            }
            let transes: &[i32] = if herm {
                &[plan::OP_N, plan::OP_C]
            } else {
                &[plan::OP_N, plan::OP_T]
            };
            for &trans in transes {
                for uplo in [plan::FILL_LOWER, plan::FILL_UPPER] {
                    let (n, k) = (7usize, 5usize);
                    let (ra, ca) = if trans == plan::OP_N { (n, k) } else { (k, n) };
                    let (lda, ldc) = (ra + 2, n + 1);
                    let a = rng.vec(ty, lda * ca);
                    let b = rng.vec(ty, lda * ca);
                    let c0 = rng.vec(ty, ldc * n);
                    let alpha = if herm { (0.75, 0.0) } else { rng.cx(ty) };
                    let alpha2c = rng.cx(ty);
                    let beta = if herm { (-0.5, 0.0) } else { rng.cx(ty) };
                    let opT = if herm { plan::OP_C } else { plan::OP_T };
                    // op(X)(i,l), and the second factor op(Y)^T(^H)(l,j) = op'(op(Y)(j,l))
                    let f = |x: &[Cx], i: usize, l: usize| opx(x, lda, trans, i, l);
                    let g = |y: &[Cx], l: usize, j: usize| {
                        let v = opx(y, lda, trans, j, l);
                        if opT == plan::OP_C { conj(v) } else { v }
                    };
                    let tri_ref = |pairs: &[(&[Cx], &[Cx], Cx)], beta: Cx| -> Vec<Cx> {
                        let mut w = c0.clone();
                        for j in 0..n {
                            for i in 0..n {
                                if (uplo == plan::FILL_LOWER && i < j)
                                    || (uplo == plan::FILL_UPPER && i > j)
                                {
                                    continue;
                                }
                                let mut v = plan::cmul(beta, c0[i + j * ldc]);
                                for &(x, y, al) in pairs {
                                    let mut acc = (0.0, 0.0);
                                    for l in 0..k {
                                        acc = plan::cadd(acc, plan::cmul(f(x, i, l), g(y, l, j)));
                                    }
                                    v = plan::cadd(v, plan::cmul(al, acc));
                                }
                                if herm && i == j {
                                    v.1 = 0.0;
                                }
                                w[i + j * ldc] = v;
                            }
                        }
                        w
                    };
                    let (da, db) = (gpu.upload(&enc(ty, &a)), gpu.upload(&enc(ty, &b)));
                    let sc = |v: Cx| scalar_of(ty, v);
                    // syrk / herk
                    let dc = gpu.upload(&enc(ty, &c0));
                    run(
                        &mut gpu,
                        plan::rank_k(
                            ty,
                            uplo,
                            trans,
                            n as i32,
                            k as i32,
                            da,
                            lda as i32,
                            da,
                            lda as i32,
                            dc,
                            ldc as i32,
                            sc(alpha),
                            sc(beta),
                            herm,
                        )
                        .unwrap(),
                    );
                    let want = tri_ref(&[(&a, &a, alpha)], beta);
                    let err = max_err(&dec(ty, &gpu.download(dc, c0.len() * es)), &want);
                    assert!(
                        err < tol(ty),
                        "{ty:?} rank-k herm={herm} trans={trans} uplo={uplo}: {err:e}"
                    );
                    // syrkx / herkx (complex alpha)
                    let alx = if herm { alpha2c } else { alpha };
                    let dc = gpu.upload(&enc(ty, &c0));
                    run(
                        &mut gpu,
                        plan::rank_k(
                            ty,
                            uplo,
                            trans,
                            n as i32,
                            k as i32,
                            da,
                            lda as i32,
                            db,
                            lda as i32,
                            dc,
                            ldc as i32,
                            sc(alx),
                            sc(beta),
                            herm,
                        )
                        .unwrap(),
                    );
                    let want = tri_ref(&[(&a, &b, alx)], beta);
                    let err = max_err(&dec(ty, &gpu.download(dc, c0.len() * es)), &want);
                    assert!(
                        err < tol(ty),
                        "{ty:?} rank-kx herm={herm} trans={trans} uplo={uplo}: {err:e}"
                    );
                    // syr2k / her2k
                    let al2 = if herm { conj(alpha2c) } else { alpha2c };
                    let dc = gpu.upload(&enc(ty, &c0));
                    run(
                        &mut gpu,
                        plan::rank_2k(
                            ty,
                            uplo,
                            trans,
                            n as i32,
                            k as i32,
                            da,
                            lda as i32,
                            db,
                            lda as i32,
                            dc,
                            ldc as i32,
                            sc(alpha2c),
                            sc(al2),
                            sc(beta),
                            herm,
                        )
                        .unwrap(),
                    );
                    let want = tri_ref(&[(&a, &b, alpha2c), (&b, &a, al2)], beta);
                    let err = max_err(&dec(ty, &gpu.download(dc, c0.len() * es)), &want);
                    assert!(
                        err < tol(ty),
                        "{ty:?} rank-2k herm={herm} trans={trans} uplo={uplo}: {err:e}"
                    );
                }
            }
        }
        // ---- symm / hemm, symv / hemv ----
        for herm in [false, true] {
            if herm && !cx {
                continue;
            }
            for side in [plan::SIDE_LEFT, plan::SIDE_RIGHT] {
                for uplo in [plan::FILL_LOWER, plan::FILL_UPPER] {
                    let (m, n) = (6usize, 5usize);
                    let ord = if side == plan::SIDE_LEFT { m } else { n };
                    let (lda, ldb, ldc) = (ord + 1, m + 2, m + 1);
                    let a = rng.vec(ty, lda * ord); // only the uplo triangle is meaningful
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
                    let b = rng.vec(ty, ldb * n);
                    let c0 = rng.vec(ty, ldc * n);
                    let (alpha, beta) = (rng.cx(ty), rng.cx(ty));
                    let mut want = c0.clone();
                    for j in 0..n {
                        for i in 0..m {
                            let mut acc = (0.0, 0.0);
                            for l in 0..ord {
                                let p = if side == plan::SIDE_LEFT {
                                    plan::cmul(full(i, l), at(&b, ldb, l, j))
                                } else {
                                    plan::cmul(at(&b, ldb, i, l), full(l, j))
                                };
                                acc = plan::cadd(acc, p);
                            }
                            want[i + j * ldc] = plan::cadd(
                                plan::cmul(alpha, acc),
                                plan::cmul(beta, c0[i + j * ldc]),
                            );
                        }
                    }
                    let (da, db, dc) = (
                        gpu.upload(&enc(ty, &a)),
                        gpu.upload(&enc(ty, &b)),
                        gpu.upload(&enc(ty, &c0)),
                    );
                    run(
                        &mut gpu,
                        plan::symm(
                            ty,
                            side,
                            uplo,
                            m as i32,
                            n as i32,
                            da,
                            lda as i32,
                            db,
                            ldb as i32,
                            dc,
                            ldc as i32,
                            scalar_of(ty, alpha),
                            scalar_of(ty, beta),
                            herm,
                        )
                        .unwrap(),
                    );
                    let err = max_err(&dec(ty, &gpu.download(dc, c0.len() * es)), &want);
                    assert!(
                        err < tol(ty),
                        "{ty:?} symm herm={herm} side={side} uplo={uplo}: {err:e}"
                    );
                    if side == plan::SIDE_LEFT {
                        // symv / hemv with x stride -2 and y stride 3
                        let x = rng.vec(ty, 2 * m);
                        let y0 = rng.vec(ty, 3 * m);
                        let mut want = y0.clone();
                        for i in 0..m {
                            let mut acc = (0.0, 0.0);
                            for l in 0..m {
                                acc = plan::cadd(acc, plan::cmul(full(i, l), x[(m - 1 - l) * 2]));
                            }
                            want[i * 3] =
                                plan::cadd(plan::cmul(alpha, acc), plan::cmul(beta, y0[i * 3]));
                        }
                        let (dx, dy) = (gpu.upload(&enc(ty, &x)), gpu.upload(&enc(ty, &y0)));
                        run(
                            &mut gpu,
                            plan::symv(
                                ty,
                                uplo,
                                m as i32,
                                da,
                                lda as i32,
                                dx,
                                -2,
                                dy,
                                3,
                                scalar_of(ty, alpha),
                                scalar_of(ty, beta),
                                herm,
                            )
                            .unwrap(),
                        );
                        let err = max_err(&dec(ty, &gpu.download(dy, y0.len() * es)), &want);
                        assert!(
                            err < tol(ty),
                            "{ty:?} symv herm={herm} uplo={uplo}: {err:e}"
                        );
                    }
                }
            }
        }
        // ---- trmm (out of place), trmv (from a contiguous copy), trsv ----
        let transes: &[i32] = if cx {
            &[plan::OP_N, plan::OP_T, plan::OP_C]
        } else {
            &[plan::OP_N, plan::OP_T]
        };
        for side in [plan::SIDE_LEFT, plan::SIDE_RIGHT] {
            for uplo in [plan::FILL_LOWER, plan::FILL_UPPER] {
                for &trans in transes {
                    for diag in [plan::DIAG_NON_UNIT, plan::DIAG_UNIT] {
                        let (m, n) = (6usize, 4usize);
                        let ord = if side == plan::SIDE_LEFT { m } else { n };
                        let (lda, ldb, ldc) = (ord + 2, m + 1, m + 3);
                        let mut a = rng.vec(ty, lda * ord);
                        for d in 0..ord {
                            a[d + d * lda] = plan::cadd(a[d + d * lda], (2.0, 0.0));
                        }
                        let tri = |r: usize, c: usize| -> Cx {
                            let (sr, sc) = if trans == plan::OP_N { (r, c) } else { (c, r) };
                            let inside = if uplo == plan::FILL_LOWER {
                                sr >= sc
                            } else {
                                sr <= sc
                            };
                            if sr == sc && diag == plan::DIAG_UNIT {
                                return one;
                            }
                            if !inside {
                                return (0.0, 0.0);
                            }
                            opx(&a, lda, trans, r, c)
                        };
                        let b = rng.vec(ty, ldb * n);
                        let c0 = rng.vec(ty, ldc * n);
                        let alpha = rng.cx(ty);
                        let mut want = c0.clone();
                        for j in 0..n {
                            for i in 0..m {
                                let mut acc = (0.0, 0.0);
                                for l in 0..ord {
                                    let p = if side == plan::SIDE_LEFT {
                                        plan::cmul(tri(i, l), at(&b, ldb, l, j))
                                    } else {
                                        plan::cmul(at(&b, ldb, i, l), tri(l, j))
                                    };
                                    acc = plan::cadd(acc, p);
                                }
                                want[i + j * ldc] = plan::cmul(alpha, acc);
                            }
                        }
                        let (da, db, dc) = (
                            gpu.upload(&enc(ty, &a)),
                            gpu.upload(&enc(ty, &b)),
                            gpu.upload(&enc(ty, &c0)),
                        );
                        run(
                            &mut gpu,
                            plan::trmm(
                                ty,
                                side,
                                uplo,
                                trans,
                                diag,
                                m as i32,
                                n as i32,
                                da,
                                lda as i32,
                                db,
                                ldb as i32,
                                dc,
                                ldc as i32,
                                scalar_of(ty, alpha),
                            )
                            .unwrap(),
                        );
                        let err = max_err(&dec(ty, &gpu.download(dc, c0.len() * es)), &want);
                        assert!(
                            err < tol(ty),
                            "{ty:?} trmm side={side} uplo={uplo} trans={trans} diag={diag}: {err:e}"
                        );
                        if side == plan::SIDE_LEFT {
                            // trmv: x (stride 2) = op(A) x, from a contiguous copy
                            let x = rng.vec(ty, 2 * m);
                            let mut want = x.clone();
                            for i in 0..m {
                                let mut acc = (0.0, 0.0);
                                for l in 0..m {
                                    acc = plan::cadd(acc, plan::cmul(tri(i, l), x[2 * l]));
                                }
                                want[2 * i] = acc;
                            }
                            let dx = gpu.upload(&enc(ty, &x));
                            let tmp = gpu.alloc(m * es);
                            let mut l = plan::copy(ty, m as i32, dx, 2, tmp, 1, false).unwrap();
                            l.extend(
                                plan::trmv(
                                    ty, uplo, trans, diag, m as i32, da, lda as i32, tmp, dx, 2,
                                )
                                .unwrap(),
                            );
                            run(&mut gpu, l);
                            let err = max_err(&dec(ty, &gpu.download(dx, x.len() * es)), &want);
                            assert!(
                                err < tol(ty),
                                "{ty:?} trmv uplo={uplo} trans={trans} diag={diag}: {err:e}"
                            );
                            // trsv inverts it: solve op(A) z = want (stride -1)
                            let rhs: Vec<Cx> = (0..m).map(|i| want[2 * (m - 1 - i)]).collect();
                            let dr = gpu.upload(&enc(ty, &rhs));
                            let mut p = plan::tri(
                                ty,
                                plan::SIDE_LEFT,
                                uplo,
                                trans,
                                diag,
                                m as i32,
                                1,
                                lda as i32,
                                m as i32,
                            )
                            .unwrap();
                            (p.b_vs, p.b_es) = (0, (-1i64) as u64);
                            run(
                                &mut gpu,
                                plan::trsm(
                                    ty,
                                    &p,
                                    scalar_of(ty, one),
                                    da,
                                    plan::vec_base(dr, m as i64, -1, ty),
                                    1,
                                    false,
                                )
                                .unwrap(),
                            );
                            let got = dec(ty, &gpu.download(dr, m * es));
                            let xs: Vec<Cx> = (0..m).map(|i| x[2 * (m - 1 - i)]).collect();
                            let err = max_err(&got, &xs);
                            assert!(
                                err < tol(ty) * 10.0,
                                "{ty:?} trsv uplo={uplo} trans={trans} diag={diag}: {err:e}"
                            );
                        }
                    }
                }
            }
        }
        // ---- ger / gerc, geam, dgmm, copy / swap, amax / amin ----
        let (m, n) = (9usize, 6usize);
        let lda = m + 2;
        let a0 = rng.vec(ty, lda * n);
        let (x, y) = (rng.vec(ty, 3 * m), rng.vec(ty, 2 * n));
        for conj_y in [false, true] {
            if conj_y && !cx {
                continue;
            }
            let alpha = rng.cx(ty);
            let mut want = a0.clone();
            for j in 0..n {
                for i in 0..m {
                    let yv = y[(n - 1 - j) * 2];
                    let yv = if conj_y { conj(yv) } else { yv };
                    want[i + j * lda] =
                        plan::cadd(a0[i + j * lda], plan::cmul(alpha, plan::cmul(x[3 * i], yv)));
                }
            }
            let (da, dx, dy) = (
                gpu.upload(&enc(ty, &a0)),
                gpu.upload(&enc(ty, &x)),
                gpu.upload(&enc(ty, &y)),
            );
            run(
                &mut gpu,
                plan::ger(
                    ty,
                    m as i32,
                    n as i32,
                    dx,
                    3,
                    dy,
                    -2,
                    da,
                    lda as i32,
                    scalar_of(ty, alpha),
                    conj_y,
                )
                .unwrap(),
            );
            let err = max_err(&dec(ty, &gpu.download(da, a0.len() * es)), &want);
            assert!(err < tol(ty), "{ty:?} ger conj={conj_y}: {err:e}");
        }
        for (ta, tb, beta0) in [
            (plan::OP_N, plan::OP_N, false),
            (plan::OP_T, plan::OP_N, true),
            (plan::OP_N, if cx { plan::OP_C } else { plan::OP_T }, false),
        ] {
            let ldb = lda + 1;
            let a = rng.vec(ty, lda * m.max(n));
            let b = rng.vec(ty, ldb * m.max(n));
            let c0 = vec![(f64::NAN, 0.0); lda * n];
            let (alpha, beta) = (rng.cx(ty), if beta0 { (0.0, 0.0) } else { rng.cx(ty) });
            let mut want = c0.clone();
            for j in 0..n {
                for i in 0..m {
                    let mut v = plan::cmul(alpha, opx(&a, lda, ta, i, j));
                    if !beta0 {
                        v = plan::cadd(v, plan::cmul(beta, opx(&b, ldb, tb, i, j)));
                    }
                    want[i + j * lda] = v;
                }
            }
            let (da, db, dc) = (
                gpu.upload(&enc(ty, &a)),
                gpu.upload(&enc(ty, &b)),
                gpu.upload(&enc(ty, &c0)),
            );
            run(
                &mut gpu,
                plan::geam(
                    ty,
                    ta,
                    tb,
                    m as i32,
                    n as i32,
                    da,
                    lda as i32,
                    db,
                    ldb as i32,
                    dc,
                    lda as i32,
                    scalar_of(ty, alpha),
                    scalar_of(ty, beta),
                )
                .unwrap(),
            );
            let err = max_err(&dec(ty, &gpu.download(dc, c0.len() * es)), &want);
            assert!(err < tol(ty), "{ty:?} geam ta={ta} tb={tb}: {err:e}");
        }
        {
            // Transpose only, as cuSOLVER's gesvdj does for a wide matrix: B is null with
            // ldb 0 and beta 0, so B is neither validated nor read.
            let a = rng.vec(ty, lda * m.max(n));
            let c0 = vec![(f64::NAN, 0.0); lda * n];
            let alpha = rng.cx(ty);
            let mut want = c0.clone();
            for j in 0..n {
                for i in 0..m {
                    want[i + j * lda] = plan::cmul(alpha, opx(&a, lda, plan::OP_T, i, j));
                }
            }
            let (da, dc) = (gpu.upload(&enc(ty, &a)), gpu.upload(&enc(ty, &c0)));
            let zero = scalar_of(ty, (0.0, 0.0));
            let launches = plan::geam(
                ty, plan::OP_T, plan::OP_N, m as i32, n as i32, da, lda as i32, 0, 0, dc, lda as i32,
                scalar_of(ty, alpha), zero,
            )
            .unwrap();
            run(&mut gpu, launches);
            let err = max_err(&dec(ty, &gpu.download(dc, c0.len() * es)), &want);
            assert!(err < tol(ty), "{ty:?} geam transpose with null B: {err:e}");
            assert!(plan::geam(ty, plan::OP_T, plan::OP_N, m as i32, n as i32, da, lda as i32, 8, 0, dc, lda as i32, zero, zero).is_err());
        }
        for side in [plan::SIDE_LEFT, plan::SIDE_RIGHT] {
            let len = if side == plan::SIDE_LEFT { m } else { n };
            let xv = rng.vec(ty, 2 * len);
            let mut want = a0.clone();
            for j in 0..n {
                for i in 0..m {
                    let d = xv[2 * if side == plan::SIDE_LEFT { i } else { j }];
                    want[i + j * lda] = plan::cmul(a0[i + j * lda], d);
                }
            }
            let (da, dx, dc) = (
                gpu.upload(&enc(ty, &a0)),
                gpu.upload(&enc(ty, &xv)),
                gpu.upload(&enc(ty, &a0)),
            );
            run(
                &mut gpu,
                plan::dgmm(
                    ty, side, m as i32, n as i32, da, lda as i32, dx, 2, dc, lda as i32,
                )
                .unwrap(),
            );
            let err = max_err(&dec(ty, &gpu.download(dc, a0.len() * es)), &want);
            assert!(err < tol(ty), "{ty:?} dgmm side={side}: {err:e}");
        }
        // copy with strides, swap, amax / amin (ties go to the first index)
        let nn = 300usize;
        let mut v = rng.vec(ty, 2 * nn);
        for e in v.iter_mut().filter(|e| e.0 == 0.0 && e.1 == 0.0) {
            *e = (0.5, 0.0); // a single exact zero below
        }
        v[2 * 17] = (3.0, 0.0);
        v[2 * 200] = if cx { (-1.5, 1.5) } else { (-3.0, 0.0) };
        v[2 * 50] = (0.0, 0.0);
        let dv = gpu.upload(&enc(ty, &v));
        let dw = gpu.upload(&enc(ty, &vec![(0.0, 0.0); nn]));
        run(
            &mut gpu,
            plan::copy(ty, nn as i32, dv, 2, dw, -1, false).unwrap(),
        );
        let got = dec(ty, &gpu.download(dw, nn * es));
        let want: Vec<Cx> = (0..nn).map(|i| v[2 * (nn - 1 - i)]).collect();
        assert_eq!(max_err(&got, &want), 0.0, "{ty:?} copy");
        run(
            &mut gpu,
            plan::copy(ty, nn as i32, dv, 2, dw, 1, true).unwrap(),
        );
        assert_eq!(
            max_err(
                &dec(ty, &gpu.download(dw, nn * es)),
                &(0..nn).map(|i| v[2 * i]).collect::<Vec<_>>()
            ),
            0.0,
            "{ty:?} swap y"
        );
        let dv = gpu.upload(&enc(ty, &v)); // the swap above changed it
        let out = gpu.alloc(8);
        for (min, expect) in [(false, 18u32), (true, 51)] {
            run(&mut gpu, vec![plan::amax(ty, nn as i32, dv, 2, out, min)]);
            let got = u32::from_le_bytes(gpu.download(out, 4).try_into().unwrap());
            assert_eq!(got, expect, "{ty:?} i?am{}", if min { "in" } else { "ax" });
        }
        run(&mut gpu, vec![plan::amax(ty, 0, dv, 1, out, false)]);
        assert_eq!(
            u32::from_le_bytes(gpu.download(out, 4).try_into().unwrap()),
            0
        );
    }
}

#[test]
fn sim_getrf_getrs_batched() {
    let mut sims = Sims::default();
    let mut rng = Rng(0x1u64 << 40 | 0x5151);
    for ty in Ty::BLAS {
        let es = ty.size() as usize;
        let (n, nrhs, lda, ldb, batch) = (9usize, 3usize, 11usize, 10usize, 3usize);
        let mut a = rng.vec(ty, lda * n * batch);
        // matrix 1 needs pivoting (zero leading entry); matrix 2 is singular in column 4
        a[lda * n] = (0.0, 0.0);
        for i in 0..lda {
            a[2 * lda * n + 4 * lda + i] = (0.0, 0.0);
        }
        let b = rng.vec(ty, ldb * nrhs * batch);
        let mut gpu = Gpu::default();
        let (da, db) = (gpu.upload(&enc(ty, &a)), gpu.upload(&enc(ty, &b)));
        let tab = |base: u64, s: usize, gpu: &mut Gpu| {
            gpu.upload(
                &(0..batch)
                    .flat_map(|z| (base + (z * s * es) as u64).to_le_bytes())
                    .collect::<Vec<u8>>(),
            )
        };
        let (ta, tb) = (tab(da, lda * n, &mut gpu), tab(db, ldb * nrhs, &mut gpu));
        let (piv, info) = (gpu.alloc(n * batch * 4), gpu.alloc(batch * 4));
        sims.run(
            &mut gpu,
            &plan::getrf(ty, n as i32, ta, lda as i32, piv, info, batch as i32).unwrap(),
        );
        let infos: Vec<u32> = gpu
            .download(info, batch * 4)
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!((infos[0], infos[1]), (0, 0), "{ty:?}");
        assert!(infos[2] >= 5, "{ty:?} singular matrix info {}", infos[2]);
        let lu = dec(ty, &gpu.download(da, a.len() * es));
        let pv: Vec<u32> = gpu
            .download(piv, n * batch * 4)
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        // P A = L U for the two regular matrices
        for z in 0..2 {
            let base = z * lda * n;
            let mut pa: Vec<Cx> = (0..lda * n).map(|i| a[base + i]).collect();
            for k in 0..n {
                let p = pv[z * n + k] as usize - 1;
                assert!(p >= k && p < n);
                for j in 0..n {
                    pa.swap(k + j * lda, p + j * lda);
                }
            }
            let mut want = vec![(0.0, 0.0); lda * n];
            for j in 0..n {
                for i in 0..n {
                    let mut acc = (0.0, 0.0);
                    for l in 0..=i.min(j) {
                        let lv = if l == i {
                            (1.0, 0.0)
                        } else {
                            lu[base + i + l * lda]
                        };
                        acc = plan::cadd(acc, plan::cmul(lv, lu[base + l + j * lda]));
                    }
                    want[i + j * lda] = acc;
                }
            }
            let got: Vec<Cx> = (0..n)
                .flat_map(|j| (0..n).map(move |i| (i, j)))
                .map(|(i, j)| pa[i + j * lda])
                .collect();
            let w: Vec<Cx> = (0..n)
                .flat_map(|j| (0..n).map(move |i| (i, j)))
                .map(|(i, j)| want[i + j * lda])
                .collect();
            assert!(
                max_err(&got, &w) < tol(ty) * 10.0,
                "{ty:?} getrf PA = LU matrix {z}"
            );
        }
        // getrs on the two regular matrices for op N, T and C: op(A) X = B
        let ops: &[i32] = if ty.complex() {
            &[plan::OP_N, plan::OP_T, plan::OP_C]
        } else {
            &[plan::OP_N, plan::OP_T]
        };
        for &op in ops {
            let db = gpu.upload(&enc(ty, &b));
            let tb2 = tab(db, ldb * nrhs, &mut gpu);
            sims.run(
                &mut gpu,
                &plan::getrs(
                    ty,
                    op,
                    n as i32,
                    nrhs as i32,
                    ta,
                    lda as i32,
                    piv,
                    tb2,
                    ldb as i32,
                    2,
                )
                .unwrap(),
            );
            let x = dec(ty, &gpu.download(db, b.len() * es));
            for z in 0..2 {
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
                assert!(
                    max_err(&back, &want) < tol(ty) * 100.0,
                    "{ty:?} getrs op={op} matrix {z}: {:e}",
                    max_err(&back, &want)
                );
            }
        }
        assert_eq!(plan::getrs(ty, 5, 2, 1, 0, 2, 0, 0, 2, 1), Err(-2));
        assert_eq!(plan::getrs(ty, 0, 4, 1, 0, 3, 0, 0, 4, 1), Err(-6));
        let _ = tb;
    }
}
