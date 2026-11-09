//! cuSPARSE shim kernels: translation, spirv-val, and the launch lists `plan` builds for
//! SpMV / SpMM / SDDMM / BSR and the index helpers, run in the SPIR-V simulator against a
//! dense f64 reference (`crates/cusparse_shim/src/{plan,ptxgen}.rs`).
use super::cusparse_src::plan::{self, Choice, DnMat, Scalars, SpMat};
use super::cusparse_src::ptxgen as pg;
use super::sim::{Gpu, Launch, Sim};
use super::*;
use std::collections::HashMap;

fn spirv_val(words: &[u32], name: &str) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let validator = std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
    let Ok(mut child) = Command::new(&validator).args(["--target-env", "opencl2.2", "-"]).stdin(Stdio::piped()).spawn() else {
        eprintln!("spirv-val not found; skipping validation of {name}");
        return;
    };
    let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
    let _ = child.stdin.take().unwrap().write_all(&bytes);
    assert!(child.wait().unwrap().success(), "invalid SPIR-V for {name}");
}

#[test]
fn cusparse_kernels_translate_and_validate() {
    let entries = pg::all_entries();
    assert_eq!(entries.len(), 8 * 6 + 3);
    for (name, src, n) in entries {
        let t = translate(&src).unwrap_or_else(|e| panic!("{name}: {e}\n{src}"));
        assert_eq!(t.kernels[0].name, name);
        assert_eq!(t.kernels[0].params.len(), n, "{name}");
        spirv_val(&t.words, &name);
    }
    for v in pg::Variant::all() {
        assert_eq!(translate(&pg::module(v)).unwrap().kernels.len(), 6);
    }
    assert_eq!(translate(&pg::index_module()).unwrap().kernels.len(), 3);
}

#[derive(Default)]
struct Sims {
    by_name: HashMap<String, Sim>,
    src: HashMap<String, String>,
}

impl Sims {
    fn new() -> Self {
        let src = pg::all_entries().into_iter().map(|(n, s, _)| (n, s)).collect();
        Sims { by_name: HashMap::new(), src }
    }

    fn run_named(&mut self, gpu: &mut Gpu, name: &str, grid: [u32; 3], block: u32, args: Vec<u64>) {
        let src = &self.src[name];
        let sim = self.by_name.entry(name.to_owned()).or_insert_with(|| {
            let t = translate(src).unwrap();
            Sim::new(&rspirv::dr::load_words(&t.words).unwrap())
        });
        sim.run(gpu, &Launch::new(grid, [block, 1, 1], args)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(gpu.violations.borrow().is_empty(), "{name}: {:?}", gpu.violations.borrow());
    }

    fn run(&mut self, gpu: &mut Gpu, launches: &[plan::Launch]) {
        for l in launches {
            let name = pg::entry_name(l.kernel, l.variant);
            self.run_named(gpu, &name, [l.blocks as u32, l.batch, 1], l.block, l.args.clone());
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
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n.max(1)
    }
    fn val(&mut self) -> f64 {
        (self.below(2001) as f64) / 1000.0 - 1.0
    }
}

/// Random CSR (row offsets, columns, values) with empty rows and some long rows.
fn random_csr(rng: &mut Rng, rows: usize, cols: usize, density: f64, double: bool) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let (mut ptr, mut idx, mut val) = (vec![0i64], Vec::new(), Vec::new());
    for r in 0..rows {
        if r % 5 != 3 {
            for c in 0..cols {
                if (rng.below(10_000) as f64) < density * 10_000.0 {
                    idx.push(c as i64);
                    let v = rng.val();
                    val.push(if double { v } else { f64::from(v as f32) });
                }
            }
        }
        ptr.push(idx.len() as i64);
    }
    (ptr, idx, val)
}

fn enc_idx(v: &[i64], wide: bool, base: i64) -> Vec<u8> {
    if wide { v.iter().flat_map(|x| (x + base).to_le_bytes()).collect() } else { v.iter().flat_map(|x| ((x + base) as i32).to_le_bytes()).collect() }
}

fn enc(double: bool, v: &[f64]) -> Vec<u8> {
    if double { v.iter().flat_map(|x| x.to_le_bytes()).collect() } else { v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect() }
}

fn dec(double: bool, b: &[u8]) -> Vec<f64> {
    if double { b.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect() } else { b.chunks(4).map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap()))).collect() }
}

fn upload(gpu: &mut Gpu, bytes: &[u8]) -> u64 {
    if bytes.is_empty() { gpu.alloc(8) } else { gpu.upload(bytes) }
}

/// Dense row-major `r x c` values from a stored dense matrix.
fn dense_logical(d: &DnMat, data: &[f64]) -> Vec<f64> {
    let (sr, sc) = d.strides();
    let mut out = vec![0.0; (d.rows * d.cols) as usize];
    for i in 0..d.rows as usize {
        for j in 0..d.cols as usize {
            out[i * d.cols as usize + j] = data[i * sr as usize + j * sc as usize];
        }
    }
    out
}

fn op_dense(d: &[f64], r: usize, c: usize, op: i32) -> (Vec<f64>, usize, usize) {
    if op == plan::OP_N {
        return (d.to_vec(), r, c);
    }
    let mut t = vec![0.0; r * c];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = d[i * c + j];
        }
    }
    (t, c, r)
}

fn max_rel(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / scale
}

#[derive(Clone, Copy, Debug)]
enum Fmt {
    Csr,
    Csc,
    Coo,
}

/// Builds `a` (logical `rows x cols`) in the given storage, returns (descriptor, dense).
#[allow(clippy::too_many_arguments)]
fn sparse_on(gpu: &mut Gpu, rng: &mut Rng, fmt: Fmt, rows: usize, cols: usize, density: f64, double: bool, wide: (bool, bool), base: i64) -> (SpMat, Vec<f64>) {
    // the storage is a CSR of the logical matrix (CSR/COO) or of its transpose (CSC)
    let (sr, sc) = if matches!(fmt, Fmt::Csc) { (cols, rows) } else { (rows, cols) };
    let (ptr, idx, val) = random_csr(rng, sr, sc, density, double);
    let nnz = idx.len() as i64;
    let ity = |w: bool| if w { plan::INDEX_64I } else { plan::INDEX_32I };
    let vty = if double { plan::R_64F } else { plan::R_32F };
    let (format, p0) = match fmt {
        Fmt::Csr => (plan::FORMAT_CSR, ptr.clone()),
        Fmt::Csc => (plan::FORMAT_CSC, ptr.clone()),
        Fmt::Coo => {
            let mut rr = Vec::new();
            for r in 0..sr {
                for _ in ptr[r]..ptr[r + 1] {
                    rr.push(r as i64);
                }
            }
            (plan::FORMAT_COO, rr)
        }
    };
    let pw = if matches!(fmt, Fmt::Coo) { wide.1 } else { wide.0 };
    let dp = upload(gpu, &enc_idx(&p0, pw, base));
    let di = upload(gpu, &enc_idx(&idx, wide.1, base));
    let dv = upload(gpu, &enc(double, &val));
    let a = SpMat::new(format, rows as i64, cols as i64, nnz, dp, di, dv, ity(pw), ity(wide.1), base as i32, vty).unwrap();
    let based: Vec<i64> = p0.iter().map(|x| x + base).collect();
    let based_i: Vec<i64> = idx.iter().map(|x| x + base).collect();
    let dense = plan::dense_of(&a, &based, &based_i, &val);
    (a, dense)
}

#[test]
fn sim_spmv_spmm_all_formats_ops_and_types() {
    let mut sims = Sims::new();
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let mut cases = 0;
    for double in [false, true] {
        let tol = if double { 1e-12 } else { 2e-5 };
        let vty = if double { plan::R_64F } else { plan::R_32F };
        for (fi, fmt) in [Fmt::Csr, Fmt::Csc, Fmt::Coo].into_iter().enumerate() {
            for op_a in [plan::OP_N, plan::OP_T] {
                for (wi, wide) in [(false, false), (true, true), (true, false)].into_iter().enumerate() {
                    if matches!(fmt, Fmt::Coo) && wide.0 != wide.1 {
                        continue;
                    }
                    let base = ((fi + wi) % 2) as i64;
                    let (rows, cols) = (13 + wi * 7, 9 + fi * 5);
                    let mut gpu = Gpu::default();
                    let (a, dense) = sparse_on(&mut gpu, &mut rng, fmt, rows, cols, 0.35, double, wide, base);
                    let (opd, m, k) = op_dense(&dense, rows, cols, op_a);
                    // SpMV and SpMM (n = 3, B row-major op N, B column-major op T)
                    for (n, op_b, order) in [(1usize, plan::OP_N, plan::ORDER_COL), (3, plan::OP_N, plan::ORDER_ROW), (3, plan::OP_T, plan::ORDER_COL), (2, plan::OP_T, plan::ORDER_ROW)] {
                        for (alpha, beta, choice) in [(1.0, 0.0, Choice::Scalar), (0.75, -0.5, Choice::Vector), (-1.25, 2.0, Choice::Auto)] {
                            let (br, bc) = if op_b == plan::OP_N { (k, n) } else { (n, k) };
                            let bld = if order == plan::ORDER_COL { br + 1 } else { bc + 2 };
                            let b_store: Vec<f64> = (0..bld * if order == plan::ORDER_COL { bc } else { br }).map(|_| rng.val()).collect();
                            let bdesc = DnMat::new(br as i64, bc as i64, bld as i64, upload(&mut gpu, &enc(double, &b_store)), vty, order).unwrap();
                            let mut c_store: Vec<f64> = (0..m * n).map(|_| rng.val()).collect();
                            if beta == 0.0 {
                                c_store[0] = f64::NAN; // beta 0 must not read C
                            }
                            let cdesc = DnMat::new(m as i64, n as i64, m.max(1) as i64, upload(&mut gpu, &enc(double, &c_store)), vty, plan::ORDER_COL).unwrap();
                            let s = Scalars::host(alpha, beta, double);
                            let launches = plan::spmm(op_a, op_b, &a, &bdesc, &cdesc, vty, s, choice).unwrap();
                            sims.run(&mut gpu, &launches);
                            let got = dec(double, &gpu.download(cdesc.val, m * n * if double { 8 } else { 4 }));
                            let bl = dense_logical(&bdesc, &dec(double, &enc(double, &b_store)));
                            let (bop, _, _) = op_dense(&bl, br, bc, op_b);
                            let cl0 = dense_logical(&cdesc, &dec(double, &enc(double, &c_store)));
                            let mut want = vec![0.0; m * n];
                            for i in 0..m {
                                for j in 0..n {
                                    let dot: f64 = (0..k).map(|l| opd[i * k + l] * bop[l * n + j]).sum();
                                    let prev = if beta == 0.0 { 0.0 } else { beta * cl0[i * n + j] };
                                    want[i * n + j] = alpha * dot + prev;
                                }
                            }
                            let gl = dense_logical(&cdesc, &got);
                            let e = max_rel(&gl, &want);
                            assert!(e < tol, "{fmt:?} op_a={op_a} op_b={op_b} n={n} wide={wide:?} base={base} double={double} {choice:?}: {e}");
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert!(cases > 300, "{cases}");
}

#[test]
fn sim_spmm_batched_and_device_scalars() {
    let mut sims = Sims::new();
    let mut rng = Rng(0xfeed_beef_0101_2323);
    for double in [false, true] {
        let vty = if double { plan::R_64F } else { plan::R_32F };
        let sz = if double { 8 } else { 4 };
        let mut gpu = Gpu::default();
        let (rows, cols, n, batch) = (6usize, 5usize, 4usize, 3usize);
        // three CSR matrices with the same nnz layout but distinct values, stored back to back
        let (ptr, idx, _) = random_csr(&mut rng, rows, cols, 0.5, double);
        let nnz = idx.len();
        let vals: Vec<f64> = (0..nnz * batch).map(|_| rng.val()).collect();
        let pb: Vec<i64> = (0..batch).flat_map(|_| ptr.iter().copied()).collect();
        let ib: Vec<i64> = (0..batch).flat_map(|_| idx.iter().copied()).collect();
        let mut a = SpMat::new(plan::FORMAT_CSR, rows as i64, cols as i64, nnz as i64, upload(&mut gpu, &enc_idx(&pb, false, 0)), upload(&mut gpu, &enc_idx(&ib, false, 0)), upload(&mut gpu, &enc(double, &vals)), plan::INDEX_32I, plan::INDEX_32I, 0, vty).unwrap();
        a.batch = batch as i32;
        let bv: Vec<f64> = (0..cols * n * batch).map(|_| rng.val()).collect();
        let mut b = DnMat::new(cols as i64, n as i64, n as i64, upload(&mut gpu, &enc(double, &bv)), vty, plan::ORDER_ROW).unwrap();
        b.batch = batch as i32;
        let mut c = DnMat::new(rows as i64, n as i64, n as i64, gpu.alloc(rows * n * batch * sz), vty, plan::ORDER_ROW).unwrap();
        c.batch = batch as i32;
        let (alpha, beta) = (1.5f64, 0.0f64);
        let sbytes = [enc(double, &[alpha]), enc(double, &[beta])].concat();
        let sp = upload(&mut gpu, &sbytes);
        let s = Scalars { alpha: 0, beta: 0, alpha_ptr: sp, beta_ptr: sp + sz as u64 };
        for choice in [Choice::Scalar, Choice::Vector] {
            sims.run(&mut gpu, &plan::spmm(plan::OP_N, plan::OP_N, &a, &b, &c, vty, s, choice).unwrap());
            let got = dec(double, &gpu.download(c.val, rows * n * batch * sz));
            for bi in 0..batch {
                for i in 0..rows {
                    for j in 0..n {
                        let mut want = 0.0;
                        for e in ptr[i] as usize..ptr[i + 1] as usize {
                            want += vals[bi * nnz + e] * bv[bi * cols * n + idx[e] as usize * n + j];
                        }
                        let g = got[bi * rows * n + i * n + j];
                        assert!((g - alpha * want).abs() < 1e-4, "batch {bi} ({i},{j}) {choice:?}: {g} vs {}", alpha * want);
                    }
                }
            }
        }
    }
}

#[test]
fn sim_sddmm_and_bsr() {
    let mut sims = Sims::new();
    let mut rng = Rng(0x0bad_cafe_1357_9bdf);
    for double in [false, true] {
        let vty = if double { plan::R_64F } else { plan::R_32F };
        let sz = if double { 8 } else { 4 };
        let tol = if double { 1e-12 } else { 2e-5 };
        // SDDMM: C (CSR m x n) = alpha * (op(A) op(B)) .* spy(C) + beta * C
        for (op_a, op_b) in [(plan::OP_N, plan::OP_N), (plan::OP_T, plan::OP_T)] {
            let mut gpu = Gpu::default();
            let (m, n, k) = (7usize, 6usize, 5usize);
            let (ptr, idx, val) = random_csr(&mut rng, m, n, 0.4, double);
            let c = SpMat::new(plan::FORMAT_CSR, m as i64, n as i64, idx.len() as i64, upload(&mut gpu, &enc_idx(&ptr, true, 0)), upload(&mut gpu, &enc_idx(&idx, false, 0)), upload(&mut gpu, &enc(double, &val)), plan::INDEX_64I, plan::INDEX_32I, 0, vty).unwrap();
            let (ar, ac) = if op_a == plan::OP_N { (m, k) } else { (k, m) };
            let (br, bc) = if op_b == plan::OP_N { (k, n) } else { (n, k) };
            let av: Vec<f64> = (0..ar * ac).map(|_| rng.val()).collect();
            let bv: Vec<f64> = (0..br * bc).map(|_| rng.val()).collect();
            let a = DnMat::new(ar as i64, ac as i64, ac as i64, upload(&mut gpu, &enc(double, &av)), vty, plan::ORDER_ROW).unwrap();
            let b = DnMat::new(br as i64, bc as i64, br as i64, upload(&mut gpu, &enc(double, &bv)), vty, plan::ORDER_COL).unwrap();
            let (alpha, beta) = (0.5, 2.0);
            sims.run(&mut gpu, &plan::sddmm(op_a, op_b, &a, &b, &c, vty, Scalars::host(alpha, beta, double)).unwrap());
            let got = dec(double, &gpu.download(c.val, idx.len() * sz));
            let (aop, _, _) = op_dense(&dense_logical(&a, &dec(double, &enc(double, &av))), ar, ac, op_a);
            let (bop, _, _) = op_dense(&dense_logical(&b, &dec(double, &enc(double, &bv))), br, bc, op_b);
            let mut want = Vec::new();
            for i in 0..m {
                for e in ptr[i] as usize..ptr[i + 1] as usize {
                    let j = idx[e] as usize;
                    let dot: f64 = (0..k).map(|l| aop[i * k + l] * bop[l * n + j]).sum();
                    want.push(alpha * dot + beta * val[e]);
                }
            }
            assert!(max_rel(&got, &want) < tol, "sddmm {op_a} {op_b} double={double}");
        }
        // BSR: mb x kb blocks of bd x bd, both block orders, op(B) N and T
        for (dir, op_b, bd, base) in [(0, plan::OP_N, 3i64, 0i32), (1, plan::OP_T, 2, 1), (0, plan::OP_T, 1, 1)] {
            let mut gpu = Gpu::default();
            let (mb, kb, n) = (4usize, 3usize, 5usize);
            let (ptr, idx, _) = random_csr(&mut rng, mb, kb, 0.6, double);
            let nnzb = idx.len();
            let bdu = bd as usize;
            let vals: Vec<f64> = (0..nnzb * bdu * bdu).map(|_| rng.val()).collect();
            let (m, k) = (mb * bdu, kb * bdu);
            let (br, bc) = if op_b == plan::OP_N { (k, n) } else { (n, k) };
            let ldb = br + 1;
            let bv: Vec<f64> = (0..ldb * bc).map(|_| rng.val()).collect();
            let cv: Vec<f64> = (0..m * n).map(|_| rng.val()).collect();
            let (dp, di, dv) = (upload(&mut gpu, &enc_idx(&ptr, false, base as i64)), upload(&mut gpu, &enc_idx(&idx, false, base as i64)), upload(&mut gpu, &enc(double, &vals)));
            let (db, dc) = (upload(&mut gpu, &enc(double, &bv)), upload(&mut gpu, &enc(double, &cv)));
            let (alpha, beta) = (1.0, -1.0);
            let s = Scalars::host(alpha, beta, double);
            sims.run(&mut gpu, &plan::bsrmm(double, dir, mb as i64, n as i64, kb as i64, nnzb as i64, bd, base, op_b, dp, di, dv, db, ldb as i64, dc, m as i64, s).unwrap());
            let got = dec(double, &gpu.download(dc, m * n * sz));
            let bq = dec(double, &enc(double, &bv));
            let cq = dec(double, &enc(double, &cv));
            let mut dense = vec![0.0; m * k];
            for ib in 0..mb {
                for e in ptr[ib] as usize..ptr[ib + 1] as usize {
                    for ri in 0..bdu {
                        for cj in 0..bdu {
                            let v = vals[e * bdu * bdu + if dir == 0 { ri * bdu + cj } else { cj * bdu + ri }];
                            dense[(ib * bdu + ri) * k + idx[e] as usize * bdu + cj] = dec(double, &enc(double, &[v]))[0];
                        }
                    }
                }
            }
            for i in 0..m {
                for j in 0..n {
                    let bij = |l: usize| if op_b == plan::OP_N { bq[l + j * ldb] } else { bq[j + l * ldb] };
                    let dot: f64 = (0..k).map(|l| dense[i * k + l] * bij(l)).sum();
                    let want = alpha * dot + beta * cq[i + j * m];
                    assert!((got[i + j * m] - want).abs() < tol * 10.0, "bsr dir={dir} op_b={op_b} bd={bd} ({i},{j}): {} vs {want}", got[i + j * m]);
                }
            }
        }
    }
}

#[test]
fn sim_index_kernels() {
    let mut sims = Sims::new();
    let mut rng = Rng(0x5555_aaaa_1234_4321);
    for base in [0i64, 1] {
        let mut gpu = Gpu::default();
        let m = 11usize;
        let (ptr, _, _) = random_csr(&mut rng, m, 8, 0.4, true);
        let nnz = *ptr.last().unwrap() as usize;
        let mut rows = Vec::new();
        for r in 0..m {
            for _ in ptr[r]..ptr[r + 1] {
                rows.push(r as i64);
            }
        }
        let dcoo = upload(&mut gpu, &enc_idx(&rows, false, base));
        let dcsr = gpu.alloc((m + 1) * 4);
        let grid = |t: usize| [(t as u32).div_ceil(plan::BLOCK), 1, 1];
        sims.run_named(&mut gpu, pg::COO2CSR, grid(m + 1), plan::BLOCK, vec![dcoo, nnz as u64, m as u64, dcsr, base as u64]);
        let got: Vec<i64> = gpu.download(dcsr, (m + 1) * 4).chunks(4).map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap()))).collect();
        let want: Vec<i64> = ptr.iter().map(|p| p + base).collect();
        assert_eq!(got, want, "coo2csr base {base}");
        let dout = gpu.alloc(nnz.max(1) * 4);
        sims.run_named(&mut gpu, pg::CSR2COO, grid(m), plan::BLOCK, vec![dcsr, nnz as u64, m as u64, dout, base as u64]);
        let got: Vec<i64> = gpu.download(dout, nnz * 4).chunks(4).map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap()))).collect();
        assert_eq!(got, rows.iter().map(|r| r + base).collect::<Vec<_>>(), "csr2coo base {base}");
    }
    let mut gpu = Gpu::default();
    let p = gpu.alloc(37 * 4);
    sims.run_named(&mut gpu, pg::IDENTITY, [1, 1, 1], plan::BLOCK, vec![p, 37, 0, 0, 0]);
    let got: Vec<i32> = gpu.download(p, 37 * 4).chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
    assert_eq!(got, (0..37).collect::<Vec<_>>());
}
