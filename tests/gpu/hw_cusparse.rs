//! cuSPARSE shim on the GPU (`cusparse64_12.dll`) against a host f64 reference: SpMV and
//! SpMM over CSR/CSC/COO with random sparsity and empty rows, op(A) N/T, op(B) N/T, both
//! dense orders, 32/64-bit indices, base 0/1, alpha/beta (incl. beta 0 over NaN), scalar and
//! vector CSR kernels, strided batches, device pointer mode, SDDMM, bsrmv/bsrmm and the
//! index helpers; f16 values return NOT_SUPPORTED.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::c_void;
use std::ptr::null_mut;

type P = *mut c_void;

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

/// CSR of a `rows x cols` matrix: every fifth row empty, `long` rows dense-ish.
fn random_csr(rng: &mut Rng, rows: usize, cols: usize, per_row: usize, long: bool) -> (Vec<i64>, Vec<i64>, Vec<f64>) {
    let (mut ptr, mut idx, mut val) = (vec![0i64], Vec::new(), Vec::new());
    for r in 0..rows {
        if r % 5 != 3 {
            let want = if long && r % 2 == 0 { cols * 3 / 4 } else { rng.below(per_row as u64 * 2 + 1) as usize };
            let mut cs: Vec<usize> = (0..cols).collect();
            for i in 0..want.min(cols) {
                let j = i + rng.below((cols - i) as u64) as usize;
                cs.swap(i, j);
            }
            let mut pick = cs[..want.min(cols)].to_vec();
            pick.sort();
            for c in pick {
                idx.push(c as i64);
                val.push(rng.val());
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

fn round(double: bool, v: &[f64]) -> Vec<f64> {
    dec(double, &enc(double, v))
}

fn max_rel(got: &[f64], want: &[f64]) -> f64 {
    let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
    got.iter().zip(want).map(|(a, b)| if a.is_nan() { f64::INFINITY } else { (a - b).abs() }).fold(0.0, f64::max) / scale
}

type CreateCsr = unsafe extern "system" fn(*mut P, i64, i64, i64, P, P, P, i32, i32, i32, i32) -> i32;
type CreateCoo = unsafe extern "system" fn(*mut P, i64, i64, i64, P, P, P, i32, i32, i32) -> i32;
type CreateDnVec = unsafe extern "system" fn(*mut P, i64, P, i32) -> i32;
type CreateDnMat = unsafe extern "system" fn(*mut P, i64, i64, i64, P, i32, i32) -> i32;
type Destroy = unsafe extern "system" fn(P) -> i32;
type SpMV = unsafe extern "system" fn(P, i32, *const c_void, P, P, *const c_void, P, i32, i32, P) -> i32;
type SpMVSize = unsafe extern "system" fn(P, i32, *const c_void, P, P, *const c_void, P, i32, i32, *mut usize) -> i32;
type SpMM = unsafe extern "system" fn(P, i32, i32, *const c_void, P, P, *const c_void, P, i32, i32, P) -> i32;
type SpMMSize = unsafe extern "system" fn(P, i32, i32, *const c_void, P, P, *const c_void, P, i32, i32, *mut usize) -> i32;
type Bsrmm = unsafe extern "system" fn(P, i32, i32, i32, i32, i32, i32, i32, *const c_void, P, P, P, P, i32, P, i32, *const c_void, P, i32) -> i32;
type Bsrmv = unsafe extern "system" fn(P, i32, i32, i32, i32, i32, *const c_void, P, P, P, P, i32, P, *const c_void, P) -> i32;

struct Api {
    lib: libloading::Library,
}

impl Api {
    fn f<T: Copy>(&self, name: &str) -> T {
        *unsafe { self.lib.get::<T>(format!("{name}\0").as_bytes()) }.unwrap_or_else(|_| panic!("{name}"))
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Fmt {
    Csr,
    Csc,
    Coo,
}

#[test]
#[ignore = "requires a built nvcuda.dll, cusparse64_12.dll and an Intel Level Zero GPU"]
fn hw_cusparse_matches_host_reference() {
    let dir = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().to_path_buf();
    let api = Api { lib: unsafe { libloading::Library::new(dir.join("cusparse64_12.dll")) }.expect("cusparse64_12.dll") };
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
        cuStreamSynchronize(Handle),
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

    let create: unsafe extern "system" fn(*mut P) -> i32 = api.f("cusparseCreate");
    let destroy_h: Destroy = api.f("cusparseDestroy");
    let set_stream: unsafe extern "system" fn(P, Handle) -> i32 = api.f("cusparseSetStream");
    let set_mode: unsafe extern "system" fn(P, i32) -> i32 = api.f("cusparseSetPointerMode");
    let create_csr: CreateCsr = api.f("cusparseCreateCsr");
    let create_csc: CreateCsr = api.f("cusparseCreateCsc");
    let create_coo: CreateCoo = api.f("cusparseCreateCoo");
    let create_vec: CreateDnVec = api.f("cusparseCreateDnVec");
    let create_mat: CreateDnMat = api.f("cusparseCreateDnMat");
    let destroy_sp: Destroy = api.f("cusparseDestroySpMat");
    let destroy_vec: Destroy = api.f("cusparseDestroyDnVec");
    let destroy_mat: Destroy = api.f("cusparseDestroyDnMat");
    let spmv: SpMV = api.f("cusparseSpMV");
    let spmv_size: SpMVSize = api.f("cusparseSpMV_bufferSize");
    let spmm: SpMM = api.f("cusparseSpMM");
    let spmm_size: SpMMSize = api.f("cusparseSpMM_bufferSize");
    let csr_batch: unsafe extern "system" fn(P, i32, i64, i64) -> i32 = api.f("cusparseCsrSetStridedBatch");
    let mat_batch: unsafe extern "system" fn(P, i32, i64) -> i32 = api.f("cusparseDnMatSetStridedBatch");

    let mut h: P = null_mut();
    assert_eq!(unsafe { create(&mut h) }, 0);
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 0));
    assert_eq!(unsafe { set_stream(h, stream) }, 0);

    let mut rng = Rng(0x2468_ace0_1357_9bdf);
    let mut cases = 0;
    let mut worst = [0.0f64; 2];

    // Builds A in storage `fmt` for the logical rows x cols matrix; returns (descr, dense row-major).
    let make_sparse = |rng: &mut Rng, fmt: Fmt, rows: usize, cols: usize, per_row: usize, long: bool, double: bool, wide: bool, base: i64| -> (P, Vec<f64>, usize) {
        let (sr, sc) = if fmt == Fmt::Csc { (cols, rows) } else { (rows, cols) };
        let (ptr, idx, val) = random_csr(rng, sr, sc, per_row, long);
        let val = round(double, &val);
        let nnz = idx.len();
        let mut dense = vec![0.0; rows * cols];
        for r in 0..sr {
            for e in ptr[r] as usize..ptr[r + 1] as usize {
                let c = idx[e] as usize;
                if fmt == Fmt::Csc { dense[c * cols + r] += val[e] } else { dense[r * cols + c] += val[e] }
            }
        }
        let ity = if wide { 3 } else { 2 };
        let vty = if double { 1 } else { 0 };
        let dv = up(&enc(double, &val)) as P;
        let di = up(&enc_idx(&idx, wide, base)) as P;
        let mut d: P = null_mut();
        let st = match fmt {
            Fmt::Csr => unsafe { create_csr(&mut d, rows as i64, cols as i64, nnz as i64, up(&enc_idx(&ptr, wide, base)) as P, di, dv, ity, ity, base as i32, vty) },
            Fmt::Csc => unsafe { create_csc(&mut d, rows as i64, cols as i64, nnz as i64, up(&enc_idx(&ptr, !wide, base)) as P, di, dv, if wide { 2 } else { 3 }, ity, base as i32, vty) },
            Fmt::Coo => {
                let mut rr = Vec::new();
                for r in 0..sr {
                    for _ in ptr[r]..ptr[r + 1] {
                        rr.push(r as i64);
                    }
                }
                unsafe { create_coo(&mut d, rows as i64, cols as i64, nnz as i64, up(&enc_idx(&rr, wide, base)) as P, di, dv, ity, base as i32, vty) }
            }
        };
        assert_eq!(st, 0, "create {fmt:?}");
        (d, dense, nnz)
    };

    // --- SpMV and SpMM over formats, ops, index widths, bases, scalars and orders
    let shapes: &[(usize, usize, usize, bool)] = &[(37, 29, 4, false), (300, 211, 6, false), (64, 900, 0, true), (1, 1, 1, false), (2000, 1500, 3, false)];
    for double in [false, true] {
        let (vty, sz) = if double { (1, 8) } else { (0, 4) };
        let tol = if double { 1e-12 } else { 1e-5 };
        for (si, &(rows, cols, per_row, long)) in shapes.iter().enumerate() {
            for fmt in [Fmt::Csr, Fmt::Csc, Fmt::Coo] {
                for op_a in [0, 1] {
                    let wide = (si + op_a as usize) % 2 == 1;
                    let base = ((si + fmt as usize) % 2) as i64;
                    let (a, dense, nnz) = make_sparse(&mut rng, fmt, rows, cols, per_row, long, double, wide, base);
                    let (m, k) = if op_a == 0 { (rows, cols) } else { (cols, rows) };
                    let opd = |i: usize, l: usize| if op_a == 0 { dense[i * cols + l] } else { dense[l * cols + i] };
                    for &(alpha, beta) in &[(1.0, 0.0), (0.75, -0.5)] {
                        let (av, bv) = (enc(double, &[alpha]), enc(double, &[beta]));
                        // SpMV
                        let x = round(double, &(0..k).map(|_| rng.val()).collect::<Vec<_>>());
                        let mut y0 = round(double, &(0..m).map(|_| rng.val()).collect::<Vec<_>>());
                        if beta == 0.0 {
                            y0[0] = f64::NAN;
                        }
                        let (dx, dy) = (up(&enc(double, &x)), up(&enc(double, &y0)));
                        let (mut vx, mut vy) = (null_mut(), null_mut());
                        assert_eq!(unsafe { create_vec(&mut vx, k as i64, dx as P, vty) }, 0);
                        assert_eq!(unsafe { create_vec(&mut vy, m as i64, dy as P, vty) }, 0);
                        let mut ws = 1usize;
                        assert_eq!(unsafe { spmv_size(h, op_a, av.as_ptr().cast(), a, vx, bv.as_ptr().cast(), vy, vty, 0, &mut ws) }, 0);
                        assert_eq!(unsafe { spmv(h, op_a, av.as_ptr().cast(), a, vx, bv.as_ptr().cast(), vy, vty, 0, null_mut()) }, 0, "spmv {fmt:?}");
                        ok!(cuStreamSynchronize(stream));
                        let got = dec(double, &down(dy, m * sz));
                        let want: Vec<f64> = (0..m).map(|i| alpha * (0..k).map(|l| opd(i, l) * x[l]).sum::<f64>() + if beta == 0.0 { 0.0 } else { beta * y0[i] }).collect();
                        let e = max_rel(&got, &want);
                        worst[double as usize] = worst[double as usize].max(e);
                        assert!(e < tol, "SpMV {fmt:?} {rows}x{cols} nnz={nnz} op={op_a} wide={wide} base={base} double={double} a={alpha} b={beta}: {e}");
                        unsafe {
                            destroy_vec(vx);
                            destroy_vec(vy);
                        }
                        cases += 1;
                        // SpMM: n = 5, op(B) N row-major / T column-major
                        if rows * cols > 1_000_000 {
                            continue;
                        }
                        for (op_b, order) in [(0, 2), (1, 1)] {
                            let n = 5usize;
                            let (br, bc) = if op_b == 0 { (k, n) } else { (n, k) };
                            let ld = if order == 1 { br + 3 } else { bc + 1 };
                            let bs = round(double, &(0..ld * if order == 1 { bc } else { br }).map(|_| rng.val()).collect::<Vec<_>>());
                            let bat = |i: usize, j: usize| if order == 1 { bs[i + j * ld] } else { bs[i * ld + j] };
                            let opb = |l: usize, j: usize| if op_b == 0 { bat(l, j) } else { bat(j, l) };
                            let c0 = round(double, &(0..m * n).map(|_| rng.val()).collect::<Vec<_>>());
                            let (db, dc) = (up(&enc(double, &bs)), up(&enc(double, &c0)));
                            let (mut mb, mut mc) = (null_mut(), null_mut());
                            assert_eq!(unsafe { create_mat(&mut mb, br as i64, bc as i64, ld as i64, db as P, vty, order) }, 0);
                            assert_eq!(unsafe { create_mat(&mut mc, m as i64, n as i64, n as i64, dc as P, vty, 2) }, 0);
                            assert_eq!(unsafe { spmm_size(h, op_a, op_b, av.as_ptr().cast(), a, mb, bv.as_ptr().cast(), mc, vty, 0, &mut ws) }, 0);
                            assert_eq!(unsafe { spmm(h, op_a, op_b, av.as_ptr().cast(), a, mb, bv.as_ptr().cast(), mc, vty, 0, null_mut()) }, 0);
                            let got = dec(double, &down(dc, m * n * sz));
                            let mut want = vec![0.0; m * n];
                            for i in 0..m {
                                for j in 0..n {
                                    want[i * n + j] = alpha * (0..k).map(|l| opd(i, l) * opb(l, j)).sum::<f64>() + beta * c0[i * n + j];
                                }
                            }
                            let e = max_rel(&got, &want);
                            worst[double as usize] = worst[double as usize].max(e);
                            assert!(e < tol, "SpMM {fmt:?} {rows}x{cols} op_a={op_a} op_b={op_b} order={order} double={double}: {e}");
                            unsafe {
                                destroy_mat(mb);
                                destroy_mat(mc);
                            }
                            cases += 1;
                        }
                    }
                    assert_eq!(unsafe { destroy_sp(a) }, 0);
                }
            }
        }
    }
    println!("  SpMV/SpMM: {cases} cases, worst rel err f32 {:.2e}, f64 {:.2e}", worst[0], worst[1]);

    // --- strided batch (torch's batched CSR @ dense) and device pointer mode
    for double in [false, true] {
        let (vty, sz) = if double { (1, 8) } else { (0, 4) };
        let (rows, cols, n, batch) = (40usize, 30usize, 6usize, 3usize);
        let (ptr, idx, _) = random_csr(&mut rng, rows, cols, 5, false);
        let nnz = idx.len();
        let vals = round(double, &(0..nnz * batch).map(|_| rng.val()).collect::<Vec<_>>());
        let pb: Vec<i64> = (0..batch).flat_map(|_| ptr.iter().copied()).collect();
        let ib: Vec<i64> = (0..batch).flat_map(|_| idx.iter().copied()).collect();
        let mut a = null_mut();
        assert_eq!(unsafe { create_csr(&mut a, rows as i64, cols as i64, nnz as i64, up(&enc_idx(&pb, true, 0)) as P, up(&enc_idx(&ib, true, 0)) as P, up(&enc(double, &vals)) as P, 3, 3, 0, vty) }, 0);
        assert_eq!(unsafe { csr_batch(a, batch as i32, (rows + 1) as i64, nnz as i64) }, 0);
        let bvals = round(double, &(0..cols * n * batch).map(|_| rng.val()).collect::<Vec<_>>());
        let (db, dc) = (up(&enc(double, &bvals)), up(&vec![0u8; rows * n * batch * sz]));
        let (mut mb, mut mc) = (null_mut(), null_mut());
        assert_eq!(unsafe { create_mat(&mut mb, cols as i64, n as i64, n as i64, db as P, vty, 2) }, 0);
        assert_eq!(unsafe { create_mat(&mut mc, rows as i64, n as i64, n as i64, dc as P, vty, 2) }, 0);
        assert_eq!(unsafe { mat_batch(mb, batch as i32, (cols * n) as i64) }, 0);
        assert_eq!(unsafe { mat_batch(mc, batch as i32, (rows * n) as i64) }, 0);
        let scal = up(&enc(double, &[2.0, 0.0]));
        assert_eq!(unsafe { set_mode(h, 1) }, 0);
        assert_eq!(unsafe { spmm(h, 0, 0, scal as *const c_void, a, mb, (scal + sz as u64) as *const c_void, mc, vty, 0, null_mut()) }, 0);
        assert_eq!(unsafe { set_mode(h, 0) }, 0);
        let got = dec(double, &down(dc, rows * n * batch * sz));
        let mut want = vec![0.0; rows * n * batch];
        for b in 0..batch {
            for i in 0..rows {
                for j in 0..n {
                    want[b * rows * n + i * n + j] = 2.0 * (ptr[i] as usize..ptr[i + 1] as usize).map(|e| vals[b * nnz + e] * bvals[b * cols * n + idx[e] as usize * n + j]).sum::<f64>();
                }
            }
        }
        let e = max_rel(&got, &want);
        println!("  batched SpMM x{batch} device scalars double={double}: rel err {e:.2e}");
        assert!(e < if double { 1e-12 } else { 1e-5 });
        unsafe {
            destroy_mat(mb);
            destroy_mat(mc);
            destroy_sp(a);
        }
        cases += 1;
    }

    // --- SDDMM: C (CSR) = alpha * (A B) .* spy(C) + beta * C
    {
        let sddmm: SpMM = api.f("cusparseSDDMM");
        for double in [false, true] {
            let (vty, sz) = if double { (1, 8) } else { (0, 4) };
            let (m, n, k) = (50usize, 40usize, 17usize);
            let (ptr, idx, val) = random_csr(&mut rng, m, n, 6, false);
            let val = round(double, &val);
            let av = round(double, &(0..m * k).map(|_| rng.val()).collect::<Vec<_>>());
            let bvv = round(double, &(0..k * n).map(|_| rng.val()).collect::<Vec<_>>());
            let dval = up(&enc(double, &val));
            let mut c = null_mut();
            assert_eq!(unsafe { create_csr(&mut c, m as i64, n as i64, idx.len() as i64, up(&enc_idx(&ptr, false, 0)) as P, up(&enc_idx(&idx, false, 0)) as P, dval as P, 2, 2, 0, vty) }, 0);
            let (mut ma, mut mb) = (null_mut(), null_mut());
            assert_eq!(unsafe { create_mat(&mut ma, m as i64, k as i64, k as i64, up(&enc(double, &av)) as P, vty, 2) }, 0);
            assert_eq!(unsafe { create_mat(&mut mb, k as i64, n as i64, k as i64, up(&enc(double, &bvv)) as P, vty, 1) }, 0);
            let (alpha, beta) = (enc(double, &[1.5]), enc(double, &[-1.0]));
            assert_eq!(unsafe { sddmm(h, 0, 0, alpha.as_ptr().cast(), ma, mb, beta.as_ptr().cast(), c, vty, 0, null_mut()) }, 0);
            let got = dec(double, &down(dval, val.len() * sz));
            let mut want = Vec::new();
            for i in 0..m {
                for e in ptr[i] as usize..ptr[i + 1] as usize {
                    let j = idx[e] as usize;
                    want.push(1.5 * (0..k).map(|l| av[i * k + l] * bvv[l + j * k]).sum::<f64>() - val[e]);
                }
            }
            let e = max_rel(&got, &want);
            println!("  SDDMM double={double}: rel err {e:.2e}");
            assert!(e < if double { 1e-12 } else { 1e-5 });
            unsafe {
                destroy_mat(ma);
                destroy_mat(mb);
                destroy_sp(c);
            }
            cases += 1;
        }
    }

    // --- legacy BSR products and index helpers
    {
        let create_descr: unsafe extern "system" fn(*mut P) -> i32 = api.f("cusparseCreateMatDescr");
        let set_base: unsafe extern "system" fn(P, i32) -> i32 = api.f("cusparseSetMatIndexBase");
        let mut descr = null_mut();
        assert_eq!(unsafe { create_descr(&mut descr) }, 0);
        for (double, dir, bd, base) in [(false, 0, 3usize, 0i64), (true, 1, 2, 1), (false, 1, 4, 1), (true, 0, 1, 0)] {
            let sz = if double { 8 } else { 4 };
            assert_eq!(unsafe { set_base(descr, base as i32) }, 0);
            let (mb, kb, n) = (23usize, 17usize, 4usize);
            let (ptr, idx, _) = random_csr(&mut rng, mb, kb, 3, false);
            let nnzb = idx.len();
            let vals = round(double, &(0..nnzb * bd * bd).map(|_| rng.val()).collect::<Vec<_>>());
            let (m, k) = (mb * bd, kb * bd);
            let mut dense = vec![0.0; m * k];
            for ib in 0..mb {
                for e in ptr[ib] as usize..ptr[ib + 1] as usize {
                    for ri in 0..bd {
                        for cj in 0..bd {
                            dense[(ib * bd + ri) * k + idx[e] as usize * bd + cj] = vals[e * bd * bd + if dir == 0 { ri * bd + cj } else { cj * bd + ri }];
                        }
                    }
                }
            }
            let (dp, di, dv) = (up(&enc_idx(&ptr, false, base)) as P, up(&enc_idx(&idx, false, base)) as P, up(&enc(double, &vals)) as P);
            let (alpha, beta) = (enc(double, &[0.5]), enc(double, &[2.0]));
            // bsrmm with op(B) = T: B is n x k column-major
            let ldb = n + 1;
            let bvals = round(double, &(0..ldb * k).map(|_| rng.val()).collect::<Vec<_>>());
            let c0 = round(double, &(0..m * n).map(|_| rng.val()).collect::<Vec<_>>());
            let (db, dc) = (up(&enc(double, &bvals)), up(&enc(double, &c0)));
            let bsrmm: Bsrmm = api.f(if double { "cusparseDbsrmm" } else { "cusparseSbsrmm" });
            assert_eq!(unsafe { bsrmm(h, dir, 0, 1, mb as i32, n as i32, kb as i32, nnzb as i32, alpha.as_ptr().cast(), descr, dv, dp, di, bd as i32, db as P, ldb as i32, beta.as_ptr().cast(), dc as P, m as i32) }, 0);
            let got = dec(double, &down(dc, m * n * sz));
            let mut want = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    want[i + j * m] = 0.5 * (0..k).map(|l| dense[i * k + l] * bvals[j + l * ldb]).sum::<f64>() + 2.0 * c0[i + j * m];
                }
            }
            let e1 = max_rel(&got, &want);
            // bsrmv
            let x = round(double, &(0..k).map(|_| rng.val()).collect::<Vec<_>>());
            let y0 = round(double, &(0..m).map(|_| rng.val()).collect::<Vec<_>>());
            let (dx, dy) = (up(&enc(double, &x)), up(&enc(double, &y0)));
            let bsrmv: Bsrmv = api.f(if double { "cusparseDbsrmv" } else { "cusparseSbsrmv" });
            assert_eq!(unsafe { bsrmv(h, dir, 0, mb as i32, kb as i32, nnzb as i32, alpha.as_ptr().cast(), descr, dv, dp, di, bd as i32, dx as P, beta.as_ptr().cast(), dy as P) }, 0);
            let got = dec(double, &down(dy, m * sz));
            let want: Vec<f64> = (0..m).map(|i| 0.5 * (0..k).map(|l| dense[i * k + l] * x[l]).sum::<f64>() + 2.0 * y0[i]).collect();
            let e2 = max_rel(&got, &want);
            println!("  bsrmm/bsrmv double={double} dir={dir} bd={bd} base={base}: rel err {e1:.2e} / {e2:.2e}");
            let tol = if double { 1e-12 } else { 1e-5 };
            assert!(e1 < tol && e2 < tol);
            cases += 2;
        }
        // transA != N is not supported, as in cuSPARSE
        let bsrmv: Bsrmv = api.f("cusparseSbsrmv");
        let one = 1.0f32;
        assert_eq!(unsafe { bsrmv(h, 0, 1, 1, 1, 0, (&one as *const f32).cast(), descr, null_mut(), null_mut(), null_mut(), 1, null_mut(), (&one as *const f32).cast(), null_mut()) }, 10);

        let coo2csr: unsafe extern "system" fn(P, P, i32, i32, P, i32) -> i32 = api.f("cusparseXcoo2csr");
        let csr2coo: unsafe extern "system" fn(P, P, i32, i32, P, i32) -> i32 = api.f("cusparseXcsr2coo");
        let ident: unsafe extern "system" fn(P, i32, P) -> i32 = api.f("cusparseCreateIdentityPermutation");
        for base in [0i64, 1] {
            let m = 1000usize;
            let (ptr, _, _) = random_csr(&mut rng, m, 50, 4, false);
            let nnz = *ptr.last().unwrap() as usize;
            let mut rows = Vec::new();
            for r in 0..m {
                for _ in ptr[r]..ptr[r + 1] {
                    rows.push(r as i64);
                }
            }
            let dcoo = up(&enc_idx(&rows, false, base));
            let dcsr = up(&vec![0u8; (m + 1) * 4]);
            assert_eq!(unsafe { coo2csr(h, dcoo as P, nnz as i32, m as i32, dcsr as P, base as i32) }, 0);
            let got: Vec<i64> = down(dcsr, (m + 1) * 4).chunks(4).map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap()))).collect();
            assert_eq!(got, ptr.iter().map(|p| p + base).collect::<Vec<_>>(), "coo2csr base {base}");
            let dback = up(&vec![0u8; nnz * 4]);
            assert_eq!(unsafe { csr2coo(h, dcsr as P, nnz as i32, m as i32, dback as P, base as i32) }, 0);
            let got: Vec<i64> = down(dback, nnz * 4).chunks(4).map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap()))).collect();
            assert_eq!(got, rows.iter().map(|r| r + base).collect::<Vec<_>>(), "csr2coo base {base}");
            cases += 2;
        }
        let dp = up(&vec![0u8; 777 * 4]);
        assert_eq!(unsafe { ident(h, 777, dp as P) }, 0);
        let got: Vec<i32> = down(dp, 777 * 4).chunks(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect();
        assert_eq!(got, (0..777).collect::<Vec<_>>());
        cases += 1;
        println!("  bsr, coo2csr, csr2coo, identity: ok");
    }

    // --- SpGEMM (C = alpha A B) through the three-phase generic API, and csrgeam2
    {
        type Work = unsafe extern "system" fn(P, i32, i32, *const c_void, P, P, *const c_void, P, i32, i32, P, *mut usize, P) -> i32;
        type Copy = unsafe extern "system" fn(P, i32, i32, *const c_void, P, P, *const c_void, P, i32, i32, P) -> i32;
        let gcreate: unsafe extern "system" fn(*mut P) -> i32 = api.f("cusparseSpGEMM_createDescr");
        let gdestroy: Destroy = api.f("cusparseSpGEMM_destroyDescr");
        let work: Work = api.f("cusparseSpGEMM_workEstimation");
        let compute: Work = api.f("cusparseSpGEMM_compute");
        let copy: Copy = api.f("cusparseSpGEMM_copy");
        let get_size: unsafe extern "system" fn(P, *mut i64, *mut i64, *mut i64) -> i32 = api.f("cusparseSpMatGetSize");
        let set_ptrs: unsafe extern "system" fn(P, P, P, P) -> i32 = api.f("cusparseCsrSetPointers");
        let dense = |rows: usize, cols: usize, ptr: &[i64], idx: &[i64], val: &[f64]| {
            let mut d = vec![0.0; rows * cols];
            for r in 0..rows {
                for e in ptr[r] as usize..ptr[r + 1] as usize {
                    d[r * cols + idx[e] as usize] += val[e];
                }
            }
            d
        };
        let dec_idx = |b: &[u8], wide: bool, base: i64| -> Vec<i64> {
            if wide { b.chunks(8).map(|c| i64::from_le_bytes(c.try_into().unwrap()) - base).collect() } else { b.chunks(4).map(|c| i64::from(i32::from_le_bytes(c.try_into().unwrap())) - base).collect() }
        };
        for double in [false, true] {
            for (m, k, n, per_row, wide, base) in [(200usize, 150usize, 120usize, 6usize, false, 0i64), (64, 300, 80, 20, true, 1), (1000, 1000, 1000, 3, true, 0), (37, 29, 41, 4, false, 1)] {
                let vt = if double { 1 } else { 0 };
                let it = if wide { 3 } else { 2 };
                let (ap, ai, av) = random_csr(&mut rng, m, k, per_row, m < 100);
                let (bp, bi, bv) = random_csr(&mut rng, k, n, per_row, false);
                let (av, bv) = (round(double, &av), round(double, &bv));
                let (mut a, mut b, mut c, mut g) = (null_mut(), null_mut(), null_mut(), null_mut());
                assert_eq!(unsafe { create_csr(&mut a, m as i64, k as i64, ai.len() as i64, up(&enc_idx(&ap, wide, base)) as P, up(&enc_idx(&ai, wide, base)) as P, up(&enc(double, &av)) as P, it, it, base as i32, vt) }, 0);
                assert_eq!(unsafe { create_csr(&mut b, k as i64, n as i64, bi.len() as i64, up(&enc_idx(&bp, wide, base)) as P, up(&enc_idx(&bi, wide, base)) as P, up(&enc(double, &bv)) as P, it, it, base as i32, vt) }, 0);
                assert_eq!(unsafe { create_csr(&mut c, m as i64, n as i64, 0, null_mut(), null_mut(), null_mut(), it, it, base as i32, vt) }, 0);
                assert_eq!(unsafe { gcreate(&mut g) }, 0);
                let (alpha, beta) = (1.25f64, 0.0f64);
                let (a32, b32) = (alpha as f32, beta as f32);
                let (pa, pb): (*const c_void, *const c_void) = if double { ((&alpha as *const f64).cast(), (&beta as *const f64).cast()) } else { ((&a32 as *const f32).cast(), (&b32 as *const f32).cast()) };
                let (mut s1, mut s2) = (0usize, 0usize);
                assert_eq!(unsafe { work(h, 0, 0, pa, a, b, pb, c, vt, 0, g, &mut s1, null_mut()) }, 0);
                assert_eq!(s1, (m + 1) * 8, "buffer 1 size");
                let b1 = up(&vec![0u8; s1]);
                assert_eq!(unsafe { work(h, 0, 0, pa, a, b, pb, c, vt, 0, g, &mut s1, b1 as P) }, 0);
                assert_eq!(unsafe { compute(h, 0, 0, pa, a, b, pb, c, vt, 0, g, &mut s2, null_mut()) }, 0);
                assert!(s2 >= m * 8);
                let b2 = up(&vec![0u8; s2]);
                assert_eq!(unsafe { compute(h, 0, 0, pa, a, b, pb, c, vt, 0, g, &mut s2, b2 as P) }, 0);
                let (mut rows, mut cols, mut nnz) = (0i64, 0i64, 0i64);
                assert_eq!(unsafe { get_size(c, &mut rows, &mut cols, &mut nnz) }, 0);
                assert_eq!((rows, cols), (m as i64, n as i64));
                let iw = if wide { 8 } else { 4 };
                let es = if double { 8 } else { 4 };
                let (cp, ci, cv) = (up(&vec![0u8; (m + 1) * iw]), up(&vec![0u8; nnz.max(1) as usize * iw]), up(&vec![0u8; nnz.max(1) as usize * es]));
                assert_eq!(unsafe { set_ptrs(c, cp as P, ci as P, cv as P) }, 0);
                assert_eq!(unsafe { copy(h, 0, 0, pa, a, b, pb, c, vt, 0, g) }, 0);
                ok!(cuStreamSynchronize(stream));
                let cpv = dec_idx(&down(cp, (m + 1) * iw), wide, base);
                let civ = dec_idx(&down(ci, nnz as usize * iw), wide, base);
                let cvv = dec(double, &down(cv, nnz as usize * es));
                assert_eq!(cpv[m], nnz);
                for r in 0..m {
                    assert!(civ[cpv[r] as usize..cpv[r + 1] as usize].windows(2).all(|w| w[0] < w[1]), "C row {r} sorted");
                }
                let (da, db) = (dense(m, k, &ap, &ai, &av), dense(k, n, &bp, &bi, &bv));
                let mut want = vec![0.0; m * n];
                for i in 0..m {
                    for l in 0..k {
                        let x = da[i * k + l];
                        if x != 0.0 {
                            for j in 0..n {
                                want[i * n + j] += alpha * x * db[l * n + j];
                            }
                        }
                    }
                }
                let e = max_rel(&dense(m, n, &cpv, &civ, &cvv), &want);
                let tol = if double { 1e-13 } else { 1e-5 };
                println!("  SpGEMM {m}x{k}x{n} f{} idx{} base {base}: nnz(C) {nnz}, rel err {e:.2e}", if double { 64 } else { 32 }, if wide { 64 } else { 32 });
                assert!(e < tol, "SpGEMM {m}x{k}x{n} double={double}: {e}");
                cases += 1;
                unsafe {
                    gdestroy(g);
                    destroy_sp(a);
                    destroy_sp(b);
                    destroy_sp(c);
                }
            }
            // csrgeam2 (32-bit legacy API): C = alpha A + beta B, sizes from bufferSizeExt
            let t = if double { "D" } else { "S" };
            type GeamSize = unsafe extern "system" fn(P, i32, i32, *const c_void, P, i32, P, P, P, *const c_void, P, i32, P, P, P, P, P, P, P, *mut usize) -> i32;
            type GeamNnz = unsafe extern "system" fn(P, i32, i32, P, i32, P, P, P, i32, P, P, P, P, *mut i32, P) -> i32;
            type Geam = unsafe extern "system" fn(P, i32, i32, *const c_void, P, i32, P, P, P, *const c_void, P, i32, P, P, P, P, P, P, P, P) -> i32;
            let gsize: GeamSize = api.f(&format!("cusparse{t}csrgeam2_bufferSizeExt"));
            let gnnz: GeamNnz = api.f("cusparseXcsrgeam2Nnz");
            let geam: Geam = api.f(&format!("cusparse{t}csrgeam2"));
            let create_descr: unsafe extern "system" fn(*mut P) -> i32 = api.f("cusparseCreateMatDescr");
            let set_base: unsafe extern "system" fn(P, i32) -> i32 = api.f("cusparseSetMatIndexBase");
            for (m, n, base) in [(300usize, 200usize, 0i64), (50, 70, 1)] {
                let (ap, ai, av) = random_csr(&mut rng, m, n, 8, false);
                let (bp, bi, bv) = random_csr(&mut rng, m, n, 5, true);
                let (av, bv) = (round(double, &av), round(double, &bv));
                let mut descr = null_mut();
                assert_eq!(unsafe { create_descr(&mut descr) }, 0);
                assert_eq!(unsafe { set_base(descr, base as i32) }, 0);
                let (dap, dai, dav) = (up(&enc_idx(&ap, false, base)) as P, up(&enc_idx(&ai, false, base)) as P, up(&enc(double, &av)) as P);
                let (dbp, dbi, dbv) = (up(&enc_idx(&bp, false, base)) as P, up(&enc_idx(&bi, false, base)) as P, up(&enc(double, &bv)) as P);
                let (alpha, beta) = (0.75f64, -1.5f64);
                let (a32, b32) = (alpha as f32, beta as f32);
                let (pa, pb): (*const c_void, *const c_void) = if double { ((&alpha as *const f64).cast(), (&beta as *const f64).cast()) } else { ((&a32 as *const f32).cast(), (&b32 as *const f32).cast()) };
                let dcp = up(&vec![0u8; (m + 1) * 4]) as P;
                let mut size = 0usize;
                assert_eq!(unsafe { gsize(h, m as i32, n as i32, pa, descr, ai.len() as i32, dav, dap, dai, pb, descr, bi.len() as i32, dbv, dbp, dbi, descr, null_mut(), dcp, null_mut(), &mut size) }, 0);
                assert_eq!(size, m * 8, "csrgeam2 buffer size");
                let buf = up(&vec![0u8; size]) as P;
                let mut nnz = 0i32;
                assert_eq!(unsafe { gnnz(h, m as i32, n as i32, descr, ai.len() as i32, dap, dai, descr, bi.len() as i32, dbp, dbi, descr, dcp, &mut nnz, buf) }, 0);
                let es = if double { 8 } else { 4 };
                let (dci, dcv) = (up(&vec![0u8; nnz.max(1) as usize * 4]) as P, up(&vec![0u8; nnz.max(1) as usize * es]) as P);
                assert_eq!(unsafe { geam(h, m as i32, n as i32, pa, descr, ai.len() as i32, dav, dap, dai, pb, descr, bi.len() as i32, dbv, dbp, dbi, descr, dcv, dcp, dci, buf) }, 0);
                ok!(cuStreamSynchronize(stream));
                let cpv = dec_idx(&down(dcp as u64, (m + 1) * 4), false, base);
                let civ = dec_idx(&down(dci as u64, nnz as usize * 4), false, base);
                let cvv = dec(double, &down(dcv as u64, nnz as usize * es));
                let want: Vec<f64> = dense(m, n, &ap, &ai, &av).iter().zip(dense(m, n, &bp, &bi, &bv)).map(|(x, y)| alpha * x + beta * y).collect();
                let e = max_rel(&dense(m, n, &cpv, &civ, &cvv), &want);
                println!("  {t}csrgeam2 {m}x{n} base {base}: nnz(C) {nnz}, rel err {e:.2e}");
                assert!(e < if double { 1e-14 } else { 1e-6 }, "{t}csrgeam2: {e}");
                cases += 1;
            }
        }
    }

    // --- f16 values: NOT_SUPPORTED, not a crash
    {
        let mut a = null_mut();
        let d = up(&[0u8; 64]) as P;
        assert_eq!(unsafe { create_csr(&mut a, 2, 2, 1, d, d, d, 2, 2, 0, 2) }, 0);
        let (mut vx, mut vy) = (null_mut(), null_mut());
        assert_eq!(unsafe { create_vec(&mut vx, 2, d, 2) }, 0);
        assert_eq!(unsafe { create_vec(&mut vy, 2, d, 2) }, 0);
        let one = 1.0f32;
        assert_eq!(unsafe { spmv(h, 0, (&one as *const f32).cast(), a, vx, (&one as *const f32).cast(), vy, 0, 0, null_mut()) }, 10);
        unsafe {
            destroy_vec(vx);
            destroy_vec(vy);
            destroy_sp(a);
        }
    }

    println!("  hw_cusparse: {cases} cases passed");
    assert_eq!(unsafe { destroy_h(h) }, 0);
    for d in allocs.borrow().iter() {
        ok!(cuMemFree_v2(*d));
    }
    ok!(cuStreamDestroy_v2(stream));
    ok!(cuCtxDestroy_v2(ctx));
}
