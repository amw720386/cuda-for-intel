//! cuFFT shim kernels: translation, spirv-val, and the full gather/stage/scatter pipeline
//! in the SPIR-V simulator against a naive DFT (`crates/cufft_shim/src/{plan,ptxgen}.rs`).
use super::cufft_plan::{self as plan, C, Desc, FftType};
use super::cufft_ptxgen as pg;
use super::sim::{Gpu, Launch, Sim};
use super::*;

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
fn cufft_kernels_translate_and_validate() {
    for (name, src, n) in pg::all_entries() {
        let t = translate(&src).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(t.kernels[0].name, name);
        assert_eq!(t.kernels[0].params.len(), n, "{name}");
        spirv_val(&t.words, &name);
    }
    // the three-entry module the shim loads
    for double in [false, true] {
        let t = translate(&pg::module(double)).unwrap();
        assert_eq!(t.kernels.len(), 5);
    }
}

struct Kernels {
    sims: Vec<(String, Sim)>,
}

impl Kernels {
    fn new(double: bool) -> Self {
        let sims = pg::all_entries()
            .into_iter()
            .filter(|(n, _, _)| n.ends_with(pg::suffix(double)))
            .map(|(name, src, _)| {
                let t = translate(&src).unwrap();
                let module = rspirv::dr::load_words(&t.words).unwrap();
                (name, Sim::new(&module))
            })
            .collect();
        Kernels { sims }
    }
    fn run(&mut self, idx: usize, gpu: &mut Gpu, total: u64, args: Vec<u64>) {
        self.run_groups(idx, gpu, (total as u32).div_ceil(64), 64, args);
    }
    fn run_groups(&mut self, idx: usize, gpu: &mut Gpu, grid: u32, block: u32, args: Vec<u64>) {
        let (name, sim) = &mut self.sims[idx];
        sim.run(gpu, &Launch::new([grid, 1, 1], [block, 1, 1], args)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert!(gpu.violations.borrow().is_empty(), "{name}: {:?}", gpu.violations.borrow());
    }
}

fn enc(double: bool, v: &[f64]) -> Vec<u8> {
    if double { v.iter().flat_map(|x| x.to_le_bytes()).collect() } else { v.iter().flat_map(|x| (*x as f32).to_le_bytes()).collect() }
}

fn dec(double: bool, b: &[u8]) -> Vec<f64> {
    if double {
        b.chunks(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect()
    } else {
        b.chunks(4).map(|c| f64::from(f32::from_le_bytes(c.try_into().unwrap()))).collect()
    }
}

/// Runs the device pipeline in the simulator. `input` holds scalars; returns `out_scalars`
/// scalars of the output buffer (the input buffer itself when `in_place`).
fn run_sim(d: &Desc, input: &[f64], out_scalars: usize, inverse: bool, in_place: bool) -> Vec<f64> {
    let dbl = d.ty.double;
    let rs = d.ty.real_size() as usize;
    let mut k = Kernels::new(dbl);
    let mut gpu = Gpu::default();
    let mut ibytes = enc(dbl, input);
    if in_place {
        ibytes.resize(ibytes.len().max(out_scalars * rs), 0);
    }
    let src = gpu.upload(&ibytes);
    let dst = if in_place { src } else { gpu.alloc(out_scalars * rs) };
    let tw = gpu.upload(&d.twiddle_bytes());
    let work = gpu.alloc(d.work_size() as usize);
    let (is, os) = d.exec_strides(in_place);
    if !d.direct {
        k.run(0, &mut gpu, d.total(), d.gather_args(src, work, &is));
    }
    for p in &d.passes {
        let (a, b, inv) = (d.buf(work, p.src, src, dst), d.buf(work, p.dst, src, dst), p.dir.inverse(inverse));
        match &p.op {
            plan::Op::Stage(s) => k.run(1, &mut gpu, p.total, d.stage_args(s, p.total, a, b, tw, inv)),
            plan::Op::Local(l) => k.run_groups(3, &mut gpu, Desc::local_groups(l), pg::LOCAL_BLOCK, d.local_args(l, a, b, tw, inv)),
            plan::Op::Chirp(c) => k.run(4, &mut gpu, p.total, d.chirp_args(c, p.total, a, b, tw, inverse)),
        }
    }
    if !d.direct {
        k.run(2, &mut gpu, d.out_total(), d.scatter_args(d.buf(work, d.result, 0, 0), dst, &os));
    }
    dec(dbl, &gpu.download(dst, out_scalars * rs))
}

fn ty(code: i32) -> FftType {
    FftType::from_cufft(code).unwrap()
}

fn sample(len: usize, seed: u64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 2001) as f64 / 1000.0 - 1.0
        })
        .collect()
}

fn pairs(v: &[f64]) -> Vec<C> {
    v.chunks(2).map(|c| (c[0], c[1])).collect()
}

fn rel_err(got: &[C], want: &[C]) -> f64 {
    let scale = want.iter().map(|v| v.0.hypot(v.1)).fold(1e-30, f64::max);
    got.iter().zip(want).map(|(x, y)| (x.0 - y.0).hypot(x.1 - y.1)).fold(0.0, f64::max) / scale
}

#[test]
fn sim_c2c_z2z_sizes_and_batches() {
    // power-of-two, mixed radix (3*5*7), prime (direct-DFT stage), 2D and 3D, batched
    let cases: &[(&[i64], i64, i32, f64)] = &[
        (&[16], 3, 0x29, 2e-6),
        (&[105], 1, 0x29, 5e-6),
        (&[13], 2, 0x29, 2e-6),
        (&[6, 8], 2, 0x29, 2e-6),
        (&[2, 3, 4], 1, 0x29, 2e-6),
        (&[32], 2, 0x69, 1e-13),
        (&[45], 1, 0x69, 1e-13),
        (&[4, 5, 3], 1, 0x69, 1e-13),
        // Bluestein (prime 67 > 63, also along a non-last dimension) and four-step (f64 cap 1024)
        (&[67], 2, 0x29, 1e-5),
        (&[67, 3], 1, 0x69, 1e-12),
        (&[2048], 1, 0x69, 1e-12),
    ];
    for &(n, batch, code, tol) in cases {
        let d = plan::describe(n.len() as i64, n, None, 0, 0, None, 0, 0, ty(code), batch).unwrap();
        let total = d.total() as usize;
        let mut x = sample(2 * total, total as u64);
        if code == 0x29 {
            x = x.iter().map(|v| f64::from(*v as f32)).collect();
        }
        for inverse in [false, true] {
            let got = run_sim(&d, &x, 2 * total, inverse, false);
            let want = plan::naive_dft(d.n.map(|v| v as usize), batch as usize, &pairs(&x), inverse);
            let e = rel_err(&pairs(&got), &want);
            assert!(e < tol, "{n:?} type {code:#x} batch {batch} inverse {inverse}: err {e}");
        }
    }
}

#[test]
fn sim_r2c_c2r_out_of_place_and_in_place() {
    for (code, icode, tol) in [(0x2a, 0x2c, 2e-6), (0x6a, 0x6c, 1e-13)] {
        for (n, batch) in [(&[12i64][..], 2i64), (&[9], 1), (&[4, 6], 2)] {
            let d = plan::describe(n.len() as i64, n, None, 0, 0, None, 0, 0, ty(code), batch).unwrap();
            let inv = plan::describe(n.len() as i64, n, None, 0, 0, None, 0, 0, ty(icode), batch).unwrap();
            let n3 = d.n.map(|v| v as usize);
            let total = d.total() as usize;
            let m2 = n3[2] / 2 + 1;
            let x: Vec<f64> = sample(total, 5).iter().map(|v| f64::from(*v as f32)).collect();
            let full = plan::naive_dft(n3, batch as usize, &x.iter().map(|&v| (v, 0.0)).collect::<Vec<_>>(), false);
            let want: Vec<C> = full.iter().enumerate().filter(|(i, _)| i % n3[2] < m2).map(|(_, v)| *v).collect();
            let y = run_sim(&d, &x, 2 * d.out_total() as usize, false, false);
            assert!(rel_err(&pairs(&y), &want) < tol, "R2C {n:?} {code:#x}");
            let z = run_sim(&inv, &y, total, true, false);
            let per = total as f64 / batch as f64;
            let e = z.iter().zip(&x).map(|(a, b)| (a / per - b).abs()).fold(0.0, f64::max);
            assert!(e < tol * 10.0, "C2R {n:?} {code:#x}: {e}");
            // in place: rows padded to 2 * m2 real scalars
            let rows = total / n3[2];
            let mut padded = vec![0.0; rows * 2 * m2];
            for r in 0..rows {
                padded[r * 2 * m2..r * 2 * m2 + n3[2]].copy_from_slice(&x[r * n3[2]..(r + 1) * n3[2]]);
            }
            let y2 = run_sim(&d, &padded, rows * 2 * m2, false, true);
            assert!(rel_err(&pairs(&y2), &want) < tol, "in-place R2C {n:?} {code:#x}");
            let z2 = run_sim(&inv, &y2, rows * 2 * m2, true, true);
            for r in 0..rows {
                for c in 0..n3[2] {
                    assert!((z2[r * 2 * m2 + c] / per - x[r * n3[2] + c]).abs() < tol * 10.0, "in-place C2R {n:?}");
                }
            }
        }
    }
}

#[test]
fn sim_advanced_layout_interleaved_batches() {
    // istride = batch, idist = 1 (as torch emits for a transform over a non-last dim)
    let (n, batch) = (10usize, 3usize);
    let d = plan::describe(1, &[n as i64], Some(&[n as i64]), batch as i64, 1, Some(&[n as i64]), 1, n as i64, ty(0x69), batch as i64).unwrap();
    let x = sample(2 * n * batch, 11);
    let got = pairs(&run_sim(&d, &x, 2 * n * batch, false, false));
    for b in 0..batch {
        let xb: Vec<C> = (0..n).map(|i| (x[2 * (i * batch + b)], x[2 * (i * batch + b) + 1])).collect();
        let want = plan::naive_dft([1, 1, n], 1, &xb, false);
        assert!(rel_err(&got[b * n..(b + 1) * n], &want) < 1e-13);
    }
}
