//! CUDA C++ -> (nvrtc shim, clang NVPTX) -> PTX -> translator -> SPIR-V simulator, checked against
//! Rust references. Skips when no clang with NVPTX is installed (set `NVRTC_SHIM_CLANG`).
use super::ggml_conformance::{Arg, Kernel};
use super::sim::Gpu;
use super::*;
use nvrtc64_120_0::compiler::{self, Request};

struct Compiled {
    ptx: String,
    prepared: String,
    lowered: std::collections::BTreeMap<String, String>,
}

fn compile(src: &str, names: &[&str]) -> Option<Compiled> {
    if compiler::find_clang().is_err() {
        eprintln!("SKIP nvrtc_tests: no clang with NVPTX");
        return None;
    }
    let req = Request {
        source: src.into(),
        name: "k.cu".into(),
        name_expressions: names.iter().map(|s| (*s).to_owned()).collect(),
        ..Default::default()
    };
    let out = compiler::compile(&req).unwrap_or_else(|e| panic!("nvrtc compile failed:\n{}", e.log()));
    let prepared = prepare_source(&out.ptx).unwrap_or_else(|e| panic!("prepare: {e}\n{}", out.ptx));
    Some(Compiled { ptx: out.ptx, prepared, lowered: out.lowered })
}

fn load(c: &Compiled, entry: &str) -> Kernel {
    let t = translate_entry(&c.prepared, entry).unwrap_or_else(|e| panic!("translate {entry}: {e}\n{}", c.ptx));
    super::coverage_tests::validate(&t.words);
    let module = rspirv::dr::load_words(&t.words).unwrap();
    let mut t = t;
    Kernel { meta: t.kernels.remove(0), module, words: t.words, has_dot_op: false }
}

fn f32s(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u32s(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn down_u32(g: &Gpu, a: u64, n: usize) -> Vec<u32> {
    g.download(a, n * 4).chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

struct Rng(u64);
impl Rng {
    fn f(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
    }
}

#[test]
fn vector_add() {
    let Some(c) = compile(include_str!("../../crates/nvrtc_shim/tests/kernels/vadd.cu"), &[]) else { return };
    let k = load(&c, "vadd");
    let n = 1000usize;
    let mut r = Rng(1);
    let a: Vec<f32> = (0..n).map(|_| r.f()).collect();
    let b: Vec<f32> = (0..n).map(|_| r.f()).collect();
    let mut g = Gpu::default();
    let (pa, pb, pc) = (g.upload(&f32s(&a)), g.upload(&f32s(&b)), g.alloc(n * 4));
    k.run(&mut g, [4, 1, 1], [256, 1, 1], &[Arg::P(pa), Arg::P(pb), Arg::P(pc), Arg::U(n as u32)]).unwrap();
    let got = g.download_f32(pc, n);
    for i in 0..n {
        assert_eq!(got[i], a[i] + b[i], "i={i}");
    }
}

#[test]
fn shared_memory_reduction() {
    let Some(c) = compile(include_str!("../../crates/nvrtc_shim/tests/kernels/reduce.cu"), &[]) else { return };
    let k = load(&c, "block_sum");
    let (n, bs) = (500usize, 128usize);
    let blocks = n.div_ceil(bs);
    let mut r = Rng(2);
    let x: Vec<f32> = (0..n).map(|_| r.f()).collect();
    let mut g = Gpu::default();
    let (pi, po) = (g.upload(&f32s(&x)), g.alloc(blocks * 4));
    k.run(&mut g, [blocks as u32, 1, 1], [bs as u32, 1, 1], &[Arg::P(pi), Arg::P(po), Arg::U(n as u32)]).unwrap();
    let got = g.download_f32(po, blocks);
    for b in 0..blocks {
        let want: f32 = x[b * bs..((b + 1) * bs).min(n)].iter().sum();
        assert!((got[b] - want).abs() < 1e-3, "block {b}: {} vs {want}", got[b]);
    }
}

#[test]
fn warp_shuffle_vote() {
    let Some(c) = compile(include_str!("../../crates/nvrtc_shim/tests/kernels/warp.cu"), &[]) else { return };
    let k = load(&c, "warp_ops");
    let mut r = Rng(3);
    let x: Vec<f32> = (0..64).map(|_| r.f()).collect();
    let mut g = Gpu::default();
    let (pi, ps, pm, pv) = (g.upload(&f32s(&x)), g.alloc(8), g.alloc(8), g.alloc(8));
    k.run(&mut g, [1, 1, 1], [64, 1, 1], &[Arg::P(pi), Arg::P(ps), Arg::P(pm), Arg::P(pv)]).unwrap();
    let (s, m, v) = (g.download_f32(ps, 2), g.download_f32(pm, 2), down_u32(&g, pv, 2));
    for w in 0..2 {
        let lane = &x[w * 32..(w + 1) * 32];
        let want_s: f32 = lane.iter().sum();
        let want_m = lane.iter().cloned().fold(f32::MIN, f32::max);
        assert!((s[w] - want_s).abs() < 1e-3, "sum {w}");
        assert_eq!(m[w], want_m, "max {w}");
        assert_eq!(v[w], lane.iter().filter(|&&q| q > 0.0).count() as u32, "votes {w}");
    }
}

#[test]
fn atomics_histogram_and_float_sum() {
    let Some(c) = compile(include_str!("../../crates/nvrtc_shim/tests/kernels/atomics.cu"), &[]) else { return };
    let k = load(&c, "atomics");
    let n = 200usize;
    let keys: Vec<u32> = (0..n as u32).map(|i| (i * 7 + 3) % 13).collect();
    let vals: Vec<f32> = (0..n).map(|i| (i % 5) as f32 * 0.5).collect();
    let mut g = Gpu::default();
    let (pk, pv, ph, pt) = (g.upload(&u32s(&keys)), g.upload(&f32s(&vals)), g.alloc(32), g.alloc(4));
    k.run(&mut g, [2, 1, 1], [128, 1, 1], &[Arg::P(pk), Arg::P(pv), Arg::P(ph), Arg::P(pt), Arg::U(n as u32)]).unwrap();
    let mut want = [0u32; 8];
    for key in &keys {
        want[(key & 7) as usize] += 1;
    }
    assert_eq!(down_u32(&g, ph, 8), want);
    let total: f32 = vals.iter().sum();
    assert!((g.download_f32(pt, 1)[0] - total).abs() < 1e-3);
}

#[test]
fn template_instantiation_via_name_expressions() {
    let src = include_str!("../../crates/nvrtc_shim/tests/kernels/templ.cu");
    let Some(c) = compile(src, &["axpy<float>", "axpy<int>", "ns::scale<3>"]) else { return };
    let n = 70usize;
    let x: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let y: Vec<f32> = (0..n).map(|i| 1.0 + i as f32 * 0.25).collect();
    let mut g = Gpu::default();
    let (px, py) = (g.upload(&f32s(&x)), g.upload(&f32s(&y)));
    load(&c, &c.lowered["axpy<float>"])
        .run(&mut g, [1, 1, 1], [128, 1, 1], &[Arg::F(2.5), Arg::P(px), Arg::P(py), Arg::U(n as u32)])
        .unwrap();
    for (i, v) in g.download_f32(py, n).iter().enumerate() {
        assert_eq!(*v, 2.5 * x[i] + y[i], "axpy<float> i={i}");
    }
    let xi: Vec<u32> = (0..n as u32).collect();
    let yi: Vec<u32> = (0..n as u32).map(|i| i * 3).collect();
    let (pxi, pyi) = (g.upload(&u32s(&xi)), g.upload(&u32s(&yi)));
    load(&c, &c.lowered["axpy<int>"])
        .run(&mut g, [1, 1, 1], [128, 1, 1], &[Arg::U(4), Arg::P(pxi), Arg::P(pyi), Arg::U(n as u32)])
        .unwrap();
    for (i, v) in down_u32(&g, pyi, n).iter().enumerate() {
        assert_eq!(*v, 4 * xi[i] + yi[i], "axpy<int> i={i}");
    }
    let ps = g.upload(&f32s(&x));
    load(&c, &c.lowered["ns::scale<3>"]).run(&mut g, [1, 1, 1], [128, 1, 1], &[Arg::P(ps), Arg::U(n as u32)]).unwrap();
    for (i, v) in g.download_f32(ps, n).iter().enumerate() {
        assert_eq!(*v, 3.0 * x[i]);
    }
}

fn f16_bits(x: f32) -> u16 {
    // round-to-nearest-even f32 -> f16 for test data (values are small and normal)
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32 - 127 + 15;
    let man = b & 0x7f_ffff;
    if x == 0.0 {
        return sign;
    }
    let mut h = ((exp as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (h & 1) == 1) {
        h += 1;
    }
    sign | h as u16
}

#[test]
fn half_math() {
    let Some(c) = compile(include_str!("../../crates/nvrtc_shim/tests/kernels/half.cu"), &[]) else { return };
    let k = load(&c, "half_ops");
    let n = 40usize;
    let mut r = Rng(5);
    let mk = |r: &mut Rng| -> Vec<u16> { (0..n).map(|_| f16_bits(r.f())).collect() };
    let (x, w, b) = (mk(&mut r), mk(&mut r), mk(&mut r));
    let bytes = |v: &[u16]| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let mut g = Gpu::default();
    let (px, pw, pb, py, pe) = (g.upload(&bytes(&x)), g.upload(&bytes(&w)), g.upload(&bytes(&b)), g.alloc(n * 2), g.alloc(n * 4));
    k.run(&mut g, [1, 1, 1], [64, 1, 1], &[Arg::P(px), Arg::P(pw), Arg::P(pb), Arg::P(py), Arg::P(pe), Arg::U(n as u32)]).unwrap();
    let y = g.download(py, n * 2);
    let e = g.download_f32(pe, n);
    let to_f = |h: u16| super::sim::f16_to_f32(u32::from(h));
    for i in 0..n {
        let want = f16_bits(to_f(x[i]).mul_add(to_f(w[i]), to_f(b[i])));
        let got = u16::from_le_bytes([y[2 * i], y[2 * i + 1]]);
        assert!((i32::from(got) - i32::from(want)).abs() <= 1, "hfma i={i}: {got:#x} vs {want:#x}");
        let we = to_f(x[i]).exp();
        assert!((e[i] - we).abs() <= we * 1e-5, "exp i={i}: {} vs {we}", e[i]);
    }
}

#[test]
fn cooperative_groups_tile_reduce() {
    let Some(c) = compile(include_str!("../../crates/nvrtc_shim/tests/kernels/cg.cu"), &[]) else { return };
    let k = load(&c, "cg_sum");
    let mut r = Rng(6);
    let x: Vec<f32> = (0..64).map(|_| r.f()).collect();
    let mut g = Gpu::default();
    let (pi, po) = (g.upload(&f32s(&x)), g.alloc(8));
    k.run(&mut g, [1, 1, 1], [64, 1, 1], &[Arg::P(pi), Arg::P(po)]).unwrap();
    let got = g.download_f32(po, 2);
    for w in 0..2 {
        let want: f32 = x[w * 32..(w + 1) * 32].iter().sum();
        assert!((got[w] - want).abs() < 1e-3);
    }
}

#[test]
fn math_functions_within_tolerance() {
    let Some(c) = compile(include_str!("../../crates/nvrtc_shim/tests/kernels/mathfn.cu"), &[]) else { return };
    let k = load(&c, "mathfn");
    let n = 33usize;
    let x: Vec<f32> = (0..n).map(|i| 0.05 + i as f32 * 0.37).collect();
    let mut g = Gpu::default();
    let (pi, po) = (g.upload(&f32s(&x)), g.alloc(n * 20));
    k.run(&mut g, [1, 1, 1], [64, 1, 1], &[Arg::P(pi), Arg::P(po), Arg::U(n as u32)]).unwrap();
    let o = g.download_f32(po, n * 5);
    for i in 0..n {
        let v = x[i];
        let want = [v.sqrt(), (-v).exp(), v.ln(), v.sin(), v.tanh()];
        for (j, w) in want.iter().enumerate() {
            let tol = 2e-5 * w.abs().max(1.0);
            assert!((o[5 * i + j] - w).abs() <= tol, "fn {j} x={v}: {} vs {w}", o[5 * i + j]);
        }
    }
}
