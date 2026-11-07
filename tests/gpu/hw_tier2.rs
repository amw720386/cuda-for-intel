//! Hardware validation of the tier-2 PTX paths: f64, f16/bf16, wmma, approximate math,
//! cp.async, redux/match/vote/elect and 64-bit/float atomics.
use super::hw_common::*;

const GUARD: &str = "\tmov.u32 %r0, %tid.x;\n\tmov.u32 %r1, %ctaid.x;\n\tmov.u32 %r2, %ntid.x;\n\tmad.lo.s32 %r3, %r1, %r2, %r0;\n\tsetp.ge.u32 %p15, %r3, %r100;\n\t@%p15 bra $done;\n";
const DECLS: &str = "\t.reg .f64 %fd<16>;\n\t.reg .f16 %h<8>;\n\t.reg .bf16 %hb<8>;\n\t.reg .b32 %q<16>;\n\t.reg .b64 %l<8>;\n";

/// Elementwise kernel: inputs `ins` = (ld op, size) loaded into the registers `names`,
/// `ops` run, then `store` writes `out_reg` with `st_op`. Scalar n is the last param.
fn ew(ins: &[(&str, u32, &str)], st: (&str, u32, &str), ops: &str) -> String {
    let mut s = prologue("ew_k", ins.len() + 1, 1, DECLS);
    s += GUARD;
    for (i, (ld, size, reg)) in ins.iter().enumerate() {
        s += &format!(
            "\tmul.wide.u32 %rd20, %r3, {size};\n\tadd.s64 %rd21, %rd{}, %rd20;\n\t{ld} {reg}, [%rd21];\n",
            i + 1
        );
    }
    s += ops;
    s += &format!(
        "\tmul.wide.u32 %rd20, %r3, {};\n\tadd.s64 %rd21, %rd{}, %rd20;\n\t{} [%rd21], {};\n$done:\n\tret;\n}}\n",
        st.1,
        ins.len() + 1,
        st.0,
        st.2
    );
    s
}

fn run_ew(ptx: &str, inputs: Vec<Vec<u8>>, out_bytes: usize, n: usize) -> Vec<u8> {
    let mut bufs = inputs;
    bufs.push(vec![0u8; out_bytes]);
    launch(ptx, "ew_k", n.div_ceil(128) as u32, 128, &mut bufs, &[n as u32])
        .unwrap_or_else(|e| panic!("{e}"));
    bufs.pop().unwrap()
}

fn b64(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}
fn u64s(b: &[u8]) -> Vec<u64> {
    b.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn f64_inputs(rng: &mut Rng, n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| match i % 8 {
            0 => f64::from_bits(((rng.u32() as u64) << 32) | rng.u32() as u64),
            1 => (rng.range(-100000, 100000) as f64) / 64.0,
            2 => [0.0, -0.0, 1.0, -1.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, f64::MAX, f64::MIN_POSITIVE, 5e-324][(rng.u32() % 10) as usize],
            _ => (rng.range(-1 << 20, 1 << 20) as f64) * 2f64.powi(rng.range(-40, 40)),
        })
        .collect()
}

fn same64(got: u64, want: f64) -> bool {
    if want.is_nan() { f64::from_bits(got).is_nan() } else { got == want.to_bits() }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_f64_arithmetic_matches_cpu() {
    let mut rng = Rng(0xf640_f640_f640_f640);
    let n = 1 << 14;
    let a = f64_inputs(&mut rng, n);
    let b = f64_inputs(&mut rng, n);
    let c = f64_inputs(&mut rng, n);
    type F2 = fn(f64, f64) -> f64;
    let bin: [(&str, F2); 8] = [
        ("add.rn.f64", |x, y| x + y),
        ("sub.rn.f64", |x, y| x - y),
        ("mul.rn.f64", |x, y| x * y),
        ("div.rn.f64", |x, y| x / y),
        ("min.f64", |x, y| if x.is_nan() { y } else if y.is_nan() { x } else if x == y && x == 0.0 { if x.is_sign_negative() { x } else { y } } else { x.min(y) }),
        ("max.f64", |x, y| if x.is_nan() { y } else if y.is_nan() { x } else if x == y && x == 0.0 { if x.is_sign_negative() { y } else { x } } else { x.max(y) }),
        ("add.f64", |x, y| x + y),
        ("mul.f64", |x, y| x * y),
    ];
    let mut report = vec![];
    for (op, f) in bin {
        let ptx = ew(&[("ld.global.b64", 8, "%fd1"), ("ld.global.b64", 8, "%fd2")], ("st.global.b64", 8, "%fd3"), &format!("\t{op} %fd3, %fd1, %fd2;\n"));
        let out = u64s(&run_ew(&ptx, vec![b64(&a.iter().map(|x| x.to_bits()).collect::<Vec<_>>()), b64(&b.iter().map(|x| x.to_bits()).collect::<Vec<_>>())], n * 8, n));
        let bad = (0..n).filter(|&i| !same64(out[i], f(a[i], b[i]))).count();
        report.push(format!("{op}: {bad} mismatches of {n}"));
        if let Some(i) = (0..n).find(|&i| !same64(out[i], f(a[i], b[i]))) {
            panic!("{op} #{i}: a={:016x} b={:016x} got={:016x} want={:016x}", a[i].to_bits(), b[i].to_bits(), out[i], f(a[i], b[i]).to_bits());
        }
    }
    // fma, sqrt, neg, abs
    let ptx = ew(&[("ld.global.b64", 8, "%fd1"), ("ld.global.b64", 8, "%fd2"), ("ld.global.b64", 8, "%fd3")], ("st.global.b64", 8, "%fd4"), "\tfma.rn.f64 %fd4, %fd1, %fd2, %fd3;\n");
    let bits = |v: &Vec<f64>| b64(&v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    let out = u64s(&run_ew(&ptx, vec![bits(&a), bits(&b), bits(&c)], n * 8, n));
    let bad = (0..n).filter(|&i| !same64(out[i], a[i].mul_add(b[i], c[i]))).count();
    report.push(format!("fma.rn.f64: {bad} mismatches of {n}"));
    for (op, f) in [("sqrt.rn.f64", f64::sqrt as fn(f64) -> f64), ("neg.f64", |x| -x), ("abs.f64", f64::abs)] {
        let ptx = ew(&[("ld.global.b64", 8, "%fd1")], ("st.global.b64", 8, "%fd2"), &format!("\t{op} %fd2, %fd1;\n"));
        let out = u64s(&run_ew(&ptx, vec![bits(&a)], n * 8, n));
        let bad = (0..n).filter(|&i| !same64(out[i], f(a[i]))).count();
        report.push(format!("{op}: {bad} mismatches of {n}"));
    }
    println!("{report:#?}");
    assert!(report.iter().all(|r| r.contains(": 0 mismatches")), "{report:#?}");
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_f64_conversions_match_cpu() {
    let mut rng = Rng(0xf64c_f64c_f64c_f64c);
    let n = 1 << 14;
    let a = f64_inputs(&mut rng, n);
    let bits: Vec<u8> = b64(&a.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    // f64 -> f32 (rn), f32 -> f64
    let ptx = ew(&[("ld.global.b64", 8, "%fd1")], ("st.global.b32", 4, "%f1"), "\tcvt.rn.f32.f64 %f1, %fd1;\n");
    let out = u32s_of(&run_ew(&ptx, vec![bits.clone()], n * 4, n));
    for i in 0..n {
        let want = a[i] as f32;
        assert!(if want.is_nan() { f32::from_bits(out[i]).is_nan() } else { out[i] == want.to_bits() }, "cvt.rn.f32.f64 {:016x}", a[i].to_bits());
    }
    let f32s: Vec<f32> = a.iter().map(|x| *x as f32).collect();
    let ptx = ew(&[("ld.global.b32", 4, "%f1")], ("st.global.b64", 8, "%fd1"), "\tcvt.f64.f32 %fd1, %f1;\n");
    let out = u64s(&run_ew(&ptx, vec![bytes_of_u32(&f32s.iter().map(|x| x.to_bits()).collect::<Vec<_>>())], n * 8, n));
    for i in 0..n {
        assert!(same64(out[i], f32s[i] as f64), "cvt.f64.f32 {:08x}", f32s[i].to_bits());
    }
    // int -> f64 and f64 -> int (saturating)
    let ints: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
    let ptx = ew(&[("ld.global.b32", 4, "%r10")], ("st.global.b64", 8, "%fd1"), "\tcvt.rn.f64.s32 %fd1, %r10;\n");
    let out = u64s(&run_ew(&ptx, vec![bytes_of_u32(&ints)], n * 8, n));
    for i in 0..n {
        assert_eq!(out[i], (ints[i] as i32 as f64).to_bits(), "cvt.rn.f64.s32");
    }
    let ptx = ew(&[("ld.global.b32", 4, "%r10")], ("st.global.b64", 8, "%fd1"), "\tcvt.rn.f64.u32 %fd1, %r10;\n");
    let out = u64s(&run_ew(&ptx, vec![bytes_of_u32(&ints)], n * 8, n));
    for i in 0..n {
        assert_eq!(out[i], (ints[i] as f64).to_bits(), "cvt.rn.f64.u32");
    }
    for (instr, f) in [("cvt.rzi.s32.f64", 0u8), ("cvt.rni.s32.f64", 1), ("cvt.rmi.s32.f64", 2), ("cvt.rpi.s32.f64", 3)] {
        let ptx = ew(&[("ld.global.b64", 8, "%fd1")], ("st.global.b32", 4, "%r11"), &format!("\t{instr} %r11, %fd1;\n"));
        let out = u32s_of(&run_ew(&ptx, vec![bits.clone()], n * 4, n));
        for i in 0..n {
            let x = a[i];
            let r = match f { 0 => x.trunc(), 1 => { let r = x.round(); if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r } }, 2 => x.floor(), _ => x.ceil() };
            let want = if x.is_nan() { 0 } else { r.clamp(-2147483648.0, 2147483647.0) as i32 };
            assert_eq!(out[i] as i32, want, "{instr} {x:e}");
        }
    }
}

// ---------------------------------------------------------------- f16 / bf16

fn bf16_of(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() { return 0x7fc0; }
    (((b as u64 + 0x7fff + ((b >> 16) & 1) as u64) >> 16) & 0xffff) as u16
}
fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_f16_and_bf16_scalar_arithmetic() {
    let mut rng = Rng(0x1616_b516_1616_b516);
    let n = 1 << 14;
    let small = |rng: &mut Rng| ((rng.range(-2048, 2048) as f32) / 64.0) * 2f32.powi(rng.range(-4, 4));
    let av: Vec<f32> = (0..n).map(|_| small(&mut rng)).collect();
    let bv: Vec<f32> = (0..n).map(|_| small(&mut rng)).collect();
    let pack = |v: &[f32], f: &dyn Fn(f32) -> u16| -> Vec<u8> { v.iter().flat_map(|x| f(*x).to_le_bytes()).collect() };
    for (ty, reg, enc, dec) in [
        ("f16", "%h", f32_to_f16 as fn(f32) -> u16, f16_to_f32 as fn(u16) -> f32),
        ("bf16", "%hb", bf16_of as fn(f32) -> u16, bf16_to_f32 as fn(u16) -> f32),
    ] {
        let ab = pack(&av, &enc);
        let bb = pack(&bv, &enc);
        let ha: Vec<f32> = ab.chunks_exact(2).map(|c| dec(u16::from_le_bytes([c[0], c[1]]))).collect();
        let hb: Vec<f32> = bb.chunks_exact(2).map(|c| dec(u16::from_le_bytes([c[0], c[1]]))).collect();
        for (op, f) in [
            ("add.rn", (|x, y| x + y) as fn(f32, f32) -> f32),
            ("sub.rn", |x, y| x - y),
            ("mul.rn", |x, y| x * y),
            ("min", |x, y| x.min(y)),
            ("max", |x, y| x.max(y)),
        ] {
            let ptx = ew(&[("ld.global.b16", 2, &format!("{reg}1")), ("ld.global.b16", 2, &format!("{reg}2"))], ("st.global.b16", 2, &format!("{reg}3")), &format!("\t{op}.{ty} {reg}3, {reg}1, {reg}2;\n"));
            let out = run_ew(&ptx, vec![ab.clone(), bb.clone()], n * 2, n);
            for i in 0..n {
                let got = u16::from_le_bytes([out[2 * i], out[2 * i + 1]]);
                let want = enc(f(ha[i], hb[i]));
                assert_eq!(got, want, "{op}.{ty} {} {}", ha[i], hb[i]);
            }
        }
        let ptx = ew(&[("ld.global.b16", 2, &format!("{reg}1")), ("ld.global.b16", 2, &format!("{reg}2")), ("ld.global.b16", 2, &format!("{reg}4"))], ("st.global.b16", 2, &format!("{reg}3")), &format!("\tfma.rn.{ty} {reg}3, {reg}1, {reg}2, {reg}4;\n"));
        let out = run_ew(&ptx, vec![ab.clone(), bb.clone(), ab.clone()], n * 2, n);
        let mut off_by_one = 0;
        for i in 0..n {
            let got = u16::from_le_bytes([out[2 * i], out[2 * i + 1]]);
            let want = enc(ha[i].mul_add(hb[i], ha[i]));
            if got != want {
                off_by_one += 1;
                assert!((got as i32 - want as i32).abs() <= 1, "fma.{ty}");
            }
        }
        println!("fma.rn.{ty}: {off_by_one} of {n} differ by one half-ulp from single-rounded CPU fma (documented double rounding)");
        // neg / abs
        for (op, f) in [("neg", (|x: f32| -x) as fn(f32) -> f32), ("abs", f32::abs)] {
            let ptx = ew(&[("ld.global.b16", 2, &format!("{reg}1"))], ("st.global.b16", 2, &format!("{reg}2")), &format!("\t{op}.{ty} {reg}2, {reg}1;\n"));
            let out = run_ew(&ptx, vec![ab.clone()], n * 2, n);
            for i in 0..n {
                let got = u16::from_le_bytes([out[2 * i], out[2 * i + 1]]);
                assert_eq!(dec(got), f(ha[i]), "{op}.{ty}");
            }
        }
    }
    // conversions f32 <-> bf16 / f16 including specials
    let specials: Vec<f32> = av.iter().copied().chain([0.0, -0.0, f32::INFINITY, f32::NEG_INFINITY, 1e-40, 3.4e38, 65519.0, 65520.0, 1.0 + 1.0 / 256.0]).collect();
    let m = specials.len();
    let fb = bytes_of_u32(&specials.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    let ptx = ew(&[("ld.global.b32", 4, "%f1")], ("st.global.b16", 2, "%hb1"), "\tcvt.rn.bf16.f32 %hb1, %f1;\n");
    let out = run_ew(&ptx, vec![fb.clone()], m * 2, m);
    for i in 0..m {
        assert_eq!(u16::from_le_bytes([out[2 * i], out[2 * i + 1]]), bf16_of(specials[i]), "cvt.rn.bf16.f32 {}", specials[i]);
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_packed_half2_arithmetic() {
    let mut rng = Rng(0x4a1f_2a1f_4a1f_2a1f);
    let n = 1 << 13;
    let word = |rng: &mut Rng| -> u32 {
        let h = |rng: &mut Rng| f32_to_f16((rng.range(-512, 512) as f32) / 16.0) as u32;
        h(rng) | (h(rng) << 16)
    };
    let a: Vec<u32> = (0..n).map(|_| word(&mut rng)).collect();
    let b: Vec<u32> = (0..n).map(|_| word(&mut rng)).collect();
    for (op, f) in [("add.rn.f16x2", (|x, y| x + y) as fn(f32, f32) -> f32), ("sub.rn.f16x2", |x, y| x - y), ("mul.rn.f16x2", |x, y| x * y)] {
        let ptx = ew(&[("ld.global.b32", 4, "%q1"), ("ld.global.b32", 4, "%q2")], ("st.global.b32", 4, "%q3"), &format!("\t{op} %q3, %q1, %q2;\n"));
        let out = u32s_of(&run_ew(&ptx, vec![bytes_of_u32(&a), bytes_of_u32(&b)], n * 4, n));
        for i in 0..n {
            let lane = |w: u32, s: u32| f16_to_f32((w >> s) as u16);
            let want = f32_to_f16(f(lane(a[i], 0), lane(b[i], 0))) as u32 | ((f32_to_f16(f(lane(a[i], 16), lane(b[i], 16))) as u32) << 16);
            assert_eq!(out[i], want, "{op}");
        }
    }
}

// ---------------------------------------------------------------- approximate math

fn ulp32(got: f32, want: f64) -> f64 {
    if got.is_nan() || !want.is_finite() { return if got.is_nan() == want.is_nan() { 0.0 } else { f64::INFINITY }; }
    let w = want as f32;
    let ulp = (f32::from_bits(w.to_bits() & 0x7f80_0000).max(f32::MIN_POSITIVE) * 2f32.powi(-23)) as f64;
    ((got as f64) - want).abs() / ulp
}

fn approx_run(op: &str, xs: &[f32]) -> Vec<f32> {
    let ptx = ew(&[("ld.global.b32", 4, "%f1")], ("st.global.b32", 4, "%f2"), &format!("\t{op} %f2, %f1;\n"));
    u32s_of(&run_ew(&ptx, vec![bytes_of_u32(&xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>())], xs.len() * 4, xs.len())).into_iter().map(f32::from_bits).collect()
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_approximate_math_error_bounds() {
    let mut rng = Rng(0x3a11_0a11_3a11_0a11);
    let n = 1 << 17;
    let uni = |rng: &mut Rng, lo: f64, hi: f64| lo as f32 + (hi - lo) as f32 * (rng.u32() as f32 / u32::MAX as f32);
    let mut lines = vec![];
    // (op, domain, reference, max ulp bound, abs bound (0 = none))
    type R = fn(f64) -> f64;
    let cases: [(&str, (f64, f64), R, f64, f64); 7] = [
        ("ex2.approx.f32", (-20.0, 20.0), |x| x.exp2(), 4.0, 0.0),
        ("lg2.approx.f32", (1e-6, 1e6), |x| x.log2(), 0.0, 2f64.powi(-20)),
        ("rcp.approx.f32", (0.001, 1000.0), |x| 1.0 / x, 2.0, 0.0),
        ("rsqrt.approx.f32", (0.001, 1000.0), |x| 1.0 / x.sqrt(), 4.0, 0.0),
        ("sin.approx.f32", (-std::f64::consts::PI, std::f64::consts::PI), |x| x.sin(), 0.0, 2f64.powi(-19)),
        ("cos.approx.f32", (-std::f64::consts::PI, std::f64::consts::PI), |x| x.cos(), 0.0, 2f64.powi(-19)),
        ("tanh.approx.f32", (-8.0, 8.0), |x| x.tanh(), 0.0, 0.0),
    ];
    for (op, (lo, hi), f, max_ulp, abs_bound) in cases {
        let xs: Vec<f32> = (0..n).map(|_| uni(&mut rng, lo, hi)).collect();
        let out = approx_run(op, &xs);
        let (mut worst_ulp, mut worst_abs, mut worst_rel) = (0f64, 0f64, 0f64);
        for i in 0..n {
            let want = f(xs[i] as f64);
            worst_ulp = worst_ulp.max(ulp32(out[i], want));
            worst_abs = worst_abs.max((out[i] as f64 - want).abs());
            if want != 0.0 { worst_rel = worst_rel.max(((out[i] as f64 - want) / want).abs()); }
        }
        lines.push(format!("{op}: max {worst_ulp:.2} ulp, max abs err {worst_abs:.3e} (2^{:.1}), max rel err {worst_rel:.3e}", worst_abs.log2()));
        if max_ulp > 0.0 { assert!(worst_ulp <= max_ulp, "{op}: {worst_ulp} ulp > {max_ulp}"); }
        if abs_bound > 0.0 && !op.starts_with("lg2") { assert!(worst_abs <= abs_bound, "{op}: abs err {worst_abs:e} > {abs_bound:e}"); }
        if op.starts_with("lg2") { assert!(worst_ulp <= 4.0 || worst_abs <= 2f64.powi(-21) * 4.0, "lg2: {worst_ulp} ulp / {worst_abs:e} abs"); }
        if op.starts_with("tanh") { assert!(worst_rel <= 2f64.powi(-10), "tanh rel {worst_rel:e}"); }
    }
    // special values
    let sp = [0.0f32, -0.0, f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
    for op in ["ex2.approx.f32", "rcp.approx.f32", "rsqrt.approx.f32", "tanh.approx.f32"] {
        lines.push(format!("{op} specials [0,-0,inf,-inf,nan] -> {:?}", approx_run(op, &sp)));
    }
    println!("{lines:#?}");
}

// ---------------------------------------------------------------- cp.async

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_cp_async_copies_through_shared_memory() {
    for size in [4u32, 8, 16] {
        let words = (size / 4) as usize;
        let mut s = prologue("cpa_k", 2, 0, "\t.shared .align 16 .b8 sbuf[2048];\n");
        s += "\tmov.u32 %r0, %tid.x;\n\tmov.u32 %r1, sbuf;\n";
        s += &format!("\tmul.lo.u32 %r2, %r0, {size};\n\tadd.u32 %r3, %r1, %r2;\n\tcvt.u64.u32 %rd20, %r2;\n\tadd.s64 %rd21, %rd1, %rd20;\n");
        s += &format!("\tcp.async.ca.shared.global [%r3], [%rd21], {size};\n\tcp.async.commit_group;\n\tcp.async.wait_group 0;\n\tbar.sync 0;\n");
        // read the neighbour's chunk to prove cross-thread visibility
        s += &format!("\tadd.u32 %r4, %r0, 1;\n\tand.b32 %r4, %r4, 63;\n\tmul.lo.u32 %r5, %r4, {size};\n\tadd.u32 %r5, %r5, %r1;\n\tmul.wide.u32 %rd22, %r0, {size};\n\tadd.s64 %rd23, %rd2, %rd22;\n");
        for w in 0..words {
            s += &format!("\tld.shared.u32 %r{}, [%r5+{}];\n\tst.global.u32 [%rd23+{}], %r{};\n", 10 + w, 4 * w, 4 * w, 10 + w);
        }
        s += "\tret;\n}\n";
        let mut rng = Rng(0xc9a5_0000 + size as u64);
        let data: Vec<u32> = (0..64 * words).map(|_| rng.u32()).collect();
        let mut bufs = vec![bytes_of_u32(&data), vec![0u8; 64 * words * 4]];
        launch(&s, "cpa_k", 1, 64, &mut bufs, &[]).unwrap_or_else(|e| panic!("cp.async size {size}: {e}"));
        let out = u32s_of(&bufs[1]);
        for t in 0..64usize {
            let nb = (t + 1) & 63;
            for w in 0..words {
                assert_eq!(out[t * words + w], data[nb * words + w], "cp.async {size} bytes, thread {t} word {w}");
            }
        }
    }
}

// ---------------------------------------------------------------- redux / match / vote / elect

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_redux_match_vote_elect() {
    let mut s = prologue("coll_k", 2, 0, "");
    s += "\tmov.u32 %r0, %tid.x;\n\tmul.wide.u32 %rd20, %r0, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tld.global.u32 %r4, [%rd21];\n";
    let ops = [
        "redux.sync.add.u32", "redux.sync.min.u32", "redux.sync.max.u32", "redux.sync.min.s32", "redux.sync.max.s32",
        "redux.sync.and.b32", "redux.sync.or.b32", "redux.sync.xor.b32",
    ];
    for (i, op) in ops.iter().enumerate() {
        s += &format!("\t{op} %r{}, %r4, 0xffffffff;\n", 10 + i);
    }
    s += "\tmatch.any.sync.b32 %r20, %r4, 0xffffffff;\n\tmatch.all.sync.b32 %r21|%p1, %r4, 0xffffffff;\n\tselp.u32 %r22, 1, 0, %p1;\n";
    s += "\tsetp.lt.u32 %p2, %r4, 100;\n\tvote.sync.uni.pred %p3, %p2, 0xffffffff;\n\tselp.u32 %r23, 1, 0, %p3;\n";
    s += "\telect.sync %r24|%p4, 0xffffffff;\n\tselp.u32 %r25, 1, 0, %p4;\n";
    s += "\tmul.wide.u32 %rd22, %r0, 64;\n\tadd.s64 %rd23, %rd2, %rd22;\n";
    for (k, r) in (10..=18).chain(20..=25).enumerate() {
        if r == 19 { continue; }
        s += &format!("\tst.global.u32 [%rd23+{}], %r{r};\n", 4 * k.min(14));
    }
    s += "\tret;\n}\n";
    // store layout: 8 redux, 20, 21(? skip 19) -> recompute explicitly below
    let mut rng = Rng(0x5eed_c011_5eed_c011);
    for case in 0..12 {
        let vals: Vec<u32> = (0..32)
            .map(|i| match case % 4 {
                0 => rng.u32(),
                1 => rng.u32() % 8,
                2 => 77,
                _ => if i % 2 == 0 { 5 } else { 5000 },
            })
            .collect();
        // Rebuild the kernel with an explicit simple store map instead of the loop above.
        let mut k = prologue("coll_k", 2, 0, "");
        k += "\tmov.u32 %r0, %tid.x;\n\tmul.wide.u32 %rd20, %r0, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tld.global.u32 %r4, [%rd21];\n";
        for (i, op) in ops.iter().enumerate() {
            k += &format!("\t{op} %r{}, %r4, 0xffffffff;\n", 10 + i);
        }
        k += "\tmatch.any.sync.b32 %r20, %r4, 0xffffffff;\n\tmatch.all.sync.b32 %r21|%p1, %r4, 0xffffffff;\n\tselp.u32 %r22, 1, 0, %p1;\n";
        k += "\tsetp.lt.u32 %p2, %r4, 100;\n\tvote.sync.uni.pred %p3, %p2, 0xffffffff;\n\tselp.u32 %r23, 1, 0, %p3;\n";
        k += "\telect.sync %r24|%p4, 0xffffffff;\n\tselp.u32 %r25, 1, 0, %p4;\n";
        k += "\tmul.wide.u32 %rd22, %r0, 64;\n\tadd.s64 %rd23, %rd2, %rd22;\n";
        let regs = [10, 11, 12, 13, 14, 15, 16, 17, 20, 22, 23, 25];
        for (k2, r) in regs.iter().enumerate() {
            k += &format!("\tst.global.u32 [%rd23+{}], %r{r};\n", 4 * k2);
        }
        k += "\tret;\n}\n";
        let mut bufs = vec![bytes_of_u32(&vals), vec![0u8; 32 * 64]];
        launch(&k, "coll_k", 1, 32, &mut bufs, &[]).unwrap_or_else(|e| panic!("collectives: {e}"));
        let out = u32s_of(&bufs[1]);
        let sum = vals.iter().fold(0u32, |a, v| a.wrapping_add(*v));
        let expect = [
            sum,
            *vals.iter().min().unwrap(),
            *vals.iter().max().unwrap(),
            *vals.iter().map(|v| *v as i32).min().as_ref().map(|v| v).unwrap() as u32,
            *vals.iter().map(|v| *v as i32).max().as_ref().map(|v| v).unwrap() as u32,
            vals.iter().fold(u32::MAX, |a, v| a & v),
            vals.iter().fold(0, |a, v| a | v),
            vals.iter().fold(0, |a, v| a ^ v),
        ];
        for lane in 0..32usize {
            let r = &out[lane * 16..lane * 16 + 12];
            assert_eq!(&r[..8], &expect, "redux case {case} lane {lane}");
            let any = (0..32).filter(|&j| vals[j] == vals[lane]).fold(0u32, |m, j| m | 1 << j);
            assert_eq!(r[8], any, "match.any case {case} lane {lane}");
            assert_eq!(r[9], vals.iter().all(|v| *v == vals[0]) as u32, "match.all case {case}");
            assert_eq!(r[10], (vals.iter().all(|v| *v < 100) || vals.iter().all(|v| *v >= 100)) as u32, "vote.uni case {case}");
            assert_eq!(r[11], (lane == 0) as u32, "elect.sync case {case} lane {lane}");
        }
    }
    let _ = s;
}

// ---------------------------------------------------------------- wmma

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_wmma_m16n16k16_f16_f32() {
    for (a_row, b_row) in [(true, true), (true, false), (false, true), (false, false)] {
        let (al, bl) = (if a_row { "row" } else { "col" }, if b_row { "row" } else { "col" });
        let list = |p: &str| (0..8).map(|i| format!("%{p}{i}")).collect::<Vec<_>>().join(", ");
        let regs = "\t.reg .b32 %wa<8>;\n\t.reg .b32 %wb<8>;\n\t.reg .f32 %wc<8>;\n\t.reg .f32 %wd<8>;\n";
        let mut s = prologue("wmma_k", 3, 0, regs);
        for i in 0..8 {
            s += &format!("\tmov.b32 %wc{i}, 0;\n");
        }
        s += &format!(
            "\twmma.load.a.sync.aligned.{al}.m16n16k16.global.f16 {{{a}}}, [%rd1], 16;\n\twmma.load.b.sync.aligned.{bl}.m16n16k16.global.f16 {{{b}}}, [%rd2], 16;\n\twmma.mma.sync.aligned.{al}.{bl}.m16n16k16.f32.f32 {{{d}}}, {{{a}}}, {{{b}}}, {{{c}}};\n\twmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%rd3], {{{d}}}, 16;\n\tret;\n}}\n",
            a = list("wa"), b = list("wb"), c = list("wc"), d = list("wd")
        );
        let mut rng = Rng(0x3a77_a000 + a_row as u64 * 2 + b_row as u64);
        let a: Vec<Vec<f32>> = (0..16).map(|_| (0..16).map(|_| rng.range(-32, 31) as f32 / 8.0).collect()).collect();
        let b: Vec<Vec<f32>> = (0..16).map(|_| (0..16).map(|_| rng.range(-32, 31) as f32 / 8.0).collect()).collect();
        let (mut ab, mut bb) = (vec![0u8; 512], vec![0u8; 512]);
        for r in 0..16 {
            for c in 0..16 {
                let (ai, bi) = (if a_row { r * 16 + c } else { c * 16 + r }, if b_row { r * 16 + c } else { c * 16 + r });
                ab[2 * ai..2 * ai + 2].copy_from_slice(&f32_to_f16(a[r][c]).to_le_bytes());
                bb[2 * bi..2 * bi + 2].copy_from_slice(&f32_to_f16(b[r][c]).to_le_bytes());
            }
        }
        let mut bufs = vec![ab, bb, vec![0u8; 1024]];
        launch(&s, "wmma_k", 1, 32, &mut bufs, &[]).unwrap_or_else(|e| panic!("wmma {al}/{bl}: {e}"));
        let d = u32s_of(&bufs[2]);
        for r in 0..16 {
            for c in 0..16 {
                let want: f32 = (0..16).map(|k| a[r][k] * b[k][c]).sum();
                assert_eq!(f32::from_bits(d[r * 16 + c]), want, "wmma a={al} b={bl} D[{r}][{c}]");
            }
        }
    }
}

// ---------------------------------------------------------------- atomics under contention

fn contended_ptx(op: &str, ty: &str) -> String {
    let mut s = prologue("cont_k", 3, 1, DECLS);
    s += GUARD;
    s += "\tmul.wide.u32 %rd20, %r3, 8;\n\tadd.s64 %rd21, %rd2, %rd20;\n\tadd.s64 %rd22, %rd3, %rd20;\n";
    s += "\tld.global.b64 %l1, [%rd21];\n";
    if ty == "f64" {
        s += &format!("\tmov.b64 %fd1, %l1;\n\tatom.global.{op}.f64 %fd2, [%rd1], %fd1;\n\tst.global.b64 [%rd22], %fd2;\n");
    } else {
        s += &format!("\tatom.global.{op}.{ty} %l2, [%rd1], %l1;\n\tst.global.b64 [%rd22], %l2;\n");
    }
    s += "$done:\n\tret;\n}\n";
    s
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_atomics_64bit_and_f64_under_heavy_contention() {
    let n = 200_000usize;
    let mut rng = Rng(0xc0c0_a70a_c0c0_a70a);
    // u64 add / max / xor on one address from 200k threads across many blocks.
    for (op, ty) in [("add", "u64"), ("max", "u64"), ("min", "s64"), ("xor", "b64"), ("or", "b64")] {
        let vals: Vec<u64> = (0..n).map(|_| ((rng.u32() as u64) << 20) ^ rng.u32() as u64).collect();
        let init = if op == "min" { i64::MAX as u64 } else { 0 };
        let mut bufs = vec![init.to_le_bytes().to_vec(), b64(&vals), vec![0u8; n * 8]];
        launch(&contended_ptx(op, ty), "cont_k", (n / 250) as u32, 250, &mut bufs, &[n as u32]).unwrap_or_else(|e| panic!("atom.{op}.{ty}: {e}"));
        let fin = u64s(&bufs[0])[0];
        let want = match op {
            "add" => vals.iter().fold(0u64, |a, v| a.wrapping_add(*v)),
            "max" => *vals.iter().max().unwrap(),
            "min" => (*vals.iter().map(|v| *v as i64).min().as_ref().unwrap()) as u64,
            "xor" => vals.iter().fold(0, |a, v| a ^ v),
            _ => vals.iter().fold(0, |a, v| a | v),
        };
        assert_eq!(fin, want, "atom.{op}.{ty} final value under contention (lost updates?)");
    }
    // f64 add: returned olds must form one serial chain.
    let vals: Vec<f64> = (0..n).map(|_| (1 + rng.u32() % 8) as f64 * 0.25).collect();
    let mut bufs = vec![0f64.to_bits().to_le_bytes().to_vec(), b64(&vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>()), vec![0u8; n * 8]];
    launch(&contended_ptx("add", "f64"), "cont_k", (n / 250) as u32, 250, &mut bufs, &[n as u32]).unwrap_or_else(|e| panic!("atom.add.f64: {e}"));
    let total: f64 = vals.iter().sum();
    assert_eq!(f64::from_bits(u64s(&bufs[0])[0]), total, "atom.add.f64 final");
    let mut seq: Vec<(f64, f64)> = u64s(&bufs[2]).iter().zip(&vals).map(|(o, v)| (f64::from_bits(*o), *v)).collect();
    seq.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    let mut expect = 0.0;
    for (old, v) in seq {
        assert_eq!(old, expect, "atom.add.f64 returned olds are not a serial chain");
        expect += v;
    }
}
