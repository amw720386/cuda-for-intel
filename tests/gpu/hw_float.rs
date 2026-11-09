//! Directed-rounding arithmetic and cvt rounding vs CPU references.
use super::hw_common::*;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Mode {
    Zero,
    Down,
    Up,
}

fn next_up(x: f32) -> f32 {
    if x.is_nan() || x == f32::INFINITY {
        return x;
    }
    if x == 0.0 {
        return f32::from_bits(1);
    }
    let b = x.to_bits();
    f32::from_bits(if x > 0.0 { b + 1 } else { b - 1 })
}

fn next_down(x: f32) -> f32 {
    -next_up(-x)
}

/// Correctly rounded (directed) result of a + b (op 0), a - b (1) or a * b (2).
pub fn directed(op: u8, mode: Mode, a: f32, b: f32) -> u32 {
    let b = if op == 1 { -b } else { b };
    let nearest = if op == 2 { a * b } else { a + b };
    if nearest.is_nan() {
        return 0x7fc0_0000;
    }
    if a.is_infinite() || b.is_infinite() {
        return nearest.to_bits();
    }
    let (x, y) = (a as f64, b as f64);
    // Exact value as s + e (s, e f64).
    let (s, e) = if op == 2 {
        (x * y, 0.0) // 24x24 bit product is exact in f64
    } else {
        let s = x + y;
        let bb = s - x;
        let e = (x - (s - bb)) + (y - bb);
        (s, e)
    };
    if s == 0.0 && e == 0.0 {
        // exact zero
        let sign = if op == 2 {
            a.is_sign_negative() ^ b.is_sign_negative()
        } else if a == 0.0 && b == 0.0 {
            a.is_sign_negative() && b.is_sign_negative()
        } else {
            mode == Mode::Down // x + (-x)
        };
        return if sign { 0x8000_0000 } else { 0 };
    }
    let r = s as f32; // round to nearest of s (double rounding resolved below)
    // sign of (exact - r)
    let sgn = if r.is_infinite() {
        -(r.signum() as i32)
    } else {
        let d = s - r as f64; // exact
        if d > -e { 1 } else if d < -e { -1 } else { 0 }
    };
    let exact_positive = s > 0.0 || (s == 0.0 && e > 0.0);
    let down = |r: f32, sgn: i32| if sgn < 0 { next_down(r) } else { r };
    let up = |r: f32, sgn: i32| if sgn > 0 { next_up(r) } else { r };
    let out = match mode {
        Mode::Down => down(r, sgn),
        Mode::Up => up(r, sgn),
        Mode::Zero => {
            if exact_positive { down(r, sgn) } else { up(r, sgn) }
        }
    };
    out.to_bits()
}

fn interesting(rng: &mut Rng) -> u32 {
    match rng.u32() % 8 {
        0 => rng.u32(),                                                        // anything
        1 => (rng.u32() & 0x807f_ffff) | ((100 + rng.u32() % 55) << 23),       // moderate exponents
        2 => (rng.u32() & 0x807f_ffff) | ((120 + rng.u32() % 16) << 23),       // near 1
        3 => (rng.u32() & 0x807f_ffff) | ((rng.u32() % 4) << 23),              // tiny / denormal
        4 => (rng.u32() & 0x807f_ffff) | ((250 + rng.u32() % 5) << 23),        // near overflow
        5 => [0, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x7f7f_ffff, 0xff7f_ffff, 0x0080_0000, 0x0000_0001][(rng.u32() % 8) as usize],
        _ => (rng.u32() & 0x807f_ffff) | ((110 + rng.u32() % 36) << 23),
    }
}

fn float_pairs(rng: &mut Rng, n: usize) -> Vec<(u32, u32)> {
    (0..n)
        .map(|i| {
            let a = interesting(rng);
            let b = match i % 5 {
                0 => a ^ 0x8000_0000, // exact cancellation
                1 => (a ^ 0x8000_0000).wrapping_add(rng.u32() % 5), // near cancellation
                2 => (a & 0x8000_0000 ^ 0x8000_0000) | (a.wrapping_add(rng.u32() % 3) & 0x7fff_ffff), // opposite sign close
                _ => interesting(rng),
            };
            (a, b)
        })
        .collect()
}

fn is_denormal(bits: u32) -> bool {
    bits & 0x7f80_0000 == 0 && bits & 0x007f_ffff != 0
}

fn directed_ptx(op: &str, mode: &str) -> String {
    let mut s = prologue("dir_k", 3, 1, "");
    s += "\tmov.u32 %r0, %tid.x;\n\tmov.u32 %r1, %ctaid.x;\n\tmov.u32 %r2, %ntid.x;\n\tmad.lo.s32 %r3, %r1, %r2, %r0;\n\tsetp.ge.u32 %p1, %r3, %r100;\n\t@%p1 bra $done;\n";
    s += "\tmul.wide.u32 %rd20, %r3, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tadd.s64 %rd22, %rd2, %rd20;\n\tadd.s64 %rd23, %rd3, %rd20;\n";
    s += "\tld.global.f32 %f1, [%rd21];\n\tld.global.f32 %f2, [%rd22];\n";
    s += &format!("\t{op}.{mode}.f32 %f3, %f1, %f2;\n");
    s += "\tst.global.f32 [%rd23], %f3;\n$done:\n\tret;\n}\n";
    s
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_directed_rounding_add_sub_mul_rz_rm_rp() {
    let mut rng = Rng(0xabcdef0123456789);
    let n = 1 << 17;
    let mut report = Vec::new();
    for (opn, op) in ["add", "sub", "mul"].iter().enumerate() {
        for (mode, name) in [(Mode::Zero, "rz"), (Mode::Down, "rm"), (Mode::Up, "rp")] {
            let pairs = float_pairs(&mut rng, n);
            let a: Vec<u32> = pairs.iter().map(|p| p.0).collect();
            let b: Vec<u32> = pairs.iter().map(|p| p.1).collect();
            let mut bufs = vec![bytes_of_u32(&a), bytes_of_u32(&b), vec![0u8; n * 4]];
            launch(&directed_ptx(op, name), "dir_k", (n / 256) as u32, 256, &mut bufs, &[n as u32]).unwrap();
            let got = u32s_of(&bufs[2]);
            let (mut bad, mut bad_denorm, mut first) = (0, 0, None);
            for i in 0..n {
                let (fa, fb) = (f32::from_bits(a[i]), f32::from_bits(b[i]));
                let want = directed(opn as u8, mode, fa, fb);
                let both_nan = f32::from_bits(want).is_nan() && f32::from_bits(got[i]).is_nan();
                if got[i] != want && !both_nan {
                    // Denormal operands or results may legitimately be flushed by the device.
                    if is_denormal(want) || is_denormal(a[i]) || is_denormal(b[i]) || is_denormal(got[i]) {
                        bad_denorm += 1;
                        if bad_denorm <= 4 {
                            println!("{op}.{name} denormal-case #{i}: a={:08x} b={:08x} got={:08x} want={:08x}", a[i], b[i], got[i], want);
                        }
                    } else {
                        bad += 1;
                        first.get_or_insert((i, a[i], b[i], got[i], want));
                    }
                }
            }
            report.push(format!("{op}.{name}: {bad} wrong, {bad_denorm} denormal diffs of {n}"));
            if let Some((i, x, y, g, w)) = first {
                panic!("{op}.{name} first mismatch #{i}: a={x:08x} b={y:08x} got={g:08x} want={w:08x}; {} bad", bad);
            }
            assert_eq!(bad_denorm, 0, "{op}.{name}: denormal handling differs from CPU (see report): {report:?}");
        }
    }
    println!("{report:#?}");
}

/// `div.rn.f32` and `sqrt.rn.f32` must be correctly rounded (IEEE) like on NVIDIA. Intel's OpenCL
/// compiler only guarantees 2.5 ulp for `/` and 3 ulp for `sqrt` unless it is built with
/// `-cl-fp32-correctly-rounded-divide-sqrt`; without it ggml's `x / d` in quantize_q8_1 (and every
/// softmax/norm division) rounds differently from CUDA, which flips q8 activation bytes and
/// decorrelates greedy decoding (docs/qwen05b-divergence.md).
#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_div_sqrt_rn_f32_correctly_rounded() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let n = 1 << 18;
    let mut pairs = float_pairs(&mut rng, n);
    // Typical quantisation ratios x / (amax / 127): normal range operands.
    for (i, p) in pairs.iter_mut().enumerate() {
        if i % 2 == 0 {
            let a = f32::from_bits((rng.u32() & 0x807f_ffff) | ((118 + rng.u32() % 12) << 23));
            let amax = f32::from_bits((rng.u32() & 0x007f_ffff) | ((118 + rng.u32() % 12) << 23));
            *p = (a.to_bits(), (amax / 127.0).to_bits());
        }
    }
    let a: Vec<u32> = pairs.iter().map(|p| p.0).collect();
    let b: Vec<u32> = pairs.iter().map(|p| p.1).collect();
    let mut bufs = vec![bytes_of_u32(&a), bytes_of_u32(&b), vec![0u8; n * 4]];
    launch(&directed_ptx("div", "rn"), "dir_k", (n / 256) as u32, 256, &mut bufs, &[n as u32]).unwrap();
    let got = u32s_of(&bufs[2]);
    let mut bad = 0;
    let mut first = None;
    for i in 0..n {
        let want = (f32::from_bits(a[i]) / f32::from_bits(b[i])).to_bits();
        let both_nan = f32::from_bits(want).is_nan() && f32::from_bits(got[i]).is_nan();
        if got[i] != want && !both_nan && !is_denormal(want) && !is_denormal(got[i]) && !is_denormal(a[i]) && !is_denormal(b[i]) {
            bad += 1;
            first.get_or_insert((i, a[i], b[i], got[i], want));
        }
    }
    println!("div.rn.f32: {bad} not correctly rounded of {n}; first {first:x?}");

    // sqrt.rn.f32
    let xs: Vec<u32> = (0..n)
        .map(|i| if i % 2 == 0 { (rng.u32() & 0x007f_ffff) | ((100 + rng.u32() % 55) << 23) } else { interesting(&mut rng) & 0x7fff_ffff })
        .collect();
    let mut ptx = prologue("sq_k", 2, 1, "");
    ptx += "\tmov.u32 %r0, %tid.x;\n\tmov.u32 %r1, %ctaid.x;\n\tmov.u32 %r2, %ntid.x;\n\tmad.lo.s32 %r3, %r1, %r2, %r0;\n\tsetp.ge.u32 %p1, %r3, %r100;\n\t@%p1 bra $done;\n";
    ptx += "\tmul.wide.u32 %rd20, %r3, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tadd.s64 %rd22, %rd2, %rd20;\n";
    ptx += "\tld.global.f32 %f1, [%rd21];\n\tsqrt.rn.f32 %f3, %f1;\n\tst.global.f32 [%rd22], %f3;\n$done:\n\tret;\n}\n";
    let mut bufs = vec![bytes_of_u32(&xs), vec![0u8; n * 4]];
    launch(&ptx, "sq_k", (n / 256) as u32, 256, &mut bufs, &[n as u32]).unwrap();
    let got = u32s_of(&bufs[1]);
    let (mut sbad, mut sfirst) = (0, None);
    for i in 0..n {
        let want = f32::from_bits(xs[i]).sqrt().to_bits();
        let both_nan = f32::from_bits(want).is_nan() && f32::from_bits(got[i]).is_nan();
        if got[i] != want && !both_nan && !is_denormal(want) && !is_denormal(got[i]) && !is_denormal(xs[i]) {
            sbad += 1;
            sfirst.get_or_insert((i, xs[i], got[i], want));
        }
    }
    println!("sqrt.rn.f32: {sbad} not correctly rounded of {n}; first {sfirst:x?}");
    assert_eq!(bad, 0, "div.rn.f32 is not correctly rounded; first {first:x?}");
    assert_eq!(sbad, 0, "sqrt.rn.f32 is not correctly rounded; first {sfirst:x?}");
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_directed_rounding_regressions() {
    // (op, mode, a, b, expected bits): each failed before the matching fix in rounded.rs.
    let cases: [(u8, &str, Mode, u32, u32, u32); 6] = [
        // TwoSum overflowed for opposite-sign operands near +-MAX, losing the error term.
        (0, "add", Mode::Zero, 0x7e90_b523, 0xff7f_ffff, 0xff37_a56d),
        // fma error term underflowed below the smallest subnormal.
        (2, "mul", Mode::Zero, 0x3f5d_0af0, 0x01ec_c9e2, 0x01cc_7465),
        (2, "mul", Mode::Zero, 0xa01b_1610, 0x201b_1610, 0x80bb_e772),
        // product underflows to -0: round-down must give -min subnormal.
        (2, "mul", Mode::Down, 0x0183_3178, 0x8183_3178, 0x8000_0001),
        (2, "mul", Mode::Down, 0x0000_0001, 0x8000_0003, 0x8000_0001),
        (2, "mul", Mode::Up, 0x0000_0001, 0x0000_0003, 0x0000_0001),
    ];
    for (op, name, mode, a, b, want) in cases {
        let m = match mode { Mode::Zero => "rz", Mode::Down => "rm", Mode::Up => "rp" };
        assert_eq!(directed(op, mode, f32::from_bits(a), f32::from_bits(b)), want, "reference {name}.{m}");
        let mut bufs = vec![bytes_of_u32(&[a]), bytes_of_u32(&[b]), vec![0u8; 4]];
        launch(&directed_ptx(name, m), "dir_k", 1, 1, &mut bufs, &[1]).unwrap();
        assert_eq!(u32s_of(&bufs[2])[0], want, "{name}.{m} {a:08x} {b:08x}");
    }
}

#[test]
fn directed_reference_self_check() {
    // 1 + 2^-30 rounds up to the next float under rp, to 1.0 otherwise.
    let tiny = f32::from_bits(0x3080_0000); // 2^-30
    assert_eq!(directed(0, Mode::Up, 1.0, tiny), 0x3f80_0001);
    assert_eq!(directed(0, Mode::Down, 1.0, tiny), 0x3f80_0000);
    assert_eq!(directed(0, Mode::Zero, 1.0, tiny), 0x3f80_0000);
    assert_eq!(directed(0, Mode::Down, -1.0, -tiny), 0xbf80_0001);
    assert_eq!(directed(0, Mode::Zero, -1.0, -tiny), 0xbf80_0000);
    assert_eq!(directed(2, Mode::Zero, f32::MAX, 2.0), 0x7f7f_ffff);
    assert_eq!(directed(2, Mode::Up, f32::MAX, 2.0), 0x7f80_0000);
    assert_eq!(directed(0, Mode::Down, 1.0, -1.0), 0x8000_0000);
    assert_eq!(directed(0, Mode::Up, 1.0, -1.0), 0);
}

// ------------------------------------------------------------------ cvt

fn cvt_ptx(instr: &str, input: &str, output: &str) -> String {
    // input/output in {"f32","b32","b16"}; element sizes 4 / 4 / 2 bytes.
    let (isz, osz) = (if input == "b16" { 2 } else { 4 }, if output == "b16" { 2 } else { 4 });
    let mut s = prologue("cvt_k", 2, 1, "\t.reg .b16 %h<8>;\n");
    s += "\tmov.u32 %r0, %tid.x;\n\tmov.u32 %r1, %ctaid.x;\n\tmov.u32 %r2, %ntid.x;\n\tmad.lo.s32 %r3, %r1, %r2, %r0;\n\tsetp.ge.u32 %p1, %r3, %r100;\n\t@%p1 bra $done;\n";
    s += &format!("\tmul.wide.u32 %rd20, %r3, {isz};\n\tadd.s64 %rd21, %rd1, %rd20;\n\tmul.wide.u32 %rd22, %r3, {osz};\n\tadd.s64 %rd23, %rd2, %rd22;\n");
    let (src, dst) = (
        match input { "f32" => "%f1", "b16" => "%h1", _ => "%r10" },
        match output { "f32" => "%f2", "b16" => "%h2", _ => "%r11" },
    );
    s += &match input {
        "f32" => "\tld.global.f32 %f1, [%rd21];\n".to_string(),
        "b16" => "\tld.global.u16 %h1, [%rd21];\n".to_string(),
        _ => "\tld.global.u32 %r10, [%rd21];\n".to_string(),
    };
    s += &format!("\t{instr} {dst}, {src};\n");
    s += &match output {
        "f32" => "\tst.global.f32 [%rd23], %f2;\n".to_string(),
        "b16" => "\tst.global.u16 [%rd23], %h2;\n".to_string(),
        _ => "\tst.global.u32 [%rd23], %r11;\n".to_string(),
    };
    s += "$done:\n\tret;\n}\n";
    s
}

fn cvt_inputs(rng: &mut Rng, n: usize) -> Vec<u32> {
    let specials = [
        0u32, 0x8000_0000, 0x3f00_0000, 0xbf00_0000, 0x3fc0_0000, 0xbfc0_0000, 0x4020_0000, 0x4000_0000,
        0x3f80_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0x4f00_0000, 0xcf00_0000, 0x4f80_0000, 0x4effffff,
        0xcf000001, 0x4b00_0000, 0x4b00_0001, 0x4aff_ffff, 0x477f_e000, 0x477f_f000, 0x477f_efff, 0x3f7f_ffff,
        0x0000_0001, 0x3380_0000, 0x387f_c000, 0x3880_0000,
    ];
    (0..n)
        .map(|i| {
            if i < specials.len() {
                specials[i]
            } else if i % 3 == 0 {
                // values with a fractional part around integers
                let int = rng.range(-300, 300) as f32;
                let frac = [0.0, 0.25, 0.5, 0.75, 0.499_999_97, 0.500_000_06][(rng.u32() % 6) as usize];
                (int + frac).to_bits()
            } else {
                interesting(rng)
            }
        })
        .collect()
}

fn run_cvt(instr: &str, input: &str, output: &str, inputs: &[u32]) -> Vec<u32> {
    let n = inputs.len();
    let osz = if output == "b16" { 2 } else { 4 };
    let isz_bytes: Vec<u8> = if input == "b16" {
        inputs.iter().flat_map(|v| (*v as u16).to_le_bytes()).collect()
    } else {
        bytes_of_u32(inputs)
    };
    let mut bufs = vec![isz_bytes, vec![0u8; n * osz]];
    launch(&cvt_ptx(instr, input, output), "cvt_k", n.div_ceil(256) as u32, 256, &mut bufs, &[n as u32])
        .unwrap_or_else(|e| panic!("{instr}: {e}"));
    if osz == 2 {
        bufs[1].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]]) as u32).collect()
    } else {
        u32s_of(&bufs[1])
    }
}

fn round_half_even(x: f32) -> f32 {
    let r = x.round(); // half away from zero
    let r = if (x - x.trunc()).abs() == 0.5 && r % 2.0 != 0.0 { r - x.signum() } else { r };
    r.copysign(x)
}

fn f32_eq_or_nan(got: u32, want: f32) -> bool {
    if want.is_nan() { f32::from_bits(got).is_nan() } else { got == want.to_bits() }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_cvt_float_to_integral_float_and_int_rounding() {
    let mut rng = Rng(0x1357_9bdf_2468_ace0);
    let inputs = cvt_inputs(&mut rng, 1 << 16);
    type FloatRef = fn(f32) -> f32;
    let cases: [(&str, FloatRef); 4] = [
        ("cvt.rzi.f32.f32", f32::trunc),
        ("cvt.rni.f32.f32", round_half_even),
        ("cvt.rmi.f32.f32", f32::floor),
        ("cvt.rpi.f32.f32", f32::ceil),
    ];
    for (instr, reference) in cases {
        let got = run_cvt(instr, "f32", "f32", &inputs);
        for (i, &x) in inputs.iter().enumerate() {
            let want = reference(f32::from_bits(x));
            assert!(
                f32_eq_or_nan(got[i], want),
                "{instr} x={x:08x} got={:08x} want={:08x}", got[i], want.to_bits()
            );
        }
    }
    // float -> int32 / uint32 (saturating, NaN -> 0)
    let s32 = |x: f32, r: fn(f32) -> f32| -> u32 {
        if x.is_nan() { 0 } else { (r(x).clamp(-2147483648.0, 2147483520.0) as i32) as u32 }
    };
    let u32f = |x: f32, r: fn(f32) -> f32| -> u32 {
        if x.is_nan() { 0 } else { r(x).clamp(0.0, 4294967040.0) as u32 }
    };
    for (instr, reference, signed) in [
        ("cvt.rzi.s32.f32", f32::trunc as fn(f32) -> f32, true),
        ("cvt.rni.s32.f32", round_half_even, true),
        ("cvt.rzi.u32.f32", f32::trunc, false),
        ("cvt.rni.u32.f32", round_half_even, false),
    ] {
        let got = run_cvt(instr, "f32", "b32", &inputs);
        for (i, &x) in inputs.iter().enumerate() {
            let fx = f32::from_bits(x);
            // saturation edge: 2^31 and above
            let want = if signed {
                if fx >= 2147483648.0 { 0x7fff_ffff } else { s32(fx, reference) }
            } else if fx >= 4294967296.0 {
                0xffff_ffff
            } else {
                u32f(fx, reference)
            };
            assert_eq!(got[i], want, "{instr} x={x:08x} ({fx})");
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_cvt_int_to_float_and_half_conversions() {
    let mut rng = Rng(0x2468_ace0_1357_9bdf);
    let ints: Vec<u32> = (0..1 << 16)
        .map(|i| match i % 4 {
            0 => rng.u32(),
            1 => rng.u32() >> (rng.u32() % 32),
            2 => (1u32 << (rng.u32() % 32)).wrapping_add(rng.u32() % 3).wrapping_sub(1),
            _ => rng.u32() | 0x8000_0000,
        })
        .collect();
    let got = run_cvt("cvt.rn.f32.u32", "b32", "f32", &ints);
    for (i, &v) in ints.iter().enumerate() {
        assert_eq!(got[i], (v as f32).to_bits(), "cvt.rn.f32.u32 {v}");
    }
    let got = run_cvt("cvt.rn.f32.s32", "b32", "f32", &ints);
    for (i, &v) in ints.iter().enumerate() {
        assert_eq!(got[i], (v as i32 as f32).to_bits(), "cvt.rn.f32.s32 {}", v as i32);
    }
    // exhaustive f16 -> f32
    let halves: Vec<u32> = (0..65536).collect();
    let got = run_cvt("cvt.f32.f16", "b16", "f32", &halves);
    for h in 0..65536usize {
        let want = f16_to_f32(h as u16);
        assert!(f32_eq_or_nan(got[h], want), "cvt.f32.f16 {h:04x} got={:08x} want={:08x}", got[h], want.to_bits());
    }
    // f32 -> f16, rn and rz
    let inputs = cvt_inputs(&mut rng, 1 << 18);
    let rz = |x: f32| -> u16 {
        let b = x.to_bits();
        if x.is_nan() { return 0x7e00; }
        let sign = ((b >> 16) & 0x8000) as u16;
        let a = x.abs();
        if a.is_infinite() { return sign | 0x7c00; }
        if a >= 65536.0 { return sign | 0x7bff; }
        let rn = f32_to_f16(a);
        // rz: if rn rounded up (value above a), step down one
        let back = f16_to_f32(rn);
        sign | if back > a { rn - 1 } else { rn }
    };
    for (instr, is_rz) in [("cvt.rn.f16.f32", false), ("cvt.rz.f16.f32", true)] {
        let got = run_cvt(instr, "f32", "b16", &inputs);
        for (i, &x) in inputs.iter().enumerate() {
            let fx = f32::from_bits(x);
            let want = if is_rz { rz(fx) } else { f32_to_f16(fx) };
            if fx.is_nan() {
                assert_eq!(got[i] & 0x7c00, 0x7c00, "{instr} NaN x={x:08x}");
                assert_ne!(got[i] & 0x3ff, 0, "{instr} NaN x={x:08x}");
                continue;
            }
            // f16 denormal results may be flushed by some devices; report separately
            assert_eq!(got[i] as u16, want, "{instr} x={x:08x} ({fx})");
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_cvt_sat_and_ftz_variants() {
    let mut rng = Rng(0x5a75_f700_5a75_f700);
    let mut inputs = cvt_inputs(&mut rng, 1 << 14);
    inputs.extend([0x3f80_0001, 0xbf80_0000, 0x7f7f_ffff, 0x3f7f_ffff, 0x8000_0001]);
    for instr in ["cvt.sat.f32.f32", "cvt.rn.sat.f32.f32", "cvt.sat.ftz.f32.f32", "cvt.rn.ftz.sat.f32.f32"] {
        let got = run_cvt(instr, "f32", "f32", &inputs);
        for (i, &x) in inputs.iter().enumerate() {
            let fx = f32::from_bits(x);
            let want = if fx.is_nan() { 0.0 } else { fx.clamp(0.0, 1.0) };
            assert!(got[i] == want.to_bits() || (want == 0.0 && got[i] & 0x7fff_ffff == 0), "{instr} x={x:08x} got={:08x}", got[i]);
            assert_ne!(got[i], 0x8000_0000, "{instr}: saturate must not return -0 (x={x:08x})");
        }
    }
    let plain = [
        ("cvt.rzi.s32.f32", "cvt.rzi.ftz.s32.f32"),
        ("cvt.rni.s32.f32", "cvt.rni.ftz.s32.f32"),
        ("cvt.rzi.u32.f32", "cvt.rzi.ftz.u32.f32"),
        ("cvt.rni.u32.f32", "cvt.rni.ftz.u32.f32"),
    ];
    let normal: Vec<u32> = inputs.iter().copied().filter(|b| b & 0x7f80_0000 != 0 || b & 0x7fff_ffff == 0).collect();
    for (a, b) in plain {
        assert_eq!(run_cvt(a, "f32", "b32", &normal), run_cvt(b, "f32", "b32", &normal), "{b}");
    }
    let halves: Vec<u32> = (0..65536).collect();
    assert_eq!(run_cvt("cvt.f32.f16", "b16", "f32", &halves), run_cvt("cvt.ftz.f32.f16", "b16", "f32", &halves));
    let big: Vec<u32> = normal.iter().copied().filter(|b| (b >> 23) & 0xff >= 113).collect();
    assert_eq!(run_cvt("cvt.rn.f16.f32", "f32", "b16", &big), run_cvt("cvt.rn.ftz.f16.f32", "f32", "b16", &big));
}
