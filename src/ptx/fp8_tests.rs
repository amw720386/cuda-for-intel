//! fp8 conversions and vector registers (roadmap item 7), run on the SPIR-V simulator.
//!
//! The fp8 reference is deliberately unlike the lowering in `ext_fp8.rs`: it decodes all 256
//! encodings to `f64` and picks the nearest finite one (ties to the even encoding), clamping
//! beyond the largest finite value, which is what `.rn.satfinite` specifies.
use super::sim::{Gpu, Launch, Sim};
use super::translate;

const HEADER: &str = ".version 8.8\n.target sm_90\n.address_size 64\n";

/// Runs `body` once per element: `%r4` is the global thread index, `%rd1`/`%rd2` the input and
/// output buffers. Returns the output bytes.
fn run(body: &str, input: &[u8], out_bytes: usize, threads: u32) -> Vec<u8> {
    let src = format!(
        "{HEADER}.visible .entry k(.param .u64 pin, .param .u64 pout)\n{{\n\
         .reg .pred %p<4>;\n.reg .b16 %h<8>;\n.reg .b32 %r<16>;\n.reg .b64 %rd<16>;\n.reg .f32 %f<8>;\n\
         ld.param.u64 %rd1, [pin];\nld.param.u64 %rd2, [pout];\n\
         cvta.to.global.u64 %rd1, %rd1;\ncvta.to.global.u64 %rd2, %rd2;\n\
         mov.u32 %r1, %tid.x;\nmov.u32 %r2, %ctaid.x;\nmov.u32 %r3, %ntid.x;\n\
         mad.lo.s32 %r4, %r2, %r3, %r1;\n{body}\nret;\n}}\n"
    );
    let t = translate(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    let module = rspirv::dr::load_words(&t.words).unwrap();
    let mut gpu = Gpu::default();
    let input = gpu.upload(input);
    let out = gpu.alloc(out_bytes);
    let block = threads.min(256);
    let mut sim = Sim::new(&module);
    sim.run(&mut gpu, &Launch::new([threads / block, 1, 1], [block, 1, 1], vec![input, out])).unwrap();
    assert!(gpu.violations.borrow().is_empty(), "{:?}", gpu.violations.borrow());
    gpu.download(out, out_bytes)
}

/// Value of an fp8 encoding (`None` for NaN).
fn fp8_value(x: u8, e5m2: bool) -> Option<f64> {
    let (m_bits, bias) = if e5m2 { (2, 15) } else { (3, 7) };
    let sign = if x & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = i32::from((x & 0x7f) >> m_bits);
    let mant = f64::from(x & ((1 << m_bits) - 1));
    let top = if e5m2 { 31 } else { 15 };
    if exp == top && (e5m2 || x & 0x7f == 0x7f) {
        // E5M2 reserves the top exponent for infinities and NaN; E4M3 only 0x7f/0xff.
        return if e5m2 && mant == 0.0 { Some(sign * f64::INFINITY) } else { None };
    }
    let scale = f64::from(1 << m_bits);
    Some(if exp == 0 {
        sign * mant / scale * 2f64.powi(1 - bias)
    } else {
        sign * (1.0 + mant / scale) * 2f64.powi(exp - bias)
    })
}

/// `cvt.rn.satfinite[.relu]` of one value, by exhaustive nearest search.
fn reference_to_fp8(x: f32, e5m2: bool, relu: bool) -> u8 {
    if x.is_nan() {
        return 0x7f;
    }
    if relu && x.is_sign_negative() {
        return 0;
    }
    // Saturation: anything beyond the largest finite value (infinity included) clamps to it.
    let max = if e5m2 { 57344.0 } else { 448.0 };
    let magnitude = f64::from(x.abs()).min(max);
    let mut best = 0u8;
    let mut best_err = f64::INFINITY;
    for enc in 0u8..0x80 {
        let Some(v) = fp8_value(enc, e5m2).filter(|v| v.is_finite()) else { continue };
        let err = (v - magnitude).abs();
        if err < best_err || (err == best_err && enc & 1 == 0) {
            best = enc;
            best_err = err;
        }
    }
    best | if x.is_sign_negative() { 0x80 } else { 0 }
}

/// f16 bits of every fp8 magnitude (`0..0x80`), found by searching the f16 encodings.
fn f16_table(e5m2: bool) -> Vec<Option<u16>> {
    let h = super::sim::f16_to_f32;
    let mut by_value: std::collections::HashMap<u64, u16> = std::collections::HashMap::new();
    for b in (0u16..0x8000).rev() {
        by_value.insert(f64::from(h(u32::from(b))).to_bits(), b);
    }
    (0u8..0x80).map(|x| fp8_value(x, e5m2).map(|v| by_value[&v.to_bits()])).collect()
}

/// Reference widening of one fp8 byte to f16 bits.
fn reference_to_f16(table: &[Option<u16>], x: u8, e5m2: bool, relu: bool) -> u16 {
    let negative = x & 0x80 != 0;
    match table[usize::from(x & 0x7f)] {
        None if e5m2 && !relu => u16::from(x) << 8,
        None => 0x7fff,
        Some(_) if relu && negative => 0,
        Some(bits) => bits | if negative { 0x8000 } else { 0 },
    }
}

/// Inputs for narrowing: every f16 value widened to f32, plus every rounding midpoint between
/// adjacent fp8 magnitudes (and the overflow midpoint) with its f32 neighbours, both signs.
fn narrowing_inputs(e5m2: bool) -> Vec<f32> {
    let mut inputs: Vec<f32> = (0u32..=0xffff).map(super::sim::f16_to_f32).collect();
    let mut magnitudes: Vec<f64> = (0u8..0x80).filter_map(|e| fp8_value(e, e5m2)).filter(|v| v.is_finite()).collect();
    let max = *magnitudes.last().unwrap();
    // The value one step above the largest finite one (its midpoint decides saturation).
    magnitudes.push(if e5m2 { 61440.0 } else { 480.0 });
    for pair in magnitudes.windows(2) {
        let mid = ((pair[0] + pair[1]) / 2.0) as f32;
        for v in [mid, f32::from_bits(mid.to_bits() - 1), f32::from_bits(mid.to_bits() + 1)] {
            inputs.extend([v, -v]);
        }
    }
    let extremes = [max as f32 * 4.0, 1e30, f32::MAX, f32::INFINITY, f32::MIN_POSITIVE, 1e-40, 1e-45, f32::NAN, -f32::NAN];
    for v in extremes {
        inputs.extend([v, -v]);
    }
    if inputs.len() % 512 != 0 {
        inputs.resize(inputs.len().next_multiple_of(512), 0.0);
    }
    inputs
}

fn check_narrowing(e5m2: bool, relu: bool) {
    let ty = if e5m2 { "e5m2x2" } else { "e4m3x2" };
    let relu_mod = if relu { ".relu" } else { "" };
    let inputs = narrowing_inputs(e5m2);
    let pairs = inputs.len() / 2;
    let bytes: Vec<u8> = inputs.iter().flat_map(|f| f.to_le_bytes()).collect();
    let body = format!(
        "mul.wide.u32 %rd3, %r4, 8;\nadd.s64 %rd4, %rd1, %rd3;\n\
         ld.global.f32 %f1, [%rd4];\nld.global.f32 %f2, [%rd4+4];\n\
         cvt.rn.satfinite{relu_mod}.{ty}.f32 %h1, %f1, %f2;\n\
         mul.wide.u32 %rd5, %r4, 2;\nadd.s64 %rd6, %rd2, %rd5;\nst.global.b16 [%rd6], %h1;"
    );
    let out = run(&body, &bytes, 2 * pairs, pairs as u32);
    for i in 0..pairs {
        let (a, b) = (inputs[2 * i], inputs[2 * i + 1]);
        let want = u16::from(reference_to_fp8(a, e5m2, relu)) << 8 | u16::from(reference_to_fp8(b, e5m2, relu));
        let got = u16::from_le_bytes([out[2 * i], out[2 * i + 1]]);
        assert_eq!(got, want, "{ty}{relu_mod} of ({a:e} [{:#010x}], {b:e} [{:#010x}])", a.to_bits(), b.to_bits());
    }
}

#[test]
fn fp8_e4m3_narrowing_matches_reference() {
    check_narrowing(false, false);
}

#[test]
fn fp8_e5m2_narrowing_matches_reference() {
    check_narrowing(true, false);
}

#[test]
fn fp8_relu_narrowing_matches_reference() {
    check_narrowing(false, true);
    check_narrowing(true, true);
}

/// `cvt.rn.satfinite.{e4m3x2,e5m2x2}.f16x2` over all 65536 f16 values (each in both lanes).
#[test]
fn fp8_narrowing_from_f16x2_is_exhaustive() {
    for e5m2 in [false, true] {
        let ty = if e5m2 { "e5m2x2" } else { "e4m3x2" };
        // Thread i converts the pair (i, i ^ 0x8001) so every value meets both lanes.
        let words: Vec<u32> = (0u32..0x10000).map(|i| (i << 16) | (i ^ 0x8001)).collect();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let body = format!(
            "mul.wide.u32 %rd3, %r4, 4;\nadd.s64 %rd4, %rd1, %rd3;\nld.global.b32 %r5, [%rd4];\n\
             cvt.rn.satfinite.{ty}.f16x2 %h1, %r5;\n\
             mul.wide.u32 %rd5, %r4, 2;\nadd.s64 %rd6, %rd2, %rd5;\nst.global.b16 [%rd6], %h1;"
        );
        let out = run(&body, &bytes, 2 * words.len(), words.len() as u32);
        let lane = |h: u32| reference_to_fp8(super::sim::f16_to_f32(h & 0xffff), e5m2, false);
        for (i, w) in words.iter().enumerate() {
            let want = u16::from(lane(w >> 16)) << 8 | u16::from(lane(*w));
            let got = u16::from_le_bytes([out[2 * i], out[2 * i + 1]]);
            assert_eq!(got, want, "{ty}.f16x2 of {w:#010x}");
        }
    }
}

/// `cvt.rn[.relu].f16x2.{e4m3x2,e5m2x2}` over all 65536 fp8 pairs.
#[test]
fn fp8_widening_is_exhaustive() {
    let bytes: Vec<u8> = (0u32..0x10000).flat_map(|i| (i as u16).to_le_bytes()).collect();
    for e5m2 in [false, true] {
        for relu in [false, true] {
            let ty = if e5m2 { "e5m2x2" } else { "e4m3x2" };
            let relu_mod = if relu { ".relu" } else { "" };
            let body = format!(
                "mul.wide.u32 %rd3, %r4, 2;\nadd.s64 %rd4, %rd1, %rd3;\nld.global.b16 %h1, [%rd4];\n\
                 cvt.rn{relu_mod}.f16x2.{ty} %r5, %h1;\n\
                 mul.wide.u32 %rd5, %r4, 4;\nadd.s64 %rd6, %rd2, %rd5;\nst.global.b32 [%rd6], %r5;"
            );
            let out = run(&body, &bytes, 4 * 0x10000, 0x10000);
            let table = f16_table(e5m2);
            for i in 0u32..0x10000 {
                let want = u32::from(reference_to_f16(&table, (i >> 8) as u8, e5m2, relu)) << 16
                    | u32::from(reference_to_f16(&table, i as u8, e5m2, relu));
                let k = 4 * i as usize;
                let got = u32::from_le_bytes([out[k], out[k + 1], out[k + 2], out[k + 3]]);
                assert_eq!(got, want, "f16x2{relu_mod}.{ty} of {i:#06x}");
            }
        }
    }
}

/// The ZLUDA vectors for the four corpus kernels.
#[test]
fn fp8_zluda_vectors() {
    let narrow = |ty: &str, a: f32, b: f32| -> u16 {
        let bytes: Vec<u8> = [a, b].iter().flat_map(|f| f.to_le_bytes()).collect();
        let body = format!(
            "ld.global.f32 %f1, [%rd1];\nld.global.f32 %f2, [%rd1+4];\n\
             cvt.rn.satfinite.{ty}.f32 %h1, %f1, %f2;\nst.global.b16 [%rd2], %h1;"
        );
        let out = run(&body, &bytes, 2, 1);
        u16::from_le_bytes([out[0], out[1]])
    };
    let widen = |ty: &str, x: u16| -> u32 {
        let body = format!("ld.global.b16 %h1, [%rd1];\ncvt.rn.f16x2.{ty} %r5, %h1;\nst.global.b32 [%rd2], %r5;");
        let out = run(&body, &x.to_le_bytes(), 4, 1);
        u32::from_le_bytes([out[0], out[1], out[2], out[3]])
    };
    assert_eq!(narrow("e4m3x2", 0.40625, 12.9), 0x2D55);
    assert_eq!(narrow("e5m2x2", 0.375, -5256.6), 0x36ED);
    assert_eq!(widen("e4m3x2", 0x2D55), 0x3680_4a80);
    assert_eq!(widen("e5m2x2", 0x36ED), 0x3600_ED00);
}

#[test]
fn fp8_unsupported_forms_are_rejected() {
    for line in [
        "cvt.rn.e4m3x2.f32 %h1, %f1, %f2;",     // .satfinite is mandatory
        "cvt.rz.satfinite.e5m2x2.f32 %h1, %f1, %f2;",
        "cvt.rn.satfinite.e4m3x2.f32 %h1, %f1;",
    ] {
        let src = format!(
            "{HEADER}.visible .entry k(.param .u64 p)\n{{\n.reg .b16 %h<4>;\n.reg .f32 %f<4>;\n{line}\nret;\n}}\n"
        );
        let err = translate(&src).expect_err(line).to_string();
        assert!(err.contains("unsupported PTX instruction"), "{line}: {err}");
    }
}

/// Vector registers: lanes, whole-vector loads/stores, pack/unpack moves, ranges and `.func`
/// vector parameters.
#[test]
fn vector_registers_lower_to_lanes() {
    let src = format!(
        "{HEADER}.func (.reg .v2 .u32 %o) swap(.reg .v2 .u32 %i)\n{{\n\
         mov.v2.u32 %o, {{%i.y, %i.x}};\nret;\n}}\n\
         .visible .entry k(.param .u64 pin, .param .u64 pout)\n{{\n\
         .reg .b64 %rd<8>;\n.reg .b32 %r<8>;\n.reg .v4 .u32 %v;\n.reg .v2 .u32 %w;\n.reg .v2.b16 %h<2>;\n\
         ld.param.u64 %rd1, [pin];\nld.param.u64 %rd2, [pout];\n\
         ld.global.v4.u32 %v, [%rd1];\n\
         mov.v4.u32 %v, {{%v.w, %v.x, %v.y, %v.z}};\n\
         st.global.v4.u32 [%rd2], %v;\n\
         mov.v2.u32 %w, {{%v.r, %v.g}};\n\
         call (%w), swap, (%w);\n\
         mov.b64 %rd3, %w;\nst.global.u64 [%rd2+16], %rd3;\n\
         mov.b32 %h1, %v.b;\nmov.b32 %r1, {{%h1.y, %h1.x}};\nst.global.u32 [%rd2+24], %r1;\n\
         ret;\n}}\n"
    );
    let t = translate(&src).unwrap_or_else(|e| panic!("{e}"));
    let module = rspirv::dr::load_words(&t.words).unwrap();
    let mut gpu = Gpu::default();
    let input: Vec<u8> = [0x11u32, 0x22, 0x3333_4444, 0x55].iter().flat_map(|w| w.to_le_bytes()).collect();
    let input = gpu.upload(&input);
    let out = gpu.alloc(28);
    Sim::new(&module).run(&mut gpu, &Launch::new([1, 1, 1], [1, 1, 1], vec![input, out])).unwrap();
    let w = |k: u64| gpu.mem.read(out + 4 * k, 4) as u32;
    // Rotated in parallel: {w, x, y, z}.
    assert_eq!([w(0), w(1), w(2), w(3)], [0x55, 0x11, 0x22, 0x3333_4444]);
    // %w = {0x55, 0x11}, swapped by the call, packed low lane first.
    assert_eq!([w(4), w(5)], [0x11, 0x55]);
    // %v.b = 0x22 unpacked into halves (0x22, 0) and repacked swapped.
    assert_eq!(w(6), 0x0022_0000);
}

#[test]
fn vector_v8_and_parallel_moves() {
    let input: Vec<u8> = (1u8..=32).collect();
    // ld/st.v8.u32 and .v8.u8 split into two .v4 halves; mov.v8 rotates in parallel.
    let body = "ld.global.v8.u32 {%r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8}, [%rd1];\n\
                st.global.v8.u32 [%rd2], {%r8, %r7, %r6, %r5, %r4, %r3, %r2, %r1};\n\
                ld.global.v8.u8 {%h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7}, [%rd1];\n\
                mov.v8.u16 {%h1, %h2, %h3, %h4, %h5, %h6, %h7, %h0}, {%h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7};\n\
                st.global.v8.u8 [%rd2+32], {%h0, %h1, %h2, %h3, %h4, %h5, %h6, %h7};";
    let out = run(&body, &input, 40, 1);
    let word = |k: usize| u32::from_le_bytes([out[4 * k], out[4 * k + 1], out[4 * k + 2], out[4 * k + 3]]);
    let input_word = |k: u32| u32::from_le_bytes([(4 * k + 1) as u8, (4 * k + 2) as u8, (4 * k + 3) as u8, (4 * k + 4) as u8]);
    for k in 0..8 {
        assert_eq!(word(k), input_word(7 - k as u32), "word {k}");
    }
    assert_eq!(&out[32..40], &[8, 1, 2, 3, 4, 5, 6, 7]);
}
