//! Regression tests for translator bugs found by vosen/ptx_tests (tests/ptx_tests). Each runs a
//! small kernel on the SPIR-V simulator; the test name of the upstream case is in the comment.
use super::sim::{Gpu, Launch, Sim};
use super::translate;

/// Runs `body` on one thread with `%in`/`%out` global pointers; returns `out_len` bytes.
fn run(body: &str, regs: &str, input: &[u8], out_len: usize) -> Vec<u8> {
    let src = format!(
        ".version 7.8\n.target sm_90\n.address_size 64\n\
         .visible .entry k(.param .u64 k_in, .param .u64 k_out)\n{{\n\
         .reg .b64 %in; .reg .b64 %out;\n{regs}\n\
         ld.param.u64 %in, [k_in];\nld.param.u64 %out, [k_out];\n{body}\nret;\n}}\n"
    );
    run_source(&src, input, out_len)
}

fn run_source(src: &str, input: &[u8], out_len: usize) -> Vec<u8> {
    let t = translate(src).unwrap_or_else(|e| panic!("{e}\n{src}"));
    let module = rspirv::dr::load_words(&t.words).unwrap();
    let mut gpu = Gpu::default();
    let inp = gpu.upload(input);
    let out = gpu.alloc(out_len);
    Sim::new(&module).run(&mut gpu, &Launch::new([1, 1, 1], [1, 1, 1], vec![inp, out])).unwrap();
    gpu.download(out, out_len)
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
}

fn le(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// Loads `n` u32 inputs into `%a0..`, runs `op` on each (`{d}`, `{a}` placeholders) into
/// `%d0..` and stores the results.
fn unary(op: &str, regs: &str, inputs: &[u32]) -> Vec<u32> {
    let mut body = String::new();
    for i in 0..inputs.len() {
        body.push_str(&format!("ld.global.u32 %a{i}, [%in+{}];\n", 4 * i));
        body.push_str(&op.replace("{d}", &format!("%d{i}")).replace("{a}", &format!("%a{i}")));
        body.push_str(&format!("\nst.global.u32 [%out+{}], %d{i};\n", 4 * i));
    }
    let n = inputs.len();
    let regs = format!(".reg .b32 %a<{n}>; .reg .b32 %d<{n}>; {regs}");
    words(&run(&body, &regs, &le(inputs), 4 * n))
}

#[test]
fn bare_register_names_translate() {
    // Every ptx_tests kernel: `.reg .u64 input_addr;` without the `%` prefix is valid PTX.
    let src = ".version 7.8\n.target sm_90\n.address_size 64\n\
               .entry run(.param .u64 input, .param .u64 output)\n{\n\
               .reg .u64 input_addr; ld.param.u64 input_addr, [input];\n\
               .reg .u64 output_addr; ld.param.u64 output_addr, [output];\n\
               .reg .u32 value; .reg .u32 value2; .reg .pred p;\n\
               ld.global.u32 value, [input_addr];\n\
               setp.gt.u32 p, value, 4;\n\
               @p add.u32 value2, value, 1;\n\
               @!p mov.u32 value2, 0;\n\
               st.global.u32 [output_addr], value2;\nret;\n}";
    assert_eq!(words(&run_source(src, &le(&[41]), 4)), [42]);
    assert_eq!(words(&run_source(src, &le(&[3]), 4)), [0]);
}

#[test]
fn cvt_sat_from_signed_16_bit_clamps_negative_to_zero() {
    // cvt_sat_u32_s16, cvt_sat_u64_s16: the .sat clamp was dropped for 32/64-bit destinations.
    let regs = ".reg .s16 %h; .reg .u64 %rd;";
    let got = unary("cvt.u16.u32 %h, {a};\ncvt.sat.u32.s16 {d}, %h;", regs, &[0xFFFF, 0x8000, 0x7FFF, 5]);
    assert_eq!(got, [0, 0, 0x7FFF, 5]);
    let got = unary("cvt.u16.u32 %h, {a};\ncvt.sat.u64.s16 %rd, %h;\ncvt.u32.u64 {d}, %rd;", regs, &[0xFFFB, 7]);
    assert_eq!(got, [0, 7]);
}

#[test]
fn cvt_integer_rounding_to_a_float_destination_rounds() {
    // cvt_rni_f16_f16, cvt_rzi_sat_f32_f32, cvt_rpi_sat_f32_f32: the integer rounding was ignored.
    let regs = ".reg .b16 %h<2>;";
    let half = |op: &str, x: u32| {
        unary(&format!("cvt.u16.u32 %h0, {{a}};\n{op} %h1, %h0;\ncvt.u32.u16 {{d}}, %h1;"), regs, &[x])[0]
    };
    assert_eq!(half("cvt.rni.f16.f16", 0x3E00), 0x4000, "1.5 -> 2");
    assert_eq!(half("cvt.rni.f16.f16", 0x4100), 0x4000, "2.5 -> 2 (ties to even)");
    assert_eq!(half("cvt.rzi.f16.f16", 0xBE00), 0xBC00, "-1.5 -> -1");
    assert_eq!(half("cvt.rni.f16.f16", 0x0001), 0x0000, "subnormal -> 0");
    assert_eq!(half("cvt.rmi.sat.f16.f16", 0x3E00), 0x3C00, "1.5 -> 1");
    let f = |op: &str, x: f32| f32::from_bits(unary(&format!("{op} {{d}}, {{a}};"), "", &[x.to_bits()])[0]);
    assert_eq!(f("cvt.rzi.sat.f32.f32", 0.75), 0.0);
    assert_eq!(f("cvt.rni.sat.f32.f32", 0.75), 1.0);
    assert_eq!(f("cvt.rpi.sat.f32.f32", f32::from_bits(1)), 1.0);
    assert_eq!(f("cvt.rmi.sat.f32.f32", 2.5), 1.0);
}

#[test]
fn testp_normal_is_true_for_zero() {
    // testp_normal_f32: NVIDIA hardware reports +-0 as normal.
    let got = unary("testp.normal.f32 %p, {a};\nselp.u32 {d}, 1, 0, %p;", ".reg .pred %p;", &[0, 0x8000_0000, 1, 0x3F80_0000, 0x7F80_0000, 0x7FC0_0000]);
    assert_eq!(got, [1, 1, 0, 1, 0, 0]);
}

#[test]
fn cvt_between_32_and_64_bits_extends_by_the_source_type() {
    // cvt.u64.s32 and cvt.s64.u32 were unsupported.
    let regs = ".reg .b64 %rd;";
    let lo_hi = |op: &str, x: u32| {
        let got = unary(&format!("{op} %rd, {{a}};\nshr.u64 %rd, %rd, 32;\ncvt.u32.u64 {{d}}, %rd;"), regs, &[x]);
        got[0]
    };
    assert_eq!(lo_hi("cvt.u64.s32", 0xFFFF_FFFE), 0xFFFF_FFFF);
    assert_eq!(lo_hi("cvt.s64.u32", 0xFFFF_FFFE), 0);
}

#[test]
fn cvt_ftz_from_an_integer_is_accepted() {
    // cvt_rn_ftz_f32_s32 and friends: .ftz is meaningless on an integer source.
    let got = unary("cvt.rn.ftz.f32.s32 {d}, {a};", "", &[(-3i32) as u32]);
    assert_eq!(f32::from_bits(got[0]), -3.0);
    let got = unary("cvt.rn.ftz.f32.u32 {d}, {a};", "", &[7]);
    assert_eq!(f32::from_bits(got[0]), 7.0);
}

#[test]
fn float_set_with_a_boolean_combine() {
    // set_lt_and_f32_f32 and the other set.CMP.BOOL.f32.f32 forms were unsupported.
    let regs = ".reg .pred %p<2>; .reg .b32 %x;";
    let run_set = |op: &str, a: f32, b: f32, c: bool| {
        let body = format!(
            "ld.global.f32 %a0, [%in]; ld.global.f32 %x, [%in+4]; ld.global.u32 %d0, [%in+8];\n\
             setp.ne.u32 %p1, %d0, 0;\n{op} %d0, %a0, %x, %p1;\nst.global.u32 [%out], %d0;"
        );
        let regs = format!(".reg .b32 %a0; .reg .b32 %d0; {regs}");
        words(&run(&body, &regs, &le(&[a.to_bits(), b.to_bits(), c as u32]), 4))[0]
    };
    assert_eq!(run_set("set.lt.and.f32.f32", 1.0, 2.0, true), 0x3F80_0000);
    assert_eq!(run_set("set.lt.and.f32.f32", 1.0, 2.0, false), 0);
    assert_eq!(run_set("set.gtu.or.ftz.f32.f32", 1.0, f32::NAN, false), 0x3F80_0000);
    assert_eq!(run_set("set.eq.xor.u32.f32", 1.0, 1.0, true), 0);
    assert_eq!(run_set("set.eq.xor.s32.f32", 1.0, 2.0, true), u32::MAX);
    let negated = |c: bool| {
        let body = "ld.global.f32 %a0, [%in]; ld.global.f32 %x, [%in+4]; ld.global.u32 %d0, [%in+8];\n\
                    setp.ne.u32 %p1, %d0, 0;\nsetp.ge.or.f32 %p0, %a0, %x, !%p1;\nselp.u32 %d0, 1, 0, %p0;\n\
                    st.global.u32 [%out], %d0;";
        let regs = ".reg .b32 %a0; .reg .b32 %d0; .reg .pred %p<2>; .reg .b32 %x;";
        words(&run(body, regs, &le(&[1.0f32.to_bits(), 2.0f32.to_bits(), c as u32]), 4))[0]
    };
    assert_eq!(negated(true), 0);
    assert_eq!(negated(false), 1);
}

#[test]
fn directed_rounding_with_saturation() {
    // add_rz_sat_f32 and the other {add,sub,mul,fma}.{rz,rm,rp}.sat.f32 forms were unsupported.
    let pair = |op: &str, a: f32, b: f32| {
        let body = format!("ld.global.f32 %a0, [%in]; ld.global.f32 %x, [%in+4];\n{op} %d0, %a0, %x;\nst.global.u32 [%out], %d0;");
        words(&run(&body, ".reg .b32 %a0; .reg .b32 %d0; .reg .b32 %x;", &le(&[a.to_bits(), b.to_bits()]), 4))[0]
    };
    let tiny = f32::from_bits(0x3080_0000); // 2^-30
    assert_eq!(pair("add.rz.sat.f32", 1.0, -tiny), 0x3F7F_FFFF);
    assert_eq!(pair("add.rp.sat.f32", 1.0, -tiny), 0x3F80_0000);
    assert_eq!(pair("sub.rm.sat.f32", 1.0, tiny), 0x3F7F_FFFF);
    assert_eq!(pair("mul.rm.sat.f32", 3.0, 2.0), 0x3F80_0000, "clamped to 1");
    assert_eq!(pair("add.rz.sat.f32", -1.0, 0.5), 0, "clamped to 0");
    assert_eq!(pair("mul.rp.sat.f32", f32::NAN, 1.0), 0, "NaN -> 0");
    let body = "ld.global.f32 %a0, [%in]; ld.global.f32 %x, [%in+4];\nfma.rz.sat.f32 %d0, %a0, %x, %a0;\nst.global.u32 [%out], %d0;";
    let got = words(&run(body, ".reg .b32 %a0; .reg .b32 %d0; .reg .b32 %x;", &le(&[1.0f32.to_bits(), (-tiny).to_bits()]), 4));
    assert_eq!(got[0], 0x3F7F_FFFF);
}

#[test]
fn directed_division_square_root_and_reciprocal() {
    // div_rz_f32, sqrt_rp, rcp_rm and the other directed f32 forms were unsupported.
    let pair = |op: &str, a: f32, b: f32| {
        let body = format!("ld.global.f32 %a0, [%in]; ld.global.f32 %x, [%in+4];\n{op} %d0, %a0, %x;\nst.global.u32 [%out], %d0;");
        words(&run(&body, ".reg .b32 %a0; .reg .b32 %d0; .reg .b32 %x;", &le(&[a.to_bits(), b.to_bits()]), 4))[0]
    };
    // 1/3 = 0x3EAAAAAA.AAA..., to nearest 0x3EAAAAAB.
    assert_eq!(pair("div.rz.f32", 1.0, 3.0), 0x3EAA_AAAA);
    assert_eq!(pair("div.rm.f32", 1.0, 3.0), 0x3EAA_AAAA);
    assert_eq!(pair("div.rp.f32", 1.0, 3.0), 0x3EAA_AAAB);
    assert_eq!(pair("div.rm.f32", -1.0, 3.0), 0xBEAA_AAAB);
    assert_eq!(pair("div.rz.f32", 6.0, 3.0), 0x4000_0000, "exact quotients stay exact");
    assert_eq!(pair("div.rz.f32", 3.0e38, 0.5), 0x7F7F_FFFF, "overflow toward zero is the largest finite");
    assert_eq!(pair("div.rp.f32", 3.0e38, 0.5), 0x7F80_0000);
    let one = |op: &str, a: f32| unary(&format!("{op} {{d}}, {{a}};"), "", &[a.to_bits()])[0];
    // sqrt(2) = 0x3FB504F3.3..., to nearest 0x3FB504F3.
    assert_eq!(one("sqrt.rz.f32", 2.0), 0x3FB5_04F3);
    assert_eq!(one("sqrt.rp.f32", 2.0), 0x3FB5_04F4);
    assert_eq!(one("sqrt.rm.f32", 4.0), 0x4000_0000);
    assert_eq!(one("rcp.rm.f32", 3.0), 0x3EAA_AAAA);
    assert_eq!(one("rcp.rp.f32", 3.0), 0x3EAA_AAAB);
    assert!(f32::from_bits(one("sqrt.rz.f32", -1.0)).is_nan());
}

#[test]
fn sixteen_bit_mad_hi_and_sad_and_saturating_mad_hi() {
    // mad_hi_u16, mad_hi_s16, sad_u16, sad_s16 and mad_hi_sat_s32 were unsupported.
    let three = |op: &str, ty: &str, a: u32, b: u32, c: u32| {
        let body = format!(
            "ld.global.u32 %r0, [%in]; ld.global.u32 %r1, [%in+4]; ld.global.u32 %r2, [%in+8];\n\
             cvt.u16.u32 %h0, %r0; cvt.u16.u32 %h1, %r1; cvt.u16.u32 %h2, %r2;\n\
             {op}.{ty} %h3, %h0, %h1, %h2;\ncvt.u32.u16 %r3, %h3;\nst.global.u32 [%out], %r3;"
        );
        words(&run(&body, ".reg .b32 %r<4>; .reg .b16 %h<4>;", &le(&[a, b, c]), 4))[0]
    };
    assert_eq!(three("mad.hi", "u16", 0xFFFF, 0xFFFF, 1), 0xFFFF);
    assert_eq!(three("mad.hi", "s16", 0xFFFE, 3, 5), 4, "hi(-6) = -1, + 5");
    assert_eq!(three("sad", "u16", 3, 10, 1), 8);
    assert_eq!(three("sad", "s16", 0xFFFD, 4, 0), 7);
    let body = "ld.global.u32 %r0, [%in]; ld.global.u32 %r1, [%in+4]; ld.global.u32 %r2, [%in+8];\n\
                mad.hi.sat.s32 %r3, %r0, %r1, %r2;\nst.global.u32 [%out], %r3;";
    let sat = |a: u32, b: u32, c: u32| words(&run(body, ".reg .b32 %r<4>;", &le(&[a, b, c]), 4))[0];
    assert_eq!(sat(0x7FFF_FFFF, 0x7FFF_FFFF, 0x7FFF_FFFF), 0x7FFF_FFFF);
    assert_eq!(sat(0x7FFF_FFFF, 0x8000_0000, 0x8000_0000), 0x8000_0000);
    assert_eq!(sat(0x0001_0000, 0x0003_0000, 2), 5);
}

#[test]
fn div_approx_returns_zero_for_huge_divisors() {
    // div_approx_f32: for 2^126 < |b| the result is 0 (NaN for infinite a); div.full is precise.
    let pair = |op: &str, a: f32, b: f32| {
        let body = format!("ld.global.f32 %a0, [%in]; ld.global.f32 %x, [%in+4];\n{op} %d0, %a0, %x;\nst.global.u32 [%out], %d0;");
        words(&run(&body, ".reg .b32 %a0; .reg .b32 %d0; .reg .b32 %x;", &le(&[a.to_bits(), b.to_bits()]), 4))[0]
    };
    let huge = f32::from_bits(0x7F00_0000); // 2^127
    assert_eq!(pair("div.approx.f32", 1.0, huge) & 0x7FFF_FFFF, 0);
    assert!(f32::from_bits(pair("div.approx.ftz.f32", f32::INFINITY, -huge)).is_nan());
    assert_eq!(pair("div.approx.f32", 6.0, 3.0), 0x4000_0000);
    assert_eq!(pair("div.full.f32", 1.0, huge), 0x0040_0000, "2^-127 is subnormal, not flushed");
}

#[test]
fn simulator_models_directed_conversion_to_half() {
    // cvt_rm_f16_f32 and cvt_rp_f16_f32 stopped the simulator ("unmodelled FConvert rounding").
    let regs = ".reg .b16 %h;";
    let half = |op: &str, x: f32| unary(&format!("{op} %h, {{a}};\ncvt.u32.u16 {{d}}, %h;"), regs, &[x.to_bits()])[0];
    let x = 1.0 + f32::from_bits(0x3980_0000); // 1 + 2^-12, between f16 1.0 and 1.0009765625
    assert_eq!(half("cvt.rm.f16.f32", x), 0x3C00);
    assert_eq!(half("cvt.rp.f16.f32", x), 0x3C01);
    assert_eq!(half("cvt.rm.f16.f32", -x), 0xBC01);
    assert_eq!(half("cvt.rp.f16.f32", -x), 0xBC00);
    assert_eq!(half("cvt.rm.f16.f32", 1.0e6), 0x7BFF, "overflow toward -inf is the largest finite");
    assert_eq!(half("cvt.rp.f16.f32", 1.0e6), 0x7C00);
}

#[test]
fn exact_ftz_flushes_subnormal_sources_and_results() {
    // add_rn_ftz_f32, min_ftz (f16), cvt_ftz_f32_f32, setp/set .ftz: with CFI_EXACT_FTZ=1 subnormals
    // become signed zeros; by default .ftz is ignored.
    struct Reset;
    impl Drop for Reset {
        fn drop(&mut self) {
            super::set_exact_ftz(None);
        }
    }
    let _reset = Reset;
    let sub = 0x0000_0001u32; // smallest positive f32 subnormal
    let neg_sub = 0x8000_0010u32;
    let pair = |op: &str, a: u32, b: u32| {
        let body = format!("ld.global.u32 %a0, [%in]; ld.global.u32 %x, [%in+4];\n{op} %d0, %a0, %x;\nst.global.u32 [%out], %d0;");
        words(&run(&body, ".reg .b32 %a0; .reg .b32 %d0; .reg .b32 %x;", &le(&[a, b]), 4))[0]
    };
    super::set_exact_ftz(Some(false));
    assert_eq!(pair("add.rn.ftz.f32", sub, 0), sub, "ignored by default");
    super::set_exact_ftz(Some(true));
    assert_eq!(pair("add.rn.ftz.f32", sub, 0), 0);
    assert_eq!(pair("add.rn.ftz.f32", neg_sub, 0x8000_0000), 0x8000_0000, "sign kept");
    assert_eq!(pair("mul.rn.ftz.f32", 0x1F80_0000, 0x1F80_0000), 0, "subnormal result flushed");
    assert_eq!(pair("mul.rn.ftz.f32", 0x3F80_0000, 0x4000_0000), 0x4000_0000, "normal values untouched");
    assert_eq!(pair("min.ftz.f32", neg_sub, sub) , 0x8000_0000);
    assert_eq!(pair("min.ftz.f16", 0x3B87, 0x8351), 0x8000, "f16 lanes flush too");
    assert_eq!(pair("max.ftz.f16x2", 0x0001_3C00, 0x0000_0000), 0x0000_3C00, "per lane");
    let one = |op: &str, a: u32| unary(&format!("{op} {{d}}, {{a}};"), "", &[a])[0];
    assert_eq!(one("cvt.ftz.f32.f32", sub), 0);
    assert_eq!(one("abs.ftz.f32", neg_sub), 0);
    assert_eq!(one("sqrt.rn.ftz.f32", sub), 0);
    let body = "ld.global.u32 %a0, [%in]; ld.global.u32 %x, [%in+4];\nsetp.gt.ftz.f32 %p, %a0, %x;\n\
                selp.u32 %d0, 1, 0, %p;\nst.global.u32 [%out], %d0;";
    let got = words(&run(body, ".reg .b32 %a0; .reg .b32 %d0; .reg .b32 %x; .reg .pred %p;", &le(&[sub, 0]), 4));
    assert_eq!(got, [0], "subnormal compares equal to zero");
    let body = "ld.global.u32 %a0, [%in];\n@%p add.rn.ftz.f32 %d0, %a0, %a0;\nst.global.u32 [%out], %d0;";
    let got = words(&run(&format!("setp.ne.u32 %p, 0, 0;\nmov.b32 %d0, 7;\n{body}"), ".reg .b32 %a0; .reg .b32 %d0; .reg .pred %p;", &le(&[sub]), 4));
    assert_eq!(got, [7], "the guard still applies");
}

#[test]
fn round_down_of_an_exact_zero_sum_is_negative_zero() {
    // sub_rm_ftz_f32 on the B390: sub.rm.f32 of +0 and +0 gave +0; IEEE 754 gives -0 for every
    // exact zero sum under round-down except (+0) + (+0).
    let pair = |op: &str, a: u32, b: u32| {
        let body = format!("ld.global.u32 %a0, [%in]; ld.global.u32 %x, [%in+4];\n{op} %d0, %a0, %x;\nst.global.u32 [%out], %d0;");
        words(&run(&body, ".reg .b32 %a0; .reg .b32 %d0; .reg .b32 %x;", &le(&[a, b]), 4))[0]
    };
    assert_eq!(pair("sub.rm.f32", 0, 0), 0x8000_0000);
    assert_eq!(pair("add.rm.f32", 0, 0x8000_0000), 0x8000_0000);
    assert_eq!(pair("add.rm.f32", 0, 0), 0);
    assert_eq!(pair("sub.rm.f32", 0x3F80_0000, 0x3F80_0000), 0x8000_0000);
    assert_eq!(pair("sub.rz.f32", 0x3F80_0000, 0x3F80_0000), 0);
    assert_eq!(pair("add.rm.f32", 0x8000_0000, 0x8000_0000), 0x8000_0000);
}
