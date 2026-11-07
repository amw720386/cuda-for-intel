//! Tests for the tier-3 PTX lowering work (`tier3.rs` and friends). Executed on
//! the SPIR-V interpreter from `coverage_tests` and validated with `spirv-val`.
use super::coverage_tests::{interp, kernel, lower};

fn eval(regs: &str, body: &str, words: usize) -> Vec<u32> {
    let module = lower(&kernel(regs, body));
    let mut machine = interp::Machine {
        params: vec![interp::GLOBAL, 0],
        ..Default::default()
    };
    interp::run(&module, &mut machine);
    (0..words).map(|i| machine.read(interp::GLOBAL + 4 * i as u64, 4) as u32).collect()
}

fn stores(names: &[&str]) -> String {
    names.iter().enumerate().map(|(i, n)| format!("    st.global.u32 [%rd15+{}], {n};\n", 4 * i)).collect()
}

#[test]
fn carry_chain_32_bit_matches_reference() {
    let body = format!(
        r#"    mov.u32 %r1, 0xffffffff;
    mov.u32 %r2, 1;
    mov.u32 %r3, 7;
    mov.u32 %r4, 0;
    add.cc.u32 %r5, %r1, %r2;
    addc.u32 %r6, %r3, %r4;
    addc.cc.u32 %r7, %r3, %r4;
    addc.u32 %r8, %r4, %r4;
    sub.cc.u32 %r9, %r4, %r2;
    subc.u32 %r10, %r3, %r4;
    sub.cc.u32 %r11, %r3, %r2;
    subc.u32 %r12, %r3, %r4;
    mad.lo.cc.u32 %r13, %r1, %r3, %r1;
    madc.hi.u32 %r14, %r1, %r3, %r4;
{}"#,
        stores(&["%r5", "%r6", "%r7", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14"])
    );
    let out = eval("", &body, 10);
    // 0xffffffff + 1 -> 0 carry 1; 7 + 0 + 1 = 8; carry cleared by addc.cc; 0 + 0 + 0.
    let lo = 0xffff_ffffu32.wrapping_mul(7);
    let (m, c) = lo.overflowing_add(0xffff_ffff);
    let hi = ((0xffff_ffffu64 * 7) >> 32) as u32;
    assert_eq!(out, [0, 8, 8, 0, 0xffff_ffff, 6, 6, 7, m, hi + u32::from(c)]);
}

#[test]
fn carry_chain_64_bit() {
    let body = r#"    mov.u64 %rd1, 0xffffffffffffffff;
    mov.u64 %rd2, 1;
    mov.u64 %rd3, 5;
    add.cc.u64 %rd4, %rd1, %rd2;
    addc.u64 %rd5, %rd3, %rd2;
    st.global.u64 [%rd15], %rd4;
    st.global.u64 [%rd15+8], %rd5;"#;
    let out = eval("", body, 4);
    assert_eq!(out, [0, 0, 7, 0]);
}

#[test]
fn video_scalar_and_simd_match_reference() {
    let body = format!(
        r#"    mov.u32 %r1, 0x01ff7f80;
    mov.u32 %r2, 0x02017f81;
    vadd.u32.u32.u32 %r3, %r1.b0, %r2.b1, 0;
    vadd.s32.s32.s32.sat %r4, %r1.b0, %r2.b0, 0;
    vabsdiff.u32.u32.u32 %r5, %r1.h1, %r2.h0, 0;
    vmax.s32.s32.s32.add %r6, %r1.b3, %r2.b3, 10;
    vset.u32.u32.lt %r7, %r1.b0, %r2.b0, 0;
    vadd4.u32.u32.u32 %r8, %r1, %r2, 0;
    vabsdiff4.u32.u32.u32 %r9, %r1, %r2, 0;
    vmin2.u32.u32.u32 %r10, %r1, %r2, 0;
    vsub2.u32.u32.u32.add %r11, %r1, %r2, 3;
    vset4.u32.u32.eq %r12, %r1, %r2, 0;
    vavrg4.u32.u32.u32 %r13, %r1, %r2, 0;
    vadd4.s32.s32.s32.sat %r14, %r1, %r2, 0;
{}"#,
        stores(&["%r3", "%r4", "%r5", "%r6", "%r7", "%r8", "%r9", "%r10", "%r11", "%r12", "%r13", "%r14"])
    );
    let out = eval("", &body, 12);
    let a: u32 = 0x01ff7f80;
    let b: u32 = 0x02017f81;
    let lane8 = |v: u32, i: u32| (v >> (8 * i)) & 0xff;
    let lane16 = |v: u32, i: u32| (v >> (16 * i)) & 0xffff;
    let pack4 = |f: &dyn Fn(i32, i32) -> i32| (0..4).fold(0u32, |acc, i| acc | (((f(lane8(a, i) as i32, lane8(b, i) as i32)) as u32 & 0xff) << (8 * i)));
    let s8 = |v: u32| v as u8 as i8 as i32;
    let expect = [
        lane8(a, 0) + lane8(b, 1),
        (s8(0x80) + s8(0x81)).clamp(i32::MIN, i32::MAX) as u32,
        lane16(a, 1).abs_diff(lane16(b, 0)),
        (s8(lane8(a, 3)).max(s8(lane8(b, 3))) + 10) as u32,
        u32::from(lane8(a, 0) < lane8(b, 0)),
        pack4(&|x, y| x + y),
        pack4(&|x, y| (x - y).abs()),
        (lane16(a, 0).min(lane16(b, 0))) | (lane16(a, 1).min(lane16(b, 1)) << 16),
        (lane16(a, 0) as i32 - lane16(b, 0) as i32 + lane16(a, 1) as i32 - lane16(b, 1) as i32 + 3) as u32,
        pack4(&|x, y| i32::from(x == y)),
        pack4(&|x, y| (x + y + 1) >> 1),
        // signed saturating byte adds: 0x80+0x81 -> -128, 0x7f+0x7f -> 127, 0xff+0x01 -> 0, 0x01+0x02 -> 3
        (-128i32 as u32 & 0xff) | (127 << 8) | (0 << 16) | (3 << 24),
    ];
    assert_eq!(out[0], expect[0]);
    assert_eq!(out[1], expect[1]);
    assert_eq!(out[2], expect[2]);
    assert_eq!(out[3], expect[3]);
    assert_eq!(out[4], expect[4]);
    assert_eq!(out[5], expect[5]);
    assert_eq!(out[6], expect[6]);
    assert_eq!(out[7], expect[7]);
    assert_eq!(out[8], expect[8]);
    assert_eq!(out[9], expect[9]);
    assert_eq!(out[10], expect[10]);
    assert_eq!(out[11], expect[11]);
}

#[test]
fn setp_pairs_and_literal_selp() {
    let body = format!(
        r#"    mov.u32 %r1, 3;
    mov.u32 %r2, 9;
    setp.lt.u32 %p1|%p2, %r1, %r2;
    selp.u32 %r3, 1, 0, %p1;
    selp.u32 %r4, 1, 0, %p2;
    setp.lt.xor.u32 %p3|%p4, %r2, %r1, %p1;
    selp.u32 %r5, 1, 0, %p3;
    selp.u32 %r6, 1, 0, %p4;
    selp.u32 %r7, %r1, %r2, 1;
    selp.u32 %r8, %r1, %r2, 0;
{}"#,
        stores(&["%r3", "%r4", "%r5", "%r6", "%r7", "%r8"])
    );
    assert_eq!(eval("", &body, 6), [1, 0, 1, 0, 3, 9]);
}

#[test]
fn partial_masks_const_loads_and_pack() {
    let body = format!(
        r#"    mov.u32 %r1, 300;
    mov.u32 %r2, 0xffffff9c;
    cvt.pack.sat.u8.s32.b32 %r3, %r1, %r2, 0xab;
    cvt.pack.sat.s16.s32.b32 %r4, %r1, %r2, 0;
    cvt.s32.u32 %r5, %r1;
    setp.eq.u32 %p1, %r1, %r1;
    mov.u32 %r6, WARP_SZ;
{}"#,
        stores(&["%r3", "%r4", "%r5", "%r6"])
    );
    let out = eval("", &body, 4);
    assert_eq!(out[0], (0xab << 16) | (255 << 8));
    assert_eq!(out[1], (300 << 16) | (0xffff_ff9cu32 as i32).clamp(-32768, 32767) as u32 & 0xffff);
    assert_eq!(out[2], 300);
    assert_eq!(out[3], 32);
}

#[test]
fn partial_mask_collectives_translate_and_validate() {
    lower(&kernel(
        "",
        r#"    mov.u32 %r1, 300;
    setp.eq.u32 %p1, %r1, %r1;
    vote.sync.ballot.b32 %r6, %p1, 0x0000000f;
    vote.sync.any.pred %p2, %p1, 0x0000ffff;
    shfl.sync.bfly.b32 %r7, %r1, 1, 0x1c1f, 0x0000ffff;
    shfl.sync.up.b32 %r8, %r1, 1, 0x0, 0x00ff00ff;
    shfl.sync.idx.b32 %r9, %r1, 0, 0x1f, 0x00ff00ff;
    st.global.u32 [%rd15], %r7;
    st.global.u32 [%rd15+4], %r8;
    st.global.u32 [%rd15+8], %r9;
    st.global.u32 [%rd15+12], %r6;"#,
    ));
}

#[test]
fn constant_space_is_ordinary_memory() {
    let out = eval(
        "",
        r#"    mov.u32 %r1, 41;
    st.global.u32 [%rd15], %r1;
    cvta.const.u64 %rd1, %rd15;
    ld.const.u32 %r2, [%rd1];
    add.u32 %r3, %r2, 1;
    st.global.u32 [%rd15+4], %r3;"#,
        2,
    );
    assert_eq!(out, [41, 42]);
}
