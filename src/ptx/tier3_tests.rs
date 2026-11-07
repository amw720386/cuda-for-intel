//! Tests for the tier-3 PTX lowering work (`tier3.rs` and friends). Executed on
//! the SPIR-V interpreter from `coverage_tests` and validated with `spirv-val`.
use super::coverage_tests::{interp, kernel, kernel_with_shared, lower};
use super::translate;

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

#[test]
fn counted_barriers_and_bar_red_translate_and_validate() {
    lower(&kernel(
        "",
        r#"    setp.eq.u32 %p1, %r0, 0;
    bar.sync 1, 128;
    bar.arrive 2, 64;
    bar.sync 2, 64;
    bar.red.popc.u32 %r2, 0, %p1;
    bar.red.and.pred %p2, 0, %p1;
    bar.red.or.pred %p3, 0, !%p1;
    barrier.cta.sync 3, %r0;
    bar.warp.sync 0xffffffff;
    selp.u32 %r3, 1, 0, %p2;
    selp.u32 %r4, 1, 0, %p3;
    st.global.u32 [%rd15], %r2;
    st.global.u32 [%rd15+4], %r3;
    st.global.u32 [%rd15+8], %r4;"#,
    ));
}

#[test]
fn bar_red_single_thread_semantics() {
    // One interpreted thread: popc counts 1, and-of-true is true, or-of-false is false.
    let out = eval(
        "",
        r#"    mov.u32 %r1, 1;
    setp.eq.u32 %p1, %r1, 1;
    bar.red.popc.u32 %r2, 0, %p1;
    bar.red.and.pred %p2, 0, %p1;
    bar.red.or.pred %p3, 0, !%p1;
    selp.u32 %r3, 1, 0, %p2;
    selp.u32 %r4, 1, 0, %p3;
    st.global.u32 [%rd15], %r2;
    st.global.u32 [%rd15+4], %r3;
    st.global.u32 [%rd15+8], %r4;"#,
        3,
    );
    assert_eq!(out, [1, 1, 0]);
}

#[test]
fn several_local_arrays_and_symbol_offsets() {
    let out = super::coverage_tests::kernel_with_shared(
        "",
        "",
        r#"    mov.u32 %r1, 11;
    mov.u32 %r2, 22;
    st.local.u32 [la+4], %r1;
    st.local.u32 [lb], %r2;
    mov.u64 %rd1, lb;
    ld.local.u32 %r3, [%rd1];
    ld.local.u32 %r4, [la+4];
    st.global.u32 [%rd15], %r3;
    st.global.u32 [%rd15+4], %r4;"#,
    )
    .replace("    .reg .pred %p<8>;", "    .local .align 4 .b8 la[12];\n    .local .align 8 .b8 lb[6];\n    .reg .pred %p<8>;");
    let module = lower(&out);
    let mut machine = interp::Machine { params: vec![interp::GLOBAL, 0], ..Default::default() };
    interp::run(&module, &mut machine);
    assert_eq!([machine.read(interp::GLOBAL, 4), machine.read(interp::GLOBAL + 4, 4)], [22, 11]);
}

#[test]
fn shared_symbols_with_offsets_and_64_bit_accesses() {
    let source = super::coverage_tests::kernel_with_shared(
        "",
        ".shared .align 8 .b32 words[16];",
        r#"    mov.u64 %rd1, 0x0000000500000007;
    st.shared.u64 [words+8], %rd1;
    ld.shared.u32 %r1, [words+8];
    ld.shared.u32 %r2, [words+12];
    st.global.u32 [%rd15], %r1;
    st.global.u32 [%rd15+4], %r2;"#,
    );
    let module = lower(&source);
    let mut machine = interp::Machine { params: vec![interp::GLOBAL, 0], ..Default::default() };
    interp::run(&module, &mut machine);
    assert_eq!([machine.read(interp::GLOBAL, 4), machine.read(interp::GLOBAL + 4, 4)], [7, 5]);
}

#[test]
fn stmatrix_scatters_row_addresses() {
    // Matrix 0 rows are 16 bytes apart in a 256-byte shared buffer; lane j supplies row j.
    let regs = "    .reg .b32 %m<4>;";
    let body = r#"    mov.u32 %r1, sb;
    mov.u32 %r2, %laneid;
    shl.b32 %r3, %r2, 4;
    add.u32 %r3, %r3, %r1;
    mov.u32 %m0, %r2;
    mov.u32 %m1, %r2;
    stmatrix.sync.aligned.m8n8.x2.shared.b16 [%r3], {%m0, %m1};
    stmatrix.sync.aligned.m8n8.x1.trans.shared.b16 [%r3+128], {%m0};
    bar.sync 0;
    mov.u32 %r4, %tid.x;
    ld.shared.u32 %r5, [%r3];
    st.global.u32 [%rd15], %r5;"#;
    lower(&kernel_with_shared(regs, ".shared .align 16 .b8 sb[512];", body));
}

#[test]
fn debug_sections_and_texture_errors() {
    let source = kernel("", "    st.global.u32 [%rd15], %r0;").replace(
        ".address_size 64",
        ".address_size 64\n.file 1 \"/tmp/a.cu\"\n.section .debug_abbrev\n{\n.b8 1, 17, 1\n}\n",
    );
    lower(&source);
    let tex = kernel("    .reg .f32 %t<4>;", "    tex.1d.v4.f32.f32 {%t0, %t1, %t2, %t3}, [%rd15, {%f0}];");
    let err = translate(&tex).unwrap_err().to_string();
    assert!(err.contains("unsupported") && err.contains("bindless"), "{err}");
}

fn run_stmatrix(trans: bool) -> Vec<u32> {
    let regs = "    .reg .b32 %m<4>;";
    let body = format!(
        r#"    mov.u32 %r1, sb;
    mov.u32 %r2, %laneid;
    shl.b32 %r3, %r2, 4;
    add.u32 %r3, %r3, %r1;
    shl.b32 %m0, %r2, 0;
    add.u32 %r6, %r2, 100;
    shl.b32 %r6, %r6, 16;
    or.b32 %m0, %m0, %r6;
    stmatrix.sync.aligned.m8n8.x1{}.shared.b16 [%r3], {{%m0}};
    bar.sync 0;
    shl.b32 %r4, %r2, 2;
    add.u32 %r4, %r4, %r1;
    ld.shared.u16 %r5, [%r4];
    ld.shared.u16 %r7, [%r4+2];
    mul.wide.u32 %rd1, %r2, 8;
    add.s64 %rd2, %rd15, %rd1;
    st.global.u32 [%rd2], %r5;
    st.global.u32 [%rd2+4], %r7;"#,
        if trans { ".trans" } else { "" }
    );
    let module = lower(&kernel_with_shared(regs, ".shared .align 16 .b8 sb[512];", &body));
    let mut machine = interp::Machine { params: vec![interp::GLOBAL, 0], ..Default::default() };
    interp::run_lanes(&module, &mut machine, 32);
    (0..64).map(|i| machine.read(interp::GLOBAL + 4 * i, 4) as u32).collect()
}

#[test]
fn stmatrix_places_fragments_like_hardware() {
    // Rows are 16 bytes apart (8 halves per row of the 8x8 b16 matrix, stored densely at stride 8 halves).
    let plain = run_stmatrix(false);
    // Memory half index h = 8*row + col; thread L = 4*row + col/2 holds (L, L+100) for cols 2*(L%4)+{0,1}.
    // Row addresses are 16 bytes apart but lanes read dense halves: row r starts at byte 16 r -> half 8 r.
    for row in 0..8usize {
        for col in 0..8usize {
            let lane = 4 * row + col / 2;
            let want = if col % 2 == 0 { lane as u32 } else { lane as u32 + 100 };
            assert_eq!(plain[8 * row + col], want, "plain ({row},{col})");
        }
    }
    let t = run_stmatrix(true);
    for row in 0..8usize {
        for col in 0..8usize {
            // Stored matrix element (row, col) comes from the register element (col, row).
            let lane = 4 * col + row / 2;
            let want = if row % 2 == 0 { lane as u32 } else { lane as u32 + 100 };
            assert_eq!(t[8 * row + col], want, "trans ({row},{col})");
        }
    }
}

#[test]
fn f64_directed_rounding_matches_exact_arithmetic() {
    let eps = f64::EPSILON; // 2^-52
    let up = |x: f64| f64::from_bits(x.to_bits() + 1);
    let down = |x: f64| f64::from_bits(x.to_bits() - 1);
    let x = 1.0 + eps;
    // (instruction, operands, expected result) with hand-derived exact results.
    let cases: Vec<(&str, Vec<f64>, f64)> = vec![
        ("add.rz.f64", vec![1.0, 2f64.powi(-60)], 1.0),
        ("add.rp.f64", vec![1.0, 2f64.powi(-60)], up(1.0)),
        ("add.rm.f64", vec![1.0, -(2f64.powi(-60))], down(1.0)),
        ("add.rz.f64", vec![-1.0, 2f64.powi(-60)], -down(1.0)),
        ("add.rp.f64", vec![-1.0, 2f64.powi(-60)], -down(1.0)),
        ("add.rm.f64", vec![-1.0, 2f64.powi(-60)], -1.0),
        ("sub.rp.f64", vec![1.0, -(2f64.powi(-60))], up(1.0)),
        ("mul.rp.f64", vec![x, x], up(1.0 + 2.0 * eps)),
        ("mul.rz.f64", vec![x, x], 1.0 + 2.0 * eps),
        ("mul.rm.f64", vec![x, x], 1.0 + 2.0 * eps),
        ("mul.rm.f64", vec![-x, x], -up(1.0 + 2.0 * eps)),
        ("div.rp.f64", vec![1.0, 3.0], up(1.0 / 3.0)),
        ("div.rz.f64", vec![1.0, 3.0], 1.0 / 3.0),
        ("div.rm.f64", vec![-1.0, 3.0], -up(1.0 / 3.0)),
        ("rcp.rm.f64", vec![3.0], 1.0 / 3.0),
        ("rcp.rp.f64", vec![3.0], up(1.0 / 3.0)),
        ("sqrt.rz.f64", vec![2.0], down(2f64.sqrt())),
        ("sqrt.rm.f64", vec![2.0], down(2f64.sqrt())),
        ("sqrt.rp.f64", vec![2.0], 2f64.sqrt()),
        ("fma.rp.f64", vec![x, x, 2f64.powi(-60)], up(1.0 + 2.0 * eps)),
        ("fma.rz.f64", vec![x, x, 2f64.powi(-60)], 1.0 + 2.0 * eps),
    ];
    let mut body = String::new();
    for (i, (op, operands, _)) in cases.iter().enumerate() {
        let regs: Vec<String> = operands
            .iter()
            .enumerate()
            .map(|(j, v)| {
                body.push_str(&format!("    mov.b64 %fd{}, 0x{:016x};\n", j + 1, v.to_bits()));
                format!("%fd{}", j + 1)
            })
            .collect();
        body.push_str(&format!("    {op} %fd7, {};\n    st.global.u64 [%rd15+{}], %fd7;\n", regs.join(", "), 8 * i));
    }
    let module = lower(&kernel("    .reg .b64 %fd<8>;", &body));
    let mut machine = interp::Machine { params: vec![interp::GLOBAL, 0], ..Default::default() };
    interp::run(&module, &mut machine);
    for (i, (op, operands, want)) in cases.iter().enumerate() {
        let got = f64::from_bits(machine.read(interp::GLOBAL + 8 * i as u64, 8));
        assert_eq!(got.to_bits(), want.to_bits(), "{op} {operands:?}: got {got:e} want {want:e}");
    }
}

#[test]
fn module_scope_const_and_global_data() {
    let source = r#".version 9.0
.target sm_80
.address_size 64
.const .align 4 .u32 ctbl[4] = {11, 22, 33, 0x2c};
.global .align 8 .b8 gtbl[8] = {1, 2, 3, 4, 5, 6, 7, 8};
.global .align 4 .f32 fv[2] = {0f3F800000, 0f40000000};
.visible .entry data(.param .u64 out_param)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<4>;
    ld.param.u64 %rd0, [out_param];
    cvta.to.global.u64 %rd1, %rd0;
    ld.const.u32 %r1, [ctbl+8];
    mov.u64 %rd2, ctbl;
    ld.const.u32 %r2, [%rd2+12];
    ld.global.u8 %r3, [gtbl+5];
    ld.global.f32 %f1, [fv+4];
    mov.u64 %rd3, gtbl+2;
    ld.global.u8 %r4, [%rd3];
    st.global.u32 [%rd1], %r1;
    st.global.u32 [%rd1+4], %r2;
    st.global.u32 [%rd1+8], %r3;
    st.global.f32 [%rd1+12], %f1;
    st.global.u32 [%rd1+16], %r4;
    ret;
}
"#;
    // Translation and spirv-val; the interpreter has no program-scope variables.
    let module = lower(source);
    assert!(super::coverage_tests::has_op(&module, rspirv::spirv::Op::Variable));
    assert!(module.types_global_values.iter().any(|i| i.class.opcode == rspirv::spirv::Op::Variable
        && i.operands.first() == Some(&rspirv::dr::Operand::StorageClass(rspirv::spirv::StorageClass::CrossWorkgroup))));
}

/// Debug helper: `PTX_IN=file.ptx cargo test show_normalized -- --ignored --nocapture`.
#[test]
#[ignore]
fn show_normalized() {
    let path = std::env::var("PTX_IN").unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    println!("{}", super::normalize::run(&text).unwrap());
}

/// Debug helper: prints the real parse/emit error that `translate` summarises.
#[test]
#[ignore]
fn show_translate_error() {
    let path = std::env::var("PTX_IN").unwrap();
    let text = std::fs::read_to_string(path).unwrap();
    let norm = super::prepare(&text).unwrap();
    println!("{norm}");
    match super::parse::parse_module(&norm) {
        Ok((entries, vars)) => match super::spirv::emit(entries, vars.device) {
            Ok(_) => println!("OK"),
            Err(e) => println!("EMIT ERROR {e}"),
        },
        Err(e) => println!("PARSE ERROR {e}"),
    }
}

#[test]
fn sixteen_bit_integer_and_predicate_ops_match_reference() {
    // u = 0x1234 (4660), v = 0xfff9 (65529 / -7 signed), w = 3.
    let regs = ".reg .b16 %h<6>;\n    .reg .u16 %u<6>;\n    .reg .s16 %s<6>;\n    .reg .pred %q<6>;";
    let body = format!(
        r#"    mov.u16 %u1, 0x1234;
    mov.u16 %u2, 0xfff9;
    mov.u16 %u3, 3;
    mov.s16 %s1, -7;
    mov.s16 %s2, 3;
    mov.b16 %h1, 0x8001;
    mov.b16 %h2, 0x8001;
    mov.u32 %r1, 5;
    setp.eq.b16 %q1, %h1, %h2;
    setp.ne.b16 %q2, %h1, %h2;
    setp.lt.u16 %q3, %u1, %u2;
    setp.lt.s16 %q4, %s1, %s2;
    selp.u32 %r10, 1, 0, %q1;
    selp.u32 %r11, 1, 0, %q2;
    selp.u32 %r12, 1, 0, %q3;
    selp.u32 %r13, 1, 0, %q4;
    shl.b16 %h3, %h1, 1;
    shl.b16 %h4, %h1, %u3;
    shr.b16 %h5, %h1, 15;
    not.b16 %h5, %h5;
    add.u16 %u4, %u2, %u3;
    sub.u16 %u5, %u3, %u2;
    mul.lo.u16 %u3, %u1, %u2;
    mul.hi.u16 %u4, %u1, %u2;
    mul.hi.s16 %s3, %s1, %s2;
    cvt.s32.s16 %r37, %s3;
    mul.wide.s16 %r20, %s1, %s2;
    mul.wide.u16 %r21, %u1, %u2;
    min.s16 %s4, %s1, %s2;
    max.u16 %u5, %u1, %u2;
    mad.lo.s16 %s5, %s1, %s2, %s2;
    abs.s16 %s5, %s1;
    cvt.s32.s16 %r22, %s1;
    cvt.u32.s16 %r23, %s1;
    cvt.u32.u16 %r24, %u2;
    cvt.s16.s32 %s4, %r1;
    cvt.sat.s16.s32 %s3, %r1;
    cvt.u64.u16 %rd1, %u2;
    cvt.s64.s16 %rd2, %s1;
    xor.pred %q5, %q1, %q2;
    selp.u32 %r25, 1, 0, %q5;
    cvt.u32.u16 %r26, %h3;
    cvt.u32.u16 %r27, %h4;
    cvt.u32.u16 %r28, %h5;
    cvt.u32.u16 %r29, %u3;
    cvt.u32.u16 %r30, %u4;
    cvt.u32.u16 %r31, %u5;
    cvt.s32.s16 %r32, %s3;
    cvt.s32.s16 %r33, %s4;
    cvt.s32.s16 %r34, %s5;
    cvt.u32.u16 %r35, %u5;
    cvt.s32.s16 %r36, %s5;
{}"#,
        stores(&[
            "%r10", "%r11", "%r12", "%r13", "%r26", "%r27", "%r28", "%r29", "%r30", "%r31", "%r32", "%r33",
            "%r34", "%r20", "%r21", "%r22", "%r23", "%r24", "%r25", "%r37",
        ])
    );
    let out = eval(regs, &body, 20);
    let (u1, u2) = (0x1234u32, 0xfff9u32);
    assert_eq!(out[0..4], [1, 0, 1, 1]);
    assert_eq!(out[4], 0x0002); // 0x8001 << 1 truncated to 16 bits
    assert_eq!(out[5], 0x0008); // 0x8001 << 3
    assert_eq!(out[6], 0xfffe); // !(0x8001 >> 15)
    assert_eq!(out[7], (u1 * u2) & 0xffff); // mul.lo.u16
    assert_eq!(out[8], (u1 * u2) >> 16); // mul.hi.u16
    assert_eq!(out[9], u2); // max.u16
    assert_eq!(out[10], 5); // cvt.sat.s16.s32
    assert_eq!(out[11], 5); // cvt.s16.s32
    assert_eq!(out[12], 7); // abs.s16
    assert_eq!(out[13] as i32, -21); // mul.wide.s16
    assert_eq!(out[14], u1 * u2); // mul.wide.u16
    assert_eq!(out[15] as i32, -7); // cvt.s32.s16
    assert_eq!(out[16] as i32, -7); // cvt.u32.s16 sign-extends
    assert_eq!(out[17], u2); // cvt.u32.u16
    assert_eq!(out[18], 1); // xor.pred
    assert_eq!(out[19] as i32, -1); // mul.hi.s16 of -7*3
}
