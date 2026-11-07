//! Tests for the tier-2 PTX coverage work (normalisation, inlining, extra
//! instructions). Integer semantics are executed on the SPIR-V interpreter in
//! `coverage_tests`; everything is validated with `spirv-val` when available.
use super::coverage_tests::{interp, kernel, lower};
use super::*;
use rspirv::spirv::Op;

/// Runs `source` (a full module whose single entry takes `(out, n)`) and
/// returns the first `words` words written to the output buffer.
fn run(source: &str, n: u32, words: usize) -> Vec<u32> {
    let module = lower(source);
    let mut machine = interp::Machine {
        params: vec![interp::GLOBAL, u64::from(n)],
        ..Default::default()
    };
    interp::run(&module, &mut machine);
    (0..words)
        .map(|i| machine.read(interp::GLOBAL + 4 * i as u64, 4) as u32)
        .collect()
}

fn eval(regs: &str, body: &str, words: usize) -> Vec<u32> {
    run(&kernel(regs, body), 0, words)
}

#[test]
fn accepts_modern_targets_and_isa_versions() {
    for (version, target) in [
        ("8.0", "sm_90"),
        ("8.3", "sm_90a"),
        ("8.7", "sm_100a"),
        ("9.0", "sm_120"),
        ("8.5", "sm_120a"),
        ("7.8", "sm_89"),
    ] {
        let source = kernel("", "    st.global.u32 [%rd15], %r0;")
            .replace(".version 9.0", &format!(".version {version}"))
            .replace("sm_75", target);
        lower(&source);
    }
    // Trailing target options are ignored.
    lower(&kernel("", "").replace(".target sm_75", ".target sm_80, debug"));
}

#[test]
fn drops_cache_hints_and_memory_order_qualifiers() {
    let out = eval(
        "",
        r#"    mov.u32 %r1, 77;
    st.global.cs.u32 [%rd15], %r1;
    st.release.gpu.global.u32 [%rd15+4], %r1;
    ld.global.L2::128B.u32 %r2, [%rd15];
    ld.acquire.gpu.global.u32 %r3, [%rd15+4];
    ld.relaxed.sys.global.L1::no_allocate.u32 %r4, [%rd15];
    ld.volatile.global.u32 %r5, [%rd15+4];
    ld.global.lu.u32 %r6, [%rd15];
    prefetch.global.L2 [%rd15];
    prefetch.global.L2::evict_last [%rd15];
    st.global.u32 [%rd15+8], %r2;
    st.global.u32 [%rd15+12], %r3;
    st.global.u32 [%rd15+16], %r4;
    st.global.u32 [%rd15+20], %r5;
    st.global.u32 [%rd15+24], %r6;"#,
        7,
    );
    assert_eq!(out[2..], [77; 5]);
}

#[test]
fn wide_vector_accesses_split_into_scalars() {
    let out = eval(
        "    .reg .b64 %rl<4>;",
        r#"    mov.u64 %rl0, 0x1111111122222222;
    mov.u64 %rl1, 0x3333333344444444;
    st.global.v2.u64 [%rd15], {%rl0, %rl1};
    ld.global.v2.u64 {%rl2, %rl3}, [%rd15];
    st.global.u64 [%rd15+16], %rl3;"#,
        6,
    );
    assert_eq!(out, [0x22222222, 0x11111111, 0x44444444, 0x33333333, 0x44444444, 0x33333333]);
}

#[test]
fn named_and_register_barriers_lower_to_a_full_barrier() {
    for form in ["bar.sync %r0", "bar.sync 3", "barrier.sync 1", "barrier.sync.aligned 0", "barrier.sync %r0"] {
        let module = lower(&kernel("", &format!("    {form};\n    st.global.u32 [%rd15], %r0;")));
        assert!(has_op(&module, Op::ControlBarrier), "{form}");
    }
    // A thread count that is not the whole block cannot be a workgroup barrier.
    assert!(translate(&kernel("", "    bar.sync 1, 64;")).is_err());
}

fn has_op(module: &rspirv::dr::Module, op: Op) -> bool {
    module.all_inst_iter().any(|inst| inst.class.opcode == op)
}

#[test]
fn legacy_warp_intrinsics_gain_a_full_mask() {
    lower(&kernel(
        "",
        r#"    setp.lt.u32 %p1, %r0, 4;
    vote.ballot.b32 %r1, %p1;
    vote.all.pred %p2, %p1;
    shfl.bfly.b32 %r2, %r1, 1, 0x1f;
    shfl.down.b32 %r3, %r1, 1, 0x1f;
    st.global.u32 [%rd15], %r2;"#,
    ));
}

#[test]
fn special_register_constants_and_lane_masks() {
    let out = eval(
        "",
        r#"    mov.u32 %r1, %lanemask_lt;
    mov.u32 %r2, %lanemask_le;
    mov.u32 %r3, %lanemask_ge;
    mov.u32 %r4, %lanemask_gt;
    mov.u32 %r5, %lanemask_eq;
    mov.u32 %r6, %WARP_SZ;
    mov.u32 %r7, %smid;
    st.global.u32 [%rd15], %r1;
    st.global.u32 [%rd15+4], %r2;
    st.global.u32 [%rd15+8], %r3;
    st.global.u32 [%rd15+12], %r4;
    st.global.u32 [%rd15+16], %r5;
    st.global.u32 [%rd15+20], %r6;
    st.global.u32 [%rd15+24], %r7;"#,
        7,
    );
    // The interpreter simulates lane 0.
    assert_eq!(out, [0, 1, 0xffff_ffff, 0xffff_fffe, 1, 32, 0]);
}

#[test]
fn exit_and_trap_end_the_thread() {
    let out = eval(
        "",
        r#"    mov.u32 %r1, 5;
    st.global.u32 [%rd15], %r1;
    setp.lt.u32 %p1, %r1, 9;
    @%p1 exit;
    st.global.u32 [%rd15+4], %r1;
    trap;"#,
        2,
    );
    assert_eq!(out, [5, 0]);
}

#[test]
fn cp_async_is_a_synchronous_copy_with_no_op_group_operations() {
    let source = kernel_shared(
        r#"    st.global.u32 [%rd15+64], %r0;
    mov.u32 %r6, 11;
    st.global.u32 [%rd15+68], %r6;
    mov.u32 %r1, sbuf;
    cp.async.ca.shared.global [%r1], [%rd15+64], 4;
    cp.async.ca.shared.global [%r1+4], [%rd15+68], 4;
    cp.async.commit_group;
    cp.async.wait_group 0;
    bar.sync 0;
    ld.shared.u32 %r2, [%r1];
    ld.shared.u32 %r3, [%r1+4];
    st.global.u32 [%rd15], %r2;
    st.global.u32 [%rd15+4], %r3;"#,
    );
    assert_eq!(run(&source, 9, 2), [9, 11]);
    // The zero-fill form copies nothing when the source size is zero.
    let source = kernel_shared(
        r#"    mov.u32 %r6, 99;
    st.global.u32 [%rd15+64], %r6;
    st.global.u32 [%rd15+68], %r6;
    st.global.u32 [%rd15+72], %r6;
    st.global.u32 [%rd15+76], %r6;
    mov.u32 %r1, sbuf;
    mov.u32 %r5, 0;
    cp.async.ca.shared.global [%r1], [%rd15+64], 4, %r5;
    cp.async.wait_all;
    ld.shared.u32 %r2, [%r1];
    st.global.u32 [%rd15], %r2;"#,
    );
    assert_eq!(run(&source, 0, 1), [0]);
    // Vector widths and predicated copies translate and validate.
    lower(&kernel_shared(
        r#"    mov.u32 %r1, sbuf;
    mov.u32 %r5, 16;
    setp.lt.u32 %p1, %r0, 3;
    cp.async.cg.shared.global.L2::128B [%r1+16], [%rd15+64], 16, %r5;
    @%p1 cp.async.ca.shared.global [%r1+32], [%rd15+64], 8;
    @%p1 cp.async.cg.shared.global [%r1+48], [%rd15+64], 16, %r5;"#,
    ));
}

fn kernel_shared(body: &str) -> String {
    super::coverage_tests::kernel_with_shared("", ".shared .align 16 .b8 sbuf[64];", body)
}

#[test]
fn inlines_func_definitions_and_calls() {
    let source = r#".version 8.0
.target sm_80
.address_size 64
.visible .func (.param .b32 func_retval0) addmul(.param .b32 addmul_param_0, .param .b32 addmul_param_1)
{
    .reg .b32 %r<5>;
    ld.param.u32 %r1, [addmul_param_0];
    ld.param.u32 %r2, [addmul_param_1];
    mad.lo.s32 %r3, %r1, %r2, %r1;
    st.param.b32 [func_retval0+0], %r3;
    ret;
}
.visible .func (.param .b32 func_retval0) twice(.param .b32 twice_param_0)
{
    .reg .b32 %r<5>;
    .reg .b32 %t<3>;
    .reg .pred %p<2>;
    ld.param.u32 %r1, [twice_param_0];
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra $L_zero;
    {
        .param .b32 param0;
        st.param.b32 [param0+0], %r1;
        .param .b32 param1;
        st.param.b32 [param1+0], 2;
        .param .b32 retval0;
        call.uni (retval0), addmul, (param0, param1);
        ld.param.b32 %r2, [retval0+0];
    }
    st.param.b32 [func_retval0+0], %r2;
    ret;
$L_zero:
    st.param.b32 [func_retval0+0], 1000;
    ret;
}
.visible .entry probe(.param .u64 out_param, .param .u32 n_param)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd0, [out_param];
    cvta.to.global.u64 %rd1, %rd0;
    ld.param.u32 %r0, [n_param];
    {
        .param .b32 param0;
        st.param.b32 [param0+0], %r0;
        .param .b32 retval0;
        call.uni (retval0), twice, (param0);
        ld.param.b32 %r1, [retval0+0];
    }
    {
        .param .b32 param0;
        st.param.b32 [param0+0], 7;
        .param .b32 retval0;
        call.uni (retval0), twice, (param0);
        ld.param.b32 %r2, [retval0+0];
    }
    st.global.u32 [%rd1], %r1;
    st.global.u32 [%rd1+4], %r2;
    ret;
}
"#;
    // twice(n) = n + 2n = 3n ; twice(0) = 1000.
    assert_eq!(run(source, 5, 2), [15, 21]);
    assert_eq!(run(source, 0, 2), [1000, 21]);
}

#[test]
fn assert_and_printf_calls_are_dropped() {
    let source = kernel(
        "",
        r#"    {
        .param .b64 param0;
        st.param.b64 [param0+0], %rd15;
        .param .b32 retval0;
        call.uni (retval0), vprintf, (param0, param0);
        ld.param.b32 %r1, [retval0+0];
    }
    {
        .param .b64 param0;
        st.param.b64 [param0+0], %rd15;
        call.uni __assertfail, (param0);
    }
    st.global.u32 [%rd15], %r1;"#,
    )
    .replace(
        ".visible .entry",
        ".extern .func (.param .b32 func_retval0) vprintf (.param .b64 vprintf_param_0, .param .b64 vprintf_param_1);\n.extern .func __assertfail (.param .b64 a);\n.visible .entry",
    );
    assert_eq!(run(&source, 0, 1), [0]);
}

fn stores(names: &[&str]) -> String {
    names
        .iter()
        .enumerate()
        .map(|(i, r)| format!("    st.global.u32 [%rd15+{}], {r};\n", 4 * i))
        .collect()
}

#[test]
fn bit_manipulation_matches_reference() {
    let a = 0x8f3c_5a96u32;
    let body = format!(
        r#"    mov.u32 %r1, 0x8f3c5a96;
    brev.b32 %r2, %r1;
    mov.u32 %r3, 8;
    mov.u32 %r4, 12;
    mov.u32 %r5, 40;
    bmsk.wrap.b32 %r6, %r3, %r4;
    bmsk.wrap.b32 %r7, %r5, %r4;
    bmsk.clamp.b32 %r8, %r5, %r4;
    szext.wrap.s32 %r9, %r1, %r4;
    szext.clamp.u32 %r10, %r1, %r4;
    szext.clamp.s32 %r11, %r1, %r5;
    mov.u32 %r12, 0x123456;
    mov.u32 %r13, 0xff9abcde;
    mul24.lo.u32 %r14, %r12, %r13;
    mul24.hi.u32 %r15, %r12, %r13;
    mul24.lo.s32 %r16, %r12, %r13;
    mul24.hi.s32 %r17, %r12, %r13;
    mad24.lo.u32 %r18, %r12, %r13, %r3;
    sad.s32 %r19, %r1, %r12, %r3;
    sad.u32 %r20, %r1, %r12, %r3;
    cnot.b32 %r21, %r3;
    mov.u32 %r22, 0;
    cnot.b32 %r23, %r22;
    mov.u32 %r24, 0x00ff0000;
    bfind.u32 %r25, %r24;
    bfind.shiftamt.u32 %r26, %r24;
    bfind.u32 %r27, %r22;
    mov.u32 %r28, 0xffff0000;
    bfind.s32 %r29, %r28;
    mad.hi.s32 %r30, %r1, %r12, %r3;
    mad.hi.u32 %r31, %r1, %r12, %r3;
    shf.l.wrap.b32 %r32, %r1, %r12, %r3;
    shf.r.clamp.b32 %r33, %r1, %r12, %r5;
    shf.r.wrap.b32 %r34, %r1, %r12, %r3;
    shf.l.clamp.b32 %r35, %r1, %r12, %r5;
    mov.u32 %r36, 1;
    slct.b32.s32 %r37, %r1, %r12, %r36;
    mov.u32 %r38, -1;
    slct.b32.s32 %r39, %r1, %r12, %r38;
{}"#,
        stores(&[
            "%r2", "%r6", "%r7", "%r8", "%r9", "%r10", "%r11", "%r14", "%r15", "%r16", "%r17",
            "%r18", "%r19", "%r20", "%r21", "%r23", "%r25", "%r26", "%r27", "%r29", "%r30",
            "%r31", "%r32", "%r33", "%r34", "%r35", "%r37", "%r39"
        ])
    );
    let out = eval("", &body, 28);
    let (m, n) = (0x123456u32, 0xff9a_bcdeu32);
    let (mu, nu) = (m & 0xff_ffff, n & 0xff_ffff);
    let sx = |x: u32| (((x << 8) as i32) >> 8) as i64;
    let sad_s = (a as i32).wrapping_sub(m as i32).unsigned_abs().wrapping_add(8);
    let sad_u = a.abs_diff(m).wrapping_add(8);
    let hi_s = ((i64::from(a as i32) * i64::from(m as i32)) >> 32) as u32;
    let hi_u = ((u64::from(a) * u64::from(m)) >> 32) as u32;
    let pair = (u64::from(m) << 32) | u64::from(a);
    let expected = [
        a.reverse_bits(),
        0x000f_ff00,
        0x000f_ff00,
        0,
        0xffff_fa96,
        0xa96,
        a,
        mu.wrapping_mul(nu),
        ((u64::from(mu) * u64::from(nu)) >> 16) as u32,
        (sx(m) * sx(n)) as u32,
        ((sx(m) * sx(n)) >> 16) as u32,
        mu.wrapping_mul(nu).wrapping_add(8),
        sad_s,
        sad_u,
        0,
        1,
        23,
        8,
        u32::MAX,
        15,
        hi_s.wrapping_add(8),
        hi_u.wrapping_add(8),
        ((pair << 8) >> 32) as u32,
        m,
        ((pair >> 8) & 0xffff_ffff) as u32,
        a,
        a,
        m,
    ];
    for (i, (got, want)) in out.iter().zip(expected).enumerate() {
        assert_eq!(*got, want, "word {i}: {got:#x} != {want:#x}");
    }
}

#[test]
fn prmt_modes_dot_products_and_fns() {
    let a = 0x8f3c_5a96u32;
    let b = 0x1122_3344u32;
    let body = format!(
        r#"    mov.u32 %r1, 0x8f3c5a96;
    mov.u32 %r2, 0x11223344;
    mov.u32 %r3, 2;
    prmt.b32.f4e %r4, %r1, %r2, %r3;
    prmt.b32.b4e %r5, %r1, %r2, %r3;
    prmt.b32.rc8 %r6, %r1, %r2, %r3;
    prmt.b32.ecl %r7, %r1, %r2, %r3;
    prmt.b32.ecr %r8, %r1, %r2, %r3;
    prmt.b32.rc16 %r9, %r1, %r2, %r3;
    mov.u32 %r10, 0x80ff017f;
    mov.u32 %r11, 0x7f81ff02;
    mov.u32 %r12, 100;
    dp4a.s32.u32 %r13, %r10, %r11, %r12;
    dp4a.u32.s32 %r14, %r10, %r11, %r12;
    dp4a.u32.u32 %r15, %r10, %r11, %r12;
    dp4a.s32.s32 %r16, %r10, %r11, %r12;
    dp2a.lo.s32.u32 %r17, %r10, %r11, %r12;
    dp2a.hi.u32.s32 %r18, %r10, %r11, %r12;
    mov.u32 %r19, 0x2d;
    mov.u32 %r20, 0;
    mov.u32 %r21, 1;
    mov.u32 %r22, -1;
    mov.u32 %r23, 5;
    fns.b32 %r24, %r19, %r20, %r21;
    fns.b32 %r25, %r19, %r20, %r3;
    fns.b32 %r26, %r19, %r20, %r20;
    fns.b32 %r27, %r19, %r21, %r20;
    fns.b32 %r28, %r19, %r23, %r22;
    mov.u32 %r31, -2;
    mov.u32 %r32, -3;
    fns.b32 %r29, %r19, %r23, %r31;
    fns.b32 %r30, %r19, %r23, %r32;
{}"#,
        stores(&[
            "%r4", "%r5", "%r6", "%r7", "%r8", "%r9", "%r13", "%r14", "%r15", "%r16", "%r17",
            "%r18", "%r24", "%r25", "%r26", "%r27", "%r28", "%r29", "%r30"
        ])
    );
    let out = eval("", &body, 19);
    let pair = (u64::from(b) << 32) | u64::from(a);
    let byte = |i: u32| ((pair >> (8 * (i & 7))) & 0xff) as u32;
    let build = |sel: [u32; 4]| (0..4).fold(0u32, |acc, k| acc | (byte(sel[k as usize]) << (8 * k)));
    let n = 2u32;
    let f4e = build([n, n + 1, n + 2, n + 3]);
    let b4e = build([n, (n.wrapping_sub(1)) & 7, (n.wrapping_sub(2)) & 7, (n.wrapping_sub(3)) & 7]);
    let rc8 = build([n; 4]);
    let ecl = build([n.max(0), n.max(1), n.max(2), n.max(3)]);
    let ecr = build([n.min(0), n.min(1), n.min(2), n.min(3)]);
    let rc16 = build([0, 1, 0, 1]);
    let (x, y, z) = (0x80ff_017fu32, 0x7f81_ff02u32, 100u32);
    let dot = |sa: bool, sb: bool| {
        (0..4).fold(z as i64, |acc, k| {
            let p = ((x >> (8 * k)) & 0xff) as u8;
            let q = ((y >> (8 * k)) & 0xff) as u8;
            let p = if sa { i64::from(p as i8) } else { i64::from(p) };
            let q = if sb { i64::from(q as i8) } else { i64::from(q) };
            acc + p * q
        }) as u32
    };
    let dp2 = |hi: bool, sa: bool, sb: bool| {
        (0..2).fold(z as i64, |acc, k| {
            let lane = ((x >> (16 * k)) & 0xffff) as u16;
            let lane = if sa { i64::from(lane as i16) } else { i64::from(lane) };
            let q = ((y >> (8 * (k + if hi { 2 } else { 0 }))) & 0xff) as u8;
            let q = if sb { i64::from(q as i8) } else { i64::from(q) };
            acc + lane * q
        }) as u32
    };
    // mask 0x2d has bits 0, 2, 3, 5.
    let expected = [
        f4e, b4e, rc8, ecl, ecr, rc16,
        dot(true, false), dot(false, true), dot(false, false), dot(true, true),
        dp2(false, true, false), dp2(true, false, true),
        2, 3, 0, u32::MAX, 3, 2, 0,
    ];
    for (i, (got, want)) in out.iter().zip(expected).enumerate() {
        assert_eq!(*got, want, "word {i}: {got:#x} != {want:#x}");
    }
}

#[test]
fn wide_moves_and_64_bit_bit_manipulation() {
    let body = r#"    mov.u32 %r1, 0x11223344;
    mov.u32 %r2, 0x99aabbcc;
    mov.b64 %rd1, {%r1, %r2};
    mov.b64 {%r3, %r4}, %rd1;
    brev.b64 %rd2, %rd1;
    clz.b64 %r5, %rd1;
    cnot.b64 %rd3, %rd1;
    mov.u32 %r6, 8;
    mov.u32 %r7, 16;
    bfe.u64 %rd4, %rd1, %r6, %r7;
    bfe.s64 %rd5, %rd1, %r6, %r7;
    bfi.b64 %rd6, %rd1, %rd2, %r7, %r6;
    mul.hi.u64 %rd7, %rd1, %rd1;
    mov.b32 {%r8, %r9}, %r1;
    st.global.u32 [%rd15], %r3;
    st.global.u32 [%rd15+4], %r4;
    st.global.u64 [%rd15+8], %rd2;
    st.global.u32 [%rd15+16], %r5;
    st.global.u64 [%rd15+24], %rd3;
    st.global.u64 [%rd15+32], %rd4;
    st.global.u64 [%rd15+40], %rd5;
    st.global.u64 [%rd15+48], %rd6;
    st.global.u64 [%rd15+56], %rd7;
    st.global.u32 [%rd15+64], %r8;
    st.global.u32 [%rd15+68], %r9;"#;
    let out = eval("", body, 18);
    let x = 0x99aa_bbcc_1122_3344u64;
    let word = |i: usize| (u64::from(out[i + 1]) << 32) | u64::from(out[i]);
    assert_eq!((out[0], out[1]), (0x1122_3344, 0x99aa_bbcc));
    assert_eq!(word(2), x.reverse_bits());
    assert_eq!(out[4], 0);
    assert_eq!(word(6), 0);
    assert_eq!(word(8), (x >> 8) & 0xffff);
    assert_eq!(word(10), 0x0000_0000_0000_0000 | ((x >> 8) & 0xffff));
    let reversed = x.reverse_bits();
    let field = 0xffu64 << 16;
    assert_eq!(word(12), (reversed & !field) | ((x << 16) & field));
    assert_eq!(word(14), ((u128::from(x) * u128::from(x)) >> 64) as u64);
    assert_eq!((out[16], out[17]), (0x3344, 0x1122));
}

#[test]
fn warp_collectives_translate_and_validate() {
    let module = lower(&kernel(
        "    .reg .b64 %rl<4>;",
        r#"    mov.u32 %r1, %laneid;
    setp.lt.u32 %p1, %r1, 5;
    redux.sync.add.u32 %r2, %r1, 0xffffffff;
    redux.sync.min.s32 %r3, %r1, 0xffffffff;
    redux.sync.max.u32 %r4, %r1, 0xffffffff;
    redux.sync.and.b32 %r5, %r1, 0xffffffff;
    redux.sync.or.b32 %r6, %r1, 0xffffffff;
    redux.sync.xor.b32 %r7, %r1, 0xffffffff;
    match.any.sync.b32 %r8, %r1, 0xffffffff;
    match.all.sync.b32 %r9|%p2, %r1, 0xffffffff;
    cvt.u64.u32 %rl0, %r1;
    match.any.sync.b64 %r10, %rl0, 0xffffffff;
    vote.sync.uni.pred %p3, %p1, 0xffffffff;
    vote.sync.any.pred %p4, !%p1, 0xffffffff;
    elect.sync %r11|%p5, 0xffffffff;
    shfl.sync.down.b32 %r12|%p6, %r1, %r1, %r1, 0xffffffff;
    shfl.sync.idx.b32 %r13, %r1, %r1, %r1, 0xffffffff;
    shfl.sync.bfly.b32 %r14, %r1, %r1, %r1, 0xffffffff;
    mov.u32 %r15, %warpid;
    mov.u32 %r16, %nwarpid;
    selp.u32 %r17, 1, 0, %p3;
    selp.u32 %r18, 1, 0, %p2;
    selp.u32 %r19, 1, 0, %p4;
    selp.u32 %r20, 1, 0, %p5;
    st.global.u32 [%rd15], %r2;
    st.global.u32 [%rd15+4], %r9;
    st.global.u32 [%rd15+8], %r17;"#,
    ));
    assert!(has_op(&module, Op::GroupNonUniformIAdd));
    assert!(has_op(&module, Op::GroupNonUniformAllEqual));
    assert!(has_op(&module, Op::GroupNonUniformElect));
    assert!(has_op(&module, Op::GroupNonUniformBitwiseXor));
    // Partial membermasks cannot be honoured.
    assert!(translate(&kernel("", "    shfl.sync.up.b32 %r2, %r1, %r1, %r1, 0xffff;")).is_err());
}

#[test]
fn wide_and_extra_atomics_translate_and_validate() {
    let module = lower(&kernel(
        "    .reg .b64 %rl<8>;\n    .reg .f32 %fa<2>;\n    .reg .b64 %fd<4>;",
        r#"    mov.u64 %rl0, 5;
    mov.u64 %rl1, 7;
    mov.u32 %r30, 0x40a00000;
    mov.b32 %f2, %r30;
    atom.global.add.u64 %rl2, [%rd15], %rl0;
    atom.global.min.u64 %rl3, [%rd15+8], %rl0;
    atom.global.max.s64 %rl4, [%rd15+16], %rl0;
    atom.global.exch.b64 %rl5, [%rd15+24], %rl0;
    atom.global.cas.b64 %rl6, [%rd15+32], %rl0, %rl1;
    atom.global.and.b64 %rl7, [%rd15+40], %rl0;
    red.global.add.u64 [%rd15+48], %rl0;
    red.global.or.b64 [%rd15+56], %rl0;
    atom.global.inc.u32 %r1, [%rd15+64], %r0;
    atom.global.dec.u32 %r2, [%rd15+68], %r0;
    red.global.inc.u32 [%rd15+72], %r0;
    atom.global.min.f32 %f1, [%rd15+76], %f2;
    atom.global.max.f32 %f1, [%rd15+80], %f2;
    atom.relaxed.gpu.global.cas.b32 %r3, [%rd15+84], %r0, %r1;
    atom.global.add.f64 %fd0, [%rd15+88], %fd1;
    st.global.u64 [%rd15+96], %rl2;"#,
    )
    .replace("%fd1;", "%rl1;"));
    assert!(has_op(&module, Op::AtomicIAdd));
    assert!(has_op(&module, Op::AtomicCompareExchange));
    assert!(has_op(&module, Op::LoopMerge) || has_op(&module, Op::BranchConditional));
}
