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

// ---- floating-point reference helpers -------------------------------------

fn h2f(h: u16) -> f32 {
    let (sign, exp, man) = ((h >> 15) & 1, i32::from((h >> 10) & 0x1f), f32::from(h & 0x3ff));
    let magnitude = match exp {
        0 => man * 2f32.powi(-24),
        0x1f => if man == 0.0 { f32::INFINITY } else { f32::NAN },
        _ => (1.0 + man / 1024.0) * 2f32.powi(exp - 15),
    };
    if sign == 1 { -magnitude } else { magnitude }
}

fn f2h(x: f32) -> u16 {
    interp::f32_to_f16_bits(x)
}

fn bf2f(b: u16) -> f32 {
    f32::from_bits(u32::from(b) << 16)
}

fn f2bf(x: f32) -> u16 {
    if x.is_nan() {
        return (((x.to_bits() >> 16) & 0x8000) | 0x7fff) as u16;
    }
    let u = x.to_bits();
    ((u + 0x7fff + ((u >> 16) & 1)) >> 16) as u16
}

const FREGS: &str = "    .reg .f64 %fd<8>;\n    .reg .bf16 %hb<8>;\n    .reg .f16 %h<8>;\n    .reg .b32 %q<16>;";

#[test]
fn bf16_and_f16_scalar_arithmetic_matches_reference() {
    // Values chosen so that several results need real rounding.
    let a = 0x3fc1u16; // bf16 1.5078125
    let b = 0xc011u16; // bf16 -2.265625
    let c = 0x3f81u16; // bf16 1.0078125
    let (ha, hb, hc) = (f2h(1.0009765), f2h(-2.7), f2h(0.3333));
    let body = format!(
        r#"    mov.b16 %hb1, {a};
    mov.b16 %hb2, {b};
    mov.b16 %hb3, {c};
    add.rn.bf16 %hb4, %hb1, %hb2;
    mul.rn.bf16 %hb5, %hb1, %hb2;
    fma.rn.bf16 %hb6, %hb1, %hb2, %hb3;
    sub.rn.bf16 %hb7, %hb1, %hb2;
    neg.bf16 %q1, %hb1;
    abs.bf16 %q2, %hb2;
    max.bf16 %q3, %hb1, %hb2;
    min.bf16 %q4, %hb1, %hb2;
    mov.b16 %h1, {ha};
    mov.b16 %h2, {hb};
    mov.b16 %h3, {hc};
    add.rn.f16 %h4, %h1, %h2;
    mul.rn.f16 %h5, %h1, %h2;
    fma.rn.f16 %h6, %h1, %h2, %h3;
    fma.rn.relu.f16 %h7, %h1, %h2, %h3;
    setp.lt.f16 %p1, %h2, %h1;
    selp.u32 %q5, 1, 0, %p1;
    setp.gt.bf16 %p2, %hb2, %hb1;
    selp.u32 %q6, 1, 0, %p2;
{}"#,
        stores(&[
            "%hb4", "%hb5", "%hb6", "%hb7", "%q1", "%q2", "%q3", "%q4", "%h4", "%h5", "%h6",
            "%h7", "%q5", "%q6"
        ])
    );
    let out = eval(FREGS, &body, 14);
    let (fa, fb, fc) = (bf2f(a), bf2f(b), bf2f(c));
    let (fha, fhb, fhc) = (h2f(ha), h2f(hb), h2f(hc));
    let expected = [
        u32::from(f2bf(fa + fb)),
        u32::from(f2bf(fa * fb)),
        u32::from(f2bf(fa.mul_add(fb, fc))),
        u32::from(f2bf(fa - fb)),
        u32::from(a ^ 0x8000),
        u32::from(b & 0x7fff),
        u32::from(a),
        u32::from(b),
        u32::from(f2h(fha + fhb)),
        u32::from(f2h(fha * fhb)),
        u32::from(f2h(fha.mul_add(fhb, fhc))),
        0, // the fused result is negative, so relu clamps it to +0
        1,
        0,
    ];
    for (i, (got, want)) in out.iter().zip(expected).enumerate() {
        assert_eq!(*got, want, "word {i}: {got:#x} != {want:#x}");
    }
}

#[test]
fn packed_half_arithmetic_and_conversions_match_reference() {
    let pack = |hi: u16, lo: u16| (u32::from(hi) << 16) | u32::from(lo);
    let (x, y, z) = (pack(f2h(1.5), f2h(-0.75)), pack(f2h(2.25), f2h(0.1)), pack(f2h(-1.0), f2h(3.0)));
    let (bx, by) = (pack(f2bf(1.5), f2bf(-0.75)), pack(f2bf(2.3), f2bf(0.1)));
    let nan = f32::NAN;
    let body = format!(
        r#"    mov.b32 %q1, {x};
    mov.b32 %q2, {y};
    mov.b32 %q3, {z};
    add.rn.f16x2 %q4, %q1, %q2;
    mul.rn.f16x2 %q5, %q1, %q2;
    fma.rn.f16x2 %q6, %q1, %q2, %q3;
    neg.f16x2 %q7, %q1;
    abs.f16x2 %q8, %q3;
    max.f16x2 %q9, %q1, %q3;
    mov.b32 %q10, {bx};
    mov.b32 %q11, {by};
    add.rn.bf16x2 %q12, %q10, %q11;
    fma.rn.bf16x2 %q13, %q10, %q11, %q10;
    mov.b32 %f1, {f1};
    mov.b32 %f2, {f2};
    cvt.rn.f16x2.f32 %q14, %f1, %f2;
    cvt.rn.bf16x2.f32 %q15, %f1, %f2;
    set.gt.f16x2.f16x2 %q0, %q1, %q3;
{}"#,
        stores(&["%q4", "%q5", "%q6", "%q7", "%q8", "%q9", "%q12", "%q13", "%q14", "%q15", "%q0"]),
        f1 = 0.3333f32.to_bits(),
        f2 = (-70000.0f32).to_bits(),
    );
    let out = eval(FREGS, &body, 11);
    let lane = |v: u32, hi: bool| if hi { (v >> 16) as u16 } else { v as u16 };
    let both = |f: &dyn Fn(bool) -> u16| pack(f(true), f(false));
    let expected = [
        both(&|hi| f2h(h2f(lane(x, hi)) + h2f(lane(y, hi)))),
        both(&|hi| f2h(h2f(lane(x, hi)) * h2f(lane(y, hi)))),
        both(&|hi| f2h(h2f(lane(x, hi)).mul_add(h2f(lane(y, hi)), h2f(lane(z, hi))))),
        x ^ 0x8000_8000,
        z & 0x7fff_7fff,
        both(&|hi| f2h(h2f(lane(x, hi)).max(h2f(lane(z, hi))))),
        both(&|hi| f2bf(bf2f(lane(bx, hi)) + bf2f(lane(by, hi)))),
        both(&|hi| f2bf(bf2f(lane(bx, hi)).mul_add(bf2f(lane(by, hi)), bf2f(lane(bx, hi))))),
        pack(f2h(0.3333), f2h(-70000.0)),
        pack(f2bf(0.3333), f2bf(-70000.0)),
        // 1.5 > -1.0 in the upper lane, -0.75 > 3.0 is false in the lower lane: 1.0 / 0.0.
        pack(0x3c00, 0),
    ];
    let _ = nan;
    for (i, (got, want)) in out.iter().zip(expected).enumerate() {
        assert_eq!(*got, want, "word {i}: {got:#x} != {want:#x}");
    }
}

#[test]
fn bf16_conversion_rounds_to_nearest_even_and_keeps_nan() {
    let inputs = [
        1.0f32,
        f32::from_bits(0x3f80_8000), // exact tie, rounds to even (down)
        f32::from_bits(0x3f81_8000), // exact tie, rounds to even (up)
        f32::from_bits(0x3f80_8001), // just above a tie
        3.4e38,
        f32::from_bits(0x7f7f_ffff), // largest finite: rounds to infinity
        f32::NEG_INFINITY,
        f32::NAN,
    ];
    let mut body = String::new();
    for (i, v) in inputs.iter().enumerate() {
        body.push_str(&format!(
            "    mov.b32 %f1, {};\n    cvt.rn.bf16.f32 %hb1, %f1;\n    cvt.f32.bf16 %f2, %hb1;\n    st.global.u32 [%rd15+{}], %hb1;\n    st.global.u32 [%rd15+{}], %f2;\n",
            v.to_bits(),
            8 * i,
            8 * i + 4
        ));
    }
    let out = eval(FREGS, &body, 2 * inputs.len());
    for (i, v) in inputs.iter().enumerate() {
        let want = f2bf(*v);
        assert_eq!(out[2 * i], u32::from(want), "input {i}: {:#x}", v.to_bits());
        assert_eq!(out[2 * i + 1], bf2f(want).to_bits(), "widened {i}");
    }
}

#[test]
fn f64_arithmetic_conversions_and_compares() {
    let (p, q) = (2.5f64, -0.75f64);
    let body = format!(
        r#"    mov.b64 %fd1, 0d{:016X};
    mov.b64 %fd2, 0d{:016X};
    add.f64 %fd3, %fd1, %fd2;
    sub.rn.f64 %fd4, %fd1, %fd2;
    mul.rn.f64 %fd5, %fd1, %fd2;
    div.rn.f64 %fd6, %fd1, %fd2;
    fma.rn.f64 %fd7, %fd1, %fd2, %fd1;
    sqrt.rn.f64 %fd0, %fd1;
    st.global.u64 [%rd15], %fd3;
    st.global.u64 [%rd15+8], %fd4;
    st.global.u64 [%rd15+16], %fd5;
    st.global.u64 [%rd15+24], %fd6;
    st.global.u64 [%rd15+32], %fd7;
    st.global.u64 [%rd15+40], %fd0;
    neg.f64 %fd3, %fd1;
    abs.f64 %fd4, %fd2;
    min.f64 %fd5, %fd1, %fd2;
    max.f64 %fd6, %fd1, %fd2;
    rcp.rn.f64 %fd7, %fd2;
    st.global.u64 [%rd15+48], %fd3;
    st.global.u64 [%rd15+56], %fd4;
    st.global.u64 [%rd15+64], %fd5;
    st.global.u64 [%rd15+72], %fd6;
    st.global.u64 [%rd15+80], %fd7;
    cvt.rn.f32.f64 %f1, %fd2;
    cvt.f64.f32 %fd3, %f1;
    mov.u32 %r1, -7;
    cvt.rn.f64.s32 %fd4, %r1;
    cvt.rzi.s32.f64 %r2, %fd1;
    cvt.rni.s32.f64 %r3, %fd2;
    cvt.rzi.u32.f64 %r4, %fd2;
    cvt.rmi.s64.f64 %rd1, %fd2;
    mov.b64 %fd5, 0d7FF0000000000000;
    cvt.rzi.s32.f64 %r5, %fd5;
    setp.lt.f64 %p1, %fd2, %fd1;
    setp.nan.f64 %p2, %fd5, %fd1;
    testp.infinite.f64 %p3, %fd5;
    selp.u32 %r6, 1, 0, %p1;
    selp.u32 %r7, 1, 0, %p2;
    selp.u32 %r8, 1, 0, %p3;
    st.global.u32 [%rd15+88], %f1;
    st.global.u64 [%rd15+96], %fd3;
    st.global.u64 [%rd15+104], %fd4;
    st.global.u32 [%rd15+112], %r2;
    st.global.u32 [%rd15+116], %r3;
    st.global.u32 [%rd15+120], %r4;
    st.global.u64 [%rd15+128], %rd1;
    st.global.u32 [%rd15+136], %r5;
    st.global.u32 [%rd15+140], %r6;
    st.global.u32 [%rd15+144], %r7;
    st.global.u32 [%rd15+148], %r8;"#,
        p.to_bits(),
        q.to_bits()
    );
    let out = eval("    .reg .f64 %fd<8>;", &body, 38);
    let wide = |i: usize| f64::from_bits((u64::from(out[i + 1]) << 32) | u64::from(out[i]));
    assert_eq!(wide(0), p + q);
    assert_eq!(wide(2), p - q);
    assert_eq!(wide(4), p * q);
    assert_eq!(wide(6), p / q);
    assert_eq!(wide(8), p.mul_add(q, p));
    assert_eq!(wide(10), p.sqrt());
    assert_eq!(wide(12), -p);
    assert_eq!(wide(14), q.abs());
    assert_eq!(wide(16), q);
    assert_eq!(wide(18), p);
    assert_eq!(wide(20), 1.0 / q);
    assert_eq!(out[22], (q as f32).to_bits());
    assert_eq!(wide(24), f64::from(q as f32));
    assert_eq!(wide(26), -7.0);
    assert_eq!(out[28] as i32, 2);
    assert_eq!(out[29] as i32, -1);
    assert_eq!(out[30], 0);
    assert_eq!(((u64::from(out[33]) << 32) | u64::from(out[32])) as i64, -1);
    assert_eq!(out[34] as i32, i32::MAX);
    assert_eq!((out[35], out[36], out[37]), (1, 0, 1));
}

#[test]
fn saturating_nan_aware_f32_and_integer_conversions() {
    let body = format!(
        r#"    mov.b32 %f1, {a};
    mov.b32 %f2, {b};
    mov.b32 %f3, {n};
    add.sat.f32 %f4, %f1, %f2;
    mul.sat.f32 %f5, %f1, %f1;
    fma.rn.sat.f32 %f6, %f1, %f2, %f2;
    max.NaN.f32 %f7, %f1, %f3;
    min.NaN.f32 %f8, %f1, %f2;
    max.f32 %f9, %f1, %f3;
    st.global.u32 [%rd15], %f4;
    st.global.u32 [%rd15+4], %f5;
    st.global.u32 [%rd15+8], %f6;
    st.global.u32 [%rd15+12], %f7;
    st.global.u32 [%rd15+16], %f8;
    st.global.u32 [%rd15+20], %f9;
    cvt.rpi.s32.f32 %r1, %f1;
    cvt.rmi.s32.f32 %r2, %f1;
    cvt.rzi.u16.f32 %r3, %f2;
    cvt.rni.s64.f32 %rd1, %f2;
    mov.u32 %r4, -300;
    cvt.sat.u8.s32 %r5, %r4;
    mov.u32 %r6, 70000;
    cvt.sat.s16.s32 %r7, %r6;
    cvt.sat.u32.s32 %r8, %r4;
    cvt.rz.f32.s32 %f10, %r6;
    cvt.rp.f32.u32 %f11, %r6;
    st.global.u32 [%rd15+24], %r1;
    st.global.u32 [%rd15+28], %r2;
    st.global.u32 [%rd15+32], %r3;
    st.global.u64 [%rd15+40], %rd1;
    st.global.u32 [%rd15+48], %r5;
    st.global.u32 [%rd15+52], %r7;
    st.global.u32 [%rd15+56], %r8;
    st.global.u32 [%rd15+60], %f10;
    st.global.u32 [%rd15+64], %f11;"#,
        a = 1.25f32.to_bits(),
        b = 70000.5f32.to_bits(),
        n = f32::NAN.to_bits(),
    );
    let out = eval("", &body, 17);
    assert_eq!(f32::from_bits(out[0]), 1.0); // 1.25 + 70000.5 saturates
    assert_eq!(f32::from_bits(out[1]), 1.0); // 1.5625 saturates
    assert_eq!(f32::from_bits(out[2]), 1.0);
    assert!(f32::from_bits(out[3]).is_nan()); // max.NaN propagates
    assert_eq!(f32::from_bits(out[4]), 1.25);
    assert_eq!(f32::from_bits(out[5]), 1.25); // plain max ignores NaN
    assert_eq!(out[6] as i32, 2);
    assert_eq!(out[7] as i32, 1);
    assert_eq!(out[8], 0xffff); // 70000.5 saturates in u16
    assert_eq!(((u64::from(out[11]) << 32) | u64::from(out[10])) as i64, 70000);
    assert_eq!(out[12], 0);
    assert_eq!(out[13] as i32, 0x7fff);
    assert_eq!(out[14], 0);
    assert_eq!(f32::from_bits(out[15]), 70000.0);
    assert_eq!(f32::from_bits(out[16]), 70000.0);
}

#[test]
fn extern_shared_arrays_alias_one_dynamic_window_after_static_shared() {
    let source = r#".version 8.0
.target sm_80
.address_size 64
.extern .shared .align 16 .b8 dyn_a[];
.extern .shared .align 16 .b8 dyn_b[];
.visible .entry probe(.param .u64 out_param, .param .u32 n_param)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    .shared .align 4 .b8 fixed[16];
    ld.param.u64 %rd0, [out_param];
    cvta.to.global.u64 %rd1, %rd0;
    ld.param.u32 %r0, [n_param];
    st.shared.u32 [dyn_a], %r0;
    ld.shared.u32 %r1, [dyn_b];
    mov.u32 %r3, 7;
    st.shared.u32 [fixed], %r3;
    mov.u32 %r2, dyn_a;
    st.global.u32 [%rd1], %r1;
    st.global.u32 [%rd1+4], %r2;
    ret;
}
"#;
    // dyn_b aliases dyn_a; the window starts after the 16 static bytes.
    assert_eq!(run(source, 9, 2), [9, 16]);
}

#[test]
fn saturating_and_wide_integer_arithmetic() {
    let body = r#"    mov.u32 %r1, 0x7ffffff0;
    mov.u32 %r2, 0x100;
    mov.u32 %r3, 0x80000010;
    add.sat.s32 %r4, %r1, %r2;
    sub.sat.s32 %r5, %r3, %r2;
    add.sat.s32 %r6, %r2, %r2;
    mov.u64 %rd1, 5;
    mad.wide.s32 %rd2, %r3, %r2, %rd1;
    mad.wide.u32 %rd3, %r3, %r2, %rd1;
    st.global.u32 [%rd15], %r4;
    st.global.u32 [%rd15+4], %r5;
    st.global.u32 [%rd15+8], %r6;
    st.global.u64 [%rd15+16], %rd2;
    st.global.u64 [%rd15+24], %rd3;"#;
    let out = eval("", body, 8);
    assert_eq!(out[0], i32::MAX as u32);
    assert_eq!(out[1], i32::MIN as u32);
    assert_eq!(out[2], 0x200);
    let wide = |i: usize| (u64::from(out[i + 1]) << 32) | u64::from(out[i]);
    assert_eq!(wide(4), ((0x8000_0010u32 as i32 as i64) * 0x100 + 5) as u64);
    assert_eq!(wide(6), 0x8000_0010u64 * 0x100 + 5);
}

#[test]
fn half2_atomic_add_translates_and_validates() {
    lower(&kernel(
        "",
        r#"    mov.b32 %r1, 0x3c003c00;
    atom.global.add.noftz.f16x2 %r2, [%rd15], %r1;
    red.global.add.noftz.f16x2 [%rd15+4], %r1;
    atom.global.add.noftz.bf16x2 %r3, [%rd15+8], %r1;
    st.global.u32 [%rd15+12], %r2;"#,
    ));
}

fn run_wmma(a_row: bool, b_row: bool, shared_staging: bool) -> Vec<f32> {
    let (a_at, b_at) = (
        |r: usize, c: usize| ((r * 3 + c * 5 + 1) % 7) as f32 - 3.0,
        |r: usize, c: usize| ((r * 2 + c * 7 + 3) % 5) as f32 - 2.0,
    );
    let (al, bl) = (if a_row { "row" } else { "col" }, if b_row { "row" } else { "col" });
    let regs = "    .reg .b32 %wa<8>;\n    .reg .b32 %wb<8>;\n    .reg .f32 %wc<8>;\n    .reg .f32 %wd<8>;";
    let list = |p: &str| (0..8).map(|i| format!("%{p}{i}")).collect::<Vec<_>>().join(", ");
    let zero: String = (0..8).map(|i| format!("    mov.b32 %wc{i}, 0;\n")).collect();
    let body = if shared_staging {
        // Copy the inputs through shared memory and load the fragments from there.
        format!(
            r#"    add.s64 %rd1, %rd15, 65536;
    mov.u32 %r1, sbuf;
    mov.u32 %r2, %tid.x;
    mov.u32 %r10, 0;
$copy:
    mul.wide.u32 %rd3, %r10, 4;
    add.s64 %rd4, %rd1, %rd3;
    ld.global.u32 %r11, [%rd4];
    shl.b32 %r12, %r10, 2;
    add.u32 %r13, %r1, %r12;
    st.shared.u32 [%r13], %r11;
    add.u32 %r10, %r10, 1;
    setp.lt.u32 %p2, %r10, 256;
    @%p2 bra $copy;
    bar.sync 0;
    add.u32 %r3, %r1, 512;
    wmma.load.a.sync.aligned.{al}.m16n16k16.shared.f16 {{{a}}}, [%r1], 16;
    wmma.load.b.sync.aligned.{bl}.m16n16k16.shared.f16 {{{b}}}, [%r3], 16;
{zero}    wmma.mma.sync.aligned.{al}.{bl}.m16n16k16.f32.f32 {{{d}}}, {{{a}}}, {{{b}}}, {{{c}}};
    wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%rd15], {{{d}}}, 16;"#,
            a = list("wa"),
            b = list("wb"),
            c = list("wc"),
            d = list("wd"),
        )
    } else {
        format!(
            r#"    add.s64 %rd1, %rd15, 65536;
    add.s64 %rd2, %rd1, 512;
    wmma.load.a.sync.aligned.{al}.m16n16k16.global.f16 {{{a}}}, [%rd1], 16;
    wmma.load.b.sync.aligned.{bl}.m16n16k16.global.f16 {{{b}}}, [%rd2], 16;
{zero}    wmma.mma.sync.aligned.{al}.{bl}.m16n16k16.f32.f32 {{{d}}}, {{{a}}}, {{{b}}}, {{{c}}};
    wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%rd15], {{{d}}}, 16;"#,
            a = list("wa"),
            b = list("wb"),
            c = list("wc"),
            d = list("wd"),
        )
    };
    let shared = if shared_staging { ".shared .align 16 .b8 sbuf[1024];" } else { "" };
    let source = super::coverage_tests::kernel_with_shared(regs, shared, &body);
    let module = lower(&source);
    let mut machine = interp::Machine {
        params: vec![interp::GLOBAL, 0],
        ..Default::default()
    };
    let input = interp::GLOBAL + 0x10000;
    for r in 0..16 {
        for c in 0..16 {
            let a_addr = if a_row { r * 16 + c } else { c * 16 + r };
            let b_addr = if b_row { r * 16 + c } else { c * 16 + r };
            machine.write(input + 2 * a_addr as u64, 2, u64::from(f2h(a_at(r, c))));
            machine.write(input + 512 + 2 * b_addr as u64, 2, u64::from(f2h(b_at(r, c))));
        }
    }
    interp::run_lanes(&module, &mut machine, 32);
    let mut expected = Vec::new();
    for r in 0..16 {
        for c in 0..16 {
            let want: f32 = (0..16).map(|k| a_at(r, k) * b_at(k, c)).sum();
            expected.push(want);
            let got = f32::from_bits(machine.read(interp::GLOBAL + 4 * (r * 16 + c) as u64, 4) as u32);
            assert_eq!(got, want, "D[{r}][{c}] (a_row={a_row}, b_row={b_row})");
        }
    }
    expected
}

#[test]
fn wmma_m16n16k16_f16_is_emulated_on_top_of_mma() {
    run_wmma(true, false, false);
    // Layouts that need element-wise 16-bit gathers.
    run_wmma(false, true, false);
    run_wmma(true, true, false);
}

#[test]
fn wmma_loads_from_shared_memory() {
    run_wmma(true, false, true);
}

#[test]
fn wmma_with_half_accumulators_translates_and_validates() {
    let regs = "    .reg .b32 %wa<8>;\n    .reg .b32 %wb<8>;\n    .reg .b32 %wc<4>;\n    .reg .b32 %wd<4>;";
    let body = r#"    add.s64 %rd1, %rd15, 65536;
    wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {%wa0, %wa1, %wa2, %wa3, %wa4, %wa5, %wa6, %wa7}, [%rd1], 16;
    wmma.load.b.sync.aligned.col.m16n16k16.global.f16 {%wb0, %wb1, %wb2, %wb3, %wb4, %wb5, %wb6, %wb7}, [%rd1+512], 16;
    wmma.load.c.sync.aligned.row.m16n16k16.global.f16 {%wc0, %wc1, %wc2, %wc3}, [%rd1+1024], 16;
    wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16 {%wd0, %wd1, %wd2, %wd3}, {%wa0, %wa1, %wa2, %wa3, %wa4, %wa5, %wa6, %wa7}, {%wb0, %wb1, %wb2, %wb3, %wb4, %wb5, %wb6, %wb7}, {%wc0, %wc1, %wc2, %wc3};
    wmma.store.d.sync.aligned.row.m16n16k16.global.f16 [%rd15], {%wd0, %wd1, %wd2, %wd3}, 16;"#;
    lower(&super::coverage_tests::kernel_with_shared(regs, "", body));
}
