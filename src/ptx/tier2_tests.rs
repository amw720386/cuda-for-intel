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
