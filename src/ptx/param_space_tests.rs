//! Computed-address kernel-parameter space, run on the SPIR-V interpreter.
use super::coverage_tests::{interp, lower};

const MODULE: &str = r#"
.version 7.8
.target sm_37
.address_size 64
.visible .entry k(.param .u64 out, .param .align 8 .b8 s[32], .param .u8 b)
{
    .reg .b16 %rs<4>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, s;
    add.s64 %rd4, %rd3, 4;
    ld.param.u8 %r1, [%rd4+1];
    ld.param.s8 %r2, [%rd3+8];
    ld.param.u16 %r3, [%rd3+10];
    ld.param.s16 %r4, [%rd3+8];
    ld.param.u64 %rd5, [%rd3+16];
    ld.param.v2.u32 {%r5, %r6}, [%rd3+12];
    add.s64 %rd6, %rd3, 20;
    ld.param.u32 %r7, [%rd6+-8];
    mov.u64 %rd7, b;
    ld.param.u8 %rs1, [%rd7];
    cvt.u32.u16 %r8, %rs1;
    ld.param.u8 %rs2, [b];
    cvt.u32.u16 %r9, %rs2;
    ld.param.v4.u32 {%r10, %r11, %r12, %r13}, [%rd3];
    st.global.u32 [%rd2], %r1;
    st.global.u32 [%rd2+4], %r2;
    st.global.u32 [%rd2+8], %r3;
    st.global.u32 [%rd2+12], %r4;
    st.global.u64 [%rd2+16], %rd5;
    st.global.u32 [%rd2+24], %r5;
    st.global.u32 [%rd2+28], %r6;
    st.global.u32 [%rd2+32], %r7;
    st.global.u32 [%rd2+36], %r8;
    st.global.u32 [%rd2+40], %r9;
    st.global.u32 [%rd2+44], %r10;
    st.global.u32 [%rd2+48], %r13;
    ret;
}
"#;

#[test]
fn computed_param_addresses_read_the_argument_buffer() {
    let module = lower(MODULE);
    let words = [0x0403_0201u32, 0x0807_0605, 0xfedc_ba98, 0x7654_3210, 0x5566_7788, 0x1122_3344, 0xdead_beef, 0x0bad_f00d];
    let mut params = vec![interp::GLOBAL];
    params.extend(words.iter().map(|&w| u64::from(w)));
    params.push(0xab);
    let mut machine = interp::Machine { params, ..Default::default() };
    interp::run(&module, &mut machine);
    let word = |i: u64| machine.read(interp::GLOBAL + 4 * i, 4) as u32;
    assert_eq!(word(0), 0x06); // byte 5 of the struct
    assert_eq!(word(1), 0xffff_ff98);
    assert_eq!(word(2), 0xfedc);
    assert_eq!(word(3), 0xffff_ba98);
    assert_eq!((word(4), word(5)), (0x5566_7788, 0x1122_3344));
    assert_eq!((word(6), word(7)), (0x7654_3210, 0x5566_7788));
    assert_eq!(word(8), 0x7654_3210); // [s+20-8] = word 3
    assert_eq!(word(9), 0xab);
    assert_eq!(word(10), 0xab);
    assert_eq!((word(11), word(12)), (0x0403_0201, 0x7654_3210));
}

/// A register read in a loop header before the (later-lowered) write that feeds it
/// behaves as zero on the first trip instead of failing translation.
#[test]
fn loop_carried_register_without_prior_write_reads_as_zero() {
    let module = lower(
        r#"
.version 7.8
.target sm_37
.address_size 64
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 0;
$L_loop:
    add.s64 %rd5, %rd4, 5;
    mov.u64 %rd4, %rd5;
    add.s32 %r1, %r1, 1;
    setp.lt.u32 %p1, %r1, 3;
    @%p1 bra $L_loop;
    st.global.u64 [%rd2], %rd4;
    ret;
}
"#,
    );
    let mut machine = interp::Machine { params: vec![interp::GLOBAL], ..Default::default() };
    interp::run(&module, &mut machine);
    assert_eq!(machine.read(interp::GLOBAL, 8), 15);
}

/// Aggregates above `STAGE_BYTES` arrive as a pointer to a staged copy.
#[test]
fn staged_aggregates_are_read_through_their_pointer() {
    let module = lower(
        r#"
.version 7.8
.target sm_37
.address_size 64
.visible .entry k(.param .u64 out, .param .align 8 .b8 s[2000], .param .u8 b)
{
    .reg .b16 %rs<2>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, s;
    add.s64 %rd4, %rd3, 8;
    ld.param.u32 %r1, [%rd4+4];
    ld.param.u8 %r2, [%rd3+1];
    ld.param.u64 %rd5, [s+16];
    ld.param.u32 %r3, [s+1996];
    ld.param.s8 %r4, [b];
    st.global.u32 [%rd2], %r1;
    st.global.u32 [%rd2+4], %r2;
    st.global.u64 [%rd2+8], %rd5;
    st.global.u32 [%rd2+16], %r3;
    st.global.u32 [%rd2+20], %r4;
    ret;
}
"#,
    );
    let data = interp::GLOBAL + 0x4000;
    let mut machine = interp::Machine { params: vec![interp::GLOBAL, data, 0xf0], ..Default::default() };
    machine.write(data + 1, 1, 0x7e);
    machine.write(data + 12, 4, 0xabcd_0123);
    machine.write(data + 16, 8, 0x1122_3344_5566_7788);
    machine.write(data + 1996, 4, 77);
    interp::run(&module, &mut machine);
    assert_eq!(machine.read(interp::GLOBAL, 4), 0xabcd_0123);
    assert_eq!(machine.read(interp::GLOBAL + 4, 4), 0x7e);
    assert_eq!(machine.read(interp::GLOBAL + 8, 8), 0x1122_3344_5566_7788);
    assert_eq!(machine.read(interp::GLOBAL + 16, 4), 77);
    assert_eq!(machine.read(interp::GLOBAL + 20, 4), 0xffff_fff0);
}
