//! Hardware checks for translator edge cases the simulator cannot decide. Written during the
//! translator-hardening pass and NOT yet run on hardware (the test box was unavailable).
//! Each test names the question it answers; expectations follow the PTX ISA / NVIDIA behaviour.
use super::hw_common::*;

const TID: &str = "\tmov.u32 %r0, %tid.x;\n\tmul.wide.u32 %rd20, %r0, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tadd.s64 %rd22, %rd2, %rd20;\n";

fn run(entry: &str, body: &str, block: u32, input: &[u32], out_words: usize) -> Vec<u32> {
    let mut s = prologue(entry, 2, 0, "");
    s += TID;
    s += body;
    s += "\tret;\n}\n";
    let mut bufs = vec![bytes_of_u32(input), vec![0xeeu8; out_words * 4]];
    launch(&s, entry, 1, block, &mut bufs, &[]).unwrap();
    u32s_of(&bufs[1])
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU; not yet run"]
fn hw_integer_division_edge_cases_are_defined() {
    // inputs: a[i], b[i]; out: div.u32, rem.u32, div.s32, rem.s32 per thread (stride 4 words)
    let pairs: [(u32, u32); 6] = [(7, 0), (0x8000_0000, 0xffff_ffff), (100, 7), (0xffff_fff0, 3), (0, 0), (5, 0xffff_ffff)];
    let mut input = vec![0u32; 64];
    for (i, (a, b)) in pairs.iter().enumerate() {
        input[i] = *a;
        input[32 + i] = *b;
    }
    let body = "\tld.global.u32 %r10, [%rd21];\n\tld.global.u32 %r11, [%rd21+128];\n\tdiv.u32 %r12, %r10, %r11;\n\trem.u32 %r13, %r10, %r11;\n\tdiv.s32 %r14, %r10, %r11;\n\trem.s32 %r15, %r10, %r11;\n\tshl.b32 %r16, %r0, 2;\n\tmul.wide.u32 %rd23, %r16, 4;\n\tadd.s64 %rd24, %rd2, %rd23;\n\tst.global.u32 [%rd24], %r12;\n\tst.global.u32 [%rd24+4], %r13;\n\tst.global.u32 [%rd24+8], %r14;\n\tst.global.u32 [%rd24+12], %r15;\n";
    let out = run("div_edge", body, 32, &input, 128);
    for (i, (a, b)) in pairs.iter().enumerate() {
        let (ud, ur) = (a.checked_div(*b).unwrap_or(u32::MAX), a.checked_rem(*b).unwrap_or(*a));
        let (sd, sr) = if *b == 0 { (u32::MAX, *a) } else { ((*a as i32).wrapping_div(*b as i32) as u32, (*a as i32).wrapping_rem(*b as i32) as u32) };
        assert_eq!(&out[i * 4..i * 4 + 4], &[ud, ur, sd, sr], "a={a:#x} b={b:#x}");
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU; not yet run"]
fn hw_shfl_lane_operand_uses_only_low_five_bits() {
    let input: Vec<u32> = (0..32).map(|i| 1000 + i).collect();
    let mut s = String::new();
    for (name, op) in [("up", "up"), ("down", "down"), ("bfly", "bfly"), ("idx", "idx")] {
        s += &format!("\tshfl.sync.{op}.b32 %r2{}, %r10, 33, 31, 0xffffffff;\n", ["up", "down", "bfly", "idx"].iter().position(|n| *n == name).unwrap());
    }
    let body = format!("\tld.global.u32 %r10, [%rd21];\n{s}\tshl.b32 %r16, %r0, 2;\n\tmul.wide.u32 %rd23, %r16, 4;\n\tadd.s64 %rd24, %rd2, %rd23;\n\tst.global.u32 [%rd24], %r20;\n\tst.global.u32 [%rd24+4], %r21;\n\tst.global.u32 [%rd24+8], %r22;\n\tst.global.u32 [%rd24+12], %r23;\n");
    let out = run("shfl_b33", &body, 32, &input, 128);
    for l in 0..32usize {
        let up = if l >= 1 { input[l - 1] } else { input[l] };
        let down = if l + 1 <= 31 { input[l + 1] } else { input[l] };
        assert_eq!(&out[l * 4..l * 4 + 4], &[up, down, input[l ^ 1], input[1]], "lane {l}");
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU; not yet run"]
fn hw_barrier_with_exited_threads_completes() {
    // Threads >= 48 exit before bar.sync 0; PTX counts exited threads as arrived.
    // Registers start at %rd12: prologue() keeps the param pointers in %rd8/%rd9, and the translator
    // types a register by name, so reusing %rd8 for a shared address made the module fail to load (218).
    let body = "\tsetp.ge.u32 %p1, %r0, 48;\n\t@%p1 bra $exit;\n\tmov.u64 %rd12, sh;\n\tcvt.u64.u32 %rd13, %r0;\n\tshl.b64 %rd13, %rd13, 2;\n\tadd.s64 %rd14, %rd12, %rd13;\n\tst.shared.u32 [%rd14], %r0;\n\tbar.sync 0;\n\txor.b32 %r5, %r0, 1;\n\tcvt.u64.u32 %rd15, %r5;\n\tshl.b64 %rd15, %rd15, 2;\n\tadd.s64 %rd16, %rd12, %rd15;\n\tld.shared.u32 %r6, [%rd16];\n\tst.global.u32 [%rd22], %r6;\n$exit:\n";
    let mut s = prologue("bar_exit", 2, 0, "\t.shared .align 4 .b8 sh[1024];\n");
    s += TID;
    s += body;
    s += "\tret;\n}\n";
    let mut bufs = vec![vec![0u8; 256], vec![0xeeu8; 256]];
    launch(&s, "bar_exit", 1, 64, &mut bufs, &[]).unwrap();
    let out = u32s_of(&bufs[1]);
    for t in 0..48usize {
        assert_eq!(out[t], (t ^ 1) as u32, "tid {t}");
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU; not yet run"]
fn hw_directed_rounding_and_overflow() {
    // in: [a, b, c] bit patterns; out: fma.rz, fma.rm(+overflow), add.rz, cvt.rz.f32.s32
    let big = 0x7f7f_ffffu32;
    let input = [big, big, 0x3f80_0000, 0x7f7f_ffff, 0x4f00_0001, 0x3dcc_cccd];
    let body = "\tld.global.u32 %r10, [%rd21];\n\tmov.b32 %f1, %r10;\n\tmov.u32 %r11, 0x40000000;\n\tmov.b32 %f2, %r11;\n\tfma.rz.f32 %f3, %f1, %f2, %f1;\n\tmov.b32 %r20, %f3;\n\tmov.u32 %r12, 0x3dcccccd;\n\tmov.b32 %f4, %r12;\n\tadd.rz.f32 %f5, %f1, %f4;\n\tmov.b32 %r21, %f5;\n\tmov.u32 %r13, 0x7fffffff;\n\tcvt.rz.f32.s32 %f6, %r13;\n\tmov.b32 %r22, %f6;\n\tst.global.u32 [%rd22], %r20;\n\tst.global.u32 [%rd22+128], %r21;\n\tst.global.u32 [%rd22+256], %r22;\n";
    let out = run("dir_round", body, 1, &input, 128);
    assert_eq!(out[0], 0x7f7f_ffff, "fma.rz overflow must saturate to MAX");
    assert_eq!(out[32], 0x7f7f_ffff, "add.rz of MAX + 0.1");
    assert_eq!(out[64], 0x4eff_ffff, "cvt.rz.f32.s32 of INT_MAX");
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU; not yet run"]
fn hw_ftz_and_denormals_are_observable() {
    // Records (does not assert) whether .ftz flushes: PTX requires 0; the translator ignores .ftz.
    let body = "\tmov.u32 %r10, 0x00400000;\n\tmov.b32 %f1, %r10;\n\tadd.ftz.f32 %f2, %f1, %f1;\n\tmov.b32 %r20, %f2;\n\tmul.ftz.f32 %f3, %f1, 0f3f800000;\n\tmov.b32 %r21, %f3;\n\tst.global.u32 [%rd22], %r20;\n\tst.global.u32 [%rd22+4], %r21;\n";
    let out = run("ftz_probe", body, 1, &[0], 4);
    eprintln!("ftz probe: add.ftz(denorm,denorm) = {:#x} (PTX: 0), mul.ftz(denorm,1) = {:#x} (PTX: 0)", out[0], out[1]);
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU; not yet run"]
fn hw_cvt_sat_s8_result_is_sign_extended_in_16_bits() {
    let body = "\t.reg .b16 %h<4>;\n";
    let _ = body;
    let mut s = prologue("sat_s8", 2, 0, "\t.reg .b16 %h<8>;\n");
    s += TID;
    s += "\tmov.u32 %r10, 0xfffffff0;\n\tcvt.sat.s8.s32 %h1, %r10;\n\tcvt.s32.s16 %r20, %h1;\n\tmov.u32 %r11, 1000;\n\tcvt.sat.s8.s32 %h2, %r11;\n\tcvt.s32.s16 %r21, %h2;\n\tst.global.u32 [%rd22], %r20;\n\tst.global.u32 [%rd22+4], %r21;\n\tret;\n}\n";
    let mut bufs = vec![vec![0u8; 16], vec![0xeeu8; 16]];
    launch(&s, "sat_s8", 1, 1, &mut bufs, &[]).unwrap();
    let out = u32s_of(&bufs[1]);
    assert_eq!(out[0], (-16i32) as u32);
    assert_eq!(out[1], 127);
}
