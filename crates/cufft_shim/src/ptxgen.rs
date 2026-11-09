//! PTX for the cuFFT shim's three kernels, one module per precision. The Driver API
//! translates it to SPIR-V like any application PTX (the same route the cuBLAS shim's GEMMs
//! take). Index math matches `plan::execute_host`; parameter order matches the
//! `plan::Desc::*_args` builders. Dependency-free so `src/ptx/cufft_tests.rs` can include it.
//!
//! * `cfi_fft_gather_<p>(src, dst, total, n0, n1, n2, sb, s0, s1, s2, mode)`:
//!   mode 0 complex, 1 real (imaginary part 0), 2 Hermitian half (mirrored and conjugated
//!   for `x2 > n2 / 2`).
//! * `cfi_fft_stage_<p>(src, dst, tw, total, n, sd, r, p, twstep, inverse)`: one radix-`r`
//!   Stockham pass, one output per thread, twiddles from a table of `W_n^t`.
//! * `cfi_fft_scatter_<p>(src, dst, total, n0, n1, m2, n2, sb, s0, s1, s2, mode)`:
//!   mode 0 complex, 1 real part only.

/// Number of parameters of (gather, stage, scatter).
pub const PARAMS: [usize; 3] = [11, 10, 12];

pub fn suffix(double: bool) -> &'static str {
    if double { "f64" } else { "f32" }
}

pub fn gather_name(double: bool) -> String {
    format!("cfi_fft_gather_{}", suffix(double))
}

pub fn stage_name(double: bool) -> String {
    format!("cfi_fft_stage_{}", suffix(double))
}

pub fn scatter_name(double: bool) -> String {
    format!("cfi_fft_scatter_{}", suffix(double))
}

fn params(name: &str, kinds: &[&str]) -> String {
    let p: Vec<String> = kinds.iter().enumerate().map(|(i, k)| format!(".param .{k} {name}_p{i}")).collect();
    format!(".visible .entry {name}(\n{}\n)\n", p.join(",\n"))
}

fn ld_params(name: &str, regs: &[(&str, &str)]) -> String {
    regs.iter()
        .enumerate()
        .map(|(i, (k, r))| format!("ld.param.{k} {r}, [{name}_p{i}];\n"))
        .collect()
}

const GID: &str = "mov.u32 %r20, %ctaid.x;\nmov.u32 %r21, %ntid.x;\nmov.u32 %r22, %tid.x;\nmad.lo.u32 %r9, %r20, %r21, %r22;\n";

fn decls(t: &str) -> String {
    format!(".reg .pred %p<8>;\n.reg .b32 %r<32>;\n.reg .b64 %rd<32>;\n.reg .{t} %f<16>;\n")
}

fn gather(double: bool) -> String {
    let (t, sz) = if double { ("f64", 8) } else { ("f32", 4) };
    let csz = 2 * sz;
    let name = gather_name(double);
    let zero = if double { "0d0000000000000000" } else { "0f00000000" };
    let mut s = params(&name, &["u64", "u64", "u32", "u32", "u32", "u32", "u64", "u64", "u64", "u64", "u32"]);
    s += "{\n";
    s += &decls(t);
    s += &ld_params(
        &name,
        &[("u64", "%rd1"), ("u64", "%rd2"), ("u32", "%r1"), ("u32", "%r2"), ("u32", "%r3"), ("u32", "%r4"),
          ("u64", "%rd3"), ("u64", "%rd4"), ("u64", "%rd5"), ("u64", "%rd6"), ("u32", "%r5")],
    );
    s += "cvta.to.global.u64 %rd1, %rd1;\ncvta.to.global.u64 %rd2, %rd2;\n";
    s += GID;
    s += "setp.ge.u32 %p1, %r9, %r1;\n@%p1 bra $L_done;\n";
    // x2 = r10, x1 = r12, x0 = r13, b = r14
    s += "rem.u32 %r10, %r9, %r4;\ndiv.u32 %r11, %r9, %r4;\nrem.u32 %r12, %r11, %r3;\ndiv.u32 %r11, %r11, %r3;\n\
          rem.u32 %r13, %r11, %r2;\ndiv.u32 %r14, %r11, %r2;\n";
    // Hermitian mirror (mode 2, x2 > n2/2): r16 = conjugate flag
    s += "mov.u32 %r16, 0;\nsetp.ne.u32 %p2, %r5, 2;\n@%p2 bra $L_index;\nshr.u32 %r15, %r4, 1;\nsetp.le.u32 %p3, %r10, %r15;\n@%p3 bra $L_index;\n\
          sub.u32 %r10, %r4, %r10;\nsub.u32 %r12, %r3, %r12;\nrem.u32 %r12, %r12, %r3;\nsub.u32 %r13, %r2, %r13;\nrem.u32 %r13, %r13, %r2;\nmov.u32 %r16, 1;\n$L_index:\n";
    // off = b*sb + x0*s0 + x1*s1 + x2*s2
    s += "cvt.u64.u32 %rd7, %r14;\nmul.lo.u64 %rd8, %rd7, %rd3;\ncvt.u64.u32 %rd7, %r13;\nmad.lo.u64 %rd8, %rd7, %rd4, %rd8;\n\
          cvt.u64.u32 %rd7, %r12;\nmad.lo.u64 %rd8, %rd7, %rd5, %rd8;\ncvt.u64.u32 %rd7, %r10;\nmad.lo.u64 %rd8, %rd7, %rd6, %rd8;\n";
    s += &format!("setp.ne.u32 %p4, %r5, 1;\n@%p4 bra $L_complex;\nmul.lo.u64 %rd9, %rd8, {sz};\nadd.u64 %rd9, %rd1, %rd9;\nld.global.{t} %f1, [%rd9];\nmov.{t} %f2, {zero};\nbra $L_store;\n");
    s += &format!("$L_complex:\nmul.lo.u64 %rd9, %rd8, {csz};\nadd.u64 %rd9, %rd1, %rd9;\nld.global.{t} %f1, [%rd9];\nld.global.{t} %f2, [%rd9+{sz}];\n\
                   setp.ne.u32 %p5, %r16, 0;\n@%p5 neg.{t} %f2, %f2;\n");
    s += &format!("$L_store:\nmul.wide.u32 %rd10, %r9, {csz};\nadd.u64 %rd10, %rd2, %rd10;\nst.global.{t} [%rd10], %f1;\nst.global.{t} [%rd10+{sz}], %f2;\n");
    s += "$L_done:\nret;\n}\n";
    s
}

fn stage(double: bool) -> String {
    let (t, sz) = if double { ("f64", 8) } else { ("f32", 4) };
    let csz = 2 * sz;
    let name = stage_name(double);
    let zero = if double { "0d0000000000000000" } else { "0f00000000" };
    let mut s = params(&name, &["u64", "u64", "u64", "u32", "u32", "u32", "u32", "u32", "u32", "u32"]);
    s += "{\n";
    s += &decls(t);
    s += &ld_params(
        &name,
        &[("u64", "%rd1"), ("u64", "%rd2"), ("u64", "%rd3"), ("u32", "%r1"), ("u32", "%r2"), ("u32", "%r3"),
          ("u32", "%r4"), ("u32", "%r5"), ("u32", "%r6"), ("u32", "%r7")],
    );
    s += "cvta.to.global.u64 %rd1, %rd1;\ncvta.to.global.u64 %rd2, %rd2;\ncvta.to.global.u64 %rd3, %rd3;\n";
    s += GID;
    s += "setp.ge.u32 %p1, %r9, %r1;\n@%p1 bra $L_done;\nsetp.ne.u32 %p2, %r7, 0;\n";
    // inner = r10, o = r11, outer = r12; base = r13
    s += "rem.u32 %r10, %r9, %r3;\ndiv.u32 %r14, %r9, %r3;\nrem.u32 %r11, %r14, %r2;\ndiv.u32 %r12, %r14, %r2;\n\
          mul.lo.u32 %r14, %r12, %r2;\nmad.lo.u32 %r13, %r14, %r3, %r10;\n";
    // pR = r15, rem = r16, i = r17, m = r18; idx = r19, step = r23, e = r24, q = r25
    s += "mul.lo.u32 %r15, %r5, %r4;\nrem.u32 %r16, %r11, %r15;\ndiv.u32 %r14, %r11, %r15;\nrem.u32 %r17, %r16, %r5;\nmad.lo.u32 %r17, %r14, %r5, %r17;\n\
          div.u32 %r18, %r2, %r4;\nmad.lo.u32 %r19, %r17, %r3, %r13;\nmul.lo.u32 %r23, %r18, %r3;\nmov.u32 %r24, 0;\nmov.u32 %r25, 0;\n";
    s += &format!("mov.{t} %f1, {zero};\nmov.{t} %f2, {zero};\n");
    s += "$L_loop:\n";
    s += &format!("mul.wide.u32 %rd4, %r19, {csz};\nadd.u64 %rd4, %rd1, %rd4;\nld.global.{t} %f3, [%rd4];\nld.global.{t} %f4, [%rd4+{sz}];\n");
    s += &format!("mul.lo.u32 %r26, %r24, %r6;\nmul.wide.u32 %rd5, %r26, {csz};\nadd.u64 %rd5, %rd3, %rd5;\nld.global.{t} %f5, [%rd5];\nld.global.{t} %f6, [%rd5+{sz}];\n@%p2 neg.{t} %f6, %f6;\n");
    // acc += x * w
    s += &format!("fma.rn.{t} %f1, %f3, %f5, %f1;\nneg.{t} %f7, %f4;\nfma.rn.{t} %f1, %f7, %f6, %f1;\nfma.rn.{t} %f2, %f3, %f6, %f2;\nfma.rn.{t} %f2, %f4, %f5, %f2;\n");
    s += "add.u32 %r19, %r19, %r23;\nadd.u32 %r24, %r24, %r16;\nsetp.ge.u32 %p3, %r24, %r15;\n@%p3 sub.u32 %r24, %r24, %r15;\n\
          add.u32 %r25, %r25, 1;\nsetp.lt.u32 %p4, %r25, %r4;\n@%p4 bra $L_loop;\n";
    s += &format!("mad.lo.u32 %r26, %r11, %r3, %r13;\nmul.wide.u32 %rd6, %r26, {csz};\nadd.u64 %rd6, %rd2, %rd6;\nst.global.{t} [%rd6], %f1;\nst.global.{t} [%rd6+{sz}], %f2;\n");
    s += "$L_done:\nret;\n}\n";
    s
}

fn scatter(double: bool) -> String {
    let (t, sz) = if double { ("f64", 8) } else { ("f32", 4) };
    let csz = 2 * sz;
    let name = scatter_name(double);
    let mut s = params(&name, &["u64", "u64", "u32", "u32", "u32", "u32", "u32", "u64", "u64", "u64", "u64", "u32"]);
    s += "{\n";
    s += &decls(t);
    s += &ld_params(
        &name,
        &[("u64", "%rd1"), ("u64", "%rd2"), ("u32", "%r1"), ("u32", "%r2"), ("u32", "%r3"), ("u32", "%r4"), ("u32", "%r6"),
          ("u64", "%rd3"), ("u64", "%rd4"), ("u64", "%rd5"), ("u64", "%rd6"), ("u32", "%r5")],
    );
    s += "cvta.to.global.u64 %rd1, %rd1;\ncvta.to.global.u64 %rd2, %rd2;\n";
    s += GID;
    s += "setp.ge.u32 %p1, %r9, %r1;\n@%p1 bra $L_done;\n";
    // x2 = r10, x1 = r12, x0 = r13, b = r14 (over the m2-wide output)
    s += "rem.u32 %r10, %r9, %r4;\ndiv.u32 %r11, %r9, %r4;\nrem.u32 %r12, %r11, %r3;\ndiv.u32 %r11, %r11, %r3;\n\
          rem.u32 %r13, %r11, %r2;\ndiv.u32 %r14, %r11, %r2;\n";
    // dense source index ((b*n0 + x0)*n1 + x1)*n2 + x2
    s += "mad.lo.u32 %r15, %r14, %r2, %r13;\nmad.lo.u32 %r15, %r15, %r3, %r12;\nmad.lo.u32 %r15, %r15, %r6, %r10;\n";
    s += &format!("mul.wide.u32 %rd9, %r15, {csz};\nadd.u64 %rd9, %rd1, %rd9;\nld.global.{t} %f1, [%rd9];\nld.global.{t} %f2, [%rd9+{sz}];\n");
    s += "cvt.u64.u32 %rd7, %r14;\nmul.lo.u64 %rd8, %rd7, %rd3;\ncvt.u64.u32 %rd7, %r13;\nmad.lo.u64 %rd8, %rd7, %rd4, %rd8;\n\
          cvt.u64.u32 %rd7, %r12;\nmad.lo.u64 %rd8, %rd7, %rd5, %rd8;\ncvt.u64.u32 %rd7, %r10;\nmad.lo.u64 %rd8, %rd7, %rd6, %rd8;\n";
    s += &format!("setp.ne.u32 %p4, %r5, 1;\n@%p4 bra $L_complex;\nmul.lo.u64 %rd10, %rd8, {sz};\nadd.u64 %rd10, %rd2, %rd10;\nst.global.{t} [%rd10], %f1;\nbra $L_done;\n");
    s += &format!("$L_complex:\nmul.lo.u64 %rd10, %rd8, {csz};\nadd.u64 %rd10, %rd2, %rd10;\nst.global.{t} [%rd10], %f1;\nst.global.{t} [%rd10+{sz}], %f2;\n");
    s += "$L_done:\nret;\n}\n";
    s
}

/// The whole module for one precision (all three entries).
pub fn module(double: bool) -> String {
    format!(".version 7.0\n.target sm_60\n.address_size 64\n\n{}\n{}\n{}", gather(double), stage(double), scatter(double))
}

/// `(entry name, single-entry module source, parameter count)` for every kernel.
pub fn all_entries() -> Vec<(String, String, usize)> {
    let mut v = Vec::new();
    for double in [false, true] {
        let head = ".version 7.0\n.target sm_60\n.address_size 64\n\n";
        v.push((gather_name(double), format!("{head}{}", gather(double)), PARAMS[0]));
        v.push((stage_name(double), format!("{head}{}", stage(double)), PARAMS[1]));
        v.push((scatter_name(double), format!("{head}{}", scatter(double)), PARAMS[2]));
    }
    v
}
