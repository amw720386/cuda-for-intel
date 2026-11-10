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
//! * `cfi_fft_local_<p>(src, dst, tw, tw2, nl, l, ni, sa, sb, sj, da, db, dj, radices,
//!   inverse, g, log2 g)`: whole `l`-point transforms of `g` (a power of two) lines per work-group of
//!   [`LOCAL_BLOCK`] threads in shared memory (`plan::Local`), every stage between barriers.
//! * `cfi_fft_chirp_<p>(src, dst, tab, total, nd, ns, sd)`: Bluestein's pointwise products
//!   (`plan::Chirp`).

/// Number of parameters of (gather, stage, scatter, local, chirp).
pub const PARAMS: [usize; 5] = [11, 10, 12, 17, 7];

/// Threads of a `local` work-group.
pub const LOCAL_BLOCK: u32 = 256;

pub fn local_name(double: bool) -> String {
    format!("cfi_fft_local_{}", suffix(double))
}

pub fn chirp_name(double: bool) -> String {
    format!("cfi_fft_chirp_{}", suffix(double))
}

/// Shared-memory FFT of whole lines (see `plan::Local` and `plan::local_host`).
fn local(double: bool) -> String {
    let (t, sz) = if double { ("f64", 8) } else { ("f32", 4) };
    let csz = 2 * sz;
    let cap = crate_local_cap(double);
    let name = local_name(double);
    let zero = if double { "0d0000000000000000" } else { "0f00000000" };
    let mut s = params(&name, &["u64"; 17]);
    s += "{\n";
    s += &format!(".reg .pred %p<16>;\n.reg .b32 %r<64>;\n.reg .b64 %rd<32>;\n.reg .{t} %f<16>;\n.shared .align 16 .b8 smem[{}];\n", 2 * cap * csz);
    for i in 0..17 {
        s += &format!("ld.param.u64 %rd{i}, [{name}_p{i}];\n");
    }
    // log2 g (parameter 16) before %rd16 is reused for the source pointer
    s += "cvt.u32.u64 %r42, %rd16;\n";
    s += "cvta.to.global.u64 %rd16, %rd0;\ncvta.to.global.u64 %rd17, %rd1;\ncvta.to.global.u64 %rd18, %rd2;\ncvta.to.global.u64 %rd19, %rd3;\n";
    s += "mov.u32 %r1, %ctaid.x;\nmov.u32 %r2, %tid.x;\n";
    for (r, rd) in [(3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 15)] {
        s += &format!("cvt.u32.u64 %r{r}, %rd{rd};\n");
    }
    s += &format!("sub.u32 %r43, %r12, 1;\nmul.lo.u32 %r13, %r12, %r4;\nmul.lo.u32 %r14, %r1, %r12;\nsetp.ne.u64 %p9, %rd14, 0;\nsetp.ne.u64 %p10, %rd3, 0;\nmov.u32 %r15, smem;\nadd.u32 %r16, %r15, {};\n", cap * csz);
    // load: element e = (line e % g, index e / g) to shared slot e
    s += "mov.u32 %r17, %r2;\n$L_ld:\nsetp.ge.u32 %p1, %r17, %r13;\n@%p1 bra $L_ldd;\nand.b32 %r18, %r17, %r43;\nshr.u32 %r19, %r17, %r42;\nadd.u32 %r20, %r14, %r18;\n";
    s += &format!("mov.{t} %f1, {zero};\nmov.{t} %f2, {zero};\nsetp.ge.u32 %p2, %r20, %r3;\n@%p2 bra $L_lz;\n");
    s += "div.u32 %r21, %r20, %r5;\nrem.u32 %r22, %r20, %r5;\nmul.lo.u32 %r23, %r21, %r6;\nmad.lo.u32 %r23, %r22, %r7, %r23;\nmad.lo.u32 %r23, %r19, %r8, %r23;\n";
    s += &format!("mul.wide.u32 %rd20, %r23, {csz};\nadd.u64 %rd20, %rd16, %rd20;\nld.global.{t} %f1, [%rd20];\nld.global.{t} %f2, [%rd20+{sz}];\n$L_lz:\n");
    s += &format!("mad.lo.u32 %r24, %r17, {csz}, %r15;\nst.shared.{t} [%r24], %f1;\nst.shared.{t} [%r24+{sz}], %f2;\nadd.u32 %r17, %r17, {LOCAL_BLOCK};\nbra $L_ld;\n$L_ldd:\nbar.sync 0;\n");
    // stages: radix r (6 bits each of the packed list), p = product so far
    s += "mov.u64 %rd21, %rd13;\nmov.u32 %r25, 1;\nmov.u32 %r26, %r15;\nmov.u32 %r27, %r16;\n$L_st:\ncvt.u32.u64 %r28, %rd21;\nand.b32 %r28, %r28, 63;\nsetp.eq.u32 %p3, %r28, 0;\n@%p3 bra $L_std;\nshr.u64 %rd21, %rd21, 6;\n";
    s += "mul.lo.u32 %r29, %r25, %r28;\ndiv.u32 %r30, %r4, %r28;\ndiv.u32 %r31, %r4, %r29;\nmul.lo.u32 %r36, %r30, %r12;\nmov.u32 %r17, %r2;\n";
    s += "$L_so:\nsetp.ge.u32 %p1, %r17, %r13;\n@%p1 bra $L_sod;\nand.b32 %r18, %r17, %r43;\nshr.u32 %r19, %r17, %r42;\n";
    s += "rem.u32 %r32, %r19, %r29;\ndiv.u32 %r33, %r19, %r29;\nrem.u32 %r34, %r32, %r25;\nmad.lo.u32 %r34, %r33, %r25, %r34;\nmad.lo.u32 %r35, %r34, %r12, %r18;\n";
    s += &format!("mad.lo.u32 %r39, %r35, {csz}, %r26;\nld.shared.{t} %f3, [%r39];\nld.shared.{t} %f4, [%r39+{sz}];\nadd.u32 %r35, %r35, %r36;\nmov.u32 %r37, %r32;\nsetp.ge.u32 %p4, %r37, %r29;\n@%p4 sub.u32 %r37, %r37, %r29;\nmov.u32 %r38, 1;\n$L_q:\n");
    let _ = zero;
    s += &format!("mad.lo.u32 %r39, %r35, {csz}, %r26;\nld.shared.{t} %f5, [%r39];\nld.shared.{t} %f6, [%r39+{sz}];\n");
    s += &format!("mul.lo.u32 %r40, %r37, %r31;\nmul.wide.u32 %rd22, %r40, {csz};\nadd.u64 %rd22, %rd18, %rd22;\nld.global.{t} %f7, [%rd22];\nld.global.{t} %f8, [%rd22+{sz}];\n@%p9 neg.{t} %f8, %f8;\n");
    s += &format!("fma.rn.{t} %f3, %f5, %f7, %f3;\nneg.{t} %f9, %f6;\nfma.rn.{t} %f3, %f9, %f8, %f3;\nfma.rn.{t} %f4, %f5, %f8, %f4;\nfma.rn.{t} %f4, %f6, %f7, %f4;\n");
    s += "add.u32 %r35, %r35, %r36;\nadd.u32 %r37, %r37, %r32;\nsetp.ge.u32 %p4, %r37, %r29;\n@%p4 sub.u32 %r37, %r37, %r29;\nadd.u32 %r38, %r38, 1;\nsetp.lt.u32 %p5, %r38, %r28;\n@%p5 bra $L_q;\n";
    s += &format!("mad.lo.u32 %r39, %r17, {csz}, %r27;\nst.shared.{t} [%r39], %f3;\nst.shared.{t} [%r39+{sz}], %f4;\nadd.u32 %r17, %r17, {LOCAL_BLOCK};\nbra $L_so;\n");
    s += "$L_sod:\nbar.sync 0;\nmov.u32 %r41, %r26;\nmov.u32 %r26, %r27;\nmov.u32 %r27, %r41;\nmov.u32 %r25, %r29;\nbra $L_st;\n$L_std:\n";
    // store (optional four-step twiddle W_n^(b o)), transposing through the strides
    s += "mov.u32 %r17, %r2;\n$L_wr:\nsetp.ge.u32 %p1, %r17, %r13;\n@%p1 bra $L_done;\nand.b32 %r18, %r17, %r43;\nshr.u32 %r19, %r17, %r42;\nadd.u32 %r20, %r14, %r18;\nsetp.ge.u32 %p2, %r20, %r3;\n@%p2 bra $L_wn;\n";
    s += &format!("mad.lo.u32 %r39, %r17, {csz}, %r26;\nld.shared.{t} %f1, [%r39];\nld.shared.{t} %f2, [%r39+{sz}];\ndiv.u32 %r21, %r20, %r5;\nrem.u32 %r22, %r20, %r5;\n");
    s += &format!("@!%p10 bra $L_nt;\nmul.lo.u32 %r40, %r22, %r19;\nmul.wide.u32 %rd22, %r40, {csz};\nadd.u64 %rd22, %rd19, %rd22;\nld.global.{t} %f7, [%rd22];\nld.global.{t} %f8, [%rd22+{sz}];\n@%p9 neg.{t} %f8, %f8;\n");
    s += &format!("mul.rn.{t} %f3, %f1, %f7;\nneg.{t} %f9, %f2;\nfma.rn.{t} %f3, %f9, %f8, %f3;\nmul.rn.{t} %f4, %f1, %f8;\nfma.rn.{t} %f4, %f2, %f7, %f4;\nmov.{t} %f1, %f3;\nmov.{t} %f2, %f4;\n$L_nt:\n");
    s += "mul.lo.u32 %r23, %r21, %r9;\nmad.lo.u32 %r23, %r22, %r10, %r23;\nmad.lo.u32 %r23, %r19, %r11, %r23;\n";
    s += &format!("mul.wide.u32 %rd20, %r23, {csz};\nadd.u64 %rd20, %rd17, %rd20;\nst.global.{t} [%rd20], %f1;\nst.global.{t} [%rd20+{sz}], %f2;\n");
    s += &format!("$L_wn:\nadd.u32 %r17, %r17, {LOCAL_BLOCK};\nbra $L_wr;\n$L_done:\nret;\n}}\n");
    s
}

/// Bluestein's chirp products (see `plan::Chirp`).
fn chirp(double: bool) -> String {
    let (t, sz) = if double { ("f64", 8) } else { ("f32", 4) };
    let csz = 2 * sz;
    let name = chirp_name(double);
    let zero = if double { "0d0000000000000000" } else { "0f00000000" };
    let mut s = params(&name, &["u64"; 7]);
    s += "{\n";
    s += &decls(t);
    for i in 0..7 {
        s += &format!("ld.param.u64 %rd{i}, [{name}_p{i}];\n");
    }
    s += "cvta.to.global.u64 %rd10, %rd0;\ncvta.to.global.u64 %rd11, %rd1;\ncvta.to.global.u64 %rd12, %rd2;\n";
    s += "cvt.u32.u64 %r1, %rd3;\ncvt.u32.u64 %r2, %rd4;\ncvt.u32.u64 %r3, %rd5;\ncvt.u32.u64 %r4, %rd6;\n";
    s += GID;
    s += "setp.ge.u32 %p1, %r9, %r1;\n@%p1 bra $L_done;\nrem.u32 %r10, %r9, %r4;\ndiv.u32 %r11, %r9, %r4;\nrem.u32 %r12, %r11, %r2;\ndiv.u32 %r13, %r11, %r2;\n";
    s += &format!("mov.{t} %f3, {zero};\nmov.{t} %f4, {zero};\nsetp.ge.u32 %p2, %r12, %r3;\n@%p2 bra $L_st;\n");
    s += "mad.lo.u32 %r14, %r13, %r3, %r12;\nmad.lo.u32 %r14, %r14, %r4, %r10;\n";
    s += &format!("mul.wide.u32 %rd13, %r14, {csz};\nadd.u64 %rd13, %rd10, %rd13;\nld.global.{t} %f1, [%rd13];\nld.global.{t} %f2, [%rd13+{sz}];\n");
    s += &format!("mul.wide.u32 %rd14, %r12, {csz};\nadd.u64 %rd14, %rd12, %rd14;\nld.global.{t} %f5, [%rd14];\nld.global.{t} %f6, [%rd14+{sz}];\n");
    s += &format!("mul.rn.{t} %f3, %f1, %f5;\nneg.{t} %f7, %f2;\nfma.rn.{t} %f3, %f7, %f6, %f3;\nmul.rn.{t} %f4, %f1, %f6;\nfma.rn.{t} %f4, %f2, %f5, %f4;\n");
    s += &format!("$L_st:\nmul.wide.u32 %rd15, %r9, {csz};\nadd.u64 %rd15, %rd11, %rd15;\nst.global.{t} [%rd15], %f3;\nst.global.{t} [%rd15+{sz}], %f4;\n$L_done:\nret;\n}}\n");
    s
}

/// `plan::local_cap` (this file is dependency-free).
fn crate_local_cap(double: bool) -> u32 {
    if double { 1024 } else { 2048 }
}

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

/// The whole module for one precision (all five entries).
pub fn module(double: bool) -> String {
    format!(
        ".version 7.0\n.target sm_60\n.address_size 64\n\n{}\n{}\n{}\n{}\n{}",
        gather(double),
        stage(double),
        scatter(double),
        local(double),
        chirp(double)
    )
}

/// `(entry name, single-entry module source, parameter count)` for every kernel.
pub fn all_entries() -> Vec<(String, String, usize)> {
    let mut v = Vec::new();
    for double in [false, true] {
        let head = ".version 7.0\n.target sm_60\n.address_size 64\n\n";
        v.push((gather_name(double), format!("{head}{}", gather(double)), PARAMS[0]));
        v.push((stage_name(double), format!("{head}{}", stage(double)), PARAMS[1]));
        v.push((scatter_name(double), format!("{head}{}", scatter(double)), PARAMS[2]));
        v.push((local_name(double), format!("{head}{}", local(double)), PARAMS[3]));
        v.push((chirp_name(double), format!("{head}{}", chirp(double)), PARAMS[4]));
    }
    v
}
