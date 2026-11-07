//! Generator for the shim's tiled shared-memory GEMM kernels (PTX text).
//!
//! One kernel family covers every cuBLAS operand layout: the host reduces
//! `transa`/`transb`/`lda`/`ldb` to element strides (see `plan::Strides`), so
//! the kernel is op-agnostic. Variants differ only in A/B/C element type
//! (f32, f16, bf16) and in how batch base pointers are found:
//!
//! * `s` strided: base = param + z * batch_stride (elements, wrapping u64)
//! * `p` pointer arrays: params are device tables; base = table[z]
//!
//! Accumulation is always f32. `beta == 0` never reads C. Only instructions the
//! repo's PTX-to-SPIR-V translator accepts are emitted (checked by a unit test
//! in `src/ptx.rs` that translates every variant). No thread returns before a
//! barrier: out-of-range loads are clamped and zeroed instead of skipped.
//!
//! This file is also `#[path]`-included by the main crate's tests, so it must
//! stay free of dependencies on the rest of the shim.

use std::fmt::Write;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Elem {
    F32,
    F16,
    Bf16,
}

impl Elem {
    fn tag(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
    fn shift(self) -> u32 {
        match self {
            Self::F32 => 2,
            _ => 1,
        }
    }
}

pub fn entry_name(a: Elem, b: Elem, c: Elem, pointer_arrays: bool) -> String {
    format!(
        "cfi_gemm_{}{}{}{}",
        a.tag(),
        b.tag(),
        c.tag(),
        if pointer_arrays { "p" } else { "s" }
    )
}

/// Emit PTX that loads one element of type `e` from global address `%addr`
/// into f32 register `%dst`. Uses scratch `%rs1`, `%r60`.
fn emit_load(out: &mut String, e: Elem, addr: &str, dst: &str) {
    match e {
        Elem::F32 => {
            let _ = writeln!(out, "ld.global.f32 {dst}, [{addr}];");
        }
        Elem::F16 => {
            let _ = writeln!(out, "ld.global.u16 %rs1, [{addr}];");
            let _ = writeln!(out, "cvt.f32.f16 {dst}, %rs1;");
        }
        Elem::Bf16 => {
            let _ = writeln!(out, "ld.global.u16 %rs1, [{addr}];");
            let _ = writeln!(out, "cvt.u32.u16 %r60, %rs1;");
            let _ = writeln!(out, "shl.b32 %r60, %r60, 16;");
            let _ = writeln!(out, "mov.b32 {dst}, %r60;");
        }
    }
}

/// Emit PTX that rounds f32 register `src` to type `e` and stores it at `%addr`.
/// Uses scratch `%rs2`, `%r60..%r62`, `%p6`.
fn emit_store(out: &mut String, e: Elem, addr: &str, src: &str) {
    match e {
        Elem::F32 => {
            let _ = writeln!(out, "st.global.f32 [{addr}], {src};");
        }
        Elem::F16 => {
            let _ = writeln!(out, "cvt.rn.f16.f32 %rs2, {src};");
            let _ = writeln!(out, "st.global.u16 [{addr}], %rs2;");
        }
        Elem::Bf16 => {
            // Round to nearest even on the integer image; NaN -> 0x7fc0.
            let _ = writeln!(out, "mov.b32 %r60, {src};");
            let _ = writeln!(out, "setp.ne.f32 %p6, {src}, {src};");
            let _ = writeln!(out, "shr.u32 %r61, %r60, 16;");
            let _ = writeln!(out, "and.b32 %r61, %r61, 1;");
            let _ = writeln!(out, "add.s32 %r61, %r61, 32767;");
            let _ = writeln!(out, "add.s32 %r60, %r60, %r61;");
            let _ = writeln!(out, "shr.u32 %r60, %r60, 16;");
            let _ = writeln!(out, "selp.u32 %r60, 32704, %r60, %p6;");
            let _ = writeln!(out, "cvt.u16.u32 %rs2, %r60;");
            let _ = writeln!(out, "st.global.u16 [{addr}], %rs2;");
        }
    }
}

/// Generate the PTX module for one kernel variant.
///
/// Kernel parameters (all by value, in order):
/// 0 A, 1 B, 2 C (u64 device pointers, or pointer tables in `p` variants),
/// 3 alpha, 4 beta (f32), 5 m, 6 n, 7 k, 8 a_si, 9 a_sl, 10 b_sl, 11 b_sj,
/// 12 ldc (u32), 13 stride_a, 14 stride_b, 15 stride_c (u64, elements; ignored by `p`).
/// Launch: block (16,16,1), grid (ceil(m/16), ceil(n/16), batch).
pub fn generate(a: Elem, b: Elem, c: Elem, pointer_arrays: bool) -> String {
    let name = entry_name(a, b, c, pointer_arrays);
    let mut s = String::new();
    let _ = writeln!(s, ".version 8.8\n.target sm_80\n.address_size 64\n");
    let _ = writeln!(s, ".visible .entry {name}(");
    for (i, ty) in [
        "u64", "u64", "u64", "f32", "f32", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "u32",
        "u64", "u64", "u64",
    ]
    .iter()
    .enumerate()
    {
        let comma = if i == 15 { "" } else { "," };
        let _ = writeln!(s, ".param .{ty} {name}_param_{i}{comma}");
    }
    let _ = writeln!(s, ")\n{{");
    let _ = writeln!(s, ".reg .pred %p<8>;\n.reg .b16 %rs<4>;\n.reg .f32 %f<24>;");
    let _ = writeln!(s, ".reg .b32 %r<72>;\n.reg .b64 %rd<48>;");
    let _ = writeln!(s, ".shared .align 4 .b8 tileA[1024];\n.shared .align 4 .b8 tileB[1024];");

    // Parameters.
    let p = |i: usize| format!("[{name}_param_{i}]");
    let _ = writeln!(s, "ld.param.u64 %rd10, {};", p(0));
    let _ = writeln!(s, "ld.param.u64 %rd11, {};", p(1));
    let _ = writeln!(s, "ld.param.u64 %rd12, {};", p(2));
    let _ = writeln!(s, "ld.param.f32 %f1, {};", p(3));
    let _ = writeln!(s, "ld.param.f32 %f2, {};", p(4));
    for (reg, idx) in [(1, 5), (2, 6), (3, 7), (4, 8), (5, 9), (6, 10), (7, 11), (8, 12)] {
        let _ = writeln!(s, "ld.param.u32 %r{reg}, {};", p(idx));
    }
    let _ = writeln!(s, "ld.param.u64 %rd13, {};", p(13));
    let _ = writeln!(s, "ld.param.u64 %rd14, {};", p(14));
    let _ = writeln!(s, "ld.param.u64 %rd15, {};", p(15));
    let _ = writeln!(s, "cvta.to.global.u64 %rd1, %rd10;");
    let _ = writeln!(s, "cvta.to.global.u64 %rd2, %rd11;");
    let _ = writeln!(s, "cvta.to.global.u64 %rd3, %rd12;");
    let _ = writeln!(s, "mov.f32 %f0, 0f00000000;");

    // Batch base pointers -> %rd20 (A), %rd21 (B), %rd22 (C). Each defined once.
    let _ = writeln!(s, "mov.u32 %r9, %ctaid.z;");
    let _ = writeln!(s, "cvt.u64.u32 %rd16, %r9;");
    if pointer_arrays {
        let _ = writeln!(s, "shl.b64 %rd17, %rd16, 3;");
        for (src, dst) in [(1, 20), (2, 21), (3, 22)] {
            let _ = writeln!(s, "add.s64 %rd18, %rd{src}, %rd17;");
            let _ = writeln!(s, "ld.global.u64 %rd{dst}, [%rd18];");
        }
    } else {
        for (src, stride, dst, e, tmp) in [(1, 13, 20, a, 23), (2, 14, 21, b, 24), (3, 15, 22, c, 25)] {
            let _ = writeln!(s, "mul.lo.u64 %rd{tmp}, %rd16, %rd{stride};");
            let _ = writeln!(s, "shl.b64 %rd{tmp}, %rd{tmp}, {};", e.shift());
            let _ = writeln!(s, "add.s64 %rd{dst}, %rd{src}, %rd{tmp};");
        }
    }

    // Thread / tile coordinates.
    let _ = writeln!(s, "mov.u32 %r10, %tid.x;");
    let _ = writeln!(s, "mov.u32 %r11, %tid.y;");
    let _ = writeln!(s, "mov.u32 %r12, %ctaid.x;");
    let _ = writeln!(s, "mov.u32 %r13, %ctaid.y;");
    let _ = writeln!(s, "mad.lo.s32 %r14, %r12, 16, %r10;"); // i
    let _ = writeln!(s, "mad.lo.s32 %r15, %r13, 16, %r11;"); // j
    let _ = writeln!(s, "mad.lo.s32 %r16, %r11, 16, %r10;");
    let _ = writeln!(s, "shl.b32 %r16, %r16, 2;"); // (ty*16+tx)*4
    let _ = writeln!(s, "mov.u32 %r17, tileA;");
    let _ = writeln!(s, "add.s32 %r18, %r17, %r16;"); // store addr A
    let _ = writeln!(s, "mov.u32 %r19, tileB;");
    let _ = writeln!(s, "add.s32 %r20, %r19, %r16;"); // store addr B
    let _ = writeln!(s, "shl.b32 %r21, %r10, 2;");
    let _ = writeln!(s, "add.s32 %r21, %r17, %r21;"); // &tileA[0][tx]
    let _ = writeln!(s, "shl.b32 %r22, %r11, 6;");
    let _ = writeln!(s, "add.s32 %r22, %r19, %r22;"); // &tileB[ty][0]
    let _ = writeln!(s, "sub.s32 %r35, %r1, 1;"); // m-1
    let _ = writeln!(s, "sub.s32 %r36, %r2, 1;"); // n-1
    let _ = writeln!(s, "sub.s32 %r37, %r3, 1;"); // k-1
    let _ = writeln!(s, "mov.f32 %f3, 0f00000000;"); // acc
    let _ = writeln!(s, "mov.u32 %r30, 0;"); // kk
    let _ = writeln!(s, "setp.ge.u32 %p1, %r30, %r3;");
    let _ = writeln!(s, "@%p1 bra $L_epilogue;");

    let _ = writeln!(s, "$L_k:");
    // A tile: element (i = i0+tx, l = kk+ty), clamped then zeroed when out of range.
    let _ = writeln!(s, "add.s32 %r31, %r30, %r11;");
    let _ = writeln!(s, "setp.lt.u32 %p2, %r14, %r1;");
    let _ = writeln!(s, "setp.lt.u32 %p3, %r31, %r3;");
    let _ = writeln!(s, "selp.u32 %r33, %r14, %r35, %p2;");
    let _ = writeln!(s, "selp.u32 %r34, %r31, %r37, %p3;");
    let _ = writeln!(s, "mul.wide.u32 %rd30, %r33, %r4;");
    let _ = writeln!(s, "mul.wide.u32 %rd31, %r34, %r5;");
    let _ = writeln!(s, "add.s64 %rd32, %rd30, %rd31;");
    let _ = writeln!(s, "shl.b64 %rd32, %rd32, {};", a.shift());
    let _ = writeln!(s, "add.s64 %rd33, %rd20, %rd32;");
    emit_load(&mut s, a, "%rd33", "%f10");
    let _ = writeln!(s, "selp.f32 %f10, %f10, %f0, %p2;");
    let _ = writeln!(s, "selp.f32 %f10, %f10, %f0, %p3;");
    let _ = writeln!(s, "st.shared.f32 [%r18], %f10;");
    // B tile: element (l = kk+tx, j = j0+ty).
    let _ = writeln!(s, "add.s32 %r38, %r30, %r10;");
    let _ = writeln!(s, "setp.lt.u32 %p4, %r15, %r2;");
    let _ = writeln!(s, "setp.lt.u32 %p5, %r38, %r3;");
    let _ = writeln!(s, "selp.u32 %r39, %r15, %r36, %p4;");
    let _ = writeln!(s, "selp.u32 %r40, %r38, %r37, %p5;");
    let _ = writeln!(s, "mul.wide.u32 %rd34, %r40, %r6;");
    let _ = writeln!(s, "mul.wide.u32 %rd35, %r39, %r7;");
    let _ = writeln!(s, "add.s64 %rd36, %rd34, %rd35;");
    let _ = writeln!(s, "shl.b64 %rd36, %rd36, {};", b.shift());
    let _ = writeln!(s, "add.s64 %rd37, %rd21, %rd36;");
    emit_load(&mut s, b, "%rd37", "%f11");
    let _ = writeln!(s, "selp.f32 %f11, %f11, %f0, %p4;");
    let _ = writeln!(s, "selp.f32 %f11, %f11, %f0, %p5;");
    let _ = writeln!(s, "st.shared.f32 [%r20], %f11;");
    let _ = writeln!(s, "bar.sync 0;");
    for l in 0..16u32 {
        let _ = writeln!(s, "add.s32 %r41, %r21, {};", l * 64);
        let _ = writeln!(s, "add.s32 %r42, %r22, {};", l * 4);
        let _ = writeln!(s, "ld.shared.f32 %f12, [%r41];");
        let _ = writeln!(s, "ld.shared.f32 %f13, [%r42];");
        let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f12, %f13, %f3;");
    }
    let _ = writeln!(s, "bar.sync 0;");
    let _ = writeln!(s, "add.s32 %r30, %r30, 16;");
    let _ = writeln!(s, "setp.lt.u32 %p1, %r30, %r3;");
    let _ = writeln!(s, "@%p1 bra $L_k;");

    let _ = writeln!(s, "$L_epilogue:");
    let _ = writeln!(s, "setp.ge.u32 %p4, %r14, %r1;");
    let _ = writeln!(s, "@%p4 bra $L_out;");
    let _ = writeln!(s, "setp.ge.u32 %p5, %r15, %r2;");
    let _ = writeln!(s, "@%p5 bra $L_out;");
    let _ = writeln!(s, "mul.ftz.f32 %f3, %f3, %f1;");
    let _ = writeln!(s, "mul.wide.u32 %rd40, %r15, %r8;");
    let _ = writeln!(s, "cvt.u64.u32 %rd41, %r14;");
    let _ = writeln!(s, "add.s64 %rd42, %rd40, %rd41;");
    let _ = writeln!(s, "shl.b64 %rd42, %rd42, {};", c.shift());
    let _ = writeln!(s, "add.s64 %rd43, %rd22, %rd42;");
    let _ = writeln!(s, "setp.eq.ftz.f32 %p7, %f2, 0f00000000;");
    let _ = writeln!(s, "@%p7 bra $L_store;");
    emit_load(&mut s, c, "%rd43", "%f14");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f2, %f14, %f3;");
    let _ = writeln!(s, "$L_store:");
    emit_store(&mut s, c, "%rd43", "%f3");
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}

/// All 54 (a, b, c, pointer-array) combinations the shim can instantiate.
pub fn all_variants() -> Vec<(Elem, Elem, Elem, bool)> {
    let es = [Elem::F32, Elem::F16, Elem::Bf16];
    let mut v = Vec::new();
    for &a in &es {
        for &b in &es {
            for &c in &es {
                for p in [false, true] {
                    v.push((a, b, c, p));
                }
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_variant_is_well_formed() {
        for (a, b, c, p) in all_variants() {
            let src = generate(a, b, c, p);
            let name = entry_name(a, b, c, p);
            assert!(src.contains(&format!(".entry {name}(")));
            assert_eq!(src.matches(".param ").count(), 16, "{name}");
            assert_eq!(src.matches("bar.sync 0;").count(), 2, "{name}");
            // Barriers must be reachable by every thread: no early `ret`/bra-to-out
            // appears before the last barrier.
            let last_bar = src.rfind("bar.sync").unwrap();
            let first_out = src.find("bra $L_out").unwrap();
            assert!(first_out > last_bar, "{name}");
            assert_eq!(src.matches("\n$L_k:").count(), 1);
            assert_eq!(src.contains("ld.global.u64 %rd20"), p, "{name}");
        }
    }

    #[test]
    fn dtype_snippets() {
        let s = generate(Elem::Bf16, Elem::F16, Elem::F32, false);
        assert!(s.contains("shl.b32 %r60, %r60, 16;")); // bf16 widen
        assert!(s.contains("cvt.f32.f16")); // f16 widen
        assert!(s.contains("st.global.f32"));
        assert!(!s.contains("cvt.rn.f16.f32"));
        let s = generate(Elem::F32, Elem::F32, Elem::Bf16, false);
        assert!(s.contains("selp.u32 %r60, 32704"));
        let s = generate(Elem::F16, Elem::F16, Elem::F16, true);
        assert!(s.contains("cvt.rn.f16.f32 %rs2"));
    }

    #[test]
    fn beta_zero_skips_c_load() {
        let s = generate(Elem::F32, Elem::F32, Elem::F32, false);
        let skip = s.find("@%p7 bra $L_store;").unwrap();
        let load = s.rfind("ld.global.f32 %f14").unwrap();
        assert!(skip < load);
    }
}
