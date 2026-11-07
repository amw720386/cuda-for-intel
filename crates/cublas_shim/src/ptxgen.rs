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

/// Fused epilogue applied to `alpha*A*B + beta*C` before the store
/// (cublasLt `EPILOGUE_*`). Bias is a length-`m` vector indexed by the row `i`
/// of C and added after the `beta*C` term; activation follows the bias.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Epilogue {
    None,
    Relu,
    Gelu,
    Bias,
    ReluBias,
    GeluBias,
}

impl Epilogue {
    pub fn has_bias(self) -> bool {
        matches!(self, Self::Bias | Self::ReluBias | Self::GeluBias)
    }
    fn tag(self) -> &'static str {
        match self {
            Self::None => "",
            Self::Relu => "relu",
            Self::Gelu => "gelu",
            Self::Bias => "bias",
            Self::ReluBias => "relubias",
            Self::GeluBias => "gelubias",
        }
    }
    fn act(self) -> u8 {
        match self {
            Self::Relu | Self::ReluBias => 1,
            Self::Gelu | Self::GeluBias => 2,
            _ => 0,
        }
    }
}

/// Kernel options beyond the A/B/C types. `Opts::default()` reproduces the
/// plain cuBLAS GEMM kernel (identical PTX and entry name as [`generate`]).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Opts {
    pub epilogue: Epilogue,
    /// Element type of the bias vector (only used when the epilogue has bias).
    pub bias: Elem,
    /// Emulate `CUBLAS_COMPUTE_16F`: round the accumulator (and the alpha/beta
    /// terms) to f16 after every operation. Only meaningful for f16 A/B/C.
    pub acc16: bool,
}

impl Default for Opts {
    fn default() -> Self {
        Self { epilogue: Epilogue::None, bias: Elem::F32, acc16: false }
    }
}

impl Opts {
    /// Number of kernel parameters: 16, plus the bias pointer when present.
    pub fn param_count(&self) -> usize {
        16 + self.epilogue.has_bias() as usize
    }
}

pub fn entry_name_ex(a: Elem, b: Elem, c: Elem, pointer_arrays: bool, o: &Opts) -> String {
    let mut n = entry_name(a, b, c, pointer_arrays);
    if o.epilogue != Epilogue::None {
        n.push('_');
        n.push_str(o.epilogue.tag());
        if o.epilogue.has_bias() {
            n.push_str(o.bias.tag());
        }
    }
    if o.acc16 {
        n.push_str("_h16");
    }
    n
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

pub const GELU_CUBIC: f32 = 0.044715;
pub const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;

/// Round f32 register `r` to the nearest f16 value (kept in f32). Scratch `%rs3`.
fn round_f16(out: &mut String, r: &str) {
    let _ = writeln!(out, "cvt.rn.f16.f32 %rs3, {r};");
    let _ = writeln!(out, "cvt.f32.f16 {r}, %rs3;");
}

/// Generate the PTX module for one kernel variant.
///
/// Kernel parameters (all by value, in order):
/// 0 A, 1 B, 2 C (u64 device pointers, or pointer tables in `p` variants),
/// 3 alpha, 4 beta (f32), 5 m, 6 n, 7 k, 8 a_si, 9 a_sl, 10 b_sl, 11 b_sj,
/// 12 ldc (u32), 13 stride_a, 14 stride_b, 15 stride_c (u64, elements; ignored by `p`).
/// Launch: block (16,16,1), grid (ceil(m/16), ceil(n/16), batch).
pub fn generate(a: Elem, b: Elem, c: Elem, pointer_arrays: bool) -> String {
    generate_ex(a, b, c, pointer_arrays, &Opts::default())
}

/// Like [`generate`] with a fused epilogue and/or f16 accumulation. With a
/// bias epilogue there is one extra trailing parameter 16: `bias` (u64 device
/// pointer, element type `o.bias`, length m).
pub fn generate_ex(a: Elem, b: Elem, c: Elem, pointer_arrays: bool, o: &Opts) -> String {
    let name = entry_name_ex(a, b, c, pointer_arrays, o);
    let nparams = o.param_count();
    let mut s = String::new();
    let _ = writeln!(s, ".version 8.8\n.target sm_80\n.address_size 64\n");
    let _ = writeln!(s, ".visible .entry {name}(");
    for (i, ty) in [
        "u64", "u64", "u64", "f32", "f32", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "u32",
        "u64", "u64", "u64", "u64",
    ]
    .iter()
    .take(nparams)
    .enumerate()
    {
        let comma = if i == nparams - 1 { "" } else { "," };
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
    if o.epilogue.has_bias() {
        let _ = writeln!(s, "ld.param.u64 %rd19, {};", p(16));
        let _ = writeln!(s, "cvta.to.global.u64 %rd4, %rd19;");
    }
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
        if o.acc16 {
            round_f16(&mut s, "%f3");
        }
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
    if o.acc16 {
        round_f16(&mut s, "%f3");
    }
    let _ = writeln!(s, "mul.wide.u32 %rd40, %r15, %r8;");
    let _ = writeln!(s, "cvt.u64.u32 %rd41, %r14;");
    let _ = writeln!(s, "add.s64 %rd42, %rd40, %rd41;");
    let _ = writeln!(s, "shl.b64 %rd42, %rd42, {};", c.shift());
    let _ = writeln!(s, "add.s64 %rd43, %rd22, %rd42;");
    let _ = writeln!(s, "setp.eq.ftz.f32 %p7, %f2, 0f00000000;");
    let _ = writeln!(s, "@%p7 bra $L_store;");
    emit_load(&mut s, c, "%rd43", "%f14");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f2, %f14, %f3;");
    if o.acc16 {
        round_f16(&mut s, "%f3");
    }
    let _ = writeln!(s, "$L_store:");
    if o.epilogue.has_bias() {
        // %rd41 = i (zero-extended). bias[i] converted to f32 in %f15.
        let _ = writeln!(s, "shl.b64 %rd44, %rd41, {};", o.bias.shift());
        let _ = writeln!(s, "add.s64 %rd45, %rd4, %rd44;");
        emit_load(&mut s, o.bias, "%rd45", "%f15");
        let _ = writeln!(s, "add.ftz.f32 %f3, %f3, %f15;");
    }
    match o.epilogue.act() {
        1 => {
            let _ = writeln!(s, "max.ftz.f32 %f3, %f3, %f0;");
        }
        2 => {
            // tanh-approximated GELU: 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))).
            let lit = |v: f32| format!("0f{:08X}", v.to_bits());
            let _ = writeln!(s, "mul.ftz.f32 %f16, %f3, %f3;");
            let _ = writeln!(s, "mul.ftz.f32 %f17, %f16, %f3;");
            let _ = writeln!(s, "mov.f32 %f18, {};", lit(GELU_CUBIC));
            let _ = writeln!(s, "fma.rn.ftz.f32 %f17, %f17, %f18, %f3;");
            let _ = writeln!(s, "mov.f32 %f18, {};", lit(GELU_SQRT_2_OVER_PI));
            let _ = writeln!(s, "mul.ftz.f32 %f17, %f17, %f18;");
            let _ = writeln!(s, "tanh.approx.f32 %f19, %f17;");
            let _ = writeln!(s, "mov.f32 %f18, {};", lit(1.0));
            let _ = writeln!(s, "add.ftz.f32 %f19, %f19, %f18;");
            let _ = writeln!(s, "mov.f32 %f18, {};", lit(0.5));
            let _ = writeln!(s, "mul.ftz.f32 %f20, %f3, %f18;");
            let _ = writeln!(s, "mul.ftz.f32 %f3, %f20, %f19;");
        }
        _ => {}
    }
    emit_store(&mut s, c, "%rd43", "%f3");
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}


pub const TRSM_ENTRY: &str = "cfi_strsm_batched";

/// Batched triangular solve, in place, one thread per right-hand-side vector.
///
/// Solves `T x = alpha * b` for every vector, where `T(r,c) = A[r*a_rs + c*a_cs]`
/// is lower (`lower != 0`) or upper triangular of order `len`. Vector `v` of the
/// batch's B matrix lives at `B + v*b_vs + e*b_es` (elements, e < len). The host
/// maps cuBLAS side/uplo/trans onto this (see `plan::trsm_plan`).
///
/// Parameters: 0 A table (u64), 1 B table (u64), 2 alpha (f32), 3 len, 4 nvec,
/// 5 a_rs, 6 a_cs, 7 b_vs, 8 b_es, 9 lower, 10 unit (u32).
/// Launch: block (64,1,1), grid (ceil(nvec/64), batch, 1).
pub fn generate_trsm() -> String {
    let name = TRSM_ENTRY;
    let mut s = String::new();
    let _ = writeln!(s, ".version 8.8\n.target sm_80\n.address_size 64\n");
    let _ = writeln!(s, ".visible .entry {name}(");
    let tys = ["u64", "u64", "f32", "u32", "u32", "u32", "u32", "u32", "u32", "u32", "u32"];
    for (i, ty) in tys.iter().enumerate() {
        let comma = if i == tys.len() - 1 { "" } else { "," };
        let _ = writeln!(s, ".param .{ty} {name}_param_{i}{comma}");
    }
    let _ = writeln!(s, ")\n{{");
    let _ = writeln!(s, ".reg .pred %p<8>;\n.reg .f32 %f<8>;\n.reg .b32 %r<40>;\n.reg .b64 %rd<24>;");
    let p = |i: usize| format!("[{name}_param_{i}]");
    let _ = writeln!(s, "ld.param.u64 %rd1, {};", p(0));
    let _ = writeln!(s, "ld.param.u64 %rd2, {};", p(1));
    let _ = writeln!(s, "ld.param.f32 %f1, {};", p(2));
    // r1 len, r2 nvec, r3 a_rs, r4 a_cs, r5 b_vs, r6 b_es, r7 lower, r8 unit
    for (reg, idx) in [(1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10)] {
        let _ = writeln!(s, "ld.param.u32 %r{reg}, {};", p(idx));
    }
    let _ = writeln!(s, "mov.u32 %r10, %ctaid.x;");
    let _ = writeln!(s, "mov.u32 %r11, %tid.x;");
    let _ = writeln!(s, "mad.lo.s32 %r12, %r10, 64, %r11;");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r12, %r2;");
    let _ = writeln!(s, "@%p1 bra $T_done;");
    let _ = writeln!(s, "mov.u32 %r13, %ctaid.y;");
    let _ = writeln!(s, "cvt.u64.u32 %rd3, %r13;");
    let _ = writeln!(s, "shl.b64 %rd3, %rd3, 3;");
    let _ = writeln!(s, "cvta.to.global.u64 %rd4, %rd1;");
    let _ = writeln!(s, "cvta.to.global.u64 %rd5, %rd2;");
    let _ = writeln!(s, "add.s64 %rd6, %rd4, %rd3;");
    let _ = writeln!(s, "add.s64 %rd7, %rd5, %rd3;");
    let _ = writeln!(s, "ld.global.u64 %rd20, [%rd6];");
    let _ = writeln!(s, "ld.global.u64 %rd21, [%rd7];");
    let _ = writeln!(s, "cvta.to.global.u64 %rd8, %rd20;");
    let _ = writeln!(s, "cvta.to.global.u64 %rd9, %rd21;");
    let _ = writeln!(s, "mul.wide.u32 %rd10, %r12, %r5;");
    let _ = writeln!(s, "shl.b64 %rd10, %rd10, 2;");
    let _ = writeln!(s, "add.s64 %rd11, %rd9, %rd10;");
    let _ = writeln!(s, "setp.ne.u32 %p2, %r7, 0;");
    let _ = writeln!(s, "setp.ne.u32 %p3, %r8, 0;");
    let _ = writeln!(s, "sub.s32 %r14, %r1, 1;");
    let _ = writeln!(s, "mov.u32 %r15, 0;");
    let _ = writeln!(s, "setp.ge.u32 %p4, %r15, %r1;");
    let _ = writeln!(s, "@%p4 bra $T_done;");
    let _ = writeln!(s, "$T_row:");
    let _ = writeln!(s, "sub.s32 %r16, %r14, %r15;");
    let _ = writeln!(s, "selp.u32 %r17, %r15, %r16, %p2;");
    let _ = writeln!(s, "mul.wide.u32 %rd12, %r17, %r6;");
    let _ = writeln!(s, "shl.b64 %rd12, %rd12, 2;");
    let _ = writeln!(s, "add.s64 %rd13, %rd11, %rd12;");
    let _ = writeln!(s, "ld.global.f32 %f2, [%rd13];");
    let _ = writeln!(s, "mul.ftz.f32 %f2, %f2, %f1;");
    let _ = writeln!(s, "mul.lo.u32 %r18, %r17, %r3;");
    let _ = writeln!(s, "mov.u32 %r19, 0;");
    let _ = writeln!(s, "setp.ge.u32 %p5, %r19, %r15;");
    let _ = writeln!(s, "@%p5 bra $T_diag;");
    let _ = writeln!(s, "$T_col:");
    let _ = writeln!(s, "sub.s32 %r20, %r14, %r19;");
    let _ = writeln!(s, "selp.u32 %r21, %r19, %r20, %p2;");
    let _ = writeln!(s, "mul.lo.u32 %r22, %r21, %r4;");
    let _ = writeln!(s, "add.s32 %r23, %r18, %r22;");
    let _ = writeln!(s, "cvt.u64.u32 %rd14, %r23;");
    let _ = writeln!(s, "shl.b64 %rd14, %rd14, 2;");
    let _ = writeln!(s, "add.s64 %rd15, %rd8, %rd14;");
    let _ = writeln!(s, "ld.global.f32 %f3, [%rd15];");
    let _ = writeln!(s, "mul.wide.u32 %rd16, %r21, %r6;");
    let _ = writeln!(s, "shl.b64 %rd16, %rd16, 2;");
    let _ = writeln!(s, "add.s64 %rd17, %rd11, %rd16;");
    let _ = writeln!(s, "ld.global.f32 %f4, [%rd17];");
    let _ = writeln!(s, "neg.ftz.f32 %f3, %f3;");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f2, %f3, %f4, %f2;");
    let _ = writeln!(s, "add.s32 %r19, %r19, 1;");
    let _ = writeln!(s, "setp.lt.u32 %p5, %r19, %r15;");
    let _ = writeln!(s, "@%p5 bra $T_col;");
    let _ = writeln!(s, "$T_diag:");
    let _ = writeln!(s, "@%p3 bra $T_store;");
    let _ = writeln!(s, "mul.lo.u32 %r24, %r17, %r4;");
    let _ = writeln!(s, "add.s32 %r24, %r18, %r24;");
    let _ = writeln!(s, "cvt.u64.u32 %rd18, %r24;");
    let _ = writeln!(s, "shl.b64 %rd18, %rd18, 2;");
    let _ = writeln!(s, "add.s64 %rd19, %rd8, %rd18;");
    let _ = writeln!(s, "ld.global.f32 %f5, [%rd19];");
    let _ = writeln!(s, "div.rn.f32 %f2, %f2, %f5;");
    let _ = writeln!(s, "$T_store:");
    let _ = writeln!(s, "st.global.f32 [%rd13], %f2;");
    let _ = writeln!(s, "add.s32 %r15, %r15, 1;");
    let _ = writeln!(s, "setp.lt.u32 %p4, %r15, %r1;");
    let _ = writeln!(s, "@%p4 bra $T_row;");
    let _ = writeln!(s, "$T_done:\nret;\n}}");
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
    fn default_opts_match_plain_generator() {
        for (a, b, c, p) in all_variants() {
            assert_eq!(generate(a, b, c, p), generate_ex(a, b, c, p, &Opts::default()));
        }
    }

    #[test]
    fn epilogue_variants_are_well_formed() {
        for epi in [
            Epilogue::Relu, Epilogue::Gelu, Epilogue::Bias, Epilogue::ReluBias, Epilogue::GeluBias,
        ] {
            for bias in [Elem::F32, Elem::F16, Elem::Bf16] {
                for acc16 in [false, true] {
                    let o = Opts { epilogue: epi, bias, acc16 };
                    let src = generate_ex(Elem::F16, Elem::F16, Elem::F16, false, &o);
                    let name = entry_name_ex(Elem::F16, Elem::F16, Elem::F16, false, &o);
                    assert!(src.contains(&format!(".entry {name}(")));
                    assert_eq!(src.matches(".param ").count(), o.param_count());
                    assert_eq!(src.contains("ld.param.u64 %rd19"), epi.has_bias());
                    assert_eq!(src.contains("max.ftz.f32"), matches!(epi, Epilogue::Relu | Epilogue::ReluBias));
                    assert_eq!(src.contains("tanh.approx.f32"), matches!(epi, Epilogue::Gelu | Epilogue::GeluBias));
                    assert_eq!(src.contains("cvt.f32.f16 %f3, %rs3"), acc16);
                    let lab = src.find("$L_store:").unwrap();
                    let st = src.rfind("st.global").unwrap();
                    if epi.has_bias() {
                        let b = src.find("add.ftz.f32 %f3, %f3, %f15").unwrap();
                        assert!(lab < b && b < st);
                    }
                    assert_eq!(src.matches("bar.sync 0;").count(), 2);
                }
            }
        }
        let a = entry_name_ex(Elem::F32, Elem::F32, Elem::F32, false, &Opts { epilogue: Epilogue::Bias, bias: Elem::F32, acc16: false });
        let b = entry_name_ex(Elem::F32, Elem::F32, Elem::F32, false, &Opts { epilogue: Epilogue::Bias, bias: Elem::F16, acc16: false });
        assert_ne!(a, b);
    }

    #[test]
    fn gelu_constants_are_f32_hex() {
        let s = generate_ex(Elem::F32, Elem::F32, Elem::F32, false, &Opts { epilogue: Epilogue::Gelu, ..Opts::default() });
        assert!(s.contains(&format!("0f{:08X}", GELU_CUBIC.to_bits())));
        assert!(s.contains("0f3D372713"));
        assert!(s.contains("0f3F4C422A"));
    }

    #[test]
    fn trsm_kernel_shape() {
        let s = generate_trsm();
        assert_eq!(s.matches(".param ").count(), 11);
        assert!(!s.contains("bar.sync"));
        assert!(s.contains("div.rn.f32"));
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
