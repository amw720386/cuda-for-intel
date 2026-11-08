//! PTX generators for the cuDNN shim kernels (direct convolution, tensor add,
//! batch normalization). Thread-per-output formulations, f32 accumulation,
//! element types f32 / f16 / bf16 for tensors (batch-norm statistics and
//! scale/bias are always f32, as in cuDNN).
//!
//! Layouts are handled by passing element strides for every tensor, so NCHW,
//! NHWC and arbitrary packed strides use the same kernel. Only 2-D spatial
//! convolution is generated (1-D is expressed with H = 1 by callers).
//!
//! This file is also `#[path]`-included by the main crate's tests (translator
//! and simulator checks), so it must not depend on the rest of the shim.
//! No kernel returns before a barrier: barrier kernels clamp instead.

use std::fmt::Write;

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Elem {
    F32,
    F16,
    Bf16,
}

impl Elem {
    pub fn tag(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
    pub fn shift(self) -> u32 {
        match self {
            Self::F32 => 2,
            _ => 1,
        }
    }
    pub fn all() -> [Elem; 3] {
        [Self::F32, Self::F16, Self::Bf16]
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ConvKind {
    Fwd,
    BwdData,
    BwdFilter,
}

impl ConvKind {
    fn tag(self) -> &'static str {
        match self {
            Self::Fwd => "fwd",
            Self::BwdData => "bwd_data",
            Self::BwdFilter => "bwd_filter",
        }
    }
}

pub fn conv_entry(kind: ConvKind, e: Elem) -> String {
    format!("cfi_cudnn_conv_{}_{}", kind.tag(), e.tag())
}
pub fn add_tensor_entry(e: Elem) -> String {
    format!("cfi_cudnn_add_tensor_{}", e.tag())
}
pub fn relu_entry(e: Elem) -> String {
    format!("cfi_cudnn_relu_{}", e.tag())
}
pub fn bn_train_entry(e: Elem) -> String {
    format!("cfi_cudnn_bn_train_{}", e.tag())
}
pub fn bn_infer_entry(e: Elem) -> String {
    format!("cfi_cudnn_bn_infer_{}", e.tag())
}
pub fn bn_bwd_entry(e: Elem) -> String {
    format!("cfi_cudnn_bn_bwd_{}", e.tag())
}

/// Conv kernel parameter count: 3 pointers, alpha, beta, 30 u32 (see [`generate_conv`]).
pub const CONV_PARAMS: usize = 35;
pub const ADD_PARAMS: usize = 17;
pub const BN_TRAIN_PARAMS: usize = 20;
pub const BN_INFER_PARAMS: usize = 17;
pub const BN_BWD_PARAMS: usize = 21;
pub const RELU_PARAMS: usize = 10;
pub const BN_BLOCK: u32 = 256;

fn header(s: &mut String, name: &str, tys: &[&str]) {
    let _ = writeln!(s, ".version 8.8\n.target sm_80\n.address_size 64\n");
    let _ = writeln!(s, ".visible .entry {name}(");
    for (i, ty) in tys.iter().enumerate() {
        let comma = if i + 1 == tys.len() { "" } else { "," };
        let _ = writeln!(s, ".param .{ty} {name}_param_{i}{comma}");
    }
    let _ = writeln!(s, ")\n{{");
    let _ = writeln!(s, ".reg .pred %p<16>;\n.reg .b16 %rs<4>;\n.reg .f32 %f<48>;");
    let _ = writeln!(s, ".reg .b32 %r<120>;\n.reg .b64 %rd<64>;");
}

/// Load element of type `e` at global `addr` into f32 `dst`. Scratch `%rs1`, `%r110`.
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
            let _ = writeln!(out, "cvt.u32.u16 %r110, %rs1;");
            let _ = writeln!(out, "shl.b32 %r110, %r110, 16;");
            let _ = writeln!(out, "mov.b32 {dst}, %r110;");
        }
    }
}

/// Round f32 `src` to `e` and store at `addr`. Scratch `%rs2`, `%r110..%r112`, `%p15`.
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
            let _ = writeln!(out, "mov.b32 %r110, {src};");
            let _ = writeln!(out, "setp.ne.f32 %p15, {src}, {src};");
            let _ = writeln!(out, "shr.u32 %r111, %r110, 16;");
            let _ = writeln!(out, "and.b32 %r111, %r111, 1;");
            let _ = writeln!(out, "add.s32 %r111, %r111, 32767;");
            let _ = writeln!(out, "add.s32 %r110, %r110, %r111;");
            let _ = writeln!(out, "shr.u32 %r110, %r110, 16;");
            let _ = writeln!(out, "selp.u32 %r110, 32704, %r110, %p15;");
            let _ = writeln!(out, "cvt.u16.u32 %rs2, %r110;");
            let _ = writeln!(out, "st.global.u16 [{addr}], %rs2;");
        }
    }
}

/// `%rdD = base + (a*sa + b*sb + c*sc + d*sd) << shift`, all u32 inputs, u64 math.
/// Scratch `%rd60..%rd62`.
fn emit_addr(s: &mut String, dst: &str, base: &str, terms: &[(&str, &str)], shift: u32) {
    let _ = writeln!(s, "mov.u64 %rd60, 0;");
    for (a, b) in terms {
        let _ = writeln!(s, "mul.wide.u32 %rd61, {a}, {b};");
        let _ = writeln!(s, "add.s64 %rd60, %rd60, %rd61;");
    }
    let _ = writeln!(s, "shl.b64 %rd60, %rd60, {shift};");
    let _ = writeln!(s, "add.s64 {dst}, {base}, %rd60;");
}

// Conv parameter indices.
// 0 a, 1 b, 2 out (u64); 3 alpha, 4 beta (f32); then u32:
// 5 N, 6 C, 7 H, 8 W, 9 K, 10 R, 11 S, 12 P, 13 Q, 14 padh, 15 padw, 16 strh, 17 strw,
// 18 dilh, 19 dilw, 20 groups, 21 flip, 22..25 x strides (n,c,h,w), 26..29 w strides
// (k,c,r,s), 30..33 y strides (n,k,p,q), 34 total (number of output elements).
// Roles: Fwd a=x b=w out=y; BwdData a=dy b=w out=dx; BwdFilter a=x b=dy out=dw.
// Registers: param i (5..=34) is loaded into %r{i}. Pointers: %rd1 a, %rd2 b, %rd3 out.

/// Generate a 2-D convolution kernel. Launch: block (256,1,1), grid ceil(total/256).
pub fn generate_conv(kind: ConvKind, e: Elem) -> String {
    let name = conv_entry(kind, e);
    let mut s = String::new();
    let mut tys = vec!["u64", "u64", "u64", "f32", "f32"];
    tys.extend(std::iter::repeat_n("u32", 30));
    header(&mut s, &name, &tys);
    let p = |i: usize| format!("[{name}_param_{i}]");
    for (i, rd) in [(0, 1), (1, 2), (2, 3)] {
        let _ = writeln!(s, "ld.param.u64 %rd{rd}, {};", p(i));
        let _ = writeln!(s, "cvta.to.global.u64 %rd{rd}, %rd{rd};");
    }
    let _ = writeln!(s, "ld.param.f32 %f1, {};", p(3));
    let _ = writeln!(s, "ld.param.f32 %f2, {};", p(4));
    for i in 5..CONV_PARAMS {
        let _ = writeln!(s, "ld.param.u32 %r{i}, {};", p(i));
    }
    let _ = writeln!(s, "mov.f32 %f0, 0f00000000;");
    let _ = writeln!(s, "mov.u32 %r40, %ctaid.x;");
    let _ = writeln!(s, "mov.u32 %r41, %ntid.x;");
    let _ = writeln!(s, "mov.u32 %r42, %tid.x;");
    let _ = writeln!(s, "mad.lo.s32 %r43, %r40, %r41, %r42;"); // idx
    let _ = writeln!(s, "setp.ge.u32 %p1, %r43, %r34;");
    let _ = writeln!(s, "@%p1 bra $L_out;");
    // Cg = C/groups, Kg = K/groups
    let _ = writeln!(s, "div.u32 %r44, %r6, %r20;");
    let _ = writeln!(s, "div.u32 %r45, %r9, %r20;");
    let _ = writeln!(s, "mov.f32 %f3, 0f00000000;"); // acc
    match kind {
        ConvKind::Fwd => gen_fwd(&mut s, e),
        ConvKind::BwdData => gen_bwd_data(&mut s, e),
        ConvKind::BwdFilter => gen_bwd_filter(&mut s, e),
    }
    // Epilogue: %f3 = acc, %rd10 = output address.
    let _ = writeln!(s, "mul.ftz.f32 %f3, %f3, %f1;");
    let _ = writeln!(s, "setp.eq.ftz.f32 %p7, %f2, 0f00000000;");
    let _ = writeln!(s, "@%p7 bra $L_store;");
    emit_load(&mut s, e, "%rd10", "%f14");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f2, %f14, %f3;");
    let _ = writeln!(s, "$L_store:");
    emit_store(&mut s, e, "%rd10", "%f3");
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}

fn gen_fwd(s: &mut String, e: Elem) {
    // idx -> q, p, k, n
    let _ = writeln!(s, "rem.u32 %r50, %r43, %r13;"); // q
    let _ = writeln!(s, "div.u32 %r51, %r43, %r13;");
    let _ = writeln!(s, "rem.u32 %r52, %r51, %r12;"); // p
    let _ = writeln!(s, "div.u32 %r51, %r51, %r12;");
    let _ = writeln!(s, "rem.u32 %r53, %r51, %r9;"); // k
    let _ = writeln!(s, "div.u32 %r54, %r51, %r9;"); // n
    let _ = writeln!(s, "div.u32 %r55, %r53, %r45;"); // g
    let _ = writeln!(s, "mul.lo.u32 %r56, %r55, %r44;"); // c0
    let _ = writeln!(s, "mov.u32 %r60, 0;"); // c
    let _ = writeln!(s, "setp.ne.u32 %p2, %r21, 0;"); // flip
    let _ = writeln!(s, "$L_c:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r60, %r44;");
    let _ = writeln!(s, "@%p1 bra $L_cdone;");
    let _ = writeln!(s, "add.s32 %r61, %r56, %r60;"); // cabs
    emit_addr(s, "%rd13", "%rd1", &[("%r54", "%r22"), ("%r61", "%r23")], e.shift());
    emit_addr(s, "%rd14", "%rd2", &[("%r53", "%r26"), ("%r60", "%r27")], e.shift());
    let _ = writeln!(s, "mov.u32 %r62, 0;"); // r
    let _ = writeln!(s, "$L_r:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r62, %r10;");
    let _ = writeln!(s, "@%p1 bra $L_rdone;");
    let _ = writeln!(s, "mul.lo.s32 %r63, %r52, %r16;");
    let _ = writeln!(s, "sub.s32 %r63, %r63, %r14;");
    let _ = writeln!(s, "mad.lo.s32 %r63, %r62, %r18, %r63;"); // ih
    let _ = writeln!(s, "setp.ge.u32 %p3, %r63, %r7;"); // ih out of range
    let _ = writeln!(s, "sub.s32 %r64, %r10, 1;");
    let _ = writeln!(s, "sub.s32 %r64, %r64, %r62;");
    let _ = writeln!(s, "selp.u32 %r64, %r64, %r62, %p2;"); // rw
    let _ = writeln!(s, "mov.u32 %r65, 0;"); // s
    let _ = writeln!(s, "$L_s:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r65, %r11;");
    let _ = writeln!(s, "@%p1 bra $L_sdone;");
    let _ = writeln!(s, "@%p3 bra $L_sn;");
    let _ = writeln!(s, "mul.lo.s32 %r66, %r50, %r17;");
    let _ = writeln!(s, "sub.s32 %r66, %r66, %r15;");
    let _ = writeln!(s, "mad.lo.s32 %r66, %r65, %r19, %r66;"); // iw
    let _ = writeln!(s, "setp.ge.u32 %p4, %r66, %r8;");
    let _ = writeln!(s, "@%p4 bra $L_sn;");
    let _ = writeln!(s, "sub.s32 %r67, %r11, 1;");
    let _ = writeln!(s, "sub.s32 %r67, %r67, %r65;");
    let _ = writeln!(s, "selp.u32 %r67, %r67, %r65, %p2;"); // sw
    let _ = writeln!(s, "mul.lo.u32 %r71, %r63, %r24;\nmad.lo.u32 %r71, %r66, %r25, %r71;");
    let _ = writeln!(s, "mul.lo.u32 %r72, %r64, %r28;\nmad.lo.u32 %r72, %r67, %r29, %r72;");
    let _ = writeln!(s, "mul.wide.u32 %rd61, %r71, {};\nadd.s64 %rd11, %rd13, %rd61;", 1u32 << e.shift());
    let _ = writeln!(s, "mul.wide.u32 %rd61, %r72, {};\nadd.s64 %rd12, %rd14, %rd61;", 1u32 << e.shift());
    emit_load(s, e, "%rd11", "%f10");
    emit_load(s, e, "%rd12", "%f11");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f10, %f11, %f3;");
    let _ = writeln!(s, "$L_sn:");
    let _ = writeln!(s, "add.s32 %r65, %r65, 1;");
    let _ = writeln!(s, "bra $L_s;");
    let _ = writeln!(s, "$L_sdone:");
    let _ = writeln!(s, "add.s32 %r62, %r62, 1;");
    let _ = writeln!(s, "bra $L_r;");
    let _ = writeln!(s, "$L_rdone:");
    let _ = writeln!(s, "add.s32 %r60, %r60, 1;");
    let _ = writeln!(s, "bra $L_c;");
    let _ = writeln!(s, "$L_cdone:");
    emit_addr(s, "%rd10", "%rd3", &[("%r54", "%r30"), ("%r53", "%r31"), ("%r52", "%r32"), ("%r50", "%r33")], e.shift());
}

fn gen_bwd_data(s: &mut String, e: Elem) {
    // idx -> w, h, c, n over dx's logical NCHW extents (N,C,H,W)
    let _ = writeln!(s, "rem.u32 %r50, %r43, %r8;"); // iw
    let _ = writeln!(s, "div.u32 %r51, %r43, %r8;");
    let _ = writeln!(s, "rem.u32 %r52, %r51, %r7;"); // ih
    let _ = writeln!(s, "div.u32 %r51, %r51, %r7;");
    let _ = writeln!(s, "rem.u32 %r53, %r51, %r6;"); // c (absolute)
    let _ = writeln!(s, "div.u32 %r54, %r51, %r6;"); // n
    let _ = writeln!(s, "div.u32 %r55, %r53, %r44;"); // g
    let _ = writeln!(s, "mul.lo.u32 %r56, %r55, %r44;");
    let _ = writeln!(s, "sub.s32 %r57, %r53, %r56;"); // c local
    let _ = writeln!(s, "mul.lo.u32 %r58, %r55, %r45;"); // k0
    let _ = writeln!(s, "setp.ne.u32 %p2, %r21, 0;");
    let _ = writeln!(s, "mov.u32 %r60, 0;"); // kl
    let _ = writeln!(s, "$L_k:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r60, %r45;");
    let _ = writeln!(s, "@%p1 bra $L_kdone;");
    let _ = writeln!(s, "add.s32 %r61, %r58, %r60;"); // kabs
    let _ = writeln!(s, "mov.u32 %r62, 0;"); // r
    let _ = writeln!(s, "$L_r:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r62, %r10;");
    let _ = writeln!(s, "@%p1 bra $L_rdone;");
    // num = ih + padh - r*dilh; need num>=0, num%strh==0, p=num/strh < P
    let _ = writeln!(s, "add.s32 %r63, %r52, %r14;");
    let _ = writeln!(s, "mul.lo.s32 %r64, %r62, %r18;");
    let _ = writeln!(s, "sub.s32 %r63, %r63, %r64;");
    let _ = writeln!(s, "setp.lt.s32 %p3, %r63, 0;");
    let _ = writeln!(s, "rem.u32 %r64, %r63, %r16;");
    let _ = writeln!(s, "setp.ne.u32 %p5, %r64, 0;");
    let _ = writeln!(s, "div.u32 %r68, %r63, %r16;"); // p
    let _ = writeln!(s, "setp.ge.u32 %p6, %r68, %r12;");
    let _ = writeln!(s, "or.pred %p3, %p3, %p5;");
    let _ = writeln!(s, "or.pred %p3, %p3, %p6;");
    let _ = writeln!(s, "sub.s32 %r64, %r10, 1;");
    let _ = writeln!(s, "sub.s32 %r64, %r64, %r62;");
    let _ = writeln!(s, "selp.u32 %r64, %r64, %r62, %p2;"); // rw
    let _ = writeln!(s, "mov.u32 %r65, 0;"); // s
    let _ = writeln!(s, "$L_s:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r65, %r11;");
    let _ = writeln!(s, "@%p1 bra $L_sdone;");
    let _ = writeln!(s, "@%p3 bra $L_sn;");
    let _ = writeln!(s, "add.s32 %r66, %r50, %r15;");
    let _ = writeln!(s, "mul.lo.s32 %r69, %r65, %r19;");
    let _ = writeln!(s, "sub.s32 %r66, %r66, %r69;");
    let _ = writeln!(s, "setp.lt.s32 %p4, %r66, 0;");
    let _ = writeln!(s, "rem.u32 %r69, %r66, %r17;");
    let _ = writeln!(s, "setp.ne.u32 %p5, %r69, 0;");
    let _ = writeln!(s, "div.u32 %r70, %r66, %r17;"); // q
    let _ = writeln!(s, "setp.ge.u32 %p6, %r70, %r13;");
    let _ = writeln!(s, "or.pred %p4, %p4, %p5;");
    let _ = writeln!(s, "or.pred %p4, %p4, %p6;");
    let _ = writeln!(s, "@%p4 bra $L_sn;");
    let _ = writeln!(s, "sub.s32 %r67, %r11, 1;");
    let _ = writeln!(s, "sub.s32 %r67, %r67, %r65;");
    let _ = writeln!(s, "selp.u32 %r67, %r67, %r65, %p2;"); // sw
    // a = dy[n,kabs,p,q] (y strides 30..33), b = w[kabs, c local, rw, sw]
    emit_addr(s, "%rd11", "%rd1", &[("%r54", "%r30"), ("%r61", "%r31"), ("%r68", "%r32"), ("%r70", "%r33")], e.shift());
    emit_addr(s, "%rd12", "%rd2", &[("%r61", "%r26"), ("%r57", "%r27"), ("%r64", "%r28"), ("%r67", "%r29")], e.shift());
    emit_load(s, e, "%rd11", "%f10");
    emit_load(s, e, "%rd12", "%f11");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f10, %f11, %f3;");
    let _ = writeln!(s, "$L_sn:");
    let _ = writeln!(s, "add.s32 %r65, %r65, 1;");
    let _ = writeln!(s, "bra $L_s;");
    let _ = writeln!(s, "$L_sdone:");
    let _ = writeln!(s, "add.s32 %r62, %r62, 1;");
    let _ = writeln!(s, "bra $L_r;");
    let _ = writeln!(s, "$L_rdone:");
    let _ = writeln!(s, "add.s32 %r60, %r60, 1;");
    let _ = writeln!(s, "bra $L_k;");
    let _ = writeln!(s, "$L_kdone:");
    emit_addr(s, "%rd10", "%rd3", &[("%r54", "%r22"), ("%r53", "%r23"), ("%r52", "%r24"), ("%r50", "%r25")], e.shift());
}

fn gen_bwd_filter(s: &mut String, e: Elem) {
    // idx -> s, r, c(local), k over dw's logical (K, Cg, R, S)
    let _ = writeln!(s, "rem.u32 %r50, %r43, %r11;"); // s
    let _ = writeln!(s, "div.u32 %r51, %r43, %r11;");
    let _ = writeln!(s, "rem.u32 %r52, %r51, %r10;"); // r
    let _ = writeln!(s, "div.u32 %r51, %r51, %r10;");
    let _ = writeln!(s, "rem.u32 %r53, %r51, %r44;"); // c local
    let _ = writeln!(s, "div.u32 %r54, %r51, %r44;"); // k
    let _ = writeln!(s, "div.u32 %r55, %r54, %r45;"); // g
    let _ = writeln!(s, "mad.lo.s32 %r56, %r55, %r44, %r53;"); // cabs
    let _ = writeln!(s, "setp.ne.u32 %p2, %r21, 0;");
    // positions: rpos = flip ? R-1-r : r
    let _ = writeln!(s, "sub.s32 %r57, %r10, 1;");
    let _ = writeln!(s, "sub.s32 %r57, %r57, %r52;");
    let _ = writeln!(s, "selp.u32 %r57, %r57, %r52, %p2;");
    let _ = writeln!(s, "sub.s32 %r58, %r11, 1;");
    let _ = writeln!(s, "sub.s32 %r58, %r58, %r50;");
    let _ = writeln!(s, "selp.u32 %r58, %r58, %r50, %p2;");
    let _ = writeln!(s, "mov.u32 %r60, 0;"); // n
    let _ = writeln!(s, "$L_n:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r60, %r5;");
    let _ = writeln!(s, "@%p1 bra $L_ndone;");
    let _ = writeln!(s, "mov.u32 %r62, 0;"); // p
    let _ = writeln!(s, "$L_p:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r62, %r12;");
    let _ = writeln!(s, "@%p1 bra $L_pdone;");
    let _ = writeln!(s, "mul.lo.s32 %r63, %r62, %r16;");
    let _ = writeln!(s, "sub.s32 %r63, %r63, %r14;");
    let _ = writeln!(s, "mad.lo.s32 %r63, %r57, %r18, %r63;"); // ih
    let _ = writeln!(s, "setp.ge.u32 %p3, %r63, %r7;");
    let _ = writeln!(s, "mov.u32 %r65, 0;"); // q
    let _ = writeln!(s, "$L_q:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r65, %r13;");
    let _ = writeln!(s, "@%p1 bra $L_qdone;");
    let _ = writeln!(s, "@%p3 bra $L_qn;");
    let _ = writeln!(s, "mul.lo.s32 %r66, %r65, %r17;");
    let _ = writeln!(s, "sub.s32 %r66, %r66, %r15;");
    let _ = writeln!(s, "mad.lo.s32 %r66, %r58, %r19, %r66;"); // iw
    let _ = writeln!(s, "setp.ge.u32 %p4, %r66, %r8;");
    let _ = writeln!(s, "@%p4 bra $L_qn;");
    emit_addr(s, "%rd11", "%rd1", &[("%r60", "%r22"), ("%r56", "%r23"), ("%r63", "%r24"), ("%r66", "%r25")], e.shift());
    emit_addr(s, "%rd12", "%rd2", &[("%r60", "%r30"), ("%r54", "%r31"), ("%r62", "%r32"), ("%r65", "%r33")], e.shift());
    emit_load(s, e, "%rd11", "%f10");
    emit_load(s, e, "%rd12", "%f11");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f10, %f11, %f3;");
    let _ = writeln!(s, "$L_qn:");
    let _ = writeln!(s, "add.s32 %r65, %r65, 1;");
    let _ = writeln!(s, "bra $L_q;");
    let _ = writeln!(s, "$L_qdone:");
    let _ = writeln!(s, "add.s32 %r62, %r62, 1;");
    let _ = writeln!(s, "bra $L_p;");
    let _ = writeln!(s, "$L_pdone:");
    let _ = writeln!(s, "add.s32 %r60, %r60, 1;");
    let _ = writeln!(s, "bra $L_n;");
    let _ = writeln!(s, "$L_ndone:");
    emit_addr(s, "%rd10", "%rd3", &[("%r54", "%r26"), ("%r53", "%r27"), ("%r52", "%r28"), ("%r50", "%r29")], e.shift());
}

// --- Explicit-GEMM convolution helpers: im2col / col2im ------------------------------------
// The column buffer is addressed through four element strides (group, image, kc row, pq column)
// so one kernel serves every GEMM layout; `kc` is the flattened (c,r,s) index (`order` 0) or the
// (r,s,c) index (`order` 1, matches channels-last filters).

pub fn im2col_entry(e: Elem, kc_fast: bool) -> String {
    format!("cfi_cudnn_im2col_{}_{}", if kc_fast { "kc" } else { "pq" }, e.tag())
}
pub fn col2im_entry(e: Elem, c_fast: bool) -> String {
    format!("cfi_cudnn_col2im_{}_{}", if c_fast { "c" } else { "w" }, e.tag())
}
pub const IM2COL_PARAMS: usize = 27;
pub const COL2IM_PARAMS: usize = 30;

/// `col[g*cs_g + n*cs_n + kc*cs_kc + pq*cs_pq] = x[(n0+n), g*Cg + c, p*str - pad + r*dil, ...]` (0 outside).
/// Params: 0 x, 1 col (u64); u32: 2 total (G*nl*Kc*PQ), 3 n0, 4 nl, 5 Cg, 6 H, 7 W, 8 R, 9 S, 10 P, 11 Q,
/// 12 padh, 13 padw, 14 strh, 15 strw, 16 dilh, 17 dilw, 18 order, 19..=22 x strides n,c,h,w,
/// 23..=26 col strides g,n,kc,pq. Thread index: `kc_fast` makes kc the fastest dimension, else pq.
pub fn generate_im2col(e: Elem, kc_fast: bool) -> String {
    let name = im2col_entry(e, kc_fast);
    let mut s = String::new();
    let mut tys = vec!["u64", "u64"];
    tys.extend(std::iter::repeat_n("u32", IM2COL_PARAMS - 2));
    header(&mut s, &name, &tys);
    let p = |i: usize| format!("[{name}_param_{i}]");
    for (i, rd) in [(0, 1), (1, 2)] {
        let _ = writeln!(s, "ld.param.u64 %rd{rd}, {};\ncvta.to.global.u64 %rd{rd}, %rd{rd};", p(i));
    }
    for i in 2..IM2COL_PARAMS {
        let _ = writeln!(s, "ld.param.u32 %r{i}, {};", p(i));
    }
    let _ = writeln!(s, "mov.u32 %r40, %ctaid.x;\nmov.u32 %r41, %ntid.x;\nmov.u32 %r42, %tid.x;");
    let _ = writeln!(s, "mad.lo.s32 %r43, %r40, %r41, %r42;"); // idx
    let _ = writeln!(s, "setp.ge.u32 %p1, %r43, %r2;\n@%p1 bra $L_out;");
    let _ = writeln!(s, "mul.lo.u32 %r44, %r10, %r11;"); // PQ
    let _ = writeln!(s, "mul.lo.u32 %r45, %r5, %r8;\nmul.lo.u32 %r45, %r45, %r9;"); // Kc
    // decompose idx -> pq %r46, kc %r47, n %r48, g %r49
    if kc_fast {
        let _ = writeln!(s, "rem.u32 %r47, %r43, %r45;\ndiv.u32 %r50, %r43, %r45;");
        let _ = writeln!(s, "rem.u32 %r46, %r50, %r44;\ndiv.u32 %r50, %r50, %r44;");
    } else {
        let _ = writeln!(s, "rem.u32 %r46, %r43, %r44;\ndiv.u32 %r50, %r43, %r44;");
        let _ = writeln!(s, "rem.u32 %r47, %r50, %r45;\ndiv.u32 %r50, %r50, %r45;");
    }
    let _ = writeln!(s, "rem.u32 %r48, %r50, %r4;\ndiv.u32 %r49, %r50, %r4;");
    let _ = writeln!(s, "rem.u32 %r51, %r46, %r11;\ndiv.u32 %r52, %r46, %r11;"); // q, p
    // kc -> r %r53, s %r54, c %r55
    let _ = writeln!(s, "setp.ne.u32 %p2, %r18, 0;\n@%p2 bra $L_rsc;");
    let _ = writeln!(s, "rem.u32 %r54, %r47, %r9;\ndiv.u32 %r56, %r47, %r9;");
    let _ = writeln!(s, "rem.u32 %r53, %r56, %r8;\ndiv.u32 %r55, %r56, %r8;");
    let _ = writeln!(s, "bra $L_decoded;\n$L_rsc:");
    let _ = writeln!(s, "rem.u32 %r55, %r47, %r5;\ndiv.u32 %r56, %r47, %r5;");
    let _ = writeln!(s, "rem.u32 %r54, %r56, %r9;\ndiv.u32 %r53, %r56, %r9;");
    let _ = writeln!(s, "$L_decoded:");
    // ih %r57, iw %r58
    let _ = writeln!(s, "mul.lo.s32 %r57, %r52, %r14;\nsub.s32 %r57, %r57, %r12;\nmad.lo.s32 %r57, %r53, %r16, %r57;");
    let _ = writeln!(s, "mul.lo.s32 %r58, %r51, %r15;\nsub.s32 %r58, %r58, %r13;\nmad.lo.s32 %r58, %r54, %r17, %r58;");
    let _ = writeln!(s, "mov.f32 %f10, 0f00000000;");
    let _ = writeln!(s, "setp.lt.u32 %p3, %r57, %r6;\nsetp.lt.u32 %p4, %r58, %r7;\nand.pred %p3, %p3, %p4;");
    let _ = writeln!(s, "@!%p3 bra $L_store;");
    let _ = writeln!(s, "add.u32 %r59, %r48, %r3;\nmad.lo.u32 %r60, %r49, %r5, %r55;"); // image, absolute channel
    emit_addr(&mut s, "%rd10", "%rd1", &[("%r59", "%r19"), ("%r60", "%r20"), ("%r57", "%r21"), ("%r58", "%r22")], e.shift());
    emit_load(&mut s, e, "%rd10", "%f10");
    let _ = writeln!(s, "$L_store:");
    emit_addr(&mut s, "%rd11", "%rd2", &[("%r49", "%r23"), ("%r48", "%r24"), ("%r47", "%r25"), ("%r46", "%r26")], e.shift());
    emit_store(&mut s, e, "%rd11", "%f10");
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}

/// Gather form of col2im for the data gradient:
/// `dx[n0+n, c, h, w] = alpha * sum_{r,s} dcol[..] + beta * dx[..]`.
/// Params: 0 dcol, 1 dx (u64); 2 alpha, 3 beta (f32); u32: 4 total (nl*C*H*W), 5 n0, 6 nl, 7 Cg, 8 H, 9 W,
/// 10 R, 11 S, 12 P, 13 Q, 14 padh, 15 padw, 16 strh, 17 strw, 18 dilh, 19 dilw, 20 order, 21..=24 dx strides
/// n,c,h,w, 25..=28 col strides g,n,kc,pq, 29 C. `c_fast`: channel is the fastest thread dimension, else w.
pub fn generate_col2im(e: Elem, c_fast: bool) -> String {
    let name = col2im_entry(e, c_fast);
    let mut s = String::new();
    let mut tys = vec!["u64", "u64", "f32", "f32"];
    tys.extend(std::iter::repeat_n("u32", COL2IM_PARAMS - 4));
    header(&mut s, &name, &tys);
    let p = |i: usize| format!("[{name}_param_{i}]");
    for (i, rd) in [(0, 1), (1, 2)] {
        let _ = writeln!(s, "ld.param.u64 %rd{rd}, {};\ncvta.to.global.u64 %rd{rd}, %rd{rd};", p(i));
    }
    let _ = writeln!(s, "ld.param.f32 %f1, {};\nld.param.f32 %f2, {};", p(2), p(3));
    for i in 4..COL2IM_PARAMS {
        let _ = writeln!(s, "ld.param.u32 %r{i}, {};", p(i));
    }
    let _ = writeln!(s, "mov.u32 %r40, %ctaid.x;\nmov.u32 %r41, %ntid.x;\nmov.u32 %r42, %tid.x;");
    let _ = writeln!(s, "mad.lo.s32 %r43, %r40, %r41, %r42;");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r43, %r4;\n@%p1 bra $L_out;");
    // decompose idx -> c %r46, w %r47, h %r48, n %r49
    if c_fast {
        let _ = writeln!(s, "rem.u32 %r46, %r43, %r29;\ndiv.u32 %r50, %r43, %r29;");
        let _ = writeln!(s, "rem.u32 %r47, %r50, %r9;\ndiv.u32 %r50, %r50, %r9;");
        let _ = writeln!(s, "rem.u32 %r48, %r50, %r8;\ndiv.u32 %r49, %r50, %r8;");
    } else {
        let _ = writeln!(s, "rem.u32 %r47, %r43, %r9;\ndiv.u32 %r50, %r43, %r9;");
        let _ = writeln!(s, "rem.u32 %r48, %r50, %r8;\ndiv.u32 %r50, %r50, %r8;");
        let _ = writeln!(s, "rem.u32 %r46, %r50, %r29;\ndiv.u32 %r49, %r50, %r29;");
    }
    let _ = writeln!(s, "rem.u32 %r51, %r46, %r7;\ndiv.u32 %r52, %r46, %r7;"); // c local, g
    let _ = writeln!(s, "mul.lo.u32 %r53, %r52, %r25;"); // g * cs_g
    let _ = writeln!(s, "mad.lo.u32 %r53, %r49, %r26, %r53;"); // + n * cs_n
    let _ = writeln!(s, "mov.f32 %f3, 0f00000000;");
    let _ = writeln!(s, "setp.ne.u32 %p2, %r20, 0;");
    let _ = writeln!(s, "mov.u32 %r60, 0;"); // r
    let _ = writeln!(s, "$L_r:\nsetp.ge.u32 %p1, %r60, %r10;\n@%p1 bra $L_rdone;");
    let _ = writeln!(s, "add.s32 %r61, %r48, %r14;\nmul.lo.s32 %r62, %r60, %r18;\nsub.s32 %r61, %r61, %r62;"); // numh
    let _ = writeln!(s, "setp.lt.s32 %p3, %r61, 0;\nrem.u32 %r62, %r61, %r16;\nsetp.ne.u32 %p5, %r62, 0;");
    let _ = writeln!(s, "div.u32 %r63, %r61, %r16;\nsetp.ge.u32 %p6, %r63, %r12;"); // p
    let _ = writeln!(s, "or.pred %p3, %p3, %p5;\nor.pred %p3, %p3, %p6;");
    let _ = writeln!(s, "mov.u32 %r64, 0;"); // s
    let _ = writeln!(s, "$L_s:\nsetp.ge.u32 %p1, %r64, %r11;\n@%p1 bra $L_sdone;\n@%p3 bra $L_sn;");
    let _ = writeln!(s, "add.s32 %r65, %r47, %r15;\nmul.lo.s32 %r66, %r64, %r19;\nsub.s32 %r65, %r65, %r66;"); // numw
    let _ = writeln!(s, "setp.lt.s32 %p4, %r65, 0;\nrem.u32 %r66, %r65, %r17;\nsetp.ne.u32 %p5, %r66, 0;");
    let _ = writeln!(s, "div.u32 %r67, %r65, %r17;\nsetp.ge.u32 %p6, %r67, %r13;"); // q
    let _ = writeln!(s, "or.pred %p4, %p4, %p5;\nor.pred %p4, %p4, %p6;\n@%p4 bra $L_sn;");
    // kc
    let _ = writeln!(s, "@%p2 bra $L_kc_rsc;");
    let _ = writeln!(s, "mad.lo.u32 %r68, %r51, %r10, %r60;\nmad.lo.u32 %r68, %r68, %r11, %r64;\nbra $L_kc_done;");
    let _ = writeln!(s, "$L_kc_rsc:\nmad.lo.u32 %r68, %r60, %r11, %r64;\nmad.lo.u32 %r68, %r68, %r7, %r51;");
    let _ = writeln!(s, "$L_kc_done:");
    let _ = writeln!(s, "mad.lo.u32 %r69, %r63, %r13, %r67;"); // pq
    emit_addr(&mut s, "%rd11", "%rd1", &[("%r68", "%r27"), ("%r69", "%r28")], e.shift());
    // + (g*cs_g + n*cs_n) precomputed in %r53
    let _ = writeln!(s, "mul.wide.u32 %rd62, %r53, {};\nadd.s64 %rd11, %rd11, %rd62;", 1u32 << e.shift());
    emit_load(&mut s, e, "%rd11", "%f10");
    let _ = writeln!(s, "add.ftz.f32 %f3, %f3, %f10;");
    let _ = writeln!(s, "$L_sn:\nadd.s32 %r64, %r64, 1;\nbra $L_s;\n$L_sdone:");
    let _ = writeln!(s, "add.s32 %r60, %r60, 1;\nbra $L_r;\n$L_rdone:");
    let _ = writeln!(s, "add.u32 %r59, %r49, %r5;");
    emit_addr(&mut s, "%rd10", "%rd2", &[("%r59", "%r21"), ("%r46", "%r22"), ("%r48", "%r23"), ("%r47", "%r24")], e.shift());
    let _ = writeln!(s, "mul.ftz.f32 %f3, %f3, %f1;");
    let _ = writeln!(s, "setp.eq.ftz.f32 %p7, %f2, 0f00000000;\n@%p7 bra $L_store;");
    emit_load(&mut s, e, "%rd10", "%f14");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f2, %f14, %f3;");
    let _ = writeln!(s, "$L_store:");
    emit_store(&mut s, e, "%rd10", "%f3");
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}

/// Entry of the work-group-per-element weight-gradient kernel (few outputs, long reductions such as
/// depthwise filters). Same parameters as [`generate_conv`]; launch grid = number of dw elements, block 256.
pub fn conv_bwd_filter_red_entry(e: Elem) -> String {
    format!("cfi_cudnn_conv_bwd_filter_red_{}", e.tag())
}

pub fn generate_conv_bwd_filter_red(e: Elem) -> String {
    let name = conv_bwd_filter_red_entry(e);
    let mut s = String::new();
    let mut tys = vec!["u64", "u64", "u64", "f32", "f32"];
    tys.extend(std::iter::repeat_n("u32", 30));
    header(&mut s, &name, &tys);
    let _ = writeln!(s, ".shared .align 4 .b8 red[{}];", BN_BLOCK * 4);
    let p = |i: usize| format!("[{name}_param_{i}]");
    for (i, rd) in [(0, 1), (1, 2), (2, 3)] {
        let _ = writeln!(s, "ld.param.u64 %rd{rd}, {};\ncvta.to.global.u64 %rd{rd}, %rd{rd};", p(i));
    }
    let _ = writeln!(s, "ld.param.f32 %f1, {};\nld.param.f32 %f2, {};", p(3), p(4));
    for i in 5..CONV_PARAMS {
        let _ = writeln!(s, "ld.param.u32 %r{i}, {};", p(i));
    }
    let _ = writeln!(s, "mov.f32 %f0, 0f00000000;");
    let _ = writeln!(s, "mov.u32 %r43, %ctaid.x;\nmov.u32 %r42, %tid.x;");
    let _ = writeln!(s, "div.u32 %r44, %r6, %r20;\ndiv.u32 %r45, %r9, %r20;"); // Cg, Kg
    // element -> s, r, c local, k (as in gen_bwd_filter)
    let _ = writeln!(s, "rem.u32 %r50, %r43, %r11;\ndiv.u32 %r51, %r43, %r11;");
    let _ = writeln!(s, "rem.u32 %r52, %r51, %r10;\ndiv.u32 %r51, %r51, %r10;");
    let _ = writeln!(s, "rem.u32 %r53, %r51, %r44;\ndiv.u32 %r54, %r51, %r44;");
    let _ = writeln!(s, "div.u32 %r55, %r54, %r45;\nmad.lo.s32 %r56, %r55, %r44, %r53;");
    let _ = writeln!(s, "setp.ne.u32 %p2, %r21, 0;");
    let _ = writeln!(s, "sub.s32 %r57, %r10, 1;\nsub.s32 %r57, %r57, %r52;\nselp.u32 %r57, %r57, %r52, %p2;");
    let _ = writeln!(s, "sub.s32 %r58, %r11, 1;\nsub.s32 %r58, %r58, %r50;\nselp.u32 %r58, %r58, %r50, %p2;");
    let _ = writeln!(s, "mul.lo.u32 %r59, %r12, %r13;"); // PQ
    let _ = writeln!(s, "mul.lo.u32 %r60, %r59, %r5;"); // N*PQ
    let _ = writeln!(s, "mov.f32 %f3, 0f00000000;");
    let _ = writeln!(s, "mov.u32 %r61, %r42;"); // m
    let _ = writeln!(s, "$L_m:\nsetp.ge.u32 %p1, %r61, %r60;\n@%p1 bra $L_mdone;");
    let _ = writeln!(s, "div.u32 %r62, %r61, %r59;\nrem.u32 %r63, %r61, %r59;"); // n, pq
    let _ = writeln!(s, "div.u32 %r64, %r63, %r13;\nrem.u32 %r65, %r63, %r13;"); // p, q
    let _ = writeln!(s, "mul.lo.s32 %r66, %r64, %r16;\nsub.s32 %r66, %r66, %r14;\nmad.lo.s32 %r66, %r57, %r18, %r66;"); // ih
    let _ = writeln!(s, "mul.lo.s32 %r67, %r65, %r17;\nsub.s32 %r67, %r67, %r15;\nmad.lo.s32 %r67, %r58, %r19, %r67;"); // iw
    let _ = writeln!(s, "setp.ge.u32 %p3, %r66, %r7;\nsetp.ge.u32 %p4, %r67, %r8;\nor.pred %p3, %p3, %p4;\n@%p3 bra $L_mn;");
    emit_addr(&mut s, "%rd11", "%rd1", &[("%r62", "%r22"), ("%r56", "%r23"), ("%r66", "%r24"), ("%r67", "%r25")], e.shift());
    emit_addr(&mut s, "%rd12", "%rd2", &[("%r62", "%r30"), ("%r54", "%r31"), ("%r64", "%r32"), ("%r65", "%r33")], e.shift());
    emit_load(&mut s, e, "%rd11", "%f10");
    emit_load(&mut s, e, "%rd12", "%f11");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f10, %f11, %f3;");
    let _ = writeln!(s, "$L_mn:\nadd.s32 %r61, %r61, {BN_BLOCK};\nbra $L_m;\n$L_mdone:");
    bn_block_sum(&mut s, "%f3", "red");
    // thread 0 stores
    let _ = writeln!(s, "setp.ne.u32 %p8, %r42, 0;\n@%p8 bra $L_out;");
    emit_addr(&mut s, "%rd10", "%rd3", &[("%r54", "%r26"), ("%r53", "%r27"), ("%r52", "%r28"), ("%r50", "%r29")], e.shift());
    let _ = writeln!(s, "mul.ftz.f32 %f3, %f30, %f1;");
    let _ = writeln!(s, "setp.eq.ftz.f32 %p7, %f2, 0f00000000;\n@%p7 bra $L_store;");
    emit_load(&mut s, e, "%rd10", "%f14");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f2, %f14, %f3;");
    let _ = writeln!(s, "$L_store:");
    emit_store(&mut s, e, "%rd10", "%f3");
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}

/// `C = alpha * broadcast(A) + beta * C` over a 4-D logical shape (cudnnAddTensor).
/// Params: 0 A, 1 C (u64), 2 alpha, 3 beta (f32), 4..7 dims n,c,h,w, 8..11 A strides
/// (0 where A's dim is 1), 12..14 C strides n,c,h (w stride in 15 -> see below).
/// To keep the count at [`ADD_PARAMS`] the C `w` stride is assumed 1 or passed as
/// the last param: layout is 4 dims, 4 A strides, 4 C strides, 1 total = 13 u32 + 2 ptr = 15 + 2 f32 = 17.
pub fn generate_add_tensor(e: Elem) -> String {
    let name = add_tensor_entry(e);
    let mut s = String::new();
    // 0 A, 1 C, 2 alpha, 3 beta, 4..=7 dims, 8..=11 A strides, 12..=15 C strides, 16 total
    let mut tys = vec!["u64", "u64", "f32", "f32"];
    tys.extend(std::iter::repeat_n("u32", 13));
    header(&mut s, &name, &tys);
    let p = |i: usize| format!("[{name}_param_{i}]");
    let _ = writeln!(s, "ld.param.u64 %rd1, {};\ncvta.to.global.u64 %rd1, %rd1;", p(0));
    let _ = writeln!(s, "ld.param.u64 %rd2, {};\ncvta.to.global.u64 %rd2, %rd2;", p(1));
    let _ = writeln!(s, "ld.param.f32 %f1, {};\nld.param.f32 %f2, {};", p(2), p(3));
    for i in 4..17 {
        let _ = writeln!(s, "ld.param.u32 %r{i}, {};", p(i));
    }
    let _ = writeln!(s, "mov.u32 %r40, %ctaid.x;\nmov.u32 %r41, %ntid.x;\nmov.u32 %r42, %tid.x;");
    let _ = writeln!(s, "mad.lo.s32 %r43, %r40, %r41, %r42;");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r43, %r16;\n@%p1 bra $L_out;");
    let _ = writeln!(s, "rem.u32 %r50, %r43, %r7;\ndiv.u32 %r51, %r43, %r7;"); // w
    let _ = writeln!(s, "rem.u32 %r52, %r51, %r6;\ndiv.u32 %r51, %r51, %r6;"); // h
    let _ = writeln!(s, "rem.u32 %r53, %r51, %r5;\ndiv.u32 %r54, %r51, %r5;"); // c, n
    emit_addr(&mut s, "%rd10", "%rd1", &[("%r54", "%r8"), ("%r53", "%r9"), ("%r52", "%r10"), ("%r50", "%r11")], e.shift());
    emit_addr(&mut s, "%rd11", "%rd2", &[("%r54", "%r12"), ("%r53", "%r13"), ("%r52", "%r14"), ("%r50", "%r15")], e.shift());
    emit_load(&mut s, e, "%rd10", "%f10");
    let _ = writeln!(s, "mul.ftz.f32 %f3, %f10, %f1;");
    let _ = writeln!(s, "setp.eq.ftz.f32 %p7, %f2, 0f00000000;\n@%p7 bra $L_store;");
    emit_load(&mut s, e, "%rd11", "%f14");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f3, %f2, %f14, %f3;");
    let _ = writeln!(s, "$L_store:");
    emit_store(&mut s, e, "%rd11", "%f3");
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}

/// In-place `y = max(y, 0)` over a 4-D logical shape (used by ConvolutionBiasActivationForward).
/// Params: 0 y (u64), 1..=4 dims n,c,h,w, 5..=8 strides, 9 total.
pub fn generate_relu(e: Elem) -> String {
    let name = relu_entry(e);
    let mut s = String::new();
    let mut tys = vec!["u64"];
    tys.extend(["u32"; 9]);
    header(&mut s, &name, &tys);
    let p = |i: usize| format!("[{name}_param_{i}]");
    let _ = writeln!(s, "ld.param.u64 %rd1, {};\ncvta.to.global.u64 %rd1, %rd1;", p(0));
    for i in 1..10 {
        let _ = writeln!(s, "ld.param.u32 %r{i}, {};", p(i));
    }
    let _ = writeln!(s, "mov.f32 %f0, 0f00000000;");
    let _ = writeln!(s, "mov.u32 %r40, %ctaid.x;\nmov.u32 %r41, %ntid.x;\nmov.u32 %r42, %tid.x;");
    let _ = writeln!(s, "mad.lo.s32 %r43, %r40, %r41, %r42;");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r43, %r9;\n@%p1 bra $L_out;");
    let _ = writeln!(s, "rem.u32 %r50, %r43, %r4;\ndiv.u32 %r51, %r43, %r4;");
    let _ = writeln!(s, "rem.u32 %r52, %r51, %r3;\ndiv.u32 %r51, %r51, %r3;");
    let _ = writeln!(s, "rem.u32 %r53, %r51, %r2;\ndiv.u32 %r54, %r51, %r2;");
    emit_addr(&mut s, "%rd10", "%rd1", &[("%r54", "%r5"), ("%r53", "%r6"), ("%r52", "%r7"), ("%r50", "%r8")], e.shift());
    emit_load(&mut s, e, "%rd10", "%f10");
    let _ = writeln!(s, "max.ftz.f32 %f10, %f10, %f0;");
    emit_store(&mut s, e, "%rd10", "%f10");
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}

// --- Batch normalization (spatial) ---------------------------------------
// Element (n, c, i) with i in 0..HW (h*W+w flattened) lives at
// n*sn + c*sc + (i / W)*sh + (i % W)*sw (elements). M = N*HW.
// One block of BN_BLOCK threads per channel.

/// Per-channel element address into `%rdDst` from `base`: uses n=%r90, c=%r91, i=%r92.
fn bn_addr(s: &mut String, dst: &str, base: &str, e: Elem) {
    // %r93 = i / W, %r94 = i % W
    let _ = writeln!(s, "div.u32 %r93, %r92, %r8;");
    let _ = writeln!(s, "rem.u32 %r94, %r92, %r8;");
    emit_addr(s, dst, base, &[("%r90", "%r9"), ("%r91", "%r10"), ("%r93", "%r11"), ("%r94", "%r12")], e.shift());
}

/// Shared tree reduction of f32 `%fSrc` over the block into `%f30` (all threads get result).
fn bn_block_sum(s: &mut String, src: &str, tag: &str) {
    let _ = writeln!(s, "mov.u32 %r95, red;");
    let _ = writeln!(s, "shl.b32 %r96, %r42, 2;");
    let _ = writeln!(s, "add.s32 %r96, %r95, %r96;");
    let _ = writeln!(s, "st.shared.f32 [%r96], {src};");
    let _ = writeln!(s, "bar.sync 0;");
    let mut step = BN_BLOCK / 2;
    while step >= 1 {
        let _ = writeln!(s, "setp.ge.u32 %p9, %r42, {step};");
        let _ = writeln!(s, "@%p9 bra $L_{tag}_skip{step};");
        let _ = writeln!(s, "ld.shared.f32 %f31, [%r96];");
        let _ = writeln!(s, "ld.shared.f32 %f32, [%r96+{}];", step * 4);
        let _ = writeln!(s, "add.ftz.f32 %f31, %f31, %f32;");
        let _ = writeln!(s, "st.shared.f32 [%r96], %f31;");
        let _ = writeln!(s, "$L_{tag}_skip{step}:");
        let _ = writeln!(s, "bar.sync 0;");
        step /= 2;
    }
    let _ = writeln!(s, "ld.shared.f32 %f30, [%r95];");
    let _ = writeln!(s, "bar.sync 0;");
}

/// Loop `for n in 0..N, for i = tid; i < HW; i += BN_BLOCK` body at label tag.
fn bn_loop_begin(s: &mut String, tag: &str) {
    let _ = writeln!(s, "mov.u32 %r90, 0;");
    let _ = writeln!(s, "$L_{tag}_n:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r90, %r5;\n@%p1 bra $L_{tag}_ndone;");
    let _ = writeln!(s, "mov.u32 %r92, %r42;");
    let _ = writeln!(s, "$L_{tag}_i:");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r92, %r7;\n@%p1 bra $L_{tag}_idone;");
}
fn bn_loop_end(s: &mut String, tag: &str) {
    let _ = writeln!(s, "add.s32 %r92, %r92, {BN_BLOCK};\nbra $L_{tag}_i;");
    let _ = writeln!(s, "$L_{tag}_idone:\nadd.s32 %r90, %r90, 1;\nbra $L_{tag}_n;\n$L_{tag}_ndone:");
}

/// Common BN prologue. Params shared prefix handled by callers; loads dims into
/// %r5 N, %r6 C, %r7 HW, %r8 W, %r9..%r12 strides (n,c,h,w); %r42 tid, %r91 channel.
fn bn_dims_prologue(s: &mut String, first_u32_param: usize, name: &str) {
    let p = |i: usize| format!("[{name}_param_{i}]");
    for k in 0..8 {
        let _ = writeln!(s, "ld.param.u32 %r{}, {};", 5 + k, p(first_u32_param + k));
    }
    let _ = writeln!(s, "mov.u32 %r42, %tid.x;\nmov.u32 %r91, %ctaid.x;");
}

/// Batch-norm forward training. Params: 0 x, 1 y, 2 scale, 3 bias, 4 runMean, 5 runVar,
/// 6 saveMean, 7 saveInvVar (u64; running/save may be null), 8 alpha, 9 beta, 10 expAvgFactor,
/// 11 eps (f32), 12..19 u32: N, C, HW, W, sn, sc, sh, sw (same strides for x and y).
/// Launch: grid (C), block (BN_BLOCK).
pub fn generate_bn_train(e: Elem) -> String {
    let name = bn_train_entry(e);
    let mut s = String::new();
    let mut tys = vec!["u64"; 8];
    tys.extend(["f32"; 4]);
    tys.extend(["u32"; 8]);
    header(&mut s, &name, &tys);
    let _ = writeln!(s, ".shared .align 4 .b8 red[{}];", BN_BLOCK * 4);
    let p = |i: usize| format!("[{name}_param_{i}]");
    for i in 0..8 {
        let _ = writeln!(s, "ld.param.u64 %rd{}, {};", i + 1, p(i));
        let _ = writeln!(s, "cvta.to.global.u64 %rd{0}, %rd{0};", i + 1);
    }
    for i in 0..4 {
        let _ = writeln!(s, "ld.param.f32 %f{}, {};", i + 1, p(8 + i));
    }
    bn_dims_prologue(&mut s, 12, &name);
    let _ = writeln!(s, "mov.f32 %f0, 0f00000000;");
    let _ = writeln!(s, "cvt.rn.f32.u32 %f5, %r5;\ncvt.rn.f32.u32 %f6, %r7;\nmul.ftz.f32 %f5, %f5, %f6;"); // M
    // pass 1: sum
    let _ = writeln!(s, "mov.f32 %f7, 0f00000000;");
    bn_loop_begin(&mut s, "a");
    bn_addr(&mut s, "%rd10", "%rd1", e);
    emit_load(&mut s, e, "%rd10", "%f10");
    let _ = writeln!(s, "add.ftz.f32 %f7, %f7, %f10;");
    bn_loop_end(&mut s, "a");
    bn_block_sum(&mut s, "%f7", "sa");
    let _ = writeln!(s, "div.rn.ftz.f32 %f8, %f30, %f5;"); // mean
    // pass 2: variance
    let _ = writeln!(s, "mov.f32 %f7, 0f00000000;");
    bn_loop_begin(&mut s, "b");
    bn_addr(&mut s, "%rd10", "%rd1", e);
    emit_load(&mut s, e, "%rd10", "%f10");
    let _ = writeln!(s, "sub.ftz.f32 %f10, %f10, %f8;");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f7, %f10, %f10, %f7;");
    bn_loop_end(&mut s, "b");
    bn_block_sum(&mut s, "%f7", "sb");
    let _ = writeln!(s, "div.rn.ftz.f32 %f9, %f30, %f5;"); // biased var
    let _ = writeln!(s, "add.ftz.f32 %f12, %f9, %f4;"); // var + eps
    let _ = writeln!(s, "sqrt.rn.ftz.f32 %f12, %f12;");
    let _ = writeln!(s, "mov.f32 %f13, 0f3F800000;");
    let _ = writeln!(s, "div.rn.ftz.f32 %f12, %f13, %f12;"); // invstd
    // scale/bias for this channel
    let _ = writeln!(s, "mul.wide.u32 %rd20, %r91, 4;");
    let _ = writeln!(s, "add.s64 %rd21, %rd3, %rd20;\nld.global.f32 %f14, [%rd21];"); // scale
    let _ = writeln!(s, "add.s64 %rd21, %rd4, %rd20;\nld.global.f32 %f15, [%rd21];"); // bias
    // pass 3: write y = alpha*(scale*(x-mean)*invstd + bias) + beta*y
    let _ = writeln!(s, "mul.ftz.f32 %f16, %f14, %f12;"); // scale*invstd
    let _ = writeln!(s, "setp.eq.ftz.f32 %p7, %f2, 0f00000000;");
    bn_loop_begin(&mut s, "c");
    bn_addr(&mut s, "%rd10", "%rd1", e);
    emit_load(&mut s, e, "%rd10", "%f10");
    let _ = writeln!(s, "sub.ftz.f32 %f10, %f10, %f8;");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f10, %f10, %f16, %f15;");
    let _ = writeln!(s, "mul.ftz.f32 %f10, %f10, %f1;");
    bn_addr(&mut s, "%rd11", "%rd2", e);
    let _ = writeln!(s, "@%p7 bra $L_c_store;");
    emit_load(&mut s, e, "%rd11", "%f17");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f10, %f2, %f17, %f10;");
    let _ = writeln!(s, "$L_c_store:");
    emit_store(&mut s, e, "%rd11", "%f10");
    bn_loop_end(&mut s, "c");
    // thread 0 writes running stats and saved stats
    let _ = writeln!(s, "setp.ne.u32 %p10, %r42, 0;\n@%p10 bra $L_out;");
    let _ = writeln!(s, "setp.eq.u64 %p11, %rd5, 0;\n@%p11 bra $L_norun;");
    let _ = writeln!(s, "mov.f32 %f18, 0f3F800000;\nsub.ftz.f32 %f18, %f18, %f3;"); // 1 - factor
    let _ = writeln!(s, "add.s64 %rd21, %rd5, %rd20;\nld.global.f32 %f19, [%rd21];");
    let _ = writeln!(s, "mul.ftz.f32 %f19, %f19, %f18;\nfma.rn.ftz.f32 %f19, %f3, %f8, %f19;");
    let _ = writeln!(s, "st.global.f32 [%rd21], %f19;");
    // unbiased var = var * M / max(M-1, 1)
    let _ = writeln!(s, "mov.f32 %f20, 0f3F800000;\nsub.ftz.f32 %f21, %f5, %f20;\nmax.ftz.f32 %f21, %f21, %f20;");
    let _ = writeln!(s, "mul.ftz.f32 %f22, %f9, %f5;\ndiv.rn.ftz.f32 %f22, %f22, %f21;");
    let _ = writeln!(s, "add.s64 %rd21, %rd6, %rd20;\nld.global.f32 %f23, [%rd21];");
    let _ = writeln!(s, "mul.ftz.f32 %f23, %f23, %f18;\nfma.rn.ftz.f32 %f23, %f3, %f22, %f23;");
    let _ = writeln!(s, "st.global.f32 [%rd21], %f23;");
    let _ = writeln!(s, "$L_norun:");
    let _ = writeln!(s, "setp.eq.u64 %p11, %rd7, 0;\n@%p11 bra $L_nosave;");
    let _ = writeln!(s, "add.s64 %rd21, %rd7, %rd20;\nst.global.f32 [%rd21], %f8;");
    let _ = writeln!(s, "add.s64 %rd21, %rd8, %rd20;\nst.global.f32 [%rd21], %f12;");
    let _ = writeln!(s, "$L_nosave:");
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}

/// Batch-norm inference (elementwise). Params: 0 x, 1 y, 2 scale, 3 bias, 4 mean, 5 var (u64),
/// 6 alpha, 7 beta, 8 eps (f32), 9..16 u32: N, C, HW, W, sn, sc, sh, sw — followed by
/// total in param 17 (u32) => [`BN_INFER_PARAMS`] = 17 + ... see tys below.
/// Launch: block 256, grid ceil(total/256).
pub fn generate_bn_infer(e: Elem) -> String {
    let name = bn_infer_entry(e);
    let mut s = String::new();
    let mut tys = vec!["u64"; 6];
    tys.extend(["f32"; 3]);
    tys.extend(["u32"; 8]);
    header(&mut s, &name, &tys);
    let p = |i: usize| format!("[{name}_param_{i}]");
    for i in 0..6 {
        let _ = writeln!(s, "ld.param.u64 %rd{}, {};\ncvta.to.global.u64 %rd{0}, %rd{0};", i + 1, p(i));
    }
    for i in 0..3 {
        let _ = writeln!(s, "ld.param.f32 %f{}, {};", i + 1, p(6 + i));
    }
    for k in 0..8 {
        let _ = writeln!(s, "ld.param.u32 %r{}, {};", 5 + k, p(9 + k));
    }
    // total elements = N*C*HW computed here.
    let _ = writeln!(s, "mul.lo.u32 %r20, %r5, %r6;\nmul.lo.u32 %r20, %r20, %r7;");
    let _ = writeln!(s, "mov.u32 %r40, %ctaid.x;\nmov.u32 %r41, %ntid.x;\nmov.u32 %r42, %tid.x;");
    let _ = writeln!(s, "mad.lo.s32 %r43, %r40, %r41, %r42;");
    let _ = writeln!(s, "setp.ge.u32 %p1, %r43, %r20;\n@%p1 bra $L_out;");
    let _ = writeln!(s, "rem.u32 %r92, %r43, %r7;\ndiv.u32 %r51, %r43, %r7;");
    let _ = writeln!(s, "rem.u32 %r91, %r51, %r6;\ndiv.u32 %r90, %r51, %r6;");
    bn_addr(&mut s, "%rd10", "%rd1", e);
    bn_addr(&mut s, "%rd11", "%rd2", e);
    let _ = writeln!(s, "mul.wide.u32 %rd20, %r91, 4;");
    for (rd, f) in [(3, 14), (4, 15), (5, 16), (6, 17)] {
        let _ = writeln!(s, "add.s64 %rd21, %rd{rd}, %rd20;\nld.global.f32 %f{f}, [%rd21];");
    }
    emit_load(&mut s, e, "%rd10", "%f10");
    let _ = writeln!(s, "add.ftz.f32 %f18, %f17, %f3;\nsqrt.rn.ftz.f32 %f18, %f18;");
    let _ = writeln!(s, "mov.f32 %f19, 0f3F800000;\ndiv.rn.ftz.f32 %f18, %f19, %f18;");
    let _ = writeln!(s, "sub.ftz.f32 %f10, %f10, %f16;");
    let _ = writeln!(s, "mul.ftz.f32 %f20, %f14, %f18;");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f10, %f10, %f20, %f15;");
    let _ = writeln!(s, "mul.ftz.f32 %f10, %f10, %f1;");
    let _ = writeln!(s, "setp.eq.ftz.f32 %p7, %f2, 0f00000000;\n@%p7 bra $L_store;");
    emit_load(&mut s, e, "%rd11", "%f21");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f10, %f2, %f21, %f10;");
    let _ = writeln!(s, "$L_store:");
    emit_store(&mut s, e, "%rd11", "%f10");
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}

/// Batch-norm backward. Params: 0 x, 1 dy, 2 dx, 3 scale, 4 dScale, 5 dBias, 6 saveMean,
/// 7 saveInvVar (u64; save* may be null: statistics are recomputed), 8 alphaData, 9 betaData,
/// 10 alphaParam, 11 betaParam, 12 eps (f32), 13..20 u32: N, C, HW, W, sn, sc, sh, sw.
/// Launch: grid (C), block (BN_BLOCK).
pub fn generate_bn_bwd(e: Elem) -> String {
    let name = bn_bwd_entry(e);
    let mut s = String::new();
    let mut tys = vec!["u64"; 8];
    tys.extend(["f32"; 5]);
    tys.extend(["u32"; 8]);
    header(&mut s, &name, &tys);
    let _ = writeln!(s, ".shared .align 4 .b8 red[{}];", BN_BLOCK * 4);
    let p = |i: usize| format!("[{name}_param_{i}]");
    for i in 0..8 {
        let _ = writeln!(s, "ld.param.u64 %rd{}, {};\ncvta.to.global.u64 %rd{0}, %rd{0};", i + 1, p(i));
    }
    for i in 0..5 {
        let _ = writeln!(s, "ld.param.f32 %f{}, {};", i + 1, p(8 + i));
    }
    bn_dims_prologue(&mut s, 13, &name);
    let _ = writeln!(s, "mov.f32 %f0, 0f00000000;\nmov.f32 %f13, 0f3F800000;");
    let _ = writeln!(s, "cvt.rn.f32.u32 %f6, %r5;\ncvt.rn.f32.u32 %f7, %r7;\nmul.ftz.f32 %f6, %f6, %f7;"); // M
    let _ = writeln!(s, "mul.wide.u32 %rd20, %r91, 4;");
    let _ = writeln!(s, "setp.eq.u64 %p11, %rd7, 0;\n@%p11 bra $L_recompute;");
    let _ = writeln!(s, "add.s64 %rd21, %rd7, %rd20;\nld.global.f32 %f8, [%rd21];"); // mean
    let _ = writeln!(s, "add.s64 %rd21, %rd8, %rd20;\nld.global.f32 %f12, [%rd21];"); // invstd
    let _ = writeln!(s, "bra $L_haveStats;");
    let _ = writeln!(s, "$L_recompute:");
    let _ = writeln!(s, "mov.f32 %f9, 0f00000000;");
    bn_loop_begin(&mut s, "m1");
    bn_addr(&mut s, "%rd10", "%rd1", e);
    emit_load(&mut s, e, "%rd10", "%f10");
    let _ = writeln!(s, "add.ftz.f32 %f9, %f9, %f10;");
    bn_loop_end(&mut s, "m1");
    bn_block_sum(&mut s, "%f9", "s1");
    let _ = writeln!(s, "div.rn.ftz.f32 %f8, %f30, %f6;");
    let _ = writeln!(s, "mov.f32 %f9, 0f00000000;");
    bn_loop_begin(&mut s, "m2");
    bn_addr(&mut s, "%rd10", "%rd1", e);
    emit_load(&mut s, e, "%rd10", "%f10");
    let _ = writeln!(s, "sub.ftz.f32 %f10, %f10, %f8;\nfma.rn.ftz.f32 %f9, %f10, %f10, %f9;");
    bn_loop_end(&mut s, "m2");
    bn_block_sum(&mut s, "%f9", "s2");
    let _ = writeln!(s, "div.rn.ftz.f32 %f12, %f30, %f6;\nadd.ftz.f32 %f12, %f12, %f5;");
    let _ = writeln!(s, "sqrt.rn.ftz.f32 %f12, %f12;\ndiv.rn.ftz.f32 %f12, %f13, %f12;");
    let _ = writeln!(s, "$L_haveStats:");
    // sums: dbias = sum dy, dscale = sum dy * (x-mean)*invstd
    let _ = writeln!(s, "mov.f32 %f9, 0f00000000;\nmov.f32 %f14, 0f00000000;");
    bn_loop_begin(&mut s, "d");
    bn_addr(&mut s, "%rd10", "%rd1", e);
    bn_addr(&mut s, "%rd11", "%rd2", e);
    emit_load(&mut s, e, "%rd10", "%f10");
    emit_load(&mut s, e, "%rd11", "%f15");
    let _ = writeln!(s, "sub.ftz.f32 %f10, %f10, %f8;\nmul.ftz.f32 %f10, %f10, %f12;");
    let _ = writeln!(s, "add.ftz.f32 %f9, %f9, %f15;\nfma.rn.ftz.f32 %f14, %f15, %f10, %f14;");
    bn_loop_end(&mut s, "d");
    bn_block_sum(&mut s, "%f9", "s3");
    let _ = writeln!(s, "mov.f32 %f16, %f30;"); // dbias
    bn_block_sum(&mut s, "%f14", "s4");
    let _ = writeln!(s, "mov.f32 %f17, %f30;"); // dscale
    // dx = scale*invstd*(dy - dbias/M - xhat*dscale/M)
    let _ = writeln!(s, "add.s64 %rd21, %rd4, %rd20;\nld.global.f32 %f18, [%rd21];"); // scale
    let _ = writeln!(s, "mul.ftz.f32 %f18, %f18, %f12;");
    let _ = writeln!(s, "div.rn.ftz.f32 %f19, %f16, %f6;\ndiv.rn.ftz.f32 %f20, %f17, %f6;");
    let _ = writeln!(s, "setp.eq.ftz.f32 %p7, %f2, 0f00000000;");
    bn_loop_begin(&mut s, "e");
    bn_addr(&mut s, "%rd10", "%rd1", e);
    bn_addr(&mut s, "%rd11", "%rd2", e);
    bn_addr(&mut s, "%rd12", "%rd3", e);
    emit_load(&mut s, e, "%rd10", "%f10");
    emit_load(&mut s, e, "%rd11", "%f15");
    let _ = writeln!(s, "sub.ftz.f32 %f10, %f10, %f8;\nmul.ftz.f32 %f10, %f10, %f12;"); // xhat
    let _ = writeln!(s, "sub.ftz.f32 %f15, %f15, %f19;");
    let _ = writeln!(s, "mul.ftz.f32 %f10, %f10, %f20;\nsub.ftz.f32 %f15, %f15, %f10;");
    let _ = writeln!(s, "mul.ftz.f32 %f15, %f15, %f18;\nmul.ftz.f32 %f15, %f15, %f1;");
    let _ = writeln!(s, "@%p7 bra $L_e_store;");
    emit_load(&mut s, e, "%rd12", "%f21");
    let _ = writeln!(s, "fma.rn.ftz.f32 %f15, %f2, %f21, %f15;");
    let _ = writeln!(s, "$L_e_store:");
    emit_store(&mut s, e, "%rd12", "%f15");
    bn_loop_end(&mut s, "e");
    // thread 0 writes dscale/dbias (f32): out = alphaParam*v + betaParam*old
    let _ = writeln!(s, "setp.ne.u32 %p10, %r42, 0;\n@%p10 bra $L_out;");
    let _ = writeln!(s, "setp.eq.ftz.f32 %p8, %f4, 0f00000000;");
    for (rd, f) in [(5, 17), (6, 16)] {
        // rd5 = dScale, rd6 = dBias (params 4, 5)
        let _ = writeln!(s, "add.s64 %rd21, %rd{rd}, %rd20;");
        let _ = writeln!(s, "mul.ftz.f32 %f22, %f{f}, %f3;");
        let _ = writeln!(s, "@%p8 bra $L_ps{rd};");
        let _ = writeln!(s, "ld.global.f32 %f23, [%rd21];\nfma.rn.ftz.f32 %f22, %f4, %f23, %f22;");
        let _ = writeln!(s, "$L_ps{rd}:\nst.global.f32 [%rd21], %f22;");
    }
    let _ = writeln!(s, "$L_out:\nret;\n}}");
    s
}

pub fn all_entries() -> Vec<(String, String, usize)> {
    let mut v = Vec::new();
    for e in Elem::all() {
        for k in [ConvKind::Fwd, ConvKind::BwdData, ConvKind::BwdFilter] {
            v.push((conv_entry(k, e), generate_conv(k, e), CONV_PARAMS));
        }
        v.push((conv_bwd_filter_red_entry(e), generate_conv_bwd_filter_red(e), CONV_PARAMS));
        for f in [false, true] {
            v.push((im2col_entry(e, f), generate_im2col(e, f), IM2COL_PARAMS));
            v.push((col2im_entry(e, f), generate_col2im(e, f), COL2IM_PARAMS));
        }
        v.push((add_tensor_entry(e), generate_add_tensor(e), ADD_PARAMS));
        v.push((relu_entry(e), generate_relu(e), RELU_PARAMS));
        v.push((bn_train_entry(e), generate_bn_train(e), BN_TRAIN_PARAMS));
        v.push((bn_infer_entry(e), generate_bn_infer(e), BN_INFER_PARAMS));
        v.push((bn_bwd_entry(e), generate_bn_bwd(e), BN_BWD_PARAMS));
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entries_have_expected_param_counts() {
        for (name, src, n) in all_entries() {
            assert!(src.contains(&format!(".entry {name}(")), "{name}");
            assert_eq!(src.matches(".param ").count(), n, "{name}");
        }
    }

    #[test]
    fn no_return_before_barrier() {
        for (name, src, _) in all_entries() {
            if let Some(last_bar) = src.rfind("bar.sync") {
                let first_ret_branch = src.find("bra $L_out").unwrap();
                // BN kernels branch to $L_out only after the last barrier.
                assert!(first_ret_branch > last_bar, "{name}");
            }
        }
    }
}
