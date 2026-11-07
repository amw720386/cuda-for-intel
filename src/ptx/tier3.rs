//! Tier-3 PTX lowering: carry chains, video SIMD, partial-mask warp
//! collectives and a handful of remaining odd spellings.
//!
//! Like `normalize.rs` this works on text: each rewrite expands one PTX
//! instruction into a short run of instructions the parser already knows. It is
//! a child module of `normalize` (see the `#[path]` line there) so it can use
//! `Ctx`, `Inst` and `render` directly. `rewrite` is consulted first; `None`
//! means "not mine" and the legacy rewrites continue.

use super::{Ctx, Inst, render};

#[path = "tier3_int16.rs"]
mod int16;

/// Per-entry state owned by this module (embedded in `Ctx`).
#[derive(Default)]
pub(super) struct State {
    /// Scratch registers requested by tier-3 rewrites of the current entry.
    x32: u32,
    x64: u32,
    xp: u32,
    /// Scratch registers of the tensor-core emulation (`tier3_mma.rs`).
    pub(super) m32: u32,
    pub(super) m64: u32,
    pub(super) mp: u32,
    pub(super) addr_seq: u32,
    pub(super) addr_max: u32,
    /// The condition-code carry flag `CC.CF` is used by the entry.
    cc: bool,
    /// Named-barrier and `bar.red` lowering need shared counters.
    named_barriers: bool,
    reductions: bool,
    labels: u32,
    /// `.local` arrays of the current entry merged into one: (name, byte offset).
    locals: Vec<(String, u32)>,
    local_bytes: u32,
    local_align: u32,
    /// `.shared` arrays of the current entry.
    shared: Vec<String>,
    uses_lsym: bool,
    uses_ssym: bool,
    /// Names of module-scope `.global` / `.const` data (kept across entries).
    module_globals: Vec<String>,
    sink_count: u32,
    sink_max: u32,
    sink_wide: bool,
    uses_sr: bool,
}

impl State {
    /// Declarations (and initialisers) to splice in front of the entry body.
    pub(super) fn take_decls(&mut self) -> Vec<String> {
        let mut decls = Vec::new();
        if self.x32 > 0 {
            decls.push(format!(".reg .b32 %__x32_<{}> ;", self.x32));
        }
        if self.x64 > 0 {
            decls.push(format!(".reg .b64 %__x64_<{}> ;", self.x64));
        }
        if self.xp > 0 {
            decls.push(format!(".reg .pred %__xp_<{}> ;", self.xp));
        }
        if self.m32 > 0 {
            decls.push(format!(".reg .b32 %__m32_<{}> ;", self.m32));
        }
        if self.addr_max > 0 {
            decls.push(format!(".reg .b64 %__ma_<{}> ;", self.addr_max));
        }
        if self.m64 > 0 {
            decls.push(format!(".reg .b64 %__m64_<{}> ;", self.m64));
        }
        if self.mp > 0 {
            decls.push(format!(".reg .pred %__mp_<{}> ;", self.mp));
        }
        if self.local_bytes > 0 {
            decls.push(format!(".local .align {} .b8 __t3_local [ {} ] ;", self.local_align, self.local_bytes));
        }
        if self.uses_lsym {
            decls.push(".reg .b64 %__lsym ;".into());
        }
        if self.uses_ssym {
            decls.push(".reg .b32 %__ssym ;".into());
        }
        if self.uses_sr {
            decls.push(".reg .b32 %__sr ;".into());
        }
        if self.sink_max > 0 {
            decls.push(format!(".reg .b32 %__sink_<{}> ;", self.sink_max));
        }
        if self.cc {
            decls.push(".reg .b32 %__cc ;".into());
            decls.push("mov.b32 %__cc , 0 ;".into());
        }
        if self.named_barriers || self.reductions {
            // 16 named barriers of (arrival count, generation) plus one reduction word.
            decls.push(".shared .align 4 .b8 __t3_bar [ 136 ] ;".into());
            decls.push(".reg .b32 %__bl ;".into());
            decls.push(".reg .b32 %__bt ;".into());
            decls.push(".reg .b32 %__bs ;".into());
            decls.push(".reg .b32 %__bz ;".into());
            decls.push(".reg .b32 %__bone ;".into());
            decls.push("mov.b32 %__bz , 0 ;".into());
            decls.push("mov.b32 %__bone , 1 ;".into());
            decls.push("mov.u32 %__bs , __t3_bar ;".into());
            decls.push(".reg .pred %__bp ;".into());
            // Zero the barrier words once: thread i of the first 34 clears word i.
            for line in [
                "mov.u32 %__bt , %ntid.y ;",
                "mov.u32 %__bl , %tid.z ;",
                "mul.lo.u32 %__bt , %__bt , %__bl ;",
                "mov.u32 %__bl , %tid.y ;",
                "add.u32 %__bt , %__bt , %__bl ;",
                "mov.u32 %__bl , %ntid.x ;",
                "mul.lo.u32 %__bt , %__bt , %__bl ;",
                "mov.u32 %__bl , %tid.x ;",
                "add.u32 %__bt , %__bt , %__bl ;",
            ] {
                decls.push(line.into());
            }
            for i in 0..34 {
                decls.push(format!("setp.eq.u32 %__bp , %__bt , {i} ;"));
                decls.push(format!("@ %__bp st.shared.u32 [%__bs+{}] , %__bz ;", 4 * i));
            }
            decls.push("bar.sync 0 ;".into());
        }
        let keep = std::mem::take(&mut self.module_globals);
        *self = State::default();
        self.module_globals = keep;
        decls
    }
}

/// Scratch-register allocator for one rewrite.
struct B<'a> {
    st: &'a mut State,
    lines: Vec<String>,
    n32: u32,
    n64: u32,
    np: u32,
}

impl<'a> B<'a> {
    fn new(st: &'a mut State) -> Self {
        B { st, lines: Vec::new(), n32: 0, n64: 0, np: 0 }
    }
    fn t32(&mut self) -> String {
        self.n32 += 1;
        format!("%__x32_{}", self.n32 - 1)
    }
    fn t64(&mut self) -> String {
        self.n64 += 1;
        format!("%__x64_{}", self.n64 - 1)
    }
    fn tp(&mut self) -> String {
        self.np += 1;
        format!("%__xp_{}", self.np - 1)
    }
    fn tn(&mut self, bits: u32) -> String {
        if bits == 64 { self.t64() } else { self.t32() }
    }
    fn p(&mut self, line: impl Into<String>) {
        self.lines.push(line.into());
    }
    fn finish(self) -> Vec<String> {
        self.st.x32 = self.st.x32.max(self.n32);
        self.st.x64 = self.st.x64.max(self.n64);
        self.st.xp = self.st.xp.max(self.np);
        self.lines
    }
    /// A register holding `operand`: immediates are moved into a scratch.
    fn reg(&mut self, operand: &str, bits: u32) -> String {
        if is_imm(operand) {
            let t = self.tn(bits);
            self.p(format!("mov.b{bits} {t} , {operand} ;"));
            t
        } else {
            operand.to_owned()
        }
    }
}

fn is_imm(text: &str) -> bool {
    let t = text.trim();
    t.starts_with(|c: char| c.is_ascii_digit()) || t.starts_with('-') || t.starts_with("0x")
}

/// Runs the rest of the rewrite pipeline on `inst` (used after renaming).
fn again(ctx: &mut Ctx, inst: &Inst) -> Vec<String> {
    super::rewrite_inst(ctx, inst).unwrap_or_else(|| vec![inst.render()])
}

fn with_op(inst: &Inst, op: String) -> Inst {
    Inst { guard: inst.guard.clone(), op, operands: inst.operands.clone() }
}

/// `0xffU` / `12u` literals lose their suffix.
fn strip_suffix_literal(operand: &str) -> Option<String> {
    let t = operand.trim();
    let body = t.strip_suffix(['U', 'u'])?;
    let digits = body.strip_prefix("0x").or_else(|| body.strip_prefix("0X")).unwrap_or(body);
    (!digits.is_empty() && digits.bytes().all(|b| b.is_ascii_hexdigit()) && body.starts_with(|c: char| c.is_ascii_digit()))
        .then(|| body.to_owned())
}

pub(super) fn rewrite(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    if inst.operands.iter().any(|o| strip_suffix_literal(o).is_some()) {
        let mut copy = with_op(inst, inst.op.clone());
        for o in &mut copy.operands {
            if let Some(clean) = strip_suffix_literal(o) {
                *o = clean;
            }
        }
        return Some(again(ctx, &copy));
    }
    if inst.operands.iter().any(|o| o == "WARP_SZ" || o == "%WARP_SZ") {
        let mut copy = with_op(inst, inst.op.clone());
        for o in &mut copy.operands {
            if o == "WARP_SZ" || o == "%WARP_SZ" {
                *o = "32".into();
            }
        }
        return Some(again(ctx, &copy));
    }
    if let Some(lines) = sinks_and_special_registers(ctx, inst) {
        return Some(lines);
    }
    if let Some(lines) = symbol_operands(ctx, inst) {
        return Some(lines);
    }
    let family = inst.op.split('.').next().unwrap_or("");
    if matches!(family, "mma" | "wmma") {
        return super::tier3_mma::rewrite(ctx, inst);
    }
    if let Some(lines) = int16::rewrite(&mut ctx.t3, inst) {
        return Some(lines);
    }
    let st = &mut ctx.t3;
    match family {
        "ld" if inst.op.split('.').any(|p| p == "param") && inst.op.split('.').any(|p| matches!(p, "v2" | "v4")) => {
            // Vector parameter loads (byval structs) become one scalar load per element.
            let parts: Vec<&str> = inst.op.split('.').collect();
            let ty = *parts.last()?;
            let size: i64 = match ty {
                "u8" | "s8" | "b8" => 1,
                "u16" | "s16" | "b16" | "f16" | "bf16" => 2,
                "u32" | "s32" | "b32" | "f32" | "f16x2" | "bf16x2" => 4,
                "u64" | "s64" | "b64" | "f64" => 8,
                _ => return None,
            };
            let regs: Vec<String> = inst.operand(0).trim().trim_start_matches('{').trim_end_matches('}').split(',').map(|r| r.trim().to_owned()).filter(|r| !r.is_empty()).collect();
            let (name, base) = bracket_symbol(inst.operand(1))?;
            let mut lines = Vec::new();
            for (i, reg) in regs.iter().enumerate() {
                let copy = Inst {
                    guard: inst.guard.clone(),
                    op: format!("ld.param.{ty}"),
                    operands: vec![reg.clone(), format!("[{name}+{}]", base + size * i as i64)],
                };
                lines.extend(again(ctx, &copy));
            }
            Some(lines)
        }
        "ld" | "ldu" if inst.op.split('.').any(|p| p == "const") => {
            // Constant-bank loads read ordinary device memory.
            let parts: Vec<&str> = inst.op.split('.').filter(|p| *p != "const").collect();
            let mut op = parts.join(".");
            op = op.replacen("ld.", "ld.global.", 1);
            let copy = with_op(inst, op);
            Some(again(ctx, &copy))
        }
        "cvta" if inst.op.split('.').any(|p| p == "const") && inst.operands.len() == 2 => {
            Some(vec![render(&inst.guard, "cvta.global.u64", &inst.operands)])
        }
        "mov" if matches!(inst.operand(1), "%dynamic_smem_size" | "%total_smem_size") && inst.operands.len() == 2 => {
            let bits = if inst.op.ends_with("64") { 64 } else { 32 };
            Some(vec![render(&inst.guard, &format!("mov.b{bits}"), &[inst.operand(0).to_owned(), "49152".into()])])
        }
        "bar" | "barrier" => barrier(ctx, inst),
        "stmatrix" => stmatrix(ctx, inst),
        "atom" | "red" => atomic_forms(ctx, inst),
        "mov" if matches!(inst.op.as_str(), "mov.b32" | "mov.s32")
            && inst.operands.len() == 2
            && (inst.operand(1).starts_with("%tid.") || inst.operand(1).starts_with("%ntid.")
                || inst.operand(1).starts_with("%ctaid.") || inst.operand(1).starts_with("%nctaid.")
                || inst.operand(1) == "%laneid") =>
        {
            Some(vec![render(&inst.guard, "mov.u32", &inst.operands)])
        }
        "shfl" => shfl_masks(ctx, inst),
        "vote" => vote_masks(ctx, inst),
        "cvt" => cvt_misc(ctx, inst),
        "fma" if matches!(inst.op.rsplit('.').next(), Some("f16" | "bf16" | "f16x2" | "bf16x2"))
            && !inst.op.split('.').any(|p| matches!(p, "relu" | "sat" | "rz" | "rm" | "rp")) =>
        {
            half_fma(st, inst)
        }
        "add" | "sub" | "mul" | "fma" | "div" | "sqrt" | "rcp"
            if inst.op.ends_with(".f64") && inst.op.split('.').any(|p| matches!(p, "rz" | "rm" | "rp")) =>
        {
            f64_directed(st, inst)
        }
        "add" | "sub" | "addc" | "subc" | "madc" | "mad" => carry(st, inst),
        "vadd" | "vsub" | "vabsdiff" | "vmin" | "vmax" | "vset" | "vshl" | "vshr" | "vmad" | "vavrg" => {
            video_scalar(st, inst)
        }
        "vadd2" | "vsub2" | "vabsdiff2" | "vmin2" | "vmax2" | "vavrg2" | "vset2" | "vadd4" | "vsub4"
        | "vabsdiff4" | "vmin4" | "vmax4" | "vavrg4" | "vset4" => video_simd(st, inst),
        "setp" | "set" => setp_pair(st, inst),
        "selp" => selp_literal(st, inst),
        _ => None,
    }
}

// ---- warp collectives with partial masks ---------------------------------

fn full_mask(text: &str) -> bool {
    matches!(super::immediate(text), Some(0xffff_ffff))
}

/// A partial `membermask` names the threads that execute the instruction; SPIR-V
/// subgroup operations act on the active invocations, which are exactly those,
/// so an immediate partial mask is replaced by the full mask. Out-of-range
/// bfly segments need the register-control form.
fn shfl_masks(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    if parts.get(1) != Some(&"sync") || inst.operands.len() != 5 {
        return None;
    }
    let mask = inst.operand(4);
    let control = inst.operand(3);
    let partial = super::immediate(mask).is_some() && !full_mask(mask);
    let bfly_segment = parts.get(2) == Some(&"bfly") && super::immediate(control).is_some_and(|c| c != 0x1f);
    if !partial && !bfly_segment {
        return None;
    }
    let mut copy = with_op(inst, inst.op.clone());
    if partial {
        copy.operands[4] = "0xffffffff".into();
    }
    let mut lines = Vec::new();
    if bfly_segment {
        let st = &mut ctx.t3;
        st.x32 = st.x32.max(1);
        lines.push(format!("mov.b32 %__x32_0 , {control} ;"));
        copy.operands[3] = "%__x32_0".into();
    }
    lines.extend(again(ctx, &copy));
    Some(lines)
}

/// `vote.sync.ballot` returns only the bits of the named threads.
fn vote_masks(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    if parts.get(1) != Some(&"sync") || parts.get(2) != Some(&"ballot") || inst.operands.len() != 3 {
        return None;
    }
    let mask = inst.operand(2);
    if full_mask(mask) {
        return None;
    }
    let dst = inst.operand(0).to_owned();
    let mut copy = with_op(inst, inst.op.clone());
    copy.guard = String::new();
    copy.operands[2] = "0xffffffff".into();
    let mut lines = again(ctx, &copy);
    lines.push(render(&inst.guard, "and.b32", &[dst.clone(), dst, mask.to_owned()]));
    Some(lines)
}

/// `cvt.pack.sat` and same-width integer reinterpretations.
fn cvt_misc(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    if parts.get(1) == Some(&"pack") && parts.get(2) == Some(&"sat") && parts.len() == 6 && inst.operands.len() == 4 {
        // cvt.pack.sat.{u8,s8,u16,s16}.{s32}.b32 d, a, b, c: d = (c << 2w) | (sat(a) << w) | sat(b)
        let (w, lo, hi): (u32, i64, i64) = match parts[3] {
            "u8" => (8, 0, 255),
            "s8" => (8, -128, 127),
            "u16" => (16, 0, 65535),
            "s16" => (16, -32768, 32767),
            _ => return None,
        };
        if parts[4] != "s32" || parts[5] != "b32" {
            return None;
        }
        let mut b = B::new(&mut ctx.t3);
        let dst = inst.operand(0);
        let (a, bb, c) = (inst.operand(1), inst.operand(2), inst.operand(3));
        let (sa, sb) = (b.t32(), b.t32());
        for (src, out) in [(a, &sa), (bb, &sb)] {
            let t = b.t32();
            b.p(format!("max.s32 {t} , {src} , {lo} ;"));
            b.p(format!("min.s32 {out} , {t} , {hi} ;"));
        }
        let acc = b.t32();
        if w == 8 {
            b.p(format!("shl.b32 {acc} , {c} , 16 ;"));
        } else {
            b.p(format!("mov.b32 {acc} , 0 ;"));
        }
        b.p(format!("bfi.b32 {acc} , {sb} , {acc} , 0 , {w} ;"));
        b.p(format!("bfi.b32 {acc} , {sa} , {acc} , {w} , {w} ;"));
        b.p(render(&inst.guard, "mov.b32", &[dst.to_owned(), acc]));
        return Some(b.finish());
    }
    if parts.len() == 3 && matches!(parts[1], "s32" | "u32") && matches!(parts[2], "s32" | "u32") && inst.operands.len() == 2 {
        return Some(vec![render(&inst.guard, "mov.b32", &inst.operands)]);
    }
    None
}

// ---- carry chains -------------------------------------------------------

fn width_of(ty: &str) -> Option<u32> {
    match ty {
        "u32" | "s32" | "b32" => Some(32),
        "u64" | "s64" | "b64" => Some(64),
        _ => None,
    }
}

fn carry(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let base = parts[0];
    let has_cc = parts.contains(&"cc");
    let bits = width_of(parts.last()?)?;
    let ty = *parts.last()?;
    let ut = format!("u{bits}");
    let (carry_in, carry_out, sub, mul) = match base {
        "add" if has_cc => (false, true, false, None),
        "sub" if has_cc => (false, true, true, None),
        "addc" => (true, has_cc, false, None),
        "subc" => (true, has_cc, true, None),
        "madc" => (true, has_cc, false, parts.iter().copied().find(|p| matches!(*p, "lo" | "hi"))),
        "mad" if has_cc => (false, true, false, parts.iter().copied().find(|p| matches!(*p, "lo" | "hi"))),
        _ => return None,
    };
    if mul.is_none() && matches!(base, "madc" | "mad") {
        return None;
    }
    let want = if mul.is_some() { 4 } else { 3 };
    if inst.operands.len() != want {
        return None;
    }
    st.cc = true;
    let mut b = B::new(st);
    let dst = inst.operand(0).to_owned();
    let inst = &{
        let mut copy = with_op(inst, inst.op.clone());
        if bits == 32 {
            for o in copy.operands.iter_mut().skip(1) {
                if let Some(v) = super::immediate(o) {
                    *o = (v as u32).to_string();
                }
            }
        }
        copy
    };
    let (a, bb) = match mul {
        Some(kind) => {
            let m = b.tn(bits);
            let (x, y) = (inst.operand(1), inst.operand(2));
            let x = b.reg(x, bits);
            b.p(format!("mul.{kind}.{ty} {m} , {x} , {y} ;"));
            (m, inst.operand(3).to_owned())
        }
        None => (inst.operand(1).to_owned(), inst.operand(2).to_owned()),
    };
    let a = if sub { b.reg(&a, bits) } else { a };
    let x0 = b.tn(bits);
    b.p(format!("{}.{ut} {x0} , {a} , {bb} ;", if sub { "sub" } else { "add" }));
    let mut result = x0.clone();
    let mut cin = String::new();
    if carry_in {
        cin = if bits == 64 {
            let c = b.t64();
            b.p(format!("cvt.u64.u32 {c} , %__cc ;"));
            c
        } else {
            "%__cc".to_owned()
        };
        let x1 = b.tn(bits);
        b.p(format!("{}.{ut} {x1} , {x0} , {cin} ;", if sub { "sub" } else { "add" }));
        result = x1;
    }
    if carry_out {
        let p0 = b.tp();
        if sub {
            b.p(format!("setp.lt.{ut} {p0} , {a} , {bb} ;"));
        } else {
            b.p(format!("setp.lt.{ut} {p0} , {x0} , {a} ;"));
        }
        if carry_in {
            let p1 = b.tp();
            if sub {
                b.p(format!("setp.lt.{ut} {p1} , {x0} , {cin} ;"));
            } else {
                b.p(format!("setp.lt.{ut} {p1} , {result} , {x0} ;"));
            }
            b.p(format!("or.pred {p0} , {p0} , {p1} ;"));
        }
        b.p(render(&inst.guard, "selp.b32", &["%__cc".into(), "1".into(), "0".into(), p0]));
    }
    b.p(render(&inst.guard, &format!("mov.b{bits}"), &[dst, result]));
    Some(b.finish())
}

// ---- video instructions ---------------------------------------------------

/// A source operand with an optional byte/half selector (`%r1.b2`, `%r1 .h0`).
struct Sel {
    reg: String,
    /// (position, length)
    field: Option<(u32, u32)>,
}

fn parse_sel(text: &str) -> Sel {
    let compact: String = text.split_whitespace().collect();
    for (suffix, pos, len) in [
        (".b0", 0, 8), (".b1", 8, 8), (".b2", 16, 8), (".b3", 24, 8), (".h0", 0, 16), (".h1", 16, 16),
    ] {
        if let Some(reg) = compact.strip_suffix(suffix) {
            return Sel { reg: reg.to_owned(), field: Some((pos, len)) };
        }
    }
    Sel { reg: compact, field: None }
}

/// Emits `dst = extract(sel)` into a fresh 32-bit register; returns it.
fn extract(b: &mut B, sel: &Sel, signed: bool) -> String {
    let t = b.t32();
    match sel.field {
        Some((pos, len)) => {
            b.p(format!("bfe.{}32 {t} , {} , {pos} , {len} ;", if signed { "s" } else { "u" }, sel.reg))
        }
        None => b.p(format!("mov.b32 {t} , {} ;", sel.reg)),
    }
    t
}

/// Widens a 32-bit value to a signed 64-bit scratch.
fn widen(b: &mut B, v: &str, signed: bool) -> String {
    let t = b.t64();
    b.p(format!("cvt.{}64.{}32 {t} , {v} ;", if signed { "s" } else { "u" }, if signed { "s" } else { "u" }));
    t
}

fn clamp_s64(b: &mut B, v: &str, lo: i64, hi: i64) -> String {
    let t = b.t64();
    b.p(format!("max.s64 {t} , {v} , {lo} ;"));
    let u = b.t64();
    b.p(format!("min.s64 {u} , {t} , {hi} ;"));
    u
}

/// Combines the primary result with `c` through the secondary operation.
fn secondary(b: &mut B, r: &str, c: &str, op2: &str, c_signed: bool) -> String {
    let c32 = b.reg(c, 32);
    let cw = widen(b, &c32, c_signed);
    let t = b.t64();
    match op2 {
        "add" => b.p(format!("add.s64 {t} , {r} , {cw} ;")),
        "min" => b.p(format!("min.s64 {t} , {r} , {cw} ;")),
        _ => b.p(format!("max.s64 {t} , {r} , {cw} ;")),
    }
    t
}

fn cmp_name(c: &str) -> Option<&'static str> {
    Some(match c {
        "eq" => "eq",
        "ne" => "ne",
        "lt" => "lt",
        "le" => "le",
        "gt" => "gt",
        "ge" => "ge",
        _ => return None,
    })
}

/// `vadd.dtype.atype.btype[.sat][.op2] d, a[.sel], b[.sel], c;` and friends.
fn video_scalar(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let name = parts[0];
    let mut rest = &parts[1..];
    let mut cmp = None;
    if matches!(name, "vmad") {
        return None;
    }
    let is_ty = |s: &str| matches!(s, "u32" | "s32");
    let vset_fixed;
    if name == "vset" {
        // vset.atype.btype.cmp: no destination type.
        if rest.len() != 3 || !is_ty(rest[0]) || !is_ty(rest[1]) {
            return None;
        }
        cmp = Some(cmp_name(rest[2])?);
        vset_fixed = vec!["u32", rest[0], rest[1]];
        rest = &vset_fixed;
    }
    if rest.len() < 3 || !rest[..3].iter().all(|t| is_ty(t)) {
        return None;
    }
    let (dt, at, bt) = (rest[0], rest[1], rest[2]);
    let mut sat = false;
    let mut clamp = false;
    let mut op2 = None;
    for m in &rest[3..] {
        match *m {
            "sat" => sat = true,
            "clamp" => clamp = true,
            "wrap" => {}
            "add" | "min" | "max" => op2 = Some(*m),
            _ => return None,
        }
    }
    let nops = if name == "vset" || op2.is_some() || name == "vshl" || name == "vshr" { 4 } else { 4 };
    if inst.operands.len() != nops && inst.operands.len() != 3 {
        return None;
    }
    let mut b = B::new(st);
    let a_sel = parse_sel(inst.operand(1));
    let b_sel = parse_sel(inst.operand(2));
    let a32 = extract(&mut b, &a_sel, at == "s32");
    let b32 = extract(&mut b, &b_sel, bt == "s32");
    let aw = widen(&mut b, &a32, at == "s32");
    let r = match name {
        "vshl" | "vshr" => {
            // The shift amount is the unsigned `b`; `.clamp` saturates it at 32, otherwise it wraps.
            let amount = b.t32();
            if clamp {
                b.p(format!("min.u32 {amount} , {b32} , 32 ;"));
            } else {
                b.p(format!("and.b32 {amount} , {b32} , 31 ;"));
            }
            let r = b.t64();
            let w = widen(&mut b, &a32, at == "s32");
            let amount64 = b.t64();
            b.p(format!("cvt.u64.u32 {amount64} , {amount} ;"));
            if name == "vshl" {
                b.p(format!("shl.b64 {r} , {w} , {amount64} ;"));
            } else if at == "s32" {
                b.p(format!("shr.s64 {r} , {w} , {amount64} ;"));
            } else {
                b.p(format!("shr.u64 {r} , {w} , {amount64} ;"));
            }
            r
        }
        _ => {
            let bw = widen(&mut b, &b32, bt == "s32");
            let r = b.t64();
            match name {
                "vadd" => b.p(format!("add.s64 {r} , {aw} , {bw} ;")),
                "vsub" => b.p(format!("sub.s64 {r} , {aw} , {bw} ;")),
                "vabsdiff" => {
                    let d = b.t64();
                    b.p(format!("sub.s64 {d} , {aw} , {bw} ;"));
                    b.p(format!("abs.s64 {r} , {d} ;"));
                }
                "vmin" => b.p(format!("min.s64 {r} , {aw} , {bw} ;")),
                "vmax" => b.p(format!("max.s64 {r} , {aw} , {bw} ;")),
                "vavrg" => {
                    let s = b.t64();
                    b.p(format!("add.s64 {s} , {aw} , {bw} ;"));
                    let s1 = b.t64();
                    b.p(format!("add.s64 {s1} , {s} , 1 ;"));
                    b.p(format!("shr.s64 {r} , {s1} , 1 ;"));
                }
                "vset" => {
                    let p = b.tp();
                    b.p(format!("setp.{}.s64 {p} , {aw} , {bw} ;", cmp?));
                    let one = b.t32();
                    b.p(format!("selp.b32 {one} , 1 , 0 , {p} ;"));
                    b.p(format!("cvt.u64.u32 {r} , {one} ;"));
                }
                _ => return None,
            }
            r
        }
    };
    let r = if sat && name != "vset" {
        let (lo, hi) = if dt == "s32" { (i64::from(i32::MIN), i64::from(i32::MAX)) } else { (0, i64::from(u32::MAX)) };
        clamp_s64(&mut b, &r, lo, hi)
    } else {
        r
    };
    let r = match op2 {
        Some(op2) if inst.operands.len() == 4 => secondary(&mut b, &r, inst.operand(3), op2, dt == "s32"),
        _ => r,
    };
    let out = b.t32();
    b.p(format!("cvt.u32.u64 {out} , {r} ;"));
    b.p(render(&inst.guard, "mov.b32", &[inst.operand(0).to_owned(), out]));
    Some(b.finish())
}

/// `vadd2/vadd4/...` SIMD-in-a-register forms.
fn video_simd(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let full = parts[0];
    let lanes: u32 = if full.ends_with('2') { 2 } else { 4 };
    let name = &full[..full.len() - 1];
    let width = 32 / lanes;
    let mut rest = &parts[1..];
    let mut cmp = None;
    let is_ty = |s: &str| matches!(s, "u32" | "s32");
    let vset_fixed;
    if name == "vset" {
        if rest.len() != 3 || !is_ty(rest[0]) || !is_ty(rest[1]) {
            return None;
        }
        cmp = Some(cmp_name(rest[2])?);
        vset_fixed = vec!["u32", rest[0], rest[1]];
        rest = &vset_fixed;
    }
    if rest.len() < 3 || !rest[..3].iter().all(|t| is_ty(t)) {
        return None;
    }
    let (dt, at, bt) = (rest[0], rest[1], rest[2]);
    let mut sat = false;
    let mut op2 = None;
    for m in &rest[3..] {
        match *m {
            "sat" => sat = true,
            "add" => op2 = Some("add"),
            _ => return None,
        }
    }
    if inst.operands.len() != 4 {
        return None;
    }
    let mut b = B::new(st);
    let (a, bb) = (inst.operand(1).to_owned(), inst.operand(2).to_owned());
    let a = b.reg(&a, 32);
    let bb = b.reg(&bb, 32);
    let out = b.t32();
    b.p(format!("mov.b32 {out} , 0 ;"));
    let sum = b.t32();
    b.p(format!("mov.b32 {sum} , 0 ;"));
    let (lo, hi) = match (dt == "s32", width) {
        (true, 8) => (-128i64, 127i64),
        (true, _) => (-32768, 32767),
        (false, 8) => (0, 255),
        (false, _) => (0, 65535),
    };
    for lane in 0..lanes {
        let pos = lane * width;
        let x = b.t32();
        let y = b.t32();
        b.p(format!("bfe.{}32 {x} , {a} , {pos} , {width} ;", if at == "s32" { "s" } else { "u" }));
        b.p(format!("bfe.{}32 {y} , {bb} , {pos} , {width} ;", if bt == "s32" { "s" } else { "u" }));
        let r = b.t32();
        match name {
            "vadd" => b.p(format!("add.s32 {r} , {x} , {y} ;")),
            "vsub" => b.p(format!("sub.s32 {r} , {x} , {y} ;")),
            "vabsdiff" => {
                let d = b.t32();
                b.p(format!("sub.s32 {d} , {x} , {y} ;"));
                b.p(format!("abs.s32 {r} , {d} ;"));
            }
            "vmin" => b.p(format!("min.s32 {r} , {x} , {y} ;")),
            "vmax" => b.p(format!("max.s32 {r} , {x} , {y} ;")),
            "vavrg" => {
                let s = b.t32();
                b.p(format!("add.s32 {s} , {x} , {y} ;"));
                let s1 = b.t32();
                b.p(format!("add.s32 {s1} , {s} , 1 ;"));
                b.p(format!("shr.s32 {r} , {s1} , 1 ;"));
            }
            "vset" => {
                let p = b.tp();
                b.p(format!("setp.{}.s32 {p} , {x} , {y} ;", cmp?));
                b.p(format!("selp.b32 {r} , 1 , 0 , {p} ;"));
            }
            _ => return None,
        }
        let r = if sat && name != "vset" {
            let t = b.t32();
            b.p(format!("max.s32 {t} , {r} , {lo} ;"));
            let u = b.t32();
            b.p(format!("min.s32 {u} , {t} , {hi} ;"));
            u
        } else {
            r
        };
        let s2 = b.t32();
        b.p(format!("add.s32 {s2} , {sum} , {r} ;"));
        b.p(format!("mov.b32 {sum} , {s2} ;"));
        b.p(format!("bfi.b32 {out} , {r} , {out} , {pos} , {width} ;"));
    }
    let result = if op2.is_some() {
        let c = b.reg(&inst.operand(3).to_owned(), 32);
        let t = b.t32();
        b.p(format!("add.s32 {t} , {sum} , {c} ;"));
        t
    } else {
        out
    };
    b.p(render(&inst.guard, "mov.b32", &[inst.operand(0).to_owned(), result]));
    Some(b.finish())
}

// ---- small predicate forms ----------------------------------------------

/// `setp.lt.u32 p|q, a, b` writes `p = cmp` and `q = !cmp`.
fn setp_pair(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    if !inst.op.starts_with("setp.") {
        return None;
    }
    let dst = inst.operand(0);
    if !dst.contains('|') {
        // `setp.cmp.and.T p, a, b, !c`: the negated combine predicate needs its own register.
        let c = inst.operands.get(3)?;
        let name = c.strip_prefix('!')?.trim().to_owned();
        let mut b = B::new(st);
        let t = b.tp();
        b.p(format!("not.pred {t} , {name} ;"));
        let mut operands = inst.operands.clone();
        operands[3] = t;
        b.p(render(&inst.guard, &inst.op, &operands));
        return Some(b.finish());
    }
    let (p, q) = dst.split_once('|')?;
    let (p, q) = (p.trim(), q.trim());
    let mut b = B::new(st);
    let mut operands = inst.operands.clone();
    operands[0] = p.to_owned();
    // With a boolean combine `q = !cmp && c` has no single-instruction form; the
    // negation of the first result is only correct without a combine.
    let combined = inst.op.split('.').any(|part| matches!(part, "and" | "or" | "xor"));
    if combined {
        let tmp = b.tp();
        let mut plain = inst.op.split('.').filter(|part| !matches!(*part, "and" | "or" | "xor")).collect::<Vec<_>>().join(".");
        plain.push_str("");
        let compare_operands = vec![tmp.clone(), inst.operand(1).to_owned(), inst.operand(2).to_owned()];
        b.p(render("", &plain, &compare_operands));
        let comb = inst.op.split('.').find(|part| matches!(*part, "and" | "or" | "xor"))?;
        let c = inst.operand(3);
        let (c_src, c_neg) = match c.strip_prefix('!') {
            Some(n) => (n.trim().to_owned(), true),
            None => (c.to_owned(), false),
        };
        let c_reg = if c_neg {
            let t = b.tp();
            b.p(format!("not.pred {t} , {c_src} ;"));
            t
        } else {
            c_src
        };
        let notcmp = b.tp();
        b.p(format!("not.pred {notcmp} , {tmp} ;"));
        for (target, value) in [(p, &tmp), (q, &notcmp)] {
            match comb {
                "and" => b.p(format!("and.pred {target} , {value} , {c_reg} ;")),
                "or" => b.p(format!("or.pred {target} , {value} , {c_reg} ;")),
                _ => {
                    let (x, y) = (b.t32(), b.t32());
                    b.p(format!("selp.b32 {x} , 1 , 0 , {value} ;"));
                    b.p(format!("selp.b32 {y} , 1 , 0 , {c_reg} ;"));
                    b.p(format!("xor.b32 {x} , {x} , {y} ;"));
                    b.p(format!("setp.ne.u32 {target} , {x} , 0 ;"));
                }
            }
        }
        return Some(b.finish());
    }
    b.p(render(&inst.guard, &inst.op, &operands));
    b.p(render(&inst.guard, "not.pred", &[q.to_owned(), p.to_owned()]));
    Some(b.finish())
}

/// `selp.T d, a, b, 0|1` with a literal predicate becomes a move.
fn selp_literal(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    if inst.operands.len() != 4 {
        return None;
    }
    let pick = match inst.operand(3) {
        "1" => 1,
        "0" => 0,
        _ => return None,
    };
    let parts: Vec<&str> = inst.op.split('.').collect();
    let ty = *parts.get(1)?;
    let bits = match ty {
        "u64" | "s64" | "b64" | "f64" => 64,
        "u32" | "s32" | "b32" | "f32" => 32,
        _ => 32,
    };
    let _ = st;
    let src = if pick == 1 { inst.operand(1) } else { inst.operand(2) };
    Some(vec![render(&inst.guard, &format!("mov.b{bits}"), &[inst.operand(0).to_owned(), src.to_owned()])])
}

// ---- barriers ------------------------------------------------------------

/// Named barriers with thread counts, `bar.arrive` and `bar.red.*`.
///
/// Counted barriers are emulated with shared-memory arrival counters and a
/// generation word per barrier id (polled with atomics so the loop cannot be
/// hoisted). Plain full-CTA barriers stay with the legacy rewrite.
fn barrier(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').filter(|p| *p != "cta" && *p != "aligned").collect();
    let kind = *parts.get(1)?;
    match kind {
        "warp" => return Some(vec![render(&inst.guard, "membar.gl", &[])]),
        "sync" => {
            let count = match inst.operands.len() {
                1 => return None,
                2 => inst.operand(1),
                _ => return None,
            };
            if super::immediate(count).is_some_and(|c| Some(c as u32) == ctx.block_threads) {
                return None;
            }
            Some(counted(ctx, inst, inst.operand(0), count, true))
        }
        "arrive" if inst.operands.len() == 2 => Some(counted(ctx, inst, inst.operand(0), inst.operand(1), false)),
        "red" => reduction(ctx, inst, &parts),
        _ => None,
    }
}

fn counted(ctx: &mut Ctx, inst: &Inst, id: &str, count: &str, wait: bool) -> Vec<String> {
    ctx.t3.named_barriers = true;
    ctx.t3.labels += 1;
    let label = format!("$L__t3_wait{}", ctx.t3.labels);
    let base = match super::immediate(id) {
        Some(n) if n < 16 => 8 * n,
        _ => 0,
    };
    // Register ids select the slot through a computed shared address.
    let mut b = B::new(&mut ctx.t3);
    let (cnt, gword) = if super::immediate(id).is_some() {
        (format!("[%__bs+{base}]"), format!("[%__bs+{}]", base + 4))
    } else {
        let a = b.t32();
        b.p(format!("and.b32 {a} , {id} , 15 ;"));
        b.p(format!("shl.b32 {a} , {a} , 3 ;"));
        b.p(format!("add.u32 {a} , {a} , %__bs ;"));
        (format!("[{a}]"), format!("[{a}+4]"))
    };
    let (g, old, junk, last, pw, nm1, c) = (b.t32(), b.t32(), b.t32(), b.tp(), b.tp(), b.t32(), b.reg(count, 32));
    b.p("membar.gl ;");
    if wait {
        b.p(format!("atom.shared.add.u32 {g} , {gword} , %__bz ;"));
    }
    b.p(format!("atom.shared.add.u32 {old} , {cnt} , %__bone ;"));
    b.p(format!("sub.u32 {nm1} , {c} , 1 ;"));
    b.p(format!("setp.eq.u32 {last} , {old} , {nm1} ;"));
    b.p(format!("@ {last} atom.shared.exch.b32 {junk} , {cnt} , %__bz ;"));
    b.p(format!("@ {last} atom.shared.add.u32 {junk} , {gword} , %__bone ;"));
    if wait {
        b.p(format!("{label} :"));
        b.p(format!("atom.shared.add.u32 {old} , {gword} , %__bz ;"));
        b.p(format!("setp.eq.u32 {pw} , {old} , {g} ;"));
        b.p(format!("@ {pw} bra {label} ;"));
        b.p("membar.gl ;");
    }
    let _ = inst;
    b.finish()
}

/// `bar.red.{popc,and,or}` over the whole CTA through one shared counter.
fn reduction(ctx: &mut Ctx, inst: &Inst, parts: &[&str]) -> Option<Vec<String>> {
    let op = *parts.get(2)?;
    if !matches!(op, "popc" | "and" | "or") || inst.operands.len() != 3 {
        return None;
    }
    // An id other than 0 is fine for a full barrier; a count is not supported here.
    let pred = inst.operand(2);
    let dst = inst.operand(0);
    ctx.t3.reductions = true;
    let word = "[%__bs+128]";
    let mut b = B::new(&mut ctx.t3);
    let (neg, pname) = match pred.strip_prefix('!') {
        Some(n) => (true, n.trim().to_owned()),
        None => (false, pred.to_owned()),
    };
    let (bit, junk, total, pz) = (b.t32(), b.t32(), b.t32(), b.tp());
    // Leader zeroes the word between two barriers so back-to-back reductions cannot race.
    b.p("bar.sync 0 ;");
    for (reg, name) in [("%tid.x", "x"), ("%tid.y", "y"), ("%tid.z", "z")] {
        let t = b.t32();
        b.p(format!("mov.u32 {t} , {reg} ;"));
        let _ = name;
        b.p(format!("setp.eq.u32 {pz} , {t} , 0 ;"));
        if name == "x" {
            b.p(format!("selp.b32 {total} , 1 , 0 , {pz} ;"));
        } else {
            b.p(format!("selp.b32 {t} , 1 , 0 , {pz} ;"));
            b.p(format!("and.b32 {total} , {total} , {t} ;"));
        }
    }
    b.p(format!("setp.ne.u32 {pz} , {total} , 0 ;"));
    b.p(format!("@ {pz} st.shared.u32 {word} , %__bz ;"));
    b.p("bar.sync 0 ;");
    // and counts the false predicates; or and popc count the true ones.
    let count_true = op != "and" && !neg || op == "and" && neg;
    if count_true {
        b.p(format!("selp.b32 {bit} , 1 , 0 , {pname} ;"));
    } else {
        b.p(format!("selp.b32 {bit} , 0 , 1 , {pname} ;"));
    }
    b.p(format!("atom.shared.add.u32 {junk} , {word} , {bit} ;"));
    b.p("bar.sync 0 ;");
    b.p(format!("ld.shared.u32 {total} , {word} ;"));
    match op {
        "popc" => b.p(render(&inst.guard, "mov.b32", &[dst.to_owned(), total])),
        "or" => b.p(render(&inst.guard, "setp.ne.u32", &[dst.to_owned(), total, "0".into()])),
        _ => b.p(render(&inst.guard, "setp.eq.u32", &[dst.to_owned(), total, "0".into()])),
    }
    Some(b.finish())
}

// ---- local / shared declarations ------------------------------------------------

/// `.local|.shared [.align N] .TYPE name[count] ;` -> (space, align, bytes, name)
fn parse_decl(stmt: &[String]) -> Option<(String, u32, u32, String)> {
    let space = stmt.first()?.clone();
    if space != ".local" && space != ".shared" {
        return None;
    }
    let mut i = 1;
    let mut align = 0u32;
    if stmt.get(i).is_some_and(|t| t == ".align") {
        align = stmt.get(i + 1)?.parse().ok()?;
        i += 2;
    }
    let elem: u32 = match stmt.get(i)?.as_str() {
        ".b8" | ".u8" | ".s8" => 1,
        ".b16" | ".u16" | ".s16" | ".f16" | ".bf16" => 2,
        ".b32" | ".u32" | ".s32" | ".f32" | ".f16x2" => 4,
        ".b64" | ".u64" | ".s64" | ".f64" => 8,
        _ => return None,
    };
    let name = stmt.get(i + 1)?.clone();
    let count: u32 = if stmt.get(i + 2).is_some_and(|t| t == "[") {
        stmt.get(i + 3)?.parse().ok()?
    } else {
        1
    };
    Some((space, align.max(elem), count.checked_mul(elem)?, name))
}

/// Entry-level pre-pass: the `.local` arrays of every entry, in order, so uses
/// can be rebased onto one merged array.
pub(super) fn entry_locals(statements: &[super::Stmt]) -> Vec<Vec<(String, u32, u32)>> {
    let mut out = Vec::new();
    let mut current: Option<Vec<(String, u32, u32)>> = None;
    let mut depth = 0i32;
    for stmt in statements {
        if super::is_entry_header(stmt) {
            current = Some(Vec::new());
            depth = 1;
            continue;
        }
        match stmt.first().map(String::as_str) {
            Some("{") => depth += 1,
            Some("}") => {
                depth -= 1;
                if depth == 0 {
                    if let Some(c) = current.take() {
                        out.push(c);
                    }
                }
            }
            _ => {}
        }
        if let (Some(c), Some((space, align, bytes, name))) = (current.as_mut(), parse_decl(stmt)) {
            if space == ".local" {
                c.push((name, align, bytes));
            }
        }
    }
    out
}

/// Starts an entry: lays the entry's `.local` arrays out in one merged array.
pub(super) fn begin_entry(st: &mut State, locals: Vec<(String, u32, u32)>, dynamic: Option<&str>) {
    st.locals.clear();
    st.sink_count = 0;
    st.shared.clear();
    st.local_bytes = 0;
    let mut offset = 0u32;
    let mut align = 4u32;
    for (name, a, bytes) in locals {
        offset = offset.next_multiple_of(a.max(1));
        st.locals.push((name, offset));
        offset += bytes;
        align = align.max(a);
    }
    if let Some(name) = dynamic {
        st.shared.push(name.to_owned());
    }
    st.local_bytes = offset.next_multiple_of(4);
    st.local_align = if align >= 16 { 16 } else if align >= 8 { 8 } else { 4 };
}

/// Rewrites `.local` / `.shared` declarations to the canonical byte-array form.
pub(super) fn decl(ctx: &mut Ctx, stmt: &[String]) -> Option<Vec<String>> {
    let (space, align, bytes, name) = parse_decl(stmt)?;
    if space == ".local" {
        // The merged array is emitted with the entry's other declarations.
        let _ = (align, bytes, name);
        return Some(Vec::new());
    }
    ctx.t3.shared.push(name.clone());
    // The kernel parser honours power-of-two alignments up to 4096 (verified on hardware).
    let align = if align.is_power_of_two() && align <= 4096 { align.max(4) } else if align >= 16 { 16 } else if align >= 8 { 8 } else { 4 };
    Some(vec![format!(".shared .align {align} .b8 {name} [ {} ] ;", bytes.next_multiple_of(4))])
}

/// Splits `[sym+off]` / `[sym]` into (sym, extra offset text).
fn bracket_symbol(operand: &str) -> Option<(String, i64)> {
    let inner = operand.trim().strip_prefix('[')?.strip_suffix(']')?.replace(' ', "");
    if inner.starts_with('%') {
        return None;
    }
    let (base, off) = match inner.find(['+', '-']) {
        Some(at) => (&inner[..at], inner[at..].replace('+', "").parse::<i64>().ok()?),
        None => (inner.as_str(), 0),
    };
    Some((base.to_owned(), off))
}

/// Memory operands and moves that name a declared `.local`/`.shared` array are
/// rewritten to the register form (`mov` of the symbol, then `[reg+offset]`).
fn symbol_operands(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let family = inst.op.split('.').next()?;
    if ctx.t3.locals.is_empty() && ctx.t3.shared.is_empty() && ctx.t3.module_globals.is_empty() {
        return None;
    }
    let mut lines = Vec::new();
    let mut operands = inst.operands.clone();
    let mut changed = false;
    if family == "mov" && inst.operands.len() == 2 {
        let src = inst.operand(1).replace(' ', "");
        let (base, off) = match src.find('+') {
            Some(at) => (src[..at].to_owned(), src[at + 1..].parse::<i64>().ok()?),
            None => (src.clone(), 0),
        };
        if let Some((_, o)) = ctx.t3.locals.iter().find(|(n, _)| *n == base) {
            let o = i64::from(*o) + off;
            let dst = inst.operand(0).to_owned();
            lines.push(render(&inst.guard, "mov.u64", &[dst.clone(), "__t3_local".into()]));
            if o != 0 {
                lines.push(render(&inst.guard, "add.u64", &[dst.clone(), dst, o.to_string()]));
            }
            return Some(lines);
        }
        if ctx.t3.module_globals.contains(&base) && (off != 0 || inst.op != "mov.u64") {
            let dst = inst.operand(0).to_owned();
            let mut lines = vec![render(&inst.guard, "mov.u64", &[dst.clone(), base])];
            if off != 0 {
                lines.push(render(&inst.guard, "add.u64", &[dst.clone(), dst, off.to_string()]));
            }
            return Some(lines);
        }
        return None;
    }
    if !matches!(family, "ld" | "ldu" | "st" | "atom" | "red") {
        return None;
    }
    for operand in &mut operands {
        let Some((base, off)) = bracket_symbol(operand) else { continue };
        if let Some((_, o)) = ctx.t3.locals.iter().find(|(n, _)| *n == base) {
            ctx.t3.uses_lsym = true;
            lines.push(format!("mov.u64 %__lsym , __t3_local ;"));
            *operand = format!("[%__lsym+{}]", i64::from(*o) + off);
            changed = true;
        } else if ctx.t3.shared.iter().any(|n| *n == base)
            && (off != 0 || inst.op.split('.').any(|p| matches!(p, "u64" | "s64" | "b64" | "f64")))
        {
            ctx.t3.uses_ssym = true;
            lines.push(format!("mov.u32 %__ssym , {base} ;"));
            *operand = format!("[%__ssym+{off}]");
            changed = true;
        }
    }
    if !changed {
        return None;
    }
    let copy = Inst { guard: inst.guard.clone(), op: inst.op.clone(), operands };
    lines.extend(again(ctx, &copy));
    Some(lines)
}

// ---- source-level clean-up ----------------------------------------------------

/// Drops debug-only directives (`.file`, `.loc`, `.section` blocks) that carry
/// string literals or data the tokenizer has no use for.
pub(super) fn strip_debug(source: &str) -> std::borrow::Cow<'_, str> {
    if !source.contains(".file") && !source.contains(".loc") && !source.contains(".section") {
        return std::borrow::Cow::Borrowed(source);
    }
    let mut out = String::with_capacity(source.len());
    let mut skipping: Option<i32> = None;
    for line in source.lines() {
        let code = line.split("//").next().unwrap_or("");
        let trimmed = code.trim_start();
        if let Some(depth) = skipping.as_mut() {
            *depth += code.matches('{').count() as i32 - code.matches('}').count() as i32;
            if *depth <= 0 {
                skipping = None;
            }
            continue;
        }
        if trimmed.starts_with(".file") || trimmed.starts_with(".loc") && !trimmed.starts_with(".local") {
            continue;
        }
        if trimmed.starts_with(".section") {
            let depth = code.matches('{').count() as i32 - code.matches('}').count() as i32;
            if depth > 0 || !code.contains('{') && !code.contains(';') {
                skipping = Some(depth);
            }
            continue;
        }
        out.push_str(line);
        out.push('\n');
    }
    std::borrow::Cow::Owned(out)
}

/// Texture, surface and sampler instructions need bindless image objects, which
/// the OpenCL-environment SPIR-V target does not provide. Fail with a clear message.
pub(super) fn check_unsupported(statements: &[super::Stmt]) -> Result<(), super::Error> {
    for stmt in statements {
        let first = stmt.first().map_or("", String::as_str);
        let op = if first == "@" { stmt.get(2).or(stmt.get(3)).map_or("", String::as_str) } else { first };
        let family = op.split('.').next().unwrap_or("");
        if matches!(family, "tex" | "tld4" | "txq" | "suld" | "sust" | "sured" | "suq")
            || matches!(first, ".texref" | ".surfref" | ".samplerref")
            || stmt.iter().any(|t| matches!(t.as_str(), ".texref" | ".surfref" | ".samplerref"))
        {
            return Err(super::Error::new(format!(
                "unsupported PTX instruction: {} (texture and surface objects need bindless images, which the OpenCL-environment SPIR-V target does not expose)",
                stmt.join(" ")
            )));
        }
    }
    Ok(())
}

// ---- stmatrix --------------------------------------------------------------------

/// `stmatrix.sync.aligned.m8n8.xN[.trans].shared.b16 [addr], {r...}`: lane `8i+j` supplies the
/// address of row `j` of matrix `i`; each thread stores its 32-bit fragment to that row
/// (or, transposed, two 16-bit halves to two rows).
fn stmatrix(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    if parts.get(1) != Some(&"sync") || inst.operands.len() != 2 {
        return None;
    }
    let count: usize = parts.iter().find_map(|p| p.strip_prefix('x')?.parse().ok())?;
    let trans = parts.contains(&"trans");
    let regs: Vec<String> = inst.operand(1).trim().trim_start_matches('{').trim_end_matches('}').split(',').map(|r| r.trim().to_owned()).filter(|r| !r.is_empty()).collect();
    if regs.len() != count {
        return None;
    }
    let addr = inst.operand(0).trim().trim_start_matches('[').trim_end_matches(']').trim().to_owned();
    // Only 32-bit shared offsets can travel through a shuffle.
    let (base, off) = match addr.split_once('+') {
        Some((b, o)) => (b.trim().to_owned(), super::immediate(o)?),
        None => (addr.clone(), 0),
    };
    if ctx.width_of(&base) == Some(64) {
        return None;
    }
    let mut b = B::new(&mut ctx.t3);
    let ctl = b.t32();
    b.p(format!("mov.b32 {ctl} , 31 ;"));
    let (lane, row, tl, src, from, my) = (b.t32(), b.t32(), b.t32(), b.t32(), b.t32(), b.t32());
    b.p(format!("mov.u32 {lane} , %laneid ;"));
    b.p(format!("shr.u32 {row} , {lane} , 2 ;"));
    b.p(format!("and.b32 {tl} , {lane} , 3 ;"));
    b.p(format!("add.u32 {my} , {base} , {off} ;"));
    for (i, reg) in regs.iter().enumerate() {
        if !trans {
            b.p(format!("add.u32 {from} , {row} , {} ;", 8 * i));
            b.p(format!("shfl.sync.idx.b32 {src} , {my} , {from} , {ctl} , 0xffffffff ;"));
            let at = b.t32();
            b.p(format!("shl.b32 {at} , {tl} , 2 ;"));
            b.p(format!("add.u32 {at} , {src} , {at} ;"));
            b.p(render(&inst.guard, "st.shared.u32", &[format!("[{at}]"), reg.clone()]));
        } else {
            for j in 0..2 {
                // Column 2*tl+j of this thread's row becomes row 2*tl+j of the stored matrix.
                b.p(format!("shl.b32 {from} , {tl} , 1 ;"));
                b.p(format!("add.u32 {from} , {from} , {} ;", 8 * i + j));
                b.p(format!("shfl.sync.idx.b32 {src} , {my} , {from} , {ctl} , 0xffffffff ;"));
                let at = b.t32();
                b.p(format!("shl.b32 {at} , {row} , 1 ;"));
                b.p(format!("add.u32 {at} , {src} , {at} ;"));
                let value = b.t32();
                if j == 0 {
                    b.p(format!("mov.b32 {value} , {reg} ;"));
                } else {
                    b.p(format!("shr.u32 {value} , {reg} , 16 ;"));
                }
                b.p(render(&inst.guard, "st.shared.u16", &[format!("[{at}]"), value]));
            }
        }
    }
    Some(b.finish())
}

/// Generic-space atomics are global; immediates in value positions move into registers.
fn atomic_forms(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let has_space = parts.iter().any(|p| matches!(*p, "global" | "shared") || p.starts_with("shared::") || p.starts_with("global::"));
    if has_space {
        return None;
    }
    let mut op = parts.clone();
    op.insert(1, "global");
    let copy = with_op(inst, op.join("."));
    Some(again(ctx, &copy))
}

// ---- f64 directed rounding ------------------------------------------------------

/// `add/sub/mul/fma/div/sqrt/rcp.{rz,rm,rp}.f64`: compute the round-to-nearest result and
/// the sign of its exact error (error-free transformations), then step one ulp when the
/// requested direction disagrees with the nearest result.
fn f64_directed(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let name = parts[0];
    let mode = *parts.iter().find(|p| matches!(**p, "rz" | "rm" | "rp"))?;
    let want = match name {
        "fma" => 4,
        "add" | "sub" | "mul" | "div" => 3,
        _ => 2,
    };
    if inst.operands.len() != want {
        return None;
    }
    let mut b = B::new(st);
    let (zero, one) = (b.t64(), b.t64());
    b.p(format!("mov.b64 {zero} , 0 ;"));
    b.p(format!("mov.b64 {one} , 0x3FF0000000000000 ;"));
    let a = inst.operand(1).to_owned();
    let neg = |b: &mut B, x: &str, zero: &str| {
        let n = b.t64();
        b.p(format!("sub.rn.f64 {n} , {zero} , {x} ;"));
        n
    };
    let (r, e_pos, e_neg);
    match name {
        "add" | "sub" => {
            let bb = if name == "sub" { neg(&mut b, inst.operand(2), &zero) } else { inst.operand(2).to_owned() };
            r = b.t64();
            let (t, u, v, w, e) = (b.t64(), b.t64(), b.t64(), b.t64(), b.t64());
            b.p(format!("add.rn.f64 {r} , {a} , {bb} ;"));
            b.p(format!("sub.rn.f64 {t} , {r} , {a} ;"));
            b.p(format!("sub.rn.f64 {u} , {r} , {t} ;"));
            b.p(format!("sub.rn.f64 {v} , {a} , {u} ;"));
            b.p(format!("sub.rn.f64 {w} , {bb} , {t} ;"));
            b.p(format!("add.rn.f64 {e} , {v} , {w} ;"));
            let (p, n) = (b.tp(), b.tp());
            b.p(format!("setp.gt.f64 {p} , {e} , {zero} ;"));
            b.p(format!("setp.lt.f64 {n} , {e} , {zero} ;"));
            e_pos = p;
            e_neg = n;
        }
        "mul" => {
            r = b.t64();
            let e = b.t64();
            b.p(format!("mul.rn.f64 {r} , {a} , {} ;", inst.operand(2)));
            let nr = neg(&mut b, &r, &zero);
            b.p(format!("fma.rn.f64 {e} , {a} , {} , {nr} ;", inst.operand(2)));
            let (p, n) = (b.tp(), b.tp());
            b.p(format!("setp.gt.f64 {p} , {e} , {zero} ;"));
            b.p(format!("setp.lt.f64 {n} , {e} , {zero} ;"));
            e_pos = p;
            e_neg = n;
        }
        "fma" => {
            let (bb, c) = (inst.operand(2).to_owned(), inst.operand(3).to_owned());
            r = b.t64();
            b.p(format!("fma.rn.f64 {r} , {a} , {bb} , {c} ;"));
            let (pr, pe, u, t, ue1, ue2, ue, w, d, e) =
                (b.t64(), b.t64(), b.t64(), b.t64(), b.t64(), b.t64(), b.t64(), b.t64(), b.t64(), b.t64());
            b.p(format!("mul.rn.f64 {pr} , {a} , {bb} ;"));
            let npr = neg(&mut b, &pr, &zero);
            b.p(format!("fma.rn.f64 {pe} , {a} , {bb} , {npr} ;"));
            b.p(format!("add.rn.f64 {u} , {pr} , {c} ;"));
            b.p(format!("sub.rn.f64 {t} , {u} , {pr} ;"));
            let ut = b.t64();
            b.p(format!("sub.rn.f64 {ut} , {u} , {t} ;"));
            b.p(format!("sub.rn.f64 {ue1} , {pr} , {ut} ;"));
            b.p(format!("sub.rn.f64 {ue2} , {c} , {t} ;"));
            b.p(format!("add.rn.f64 {ue} , {ue1} , {ue2} ;"));
            b.p(format!("add.rn.f64 {w} , {ue} , {pe} ;"));
            b.p(format!("sub.rn.f64 {d} , {u} , {r} ;"));
            b.p(format!("add.rn.f64 {e} , {d} , {w} ;"));
            let (p, n) = (b.tp(), b.tp());
            b.p(format!("setp.gt.f64 {p} , {e} , {zero} ;"));
            b.p(format!("setp.lt.f64 {n} , {e} , {zero} ;"));
            e_pos = p;
            e_neg = n;
        }
        "div" | "rcp" => {
            let (num, den) = if name == "rcp" { (one.clone(), a.clone()) } else { (a.clone(), inst.operand(2).to_owned()) };
            r = b.t64();
            let rem = b.t64();
            b.p(format!("div.rn.f64 {r} , {num} , {den} ;"));
            let nq = neg(&mut b, &r, &zero);
            b.p(format!("fma.rn.f64 {rem} , {nq} , {den} , {num} ;"));
            let (rp, rn, bn, nbn, x1, x2, y1, y2, p, n) =
                (b.tp(), b.tp(), b.tp(), b.tp(), b.tp(), b.tp(), b.tp(), b.tp(), b.tp(), b.tp());
            b.p(format!("setp.gt.f64 {rp} , {rem} , {zero} ;"));
            b.p(format!("setp.lt.f64 {rn} , {rem} , {zero} ;"));
            b.p(format!("setp.lt.f64 {bn} , {den} , {zero} ;"));
            b.p(format!("not.pred {nbn} , {bn} ;"));
            b.p(format!("and.pred {x1} , {rp} , {nbn} ;"));
            b.p(format!("and.pred {x2} , {rn} , {bn} ;"));
            b.p(format!("or.pred {p} , {x1} , {x2} ;"));
            b.p(format!("and.pred {y1} , {rn} , {nbn} ;"));
            b.p(format!("and.pred {y2} , {rp} , {bn} ;"));
            b.p(format!("or.pred {n} , {y1} , {y2} ;"));
            e_pos = p;
            e_neg = n;
        }
        _ => {
            // sqrt
            r = b.t64();
            let rem = b.t64();
            b.p(format!("sqrt.rn.f64 {r} , {a} ;"));
            let ns = neg(&mut b, &r, &zero);
            b.p(format!("fma.rn.f64 {rem} , {ns} , {r} , {a} ;"));
            let (p, n) = (b.tp(), b.tp());
            b.p(format!("setp.gt.f64 {p} , {rem} , {zero} ;"));
            b.p(format!("setp.lt.f64 {n} , {rem} , {zero} ;"));
            e_pos = p;
            e_neg = n;
        }
    }
    let (r_pos, r_neg, up, down, step, res) = (b.tp(), b.tp(), b.t64(), b.t64(), b.t64(), b.t64());
    b.p(format!("setp.gt.f64 {r_pos} , {r} , {zero} ;"));
    b.p(format!("setp.lt.f64 {r_neg} , {r} , {zero} ;"));
    b.p(format!("add.s64 {up} , {r} , 1 ;"));
    b.p(format!("sub.s64 {down} , {r} , 1 ;"));
    let cond = b.tp();
    match mode {
        "rz" => {
            let (x, y) = (b.tp(), b.tp());
            b.p(format!("and.pred {x} , {r_pos} , {e_neg} ;"));
            b.p(format!("and.pred {y} , {r_neg} , {e_pos} ;"));
            b.p(format!("or.pred {cond} , {x} , {y} ;"));
            b.p(format!("mov.b64 {step} , {down} ;"));
        }
        "rm" => {
            b.p(format!("and.pred {cond} , {e_neg} , {e_neg} ;"));
            let (tiny, t1) = (b.t64(), b.t64());
            b.p(format!("mov.b64 {tiny} , 0x8000000000000001 ;"));
            b.p(format!("selp.b64 {t1} , {up} , {tiny} , {r_neg} ;"));
            b.p(format!("selp.b64 {step} , {down} , {t1} , {r_pos} ;"));
        }
        _ => {
            b.p(format!("and.pred {cond} , {e_pos} , {e_pos} ;"));
            let (tiny, t1) = (b.t64(), b.t64());
            b.p(format!("mov.b64 {tiny} , 1 ;"));
            b.p(format!("selp.b64 {t1} , {down} , {tiny} , {r_neg} ;"));
            b.p(format!("selp.b64 {step} , {up} , {t1} , {r_pos} ;"));
        }
    }
    b.p(format!("selp.b64 {res} , {step} , {r} , {cond} ;"));
    b.p(render(&inst.guard, "mov.b64", &[inst.operand(0).to_owned(), res]));
    Some(b.finish())
}

// ---- half-precision fma with a single rounding -------------------------------------

/// `fma.rn.{f16,bf16,f16x2,bf16x2}`: the f16/bf16 product is exact in f64 and the f64
/// sum keeps 42 more bits than the target, so rounding once from f64 matches the
/// hardware except in double-rounding ties that need a >31-bit exponent gap.
fn half_fma(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    let ty = inst.op.rsplit('.').next()?;
    if inst.operands.len() != 4 {
        return None;
    }
    let packed = ty.ends_with("x2");
    let h = if ty.starts_with("bf16") { "bf16" } else { "f16" };
    let mut b = B::new(st);
    let lanes = if packed { 2 } else { 1 };
    let mut results = Vec::new();
    for lane in 0..lanes {
        let mut wide = Vec::new();
        for k in 1..=3 {
            let src = inst.operand(k);
            let raw = if packed {
                let t = b.t32();
                if lane == 0 {
                    b.p(format!("and.b32 {t} , {src} , 65535 ;"));
                } else {
                    b.p(format!("shr.u32 {t} , {src} , 16 ;"));
                }
                t
            } else {
                src.to_owned()
            };
            let w = b.t64();
            b.p(format!("cvt.f64.{h} {w} , {raw} ;"));
            wide.push(w);
        }
        let r = b.t64();
        b.p(format!("fma.rn.f64 {r} , {} , {} , {} ;", wide[0], wide[1], wide[2]));
        let n = b.t32();
        b.p(format!("cvt.rn.{h}.f64 {n} , {r} ;"));
        results.push(n);
    }
    if packed {
        let (lo, hi) = (b.t32(), b.t32());
        b.p(format!("and.b32 {lo} , {} , 65535 ;", results[0]));
        b.p(format!("shl.b32 {hi} , {} , 16 ;", results[1]));
        b.p(format!("or.b32 {lo} , {lo} , {hi} ;"));
        b.p(render(&inst.guard, "mov.b32", &[inst.operand(0).to_owned(), lo]));
    } else {
        b.p(render(&inst.guard, "mov.b32", &[inst.operand(0).to_owned(), results[0].clone()]));
    }
    Some(b.finish())
}

/// Records the name of a module-scope `.global` / `.const` data declaration.
pub(super) fn module_decl(ctx: &mut Ctx, stmt: &[String]) {
    let mut i = 0;
    while stmt.get(i).is_some_and(|t| matches!(t.as_str(), ".visible" | ".weak" | ".extern" | ".common")) {
        if stmt[i] == ".extern" {
            return;
        }
        i += 1;
    }
    if !stmt.get(i).is_some_and(|t| matches!(t.as_str(), ".global" | ".const")) {
        return;
    }
    // The name is the token before `[`, `=` or `;`.
    let mut j = i + 1;
    let mut name = None;
    while let Some(t) = stmt.get(j) {
        if matches!(t.as_str(), "[" | "=" | ";") {
            break;
        }
        if !t.starts_with('.') && !t.chars().all(|c| c.is_ascii_digit()) {
            name = Some(t.clone());
        }
        j += 1;
    }
    // Only `mov dst, sym+off` is rewritten here; everything else is native in `globals.rs`.
    if let Some(name) = name {
        if !ctx.t3.module_globals.contains(&name) {
            ctx.t3.module_globals.push(name);
        }
    }
}

/// `_` sink operands get a scratch register; special registers used outside `mov`
/// are first copied into a register.
fn sinks_and_special_registers(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    let is_sink = |t: &str| t == "_";
    let special = |t: &str| {
        let name = t.trim().strip_prefix('%').unwrap_or("");
        let (base, comp) = name.split_once('.').unwrap_or((name, ""));
        matches!(base, "tid" | "ntid" | "ctaid" | "nctaid") && matches!(comp, "x" | "y" | "z")
            || matches!(base, "laneid" | "warpid" | "nwarpid" | "smid") && comp.is_empty()
    };
    let has_sink = inst.operands.iter().any(|o| o.split_whitespace().any(is_sink));
    let has_special = !inst.op.starts_with("mov.") && inst.operands.iter().any(|o| special(o));
    if !has_sink && !has_special {
        return None;
    }
    let mut lines = Vec::new();
    let mut operands = inst.operands.clone();
    if has_sink {
        let wide = inst.op.contains("64") && !inst.op.starts_with("mov.");
        let mut next = ctx.t3.sink_count;
        for o in &mut operands {
            let tokens: Vec<String> = o
                .split_whitespace()
                .map(|t| {
                    if is_sink(t) {
                        let name = format!("%__sink_{next}");
                        next += 1;
                        name
                    } else {
                        t.to_owned()
                    }
                })
                .collect();
            *o = tokens.join(" ");
        }
        let used = next - ctx.t3.sink_count;
        ctx.t3.sink_count = next;
        ctx.t3.sink_max = ctx.t3.sink_max.max(next);
        ctx.t3.sink_wide = wide;
        let _ = used;
    }
    if has_special {
        for o in &mut operands {
            if special(o) {
                ctx.t3.uses_sr = true;
                lines.push(format!("mov.u32 %__sr , {} ;", o.trim()));
                *o = "%__sr".into();
                break;
            }
        }
    }
    let copy = Inst { guard: inst.guard.clone(), op: inst.op.clone(), operands };
    lines.extend(again(ctx, &copy));
    Some(lines)
}
