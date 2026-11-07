//! Tier-3 PTX lowering: carry chains, video SIMD, partial-mask warp
//! collectives and a handful of remaining odd spellings.
//!
//! Like `normalize.rs` this works on text: each rewrite expands one PTX
//! instruction into a short run of instructions the parser already knows. It is
//! a child module of `normalize` (see the `#[path]` line there) so it can use
//! `Ctx`, `Inst` and `render` directly. `rewrite` is consulted first; `None`
//! means "not mine" and the legacy rewrites continue.

use super::{Ctx, Inst, render};

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
        *self = State::default();
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

pub(super) fn rewrite(ctx: &mut Ctx, inst: &Inst) -> Option<Vec<String>> {
    if inst.operands.iter().any(|o| o == "WARP_SZ" || o == "%WARP_SZ") {
        let mut copy = with_op(inst, inst.op.clone());
        for o in &mut copy.operands {
            if o == "WARP_SZ" || o == "%WARP_SZ" {
                *o = "32".into();
            }
        }
        return Some(again(ctx, &copy));
    }
    let family = inst.op.split('.').next().unwrap_or("");
    if matches!(family, "mma" | "wmma") {
        return super::tier3_mma::rewrite(ctx, inst);
    }
    let st = &mut ctx.t3;
    match family {
        "ld" | "ldu" if inst.op.split('.').any(|p| p == "const") => {
            // Constant-bank loads read ordinary device memory.
            let parts: Vec<&str> = inst.op.split('.').filter(|p| *p != "const").collect();
            let mut op = parts.join(".");
            op = op.replacen("ld.", "ld.global.", 1);
            let copy = with_op(inst, op);
            Some(again(ctx, &copy))
        }
        "cvta" if inst.op.split('.').any(|p| p == "const") && inst.operands.len() == 2 => {
            Some(vec![render(&inst.guard, "mov.b64", &inst.operands)])
        }
        "mov" if matches!(inst.operand(1), "%dynamic_smem_size" | "%total_smem_size") && inst.operands.len() == 2 => {
            let bits = if inst.op.ends_with("64") { 64 } else { 32 };
            Some(vec![render(&inst.guard, &format!("mov.b{bits}"), &[inst.operand(0).to_owned(), "49152".into()])])
        }
        "bar" | "barrier" => barrier(ctx, inst),
        "shfl" => shfl_masks(ctx, inst),
        "vote" => vote_masks(ctx, inst),
        "cvt" => cvt_misc(ctx, inst),
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
