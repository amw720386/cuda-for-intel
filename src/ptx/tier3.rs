//! Tier-3 PTX lowering: carry chains, video SIMD, partial-mask warp
//! collectives and a handful of remaining odd spellings.
//!
//! Like `normalize.rs` this works on text: each rewrite expands one PTX
//! instruction into a short run of instructions the parser already knows. It is
//! a child module of `normalize` (see the `#[path]` line there) so it can use
//! `Ctx`, `Inst` and `render` directly. `rewrite` is consulted first; `None`
//! means "not mine" and the legacy rewrites continue.

use super::{Inst, render};

/// Per-entry state owned by this module (embedded in `Ctx`).
#[derive(Default)]
pub(super) struct State {
    /// Scratch registers requested by tier-3 rewrites of the current entry.
    x32: u32,
    x64: u32,
    xp: u32,
    /// The condition-code carry flag `CC.CF` is used by the entry.
    cc: bool,
    /// Named-barrier lowering needs shared counters.
    pub(super) named_barriers: bool,
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
        if self.cc {
            decls.push(".reg .b32 %__cc ;".into());
            decls.push("mov.b32 %__cc , 0 ;".into());
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

pub(super) fn rewrite(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    let family = inst.op.split('.').next().unwrap_or("");
    match family {
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
