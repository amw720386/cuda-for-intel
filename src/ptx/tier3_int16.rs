//! 16-bit integer and predicate lowering. A `.b16`/`.u16`/`.s16` register lives
//! in a 32-bit slot whose upper half is not trusted, so every operand is
//! re-extended (`and 0xffff` for unsigned/bits, `bfe.s32` for signed) before a
//! 32-bit operation, and every result is truncated again (sign-extended for
//! `.s16` destinations to match `ld.s16`). Child module of `tier3`.

use super::{B, Inst, State, is_imm, render};

fn ty16(op: &str) -> Option<&str> {
    let t = op.rsplit('.').next()?;
    matches!(t, "b16" | "u16" | "s16").then_some(t)
}

/// Source operand extended to 32 bits (immediates pass through).
fn rd(b: &mut B, operand: &str, ty: &str) -> String {
    if is_imm(operand) {
        return operand.to_owned();
    }
    let t = b.t32();
    if ty.starts_with('s') {
        b.p(format!("bfe.s32 {t} , {operand} , 0 , 16 ;"));
    } else {
        b.p(format!("and.b32 {t} , {operand} , 65535 ;"));
    }
    t
}

/// Truncates `value` into the 16-bit destination (guarded).
fn wr(b: &mut B, guard: &str, dst: &str, value: &str, ty: &str) {
    if ty.starts_with('s') {
        b.p(render(guard, "bfe.s32", &[dst.to_owned(), value.to_owned(), "0".into(), "16".into()]));
    } else {
        b.p(render(guard, "and.b32", &[dst.to_owned(), value.to_owned(), "65535".into()]));
    }
}

fn width_signed(t: &str) -> Option<(u32, bool)> {
    Some(match t {
        "u8" | "b8" => (8, false),
        "s8" => (8, true),
        "u16" | "b16" => (16, false),
        "s16" => (16, true),
        "u32" | "b32" => (32, false),
        "s32" => (32, true),
        "u64" | "b64" => (64, false),
        "s64" => (64, true),
        _ => return None,
    })
}

pub(super) fn rewrite(st: &mut State, inst: &Inst) -> Option<Vec<String>> {
    let parts: Vec<&str> = inst.op.split('.').collect();
    let family = parts[0];
    if family == "cvt" {
        return cvt(st, inst, &parts);
    }
    if inst.op == "xor.pred" && inst.operands.len() == 3 {
        let mut b = B::new(st);
        let (x, y) = (b.t32(), b.t32());
        b.p(format!("selp.b32 {x} , 1 , 0 , {} ;", inst.operand(1)));
        b.p(format!("selp.b32 {y} , 1 , 0 , {} ;", inst.operand(2)));
        b.p(format!("xor.b32 {x} , {x} , {y} ;"));
        b.p(render(&inst.guard, "setp.ne.u32", &[inst.operand(0).to_owned(), x, "0".into()]));
        return Some(b.finish());
    }
    if family == "slct" && parts.len() == 3 && matches!(parts[1], "b16" | "u16" | "s16") && inst.operands.len() == 4 {
        // slct.T.s32 d, a, b, c: a if c >= 0 else b.
        let t = parts[1];
        let mut b = B::new(st);
        let (x, y) = (rd(&mut b, inst.operand(1), t), rd(&mut b, inst.operand(2), t));
        let p = b.tp();
        b.p(format!("setp.ge.s32 {p} , {} , 0 ;", inst.operand(3)));
        let r = b.t32();
        b.p(format!("selp.b32 {r} , {x} , {y} , {p} ;"));
        wr(&mut b, &inst.guard, inst.operand(0), &r, t);
        return Some(b.finish());
    }
    let t = ty16(&inst.op)?;
    // `mul.wide`/`mad.wide` and `setp` carry the type last too.
    let signed = t.starts_with('s');
    let sfx = if signed { "s32" } else { "u32" };
    let g = inst.guard.as_str();
    let mut b = B::new(st);
    let a = |i: usize| inst.operand(i).to_owned();
    match family {
        "add" | "sub" | "mul" | "mad" | "min" | "max" if !parts.contains(&"wide") && !parts.contains(&"hi") && !parts.contains(&"sat") => {
            let n = if family == "mad" { 4 } else { 3 };
            if inst.operands.len() != n || parts.iter().any(|p| matches!(*p, "cc" | "rn" | "rz" | "relu" | "ftz")) {
                return None;
            }
            let mut ops = vec![String::new()];
            for i in 1..n {
                let v = rd(&mut b, inst.operand(i), t);
                ops.push(v);
            }
            let r = b.t32();
            let op = match family {
                "mul" => format!("mul.lo.{sfx}"),
                "mad" => format!("mad.lo.{sfx}"),
                f => format!("{f}.{sfx}"),
            };
            ops[0] = r.clone();
            b.p(render("", &op, &ops));
            wr(&mut b, g, inst.operand(0), &r, t);
        }
        "mul" if parts.contains(&"hi") && inst.operands.len() == 3 => {
            let (x, y) = (rd(&mut b, inst.operand(1), t), rd(&mut b, inst.operand(2), t));
            let r = b.t32();
            b.p(format!("mul.lo.{sfx} {r} , {x} , {y} ;"));
            b.p(format!("shr.{sfx} {r} , {r} , 16 ;"));
            wr(&mut b, g, inst.operand(0), &r, t);
        }
        "mad" if parts.contains(&"hi") && !parts.contains(&"sat") && inst.operands.len() == 4 => {
            // mad.hi.{u16,s16}: high half of the 32-bit product plus c, truncated.
            let (x, y, z) = (rd(&mut b, inst.operand(1), t), rd(&mut b, inst.operand(2), t), rd(&mut b, inst.operand(3), t));
            let r = b.t32();
            b.p(format!("mul.lo.{sfx} {r} , {x} , {y} ;"));
            b.p(format!("shr.{sfx} {r} , {r} , 16 ;"));
            b.p(format!("add.u32 {r} , {r} , {z} ;"));
            wr(&mut b, g, inst.operand(0), &r, t);
        }
        "sad" if inst.operands.len() == 4 => {
            // sad.{u16,s16}: |a - b| + c on the extended values, truncated.
            let (x, y, z) = (rd(&mut b, inst.operand(1), t), rd(&mut b, inst.operand(2), t), rd(&mut b, inst.operand(3), t));
            let r = b.t32();
            b.p(format!("sad.{sfx} {r} , {x} , {y} , {z} ;"));
            wr(&mut b, g, inst.operand(0), &r, t);
        }
        "mul" | "mad" if parts.contains(&"wide") => {
            let n = if family == "mad" { 4 } else { 3 };
            if inst.operands.len() != n {
                return None;
            }
            let (x, y) = (rd(&mut b, inst.operand(1), t), rd(&mut b, inst.operand(2), t));
            if family == "mul" {
                b.p(render(g, &format!("mul.lo.{sfx}"), &[a(0), x, y]));
            } else {
                b.p(render(g, &format!("mad.lo.{sfx}"), &[a(0), x, y, a(3)]));
            }
        }
        "div" | "rem" | "abs" | "neg" if !parts.contains(&"f16") => {
            let n = if matches!(family, "div" | "rem") { 3 } else { 2 };
            if inst.operands.len() != n {
                return None;
            }
            let mut ops = vec![String::new()];
            for i in 1..n {
                let v = rd(&mut b, inst.operand(i), t);
                ops.push(v);
            }
            let r = b.t32();
            ops[0] = r.clone();
            b.p(render("", &format!("{family}.{sfx}"), &ops));
            wr(&mut b, g, inst.operand(0), &r, t);
        }
        // Legacy parser lowering of `shr.u16`/`shr.s16` is hardware-verified; keep it.
        "shr" if t != "b16" => return None,
        "shl" | "shr" if inst.operands.len() == 3 => {
            let x = rd(&mut b, inst.operand(1), t);
            let r = b.t32();
            let op = if family == "shl" { "shl.b32".to_owned() } else { format!("shr.{sfx}") };
            b.p(format!("{op} {r} , {x} , {} ;", inst.operand(2)));
            wr(&mut b, g, inst.operand(0), &r, t);
        }
        "not" | "cnot" if inst.operands.len() == 2 => {
            let x = rd(&mut b, inst.operand(1), t);
            let r = b.t32();
            if family == "not" {
                b.p(format!("not.b32 {r} , {x} ;"));
            } else {
                let p = b.tp();
                b.p(format!("setp.eq.u32 {p} , {x} , 0 ;"));
                b.p(format!("selp.b32 {r} , 1 , 0 , {p} ;"));
            }
            wr(&mut b, g, inst.operand(0), &r, t);
        }
        "and" | "or" | "xor" if inst.operands.len() == 3 => {
            let (x, y) = (rd(&mut b, inst.operand(1), t), rd(&mut b, inst.operand(2), t));
            let r = b.t32();
            b.p(format!("{family}.b32 {r} , {x} , {y} ;"));
            wr(&mut b, g, inst.operand(0), &r, t);
        }
        "selp" if inst.operands.len() == 4 && !is_imm(inst.operand(3)) => {
            let (x, y) = (rd(&mut b, inst.operand(1), t), rd(&mut b, inst.operand(2), t));
            let r = b.t32();
            b.p(render("", "selp.b32", &[r.clone(), x, y, a(3)]));
            wr(&mut b, g, inst.operand(0), &r, t);
        }
        "set" if parts.len() >= 4 && inst.operands.len() >= 3 && matches!(parts[parts.len() - 2], "u32" | "s32") => {
            // set.cmp[.bool].{u32,s32}.AT d, a, b[, c]: all-ones when true.
            let (x, y) = (rd(&mut b, inst.operand(1), t), rd(&mut b, inst.operand(2), t));
            let p = b.tp();
            let mut ops = inst.operands.clone();
            ops[0] = p.clone();
            ops[1] = x;
            ops[2] = y;
            let mut name: Vec<&str> = vec!["setp"];
            name.extend(parts[1..parts.len() - 2].iter().map(|p| match *p {
                "lo" => "lt",
                "ls" => "le",
                "hi" => "gt",
                "hs" => "ge",
                p => p,
            }));
            name.push(if t == "b16" { "u32" } else { sfx });
            b.p(render("", &name.join("."), &ops));
            b.p(render(g, "selp.b32", &[a(0), "-1".into(), "0".into(), p]));
        }
        "setp" if parts.len() >= 3 && inst.operands.len() >= 3 => {
            // setp.cmp[.bool].T p[|q], a, b[, c]: compare as 32-bit after extension.
            let (x, y) = (rd(&mut b, inst.operand(1), t), rd(&mut b, inst.operand(2), t));
            let mut ops = inst.operands.clone();
            ops[1] = x;
            ops[2] = y;
            let mut name: Vec<&str> = parts[..parts.len() - 1]
                .iter()
                .map(|p| match *p {
                    "lo" => "lt",
                    "ls" => "le",
                    "hi" => "gt",
                    "hs" => "ge",
                    p => p,
                })
                .collect();
            name.push(if t == "b16" { "u32" } else { sfx });
            b.p(render(g, &name.join("."), &ops));
        }
        _ => return None,
    }
    Some(b.finish())
}

/// Integer `cvt` where either side is 8/16 bits and the other is an integer.
fn cvt(st: &mut State, inst: &Inst, parts: &[&str]) -> Option<Vec<String>> {
    if inst.operands.len() != 2 {
        return None;
    }
    let sat = parts.contains(&"sat");
    let types: Vec<&str> = parts[1..].iter().copied().filter(|p| width_signed(p).is_some()).collect();
    if types.len() != 2 || parts[1..].iter().any(|p| !matches!(*p, "sat") && width_signed(p).is_none()) {
        return None;
    }
    let ((dw, ds), (sw, ss)) = (width_signed(types[0])?, width_signed(types[1])?);
    // 8-bit <-> 64-bit pairs have no legacy lowering; everything else needs a 16-bit side.
    let byte_wide = (dw == 8 && sw == 64) || (dw == 64 && sw == 8);
    if !byte_wide && (dw > 16 && sw > 16 || dw == 8 && sw == 8 || (dw != 16 && sw != 16)) {
        return None;
    }
    let g = inst.guard.as_str();
    let mut b = B::new(st);
    // Source extended to 32 bits by its own signedness.
    let src = inst.operand(1);
    let x = if is_imm(src) {
        src.to_owned()
    } else if sw >= 32 {
        // 64-bit sources contribute their low half.
        if sw == 64 {
            let t = b.t32();
            b.p(format!("cvt.u32.u64 {t} , {src} ;"));
            t
        } else {
            src.to_owned()
        }
    } else {
        let t = b.t32();
        if ss {
            b.p(format!("bfe.s32 {t} , {src} , 0 , {sw} ;"));
        } else {
            b.p(format!("bfe.u32 {t} , {src} , 0 , {sw} ;"));
        }
        t
    };
    // Signedness of the extended value for clamping/widening.
    let xs = ss || sw < 32 && ss;
    let mut v = x;
    if dw < 32 {
        if sat {
            let (lo, hi): (i64, i64) = match (dw, ds) {
                (16, true) => (-32768, 32767),
                (16, false) => (0, 65535),
                (8, true) => (-128, 127),
                _ => (0, 255),
            };
            let t = b.t32();
            if xs {
                b.p(format!("max.s32 {t} , {v} , {lo} ;"));
                b.p(format!("min.s32 {t} , {t} , {hi} ;"));
            } else {
                b.p(format!("min.u32 {t} , {v} , {hi} ;"));
            }
            v = t;
        }
        let t = b.t32();
        if ds {
            b.p(format!("bfe.s32 {t} , {v} , 0 , {dw} ;"));
        } else {
            b.p(format!("bfe.u32 {t} , {v} , 0 , {dw} ;"));
        }
        v = t;
    } else if sat && xs && !ds {
        // cvt.sat.{u32,u64}.{s8,s16}: negative sources clamp to 0. Wider destinations
        // need no other clamp.
        let t = b.t32();
        b.p(format!("max.s32 {t} , {v} , 0 ;"));
        v = t;
    }
    if dw == 64 {
        let op = if xs { "cvt.s64.s32" } else { "cvt.u64.u32" };
        b.p(render(g, op, &[inst.operand(0).to_owned(), v]));
    } else {
        b.p(render(g, "mov.b32", &[inst.operand(0).to_owned(), v]));
    }
    Some(b.finish())
}
