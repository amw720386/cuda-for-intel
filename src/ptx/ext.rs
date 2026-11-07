//! Parsing of PTX instructions beyond the original GGML subset.
//!
//! `parse` is consulted before the legacy pattern table in `parse.rs`, so it
//! must only claim spellings that table does not already lower. Emission lives
//! in `ext_emit.rs` (a child of `spirv.rs`, which owns the emitter state).

use super::{AtomicOp, Comparison, Error, Instruction, Space};

/// `prmt` selector modes beyond the default byte-permute.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PrmtMode {
    F4e,
    B4e,
    Rc8,
    Ecl,
    Ecr,
    Rc16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReduxOp {
    Add,
    Min,
    Max,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ShflMode {
    Idx,
    Up,
    Down,
    Bfly,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Combine {
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ClockKind {
    Clock32,
    Clock64,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum ExtOp {
    // 32/64-bit integer helpers. Sources are `src[..]`, the result is `dst[0]`.
    Brev { wide: bool },
    Bmsk { wrap: bool },
    Fns,
    Szext { wrap: bool, signed: bool },
    Mul24 { hi: bool, signed: bool },
    Mad24 { hi: bool, signed: bool },
    Sad { signed: bool },
    Cnot { wide: bool },
    Bfind { wide: bool, signed: bool, shiftamt: bool },
    Clz64,
    MadHi { signed: bool },
    MulHi64 { signed: bool },
    Bfe64 { signed: bool },
    Bfi64,
    Prmt(PrmtMode),
    Dp4a { signed: [bool; 2] },
    Dp2a { hi: bool, signed: [bool; 2] },
    Shf { left: bool, clamp: bool },
    Slct { wide: bool, float_selector: bool },
    SetpCombine { cmp: Comparison, signed: bool, wide: bool, combine: Combine },
    // Register moves and special registers.
    PackB64,
    UnpackB64,
    UnpackB32,
    WarpId,
    NWarpId,
    Clock(ClockKind),
    /// `cvta.local/global`, `cvta.to.local`: the pointer value is unchanged.
    CvtaCopy,
    IsSpace(Space),
    // Warp-level collectives.
    Redux { op: ReduxOp, signed: bool },
    MatchAny { wide: bool },
    MatchAll { wide: bool },
    VoteUni,
    Elect,
    ShflDyn(ShflMode),
    // Atomics beyond the 32-bit integer forms handled by `Instruction::Atomic`.
    Atom { op: AtomicOp, ty: AtomTy, space: Space, returns: bool },
    AtomIncDec { dec: bool, space: Space, returns: bool },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AtomTy {
    U64,
    S64,
    F32,
    F64,
}

impl ExtOp {
    /// True when lowering needs 32-lane subgroup execution.
    pub(super) fn uses_subgroup(self) -> bool {
        matches!(
            self,
            Self::Redux { .. }
                | Self::MatchAny { .. }
                | Self::MatchAll { .. }
                | Self::VoteUni
                | Self::Elect
                | Self::ShflDyn(_)
                | Self::WarpId
                | Self::NWarpId
        )
    }
}

fn unsupported(body: &[&str]) -> Error {
    Error::new(format!("unsupported PTX instruction: {}", body.join(" ")))
}

/// Splits `body[1..]` at top-level commas. Brace/bracket groups stay whole.
fn operands<'a>(body: &[&'a str]) -> Vec<Vec<&'a str>> {
    let mut out = Vec::new();
    let mut current = Vec::new();
    let mut depth = 0i32;
    for &token in &body[1..] {
        match token {
            "{" | "[" => depth += 1,
            "}" | "]" => depth -= 1,
            _ => {}
        }
        if token == "," && depth == 0 {
            out.push(std::mem::take(&mut current));
        } else {
            current.push(token);
        }
    }
    if !current.is_empty() {
        out.push(current);
    }
    out
}

/// A single-token operand.
fn reg<'a>(ops: &[Vec<&'a str>], index: usize) -> Option<&'a str> {
    match ops.get(index)?.as_slice() {
        [one] => Some(one),
        _ => None,
    }
}

/// `[ addr ]` operand.
fn address<'a>(ops: &[Vec<&'a str>], index: usize) -> Option<&'a str> {
    match ops.get(index)?.as_slice() {
        ["[", addr, "]"] => Some(addr),
        _ => None,
    }
}

fn parse_u32(text: &str) -> Option<u32> {
    if let Some(hex) = text.strip_prefix("0x").or_else(|| text.strip_prefix("0X")) {
        u32::from_str_radix(hex, 16).ok()
    } else {
        text.parse::<u32>().ok()
    }
}

/// Returns `Ok(None)` for instructions this module does not own.
pub(super) fn parse<'a>(body: &[&'a str]) -> Result<Option<Instruction<'a>>, Error> {
    let Some(&op) = body.first() else {
        return Ok(None);
    };
    let ops = operands(body);
    let parts: Vec<&str> = op.split('.').collect();
    let fail = || unsupported(body);
    let ext = |op: ExtOp, dst: [&'a str; 2], src: [&'a str; 4]| {
        Ok(Some(Instruction::Ext { op, dst, src }))
    };
    const NONE: &str = "0";
    // Operand shorthands.
    let r = |i: usize| reg(&ops, i);
    macro_rules! need {
        ($e:expr) => {
            $e.ok_or_else(fail)?
        };
    }
    match parts.as_slice() {
        ["brev", ty] if matches!(*ty, "b32" | "b64") => {
            let (d, a) = (need!(r(0)), need!(r(1)));
            ext(ExtOp::Brev { wide: *ty == "b64" }, [d, ""], [a, NONE, NONE, NONE])
        }
        ["bmsk", mode @ ("wrap" | "clamp"), "b32"] => {
            let (d, a, b) = (need!(r(0)), need!(r(1)), need!(r(2)));
            ext(ExtOp::Bmsk { wrap: *mode == "wrap" }, [d, ""], [a, b, NONE, NONE])
        }
        ["fns", "b32"] => {
            let (d, a, b, c) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(ExtOp::Fns, [d, ""], [a, b, c, NONE])
        }
        ["szext", mode @ ("wrap" | "clamp"), ty @ ("s32" | "u32")] => {
            let (d, a, b) = (need!(r(0)), need!(r(1)), need!(r(2)));
            ext(
                ExtOp::Szext { wrap: *mode == "wrap", signed: *ty == "s32" },
                [d, ""],
                [a, b, NONE, NONE],
            )
        }
        ["mul24", half @ ("lo" | "hi"), ty @ ("s32" | "u32")] => {
            let (d, a, b) = (need!(r(0)), need!(r(1)), need!(r(2)));
            ext(
                ExtOp::Mul24 { hi: *half == "hi", signed: *ty == "s32" },
                [d, ""],
                [a, b, NONE, NONE],
            )
        }
        ["mad24", half @ ("lo" | "hi"), ty @ ("s32" | "u32")] => {
            let (d, a, b, c) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(
                ExtOp::Mad24 { hi: *half == "hi", signed: *ty == "s32" },
                [d, ""],
                [a, b, c, NONE],
            )
        }
        ["sad", ty @ ("s32" | "u32")] => {
            let (d, a, b, c) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(ExtOp::Sad { signed: *ty == "s32" }, [d, ""], [a, b, c, NONE])
        }
        ["cnot", ty @ ("b32" | "b64")] => {
            let (d, a) = (need!(r(0)), need!(r(1)));
            ext(ExtOp::Cnot { wide: *ty == "b64" }, [d, ""], [a, NONE, NONE, NONE])
        }
        ["bfind", rest @ ..] => {
            let (shiftamt, rest) = match rest {
                ["shiftamt", rest @ ..] => (true, rest),
                other => (false, other),
            };
            let (wide, signed) = match rest {
                ["u32"] => (false, false),
                ["s32"] => (false, true),
                ["u64"] => (true, false),
                ["s64"] => (true, true),
                _ => return Err(fail()),
            };
            let (d, a) = (need!(r(0)), need!(r(1)));
            ext(ExtOp::Bfind { wide, signed, shiftamt }, [d, ""], [a, NONE, NONE, NONE])
        }
        ["clz", "b64"] => {
            let (d, a) = (need!(r(0)), need!(r(1)));
            ext(ExtOp::Clz64, [d, ""], [a, NONE, NONE, NONE])
        }
        ["mad", "hi", ty @ ("s32" | "u32")] => {
            let (d, a, b, c) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(ExtOp::MadHi { signed: *ty == "s32" }, [d, ""], [a, b, c, NONE])
        }
        ["mul", "hi", ty @ ("s64" | "u64")] => {
            let (d, a, b) = (need!(r(0)), need!(r(1)), need!(r(2)));
            ext(ExtOp::MulHi64 { signed: *ty == "s64" }, [d, ""], [a, b, NONE, NONE])
        }
        ["bfe", ty @ ("s64" | "u64")] => {
            let (d, a, p, l) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(ExtOp::Bfe64 { signed: *ty == "s64" }, [d, ""], [a, p, l, NONE])
        }
        ["bfi", "b64"] => {
            let (d, f, a, p, l) =
                (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)), need!(r(4)));
            // Four sources: pack (f, a, pos, len).
            ext(ExtOp::Bfi64, [d, ""], [f, a, p, l])
        }
        ["prmt", "b32", mode] => {
            let mode = match *mode {
                "f4e" => PrmtMode::F4e,
                "b4e" => PrmtMode::B4e,
                "rc8" => PrmtMode::Rc8,
                "ecl" => PrmtMode::Ecl,
                "ecr" => PrmtMode::Ecr,
                "rc16" => PrmtMode::Rc16,
                _ => return Err(fail()),
            };
            let (d, a, b, c) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(ExtOp::Prmt(mode), [d, ""], [a, b, c, NONE])
        }
        ["dp4a", a @ ("s32" | "u32"), b @ ("s32" | "u32")] => {
            let (d, x, y, z) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(
                ExtOp::Dp4a { signed: [*a == "s32", *b == "s32"] },
                [d, ""],
                [x, y, z, NONE],
            )
        }
        ["dp2a", half @ ("lo" | "hi"), a @ ("s32" | "u32"), b @ ("s32" | "u32")] => {
            let (d, x, y, z) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(
                ExtOp::Dp2a { hi: *half == "hi", signed: [*a == "s32", *b == "s32"] },
                [d, ""],
                [x, y, z, NONE],
            )
        }
        ["shf", dir @ ("l" | "r"), mode @ ("wrap" | "clamp"), "b32"] => {
            let (d, a, b, c) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(
                ExtOp::Shf { left: *dir == "l", clamp: *mode == "clamp" },
                [d, ""],
                [a, b, c, NONE],
            )
        }
        ["slct", dty, sty @ ("s32" | "f32")] => {
            let wide = match *dty {
                "b32" | "u32" | "s32" | "f32" => false,
                "b64" | "u64" | "s64" | "f64" => true,
                _ => return Err(fail()),
            };
            let (d, a, b, c) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(
                ExtOp::Slct { wide, float_selector: *sty == "f32" },
                [d, ""],
                [a, b, c, NONE],
            )
        }
        ["setp", cmp, combine @ ("and" | "or" | "xor"), ty]
            if ops.len() == 4 && !ty.starts_with('f') && !ty.starts_with('b') =>
        {
            let cmp = match *cmp {
                "eq" => Comparison::Equal,
                "ne" => Comparison::NotEqual,
                "lt" | "lo" => Comparison::Less,
                "le" | "ls" => Comparison::LessEqual,
                "gt" | "hi" => Comparison::Greater,
                "ge" | "hs" => Comparison::GreaterEqual,
                _ => return Err(fail()),
            };
            let (signed, wide) = match *ty {
                "s32" => (true, false),
                "u32" => (false, false),
                "s64" => (true, true),
                "u64" => (false, true),
                _ => return Err(fail()),
            };
            let combine = match *combine {
                "and" => Combine::And,
                "or" => Combine::Or,
                _ => Combine::Xor,
            };
            let (d, a, b, q) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(
                ExtOp::SetpCombine { cmp, signed, wide, combine },
                [d, ""],
                [a, b, q, NONE],
            )
        }
        ["mov", "b64" | "u64"] if ops.len() == 2 => {
            if let ["{", lo, ",", hi, "}"] = ops[1].as_slice() {
                let d = need!(r(0));
                return ext(ExtOp::PackB64, [d, ""], [lo, hi, NONE, NONE]);
            }
            if let ["{", lo, ",", hi, "}"] = ops[0].as_slice() {
                let s = need!(r(1));
                return ext(ExtOp::UnpackB64, [lo, hi], [s, NONE, NONE, NONE]);
            }
            Ok(None)
        }
        ["mov", "b32" | "u32"] if ops.len() == 2 => {
            if let ["{", lo, ",", hi, "}"] = ops[0].as_slice() {
                let s = need!(r(1));
                return ext(ExtOp::UnpackB32, [lo, hi], [s, NONE, NONE, NONE]);
            }
            match r(1) {
                Some("%warpid") => ext(ExtOp::WarpId, [need!(r(0)), ""], [NONE; 4]),
                Some("%nwarpid") => ext(ExtOp::NWarpId, [need!(r(0)), ""], [NONE; 4]),
                Some("%clock") => {
                    ext(ExtOp::Clock(ClockKind::Clock32), [need!(r(0)), ""], [NONE; 4])
                }
                _ => Ok(None),
            }
        }
        ["mov", "u64" | "b64"] if matches!(r(1), Some("%clock64" | "%globaltimer")) => {
            ext(ExtOp::Clock(ClockKind::Clock64), [need!(r(0)), ""], [NONE; 4])
        }
        ["cvta", "local" | "global", "u64"] | ["cvta", "to", "local", "u64"] => {
            let (d, a) = (need!(r(0)), need!(r(1)));
            ext(ExtOp::CvtaCopy, [d, ""], [a, NONE, NONE, NONE])
        }
        ["isspacep", space @ ("shared" | "global" | "local")] => {
            let (d, a) = (need!(r(0)), need!(r(1)));
            let space = match *space {
                "shared" => Space::Shared,
                "local" => Space::Local,
                _ => Space::Global,
            };
            ext(ExtOp::IsSpace(space), [d, ""], [a, NONE, NONE, NONE])
        }
        ["redux", "sync", kind, ty @ ("u32" | "s32" | "b32")] => {
            let op = match *kind {
                "add" => ReduxOp::Add,
                "min" => ReduxOp::Min,
                "max" => ReduxOp::Max,
                "and" => ReduxOp::And,
                "or" => ReduxOp::Or,
                "xor" => ReduxOp::Xor,
                _ => return Err(fail()),
            };
            let (d, a, m) = (need!(r(0)), need!(r(1)), need!(r(2)));
            ext(ExtOp::Redux { op, signed: *ty == "s32" }, [d, ""], [a, m, NONE, NONE])
        }
        ["match", "any", "sync", ty @ ("b32" | "b64")] => {
            let (d, a, m) = (need!(r(0)), need!(r(1)), need!(r(2)));
            ext(ExtOp::MatchAny { wide: *ty == "b64" }, [d, ""], [a, m, NONE, NONE])
        }
        ["match", "all", "sync", ty @ ("b32" | "b64")] => {
            let dst = need!(r(0));
            let (d, p) = match dst.split_once('|') {
                Some((d, p)) => (d, p),
                None => (dst, ""),
            };
            let (a, m) = (need!(r(1)), need!(r(2)));
            ext(ExtOp::MatchAll { wide: *ty == "b64" }, [d, p], [a, m, NONE, NONE])
        }
        ["vote", "sync", "uni", "pred"] => {
            let (d, a, m) = (need!(r(0)), need!(r(1)), need!(r(2)));
            ext(ExtOp::VoteUni, [d, ""], [a, m, NONE, NONE])
        }
        ["elect", "sync"] => {
            let dst = need!(r(0));
            let (d, p) = match dst.split_once('|') {
                Some((d, p)) => (d, p),
                None => ("", dst),
            };
            let m = need!(r(1));
            ext(ExtOp::Elect, [d, p], [m, NONE, NONE, NONE])
        }
        ["shfl", "sync", mode, "b32"] if ops.len() == 5 && parse_u32(r(3).unwrap_or("")).is_none() => {
            let mode = match *mode {
                "idx" => ShflMode::Idx,
                "up" => ShflMode::Up,
                "down" => ShflMode::Down,
                "bfly" => ShflMode::Bfly,
                _ => return Err(fail()),
            };
            let dst = need!(r(0));
            let (d, p) = match dst.split_once('|') {
                Some((d, p)) => (d, p),
                None => (dst, ""),
            };
            let (a, b, c) = (need!(r(1)), need!(r(2)), need!(r(3)));
            if r(4).and_then(parse_u32).is_some_and(|mask| mask != u32::MAX) {
                return Err(Error::new(
                    "only full-warp shfl.sync membermask 0xffffffff is supported",
                ));
            }
            ext(ExtOp::ShflDyn(mode), [d, p], [a, b, c, NONE])
        }
        ["atom" | "red", rest @ ..] => parse_atomic(body, &ops, parts[0] == "atom", rest),
        _ => Ok(None),
    }
}

fn parse_atomic<'a>(
    body: &[&'a str],
    ops: &[Vec<&'a str>],
    returns: bool,
    parts: &[&str],
) -> Result<Option<Instruction<'a>>, Error> {
    let fail = || unsupported(body);
    let mut space = Space::Global;
    let mut kind = None;
    let mut ty = None;
    for part in parts {
        match *part {
            "global" => space = Space::Global,
            "shared" | "shared::cta" => space = Space::Shared,
            "relaxed" | "acquire" | "release" | "acq_rel" | "gpu" | "cta" | "sys" | "cluster"
            | "sem" => {}
            "add" => kind = Some(AtomicOp::Add),
            "min" => kind = Some(AtomicOp::Min),
            "max" => kind = Some(AtomicOp::Max),
            "and" => kind = Some(AtomicOp::And),
            "or" => kind = Some(AtomicOp::Or),
            "xor" => kind = Some(AtomicOp::Xor),
            "exch" => kind = Some(AtomicOp::Exch),
            "cas" => kind = Some(AtomicOp::Cas),
            "inc" => kind = None,
            "dec" => kind = None,
            "u64" | "b64" | "s64" | "f32" | "f64" | "u32" | "b32" | "s32" => ty = Some(*part),
            _ => return Ok(None),
        }
    }
    let inc = parts.contains(&"inc");
    let dec = parts.contains(&"dec");
    let Some(ty) = ty else {
        return Ok(None);
    };
    if inc || dec {
        if ty != "u32" {
            return Ok(None);
        }
        let (dst, rest) = if returns {
            (reg(ops, 0).ok_or_else(fail)?, &ops[1..])
        } else {
            ("", ops)
        };
        let addr = address(rest, 0).ok_or_else(fail)?;
        let value = reg(rest, 1).ok_or_else(fail)?;
        return Ok(Some(Instruction::Ext {
            op: ExtOp::AtomIncDec { dec, space, returns },
            dst: [dst, ""],
            src: [addr, value, "0", "0"],
        }));
    }
    let Some(op) = kind else {
        return Ok(None);
    };
    let atom_ty = match ty {
        "u64" | "b64" => AtomTy::U64,
        "s64" => AtomTy::S64,
        "f64" => AtomTy::F64,
        "f32" if matches!(op, AtomicOp::Min | AtomicOp::Max | AtomicOp::Exch | AtomicOp::Cas) => {
            AtomTy::F32
        }
        _ => return Ok(None),
    };
    let (dst, rest) = if returns {
        (reg(ops, 0).ok_or_else(fail)?, &ops[1..])
    } else {
        ("", ops)
    };
    let addr = address(rest, 0).ok_or_else(fail)?;
    let (value, compare) = if op == AtomicOp::Cas {
        // PTX order: [addr], compare, value.
        (reg(rest, 2).ok_or_else(fail)?, reg(rest, 1).ok_or_else(fail)?)
    } else {
        (reg(rest, 1).ok_or_else(fail)?, "0")
    };
    if returns && dst.is_empty() {
        return Err(fail());
    }
    Ok(Some(Instruction::Ext {
        op: ExtOp::Atom { op, ty: atom_ty, space, returns },
        dst: [dst, ""],
        src: [addr, value, compare, "0"],
    }))
}
