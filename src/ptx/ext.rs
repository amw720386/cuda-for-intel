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

/// Scalar numeric types of `cvt` and the floating-point instructions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum NumTy {
    S8,
    U8,
    S16,
    U16,
    S32,
    U32,
    S64,
    U64,
    F16,
    Bf16,
    F16x2,
    Bf16x2,
    F32,
    F64,
    Tf32,
}

impl NumTy {
    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "s8" => Self::S8,
            "u8" | "b8" => Self::U8,
            "s16" => Self::S16,
            "u16" | "b16" => Self::U16,
            "s32" => Self::S32,
            "u32" | "b32" => Self::U32,
            "s64" => Self::S64,
            "u64" | "b64" => Self::U64,
            "f16" => Self::F16,
            "bf16" => Self::Bf16,
            "f16x2" => Self::F16x2,
            "bf16x2" => Self::Bf16x2,
            "f32" => Self::F32,
            "f64" => Self::F64,
            "tf32" => Self::Tf32,
            _ => return None,
        })
    }

    pub(super) fn is_float(self) -> bool {
        matches!(
            self,
            Self::F16 | Self::Bf16 | Self::F16x2 | Self::Bf16x2 | Self::F32 | Self::F64 | Self::Tf32
        )
    }

    pub(super) fn is_signed(self) -> bool {
        matches!(self, Self::S8 | Self::S16 | Self::S32 | Self::S64)
    }

    pub(super) fn is_half(self) -> bool {
        matches!(self, Self::F16 | Self::Bf16 | Self::F16x2 | Self::Bf16x2)
    }

    pub(super) fn is_packed(self) -> bool {
        matches!(self, Self::F16x2 | Self::Bf16x2)
    }

    pub(super) fn is_bf16(self) -> bool {
        matches!(self, Self::Bf16 | Self::Bf16x2)
    }

    /// Integer width in bits (floats report their storage width).
    pub(super) fn bits(self) -> u32 {
        match self {
            Self::S8 | Self::U8 => 8,
            Self::S16 | Self::U16 | Self::F16 | Self::Bf16 => 16,
            Self::S64 | Self::U64 | Self::F64 => 64,
            _ => 32,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Round {
    None,
    Rn,
    Rz,
    Rm,
    Rp,
    Rni,
    Rzi,
    Rmi,
    Rpi,
    Rna,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(super) struct Mods {
    pub(super) sat: bool,
    pub(super) relu: bool,
    pub(super) nan: bool,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FOp {
    Add,
    Sub,
    Mul,
    Fma,
    Min,
    Max,
    Neg,
    Abs,
    Ex2,
    Tanh,
    Rcp,
    Sqrt,
    Rsqrt,
    Div,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum FCmp {
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    Equ,
    Neu,
    Ltu,
    Leu,
    Gtu,
    Geu,
    Num,
    Nan,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum TestKind {
    Finite,
    Infinite,
    Number,
    NotANumber,
    Normal,
    Subnormal,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum ExtOp {
    /// Floating-point arithmetic on f64, f16/bf16 (scalar or packed) and
    /// saturating / NaN-propagating f32. Sources are `src[0..3]`.
    Float { op: FOp, ty: NumTy, mods: Mods },
    /// Generic `cvt`. Packed destinations take two sources (`src[0]` is the
    /// upper lane).
    Cvt { d: NumTy, s: NumTy, round: Round, mods: Mods },
    FSetp { cmp: FCmp, ty: NumTy },
    FSet { cmp: FCmp, d: NumTy, s: NumTy },
    Testp { kind: TestKind, ty: NumTy },
    // 32/64-bit integer helpers. Sources are `src[..]`, the result is `dst[0]`.
    Brev { wide: bool },
    Bmsk { wrap: bool },
    Fns,
    Szext { wrap: bool, signed: bool },
    Mul24 { hi: bool, signed: bool },
    Mad24 { hi: bool, signed: bool },
    Sad { signed: bool },
    Cnot { wide: bool },
    /// `add.sat.s32` / `sub.sat.s32`.
    SatAddSub { sub: bool },
    /// `mad.wide.{s32,u32} d64, a, b, c64`.
    MadWide { signed: bool },
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
    /// fp8 pair conversions (`ext_fp8.rs`). `e5m2` selects E5M2 over E4M3 (OCP FP8).
    Fp8 { dir: Fp8Dir, e5m2: bool, relu: bool },
}

/// Direction and source type of an fp8 pair conversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Fp8Dir {
    /// `cvt.rn.satfinite[.relu].{e4m3x2,e5m2x2}.f32 d, a, b`: `a` fills the upper byte.
    FromF32,
    /// `cvt.rn.satfinite[.relu].{e4m3x2,e5m2x2}.f16x2 d, a`.
    FromF16x2,
    /// `cvt.rn[.relu].f16x2.{e4m3x2,e5m2x2} d, a`: the upper byte fills the upper half.
    ToF16x2,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum AtomTy {
    U64,
    S64,
    F32,
    F64,
    F16x2,
    Bf16x2,
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
        [name @ ("add" | "sub"), "sat", "s32"] => {
            let (d, a, b) = (need!(r(0)), need!(r(1)), need!(r(2)));
            ext(ExtOp::SatAddSub { sub: *name == "sub" }, [d, ""], [a, b, NONE, NONE])
        }
        ["mad", "wide", ty @ ("s32" | "u32")] => {
            let (d, a, b, c) = (need!(r(0)), need!(r(1)), need!(r(2)), need!(r(3)));
            ext(ExtOp::MadWide { signed: *ty == "s32" }, [d, ""], [a, b, c, NONE])
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
            if matches!(r(1), Some("%clock64" | "%globaltimer")) {
                return ext(ExtOp::Clock(ClockKind::Clock64), [need!(r(0)), ""], [NONE; 4]);
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
        ["cvt", ..] => parse_cvt(body, &ops, &parts),
        ["setp", ..] => parse_fsetp(body, &ops, &parts),
        ["set", ..] => parse_fset(body, &ops, &parts),
        ["testp", kind, ty] => {
            let kind = match *kind {
                "finite" => TestKind::Finite,
                "infinite" => TestKind::Infinite,
                "number" => TestKind::Number,
                "notanumber" => TestKind::NotANumber,
                "normal" => TestKind::Normal,
                "subnormal" => TestKind::Subnormal,
                _ => return Err(fail()),
            };
            let ty = NumTy::parse(ty).filter(|t| matches!(t, NumTy::F32 | NumTy::F64)).ok_or_else(fail)?;
            let (d, a) = (need!(r(0)), need!(r(1)));
            ext(ExtOp::Testp { kind, ty }, [d, ""], [a, NONE, NONE, NONE])
        }
        [name, ..] if float_op(name).is_some() => parse_float(body, &ops, &parts),
        _ => Ok(None),
    }
}

fn float_op(name: &str) -> Option<FOp> {
    Some(match name {
        "add" => FOp::Add,
        "sub" => FOp::Sub,
        "mul" => FOp::Mul,
        "fma" | "mad" => FOp::Fma,
        "min" => FOp::Min,
        "max" => FOp::Max,
        "neg" => FOp::Neg,
        "abs" => FOp::Abs,
        "ex2" => FOp::Ex2,
        "tanh" => FOp::Tanh,
        "rcp" => FOp::Rcp,
        "sqrt" => FOp::Sqrt,
        "rsqrt" => FOp::Rsqrt,
        "div" => FOp::Div,
        _ => return None,
    })
}

fn parse_float<'a>(
    body: &[&'a str],
    ops: &[Vec<&'a str>],
    parts: &[&str],
) -> Result<Option<Instruction<'a>>, Error> {
    let fail = || unsupported(body);
    let Some(op) = float_op(parts[0]) else {
        return Ok(None);
    };
    let Some(ty) = parts.last().and_then(|t| NumTy::parse(t)) else {
        return Ok(None);
    };
    if !matches!(ty, NumTy::F16 | NumTy::Bf16 | NumTy::F16x2 | NumTy::Bf16x2 | NumTy::F32 | NumTy::F64) {
        return Ok(None);
    }
    let mut mods = Mods::default();
    for part in &parts[1..parts.len() - 1] {
        match *part {
            "rn" | "ftz" | "approx" | "full" | "noftz" => {}
            "sat" => mods.sat = true,
            "relu" => mods.relu = true,
            "NaN" => mods.nan = true,
            _ if ty == NumTy::F32 => return Ok(None),
            _ => return Err(fail()),
        }
    }
    // f32 is mostly handled by the original table; only claim the new modifiers.
    if ty == NumTy::F32
        && !(matches!(op, FOp::Add | FOp::Sub | FOp::Mul | FOp::Fma | FOp::Min | FOp::Max)
            && (mods.sat || mods.nan))
    {
        return Ok(None);
    }
    let count = match op {
        FOp::Fma => 4,
        FOp::Neg | FOp::Abs | FOp::Ex2 | FOp::Tanh | FOp::Rcp | FOp::Sqrt | FOp::Rsqrt => 2,
        _ => 3,
    };
    if ops.len() != count || ops.iter().any(|o| o.len() != 1) {
        return Err(fail());
    }
    let mut src = ["0"; 4];
    for (slot, operand) in src.iter_mut().zip(&ops[1..]) {
        *slot = operand[0];
    }
    Ok(Some(Instruction::Ext {
        op: ExtOp::Float { op, ty, mods },
        dst: [ops[0][0], ""],
        src,
    }))
}

fn parse_cvt<'a>(
    body: &[&'a str],
    ops: &[Vec<&'a str>],
    parts: &[&str],
) -> Result<Option<Instruction<'a>>, Error> {
    let fail = || unsupported(body);
    // cvt [.rnd] [.ftz] [.sat] [.relu] [.satfinite] .dtype .stype
    if parts.len() < 3 {
        return Ok(None);
    }
    if let Some(fp8) = parse_fp8_cvt(ops, parts) {
        return fp8.map(Some).ok_or_else(fail);
    }
    let (Some(d), Some(s)) = (
        NumTy::parse(parts[parts.len() - 2]),
        NumTy::parse(parts[parts.len() - 1]),
    ) else {
        // cvt.pack and the fp4 / fp6 / ue8m0 forms are not implemented.
        return Ok(None);
    };
    let mut round = Round::None;
    let mut mods = Mods::default();
    for part in &parts[1..parts.len() - 2] {
        match *part {
            "rn" => round = Round::Rn,
            "rz" => round = Round::Rz,
            "rm" => round = Round::Rm,
            "rp" => round = Round::Rp,
            "rni" => round = Round::Rni,
            "rzi" => round = Round::Rzi,
            "rmi" => round = Round::Rmi,
            "rpi" => round = Round::Rpi,
            "rna" => round = Round::Rna,
            "ftz" => {}
            "sat" | "satfinite" => mods.sat = true,
            "relu" => mods.relu = true,
            _ => return Err(fail()),
        }
    }
    let both_int = !d.is_float() && !s.is_float();
    if both_int && !mods.sat {
        // Plain integer conversions are handled by the original table.
        return Ok(None);
    }
    let original = !mods.sat
        && !mods.relu
        && matches!(
            (d, s, round),
            (NumTy::F16, NumTy::F32, Round::Rn | Round::Rz)
                | (NumTy::F32, NumTy::F16, Round::None | Round::Rn)
                | (NumTy::F32, NumTy::U32 | NumTy::S32, Round::Rn)
                | (NumTy::F32, NumTy::S16, Round::Rn)
                | (NumTy::S32 | NumTy::U32, NumTy::F32, Round::Rni | Round::Rzi)
                | (NumTy::F32, NumTy::F32, Round::Rni | Round::Rzi | Round::Rmi | Round::Rpi)
        );
    if original {
        return Ok(None);
    }
    let want = if d.is_packed() { 3 } else { 2 };
    if ops.len() != want || ops.iter().any(|o| o.len() != 1) {
        return Err(fail());
    }
    let mut src = ["0"; 4];
    for (slot, operand) in src.iter_mut().zip(&ops[1..]) {
        *slot = operand[0];
    }
    Ok(Some(Instruction::Ext {
        op: ExtOp::Cvt { d, s, round, mods },
        dst: [ops[0][0], ""],
        src,
    }))
}

/// `Some(None)` for a malformed fp8 conversion, `None` when `parts` is not one.
fn parse_fp8_cvt<'a>(ops: &[Vec<&'a str>], parts: &[&str]) -> Option<Option<Instruction<'a>>> {
    let fp8 = |t: &str| match t {
        "e4m3x2" => Some(false),
        "e5m2x2" => Some(true),
        _ => None,
    };
    let (d, s) = (parts[parts.len() - 2], parts[parts.len() - 1]);
    let (dir, e5m2) = match (fp8(d), fp8(s)) {
        (Some(e5m2), None) if s == "f32" => (Fp8Dir::FromF32, e5m2),
        (Some(e5m2), None) if s == "f16x2" => (Fp8Dir::FromF16x2, e5m2),
        (None, Some(e5m2)) if d == "f16x2" => (Fp8Dir::ToF16x2, e5m2),
        _ => return None,
    };
    let modifiers = &parts[1..parts.len() - 2];
    let relu = modifiers.contains(&"relu");
    // Narrowing requires `.rn.satfinite`; widening is exact and takes `.rn`.
    let rounding_ok = modifiers.contains(&"rn")
        && (dir == Fp8Dir::ToF16x2 || modifiers.contains(&"satfinite"))
        && modifiers.iter().all(|m| matches!(*m, "rn" | "satfinite" | "relu"));
    let want = if dir == Fp8Dir::FromF32 { 3 } else { 2 };
    if !rounding_ok || ops.len() != want || ops.iter().any(|o| o.len() != 1) {
        return Some(None);
    }
    let mut src = ["0"; 4];
    for (slot, operand) in src.iter_mut().zip(&ops[1..]) {
        *slot = operand[0];
    }
    Some(Some(Instruction::Ext { op: ExtOp::Fp8 { dir, e5m2, relu }, dst: [ops[0][0], ""], src }))
}

fn fcmp(name: &str) -> Option<FCmp> {
    Some(match name {
        "eq" => FCmp::Eq,
        "ne" => FCmp::Ne,
        "lt" => FCmp::Lt,
        "le" => FCmp::Le,
        "gt" => FCmp::Gt,
        "ge" => FCmp::Ge,
        "equ" => FCmp::Equ,
        "neu" => FCmp::Neu,
        "ltu" => FCmp::Ltu,
        "leu" => FCmp::Leu,
        "gtu" => FCmp::Gtu,
        "geu" => FCmp::Geu,
        "num" => FCmp::Num,
        "nan" => FCmp::Nan,
        _ => return None,
    })
}

fn parse_fsetp<'a>(
    body: &[&'a str],
    ops: &[Vec<&'a str>],
    parts: &[&str],
) -> Result<Option<Instruction<'a>>, Error> {
    let fail = || unsupported(body);
    let Some(ty) = parts.last().and_then(|t| NumTy::parse(t)) else {
        return Ok(None);
    };
    if !matches!(ty, NumTy::F16 | NumTy::Bf16 | NumTy::F32 | NumTy::F64) || parts.len() < 3 {
        return Ok(None);
    }
    let Some(cmp) = fcmp(parts[1]) else {
        return Ok(None);
    };
    // Plain ordered f32 comparisons stay on the original path.
    let ordered = matches!(cmp, FCmp::Eq | FCmp::Ne | FCmp::Lt | FCmp::Le | FCmp::Gt | FCmp::Ge);
    if ty == NumTy::F32 && ordered && parts.len() <= 4 {
        return Ok(None);
    }
    if parts[2..parts.len() - 1].iter().any(|p| !matches!(*p, "ftz")) {
        return Err(fail());
    }
    if ops.len() != 3 || ops.iter().any(|o| o.len() != 1) {
        return Err(fail());
    }
    Ok(Some(Instruction::Ext {
        op: ExtOp::FSetp { cmp, ty },
        dst: [ops[0][0], ""],
        src: [ops[1][0], ops[2][0], "0", "0"],
    }))
}

fn parse_fset<'a>(
    body: &[&'a str],
    ops: &[Vec<&'a str>],
    parts: &[&str],
) -> Result<Option<Instruction<'a>>, Error> {
    let fail = || unsupported(body);
    // set.CMP[.ftz].dtype.stype d, a, b
    if parts.len() < 4 {
        return Ok(None);
    }
    let (Some(d), Some(s)) = (
        NumTy::parse(parts[parts.len() - 2]),
        NumTy::parse(parts[parts.len() - 1]),
    ) else {
        return Ok(None);
    };
    let Some(cmp) = fcmp(parts[1]) else {
        return Ok(None);
    };
    if !matches!(s, NumTy::F16 | NumTy::Bf16 | NumTy::F16x2 | NumTy::Bf16x2 | NumTy::F32 | NumTy::F64) {
        return Ok(None);
    }
    if ops.len() != 3 || ops.iter().any(|o| o.len() != 1) {
        return Err(fail());
    }
    Ok(Some(Instruction::Ext {
        op: ExtOp::FSet { cmp, d, s },
        dst: [ops[0][0], ""],
        src: [ops[1][0], ops[2][0], "0", "0"],
    }))
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
            "u64" | "b64" | "s64" | "f32" | "f64" | "u32" | "b32" | "s32" | "f16x2" | "bf16x2" => {
                ty = Some(*part)
            }
            "noftz" => {}
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
        "f16x2" if op == AtomicOp::Add => AtomTy::F16x2,
        "bf16x2" if op == AtomicOp::Add => AtomTy::Bf16x2,
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
