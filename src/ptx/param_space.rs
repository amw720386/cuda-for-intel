//! Kernel-parameter space accessed through a computed address.
//!
//! nvcc turns large by-value structs (ATen `TensorListMetadata`, `CatArrInputTensor`
//! tables) into `.param .align N .b8 p[bytes]` and addresses them with
//! `mov.u64 %rd, p; ... ld.param.u8 %r, [%rd+1536]`. The lowering keeps all kernel
//! parameters in a workgroup array laid out like the launch argument buffer and turns
//! `mov.u64 %rd, p` into the byte offset of `p` within it. Sub-word, 64-bit and vector
//! loads at any (possibly negative) immediate displacement are supported.
//! Emission lives in `param_space_emit.rs` (a child of `spirv.rs`).

use super::{Error, Instruction, Param};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum PTy {
    U8,
    S8,
    U16,
    S16,
    U32,
    S32,
    U64,
}

impl PTy {
    pub(super) fn bytes(self) -> u32 {
        match self {
            Self::U8 | Self::S8 => 1,
            Self::U16 | Self::S16 => 2,
            Self::U32 | Self::S32 => 4,
            Self::U64 => 8,
        }
    }

    pub(super) fn signed(self) -> bool {
        matches!(self, Self::S8 | Self::S16 | Self::S32)
    }

    fn parse(text: &str) -> Option<Self> {
        Some(match text {
            "u8" | "b8" => Self::U8,
            "s8" => Self::S8,
            "u16" | "b16" | "f16" => Self::U16,
            "s16" => Self::S16,
            "u32" | "b32" | "f32" => Self::U32,
            "s32" => Self::S32,
            "u64" | "b64" | "s64" | "f64" => Self::U64,
            _ => return None,
        })
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PLoad<'a> {
    pub(super) dst: [&'a str; 4],
    pub(super) count: u8,
    /// Register holding the param-space address, or the parameter name when `!dynamic`.
    pub(super) base: &'a str,
    pub(super) ty: PTy,
    pub(super) off: i32,
    pub(super) dynamic: bool,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum ParamOp<'a> {
    /// `mov.u64 %rd, param_name`: byte offset of the parameter in param space.
    Addr { dst: &'a str, name: &'a str },
    Load(PLoad<'a>),
}

impl ParamOp<'_> {
    pub(super) fn is_dynamic(&self) -> bool {
        matches!(self, Self::Addr { .. } | Self::Load(PLoad { dynamic: true, .. }))
    }
}

/// `[sym]`, `[sym+off]`, `[%r+off]`, `[%r+-off]`, `[%r-off]`.
fn split_address(text: &str) -> Option<(&str, i32)> {
    if let Some(at) = text[1.min(text.len())..].find(['+', '-']) {
        let at = at + 1.min(text.len());
        let (base, rest) = text.split_at(at);
        let rest = rest.strip_prefix('+').unwrap_or(rest);
        let value = if let Some(hex) = rest.strip_prefix("0x") {
            i32::from_str_radix(hex, 16).ok()?
        } else {
            rest.parse::<i32>().ok()?
        };
        Some((base, value))
    } else {
        Some((text, 0))
    }
}

pub(super) fn parse<'a>(body: &[&'a str], names: &[&'a str]) -> Result<Option<Instruction<'a>>, Error> {
    let Some(&op) = body.first() else {
        return Ok(None);
    };
    if matches!(op, "mov.u64" | "mov.b64") {
        if let [_, dst, ",", src] = body
            && names.contains(src)
        {
            return Ok(Some(Instruction::ParamSpace(ParamOp::Addr { dst, name: src })));
        }
        return Ok(None);
    }
    let Some(rest) = op.strip_prefix("ld.param.") else {
        return Ok(None);
    };
    let (count, ty) = match rest.split_once('.') {
        Some(("v2", ty)) => (2u8, ty),
        Some(("v4", ty)) => (4u8, ty),
        Some(_) => return Ok(None),
        None => (1u8, rest),
    };
    let Some(pty) = PTy::parse(ty) else {
        return Ok(None);
    };
    let mut dst = [""; 4];
    let tail: &[&'a str] = if count == 1 {
        let [_, d, ",", rest @ ..] = body else {
            return Ok(None);
        };
        dst[0] = d;
        rest
    } else {
        let mut at = 2; // body[1] == "{"
        if body.get(1) != Some(&"{") {
            return Ok(None);
        }
        for slot in dst.iter_mut().take(usize::from(count)) {
            *slot = body.get(at).copied().ok_or_else(|| Error::new("malformed ld.param vector"))?;
            at += 1;
            if body.get(at) == Some(&",") {
                at += 1;
            }
        }
        if body.get(at) != Some(&"}") || body.get(at + 1) != Some(&",") {
            return Err(Error::new(format!("unsupported PTX instruction: {}", body.join(" "))));
        }
        &body[at + 2..]
    };
    let ["[", address, "]"] = tail else {
        return Ok(None);
    };
    let Some((base, off)) = split_address(address) else {
        return Err(Error::new("invalid PTX parameter offset"));
    };
    let dynamic = base.starts_with('%');
    if !dynamic {
        // Spellings the legacy table already lowers stay there.
        let legacy = count == 1 && matches!(ty, "u32" | "b32" | "s32" | "f32" | "u64" | "b64" | "s8")
            || count > 1 && ty == "f32";
        if legacy || off < 0 {
            return Ok(None);
        }
    }
    Ok(Some(Instruction::ParamSpace(ParamOp::Load(PLoad {
        dst,
        count,
        base,
        ty: pty,
        off,
        dynamic,
    }))))
}

/// Byte offset of every parameter in the launch argument buffer and the total size.
pub(super) fn layout(params: &[(&str, Param)]) -> (Vec<u32>, u32) {
    let mut offsets = Vec::with_capacity(params.len());
    let mut bytes = 0usize;
    for &(_, param) in params {
        bytes = bytes.next_multiple_of(param.alignment());
        offsets.push(bytes as u32);
        bytes += param.size();
    }
    (offsets, bytes as u32)
}
