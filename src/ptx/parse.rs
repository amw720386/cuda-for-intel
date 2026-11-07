use super::{
    Block, Comparison, Entry, Error, FloatOp, Instruction, IntegerOp, Param, Space, Terminator,
    Width,
};
use std::collections::HashMap;

pub(super) fn tokens(mut source: &str) -> Result<Vec<&str>, Error> {
    fn punctuation(c: char) -> bool {
        "(){}[],;:<>@!".contains(c)
    }

    let mut tokens = Vec::new();
    loop {
        // nvcc fatbin PTX often pads with NULs after the last kernel.
        source = source.trim_start_matches(|c: char| c.is_ascii_whitespace() || c == '\0');
        let Some(first) = source.chars().next() else {
            return Ok(tokens);
        };
        if let Some(comment) = source.strip_prefix("//") {
            source = comment.find('\n').map_or("", |end| &comment[end + 1..]);
        } else if let Some(comment) = source.strip_prefix("/*") {
            let end = comment
                .find("*/")
                .ok_or_else(|| Error::new("unterminated PTX comment"))?;
            source = &comment[end + 2..];
        } else {
            let end = if punctuation(first) {
                1
            } else {
                source
                    .find(|c: char| c.is_ascii_whitespace() || punctuation(c) || c == '/')
                    .unwrap_or(source.len())
            };
            if end == 0 {
                return Err(Error::new("unexpected slash in PTX"));
            }
            let (token, rest) = source.split_at(end);
            tokens.push(token);
            source = rest;
        }
    }
}

struct Parser<'a> {
    tokens: std::vec::IntoIter<&'a str>,
}

impl<'a> Parser<'a> {
    fn peek(&self) -> Option<&'a str> {
        self.tokens.as_slice().first().copied()
    }

    fn take(&mut self) -> Result<&'a str, Error> {
        self.tokens
            .next()
            .ok_or_else(|| Error::new("unexpected end of PTX"))
    }

    fn expect(&mut self, expected: &str) -> Result<(), Error> {
        let token = self.take()?;
        if token != expected {
            return Err(Error::new(format!("expected {expected}, got {token}")));
        }
        Ok(())
    }

    fn eat(&mut self, token: &str) -> bool {
        if self.peek() == Some(token) {
            self.tokens.next();
            true
        } else {
            false
        }
    }

    fn identifier(&mut self) -> Result<&'a str, Error> {
        let name = self.take()?;
        let mut bytes = name.bytes();
        let valid = |b: u8| b.is_ascii_alphabetic() || matches!(b, b'_' | b'$' | b'%');
        if !bytes.next().is_some_and(valid) || !bytes.all(|b| valid(b) || b.is_ascii_digit()) {
            return Err(Error::new(format!("invalid PTX identifier {name}")));
        }
        Ok(name)
    }

    fn param(&mut self) -> Result<(&'a str, Param), Error> {
        self.expect(".param")?;
        let ty = match self.take()? {
            ".align" => {
                let align = self
                    .take()?
                    .parse::<u16>()
                    .ok()
                    .filter(|n| matches!(n, 4 | 8 | 16))
                    .ok_or_else(|| Error::new("unsupported PTX parameter alignment"))?;
                self.expect(".b8")?;
                let name = self.identifier()?;
                self.expect("[")?;
                let bytes = self
                    .take()?
                    .parse::<u16>()
                    .ok()
                    .filter(|n| {
                        (4..=4096).contains(n) && n.is_multiple_of(4) && n.is_multiple_of(align)
                    })
                    .ok_or_else(|| Error::new("unsupported PTX parameter aggregate size"))?;
                self.expect("]")?;
                return Ok((
                    name,
                    Param::Words {
                        words: bytes / 4,
                        align,
                    },
                ));
            }
            ".u8" | ".s8" => Param::U8,
            ".u32" | ".b32" => Param::U32,
            ".f32" => Param::F32,
            ".u64" | ".b64" if self.eat(".ptr") => {
                // Generic pointer parameters are restricted to device memory at launch.
                self.eat(".global");
                if self.eat(".align") && !matches!(self.take()?, "1" | "2" | "4") {
                    return Err(Error::new("unsupported PTX pointer alignment"));
                }
                Param::GlobalPtr
            }
            ".u64" | ".b64" => Param::U64,
            ty => return Err(Error::new(format!("unsupported PTX parameter type {ty}"))),
        };
        Ok((self.identifier()?, ty))
    }

    fn params(&mut self) -> Result<Vec<(&'a str, Param)>, Error> {
        let mut params = Vec::new();
        let mut names = std::collections::HashSet::new();
        let mut bytes = 0usize;
        self.expect("(")?;
        if self.eat(")") {
            return Ok(params);
        }
        loop {
            let (name, ty) = self.param()?;
            if !names.insert(name) {
                return Err(Error::new(format!("duplicate PTX parameter {name}")));
            }
            bytes = bytes.next_multiple_of(ty.alignment()) + ty.size();
            // Bound aggregate expansion before constructing the SPIR-V signature.
            if bytes > 32_764 {
                return Err(Error::new("PTX parameter list is too large"));
            }
            params.push((name, ty));
            if self.eat(")") {
                return Ok(params);
            }
            self.expect(",")?;
        }
    }

    fn registers(&mut self) -> Result<HashMap<String, Width>, Error> {
        let mut registers = HashMap::new();
        while self.eat(".reg") {
            let width = match self.take()? {
                ".pred" => Width::Predicate,
                // 16-bit regs are tracked as B32 slots holding zero-extended values.
                ".b16" | ".u16" | ".s16" | ".b32" | ".u32" | ".s32" | ".f32" => Width::B32,
                ".b64" | ".u64" | ".s64" => Width::B64,
                ty => return Err(Error::new(format!("unsupported PTX register type {ty}"))),
            };
            let name = self.identifier()?;
            if !name.starts_with('%') || name.len() < 2 {
                return Err(Error::new("expected a percent-prefixed PTX register"));
            }
            if self.eat("<") {
                let count = self
                    .take()?
                    .parse::<usize>()
                    .ok()
                    .filter(|&n| (1..=4096 - registers.len()).contains(&n))
                    .ok_or_else(|| Error::new("unsupported PTX register range"))?;
                self.expect(">")?;
                for index in 0..count {
                    if registers.insert(format!("{name}{index}"), width).is_some() {
                        return Err(Error::new("duplicate PTX register"));
                    }
                }
            } else {
                if registers.len() == 4096 {
                    return Err(Error::new("too many PTX registers"));
                }
                if registers.insert(name.to_owned(), width).is_some() {
                    return Err(Error::new("duplicate PTX register"));
                }
            }
            self.expect(";")?;
        }
        Ok(registers)
    }

    fn instruction(&mut self) -> Result<Instruction<'a>, Error> {
        let slice = self.tokens.as_slice();
        let end = slice
            .iter()
            .position(|&token| token == ";")
            .ok_or_else(|| Error::new("unterminated PTX instruction"))?;
        // nvcc wraps some helpers (notably __half2float) as `{ inst; }`.
        let braced = slice.first() == Some(&"{");
        let body = if braced { &slice[1..end] } else { &slice[..end] };
        let result = match body {
            [
                "ld.param.b32" | "ld.param.u32" | "ld.param.f32",
                dst,
                ",",
                "[",
                src,
                "]",
            ] => {
                let (name, offset) = if let Some((name, offset)) = src.split_once('+') {
                    (
                        name,
                        offset
                            .parse()
                            .map_err(|_| Error::new("invalid PTX parameter offset"))?,
                    )
                } else {
                    (*src, 0)
                };
                Instruction::Param(Width::B32, dst, name, offset)
            }
            ["ld.param.s8", dst, ",", "[", src, "]"] => Instruction::ParamS8(dst, src),
            ["ld.param.b64" | "ld.param.u64", dst, ",", "[", src, "]"] => {
                let (name, offset) = if let Some((name, offset)) = src.split_once('+') {
                    (
                        name,
                        offset
                            .parse()
                            .map_err(|_| Error::new("invalid PTX parameter offset"))?,
                    )
                } else {
                    (*src, 0)
                };
                Instruction::Param(Width::B64, dst, name, offset)
            }
            [
                op @ ("ld.param.v2.f32" | "ld.param.v4.f32"),
                "{",
                a,
                ",",
                b,
                rest @ ..,
            ] => {
                let count = if *op == "ld.param.v4.f32" { 4u8 } else { 2 };
                let (c, d, src) = if count == 4 {
                    match rest {
                        [",", c, ",", d, "}", ",", "[", src, "]"] => (*c, *d, *src),
                        _ => {
                            return Err(Error::new(format!(
                                "unsupported PTX instruction: {}",
                                slice[..end].join(" ")
                            )));
                        }
                    }
                } else {
                    match rest {
                        ["}", ",", "[", src, "]"] => ("", "", *src),
                        _ => {
                            return Err(Error::new(format!(
                                "unsupported PTX instruction: {}",
                                slice[..end].join(" ")
                            )));
                        }
                    }
                };
                let (name, offset) = if let Some((name, offset)) = src.split_once('+') {
                    (
                        name,
                        offset
                            .parse()
                            .map_err(|_| Error::new("invalid PTX parameter offset"))?,
                    )
                } else {
                    (src, 0)
                };
                Instruction::ParamVec {
                    count,
                    dst: [a, b, c, d],
                    name,
                    offset,
                }
            }
            ["cvta.to.global.u64", dst, ",", src] => Instruction::ToGlobal(dst, src),
            ["mov.b64", dst, ",", src]
                if !src.starts_with('%')
                    && !src.starts_with("0x")
                    && !src.starts_with("0X")
                    && src.parse::<i64>().is_err() =>
            {
                // `mov.b64 %rd, shared_symbol` — shared base address.
                Instruction::SharedAddress(dst, src)
            }
            ["bar.sync", "0"] => Instruction::Barrier,
            [
                "shfl.sync.bfly.b32",
                dst,
                ",",
                value,
                ",",
                xor_mask,
                ",",
                clamp,
                ",",
                membermask,
            ] => {
                let (dst, pred) = match dst.split_once('|') {
                    Some((dst, pred)) => (dst, Some(pred)),
                    None => (*dst, None),
                };
                Instruction::ShuffleBfly {
                    dst,
                    pred,
                    src: [value, xor_mask, clamp, membermask],
                }
            }
            [
                "atom.global.add.u32",
                dst,
                ",",
                "[",
                address,
                "]",
                ",",
                value,
            ] => Instruction::AtomicAdd32(dst, address, value),
            ["mov.u32", dst, ",", src] => {
                if src.contains('.') {
                    Instruction::Builtin(dst, src)
                } else if !src.starts_with('%')
                    && !src.starts_with("0x")
                    && !src.starts_with("0X")
                    && src.parse::<i64>().is_err()
                {
                    // `mov.u32 %r, data_soft_max_f32` — address of extern/dynamic shared.
                    Instruction::SharedSymbol32(dst, src)
                } else {
                    Instruction::Move32(dst, src)
                }
            }
            ["mov.u64", dst, ",", src]
                if !src.starts_with('%')
                    && !src.starts_with("0x")
                    && !src.starts_with("0X")
                    && src.parse::<i64>().is_err() =>
            {
                // `mov.u64 %SPL, __local_depotN` — address of CUDA .local depot.
                Instruction::LocalSymbol64(dst, src)
            }
            ["mov.u64" | "mov.b64", dst, ",", src] => Instruction::Move64(dst, src),
            ["mad.lo.s32" | "mad.lo.u32", dst, ",", a, ",", b, ",", c] => {
                Instruction::Mad32(dst, [a, b, c])
            }
            ["dp4a.s32.s32", dst, ",", a, ",", b, ",", c] => Instruction::Dp4a(dst, [a, b, c]),
            [
                op @ ("setp.ge.u32" | "setp.gt.u32" | "setp.lt.u32" | "setp.le.u32" | "setp.eq.u32"
                | "setp.ne.u32" | "setp.eq.b32" | "setp.ne.b32" | "setp.ge.s32"
                | "setp.gt.s32" | "setp.lt.s32" | "setp.le.s32" | "setp.eq.s32"
                | "setp.ne.s32" | "setp.eq.s16" | "setp.ne.s16" | "setp.lt.s16"
                | "setp.le.s16" | "setp.gt.s16" | "setp.ge.s16"),
                dst,
                ",",
                a,
                ",",
                b,
            ] => {
                let bits16 = op.ends_with(".s16") || op.ends_with(".u16");
                let signed = op.ends_with(".s32") || op.ends_with(".s16");
                let comparison = match &op[5..7] {
                    "ge" => Comparison::GreaterEqual,
                    "gt" => Comparison::Greater,
                    "lt" => Comparison::Less,
                    "le" => Comparison::LessEqual,
                    "eq" => Comparison::Equal,
                    _ => Comparison::NotEqual,
                };
                Instruction::Compare {
                    op: comparison,
                    signed,
                    bits16,
                    dst,
                    src: [a, b],
                }
            }
            [
                op @ ("setp.eq.s64" | "setp.ne.s64" | "setp.lt.s64" | "setp.le.s64"
                | "setp.gt.s64" | "setp.ge.s64" | "setp.eq.u64" | "setp.ne.u64"
                | "setp.lt.u64" | "setp.le.u64" | "setp.gt.u64" | "setp.ge.u64"),
                dst,
                ",",
                a,
                ",",
                b,
            ] => {
                let signed = op.ends_with(".s64");
                let comparison = match &op[5..7] {
                    "ge" => Comparison::GreaterEqual,
                    "gt" => Comparison::Greater,
                    "lt" => Comparison::Less,
                    "le" => Comparison::LessEqual,
                    "eq" => Comparison::Equal,
                    _ => Comparison::NotEqual,
                };
                Instruction::Compare64 {
                    op: comparison,
                    signed,
                    dst,
                    src: [a, b],
                }
            }
            [
                op @ ("selp.b32" | "selp.u32" | "selp.s32" | "selp.f32" | "selp.b64"
                | "selp.u64" | "selp.s64"),
                dst,
                ",",
                a,
                ",",
                b,
                ",",
                pred,
            ] => {
                let kind = if op.ends_with(".b64") || op.ends_with(".u64") || op.ends_with(".s64")
                {
                    Width::B64
                } else {
                    Width::B32
                };
                Instruction::Select {
                    kind,
                    dst,
                    src: [a, b, pred],
                }
            }
            [
                op @ ("setp.eq.f32" | "setp.eq.ftz.f32" | "setp.ne.f32" | "setp.ne.ftz.f32"
                | "setp.lt.f32" | "setp.lt.ftz.f32" | "setp.le.f32" | "setp.le.ftz.f32"
                | "setp.gt.f32" | "setp.gt.ftz.f32" | "setp.ge.f32" | "setp.ge.ftz.f32"),
                dst,
                ",",
                a,
                ",",
                b,
            ] => {
                let comparison = match &op[5..7] {
                    "ge" => Comparison::GreaterEqual,
                    "gt" => Comparison::Greater,
                    "lt" => Comparison::Less,
                    "le" => Comparison::LessEqual,
                    "eq" => Comparison::Equal,
                    _ => Comparison::NotEqual,
                };
                Instruction::CompareFloat {
                    op: comparison,
                    dst,
                    src: [a, b],
                }
            }
            [op @ ("and.pred" | "or.pred"), dst, ",", a, ",", b] => Instruction::PredicateLogic {
                and: *op == "and.pred",
                dst,
                src: [a, b],
            },
            ["not.pred", dst, ",", src] => Instruction::NotPred(dst, src),
            ["neg.s32", dst, ",", src] => Instruction::Integer32(IntegerOp::Sub, dst, ["0", src]),
            [
                op @ ("add.s32" | "sub.s32" | "mul.lo.s32" | "mul.hi.u32" | "shl.b32"
                | "shr.u32" | "shr.s32" | "and.b32" | "or.b32" | "div.s32" | "div.u32"
                | "rem.s32" | "rem.u32"),
                dst,
                ",",
                a,
                ",",
                b,
            ] => {
                let op = match *op {
                    "add.s32" => IntegerOp::Add,
                    "sub.s32" => IntegerOp::Sub,
                    "mul.lo.s32" => IntegerOp::Mul,
                    "and.b32" => IntegerOp::And,
                    "or.b32" => IntegerOp::Or,
                    "shl.b32" => IntegerOp::ShiftLeft,
                    "mul.hi.u32" => IntegerOp::MulHighUnsigned,
                    "shr.s32" => IntegerOp::ShiftRightArithmetic,
                    "div.s32" => IntegerOp::Div,
                    "div.u32" => IntegerOp::DivUnsigned,
                    "rem.s32" => IntegerOp::Rem,
                    "rem.u32" => IntegerOp::RemUnsigned,
                    _ => IntegerOp::ShiftRightLogical,
                };
                Instruction::Integer32(op, dst, [a, b])
            }
            ["and.b64", dst, ",", a, ",", b] => Instruction::And64(dst, [a, b]),
            ["or.b64", dst, ",", a, ",", b] => Instruction::Or64(dst, [a, b]),
            ["shr.u64", dst, ",", a, ",", b] => Instruction::ShiftRightLogical64(dst, [a, b]),
            ["rem.s64", dst, ",", a, ",", b] => Instruction::Rem64 {
                signed: true,
                dst,
                src: [a, b],
            },
            ["rem.u64", dst, ",", a, ",", b] => Instruction::Rem64 {
                signed: false,
                dst,
                src: [a, b],
            },
            ["cvt.s32.s8", dst, ",", src] => {
                Instruction::Integer32(IntegerOp::SignExtend8, dst, [src, src])
            }
            ["cvt.u16.u32", dst, ",", src] => {
                Instruction::Integer32(IntegerOp::ToU16, dst, [src, src])
            }
            ["cvt.s16.s8", dst, ",", src] => {
                Instruction::Integer32(IntegerOp::SignExtend8To16, dst, [src, src])
            }
            ["shr.s16", dst, ",", a, ",", b] => {
                Instruction::Integer32(IntegerOp::ShiftRightArithmetic16, dst, [a, b])
            }
            ["not.b32", dst, ",", src] => Instruction::Not32(dst, src),
            ["not.b64" | "not.u64" | "not.s64", dst, ",", src] => Instruction::Not64(dst, src),
            ["mov.b32" | "mov.f32", dst, ",", src] => Instruction::Move32(dst, src),
            ["mov.b32", dst, ",", "{", lo, ",", hi, "}"] => Instruction::PackB16(dst, [lo, hi]),
            ["cvt.u64.u32", dst, ",", src] => Instruction::Widen {
                signed: false,
                dst,
                src,
            },
            ["cvt.s64.s32", dst, ",", src] => Instruction::Widen {
                signed: true,
                dst,
                src,
            },
            ["cvt.u32.u64", dst, ",", src] => Instruction::Narrow(dst, src),
            ["mul.wide.u32", dst, ",", a, ",", b] => Instruction::MulWide32 {
                signed: false,
                dst,
                src: [a, b],
            },
            ["mul.wide.s32", dst, ",", a, ",", b] => Instruction::MulWide32 {
                signed: true,
                dst,
                src: [a, b],
            },
            ["shl.b64", dst, ",", a, ",", b] => Instruction::ShiftLeft64(dst, [a, b]),
            ["add.s64" | "add.u64", dst, ",", a, ",", b] => Instruction::Add64(dst, [a, b]),
            ["sub.s64" | "sub.u64", dst, ",", a, ",", b] => Instruction::Sub64(dst, [a, b]),
            ["shr.s64", dst, ",", a, ",", b] => Instruction::ShiftRightArithmetic64(dst, [a, b]),
            ["redux.sync.add.s32", dst, ",", src, ",", membermask] => Instruction::ReduxSyncAdd {
                dst,
                src,
                membermask,
            },
            ["mul.lo.s64" | "mul.lo.u64", dst, ",", a, ",", b] => {
                Instruction::Mul64(dst, [a, b])
            }
            ["div.s64" | "div.u64", dst, ",", a, ",", b] => Instruction::Div64 {
                signed: body[0] == "div.s64",
                dst,
                src: [a, b],
            },
            [
                op @ ("ld.global.b32"
                | "ld.global.u32"
                | "ld.global.f32"
                | "ld.global.nc.u32"
                | "ld.global.nc.f32"
                | "ld.shared.b32"
                | "ld.shared.f32"
                | "ld.shared.u32"
                | "ld.local.b32"
                | "ld.local.u32"
                | "ld.local.f32"),
                dst,
                ",",
                "[",
                addr,
                "]",
            ] => {
                let space = if op.starts_with("ld.shared.") {
                    Space::Shared
                } else if op.starts_with("ld.local.") {
                    Space::Local
                } else {
                    Space::Global
                };
                Instruction::Load32(space, dst, addr)
            }
            [
                "ld.global.u64" | "ld.global.b64" | "ld.global.nc.u64" | "ld.global.nc.b64",
                dst,
                ",",
                "[",
                addr,
                "]",
            ] => Instruction::Load64(Space::Global, dst, addr),
            [
                op @ ("ld.global.u16" | "ld.global.b16" | "ld.global.nc.u16" | "ld.global.nc.b16"),
                dst,
                ",",
                "[",
                addr,
                "]",
            ] => {
                let _ = op;
                Instruction::Load16(Space::Global, dst, addr)
            }
            [
                op @ ("ld.global.u8" | "ld.global.b8" | "ld.global.nc.u8" | "ld.global.nc.b8"),
                dst,
                ",",
                "[",
                addr,
                "]",
            ] => {
                let _ = op;
                Instruction::Load8(Space::Global, dst, addr)
            }
            [
                op @ ("ld.global.v2.u16"
                | "ld.global.nc.v2.u16"
                | "ld.global.v2.b16"
                | "ld.global.nc.v2.b16"
                | "ld.global.v4.u16"
                | "ld.global.nc.v4.u16"
                | "ld.global.v4.b16"
                | "ld.global.nc.v4.b16"),
                "{",
                a,
                ",",
                b,
                rest @ ..,
            ] => {
                let count = if op.contains("v4") { 4u8 } else { 2 };
                let (c, d, addr) = if count == 4 {
                    match rest {
                        [",", c, ",", d, "}", ",", "[", addr, "]"] => (*c, *d, *addr),
                        _ => {
                            return Err(Error::new(format!(
                                "unsupported PTX instruction: {}",
                                slice[..end].join(" ")
                            )));
                        }
                    }
                } else {
                    match rest {
                        ["}", ",", "[", addr, "]"] => ("", "", *addr),
                        _ => {
                            return Err(Error::new(format!(
                                "unsupported PTX instruction: {}",
                                slice[..end].join(" ")
                            )));
                        }
                    }
                };
                Instruction::LoadVecU16 {
                    space: Space::Global,
                    count,
                    dst: [a, b, c, d],
                    addr,
                }
            }
            [
                op @ ("ld.global.v2.f32"
                | "ld.global.nc.v2.f32"
                | "ld.global.v4.f32"
                | "ld.global.nc.v4.f32"
                | "ld.shared.v2.f32"
                | "ld.local.v2.f32"),
                "{",
                a,
                ",",
                b,
                rest @ ..,
            ] => {
                let (count, space) = match *op {
                    "ld.global.v4.f32" | "ld.global.nc.v4.f32" => (4u8, Space::Global),
                    "ld.global.v2.f32" | "ld.global.nc.v2.f32" => (2, Space::Global),
                    "ld.local.v2.f32" => (2, Space::Local),
                    _ => (2, Space::Shared),
                };
                let (c, d, addr) = if count == 4 {
                    match rest {
                        [",", c, ",", d, "}", ",", "[", addr, "]"] => (*c, *d, *addr),
                        _ => {
                            return Err(Error::new(format!(
                                "unsupported PTX instruction: {}",
                                slice[..end].join(" ")
                            )));
                        }
                    }
                } else {
                    match rest {
                        ["}", ",", "[", addr, "]"] => ("", "", *addr),
                        _ => {
                            return Err(Error::new(format!(
                                "unsupported PTX instruction: {}",
                                slice[..end].join(" ")
                            )));
                        }
                    }
                };
                Instruction::LoadVec {
                    space,
                    count,
                    dst: [a, b, c, d],
                    addr,
                }
            }
            [
                "add.f32" | "add.rn.f32" | "add.ftz.f32" | "add.rz.ftz.f32" | "add.rz.f32",
                dst,
                ",",
                a,
                ",",
                b,
            ] => Instruction::Float32(dst, FloatOp::Add([a, b])),
            ["abs.ftz.f32" | "abs.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::Abs([src]))
            }
            ["copysign.f32", dst, ",", sign, ",", magnitude] => {
                Instruction::Float32(dst, FloatOp::CopySign([sign, magnitude]))
            }
            ["neg.ftz.f32" | "neg.f32" | "neg.rn.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::Neg([src]))
            }
            ["tanh.approx.f32" | "tanh.approx.ftz.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::TanhApprox([src]))
            }
            ["cvt.rzi.f32.f32" | "cvt.rzi.ftz.f32.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::Trunc([src]))
            }
            ["cvt.rzi.ftz.s32.f32" | "cvt.rzi.s32.f32", dst, ",", src] => {
                Instruction::CvtRziS32(dst, src)
            }
            ["cvt.rn.f16.f32" | "cvt.rz.f16.f32", dst, ",", src] => Instruction::CvtToHalf(dst, src),
            ["sub.f32" | "sub.rn.f32" | "sub.ftz.f32", dst, ",", a, ",", b] => {
                Instruction::Float32(dst, FloatOp::Sub([a, b]))
            }
            [
                op @ ("min.f32" | "max.f32" | "min.ftz.f32" | "max.ftz.f32"),
                dst,
                ",",
                a,
                ",",
                b,
            ] => Instruction::Float32(
                dst,
                FloatOp::MinMax {
                    min: op.starts_with("min."),
                    src: [a, b],
                },
            ),
            ["mul.f32" | "mul.rn.f32" | "mul.ftz.f32", dst, ",", a, ",", b] => {
                Instruction::Float32(dst, FloatOp::Mul([a, b]))
            }
            ["fma.rn.f32" | "fma.rn.ftz.f32", dst, ",", a, ",", b, ",", c] => {
                Instruction::Float32(dst, FloatOp::Fma([a, b, c]))
            }
            ["cvt.rn.f32.u32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::FromUnsigned([src]))
            }
            ["cvt.rn.f32.s32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::FromSigned([src]))
            }
            ["cvt.f32.f16" | "cvt.rn.f32.f16", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::FromHalf([src]))
            }
            ["cvt.rn.f32.s16", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::FromSigned16([src]))
            }
            [
                "div.rn.f32" | "div.approx.f32" | "div.approx.ftz.f32" | "div.ftz.f32",
                dst,
                ",",
                a,
                ",",
                b,
            ] => Instruction::Float32(dst, FloatOp::Div([a, b])),
            ["sqrt.rn.f32" | "sqrt.approx.f32" | "sqrt.approx.ftz.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::Sqrt([src]))
            }
            ["rsqrt.approx.f32" | "rsqrt.approx.ftz.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::RsqrtApprox([src]))
            }
            ["rcp.approx.f32" | "rcp.approx.ftz.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::RcpApprox([src]))
            }
            ["sin.approx.f32" | "sin.approx.ftz.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::SinApprox([src]))
            }
            ["cos.approx.f32" | "cos.approx.ftz.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::CosApprox([src]))
            }
            ["lg2.approx.f32" | "lg2.approx.ftz.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::Lg2Approx([src]))
            }
            ["ex2.approx.f32" | "ex2.approx.ftz.f32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::Ex2Approx([src]))
            }
            [
                "st.global.u8" | "st.global.b8" | "st.shared.u8" | "st.shared.b8",
                "[",
                addr,
                "]",
                ",",
                src,
            ] => {
                let space = if body[0].starts_with("st.shared") {
                    Space::Shared
                } else {
                    Space::Global
                };
                Instruction::Store8(space, addr, src)
            }
            [
                "st.global.u16" | "st.global.b16",
                "[",
                addr,
                "]",
                ",",
                src,
            ] => Instruction::Store16(Space::Global, addr, src),
            [
                "st.global.u64" | "st.global.b64",
                "[",
                addr,
                "]",
                ",",
                src,
            ] => Instruction::Store64(Space::Global, addr, src),
            [
                "st.global.v2.u16" | "st.global.v2.b16",
                "[",
                addr,
                "]",
                ",",
                "{",
                a,
                ",",
                b,
                "}",
            ] => Instruction::StoreVecU16(Space::Global, addr, [a, b]),
            [
                op @ ("st.global.v4.u16" | "st.global.v4.b16" | "st.global.v4.u8"),
                "[",
                addr,
                "]",
                ",",
                "{",
                a,
                ",",
                b,
                ",",
                c,
                ",",
                d,
                "}",
            ] => {
                if *op == "st.global.v4.u8" {
                    Instruction::StoreVec4U8(addr, [a, b, c, d])
                } else {
                    Instruction::StoreVec4U16(Space::Global, addr, [a, b, c, d])
                }
            }
            [
                op @ ("st.global.b32"
                | "st.global.u32"
                | "st.global.f32"
                | "st.shared.b32"
                | "st.shared.f32"
                | "st.shared.u32"
                | "st.local.b32"
                | "st.local.u32"
                | "st.local.f32"),
                "[",
                addr,
                "]",
                ",",
                src,
            ] => {
                let space = if op.starts_with("st.shared.") {
                    Space::Shared
                } else if op.starts_with("st.local.") {
                    Space::Local
                } else {
                    Space::Global
                };
                Instruction::Store32(space, addr, src)
            }
            [
                op @ ("st.global.v2.f32" | "st.global.v2.u32" | "st.global.v4.f32"
                | "st.shared.v2.f32"
                | "st.local.v2.f32"),
                "[",
                addr,
                "]",
                ",",
                "{",
                a,
                ",",
                b,
                rest @ ..,
            ] => {
                // v2.u32 shares the float-vector store path; bit patterns are preserved.
                let (count, space) = match *op {
                    "st.global.v4.f32" => (4u8, Space::Global),
                    "st.global.v2.f32" | "st.global.v2.u32" => (2, Space::Global),
                    "st.local.v2.f32" => (2, Space::Local),
                    _ => (2, Space::Shared),
                };
                let (c, d) = if count == 4 {
                    match rest {
                        [",", c, ",", d, "}"] => (*c, *d),
                        _ => {
                            return Err(Error::new(format!(
                                "unsupported PTX instruction: {}",
                                slice[..end].join(" ")
                            )));
                        }
                    }
                } else {
                    match rest {
                        ["}"] => ("", ""),
                        _ => {
                            return Err(Error::new(format!(
                                "unsupported PTX instruction: {}",
                                slice[..end].join(" ")
                            )));
                        }
                    }
                };
                Instruction::StoreVec {
                    space,
                    count,
                    addr,
                    src: [a, b, c, d],
                }
            }
            _ => {
                return Err(Error::new(format!(
                    "unsupported PTX instruction: {}",
                    slice[..end].join(" ")
                )));
            }
        };
        self.tokens.nth(end);
        if braced {
            self.expect("}")?;
        }
        Ok(result)
    }

    fn label(&self) -> bool {
        self.tokens.as_slice().get(1) == Some(&":")
    }

    fn blocks(&mut self) -> Result<Vec<Block<'a>>, Error> {
        enum End<'a> {
            Return,
            Jump(&'a str),
            Branch(&'a str, bool, &'a str),
            Fallthrough,
        }
        let mut labels = HashMap::new();
        let mut blocks = Vec::new();
        while !self.eat("}") {
            if blocks.len() == 512 {
                return Err(Error::new("too many PTX blocks"));
            }
            while self.label() {
                let label = self.identifier()?;
                self.expect(":")?;
                if labels.insert(label, blocks.len()).is_some() {
                    return Err(Error::new(format!("duplicate PTX label {label}")));
                }
            }
            let mut instructions = Vec::new();
            let end = loop {
                if self.label() || self.peek() == Some("}") {
                    break End::Fallthrough;
                }
                if self.eat(".pragma") {
                    self.expect("\"nounroll\"")?;
                    self.expect(";")?;
                    continue;
                }
                let end = if self.eat("ret") {
                    End::Return
                } else if self.eat("@") {
                    let invert = self.eat("!");
                    let predicate = self.identifier()?;
                    if !self.eat("bra") {
                        self.expect("bra.uni")?;
                    }
                    End::Branch(predicate, invert, self.identifier()?)
                } else if self.eat("bra") || self.eat("bra.uni") {
                    End::Jump(self.identifier()?)
                } else {
                    instructions.push(self.instruction()?);
                    continue;
                };
                self.expect(";")?;
                break end;
            };
            blocks.push((instructions, end));
        }
        let count = blocks.len();
        blocks
            .into_iter()
            .enumerate()
            .map(|(index, (instructions, end))| {
                let next = || {
                    (index + 1 < count)
                        .then_some(index + 1)
                        .ok_or_else(|| Error::new("PTX block falls off the end of the entry"))
                };
                let target = |label: &str| {
                    labels
                        .get(label)
                        .copied()
                        .ok_or_else(|| Error::new(format!("unknown PTX label {label}")))
                };
                let end = match end {
                    End::Return => Terminator::Return,
                    End::Jump(label) => Terminator::Jump(target(label)?),
                    End::Fallthrough => Terminator::Jump(next()?),
                    End::Branch(predicate, invert, label) => {
                        let (mut taken, mut other) = (target(label)?, next()?);
                        if invert {
                            std::mem::swap(&mut taken, &mut other);
                        }
                        Terminator::Branch {
                            predicate,
                            taken,
                            other,
                        }
                    }
                };
                Ok(Block { instructions, end })
            })
            .collect()
    }
}

/// Skip a module-level declaration (`.global` / `.const` / …) through its
/// terminating `;`, including `{…}` initializers used by nvcc lookup tables.
fn skip_module_declaration(parser: &mut Parser<'_>) -> Result<(), Error> {
    let mut depth = 0i32;
    loop {
        let token = parser.take()?;
        match token {
            "{" => depth += 1,
            "}" => depth -= 1,
            ";" if depth == 0 => return Ok(()),
            _ => {}
        }
        if depth < 0 {
            return Err(Error::new("unbalanced braces in PTX module declaration"));
        }
    }
}

/// Consume storage-class / linkage prefixes that are not `.entry`.
fn skip_module_directives(parser: &mut Parser<'_>) -> Result<(), Error> {
    loop {
        let Some(first) = parser.peek() else {
            return Ok(());
        };
        match first {
            ".entry" => return Ok(()),
            ".visible" | ".weak" | ".extern" | ".common" => {
                // `.visible .entry` begins a kernel — leave the attribute for
                // `parse_entry`. Other `.visible .global` forms are skipped.
                let rest = parser.tokens.as_slice();
                if rest.get(1) == Some(&".entry") {
                    return Ok(());
                }
                parser.take()?;
                // Optional second linkage/storage token, then declaration body.
                if matches!(
                    parser.peek(),
                    Some(".global" | ".const" | ".shared" | ".local" | ".tex")
                ) {
                    parser.take()?;
                }
                skip_module_declaration(parser)?;
            }
            ".global" | ".const" | ".shared" | ".local" | ".tex" => {
                parser.take()?;
                skip_module_declaration(parser)?;
            }
            other if other.chars().all(|c| c == '\0') => {
                parser.take()?;
            }
            other => {
                return Err(Error::new(format!(
                    "unsupported PTX module directive {other}"
                )));
            }
        }
    }
}

/// Advance past a failed `.entry` body so remaining kernels can still load.
fn skip_failed_entry(parser: &mut Parser<'_>) -> Result<(), Error> {
    // We may be mid-signature or past `{`. Seek the entry's opening brace if
    // needed, then balance to the matching close.
    let mut depth = 0i32;
    let mut seen_brace = false;
    while let Some(token) = parser.peek() {
        if token == "{" {
            parser.take()?;
            depth += 1;
            seen_brace = true;
            break;
        }
        if token == ".entry" || token == ".visible" {
            // No body yet — abandon this stub.
            return Ok(());
        }
        parser.take()?;
    }
    if !seen_brace {
        return Ok(());
    }
    while depth > 0 {
        match parser.take()? {
            "{" => depth += 1,
            "}" => depth -= 1,
            _ => {}
        }
    }
    Ok(())
}

/// Rewrite nvcc half2 extract scopes into plain bitwise ops our parser accepts.
///
/// ```ptx
/// {.reg .f16 low,high;
/// mov.b32 {low,high}, %rN;
/// mov.b16 %rsM, low;}
/// ```
/// becomes `and.b32 %rsM, %rN, 65535;` (low 16 bits). High-lane extracts become
/// `shr.u32 %rsM, %rN, 16;`.
fn rewrite_half_extract_scopes(source: &str) -> String {
    let mut out = String::with_capacity(source.len());
    let bytes = source.as_bytes();
    let mut i = 0usize;
    while i < bytes.len() {
        // Only attempt a match at `{`. Trimming whitespace before `{` would
        // steal the newline after a `//` comment and glue the replacement onto
        // that comment line (`//\n{…}` → `//and.b32…`), which then disappears.
        if bytes[i] == b'{'
            && let Some(caps) = HalfExtract::match_prefix(&source[i..])
        {
            out.push_str(&caps.replacement);
            i += caps.consumed;
            continue;
        }
        out.push(source[i..].chars().next().unwrap());
        i += source[i..].chars().next().unwrap().len_utf8();
    }
    out
}

struct HalfExtract {
    replacement: String,
    consumed: usize,
}

impl HalfExtract {
    /// `rest` must begin at the opening `{` of a half-extract scope.
    fn match_prefix(rest: &str) -> Option<Self> {
        if !rest.starts_with('{') {
            return None;
        }
        let after_brace = rest[1..].trim_start();
        if !after_brace.starts_with(".reg") {
            return None;
        }
        // Find matching `}` that closes the scope (not the mov.b32 {low,high}).
        let mut depth = 0i32;
        let mut end = None;
        for (idx, ch) in rest.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = Some(idx + 1);
                        break;
                    }
                }
                _ => {}
            }
        }
        let end = end?;
        let block = &rest[..end];
        // Require the nvcc half2 unpack pattern specifically.
        if !block.contains(".f16") {
            return None;
        }
        // mov.b32 {low,high}, %rSRC;  (optional space after comma)
        let src = {
            let key = "mov.b32 {low,high},";
            let pos = block.find(key)?;
            let after = block[pos + key.len()..].trim_start();
            let name = after.split(';').next()?.trim();
            if !name.starts_with('%') {
                return None;
            }
            name.to_string()
        };
        // Pattern A: mov.b16 %rsDST, low|high;
        // Pattern B: cvt.f32.f16 %fDST, low|high;  (extract + convert in one scope)
        let replacement = if let Some(pos) = block.find("mov.b16") {
            let after = block[pos + "mov.b16".len()..].trim_start();
            let mut parts = after.split(',');
            let dst = parts.next()?.trim().to_string();
            let lane = parts
                .next()?
                .trim()
                .trim_matches(|c: char| c == ';' || c == '}' || c.is_whitespace());
            if !dst.starts_with('%') || !matches!(lane, "low" | "high") {
                return None;
            }
            if lane == "low" {
                format!("and.b32 {dst}, {src}, 65535;\n")
            } else {
                format!("shr.u32 {dst}, {src}, 16;\n")
            }
        } else if let Some(pos) = block.find("cvt.f32.f16") {
            let after = block[pos + "cvt.f32.f16".len()..].trim_start();
            let mut parts = after.split(',');
            let dst = parts.next()?.trim().to_string();
            let lane = parts
                .next()?
                .trim()
                .trim_matches(|c: char| c == ';' || c == '}' || c.is_whitespace());
            if !dst.starts_with('%') || !matches!(lane, "low" | "high") {
                return None;
            }
            // Mask/shift into the f32 slot (B32 bits), then convert half→float in place.
            if lane == "low" {
                format!("and.b32 {dst}, {src}, 65535;\ncvt.f32.f16 {dst}, {dst};\n")
            } else {
                format!("shr.u32 {dst}, {src}, 16;\ncvt.f32.f16 {dst}, {dst};\n")
            }
        } else {
            return None;
        };
        Some(Self {
            replacement,
            consumed: end,
        })
    }
}

pub(super) fn rewrite_half_scopes(source: &str) -> String {
    rewrite_half_extract_scopes(source)
}

pub(super) fn parse(source: &str) -> Result<Vec<Entry<'_>>, Error> {
    let mut parser = Parser {
        tokens: tokens(source)?.into_iter(),
    };
    parser.expect(".version")?;
    if !matches!(parser.take()?, "4.1" | "8.8" | "9.0") {
        return Err(Error::new("unsupported PTX version"));
    }
    parser.expect(".target")?;
    let target = parser.take()?;
    if !target
        .strip_prefix("sm_")
        .is_some_and(|value| value.parse::<u16>().is_ok())
    {
        return Err(Error::new("expected a PTX sm target"));
    }
    parser.expect(".address_size")?;
    parser.expect("64")?;
    let mut names = std::collections::HashSet::new();
    let mut entries = Vec::new();
    let mut first_skip: Option<Error> = None;
    loop {
        skip_module_directives(&mut parser)?;
        if parser.peek().is_none() {
            break;
        }
        // Snapshot so a failed kernel can be skipped without aborting the module.
        let checkpoint = parser.tokens.clone();
        match parse_entry(&mut parser) {
            Ok(entry) => {
                if !names.insert(entry.name) {
                    return Err(Error::new("duplicate PTX entry name"));
                }
                entries.push(entry);
            }
            Err(error) => {
                if first_skip.is_none() {
                    first_skip = Some(error.clone());
                }
                parser.tokens = checkpoint;
                // Consume optional `.visible` then `.entry name … { … }`.
                parser.eat(".visible");
                if parser.eat(".entry") {
                    let _ = parser.identifier();
                    skip_failed_entry(&mut parser)?;
                } else {
                    return Err(error);
                }
            }
        }
        if parser.peek().is_none() {
            break;
        }
    }
    if entries.is_empty() {
        return Err(first_skip.unwrap_or_else(|| {
            Error::new("PTX module contains no supported .entry kernels")
        }));
    }
    Ok(entries)
}

pub(super) fn parse_named<'a>(source: &'a str, name: &str) -> Result<Entry<'a>, Error> {
    let mut tokens = tokens(source)?;
    let start = tokens
        .windows(3)
        .position(|tokens| tokens == [".entry", name, "("])
        .ok_or_else(|| Error::new("PTX entry is absent"))?;
    tokens.drain(..start);
    let mut parser = Parser {
        tokens: tokens.into_iter(),
    };
    parse_entry(&mut parser)
}

fn parse_entry<'a>(parser: &mut Parser<'a>) -> Result<Entry<'a>, Error> {
    parser.eat(".visible");
    parser.expect(".entry")?;
    let name = parser.identifier()?;
    let mut params = parser.params()?;
    // Performance / occupancy hints may appear in any order before `{`.
    let mut max_threads = None;
    loop {
        if parser.eat(".maxntid") {
            let mut total = 1u32;
            for axis in 0..3 {
                total = parser
                    .take()?
                    .parse::<u32>()
                    .ok()
                    .filter(|&n| n != 0)
                    .and_then(|n| total.checked_mul(n))
                    .ok_or_else(|| Error::new("invalid PTX .maxntid product"))?;
                if !parser.eat(",") {
                    break;
                }
                if axis == 2 {
                    return Err(Error::new("too many PTX .maxntid dimensions"));
                }
            }
            max_threads = Some(total);
            continue;
        }
        if parser.eat(".minnctapersm") || parser.eat(".maxnctapersm") {
            let _ = parser.take()?; // integer count
            continue;
        }
        if parser.eat(".reqntid") {
            // Same shape as .maxntid; ignore beyond validating tokens.
            for axis in 0..3 {
                let _ = parser.take()?;
                if !parser.eat(",") {
                    break;
                }
                if axis == 2 {
                    return Err(Error::new("too many PTX .reqntid dimensions"));
                }
            }
            continue;
        }
        break;
    }
    parser.expect("{")?;
    // nvcc may emit `.local` depots before `.reg` declarations.
    let local = if parser.eat(".local") {
        parser.expect(".align")?;
        let align = parser.take()?;
        if !matches!(align, "4" | "8" | "16") {
            return Err(Error::new("unsupported PTX .local alignment"));
        }
        parser.expect(".b8")?;
        let name = parser.identifier()?;
        if name.starts_with('%') {
            return Err(Error::new("invalid PTX local symbol"));
        }
        parser.expect("[")?;
        let size = parser
            .take()?
            .parse::<u32>()
            .ok()
            .filter(|&size| size != 0 && size.is_multiple_of(4))
            .ok_or_else(|| Error::new("local array size must be a positive multiple of four"))?;
        parser.expect("]")?;
        parser.expect(";")?;
        Some((name, size))
    } else {
        None
    };
    let registers = parser.registers()?;
    let mut shared = Vec::new();
    while parser.eat(".shared") {
        parser.expect(".align")?;
        let align = parser.take()?;
        if !matches!(align, "4" | "8" | "16") {
            return Err(Error::new("unsupported PTX .shared alignment"));
        }
        parser.expect(".b8")?;
        let name = parser.identifier()?;
        if name.starts_with('%') {
            return Err(Error::new("invalid PTX shared symbol"));
        }
        parser.expect("[")?;
        let size = parser
            .take()?
            .parse::<u32>()
            .ok()
            .filter(|&size| size != 0 && size.is_multiple_of(4))
            .ok_or_else(|| Error::new("shared array size must be a positive multiple of four"))?;
        parser.expect("]")?;
        parser.expect(";")?;
        shared.push((name, size));
    }
    let blocks = parser.blocks()?;
    infer_direct_global_params(&mut params, &blocks);
    // Dynamic / extern shared: nvcc emits `mov.u32 %r, symbol` without a local .shared decl.
    if shared.is_empty() {
        for inst in blocks.iter().flat_map(|block| &block.instructions) {
            if let Instruction::SharedSymbol32(_, name) = inst {
                // Allocate a generous workgroup buffer; CUDA launch shared size selects the
                // live prefix. Softmax and similar kernels index from symbol address 0.
                shared.push((*name, 48 * 1024));
                break;
            }
        }
    }
    Ok(Entry {
        name,
        params,
        registers,
        shared,
        local,
        max_threads,
        blocks,
    })
}

/// nvcc may omit .ptr. Only promote direct, uniquely defined parameter loads.
/// The emitter still checks dominance and requires compatible uses of the pointer.
fn infer_direct_global_params(params: &mut [(&str, Param)], blocks: &[Block<'_>]) {
    let mut definitions = HashMap::new();
    let mut defined = Vec::new();
    for inst in blocks.iter().flat_map(|block| &block.instructions) {
        if inst.collect_definitions(&mut defined).is_some() {
            for &dst in &defined {
                definitions
                    .entry(dst)
                    .and_modify(|def| *def = None)
                    .or_insert(Some(inst));
            }
        }
    }
    for inst in blocks.iter().flat_map(|block| &block.instructions) {
        if let Instruction::ToGlobal(_, src) = inst
            && let Some(Some(Instruction::Param(Width::B64, _, name, 0))) = definitions.get(src)
            && let Some((_, ty)) = params.iter_mut().find(|(param, _)| param == name)
            && *ty == Param::U64
        {
            *ty = Param::GlobalPtr;
        }
    }
}
