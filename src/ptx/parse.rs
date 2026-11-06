use super::{
    Block, Comparison, Entry, Error, FloatOp, Instruction, IntegerOp, Param, Space, Terminator,
    Width,
};
use std::collections::HashMap;

fn tokens(mut source: &str) -> Result<Vec<&str>, Error> {
    fn punctuation(c: char) -> bool {
        "(){}[],;:<>@!".contains(c)
    }

    let mut tokens = Vec::new();
    loop {
        source = source.trim_start_matches(|c: char| c.is_ascii_whitespace());
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
                self.expect("4")?;
                self.expect(".b8")?;
                let name = self.identifier()?;
                self.expect("[")?;
                let bytes = self
                    .take()?
                    .parse::<u16>()
                    .ok()
                    .filter(|n| (4..=4096).contains(n) && n.is_multiple_of(4))
                    .ok_or_else(|| Error::new("unsupported PTX parameter aggregate size"))?;
                self.expect("]")?;
                return Ok((name, Param::Words(bytes / 4)));
            }
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
                ".b32" | ".u32" | ".s32" | ".f32" => Width::B32,
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
        let result = match &slice[..end] {
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
            ["ld.param.b64" | "ld.param.u64", dst, ",", "[", src, "]"] => {
                Instruction::Param(Width::B64, dst, src, 0)
            }
            ["cvta.to.global.u64", dst, ",", src] => Instruction::ToGlobal(dst, src),
            ["mov.b64", dst, ",", src] => Instruction::SharedAddress(dst, src),
            ["bar.sync", "0"] => Instruction::Barrier,
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
            ["mov.u32", dst, ",", src] => Instruction::Builtin(dst, src),
            ["mad.lo.s32" | "mad.lo.u32", dst, ",", a, ",", b, ",", c] => {
                Instruction::Mad32(dst, [a, b, c])
            }
            [
                op
                @ ("setp.ge.u32" | "setp.gt.u32" | "setp.lt.u32" | "setp.le.u32" | "setp.eq.b32"),
                dst,
                ",",
                a,
                ",",
                b,
            ] => {
                let comparison = match *op {
                    "setp.ge.u32" => Comparison::GreaterEqual,
                    "setp.gt.u32" => Comparison::Greater,
                    "setp.lt.u32" => Comparison::Less,
                    "setp.le.u32" => Comparison::LessEqual,
                    _ => Comparison::Equal,
                };
                Instruction::Compare(comparison, dst, [a, b])
            }
            [op @ ("and.pred" | "or.pred"), dst, ",", a, ",", b] => Instruction::PredicateLogic {
                and: *op == "and.pred",
                dst,
                src: [a, b],
            },
            [
                op @ ("add.s32" | "sub.s32" | "mul.lo.s32" | "mul.hi.u32" | "shr.u32"),
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
                    "mul.hi.u32" => IntegerOp::MulHighUnsigned,
                    _ => IntegerOp::ShiftRight,
                };
                Instruction::Integer32(op, dst, [a, b])
            }
            ["mov.b32", dst, ",", src] => Instruction::Move32(dst, src),
            ["cvt.u64.u32", dst, ",", src] => Instruction::WidenUnsigned(dst, src),
            ["mul.wide.u32", dst, ",", a, ",", b] => Instruction::MulWide32(dst, [a, b]),
            ["shl.b64", dst, ",", a, ",", b] => Instruction::ShiftLeft64(dst, [a, b]),
            ["add.s64" | "add.u64", dst, ",", a, ",", b] => Instruction::Add64(dst, [a, b]),
            [
                op @ ("ld.global.b32" | "ld.global.f32" | "ld.shared.b32"),
                dst,
                ",",
                "[",
                addr,
                "]",
            ] => {
                let space = if *op != "ld.shared.b32" {
                    Space::Global
                } else {
                    Space::Shared
                };
                Instruction::Load32(space, dst, addr)
            }
            ["add.f32" | "add.rn.f32", dst, ",", a, ",", b] => {
                Instruction::Float32(dst, FloatOp::Add([a, b]))
            }
            [op @ ("min.f32" | "max.f32"), dst, ",", a, ",", b] => Instruction::Float32(
                dst,
                FloatOp::MinMax {
                    min: *op == "min.f32",
                    src: [a, b],
                },
            ),
            ["mul.rn.f32", dst, ",", a, ",", b] => Instruction::Float32(dst, FloatOp::Mul([a, b])),
            ["fma.rn.f32", dst, ",", a, ",", b, ",", c] => {
                Instruction::Float32(dst, FloatOp::Fma([a, b, c]))
            }
            ["cvt.rn.f32.u32", dst, ",", src] => {
                Instruction::Float32(dst, FloatOp::FromUnsigned([src]))
            }
            ["div.rn.f32", dst, ",", a, ",", b] => Instruction::Float32(dst, FloatOp::Div([a, b])),
            ["sqrt.rn.f32", dst, ",", src] => Instruction::Float32(dst, FloatOp::Sqrt([src])),
            [
                op @ ("st.global.b32" | "st.global.f32" | "st.shared.b32"),
                "[",
                addr,
                "]",
                ",",
                src,
            ] => {
                let space = if *op != "st.shared.b32" {
                    Space::Global
                } else {
                    Space::Shared
                };
                Instruction::Store32(space, addr, src)
            }
            _ => {
                return Err(Error::new(format!(
                    "unsupported PTX instruction: {}",
                    slice[..end].join(" ")
                )));
            }
        };
        self.tokens.nth(end);
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

pub(super) fn parse(source: &str) -> Result<Vec<Entry<'_>>, Error> {
    let mut parser = Parser {
        tokens: tokens(source)?.into_iter(),
    };
    parser.expect(".version")?;
    if !matches!(parser.take()?, "4.1" | "8.8") {
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
    loop {
        let entry = parse_entry(&mut parser)?;
        if !names.insert(entry.name) {
            return Err(Error::new("duplicate PTX entry name"));
        }
        entries.push(entry);
        if parser.peek().is_none() {
            return Ok(entries);
        }
    }
}

fn parse_entry<'a>(parser: &mut Parser<'a>) -> Result<Entry<'a>, Error> {
    parser.eat(".visible");
    parser.expect(".entry")?;
    let name = parser.identifier()?;
    let mut params = parser.params()?;
    let max_threads = if parser.eat(".maxntid") {
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
        Some(total)
    } else {
        None
    };
    parser.expect("{")?;
    let registers = parser.registers()?;
    let shared = if parser.eat(".shared") {
        parser.expect(".align")?;
        parser.expect("4")?;
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
        Some((name, size))
    } else {
        None
    };
    let blocks = parser.blocks()?;
    infer_direct_global_params(&mut params, &blocks);
    Ok(Entry {
        name,
        params,
        registers,
        shared,
        max_threads,
        blocks,
    })
}

/// nvcc may omit .ptr. Only promote direct, uniquely defined parameter loads.
/// The emitter still checks dominance and requires compatible uses of the pointer.
fn infer_direct_global_params(params: &mut [(&str, Param)], blocks: &[Block<'_>]) {
    let mut definitions = HashMap::new();
    for inst in blocks.iter().flat_map(|block| &block.instructions) {
        if let Some((dst, _)) = inst.assignment() {
            definitions
                .entry(dst)
                .and_modify(|def| *def = None)
                .or_insert(Some(inst));
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
