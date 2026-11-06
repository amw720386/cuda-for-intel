use super::{Entry, Error, Instruction, Param, Space, Width};
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

    fn params(&mut self) -> Result<Vec<(&'a str, Param)>, Error> {
        let mut params = Vec::new();
        self.expect("(")?;
        if self.eat(")") {
            return Ok(params);
        }
        loop {
            self.expect(".param")?;
            let ty = match self.take()? {
                ".u32" | ".b32" => Param::U32,
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
            let name = self.identifier()?;
            if params.iter().any(|&(param, _)| param == name) {
                return Err(Error::new(format!("duplicate PTX parameter {name}")));
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
                ".b32" | ".u32" | ".s32" => Width::B32,
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
            ["ld.param.b32" | "ld.param.u32", dst, ",", "[", src, "]"] => {
                Instruction::Param(Width::B32, dst, src)
            }
            ["ld.param.b64" | "ld.param.u64", dst, ",", "[", src, "]"] => {
                Instruction::Param(Width::B64, dst, src)
            }
            ["cvta.to.global.u64", dst, ",", src] => Instruction::ToGlobal(dst, src),
            ["mov.b64", dst, ",", src] => Instruction::SharedAddress(dst, src),
            ["bar.sync", "0"] => Instruction::Barrier,
            ["mov.u32", dst, ",", src] => Instruction::Builtin(dst, src),
            ["mad.lo.s32" | "mad.lo.u32", dst, ",", a, ",", b, ",", c] => {
                Instruction::Mad32(dst, [a, b, c])
            }
            ["setp.ge.u32", dst, ",", a, ",", b] => Instruction::Ge32(dst, [a, b]),
            ["mul.wide.u32", dst, ",", a, ",", b] => Instruction::MulWide32(dst, [a, b]),
            ["add.s64" | "add.u64", dst, ",", a, ",", b] => Instruction::Add64(dst, [a, b]),
            [
                op @ ("ld.global.b32" | "ld.shared.b32"),
                dst,
                ",",
                "[",
                addr,
                "]",
            ] => {
                let space = if *op == "ld.global.b32" {
                    Space::Global
                } else {
                    Space::Shared
                };
                Instruction::Load32(space, dst, addr)
            }
            ["add.f32" | "add.rn.f32", dst, ",", a, ",", b] => Instruction::AddFloat32(dst, [a, b]),
            [
                op @ ("st.global.b32" | "st.shared.b32"),
                "[",
                addr,
                "]",
                ",",
                src,
            ] => {
                let space = if *op == "st.global.b32" {
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
}

pub(super) fn parse(source: &str) -> Result<Entry<'_>, Error> {
    let mut parser = Parser {
        tokens: tokens(source)?.into_iter(),
    };
    parser.expect(".version")?;
    parser.expect("4.1")?;
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
    parser.eat(".visible");
    parser.expect(".entry")?;
    let name = parser.identifier()?;
    let params = parser.params()?;
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
    let mut prefix = Vec::new();
    while !matches!(parser.peek(), Some("@" | "ret")) {
        prefix.push(parser.instruction()?);
    }
    let guarded = if parser.eat("@") {
        let predicate = parser.identifier()?;
        parser.expect("bra")?;
        let target = parser.identifier()?;
        parser.expect(";")?;
        let mut body = Vec::new();
        while parser.peek() != Some(target) {
            let inst = parser.instruction()?;
            if matches!(inst, Instruction::Barrier) {
                return Err(Error::new("workgroup barrier in a conditional region"));
            }
            body.push(inst);
        }
        parser.expect(target)?;
        parser.expect(":")?;
        Some((predicate, body))
    } else {
        None
    };
    parser.expect("ret")?;
    parser.expect(";")?;
    parser.expect("}")?;
    if parser.peek().is_some() {
        return Err(Error::new("only a single PTX entry is supported"));
    }
    Ok(Entry {
        name,
        params,
        registers,
        shared,
        prefix,
        guarded,
    })
}
