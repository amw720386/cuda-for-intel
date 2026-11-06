//! PTX subset to OpenCL-environment SPIR-V.
mod parse;
mod spirv;

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Param {
    U32,
    U64,
    GlobalPtr,
}

impl Param {
    pub(crate) fn size(self) -> usize {
        match self {
            Self::U32 => 4,
            Self::U64 | Self::GlobalPtr => 8,
        }
    }
}

#[derive(Debug)]
pub struct Error(String);

impl Error {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

impl From<rspirv::dr::Error> for Error {
    fn from(error: rspirv::dr::Error) -> Self {
        Self::new(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Width {
    Predicate,
    B32,
    B64,
}

#[derive(Clone, Copy, Debug)]
enum Instruction<'a> {
    Param(Width, &'a str, &'a str),
    ToGlobal(&'a str, &'a str),
    Builtin(&'a str, &'a str),
    Mad32(&'a str, [&'a str; 3]),
    Ge32(&'a str, [&'a str; 2]),
    MulWide32(&'a str, [&'a str; 2]),
    Add64(&'a str, [&'a str; 2]),
    Load32(&'a str, &'a str),
    AddFloat32(&'a str, [&'a str; 2]),
    Store32(&'a str, &'a str),
}

#[derive(Debug)]
struct Entry<'a> {
    name: &'a str,
    params: Vec<(&'a str, Param)>,
    registers: HashMap<String, Width>,
    prefix: Vec<Instruction<'a>>,
    guarded: Option<(&'a str, Vec<Instruction<'a>>)>,
}

#[derive(Debug)]
pub(crate) struct Translation {
    pub(crate) name: String,
    pub(crate) params: Vec<Param>,
    pub(crate) words: Vec<u32>,
}

pub(crate) fn translate(source: &str) -> Result<Translation, Error> {
    spirv::emit(parse::parse(source)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::spirv::Op;

    const VECTOR_ADD: &str = include_str!("../examples/vector_add.ptx");

    #[test]
    fn parses_compiler_generated_ptx() {
        let entry = parse::parse(VECTOR_ADD).unwrap();
        assert_eq!(entry.name, "vector_add");
        assert_eq!(
            entry.params.iter().map(|&(_, ty)| ty).collect::<Vec<_>>(),
            [
                Param::GlobalPtr,
                Param::GlobalPtr,
                Param::GlobalPtr,
                Param::U32
            ]
        );
        assert_eq!(entry.registers["%rd10"], Width::B64);
        assert_eq!(entry.guarded.as_ref().unwrap().0, "%p1");
    }

    #[test]
    fn lowers_bounds_check_and_float_add_to_spirv() {
        let translated = translate(VECTOR_ADD).unwrap();
        assert_eq!(translated.name, "vector_add");
        assert_eq!(
            translated
                .params
                .iter()
                .map(|param| param.size())
                .collect::<Vec<_>>(),
            [8, 8, 8, 4]
        );
        let module = rspirv::dr::load_words(&translated.words).unwrap();
        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].parameters.len(), 4);
        let opcodes: Vec<_> = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .map(|inst| inst.class.opcode)
            .collect();
        for expected in [
            Op::IMul,
            Op::UGreaterThanEqual,
            Op::BranchConditional,
            Op::FAdd,
            Op::Store,
        ] {
            assert!(opcodes.contains(&expected), "missing {expected:?}");
        }
    }

    #[test]
    fn rejects_unsupported_or_invalid_ptx() {
        for (old, new) in [
            (".address_size 64", ".address_size 32"),
            ("add.f32", "add.ftz.f32"),
            ("add.f32 \t%r8, %r6, %r7", "add.f32 %r8, %r6, %r0"),
            ("@%p1 bra", "@!%p1 bra"),
            ("$L__BB0_2:", "other_label:"),
            ("[%rd3]", "[%rd3+4]"),
            (".reg .b64", ".reg .b32"),
            (".ptr .align 1", ""),
        ] {
            assert!(VECTOR_ADD.contains(old), "fixture lacks {old}");
            assert!(
                translate(&VECTOR_ADD.replace(old, new)).is_err(),
                "accepted {new}"
            );
        }
        assert!(translate(&format!("{VECTOR_ADD}\n.visible .entry other() {{ ret; }}")).is_err());
        assert!(translate("not PTX").is_err());
    }
}
