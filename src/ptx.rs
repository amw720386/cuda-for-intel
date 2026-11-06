//! PTX subset to OpenCL-environment SPIR-V.
mod flow;
mod parse;
mod rounded;
mod spirv;

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Param {
    U32,
    F32,
    U64,
    GlobalPtr,
}

impl Param {
    pub(crate) fn size(self) -> usize {
        match self {
            Self::U32 | Self::F32 => 4,
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Space {
    Global,
    Shared,
}

#[derive(Clone, Copy, Debug)]
enum Comparison {
    Equal,
    Less,
    Greater,
    GreaterEqual,
}

#[derive(Clone, Copy, Debug)]
enum IntegerOp {
    Add,
    Sub,
    Mul,
    ShiftRight,
}

#[derive(Clone, Copy, Debug)]
enum FloatOp<'a> {
    Add([&'a str; 2]),
    Mul([&'a str; 2]),
    Fma([&'a str; 3]),
    FromUnsigned([&'a str; 1]),
    Div([&'a str; 2]),
    Sqrt([&'a str; 1]),
}

impl<'a> FloatOp<'a> {
    fn inputs(&self) -> &[&'a str] {
        match self {
            Self::Add(inputs) | Self::Mul(inputs) | Self::Div(inputs) => inputs,
            Self::Fma(inputs) => inputs,
            Self::FromUnsigned(inputs) | Self::Sqrt(inputs) => inputs,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Instruction<'a> {
    Param(Width, &'a str, &'a str),
    ToGlobal(&'a str, &'a str),
    SharedAddress(&'a str, &'a str),
    Builtin(&'a str, &'a str),
    Mad32(&'a str, [&'a str; 3]),
    Compare(Comparison, &'a str, [&'a str; 2]),
    Integer32(IntegerOp, &'a str, [&'a str; 2]),
    Move32(&'a str, &'a str),
    MulWide32(&'a str, [&'a str; 2]),
    Add64(&'a str, [&'a str; 2]),
    Load32(Space, &'a str, &'a str),
    Float32(&'a str, FloatOp<'a>),
    Store32(Space, &'a str, &'a str),
    AtomicAdd32(&'a str, &'a str, &'a str),
    Barrier,
}

impl<'a> Instruction<'a> {
    fn assignment(&self) -> Option<(&'a str, &[&'a str])> {
        use Instruction::*;
        Some(match self {
            Param(_, dst, _) | SharedAddress(dst, _) | Load32(_, dst, _) | AtomicAdd32(dst, ..) => {
                (*dst, &[])
            }
            ToGlobal(dst, src) | Builtin(dst, src) | Move32(dst, src) => {
                (*dst, std::slice::from_ref(src))
            }
            Mad32(dst, src) => (*dst, src),
            Compare(_, dst, src)
            | Integer32(_, dst, src)
            | MulWide32(dst, src)
            | Add64(dst, src) => (*dst, src),
            Float32(dst, op) => (*dst, op.inputs()),
            Store32(..) | Barrier => return None,
        })
    }
}

#[derive(Clone, Copy, Debug)]
enum Terminator<'a> {
    Return,
    Jump(usize),
    Branch {
        predicate: &'a str,
        taken: usize,
        other: usize,
    },
}

impl Terminator<'_> {
    fn successors(self) -> impl Iterator<Item = usize> {
        match self {
            Self::Return => [None, None],
            Self::Jump(target) => [Some(target), None],
            Self::Branch { taken, other, .. } => [Some(taken), Some(other)],
        }
        .into_iter()
        .flatten()
    }
}

#[derive(Debug)]
struct Block<'a> {
    instructions: Vec<Instruction<'a>>,
    end: Terminator<'a>,
}

#[derive(Debug)]
struct Entry<'a> {
    name: &'a str,
    params: Vec<(&'a str, Param)>,
    registers: HashMap<String, Width>,
    shared: Option<(&'a str, u32)>,
    max_threads: Option<u32>,
    blocks: Vec<Block<'a>>,
}

#[derive(Debug)]
pub(crate) struct Translation {
    pub(crate) name: String,
    pub(crate) params: Vec<Param>,
    pub(crate) words: Vec<u32>,
    pub(crate) shared_bytes: u32,
    pub(crate) max_threads: Option<u32>,
    pub(crate) device_atomics: bool,
    pub(crate) fp32: bool,
    pub(crate) fp32_fma: bool,
}

pub(crate) fn translate(source: &str) -> Result<Translation, Error> {
    spirv::emit(parse::parse(source)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rspirv::spirv::Op;

    const VECTOR_ADD: &str = include_str!("../examples/vector_add.ptx");
    const BLOCK_REDUCE: &str = include_str!("../examples/block_reduce.ptx");
    const ATOMIC_TICKETS: &str = include_str!("../examples/atomic_tickets.ptx");
    const RMSNORM: &str = include_str!("../examples/rmsnorm.ptx");

    const SHARED: &str = "
        .version 4.1
        .target sm_52
        .address_size 64
        .entry shared_test() {
            .reg .b32 %r;
            .reg .b64 %a;
            .shared .align 4 .b8 scratch[16];
            mov.b64 %a, scratch;
            ld.shared.b32 %r, [scratch];
            st.shared.b32 [%a], %r;
            bar.sync 0;
            ret;
        }
    ";

    #[test]
    fn lowers_static_shared_memory_and_full_block_barrier() {
        use rspirv::dr::Operand;
        use rspirv::spirv::{MemorySemantics, Scope, StorageClass};

        let translated = translate(SHARED).unwrap();
        assert_eq!(translated.shared_bytes, 16);
        let module = rspirv::dr::load_words(&translated.words).unwrap();
        assert!(module.types_global_values.iter().any(|inst| {
            inst.class.opcode == Op::Variable
                && inst.operands == [Operand::StorageClass(StorageClass::Workgroup)]
        }));
        let barrier = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .find(|inst| inst.class.opcode == Op::ControlBarrier)
            .unwrap();
        let values: Vec<_> = barrier
            .operands
            .iter()
            .map(|operand| {
                let (Operand::IdScope(id) | Operand::IdMemorySemantics(id)) = *operand else {
                    panic!("expected a constant ID")
                };
                let inst = module
                    .types_global_values
                    .iter()
                    .find(|inst| inst.result_id == Some(id))
                    .unwrap();
                inst.operands[0].clone()
            })
            .collect();
        assert_eq!(
            values,
            [
                Operand::LiteralBit32(Scope::Workgroup as u32),
                Operand::LiteralBit32(Scope::Workgroup as u32),
                Operand::LiteralBit32(
                    (MemorySemantics::ACQUIRE_RELEASE
                        | MemorySemantics::WORKGROUP_MEMORY
                        | MemorySemantics::CROSS_WORKGROUP_MEMORY)
                        .bits()
                ),
            ]
        );
    }

    #[test]
    fn rejects_unsupported_shared_memory_and_barriers() {
        for (old, new) in [
            ("[16]", "[3]"),
            (".align 4", ".align 8"),
            ("[scratch]", "[missing]"),
            ("st.shared", "st.global"),
            ("bar.sync 0", "bar.sync 1"),
            ("bar.sync 0", "bar.sync 0, 32"),
            ("bar.sync 0", "bar.arrive 0, 32"),
        ] {
            assert!(
                translate(&SHARED.replace(old, new)).is_err(),
                "accepted {new}"
            );
        }
    }

    #[test]
    fn lowers_reduction_control_flow_and_static_limits() {
        let translated = translate(BLOCK_REDUCE).unwrap();
        assert_eq!(translated.max_threads, Some(128));
        assert_eq!(translated.shared_bytes, 512);
        let module = rspirv::dr::load_words(&translated.words).unwrap();
        let instructions = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions);
        assert_eq!(
            instructions
                .clone()
                .filter(|inst| inst.class.opcode == Op::ControlBarrier)
                .count(),
            2
        );
        assert_eq!(
            instructions
                .filter(|inst| inst.class.opcode == Op::BranchConditional)
                .count(),
            5
        );
    }

    #[test]
    fn accepts_only_device_scope_relaxed_unsigned_atomic_add() {
        let translated = translate(ATOMIC_TICKETS).unwrap();
        assert!(translated.device_atomics);
        assert!(!translated.fp32);
        let module = rspirv::dr::load_words(&translated.words).unwrap();
        assert!(
            module.functions[0]
                .blocks
                .iter()
                .flat_map(|block| &block.instructions)
                .any(|inst| inst.class.opcode == Op::AtomicIAdd)
        );
        for op in [
            "atom.global.add.u64",
            "atom.acquire.gpu.global.add.u32",
            "atom.shared.add.u32",
        ] {
            assert!(translate(&ATOMIC_TICKETS.replace("atom.global.add.u32", op)).is_err());
        }
    }

    #[test]
    fn rejects_missing_register_definitions_at_joins_and_divergent_barriers() {
        let undefined = BLOCK_REDUCE.replace("mov.b32 \t%r12, 0f00000000;", "");
        assert_ne!(undefined, BLOCK_REDUCE);
        assert!(
            translate(&undefined)
                .unwrap_err()
                .to_string()
                .contains("undefined PTX register %r12")
        );

        let divergent = BLOCK_REDUCE.replace("@%p3 bra \t$L__BB0_8;", "@%p3 bra \t$L__BB0_3;");
        assert_ne!(divergent, BLOCK_REDUCE);
        assert!(
            translate(&divergent)
                .unwrap_err()
                .to_string()
                .contains("barrier may execute divergently")
        );
    }

    #[test]
    #[ignore = "requires the Khronos spirv-val executable"]
    fn validate_opencl_spirv() {
        use std::io::Write;
        use std::process::{Command, Stdio};

        let validator = std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
        for source in [
            VECTOR_ADD,
            SHARED,
            BLOCK_REDUCE,
            ATOMIC_TICKETS,
            RMSNORM,
            include_str!("../examples/forge_residual.ptx"),
            include_str!("../examples/forge_reduce.ptx"),
        ] {
            let translated = translate(source).unwrap();
            let mut process = Command::new(&validator)
                .args(["--target-env", "opencl2.2", "-"])
                .stdin(Stdio::piped())
                .spawn()
                .expect("start spirv-val");
            let bytes: Vec<_> = translated
                .words
                .iter()
                .flat_map(|word| word.to_le_bytes())
                .collect();
            let written = process.stdin.take().unwrap().write_all(&bytes);
            assert!(
                process.wait().unwrap().success(),
                "invalid SPIR-V for {}",
                translated.name
            );
            written.unwrap();
        }
    }

    #[test]
    fn lowers_round_to_nearest_fp32_operations_without_implicit_contraction() {
        use rspirv::dr::Operand;
        use rspirv::spirv::{ExecutionMode, OpenclStd100Op};

        let translated = translate(RMSNORM).unwrap();
        assert_eq!(translated.params.last(), Some(&Param::F32));
        assert!(translated.fp32 && translated.fp32_fma);
        let module = rspirv::dr::load_words(&translated.words).unwrap();
        assert!(module.execution_modes.iter().any(|inst| {
            inst.operands
                .contains(&Operand::ExecutionMode(ExecutionMode::ContractionOff))
        }));
        let kernel = module.functions.last().unwrap();
        let instructions = kernel.blocks.iter().flat_map(|block| &block.instructions);
        assert!(instructions.clone().any(|inst| {
            inst.class.opcode == Op::ExtInst
                && inst.operands[1] == Operand::LiteralExtInstInteger(OpenclStd100Op::fma as u32)
        }));
        assert!(
            instructions
                .clone()
                .any(|inst| inst.class.opcode == Op::ConvertUToF)
        );
        assert!(
            instructions
                .clone()
                .any(|inst| inst.class.opcode == Op::FMul)
        );
        assert!(
            instructions
                .clone()
                .any(|inst| inst.class.opcode == Op::ShiftRightArithmetic)
        );
        assert!(
            !instructions
                .clone()
                .any(|inst| inst.class.opcode == Op::FDiv)
        );
    }

    #[test]
    fn rejects_unimplemented_fp32_rounding_and_approximation_modes() {
        for (original, unsupported) in [
            ("mul.rn.f32", "mul.rz.f32"),
            ("fma.rn.f32", "fma.rn.ftz.f32"),
            ("fma.rn.f32", "fma.rn.sat.f32"),
            ("cvt.rn.f32.u32", "cvt.rz.f32.u32"),
            ("div.rn.f32", "div.approx.f32"),
            ("sqrt.rn.f32", "sqrt.approx.f32"),
        ] {
            assert!(RMSNORM.contains(original));
            assert!(translate(&RMSNORM.replace(original, unsupported)).is_err());
        }
    }

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
        assert!(matches!(
            entry.blocks[0].end,
            Terminator::Branch {
                predicate: "%p1",
                ..
            }
        ));
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
            ("@%p1 bra", "@%p0 bra"),
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
