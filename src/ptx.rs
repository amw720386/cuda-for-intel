//! PTX subset to OpenCL-environment SPIR-V.
mod flow;
mod parse;
mod rounded;
mod spirv;

use std::collections::HashMap;
use std::fmt;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Param {
    U8,
    U32,
    F32,
    U64,
    GlobalPtr,
    /// `.param .align N .b8 name[bytes]` — `words` is `bytes/4`, `align` is 4/8/16.
    Words { words: u16, align: u16 },
}

impl Param {
    pub(crate) fn alignment(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U64 | Self::GlobalPtr => 8,
            Self::Words { align, .. } => usize::from(align),
            Self::U32 | Self::F32 => 4,
        }
    }

    pub(crate) fn size(self) -> usize {
        match self {
            Self::U8 => 1,
            Self::U32 | Self::F32 => 4,
            Self::U64 | Self::GlobalPtr => 8,
            Self::Words { words, .. } => usize::from(words) * 4,
        }
    }
}

#[derive(Clone, Debug)]
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
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

#[derive(Clone, Copy, Debug)]
enum IntegerOp {
    Add,
    Sub,
    Mul,
    And,
    Or,
    ShiftLeft,
    ShiftRightLogical,
    ShiftRightArithmetic,
    Div,
    DivUnsigned,
    Rem,
    RemUnsigned,
    MulHighUnsigned,
    SignExtend8,
    /// Zero-extend / truncate to u16 in the low 16 bits (`cvt.u16.u32`).
    ToU16,
    /// Sign-extend low 8 bits into a 16-bit value (`cvt.s16.s8`).
    SignExtend8To16,
    /// Arithmetic shift of a 16-bit value (`shr.s16`).
    ShiftRightArithmetic16,
}

#[derive(Clone, Copy, Debug)]
enum FloatOp<'a> {
    Add([&'a str; 2]),
    Sub([&'a str; 2]),
    Mul([&'a str; 2]),
    MinMax { min: bool, src: [&'a str; 2] },
    Fma([&'a str; 3]),
    FromUnsigned([&'a str; 1]),
    FromSigned([&'a str; 1]),
    FromHalf([&'a str; 1]),
    FromSigned16([&'a str; 1]),
    Div([&'a str; 2]),
    Sqrt([&'a str; 1]),
    RsqrtApprox([&'a str; 1]),
    RcpApprox([&'a str; 1]),
    SinApprox([&'a str; 1]),
    CosApprox([&'a str; 1]),
    Lg2Approx([&'a str; 1]),
    Ex2Approx([&'a str; 1]),
}

impl<'a> FloatOp<'a> {
    fn inputs(&self) -> &[&'a str] {
        match self {
            Self::Add(inputs) | Self::Sub(inputs) | Self::Mul(inputs) | Self::Div(inputs) => {
                inputs
            }
            Self::Fma(inputs) => inputs,
            Self::MinMax { src, .. } => src,
            Self::FromUnsigned(inputs)
            | Self::FromSigned(inputs)
            | Self::FromHalf(inputs)
            | Self::FromSigned16(inputs)
            | Self::Sqrt(inputs)
            | Self::RsqrtApprox(inputs)
            | Self::RcpApprox(inputs)
            | Self::SinApprox(inputs)
            | Self::CosApprox(inputs)
            | Self::Lg2Approx(inputs)
            | Self::Ex2Approx(inputs) => inputs,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Instruction<'a> {
    Param(Width, &'a str, &'a str, usize),
    /// `ld.param.s8` — load a byte parameter and sign-extend into a b16/b32 slot.
    ParamS8(&'a str, &'a str),
    /// `ld.param.v2.f32` / `ld.param.v4.f32` from an aggregate `.b8` parameter.
    ParamVec {
        count: u8,
        dst: [&'a str; 4],
        name: &'a str,
        offset: usize,
    },
    ToGlobal(&'a str, &'a str),
    SharedAddress(&'a str, &'a str),
    /// `mov.u32 %r, shared_symbol` — 32-bit address of a (possibly dynamic) shared array.
    SharedSymbol32(&'a str, &'a str),
    Builtin(&'a str, &'a str),
    Mad32(&'a str, [&'a str; 3]),
    /// CUDA `dp4a.s32.s32` — signed 4×8-bit dot-product accumulate.
    Dp4a(&'a str, [&'a str; 3]),
    Compare {
        op: Comparison,
        signed: bool,
        /// When true, compare the low 16 bits as signed/unsigned 16-bit values.
        bits16: bool,
        dst: &'a str,
        src: [&'a str; 2],
    },
    CompareFloat {
        op: Comparison,
        dst: &'a str,
        src: [&'a str; 2],
    },
    Compare64 {
        op: Comparison,
        signed: bool,
        dst: &'a str,
        src: [&'a str; 2],
    },
    /// `selp.*` — `dst = src[2] ? src[0] : src[1]`.
    Select {
        kind: Width,
        dst: &'a str,
        src: [&'a str; 3],
    },
    PredicateLogic {
        and: bool,
        dst: &'a str,
        src: [&'a str; 2],
    },
    Integer32(IntegerOp, &'a str, [&'a str; 2]),
    Not32(&'a str, &'a str),
    Move32(&'a str, &'a str),
    MulWide32 {
        signed: bool,
        dst: &'a str,
        src: [&'a str; 2],
    },
    Widen {
        signed: bool,
        dst: &'a str,
        src: &'a str,
    },
    /// `cvt.u32.u64` — truncate the low 32 bits of a 64-bit value.
    Narrow(&'a str, &'a str),
    ShiftLeft64(&'a str, [&'a str; 2]),
    ShiftRightLogical64(&'a str, [&'a str; 2]),
    And64(&'a str, [&'a str; 2]),
    Rem64 {
        signed: bool,
        dst: &'a str,
        src: [&'a str; 2],
    },
    Add64(&'a str, [&'a str; 2]),
    Mul64(&'a str, [&'a str; 2]),
    Load32(Space, &'a str, &'a str),
    Load64(Space, &'a str, &'a str),
    Load16(Space, &'a str, &'a str),
    LoadVec {
        space: Space,
        count: u8,
        dst: [&'a str; 4],
        addr: &'a str,
    },
    Float32(&'a str, FloatOp<'a>),
    Store32(Space, &'a str, &'a str),
    StoreVec {
        space: Space,
        count: u8,
        addr: &'a str,
        src: [&'a str; 4],
    },
    AtomicAdd32(&'a str, &'a str, &'a str),
    /// CUDA `shfl.sync.bfly.b32` — warp butterfly exchange with an active mask.
    /// Sources are `[value, xor_mask, clamp, membermask]`.
    /// Optional `pred` is the dual-destination lane-valid flag (`dst|pred`).
    ShuffleBfly {
        dst: &'a str,
        pred: Option<&'a str>,
        src: [&'a str; 4],
    },
    Barrier,
}

impl<'a> Instruction<'a> {
    /// Appends registers defined by this instruction. Returns source operands when it defines any.
    fn collect_definitions(&self, out: &mut Vec<&'a str>) -> Option<&[&'a str]> {
        use Instruction::*;
        out.clear();
        Some(match self {
            Param(_, dst, ..)
            | ParamS8(dst, _)
            | SharedAddress(dst, _)
            | SharedSymbol32(dst, _)
            | AtomicAdd32(dst, ..) => {
                out.push(*dst);
                &[]
            }
            ParamVec { count, dst, .. } => {
                out.extend_from_slice(&dst[..usize::from(*count)]);
                &[]
            }
            Mad32(dst, src) | Dp4a(dst, src) => {
                out.push(*dst);
                src
            }
            Load32(_, dst, addr) | Load64(_, dst, addr) | Load16(_, dst, addr) => {
                out.push(*dst);
                std::slice::from_ref(addr)
            }
            LoadVec {
                count, dst, addr, ..
            } => {
                out.extend_from_slice(&dst[..usize::from(*count)]);
                std::slice::from_ref(addr)
            }
            ToGlobal(dst, src) | Builtin(dst, src) | Move32(dst, src) | Not32(dst, src) => {
                out.push(*dst);
                std::slice::from_ref(src)
            }
            Widen { dst, src, .. } | Narrow(dst, src) => {
                out.push(*dst);
                std::slice::from_ref(src)
            }
            Compare { dst, src, .. }
            | CompareFloat { dst, src, .. }
            | Compare64 { dst, src, .. }
            | PredicateLogic { dst, src, .. }
            | Integer32(_, dst, src)
            | MulWide32 { dst, src, .. }
            | ShiftLeft64(dst, src)
            | ShiftRightLogical64(dst, src)
            | And64(dst, src)
            | Rem64 { dst, src, .. }
            | Add64(dst, src)
            | Mul64(dst, src) => {
                out.push(*dst);
                src
            }
            Select { dst, src, .. } => {
                out.push(*dst);
                src
            }
            Float32(dst, op) => {
                out.push(*dst);
                op.inputs()
            }
            ShuffleBfly { dst, pred, src } => {
                out.push(*dst);
                if let Some(pred) = pred {
                    out.push(*pred);
                }
                src
            }
            Store32(..) | StoreVec { .. } | Barrier => return None,
        })
    }

    fn is_memory_load(&self) -> bool {
        matches!(
            self,
            Self::Load32(..)
                | Self::Load64(..)
                | Self::Load16(..)
                | Self::LoadVec { .. }
                | Self::AtomicAdd32(..)
        )
    }

    fn is_convergent(&self) -> bool {
        matches!(self, Self::Barrier | Self::ShuffleBfly { .. })
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
pub(crate) struct KernelMetadata {
    pub(crate) name: String,
    pub(crate) params: Vec<Param>,
    pub(crate) shared_bytes: u32,
    pub(crate) max_threads: Option<u32>,
    pub(crate) device_atomics: bool,
    pub(crate) fp32: bool,
    pub(crate) fp32_fma: bool,
}

#[derive(Debug)]
pub(crate) struct Translation {
    pub(crate) words: Vec<u32>,
    pub(crate) kernels: Vec<KernelMetadata>,
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
    const GGML_MODERN: &str = include_str!("../examples/ggml_scale_clamp.ptx");
    const GGML_ADD: &str = include_str!("../examples/ggml_add_f32.ptx");

    #[test]
    fn lowers_ggml_grid_stride_instructions() {
        use rspirv::dr::Operand;
        use rspirv::spirv::{BuiltIn, Op};

        let translated = translate(GGML_ADD).unwrap();
        assert_eq!(
            translated.kernels[0].params,
            [vec![Param::GlobalPtr; 3], vec![Param::U32; 5]].concat()
        );
        let module = rspirv::dr::load_words(&translated.words).unwrap();
        assert!(module.annotations.iter().any(|inst| {
            inst.operands
                .contains(&Operand::BuiltIn(BuiltIn::NumWorkgroups))
        }));
        for opcode in [
            Op::LogicalOr,
            Op::LogicalAnd,
            Op::ULessThanEqual,
            Op::ShiftLeftLogical,
        ] {
            assert!(
                module
                    .all_inst_iter()
                    .any(|inst| inst.class.opcode == opcode)
            );
        }
        assert!(translate(&GGML_ADD.replace("%rd10, 2;", "%rd10, 64;")).is_err());
        assert!(translate(&GGML_ADD.replace("%p3, %p1, %p2;", "%p3, %r4, %p2;")).is_err());
    }

    #[test]
    fn translates_authentic_ggml_rms_norm_f32() {
        let source = include_str!("../examples/ggml_rms_norm_auth.ptx");
        translate(source).unwrap_or_else(|e| panic!("rms reject: {e}"));
    }

    #[test]
    fn translates_authentic_ggml_scale_f32_module() {
        let source = include_str!("../examples/ggml_scale_f32_auth.ptx");
        let translated = translate(source).unwrap();
        assert_eq!(translated.kernels.len(), 1);
        assert!(translated.kernels[0].name.contains("scale_f32IfEv"));
    }

    #[test]
    fn translates_authentic_ggml_scale_fatbin_skipping_globals_and_bf16() {
        // Real nvcc fatbin payload: 11 `.global` LUT tables + bf16 + f32 entries.
        let source = include_str!("../examples/ggml_scale_module_full.ptx");
        let translated = translate(source).unwrap();
        assert!(
            translated
                .kernels
                .iter()
                .any(|k| k.name.contains("scale_f32IfEv")),
            "f32 scale entry must survive best-effort module parse"
        );
    }

    #[test]
    fn lowers_nvcc_aggregate_and_scalar_abi() {
        let translated = translate(GGML_MODERN).unwrap();
        assert_eq!(translated.kernels.len(), 2);
        let module = rspirv::dr::load_words(&translated.words).unwrap();
        for (kernel, function) in translated.kernels.iter().zip(&module.functions) {
            assert_eq!(kernel.params.len(), 14);
            assert_eq!(kernel.params[..2], [Param::GlobalPtr; 2]);
            assert_eq!(
                kernel.params[5..8],
                [Param::Words { words: 3, align: 4 }; 3]
            );
            assert_eq!(function.parameters.len(), 20);
        }
    }

    #[test]
    fn translates_ggml_norm_f32_li32_snippet() {
        let source = include_str!("../examples/ggml_norm_f32_32.ptx");
        let translated = translate(source).unwrap_or_else(|error| panic!("{error}"));
        assert!(translated.kernels[0].fp32);
        assert!(translated.kernels[0].shared_bytes >= 32 * 4);
    }

    #[test]
    fn translates_ggml_rms_norm_f32_li32() {
        let translated = translate(include_str!("../examples/ggml_rms_norm_f32_32.ptx"))
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(translated.kernels[0].fp32);
    }

    #[test]
    fn translates_ggml_rope_norm_f32_fast_math() {
        let translated = translate(include_str!("../examples/ggml_rope_norm_f32.ptx"))
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(translated.kernels[0].fp32);
        assert!(translated.kernels[0].fp32_fma);
    }

    #[test]
    fn translates_authentic_ggml_rope_norm_entry() {
        let translated = translate(include_str!("../examples/ggml_rope_auth.ptx"))
            .unwrap_or_else(|error| panic!("{error}"));
        let kernel = &translated.kernels[0];
        assert!(kernel.fp32 && kernel.fp32_fma);
        assert_eq!(kernel.params.last(), Some(&Param::U8));
        assert!(
            kernel.params.iter().any(|param| {
                matches!(param, Param::Words { words: 2, align: 4 })
            }),
            "expected rope_corr_dims .b8[8] aggregate"
        );
    }

    #[test]
    fn translates_authentic_ggml_soft_max_f32_entry() {
        let translated = translate(include_str!("../examples/ggml_softmax_auth.ptx"))
            .unwrap_or_else(|error| panic!("{error}"));
        let kernel = &translated.kernels[0];
        assert!(kernel.fp32);
        assert!(
            kernel
                .params
                .iter()
                .any(|param| matches!(param, Param::Words { align: 8, .. })),
            "expected soft_max_params .align 8 aggregate"
        );
    }

    #[test]
    fn translates_ggml_q8_0_vec_dot_soft_half() {
        let translated = translate(include_str!("../examples/ggml_q8_0_vec_dot.ptx"))
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(translated.kernels[0].fp32);
        assert!(translated.kernels[0].fp32_fma);
    }

    #[test]
    fn translates_ggml_half_roundtrip_f16() {
        let translated = translate(include_str!("../examples/ggml_half_roundtrip.ptx"))
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(translated.kernels[0].fp32);
    }

    #[test]
    fn translates_ggml_dp4a_smoke() {
        translate(include_str!("../examples/ggml_dp4a_smoke.ptx"))
            .unwrap_or_else(|error| panic!("{error}"));
    }

    #[test]
    fn translates_ggml_q8_0_auth_layout() {
        let translated = translate(include_str!("../examples/ggml_q8_0_auth.ptx"))
            .unwrap_or_else(|error| panic!("{error}"));
        assert!(translated.kernels[0].fp32);
        assert!(translated.kernels[0].fp32_fma);
    }

    #[test]
    fn lowers_shfl_sync_bfly_through_shared_scratch() {
        const SHFL: &str = r#"
            .version 8.8
            .target sm_80
            .address_size 64
            .visible .entry shfl_bfly(
                .param .u64 .ptr .align 1 shfl_bfly_param_0
            )
            .maxntid 32
            {
                .reg .pred %p<2>;
                .reg .b32 %r<8>;
                .reg .b64 %rd<4>;
                ld.param.u64 %rd1, [shfl_bfly_param_0];
                cvta.to.global.u64 %rd2, %rd1;
                mov.u32 %r1, %tid.x;
                mul.wide.u32 %rd3, %r1, 4;
                add.s64 %rd2, %rd2, %rd3;
                ld.global.b32 %r2, [%rd2];
                mov.u32 %r5, 16;
                mov.u32 %r6, 31;
                mov.u32 %r7, -1;
                shfl.sync.bfly.b32 %r3|%p1, %r2, %r5, %r6, %r7;
                shfl.sync.bfly.b32 %r4, %r3, 8, 0x1f, 0xffffffff;
                st.global.b32 [%rd2], %r4;
                ret;
            }
        "#;
        let translated = translate(SHFL).unwrap();
        assert_eq!(translated.kernels[0].shared_bytes, 32 * 4);
        let module = rspirv::dr::load_words(&translated.words).unwrap();
        let opcodes: Vec<_> = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .map(|inst| inst.class.opcode)
            .collect();
        assert!(opcodes.iter().filter(|&&op| op == Op::ControlBarrier).count() >= 2);
        assert!(opcodes.contains(&Op::BitwiseXor));
        assert!(
            translate(&SHFL.replace("0xffffffff", "0x0000ffff"))
                .unwrap_err()
                .to_string()
                .contains("membermask")
        );
        assert!(
            translate(&SHFL.replace("shfl.sync.bfly.b32", "shfl.sync.up.b32"))
                .is_err()
        );
    }

    #[test]
    fn rejects_unsupported_nvcc_operand_forms() {
        for (from, to) in [
            (".version 8.8", ".version 9.0"),
            (".align 4 .b8", ".align 8 .b8"),
            ("param_5[12]", "param_5[10]"),
            ("param_5+8]", "param_5+12]"),
            ("param_5+8]", "param_5+1]"),
            ("param_4]", "param_4+4]"),
            ("cvta.to.global.u64", "cvta.to.shared.u64"),
            ("mul.hi.u32", "mul.hi.s32"),
            ("cvt.u64.u32", "cvt.u64.s32"),
            ("min.f32", "min.ftz.f32"),
            ("max.f32", "max.NaN.f32"),
            ("ld.global.f32", "ld.global.v2.f32"),
            ("st.global.f32", "st.global.f16"),
            ("shr.u32", "shfl.sync.up.b32"),
        ] {
            assert!(GGML_MODERN.contains(from));
            assert!(
                translate(&GGML_MODERN.replacen(from, to, 1)).is_err(),
                "accepted {to}"
            );
        }
    }

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
        assert_eq!(translated.kernels[0].shared_bytes, 16);
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
        assert_eq!(translated.kernels[0].max_threads, Some(128));
        assert_eq!(translated.kernels[0].shared_bytes, 512);
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
        assert!(translated.kernels[0].device_atomics);
        assert!(!translated.kernels[0].fp32);
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
        // A register that is never written on any path must still be rejected.
        let undefined = r#"
.version 8.8
.target sm_80
.address_size 64
.visible .entry never_defined(
	.param .u64 never_defined_param_0
)
{
	.reg .b32 %r<2>;
	.reg .b64 %rd<3>;
	ld.param.b64 %rd1, [never_defined_param_0];
	cvta.to.global.u64 %rd0, %rd1;
	st.global.b32 [%rd0], %r1;
	ret;
}
"#;
        let err = translate(undefined).unwrap_err().to_string();
        assert!(
            err.contains("undefined PTX register %r1"),
            "unexpected error: {err}"
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
            GGML_ADD,
            GGML_MODERN,
            include_str!("../examples/shfl_bfly.ptx"),
            include_str!("../tests/gpu/multi_entry.ptx"),
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
                translated.kernels[0].name
            );
            written.unwrap();
        }
    }

    #[test]
    fn lowers_round_to_nearest_fp32_operations_without_implicit_contraction() {
        use rspirv::dr::Operand;
        use rspirv::spirv::{ExecutionMode, OpenclStd100Op};

        let translated = translate(RMSNORM).unwrap();
        assert_eq!(translated.kernels[0].params.last(), Some(&Param::F32));
        assert!(translated.kernels[0].fp32 && translated.kernels[0].fp32_fma);
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
            ("fma.rn.f32", "fma.rz.f32"),
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
    fn lowers_signed_compares_widens_and_approx_rsqrt() {
        use rspirv::spirv::{Op, OpenclStd100Op};

        const SOURCE: &str = "
            .version 8.8
            .target sm_80
            .address_size 64
            .entry norm_ops(
                .param .u64 .ptr .global a,
                .param .u64 .ptr .global b
            ) {
                .reg .pred %p<4>;
                .reg .b32 %r<8>;
                .reg .f32 %f<4>;
                .reg .b64 %rd<4>;
                ld.param.u64 %rd1, [a];
                ld.param.u64 %rd2, [b];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                ld.global.b32 %r1, [%rd1];
                mov.b32 %r2, 4294967294;
                setp.lt.s32 %p1, %r1, %r2;
                setp.ne.s32 %p2, %r1, %r2;
                @%p1 bra skip;
                cvt.s64.s32 %rd3, %r1;
                cvt.rn.f32.s32 %f1, %r1;
                rsqrt.approx.f32 %f2, %f1;
                st.global.f32 [%rd2], %f2;
            skip:
                ret;
            }
        ";
        let translated = translate(SOURCE).unwrap();
        let module = rspirv::dr::load_words(&translated.words).unwrap();
        let ops: Vec<_> = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .map(|inst| inst.class.opcode)
            .collect();
        assert!(ops.contains(&Op::SLessThan));
        assert!(ops.contains(&Op::INotEqual));
        assert!(ops.contains(&Op::SConvert));
        assert!(ops.contains(&Op::ConvertSToF));
        assert!(
            module.functions[0]
                .blocks
                .iter()
                .flat_map(|b| &b.instructions)
                .any(|inst| {
                    inst.class.opcode == Op::ExtInst
                        && inst.operands.get(1)
                            == Some(&rspirv::dr::Operand::LiteralExtInstInteger(
                                OpenclStd100Op::rsqrt as u32,
                            ))
                })
        );
        assert!(translate(&SOURCE.replace("rsqrt.approx.f32", "rsqrt.rn.f32")).is_err());
        assert!(translate(&SOURCE.replace("cvt.s64.s32", "cvt.s64.u32")).is_err());
    }

    #[test]
    fn lowers_vector_float_memory() {
        use rspirv::spirv::Op;

        const SOURCE: &str = "
            .version 8.8
            .target sm_80
            .address_size 64
            .entry vector_mem(
                .param .u64 .ptr .global in,
                .param .u64 .ptr .global out
            ) {
                .reg .f32 %f<8>;
                .reg .b64 %rd<4>;
                .shared .align 4 .b8 scratch[16];
                ld.param.u64 %rd1, [in];
                ld.param.u64 %rd2, [out];
                cvta.to.global.u64 %rd1, %rd1;
                cvta.to.global.u64 %rd2, %rd2;
                ld.global.v4.f32 {%f1, %f2, %f3, %f4}, [%rd1];
                st.shared.v2.f32 [scratch], {%f1, %f2};
                bar.sync 0;
                ld.shared.v2.f32 {%f5, %f6}, [scratch];
                st.global.v4.f32 [%rd2], {%f5, %f6, %f3, %f4};
                ret;
            }
        ";
        let translated = translate(SOURCE).unwrap();
        assert_eq!(translated.kernels[0].shared_bytes, 16);
        let module = rspirv::dr::load_words(&translated.words).unwrap();
        let ops: Vec<_> = module.functions[0]
            .blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .map(|inst| inst.class.opcode)
            .collect();
        assert!(ops.contains(&Op::Load));
        assert!(ops.contains(&Op::Store));
        assert!(ops.contains(&Op::CompositeExtract));
        assert!(ops.contains(&Op::CompositeConstruct));
        assert!(translate(&SOURCE.replace("ld.global.v4.f32", "ld.global.v8.f32")).is_err());
    }

    #[test]
    fn parses_compiler_generated_ptx() {
        let entries = parse::parse(VECTOR_ADD).unwrap();
        let entry = &entries[0];
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
        assert_eq!(translated.kernels[0].name, "vector_add");
        assert_eq!(
            translated.kernels[0]
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
            ("[%rd3]", "[%rd3+x]"),
            (".reg .b64", ".reg .b32"),
            (".ptr .align 1", ".ptr .shared"),
        ] {
            assert!(VECTOR_ADD.contains(old), "fixture lacks {old}");
            assert!(
                translate(&VECTOR_ADD.replace(old, new)).is_err(),
                "accepted {new}"
            );
        }
        assert!(
            translate(&format!(
                "{VECTOR_ADD}\n.visible .entry vector_add() {{ ret; }}"
            ))
            .is_err()
        );
        assert!(translate("not PTX").is_err());
    }
}
