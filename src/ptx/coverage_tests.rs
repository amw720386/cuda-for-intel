//! Instruction-coverage tests. Every snippet is translated and, when a
//! `spirv-val` executable is available (`SPIRV_VAL` or `PATH`), validated.
use super::*;
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::spirv::{ExecutionMode, Op};

/// Wraps `body` in a kernel with one global output pointer (`%rdo`, already
/// converted to a global address) and the given extra register declarations.
pub(super) fn kernel(regs: &str, body: &str) -> String {
    format!(
        r#".version 9.0
.target sm_75
.address_size 64
.visible .entry probe(.param .u64 out_param, .param .u32 n_param)
{{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<16>;
    {regs}
    ld.param.u64 %rd0, [out_param];
    cvta.to.global.u64 %rdo, %rd0;
    ld.param.u32 %r0, [n_param];
{body}
    ret;
}}
"#
    )
    .replace("%rdo", "%rd15")
}

pub(super) fn validate(words: &[u32]) {
    use std::io::Write;
    use std::process::{Command, Stdio};
    let validator = std::env::var_os("SPIRV_VAL").unwrap_or_else(|| "spirv-val".into());
    let mut module = rspirv::dr::load_words(words).unwrap();
    // spirv-val does not know the Intel SubgroupSize execution mode.
    module.execution_modes.retain(|inst| {
        !matches!(
            inst.operands.get(1),
            Some(Operand::ExecutionMode(ExecutionMode::SubgroupSize))
        )
    });
    let bytes: Vec<u8> = module
        .assemble()
        .iter()
        .flat_map(|word| word.to_le_bytes())
        .collect();
    let Ok(mut process) = Command::new(&validator)
        .args(["--target-env", "spv1.3", "-"])
        .stdin(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
    else {
        return; // validator not installed
    };
    let mut stdin = process.stdin.take().unwrap();
    let writer = std::thread::spawn(move || stdin.write_all(&bytes));
    let output = process.wait_with_output().unwrap();
    let _ = writer.join();
    assert!(
        output.status.success(),
        "spirv-val rejected the module: {}",
        String::from_utf8_lossy(&output.stderr)
    );
}

/// Translates and validates; returns the decoded module.
pub(super) fn lower(source: &str) -> rspirv::dr::Module {
    let translated = translate(source).unwrap_or_else(|e| panic!("{e}\n{source}"));
    validate(&translated.words);
    rspirv::dr::load_words(&translated.words).unwrap()
}

pub(super) fn has_op(module: &rspirv::dr::Module, op: Op) -> bool {
    module.all_inst_iter().any(|inst| inst.class.opcode == op)
}

#[test]
fn flattens_register_scopes_from_inline_asm_helpers() {
    let source = kernel(
        "",
        r#"    mov.u32 %r1, 7;
    {
        .reg .b64 u64addr;
        .reg .b32 t<2>;
        cvt.u64.u32 u64addr, %r1;
        add.s32 t0, %r1, 1;
        st.global.u32 [%rd15], t0;
        st.global.u64 [%rd15+8], u64addr;
    }
    {
        .reg .b64 u64addr;
        cvt.u64.u32 u64addr, %r0;
        st.global.u64 [%rd15+16], u64addr;
    }"#,
    );
    lower(&source);
}

#[test]
fn executes_guarded_instructions_through_conditional_blocks() {
    let source = kernel(
        "",
        r#"    mov.u32 %r1, 0;
    setp.lt.u32 %p1, %r0, 4;
    @%p1 mov.u32 %r1, 5;
    @!%p1 add.s32 %r1, %r1, 9;
    st.global.u32 [%rd15], %r1;
    @%p1 ret;
    st.global.u32 [%rd15+4], %r1;"#,
    );
    let module = lower(&source);
    assert!(has_op(&module, Op::BranchConditional));
}

/// A minimal SPIR-V interpreter for the integer subset the translator emits.
/// It is only used to check the semantics of lowered instruction sequences
/// without a GPU. Memory is a flat byte map; one invocation is simulated.
pub(super) mod interp {
    use rspirv::dr::{Instruction, Module, Operand};
    use rspirv::spirv::{Op, StorageClass};
    use std::collections::HashMap;

    pub const GLOBAL: u64 = 0x1000_0000;
    pub const SHARED: u64 = 0x2000_0000;

    #[derive(Default)]
    pub struct Machine {
        pub memory: HashMap<u64, u8>,
        /// Values of kernel parameters in declaration order.
        pub params: Vec<u64>,
        /// Lane id reported to subgroup operations.
        pub lane: u64,
    }

    impl Machine {
        pub fn read(&self, addr: u64, bytes: u32) -> u64 {
            (0..bytes as u64).fold(0u64, |acc, i| {
                acc | (u64::from(*self.memory.get(&(addr + i)).unwrap_or(&0)) << (8 * i))
            })
        }
        pub fn write(&mut self, addr: u64, bytes: u32, value: u64) {
            for i in 0..bytes as u64 {
                self.memory.insert(addr + i, (value >> (8 * i)) as u8);
            }
        }
    }

    fn mask(width: u32, v: u64) -> u64 {
        if width >= 64 { v } else { v & ((1u64 << width) - 1) }
    }
    fn sext(width: u32, v: u64) -> i64 {
        if width >= 64 { v as i64 } else { ((v << (64 - width)) as i64) >> (64 - width) }
    }
    pub(super) fn id(op: &Operand) -> u32 {
        match op {
            Operand::IdRef(id) => *id,
            other => panic!("expected id, got {other:?}"),
        }
    }
    fn lit(op: &Operand) -> u32 {
        match op {
            Operand::LiteralBit32(v) | Operand::LiteralExtInstInteger(v) => *v,
            other => panic!("expected literal, got {other:?}"),
        }
    }

    pub fn run(module: &Module, machine: &mut Machine) {
        let mut widths: HashMap<u32, u32> = HashMap::new(); // type id -> bits
        let mut pointees: HashMap<u32, u32> = HashMap::new();
        for inst in &module.types_global_values {
            match inst.class.opcode {
                Op::TypeInt | Op::TypeFloat => {
                    if let Operand::LiteralBit32(w) = inst.operands[0] {
                        widths.insert(inst.result_id.unwrap(), w);
                    }
                }
                Op::TypeBool => {
                    widths.insert(inst.result_id.unwrap(), 1);
                }
                Op::TypePointer => {
                    widths.insert(inst.result_id.unwrap(), 64);
                    pointees.insert(inst.result_id.unwrap(), id(&inst.operands[1]));
                }
                _ => {}
            }
        }
        let mut values: HashMap<u32, u64> = HashMap::new();
        let mut next_shared = SHARED;
        for inst in &module.types_global_values {
            match inst.class.opcode {
                Op::Constant => {
                    let w = widths[&inst.result_type.unwrap()];
                    let v = match inst.operands[0] {
                        Operand::LiteralBit32(v) => u64::from(v),
                        Operand::LiteralBit64(v) => v,
                        _ => unreachable!(),
                    };
                    values.insert(inst.result_id.unwrap(), mask(w, v));
                    VBITS.with(|m| m.borrow_mut().insert(inst.result_id.unwrap(), w));
                }
                Op::ConstantTrue => {
                    values.insert(inst.result_id.unwrap(), 1);
                }
                Op::ConstantFalse => {
                    values.insert(inst.result_id.unwrap(), 0);
                }
                Op::Variable => {
                    if let Some(Operand::StorageClass(StorageClass::Workgroup)) =
                        inst.operands.first()
                    {
                        values.insert(inst.result_id.unwrap(), next_shared);
                        next_shared += 0x10000;
                    }
                }
                _ => {}
            }
        }
        let function = &module.functions[0];
        for (index, param) in function.parameters.iter().enumerate() {
            values.insert(param.result_id.unwrap(), machine.params[index]);
            let pw = widths[&param.result_type.unwrap()];
            VBITS.with(|m| m.borrow_mut().insert(param.result_id.unwrap(), pw));
        }
        // Blocks by label.
        let blocks: HashMap<u32, usize> = function
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.label.as_ref().unwrap().result_id.unwrap(), i))
            .collect();
        let mut next_slot = 0xF000_0000u64;
        for block in &function.blocks {
            for inst in &block.instructions {
                if inst.class.opcode == Op::Variable {
                    values.insert(inst.result_id.unwrap(), next_slot);
                    VBITS.with(|m| m.borrow_mut().insert(inst.result_id.unwrap(), 64));
                    next_slot += 0x1_0000;
                }
            }
        }
        let mut current = 0usize;
        let mut previous = 0u32;
        let mut steps = 0;
        'blocks: loop {
            let block = &function.blocks[current];
            let label = block.label.as_ref().unwrap().result_id.unwrap();
            for inst in &block.instructions {
                steps += 1;
                assert!(steps < 200_000, "interpreter step limit");
                if let Some(done) = step(inst, &mut values, &widths, &pointees, machine, &blocks, previous, label, &mut current) {
                    if done {
                        break 'blocks;
                    }
                    previous = label;
                    continue 'blocks;
                }
            }
            panic!("block without terminator");
        }
    }

    /// Returns `None` for a plain instruction, `Some(false)` after a branch
    /// (with `current` updated) and `Some(true)` on return.
    #[allow(clippy::too_many_arguments)]
    fn step(
        inst: &Instruction,
        values: &mut HashMap<u32, u64>,
        widths: &HashMap<u32, u32>,
        pointees: &HashMap<u32, u32>,
        machine: &mut Machine,
        blocks: &HashMap<u32, usize>,
        _previous: u32,
        _label: u32,
        current: &mut usize,
    ) -> Option<bool> {
        let rt = inst.result_type;
        let w = rt.and_then(|t| widths.get(&t).copied()).unwrap_or(0);
        let get = |values: &HashMap<u32, u64>, i: usize| -> u64 {
            *values
                .get(&id(&inst.operands[i]))
                .unwrap_or_else(|| panic!("undefined id in {:?}", inst.class.opcode))
        };
        let result = match inst.class.opcode {
            Op::Variable => return None,
            Op::Load => {
                let addr = get(values, 0);
                machine.read(addr, w.max(8) / 8)
            }
            Op::Store => {
                let addr = get(values, 0);
                let value = get(values, 1);
                // Width comes from the stored value's type; track via the id's definition.
                let bytes = stored_bytes(inst, values);
                machine.write(addr, bytes, value);
                return None;
            }
            Op::SelectionMerge | Op::LoopMerge | Op::Nop | Op::Line | Op::NoLine => return None,
            Op::Branch => {
                *current = blocks[&id(&inst.operands[0])];
                return Some(false);
            }
            Op::BranchConditional => {
                let cond = get(values, 0) != 0;
                *current = blocks[&id(&inst.operands[if cond { 1 } else { 2 }])];
                return Some(false);
            }
            Op::Return => return Some(true),
            Op::Bitcast | Op::ConvertPtrToU | Op::ConvertUToPtr | Op::GenericCastToPtr
            | Op::PtrCastToGeneric | Op::CopyObject => get(values, 0),
            Op::UConvert => mask(w, get(values, 0)),
            Op::SConvert => {
                let sw = value_width(inst, values, widths);
                mask(w, sext(sw, get(values, 0)) as u64)
            }
            Op::IAdd => mask(w, get(values, 0).wrapping_add(get(values, 1))),
            Op::ISub => mask(w, get(values, 0).wrapping_sub(get(values, 1))),
            Op::IMul => mask(w, get(values, 0).wrapping_mul(get(values, 1))),
            Op::UDiv => mask(w, get(values, 0) / get(values, 1)),
            Op::UMod => mask(w, get(values, 0) % get(values, 1)),
            Op::SDiv => mask(w, (sext(w, get(values, 0)) / sext(w, get(values, 1))) as u64),
            Op::SRem => mask(w, (sext(w, get(values, 0)) % sext(w, get(values, 1))) as u64),
            Op::ShiftLeftLogical => {
                let s = get(values, 1);
                assert!(s < u64::from(w), "shift amount {s} out of range for {w} bits");
                mask(w, get(values, 0) << s)
            }
            Op::ShiftRightLogical => {
                let s = get(values, 1);
                assert!(s < u64::from(w), "shift amount {s} out of range for {w} bits");
                get(values, 0) >> s
            }
            Op::ShiftRightArithmetic => {
                let s = get(values, 1);
                assert!(s < u64::from(w), "shift amount {s} out of range for {w} bits");
                mask(w, (sext(w, get(values, 0)) >> s) as u64)
            }
            Op::BitwiseAnd => get(values, 0) & get(values, 1),
            Op::BitwiseOr => get(values, 0) | get(values, 1),
            Op::BitwiseXor => get(values, 0) ^ get(values, 1),
            Op::Not => mask(w, !get(values, 0)),
            Op::BitCount => u64::from(get(values, 0).count_ones()),
            Op::LogicalAnd => get(values, 0) & get(values, 1),
            Op::LogicalOr => get(values, 0) | get(values, 1),
            Op::LogicalNot => get(values, 0) ^ 1,
            Op::Select => {
                if get(values, 0) != 0 { get(values, 1) } else { get(values, 2) }
            }
            Op::IEqual => u64::from(get(values, 0) == get(values, 1)),
            Op::INotEqual => u64::from(get(values, 0) != get(values, 1)),
            Op::ULessThan => u64::from(get(values, 0) < get(values, 1)),
            Op::ULessThanEqual => u64::from(get(values, 0) <= get(values, 1)),
            Op::UGreaterThan => u64::from(get(values, 0) > get(values, 1)),
            Op::UGreaterThanEqual => u64::from(get(values, 0) >= get(values, 1)),
            Op::SLessThan | Op::SLessThanEqual | Op::SGreaterThan | Op::SGreaterThanEqual => {
                let sw = value_width(inst, values, widths);
                let (a, b) = (sext(sw, get(values, 0)), sext(sw, get(values, 1)));
                u64::from(match inst.class.opcode {
                    Op::SLessThan => a < b,
                    Op::SLessThanEqual => a <= b,
                    Op::SGreaterThan => a > b,
                    _ => a >= b,
                })
            }
            Op::AccessChain | Op::InBoundsAccessChain => get(values, 0),
            Op::PtrAccessChain | Op::InBoundsPtrAccessChain => {
                // Element size is the pointee size of the base pointer's type.
                let element = element_bytes(inst, values, widths, pointees);
                get(values, 0).wrapping_add((sext(64, get(values, 1)) as u64).wrapping_mul(element))
            }
            Op::ExtInst => {
                let n = lit(&inst.operands[1]);
                let arg = |i: usize| get(values, 2 + i);
                use rspirv::spirv::OpenclStd100Op as C;
                let op = n;
                if op == C::s_min as u32 {
                    mask(w, sext(w, arg(0)).min(sext(w, arg(1))) as u64)
                } else if op == C::s_max as u32 {
                    mask(w, sext(w, arg(0)).max(sext(w, arg(1))) as u64)
                } else if op == C::u_min as u32 {
                    arg(0).min(arg(1))
                } else if op == C::u_max as u32 {
                    arg(0).max(arg(1))
                } else if op == C::s_abs as u32 {
                    mask(w, sext(w, arg(0)).unsigned_abs())
                } else if op == C::clz as u32 {
                    u64::from((arg(0) << (64 - w)).leading_zeros().min(w))
                } else if op == C::u_mul_hi as u32 {
                    ((u128::from(arg(0)) * u128::from(arg(1))) >> w) as u64
                } else if op == C::s_mul_hi as u32 {
                    mask(w, ((i128::from(sext(w, arg(0))) * i128::from(sext(w, arg(1)))) >> w) as u64)
                } else {
                    panic!("unsupported OpenCL.std instruction {op}")
                }
            }
            other => panic!("interpreter does not support {other:?}"),
        };
        values.insert(inst.result_id.unwrap(), result);
        VBITS.with(|m| m.borrow_mut().insert(inst.result_id.unwrap(), w));
        None
    }

    thread_local! {
        /// Bit width of every value id defined so far (pointers are 64).
        static VBITS: std::cell::RefCell<HashMap<u32, u32>> = Default::default();
    }

    fn bits_of(id: u32) -> u32 {
        VBITS.with(|m| m.borrow().get(&id).copied().unwrap_or(32))
    }
    fn stored_bytes(inst: &Instruction, _: &HashMap<u32, u64>) -> u32 {
        bits_of(super::interp::id(&inst.operands[1])).max(8) / 8
    }
    fn value_width(inst: &Instruction, _: &HashMap<u32, u64>, _: &HashMap<u32, u32>) -> u32 {
        bits_of(super::interp::id(&inst.operands[0]))
    }
    fn element_bytes(
        inst: &Instruction,
        _: &HashMap<u32, u64>,
        widths: &HashMap<u32, u32>,
        pointees: &HashMap<u32, u32>,
    ) -> u64 {
        let pointee = pointees[&inst.result_type.unwrap()];
        u64::from(widths.get(&pointee).copied().unwrap_or(8).max(8) / 8)
    }
}

/// Runs a `kernel()` body on the interpreter; returns the first `words`
/// 32-bit words written to the output buffer.
pub(super) fn eval(regs: &str, body: &str, n: u32, words: usize) -> Vec<u32> {
    let module = lower(&kernel(regs, body));
    let mut machine = interp::Machine {
        params: vec![interp::GLOBAL, u64::from(n)],
        ..Default::default()
    };
    interp::run(&module, &mut machine);
    (0..words)
        .map(|i| machine.read(interp::GLOBAL + 4 * i as u64, 4) as u32)
        .collect()
}

#[test]
fn integer_minmax_xor_abs_popc_clz() {
    let out = eval(
        "",
        r#"    mov.u32 %r1, 0xfffffff0;
    mov.u32 %r2, 5;
    min.s32 %r3, %r1, %r2;
    max.s32 %r4, %r1, %r2;
    min.u32 %r5, %r1, %r2;
    max.u32 %r6, %r1, %r2;
    xor.b32 %r7, %r1, %r2;
    abs.s32 %r8, %r1;
    popc.b32 %r9, %r1;
    clz.b32 %r10, %r2;
    mul.hi.s32 %r11, %r1, %r2;
    add.u32 %r12, %r2, %r2;
    sub.u32 %r13, %r2, %r1;
    mul.lo.u32 %r14, %r2, %r2;
    shr.b32 %r15, %r1, 28;
    st.global.u32 [%rd15], %r3;
    st.global.u32 [%rd15+4], %r4;
    st.global.u32 [%rd15+8], %r5;
    st.global.u32 [%rd15+12], %r6;
    st.global.u32 [%rd15+16], %r7;
    st.global.u32 [%rd15+20], %r8;
    st.global.u32 [%rd15+24], %r9;
    st.global.u32 [%rd15+28], %r10;
    st.global.u32 [%rd15+32], %r11;
    st.global.u32 [%rd15+36], %r12;
    st.global.u32 [%rd15+40], %r13;
    st.global.u32 [%rd15+44], %r14;
    st.global.u32 [%rd15+48], %r15;"#,
        0,
        13,
    );
    assert_eq!(
        out,
        [
            0xfffffff0, 5, 5, 0xfffffff0, 0xfffffff5, 16, 28, 29, 0xffffffff, 10, 21, 25, 15
        ]
    );
}

#[test]
fn bit_field_extract_insert_permute_and_lop3() {
    let out = eval(
        "",
        r#"    mov.u32 %r1, 0x8f3c5a96;
    bfe.u32 %r2, %r1, 4, 8;
    bfe.s32 %r3, %r1, 4, 8;
    bfe.s32 %r4, %r1, 28, 4;
    bfe.u32 %r5, %r1, 40, 8;
    bfe.u32 %r6, %r1, 0, 0;
    mov.u32 %r7, 0xab;
    bfi.b32 %r8, %r7, %r1, 8, 8;
    bfi.b32 %r9, %r7, %r1, 40, 8;
    mov.u32 %r10, 0x11223344;
    prmt.b32 %r11, %r1, %r10, 0x7531;
    prmt.b32 %r12, %r1, %r10, 0xba98;
    lop3.b32 %r13, %r1, %r10, %r7, 0xe8;
    lop3.b32 %r14, %r1, %r10, %r7, 0x96;
    st.global.u32 [%rd15], %r2;
    st.global.u32 [%rd15+4], %r3;
    st.global.u32 [%rd15+8], %r4;
    st.global.u32 [%rd15+12], %r5;
    st.global.u32 [%rd15+16], %r6;
    st.global.u32 [%rd15+20], %r8;
    st.global.u32 [%rd15+24], %r9;
    st.global.u32 [%rd15+28], %r11;
    st.global.u32 [%rd15+32], %r12;
    st.global.u32 [%rd15+36], %r13;
    st.global.u32 [%rd15+40], %r14;"#,
        0,
        11,
    );
    let (a, b, c) = (0x8f3c5a96u32, 0x11223344u32, 0xabu32);
    // prmt: bytes of {b:a} = [96 5a 3c 8f 44 33 22 11]; 0x7531 picks bytes 1,3,5,7 (low nibble first).
    assert_eq!(
        out,
        [
            0xa9,
            0xffffffa9,
            0xfffffff8,
            0,
            0,
            0x8f3cab96,
            a,
            0x1133_8f5a,
            // selectors 8..b replicate the sign of bytes 0..3: 0x96->ff, 0x5a->00, 0x3c->00, 0x8f->ff
            0xff00_00ff,
            (a & b) | (a & c) | (b & c),
            a ^ b ^ c,
        ]
    );
}
