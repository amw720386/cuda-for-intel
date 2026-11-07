//! Instruction-coverage tests. Every snippet is translated and, when a
//! `spirv-val` executable is available (`SPIRV_VAL` or `PATH`), validated.
use super::*;
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::spirv::{ExecutionMode, Op};

/// Wraps `body` in a kernel with one global output pointer (`%rdo`, already
/// converted to a global address) and the given extra register declarations.
pub(super) fn kernel(regs: &str, body: &str) -> String {
    kernel_with_shared(regs, "", body)
}

pub(super) fn kernel_with_shared(regs: &str, shared: &str, body: &str) -> String {
    format!(
        r#".version 9.0
.target sm_75
.address_size 64
.visible .entry probe(.param .u64 out_param, .param .u32 n_param)
{{
    .reg .pred %p<8>;
    .reg .b32 %r<64>;
    .reg .b64 %rd<16>;
    .reg .f32 %f<16>;
    {regs}
    {shared}
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
        run_lanes(module, machine, 1);
    }

    /// Runs `lanes` invocations of one subgroup in lockstep (uniform control
    /// flow only). Invocation `l` has `tid.x == l` and a workgroup of `lanes`.
    pub fn run_lanes(module: &Module, machine: &mut Machine, lanes: usize) {
        let mut widths: HashMap<u32, u32> = HashMap::new();
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
                Op::TypeVector => {
                    let element = widths[&id(&inst.operands[0])];
                    widths.insert(inst.result_id.unwrap(), element * lit(&inst.operands[1]));
                }
                _ => {}
            }
        }
        let mut shared: HashMap<u32, u64> = HashMap::new();
        let mut constants: HashMap<u32, u64> = HashMap::new();
        let mut builtin_vars: Vec<u32> = Vec::new();
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
                    constants.insert(inst.result_id.unwrap(), mask(w, v));
                    VBITS.with(|m| m.borrow_mut().insert(inst.result_id.unwrap(), w));
                }
                Op::ConstantTrue => {
                    constants.insert(inst.result_id.unwrap(), 1);
                }
                Op::ConstantFalse => {
                    constants.insert(inst.result_id.unwrap(), 0);
                }
                Op::Variable => match inst.operands.first() {
                    Some(Operand::StorageClass(StorageClass::Workgroup)) => {
                        shared.insert(inst.result_id.unwrap(), next_shared);
                        next_shared += 0x10000;
                    }
                    Some(Operand::StorageClass(StorageClass::Input)) => {
                        builtin_vars.push(inst.result_id.unwrap());
                    }
                    _ => {}
                },
                _ => {}
            }
        }
        // Built-in decorations: order of BUILTINS is ctaid, ntid, tid, nctaid, but
        // look the kind up from the decoration to stay independent of it.
        let mut kind_of: HashMap<u32, rspirv::spirv::BuiltIn> = HashMap::new();
        for inst in &module.annotations {
            if inst.class.opcode == Op::Decorate
                && let [Operand::IdRef(target), Operand::Decoration(rspirv::spirv::Decoration::BuiltIn), Operand::BuiltIn(kind)] =
                    inst.operands[..]
            {
                kind_of.insert(target, kind);
            }
        }
        let function = &module.functions[0];
        let blocks: HashMap<u32, usize> = function
            .blocks
            .iter()
            .enumerate()
            .map(|(i, b)| (b.label.as_ref().unwrap().result_id.unwrap(), i))
            .collect();
        let mut states: Vec<HashMap<u32, u64>> = Vec::new();
        for lane in 0..lanes as u64 {
            let mut values = constants.clone();
            values.extend(shared.iter().map(|(k, v)| (*k, *v)));
            for (index, param) in function.parameters.iter().enumerate() {
                values.insert(param.result_id.unwrap(), machine.params[index]);
                let pw = widths[&param.result_type.unwrap()];
                VBITS.with(|m| m.borrow_mut().insert(param.result_id.unwrap(), pw));
            }
            for (index, var) in builtin_vars.iter().enumerate() {
                let address = 0xE000_0000 + (lane << 20) + index as u64 * 0x100;
                let components = match kind_of.get(var) {
                    Some(rspirv::spirv::BuiltIn::LocalInvocationId) => [lane, 0, 0],
                    Some(rspirv::spirv::BuiltIn::WorkgroupSize) => [lanes as u64, 1, 1],
                    Some(rspirv::spirv::BuiltIn::NumWorkgroups) => [1, 1, 1],
                    _ => [0, 0, 0],
                };
                for (i, c) in components.iter().enumerate() {
                    machine.write(address + 8 * i as u64, 8, *c);
                }
                values.insert(*var, address);
            }
            let mut slot = 0xF000_0000u64 + (lane << 28);
            for block in &function.blocks {
                for inst in &block.instructions {
                    if inst.class.opcode == Op::Variable {
                        values.insert(inst.result_id.unwrap(), slot);
                        VBITS.with(|m| m.borrow_mut().insert(inst.result_id.unwrap(), 64));
                        slot += 0x1_0000;
                    }
                }
            }
            states.push(values);
        }
        let mut current = 0usize;
        let mut steps = 0;
        'blocks: loop {
            let block = &function.blocks[current];
            let label = block.label.as_ref().unwrap().result_id.unwrap();
            for inst in &block.instructions {
                steps += 1;
                assert!(steps < 200_000, "interpreter step limit");
                if inst.class.opcode == Op::GroupNonUniformShuffle {
                    let results: Vec<u64> = (0..lanes)
                        .map(|l| {
                            let source = *states[l].get(&id(&inst.operands[2])).unwrap() as usize;
                            *states[source].get(&id(&inst.operands[1])).unwrap()
                        })
                        .collect();
                    for (l, r) in results.into_iter().enumerate() {
                        states[l].insert(inst.result_id.unwrap(), r);
                    }
                    VBITS.with(|m| m.borrow_mut().insert(inst.result_id.unwrap(), 32));
                    continue;
                }
                let mut outcome = None;
                let mut next = current;
                for values in states.iter_mut() {
                    let mut target = current;
                    let r = step(inst, values, &widths, &pointees, machine, &blocks, 0, label, &mut target);
                    if r.is_some() {
                        assert!(outcome.is_none() || next == target, "divergent control flow");
                        outcome = r;
                        next = target;
                    }
                }
                if let Some(done) = outcome {
                    if done {
                        break 'blocks;
                    }
                    current = next;
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
                if w > 64 {
                    // Vector built-ins stay in memory; the value is their address.
                    addr
                } else {
                    machine.read(addr, w.max(8) / 8)
                }
            }
            Op::Store => {
                let addr = get(values, 0);
                let value = get(values, 1);
                // Width comes from the stored value's type; track via the id's definition.
                let bytes = stored_bytes(inst, values);
                machine.write(addr, bytes, value);
                return None;
            }
            Op::ControlBarrier | Op::MemoryBarrier | Op::SelectionMerge | Op::LoopMerge | Op::Nop | Op::Line | Op::NoLine => return None,
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
            Op::CompositeExtract => {
                let index = lit(&inst.operands[1]) as u64;
                machine.read(get(values, 0) + 8 * index, 8)
            }
            Op::FConvert => {
                let sw = value_width(inst, values, widths);
                assert_eq!((sw, w), (16, 32), "only half -> f32 is modelled");
                let h = get(values, 0) as u32;
                let (sign, exp, man) = (h >> 15, (h >> 10) & 0x1f, h & 0x3ff);
                let magnitude = match exp {
                    0 => man as f32 * 2f32.powi(-24),
                    0x1f => f32::INFINITY,
                    _ => (1.0 + man as f32 / 1024.0) * 2f32.powi(exp as i32 - 15),
                };
                u64::from((if sign == 1 { -magnitude } else { magnitude }).to_bits())
            }
            Op::FMul => u64::from((f32::from_bits(get(values, 0) as u32) * f32::from_bits(get(values, 1) as u32)).to_bits()),
            Op::FAdd => u64::from((f32::from_bits(get(values, 0) as u32) + f32::from_bits(get(values, 1) as u32)).to_bits()),
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

/// Output buffer base used by lockstep tests; the input buffer sits 64 KiB above.
const OUT: u64 = interp::GLOBAL;
const IN: u64 = interp::GLOBAL + 0x10000;

/// Runs a 32-lane kernel after letting `init` fill the input buffer.
fn eval_warp(regs: &str, shared: &str, body: &str, init: impl FnOnce(&mut interp::Machine)) -> interp::Machine {
    let module = lower(&kernel_with_shared(regs, shared, body));
    let mut machine = interp::Machine {
        params: vec![OUT, 0],
        ..Default::default()
    };
    init(&mut machine);
    interp::run_lanes(&module, &mut machine, 32);
    machine
}

fn f16_bits(value: f32) -> u16 {
    // Exact for the small integers used in tests.
    if value == 0.0 {
        return 0;
    }
    let bits = value.abs().to_bits();
    let exponent = ((bits >> 23) as i32 - 127 + 15) as u16;
    let mantissa = ((bits >> 13) & 0x3ff) as u16;
    (u16::from(value < 0.0) << 15) | (exponent << 10) | mantissa
}

/// Integer MMA reference: layouts follow the PTX ISA fragment tables.
fn check_int8_mma(shape: &str, m: usize, k: usize, a_signed: bool, b_signed: bool) {
    let n = 8;
    let regs_a = (m / 8) * (k / 16);
    let regs_b = k / 16;
    let regs_c = m * n / 32;
    let a_at = |r: usize, c: usize| -> i32 {
        let v = ((r * 7 + c * 3 + 1) % 23) as i32 - if a_signed { 11 } else { 0 };
        if a_signed { v } else { v & 0xff }
    };
    let b_at = |r: usize, c: usize| -> i32 {
        let v = ((r * 5 + c * 11 + 2) % 29) as i32 - if b_signed { 14 } else { 0 };
        if b_signed { v } else { v & 0xff }
    };
    let c_at = |r: usize, c: usize| -> i32 { (r * 100 + c) as i32 - 300 };
    let byte = |v: i32| (v as u32) & 0xff;
    // Per-lane input words: a regs, b regs, c regs.
    let stride = 16;
    let a_name = if a_signed { "s8" } else { "u8" };
    let b_name = if b_signed { "s8" } else { "u8" };
    let list = |prefix: &str, base: usize, count: usize| {
        (0..count).map(|i| format!("%r{}", base + i)).collect::<Vec<_>>().join(", ") + prefix
    };
    let loads: String = (0..regs_a + regs_b + regs_c)
        .map(|i| format!("    ld.global.u32 %r{}, [%rd14+{}];\n", 10 + i, 4 * i))
        .collect();
    let stores: String = (0..regs_c)
        .map(|i| format!("    st.global.u32 [%rd13+{}], %r{};\n", 4 * i, 20 + i))
        .collect();
    let body = format!(
        "    mov.u32 %r1, %laneid;\n    mul.wide.u32 %rd12, %r1, {bytes};\n    add.s64 %rd14, %rd15, 65536;\n    add.s64 %rd14, %rd14, %rd12;\n    mul.wide.u32 %rd11, %r1, {out_bytes};\n    add.s64 %rd13, %rd15, %rd11;\n{loads}    mma.sync.aligned.{shape}.row.col.s32.{a_name}.{b_name}.s32 {{{d}}}, {{{a}}}, {{{b}}}, {{{c}}};\n{stores}",
        bytes = stride * 4,
        out_bytes = regs_c * 4,
        d = list("", 20, regs_c),
        a = list("", 10, regs_a),
        b = list("", 10 + regs_a, regs_b),
        c = list("", 10 + regs_a + regs_b, regs_c),
    );
    let machine = eval_warp("", "", &body, |machine| {
        for lane in 0..32usize {
            let (row, quad) = (lane / 4, lane % 4);
            let mut words = Vec::new();
            for reg in 0..regs_a {
                let (half, high) = (reg % 2, reg / 2);
                let mut word = 0u32;
                for i in 0..4 {
                    word |= byte(a_at(row + 8 * half, quad * 4 + i + 16 * high)) << (8 * i);
                }
                words.push(word);
            }
            for reg in 0..regs_b {
                let mut word = 0u32;
                for i in 0..4 {
                    word |= byte(b_at(quad * 4 + i + 16 * reg, row)) << (8 * i);
                }
                words.push(word);
            }
            for reg in 0..regs_c {
                words.push(c_at(row + 8 * (reg / 2), quad * 2 + reg % 2) as u32);
            }
            for (i, w) in words.iter().enumerate() {
                machine.write(IN + (lane * stride * 4 + i * 4) as u64, 4, u64::from(*w));
            }
        }
    });
    for lane in 0..32usize {
        let (row, quad) = (lane / 4, lane % 4);
        for reg in 0..regs_c {
            let (r, c) = (row + 8 * (reg / 2), quad * 2 + reg % 2);
            let expected: i32 =
                c_at(r, c) + (0..k).map(|x| a_at(r, x) * b_at(x, c)).sum::<i32>();
            let got = machine.read(OUT + (lane * regs_c * 4 + reg * 4) as u64, 4) as u32 as i32;
            assert_eq!(got, expected, "{shape} lane {lane} reg {reg}");
        }
    }
}

#[test]
fn mma_m8n8k16_s8_matches_reference() {
    check_int8_mma("m8n8k16", 8, 16, true, true);
}

#[test]
fn mma_m16n8k16_s8_matches_reference() {
    check_int8_mma("m16n8k16", 16, 16, true, true);
}

#[test]
fn mma_m16n8k32_mixed_signedness_matches_reference() {
    check_int8_mma("m16n8k32", 16, 32, true, true);
    check_int8_mma("m16n8k32", 16, 32, false, true);
    check_int8_mma("m16n8k32", 16, 32, true, false);
}

fn check_f16_mma(shape: &str, k: usize) {
    let (m, n) = (16usize, 8usize);
    let regs_a = 2 * (k / 8);
    let regs_b = k / 8;
    let a_at = |r: usize, c: usize| ((r + 2 * c) % 7) as f32 - 3.0;
    let b_at = |r: usize, c: usize| ((3 * r + c) % 5) as f32 - 2.0;
    let c_at = |r: usize, c: usize| (r * 8 + c) as f32 * 0.5;
    let pack = |lo: f32, hi: f32| u32::from(f16_bits(lo)) | (u32::from(f16_bits(hi)) << 16);
    let loads: String = (0..regs_a + regs_b + 4)
        .map(|i| format!("    ld.global.u32 %r{}, [%rd14+{}];\n", 10 + i, 4 * i))
        .collect();
    let list = |base: usize, count: usize| {
        (0..count).map(|i| format!("%r{}", base + i)).collect::<Vec<_>>().join(", ")
    };
    let stores: String = (0..4)
        .map(|i| format!("    st.global.u32 [%rd13+{}], %r{};\n", 4 * i, 30 + i))
        .collect();
    let body = format!(
        "    mov.u32 %r1, %laneid;\n    mul.wide.u32 %rd12, %r1, 64;\n    add.s64 %rd14, %rd15, 65536;\n    add.s64 %rd14, %rd14, %rd12;\n    mul.wide.u32 %rd11, %r1, 16;\n    add.s64 %rd13, %rd15, %rd11;\n{loads}    mma.sync.aligned.{shape}.row.col.f32.f16.f16.f32 {{{d}}}, {{{a}}}, {{{b}}}, {{{c}}};\n{stores}",
        d = list(30, 4),
        a = list(10, regs_a),
        b = list(10 + regs_a, regs_b),
        c = list(10 + regs_a + regs_b, 4),
    );
    let machine = eval_warp("", "", &body, |machine| {
        for lane in 0..32usize {
            let (row, quad) = (lane / 4, lane % 4);
            let mut words = Vec::new();
            for reg in 0..regs_a {
                let (half, high) = (reg % 2, reg / 2);
                let (r, c) = (row + 8 * half, quad * 2 + 8 * high);
                words.push(pack(a_at(r, c), a_at(r, c + 1)));
            }
            for reg in 0..regs_b {
                let (r, c) = (quad * 2 + 8 * reg, row);
                words.push(pack(b_at(r, c), b_at(r + 1, c)));
            }
            for reg in 0..4 {
                words.push(c_at(row + 8 * (reg / 2), quad * 2 + reg % 2).to_bits());
            }
            for (i, w) in words.iter().enumerate() {
                machine.write(IN + (lane * 64 + i * 4) as u64, 4, u64::from(*w));
            }
        }
    });
    let _ = (m, n);
    for lane in 0..32usize {
        let (row, quad) = (lane / 4, lane % 4);
        for reg in 0..4 {
            let (r, c) = (row + 8 * (reg / 2), quad * 2 + reg % 2);
            let expected = c_at(r, c) + (0..k).map(|x| a_at(r, x) * b_at(x, c)).sum::<f32>();
            let got = f32::from_bits(machine.read(OUT + (lane * 16 + reg * 4) as u64, 4) as u32);
            assert_eq!(got, expected, "{shape} lane {lane} reg {reg}");
        }
    }
}

#[test]
fn mma_f16_shapes_match_reference() {
    check_f16_mma("m16n8k8", 8);
    check_f16_mma("m16n8k16", 16);
}

#[test]
fn signed_subword_loads_and_shared_widths() {
    let machine = eval_warp(
        "",
        ".shared .align 8 .b8 smem[64];",
        r#"    mov.u32 %r1, smem;
    mov.u32 %r2, 0xfedcba98;
    st.shared.u32 [%r1], %r2;
    st.shared.u32 [%r1+4], %r2;
    ld.shared.s8 %r3, [%r1];
    ld.shared.u16 %r4, [%r1+2];
    ld.shared.s16 %r5, [%r1+2];
    ld.shared.u64 %rd1, [%r1];
    ld.shared.u8 %r6, [%r1+3];
    st.shared.u16 [%r1+8], %r2;
    ld.shared.u32 %r7, [%r1+8];
    mov.u64 %rd2, 0x1122334455667788;
    st.shared.u64 [%r1+16], %rd2;
    ld.shared.u32 %r8, [%r1+20];
    st.global.u32 [%rd15], %r3;
    st.global.u32 [%rd15+4], %r4;
    st.global.u32 [%rd15+8], %r5;
    st.global.u64 [%rd15+16], %rd1;
    st.global.u32 [%rd15+12], %r6;
    st.global.u32 [%rd15+24], %r7;
    st.global.u32 [%rd15+28], %r8;"#,
        |_| {},
    );
    let word = |offset: u64| machine.read(OUT + offset, 4) as u32;
    // Every lane stores the same values, so lane order does not matter.
    assert_eq!(word(0), 0xffff_ff98);
    assert_eq!(word(4), 0xfedc);
    assert_eq!(word(8), 0xffff_fedc);
    assert_eq!(machine.read(OUT + 16, 8), 0xfedc_ba98_fedc_ba98);
    assert_eq!(word(12), 0xfe);
    assert_eq!(word(24) & 0xffff, 0xba98);
    assert_eq!(word(28), 0x1122_3344);
}

#[test]
fn shared_address_conversions_round_trip_through_generic_pointers() {
    // The nvcc idiom for __cvta_generic_to_shared and its inverse.
    let machine = eval_warp(
        "",
        ".shared .align 4 .b8 first[16];\n    .shared .align 4 .b8 second[16];",
        r#"    mov.u32 %r1, second;
    {
        .reg .b64 u64addr;
        cvt.u64.u32 u64addr, %r1;
        cvta.shared.u64 %rd1, u64addr;
    }
    mov.u32 %r2, 0x600dcafe;
    st.u32 [%rd1+4], %r2;
    ld.u32 %r3, [%rd1+4];
    ld.shared.u32 %r4, [%r1+4];
    {
        .reg .u64 tmp;
        cvta.to.shared.u64 tmp, %rd1;
        cvt.u32.u64 %r5, tmp;
    }
    st.global.u32 [%rd15], %r3;
    st.global.u32 [%rd15+4], %r4;
    st.global.u32 [%rd15+8], %r5;
    st.global.u32 [%rd15+12], %r1;"#,
        |_| {},
    );
    assert_eq!(machine.read(OUT, 4), 0x600d_cafe);
    assert_eq!(machine.read(OUT + 4, 4), 0x600d_cafe);
    assert_eq!(machine.read(OUT + 8, 4), 16);
    assert_eq!(machine.read(OUT + 12, 4), 16);
}

#[test]
fn vector_u32_accesses_in_all_spaces_validate() {
    lower(&kernel_with_shared(
        "",
        ".shared .align 16 .b8 smem[64];",
        r#"    mov.u32 %r1, smem;
    ld.global.nc.v4.u32 {%r2, %r3, %r4, %r5}, [%rd15];
    st.shared.v4.u32 [%r1], {%r2, %r3, %r4, %r5};
    ld.shared.v2.u32 {%r6, %r7}, [%r1+8];
    st.global.v4.b32 [%rd15+16], {%r6, %r7, %r2, %r3};
    ld.v4.u32 {%r8, %r9, %r10, %r11}, [%rd15+16];"#,
    ));
}

fn ldmatrix_case(count: u8, trans: bool) {
    let regs = (0..count).map(|i| format!("%r{}", 20 + i)).collect::<Vec<_>>().join(", ");
    let trans_suffix = if trans { ".trans" } else { "" };
    let stores: String = (0..count)
        .map(|i| format!("    st.global.u32 [%rd13+{}], %r{};\n", 4 * i, 20 + i))
        .collect();
    let body = format!(
        r#"    mov.u32 %r1, %laneid;
    mov.u32 %r2, smem;
    shl.b32 %r3, %r1, 4;
    add.s32 %r3, %r3, %r2;
    add.s64 %rd14, %rd15, 65536;
    mul.wide.u32 %rd12, %r1, 16;
    add.s64 %rd14, %rd14, %rd12;
    ld.global.u32 %r4, [%rd14];
    ld.global.u32 %r5, [%rd14+4];
    ld.global.u32 %r6, [%rd14+8];
    ld.global.u32 %r7, [%rd14+12];
    st.shared.u32 [%r3], %r4;
    st.shared.u32 [%r3+4], %r5;
    st.shared.u32 [%r3+8], %r6;
    st.shared.u32 [%r3+12], %r7;
    bar.sync 0;
    shr.u32 %r8, %r1, 3;
    shl.b32 %r8, %r8, 7;
    and.b32 %r9, %r1, 7;
    shl.b32 %r9, %r9, 4;
    add.s32 %r8, %r8, %r9;
    add.s32 %r8, %r8, %r2;
    ldmatrix.sync.aligned.m8n8.x{count}{trans_suffix}.shared.b16 {{{regs}}}, [%r8];
    mul.wide.u32 %rd11, %r1, 16;
    add.s64 %rd13, %rd15, %rd11;
{stores}"#
    );
    let machine = eval_warp("", ".shared .align 16 .b8 smem[512];", &body, |machine| {
        // 512 bytes of shared memory = four 8x8 b16 matrices (row-major, 16 B rows).
        for byte in 0..512u64 {
            machine.write(IN + byte, 1, (byte * 37 + 11) & 0xff);
        }
    });
    let element = |matrix: u64, row: u64, col: u64| -> u32 {
        let at = (matrix * 128 + row * 16 + col * 2) as u64;
        (((at * 37 + 11) & 0xff) | ((((at + 1) * 37 + 11) & 0xff) << 8)) as u32
    };
    for lane in 0..32u64 {
        for matrix in 0..u64::from(count) {
            let (row, quad) = (lane / 4, lane % 4);
            let expected = if trans {
                element(matrix, quad * 2, row) | (element(matrix, quad * 2 + 1, row) << 16)
            } else {
                element(matrix, row, quad * 2) | (element(matrix, row, quad * 2 + 1) << 16)
            };
            let got = machine.read(OUT + lane * 16 + matrix * 4, 4) as u32;
            assert_eq!(got, expected, "x{count} trans={trans} lane {lane} matrix {matrix}");
        }
    }
}

#[test]
fn ldmatrix_matches_reference_fragment_layout() {
    for count in [1, 2, 4] {
        ldmatrix_case(count, false);
        ldmatrix_case(count, true);
    }
}
