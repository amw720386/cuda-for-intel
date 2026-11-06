use super::flow::Flow;
use super::rounded;
use super::{
    Comparison, Entry, Error, FloatOp, Instruction, IntegerOp, Param, Space, Terminator,
    Translation, Width,
};
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Instruction as SpirvInstruction, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FPRoundingMode, FunctionControl, MemoryModel, MemorySemantics, Op, OpenclStd100Op, Scope,
    StorageClass, Word,
};
use std::collections::{HashMap, HashSet};

const BUILTINS: [BuiltIn; 4] = [
    BuiltIn::WorkgroupId,
    BuiltIn::WorkgroupSize,
    BuiltIn::LocalInvocationId,
    BuiltIn::NumWorkgroups,
];

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    Predicate,
    B32,
    B64,
    Address(Space),
}

impl Kind {
    fn width(self) -> Width {
        match self {
            Self::Predicate => Width::Predicate,
            Self::B32 => Width::B32,
            Self::B64 | Self::Address(_) => Width::B64,
        }
    }
}

#[derive(Clone, Copy)]
struct Value {
    id: Word,
    kind: Kind,
}

impl Value {
    fn require(self, kind: Kind, operand: &str) -> Result<Word, Error> {
        if self.kind != kind {
            return Err(Error::new(format!(
                "expected {kind:?} for {operand}, got {:?}",
                self.kind
            )));
        }
        Ok(self.id)
    }
}

struct Emitter<'a> {
    builder: Builder,
    bool_type: Word,
    u32_type: Word,
    u64_type: Word,
    f32_type: Word,
    global_byte_ptr: Word,
    global_word_ptr: Word,
    shared_byte_ptr: Word,
    shared_word_ptr: Word,
    shared: Option<(&'a str, Word)>,
    builtin_type: Word,
    builtins: [Word; BUILTINS.len()],
    declarations: HashMap<String, Width>,
    registers: HashMap<&'a str, Value>,
    initialized: HashSet<&'a str>,
    locals: Vec<SpirvInstruction>,
    params: HashMap<&'a str, Value>,
    device_atomics: bool,
    fp32: bool,
    fp32_fma: bool,
    opencl: Word,
}

impl<'a> Emitter<'a> {
    fn new(declarations: HashMap<String, Width>, shared: Option<(&'a str, u32)>) -> Self {
        let mut builder = Builder::new();
        builder.set_version(1, 0);
        for capability in [
            Capability::Addresses,
            Capability::Kernel,
            Capability::Int8,
            Capability::Int64,
        ] {
            builder.capability(capability);
        }
        builder.memory_model(AddressingModel::Physical64, MemoryModel::OpenCL);
        let opencl = builder.ext_inst_import("OpenCL.std");

        let bool_type = builder.type_bool();
        let u8_type = builder.type_int(8, 0);
        let u32_type = builder.type_int(32, 0);
        let u64_type = builder.type_int(64, 0);
        let f32_type = builder.type_float(32, None);
        let global_byte_ptr = builder.type_pointer(None, StorageClass::CrossWorkgroup, u8_type);
        let global_word_ptr = builder.type_pointer(None, StorageClass::CrossWorkgroup, u32_type);
        let shared_byte_ptr = builder.type_pointer(None, StorageClass::Workgroup, u8_type);
        let shared_word_ptr = builder.type_pointer(None, StorageClass::Workgroup, u32_type);
        let shared = shared.map(|(name, bytes)| {
            // Use a u32 array to retain the PTX declaration's four-byte alignment.
            let length = builder.constant_bit32(u32_type, bytes / 4);
            let array = builder.type_array(u32_type, length);
            let pointer = builder.type_pointer(None, StorageClass::Workgroup, array);
            let variable = builder.variable(pointer, None, StorageClass::Workgroup, None);
            (name, variable)
        });
        // OpenCL's Physical64 environment uses size_t vectors for these built-ins.
        let builtin_type = builder.type_vector(u64_type, 3);
        let builtin_ptr = builder.type_pointer(None, StorageClass::Input, builtin_type);
        let builtins = BUILTINS.map(|builtin| {
            let id = builder.variable(builtin_ptr, None, StorageClass::Input, None);
            builder.decorate(id, Decoration::BuiltIn, [Operand::BuiltIn(builtin)]);
            id
        });
        Self {
            builder,
            bool_type,
            u32_type,
            u64_type,
            f32_type,
            global_byte_ptr,
            global_word_ptr,
            shared_byte_ptr,
            shared_word_ptr,
            shared,
            builtin_type,
            builtins,
            declarations,
            registers: HashMap::new(),
            initialized: HashSet::new(),
            locals: Vec::new(),
            params: HashMap::new(),
            device_atomics: false,
            fp32: false,
            fp32_fma: false,
            opencl,
        }
    }

    fn param_type(&self, param: Param) -> Word {
        match param {
            Param::U32 => self.u32_type,
            Param::F32 => self.f32_type,
            Param::U64 => self.u64_type,
            Param::GlobalPtr => self.global_byte_ptr,
        }
    }

    fn value_type(&self, kind: Kind) -> Word {
        match kind {
            Kind::Predicate => self.bool_type,
            Kind::B32 => self.u32_type,
            Kind::B64 => self.u64_type,
            Kind::Address(Space::Global) => self.global_byte_ptr,
            Kind::Address(Space::Shared) => self.shared_byte_ptr,
        }
    }

    fn read(&mut self, operand: &str) -> Result<Value, Error> {
        let slot = self
            .registers
            .get(operand)
            .copied()
            .filter(|_| self.initialized.contains(operand))
            .ok_or_else(|| {
                Error::new(format!(
                    "undefined PTX register {operand} on an incoming path"
                ))
            })?;
        let id = self
            .builder
            .load(self.value_type(slot.kind), None, slot.id, None, [])?;
        Ok(Value {
            id,
            kind: slot.kind,
        })
    }

    fn write(&mut self, dst: &'a str, value: Value) -> Result<(), Error> {
        if self.declarations.get(dst) != Some(&value.kind.width()) {
            return Err(Error::new(format!(
                "missing or mismatched PTX declaration for {dst}"
            )));
        }
        let slot = if let Some(slot) = self.registers.get(dst).copied() {
            slot.require(value.kind, dst)?
        } else {
            let pointer = self.builder.type_pointer(
                None,
                StorageClass::Function,
                self.value_type(value.kind),
            );
            let slot = self.builder.id();
            // Register types are discovered during lowering. Emit the slots in the prologue.
            self.locals.push(SpirvInstruction::new(
                Op::Variable,
                Some(pointer),
                Some(slot),
                vec![Operand::StorageClass(StorageClass::Function)],
            ));
            self.registers.insert(
                dst,
                Value {
                    id: slot,
                    kind: value.kind,
                },
            );
            slot
        };
        self.builder.store(slot, value.id, None, [])?;
        self.initialized.insert(dst);
        Ok(())
    }

    fn integer(&mut self, operand: &str, kind: Kind) -> Result<Word, Error> {
        if operand.starts_with('%') {
            return self.read(operand)?.require(kind, operand);
        }
        match kind {
            Kind::B32 => operand
                .parse::<u32>()
                .ok()
                .map(|n| self.builder.constant_bit32(self.u32_type, n)),
            Kind::B64 => operand
                .parse()
                .ok()
                .map(|n| self.builder.constant_bit64(self.u64_type, n)),
            _ => None,
        }
        .ok_or_else(|| Error::new(format!("unsupported integer operand {operand}")))
    }

    fn builtin(&mut self, src: &str) -> Result<Word, Error> {
        let (name, axis) = src
            .split_once('.')
            .ok_or_else(|| Error::new(format!("unsupported PTX built-in {src}")))?;
        let builtin = match name {
            "%ctaid" => self.builtins[0],
            "%ntid" => self.builtins[1],
            "%tid" => self.builtins[2],
            "%nctaid" => self.builtins[3],
            _ => return Err(Error::new(format!("unsupported PTX built-in {src}"))),
        };
        let axis = match axis {
            "x" => 0,
            "y" => 1,
            "z" => 2,
            _ => return Err(Error::new(format!("unsupported PTX built-in {src}"))),
        };
        let vector = self
            .builder
            .load(self.builtin_type, None, builtin, None, [])?;
        let component = self
            .builder
            .composite_extract(self.u64_type, None, vector, [axis])?;
        Ok(self.builder.u_convert(self.u32_type, None, component)?)
    }

    fn add64(&mut self, a: &str, b: &str) -> Result<Value, Error> {
        let address = |name: &str| {
            self.registers
                .get(name)
                .filter(|v| matches!(v.kind, Kind::Address(_)))
                .copied()
        };
        let (base_name, offset) = match (address(a), address(b)) {
            (Some(_), None) => (a, b),
            (None, Some(_)) => (b, a),
            (None, None) => {
                let a = self.integer(a, Kind::B64)?;
                let b = self.integer(b, Kind::B64)?;
                let id = self.builder.i_add(self.u64_type, None, a, b)?;
                return Ok(Value {
                    id,
                    kind: Kind::B64,
                });
            }
            (Some(_), Some(_)) => {
                return Err(Error::new("adding two PTX addresses is unsupported"));
            }
        };
        let offset = self.integer(offset, Kind::B64)?;
        let base = self.read(base_name)?;
        let pointer_type = self.value_type(base.kind);
        // PTX adds byte offsets. Keep the base typed as a pointer for driver residency.
        let id = self
            .builder
            .ptr_access_chain(pointer_type, None, base.id, offset, [])?;
        Ok(Value {
            id,
            kind: base.kind,
        })
    }

    fn shared_base(&mut self, name: &str) -> Result<Word, Error> {
        let (_, array) = self
            .shared
            .filter(|(symbol, _)| *symbol == name)
            .ok_or_else(|| Error::new(format!("unknown PTX shared symbol {name}")))?;
        let zero = self.builder.constant_bit32(self.u32_type, 0);
        Ok(self
            .builder
            .access_chain(self.shared_word_ptr, None, array, [zero])?)
    }

    fn word_pointer(&mut self, space: Space, address: &str) -> Result<Word, Error> {
        if space == Space::Shared && !address.starts_with('%') {
            return self.shared_base(address);
        }
        let pointer = self.read(address)?.require(Kind::Address(space), address)?;
        let ty = match space {
            Space::Global => self.global_word_ptr,
            Space::Shared => self.shared_word_ptr,
        };
        Ok(self.builder.bitcast(ty, None, pointer)?)
    }

    fn float_bits(&mut self, source: &str) -> Result<Word, Error> {
        if let Some(hex) = source.strip_prefix("0f").filter(|hex| hex.len() == 8) {
            let bits = u32::from_str_radix(hex, 16)
                .map_err(|_| Error::new("invalid PTX float bit literal"))?;
            return Ok(self.builder.constant_bit32(self.u32_type, bits));
        }
        self.read(source)?.require(Kind::B32, source)
    }

    fn float_value(&mut self, source: &str) -> Result<Word, Error> {
        let bits = self.float_bits(source)?;
        Ok(self.builder.bitcast(self.f32_type, None, bits)?)
    }

    fn float_op(&mut self, op: FloatOp<'_>) -> Result<Word, Error> {
        let ty = self.f32_type;
        let result = match op {
            FloatOp::Div([a, b]) => {
                let a = self.float_bits(a)?;
                let b = self.float_bits(b)?;
                return rounded::div(&mut self.builder, self.opencl, a, b);
            }
            FloatOp::Sqrt([source]) => {
                let value = self.float_bits(source)?;
                return rounded::sqrt(&mut self.builder, self.opencl, value);
            }
            FloatOp::Add([a, b]) | FloatOp::Mul([a, b]) => {
                let a = self.float_value(a)?;
                let b = self.float_value(b)?;
                match op {
                    FloatOp::Add(_) => self.builder.f_add(ty, None, a, b)?,
                    _ => self.builder.f_mul(ty, None, a, b)?,
                }
            }
            FloatOp::Fma([a, b, c]) => {
                let a = self.float_value(a)?;
                let b = self.float_value(b)?;
                let c = self.float_value(c)?;
                self.fp32_fma = true;
                self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    OpenclStd100Op::fma as u32,
                    [a, b, c].map(Operand::IdRef),
                )?
            }
            FloatOp::FromUnsigned([source]) => {
                let value = self.integer(source, Kind::B32)?;
                let converted = self.builder.convert_u_to_f(ty, None, value)?;
                self.builder.decorate(
                    converted,
                    Decoration::FPRoundingMode,
                    [Operand::FPRoundingMode(FPRoundingMode::RTE)],
                );
                converted
            }
        };
        self.fp32 = true;
        Ok(self.builder.bitcast(self.u32_type, None, result)?)
    }

    fn instruction(&mut self, inst: Instruction<'a>) -> Result<(), Error> {
        let (dst, id, kind) = match inst {
            Instruction::Param(width, dst, src) => {
                let value = self
                    .params
                    .get(src)
                    .copied()
                    .ok_or_else(|| Error::new(format!("unknown PTX parameter {src}")))?;
                if width != value.kind.width() {
                    return Err(Error::new(format!(
                        "mismatched PTX parameter load for {src}"
                    )));
                }
                (dst, value.id, value.kind)
            }
            Instruction::ToGlobal(dst, src) => {
                // The frontend accepts only pointers to device allocations for this state space.
                let kind = Kind::Address(Space::Global);
                let id = self.read(src)?.require(kind, src)?;
                (dst, id, kind)
            }
            Instruction::SharedAddress(dst, src) => {
                let pointer = self.shared_base(src)?;
                let id = self.builder.bitcast(self.shared_byte_ptr, None, pointer)?;
                (dst, id, Kind::Address(Space::Shared))
            }
            Instruction::Builtin(dst, src) => (dst, self.builtin(src)?, Kind::B32),
            Instruction::Mad32(dst, [a, b, c]) => {
                let a = self.integer(a, Kind::B32)?;
                let b = self.integer(b, Kind::B32)?;
                let c = self.integer(c, Kind::B32)?;
                let product = self.builder.i_mul(self.u32_type, None, a, b)?;
                (
                    dst,
                    self.builder.i_add(self.u32_type, None, product, c)?,
                    Kind::B32,
                )
            }
            Instruction::Compare(op, dst, [a, b]) => {
                let a = self.integer(a, Kind::B32)?;
                let b = self.integer(b, Kind::B32)?;
                let ty = self.bool_type;
                let id = match op {
                    Comparison::Equal => self.builder.i_equal(ty, None, a, b),
                    Comparison::Less => self.builder.u_less_than(ty, None, a, b),
                    Comparison::LessEqual => self.builder.u_less_than_equal(ty, None, a, b),
                    Comparison::Greater => self.builder.u_greater_than(ty, None, a, b),
                    Comparison::GreaterEqual => self.builder.u_greater_than_equal(ty, None, a, b),
                }?;
                (dst, id, Kind::Predicate)
            }
            Instruction::PredicateLogic {
                and,
                dst,
                src: [a, b],
            } => {
                let a = self.read(a)?.require(Kind::Predicate, a)?;
                let b = self.read(b)?.require(Kind::Predicate, b)?;
                let ty = self.bool_type;
                let id = if and {
                    self.builder.logical_and(ty, None, a, b)?
                } else {
                    self.builder.logical_or(ty, None, a, b)?
                };
                (dst, id, Kind::Predicate)
            }
            Instruction::Integer32(op, dst, [a, b]) => {
                if matches!(op, IntegerOp::ShiftRight) && !b.parse::<u32>().is_ok_and(|n| n < 32) {
                    return Err(Error::new(
                        "only immediate PTX right shifts below 32 are supported",
                    ));
                }
                let a = self.integer(a, Kind::B32)?;
                let b = self.integer(b, Kind::B32)?;
                let ty = self.u32_type;
                let id = match op {
                    IntegerOp::Add => self.builder.i_add(ty, None, a, b),
                    IntegerOp::Sub => self.builder.i_sub(ty, None, a, b),
                    IntegerOp::Mul => self.builder.i_mul(ty, None, a, b),
                    IntegerOp::ShiftRight => self.builder.shift_right_logical(ty, None, a, b),
                }?;
                (dst, id, Kind::B32)
            }
            Instruction::Move32(dst, src) => {
                let id = if let Some(hex) = src.strip_prefix("0f").filter(|s| s.len() == 8) {
                    let bits = u32::from_str_radix(hex, 16)
                        .map_err(|_| Error::new("invalid PTX bit literal"))?;
                    self.builder.constant_bit32(self.u32_type, bits)
                } else {
                    self.integer(src, Kind::B32)?
                };
                (dst, id, Kind::B32)
            }
            Instruction::MulWide32(dst, [a, b]) => {
                let a = self.integer(a, Kind::B32)?;
                let b = self.integer(b, Kind::B32)?;
                let a = self.builder.u_convert(self.u64_type, None, a)?;
                let b = self.builder.u_convert(self.u64_type, None, b)?;
                (
                    dst,
                    self.builder.i_mul(self.u64_type, None, a, b)?,
                    Kind::B64,
                )
            }
            Instruction::Add64(dst, [a, b]) => {
                let value = self.add64(a, b)?;
                (dst, value.id, value.kind)
            }
            Instruction::ShiftLeft64(dst, [value, shift]) => {
                if !shift.parse::<u32>().is_ok_and(|n| n < 64) {
                    return Err(Error::new(
                        "only immediate PTX left shifts below 64 are supported",
                    ));
                }
                let value = self.integer(value, Kind::B64)?;
                let shift = self.integer(shift, Kind::B32)?;
                let id = self
                    .builder
                    .shift_left_logical(self.u64_type, None, value, shift)?;
                (dst, id, Kind::B64)
            }
            Instruction::Load32(space, dst, address) => {
                let pointer = self.word_pointer(space, address)?;
                (
                    dst,
                    self.builder.load(self.u32_type, None, pointer, None, [])?,
                    Kind::B32,
                )
            }
            Instruction::Float32(dst, op) => (dst, self.float_op(op)?, Kind::B32),
            Instruction::Store32(space, address, src) => {
                let pointer = self.word_pointer(space, address)?;
                let value = self.read(src)?.require(Kind::B32, src)?;
                self.builder.store(pointer, value, None, [])?;
                return Ok(());
            }
            Instruction::Barrier => {
                let scope = self
                    .builder
                    .constant_bit32(self.u32_type, Scope::Workgroup as u32);
                let semantics = MemorySemantics::ACQUIRE_RELEASE
                    | MemorySemantics::WORKGROUP_MEMORY
                    | MemorySemantics::CROSS_WORKGROUP_MEMORY;
                let semantics = self.builder.constant_bit32(self.u32_type, semantics.bits());
                self.builder.control_barrier(scope, scope, semantics)?;
                return Ok(());
            }
            Instruction::AtomicAdd32(dst, address, value) => {
                let pointer = self.word_pointer(Space::Global, address)?;
                let value = self.integer(value, Kind::B32)?;
                let scope = self
                    .builder
                    .constant_bit32(self.u32_type, Scope::Device as u32);
                let relaxed = self
                    .builder
                    .constant_bit32(self.u32_type, MemorySemantics::empty().bits());
                self.device_atomics = true;
                let previous = self.builder.atomic_i_add(
                    self.u32_type,
                    None,
                    pointer,
                    scope,
                    relaxed,
                    value,
                )?;
                (dst, previous, Kind::B32)
            }
        };
        self.write(dst, Value { id, kind })
    }
}

pub(super) fn emit(entry: Entry<'_>) -> Result<Translation, Error> {
    let flow = Flow::analyze(&entry.blocks)?;
    let mut emitter = Emitter::new(entry.registers, entry.shared);
    let types: Vec<_> = entry
        .params
        .iter()
        .map(|&(_, param)| emitter.param_type(param))
        .collect();
    let void = emitter.builder.type_void();
    let signature = emitter.builder.type_function(void, types.iter().copied());
    let function = emitter
        .builder
        .begin_function(void, None, FunctionControl::NONE, signature)?;
    emitter.builder.entry_point(
        ExecutionModel::Kernel,
        function,
        entry.name,
        emitter.builtins,
    );
    let values = types
        .into_iter()
        .map(|ty| emitter.builder.function_parameter(ty))
        .collect::<Result<Vec<_>, _>>()?;
    emitter.builder.begin_block(None)?;
    for ((name, param), mut id) in entry.params.iter().zip(values) {
        let kind = match param {
            Param::U32 => Kind::B32,
            Param::F32 => {
                id = emitter.builder.bitcast(emitter.u32_type, None, id)?;
                Kind::B32
            }
            Param::U64 => Kind::B64,
            Param::GlobalPtr => Kind::Address(Space::Global),
        };
        emitter.params.insert(name, Value { id, kind });
    }
    let labels: Vec<_> = entry.blocks.iter().map(|_| emitter.builder.id()).collect();
    emitter.builder.branch(labels[0])?;
    for index in flow.order {
        emitter.initialized.clone_from(&flow.inputs[index]);
        emitter.builder.begin_block(Some(labels[index]))?;
        let block = &entry.blocks[index];
        for &inst in &block.instructions {
            emitter.instruction(inst)?;
        }
        match block.end {
            Terminator::Return => emitter.builder.ret()?,
            Terminator::Jump(target) => emitter.builder.branch(labels[target])?,
            Terminator::Branch {
                predicate,
                taken,
                other,
            } => {
                let condition = emitter
                    .read(predicate)?
                    .require(Kind::Predicate, predicate)?;
                emitter
                    .builder
                    .branch_conditional(condition, labels[taken], labels[other], [])?;
            }
        }
    }
    emitter.builder.end_function()?;
    if emitter.fp32 {
        emitter
            .builder
            .execution_mode(function, ExecutionMode::ContractionOff, []);
    }
    let mut module = emitter.builder.module();
    module.functions[0].blocks[0]
        .instructions
        .splice(0..0, emitter.locals);
    Ok(Translation {
        name: entry.name.to_owned(),
        params: entry.params.into_iter().map(|(_, param)| param).collect(),
        words: module.assemble(),
        shared_bytes: entry.shared.map_or(0, |(_, bytes)| bytes),
        max_threads: entry.max_threads,
        device_atomics: emitter.device_atomics,
        fp32: emitter.fp32,
        fp32_fma: emitter.fp32_fma,
    })
}
