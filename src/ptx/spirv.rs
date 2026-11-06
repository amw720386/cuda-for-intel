use super::flow::Flow;
use super::rounded;
use super::{
    Comparison, Entry, Error, FloatOp, Instruction, IntegerOp, KernelMetadata, Param, Space,
    Terminator, Translation, Width,
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
    u16_type: Word,
    u32_type: Word,
    u64_type: Word,
    f16_type: Word,
    f32_type: Word,
    global_byte_ptr: Word,
    global_half_ptr: Word,
    global_word_ptr: Word,
    shared_byte_ptr: Word,
    shared_word_ptr: Word,
    shared: Option<(&'a str, Word)>,
    /// Workgroup scratch used to emulate CUDA warp shuffles across Intel subgroup widths.
    shuffle: Option<Word>,
    shuffle_words: u32,
    builtin_type: Word,
    builtins: [Word; BUILTINS.len()],
    declarations: HashMap<String, Width>,
    registers: HashMap<&'a str, Value>,
    initialized: HashSet<&'a str>,
    locals: Vec<SpirvInstruction>,
    params: HashMap<(&'a str, usize), Value>,
    device_atomics: bool,
    fp32: bool,
    fp32_fma: bool,
    opencl: Word,
}

impl<'a> Emitter<'a> {
    fn new() -> Self {
        let mut builder = Builder::new();
        builder.set_version(1, 0);
        for capability in [
            Capability::Addresses,
            Capability::Kernel,
            Capability::Int8,
            Capability::Int16,
            Capability::Int64,
            Capability::Float16,
        ] {
            builder.capability(capability);
        }
        builder.memory_model(AddressingModel::Physical64, MemoryModel::OpenCL);
        let opencl = builder.ext_inst_import("OpenCL.std");

        let bool_type = builder.type_bool();
        let u8_type = builder.type_int(8, 0);
        let u16_type = builder.type_int(16, 0);
        let u32_type = builder.type_int(32, 0);
        let u64_type = builder.type_int(64, 0);
        let f16_type = builder.type_float(16, None);
        let f32_type = builder.type_float(32, None);
        let global_byte_ptr = builder.type_pointer(None, StorageClass::CrossWorkgroup, u8_type);
        let global_half_ptr = builder.type_pointer(None, StorageClass::CrossWorkgroup, u16_type);
        let global_word_ptr = builder.type_pointer(None, StorageClass::CrossWorkgroup, u32_type);
        let shared_byte_ptr = builder.type_pointer(None, StorageClass::Workgroup, u8_type);
        let shared_word_ptr = builder.type_pointer(None, StorageClass::Workgroup, u32_type);
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
            u16_type,
            u32_type,
            u64_type,
            f16_type,
            f32_type,
            global_byte_ptr,
            global_half_ptr,
            global_word_ptr,
            shared_byte_ptr,
            shared_word_ptr,
            shared: None,
            shuffle: None,
            shuffle_words: 0,
            builtin_type,
            builtins,
            declarations: HashMap::new(),
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
            Param::U32 | Param::Words(_) => self.u32_type,
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
        // A register slot must have been written on some path. CUDA PTX often
        // guards both the defining load and later uses with the same predicate,
        // so the join into the use can look uninitialized under path-insensitive
        // analysis even though the use is unreachable when the value is absent.
        let slot = self.registers.get(operand).copied().ok_or_else(|| {
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

    fn parse_u32(operand: &str) -> Option<u32> {
        if let Some(hex) = operand
            .strip_prefix("0x")
            .or_else(|| operand.strip_prefix("0X"))
        {
            u32::from_str_radix(hex, 16).ok()
        } else if let Ok(n) = operand.parse::<u32>() {
            Some(n)
        } else {
            // PTX often writes all-ones as `-1`.
            operand.parse::<i32>().ok().map(|n| n as u32)
        }
    }

    fn parse_u64(operand: &str) -> Option<u64> {
        if let Some(hex) = operand
            .strip_prefix("0x")
            .or_else(|| operand.strip_prefix("0X"))
        {
            u64::from_str_radix(hex, 16).ok()
        } else if let Ok(n) = operand.parse::<u64>() {
            Some(n)
        } else {
            operand.parse::<i64>().ok().map(|n| n as u64)
        }
    }

    fn integer(&mut self, operand: &str, kind: Kind) -> Result<Word, Error> {
        if operand.starts_with('%') {
            return self.read(operand)?.require(kind, operand);
        }
        match kind {
            Kind::B32 => Self::parse_u32(operand)
                .map(|n| self.builder.constant_bit32(self.u32_type, n)),
            Kind::B64 => Self::parse_u64(operand)
                .map(|n| self.builder.constant_bit64(self.u64_type, n)),
            _ => None,
        }
        .ok_or_else(|| Error::new(format!("unsupported integer operand {operand}")))
    }

    fn shuffle_slot(&mut self, index: Word) -> Result<Word, Error> {
        let array = self
            .shuffle
            .ok_or_else(|| Error::new("missing warp shuffle scratch buffer"))?;
        Ok(self
            .builder
            .access_chain(self.shared_word_ptr, None, array, [index])?)
    }

    fn emit_shuffle_bfly(
        &mut self,
        dst: &'a str,
        pred: Option<&'a str>,
        value: &str,
        xor_mask: &str,
        clamp: &str,
        membermask: &str,
    ) -> Result<(), Error> {
        // Full-warp CUDA reductions use clamp=31 / membermask=all-ones. Accept those
        // as immediates or as registers that nvcc filled with the same constants.
        if let Some(clamp) = Self::parse_u32(clamp)
            && clamp != 0x1f
        {
            return Err(Error::new(
                "only full-warp shfl.sync.bfly.b32 clamp 0x1f is supported",
            ));
        }
        if let Some(membermask) = Self::parse_u32(membermask)
            && membermask != u32::MAX
        {
            return Err(Error::new(
                "only full-warp shfl.sync.bfly.b32 membermask 0xffffffff is supported",
            ));
        }
        // Keep register clamp/membermask live for ABI fidelity even though the
        // shared-memory emulation currently assumes a full 32-lane warp.
        let _clamp = self.integer(clamp, Kind::B32)?;
        let _membermask = self.integer(membermask, Kind::B32)?;
        let value = self.read(value)?.require(Kind::B32, value)?;
        let xor_mask = self.integer(xor_mask, Kind::B32)?;
        let tid = self.builtin("%tid.x")?;
        let lane_mask = self.builder.constant_bit32(self.u32_type, 31);
        let warp_mask = self.builder.constant_bit32(self.u32_type, !31u32);
        let slot = self.shuffle_slot(tid)?;
        self.builder.store(slot, value, None, [])?;
        let scope = self
            .builder
            .constant_bit32(self.u32_type, Scope::Workgroup as u32);
        let semantics = MemorySemantics::ACQUIRE_RELEASE
            | MemorySemantics::WORKGROUP_MEMORY
            | MemorySemantics::CROSS_WORKGROUP_MEMORY;
        let semantics = self.builder.constant_bit32(self.u32_type, semantics.bits());
        self.builder.control_barrier(scope, scope, semantics)?;
        let lane = self.builder.bitwise_and(self.u32_type, None, tid, lane_mask)?;
        let warp = self.builder.bitwise_and(self.u32_type, None, tid, warp_mask)?;
        let flipped = self.builder.bitwise_xor(self.u32_type, None, lane, xor_mask)?;
        let source_lane = self
            .builder
            .bitwise_and(self.u32_type, None, flipped, lane_mask)?;
        let source = self.builder.bitwise_or(self.u32_type, None, warp, source_lane)?;
        let source_slot = self.shuffle_slot(source)?;
        let result = self
            .builder
            .load(self.u32_type, None, source_slot, None, [])?;
        self.write(
            dst,
            Value {
                id: result,
                kind: Kind::B32,
            },
        )?;
        if let Some(pred) = pred {
            // Full-warp clamp keeps every participating lane in range.
            let truth = self.builder.constant_true(self.bool_type);
            self.write(
                pred,
                Value {
                    id: truth,
                    kind: Kind::Predicate,
                },
            )?;
        }
        Ok(())
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

    fn addressed_byte_pointer(&mut self, space: Space, address: &str) -> Result<Word, Error> {
        let (base, offset) = match address.split_once('+') {
            Some((base, offset)) => (
                base,
                Self::parse_u64(offset).ok_or_else(|| {
                    Error::new(format!("unsupported PTX address offset {offset}"))
                })?,
            ),
            None => (address, 0),
        };
        let pointer = self.read(base)?.require(Kind::Address(space), base)?;
        let byte_ty = match space {
            Space::Global => self.global_byte_ptr,
            Space::Shared => self.shared_byte_ptr,
        };
        let bytes_ptr = self.builder.bitcast(byte_ty, None, pointer)?;
        if offset == 0 {
            Ok(bytes_ptr)
        } else {
            let bytes = self.builder.constant_bit64(self.u64_type, offset);
            Ok(self
                .builder
                .ptr_access_chain(byte_ty, None, bytes_ptr, bytes, [])?)
        }
    }

    fn word_pointer(&mut self, space: Space, address: &str) -> Result<Word, Error> {
        if space == Space::Shared && !address.starts_with('%') {
            return self.shared_base(address);
        }
        let bytes_ptr = self.addressed_byte_pointer(space, address)?;
        let ty = match space {
            Space::Global => self.global_word_ptr,
            Space::Shared => self.shared_word_ptr,
        };
        Ok(self.builder.bitcast(ty, None, bytes_ptr)?)
    }

    fn half_pointer(&mut self, space: Space, address: &str) -> Result<Word, Error> {
        if space != Space::Global {
            return Err(Error::new("only global PTX u16 loads are supported"));
        }
        let bytes_ptr = self.addressed_byte_pointer(space, address)?;
        Ok(self.builder.bitcast(self.global_half_ptr, None, bytes_ptr)?)
    }

    fn float_vector_pointer(
        &mut self,
        space: Space,
        address: &str,
        count: u32,
    ) -> Result<(Word, Word), Error> {
        let vector = self.builder.type_vector(self.f32_type, count);
        let storage = match space {
            Space::Global => StorageClass::CrossWorkgroup,
            Space::Shared => StorageClass::Workgroup,
        };
        let pointer_type = self.builder.type_pointer(None, storage, vector);
        let byte_pointer = if space == Space::Shared && !address.starts_with('%') {
            let array = self.shared_base(address)?;
            self.builder.bitcast(self.shared_byte_ptr, None, array)?
        } else {
            self.read(address)?.require(Kind::Address(space), address)?
        };
        Ok((
            vector,
            self.builder.bitcast(pointer_type, None, byte_pointer)?,
        ))
    }

    fn load_float_vector(
        &mut self,
        space: Space,
        count: u8,
        dst: &[&'a str; 4],
        addr: &str,
    ) -> Result<(), Error> {
        let count = u32::from(count);
        let (vector, pointer) = self.float_vector_pointer(space, addr, count)?;
        let loaded = self.builder.load(vector, None, pointer, None, [])?;
        self.fp32 = true;
        for index in 0..count {
            let component = self
                .builder
                .composite_extract(self.f32_type, None, loaded, [index])?;
            let bits = self.builder.bitcast(self.u32_type, None, component)?;
            self.write(
                dst[index as usize],
                Value {
                    id: bits,
                    kind: Kind::B32,
                },
            )?;
        }
        Ok(())
    }

    fn store_float_vector(
        &mut self,
        space: Space,
        count: u8,
        addr: &str,
        src: &[&'a str; 4],
    ) -> Result<(), Error> {
        let count = u32::from(count);
        let (vector, pointer) = self.float_vector_pointer(space, addr, count)?;
        let mut components = [0; 4];
        for index in 0..count {
            components[index as usize] = self.float_value(src[index as usize])?;
        }
        let value = self.builder.composite_construct(
            vector,
            None,
            components[..count as usize].iter().copied(),
        )?;
        self.fp32 = true;
        self.builder.store(pointer, value, None, [])?;
        Ok(())
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
            FloatOp::Add([a, b]) | FloatOp::Sub([a, b]) | FloatOp::Mul([a, b]) => {
                let a = self.float_value(a)?;
                let b = self.float_value(b)?;
                match op {
                    FloatOp::Add(_) => self.builder.f_add(ty, None, a, b)?,
                    FloatOp::Sub(_) => self.builder.f_sub(ty, None, a, b)?,
                    _ => self.builder.f_mul(ty, None, a, b)?,
                }
            }
            FloatOp::MinMax { min, src: [a, b] } => {
                let a = self.float_bits(a)?;
                let b = self.float_bits(b)?;
                let magnitude_mask = self.builder.constant_bit32(self.u32_type, 0x7fff_ffff);
                let combined = self.builder.bitwise_or(self.u32_type, None, a, b)?;
                let magnitude =
                    self.builder
                        .bitwise_and(self.u32_type, None, combined, magnitude_mask)?;
                let zero = self.builder.constant_bit32(self.u32_type, 0);
                let both_zero = self
                    .builder
                    .i_equal(self.bool_type, None, magnitude, zero)?;
                // PTX orders -0 below +0. OpenCL fmin/fmax do not guarantee the zero sign.
                let signed_zero = if min {
                    combined
                } else {
                    self.builder.bitwise_and(self.u32_type, None, a, b)?
                };
                let a = self.builder.bitcast(ty, None, a)?;
                let b = self.builder.bitcast(ty, None, b)?;
                let op = if min {
                    OpenclStd100Op::fmin
                } else {
                    OpenclStd100Op::fmax
                };
                let numeric = self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    op as u32,
                    [a, b].map(Operand::IdRef),
                )?;
                let signed_zero = self.builder.bitcast(ty, None, signed_zero)?;
                self.builder
                    .select(ty, None, both_zero, signed_zero, numeric)?
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
            FloatOp::FromSigned([source]) => {
                let value = self.integer(source, Kind::B32)?;
                let converted = self.builder.convert_s_to_f(ty, None, value)?;
                self.builder.decorate(
                    converted,
                    Decoration::FPRoundingMode,
                    [Operand::FPRoundingMode(FPRoundingMode::RTE)],
                );
                converted
            }
            FloatOp::FromHalf([source]) => {
                // PTX cvt.f32.f16 reads the low 16 bits of the source as IEEE half.
                let bits32 = self.integer(source, Kind::B32)?;
                let bits16 = self.builder.u_convert(self.u16_type, None, bits32)?;
                let half = self.builder.bitcast(self.f16_type, None, bits16)?;
                self.builder.f_convert(ty, None, half)?
            }
            FloatOp::RsqrtApprox([source]) => {
                let value = self.float_value(source)?;
                self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    OpenclStd100Op::rsqrt as u32,
                    [Operand::IdRef(value)],
                )?
            }
            FloatOp::SinApprox([source])
            | FloatOp::CosApprox([source])
            | FloatOp::Lg2Approx([source])
            | FloatOp::Ex2Approx([source]) => {
                let value = self.float_value(source)?;
                let opcode = match op {
                    FloatOp::SinApprox(_) => OpenclStd100Op::native_sin,
                    FloatOp::CosApprox(_) => OpenclStd100Op::native_cos,
                    FloatOp::Lg2Approx(_) => OpenclStd100Op::native_log2,
                    _ => OpenclStd100Op::native_exp2,
                };
                self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    opcode as u32,
                    [Operand::IdRef(value)],
                )?
            }
        };
        self.fp32 = true;
        Ok(self.builder.bitcast(self.u32_type, None, result)?)
    }

    fn instruction(&mut self, inst: Instruction<'a>) -> Result<(), Error> {
        let (dst, id, kind) = match inst {
            Instruction::Param(width, dst, src, offset) => {
                let value = self.params.get(&(src, offset)).copied().ok_or_else(|| {
                    Error::new(format!("unknown PTX parameter word {src}+{offset}"))
                })?;
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
            Instruction::Compare {
                op,
                signed,
                dst,
                src: [a, b],
            } => {
                let a = self.integer(a, Kind::B32)?;
                let b = self.integer(b, Kind::B32)?;
                let ty = self.bool_type;
                let id = match (op, signed) {
                    (Comparison::Equal, _) => self.builder.i_equal(ty, None, a, b),
                    (Comparison::NotEqual, _) => self.builder.i_not_equal(ty, None, a, b),
                    (Comparison::Less, false) => self.builder.u_less_than(ty, None, a, b),
                    (Comparison::LessEqual, false) => {
                        self.builder.u_less_than_equal(ty, None, a, b)
                    }
                    (Comparison::Greater, false) => self.builder.u_greater_than(ty, None, a, b),
                    (Comparison::GreaterEqual, false) => {
                        self.builder.u_greater_than_equal(ty, None, a, b)
                    }
                    (Comparison::Less, true) => self.builder.s_less_than(ty, None, a, b),
                    (Comparison::LessEqual, true) => self.builder.s_less_than_equal(ty, None, a, b),
                    (Comparison::Greater, true) => self.builder.s_greater_than(ty, None, a, b),
                    (Comparison::GreaterEqual, true) => {
                        self.builder.s_greater_than_equal(ty, None, a, b)
                    }
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
                let a = self.integer(a, Kind::B32)?;
                let b = self.integer(b, Kind::B32)?;
                let ty = self.u32_type;
                let id = match op {
                    IntegerOp::Add => self.builder.i_add(ty, None, a, b),
                    IntegerOp::Sub => self.builder.i_sub(ty, None, a, b),
                    IntegerOp::Mul => self.builder.i_mul(ty, None, a, b),
                    IntegerOp::And => self.builder.bitwise_and(ty, None, a, b),
                    IntegerOp::Or => self.builder.bitwise_or(ty, None, a, b),
                    IntegerOp::Div => self.builder.s_div(ty, None, a, b),
                    IntegerOp::Rem => self.builder.s_rem(ty, None, a, b),
                    IntegerOp::SignExtend8 => {
                        // PTX cvt.s32.s8 sign-extends the low 8 bits of the source.
                        let shift = self.builder.constant_bit32(ty, 24);
                        let raised = self.builder.shift_left_logical(ty, None, a, shift)?;
                        self.builder.shift_right_arithmetic(ty, None, raised, shift)
                    }
                    IntegerOp::ShiftLeft
                    | IntegerOp::ShiftRightLogical
                    | IntegerOp::ShiftRightArithmetic => {
                        // PTX clamps at 32, but SPIR-V shifts at or above 32 are undefined.
                        let bits = self.builder.constant_bit32(ty, 32);
                        let zero = self.builder.constant_bit32(ty, 0);
                        let in_range = self.builder.u_less_than(self.bool_type, None, b, bits)?;
                        let safe = self.builder.select(ty, None, in_range, b, zero)?;
                        let shifted = match op {
                            IntegerOp::ShiftLeft => {
                                self.builder.shift_left_logical(ty, None, a, safe)?
                            }
                            IntegerOp::ShiftRightArithmetic => {
                                self.builder.shift_right_arithmetic(ty, None, a, safe)?
                            }
                            _ => self.builder.shift_right_logical(ty, None, a, safe)?,
                        };
                        self.builder.select(ty, None, in_range, shifted, zero)
                    }
                    IntegerOp::MulHighUnsigned => self.builder.ext_inst(
                        ty,
                        None,
                        self.opencl,
                        OpenclStd100Op::u_mul_hi as u32,
                        [a, b].map(Operand::IdRef),
                    ),
                }?;
                (dst, id, Kind::B32)
            }
            Instruction::Not32(dst, src) => {
                let src = self.integer(src, Kind::B32)?;
                (
                    dst,
                    self.builder.not(self.u32_type, None, src)?,
                    Kind::B32,
                )
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
            Instruction::MulWide32 {
                signed,
                dst,
                src: [a, b],
            } => {
                let a = self.integer(a, Kind::B32)?;
                let b = self.integer(b, Kind::B32)?;
                let a = if signed {
                    self.builder.s_convert(self.u64_type, None, a)?
                } else {
                    self.builder.u_convert(self.u64_type, None, a)?
                };
                let b = if signed {
                    self.builder.s_convert(self.u64_type, None, b)?
                } else {
                    self.builder.u_convert(self.u64_type, None, b)?
                };
                (
                    dst,
                    self.builder.i_mul(self.u64_type, None, a, b)?,
                    Kind::B64,
                )
            }
            Instruction::Mul64(dst, [a, b]) => {
                let a = self.integer(a, Kind::B64)?;
                let b = self.integer(b, Kind::B64)?;
                (
                    dst,
                    self.builder.i_mul(self.u64_type, None, a, b)?,
                    Kind::B64,
                )
            }
            Instruction::Widen { signed, dst, src } => {
                let src = self.integer(src, Kind::B32)?;
                let id = if signed {
                    self.builder.s_convert(self.u64_type, None, src)?
                } else {
                    self.builder.u_convert(self.u64_type, None, src)?
                };
                (dst, id, Kind::B64)
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
            Instruction::Load16(space, dst, address) => {
                let pointer = self.half_pointer(space, address)?;
                let bits16 = self
                    .builder
                    .load(self.u16_type, None, pointer, None, [])?;
                let bits32 = self.builder.u_convert(self.u32_type, None, bits16)?;
                (dst, bits32, Kind::B32)
            }
            Instruction::LoadVec {
                space,
                count,
                dst,
                addr,
            } => {
                self.load_float_vector(space, count, &dst, addr)?;
                return Ok(());
            }
            Instruction::Float32(dst, op) => (dst, self.float_op(op)?, Kind::B32),
            Instruction::Store32(space, address, src) => {
                let pointer = self.word_pointer(space, address)?;
                let value = self.read(src)?.require(Kind::B32, src)?;
                self.builder.store(pointer, value, None, [])?;
                return Ok(());
            }
            Instruction::StoreVec {
                space,
                count,
                addr,
                src,
            } => {
                self.store_float_vector(space, count, addr, &src)?;
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
            Instruction::ShuffleBfly {
                dst,
                pred,
                src: [value, xor_mask, clamp, membermask],
            } => {
                self.emit_shuffle_bfly(dst, pred, value, xor_mask, clamp, membermask)?;
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

pub(super) fn emit(entries: Vec<Entry<'_>>) -> Result<Translation, Error> {
    let mut emitter = Emitter::new();
    let kernels = entries
        .into_iter()
        .map(|entry| emitter.emit_entry(entry))
        .collect::<Result<_, _>>()?;
    Ok(Translation {
        words: emitter.builder.module().assemble(),
        kernels,
    })
}

impl<'a> Emitter<'a> {
    fn emit_entry(&mut self, entry: Entry<'a>) -> Result<KernelMetadata, Error> {
        let flow = Flow::analyze(&entry.blocks)?;
        self.declarations = entry.registers;
        self.registers.clear();
        self.initialized.clear();
        self.params.clear();
        self.device_atomics = false;
        self.fp32 = false;
        self.fp32_fma = false;
        let needs_shuffle = entry.blocks.iter().any(|block| {
            block
                .instructions
                .iter()
                .any(|inst| matches!(inst, Instruction::ShuffleBfly { .. }))
        });
        self.shuffle_words = if needs_shuffle {
            entry.max_threads.unwrap_or(1024).max(32)
        } else {
            0
        };
        self.shuffle = (self.shuffle_words > 0).then(|| {
            let length = self
                .builder
                .constant_bit32(self.u32_type, self.shuffle_words);
            let array = self.builder.type_array(self.u32_type, length);
            let pointer = self
                .builder
                .type_pointer(None, StorageClass::Workgroup, array);
            self.builder
                .variable(pointer, None, StorageClass::Workgroup, None)
        });
        self.shared = entry.shared.map(|(name, bytes)| {
            // Use a u32 array to retain the PTX declaration's four-byte alignment.
            let length = self.builder.constant_bit32(self.u32_type, bytes / 4);
            let array = self.builder.type_array(self.u32_type, length);
            let pointer = self
                .builder
                .type_pointer(None, StorageClass::Workgroup, array);
            let variable = self
                .builder
                .variable(pointer, None, StorageClass::Workgroup, None);
            (name, variable)
        });
        let types: Vec<_> = entry
            .params
            .iter()
            .flat_map(|&(_, param)| {
                let count = match param {
                    Param::Words(words) => usize::from(words),
                    _ => 1,
                };
                std::iter::repeat_n(self.param_type(param), count)
            })
            .collect();
        let void = self.builder.type_void();
        let signature = self.builder.type_function(void, types.iter().copied());
        let function = self
            .builder
            .begin_function(void, None, FunctionControl::NONE, signature)?;
        self.builder
            .entry_point(ExecutionModel::Kernel, function, entry.name, self.builtins);
        let values = types
            .into_iter()
            .map(|ty| self.builder.function_parameter(ty))
            .collect::<Result<Vec<_>, _>>()?;
        self.builder.begin_block(None)?;
        let mut values = values.into_iter();
        for &(name, param) in &entry.params {
            // Keep one CUDA argument, but pass each aligned word to Level Zero
            // as a scalar so the device ABI needs no by-value struct pointer.
            if let Param::Words(words) = param {
                for offset in 0..usize::from(words) {
                    let id = values.next().unwrap();
                    self.params.insert(
                        (name, offset * 4),
                        Value {
                            id,
                            kind: Kind::B32,
                        },
                    );
                }
                continue;
            }
            let mut id = values.next().unwrap();
            let kind = match param {
                Param::U32 => Kind::B32,
                Param::F32 => {
                    id = self.builder.bitcast(self.u32_type, None, id)?;
                    Kind::B32
                }
                Param::U64 => Kind::B64,
                Param::GlobalPtr => Kind::Address(Space::Global),
                Param::Words(_) => unreachable!(),
            };
            self.params.insert((name, 0), Value { id, kind });
        }
        let labels: Vec<_> = entry.blocks.iter().map(|_| self.builder.id()).collect();
        self.builder.branch(labels[0])?;
        for index in flow.order {
            self.initialized.clone_from(&flow.inputs[index]);
            self.builder.begin_block(Some(labels[index]))?;
            let block = &entry.blocks[index];
            for &inst in &block.instructions {
                self.instruction(inst)?;
            }
            match block.end {
                Terminator::Return => self.builder.ret()?,
                Terminator::Jump(target) => self.builder.branch(labels[target])?,
                Terminator::Branch {
                    predicate,
                    taken,
                    other,
                } => {
                    let condition = self.read(predicate)?.require(Kind::Predicate, predicate)?;
                    self.builder
                        .branch_conditional(condition, labels[taken], labels[other], [])?;
                }
            }
        }
        self.builder.end_function()?;
        if self.fp32 {
            self.builder
                .execution_mode(function, ExecutionMode::ContractionOff, []);
        }
        self.builder
            .module_mut()
            .functions
            .last_mut()
            .unwrap()
            .blocks[0]
            .instructions
            .splice(0..0, self.locals.drain(..));
        Ok(KernelMetadata {
            name: entry.name.to_owned(),
            params: entry.params.into_iter().map(|(_, param)| param).collect(),
            shared_bytes: entry.shared.map_or(0, |(_, bytes)| bytes)
                + self.shuffle_words.saturating_mul(4),
            max_threads: entry.max_threads,
            device_atomics: self.device_atomics,
            fp32: self.fp32,
            fp32_fma: self.fp32_fma,
        })
    }
}
