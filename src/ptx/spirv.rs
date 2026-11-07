use super::flow::Flow;
use super::globals::GlobalVar;
use super::rounded;
use super::{
    Alu3Op, Alu64Op, AtomicOp, Comparison, MmaShape, ShuffleMode, VoteMode, Entry, Error, FloatOp, Instruction, IntegerOp, KernelMetadata, Param, Space,
    Terminator, Translation, Width,
};
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Instruction as SpirvInstruction, Operand};
use rspirv::spirv::{
    LinkageType, AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FPRoundingMode, FunctionControl, MemoryModel, MemorySemantics, Op,
    OpenclStd100Op, Scope, StorageClass, Word,
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

#[path = "ext_emit.rs"]
mod ext_emit;
#[path = "ext_float.rs"]
mod ext_float;

struct Emitter<'a> {
    builder: Builder,
    bool_type: Word,
    u8_type: Word,
    u16_type: Word,
    u32_type: Word,
    u64_type: Word,
    i64_type: Word,
    f16_type: Word,
    f32_type: Word,
    global_byte_ptr: Word,
    global_half_ptr: Word,
    global_word_ptr: Word,
    global_qword_ptr: Word,
    shared_byte_ptr: Word,
    shared_word_ptr: Word,
    local_byte_ptr: Word,
    local_word_ptr: Word,
    /// Unified workgroup array backing all `.shared` symbols.
    shared_array: Option<Word>,
    /// Byte offset of each `.shared` symbol within `shared_array`.
    shared_offsets: HashMap<&'a str, u32>,
    /// Module-scope `.global`/`.const` variables by PTX name.
    globals: HashMap<String, Word>,
    local: Option<(&'a str, Word)>,
    /// Set when the entry uses CUDA warp shuffles / redux (forces SubgroupSize 32).
    warp_ops: bool,
    /// Capabilities / extensions already declared on the module.
    declared: HashSet<&'static str>,
    builtin_type: Word,
    builtins: [Word; BUILTINS.len()],
    declarations: HashMap<String, Width>,
    registers: HashMap<&'a str, Value>,
    initialized: HashSet<&'a str>,
    /// Registers assigned somewhere in the entry; others read as zero (undefined in PTX).
    ever_defined: HashSet<&'a str>,
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
        let i64_type = builder.type_int(64, 1);
        let f16_type = builder.type_float(16, None);
        let f32_type = builder.type_float(32, None);
        let global_byte_ptr = builder.type_pointer(None, StorageClass::CrossWorkgroup, u8_type);
        let global_half_ptr = builder.type_pointer(None, StorageClass::CrossWorkgroup, u16_type);
        let global_word_ptr = builder.type_pointer(None, StorageClass::CrossWorkgroup, u32_type);
        let global_qword_ptr = builder.type_pointer(None, StorageClass::CrossWorkgroup, u64_type);
        let shared_byte_ptr = builder.type_pointer(None, StorageClass::Workgroup, u8_type);
        let shared_word_ptr = builder.type_pointer(None, StorageClass::Workgroup, u32_type);
        // CUDA `.local` → SPIR-V Function (per-invocation private).
        let local_byte_ptr = builder.type_pointer(None, StorageClass::Function, u8_type);
        let local_word_ptr = builder.type_pointer(None, StorageClass::Function, u32_type);
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
            u8_type,
            u16_type,
            u32_type,
            u64_type,
            i64_type,
            f16_type,
            f32_type,
            global_byte_ptr,
            global_half_ptr,
            global_word_ptr,
            global_qword_ptr,
            shared_byte_ptr,
            shared_word_ptr,
            local_byte_ptr,
            local_word_ptr,
            shared_array: None,
            shared_offsets: HashMap::new(),
            globals: HashMap::new(),
            local: None,
            warp_ops: false,
            declared: HashSet::new(),
            builtin_type,
            builtins,
            declarations: HashMap::new(),
            registers: HashMap::new(),
            initialized: HashSet::new(),
            ever_defined: HashSet::new(),
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
            Param::U8 => self.u8_type,
            Param::U16 => self.u16_type,
            Param::U32 | Param::Words { .. } => self.u32_type,
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
            Kind::Address(Space::Local) => self.local_byte_ptr,
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
        let mut value = value;
        let slot = if let Some(slot) = self.registers.get(dst).copied() {
            // PTX registers are untyped 64-bit values: nvcc reuses one `%rd` for a raw
            // integer and for a pointer. Coerce between the slot's kind and the new value.
            if slot.kind != value.kind {
                let wide = |kind: Kind| matches!(kind, Kind::B64 | Kind::Address(_));
                if !(wide(slot.kind) && wide(value.kind)) {
                    slot.require(value.kind, dst)?;
                }
                let mut id = value.id;
                if let Kind::Address(_) = value.kind {
                    id = self.builder.convert_ptr_to_u(self.u64_type, None, id)?;
                }
                if let Kind::Address(_) = slot.kind {
                    let pointer_type = self.value_type(slot.kind);
                    id = self.builder.convert_u_to_ptr(pointer_type, None, id)?;
                }
                value = Value { id, kind: slot.kind };
            }
            slot.id
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
        // `0fXXXXXXXX` float literal used where the translator wants raw 32-bit bits (selp.f32, mov.f32).
        if let Some(bits) = operand.strip_prefix("0f").or_else(|| operand.strip_prefix("0F")).filter(|h| h.len() == 8) {
            return u32::from_str_radix(bits, 16).ok();
        }
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
        if operand.starts_with('%')
            && !self.registers.contains_key(operand)
            && !self.ever_defined.contains(operand)
            && !self.params.contains_key(&(operand, 0))
            && matches!(kind, Kind::B32 | Kind::B64)
            && !operand.contains('.')
        {
            // A register that is never assigned holds an undefined value (nvcc emits
            // such operands for dead lanes of packed moves); read it as zero.
            return Ok(if kind == Kind::B32 {
                self.u32_const(0)
            } else {
                self.builder.constant_bit64(self.u64_type, 0)
            });
        }
        if operand.starts_with('%') {
            let value = self.read(operand)?;
            if value.kind == kind {
                return Ok(value.id);
            }
            // Pointer params are often compared/added as .u64 before cvta.to.global.
            if kind == Kind::B64 && matches!(value.kind, Kind::Address(_)) {
                return Ok(self.builder.bitcast(self.u64_type, None, value.id)?);
            }
            return Ok(value.require(kind, operand)?);
        }
        match kind {
            Kind::B32 if operand.len() == 10 && (operand.starts_with("0f") || operand.starts_with("0F")) => {
                u32::from_str_radix(&operand[2..], 16)
                    .ok()
                    .map(|n| self.builder.constant_bit32(self.u32_type, n))
            }
            Kind::B32 => Self::parse_u32(operand)
                .map(|n| self.builder.constant_bit32(self.u32_type, n)),
            Kind::B64 => Self::parse_u64(operand)
                .map(|n| self.builder.constant_bit64(self.u64_type, n)),
            _ => None,
        }
        .ok_or_else(|| Error::new(format!("unsupported integer operand {operand}")))
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
        // Keep register clamp/membermask live for ABI fidelity.
        let _clamp = self.integer(clamp, Kind::B32)?;
        let _membermask = self.integer(membermask, Kind::B32)?;
        let value = self.read(value)?.require(Kind::B32, value)?;
        let xor_mask = self.integer(xor_mask, Kind::B32)?;
        // Native subgroup shuffle: no workgroup barrier, so warps that early-exit
        // after bar.sync (GGML mmvq) cannot hang the remaining warp's reduction.
        // SubgroupSize=32 (set on the entry) makes Intel subgroups match CUDA warps.
        let scope = self
            .builder
            .constant_bit32(self.u32_type, Scope::Subgroup as u32);
        let result = self.builder.group_non_uniform_shuffle_xor(
            self.u32_type,
            None,
            scope,
            value,
            xor_mask,
        )?;
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

    fn emit_redux_add(
        &mut self,
        dst: &'a str,
        value: &str,
        membermask: &str,
    ) -> Result<(), Error> {
        if let Some(membermask) = Self::parse_u32(membermask)
            && membermask != u32::MAX
        {
            return Err(Error::new(
                "only full-warp redux.sync.add.s32 membermask 0xffffffff is supported",
            ));
        }
        let _membermask = self.integer(membermask, Kind::B32)?;
        // Emulate CUDA `redux.sync.add.s32` with a 32-lane ShuffleXor butterfly.
        // Isolated `redux` and combined float-max+redux probes pass on Iris; the
        // native GroupNonUniformIAdd path was not reliable for GGML quantize.
        let scope = self
            .builder
            .constant_bit32(self.u32_type, Scope::Subgroup as u32);
        let mut acc = self.read(value)?.require(Kind::B32, value)?;
        for delta in [16u32, 8, 4, 2, 1] {
            let xor_mask = self.builder.constant_bit32(self.u32_type, delta);
            let other = self.builder.group_non_uniform_shuffle_xor(
                self.u32_type,
                None,
                scope,
                acc,
                xor_mask,
            )?;
            acc = self.builder.i_add(self.u32_type, None, acc, other)?;
        }
        self.write(
            dst,
            Value {
                id: acc,
                kind: Kind::B32,
            },
        )?;
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

    fn shared_symbol_offset(&self, name: &str) -> Result<u32, Error> {
        self.shared_offsets
            .get(name)
            .copied()
            .ok_or_else(|| Error::new(format!("unknown PTX shared symbol {name}")))
    }

    fn shared_window_bytes(&mut self) -> Result<Word, Error> {
        let array = self.shared_array.ok_or_else(|| {
            Error::new("shared memory access without a shared declaration")
        })?;
        let zero = self.builder.constant_bit32(self.u32_type, 0);
        let word_ptr = self
            .builder
            .access_chain(self.shared_word_ptr, None, array, [zero])?;
        Ok(self.builder.bitcast(self.shared_byte_ptr, None, word_ptr)?)
    }

    fn shared_base(&mut self, name: &str) -> Result<Word, Error> {
        let offset = self.shared_symbol_offset(name)?;
        let base = self.shared_window_bytes()?;
        if offset == 0 {
            // Return a word pointer at the window start (historical callers bitcast).
            Ok(self.builder.bitcast(self.shared_word_ptr, None, base)?)
        } else {
            let bytes = self.builder.constant_bit64(self.u64_type, u64::from(offset));
            let at = self
                .builder
                .ptr_access_chain(self.shared_byte_ptr, None, base, bytes, [])?;
            Ok(self.builder.bitcast(self.shared_word_ptr, None, at)?)
        }
    }

    fn local_base(&mut self, name: &str) -> Result<Word, Error> {
        let (_, array) = self
            .local
            .filter(|(symbol, _)| *symbol == name)
            .ok_or_else(|| Error::new(format!("unknown PTX local symbol {name}")))?;
        let zero = self.builder.constant_bit32(self.u32_type, 0);
        Ok(self
            .builder
            .access_chain(self.local_word_ptr, None, array, [zero])?)
    }

    fn parse_byte_offset(operand: &str) -> Option<i64> {
        if let Some(hex) = operand
            .strip_prefix("0x")
            .or_else(|| operand.strip_prefix("0X"))
        {
            i64::from_str_radix(hex, 16).ok()
        } else {
            operand.parse::<i64>().ok()
        }
    }

    fn addressed_byte_pointer(&mut self, space: Space, address: &str) -> Result<Word, Error> {
        // PTX allows `[%rd+18]` and `[%rd+-18]` (nvcc emits the latter for struct fields).
        let (base, offset) = match address.split_once('+') {
            Some((base, offset)) => (
                base,
                Self::parse_byte_offset(offset).ok_or_else(|| {
                    Error::new(format!("unsupported PTX address offset {offset}"))
                })?,
            ),
            None => (address, 0),
        };
        let byte_ty = match space {
            Space::Global => self.global_byte_ptr,
            Space::Shared => self.shared_byte_ptr,
            Space::Local => self.local_byte_ptr,
        };
        if space == Space::Global
            && let Some(&variable) = self.globals.get(base)
        {
            // Module-scope variable named directly in the address operand.
            let pointer = self.builder.bitcast(byte_ty, None, variable)?;
            if offset == 0 {
                return Ok(pointer);
            }
            let bytes = self.builder.constant_bit64(self.i64_type, offset as u64);
            return Ok(self.builder.ptr_access_chain(byte_ty, None, pointer, bytes, [])?);
        }
        let value = self.read(base)?;
        let bytes_ptr = match (space, value.kind) {
            (Space::Shared, Kind::B32) => {
                // 32-bit shared addressing: register holds a byte offset from the
                // unified shared window (symbol mov.u32 contributes its prefix offset).
                let base_bytes = self.shared_window_bytes()?;
                let off32 = value.id;
                let off64 = self.builder.u_convert(self.u64_type, None, off32)?;
                self.builder
                    .ptr_access_chain(byte_ty, None, base_bytes, off64, [])?
            }
            (_, Kind::Address(actual)) if actual == space => {
                self.builder.bitcast(byte_ty, None, value.id)?
            }
            // nvcc often keeps post-cvta pointers in .b64 via selp/mov; treat as
            // generic global addresses when used by ld.global/st.global.
            (Space::Global, Kind::B64) => self.builder.bitcast(byte_ty, None, value.id)?,
            // Same for `.local` after integer address math that lost Address typing.
            (Space::Local, Kind::B64) => self.builder.bitcast(byte_ty, None, value.id)?,
            _ => {
                return Err(Error::new(format!(
                    "expected {:?} address for {base}, got {:?}",
                    space, value.kind
                )));
            }
        };
        if offset == 0 {
            Ok(bytes_ptr)
        } else {
            // Use a signed i64 index so nvcc `[%rd+-N]` lowers correctly on Level Zero.
            let bytes = self
                .builder
                .constant_bit64(self.i64_type, offset as u64);
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
            Space::Local => self.local_word_ptr,
        };
        Ok(self.builder.bitcast(ty, None, bytes_ptr)?)
    }

    fn typed_pointer(&mut self, space: Space, element: Word) -> Word {
        let storage = match space {
            Space::Global => StorageClass::CrossWorkgroup,
            Space::Shared => StorageClass::Workgroup,
            Space::Local => StorageClass::Function,
        };
        self.builder.type_pointer(None, storage, element)
    }

    fn half_pointer(&mut self, space: Space, address: &str) -> Result<Word, Error> {
        let bytes_ptr = self.addressed_byte_pointer(space, address)?;
        let ty = self.typed_pointer(space, self.u16_type);
        Ok(self.builder.bitcast(ty, None, bytes_ptr)?)
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
            Space::Local => StorageClass::Function,
        };
        let pointer_type = self.builder.type_pointer(None, storage, vector);
        let byte_pointer = if space == Space::Shared && !address.starts_with('%') {
            let array = self.shared_base(address)?;
            self.builder.bitcast(self.shared_byte_ptr, None, array)?
        } else {
            self.addressed_byte_pointer(space, address)?
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
            FloatOp::FmaDirected { mode, src: [a, b, c] } => {
                let a = self.float_bits(a)?;
                let b = self.float_bits(b)?;
                let c = self.float_bits(c)?;
                self.fp32_fma = true;
                return rounded::fma_directed(&mut self.builder, self.opencl, mode, a, b, c);
            }
            FloatOp::Directed { kind, mode, src: [a, b] } => {
                let a = self.float_bits(a)?;
                let b = self.float_bits(b)?;
                self.fp32_fma = true;
                return rounded::directed(&mut self.builder, self.opencl, kind, mode, a, b);
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
            FloatOp::FromSigned16([source]) => {
                // Low 16 bits are a signed integer; sign-extend before f32 convert.
                let bits32 = self.integer(source, Kind::B32)?;
                let shift = self.builder.constant_bit32(self.u32_type, 16);
                let raised = self
                    .builder
                    .shift_left_logical(self.u32_type, None, bits32, shift)?;
                let sext = self
                    .builder
                    .shift_right_arithmetic(self.u32_type, None, raised, shift)?;
                let converted = self.builder.convert_s_to_f(ty, None, sext)?;
                self.builder.decorate(
                    converted,
                    Decoration::FPRoundingMode,
                    [Operand::FPRoundingMode(FPRoundingMode::RTE)],
                );
                converted
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
            FloatOp::RcpApprox([source]) => {
                let value = self.float_value(source)?;
                self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    OpenclStd100Op::native_recip as u32,
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
            FloatOp::Neg([source]) => {
                let value = self.float_value(source)?;
                self.builder.f_negate(ty, None, value)?
            }
            FloatOp::TanhApprox([source]) => {
                let value = self.float_value(source)?;
                self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    OpenclStd100Op::tanh as u32,
                    [Operand::IdRef(value)],
                )?
            }
            FloatOp::Abs([source]) => {
                let value = self.float_value(source)?;
                self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    OpenclStd100Op::fabs as u32,
                    [Operand::IdRef(value)],
                )?
            }
            FloatOp::CopySign([sign, magnitude]) => {
                let sign = self.float_bits(sign)?;
                let magnitude = self.float_bits(magnitude)?;
                let u32_type = self.u32_type;
                let sign_mask = self.builder.constant_bit32(u32_type, 0x8000_0000);
                let magnitude_mask = self.builder.constant_bit32(u32_type, 0x7fff_ffff);
                let sign = self.builder.bitwise_and(u32_type, None, sign, sign_mask)?;
                let magnitude = self.builder.bitwise_and(u32_type, None, magnitude, magnitude_mask)?;
                return Ok(self.builder.bitwise_or(u32_type, None, sign, magnitude)?);
            }
            FloatOp::Saturate([source]) => {
                let value = self.float_value(source)?;
                let zero = self.builder.constant_bit32(ty, 0);
                let one = self.builder.constant_bit32(ty, 0x3f80_0000);
                let low = self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    OpenclStd100Op::fmax as u32,
                    [value, zero].map(Operand::IdRef),
                )?;
                self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    OpenclStd100Op::fmin as u32,
                    [low, one].map(Operand::IdRef),
                )?
            }
            FloatOp::RoundNearest([source]) | FloatOp::Floor([source]) | FloatOp::Ceil([source]) => {
                let value = self.float_value(source)?;
                let op = match op {
                    FloatOp::RoundNearest(_) => OpenclStd100Op::rint,
                    FloatOp::Floor(_) => OpenclStd100Op::floor,
                    _ => OpenclStd100Op::ceil,
                };
                self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    op as u32,
                    [Operand::IdRef(value)],
                )?
            }
            FloatOp::Trunc([source]) => {
                let value = self.float_value(source)?;
                self.builder.ext_inst(
                    ty,
                    None,
                    self.opencl,
                    OpenclStd100Op::trunc as u32,
                    [Operand::IdRef(value)],
                )?
            }
        };
        self.fp32 = true;
        Ok(self.builder.bitcast(self.u32_type, None, result)?)
    }

    fn opencl_int(
        &mut self,
        ty: Word,
        op: OpenclStd100Op,
        args: &[Word],
    ) -> Result<Word, rspirv::dr::Error> {
        self.builder.ext_inst(
            ty,
            None,
            self.opencl,
            op as u32,
            args.iter().copied().map(Operand::IdRef),
        )
    }

    /// `1 << clamp(len)` style mask: all-ones for `len >= 32`, zero for `len == 0`.
    fn low_mask(&mut self, len: Word) -> Result<Word, Error> {
        let ty = self.u32_type;
        let c32 = self.builder.constant_bit32(ty, 32);
        let zero = self.builder.constant_bit32(ty, 0);
        let ones = self.builder.constant_bit32(ty, u32::MAX);
        let small = self.builder.u_less_than(self.bool_type, None, len, c32)?;
        let safe = self.builder.select(ty, None, small, len, zero)?;
        let shifted = self.builder.shift_left_logical(ty, None, ones, safe)?;
        let mask = self.builder.not(ty, None, shifted)?;
        // len == 0 -> mask 0 (shift by 0 gives ones -> not -> 0); len >= 32 -> ones.
        Ok(self.builder.select(ty, None, small, mask, ones)?)
    }

    fn alu3(&mut self, op: Alu3Op, src: [&str; 4]) -> Result<Word, Error> {
        let ty = self.u32_type;
        let c = |e: &mut Self, v: u32| e.builder.constant_bit32(ty, v);
        match op {
            Alu3Op::BfeUnsigned | Alu3Op::BfeSigned => {
                let a = self.integer(src[0], Kind::B32)?;
                let pos = self.integer(src[1], Kind::B32)?;
                let len = self.integer(src[2], Kind::B32)?;
                let ff = c(self, 0xff);
                let pos = self.builder.bitwise_and(ty, None, pos, ff)?;
                let len = self.builder.bitwise_and(ty, None, len, ff)?;
                let mask = self.low_mask(len)?;
                let c32 = c(self, 32);
                let c31 = c(self, 31);
                let zero = c(self, 0);
                if matches!(op, Alu3Op::BfeUnsigned) {
                    let in_range = self.builder.u_less_than(self.bool_type, None, pos, c32)?;
                    let safe = self.builder.select(ty, None, in_range, pos, zero)?;
                    let shifted = self.builder.shift_right_logical(ty, None, a, safe)?;
                    let shifted = self.builder.select(ty, None, in_range, shifted, zero)?;
                    return Ok(self.builder.bitwise_and(ty, None, shifted, mask)?);
                }
                // Signed: replicate bit min(pos + len - 1, 31) above the field.
                let big = self.builder.u_less_than(self.bool_type, None, c31, pos)?;
                let pos31 = self.builder.select(ty, None, big, c31, pos)?;
                let shifted = self.builder.shift_right_arithmetic(ty, None, a, pos31)?;
                let field = self.builder.bitwise_and(ty, None, shifted, mask)?;
                let one = c(self, 1);
                let last = self.builder.i_add(ty, None, pos, len)?;
                let last = self.builder.i_sub(ty, None, last, one)?;
                let over = self.builder.u_less_than(self.bool_type, None, c31, last)?;
                let last = self.builder.select(ty, None, over, c31, last)?;
                let sign = self.builder.shift_right_logical(ty, None, a, last)?;
                let sign = self.builder.bitwise_and(ty, None, sign, one)?;
                let len_zero = self.builder.i_equal(self.bool_type, None, len, zero)?;
                let sign = self.builder.select(ty, None, len_zero, zero, sign)?;
                let negative = self.builder.i_not_equal(self.bool_type, None, sign, zero)?;
                let inverse = self.builder.not(ty, None, mask)?;
                let extended = self.builder.bitwise_or(ty, None, field, inverse)?;
                Ok(self.builder.select(ty, None, negative, extended, field)?)
            }
            Alu3Op::Bfi => {
                let f = self.integer(src[0], Kind::B32)?;
                let a = self.integer(src[1], Kind::B32)?;
                let pos = self.integer(src[2], Kind::B32)?;
                let len = self.integer(src[3], Kind::B32)?;
                let ff = c(self, 0xff);
                let pos = self.builder.bitwise_and(ty, None, pos, ff)?;
                let len = self.builder.bitwise_and(ty, None, len, ff)?;
                let mask = self.low_mask(len)?;
                let c32 = c(self, 32);
                let zero = c(self, 0);
                let in_range = self.builder.u_less_than(self.bool_type, None, pos, c32)?;
                let safe = self.builder.select(ty, None, in_range, pos, zero)?;
                let field = self.builder.shift_left_logical(ty, None, mask, safe)?;
                let moved = self.builder.shift_left_logical(ty, None, f, safe)?;
                let moved = self.builder.bitwise_and(ty, None, moved, field)?;
                let keep = self.builder.not(ty, None, field)?;
                let keep = self.builder.bitwise_and(ty, None, a, keep)?;
                let merged = self.builder.bitwise_or(ty, None, keep, moved)?;
                Ok(self.builder.select(ty, None, in_range, merged, a)?)
            }
            Alu3Op::Prmt => {
                let a = self.integer(src[0], Kind::B32)?;
                let b = self.integer(src[1], Kind::B32)?;
                let sel = self.integer(src[2], Kind::B32)?;
                let a64 = self.builder.u_convert(self.u64_type, None, a)?;
                let b64 = self.builder.u_convert(self.u64_type, None, b)?;
                let c32 = self.builder.constant_bit64(self.u64_type, 32);
                let hi = self.builder.shift_left_logical(self.u64_type, None, b64, c32)?;
                let pair = self.builder.bitwise_or(self.u64_type, None, a64, hi)?;
                let mut result = c(self, 0);
                for lane in 0..4u32 {
                    let nibble_shift = c(self, 4 * lane);
                    let nibble = self.builder.shift_right_logical(ty, None, sel, nibble_shift)?;
                    let seven = c(self, 7);
                    let index = self.builder.bitwise_and(ty, None, nibble, seven)?;
                    let three = c(self, 3);
                    let bit_index = self.builder.shift_left_logical(ty, None, index, three)?;
                    let bit_index = self.builder.u_convert(self.u64_type, None, bit_index)?;
                    let picked = self.builder.shift_right_logical(self.u64_type, None, pair, bit_index)?;
                    let picked = self.builder.u_convert(ty, None, picked)?;
                    let byte_mask = c(self, 0xff);
                    let byte = self.builder.bitwise_and(ty, None, picked, byte_mask)?;
                    // Selector bit 3 replicates the byte's sign bit.
                    let seven_bits = c(self, 7);
                    let sign = self.builder.shift_right_logical(ty, None, byte, seven_bits)?;
                    let zero = c(self, 0);
                    let negated = self.builder.i_sub(ty, None, zero, sign)?;
                    let smear = self.builder.bitwise_and(ty, None, negated, byte_mask)?;
                    let eight = c(self, 8);
                    let flag = self.builder.bitwise_and(ty, None, nibble, eight)?;
                    let replicate = self.builder.i_not_equal(self.bool_type, None, flag, zero)?;
                    let value = self.builder.select(ty, None, replicate, smear, byte)?;
                    let lane_shift = c(self, 8 * lane);
                    let placed = self.builder.shift_left_logical(ty, None, value, lane_shift)?;
                    result = self.builder.bitwise_or(ty, None, result, placed)?;
                }
                Ok(result)
            }
            Alu3Op::Lop3 => {
                let operands = [
                    self.integer(src[0], Kind::B32)?,
                    self.integer(src[1], Kind::B32)?,
                    self.integer(src[2], Kind::B32)?,
                ];
                let lut = Self::parse_u32(src[3])
                    .filter(|&lut| lut <= 0xff)
                    .ok_or_else(|| Error::new("lop3.b32 requires an immediate lookup table"))?;
                let mut inverted = Vec::new();
                for operand in operands {
                    inverted.push(self.builder.not(ty, None, operand)?);
                }
                let mut result = c(self, 0);
                for index in 0..8u32 {
                    if lut >> index & 1 == 0 {
                        continue;
                    }
                    // Index bit 2 selects `a`, bit 1 selects `b`, bit 0 selects `c`.
                    let mut term = c(self, u32::MAX);
                    for (slot, shift) in [(0usize, 2u32), (1, 1), (2, 0)] {
                        let wanted = if index >> shift & 1 == 1 {
                            operands[slot]
                        } else {
                            inverted[slot]
                        };
                        term = self.builder.bitwise_and(ty, None, term, wanted)?;
                    }
                    result = self.builder.bitwise_or(ty, None, result, term)?;
                }
                Ok(result)
            }
        }
    }

    fn alu64(&mut self, op: Alu64Op, src: [&str; 3]) -> Result<(Word, Kind), Error> {
        let ty = self.u64_type;
        let a = self.integer(src[0], Kind::B64)?;
        let b = if matches!(op, Alu64Op::Neg | Alu64Op::AbsSigned | Alu64Op::PopCount) {
            a
        } else {
            self.integer(src[1], Kind::B64)?
        };
        let id = match op {
            Alu64Op::Xor => self.builder.bitwise_xor(ty, None, a, b)?,
            Alu64Op::MaxSigned => self.opencl_int(ty, OpenclStd100Op::s_max, &[a, b])?,
            Alu64Op::MinUnsigned => self.opencl_int(ty, OpenclStd100Op::u_min, &[a, b])?,
            Alu64Op::MaxUnsigned => self.opencl_int(ty, OpenclStd100Op::u_max, &[a, b])?,
            Alu64Op::Neg => {
                let zero = self.builder.constant_bit64(ty, 0);
                self.builder.i_sub(ty, None, zero, a)?
            }
            Alu64Op::AbsSigned => self.opencl_int(ty, OpenclStd100Op::s_abs, &[a])?,
            Alu64Op::MadLo => {
                let c = self.integer(src[2], Kind::B64)?;
                let product = self.builder.i_mul(ty, None, a, b)?;
                self.builder.i_add(ty, None, product, c)?
            }
            Alu64Op::PopCount => {
                let count = self.builder.bit_count(ty, None, a)?;
                return Ok((self.builder.u_convert(self.u32_type, None, count)?, Kind::B32));
            }
        };
        Ok((id, Kind::B64))
    }

    fn subgroup_shuffle(&mut self, value: Word, lane: Word) -> Result<Word, Error> {
        let scope = self
            .builder
            .constant_bit32(self.u32_type, Scope::Subgroup as u32);
        Ok(self.builder.group_non_uniform_shuffle(
            self.u32_type,
            None,
            scope,
            value,
            lane,
        )?)
    }

    /// CUDA lane id: the linear thread index modulo 32 (subgroups are 32 wide).
    fn lane_id(&mut self) -> Result<Word, Error> {
        let ty = self.u32_type;
        let linear = self.linear_tid()?;
        let mask = self.builder.constant_bit32(ty, 31);
        Ok(self.builder.bitwise_and(ty, None, linear, mask)?)
    }

    fn u32_const(&mut self, value: u32) -> Word {
        self.builder.constant_bit32(self.u32_type, value)
    }

    /// Four-way byte dot product with per-operand signedness, plus accumulate.
    fn dot4(&mut self, a: Word, b: Word, acc: Word, signed: [bool; 2]) -> Result<Word, Error> {
        let ty = self.u32_type;
        let c24 = self.u32_const(24);
        let ff = self.u32_const(0xff);
        let mut acc = acc;
        for lane in 0..4u32 {
            let shift = self.u32_const(lane * 8);
            let mut bytes = [0; 2];
            for (slot, (word, signed)) in [(a, signed[0]), (b, signed[1])].into_iter().enumerate() {
                let moved = self.builder.shift_right_logical(ty, None, word, shift)?;
                bytes[slot] = if signed {
                    let raised = self.builder.shift_left_logical(ty, None, moved, c24)?;
                    self.builder.shift_right_arithmetic(ty, None, raised, c24)?
                } else {
                    self.builder.bitwise_and(ty, None, moved, ff)?
                };
            }
            let product = self.builder.i_mul(ty, None, bytes[0], bytes[1])?;
            acc = self.builder.i_add(ty, None, acc, product)?;
        }
        Ok(acc)
    }

    /// Two IEEE halves packed in a word, as f32 products summed onto `acc`.
    fn half2_dot(&mut self, a: Word, b: Word, acc: Word) -> Result<Word, Error> {
        self.fp32 = true;
        let mut acc = acc;
        for half in 0..2u32 {
            let shift = self.u32_const(half * 16);
            let mut values = [0; 2];
            for (slot, word) in [a, b].into_iter().enumerate() {
                let moved = self.builder.shift_right_logical(self.u32_type, None, word, shift)?;
                let bits = self.builder.u_convert(self.u16_type, None, moved)?;
                let half = self.builder.bitcast(self.f16_type, None, bits)?;
                values[slot] = self.builder.f_convert(self.f32_type, None, half)?;
            }
            let product = self.builder.f_mul(self.f32_type, None, values[0], values[1])?;
            acc = self.builder.f_add(self.f32_type, None, acc, product)?;
        }
        Ok(acc)
    }

    fn emit_mma(
        &mut self,
        shape: MmaShape,
        signed: [bool; 2],
        dst: [&'a str; 4],
        src: [&'a str; 10],
    ) -> Result<(), Error> {
        let ty = self.u32_type;
        let accumulators = shape.accumulators();
        let groups = shape.groups();
        let halves = accumulators / 2;
        let mut a = Vec::new();
        for name in &src[..halves * (groups / 4)] {
            a.push(self.integer(name, Kind::B32)?);
        }
        let mut b = Vec::new();
        for name in &src[4..4 + groups / 4] {
            b.push(self.integer(name, Kind::B32)?);
        }
        let mut c = Vec::new();
        for name in &src[6..6 + accumulators] {
            c.push(self.integer(name, Kind::B32)?);
        }
        let lane = self.lane_id()?;
        let two = self.u32_const(2);
        let three = self.u32_const(3);
        let four = self.u32_const(4);
        let row = self.builder.shift_right_logical(ty, None, lane, two)?;
        let quad = self.builder.bitwise_and(ty, None, lane, three)?;
        let row4 = self.builder.i_mul(ty, None, row, four)?;
        let one_bit = self.u32_const(1);
        let col_base = self.builder.shift_left_logical(ty, None, quad, one_bit)?;
        // A words are shared by both accumulator columns of a row half; B words by both rows.
        let mut a_words = HashMap::new();
        let mut b_words = HashMap::new();
        let mut results = Vec::new();
        for (j, &start) in c.iter().enumerate() {
            let (half, parity) = (j / 2, j % 2);
            let column = if parity == 0 {
                col_base
            } else {
                let one = self.u32_const(1);
                self.builder.i_add(ty, None, col_base, one)?
            };
            let column4 = self.builder.i_mul(ty, None, column, four)?;
            let mut acc = start;
            if !shape.is_integer() {
                acc = self.builder.bitcast(self.f32_type, None, acc)?;
            }
            for group in 0..groups {
                let low = self.u32_const((group % 4) as u32);
                let a_word = match a_words.get(&(half, group)) {
                    Some(&word) => word,
                    None => {
                        let source = self.builder.i_add(ty, None, row4, low)?;
                        let word = self.subgroup_shuffle(a[half + 2 * (group / 4)], source)?;
                        a_words.insert((half, group), word);
                        word
                    }
                };
                let b_word = match b_words.get(&(parity, group)) {
                    Some(&word) => word,
                    None => {
                        let source = self.builder.i_add(ty, None, column4, low)?;
                        let word = self.subgroup_shuffle(b[group / 4], source)?;
                        b_words.insert((parity, group), word);
                        word
                    }
                };
                acc = if shape.is_integer() {
                    self.dot4(a_word, b_word, acc, signed)?
                } else {
                    self.half2_dot(a_word, b_word, acc)?
                };
            }
            if !shape.is_integer() {
                acc = self.builder.bitcast(ty, None, acc)?;
            }
            results.push(acc);
        }
        for (name, id) in dst.iter().zip(results) {
            self.write(name, Value { id, kind: Kind::B32 })?;
        }
        Ok(())
    }

    /// 32-bit byte offset into the shared window for a shared address operand.
    fn shared_offset32(&mut self, operand: &str) -> Result<Word, Error> {
        let (base, offset) = match operand.split_once('+') {
            Some((base, offset)) => (
                base,
                Self::parse_byte_offset(offset)
                    .ok_or_else(|| Error::new(format!("unsupported PTX address offset {offset}")))?,
            ),
            None => (operand, 0),
        };
        let value = self.read(base)?;
        let ty = self.u32_type;
        let id = match value.kind {
            Kind::B32 => value.id,
            Kind::B64 => self.builder.u_convert(ty, None, value.id)?,
            Kind::Address(Space::Shared) => {
                let window = self.shared_window_bytes()?;
                let pointer = self.builder.convert_ptr_to_u(self.u64_type, None, value.id)?;
                let origin = self.builder.convert_ptr_to_u(self.u64_type, None, window)?;
                let delta = self.builder.i_sub(self.u64_type, None, pointer, origin)?;
                self.builder.u_convert(ty, None, delta)?
            }
            other => {
                return Err(Error::new(format!(
                    "expected a shared address for {base}, got {other:?}"
                )));
            }
        };
        if offset == 0 {
            return Ok(id);
        }
        let delta = self.u32_const(offset as u32);
        Ok(self.builder.i_add(ty, None, id, delta)?)
    }

    fn shared_pointer_at(&mut self, offset: Word) -> Result<Word, Error> {
        let window = self.shared_window_bytes()?;
        let wide = self.builder.u_convert(self.u64_type, None, offset)?;
        Ok(self
            .builder
            .ptr_access_chain(self.shared_byte_ptr, None, window, wide, [])?)
    }

    fn emit_ldmatrix(
        &mut self,
        count: u8,
        trans: bool,
        dst: [&'a str; 4],
        addr: &str,
    ) -> Result<(), Error> {
        let ty = self.u32_type;
        let offset = self.shared_offset32(addr)?;
        let lane = self.lane_id()?;
        let two = self.u32_const(2);
        let three = self.u32_const(3);
        let row = self.builder.shift_right_logical(ty, None, lane, two)?;
        let quad = self.builder.bitwise_and(ty, None, lane, three)?;
        let half_ptr = self.builder.type_pointer(None, StorageClass::Workgroup, self.u16_type);
        let word_ptr = self.shared_word_ptr;
        let mut words = Vec::new();
        for matrix in 0..u32::from(count) {
            let base_lane = self.u32_const(8 * matrix);
            let word = if !trans {
                // Lane j of the matrix supplies the address of row j.
                let source = self.builder.i_add(ty, None, base_lane, row)?;
                let row_offset = self.subgroup_shuffle(offset, source)?;
                let four = self.u32_const(4);
                let into = self.builder.i_mul(ty, None, quad, four)?;
                let at = self.builder.i_add(ty, None, row_offset, into)?;
                let bytes = self.shared_pointer_at(at)?;
                let pointer = self.builder.bitcast(word_ptr, None, bytes)?;
                self.builder.load(ty, None, pointer, None, [])?
            } else {
                // The thread owns elements [2q][row] and [2q+1][row] of the matrix.
                let one = self.u32_const(1);
                let first = self.builder.shift_left_logical(ty, None, quad, one)?;
                let first = self.builder.i_add(ty, None, base_lane, first)?;
                let second = self.builder.i_add(ty, None, first, one)?;
                let into = self.builder.shift_left_logical(ty, None, row, one)?;
                let mut halves = [0; 2];
                for (slot, source) in [first, second].into_iter().enumerate() {
                    let row_offset = self.subgroup_shuffle(offset, source)?;
                    let at = self.builder.i_add(ty, None, row_offset, into)?;
                    let bytes = self.shared_pointer_at(at)?;
                    let pointer = self.builder.bitcast(half_ptr, None, bytes)?;
                    let value = self.builder.load(self.u16_type, None, pointer, None, [])?;
                    halves[slot] = self.builder.u_convert(ty, None, value)?;
                }
                let sixteen = self.u32_const(16);
                let high = self.builder.shift_left_logical(ty, None, halves[1], sixteen)?;
                self.builder.bitwise_or(ty, None, halves[0], high)?
            };
            words.push(word);
        }
        for (name, id) in dst.iter().zip(words) {
            self.write(name, Value { id, kind: Kind::B32 })?;
        }
        Ok(())
    }

    /// Space of the pointer held by the base register of `address`, for
    /// generic (`Space::Global` as parsed) accesses through non-global pointers.
    fn generic_space(&self, space: Space, address: &str) -> Space {
        let base = address.split('+').next().unwrap_or(address);
        match self.registers.get(base) {
            Some(Value { kind: Kind::Address(actual), .. }) if space == Space::Global => *actual,
            _ => space,
        }
    }

    fn resolve_generic(&self, inst: Instruction<'a>) -> Instruction<'a> {
        use Instruction::*;
        match inst {
            Load32(s, d, a) => Load32(self.generic_space(s, a), d, a),
            Load64(s, d, a) => Load64(self.generic_space(s, a), d, a),
            Load16(s, d, a) => Load16(self.generic_space(s, a), d, a),
            Load8(s, d, a) => Load8(self.generic_space(s, a), d, a),
            LoadSigned { space, bits, dst, addr } => LoadSigned {
                space: self.generic_space(space, addr),
                bits,
                dst,
                addr,
            },
            LoadVec { space, count, dst, addr } => LoadVec {
                space: self.generic_space(space, addr),
                count,
                dst,
                addr,
            },
            Store32(s, a, v) => Store32(self.generic_space(s, a), a, v),
            Store64(s, a, v) => Store64(self.generic_space(s, a), a, v),
            Store16(s, a, v) => Store16(self.generic_space(s, a), a, v),
            Store8(s, a, v) => Store8(self.generic_space(s, a), a, v),
            StoreVec { space, count, addr, src } => StoreVec {
                space: self.generic_space(space, addr),
                count,
                addr,
                src,
            },
            other => other,
        }
    }

    fn declare(&mut self, name: &'static str, capability: Option<Capability>, extension: Option<&str>) {
        if self.declared.insert(name) {
            if let Some(capability) = capability {
                self.builder.capability(capability);
            }
            if let Some(extension) = extension {
                self.builder.extension(extension);
            }
        }
    }

    fn emit_atomic(
        &mut self,
        op: AtomicOp,
        space: Space,
        float: bool,
        wide: bool,
        signed: bool,
        dst: Option<&'a str>,
        addr: &str,
        src: [&str; 2],
    ) -> Result<(), Error> {
        if space == Space::Local {
            return Err(Error::new("atomics on local memory are not supported"));
        }
        let (ty, kind) = if wide {
            self.declare("int64_atomics", Some(Capability::Int64Atomics), None);
            (self.u64_type, Kind::B64)
        } else {
            (self.u32_type, Kind::B32)
        };
        let pointer = if wide {
            let bytes = self.addressed_byte_pointer(space, addr)?;
            let wide_ptr = self.typed_pointer(space, self.u64_type);
            self.builder.bitcast(wide_ptr, None, bytes)?
        } else {
            self.word_pointer(space, addr)?
        };
        let scope_kind = if space == Space::Global { Scope::Device } else { Scope::Workgroup };
        if space == Space::Global {
            self.device_atomics = true;
        }
        let scope = self.builder.constant_bit32(self.u32_type, scope_kind as u32);
        let relaxed = self.builder.constant_bit32(self.u32_type, MemorySemantics::empty().bits());
        let value = self.integer(src[0], kind)?;
        let previous = match op {
            AtomicOp::Inc | AtomicOp::Dec => {
                // atom.inc: old >= b ? 0 : old + 1; atom.dec: (old == 0 || old > b) ? b : old - 1.
                // Compare-exchange loop on the 32-bit word.
                let zero = self.u32_const(0);
                let one = self.u32_const(1);
                let init = self.builder.atomic_i_add(ty, None, pointer, scope, relaxed, zero)?;
                let before = self.builder.id();
                self.builder.branch(before)?;
                self.builder.begin_block(Some(before))?;
                let head = self.builder.id();
                let exit = self.builder.id();
                self.builder.branch(head)?;
                self.builder.begin_block(Some(head))?;
                let got = self.builder.id();
                let old = self.builder.phi(ty, None, [(init, before), (got, head)])?;
                let new = if op == AtomicOp::Inc {
                    let wrap = self.builder.u_greater_than_equal(self.bool_type, None, old, value)?;
                    let next = self.builder.i_add(ty, None, old, one)?;
                    self.builder.select(ty, None, wrap, zero, next)?
                } else {
                    let is_zero = self.builder.i_equal(self.bool_type, None, old, zero)?;
                    let above = self.builder.u_greater_than(self.bool_type, None, old, value)?;
                    let wrap = self.builder.logical_or(self.bool_type, None, is_zero, above)?;
                    let next = self.builder.i_sub(ty, None, old, one)?;
                    self.builder.select(ty, None, wrap, value, next)?
                };
                self.builder.atomic_compare_exchange(
                    ty, Some(got), pointer, scope, relaxed, relaxed, new, old,
                )?;
                let done = self.builder.i_equal(self.bool_type, None, got, old)?;
                self.builder.branch_conditional(done, exit, head, [])?;
                self.builder.begin_block(Some(exit))?;
                old
            }
            AtomicOp::Add if float => {
                self.declare(
                    "atomic_f32_add",
                    Some(Capability::AtomicFloat32AddEXT),
                    Some("SPV_EXT_shader_atomic_float_add"),
                );
                let storage = if space == Space::Global {
                    StorageClass::CrossWorkgroup
                } else {
                    StorageClass::Workgroup
                };
                let float_ptr = self.builder.type_pointer(None, storage, self.f32_type);
                let float_ptr = self.builder.bitcast(float_ptr, None, pointer)?;
                let value = self.builder.bitcast(self.f32_type, None, value)?;
                let old = self.builder.atomic_f_add_ext(
                    self.f32_type,
                    None,
                    float_ptr,
                    scope,
                    relaxed,
                    value,
                )?;
                self.builder.bitcast(ty, None, old)?
            }
            AtomicOp::Add => self.builder.atomic_i_add(ty, None, pointer, scope, relaxed, value)?,
            AtomicOp::Min if signed => {
                self.builder.atomic_s_min(ty, None, pointer, scope, relaxed, value)?
            }
            AtomicOp::Min => self.builder.atomic_u_min(ty, None, pointer, scope, relaxed, value)?,
            AtomicOp::Max if signed => {
                self.builder.atomic_s_max(ty, None, pointer, scope, relaxed, value)?
            }
            AtomicOp::Max => self.builder.atomic_u_max(ty, None, pointer, scope, relaxed, value)?,
            AtomicOp::And => self.builder.atomic_and(ty, None, pointer, scope, relaxed, value)?,
            AtomicOp::Or => self.builder.atomic_or(ty, None, pointer, scope, relaxed, value)?,
            AtomicOp::Xor => self.builder.atomic_xor(ty, None, pointer, scope, relaxed, value)?,
            AtomicOp::Exch => {
                self.builder.atomic_exchange(ty, None, pointer, scope, relaxed, value)?
            }
            AtomicOp::Cas => {
                let compare = self.integer(src[1], kind)?;
                self.builder.atomic_compare_exchange(
                    ty, None, pointer, scope, relaxed, relaxed, value, compare,
                )?
            }
        };
        if let Some(dst) = dst {
            self.write(dst, Value { id: previous, kind })?;
        }
        Ok(())
    }

    fn emit_shuffle(
        &mut self,
        mode: ShuffleMode,
        dst: &'a str,
        pred: Option<&'a str>,
        src: [&str; 4],
    ) -> Result<(), Error> {
        let ty = self.u32_type;
        // The packed clamp/segment-mask operand is an immediate or a register nvcc filled
        // with a constant; compute it from the operand at run time (constants fold).
        if Self::parse_u32(src[3]).is_some_and(|mask| mask != u32::MAX) {
            return Err(Error::new("only full-warp shfl.sync membermask 0xffffffff is supported"));
        }
        let control = self.integer(src[2], Kind::B32)?;
        let c31 = self.u32_const(31);
        let c8 = self.u32_const(8);
        let clamp = self.builder.bitwise_and(ty, None, control, c31)?;
        let shifted = self.builder.shift_right_logical(ty, None, control, c8)?;
        let segment = self.builder.bitwise_and(ty, None, shifted, c31)?;
        let keep = self.builder.bitwise_xor(ty, None, segment, c31)?;
        let value = self.read(src[0])?.require(Kind::B32, src[0])?;
        let delta = self.integer(src[1], Kind::B32)?;
        let lane = self.lane_id()?;
        let min_lane = self.builder.bitwise_and(ty, None, lane, segment)?;
        let (source, valid) = match mode {
            ShuffleMode::Idx | ShuffleMode::Down => {
                let upper = self.builder.bitwise_and(ty, None, clamp, keep)?;
                let max_lane = self.builder.bitwise_or(ty, None, min_lane, upper)?;
                let j = if mode == ShuffleMode::Idx {
                    let low = self.builder.bitwise_and(ty, None, delta, keep)?;
                    self.builder.bitwise_or(ty, None, low, min_lane)?
                } else {
                    self.builder.i_add(ty, None, lane, delta)?
                };
                let valid = self.builder.u_less_than_equal(self.bool_type, None, j, max_lane)?;
                (j, valid)
            }
            ShuffleMode::Up => {
                let room = self.builder.i_sub(ty, None, lane, min_lane)?;
                let valid = self.builder.u_less_than_equal(self.bool_type, None, delta, room)?;
                (self.builder.i_sub(ty, None, lane, delta)?, valid)
            }
        };
        let source = self.builder.select(ty, None, valid, source, lane)?;
        let result = self.subgroup_shuffle(value, source)?;
        self.write(dst, Value { id: result, kind: Kind::B32 })?;
        if let Some(pred) = pred {
            self.write(pred, Value { id: valid, kind: Kind::Predicate })?;
        }
        Ok(())
    }

    fn emit_vote(&mut self, mode: VoteMode, dst: &'a str, src: &str) -> Result<(), Error> {
        let predicate = if src == "1" {
            self.builder.constant_true(self.bool_type)
        } else {
            self.read(src)?.require(Kind::Predicate, src)?
        };
        let scope = self.builder.constant_bit32(self.u32_type, Scope::Subgroup as u32);
        match mode {
            VoteMode::Ballot => {
                self.declare("ballot", Some(Capability::GroupNonUniformBallot), None);
                let vec4 = self.builder.type_vector(self.u32_type, 4);
                let mask = self.builder.group_non_uniform_ballot(vec4, None, scope, predicate)?;
                let low = self.builder.composite_extract(self.u32_type, None, mask, [0])?;
                self.write(dst, Value { id: low, kind: Kind::B32 })
            }
            VoteMode::All | VoteMode::Any => {
                self.declare("vote", Some(Capability::GroupNonUniformVote), None);
                let id = if mode == VoteMode::All {
                    self.builder.group_non_uniform_all(self.bool_type, None, scope, predicate)?
                } else {
                    self.builder.group_non_uniform_any(self.bool_type, None, scope, predicate)?
                };
                self.write(dst, Value { id, kind: Kind::Predicate })
            }
        }
    }

    fn instruction(&mut self, inst: Instruction<'a>) -> Result<(), Error> {
        let inst = self.resolve_generic(inst);
        let (dst, id, kind) = match inst {
            Instruction::Atomic { op, space, float, wide, signed, dst, addr, src } => {
                self.emit_atomic(op, space, float, wide, signed, dst, addr, src)?;
                return Ok(());
            }
            Instruction::Shuffle { mode, dst, pred, src } => {
                self.emit_shuffle(mode, dst, pred, src)?;
                return Ok(());
            }
            Instruction::Vote { mode, dst, src } => {
                self.emit_vote(mode, dst, src)?;
                return Ok(());
            }
            Instruction::Ext { op, dst, src } => {
                self.emit_ext(op, dst, src)?;
                return Ok(());
            }
            Instruction::Fence => {
                let ty = self.u32_type;
                let scope = self.builder.constant_bit32(ty, Scope::Device as u32);
                let semantics = self.builder.constant_bit32(
                    ty,
                    (MemorySemantics::ACQUIRE_RELEASE
                        | MemorySemantics::CROSS_WORKGROUP_MEMORY
                        | MemorySemantics::WORKGROUP_MEMORY)
                        .bits(),
                );
                self.builder.memory_barrier(scope, semantics)?;
                return Ok(());
            }
            Instruction::LoadSigned { space, bits, dst, addr } => {
                let ty = self.u32_type;
                let loaded = if bits == 8 {
                    let bytes = self.addressed_byte_pointer(space, addr)?;
                    let value = self.builder.load(self.u8_type, None, bytes, None, [])?;
                    self.builder.s_convert(ty, None, value)?
                } else {
                    let pointer = self.half_pointer(space, addr)?;
                    let value = self.builder.load(self.u16_type, None, pointer, None, [])?;
                    self.builder.s_convert(ty, None, value)?
                };
                (dst, loaded, Kind::B32)
            }
            Instruction::CvtaShared(dst, src) => {
                let value = self.read(src)?;
                let id = match value.kind {
                    Kind::Address(Space::Shared) => value.id,
                    Kind::B32 | Kind::B64 => {
                        let offset = self.shared_offset32(src)?;
                        let bytes = self.shared_pointer_at(offset)?;
                        bytes
                    }
                    other => {
                        return Err(Error::new(format!(
                            "expected a shared address for {src}, got {other:?}"
                        )));
                    }
                };
                (dst, id, Kind::Address(Space::Shared))
            }
            Instruction::CvtaToShared { wide, dst, src } => {
                let offset = self.shared_offset32(src)?;
                if wide {
                    (dst, self.builder.u_convert(self.u64_type, None, offset)?, Kind::B64)
                } else {
                    (dst, offset, Kind::B32)
                }
            }
            Instruction::LaneId(dst) => (dst, self.lane_id()?, Kind::B32),
            Instruction::Special(dst, name) => {
                let ty = self.u32_type;
                let lane = self.lane_id()?;
                let one = self.u32_const(1);
                let bit = self.builder.shift_left_logical(ty, None, one, lane)?;
                let below = self.builder.i_sub(ty, None, bit, one)?;
                let id = match name {
                    "%lanemask_eq" => bit,
                    "%lanemask_lt" => below,
                    "%lanemask_le" => self.builder.bitwise_or(ty, None, below, bit)?,
                    "%lanemask_gt" => {
                        let le = self.builder.bitwise_or(ty, None, below, bit)?;
                        self.builder.not(ty, None, le)?
                    }
                    "%lanemask_ge" => self.builder.not(ty, None, below)?,
                    _ => {
                        let linear = self.linear_tid()?;
                        let five = self.u32_const(5);
                        self.builder.shift_right_logical(ty, None, linear, five)?
                    }
                };
                (dst, id, Kind::B32)
            }
            Instruction::Mma {
                shape,
                signed,
                dst,
                src,
            } => {
                self.emit_mma(shape, signed, dst, src)?;
                return Ok(());
            }
            Instruction::LdMatrix {
                count,
                trans,
                dst,
                addr,
            } => {
                self.emit_ldmatrix(count, trans, dst, addr)?;
                return Ok(());
            }
            Instruction::Alu3(op, dst, src) => (dst, self.alu3(op, src)?, Kind::B32),
            Instruction::Alu64(op, dst, src) => {
                let (id, kind) = self.alu64(op, src)?;
                (dst, id, kind)
            }
            Instruction::Param(width, dst, src, offset) => {
                if width == Width::B64 {
                    if let Some(value) = self.params.get(&(src, offset)).copied() {
                        // Native .u64 / promoted GlobalPtr both occupy a 64-bit slot.
                        if value.kind.width() == Width::B64 {
                            (dst, value.id, value.kind)
                        } else if value.kind == Kind::B32 {
                            // Aggregate `.b8` params are stored as B32 words; assemble a u64.
                            let hi = self
                                .params
                                .get(&(src, offset + 4))
                                .copied()
                                .ok_or_else(|| {
                                    Error::new(format!(
                                        "unknown PTX parameter word {src}+{}",
                                        offset + 4
                                    ))
                                })?;
                            if hi.kind != Kind::B32 {
                                return Err(Error::new(format!(
                                    "mismatched PTX parameter load for {src}+{offset}"
                                )));
                            }
                            // SPIR-V requires the shift operand bitwidth to match
                            // the base. A u32 shift here made IGC miscompile
                            // aggregate `ld.param.u64` (soft_max ncols → 0).
                            let lo64 = self.builder.u_convert(self.u64_type, None, value.id)?;
                            let hi64 = self.builder.u_convert(self.u64_type, None, hi.id)?;
                            let shift = self.builder.constant_bit64(self.u64_type, 32);
                            let hi64 = self.builder.shift_left_logical(
                                self.u64_type,
                                None,
                                hi64,
                                shift,
                            )?;
                            let id =
                                self.builder.bitwise_or(self.u64_type, None, lo64, hi64)?;
                            (dst, id, Kind::B64)
                        } else {
                            return Err(Error::new(format!(
                                "mismatched PTX parameter load for {src}+{offset}"
                            )));
                        }
                    } else {
                        return Err(Error::new(format!(
                            "unknown PTX parameter word {src}+{offset}"
                        )));
                    }
                } else {
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
            }
            Instruction::ParamS32(dst, src, offset) => {
                let value = self.params.get(&(src, offset)).copied().ok_or_else(|| {
                    Error::new(format!("unknown PTX parameter word {src}+{offset}"))
                })?;
                let id = value.require(Kind::B32, src)?;
                if self.declarations.get(dst) == Some(&Width::B64) {
                    (dst, self.builder.s_convert(self.u64_type, None, id)?, Kind::B64)
                } else {
                    (dst, id, Kind::B32)
                }
            }
            Instruction::ParamS8(dst, src) => {
                let value = self.params.get(&(src, 0)).copied().ok_or_else(|| {
                    Error::new(format!("unknown PTX parameter word {src}"))
                })?;
                if value.kind != Kind::B32 {
                    return Err(Error::new(format!(
                        "mismatched PTX parameter load for {src}"
                    )));
                }
                // Sign-extend the low 8 bits into the b16/b32 destination slot.
                let ty = self.u32_type;
                let shift = self.builder.constant_bit32(ty, 24);
                let raised = self
                    .builder
                    .shift_left_logical(ty, None, value.id, shift)?;
                let id = self
                    .builder
                    .shift_right_arithmetic(ty, None, raised, shift)?;
                (dst, id, Kind::B32)
            }
            Instruction::ParamVec {
                count,
                dst,
                name,
                offset,
            } => {
                for index in 0..usize::from(count) {
                    let word = offset + index * 4;
                    let value = self.params.get(&(name, word)).copied().ok_or_else(|| {
                        Error::new(format!("unknown PTX parameter word {name}+{word}"))
                    })?;
                    if value.kind != Kind::B32 {
                        return Err(Error::new(format!(
                            "mismatched PTX parameter load for {name}+{word}"
                        )));
                    }
                    self.write(dst[index], value)?;
                }
                return Ok(());
            }
            Instruction::ToGlobal(dst, src) => {
                // Accept either an already-typed global pointer or a raw .u64 address.
                let kind = Kind::Address(Space::Global);
                let value = self.read(src)?;
                let id = match value.kind {
                    Kind::Address(Space::Global) => value.id,
                    Kind::B64 => self.builder.bitcast(self.global_byte_ptr, None, value.id)?,
                    Kind::Address(Space::Shared) => {
                        return Err(Error::new(format!(
                            "cannot cvta.to.global from shared pointer {src}"
                        )));
                    }
                    Kind::Address(Space::Local) => {
                        return Err(Error::new(format!(
                            "cannot cvta.to.global from local pointer {src}"
                        )));
                    }
                    other => {
                        return Err(Error::new(format!(
                            "expected B64 or global pointer for {src}, got {other:?}"
                        )));
                    }
                };
                (dst, id, kind)
            }
            Instruction::SharedAddress(dst, src) => {
                let pointer = self.shared_base(src)?;
                let id = self.builder.bitcast(self.shared_byte_ptr, None, pointer)?;
                (dst, id, Kind::Address(Space::Shared))
            }
            Instruction::SharedSymbol32(dst, src) => {
                // CUDA 32-bit shared: symbol address is the byte offset in the window.
                let offset = self.shared_symbol_offset(src)?;
                let id = self.builder.constant_bit32(self.u32_type, offset);
                (dst, id, Kind::B32)
            }
            Instruction::GlobalAddress(dst, src) => {
                let variable = *self
                    .globals
                    .get(src)
                    .ok_or_else(|| Error::new(format!("unknown PTX global symbol {src}")))?;
                let id = self.builder.bitcast(self.global_byte_ptr, None, variable)?;
                (dst, id, Kind::Address(Space::Global))
            }
            Instruction::LocalSymbol64(dst, src) => {
                let pointer = self.local_base(src)?;
                let id = self.builder.bitcast(self.local_byte_ptr, None, pointer)?;
                (dst, id, Kind::Address(Space::Local))
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
            Instruction::Dp4a(dst, [a, b, c]) => {
                // Soft-emulate CUDA dp4a.s32.s32: four signed i8×i8 products + accumulate.
                let a = self.integer(a, Kind::B32)?;
                let b = self.integer(b, Kind::B32)?;
                let mut acc = self.integer(c, Kind::B32)?;
                let ty = self.u32_type;
                let shift24 = self.builder.constant_bit32(ty, 24);
                for lane in 0..4u32 {
                    let shift = self.builder.constant_bit32(ty, lane * 8);
                    let a_byte = self.builder.shift_right_logical(ty, None, a, shift)?;
                    let b_byte = self.builder.shift_right_logical(ty, None, b, shift)?;
                    let a_sext = {
                        let raised = self.builder.shift_left_logical(ty, None, a_byte, shift24)?;
                        self.builder
                            .shift_right_arithmetic(ty, None, raised, shift24)?
                    };
                    let b_sext = {
                        let raised = self.builder.shift_left_logical(ty, None, b_byte, shift24)?;
                        self.builder
                            .shift_right_arithmetic(ty, None, raised, shift24)?
                    };
                    let prod = self.builder.i_mul(ty, None, a_sext, b_sext)?;
                    acc = self.builder.i_add(ty, None, acc, prod)?;
                }
                (dst, acc, Kind::B32)
            }
            Instruction::Compare {
                op,
                signed,
                bits16,
                dst,
                src: [a, b],
            } => {
                let mut a = self.integer(a, Kind::B32)?;
                let mut b = self.integer(b, Kind::B32)?;
                if bits16 {
                    let shift = self.builder.constant_bit32(self.u32_type, 16);
                    let narrow = |emitter: &mut Self, value: Word, signed: bool| -> Result<Word, Error> {
                        let raised =
                            emitter
                                .builder
                                .shift_left_logical(emitter.u32_type, None, value, shift)?;
                        if signed {
                            Ok(emitter.builder.shift_right_arithmetic(
                                emitter.u32_type,
                                None,
                                raised,
                                shift,
                            )?)
                        } else {
                            let mask = emitter.builder.constant_bit32(emitter.u32_type, 0xffff);
                            Ok(emitter.builder.bitwise_and(emitter.u32_type, None, value, mask)?)
                        }
                    };
                    a = narrow(self, a, signed)?;
                    b = narrow(self, b, signed)?;
                }
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
            Instruction::CompareFloat {
                op,
                dst,
                src: [a, b],
            } => {
                let a = self.float_value(a)?;
                let b = self.float_value(b)?;
                let ty = self.bool_type;
                // PTX float setp uses ordered compares (NaN → false for eq/lt/…).
                let id = match op {
                    Comparison::Equal => self.builder.f_ord_equal(ty, None, a, b),
                    Comparison::NotEqual => self.builder.f_ord_not_equal(ty, None, a, b),
                    Comparison::Less => self.builder.f_ord_less_than(ty, None, a, b),
                    Comparison::LessEqual => self.builder.f_ord_less_than_equal(ty, None, a, b),
                    Comparison::Greater => self.builder.f_ord_greater_than(ty, None, a, b),
                    Comparison::GreaterEqual => {
                        self.builder.f_ord_greater_than_equal(ty, None, a, b)
                    }
                }?;
                self.fp32 = true;
                (dst, id, Kind::Predicate)
            }
            Instruction::Compare64 {
                op,
                signed,
                dst,
                src: [a, b],
            } => {
                let a = self.integer(a, Kind::B64)?;
                let b = self.integer(b, Kind::B64)?;
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
            Instruction::Select {
                kind,
                dst,
                src: [a, b, pred],
            } => {
                let cond = self.read(pred)?.require(Kind::Predicate, pred)?;
                let a_global = matches!(
                    self.registers.get(a).map(|v| v.kind),
                    Some(Kind::Address(Space::Global))
                );
                let b_global = matches!(
                    self.registers.get(b).map(|v| v.kind),
                    Some(Kind::Address(Space::Global))
                );
                // Preserve global pointers through `selp.b64` so later address
                // arithmetic (`add.s64` + `ld.global`) keeps Address typing.
                if kind == Width::B64 && (a_global || b_global) {
                    let mut to_ptr = |name: &str, is_addr: bool| -> Result<Word, Error> {
                        if is_addr {
                            Ok(self.read(name)?.id)
                        } else if Self::parse_u64(name) == Some(0) {
                            let zero = self.builder.constant_bit64(self.u64_type, 0);
                            Ok(self.builder.bitcast(self.global_byte_ptr, None, zero)?)
                        } else {
                            let bits = self.integer(name, Kind::B64)?;
                            Ok(self.builder.bitcast(self.global_byte_ptr, None, bits)?)
                        }
                    };
                    let a = to_ptr(a, a_global)?;
                    let b = to_ptr(b, b_global)?;
                    let id = self
                        .builder
                        .select(self.global_byte_ptr, None, cond, a, b)?;
                    (dst, id, Kind::Address(Space::Global))
                } else {
                    let value_kind = match kind {
                        Width::B32 => Kind::B32,
                        Width::B64 => Kind::B64,
                        Width::Predicate => {
                            return Err(Error::new("selp predicate destination is unsupported"));
                        }
                    };
                    let a = self.integer(a, value_kind)?;
                    let b = self.integer(b, value_kind)?;
                    let ty = self.value_type(value_kind);
                    let id = self.builder.select(ty, None, cond, a, b)?;
                    (dst, id, value_kind)
                }
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
            Instruction::NotPred(dst, src) => {
                let src = self.read(src)?.require(Kind::Predicate, src)?;
                let id = self.builder.logical_not(self.bool_type, None, src)?;
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
                    IntegerOp::DivUnsigned => self.builder.u_div(ty, None, a, b),
                    IntegerOp::Rem => self.builder.s_rem(ty, None, a, b),
                    IntegerOp::RemUnsigned => self.builder.u_mod(ty, None, a, b),
                    IntegerOp::SignExtend8 => {
                        // PTX cvt.s32.s8 sign-extends the low 8 bits of the source.
                        let shift = self.builder.constant_bit32(ty, 24);
                        let raised = self.builder.shift_left_logical(ty, None, a, shift)?;
                        self.builder.shift_right_arithmetic(ty, None, raised, shift)
                    }
                    IntegerOp::ToU16 => {
                        let mask = self.builder.constant_bit32(ty, 0xffff);
                        self.builder.bitwise_and(ty, None, a, mask)
                    }
                    IntegerOp::SignExtend8To16 => {
                        // Low 8 bits → signed 16-bit value (still held in a B32 slot).
                        let shift = self.builder.constant_bit32(ty, 24);
                        let raised = self.builder.shift_left_logical(ty, None, a, shift)?;
                        let sext32 =
                            self.builder
                                .shift_right_arithmetic(ty, None, raised, shift)?;
                        let mask = self.builder.constant_bit32(ty, 0xffff);
                        self.builder.bitwise_and(ty, None, sext32, mask)
                    }
                    IntegerOp::ShiftRightArithmetic16 => {
                        // Operate on the low 16 bits; keep the result as a sign-extended s16
                        // in a B32 slot (high bits clear after mask of arithmetic result).
                        let shift16 = self.builder.constant_bit32(ty, 16);
                        let raised = self.builder.shift_left_logical(ty, None, a, shift16)?;
                        let as_s32 =
                            self.builder
                                .shift_right_arithmetic(ty, None, raised, shift16)?;
                        let bits = self.builder.constant_bit32(ty, 16);
                        let max = self.builder.constant_bit32(ty, 15);
                        let in_range =
                            self.builder.u_less_than(self.bool_type, None, b, bits)?;
                        // Counts of 16 or more fill with the sign bit (clamp to 15).
                        let safe = self.builder.select(ty, None, in_range, b, max)?;
                        let masked_shift =
                            self.builder
                                .shift_right_arithmetic(ty, None, as_s32, safe)?;
                        // Re-pack as s16 in the low half for subsequent cvt.rn.f32.s16.
                        let raised = self
                            .builder
                            .shift_left_logical(ty, None, masked_shift, shift16)?;
                        let sext =
                            self.builder
                                .shift_right_arithmetic(ty, None, raised, shift16)?;
                        let mask = self.builder.constant_bit32(ty, 0xffff);
                        self.builder.bitwise_and(ty, None, sext, mask)
                    }
                    IntegerOp::ShiftLeft
                    | IntegerOp::ShiftRightLogical
                    | IntegerOp::ShiftRightArithmetic => {
                        // PTX clamps at 32, but SPIR-V shifts at or above 32 are undefined.
                        let bits = self.builder.constant_bit32(ty, 32);
                        let zero = self.builder.constant_bit32(ty, 0);
                        let in_range = self.builder.u_less_than(self.bool_type, None, b, bits)?;
                        if matches!(op, IntegerOp::ShiftRightArithmetic) {
                            // `shr.s32` by 32 or more fills with the sign bit, so
                            // clamp the count to 31 instead of producing zero.
                            let max = self.builder.constant_bit32(ty, 31);
                            let safe = self.builder.select(ty, None, in_range, b, max)?;
                            self.builder.shift_right_arithmetic(ty, None, a, safe)
                        } else {
                            let safe = self.builder.select(ty, None, in_range, b, zero)?;
                            let shifted = if matches!(op, IntegerOp::ShiftLeft) {
                                self.builder.shift_left_logical(ty, None, a, safe)?
                            } else {
                                self.builder.shift_right_logical(ty, None, a, safe)?
                            };
                            self.builder.select(ty, None, in_range, shifted, zero)
                        }
                    }
                    IntegerOp::MulHighUnsigned => self.builder.ext_inst(
                        ty,
                        None,
                        self.opencl,
                        OpenclStd100Op::u_mul_hi as u32,
                        [a, b].map(Operand::IdRef),
                    ),
                    IntegerOp::Xor => self.builder.bitwise_xor(ty, None, a, b),
                    IntegerOp::MinSigned => self.opencl_int(ty, OpenclStd100Op::s_min, &[a, b]),
                    IntegerOp::MaxSigned => self.opencl_int(ty, OpenclStd100Op::s_max, &[a, b]),
                    IntegerOp::MinUnsigned => self.opencl_int(ty, OpenclStd100Op::u_min, &[a, b]),
                    IntegerOp::MaxUnsigned => self.opencl_int(ty, OpenclStd100Op::u_max, &[a, b]),
                    IntegerOp::MulHighSigned => {
                        self.opencl_int(ty, OpenclStd100Op::s_mul_hi, &[a, b])
                    }
                    IntegerOp::AbsSigned => self.opencl_int(ty, OpenclStd100Op::s_abs, &[a]),
                    IntegerOp::LeadingZeros => self.opencl_int(ty, OpenclStd100Op::clz, &[a]),
                    IntegerOp::PopCount => self.builder.bit_count(ty, None, a),
                    IntegerOp::SignExtend16 => {
                        let shift = self.builder.constant_bit32(ty, 16);
                        let raised = self.builder.shift_left_logical(ty, None, a, shift)?;
                        self.builder.shift_right_arithmetic(ty, None, raised, shift)
                    }
                    IntegerOp::ToU8 => {
                        let mask = self.builder.constant_bit32(ty, 0xff);
                        self.builder.bitwise_and(ty, None, a, mask)
                    }
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
            Instruction::Not64(dst, src) => {
                let src = self.integer(src, Kind::B64)?;
                (
                    dst,
                    self.builder.not(self.u64_type, None, src)?,
                    Kind::B64,
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
            Instruction::Move64(dst, src) => {
                // Preserve pointer typing across `mov.u64 %rdD, %rdS`.
                if src.starts_with('%') {
                    let value = self.read(src)?;
                    match value.kind {
                        Kind::Address(_) | Kind::B64 => (dst, value.id, value.kind),
                        other => {
                            return Err(Error::new(format!(
                                "expected B64 or address for {src}, got {other:?}"
                            )));
                        }
                    }
                } else {
                    let id = self.integer(src, Kind::B64)?;
                    (dst, id, Kind::B64)
                }
            }
            Instruction::PackB16(dst, [lo, hi]) => {
                let lo = self.integer(lo, Kind::B32)?;
                let hi = self.integer(hi, Kind::B32)?;
                let mask = self.builder.constant_bit32(self.u32_type, 0xffff);
                let lo = self.builder.bitwise_and(self.u32_type, None, lo, mask)?;
                let hi = self.builder.bitwise_and(self.u32_type, None, hi, mask)?;
                let shift = self.builder.constant_bit32(self.u32_type, 16);
                let hi = self
                    .builder
                    .shift_left_logical(self.u32_type, None, hi, shift)?;
                (
                    dst,
                    self.builder.bitwise_or(self.u32_type, None, lo, hi)?,
                    Kind::B32,
                )
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
                // nvcc may feed a .b64 register to cvt.*.s32/u32, meaning "low 32 bits".
                let src32 = if let Ok(value) = self.read(src) {
                    match value.kind {
                        Kind::B32 => value.id,
                        Kind::B64 => self.builder.u_convert(self.u32_type, None, value.id)?,
                        other => {
                            return Err(Error::new(format!(
                                "expected B32 for {src}, got {other:?}"
                            )));
                        }
                    }
                } else {
                    self.integer(src, Kind::B32)?
                };
                let id = if signed {
                    self.builder.s_convert(self.u64_type, None, src32)?
                } else {
                    self.builder.u_convert(self.u64_type, None, src32)?
                };
                (dst, id, Kind::B64)
            }
            Instruction::Narrow(dst, src) => {
                let src = self.integer(src, Kind::B64)?;
                let id = self.builder.u_convert(self.u32_type, None, src)?;
                (dst, id, Kind::B32)
            }
            Instruction::Add64(dst, [a, b]) => {
                let value = self.add64(a, b)?;
                (dst, value.id, value.kind)
            }
            Instruction::Sub64(dst, [a, b]) => {
                let a = self.integer(a, Kind::B64)?;
                let b = self.integer(b, Kind::B64)?;
                (
                    dst,
                    self.builder.i_sub(self.u64_type, None, a, b)?,
                    Kind::B64,
                )
            }
            Instruction::ShiftRightArithmetic64(dst, [value, shift]) => {
                // SPIR-V requires the shift operand bitwidth to match the base
                // (u64). Passing a u32 immediate made IGC miscompile GGML's
                // signed `/32` idiom (`shr.s64`/`shr.u64`) and OOB-fault
                // quantize_q8_1 stores.
                let value = self.integer(value, Kind::B64)?;
                let shift = self.integer(shift, Kind::B64)?;
                let bits = self.builder.constant_bit64(self.u64_type, 64);
                let max = self.builder.constant_bit64(self.u64_type, 63);
                let in_range = self
                    .builder
                    .u_less_than(self.bool_type, None, shift, bits)?;
                // Counts of 64 or more fill with the sign bit (clamp to 63).
                let safe = self
                    .builder
                    .select(self.u64_type, None, in_range, shift, max)?;
                let id = self
                    .builder
                    .shift_right_arithmetic(self.u64_type, None, value, safe)?;
                (dst, id, Kind::B64)
            }
            Instruction::ShiftLeft64(dst, [value, shift]) => {
                if !shift.parse::<u32>().is_ok_and(|n| n < 64) {
                    return Err(Error::new(
                        "only immediate PTX left shifts below 64 are supported",
                    ));
                }
                let value = self.integer(value, Kind::B64)?;
                let shift = self.integer(shift, Kind::B64)?;
                let id = self
                    .builder
                    .shift_left_logical(self.u64_type, None, value, shift)?;
                (dst, id, Kind::B64)
            }
            Instruction::ShiftRightLogical64(dst, [value, shift]) => {
                let value = self.integer(value, Kind::B64)?;
                let shift = self.integer(shift, Kind::B64)?;
                let bits = self.builder.constant_bit64(self.u64_type, 64);
                let zero = self.builder.constant_bit64(self.u64_type, 0);
                let in_range = self
                    .builder
                    .u_less_than(self.bool_type, None, shift, bits)?;
                let safe = self
                    .builder
                    .select(self.u64_type, None, in_range, shift, zero)?;
                let shifted = self
                    .builder
                    .shift_right_logical(self.u64_type, None, value, safe)?;
                let id = self
                    .builder
                    .select(self.u64_type, None, in_range, shifted, zero)?;
                (dst, id, Kind::B64)
            }
            Instruction::And64(dst, [a, b]) => {
                let a = self.integer(a, Kind::B64)?;
                let b = self.integer(b, Kind::B64)?;
                let id = self.builder.bitwise_and(self.u64_type, None, a, b)?;
                (dst, id, Kind::B64)
            }
            Instruction::Or64(dst, [a, b]) => {
                let a = self.integer(a, Kind::B64)?;
                let b = self.integer(b, Kind::B64)?;
                let id = self.builder.bitwise_or(self.u64_type, None, a, b)?;
                (dst, id, Kind::B64)
            }
            Instruction::MinS64(dst, [a, b]) => {
                let a = self.integer(a, Kind::B64)?;
                let b = self.integer(b, Kind::B64)?;
                let less = self.builder.s_less_than(self.bool_type, None, a, b)?;
                let id = self.builder.select(self.u64_type, None, less, a, b)?;
                (dst, id, Kind::B64)
            }
            Instruction::Div64 {
                signed,
                dst,
                src: [a, b],
            } => {
                let a = self.integer(a, Kind::B64)?;
                let b = self.integer(b, Kind::B64)?;
                let id = if signed {
                    self.builder.s_div(self.u64_type, None, a, b)?
                } else {
                    self.builder.u_div(self.u64_type, None, a, b)?
                };
                (dst, id, Kind::B64)
            }
            Instruction::Rem64 {
                signed,
                dst,
                src: [a, b],
            } => {
                let a = self.integer(a, Kind::B64)?;
                let b = self.integer(b, Kind::B64)?;
                let id = if signed {
                    self.builder.s_rem(self.u64_type, None, a, b)?
                } else {
                    self.builder.u_mod(self.u64_type, None, a, b)?
                };
                (dst, id, Kind::B64)
            }
            Instruction::Load64(space, dst, address) => {
                let bytes_ptr = self.addressed_byte_pointer(space, address)?;
                let qword_ptr = self.typed_pointer(space, self.u64_type);
                let pointer = self.builder.bitcast(qword_ptr, None, bytes_ptr)?;
                let id = self
                    .builder
                    .load(self.u64_type, None, pointer, None, [])?;
                (dst, id, Kind::B64)
            }
            Instruction::Load32(space, dst, address) => {
                let id = if space == Space::Global {
                    // PTX requires natural alignment, so the common case is one aligned
                    // 32-bit load (no tearing against atomics or flag polling). ggml also
                    // reads q8_0 words at +2 offsets, so tolerate misalignment by combining
                    // the two aligned words that cover the access; for an aligned address
                    // both loads hit the same word, so nothing out of bounds is touched.
                    let bytes = self.addressed_byte_pointer(space, address)?;
                    let addr = self.builder.convert_ptr_to_u(self.u64_type, None, bytes)?;
                    let c = |s: &mut Self, v: u64| s.builder.constant_bit64(s.u64_type, v);
                    let (c3, cnot3) = (c(self, 3), c(self, !3u64));
                    let low_bits = self.builder.bitwise_and(self.u64_type, None, addr, c3)?;
                    let lo_addr = self.builder.bitwise_and(self.u64_type, None, addr, cnot3)?;
                    let up = self.builder.i_add(self.u64_type, None, addr, c3)?;
                    let hi_addr = self.builder.bitwise_and(self.u64_type, None, up, cnot3)?;
                    let mut word = |s: &mut Self, a: Word| -> Result<Word, Error> {
                        let p = s.builder.convert_u_to_ptr(s.global_word_ptr, None, a)?;
                        Ok(s.builder.load(s.u32_type, None, p, None, [])?)
                    };
                    let lo = word(self, lo_addr)?;
                    let hi = word(self, hi_addr)?;
                    let low_bits32 = self.builder.u_convert(self.u32_type, None, low_bits)?;
                    let eight = self.builder.constant_bit32(self.u32_type, 3);
                    let shift = self
                        .builder
                        .shift_left_logical(self.u32_type, None, low_bits32, eight)?;
                    let zero = self.builder.constant_bit32(self.u32_type, 0);
                    let thirty_two = self.builder.constant_bit32(self.u32_type, 32);
                    let inv = self.builder.i_sub(self.u32_type, None, thirty_two, shift)?;
                    let from_lo = self
                        .builder
                        .shift_right_logical(self.u32_type, None, lo, shift)?;
                    let c31 = self.builder.constant_bit32(self.u32_type, 31);
                    let inv_masked = self.builder.bitwise_and(
                        self.u32_type,
                        None,
                        inv,
                        c31,
                    )?;
                    let from_hi = self
                        .builder
                        .shift_left_logical(self.u32_type, None, hi, inv_masked)?;
                    let aligned = self.builder.i_equal(self.bool_type, None, shift, zero)?;
                    let from_hi = self
                        .builder
                        .select(self.u32_type, None, aligned, zero, from_hi)?;
                    self.builder.bitwise_or(self.u32_type, None, from_lo, from_hi)?
                } else {
                    let pointer = self.word_pointer(space, address)?;
                    self.builder.load(self.u32_type, None, pointer, None, [])?
                };
                (dst, id, Kind::B32)
            }
            Instruction::Load16(space, dst, address) => {
                let pointer = self.half_pointer(space, address)?;
                let bits16 = self
                    .builder
                    .load(self.u16_type, None, pointer, None, [])?;
                let bits32 = self.builder.u_convert(self.u32_type, None, bits16)?;
                (dst, bits32, Kind::B32)
            }
            Instruction::Load8(space, dst, address) => {
                let bytes = self.addressed_byte_pointer(space, address)?;
                let bits8 = self.builder.load(self.u8_type, None, bytes, None, [])?;
                let bits32 = self.builder.u_convert(self.u32_type, None, bits8)?;
                (dst, bits32, Kind::B32)
            }
            Instruction::LoadVecU16 {
                space,
                count,
                dst,
                addr,
            } => {
                let (base, base_off) = match addr.split_once('+') {
                    Some((base, offset)) => (
                        base,
                        Self::parse_byte_offset(offset).ok_or_else(|| {
                            Error::new(format!("unsupported PTX address offset {offset}"))
                        })?,
                    ),
                    None => (addr, 0i64),
                };
                for index in 0..usize::from(count) {
                    let off = base_off + (index as i64) * 2;
                    let lane_addr = if off == 0 {
                        base.to_string()
                    } else {
                        format!("{base}+{off}")
                    };
                    let pointer = self.half_pointer(space, &lane_addr)?;
                    let bits16 = self
                        .builder
                        .load(self.u16_type, None, pointer, None, [])?;
                    let bits32 = self.builder.u_convert(self.u32_type, None, bits16)?;
                    self.write(
                        dst[index],
                        Value {
                            id: bits32,
                            kind: Kind::B32,
                        },
                    )?;
                }
                return Ok(());
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
            Instruction::CvtRziS32(dst, src) => {
                let value = self.float_value(src)?;
                let converted = self.builder.convert_f_to_s(self.u32_type, None, value)?;
                self.builder.decorate(
                    converted,
                    Decoration::FPRoundingMode,
                    [Operand::FPRoundingMode(FPRoundingMode::RTZ)],
                );
                // PTX float-to-int conversion saturates and maps NaN to zero.
                self.builder
                    .decorate(converted, Decoration::SaturatedConversion, []);
                (dst, converted, Kind::B32)
            }
            Instruction::CvtRniS32(dst, src) => {
                let value = self.float_value(src)?;
                let rounded = self.builder.ext_inst(
                    self.f32_type,
                    None,
                    self.opencl,
                    OpenclStd100Op::rint as u32,
                    [Operand::IdRef(value)],
                )?;
                let converted = self.builder.convert_f_to_s(self.u32_type, None, rounded)?;
                self.builder.decorate(
                    converted,
                    Decoration::FPRoundingMode,
                    [Operand::FPRoundingMode(FPRoundingMode::RTZ)],
                );
                self.fp32 = true;
                (dst, converted, Kind::B32)
            }
            Instruction::CvtToU32 { nearest, dst, src } => {
                let mut value = self.float_value(src)?;
                if nearest {
                    value = self.builder.ext_inst(
                        self.f32_type,
                        None,
                        self.opencl,
                        OpenclStd100Op::rint as u32,
                        [Operand::IdRef(value)],
                    )?;
                }
                // PTX saturates: negatives (and NaN) give 0.
                let zero = self.builder.constant_bit32(self.f32_type, 0);
                let negative = self.builder.f_ord_less_than(self.bool_type, None, value, zero)?;
                let clamped = self.builder.select(self.f32_type, None, negative, zero, value)?;
                let converted = self.builder.convert_f_to_u(self.u32_type, None, clamped)?;
                self.builder.decorate(
                    converted,
                    Decoration::FPRoundingMode,
                    [Operand::FPRoundingMode(FPRoundingMode::RTZ)],
                );
                self.fp32 = true;
                (dst, converted, Kind::B32)
            }
            Instruction::CvtToHalf(dst, src, toward_zero) => {
                let value = self.float_value(src)?;
                let half = self.builder.f_convert(self.f16_type, None, value)?;
                let mode = if toward_zero {
                    FPRoundingMode::RTZ
                } else {
                    FPRoundingMode::RTE
                };
                self.builder.decorate(
                    half,
                    Decoration::FPRoundingMode,
                    [Operand::FPRoundingMode(mode)],
                );
                let bits16 = self.builder.bitcast(self.u16_type, None, half)?;
                let bits32 = self.builder.u_convert(self.u32_type, None, bits16)?;
                self.fp32 = true;
                (dst, bits32, Kind::B32)
            }
            Instruction::Store32(space, address, src) => {
                let pointer = self.word_pointer(space, address)?;
                let value = self.read(src)?.require(Kind::B32, src)?;
                self.builder.store(pointer, value, None, [])?;
                return Ok(());
            }
            Instruction::Store8(space, address, src) => {
                let bytes = self.addressed_byte_pointer(space, address)?;
                let value = self.integer(src, Kind::B32)?;
                let byte = self.builder.u_convert(self.u8_type, None, value)?;
                self.builder.store(bytes, byte, None, [])?;
                return Ok(());
            }
            Instruction::Store16(space, address, src) => {
                let half_ptr = self.half_pointer(space, address)?;
                let value = self.integer(src, Kind::B32)?;
                let bits16 = self.builder.u_convert(self.u16_type, None, value)?;
                self.builder.store(half_ptr, bits16, None, [])?;
                return Ok(());
            }
            Instruction::Store64(space, address, src) => {
                let bytes_ptr = self.addressed_byte_pointer(space, address)?;
                let qword_ptr = self.typed_pointer(space, self.u64_type);
                let pointer = self.builder.bitcast(qword_ptr, None, bytes_ptr)?;
                let value = self.integer(src, Kind::B64)?;
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
            Instruction::StoreVecU16(space, addr, [a, b]) => {
                let lo = self.integer(a, Kind::B32)?;
                let hi = self.integer(b, Kind::B32)?;
                let mask = self.builder.constant_bit32(self.u32_type, 0xffff);
                let lo = self.builder.bitwise_and(self.u32_type, None, lo, mask)?;
                let hi = self.builder.bitwise_and(self.u32_type, None, hi, mask)?;
                let shift = self.builder.constant_bit32(self.u32_type, 16);
                let hi = self
                    .builder
                    .shift_left_logical(self.u32_type, None, hi, shift)?;
                let packed = self.builder.bitwise_or(self.u32_type, None, lo, hi)?;
                let pointer = self.word_pointer(space, addr)?;
                self.builder.store(pointer, packed, None, [])?;
                return Ok(());
            }
            Instruction::StoreVec4U8(addr, bytes) => {
                let ty = self.u32_type;
                let mask = self.builder.constant_bit32(ty, 0xff);
                let mut packed = self.builder.constant_bit32(ty, 0);
                for (index, byte) in bytes.into_iter().enumerate() {
                    let byte = self.integer(byte, Kind::B32)?;
                    let byte = self.builder.bitwise_and(ty, None, byte, mask)?;
                    let shift = self.builder.constant_bit32(ty, index as u32 * 8);
                    let byte = self.builder.shift_left_logical(ty, None, byte, shift)?;
                    packed = self.builder.bitwise_or(ty, None, packed, byte)?;
                }
                let pointer = self.word_pointer(Space::Global, addr)?;
                self.builder.store(pointer, packed, None, [])?;
                return Ok(());
            }
            Instruction::StoreVec4U16(space, addr, [a, b, c, d]) => {
                // Two packed u32 stores: [a|b] at addr, [c|d] at addr+4.
                let mut pack2 = |lo: &str, hi: &str| -> Result<Word, Error> {
                    let lo = self.integer(lo, Kind::B32)?;
                    let hi = self.integer(hi, Kind::B32)?;
                    let mask = self.builder.constant_bit32(self.u32_type, 0xffff);
                    let lo = self.builder.bitwise_and(self.u32_type, None, lo, mask)?;
                    let hi = self.builder.bitwise_and(self.u32_type, None, hi, mask)?;
                    let shift = self.builder.constant_bit32(self.u32_type, 16);
                    let hi = self
                        .builder
                        .shift_left_logical(self.u32_type, None, hi, shift)?;
                    Ok(self.builder.bitwise_or(self.u32_type, None, lo, hi)?)
                };
                let lo_word = pack2(a, b)?;
                let hi_word = pack2(c, d)?;
                let lo_ptr = self.word_pointer(space, addr)?;
                self.builder.store(lo_ptr, lo_word, None, [])?;
                let hi_addr = if let Some((base, off)) = addr.split_once('+') {
                    let base_off = Self::parse_byte_offset(off).unwrap_or(0);
                    format!("{base}+{}", base_off + 4)
                } else {
                    format!("{addr}+4")
                };
                let hi_ptr = self.word_pointer(space, &hi_addr)?;
                self.builder.store(hi_ptr, hi_word, None, [])?;
                return Ok(());
            }
            Instruction::ReduxSyncAdd {
                dst,
                src,
                membermask,
            } => {
                self.emit_redux_add(dst, src, membermask)?;
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

impl<'a> Emitter<'a> {
    /// Emit every module-scope `.global`/`.const` variable as an exported
    /// `CrossWorkgroup` variable. Level Zero resolves the exported name with
    /// `zeModuleGetGlobalPointer`.
    fn declare_globals(&mut self, vars: &[GlobalVar]) {
        if vars.is_empty() {
            return;
        }
        self.builder.capability(Capability::Linkage);
        let mut words: HashMap<u32, Word> = HashMap::new();
        for var in vars {
            let count = var.size.div_ceil(4);
            let length = self.builder.constant_bit32(self.u32_type, count);
            let array = self.builder.type_array(self.u32_type, length);
            let pointer = self
                .builder
                .type_pointer(None, StorageClass::CrossWorkgroup, array);
            let initializer = match &var.init {
                None => self.builder.constant_null(array),
                Some(bytes) => {
                    let mut elements = Vec::with_capacity(count as usize);
                    for chunk in bytes.chunks(4) {
                        let mut word = [0u8; 4];
                        word[..chunk.len()].copy_from_slice(chunk);
                        let value = u32::from_le_bytes(word);
                        let builder = &mut self.builder;
                        let u32_type = self.u32_type;
                        let id = *words
                            .entry(value)
                            .or_insert_with(|| builder.constant_bit32(u32_type, value));
                        elements.push(id);
                    }
                    self.builder.constant_composite(array, elements)
                }
            };
            let variable = self.builder.variable(
                pointer,
                None,
                StorageClass::CrossWorkgroup,
                Some(initializer),
            );
            self.builder.decorate(
                variable,
                Decoration::Alignment,
                [Operand::LiteralBit32(var.align.max(4))],
            );
            self.builder.decorate(
                variable,
                Decoration::LinkageAttributes,
                [
                    Operand::LiteralString(var.name.clone()),
                    Operand::LinkageType(LinkageType::Export),
                ],
            );
            self.globals.insert(var.name.clone(), variable);
        }
    }
}

pub(super) fn emit(
    entries: Vec<Entry<'_>>,
    globals: Vec<GlobalVar>,
) -> Result<Translation, Error> {
    let mut emitter = Emitter::new();
    emitter.declare_globals(&globals);
    let kernels = entries
        .into_iter()
        .map(|entry| emitter.emit_entry(entry))
        .collect::<Result<_, _>>()?;
    Ok(Translation {
        words: emitter.builder.module().assemble(),
        kernels,
        globals,
    })
}

impl<'a> Emitter<'a> {
    fn emit_entry(&mut self, entry: Entry<'a>) -> Result<KernelMetadata, Error> {
        let flow = Flow::analyze(&entry.blocks)?;
        self.declarations = entry.registers;
        self.registers.clear();
        self.initialized.clear();
        self.params.clear();
        self.local = None;
        self.device_atomics = false;
        self.fp32 = false;
        self.fp32_fma = false;
        self.warp_ops = entry.blocks.iter().any(|block| {
            block.instructions.iter().any(|inst| {
                matches!(
                    inst,
                    Instruction::ShuffleBfly { .. }
                        | Instruction::ReduxSyncAdd { .. }
                        | Instruction::Mma { .. }
                        | Instruction::LdMatrix { .. }
                        | Instruction::LaneId(..)
                        | Instruction::Shuffle { .. }
                        | Instruction::Vote { .. }
                ) || matches!(inst, Instruction::Ext { op, .. } if op.uses_subgroup())
            })
        });
        if self.warp_ops {
            // GroupNonUniform requires SPIR-V 1.3; force 32-lane subgroups so CUDA
            // warp shuffles map 1:1 onto Intel subgroup shuffles.
            self.builder.set_version(1, 3);
            self.builder.capability(Capability::GroupNonUniform);
            self.builder
                .capability(Capability::GroupNonUniformShuffle);
        }
        self.shared_offsets.clear();
        self.shared_array = None;
        let shared_total = entry
            .shared
            .iter()
            .try_fold(0u32, |total, &(_, bytes)| total.checked_add(bytes))
            .ok_or_else(|| Error::new("PTX shared memory size overflows 32 bits"))?;
        if shared_total != 0 {
            let mut offset = 0u32;
            for &(name, bytes) in &entry.shared {
                self.shared_offsets.insert(name, offset);
                offset = offset.saturating_add(bytes);
            }
            // One workgroup array for the whole shared window (multi-symbol offsets).
            let length = self.builder.constant_bit32(self.u32_type, shared_total / 4);
            let array = self.builder.type_array(self.u32_type, length);
            let pointer = self
                .builder
                .type_pointer(None, StorageClass::Workgroup, array);
            let variable = self
                .builder
                .variable(pointer, None, StorageClass::Workgroup, None);
            self.shared_array = Some(variable);
        }
        // CUDA `.local` depot: Function-scope private array (spliced into block 0).
        self.local = entry.local.map(|(name, bytes)| {
            let words = bytes.div_ceil(4).max(1);
            let length = self.builder.constant_bit32(self.u32_type, words);
            let array = self.builder.type_array(self.u32_type, length);
            let pointer = self
                .builder
                .type_pointer(None, StorageClass::Function, array);
            let variable = self.builder.id();
            self.locals.push(SpirvInstruction::new(
                Op::Variable,
                Some(pointer),
                Some(variable),
                vec![Operand::StorageClass(StorageClass::Function)],
            ));
            (name, variable)
        });
        let types: Vec<_> = entry
            .params
            .iter()
            .flat_map(|&(_, param)| {
                let count = match param {
                    Param::Words { words, .. } => usize::from(words),
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
        if self.warp_ops {
            self.builder
                .execution_mode(function, ExecutionMode::SubgroupSize, [32]);
        }
        let values = types
            .into_iter()
            .map(|ty| self.builder.function_parameter(ty))
            .collect::<Result<Vec<_>, _>>()?;
        self.builder.begin_block(None)?;
        let mut values = values.into_iter();
        for &(name, param) in &entry.params {
            // Keep one CUDA argument, but pass each aligned word to Level Zero
            // as a scalar so the device ABI needs no by-value struct pointer.
            if let Param::Words { words, .. } = param {
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
                Param::U8 => {
                    // Promote byte params to B32 slots; ld.param.s8 sign-extends later.
                    id = self.builder.u_convert(self.u32_type, None, id)?;
                    Kind::B32
                }
                Param::U16 => {
                    id = self.builder.u_convert(self.u32_type, None, id)?;
                    Kind::B32
                }
                Param::U32 => Kind::B32,
                Param::F32 => {
                    id = self.builder.bitcast(self.u32_type, None, id)?;
                    Kind::B32
                }
                Param::U64 => Kind::B64,
                Param::GlobalPtr => Kind::Address(Space::Global),
                Param::Words { .. } => unreachable!(),
            };
            self.params.insert((name, 0), Value { id, kind });
        }
        self.ever_defined.clear();
        let mut scratch = Vec::new();
        for block in &entry.blocks {
            for inst in &block.instructions {
                if inst.collect_definitions(&mut scratch).is_some() {
                    self.ever_defined.extend(scratch.iter().copied());
                }
            }
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
            shared_bytes: shared_total,
            max_threads: entry.max_threads,
            device_atomics: self.device_atomics,
            fp32: self.fp32,
            fp32_fma: self.fp32_fma,
        })
    }
}
