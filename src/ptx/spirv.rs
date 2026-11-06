use super::{Entry, Error, Instruction, Param, Space, Translation, Width};
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionModel, FunctionControl, MemoryModel,
    MemorySemantics, Scope, StorageClass, Word,
};
use std::collections::HashMap;

const BUILTINS: [BuiltIn; 3] = [
    BuiltIn::WorkgroupId,
    BuiltIn::WorkgroupSize,
    BuiltIn::LocalInvocationId,
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
    builtins: [Word; 3],
    declarations: HashMap<String, Width>,
    registers: HashMap<&'a str, Value>,
    params: HashMap<&'a str, Value>,
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
            params: HashMap::new(),
        }
    }

    fn param_type(&self, param: Param) -> Word {
        match param {
            Param::U32 => self.u32_type,
            Param::U64 => self.u64_type,
            Param::GlobalPtr => self.global_byte_ptr,
        }
    }

    fn read(&self, operand: &str) -> Result<Value, Error> {
        self.registers
            .get(operand)
            .copied()
            .ok_or_else(|| Error::new(format!("undefined PTX register {operand}")))
    }

    fn write(&mut self, dst: &'a str, value: Value) -> Result<(), Error> {
        if self.declarations.get(dst) != Some(&value.kind.width()) {
            return Err(Error::new(format!(
                "missing or mismatched PTX declaration for {dst}"
            )));
        }
        self.registers.insert(dst, value);
        Ok(())
    }

    fn integer(&mut self, operand: &str, kind: Kind) -> Result<Word, Error> {
        if operand.starts_with('%') {
            return self.read(operand)?.require(kind, operand);
        }
        match kind {
            Kind::B32 => operand
                .parse()
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
        let (base, offset) = match (address(a), address(b)) {
            (Some(base), None) => (base, b),
            (None, Some(base)) => (base, a),
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
        let pointer_type = match base.kind {
            Kind::Address(Space::Shared) => self.shared_byte_ptr,
            _ => self.global_byte_ptr,
        };
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
            Instruction::Ge32(dst, [a, b]) => {
                let a = self.integer(a, Kind::B32)?;
                let b = self.integer(b, Kind::B32)?;
                (
                    dst,
                    self.builder
                        .u_greater_than_equal(self.bool_type, None, a, b)?,
                    Kind::Predicate,
                )
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
            Instruction::Load32(space, dst, address) => {
                let pointer = self.word_pointer(space, address)?;
                (
                    dst,
                    self.builder.load(self.u32_type, None, pointer, None, [])?,
                    Kind::B32,
                )
            }
            Instruction::AddFloat32(dst, [a, b]) => {
                let a = self.read(a)?.require(Kind::B32, a)?;
                let b = self.read(b)?.require(Kind::B32, b)?;
                let a = self.builder.bitcast(self.f32_type, None, a)?;
                let b = self.builder.bitcast(self.f32_type, None, b)?;
                let sum = self.builder.f_add(self.f32_type, None, a, b)?;
                (
                    dst,
                    self.builder.bitcast(self.u32_type, None, sum)?,
                    Kind::B32,
                )
            }
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
        };
        self.write(dst, Value { id, kind })
    }
}

pub(super) fn emit(entry: Entry<'_>) -> Result<Translation, Error> {
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
    for ((name, param), ty) in entry.params.iter().zip(types) {
        let kind = match param {
            Param::U32 => Kind::B32,
            Param::U64 => Kind::B64,
            Param::GlobalPtr => Kind::Address(Space::Global),
        };
        let id = emitter.builder.function_parameter(ty)?;
        emitter.params.insert(name, Value { id, kind });
    }
    emitter.builder.begin_block(None)?;
    for inst in entry.prefix {
        emitter.instruction(inst)?;
    }
    if let Some((predicate, instructions)) = entry.guarded {
        let condition = emitter
            .read(predicate)?
            .require(Kind::Predicate, predicate)?;
        let body = emitter.builder.id();
        let done = emitter.builder.id();
        emitter
            .builder
            .branch_conditional(condition, done, body, [])?;
        emitter.builder.begin_block(Some(body))?;
        for inst in instructions {
            emitter.instruction(inst)?;
        }
        emitter.builder.branch(done)?;
        emitter.builder.begin_block(Some(done))?;
    }
    emitter.builder.ret()?;
    emitter.builder.end_function()?;
    Ok(Translation {
        name: entry.name.to_owned(),
        params: entry.params.into_iter().map(|(_, param)| param).collect(),
        words: emitter.builder.module().assemble(),
        shared_bytes: entry.shared.map_or(0, |(_, bytes)| bytes),
    })
}
