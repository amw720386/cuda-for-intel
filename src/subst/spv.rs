//! A small structured SPIR-V (OpenCL environment, Physical64) kernel builder used by the native
//! kernel substitutions. Locals are Function variables (spliced into the first block when the
//! module is finished), control flow is structured, and all addresses are plain `u64` integers
//! converted to CrossWorkgroup pointers at each access, the same way the PTX translator does.
use rspirv::binary::Assemble;
use rspirv::dr::{Builder, Instruction, Operand};
use rspirv::spirv::{
    AddressingModel, BuiltIn, Capability, Decoration, ExecutionMode, ExecutionModel,
    FunctionControl, LoopControl, MemoryAccess, MemoryModel, MemorySemantics, Op,
    OpenclStd100Op, PackedVectorFormat, Scope, SelectionControl, StorageClass, Word,
};

pub(crate) type V = Word;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Ty {
    Bool,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
}

/// A kernel parameter as Level Zero sees it (one scalar per 32-bit word of a CUDA aggregate).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Arg {
    Ptr,
    U32,
    U64,
    F32,
}

#[derive(Clone, Copy)]
pub(crate) struct Var {
    id: Word,
    ty: Ty,
}

pub(crate) struct Kb {
    pub(crate) b: Builder,
    opencl: Word,
    builtins: [Word; 3],
    vec3: Word,
    shared: Option<(Word, Word)>,
    vars: Vec<Instruction>,
    terminated: bool,
    dot: bool,
    pub(crate) params: Vec<V>,
    /// A runtime zero the compiler cannot fold (see `opaque_const`).
    pub(crate) opaque_zero: Option<V>,
}

fn scope(s: Scope) -> Scope {
    s
}

impl Kb {
    /// `shared_words` is the size in 32-bit words of the one workgroup array (0 for none);
    /// `subgroup32` forces 32-lane subgroups (needed for shuffles to mirror CUDA warps).
    pub(crate) fn new(name: &str, args: &[Arg], shared_words: u32, subgroup32: bool, dot: bool) -> Self {
        let mut b = Builder::new();
        b.set_version(1, 3);
        for c in [
            Capability::Addresses,
            Capability::Kernel,
            Capability::Int8,
            Capability::Int16,
            Capability::Int64,
            Capability::Float16,
            Capability::GroupNonUniform,
            Capability::GroupNonUniformShuffle,
        ] {
            b.capability(c);
        }
        if dot {
            b.capability(Capability::DotProduct);
            b.capability(Capability::DotProductInput4x8BitPacked);
            b.extension("SPV_KHR_integer_dot_product");
        }
        b.memory_model(AddressingModel::Physical64, MemoryModel::OpenCL);
        let opencl = b.ext_inst_import("OpenCL.std");
        let u64t = b.type_int(64, 0);
        let vec3 = b.type_vector(u64t, 3);
        let vec3_ptr = b.type_pointer(None, StorageClass::Input, vec3);
        let builtins = [BuiltIn::WorkgroupId, BuiltIn::LocalInvocationId, BuiltIn::NumWorkgroups].map(|kind| {
            let id = b.variable(vec3_ptr, None, StorageClass::Input, None);
            b.decorate(id, Decoration::BuiltIn, [Operand::BuiltIn(kind)]);
            id
        });
        let mut kb = Self {
            b,
            opencl,
            builtins,
            vec3,
            shared: None,
            vars: Vec::new(),
            terminated: false,
            dot,
            params: Vec::new(),
            opaque_zero: None,
        };
        if shared_words != 0 {
            let u32t = kb.t(Ty::U32);
            let len = kb.b.constant_bit32(u32t, shared_words);
            let arr = kb.b.type_array(u32t, len);
            let ptr = kb.b.type_pointer(None, StorageClass::Workgroup, arr);
            let var = kb.b.variable(ptr, None, StorageClass::Workgroup, None);
            kb.shared = Some((var, u32t));
        }
        let void = kb.b.type_void();
        let tys: Vec<Word> = args
            .iter()
            .map(|a| match a {
                Arg::Ptr => {
                    let u8t = kb.t(Ty::U8);
                    kb.b.type_pointer(None, StorageClass::CrossWorkgroup, u8t)
                }
                Arg::U32 => kb.t(Ty::U32),
                Arg::U64 => kb.t(Ty::U64),
                Arg::F32 => kb.t(Ty::F32),
            })
            .collect();
        let sig = kb.b.type_function(void, tys.iter().copied());
        let f = kb.b.begin_function(void, None, FunctionControl::NONE, sig).unwrap();
        let mut interface = kb.builtins.to_vec();
        interface.sort_unstable();
        kb.b.entry_point(ExecutionModel::Kernel, f, name, interface);
        if subgroup32 {
            kb.b.execution_mode(f, ExecutionMode::SubgroupSize, [32]);
        }
        kb.b.execution_mode(f, ExecutionMode::ContractionOff, []);
        kb.params = tys.iter().map(|&t| kb.b.function_parameter(t).unwrap()).collect();
        kb.b.begin_block(None).unwrap();
        kb
    }

    pub(crate) fn finish(mut self) -> Vec<u32> {
        if !self.terminated {
            self.b.ret().unwrap();
        }
        self.b.end_function().unwrap();
        let mut module = self.b.module();
        let first = &mut module.functions[0].blocks[0].instructions;
        for (i, v) in self.vars.drain(..).enumerate() {
            first.insert(i, v);
        }
        module.assemble()
    }

    pub(crate) fn t(&mut self, ty: Ty) -> Word {
        match ty {
            Ty::Bool => self.b.type_bool(),
            Ty::U8 => self.b.type_int(8, 0),
            Ty::U16 => self.b.type_int(16, 0),
            Ty::U32 => self.b.type_int(32, 0),
            Ty::U64 => self.b.type_int(64, 0),
            Ty::F16 => self.b.type_float(16, None),
            Ty::F32 => self.b.type_float(32, None),
        }
    }

    // ---- constants ----
    pub(crate) fn c32(&mut self, v: u32) -> V {
        let t = self.t(Ty::U32);
        self.b.constant_bit32(t, v)
    }
    pub(crate) fn c64(&mut self, v: u64) -> V {
        let t = self.t(Ty::U64);
        self.b.constant_bit64(t, v)
    }
    pub(crate) fn cf(&mut self, v: f32) -> V {
        let t = self.t(Ty::F32);
        self.b.constant_bit32(t, v.to_bits())
    }

    // ---- integer ops (u32 unless noted) ----
    pub(crate) fn add(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U32);
        self.b.i_add(t, None, a, b).unwrap()
    }
    pub(crate) fn sub(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U32);
        self.b.i_sub(t, None, a, b).unwrap()
    }
    pub(crate) fn mul(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U32);
        self.b.i_mul(t, None, a, b).unwrap()
    }
    pub(crate) fn udiv(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U32);
        self.b.u_div(t, None, a, b).unwrap()
    }
    pub(crate) fn umod(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U32);
        self.b.u_mod(t, None, a, b).unwrap()
    }
    pub(crate) fn and(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U32);
        self.b.bitwise_and(t, None, a, b).unwrap()
    }
    pub(crate) fn or(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U32);
        self.b.bitwise_or(t, None, a, b).unwrap()
    }
    pub(crate) fn shl(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U32);
        self.b.shift_left_logical(t, None, a, b).unwrap()
    }
    pub(crate) fn shr(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U32);
        self.b.shift_right_logical(t, None, a, b).unwrap()
    }
    pub(crate) fn sar(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U32);
        self.b.shift_right_arithmetic(t, None, a, b).unwrap()
    }
    pub(crate) fn addc(&mut self, a: V, k: u32) -> V {
        let k = self.c32(k);
        self.add(a, k)
    }
    pub(crate) fn mulc(&mut self, a: V, k: u32) -> V {
        let k = self.c32(k);
        self.mul(a, k)
    }
    pub(crate) fn andc(&mut self, a: V, k: u32) -> V {
        let k = self.c32(k);
        self.and(a, k)
    }
    pub(crate) fn shrc(&mut self, a: V, k: u32) -> V {
        let k = self.c32(k);
        self.shr(a, k)
    }
    pub(crate) fn shlc(&mut self, a: V, k: u32) -> V {
        let k = self.c32(k);
        self.shl(a, k)
    }
    pub(crate) fn sarc(&mut self, a: V, k: u32) -> V {
        let k = self.c32(k);
        self.sar(a, k)
    }
    /// Saturating-free per-byte `a - b` (wrapping per lane) of four packed bytes (`__vsubss4`
    /// is only used on values whose differences fit in a signed byte, so wrapping is exact).
    pub(crate) fn vsub4(&mut self, a: V, b: V) -> V {
        // ((a | 0x80808080) - (b & 0x7f7f7f7f)) ^ ((a ^ ~b) & 0x80808080)
        let hi = self.c32(0x8080_8080);
        let lo = self.c32(0x7f7f_7f7f);
        let a_or = self.or(a, hi);
        let b_lo = self.and(b, lo);
        let diff = self.sub(a_or, b_lo);
        let t = self.t(Ty::U32);
        let nb = self.b.not(t, None, b).unwrap();
        let x = self.b.bitwise_xor(t, None, a, nb).unwrap();
        let x = self.and(x, hi);
        self.b.bitwise_xor(t, None, diff, x).unwrap()
    }
    pub(crate) fn shr64c(&mut self, a: V, k: u64) -> V {
        let t = self.t(Ty::U64);
        let k = self.c64(k);
        self.b.shift_right_logical(t, None, a, k).unwrap()
    }
    pub(crate) fn bitcast(&mut self, to: Ty, a: V) -> V {
        let t = self.t(to);
        self.b.bitcast(t, None, a).unwrap()
    }
    pub(crate) fn uconv(&mut self, to: Ty, a: V) -> V {
        let t = self.t(to);
        self.b.u_convert(t, None, a).unwrap()
    }
    pub(crate) fn sconv(&mut self, to: Ty, a: V) -> V {
        let t = self.t(to);
        self.b.s_convert(t, None, a).unwrap()
    }
    pub(crate) fn add64(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U64);
        self.b.i_add(t, None, a, b).unwrap()
    }
    pub(crate) fn mul64(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::U64);
        self.b.i_mul(t, None, a, b).unwrap()
    }
    /// `base + (u32 index) * stride` as a u64 address.
    pub(crate) fn addr(&mut self, base: V, index: V, stride: u32) -> V {
        let i = self.uconv(Ty::U64, index);
        let s = self.c64(u64::from(stride));
        let o = self.mul64(i, s);
        self.add64(base, o)
    }
    pub(crate) fn addr_off(&mut self, base: V, bytes: u32) -> V {
        let o = self.c64(u64::from(bytes));
        self.add64(base, o)
    }
    pub(crate) fn addr_u64off(&mut self, base: V, bytes: V) -> V {
        self.add64(base, bytes)
    }
    pub(crate) fn ptr_to_u64(&mut self, p: V) -> V {
        let t = self.t(Ty::U64);
        self.b.convert_ptr_to_u(t, None, p).unwrap()
    }

    /// `value`, computed so the compiler cannot constant-fold it into a dot-product operand
    /// (hardware showed OpSDot with a folded constant operand giving wrong results on Arc).
    pub(crate) fn opaque_const(&mut self, value: u32) -> V {
        let z = self.opaque_zero.expect("opaque_zero not set");
        let c = self.c32(value);
        self.or(c, z)
    }

    // ---- comparisons / select ----
    pub(crate) fn ult(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::Bool);
        self.b.u_less_than(t, None, a, b).unwrap()
    }
    pub(crate) fn ule(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::Bool);
        self.b.u_less_than_equal(t, None, a, b).unwrap()
    }
    pub(crate) fn ugt(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::Bool);
        self.b.u_greater_than(t, None, a, b).unwrap()
    }
    pub(crate) fn ult64(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::Bool);
        self.b.u_less_than(t, None, a, b).unwrap()
    }
    pub(crate) fn cf_zero(&mut self) -> V {
        self.cf(0.0)
    }
    pub(crate) fn eq(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::Bool);
        self.b.i_equal(t, None, a, b).unwrap()
    }
    pub(crate) fn ne(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::Bool);
        self.b.i_not_equal(t, None, a, b).unwrap()
    }
    pub(crate) fn land(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::Bool);
        self.b.logical_and(t, None, a, b).unwrap()
    }
    pub(crate) fn select(&mut self, ty: Ty, c: V, a: V, b: V) -> V {
        let t = self.t(ty);
        self.b.select(t, None, c, a, b).unwrap()
    }
    pub(crate) fn fogt(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::Bool);
        self.b.f_ord_greater_than(t, None, a, b).unwrap()
    }
    pub(crate) fn foeq(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::Bool);
        self.b.f_ord_equal(t, None, a, b).unwrap()
    }

    // ---- floats ----
    pub(crate) fn fadd(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::F32);
        self.b.f_add(t, None, a, b).unwrap()
    }
    pub(crate) fn fsub(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::F32);
        self.b.f_sub(t, None, a, b).unwrap()
    }
    pub(crate) fn fmul(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::F32);
        self.b.f_mul(t, None, a, b).unwrap()
    }
    pub(crate) fn fdiv(&mut self, a: V, b: V) -> V {
        let t = self.t(Ty::F32);
        self.b.f_div(t, None, a, b).unwrap()
    }
    pub(crate) fn s2f(&mut self, a: V) -> V {
        let t = self.t(Ty::F32);
        self.b.convert_s_to_f(t, None, a).unwrap()
    }
    pub(crate) fn f2s(&mut self, a: V) -> V {
        let t = self.t(Ty::U32);
        self.b.convert_f_to_s(t, None, a).unwrap()
    }
    pub(crate) fn h2f(&mut self, bits_u16: V) -> V {
        let h = self.bitcast(Ty::F16, bits_u16);
        let t = self.t(Ty::F32);
        self.b.f_convert(t, None, h).unwrap()
    }
    /// Round-to-nearest-even float to half, returned as the raw 16 bits.
    pub(crate) fn f2h_bits(&mut self, f: V) -> V {
        let h = self.t(Ty::F16);
        let h = self.b.f_convert(h, None, f).unwrap();
        self.b.decorate(h, Decoration::FPRoundingMode, [Operand::FPRoundingMode(rspirv::spirv::FPRoundingMode::RTE)]);
        self.bitcast(Ty::U16, h)
    }
    fn ext(&mut self, ty: Ty, op: OpenclStd100Op, args: &[V]) -> V {
        let t = self.t(ty);
        self.b
            .ext_inst(t, None, self.opencl, op as u32, args.iter().map(|&a| Operand::IdRef(a)))
            .unwrap()
    }
    pub(crate) fn fabs(&mut self, a: V) -> V {
        self.ext(Ty::F32, OpenclStd100Op::fabs, &[a])
    }
    pub(crate) fn fmax(&mut self, a: V, b: V) -> V {
        self.ext(Ty::F32, OpenclStd100Op::fmax, &[a, b])
    }
    pub(crate) fn exp(&mut self, a: V) -> V {
        self.ext(Ty::F32, OpenclStd100Op::exp, &[a])
    }
    pub(crate) fn round(&mut self, a: V) -> V {
        self.ext(Ty::F32, OpenclStd100Op::round, &[a])
    }
    pub(crate) fn fma(&mut self, a: V, b: V, c: V) -> V {
        self.ext(Ty::F32, OpenclStd100Op::fma, &[a, b, c])
    }

    /// Wrapping four-way byte dot product `acc + dot(a, b)` with both operands signed.
    pub(crate) fn dp4a(&mut self, a: V, b: V, acc: V) -> V {
        let t = self.t(Ty::U32);
        if self.dot {
            let d = self
                .b
                .s_dot(t, None, a, b, Some(PackedVectorFormat::PackedVectorFormat4x8Bit))
                .unwrap();
            return self.add(acc, d);
        }
        let c24 = self.c32(24);
        let mut acc = acc;
        for lane in 0..4u32 {
            let sh = self.c32(lane * 8);
            let mut by = [0; 2];
            for (i, w) in [a, b].into_iter().enumerate() {
                let m = self.shr(w, sh);
                let r = self.shl(m, c24);
                by[i] = self.sar(r, c24);
            }
            let p = self.mul(by[0], by[1]);
            acc = self.add(acc, p);
        }
        acc
    }

    // ---- memory ----
    fn gptr(&mut self, ty: Ty, addr: V) -> V {
        let el = self.t(ty);
        let pt = self.b.type_pointer(None, StorageClass::CrossWorkgroup, el);
        self.b.convert_u_to_ptr(pt, None, addr).unwrap()
    }
    pub(crate) fn ld(&mut self, ty: Ty, addr: V, align: u32) -> V {
        let p = self.gptr(ty, addr);
        let t = self.t(ty);
        self.b.load(t, None, p, Some(MemoryAccess::ALIGNED), [Operand::LiteralBit32(align)]).unwrap()
    }
    pub(crate) fn st(&mut self, ty: Ty, addr: V, val: V, align: u32) {
        let p = self.gptr(ty, addr);
        self.b.store(p, val, Some(MemoryAccess::ALIGNED), [Operand::LiteralBit32(align)]).unwrap();
    }
    /// Little-endian 32-bit load from a 2-byte-aligned address as two 16-bit loads (ggml's
    /// `get_int_b2`).
    pub(crate) fn ld_b2(&mut self, addr: V) -> V {
        let lo = self.ld(Ty::U16, addr, 2);
        let a2 = self.addr_off(addr, 2);
        let hi = self.ld(Ty::U16, a2, 2);
        let lo = self.uconv(Ty::U32, lo);
        let hi = self.uconv(Ty::U32, hi);
        let hi = self.shlc(hi, 16);
        self.or(lo, hi)
    }
    pub(crate) fn ld_b4(&mut self, addr: V) -> V {
        self.ld(Ty::U32, addr, 4)
    }
    pub(crate) fn ld_u8(&mut self, addr: V) -> V {
        let v = self.ld(Ty::U8, addr, 1);
        self.uconv(Ty::U32, v)
    }
    pub(crate) fn ld_f16(&mut self, addr: V) -> V {
        let v = self.ld(Ty::U16, addr, 2);
        self.h2f(v)
    }

    pub(crate) fn sh_ptr(&mut self, index: V) -> V {
        let (var, u32t) = self.shared.expect("kernel declared no shared memory");
        let pt = self.b.type_pointer(None, StorageClass::Workgroup, u32t);
        self.b.in_bounds_access_chain(pt, None, var, [index]).unwrap()
    }
    pub(crate) fn sh_ld_f(&mut self, index: V) -> V {
        let p = self.sh_ptr(index);
        let t = self.t(Ty::F32);
        let fp = self.b.type_pointer(None, StorageClass::Workgroup, t);
        let p = self.b.bitcast(fp, None, p).unwrap();
        self.b.load(t, None, p, Some(MemoryAccess::ALIGNED), [Operand::LiteralBit32(4)]).unwrap()
    }
    pub(crate) fn sh_st_f(&mut self, index: V, val: V) {
        let p = self.sh_ptr(index);
        let t = self.t(Ty::F32);
        let fp = self.b.type_pointer(None, StorageClass::Workgroup, t);
        let p = self.b.bitcast(fp, None, p).unwrap();
        self.b.store(p, val, Some(MemoryAccess::ALIGNED), [Operand::LiteralBit32(4)]).unwrap();
    }
    /// Four consecutive f32 of the workgroup array from `index` (a multiple of 4: 16-byte aligned).
    pub(crate) fn sh_ld_f4(&mut self, index: V) -> [V; 4] {
        let p = self.sh_ptr(index);
        let f = self.t(Ty::F32);
        let v4 = self.b.type_vector(f, 4);
        let vp = self.b.type_pointer(None, StorageClass::Workgroup, v4);
        let p = self.b.bitcast(vp, None, p).unwrap();
        let v = self.b.load(v4, None, p, Some(MemoryAccess::ALIGNED), [Operand::LiteralBit32(16)]).unwrap();
        [0u32, 1, 2, 3].map(|i| self.b.composite_extract(f, None, v, [i]).unwrap())
    }
    pub(crate) fn sh_ld(&mut self, index: V) -> V {
        let p = self.sh_ptr(index);
        let t = self.t(Ty::U32);
        self.b.load(t, None, p, Some(MemoryAccess::ALIGNED), [Operand::LiteralBit32(4)]).unwrap()
    }
    pub(crate) fn sh_st(&mut self, index: V, val: V) {
        let p = self.sh_ptr(index);
        self.b.store(p, val, Some(MemoryAccess::ALIGNED), [Operand::LiteralBit32(4)]).unwrap();
    }

    pub(crate) fn barrier(&mut self) {
        let sc = self.c32(scope(Scope::Workgroup) as u32);
        let sem = self.c32((MemorySemantics::ACQUIRE_RELEASE | MemorySemantics::WORKGROUP_MEMORY).bits());
        self.b.control_barrier(sc, sc, sem).unwrap();
    }

    /// `__shfl_xor_sync` on a 32-lane subgroup.
    pub(crate) fn shfl_xor_f(&mut self, val: V, mask: u32) -> V {
        let sc = self.c32(Scope::Subgroup as u32);
        let m = self.c32(mask);
        let t = self.t(Ty::F32);
        self.b.group_non_uniform_shuffle_xor(t, None, sc, val, m).unwrap()
    }

    // ---- builtins ----
    fn builtin(&mut self, which: usize, comp: u32) -> V {
        let v = self.b.load(self.vec3, None, self.builtins[which], None, []).unwrap();
        let u64t = self.t_u64();
        let x = self.b.composite_extract(u64t, None, v, [comp]).unwrap();
        self.uconv(Ty::U32, x)
    }
    fn t_u64(&mut self) -> Word {
        self.t(Ty::U64)
    }
    pub(crate) fn group_id(&mut self, c: u32) -> V {
        self.builtin(0, c)
    }
    pub(crate) fn local_id(&mut self, c: u32) -> V {
        self.builtin(1, c)
    }
    pub(crate) fn num_groups(&mut self, c: u32) -> V {
        self.builtin(2, c)
    }

    // ---- variables ----
    pub(crate) fn var(&mut self, ty: Ty, init: V) -> Var {
        let el = self.t(ty);
        let pt = self.b.type_pointer(None, StorageClass::Function, el);
        let id = self.b.id();
        self.vars.push(Instruction::new(
            Op::Variable,
            Some(pt),
            Some(id),
            vec![Operand::StorageClass(StorageClass::Function)],
        ));
        let v = Var { id, ty };
        self.set(v, init);
        v
    }
    pub(crate) fn get(&mut self, v: Var) -> V {
        let t = self.t(v.ty);
        self.b.load(t, None, v.id, None, []).unwrap()
    }
    pub(crate) fn set(&mut self, v: Var, val: V) {
        self.b.store(v.id, val, None, []).unwrap();
    }

    // ---- control flow ----
    fn begin(&mut self, label: Word) {
        self.b.begin_block(Some(label)).unwrap();
        self.terminated = false;
    }
    pub(crate) fn ret(&mut self) {
        self.b.ret().unwrap();
        self.terminated = true;
    }
    pub(crate) fn if_(&mut self, cond: V, then: impl FnOnce(&mut Self)) {
        let (t, m) = (self.b.id(), self.b.id());
        self.b.selection_merge(m, SelectionControl::NONE).unwrap();
        self.b.branch_conditional(cond, t, m, []).unwrap();
        self.begin(t);
        then(self);
        if !self.terminated {
            self.b.branch(m).unwrap();
        }
        self.begin(m);
    }
    pub(crate) fn while_(&mut self, cond: impl FnOnce(&mut Self) -> V, body: impl FnOnce(&mut Self)) {
        let (header, test, body_l, cont, merge) =
            (self.b.id(), self.b.id(), self.b.id(), self.b.id(), self.b.id());
        self.b.branch(header).unwrap();
        self.begin(header);
        self.b.loop_merge(merge, cont, LoopControl::NONE, []).unwrap();
        self.b.branch(test).unwrap();
        self.begin(test);
        let c = cond(self);
        self.b.branch_conditional(c, body_l, merge, []).unwrap();
        self.begin(body_l);
        body(self);
        if !self.terminated {
            self.b.branch(cont).unwrap();
        }
        self.begin(cont);
        self.b.branch(header).unwrap();
        self.begin(merge);
    }
    /// `for (i = start; i < end; i += step)` over u32 values.
    pub(crate) fn for_(&mut self, start: V, end: V, step: V, body: impl FnOnce(&mut Self, V)) {
        let i = self.var(Ty::U32, start);
        self.while_(
            |kb| {
                let v = kb.get(i);
                kb.ult(v, end)
            },
            |kb| {
                let v = kb.get(i);
                body(kb, v);
                let v = kb.get(i);
                let n = kb.add(v, step);
                kb.set(i, n);
            },
        );
    }
}
