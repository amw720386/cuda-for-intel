//! Offline verification of the generated XMX modules: a small interpreter for
//! exactly the SPIR-V subset `xmx::build` emits, including the semantics of
//! `OpSubgroupMatrixMultiplyAccumulateINTEL`, driven against the CPU reference
//! in `plan.rs` on random and edge shapes.
//!
//! Execution model: each of the 16 invocations of the (single) subgroup runs
//! independently with its own registers; they rendezvous only at the DPAS
//! instruction, which is a subgroup-collective operation. Memory is one flat
//! arena addressed by 64-bit pointers; every access is bounds checked (so the
//! clamped-load design is verified: a load outside an operand's allocation
//! panics) and `Aligned` memory operands are checked against the address.
//!
//! DPAS semantics implemented (extension text, Xe2 shapes, 16-bit inputs):
//! the result type `<M x f32>` fixes `M` (<= 8) and N is the subgroup size (16).
//! Matrix A is `<M x i16>`: component r of invocation l is `A[r][l]` (K = 16 =
//! subgroup size). Matrix B is `<K/2 x i32>`: component p of invocation n holds
//! `B[2p][n]` in the low half and `B[2p+1][n]` in the high half. Matrix C and the
//! result are `<M x f32>`: component r of invocation n is `C[r][n]`. The 16-bit
//! elements are fp16 or bf16 as selected by the operand mask; products are exact
//! and accumulation is in f32.
use crate::plan::*;
use crate::xmx::{self, Act, Cfg, Flags, Input, Ty};
use crate::xmx_host::{self, Arg, Call, Mode};
use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{BuiltIn, Decoration, Op, OpenclStd100Op, Word};
use std::collections::HashMap;

#[derive(Clone, Debug, PartialEq)]
enum V {
    I(u64),
    F(f32),
    H(u16), // 16-bit float bit pattern (only produced by FConvert/Bitcast)
    B(bool),
    Vec(Vec<V>),
}

#[derive(Clone, Debug)]
enum T {
    Void,
    Bool,
    Int(u32),
    Float(u32),
    Vector(Word, u32),
    Ptr(Word),
}

const BASE: u64 = 0x4000_0000;

pub struct Mem {
    bytes: Vec<u8>,
}

impl Mem {
    pub fn new() -> Self {
        Self { bytes: Vec::new() }
    }
    /// Copy `data` into a fresh 256-byte aligned allocation; returns its device address.
    pub fn alloc(&mut self, data: &[u8]) -> u64 {
        while self.bytes.len() % 256 != 0 {
            self.bytes.push(0xCD);
        }
        let addr = BASE + self.bytes.len() as u64;
        self.bytes.extend_from_slice(data);
        // Guard band so overruns land in poison, not in a neighbour.
        self.bytes.extend(std::iter::repeat_n(0xCD, 64));
        addr
    }
    pub fn read(&self, addr: u64, len: usize) -> &[u8] {
        let off = addr.checked_sub(BASE).unwrap_or_else(|| panic!("read below arena: {addr:#x}")) as usize;
        let slice = self.bytes.get(off..off + len).unwrap_or_else(|| panic!("read out of arena: {addr:#x}+{len}"));
        slice
    }
    fn write(&mut self, addr: u64, data: &[u8]) {
        let off = addr.checked_sub(BASE).unwrap_or_else(|| panic!("write below arena: {addr:#x}")) as usize;
        self.bytes
            .get_mut(off..off + data.len())
            .unwrap_or_else(|| panic!("write out of arena: {addr:#x}+{}", data.len()))
            .copy_from_slice(data);
    }
}

struct Prog {
    types: HashMap<Word, T>,
    consts: HashMap<Word, V>,
    builtin_vars: HashMap<Word, BuiltIn>,
    blocks: Vec<(Word, Vec<Instruction>)>,
    label_index: HashMap<Word, usize>,
    params: Vec<Word>,
    ocl: Word,
    bound: usize,
}

fn id(op: &Operand) -> Word {
    match op {
        Operand::IdRef(w) => *w,
        other => panic!("expected id operand, got {other:?}"),
    }
}

fn lit32(op: &Operand) -> u32 {
    match op {
        Operand::LiteralBit32(v) => *v,
        other => panic!("expected literal, got {other:?}"),
    }
}

fn mask(x: u64, w: u32) -> u64 {
    if w >= 64 { x } else { x & ((1u64 << w) - 1) }
}

impl Prog {
    fn load(module: &Module) -> Prog {
        let mut types = HashMap::new();
        let mut consts = HashMap::new();
        let mut builtin_vars = HashMap::new();
        for a in &module.annotations {
            if a.class.opcode == Op::Decorate {
                if let (Operand::IdRef(t), Operand::Decoration(Decoration::BuiltIn), Operand::BuiltIn(b)) =
                    (&a.operands[0], &a.operands[1], &a.operands[2])
                {
                    builtin_vars.insert(*t, *b);
                }
            }
        }
        let mut ocl = 0;
        for i in &module.ext_inst_imports {
            ocl = i.result_id.unwrap();
        }
        for i in &module.types_global_values {
            let rid = i.result_id;
            match i.class.opcode {
                Op::TypeVoid => {
                    types.insert(rid.unwrap(), T::Void);
                }
                Op::TypeBool => {
                    types.insert(rid.unwrap(), T::Bool);
                }
                Op::TypeInt => {
                    types.insert(rid.unwrap(), T::Int(lit32(&i.operands[0])));
                }
                Op::TypeFloat => {
                    types.insert(rid.unwrap(), T::Float(lit32(&i.operands[0])));
                }
                Op::TypeVector => {
                    types.insert(rid.unwrap(), T::Vector(id(&i.operands[0]), lit32(&i.operands[1])));
                }
                Op::TypePointer => {
                    types.insert(rid.unwrap(), T::Ptr(id(&i.operands[1])));
                }
                Op::TypeFunction | Op::Variable => {}
                Op::Constant | Op::ConstantNull => {}
                other => panic!("interpreter: unsupported global {other:?}"),
            }
        }
        let mut p = Prog {
            types,
            consts: HashMap::new(),
            builtin_vars,
            blocks: Vec::new(),
            label_index: HashMap::new(),
            params: Vec::new(),
            ocl,
            bound: module.header.as_ref().unwrap().bound as usize,
        };
        for i in &module.types_global_values {
            match i.class.opcode {
                Op::Constant => {
                    let ty = i.result_type.unwrap();
                    let raw = match &i.operands[0] {
                        Operand::LiteralBit32(v) => *v as u64,
                        Operand::LiteralBit64(v) => *v,
                        o => panic!("constant operand {o:?}"),
                    };
                    let v = match p.types[&ty] {
                        T::Int(w) => V::I(mask(raw, w)),
                        T::Float(32) => V::F(f32::from_bits(raw as u32)),
                        ref t => panic!("constant of type {t:?}"),
                    };
                    consts.insert(i.result_id.unwrap(), v);
                }
                Op::ConstantNull => {
                    let ty = i.result_type.unwrap();
                    consts.insert(i.result_id.unwrap(), p.zero(ty));
                }
                _ => {}
            }
        }
        p.consts = consts;
        let f = &module.functions[0];
        p.params = f.parameters.iter().map(|i| i.result_id.unwrap()).collect();
        for (idx, blk) in f.blocks.iter().enumerate() {
            let label = blk.label.as_ref().unwrap().result_id.unwrap();
            p.label_index.insert(label, idx);
            p.blocks.push((label, blk.instructions.clone()));
        }
        p
    }

    fn zero(&self, ty: Word) -> V {
        match &self.types[&ty] {
            T::Int(_) => V::I(0),
            T::Float(32) => V::F(0.0),
            T::Float(_) => V::H(0),
            T::Bool => V::B(false),
            T::Vector(e, n) => V::Vec((0..*n).map(|_| self.zero(*e)).collect()),
            t => panic!("zero of {t:?}"),
        }
    }

    fn size(&self, ty: Word) -> usize {
        match &self.types[&ty] {
            T::Int(w) | T::Float(w) => (*w / 8) as usize,
            T::Vector(e, n) => self.size(*e) * *n as usize,
            T::Ptr(_) => 8,
            t => panic!("size of {t:?}"),
        }
    }

    fn ser(&self, v: &V, ty: Word, out: &mut Vec<u8>) {
        match (&self.types[&ty], v) {
            (T::Int(w), V::I(x)) => out.extend_from_slice(&x.to_le_bytes()[..(*w / 8) as usize]),
            (T::Float(32), V::F(x)) => out.extend_from_slice(&x.to_le_bytes()),
            (T::Float(16), V::H(x)) => out.extend_from_slice(&x.to_le_bytes()),
            (T::Vector(e, _), V::Vec(xs)) => xs.iter().for_each(|x| self.ser(x, *e, out)),
            (t, v) => panic!("cannot serialise {v:?} as {t:?}"),
        }
    }

    fn de(&self, bytes: &[u8], ty: Word) -> V {
        match &self.types[&ty] {
            T::Int(w) => {
                let mut b = [0u8; 8];
                b[..(*w / 8) as usize].copy_from_slice(&bytes[..(*w / 8) as usize]);
                V::I(u64::from_le_bytes(b))
            }
            T::Float(32) => V::F(f32::from_le_bytes(bytes[..4].try_into().unwrap())),
            T::Float(16) => V::H(u16::from_le_bytes(bytes[..2].try_into().unwrap())),
            T::Vector(e, n) => {
                let es = self.size(*e);
                V::Vec((0..*n as usize).map(|i| self.de(&bytes[i * es..], *e)).collect())
            }
            t => panic!("cannot deserialise {t:?}"),
        }
    }
}

struct Lane {
    regs: Vec<Option<V>>,
    blk: usize,
    pc: usize,
    done: bool,
    waiting: bool,
}

fn as_i(v: &V) -> u64 {
    match v {
        V::I(x) => *x,
        other => panic!("expected integer, got {other:?}"),
    }
}
fn as_f(v: &V) -> f32 {
    match v {
        V::F(x) => *x,
        other => panic!("expected f32, got {other:?}"),
    }
}
fn as_b(v: &V) -> bool {
    match v {
        V::B(x) => *x,
        other => panic!("expected bool, got {other:?}"),
    }
}

impl Lane {
    fn get<'a>(&'a self, p: &'a Prog, w: Word) -> &'a V {
        p.consts
            .get(&w)
            .or(self.regs[w as usize].as_ref())
            .unwrap_or_else(|| panic!("use of undefined id %{w}"))
    }
}

fn enter(p: &Prog, lane: &mut Lane, target: Word, from: Word) {
    let idx = p.label_index[&target];
    lane.blk = idx;
    let insts = &p.blocks[idx].1;
    let phis: Vec<(Word, V)> = insts
        .iter()
        .take_while(|i| i.class.opcode == Op::Phi)
        .map(|i| {
            let mut pairs = i.operands.chunks(2);
            let v = pairs
                .find(|c| id(&c[1]) == from)
                .map(|c| lane.get(p, id(&c[0])).clone())
                .unwrap_or_else(|| panic!("phi without incoming edge from %{from}"));
            (i.result_id.unwrap(), v)
        })
        .collect();
    lane.pc = phis.len();
    for (r, v) in phis {
        lane.regs[r as usize] = Some(v);
    }
}

/// Execute one instruction of `lane`. Returns when the lane finishes or blocks on DPAS.
fn step(p: &Prog, lane: &mut Lane, mem: &mut Mem, params: &[V], wg: [u64; 3], lid: u64) {
    let (label, insts) = &p.blocks[lane.blk];
    let inst = &insts[lane.pc];
    let rt = inst.result_type;
    let op = inst.class.opcode;
    let o = &inst.operands;
    macro_rules! set {
        ($v:expr) => {{
            let v = $v;
            lane.regs[inst.result_id.unwrap() as usize] = Some(v);
        }};
    }
    let g = |lane: &Lane, i: usize| -> V { lane.get(p, id(&o[i])).clone() };
    let width = |ty: Word| match p.types[&ty] {
        T::Int(w) => w,
        ref t => panic!("integer op on {t:?}"),
    };
    lane.pc += 1;
    match op {
        Op::Load => {
            let ty = rt.unwrap();
            if let Some(var) = o.first().map(id).filter(|v| p.builtin_vars.contains_key(v)) {
                let v = match p.builtin_vars[&var] {
                    BuiltIn::WorkgroupId => V::Vec(wg.iter().map(|&x| V::I(x)).collect()),
                    BuiltIn::LocalInvocationId => V::Vec(vec![V::I(lid), V::I(0), V::I(0)]),
                    b => panic!("builtin {b:?}"),
                };
                set!(v);
            } else {
                let ptr = as_i(&g(lane, 0));
                if let (Some(Operand::MemoryAccess(_)), Some(a)) = (o.get(1), o.get(2)) {
                    let al = lit32(a) as u64;
                    assert!(ptr % al == 0, "load of %{} misaligned: {ptr:#x} not {al}-aligned", inst.result_id.unwrap());
                }
                let bytes = mem.read(ptr, p.size(ty)).to_vec();
                set!(p.de(&bytes, ty));
            }
        }
        Op::Store => {
            let ptr = as_i(&g(lane, 0));
            let v = g(lane, 1);
            // The stored type is recovered from the value shape via the pointee of an earlier
            // ConvertUToPtr; all stores here are scalars, so serialise by value kind.
            let mut bytes = Vec::new();
            match &v {
                V::I(x) => {
                    // Width comes from the memory operand alignment (2/4/8 bytes).
                    let al = lit32(&o[3]) as usize;
                    bytes.extend_from_slice(&x.to_le_bytes()[..al]);
                }
                V::F(x) => bytes.extend_from_slice(&x.to_le_bytes()),
                V::H(x) => bytes.extend_from_slice(&x.to_le_bytes()),
                other => panic!("store of {other:?}"),
            }
            let al = lit32(&o[3]) as u64;
            assert!(ptr % al == 0, "store misaligned: {ptr:#x} not {al}-aligned");
            mem.write(ptr, &bytes);
        }
        Op::ConvertUToPtr => set!(g(lane, 0)),
        Op::IAdd | Op::ISub | Op::IMul => {
            let w = width(rt.unwrap());
            let (a, b) = (as_i(&g(lane, 0)), as_i(&g(lane, 1)));
            let r = match op {
                Op::IAdd => a.wrapping_add(b),
                Op::ISub => a.wrapping_sub(b),
                _ => a.wrapping_mul(b),
            };
            set!(V::I(mask(r, w)));
        }
        Op::ShiftLeftLogical => {
            let w = width(rt.unwrap());
            set!(V::I(mask(as_i(&g(lane, 0)) << as_i(&g(lane, 1)), w)));
        }
        Op::ShiftRightLogical => set!(V::I(as_i(&g(lane, 0)) >> as_i(&g(lane, 1)))),
        Op::BitwiseAnd => set!(V::I(as_i(&g(lane, 0)) & as_i(&g(lane, 1)))),
        Op::BitwiseOr => set!(V::I(as_i(&g(lane, 0)) | as_i(&g(lane, 1)))),
        Op::UConvert => set!(V::I(mask(as_i(&g(lane, 0)), width(rt.unwrap())))),
        Op::Select => {
            let c = as_b(&g(lane, 0));
            set!(if c { g(lane, 1) } else { g(lane, 2) });
        }
        Op::ULessThan => set!(V::B(as_i(&g(lane, 0)) < as_i(&g(lane, 1)))),
        Op::ULessThanEqual => set!(V::B(as_i(&g(lane, 0)) <= as_i(&g(lane, 1)))),
        Op::LogicalAnd => set!(V::B(as_b(&g(lane, 0)) && as_b(&g(lane, 1)))),
        Op::IsNan => set!(V::B(as_f(&g(lane, 0)).is_nan())),
        Op::FMul => set!(V::F(as_f(&g(lane, 0)) * as_f(&g(lane, 1)))),
        Op::FAdd => set!(V::F(as_f(&g(lane, 0)) + as_f(&g(lane, 1)))),
        Op::FConvert => {
            let v = g(lane, 0);
            let out = match (&p.types[&rt.unwrap()], v) {
                (T::Float(16), V::F(x)) => V::H(f32_to_f16_bits(x)),
                (T::Float(32), V::H(x)) => V::F(f16_bits_to_f32(x)),
                (t, v) => panic!("FConvert {v:?} -> {t:?}"),
            };
            set!(out);
        }
        Op::Bitcast => {
            let src = g(lane, 0);
            // Find the source type by serialising generically: ints/floats/vectors only.
            let mut bytes = Vec::new();
            fn flat(v: &V, out: &mut Vec<u8>, int_bytes: usize) {
                match v {
                    V::I(x) => out.extend_from_slice(&x.to_le_bytes()[..int_bytes]),
                    V::F(x) => out.extend_from_slice(&x.to_le_bytes()),
                    V::H(x) => out.extend_from_slice(&x.to_le_bytes()),
                    V::Vec(xs) => xs.iter().for_each(|x| flat(x, out, int_bytes)),
                    V::B(_) => panic!("bitcast of bool"),
                }
            }
            // Integer element width of the source = destination's total size / element count heuristic is
            // ambiguous, so use the operand's declared type recorded in the register file.
            let src_ty = type_of_id(p, id(&o[0]), insts, lane.blk);
            let int_bytes = match &p.types[&src_ty] {
                T::Int(w) => (*w / 8) as usize,
                T::Vector(e, _) => match &p.types[e] {
                    T::Int(w) => (*w / 8) as usize,
                    _ => 4,
                },
                _ => 4,
            };
            flat(&src, &mut bytes, int_bytes);
            set!(p.de(&bytes, rt.unwrap()));
        }
        Op::CompositeConstruct => {
            set!(V::Vec(o.iter().map(|x| lane.get(p, id(x)).clone()).collect()));
        }
        Op::CompositeExtract => match g(lane, 0) {
            V::Vec(xs) => set!(xs[lit32(&o[1]) as usize].clone()),
            other => panic!("extract from {other:?}"),
        },
        Op::ExtInst => {
            let num = match &o[1] {
                Operand::LiteralExtInstInteger(n) => *n,
                other => panic!("ext inst number {other:?}"),
            };
            assert_eq!(id(&o[0]), p.ocl);
            let args: Vec<V> = o[2..].iter().map(|x| lane.get(p, id(x)).clone()).collect();
            let r = if num == OpenclStd100Op::u_min as u32 {
                V::I(as_i(&args[0]).min(as_i(&args[1])))
            } else if num == OpenclStd100Op::fma as u32 {
                V::F(as_f(&args[0]).mul_add(as_f(&args[1]), as_f(&args[2])))
            } else if num == OpenclStd100Op::fmax as u32 {
                V::F(as_f(&args[0]).max(as_f(&args[1])))
            } else if num == OpenclStd100Op::tanh as u32 {
                V::F(as_f(&args[0]).tanh())
            } else {
                panic!("ext inst {num}")
            };
            set!(r);
        }
        Op::SubgroupMatrixMultiplyAccumulateINTEL => {
            lane.pc -= 1; // re-executed by the rendezvous
            lane.waiting = true;
        }
        Op::SelectionMerge | Op::LoopMerge => {}
        Op::Branch => {
            let t = id(&o[0]);
            enter(p, lane, t, *label);
        }
        Op::BranchConditional => {
            let c = as_b(&g(lane, 0));
            let t = if c { id(&o[1]) } else { id(&o[2]) };
            enter(p, lane, t, *label);
        }
        Op::Return => lane.done = true,
        other => panic!("interpreter: unsupported op {other:?}"),
    }
    let _ = params;
}

/// Static type of an SSA id (for bitcast width recovery): scan defining instruction.
fn type_of_id(p: &Prog, w: Word, _insts: &[Instruction], _blk: usize) -> Word {
    for (_, insts) in &p.blocks {
        for i in insts {
            if i.result_id == Some(w) {
                return i.result_type.unwrap();
            }
        }
    }
    panic!("type of %{w} unknown (bitcast of a constant/parameter)")
}

fn dpas(p: &Prog, lanes: &mut [Lane]) {
    let (b, pc) = (lanes[0].blk, lanes[0].pc);
    assert!(lanes.iter().all(|l| l.waiting && l.blk == b && l.pc == pc), "DPAS reached by diverged invocations");
    let inst = p.blocks[b].1[pc].clone();
    let o = &inst.operands;
    let m = match &p.types[&inst.result_type.unwrap()] {
        T::Vector(_, n) => *n as usize,
        t => panic!("DPAS result {t:?}"),
    };
    assert!(m <= 8, "DPAS rows must be <= 8");
    assert_eq!(lanes.len(), 16);
    let k = as_i(lanes[0].get(p, id(&o[0]))) as usize;
    assert_eq!(k, 16, "interpreter models K = 16 (16-bit inputs)");
    let flags = match o.get(4) {
        Some(Operand::MatrixMultiplyAccumulateOperands(f)) => f.bits(),
        _ => 0,
    };
    let bf16 = flags & (4096 | 8192) != 0;
    let to_f = |bits: u16| if bf16 { bf16_bits_to_f32(bits) } else { f16_bits_to_f32(bits) };
    let comp = |lane: &Lane, i: usize, c: usize| -> V {
        match lane.get(p, id(&o[i])) {
            V::Vec(xs) => xs[c].clone(),
            other => panic!("DPAS operand {other:?}"),
        }
    };
    // A[r][l] = lane l component r; B[2q+h][n] = half h of lane n component q.
    let a = |r: usize, l: usize| to_f(as_i(&comp(&lanes[l], 1, r)) as u16);
    let bm = |kk: usize, n: usize| {
        let w = as_i(&comp(&lanes[n], 2, kk / 2));
        to_f(((w >> (16 * (kk % 2))) & 0xffff) as u16)
    };
    let mut result = vec![vec![0f32; 16]; m];
    for r in 0..m {
        for n in 0..16 {
            let mut acc = as_f(&comp(&lanes[n], 3, r));
            let mut sum = 0f32;
            for kk in 0..16 {
                sum += a(r, kk) * bm(kk, n);
            }
            acc += sum;
            result[r][n] = acc;
        }
    }
    for (n, lane) in lanes.iter_mut().enumerate() {
        lane.regs[inst.result_id.unwrap() as usize] = Some(V::Vec((0..m).map(|r| V::F(result[r][n])).collect()));
        lane.pc += 1;
        lane.waiting = false;
    }
}

/// Run `words` (one kernel, 16-invocation workgroups) over `grid`.
pub fn run(words: &[u32], args: &[Arg], grid: (u32, u32, u32), mem: &mut Mem) {
    let module = rspirv::dr::load_words(words).expect("parse spirv");
    let p = Prog::load(&module);
    assert_eq!(p.params.len(), args.len());
    let params: Vec<V> = args
        .iter()
        .map(|a| match *a {
            Arg::U64(v) => V::I(v),
            Arg::U32(v) => V::I(v as u64),
            Arg::F32(v) => V::F(v),
        })
        .collect();
    for z in 0..grid.2 as u64 {
        for y in 0..grid.1 as u64 {
            for x in 0..grid.0 as u64 {
                let mut lanes: Vec<Lane> = (0..16)
                    .map(|_| {
                        let mut regs = vec![None; p.bound];
                        for (pid, v) in p.params.iter().zip(&params) {
                            regs[*pid as usize] = Some(v.clone());
                        }
                        Lane { regs, blk: 0, pc: 0, done: false, waiting: false }
                    })
                    .collect();
                let mut guard = 0u64;
                while lanes.iter().any(|l| !l.done) {
                    for (lid, lane) in lanes.iter_mut().enumerate() {
                        while !lane.done && !lane.waiting {
                            step(&p, lane, mem, &params, [x, y, z], lid as u64);
                            guard += 1;
                            assert!(guard < 200_000_000, "interpreter step limit");
                        }
                    }
                    if lanes.iter().any(|l| l.waiting) {
                        dpas(&p, &mut lanes);
                    }
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tests against the CPU reference
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn val(&mut self) -> f32 {
            (self.next() % 33) as f32 / 8.0 - 2.0 // multiples of 1/8 in [-2, 2]: exact in f16 and bf16
        }
    }

    fn enc(dt: DType, v: f32) -> Vec<u8> {
        let mut b = vec![0u8; dt.size()];
        encode(dt, &mut b, 0, v);
        b
    }

    #[derive(Clone, Copy, Debug)]
    struct Case {
        m: usize,
        n: usize,
        k: usize,
        ta: bool,
        tb: bool,
        input: Input,
        out: Ty,
        batch: usize,
        ptrs: bool,
        alpha: f32,
        beta: f32,
        bias: Option<Ty>,
        act: Act,
        tile: (u32, u32),
        pad: usize,
        flags: Flags,
        /// CUBLAS_COMPUTE_16F: f16 scalars, `opts.acc16` set (f16 in/out only).
        acc16: bool,
        /// Pointer-table operands start 2 bytes past a 256-byte boundary (ggml views).
        misalign: bool,
    }

    fn dt_of(t: Ty) -> DType {
        match t {
            Ty::F16 => DType::F16,
            Ty::Bf16 => DType::Bf16,
            Ty::F32 => DType::F32,
        }
    }

    /// Inputs, options and the CPU-reference result of one case.
    struct Prep {
        c: Case,
        d: GemmDesc,
        opts: crate::ptxgen::Opts,
        abytes: Vec<u8>,
        bbytes: Vec<u8>,
        cbytes: Vec<u8>,
        bias_bytes: Vec<u8>,
        want: Vec<u8>,
        ea: usize,
        eb: usize,
        ec: usize,
        ldc: usize,
    }

    fn prepare(c: Case, seed: u64) -> Prep {
        let mut rng = Rng(seed | 1);
        let in_dt = if c.input == Input::F16 { DType::F16 } else { DType::Bf16 };
        let out_dt = dt_of(c.out);
        let (rows_a, cols_a) = if c.ta { (c.k, c.m) } else { (c.m, c.k) };
        let (rows_b, cols_b) = if c.tb { (c.n, c.k) } else { (c.k, c.n) };
        let (lda, ldb, ldc) = (rows_a + c.pad, rows_b + c.pad, c.m + c.pad);
        let d = GemmDesc {
            trans_a: c.ta,
            trans_b: c.tb,
            m: c.m as u32,
            n: c.n as u32,
            k: c.k as u32,
            lda: lda as u32,
            ldb: ldb as u32,
            ldc: ldc as u32,
            a: in_dt,
            b: in_dt,
            c: out_dt,
            scalar: if c.acc16 { ScalarKind::F16 } else { ScalarKind::F32 },
        };
        let (ea, eb, ec) = (lda * cols_a, ldb * cols_b, ldc * c.n);
        // Distinct matrices per batch entry; the pointer variant broadcasts A across the batch.
        let a_mats = if c.ptrs { 1 } else { c.batch };
        let mut abytes = Vec::new();
        for _ in 0..ea * a_mats {
            abytes.extend(enc(in_dt, rng.val()));
        }
        let mut bbytes = Vec::new();
        for _ in 0..eb * c.batch {
            bbytes.extend(enc(in_dt, rng.val()));
        }
        let mut cbytes = Vec::new();
        for _ in 0..ec * c.batch {
            cbytes.extend(enc(out_dt, rng.val()));
        }
        let bias_vals: Vec<f32> = (0..c.m).map(|_| rng.val()).collect();
        let mut bias_bytes = Vec::new();
        if let Some(bt) = c.bias {
            for v in &bias_vals {
                bias_bytes.extend(enc(dt_of(bt), *v));
            }
        }
        // Expected result (f32 epilogue before rounding to C's type).
        let mut want = cbytes.clone();
        for z in 0..c.batch {
            let a_off = if c.ptrs { 0 } else { z * ea };
            for j in 0..c.n {
                for i in 0..c.m {
                    let mut acc = 0f32;
                    for l in 0..c.k {
                        let ai = if c.ta { l + i * lda } else { i + l * lda };
                        let bi = if c.tb { j + l * ldb } else { l + j * ldb };
                        acc += decode(in_dt, &abytes, a_off + ai) * decode(in_dt, &bbytes, z * eb + bi);
                    }
                    let ci = z * ec + i + j * ldc;
                    let mut x = acc * c.alpha;
                    if c.beta != 0.0 {
                        x += c.beta * decode(out_dt, &cbytes, ci);
                    }
                    if let Some(bt) = c.bias {
                        x += decode(dt_of(bt), &bias_bytes, i);
                    }
                    x = match c.act {
                        Act::None => x,
                        Act::Relu => x.max(0.0),
                        Act::Gelu => {
                            let t = (crate::ptxgen::GELU_SQRT_2_OVER_PI * (x + crate::ptxgen::GELU_CUBIC * x * x * x)).tanh();
                            0.5 * x * (1.0 + t)
                        }
                    };
                    encode(out_dt, &mut want, ci, x);
                }
            }
        }
        let opts = crate::ptxgen::Opts {
            epilogue: match (c.act, c.bias.is_some()) {
                (Act::None, false) => crate::ptxgen::Epilogue::None,
                (Act::None, true) => crate::ptxgen::Epilogue::Bias,
                (Act::Relu, false) => crate::ptxgen::Epilogue::Relu,
                (Act::Relu, true) => crate::ptxgen::Epilogue::ReluBias,
                (Act::Gelu, false) => crate::ptxgen::Epilogue::Gelu,
                (Act::Gelu, true) => crate::ptxgen::Epilogue::GeluBias,
            },
            bias: match c.bias {
                Some(Ty::F16) => crate::ptxgen::Elem::F16,
                Some(Ty::Bf16) => crate::ptxgen::Elem::Bf16,
                _ => crate::ptxgen::Elem::F32,
            },
            acc16: c.acc16,
        };
        Prep { c, d, opts, abytes, bbytes, cbytes, bias_bytes, want, ea, eb, ec, ldc }
    }

    /// Padding rows of C must be untouched; valid entries within rounding tolerance.
    fn verify(p: &Prep, got: &[u8]) {
        let (c, out_dt) = (p.c, p.d.c);
        for z in 0..c.batch {
            for j in 0..c.n {
                for i in 0..p.ldc {
                    let ci = z * p.ec + i + j * p.ldc;
                    let sz = out_dt.size();
                    if i >= c.m {
                        assert_eq!(&got[ci * sz..(ci + 1) * sz], &p.cbytes[ci * sz..(ci + 1) * sz], "padding clobbered {c:?} z={z} i={i} j={j}");
                        continue;
                    }
                    let (g, w) = (decode(out_dt, got, ci), decode(out_dt, &p.want, ci));
                    let tol = match out_dt {
                        DType::F32 => 2e-4 * (1.0 + w.abs()),
                        DType::F16 => 2e-3 * w.abs() + 2e-3,
                        DType::Bf16 => 1.6e-2 * w.abs() + 2e-2,
                    };
                    assert!((g - w).abs() <= tol, "{c:?} z={z} i={i} j={j}: got {g} want {w}");
                }
            }
        }
    }

    /// Returns the Cfg actually used (for coverage assertions).
    fn check(c: Case, seed: u64) -> Cfg {
        let p = prepare(c, seed);
        let (d, in_dt, out_dt) = (p.d, p.d.a, p.d.c);
        let mut mem = Mem::new();
        let shift = if c.misalign && c.ptrs { 2usize } else { 0 };
        let shifted = |mem: &mut Mem, b: &[u8]| -> u64 {
            let mut v = vec![0u8; shift];
            v.extend_from_slice(b);
            mem.alloc(&v) + shift as u64
        };
        let (da, db) = (shifted(&mut mem, &p.abytes), shifted(&mut mem, &p.bbytes));
        let dc = mem.alloc(&p.cbytes);
        let dbias = if c.bias.is_some() { mem.alloc(&p.bias_bytes) } else { 0 };
        let (sa, sb, sc) = (p.ea as i64, p.eb as i64, p.ec as i64);
        let call = Call { d, pointer_arrays: c.ptrs, beta: c.beta, opts: p.opts, a: da, b: db, sa, sb, batch: c.batch as u32 };
        let cfg = xmx_host::qualify(&call, Mode::Force, c.tile, c.flags).expect("shape qualifies");
        let words = xmx::build(&cfg).unwrap();
        let (gx, gy) = xmx_host::grid(&d, &cfg);
        let args = if c.ptrs {
            let tab = |mem: &mut Mem, base: u64, per: u64| -> u64 {
                let b: Vec<u8> = (0..c.batch as u64).flat_map(|z| (base + z * per).to_le_bytes()).collect();
                mem.alloc(&b)
            };
            let ta = tab(&mut mem, da, 0); // broadcast A
            let tb = tab(&mut mem, db, (p.eb * in_dt.size()) as u64);
            let tc = tab(&mut mem, dc, (p.ec * out_dt.size()) as u64);
            xmx_host::kernel_args(&d, ta, tb, tc, c.alpha, c.beta, (0, 0, 0), dbias)
        } else {
            xmx_host::kernel_args(&d, da, db, dc, c.alpha, c.beta, (sa as u64, sb as u64, sc as u64), dbias)
        };
        run(&words, &args, (gx, gy, c.batch as u32), &mut mem);
        verify(&p, mem.read(dc, p.cbytes.len()));
        cfg
    }

    /// Hardware: the same oracle, but the GEMM runs through `run_gemm_ex` on the GPU
    /// (so the probe, kernel cache, launcher and every epilogue are exercised).
    /// Run with `cargo test -p cublas_shim -- --ignored gpu_xmx`.
    #[test]
    #[ignore = "requires nvcuda with an Intel GPU (Xe2/Xe3) and a current CUDA context"]
    fn gpu_xmx_epilogues_and_edges_match_cpu_reference() {
        use crate::{Handle, driver, run_gemm_ex};
        use std::ffi::c_void;
        unsafe { std::env::set_var("CFI_XMX", "2") };
        let drv = driver().expect("driver");
        let lib = &drv._lib;
        type Init = unsafe extern "system" fn(u32) -> i32;
        type Create = unsafe extern "system" fn(*mut *mut c_void, u32, i32) -> i32;
        type Alloc = unsafe extern "system" fn(*mut u64, usize) -> i32;
        type HtoD = unsafe extern "system" fn(u64, *const c_void, usize) -> i32;
        type DtoH = unsafe extern "system" fn(*mut c_void, u64, usize) -> i32;
        unsafe {
            let init: libloading::Symbol<Init> = lib.get(b"cuInit\0").unwrap();
            let create: libloading::Symbol<Create> = lib.get(b"cuCtxCreate_v2\0").unwrap();
            let alloc: libloading::Symbol<Alloc> = lib.get(b"cuMemAlloc_v2\0").unwrap();
            let htod: libloading::Symbol<HtoD> = lib.get(b"cuMemcpyHtoD_v2\0").unwrap();
            let dtoh: libloading::Symbol<DtoH> = lib.get(b"cuMemcpyDtoH_v2\0").unwrap();
            assert_eq!(init(0), 0);
            let mut ctx = std::ptr::null_mut();
            assert_eq!(create(&mut ctx, 0, 0), 0);
            let up = |b: &[u8]| -> u64 {
                let mut p = 0u64;
                assert_eq!(alloc(&mut p, b.len().max(8)), 0);
                assert_eq!(htod(p, b.as_ptr().cast(), b.len()), 0);
                p
            };
            let mut h = Handle {
                stream: std::ptr::null_mut(),
                math_mode: 0,
                pointer_mode: POINTER_MODE_HOST,
                workspace: (0, 0),
            };
            let mut seed = 31;
            let mut n_cases = 0;
            let before = crate::xmx_host::cfiXmxLaunchCount();
            for input in [Input::F16, Input::Bf16] {
                for out in [Ty::F32, if input == Input::F16 { Ty::F16 } else { Ty::Bf16 }] {
                    for act in [Act::None, Act::Relu, Act::Gelu] {
                        for bias in [None, Some(Ty::F32), Some(Ty::F16), Some(Ty::Bf16)] {
                            for (ta, tb, m, n, k, beta) in [(true, false, 37usize, 19usize, 45usize, 0.0f32), (false, true, 64, 64, 64, 0.5), (true, true, 17, 33, 20, 0.0)] {
                                seed += 1;
                                let c = Case { m, n, k, ta, tb, input, out, batch: 2, ptrs: false, alpha: 1.0, beta, bias, act, tile: (2, 4), pad: 1, flags: Flags::Packed, acc16: false, misalign: false };
                                let pr = prepare(c, seed);
                                let (da, db, dc) = (up(&pr.abytes), up(&pr.bbytes), up(&pr.cbytes));
                                let dbias = if bias.is_some() { up(&pr.bias_bytes) } else { 0 };
                                let (alpha, beta) = (c.alpha, c.beta);
                                let batching = crate::Batching::Strided { a: da, b: db, c: dc, sa: pr.ea as i64, sb: pr.eb as i64, sc: pr.ec as i64 };
                                let st = run_gemm_ex(&mut h, &pr.d, std::ptr::from_ref(&alpha).cast(), std::ptr::from_ref(&beta).cast(), 2, batching, &pr.opts, dbias);
                                assert_eq!(st, 0, "{c:?}");
                                let mut got = vec![0u8; pr.cbytes.len()];
                                assert_eq!(dtoh(got.as_mut_ptr().cast(), dc, got.len()), 0);
                                verify(&pr, &got);
                                n_cases += 1;
                            }
                        }
                    }
                }
            }
            let launched = crate::xmx_host::cfiXmxLaunchCount() - before;
            println!("gpu_xmx: {n_cases} cases, {launched} XMX launches, probe status {:#06b}", crate::xmx_host::cfiXmxProbeStatus());
            assert!(launched >= n_cases, "XMX path was not taken for every case");
        }
    }

    fn base() -> Case {
        Case {
            m: 16, n: 8, k: 16, ta: false, tb: false, input: Input::F16, out: Ty::F32, batch: 1, ptrs: false,
            alpha: 1.0, beta: 0.0, bias: None, act: Act::None, tile: (1, 1), pad: 0, flags: Flags::Packed, acc16: false, misalign: false,
        }
    }

    #[test]
    fn exact_tile_both_inputs() {
        for input in [Input::F16, Input::Bf16] {
            check(Case { input, ..base() }, 1);
        }
    }

    #[test]
    fn edge_shapes_all_transposes() {
        let sizes = [1usize, 7, 16, 17, 31, 33];
        let mut seed = 10;
        for &m in &sizes {
            for &n in &[1usize, 9, 16, 24] {
                for &k in &[1usize, 15, 16, 17, 40] {
                    for (ta, tb) in [(false, false), (true, false), (false, true), (true, true)] {
                        seed += 1;
                        check(Case { m, n, k, ta, tb, tile: (1, 1), pad: (seed % 3) as usize, ..base() }, seed);
                    }
                }
            }
        }
    }

    #[test]
    fn production_tile_and_output_types() {
        let mut seed = 500;
        for out in [Ty::F32, Ty::F16, Ty::Bf16] {
            for input in [Input::F16, Input::Bf16] {
                for (ta, tb) in [(true, false), (false, true), (false, false), (true, true)] {
                    seed += 1;
                    check(Case { m: 37, n: 41, k: 50, ta, tb, input, out, tile: (2, 4), pad: 2, alpha: 0.5, beta: 0.25, ..base() }, seed);
                }
            }
        }
    }

    #[test]
    fn beta_zero_never_reads_c_and_fast_loads_are_used() {
        // m,k even and n % 8 == 0 with op T on both: pair_a and vec_b variants.
        let cfg = check(Case { m: 48, n: 40, k: 48, ta: true, tb: true, tile: (2, 4), pad: 0, ..base() }, 77);
        assert!(cfg.pair_a && cfg.vec_b && cfg.beta0);
        let cfg = check(Case { m: 33, n: 24, k: 37, ta: true, tb: true, tile: (2, 4), pad: 8, beta: 1.0, ..base() }, 78);
        assert!(!cfg.beta0 && cfg.vec_b);
        let _ = cfg;
    }

    #[test]
    fn epilogues_and_bias_types() {
        let mut seed = 900;
        for act in [Act::None, Act::Relu, Act::Gelu] {
            for bias in [None, Some(Ty::F32), Some(Ty::F16), Some(Ty::Bf16)] {
                seed += 1;
                check(Case { m: 35, n: 19, k: 33, out: Ty::F16, bias, act, tile: (2, 4), beta: 0.5, ..base() }, seed);
            }
        }
    }

    #[test]
    fn batched_strided_and_pointer_tables() {
        let mut seed = 1200;
        for ptrs in [false, true] {
            for out in [Ty::F32, Ty::Bf16] {
                seed += 1;
                check(Case { m: 20, n: 11, k: 23, batch: 3, ptrs, out, tile: (1, 2), pad: 1, ..base() }, seed);
            }
        }
    }

    #[test]
    fn compute_16f_rounds_once_from_f32_accumulators() {
        // ggml KQV shape family: f16 A/B/C, f16 alpha/beta, pointer tables and strided batches.
        let mut seed = 2000;
        for ptrs in [true, false] {
            for (m, n, k, ta, tb) in [(64usize, 40usize, 50usize, false, false), (33, 17, 70, true, false), (16, 16, 16, true, true)] {
                seed += 1;
                check(Case { m, n, k, ta, tb, input: Input::F16, out: Ty::F16, batch: 4, ptrs, acc16: true, alpha: 0.5, beta: 0.25, tile: (2, 4), pad: 1, ..base() }, seed);
            }
        }
        // A long K accumulation that an f16 accumulator would round away: sum of 2048 values near 1/8.
        let c = Case { m: 16, n: 8, k: 2048, input: Input::F16, out: Ty::F16, acc16: true, ..base() };
        check(c, 9);
    }

    #[test]
    fn two_byte_aligned_pointer_table_operands() {
        // ggml views can hand out 2-byte aligned addresses; the pointer-table variant must use
        // 16-bit loads only (the interpreter asserts every Aligned operand).
        let mut seed = 3000;
        for (ta, tb) in [(true, false), (false, false), (true, true), (false, true)] {
            seed += 1;
            let cfg = check(Case { m: 48, n: 24, k: 48, ta, tb, ptrs: true, batch: 3, misalign: true, tile: (2, 4), ..base() }, seed);
            assert!(cfg.ptrs && !cfg.pair_a && !cfg.vec_b);
        }
    }

    /// Evidence for leaving f32 GEMMs (llama.cpp QK) on the PTX kernels: error of split-precision
    /// XMX schemes against an f64 reference, relative to sum|a||b|, versus the fp32 PTX kernel
    /// (sequential f32 fma). bf16 2-term splits are ~20x worse than fp32; a 3-way bf16 split
    /// (6 DPAS products) gets within ~3x of fp32 error, which divides the 5-8 TFLOP/s measured for
    /// f16 XMX by 6 and adds the split ALU cost, i.e. break-even with the ~1 TFLOP/s PTX kernel.
    #[test]
    fn f32_split_precision_study() {
        let mut rng = Rng(0xfeed);
        let (k, trials) = (64usize, 400usize);
        let gauss = |rng: &mut Rng| -> f32 {
            let u = |r: &mut Rng| (r.next() % 1_000_003) as f64 / 1_000_003.0 + 1e-9;
            ((-2.0 * u(rng).ln()).sqrt() * (6.283185307179586 * u(rng)).cos()) as f32
        };
        let bf = |x: f32| bf16_bits_to_f32(f32_to_bf16_bits(x));
        let hf = |x: f32| f16_bits_to_f32(f32_to_f16_bits(x));
        let split_bf = |x: f32| {
            let h = bf(x);
            let m = bf(x - h);
            let l = bf(x - h - m);
            [h, m, l]
        };
        let split_h = |x: f32| {
            let h = hf(x);
            [h, hf(x - h)]
        };
        let (mut e32, mut e_bf3, mut e_bf2, mut e_h2) = (0f64, 0f64, 0f64, 0f64);
        for _ in 0..trials {
            let a: Vec<f32> = (0..k).map(|_| gauss(&mut rng)).collect();
            let b: Vec<f32> = (0..k).map(|_| gauss(&mut rng)).collect();
            let exact: f64 = a.iter().zip(&b).map(|(x, y)| *x as f64 * *y as f64).sum();
            let scale: f64 = a.iter().zip(&b).map(|(x, y)| (*x as f64 * *y as f64).abs()).sum();
            let mut f = 0f32;
            for i in 0..k {
                f = a[i].mul_add(b[i], f);
            }
            let (mut s6, mut s3b, mut s3h) = (0f32, 0f32, 0f32);
            for i in 0..k {
                let (sa, sb) = (split_bf(a[i]), split_bf(b[i]));
                for (x, y) in [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1), (2, 0)] {
                    s6 += sa[x] * sb[y];
                }
                let (ta, tb) = ([sa[0], sa[1]], [sb[0], sb[1]]);
                for (x, y) in [(0, 0), (0, 1), (1, 0)] {
                    s3b += ta[x] * tb[y];
                }
                let (ha, hb) = (split_h(a[i]), split_h(b[i]));
                for (x, y) in [(0, 0), (0, 1), (1, 0)] {
                    s3h += ha[x] * hb[y];
                }
            }
            let rel = |v: f32| (v as f64 - exact).abs() / scale;
            e32 = e32.max(rel(f));
            e_bf3 = e_bf3.max(rel(s6));
            e_bf2 = e_bf2.max(rel(s3b));
            e_h2 = e_h2.max(rel(s3h));
        }
        println!("f32 split study (max err/sum|ab|, k={k}): fp32 fma {e32:.2e}, bf16x3 6-product {e_bf3:.2e}, bf16x2 3-product {e_bf2:.2e}, f16x2 3-product {e_h2:.2e}");
        // Measured: bf16x2 ~20x, bf16x3 ~3x, f16x2 ~2x the fp32 kernel's error.
        assert!(e_bf2 > 10.0 * e32, "bf16x2 is not fp32-class");
        assert!(e_bf3 > 2.0 * e32, "bf16x3 (6 products) is not within 2x either");
        assert!(e_h2 < 3.0 * e32, "f16x2 is the only 3-product scheme near fp32 on gaussian data");
    }

    #[test]
    fn bare_flags_variant_matches_packed_for_f16() {
        check(Case { flags: Flags::Bare, m: 17, n: 9, k: 33, ..base() }, 5);
    }

    #[test]
    fn plan_reference_agrees_with_test_oracle_on_one_case() {
        // Guards the oracle itself against plan::reference_gemm.
        let d = GemmDesc {
            trans_a: true, trans_b: false, m: 5, n: 3, k: 4, lda: 4, ldb: 4, ldc: 5,
            a: DType::F16, b: DType::F16, c: DType::F32, scalar: ScalarKind::F32,
        };
        let mut rng = Rng(3);
        let a: Vec<u8> = (0..20).flat_map(|_| enc(DType::F16, rng.val())).collect();
        let b: Vec<u8> = (0..12).flat_map(|_| enc(DType::F16, rng.val())).collect();
        let mut c = vec![0u8; 15 * 4];
        reference_gemm(&d, 1.0, 0.0, &a, &b, &mut c);
        let mut acc = 0f32;
        for l in 0..4 {
            acc += decode(DType::F16, &a, l + 2 * 4) * decode(DType::F16, &b, l + 1 * 4);
        }
        assert_eq!(decode(DType::F32, &c, 2 + 1 * 5), acc);
    }
}
