//! Test-only SPIR-V interpreter for the native kernels: multiple workgroups, per-invocation
//! control flow, workgroup memory, barriers and 32-lane subgroup shuffles. It models exactly the
//! instruction subset `spv::Kb` emits.
use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{Op, OpenclStd100Op as C, StorageClass};
use std::collections::HashMap;

pub const GLOBAL: u64 = 0x10_0000_0000;
const SHARED: u64 = 0x20_0000_0000;
const PRIV: u64 = 0x30_0000_0000;
const BUILTIN: u64 = 0x40_0000_0000;
const PRIV_SPAN: u64 = 1 << 20;

#[derive(Clone, Copy, Debug)]
pub enum ArgVal {
    Ptr(u64),
    U32(u32),
    U64(u64),
    F32(f32),
}

#[derive(Default)]
pub struct Dev {
    pub mem: Vec<u8>,
}

impl Dev {
    pub fn alloc(&mut self, bytes: usize) -> u64 {
        let start = self.mem.len().next_multiple_of(256);
        self.mem.resize(start + bytes.max(1) + 256, 0);
        GLOBAL + start as u64
    }
    pub fn upload(&mut self, addr: u64, data: &[u8]) {
        let o = (addr - GLOBAL) as usize;
        self.mem[o..o + data.len()].copy_from_slice(data);
    }
    pub fn download(&self, addr: u64, n: usize) -> Vec<u8> {
        let o = (addr - GLOBAL) as usize;
        self.mem[o..o + n].to_vec()
    }
    pub fn put(&mut self, data: &[u8]) -> u64 {
        let a = self.alloc(data.len());
        self.upload(a, data);
        a
    }
}

type Val = (u64, u32);

fn id(op: &Operand) -> u32 {
    match op {
        Operand::IdRef(i) => *i,
        o => panic!("expected id, got {o:?}"),
    }
}
fn lit(op: &Operand) -> u32 {
    match op {
        Operand::LiteralBit32(v) | Operand::LiteralExtInstInteger(v) => *v,
        o => panic!("expected literal, got {o:?}"),
    }
}
fn mask(w: u32, v: u64) -> u64 {
    if w >= 64 { v } else { v & ((1u64 << w) - 1) }
}
fn sext(w: u32, v: u64) -> i64 {
    if w >= 64 { v as i64 } else { ((v << (64 - w)) as i64) >> (64 - w) }
}
pub fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as u32);
    let mag = match e {
        0 => m as f32 * 2f32.powi(-24),
        0x1f => {
            if m == 0 { f32::INFINITY } else { f32::NAN }
        }
        _ => (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15),
    };
    if s == 1 { -mag } else { mag }
}
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let mant = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut half = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rem > halfway || (rem == halfway && half & 1 == 1) {
            half += 1;
        }
        return sign | half as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

struct Ctx<'a> {
    module: &'a Module,
    widths: HashMap<u32, u32>,
    elem_bytes: HashMap<u32, u64>,
    consts: HashMap<u32, Val>,
    blocks: HashMap<u32, usize>,
    builtin_kind: HashMap<u32, usize>,
    shared_var: Option<u32>,
}

#[derive(PartialEq)]
enum Status {
    Running,
    AtBarrier,
    AtShuffle(u32),
    Done,
}

struct Inv {
    vals: HashMap<u32, Val>,
    vecs: HashMap<u32, [u64; 3]>,
    block: usize,
    ip: usize,
    status: Status,
    index: u64,
    wg: [u32; 3],
    local: [u32; 3],
    groups: [u32; 3],
    shuffle_result: Option<u64>,
    steps: u64,
}

struct Mem<'a> {
    global: &'a mut Vec<u8>,
    shared: Vec<u8>,
    private: Vec<Vec<u8>>,
}

impl Mem<'_> {
    fn slot(&mut self, addr: u64, inv: u64, len: usize) -> &mut [u8] {
        if addr >= PRIV {
            let off = ((addr - PRIV) % PRIV_SPAN) as usize;
            let owner = ((addr - PRIV) / PRIV_SPAN) as usize;
            let _ = inv;
            &mut self.private[owner][off..off + len]
        } else if addr >= SHARED {
            let off = (addr - SHARED) as usize;
            &mut self.shared[off..off + len]
        } else {
            assert!(addr >= GLOBAL, "wild pointer {addr:#x}");
            let off = (addr - GLOBAL) as usize;
            assert!(off + len <= self.global.len(), "global access out of range: {addr:#x}+{len}");
            &mut self.global[off..off + len]
        }
    }
    fn read(&mut self, addr: u64, inv: u64, bytes: usize) -> u64 {
        let s = self.slot(addr, inv, bytes);
        s.iter().enumerate().fold(0u64, |a, (i, b)| a | (u64::from(*b) << (8 * i)))
    }
    fn write(&mut self, addr: u64, inv: u64, bytes: usize, v: u64) {
        let s = self.slot(addr, inv, bytes);
        for (i, b) in s.iter_mut().enumerate() {
            *b = (v >> (8 * i)) as u8;
        }
    }
}

pub fn run(dev: &mut Dev, words: &[u32], args: &[ArgVal], grid: [u32; 3], block: [u32; 3]) {
    let module = rspirv::dr::load_words(words).expect("module decodes");
    let mut ctx = Ctx {
        module: &module,
        widths: HashMap::new(),
        elem_bytes: HashMap::new(),
        consts: HashMap::new(),
        blocks: HashMap::new(),
        builtin_kind: HashMap::new(),
        shared_var: None,
    };
    let mut pointees: HashMap<u32, u32> = HashMap::new();
    for inst in &module.types_global_values {
        let r = inst.result_id;
        match inst.class.opcode {
            Op::TypeInt | Op::TypeFloat => {
                ctx.widths.insert(r.unwrap(), lit(&inst.operands[0]));
            }
            Op::TypeBool => {
                ctx.widths.insert(r.unwrap(), 1);
            }
            Op::TypePointer => {
                ctx.widths.insert(r.unwrap(), 64);
                pointees.insert(r.unwrap(), id(&inst.operands[1]));
            }
            Op::TypeArray => {
                ctx.widths.insert(r.unwrap(), 0);
                ctx.elem_bytes.insert(r.unwrap(), u64::from(ctx.widths[&id(&inst.operands[0])] / 8));
            }
            Op::TypeVector => {
                ctx.widths.insert(r.unwrap(), 0);
            }
            _ => {}
        }
    }
    for inst in &module.types_global_values {
        match inst.class.opcode {
            Op::Constant => {
                let w = ctx.widths[&inst.result_type.unwrap()];
                let v = match inst.operands[0] {
                    Operand::LiteralBit32(v) => u64::from(v),
                    Operand::LiteralBit64(v) => v,
                    _ => unreachable!(),
                };
                ctx.consts.insert(inst.result_id.unwrap(), (mask(w, v), w));
            }
            Op::Variable => {
                if let Some(Operand::StorageClass(sc)) = inst.operands.first() {
                    if *sc == StorageClass::Workgroup {
                        ctx.shared_var = inst.result_id;
                        ctx.consts.insert(inst.result_id.unwrap(), (SHARED, 64));
                    }
                }
            }
            _ => {}
        }
    }
    // Pointer-to-array element size for access chains on the workgroup array.
    let shared_elem = ctx
        .shared_var
        .map(|v| {
            let ptr_ty = module.types_global_values.iter().find(|i| i.result_id == Some(v)).unwrap().result_type.unwrap();
            ctx.elem_bytes[&pointees[&ptr_ty]]
        })
        .unwrap_or(4);
    let mut kinds: HashMap<u32, rspirv::spirv::BuiltIn> = HashMap::new();
    for inst in &module.annotations {
        if inst.class.opcode == Op::Decorate
            && let [Operand::IdRef(t), Operand::Decoration(rspirv::spirv::Decoration::BuiltIn), Operand::BuiltIn(k)] = inst.operands[..]
        {
            kinds.insert(t, k);
        }
    }
    for (i, inst) in module.types_global_values.iter().enumerate() {
        let _ = i;
        if inst.class.opcode == Op::Variable
            && let Some(Operand::StorageClass(StorageClass::Input)) = inst.operands.first()
        {
            let v = inst.result_id.unwrap();
            let which = match kinds[&v] {
                rspirv::spirv::BuiltIn::WorkgroupId => 0,
                rspirv::spirv::BuiltIn::LocalInvocationId => 1,
                _ => 2,
            };
            ctx.builtin_kind.insert(v, which);
            ctx.consts.insert(v, (BUILTIN + which as u64, 64));
        }
    }
    let f = &module.functions[0];
    for (i, b) in f.blocks.iter().enumerate() {
        ctx.blocks.insert(b.label.as_ref().unwrap().result_id.unwrap(), i);
    }
    let n = (block[0] * block[1] * block[2]) as usize;
    for gz in 0..grid[2] {
        for gy in 0..grid[1] {
            for gx in 0..grid[0] {
                let mut mem = Mem {
                    global: &mut dev.mem,
                    shared: vec![0u8; 1 << 17],
                    private: (0..n).map(|_| vec![0u8; PRIV_SPAN as usize / 16]).collect(),
                };
                let mut invs: Vec<Inv> = (0..n)
                    .map(|t| {
                        let t = t as u32;
                        let mut vals = ctx.consts.clone();
                        for (p, a) in f.parameters.iter().zip(args) {
                            let w = ctx.widths[&p.result_type.unwrap()];
                            let bits = match *a {
                                ArgVal::Ptr(p) => p,
                                ArgVal::U32(v) => u64::from(v),
                                ArgVal::U64(v) => v,
                                ArgVal::F32(v) => u64::from(v.to_bits()),
                            };
                            vals.insert(p.result_id.unwrap(), (bits, w));
                        }
                        let mut slot = 0u64;
                        for blk in &f.blocks {
                            for i in &blk.instructions {
                                if i.class.opcode == Op::Variable {
                                    vals.insert(i.result_id.unwrap(), (PRIV + u64::from(t) * PRIV_SPAN + slot, 64));
                                    slot += 16;
                                }
                            }
                        }
                        Inv {
                            vals,
                            vecs: HashMap::new(),
                            block: 0,
                            ip: 0,
                            status: Status::Running,
                            index: u64::from(t),
                            wg: [gx, gy, gz],
                            local: [t % block[0], (t / block[0]) % block[1], t / (block[0] * block[1])],
                            groups: grid,
                            shuffle_result: None,
                            steps: 0,
                        }
                    })
                    .collect();
                loop {
                    for inv in invs.iter_mut() {
                        if inv.status == Status::Running {
                            exec(&ctx, inv, &mut mem, shared_elem, &pointees);
                        }
                    }
                    if invs.iter().all(|i| i.status == Status::Done) {
                        break;
                    }
                    // Release a barrier only when every live invocation has reached it.
                    let live: Vec<usize> = (0..n).filter(|&i| invs[i].status != Status::Done).collect();
                    if !live.is_empty() && live.iter().all(|&i| invs[i].status == Status::AtBarrier) {
                        for &i in &live {
                            invs[i].status = Status::Running;
                        }
                        continue;
                    }
                    // Subgroup shuffles: complete per 32-lane subgroup.
                    let mut progressed = false;
                    for sg in 0..n.div_ceil(32) {
                        let members: Vec<usize> = (sg * 32..((sg + 1) * 32).min(n)).collect();
                        let waiting: Vec<usize> = members.iter().copied().filter(|&i| matches!(invs[i].status, Status::AtShuffle(_))).collect();
                        if waiting.is_empty() {
                            continue;
                        }
                        let others_ready = members.iter().all(|&i| matches!(invs[i].status, Status::AtShuffle(_) | Status::Done));
                        if !others_ready {
                            continue;
                        }
                        let results: Vec<(usize, u64)> = waiting
                            .iter()
                            .map(|&i| {
                                let Status::AtShuffle(inst_idx) = invs[i].status else { unreachable!() };
                                let inst = ctx.module.functions[0].blocks[invs[i].block].instructions[inst_idx as usize].clone();
                                let mask_v = invs[i].vals[&id(&inst.operands[2])].0 as usize;
                                let src = (i % 32) ^ mask_v;
                                let src_idx = sg * 32 + src;
                                let value = invs[src_idx].vals[&id(&inst.operands[1])];
                                (i, value.0)
                            })
                            .collect();
                        for (i, v) in results {
                            invs[i].shuffle_result = Some(v);
                            invs[i].status = Status::Running;
                        }
                        progressed = true;
                    }
                    assert!(progressed, "deadlock: barrier/shuffle divergence");
                }
            }
        }
    }
}

fn exec(ctx: &Ctx, inv: &mut Inv, mem: &mut Mem, shared_elem: u64, pointees: &HashMap<u32, u32>) {
    let f = &ctx.module.functions[0];
    loop {
        let block = &f.blocks[inv.block];
        let inst = &block.instructions[inv.ip];
        inv.steps += 1;
        assert!(inv.steps < 20_000_000, "interpreter step limit");
        inv.ip += 1;
        match step(ctx, inv, mem, inst, shared_elem, pointees) {
            Flow::Next => {}
            Flow::Jump(label) => {
                inv.block = ctx.blocks[&label];
                inv.ip = 0;
            }
            Flow::Barrier => {
                inv.status = Status::AtBarrier;
                return;
            }
            Flow::Shuffle => {
                inv.status = Status::AtShuffle(inv.ip as u32);
                return;
            }
            Flow::Return => {
                inv.status = Status::Done;
                return;
            }
        }
    }
}

enum Flow {
    Next,
    Jump(u32),
    Barrier,
    Shuffle,
    Return,
}

fn step(ctx: &Ctx, inv: &mut Inv, mem: &mut Mem, inst: &Instruction, shared_elem: u64, pointees: &HashMap<u32, u32>) -> Flow {
    let w = inst.result_type.map_or(0, |t| ctx.widths.get(&t).copied().unwrap_or(0));
    let get = |inv: &Inv, i: usize| -> Val {
        *inv.vals.get(&id(&inst.operands[i])).unwrap_or_else(|| panic!("undefined operand {i} of {:?}", inst.class.opcode))
    };
    let result: u64 = match inst.class.opcode {
        Op::Variable | Op::SelectionMerge | Op::LoopMerge | Op::Line | Op::NoLine | Op::Nop => return Flow::Next,
        Op::Branch => return Flow::Jump(id(&inst.operands[0])),
        Op::BranchConditional => {
            let c = get(inv, 0).0 != 0;
            return Flow::Jump(id(&inst.operands[if c { 1 } else { 2 }]));
        }
        Op::Return => return Flow::Return,
        Op::ControlBarrier => return Flow::Barrier,
        Op::GroupNonUniformShuffleXor => {
            if let Some(r) = inv.shuffle_result.take() {
                inv.vals.insert(inst.result_id.unwrap(), (r, w));
                return Flow::Next;
            }
            inv.ip -= 1;
            return Flow::Shuffle;
        }
        Op::Load => {
            let addr = get(inv, 0).0;
            if addr >= BUILTIN && addr < BUILTIN + 8 {
                let which = (addr - BUILTIN) as usize;
                let v = match which {
                    0 => inv.wg.map(u64::from),
                    1 => inv.local.map(u64::from),
                    _ => inv.groups.map(u64::from),
                };
                inv.vecs.insert(inst.result_id.unwrap(), v);
                return Flow::Next;
            }
            let bytes = (w.max(8) / 8) as usize;
            mem.read(addr, inv.index, bytes)
        }
        Op::Store => {
            let addr = get(inv, 0).0;
            let (v, vw) = get(inv, 1);
            mem.write(addr, inv.index, (vw.max(8) / 8) as usize, v);
            return Flow::Next;
        }
        Op::CompositeExtract => {
            let v = inv.vecs[&id(&inst.operands[0])];
            v[lit(&inst.operands[1]) as usize]
        }
        Op::ConvertUToPtr | Op::ConvertPtrToU | Op::Bitcast | Op::CopyObject => get(inv, 0).0,
        Op::InBoundsAccessChain | Op::AccessChain => {
            // Only the workgroup array: base + index * element size.
            let base = get(inv, 0).0;
            let index = get(inv, 1).0;
            let _ = pointees;
            base + index * shared_elem
        }
        Op::UConvert => mask(w, get(inv, 0).0),
        Op::SConvert => {
            let (v, sw) = get(inv, 0);
            mask(w, sext(sw, v) as u64)
        }
        Op::IAdd => mask(w, get(inv, 0).0.wrapping_add(get(inv, 1).0)),
        Op::ISub => mask(w, get(inv, 0).0.wrapping_sub(get(inv, 1).0)),
        Op::IMul => mask(w, get(inv, 0).0.wrapping_mul(get(inv, 1).0)),
        Op::UDiv => mask(w, get(inv, 0).0 / get(inv, 1).0),
        Op::UMod => mask(w, get(inv, 0).0 % get(inv, 1).0),
        Op::ShiftLeftLogical => mask(w, get(inv, 0).0 << get(inv, 1).0),
        Op::ShiftRightLogical => get(inv, 0).0 >> get(inv, 1).0,
        Op::ShiftRightArithmetic => {
            let (v, sw) = get(inv, 0);
            mask(w, (sext(sw, v) >> get(inv, 1).0) as u64)
        }
        Op::BitwiseAnd => get(inv, 0).0 & get(inv, 1).0,
        Op::BitwiseOr => get(inv, 0).0 | get(inv, 1).0,
        Op::BitwiseXor => get(inv, 0).0 ^ get(inv, 1).0,
        Op::Not => mask(w, !get(inv, 0).0),
        Op::LogicalAnd => get(inv, 0).0 & get(inv, 1).0,
        Op::LogicalOr => get(inv, 0).0 | get(inv, 1).0,
        Op::LogicalNot => get(inv, 0).0 ^ 1,
        Op::Select => {
            if get(inv, 0).0 != 0 { get(inv, 1).0 } else { get(inv, 2).0 }
        }
        Op::IEqual => u64::from(get(inv, 0).0 == get(inv, 1).0),
        Op::INotEqual => u64::from(get(inv, 0).0 != get(inv, 1).0),
        Op::ULessThan => u64::from(get(inv, 0).0 < get(inv, 1).0),
        Op::ULessThanEqual => u64::from(get(inv, 0).0 <= get(inv, 1).0),
        Op::UGreaterThan => u64::from(get(inv, 0).0 > get(inv, 1).0),
        Op::UGreaterThanEqual => u64::from(get(inv, 0).0 >= get(inv, 1).0),
        Op::FOrdGreaterThan => u64::from(f32::from_bits(get(inv, 0).0 as u32) > f32::from_bits(get(inv, 1).0 as u32)),
        Op::FOrdEqual => u64::from(f32::from_bits(get(inv, 0).0 as u32) == f32::from_bits(get(inv, 1).0 as u32)),
        Op::FOrdLessThan => u64::from(f32::from_bits(get(inv, 0).0 as u32) < f32::from_bits(get(inv, 1).0 as u32)),
        Op::FAdd | Op::FSub | Op::FMul | Op::FDiv => {
            let (a, b) = (f32::from_bits(get(inv, 0).0 as u32), f32::from_bits(get(inv, 1).0 as u32));
            u64::from(
                match inst.class.opcode {
                    Op::FAdd => a + b,
                    Op::FSub => a - b,
                    Op::FMul => a * b,
                    _ => a / b,
                }
                .to_bits(),
            )
        }
        Op::FNegate => u64::from((-f32::from_bits(get(inv, 0).0 as u32)).to_bits()),
        Op::ConvertSToF => u64::from((sext(get(inv, 0).1, get(inv, 0).0) as f32).to_bits()),
        Op::ConvertFToS => {
            let v = f32::from_bits(get(inv, 0).0 as u32).trunc();
            mask(w, (v as i64) as u64)
        }
        Op::FConvert => {
            let (v, sw) = get(inv, 0);
            if sw == 16 {
                u64::from(f16_to_f32(v as u16).to_bits())
            } else {
                u64::from(f32_to_f16(f32::from_bits(v as u32)))
            }
        }
        Op::SDot => {
            let (a, b) = (get(inv, 0).0 as u32, get(inv, 1).0 as u32);
            let mut s = 0i32;
            for k in 0..4 {
                s += i32::from((a >> (8 * k)) as u8 as i8) * i32::from((b >> (8 * k)) as u8 as i8);
            }
            mask(w, s as u32 as u64)
        }
        Op::ExtInst => {
            let op = lit(&inst.operands[1]);
            let arg = |i: usize| f32::from_bits(get(inv, 2 + i).0 as u32);
            let out = if op == C::fabs as u32 {
                arg(0).abs()
            } else if op == C::fmax as u32 {
                let (a, b) = (arg(0), arg(1));
                if a.is_nan() { b } else if b.is_nan() { a } else { a.max(b) }
            } else if op == C::round as u32 {
                arg(0).round()
            } else if op == C::fma as u32 {
                arg(0).mul_add(arg(1), arg(2))
            } else {
                panic!("unsupported OpenCL.std instruction {op}")
            };
            u64::from(out.to_bits())
        }
        other => panic!("interpreter does not support {other:?}"),
    };
    inv.vals.insert(inst.result_id.unwrap(), (result, w));
    Flow::Next
}
