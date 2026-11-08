//! A multi-threaded SPIR-V (OpenCL kernel flavour) simulator used to run real GGML kernels
//! offline. Unlike the lockstep interpreter in `coverage_tests`, every invocation owns a program
//! counter, so divergent control flow, workgroup barriers, subgroup collectives (shuffle, vote,
//! ballot, reduce), atomics, shared memory, half precision and the integer dot-product
//! instructions behave as the SPIR-V specification describes. Global accesses outside the
//! allocated buffers are reported, which turns buffer over-reads into test failures.
use rspirv::dr::{Instruction, Module, Operand};
use rspirv::spirv::{BuiltIn, Op, StorageClass};
use std::collections::HashMap;

pub const GLOBAL_BASE: u64 = 0x4000_0000;
pub const SHARED_BASE: u64 = 0x2000_0000;
const PRIVATE_BASE: u64 = 0xF000_0000;
const PAGE: u64 = 4096;

#[derive(Default)]
pub struct Mem {
    pages: HashMap<u64, Box<[u8; PAGE as usize]>>,
}

impl Mem {
    pub fn read(&self, addr: u64, bytes: u32) -> u64 {
        let mut v = 0u64;
        for i in 0..u64::from(bytes) {
            let a = addr + i;
            let b = self.pages.get(&(a / PAGE)).map_or(0, |p| p[(a % PAGE) as usize]);
            v |= u64::from(b) << (8 * i);
        }
        v
    }
    pub fn write(&mut self, addr: u64, bytes: u32, value: u64) {
        for i in 0..u64::from(bytes) {
            let a = addr + i;
            let p = self.pages.entry(a / PAGE).or_insert_with(|| Box::new([0; PAGE as usize]));
            p[(a % PAGE) as usize] = (value >> (8 * i)) as u8;
        }
    }
}

/// Global memory with a bump allocator; accesses outside live allocations are recorded.
pub struct Gpu {
    pub mem: Mem,
    regions: Vec<(u64, u64)>,
    next: u64,
    pub violations: std::cell::RefCell<Vec<String>>,
}

impl Default for Gpu {
    fn default() -> Self {
        Self { mem: Mem::default(), regions: Vec::new(), next: GLOBAL_BASE, violations: Default::default() }
    }
}

impl Gpu {
    /// Allocates `bytes` (zeroed) with a 256-byte gap after it, so over-reads are visible.
    pub fn alloc(&mut self, bytes: usize) -> u64 {
        let base = self.next;
        self.regions.push((base, bytes as u64));
        self.next = (base + bytes as u64 + 256 + 255) & !255;
        base
    }
    pub fn upload(&mut self, data: &[u8]) -> u64 {
        let base = self.alloc(data.len());
        for (i, b) in data.iter().enumerate() {
            self.mem.write(base + i as u64, 1, u64::from(*b));
        }
        base
    }
    pub fn download(&self, addr: u64, bytes: usize) -> Vec<u8> {
        (0..bytes as u64).map(|i| self.mem.read(addr + i, 1) as u8).collect()
    }
    pub fn download_f32(&self, addr: u64, count: usize) -> Vec<f32> {
        (0..count as u64).map(|i| f32::from_bits(self.mem.read(addr + 4 * i, 4) as u32)).collect()
    }
    fn check(&self, addr: u64, bytes: u32, what: &str) {
        if !(GLOBAL_BASE..PRIVATE_BASE).contains(&addr) {
            return;
        }
        let end = addr + u64::from(bytes);
        let ok = self.regions.iter().any(|&(s, l)| addr >= s && end <= s + l);
        if !ok {
            let mut v = self.violations.borrow_mut();
            if v.len() < 8 {
                v.push(format!("{what} of {bytes} bytes at {addr:#x} outside allocations"));
            }
        }
    }
}

pub struct Launch {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Flattened kernel arguments, one entry per SPIR-V function parameter.
    pub args: Vec<u64>,
    /// Bytes of dynamic (extern) shared memory already part of the module; informational.
    pub max_steps: u64,
}

impl Launch {
    pub fn new(grid: [u32; 3], block: [u32; 3], args: Vec<u64>) -> Self {
        Self { grid, block, args, max_steps: 400_000_000 }
    }
}

struct Ins {
    op: Op,
    res: u32,
    w: u32,
    /// Raw operands: ids, 32-bit literals; everything else 0.
    ops: Vec<u32>,
    /// Second operand literal for 64-bit constants.
    stride: u64,
    decorated_rounding: Option<rspirv::spirv::FPRoundingMode>,
}

pub struct Sim {
    blocks: Vec<Vec<Ins>>,
    label_index: HashMap<u32, usize>,
    /// Static bit width of every id (pointers 64, bool 1); vectors report the element width.
    vw: Vec<u32>,
    vlen: Vec<u32>,
    constants: Vec<(u32, u64)>,
    params: Vec<u32>,
    shared_vars: Vec<(u32, u64)>,
    builtins: HashMap<u32, BuiltIn>,
    function_vars: Vec<(u32, u32)>,
    bound: usize,
    global_vars: Vec<(u32, Vec<u8>)>,
    entry_label: u32,
    placed: Vec<(u32, u64)>,
    pub shared_bytes: u64,
    pub opcode_counts: HashMap<String, u64>,
    pub count_ops: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum State {
    Run,
    Barrier,
    Collective,
    Done,
}

struct Thread {
    vals: Vec<u64>,
    vecs: HashMap<u32, Vec<u64>>,
    block: usize,
    pc: usize,
    state: State,
    tid: [u32; 3],
    linear: u32,
    private: u64,
    label: u32,
}

fn id_of(o: &Operand) -> u32 {
    match o {
        Operand::IdRef(i) | Operand::IdScope(i) | Operand::IdMemorySemantics(i) => *i,
        Operand::LiteralBit32(v) | Operand::LiteralExtInstInteger(v) => *v,
        Operand::PackedVectorFormat(_) => 0,
        _ => 0,
    }
}

fn mask(w: u32, v: u64) -> u64 {
    if w >= 64 { v } else { v & ((1u64 << w) - 1) }
}
fn sext(w: u32, v: u64) -> i64 {
    if w >= 64 || w == 0 { v as i64 } else { ((v << (64 - w)) as i64) >> (64 - w) }
}

pub fn f16_to_f32(h: u32) -> f32 {
    let (sign, exp, man) = (h >> 15 & 1, (h >> 10) & 0x1f, h & 0x3ff);
    let magnitude = match exp {
        0 => man as f32 * 2f32.powi(-24),
        0x1f => {
            if man == 0 { f32::INFINITY } else { f32::NAN }
        }
        _ => (1.0 + man as f32 / 1024.0) * 2f32.powi(exp as i32 - 15),
    };
    if sign == 1 { -magnitude } else { magnitude }
}

fn fv(w: u32, bits: u64) -> f64 {
    match w {
        16 => f64::from(f16_to_f32(bits as u32)),
        32 => f64::from(f32::from_bits(bits as u32)),
        _ => f64::from_bits(bits),
    }
}
fn fb(w: u32, value: f64) -> u64 {
    match w {
        16 => u64::from(super::coverage_tests::interp::f32_to_f16_bits(value as f32)),
        32 => u64::from((value as f32).to_bits()),
        _ => value.to_bits(),
    }
}

impl Sim {
    pub fn new(module: &Module) -> Self {
        let bound = module.header.as_ref().map_or(4096, |h| h.bound as usize) + 8;
        let mut type_w: HashMap<u32, u32> = HashMap::new();
        let mut type_len: HashMap<u32, u32> = HashMap::new();
        let mut vw = vec![0u32; bound];
        let mut vlen = vec![1u32; bound];
        for inst in &module.types_global_values {
            let Some(rid) = inst.result_id else { continue };
            match inst.class.opcode {
                Op::TypeInt | Op::TypeFloat => {
                    if let Operand::LiteralBit32(w) = inst.operands[0] {
                        type_w.insert(rid, w);
                    }
                }
                Op::TypeBool => {
                    type_w.insert(rid, 1);
                }
                Op::TypePointer => {
                    type_w.insert(rid, 64);
                }
                Op::TypeVector => {
                    type_w.insert(rid, type_w[&id_of(&inst.operands[0])]);
                    type_len.insert(rid, id_of(&inst.operands[1]));
                }
                _ => {}
            }
        }
        let mut rounding: HashMap<u32, rspirv::spirv::FPRoundingMode> = HashMap::new();
        let mut builtins = HashMap::new();
        for inst in &module.annotations {
            if inst.class.opcode == Op::Decorate {
                match inst.operands[..] {
                    [Operand::IdRef(t), Operand::Decoration(rspirv::spirv::Decoration::BuiltIn), Operand::BuiltIn(k)] => {
                        builtins.insert(t, k);
                    }
                    [Operand::IdRef(t), Operand::Decoration(rspirv::spirv::Decoration::FPRoundingMode), Operand::FPRoundingMode(m)] => {
                        rounding.insert(t, m);
                    }
                    _ => {}
                }
            }
        }
        let mut constants = Vec::new();
        let mut shared_vars = Vec::new();
        let mut shared_bytes = 0u64;
        let mut array_bytes: HashMap<u32, u64> = HashMap::new();
        let mut const_val: HashMap<u32, u64> = HashMap::new();
        let mut ptr_pointee: HashMap<u32, u32> = HashMap::new();
        let mut composites: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut global_vars: Vec<(u32, Vec<u8>)> = Vec::new();
        for inst in &module.types_global_values {
            let Some(rid) = inst.result_id else { continue };
            match inst.class.opcode {
                Op::Constant => {
                    let t = inst.result_type.unwrap();
                    let w = type_w[&t];
                    let v = match inst.operands[0] {
                        Operand::LiteralBit32(v) => u64::from(v),
                        Operand::LiteralBit64(v) => v,
                        _ => unreachable!(),
                    };
                    constants.push((rid, mask(w, v)));
                    const_val.insert(rid, mask(w, v));
                    vw[rid as usize] = w;
                }
                Op::ConstantTrue | Op::ConstantFalse => {
                    constants.push((rid, u64::from(inst.class.opcode == Op::ConstantTrue)));
                    vw[rid as usize] = 1;
                }
                Op::ConstantNull => {
                    constants.push((rid, 0));
                    vw[rid as usize] = type_w.get(&inst.result_type.unwrap()).copied().unwrap_or(64);
                }
                Op::TypePointer => {
                    ptr_pointee.insert(rid, id_of(&inst.operands[1]));
                }
                Op::TypeArray => {
                    let elem = id_of(&inst.operands[0]);
                    let n = const_val[&id_of(&inst.operands[1])];
                    let ew = array_bytes.get(&elem).copied().unwrap_or_else(|| u64::from(type_w[&elem]) / 8);
                    array_bytes.insert(rid, ew * n);
                }
                Op::ConstantComposite => {
                    composites.insert(rid, inst.operands.iter().map(id_of).collect());
                }
                Op::Variable => {
                    let t = inst.result_type.unwrap();
                    vw[rid as usize] = 64;
                    if let Some(Operand::StorageClass(StorageClass::CrossWorkgroup)) = inst.operands.first() {
                        let pointee = ptr_pointee[&t];
                        let bytes = array_bytes.get(&pointee).copied().unwrap_or(4);
                        let mut data = vec![0u8; bytes as usize];
                        if let Some(init) = inst.operands.get(1).map(id_of)
                            && let Some(words) = composites.get(&init)
                        {
                            for (i, w) in words.iter().enumerate() {
                                let v = const_val.get(w).copied().unwrap_or(0) as u32;
                                data[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
                            }
                        }
                        global_vars.push((rid, data));
                    }
                    if let Some(Operand::StorageClass(StorageClass::Workgroup)) = inst.operands.first() {
                        let pointee = ptr_pointee[&t];
                        let bytes = array_bytes.get(&pointee).copied().unwrap_or_else(|| u64::from(type_w.get(&pointee).copied().unwrap_or(32)) / 8);
                        shared_vars.push((rid, shared_bytes));
                        shared_bytes += (bytes + 255) & !255;
                    }
                }
                _ => {}
            }
        }
        // Element stride of every pointer type, for PtrAccessChain.
        let mut ptr_stride: HashMap<u32, u64> = HashMap::new();
        for (&pt, &pointee) in &ptr_pointee {
            let bytes = array_bytes
                .get(&pointee)
                .copied()
                .unwrap_or_else(|| u64::from(type_w.get(&pointee).copied().unwrap_or(8)).max(8) / 8
                    * u64::from(type_len.get(&pointee).copied().unwrap_or(1)));
            ptr_stride.insert(pt, bytes);
        }
        let function = &module.functions[0];
        let mut params = Vec::new();
        for p in &function.parameters {
            let id = p.result_id.unwrap();
            vw[id as usize] = type_w[&p.result_type.unwrap()];
            params.push(id);
        }
        let mut label_index = HashMap::new();
        let mut blocks = Vec::new();
        let mut function_vars = Vec::new();
        for (bi, b) in function.blocks.iter().enumerate() {
            label_index.insert(b.label.as_ref().unwrap().result_id.unwrap(), bi);
            let mut list = Vec::new();
            for inst in &b.instructions {
                if inst.class.opcode == Op::Variable {
                    let id = inst.result_id.unwrap();
                    vw[id as usize] = 64;
                    let pointee = ptr_pointee[&inst.result_type.unwrap()];
                    let bytes = array_bytes.get(&pointee).copied().unwrap_or_else(|| u64::from(type_w.get(&pointee).copied().unwrap_or(64)) / 8);
                    function_vars.push((id, bytes.max(8) as u32));
                    continue;
                }
                list.push(Self::decode(inst, &type_w, &type_len, &mut vw, &mut vlen, &rounding, &ptr_stride));
            }
            blocks.push(list);
        }
        Self {
            blocks,
            label_index,
            vw,
            vlen,
            constants,
            params,
            shared_vars,
            builtins,
            function_vars,
            bound,
            global_vars,
            placed: Vec::new(),
            entry_label: function.blocks[0].label.as_ref().unwrap().result_id.unwrap(),
            shared_bytes,
            opcode_counts: HashMap::new(),
            count_ops: false,
        }
    }

    fn decode(
        inst: &Instruction,
        type_w: &HashMap<u32, u32>,
        type_len: &HashMap<u32, u32>,
        vw: &mut [u32],
        vlen: &mut [u32],
        rounding: &HashMap<u32, rspirv::spirv::FPRoundingMode>,
        ptr_stride: &HashMap<u32, u64>,
    ) -> Ins {
        let w = inst.result_type.map_or(0, |t| type_w.get(&t).copied().unwrap_or(0));
        let res = inst.result_id.unwrap_or(0);
        if res != 0 {
            vw[res as usize] = w;
            if let Some(t) = inst.result_type {
                vlen[res as usize] = type_len.get(&t).copied().unwrap_or(1);
            }
        }
        let stride = inst
            .result_type
            .and_then(|t| ptr_stride.get(&t).copied())
            .unwrap_or(1);
        Ins {
            op: inst.class.opcode,
            res,
            w,
            ops: inst.operands.iter().map(id_of).collect(),
            stride,
            decorated_rounding: rounding.get(&res).copied(),
        }
    }

    /// Runs every workgroup of the launch; returns Err with a description on a simulation fault.
    pub fn run(&mut self, gpu: &mut Gpu, launch: &Launch) -> Result<(), String> {
        let mut steps = 0u64;
        self.placed = self
            .global_vars
            .iter()
            .map(|(id, data)| (*id, gpu.upload(data)))
            .collect();
        for z in 0..launch.grid[2] {
            for y in 0..launch.grid[1] {
                for x in 0..launch.grid[0] {
                    self.run_group(gpu, launch, [x, y, z], &mut steps)?;
                }
            }
        }
        let v = gpu.violations.borrow();
        if !v.is_empty() {
            return Err(format!("memory violations: {:?}", *v));
        }
        Ok(())
    }

    fn run_group(&mut self, gpu: &mut Gpu, launch: &Launch, group: [u32; 3], steps: &mut u64) -> Result<(), String> {
        let n = (launch.block[0] * launch.block[1] * launch.block[2]) as usize;
        let mut shared = Mem::default();
        let mut priv_mem = Mem::default();
        let mut threads: Vec<Thread> = Vec::with_capacity(n);
        for linear in 0..n as u32 {
            let tid = [
                linear % launch.block[0],
                (linear / launch.block[0]) % launch.block[1],
                linear / (launch.block[0] * launch.block[1]),
            ];
            let mut vals = vec![0u64; self.bound];
            for &(id, v) in &self.constants {
                vals[id as usize] = v;
            }
            for (i, &p) in self.params.iter().enumerate() {
                vals[p as usize] = launch.args[i];
            }
            for &(id, addr) in &self.placed {
                vals[id as usize] = addr;
            }
            for &(id, off) in &self.shared_vars {
                vals[id as usize] = SHARED_BASE + off;
            }
            let private = PRIVATE_BASE + (u64::from(linear) << 28);
            let mut slot = private;
            for &(id, bytes) in &self.function_vars {
                vals[id as usize] = slot;
                slot += u64::from(bytes).next_multiple_of(16);
            }
            threads.push(Thread { vals, vecs: HashMap::new(), block: 0, pc: 0, state: State::Run, tid, linear, private, label: self.entry_label });
            let _ = threads.last().unwrap().private;
        }
        let ctx = Ctx { launch, group };
        loop {
            let mut progressed = false;
            for t in 0..n {
                while threads[t].state == State::Run {
                    self.step(&mut threads[t], gpu, &mut shared, &mut priv_mem, &ctx)?;
                    *steps += 1;
                    progressed = true;
                    if *steps > launch.max_steps {
                        return Err("simulator step limit".into());
                    }
                }
            }
            let _ = progressed;
            if threads.iter().all(|t| t.state == State::Done) {
                return Ok(());
            }
            // Everyone is blocked: resolve a barrier, then subgroup collectives.
            let at_barrier = threads.iter().filter(|t| t.state == State::Barrier).count();
            if at_barrier > 0 {
                if threads.iter().any(|t| t.state == State::Collective) {
                    return Err("workgroup barrier reached while other invocations wait in a subgroup operation".into());
                }
                // Every waiting invocation must be at the same barrier instruction: `bar.sync` with
                // some threads at a different barrier (another loop iteration's exit path, another
                // branch) is undefined behaviour in CUDA and a hang on hardware.
                let mut sites = threads.iter().filter(|t| t.state == State::Barrier).map(|t| (t.block, t.pc));
                let first_site = sites.next();
                if sites.any(|site| Some(site) != first_site) {
                    return Err("divergent barrier: invocations wait at different barrier instructions".into());
                }
                // Invocations that already returned are ignored (PTX bar.sync counts exited threads).
                for t in threads.iter_mut().filter(|t| t.state == State::Barrier) {
                    t.pc += 1;
                    t.state = State::Run;
                }
                continue;
            }
            self.resolve_collectives(&mut threads)?;
        }
    }

    fn resolve_collectives(&self, threads: &mut [Thread]) -> Result<(), String> {
        let n = threads.len();
        let sub = 32usize;
        for base in (0..n).step_by(sub) {
            let lanes: Vec<usize> = (base..(base + sub).min(n)).collect();
            let waiting: Vec<usize> = lanes.iter().copied().filter(|&l| threads[l].state == State::Collective).collect();
            if waiting.is_empty() {
                continue;
            }
            let first = &self.blocks[threads[waiting[0]].block][threads[waiting[0]].pc];
            for &l in &waiting {
                let i = &self.blocks[threads[l].block][threads[l].pc];
                if i.op != first.op || i.ops != first.ops {
                    return Err(format!("subgroup divergence: lanes execute different collectives ({:?} vs {:?})", first.op, i.op));
                }
            }
            let op = first.op;
            let ops = first.ops.clone();
            let w = first.w;
            let res = first.res;
            // Value operand index per op.
            let val = |l: usize, idx: usize| threads[l].vals[ops[idx] as usize];
            let lane_of = |l: usize| l - base;
            let mut out: Vec<(usize, u64)> = Vec::new();
            let mut vec_out: Vec<(usize, Vec<u64>)> = Vec::new();
            let participating: Vec<usize> = waiting.clone();
            match op {
                Op::GroupNonUniformShuffle
                | Op::GroupNonUniformShuffleXor
                | Op::GroupNonUniformShuffleUp
                | Op::GroupNonUniformShuffleDown => {
                    for &l in &waiting {
                        let d = val(l, 2) as usize;
                        let me = lane_of(l);
                        let src = match op {
                            Op::GroupNonUniformShuffle => d,
                            Op::GroupNonUniformShuffleXor => me ^ d,
                            Op::GroupNonUniformShuffleUp => me.wrapping_sub(d),
                            _ => me + d,
                        };
                        if src >= sub || base + src >= n || !participating.contains(&(base + src)) {
                            return Err(format!(
                                "{op:?}: lane {me} reads lane {src}, which is out of range or not executing the shuffle"
                            ));
                        }
                        out.push((l, val(base + src, 1)));
                    }
                }
                Op::GroupNonUniformBroadcast => {
                    let src = ops[2] as usize;
                    let src = threads[waiting[0]].vals[src] as usize;
                    for &l in &waiting {
                        out.push((l, val(base + src, 1)));
                    }
                }
                Op::GroupNonUniformBroadcastFirst => {
                    let v = val(waiting[0], 1);
                    for &l in &waiting {
                        out.push((l, v));
                    }
                }
                Op::GroupNonUniformElect => {
                    for &l in &waiting {
                        out.push((l, u64::from(l == waiting[0])));
                    }
                }
                Op::GroupNonUniformAll | Op::GroupNonUniformAny => {
                    let all = waiting.iter().all(|&l| val(l, 1) != 0);
                    let any = waiting.iter().any(|&l| val(l, 1) != 0);
                    for &l in &waiting {
                        out.push((l, u64::from(if op == Op::GroupNonUniformAll { all } else { any })));
                    }
                }
                Op::GroupNonUniformBallot => {
                    let mut m = 0u64;
                    for &l in &waiting {
                        if val(l, 1) != 0 {
                            m |= 1 << lane_of(l);
                        }
                    }
                    for &l in &waiting {
                        vec_out.push((l, vec![m & 0xffff_ffff, 0, 0, 0]));
                    }
                }
                Op::GroupNonUniformIAdd
                | Op::GroupNonUniformSMin
                | Op::GroupNonUniformUMin
                | Op::GroupNonUniformSMax
                | Op::GroupNonUniformUMax
                | Op::GroupNonUniformBitwiseAnd
                | Op::GroupNonUniformBitwiseOr
                | Op::GroupNonUniformBitwiseXor => {
                    // operands: scope, group op (0 = Reduce), value
                    let vals: Vec<u64> = waiting.iter().map(|&l| val(l, 2)).collect();
                    let reduce = |vals: &[u64]| -> u64 {
                        match op {
                            Op::GroupNonUniformIAdd => vals.iter().fold(0u64, |a, &b| mask(w, a.wrapping_add(b))),
                            Op::GroupNonUniformSMin => vals.iter().map(|&v| sext(w, v)).min().map(|v| mask(w, v as u64)).unwrap(),
                            Op::GroupNonUniformUMin => *vals.iter().min().unwrap(),
                            Op::GroupNonUniformSMax => vals.iter().map(|&v| sext(w, v)).max().map(|v| mask(w, v as u64)).unwrap(),
                            Op::GroupNonUniformUMax => *vals.iter().max().unwrap(),
                            Op::GroupNonUniformBitwiseAnd => vals.iter().fold(u64::MAX, |a, &b| a & b),
                            Op::GroupNonUniformBitwiseOr => vals.iter().fold(0, |a, &b| a | b),
                            _ => vals.iter().fold(0, |a, &b| a ^ b),
                        }
                    };
                    if ops[1] != 0 {
                        return Err("only Reduce group operations are modelled".into());
                    }
                    let r = reduce(&vals);
                    for &l in &waiting {
                        out.push((l, r));
                    }
                }
                other => return Err(format!("unmodelled collective {other:?}")),
            }
            for (l, v) in out {
                threads[l].vals[res as usize] = v;
                threads[l].pc += 1;
                threads[l].state = State::Run;
            }
            for (l, v) in vec_out {
                threads[l].vecs.insert(res, v);
                threads[l].pc += 1;
                threads[l].state = State::Run;
            }
        }
        Ok(())
    }

    fn step(&mut self, t: &mut Thread, gpu: &mut Gpu, shared: &mut Mem, priv_mem: &mut Mem, ctx: &Ctx) -> Result<(), String> {
        let ins = &self.blocks[t.block][t.pc];
        if self.count_ops {
            *self.opcode_counts.entry(format!("{:?}", ins.op)).or_default() += 1;
        }
        let w = ins.w;
        let o = &ins.ops;
        let vw = &self.vw;
        macro_rules! g {
            ($i:expr) => {
                t.vals[o[$i] as usize]
            };
        }
        let sw = |i: usize| vw[o[i] as usize];
        let mut next_pc = true;
        let result: Option<u64> = match ins.op {
            Op::Load => {
                let addr = g!(0);
                if self.vlen[ins.res as usize] > 1 {
                    let n = self.vlen[ins.res as usize];
                    if let Some(kind) = self.builtins.get(&o[0]) {
                        let v = match kind {
                            BuiltIn::LocalInvocationId => t.tid.map(u64::from),
                            BuiltIn::WorkgroupId => ctx.group.map(u64::from),
                            BuiltIn::WorkgroupSize => ctx.launch.block.map(u64::from),
                            BuiltIn::NumWorkgroups => ctx.launch.grid.map(u64::from),
                            BuiltIn::GlobalInvocationId => [0, 1, 2].map(|k| u64::from(ctx.group[k]) * u64::from(ctx.launch.block[k]) + u64::from(t.tid[k])),
                            other => return Err(format!("unsupported builtin {other:?}")),
                        };
                        t.vecs.insert(ins.res, v.to_vec());
                    } else {
                        let ew = w.max(8) / 8;
                        let v = (0..n).map(|k| Self::mem_read(gpu, shared, priv_mem, addr + u64::from(k * ew), ew)).collect();
                        t.vecs.insert(ins.res, v);
                    }
                    None
                } else {
                    Some(Self::mem_read(gpu, shared, priv_mem, addr, w.max(8) / 8))
                }
            }
            Op::Store => {
                let addr = g!(0);
                let bytes = sw(1).max(8) / 8;
                if self.vlen[o[1] as usize] > 1 {
                    let v = t.vecs.get(&o[1]).cloned().unwrap_or_default();
                    for (k, e) in v.iter().enumerate() {
                        Self::mem_write(gpu, shared, priv_mem, addr + (k as u64) * u64::from(bytes), bytes, *e);
                    }
                } else {
                    Self::mem_write(gpu, shared, priv_mem, addr, bytes, g!(1));
                }
                None
            }
            Op::AtomicFAddEXT => {
                let (addr, value) = (g!(0), g!(3));
                let bytes = w.max(8) / 8;
                let old = Self::mem_read(gpu, shared, priv_mem, addr, bytes);
                let new = if w == 32 {
                    u64::from((f32::from_bits(old as u32) + f32::from_bits(value as u32)).to_bits())
                } else {
                    (f64::from_bits(old) + f64::from_bits(value)).to_bits()
                };
                Self::mem_write(gpu, shared, priv_mem, addr, bytes, new);
                Some(old)
            }
            Op::AtomicIAdd | Op::AtomicISub | Op::AtomicExchange | Op::AtomicSMax | Op::AtomicUMax | Op::AtomicSMin
            | Op::AtomicUMin | Op::AtomicAnd | Op::AtomicOr | Op::AtomicXor => {
                let (addr, value) = (g!(0), g!(3));
                let bytes = w.max(8) / 8;
                let old = Self::mem_read(gpu, shared, priv_mem, addr, bytes);
                let new = match ins.op {
                    Op::AtomicIAdd => mask(w, old.wrapping_add(value)),
                    Op::AtomicISub => mask(w, old.wrapping_sub(value)),
                    Op::AtomicExchange => value,
                    Op::AtomicSMax => mask(w, sext(w, old).max(sext(w, value)) as u64),
                    Op::AtomicSMin => mask(w, sext(w, old).min(sext(w, value)) as u64),
                    Op::AtomicUMax => old.max(value),
                    Op::AtomicUMin => old.min(value),
                    Op::AtomicAnd => old & value,
                    Op::AtomicOr => old | value,
                    _ => old ^ value,
                };
                Self::mem_write(gpu, shared, priv_mem, addr, bytes, new);
                Some(old)
            }
            Op::AtomicCompareExchange => {
                let (addr, value, cmp) = (g!(0), g!(4), g!(5));
                let bytes = w.max(8) / 8;
                let old = Self::mem_read(gpu, shared, priv_mem, addr, bytes);
                if old == cmp {
                    Self::mem_write(gpu, shared, priv_mem, addr, bytes, value);
                }
                Some(old)
            }
            Op::AtomicLoad => Some(Self::mem_read(gpu, shared, priv_mem, g!(0), w.max(8) / 8)),
            Op::AtomicStore => {
                Self::mem_write(gpu, shared, priv_mem, g!(0), sw(3).max(8) / 8, g!(3));
                None
            }
            Op::ControlBarrier => {
                t.state = State::Barrier;
                next_pc = false;
                None
            }
            Op::MemoryBarrier | Op::SelectionMerge | Op::LoopMerge | Op::Nop | Op::Line | Op::NoLine => None,
            Op::Branch => {
                let target = o[0];
                self.enter(t, target);
                return Ok(());
            }
            Op::BranchConditional => {
                let cond = g!(0) != 0;
                let target = o[if cond { 1 } else { 2 }];
                self.enter(t, target);
                return Ok(());
            }
            Op::Switch => {
                let sel = g!(0);
                let mut target = o[1];
                // Remaining operands: (literal, label) pairs; 32-bit selectors only.
                let mut i = 2;
                while i + 1 < o.len() {
                    if u64::from(o[i]) == sel {
                        target = o[i + 1];
                        break;
                    }
                    i += 2;
                }
                self.enter(t, target);
                return Ok(());
            }
            Op::Return => {
                t.state = State::Done;
                return Ok(());
            }
            Op::Unreachable => return Err("reached OpUnreachable".into()),
            Op::GroupNonUniformShuffle
            | Op::GroupNonUniformShuffleXor
            | Op::GroupNonUniformShuffleUp
            | Op::GroupNonUniformShuffleDown
            | Op::GroupNonUniformBroadcast
            | Op::GroupNonUniformBroadcastFirst
            | Op::GroupNonUniformElect
            | Op::GroupNonUniformAll
            | Op::GroupNonUniformAny
            | Op::GroupNonUniformBallot
            | Op::GroupNonUniformIAdd
            | Op::GroupNonUniformSMin
            | Op::GroupNonUniformUMin
            | Op::GroupNonUniformSMax
            | Op::GroupNonUniformUMax
            | Op::GroupNonUniformBitwiseAnd
            | Op::GroupNonUniformBitwiseOr
            | Op::GroupNonUniformBitwiseXor => {
                t.state = State::Collective;
                next_pc = false;
                None
            }
            Op::Bitcast if self.vlen[ins.res as usize] > 1 => {
                let n = self.vlen[ins.res as usize];
                let src = g!(0);
                let v = (0..n).map(|k| mask(w, src >> (k * w))).collect();
                t.vecs.insert(ins.res, v);
                None
            }
            Op::Bitcast | Op::ConvertPtrToU | Op::ConvertUToPtr | Op::GenericCastToPtr | Op::PtrCastToGeneric | Op::CopyObject => {
                Some(mask(w.max(1), g!(0)))
            }
            Op::UConvert => Some(mask(w, g!(0))),
            Op::SConvert => Some(mask(w, sext(sw(0), g!(0)) as u64)),
            Op::IAdd => Some(mask(w, g!(0).wrapping_add(g!(1)))),
            Op::ISub => Some(mask(w, g!(0).wrapping_sub(g!(1)))),
            Op::IMul => Some(mask(w, g!(0).wrapping_mul(g!(1)))),
            Op::SNegate => Some(mask(w, (sext(w, g!(0)).wrapping_neg()) as u64)),
            Op::UDiv => {
                if g!(1) == 0 { return Err("division by zero".into()); }
                Some(mask(w, g!(0) / g!(1)))
            }
            Op::UMod => {
                if g!(1) == 0 { return Err("modulo by zero".into()); }
                Some(mask(w, g!(0) % g!(1)))
            }
            Op::SDiv => {
                if g!(1) == 0 { return Err("division by zero".into()); }
                Some(mask(w, sext(w, g!(0)).wrapping_div(sext(w, g!(1))) as u64))
            }
            Op::SRem => {
                if g!(1) == 0 { return Err("remainder by zero".into()); }
                Some(mask(w, sext(w, g!(0)).wrapping_rem(sext(w, g!(1))) as u64))
            }
            Op::ShiftLeftLogical | Op::ShiftRightLogical | Op::ShiftRightArithmetic => {
                let s = g!(1);
                if s >= u64::from(w) {
                    return Err(format!("shift amount {s} out of range for {w} bits (undefined in SPIR-V)"));
                }
                Some(match ins.op {
                    Op::ShiftLeftLogical => mask(w, g!(0) << s),
                    Op::ShiftRightLogical => g!(0) >> s,
                    _ => mask(w, (sext(w, g!(0)) >> s) as u64),
                })
            }
            Op::BitwiseAnd => Some(g!(0) & g!(1)),
            Op::BitwiseOr => Some(g!(0) | g!(1)),
            Op::BitwiseXor => Some(g!(0) ^ g!(1)),
            Op::Not => Some(mask(w, !g!(0))),
            Op::BitCount => Some(u64::from(g!(0).count_ones())),
            Op::LogicalAnd => Some(g!(0) & g!(1)),
            Op::LogicalOr => Some(g!(0) | g!(1)),
            Op::LogicalNot => Some(g!(0) ^ 1),
            Op::LogicalEqual => Some(u64::from(g!(0) == g!(1))),
            Op::LogicalNotEqual => Some(u64::from(g!(0) != g!(1))),
            Op::Select => Some(if g!(0) != 0 { g!(1) } else { g!(2) }),
            Op::IEqual => Some(u64::from(g!(0) == g!(1))),
            Op::INotEqual => Some(u64::from(g!(0) != g!(1))),
            Op::ULessThan => Some(u64::from(g!(0) < g!(1))),
            Op::ULessThanEqual => Some(u64::from(g!(0) <= g!(1))),
            Op::UGreaterThan => Some(u64::from(g!(0) > g!(1))),
            Op::UGreaterThanEqual => Some(u64::from(g!(0) >= g!(1))),
            Op::SLessThan | Op::SLessThanEqual | Op::SGreaterThan | Op::SGreaterThanEqual => {
                let ow = sw(0);
                let (a, b) = (sext(ow, g!(0)), sext(ow, g!(1)));
                Some(u64::from(match ins.op {
                    Op::SLessThan => a < b,
                    Op::SLessThanEqual => a <= b,
                    Op::SGreaterThan => a > b,
                    _ => a >= b,
                }))
            }
            Op::CompositeExtract => {
                let idx = o[1] as usize;
                Some(t.vecs.get(&o[0]).ok_or("extract from unknown vector")?[idx])
            }
            Op::CompositeConstruct => {
                let v: Vec<u64> = o.iter().map(|&i| t.vals[i as usize]).collect();
                t.vecs.insert(ins.res, v);
                None
            }
            Op::CompositeInsert => {
                let mut v = t.vecs.get(&o[1]).cloned().ok_or("insert into unknown vector")?;
                v[o[2] as usize] = g!(0);
                t.vecs.insert(ins.res, v);
                None
            }
            Op::FConvert => {
                let (from, to) = (sw(0), w);
                let x = fv(from, g!(0));
                Some(match (to, ins.decorated_rounding) {
                    (16, Some(rspirv::spirv::FPRoundingMode::RTZ)) => {
                        // Round toward zero: truncate the f32 mantissa to 10 bits.
                        let h = super::coverage_tests::interp::f32_to_f16_bits(x as f32);
                        let back = f64::from(f16_to_f32(u32::from(h)));
                        let h = if back.abs() > x.abs() && !x.is_nan() { h.wrapping_sub(1) } else { h };
                        u64::from(h)
                    }
                    (_, None) | (_, Some(rspirv::spirv::FPRoundingMode::RTE)) => fb(to, x),
                    (_, Some(m)) => return Err(format!("unmodelled FConvert rounding {m:?}")),
                })
            }
            Op::FMul | Op::FAdd | Op::FSub | Op::FDiv => {
                let (a, b) = (fv(w, g!(0)), fv(w, g!(1)));
                Some(fb(w, match ins.op {
                    Op::FMul => a * b,
                    Op::FAdd => a + b,
                    Op::FSub => a - b,
                    _ => a / b,
                }))
            }
            Op::FNegate => Some(fb(w, -fv(w, g!(0)))),
            Op::FOrdEqual | Op::FOrdNotEqual | Op::FOrdLessThan | Op::FOrdLessThanEqual | Op::FOrdGreaterThan
            | Op::FOrdGreaterThanEqual | Op::FUnordEqual | Op::FUnordNotEqual | Op::FUnordLessThan
            | Op::FUnordLessThanEqual | Op::FUnordGreaterThan | Op::FUnordGreaterThanEqual => {
                let ow = sw(0);
                let (a, b) = (fv(ow, g!(0)), fv(ow, g!(1)));
                let unordered = a.is_nan() || b.is_nan();
                let op = ins.op;
                let ord = matches!(
                    op,
                    Op::FOrdEqual | Op::FOrdNotEqual | Op::FOrdLessThan | Op::FOrdLessThanEqual | Op::FOrdGreaterThan | Op::FOrdGreaterThanEqual
                );
                let value = match op {
                    Op::FOrdEqual | Op::FUnordEqual => a == b,
                    Op::FOrdNotEqual | Op::FUnordNotEqual => a != b,
                    Op::FOrdLessThan | Op::FUnordLessThan => a < b,
                    Op::FOrdLessThanEqual | Op::FUnordLessThanEqual => a <= b,
                    Op::FOrdGreaterThan | Op::FUnordGreaterThan => a > b,
                    _ => a >= b,
                };
                Some(u64::from(if unordered { !ord } else { value }))
            }
            Op::IsNan => Some(u64::from(fv(sw(0), g!(0)).is_nan())),
            Op::IsInf => Some(u64::from(fv(sw(0), g!(0)).is_infinite())),
            Op::ConvertSToF | Op::ConvertUToF => {
                let ow = sw(0);
                let raw = g!(0);
                let signed = ins.op == Op::ConvertSToF;
                Some(if w == 32 && ow == 64 {
                    u64::from(if signed { sext(ow, raw) as f32 } else { raw as f32 }.to_bits())
                } else if w == 16 {
                    let v = if signed { sext(ow, raw) as f32 } else { raw as f32 };
                    u64::from(super::coverage_tests::interp::f32_to_f16_bits(v))
                } else {
                    fb(w, if signed { sext(ow, raw) as f64 } else { raw as f64 })
                })
            }
            Op::ConvertFToS | Op::ConvertFToU => {
                let v = fv(sw(0), g!(0)).trunc();
                Some(if ins.op == Op::ConvertFToS {
                    let (lo, hi) = (-(2f64.powi(w as i32 - 1)), 2f64.powi(w as i32 - 1) - 1.0);
                    let c = if v.is_nan() { 0.0 } else { v.clamp(lo, hi) };
                    mask(w, (c as i64) as u64)
                } else {
                    let hi = if w == 64 { u64::MAX as f64 } else { 2f64.powi(w as i32) - 1.0 };
                    let c = if v.is_nan() { 0.0 } else { v.clamp(0.0, hi) };
                    mask(w, c as u64)
                })
            }
            Op::AccessChain | Op::InBoundsAccessChain => Some(g!(0)),
            Op::PtrAccessChain | Op::InBoundsPtrAccessChain => {
                // Element size comes from the result pointer type; the translator only uses
                // byte-sized or scalar element pointers here, so derive it from the id width table.
                let elem = self.elem_bytes(ins);
                Some(g!(0).wrapping_add((sext(64, g!(1)) as u64).wrapping_mul(elem)))
            }
            Op::SDot | Op::UDot | Op::SUDot | Op::SDotAccSat | Op::UDotAccSat | Op::SUDotAccSat => {
                let pack = |t: &Thread, i: usize| -> u32 {
                    match t.vecs.get(&o[i]) {
                        Some(v) => v.iter().enumerate().fold(0u32, |a, (k, e)| a | (((*e as u32) & 0xff) << (8 * k))),
                        None => t.vals[o[i] as usize] as u32,
                    }
                };
                let (a, b) = (pack(t, 0), pack(t, 1));
                let mut sum = 0i64;
                for k in 0..4 {
                    let (x, y) = ((a >> (8 * k)) & 0xff, (b >> (8 * k)) & 0xff);
                    let xs = match ins.op {
                        Op::SDot | Op::SUDot | Op::SDotAccSat | Op::SUDotAccSat => i64::from(x as u8 as i8),
                        _ => i64::from(x),
                    };
                    let ys = match ins.op {
                        Op::SDot | Op::SDotAccSat => i64::from(y as u8 as i8),
                        _ => i64::from(y),
                    };
                    sum += xs * ys;
                }
                match ins.op {
                    Op::SDot | Op::UDot | Op::SUDot => Some(mask(w, sum as u64)),
                    Op::UDotAccSat => Some((sum + i64::from(g!(2) as u32)).clamp(0, i64::from(u32::MAX)) as u64),
                    _ => Some((sum + i64::from(g!(2) as u32 as i32)).clamp(i64::from(i32::MIN), i64::from(i32::MAX)) as u64 & 0xffff_ffff),
                }
            }
            Op::ExtInst => Some(self.ext_inst(ins, &t.vals, w)?),
            other => return Err(format!("simulator does not support {other:?}")),
        };
        if let Some(v) = result {
            t.vals[ins.res as usize] = v;
        }
        if next_pc {
            t.pc += 1;
        }
        Ok(())
    }

    /// Transfers control to `target`, evaluating its leading OpPhi instructions in parallel.
    fn enter(&self, t: &mut Thread, target: u32) {
        let from = t.label;
        let block = self.label_index[&target];
        let mut pc = 0;
        let mut assigned = Vec::new();
        while let Some(i) = self.blocks[block].get(pc) {
            if i.op != Op::Phi {
                break;
            }
            let pair = i.ops.chunks(2).find(|p| p[1] == from).unwrap_or_else(|| panic!("phi without incoming edge from {from}"));
            assigned.push((i.res, t.vals[pair[0] as usize]));
            pc += 1;
        }
        for (r, v) in assigned {
            t.vals[r as usize] = v;
        }
        t.block = block;
        t.pc = pc;
        t.label = target;
    }

    fn mem_read(gpu: &Gpu, shared: &Mem, private: &Mem, addr: u64, bytes: u32) -> u64 {
        if addr >= PRIVATE_BASE {
            private.read(addr, bytes)
        } else if addr >= SHARED_BASE && addr < GLOBAL_BASE {
            shared.read(addr, bytes)
        } else {
            gpu.check(addr, bytes, "load");
            gpu.mem.read(addr, bytes)
        }
    }

    fn mem_write(gpu: &mut Gpu, shared: &mut Mem, private: &mut Mem, addr: u64, bytes: u32, value: u64) {
        if addr >= PRIVATE_BASE {
            private.write(addr, bytes, value);
        } else if addr >= SHARED_BASE && addr < GLOBAL_BASE {
            shared.write(addr, bytes, value);
        } else {
            gpu.check(addr, bytes, "store");
            gpu.mem.write(addr, bytes, value);
        }
    }

    fn ext_inst(&self, ins: &Ins, vals: &[u64], w: u32) -> Result<u64, String> {
        use rspirv::spirv::OpenclStd100Op as C;
        let op = ins.ops[1];
        let arg = |i: usize| vals[ins.ops[2 + i] as usize];
        let aw = |i: usize| self.vw[ins.ops[2 + i] as usize];
        let is = |c: C| op == c as u32;
        Ok(if is(C::s_min) {
            mask(w, sext(w, arg(0)).min(sext(w, arg(1))) as u64)
        } else if is(C::s_max) {
            mask(w, sext(w, arg(0)).max(sext(w, arg(1))) as u64)
        } else if is(C::u_min) {
            arg(0).min(arg(1))
        } else if is(C::u_max) {
            arg(0).max(arg(1))
        } else if is(C::s_abs) {
            mask(w, sext(w, arg(0)).unsigned_abs())
        } else if is(C::clz) {
            u64::from((arg(0) << (64 - w)).leading_zeros().min(w))
        } else if is(C::u_mul_hi) {
            ((u128::from(arg(0)) * u128::from(arg(1))) >> w) as u64
        } else if is(C::s_mul_hi) {
            mask(w, ((i128::from(sext(w, arg(0))) * i128::from(sext(w, arg(1)))) >> w) as u64)
        } else if is(C::popcount) {
            u64::from(arg(0).count_ones())
        } else if is(C::fabs) {
            fb(w, fv(w, arg(0)).abs())
        } else if is(C::fmin) || is(C::fmax) {
            let (a, b) = (fv(w, arg(0)), fv(w, arg(1)));
            let r = if a.is_nan() { b } else if b.is_nan() { a } else if is(C::fmin) { a.min(b) } else { a.max(b) };
            fb(w, r)
        } else if is(C::fma) {
            // Single rounding in the operand precision.
            match w {
                32 => u64::from(
                    f32::from_bits(arg(0) as u32).mul_add(f32::from_bits(arg(1) as u32), f32::from_bits(arg(2) as u32)).to_bits(),
                ),
                _ => fb(w, fv(w, arg(0)).mul_add(fv(w, arg(1)), fv(w, arg(2)))),
            }
        } else if is(C::trunc) {
            fb(w, fv(aw(0), arg(0)).trunc())
        } else if is(C::rint) {
            fb(w, fv(aw(0), arg(0)).round_ties_even())
        } else if is(C::floor) {
            fb(w, fv(aw(0), arg(0)).floor())
        } else if is(C::ceil) {
            fb(w, fv(aw(0), arg(0)).ceil())
        } else if is(C::sqrt) {
            fb(w, fv(aw(0), arg(0)).sqrt())
        } else if is(C::native_exp2) {
            fb(w, fv(aw(0), arg(0)).exp2())
        } else if is(C::native_log2) || is(C::log2) {
            fb(w, fv(aw(0), arg(0)).log2())
        } else if is(C::log) {
            fb(w, fv(aw(0), arg(0)).ln())
        } else if is(C::native_sin) || is(C::sin) {
            fb(w, fv(aw(0), arg(0)).sin())
        } else if is(C::native_cos) || is(C::cos) {
            fb(w, fv(aw(0), arg(0)).cos())
        } else if is(C::native_sqrt) {
            fb(w, fv(aw(0), arg(0)).sqrt())
        } else if is(C::native_rsqrt) || is(C::rsqrt) {
            fb(w, 1.0 / fv(aw(0), arg(0)).sqrt())
        } else if is(C::native_recip) {
            fb(w, 1.0 / fv(aw(0), arg(0)))
        } else if is(C::native_divide) {
            fb(w, fv(aw(0), arg(0)) / fv(aw(1), arg(1)))
        } else if is(C::exp2) {
            fb(w, fv(aw(0), arg(0)).exp2())
        } else if is(C::exp) {
            fb(w, fv(aw(0), arg(0)).exp())
        } else if is(C::tanh) {
            fb(w, fv(aw(0), arg(0)).tanh())
        } else if is(C::u_clamp) {
            arg(0).clamp(arg(1), arg(2))
        } else if is(C::s_clamp) {
            mask(w, sext(w, arg(0)).clamp(sext(w, arg(1)), sext(w, arg(2))) as u64)
        } else if is(C::rotate) {
            let s = arg(1) % u64::from(w);
            if s == 0 { arg(0) } else { mask(w, (arg(0) << s) | (arg(0) >> (u64::from(w) - s))) }
        } else {
            return Err(format!("unsupported OpenCL.std instruction {op}"));
        })
    }

    fn elem_bytes(&self, ins: &Ins) -> u64 {
        // Pointer arithmetic strides are recorded per result pointer type by the decoder.
        ins.stride
    }
}

struct Ctx<'a> {
    launch: &'a Launch,
    group: [u32; 3],
}
