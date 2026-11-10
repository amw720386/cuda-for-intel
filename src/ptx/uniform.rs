//! Block-uniformity analysis for barrier legality.
//!
//! Abstract value domain, one element per PTX register (flow-insensitive: the join of every
//! definition, which over-approximates the value at every program point):
//!
//! * `Mul`  - a multiple of `ntid.x` (`0`, `ntid.x`, `n - n % ntid.x`, ...); uniform
//! * `Ntid` - exactly `ntid.x`; uniform
//! * `Rem(a)` - `a % ntid.x` for the single-definition uniform register `a`; uniform
//! * `Eq(a)` - provably equal to the single-definition uniform register `a`; uniform
//! * `Uni`  - the same in every thread of the block
//! * `Tid`  - `tid.x + m * ntid.x` where `m` is block-uniform (`tid.x` itself, and the
//!   induction variable of `for (i = tid.x; ...; i += blockDim.x)`)
//! * `Top`  - unknown (memory loads and anything unmodelled), trusted as in the surrounding rules
//! * `Var`  - depends on the thread index in a way that is not modelled
//!
//! A comparison `Tid < K` / `Tid >= K` (either operand order) with `K` a multiple of `ntid.x` is
//! block-uniform: `tid.x + m*ntid.x < c*ntid.x` holds iff `m < c`, because `0 <= tid.x < ntid.x`.
//! That is what makes the trip count of a block-stride loop uniform, and with it every barrier
//! inside the loop. Arithmetic is assumed not to wrap, as for any index computation.
//!
//! Soundness of "`m` is uniform" is by induction on loop iterations and holds only while control
//! reaching the definition is itself uniform, so every definition inside the region of a branch
//! on a `Var` predicate (up to the branch's postdominator) is degraded to `Var`.

use super::flow::Graphs;
use super::{Block, Comparison, Instruction, IntegerOp, Terminator};
use std::collections::HashMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Cls {
    Bot,
    Mul,
    Ntid,
    Rem(u32, bool),
    Eq(u32),
    Uni,
    Tid,
    Top,
    Var,
}

impl Cls {
    fn uniform(self) -> bool {
        matches!(self, Cls::Mul | Cls::Ntid | Cls::Rem(..) | Cls::Eq(_) | Cls::Uni)
    }

    fn join(self, other: Cls) -> Cls {
        use Cls::*;
        if self == other || other == Bot {
            return self;
        }
        if self == Bot {
            return other;
        }
        match (self, other) {
            (Var, _) | (_, Var) => Var,
            (Mul, Ntid) | (Ntid, Mul) => Mul,
            (Rem(a, x), Rem(b, y)) if a == b => Rem(a, x || y),
            (a, b) if a.uniform() && b.uniform() => Uni,
            (Top, b) | (b, Top) if b.uniform() => Top,
            _ => Var,
        }
    }
}

pub(super) struct Uniformity {
    /// Class of each block's branch predicate at the terminator.
    branch_cls: Vec<Cls>,
}

impl Uniformity {
    /// True when the branch ending `block` is provably taken identically by every thread.
    pub(super) fn branch_uniform(&self, block: usize) -> bool {
        self.branch_cls[block].uniform()
    }
}

struct Ctx<'a, 'b> {
    ids: &'b HashMap<&'a str, u32>,
    cls: &'b [Cls],
    single: &'b [bool],
    hi_zero: &'b [(usize, u32)],
    graphs: &'b Graphs,
    /// Strong updates within the current block: the class of the latest definition so far.
    local: &'b HashMap<u32, Cls>,
    /// Definitions of registers that are tracked flow-sensitively (reaching definitions).
    md_defs: &'b [Vec<u32>],
    md_cls: &'b [Cls],
    /// Reaching-definition bits at entry of the current block; empty = flow-insensitive.
    reach: &'b [u64],
}

impl<'a> Ctx<'a, '_> {
    fn operand(&self, s: &str) -> Cls {
        let s = s.trim();
        if s.starts_with("%tid.x") {
            return Cls::Tid;
        }
        if s.starts_with("%ntid.x") {
            return Cls::Ntid;
        }
        if s.starts_with("%tid.") || s.starts_with("%laneid") || s.starts_with("%warpid") || s.starts_with("%lanemask") {
            return Cls::Var;
        }
        if s.starts_with("%ntid.") || s.starts_with("%ctaid.") || s.starts_with("%nctaid.") {
            return Cls::Uni;
        }
        if s.as_bytes().first().is_some_and(|c| c.is_ascii_digit() || *c == b'-') {
            return if s == "0" { Cls::Mul } else { Cls::Uni };
        }
        let base = s.split('+').next().unwrap_or(s);
        let Some(&id) = self.ids.get(base) else { return Cls::Top };
        if let Some(&c) = self.local.get(&id) {
            return c;
        }
        let defs = &self.md_defs[id as usize];
        if defs.is_empty() {
            return self.cls[id as usize];
        }
        let mut c = Cls::Bot;
        for &d in defs {
            if self.reach.is_empty() || self.reach[d as usize / 64] >> (d % 64) & 1 == 1 {
                c = c.join(self.md_cls[d as usize]);
            }
        }
        c
    }

    fn id(&self, s: &str) -> Option<u32> {
        self.ids.get(s.split('+').next().unwrap_or(s)).copied()
    }

    /// Register whose value equals `s` exactly and never changes (single definition).
    fn root(&self, s: &str) -> Option<u32> {
        if let Cls::Eq(a) = self.operand(s) {
            return Some(a);
        }
        let id = self.id(s)?;
        (self.single[id as usize] && self.cls[id as usize].uniform()).then_some(id)
    }

    /// `root` is known to fit in 32 bits (upper half zero) in `block`.
    fn hi_zero_at(&self, root: u32, block: usize) -> bool {
        self.hi_zero.iter().any(|&(edge, reg)| reg == root && self.graphs.dom.dominates(edge, block))
    }
}

fn arith(ctx: &Ctx<'_, '_>, op: &IntegerOp, signed_hint: Option<bool>, a: &str, b: &str) -> Cls {
    let (ca, cb) = (ctx.operand(a), ctx.operand(b));
    if ca == Cls::Bot || cb == Cls::Bot {
        return Cls::Bot;
    }
    use IntegerOp::*;
    let both = ca.uniform() && cb.uniform();
    let mulish = |c: Cls| matches!(c, Cls::Mul | Cls::Ntid);
    match op {
        Add if both => if mulish(ca) && mulish(cb) { Cls::Mul } else { Cls::Uni },
        Add if (ca == Cls::Tid && mulish(cb)) || (cb == Cls::Tid && mulish(ca)) => Cls::Tid,
        Sub if both => match (cb, ctx.root(a)) {
            (Cls::Rem(r, _), Some(root)) if r == root => Cls::Mul,
            _ if mulish(ca) && mulish(cb) => Cls::Mul,
            _ => Cls::Uni,
        },
        Mul if both => if (mulish(ca) && cb.uniform()) || (mulish(cb) && ca.uniform()) { Cls::Mul } else { Cls::Uni },
        ShiftLeft if both => if mulish(ca) { Cls::Mul } else { Cls::Uni },
        RemUnsigned | Rem if both && cb == Cls::Ntid => match ctx.root(a) {
            Some(root) => Cls::Rem(root, matches!(op, Rem)),
            None => Cls::Uni,
        },
        _ if both => Cls::Uni,
        _ => {
            let _ = signed_hint;
            if ca == Cls::Var || cb == Cls::Var || ca == Cls::Tid || cb == Cls::Tid { Cls::Var } else { Cls::Top }
        }
    }
}

fn pure_default(inst: &Instruction<'_>) -> bool {
    use Instruction::*;
    !(inst.is_memory_load()
        || matches!(
            inst,
            Shuffle { .. }
                | ShuffleBfly { .. }
                | Vote { .. }
                | Mma { .. }
                | LdMatrix { .. }
                | LaneId(_)
                | Special(..)
                | ReduxSyncAdd { .. }
                | Ext { .. }
        ))
}

fn transfer<'a>(ctx: &Ctx<'a, '_>, inst: &Instruction<'a>, block: usize, defs: &[&'a str], inputs: &[&'a str]) -> Cls {
    use Instruction::*;
    let widen = |c: Cls, signed: bool| match c {
        Cls::Rem(a, rs) if rs && !signed => {
            let _ = a;
            Cls::Uni
        }
        c => c,
    };
    let _ = defs;
    match inst {
        Builtin(_, src) => ctx.operand(src),
        LaneId(_) | Special(..) => Cls::Var,
        Param(..) | ParamS8(..) | ParamS32(..) | ParamVec { .. } => Cls::Uni,
        Move32(_, s) | Move64(_, s) => ctx.operand(s),
        Integer32(op, _, [a, b]) => arith(ctx, op, None, a, b),
        Add64(_, [a, b]) => arith(ctx, &IntegerOp::Add, None, a, b),
        Sub64(_, [a, b]) => arith(ctx, &IntegerOp::Sub, None, a, b),
        Mul64(_, [a, b]) => arith(ctx, &IntegerOp::Mul, None, a, b),
        ShiftLeft64(_, [a, b]) => arith(ctx, &IntegerOp::ShiftLeft, None, a, b),
        Rem64 { signed, src: [a, b], .. } => {
            arith(ctx, if *signed { &IntegerOp::Rem } else { &IntegerOp::RemUnsigned }, None, a, b)
        }
        Widen { signed, src, .. } => match ctx.operand(src) {
            Cls::Eq(a) => Cls::Eq(a),
            c => widen(c, *signed),
        },
        Narrow(_, s) => match ctx.operand(s) {
            Cls::Tid => Cls::Tid,
            Cls::Ntid => Cls::Ntid,
            Cls::Bot => Cls::Bot,
            c if c.uniform() => match ctx.root(s) {
                Some(root) if ctx.hi_zero_at(root, block) => Cls::Eq(root),
                _ => Cls::Uni,
            },
            Cls::Var => Cls::Var,
            _ => Cls::Top,
        },
        Compare { op, src: [a, b], bits16: false, .. } | Compare64 { op, src: [a, b], .. } => {
            let (ca, cb) = (ctx.operand(a), ctx.operand(b));
            let mulish = |c: Cls| matches!(c, Cls::Mul | Cls::Ntid);
            if ca == Cls::Bot || cb == Cls::Bot {
                Cls::Bot
            } else if ca.uniform() && cb.uniform() {
                Cls::Uni
            } else if (ca == Cls::Tid && mulish(cb) && matches!(op, Comparison::Less | Comparison::GreaterEqual))
                || (mulish(ca) && cb == Cls::Tid && matches!(op, Comparison::Greater | Comparison::LessEqual))
            {
                Cls::Uni
            } else if matches!(ca, Cls::Var | Cls::Tid) || matches!(cb, Cls::Var | Cls::Tid) {
                Cls::Var
            } else {
                Cls::Top
            }
        }
        _ => {
            let mut result = Cls::Bot;
            for &src in inputs {
                result = match (result, ctx.operand(src)) {
                    (Cls::Bot, c) | (c, Cls::Bot) => c,
                    (a, b) if a.uniform() && b.uniform() => Cls::Uni,
                    (a, b) if a == Cls::Var || b == Cls::Var || a == Cls::Tid || b == Cls::Tid => Cls::Var,
                    _ => Cls::Top,
                };
            }
            if result == Cls::Bot && !inputs.is_empty() {
                return Cls::Bot;
            }
            if !pure_default(inst) {
                return if result == Cls::Var { Cls::Var } else { Cls::Top };
            }
            if result == Cls::Bot { Cls::Uni } else if result.uniform() { Cls::Uni } else { result }
        }
    }
}

pub(super) fn analyze<'a>(
    blocks: &[Block<'a>],
    order: &[usize],
    predecessors: &[Vec<usize>],
    graphs: &Graphs,
) -> Uniformity {
    // Debug switches, read once per call rather than per branch and per tainted block.
    let debug = std::env::var_os("CFI_UNIFORM_DEBUG").is_some();
    let watch = std::env::var_os("CFI_UNIFORM_WATCH");
    let mut ids: HashMap<&'a str, u32> = HashMap::new();
    let mut defcount: Vec<u32> = Vec::new();
    let mut scratch = Vec::new();
    let mut def_inst: Vec<Option<(usize, usize)>> = Vec::new();
    for (bi, block) in blocks.iter().enumerate() {
        for (ii, inst) in block.instructions.iter().enumerate() {
            if inst.collect_definitions(&mut scratch).is_some() {
                for &d in &scratch {
                    let next = ids.len() as u32;
                    let id = *ids.entry(d).or_insert(next);
                    if id as usize == defcount.len() {
                        defcount.push(0);
                        def_inst.push(None);
                    }
                    defcount[id as usize] += 1;
                    def_inst[id as usize] = Some((bi, ii));
                }
            }
        }
    }
    let single: Vec<bool> = defcount.iter().map(|&c| c == 1).collect();

    // Registers with a few definitions (loop counters reused for unrelated loops) are classified
    // per reaching definition; lowering scratch registers and heavily redefined ones fall back
    // to the join over all definitions.
    const MAX_TRACKED_DEFS: u32 = 64;
    const MAX_REACH_BYTES: usize = 256 << 20;
    let mut names: Vec<&str> = vec![""; ids.len()];
    for (&n, &id) in &ids {
        names[id as usize] = n;
    }
    let mut md_defs: Vec<Vec<u32>> = vec![Vec::new(); ids.len()];
    let mut md_reg: Vec<u32> = Vec::new();
    let mut block_md: Vec<Vec<u32>> = vec![Vec::new(); blocks.len()];
    for (bi, block) in blocks.iter().enumerate() {
        for inst in &block.instructions {
            if inst.collect_definitions(&mut scratch).is_some() {
                for &d in &scratch {
                    let id = ids[d];
                    let tracked = defcount[id as usize] > 1
                        && defcount[id as usize] <= MAX_TRACKED_DEFS
                        && !d.starts_with("%__");
                    if tracked {
                        let index = md_reg.len() as u32;
                        md_reg.push(id);
                        md_defs[id as usize].push(index);
                        block_md[bi].push(index);
                    } else {
                        block_md[bi].push(u32::MAX);
                    }
                }
            }
        }
    }
    // Last tracked definition of each register per block.
    let block_last: Vec<Vec<(u32, u32)>> = block_md
        .iter()
        .map(|list| {
            let mut last: Vec<(u32, u32)> = Vec::new();
            for &d in list.iter().filter(|&&d| d != u32::MAX) {
                let reg = md_reg[d as usize];
                match last.iter_mut().find(|(r, _)| *r == reg) {
                    Some(entry) => entry.1 = d,
                    None => last.push((reg, d)),
                }
            }
            last
        })
        .collect();
    let words = md_reg.len().div_ceil(64);
    let mut reach_in: Vec<u64> = Vec::new();
    if words > 0 && blocks.len().saturating_mul(words).saturating_mul(8) <= MAX_REACH_BYTES {
        reach_in = vec![0u64; blocks.len() * words];
        let mut row = vec![0u64; words];
        let mut tmp = vec![0u64; words];
        loop {
            let mut changed = false;
            for &b in order {
                row.fill(0);
                for &p in &predecessors[b] {
                    tmp.copy_from_slice(&reach_in[p * words..(p + 1) * words]);
                    for &(reg, last) in &block_last[p] {
                        for &d in &md_defs[reg as usize] {
                            tmp[d as usize / 64] &= !(1u64 << (d % 64));
                        }
                        tmp[last as usize / 64] |= 1u64 << (last % 64);
                    }
                    for (r, t) in row.iter_mut().zip(&tmp) {
                        *r |= t;
                    }
                }
                let slot = &mut reach_in[b * words..(b + 1) * words];
                if slot != row.as_slice() {
                    slot.copy_from_slice(&row);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
    let mut md_cls = vec![Cls::Bot; md_reg.len()];

    // Edges that prove the upper 32 bits of a register are zero:
    // `and.b64 t, s, 0xFFFFFFFF00000000; setp.{ne,eq}.s64 p, t, 0; @p bra ...`.
    let mut hi_zero: Vec<(usize, u32)> = Vec::new();
    for (bi, block) in blocks.iter().enumerate() {
        let Terminator::Branch { predicate, taken, other } = block.end else { continue };
        let Some(&pid) = ids.get(predicate) else { continue };
        if !single[pid as usize] {
            continue;
        }
        let (db, di) = def_inst[pid as usize].unwrap();
        let Instruction::Compare64 { op, src: [t, zero], .. } = &blocks[db].instructions[di] else { continue };
        if *zero != "0" {
            continue;
        }
        let zero_edge = match op {
            Comparison::NotEqual => other,
            Comparison::Equal => taken,
            _ => continue,
        };
        let Some(&tid_) = ids.get(*t) else { continue };
        if !single[tid_ as usize] {
            continue;
        }
        let (tb, ti) = def_inst[tid_ as usize].unwrap();
        let Instruction::And64(_, [s, mask]) = &blocks[tb].instructions[ti] else { continue };
        let (s, mask) = if matches!(*mask, "-4294967296" | "18446744069414584320" | "0xFFFFFFFF00000000") {
            (*s, *mask)
        } else if matches!(*s, "-4294967296" | "18446744069414584320" | "0xFFFFFFFF00000000") {
            (*mask, *s)
        } else {
            continue;
        };
        let _ = mask;
        let Some(&sid) = ids.get(s) else { continue };
        // The proof holds in blocks dominated by the zero edge's target, which must be entered
        // only from this branch, and only while `s` cannot change (single definition).
        if single[sid as usize] && predecessors[zero_edge].len() == 1 && predecessors[zero_edge][0] == bi {
            hi_zero.push((zero_edge, sid));
        }
    }

    let mut cls = vec![Cls::Bot; ids.len()];
    let mut branch_cls = vec![Cls::Bot; blocks.len()];
    let mut local: HashMap<u32, Cls> = HashMap::new();
    let mut tainted = vec![false; blocks.len()];
    let mut stamp = vec![0u32; blocks.len()];
    let mut generation = 0u32;
    let mut budget: u64 = 50_000_000;
    let mut defs = Vec::new();
    loop {
        loop {
            let mut changed = false;
            for (bi, block) in blocks.iter().enumerate() {
                local.clear();
                let reach: &[u64] = if reach_in.is_empty() { &[] } else { &reach_in[bi * words..(bi + 1) * words] };
                let mut cursor = 0usize;
                for inst in &block.instructions {
                    let Some(inputs) = inst.collect_definitions(&mut defs) else { continue };
                    let inputs: Vec<&str> = inputs.to_vec();
                    let ctx = Ctx { ids: &ids, cls: &cls, single: &single, hi_zero: &hi_zero, graphs, local: &local, md_defs: &md_defs, md_cls: &md_cls, reach };
                    let mut value = transfer(&ctx, inst, bi, &defs, &inputs);
                    if tainted[bi] && value != Cls::Bot {
                        value = Cls::Var;
                    }
                    for &d in &defs {
                        let id = ids[d];
                        local.insert(id, value);
                        let md = block_md[bi][cursor];
                        cursor += 1;
                        if md != u32::MAX {
                            let joined = md_cls[md as usize].join(value);
                            if joined != md_cls[md as usize] {
                                md_cls[md as usize] = joined;
                                changed = true;
                            }
                        }
                        let joined = cls[id as usize].join(value);
                        if joined != cls[id as usize] {
                            cls[id as usize] = joined;
                            changed = true;
                        }
                    }
                }
                if let Terminator::Branch { predicate, .. } = block.end {
                    let ctx = Ctx { ids: &ids, cls: &cls, single: &single, hi_zero: &hi_zero, graphs, local: &local, md_defs: &md_defs, md_cls: &md_cls, reach };
                    let c = ctx.operand(predicate);
                    if c != branch_cls[bi] {
                        branch_cls[bi] = c;
                        changed = true;
                    }
                }
            }
            if !changed {
                break;
            }
        }
        // Taint the region of every branch on a thread-dependent predicate.
        let mut grew = false;
        for (bi, block) in blocks.iter().enumerate() {
            let Terminator::Branch { predicate, taken, other } = block.end else { continue };
            if taken == other {
                continue;
            }
            let c = branch_cls[bi];
            if !matches!(c, Cls::Var | Cls::Tid) {
                continue;
            }
            // Threads that leave through a side which never meets the other side again (an early
            // `return`) cannot observe the values the surviving side computes, so the survivors
            // stay in lock step. Only sides that reconverge on a block with real work taint.
            let meets = |from_a: usize, from_b: usize, mark: &mut Vec<u32>, generation: &mut u32, budget: &mut u64| {
                *generation += 2;
                let (ga, gb) = (*generation - 1, *generation);
                let mut stack = vec![from_a];
                while let Some(n) = stack.pop() {
                    if std::mem::replace(&mut mark[n], ga) == ga {
                        continue;
                    }
                    *budget = budget.saturating_sub(1);
                    stack.extend(blocks[n].end.successors());
                }
                stack.push(from_b);
                while let Some(n) = stack.pop() {
                    if mark[n] == gb {
                        continue;
                    }
                    let seen_by_a = mark[n] == ga;
                    mark[n] = gb;
                    if seen_by_a && !(blocks[n].instructions.is_empty() && matches!(blocks[n].end, Terminator::Return)) {
                        return true;
                    }
                    *budget = budget.saturating_sub(1);
                    stack.extend(blocks[n].end.successors());
                }
                false
            };
            if !meets(taken, other, &mut stamp, &mut generation, &mut budget) {
                continue;
            }
            if debug {
                eprintln!("TAINT branch {predicate} in block {bi} class {c:?}");
            }
            let mut stack = vec![taken, other];
            generation += 1;
            while let Some(next) = stack.pop() {
                if graphs.postdominates(bi, next) && next != bi && !false {
                    continue;
                }
                if std::mem::replace(&mut stamp[next], generation) == generation {
                    continue;
                }
                if budget == 0 {
                    // Give up precision rather than time: nothing downstream stays provable.
                    branch_cls.iter_mut().for_each(|c| *c = Cls::Var);
                    return Uniformity { branch_cls };
                }
                budget -= 1;
                if !tainted[next] {
                    tainted[next] = true;
                    grew = true;
                    if let Some(watch) = &watch {
                        let w = watch.to_string_lossy();
                        let mut d = Vec::new();
                        if blocks[next].instructions.iter().any(|i| i.collect_definitions(&mut d).is_some() && d.iter().any(|x| *x == w)) {
                            eprintln!("WATCH {w} block {next} tainted by branch {predicate} in block {bi}");
                        }
                    }
                }
                stack.extend(blocks[next].end.successors());
            }
        }
        if !grew {
            if debug {
                for (name, &id) in &ids {
                    eprintln!("UNI {name} {:?} single={}", cls[id as usize], single[id as usize]);
                }
            }
            return Uniformity { branch_cls };
        }
    }
}
