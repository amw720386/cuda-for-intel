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
            (a, b) if a.uniform() && b.uniform() => Uni,
            (Top, b) | (b, Top) if b.uniform() => Top,
            _ => Var,
        }
    }
}

pub(super) struct Uniformity<'a> {
    ids: HashMap<&'a str, u32>,
    cls: Vec<Cls>,
}

impl<'a> Uniformity<'a> {
    /// True when the predicate is provably the same for every thread of the block.
    pub(super) fn predicate_uniform(&self, name: &str) -> bool {
        self.ids.get(name).is_some_and(|&id| self.cls[id as usize].uniform())
    }
}

struct Ctx<'a, 'b> {
    ids: &'b HashMap<&'a str, u32>,
    cls: &'b [Cls],
    single: &'b [bool],
    hi_zero: &'b [(usize, u32)],
    graphs: &'b Graphs,
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
        self.ids.get(base).map_or(Cls::Top, |&id| self.cls[id as usize])
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

pub(super) fn analyze<'a>(blocks: &[Block<'a>], predecessors: &[Vec<usize>], graphs: &Graphs) -> Uniformity<'a> {
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
    let mut tainted = vec![false; blocks.len()];
    let mut stamp = vec![0u32; blocks.len()];
    let mut generation = 0u32;
    let mut budget: u64 = 50_000_000;
    let mut defs = Vec::new();
    loop {
        loop {
            let mut changed = false;
            for (bi, block) in blocks.iter().enumerate() {
                for inst in &block.instructions {
                    let Some(inputs) = inst.collect_definitions(&mut defs) else { continue };
                    let inputs: Vec<&str> = inputs.to_vec();
                    let ctx = Ctx { ids: &ids, cls: &cls, single: &single, hi_zero: &hi_zero, graphs };
                    let mut value = transfer(&ctx, inst, bi, &defs, &inputs);
                    if tainted[bi] && value != Cls::Bot {
                        value = Cls::Var;
                    }
                    for &d in &defs {
                        let id = ids[d] as usize;
                        let joined = cls[id].join(value);
                        if joined != cls[id] {
                            cls[id] = joined;
                            changed = true;
                        }
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
            let c = ids.get(predicate).map_or(Cls::Top, |&id| cls[id as usize]);
            if !matches!(c, Cls::Var | Cls::Tid) {
                continue;
            }
            generation += 1;
            let mut stack = vec![taken, other];
            while let Some(next) = stack.pop() {
                if graphs.postdominates(bi, next) && next != bi && !false {
                    continue;
                }
                if std::mem::replace(&mut stamp[next], generation) == generation {
                    continue;
                }
                if budget == 0 {
                    // Give up precision rather than time: nothing downstream stays provable.
                    cls.iter_mut().for_each(|c| *c = Cls::Var);
                    return Uniformity { ids, cls };
                }
                budget -= 1;
                if !tainted[next] {
                    tainted[next] = true;
                    grew = true;
                }
                stack.extend(blocks[next].end.successors());
            }
        }
        if !grew {
            return Uniformity { ids, cls };
        }
    }
}
