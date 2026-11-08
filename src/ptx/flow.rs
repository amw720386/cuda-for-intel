use super::{Block, Comparison, Error, Instruction, IntegerOp, Terminator};
use std::collections::HashSet;

/// Memory ceiling for the definite-assignment bit matrices.
const MAX_FLOW_BYTES: usize = 1 << 30;

pub(super) struct Flow<'a> {
    pub(super) order: Vec<usize>,
    names: Vec<&'a str>,
    words: usize,
    /// Row-major bitsets: registers definitely written on every path into each block.
    inputs: Vec<u64>,
}

impl<'a> Flow<'a> {
    /// Fills `into` with the registers definitely initialized on entry to `index`.
    pub(super) fn load_inputs(&self, index: usize, into: &mut HashSet<&'a str>) {
        into.clear();
        let row = &self.inputs[index * self.words..(index + 1) * self.words];
        for (w, &word) in row.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                into.insert(self.names[w * 64 + bit]);
            }
        }
    }

    pub(super) fn analyze(blocks: &[Block<'a>]) -> Result<Self, Error> {
        if blocks.is_empty() {
            return Err(Error::new("PTX entry has no blocks"));
        }
        let mut predecessors = vec![Vec::new(); blocks.len()];
        for (index, block) in blocks.iter().enumerate() {
            for next in block.end.successors() {
                predecessors[next].push(index);
            }
        }

        let mut order = Vec::new();
        let mut visited = vec![false; blocks.len()];
        let mut stack = vec![(0, false)];
        while let Some((index, done)) = stack.pop() {
            if done {
                order.push(index);
            } else if !std::mem::replace(&mut visited[index], true) {
                stack.push((index, true));
                stack.extend(blocks[index].end.successors().map(|next| (next, false)));
            }
        }
        if order.len() != blocks.len() {
            return Err(Error::new("unreachable PTX block"));
        }
        order.reverse();

        let mut exits: Vec<_> = blocks
            .iter()
            .enumerate()
            .filter_map(|(index, block)| matches!(block.end, Terminator::Return).then_some(index))
            .collect();
        visited.fill(false);
        while let Some(index) = exits.pop() {
            if !std::mem::replace(&mut visited[index], true) {
                exits.extend(predecessors[index].iter().copied());
            }
        }
        if visited.contains(&false) {
            return Err(Error::new("PTX block has no path to a return"));
        }
        if blocks
            .iter()
            .flat_map(|block| &block.instructions)
            .any(|inst| matches!(inst, Instruction::Barrier))
        {
            verify_barriers(blocks, &order, &predecessors)?;
        }

        // Definite assignment: forward must-analysis over interned registers. Definitions only
        // generate, so out = in | gen, and the join is intersection.
        let mut ids: std::collections::HashMap<&'a str, usize> = std::collections::HashMap::new();
        let mut names: Vec<&'a str> = Vec::new();
        let mut scratch = Vec::new();
        let mut gen_lists: Vec<Vec<usize>> = Vec::with_capacity(blocks.len());
        for block in blocks {
            let mut list = Vec::new();
            for inst in &block.instructions {
                if inst.collect_definitions(&mut scratch).is_some() {
                    for &name in &scratch {
                        let next = names.len();
                        let id = *ids.entry(name).or_insert(next);
                        if id == next {
                            names.push(name);
                        }
                        list.push(id);
                    }
                }
            }
            list.sort_unstable();
            list.dedup();
            gen_lists.push(list);
        }
        let words = names.len().div_ceil(64).max(1);
        // Two bit matrices of blocks x registers; refuse inputs that would not fit a sane budget.
        if blocks.len().saturating_mul(words).saturating_mul(16) > MAX_FLOW_BYTES {
            return Err(Error::new("PTX entry too large for flow analysis"));
        }
        let mut inputs = vec![0u64; blocks.len() * words];
        let mut outputs = vec![!0u64; blocks.len() * words];
        let mut row = vec![0u64; words];
        loop {
            let mut changed = false;
            for &index in &order {
                if index == 0 {
                    row.fill(0);
                } else {
                    row.fill(!0);
                    for &prev in &predecessors[index] {
                        let out = &outputs[prev * words..(prev + 1) * words];
                        for (r, o) in row.iter_mut().zip(out) {
                            *r &= o;
                        }
                    }
                }
                inputs[index * words..(index + 1) * words].copy_from_slice(&row);
                for &id in &gen_lists[index] {
                    row[id / 64] |= 1 << (id % 64);
                }
                let out = &mut outputs[index * words..(index + 1) * words];
                if out != row.as_slice() {
                    out.copy_from_slice(&row);
                    changed = true;
                }
            }
            if !changed {
                return Ok(Self { order, names, words, inputs });
            }
        }
    }
}

/// Dominator tree (Cooper-Harvey-Kennedy) with Euler-interval numbering for O(1) ancestor
/// queries. Built over an explicit graph so the same code serves dominators of the CFG and
/// postdominators (the reverse CFG rooted at a virtual exit).
pub(super) struct DomTree {
    tin: Vec<u32>,
    tout: Vec<u32>,
}

impl DomTree {
    /// `succ[v]` are the graph successors of `v`, `pred[v]` its graph predecessors.
    fn build(root: usize, succ: &[Vec<usize>], pred: &[Vec<usize>]) -> Self {
        let n = succ.len();
        let mut seen = vec![false; n];
        let mut post = Vec::with_capacity(n);
        let mut stack: Vec<(usize, usize)> = vec![(root, 0)];
        seen[root] = true;
        while let Some(&mut (node, ref mut i)) = stack.last_mut() {
            if let Some(&next) = succ[node].get(*i) {
                *i += 1;
                if !std::mem::replace(&mut seen[next], true) {
                    stack.push((next, 0));
                }
            } else {
                post.push(node);
                stack.pop();
            }
        }
        post.reverse();
        let mut rpo = vec![u32::MAX; n];
        for (i, &node) in post.iter().enumerate() {
            rpo[node] = i as u32;
        }
        let mut idom = vec![usize::MAX; n];
        idom[root] = root;
        loop {
            let mut changed = false;
            for &node in post.iter().skip(1) {
                let mut new = usize::MAX;
                for &p in &pred[node] {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    new = if new == usize::MAX {
                        p
                    } else {
                        let (mut a, mut b) = (p, new);
                        while a != b {
                            while rpo[a] > rpo[b] {
                                a = idom[a];
                            }
                            while rpo[b] > rpo[a] {
                                b = idom[b];
                            }
                        }
                        a
                    };
                }
                if new != idom[node] {
                    idom[node] = new;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        let mut head = vec![usize::MAX; n];
        let mut next_sibling = vec![usize::MAX; n];
        for node in 0..n {
            let parent = idom[node];
            if parent != usize::MAX && node != root {
                next_sibling[node] = head[parent];
                head[parent] = node;
            }
        }
        let mut tin = vec![u32::MAX; n];
        let mut tout = vec![0u32; n];
        let mut clock = 0u32;
        let mut stack = vec![(root, false)];
        while let Some((node, done)) = stack.pop() {
            if done {
                tout[node] = clock;
                clock += 1;
                continue;
            }
            tin[node] = clock;
            clock += 1;
            stack.push((node, true));
            let mut child = head[node];
            while child != usize::MAX {
                stack.push((child, false));
                child = next_sibling[child];
            }
        }
        Self { tin, tout }
    }

    /// True when `ancestor` is an ancestor of (or equal to) `node` in the tree.
    pub(super) fn dominates(&self, ancestor: usize, node: usize) -> bool {
        self.tin[ancestor] <= self.tin[node] && self.tout[node] <= self.tout[ancestor]
    }
}

pub(super) struct Graphs {
    pub(super) dom: DomTree,
    post: DomTree,
}

impl Graphs {
    pub(super) fn build(blocks: &[Block<'_>], predecessors: &[Vec<usize>]) -> Self {
        let n = blocks.len();
        let forward: Vec<Vec<usize>> = blocks.iter().map(|b| b.end.successors().collect()).collect();
        let dom = DomTree::build(0, &forward, predecessors);
        // Reverse graph with a virtual exit node `n`.
        let mut rsucc: Vec<Vec<usize>> = predecessors.to_vec();
        let mut rpred: Vec<Vec<usize>> = forward.clone();
        rsucc.push(Vec::new());
        rpred.push(Vec::new());
        for (i, b) in blocks.iter().enumerate() {
            if matches!(b.end, Terminator::Return) {
                rsucc[n].push(i);
                rpred[i].push(n);
            }
        }
        Self { dom, post: DomTree::build(n, &rsucc, &rpred) }
    }

    /// True when `candidate` postdominates `block` (reflexive).
    pub(super) fn postdominates(&self, block: usize, candidate: usize) -> bool {
        self.post.dominates(candidate, block)
    }
}

/// Strongly connected components; `in_cycle[b]` is true when `b` lies on a cycle.
fn cyclic_blocks(blocks: &[Block<'_>]) -> Vec<bool> {
    let n = blocks.len();
    let mut index_of = vec![u32::MAX; n];
    let mut low = vec![0u32; n];
    let mut on_stack = vec![false; n];
    let mut in_cycle = vec![false; n];
    let mut scc_stack = Vec::new();
    let mut counter = 0u32;
    for root in 0..n {
        if index_of[root] != u32::MAX {
            continue;
        }
        let mut call: Vec<(usize, usize)> = vec![(root, 0)];
        index_of[root] = counter;
        low[root] = counter;
        counter += 1;
        scc_stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (node, ref mut i)) = call.last_mut() {
            let succ = blocks[node].end.successors().nth(*i);
            if let Some(next) = succ {
                *i += 1;
                if next == node {
                    in_cycle[node] = true;
                }
                if index_of[next] == u32::MAX {
                    index_of[next] = counter;
                    low[next] = counter;
                    counter += 1;
                    scc_stack.push(next);
                    on_stack[next] = true;
                    call.push((next, 0));
                } else if on_stack[next] {
                    low[node] = low[node].min(index_of[next]);
                }
            } else {
                call.pop();
                if let Some(&(parent, _)) = call.last() {
                    low[parent] = low[parent].min(low[node]);
                }
                if low[node] == index_of[node] {
                    let mut members = Vec::new();
                    loop {
                        let m = scc_stack.pop().unwrap();
                        on_stack[m] = false;
                        members.push(m);
                        if m == node {
                            break;
                        }
                    }
                    if members.len() > 1 {
                        for m in members {
                            in_cycle[m] = true;
                        }
                    }
                }
            }
        }
    }
    in_cycle
}

fn verify_barriers(blocks: &[Block<'_>], order: &[usize], predecessors: &[Vec<usize>]) -> Result<(), Error> {
    // A branch's strict postdominators are guaranteed reconvergence points.
    let postdom = Graphs::build(blocks, predecessors);
    let uniformity = super::uniform::analyze(blocks, order, predecessors, &postdom);
    let in_cycle = cyclic_blocks(blocks);
    // Blocks from which a convergent instruction is reachable (backward closure, linear).
    let mut reaches_conv = vec![false; blocks.len()];
    let mut work: Vec<usize> = Vec::new();
    for (i, block) in blocks.iter().enumerate() {
        if block.instructions.iter().any(|inst| inst.is_convergent()) {
            reaches_conv[i] = true;
            work.push(i);
        }
    }
    while let Some(i) = work.pop() {
        for &p in &predecessors[i] {
            if !std::mem::replace(&mut reaches_conv[p], true) {
                work.push(p);
            }
        }
    }
    let mut stamp = vec![0u32; blocks.len()];
    let mut generation = 0u32;

    // Treat memory loads and all definitions under divergent control as potentially varying.
    // Repeat because a loop-carried definition can affect an earlier branch.
    let mut varying = HashSet::new();
    // Values computed from `%tid.x` by ALU instructions alone (`%tid.y/z` conditions guard
    // rows that PTX programmers keep uniform by launching with `ntid.y == 1` where a block
    // cooperates along x; that cannot be proven statically) (no memory in between): a branch
    // on one of these is provably thread-divergent. Branches that are only "varying" because
    // of an unknown load or a divergent region are assumed to be taken by all threads, as
    // PTX programmers do for `__syncthreads` under data-dependent but block-uniform control.
    let mut tid_derived: HashSet<&str> = HashSet::new();
    let mut tid_defs: std::collections::HashMap<&str, (String, Vec<&str>)> = std::collections::HashMap::new();
    let mut local_derived: HashSet<&str> = HashSet::new();
    let mut checked_branches = vec![false; blocks.len()];
    // Value facts for block-uniform bounds. `K = n - n % ntid.x` is a multiple of the block size,
    // so `tid.x >= K` (or `<`) is the same for every thread of a block: all of them (K == 0) or
    // none (K >= ntid.x > tid.x). PyTorch's top-k / kthvalue loops are built on this.
    let mut tid_regs: HashSet<&str> = HashSet::new();
    let mut ntid_regs: HashSet<&str> = HashSet::new();
    let mut rem_of: std::collections::HashMap<&str, &str> = std::collections::HashMap::new();
    let mut multiple_of_ntid: HashSet<&str> = HashSet::new();
    let mut uniform_compare: HashSet<&str> = HashSet::new();
    loop {
        let old_count = varying.len();
        for &index in order {
            let block = &blocks[index];
            for inst in &block.instructions {
                match inst {
                    Instruction::Builtin(dst, src) if src.starts_with("%tid.x") => {
                        tid_regs.insert(*dst);
                    }
                    Instruction::Builtin(dst, src) if src.starts_with("%ntid.x") => {
                        ntid_regs.insert(*dst);
                    }
                    Instruction::Integer32(IntegerOp::RemUnsigned | IntegerOp::Rem, dst, [a, b])
                        if ntid_regs.contains(b) =>
                    {
                        rem_of.insert(*dst, *a);
                    }
                    Instruction::Integer32(IntegerOp::Sub, dst, [a, r]) if rem_of.get(r) == Some(a) => {
                        multiple_of_ntid.insert(*dst);
                    }
                    Instruction::Compare { op, dst, src: [x, k], .. }
                        if tid_regs.contains(x)
                            && multiple_of_ntid.contains(k)
                            && matches!(op, Comparison::GreaterEqual | Comparison::Less) =>
                    {
                        uniform_compare.insert(*dst);
                    }
                    _ => {}
                }
                let mut defined = Vec::new();
                if let Some(inputs) = inst.collect_definitions(&mut defined) {
                    let base = |src: &&str| src.split('+').next().unwrap_or(src).to_string();
                    let thread_dependent = inputs.iter().any(|src| {
                        let b = base(src);
                        src.starts_with("%tid.") || varying.contains(src) || varying.contains(b.as_str())
                    });
                    let private = matches!(inst, super::Instruction::LocalSymbol64(..))
                        || inputs.iter().any(|src| local_derived.contains(base(src).as_str()));
                    if private {
                        local_derived.extend(defined.iter().copied());
                    }
                    // A plain load from an address that is the same in every thread
                    // (and not per-thread local memory) yields a block-uniform value.
                    // Atomics return distinct values per thread.
                    let uniform_load = inst.is_plain_load()
                        && !thread_dependent
                        && !private
                        && !inst.loads_local_space();
                    if thread_dependent || (inst.is_memory_load() && !uniform_load) {
                        varying.extend(defined.iter().copied());
                    } else {
                        // Lowering scratch registers (`%__x32_0`, `%__t64_1`, scoped `%__sc..`)
                        // are rewritten by every lowered statement and never live across them;
                        // a fresh uniform definition must not inherit an earlier statement's taint.
                        for d in &defined {
                            if d.starts_with("%__") {
                                varying.remove(d);
                            }
                        }
                    }
                    let tid_dependent = !inst.is_memory_load()
                        && inputs.iter().any(|src| {
                            let b = base(src);
                            src.starts_with("%tid.x") || tid_derived.contains(src) || tid_derived.contains(b.as_str())
                        });
                    if tid_dependent {
                        if std::env::var_os("CFI_FLOW_DEBUG").is_some() {
                            for d in &defined {
                                tid_defs.entry(*d).or_insert_with(|| (format!("{inst:?}"), inputs.iter().copied().collect()));
                            }
                        }
                        tid_derived.extend(defined.iter().copied());
                    } else {
                        for d in &defined {
                            if d.starts_with("%__") {
                                tid_derived.remove(d);
                            }
                        }
                    }
                }
            }
            let Terminator::Branch {
                predicate,
                taken,
                other,
            } = block.end
            else {
                continue;
            };
            if taken == other || !varying.contains(predicate) || checked_branches[index] {
                continue;
            }
            // An early-exit guard (`@p bra $exit` where the target only returns) lets some
            // threads leave before a barrier. CUDA counts exited threads as arrived, and the
            // hardware barrier does the same for terminated invocations, so it is not a
            // divergence hazard for the barriers that follow.
            let exit_only = |block: usize| {
                blocks[block].instructions.is_empty()
                    && matches!(blocks[block].end, Terminator::Return)
            };
            if exit_only(taken) || exit_only(other) {
                continue;
            }
            if uniform_compare.contains(predicate) || uniformity.branch_uniform(index) {
                continue;
            }
            // Generalised early exit: if no barrier is reachable from one side, the threads
            // taking it only run to completion and exit, which the hardware barrier counts
            // as arrived (GGML's `if (col >= ncols) { ...no barrier...; return; }`).
            let reaches_barrier = |start: usize| reaches_conv[start];
            if !reaches_barrier(taken) || !reaches_barrier(other) || !tid_derived.contains(predicate) {
                continue;
            }
            // A tid-dependent *loop* test (`for (i = tid; i < rounded_up_n; i += blockDim)`):
            // CUDA requires all threads to run the same number of barrier-containing
            // iterations, which PyTorch's top-k / sort / scan helpers guarantee by rounding the
            // bound up to a multiple of the block size. Uniformity of the trip count is a
            // value fact we cannot prove, so trust the program as CUDA does.
            let in_loop = in_cycle[index];
            if in_loop && std::env::var_os("CFI_STRICT_BARRIERS").is_none() {
                continue;
            }
            checked_branches[index] = true;
            generation += 1;
            let mut stack = vec![taken, other];
            while let Some(next) = stack.pop() {
                if (next != index && postdom.postdominates(index, next))
                    || std::mem::replace(&mut stamp[next], generation) == generation
                {
                    continue;
                }
                for inst in &blocks[next].instructions {
                    if inst.is_convergent() {
                        if std::env::var_os("CFI_FLOW_DEBUG").is_some() {
                            eprintln!("convergent inst {inst:?} in block {next}");
                            let mut work: Vec<&str> = vec![predicate];
                            let mut seen_names = HashSet::new();
                            while let Some(n) = work.pop() {
                                if !seen_names.insert(n) { continue; }
                                if let Some((text, ins)) = tid_defs.get(n) {
                                    eprintln!("  {n} <- {text}");
                                    work.extend(ins.iter().copied());
                                }
                            }
                        }
                        return Err(Error::new(format!(
                            "barrier may execute divergently (branch on {predicate} in block {index})"
                        )));
                    }
                    let mut defined = Vec::new();
                    if inst.collect_definitions(&mut defined).is_some() {
                        varying.extend(defined.iter().copied());
                    }
                }
                stack.extend(blocks[next].end.successors());
            }
        }
        if old_count == varying.len() {
            return Ok(());
        }
    }
}
