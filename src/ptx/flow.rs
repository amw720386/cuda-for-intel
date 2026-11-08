use super::{Block, Comparison, Error, Instruction, IntegerOp, Terminator};
use super::fxhash::{FxHashMap as HashMap, FxHashSet as HashSet};

pub(super) struct Flow {
    pub(super) order: Vec<usize>,
}

impl Flow {
    pub(super) fn analyze(blocks: &[Block<'_>]) -> Result<Self, Error> {
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
            verify_barriers(blocks, &order)?;
        }

        Ok(Self { order })
    }
}

fn verify_barriers<'a>(blocks: &[Block<'a>], order: &[usize]) -> Result<(), Error> {
    // A branch's strict postdominators are guaranteed reconvergence points.
    let postdom = PostDom::new(blocks);

    // Treat memory loads and all definitions under divergent control as potentially varying.
    // Repeat because a loop-carried definition can affect an earlier branch.
    let mut varying = HashSet::default();
    // Values computed from `%tid.x` by ALU instructions alone (`%tid.y/z` conditions guard
    // rows that PTX programmers keep uniform by launching with `ntid.y == 1` where a block
    // cooperates along x; that cannot be proven statically) (no memory in between): a branch
    // on one of these is provably thread-divergent. Branches that are only "varying" because
    // of an unknown load or a divergent region are assumed to be taken by all threads, as
    // PTX programmers do for `__syncthreads` under data-dependent but block-uniform control.
    let mut tid_derived: HashSet<&str> = HashSet::default();
    let mut tid_defs: HashMap<&str, (String, Vec<&str>)> = HashMap::default();
    let mut local_derived: HashSet<&str> = HashSet::default();
    let mut checked_branches = vec![false; blocks.len()];
    let barrier_reach = barrier_reachability(blocks);
    let mut cyclic: Option<Vec<bool>> = None;
    // Value facts for block-uniform bounds. `K = n - n % ntid.x` is a multiple of the block size,
    // so `tid.x >= K` (or `<`) is the same for every thread of a block: all of them (K == 0) or
    // none (K >= ntid.x > tid.x). PyTorch's top-k / kthvalue loops are built on this.
    let mut tid_regs: HashSet<&str> = HashSet::default();
    let mut ntid_regs: HashSet<&str> = HashSet::default();
    let mut rem_of: HashMap<&str, &str> = HashMap::default();
    let mut multiple_of_ntid: HashSet<&str> = HashSet::default();
    let mut uniform_compare: HashSet<&str> = HashSet::default();
    // 64-bit variant: `for (i = tid; i < K; i += blockDim)` with `K = n - n % blockDim`. Every
    // thread's induction variable is congruent to its tid modulo blockDim, so comparing it with
    // a multiple of blockDim gives the same answer for the whole block.
    {
        let mut defs: std::collections::HashMap<&str, Vec<&Instruction>> = std::collections::HashMap::new();
        for block in blocks {
            for inst in &block.instructions {
                let mut d = Vec::new();
                if inst.collect_definitions(&mut d).is_some() {
                    for r in d {
                        defs.entry(r).or_default().push(inst);
                    }
                }
            }
        }
        let all = |r: &str, ok: &dyn Fn(&Instruction) -> bool| defs.get(r).is_some_and(|v| !v.is_empty() && v.iter().all(|i| ok(i)));
        // ntid as read from the builtin, widened (`cvt.u64.u32`) and narrowed again (`cvt.u32.u64`).
        let ntid_builtin = |r: &str| all(r, &|i| matches!(i, Instruction::Builtin(_, s) if s.starts_with("%ntid.x")));
        let ntid64 = |r: &str| all(r, &|i| matches!(i, Instruction::Widen { src, .. } if ntid_builtin(src)));
        let is_ntid32 = |r: &str| {
            ntid_builtin(r) || all(r, &|i| matches!(i, Instruction::Narrow(_, src) if ntid64(src)))
        };
        let is_ntid = |r: &str| ntid_builtin(r) || ntid64(r);
        // tid congruence (fixed point over self-increments)
        let is_tid32 = |r: &str| all(r, &|i| matches!(i, Instruction::Builtin(_, s) if s.starts_with("%tid.x")));
        let is_cong = |r: &str| {
            all(r, &|i| match i {
                Instruction::Builtin(_, s) => s.starts_with("%tid.x"),
                Instruction::Widen { src, .. } => is_tid32(src),
                Instruction::Add64(dst, [a, b]) => (dst == a && is_ntid(b)) || (dst == b && is_ntid(a)),
                _ => false,
            })
        };
        let rem_ok = |r: &str, a: &str| {
            all(r, &|i| match i {
                Instruction::Rem64 { signed: false, src: [a2, b], .. } => *a2 == a && is_ntid(b),
                Instruction::Widen { signed: false, src, .. } => all(src, &|j| {
                    matches!(j, Instruction::Integer32(IntegerOp::RemUnsigned, _, [_, b]) if is_ntid32(b))
                }),
                _ => false,
            })
        };
        let is_multiple = |k: &str| {
            all(k, &|i| matches!(i, Instruction::Sub64(_, [a, r]) if rem_ok(r, a)))
        };
        for block in blocks {
            for inst in &block.instructions {
                if let Instruction::Compare64 { op, dst, src: [x, y], .. } = inst {
                    let uniform = match op {
                        // x < K, x >= K
                        Comparison::Less | Comparison::GreaterEqual => is_cong(x) && is_multiple(y),
                        // K <= x, K > x
                        Comparison::LessEqual | Comparison::Greater => is_cong(y) && is_multiple(x),
                        _ => false,
                    };
                    if std::env::var_os("CFI_FLOW_DEBUG").is_some() && !uniform {
                        eprintln!("cmp64 {dst} {op:?} {x} {y}: cong(x)={} mult(y)={} cong(y)={} mult(x)={}", is_cong(x), is_multiple(y), is_cong(y), is_multiple(x));
                    }
                    if uniform {
                        uniform_compare.insert(*dst);
                    }
                }
            }
        }
    }
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
                    let base = |src: &&'a str| -> &'a str { src.split('+').next().unwrap_or(src) };
                    let thread_dependent = inputs.iter().any(|src| {
                        let b = base(src);
                        src.starts_with("%tid.") || varying.contains(src) || varying.contains(b)
                    });
                    let private = matches!(inst, super::Instruction::LocalSymbol64(..))
                        || inputs.iter().any(|src| local_derived.contains(base(src)));
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
                            src.starts_with("%tid.x") || tid_derived.contains(src) || tid_derived.contains(b)
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
            if uniform_compare.contains(predicate) {
                continue;
            }
            // Generalised early exit: if no barrier is reachable from one side, the threads
            // taking it only run to completion and exit, which the hardware barrier counts
            // as arrived (GGML's `if (col >= ncols) { ...no barrier...; return; }`).
            let reaches_barrier = |start: usize| barrier_reach[start];
            if !reaches_barrier(taken) || !reaches_barrier(other) || !tid_derived.contains(predicate) {
                continue;
            }
            // A tid-dependent *loop* test (`for (i = tid; i < rounded_up_n; i += blockDim)`):
            // CUDA requires all threads to run the same number of barrier-containing
            // iterations, which PyTorch's top-k / sort / scan helpers guarantee by rounding the
            // bound up to a multiple of the block size. Uniformity of the trip count is a
            // value fact we cannot prove, so trust the program as CUDA does.
            let in_loop = cyclic.get_or_insert_with(|| cyclic_blocks(blocks))[index];
            if in_loop && std::env::var_os("CFI_STRICT_BARRIERS").is_none() {
                continue;
            }
            checked_branches[index] = true;
            let mut seen = vec![false; blocks.len()];
            let mut stack = vec![taken, other];
            while let Some(next) = stack.pop() {
                if (next != index && postdom.postdominates(next, index))
                    || std::mem::replace(&mut seen[next], true)
                {
                    continue;
                }
                for inst in &blocks[next].instructions {
                    if inst.is_convergent() {
                        if std::env::var_os("CFI_FLOW_DEBUG").is_some() {
                            eprintln!("convergent inst {inst:?} in block {next}");
                            let mut work: Vec<&str> = vec![predicate];
                            let mut seen_names = std::collections::HashSet::new();
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

/// `reach[b]`: some block reachable from `b` (including `b`) holds a convergent instruction.
fn barrier_reachability(blocks: &[Block<'_>]) -> Vec<bool> {
    let mut predecessors = vec![Vec::new(); blocks.len()];
    for (index, block) in blocks.iter().enumerate() {
        for next in block.end.successors() {
            predecessors[next].push(index);
        }
    }
    let mut reach = vec![false; blocks.len()];
    let mut stack: Vec<usize> = (0..blocks.len())
        .filter(|&b| blocks[b].instructions.iter().any(|inst| inst.is_convergent()))
        .collect();
    while let Some(b) = stack.pop() {
        if !std::mem::replace(&mut reach[b], true) {
            stack.extend(predecessors[b].iter().copied());
        }
    }
    reach
}

/// `cyclic[b]`: `b` lies on a control-flow cycle (iterative Tarjan SCC; self-edges count).
fn cyclic_blocks(blocks: &[Block<'_>]) -> Vec<bool> {
    let n = blocks.len();
    let succ: Vec<Vec<usize>> = blocks.iter().map(|b| b.end.successors().collect()).collect();
    let mut cyclic = vec![false; n];
    let mut index = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut on_stack = vec![false; n];
    let mut scc_stack = Vec::new();
    let mut counter = 0;
    for root in 0..n {
        if index[root] != usize::MAX {
            continue;
        }
        let mut call = vec![(root, 0usize)];
        index[root] = counter;
        low[root] = counter;
        counter += 1;
        scc_stack.push(root);
        on_stack[root] = true;
        while let Some(&mut (v, ref mut at)) = call.last_mut() {
            if *at < succ[v].len() {
                let w = succ[v][*at];
                *at += 1;
                if w == v {
                    cyclic[v] = true;
                }
                if index[w] == usize::MAX {
                    index[w] = counter;
                    low[w] = counter;
                    counter += 1;
                    scc_stack.push(w);
                    on_stack[w] = true;
                    call.push((w, 0));
                } else if on_stack[w] {
                    low[v] = low[v].min(index[w]);
                }
            } else {
                call.pop();
                if let Some(&(parent, _)) = call.last() {
                    low[parent] = low[parent].min(low[v]);
                }
                if low[v] == index[v] {
                    let mut members = Vec::new();
                    loop {
                        let w = scc_stack.pop().unwrap();
                        on_stack[w] = false;
                        members.push(w);
                        if w == v {
                            break;
                        }
                    }
                    if members.len() > 1 {
                        for w in members {
                            cyclic[w] = true;
                        }
                    }
                }
            }
        }
    }
    cyclic
}

/// Post-dominator tree (Cooper-Harvey-Kennedy on the reverse CFG with a virtual exit) with
/// DFS intervals, so "does `a` post-dominate `b`" is O(1) instead of an N x N bit matrix.
struct PostDom {
    tin: Vec<u32>,
    tout: Vec<u32>,
}

impl PostDom {
    fn new(blocks: &[Block<'_>]) -> Self {
        let n = blocks.len();
        let exit = n;
        // Reverse-graph successors of a node are its forward predecessors; the virtual exit
        // precedes every Return block.
        let mut predecessors = vec![Vec::new(); n + 1];
        for (index, block) in blocks.iter().enumerate() {
            if matches!(block.end, Terminator::Return) {
                predecessors[exit].push(index);
            }
            for next in block.end.successors() {
                predecessors[next].push(index);
            }
        }
        // Reverse postorder over the reverse graph, rooted at the exit.
        let mut visited = vec![false; n + 1];
        let mut post = Vec::with_capacity(n + 1);
        let mut stack = vec![(exit, 0usize)];
        visited[exit] = true;
        while let Some(&mut (v, ref mut at)) = stack.last_mut() {
            if *at < predecessors[v].len() {
                let w = predecessors[v][*at];
                *at += 1;
                if !std::mem::replace(&mut visited[w], true) {
                    stack.push((w, 0));
                }
            } else {
                post.push(v);
                stack.pop();
            }
        }
        let mut number = vec![usize::MAX; n + 1];
        for (i, &v) in post.iter().enumerate() {
            number[v] = i;
        }
        // Forward successors (plus the exit for Return blocks) are the reverse-graph predecessors.
        let rev_preds = |v: usize| -> Vec<usize> {
            match blocks[v].end {
                Terminator::Return => vec![exit],
                end => end.successors().collect(),
            }
        };
        let mut idom = vec![usize::MAX; n + 1];
        idom[exit] = exit;
        let mut changed = true;
        while changed {
            changed = false;
            for &v in post.iter().rev() {
                if v == exit {
                    continue;
                }
                let mut new = usize::MAX;
                for p in rev_preds(v) {
                    if idom[p] == usize::MAX {
                        continue;
                    }
                    new = if new == usize::MAX {
                        p
                    } else {
                        let (mut a, mut b) = (p, new);
                        while a != b {
                            while number[a] < number[b] {
                                a = idom[a];
                            }
                            while number[b] < number[a] {
                                b = idom[b];
                            }
                        }
                        a
                    };
                }
                if new != idom[v] {
                    idom[v] = new;
                    changed = true;
                }
            }
        }
        // Tree children, then DFS intervals.
        let mut children = vec![Vec::new(); n + 1];
        for v in 0..n {
            if idom[v] != usize::MAX {
                children[idom[v]].push(v);
            }
        }
        let mut tin = vec![0u32; n + 1];
        let mut tout = vec![0u32; n + 1];
        let mut clock = 0u32;
        let mut stack = vec![(exit, 0usize)];
        tin[exit] = clock;
        clock += 1;
        while let Some(&mut (v, ref mut at)) = stack.last_mut() {
            if *at < children[v].len() {
                let w = children[v][*at];
                *at += 1;
                tin[w] = clock;
                clock += 1;
                stack.push((w, 0));
            } else {
                tout[v] = clock;
                clock += 1;
                stack.pop();
            }
        }
        Self { tin, tout }
    }

    /// `candidate` post-dominates `node` (every block post-dominates itself).
    fn postdominates(&self, candidate: usize, node: usize) -> bool {
        self.tin[candidate] <= self.tin[node] && self.tout[node] <= self.tout[candidate]
    }
}
