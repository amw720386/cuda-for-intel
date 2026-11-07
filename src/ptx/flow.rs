use super::{Block, Error, Instruction, Terminator};
use std::collections::HashSet;

pub(super) struct Flow<'a> {
    pub(super) order: Vec<usize>,
    pub(super) inputs: Vec<HashSet<&'a str>>,
}

impl<'a> Flow<'a> {
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
            verify_barriers(blocks, &order)?;
        }

        let mut scratch = Vec::new();
        let definitions: Vec<HashSet<_>> = blocks
            .iter()
            .map(|block| {
                let mut defined = HashSet::new();
                for inst in &block.instructions {
                    if inst.collect_definitions(&mut scratch).is_some() {
                        defined.extend(scratch.iter().copied());
                    }
                }
                defined
            })
            .collect();
        let all: HashSet<_> = definitions.iter().flatten().copied().collect();
        let mut outputs = vec![all; blocks.len()];
        let mut inputs = vec![HashSet::new(); blocks.len()];
        loop {
            let mut changed = false;
            for &index in &order {
                inputs[index] = if index == 0 {
                    HashSet::new()
                } else {
                    let mut pred = predecessors[index].iter().copied();
                    let mut common = outputs[pred.next().expect("reachable block")].clone();
                    for prev in pred {
                        common.retain(|reg| outputs[prev].contains(reg));
                    }
                    common
                };
                let output = inputs[index].union(&definitions[index]).copied().collect();
                if outputs[index] != output {
                    outputs[index] = output;
                    changed = true;
                }
            }
            if !changed {
                return Ok(Self { order, inputs });
            }
        }
    }
}

fn verify_barriers(blocks: &[Block<'_>], order: &[usize]) -> Result<(), Error> {
    // A branch's strict postdominators are guaranteed reconvergence points.
    let mut postdominates = vec![vec![true; blocks.len()]; blocks.len()];
    loop {
        let mut changed = false;
        for &index in order.iter().rev() {
            let end = blocks[index].end;
            let set: Vec<_> = (0..blocks.len())
                .map(|candidate| {
                    candidate == index
                        || (!matches!(end, Terminator::Return)
                            && end.successors().all(|next| postdominates[next][candidate]))
                })
                .collect();
            if postdominates[index] != set {
                postdominates[index] = set;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    // Treat memory loads and all definitions under divergent control as potentially varying.
    // Repeat because a loop-carried definition can affect an earlier branch.
    let mut varying = HashSet::new();
    let mut local_derived: HashSet<&str> = HashSet::new();
    let mut checked_branches = vec![false; blocks.len()];
    loop {
        let old_count = varying.len();
        for &index in order {
            let block = &blocks[index];
            for inst in &block.instructions {
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
            checked_branches[index] = true;
            let mut seen = vec![false; blocks.len()];
            let mut stack = vec![taken, other];
            while let Some(next) = stack.pop() {
                if (next != index && postdominates[index][next])
                    || std::mem::replace(&mut seen[next], true)
                {
                    continue;
                }
                for inst in &blocks[next].instructions {
                    if inst.is_convergent() {
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
