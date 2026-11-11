//! Recorded graph replay (`CUDA_FOR_INTEL_GRAPH_RECORD=1`, off by default).
//!
//! The default `cuGraphLaunch` re-issues every node as its own append to the stream's
//! immediate command list, so a 700-kernel decode graph costs 700 launch-path trips per token.
//! With recording on, the first launch of an executable graph (and the first launch after
//! `cuGraphExecUpdate` or an `cuGraphExec*SetParams` call changed it) appends its nodes to a
//! Level Zero regular command list in dependency order and closes it. That launch and every
//! later one run the list with a single `zeCommandListImmediateAppendCommandListsExp` on the
//! launch stream's immediate list. This is the extension Intel's SYCL graph support uses.
//! Kernel substitution (`subst_hook`) is decided at record time exactly as at launch time.
//!
//! A list is also re-recorded after any `cuMemFree` in the context (or module-variable
//! retirement): Intel's driver keeps the allocation set of a regular list's appends, and a pool
//! may hand the freed address back as a new allocation. Frees in another (peer) context are not
//! tracked, as the per-node validation never tracked them either.
//!
//! Recorded: kernel, device-to-device copy, memset and empty nodes (child graphs flattened).
//! A graph with host-function, event or host-memory copy nodes, a stream with experimental
//! batching, or a driver without the extension keeps the re-issue path, as does a graph
//! whose recording failed. Changing a graph re-records it; patching the closed list in place
//! (`ZE_experimental_mutable_command_list`) is not implemented. Its availability is only
//! reported in the export log.
use super::graphs::{CopyKind, Node, flatten};
use super::*;
use crate::level_zero::{CommandList, Fence};
use std::sync::Arc;

/// Driver extension behind `CommandList::append_lists`.
const IMMEDIATE_APPEND_EXTENSION: &str = "ZE_experimental_immediate_command_list_append";
/// Driver extensions that patch kernel arguments of a closed regular list in place
/// (`zeCommandListUpdateMutableCommandsExp`). Only reported for now.
const MUTABLE_LIST_EXTENSIONS: [&str; 2] = ["ZE_experimental_mutable_command_list", "ZE_extension_mutable_command_list"];

/// `CUDA_FOR_INTEL_GRAPH_RECORD=1` switches recording on. Read at every graph launch (one
/// lookup per graph, not per kernel) so a process can compare both paths.
pub(super) fn enabled() -> bool {
    #[cfg(test)]
    if let Some(forced) = TEST_ENABLED.get() {
        return forced;
    }
    std::env::var_os("CUDA_FOR_INTEL_GRAPH_RECORD").is_some_and(|value| value == "1")
}

#[cfg(test)]
thread_local! {
    pub(super) static TEST_ENABLED: std::cell::Cell<Option<bool>> = const { std::cell::Cell::new(None) };
}

struct Recorded {
    /// `Exec::version` the list was recorded from.
    version: u64,
    list: CommandList,
    /// Work-groups of all kernels, for the stream's in-flight bound.
    groups: u64,
    /// Completion of the latest run; the list is destroyed only after it.
    last: Option<Arc<Fence>>,
    /// `CudaContext::alloc_frees` when the list was recorded (after the nodes were validated).
    /// Any later free re-records: a regular list keeps the allocation set of its appends, and a
    /// pool can hand the freed address back as a different allocation.
    frees: u64,
}

enum Slot {
    Ready(Recorded),
    /// This version cannot be recorded; it uses the re-issue path.
    Unsupported(u64),
}

/// Recorded lists of one context, keyed by `CUgraphExec` handle.
#[derive(Default)]
pub(super) struct Recordings {
    live: HashMap<usize, Slot>,
    retired: Vec<(CommandList, Option<Arc<Fence>>)>,
    pub(super) stats: Stats,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(super) struct Stats {
    /// Lists recorded (first launches and re-recordings after a change).
    pub(super) recorded: usize,
    /// Graph launches that ran a recorded list.
    pub(super) replayed: usize,
    /// Graph launches that kept the re-issue path.
    pub(super) reissued: usize,
    /// Launches that waited for the previous run of the same list.
    pub(super) waited: usize,
}

impl Recordings {
    /// The graph `exec` was destroyed or changed: its list goes once its last run finished.
    pub(super) fn retire(&mut self, exec: usize) {
        if let Some(Slot::Ready(recorded)) = self.live.remove(&exec) {
            self.retired.push((recorded.list, recorded.last));
        }
    }

    /// Kernels or memory the lists refer to are going away (module unload).
    pub(super) fn retire_all(&mut self) {
        let execs: Vec<usize> = self.live.keys().copied().collect();
        for exec in execs {
            self.retire(exec);
        }
    }

    /// Whether `exec` at `version` has a recorded list that stays usable while the context has
    /// freed `frees` allocations. Its nodes were validated before recording and no allocation
    /// went away since, so they still are valid.
    pub(super) fn validated(&self, exec: usize, version: u64, frees: u64) -> bool {
        matches!(self.live.get(&exec), Some(Slot::Ready(recorded))
            if recorded.version == version && recorded.frees == frees)
    }

    /// Destroy retired lists whose last run finished.
    pub(super) fn collect(&mut self, api: &Api) -> Result<(), CuResult> {
        let mut index = 0;
        while index < self.retired.len() {
            let done = match &self.retired[index].1 {
                Some(fence) => fence.ready(api).map_err(backend_error)?,
                None => true,
            };
            if done {
                let (list, _) = self.retired.swap_remove(index);
                list.destroy(api).map_err(backend_error)?;
            } else {
                index += 1;
            }
        }
        Ok(())
    }

    /// Context teardown, after every stream drained.
    pub(super) fn destroy(&mut self, api: &Api) -> Result<(), CuResult> {
        self.retire_all();
        for (list, _) in self.retired.drain(..) {
            list.destroy(api).map_err(backend_error)?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn live_lists(&self) -> usize {
        self.live.values().filter(|slot| matches!(slot, Slot::Ready(_))).count() + self.retired.len()
    }
}

/// Whether every step can go into a regular list.
fn recordable(steps: &[&Node]) -> bool {
    steps.iter().any(|node| matches!(node, Node::Kernel(_)))
        && steps.iter().all(|node| match node {
            Node::Kernel(_) | Node::Fill(_) | Node::Empty => true,
            Node::Copy(copy) => copy.kind == CopyKind::DeviceToDevice,
            _ => false,
        })
}

fn supported(runtime: &Runtime, context: &CudaContext) -> bool {
    static REPORTED: std::sync::Once = std::sync::Once::new();
    let extensions = &runtime.devices[context.device as usize].info.driver_extensions;
    let has = |name: &str| extensions.iter().any(|(extension, _)| extension == name);
    #[cfg(test)]
    let fake = runtime.api.is_fake();
    #[cfg(not(test))]
    let fake = false;
    let append = runtime.api.append_lists.is_some() && (has(IMMEDIATE_APPEND_EXTENSION) || fake);
    REPORTED.call_once(|| {
        let mutable = MUTABLE_LIST_EXTENSIONS.iter().find(|name| has(name));
        export_log_always(format_args!(
            "graph record: immediate append {} (loader entry point {}), mutable command lists {}",
            if append { "available" } else { "UNAVAILABLE, graphs keep re-issuing nodes" },
            if runtime.api.append_lists.is_some() { "found" } else { "missing" },
            mutable.map_or("not reported", |name| *name),
        ));
    });
    append
}

/// Run executable graph `exec` (content `version`, nodes `nodes`) on stream `key` from a
/// recorded list. `Ok(false)`: not recordable here, the caller re-issues the nodes.
pub(super) fn launch(
    runtime: &Runtime,
    context: &mut CudaContext,
    exec: usize,
    version: u64,
    nodes: &[Node],
    key: usize,
) -> Result<bool, CuResult> {
    let api = &runtime.api;
    context.graph_lists.collect(api)?;
    let reuse = match context.graph_lists.live.get(&exec) {
        Some(Slot::Ready(recorded)) => recorded.version == version && recorded.frees == context.alloc_frees,
        Some(Slot::Unsupported(stale)) if *stale == version => {
            context.graph_lists.stats.reissued += 1;
            return Ok(false);
        }
        _ => false,
    };
    if !reuse {
        context.graph_lists.retire(exec);
        let mut steps = Vec::new();
        flatten(nodes, &mut steps);
        if !recordable(&steps) || !context.streams.is_immediate(key)? || !supported(runtime, context) {
            context.graph_lists.live.insert(exec, Slot::Unsupported(version));
            context.graph_lists.stats.reissued += 1;
            return Ok(false);
        }
        match record(api, context, &steps, key) {
            Ok((list, groups)) => {
                context.graph_lists.stats.recorded += 1;
                context.graph_lists.live.insert(
                    exec,
                    Slot::Ready(Recorded {
                        version,
                        list,
                        groups,
                        last: None,
                        frees: context.alloc_frees,
                    }),
                );
            }
            Err(error) => {
                export_log_always(format_args!(
                    "graph record: recording {} nodes failed ({error:?}); re-issuing them",
                    steps.len()
                ));
                context.graph_lists.live.insert(exec, Slot::Unsupported(version));
                context.graph_lists.stats.reissued += 1;
                return Ok(false);
            }
        }
    }
    let Some(Slot::Ready(recorded)) = context.graph_lists.live.get_mut(&exec) else {
        return Err(Unknown);
    };
    // One run of a list at a time: the Level Zero specification does not promise that a
    // regular list may be appended again while it executes. ggml synchronizes every token, so
    // the previous run has normally finished and this wait is a status query.
    if let Some(last) = &recorded.last
        && !last.ready(api).map_err(backend_error)?
    {
        context.graph_lists.stats.waited += 1;
        last.synchronize(api.event_wait_fn()).map_err(backend_error)?;
    }
    let fence = context
        .streams
        .launch_recorded(api, &context.backend, key, &recorded.list, recorded.groups)?;
    recorded.last = Some(fence);
    context.graph_lists.stats.replayed += 1;
    Ok(true)
}

/// Append `steps` to a new regular list for stream `key` and close it.
fn record(api: &Api, context: &mut CudaContext, steps: &[&Node], key: usize) -> Result<(CommandList, u64), CuResult> {
    let list = context
        .backend
        .create_list(api)
        .inspect_err(|error| context.backend.health.observe(error))
        .map_err(backend_error)?;
    let mut groups = 0u64;
    let result = (|| -> Result<(), CuResult> {
        for node in steps {
            match node {
                Node::Kernel(kernel) => {
                    groups = groups.saturating_add(kernel.geometry.0.iter().map(|&g| u64::from(g)).product());
                    kernel.record(api, context, &list, key)?;
                }
                Node::Copy(copy) => unsafe {
                    list.copy(
                        api,
                        (
                            std::ptr::with_exposed_provenance_mut(copy.dst),
                            std::ptr::with_exposed_provenance(copy.src),
                        ),
                        copy.size,
                        None,
                        &[],
                    )
                }
                .map_err(backend_error)?,
                Node::Fill(fill) => list
                    .fill(
                        api,
                        (std::ptr::with_exposed_provenance_mut(fill.dst), fill.value, fill.size),
                        None,
                        &[],
                    )
                    .map_err(backend_error)?,
                Node::Empty => {}
                _ => return Err(NotSupported),
            }
        }
        list.close(api).map_err(backend_error)
    })();
    match result {
        Ok(()) => Ok((list, groups)),
        Err(error) => {
            let _ = list.destroy(api);
            Err(error)
        }
    }
}
