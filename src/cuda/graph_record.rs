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
//! A list is also re-recorded when memory it uses was freed: Intel's driver keeps the
//! allocation set of a regular list's appends, and a pool may hand the freed address back as a
//! new allocation. "Uses" means the allocations that contain a pointer-sized kernel argument or a
//! copy or memset address of the graph, collected at record time. Frees of other memory keep the
//! list: Ollama allocates and frees its per-token input buffers around every graph launch.
//! The context keeps the last `FREE_LOG` frees; a list older than that is re-recorded. Frees in
//! another (peer) context are not tracked, as the per-node validation never tracked them either.
//!
//! Every decision is counted and the first ones of each kind, then a summary every
//! `SUMMARY_EVERY` launches, go to the export log as `graph record:` lines.
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
    /// `CudaContext::alloc_frees` up to which no free touched `uses` (the nodes were validated
    /// at that count). A later free of a used allocation re-records.
    frees: u64,
    /// Allocations (base, length) the recorded commands name, sorted.
    uses: Vec<(usize, usize)>,
}

/// Frees remembered per context for the overlap check.
pub(super) const FREE_LOG: usize = 4096;
/// Launches between two summary lines in the export log.
const SUMMARY_EVERY: usize = 1000;
/// Detailed log lines per kind of decision.
const DETAIL_LINES: usize = 8;

/// Frees of a context, for deciding whether a recorded list touches freed memory.
#[derive(Default)]
pub(super) struct FreeLog {
    /// (free number, base, length) of the latest frees, oldest first.
    entries: std::collections::VecDeque<(u64, usize, usize)>,
}

impl FreeLog {
    pub(super) fn push(&mut self, number: u64, base: usize, length: usize) {
        if self.entries.len() == FREE_LOG {
            self.entries.pop_front();
        }
        self.entries.push_back((number, base, length));
    }

    /// Whether a free numbered above `since` (up to `now`) overlaps one of `uses`. `None` when
    /// the log no longer holds all of them.
    pub(super) fn overlaps_since(&self, since: u64, now: u64, uses: &[(usize, usize)]) -> Option<bool> {
        let retained = self.entries.front().map_or(now + 1, |&(number, _, _)| number);
        if since + 1 < retained && since < now {
            return None;
        }
        Some(self.entries.iter().filter(|&&(number, _, _)| number > since).any(|&(_, base, length)| {
            uses.iter().any(|&(used, used_length)| {
                base < used.saturating_add(used_length.max(1)) && used < base.saturating_add(length.max(1))
            })
        }))
    }
}

/// Why a launch recorded a list or kept re-issuing nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Event {
    Replayed,
    RecordedFirst,
    RecordedChanged,
    RecordedFreed,
    RecordedFreeLogOverflow,
    ReissuedNodes,
    ReissuedBatchingStream,
    ReissuedNoExtension,
    ReissuedRecordFailed,
    ReissuedCached,
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
    events: EventLog,
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
    /// Frees of other memory that a recorded list survived.
    pub(super) kept_after_free: usize,
}

#[derive(Default)]
struct EventLog {
    counts: HashMap<Event, usize>,
    launches: usize,
}

impl EventLog {
    fn note(&mut self, event: Event, detail: impl FnOnce() -> String) {
        let count = self.counts.entry(event).or_insert(0);
        *count += 1;
        if *count <= DETAIL_LINES {
            export_log_always(format_args!("graph record: {event:?} #{count}: {}", detail()));
        }
        self.launches += 1;
        if self.launches % SUMMARY_EVERY == 0 {
            let mut counts: Vec<_> = self.counts.iter().map(|(event, count)| format!("{event:?} {count}")).collect();
            counts.sort();
            export_log_always(format_args!("graph record: {} launches: {}", self.launches, counts.join(", ")));
        }
    }
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

/// [`unrecordable`] for the nodes of a graph (child graphs flattened).
#[cfg(test)]
pub(super) fn unrecordable_kinds(nodes: &[Node]) -> Option<String> {
    let mut steps = Vec::new();
    flatten(nodes, &mut steps);
    unrecordable(&steps)
}

/// `None` when every step can go into a regular list, else the kinds of the steps that cannot.
fn unrecordable(steps: &[&Node]) -> Option<String> {
    let mut kinds = std::collections::BTreeMap::<&str, usize>::new();
    for node in steps {
        let kind = match node {
            Node::Kernel(_) | Node::Fill(_) | Node::Empty => continue,
            Node::Copy(copy) => match copy.kind {
                CopyKind::DeviceToDevice => continue,
                CopyKind::HostToDevice => "host-to-device copy",
                CopyKind::DeviceToHost => "device-to-host copy",
            },
            Node::Host(_) => "host function",
            Node::EventRecord(_) => "event record",
            Node::EventWait(_) => "event wait",
            Node::Child(_) => "child graph",
        };
        *kinds.entry(kind).or_default() += 1;
    }
    if !steps.iter().any(|node| matches!(node, Node::Kernel(_))) {
        *kinds.entry("no kernel").or_default() += 1;
    }
    (!kinds.is_empty()).then(|| format!("{} nodes, unrecordable: {kinds:?}", steps.len()))
}

/// The allocation (base, length) of `map` that contains `address`.
fn containing(map: &BTreeMap<usize, usize>, address: usize) -> Option<(usize, usize)> {
    map.range(..=address)
        .next_back()
        .filter(|&(&base, &length)| address < base.saturating_add(length.max(1)))
        .map(|(&base, &length)| (base, length))
}

/// Allocations named by `steps`: every 8-byte kernel argument that points into one (pointer
/// parameters and pointers passed as integers alike), and copy and memset addresses.
/// A pointer inside an aggregate (`.b8` array) parameter is not scanned: such kernels always keep
/// indirect USM access (the indirect-access analysis never treats `.b8` parameters as direct
/// pointers, so even `INDIRECT_ACCESS=auto` leaves them indirect), and staged parameters are
/// refused under capture.
fn used_allocations(context: &CudaContext, steps: &[&Node]) -> Vec<(usize, usize)> {
    let lookup = |address: usize| {
        containing(&context.allocations, address).or_else(|| containing(&context.host_allocations, address))
    };
    let mut uses = Vec::new();
    for node in steps {
        match node {
            Node::Kernel(kernel) => {
                for arg in kernel.args.iter().filter(|arg| arg.len() == 8) {
                    let value = unsafe { arg.as_ptr().cast::<u64>().read_unaligned() } as usize;
                    uses.extend(lookup(value));
                }
            }
            Node::Copy(copy) => uses.extend([lookup(copy.dst), lookup(copy.src)].into_iter().flatten()),
            Node::Fill(fill) => uses.extend(lookup(fill.dst)),
            _ => {}
        }
    }
    uses.sort_unstable();
    uses.dedup();
    uses
}

impl CudaContext {
    /// Count a free of [`base`, `base + length`) for recorded graphs (`graph_record`).
    pub(super) fn note_free(&mut self, base: usize, length: usize) {
        self.alloc_frees += 1;
        self.freed.push(self.alloc_frees, base, length);
    }
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
    let now = context.alloc_frees;
    let reason = match context.graph_lists.live.get(&exec) {
        None => Some(Event::RecordedFirst),
        Some(Slot::Unsupported(stale)) if *stale == version => {
            context.graph_lists.stats.reissued += 1;
            context.graph_lists.events.note(Event::ReissuedCached, || format!("graph {exec:#x}"));
            return Ok(false);
        }
        Some(Slot::Unsupported(_)) => Some(Event::RecordedChanged),
        Some(Slot::Ready(recorded)) if recorded.version != version => Some(Event::RecordedChanged),
        Some(Slot::Ready(recorded)) if recorded.frees != now => {
            match context.freed.overlaps_since(recorded.frees, now, &recorded.uses) {
                Some(true) => Some(Event::RecordedFreed),
                None => Some(Event::RecordedFreeLogOverflow),
                Some(false) => None,
            }
        }
        Some(Slot::Ready(_)) => None,
    };
    if let Some(reason) = reason {
        context.graph_lists.retire(exec);
        let mut steps = Vec::new();
        flatten(nodes, &mut steps);
        let refused = if let Some(why) = unrecordable(&steps) {
            Some((Event::ReissuedNodes, why))
        } else if !context.streams.is_immediate(key)? {
            Some((Event::ReissuedBatchingStream, "stream uses experimental batching".to_string()))
        } else if !supported(runtime, context) {
            Some((Event::ReissuedNoExtension, "no immediate-append extension".to_string()))
        } else {
            None
        };
        if let Some((event, why)) = refused {
            context.graph_lists.live.insert(exec, Slot::Unsupported(version));
            context.graph_lists.stats.reissued += 1;
            context.graph_lists.events.note(event, || format!("graph {exec:#x}: {why}"));
            return Ok(false);
        }
        let uses = used_allocations(context, &steps);
        // Level Zero captures kernel arguments at append, so cached values would be correct here
        // too, but recording is rare and must not depend on what immediate launches left in the
        // argument cache: drop it so every recorded launch sets all of its arguments, and drop
        // it again so the next immediate launches do not rely on state the recording changed.
        crate::level_zero::invalidate_arg_cache();
        let recorded = record(api, context, &steps, key);
        crate::level_zero::invalidate_arg_cache();
        match recorded {
            Ok((list, groups)) => {
                context.graph_lists.stats.recorded += 1;
                let count = steps.len();
                context.graph_lists.events.note(reason, || {
                    format!("graph {exec:#x}: {count} nodes, {} allocations used, frees so far {now}", uses.len())
                });
                context.graph_lists.live.insert(
                    exec,
                    Slot::Ready(Recorded {
                        version,
                        list,
                        groups,
                        last: None,
                        frees: now,
                        uses,
                    }),
                );
            }
            Err(error) => {
                let count = steps.len();
                context.graph_lists.live.insert(exec, Slot::Unsupported(version));
                context.graph_lists.stats.reissued += 1;
                context.graph_lists.events.note(Event::ReissuedRecordFailed, || {
                    format!("graph {exec:#x}: recording {count} nodes failed ({error:?})")
                });
                return Ok(false);
            }
        }
    } else {
        // No free since recording touched memory this list uses.
        if let Some(Slot::Ready(recorded)) = context.graph_lists.live.get_mut(&exec)
            && recorded.frees != now
        {
            recorded.frees = now;
            context.graph_lists.stats.kept_after_free += 1;
        }
        context.graph_lists.events.note(Event::Replayed, || format!("graph {exec:#x}"));
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
