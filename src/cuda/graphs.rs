//! Graph model, stream capture and replay of supported CUDA operations.
//!
//! A graph is a [`Dag`]: nodes with handles and dependency edges. Capture
//! records kernel launches, memcpy, memset, host functions and
//! nested graphs; every stream keeps the set of nodes its next operation
//! depends on, and events carry that set from one stream to another, so
//! fork/join patterns produce real edges. `cuStreamGetCaptureInfo` reports the
//! graph being built and these dependencies, and
//! `cuStreamUpdateCaptureDependencies` edits them. Instantiation orders nodes
//! topologically (ready nodes first, earliest created first) and
//! `cuGraphLaunch` issues them on the launch stream in that order. Independent
//! branches therefore run in program order on one stream, which is always a
//! legal schedule. Host functions run between submitted segments with the
//! runtime lock released. The explicit construction API is in `graph_api.rs`.
//! Conditional and memory-allocation nodes are not implemented.

use super::*;
use crate::level_zero::KernelArg;
use std::cell::Cell;
use std::collections::{BTreeSet, HashSet};
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use std::thread::ThreadId;

#[derive(Default)]
pub(super) struct Graphs {
    pub(super) graphs: HashMap<usize, Dag>,
    pub(super) executable: HashMap<usize, Exec>,
    /// Argument blocks handed out by `cuGraphKernelNodeGetParams`, valid until
    /// the next query of the same node.
    pub(super) param_storage: HashMap<usize, (Vec<Box<[u8]>>, Vec<usize>)>,
}

impl Graphs {
    pub(super) fn drop_context(&mut self, context: usize) {
        self.graphs.retain(|_, graph| graph.context != context);
        self.executable.retain(|_, graph| graph.context != context);
    }

    /// The graph that owns `node`.
    pub(super) fn owner(&self, node: usize) -> Option<usize> {
        self.graphs
            .iter()
            .find(|(_, dag)| dag.find(node).is_some())
            .map(|(&handle, _)| handle)
    }
}

/// Node handles are unique across graphs and never collide with other handles.
static NEXT_NODE: AtomicUsize = AtomicUsize::new(0x4000_0000_0001);

pub(super) fn new_node_id() -> usize {
    NEXT_NODE.fetch_add(1, Ordering::Relaxed)
}

/// One node of a graph under construction with the nodes it depends on.
#[derive(Clone)]
pub(super) struct Entry {
    pub(super) id: usize,
    pub(super) node: Node,
    pub(super) deps: Vec<usize>,
}

/// A graph with node handles and dependency edges, in node creation order.
#[derive(Clone, Default)]
pub(super) struct Dag {
    pub(super) context: usize,
    pub(super) entries: Vec<Entry>,
    /// A stream capture is appending nodes to this graph.
    pub(super) capturing: bool,
    /// Original node -> node in this graph, for graphs made by `cuGraphClone`.
    pub(super) clone_map: HashMap<usize, usize>,
}

impl Dag {
    pub(super) fn new(context: usize) -> Self {
        Self {
            context,
            ..Self::default()
        }
    }

    pub(super) fn find(&self, id: usize) -> Option<usize> {
        self.entries.iter().position(|entry| entry.id == id)
    }

    pub(super) fn entry(&self, id: usize) -> Result<&Entry, CuResult> {
        self.find(id)
            .map(|index| &self.entries[index])
            .ok_or(InvalidValue)
    }

    pub(super) fn entry_mut(&mut self, id: usize) -> Result<&mut Entry, CuResult> {
        let index = self.find(id).ok_or(InvalidValue)?;
        Ok(&mut self.entries[index])
    }

    /// Add a node depending on `deps`, which must be distinct nodes of this graph.
    pub(super) fn add(&mut self, node: Node, deps: &[usize]) -> Result<usize, CuResult> {
        let mut unique = Vec::with_capacity(deps.len());
        for &dep in deps {
            if self.find(dep).is_none() || unique.contains(&dep) {
                return Err(InvalidValue);
            }
            unique.push(dep);
        }
        let id = new_node_id();
        self.entries.push(Entry {
            id,
            node,
            deps: unique,
        });
        Ok(id)
    }

    pub(super) fn children(&self, id: usize) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|entry| entry.deps.contains(&id))
            .map(|entry| entry.id)
            .collect()
    }

    pub(super) fn roots(&self) -> Vec<usize> {
        self.entries
            .iter()
            .filter(|entry| entry.deps.is_empty())
            .map(|entry| entry.id)
            .collect()
    }

    /// Whether `from` already reaches `to` (so an edge `to -> from` would close a cycle).
    pub(super) fn reaches(&self, from: usize, to: usize) -> bool {
        let mut seen = HashSet::new();
        let mut stack = vec![from];
        while let Some(id) = stack.pop() {
            if id == to {
                return true;
            }
            if seen.insert(id) {
                stack.extend(self.children(id));
            }
        }
        false
    }

    pub(super) fn add_edge(&mut self, from: usize, to: usize) -> Result<(), CuResult> {
        self.find(from).ok_or(InvalidValue)?;
        self.find(to).ok_or(InvalidValue)?;
        if self.entry(to)?.deps.contains(&from) || from == to || self.reaches(to, from) {
            return Err(InvalidValue);
        }
        self.entry_mut(to)?.deps.push(from);
        Ok(())
    }

    pub(super) fn remove_edge(&mut self, from: usize, to: usize) -> Result<(), CuResult> {
        let deps = &mut self.entry_mut(to)?.deps;
        let position = deps.iter().position(|&dep| dep == from).ok_or(InvalidValue)?;
        deps.remove(position);
        Ok(())
    }

    pub(super) fn remove_node(&mut self, id: usize) -> Result<(), CuResult> {
        let index = self.find(id).ok_or(InvalidValue)?;
        self.entries.remove(index);
        for entry in &mut self.entries {
            entry.deps.retain(|&dep| dep != id);
        }
        Ok(())
    }

    /// Indices in dependency order. A node is issued as soon as all of its
    /// dependencies have been, earliest-created ready node first.
    pub(super) fn order(&self) -> Vec<usize> {
        let position: HashMap<usize, usize> = self
            .entries
            .iter()
            .enumerate()
            .map(|(index, entry)| (entry.id, index))
            .collect();
        let mut waiting: Vec<usize> = self.entries.iter().map(|entry| entry.deps.len()).collect();
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); self.entries.len()];
        for (index, entry) in self.entries.iter().enumerate() {
            for dep in &entry.deps {
                dependents[position[dep]].push(index);
            }
        }
        let mut ready: BTreeSet<usize> = (0..self.entries.len())
            .filter(|&index| waiting[index] == 0)
            .collect();
        let mut order = Vec::with_capacity(self.entries.len());
        while let Some(index) = ready.pop_first() {
            order.push(index);
            for &next in &dependents[index] {
                waiting[next] -= 1;
                if waiting[next] == 0 {
                    ready.insert(next);
                }
            }
        }
        order
    }

    /// The executable form: nodes in dependency order with their handles.
    pub(super) fn exec(&self) -> Exec {
        let order = self.order();
        Exec {
            context: self.context,
            nodes: order.iter().map(|&i| self.entries[i].node.clone()).collect(),
            ids: order.iter().map(|&i| self.entries[i].id).collect(),
            version: next_exec_version(),
        }
    }

    pub(super) fn flatten(&self) -> Arc<[Node]> {
        self.exec().nodes
    }

    /// Copy with fresh node handles.
    pub(super) fn duplicate(&self) -> Dag {
        let mut map = HashMap::new();
        let mut copy = Dag::new(self.context);
        for entry in &self.entries {
            let id = new_node_id();
            map.insert(entry.id, id);
            copy.entries.push(Entry {
                id,
                node: entry.node.clone(),
                deps: entry.deps.iter().map(|dep| map[dep]).collect(),
            });
        }
        copy.clone_map = map;
        copy
    }
}

/// An instantiated graph: nodes in dependency order and the handles of the
/// graph nodes they came from.
#[derive(Clone)]
pub(super) struct Exec {
    pub(super) context: usize,
    pub(super) nodes: Arc<[Node]>,
    pub(super) ids: Arc<[usize]>,
    /// Unique per content: every change of `nodes` takes a new value, so a command list
    /// recorded from this graph (`graph_record`) is stale when the versions differ.
    pub(super) version: u64,
}

static EXEC_VERSION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

fn next_exec_version() -> u64 {
    EXEC_VERSION.fetch_add(1, Ordering::Relaxed)
}

impl Exec {
    pub(super) fn replace(&mut self, id: usize, node: Node) -> Result<(), CuResult> {
        let index = self.ids.iter().position(|&candidate| candidate == id).ok_or(InvalidValue)?;
        let mut nodes = self.nodes.to_vec();
        nodes[index] = node;
        self.nodes = nodes.into();
        self.version = next_exec_version();
        Ok(())
    }

    pub(super) fn node(&self, id: usize) -> Result<&Node, CuResult> {
        let index = self.ids.iter().position(|&candidate| candidate == id).ok_or(InvalidValue)?;
        Ok(&self.nodes[index])
    }
}

/// A host function launched by a graph or stream.
#[derive(Clone, Copy)]
pub(super) struct HostFn {
    pub(super) function: usize,
    pub(super) data: usize,
}

#[derive(Clone)]
pub(super) enum Node {
    Kernel(Kernel),
    Copy(Copy),
    Fill(Fill),
    Empty,
    Host(HostFn),
    EventRecord(usize),
    EventWait(usize),
    /// A previously captured graph launched into a capturing stream, or a
    /// child graph node.
    Child(Arc<[Node]>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CopyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

#[derive(Clone)]
pub(super) struct Copy {
    pub(super) dst: usize,
    pub(super) src: usize,
    pub(super) size: usize,
    pub(super) kind: CopyKind,
}

#[derive(Clone)]
pub(super) struct Fill {
    pub(super) dst: usize,
    pub(super) value: u8,
    pub(super) size: usize,
}

impl Node {
    pub(super) fn validate(&self, context: &CudaContext) -> Result<(), CuResult> {
        match self {
            Node::Kernel(kernel) => kernel.validate(context),
            Node::Copy(copy) => {
                let device = |address| contains_range(&context.allocations, address, copy.size);
                let valid = match copy.kind {
                    CopyKind::HostToDevice => device(copy.dst) && copy.src != 0,
                    CopyKind::DeviceToHost => device(copy.src) && copy.dst != 0,
                    CopyKind::DeviceToDevice => device(copy.dst) && device(copy.src),
                };
                valid.then_some(()).ok_or(InvalidValue)
            }
            Node::Fill(fill) => contains_range(&context.allocations, fill.dst, fill.size)
                .then_some(())
                .ok_or(InvalidValue),
            Node::Empty | Node::Host(_) | Node::EventRecord(_) | Node::EventWait(_) => Ok(()),
            Node::Child(nodes) => nodes.iter().try_for_each(|node| node.validate(context)),
        }
    }

    pub(super) fn replay(&self, runtime: &Runtime, context: &mut CudaContext, stream: usize) -> Result<(), CuResult> {
        match self {
            Node::Kernel(kernel) => kernel.replay(&runtime.api, context, stream),
            Node::Copy(copy) => match copy.kind {
                CopyKind::HostToDevice => unsafe {
                    copy_host_async(
                        runtime,
                        context,
                        stream,
                        std::ptr::with_exposed_provenance_mut(copy.src),
                        copy.dst as u64,
                        copy.size,
                        true,
                    )
                },
                CopyKind::DeviceToHost => unsafe {
                    copy_host_async(
                        runtime,
                        context,
                        stream,
                        std::ptr::with_exposed_provenance_mut(copy.dst),
                        copy.src as u64,
                        copy.size,
                        false,
                    )
                },
                CopyKind::DeviceToDevice => unsafe {
                    context.streams.copy(
                        &runtime.api,
                        &context.backend,
                        stream,
                        (
                            std::ptr::with_exposed_provenance_mut(copy.dst),
                            std::ptr::with_exposed_provenance(copy.src),
                        ),
                        copy.size,
                    )
                },
            },
            Node::Fill(fill) => context.streams.fill(
                &runtime.api,
                &context.backend,
                stream,
                (
                    std::ptr::with_exposed_provenance_mut(fill.dst),
                    fill.value,
                    fill.size,
                ),
            ),
            Node::Empty => Ok(()),
            // Host functions run between segments in `cuGraphLaunch`, outside the lock.
            Node::Host(_) => Ok(()),
            Node::EventRecord(event) => {
                context
                    .streams
                    .replay_event_record(&runtime.api, &context.backend, *event, stream)
            }
            Node::EventWait(event) => {
                context
                    .streams
                    .replay_event_wait(&runtime.api, &context.backend, *event, stream)
            }
            Node::Child(nodes) => nodes
                .iter()
                .try_for_each(|node| node.replay(runtime, context, stream)),
        }
    }

    /// Whether `new` can replace `self` in an executable graph. The error is a
    /// CUgraphExecUpdateResult code.
    fn updatable(&self, new: &Node) -> Result<(), c_int> {
        const TOPOLOGY: c_int = 2;
        const NODE_TYPE: c_int = 3;
        const FUNCTION: c_int = 4;
        const PARAMETERS: c_int = 5;
        match (self, new) {
            (Node::Kernel(old), Node::Kernel(new)) => {
                if old.function == new.function {
                    Ok(())
                } else {
                    Err(FUNCTION)
                }
            }
            (Node::Copy(old), Node::Copy(new)) => {
                if old.kind == new.kind {
                    Ok(())
                } else {
                    Err(PARAMETERS)
                }
            }
            (Node::Fill(_), Node::Fill(_)) | (Node::Empty, Node::Empty) => Ok(()),
            (Node::Host(_), Node::Host(_)) => Ok(()),
            (Node::EventRecord(_), Node::EventRecord(_)) | (Node::EventWait(_), Node::EventWait(_)) => Ok(()),
            (Node::Child(old), Node::Child(new)) => nodes_updatable(old, new),
            _ => {
                let _ = TOPOLOGY;
                Err(NODE_TYPE)
            }
        }
    }
}

fn nodes_updatable(old: &[Node], new: &[Node]) -> Result<(), c_int> {
    if old.len() != new.len() {
        return Err(2);
    }
    old.iter().zip(new).try_for_each(|(old, new)| old.updatable(new))
}

#[derive(Default)]
struct Member {
    /// Operations issued on this stream inside the capture.
    sequence: u64,
    /// Sequence covered by an event another capturing stream waited for.
    joined: Option<u64>,
}

struct Inner {
    id: u64,
    origin: usize,
    #[allow(dead_code)]
    mode: c_int,
    /// Graph the nodes are appended to when the capture ends.
    graph: usize,
    /// The capture appends to a caller-supplied graph (`cuStreamBeginCaptureToGraph`).
    to_graph: bool,
    invalidated: bool,
    entries: Vec<Entry>,
    members: HashMap<usize, Member>,
    /// Nodes the next operation on each stream depends on.
    frontier: HashMap<usize, Vec<usize>>,
    /// Frontier of the stream when each event was recorded.
    snapshots: HashMap<usize, Vec<usize>>,
    /// Storage for the dependency list handed out by `cuStreamGetCaptureInfo`.
    reported: Vec<usize>,
}

/// A capture session shared by the origin stream and every stream forked into it.
#[derive(Clone)]
pub(super) struct Capture(Arc<Mutex<Inner>>);

/// Captures in progress, for rejecting calls that are unsafe during capture.
static ACTIVE: Mutex<Vec<(Weak<Mutex<Inner>>, ThreadId, c_int)>> = Mutex::new(Vec::new());

impl Capture {
    pub(super) fn new(
        id: u64,
        origin: usize,
        mode: c_int,
        graph: usize,
        to_graph: bool,
        dependencies: Vec<usize>,
    ) -> Self {
        let mut members = HashMap::new();
        members.insert(origin, Member::default());
        let thread = std::thread::current().id();
        let inner = Arc::new(Mutex::new(Inner {
            id,
            origin,
            mode,
            graph,
            to_graph,
            invalidated: false,
            entries: Vec::new(),
            members,
            frontier: HashMap::from([(origin, dependencies)]),
            snapshots: HashMap::new(),
            reported: Vec::new(),
        }));
        let mut active = ACTIVE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        active.retain(|(weak, _, _)| weak.strong_count() > 0);
        active.push((Arc::downgrade(&inner), thread, mode));
        Self(inner)
    }

    fn inner(&self) -> MutexGuard<'_, Inner> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    pub(super) fn same(&self, other: &Capture) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }

    pub(super) fn id(&self) -> u64 {
        self.inner().id
    }

    pub(super) fn graph(&self) -> usize {
        self.inner().graph
    }

    pub(super) fn to_graph(&self) -> bool {
        self.inner().to_graph
    }

    /// Whether `node` was captured so far.
    pub(super) fn contains(&self, node: usize) -> bool {
        self.inner().entries.iter().any(|entry| entry.id == node)
    }

    pub(super) fn is_origin(&self, stream: usize) -> bool {
        self.inner().origin == stream
    }

    pub(super) fn is_invalidated(&self) -> bool {
        self.inner().invalidated
    }

    pub(super) fn members(&self) -> Vec<usize> {
        self.inner().members.keys().copied().collect()
    }

    /// Append a node produced by `stream`, depending on the stream's frontier.
    pub(super) fn push(&self, stream: usize, node: Node) -> Result<(), CuResult> {
        self.push_node(stream, node).map(|_| ())
    }

    pub(super) fn push_node(&self, stream: usize, node: Node) -> Result<usize, CuResult> {
        let mut inner = self.inner();
        if inner.invalidated {
            return Err(StreamCaptureInvalidated);
        }
        inner.members.entry(stream).or_default().sequence += 1;
        let id = new_node_id();
        let deps = inner.frontier.insert(stream, vec![id]).unwrap_or_default();
        inner.entries.push(Entry { id, node, deps });
        Ok(id)
    }

    /// Graph and dependencies reported for `stream`; the list stays valid until
    /// the next call.
    pub(super) fn info(&self, stream: usize) -> (usize, *const Opaque, usize) {
        let mut inner = self.inner();
        let frontier = inner.frontier.get(&stream).cloned().unwrap_or_default();
        inner.reported = frontier;
        (
            inner.graph,
            inner.reported.as_ptr().cast(),
            inner.reported.len(),
        )
    }

    /// Replace (`set`) or extend the dependencies of the next operation on `stream`.
    pub(super) fn update_dependencies(&self, stream: usize, nodes: &[usize], set: bool) {
        let mut inner = self.inner();
        let frontier = inner.frontier.entry(stream).or_default();
        if set {
            frontier.clear();
        }
        for &node in nodes {
            if !frontier.contains(&node) {
                frontier.push(node);
            }
        }
    }

    pub(super) fn invalidate(&self) -> CuResult {
        self.invalidate_as(StreamCaptureUnsupported)
    }

    pub(super) fn invalidate_as(&self, error: CuResult) -> CuResult {
        self.inner().invalidated = true;
        error
    }

    /// Mark `stream` as a member (a fork). Returns false if it already was.
    pub(super) fn join(&self, stream: usize) -> bool {
        let mut inner = self.inner();
        let new = !inner.members.contains_key(&stream);
        inner.members.entry(stream).or_default();
        new
    }

    /// Sequence number an event recorded now on `stream` stands for.
    pub(super) fn record_event(&self, stream: usize) -> u64 {
        self.inner().members.entry(stream).or_default().sequence
    }

    /// Remember what `event`, recorded now on `stream`, depends on.
    pub(super) fn snapshot_event(&self, event: usize, stream: usize) {
        let mut inner = self.inner();
        let frontier = inner.frontier.get(&stream).cloned().unwrap_or_default();
        inner.snapshots.insert(event, frontier);
    }

    /// `waiter` waited for `event`: its next operation depends on the event's nodes.
    pub(super) fn apply_wait(&self, event: usize, waiter: usize) {
        let mut inner = self.inner();
        let snapshot = inner.snapshots.get(&event).cloned().unwrap_or_default();
        let frontier = inner.frontier.entry(waiter).or_default();
        for node in snapshot {
            if !frontier.contains(&node) {
                frontier.push(node);
            }
        }
    }

    /// `waiter` waited for an event recorded on `producer` at `sequence`.
    pub(super) fn note_wait(&self, producer: usize, waiter: usize, sequence: u64) {
        if producer == waiter {
            return;
        }
        if let Some(member) = self.inner().members.get_mut(&producer)
            && member.sequence == sequence
        {
            member.joined = Some(sequence);
        }
    }

    /// Finish the session, returning the captured nodes.
    fn finish(&self) -> Result<Vec<Entry>, CuResult> {
        let mut inner = self.inner();
        if inner.invalidated {
            return Err(StreamCaptureInvalidated);
        }
        let origin = inner.origin;
        if inner
            .members
            .iter()
            .any(|(&stream, member)| stream != origin && member.joined != Some(member.sequence))
        {
            return Err(StreamCaptureUnjoined);
        }
        Ok(std::mem::take(&mut inner.entries))
    }
}

/// Reject a call that CUDA forbids while a capture could be affected: global
/// captures forbid it on every thread, thread-local ones on the capturing
/// thread, and relaxed ones never. The calling thread's mode, set with
/// `cuThreadExchangeStreamCaptureMode`, can opt out. The capture is
/// invalidated, as in CUDA.
pub(super) fn unsafe_call() -> Result<(), CuResult> {
    let thread_mode = THREAD_MODE.get();
    if thread_mode == 2 {
        return Ok(());
    }
    let me = std::thread::current().id();
    let mut active = ACTIVE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    active.retain(|(weak, _, _)| weak.strong_count() > 0);
    for (weak, thread, mode) in active.iter() {
        let hit = (*mode == 0 && thread_mode == 0) || (*mode != 2 && *thread == me);
        if hit && let Some(inner) = weak.upgrade() {
            return Err(Capture(inner).invalidate());
        }
    }
    Ok(())
}

fn deactivate(capture: &Capture) {
    let mut active = ACTIVE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    active.retain(|(weak, _, _)| weak.strong_count() > 0 && !std::ptr::eq(weak.as_ptr(), Arc::as_ptr(&capture.0)));
}

#[derive(Clone)]
pub(super) struct Kernel {
    pub(super) function: usize,
    pub(super) geometry: ([u32; 3], [u32; 3]),
    pub(super) shared: u32,
    pub(super) args: Vec<Box<[MaybeUninit<u8>]>>,
}

impl Kernel {
    pub(super) unsafe fn capture(
        function: usize,
        geometry: ([u32; 3], [u32; 3]),
        args: &[KernelArg],
    ) -> Self {
        let args = args
            .iter()
            .map(|arg| {
                // Aggregate parameters can contain uninitialized padding.
                let mut bytes = Box::<[u8]>::new_uninit_slice(arg.size);
                unsafe {
                    std::ptr::copy_nonoverlapping(arg.value.cast(), bytes.as_mut_ptr(), arg.size)
                };
                bytes
            })
            .collect();
        Self {
            function,
            geometry,
            shared: 0,
            args,
        }
    }

    fn entry<'a>(&self, context: &'a CudaContext) -> Result<&'a CudaEntry, CuResult> {
        let FunctionRef { module, name, .. } = context.functions.get(&self.function).ok_or(InvalidHandle)?;
        let module = context.modules.get(module).ok_or(InvalidHandle)?;
        module.entries.get(name).ok_or(InvalidHandle)
    }

    pub(super) fn validate(&self, context: &CudaContext) -> Result<(), CuResult> {
        let entry = self.entry(context)?;
        entry.function.ok_or(InvalidHandle)?;
        let mut index = 0;
        for param in &entry.params {
            if *param == Param::GlobalPtr {
                let address = unsafe { self.args[index].as_ptr().cast::<u64>().read_unaligned() };
                if address != 0 && !contains_range(&context.allocations, address as usize, 1) {
                    return Err(InvalidValue);
                }
            }
            index += match param {
                Param::Words { words, .. } => usize::from(*words),
                _ => 1,
            };
        }
        Ok(())
    }

    /// Append this launch to the regular list `list` that later runs on `stream`, with the
    /// same substitution as a replay.
    pub(super) fn record(
        &self,
        api: &Api,
        context: &mut CudaContext,
        list: &crate::level_zero::CommandList,
        stream: usize,
    ) -> Result<(), CuResult> {
        let entry = self.entry(context)?;
        let (_, kernel) = entry.function.ok_or(InvalidHandle)?;
        let subst_plan = entry.subst.clone();
        let args: Vec<_> = self
            .args
            .iter()
            .map(|arg| KernelArg {
                value: arg.as_ptr().cast(),
                size: arg.len(),
            })
            .collect();
        super::subst_hook::launch_to(
            api,
            context,
            subst_plan.as_ref(),
            kernel,
            super::subst_hook::Target::List { list, stream },
            self.geometry,
            &args,
        )
    }

    fn replay(&self, api: &Api, context: &mut CudaContext, stream: usize) -> Result<(), CuResult> {
        let entry = self.entry(context)?;
        let (_, kernel) = entry.function.ok_or(InvalidHandle)?;
        let subst_plan = entry.subst.clone();
        let args: Vec<_> = self
            .args
            .iter()
            .map(|arg| KernelArg {
                value: arg.as_ptr().cast(),
                size: arg.len(),
            })
            .collect();
        super::subst_hook::launch(api, context, subst_plan.as_ref(), kernel, stream, self.geometry, &args)
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamBeginCapture_v2(stream: Opaque, mode: c_int) -> CuResult {
    trace::call("cuStreamBeginCapture_v2", || begin_capture(stream, mode, None))
}

/// Pre-CUDA 10.1 form: always global mode.
#[unsafe(no_mangle)]
pub extern "system" fn cuStreamBeginCapture(stream: Opaque) -> CuResult {
    trace::call("cuStreamBeginCapture", || begin_capture(stream, 0, None))
}

/// Capture into a graph the caller owns; the first captured nodes depend on `dependencies`.
pub(super) fn begin_capture(
    stream: Opaque,
    mode: c_int,
    target: Option<(usize, Vec<usize>)>,
) -> CuResult {
    status(state(|state| {
        let owner = current().ok_or(InvalidContext)?;
        let id = reserve_handle(&mut state.next_handle)?;
        let new_graph = reserve_handle(&mut state.next_handle)?;
        let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
        context.check_live()?;
        let key = context.streams.launch_key(stream)?;
        if !matches!(mode, 0..=2) {
            return Err(InvalidValue);
        }
        // The legacy default stream cannot be captured.
        if key == 0 {
            return Err(StreamCaptureUnsupported);
        }
        let slot = context.streams.capture_slot(key)?;
        if slot.is_some() {
            return Err(IllegalState);
        }
        let (graph, to_graph, dependencies) = match target {
            Some((graph, dependencies)) => {
                let dag = state.graphs.graphs.get_mut(&graph).ok_or(InvalidValue)?;
                if dag.context != owner {
                    return Err(InvalidContext);
                }
                if dag.capturing {
                    return Err(IllegalState);
                }
                if dependencies.iter().any(|&node| dag.find(node).is_none()) {
                    return Err(InvalidValue);
                }
                dag.capturing = true;
                (graph, true, dependencies)
            }
            None => {
                let mut dag = Dag::new(owner);
                dag.capturing = true;
                state.graphs.graphs.insert(new_graph, dag);
                (new_graph, false, Vec::new())
            }
        };
        *slot = Some(Capture::new(id as u64, key, mode, graph, to_graph, dependencies));
        Ok(())
    }))
}

thread_local! {
    static THREAD_MODE: Cell<c_int> = const { Cell::new(0) };
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuThreadExchangeStreamCaptureMode(mode: *mut c_int) -> CuResult {
    trace::call("cuThreadExchangeStreamCaptureMode", || {
        if mode.is_null() {
            return InvalidValue;
        }
        let requested = unsafe { mode.read() };
        if !matches!(requested, 0..=2) {
            return InvalidValue;
        }
        unsafe { mode.write(THREAD_MODE.replace(requested)) };
        Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamEndCapture(stream: Opaque, out: *mut Opaque) -> CuResult {
    trace::call("cuStreamEndCapture", || {
        if out.is_null() {
            return InvalidValue;
        }
        unsafe { out.write(std::ptr::null_mut()) };
        status(state(|state| {
            let owner = current().ok_or(InvalidContext)?;
            let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
            context.check_live()?;
            let key = context.streams.launch_key(stream)?;
            let capture = context
                .streams
                .capture(key)?
                .cloned()
                .ok_or(StreamCaptureUnmatched)?;
            if !capture.is_origin(key) {
                return Err(StreamCaptureUnmatched);
            }
            // The session ends for every member, whether or not it succeeded.
            for member in capture.members() {
                if let Ok(slot) = context.streams.capture_slot(member) {
                    *slot = None;
                }
            }
            context
                .streams
                .captured
                .retain(|_, (candidate, _, _)| !candidate.same(&capture));
            deactivate(&capture);
            let graph = capture.graph();
            let entries = match capture.finish() {
                Ok(entries) => entries,
                Err(error) => {
                    if capture.to_graph() {
                        if let Some(dag) = state.graphs.graphs.get_mut(&graph) {
                            dag.capturing = false;
                        }
                    } else {
                        state.graphs.graphs.remove(&graph);
                    }
                    return Err(error);
                }
            };
            let dag = state.graphs.graphs.get_mut(&graph).ok_or(InvalidHandle)?;
            dag.entries.extend(entries);
            dag.capturing = false;
            unsafe { out.write(std::ptr::without_provenance_mut(graph)) };
            Ok(())
        }))
    })
}

fn capture_status(stream: Opaque) -> Result<(c_int, u64), CuResult> {
    with_context(|_, context| {
        let key = context.streams.launch_key(stream)?;
        Ok(match context.streams.capture(key)? {
            None => (0, 0),
            Some(capture) => (if capture.is_invalidated() { 2 } else { 1 }, capture.id()),
        })
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamIsCapturing(stream: Opaque, out: *mut c_int) -> CuResult {
    trace::call("cuStreamIsCapturing", || {
        if out.is_null() {
            return InvalidValue;
        }
        status(capture_status(stream).map(|(mode, _)| unsafe { out.write(mode) }))
    })
}

/// CUDA 10.1 form: status and capture id only.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamGetCaptureInfo(
    stream: Opaque,
    status_out: *mut c_int,
    id_out: *mut u64,
) -> CuResult {
    trace::call("cuStreamGetCaptureInfo", || {
        status(capture_status(stream).map(|(mode, id)| unsafe {
            if !status_out.is_null() {
                status_out.write(mode);
            }
            if !id_out.is_null() {
                id_out.write(id);
            }
        }))
    })
}

/// Graph being captured into and the nodes the next operation depends on.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamGetCaptureInfo_v2(
    stream: Opaque,
    status_out: *mut c_int,
    id_out: *mut u64,
    graph_out: *mut Opaque,
    dependencies_out: *mut *const Opaque,
    count_out: *mut usize,
) -> CuResult {
    trace::call("cuStreamGetCaptureInfo_v2", || unsafe {
        let result = cuStreamGetCaptureInfo(stream, status_out, id_out);
        if result != Success {
            return result;
        }
        let info = with_context(|_, context| {
            let key = context.streams.launch_key(stream)?;
            Ok(context.streams.capture(key)?.map(|capture| capture.info(key)))
        });
        let (graph, dependencies, count) = match info {
            Ok(Some(info)) => info,
            Ok(None) => (0, std::ptr::null(), 0),
            Err(error) => return error,
        };
        if !graph_out.is_null() {
            graph_out.write(std::ptr::without_provenance_mut(graph));
        }
        if !dependencies_out.is_null() {
            dependencies_out.write(dependencies);
        }
        if !count_out.is_null() {
            count_out.write(count);
        }
        Success
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuStreamGetCaptureInfo_v3(
    stream: Opaque,
    status_out: *mut c_int,
    id_out: *mut u64,
    graph_out: *mut Opaque,
    dependencies_out: *mut *const Opaque,
    edges_out: *mut *const c_void,
    count_out: *mut usize,
) -> CuResult {
    trace::call("cuStreamGetCaptureInfo_v3", || unsafe {
        let result = cuStreamGetCaptureInfo_v2(
            stream,
            status_out,
            id_out,
            graph_out,
            dependencies_out,
            count_out,
        );
        // Every edge is a default edge, which CUDA reports as no edge data.
        if result == Success && !edges_out.is_null() {
            edges_out.write(std::ptr::null());
        }
        result
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphDestroy(graph: Opaque) -> CuResult {
    trace::call("cuGraphDestroy", || {
        status(state(|state| {
            if state
                .graphs
                .graphs
                .get(&graph.addr())
                .ok_or(InvalidHandle)?
                .capturing
            {
                return Err(IllegalState);
            }
            state.graphs.graphs.remove(&graph.addr());
            Ok(())
        }))
    })
}

fn instantiate(out: *mut Opaque, graph: Opaque) -> Result<(), CuResult> {
    if out.is_null() {
        return Err(InvalidValue);
    }
    state(|state| {
        let dag = state.graphs.graphs.get(&graph.addr()).ok_or(InvalidHandle)?;
        if dag.capturing {
            return Err(IllegalState);
        }
        let exec = dag.exec();
        let handle = reserve_handle(&mut state.next_handle)?;
        state.graphs.executable.insert(handle, exec);
        unsafe { out.write(std::ptr::without_provenance_mut(handle)) };
        Ok(())
    })
}

fn instantiate_flags(flags: u64) -> Result<(), CuResult> {
    // 1 auto-free-on-launch and 8 node priority have no effect without
    // allocation nodes or priorities; 2 (upload) is a hint.
    match flags {
        f if f & !0xf != 0 => Err(InvalidValue),
        f if f & 4 != 0 => Err(NotSupported), // device launch
        _ => Ok(()),
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphInstantiateWithFlags(
    out: *mut Opaque,
    graph: Opaque,
    flags: u64,
) -> CuResult {
    trace::call("cuGraphInstantiateWithFlags", || {
        status(instantiate_flags(flags).and_then(|()| instantiate(out, graph)))
    })
}

/// CUDA 10 and 11 form with an error-node and log-buffer reporting.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphInstantiate_v1(
    out: *mut Opaque,
    graph: Opaque,
    error_node: *mut Opaque,
    log: *mut c_char,
    log_size: usize,
) -> CuResult {
    trace::call("cuGraphInstantiate_v1", || {
        if !error_node.is_null() {
            unsafe { error_node.write(std::ptr::null_mut()) };
        }
        if !log.is_null() && log_size > 0 {
            unsafe { log.write(0) };
        }
        status(instantiate(out, graph))
    })
}

/// Layout of CUDA_GRAPH_INSTANTIATE_PARAMS.
#[repr(C)]
pub struct InstantiateParams {
    flags: u64,
    upload_stream: Opaque,
    error_node: Opaque,
    result: c_int,
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphInstantiateWithParams(
    out: *mut Opaque,
    graph: Opaque,
    params: *mut InstantiateParams,
) -> CuResult {
    trace::call("cuGraphInstantiateWithParams", || {
        if params.is_null() {
            return InvalidValue;
        }
        let flags = unsafe { (*params).flags };
        let outcome = instantiate_flags(flags).and_then(|()| instantiate(out, graph));
        unsafe {
            (*params).error_node = std::ptr::null_mut();
            // CUDA_GRAPH_INSTANTIATE_SUCCESS = 0, ERROR = 1, INVALID_STRUCTURE = 2.
            (*params).result = match outcome {
                Ok(()) => 0,
                Err(InvalidHandle) => 2,
                Err(_) => 1,
            };
        }
        status(outcome)
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphExecDestroy(graph: Opaque) -> CuResult {
    trace::call("cuGraphExecDestroy", || {
        status(state(|state| {
            let exec = state
                .graphs
                .executable
                .remove(&graph.addr())
                .ok_or(InvalidHandle)?;
            if let Some(context) = state.contexts.get_mut(&exec.context) {
                context.graph_lists.retire(graph.addr());
            }
            Ok(())
        }))
    })
}

/// Layout of CUgraphExecUpdateResultInfo.
#[repr(C)]
pub struct ExecUpdateInfo {
    result: c_int,
    error_node: Opaque,
    error_from_node: Opaque,
}

fn exec_update(exec: Opaque, graph: Opaque) -> (CuResult, c_int) {
    let mut result = 0;
    let outcome = state(|state| {
        let new = state
            .graphs
            .graphs
            .get(&graph.addr())
            .ok_or(InvalidHandle)?
            .exec();
        let old = state
            .graphs
            .executable
            .get_mut(&exec.addr())
            .ok_or(InvalidHandle)?;
        if old.context != new.context {
            result = 6; // CU_GRAPH_EXEC_UPDATE_ERROR_NOT_SUPPORTED
            return Err(GraphExecUpdateFailure);
        }
        if let Err(code) = nodes_updatable(&old.nodes, &new.nodes) {
            result = code;
            return Err(GraphExecUpdateFailure);
        }
        old.nodes = new.nodes;
        old.ids = new.ids;
        old.version = new.version;
        Ok(())
    });
    (status_of(outcome), result)
}

fn status_of(outcome: Result<(), CuResult>) -> CuResult {
    outcome.err().unwrap_or(Success)
}

/// CUDA 12 form.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphExecUpdate_v2(
    exec: Opaque,
    graph: Opaque,
    info: *mut ExecUpdateInfo,
) -> CuResult {
    trace::call("cuGraphExecUpdate_v2", || {
        let (result, code) = exec_update(exec, graph);
        if !info.is_null() {
            unsafe {
                (*info).result = code;
                (*info).error_node = std::ptr::null_mut();
                (*info).error_from_node = std::ptr::null_mut();
            }
        }
        result
    })
}

/// CUDA 10 and 11 form.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuGraphExecUpdate(
    exec: Opaque,
    graph: Opaque,
    error_node: *mut Opaque,
    result_out: *mut c_int,
) -> CuResult {
    trace::call("cuGraphExecUpdate", || {
        let (result, code) = exec_update(exec, graph);
        unsafe {
            if !error_node.is_null() {
                error_node.write(std::ptr::null_mut());
            }
            if !result_out.is_null() {
                result_out.write(code);
            }
        }
        result
    })
}

/// Upload is a scheduling hint; replay needs no preparation.
#[unsafe(no_mangle)]
pub extern "system" fn cuGraphUpload(graph: Opaque, stream: Opaque) -> CuResult {
    trace::call("cuGraphUpload", || {
        status(state(|state| {
            state
                .graphs
                .executable
                .get(&graph.addr())
                .ok_or(InvalidHandle)?;
            let context = state
                .contexts
                .get_mut(&current().ok_or(InvalidContext)?)
                .ok_or(ContextIsDestroyed)?;
            context.streams.launch_key(stream).map(|_| ())
        }))
    })
}

pub(super) fn flatten<'a>(nodes: &'a [Node], steps: &mut Vec<&'a Node>) {
    for node in nodes {
        match node {
            Node::Child(children) => flatten(children, steps),
            node => steps.push(node),
        }
    }
}

/// Call a host function after the work issued before it on `stream` finished.
/// Runs with the runtime lock released, so the function may use the API.
pub(super) fn call_host(stream: Opaque, host: HostFn) -> CuResult {
    let result = super::streams::cuStreamSynchronize(stream);
    if result != Success {
        return result;
    }
    let function: unsafe extern "system" fn(*mut c_void) =
        unsafe { std::mem::transmute(host.function) };
    unsafe { function(std::ptr::with_exposed_provenance_mut(host.data)) };
    Success
}

/// Replay a graph in dependency order on one stream. Independent branches are
/// issued one after another (ready-order), which is always a legal schedule.
#[unsafe(no_mangle)]
pub extern "system" fn cuGraphLaunch(graph: Opaque, stream: Opaque) -> CuResult {
    trace::call("cuGraphLaunch", || {
        let exec_handle = graph.addr();
        let prepared = state(|state| {
            let graph = state
                .graphs
                .executable
                .get(&graph.addr())
                .ok_or(InvalidHandle)?;
            if current() != Some(graph.context) {
                return Err(InvalidContext);
            }
            let context = state
                .contexts
                .get_mut(&graph.context)
                .ok_or(ContextIsDestroyed)?;
            context.check_live()?;
            let key = context.streams.launch_key(stream)?;
            // A recorded graph whose nodes and the allocations they name are unchanged since
            // its last launch needs no second walk over every node.
            if !(super::graph_record::enabled()
                && context.graph_lists.validated(exec_handle, graph.version, context.alloc_frees))
            {
                for node in &*graph.nodes {
                    node.validate(context)?;
                }
            }
            if let Some(capture) = context.streams.capture(key)? {
                // Launching into a capturing stream adds the graph as a child node.
                capture.push(key, Node::Child(Arc::clone(&graph.nodes)))?;
                return Ok(None);
            }
            if super::graph_record::enabled()
                && super::graph_record::launch(&state.runtime, context, exec_handle, graph.version, &graph.nodes, key)?
            {
                return Ok(None);
            }
            Ok(Some((graph.context, key, Arc::clone(&graph.nodes))))
        });
        let (owner, key, nodes) = match prepared {
            Ok(Some(prepared)) => prepared,
            Ok(None) => return Success,
            Err(error) => return error,
        };
        let mut steps = Vec::new();
        flatten(&nodes, &mut steps);
        let mut next = 0;
        while next < steps.len() {
            let host = state(|state| {
                let context = state.contexts.get_mut(&owner).ok_or(ContextIsDestroyed)?;
                context.check_live()?;
                while next < steps.len() {
                    let node = steps[next];
                    next += 1;
                    if let Node::Host(host) = node {
                        return Ok(Some(*host));
                    }
                    node.replay(&state.runtime, context, key)?;
                }
                Ok(None)
            });
            match host {
                Ok(Some(host)) => {
                    let result = call_host(stream, host);
                    if result != Success {
                        return result;
                    }
                }
                Ok(None) => {}
                Err(error) => return error,
            }
        }
        Success
    })
}
