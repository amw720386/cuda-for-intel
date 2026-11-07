//! Stream capture and replay of supported CUDA operations.
//!
//! Capture records kernel launches, memcpy and memset operations into a node
//! list instead of submitting them. Events recorded on a capturing stream and
//! waited on by another stream fork that stream into the same capture, so
//! multi-stream fork/join patterns are accepted. The recorded order is the
//! program order of the capturing calls, which is a valid topological order of
//! the CUDA dependency graph. Launching a graph re-issues the nodes in that
//! order on the launch stream: correct first, with no concurrency between
//! branches. Graphs have no public node handles; explicit graph building,
//! node getters and conditional/memory-allocation nodes are not implemented.

use super::*;
use crate::level_zero::KernelArg;
use std::cell::Cell;
use std::mem::MaybeUninit;
use std::sync::{Arc, Mutex, MutexGuard};

#[derive(Default)]
pub(super) struct Graphs {
    graphs: HashMap<usize, Graph>,
    executable: HashMap<usize, Graph>,
}

impl Graphs {
    pub(super) fn drop_context(&mut self, context: usize) {
        self.graphs.retain(|_, graph| graph.context != context);
        self.executable.retain(|_, graph| graph.context != context);
    }
}

#[derive(Clone)]
struct Graph {
    context: usize,
    nodes: Arc<[Node]>,
}

pub(super) enum Node {
    Kernel(Kernel),
    Copy(Copy),
    Fill(Fill),
    /// A previously captured graph launched into a capturing stream.
    Child(Arc<[Node]>),
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(super) enum CopyKind {
    HostToDevice,
    DeviceToHost,
    DeviceToDevice,
}

pub(super) struct Copy {
    pub(super) dst: usize,
    pub(super) src: usize,
    pub(super) size: usize,
    pub(super) kind: CopyKind,
}

pub(super) struct Fill {
    pub(super) dst: usize,
    pub(super) value: u8,
    pub(super) size: usize,
}

impl Node {
    fn validate(&self, context: &CudaContext) -> Result<(), CuResult> {
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
            Node::Child(nodes) => nodes.iter().try_for_each(|node| node.validate(context)),
        }
    }

    fn replay(&self, runtime: &Runtime, context: &mut CudaContext, stream: usize) -> Result<(), CuResult> {
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
            (Node::Fill(_), Node::Fill(_)) => Ok(()),
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
    invalidated: bool,
    nodes: Vec<Node>,
    members: HashMap<usize, Member>,
}

/// A capture session shared by the origin stream and every stream forked into it.
#[derive(Clone)]
pub(super) struct Capture(Arc<Mutex<Inner>>);

impl Capture {
    pub(super) fn new(id: u64, origin: usize, mode: c_int) -> Self {
        let mut members = HashMap::new();
        members.insert(origin, Member::default());
        Self(Arc::new(Mutex::new(Inner {
            id,
            origin,
            mode,
            invalidated: false,
            nodes: Vec::new(),
            members,
        })))
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

    pub(super) fn is_origin(&self, stream: usize) -> bool {
        self.inner().origin == stream
    }

    pub(super) fn is_invalidated(&self) -> bool {
        self.inner().invalidated
    }

    pub(super) fn members(&self) -> Vec<usize> {
        self.inner().members.keys().copied().collect()
    }

    /// Append a node produced by `stream`.
    pub(super) fn push(&self, stream: usize, node: Node) -> Result<(), CuResult> {
        let mut inner = self.inner();
        if inner.invalidated {
            return Err(StreamCaptureInvalidated);
        }
        inner.members.entry(stream).or_default().sequence += 1;
        inner.nodes.push(node);
        Ok(())
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

    /// Finish the session, returning its nodes.
    fn finish(&self) -> Result<Vec<Node>, CuResult> {
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
        Ok(std::mem::take(&mut inner.nodes))
    }
}

pub(super) struct Kernel {
    function: usize,
    geometry: ([u32; 3], [u32; 3]),
    args: Vec<Box<[MaybeUninit<u8>]>>,
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
            args,
        }
    }

    fn entry<'a>(&self, context: &'a CudaContext) -> Result<&'a CudaEntry, CuResult> {
        let (module, name) = context.functions.get(&self.function).ok_or(InvalidHandle)?;
        let module = context.modules.get(module).ok_or(InvalidHandle)?;
        module.entries.get(name).ok_or(InvalidHandle)
    }

    fn validate(&self, context: &CudaContext) -> Result<(), CuResult> {
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

    fn replay(&self, api: &Api, context: &mut CudaContext, stream: usize) -> Result<(), CuResult> {
        let (_, kernel) = self.entry(context)?.function.ok_or(InvalidHandle)?;
        let args: Vec<_> = self
            .args
            .iter()
            .map(|arg| KernelArg {
                value: arg.as_ptr().cast(),
                size: arg.len(),
            })
            .collect();
        context
            .streams
            .launch(api, &context.backend, kernel, stream, self.geometry, &args)
    }
}

#[unsafe(no_mangle)]
pub extern "system" fn cuStreamBeginCapture_v2(stream: Opaque, mode: c_int) -> CuResult {
    trace::call("cuStreamBeginCapture_v2", || begin_capture(stream, mode))
}

/// Pre-CUDA 10.1 form: always global mode.
#[unsafe(no_mangle)]
pub extern "system" fn cuStreamBeginCapture(stream: Opaque) -> CuResult {
    trace::call("cuStreamBeginCapture", || begin_capture(stream, 0))
}

fn begin_capture(stream: Opaque, mode: c_int) -> CuResult {
    status(state(|state| {
        let owner = current().ok_or(InvalidContext)?;
        let id = reserve_handle(&mut state.next_handle)?;
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
        // All modes capture identically; global/thread-local do not yet reject
        // unsafe calls (such as allocation) made during capture.
        *slot = Some(Capture::new(id as u64, key, mode));
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
            let nodes = capture.finish()?;
            let graph = reserve_handle(&mut state.next_handle)?;
            state.graphs.graphs.insert(
                graph,
                Graph {
                    context: owner,
                    nodes: nodes.into(),
                },
            );
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

/// There is no graph object while capture is in progress, so the graph and the
/// dependency set are reported as empty.
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
        if !graph_out.is_null() {
            graph_out.write(std::ptr::null_mut());
        }
        if !dependencies_out.is_null() {
            dependencies_out.write(std::ptr::null());
        }
        if !count_out.is_null() {
            count_out.write(0);
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
            state
                .graphs
                .graphs
                .remove(&graph.addr())
                .map(|_| ())
                .ok_or(InvalidHandle)
        }))
    })
}

fn instantiate(out: *mut Opaque, graph: Opaque) -> Result<(), CuResult> {
    if out.is_null() {
        return Err(InvalidValue);
    }
    state(|state| {
        let graph = state
            .graphs
            .graphs
            .get(&graph.addr())
            .ok_or(InvalidHandle)?
            .clone();
        let handle = reserve_handle(&mut state.next_handle)?;
        state.graphs.executable.insert(handle, graph);
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
            state
                .graphs
                .executable
                .remove(&graph.addr())
                .map(|_| ())
                .ok_or(InvalidHandle)
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
            .clone();
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

#[unsafe(no_mangle)]
pub extern "system" fn cuGraphLaunch(graph: Opaque, stream: Opaque) -> CuResult {
    trace::call("cuGraphLaunch", || {
        status(state(|state| {
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
            for node in &*graph.nodes {
                node.validate(context)?;
            }
            if let Some(capture) = context.streams.capture(key)? {
                // Launching into a capturing stream adds the graph as a child node.
                return capture.push(key, Node::Child(Arc::clone(&graph.nodes)));
            }
            for node in &*graph.nodes {
                node.replay(&state.runtime, context, key)?;
            }
            Ok(())
        }))
    })
}
