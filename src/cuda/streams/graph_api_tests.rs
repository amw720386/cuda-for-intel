//! Explicit graph building, capture dependencies and host functions against
//! the fake Level Zero backend.
use super::events::{cuEventDestroy_v2, cuEventRecord, cuStreamWaitEvent};
use super::fake_tests::{Fixture, allocations, assert_baseline, context, event, function, launch, module, stream};
use super::*;
use crate::cuda::graph_api::*;
use crate::cuda::graphs::*;
use crate::cuda::*;
use crate::level_zero::fake;
use std::ptr::null_mut;
use std::sync::atomic::{AtomicUsize, Ordering};

const GLOBAL: c_int = 0;
const THREAD_LOCAL: c_int = 1;
const RELAXED: c_int = 2;

fn work() -> usize {
    let s = fake::snapshot();
    s.recorded + s.ledger.submitted
}

fn graph() -> Opaque {
    let mut graph = null_mut();
    assert_eq!(unsafe { cuGraphCreate(&mut graph, 0) }, Success);
    graph
}

fn begin(stream: Opaque, mode: c_int) {
    assert_eq!(cuStreamBeginCapture_v2(stream, mode), Success);
}

fn end(stream: Opaque) -> Opaque {
    let mut graph = null_mut();
    assert_eq!(unsafe { cuStreamEndCapture(stream, &mut graph) }, Success);
    graph
}

fn instantiate(graph: Opaque) -> Opaque {
    let mut exec = null_mut();
    assert_eq!(unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) }, Success);
    exec
}

fn nodes(graph: Opaque) -> Vec<Opaque> {
    let mut count = 0;
    assert_eq!(unsafe { cuGraphGetNodes(graph, null_mut(), &mut count) }, Success);
    let mut list = vec![null_mut(); count + 2];
    let mut capacity = count + 2;
    assert_eq!(unsafe { cuGraphGetNodes(graph, list.as_mut_ptr(), &mut capacity) }, Success);
    assert_eq!(capacity, count);
    list.truncate(count);
    list
}

fn edges(graph: Opaque) -> Vec<(Opaque, Opaque)> {
    let mut count = 0;
    assert_eq!(unsafe { cuGraphGetEdges(graph, null_mut(), null_mut(), &mut count) }, Success);
    let (mut from, mut to) = (vec![null_mut(); count], vec![null_mut(); count]);
    assert_eq!(
        unsafe { cuGraphGetEdges(graph, from.as_mut_ptr(), to.as_mut_ptr(), &mut count) },
        Success
    );
    from.into_iter().zip(to).collect()
}

fn kind(node: Opaque) -> c_int {
    let mut kind = -1;
    assert_eq!(unsafe { cuGraphNodeGetType(node, &mut kind) }, Success);
    kind
}

fn empty(graph: Opaque, deps: &[Opaque]) -> Opaque {
    let mut node = null_mut();
    assert_eq!(
        unsafe { cuGraphAddEmptyNode(&mut node, graph, deps.as_ptr(), deps.len()) },
        Success
    );
    node
}

fn kernel_params(f: Opaque, address: &u64) -> (KernelNodeParams, Box<[Opaque; 1]>) {
    let arg: Box<[Opaque; 1]> = Box::new([std::ptr::from_ref(address).cast_mut().cast()]);
    (
        KernelNodeParams {
            function: f,
            grid: [1, 1, 1],
            block: [1, 1, 1],
            shared: 0,
            kernel_params: arg.as_ptr().cast_mut(),
            extra: null_mut(),
            kernel: null_mut(),
            context: null_mut(),
        },
        arg,
    )
}

fn kernel(graph: Opaque, deps: &[Opaque], f: Opaque, address: &u64) -> Opaque {
    let (params, _arg) = kernel_params(f, address);
    let mut node = null_mut();
    assert_eq!(
        unsafe { cuGraphAddKernelNode_v2(&mut node, graph, deps.as_ptr(), deps.len(), &params) },
        Success
    );
    node
}

fn memset(graph: Opaque, deps: &[Opaque], address: u64) -> Opaque {
    let params = MemsetNodeParams {
        destination: address,
        pitch: 0,
        value: 0x07,
        element_size: 1,
        width: 16,
        height: 1,
    };
    let mut node = null_mut();
    assert_eq!(
        unsafe { cuGraphAddMemsetNode(&mut node, graph, deps.as_ptr(), deps.len(), &params, null_mut()) },
        Success
    );
    node
}

fn memcpy_htod(device: u64, host: Opaque, size: usize) -> MemcpyNodeParams {
    MemcpyNodeParams {
        src_x: 0,
        src_y: 0,
        src_z: 0,
        src_lod: 0,
        src_type: 1,
        src_host: host,
        src_device: 0,
        src_array: null_mut(),
        reserved0: null_mut(),
        src_pitch: size,
        src_height: 1,
        dst_x: 0,
        dst_y: 0,
        dst_z: 0,
        dst_lod: 0,
        dst_type: 2,
        dst_host: null_mut(),
        dst_device: device,
        dst_array: null_mut(),
        reserved1: null_mut(),
        dst_pitch: size,
        dst_height: 1,
        width: size,
        height: 1,
        depth: 1,
    }
}

fn exec_ids(exec: Opaque) -> Vec<usize> {
    state(|state| Ok(state.graphs.executable[&exec.addr()].ids.to_vec())).unwrap()
}

#[test]
fn fake_explicit_graph_structure_and_queries() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let f = function(module());
    let (device, host) = allocations();
    let g = graph();
    assert!(nodes(g).is_empty());

    let a = empty(g, &[]);
    let b = kernel(g, &[a], f, &device);
    let c = memset(g, &[a], device + 32);
    let copy = memcpy_htod(device + 16, host, 8);
    let mut d = null_mut();
    assert_eq!(
        unsafe { cuGraphAddMemcpyNode(&mut d, g, [b, c].as_ptr(), 2, &copy, null_mut()) },
        Success
    );
    assert_eq!(nodes(g), [a, b, c, d]);
    assert_eq!([a, b, c, d].map(kind), [5, 0, 2, 1]);
    assert_eq!(edges(g), [(a, b), (a, c), (b, d), (c, d)]);

    let mut count = 0;
    let mut roots = [null_mut(); 3];
    count = 3.max(count);
    assert_eq!(unsafe { cuGraphGetRootNodes(g, roots.as_mut_ptr(), &mut count) }, Success);
    assert_eq!((count, roots[0], roots[1]), (1, a, null_mut()));

    let mut deps = [null_mut(); 4];
    count = 4;
    assert_eq!(unsafe { cuGraphNodeGetDependencies(d, deps.as_mut_ptr(), &mut count) }, Success);
    assert_eq!((count, deps[0], deps[1]), (2, b, c));
    count = 4;
    assert_eq!(unsafe { cuGraphNodeGetDependentNodes(a, deps.as_mut_ptr(), &mut count) }, Success);
    assert_eq!((count, deps[0], deps[1]), (2, b, c));

    // Edges: no cycles, duplicates or foreign nodes.
    assert_eq!(unsafe { cuGraphAddDependencies(g, [d].as_ptr(), [a].as_ptr(), 1) }, InvalidValue);
    assert_eq!(unsafe { cuGraphAddDependencies(g, [a].as_ptr(), [b].as_ptr(), 1) }, InvalidValue);
    assert_eq!(unsafe { cuGraphAddDependencies(g, [a].as_ptr(), [a].as_ptr(), 1) }, InvalidValue);
    assert_eq!(unsafe { cuGraphAddDependencies(g, [a].as_ptr(), [d].as_ptr(), 1) }, Success);
    assert_eq!(unsafe { cuGraphRemoveDependencies(g, [a].as_ptr(), [d].as_ptr(), 1) }, Success);
    assert_eq!(unsafe { cuGraphRemoveDependencies(g, [a].as_ptr(), [d].as_ptr(), 1) }, InvalidValue);
    // A failing batch changes nothing.
    assert_eq!(
        unsafe { cuGraphAddDependencies(g, [a, d].as_ptr(), [d, a].as_ptr(), 2) },
        InvalidValue
    );
    assert_eq!(edges(g).len(), 4);

    // Bad handles.
    let mut node = null_mut();
    assert_eq!(unsafe { cuGraphAddEmptyNode(&mut node, g, [std::ptr::dangling_mut()].as_ptr(), 1) }, InvalidValue);
    assert_eq!(unsafe { cuGraphAddEmptyNode(&mut node, g, [a, a].as_ptr(), 2) }, InvalidValue);
    assert_eq!(unsafe { cuGraphNodeGetType(std::ptr::dangling_mut(), &mut 0) }, InvalidValue);
    assert_eq!(unsafe { cuGraphCreate(&mut node, 1) }, InvalidValue);

    // Removing a node drops its edges.
    assert_eq!(cuGraphDestroyNode(c), Success);
    assert_eq!(nodes(g), [a, b, d]);
    assert_eq!(edges(g), [(a, b), (b, d)]);

    assert_eq!(cuGraphDestroy(g), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}

#[test]
fn fake_explicit_graph_launch_follows_dependencies() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    let f = function(module());
    let (device, _) = allocations();
    let g = graph();
    let a = empty(g, &[]);
    let b = kernel(g, &[], f, &device);
    let c = kernel(g, &[b], f, &device);
    // Make `a` depend on `c`: replay must issue c's chain before a.
    assert_eq!(unsafe { cuGraphAddDependencies(g, [c].as_ptr(), [a].as_ptr(), 1) }, Success);
    let exec = instantiate(g);
    assert_eq!(exec_ids(exec), [b, c, a].map(Opaque::addr));

    let before = work();
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert!(work() - before >= 2);
    assert_eq!(cuStreamSynchronize(s), Success);

    // Independent branches are issued in creation order.
    let g2 = graph();
    let x1 = kernel(g2, &[], f, &device);
    let y1 = kernel(g2, &[], f, &device);
    let x2 = kernel(g2, &[x1], f, &device);
    let y2 = kernel(g2, &[y1], f, &device);
    let join = empty(g2, &[x2, y2]);
    let exec2 = instantiate(g2);
    assert_eq!(exec_ids(exec2), [x1, y1, x2, y2, join].map(Opaque::addr));

    for exec in [exec, exec2] {
        assert_eq!(cuGraphExecDestroy(exec), Success);
    }
    assert_eq!(cuGraphDestroy(g), Success);
    assert_eq!(cuGraphDestroy(g2), Success);
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn fake_graph_clone_and_node_params() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let f = function(module());
    let (device, host) = allocations();
    let g = graph();
    let k = kernel(g, &[], f, &device);
    let m = memset(g, &[k], device + 32);
    let copy = memcpy_htod(device + 16, host, 8);
    let mut c = null_mut();
    assert_eq!(unsafe { cuGraphAddMemcpyNode(&mut c, g, [m].as_ptr(), 1, &copy, null_mut()) }, Success);

    let mut clone = null_mut();
    assert_eq!(unsafe { cuGraphClone(&mut clone, g) }, Success);
    assert_ne!(clone, g);
    let copies = nodes(clone);
    assert_eq!(copies.len(), 3);
    assert!(copies.iter().all(|node| ![k, m, c].contains(node)));
    let mut found = null_mut();
    assert_eq!(unsafe { cuGraphNodeFindInClone(&mut found, m, clone) }, Success);
    assert_eq!(found, copies[1]);
    assert_eq!(unsafe { cuGraphNodeFindInClone(&mut found, m, g) }, InvalidValue);
    assert_eq!(edges(clone), [(copies[0], copies[1]), (copies[1], copies[2])]);
    assert_eq!(cuGraphDestroyNode(copies[0]), Success);
    assert_eq!(nodes(g).len(), 3, "the original is unaffected");

    // Kernel parameters round trip and can be replaced.
    let mut got = kernel_params(f, &0).0;
    assert_eq!(unsafe { cuGraphKernelNodeGetParams_v2(k, &mut got) }, Success);
    assert_eq!((got.function, got.grid, got.block), (f, [1, 1, 1], [1, 1, 1]));
    let value = unsafe { (*got.kernel_params).cast::<u64>().read_unaligned() };
    assert_eq!(value, device);
    let other = device + 8;
    let (mut set, _arg) = kernel_params(f, &other);
    set.grid = [2, 1, 1];
    assert_eq!(unsafe { cuGraphKernelNodeSetParams_v2(k, &set) }, Success);
    assert_eq!(unsafe { cuGraphKernelNodeGetParams_v2(k, &mut got) }, Success);
    assert_eq!(got.grid, [2, 1, 1]);
    assert_eq!(unsafe { (*got.kernel_params).cast::<u64>().read_unaligned() }, other);
    // Wrong node type, bad address.
    assert_eq!(unsafe { cuGraphKernelNodeGetParams_v2(m, &mut got) }, InvalidValue);
    let bad = 0xdead_0000u64;
    let (bad_params, _arg) = kernel_params(f, &bad);
    assert_eq!(unsafe { cuGraphKernelNodeSetParams_v2(k, &bad_params) }, InvalidValue);

    // Memset and memcpy parameters.
    let mut memset_params = MemsetNodeParams { destination: 0, pitch: 0, value: 0, element_size: 0, width: 0, height: 0 };
    assert_eq!(unsafe { cuGraphMemsetNodeGetParams(m, &mut memset_params) }, Success);
    assert_eq!((memset_params.destination, memset_params.width, memset_params.value), (device + 32, 16, 7));
    memset_params.width = 8;
    assert_eq!(unsafe { cuGraphMemsetNodeSetParams(m, &memset_params) }, Success);
    memset_params.value = 0x0102;
    memset_params.element_size = 2;
    assert_eq!(unsafe { cuGraphMemsetNodeSetParams(m, &memset_params) }, NotSupported);
    let mut copy_params = memcpy_htod(0, null_mut(), 0);
    assert_eq!(unsafe { cuGraphMemcpyNodeGetParams(c, &mut copy_params) }, Success);
    assert_eq!((copy_params.src_type, copy_params.dst_type, copy_params.width), (1, 2, 8));
    assert_eq!(copy_params.dst_device, device + 16);
    copy_params.width = 4;
    assert_eq!(unsafe { cuGraphMemcpyNodeSetParams(c, &copy_params) }, Success);
    copy_params.height = 2;
    assert_eq!(unsafe { cuGraphMemcpyNodeSetParams(c, &copy_params) }, NotSupported);

    for graph in [g, clone] {
        assert_eq!(cuGraphDestroy(graph), Success);
    }
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}

#[test]
fn fake_exec_kernel_node_set_params() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let f = function(module());
    let (device, _) = allocations();
    let g = graph();
    let k = kernel(g, &[], f, &device);
    let exec = instantiate(g);
    let other = device + 8;
    let (params, _arg) = kernel_params(f, &other);
    assert_eq!(unsafe { cuGraphExecKernelNodeSetParams_v2(exec, k, &params) }, Success);
    let word = state(|state| {
        let Node::Kernel(kernel) = state.graphs.executable[&exec.addr()].node(k.addr())? else {
            return Err(InvalidValue);
        };
        Ok(unsafe { kernel.args[0].as_ptr().cast::<u64>().read_unaligned() })
    })
    .unwrap();
    assert_eq!(word, other);
    // The graph itself is unchanged.
    let mut got = kernel_params(f, &0).0;
    assert_eq!(unsafe { cuGraphKernelNodeGetParams_v2(k, &mut got) }, Success);
    assert_eq!(unsafe { (*got.kernel_params).cast::<u64>().read_unaligned() }, device);
    let unknown = empty(g, &[]);
    assert_eq!(unsafe { cuGraphExecKernelNodeSetParams_v2(exec, unknown, &params) }, InvalidValue);
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuGraphDestroy(g), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}

#[test]
fn fake_capture_builds_dependency_edges() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let (s1, s2) = (stream(), stream());
    let f = function(module());
    let (device, _) = allocations();
    let (fork, joined) = (event(), event());

    begin(s1, RELAXED);
    let (mut status, mut id, mut g, mut deps, mut count) = (0, 0, null_mut(), std::ptr::null(), 9);
    assert_eq!(
        unsafe { cuStreamGetCaptureInfo_v2(s1, &mut status, &mut id, &mut g, &mut deps, &mut count) },
        Success
    );
    assert!(!g.is_null());
    assert_eq!(count, 0);
    assert_eq!(launch(f, s1, device), Success);
    assert_eq!(
        unsafe { cuStreamGetCaptureInfo_v2(s1, &mut status, &mut id, &mut g, &mut deps, &mut count) },
        Success
    );
    assert_eq!(count, 1);
    let first = unsafe { deps.read() };
    assert_eq!(cuEventRecord(fork, s1), Success);
    assert_eq!(cuStreamWaitEvent(s2, fork, 0), Success);
    assert_eq!(launch(f, s2, device), Success);
    assert_eq!(cuEventRecord(joined, s2), Success);
    assert_eq!(launch(f, s1, device), Success);
    assert_eq!(cuStreamWaitEvent(s1, joined, 0), Success);
    assert_eq!(launch(f, s1, device), Success);
    let captured = end(s1);
    assert_eq!(captured, g, "the graph reported during capture is the result");

    let list = nodes(captured);
    assert_eq!(list.len(), 4);
    assert_eq!(list[0], first);
    let (k1, k2, k3, k4) = (list[0], list[1], list[2], list[3]);
    let found = edges(captured);
    for edge in [(k1, k2), (k1, k3), (k2, k4), (k3, k4)] {
        assert!(found.contains(&edge), "missing edge {edge:?} in {found:?}");
    }
    assert_eq!(found.len(), 4);
    assert_eq!(unsafe { cuGraphDestroy(captured) }, Success);

    // A stream that is not capturing has no graph or dependencies.
    assert_eq!(
        unsafe { cuStreamGetCaptureInfo_v2(s1, &mut status, &mut id, &mut g, &mut deps, &mut count) },
        Success
    );
    assert!(g.is_null() && deps.is_null() && count == 0);
    for e in [fork, joined] {
        assert_eq!(cuEventDestroy_v2(e), Success);
    }
    for s in [s1, s2] {
        assert_eq!(cuStreamDestroy_v2(s), Success);
    }
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn fake_update_capture_dependencies_and_capture_to_graph() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    let f = function(module());
    let (device, _) = allocations();

    begin(s, GLOBAL);
    assert_eq!(launch(f, s, device), Success);
    let (mut status, mut id, mut g, mut deps, mut count) = (0, 0, null_mut(), std::ptr::null(), 0);
    assert_eq!(
        unsafe { cuStreamGetCaptureInfo_v2(s, &mut status, &mut id, &mut g, &mut deps, &mut count) },
        Success
    );
    // Add a node to the capturing graph by hand and make the stream depend on it only.
    let manual = empty(g, &[]);
    assert_eq!(unsafe { cuStreamUpdateCaptureDependencies(s, [manual].as_ptr(), 1, 1) }, Success);
    assert_eq!(launch(f, s, device), Success);
    assert_eq!(
        unsafe { cuStreamUpdateCaptureDependencies(s, [std::ptr::dangling_mut()].as_ptr(), 1, 0) },
        InvalidValue
    );
    assert_eq!(unsafe { cuStreamUpdateCaptureDependencies_v2(s, null_mut(), std::ptr::null(), 0, 7) }, InvalidValue);
    assert_eq!(cuGraphDestroy(g), IllegalState, "a capturing graph cannot be destroyed");
    let mut exec = null_mut();
    assert_eq!(unsafe { cuGraphInstantiateWithFlags(&mut exec, g, 0) }, IllegalState);
    assert_eq!(end(s), g);
    let list = nodes(g);
    assert_eq!(list.len(), 3);
    let found = edges(g);
    assert_eq!(found, [(manual, list[2])], "the second kernel depends on the manual node only: {found:?}");
    assert_eq!(unsafe { cuStreamUpdateCaptureDependencies(s, std::ptr::null(), 0, 0) }, IllegalState);
    assert_eq!(cuGraphDestroy(g), Success);

    // Capture into an existing graph after a dependency.
    let target = graph();
    let anchor = empty(target, &[]);
    let missing = std::ptr::dangling_mut();
    assert_eq!(
        unsafe { cuStreamBeginCaptureToGraph(s, target, [missing].as_ptr(), std::ptr::null(), 1, RELAXED) },
        InvalidValue
    );
    assert_eq!(capture_state(s), 0);
    assert_eq!(
        unsafe { cuStreamBeginCaptureToGraph(s, target, [anchor].as_ptr(), std::ptr::null(), 1, RELAXED) },
        Success
    );
    assert_eq!(launch(f, s, device), Success);
    assert_eq!(end(s), target);
    let list = nodes(target);
    assert_eq!(list.len(), 2);
    assert_eq!(edges(target), [(anchor, list[1])]);
    assert_eq!(cuGraphDestroy(target), Success);
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}

fn capture_state(stream: Opaque) -> c_int {
    let mut status = -1;
    assert_eq!(unsafe { cuStreamIsCapturing(stream, &mut status) }, Success);
    status
}

#[test]
fn fake_unsafe_calls_are_rejected_by_capture_mode() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    let mut address = 0u64;

    // Relaxed capture allows allocation.
    begin(s, RELAXED);
    assert_eq!(unsafe { cuMemAlloc_v2(&mut address, 64) }, Success);
    assert_eq!(cuMemFree_v2(address), Success);
    assert_eq!(capture_state(s), 1);
    assert_eq!(cuGraphDestroy(end(s)), Success);

    for mode in [GLOBAL, THREAD_LOCAL] {
        begin(s, mode);
        assert_eq!(unsafe { cuMemAlloc_v2(&mut address, 64) }, StreamCaptureUnsupported);
        assert_eq!(cuCtxSynchronize(), StreamCaptureUnsupported);
        assert_eq!(capture_state(s), 2, "the capture is invalidated");
        let mut g = null_mut();
        assert_eq!(unsafe { cuStreamEndCapture(s, &mut g) }, StreamCaptureInvalidated);
        // Capture over: calls work again.
        assert_eq!(unsafe { cuMemAlloc_v2(&mut address, 64) }, Success);
        assert_eq!(cuMemFree_v2(address), Success);
    }

    // The calling thread can opt out of a global capture.
    begin(s, GLOBAL);
    let mut thread_mode = RELAXED;
    assert_eq!(unsafe { cuThreadExchangeStreamCaptureMode(&mut thread_mode) }, Success);
    assert_eq!(thread_mode, GLOBAL);
    assert_eq!(unsafe { cuMemAlloc_v2(&mut address, 64) }, Success);
    assert_eq!(cuMemFree_v2(address), Success);
    assert_eq!(unsafe { cuThreadExchangeStreamCaptureMode(&mut thread_mode) }, Success);
    assert_eq!(thread_mode, RELAXED);
    assert_eq!(cuGraphDestroy(end(s)), Success);

    // A global capture on this thread blocks other threads; thread-local does not.
    for (mode, blocked) in [(GLOBAL, true), (THREAD_LOCAL, false)] {
        begin(s, mode);
        let other = std::thread::spawn(|| {
            let mut address = 0u64;
            unsafe { cuMemAlloc_v2(&mut address, 64) }
        })
        .join()
        .unwrap();
        assert_eq!(other == StreamCaptureUnsupported, blocked, "mode {mode}: {other:?}");
        let _ = cuGraphDestroy(end_any(s));
    }
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}

/// End a capture that may have been invalidated; returns a graph handle or null.
fn end_any(stream: Opaque) -> Opaque {
    let mut g = null_mut();
    let _ = unsafe { cuStreamEndCapture(stream, &mut g) };
    g
}

static HOST_CALLS: AtomicUsize = AtomicUsize::new(0);

extern "system" fn bump(data: *mut c_void) {
    HOST_CALLS.fetch_add(data.addr(), Ordering::SeqCst);
}

#[test]
fn fake_host_functions_run_after_stream_work() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    let f = function(module());
    let (device, _) = allocations();
    HOST_CALLS.store(0, Ordering::SeqCst);
    let function: Opaque = (bump as extern "system" fn(*mut c_void)) as *const () as Opaque;

    assert_eq!(unsafe { cuLaunchHostFunc(s, null_mut(), null_mut()) }, InvalidValue);
    assert_eq!(launch(f, s, device), Success);
    assert_eq!(unsafe { cuLaunchHostFunc(s, function, 1 as Opaque) }, Success);
    assert_eq!(HOST_CALLS.load(Ordering::SeqCst), 1);

    // In a capture the function becomes a node and runs on every launch.
    begin(s, RELAXED);
    assert_eq!(launch(f, s, device), Success);
    assert_eq!(unsafe { cuLaunchHostFunc(s, function, 10 as Opaque) }, Success);
    assert_eq!(launch(f, s, device), Success);
    let g = end(s);
    assert_eq!(HOST_CALLS.load(Ordering::SeqCst), 1);
    let list = nodes(g);
    assert_eq!(list.iter().map(|&n| kind(n)).collect::<Vec<_>>(), [0, 3, 0]);
    let exec = instantiate(g);
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(HOST_CALLS.load(Ordering::SeqCst), 11);

    // An explicit host node, with parameters that can be read and replaced.
    let explicit = graph();
    let params = HostNodeParams { function, data: 100 as Opaque };
    let mut node = null_mut();
    assert_eq!(unsafe { cuGraphAddHostNode(&mut node, explicit, std::ptr::null(), 0, &params) }, Success);
    let after = kernel(explicit, &[node], f, &device);
    let exec2 = instantiate(explicit);
    assert_eq!(cuGraphLaunch(exec2, s), Success);
    assert_eq!(HOST_CALLS.load(Ordering::SeqCst), 111);
    let mut got = HostNodeParams { function: null_mut(), data: null_mut() };
    assert_eq!(unsafe { cuGraphHostNodeGetParams(node, &mut got) }, Success);
    assert_eq!((got.function, got.data), (function, 100 as Opaque));
    let replacement = HostNodeParams { function, data: 1000 as Opaque };
    assert_eq!(unsafe { cuGraphExecHostNodeSetParams(exec2, node, &replacement) }, Success);
    assert_eq!(cuGraphLaunch(exec2, s), Success);
    assert_eq!(HOST_CALLS.load(Ordering::SeqCst), 1111);
    assert_eq!(unsafe { cuGraphAddHostNode(&mut node, explicit, std::ptr::null(), 0, &HostNodeParams { function: null_mut(), data: null_mut() }) }, InvalidValue);
    let _ = after;

    // Launching the graph into a capture nests the host node.
    let outer = stream();
    begin(outer, RELAXED);
    assert_eq!(cuGraphLaunch(exec, outer), Success);
    let nested = end(outer);
    let nested_exec = instantiate(nested);
    assert_eq!(cuGraphLaunch(nested_exec, s), Success);
    assert_eq!(HOST_CALLS.load(Ordering::SeqCst), 1121);

    for e in [exec, exec2, nested_exec] {
        assert_eq!(cuGraphExecDestroy(e), Success);
    }
    for graph in [g, explicit, nested] {
        assert_eq!(cuGraphDestroy(graph), Success);
    }
    assert_eq!(cuStreamDestroy_v2(outer), Success);
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn fake_event_nodes_and_child_graphs() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let (s, t) = (stream(), stream());
    let f = function(module());
    let (device, _) = allocations();
    let e = event();

    let child = graph();
    let inner = kernel(child, &[], f, &device);
    let g = graph();
    let start = kernel(g, &[], f, &device);
    let mut nested = null_mut();
    assert_eq!(unsafe { cuGraphAddChildGraphNode(&mut nested, g, [start].as_ptr(), 1, child) }, Success);
    let mut record = null_mut();
    assert_eq!(unsafe { cuGraphAddEventRecordNode(&mut record, g, [nested].as_ptr(), 1, e) }, Success);
    assert_eq!([start, nested, record].map(kind), [0, 4, 7]);
    let mut got = null_mut();
    assert_eq!(unsafe { cuGraphEventRecordNodeGetEvent(record, &mut got) }, Success);
    assert_eq!(got, e);
    assert_eq!(unsafe { cuGraphEventWaitNodeGetEvent(record, &mut got) }, InvalidValue);
    let e2 = event();
    assert_eq!(cuGraphEventRecordNodeSetEvent(record, e2), Success);
    assert_eq!(cuGraphEventRecordNodeSetEvent(record, e), Success);

    let exec = instantiate(g);
    let before = work();
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert!(work() - before >= 2, "child kernel and kernel are replayed");
    // A second graph, on another stream, waits for the event the first one records.
    let waiter = graph();
    let mut wait = null_mut();
    assert_eq!(unsafe { cuGraphAddEventWaitNode(&mut wait, waiter, std::ptr::null(), 0, e) }, Success);
    kernel(waiter, &[wait], f, &device);
    let wait_exec = instantiate(waiter);
    assert_eq!(cuGraphLaunch(wait_exec, t), Success);
    assert_eq!(cuStreamSynchronize(t), Success);
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuGraphExecEventWaitNodeSetEvent(wait_exec, wait, e2), Success);
    assert_eq!(cuGraphExecEventRecordNodeSetEvent(wait_exec, wait, e2), InvalidValue);
    let _ = inner;

    let path = std::env::temp_dir().join(format!("cfi-graph-{}.dot", std::process::id()));
    let c_path = std::ffi::CString::new(path.to_str().unwrap()).unwrap();
    assert_eq!(unsafe { cuGraphDebugDotPrint(g, c_path.as_ptr(), 0) }, Success);
    let dot = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(dot.starts_with("digraph") && dot.contains("EVENT_RECORD") && dot.contains("->"));

    for x in [exec, wait_exec] {
        assert_eq!(cuGraphExecDestroy(x), Success);
    }
    for graph in [g, child, waiter] {
        assert_eq!(cuGraphDestroy(graph), Success);
    }
    for x in [e, e2] {
        assert_eq!(cuEventDestroy_v2(x), Success);
    }
    for stream in [s, t] {
        assert_eq!(cuStreamDestroy_v2(stream), Success);
    }
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}
