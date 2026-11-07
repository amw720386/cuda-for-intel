//! Stream capture and graph replay against the fake Level Zero backend.
use super::events::{
    cuEventCreate, cuEventDestroy_v2, cuEventQuery, cuEventRecord, cuEventRecordWithFlags, cuEventSynchronize,
    cuStreamWaitEvent,
};
use super::fake_tests::{
    BATCH, Fixture, allocations, assert_baseline, context, event, function, launch, module, stream,
};
use super::*;
use crate::cuda::graphs::*;
use crate::cuda::*;
use crate::level_zero::fake;
use std::ptr::null_mut;

const GLOBAL: c_int = 0;
const THREAD_LOCAL: c_int = 1;
const RELAXED: c_int = 2;

/// Native work issued so far, recorded or submitted.
fn work() -> usize {
    let s = fake::snapshot();
    s.recorded + s.ledger.submitted
}

fn begin(stream: Opaque, mode: c_int) {
    assert_eq!(cuStreamBeginCapture_v2(stream, mode), Success);
}

fn end(stream: Opaque) -> Opaque {
    let mut graph = null_mut();
    assert_eq!(unsafe { cuStreamEndCapture(stream, &mut graph) }, Success);
    assert!(!graph.is_null());
    graph
}

fn instantiate(graph: Opaque) -> Opaque {
    let mut exec = null_mut();
    assert_eq!(
        unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) },
        Success
    );
    exec
}

fn capturing(stream: Opaque) -> c_int {
    let mut status = -1;
    assert_eq!(unsafe { cuStreamIsCapturing(stream, &mut status) }, Success);
    status
}

/// Capture `launches` kernels, one copy in each direction, a memset and a
/// device copy on `stream`.
fn record_mixed(f: Opaque, stream: Opaque, device: u64, host: Opaque, launches: usize) {
    for _ in 0..launches {
        assert_eq!(launch(f, stream, device), Success);
    }
    assert_eq!(
        unsafe { cuMemcpyHtoDAsync_v2(device, host, 16, stream) },
        Success
    );
    assert_eq!(
        unsafe { cuMemcpyDtoHAsync_v2(host, device + 16, 16, stream) },
        Success
    );
    assert_eq!(cuMemsetD8Async(device + 32, 7, 16, stream), Success);
    assert_eq!(
        unsafe { cuMemcpyDtoDAsync_v2(device + 48, device, 8, stream) },
        Success
    );
}

#[test]
fn fake_capture_defers_work_and_launch_replays_it() {
    let _fixture = Fixture::new();
    for policy in [None, BATCH] {
        let ctx = context(policy);
        let s = stream();
        let f = function(module());
        let (device, host) = allocations();
        fake::pause();
        let before = work();

        assert_eq!(capturing(s), 0);
        begin(s, RELAXED);
        assert_eq!(capturing(s), 1);
        record_mixed(f, s, device, host, 2);
        assert_eq!(work(), before, "capture must not submit anything");
        let graph = end(s);
        assert_eq!(capturing(s), 0);
        let exec = instantiate(graph);
        assert_eq!(work(), before, "instantiate must not submit anything");

        // The executable keeps the recorded nodes after the graph is destroyed.
        assert_eq!(cuGraphDestroy(graph), Success);
        assert_eq!(cuGraphDestroy(graph), InvalidHandle);
        assert_eq!(cuGraphLaunch(exec, s), Success);
        let one = work() - before;
        assert!(one >= 6, "replay submitted {one} operations");
        assert_eq!(cuGraphLaunch(exec, s), Success);
        assert_eq!(work() - before, 2 * one, "each launch replays every node");

        fake::complete();
        assert_eq!(cuStreamSynchronize(s), Success);
        assert_eq!(cuGraphExecDestroy(exec), Success);
        assert_eq!(cuGraphExecDestroy(exec), InvalidHandle);
        assert_eq!(cuStreamDestroy_v2(s), Success);
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}

#[test]
fn fake_capture_modes_and_state_errors() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    let f = function(module());
    let (device, _) = allocations();

    // The legacy stream cannot be captured and invalid modes are rejected.
    assert_eq!(cuStreamBeginCapture_v2(null_mut(), GLOBAL), StreamCaptureUnsupported);
    assert_eq!(cuStreamBeginCapture_v2(s, 3), InvalidValue);
    let mut graph = null_mut();
    assert_eq!(
        unsafe { cuStreamEndCapture(s, &mut graph) },
        StreamCaptureUnmatched
    );
    assert!(graph.is_null());

    for mode in [GLOBAL, THREAD_LOCAL, RELAXED] {
        begin(s, mode);
        assert_eq!(cuStreamBeginCapture_v2(s, mode), IllegalState);
        assert_eq!(launch(f, s, device), Success);
        assert_eq!(cuGraphDestroy(end(s)), Success);
    }

    // Host synchronization invalidates the capture; ending it reports that and
    // leaves the stream usable.
    begin(s, RELAXED);
    assert_eq!(cuStreamSynchronize(s), StreamCaptureUnsupported);
    assert_eq!(capturing(s), 2);
    assert_eq!(launch(f, s, device), StreamCaptureInvalidated);
    graph = null_mut();
    assert_eq!(
        unsafe { cuStreamEndCapture(s, &mut graph) },
        StreamCaptureInvalidated
    );
    assert!(graph.is_null());
    assert_eq!(capturing(s), 0);
    assert_eq!(launch(f, s, device), Success);
    assert_eq!(cuStreamSynchronize(s), Success);

    // A query is also host synchronization.
    begin(s, RELAXED);
    assert_eq!(cuStreamQuery(s), StreamCaptureUnsupported);
    assert_eq!(
        unsafe { cuStreamEndCapture(s, &mut graph) },
        StreamCaptureInvalidated
    );

    // Out-of-range allocations are rejected at capture time.
    begin(s, RELAXED);
    assert_eq!(launch(f, s, 0xdead_0000), InvalidValue);
    assert_eq!(cuMemsetD8Async(0xdead_0000, 0, 8, s), InvalidValue);
    assert_eq!(cuGraphDestroy(end(s)), Success);

    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn fake_capture_info_and_thread_mode() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    let (mut status, mut id, mut graph, mut deps, mut edges, mut count) =
        (-1, 99u64, 1 as Opaque, 1 as *const Opaque, 1 as *const c_void, 9usize);

    assert_eq!(
        unsafe { cuStreamGetCaptureInfo_v3(s, &mut status, &mut id, &mut graph, &mut deps, &mut edges, &mut count) },
        Success
    );
    assert_eq!((status, id, count), (0, 0, 0));
    assert!(graph.is_null() && deps.is_null() && edges.is_null());

    begin(s, RELAXED);
    assert_eq!(unsafe { cuStreamGetCaptureInfo(s, &mut status, &mut id) }, Success);
    assert_eq!(status, 1);
    assert_ne!(id, 0);
    let first = id;
    // All optional outputs may be null.
    assert_eq!(
        unsafe {
            cuStreamGetCaptureInfo_v2(s, null_mut(), null_mut(), null_mut(), null_mut(), null_mut())
        },
        Success
    );
    assert_eq!(cuGraphDestroy(end(s)), Success);
    begin(s, RELAXED);
    assert_eq!(unsafe { cuStreamGetCaptureInfo(s, &mut status, &mut id) }, Success);
    assert_ne!(id, first, "each capture has its own id");
    assert_eq!(cuGraphDestroy(end(s)), Success);

    let mut mode = RELAXED;
    assert_eq!(unsafe { cuThreadExchangeStreamCaptureMode(&mut mode) }, Success);
    assert_eq!(mode, GLOBAL);
    mode = THREAD_LOCAL;
    assert_eq!(unsafe { cuThreadExchangeStreamCaptureMode(&mut mode) }, Success);
    assert_eq!(mode, RELAXED);
    mode = 9;
    assert_eq!(unsafe { cuThreadExchangeStreamCaptureMode(&mut mode) }, InvalidValue);
    mode = GLOBAL;
    assert_eq!(unsafe { cuThreadExchangeStreamCaptureMode(&mut mode) }, Success);

    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn fake_capture_fork_join_through_events() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let (origin, worker) = (stream(), stream());
    let (fork, join) = (event(), event());
    let f = function(module());
    let (device, _) = allocations();
    fake::pause();
    let before = work();

    // origin -> fork -> worker -> join -> origin
    begin(origin, RELAXED);
    assert_eq!(launch(f, origin, device), Success);
    assert_eq!(cuEventRecord(fork, origin), Success);
    assert_eq!(cuStreamWaitEvent(worker, fork, 0), Success);
    assert_eq!(capturing(worker), 1, "waiting on a captured event joins the capture");
    assert_eq!(launch(f, worker, device), Success);
    assert_eq!(cuEventRecord(join, worker), Success);
    assert_eq!(cuStreamWaitEvent(origin, join, 0), Success);
    assert_eq!(launch(f, origin, device), Success);
    assert_eq!(cuEventQuery(fork), CapturedEvent);
    assert_eq!(cuEventSynchronize(join), CapturedEvent);
    assert_eq!(work(), before);
    let graph = end(origin);
    assert_eq!(capturing(worker), 0, "ending capture releases joined streams");
    // Once capture is over the events are ordinary again.
    assert_eq!(cuEventQuery(fork), Success);

    let exec = instantiate(graph);
    assert_eq!(cuGraphLaunch(exec, origin), Success);
    assert_eq!(work() - before, 3);
    fake::complete();
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuGraphDestroy(graph), Success);

    // A forked stream that never rejoins makes the capture invalid.
    fake::pause();
    begin(origin, RELAXED);
    assert_eq!(cuEventRecord(fork, origin), Success);
    assert_eq!(cuStreamWaitEvent(worker, fork, 0), Success);
    assert_eq!(launch(f, worker, device), Success);
    let mut out = null_mut();
    assert_eq!(
        unsafe { cuStreamEndCapture(origin, &mut out) },
        StreamCaptureUnjoined
    );
    assert!(out.is_null());
    assert_eq!(capturing(origin), 0);
    assert_eq!(capturing(worker), 0);

    // Work issued after the join event was recorded is not covered by it.
    begin(origin, RELAXED);
    assert_eq!(cuEventRecord(fork, origin), Success);
    assert_eq!(cuStreamWaitEvent(worker, fork, 0), Success);
    assert_eq!(cuEventRecord(join, worker), Success);
    assert_eq!(launch(f, worker, device), Success);
    assert_eq!(cuStreamWaitEvent(origin, join, 0), Success);
    assert_eq!(
        unsafe { cuStreamEndCapture(origin, &mut out) },
        StreamCaptureUnjoined
    );

    // An empty fork that rejoins is fine.
    begin(origin, RELAXED);
    assert_eq!(cuEventRecord(fork, origin), Success);
    assert_eq!(cuStreamWaitEvent(worker, fork, 0), Success);
    assert_eq!(cuEventRecord(join, worker), Success);
    assert_eq!(cuStreamWaitEvent(origin, join, 0), Success);
    assert_eq!(cuGraphDestroy(end(origin)), Success);

    fake::complete();
    assert_eq!(cuEventDestroy_v2(fork), Success);
    assert_eq!(cuEventDestroy_v2(join), Success);
    assert_eq!(cuStreamDestroy_v2(origin), Success);
    assert_eq!(cuStreamDestroy_v2(worker), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn fake_capture_isolation_and_merge_errors() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let (a, b) = (stream(), stream());
    let (outside, inside) = (event(), event());
    let mut out = null_mut();

    // A capturing stream may not wait for work outside the capture.
    assert_eq!(cuEventRecord(outside, b), Success);
    begin(a, RELAXED);
    assert_eq!(cuStreamWaitEvent(a, outside, 0), StreamCaptureIsolation);
    assert_eq!(capturing(a), 2);
    assert_eq!(unsafe { cuStreamEndCapture(a, &mut out) }, StreamCaptureInvalidated);

    // An event that was never recorded imposes no dependency.
    let unused = event();
    begin(a, RELAXED);
    assert_eq!(cuStreamWaitEvent(a, unused, 0), Success);
    assert_eq!(cuGraphDestroy(end(a)), Success);

    // Two captures cannot be merged.
    begin(a, RELAXED);
    begin(b, RELAXED);
    assert_eq!(cuEventRecord(inside, a), Success);
    assert_eq!(cuStreamWaitEvent(b, inside, 0), StreamCaptureMerge);
    assert_eq!(unsafe { cuStreamEndCapture(b, &mut out) }, StreamCaptureInvalidated);
    // The other capture is unaffected.
    assert_eq!(capturing(a), 1);
    assert_eq!(cuGraphDestroy(end(a)), Success);

    // Joined streams cannot end a capture they did not start.
    begin(a, RELAXED);
    assert_eq!(cuEventRecord(inside, a), Success);
    assert_eq!(cuStreamWaitEvent(b, inside, 0), Success);
    assert_eq!(unsafe { cuStreamEndCapture(b, &mut out) }, StreamCaptureUnmatched);
    assert_eq!(cuEventRecord(inside, b), Success);
    assert_eq!(cuStreamWaitEvent(a, inside, 0), Success);
    assert_eq!(cuGraphDestroy(end(a)), Success);

    // Events record normally outside capture; external records are unsupported.
    assert_eq!(cuEventRecordWithFlags(outside, a, 0), Success);
    assert_eq!(cuEventRecordWithFlags(outside, a, 1), NotSupported);
    assert_eq!(cuEventRecordWithFlags(outside, a, 2), InvalidValue);
    let mut late = null_mut();
    assert_eq!(unsafe { cuEventCreate(&mut late, 2) }, Success);
    assert_eq!(cuEventDestroy_v2(late), Success);

    for e in [outside, inside, unused] {
        assert_eq!(cuEventDestroy_v2(e), Success);
    }
    assert_eq!(cuStreamDestroy_v2(a), Success);
    assert_eq!(cuStreamDestroy_v2(b), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn fake_graph_launch_into_capture_nests_and_validates() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let (s, other) = (stream(), stream());
    let f = function(module());
    let (device, host) = allocations();
    fake::pause();
    let before = work();

    begin(s, RELAXED);
    record_mixed(f, s, device, host, 1);
    let inner = end(s);
    let inner_exec = instantiate(inner);

    begin(s, RELAXED);
    assert_eq!(launch(f, s, device), Success);
    assert_eq!(cuGraphLaunch(inner_exec, s), Success);
    assert_eq!(work(), before);
    let outer = end(s);
    let outer_exec = instantiate(outer);
    assert_eq!(cuGraphLaunch(outer_exec, other), Success);
    let issued = work() - before;
    assert!(issued >= 6, "child nodes replay with the parent: {issued}");

    // Nodes are validated against live allocations at launch.
    let (stale, _) = allocations();
    begin(s, RELAXED);
    assert_eq!(launch(f, s, stale), Success);
    let stale_graph = end(s);
    let stale_exec = instantiate(stale_graph);
    fake::complete();
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuStreamSynchronize(other), Success);
    assert_eq!(cuMemFree_v2(stale), Success);
    let after = work();
    assert_eq!(cuGraphLaunch(stale_exec, s), InvalidValue);
    assert_eq!(work(), after, "a rejected graph issues nothing");

    for exec in [inner_exec, outer_exec, stale_exec] {
        assert_eq!(cuGraphExecDestroy(exec), Success);
    }
    for graph in [inner, outer, stale_graph] {
        assert_eq!(cuGraphDestroy(graph), Success);
    }
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuStreamDestroy_v2(other), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn fake_graph_instantiate_variants_and_exec_update() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    let f = function(module());
    let g = {
        let mut out = null_mut();
        assert_eq!(
            unsafe {
                cuModuleGetFunction(
                    &mut out,
                    module(),
                    c"k".as_ptr(),
                )
            },
            Success
        );
        out
    };
    let (device, host) = allocations();

    begin(s, RELAXED);
    record_mixed(f, s, device, host, 2);
    let base = end(s);

    // Instantiate forms.
    let mut exec = null_mut();
    assert_eq!(unsafe { cuGraphInstantiateWithFlags(&mut exec, base, 4) }, NotSupported);
    assert_eq!(unsafe { cuGraphInstantiateWithFlags(&mut exec, base, 0x100) }, InvalidValue);
    assert_eq!(unsafe { cuGraphInstantiateWithFlags(&mut exec, null_mut(), 0) }, InvalidHandle);
    let mut log = [1 as c_char; 4];
    let mut error_node = 1 as Opaque;
    assert_eq!(
        unsafe { cuGraphInstantiate_v1(&mut exec, base, &mut error_node, log.as_mut_ptr(), 4) },
        Success
    );
    assert!(error_node.is_null() && log[0] == 0);
    assert_eq!(cuGraphExecDestroy(exec), Success);
    let mut params = InstantiateParamsForTest::new(2);
    assert_eq!(
        unsafe { cuGraphInstantiateWithParams(&mut exec, base, std::ptr::from_mut(&mut params).cast()) },
        Success
    );
    assert_eq!(params.result, 0);
    assert_eq!(cuGraphUpload(exec, s), Success);
    assert_eq!(cuGraphUpload(null_mut(), s), InvalidHandle);

    // Same shape, new parameters: update succeeds.
    begin(s, RELAXED);
    record_mixed(f, s, device + 8, host, 2);
    let same = end(s);
    let mut info = ExecUpdateInfoForTest::default();
    assert_eq!(
        unsafe { cuGraphExecUpdate_v2(exec, same, std::ptr::from_mut(&mut info).cast()) },
        Success
    );
    assert_eq!(info.result, 0);

    // A different node count is a topology change and leaves the exec usable.
    begin(s, RELAXED);
    record_mixed(f, s, device, host, 3);
    let longer = end(s);
    assert_eq!(
        unsafe { cuGraphExecUpdate_v2(exec, longer, std::ptr::from_mut(&mut info).cast()) },
        GraphExecUpdateFailure
    );
    assert_eq!(info.result, 2);
    let (mut node, mut result) = (1 as Opaque, -1);
    assert_eq!(
        unsafe { cuGraphExecUpdate(exec, longer, &mut node, &mut result) },
        GraphExecUpdateFailure
    );
    assert_eq!(result, 2);
    assert!(node.is_null());

    // A different node kind or kernel function cannot be updated.
    begin(s, RELAXED);
    assert_eq!(cuMemsetD8Async(device, 0, 8, s), Success);
    for _ in 0..5 {
        assert_eq!(cuMemsetD8Async(device, 0, 8, s), Success);
    }
    let kinds = end(s);
    assert_eq!(
        unsafe { cuGraphExecUpdate_v2(exec, kinds, std::ptr::from_mut(&mut info).cast()) },
        GraphExecUpdateFailure
    );
    assert_eq!(info.result, 3);
    begin(s, RELAXED);
    assert_eq!(launch(g, s, device), Success);
    let other_function = end(s);
    begin(s, RELAXED);
    assert_eq!(launch(f, s, device), Success);
    let one_kernel = end(s);
    let one_exec = instantiate(one_kernel);
    assert_eq!(
        unsafe { cuGraphExecUpdate_v2(one_exec, other_function, std::ptr::from_mut(&mut info).cast()) },
        GraphExecUpdateFailure
    );
    assert_eq!(info.result, 4);
    assert_eq!(
        unsafe { cuGraphExecUpdate_v2(null_mut(), same, null_mut()) },
        InvalidHandle
    );

    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(cuStreamSynchronize(s), Success);
    for e in [exec, one_exec] {
        assert_eq!(cuGraphExecDestroy(e), Success);
    }
    for graph in [base, same, longer, kinds, other_function, one_kernel] {
        assert_eq!(cuGraphDestroy(graph), Success);
    }
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[repr(C)]
struct InstantiateParamsForTest {
    flags: u64,
    upload_stream: Opaque,
    error_node: Opaque,
    result: c_int,
}

impl InstantiateParamsForTest {
    fn new(flags: u64) -> Self {
        Self {
            flags,
            upload_stream: null_mut(),
            error_node: 1 as Opaque,
            result: -1,
        }
    }
}

#[repr(C)]
struct ExecUpdateInfoForTest {
    result: c_int,
    error_node: Opaque,
    error_from_node: Opaque,
}

impl Default for ExecUpdateInfoForTest {
    fn default() -> Self {
        Self {
            result: -1,
            error_node: null_mut(),
            error_from_node: null_mut(),
        }
    }
}

#[test]
fn fake_destroying_a_context_drops_its_graphs() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    begin(s, RELAXED);
    let exec = instantiate(end(s));
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_eq!(cuGraphExecDestroy(exec), InvalidHandle);
    assert_baseline();
}
