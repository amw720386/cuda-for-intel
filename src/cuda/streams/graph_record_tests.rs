//! Recorded graph replay (`CUDA_FOR_INTEL_GRAPH_RECORD`) on the fake Level Zero backend.
use super::fake_tests::Fixture;
use crate::cuda::*;
use crate::level_zero::{self, fake, Fence};
use std::sync::Arc;

/// A closed regular list runs once per append into an immediate list, in that list's order,
/// with or without a completion signal and waits.
#[test]
fn regular_list_replays_through_immediate_append() {
    let _fixture = Fixture::new();
    state(|state| {
        let api = &state.runtime.api;
        let mut backend = level_zero::Context::create(api, &state.runtime.devices[0]).unwrap();
        backend.initialize(api).unwrap();
        let immediate = backend.create_stream(api).unwrap();
        let recorded = backend.create_list(api).unwrap();
        let buffer = backend.alloc(api, 256).unwrap();
        for _ in 0..3 {
            recorded
                .fill(api, (std::ptr::with_exposed_provenance_mut(buffer), 7, 256), None, &[])
                .unwrap();
        }
        recorded.close(api).unwrap();
        let before = fake::snapshot().ledger.completed;
        // The bare shape: no signal, no waits.
        immediate.append_lists(api, &recorded, None, &[]).unwrap();
        let mut fences = level_zero::Fences::default();
        let done: Arc<Fence> = fences.acquire(api, &backend).unwrap();
        immediate.append_lists(api, &recorded, Some(&done), &[]).unwrap();
        let after: Arc<Fence> = fences.acquire(api, &backend).unwrap();
        immediate
            .append_lists(api, &recorded, Some(&after), std::slice::from_ref(&done))
            .unwrap();
        immediate.wait(api, u64::MAX).unwrap();
        assert!(done.ready(api).unwrap() && after.ready(api).unwrap());
        let stats = fake::snapshot();
        // Three fills per replay, plus a signal marker (second and third) and a wait marker (third).
        assert_eq!(stats.ledger.completed - before, 3 + 4 + 5);
        assert!(stats.ledger.violations.is_empty(), "{stats:?}");
        // A list whose memory was freed must not run again.
        backend.free(api, buffer).unwrap();
        assert!(immediate.append_lists(api, &recorded, None, &[]).is_err());
        assert_eq!(fake::snapshot().ledger.violations, ["executed a list referencing a destroyed resource"]);
        drop((done, after));
        fences.destroy(api).unwrap();
        recorded.destroy(api).unwrap();
        immediate.destroy(api).unwrap();
        backend.destroy(api).unwrap();
        Ok::<(), CuResult>(())
    })
    .unwrap();
}

use super::fake_tests::{BATCH, allocations, assert_baseline, context, function, launch, module, stream};
use crate::cuda::graph_record::{Stats, TEST_ENABLED};
use super::*;
use crate::cuda::graph_api::{
    KernelNodeParams, cuGraphExecKernelNodeSetParams_v2, cuGraphGetNodes, cuGraphKernelNodeGetParams_v2,
};
use crate::cuda::graphs::*;
use crate::level_zero::fake::Kind;
use std::ptr::null_mut;

const RELAXED: c_int = 2;

/// Enables recording for one test thread; restores the environment default on drop.
struct Recording;

impl Recording {
    fn on() -> Self {
        TEST_ENABLED.set(Some(true));
        Self
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        TEST_ENABLED.set(None);
    }
}

fn capture(s: Opaque, body: impl FnOnce()) -> Opaque {
    assert_eq!(cuStreamBeginCapture_v2(s, RELAXED), Success);
    body();
    let mut graph = null_mut();
    assert_eq!(unsafe { cuStreamEndCapture(s, &mut graph) }, Success);
    graph
}

fn instantiate(graph: Opaque) -> Opaque {
    let mut exec = null_mut();
    assert_eq!(unsafe { cuGraphInstantiateWithFlags(&mut exec, graph, 0) }, Success);
    exec
}

/// A ggml-like decode chain: `kernels` launches, then a memset and a device copy.
fn chain(f: Opaque, s: Opaque, device: u64, kernels: usize) {
    for _ in 0..kernels {
        assert_eq!(launch(f, s, device), Success);
    }
    assert_eq!(cuMemsetD8Async(device + 32, 7, 16, s), Success);
    assert_eq!(unsafe { cuMemcpyDtoDAsync_v2(device + 48, device, 8, s) }, Success);
}

fn stats() -> Stats {
    state(|state| Ok(state.contexts.values().map(|c| c.graph_lists.stats).next().unwrap())).unwrap()
}

fn live_lists() -> usize {
    state(|state| Ok(state.contexts.values().map(|c| c.graph_lists.live_lists()).sum())).unwrap()
}

fn regular_lists() -> usize {
    fake::snapshot().live.get(&Kind::List).copied().unwrap_or(0)
}

#[test]
fn recorded_graph_runs_one_list_per_launch() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let (s, f) = (stream(), function(module()));
    let (device, _) = allocations();
    let exec = instantiate(capture(s, || chain(f, s, device, 120)));

    // Default: every node is re-issued on the stream.
    let before = fake::snapshot().ledger.submitted;
    assert_eq!(cuGraphLaunch(exec, s), Success);
    let reissued = fake::snapshot().ledger.submitted - before;
    assert_eq!(stats(), Stats::default(), "recording is off by default");

    let _on = Recording::on();
    let lists = regular_lists();
    for round in 1..=3 {
        let before = fake::snapshot().ledger.submitted;
        assert_eq!(cuGraphLaunch(exec, s), Success);
        // The same device work as the re-issue path, plus the completion signal.
        assert_eq!(fake::snapshot().ledger.submitted - before, reissued + 1);
        assert_eq!(stats(), Stats { recorded: 1, replayed: round, ..Stats::default() });
    }
    assert_eq!(regular_lists(), lists + 1, "one recorded list, reused");
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn exec_update_and_node_params_rerecord() {
    let _fixture = Fixture::new();
    let _on = Recording::on();
    let ctx = context(None);
    let (s, f) = (stream(), function(module()));
    let (device, _) = allocations();
    let (other, _) = allocations();
    let exec = instantiate(capture(s, || chain(f, s, device, 8)));
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats().recorded, 1);

    // ggml's per-token path: capture again, cuGraphExecUpdate, launch.
    let update = capture(s, || chain(f, s, other, 8));
    let mut info = std::mem::MaybeUninit::<ExecUpdateInfo>::zeroed();
    assert_eq!(unsafe { cuGraphExecUpdate_v2(exec, update, info.as_mut_ptr()) }, Success);
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats(), Stats { recorded: 2, replayed: 3, ..Stats::default() });

    // A per-node parameter change also invalidates the list.
    let mut nodes = [null_mut(); 16];
    let mut count = nodes.len();
    assert_eq!(unsafe { cuGraphGetNodes(update, nodes.as_mut_ptr(), &mut count) }, Success);
    let mut params = std::mem::MaybeUninit::<KernelNodeParams>::zeroed();
    let kernel = nodes[..count]
        .iter()
        .copied()
        .find(|&node| unsafe { cuGraphKernelNodeGetParams_v2(node, params.as_mut_ptr()) } == Success)
        .unwrap();
    assert_eq!(unsafe { cuGraphExecKernelNodeSetParams_v2(exec, kernel, params.as_ptr()) }, Success);
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats(), Stats { recorded: 3, replayed: 4, ..Stats::default() });
    // Stale lists are destroyed once their runs finished; one stays live.
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(live_lists(), 1);
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuGraphDestroy(update), Success);
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn unrecordable_graphs_keep_reissuing() {
    let _fixture = Fixture::new();
    let _on = Recording::on();
    for policy in [None, BATCH] {
        let ctx = context(policy);
        let (s, f) = (stream(), function(module()));
        let (device, host) = allocations();
        // A host-to-device copy reads pageable or pinned host memory: not recorded.
        let mixed = instantiate(capture(s, || {
            assert_eq!(launch(f, s, device), Success);
            assert_eq!(unsafe { cuMemcpyHtoDAsync_v2(device, host, 16, s) }, Success);
        }));
        let kernels = instantiate(capture(s, || chain(f, s, device, 4)));
        for _ in 0..2 {
            assert_eq!(cuGraphLaunch(mixed, s), Success);
            assert_eq!(cuGraphLaunch(kernels, s), Success);
        }
        let expected = if policy.is_none() {
            Stats { recorded: 1, replayed: 2, reissued: 2, waited: 0 }
        } else {
            // Batching streams have no immediate list to append to.
            Stats { reissued: 4, ..Stats::default() }
        };
        assert_eq!(stats(), expected);
        assert_eq!(cuStreamSynchronize(s), Success);
        assert_eq!(cuGraphExecDestroy(mixed), Success);
        assert_eq!(cuGraphExecDestroy(kernels), Success);
        assert_eq!(cuStreamDestroy_v2(s), Success);
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}

#[test]
fn module_unload_and_free_invalidate_recorded_graphs() {
    let _fixture = Fixture::new();
    let _on = Recording::on();
    let ctx = context(None);
    let s = stream();
    let m = module();
    let f = function(m);
    let (device, _) = allocations();
    let exec = instantiate(capture(s, || chain(f, s, device, 4)));
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(live_lists(), 1);
    assert_eq!(cuModuleUnload(m), Success);
    assert_eq!(live_lists(), 0, "unload drops lists that name its kernels");
    // As without recording, the stale graph is rejected before anything runs.
    assert_eq!(cuGraphLaunch(exec, s), InvalidHandle);

    let f = function(module());
    let exec2 = instantiate(capture(s, || chain(f, s, device, 4)));
    assert_eq!(cuGraphLaunch(exec2, s), Success);
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(stats().recorded, 2);
    // Any free re-records (the address may come back as another allocation), even one the
    // graph does not use.
    let (spare, _) = allocations();
    assert_eq!(cuGraphLaunch(exec2, s), Success);
    assert_eq!(stats().recorded, 2, "a new allocation keeps the list");
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuMemFree_v2(spare), Success);
    assert_eq!(cuGraphLaunch(exec2, s), Success);
    assert_eq!(stats().recorded, 3);
    assert_eq!(cuGraphLaunch(exec2, s), Success);
    assert_eq!(stats().recorded, 3);
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuMemFree_v2(device), Success);
    assert_eq!(cuGraphLaunch(exec2, s), InvalidValue);
    assert!(fake::snapshot().ledger.violations.is_empty());
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuGraphExecDestroy(exec2), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

/// The bug class from review: a graph names a pointer, the memory is freed and a new allocation
/// takes its place (ggml's pool does this). The old list must not run again: the next launch
/// validates against the new allocation table and records a fresh list.
#[test]
fn free_then_reallocate_rerecords() {
    let _fixture = Fixture::new();
    let _on = Recording::on();
    let ctx = context(None);
    let (s, f) = (stream(), function(module()));
    let mut buffer = 0u64;
    assert_eq!(unsafe { cuMemAlloc_v2(&mut buffer, 256) }, Success);
    let (scratch, _) = allocations();
    // The graph names only `buffer`; `scratch` is unrelated memory that comes and goes.
    let exec = instantiate(capture(s, || chain(f, s, buffer, 6)));
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats(), Stats { recorded: 1, replayed: 2, ..Stats::default() });
    assert_eq!(cuStreamSynchronize(s), Success);

    // Free and reallocate: whatever address the new allocation gets, the list is re-recorded.
    assert_eq!(cuMemFree_v2(scratch), Success);
    let mut again = 0u64;
    assert_eq!(unsafe { cuMemAlloc_v2(&mut again, 64) }, Success);
    let regular = regular_lists();
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats(), Stats { recorded: 2, replayed: 3, ..Stats::default() });
    assert_eq!(cuStreamSynchronize(s), Success);
    // The stale list was retired and destroyed once its last run finished.
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats(), Stats { recorded: 2, replayed: 4, ..Stats::default() });
    assert_eq!(regular_lists(), regular, "old list destroyed, new one live");
    assert_eq!(live_lists(), 1);

    // A pinned host free counts too.
    let mut host = null_mut();
    assert_eq!(unsafe { cuMemHostAlloc(&mut host, 64, 0) }, Success);
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats().recorded, 2);
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuMemFreeHost(host), Success);
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats().recorded, 3);

    // Freeing the graph's own buffer makes the stale graph fail validation, as without recording.
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuMemFree_v2(buffer), Success);
    assert_eq!(cuGraphLaunch(exec, s), InvalidValue);
    assert!(fake::snapshot().ledger.violations.is_empty());
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn nested_graph_and_destroyed_exec_clean_up() {
    let _fixture = Fixture::new();
    let _on = Recording::on();
    let ctx = context(None);
    let (s, f) = (stream(), function(module()));
    let (device, _) = allocations();
    let exec = instantiate(capture(s, || chain(f, s, device, 4)));
    assert_eq!(cuGraphLaunch(exec, s), Success);
    // Launching into a capture still nests the graph; the outer graph records separately.
    let outer = instantiate(capture(s, || assert_eq!(cuGraphLaunch(exec, s), Success)));
    assert_eq!(cuGraphLaunch(outer, s), Success);
    assert_eq!(stats(), Stats { recorded: 2, replayed: 2, ..Stats::default() });
    // Destroying an executable while its list may run defers the list's destruction.
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuGraphExecDestroy(outer), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

const QUANTIZE: &std::ffi::CStr = c".version 4.1 .target sm_52 .address_size 64 \
.entry _Z13quantize_q8_1PKfPvxxxxxj5uint3(.param .u64 .ptr .global x, .param .u64 .ptr .global y, \
.param .u64 a, .param .u64 b, .param .u64 c, .param .u64 d, .param .u64 e, .param .u32 f, \
.param .align 4 .b8 g[12]) { ret; }";

#[test]
fn substitution_is_decided_when_the_graph_is_recorded() {
    let _fixture = Fixture::new();
    let _on = Recording::on();
    let ctx = context(None);
    let s = stream();
    let (a, _) = allocations();
    let (b, _) = allocations();
    // SAFETY: tests that touch the fake runtime are serialised by the fixture.
    unsafe { std::env::set_var("CUDA_FOR_INTEL_SUBST", "quantize_q8_1") };
    let mut module = null_mut();
    assert_eq!(unsafe { cuModuleLoadData(&mut module, QUANTIZE.as_ptr().cast()) }, Success);
    let mut f = null_mut();
    assert_eq!(
        unsafe { cuModuleGetFunction(&mut f, module, c"_Z13quantize_q8_1PKfPvxxxxxj5uint3".as_ptr()) },
        Success
    );
    let natives = || state(|state| Ok(state.contexts.values().map(|c| c.subst_len()).sum::<usize>())).unwrap();
    let exec = instantiate(capture(s, || {
        let (mut x, mut y) = (a, b);
        let mut u = [0u64; 5];
        let mut n = 0u32;
        let mut g = [0u32; 3];
        let mut params: Vec<*mut c_void> = vec![(&raw mut x).cast(), (&raw mut y).cast()];
        params.extend(u.iter_mut().map(|v| std::ptr::from_mut(v).cast::<c_void>()));
        params.push((&raw mut n).cast());
        params.push(g.as_mut_ptr().cast());
        let launched =
            unsafe { cuLaunchKernel(f, 1, 1, 1, 256, 1, 1, 0, s, params.as_mut_ptr().cast(), null_mut()) };
        assert_eq!(launched, Success);
    }));
    assert_eq!(natives(), 0, "capture issues nothing");
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(natives(), 1, "the native kernel is built when the graph is recorded");
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats(), Stats { recorded: 1, replayed: 2, ..Stats::default() });
    unsafe { std::env::remove_var("CUDA_FOR_INTEL_SUBST") };
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

/// Host cost of `cuGraphLaunch` for a decode-sized graph on the fake backend, re-issued against
/// recorded (ignored: timing only; the fake driver costs almost nothing, so this is our own
/// host path, not a GPU number).
#[test]
#[ignore = "timing only"]
fn graph_launch_host_cost_reissue_vs_recorded() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let (s, f) = (stream(), function(module()));
    let (device, _) = allocations();
    let kernels = 725;
    let exec = instantiate(capture(s, || chain(f, s, device, kernels)));
    let rounds = 200;
    for recorded in [false, true, false, true] {
        TEST_ENABLED.set(Some(recorded));
        assert_eq!(cuGraphLaunch(exec, s), Success);
        assert_eq!(cuStreamSynchronize(s), Success);
        let start = std::time::Instant::now();
        for _ in 0..rounds {
            assert_eq!(cuGraphLaunch(exec, s), Success);
            assert_eq!(cuStreamSynchronize(s), Success);
        }
        let per = start.elapsed().as_secs_f64() * 1e6 / rounds as f64;
        println!(
            "recorded={recorded}: {per:.1} us per cuGraphLaunch of {kernels} kernels ({:.3} us/kernel)",
            per / kernels as f64
        );
    }
    TEST_ENABLED.set(None);
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}
