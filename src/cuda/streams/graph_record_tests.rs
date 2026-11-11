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
            Stats { recorded: 1, replayed: 2, reissued: 2, ..Stats::default() }
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
    // Allocating, and freeing memory the graph does not use, keep the list.
    let (spare, _) = allocations();
    assert_eq!(cuGraphLaunch(exec2, s), Success);
    assert_eq!(stats().recorded, 2, "a new allocation keeps the list");
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuMemFree_v2(spare), Success);
    assert_eq!(cuGraphLaunch(exec2, s), Success);
    assert_eq!(stats().recorded, 2, "an unrelated free keeps the list");
    assert_eq!(stats().kept_after_free, 1);
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuMemFree_v2(device), Success);
    assert_eq!(cuGraphLaunch(exec2, s), InvalidValue);
    assert!(fake::snapshot().ledger.violations.is_empty());
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuGraphExecDestroy(exec2), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

/// The bug class from review: a graph names a buffer, the buffer is freed and a new allocation
/// takes the same address (ggml's pool, or the driver, can do this). The old list must not run
/// again: the next launch validates against the new allocation table and records a fresh list.
#[test]
fn free_then_reallocate_at_the_same_address_rerecords() {
    let _fixture = Fixture::new();
    let _on = Recording::on();
    let ctx = context(None);
    let (s, f) = (stream(), function(module()));
    let mut buffer = 0u64;
    assert_eq!(unsafe { cuMemAlloc_v2(&mut buffer, 256) }, Success);
    let exec = instantiate(capture(s, || chain(f, s, buffer, 6)));
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats(), Stats { recorded: 1, replayed: 2, ..Stats::default() });
    assert_eq!(cuStreamSynchronize(s), Success);

    assert_eq!(cuMemFree_v2(buffer), Success);
    fake::reuse_address(buffer as usize);
    let mut again = 0u64;
    assert_eq!(unsafe { cuMemAlloc_v2(&mut again, 256) }, Success);
    assert_eq!(again, buffer, "the fake handed the freed address back");
    let regular = regular_lists();
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats(), Stats { recorded: 2, replayed: 3, ..Stats::default() });
    assert_eq!(cuStreamSynchronize(s), Success);
    // The stale list was retired and destroyed once its last run finished.
    assert_eq!(cuGraphLaunch(exec, s), Success);
    assert_eq!(stats(), Stats { recorded: 2, replayed: 4, ..Stats::default() });
    assert_eq!(regular_lists(), regular, "old list destroyed, new one live");
    assert_eq!(live_lists(), 1);
    assert!(fake::snapshot().ledger.violations.is_empty());
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

/// What ggml in Ollama does per decode token: allocate the token's input buffers, copy the inputs
/// in (outside the graph), copy the CPY-indirection pointer table, launch the graph, synchronize,
/// free the inputs. Every few tokens the KV length changes, ggml re-captures and calls
/// cuGraphExecUpdate. The recorded list must survive the input frees and be re-recorded only on
/// the updates.
#[test]
fn ollama_token_pattern_keeps_the_list_across_input_frees() {
    let _fixture = Fixture::new();
    let _on = Recording::on();
    let ctx = context(None);
    let (s, f) = (stream(), function(module()));
    let (weights, host) = allocations();
    let (table, _) = allocations();
    let exec = instantiate(capture(s, || chain(f, s, weights, 20)));
    let tokens = 24;
    let update_every = 8;
    for token in 0..tokens {
        let mut input = 0u64;
        assert_eq!(unsafe { cuMemAlloc_v2(&mut input, 64) }, Success);
        assert_eq!(unsafe { cuMemcpyHtoDAsync_v2(input, host, 16, s) }, Success);
        assert_eq!(unsafe { cuMemcpyHtoDAsync_v2(table, host, 16, s) }, Success);
        if token > 0 && token % update_every == 0 {
            let update = capture(s, || chain(f, s, weights, 20));
            let mut info = std::mem::MaybeUninit::<ExecUpdateInfo>::zeroed();
            assert_eq!(unsafe { cuGraphExecUpdate_v2(exec, update, info.as_mut_ptr()) }, Success);
            assert_eq!(cuGraphDestroy(update), Success);
        }
        assert_eq!(cuGraphLaunch(exec, s), Success);
        assert_eq!(cuStreamSynchronize(s), Success);
        assert_eq!(cuMemFree_v2(input), Success);
    }
    let stats = stats();
    assert_eq!(stats.recorded, 1 + (tokens - 1) / update_every, "{stats:?}");
    assert_eq!(stats.replayed, tokens, "{stats:?}");
    assert_eq!(stats.reissued, 0, "{stats:?}");
    assert!(stats.kept_after_free >= tokens - stats.recorded, "{stats:?}");
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

/// A capture holding host-to-device copies or host functions keeps re-issuing,
/// and says why.
#[test]
fn unrecordable_nodes_are_named() {
    use crate::cuda::graph_record::unrecordable_kinds;
    let _fixture = Fixture::new();
    let _on = Recording::on();
    let ctx = context(None);
    let (s, f) = (stream(), function(module()));
    let (device, host) = allocations();
    let mut e = null_mut();
    assert_eq!(unsafe { super::events::cuEventCreate(&mut e, 2) }, Success);
    unsafe extern "system" fn nothing(_: *mut c_void) {}
    let graph = capture(s, || {
        assert_eq!(launch(f, s, device), Success);
        assert_eq!(unsafe { cuMemcpyHtoDAsync_v2(device, host, 16, s) }, Success);
        assert_eq!(super::events::cuEventRecord(e, s), Success);
        assert_eq!(unsafe { crate::cuda::graph_api::cuLaunchHostFunc(s, nothing as *mut c_void, null_mut()) }, Success);
    });
    let why = state(|state| Ok(unrecordable_kinds(&state.graphs.graphs[&graph.addr()].flatten()))).unwrap();
    // An event recorded on the capturing stream itself adds no node (only cross-stream joins do).
    assert!(why.as_deref().is_some_and(|why| why.contains("host-to-device copy") && why.contains("host function")), "{why:?}");
    let exec = instantiate(graph);
    for _ in 0..3 {
        assert_eq!(cuGraphLaunch(exec, s), Success);
    }
    assert_eq!(stats(), Stats { reissued: 3, ..Stats::default() });
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuGraphDestroy(graph), Success);
    assert_eq!(super::events::cuEventDestroy_v2(e), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn free_log_overlap_rules() {
    use crate::cuda::graph_record::{FREE_LOG, FreeLog};
    let mut log = FreeLog::default();
    let uses = [(0x1000, 0x100), (0x5000, 0x10)];
    log.push(1, 0x2000, 0x100);
    assert_eq!(log.overlaps_since(0, 1, &uses), Some(false));
    log.push(2, 0x10f0, 0x20);
    assert_eq!(log.overlaps_since(0, 2, &uses), Some(true));
    assert_eq!(log.overlaps_since(1, 2, &uses), Some(true));
    assert_eq!(log.overlaps_since(2, 2, &uses), Some(false), "nothing freed since 2");
    for number in 3..(3 + FREE_LOG as u64) {
        log.push(number, 0x9000, 0x10);
    }
    let now = 2 + FREE_LOG as u64;
    assert_eq!(log.overlaps_since(0, now, &uses), None, "older frees fell out of the log");
    assert_eq!(log.overlaps_since(5, now, &uses), Some(false));
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

// ---- recorded graphs with l0-submit's deferred submission and in-flight window ----

/// A context whose streams use `CUDA_FOR_INTEL_SUBMIT=deferred` (cap `submit`) and/or
/// `CUDA_FOR_INTEL_INFLIGHT=window` (limit `window`).
fn submit_context(submit: Option<usize>, window: Option<u64>) -> Opaque {
    super::deferred::TEST_CONFIG.set(Some(submit.map(|cap| super::deferred::Config { cap })));
    if let Some(limit) = window {
        super::inflight::TEST_CONFIG.set(Some(super::inflight::Config { limit, window: true }));
    }
    let ctx = context(None);
    super::deferred::TEST_CONFIG.set(None);
    super::inflight::TEST_CONFIG.set(None);
    ctx
}

/// (launches buffered in the deferred submit list, window checkpoints) of `stream`.
fn submit_state(stream: Opaque) -> (usize, usize) {
    let key = stream_key(stream).unwrap();
    with_context(|_, context| {
        let stream = context.streams.get(key)?;
        Ok((stream.deferred.as_ref().map_or(0, |d| d.buffered()), stream.window.checkpoints()))
    })
    .unwrap()
}

/// Recorded graphs under deferred submit, the window and both: buffered launches are flushed
/// before the list, window checkpoints stay bounded, and a failed list append fails the stream
/// once (sticky, like every other failed append) without leaking or appending a list twice.
/// Written by the reviewer-2 teammate; the window case fails without the sticky append error.
#[test]
fn recorded_graphs_under_deferred_submit_and_window() {
    for (submit, window) in [(Some(8), None), (None, Some(64)), (Some(4), Some(64))] {
        let _fixture = Fixture::new();
        let ctx = submit_context(submit, window);
        let _rec = Recording::on();
        let (st, f) = (stream(), function(module()));
        let (device, host) = allocations();
        let exec = instantiate(capture(st, || {
            for _ in 0..6 {
                assert_eq!(launch(f, st, device), Success);
            }
            assert_eq!(unsafe { cuMemcpyDtoDAsync_v2(device + 8, device, 8, st) }, Success);
        }));
        for _ in 0..20 {
            for _ in 0..3 {
                assert_eq!(launch(f, st, device), Success);
            }
            assert_eq!(cuGraphLaunch(exec, st), Success, "{submit:?} {window:?}");
            assert_eq!(submit_state(st).0, 0, "buffered launches were flushed before the list");
            assert_eq!(launch(f, st, device), Success);
        }
        assert!(submit_state(st).1 <= 64, "window checkpoints stay bounded");
        assert_eq!(unsafe { cuMemcpyDtoHAsync_v2(host, device, 4, st) }, Success);
        assert_eq!(cuStreamSynchronize(st), Success);
        assert_eq!(stats().recorded, 1);
        assert_eq!(stats().replayed, 20);
        // the immediate append fails: buffered launches (if any) and the list are lost once,
        // the stream is failed, and teardown releases everything
        assert_eq!(launch(f, st, device), Success);
        let before = fake::snapshot().ledger.submitted;
        crate::level_zero::fault::arm("zeCommandListImmediateAppendCommandListsExp", 0x7fff_fffe);
        assert_ne!(cuGraphLaunch(exec, st), Success, "{submit:?} {window:?}");
        assert_eq!(stats().replayed, 20, "a failed append is not counted as a replay");
        assert_ne!(cuGraphLaunch(exec, st), Success, "the stream stays failed {submit:?} {window:?}");
        assert_ne!(cuStreamSynchronize(st), Success);
        let after = fake::snapshot().ledger.submitted;
        assert!(after - before <= 1, "nothing appended twice after the failure: {before} -> {after}");
        assert_eq!(cuGraphExecDestroy(exec), Success);
        assert_ne!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}

fn device_buffer() -> u64 {
    let mut p = 0;
    assert_eq!(unsafe { cuMemAlloc_v2(&mut p, 64) }, Success);
    p
}

/// The buffer each executed kernel launched with (its first allocation), in execution order.
fn executed() -> Vec<u64> {
    fake::take_executed_kernels().into_iter().map(|memory| memory[0] as u64).collect()
}

/// Recorded graphs under `CUDA_FOR_INTEL_SUBMIT=deferred`: launches buffered in the deferred
/// submit list before a graph run before it, and launches after it run after it, on the device,
/// whether the submit list is open, empty or flushed in between (window on and off).
#[test]
fn recorded_graphs_keep_stream_order_under_deferred_submit() {
    for (submit, window) in [(Some(8), None), (Some(4), Some(64)), (None, Some(64))] {
        let _fixture = Fixture::new();
        let ctx = submit_context(submit, window);
        let _rec = Recording::on();
        let (st, f) = (stream(), function(module()));
        let (before, inside, after) = (device_buffer(), device_buffer(), device_buffer());
        let exec = instantiate(capture(st, || {
            for _ in 0..3 {
                assert_eq!(launch(f, st, inside), Success);
            }
        }));
        executed();
        let mut want = Vec::new();
        for round in 0..6 {
            // 0..=5 launches before the graph: an empty, open or just-flushed submit list
            for _ in 0..round {
                assert_eq!(launch(f, st, before), Success);
                want.push(before);
            }
            assert_eq!(cuGraphLaunch(exec, st), Success, "{submit:?} {window:?}");
            want.extend([inside; 3]);
            assert_eq!(launch(f, st, after), Success);
            want.push(after);
        }
        assert_eq!(cuStreamSynchronize(st), Success);
        assert_eq!(executed(), want, "{submit:?} {window:?}");
        assert_eq!((stats().recorded, stats().replayed), (1, 6));
        assert_eq!(cuGraphExecDestroy(exec), Success);
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}

/// `zeKernelSetArgumentValue` calls so far.
fn arg_sets() -> usize {
    fake::snapshot().ledger.arg_sets
}

/// The kernel argument cache (#43) and graph recording: a recording sets every argument even
/// when immediate launches left the same values cached, the immediate launches after it set
/// theirs again, a replay sets nothing, and the device runs the right values in order.
#[test]
fn graph_recording_does_not_trust_the_argument_cache() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let _rec = Recording::on();
    let f = function(module());
    let (p, other) = (device_buffer(), device_buffer());
    let ns = stream();
    assert_eq!(launch(f, ns, p), Success);
    let before = arg_sets();
    assert_eq!(launch(f, ns, p), Success);
    assert_eq!(arg_sets(), before, "an identical relaunch is served from the cache");
    let exec = instantiate(capture(ns, || {
        assert_eq!(launch(f, ns, p), Success);
        assert_eq!(launch(f, ns, other), Success);
        assert_eq!(launch(f, ns, p), Success);
    }));
    executed();
    // Recording: the first launch sets its argument although `p` was cached.
    let before = arg_sets();
    assert_eq!(cuGraphLaunch(exec, ns), Success);
    assert_eq!(arg_sets() - before, 3, "every recorded launch sets its argument");
    let before = arg_sets();
    assert_eq!(launch(f, ns, other), Success);
    assert_eq!(launch(f, ns, other), Success);
    assert_eq!(arg_sets() - before, 1, "set once after the recording, then cached");
    assert_eq!(cuGraphLaunch(exec, ns), Success);
    assert_eq!(arg_sets() - before, 1, "a replay sets nothing");
    assert_eq!(cuStreamSynchronize(ns), Success);
    assert_eq!(executed(), [p, other, p, other, other, p, other, p]);
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuStreamDestroy_v2(ns), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

/// A failed immediate append of a recorded list must fail the stream like any other append: the
/// driver may have taken the list, so the next launch must not append it again unawaited, and
/// teardown must report the error and leak nothing.
#[test]
fn a_failed_recorded_append_fails_the_stream() {
    for (submit, window) in [(None, None), (Some(8), None), (None, Some(64))] {
        let _fixture = Fixture::new();
        let ctx = submit_context(submit, window);
        let _on = Recording::on();
        let (s, f) = (stream(), function(module()));
        let (device, _) = allocations();
        let exec = instantiate(capture(s, || chain(f, s, device, 4)));
        assert_eq!(cuGraphLaunch(exec, s), Success);
        assert_eq!(cuStreamSynchronize(s), Success);
        // nothing buffered: the next immediate append is the recorded list itself
        crate::level_zero::fault::arm("zeCommandListImmediateAppendCommandListsExp", 0x7fff_fffe);
        assert_ne!(cuGraphLaunch(exec, s), Success, "{submit:?} {window:?}");
        assert_eq!(stats().replayed, 1, "{submit:?} {window:?}: a failed append is no replay");
        assert_ne!(cuGraphLaunch(exec, s), Success, "{submit:?} {window:?}: the stream stays failed");
        assert_ne!(launch(f, s, device), Success, "{submit:?} {window:?}: the stream stays failed");
        assert_ne!(cuStreamSynchronize(s), Success, "{submit:?} {window:?}");
        assert_eq!(cuGraphExecDestroy(exec), Success);
        assert_ne!(cuCtxDestroy_v2(ctx), Success, "{submit:?} {window:?}: teardown reports the error");
        assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
        assert_baseline();
    }
}

/// The Ollama token pattern under deferred submission and the window: per-token input frees
/// outside the graph keep the recorded list, and launches around each graph launch stay valid.
#[test]
fn ollama_pattern_under_deferred_submit_keeps_the_list() {
    let _fixture = Fixture::new();
    let ctx = submit_context(Some(4), Some(64));
    let _on = Recording::on();
    let (s, f) = (stream(), function(module()));
    let (weights, host) = allocations();
    let exec = instantiate(capture(s, || chain(f, s, weights, 8)));
    for _ in 0..12 {
        let mut input = 0;
        assert_eq!(unsafe { cuMemAlloc_v2(&mut input, 256) }, Success);
        assert_eq!(unsafe { cuMemcpyHtoDAsync_v2(input, host.cast_const().cast(), 4, s) }, Success);
        assert_eq!(launch(f, s, input), Success);
        assert_eq!(cuGraphLaunch(exec, s), Success);
        assert_eq!(launch(f, s, input), Success);
        assert_eq!(cuStreamSynchronize(s), Success);
        assert_eq!(cuMemFree_v2(input), Success);
    }
    let stats = stats();
    assert_eq!((stats.recorded, stats.replayed, stats.reissued), (1, 12, 0));
    assert!(fake::snapshot().ledger.violations.is_empty(), "{:?}", fake::snapshot().ledger.violations);
    assert_eq!(cuGraphExecDestroy(exec), Success);
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}
