use super::batch::TEST_POLICY;
use super::events::{
    cuEventCreate, cuEventDestroy_v2, cuEventQuery, cuEventRecord, cuEventSynchronize,
    cuStreamWaitEvent,
};
use super::*;
use crate::cuda::*;
use crate::level_zero::{
    fake::{self, Kind},
    fault::{arm, arm_worker, consumed, nth, pause_worker_execute},
};
use std::ptr::{null_mut, without_provenance_mut};
use std::sync::{Barrier, MutexGuard};
use std::thread;

const OOM: u32 = 0x70000002;
const LOST: u32 = 0x70000001;
const FAIL: u32 = 0x7fff_fffe;
const IMAGE: &std::ffi::CStr =
    c".version 4.1 .target sm_52 .address_size 64 .entry k(.param .u64 .ptr .global dst) { ret; }";
pub(super) const BATCH: Option<Policy> = Some(Policy {
    cap: 32,
    delay: None,
    high_resolution: false,
    eager_after_idle: false,
});

pub(super) struct Fixture {
    _serial: MutexGuard<'static, ()>,
}

impl Fixture {
    pub(super) fn new() -> Self {
        let serial = TEST_RUNTIME
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        assert!(
            RUNTIME
                .lock()
                .unwrap()
                .replace(State {
                    runtime: fake::start(),
                    next_handle: 3,
                    stream_owners: HashMap::new(),
                    contexts: HandleMap::default(),
                    incomplete_contexts: Vec::new(),
                    primary: HashMap::new(),
                    libraries: library::Libraries::default(),
                    graphs: graphs::Graphs::default(),
                    lazy_modules: HashMap::new(),
                })
                .is_none()
        );
        assert_eq!(cuInit(0), Success);
        Self { _serial: serial }
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let (state, workers) = {
            let mut runtime = RUNTIME.lock().unwrap();
            let mut state = runtime.take().unwrap();
            let workers: Vec<_> = state
                .contexts
                .values_mut()
                .filter_map(|context| context.streams.stop())
                .collect();
            (state, workers)
        };
        for worker in workers {
            let _ = worker.join();
        }
        CURRENT.with_borrow_mut(Vec::clear);
        TEST_POLICY.set(None);
        println!("native ledger: {:?}", fake::snapshot());
        fake::finish();
        assert!(
            thread::panicking()
                || (state.contexts.is_empty() && state.incomplete_contexts.is_empty())
        );
    }
}

fn create_context(policy: Option<Policy>, device: i32, out: &mut Opaque) -> CuResult {
    TEST_POLICY.set(Some(policy));
    let result = unsafe { cuCtxCreate_v2(out, 0, device) };
    TEST_POLICY.set(None);
    result
}

pub(super) fn context(policy: Option<Policy>) -> Opaque {
    let mut out = null_mut();
    assert_eq!(create_context(policy, 0, &mut out), Success);
    out
}

pub(super) fn stream() -> Opaque {
    let mut out = null_mut();
    assert_eq!(unsafe { cuStreamCreate(&mut out, 1) }, Success);
    out
}

pub(super) fn event() -> Opaque {
    let mut out = null_mut();
    assert_eq!(unsafe { cuEventCreate(&mut out, 2) }, Success);
    out
}

pub(super) fn module() -> Opaque {
    let mut out = null_mut();
    assert_eq!(
        unsafe { cuModuleLoadData(&mut out, IMAGE.as_ptr().cast()) },
        Success
    );
    out
}

pub(super) fn function(module: Opaque) -> Opaque {
    let mut out = null_mut();
    assert_eq!(
        unsafe { cuModuleGetFunction(&mut out, module, c"k".as_ptr()) },
        Success
    );
    out
}

pub(super) fn allocations() -> (u64, Opaque) {
    let (mut device, mut host) = (0, null_mut());
    assert_eq!(unsafe { cuMemAlloc_v2(&mut device, 64) }, Success);
    assert_eq!(unsafe { cuMemHostAlloc(&mut host, 64, 0) }, Success);
    (device, host)
}

pub(super) fn launch(function: Opaque, stream: Opaque, address: u64) -> CuResult {
    let mut arg = std::ptr::from_ref(&address).cast_mut().cast();
    unsafe { cuLaunchKernel(function, 1, 1, 1, 1, 1, 1, 0, stream, &mut arg, null_mut()) }
}

pub(super) fn assert_baseline() {
    let stats = fake::snapshot();
    assert!(stats.live.is_empty(), "{stats:?}");
    assert_eq!(stats.pending, 0, "{stats:?}");
    assert_eq!(stats.recorded, 0, "{stats:?}");
    assert_eq!(stats.waiters, 0, "{stats:?}");
    assert!(stats.ledger.violations.is_empty(), "{stats:?}");
    assert_eq!(stats.ledger.created, stats.ledger.released, "{stats:?}");
    assert_eq!(stats.ledger.submitted, stats.ledger.completed, "{stats:?}");
    assert!(consumed());
}

#[test]
fn fake_cuda_library_loads_in_each_context_and_retires_handles() {
    use crate::cuda::library::*;
    // The one-instruction test kernel is an empty body, which the runtime otherwise rejects
    // as an architecture-guard stub.
    // SAFETY: set before any other thread of this test reads it.
    unsafe { std::env::set_var("CFI_ALLOW_EMPTY_KERNELS", "1") };
    use std::ptr::null;

    let _fixture = Fixture::new();
    let mut library = null_mut();
    assert_eq!(
        unsafe {
            cuLibraryLoadData(
                &mut library,
                IMAGE.as_ptr().cast(),
                null(),
                null(),
                0,
                null(),
                null(),
                0,
            )
        },
        Success
    );
    let mut kernel = null_mut();
    let mut function = null_mut();
    assert_eq!(
        unsafe { cuLibraryGetKernel(&mut kernel, library, c"missing".as_ptr()) },
        NotFound
    );
    assert_eq!(
        unsafe { cuLibraryGetKernel(&mut kernel, library, c"k".as_ptr()) },
        Success
    );
    assert_eq!(
        unsafe { cuKernelGetFunction(&mut function, kernel) },
        InvalidContext
    );

    let first = context(None);
    assert_eq!(
        unsafe { cuKernelGetFunction(&mut function, kernel) },
        Success
    );
    let first_function = function;
    let mut module = null_mut();
    assert_eq!(unsafe { cuLibraryGetModule(&mut module, library) }, Success);
    assert_eq!(cuModuleUnload(module), InvalidHandle);

    let second = context(None);
    assert_eq!(
        unsafe { cuKernelGetFunction(&mut function, kernel) },
        Success
    );
    assert_ne!(first_function, function);
    let (device, _) = allocations();
    assert_eq!(launch(kernel, null_mut(), device), Success);
    // Second launch: served by the context's resolved-kernel cache.
    assert_eq!(launch(kernel, null_mut(), device), Success);
    assert_eq!(cuLibraryUnload(library), Success);
    assert_eq!(
        unsafe { cuKernelGetFunction(&mut function, kernel) },
        InvalidHandle
    );
    assert_eq!(launch(function, null_mut(), device), InvalidHandle);
    // The cached resolution must not outlive the library.
    assert_eq!(launch(kernel, null_mut(), device), InvalidHandle);
    assert_eq!(cuCtxDestroy_v2(second), Success);
    assert_eq!(cuCtxDestroy_v2(first), Success);
    assert_baseline();
}

#[test]
fn fake_multi_device_context_isolation() {
    let _fixture = Fixture::new();
    let mut count = 0;
    assert_eq!(unsafe { cuDeviceGetCount(&mut count) }, Success);
    assert_eq!(count, 2);
    let mut ordinal = -1;
    assert_eq!(unsafe { cuDeviceGet(&mut ordinal, count) }, InvalidDevice);

    for policy in [None, BATCH] {
        let a = context(policy);
        let stream_a = stream();
        let event_a = event();
        let function_a = function(module());
        let (device_a, host_a) = allocations();
        let mut large = 0;
        assert_eq!(unsafe { cuMemAlloc_v2(&mut large, 256) }, Success);
        assert_eq!(launch(function_a, stream_a, device_a), Success);
        assert_eq!(cuEventRecord(event_a, stream_a), Success);

        let mut b = null_mut();
        assert_eq!(create_context(policy, 1, &mut b), Success);
        assert_eq!(unsafe { cuCtxGetDevice(&mut ordinal) }, Success);
        assert_eq!(ordinal, 1);
        let stream_b = stream();
        let function_b = function(module());
        let (device_b, host_b) = allocations();
        assert_eq!(unsafe { cuMemAlloc_v2(&mut large, 256) }, OutOfMemory);
        // CUDA frees by pointer, whichever device's context is current.
        assert_eq!(cuMemFree_v2(device_a), Success);
        assert_eq!(cuMemFreeHost(host_a), InvalidValue);
        assert_eq!(launch(function_a, stream_b, device_b), InvalidHandle);
        assert_eq!(launch(function_b, stream_a, device_b), InvalidHandle);
        assert_eq!(launch(function_b, stream_b, device_a), InvalidValue);
        // Events of another device are waited for on the host.
        assert_eq!(cuStreamWaitEvent(stream_b, event_a, 0), Success);
        assert_eq!(cuStreamSynchronize(stream_a), Success);

        // A can release its native objects while B has submitted work pending.
        fake::pause();
        assert_eq!(
            unsafe { cuMemcpyHtoDAsync_v2(device_b, host_b, 64, stream_b) },
            Success
        );
        with_context(|runtime, context| {
            context
                .streams
                .get_mut(stream_b.addr())?
                .flush(&runtime.api)
        })
        .unwrap();
        assert!(fake::snapshot().pending > 0);
        assert_eq!(cuCtxDestroy_v2(a), Success);
        assert_eq!(fake::snapshot().live.get(&Kind::Context), Some(&1));
        assert_eq!(cuStreamQuery(stream_b), NotReady);
        assert_eq!(cuStreamQuery(stream_a), InvalidHandle);
        fake::complete();
        assert_eq!(launch(function_b, stream_b, device_b), Success);
        assert_eq!(cuCtxSynchronize(), Success);
        assert_eq!(cuCtxDestroy_v2(b), Success);
        assert_baseline();
    }
}

#[test]
fn fake_fault_initialization_and_allocation() {
    let _fixture = Fixture::new();
    let mut out = null_mut();
    arm("zeContextCreate", OOM);
    assert_eq!(create_context(None, 0, &mut out), OutOfMemory);
    assert!(out.is_null());
    assert_baseline();

    for (call, policy) in [
        ("zeCommandListCreateImmediate", None),
        ("zeCommandListCreateImmediate (asynchronous)", None),
        ("zeCommandQueueCreate", BATCH),
    ] {
        arm(call, OOM);
        arm("zeContextDestroy", FAIL);
        assert_eq!(create_context(policy, 0, &mut out), OutOfMemory);
        assert!(out.is_null());
        assert_eq!(fake::snapshot().live.get(&Kind::Context), Some(&1));
        state(|state| {
            assert_eq!(state.incomplete_contexts.len(), 1);
            assert!(state.contexts.is_empty());
            Ok(())
        })
        .unwrap();
        let ctx = context(policy);
        assert_eq!(fake::snapshot().live.get(&Kind::Context), Some(&1));
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }

    for policy in [None, BATCH] {
        let ctx = context(policy);
        let mut device = u64::MAX;
        let mut host = null_mut();
        nth("zeMemAllocDevice", 2, OOM);
        assert_eq!(unsafe { cuMemAlloc_v2(&mut device, 64) }, Success);
        let mut second = u64::MAX;
        assert_eq!(unsafe { cuMemAlloc_v2(&mut second, 64) }, OutOfMemory);
        assert_eq!(second, u64::MAX);
        arm("zeMemAllocHost", OOM);
        assert_eq!(unsafe { cuMemHostAlloc(&mut host, 64, 0) }, OutOfMemory);
        assert!(host.is_null());
        arm("zeMemFree", FAIL);
        assert_eq!(cuMemFree_v2(device), Unknown);
        assert_eq!(fake::snapshot().live.get(&Kind::DeviceAllocation), Some(&1));
        assert_eq!(cuMemFree_v2(device), Success);
        assert_eq!(cuMemFree_v2(device), InvalidValue);
        for _ in 0..32 {
            let (device, host) = allocations();
            assert_eq!(cuMemFree_v2(device), Success);
            assert_eq!(cuMemFreeHost(host), Success);
        }
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}

#[test]
fn fake_fault_module_retry_and_handle_lifetimes() {
    let _fixture = Fixture::new();
    for policy in [None, BATCH] {
        let ctx = context(policy);
        let mut out = null_mut();
        arm("zeModuleCreate", OOM);
        assert_eq!(
            unsafe { cuModuleLoadData(&mut out, IMAGE.as_ptr().cast()) },
            OutOfMemory
        );
        assert!(out.is_null());
        arm("zeModuleBuildLogDestroy", FAIL);
        let m = module();
        assert_eq!(fake::snapshot().live.get(&Kind::BuildLog), Some(&1));
        arm("zeKernelCreate", OOM);
        assert_eq!(
            unsafe { cuModuleGetFunction(&mut out, m, c"k".as_ptr()) },
            OutOfMemory
        );
        assert!(out.is_null());
        let f = function(m);
        let (device, _) = allocations();
        arm("zeModuleDestroy", FAIL);
        assert_eq!(cuModuleUnload(m), Unknown);
        assert_eq!(fake::snapshot().live.get(&Kind::Module), Some(&1));
        assert_eq!(fake::snapshot().live.get(&Kind::Kernel), None);
        assert_eq!(launch(f, null_mut(), device), InvalidHandle);
        assert_eq!(
            unsafe { cuModuleGetFunction(&mut out, m, c"k".as_ptr()) },
            InvalidHandle
        );
        assert!(out.is_null());
        assert_eq!(cuModuleUnload(m), Success);
        assert_eq!(cuModuleUnload(m), InvalidHandle);
        for _ in 0..24 {
            let m = module();
            let _ = function(m);
            assert_eq!(cuModuleUnload(m), Success);
        }
        arm("zeModuleBuildLogDestroy", FAIL);
        assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
        assert_eq!(cuCtxSetCurrent(ctx), Unknown);
        assert_eq!(fake::snapshot().live.get(&Kind::BuildLog), Some(&1));
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}

#[test]
fn fake_fault_event_create_and_stream_teardown() {
    let _fixture = Fixture::new();
    for policy in [None, BATCH] {
        let ctx = context(policy);
        let s = stream();
        let e = event();
        for call in ["zeEventPoolCreate", "zeEventCreate"] {
            arm(call, OOM);
            assert_eq!(cuEventRecord(e, s), OutOfMemory, "{call}");
            assert_eq!(cuEventQuery(e), Success);
        }
        assert_eq!(cuEventRecord(e, s), Success);
        assert_eq!(cuEventSynchronize(e), Success);
        arm("zeEventQueryStatus", FAIL);
        assert_eq!(cuEventQuery(e), Unknown);
        arm("zeEventHostSynchronize", FAIL);
        assert_eq!(cuEventSynchronize(e), Unknown);
        assert_eq!(cuEventSynchronize(e), Success);
        assert_eq!(cuEventDestroy_v2(e), Success);
        assert_eq!(cuEventQuery(e), InvalidHandle);
        let destroy = if policy.is_some() {
            "zeCommandQueueDestroy"
        } else {
            "zeCommandListDestroy"
        };
        arm(destroy, FAIL);
        assert_eq!(cuStreamDestroy_v2(s), Unknown);
        assert_eq!(cuStreamQuery(s), InvalidHandle);
        for _ in 0..64 {
            assert_eq!(cuStreamDestroy_v2(stream()), Success);
        }
        arm("zeEventDestroy", FAIL);
        assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
        assert_eq!(fake::snapshot().live.get(&Kind::Event), Some(&1));
        arm("zeEventPoolDestroy", FAIL);
        assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
        assert_eq!(fake::snapshot().live.get(&Kind::Event), None);
        assert_eq!(fake::snapshot().live.get(&Kind::EventPool), Some(&1));
        arm("zeContextDestroy", FAIL);
        assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
        assert_eq!(fake::snapshot().live.get(&Kind::Context), Some(&1));
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}

#[test]
fn fake_fault_event_pool_recycles_only_unreferenced_events() {
    let _fixture = Fixture::new();
    for policy in [None, BATCH] {
        let ctx = context(policy);
        let s = stream();
        let held = event();
        let e = event();
        assert_eq!(cuEventRecord(held, s), Success);
        for _ in 0..63 {
            assert_eq!(cuEventRecord(e, s), Success);
            assert_eq!(cuStreamSynchronize(s), Success);
            assert_eq!(cuStreamQuery(s), Success);
        }
        assert_eq!(fake::snapshot().live.get(&Kind::Event), Some(&64));
        arm("zeEventHostReset", FAIL);
        assert_eq!(cuEventRecord(e, s), Unknown);
        assert_eq!(cuEventQuery(held), Success);
        assert_eq!(cuEventRecord(e, s), Success);
        assert_eq!(cuEventQuery(held), Success);
        assert_eq!(fake::snapshot().live.get(&Kind::EventPool), Some(&1));
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}

#[test]
fn fake_fault_batch_recording_and_fence_paths() {
    let _fixture = Fixture::new();
    let ctx = context(BATCH);
    let s = stream();
    let e = event();
    for call in ["zeCommandListCreate", "zeFenceCreate"] {
        arm(call, OOM);
        assert_eq!(cuEventRecord(e, s), OutOfMemory, "{call}");
        assert_eq!(fake::snapshot().live.get(&Kind::Fence), None);
    }
    assert_eq!(cuEventRecord(e, s), Success);
    arm("zeFenceHostSynchronize", FAIL);
    assert_eq!(cuStreamSynchronize(s), Unknown);
    assert_eq!(cuStreamSynchronize(s), Success);
    arm("zeFenceQueryStatus", FAIL);
    assert_eq!(cuStreamQuery(s), Unknown);
    assert_eq!(cuStreamQuery(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();

    for failure in ["zeCommandListReset", "zeFenceReset"] {
        let ctx = context(BATCH);
        let s = stream();
        assert_eq!(cuEventRecord(event(), s), Success);
        assert_eq!(cuStreamSynchronize(s), Success);
        arm(failure, FAIL);
        assert_eq!(cuStreamQuery(s), Unknown);
        assert_eq!(cuStreamQuery(s), Unknown);
        arm("zeFenceDestroy", FAIL);
        assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
        assert_eq!(fake::snapshot().live.get(&Kind::Fence), Some(&1));
        assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
        assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
        assert_baseline();
    }

    for (policy, failure) in [
        (None, "zeCommandListAppendLaunchKernel"),
        (BATCH, "zeCommandListAppendLaunchKernel"),
        (BATCH, "zeCommandListClose"),
        (BATCH, "zeCommandQueueExecuteCommandLists"),
    ] {
        let ctx = context(policy);
        let s = stream();
        let f = function(module());
        let (address, _) = allocations();
        assert_eq!(launch(f, s, address), Success);
        let before = fake::snapshot();
        arm(failure, FAIL);
        let status = if failure == "zeCommandListAppendLaunchKernel" {
            launch(f, s, address)
        } else {
            cuStreamSynchronize(s)
        };
        assert_eq!(status, Unknown, "{failure}");
        assert_eq!(cuStreamSynchronize(s), Unknown);
        assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
        let after = fake::snapshot();
        assert_eq!(after.ledger.submitted, before.ledger.submitted);
        assert_eq!(
            after.ledger.discarded_recorded - before.ledger.discarded_recorded,
            before.recorded
        );
        assert_baseline();
    }
}

#[test]
fn fake_fault_accepted_work_and_unconfirmed_cleanup() {
    let _fixture = Fixture::new();
    let ctx = context(BATCH);
    let s = stream();
    let f = function(module());
    let (device, host) = allocations();
    fake::pause();
    assert_eq!(
        unsafe { cuMemcpyHtoDAsync_v2(device, host, 64, s) },
        Success
    );
    assert_eq!(launch(f, s, device), Success);
    fake::fail_after_execute(FAIL);
    assert_eq!(cuStreamSynchronize(s), Unknown);
    let before = fake::snapshot();
    assert_eq!(before.pending, 1);
    // The allocation's zero-fill (one completed submission), then HtoD records a copy plus
    // a destination ranges barrier, then the launch.
    assert_eq!(before.ledger.submitted, 4);
    assert_eq!(before.ledger.completed, 1);
    assert_eq!(cuMemFree_v2(device), Unknown);
    arm("zeCommandQueueSynchronize", 1);
    assert_eq!(cuCtxDestroy_v2(ctx), NotReady);
    let after = fake::snapshot();
    assert_eq!(after.live, before.live);
    assert_eq!(after.ledger.released, before.ledger.released);
    assert_eq!(after.pending, 1);
    assert_eq!(cuCtxSetCurrent(ctx), NotReady);
    fake::complete();
    assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
    assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
    assert_baseline();

    for policy in [None, BATCH] {
        let ctx = context(policy);
        let mut out = 0;
        arm("zeMemAllocDevice", LOST);
        assert_eq!(unsafe { cuMemAlloc_v2(&mut out, 64) }, LaunchFailed);
        assert_eq!(out, 0);
        assert_eq!(cuStreamQuery(null_mut()), LaunchFailed);
        assert_eq!(cuCtxDestroy_v2(ctx), LaunchFailed);
        assert_baseline();
    }
}

#[test]
fn fake_fault_pending_dependencies_and_concurrent_waiters() {
    let _fixture = Fixture::new();
    for policy in [None, BATCH] {
        let ctx = context(policy);
        let producer = stream();
        let consumer = stream();
        let e = event();
        let f = function(module());
        let (device, host) = allocations();
        fake::pause();
        assert_eq!(
            unsafe { cuMemcpyHtoDAsync_v2(device, host, 64, producer) },
            Success
        );
        assert_eq!(cuEventRecord(e, producer), Success);
        assert_eq!(cuStreamWaitEvent(consumer, e, 0), Success);
        assert_eq!(cuStreamDestroy_v2(producer), Success);
        assert_eq!(cuStreamQuery(producer), InvalidHandle);
        assert_eq!(cuEventDestroy_v2(e), Success);
        assert_eq!(cuEventQuery(e), InvalidHandle);

        let (ctx_key, stream_key, f_key) = (ctx.addr(), consumer.addr(), f.addr());
        let start = Barrier::new(4);
        thread::scope(|scope| {
            for _ in 0..4 {
                scope.spawn(|| {
                    assert_eq!(cuCtxSetCurrent(without_provenance_mut(ctx_key)), Success);
                    start.wait();
                    for _ in 0..24 {
                        assert_eq!(
                            launch(
                                without_provenance_mut(f_key),
                                without_provenance_mut(stream_key),
                                device
                            ),
                            Success
                        );
                    }
                });
            }
        });
        assert_eq!(cuStreamQuery(consumer), NotReady);
        thread::scope(|scope| {
            let waiter = scope.spawn(|| cuStreamSynchronize(without_provenance_mut(stream_key)));
            let waiting = fake::until(|s| s.waiters == 1);
            let pending = fake::snapshot();
            let destroy = cuCtxDestroy_v2(ctx);
            fake::complete();
            assert!(waiting);
            assert_eq!(destroy, NotReady);
            assert!(pending.pending > 0);
            assert_eq!(pending.live.get(&Kind::DeviceAllocation), Some(&1));
            assert_eq!(pending.live.get(&Kind::HostAllocation), Some(&1));
            assert_eq!(pending.live.get(&Kind::Kernel), Some(&1));
            assert!(pending.live.contains_key(&Kind::Event));
            assert_eq!(waiter.join().unwrap(), Success);
        });
        assert_eq!(cuStreamDestroy_v2(consumer), Success);
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}

#[test]
fn fake_fault_worker_progress_failure_and_shutdown() {
    let _fixture = Fixture::new();
    let policy = BATCH.map(|p| Policy {
        delay: Some(Duration::from_micros(250)),
        ..p
    });
    let ctx = context(policy);
    let s = stream();
    fake::pause();
    assert_eq!(cuEventRecord(event(), s), Success);
    assert!(fake::until(|s| s.pending == 1));
    arm("zeCommandQueueSynchronize", 1);
    assert_eq!(cuCtxDestroy_v2(ctx), NotReady);
    assert_eq!(fake::snapshot().pending, 1);
    fake::complete();
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();

    let ctx = context(policy);
    let e = event();
    arm_worker("zeCommandQueueExecuteCommandLists", FAIL);
    assert_eq!(cuEventRecord(e, stream()), Success);
    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        match cuEventQuery(e) {
            Unknown => break,
            NotReady if Instant::now() < deadline => thread::yield_now(),
            result => panic!("worker error did not propagate: {result:?}"),
        }
    }
    assert_eq!(cuEventSynchronize(e), Unknown);
    assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
    assert_baseline();

    let ctx = context(policy);
    let (entered, resume) = pause_worker_execute();
    assert_eq!(cuEventRecord(event(), stream()), Success);
    entered.recv_timeout(Duration::from_secs(3)).unwrap();
    assert!(RUNTIME.try_lock().is_err());
    let key = ctx.addr();
    let start = Barrier::new(2);
    thread::scope(|scope| {
        let destroy = scope.spawn(|| {
            start.wait();
            cuCtxDestroy_v2(without_provenance_mut(key))
        });
        start.wait();
        resume.send(()).unwrap();
        assert_eq!(destroy.join().unwrap(), Success);
    });
    assert_baseline();

    let ctx = context(policy);
    arm_worker("batch worker shutdown", FAIL);
    assert_eq!(cuCtxDestroy_v2(ctx), OperatingSystem);
    assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
    assert_baseline();
}

#[test]
fn fake_primary_context_references_reset_and_native_failures() {
    use crate::cuda::primary::*;
    let _fixture = Fixture::new();
    for policy in [None, BATCH] {
        TEST_POLICY.set(Some(policy));
        let primary_state = || {
            let (mut flags, mut active) = (u32::MAX, -1);
            assert_eq!(
                unsafe { cuDevicePrimaryCtxGetState(0, &mut flags, &mut active) },
                Success
            );
            (flags, active)
        };
        assert_eq!(primary_state(), (0, 0));
        assert_eq!(cuDevicePrimaryCtxSetFlags_v2(0, 3), InvalidValue);
        assert_eq!(cuDevicePrimaryCtxSetFlags_v2(0, 1), NotSupported);
        assert_eq!(cuDevicePrimaryCtxSetFlags_v2(0, 0), Success);
        assert_eq!(cuDevicePrimaryCtxRelease_v2(0), InvalidContext);
        let mut primary = null_mut();
        arm("zeContextCreate", OOM);
        assert_eq!(
            unsafe { cuDevicePrimaryCtxRetain(&mut primary, 0) },
            OutOfMemory
        );
        assert!(primary.is_null());
        assert_eq!(primary_state(), (0, 0));
        assert_eq!(
            unsafe { cuDevicePrimaryCtxRetain(&mut primary, 0) },
            Success
        );
        assert_eq!(current(), None);
        assert_eq!(cuCtxSetCurrent(primary), Success);
        assert_eq!(cuCtxDestroy_v2(primary), InvalidContext);
        assert_eq!(cuDevicePrimaryCtxSetFlags(0, 0), PrimaryContextActive);
        let primary_key = primary.addr();
        thread::spawn(move || {
            let mut same = null_mut();
            assert_eq!(unsafe { cuDevicePrimaryCtxRetain(&mut same, 0) }, Success);
            assert_eq!(same.addr(), primary_key);
            assert_eq!(current(), None);
            assert_eq!(cuDevicePrimaryCtxRelease_v2(0), Success);
        })
        .join()
        .unwrap();
        assert_eq!(primary_state(), (0, 1));
        let mut pinned = null_mut();
        assert_eq!(unsafe { cuMemHostAlloc(&mut pinned, 64, 0) }, Success);
        arm("zeContextDestroy", FAIL);
        assert_eq!(cuDevicePrimaryCtxReset_v2(0), Unknown);
        assert_eq!(primary_state(), (0, 1));
        let mut unavailable = null_mut();
        assert_eq!(
            unsafe { cuDevicePrimaryCtxRetain(&mut unavailable, 0) },
            Unknown
        );
        assert_eq!(cuDevicePrimaryCtxReset_v2(0), Success);
        assert_eq!(current(), Some(primary_key));
        assert_eq!(cuCtxSynchronize(), ContextIsDestroyed);
        assert_eq!(primary_state(), (0, 0));
        let mut replacement = null_mut();
        assert_eq!(
            unsafe { cuDevicePrimaryCtxRetain(&mut replacement, 0) },
            Success
        );
        assert_ne!(replacement, primary);
        assert_eq!(cuDevicePrimaryCtxRelease_v2(0), Success);
        assert_eq!(primary_state(), (0, 1));
        assert_eq!(cuDevicePrimaryCtxRelease_v2(0), Success);
        assert_eq!(cuDevicePrimaryCtxRelease_v2(0), InvalidContext);
        assert_eq!(cuCtxSetCurrent(null_mut()), Success);
        TEST_POLICY.set(None);
        assert_baseline();
    }
}

#[test]
fn fake_multi_entry_kernel_create_and_unload_retry() {
    let _fixture = Fixture::new();
    for policy in [None, BATCH] {
        let ctx = context(policy);
        let image = c".version 4.1 .target sm_52 .address_size 64 .entry first() { ret; } .entry second() { ret; }";
        let (mut module, mut first, mut second) = (null_mut(), null_mut(), null_mut());
        assert_eq!(
            unsafe { cuModuleLoadData(&mut module, image.as_ptr().cast()) },
            Success
        );
        let lookup = |out: &mut Opaque, name: &CStr| unsafe {
            cuModuleGetFunction(out, module, name.as_ptr())
        };
        assert_eq!(lookup(&mut first, c"first"), Success);
        arm("zeKernelCreate", OOM);
        assert_eq!(lookup(&mut second, c"second"), OutOfMemory);
        assert!(second.is_null());
        assert_eq!(lookup(&mut second, c"second"), Success);
        assert_ne!(first, second);
        assert_eq!(
            unsafe {
                cuLaunchKernel(
                    first,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    null_mut(),
                    null_mut(),
                    null_mut(),
                )
            },
            Success
        );
        nth("zeKernelDestroy", 2, FAIL);
        assert_eq!(cuModuleUnload(module), Unknown);
        assert_eq!(lookup(&mut first, c"first"), InvalidHandle);
        assert_eq!(
            unsafe {
                cuLaunchKernel(
                    second,
                    1,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    null_mut(),
                    null_mut(),
                    null_mut(),
                )
            },
            InvalidHandle
        );
        assert_eq!(cuModuleUnload(module), Success);
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}
