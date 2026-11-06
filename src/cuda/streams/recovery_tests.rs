use super::batch_tests::{POLICY, high_resolution, launch, new_stream, noop, open, policy};
use super::events::{
    cuEventCreate, cuEventDestroy_v2, cuEventQuery, cuEventRecord, cuEventSynchronize,
    cuStreamWaitEvent,
};
use super::*;
use crate::cuda::{cuCtxCreate_v2, cuCtxDestroy_v2, cuCtxSetCurrent, cuInit, cuLaunchKernel};
use crate::level_zero::fault::{arm, arm_worker, consumed, pause_worker_execute};
use std::ptr::{null_mut, without_provenance_mut};
use std::thread;

const OOM: u32 = 0x70000002;
const DEVICE_LOST: u32 = 0x70000001;
const UNKNOWN: u32 = 0x7fff_fffe;

fn new_event() -> Opaque {
    let mut event = null_mut();
    assert_eq!(unsafe { cuEventCreate(&mut event, 2) }, Success);
    event
}

fn buffered() -> Policy {
    Policy {
        delay: None,
        ..policy(32, 250)
    }
}

#[test]
#[ignore = "uses a real Level Zero context with deterministic pre-call failures"]
fn gated_recovery_initialization_and_destruction_retry() {
    assert_eq!(cuInit(0), Success);
    for (call, policy) in [
        ("zeCommandListCreateImmediate", None),
        ("zeCommandListCreateImmediate (asynchronous)", None),
        ("zeCommandQueueCreate", Some(high_resolution(250))),
    ] {
        POLICY.set(Some(policy));
        arm(call, OOM);
        arm("zeContextDestroy", UNKNOWN);
        let mut out = null_mut();
        assert_eq!(unsafe { cuCtxCreate_v2(&mut out, 0, 0) }, OutOfMemory);
        POLICY.set(None);
        assert!(out.is_null());
        assert!(consumed());
        state(|state| {
            assert_eq!(state.incomplete_contexts.len(), 1);
            Ok(())
        })
        .unwrap();
        let ctx = open(policy);
        state(|state| {
            assert!(state.incomplete_contexts.is_empty());
            Ok(())
        })
        .unwrap();

        let stream = new_stream(1);
        arm(
            if policy.is_some() {
                "zeCommandQueueDestroy"
            } else {
                "zeCommandListDestroy"
            },
            UNKNOWN,
        );
        assert_eq!(cuStreamDestroy_v2(stream), Unknown);
        assert!(consumed());
        assert_eq!(cuStreamQuery(stream), InvalidHandle);
        assert_eq!(cuStreamSynchronize(new_stream(1)), Success);

        arm("zeContextDestroy", UNKNOWN);
        assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
        assert!(consumed());
        assert_eq!(cuStreamQuery(null_mut()), Unknown);
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
    }
}

#[test]
#[ignore = "uses real Level Zero lists and fences with deterministic pre-call failures"]
fn gated_recovery_create_append_submit_and_recycle() {
    let ctx = open(Some(buffered()));
    let stream = new_stream(1);
    let event = new_event();
    for call in ["zeCommandListCreate", "zeFenceCreate"] {
        arm(call, OOM);
        assert_eq!(cuEventRecord(event, stream), OutOfMemory);
        assert!(consumed());
        assert_eq!(cuEventQuery(event), Success);
    }
    assert_eq!(cuEventRecord(event, stream), Success);
    assert_eq!(cuEventSynchronize(event), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);

    for call in [
        "zeCommandListAppendBarrier",
        "zeCommandListClose",
        "zeCommandQueueExecuteCommandLists",
        "zeCommandListReset",
        "zeFenceReset",
    ] {
        let ctx = open(Some(buffered()));
        let stream = new_stream(1);
        let event = new_event();
        let queued = new_event();
        let unrelated = new_stream(1);
        // A previously successful recording remains valid. The second is
        // accepted into the open list before the injected failure.
        assert_eq!(cuEventRecord(event, stream), Success);
        assert_eq!(cuStreamSynchronize(stream), Success);
        assert_eq!(cuStreamQuery(stream), Success);
        assert_eq!(cuEventRecord(queued, stream), Success);
        arm(call, UNKNOWN);
        let result = match call {
            "zeCommandListAppendBarrier" => cuEventRecord(event, stream),
            "zeCommandListReset" | "zeFenceReset" => {
                // Synchronize captures and waits for the native fence without
                // recycling it. The subsequent query reaches the reset hook.
                assert_eq!(cuStreamSynchronize(stream), Success);
                cuStreamQuery(stream)
            }
            _ => cuStreamSynchronize(stream),
        };
        assert_eq!(result, Unknown, "{call}");
        assert!(consumed(), "{call}");
        assert_eq!(cuEventSynchronize(queued), Unknown);
        assert_eq!(cuStreamQuery(stream), Unknown);
        assert_eq!(cuStreamSynchronize(unrelated), Success);
        // Successful native cleanup still reports the async operation error.
        assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
        assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
    }
}

#[test]
#[ignore = "uses real Level Zero lists and fences to test partial cleanup"]
fn gated_recovery_partial_slot_cleanup() {
    let ctx = open(Some(buffered()));
    let stream = new_stream(1);
    let event = new_event();
    arm("zeFenceCreate", OOM);
    arm("zeCommandListDestroy", UNKNOWN);
    assert_eq!(cuEventRecord(event, stream), OutOfMemory);
    assert_eq!(cuStreamQuery(stream), Unknown);
    assert!(consumed());
    assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
    assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);

    let ctx = open(Some(buffered()));
    let stream = new_stream(1);
    let event = new_event();
    assert_eq!(cuEventRecord(event, stream), Success);
    arm("zeFenceDestroy", UNKNOWN);
    assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
    assert!(consumed());
    // Retry after the list was destroyed and only its fence is still owned.
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
}

#[test]
#[ignore = "uses real Level Zero immediate lists to test append failure"]
fn gated_recovery_immediate_append() {
    let ctx = open(None);
    let stream = new_stream(1);
    let event = new_event();
    let kernel = noop();
    launch(kernel, stream);
    arm("zeCommandListAppendBarrier", UNKNOWN);
    assert_eq!(cuEventRecord(event, stream), Unknown);
    assert!(consumed());
    assert_eq!(cuEventQuery(event), Success);
    assert_eq!(cuStreamSynchronize(stream), Unknown);
    assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
    assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
}

#[test]
#[ignore = "uses real Level Zero queues to test waiter and teardown errors"]
fn gated_recovery_waiter_and_ambiguous_execute() {
    let ctx = open(Some(buffered()));
    let stream = new_stream(1);
    let event = new_event();
    assert_eq!(cuEventRecord(event, stream), Success);
    let key = stream.addr();
    let result = thread::spawn(move || {
        arm("zeFenceHostSynchronize", UNKNOWN);
        let result = cuStreamSynchronize(without_provenance_mut(key));
        assert!(consumed());
        result
    })
    .join()
    .unwrap();
    assert_eq!(result, Unknown);
    assert_eq!(cuStreamSynchronize(stream), Success);
    assert_eq!(cuEventSynchronize(event), Success);

    assert_eq!(cuEventRecord(event, stream), Success);
    arm("zeCommandQueueExecuteCommandLists", DEVICE_LOST);
    assert_eq!(cuStreamSynchronize(stream), LaunchFailed);
    assert_eq!(cuStreamQuery(null_mut()), LaunchFailed);
    assert!(consumed());
    arm("zeCommandQueueSynchronize", DEVICE_LOST);
    assert_eq!(cuCtxDestroy_v2(ctx), LaunchFailed);
    assert!(consumed());
    // The handle and list are still owned until the native queue can be drained.
    state(|state| {
        assert!(state.contexts.contains_key(&ctx.addr()));
        Ok(())
    })
    .unwrap();
    assert_eq!(cuCtxDestroy_v2(ctx), LaunchFailed);
    assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
}

#[test]
#[ignore = "uses a real Level Zero worker and native queue"]
fn gated_recovery_worker_error_and_concurrent_teardown() {
    let ctx = open(Some(high_resolution(250)));
    let stream = new_stream(1);
    let event = new_event();
    arm_worker("zeCommandQueueExecuteCommandLists", UNKNOWN);
    assert_eq!(cuEventRecord(event, stream), Success);
    let deadline = Instant::now() + Duration::from_secs(5);
    loop {
        match cuEventQuery(event) {
            NotReady if Instant::now() < deadline => thread::sleep(Duration::from_millis(1)),
            Unknown => break,
            other => panic!("worker failed to publish submission error: {other:?}"),
        }
    }
    assert_eq!(cuEventSynchronize(event), Unknown);
    assert_eq!(cuCtxDestroy_v2(ctx), Unknown);
    assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);

    let ctx = open(Some(high_resolution(250)));
    let stream = new_stream(1);
    let event = new_event();
    let function = noop().addr();
    let (entered, resume) = pause_worker_execute();
    assert_eq!(cuEventRecord(event, stream), Success);
    entered.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(crate::cuda::RUNTIME.try_lock().is_err());

    let (owner, stream_key) = (ctx.addr(), stream.addr());
    let destroyed = race_teardown(resume, owner, stream_key, function);
    assert_eq!(destroyed, Success);
    assert_eq!(cuStreamQuery(stream), InvalidHandle);
    assert_eq!(cuCtxSetCurrent(null_mut()), Success);

    let ctx = open(Some(high_resolution(250)));
    arm_worker("batch worker shutdown", UNKNOWN);
    assert_eq!(cuCtxDestroy_v2(ctx), OperatingSystem);
    assert_eq!(cuCtxDestroy_v2(ctx), InvalidContext);
}

fn race_teardown(
    resume: std::sync::mpsc::Sender<()>,
    owner: usize,
    stream_key: usize,
    function: usize,
) -> CuResult {
    let start = std::sync::Barrier::new(3);
    thread::scope(|scope| {
        let destroy = scope.spawn(|| {
            start.wait();
            cuCtxDestroy_v2(without_provenance_mut(owner))
        });
        let caller = scope.spawn(|| {
            start.wait();
            match cuCtxSetCurrent(without_provenance_mut(owner)) {
                InvalidContext => {}
                Success => {
                    let result = unsafe {
                        cuLaunchKernel(
                            without_provenance_mut(function),
                            1,
                            1,
                            1,
                            1,
                            1,
                            1,
                            0,
                            without_provenance_mut(stream_key),
                            null_mut(),
                            null_mut(),
                        )
                    };
                    assert!(matches!(result, Success | ContextIsDestroyed), "{result:?}");
                }
                other => panic!("unexpected context selection: {other:?}"),
            }
        });
        start.wait();
        resume.send(()).unwrap();
        caller.join().unwrap();
        destroy.join().unwrap()
    })
}

#[test]
#[ignore = "uses real Level Zero events and queues"]
fn gated_recovery_buffered_and_retired_dependencies() {
    for policy in [None, Some(buffered()), Some(high_resolution(250))] {
        for _ in 0..8 {
            let ctx = open(policy);
            let producer = new_stream(1);
            let consumer = new_stream(1);
            let event = new_event();
            let kernel = noop();
            launch(kernel, producer);
            assert_eq!(cuEventRecord(event, producer), Success);
            assert_eq!(cuStreamDestroy_v2(producer), Success);
            assert_eq!(cuStreamWaitEvent(consumer, event, 0), Success);
            assert_eq!(cuEventDestroy_v2(event), Success);
            launch(kernel, consumer);
            assert_eq!(cuCtxDestroy_v2(ctx), Success);
        }
    }
}
