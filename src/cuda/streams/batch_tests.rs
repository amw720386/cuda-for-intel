use super::events::{
    cuEventCreate, cuEventDestroy_v2, cuEventQuery, cuEventRecord, cuEventSynchronize,
    cuStreamWaitEvent,
};
use super::profiling::{arg, run_bench_pipeline};
use super::*;
use crate::cuda::*;
use std::cell::Cell;
use std::hint::black_box;
use std::ptr::{null_mut, without_provenance_mut};
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

thread_local! {
    pub(super) static POLICY: Cell<Option<Option<Policy>>> = const { Cell::new(None) };
}

fn open(policy: Option<Policy>) -> Handle {
    assert_eq!(cuInit(0), Success);
    let mut ctx = null_mut();
    POLICY.set(Some(policy));
    let result = unsafe { cuCtxCreate_v2(&mut ctx, 0, 0) };
    POLICY.set(None);
    assert_eq!(result, Success);
    ctx
}

type Handle = Opaque;

fn new_stream(flags: u32) -> Handle {
    let mut stream = null_mut();
    assert_eq!(unsafe { cuStreamCreate(&mut stream, flags) }, Success);
    stream
}

fn kernel(ptx: &CStr, name: &CStr) -> Handle {
    let (mut module, mut kernel) = (null_mut(), null_mut());
    assert_eq!(
        unsafe { cuModuleLoadData(&mut module, ptx.as_ptr().cast()) },
        Success
    );
    assert_eq!(
        unsafe { cuModuleGetFunction(&mut kernel, module, name.as_ptr()) },
        Success
    );
    kernel
}

fn noop() -> Handle {
    kernel(
        c".version 4.1 .target sm_52 .address_size 64 .entry noop() { ret; }",
        c"noop",
    )
}

fn launch(function: Handle, stream: Handle) {
    assert_eq!(
        unsafe {
            cuLaunchKernel(
                function,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                stream,
                null_mut(),
                null_mut(),
            )
        },
        Success
    );
}

const fn policy(cap: usize, micros: u64) -> Policy {
    Policy {
        cap,
        delay: Some(Duration::from_micros(micros)),
    }
}

#[test]
#[ignore = "requires an Intel Level Zero GPU"]
fn gated_batch_buffered_queries_and_flush_triggers() {
    // Disable the worker only in this test to keep the pre-submit observations
    // deterministic. Each operation below explicitly exercises its flush path.
    let ctx = open(Some(Policy {
        cap: 1024,
        delay: None,
    }));
    let stream = new_stream(1);
    let consumer = new_stream(1);
    let kernel = noop();
    launch(kernel, stream);
    assert_eq!(cuStreamQuery(stream), NotReady);
    let mut event = null_mut();
    assert_eq!(unsafe { cuEventCreate(&mut event, 2) }, Success);
    assert_eq!(cuEventRecord(event, stream), Success);
    assert_eq!(cuEventQuery(event), NotReady);
    assert_eq!(cuStreamWaitEvent(consumer, event, 0), Success);
    with_context(|_, ctx| {
        assert!(
            !ctx.streams
                .get(stream.addr())?
                .batch
                .as_ref()
                .unwrap()
                .buffered()
        );
        Ok(())
    })
    .unwrap();
    assert_eq!(cuStreamQuery(consumer), NotReady);
    assert_eq!(cuEventRecord(event, consumer), Success);
    assert_eq!(cuEventSynchronize(event), Success);
    assert_eq!(cuEventQuery(event), Success);
    launch(kernel, stream);
    assert_eq!(cuStreamSynchronize(stream), Success);
    assert_eq!(cuStreamQuery(stream), Success);
    for _ in 0..1024 {
        launch(kernel, stream);
    }
    with_context(|_, ctx| {
        let batch = ctx.streams.get(stream.addr())?.batch.as_ref().unwrap();
        assert!(!batch.buffered());
        assert_eq!(batch.stats.submitted, 3);
        Ok(())
    })
    .unwrap();
    launch(kernel, stream);
    assert_eq!(cuCtxSynchronize(), Success);
    assert_eq!(cuStreamQuery(stream), Success);
    launch(kernel, stream);
    assert_eq!(cuStreamDestroy_v2(stream), Success);
    assert_eq!(cuStreamQuery(stream), InvalidHandle);
    assert_eq!(cuEventDestroy_v2(event), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}

fn observe_kernel(stream: Handle) {
    // This is the event signaled by the launch itself on a blocking stream.
    // Inspecting it here cannot submit a list. The subsequent wait is a native
    // Level Zero wait with no CUDA polling, markers, or synchronization.
    let (event, wait, _context_guard) = with_context(|runtime, ctx| {
        Ok((
            ctx.streams.get(stream.addr())?.tail().unwrap(),
            runtime.api.event_wait_fn(),
            Arc::clone(&ctx.streams.event_waiters),
        ))
    })
    .unwrap();
    event.wait(wait, 10_000_000_000).unwrap();
}

#[test]
#[ignore = "requires an Intel Level Zero GPU"]
fn gated_batch_isolated_launch_and_concurrent_recording() {
    let ctx = open(Some(policy(16, 250)));
    let stream = new_stream(0);
    let kernel = noop();
    launch(kernel, stream);
    observe_kernel(stream);
    with_context(|_, ctx| {
        let batch = ctx.streams.get(stream.addr())?.batch.as_ref().unwrap();
        assert_eq!(batch.stats.background, 1);
        assert_eq!(batch.stats.submitted, 1);
        Ok(())
    })
    .unwrap();

    // Submit concurrently to the same stream while the context worker may
    // close earlier lists. The batch count also exercises native reuse.
    let ctx_key = ctx.addr();
    let stream_key = stream.addr();
    let kernel_key = kernel.addr();
    let start = Barrier::new(5);
    thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                assert_eq!(cuCtxSetCurrent(without_provenance_mut(ctx_key)), Success);
                start.wait();
                for _ in 0..257 {
                    launch(
                        without_provenance_mut(kernel_key),
                        without_provenance_mut(stream_key),
                    );
                }
                assert_eq!(
                    cuStreamSynchronize(without_provenance_mut(stream_key)),
                    Success
                );
            });
        }
        start.wait();
    });
    assert_eq!(cuCtxSynchronize(), Success);
    with_context(|_, ctx| {
        let batch = ctx.streams.get(stream_key)?.batch.as_ref().unwrap();
        assert!(batch.stats.submitted > (4usize * 257).div_ceil(16));
        assert!(batch.stats.recycled > 0);
        Ok(())
    })
    .unwrap();
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}

#[derive(Clone, Copy)]
struct Case(&'static str, Option<Policy>);

const CASES: [Case; 6] = [
    Case("A1", None),
    Case("C16/50", Some(policy(16, 50))),
    Case("C16/250", Some(policy(16, 250))),
    Case("C32/250", Some(policy(32, 250))),
    Case("C32/1000", Some(policy(32, 1000))),
    Case("A2", None),
];

const EXPLICIT: Case = Case(
    "B32-explicit-control",
    Some(Policy {
        cap: 32,
        delay: None,
    }),
);

fn report(label: &str, values: &mut [Duration]) {
    values.sort_unstable();
    let p = |percent: usize| {
        let index = (percent * values.len()).div_ceil(100) - 1;
        values[index].as_secs_f64() * 1e6
    };
    println!(
        "{label}: n={} p50={:.3} us p95={:.3} us p99={:.3} us",
        values.len(),
        p(50),
        p(95),
        p(99)
    );
}

#[test]
#[ignore = "requires an Intel Level Zero GPU"]
fn gated_m9_throughput() {
    for Case(label, policy) in CASES.into_iter().chain([EXPLICIT]) {
        let ctx = open(policy);
        let stream = new_stream(1);
        let noop = noop();
        let ptx = CString::new(include_str!("../../../examples/rmsnorm.ptx")).unwrap();
        let rms = kernel(&ptx, c"rmsnorm");
        let mut buffers = [0u64; 3];
        let data = [1.0f32; 1024];
        for address in &mut buffers {
            assert_eq!(
                unsafe { cuMemAlloc_v2(address, size_of_val(&data)) },
                Success
            );
            assert_eq!(
                unsafe { cuMemcpyHtoD_v2(*address, data.as_ptr().cast(), size_of_val(&data)) },
                Success
            );
        }
        let [x, w, y] = buffers;
        let mut args = [arg(&x), arg(&w), arg(&y), arg(&1024u32), arg(&1e-5f32)];
        for (workload, ops) in [
            ("noop", 1),
            ("noop", 10),
            ("noop", 100),
            ("noop", 1000),
            ("rmsnorm", 1),
            ("rmsnorm", 32),
        ] {
            let mut enqueue = Vec::new();
            let mut complete = Vec::new();
            for i in 0..42 {
                let start = Instant::now();
                for _ in 0..ops {
                    if workload == "noop" {
                        launch(noop, stream);
                    } else {
                        assert_eq!(
                            unsafe {
                                cuLaunchKernel(
                                    rms,
                                    1,
                                    1,
                                    1,
                                    128,
                                    1,
                                    1,
                                    0,
                                    stream,
                                    args.as_mut_ptr(),
                                    null_mut(),
                                )
                            },
                            Success
                        );
                    }
                }
                let submit = start.elapsed();
                assert_eq!(cuStreamSynchronize(stream), Success);
                if i >= 7 {
                    enqueue.push(submit);
                    complete.push(start.elapsed());
                }
            }
            report(&format!("{label} {workload} {ops} enqueue"), &mut enqueue);
            report(&format!("{label} {workload} {ops} complete"), &mut complete);
        }
        with_context(|_, ctx| {
            if let Some(batch) = &ctx.streams.get(stream.addr())?.batch {
                let stats = &batch.stats;
                println!(
                    "{label} submitted={} background={} recycled={} mean native reset={:.3} us",
                    stats.submitted,
                    stats.background,
                    stats.recycled,
                    stats.recycle_time.as_secs_f64() * 1e6 / stats.recycled as f64
                );
            }
            Ok(())
        })
        .unwrap();
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
    }
}

#[test]
#[ignore = "requires an Intel Level Zero GPU"]
fn gated_m9_forge() {
    for Case(label, policy) in CASES.into_iter().chain([EXPLICIT]) {
        println!("backend {label}");
        POLICY.set(Some(policy));
        run_bench_pipeline();
        POLICY.set(None);
    }
}

#[test]
#[ignore = "requires an Intel Level Zero GPU and exercises CPU saturation"]
fn gated_m9_isolated_latency_under_load() {
    let cores = thread::available_parallelism().unwrap().get();
    for (label, cpu_workers, cuda_workers) in [
        ("idle", 0, 0),
        ("moderate", (cores / 2).max(1), 0),
        ("heavy", cores, 0),
        ("CUDA-callers", 0, 4),
    ] {
        for Case(case, policy) in CASES {
            let ctx = open(policy);
            let stream = new_stream(0);
            let query_stream = new_stream(1).addr();
            let kernel = noop();
            let stop = AtomicBool::new(false);
            let start = Barrier::new(1 + cpu_workers + cuda_workers);
            let (mut enqueue, mut complete, mut delay) = (Vec::new(), Vec::new(), Vec::new());
            thread::scope(|scope| {
                for _ in 0..cpu_workers {
                    scope.spawn(|| {
                        start.wait();
                        let mut work = 1u64;
                        while !stop.load(Ordering::Relaxed) {
                            for _ in 0..2048 {
                                work = black_box(
                                    work.wrapping_mul(6364136223846793005).wrapping_add(1),
                                );
                            }
                        }
                    });
                }
                for _ in 0..cuda_workers {
                    scope.spawn(|| {
                        start.wait();
                        while !stop.load(Ordering::Relaxed) {
                            let result = cuStreamQuery(without_provenance_mut(query_stream));
                            if result != Success {
                                stop.store(true, Ordering::Relaxed);
                                panic!("unrelated stream query: {result:?}");
                            }
                            thread::yield_now();
                        }
                    });
                }
                struct Stop<'a>(&'a AtomicBool);
                impl Drop for Stop<'_> {
                    fn drop(&mut self) {
                        self.0.store(true, Ordering::Relaxed);
                    }
                }
                let _stop = Stop(&stop);
                start.wait();
                for i in 0..520 {
                    let began = Instant::now();
                    launch(kernel, stream);
                    let api = began.elapsed();
                    observe_kernel(stream);
                    let done = began.elapsed();
                    if i >= 20 {
                        enqueue.push(api);
                        complete.push(done);
                    }
                    // Safe reset after observed completion. Keep recycling out
                    // of the isolated-launch measurement for every backend.
                    assert_eq!(cuStreamSynchronize(stream), Success);
                    assert_eq!(cuStreamQuery(stream), Success);
                    if policy.is_some() {
                        let measured = with_context(|_, ctx| {
                            let stats = &ctx
                                .streams
                                .get(stream.addr())?
                                .batch
                                .as_ref()
                                .unwrap()
                                .stats;
                            assert_eq!(stats.background, i + 1);
                            Ok(stats.last_delay)
                        })
                        .unwrap();
                        if i >= 20 {
                            delay.push(measured);
                        }
                    }
                }
            });
            report(&format!("{label} {case} enqueue"), &mut enqueue);
            report(&format!("{label} {case} complete"), &mut complete);
            if !delay.is_empty() {
                report(&format!("{label} {case} background submit"), &mut delay);
            }
            assert_eq!(cuCtxDestroy_v2(ctx), Success);
        }
    }
}

#[test]
#[ignore = "requires an Intel Level Zero GPU"]
fn gated_batch_sync_captures_prefix_while_more_work_arrives() {
    let ctx = open(Some(policy(16, 250)));
    let stream = new_stream(1);
    let kernel = noop();
    let gates = with_context(|runtime, ctx| {
        let api = &runtime.api;
        let first = ctx.streams.next_fence(api, &ctx.backend)?;
        let second = ctx.streams.next_fence(api, &ctx.backend)?;
        ctx.streams.get_mut(stream.addr())?.record(
            api,
            &ctx.backend,
            None,
            Vec::new(),
            |list, _, _| list.gate(api, &first),
        )?;
        Ok([first, second])
    })
    .unwrap();
    launch(kernel, stream);
    let key = stream.addr();
    let (send, receive) = std::sync::mpsc::channel();
    thread::scope(|scope| {
        scope.spawn(move || {
            send.send(cuStreamSynchronize(without_provenance_mut(key)))
                .unwrap()
        });
        let deadline = Instant::now() + Duration::from_secs(10);
        let captured = loop {
            if with_context(|_, ctx| Ok(ctx.streams.has_waiters())).unwrap() {
                break true;
            }
            if Instant::now() >= deadline {
                break false;
            }
            thread::sleep(Duration::from_millis(1));
        };
        let record = with_context(|runtime, ctx| {
            let api = &runtime.api;
            let result = ctx.streams.get_mut(key)?.record(
                api,
                &ctx.backend,
                None,
                Vec::new(),
                |list, _, _| list.gate(api, &gates[1]),
            );
            gates[0].host_signal(api).map_err(backend_error)?;
            Ok(result)
        })
        .unwrap();
        let prefix = receive.recv_timeout(Duration::from_secs(10));
        let later = cuStreamQuery(stream);
        with_context(|runtime, _| gates[1].host_signal(&runtime.api).map_err(backend_error))
            .unwrap();
        assert!(captured);
        assert_eq!(record, Ok(()));
        assert_eq!(prefix, Ok(Success));
        assert_eq!(later, NotReady);
    });
    assert_eq!(cuStreamSynchronize(stream), Success);
    drop(gates);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}
