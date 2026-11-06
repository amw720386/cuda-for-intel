use super::events::{
    cuEventCreate, cuEventDestroy_v2, cuEventQuery, cuEventRecord, cuEventSynchronize,
    cuStreamWaitEvent,
};
use super::profiling::{arg, run_bench_pipeline};
use super::*;
use crate::cuda::*;
use std::hint::black_box;
use std::ptr::{null_mut, without_provenance_mut};
use std::sync::Barrier;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::{Duration, Instant};

pub(super) use super::batch::TEST_POLICY as POLICY;

pub(super) fn open(policy: Option<Policy>) -> Handle {
    assert_eq!(cuInit(0), Success);
    let mut ctx = null_mut();
    POLICY.set(Some(policy));
    let result = unsafe { cuCtxCreate_v2(&mut ctx, 0, test_device()) };
    POLICY.set(None);
    assert_eq!(result, Success);
    ctx
}

type Handle = Opaque;

pub(super) fn new_stream(flags: u32) -> Handle {
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

pub(super) fn noop() -> Handle {
    kernel(
        c".version 4.1 .target sm_52 .address_size 64 .entry noop() { ret; }",
        c"noop",
    )
}

pub(super) fn launch(function: Handle, stream: Handle) {
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

pub(super) const fn policy(cap: usize, micros: u64) -> Policy {
    Policy {
        cap,
        delay: Some(Duration::from_micros(micros)),
        high_resolution: false,
        eager_after_idle: false,
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
        high_resolution: false,
        eager_after_idle: false,
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

fn observe_kernel(stream: Handle) -> Duration {
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
    let start = Instant::now();
    event.wait(wait, 10_000_000_000).unwrap();
    start.elapsed()
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

fn cases() -> impl Iterator<Item = Case> {
    [
        Case("A1", None),
        Case("B250", Some(policy(32, 250))),
        Case("C50", Some(high_resolution(50))),
        Case("C100", Some(high_resolution(100))),
        Case("C250", Some(high_resolution(250))),
        Case("C500", Some(high_resolution(500))),
        Case("C1000", Some(high_resolution(1000))),
        Case(
            "D250",
            Some(Policy {
                eager_after_idle: true,
                ..high_resolution(250)
            }),
        ),
        Case("A2", None),
    ]
    .into_iter()
    .filter(|case| selected("CUDA_FOR_INTEL_BENCH_CASES", case.0))
}

pub(super) const fn high_resolution(micros: u64) -> Policy {
    Policy {
        high_resolution: true,
        ..policy(32, micros)
    }
}

fn selected(var: &str, label: &str) -> bool {
    std::env::var(var).map_or(true, |values| values.split(',').any(|value| value == label))
}

fn process_cpu() -> Duration {
    #[repr(C)]
    #[derive(Default)]
    struct FileTime(u32, u32);
    impl FileTime {
        fn ticks(&self) -> u64 {
            (u64::from(self.1) << 32) | u64::from(self.0)
        }
    }
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentProcess() -> Opaque;
        fn GetProcessTimes(
            process: Opaque,
            create: &mut FileTime,
            exit: &mut FileTime,
            kernel: &mut FileTime,
            user: &mut FileTime,
        ) -> i32;
    }
    let (mut create, mut exit, mut kernel, mut user) = Default::default();
    assert_ne!(
        unsafe {
            GetProcessTimes(
                GetCurrentProcess(),
                &mut create,
                &mut exit,
                &mut kernel,
                &mut user,
            )
        },
        0
    );
    Duration::from_nanos((kernel.ticks() + user.ticks()) * 100)
}

fn report_cpu(label: &str, began: Instant, cpu: Duration, samples: usize) {
    let wall = began.elapsed().as_secs_f64();
    let cpu = (process_cpu() - cpu).as_secs_f64();
    println!(
        "{label} wall={wall:.3} s CPU={cpu:.3} s ({:.1}% of one logical CPU, {:.1} us/sample)",
        100.0 * cpu / wall,
        cpu * 1e6 / samples as f64
    );
}

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

fn loads() -> impl Iterator<Item = (&'static str, usize, usize)> {
    let cores = thread::available_parallelism().unwrap().get();
    [
        ("idle", 0, 0),
        ("moderate", (cores / 2).max(1), 0),
        ("heavy", cores, 0),
        ("CUDA-callers", 0, 4),
    ]
    .into_iter()
    .filter(|(label, _, _)| selected("CUDA_FOR_INTEL_BENCH_LOADS", label))
}

fn under_load(cpu_workers: usize, cuda_workers: usize, run: impl FnOnce()) {
    // Install the load context on a different thread so the benchmark keeps
    // its own current context while the explicit query handle is in use.
    let query_context = (cuda_workers > 0).then(|| {
        thread::spawn(|| (open(None).addr(), new_stream(1).addr()))
            .join()
            .unwrap()
    });
    let query_stream = query_context.map_or(0, |(_, stream)| stream);
    let stop = AtomicBool::new(false);
    let start = Barrier::new(1 + cpu_workers + cuda_workers);
    thread::scope(|scope| {
        for _ in 0..cpu_workers {
            scope.spawn(|| {
                start.wait();
                let mut work = 1u64;
                while !stop.load(Ordering::Relaxed) {
                    for _ in 0..2048 {
                        work = black_box(work.wrapping_mul(6364136223846793005).wrapping_add(1));
                    }
                }
            });
        }
        for _ in 0..cuda_workers {
            scope.spawn(|| {
                start.wait();
                while !stop.load(Ordering::Relaxed) {
                    assert_eq!(cuStreamQuery(without_provenance_mut(query_stream)), Success);
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
        run();
    });
    if let Some((context, _)) = query_context {
        assert_eq!(cuCtxDestroy_v2(without_provenance_mut(context)), Success);
    }
}

#[test]
#[ignore = "measures Windows timeout behavior without a GPU"]
fn gated_m10_host_timer() {
    for high_resolution in [false, true] {
        for micros in [50, 100, 250, 500, 1000] {
            let wake = Wake::new(high_resolution).unwrap();
            let mut samples = Vec::new();
            let cpu = process_cpu();
            let began = Instant::now();
            for i in 0..110 {
                let deadline = Instant::now() + Duration::from_micros(micros);
                wake.wait(Some(deadline)).unwrap();
                if i >= 10 {
                    samples.push(deadline.elapsed());
                }
            }
            let label = format!("host high_resolution={high_resolution} requested={micros} us");
            report_cpu(&label, began, cpu, 110);
            report(&format!("{label} overshoot"), &mut samples);
        }
    }
}

#[test]
#[ignore = "requires kernel timestamp events on an Intel Level Zero GPU"]
fn gated_m10_device_execution() {
    use crate::level_zero::profiling::OperationEvents;
    for Case(label, policy) in cases() {
        let ctx = open(policy);
        let stream = new_stream(1);
        let kernel = noop();
        let (mut timestamps, wait) = with_context(|runtime, ctx| {
            let info = &runtime.devices[ctx.device as usize].info;
            let events = OperationEvents::create(
                &runtime.api,
                &ctx.backend,
                1,
                info.timestamp_frequency_hz,
                info.kernel_timestamp_valid_bits,
                false,
            )
            .map_err(backend_error)?;
            Ok((events, runtime.api.event_wait_fn()))
        })
        .unwrap();
        let mut samples = Vec::new();
        for i in 0..120 {
            timestamps.arm(0);
            let started = Instant::now();
            launch(kernel, stream);
            assert!(!OperationEvents::disarm());
            timestamps.wait(0, wait, 10_000_000_000).unwrap();
            let elapsed = started.elapsed();
            let [start, end] = timestamps.query(0).unwrap();
            let timer = timestamps.timer;
            let ticks = timer.window(start, end, elapsed).unwrap();
            let duration = Duration::from_secs_f64(timer.seconds(ticks));
            assert!(duration <= elapsed);
            if i >= 20 {
                samples.push(duration);
            }
            assert_eq!(cuStreamSynchronize(stream), Success);
            assert_eq!(cuStreamQuery(stream), Success);
            with_context(|runtime, _| timestamps.reset(&runtime.api, 0).map_err(backend_error))
                .unwrap();
        }
        report(&format!("{label} isolated device execution"), &mut samples);
        assert_eq!(cuStreamDestroy_v2(stream), Success);
        with_context(|runtime, _| timestamps.destroy(&runtime.api).map_err(backend_error)).unwrap();
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
    }
}

#[test]
#[ignore = "requires an Intel Level Zero GPU"]
fn gated_m10_throughput() {
    for (load, cpu, callers) in loads() {
        under_load(cpu, callers, || run_throughput(load));
    }
}

fn run_throughput(load: &str) {
    for Case(case, policy) in cases() {
        let label = format!("{load} {case}");
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
            let cpu = process_cpu();
            let began = Instant::now();
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
            report_cpu(&format!("{label} {workload} {ops}"), began, cpu, 42);
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
fn gated_m10_forge() {
    for (load, cpu, callers) in loads() {
        under_load(cpu, callers, || {
            for Case(case, policy) in cases() {
                let label = format!("{load} {case}");
                println!("backend {label}");
                POLICY.set(Some(policy));
                let cpu = process_cpu();
                let began = Instant::now();
                run_bench_pipeline();
                report_cpu(&label, began, cpu, 1);
                POLICY.set(None);
            }
        });
    }
}

#[test]
#[ignore = "requires an Intel Level Zero GPU and exercises CPU saturation"]
fn gated_m10_isolated_latency_under_load() {
    for (label, cpu_workers, cuda_workers) in loads() {
        for Case(case, policy) in cases() {
            let spaced = std::env::var_os("CUDA_FOR_INTEL_BENCH_SPACED").is_some();
            if !spaced && policy.is_some_and(|policy| policy.eager_after_idle) {
                continue;
            }
            let spacing = spaced.then(|| Wake::new(true).unwrap());
            let ctx = open(policy);
            let stream = new_stream(0);
            let kernel = noop();
            let (mut enqueue, mut complete, mut observe, mut recycle) =
                (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            let mut phases = [const { Vec::new() }; 9];
            let mut native_event_wait = Vec::new();
            let mut last_recycle = Duration::ZERO;
            let cpu = process_cpu();
            let began = Instant::now();
            under_load(cpu_workers, cuda_workers, || {
                for i in 0..320 {
                    if let Some(spacing) = &spacing {
                        // Wait outside the latency window to exercise truly idle arrivals.
                        spacing
                            .wait(Some(Instant::now() + Duration::from_millis(2)))
                            .unwrap();
                    }
                    let began = Instant::now();
                    launch(kernel, stream);
                    let api = began.elapsed();
                    let wait = Instant::now();
                    let native = observe_kernel(stream);
                    let observed = wait.elapsed();
                    let done = began.elapsed();
                    if i >= 20 {
                        enqueue.push(api);
                        complete.push(done);
                        observe.push(observed);
                        native_event_wait.push(native);
                    }
                    // Safe reset after observed completion. Keep recycling out
                    // of the isolated-launch measurement for every backend.
                    let cleanup = Instant::now();
                    assert_eq!(cuStreamSynchronize(stream), Success);
                    let fence_wait = QueueFence::last_wait();
                    assert_eq!(cuStreamQuery(stream), Success);
                    if i >= 20 {
                        recycle.push(cleanup.elapsed());
                    }
                    if let Some(policy) = policy {
                        let measured = with_context(|_, ctx| {
                            let stats = &ctx
                                .streams
                                .get(stream.addr())?
                                .batch
                                .as_ref()
                                .unwrap()
                                .stats;
                            assert_eq!(
                                stats.background,
                                if policy.eager_after_idle { 0 } else { i + 1 }
                            );
                            Ok([
                                policy.delay.unwrap(),
                                stats.wake_late,
                                stats.lock_time,
                                stats.scan_late,
                                stats.close_time,
                                stats.execute_time,
                                stats.last_delay,
                                fence_wait,
                                stats.recycle_time - last_recycle,
                            ])
                        })
                        .unwrap();
                        last_recycle += measured[8];
                        if i >= 20 {
                            for (index, (samples, measurement)) in
                                phases.iter_mut().zip(measured).enumerate()
                            {
                                if !policy.eager_after_idle || index >= 7 {
                                    samples.push(measurement);
                                }
                            }
                        }
                    }
                }
            });
            report(&format!("{label} {case} enqueue"), &mut enqueue);
            report(&format!("{label} {case} complete"), &mut complete);
            report_cpu(&format!("{label} {case}"), began, cpu, 320);
            report(&format!("{label} {case} observe completion"), &mut observe);
            report(
                &format!("{label} {case} native event wait"),
                &mut native_event_wait,
            );
            report(&format!("{label} {case} sync and reap"), &mut recycle);
            for (phase, samples) in [
                "buffer to deadline",
                "deadline to wait return",
                "runtime lock",
                "deadline to scan",
                "list close",
                "queue execute",
                "buffer to submitted",
                "native queue fence wait",
                "native recycling",
            ]
            .into_iter()
            .zip(&mut phases)
            {
                if !samples.is_empty() {
                    report(&format!("{label} {case} {phase}"), samples);
                }
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
