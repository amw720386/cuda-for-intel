use super::events::{cuEventCreate, cuEventRecord, cuStreamWaitEvent};
use super::{cuStreamCreate, cuStreamSynchronize};
use crate::cuda::*;
use crate::level_zero::profiling::{Clock, OperationEvents};
use std::time::{Duration, Instant};

type Handle = Opaque;

struct LinkedDriver;

fn library() -> LinkedDriver {
    LinkedDriver
}

macro_rules! linked_symbols {
    ($library:ident; $( $name:ident($($arg:ty),*) ),+ $(,)?) => {
        let _ = $library;
        $(let $name: unsafe extern "system" fn($($arg),*) -> crate::cuda::CuResult = super::$name;)+
    };
}

macro_rules! linked_ok {
    ($name:ident($($arg:expr),* $(,)?)) => {
        assert_eq!(unsafe { $name($($arg),*) }, crate::cuda::CuResult::Success, "{}", stringify!($name))
    };
}

use {linked_ok as ok, linked_symbols as symbols};

#[path = "../../../tests/gpu/forge_workload.rs"]
mod forge_workload;
use forge_workload::*;
#[path = "../../../tests/gpu/pipeline.rs"]
mod pipeline;
use pipeline::{Mode, Trace};

const STAGES: [&str; 7] = [
    "HtoD input",
    "HtoD residual",
    "HtoD weights",
    "RMSNorm",
    "residual addition",
    "reduction",
    "DtoH",
];
const STAMPS: usize = BATCH * STAGES.len() * 2;
const OPERATIONS: usize = STAMPS / 2;
const WORDS: usize = STAMPS + 2;

struct Profile {
    clock: Clock,
    buffer: *mut u64,
    next: usize,
    anchor: Option<(Instant, u64)>,
    durations: [Vec<Duration>; STAGES.len()],
    span: Vec<Duration>,
    overlaps: usize,
    samples_with_overlap: usize,
}

impl Profile {
    fn new() -> Self {
        let clock = with_context(|runtime, context| {
            let info = &runtime.devices[context.device as usize].info;
            Clock::new(
                &runtime.api,
                info.timestamp_frequency_hz,
                info.timestamp_valid_bits,
            )
            .map_err(backend_error)
        })
        .expect("device profiling clock");
        // Tracked as a context allocation. Both cuMemFreeHost and context
        // destruction drain pending stream work before freeing it.
        let mut buffer = std::ptr::null_mut();
        linked_ok!(cuMemHostAlloc(&mut buffer, WORDS * size_of::<u64>(), 0));
        Self {
            clock,
            buffer: buffer.cast(),
            next: 0,
            anchor: None,
            durations: Default::default(),
            span: Vec::new(),
            overlaps: 0,
            samples_with_overlap: 0,
        }
    }

    fn device_sample(&self) -> u64 {
        with_context(|_, context| self.clock.sample(&context.backend).map_err(backend_error))
            .expect("sample global clock")
            .1
    }
}

impl Drop for Profile {
    fn drop(&mut self) {
        let result = cuMemFreeHost(self.buffer.cast());
        if !std::thread::panicking() {
            assert_eq!(result, Success);
        }
    }
}

impl Trace for Profile {
    fn begin(&mut self) {
        self.next = 0;
        // Previous samples have completed. Detect an unwritten destination.
        unsafe { self.buffer.write_bytes(0xff, WORDS) };
        self.anchor = Some((Instant::now(), self.device_sample()));
    }

    fn mark(&mut self, stream: Option<Handle>) {
        assert!(self.next < STAMPS);
        let dst = unsafe { self.buffer.add(self.next + 1) };
        with_context(|runtime, context| {
            let api = &runtime.api;
            if let Some(stream) = stream {
                let list = context.streams.get_mut(super::stream_key(stream)?)?;
                unsafe { list.list.append_timestamp(&self.clock, dst) }.map_err(backend_error)?;
                list.submitted();
            } else {
                // Match the wait already required by a blocking CUDA copy.
                context.streams.wait_legacy(api, u64::MAX)?;
                unsafe { self.clock.append_blocking(&context.backend, dst) }
                    .map_err(backend_error)?;
            }
            Ok(())
        })
        .expect("append device timestamp");
        self.next += 1;
    }

    fn end(&mut self, mode: Mode, slots: usize, sample: bool) {
        assert_eq!(self.next, STAMPS);
        let (start, first) = self.anchor.take().unwrap();
        let last = self.device_sample();
        let timer = self.clock.timer;
        let elapsed = start.elapsed();
        let window = timer
            .window(first, last, elapsed)
            .expect("trace exceeds counter period");
        // Both host samples bound the list writes. Check scale as well as order
        // so a clock reset or a different timestamp representation fails closed.
        assert!(timer.seconds(window) <= elapsed.as_secs_f64() + 0.001);
        let words = unsafe { std::slice::from_raw_parts(self.buffer, WORDS) };
        assert_eq!([words[0], words[WORDS - 1]], [u64::MAX; 2]);
        let stamps = &words[1..WORDS - 1];
        let intervals: Vec<_> = stamps.as_chunks::<2>().0.iter().map(|pair| {
            assert!(!pair.contains(&u64::MAX), "unwritten device timestamp");
            let begin = timer.ticks(first, pair[0]);
            let end = timer.ticks(first, pair[1]);
            assert!(
                begin <= end && end <= window,
                "timestamps outside ordered device window: {begin}..{end} of {window} ticks, raw {pair:x?}, anchors {first:x}..{last:x}"
            );
            (begin, end)
        }).collect();
        let overlaps = check_timeline(&intervals, mode, slots).expect("global timestamp order");
        if sample {
            let duration = |ticks| Duration::from_secs_f64(timer.seconds(ticks));
            for (i, &(begin, end)) in intervals.iter().enumerate() {
                self.durations[i % STAGES.len()].push(duration(end - begin));
            }
            self.span
                .push(duration(intervals.last().unwrap().1 - intervals[0].0));
            self.overlaps += overlaps;
            self.samples_with_overlap += usize::from(overlaps > 0);
        }
    }

    fn report(&mut self, label: &str) {
        println!(
            "{label}: bracket overlaps={}, samples with overlaps={}/{}",
            self.overlaps,
            self.samples_with_overlap,
            self.span.len()
        );
        for (stage, durations) in STAGES.into_iter().zip(&mut self.durations) {
            report(&format!("{label} {stage} bracket"), durations);
            durations.clear();
        }
        report(&format!("{label} device span"), &mut self.span);
        self.span.clear();
        self.overlaps = 0;
        self.samples_with_overlap = 0;
    }
}

fn check_timeline(intervals: &[(u64, u64)], mode: Mode, slots: usize) -> Result<usize, String> {
    let (batches, remainder) = intervals.as_chunks::<{ STAGES.len() }>();
    assert!(remainder.is_empty());
    for (index, batch) in batches.iter().enumerate() {
        for (i, pair) in batch.windows(2).enumerate() {
            if pair[0].1 > pair[1].0 {
                return Err(format!(
                    "batch {index} {} before {} by {} clock units",
                    STAGES[i + 1],
                    STAGES[i],
                    pair[0].1 - pair[1].0
                ));
            }
        }
        if index >= slots && batches[index - slots][6].1 > batch[0].0 {
            return Err(format!(
                "batch {index} reuses a slot before its prior download"
            ));
        }
        if index > 0 && matches!(mode, Mode::EventLinked) {
            let prior = batches[index - 1];
            if prior[2].1 > batch[0].0 || prior[5].1 > batch[3].0 || prior[6].1 > batch[6].0 {
                return Err(format!(
                    "batch {index} reverses an operation on the same stream"
                ));
            }
        }
    }
    let (copies, kernels): (Vec<_>, Vec<_>) = intervals
        .iter()
        .enumerate()
        .partition(|(i, _)| !matches!(i % STAGES.len(), 3..=5));
    Ok(copies
        .iter()
        .map(|(_, copy)| {
            kernels
                .iter()
                .filter(|(_, kernel)| copy.0.max(kernel.0) < copy.1.min(kernel.1))
                .count()
        })
        .sum())
}

#[test]
fn timeline_counts_overlap_only_after_checking_dependencies() {
    let intervals = [
        (0, 2),
        (2, 4),
        (4, 6),
        (6, 9),
        (9, 11),
        (11, 14),
        (14, 16),
        (7, 9),
        (9, 11),
        (11, 12),
        (14, 17),
        (17, 19),
        (19, 21),
        (21, 23),
    ];
    assert_eq!(check_timeline(&intervals, Mode::EventLinked, 2), Ok(4));
    assert!(check_timeline(&intervals, Mode::EventLinked, 1).is_err());
    let mut invalid = intervals;
    invalid[3].0 = 5;
    assert!(check_timeline(&invalid, Mode::EventLinked, 2).is_err());
}

struct EventProfile {
    events: OperationEvents,
    clock: Option<Clock>,
    host_start: u64,
    mapped_samples: usize,
    rejected: usize,
    first_rejection: Option<String>,
    overlaps: usize,
    marks: usize,
    start: Instant,
    durations: [Vec<Duration>; STAGES.len()],
}

impl EventProfile {
    fn create() -> Box<dyn Trace> {
        let (events, clock) = with_context(|runtime, context| {
            let info = &runtime.devices[context.device as usize].info;
            let mapped = info
                .driver_extensions
                .iter()
                .any(|(name, _)| name == "ZE_extension_event_query_kernel_timestamps");
            let clock = mapped
                .then(|| {
                    Clock::new(
                        &runtime.api,
                        info.timestamp_frequency_hz,
                        info.timestamp_valid_bits,
                    )
                })
                .transpose()
                .map_err(backend_error)?;
            let events = OperationEvents::create(
                &runtime.api,
                &context.backend,
                OPERATIONS as u32,
                info.timestamp_frequency_hz,
                info.kernel_timestamp_valid_bits,
                mapped,
            )
            .map_err(backend_error)?;
            Ok((events, clock))
        })
        .expect("operation timestamp events");
        Box::new(Self {
            events,
            clock,
            host_start: 0,
            mapped_samples: 0,
            rejected: 0,
            first_rejection: None,
            overlaps: 0,
            marks: 0,
            start: Instant::now(),
            durations: Default::default(),
        })
    }
}

impl EventProfile {
    fn host_sample(&self) -> Option<u64> {
        self.clock.as_ref().map(|clock| {
            with_context(|_, context| clock.sample(&context.backend).map_err(backend_error))
                .unwrap()
                .0
        })
    }
}

impl Drop for EventProfile {
    fn drop(&mut self) {
        OperationEvents::disarm();
        let result = with_context(|runtime, context| {
            context.streams.wait_all(&runtime.api)?;
            self.events.destroy(&runtime.api).map_err(backend_error)
        });
        if !std::thread::panicking() {
            result.expect("destroy operation timestamp events");
        }
    }
}

impl Trace for EventProfile {
    fn begin(&mut self) {
        if self.marks != 0 {
            with_context(|runtime, _| {
                for index in 0..OPERATIONS {
                    self.events
                        .reset(&runtime.api, index)
                        .map_err(backend_error)?;
                }
                Ok(())
            })
            .expect("reset completed timestamp events");
        }
        self.marks = 0;
        self.start = Instant::now();
        self.host_start = self.host_sample().unwrap_or_default();
    }

    fn mark(&mut self, _stream: Option<Handle>) {
        if self.marks.is_multiple_of(2) {
            self.events.arm(self.marks / 2);
        } else {
            assert!(
                !OperationEvents::disarm(),
                "operation did not consume event"
            );
        }
        self.marks += 1;
    }

    fn end(&mut self, mode: Mode, slots: usize, sample: bool) {
        assert_eq!(self.marks, STAMPS);
        let elapsed = self.start.elapsed();
        let host_end = self.host_sample();
        let timer = self.events.timer;
        let mut nonzero = [false; STAGES.len()];
        for index in 0..OPERATIONS {
            let stage = index % STAGES.len();
            let [begin, end] = self
                .events
                .query(index)
                .unwrap_or_else(|error| panic!("{} timestamp: {error}", STAGES[stage]));
            let ticks = timer
                .window(begin, end, elapsed)
                .expect("trace exceeds counter period");
            let seconds = timer.seconds(ticks);
            assert!(
                seconds <= elapsed.as_secs_f64(),
                "{} timestamp exceeds run",
                STAGES[stage]
            );
            nonzero[stage] |= ticks != 0;
            if sample {
                self.durations[stage].push(Duration::from_secs_f64(seconds));
            }
        }
        if let Some(host_end) = host_end {
            let timeline = (0..OPERATIONS)
                .map(|index| {
                    let (begin, end) = self
                        .events
                        .mapped_interval(index)
                        .map_err(|error| error.to_string())?;
                    if self.host_start > begin || end > host_end {
                        return Err(format!(
                            "operation {index} falls outside the host clock anchors"
                        ));
                    }
                    Ok((begin, end))
                })
                .collect::<Result<Vec<_>, _>>();
            match timeline.and_then(|intervals| check_timeline(&intervals, mode, slots)) {
                Ok(overlaps) if sample => {
                    self.mapped_samples += 1;
                    self.overlaps += overlaps;
                }
                Err(reason) if sample => {
                    self.rejected += 1;
                    self.first_rejection.get_or_insert(reason);
                }
                _ => (),
            }
        }
        assert!(
            nonzero.into_iter().all(|valid| valid),
            "missing operation timestamps: {nonzero:?}"
        );
    }

    fn report(&mut self, label: &str) {
        if self.clock.is_some() {
            println!(
                "{label}: mapped event overlaps={}, valid samples={}, rejected={}",
                self.overlaps, self.mapped_samples, self.rejected
            );
            if let Some(reason) = self.first_rejection.take() {
                println!("{label}: first mapped timestamp rejection: {reason}");
            }
        } else {
            println!("{label}: mapped event timestamps unavailable");
        }
        self.mapped_samples = 0;
        self.rejected = 0;
        self.overlaps = 0;
        for (stage, durations) in STAGES.into_iter().zip(&mut self.durations) {
            report(&format!("{label} {stage} event duration"), durations);
            durations.clear();
        }
    }
}

#[test]
#[ignore = "requires an Intel Level Zero GPU"]
fn gated_forge_device_profile() {
    pipeline::run_pipeline(
        Some(|| Box::new(Profile::new())),
        std::env::var_os("CUDA_FOR_INTEL_BENCH_PROFILING").is_some(),
    );
}

#[test]
#[ignore = "requires operation timestamp support on an Intel Level Zero GPU"]
fn gated_forge_operation_timestamps() {
    pipeline::run_pipeline(
        Some(EventProfile::create),
        std::env::var_os("CUDA_FOR_INTEL_BENCH_PROFILING").is_some(),
    );
}

#[test]
#[ignore = "requires an Intel Level Zero GPU"]
fn gated_timestamp_storage_survives_stream_retirement() {
    use super::{cuStreamDestroy_v2, cuStreamQuery};
    use std::ptr::null_mut;

    assert_eq!(cuInit(0), Success);
    for _ in 0..8 {
        let (mut context, mut stream) = (null_mut(), null_mut());
        linked_ok!(cuCtxCreate_v2(&mut context, 0, 0));
        linked_ok!(cuStreamCreate(&mut stream, 1));
        let mut profile = Profile::new();
        profile.begin();
        let gate = with_context(|runtime, context| {
            let api = &runtime.api;
            let gate = context
                .streams
                .fences
                .acquire(api, &context.backend)
                .map_err(backend_error)?;
            let list = context.streams.get_mut(stream.addr())?;
            list.list.gate(api, &gate).map_err(backend_error)?;
            list.submitted();
            Ok(gate)
        })
        .unwrap();
        profile.mark(Some(stream));
        let pending = cuStreamQuery(stream);
        let retired = cuStreamDestroy_v2(stream);
        with_context(|runtime, _| gate.host_signal(&runtime.api).map_err(backend_error)).unwrap();
        drop(gate);
        assert_eq!(pending, NotReady);
        assert_eq!(retired, Success);
        assert_eq!(cuCtxSynchronize(), Success);
        assert_ne!(unsafe { profile.buffer.add(1).read() }, u64::MAX);
        drop(profile);
        with_context(|_, context| {
            assert!(context.host_allocations.is_empty() && context.streams.retired.is_empty());
            Ok(())
        })
        .unwrap();
        assert_eq!(cuCtxDestroy_v2(context), Success);
    }
}
