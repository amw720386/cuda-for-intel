//! `CFI_GPU_PROFILE=1`: per-kernel GPU execution time from Level Zero kernel timestamp events,
//! without serialising launches (unlike `CFI_PROFILE`, whose per-launch wait adds the launch and
//! sync round trip to every kernel and changes the pipeline it measures).
//!
//! Every launch that carries no stream signal of its own gets a timestamp event from a ring of
//! `SLOTS`; a slot is read back when the ring reuses it or when a snapshot is written. Launches that
//! already signal a fence (blocking streams), launches of a second context and failed launches are
//! counted as `skipped`. Snapshots are cumulative and appended to `CFI_GPU_PROFILE_OUT` (default
//! `cfi-gpu-profile.tsv`) on the first launch after `CFI_GPU_PROFILE_FLUSH_MS` (default 2000) ms,
//! after reading every completed slot, so a pause between two requests gives a clean boundary:
//!
//! ```text
//! GPUPROF wall_ms=<ms since first launch> launches=<n> timed=<n> skipped=<n>
//! K <count> <total_ns> <max_ns> <kernel name>
//! END
//! ```
//!
//! Kernel time is the device's global start-to-end time of the kernel. Kernels that overlap (several
//! streams) are each counted in full. Summarise with `scripts/profile/gpu_profile.py`.
use super::Context;
use crate::level_zero::profiling::OperationEvents;
use crate::level_zero::{Api, DeviceInfo};
use std::collections::HashMap;
use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

const SLOTS: u32 = 4096;
/// A slot is only reused `SLOTS` launches after it was armed, so it has normally completed long
/// before; this bounds the wait if a kernel faulted and never signalled.
const REUSE_TIMEOUT_NS: u64 = 5_000_000_000;

pub(super) fn enabled() -> bool {
    static ON: OnceLock<bool> = OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("CFI_GPU_PROFILE").is_some_and(|v| v == "1"))
}

struct Profiler {
    events: OperationEvents,
    context: usize,
    /// Kernel id armed in each slot, `None` when the slot is free.
    slots: Vec<Option<u32>>,
    next: usize,
    ids: HashMap<String, u32>,
    names: Vec<String>,
    /// Per kernel id: count, total ns, max ns.
    stats: Vec<[u64; 3]>,
    launches: u64,
    timed: u64,
    skipped: u64,
    started: Instant,
    flushed: Instant,
    interval: Duration,
    path: std::path::PathBuf,
}

// The Level Zero handles in `events` are only used under the `PROFILER` lock.
unsafe impl Send for Profiler {}

enum State {
    Uninit,
    Failed,
    On(Profiler),
}

static PROFILER: Mutex<State> = Mutex::new(State::Uninit);

impl Profiler {
    fn create(api: &Api, info: &DeviceInfo, backend: &Context, context: usize) -> Option<Self> {
        let events = OperationEvents::create(
            api,
            backend,
            SLOTS,
            info.timestamp_frequency_hz,
            info.kernel_timestamp_valid_bits,
            false,
        )
        .inspect_err(|error| eprintln!("CUDA for Intel: CFI_GPU_PROFILE disabled: {error:?}"))
        .ok()?;
        let interval = std::env::var("CFI_GPU_PROFILE_FLUSH_MS")
            .ok()
            .and_then(|v| v.parse().ok())
            .map_or(Duration::from_secs(2), Duration::from_millis);
        let path = std::env::var_os("CFI_GPU_PROFILE_OUT").map_or_else(|| "cfi-gpu-profile.tsv".into(), Into::into);
        let now = Instant::now();
        Some(Self {
            slots: vec![None; events.len()],
            events,
            context,
            next: 0,
            ids: HashMap::new(),
            names: Vec::new(),
            stats: Vec::new(),
            launches: 0,
            timed: 0,
            skipped: 0,
            started: now,
            flushed: now,
            interval,
            path,
        })
    }

    fn id(&mut self, name: &str) -> u32 {
        if let Some(&id) = self.ids.get(name) {
            return id;
        }
        let id = self.names.len() as u32;
        self.ids.insert(name.to_owned(), id);
        self.names.push(name.to_owned());
        self.stats.push([0; 3]);
        id
    }

    /// Reads slot `index` into the statistics and frees it. Without `block`, a slot whose kernel has
    /// not finished is left alone and `false` is returned.
    fn harvest(&mut self, api: &Api, index: usize, block: bool) -> bool {
        let Some(id) = self.slots[index] else { return true };
        if block {
            if self.events.wait_runtime(api, index, REUSE_TIMEOUT_NS).is_err() {
                self.free(api, index);
                self.skipped += 1;
                return true;
            }
        } else if !self.events.ready(api, index) {
            return false;
        }
        if let Ok([start, end]) = self.events.query(index) {
            let ns = (self.events.timer.seconds(self.events.timer.ticks(start, end)) * 1e9) as u64;
            let stat = &mut self.stats[id as usize];
            stat[0] += 1;
            stat[1] += ns;
            stat[2] = stat[2].max(ns);
        } else {
            self.skipped += 1;
        }
        self.free(api, index);
        true
    }

    fn free(&mut self, api: &Api, index: usize) {
        let _ = self.events.reset(api, index);
        self.slots[index] = None;
    }

    fn snapshot(&mut self, api: &Api) {
        for index in 0..self.slots.len() {
            self.harvest(api, index, false);
        }
        let mut order: Vec<usize> = (0..self.names.len()).collect();
        order.sort_by_key(|&id| std::cmp::Reverse(self.stats[id][1]));
        let mut text = format!(
            "GPUPROF wall_ms={} launches={} timed={} skipped={}\n",
            self.started.elapsed().as_millis(),
            self.launches,
            self.timed,
            self.skipped
        );
        for id in order {
            let [count, total, max] = self.stats[id];
            text.push_str(&format!("K\t{count}\t{total}\t{max}\t{}\n", self.names[id]));
        }
        text.push_str("END\n");
        if let Ok(mut file) = std::fs::OpenOptions::new().create(true).append(true).open(&self.path) {
            let _ = file.write_all(text.as_bytes());
        }
        self.flushed = Instant::now();
    }
}

/// Arms a timestamp event for the next launch of `name` on this thread. Pass the result to [`after`].
pub(super) fn before(api: &Api, info: &DeviceInfo, backend: &Context, context: usize, name: &str) -> Option<usize> {
    let mut state = PROFILER.lock().ok()?;
    if matches!(*state, State::Uninit) {
        *state = Profiler::create(api, info, backend, context).map_or(State::Failed, State::On);
    }
    let State::On(profiler) = &mut *state else { return None };
    profiler.launches += 1;
    if profiler.context != context {
        profiler.skipped += 1;
        return None;
    }
    if profiler.flushed.elapsed() >= profiler.interval {
        profiler.snapshot(api);
    }
    let index = profiler.next;
    profiler.next = (index + 1) % profiler.slots.len();
    profiler.harvest(api, index, true);
    let id = profiler.id(name);
    profiler.slots[index] = Some(id);
    profiler.events.arm_runtime(index);
    Some(index)
}

/// Disarms the event if the launch did not take it (it had its own signal) or failed.
pub(super) fn after(api: &Api, slot: Option<usize>, launched: bool) {
    let Some(index) = slot else { return };
    let unused = OperationEvents::disarm();
    let Ok(mut state) = PROFILER.lock() else { return };
    let State::On(profiler) = &mut *state else { return };
    if unused || !launched {
        profiler.free(api, index);
        profiler.skipped += 1;
    } else {
        profiler.timed += 1;
    }
}
