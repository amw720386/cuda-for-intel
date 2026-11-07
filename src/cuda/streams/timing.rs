//! Event timestamps: `cuEventElapsedTime` and the bookkeeping behind it.
//!
//! Fences are Level Zero events signalled by barriers and carry no timestamps.
//! A timing-enabled CUDA event therefore records through two appends:
//!
//! 1. a barrier that signals a private `ready` fence once everything previously
//!    submitted to the stream (and every wait edge) has finished, and
//! 2. `zeCommandListAppendWriteGlobalTimestamp`, which waits for `ready`, writes
//!    the device global timer into a per-event host slot and signals the
//!    completion fence the rest of the stream machinery already uses.
//!
//! The completion fence therefore still means "the event has happened", so
//! `cuEventQuery`, `cuEventSynchronize` and `cuStreamWaitEvent` are unchanged and
//! the slot is valid exactly when the fence is. Elapsed time is the signed
//! difference of two slots, wrap-aware within the device's valid timer bits and
//! scaled by `timerResolution` (a frequency in Hz).
//!
//! Error mapping follows the CUDA driver: an event created with
//! `CU_EVENT_DISABLE_TIMING` has no timestamp and yields `InvalidHandle`, an
//! event that was never recorded or has not completed yields `NotReady`.
use super::{Fence, Streams};
use crate::cuda::CuResult::{self, *};
use crate::cuda::{Api, Context, Opaque, backend_error, trace, with_context};
use std::collections::HashMap;
use std::sync::Arc;

/// `CU_EVENT_DISABLE_TIMING`.
const DISABLE_TIMING: u32 = 2;
const PAGE_BYTES: usize = 4096;
const SLOTS_PER_PAGE: usize = PAGE_BYTES / 8;

/// Device timer: ticks per second and number of valid counter bits.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) struct DeviceTimer {
    hz: u64,
    bits: u32,
}

impl DeviceTimer {
    pub(super) fn new(hz: Option<u64>, bits: u32) -> Option<Self> {
        let hz = hz.filter(|&hz| hz > 0)?;
        (1..=64).contains(&bits).then_some(Self { hz, bits })
    }

    /// `end - start` in ticks, interpreted as a signed value of `bits` width so
    /// a counter rollover between the stamps still gives the right answer.
    pub(super) fn ticks(self, start: u64, end: u64) -> i64 {
        let shift = 64 - self.bits;
        (end.wrapping_sub(start) << shift) as i64 >> shift
    }

    pub(super) fn millis(self, start: u64, end: u64) -> f32 {
        (self.ticks(start, end) as f64 * 1000.0 / self.hz as f64) as f32
    }
}

enum Page {
    /// Pinned host memory visible to the device.
    Host(usize),
    /// Plain heap memory for the fake Level Zero backend of unit tests.
    #[cfg(test)]
    Heap(Box<[u64]>),
}

impl Page {
    fn base(&self) -> usize {
        match self {
            Self::Host(address) => *address,
            #[cfg(test)]
            Self::Heap(words) => words.as_ptr().addr(),
        }
    }
}

fn is_fake(api: &Api) -> bool {
    #[cfg(test)]
    {
        api.is_fake()
    }
    #[cfg(not(test))]
    {
        let _ = api;
        false
    }
}

struct Timing {
    /// Timestamp slot of the event, allocated on its first record.
    slot: Option<usize>,
}

/// Per-context event timing state.
#[derive(Default)]
pub(super) struct EventTimes {
    /// Events created with timing enabled.
    events: HashMap<usize, Timing>,
    pages: Vec<Page>,
    free: Vec<usize>,
    /// Slots of destroyed events whose last record may still write.
    quarantine: Vec<(usize, Arc<Fence>)>,
}

impl EventTimes {
    pub(super) fn register(&mut self, event: usize, flags: u32) {
        if flags & DISABLE_TIMING == 0 {
            self.events.insert(event, Timing { slot: None });
        }
    }

    #[cfg(test)]
    pub(super) fn page_count(&self) -> usize {
        self.pages.len()
    }

    pub(super) fn enabled(&self, event: usize) -> bool {
        self.events.contains_key(&event)
    }

    fn slot_of(&self, event: usize) -> Option<usize> {
        self.events.get(&event)?.slot
    }

    fn reclaim(&mut self, api: &Api) {
        let mut index = 0;
        while index < self.quarantine.len() {
            // A fence that cannot be queried keeps its slot out of circulation.
            if self.quarantine[index].1.ready(api).unwrap_or(false) {
                self.free.push(self.quarantine.swap_remove(index).0);
            } else {
                index += 1;
            }
        }
    }

    /// Slot for `event`, allocating pinned memory when none is free.
    fn slot_for(&mut self, api: &Api, context: &Context, event: usize) -> Result<usize, CuResult> {
        if let Some(slot) = self.slot_of(event) {
            return Ok(slot);
        }
        self.reclaim(api);
        if self.free.is_empty() {
            let page = if is_fake(api) {
                #[cfg(test)]
                {
                    Page::Heap(vec![0u64; SLOTS_PER_PAGE].into_boxed_slice())
                }
                #[cfg(not(test))]
                unreachable!()
            } else {
                Page::Host(
                    context
                        .alloc_host(api, PAGE_BYTES)
                        .inspect_err(|error| context.health.observe(error))
                        .map_err(backend_error)?,
                )
            };
            let base = page.base();
            self.free.extend((0..SLOTS_PER_PAGE).rev().map(|n| base + 8 * n));
            self.pages.push(page);
        }
        let slot = self.free.pop().ok_or(OutOfMemory)?;
        if let Some(timing) = self.events.get_mut(&event) {
            timing.slot = Some(slot);
        }
        Ok(slot)
    }

    /// The event is gone; its slot returns once its last record has completed.
    pub(super) fn release(&mut self, event: usize, last: Option<Arc<Fence>>) {
        if let Some(Timing { slot: Some(slot) }) = self.events.remove(&event) {
            match last {
                Some(fence) => self.quarantine.push((slot, fence)),
                None => self.free.push(slot),
            }
        }
    }

    pub(super) fn clear(&mut self) {
        self.events.clear();
        self.free.clear();
        self.quarantine.clear();
    }

    /// Free pinned pages. Call only when no record can still write.
    pub(super) fn release_pages(&mut self, api: &Api, context: &Context) -> Result<(), CuResult> {
        self.clear();
        while let Some(page) = self.pages.last() {
            if let Page::Host(address) = page {
                context
                    .free(api, *address)
                    .inspect_err(|error| context.health.observe(error))
                    .map_err(backend_error)?;
            }
            self.pages.pop();
        }
        Ok(())
    }
}

fn read_stamp(slot: usize) -> u64 {
    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
    unsafe { std::ptr::with_exposed_provenance::<u64>(slot).read_volatile() }
}

impl Streams {
    /// Record `event` on stream `key`, stamping the device timer when the event
    /// was created with timing enabled.
    pub(super) fn record_event(
        &mut self,
        api: &Api,
        context: &Context,
        key: usize,
        event: usize,
    ) -> Result<Arc<Fence>, CuResult> {
        if !self.times.enabled(event) {
            return self.event_marker(api, context, key, None);
        }
        let slot = self.times.slot_for(api, context, event)?;
        let blocking = self.get(key)?.blocking;
        self.collect_retired(api)?;
        let ready = self.next_fence(api, context)?;
        let completion = self.next_fence(api, context)?;
        let mut waits = self.legacy_waits(api, key, blocking)?;
        // The submission keeps `ready` alive until the timestamp has consumed it.
        waits.push(Arc::clone(&ready));
        self.get_mut(key)?.record(
            api,
            context,
            Some(Arc::clone(&completion)),
            waits,
            |list, signal, waits| {
                let (ready, others) = waits.split_last().unwrap();
                list.barrier(api, ready, others)?;
                list.write_timestamp(
                    api,
                    std::ptr::with_exposed_provenance_mut(slot),
                    signal.unwrap(),
                    std::slice::from_ref(ready),
                )
            },
        )?;
        Ok(completion)
    }

    pub(in crate::cuda) fn release_timestamps(&mut self, api: &Api, context: &Context) -> Result<(), CuResult> {
        self.times.release_pages(api, context)
    }
}

/// Elapsed milliseconds between two completed, timing-enabled events.
fn elapsed(start: Opaque, end: Opaque) -> Result<f32, CuResult> {
    with_context(|runtime, context| {
        let streams = &context.streams;
        for event in [start, end] {
            if streams.captured.contains_key(&event.addr()) {
                return Err(CapturedEvent);
            }
            streams.event(event)?;
            if !streams.times.enabled(event.addr()) {
                return Err(InvalidHandle);
            }
        }
        let mut stamps = [0u64; 2];
        for (stamp, event) in stamps.iter_mut().zip([start, end]) {
            // Never recorded, or recorded and still in flight.
            let Some(recording) = streams.event(event)? else {
                return Err(NotReady);
            };
            if let Ok(producer) = streams.get(recording.producer) {
                producer.check()?;
            }
            if !recording
                .fence
                .ready(&runtime.api)
                .inspect_err(|error| context.backend.health.observe(error))
                .map_err(backend_error)?
            {
                return Err(NotReady);
            }
            let slot = streams.times.slot_of(event.addr()).ok_or(NotReady)?;
            *stamp = read_stamp(slot);
        }
        let info = &runtime.devices[context.device as usize].info;
        let timer = DeviceTimer::new(info.timestamp_frequency_hz, info.timestamp_valid_bits)
            .ok_or(NotSupported)?;
        Ok(timer.millis(stamps[0], stamps[1]))
    })
}

fn elapsed_to(name: &'static str, out: *mut f32, start: Opaque, end: Opaque) -> CuResult {
    trace::call(name, || {
        if out.is_null() {
            return InvalidValue;
        }
        match elapsed(start, end) {
            Ok(ms) => {
                unsafe { out.write(ms) };
                Success
            }
            Err(error) => error,
        }
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuEventElapsedTime(out: *mut f32, start: Opaque, end: Opaque) -> CuResult {
    elapsed_to("cuEventElapsedTime", out, start, end)
}

/// CUDA 12.8 variant; the arguments are identical.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cuEventElapsedTime_v2(out: *mut f32, start: Opaque, end: Opaque) -> CuResult {
    elapsed_to("cuEventElapsedTime_v2", out, start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timer_scales_ticks_to_milliseconds() {
        let timer = DeviceTimer::new(Some(10_000_000), 36).unwrap();
        assert_eq!(timer.ticks(100, 150), 50);
        assert_eq!(timer.millis(0, 10_000), 1.0);
        assert_eq!(timer.millis(5, 5), 0.0);
        // 19.2 MHz, the usual Intel GPU timestamp clock.
        let timer = DeviceTimer::new(Some(19_200_000), 36).unwrap();
        assert!((timer.millis(0, 19_200) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn timer_handles_rollover_and_negative_intervals() {
        let timer = DeviceTimer::new(Some(1_000_000), 32).unwrap();
        // The counter wraps between the stamps: still 20 ticks.
        assert_eq!(timer.ticks(0xffff_fff8, 0x0000_000c), 20);
        assert_eq!(timer.millis(0xffff_fff8, 0x0000_000c), 0.02);
        // End before start is a negative interval, not a huge positive one.
        assert_eq!(timer.ticks(10, 4), -6);
        assert!(timer.millis(1000, 0) < 0.0);
        let full = DeviceTimer::new(Some(1), 64).unwrap();
        assert_eq!(full.ticks(u64::MAX, 1), 2);
        // Upper garbage above the valid bits is ignored.
        assert_eq!(timer.ticks(0xabcd_0000_0000_0005, 0x1234_0000_0000_0009), 4);
    }

    #[test]
    fn timer_rejects_unusable_properties() {
        assert!(DeviceTimer::new(None, 36).is_none());
        assert!(DeviceTimer::new(Some(0), 36).is_none());
        assert!(DeviceTimer::new(Some(1), 0).is_none());
        assert!(DeviceTimer::new(Some(1), 65).is_none());
    }
}
