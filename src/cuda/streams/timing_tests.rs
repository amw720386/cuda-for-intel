//! Event timing on the fake Level Zero backend. The fake device timer runs at
//! `FAKE_TIMER_HZ` with 36 valid bits and advances a configurable number of
//! ticks for every timestamp write.
use super::events::{
    cuStreamWaitEvent,
    cuEventCreate, cuEventDestroy_v2, cuEventQuery, cuEventRecord, cuEventSynchronize,
};
use super::fake_tests::{BATCH, Fixture, assert_baseline, context, stream};
use super::timing::{cuEventElapsedTime, cuEventElapsedTime_v2};
use super::*;
use crate::cuda::*;
use crate::level_zero::fake::{self, FAKE_TIMER_HZ};
use std::ptr::null_mut;

fn event(flags: u32) -> Opaque {
    let mut out = null_mut();
    assert_eq!(unsafe { cuEventCreate(&mut out, flags) }, Success);
    out
}

fn elapsed(start: Opaque, end: Opaque) -> Result<f32, CuResult> {
    let mut ms = f32::NAN;
    match unsafe { cuEventElapsedTime(&mut ms, start, end) } {
        Success => {
            let mut v2 = f32::NAN;
            assert_eq!(unsafe { cuEventElapsedTime_v2(&mut v2, start, end) }, Success);
            assert_eq!(v2, ms);
            Ok(ms)
        }
        error => {
            let mut v2 = 0.0;
            assert_eq!(unsafe { cuEventElapsedTime_v2(&mut v2, start, end) }, error);
            Err(error)
        }
    }
}

fn ms(ticks: u64) -> f32 {
    (ticks as f64 * 1000.0 / FAKE_TIMER_HZ as f64) as f32
}

#[test]
fn elapsed_time_is_the_scaled_timestamp_difference() {
    for policy in [None, BATCH] {
        let _fixture = Fixture::new();
        let ctx = context(policy);
        fake::set_clock(1_000, 500);
        let s = stream();
        let (a, b, c) = (event(0), event(1), event(0));
        assert_eq!(cuEventRecord(a, s), Success);
        assert_eq!(cuEventRecord(b, s), Success);
        assert_eq!(cuEventRecord(c, null_mut()), Success);
        assert_eq!(cuEventSynchronize(b), Success);
        assert_eq!(cuEventSynchronize(c), Success);
        assert_eq!(cuEventQuery(a), Success);
        // Each record stamps once, in submission order: 500 ticks = 50 us.
        assert_eq!(elapsed(a, b), Ok(ms(500)));
        assert_eq!(elapsed(a, b), Ok(0.05));
        assert_eq!(elapsed(a, a), Ok(0.0));
        // Reversed arguments give a negative interval.
        assert_eq!(elapsed(b, a), Ok(-0.05));
        // Re-recording replaces the stamp.
        assert_eq!(cuEventRecord(a, s), Success);
        assert_eq!(cuEventSynchronize(a), Success);
        assert_eq!(elapsed(a, b), Ok(-ms(1_000)));
        for e in [a, b, c] {
            assert_eq!(cuEventDestroy_v2(e), Success);
        }
        assert_eq!(cuStreamDestroy_v2(s), Success);
        assert_eq!(cuCtxDestroy_v2(ctx), Success);
        assert_baseline();
    }
}

#[test]
fn elapsed_time_error_codes() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    fake::set_clock(0, 100);
    let s = stream();
    let timed = event(0);
    let untimed = event(2);
    let other = event(0);
    let mut ms = 0.0f32;

    // Never recorded: the interval is not available yet.
    assert_eq!(elapsed(timed, other), Err(NotReady));
    assert_eq!(cuEventRecord(timed, s), Success);
    assert_eq!(cuEventRecord(untimed, s), Success);
    assert_eq!(cuEventSynchronize(untimed), Success);
    assert_eq!(elapsed(timed, other), Err(NotReady));

    // DISABLE_TIMING events have no timestamps, whatever their state.
    assert_eq!(elapsed(timed, untimed), Err(InvalidHandle));
    assert_eq!(elapsed(untimed, timed), Err(InvalidHandle));
    assert_eq!(elapsed(untimed, untimed), Err(InvalidHandle));

    // Bad pointers and handles.
    assert_eq!(unsafe { cuEventElapsedTime(null_mut(), timed, timed) }, InvalidValue);
    assert_eq!(unsafe { cuEventElapsedTime_v2(null_mut(), timed, timed) }, InvalidValue);
    let bogus = std::ptr::without_provenance_mut(0x5555);
    assert_eq!(unsafe { cuEventElapsedTime(&mut ms, bogus, timed) }, InvalidHandle);
    assert_eq!(unsafe { cuEventElapsedTime(&mut ms, timed, bogus) }, InvalidHandle);
    assert_eq!(unsafe { cuEventElapsedTime(&mut ms, null_mut(), timed) }, InvalidHandle);
    assert!(ms.is_nan() || ms == 0.0);

    // Destroyed events are invalid.
    assert_eq!(cuEventSynchronize(timed), Success);
    assert_eq!(cuEventDestroy_v2(other), Success);
    assert_eq!(elapsed(timed, other), Err(InvalidHandle));
    for e in [timed, untimed] {
        assert_eq!(cuEventDestroy_v2(e), Success);
    }
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn pending_events_report_not_ready_until_the_device_catches_up() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    fake::set_clock(0, 2_000);
    let s = stream();
    let (a, b) = (event(0), event(0));
    fake::pause();
    assert_eq!(cuEventRecord(a, s), Success);
    assert_eq!(cuEventRecord(b, s), Success);
    assert_eq!(cuEventQuery(a), NotReady);
    assert_eq!(elapsed(a, b), Err(NotReady));
    assert_eq!(elapsed(b, a), Err(NotReady));
    fake::complete();
    assert_eq!(cuEventSynchronize(b), Success);
    assert_eq!(cuEventQuery(b), Success);
    assert_eq!(elapsed(a, b), Ok(ms(2_000)));
    // A fresh record makes the event pending again.
    fake::pause();
    assert_eq!(cuEventRecord(a, s), Success);
    assert_eq!(elapsed(a, b), Err(NotReady));
    fake::complete();
    assert_eq!(cuEventSynchronize(a), Success);
    assert_eq!(elapsed(a, b), Ok(-ms(2_000)));
    for e in [a, b] {
        assert_eq!(cuEventDestroy_v2(e), Success);
    }
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn counter_rollover_between_stamps() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    // 36 valid bits: the second stamp wraps past 2^36.
    fake::set_clock((1u64 << 36) - 250, 500);
    let s = stream();
    let (a, b) = (event(0), event(0));
    assert_eq!(cuEventRecord(a, s), Success);
    assert_eq!(cuEventRecord(b, s), Success);
    assert_eq!(cuEventSynchronize(b), Success);
    assert_eq!(elapsed(a, b), Ok(ms(500)));
    assert_eq!(elapsed(b, a), Ok(ms(500) * -1.0));
    for e in [a, b] {
        assert_eq!(cuEventDestroy_v2(e), Success);
    }
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn timestamps_follow_stream_order_across_streams() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    fake::set_clock(0, 10);
    let (s1, s2) = (stream(), stream());
    let (a, b) = (event(0), event(0));
    fake::pause();
    assert_eq!(cuEventRecord(a, s1), Success);
    assert_eq!(cuStreamWaitEvent(s2, a, 0), Success);
    assert_eq!(cuEventRecord(b, s2), Success);
    fake::complete();
    assert_eq!(cuEventSynchronize(b), Success);
    // b waited on a, so its stamp cannot precede a's.
    assert!(elapsed(a, b).unwrap() > 0.0);
    for e in [a, b] {
        assert_eq!(cuEventDestroy_v2(e), Success);
    }
    for stream in [s1, s2] {
        assert_eq!(cuStreamDestroy_v2(stream), Success);
    }
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn slots_are_recycled_after_event_destruction() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    fake::set_clock(0, 1);
    let s = stream();
    // Far more events than one page of slots, each destroyed after completing.
    for round in 0..2_000u32 {
        let e = event(0);
        assert_eq!(cuEventRecord(e, s), Success);
        if round % 7 == 0 {
            assert_eq!(cuEventSynchronize(e), Success);
        }
        assert_eq!(cuEventDestroy_v2(e), Success);
    }
    assert_eq!(cuStreamSynchronize(s), Success);
    let pages = with_context(|_, c| Ok(c.streams.times.page_count())).unwrap();
    assert!(pages <= 2, "{pages} pages");
    // Many live events at once need more pages, which are freed with the context.
    let live: Vec<_> = (0..600).map(|_| event(0)).collect();
    for &e in &live {
        assert_eq!(cuEventRecord(e, s), Success);
    }
    assert_eq!(cuStreamSynchronize(s), Success);
    assert!(elapsed(live[0], live[599]).unwrap() > 0.0);
    let pages = with_context(|_, c| Ok(c.streams.times.page_count())).unwrap();
    assert!(pages >= 2);
    for e in live {
        assert_eq!(cuEventDestroy_v2(e), Success);
    }
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn drivers_without_a_timer_report_not_supported() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    fake::set_clock(0, 1);
    let s = stream();
    let (a, b) = (event(0), event(0));
    assert_eq!(cuEventRecord(a, s), Success);
    assert_eq!(cuEventRecord(b, s), Success);
    assert_eq!(cuEventSynchronize(b), Success);
    assert!(elapsed(a, b).is_ok());
    // A device that reports no timer resolution cannot convert ticks to time.
    state(|state| {
        state.runtime.devices[0].info.timestamp_frequency_hz = None;
        Ok(())
    })
    .unwrap();
    assert_eq!(elapsed(a, b), Err(NotSupported));
    for e in [a, b] {
        assert_eq!(cuEventDestroy_v2(e), Success);
    }
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}
