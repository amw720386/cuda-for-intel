//! cuEventElapsedTime on hardware: plausible durations, ordering, error codes.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::c_void;
use std::ptr::null_mut;
use std::time::Instant;

const NOT_READY: i32 = 600;
const INVALID_HANDLE: i32 = 400;

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn event_elapsed_time_tracks_device_work() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
        cuEventCreate(*mut Handle, u32),
        cuEventDestroy_v2(Handle),
        cuEventRecord(Handle, Handle),
        cuEventQuery(Handle),
        cuEventSynchronize(Handle),
        cuEventElapsedTime(*mut f32, Handle, Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemHostAlloc(*mut Handle, usize, u32),
        cuMemFreeHost(Handle),
        cuMemcpyHtoDAsync_v2(u64, *const c_void, usize, Handle),
        cuStreamSynchronize(Handle),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 1));
    let (mut start, mut end, mut untimed) = (null_mut(), null_mut(), null_mut());
    ok!(cuEventCreate(&mut start, 0));
    ok!(cuEventCreate(&mut end, 0));
    ok!(cuEventCreate(&mut untimed, 2));

    const BYTES: usize = 256 << 20;
    let (mut device, mut host) = (0u64, null_mut());
    ok!(cuMemAlloc_v2(&mut device, BYTES));
    ok!(cuMemHostAlloc(&mut host, BYTES, 0));

    let mut ms = -1.0f32;
    assert_eq!(unsafe { cuEventElapsedTime(&mut ms, start, end) }, NOT_READY, "never recorded");

    let wall = Instant::now();
    ok!(cuEventRecord(start, stream));
    ok!(cuMemcpyHtoDAsync_v2(device, host.cast(), BYTES, stream));
    ok!(cuEventRecord(end, stream));
    ok!(cuEventRecord(untimed, stream));
    ok!(cuEventSynchronize(end));
    let wall_ms = wall.elapsed().as_secs_f32() * 1000.0;
    ok!(cuEventQuery(end));
    ok!(cuEventElapsedTime(&mut ms, start, end));
    eprintln!("device {ms:.3} ms, host wall {wall_ms:.3} ms for {} MiB", BYTES >> 20);
    // A 256 MiB upload takes measurable time, and cannot exceed the wall clock.
    assert!(ms > 0.05, "{ms}");
    assert!(ms <= wall_ms * 1.05 + 1.0, "{ms} vs {wall_ms}");
    let mut back = -1.0f32;
    ok!(cuEventElapsedTime(&mut back, end, start));
    assert!((back + ms).abs() < 1e-3, "reversed interval {back} vs {ms}");
    assert_eq!(unsafe { cuEventElapsedTime(&mut ms, start, untimed) }, INVALID_HANDLE);
    let mut zero = -1.0f32;
    ok!(cuEventElapsedTime(&mut zero, start, start));
    assert_eq!(zero, 0.0);

    // Back-to-back records are ordered and sub-millisecond.
    ok!(cuEventRecord(start, stream));
    ok!(cuEventRecord(end, stream));
    ok!(cuStreamSynchronize(stream));
    ok!(cuEventElapsedTime(&mut ms, start, end));
    assert!((0.0..5.0).contains(&ms), "{ms}");

    ok!(cuMemFreeHost(host));
    ok!(cuMemFree_v2(device));
    for event in [start, end, untimed] {
        ok!(cuEventDestroy_v2(event));
    }
    ok!(cuStreamDestroy_v2(stream));
    ok!(cuCtxDestroy_v2(context));
}
