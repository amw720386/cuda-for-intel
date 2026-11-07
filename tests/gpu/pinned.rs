use super::{Handle, library, ok, symbols, test_device};
use std::ffi::c_void;
use std::ptr::{null_mut, without_provenance_mut};

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn pinned_async_transfers() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxSetCurrent(Handle),
        cuCtxSynchronize(),
        cuMemHostAlloc(*mut Handle, usize, u32),
        cuMemFreeHost(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoDAsync_v2(u64, *const c_void, usize, Handle),
        cuMemcpyDtoHAsync_v2(Handle, u64, usize, Handle),
        cuMemcpyDtoH_v2(Handle, u64, usize),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
        cuStreamSynchronize(Handle),
        cuEventCreate(*mut Handle, u32),
        cuEventRecord(Handle, Handle),
        cuEventSynchronize(Handle),
        cuStreamWaitEvent(Handle, Handle, u32),
    );
    ok!(cuInit(0));
    let (mut ctx, mut foreign, mut src, mut dst) = (null_mut(), null_mut(), null_mut(), null_mut());
    let (mut up, mut down, mut event) = (null_mut(), null_mut(), null_mut());
    let (mut device, mut other_device) = (0, 0);
    const SIZE: usize = 37;
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    assert_eq!(unsafe { cuMemHostAlloc(null_mut(), SIZE, 0) }, 1);
    assert_eq!(unsafe { cuMemHostAlloc(&mut src, 0, 0) }, 1);
    for flags in 1..=7 {
        assert_eq!(unsafe { cuMemHostAlloc(&mut src, SIZE, flags) }, 801);
    }
    assert_eq!(unsafe { cuMemHostAlloc(&mut src, SIZE, 8) }, 1);
    assert!(src.is_null());
    ok!(cuMemHostAlloc(&mut src, SIZE, 0));
    ok!(cuMemHostAlloc(&mut dst, SIZE, 0));
    assert_ne!(src, dst);
    assert_eq!(src.addr() % 4096, 0);
    assert_eq!(dst.addr() % 4096, 0);
    ok!(cuMemAlloc_v2(&mut device, SIZE));
    ok!(cuStreamCreate(&mut up, 0));
    ok!(cuStreamCreate(&mut down, 1));
    ok!(cuEventCreate(&mut event, 2));
    let input = std::array::from_fn::<_, SIZE, _>(|i| (i * 7) as u8);
    let src_offset = unsafe { src.byte_add(3) };
    let dst_offset = unsafe { dst.byte_add(5) };
    unsafe {
        src.cast::<[u8; SIZE]>().write(input);
        dst.cast::<u8>().write_bytes(0xA5, SIZE);
    }
    assert_eq!(unsafe { cuMemFreeHost(null_mut()) }, 1);
    assert_eq!(unsafe { cuMemFreeHost(src_offset) }, 1);
    assert_eq!(unsafe { cuMemFree_v2(src.addr() as u64) }, 1);
    assert_eq!(
        unsafe { cuMemFreeHost(without_provenance_mut(device as usize)) },
        1
    );
    assert_eq!(
        unsafe { cuMemcpyHtoDAsync_v2(device + 1, src, usize::MAX, up) },
        1
    );
    assert_eq!(
        unsafe { cuMemcpyDtoHAsync_v2(dst_offset, device, SIZE - 4, down) },
        1
    );
    assert_eq!(
        unsafe { cuMemcpyDtoHAsync_v2(dst, device + 1, SIZE, down) },
        1
    );
    assert_eq!(unsafe { cuMemcpyDtoHAsync_v2(dst, device, 0, ctx) }, 400);
    assert_eq!(
        unsafe { cuMemcpyHtoDAsync_v2(0, null_mut(), 0, without_provenance_mut(2)) },
        801
    );
    ok!(cuMemcpyHtoDAsync_v2(0, null_mut(), 0, up));
    ok!(cuMemcpyDtoHAsync_v2(null_mut(), u64::MAX, 0, null_mut()));

    ok!(cuCtxCreate_v2(&mut foreign, 0, test_device()));
    ok!(cuMemAlloc_v2(&mut other_device, SIZE));
    let (mut other_host, mut other_stream) = (null_mut(), null_mut());
    ok!(cuMemHostAlloc(&mut other_host, SIZE, 0));
    ok!(cuStreamCreate(&mut other_stream, 1));
    assert_eq!(unsafe { cuMemFreeHost(src) }, 1);
    assert_eq!(
        unsafe { cuMemcpyHtoDAsync_v2(other_device, src, 1, other_stream) },
        1
    );
    assert_eq!(
        unsafe { cuMemcpyDtoHAsync_v2(other_host, device, 1, other_stream) },
        1
    );
    assert_eq!(
        unsafe { cuMemcpyHtoDAsync_v2(other_device, other_host, 1, up) },
        400
    );
    ok!(cuCtxDestroy_v2(foreign));
    ok!(cuCtxSetCurrent(ctx));

    // An explicit blocking stream sees the legacy upload, and the nonblocking
    // consumer sees the producer through a recorded event.
    ok!(cuMemcpyHtoDAsync_v2(device + 1, src_offset, 29, null_mut()));
    ok!(cuEventRecord(event, up));
    ok!(cuStreamWaitEvent(down, event, 0));
    ok!(cuMemcpyDtoHAsync_v2(dst_offset, device + 1, 29, down));
    ok!(cuEventRecord(event, down));
    ok!(cuEventSynchronize(event));
    let mut expected = [0xA5; SIZE];
    expected[5..34].copy_from_slice(&input[3..32]);
    assert_eq!(unsafe { dst.cast::<[u8; SIZE]>().read() }, expected);

    // Several callers share a context, with independent allocations and streams.
    let ctx_key = ctx.addr();
    let worker = |byte| {
        ok!(cuCtxSetCurrent(without_provenance_mut(ctx_key)));
        let (mut host, mut stream, mut buffer) = (null_mut(), null_mut(), 0);
        ok!(cuMemHostAlloc(&mut host, 2 * SIZE, 0));
        ok!(cuMemAlloc_v2(&mut buffer, SIZE));
        ok!(cuStreamCreate(&mut stream, 1));
        let output = unsafe { host.byte_add(SIZE) };
        for value in byte..byte + 16 {
            unsafe { host.cast::<u8>().write_bytes(value, SIZE) };
            ok!(cuMemcpyHtoDAsync_v2(buffer, host, SIZE, stream));
            ok!(cuMemcpyDtoHAsync_v2(output, buffer, SIZE, stream));
            ok!(cuStreamSynchronize(stream));
            assert_eq!(unsafe { output.cast::<[u8; SIZE]>().read() }, [value; SIZE]);
        }
        ok!(cuStreamDestroy_v2(stream));
        ok!(cuMemFreeHost(host));
        ok!(cuMemFree_v2(buffer));
    };
    std::thread::scope(|scope| {
        for byte in 1..=4u8 {
            scope.spawn(move || worker(byte));
        }
    });

    // Free drains queued uses, including work on a destroyed stream.
    ok!(cuMemcpyHtoDAsync_v2(device, src, SIZE, up));
    ok!(cuMemcpyDtoHAsync_v2(dst, device, SIZE, up));
    match std::env::var("CUDA_FOR_INTEL_PINNED_COMPLETION").as_deref() {
        Ok("stream") => ok!(cuStreamSynchronize(up)),
        Ok("event") => {
            ok!(cuEventRecord(event, up));
            ok!(cuEventSynchronize(event));
        }
        Ok("context") => ok!(cuCtxSynchronize()),
        Ok("retire") | Err(std::env::VarError::NotPresent) => {}
        value => panic!("unknown pinned copy control {value:?}"),
    }
    ok!(cuStreamDestroy_v2(up));
    assert_eq!(unsafe { cuMemcpyHtoDAsync_v2(device, src, SIZE, up) }, 400);
    ok!(cuMemFreeHost(src));
    assert_eq!(unsafe { cuMemFreeHost(src) }, 1);
    assert_eq!(unsafe { cuMemcpyHtoDAsync_v2(device, src, SIZE, down) }, 1);
    let retired_output = unsafe { dst.cast::<[u8; SIZE]>().read() };
    if retired_output != input {
        // Read after capturing the failing host snapshot so the diagnostic copy
        // cannot hide a premature completion or host visibility failure.
        let mut device_snapshot = [0; SIZE];
        let result = unsafe { cuMemcpyDtoH_v2(device_snapshot.as_mut_ptr().cast(), device, SIZE) };
        eprintln!(
            "retired stream diagnostic: blocking DtoH result={result}, device_matches_input={}, device_matches_stale_output={}",
            device_snapshot == input,
            device_snapshot == retired_output
        );
    }
    assert_eq!(retired_output, input);
    for _ in 0..64 {
        let mut temporary = null_mut();
        ok!(cuMemHostAlloc(&mut temporary, SIZE, 0));
        ok!(cuMemFreeHost(temporary));
    }
    // Legacy work waits for a blocking stream's upload. A device free also
    // waits for outstanding DtoH copies before allowing address reuse.
    ok!(cuStreamCreate(&mut up, 0));
    ok!(cuMemcpyHtoDAsync_v2(device, dst, SIZE, up));
    let mut actual = [0; SIZE];
    ok!(cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), device, SIZE));
    assert_eq!(actual, input);
    let mut legacy = null_mut();
    ok!(cuMemHostAlloc(&mut legacy, SIZE, 0));
    unsafe {
        dst.cast::<u8>().write_bytes(0x39, SIZE);
        legacy.cast::<u8>().write_bytes(0, SIZE);
    }
    ok!(cuMemcpyHtoDAsync_v2(device, dst, SIZE, up));
    ok!(cuMemcpyDtoHAsync_v2(legacy, device, SIZE, null_mut()));
    ok!(cuEventRecord(event, null_mut()));
    ok!(cuEventSynchronize(event));
    assert_eq!(unsafe { legacy.cast::<[u8; SIZE]>().read() }, [0x39; SIZE]);
    ok!(cuMemFreeHost(legacy));

    unsafe { dst.cast::<u8>().write_bytes(0, SIZE) };
    ok!(cuMemcpyDtoHAsync_v2(dst, device, SIZE, down));
    ok!(cuCtxSynchronize());
    assert_eq!(unsafe { dst.cast::<[u8; SIZE]>().read() }, [0x39; SIZE]);
    unsafe { dst.cast::<u8>().write_bytes(0, SIZE) };
    ok!(cuMemcpyDtoHAsync_v2(dst, device, SIZE, down));
    ok!(cuMemFree_v2(device));
    assert_eq!(unsafe { dst.cast::<[u8; SIZE]>().read() }, [0x39; SIZE]);

    // Context destruction owns both allocations even with queued work.
    ok!(cuMemAlloc_v2(&mut device, SIZE));
    ok!(cuMemcpyHtoDAsync_v2(device, dst, SIZE, down));
    ok!(cuStreamDestroy_v2(down));
    ok!(cuCtxDestroy_v2(ctx));
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn pinned_partial_then_full_copy() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxSynchronize(),
        cuMemHostAlloc(*mut Handle, usize, u32),
        cuMemFreeHost(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoDAsync_v2(u64, *const c_void, usize, Handle),
        cuMemcpyDtoHAsync_v2(Handle, u64, usize, Handle),
        cuMemcpyDtoH_v2(Handle, u64, usize),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
        cuStreamSynchronize(Handle),
        cuEventCreate(*mut Handle, u32),
        cuEventRecord(Handle, Handle),
        cuEventSynchronize(Handle),
    );
    let (mut ctx, mut src, mut dst, mut up, mut down, mut event) = (
        null_mut(),
        null_mut(),
        null_mut(),
        null_mut(),
        null_mut(),
        null_mut(),
    );
    let mut device = 0;
    const SIZE: usize = 37;
    ok!(cuInit(0));
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    ok!(cuMemHostAlloc(&mut src, SIZE, 0));
    ok!(cuMemHostAlloc(&mut dst, SIZE, 0));
    ok!(cuMemAlloc_v2(&mut device, SIZE));
    ok!(cuStreamCreate(&mut up, 0));
    ok!(cuStreamCreate(&mut down, 1));
    ok!(cuEventCreate(&mut event, 2));
    let input = std::array::from_fn::<_, SIZE, _>(|i| (i * 7) as u8);
    unsafe {
        src.cast::<[u8; SIZE]>().write(input);
        dst.cast::<u8>().write_bytes(0xA5, SIZE);
    }

    ok!(cuMemcpyHtoDAsync_v2(device, src, SIZE, down));
    ok!(cuMemcpyDtoHAsync_v2(dst.byte_add(5), device + 3, 29, down));
    ok!(cuEventRecord(event, down));
    ok!(cuEventSynchronize(event));
    let mut partial = [0xA5; SIZE];
    partial[5..34].copy_from_slice(&input[3..32]);
    assert_eq!(unsafe { dst.cast::<[u8; SIZE]>().read() }, partial);

    ok!(cuMemcpyHtoDAsync_v2(device, src, SIZE, up));
    ok!(cuMemcpyDtoHAsync_v2(dst, device, SIZE, up));
    match std::env::var("CUDA_FOR_INTEL_PINNED_COMPLETION").as_deref() {
        Ok("stream") => ok!(cuStreamSynchronize(up)),
        Ok("event") => {
            ok!(cuEventRecord(event, up));
            ok!(cuEventSynchronize(event));
        }
        Ok("context") => ok!(cuCtxSynchronize()),
        Ok("retire") | Err(std::env::VarError::NotPresent) => {}
        value => panic!("unknown pinned copy control {value:?}"),
    }
    ok!(cuStreamDestroy_v2(up));
    ok!(cuMemFreeHost(src));
    let snapshot = unsafe { dst.cast::<[u8; SIZE]>().read() };
    if snapshot != input {
        let mut blocking = [0; SIZE];
        ok!(cuMemcpyDtoH_v2(blocking.as_mut_ptr().cast(), device, SIZE));
        eprintln!(
            "partial/full diagnostic: stale_partial={}, blocking_matches_input={}",
            snapshot == partial,
            blocking == input
        );
    }
    assert_eq!(snapshot, input);
    ok!(cuCtxDestroy_v2(ctx));
}
