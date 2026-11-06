use super::{Handle, library, ok, symbols, test_device};
use std::ptr::null_mut;
use std::time::{Duration, Instant};

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn event_linked_transfers_make_progress_without_synchronize() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemHostAlloc(*mut Handle, usize, u32),
        cuMemcpyHtoDAsync_v2(u64, Handle, usize, Handle),
        cuMemcpyDtoHAsync_v2(Handle, u64, usize, Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
        cuStreamQuery(Handle),
        cuStreamWaitEvent(Handle, Handle, u32),
        cuEventCreate(*mut Handle, u32),
        cuEventRecord(Handle, Handle),
        cuEventDestroy_v2(Handle),
    );
    ok!(cuInit(0));
    let (mut context, mut producer, mut consumer, mut uploaded) =
        (null_mut(), null_mut(), null_mut(), null_mut());
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    ok!(cuStreamCreate(&mut producer, 1));
    ok!(cuStreamCreate(&mut consumer, 1));
    ok!(cuEventCreate(&mut uploaded, 2));

    let (mut src, mut dst, mut device) = (null_mut(), null_mut(), 0);
    ok!(cuMemHostAlloc(&mut src, 1, 0));
    ok!(cuMemHostAlloc(&mut dst, 3, 0));
    ok!(cuMemAlloc_v2(&mut device, 1));
    unsafe {
        src.cast::<u8>().write(42);
        dst.cast::<[u8; 3]>().write([0xa5; 3]);
    }
    ok!(cuMemcpyHtoDAsync_v2(device, src, 1, producer));
    ok!(cuEventRecord(uploaded, producer));
    ok!(cuStreamWaitEvent(consumer, uploaded, 0));
    let result = unsafe { dst.cast::<u8>().add(1).cast() };
    ok!(cuMemcpyDtoHAsync_v2(result, device, 1, consumer));
    ok!(cuEventDestroy_v2(uploaded));
    ok!(cuStreamDestroy_v2(producer));

    // Query from a thread with no current context. A wait or sync could mask
    // missing forward progress by causing buffered commands to be submitted.
    std::thread::scope(|scope| {
        let key = consumer.addr();
        scope
            .spawn(move || {
                let deadline = Instant::now() + Duration::from_secs(10);
                loop {
                    match unsafe { cuStreamQuery(std::ptr::without_provenance_mut(key)) } {
                        0 => break,
                        600 if Instant::now() < deadline => {
                            std::thread::sleep(Duration::from_millis(1));
                        }
                        status => panic!("consumer failed to complete, CUDA status {status}"),
                    }
                }
            })
            .join()
            .unwrap();
    });
    assert_eq!(unsafe { dst.cast::<[u8; 3]>().read() }, [0xa5, 42, 0xa5]);
    ok!(cuCtxDestroy_v2(context));
}
