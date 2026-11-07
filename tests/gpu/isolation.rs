use super::isolation_capability::supports_isolation;
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{null_mut, without_provenance_mut};
use std::sync::Barrier;

const SOURCE: &std::ffi::CStr = c"
    .version 4.1
    .target sm_52
    .address_size 64
    .visible .entry store(.param .u64 .ptr .global dst, .param .u32 value) {
        .reg .b64 %ptr;
        .reg .b32 %val;
        ld.param.u64 %ptr, [dst];
        ld.param.u32 %val, [value];
        st.global.b32 [%ptr], %val;
        ret;
    }
";

#[test]
#[ignore = "requires an Intel Level Zero GPU with SPIR-V and pinned copy support"]
fn same_device_context_isolation() {
    check_isolation(test_device());
}

#[test]
#[ignore = "requires two Intel Level Zero root GPUs with SPIR-V and pinned copy support"]
fn multi_device_context_isolation() {
    let peer = nvcuda::intel_devices()
        .unwrap()
        .iter()
        .enumerate()
        .find(|(ordinal, device)| *ordinal as i32 != test_device() && supports_isolation(device))
        .map(|(ordinal, _)| ordinal as i32);
    if let Some(peer) = peer {
        check_isolation(peer);
    } else {
        println!("SKIP: no second Intel Level Zero root GPU supports the isolation workload");
    }
}

#[derive(Default)]
struct Resources {
    context: Handle,
    stream: Handle,
    event: Handle,
    module: Handle,
    function: Handle,
    device: u64,
    host: Handle,
}

#[allow(non_snake_case)]
fn check_isolation(peer: i32) {
    const INVALID_VALUE: i32 = 1;
    const INVALID_HANDLE: i32 = 400;
    const BYTES: usize = size_of::<u32>();
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuDeviceGet(*mut i32, i32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxGetCurrent(*mut Handle),
        cuCtxSetCurrent(Handle),
        cuCtxGetDevice(*mut i32),
        cuCtxSynchronize(),
        cuCtxDestroy_v2(Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamQuery(Handle),
        cuStreamSynchronize(Handle),
        cuEventCreate(*mut Handle, u32),
        cuEventRecord(Handle, Handle),
        cuEventSynchronize(Handle),
        cuStreamWaitEvent(Handle, Handle, u32),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuModuleUnload(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemHostAlloc(*mut Handle, usize, u32),
        cuMemFreeHost(Handle),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyHtoDAsync_v2(u64, *const c_void, usize, Handle),
        cuMemcpyDtoHAsync_v2(Handle, u64, usize, Handle),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let create = |ordinal| {
        let mut device = -1;
        ok!(cuDeviceGet(&mut device, ordinal));
        assert_eq!(device, ordinal);
        let mut r = Resources::default();
        ok!(cuCtxCreate_v2(&mut r.context, 0, device));
        ok!(cuCtxGetDevice(&mut device));
        assert_eq!(device, ordinal);
        ok!(cuStreamCreate(&mut r.stream, 1));
        ok!(cuEventCreate(&mut r.event, 2));
        ok!(cuModuleLoadData(&mut r.module, SOURCE.as_ptr().cast()));
        ok!(cuModuleGetFunction(
            &mut r.function,
            r.module,
            c"store".as_ptr()
        ));
        ok!(cuMemAlloc_v2(&mut r.device, BYTES));
        ok!(cuMemHostAlloc(&mut r.host, BYTES, 0));
        r
    };
    let launch = |function, stream, pointer: u64, value: u32| {
        let mut args = [
            std::ptr::from_ref(&pointer).cast_mut().cast(),
            std::ptr::from_ref(&value).cast_mut().cast(),
        ];
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
                args.as_mut_ptr(),
                null_mut(),
            )
        }
    };
    let enqueue = |r: &Resources, value| {
        assert_eq!(launch(r.function, r.stream, r.device, value), 0);
        ok!(cuEventRecord(r.event, r.stream));
        ok!(cuMemcpyDtoHAsync_v2(r.host, r.device, BYTES, r.stream));
    };

    let a = create(test_device());
    enqueue(&a, 17);
    let b = create(peer);
    assert_eq!(unsafe { cuMemFree_v2(a.device) }, INVALID_VALUE);
    assert_eq!(unsafe { cuMemFreeHost(a.host) }, INVALID_VALUE);
    assert_eq!(
        unsafe { cuMemcpyHtoD_v2(a.device, (&0u32 as *const u32).cast(), BYTES) },
        INVALID_VALUE
    );
    assert_eq!(
        unsafe { cuMemcpyDtoHAsync_v2(b.host, a.device, BYTES, b.stream) },
        INVALID_VALUE
    );
    assert_eq!(
        unsafe { cuMemcpyDtoHAsync_v2(b.host, b.device, BYTES, a.stream) },
        INVALID_HANDLE
    );
    assert_eq!(launch(a.function, b.stream, b.device, 0), INVALID_HANDLE);
    assert_eq!(launch(b.function, a.stream, b.device, 0), INVALID_HANDLE);
    assert_eq!(launch(b.function, b.stream, a.device, 0), INVALID_VALUE);
    let mut foreign_function = null_mut();
    assert_eq!(
        unsafe { cuModuleGetFunction(&mut foreign_function, a.module, c"store".as_ptr()) },
        INVALID_HANDLE
    );
    assert_eq!(unsafe { cuModuleUnload(a.module) }, INVALID_HANDLE);
    assert_eq!(unsafe { cuEventRecord(a.event, b.stream) }, INVALID_HANDLE);
    assert_eq!(unsafe { cuEventRecord(b.event, a.stream) }, INVALID_HANDLE);
    assert_eq!(
        unsafe { cuStreamWaitEvent(b.stream, a.event, 0) },
        INVALID_HANDLE
    );
    assert_eq!(unsafe { cuEventSynchronize(a.event) }, INVALID_HANDLE);

    // Explicit stream synchronization resolves the stream's owning context.
    ok!(cuStreamSynchronize(a.stream));
    ok!(cuStreamQuery(a.stream));
    assert_eq!(unsafe { a.host.cast::<u32>().read() }, 17);
    enqueue(&b, 29);

    let pair = [(a.context.addr(), test_device()), (b.context.addr(), peer)];
    let barrier = Barrier::new(2);
    std::thread::scope(|scope| {
        let cuCtxGetCurrent = &cuCtxGetCurrent;
        let cuCtxSetCurrent = &cuCtxSetCurrent;
        let cuCtxGetDevice = &cuCtxGetDevice;
        for order in [[0, 1], [1, 0]] {
            let barrier = &barrier;
            scope.spawn(move || {
                barrier.wait();
                let mut current = null_mut();
                ok!(cuCtxGetCurrent(&mut current));
                assert!(current.is_null());
                for index in order {
                    let (key, ordinal) = pair[index];
                    let handle = without_provenance_mut(key);
                    ok!(cuCtxSetCurrent(handle));
                    ok!(cuCtxGetCurrent(&mut current));
                    assert_eq!(current, handle);
                    let mut device = -1;
                    ok!(cuCtxGetDevice(&mut device));
                    assert_eq!(device, ordinal);
                }
            });
        }
    });

    ok!(cuCtxDestroy_v2(a.context));
    let mut current = null_mut();
    ok!(cuCtxGetCurrent(&mut current));
    assert_eq!(current, b.context);
    assert_eq!(unsafe { cuStreamQuery(a.stream) }, INVALID_HANDLE);
    ok!(cuCtxSynchronize());
    assert_eq!(unsafe { b.host.cast::<u32>().read() }, 29);
    enqueue(&b, 53);
    ok!(cuCtxSynchronize());
    assert_eq!(unsafe { b.host.cast::<u32>().read() }, 53);
    ok!(cuCtxDestroy_v2(b.context));
    println!(
        "Context isolation verified on CUDA ordinals {} and {peer}",
        test_device()
    );
}
