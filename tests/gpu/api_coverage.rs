//! Driver entry points added for API breadth, checked on hardware: context limits and
//! cache configuration, stream queries, host callbacks, 2-D memset, device properties and
//! cooperative launch admission.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};
use std::sync::atomic::{AtomicU32, Ordering};

static CALLBACKS: AtomicU32 = AtomicU32::new(0);

unsafe extern "system" fn count_callback(_stream: Handle, status: i32, data: *mut c_void) {
    assert_eq!(status, 0);
    CALLBACKS.fetch_add(data as usize as u32, Ordering::SeqCst);
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn context_limits_streams_and_properties() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxGetLimit(*mut usize, i32),
        cuCtxSetLimit(i32, usize),
        cuCtxGetCacheConfig(*mut i32),
        cuCtxSetCacheConfig(i32),
        cuCtxGetId(Handle, *mut u64),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
        cuStreamGetFlags(Handle, *mut u32),
        cuStreamGetPriority(Handle, *mut i32),
        cuStreamGetId(Handle, *mut u64),
        cuStreamGetDevice(Handle, *mut i32),
        cuStreamGetCtx_v2(Handle, *mut Handle, *mut Handle),
        cuStreamAddCallback(Handle, unsafe extern "system" fn(Handle, i32, *mut c_void), *mut c_void, u32),
        cuStreamSynchronize(Handle),
        cuDeviceComputeCapability(*mut i32, *mut i32, i32),
        cuDeviceGetAttribute(*mut i32, i32, i32),
        cuDeviceGetProperties(*mut i32, i32),
        cuProfilerStart(),
        cuProfilerStop(),
        cuGetErrorName(i32, *mut *const c_char),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));

    let mut limit = 0usize;
    ok!(cuCtxGetLimit(&mut limit, 0));
    assert!(limit > 0, "stack size limit");
    ok!(cuCtxSetLimit(0, 2048));
    ok!(cuCtxGetLimit(&mut limit, 0));
    assert_eq!(limit, 2048);
    assert_eq!(unsafe { cuCtxGetLimit(&mut limit, 99) }, 1);
    let mut cache = -1;
    ok!(cuCtxSetCacheConfig(1));
    ok!(cuCtxGetCacheConfig(&mut cache));
    assert_eq!(cache, 1);
    let mut id = 0u64;
    ok!(cuCtxGetId(null_mut(), &mut id));
    assert_ne!(id, 0);

    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 1));
    let (mut flags, mut priority, mut sid, mut device) = (9u32, 9, 0u64, -1);
    ok!(cuStreamGetFlags(stream, &mut flags));
    assert_eq!(flags, 1);
    ok!(cuStreamGetPriority(stream, &mut priority));
    assert_eq!(priority, 0);
    ok!(cuStreamGetId(stream, &mut sid));
    ok!(cuStreamGetDevice(stream, &mut device));
    assert_eq!(device, test_device());
    let mut owner = null_mut();
    ok!(cuStreamGetCtx_v2(stream, &mut owner, null_mut()));
    assert_eq!(owner, context);

    CALLBACKS.store(0, Ordering::SeqCst);
    ok!(cuStreamAddCallback(stream, count_callback, 3usize as *mut c_void, 0));
    ok!(cuStreamAddCallback(stream, count_callback, 4usize as *mut c_void, 0));
    ok!(cuStreamSynchronize(stream));
    assert_eq!(CALLBACKS.load(Ordering::SeqCst), 7, "both callbacks ran in order before sync returned");

    let (mut major, mut minor) = (0, 0);
    ok!(cuDeviceComputeCapability(&mut major, &mut minor, test_device()));
    let (mut a, mut b) = (0, 0);
    ok!(cuDeviceGetAttribute(&mut a, 75, test_device()));
    ok!(cuDeviceGetAttribute(&mut b, 76, test_device()));
    assert_eq!((major, minor), (a, b));
    let mut prop = [0i32; 21];
    ok!(cuDeviceGetProperties(prop.as_mut_ptr(), test_device()));
    let mut threads = 0;
    ok!(cuDeviceGetAttribute(&mut threads, 1, test_device()));
    assert_eq!(prop[0], threads);

    ok!(cuProfilerStart());
    ok!(cuProfilerStop());
    let mut name = std::ptr::null();
    ok!(cuGetErrorName(720, &mut name));
    assert_eq!(unsafe { std::ffi::CStr::from_ptr(name) }.to_str().unwrap(), "CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE");

    ok!(cuStreamDestroy_v2(stream));
    ok!(cuCtxDestroy_v2(context));
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn memset_2d_writes_only_the_requested_columns() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemsetD8_v2(u64, u8, usize),
        cuMemsetD2D8_v2(u64, usize, u8, usize, usize),
        cuMemsetD2D32_v2(u64, usize, u32, usize, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let mut buffer = 0u64;
    ok!(cuMemAlloc_v2(&mut buffer, 256));
    ok!(cuMemsetD8_v2(buffer, 0, 256));
    ok!(cuMemsetD2D8_v2(buffer, 64, 0xAB, 16, 4));
    let mut host = [0u8; 256];
    ok!(cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), buffer, 256));
    for row in 0..4 {
        assert!(host[row * 64..row * 64 + 16].iter().all(|&b| b == 0xAB), "row {row} filled");
        assert!(host[row * 64 + 16..row * 64 + 64].iter().all(|&b| b == 0), "row {row} padding untouched");
    }
    ok!(cuMemsetD2D32_v2(buffer, 64, 0x01020304, 4, 2));
    ok!(cuMemcpyDtoH_v2(host.as_mut_ptr().cast(), buffer, 256));
    assert_eq!(&host[0..4], &0x01020304u32.to_le_bytes());
    assert_eq!(&host[64..68], &0x01020304u32.to_le_bytes());
    assert_eq!(host[128], 0xAB, "third row keeps the earlier fill");
    assert_eq!(unsafe { cuMemsetD2D8_v2(buffer, 8, 1, 16, 2) }, 1, "width exceeds pitch");
    ok!(cuMemFree_v2(buffer));
    ok!(cuCtxDestroy_v2(context));
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn cooperative_launch_admits_resident_grids_only() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuCtxSynchronize(),
        cuLaunchCooperativeKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut *mut c_void),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let ptx = std::ffi::CString::new(include_str!("../../examples/ctaid_probe.ptx")).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, c"ctaid_probe".as_ptr()));
    let mut dst = 0u64;
    ok!(cuMemAlloc_v2(&mut dst, 24));
    let mut params = [from_mut(&mut dst).cast::<c_void>()];
    ok!(cuLaunchCooperativeKernel(kernel, 1, 1, 1, 32, 1, 1, 0, null_mut(), params.as_mut_ptr()));
    ok!(cuCtxSynchronize());
    // CUDA_ERROR_COOPERATIVE_LAUNCH_TOO_LARGE
    let result = unsafe { cuLaunchCooperativeKernel(kernel, 1 << 20, 1 << 10, 1, 32, 1, 1, 0, null_mut(), params.as_mut_ptr()) };
    assert_eq!(result, 720);
    let zero_block = unsafe { cuLaunchCooperativeKernel(kernel, 1, 1, 1, 0, 1, 1, 0, null_mut(), params.as_mut_ptr()) };
    assert_eq!(zero_block, 1);
    ok!(cuMemFree_v2(dst));
    ok!(cuCtxDestroy_v2(context));
}
