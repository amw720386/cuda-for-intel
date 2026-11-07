use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn store_u8_and_v2_u16_roundtrip() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
        cuCtxSynchronize(),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let ptx = std::ffi::CString::new(include_str!("../../examples/store_u8_probe.ptx")).unwrap();
    let mut module = null_mut();
    let load = unsafe { cuModuleLoadData(&mut module, ptx.as_ptr().cast()) };
    assert_eq!(load, 0, "ModuleLoad -> {load}");
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, c"store_u8_probe".as_ptr()));
    const N: usize = 32;
    let x: Vec<f32> = (0..N).map(|i| (i as f32) / 32.0).collect();
    let mut dx = 0u64;
    let mut dy = 0u64;
    ok!(cuMemAlloc_v2(&mut dx, N * 4));
    ok!(cuMemAlloc_v2(&mut dy, N + 4));
    ok!(cuMemcpyHtoD_v2(dx, x.as_ptr().cast(), N * 4));
    let zeros = vec![0u8; N + 4];
    ok!(cuMemcpyHtoD_v2(dy, zeros.as_ptr().cast(), zeros.len()));
    let mut n = N as u32;
    let mut params = [
        from_mut(&mut dx).cast::<c_void>(),
        from_mut(&mut dy).cast::<c_void>(),
        from_mut(&mut n).cast::<c_void>(),
    ];
    assert_eq!(unsafe { cuLaunchKernel(kernel, 1,1,1, N as u32,1,1, 0, null_mut(), params.as_mut_ptr(), null_mut()) }, 0);
    let sync = unsafe { cuCtxSynchronize() };
    assert_eq!(sync, 0, "sync -> {sync}");
    let mut got = vec![0u8; N + 4];
    ok!(cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), dy, got.len()));
    eprintln!("hdr={:02x?} qs={:?}", &got[..4], &got[4..]);
    assert_ne!(got[4..].iter().filter(|&&b| b != 0).count(), 0, "qs all zero");
    ok!(cuCtxDestroy_v2(context));
}
