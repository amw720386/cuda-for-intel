use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn quantize_warp_ops_no_device_lost() {
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
    let ptx = std::ffi::CString::new(include_str!("../../examples/quantize_warp_ops.ptx")).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(
        &mut kernel,
        module,
        c"quantize_warp_ops".as_ptr()
    ));
    const N: usize = 32;
    let mut x = vec![0.0f32; N];
    for (i, v) in x.iter_mut().enumerate() {
        *v = (i as f32) * 0.01 - 0.1;
    }
    let mut dx = 0u64;
    let mut dy = 0u64;
    ok!(cuMemAlloc_v2(&mut dx, N * 4));
    ok!(cuMemAlloc_v2(&mut dy, 36));
    ok!(cuMemcpyHtoD_v2(dx, x.as_ptr().cast(), N * 4));
    let zeros = [0u8; 36];
    ok!(cuMemcpyHtoD_v2(dy, zeros.as_ptr().cast(), 36));
    let mut n = N as u32;
    let mut params = [
        from_mut(&mut dx).cast::<c_void>(),
        from_mut(&mut dy).cast::<c_void>(),
        from_mut(&mut n).cast::<c_void>(),
    ];
    assert_eq!(
        unsafe {
            cuLaunchKernel(
                kernel,
                1,
                1,
                1,
                N as u32,
                1,
                1,
                0,
                null_mut(),
                params.as_mut_ptr(),
                null_mut(),
            )
        },
        0
    );
    let sync = unsafe { cuCtxSynchronize() };
    assert_eq!(sync, 0, "cuCtxSynchronize -> {sync}");
    let mut got = [0u8; 36];
    ok!(cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), dy, 36));
    eprintln!("warp_ops hdr={:02x?} qs={:?}", &got[..4], &got[4..]);
    assert!(
        got[4..].iter().any(|&b| b != 0),
        "quantize_warp_ops wrote no qs"
    );
    ok!(cuCtxDestroy_v2(context));
}
