use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn ctaid_yz_are_zero_for_1d_grid() {
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
    let ptx = std::ffi::CString::new(include_str!("../../examples/ctaid_probe.ptx")).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, c"ctaid_probe".as_ptr()));
    let mut dst = 0u64;
    ok!(cuMemAlloc_v2(&mut dst, 24));
    let zeros = [0u32; 6];
    ok!(cuMemcpyHtoD_v2(dst, zeros.as_ptr().cast(), 24));
    let mut params = [from_mut(&mut dst).cast::<c_void>()];
    for (gx, gy, gz) in [(1u32, 1u32, 1u32), (2, 3, 4)] {
        let zeros = [0u32; 6];
        ok!(cuMemcpyHtoD_v2(dst, zeros.as_ptr().cast(), 24));
        ok!(cuLaunchKernel(
            kernel,
            gx,
            gy,
            gz,
            32,
            1,
            1,
            0,
            null_mut(),
            params.as_mut_ptr(),
            null_mut(),
        ));
        let sync = unsafe { cuCtxSynchronize() };
        assert_eq!(sync, 0, "sync grid=({gx},{gy},{gz}) -> {sync}");
        let mut got = [0u32; 6];
        ok!(cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), dst, 24));
        eprintln!(
            "grid=({gx},{gy},{gz}) ctaid=({},{},{}) nctaid=({},{},{})",
            got[0], got[1], got[2], got[3], got[4], got[5]
        );
        assert_eq!(got[3], gx, "nctaid.x");
        assert_eq!(got[4], gy, "nctaid.y");
        assert_eq!(got[5], gz, "nctaid.z");
        assert!(
            got[0] < gx && got[1] < gy && got[2] < gz,
            "ctaid out of range: {got:?}"
        );
    }
    ok!(cuCtxDestroy_v2(context));
}
