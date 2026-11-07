//! Module-scope `.global`/`.const` variables on hardware: initializers, address
//! lookup with cuModuleGetGlobal, host copies, kernel access and unload.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

const PTX: &std::ffi::CStr = c"
.version 7.0
.target sm_75
.address_size 64
.global .align 4 .u32 counter = 7;
.const .align 4 .u32 table[4] = {10, 20, 30, 40};
.global .align 4 .u32 result;
.entry combine(.param .u64 .ptr .global out)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, counter;
    ld.global.u32 %r1, [%rd3];
    ld.const.u32 %r2, [table+8];
    add.u32 %r3, %r1, %r2;
    st.global.u32 [result], %r3;
    st.global.u32 [%rd2], %r3;
    ret;
}
";

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn module_globals_initializers_copies_and_kernel_access() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleUnload(Handle),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuModuleGetGlobal_v2(*mut u64, *mut usize, Handle, *const c_char),
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
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, PTX.as_ptr().cast()));
    let global = |name: &std::ffi::CStr| {
        let (mut address, mut bytes) = (0u64, 0usize);
        ok!(cuModuleGetGlobal_v2(&mut address, &mut bytes, module, name.as_ptr()));
        (address, bytes)
    };
    let (counter, bytes) = global(c"counter");
    assert_eq!(bytes, 4);
    let (table, bytes) = global(c"table");
    assert_eq!(bytes, 16);
    let (result, _) = global(c"result");

    // Initializers are visible to the host before any kernel ran.
    let mut seen = [0u32; 4];
    ok!(cuMemcpyDtoH_v2(seen.as_mut_ptr().cast(), table, 16));
    assert_eq!(seen, [10, 20, 30, 40]);
    let mut one = 0u32;
    ok!(cuMemcpyDtoH_v2(from_mut(&mut one).cast(), counter, 4));
    assert_eq!(one, 7);

    // The host can overwrite both a .global and a .const object.
    let new_counter = 100u32;
    ok!(cuMemcpyHtoD_v2(counter, std::ptr::from_ref(&new_counter).cast(), 4));
    let new_table = [1u32, 2, 3, 4];
    ok!(cuMemcpyHtoD_v2(table, new_table.as_ptr().cast(), 16));

    let mut out = 0u64;
    ok!(cuMemAlloc_v2(&mut out, 4));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, c"combine".as_ptr()));
    let mut params = [from_mut(&mut out).cast::<c_void>()];
    ok!(cuLaunchKernel(kernel, 1, 1, 1, 1, 1, 1, 0, null_mut(), params.as_mut_ptr(), null_mut()));
    ok!(cuCtxSynchronize());

    // The kernel saw the host's values at the same addresses and wrote a global back.
    let (mut from_out, mut from_result) = (0u32, 0u32);
    ok!(cuMemcpyDtoH_v2(from_mut(&mut from_out).cast(), out, 4));
    ok!(cuMemcpyDtoH_v2(from_mut(&mut from_result).cast(), result, 4));
    assert_eq!((from_out, from_result), (103, 103));

    ok!(cuMemFree_v2(out));
    ok!(cuModuleUnload(module));
    ok!(cuCtxDestroy_v2(context));
}
