use super::{Handle, c_char, library, ok, symbols, test_device};
use std::ffi::CStr;
use std::ptr::{from_mut, null, null_mut};

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn resolved_primary_context_runs_multiple_module_entries() {
    let dll = library();
    symbols!(dll;
        cuDriverGetVersion(*mut i32),
        cuGetProcAddress_v2(*const c_char, *mut Handle, i32, u64, *mut i32),
        cuDevicePrimaryCtxGetState(i32, *mut u32, *mut i32),
        cuDevicePrimaryCtxRelease_v2(i32),
        cuCtxGetCurrent(*mut Handle),
        cuCtxSetCurrent(Handle),
        cuDeviceGetPCIBusId(*mut c_char, i32, i32),
        cuDeviceGetAttribute(*mut i32, i32, i32),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyDtoH_v2(Handle, u64, usize),
        cuModuleLoadData(*mut Handle, *const std::ffi::c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuModuleUnload(Handle),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    let mut version = 0;
    ok!(cuDriverGetVersion(&mut version));
    assert_eq!(version, 13000);
    let resolve = |name: &CStr| {
        let mut result = null_mut();
        let mut status = -1;
        ok!(cuGetProcAddress_v2(
            name.as_ptr(),
            &mut result,
            version,
            1,
            &mut status
        ));
        assert_eq!(status, 0);
        assert!(!result.is_null());
        result
    };
    let cuInit: unsafe extern "system" fn(u32) -> i32 =
        unsafe { std::mem::transmute(resolve(c"cuInit")) };
    let cuDevicePrimaryCtxRetain: unsafe extern "system" fn(*mut Handle, i32) -> i32 =
        unsafe { std::mem::transmute(resolve(c"cuDevicePrimaryCtxRetain")) };
    ok!(cuInit(0));

    let dev = test_device();
    let mut pci = [0; 32];
    match unsafe { cuDeviceGetPCIBusId(pci.as_mut_ptr(), pci.len() as i32, dev) } {
        0 => {
            let pci = unsafe { CStr::from_ptr(pci.as_ptr()) }.to_str().unwrap();
            let parts: Vec<_> = pci.split([':', '.']).collect();
            assert_eq!(parts.len(), 4);
            for (part, attr) in parts.iter().zip([50, 33, 34]) {
                let mut value = -1;
                ok!(cuDeviceGetAttribute(&mut value, attr, dev));
                assert_eq!(i32::from_str_radix(part, 16).unwrap(), value);
            }
            println!("Level Zero PCI address {pci}");
        }
        801 => println!("SKIP PCI subcheck: Level Zero PCI properties unavailable"),
        result => panic!("unexpected cuDeviceGetPCIBusId status {result}"),
    }
    let mut integrated = -1;
    ok!(cuDeviceGetAttribute(&mut integrated, 18, dev));
    assert!(matches!(integrated, 0 | 1));
    assert_eq!(
        unsafe { cuDeviceGetAttribute(&mut integrated, 75, dev) },
        801
    );

    let (mut flags, mut active) = (u32::MAX, -1);
    let (mut context, mut current) = (null_mut(), null_mut());
    ok!(cuDevicePrimaryCtxRetain(&mut context, dev));
    ok!(cuCtxGetCurrent(&mut current));
    assert!(current.is_null());
    ok!(cuDevicePrimaryCtxGetState(dev, &mut flags, &mut active));
    assert_eq!((flags, active), (0, 1));
    ok!(cuCtxSetCurrent(context));

    let mut module = null_mut();
    let mut image = include_bytes!("multi_entry.ptx").to_vec();
    image.push(0);
    ok!(cuModuleLoadData(&mut module, image.as_ptr().cast()));
    let (mut marker, mut write, mut repeated) = (null_mut(), null_mut(), null_mut());
    ok!(cuModuleGetFunction(&mut marker, module, c"marker".as_ptr()));
    ok!(cuModuleGetFunction(
        &mut write,
        module,
        c"write_value".as_ptr()
    ));
    ok!(cuModuleGetFunction(
        &mut repeated,
        module,
        c"marker".as_ptr()
    ));
    assert_eq!(marker, repeated);
    assert_ne!(marker, write);
    assert_eq!(
        unsafe { cuModuleGetFunction(&mut repeated, module, null()) },
        1
    );

    let mut address = 0;
    let mut value = 0x1234_5678u32;
    let mut actual = 0u32;
    ok!(cuMemAlloc_v2(&mut address, size_of::<u32>()));
    let mut args = [from_mut(&mut address).cast(), from_mut(&mut value).cast()];
    ok!(cuLaunchKernel(
        marker,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        null_mut(),
        null_mut(),
        null_mut()
    ));
    ok!(cuLaunchKernel(
        write,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        null_mut(),
        args.as_mut_ptr(),
        null_mut()
    ));
    ok!(cuModuleUnload(module)); // Retires both handles and drains the submitted launches.
    assert_eq!(
        unsafe {
            cuLaunchKernel(
                marker,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                null_mut(),
                null_mut(),
                null_mut(),
            )
        },
        400
    );
    ok!(cuMemcpyDtoH_v2(
        from_mut(&mut actual).cast(),
        address,
        size_of::<u32>()
    ));
    assert_eq!(actual, value);
    ok!(cuDevicePrimaryCtxRelease_v2(dev)); // Also frees address.
    ok!(cuCtxSetCurrent(null_mut()));
    ok!(cuDevicePrimaryCtxGetState(dev, &mut flags, &mut active));
    assert_eq!((flags, active), (0, 0));
}
