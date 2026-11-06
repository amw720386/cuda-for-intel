//! Opt-in tests against the built CUDA DLL and an actual Intel GPU.
#![cfg(windows)]

use libloading::Library;
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;

type Handle = *mut c_void;

macro_rules! symbols {
    ($library:ident; $( $name:ident($($arg:ty),*) ),+ $(,)?) => {
        $(
            let $name = unsafe {
                $library.get::<unsafe extern "system" fn($($arg),*) -> i32>(
                    concat!(stringify!($name), "\0").as_bytes(),
                )
            }.expect(stringify!($name));
        )+
    };
}

macro_rules! ok {
    ($name:ident($($arg:expr),* $(,)?)) => {
        assert_eq!(unsafe { $name($($arg),*) }, 0, "{}", stringify!($name))
    };
}

fn library() -> Library {
    let exe = std::env::current_exe().unwrap();
    let dll = exe.parent().unwrap().parent().unwrap().join("nvcuda.dll");
    unsafe { Library::new(&dll) }.unwrap_or_else(|error| {
        panic!(
            "cannot load {} (run cargo build first): {error}",
            dll.display()
        )
    })
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn device_memory_round_trip() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuDeviceGetCount(*mut i32),
        cuDeviceGetName(*mut c_char, i32, i32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxGetCurrent(*mut Handle),
        cuCtxSetCurrent(Handle),
        cuCtxSynchronize(),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
    );
    ok!(cuInit(0));
    let mut count = 0;
    ok!(cuDeviceGetCount(&mut count));
    assert!(count > 0);
    let mut name = [0; 256];
    ok!(cuDeviceGetName(name.as_mut_ptr(), name.len() as i32, 0));
    println!(
        "Device: {}",
        unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_string_lossy()
    );

    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, 0));
    let mut current = null_mut();
    ok!(cuCtxGetCurrent(&mut current));
    assert_eq!(current, context);
    std::thread::scope(|scope| {
        scope.spawn(|| {
            let mut current = null_mut();
            ok!(cuCtxGetCurrent(&mut current));
            assert!(current.is_null());
        });
    });
    ok!(cuCtxSetCurrent(null_mut()));
    assert_eq!(unsafe { cuCtxSynchronize() }, 201);
    ok!(cuCtxSetCurrent(context));

    let input = [2u8, 5, 8, 13, 21];
    let mut output = [0u8; 5];
    let mut device = 0;
    ok!(cuMemAlloc_v2(&mut device, input.len() + 2));
    ok!(cuMemcpyHtoD_v2(
        device + 2,
        input.as_ptr().cast(),
        input.len()
    ));
    assert_eq!(
        unsafe { cuMemcpyHtoD_v2(device + 3, input.as_ptr().cast(), input.len()) },
        1
    );
    ok!(cuMemcpyDtoH_v2(
        output.as_mut_ptr().cast(),
        device + 2,
        output.len()
    ));
    ok!(cuCtxSynchronize());
    assert_eq!(output, input);
    ok!(cuMemFree_v2(device));
    assert_eq!(unsafe { cuMemFree_v2(device) }, 1);
    ok!(cuCtxDestroy_v2(context));
    assert_eq!(unsafe { cuCtxSetCurrent(context) }, 201);
}
