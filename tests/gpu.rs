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
    let key = context.addr();
    let (bound, wait_for_bind) = std::sync::mpsc::channel();
    let (destroyed, wait_for_destroy) = std::sync::mpsc::channel();
    std::thread::scope(|scope| {
        let set_current = &cuCtxSetCurrent;
        let synchronize = &cuCtxSynchronize;
        scope.spawn(move || {
            ok!(set_current(std::ptr::without_provenance_mut(key)));
            bound.send(()).unwrap();
            wait_for_destroy.recv().unwrap();
            assert_eq!(unsafe { synchronize() }, 709);
        });
        wait_for_bind.recv().unwrap();
        let result = unsafe { cuCtxDestroy_v2(context) };
        destroyed.send(()).unwrap();
        assert_eq!(result, 0);
    });
    assert_eq!(unsafe { cuCtxSetCurrent(context) }, 201);
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn compiled_ptx_vector_add() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxSynchronize(),
        cuModuleLoad(*mut Handle, *const c_char),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuModuleUnload(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, 0));

    let mut module = null_mut();
    let path = std::ffi::CString::new(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/examples/vector_add.ptx"
    ))
    .unwrap();
    ok!(cuModuleLoad(&mut module, path.as_ptr()));
    ok!(cuModuleUnload(module));
    module = null_mut();
    let elf = *b"\x7fELF";
    assert_eq!(
        unsafe { cuModuleLoadData(&mut module, elf.as_ptr().cast()) },
        200
    );
    let ptx = std::ffi::CString::new(include_str!("../examples/vector_add.ptx")).unwrap();
    assert_eq!(
        unsafe { cuModuleLoadData(&mut module, c"invalid PTX".as_ptr().cast()) },
        218
    );
    assert!(module.is_null());
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut kernel = null_mut();
    assert_eq!(
        unsafe { cuModuleGetFunction(&mut kernel, module, c"missing".as_ptr()) },
        500
    );
    ok!(cuModuleGetFunction(
        &mut kernel,
        module,
        c"vector_add".as_ptr()
    ));

    const N: usize = 1003;
    const CAPACITY: usize = 1024;
    const SENTINEL: f32 = -99_999.0;
    let mut left: Vec<f32> = (0..CAPACITY).map(|i| (i as f32 - 500.0) * 0.25).collect();
    let mut right: Vec<f32> = (0..CAPACITY).map(|i| (i as f32 % 73.0) * -0.5).collect();
    left[0] = f32::from_bits(1);
    right[0] = f32::from_bits(1);
    left[1] = 1.0;
    right[1] = f32::EPSILON / 2.0;
    let expected: Vec<f32> = (0..CAPACITY)
        .map(|i| if i < N { left[i] + right[i] } else { SENTINEL })
        .collect();
    let bytes = CAPACITY * size_of::<f32>();
    let (mut a, mut b, mut output) = (0u64, 0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut a, bytes));
    ok!(cuMemAlloc_v2(&mut b, bytes));
    ok!(cuMemAlloc_v2(&mut output, bytes));
    ok!(cuMemcpyHtoD_v2(a, left.as_ptr().cast(), bytes));
    ok!(cuMemcpyHtoD_v2(b, right.as_ptr().cast(), bytes));

    let mut count = N as u32;
    let mut args = [
        (&mut a as *mut u64).cast(),
        (&mut b as *mut u64).cast(),
        (&mut output as *mut u64).cast(),
        (&mut count as *mut u32).cast(),
    ];
    let launch = |kernel, groups, threads, args| unsafe {
        cuLaunchKernel(
            kernel,
            groups,
            1,
            1,
            threads,
            1,
            1,
            0,
            null_mut(),
            args,
            null_mut(),
        )
    };
    assert_eq!(launch(kernel, 1, 0, args.as_mut_ptr()), 1);
    assert_eq!(launch(kernel, 1, 32, null_mut()), 1);
    assert_eq!(
        unsafe {
            cuLaunchKernel(
                kernel,
                1,
                1,
                1,
                32,
                1,
                1,
                4,
                null_mut(),
                args.as_mut_ptr(),
                null_mut(),
            )
        },
        801
    );
    let mut bad_pointer = 0u64;
    let mut invalid_args = args;
    invalid_args[0] = (&mut bad_pointer as *mut u64).cast();
    assert_eq!(launch(kernel, 1, 32, invalid_args.as_mut_ptr()), 1);
    for threads in [32, 64] {
        let mut actual = vec![SENTINEL; CAPACITY];
        ok!(cuMemcpyHtoD_v2(output, actual.as_ptr().cast(), bytes));
        let groups = count.div_ceil(threads);
        assert_eq!(
            launch(kernel, groups, threads, args.as_mut_ptr()),
            0,
            "cuLaunchKernel"
        );
        ok!(cuCtxSynchronize());
        ok!(cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), output, bytes));
        for (i, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                actual.to_bits(),
                expected.to_bits(),
                "element {i}, block size {threads}"
            );
        }
        println!(
            "Verified {N} sums and {} untouched guard elements with {groups} blocks of {threads}",
            CAPACITY - N
        );
    }

    ok!(cuModuleUnload(module));
    assert_eq!(launch(kernel, 1, 32, args.as_mut_ptr()), 400);
    assert_eq!(unsafe { cuModuleUnload(module) }, 400);
    // Context destruction also releases its three live allocations.
    ok!(cuCtxDestroy_v2(context));
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn compiled_ptx_block_reduction() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxSynchronize(),
        cuDeviceGetAttribute(*mut i32, i32, i32),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, 0));
    let source = include_str!("../examples/block_reduce.ptx");
    let mut module = null_mut();
    let mut shared_limit = 0;
    ok!(cuDeviceGetAttribute(&mut shared_limit, 8, 0));
    let oversized_bytes = (shared_limit as u32 + 1).next_multiple_of(4);
    let oversized =
        std::ffi::CString::new(source.replace("[512]", &format!("[{oversized_bytes}]"))).unwrap();
    assert_eq!(
        unsafe { cuModuleLoadData(&mut module, oversized.as_ptr().cast()) },
        801
    );
    assert!(module.is_null());
    let ptx = std::ffi::CString::new(source).unwrap();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(
        &mut kernel,
        module,
        c"block_reduce".as_ptr()
    ));

    const N: usize = 1003;
    const GUARD: usize = 8;
    const SENTINEL: f32 = -99_999.0;
    const SHAPES: [u32; 6] = [1, 16, 31, 64, 95, 128];
    let mut input = vec![1000.0f32; N + 128];
    let mut output = vec![SENTINEL; N + 2 * GUARD];
    let input_bytes = input.len() * size_of::<f32>();
    let output_bytes = output.len() * size_of::<f32>();
    let (mut input_ptr, mut output_base) = (0, 0);
    ok!(cuMemAlloc_v2(&mut input_ptr, input_bytes));
    ok!(cuMemAlloc_v2(&mut output_base, output_bytes));
    let mut output_ptr = output_base + (GUARD * size_of::<f32>()) as u64;
    let mut count = N as u32;
    let mut args = [
        (&mut input_ptr as *mut u64).cast(),
        (&mut output_ptr as *mut u64).cast(),
        (&mut count as *mut u32).cast(),
    ];
    let launch = |groups, threads, args| unsafe {
        cuLaunchKernel(
            kernel,
            groups,
            1,
            1,
            threads,
            1,
            1,
            0,
            null_mut(),
            args,
            null_mut(),
        )
    };
    assert_eq!(launch(1, 129, args.as_mut_ptr()), 1);
    let mut null_input = 0u64;
    let mut bad_args = args;
    bad_args[0] = (&mut null_input as *mut u64).cast();
    assert_eq!(launch(1, 16, bad_args.as_mut_ptr()), 1);

    for (label, divisor) in [("exact", 4.0), ("rounded", 7.0)] {
        for (i, item) in input[..N].iter_mut().enumerate() {
            *item = ((i * 31 % 97) as f32 - 48.0) / divisor;
        }
        ok!(cuMemcpyHtoD_v2(
            input_ptr,
            input.as_ptr().cast(),
            input_bytes
        ));
        for threads in SHAPES {
            output.fill(SENTINEL);
            ok!(cuMemcpyHtoD_v2(
                output_base,
                output.as_ptr().cast(),
                output_bytes
            ));
            let groups = count.div_ceil(threads);
            assert_eq!(
                launch(groups, threads, args.as_mut_ptr()),
                0,
                "cuLaunchKernel, {threads} threads"
            );
            ok!(cuCtxSynchronize());
            ok!(cuMemcpyDtoH_v2(
                output.as_mut_ptr().cast(),
                output_base,
                output_bytes
            ));
            let (prefix, payload) = output.split_at(GUARD);
            let (results, suffix) = payload.split_at(groups as usize);
            assert!(prefix.iter().chain(suffix).all(|&value| value == SENTINEL));
            let mut max_error = 0.0f64;
            // Each input participates in at most ceil(log2(threads)) rounded additions.
            let u = f64::from(f32::EPSILON) / 2.0;
            let ku = f64::from(threads.next_power_of_two().ilog2()) * u;
            let gamma = ku / (1.0 - ku);
            for (block, (tile, &actual)) in
                input[..N].chunks(threads as usize).zip(results).enumerate()
            {
                let expected: f64 = tile.iter().map(|&value| f64::from(value)).sum();
                let magnitude: f64 = tile.iter().map(|&value| f64::from(value).abs()).sum();
                let error = (f64::from(actual) - expected).abs();
                let tolerance = if label == "exact" {
                    assert_eq!(
                        actual.to_bits(),
                        (expected as f32).to_bits(),
                        "block {block}, width {threads}"
                    );
                    0.0
                } else {
                    gamma * magnitude
                };
                assert!(
                    error <= tolerance,
                    "{label}, block {block}, width {threads}: got {actual}, expected {expected}, tolerance {tolerance}"
                );
                max_error = max_error.max(error);
            }
            println!(
                "Reduction ({label}), {groups} blocks of {threads}: max absolute error {max_error:.9e}, guards intact"
            );
        }
    }
    ok!(cuCtxDestroy_v2(context));
}
