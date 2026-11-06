//! Opt-in tests against the built CUDA DLL and an actual Intel GPU.
#![cfg(windows)]

use libloading::Library;
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;

type Handle = *mut c_void;

#[path = "gpu/device.rs"]
mod gpu_device;
use gpu_device::ordinal as test_device;

#[path = "gpu/discovery.rs"]
mod discovery;
#[path = "gpu/forge.rs"]
mod forge;
#[path = "gpu/ggml.rs"]
mod ggml;
#[path = "gpu/ggml_norm.rs"]
mod ggml_norm;
#[path = "gpu/isolation.rs"]
mod isolation;
#[path = "gpu/isolation_capability.rs"]
mod isolation_capability;
#[path = "gpu/ollama.rs"]
mod ollama;
#[path = "gpu/pinned.rs"]
mod pinned;
#[path = "gpu/progress.rs"]
mod progress;
#[path = "gpu/shfl.rs"]
mod shfl;

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

use {ok, symbols};

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
    ok!(cuDeviceGetName(
        name.as_mut_ptr(),
        name.len() as i32,
        test_device()
    ));
    println!(
        "Device: {}",
        unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_string_lossy()
    );

    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
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
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));

    let mut module = null_mut();
    ok!(cuModuleLoad(
        &mut module,
        c"examples/vector_add.ptx".as_ptr()
    ));
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
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let source = include_str!("../examples/block_reduce.ptx");
    let mut module = null_mut();
    let mut shared_limit = 0;
    ok!(cuDeviceGetAttribute(&mut shared_limit, 8, test_device()));
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

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn compiled_ptx_atomic_tickets() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let ptx = std::ffi::CString::new(include_str!("../examples/atomic_tickets.ptx")).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(
        &mut kernel,
        module,
        c"atomic_tickets".as_ptr()
    ));

    const GROUPS: u32 = 23;
    const THREADS: u32 = 64;
    const COUNT: u32 = GROUPS * THREADS;
    const START: u32 = u32::MAX - COUNT / 2;
    const GUARD: u32 = 0x1234_5678;
    let mut tickets = vec![GUARD; COUNT as usize + 2];
    let size = size_of::<u32>();
    let bytes = tickets.len() * size;
    let (mut counter, mut ticket_base) = (0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut counter, size));
    ok!(cuMemAlloc_v2(&mut ticket_base, bytes));
    ok!(cuMemcpyHtoD_v2(
        counter,
        (&START as *const u32).cast(),
        size
    ));
    ok!(cuMemcpyHtoD_v2(ticket_base, tickets.as_ptr().cast(), bytes));
    let mut ticket_ptr = ticket_base + size as u64;
    let mut args = [
        (&mut counter as *mut u64).cast(),
        (&mut ticket_ptr as *mut u64).cast(),
    ];

    ok!(cuLaunchKernel(
        kernel,
        GROUPS,
        1,
        1,
        THREADS,
        1,
        1,
        0,
        null_mut(),
        args.as_mut_ptr(),
        null_mut()
    ));
    let mut final_count = 0u32;
    ok!(cuMemcpyDtoH_v2(
        (&mut final_count as *mut u32).cast(),
        counter,
        size
    ));
    ok!(cuMemcpyDtoH_v2(
        tickets.as_mut_ptr().cast(),
        ticket_base,
        bytes
    ));
    assert_eq!(final_count, START.wrapping_add(COUNT));
    assert_eq!(tickets.first(), Some(&GUARD));
    assert_eq!(tickets.last(), Some(&GUARD));
    let mut offsets: Vec<_> = tickets[1..=COUNT as usize]
        .iter()
        .map(|&ticket| ticket.wrapping_sub(START))
        .collect();
    offsets.sort_unstable();
    assert!(
        offsets.into_iter().eq(0..COUNT),
        "atomic tickets must be unique and complete"
    );
    println!(
        "Verified {COUNT} atomic tickets across {GROUPS} workgroups, including counter wraparound"
    );
    ok!(cuCtxDestroy_v2(context));
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn compiled_ptx_rmsnorm() {
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
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let ptx = std::ffi::CString::new(include_str!("../examples/rmsnorm.ptx")).unwrap();
    let unsupported = std::ffi::CString::new(
        ptx.to_str()
            .unwrap()
            .replace("sqrt.rn.f32", "sqrt.approx.f32"),
    )
    .unwrap();
    let mut module = null_mut();
    assert_eq!(
        unsafe { cuModuleLoadData(&mut module, unsupported.as_ptr().cast()) },
        218
    );
    assert!(module.is_null());
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(
        &mut kernel,
        module,
        c"rmsnorm".as_ptr()
    ));

    const ROWS: usize = 4;
    const GUARD: usize = 8;
    const SENTINEL: f32 = -99_999.0;
    let mut rng = 0x9e37_79b9u32;
    let mut random = || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (rng >> 8) as f32 * (1.0 / 8_388_608.0) - 1.0
    };
    let launch = |threads, args| unsafe {
        cuLaunchKernel(
            kernel,
            ROWS as u32,
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
    let size = size_of::<f32>();

    for dim in [32usize, 96, 256, 896, 1024] {
        let mut input = Vec::with_capacity(ROWS * dim);
        for scale in [1.0f32, 4.0, 0.0, 1.0e-10] {
            input.extend((0..dim).map(|_| random() * scale));
        }
        let weights: Vec<_> = (0..dim).map(|_| 0.5 + 1.5 * random()).collect();
        let guarded = vec![SENTINEL; ROWS * dim + 2 * GUARD];
        let bytes = guarded.len() * size;
        let mut actual = guarded.clone();
        let (mut x, mut weight, mut output) = (0u64, 0u64, 0u64);
        ok!(cuMemAlloc_v2(&mut x, input.len() * size));
        ok!(cuMemAlloc_v2(&mut weight, weights.len() * size));
        ok!(cuMemAlloc_v2(&mut output, bytes));
        ok!(cuMemcpyHtoD_v2(
            x,
            input.as_ptr().cast(),
            input.len() * size
        ));
        ok!(cuMemcpyHtoD_v2(
            weight,
            weights.as_ptr().cast(),
            weights.len() * size
        ));
        let mut y = output + (GUARD * size) as u64;
        let mut columns = dim as u32;
        let mut epsilon = 1e-5f32;
        let mut args = [
            (&mut x as *mut u64).cast(),
            (&mut weight as *mut u64).cast(),
            (&mut y as *mut u64).cast(),
            (&mut columns as *mut u32).cast(),
            (&mut epsilon as *mut f32).cast(),
        ];
        assert_eq!(launch(129, args.as_mut_ptr()), 1);
        assert_eq!(launch(32, null_mut()), 1);
        let mut null_pointer = 0u64;
        let mut bad_args = args;
        bad_args[2] = (&mut null_pointer as *mut u64).cast();
        assert_eq!(launch(32, bad_args.as_mut_ptr()), 1);

        for eps in [1e-12f32, 1e-5] {
            epsilon = eps;
            for threads in [32u32, 63, 128] {
                // Nonnegative squares give a relative summation bound from the FMA chain
                // and the shared-memory tree. Allow eight more FP32 rounding steps and a
                // factor of two for the remaining normalization arithmetic.
                let depth = columns.div_ceil(threads) + threads.next_power_of_two().ilog2();
                let ku = f64::from(depth + 8) * f64::from(f32::EPSILON) / 2.0;
                let relative_limit = 2.0 * ku / (1.0 - ku);

                let mut first = Vec::new();
                for _ in 0..2 {
                    ok!(cuMemcpyHtoD_v2(output, guarded.as_ptr().cast(), bytes));
                    assert_eq!(
                        launch(threads, args.as_mut_ptr()),
                        0,
                        "launch {dim}x{threads}"
                    );
                    ok!(cuMemcpyDtoH_v2(actual.as_mut_ptr().cast(), output, bytes));
                    assert!(
                        actual[..GUARD]
                            .iter()
                            .chain(&actual[GUARD + ROWS * dim..])
                            .all(|&value| value.to_bits() == SENTINEL.to_bits())
                    );
                    let bits = actual[GUARD..GUARD + ROWS * dim]
                        .iter()
                        .map(|&value| value.to_bits());
                    if first.is_empty() {
                        first.extend(bits);
                    } else {
                        assert!(
                            first.iter().copied().eq(bits),
                            "nondeterministic {dim}x{threads}"
                        );
                    }
                }

                let mut max_abs = 0.0f64;
                let mut max_rel = 0.0f64;
                for (row, values) in input.chunks_exact(dim).enumerate() {
                    let mean =
                        values.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>() / dim as f64;
                    let rms = (mean + f64::from(epsilon)).sqrt();
                    for (col, (&x, &w)) in values.iter().zip(&weights).enumerate() {
                        let expected = f64::from(x) * f64::from(w) / rms;
                        let result = f64::from(actual[GUARD + row * dim + col]);
                        let error = (result - expected).abs();
                        assert!(
                            error <= relative_limit * expected.abs(),
                            "dim {dim}, block {threads}, epsilon {eps:e}, row {row}, col {col}: {result} != {expected}"
                        );
                        max_abs = max_abs.max(error);
                        if expected != 0.0 {
                            max_rel = max_rel.max(error / expected.abs());
                        }
                    }
                }
                println!(
                    "RMSNorm dim={dim:4} block={threads:3} eps={eps:.0e} abs={max_abs:.9e} rel={max_rel:.9e} limit={relative_limit:.9e}"
                );
            }
        }
        ok!(cuMemFree_v2(x));
        ok!(cuMemFree_v2(weight));
        ok!(cuMemFree_v2(output));
    }
    ok!(cuCtxDestroy_v2(context));
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn fp32_rounding_and_contraction() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    let source = c"
        .version 4.1
        .target sm_52
        .address_size 64
        .entry check_fp32(.param .u64 .ptr input, .param .u64 .ptr output) {
            .reg .b32 %r<13>;
            .reg .b64 %d<5>;
            ld.param.b64 %d0, [input];
            ld.param.b64 %d1, [output];
            mov.u32 %r0, %ctaid.x;
            mov.u32 %r1, %ntid.x;
            mov.u32 %r2, %tid.x;
            mad.lo.u32 %r3, %r0, %r1, %r2;
            mul.wide.u32 %d2, %r3, 12;
            add.u64 %d3, %d0, %d2;
            ld.global.b32 %r4, [%d3];
            add.u64 %d3, %d3, 4;
            ld.global.b32 %r5, [%d3];
            add.u64 %d3, %d3, 4;
            ld.global.b32 %r6, [%d3];

            div.rn.f32 %r7, %r4, %r5;
            sqrt.rn.f32 %r8, %r4;
            fma.rn.f32 %r9, %r4, %r5, %r6;
            mul.rn.f32 %r10, %r4, %r5;
            add.rn.f32 %r11, %r10, %r6;
            cvt.rn.f32.u32 %r12, %r4;

            mul.wide.u32 %d2, %r3, 20;
            add.u64 %d4, %d1, %d2;
            st.global.b32 [%d4], %r7;
            add.u64 %d4, %d4, 4;
            st.global.b32 [%d4], %r8;
            add.u64 %d4, %d4, 4;
            st.global.b32 [%d4], %r9;
            add.u64 %d4, %d4, 4;
            st.global.b32 [%d4], %r11;
            add.u64 %d4, %d4, 4;
            st.global.b32 [%d4], %r12;
            ret;
        }
    ";
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, source.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(
        &mut kernel,
        module,
        c"check_fp32".as_ptr()
    ));

    const LANES: usize = 4096;
    let tiny = f32::from_bits(1);
    let mut cases = vec![
        [1.0 + f32::EPSILON, 1.0 - f32::EPSILON, -1.0],
        [f32::MAX, 2.0, -f32::MAX],
        [tiny, 2.0, 0.0],
        [tiny * 3.0, 2.0, 0.0],
        [f32::MIN_POSITIVE, 2.0, -tiny],
        [-tiny, 2.0, tiny],
        [-0.0, 2.0, -0.0],
        [0.0, -0.0, 1.0],
        [1.0, -0.0, 1.0],
        [f32::INFINITY, f32::INFINITY, 1.0],
        [1.0, f32::NEG_INFINITY, 1.0],
        [f32::NAN, 1.0, 0.0],
        [1.0, f32::NAN, 0.0],
        [f32::from_bits(0x0100_0001), 1.0, 0.0],
        [f32::from_bits(0x0100_0003), 1.0, 0.0],
    ];
    let mut rng = 0x85eb_ca6bu32;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        f32::from_bits(rng)
    };
    cases.resize_with(LANES, || [next(), next(), next()]);
    let mut result = vec![[0.0f32; 5]; LANES];
    let in_bytes = std::mem::size_of_val(cases.as_slice());
    let out_bytes = std::mem::size_of_val(result.as_slice());
    let (mut input, mut output) = (0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut input, in_bytes));
    ok!(cuMemAlloc_v2(&mut output, out_bytes));
    ok!(cuMemcpyHtoD_v2(input, cases.as_ptr().cast(), in_bytes));
    let mut args = [
        (&mut input as *mut u64).cast(),
        (&mut output as *mut u64).cast(),
    ];
    ok!(cuLaunchKernel(
        kernel,
        LANES as u32 / 64,
        1,
        1,
        64,
        1,
        1,
        0,
        null_mut(),
        args.as_mut_ptr(),
        null_mut()
    ));
    ok!(cuMemcpyDtoH_v2(
        result.as_mut_ptr().cast(),
        output,
        out_bytes
    ));
    let names = ["divide", "sqrt", "fma", "mul + add", "convert"];
    for (index, (&[a, b, c], results)) in cases.iter().zip(&result).enumerate() {
        let reference = [
            a / b,
            a.sqrt(),
            a.mul_add(b, c),
            std::hint::black_box(a * b) + c,
            a.to_bits() as f32,
        ];
        for ((&actual, expected), name) in results.iter().zip(reference).zip(names) {
            assert!(
                (expected.is_nan() && actual.is_nan()) || expected.to_bits() == actual.to_bits(),
                "{name} case {index}, inputs {:08x?}, got {:08x}, expected {:08x}",
                [a.to_bits(), b.to_bits(), c.to_bits()],
                actual.to_bits(),
                expected.to_bits()
            );
        }
    }
    println!(
        "Verified {LANES} FP32 bit-pattern cases across division, sqrt, FMA, unfused multiply-add and conversion"
    );
    ok!(cuCtxDestroy_v2(context));
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn asynchronous_stream_order_and_lifetimes() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxSynchronize(),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
        cuStreamSynchronize(Handle),
        cuStreamQuery(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuModuleUnload(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    let ptx = c"
        .version 4.1
        .target sm_52
        .address_size 64
        .entry increment(.param .u64 .ptr data, .param .u32 delta) {
            .reg .b64 %ptr;
            .reg .b32 %a;
            .reg .b32 %b;
            ld.param.b64 %ptr, [data];
            ld.param.b32 %a, [delta];
            ld.global.b32 %b, [%ptr];
            add.s32 %b, %b, %a;
            st.global.b32 [%ptr], %b;
            ret;
        }
    ";
    ok!(cuInit(0));
    let (mut context, mut module, mut kernel) = (null_mut(), null_mut(), null_mut());
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    ok!(cuModuleGetFunction(
        &mut kernel,
        module,
        c"increment".as_ptr()
    ));
    let (mut blocking, mut other_blocking, mut first, mut second) =
        (null_mut(), null_mut(), null_mut(), null_mut());
    let legacy = std::ptr::without_provenance_mut(1);
    let per_thread = std::ptr::without_provenance_mut(2);
    let invalid = std::ptr::without_provenance_mut(usize::MAX);
    assert_eq!(unsafe { cuStreamCreate(&mut first, 2) }, 1);
    assert!(first.is_null());
    assert_eq!(unsafe { cuStreamCreate(null_mut(), 0) }, 1);
    assert_eq!(unsafe { cuStreamQuery(invalid) }, 400);
    assert_eq!(unsafe { cuStreamQuery(per_thread) }, 801);
    assert_eq!(unsafe { cuStreamDestroy_v2(legacy) }, 400);

    let size = size_of::<u32>();
    let alloc = || {
        let mut address = 0u64;
        let zero = 0u32;
        ok!(cuMemAlloc_v2(&mut address, size));
        ok!(cuMemcpyHtoD_v2(address, (&zero as *const u32).cast(), size));
        address
    };
    let shared = alloc();
    let a = alloc();
    let b = alloc();
    let read = |address| {
        let mut value = 0u32;
        ok!(cuMemcpyDtoH_v2(
            (&mut value as *mut u32).cast(),
            address,
            size
        ));
        value
    };
    let launch = |stream, mut address: u64, mut delta: u32| {
        let mut args = [
            (&mut address as *mut u64).cast(),
            (&mut delta as *mut u32).cast(),
        ];
        unsafe {
            cuLaunchKernel(
                kernel,
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
    assert_eq!(launch(invalid, shared, 1), 400);
    assert_eq!(launch(per_thread, shared, 1), 801);
    assert_eq!(launch(legacy, shared, 5), 0);
    ok!(cuStreamCreate(&mut blocking, 0));
    ok!(cuStreamCreate(&mut other_blocking, 0));
    ok!(cuStreamCreate(&mut first, 1));
    ok!(cuStreamCreate(&mut second, 1));
    ok!(cuStreamQuery(first));
    // The preceding launch may still be pending when a blocking stream is created.
    assert!(matches!(unsafe { cuStreamQuery(legacy) }, 0 | 600));
    ok!(cuStreamSynchronize(legacy));
    ok!(cuStreamQuery(legacy));

    let mut foreign_context = null_mut();
    let mut foreign_stream = null_mut();
    ok!(cuCtxCreate_v2(&mut foreign_context, 0, test_device()));
    ok!(cuStreamCreate(&mut foreign_stream, 1));
    ok!(cuStreamQuery(first)); // Explicit stream operations resolve the owner.
    assert_eq!(launch(first, a, 1), 400);
    ok!(cuCtxDestroy_v2(foreign_context));
    assert_eq!(unsafe { cuStreamQuery(foreign_stream) }, 400);

    for _ in 0..256 {
        assert_eq!(launch(blocking, shared, 1), 0);
        assert_eq!(launch(null_mut(), shared, 2), 0);
        assert_eq!(launch(other_blocking, shared, 3), 0);
        assert_eq!(launch(legacy, shared, 4), 0);
    }
    assert_eq!(launch(blocking, shared, 7), 0);
    // A blocking copy is ordered after legacy and blocking stream work.
    assert_eq!(read(shared), 2572);
    ok!(cuStreamQuery(legacy));

    for delta in 1..=1024 {
        assert_eq!(launch(first, a, delta), 0);
        assert_eq!(launch(second, b, 2 * delta), 0);
    }
    assert!(matches!(unsafe { cuStreamQuery(first) }, 0 | 600));
    ok!(cuStreamSynchronize(first));
    ok!(cuStreamQuery(first));
    assert_eq!(read(a), 1024 * 1025 / 2);
    ok!(cuStreamDestroy_v2(second));
    assert_eq!(unsafe { cuStreamQuery(second) }, 400);
    assert_eq!(unsafe { cuStreamDestroy_v2(second) }, 400);
    ok!(cuCtxSynchronize());
    assert_eq!(read(b), 1024 * 1025);

    // Module unload and device free may occur with queued work. The runtime
    // must keep their backend objects live until that work completes.
    assert_eq!(launch(first, a, 5), 0);
    ok!(cuMemFree_v2(a));
    ok!(cuStreamQuery(first));
    assert_eq!(launch(first, b, 11), 0);
    ok!(cuModuleUnload(module));
    assert_eq!(read(b), 1024 * 1025 + 11);
    assert_eq!(launch(first, b, 1), 400);
    ok!(cuStreamDestroy_v2(first));
    ok!(cuStreamDestroy_v2(blocking));
    ok!(cuStreamDestroy_v2(other_blocking));
    ok!(cuCtxDestroy_v2(context));
    println!(
        "Verified legacy ordering, independent streams, retirement, and 3,072 queued launches"
    );
}
