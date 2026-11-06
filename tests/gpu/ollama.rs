use super::*;
use std::time::{Duration, Instant};

fn report(label: &str, samples: &mut [Duration]) {
    samples.sort_unstable();
    let micros = |index: usize| samples[index].as_secs_f64() * 1e6;
    println!(
        "{label}: n={} median={:.2} us p95={:.2} us",
        samples.len(),
        micros(samples.len().div_ceil(2) - 1),
        micros((samples.len() * 95).div_ceil(100) - 1)
    );
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn ggml_cuda_add_f32() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxSetCurrent(Handle),
        cuCtxSynchronize(),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuModuleUnload(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));

    let ptx = std::ffi::CString::new(include_str!("../../examples/ggml_add_f32.ptx")).unwrap();
    let mut module = null_mut();
    let mut module_times = Vec::new();
    for index in 0..12 {
        let start = Instant::now();
        ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
        let duration = start.elapsed();
        if index == 0 {
            println!("first module load: {:.2} us", duration.as_secs_f64() * 1e6);
        } else {
            module_times.push(duration);
        }
        if index != 11 {
            ok!(cuModuleUnload(module));
        }
    }
    report("repeat module loads", &mut module_times);
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(
        &mut kernel,
        module,
        c"ggml_add_f32".as_ptr()
    ));

    const GUARD: usize = 7;
    const SENTINEL: f32 = -98765.0;
    const WARMUPS: usize = 5;
    const SAMPLES: usize = 31;
    for (cols, rows, groups, threads) in [
        (1usize, 1usize, 1, 1),
        (63, 3, 1, 64),
        (64, 2, 1, 64),
        (65, 5, 1, 64),
        (257, 7, 2, 63),
        (896, 64, 3, 128),
    ] {
        let strides = [cols + 3, cols + 5, cols + 9];
        let capacities = strides.map(|stride| GUARD * 2 + stride * rows);
        let mut left = vec![SENTINEL; capacities[0]];
        let mut right = vec![SENTINEL; capacities[1]];
        let initial = vec![SENTINEL; capacities[2]];
        let mut output = initial.clone();
        let mut expected = initial.clone();
        let edge = [
            (f32::from_bits(1), f32::from_bits(1)),
            (1.0, f32::EPSILON / 2.0),
            (-0.0, -0.0),
            (f32::MAX, f32::MAX),
            (f32::INFINITY, f32::NEG_INFINITY),
            (f32::NAN, 0.0),
            (f32::MIN_POSITIVE, -f32::MIN_POSITIVE),
        ];
        for row in 0..rows {
            for col in 0..cols {
                let index = row * cols + col;
                let (a, b) = edge.get(index).copied().unwrap_or_else(|| {
                    (
                        (index % 251) as f32 * 0.3125,
                        -((index % 83) as f32) * 0.1875,
                    )
                });
                left[GUARD + row * strides[0] + col] = a;
                right[GUARD + row * strides[1] + col] = b;
                expected[GUARD + row * strides[2] + col] = a + b;
            }
        }
        let bytes = capacities.map(|len| len * size_of::<f32>());
        let mut device = [0; 3];
        for (ptr, len) in device.iter_mut().zip(bytes) {
            ok!(cuMemAlloc_v2(ptr, len));
        }
        let mut pointers = device.map(|ptr| ptr + (GUARD * size_of::<f32>()) as u64);
        let mut shape = [
            cols as u32,
            rows as u32,
            strides[0] as u32,
            strides[1] as u32,
            strides[2] as u32,
        ];
        let mut args: Vec<Handle> = pointers
            .iter_mut()
            .map(|p| (p as *mut u64).cast())
            .collect();
        args.extend(shape.iter_mut().map(|p| (p as *mut u32).cast::<c_void>()));
        let launch = || unsafe {
            cuLaunchKernel(
                kernel,
                groups,
                rows as u32 + 1,
                1,
                threads,
                1,
                1,
                0,
                null_mut(),
                args.as_ptr().cast_mut(),
                null_mut(),
            )
        };

        let mut peer = null_mut();
        ok!(cuCtxCreate_v2(&mut peer, 0, test_device()));
        assert_eq!(launch(), 400, "foreign kernel handle");
        ok!(cuCtxDestroy_v2(peer));
        ok!(cuCtxSetCurrent(context));

        let upload = || {
            for (index, host) in [left.as_slice(), right.as_slice(), initial.as_slice()]
                .into_iter()
                .enumerate()
            {
                ok!(cuMemcpyHtoD_v2(
                    device[index],
                    host.as_ptr().cast(),
                    bytes[index]
                ));
            }
        };
        upload();
        let mut completed = Vec::new();
        let mut transfer = Vec::new();
        for index in 0..WARMUPS + SAMPLES {
            let start = Instant::now();
            assert_eq!(launch(), 0);
            ok!(cuCtxSynchronize());
            let launch_time = start.elapsed();

            let start = Instant::now();
            upload();
            assert_eq!(launch(), 0);
            ok!(cuMemcpyDtoH_v2(
                output.as_mut_ptr().cast(),
                device[2],
                bytes[2]
            ));
            let transfer_time = start.elapsed();
            if index >= WARMUPS {
                completed.push(launch_time);
                transfer.push(transfer_time);
            }
            for (offset, (&actual, &reference)) in output.iter().zip(&expected).enumerate() {
                if reference.is_nan() {
                    assert!(actual.is_nan(), "{rows}x{cols} at {offset}: {actual}");
                } else {
                    assert_eq!(
                        actual.to_bits(),
                        reference.to_bits(),
                        "{rows}x{cols} at {offset}"
                    );
                }
            }
        }
        report(&format!("{rows}x{cols} launch+wait"), &mut completed);
        report(
            &format!("{rows}x{cols} upload+launch+download"),
            &mut transfer,
        );
        for ptr in device {
            ok!(cuMemFree_v2(ptr));
        }
    }
    ok!(cuModuleUnload(module));
    ok!(cuCtxDestroy_v2(context));
}
