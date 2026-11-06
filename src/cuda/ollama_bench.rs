use super::streams::{cuStreamCreate, cuStreamSynchronize};
use super::*;
use crate::level_zero::profiling::OperationEvents;
use std::ptr::null_mut;
use std::time::{Duration, Instant};

#[test]
#[ignore = "requires an Intel Level Zero GPU with kernel timestamps"]
fn gated_ggml_add_device_time() {
    const ROWS: u32 = 64;
    const COLS: u32 = 896;
    const WARMUPS: usize = 5;
    const SAMPLES: usize = 31;
    let source = include_str!("../../examples/ggml_add_f32.ptx");
    let mut translation = Vec::new();
    for i in 0..WARMUPS + SAMPLES {
        let start = Instant::now();
        let ptx = ptx::translate(source).unwrap();
        if i >= WARMUPS {
            translation.push(start.elapsed());
        }
        std::hint::black_box(ptx);
    }

    assert_eq!(cuInit(0), Success);
    let mut context = null_mut();
    assert_eq!(
        unsafe { cuCtxCreate_v2(&mut context, 0, test_device()) },
        Success
    );
    let mut events = with_context(|runtime, context| {
        let info = &runtime.devices[context.device as usize].info;
        OperationEvents::create(
            &runtime.api,
            &context.backend,
            1,
            info.timestamp_frequency_hz,
            info.kernel_timestamp_valid_bits,
            false,
        )
        .map_err(backend_error)
    })
    .unwrap();
    let ptx = CString::new(source).unwrap();
    let (mut module, mut kernel) = (null_mut(), null_mut());
    assert_eq!(
        unsafe { cuModuleLoadData(&mut module, ptx.as_ptr().cast()) },
        Success
    );
    assert_eq!(
        unsafe { cuModuleGetFunction(&mut kernel, module, c"ggml_add_f32".as_ptr()) },
        Success
    );
    let bytes = (ROWS * COLS) as usize * size_of::<f32>();
    let input = vec![1.0f32; (ROWS * COLS) as usize];
    let mut pointers = [0u64; 3];
    for ptr in &mut pointers {
        assert_eq!(unsafe { cuMemAlloc_v2(ptr, bytes) }, Success);
        assert_eq!(
            unsafe { cuMemcpyHtoD_v2(*ptr, input.as_ptr().cast(), bytes) },
            Success
        );
    }
    let mut shape = [COLS, ROWS, COLS, COLS, COLS];
    let mut args: Vec<Opaque> = pointers
        .iter_mut()
        .map(|p| (p as *mut u64).cast())
        .collect();
    args.extend(shape.iter_mut().map(|p| (p as *mut u32).cast::<c_void>()));
    let mut device = Vec::new();
    for i in 0..WARMUPS + SAMPLES {
        events.arm(0);
        let start = Instant::now();
        let result = unsafe {
            cuLaunchKernel(
                kernel,
                3,
                ROWS + 1,
                1,
                128,
                1,
                1,
                0,
                null_mut(),
                args.as_mut_ptr(),
                null_mut(),
            )
        };
        assert!(
            !OperationEvents::disarm(),
            "launch did not consume timestamp event"
        );
        assert_eq!(result, Success);
        assert_eq!(cuCtxSynchronize(), Success);
        let elapsed = start.elapsed();
        let [begin, end] = events.query(0).unwrap();
        let ticks = events
            .timer
            .window(begin, end, elapsed)
            .expect("timestamp counter wrapped");
        let duration = Duration::from_secs_f64(events.timer.seconds(ticks));
        assert!(!duration.is_zero() && duration <= elapsed);
        if i >= WARMUPS {
            device.push(duration);
        }
        with_context(|runtime, _| events.reset(&runtime.api, 0).map_err(backend_error)).unwrap();
    }
    let mut output = vec![0.0f32; input.len()];
    assert_eq!(
        unsafe { cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), pointers[2], bytes) },
        Success
    );
    assert!(output.iter().all(|&value| value == 2.0));
    with_context(|runtime, _| events.destroy(&runtime.api).map_err(backend_error)).unwrap();
    assert_eq!(cuCtxDestroy_v2(context), Success);

    for (name, samples) in [
        ("translation only", &mut translation),
        ("Level Zero kernel event", &mut device),
    ] {
        samples.sort_unstable();
        println!(
            "64x896 {name}: n={SAMPLES} median={:.2} us p95={:.2} us",
            samples[SAMPLES.div_ceil(2) - 1].as_secs_f64() * 1e6,
            samples[(SAMPLES * 95).div_ceil(100) - 1].as_secs_f64() * 1e6,
        );
    }
}

#[test]
#[ignore = "requires release build and a Level Zero GPU with kernel timestamps"]
fn ggml_scale_clamp_timings() {
    const COLS: u32 = 896;
    const ROWS: u32 = 64;
    const COUNT: u32 = COLS * ROWS;
    const BYTES: usize = COUNT as usize * size_of::<f32>();
    const BLOCK: u32 = 256;
    const WARMUPS: usize = 5;
    const SAMPLES: usize = 31;
    let source = include_str!("../../examples/ggml_scale_clamp.ptx");
    let image = CString::new(source).unwrap();
    let mut context = null_mut();
    assert_eq!(cuInit(0), Success);
    assert_eq!(
        unsafe { cuCtxCreate_v2(&mut context, 0, test_device()) },
        Success
    );

    let measure = |name: &str, sample: &mut dyn FnMut() -> Duration| {
        let mut values: Vec<_> = (0..WARMUPS + SAMPLES)
            .map(|_| sample())
            .skip(WARMUPS)
            .collect();
        values.sort_unstable();
        println!(
            "64x896 {name}: n={SAMPLES} median={:.2} us p95={:.2} us",
            values[SAMPLES.div_ceil(2) - 1].as_secs_f64() * 1e6,
            values[(SAMPLES * 95).div_ceil(100) - 1].as_secs_f64() * 1e6,
        );
    };
    measure("translation (two entries)", &mut || {
        let start = Instant::now();
        std::hint::black_box(ptx::translate(source).unwrap());
        start.elapsed()
    });

    let load = || {
        let mut module = null_mut();
        let start = Instant::now();
        assert_eq!(
            unsafe { cuModuleLoadData(&mut module, image.as_ptr().cast()) },
            Success
        );
        (module, start.elapsed())
    };
    let (module, first) = load();
    println!(
        "64x896 first load in process: {:.2} us",
        first.as_secs_f64() * 1e6
    );
    measure("repeat module load incl translation", &mut || {
        let (module, elapsed) = load();
        assert_eq!(cuModuleUnload(module), Success);
        elapsed
    });

    let translated = ptx::translate(source).unwrap();
    measure("repeat native module creation", &mut || {
        with_context(|runtime, context| {
            let start = Instant::now();
            let mut module = context
                .backend
                .load_module(&runtime.api, &translated.words)
                .map_err(backend_error)?;
            let elapsed = start.elapsed();
            module.destroy(&runtime.api).map_err(backend_error)?;
            Ok(elapsed)
        })
        .unwrap()
    });

    let mut kernels = [null_mut(); 2];
    for (kernel, symbol) in kernels.iter_mut().zip([
        c"_Z9scale_f32IfEvPKT_PS0_ffj5uint3S4_S4_jjjjjj",
        c"_Z15op_clamp_kernelIfEvPKT_PS0_S0_S0_j5uint3S4_S4_jjjjjj",
    ]) {
        assert_eq!(
            unsafe { cuModuleGetFunction(kernel, module, symbol.as_ptr()) },
            Success
        );
    }
    let mut events = with_context(|runtime, context| {
        let info = &runtime.devices[context.device as usize].info;
        OperationEvents::create(
            &runtime.api,
            &context.backend,
            1,
            info.timestamp_frequency_hz,
            info.kernel_timestamp_valid_bits,
            false,
        )
        .map_err(backend_error)
    })
    .unwrap();
    let mut buffers = [0u64; 3];
    for buffer in &mut buffers {
        assert_eq!(unsafe { cuMemAlloc_v2(buffer, BYTES) }, Success);
    }
    let mut stream = null_mut();
    assert_eq!(unsafe { cuStreamCreate(&mut stream, 1) }, Success);
    let input = vec![1.25f32; COUNT as usize];
    let mut output = vec![0.0f32; COUNT as usize];
    let upload = || {
        assert_eq!(
            unsafe { cuMemcpyHtoD_v2(buffers[0], input.as_ptr().cast(), BYTES) },
            Success
        );
    };
    let launch = |index: usize| {
        let mut input = buffers[index];
        let mut output = buffers[index + 1];
        let (mut a, mut b) = [(1.5f32, 0.25f32), (-1.0f32, 2.0f32)][index];
        let mut count = COUNT;
        // GGML fast division constants for 896, 64, 1.
        let mut divs = [
            [
                ((1u64 << 32) * (1024 - u64::from(COLS)) / u64::from(COLS) + 1) as u32,
                10,
                COLS,
            ],
            [1, 6, ROWS],
            [1, 0, 1],
        ];
        let mut strides = [COLS, COUNT, COUNT];
        let ptr = std::ptr::from_mut;
        let mut args: Vec<Opaque> = vec![ptr(&mut input).cast(), ptr(&mut output).cast()];
        args.push(std::ptr::from_mut(&mut a).cast());
        args.push(std::ptr::from_mut(&mut b).cast());
        args.push(std::ptr::from_mut(&mut count).cast());
        args.extend(
            divs.iter_mut()
                .map(|p| std::ptr::from_mut(p).cast::<c_void>()),
        );
        let stride_args = strides
            .each_mut()
            .map(|p| std::ptr::from_mut(p).cast::<c_void>());
        args.extend(stride_args);
        args.extend(stride_args);
        assert_eq!(
            unsafe {
                cuLaunchKernel(
                    kernels[index],
                    COUNT.div_ceil(BLOCK),
                    1,
                    1,
                    BLOCK,
                    1,
                    1,
                    0,
                    stream,
                    args.as_mut_ptr(),
                    null_mut(),
                )
            },
            Success
        );
    };
    upload();
    launch(0);
    assert_eq!(cuStreamSynchronize(stream), Success);
    for (index, op) in ["scale", "clamp"].into_iter().enumerate() {
        measure(&format!("{op} launch and stream sync"), &mut || {
            let start = Instant::now();
            launch(index);
            assert_eq!(cuStreamSynchronize(stream), Success);
            start.elapsed()
        });
        measure(&format!("{op} kernel timestamp"), &mut || {
            events.arm(0);
            let start = Instant::now();
            launch(index);
            assert!(!OperationEvents::disarm());
            assert_eq!(cuStreamSynchronize(stream), Success);
            let elapsed = start.elapsed();
            let [begin, end] = events.query(0).unwrap();
            let ticks = events
                .timer
                .window(begin, end, elapsed)
                .expect("invalid timestamp window");
            let duration = Duration::from_secs_f64(events.timer.seconds(ticks));
            assert!(!duration.is_zero() && duration <= elapsed);
            with_context(|runtime, _| events.reset(&runtime.api, 0).map_err(backend_error))
                .unwrap();
            duration
        });
    }
    measure("empty stream sync", &mut || {
        let start = Instant::now();
        assert_eq!(cuStreamSynchronize(stream), Success);
        start.elapsed()
    });
    let chain = || {
        launch(0);
        launch(1);
        assert_eq!(cuStreamSynchronize(stream), Success);
    };
    measure("resident chain and stream sync", &mut || {
        let start = Instant::now();
        chain();
        start.elapsed()
    });
    measure("blocking HtoD + chain + blocking DtoH", &mut || {
        let start = Instant::now();
        upload();
        chain();
        assert_eq!(
            unsafe { cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), buffers[2], BYTES) },
            Success
        );
        start.elapsed()
    });
    assert!(output.iter().all(|&value| value == 2.0));
    with_context(|runtime, _| events.destroy(&runtime.api).map_err(backend_error)).unwrap();
    assert_eq!(cuCtxDestroy_v2(context), Success);
}
