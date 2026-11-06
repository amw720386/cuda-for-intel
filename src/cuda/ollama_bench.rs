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
