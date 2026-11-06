use super::{Handle, library, ok, symbols};
use std::ffi::{CString, c_char, c_void};
use std::ptr::null_mut;
use std::time::{Duration, Instant};

const GUARD: usize = 8;
const SENTINEL: f32 = -99_999.0;
const EPSILON: f32 = 1e-5;
const BATCH: usize = 32;

fn arg<T>(value: &T) -> Handle {
    std::ptr::from_ref(value).cast_mut().cast()
}

fn report(label: &str, samples: &mut [Duration]) {
    samples.sort_unstable();
    let micros = |duration: Duration| duration.as_secs_f64() * 1e6;
    let p95 = (95 * samples.len()).div_ceil(100) - 1;
    println!(
        "{label}: n={} median={:.3} us p95={:.3} us",
        samples.len(),
        micros(samples[samples.len() / 2]),
        micros(samples[p95]),
    );
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn dependent_forge_chain() {
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
        cuStreamCreate(*mut Handle, u32),
        cuStreamSynchronize(Handle),
        cuEventCreate(*mut Handle, u32),
        cuEventRecord(Handle, Handle),
        cuEventDestroy_v2(Handle),
        cuStreamWaitEvent(Handle, Handle, u32),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, 0));
    let mut streams = [null_mut(); 3];
    for stream in &mut streams {
        ok!(cuStreamCreate(stream, 1));
    }
    let mut events = [null_mut(); 2];
    for event in &mut events {
        ok!(cuEventCreate(event, 2));
    }
    let kernels = [
        (c"rmsnorm", include_str!("../../examples/rmsnorm.ptx")),
        (
            c"forge_residual",
            include_str!("../../examples/forge_residual.ptx"),
        ),
        (
            c"forge_reduce",
            include_str!("../../examples/forge_reduce.ptx"),
        ),
    ]
    .map(|(name, source)| {
        let (mut module, mut kernel) = (null_mut(), null_mut());
        let ptx = CString::new(source).unwrap();
        ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
        ok!(cuModuleGetFunction(&mut kernel, module, name.as_ptr()));
        kernel
    });

    let launch = |kernel, blocks, threads, stream, args: &mut [Handle]| {
        ok!(cuLaunchKernel(
            kernel,
            blocks,
            1,
            1,
            threads,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            null_mut()
        ));
    };
    let upload = |address, values: &[f32]| {
        ok!(cuMemcpyHtoD_v2(
            address,
            values.as_ptr().cast(),
            size_of_val(values)
        ));
    };
    let download = |address, values: &mut [f32]| {
        ok!(cuMemcpyDtoH_v2(
            values.as_mut_ptr().cast(),
            address,
            size_of_val(values)
        ));
    };
    let alloc = |values: &[f32]| {
        let mut address = 0;
        ok!(cuMemAlloc_v2(&mut address, size_of_val(values)));
        upload(address, values);
        address
    };
    let bytes = |elements: usize| (elements * size_of::<f32>()) as u64;
    let measure = std::env::var_os("CUDA_FOR_INTEL_BENCH_EVENTS").is_some();
    let mut rng = 0x9e37_79b9u32;
    let mut random = || {
        rng = rng.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (rng >> 8) as f32 * (1.0 / 8_388_608.0) - 1.0
    };

    for (rows, columns, threads) in [
        (1u32, 1u32, 1u32),
        (3, 17, 32),
        (7, 257, 63),
        (4, 1024, 128),
    ] {
        let elements = rows * columns;
        let tensor_len = elements as usize;
        let row_count = rows as usize;
        let tensor_stride = tensor_len + 2 * GUARD;
        let result_stride = row_count + 2 * GUARD;
        let input: Vec<_> = (0..BATCH * tensor_len).map(|_| random()).collect();
        let residual: Vec<_> = (0..input.len()).map(|_| random() * 0.5).collect();
        let weights: Vec<_> = (0..columns).map(|_| random() * 0.5 + 1.0).collect();
        let tensors = vec![SENTINEL; BATCH * tensor_stride];
        let results = vec![SENTINEL; BATCH * result_stride];
        let x = alloc(&input);
        let skip = alloc(&residual);
        let weight = alloc(&weights);
        let norm = alloc(&tensors);
        let added = alloc(&tensors);
        let output = alloc(&results);

        let submit = |multiple, batch| {
            let [first, second, third] = if multiple { streams } else { [streams[0]; 3] };
            for slot in 0..batch {
                let input = x + bytes(slot * tensor_len);
                let residual = skip + bytes(slot * tensor_len);
                let normalized = norm + bytes(slot * tensor_stride + GUARD);
                let sum = added + bytes(slot * tensor_stride + GUARD);
                let result = output + bytes(slot * result_stride + GUARD);
                launch(
                    kernels[0],
                    rows,
                    threads,
                    first,
                    &mut [
                        arg(&input),
                        arg(&weight),
                        arg(&normalized),
                        arg(&columns),
                        arg(&EPSILON),
                    ],
                );
                if multiple {
                    ok!(cuEventRecord(events[0], first));
                    ok!(cuStreamWaitEvent(second, events[0], 0));
                }
                launch(
                    kernels[1],
                    elements.div_ceil(threads),
                    threads,
                    second,
                    &mut [arg(&normalized), arg(&residual), arg(&sum), arg(&elements)],
                );
                if multiple {
                    ok!(cuEventRecord(events[1], second));
                    ok!(cuStreamWaitEvent(third, events[1], 0));
                }
                launch(
                    kernels[2],
                    rows,
                    threads,
                    third,
                    &mut [arg(&sum), arg(&result), arg(&columns)],
                );
            }
            if multiple { third } else { first }
        };

        let reference: Vec<_> = input
            .chunks_exact(columns as usize)
            .zip(residual.chunks_exact(columns as usize))
            .map(|(row, residual)| {
                let rms = (row.iter().map(|&v| f64::from(v).powi(2)).sum::<f64>()
                    / f64::from(columns)
                    + f64::from(EPSILON))
                .sqrt();
                row.iter()
                    .zip(&weights)
                    .zip(residual)
                    .map(|((&x, &w), &r)| f64::from(x) * f64::from(w) / rms + f64::from(r))
                    .sum::<f64>()
            })
            .collect();
        let mut first = Vec::new();
        let mut max_error = 0.0f64;
        for multiple in [false, true, false, true] {
            upload(norm, &tensors);
            upload(added, &tensors);
            upload(output, &results);
            ok!(cuStreamSynchronize(submit(multiple, BATCH)));
            let mut actual = results.clone();
            download(output, &mut actual);
            let mut bits = Vec::with_capacity(BATCH * row_count);
            for (slot, result) in actual.chunks_exact(result_stride).enumerate() {
                for (row, &value) in result[GUARD..GUARD + row_count].iter().enumerate() {
                    let expected = reference[slot * row_count + row];
                    let error = (f64::from(value) - expected).abs();
                    let tolerance = 2e-6 * f64::from(columns).sqrt() + 2e-6 * expected.abs();
                    assert!(
                        error <= tolerance,
                        "{rows}x{columns} slot {slot} row {row}: {value} vs {expected}, limit {tolerance}"
                    );
                    max_error = max_error.max(error);
                    bits.push(value.to_bits());
                }
                for (base, stride, len) in [
                    (norm, tensor_stride, tensor_len),
                    (added, tensor_stride, tensor_len),
                    (output, result_stride, row_count),
                ] {
                    for offset in [0, GUARD + len] {
                        let mut guard = [0.0; GUARD];
                        download(base + bytes(slot * stride + offset), &mut guard);
                        assert_eq!(guard.map(f32::to_bits), [SENTINEL.to_bits(); GUARD]);
                    }
                }
            }
            if first.is_empty() {
                first = bits;
            } else {
                assert_eq!(
                    bits, first,
                    "{rows}x{columns} differs across runs or stream modes"
                );
            }
        }
        println!(
            "Forge {rows}x{columns}, block={threads}, 4x{BATCH} chains: max abs={max_error:.9e}"
        );

        if measure && columns == 257 {
            for (pass, modes) in [[false, true], [true, false]].into_iter().enumerate() {
                for multiple in modes {
                    for (batch, count) in [(1, 201), (BATCH, 51)] {
                        for _ in 0..20 {
                            ok!(cuStreamSynchronize(submit(multiple, batch)));
                        }
                        let mut enqueue = Vec::with_capacity(count);
                        let mut completed = Vec::with_capacity(count);
                        for _ in 0..count {
                            let start = Instant::now();
                            let final_stream = submit(multiple, batch);
                            enqueue.push(start.elapsed());
                            ok!(cuStreamSynchronize(final_stream));
                            completed.push(start.elapsed());
                        }
                        let label =
                            format!("Forge pass={} multiple={multiple} batch={batch}", pass + 1);
                        report(&format!("{label} submit"), &mut enqueue);
                        report(&format!("{label} complete"), &mut completed);
                    }
                }
            }
        }
        for address in [x, skip, weight, norm, added, output] {
            ok!(cuMemFree_v2(address));
        }
    }
    for event in events {
        ok!(cuEventDestroy_v2(event));
    }
    ok!(cuCtxDestroy_v2(context));
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn event_driver_api() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuCtxSetCurrent(Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamSynchronize(Handle),
        cuEventCreate(*mut Handle, u32),
        cuEventRecord(Handle, Handle),
        cuEventQuery(Handle),
        cuEventSynchronize(Handle),
        cuEventDestroy_v2(Handle),
        cuStreamWaitEvent(Handle, Handle, u32),
    );
    ok!(cuInit(0));
    let (mut context, mut other, mut event, mut other_event) =
        (null_mut(), null_mut(), null_mut(), null_mut());
    let (mut producer, mut consumer, mut foreign) = (null_mut(), null_mut(), null_mut());
    ok!(cuCtxCreate_v2(&mut context, 0, 0));
    ok!(cuStreamCreate(&mut producer, 1));
    ok!(cuStreamCreate(&mut consumer, 1));
    assert_eq!(unsafe { cuEventCreate(null_mut(), 2) }, 1);
    for (flag, error) in [(4, 1), (8, 1), (1, 801), (3, 801), (6, 801)] {
        assert_eq!(unsafe { cuEventCreate(&mut event, flag) }, error);
        assert!(event.is_null());
    }
    ok!(cuEventCreate(&mut event, 2));
    ok!(cuEventQuery(event));
    ok!(cuEventSynchronize(event));
    ok!(cuStreamWaitEvent(consumer, event, 0));
    assert_eq!(unsafe { cuStreamWaitEvent(consumer, event, 1) }, 1);
    assert_eq!(unsafe { cuEventRecord(null_mut(), producer) }, 400);
    assert_eq!(unsafe { cuStreamWaitEvent(consumer, context, 0) }, 400);
    ok!(cuCtxCreate_v2(&mut other, 0, 0));
    ok!(cuStreamCreate(&mut foreign, 1));
    ok!(cuEventCreate(&mut other_event, 0));
    assert_eq!(unsafe { cuEventRecord(event, foreign) }, 400);
    assert_eq!(unsafe { cuEventQuery(event) }, 400);
    assert_eq!(unsafe { cuEventSynchronize(event) }, 400);
    assert_eq!(unsafe { cuEventDestroy_v2(event) }, 400);
    assert_eq!(unsafe { cuStreamWaitEvent(foreign, event, 0) }, 400);
    ok!(cuCtxSetCurrent(context));
    assert_eq!(unsafe { cuEventRecord(event, foreign) }, 400);
    assert_eq!(unsafe { cuStreamWaitEvent(foreign, event, 0) }, 400);
    let per_thread = std::ptr::without_provenance_mut(2);
    assert_eq!(unsafe { cuEventRecord(event, per_thread) }, 801);
    assert_eq!(unsafe { cuStreamWaitEvent(per_thread, event, 0) }, 801);
    ok!(cuCtxSetCurrent(other));
    ok!(cuCtxDestroy_v2(other));
    assert_eq!(unsafe { cuEventQuery(other_event) }, 400);
    ok!(cuCtxSetCurrent(null_mut()));
    assert_eq!(unsafe { cuEventQuery(event) }, 201);
    ok!(cuCtxSetCurrent(context));

    let (ctx, evt, src, dst) = (
        context.addr(),
        event.addr(),
        producer.addr(),
        consumer.addr(),
    );
    std::thread::scope(|scope| {
        for _ in 0..4 {
            scope.spawn(|| {
                let handle = std::ptr::without_provenance_mut;
                ok!(cuCtxSetCurrent(handle(ctx)));
                for _ in 0..32 {
                    ok!(cuEventRecord(handle(evt), handle(src)));
                    ok!(cuStreamWaitEvent(handle(dst), handle(evt), 0));
                    assert!(matches!(unsafe { cuEventQuery(handle(evt)) }, 0 | 600));
                    ok!(cuEventSynchronize(handle(evt)));
                }
            });
        }
    });
    ok!(cuStreamSynchronize(consumer));
    ok!(cuEventQuery(event));
    ok!(cuEventDestroy_v2(event));
    assert_eq!(unsafe { cuEventDestroy_v2(event) }, 400);
    assert_eq!(unsafe { cuEventRecord(event, producer) }, 400);
    assert_eq!(unsafe { cuStreamWaitEvent(consumer, event, 0) }, 400);

    if std::env::var_os("CUDA_FOR_INTEL_BENCH_EVENTS").is_some() {
        let mut cold = Vec::new();
        for sample in 0..111 {
            let (mut fresh, mut first) = (null_mut(), null_mut());
            ok!(cuCtxCreate_v2(&mut fresh, 0, 0));
            ok!(cuEventCreate(&mut first, 2));
            let start = Instant::now();
            ok!(cuEventRecord(first, null_mut()));
            if sample >= 10 {
                cold.push(start.elapsed());
            }
            ok!(cuEventSynchronize(first));
            ok!(cuCtxDestroy_v2(fresh));
        }
        report("First record (fresh context/pool)", &mut cold);

        let mut create = Vec::new();
        let mut destroy = Vec::new();
        let mut record = Vec::new();
        let mut query = Vec::new();
        let mut wait = Vec::new();
        for sample in 0..701 {
            let start = Instant::now();
            ok!(cuEventCreate(&mut event, 2));
            let created = start.elapsed();
            let start = Instant::now();
            ok!(cuEventRecord(event, null_mut()));
            let recorded = start.elapsed();
            let start = Instant::now();
            ok!(cuStreamWaitEvent(consumer, event, 0));
            let waited = start.elapsed();
            ok!(cuStreamSynchronize(consumer));
            let start = Instant::now();
            ok!(cuEventQuery(event));
            let queried = start.elapsed();
            let start = Instant::now();
            ok!(cuEventDestroy_v2(event));
            let destroyed = start.elapsed();
            if sample >= 200 {
                create.push(created);
                destroy.push(destroyed);
                record.push(recorded);
                query.push(queried);
                wait.push(waited);
            }
        }
        report("Event create", &mut create);
        report("Event destroy (completed)", &mut destroy);
        report("Event record (warm pool)", &mut record);
        report("Event query (completed)", &mut query);
        report("Stream wait append", &mut wait);
    }
    ok!(cuCtxDestroy_v2(context));
}
