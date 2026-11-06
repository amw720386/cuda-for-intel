use super::{BATCH, EPSILON, GUARD, KERNELS, SENTINEL, arg, report};
use crate::{Handle, library, ok, symbols};
use std::ffi::{CString, c_char, c_void};
use std::ptr::null_mut;
use std::time::Instant;

fn bytes(elements: usize) -> usize {
    elements * size_of::<f32>()
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Blocking,
    SingleStream,
    EventLinked,
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn forge_transfer_pipeline() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemHostAlloc(*mut Handle, usize, u32),
        cuMemFreeHost(Handle),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(Handle, u64, usize),
        cuMemcpyHtoDAsync_v2(u64, *const c_void, usize, Handle),
        cuMemcpyDtoHAsync_v2(Handle, u64, usize, Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamSynchronize(Handle),
        cuEventCreate(*mut Handle, u32),
        cuEventRecord(Handle, Handle),
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
    let mut events = [[null_mut(); 3]; 2];
    for event in events.iter_mut().flatten() {
        ok!(cuEventCreate(event, 2));
    }
    let kernels = KERNELS.map(|(name, source)| {
        let (mut module, mut kernel) = (null_mut(), null_mut());
        let ptx = CString::new(source).unwrap();
        ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
        ok!(cuModuleGetFunction(&mut kernel, module, name.as_ptr()));
        kernel
    });
    let launch = |kernel, grid, block, stream, args: &mut [Handle]| {
        ok!(cuLaunchKernel(
            kernel,
            grid,
            1,
            1,
            block,
            1,
            1,
            0,
            stream,
            args.as_mut_ptr(),
            null_mut()
        ));
    };
    let pinned = |values: &[f32]| {
        let mut host = null_mut();
        ok!(cuMemHostAlloc(&mut host, size_of_val(values), 0));
        let host = host.cast::<f32>();
        unsafe { host.copy_from_nonoverlapping(values.as_ptr(), values.len()) };
        host
    };
    let bench = std::env::var_os("CUDA_FOR_INTEL_BENCH_TRANSFERS").is_some();
    let mut seed = 0xf817_a029u32;
    let mut random = || {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (seed >> 8) as f32 / 8_388_608.0 - 1.0
    };
    for (rows, columns, threads) in [(7u32, 257u32, 63u32), (64, 1024, 128)] {
        let elements = rows * columns;
        let (rows_len, cols, tensor) = (rows as usize, columns as usize, elements as usize);
        let result_stride = rows_len + 2 * GUARD;
        let input: Vec<_> = (0..BATCH * tensor).map(|_| random()).collect();
        let residual: Vec<_> = (0..input.len()).map(|_| random() * 0.5).collect();
        let weight: Vec<_> = (0..cols).map(|_| 1.0 + random() * 0.5).collect();
        let reference: Vec<_> = input
            .chunks_exact(cols)
            .zip(residual.chunks_exact(cols))
            .map(|(row, skip)| {
                let rms = (row.iter().map(|&x| f64::from(x).powi(2)).sum::<f64>()
                    / f64::from(columns)
                    + f64::from(EPSILON))
                .sqrt();
                row.iter()
                    .zip(&weight)
                    .zip(skip)
                    .map(|((&x, &w), &r)| f64::from(x) * f64::from(w) / rms + f64::from(r))
                    .sum::<f64>()
            })
            .collect();
        let host_x = pinned(&input);
        let host_skip = pinned(&residual);
        let host_weight = pinned(&weight);
        let result_init = vec![SENTINEL; BATCH * result_stride];
        let host_out = pinned(&result_init);
        let device_slots = [(); 2].map(|()| {
            [
                tensor,
                tensor,
                cols,
                tensor + 2 * GUARD,
                tensor + 2 * GUARD,
                result_stride,
            ]
            .map(|len| {
                let mut ptr = 0;
                ok!(cuMemAlloc_v2(&mut ptr, bytes(len)));
                let init = vec![SENTINEL; len];
                ok!(cuMemcpyHtoD_v2(ptr, init.as_ptr().cast(), bytes(len)));
                ptr
            })
        });

        let run = |mode, slot_count| {
            let [upload, compute, download] = match mode {
                Mode::Blocking => [null_mut(); 3],
                Mode::SingleStream => [streams[0]; 3],
                Mode::EventLinked => streams,
            };
            let start = Instant::now();
            for batch in 0..BATCH {
                let slot = batch % slot_count;
                let [x, skip, w, norm, added, out] = device_slots[slot];
                let [uploaded, computed, downloaded] = events[slot];
                if matches!(mode, Mode::EventLinked) && batch >= slot_count {
                    ok!(cuStreamWaitEvent(upload, downloaded, 0));
                }
                for (device, host, len) in [
                    (x, unsafe { host_x.add(batch * tensor) }, tensor),
                    (skip, unsafe { host_skip.add(batch * tensor) }, tensor),
                    (w, host_weight, cols),
                ] {
                    if matches!(mode, Mode::Blocking) {
                        ok!(cuMemcpyHtoD_v2(device, host.cast(), bytes(len)));
                    } else {
                        ok!(cuMemcpyHtoDAsync_v2(
                            device,
                            host.cast(),
                            bytes(len),
                            upload
                        ));
                    }
                }
                if matches!(mode, Mode::EventLinked) {
                    ok!(cuEventRecord(uploaded, upload));
                    ok!(cuStreamWaitEvent(compute, uploaded, 0));
                }
                let normalized = norm + bytes(GUARD) as u64;
                let sum = added + bytes(GUARD) as u64;
                let result = out + bytes(GUARD) as u64;
                launch(
                    kernels[0],
                    rows,
                    threads,
                    compute,
                    &mut [
                        arg(&x),
                        arg(&w),
                        arg(&normalized),
                        arg(&columns),
                        arg(&EPSILON),
                    ],
                );
                launch(
                    kernels[1],
                    elements.div_ceil(threads),
                    threads,
                    compute,
                    &mut [arg(&normalized), arg(&skip), arg(&sum), arg(&elements)],
                );
                launch(
                    kernels[2],
                    rows,
                    threads,
                    compute,
                    &mut [arg(&sum), arg(&result), arg(&columns)],
                );
                if matches!(mode, Mode::EventLinked) {
                    ok!(cuEventRecord(computed, compute));
                    ok!(cuStreamWaitEvent(download, computed, 0));
                }
                let dst = unsafe { host_out.add(batch * result_stride + GUARD) };
                if matches!(mode, Mode::Blocking) {
                    ok!(cuMemcpyDtoH_v2(dst.cast(), result, bytes(rows_len)));
                } else {
                    ok!(cuMemcpyDtoHAsync_v2(
                        dst.cast(),
                        result,
                        bytes(rows_len),
                        download
                    ));
                }
                if matches!(mode, Mode::EventLinked) {
                    ok!(cuEventRecord(downloaded, download));
                }
            }
            let submitted = start.elapsed();
            let sync_start = Instant::now();
            if !matches!(mode, Mode::Blocking) {
                ok!(cuStreamSynchronize(download));
            }
            (submitted, sync_start.elapsed(), start.elapsed())
        };
        let mut baseline = Vec::new();
        let mut max_error = 0.0f64;
        for (mode, slots) in [
            (Mode::Blocking, 1),
            (Mode::SingleStream, 1),
            (Mode::EventLinked, 1),
            (Mode::EventLinked, 2),
        ] {
            for _ in 0..3 {
                unsafe {
                    host_out.copy_from_nonoverlapping(result_init.as_ptr(), result_init.len())
                };
                run(mode, slots);
                let actual = unsafe { std::slice::from_raw_parts(host_out, result_init.len()) };
                let mut bits = Vec::new();
                for (batch, result) in actual.chunks_exact(result_stride).enumerate() {
                    assert_eq!(&result[..GUARD], &[SENTINEL; GUARD]);
                    assert_eq!(&result[GUARD + rows_len..], &[SENTINEL; GUARD]);
                    for (row, &value) in result[GUARD..GUARD + rows_len].iter().enumerate() {
                        let expected = reference[batch * rows_len + row];
                        let error = (f64::from(value) - expected).abs();
                        let tolerance = 2e-6 * f64::from(columns).sqrt() + 2e-6 * expected.abs();
                        assert!(
                            error <= tolerance,
                            "{mode:?} {rows}x{columns} batch {batch} row {row}: {value} vs {expected}"
                        );
                        max_error = max_error.max(error);
                        bits.push(value.to_bits());
                    }
                }
                if baseline.is_empty() {
                    baseline = bits;
                } else {
                    assert_eq!(bits, baseline, "{mode:?} {rows}x{columns} slots={slots}");
                }
                for &[_, _, _, norm, added, out] in &device_slots[..slots] {
                    for (base, len) in [(norm, tensor), (added, tensor), (out, rows_len)] {
                        for offset in [0, GUARD + len] {
                            let mut guard = [0.0; GUARD];
                            ok!(cuMemcpyDtoH_v2(
                                guard.as_mut_ptr().cast(),
                                base + bytes(offset) as u64,
                                bytes(GUARD)
                            ));
                            assert_eq!(guard, [SENTINEL; GUARD]);
                        }
                    }
                }
            }
            if bench {
                for _ in 0..5 {
                    run(mode, slots);
                }
                let (mut submit, mut sync, mut complete) = (Vec::new(), Vec::new(), Vec::new());
                for _ in 0..31 {
                    let (s, w, c) = run(mode, slots);
                    submit.push(s);
                    sync.push(w);
                    complete.push(c);
                }
                let label =
                    format!("Forge {rows}x{columns} {mode:?} slots={slots} batches={BATCH}");
                if !matches!(mode, Mode::Blocking) {
                    report(&format!("{label} submit"), &mut submit);
                    report(&format!("{label} synchronize"), &mut sync);
                }
                report(&format!("{label} complete"), &mut complete);
            }
        }
        println!(
            "Pinned Forge {rows}x{columns}: 12x{BATCH} chains, bytes/batch={}, max abs={max_error:.9e}",
            bytes(2 * tensor + cols + rows_len)
        );
        for ptr in device_slots.into_iter().flatten() {
            ok!(cuMemFree_v2(ptr));
        }
        for ptr in [host_x, host_skip, host_weight, host_out] {
            ok!(cuMemFreeHost(ptr.cast()));
        }
    }
    ok!(cuCtxDestroy_v2(context));
}
