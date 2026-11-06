use super::streams::{cuStreamCreate, cuStreamSynchronize};
use super::*;
use std::ptr::null_mut;
use std::time::{Duration, Instant};

const WARMUP: usize = 20;
const SAMPLES: usize = 101;

fn report(label: &str, samples: &mut [Duration], bytes: Option<usize>) {
    samples.sort_unstable();
    let median = samples[samples.len() / 2].as_secs_f64();
    let p95 = samples[(95 * samples.len()).div_ceil(100) - 1].as_secs_f64();
    print!(
        "{label}: n={} median={:.3} us p95={:.3} us",
        samples.len(),
        median * 1e6,
        p95 * 1e6
    );
    if let Some(bytes) = bytes {
        print!(" effective={:.3} GB/s", bytes as f64 / median / 1e9);
    }
    println!();
}

#[test]
#[ignore = "release benchmark requiring an Intel Level Zero GPU"]
fn pinned_transfer_benchmark() {
    assert_eq!(cuInit(0), Success);
    let (mut cuda, mut stream) = (null_mut(), null_mut());
    assert_eq!(unsafe { cuCtxCreate_v2(&mut cuda, 0, 0) }, Success);
    assert_eq!(unsafe { cuStreamCreate(&mut stream, 1) }, Success);
    let runtime = Runtime::load().unwrap();
    let api = &runtime.api;
    let mut direct = Context::create(api, &runtime.devices[0]).unwrap();
    direct.initialize(api).unwrap();
    let list = direct.create_stream(api).unwrap();

    for size in [4096, 65536, 1 << 20, 16 << 20] {
        let (mut alloc, mut free) = (Vec::new(), Vec::new());
        for sample in 0..WARMUP + SAMPLES {
            let mut host = null_mut();
            let start = Instant::now();
            assert_eq!(unsafe { cuMemHostAlloc(&mut host, size, 0) }, Success);
            let allocated = start.elapsed();
            let start = Instant::now();
            assert_eq!(cuMemFreeHost(host), Success);
            let freed = start.elapsed();
            if sample >= WARMUP {
                alloc.push(allocated);
                free.push(freed);
            }
        }
        report(&format!("Pinned alloc bytes={size}"), &mut alloc, None);
        report(&format!("Pinned free bytes={size}"), &mut free, None);

        let (mut cuda_host, mut cuda_device) = (null_mut(), 0);
        assert_eq!(unsafe { cuMemHostAlloc(&mut cuda_host, size, 0) }, Success);
        assert_eq!(unsafe { cuMemAlloc_v2(&mut cuda_device, size) }, Success);
        let direct_host = direct.alloc_host(api, size).unwrap();
        let direct_device = direct.alloc(api, size).unwrap();
        let host = std::ptr::with_exposed_provenance_mut::<c_void>(direct_host);
        let device = std::ptr::with_exposed_provenance_mut::<c_void>(direct_device);
        for (backend, async_copy) in [
            ("CUDA", false),
            ("Level Zero", false),
            ("CUDA", true),
            ("Level Zero", true),
        ] {
            let host_ptr = if backend == "CUDA" { cuda_host } else { host };
            unsafe { host_ptr.cast::<u8>().write_bytes(0x67, size) };
            for (direction, upload) in [("HtoD", true), ("DtoH", false)] {
                let (mut submit, mut sync, mut complete) = (Vec::new(), Vec::new(), Vec::new());
                if !upload {
                    unsafe { host_ptr.cast::<u8>().write_bytes(0, size) };
                }
                for sample in 0..WARMUP + SAMPLES {
                    let start = Instant::now();
                    if backend == "CUDA" {
                        let result = match (async_copy, upload) {
                            (false, true) => unsafe {
                                cuMemcpyHtoD_v2(cuda_device, cuda_host, size)
                            },
                            (false, false) => unsafe {
                                cuMemcpyDtoH_v2(cuda_host, cuda_device, size)
                            },
                            (true, true) => unsafe {
                                cuMemcpyHtoDAsync_v2(cuda_device, cuda_host, size, stream)
                            },
                            (true, false) => unsafe {
                                cuMemcpyDtoHAsync_v2(cuda_host, cuda_device, size, stream)
                            },
                        };
                        assert_eq!(result, Success);
                    } else {
                        let (dst, src) = if upload {
                            (device, host)
                        } else {
                            (host, device)
                        };
                        if async_copy {
                            unsafe { list.copy(api, (dst, src), size, None, &[]) }.unwrap();
                        } else {
                            unsafe { direct.copy(api, dst, src, size) }.unwrap();
                        }
                    }
                    let submitted = start.elapsed();
                    let sync_start = Instant::now();
                    if async_copy {
                        if backend == "CUDA" {
                            assert_eq!(cuStreamSynchronize(stream), Success);
                        } else {
                            list.wait(api, u64::MAX).unwrap();
                        }
                    }
                    let synced = sync_start.elapsed();
                    let completed = start.elapsed();
                    if sample >= WARMUP {
                        submit.push(submitted);
                        sync.push(synced);
                        complete.push(completed);
                    }
                }
                let label = format!("{backend} async={async_copy} {direction} bytes={size}");
                if async_copy {
                    report(&format!("{label} submit"), &mut submit, None);
                    report(&format!("{label} synchronize"), &mut sync, None);
                }
                report(&format!("{label} complete"), &mut complete, Some(size));
            }
            let data = unsafe { std::slice::from_raw_parts(host_ptr.cast::<u8>(), size) };
            assert!(data.iter().all(|&byte| byte == 0x67));
        }
        assert_eq!(cuMemFreeHost(cuda_host), Success);
        assert_eq!(cuMemFree_v2(cuda_device), Success);
        direct.free(api, direct_host).unwrap();
        direct.free(api, direct_device).unwrap();
    }
    list.destroy(api).unwrap();
    direct.destroy(api).unwrap();
    assert_eq!(cuCtxDestroy_v2(cuda), Success);
}
