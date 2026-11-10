//! Throughput of the f64/complex cuBLAS kernels and of cuFFT (timing only, no checks beyond
//! status codes; correctness lives in `hw_cublas64`, `hw_cufft` and `hw_cusparse`).
//!
//! `cargo test --release --test gpu hw_libs_perf -- --ignored --nocapture --test-threads=1`
//! prints one line per case: GFLOP/s for BLAS, GB/s for the bandwidth-bound routines and
//! FFTs (bytes = one read and one write of the data per transform, the minimum any
//! out-of-place FFT moves), next to a device-to-device copy as the bandwidth roofline.
//! `CFI_PERF_QUICK=1` runs one repetition of each case.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::c_void;
use std::ptr::null_mut;
use std::time::Instant;

type P = *mut c_void;
type C = *const c_void;

fn fill(len: usize, seed: u64) -> Vec<u8> {
    let mut s = seed | 1;
    let mut out = Vec::with_capacity(len * 8);
    for _ in 0..len {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        let v = (s % 2001) as f64 / 1000.0 - 1.0;
        out.extend(v.to_le_bytes());
    }
    out
}

#[test]
#[ignore = "requires nvcuda.dll, cublas64_12.dll, cufft64_11.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case, clippy::type_complexity)]
fn hw_libs_perf() {
    let dir = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().to_path_buf();
    let blas = unsafe { libloading::Library::new(dir.join("cublas64_12.dll")) }.expect("cublas64_12.dll");
    let fft = unsafe { libloading::Library::new(dir.join("cufft64_11.dll")) }.expect("cufft64_11.dll");
    fn get<T: Copy>(lib: &libloading::Library, name: &str) -> T {
        *unsafe { lib.get::<T>(format!("{name}\0").as_bytes()) }.unwrap_or_else(|_| panic!("{name}"))
    }
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoD_v2(u64, u64, usize),
        cuMemFree_v2(u64),
        cuCtxSynchronize(),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let quick = std::env::var("CFI_PERF_QUICK").is_ok_and(|v| v == "1");
    let alloc = |words: usize, seed: u64| -> u64 {
        let mut d = 0u64;
        ok!(cuMemAlloc_v2(&mut d, words * 8));
        let b = fill(words, seed);
        ok!(cuMemcpyHtoD_v2(d, b.as_ptr().cast(), b.len()));
        d
    };
    // Best of `reps` timed runs (seconds) after one warm-up.
    let time = |reps: usize, f: &mut dyn FnMut() -> i32| -> f64 {
        assert_eq!(f(), 0, "warm-up status");
        ok!(cuCtxSynchronize());
        let mut best = f64::INFINITY;
        for _ in 0..if quick { 1 } else { reps } {
            let t = Instant::now();
            assert_eq!(f(), 0);
            ok!(cuCtxSynchronize());
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    };

    // Roofline: device-to-device copy of 64 MiB (read + write).
    let words = 8 << 20;
    let (x, y) = (alloc(words, 1), alloc(words, 2));
    let t = time(5, &mut || unsafe { cuMemcpyDtoD_v2(y, x, words * 8) });
    let roof = 2.0 * (words * 8) as f64 / t / 1e9;
    println!("perf copy DtoD 64MiB: {roof:.1} GB/s (roofline)");

    let create: unsafe extern "system" fn(*mut P) -> i32 = get(&blas, "cublasCreate_v2");
    let mut h: P = null_mut();
    assert_eq!(unsafe { create(&mut h) }, 0);
    let one = [1.0f64, 0.0];
    let zero = [0.0f64, 0.0];
    let (o, z) = (one.as_ptr() as C, zero.as_ptr() as C);
    let (one32, zero32) = ([1.0f32, 0.0], [0.0f32, 0.0]);

    type Gemm = unsafe extern "system" fn(P, i32, i32, i32, i32, i32, C, C, i32, C, i32, C, P, i32) -> i32;
    for (name, flops_per, sizes) in [
        ("cublasDgemm_v2", 2.0, &[128usize, 256, 512, 1024][..]),
        ("cublasZgemm_v2", 8.0, &[128, 256, 512][..]),
        ("cublasCgemm_v2", 8.0, &[256, 512][..]),
    ] {
        let g: Gemm = get(&blas, name);
        let (o, z) = if name.starts_with("cublasC") { (one32.as_ptr() as C, zero32.as_ptr() as C) } else { (o, z) };
        for &n in sizes {
            let words = n * n * 2;
            let (a, b, c) = (alloc(words, 3), alloc(words, 4), alloc(words, 5));
            for (ta, tb) in [(0, 0), (1, 0)] {
                let ni = n as i32;
                let t = time(3, &mut || unsafe { g(h, ta, tb, ni, ni, ni, o, a as C, ni, b as C, ni, z, c as P, ni) });
                let gf = flops_per * (n * n * n) as f64 / t / 1e9;
                println!("perf {name} {}{} n={n}: {gf:.2} GFLOP/s ({:.3} ms)", ["N", "T"][ta as usize], ["N", "T"][tb as usize], t * 1e3);
            }
            for d in [a, b, c] {
                ok!(cuMemFree_v2(d));
            }
        }
    }

    // Level 1 reductions at 4M elements (bandwidth bound).
    let n = 4usize << 20;
    let (xv, yv, out) = (alloc(2 * n, 6), alloc(2 * n, 7), alloc(4, 8));
    unsafe {
        let set_mode: unsafe extern "system" fn(P, i32) -> i32 = get(&blas, "cublasSetPointerMode_v2");
        assert_eq!(set_mode(h, 1), 0); // device results: time the kernel, not the copy back
        let ddot: unsafe extern "system" fn(P, i32, C, i32, C, i32, P) -> i32 = get(&blas, "cublasDdot_v2");
        let zdot: unsafe extern "system" fn(P, i32, C, i32, C, i32, P) -> i32 = get(&blas, "cublasZdotc_v2");
        let dnrm: unsafe extern "system" fn(P, i32, C, i32, P) -> i32 = get(&blas, "cublasDnrm2_v2");
        let znrm: unsafe extern "system" fn(P, i32, C, i32, P) -> i32 = get(&blas, "cublasDznrm2_v2");
        let amax: unsafe extern "system" fn(P, i32, C, i32, P) -> i32 = get(&blas, "cublasIdamax_v2");
        let ni = n as i32;
        let gb = |bytes: usize, t: f64| bytes as f64 / t / 1e9;
        let t = time(5, &mut || ddot(h, ni, xv as C, 1, yv as C, 1, out as P));
        println!("perf cublasDdot n={n}: {:.1} GB/s ({:.3} ms)", gb(16 * n, t), t * 1e3);
        let t = time(5, &mut || zdot(h, ni, xv as C, 1, yv as C, 1, out as P));
        println!("perf cublasZdotc n={n}: {:.1} GB/s ({:.3} ms)", gb(32 * n, t), t * 1e3);
        let t = time(5, &mut || dnrm(h, ni, xv as C, 1, out as P));
        println!("perf cublasDnrm2 n={n}: {:.1} GB/s ({:.3} ms)", gb(8 * n, t), t * 1e3);
        let t = time(5, &mut || znrm(h, ni, xv as C, 1, out as P));
        println!("perf cublasDznrm2 n={n}: {:.1} GB/s ({:.3} ms)", gb(16 * n, t), t * 1e3);
        let t = time(5, &mut || amax(h, ni, xv as C, 1, out as P));
        println!("perf cublasIdamax n={n}: {:.1} GB/s ({:.3} ms)", gb(8 * n, t), t * 1e3);
        assert_eq!(set_mode(h, 0), 0);
    }
    for d in [xv, yv, out] {
        ok!(cuMemFree_v2(d));
    }

    // trsm (left, lower, N, non-unit): m^2 n flops.
    type Trsm = unsafe extern "system" fn(P, i32, i32, i32, i32, i32, i32, C, C, i32, P, i32) -> i32;
    for (name, fl) in [("cublasDtrsm_v2", 1.0), ("cublasZtrsm_v2", 4.0)] {
        let f: Trsm = get(&blas, name);
        for n in [256usize, 1024] {
            // diagonally dominant lower triangle: add n on the diagonal
            let mut av = fill(2 * n * n, 9);
            let es = if fl > 1.0 { 16 } else { 8 };
            for i in 0..n {
                let o = (i * n + i) * es;
                let v = f64::from_le_bytes(av[o..o + 8].try_into().unwrap()) + n as f64;
                av[o..o + 8].copy_from_slice(&v.to_le_bytes());
            }
            let mut a = 0u64;
            ok!(cuMemAlloc_v2(&mut a, av.len()));
            ok!(cuMemcpyHtoD_v2(a, av.as_ptr().cast(), av.len()));
            let b = alloc(2 * n * n, 10);
            let ni = n as i32;
            let t = time(2, &mut || unsafe { f(h, 0, 0, 0, 0, ni, ni, o, a as C, ni, b as P, ni) });
            let gf = fl * (n * n * n) as f64 / t / 1e9;
            println!("perf {name} LLNN m=n={n}: {gf:.2} GFLOP/s ({:.3} ms)", t * 1e3);
            ok!(cuMemFree_v2(a));
            ok!(cuMemFree_v2(b));
        }
    }

    // getrfBatched with pivoting: 2/3 n^3 flops per matrix.
    type Getrf = unsafe extern "system" fn(P, i32, C, i32, P, P, i32) -> i32;
    let getrf: Getrf = get(&blas, "cublasDgetrfBatched");
    for (n, batch) in [(8usize, 10000usize), (32, 2000), (64, 256), (128, 64)] {
        let base = alloc(n * n * batch, 11);
        let table: Vec<u8> = (0..batch).flat_map(|i| (base + (i * n * n * 8) as u64).to_le_bytes()).collect();
        let mut tab = 0u64;
        ok!(cuMemAlloc_v2(&mut tab, table.len()));
        ok!(cuMemcpyHtoD_v2(tab, table.as_ptr().cast(), table.len()));
        let (piv, info) = (alloc(n * batch, 12), alloc(batch, 13));
        let fresh = alloc(n * n * batch, 11);
        let ni = n as i32;
        // refactor the same input each time (copy it back before the timed call)
        let mut f = || unsafe {
            cuMemcpyDtoD_v2(base, fresh, n * n * batch * 8);
            getrf(h, ni, tab as C, ni, piv as P, info as P, batch as i32)
        };
        let t_all = time(3, &mut f);
        let t_copy = time(3, &mut || unsafe { cuMemcpyDtoD_v2(base, fresh, n * n * batch * 8) });
        let t = (t_all - t_copy).max(1e-9);
        let gf = 2.0 / 3.0 * (n * n * n * batch) as f64 / t / 1e9;
        println!("perf cublasDgetrfBatched n={n} batch={batch}: {gf:.2} GFLOP/s ({:.3} ms)", t * 1e3);
        for d in [base, tab, piv, info, fresh] {
            ok!(cuMemFree_v2(d));
        }
    }

    // cuFFT: out-of-place forward transforms.
    let plan1d: unsafe extern "system" fn(*mut i32, i32, i32, i32) -> i32 = get(&fft, "cufftPlan1d");
    let destroy: unsafe extern "system" fn(i32) -> i32 = get(&fft, "cufftDestroy");
    let c2c: unsafe extern "system" fn(i32, P, P, i32) -> i32 = get(&fft, "cufftExecC2C");
    let z2z: unsafe extern "system" fn(i32, P, P, i32) -> i32 = get(&fft, "cufftExecZ2Z");
    for (double, n, batch) in [
        (false, 1usize << 20, 1usize),
        (false, 4096, 256),
        (false, 256, 4096),
        (false, 1000, 1000),
        (true, 1 << 18, 1),
        (true, 4096, 64),
        (false, 4099, 64),
        (false, 10007, 16),
        (true, 10007, 16),
        (false, 65537, 2),
        (false, 1 << 22, 1),
    ] {
        let es = if double { 16 } else { 8 };
        let words = n * batch * es / 8;
        let (xi, xo) = (alloc(words, 14), alloc(words, 15));
        let mut plan = 0;
        assert_eq!(unsafe { plan1d(&mut plan, n as i32, if double { 0x69 } else { 0x29 }, batch as i32) }, 0);
        let t = if double {
            time(3, &mut || unsafe { z2z(plan, xi as P, xo as P, -1) })
        } else {
            time(3, &mut || unsafe { c2c(plan, xi as P, xo as P, -1) })
        };
        let bytes = 2 * n * batch * es;
        let gbs = bytes as f64 / t / 1e9;
        let gflops = 5.0 * (n * batch) as f64 * (n as f64).log2() / t / 1e9;
        println!(
            "perf cufft {} n={n} batch={batch}: {gbs:.1} GB/s ({:.0}% of copy), {gflops:.1} GFLOP/s ({:.3} ms)",
            if double { "Z2Z" } else { "C2C" },
            100.0 * gbs / roof,
            t * 1e3
        );
        assert_eq!(unsafe { destroy(plan) }, 0);
        ok!(cuMemFree_v2(xi));
        ok!(cuMemFree_v2(xo));
    }
    ok!(cuMemFree_v2(x));
    ok!(cuMemFree_v2(y));
}
