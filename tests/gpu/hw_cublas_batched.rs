//! Batched fp16 GEMM through the cuBLAS shim as used for GQA attention
//! (cublasGemmStridedBatchedEx / cublasGemmBatchedEx), checked against a CPU reference.
//! A watchdog aborts the process if a launch hangs.
use super::hw_common::*;
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::c_void;
use std::ptr::{from_mut, null, null_mut};

type Create = unsafe extern "system" fn(*mut *mut c_void) -> i32;
type Strided = unsafe extern "system" fn(
    *mut c_void, i32, i32, i32, i32, i32, *const c_void, *const c_void, i32, i32, i64,
    *const c_void, i32, i32, i64, *const c_void, *mut c_void, i32, i32, i64, i32, i32, i32,
) -> i32;
type Batched = unsafe extern "system" fn(
    *mut c_void, i32, i32, i32, i32, i32, *const c_void, *const *const c_void, i32, i32,
    *const *const c_void, i32, i32, *const c_void, *const *mut c_void, i32, i32, i32, i32, i32,
) -> i32;

struct Case {
    name: &'static str,
    ta: i32, // 0 = N, 1 = T
    m: i32,
    n: i32,
    k: i32,
    batch_a: i32, // distinct A matrices
    ratio: i32,   // batch / batch_a
}

fn f16_matrix(rng: &mut Rng, len: usize) -> (Vec<u16>, Vec<f32>) {
    let vals: Vec<f32> = (0..len).map(|_| rng.range(-64, 64) as f32 / 64.0).collect();
    (vals.iter().map(|v| f32_to_f16(*v)).collect(), vals.iter().map(|v| f16_to_f32(f32_to_f16(*v))).collect())
}

#[allow(non_snake_case)]
fn run_case(c: &Case, pointer_arrays: bool, c_f16: bool, rng: &mut Rng) {
    let lib = unsafe { libloading::Library::new(std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("cublas64_12.dll")) }.expect("cublas64_12.dll");
    let (m, n, k) = (c.m as usize, c.n as usize, c.k as usize);
    let batch = (c.batch_a * c.ratio) as usize;
    // A: m x k column-major, stored transposed for T (k x m with lda = k)
    let (lda, a_elems) = if c.ta == 1 { (k, k * m) } else { (m, m * k) };
    let (ldb, b_elems) = (k, k * n);
    let ldc = m;
    let (a16, af) = f16_matrix(rng, a_elems * c.batch_a as usize);
    let (b16, bf) = f16_matrix(rng, b_elems * batch);
    let c_elem = if c_f16 { 2 } else { 4 };
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuMemFree_v2(u64),
        cuCtxSynchronize(),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let (ab, bb) = (bytes(&a16), bytes(&b16));
    // Exactly-sized allocations so any overrun lands outside the buffer.
    let (mut da, mut db, mut dc) = (0u64, 0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut da, ab.len()));
    ok!(cuMemAlloc_v2(&mut db, bb.len()));
    ok!(cuMemAlloc_v2(&mut dc, m * n * batch * c_elem));
    ok!(cuMemcpyHtoD_v2(da, ab.as_ptr().cast(), ab.len()));
    ok!(cuMemcpyHtoD_v2(db, bb.as_ptr().cast(), bb.len()));
    let zero = vec![0u8; m * n * batch * c_elem];
    ok!(cuMemcpyHtoD_v2(dc, zero.as_ptr().cast(), zero.len()));
    let create: libloading::Symbol<Create> = unsafe { lib.get(b"cublasCreate_v2\0") }.unwrap();
    let mut handle = null_mut();
    assert_eq!(unsafe { create(&mut handle) }, 0, "cublasCreate_v2");
    let (alpha, beta) = (1.0f32, 0.0f32);
    let (a_bytes, b_bytes, c_bytes) = (a_elems * 2, b_elems * 2, m * n * c_elem);
    let c_type = if c_f16 { 2 } else { 0 };
    let status = if pointer_arrays {
        let mut table = |base: u64, per: usize, div: usize| -> u64 {
            let ptrs: Vec<u64> = (0..batch).map(|i| base + ((i / div) * per) as u64).collect();
            let mut dev = 0u64;
            assert_eq!(unsafe { cuMemAlloc_v2(&mut dev, ptrs.len() * 8) }, 0);
            let raw: Vec<u8> = ptrs.iter().flat_map(|p| p.to_le_bytes()).collect();
            assert_eq!(unsafe { cuMemcpyHtoD_v2(dev, raw.as_ptr().cast(), raw.len()) }, 0);
            dev
        };
        let (ta, tb, tc) = (table(da, a_bytes, c.ratio as usize), table(db, b_bytes, 1), table(dc, c_bytes, 1));
        let f: libloading::Symbol<Batched> = unsafe { lib.get(b"cublasGemmBatchedEx\0") }.unwrap();
        unsafe {
            f(handle, c.ta, 0, c.m, c.n, c.k, from_ref(&alpha), ta as *const *const c_void, 2, lda as i32, tb as *const *const c_void, 2, ldb as i32, from_ref(&beta), tc as *const *mut c_void, c_type, ldc as i32, batch as i32, 68, -1)
        }
    } else {
        // Strided batching cannot express the 1/ratio broadcast: use batch_a == batch.
        assert_eq!(c.ratio, 1, "strided variant needs ratio 1");
        let f: libloading::Symbol<Strided> = unsafe { lib.get(b"cublasGemmStridedBatchedEx\0") }.unwrap();
        unsafe {
            f(handle, c.ta, 0, c.m, c.n, c.k, from_ref(&alpha), da as *const c_void, 2, lda as i32, a_elems as i64, db as *const c_void, 2, ldb as i32, b_elems as i64, from_ref(&beta), dc as *mut c_void, c_type, ldc as i32, (m * n) as i64, batch as i32, 68, -1)
        }
    };
    let sync = unsafe { cuCtxSynchronize() };
    assert_eq!((status, sync), (0, 0), "{} pointer_arrays={pointer_arrays} c_f16={c_f16}: gemm status / sync", c.name);
    let mut out = vec![0u8; m * n * batch * c_elem];
    ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), dc, out.len()));
    let get = |idx: usize| -> f32 {
        if c_f16 { f16_to_f32(u16::from_le_bytes([out[2 * idx], out[2 * idx + 1]])) } else { f32::from_le_bytes(out[4 * idx..4 * idx + 4].try_into().unwrap()) }
    };
    let mut worst = 0f64;
    for z in 0..batch {
        let ab = (z / c.ratio as usize) * a_elems;
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0f64;
                for l in 0..k {
                    let a = if c.ta == 1 { af[ab + i * lda + l] } else { af[ab + l * lda + i] };
                    acc += a as f64 * bf[z * b_elems + j * ldb + l] as f64;
                }
                let got = get(z * m * n + j * ldc + i) as f64;
                let tol = if c_f16 { 2e-3 * acc.abs() + 2e-3 } else { 1e-4 * acc.abs() + 1e-4 };
                worst = worst.max((got - acc).abs());
                assert!((got - acc).abs() <= tol, "{} z={z} i={i} j={j}: got {got} want {acc}", c.name);
            }
        }
    }
    println!("{} pointers={pointer_arrays} f16_out={c_f16}: max abs err {worst:.3e}", c.name);
}

fn from_ref<T>(v: &T) -> *const c_void {
    std::ptr::from_ref(v).cast()
}

fn with_watchdog<F: FnOnce() + Send + 'static>(secs: u64, f: F) {
    let (tx, rx) = std::sync::mpsc::channel();
    let h = std::thread::spawn(move || {
        f();
        let _ = tx.send(());
    });
    match rx.recv_timeout(std::time::Duration::from_secs(secs)) {
        Ok(()) => h.join().unwrap(),
        // The worker dropped the sender without finishing: it panicked (a failed assertion
        // is not a hang). Re-raise it so the real message is reported.
        Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => match h.join() {
            Err(p) => std::panic::resume_unwind(p),
            Ok(()) => panic!("watchdog worker ended without a result"),
        },
        Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
            eprintln!("WATCHDOG: GEMM hung for {secs}s");
            std::process::exit(99);
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll, cublas64_12.dll and an Intel Level Zero GPU"]
fn hw_gqa_batched_f16_gemm_failing_shapes() {
    with_watchdog(180, || {
        let mut rng = Rng(0x6a71_6a71_6a71_6a71);
        let cases = [
            Case { name: "kq 512x512x64", ta: 1, m: 512, n: 512, k: 64, batch_a: 2, ratio: 7 },
            Case { name: "kqv 64x32x128", ta: 0, m: 64, n: 32, k: 128, batch_a: 2, ratio: 7 },
            Case { name: "kqv 64x32x512", ta: 0, m: 64, n: 32, k: 512, batch_a: 2, ratio: 7 },
            Case { name: "kqv 64x512x512", ta: 0, m: 64, n: 512, k: 512, batch_a: 2, ratio: 7 },
        ];
        for c in &cases {
            for c_f16 in [false, true] {
                for rep in 0..3 {
                    let _ = rep;
                    run_case(c, true, c_f16, &mut rng);
                }
            }
            let strided = Case { ratio: 1, batch_a: 14, ..Case { name: c.name, ta: c.ta, m: c.m, n: c.n, k: c.k, batch_a: 0, ratio: 1 } };
            run_case(&strided, false, false, &mut rng);
        }
    });
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_k_compute_batched_ptrs_broadcast_tables() {
    let ptx = include_str!("../../examples/k_compute_batched_ptrs_dump.ptx");
    let entry = ptx
        .lines()
        .find_map(|l| l.strip_prefix(".entry ").map(|r| r.trim_end_matches('(').to_string()))
        .unwrap();
    // (ne12, ne13, r2, r3)
    for (ne12, ne13, r2, r3) in [(14i64, 1i64, 7i64, 1i64), (14, 3, 7, 3), (8, 2, 1, 1), (32, 1, 4, 1), (5, 7, 5, 7)] {
        let ne23 = ne12 * ne13;
        let (nb02, nb03, nb12, nb13, nbd2, nbd3) = (65536i64, 131072 * 3, 4096i64, 4096 * ne12, 2048i64, 2048 * ne12);
        let (s0, s1, d) = (0x1000_0000u64, 0x2000_0000u64, 0x3000_0000u64);
        let mut bufs = vec![vec![0u8; (2 * ne23 as usize) * 8], vec![0u8; ne23 as usize * 8]];
        let args = [
            Arg::Bytes(s0.to_le_bytes().to_vec()),
            Arg::Bytes(s1.to_le_bytes().to_vec()),
            Arg::Bytes(d.to_le_bytes().to_vec()),
            Arg::Buf(0),
            Arg::Buf(1),
            i64_arg(ne12), i64_arg(ne13), i64_arg(ne23),
            i64_arg(nb02), i64_arg(nb03), i64_arg(nb12), i64_arg(nb13), i64_arg(nbd2), i64_arg(nbd3),
            i64_arg(r2), i64_arg(r3),
        ];
        launch_args(ptx, &entry, [(ne13 as u32).div_ceil(16), (ne12 as u32).div_ceil(16), 1], [16, 16, 1], 0, &mut bufs, &args)
            .unwrap_or_else(|e| panic!("k_compute_batched_ptrs: {e}"));
        let src = u64s(&bufs[0]);
        let dst = u64s(&bufs[1]);
        for i13 in 0..ne13 {
            for i12 in 0..ne12 {
                let idx = (i12 + i13 * ne12) as usize;
                let (i02, i03) = (i12 / r2, i13 / r3);
                assert_eq!(src[idx], s0 + (i02 * nb02 + i03 * nb03) as u64, "A ptr (ne12={ne12} r2={r2}) i12={i12} i13={i13}");
                assert_eq!(src[ne23 as usize + idx], s1 + (i12 * nb12 + i13 * nb13) as u64, "B ptr i12={i12} i13={i13}");
                assert_eq!(dst[idx], d + (i12 * nbd2 + i13 * nbd3) as u64, "C ptr i12={i12} i13={i13}");
            }
        }
    }
}

fn u64s(b: &[u8]) -> Vec<u64> {
    b.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

/// The ggml flow: k_compute_batched_ptrs writes the pointer tables on a stream and
/// cublasGemmBatchedEx (async shim mode) reads them on the same stream with no sync
/// in between, repeated back to back. A race or ordering bug shows up as a fault.
#[test]
#[ignore = "requires a built nvcuda.dll, cublas64_12.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_gqa_ptr_table_then_gemm_back_to_back_on_a_stream() {
    unsafe { std::env::set_var("CFI_CUBLAS_ASYNC", "1") };
    with_watchdog(200, || {
        let lib = unsafe { libloading::Library::new(std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("cublas64_12.dll")) }.unwrap();
        let ptx = include_str!("../../examples/k_compute_batched_ptrs_dump.ptx");
        let entry = ptx.lines().find_map(|l| l.strip_prefix(".entry ").map(|r| r.trim_end_matches('(').to_string())).unwrap();
        let dll = library();
        symbols!(dll;
            cuInit(u32),
            cuCtxCreate_v2(*mut Handle, u32, i32),
            cuStreamCreate(*mut Handle, u32),
            cuStreamSynchronize(Handle),
            cuModuleLoadData(*mut Handle, *const c_void),
            cuModuleGetFunction(*mut Handle, Handle, *const std::ffi::c_char),
            cuMemAlloc_v2(*mut u64, usize),
            cuMemGetInfo_v2(*mut usize, *mut usize),
            cuMemcpyHtoD_v2(u64, *const c_void, usize),
            cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
        );
        ok!(cuInit(0));
        let mut ctx = null_mut();
        ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
        let mut stream = null_mut();
        ok!(cuStreamCreate(&mut stream, 0));
        let src = std::ffi::CString::new(ptx).unwrap();
        let mut module = null_mut();
        ok!(cuModuleLoadData(&mut module, src.as_ptr().cast()));
        let name = std::ffi::CString::new(entry).unwrap();
        let mut kernel = null_mut();
        ok!(cuModuleGetFunction(&mut kernel, module, name.as_ptr()));
        type SetStream = unsafe extern "system" fn(*mut c_void, Handle) -> i32;
        let create: libloading::Symbol<Create> = unsafe { lib.get(b"cublasCreate_v2\0") }.unwrap();
        let set_stream: libloading::Symbol<SetStream> = unsafe { lib.get(b"cublasSetStream_v2\0") }.unwrap();
        let gemm: libloading::Symbol<Batched> = unsafe { lib.get(b"cublasGemmBatchedEx\0") }.unwrap();
        let mut handle = null_mut();
        assert_eq!(unsafe { create(&mut handle) }, 0);
        assert_eq!(unsafe { set_stream(handle, stream) }, 0);
        for (name, ta, m, n, k) in [("kq", 1, 512, 512, 64), ("kqv a", 0, 64, 32, 128), ("kqv b", 0, 64, 32, 512), ("kqv c", 0, 64, 512, 512)] {
            let (ne12, r2) = (14i64, 7i64);
            let (a_bytes, b_bytes, c_bytes) = ((m * k * 2) as i64, (k * n * 2) as i64, (m * n * 4) as i64);
            let (mut da, mut db, mut dc, mut p_src, mut p_dst) = (0u64, 0u64, 0u64, 0u64, 0u64);
            ok!(cuMemAlloc_v2(&mut da, (2 * a_bytes) as usize));
            ok!(cuMemAlloc_v2(&mut db, (ne12 * b_bytes) as usize));
            ok!(cuMemAlloc_v2(&mut dc, (ne12 * c_bytes) as usize));
            ok!(cuMemAlloc_v2(&mut p_src, 2 * ne12 as usize * 8));
            ok!(cuMemAlloc_v2(&mut p_dst, ne12 as usize * 8));
            let zeros = vec![0u8; (ne12 * b_bytes) as usize];
            ok!(cuMemcpyHtoD_v2(db, zeros.as_ptr().cast(), zeros.len()));
            ok!(cuMemcpyHtoD_v2(da, zeros.as_ptr().cast(), (2 * a_bytes) as usize));
            let iters: usize = std::env::var("CFI_GQA_ITERS").ok().and_then(|v| v.parse().ok()).unwrap_or(100);
            let (skip_table, skip_gemm) = (std::env::var_os("CFI_GQA_SKIP_TABLE").is_some(), std::env::var_os("CFI_GQA_SKIP_GEMM").is_some());
            for iter in 0..iters {
                let mut vals: Vec<Vec<u8>> = [da, db, dc, p_src, p_dst].iter().map(|v| v.to_le_bytes().to_vec()).collect();
                for v in [ne12, 1, ne12, a_bytes, 0, b_bytes, 0, c_bytes, 0, r2, 1] {
                    vals.push(v.to_le_bytes().to_vec());
                }
                let mut params: Vec<*mut c_void> = vals.iter_mut().map(|v| v.as_mut_ptr().cast()).collect();
                if !skip_table || iter == 0 {
                    ok!(cuLaunchKernel(kernel, 1, 1, 1, 16, 16, 1, 0, stream, params.as_mut_ptr(), null_mut()));
                }
                if skip_gemm {
                    continue;
                }
                let (alpha, beta) = (1.0f32, 0.0f32);
                let status = unsafe {
                    gemm(handle, ta, 0, m, n, k, from_ref(&alpha), p_src as *const *const c_void, 2, if ta == 1 { k } else { m }, (p_src + ne12 as u64 * 8) as *const *const c_void, 2, k, from_ref(&beta), p_dst as *const *mut c_void, 0, m, ne12 as i32, 68, -1)
                };
                if iter < 12 || iter % 50 == 0 || status != 0 {
                    let (mut free, mut total) = (0usize, 0usize);
                    unsafe { cuMemGetInfo_v2(&mut free, &mut total) };
                    println!("{name} iter {iter}: status {status}, device free {} MB of {} MB", free >> 20, total >> 20);
                }
                assert_eq!(status, 0, "{name} iter {iter}: gemm status");
            }
            let sync = unsafe { cuStreamSynchronize(stream) };
            assert_eq!(sync, 0, "{name}: 100 back-to-back table+gemm launches faulted on stream sync");
            println!("{name}: {iters} back-to-back table+gemm launches ok");
        }
    });
}
