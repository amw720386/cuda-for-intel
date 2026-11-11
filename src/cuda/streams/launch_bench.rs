//! Host cost of `cuLaunchKernel` on the fake Level Zero backend, for a ggml-shaped kernel
//! (3 pointers and 21 scalars, 24 parameters) launched on a non-blocking stream. No GPU: it
//! times the frontend (`null driver`: the three per-launch Level Zero calls are no-op stubs) and
//! the frontend plus the fake driver's bookkeeping (`fake driver`), and the null driver again with
//! the `CUkernel` of a `cuLibraryLoadData` library as the launch handle (`null driver, CUkernel`:
//! the stock CUDA Runtime launches this way). Run in release:
//!   cargo test --release --lib -j 2 launch_host_benchmark -- --ignored --nocapture
//! `CFI_LAUNCH_BENCH=null` (or `fake`, `kernel`) runs one mode only, e.g. to count instructions per launch
//! under `valgrind --tool=callgrind --toggle-collect=cuLaunchKernel`.
use super::fake_tests::{Fixture, context, stream};
use super::*;
use crate::cuda::*;
use crate::level_zero::fake;
use std::ffi::CString;
use std::ptr::null_mut;
use std::time::Instant;

const POINTERS: usize = 3;
const SCALARS: usize = 21;
const LAUNCHES: usize = 20_000;
const REPS: usize = 15;
/// A ggml-length mangled name: name hashing and copies scale with it.
const NAME: &str = "_Z13mul_mat_vec_qIL9ggml_type12ELi1ELb0EEvPKvS2_PKiPfjjjjjjjjjjjjjjjjjjjjj";

/// `NAME(.u64 p0, .u64 p1, .u64 p2, .u32 s1 .. s21)`: `p0[tid] = s21`, ggml's parameter shape.
fn kernel() -> String {
    let mut s = format!(".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry {NAME}(");
    let params: Vec<String> = (0..POINTERS)
        .map(|i| format!(".param .u64 k_p{i}"))
        .chain((1..=SCALARS).map(|i| format!(".param .u32 k_s{i}")))
        .collect();
    s += &params.join(", ");
    s += ")\n{\n.reg .b32 %r<4>;\n.reg .b64 %rd<6>;\nld.param.u64 %rd1, [k_p0];\ncvta.to.global.u64 %rd2, %rd1;\n";
    s += &format!("ld.param.u32 %r1, [k_s{SCALARS}];\n");
    s += "mov.u32 %r2, %tid.x;\nmul.wide.u32 %rd3, %r2, 4;\nadd.s64 %rd4, %rd2, %rd3;\nst.global.u32 [%rd4], %r1;\nret;\n}\n";
    s
}

fn set_null_driver(on: bool) {
    let mut guard = RUNTIME.lock().unwrap();
    fake::set_null_launch(&mut guard.as_mut().unwrap().runtime.api, on);
}

#[test]
#[ignore = "host-path benchmark; run in release with --nocapture"]
fn launch_host_benchmark() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let stream = stream();
    let source = CString::new(kernel()).unwrap();
    let mut module = null_mut();
    assert_eq!(unsafe { cuModuleLoadData(&mut module, source.as_ptr().cast()) }, Success);
    let mut function = null_mut();
    let name = CString::new(NAME).unwrap();
    assert_eq!(unsafe { cuModuleGetFunction(&mut function, module, name.as_ptr()) }, Success);
    let mut library = null_mut();
    assert_eq!(
        unsafe { library::cuLibraryLoadData(&mut library, source.as_ptr().cast(), std::ptr::null(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null(), 0) },
        Success
    );
    let mut library_kernel = null_mut();
    assert_eq!(unsafe { library::cuLibraryGetKernel(&mut library_kernel, library, name.as_ptr()) }, Success);
    let mut pointers = [0u64; POINTERS];
    for pointer in &mut pointers {
        assert_eq!(unsafe { cuMemAlloc_v2(pointer, 256) }, Success);
    }
    let mut scalars: Vec<u32> = (1..=SCALARS as u32).collect();
    let mut params: Vec<Opaque> = pointers.iter_mut().map(|p| (p as *mut u64).cast()).collect();
    params.extend(scalars.iter_mut().map(|s| (s as *mut u32).cast::<std::ffi::c_void>()));
    let launch = |handle: Opaque| {
        let result = unsafe {
            cuLaunchKernel(handle, 4, 1, 1, 32, 1, 1, 0, stream, params.as_ptr().cast_mut(), null_mut())
        };
        assert_eq!(result, Success);
    };
    println!("== cuLaunchKernel host cost, {} params, {LAUNCHES} launches x {REPS} reps", POINTERS + SCALARS);
    let only = std::env::var("CFI_LAUNCH_BENCH").ok();
    let modes = [
        ("null driver", "null", true, function),
        ("null driver, CUkernel", "kernel", true, library_kernel),
        ("fake driver", "fake", false, function),
    ];
    for (label, mode, null, handle) in modes {
        if only.as_deref().is_some_and(|only| only != mode) {
            continue;
        }
        set_null_driver(null);
        for _ in 0..2_000 {
            launch(handle);
        }
        assert_eq!(cuStreamSynchronize(stream), Success);
        let mut samples = Vec::with_capacity(REPS);
        for _ in 0..REPS {
            let started = Instant::now();
            for _ in 0..LAUNCHES {
                launch(handle);
            }
            samples.push(started.elapsed().as_nanos() as f64 / LAUNCHES as f64);
            assert_eq!(cuStreamSynchronize(stream), Success);
        }
        samples.sort_by(f64::total_cmp);
        println!(
            "{label}: median {:.1} ns/launch (min {:.1}, max {:.1})",
            samples[REPS / 2],
            samples[0],
            samples[REPS - 1]
        );
    }
    set_null_driver(false);
    for pointer in pointers {
        assert_eq!(cuMemFree_v2(pointer), Success);
    }
    assert_eq!(cuModuleUnload(module), Success);
    assert_eq!(library::cuLibraryUnload(library), Success);
    assert_eq!(cuStreamDestroy_v2(stream), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}
