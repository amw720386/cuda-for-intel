//! Kernel substitution through the driver on the fake backend: selection, lazy native module
//! build, launch routing and fallback.
use super::fake_tests::*;
use super::*;
use crate::cuda::*;
use crate::level_zero::fake::{self, Kind};
use std::ptr::null_mut;

const QUANTIZE: &std::ffi::CStr = c".version 4.1 .target sm_52 .address_size 64 \
.entry _Z13quantize_q8_1PKfPvxxxxxj5uint3(.param .u64 .ptr .global x, .param .u64 .ptr .global y, \
.param .u64 a, .param .u64 b, .param .u64 c, .param .u64 d, .param .u64 e, .param .u32 f, \
.param .align 4 .b8 g[12]) { ret; }";

/// Same name but a different parameter list: must never be substituted.
const QUANTIZE_WRONG_ABI: &std::ffi::CStr = c".version 4.1 .target sm_52 .address_size 64 \
.entry _Z13quantize_q8_1PKfPvxxxxxj5uint3(.param .u64 .ptr .global x, .param .u64 .ptr .global y, \
.param .u64 a, .param .u32 f) { ret; }";

fn load(image: &std::ffi::CStr) -> Opaque {
    let mut module = null_mut();
    assert_eq!(unsafe { cuModuleLoadData(&mut module, image.as_ptr().cast()) }, Success);
    let mut function = null_mut();
    assert_eq!(
        unsafe { cuModuleGetFunction(&mut function, module, c"_Z13quantize_q8_1PKfPvxxxxxj5uint3".as_ptr()) },
        Success
    );
    function
}

fn launch_quantize(function: Opaque, block: u32, a: u64, b: u64, full_abi: bool) -> CuResult {
    let (mut a, mut b) = (a, b);
    let mut u = [0u64; 5];
    let mut f = 0u32;
    let mut g = [0u32; 3];
    let mut params: Vec<*mut std::ffi::c_void> = vec![(&raw mut a).cast(), (&raw mut b).cast()];
    if full_abi {
        for v in u.iter_mut() {
            params.push(std::ptr::from_mut(v).cast());
        }
        params.push((&raw mut f).cast());
        params.push(g.as_mut_ptr().cast());
    } else {
        params.push(std::ptr::from_mut(&mut u[0]).cast());
        params.push((&raw mut f).cast());
    }
    unsafe { cuLaunchKernel(function, 1, 1, 1, block, 1, 1, 0, null_mut(), params.as_mut_ptr().cast(), null_mut()) }
}

fn native_state() -> (usize, usize) {
    let live = fake::snapshot().live;
    let modules = live.get(&Kind::Module).copied().unwrap_or(0);
    let natives = state(|s| Ok(s.contexts.values().map(|c| c.subst_len()).sum::<usize>())).unwrap();
    (modules, natives)
}

#[test]
fn substitution_is_off_by_default_and_on_when_selected() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let (a, _) = allocations();
    let (b, _) = allocations();
    // SAFETY: tests that touch the fake runtime are serialised by the fixture.
    unsafe { std::env::remove_var("CUDA_FOR_INTEL_SUBST") };
    let plain = load(QUANTIZE);
    assert_eq!(launch_quantize(plain, 256, a, b, true), Success);
    let (modules_plain, natives_plain) = native_state();
    assert_eq!(natives_plain, 0);

    unsafe { std::env::set_var("CUDA_FOR_INTEL_SUBST", "quantize_q8_1") };
    let native = load(QUANTIZE);
    assert_eq!(launch_quantize(native, 256, a, b, true), Success);
    let (modules_native, natives_native) = native_state();
    assert_eq!(natives_native, 1, "native kernel built lazily on first launch");
    // one translated module for `native` + one native module
    assert_eq!(modules_native, modules_plain + 2);
    // second launch reuses the native module
    assert_eq!(launch_quantize(native, 256, a, b, true), Success);
    assert_eq!(native_state(), (modules_native, 1));
    // an unexpected block shape keeps the translated kernel (no new module needed, still succeeds)
    assert_eq!(launch_quantize(native, 128, a, b, true), Success);

    // layout mismatch: recognised name, different parameters -> translated only
    let before = native_state();
    let wrong = load(QUANTIZE_WRONG_ABI);
    assert_eq!(launch_quantize(wrong, 256, a, b, false), Success);
    assert_eq!(native_state().1, before.1);
    unsafe { std::env::remove_var("CUDA_FOR_INTEL_SUBST") };
    // context teardown destroys the native modules too
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

/// Host-side launch overhead of the substitution hook on the fake backend (ignored: timing only).
/// Compares per-launch cost with substitution off, on-but-not-applicable (plain kernel) and
/// on-but-falling-back (recognised kernel, unexpected block shape).
#[test]
#[ignore = "timing only"]
fn launch_overhead_with_and_without_substitution() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let (a, _) = allocations();
    let (b, _) = allocations();
    let n = 20_000;
    let mut time = |label: &str, f: Opaque, block: u32, full: bool| {
        let start = std::time::Instant::now();
        for _ in 0..n {
            assert_eq!(launch_quantize(f, block, a, b, full), Success);
        }
        assert_eq!(cuCtxSynchronize(), Success);
        println!("{label}: {:.0} ns/launch", start.elapsed().as_nanos() as f64 / n as f64);
    };
    unsafe { std::env::remove_var("CUDA_FOR_INTEL_SUBST") };
    let off = load(QUANTIZE);
    time("subst none, quantize stub", off, 256, true);
    unsafe { std::env::set_var("CUDA_FOR_INTEL_SUBST", "all") };
    let on = load(QUANTIZE);
    time("subst all, recognised kernel, native path", on, 256, true);
    time("subst all, recognised kernel, geometry fallback", on, 128, true);
    let plain = load(QUANTIZE_WRONG_ABI);
    time("subst all, layout mismatch (plain PTX)", plain, 256, false);
    unsafe { std::env::remove_var("CUDA_FOR_INTEL_SUBST") };
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
}
