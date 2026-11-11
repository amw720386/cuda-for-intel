//! Host-side cost of `cuLaunchKernel` against the number of kernel parameters, on a non-blocking
//! stream as ggml uses. Decode issues about 725 launches per token with about 22 parameters each
//! (Qwen2.5-0.5B on stock Ollama), so microseconds per launch here are milliseconds per token there.
//! One GPU slot:
//!   cargo test --release --test gpu hw_launch_host -- --ignored --nocapture --test-threads=1
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{CString, c_void};
use std::ptr::null_mut;
use std::time::Instant;

const HEADER: &str = ".version 8.0\n.target sm_80\n.address_size 64\n";
const REPS: usize = 7;
const LAUNCHES: usize = 4000;

/// `.entry k(.param .u64 dst, .param .u32 p1, ..., .param .u32 p{n-1})`: `dst[tid] = p{n-1}`.
fn kernel(n: usize) -> String {
    let mut s = format!("{HEADER}.visible .entry k(.param .u64 k_dst");
    for i in 1..n {
        s += &format!(", .param .u32 k_p{i}");
    }
    s += ")\n{\n.reg .b32 %r<4>;\n.reg .b64 %rd<6>;\nld.param.u64 %rd1, [k_dst];\ncvta.to.global.u64 %rd2, %rd1;\n";
    s += &format!("ld.param.u32 %r1, [k_p{}];\n", n - 1);
    s += "mov.u32 %r2, %tid.x;\nmul.wide.u32 %rd3, %r2, 4;\nadd.s64 %rd4, %rd2, %rd3;\nst.global.u32 [%rd4], %r1;\nret;\n}\n";
    s
}

fn percentile(sorted: &[f64], p: usize) -> f64 {
    sorted[(p * sorted.len()).div_ceil(100).max(1) - 1]
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_launch_host_overhead_by_param_count() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamSynchronize(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const std::ffi::c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 1)); // CU_STREAM_NON_BLOCKING, like ggml's stream
    let mut dst = 0u64;
    ok!(cuMemAlloc_v2(&mut dst, 32 * 4));
    println!("== cuLaunchKernel host cost, grid 1 x 32, {LAUNCHES} async launches per rep, {REPS} reps");
    for n in [2usize, 8, 16, 24, 32] {
        let source = CString::new(kernel(n)).unwrap();
        let mut module = null_mut();
        ok!(cuModuleLoadData(&mut module, source.as_ptr().cast()));
        let mut function = null_mut();
        ok!(cuModuleGetFunction(&mut function, module, c"k".as_ptr()));
        let mut pointer = dst;
        let mut values: Vec<u32> = (1..n as u32).collect();
        let mut params: Vec<*mut c_void> = Vec::with_capacity(n);
        params.push((&raw mut pointer).cast());
        for value in &mut values {
            params.push((value as *mut u32).cast());
        }
        let mut launch = || {
            ok!(cuLaunchKernel(function, 1, 1, 1, 32, 1, 1, 0, stream, params.as_mut_ptr(), null_mut()));
        };
        for _ in 0..500 {
            launch();
        }
        ok!(cuStreamSynchronize(stream));
        let (mut host, mut done) = (Vec::new(), Vec::new());
        for _ in 0..REPS {
            let start = Instant::now();
            for _ in 0..LAUNCHES {
                launch();
            }
            host.push(start.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
            ok!(cuStreamSynchronize(stream));
            done.push(start.elapsed().as_secs_f64() * 1e6 / LAUNCHES as f64);
        }
        let mut out = [0u32; 32];
        ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), dst, 32 * 4));
        assert!(out.iter().all(|&v| v == n as u32 - 1), "params={n}: kernel wrote {out:?}");
        host.sort_by(f64::total_cmp);
        done.sort_by(f64::total_cmp);
        println!(
            "params {n:2}: host p50 {:.2} us/launch (p95 {:.2}) | to completion p50 {:.2} us/launch (p95 {:.2})",
            percentile(&host, 50),
            percentile(&host, 95),
            percentile(&done, 50),
            percentile(&done, 95)
        );
    }
    ok!(cuCtxDestroy_v2(ctx));
}

/// `.entry slm(.param .u64 src, .param .u64 dst)`: 32-thread workgroups that stage 16 floats per
/// thread through a `bytes`-byte `.shared` array (only the first 128 bytes are used), barrier, and
/// write one per-thread sum. The array size alone sets how many workgroups fit per Xe-core.
fn slm_kernel(bytes: usize) -> String {
    let mut s = format!("{HEADER}.visible .entry slm(.param .u64 slm_src, .param .u64 slm_dst)\n{{\n");
    s += &format!(".shared .align 16 .b8 buf[{bytes}];\n");
    s += ".reg .pred %p<2>;\n.reg .b32 %r<8>;\n.reg .f32 %f<8>;\n.reg .b64 %rd<12>;\n";
    s += "ld.param.u64 %rd1, [slm_src];\ncvta.to.global.u64 %rd1, %rd1;\nld.param.u64 %rd2, [slm_dst];\ncvta.to.global.u64 %rd2, %rd2;\n";
    s += "mov.u32 %r1, %tid.x;\nmov.u32 %r2, %ctaid.x;\nshl.b32 %r3, %r2, 5;\nadd.s32 %r3, %r3, %r1;\n";
    s += "mov.f32 %f1, 0f00000000;\nmov.u32 %r4, 0;\nLOOP:\n";
    s += "mad.lo.s32 %r5, %r4, 1048576, %r3;\nand.b32 %r5, %r5, 16777215;\nmul.wide.u32 %rd3, %r5, 4;\nadd.s64 %rd4, %rd1, %rd3;\nld.global.f32 %f2, [%rd4];\nadd.f32 %f1, %f1, %f2;\n";
    s += "add.s32 %r4, %r4, 1;\nsetp.lt.u32 %p1, %r4, 16;\n@%p1 bra LOOP;\n";
    s += "mov.u32 %r6, buf;\nshl.b32 %r7, %r1, 2;\nadd.s32 %r6, %r6, %r7;\nst.shared.f32 [%r6], %f1;\nbar.sync 0;\n";
    s += "xor.b32 %r7, %r7, 4;\nmov.u32 %r6, buf;\nadd.s32 %r6, %r6, %r7;\nld.shared.f32 %f3, [%r6];\nadd.f32 %f1, %f1, %f3;\n";
    s += "mul.wide.u32 %rd5, %r3, 4;\nadd.s64 %rd6, %rd2, %rd5;\nst.global.f32 [%rd6], %f1;\nret;\n}\n";
    s
}

/// GPU time of a small 32-thread kernel against the size of its workgroup shared array.
#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_slm_reservation_occupancy() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamSynchronize(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const std::ffi::c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 1));
    const N: usize = 1 << 24;
    let host: Vec<f32> = (0..N).map(|i| ((i * 2654435761usize) % 1000) as f32 * 1e-3).collect();
    let (mut src, mut dst) = (0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut src, N * 4));
    ok!(cuMemcpyHtoD_v2(src, host.as_ptr().cast(), N * 4));
    ok!(cuMemAlloc_v2(&mut dst, N * 4));
    println!("== 32-thread workgroups, 16 strided loads per thread, one shared array of the given size");
    let mut reference: Option<Vec<f32>> = None;
    for groups in [448u32, 4096] {
        for bytes in [65536usize, 49152, 16384, 4096] {
            let source = CString::new(slm_kernel(bytes)).unwrap();
            let mut module = null_mut();
            ok!(cuModuleLoadData(&mut module, source.as_ptr().cast()));
            let mut function = null_mut();
            ok!(cuModuleGetFunction(&mut function, module, c"slm".as_ptr()));
            let (mut s, mut d) = (src, dst);
            let mut params: [*mut c_void; 2] = [(&raw mut s).cast(), (&raw mut d).cast()];
            let mut launch = || ok!(cuLaunchKernel(function, groups, 1, 1, 32, 1, 1, 0, stream, params.as_mut_ptr(), null_mut()));
            for _ in 0..20 {
                launch();
            }
            ok!(cuStreamSynchronize(stream));
            let mut samples = Vec::new();
            for _ in 0..REPS {
                let start = Instant::now();
                for _ in 0..200 {
                    launch();
                }
                ok!(cuStreamSynchronize(stream));
                samples.push(start.elapsed().as_secs_f64() * 1e6 / 200.0);
            }
            samples.sort_by(f64::total_cmp);
            let count = groups as usize * 32;
            let mut out = vec![0f32; count];
            ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), dst, count * 4));
            if groups == 448 {
                match &reference {
                    None => reference = Some(out),
                    Some(r) => assert_eq!(r, &out, "shared array {bytes} changed the result"),
                }
            }
            println!("groups {groups:4}, shared {bytes:5} B: p50 {:.2} us/launch (p95 {:.2})", percentile(&samples, 50), percentile(&samples, 95));
        }
    }
    ok!(cuCtxDestroy_v2(ctx));
}
