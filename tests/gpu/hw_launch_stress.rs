//! Launch-queue stress: many unsynchronised launches on one stream must not leak
//! host/device memory or hit Level Zero resource limits.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;
use std::time::Instant;

const TRIVIAL: &str = ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry triv(.param .u64 triv_p0, .param .u32 triv_p1)\n{\n.reg .b32 %r<4>;\n.reg .b64 %rd<6>;\nld.param.u64 %rd1, [triv_p0];\ncvta.to.global.u64 %rd4, %rd1;\nld.param.u32 %r1, [triv_p1];\nmov.u32 %r2, %tid.x;\nmul.wide.u32 %rd2, %r2, 4;\nadd.s64 %rd3, %rd4, %rd2;\nst.global.u32 [%rd3], %r1;\nret;\n}\n";
const BIG: &str = ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry big(.param .u64 big_p0, .param .align 8 .b8 big_p1[256])\n{\n.reg .b32 %r<4>;\n.reg .b64 %rd<6>;\nld.param.u64 %rd1, [big_p0];\ncvta.to.global.u64 %rd4, %rd1;\nld.param.u32 %r1, [big_p1+252];\nmov.u32 %r2, %tid.x;\nmul.wide.u32 %rd2, %r2, 4;\nadd.s64 %rd3, %rd4, %rd2;\nst.global.u32 [%rd3], %r1;\nret;\n}\n";

fn working_set_kb() -> u64 {
    // K32GetProcessMemoryInfo via kernel32 (psapi forwarders).
    #[repr(C)]
    struct Counters {
        cb: u32,
        page_faults: u32,
        peak_ws: usize,
        ws: usize,
        rest: [usize; 6],
    }
    unsafe {
        let lib = libloading::Library::new("kernel32.dll").unwrap();
        let get_process: libloading::Symbol<unsafe extern "system" fn() -> *mut c_void> = lib.get(b"GetCurrentProcess\0").unwrap();
        let get_info: libloading::Symbol<unsafe extern "system" fn(*mut c_void, *mut Counters, u32) -> i32> = lib.get(b"K32GetProcessMemoryInfo\0").unwrap();
        let mut c: Counters = std::mem::zeroed();
        c.cb = size_of::<Counters>() as u32;
        get_info(get_process(), &mut c, c.cb);
        (c.ws / 1024) as u64
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_launch_stress_sync_intervals_do_not_leak_or_exhaust() {
    let total: usize = std::env::var("CFI_STRESS_LAUNCHES").ok().and_then(|v| v.parse().ok()).unwrap_or(100_000);
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuStreamCreate(*mut Handle, u32),
        cuStreamSynchronize(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemsetD32Async(u64, u32, usize, Handle),
        cuMemcpyDtoDAsync_v2(u64, u64, usize, Handle),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 0));
    let load = |src: &str, name: &str| -> Handle {
        let s = std::ffi::CString::new(src).unwrap();
        let mut m = null_mut();
        ok!(cuModuleLoadData(&mut m, s.as_ptr().cast()));
        let n = std::ffi::CString::new(name).unwrap();
        let mut f = null_mut();
        ok!(cuModuleGetFunction(&mut f, m, n.as_ptr()));
        f
    };
    let (triv, big) = (load(TRIVIAL, "triv"), load(BIG, "big"));
    let (mut a, mut b) = (0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut a, 4096));
    ok!(cuMemAlloc_v2(&mut b, 4096));
    let mut report = Vec::new();
    for (mode, interval) in [("trivial", 1usize), ("trivial", 16), ("trivial", 128), ("trivial", 1024), ("big-args", 128), ("mixed", 128), ("trivial", usize::MAX)] {
        let before = working_set_kb();
        let start = Instant::now();
        let mut big_args = vec![0u8; 256];
        let mut failure = None;
        for i in 0..total {
            let mut value = i as u32;
            let mut pa = a;
            let mut params: [*mut c_void; 2] = [(&mut pa as *mut u64).cast(), (&mut value as *mut u32).cast()];
            let status = match mode {
                "big-args" => {
                    big_args[252..256].copy_from_slice(&value.to_le_bytes());
                    let mut bp: [*mut c_void; 2] = [(&mut pa as *mut u64).cast(), big_args.as_mut_ptr().cast()];
                    unsafe { cuLaunchKernel(big, 1, 1, 1, 64, 1, 1, 0, stream, bp.as_mut_ptr(), null_mut()) }
                }
                "mixed" => {
                    let s1 = unsafe { cuMemsetD32Async(b, value, 256, stream) };
                    let s2 = unsafe { cuMemcpyDtoDAsync_v2(a, b, 512, stream) };
                    let s3 = unsafe { cuLaunchKernel(triv, 1, 1, 1, 64, 1, 1, 0, stream, params.as_mut_ptr(), null_mut()) };
                    s1 | s2 | s3
                }
                _ => unsafe { cuLaunchKernel(triv, 1, 1, 1, 64, 1, 1, 0, stream, params.as_mut_ptr(), null_mut()) },
            };
            if status != 0 {
                failure = Some((i, status));
                break;
            }
            if interval != usize::MAX && (i + 1) % interval == 0 {
                let s = unsafe { cuStreamSynchronize(stream) };
                if s != 0 {
                    failure = Some((i, s));
                    break;
                }
            }
        }
        let sync = unsafe { cuStreamSynchronize(stream) };
        let after = working_set_kb();
        let line = format!(
            "{mode} sync every {} : {} launches in {:.2}s, working set {before} -> {after} KB, failure {failure:?}, final sync {sync}",
            if interval == usize::MAX { "never".to_string() } else { interval.to_string() },
            total,
            start.elapsed().as_secs_f64()
        );
        println!("{line}");
        report.push((line, failure.is_some() || sync != 0));
        if failure.is_some() {
            break;
        }
    }
    assert!(report.iter().all(|(_, bad)| !bad), "{report:#?}");
}

const VADD: &str = ".version 8.0\n.target sm_80\n.address_size 64\n.visible .entry vadd(.param .u64 vadd_p0, .param .u64 vadd_p1, .param .u64 vadd_p2, .param .u64 vadd_p3)\n{\n.reg .pred %p<2>;\n.reg .b32 %r<8>;\n.reg .b64 %rd<20>;\n.reg .f32 %f<4>;\nld.param.u64 %rd1, [vadd_p0];\nld.param.u64 %rd2, [vadd_p1];\nld.param.u64 %rd3, [vadd_p2];\nld.param.u64 %rd4, [vadd_p3];\ncvta.to.global.u64 %rd5, %rd1;\ncvta.to.global.u64 %rd6, %rd2;\ncvta.to.global.u64 %rd7, %rd3;\nmov.u32 %r1, %ctaid.x;\nmov.u32 %r2, %ntid.x;\nmov.u32 %r3, %tid.x;\nmad.lo.s32 %r4, %r1, %r2, %r3;\ncvt.u64.u32 %rd8, %r4;\nmov.u32 %r5, %nctaid.x;\nmul.lo.s32 %r6, %r5, %r2;\ncvt.u64.u32 %rd9, %r6;\n$loop:\nsetp.ge.u64 %p1, %rd8, %rd4;\n@%p1 bra $done;\nshl.b64 %rd10, %rd8, 2;\nadd.s64 %rd11, %rd5, %rd10;\nadd.s64 %rd12, %rd6, %rd10;\nadd.s64 %rd13, %rd7, %rd10;\nld.global.f32 %f1, [%rd11];\nld.global.f32 %f2, [%rd12];\nadd.f32 %f3, %f1, %f2;\nst.global.f32 [%rd13], %f3;\nadd.s64 %rd8, %rd8, %rd9;\nbra $loop;\n$done:\nret;\n}\n";

fn system_available_mb() -> u64 {
    #[repr(C)]
    struct MemoryStatus {
        length: u32,
        load: u32,
        fields: [u64; 7],
    }
    unsafe extern "system" {
        fn GlobalMemoryStatusEx(status: *mut MemoryStatus) -> i32;
    }
    let mut status = MemoryStatus { length: size_of::<MemoryStatus>() as u32, load: 0, fields: [0; 7] };
    unsafe { GlobalMemoryStatusEx(&mut status) };
    status.fields[1] >> 20 // ullAvailPhys
}

/// Unsynchronised 14336-group dispatches over large buffers. Reports system RAM at the
/// first failure so integrated-GPU memory pressure can be told apart from a runtime limit.
#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU; hold the GPU lock"]
#[allow(non_snake_case)]
fn hw_big_dispatch_loop_unsynchronised() {
    let launches: usize = std::env::var("CFI_BIG_LAUNCHES").ok().and_then(|v| v.parse().ok()).unwrap_or(1000);
    let bytes: usize = std::env::var("CFI_BIG_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(1024) << 20;
    let min_free_mb: u64 = std::env::var("CFI_MIN_AVAILABLE_MB").ok().and_then(|v| v.parse().ok()).unwrap_or(20 * 1024);
    let before = system_available_mb();
    println!("system available RAM before: {before} MB");
    if before < min_free_mb {
        println!("SKIP: available RAM {before} MB below required {min_free_mb} MB");
        return;
    }
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuStreamCreate(*mut Handle, u32),
        cuStreamSynchronize(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemGetInfo_v2(*mut usize, *mut usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 0));
    let src = std::ffi::CString::new(VADD).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, src.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, c"vadd".as_ptr()));
    let (mut a, mut b, mut c) = (0u64, 0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut a, bytes));
    ok!(cuMemAlloc_v2(&mut b, bytes));
    ok!(cuMemAlloc_v2(&mut c, bytes));
    let (mut free, mut total) = (0usize, 0usize);
    ok!(cuMemGetInfo_v2(&mut free, &mut total));
    println!("after alloc: cuMemGetInfo free {} MB / {} MB, system available {} MB", free >> 20, total >> 20, system_available_mb());
    let mut n = (bytes / 4) as u64;
    let mut failure = None;
    for i in 0..launches {
        let (mut pa, mut pb, mut pc) = (a, b, c);
        let mut params: [*mut c_void; 4] = [(&mut pa as *mut u64).cast(), (&mut pb as *mut u64).cast(), (&mut pc as *mut u64).cast(), (&mut n as *mut u64).cast()];
        let status = unsafe { cuLaunchKernel(kernel, 14336, 1, 1, 256, 1, 1, 0, stream, params.as_mut_ptr(), null_mut()) };
        if status != 0 {
            println!("FAILURE launch {i}: status {status}; system available {} MB (was {before} MB)", system_available_mb());
            failure = Some(i);
            break;
        }
        if i % 100 == 0 {
            println!("launch {i}: system available {} MB", system_available_mb());
        }
    }
    let sync = unsafe { cuStreamSynchronize(stream) };
    println!("final sync {sync}, system available {} MB", system_available_mb());
    assert!(failure.is_none() && sync == 0, "failure {failure:?} sync {sync}");
}
