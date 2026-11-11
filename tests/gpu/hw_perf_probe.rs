//! Decode-path ceilings on the device: launch overhead (async throughput and launch+sync round trip)
//! and global-memory read bandwidth for a few load widths and occupancies. Reference numbers for
//! the mmvq work (docs/hw-logs/perf-mmvq-2026-11-09). One GPU slot:
//!   cargo test --release --test gpu hw_perf_probe -- --ignored --nocapture --test-threads=1
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;
use std::time::Instant;

const HEADER: &str = ".version 8.0\n.target sm_80\n.address_size 64\n";

const TRIVIAL: &str = ".visible .entry triv(.param .u64 triv_p0, .param .u32 triv_p1)\n{\n.reg .b32 %r<4>;\n.reg .b64 %rd<6>;\nld.param.u64 %rd1, [triv_p0];\ncvta.to.global.u64 %rd4, %rd1;\nld.param.u32 %r1, [triv_p1];\nmov.u32 %r2, %tid.x;\nmul.wide.u32 %rd2, %r2, 4;\nadd.s64 %rd3, %rd4, %rd2;\nst.global.u32 [%rd3], %r1;\nret;\n}\n";

/// Grid-stride read of `n` 16-byte vectors, `unroll` independent loads per iteration.
fn read_kernel(name: &str, unroll: u32, width: u32) -> String {
    let mut s = String::new();
    s += &format!(".visible .entry {name}(.param .u64 p_src, .param .u64 p_dst, .param .u32 p_n)\n{{\n");
    s += ".reg .pred %p<4>;\n.reg .b32 %r<64>;\n.reg .b64 %rd<32>;\n";
    s += "ld.param.u64 %rd1, [p_src];\nld.param.u64 %rd2, [p_dst];\nld.param.u32 %r1, [p_n];\n";
    s += "mov.u32 %r2, %ctaid.x;\nmov.u32 %r3, %ntid.x;\nmov.u32 %r4, %tid.x;\nmad.lo.s32 %r5, %r2, %r3, %r4;\n";
    s += "mov.u32 %r6, %nctaid.x;\nmul.lo.s32 %r7, %r6, %r3;\n";
    s += &format!("mul.lo.s32 %r8, %r7, {unroll};\n");
    s += "mov.u32 %r10, 0;\nmov.u32 %r9, %r5;\n";
    // n is a multiple of total*unroll in this test: no tail handling
    s += "setp.ge.u32 %p1, %r9, %r1;\n@%p1 bra DONE;\nLOOP:\n";
    for u in 0..unroll {
        s += &format!("mad.lo.s32 %r{}, %r7, {u}, %r9;\n", 20 + u);
        s += &format!("mul.wide.u32 %rd{}, %r{}, {width};\n", 3 + 2 * u, 20 + u);
        s += &format!("add.s64 %rd{}, %rd1, %rd{};\n", 4 + 2 * u, 3 + 2 * u);
        match width {
            16 => s += &format!("ld.global.nc.v4.u32 {{%r{a},%r{b},%r{c},%r{d}}}, [%rd{}];\n", 4 + 2 * u, a = 30 + 4 * u, b = 31 + 4 * u, c = 32 + 4 * u, d = 33 + 4 * u),
            8 => s += &format!("ld.global.nc.v2.u32 {{%r{a},%r{b}}}, [%rd{}];\n", 4 + 2 * u, a = 30 + 4 * u, b = 31 + 4 * u),
            _ => s += &format!("ld.global.nc.u32 %r{a}, [%rd{}];\n", 4 + 2 * u, a = 30 + 4 * u),
        }
    }
    for u in 0..unroll {
        let lanes = width / 4;
        for l in 0..lanes {
            s += &format!("xor.b32 %r10, %r10, %r{};\n", 30 + 4 * u + l);
        }
    }
    s += "add.s32 %r9, %r9, %r8;\nsetp.lt.u32 %p2, %r9, %r1;\n@%p2 bra LOOP;\nDONE:\n";
    s += "mul.wide.u32 %rd30, %r5, 4;\nadd.s64 %rd31, %rd2, %rd30;\nst.global.u32 [%rd31], %r10;\nret;\n}\n";
    s
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_perf_probe_launch_and_bandwidth() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuStreamCreate(*mut Handle, u32),
        cuStreamSynchronize(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemsetD32_v2(u64, u32, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 0));
    let load = |body: &str, name: &str| -> Handle {
        let s = std::ffi::CString::new(format!("{HEADER}{body}")).unwrap();
        let mut m = null_mut();
        ok!(cuModuleLoadData(&mut m, s.as_ptr().cast()));
        let n = std::ffi::CString::new(name).unwrap();
        let mut f = null_mut();
        ok!(cuModuleGetFunction(&mut f, m, n.as_ptr()));
        f
    };

    // ---- launch overhead ----
    let triv = load(TRIVIAL, "triv");
    let mut a = 0u64;
    ok!(cuMemAlloc_v2(&mut a, 1 << 20));
    let launch_triv = |grid: u32, block: u32| {
        let mut pa = a;
        let mut v = 7u32;
        let mut params: [*mut c_void; 2] = [(&mut pa as *mut u64).cast(), (&mut v as *mut u32).cast()];
        ok!(cuLaunchKernel(triv, grid, 1, 1, block, 1, 1, 0, stream, params.as_mut_ptr(), null_mut()));
    };
    for _ in 0..200 {
        launch_triv(1, 32);
    }
    ok!(cuStreamSynchronize(stream));
    println!("== launch overhead (trivial kernel)");
    for (grid, block) in [(1u32, 32u32), (896, 128), (4864, 128)] {
        let n = 5000;
        let t = Instant::now();
        for _ in 0..n {
            launch_triv(grid, block);
        }
        let host = t.elapsed().as_secs_f64();
        ok!(cuStreamSynchronize(stream));
        let total = t.elapsed().as_secs_f64();
        println!(
            "grid {grid:5} x {block:3}: async {:.2} us/launch host, {:.2} us/launch to completion",
            host * 1e6 / n as f64,
            total * 1e6 / n as f64
        );
        let n = 1000;
        let t = Instant::now();
        for _ in 0..n {
            launch_triv(grid, block);
            ok!(cuStreamSynchronize(stream));
        }
        println!("grid {grid:5} x {block:3}: launch+sync round trip {:.2} us", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    }

    // ---- read bandwidth ----
    let bytes: usize = 512 << 20;
    let mut src = 0u64;
    let mut dst = 0u64;
    ok!(cuMemAlloc_v2(&mut src, bytes));
    ok!(cuMemAlloc_v2(&mut dst, 64 << 20));
    // random contents: a constant fill could be served compressed
    let mut host = vec![0u32; bytes / 4];
    let mut x = 0x9e37_79b9_7f4a_7c15u64;
    for v in host.iter_mut() {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        *v = x as u32;
    }
    ok!(cuMemcpyHtoD_v2(src, host.as_ptr().cast(), bytes));
    let _ = cuMemsetD32_v2;
    println!("== read bandwidth ({} MiB buffer)", bytes >> 20);
    println!("width unroll block   grid      GB/s");
    let mut best = 0.0f64;
    for (width, unroll) in [(4u32, 1u32), (4, 4), (8, 4), (16, 1), (16, 2), (16, 4)] {
        let name = format!("rd_w{width}_u{unroll}");
        let f = load(&read_kernel(&name, unroll, width), &name);
        for block in [64u32, 128, 256, 512] {
            for grid in [96u32 * 8, 96 * 32, 96 * 128] {
                let total = u64::from(grid) * u64::from(block) * u64::from(unroll);
                let n_vec = (bytes as u64 / u64::from(width)) / total * total;
                if n_vec == 0 {
                    continue;
                }
                let mut ps = src;
                let mut pd = dst;
                let mut pn = n_vec as u32;
                let mut params: [*mut c_void; 3] = [(&mut ps as *mut u64).cast(), (&mut pd as *mut u64).cast(), (&mut pn as *mut u32).cast()];
                for _ in 0..2 {
                    ok!(cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, stream, params.as_mut_ptr(), null_mut()));
                }
                ok!(cuStreamSynchronize(stream));
                let reps = 10;
                let t = Instant::now();
                for _ in 0..reps {
                    ok!(cuLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, stream, params.as_mut_ptr(), null_mut()));
                }
                ok!(cuStreamSynchronize(stream));
                let gbs = (n_vec * u64::from(width) * reps) as f64 / t.elapsed().as_secs_f64() / 1e9;
                // every word was read: the xor of all thread outputs equals the host xor of the prefix
                let threads = (u64::from(grid) * u64::from(block)) as usize;
                let mut out = vec![0u32; threads];
                ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), dst, threads * 4));
                let got = out.iter().fold(0u32, |a, b| a ^ b);
                let want = host[..(n_vec * u64::from(width) / 4) as usize].iter().fold(0u32, |a, b| a ^ b);
                assert_eq!(got, want, "{name} block {block} grid {grid}: not every word was read");
                best = best.max(gbs);
                println!("{width:5} {unroll:6} {block:5} {grid:6} {gbs:9.1}");
            }
        }
    }
    println!("best read bandwidth {best:.1} GB/s");
    assert!(best > 1.0);
}
