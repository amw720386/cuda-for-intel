//! Recorded graph replay (`CUDA_FOR_INTEL_GRAPH_RECORD`) on a real GPU: a ggml-decode-like
//! chain of small dependent kernels, issued per "token" three ways:
//! * `direct`: one `cuLaunchKernel` per kernel (what ggml does below cc 8.0 today);
//! * `graph re-issue`: `cuGraphLaunch` with recording off (every node appended separately);
//! * `graph recorded`: `cuGraphLaunch` with recording on (one regular list per token).
//!
//! Per mode it reports, per token: `token` = issue + `cuStreamSynchronize` (the decode pattern,
//! GPU completion included), `host` = time to issue only, and `pipelined` = back-to-back
//! tokens with one wait at the end (GPU throughput when the host is never the bottleneck).
//! The difference between `token` and the GPU time of the kernels alone is dispatch gaps.
//!
//! `CFI_GRAPH_KERNELS` (default 700, even), `CFI_GRAPH_ELEMS` (default 4096 floats per kernel,
//! 16 work-groups of 256), `CFI_GRAPH_TOKENS` (default 100) timed tokens per mode and pass.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};
use std::time::Instant;

const RELAXED: i32 = 2;

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Direct,
    Reissue,
    Recorded,
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn recorded_graph_replay_matches_and_times_against_reissue() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuMemsetD8Async(u64, u8, usize, Handle),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
        cuStreamSynchronize(Handle),
        cuStreamBeginCapture_v2(Handle, i32),
        cuStreamEndCapture(Handle, *mut Handle),
        cuGraphInstantiateWithFlags(*mut Handle, Handle, u64),
        cuGraphLaunch(Handle, Handle),
        cuGraphDestroy(Handle),
        cuGraphExecDestroy(Handle),
    );
    let kernels = env_usize("CFI_GRAPH_KERNELS", 700).max(2) & !1;
    let elems = env_usize("CFI_GRAPH_ELEMS", 4096).max(256) / 256 * 256;
    let tokens = env_usize("CFI_GRAPH_TOKENS", 100).max(1);

    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let ptx = std::ffi::CString::new(include_str!("../../examples/vector_add.ptx")).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut add = null_mut();
    ok!(cuModuleGetFunction(&mut add, module, c"vector_add".as_ptr()));
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 1));

    let bytes = elems * 4;
    let (mut x, mut y, mut ones, mut scratch) = (0u64, 0u64, 0u64, 0u64);
    for buffer in [&mut x, &mut y, &mut ones, &mut scratch] {
        ok!(cuMemAlloc_v2(buffer, bytes));
    }
    let one = vec![1f32; elems];
    ok!(cuMemcpyHtoD_v2(ones, one.as_ptr().cast(), bytes));
    let n = elems as u32;
    let grid = (elems / 256) as u32;

    // One token: a memset of an unrelated buffer (ggml graphs have a few non-kernel nodes),
    // then x -> y -> x ... where each kernel adds one, so a token leaves x == x + kernels.
    let token = |stream: Handle| {
        ok!(cuMemsetD8Async(scratch, 0, bytes, stream));
        for index in 0..kernels {
            let (mut src, mut dst) = if index % 2 == 0 { (x, y) } else { (y, x) };
            let (mut ones, mut n) = (ones, n);
            let mut params = [
                from_mut(&mut src).cast::<c_void>(),
                from_mut(&mut ones).cast::<c_void>(),
                from_mut(&mut dst).cast::<c_void>(),
                from_mut(&mut n).cast::<c_void>(),
            ];
            assert_eq!(
                unsafe { cuLaunchKernel(add, grid, 1, 1, 256, 1, 1, 0, stream, params.as_mut_ptr(), null_mut()) },
                0
            );
        }
    };
    ok!(cuStreamBeginCapture_v2(stream, RELAXED));
    token(stream);
    let mut graph = null_mut();
    ok!(cuStreamEndCapture(stream, &mut graph));
    let mut exec = null_mut();
    ok!(cuGraphInstantiateWithFlags(&mut exec, graph, 0));
    ok!(cuGraphDestroy(graph));

    let zero = vec![0f32; elems];
    let mut result = vec![0f32; elems];
    let run = |mode: Mode| match mode {
        Mode::Direct => token(stream),
        Mode::Reissue | Mode::Recorded => ok!(cuGraphLaunch(exec, stream)),
    };
    println!("chain: {kernels} kernels x {grid} work-groups of 256 + 1 memset, {tokens} tokens per measurement");
    for pass in 0..2 {
        for mode in [Mode::Direct, Mode::Reissue, Mode::Recorded] {
            let (name, record) = match mode {
                Mode::Direct => ("direct launches", "0"),
                Mode::Reissue => ("graph re-issue ", "0"),
                Mode::Recorded => ("graph recorded ", "1"),
            };
            // SAFETY: run with --test-threads=1; the DLL reads the variable at each graph launch.
            unsafe { std::env::set_var("CUDA_FOR_INTEL_GRAPH_RECORD", record) };
            ok!(cuMemcpyHtoD_v2(x, zero.as_ptr().cast(), bytes));
            // Warm-up (records the list in recorded mode) and a correctness check.
            for _ in 0..3 {
                run(mode);
            }
            ok!(cuStreamSynchronize(stream));
            ok!(cuMemcpyDtoH_v2(result.as_mut_ptr().cast(), x, bytes));
            let expected = (3 * kernels) as f32;
            assert!(result.iter().all(|&v| v == expected), "{name}: {} != {expected}", result[0]);

            let start = Instant::now();
            for _ in 0..tokens {
                run(mode);
                ok!(cuStreamSynchronize(stream));
            }
            let per_token = start.elapsed().as_secs_f64() * 1e6 / tokens as f64;
            let start = Instant::now();
            for _ in 0..tokens {
                run(mode);
            }
            let host = start.elapsed().as_secs_f64() * 1e6 / tokens as f64;
            ok!(cuStreamSynchronize(stream));
            let pipelined = start.elapsed().as_secs_f64() * 1e6 / tokens as f64;
            ok!(cuMemcpyDtoH_v2(result.as_mut_ptr().cast(), x, bytes));
            let expected = ((3 + 2 * tokens) * kernels) as f32;
            assert!(result.iter().all(|&v| v == expected), "{name}: {} != {expected}", result[0]);
            println!(
                "pass {pass} {name}: token {per_token:8.1} us ({:.2} us/kernel), host {host:8.1} us, pipelined {pipelined:8.1} us ({:.2} us/kernel)",
                per_token / kernels as f64,
                pipelined / kernels as f64
            );
        }
    }
    unsafe { std::env::remove_var("CUDA_FOR_INTEL_GRAPH_RECORD") };
    ok!(cuGraphExecDestroy(exec));
    ok!(cuStreamDestroy_v2(stream));
    ok!(cuCtxDestroy_v2(context));
}
