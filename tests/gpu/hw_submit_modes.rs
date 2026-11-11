//! GPU-side cost per launch under each command submission mode.
//!
//! Every mode runs in a fresh context (the submission settings are read when a context is created):
//!   (a) 1000 back-to-back trivial launches (1 block of 32 threads): host us/launch to enqueue, us/launch to completion;
//!   (b) launch + cuStreamSynchronize round trip;
//!   (c) a decode-shaped chain of ~700 launches per "token" (Qwen2.5-0.5B layer pattern: many 1-19 block element-wise
//!       launches, 128/896/4864-block matrix-vector shapes, one 151936-block lm_head), ending like ggml's decode step with
//!       an async 4-byte device-to-host copy and cuStreamSynchronize; ms/token, host us/launch and us/launch to completion.
//!
//! The kernel only lets thread 0 of each block add 1.0 to `p[blockIdx.x]`, so the cost is dispatch, not bandwidth; the
//! chain is verified exactly afterwards (`p[j]` counts the launches whose grid exceeds `j`).
//!
//! Modes come from `CFI_SUBMIT_MODES` (`name:VAR=value,VAR=value;name2:...`; `name:` alone means no variables) and default
//! to every mode the shim knows (see `DEFAULT_MODES`). Variables the DLL does not know leave it on its default path, so an
//! older DLL just repeats the baseline. `batch64` needs a DLL built with `--features experimental-batching`.
//!
//!   cargo build --release --locked -p cuda-for-intel
//!   cargo test --release --locked --test gpu hw_submit_modes -- --ignored --nocapture --test-threads=1
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{CString, c_void};
use std::ptr::null_mut;
use std::time::Instant;

const KERNEL: &str = ".version 8.0\n.target sm_80\n.address_size 64\n\
.visible .entry inc(.param .u64 inc_p, .param .u32 inc_n)\n{\n\
.reg .pred %p<3>;\n.reg .b32 %r<4>;\n.reg .f32 %f<3>;\n.reg .b64 %rd<5>;\n\
ld.param.u64 %rd1, [inc_p];\ncvta.to.global.u64 %rd2, %rd1;\nld.param.u32 %r1, [inc_n];\n\
mov.u32 %r2, %ctaid.x;\nmov.u32 %r3, %tid.x;\n\
setp.ne.u32 %p1, %r3, 0;\n@%p1 bra DONE;\nsetp.ge.u32 %p2, %r2, %r1;\n@%p2 bra DONE;\n\
mul.wide.u32 %rd3, %r2, 4;\nadd.s64 %rd4, %rd2, %rd3;\n\
ld.global.f32 %f1, [%rd4];\nadd.f32 %f2, %f1, 0f3F800000;\nst.global.f32 [%rd4], %f2;\n\
DONE:\nret;\n}\n";

/// Modes measured when `CFI_SUBMIT_MODES` is unset.
const DEFAULT_MODES: &str = "immediate:;\
batch64:CUDA_FOR_INTEL_BATCH_CAP=64,CUDA_FOR_INTEL_BATCH_US=250;\
deferred:CUDA_FOR_INTEL_SUBMIT=deferred;\
deferred16:CUDA_FOR_INTEL_SUBMIT=deferred,CUDA_FOR_INTEL_SUBMIT_CAP=16;\
deferred256:CUDA_FOR_INTEL_SUBMIT=deferred,CUDA_FOR_INTEL_SUBMIT_CAP=256";

/// Every variable any mode may set; cleared before each mode so modes do not leak into each other.
const KNOWN_VARS: &[&str] = &[
    "CUDA_FOR_INTEL_BATCH_CAP",
    "CUDA_FOR_INTEL_BATCH_US",
    "CUDA_FOR_INTEL_BATCH_TIMER",
    "CUDA_FOR_INTEL_SUBMIT",
    "CUDA_FOR_INTEL_SUBMIT_CAP",
];

const LAYERS: usize = 24;
const LM_HEAD: u32 = 151_936;

/// (grid, block) of one decode step: per layer 29 launches, then the output head.
fn decode_chain() -> Vec<(u32, u32)> {
    let small = |g: u32| (g, 256);
    let mmvq = |rows: u32| (rows, 128);
    let mut chain = vec![small(4)]; // get_rows
    for _ in 0..LAYERS {
        chain.extend([
            small(1),    // rms_norm
            small(4),    // mul (norm weight)
            small(4),    // quantize q8_1
            mmvq(896),   // q
            mmvq(128),   // k
            mmvq(128),   // v
            small(4),    // bias q
            small(1),    // bias k
            small(1),    // bias v
            small(7),    // rope q
            small(1),    // rope k
            small(1),    // cpy k -> cache
            small(1),    // cpy v -> cache
            small(1),    // view/cont
            (14, 128),   // flash attention (one block per head)
            small(1),    // attention combine
            small(4),    // quantize q8_1
            mmvq(896),   // o
            small(4),    // add residual
            small(1),    // rms_norm
            small(4),    // mul
            small(4),    // quantize q8_1
            mmvq(4864),  // gate
            mmvq(4864),  // up
            small(19),   // silu
            small(19),   // mul
            small(19),   // quantize q8_1
            mmvq(896),   // down
            small(4),    // add residual
        ]);
    }
    chain.extend([small(1), small(4), small(4), mmvq(LM_HEAD), small(1)]);
    chain
}

fn parse_modes(spec: &str) -> Vec<(String, Vec<(String, String)>)> {
    spec.split(';')
        .map(str::trim)
        .filter(|mode| !mode.is_empty())
        .map(|mode| {
            let (name, vars) = mode.split_once(':').unwrap_or((mode, ""));
            let vars = vars
                .split(',')
                .filter(|pair| !pair.is_empty())
                .map(|pair| {
                    let (key, value) = pair.split_once('=').expect("CFI_SUBMIT_MODES: VAR=value");
                    (key.trim().to_owned(), value.trim().to_owned())
                })
                .collect();
            (name.trim().to_owned(), vars)
        })
        .collect()
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(f64::total_cmp);
    samples[samples.len() / 2]
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_submit_modes() {
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
        cuMemAllocHost_v2(*mut *mut c_void, usize),
        cuMemFreeHost(*mut c_void),
        cuMemsetD8Async(u64, u8, usize, Handle),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuMemcpyDtoHAsync_v2(*mut c_void, u64, usize, Handle),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    ok!(cuInit(0));
    let spec = std::env::var("CFI_SUBMIT_MODES").unwrap_or_else(|_| DEFAULT_MODES.to_owned());
    let tokens: usize = std::env::var("CFI_SUBMIT_TOKENS").ok().and_then(|v| v.parse().ok()).unwrap_or(20);
    let chain = decode_chain();
    let blocks = LM_HEAD as usize;
    println!(
        "== submission modes: (a) 1000 trivial launches x 7, (b) launch+sync x 200, (c) decode chain of {} launches x {tokens} tokens",
        chain.len()
    );
    println!("mode        | (a) host us/l | (a) done us/l | (b) launch+sync us | (c) ms/token | (c) host us/l | (c) done us/l");
    for (name, vars) in parse_modes(&spec) {
        for key in KNOWN_VARS {
            unsafe { std::env::remove_var(key) };
        }
        for (key, value) in &vars {
            unsafe { std::env::set_var(key, value) };
        }
        let mut ctx = null_mut();
        ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
        let mut stream = null_mut();
        ok!(cuStreamCreate(&mut stream, 1)); // CU_STREAM_NON_BLOCKING, like ggml
        let source = CString::new(KERNEL).unwrap();
        let mut module = null_mut();
        ok!(cuModuleLoadData(&mut module, source.as_ptr().cast()));
        let mut function = null_mut();
        ok!(cuModuleGetFunction(&mut function, module, c"inc".as_ptr()));
        let mut buffer = 0u64;
        ok!(cuMemAlloc_v2(&mut buffer, blocks * 4));
        let mut pinned = null_mut();
        ok!(cuMemAllocHost_v2(&mut pinned, 4));
        let launch = |grid: u32, block: u32, n: u32| {
            let mut pointer = buffer;
            let mut count = n;
            let mut params: [*mut c_void; 2] = [(&raw mut pointer).cast(), (&raw mut count).cast()];
            ok!(cuLaunchKernel(function, grid, 1, 1, block, 1, 1, 0, stream, params.as_mut_ptr(), null_mut()));
        };

        // (a) back-to-back trivial launches.
        for _ in 0..200 {
            launch(1, 32, 1);
        }
        ok!(cuStreamSynchronize(stream));
        let (mut host, mut done) = (Vec::new(), Vec::new());
        for _ in 0..7 {
            let start = Instant::now();
            for _ in 0..1000 {
                launch(1, 32, 1);
            }
            host.push(start.elapsed().as_secs_f64() * 1e6 / 1000.0);
            ok!(cuStreamSynchronize(stream));
            done.push(start.elapsed().as_secs_f64() * 1e6 / 1000.0);
        }
        // (b) round trip.
        let mut trip = Vec::new();
        for _ in 0..200 {
            let start = Instant::now();
            launch(1, 32, 1);
            ok!(cuStreamSynchronize(stream));
            trip.push(start.elapsed().as_secs_f64() * 1e6);
        }

        // (c) decode-shaped chain, verified exactly.
        ok!(cuMemsetD8Async(buffer, 0, blocks * 4, stream));
        let (mut token_ms, mut chain_host, mut chain_done) = (Vec::new(), Vec::new(), Vec::new());
        let runs = tokens + 2;
        for run in 0..runs {
            let start = Instant::now();
            for &(grid, block) in &chain {
                launch(grid, block, grid);
            }
            let issued = start.elapsed();
            ok!(cuMemcpyDtoHAsync_v2(pinned, buffer, 4, stream));
            ok!(cuStreamSynchronize(stream));
            let total = start.elapsed();
            assert_eq!(unsafe { pinned.cast::<f32>().read() }, ((run + 1) * chain.len()) as f32, "{name}: chain lost a launch");
            if run >= 2 {
                token_ms.push(total.as_secs_f64() * 1e3);
                chain_host.push(issued.as_secs_f64() * 1e6 / chain.len() as f64);
                chain_done.push(total.as_secs_f64() * 1e6 / chain.len() as f64);
            }
        }
        let mut out = vec![0f32; blocks];
        ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), buffer, blocks * 4));
        for j in [0usize, 1, 3, 4, 13, 18, 19, 127, 128, 895, 896, 4863, 4864, blocks - 1] {
            let per_token = chain.iter().filter(|&&(grid, _)| grid as usize > j).count();
            assert_eq!(out[j], (runs * per_token) as f32, "{name}: p[{j}] after {runs} chains");
        }
        println!(
            "{name:11} | {:13.2} | {:13.2} | {:18.2} | {:12.3} | {:13.2} | {:13.2}",
            median(&mut host),
            median(&mut done),
            median(&mut trip),
            median(&mut token_ms),
            median(&mut chain_host),
            median(&mut chain_done),
        );
        ok!(cuMemFreeHost(pinned));
        ok!(cuCtxDestroy_v2(ctx));
    }
    for key in KNOWN_VARS {
        unsafe { std::env::remove_var(key) };
    }
}
