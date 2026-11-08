//! cuDNN shim on hardware: one dense test covering conv forward / backward data /
//! backward filter (NCHW + NHWC, groups, dilation, stride, padding, flip, f32 + f16),
//! AddTensor, ConvolutionBiasActivationForward, spatial batch norm (training,
//! inference, backward), the v8 backend API conv flow and NOT_SUPPORTED probes,
//! all against the CPU references in `crates/cudnn_shim/src/refmath.rs`.
//! A watchdog aborts the process if a launch hangs.
//!
//! Run (Windows, Intel GPU; cudnn64_9.dll built next to nvcuda.dll):
//!   cargo build --release -p cuda-for-intel -p cudnn_shim
//!   cargo test --release --test gpu hw_cudnn -- --ignored --nocapture --test-threads=1
use super::hw_common::{f16_to_f32, f32_to_f16};
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::c_void;
use std::ptr::null_mut;
use std::time::Instant;

#[path = "../../crates/cudnn_shim/src/refmath.rs"]
mod refmath;
use refmath::{ConvGeom, Kind, out_dim};

type Dll = libloading::Library;

static GEMM_PATH_CASES: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

fn cudnn_dll() -> Dll {
    let exe = std::env::current_exe().unwrap();
    let dll = exe.parent().unwrap().parent().unwrap().join("cudnn64_9.dll");
    unsafe { Dll::new(&dll) }.unwrap_or_else(|e| panic!("cannot load {} (cargo build -p cudnn_shim): {e}", dll.display()))
}

macro_rules! dnn {
    ($lib:expr, $name:literal, fn($($a:ty),*)) => {
        *unsafe { $lib.get::<unsafe extern "system" fn($($a),*) -> i32>(concat!($name, "\0").as_bytes()) }
            .expect($name)
    };
}

// cuDNN v9 numbering; the v8 build uses 1..14 (see docs/cudnn-abi-todo.md).
const NOT_SUPPORTED: i32 = 3000;

#[derive(Clone, Copy, PartialEq, Debug)]
enum Dt {
    F32,
    F16,
    Bf16,
}
impl Dt {
    fn code(self) -> i32 {
        match self {
            Dt::F32 => 0,
            Dt::F16 => 2,
            Dt::Bf16 => 9,
        }
    }
    fn bytes(self) -> usize {
        if self == Dt::F32 { 4 } else { 2 }
    }
    fn enc(self, v: &[f32]) -> Vec<u8> {
        match self {
            Dt::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Dt::F16 => v.iter().flat_map(|x| f32_to_f16(*x).to_le_bytes()).collect(),
            Dt::Bf16 => v.iter().flat_map(|x| f32_to_bf16(*x).to_le_bytes()).collect(),
        }
    }
    fn dec(self, b: &[u8]) -> Vec<f32> {
        match self {
            Dt::F32 => b.chunks(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect(),
            Dt::F16 => b.chunks(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect(),
            Dt::Bf16 => b.chunks(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect(),
        }
    }
}

/// Round to nearest even.
fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7fff + ((b >> 16) & 1)) >> 16) as u16
}

struct Cu {
    alloc: unsafe extern "system" fn(*mut u64, usize) -> i32,
    htod: unsafe extern "system" fn(u64, *const c_void, usize) -> i32,
    dtoh: unsafe extern "system" fn(*mut c_void, u64, usize) -> i32,
    sync: unsafe extern "system" fn() -> i32,
}

impl Cu {
    fn up(&self, bytes: &[u8]) -> u64 {
        let mut p = 0u64;
        assert_eq!(unsafe { (self.alloc)(&mut p, bytes.len().max(4)) }, 0);
        assert_eq!(unsafe { (self.htod)(p, bytes.as_ptr().cast(), bytes.len()) }, 0);
        p
    }
    fn down(&self, p: u64, n: usize) -> Vec<u8> {
        assert_eq!(unsafe { (self.sync)() }, 0, "ctx sync");
        let mut v = vec![0u8; n];
        assert_eq!(unsafe { (self.dtoh)(v.as_mut_ptr().cast(), p, n) }, 0);
        v
    }
}

fn sample(len: usize, seed: usize) -> Vec<f32> {
    (0..len).map(|i| (((i * 5 + seed * 3 + i / 7) % 16) as f32 - 8.0) / 8.0).collect()
}

fn check(name: &str, got: &[f32], want: &[f32], rel: f32) {
    assert_eq!(got.len(), want.len(), "{name} length");
    let mut worst = 0f32;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        worst = worst.max(err);
        assert!(err <= rel * (1.0 + w.abs()), "{name}[{i}]: got {g} want {w}");
    }
    println!("  {name}: max abs err {worst:.3e}");
}

struct Api<'a> {
    cudnn: &'a Dll,
    h: *mut c_void,
    /// Build descriptors with the Nd entry points, as PyTorch does.
    nd: std::cell::Cell<bool>,
}

impl Api<'_> {
    fn tensor(&self, nhwc: bool, dt: Dt, d: [i32; 4]) -> *mut c_void {
        let create = dnn!(self.cudnn, "cudnnCreateTensorDescriptor", fn(*mut *mut c_void));
        let set = dnn!(self.cudnn, "cudnnSetTensor4dDescriptor", fn(*mut c_void, i32, i32, i32, i32, i32, i32));
        let mut t = null_mut();
        assert_eq!(unsafe { create(&mut t) }, 0);
        if self.nd.get() {
            let setnd = dnn!(self.cudnn, "cudnnSetTensorNdDescriptor", fn(*mut c_void, i32, i32, *const i32, *const i32));
            let st = if nhwc { [d[1] * d[2] * d[3], 1, d[3] * d[1], d[1]] } else { [d[1] * d[2] * d[3], d[2] * d[3], d[3], 1] };
            assert_eq!(unsafe { setnd(t, dt.code(), 4, d.as_ptr(), st.as_ptr()) }, 0);
            return t;
        }
        assert_eq!(unsafe { set(t, nhwc as i32, dt.code(), d[0], d[1], d[2], d[3]) }, 0);
        t
    }
    fn filter(&self, nhwc: bool, dt: Dt, d: [i32; 4]) -> *mut c_void {
        let create = dnn!(self.cudnn, "cudnnCreateFilterDescriptor", fn(*mut *mut c_void));
        let set = dnn!(self.cudnn, "cudnnSetFilter4dDescriptor", fn(*mut c_void, i32, i32, i32, i32, i32, i32));
        let mut f = null_mut();
        assert_eq!(unsafe { create(&mut f) }, 0);
        if self.nd.get() {
            let setnd = dnn!(self.cudnn, "cudnnSetFilterNdDescriptor", fn(*mut c_void, i32, i32, i32, *const i32));
            assert_eq!(unsafe { setnd(f, dt.code(), nhwc as i32, 4, d.as_ptr()) }, 0);
            return f;
        }
        assert_eq!(unsafe { set(f, dt.code(), nhwc as i32, d[0], d[1], d[2], d[3]) }, 0);
        f
    }
    fn conv(&self, pad: i32, stride: i32, dil: i32, groups: i32, flip: bool) -> *mut c_void {
        let create = dnn!(self.cudnn, "cudnnCreateConvolutionDescriptor", fn(*mut *mut c_void));
        let set = dnn!(self.cudnn, "cudnnSetConvolution2dDescriptor", fn(*mut c_void, i32, i32, i32, i32, i32, i32, i32, i32));
        let grp = dnn!(self.cudnn, "cudnnSetConvolutionGroupCount", fn(*mut c_void, i32));
        let mut c = null_mut();
        assert_eq!(unsafe { create(&mut c) }, 0);
        if self.nd.get() {
            let setnd = dnn!(self.cudnn, "cudnnSetConvolutionNdDescriptor", fn(*mut c_void, i32, *const i32, *const i32, *const i32, i32, i32));
            assert_eq!(unsafe { setnd(c, 2, [pad; 2].as_ptr(), [stride; 2].as_ptr(), [dil; 2].as_ptr(), if flip { 0 } else { 1 }, 0) }, 0);
            assert_eq!(unsafe { grp(c, groups) }, 0);
            return c;
        }
        assert_eq!(unsafe { set(c, pad, pad, stride, stride, dil, dil, if flip { 0 } else { 1 }, 0) }, 0);
        assert_eq!(unsafe { grp(c, groups) }, 0);
        c
    }
}

struct Case {
    nhwc: bool,
    dt: Dt,
    groups: u32,
    dil: u32,
    stride: u32,
    pad: u32,
    flip: bool,
    ks: u32,
}

fn run_conv_case(cu: &Cu, api: &Api, c: &Case) {
    let (n, ch, h, w, k, r, s) = (2u32, 8u32, 17u32, 13u32, 16u32, c.ks, c.ks);
    let p = out_dim(h, c.pad, r, c.dil, c.stride).unwrap();
    let q = out_dim(w, c.pad, s, c.dil, c.stride).unwrap();
    let cg = ch / c.groups;
    let (xs, ys, ws) = if c.nhwc {
        (ConvGeom::packed_nhwc(ch, h, w), ConvGeom::packed_nhwc(k, p, q), [r * s * cg, 1, s * cg, cg])
    } else {
        (ConvGeom::packed_nchw(ch, h, w), ConvGeom::packed_nchw(k, p, q), [cg * r * s, r * s, s, 1])
    };
    let g = ConvGeom {
        n, c: ch, h, w, k, r, s, p, q, pad_h: c.pad, pad_w: c.pad, str_h: c.stride, str_w: c.stride, dil_h: c.dil,
        dil_w: c.dil, groups: c.groups, flip: c.flip, xs, ws, ys,
    };
    let rt = |v: Vec<f32>| c.dt.dec(&c.dt.enc(&v));
    let (x, wv, dy) = (rt(sample(g.x_len(), 1)), rt(sample(g.w_len(), 2)), rt(sample(g.y_len(), 3)));
    let (xd, dyd) = (api.tensor(c.nhwc, c.dt, [n as i32, ch as i32, h as i32, w as i32]), api.tensor(c.nhwc, c.dt, [n as i32, k as i32, p as i32, q as i32]));
    let wd = api.filter(c.nhwc, c.dt, [k as i32, cg as i32, r as i32, s as i32]);
    let cd = api.conv(c.pad as i32, c.stride as i32, c.dil as i32, c.groups as i32, c.flip);
    let (dx_ptr, dw_ptr, y_ptr) = (cu.up(&vec![0u8; g.x_len() * c.dt.bytes()]), cu.up(&vec![0u8; g.w_len() * c.dt.bytes()]), cu.up(&vec![0u8; g.y_len() * c.dt.bytes()]));
    let (x_ptr, w_ptr, dy_ptr) = (cu.up(&c.dt.enc(&x)), cu.up(&c.dt.enc(&wv)), cu.up(&c.dt.enc(&dy)));
    let (alpha, beta) = (1f32, 0f32);
    let tol = match c.dt {
        Dt::F32 => 1e-4,
        Dt::F16 => 2e-2,
        Dt::Bf16 => 8e-2,
    };
    let label = format!("{:?} nhwc={} g={} dil={} str={} pad={} flip={}", c.dt, c.nhwc, c.groups, c.dil, c.stride, c.pad, c.flip);
    println!("conv {label}");

    // algorithm query + workspace
    let get_algo = dnn!(api.cudnn, "cudnnGetConvolutionForwardAlgorithm_v7", fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, i32, *mut i32, *mut c_void));
    let mut perf = [0u8; 48 * 4];
    let mut count = 0;
    assert_eq!(unsafe { get_algo(api.h, xd, wd, cd, dyd, 4, &mut count, perf.as_mut_ptr().cast()) }, 0);
    assert_eq!(count, 1);

    let fwd = dnn!(api.cudnn, "cudnnConvolutionForward", fn(*mut c_void, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, i32, *mut c_void, usize, *const f32, *mut c_void, u64));
    let t = Instant::now();
    assert_eq!(unsafe { fwd(api.h, &alpha, xd, x_ptr, wd, w_ptr, cd, 0, null_mut(), 0, &beta, dyd, y_ptr) }, 0);
    let got = c.dt.dec(&cu.down(y_ptr, g.y_len() * c.dt.bytes()));
    println!("  forward {:?}", t.elapsed());
    let mut want = vec![0f32; g.y_len()];
    refmath::conv_fwd(&g, &x, &wv, &mut want, 1.0, 0.0);
    check("fwd", &got, &c.dt.dec(&c.dt.enc(&want)), tol);

    let bwd_d = dnn!(api.cudnn, "cudnnConvolutionBackwardData", fn(*mut c_void, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, i32, *mut c_void, usize, *const f32, *mut c_void, u64));
    assert_eq!(unsafe { bwd_d(api.h, &alpha, wd, w_ptr, dyd, dy_ptr, cd, 0, null_mut(), 0, &beta, xd, dx_ptr) }, 0);
    let got = c.dt.dec(&cu.down(dx_ptr, g.x_len() * c.dt.bytes()));
    let mut want = vec![0f32; g.x_len()];
    refmath::conv_bwd_data(&g, &wv, &dy, &mut want, 1.0, 0.0);
    check("bwd data", &got, &c.dt.dec(&c.dt.enc(&want)), tol);

    let bwd_f = dnn!(api.cudnn, "cudnnConvolutionBackwardFilter", fn(*mut c_void, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, i32, *mut c_void, usize, *const f32, *mut c_void, u64));
    assert_eq!(unsafe { bwd_f(api.h, &alpha, xd, x_ptr, dyd, dy_ptr, cd, 0, null_mut(), 0, &beta, wd, dw_ptr) }, 0);
    let got = c.dt.dec(&cu.down(dw_ptr, g.w_len() * c.dt.bytes()));
    let mut want = vec![0f32; g.w_len()];
    refmath::conv_bwd_filter(&g, &x, &dy, &mut want, 1.0, 0.0);
    check("bwd filter", &got, &c.dt.dec(&c.dt.enc(&want)), tol * 4.0);

    // alpha/beta blending on the forward pass: y = 2*conv + 0.5*y
    let before = c.dt.dec(&cu.down(y_ptr, g.y_len() * c.dt.bytes()));
    let (a2, b2) = (2f32, 0.5f32);
    assert_eq!(unsafe { fwd(api.h, &a2, xd, x_ptr, wd, w_ptr, cd, 0, null_mut(), 0, &b2, dyd, y_ptr) }, 0);
    let got = c.dt.dec(&cu.down(y_ptr, g.y_len() * c.dt.bytes()));
    let mut want = before.clone();
    refmath::conv_fwd(&g, &x, &wv, &mut want, 2.0, 0.5);
    check("fwd alpha/beta", &got, &c.dt.dec(&c.dt.enc(&want)), tol);

    // Same three directions with a caller-supplied workspace (explicit-GEMM path where eligible) and with a
    // workspace that is too small (must fall back to the direct kernels, never fail or corrupt).
    let ws_fwd = dnn!(api.cudnn, "cudnnGetConvolutionForwardWorkspaceSize", fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, i32, *mut usize));
    let ws_bd = dnn!(api.cudnn, "cudnnGetConvolutionBackwardDataWorkspaceSize", fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, i32, *mut usize));
    let ws_bf = dnn!(api.cudnn, "cudnnGetConvolutionBackwardFilterWorkspaceSize", fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, i32, *mut usize));
    let (mut sf, mut sd, mut sw) = (0usize, 0usize, 0usize);
    assert_eq!(unsafe { ws_fwd(api.h, xd, wd, cd, dyd, 0, &mut sf) }, 0);
    assert_eq!(unsafe { ws_bd(api.h, wd, dyd, cd, xd, 0, &mut sd) }, 0);
    assert_eq!(unsafe { ws_bf(api.h, xd, dyd, cd, wd, 0, &mut sw) }, 0);
    println!("  workspace sizes fwd/bwd-data/bwd-filter: {sf}/{sd}/{sw}");
    if sf + sd + sw > 0 {
        GEMM_PATH_CASES.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
    }
    for tiny in [false, true] {
        let wsp = |sz: usize| if tiny { (cu.up(&[0u8; 16]), 16usize) } else { (cu.up(&vec![0u8; sz.max(4)]), sz) };
        let tag = if tiny { "tiny workspace" } else { "workspace" };
        let zero_y = cu.up(&vec![0u8; g.y_len() * c.dt.bytes()]);
        let (wp, wb) = wsp(sf);
        assert_eq!(unsafe { fwd(api.h, &alpha, xd, x_ptr, wd, w_ptr, cd, 0, wp as *mut c_void, wb, &beta, dyd, zero_y) }, 0);
        let got = c.dt.dec(&cu.down(zero_y, g.y_len() * c.dt.bytes()));
        let mut want = vec![0f32; g.y_len()];
        refmath::conv_fwd(&g, &x, &wv, &mut want, 1.0, 0.0);
        check(&format!("fwd {tag}"), &got, &c.dt.dec(&c.dt.enc(&want)), tol);
        let zero_x = cu.up(&vec![0u8; g.x_len() * c.dt.bytes()]);
        let (wp, wb) = wsp(sd);
        assert_eq!(unsafe { bwd_d(api.h, &alpha, wd, w_ptr, dyd, dy_ptr, cd, 0, wp as *mut c_void, wb, &beta, xd, zero_x) }, 0);
        let got = c.dt.dec(&cu.down(zero_x, g.x_len() * c.dt.bytes()));
        let mut want = vec![0f32; g.x_len()];
        refmath::conv_bwd_data(&g, &wv, &dy, &mut want, 1.0, 0.0);
        check(&format!("bwd data {tag}"), &got, &c.dt.dec(&c.dt.enc(&want)), tol);
        // accumulate into a non-zero dw (beta = 1) to cover the chunk accumulation
        let dw0 = rt(sample(g.w_len(), 9));
        let dwp = cu.up(&c.dt.enc(&dw0));
        let (wp, wb) = wsp(sw);
        let one = 1f32;
        assert_eq!(unsafe { bwd_f(api.h, &alpha, xd, x_ptr, dyd, dy_ptr, cd, 0, wp as *mut c_void, wb, &one, wd, dwp) }, 0);
        let got = c.dt.dec(&cu.down(dwp, g.w_len() * c.dt.bytes()));
        let mut want = dw0.clone();
        refmath::conv_bwd_filter(&g, &x, &dy, &mut want, 1.0, 1.0);
        check(&format!("bwd filter {tag}"), &got, &c.dt.dec(&c.dt.enc(&want)), tol * 4.0);
    }
    let _ = Kind::Fwd;
}

#[test]
#[ignore = "requires a built nvcuda.dll, cudnn64_9.dll and an Intel Level Zero GPU"]
fn hw_cudnn_dense() {
    let (tx, rx) = std::sync::mpsc::channel();
    let handle = std::thread::spawn(move || {
        dense();
        let _ = tx.send(());
    });
    match rx.recv_timeout(std::time::Duration::from_secs(std::env::var("CFI_HW_WATCHDOG_S").ok().and_then(|v| v.parse().ok()).unwrap_or(900))) {
        Ok(()) => handle.join().unwrap(),
        Err(_) => {
            eprintln!("WATCHDOG: cudnn hardware test hung");
            std::process::exit(99);
        }
    }
}

fn dense() {
    let drv = library();
    symbols!(drv; cuInit(u32), cuCtxCreate_v2(*mut Handle, u32, i32));
    ok!(cuInit(0));
    let mut ctx = null_mut();
    ok!(cuCtxCreate_v2(&mut ctx, 0, test_device()));
    let cu = Cu {
        alloc: *unsafe { drv.get::<unsafe extern "system" fn(*mut u64, usize) -> i32>(b"cuMemAlloc_v2\0") }.unwrap(),
        htod: *unsafe { drv.get::<unsafe extern "system" fn(u64, *const c_void, usize) -> i32>(b"cuMemcpyHtoD_v2\0") }.unwrap(),
        dtoh: *unsafe { drv.get::<unsafe extern "system" fn(*mut c_void, u64, usize) -> i32>(b"cuMemcpyDtoH_v2\0") }.unwrap(),
        sync: *unsafe { drv.get::<unsafe extern "system" fn() -> i32>(b"cuCtxSynchronize\0") }.unwrap(),
    };
    // The explicit-GEMM path finds the cuBLAS shim by module name: preload it like an application would.
    let cublas = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("cublas64_12.dll");
    let cublas_loaded = match unsafe { Dll::new(&cublas) } {
        Ok(l) => {
            std::mem::forget(l);
            true
        }
        Err(e) => {
            println!("cublas64_12.dll not preloaded ({e}); GEMM path not exercised");
            false
        }
    };
    let cudnn = cudnn_dll();

    // --- library basics -------------------------------------------------
    let get_version = dnn!(cudnn, "cudnnGetVersion", fn());
    println!("cudnnGetVersion = {}", unsafe { get_version() });
    let create = dnn!(cudnn, "cudnnCreate", fn(*mut *mut c_void));
    let mut h = null_mut();
    assert_eq!(unsafe { create(&mut h) }, 0);
    let api = Api { cudnn: &cudnn, h, nd: std::cell::Cell::new(false) };
    let softmax = dnn!(cudnn, "cudnnSoftmaxForward", fn());
    assert_eq!(unsafe { softmax() }, NOT_SUPPORTED, "softmax must report NOT_SUPPORTED");

    // --- convolutions -----------------------------------------------------
    let cases = [
        Case { nhwc: false, dt: Dt::F32, groups: 1, dil: 1, stride: 1, pad: 1, flip: false, ks: 3 },
        Case { nhwc: true, dt: Dt::F32, groups: 1, dil: 1, stride: 2, pad: 1, flip: false, ks: 3 },
        Case { nhwc: false, dt: Dt::F32, groups: 2, dil: 2, stride: 1, pad: 2, flip: true, ks: 3 },
        Case { nhwc: true, dt: Dt::F32, groups: 8, dil: 1, stride: 1, pad: 0, flip: false, ks: 3 }, // depthwise
        Case { nhwc: false, dt: Dt::F16, groups: 1, dil: 1, stride: 1, pad: 1, flip: false, ks: 3 },
        Case { nhwc: true, dt: Dt::F16, groups: 4, dil: 1, stride: 2, pad: 1, flip: false, ks: 3 },
        // kernel sizes real networks use: 1x1, 5x5, 7x7 stem (stride 2, pad 3), dilated + strided
        Case { nhwc: false, dt: Dt::F32, groups: 1, dil: 1, stride: 1, pad: 0, flip: false, ks: 1 },
        Case { nhwc: true, dt: Dt::F16, groups: 1, dil: 1, stride: 2, pad: 0, flip: false, ks: 1 },
        Case { nhwc: false, dt: Dt::F32, groups: 1, dil: 1, stride: 1, pad: 2, flip: false, ks: 5 },
        Case { nhwc: false, dt: Dt::F32, groups: 1, dil: 1, stride: 2, pad: 3, flip: false, ks: 7 },
        Case { nhwc: true, dt: Dt::F16, groups: 1, dil: 1, stride: 2, pad: 3, flip: false, ks: 7 },
        Case { nhwc: false, dt: Dt::F16, groups: 2, dil: 2, stride: 2, pad: 2, flip: false, ks: 3 },
        Case { nhwc: true, dt: Dt::F32, groups: 2, dil: 2, stride: 2, pad: 2, flip: false, ks: 3 },
        Case { nhwc: false, dt: Dt::Bf16, groups: 1, dil: 1, stride: 1, pad: 1, flip: false, ks: 3 },
        Case { nhwc: true, dt: Dt::Bf16, groups: 1, dil: 1, stride: 2, pad: 1, flip: false, ks: 3 },
    ];
    for c in &cases {
        run_conv_case(&cu, &api, c);
    }
    // The same through the Nd entry points (what PyTorch calls), including non-default layouts.
    api.nd.set(true);
    for c in &cases {
        run_conv_case(&cu, &api, c);
    }
    api.nd.set(false);
    unsupported_probes(&cudnn, &cu, &api);
    let gemm_cases = GEMM_PATH_CASES.load(std::sync::atomic::Ordering::SeqCst);
    println!("cases that used the explicit-GEMM workspace: {gemm_cases}");
    assert!(gemm_cases > 0 || !cublas_loaded, "cuBLAS shim preloaded but no case reported a workspace");

    // --- AddTensor + ConvolutionBiasActivationForward ----------------------
    {
        let (n, ch, h2, w2) = (2i32, 6i32, 5i32, 4i32);
        let ydesc = api.tensor(false, Dt::F32, [n, ch, h2, w2]);
        let bdesc = api.tensor(false, Dt::F32, [1, ch, 1, 1]);
        let y0 = sample((n * ch * h2 * w2) as usize, 9);
        let bias = sample(ch as usize, 4);
        let (yp, bp) = (cu.up(&Dt::F32.enc(&y0)), cu.up(&Dt::F32.enc(&bias)));
        let add = dnn!(cudnn, "cudnnAddTensor", fn(*mut c_void, *const f32, *mut c_void, u64, *const f32, *mut c_void, u64));
        let (a, b) = (2f32, 0.5f32);
        assert_eq!(unsafe { add(h, &a, bdesc, bp, &b, ydesc, yp) }, 0);
        let got = Dt::F32.dec(&cu.down(yp, y0.len() * 4));
        let want: Vec<f32> = y0.iter().enumerate().map(|(i, v)| 2.0 * bias[(i / (h2 * w2) as usize) % ch as usize] + 0.5 * v).collect();
        check("AddTensor bias", &got, &want, 1e-5);

        // y = relu(conv(x,w) + bias), 1x1 conv, z unused (alpha2 = 0)
        let (ci, co) = (4i32, 6i32);
        let xd = api.tensor(false, Dt::F32, [n, ci, h2, w2]);
        let yd = api.tensor(false, Dt::F32, [n, co, h2, w2]);
        let wd = api.filter(false, Dt::F32, [co, ci, 1, 1]);
        let cd = api.conv(0, 1, 1, 1, false);
        let ad_create = dnn!(cudnn, "cudnnCreateActivationDescriptor", fn(*mut *mut c_void));
        let ad_set = dnn!(cudnn, "cudnnSetActivationDescriptor", fn(*mut c_void, i32, i32, f64));
        let mut act = null_mut();
        assert_eq!(unsafe { ad_create(&mut act) }, 0);
        assert_eq!(unsafe { ad_set(act, 1, 0, 0.0) }, 0);
        let g = ConvGeom { n: 2, c: 4, h: 5, w: 4, k: 6, r: 1, s: 1, p: 5, q: 4, pad_h: 0, pad_w: 0, str_h: 1, str_w: 1, dil_h: 1, dil_w: 1, groups: 1, flip: false, xs: ConvGeom::packed_nchw(4, 5, 4), ws: [4, 1, 1, 1], ys: ConvGeom::packed_nchw(6, 5, 4) };
        let (x, wv) = (sample(g.x_len(), 1), sample(g.w_len(), 2));
        let (xp, wp) = (cu.up(&Dt::F32.enc(&x)), cu.up(&Dt::F32.enc(&wv)));
        let outp = cu.up(&vec![0u8; g.y_len() * 4]);
        let cba = dnn!(cudnn, "cudnnConvolutionBiasActivationForward", fn(*mut c_void, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, i32, *mut c_void, usize, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, *mut c_void, u64));
        let (one, zero) = (1f32, 0f32);
        assert_eq!(unsafe { cba(h, &one, xd, xp, wd, wp, cd, 0, null_mut(), 0, &zero, yd, 0, bdesc_for(&api, co), bp, act, yd, outp) }, 0, "ConvBiasActivation");
        let got = Dt::F32.dec(&cu.down(outp, g.y_len() * 4));
        let mut want = vec![0f32; g.y_len()];
        refmath::conv_fwd(&g, &x, &wv, &mut want, 1.0, 0.0);
        for (i, v) in want.iter_mut().enumerate() {
            *v = (*v + bias[(i / 20) % 6]).max(0.0);
        }
        check("ConvBiasActivation relu", &got, &want, 1e-4);
    }

    // --- Batch norm (spatial, NCHW f32) --------------------------------------
    {
        let (n, c, hh, ww) = (4usize, 8usize, 9usize, 7usize);
        let hw = hh * ww;
        let x: Vec<f32> = (0..n * c * hw).map(|i| ((i * 7) % 23) as f32 / 8.0 - 1.0 + (i / (c * hw)) as f32 * 0.25).collect();
        let scale: Vec<f32> = (0..c).map(|i| 0.5 + i as f32 * 0.25).collect();
        let bias: Vec<f32> = (0..c).map(|i| i as f32 * 0.1 - 0.3).collect();
        let eps = 1e-5f32;
        let (want_y, want_mean, want_inv, want_uvar) = refmath::bn_train(&x, n, c, hw, &scale, &bias, eps);
        let xd = api.tensor(false, Dt::F32, [n as i32, c as i32, hh as i32, ww as i32]);
        let bnd = api.tensor(false, Dt::F32, [1, c as i32, 1, 1]);
        let (xp, yp) = (cu.up(&Dt::F32.enc(&x)), cu.up(&vec![0u8; x.len() * 4]));
        let (sp, bp) = (cu.up(&Dt::F32.enc(&scale)), cu.up(&Dt::F32.enc(&bias)));
        let rm0 = vec![1.0f32; c];
        let rv0 = vec![2.0f32; c];
        let (rmp, rvp) = (cu.up(&Dt::F32.enc(&rm0)), cu.up(&Dt::F32.enc(&rv0)));
        let (smp, sip) = (cu.up(&vec![0u8; c * 4]), cu.up(&vec![0u8; c * 4]));
        let train = dnn!(cudnn, "cudnnBatchNormalizationForwardTraining", fn(*mut c_void, i32, *const f32, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, u64, u64, f64, u64, u64, f64, u64, u64));
        let (one, zero) = (1f32, 0f32);
        let factor = 0.1f64;
        let t = Instant::now();
        assert_eq!(unsafe { train(h, 1, &one, &zero, xd, xp, xd, yp, bnd, sp, bp, factor, rmp, rvp, eps as f64, smp, sip) }, 0, "bn train");
        println!("bn train {:?}", t.elapsed());
        check("bn train y", &Dt::F32.dec(&cu.down(yp, x.len() * 4)), &want_y, 2e-4);
        check("bn save mean", &Dt::F32.dec(&cu.down(smp, c * 4)), &want_mean, 2e-4);
        check("bn save invstd", &Dt::F32.dec(&cu.down(sip, c * 4)), &want_inv, 2e-4);
        let want_rm: Vec<f32> = (0..c).map(|i| 0.9 * rm0[i] + 0.1 * want_mean[i]).collect();
        let want_rv: Vec<f32> = (0..c).map(|i| 0.9 * rv0[i] + 0.1 * want_uvar[i]).collect();
        check("bn running mean", &Dt::F32.dec(&cu.down(rmp, c * 4)), &want_rm, 2e-4);
        check("bn running var", &Dt::F32.dec(&cu.down(rvp, c * 4)), &want_rv, 2e-4);

        let infer = dnn!(cudnn, "cudnnBatchNormalizationForwardInference", fn(*mut c_void, i32, *const f32, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, u64, u64, u64, u64, f64));
        let yi = cu.up(&vec![0u8; x.len() * 4]);
        let var: Vec<f32> = want_inv.iter().map(|i| 1.0 / (i * i) - eps).collect();
        let (mp, vp) = (cu.up(&Dt::F32.enc(&want_mean)), cu.up(&Dt::F32.enc(&var)));
        assert_eq!(unsafe { infer(h, 1, &one, &zero, xd, xp, xd, yi, bnd, sp, bp, mp, vp, eps as f64) }, 0, "bn infer");
        check("bn inference y", &Dt::F32.dec(&cu.down(yi, x.len() * 4)), &want_y, 2e-4);

        let dy: Vec<f32> = (0..x.len()).map(|i| ((i * 3) % 7) as f32 / 4.0 - 0.75).collect();
        let (wdx, wds, wdb) = refmath::bn_bwd(&x, &dy, n, c, hw, &scale, &want_mean, &want_inv);
        let bwd = dnn!(cudnn, "cudnnBatchNormalizationBackward", fn(*mut c_void, i32, *const f32, *const f32, *const f32, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, u64, *mut c_void, u64, u64, u64, f64, u64, u64));
        for saved in [true, false] {
            let (dyp, dxp) = (cu.up(&Dt::F32.enc(&dy)), cu.up(&vec![0u8; x.len() * 4]));
            let (dsp, dbp) = (cu.up(&vec![0u8; c * 4]), cu.up(&vec![0u8; c * 4]));
            let (m, i) = if saved { (smp, sip) } else { (0, 0) };
            assert_eq!(unsafe { bwd(h, 1, &one, &zero, &one, &zero, xd, xp, xd, dyp, xd, dxp, bnd, sp, dsp, dbp, eps as f64, m, i) }, 0, "bn bwd");
            let tag = if saved { "saved stats" } else { "recomputed stats" };
            check(&format!("bn bwd dx ({tag})"), &Dt::F32.dec(&cu.down(dxp, x.len() * 4)), &wdx, 5e-4);
            check(&format!("bn bwd dscale ({tag})"), &Dt::F32.dec(&cu.down(dsp, c * 4)), &wds, 5e-4);
            check(&format!("bn bwd dbias ({tag})"), &Dt::F32.dec(&cu.down(dbp, c * 4)), &wdb, 5e-4);
        }
    }

    // --- v8 backend API: single conv graph, like cudnn_frontend --------------
    backend_flow(&cudnn, &cu, h);

    // --- Throughput record: ResNet-like 3x3 conv, NCHW f32 and NHWC f16 -----
    for (nhwc, dt) in [(false, Dt::F32), (true, Dt::F16)] {
        let (n, c, hh, ww, k) = (8i32, 64i32, 56i32, 56i32, 64i32);
        let xd = api.tensor(nhwc, dt, [n, c, hh, ww]);
        let yd = api.tensor(nhwc, dt, [n, k, hh, ww]);
        let wd = api.filter(nhwc, dt, [k, c, 3, 3]);
        let cd = api.conv(1, 1, 1, 1, false);
        let bytes = |e: usize| e * dt.bytes();
        let (xp, wp, yp) = (cu.up(&vec![0u8; bytes((n * c * hh * ww) as usize)]), cu.up(&vec![0u8; bytes((k * c * 9) as usize)]), cu.up(&vec![0u8; bytes((n * k * hh * ww) as usize)]));
        let fwd = dnn!(cudnn, "cudnnConvolutionForward", fn(*mut c_void, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, i32, *mut c_void, usize, *const f32, *mut c_void, u64));
        let (one, zero) = (1f32, 0f32);
        assert_eq!(unsafe { fwd(h, &one, xd, xp, wd, wp, cd, 0, null_mut(), 0, &zero, yd, yp) }, 0);
        cu.down(yp, 4);
        let t = Instant::now();
        for _ in 0..5 {
            assert_eq!(unsafe { fwd(h, &one, xd, xp, wd, wp, cd, 0, null_mut(), 0, &zero, yd, yp) }, 0);
        }
        cu.down(yp, 4);
        let secs = t.elapsed().as_secs_f64() / 5.0;
        let flops = 2.0 * (n * k * hh * ww) as f64 * (c * 9) as f64;
        println!("throughput conv 8x64x56x56 -> 64 3x3 nhwc={nhwc} {dt:?}: {:.2} ms, {:.1} GFLOP/s", secs * 1e3, flops / secs / 1e9);
    }
    println!("hw_cudnn_dense: ALL OK");
}

fn bdesc_for(api: &Api, k: i32) -> *mut c_void {
    api.tensor(false, Dt::F32, [1, k, 1, 1])
}

fn backend_flow(cudnn: &Dll, cu: &Cu, h: *mut c_void) {
    type Create = unsafe extern "system" fn(i32, *mut *mut c_void) -> i32;
    type SetAttr = unsafe extern "system" fn(*mut c_void, i32, i32, i64, *const c_void) -> i32;
    type GetAttr = unsafe extern "system" fn(*mut c_void, i32, i32, i64, *mut i64, *mut c_void) -> i32;
    type Fin = unsafe extern "system" fn(*mut c_void) -> i32;
    let create: Create = dnn!(cudnn, "cudnnBackendCreateDescriptor", fn(i32, *mut *mut c_void));
    let set: SetAttr = dnn!(cudnn, "cudnnBackendSetAttribute", fn(*mut c_void, i32, i32, i64, *const c_void));
    let get: GetAttr = dnn!(cudnn, "cudnnBackendGetAttribute", fn(*mut c_void, i32, i32, i64, *mut i64, *mut c_void));
    let fin: Fin = dnn!(cudnn, "cudnnBackendFinalize", fn(*mut c_void));
    let exec = dnn!(cudnn, "cudnnBackendExecute", fn(*mut c_void, *mut c_void, *mut c_void));
    let mk = |ty: i32| {
        let mut d = null_mut();
        assert_eq!(unsafe { create(ty, &mut d) }, 0);
        d
    };
    let seti64 = |d: *mut c_void, name: i32, v: &[i64]| assert_eq!(unsafe { set(d, name, 3, v.len() as i64, v.as_ptr().cast()) }, 0, "attr {name}");
    let setd = |d: *mut c_void, name: i32, v: &[*mut c_void]| assert_eq!(unsafe { set(d, name, 15, v.len() as i64, v.as_ptr().cast()) }, 0, "desc attr {name}");
    let tensor = |uid: i64, dims: [i64; 4]| {
        let t = mk(17);
        let strides = [dims[1] * dims[2] * dims[3], dims[2] * dims[3], dims[3], 1];
        seti64(t, 902, &dims);
        seti64(t, 903, &strides);
        assert_eq!(unsafe { set(t, 901, 1, 1, [0i32].as_ptr().cast()) }, 0);
        seti64(t, 906, &[uid]);
        seti64(t, 900, &[16]);
        assert_eq!(unsafe { fin(t) }, 0);
        t
    };
    let (x, w, y) = (tensor(1, [2, 4, 8, 8]), tensor(2, [6, 4, 3, 3]), tensor(3, [2, 6, 8, 8]));
    let cv = mk(1);
    seti64(cv, 106, &[2]);
    seti64(cv, 102, &[1, 1]);
    seti64(cv, 103, &[1, 1]);
    seti64(cv, 105, &[1, 1]);
    seti64(cv, 104, &[1, 1]);
    assert_eq!(unsafe { set(cv, 100, 1, 1, [0i32].as_ptr().cast()) }, 0);
    assert_eq!(unsafe { set(cv, 101, 7, 1, [1i32].as_ptr().cast()) }, 0);
    assert_eq!(unsafe { fin(cv) }, 0);
    let op = mk(10);
    setd(op, 704, &[x]);
    setd(op, 703, &[w]);
    setd(op, 705, &[y]);
    setd(op, 702, &[cv]);
    assert_eq!(unsafe { set(op, 700, 4, 1, [1f32].as_ptr().cast()) }, 0);
    assert_eq!(unsafe { set(op, 701, 4, 1, [0f32].as_ptr().cast()) }, 0);
    assert_eq!(unsafe { fin(op) }, 0, "conv op finalize");
    let graph = mk(15);
    setd(graph, 801, &[op]);
    assert_eq!(unsafe { set(graph, 800, 0, 1, [h].as_ptr().cast()) }, 0);
    assert_eq!(unsafe { fin(graph) }, 0, "graph finalize");
    let heur = mk(4);
    setd(heur, 201, &[graph]);
    assert_eq!(unsafe { set(heur, 200, 8, 1, [0i32].as_ptr().cast()) }, 0);
    assert_eq!(unsafe { fin(heur) }, 0, "heuristics finalize");
    let cfg = mk(3);
    let mut n = 0i64;
    assert_eq!(unsafe { get(heur, 202, 15, 1, &mut n, [cfg].as_ptr() as *mut c_void) }, 0, "heuristic results");
    assert_eq!(n, 1);
    let plan = mk(5);
    assert_eq!(unsafe { set(plan, 400, 0, 1, [h].as_ptr().cast()) }, 0);
    setd(plan, 401, &[cfg]);
    assert_eq!(unsafe { fin(plan) }, 0, "plan finalize");
    let mut ws = -1i64;
    assert_eq!(unsafe { get(plan, 402, 3, 1, &mut n, (&mut ws as *mut i64).cast()) }, 0);
    assert!(ws >= 0);
    println!("backend plan workspace: {ws} bytes");

    let g = ConvGeom { n: 2, c: 4, h: 8, w: 8, k: 6, r: 3, s: 3, p: 8, q: 8, pad_h: 1, pad_w: 1, str_h: 1, str_w: 1, dil_h: 1, dil_w: 1, groups: 1, flip: false, xs: ConvGeom::packed_nchw(4, 8, 8), ws: [36, 9, 3, 1], ys: ConvGeom::packed_nchw(6, 8, 8) };
    let (xv, wv) = (sample(g.x_len(), 1), sample(g.w_len(), 2));
    let (xp, wp, yp) = (cu.up(&Dt::F32.enc(&xv)), cu.up(&Dt::F32.enc(&wv)), cu.up(&vec![0u8; g.y_len() * 4]));
    // First without a workspace (must fall back to the direct kernel), then with the advertised one.
    for with_ws in [false, true] {
        let wsp: u64 = if with_ws && ws > 0 { cu.up(&vec![0u8; ws as usize]) } else { 0 };
        let yp = cu.up(&vec![0u8; g.y_len() * 4]);
        let vp = mk(16);
        seti64(vp, 1000, &[1, 2, 3]);
        assert_eq!(unsafe { set(vp, 1001, 6, 3, [xp, wp, yp].as_ptr().cast()) }, 0);
        assert_eq!(unsafe { set(vp, 1003, 6, 1, [wsp].as_ptr().cast()) }, 0);
        assert_eq!(unsafe { fin(vp) }, 0, "variant pack finalize");
        assert_eq!(unsafe { exec(h, plan, vp) }, 0, "backend execute");
        let mut want = vec![0f32; g.y_len()];
        refmath::conv_fwd(&g, &xv, &wv, &mut want, 1.0, 0.0);
        check(if with_ws { "backend conv fwd (workspace)" } else { "backend conv fwd (no workspace)" }, &Dt::F32.dec(&cu.down(yp, g.y_len() * 4)), &want, 1e-4);
    }

    // a pointwise op must not produce an engine (frontend then reports no plan)
    let pw = mk(13);
    assert_ne!(unsafe { fin(pw) }, 0, "pointwise op alone must not finalize");
}

/// Configurations without a kernel must answer NOT_SUPPORTED (or BAD_PARAM for malformed ones) from every
/// entry point that sees them: never crash, never SUCCESS with garbage.
fn unsupported_probes(cudnn: &Dll, cu: &Cu, api: &Api) {
    let tnd = dnn!(cudnn, "cudnnSetTensorNdDescriptor", fn(*mut c_void, i32, i32, *const i32, *const i32));
    let fnd = dnn!(cudnn, "cudnnSetFilterNdDescriptor", fn(*mut c_void, i32, i32, i32, *const i32));
    let cnd = dnn!(cudnn, "cudnnSetConvolutionNdDescriptor", fn(*mut c_void, i32, *const i32, *const i32, *const i32, i32, i32));
    let mk_t = || {
        let create = dnn!(cudnn, "cudnnCreateTensorDescriptor", fn(*mut *mut c_void));
        let mut t = null_mut();
        assert_eq!(unsafe { create(&mut t) }, 0);
        t
    };
    let mk_f = || {
        let create = dnn!(cudnn, "cudnnCreateFilterDescriptor", fn(*mut *mut c_void));
        let mut t = null_mut();
        assert_eq!(unsafe { create(&mut t) }, 0);
        t
    };
    let mk_c = || {
        let create = dnn!(cudnn, "cudnnCreateConvolutionDescriptor", fn(*mut *mut c_void));
        let mut t = null_mut();
        assert_eq!(unsafe { create(&mut t) }, 0);
        t
    };
    let fwd = dnn!(cudnn, "cudnnConvolutionForward", fn(*mut c_void, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, i32, *mut c_void, usize, *const f32, *mut c_void, u64));
    let bwd_d = dnn!(cudnn, "cudnnConvolutionBackwardData", fn(*mut c_void, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, i32, *mut c_void, usize, *const f32, *mut c_void, u64));
    let bwd_f = dnn!(cudnn, "cudnnConvolutionBackwardFilter", fn(*mut c_void, *const f32, *mut c_void, u64, *mut c_void, u64, *mut c_void, i32, *mut c_void, usize, *const f32, *mut c_void, u64));
    let get_algo = dnn!(cudnn, "cudnnGetConvolutionForwardAlgorithm_v7", fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, i32, *mut i32, *mut c_void));
    let ws_fwd = dnn!(cudnn, "cudnnGetConvolutionForwardWorkspaceSize", fn(*mut c_void, *mut c_void, *mut c_void, *mut c_void, *mut c_void, i32, *mut usize));
    let buf = cu.up(&vec![0u8; 1 << 16]);
    let (one, zero) = (1f32, 0f32);

    // (label, x dtype, w dtype, y dtype, spatial dims (1 = 3-D conv adds a depth axis), conv compute type)
    struct Probe {
        label: &'static str,
        xdt: i32,
        wdt: i32,
        ydt: i32,
        depth: bool,
    }
    let probes = [
        Probe { label: "3-D convolution (conv3d)", xdt: 0, wdt: 0, ydt: 0, depth: true },
        Probe { label: "3-D convolution fp16", xdt: 2, wdt: 2, ydt: 2, depth: true },
        Probe { label: "fp64", xdt: 1, wdt: 1, ydt: 1, depth: false },
        Probe { label: "int8", xdt: 3, wdt: 3, ydt: 3, depth: false },
        Probe { label: "mixed x f32 / w f16", xdt: 0, wdt: 2, ydt: 0, depth: false },
        Probe { label: "mixed y f16", xdt: 0, wdt: 0, ydt: 2, depth: false },
    ];
    for p in &probes {
        let (xd, wd, yd, cd) = (mk_t(), mk_f(), mk_t(), mk_c());
        if p.depth {
            let (xdims, xst) = ([1, 4, 4, 5, 5], [400, 100, 25, 5, 1]);
            let (ydims, yst) = ([1, 4, 4, 5, 5], [400, 100, 25, 5, 1]);
            assert_eq!(unsafe { tnd(xd, p.xdt, 5, xdims.as_ptr(), xst.as_ptr()) }, 0);
            assert_eq!(unsafe { tnd(yd, p.ydt, 5, ydims.as_ptr(), yst.as_ptr()) }, 0);
            assert_eq!(unsafe { fnd(wd, p.wdt, 0, 5, [4, 4, 3, 3, 3].as_ptr()) }, 0);
            assert_eq!(unsafe { cnd(cd, 3, [1; 3].as_ptr(), [1; 3].as_ptr(), [1; 3].as_ptr(), 1, 0) }, 0);
        } else {
            let (dims, st) = ([1, 4, 5, 5], [100, 25, 5, 1]);
            assert_eq!(unsafe { tnd(xd, p.xdt, 4, dims.as_ptr(), st.as_ptr()) }, 0);
            assert_eq!(unsafe { tnd(yd, p.ydt, 4, dims.as_ptr(), st.as_ptr()) }, 0);
            assert_eq!(unsafe { fnd(wd, p.wdt, 0, 4, [4, 4, 3, 3].as_ptr()) }, 0);
            assert_eq!(unsafe { cnd(cd, 2, [1; 2].as_ptr(), [1; 2].as_ptr(), [1; 2].as_ptr(), 1, 0) }, 0);
        }
        let b = buf as *mut c_void;
        assert_eq!(unsafe { fwd(api.h, &one, xd, buf, wd, buf, cd, 0, null_mut(), 0, &zero, yd, buf) }, NOT_SUPPORTED, "{} fwd", p.label);
        assert_eq!(unsafe { bwd_d(api.h, &one, wd, buf, yd, buf, cd, 0, null_mut(), 0, &zero, xd, buf) }, NOT_SUPPORTED, "{} bwd data", p.label);
        assert_eq!(unsafe { bwd_f(api.h, &one, xd, buf, yd, buf, cd, 0, null_mut(), 0, &zero, wd, buf) }, NOT_SUPPORTED, "{} bwd filter", p.label);
        let (mut cnt, mut perf, mut ws) = (-1i32, [0u8; 48 * 4], 7usize);
        assert_eq!(unsafe { get_algo(api.h, xd, wd, cd, yd, 4, &mut cnt, perf.as_mut_ptr().cast()) }, NOT_SUPPORTED, "{} algo query", p.label);
        assert_eq!(cnt, 0, "{} algo count", p.label);
        assert_eq!(unsafe { ws_fwd(api.h, xd, wd, cd, yd, 0, &mut ws) }, NOT_SUPPORTED, "{} workspace query", p.label);
        let _ = b;
        println!("  unsupported OK: {}", p.label);
    }

    // Malformed descriptors: output extent that disagrees with the geometry, filter/channel mismatch, zero batch of stale descriptors.
    let (xd, wd, yd, cd) = (api.tensor(false, Dt::F32, [1, 4, 5, 5]), api.filter(false, Dt::F32, [4, 4, 3, 3]), api.tensor(false, Dt::F32, [1, 4, 9, 9]), api.conv(1, 1, 1, 1, false));
    let st = unsafe { fwd(api.h, &one, xd, buf, wd, buf, cd, 0, null_mut(), 0, &zero, yd, buf) };
    assert!(st == 2000 || st == NOT_SUPPORTED, "wrong output extent -> BAD_PARAM, got {st}");
    let wd2 = api.filter(false, Dt::F32, [4, 3, 3, 3]);
    let yd2 = api.tensor(false, Dt::F32, [1, 4, 5, 5]);
    let st = unsafe { fwd(api.h, &one, xd, buf, wd2, buf, cd, 0, null_mut(), 0, &zero, yd2, buf) };
    assert!(st == 2000 || st == NOT_SUPPORTED, "channel mismatch -> BAD_PARAM, got {st}");
    let st = unsafe { fwd(api.h, &one, xd, 0, wd, buf, cd, 0, null_mut(), 0, &zero, yd2, buf) };
    assert_ne!(st, 0, "null operand pointer must not succeed");
    // fresh (never set) descriptors and a null handle
    let (xe, we, ye, ce) = (mk_t(), mk_f(), mk_t(), mk_c());
    assert_ne!(unsafe { fwd(api.h, &one, xe, buf, we, buf, ce, 0, null_mut(), 0, &zero, ye, buf) }, 0, "unset descriptors");
    assert_ne!(unsafe { fwd(null_mut(), &one, xd, buf, wd, buf, cd, 0, null_mut(), 0, &zero, yd2, buf) }, 0, "null handle");
    println!("  malformed descriptors rejected");
}
