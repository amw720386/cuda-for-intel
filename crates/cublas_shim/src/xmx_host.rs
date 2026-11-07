//! Host side of the native XMX GEMM fast path: qualification, argument
//! packing, kernel cache, one-time probe, and the launcher called from
//! `run_gemm_ex`.
//!
//! Switches (read on every call so tests can toggle them):
//! * `CFI_XMX` unset/`0` - off (the default until hardware validation);
//!   `1` - on for shapes where DPAS pays (`m>=16, n>=8, k>=16`);
//!   `2` - on for every qualifying call, any size (edge-shape testing).
//! * `CFI_XMX_TILE=IBxJB` - subgroup tile in 16-row x 8-column DPAS blocks
//!   (default `2x4`, i.e. 32 x 32 of C).
//! * `CFI_XMX_TRACE=1` - log decisions and probe results.
use crate::plan::*;
use crate::ptxgen::{Elem, Epilogue, Opts};
use crate::xmx::{self, Act, Cfg, Flags, Input, Ty};
use crate::{
    Batching, CUfunction, CUmodule, CUstream, Driver, EXECUTION_FAILED, current_context, env_flag,
};
use std::collections::HashMap;
use std::ffi::{c_int, c_void};
use std::sync::{Mutex, OnceLock};

/// One kernel argument by value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Arg {
    U64(u64),
    U32(u32),
    F32(f32),
}

impl Arg {
    /// Little-endian word the driver reads `size` bytes of.
    fn word(self) -> u64 {
        match self {
            Arg::U64(v) => v,
            Arg::U32(v) => v as u64,
            Arg::F32(v) => v.to_bits() as u64,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    Off,
    /// Only shapes where XMX is expected to beat the tiled kernels.
    Auto,
    /// Every qualifying call.
    Force,
}

pub fn mode() -> Mode {
    match std::env::var("CFI_XMX").as_deref() {
        Ok("1") => Mode::Auto,
        Ok("2") => Mode::Force,
        _ => Mode::Off,
    }
}

pub fn tile() -> (u32, u32) {
    std::env::var("CFI_XMX_TILE")
        .ok()
        .and_then(|v| {
            let (a, b) = v.split_once('x')?;
            let (a, b): (u32, u32) = (a.parse().ok()?, b.parse().ok()?);
            ((1..=4).contains(&a) && (1..=8).contains(&b)).then_some((a, b))
        })
        .unwrap_or((2, 4))
}

fn ty_of(e: Elem) -> Ty {
    match e {
        Elem::F32 => Ty::F32,
        Elem::F16 => Ty::F16,
        Elem::Bf16 => Ty::Bf16,
    }
}

fn dtype_ty(d: DType) -> Ty {
    match d {
        DType::F32 => Ty::F32,
        DType::F16 => Ty::F16,
        DType::Bf16 => Ty::Bf16,
    }
}

/// Everything the qualification needs to know about one call.
#[derive(Clone, Copy, Debug)]
pub struct Call {
    pub d: GemmDesc,
    pub pointer_arrays: bool,
    pub beta: f32,
    pub opts: Opts,
    pub a: u64,
    pub b: u64,
    pub sa: i64,
    pub sb: i64,
}

/// Pick the kernel variant for a call, or `None` when the call does not qualify
/// (the caller then uses the tiled PTX kernels). `flags` is the probed operand
/// spelling for the input type.
pub fn qualify(c: &Call, mode: Mode, tile: (u32, u32), flags: Flags) -> Option<Cfg> {
    let d = &c.d;
    if mode == Mode::Off || d.a != d.b || d.scalar != ScalarKind::F32 || c.opts.acc16 {
        return None;
    }
    let input = match d.a {
        DType::F16 => Input::F16,
        DType::Bf16 => Input::Bf16,
        DType::F32 => return None,
    };
    if d.m == 0 || d.n == 0 || d.k == 0 {
        return None;
    }
    if mode == Mode::Auto && !(d.m >= 16 && d.n >= 8 && d.k >= 16) {
        return None;
    }
    let (ib, jb) = tile;
    if d.n.div_ceil(jb * xmx::DPAS_M) > MAX_GRID_Y {
        return None;
    }
    let act = match c.opts.epilogue {
        Epilogue::Relu | Epilogue::ReluBias => Act::Relu,
        Epilogue::Gelu | Epilogue::GeluBias => Act::Gelu,
        _ => Act::None,
    };
    let bias = c.opts.epilogue.has_bias().then(|| ty_of(c.opts.bias));
    let s = d.strides();
    let strided = !c.pointer_arrays;
    // A is K-contiguous (op T) with even leading dimension and 4-byte alignment.
    let pair_a = strided
        && d.trans_a
        && d.lda % 2 == 0
        && c.a % 4 == 0
        && c.sa % 2 == 0;
    // B has eight contiguous columns per K row (op T), 16-byte aligned, n % 8 == 0.
    let vec_b = strided
        && d.trans_b
        && s.b_sj == 1
        && d.ldb % 8 == 0
        && c.b % 16 == 0
        && c.sb % 8 == 0
        && d.n % 8 == 0
        && d.n >= 8;
    Some(Cfg {
        input,
        out: dtype_ty(d.c),
        ptrs: c.pointer_arrays,
        beta0: c.beta == 0.0,
        act,
        bias,
        ib,
        jb,
        pair_a,
        vec_b,
        flags,
    })
}

/// Kernel arguments for one launch chunk (base pointers already advanced to the chunk).
#[allow(clippy::too_many_arguments)]
pub fn kernel_args(
    d: &GemmDesc,
    a: u64,
    b: u64,
    c: u64,
    alpha: f32,
    beta: f32,
    strides: (u64, u64, u64),
    bias: u64,
) -> Vec<Arg> {
    let s = d.strides();
    vec![
        Arg::U64(a),
        Arg::U64(b),
        Arg::U64(c),
        Arg::F32(alpha),
        Arg::F32(beta),
        Arg::U32(d.m),
        Arg::U32(d.n),
        Arg::U32(d.k),
        Arg::U32(s.a_si),
        Arg::U32(s.a_sl),
        Arg::U32(s.b_sl),
        Arg::U32(s.b_sj),
        Arg::U32(d.ldc),
        Arg::U64(strides.0),
        Arg::U64(strides.1),
        Arg::U64(strides.2),
        Arg::U64(bias),
    ]
}

/// Launch grid `(x, y)` for a variant.
pub fn grid(d: &GemmDesc, cfg: &Cfg) -> (u32, u32) {
    (d.m.div_ceil(cfg.tile_m()), d.n.div_ceil(cfg.tile_n()))
}

// ---------------------------------------------------------------------------
// Driver glue
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Kernel(CUfunction);
unsafe impl Send for Kernel {}
unsafe impl Sync for Kernel {}

fn trace_on() -> bool {
    env_flag("CFI_XMX_TRACE")
}

macro_rules! xtrace {
    ($($arg:tt)*) => {
        if trace_on() {
            eprintln!("CUDA for Intel xmx: {}", format!($($arg)*));
        }
    };
}

/// Load (or fetch from cache) the kernel for `cfg`. `None` if the driver rejects the module.
fn kernel(drv: &Driver, cfg: &Cfg) -> Option<Kernel> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, Cfg), Option<Kernel>>>> = OnceLock::new();
    let ctx = current_context(drv).ok()?;
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(k) = cache.get(&(ctx as usize, *cfg)) {
        return *k;
    }
    let built = (|| {
        let words = xmx::build(cfg).ok()?;
        let text = format!("{}\0", xmx::container(cfg, &words));
        let mut module: CUmodule = std::ptr::null_mut();
        let err = unsafe { (drv.cuModuleLoadData)(&mut module, text.as_ptr().cast()) };
        if err != 0 {
            xtrace!("module load of {} failed err={err}", cfg.name());
            return None;
        }
        let name = std::ffi::CString::new(cfg.name()).ok()?;
        let mut func: CUfunction = std::ptr::null_mut();
        let err = unsafe { (drv.cuModuleGetFunction)(&mut func, module, name.as_ptr()) };
        (err == 0 && !func.is_null()).then_some(Kernel(func))
    })();
    cache.insert((ctx as usize, *cfg), built);
    built
}

unsafe fn launch(
    drv: &Driver,
    k: Kernel,
    grid: (u32, u32, u32),
    stream: CUstream,
    args: &[Arg],
) -> c_int {
    let mut words: Vec<u64> = args.iter().map(|a| a.word()).collect();
    let mut ptrs: Vec<*mut c_void> = words.iter_mut().map(|w| (w as *mut u64).cast()).collect();
    unsafe {
        (drv.cuLaunchKernel)(
            k.0,
            grid.0,
            grid.1,
            grid.2,
            xmx::DPAS_N,
            1,
            1,
            0,
            stream,
            ptrs.as_mut_ptr(),
            std::ptr::null_mut(),
        )
    }
}

/// Deterministic small test values that are exact in f16 and bf16.
fn probe_val(x: usize) -> f32 {
    ((x * 7 + 3) % 9) as f32 / 4.0 - 1.0
}

/// Run a tiny GEMM (edge shape in every dimension) through the variant with the
/// given operand-flag spelling and compare against an exact host result.
fn selftest(drv: &Driver, input: Input, flags: Flags) -> bool {
    let (m, n, k) = (21usize, 10usize, 37usize);
    let enc = |v: f32| -> u16 {
        match input {
            Input::F16 => f32_to_f16_bits(v),
            Input::Bf16 => f32_to_bf16_bits(v),
        }
    };
    let a: Vec<u16> = (0..m * k).map(|i| enc(probe_val(i))).collect();
    let b: Vec<u16> = (0..k * n).map(|i| enc(probe_val(i * 5 + 1))).collect();
    let mut want = vec![0f32; m * n];
    for j in 0..n {
        for i in 0..m {
            // A is m x k column-major (lda = m), B is k x n column-major (ldb = k).
            want[i + j * m] = (0..k).map(|l| probe_val(i + l * m) * probe_val((l + j * k) * 5 + 1)).sum();
        }
    }
    let d = GemmDesc {
        trans_a: false,
        trans_b: false,
        m: m as u32,
        n: n as u32,
        k: k as u32,
        lda: m as u32,
        ldb: k as u32,
        ldc: m as u32,
        a: if input == Input::F16 { DType::F16 } else { DType::Bf16 },
        b: if input == Input::F16 { DType::F16 } else { DType::Bf16 },
        c: DType::F32,
        scalar: ScalarKind::F32,
    };
    let call = Call {
        d,
        pointer_arrays: false,
        beta: 0.0,
        opts: Opts::default(),
        a: 0,
        b: 0,
        sa: 0,
        sb: 0,
    };
    let Some(cfg) = qualify(&call, Mode::Force, (1, 1), flags) else { return false };
    let Some(kern) = kernel(drv, &cfg) else { return false };
    let bytes = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let (ab, bb) = (bytes(&a), bytes(&b));
    let (mut da, mut db, mut dc) = (0u64, 0u64, 0u64);
    unsafe {
        if (drv.cuMemAlloc_v2)(&mut da, ab.len()) != 0
            || (drv.cuMemAlloc_v2)(&mut db, bb.len()) != 0
            || (drv.cuMemAlloc_v2)(&mut dc, m * n * 4) != 0
        {
            return false;
        }
        let ok = (|| {
            if (drv.cuMemcpyHtoD_v2)(da, ab.as_ptr().cast(), ab.len()) != 0
                || (drv.cuMemcpyHtoD_v2)(db, bb.as_ptr().cast(), bb.len()) != 0
            {
                return false;
            }
            let zero = vec![0u8; m * n * 4];
            if (drv.cuMemcpyHtoD_v2)(dc, zero.as_ptr().cast(), zero.len()) != 0 {
                return false;
            }
            let (gx, gy) = grid(&d, &cfg);
            let args = kernel_args(&d, da, db, dc, 1.0, 0.0, (0, 0, 0), 0);
            if launch(drv, kern, (gx, gy, 1), std::ptr::null_mut(), &args) != 0
                || (drv.cuStreamSynchronize)(std::ptr::null_mut()) != 0
            {
                return false;
            }
            let mut out = vec![0u8; m * n * 4];
            if (drv.cuMemcpyDtoH_v2)(out.as_mut_ptr().cast(), dc, out.len()) != 0 {
                return false;
            }
            out.chunks_exact(4)
                .zip(&want)
                .all(|(g, w)| (f32::from_le_bytes(g.try_into().unwrap()) - w).abs() <= 1e-4)
        })();
        (drv.cuMemFree_v2)(da);
        (drv.cuMemFree_v2)(db);
        (drv.cuMemFree_v2)(dc);
        ok
    }
}

/// Probe once per input type: the first operand-flag spelling whose module loads
/// and whose self-test GEMM is numerically exact. `None` = XMX unusable.
fn probed_flags(drv: &Driver, input: Input) -> Option<Flags> {
    static F16: OnceLock<Option<Flags>> = OnceLock::new();
    static BF16: OnceLock<Option<Flags>> = OnceLock::new();
    let cell = match input {
        Input::F16 => &F16,
        Input::Bf16 => &BF16,
    };
    *cell.get_or_init(|| {
        let started = std::time::Instant::now();
        let found = [Flags::Packed, Flags::Bare].into_iter().find(|&f| selftest(drv, input, f));
        xtrace!("probe {input:?}: {found:?} in {} ms", started.elapsed().as_millis());
        found
    })
}

/// Fast path entry. `Some(status)` = handled (status is the cuBLAS result before
/// the shim's stream synchronisation); `None` = not eligible or unsupported, use
/// the PTX kernels.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn try_gemm(
    drv: &Driver,
    stream: CUstream,
    d: &GemmDesc,
    alpha: f32,
    beta: f32,
    batch: c_int,
    batching: Batching,
    opts: &Opts,
    bias: u64,
) -> Option<c_int> {
    let mode = mode();
    if mode == Mode::Off {
        return None;
    }
    let (a, b, sa, sb, pointer_arrays) = match batching {
        Batching::Strided { a, b, sa, sb, .. } => (a, b, sa, sb, false),
        Batching::Pointers { .. } => (0, 0, 0, 0, true),
    };
    let input = match d.a {
        DType::F16 => Input::F16,
        DType::Bf16 => Input::Bf16,
        DType::F32 => return None,
    };
    let call = Call { d: *d, pointer_arrays, beta, opts: *opts, a, b, sa, sb };
    // Cheap shape test first so ineligible calls never trigger the probe.
    qualify(&call, mode, tile(), Flags::Packed)?;
    let flags = probed_flags(drv, input)?;
    let cfg = qualify(&call, mode, tile(), flags)?;
    let kern = kernel(drv, &cfg)?;
    xtrace!("{} m={} n={} k={} batch={batch}", cfg.name(), d.m, d.n, d.k);
    let (gx, gy) = grid(d, &cfg);
    for (start, count) in batch_chunks(batch as u32) {
        let (pa, pb, pc, sa_e, sb_e, sc_e) = match batching {
            Batching::Strided { a, b, c, sa, sb, sc } => {
                let first = start as i64;
                let off = |base: u64, stride: i64, size: usize| {
                    base.wrapping_add((first.wrapping_mul(stride) as u64).wrapping_mul(size as u64))
                };
                (
                    off(a, sa, d.a.size()),
                    off(b, sb, d.b.size()),
                    off(c, sc, d.c.size()),
                    sa as u64,
                    sb as u64,
                    sc as u64,
                )
            }
            Batching::Pointers { a, b, c } => {
                let off = start as u64 * 8;
                (a + off, b + off, c + off, 0, 0, 0)
            }
        };
        let args = kernel_args(d, pa, pb, pc, alpha, beta, (sa_e, sb_e, sc_e), bias);
        let err = unsafe { launch(drv, kern, (gx, gy, count), stream, &args) };
        if err != 0 {
            eprintln!("CUDA for Intel cublas: xmx launch failed err={err}");
            return Some(EXECUTION_FAILED);
        }
    }
    Some(0)
}
