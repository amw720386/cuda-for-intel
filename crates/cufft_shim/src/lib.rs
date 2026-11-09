//! cuFFT shim over cuda-for-intel's Driver API (`nvcuda.dll`).
//!
//! Same structure as the cuBLAS shim: the transforms run as generated PTX that our Driver
//! API translates to SPIR-V (no oneAPI dependency):
//! * `plan`   - pure plan logic: type decoding, basic and advanced layouts, the
//!   mixed-radix Stockham schedule, twiddle tables, work size, host model of the kernels.
//! * `ptxgen` - the gather / stage / scatter kernels, one module per precision.
//! * this file - the exported C ABI, plan handles, kernel cache and launcher.
//!
//! Implemented: plan creation (`cufftPlan{1d,2d,3d,Many}`, `cufftCreate`,
//! `cufftMakePlan{1d,2d,3d,Many,Many64}`, `cufftXtMakePlanMany`), size queries
//! (`cufftEstimate*`, `cufftGetSize*`, `cufftXtGetSizeMany`), work area control
//! (`cufftSetWorkArea`, `cufftSetAutoAllocation`), `cufftSetStream`, `cufftDestroy`,
//! execution (`cufftExec{C2C,R2C,C2R,Z2Z,D2Z,Z2D}`, `cufftXtExec`), plan properties and
//! version queries. Everything else (callbacks, multi-GPU `cufftXt*` descriptors, guru
//! plans) returns `CUFFT_NOT_SUPPORTED` from `stubs.rs`.
//!
//! Execution is asynchronous on the plan's stream, like cuFFT. Results are not normalized.
//! `CFI_CUFFT_TRACE=1` logs each plan and exec.
#![allow(non_snake_case)]
#![allow(clippy::missing_safety_doc)]

pub mod plan;
pub mod ptxgen;
mod stubs;

use libloading::Library;
use plan::*;
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_longlong, c_void};
use std::sync::{Mutex, OnceLock};

pub const CUFFT_SUCCESS: i32 = SUCCESS;
pub const CUFFT_INVALID_VALUE: i32 = INVALID_VALUE;
pub const CUFFT_NOT_SUPPORTED: i32 = NOT_SUPPORTED;

/// Reported cuFFT version (11.4.0 -> 11400).
pub const VERSION: i32 = 11400;

type CuResult = c_int;
type CUmodule = *mut c_void;
type CUfunction = *mut c_void;
type CUstream = *mut c_void;
type CUcontext = *mut c_void;

fn trace_enabled() -> bool {
    static T: OnceLock<bool> = OnceLock::new();
    *T.get_or_init(|| std::env::var("CFI_CUFFT_TRACE").is_ok_and(|v| !v.is_empty() && v != "0"))
}

macro_rules! trace {
    ($($arg:tt)*) => {
        if trace_enabled() {
            eprintln!("CUDA for Intel cufft: {}", format!($($arg)*));
        }
    };
}

// Load the driver with the DLL on Windows, as the cuBLAS shim does.
#[cfg(windows)]
#[link(name = "nvcuda", kind = "raw-dylib")]
unsafe extern "system" {
    fn cuCtxGetCurrent(context: *mut CUcontext) -> CuResult;
}

struct Driver {
    _lib: Library,
    cuCtxGetCurrent: unsafe extern "system" fn(*mut CUcontext) -> CuResult,
    cuInit: unsafe extern "system" fn(u32) -> CuResult,
    cuDeviceGet: unsafe extern "system" fn(*mut c_int, c_int) -> CuResult,
    cuDevicePrimaryCtxRetain: unsafe extern "system" fn(*mut CUcontext, c_int) -> CuResult,
    cuCtxSetCurrent: unsafe extern "system" fn(CUcontext) -> CuResult,
    cuCtxSynchronize: unsafe extern "system" fn() -> CuResult,
    cuModuleLoadData: unsafe extern "system" fn(*mut CUmodule, *const c_void) -> CuResult,
    cuModuleGetFunction: unsafe extern "system" fn(*mut CUfunction, CUmodule, *const c_char) -> CuResult,
    cuLaunchKernel: unsafe extern "system" fn(
        CUfunction, u32, u32, u32, u32, u32, u32, u32, CUstream, *mut *mut c_void, *mut *mut c_void,
    ) -> CuResult,
    cuMemAlloc_v2: unsafe extern "system" fn(*mut u64, usize) -> CuResult,
    cuMemFree_v2: unsafe extern "system" fn(u64) -> CuResult,
    cuMemcpyHtoD_v2: unsafe extern "system" fn(u64, *const c_void, usize) -> CuResult,
}

// Function pointers into a library that is never unloaded.
unsafe impl Send for Driver {}
unsafe impl Sync for Driver {}

fn driver() -> Result<&'static Driver, i32> {
    static DRIVER: OnceLock<Result<Driver, i32>> = OnceLock::new();
    DRIVER.get_or_init(load_driver).as_ref().map_err(|e| *e)
}

fn load_driver() -> Result<Driver, i32> {
    unsafe {
        #[cfg(windows)]
        let lib = Library::new("nvcuda.dll").map_err(|_| SETUP_FAILED)?;
        #[cfg(not(windows))]
        let lib = Library::new("libcuda.so.1").map_err(|_| SETUP_FAILED)?;
        macro_rules! sym {
            ($name:literal) => {
                *lib.get(concat!($name, "\0").as_bytes()).map_err(|_| SETUP_FAILED)?
            };
        }
        Ok(Driver {
            #[cfg(windows)]
            cuCtxGetCurrent,
            #[cfg(not(windows))]
            cuCtxGetCurrent: sym!("cuCtxGetCurrent"),
            cuInit: sym!("cuInit"),
            cuDeviceGet: sym!("cuDeviceGet"),
            cuDevicePrimaryCtxRetain: sym!("cuDevicePrimaryCtxRetain"),
            cuCtxSetCurrent: sym!("cuCtxSetCurrent"),
            cuCtxSynchronize: sym!("cuCtxSynchronize"),
            cuModuleLoadData: sym!("cuModuleLoadData"),
            cuModuleGetFunction: sym!("cuModuleGetFunction"),
            cuLaunchKernel: sym!("cuLaunchKernel"),
            cuMemAlloc_v2: sym!("cuMemAlloc_v2"),
            cuMemFree_v2: sym!("cuMemFree_v2"),
            cuMemcpyHtoD_v2: sym!("cuMemcpyHtoD_v2"),
            _lib: lib,
        })
    }
}

/// The current context, or device 0's primary context made current (cuFFT, being a
/// runtime-API library, works on the primary context when none is current).
fn context(drv: &Driver) -> Result<usize, i32> {
    let mut ctx: CUcontext = std::ptr::null_mut();
    if unsafe { (drv.cuCtxGetCurrent)(&mut ctx) } == 0 && !ctx.is_null() {
        return Ok(ctx as usize);
    }
    let mut dev = 0;
    unsafe {
        if (drv.cuInit)(0) != 0 || (drv.cuDeviceGet)(&mut dev, 0) != 0 {
            return Err(SETUP_FAILED);
        }
        if (drv.cuDevicePrimaryCtxRetain)(&mut ctx, dev) != 0 || (drv.cuCtxSetCurrent)(ctx) != 0 {
            return Err(SETUP_FAILED);
        }
    }
    Ok(ctx as usize)
}

// ---------------------------------------------------------------------------
// Kernels: one module per (context, precision).
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Kernels {
    gather: usize,
    stage: usize,
    scatter: usize,
}

fn kernels(drv: &Driver, ctx: usize, double: bool) -> Result<Kernels, i32> {
    static CACHE: OnceLock<Mutex<HashMap<(usize, bool), Kernels>>> = OnceLock::new();
    let mut cache = CACHE.get_or_init(Default::default).lock().unwrap();
    if let Some(k) = cache.get(&(ctx, double)) {
        return Ok(*k);
    }
    let src = format!("{}\0", ptxgen::module(double));
    let mut module: CUmodule = std::ptr::null_mut();
    let err = unsafe { (drv.cuModuleLoadData)(&mut module, src.as_ptr().cast()) };
    if err != 0 {
        eprintln!("CUDA for Intel cufft: loading the {} FFT kernels failed (err={err})", ptxgen::suffix(double));
        return Err(if double { NOT_SUPPORTED } else { INTERNAL_ERROR });
    }
    let get = |name: String| -> Result<usize, i32> {
        let c = std::ffi::CString::new(name).unwrap();
        let mut f: CUfunction = std::ptr::null_mut();
        let err = unsafe { (drv.cuModuleGetFunction)(&mut f, module, c.as_ptr()) };
        if err != 0 || f.is_null() { Err(INTERNAL_ERROR) } else { Ok(f as usize) }
    };
    let k = Kernels { gather: get(ptxgen::gather_name(double))?, stage: get(ptxgen::stage_name(double))?, scatter: get(ptxgen::scatter_name(double))? };
    cache.insert((ctx, double), k);
    Ok(k)
}

const BLOCK: u32 = 256;

fn launch(drv: &Driver, f: usize, total: u64, stream: usize, args: &[u64]) -> Result<(), i32> {
    // Every parameter slot is 8 bytes; 32-bit parameters read the low half (little endian).
    let mut slots: Vec<u64> = args.to_vec();
    let mut ptrs: Vec<*mut c_void> = slots.iter_mut().map(|v| (v as *mut u64).cast()).collect();
    let grid = total.div_ceil(BLOCK as u64) as u32;
    let err = unsafe {
        (drv.cuLaunchKernel)(f as CUfunction, grid, 1, 1, BLOCK, 1, 1, 0, stream as CUstream, ptrs.as_mut_ptr(), std::ptr::null_mut())
    };
    if err != 0 {
        eprintln!("CUDA for Intel cufft: launch failed (err={err})");
        return Err(EXEC_FAILED);
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Plans
// ---------------------------------------------------------------------------

struct Plan {
    desc: Option<Desc>,
    stream: usize,
    auto_alloc: bool,
    user_work: u64,
    own_work: u64,
    twiddles: u64,
    ctx: usize,
    props: HashMap<i32, i64>,
}

impl Plan {
    fn new() -> Self {
        Plan { desc: None, stream: 0, auto_alloc: true, user_work: 0, own_work: 0, twiddles: 0, ctx: 0, props: HashMap::new() }
    }

    fn release(&mut self) {
        if self.own_work == 0 && self.twiddles == 0 {
            return;
        }
        // torch destroys its cached plans from an exit handler; the driver must not be entered
        // then (its thread-local state is gone). The process is ending, so nothing leaks.
        if shutting_down() {
            self.own_work = 0;
            self.twiddles = 0;
            return;
        }
        if let Ok(drv) = driver() {
            unsafe {
                // Earlier executions may still read the twiddles and work area.
                let _ = (drv.cuCtxSynchronize)();
                for p in [self.own_work, self.twiddles] {
                    if p != 0 {
                        let _ = (drv.cuMemFree_v2)(p);
                    }
                }
            }
        }
        self.own_work = 0;
        self.twiddles = 0;
    }
}

#[cfg(windows)]
#[link(name = "ntdll", kind = "raw-dylib")]
unsafe extern "system" {
    fn RtlDllShutdownInProgress() -> u8;
}

fn shutting_down() -> bool {
    #[cfg(windows)]
    return unsafe { RtlDllShutdownInProgress() } != 0;
    #[cfg(not(windows))]
    false
}

struct Table {
    next: i32,
    plans: HashMap<i32, Plan>,
}

fn table() -> std::sync::MutexGuard<'static, Table> {
    static T: OnceLock<Mutex<Table>> = OnceLock::new();
    T.get_or_init(|| Mutex::new(Table { next: 1, plans: HashMap::new() })).lock().unwrap_or_else(|e| e.into_inner())
}

fn with_plan<R>(h: c_int, f: impl FnOnce(&mut Plan) -> Result<R, i32>) -> Result<R, i32> {
    match table().plans.get_mut(&h) {
        Some(p) => f(p),
        None => Err(INVALID_PLAN),
    }
}

fn status(r: Result<(), i32>) -> c_int {
    match r {
        Ok(()) => SUCCESS,
        Err(e) => e,
    }
}

fn new_handle() -> c_int {
    let mut t = table();
    let h = t.next;
    t.next += 1;
    t.plans.insert(h, Plan::new());
    h
}

unsafe fn write_size(out: *mut usize, v: u64) {
    if !out.is_null() {
        unsafe { out.write(v as usize) };
    }
}

unsafe fn dims<T: Copy + Into<i64>>(p: *const T, rank: i64) -> Option<Vec<i64>> {
    if p.is_null() || !(1..=3).contains(&rank) {
        return None;
    }
    Some((0..rank as usize).map(|i| unsafe { *p.add(i) }.into()).collect())
}

/// Arguments of a PlanMany-style call, already widened to 64 bits.
struct Many {
    rank: i64,
    n: Option<Vec<i64>>,
    inembed: Option<Vec<i64>>,
    istride: i64,
    idist: i64,
    onembed: Option<Vec<i64>>,
    ostride: i64,
    odist: i64,
    batch: i64,
}

impl Many {
    fn simple(n: &[i64], batch: i64) -> Many {
        Many { rank: n.len() as i64, n: Some(n.to_vec()), inembed: None, istride: 1, idist: 0, onembed: None, ostride: 1, odist: 0, batch }
    }

    fn describe(&self, ty: Result<FftType, i32>) -> Result<Desc, i32> {
        let ty = ty?;
        let n = self.n.as_ref().ok_or(INVALID_VALUE)?;
        describe(self.rank, n, self.inembed.as_deref(), self.istride, self.idist, self.onembed.as_deref(), self.ostride, self.odist, ty, self.batch)
    }
}

fn cufft_type(t: c_int) -> Result<FftType, i32> {
    FftType::from_cufft(t).ok_or(INVALID_TYPE)
}

fn make_plan(h: c_int, m: &Many, ty: Result<FftType, i32>, work: *mut usize) -> c_int {
    let r = (|| {
        let desc = m.describe(ty)?;
        let drv = driver()?;
        let ctx = context(drv)?;
        // Compile the kernels now so a missing fp64 capability fails at plan time.
        kernels(drv, ctx, desc.ty.double)?;
        with_plan(h, |p| {
            p.release();
            let tw = desc.twiddle_bytes();
            let mut dev = 0u64;
            unsafe {
                if (drv.cuMemAlloc_v2)(&mut dev, tw.len().max(16)) != 0 {
                    return Err(ALLOC_FAILED);
                }
                p.twiddles = dev;
                if !tw.is_empty() && (drv.cuMemcpyHtoD_v2)(dev, tw.as_ptr().cast(), tw.len()) != 0 {
                    return Err(INTERNAL_ERROR);
                }
                if p.auto_alloc {
                    if (drv.cuMemAlloc_v2)(&mut dev, desc.work_size() as usize) != 0 {
                        return Err(ALLOC_FAILED);
                    }
                    p.own_work = dev;
                }
                write_size(work, desc.work_size());
            }
            trace!("plan {h}: {:?} n={:?} batch={} in={:?} out={:?} stages={} work={}", desc.ty, desc.n, desc.batch, desc.input, desc.output, desc.stages.len(), desc.work_size());
            p.ctx = ctx;
            p.desc = Some(desc);
            Ok(())
        })
    })();
    status(r)
}

fn estimate(m: &Many, ty: Result<FftType, i32>, work: *mut usize) -> c_int {
    status(m.describe(ty).map(|d| unsafe { write_size(work, d.work_size()) }))
}

fn plan_new(out: *mut c_int, m: &Many, ty: c_int) -> c_int {
    if out.is_null() {
        return INVALID_VALUE;
    }
    let h = new_handle();
    let st = make_plan(h, m, cufft_type(ty), std::ptr::null_mut());
    if st != SUCCESS {
        if let Some(mut p) = table().plans.remove(&h) {
            p.release();
        }
        return st;
    }
    unsafe { out.write(h) };
    SUCCESS
}

// ---------------------------------------------------------------------------
// Exported: handles and plans
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftCreate(plan: *mut c_int) -> c_int {
    if plan.is_null() {
        return INVALID_VALUE;
    }
    unsafe { plan.write(new_handle()) };
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftDestroy(plan: c_int) -> c_int {
    match table().plans.remove(&plan) {
        Some(mut p) => {
            p.release();
            SUCCESS
        }
        None => INVALID_PLAN,
    }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftPlan1d(plan: *mut c_int, nx: c_int, ty: c_int, batch: c_int) -> c_int {
    plan_new(plan, &Many::simple(&[nx.into()], batch.into()), ty)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftPlan2d(plan: *mut c_int, nx: c_int, ny: c_int, ty: c_int) -> c_int {
    plan_new(plan, &Many::simple(&[nx.into(), ny.into()], 1), ty)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftPlan3d(plan: *mut c_int, nx: c_int, ny: c_int, nz: c_int, ty: c_int) -> c_int {
    plan_new(plan, &Many::simple(&[nx.into(), ny.into(), nz.into()], 1), ty)
}

#[allow(clippy::too_many_arguments)]
unsafe fn many32(
    rank: c_int, n: *const c_int, inembed: *const c_int, istride: c_int, idist: c_int, onembed: *const c_int,
    ostride: c_int, odist: c_int, batch: c_int,
) -> Many {
    let r = rank as i64;
    unsafe {
        Many {
            rank: r, n: dims(n, r), inembed: dims(inembed, r), istride: istride.into(), idist: idist.into(),
            onembed: dims(onembed, r), ostride: ostride.into(), odist: odist.into(), batch: batch.into(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
unsafe fn many64(
    rank: c_int, n: *const c_longlong, inembed: *const c_longlong, istride: c_longlong, idist: c_longlong,
    onembed: *const c_longlong, ostride: c_longlong, odist: c_longlong, batch: c_longlong,
) -> Many {
    let r = rank as i64;
    unsafe { Many { rank: r, n: dims(n, r), inembed: dims(inembed, r), istride, idist, onembed: dims(onembed, r), ostride, odist, batch } }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftPlanMany(
    plan: *mut c_int, rank: c_int, n: *const c_int, inembed: *const c_int, istride: c_int, idist: c_int,
    onembed: *const c_int, ostride: c_int, odist: c_int, ty: c_int, batch: c_int,
) -> c_int {
    let m = unsafe { many32(rank, n, inembed, istride, idist, onembed, ostride, odist, batch) };
    plan_new(plan, &m, ty)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftMakePlan1d(plan: c_int, nx: c_int, ty: c_int, batch: c_int, work: *mut usize) -> c_int {
    make_plan(plan, &Many::simple(&[nx.into()], batch.into()), cufft_type(ty), work)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftMakePlan2d(plan: c_int, nx: c_int, ny: c_int, ty: c_int, work: *mut usize) -> c_int {
    make_plan(plan, &Many::simple(&[nx.into(), ny.into()], 1), cufft_type(ty), work)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftMakePlan3d(plan: c_int, nx: c_int, ny: c_int, nz: c_int, ty: c_int, work: *mut usize) -> c_int {
    make_plan(plan, &Many::simple(&[nx.into(), ny.into(), nz.into()], 1), cufft_type(ty), work)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftMakePlanMany(
    plan: c_int, rank: c_int, n: *const c_int, inembed: *const c_int, istride: c_int, idist: c_int,
    onembed: *const c_int, ostride: c_int, odist: c_int, ty: c_int, batch: c_int, work: *mut usize,
) -> c_int {
    let m = unsafe { many32(rank, n, inembed, istride, idist, onembed, ostride, odist, batch) };
    make_plan(plan, &m, cufft_type(ty), work)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftMakePlanMany64(
    plan: c_int, rank: c_int, n: *const c_longlong, inembed: *const c_longlong, istride: c_longlong,
    idist: c_longlong, onembed: *const c_longlong, ostride: c_longlong, odist: c_longlong, ty: c_int,
    batch: c_longlong, work: *mut usize,
) -> c_int {
    let m = unsafe { many64(rank, n, inembed, istride, idist, onembed, ostride, odist, batch) };
    make_plan(plan, &m, cufft_type(ty), work)
}

/// `cufftXtMakePlanMany`: the entry point torch 2.x plans every FFT with.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftXtMakePlanMany(
    plan: c_int, rank: c_int, n: *const c_longlong, inembed: *const c_longlong, istride: c_longlong,
    idist: c_longlong, itype: c_int, onembed: *const c_longlong, ostride: c_longlong, odist: c_longlong,
    otype: c_int, batch: c_longlong, work: *mut usize, etype: c_int,
) -> c_int {
    let m = unsafe { many64(rank, n, inembed, istride, idist, onembed, ostride, odist, batch) };
    make_plan(plan, &m, FftType::from_xt(itype, otype, etype), work)
}

// ---------------------------------------------------------------------------
// Exported: size queries and work area
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftEstimate1d(nx: c_int, ty: c_int, batch: c_int, work: *mut usize) -> c_int {
    estimate(&Many::simple(&[nx.into()], batch.into()), cufft_type(ty), work)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftEstimate2d(nx: c_int, ny: c_int, ty: c_int, work: *mut usize) -> c_int {
    estimate(&Many::simple(&[nx.into(), ny.into()], 1), cufft_type(ty), work)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftEstimate3d(nx: c_int, ny: c_int, nz: c_int, ty: c_int, work: *mut usize) -> c_int {
    estimate(&Many::simple(&[nx.into(), ny.into(), nz.into()], 1), cufft_type(ty), work)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftEstimateMany(
    rank: c_int, n: *const c_int, inembed: *const c_int, istride: c_int, idist: c_int, onembed: *const c_int,
    ostride: c_int, odist: c_int, ty: c_int, batch: c_int, work: *mut usize,
) -> c_int {
    let m = unsafe { many32(rank, n, inembed, istride, idist, onembed, ostride, odist, batch) };
    estimate(&m, cufft_type(ty), work)
}

fn plan_exists(h: c_int) -> bool {
    table().plans.contains_key(&h)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetSize1d(plan: c_int, nx: c_int, ty: c_int, batch: c_int, work: *mut usize) -> c_int {
    if !plan_exists(plan) {
        return INVALID_PLAN;
    }
    unsafe { cufftEstimate1d(nx, ty, batch, work) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetSize2d(plan: c_int, nx: c_int, ny: c_int, ty: c_int, work: *mut usize) -> c_int {
    if !plan_exists(plan) {
        return INVALID_PLAN;
    }
    unsafe { cufftEstimate2d(nx, ny, ty, work) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetSize3d(plan: c_int, nx: c_int, ny: c_int, nz: c_int, ty: c_int, work: *mut usize) -> c_int {
    if !plan_exists(plan) {
        return INVALID_PLAN;
    }
    unsafe { cufftEstimate3d(nx, ny, nz, ty, work) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetSizeMany(
    plan: c_int, rank: c_int, n: *const c_int, inembed: *const c_int, istride: c_int, idist: c_int,
    onembed: *const c_int, ostride: c_int, odist: c_int, ty: c_int, batch: c_int, work: *mut usize,
) -> c_int {
    if !plan_exists(plan) {
        return INVALID_PLAN;
    }
    unsafe { cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, ty, batch, work) }
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetSizeMany64(
    plan: c_int, rank: c_int, n: *const c_longlong, inembed: *const c_longlong, istride: c_longlong,
    idist: c_longlong, onembed: *const c_longlong, ostride: c_longlong, odist: c_longlong, ty: c_int,
    batch: c_longlong, work: *mut usize,
) -> c_int {
    if !plan_exists(plan) {
        return INVALID_PLAN;
    }
    let m = unsafe { many64(rank, n, inembed, istride, idist, onembed, ostride, odist, batch) };
    estimate(&m, cufft_type(ty), work)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftXtGetSizeMany(
    plan: c_int, rank: c_int, n: *const c_longlong, inembed: *const c_longlong, istride: c_longlong,
    idist: c_longlong, itype: c_int, onembed: *const c_longlong, ostride: c_longlong, odist: c_longlong,
    otype: c_int, batch: c_longlong, work: *mut usize, etype: c_int,
) -> c_int {
    if !plan_exists(plan) {
        return INVALID_PLAN;
    }
    let m = unsafe { many64(rank, n, inembed, istride, idist, onembed, ostride, odist, batch) };
    estimate(&m, FftType::from_xt(itype, otype, etype), work)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetSize(plan: c_int, work: *mut usize) -> c_int {
    status(with_plan(plan, |p| {
        let size = p.desc.as_ref().map_or(0, |d| d.work_size());
        unsafe { write_size(work, size) };
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftSetWorkArea(plan: c_int, area: *mut c_void) -> c_int {
    status(with_plan(plan, |p| {
        p.user_work = area as u64;
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftSetAutoAllocation(plan: c_int, auto: c_int) -> c_int {
    status(with_plan(plan, |p| {
        p.auto_alloc = auto != 0;
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftSetStream(plan: c_int, stream: *mut c_void) -> c_int {
    status(with_plan(plan, |p| {
        p.stream = stream as usize;
        Ok(())
    }))
}

/// `cufftProperty`: NVFFT_PLAN_PROPERTY_INT64_PATIENT_JIT (1) and
/// NVFFT_PLAN_PROPERTY_INT64_MAX_NUM_HOST_THREADS (2). Stored, no effect on execution.
fn known_property(prop: c_int) -> bool {
    matches!(prop, 1 | 2)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftSetPlanPropertyInt64(plan: c_int, prop: c_int, value: c_longlong) -> c_int {
    if !known_property(prop) {
        return INVALID_VALUE;
    }
    status(with_plan(plan, |p| {
        p.props.insert(prop, value);
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetPlanPropertyInt64(plan: c_int, prop: c_int, value: *mut c_longlong) -> c_int {
    if !known_property(prop) || value.is_null() {
        return INVALID_VALUE;
    }
    status(with_plan(plan, |p| {
        unsafe { value.write(p.props.get(&prop).copied().unwrap_or(0)) };
        Ok(())
    }))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftResetPlanProperty(plan: c_int, prop: c_int) -> c_int {
    if !known_property(prop) {
        return INVALID_VALUE;
    }
    status(with_plan(plan, |p| {
        p.props.remove(&prop);
        Ok(())
    }))
}

// ---------------------------------------------------------------------------
// Exported: execution
// ---------------------------------------------------------------------------

const FORWARD: c_int = -1;
const INVERSE: c_int = 1;

/// `want` is the (kind, precision) the exec function implies; `None` accepts the plan's
/// (cufftXtExec).
fn exec(h: c_int, want: Option<FftType>, idata: *mut c_void, odata: *mut c_void, direction: c_int) -> c_int {
    let r = (|| {
        let (desc, stream, work, tw) = with_plan(h, |p| {
            let d = p.desc.clone().ok_or(INVALID_PLAN)?;
            let work = if p.own_work != 0 { p.own_work } else { p.user_work };
            Ok((d, p.stream, work, p.twiddles))
        })?;
        if want.is_some_and(|w| w != desc.ty) {
            return Err(INVALID_TYPE);
        }
        let inverse = match desc.ty.kind {
            Kind::R2C => false,
            Kind::C2R => true,
            Kind::C2C if direction == FORWARD => false,
            Kind::C2C if direction == INVERSE => true,
            Kind::C2C => return Err(INVALID_VALUE),
        };
        if idata.is_null() || odata.is_null() {
            return Err(INVALID_VALUE);
        }
        if work == 0 {
            return Err(INVALID_VALUE);
        }
        let drv = driver()?;
        let ctx = context(drv)?;
        let k = kernels(drv, ctx, desc.ty.double)?;
        let in_place = idata == odata;
        let (is, os) = desc.exec_strides(in_place);
        trace!("exec plan {h}: {:?} inverse={inverse} in_place={in_place} stream={stream:#x}", desc.ty);
        let half = desc.total() * desc.ty.complex_size();
        launch(drv, k.gather, desc.total(), stream, &desc.gather_args(idata as u64, work, &is))?;
        let (mut a, mut b) = (work, work + half);
        for s in &desc.stages {
            launch(drv, k.stage, desc.total(), stream, &desc.stage_args(s, a, b, tw, inverse))?;
            std::mem::swap(&mut a, &mut b);
        }
        launch(drv, k.scatter, desc.out_total(), stream, &desc.scatter_args(a, odata as u64, &os))
    })();
    status(r)
}

fn ty_of(kind: Kind, double: bool) -> Option<FftType> {
    Some(FftType { kind, double })
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftExecC2C(plan: c_int, idata: *mut c_void, odata: *mut c_void, direction: c_int) -> c_int {
    exec(plan, ty_of(Kind::C2C, false), idata, odata, direction)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftExecR2C(plan: c_int, idata: *mut c_void, odata: *mut c_void) -> c_int {
    exec(plan, ty_of(Kind::R2C, false), idata, odata, FORWARD)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftExecC2R(plan: c_int, idata: *mut c_void, odata: *mut c_void) -> c_int {
    exec(plan, ty_of(Kind::C2R, false), idata, odata, INVERSE)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftExecZ2Z(plan: c_int, idata: *mut c_void, odata: *mut c_void, direction: c_int) -> c_int {
    exec(plan, ty_of(Kind::C2C, true), idata, odata, direction)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftExecD2Z(plan: c_int, idata: *mut c_void, odata: *mut c_void) -> c_int {
    exec(plan, ty_of(Kind::R2C, true), idata, odata, FORWARD)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftExecZ2D(plan: c_int, idata: *mut c_void, odata: *mut c_void) -> c_int {
    exec(plan, ty_of(Kind::C2R, true), idata, odata, INVERSE)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftXtExec(plan: c_int, input: *mut c_void, output: *mut c_void, direction: c_int) -> c_int {
    exec(plan, None, input, output, direction)
}

// ---------------------------------------------------------------------------
// Exported: version
// ---------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetVersion(version: *mut c_int) -> c_int {
    if version.is_null() {
        return INVALID_VALUE;
    }
    unsafe { version.write(VERSION) };
    SUCCESS
}

/// `libraryPropertyType`: MAJOR_VERSION 0, MINOR_VERSION 1, PATCH_LEVEL 2.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cufftGetProperty(kind: c_int, value: *mut c_int) -> c_int {
    if value.is_null() {
        return INVALID_VALUE;
    }
    let field = match kind {
        0 => VERSION / 1000,
        1 => (VERSION / 100) % 10,
        2 => VERSION % 100,
        _ => return INVALID_VALUE,
    };
    unsafe { value.write(field) };
    SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};

    #[test]
    fn version_queries() {
        let mut v = 0;
        assert_eq!(unsafe { cufftGetVersion(&mut v) }, 0);
        assert_eq!(v, 11400);
        assert_eq!(unsafe { cufftGetVersion(null_mut()) }, CUFFT_INVALID_VALUE);
        let (mut major, mut minor, mut patch) = (0, 0, 0);
        assert_eq!(unsafe { cufftGetProperty(0, &mut major) }, 0);
        assert_eq!(unsafe { cufftGetProperty(1, &mut minor) }, 0);
        assert_eq!(unsafe { cufftGetProperty(2, &mut patch) }, 0);
        assert_eq!((major, minor, patch), (11, 4, 0));
        assert_eq!(unsafe { cufftGetProperty(9, &mut patch) }, CUFFT_INVALID_VALUE);
    }

    /// Everything that needs no device: handles, estimates, work-area settings, properties,
    /// and the errors exec reports before touching the driver.
    #[test]
    fn handles_estimates_and_errors_without_a_device() {
        let mut h = 0;
        assert_eq!(unsafe { cufftCreate(&mut h) }, 0);
        assert!(h > 0);
        let mut ws = 0usize;
        assert_eq!(unsafe { cufftEstimate1d(1024, 0x29, 4, &mut ws) }, 0);
        assert_eq!(ws, 2 * 4096 * 8);
        assert_eq!(unsafe { cufftEstimate2d(16, 16, 0x6a, &mut ws) }, 0);
        assert_eq!(ws, 2 * 256 * 16);
        assert_eq!(unsafe { cufftEstimate3d(4, 4, 4, 0x2c, &mut ws) }, 0);
        assert_eq!(ws, 2 * 64 * 8);
        assert_eq!(unsafe { cufftEstimate1d(0, 0x29, 1, &mut ws) }, INVALID_SIZE);
        assert_eq!(unsafe { cufftEstimate1d(8, 0x2b, 1, &mut ws) }, INVALID_TYPE);
        let n = [12i32, 10];
        assert_eq!(unsafe { cufftEstimateMany(2, n.as_ptr(), null(), 1, 0, null(), 1, 0, 0x29, 3, &mut ws) }, 0);
        assert_eq!(ws, 2 * 360 * 8);
        assert_eq!(unsafe { cufftGetSize1d(h, 100, 0x69, 2, &mut ws) }, 0);
        assert_eq!(ws, 2 * 200 * 16);
        let n64 = [64i64];
        assert_eq!(unsafe { cufftXtGetSizeMany(h, 1, n64.as_ptr(), n64.as_ptr(), 1, 64, 0, n64.as_ptr(), 1, 33, 4, 5, &mut ws, 4) }, 0);
        assert_eq!(ws, 2 * 320 * 8);
        assert_eq!(unsafe { cufftXtGetSizeMany(h, 1, n64.as_ptr(), null(), 1, 64, 6, null(), 1, 33, 6, 5, &mut ws, 6) }, NOT_SUPPORTED);
        assert_eq!(unsafe { cufftGetSize(h, &mut ws) }, 0);
        assert_eq!(ws, 0, "an unmade plan needs no work area");
        assert_eq!(unsafe { cufftSetAutoAllocation(h, 0) }, 0);
        assert_eq!(unsafe { cufftSetWorkArea(h, 0x1000 as *mut c_void) }, 0);
        assert_eq!(unsafe { cufftSetStream(h, null_mut()) }, 0);
        assert_eq!(unsafe { cufftSetPlanPropertyInt64(h, 1, 7) }, 0);
        let mut v = 0i64;
        assert_eq!(unsafe { cufftGetPlanPropertyInt64(h, 1, &mut v) }, 0);
        assert_eq!(v, 7);
        assert_eq!(unsafe { cufftResetPlanProperty(h, 1) }, 0);
        assert_eq!(unsafe { cufftGetPlanPropertyInt64(h, 1, &mut v) }, 0);
        assert_eq!(v, 0);
        assert_eq!(unsafe { cufftSetPlanPropertyInt64(h, 99, 1) }, INVALID_VALUE);
        let buf = 0x1000 as *mut c_void;
        assert_eq!(unsafe { cufftExecC2C(h, buf, buf, FORWARD) }, INVALID_PLAN, "exec before MakePlan");
        assert_eq!(unsafe { cufftDestroy(h) }, 0);
        assert_eq!(unsafe { cufftDestroy(h) }, INVALID_PLAN);
        assert_eq!(unsafe { cufftSetStream(h, null_mut()) }, INVALID_PLAN);
        assert_eq!(unsafe { cufftGetSize1d(h, 8, 0x29, 1, &mut ws) }, INVALID_PLAN);
        assert_eq!(unsafe { cufftCreate(null_mut()) }, INVALID_VALUE);
        assert_eq!(unsafe { cufftPlan1d(null_mut(), 8, 0x29, 1) }, INVALID_VALUE);
    }

    #[test]
    fn exported_surface_covers_zluda_and_torch() {
        // ZLUDA's 32 implemented cuFFT entry points plus torch 2.x's cufftXt pair.
        let src = include_str!("lib.rs");
        let want = [
            "cufftCreate", "cufftDestroy", "cufftEstimate1d", "cufftEstimate2d", "cufftEstimate3d", "cufftEstimateMany",
            "cufftExecC2C", "cufftExecC2R", "cufftExecD2Z", "cufftExecR2C", "cufftExecZ2D", "cufftExecZ2Z",
            "cufftGetProperty", "cufftGetSize", "cufftGetSize1d", "cufftGetSize2d", "cufftGetSize3d", "cufftGetSizeMany",
            "cufftGetSizeMany64", "cufftGetVersion", "cufftMakePlan1d", "cufftMakePlan2d", "cufftMakePlan3d",
            "cufftMakePlanMany", "cufftMakePlanMany64", "cufftPlan1d", "cufftPlan2d", "cufftPlan3d", "cufftPlanMany",
            "cufftSetAutoAllocation", "cufftSetStream", "cufftSetWorkArea", "cufftXtMakePlanMany", "cufftXtExec",
        ];
        for name in want {
            assert!(src.contains(&format!("extern \"system\" fn {name}(")), "{name}");
        }
    }
}
