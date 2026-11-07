//! cublasLt exports (`cublasLt64_12`): descriptor bookkeeping plus execution of
//! `cublasLtMatmul` on the shim's tiled PTX GEMM with a fused bias/activation
//! epilogue. Pure logic (attributes, layout-to-GEMM translation, CPU
//! references) is in `ltplan`; this file is the C ABI.
//!
//! Struct layouts follow `cublasLt.h`: `cublasLtMatmulAlgo_t` is `uint64_t[8]`
//! and `cublasLtMatmulHeuristicResult_t` is
//! `{ algo; size_t workspaceSize; cublasStatus_t state; float wavesCount; int reserved[4]; }`
//! (96 bytes). Handles, descriptors, layouts and preferences are opaque heap
//! objects carrying a magic number so stale or foreign pointers are rejected.
//!
//! Execution model: the heuristic validates the whole problem and returns
//! exactly one algorithm (id [`ALGO_MAGIC`], workspace 0). `cublasLtMatmul`
//! re-plans from the descriptors, so a NULL algo also works. When the real
//! library would report "no algo" (type/epilogue/layout combos this shim
//! cannot run) the heuristic returns SUCCESS with `returnAlgoCount = 0`.

use crate::ltplan::*;
use crate::plan::*;
use crate::{Batching, Handle, driver, run_gemm_ex};
use std::ffi::{c_int, c_void};

const MAGIC_HANDLE: u32 = 0x4c54_4831;
const MAGIC_DESC: u32 = 0x4c54_4432;
const MAGIC_LAYOUT: u32 = 0x4c54_4c33;
const MAGIC_PREF: u32 = 0x4c54_5034;

/// First word of the single algorithm this shim hands out.
pub const ALGO_MAGIC: u64 = 0x4346_4900_4c54_0001;

/// cublasLt version reported (major*10000 + minor*100 + patch): 12.9.0.
const LT_VERSION: usize = 120900;
const CUDART_VERSION: usize = 12090;

#[repr(C)]
struct Obj {
    magic: u32,
    bag: AttrBag,
}

#[repr(C)]
pub struct LtHandle {
    magic: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct MatmulAlgo {
    pub data: [u64; 8],
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct HeuristicResult {
    pub algo: MatmulAlgo,
    pub workspace_size: usize,
    pub state: c_int,
    pub waves_count: f32,
    pub reserved: [c_int; 4],
}

fn obj_new(out: *mut *mut c_void, magic: u32, bag: AttrBag) -> c_int {
    if out.is_null() {
        return INVALID_VALUE;
    }
    unsafe { out.write(Box::into_raw(Box::new(Obj { magic, bag })).cast()) };
    SUCCESS
}

fn obj_ref<'a>(p: *const c_void, magic: u32) -> Option<&'a Obj> {
    if p.is_null() {
        return None;
    }
    let o = unsafe { &*(p as *const Obj) };
    (o.magic == magic).then_some(o)
}

fn obj_mut<'a>(p: *mut c_void, magic: u32) -> Option<&'a mut Obj> {
    if p.is_null() {
        return None;
    }
    let o = unsafe { &mut *(p as *mut Obj) };
    (o.magic == magic).then_some(o)
}

fn obj_free(p: *mut c_void, magic: u32) -> c_int {
    if p.is_null() {
        return INVALID_VALUE;
    }
    if unsafe { (*(p as *const Obj)).magic } != magic {
        return INVALID_VALUE;
    }
    unsafe {
        (*(p as *mut Obj)).magic = 0;
        drop(Box::from_raw(p as *mut Obj));
    }
    SUCCESS
}

fn set_attr(p: *mut c_void, magic: u32, id: c_int, buf: *const c_void, size: usize) -> c_int {
    let Some(o) = obj_mut(p, magic) else { return INVALID_VALUE };
    if buf.is_null() {
        return INVALID_VALUE;
    }
    let bytes = unsafe { std::slice::from_raw_parts(buf.cast::<u8>(), size) };
    o.bag.set(id, bytes)
}

fn get_attr(
    p: *const c_void,
    magic: u32,
    id: c_int,
    buf: *mut c_void,
    size: usize,
    written: *mut usize,
) -> c_int {
    let Some(o) = obj_ref(p, magic) else { return INVALID_VALUE };
    if buf.is_null() && size != 0 {
        return INVALID_VALUE;
    }
    let out: &mut [u8] = if size == 0 {
        &mut []
    } else {
        unsafe { std::slice::from_raw_parts_mut(buf.cast::<u8>(), size) }
    };
    let mut n = 0usize;
    let st = o.bag.get(id, out, &mut n);
    if !written.is_null() {
        unsafe { written.write(n) };
    }
    st
}

// ---- handle -----------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtCreate(handle: *mut *mut c_void) -> c_int {
    if handle.is_null() {
        return INVALID_VALUE;
    }
    unsafe { handle.write(Box::into_raw(Box::new(LtHandle { magic: MAGIC_HANDLE })).cast()) };
    SUCCESS
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtDestroy(handle: *mut c_void) -> c_int {
    if handle.is_null() || unsafe { (*(handle as *const LtHandle)).magic } != MAGIC_HANDLE {
        return INVALID_VALUE;
    }
    unsafe { drop(Box::from_raw(handle as *mut LtHandle)) };
    SUCCESS
}

#[unsafe(no_mangle)]
pub extern "system" fn cublasLtGetVersion() -> usize {
    LT_VERSION
}

#[unsafe(no_mangle)]
pub extern "system" fn cublasLtGetCudartVersion() -> usize {
    CUDART_VERSION
}

/// `libraryPropertyType`: MAJOR_VERSION 0, MINOR_VERSION 1, PATCH_LEVEL 2.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtGetProperty(prop: c_int, value: *mut c_int) -> c_int {
    if value.is_null() {
        return INVALID_VALUE;
    }
    let v = match prop {
        0 => (LT_VERSION / 10000) as c_int,
        1 => (LT_VERSION / 100 % 100) as c_int,
        2 => (LT_VERSION % 100) as c_int,
        _ => return INVALID_VALUE,
    };
    unsafe { value.write(v) };
    SUCCESS
}

// ---- matmul descriptor ----------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatmulDescCreate(
    desc: *mut *mut c_void,
    compute_type: c_int,
    scale_type: c_int,
) -> c_int {
    obj_new(desc, MAGIC_DESC, new_desc(compute_type, scale_type))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatmulDescDestroy(desc: *mut c_void) -> c_int {
    obj_free(desc, MAGIC_DESC)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatmulDescSetAttribute(
    desc: *mut c_void,
    attr: c_int,
    buf: *const c_void,
    size: usize,
) -> c_int {
    set_attr(desc, MAGIC_DESC, attr, buf, size)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatmulDescGetAttribute(
    desc: *const c_void,
    attr: c_int,
    buf: *mut c_void,
    size: usize,
    written: *mut usize,
) -> c_int {
    get_attr(desc, MAGIC_DESC, attr, buf, size, written)
}

// ---- matrix layout ----------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatrixLayoutCreate(
    layout: *mut *mut c_void,
    dtype: c_int,
    rows: u64,
    cols: u64,
    ld: i64,
) -> c_int {
    obj_new(layout, MAGIC_LAYOUT, new_layout(dtype, rows, cols, ld))
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatrixLayoutDestroy(layout: *mut c_void) -> c_int {
    obj_free(layout, MAGIC_LAYOUT)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatrixLayoutSetAttribute(
    layout: *mut c_void,
    attr: c_int,
    buf: *const c_void,
    size: usize,
) -> c_int {
    set_attr(layout, MAGIC_LAYOUT, attr, buf, size)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatrixLayoutGetAttribute(
    layout: *const c_void,
    attr: c_int,
    buf: *mut c_void,
    size: usize,
    written: *mut usize,
) -> c_int {
    get_attr(layout, MAGIC_LAYOUT, attr, buf, size, written)
}

// ---- preference --------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatmulPreferenceCreate(pref: *mut *mut c_void) -> c_int {
    obj_new(pref, MAGIC_PREF, new_pref())
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatmulPreferenceDestroy(pref: *mut c_void) -> c_int {
    obj_free(pref, MAGIC_PREF)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatmulPreferenceSetAttribute(
    pref: *mut c_void,
    attr: c_int,
    buf: *const c_void,
    size: usize,
) -> c_int {
    set_attr(pref, MAGIC_PREF, attr, buf, size)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatmulPreferenceGetAttribute(
    pref: *const c_void,
    attr: c_int,
    buf: *mut c_void,
    size: usize,
    written: *mut usize,
) -> c_int {
    get_attr(pref, MAGIC_PREF, attr, buf, size, written)
}

// ---- heuristic -----------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn plan_from_ptrs(
    desc: *const c_void,
    a: *const c_void,
    b: *const c_void,
    c: *const c_void,
    d: *const c_void,
) -> Result<LtPlan, c_int> {
    let desc = obj_ref(desc, MAGIC_DESC).ok_or(INVALID_VALUE)?;
    let a = obj_ref(a, MAGIC_LAYOUT).ok_or(INVALID_VALUE)?;
    let b = obj_ref(b, MAGIC_LAYOUT).ok_or(INVALID_VALUE)?;
    let d = obj_ref(d, MAGIC_LAYOUT).ok_or(INVALID_VALUE)?;
    // C may be NULL in cublasLt only when beta is zero; the layout is then ignored.
    let c = obj_ref(c, MAGIC_LAYOUT).unwrap_or(d);
    plan_matmul(&desc.bag, &a.bag, &b.bag, &c.bag, &d.bag)
}

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatmulAlgoGetHeuristic(
    handle: *mut c_void,
    desc: *mut c_void,
    a: *mut c_void,
    b: *mut c_void,
    c: *mut c_void,
    d: *mut c_void,
    pref: *mut c_void,
    requested: c_int,
    results: *mut HeuristicResult,
    returned: *mut c_int,
) -> c_int {
    if handle.is_null() || unsafe { (*(handle as *const LtHandle)).magic } != MAGIC_HANDLE {
        return INVALID_VALUE;
    }
    if returned.is_null() || results.is_null() || requested <= 0 {
        return INVALID_VALUE;
    }
    unsafe { returned.write(0) };
    if obj_ref(pref, MAGIC_PREF).is_none() {
        return INVALID_VALUE;
    }
    match plan_from_ptrs(desc, a, b, c, d) {
        Ok(_) => {}
        Err(INVALID_VALUE) => return INVALID_VALUE,
        Err(_) => return SUCCESS, // zero algos: the caller sees NOT_SUPPORTED-like "no heuristic"
    }
    let mut algo = MatmulAlgo { data: [0; 8] };
    algo.data[0] = ALGO_MAGIC;
    unsafe {
        results.write(HeuristicResult {
            algo,
            workspace_size: 0,
            state: SUCCESS,
            waves_count: 1.0,
            reserved: [0; 4],
        });
        returned.write(1);
    }
    SUCCESS
}

// ---- matmul -----------------------------------------------------------------------------

#[unsafe(no_mangle)]
pub unsafe extern "system" fn cublasLtMatmul(
    handle: *mut c_void,
    desc: *mut c_void,
    alpha: *const c_void,
    a_ptr: *const c_void,
    a: *mut c_void,
    b_ptr: *const c_void,
    b: *mut c_void,
    beta: *const c_void,
    c_ptr: *const c_void,
    c: *mut c_void,
    d_ptr: *mut c_void,
    d: *mut c_void,
    algo: *const MatmulAlgo,
    _workspace: *mut c_void,
    _workspace_size: usize,
    stream: *mut c_void,
) -> c_int {
    if handle.is_null() || unsafe { (*(handle as *const LtHandle)).magic } != MAGIC_HANDLE {
        return INVALID_VALUE;
    }
    if !algo.is_null() && unsafe { (*algo).data[0] } != ALGO_MAGIC {
        return INVALID_VALUE;
    }
    if alpha.is_null() || d_ptr.is_null() {
        return INVALID_VALUE;
    }
    let plan = match plan_from_ptrs(desc, a, b, c, d) {
        Ok(p) => p,
        Err(e) => return e,
    };
    let dobj = obj_ref(desc, MAGIC_DESC).expect("validated by plan");
    let pointer_mode = dobj.bag.i32_of(D_POINTER_MODE);
    let g = &plan.gemm;
    // The kernel reads C through D's buffer, so beta != 0 needs C == D (same layout).
    let beta_ptr = if beta.is_null() { &0f32 as *const f32 as *const c_void } else { beta };
    if !c_ptr.is_null() && c_ptr != d_ptr as *const c_void {
        let nonzero = pointer_mode == PM_DEVICE || {
            let v = unsafe { read_host_scalar(beta_ptr, g.scalar) };
            v != 0.0
        };
        if nonzero {
            return NOT_SUPPORTED;
        }
    } else if !c_ptr.is_null() && !plan.c_matches_d {
        return NOT_SUPPORTED;
    }
    let bias = if plan.opts.epilogue.has_bias() {
        let p = dobj.bag.u64_of(D_BIAS_POINTER);
        if p == 0 {
            return INVALID_VALUE;
        }
        p
    } else {
        0
    };
    if dobj.bag.u64_of(D_EPILOGUE_AUX_POINTER) != 0 {
        return NOT_SUPPORTED;
    }
    let (ap, bp) = if plan.swap { (b_ptr, a_ptr) } else { (a_ptr, b_ptr) };
    let batching = Batching::Strided {
        a: ap as u64,
        b: bp as u64,
        c: d_ptr as u64,
        sa: plan.sa,
        sb: plan.sb,
        sc: plan.sd,
    };
    let mut h = Handle {
        stream,
        math_mode: 0,
        pointer_mode: if pointer_mode == PM_DEVICE { POINTER_MODE_DEVICE } else { POINTER_MODE_HOST },
        workspace: (0, 0),
    };
    if driver().is_err() {
        return NOT_INITIALIZED;
    }
    unsafe {
        run_gemm_ex(&mut h, g, alpha, beta_ptr, plan.batch as c_int, batching, &plan.opts, bias)
    }
}

unsafe fn read_host_scalar(p: *const c_void, kind: ScalarKind) -> f32 {
    match kind {
        ScalarKind::F32 => unsafe { *(p as *const f32) },
        ScalarKind::F16 => f16_bits_to_f32(unsafe { *(p as *const u16) }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ptr::{null, null_mut};

    #[test]
    fn abi_struct_layout() {
        assert_eq!(std::mem::size_of::<MatmulAlgo>(), 64);
        assert_eq!(std::mem::size_of::<HeuristicResult>(), 96);
        assert_eq!(std::mem::offset_of!(HeuristicResult, workspace_size), 64);
        assert_eq!(std::mem::offset_of!(HeuristicResult, state), 72);
        assert_eq!(std::mem::offset_of!(HeuristicResult, waves_count), 76);
        assert_eq!(std::mem::offset_of!(HeuristicResult, reserved), 80);
    }

    struct Fx {
        handle: *mut c_void,
        desc: *mut c_void,
        a: *mut c_void,
        b: *mut c_void,
        d: *mut c_void,
        pref: *mut c_void,
    }

    unsafe fn fixture(dtype: c_int, compute: c_int, epi: u32) -> Fx {
        unsafe {
            let mut fx = Fx { handle: null_mut(), desc: null_mut(), a: null_mut(), b: null_mut(), d: null_mut(), pref: null_mut() };
            assert_eq!(cublasLtCreate(&mut fx.handle), SUCCESS);
            let scale = if compute == COMPUTE_16F { CUDA_R_16F } else { CUDA_R_32F };
            assert_eq!(cublasLtMatmulDescCreate(&mut fx.desc, compute, scale), SUCCESS);
            let t: c_int = 1;
            assert_eq!(cublasLtMatmulDescSetAttribute(fx.desc, D_TRANSA, &t as *const _ as _, 4), SUCCESS);
            assert_eq!(cublasLtMatmulDescSetAttribute(fx.desc, D_EPILOGUE, &epi as *const _ as _, 4), SUCCESS);
            // Linear(in=6, out=4), batch 3: m=4, n=3, k=6
            assert_eq!(cublasLtMatrixLayoutCreate(&mut fx.a, dtype, 6, 4, 6), SUCCESS);
            assert_eq!(cublasLtMatrixLayoutCreate(&mut fx.b, dtype, 6, 3, 6), SUCCESS);
            assert_eq!(cublasLtMatrixLayoutCreate(&mut fx.d, dtype, 4, 3, 4), SUCCESS);
            assert_eq!(cublasLtMatmulPreferenceCreate(&mut fx.pref), SUCCESS);
            let ws: u64 = 32 << 20;
            assert_eq!(cublasLtMatmulPreferenceSetAttribute(fx.pref, P_MAX_WORKSPACE_BYTES, &ws as *const _ as _, 8), SUCCESS);
            fx
        }
    }

    unsafe fn free(fx: Fx) {
        unsafe {
            assert_eq!(cublasLtMatmulPreferenceDestroy(fx.pref), SUCCESS);
            assert_eq!(cublasLtMatrixLayoutDestroy(fx.a), SUCCESS);
            assert_eq!(cublasLtMatrixLayoutDestroy(fx.b), SUCCESS);
            assert_eq!(cublasLtMatrixLayoutDestroy(fx.d), SUCCESS);
            assert_eq!(cublasLtMatmulDescDestroy(fx.desc), SUCCESS);
            assert_eq!(cublasLtDestroy(fx.handle), SUCCESS);
        }
    }

    #[test]
    fn heuristic_returns_one_algo_for_linear() {
        unsafe {
            for (dt, compute) in [(CUDA_R_32F, COMPUTE_32F), (CUDA_R_16F, COMPUTE_32F), (CUDA_R_16BF, COMPUTE_32F), (CUDA_R_16F, COMPUTE_16F)] {
                for epi in [EPI_DEFAULT, EPI_BIAS, EPI_RELU_BIAS, EPI_GELU_BIAS] {
                    let fx = fixture(dt, compute, epi);
                    let mut res = std::mem::zeroed::<HeuristicResult>();
                    let mut n = -1;
                    let st = cublasLtMatmulAlgoGetHeuristic(fx.handle, fx.desc, fx.a, fx.b, fx.d, fx.d, fx.pref, 1, &mut res, &mut n);
                    assert_eq!((st, n), (SUCCESS, 1), "dt={dt} epi={epi}");
                    assert_eq!(res.algo.data[0], ALGO_MAGIC);
                    assert_eq!(res.state, SUCCESS);
                    assert_eq!(res.workspace_size, 0);
                    free(fx);
                }
            }
        }
    }

    #[test]
    fn heuristic_reports_zero_algos_or_errors() {
        unsafe {
            let fx = fixture(CUDA_R_32F, COMPUTE_32F, EPI_GELU_AUX);
            let mut res = std::mem::zeroed::<HeuristicResult>();
            let mut n = -1;
            // unsupported epilogue -> SUCCESS with zero results
            let st = cublasLtMatmulAlgoGetHeuristic(fx.handle, fx.desc, fx.a, fx.b, fx.d, fx.d, fx.pref, 1, &mut res, &mut n);
            assert_eq!((st, n), (SUCCESS, 0));
            // bad arguments
            assert_eq!(cublasLtMatmulAlgoGetHeuristic(fx.handle, fx.desc, fx.a, fx.b, fx.d, fx.d, fx.pref, 0, &mut res, &mut n), INVALID_VALUE);
            assert_eq!(cublasLtMatmulAlgoGetHeuristic(null_mut(), fx.desc, fx.a, fx.b, fx.d, fx.d, fx.pref, 1, &mut res, &mut n), INVALID_VALUE);
            assert_eq!(cublasLtMatmulAlgoGetHeuristic(fx.handle, null_mut(), fx.a, fx.b, fx.d, fx.d, fx.pref, 1, &mut res, &mut n), INVALID_VALUE);
            // mismatched dims
            let mut bad: *mut c_void = null_mut();
            assert_eq!(cublasLtMatrixLayoutCreate(&mut bad, CUDA_R_32F, 7, 3, 7), SUCCESS);
            let epi = EPI_DEFAULT;
            cublasLtMatmulDescSetAttribute(fx.desc, D_EPILOGUE, &epi as *const _ as _, 4);
            assert_eq!(cublasLtMatmulAlgoGetHeuristic(fx.handle, fx.desc, fx.a, bad, fx.d, fx.d, fx.pref, 1, &mut res, &mut n), INVALID_VALUE);
            assert_eq!(cublasLtMatrixLayoutDestroy(bad), SUCCESS);
            free(fx);
        }
    }

    #[test]
    fn object_lifetime_and_stale_pointers() {
        unsafe {
            let mut d: *mut c_void = null_mut();
            assert_eq!(cublasLtMatmulDescCreate(&mut d, COMPUTE_32F, CUDA_R_32F), SUCCESS);
            // wrong object kind is rejected
            assert_eq!(cublasLtMatrixLayoutDestroy(d), INVALID_VALUE);
            let mut buf = [0u8; 4];
            let mut n = 0usize;
            assert_eq!(cublasLtMatmulDescGetAttribute(d, D_COMPUTE_TYPE, buf.as_mut_ptr().cast(), 4, &mut n), SUCCESS);
            assert_eq!((n, i32::from_le_bytes(buf)), (4, COMPUTE_32F));
            assert_eq!(cublasLtMatmulDescGetAttribute(d, D_TRANSA, buf.as_mut_ptr().cast(), 2, &mut n), INVALID_VALUE);
            assert_eq!(cublasLtMatmulDescSetAttribute(d, D_TRANSA, null(), 4), INVALID_VALUE);
            assert_eq!(cublasLtMatmulDescDestroy(d), SUCCESS);
            assert_eq!(cublasLtMatmulDescDestroy(null_mut()), INVALID_VALUE);
            assert_eq!(cublasLtDestroy(null_mut()), INVALID_VALUE);
        }
    }

    #[test]
    fn version_and_properties() {
        unsafe {
            assert_eq!(cublasLtGetVersion(), 120900);
            let mut v = 0;
            assert_eq!(cublasLtGetProperty(0, &mut v), SUCCESS);
            assert_eq!(v, 12);
            assert_eq!(cublasLtGetProperty(1, &mut v), SUCCESS);
            assert_eq!(v, 9);
            assert_eq!(cublasLtGetProperty(7, &mut v), INVALID_VALUE);
        }
    }

    #[test]
    fn matmul_argument_errors_precede_driver_use() {
        unsafe {
            let fx = fixture(CUDA_R_32F, COMPUTE_32F, EPI_BIAS);
            let one = 1.0f32;
            let p = &one as *const f32 as *const c_void;
            let bad_algo = MatmulAlgo { data: [1; 8] };
            let dummy = 0x1000 as *const c_void;
            // foreign algo
            assert_eq!(
                cublasLtMatmul(fx.handle, fx.desc, p, dummy, fx.a, dummy, fx.b, p, dummy, fx.d, dummy as _, fx.d, &bad_algo, null_mut(), 0, null_mut()),
                INVALID_VALUE
            );
            // null alpha / D
            assert_eq!(
                cublasLtMatmul(fx.handle, fx.desc, null(), dummy, fx.a, dummy, fx.b, p, dummy, fx.d, dummy as _, fx.d, null(), null_mut(), 0, null_mut()),
                INVALID_VALUE
            );
            // bias epilogue without a bias pointer is rejected before any driver use
            // (needs the driver to reach: only assert it is not SUCCESS)
            let st = cublasLtMatmul(fx.handle, fx.desc, p, dummy, fx.a, dummy, fx.b, null(), null(), fx.d, dummy as _, fx.d, null(), null_mut(), 0, null_mut());
            assert_eq!(st, INVALID_VALUE);
            // separate C with nonzero beta is not supported
            let beta = 0.5f32;
            let st = cublasLtMatmul(fx.handle, fx.desc, p, dummy, fx.a, dummy, fx.b, &beta as *const f32 as _, 0x2000 as _, fx.d, dummy as _, fx.d, null(), null_mut(), 0, null_mut());
            assert_eq!(st, NOT_SUPPORTED);
            free(fx);
        }
    }
}
