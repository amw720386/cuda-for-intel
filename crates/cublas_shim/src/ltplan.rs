//! Pure (driver-free) cublasLt descriptor handling.
//!
//! Attribute storage with cublasLt's sizes and defaults, translation of
//! (matmul descriptor + four matrix layouts) into the op-agnostic tiled GEMM
//! plus fused epilogue, and CPU references. Enum values follow `cublasLt.h`
//! from memory; every value not verified against the real library is listed in
//! `docs/cublaslt-abi-todo.md`.

use crate::plan::*;
use crate::ptxgen::{Elem, Epilogue, Opts};
use std::collections::BTreeMap;
use std::ffi::c_int;

// ---- cublasLtMatmulDescAttributes_t -------------------------------------
pub const D_COMPUTE_TYPE: c_int = 0;
pub const D_SCALE_TYPE: c_int = 1;
pub const D_POINTER_MODE: c_int = 2;
pub const D_TRANSA: c_int = 3;
pub const D_TRANSB: c_int = 4;
pub const D_TRANSC: c_int = 5;
pub const D_FILL_MODE: c_int = 6;
pub const D_EPILOGUE: c_int = 7;
pub const D_BIAS_POINTER: c_int = 8;
pub const D_BIAS_BATCH_STRIDE: c_int = 10;
pub const D_EPILOGUE_AUX_POINTER: c_int = 11;
pub const D_EPILOGUE_AUX_LD: c_int = 12;
pub const D_EPILOGUE_AUX_BATCH_STRIDE: c_int = 13;
pub const D_ALPHA_VECTOR_BATCH_STRIDE: c_int = 14;
pub const D_SM_COUNT_TARGET: c_int = 15;
pub const D_A_SCALE_POINTER: c_int = 17;
pub const D_B_SCALE_POINTER: c_int = 18;
pub const D_C_SCALE_POINTER: c_int = 19;
pub const D_D_SCALE_POINTER: c_int = 20;
pub const D_AMAX_D_POINTER: c_int = 21;
pub const D_EPILOGUE_AUX_DATA_TYPE: c_int = 22;
pub const D_EPILOGUE_AUX_SCALE_POINTER: c_int = 23;
pub const D_EPILOGUE_AUX_AMAX_POINTER: c_int = 24;
pub const D_FAST_ACCUM: c_int = 25;
pub const D_BIAS_DATA_TYPE: c_int = 26;
/// Unknown attribute ids up to this bound are stored opaquely (<= 64 bytes).
pub const D_OPAQUE_MAX_ID: c_int = 63;

// ---- cublasLtMatmulDescAttributes_t pointer modes ------------------------
pub const PM_HOST: c_int = 0;
pub const PM_DEVICE: c_int = 1;
pub const PM_DEVICE_VECTOR: c_int = 2;
pub const PM_ALPHA_DEVICE_VECTOR_BETA_ZERO: c_int = 3;
pub const PM_ALPHA_DEVICE_VECTOR_BETA_HOST: c_int = 4;

// ---- cublasLtEpilogue_t ---------------------------------------------------
pub const EPI_DEFAULT: u32 = 1;
pub const EPI_RELU: u32 = 2;
pub const EPI_RELU_AUX: u32 = 2 | 128;
pub const EPI_BIAS: u32 = 4;
pub const EPI_RELU_BIAS: u32 = 2 | 4;
pub const EPI_RELU_AUX_BIAS: u32 = 2 | 4 | 128;
pub const EPI_DRELU: u32 = 8 | 128;
pub const EPI_DRELU_BGRAD: u32 = 8 | 16 | 128;
pub const EPI_GELU: u32 = 32;
pub const EPI_GELU_AUX: u32 = 32 | 128;
pub const EPI_GELU_BIAS: u32 = 32 | 4;
pub const EPI_GELU_AUX_BIAS: u32 = 32 | 4 | 128;
pub const EPI_DGELU: u32 = 64 | 128;
pub const EPI_DGELU_BGRAD: u32 = 64 | 16 | 128;
pub const EPI_BGRADA: u32 = 256;
pub const EPI_BGRADB: u32 = 512;

// ---- cublasLtMatrixLayoutAttribute_t --------------------------------------
pub const L_TYPE: c_int = 0;
pub const L_ORDER: c_int = 1;
pub const L_ROWS: c_int = 2;
pub const L_COLS: c_int = 3;
pub const L_LD: c_int = 4;
pub const L_BATCH_COUNT: c_int = 5;
pub const L_STRIDED_BATCH_OFFSET: c_int = 6;
pub const L_PLANE_OFFSET: c_int = 7;

// ---- cublasLtOrder_t ------------------------------------------------------
pub const ORDER_COL: c_int = 0;
pub const ORDER_ROW: c_int = 1;

// ---- cublasLtMatmulPreferenceAttributes_t ---------------------------------
pub const P_SEARCH_MODE: c_int = 0;
pub const P_MAX_WORKSPACE_BYTES: c_int = 1;
pub const P_REDUCTION_SCHEME_MASK: c_int = 3;
pub const P_MIN_ALIGNMENT_A_BYTES: c_int = 5;
pub const P_MIN_ALIGNMENT_B_BYTES: c_int = 6;
pub const P_MIN_ALIGNMENT_C_BYTES: c_int = 7;
pub const P_MIN_ALIGNMENT_D_BYTES: c_int = 8;
pub const P_MAX_WAVES_COUNT: c_int = 9;
pub const P_IMPL_MASK: c_int = 12;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Kind {
    Desc,
    Layout,
    Pref,
}

/// Exact byte size of an attribute, or None if the id is unknown.
pub fn attr_size(kind: Kind, id: c_int) -> Option<usize> {
    match kind {
        Kind::Desc => match id {
            D_COMPUTE_TYPE | D_SCALE_TYPE | D_POINTER_MODE | D_TRANSA | D_TRANSB | D_TRANSC
            | D_FILL_MODE | D_EPILOGUE | D_SM_COUNT_TARGET | D_EPILOGUE_AUX_DATA_TYPE
            | D_BIAS_DATA_TYPE => Some(4),
            D_BIAS_POINTER | D_EPILOGUE_AUX_POINTER | D_A_SCALE_POINTER | D_B_SCALE_POINTER
            | D_C_SCALE_POINTER | D_D_SCALE_POINTER | D_AMAX_D_POINTER
            | D_EPILOGUE_AUX_SCALE_POINTER | D_EPILOGUE_AUX_AMAX_POINTER | D_BIAS_BATCH_STRIDE
            | D_EPILOGUE_AUX_LD | D_EPILOGUE_AUX_BATCH_STRIDE | D_ALPHA_VECTOR_BATCH_STRIDE => {
                Some(8)
            }
            D_FAST_ACCUM => Some(1),
            _ => None,
        },
        Kind::Layout => match id {
            L_TYPE | L_ORDER | L_BATCH_COUNT => Some(4),
            L_ROWS | L_COLS | L_LD | L_STRIDED_BATCH_OFFSET | L_PLANE_OFFSET => Some(8),
            _ => None,
        },
        Kind::Pref => match id {
            P_SEARCH_MODE | P_REDUCTION_SCHEME_MASK | P_MIN_ALIGNMENT_A_BYTES
            | P_MIN_ALIGNMENT_B_BYTES | P_MIN_ALIGNMENT_C_BYTES | P_MIN_ALIGNMENT_D_BYTES
            | P_IMPL_MASK => Some(4),
            P_MAX_WORKSPACE_BYTES => Some(8),
            P_MAX_WAVES_COUNT => Some(4),
            _ => None,
        },
    }
}

/// Attribute store shared by descriptors, layouts and preferences. Values are
/// raw little-endian bytes; absent keys read as the per-kind default.
#[derive(Clone, Debug, Default)]
pub struct AttrBag {
    kind: Option<Kind>,
    map: BTreeMap<c_int, Vec<u8>>,
}

impl AttrBag {
    pub fn new(kind: Kind) -> Self {
        Self { kind: Some(kind), map: BTreeMap::new() }
    }
    fn kind(&self) -> Kind {
        self.kind.unwrap_or(Kind::Desc)
    }

    /// cublasLt*SetAttribute.
    pub fn set(&mut self, id: c_int, bytes: &[u8]) -> c_int {
        let kind = self.kind();
        match attr_size(kind, id) {
            Some(n) => {
                if bytes.len() != n {
                    return INVALID_VALUE;
                }
            }
            None => {
                if kind != Kind::Desc || !(0..=D_OPAQUE_MAX_ID).contains(&id) || bytes.len() > 64 {
                    return INVALID_VALUE;
                }
            }
        }
        if kind == Kind::Desc {
            let v = u32_of(bytes);
            let bad = match id {
                D_TRANSA | D_TRANSB | D_TRANSC => v > 2,
                D_POINTER_MODE => v > 4,
                _ => false,
            };
            if bad {
                return INVALID_VALUE;
            }
        }
        if kind == Kind::Layout && id == L_ORDER && u32_of(bytes) > 4 {
            return INVALID_VALUE;
        }
        self.map.insert(id, bytes.to_vec());
        SUCCESS
    }

    pub fn is_set(&self, id: c_int) -> bool {
        self.map.contains_key(&id)
    }

    /// Default value bytes for `id` (zero except where cublasLt defines otherwise).
    fn default_bytes(&self, id: c_int) -> Vec<u8> {
        let n = attr_size(self.kind(), id).unwrap_or(0);
        let mut v = vec![0u8; n];
        match (self.kind(), id) {
            (Kind::Desc, D_EPILOGUE) => v.copy_from_slice(&EPI_DEFAULT.to_le_bytes()),
            (Kind::Layout, L_BATCH_COUNT) => v.copy_from_slice(&1i32.to_le_bytes()),
            (Kind::Pref, P_MAX_WAVES_COUNT) => v.copy_from_slice(&0f32.to_le_bytes()),
            _ => {}
        }
        v
    }

    /// cublasLt*GetAttribute. `size_written` receives the attribute size; a
    /// zero-length query only reports the size.
    pub fn get(&self, id: c_int, out: &mut [u8], size_written: &mut usize) -> c_int {
        let bytes = match self.map.get(&id) {
            Some(b) => b.clone(),
            None => {
                if attr_size(self.kind(), id).is_none() {
                    return INVALID_VALUE;
                }
                self.default_bytes(id)
            }
        };
        *size_written = bytes.len();
        if out.is_empty() {
            return SUCCESS;
        }
        if out.len() < bytes.len() {
            return INVALID_VALUE;
        }
        out[..bytes.len()].copy_from_slice(&bytes);
        SUCCESS
    }

    pub fn i32_of(&self, id: c_int) -> i32 {
        let b = self.map.get(&id).cloned().unwrap_or_else(|| self.default_bytes(id));
        u32_of(&b) as i32
    }
    pub fn u32_of(&self, id: c_int) -> u32 {
        self.i32_of(id) as u32
    }
    pub fn u64_of(&self, id: c_int) -> u64 {
        let b = self.map.get(&id).cloned().unwrap_or_else(|| self.default_bytes(id));
        let mut a = [0u8; 8];
        a[..b.len().min(8)].copy_from_slice(&b[..b.len().min(8)]);
        u64::from_le_bytes(a)
    }
    pub fn i64_of(&self, id: c_int) -> i64 {
        self.u64_of(id) as i64
    }
}

fn u32_of(b: &[u8]) -> u32 {
    let mut a = [0u8; 4];
    a[..b.len().min(4)].copy_from_slice(&b[..b.len().min(4)]);
    u32::from_le_bytes(a)
}

// ---- object constructors --------------------------------------------------

pub fn new_desc(compute_type: c_int, scale_type: c_int) -> AttrBag {
    let mut b = AttrBag::new(Kind::Desc);
    b.map.insert(D_COMPUTE_TYPE, compute_type.to_le_bytes().to_vec());
    b.map.insert(D_SCALE_TYPE, scale_type.to_le_bytes().to_vec());
    b
}

pub fn new_layout(dtype: c_int, rows: u64, cols: u64, ld: i64) -> AttrBag {
    let mut b = AttrBag::new(Kind::Layout);
    b.map.insert(L_TYPE, dtype.to_le_bytes().to_vec());
    b.map.insert(L_ROWS, rows.to_le_bytes().to_vec());
    b.map.insert(L_COLS, cols.to_le_bytes().to_vec());
    b.map.insert(L_LD, ld.to_le_bytes().to_vec());
    b
}

pub fn new_pref() -> AttrBag {
    AttrBag::new(Kind::Pref)
}

// ---- planning ---------------------------------------------------------------

pub fn decode_epilogue(code: u32) -> Option<Epilogue> {
    Some(match code {
        EPI_DEFAULT => Epilogue::None,
        EPI_RELU => Epilogue::Relu,
        EPI_GELU => Epilogue::Gelu,
        EPI_BIAS => Epilogue::Bias,
        EPI_RELU_BIAS => Epilogue::ReluBias,
        EPI_GELU_BIAS => Epilogue::GeluBias,
        _ => return None,
    })
}

pub fn to_elem(d: DType) -> Elem {
    match d {
        DType::F32 => Elem::F32,
        DType::F16 => Elem::F16,
        DType::Bf16 => Elem::Bf16,
    }
}

/// Stored layout of one matrix.
#[derive(Clone, Copy, Debug)]
struct Stored {
    dtype: DType,
    rows: u64,
    cols: u64,
    /// element strides of (row, col) within one batch entry
    sr: i64,
    sc: i64,
    batch: i64,
    batch_stride: i64,
}

fn stored(l: &AttrBag) -> Result<Stored, c_int> {
    let dtype = DType::from_cuda(l.i32_of(L_TYPE)).ok_or(NOT_SUPPORTED)?;
    let (rows, cols) = (l.u64_of(L_ROWS), l.u64_of(L_COLS));
    let ld = l.i64_of(L_LD);
    let (sr, sc, need) = match l.i32_of(L_ORDER) {
        ORDER_COL => (1, ld, rows),
        ORDER_ROW => (ld, 1, cols),
        _ => return Err(NOT_SUPPORTED),
    };
    if rows > u32::MAX as u64 || cols > u32::MAX as u64 {
        return Err(NOT_SUPPORTED);
    }
    if ld < 0 || (ld as u64) < need.max(1) && (rows != 0 && cols != 0) {
        return Err(INVALID_VALUE);
    }
    if ld > u32::MAX as i64 {
        return Err(NOT_SUPPORTED);
    }
    let batch = l.i32_of(L_BATCH_COUNT) as i64;
    if batch < 1 {
        return Err(INVALID_VALUE);
    }
    Ok(Stored { dtype, rows, cols, sr, sc, batch, batch_stride: l.i64_of(L_STRIDED_BATCH_OFFSET) })
}

/// Logical view of `op(X)`: dims and element strides.
#[derive(Clone, Copy, Debug)]
struct View {
    rows: u64,
    cols: u64,
    sr: i64,
    sc: i64,
}

fn op_view(s: &Stored, trans: bool) -> View {
    if trans {
        View { rows: s.cols, cols: s.rows, sr: s.sc, sc: s.sr }
    } else {
        View { rows: s.rows, cols: s.cols, sr: s.sr, sc: s.sc }
    }
}

fn transpose(v: View) -> View {
    View { rows: v.cols, cols: v.rows, sr: v.sc, sc: v.sr }
}

/// (trans, ld) reproducing strides `(si, sl)` for the shim's GemmDesc.
fn trans_ld(si: i64, sl: i64) -> Result<(bool, u32), c_int> {
    if si == 1 {
        Ok((false, sl as u32))
    } else if sl == 1 {
        Ok((true, si as u32))
    } else {
        Err(NOT_SUPPORTED)
    }
}

/// A validated matmul, ready to launch.
#[derive(Clone, Copy, Debug)]
pub struct LtPlan {
    pub gemm: GemmDesc,
    /// Operands are exchanged (D is row-major, so D^T = op(B)^T op(A)^T is computed).
    pub swap: bool,
    pub batch: u32,
    /// batch strides in elements for the kernel's A, B (after any swap) and for D
    pub sa: i64,
    pub sb: i64,
    pub sd: i64,
    pub opts: Opts,
    /// C layout is identical to D (required to honour beta != 0 with a separate C)
    pub c_matches_d: bool,
}

/// Translate descriptor + layouts into a launchable plan.
/// Errors follow cublasLt: INVALID_VALUE for malformed dimensions, NOT_SUPPORTED for
/// type/layout/epilogue combinations this shim cannot run.
pub fn plan_matmul(
    desc: &AttrBag,
    a: &AttrBag,
    b: &AttrBag,
    c: &AttrBag,
    d: &AttrBag,
) -> Result<LtPlan, c_int> {
    let compute = desc.i32_of(D_COMPUTE_TYPE);
    let scale = scalar_kind(compute).ok_or(NOT_SUPPORTED)?;
    if scalar_kind(desc.i32_of(D_SCALE_TYPE)) != Some(scale) {
        return Err(NOT_SUPPORTED);
    }
    if !matches!(desc.i32_of(D_POINTER_MODE), PM_HOST | PM_DEVICE) {
        return Err(NOT_SUPPORTED); // vector pointer modes
    }
    let ta = op_is_transposed(desc.i32_of(D_TRANSA)).ok_or(INVALID_VALUE)?;
    let tb = op_is_transposed(desc.i32_of(D_TRANSB)).ok_or(INVALID_VALUE)?;
    if desc.i32_of(D_TRANSC) != OP_N {
        return Err(NOT_SUPPORTED);
    }
    let epi_code = desc.u32_of(D_EPILOGUE);
    let epilogue = decode_epilogue(epi_code).ok_or(NOT_SUPPORTED)?;

    let (sa, sb, sc, sd) = (stored(a)?, stored(b)?, stored(c)?, stored(d)?);
    // Type rules (same as cublasGemmEx): A == B; C == D.
    if sa.dtype != sb.dtype || sc.dtype != sd.dtype {
        return Err(NOT_SUPPORTED);
    }
    match (sa.dtype, sd.dtype) {
        (DType::F32, DType::F32)
        | (DType::F16, DType::F16 | DType::F32)
        | (DType::Bf16, DType::Bf16 | DType::F32) => {}
        _ => return Err(NOT_SUPPORTED),
    }
    let acc16 = scale == ScalarKind::F16;
    if acc16 && (sa.dtype != DType::F16 || sd.dtype != DType::F16) {
        return Err(NOT_SUPPORTED);
    }

    let (va, vb) = (op_view(&sa, ta), op_view(&sb, tb));
    let (m, n, k) = (sd.rows, sd.cols, va.cols);
    if va.rows != m || vb.cols != n || vb.rows != k {
        return Err(INVALID_VALUE);
    }
    if sa.batch != sb.batch || sa.batch != sd.batch {
        return Err(INVALID_VALUE);
    }
    let batch = sa.batch;
    if batch > u32::MAX as i64 {
        return Err(NOT_SUPPORTED);
    }

    // D row-major: compute D^T (column-major n x m) = op(B)^T op(A)^T.
    let swap = d.i32_of(L_ORDER) == ORDER_ROW;
    let (ka, kb, km, kn, ldd) = if swap {
        (transpose(vb), transpose(va), n, m, sd.sr)
    } else {
        (va, vb, m, n, sd.sc)
    };
    let (a_trans, lda) = trans_ld(ka.sr, ka.sc)?;
    let (b_trans, ldb) = trans_ld(kb.sr, kb.sc)?;
    if ldd < km.max(1) as i64 && km != 0 && kn != 0 {
        return Err(INVALID_VALUE);
    }
    let (ta_dt, tb_dt) = if swap { (sb.dtype, sa.dtype) } else { (sa.dtype, sb.dtype) };
    let gemm = GemmDesc {
        trans_a: a_trans,
        trans_b: b_trans,
        m: km as u32,
        n: kn as u32,
        k: k as u32,
        lda,
        ldb,
        ldc: ldd as u32,
        a: ta_dt,
        b: tb_dt,
        c: sd.dtype,
        scalar: scale,
    };
    let (bsa, bsb) = if swap {
        (sb.batch_stride, sa.batch_stride)
    } else {
        (sa.batch_stride, sb.batch_stride)
    };

    let mut bias = Elem::F32;
    if epilogue.has_bias() {
        if swap {
            return Err(NOT_SUPPORTED); // bias would run along the kernel's column index
        }
        bias = if desc.is_set(D_BIAS_DATA_TYPE) {
            to_elem(DType::from_cuda(desc.i32_of(D_BIAS_DATA_TYPE)).ok_or(NOT_SUPPORTED)?)
        } else {
            to_elem(sd.dtype)
        };
        if desc.i64_of(D_BIAS_BATCH_STRIDE) != 0 && batch > 1 {
            return Err(NOT_SUPPORTED);
        }
    }
    let c_matches_d = sc.rows == sd.rows
        && sc.cols == sd.cols
        && sc.sr == sd.sr
        && sc.sc == sd.sc
        && sc.batch_stride == sd.batch_stride
        && sc.batch == sd.batch
        && c.i32_of(L_ORDER) == d.i32_of(L_ORDER);
    Ok(LtPlan {
        gemm,
        swap,
        batch: batch as u32,
        sa: bsa,
        sb: bsb,
        sd: sd.batch_stride,
        opts: Opts { epilogue, bias, acc16 },
        c_matches_d,
    })
}

// ---- CPU references ----------------------------------------------------------

pub fn gelu_tanh(x: f32) -> f32 {
    let u = crate::ptxgen::GELU_SQRT_2_OVER_PI * (x + crate::ptxgen::GELU_CUBIC * x * x * x);
    0.5 * x * (1.0 + u.tanh())
}

pub fn round_f16(v: f32) -> f32 {
    f16_bits_to_f32(f32_to_f16_bits(v))
}

fn apply_epilogue(e: Epilogue, mut v: f32, bias: f32) -> f32 {
    if e.has_bias() {
        v += bias;
    }
    match e {
        Epilogue::Relu | Epilogue::ReluBias => v.max(0.0),
        Epilogue::Gelu | Epilogue::GeluBias => gelu_tanh(v),
        _ => v,
    }
}

/// Literal cublasLt semantics evaluated straight from the layouts:
/// `D = epilogue(alpha * op(A) op(B) + beta * C [+ bias])`, independent of
/// [`plan_matmul`]. Buffers hold all batch entries; offsets use each layout's
/// batch stride. `bias` has `rows(D)` elements of dtype `bias_dtype`.
#[allow(clippy::too_many_arguments)]
pub fn reference_matmul(
    desc: &AttrBag,
    la: &AttrBag,
    lb: &AttrBag,
    lc: &AttrBag,
    ld_: &AttrBag,
    alpha: f32,
    beta: f32,
    a: &[u8],
    b: &[u8],
    c: &[u8],
    dout: &mut [u8],
    bias: Option<(&[u8], DType)>,
) {
    let idx = |l: &AttrBag, r: usize, col: usize, z: usize| -> usize {
        let ld = l.i64_of(L_LD) as usize;
        let off = z as i64 * l.i64_of(L_STRIDED_BATCH_OFFSET);
        let base = if l.i32_of(L_ORDER) == ORDER_ROW { r * ld + col } else { r + col * ld };
        (base as i64 + off) as usize
    };
    let ta = desc.i32_of(D_TRANSA) != OP_N;
    let tb = desc.i32_of(D_TRANSB) != OP_N;
    let epi = decode_epilogue(desc.u32_of(D_EPILOGUE)).unwrap();
    let acc16 = scalar_kind(desc.i32_of(D_COMPUTE_TYPE)) == Some(ScalarKind::F16);
    let (adt, bdt, cdt, ddt) = (
        DType::from_cuda(la.i32_of(L_TYPE)).unwrap(),
        DType::from_cuda(lb.i32_of(L_TYPE)).unwrap(),
        DType::from_cuda(lc.i32_of(L_TYPE)).unwrap(),
        DType::from_cuda(ld_.i32_of(L_TYPE)).unwrap(),
    );
    let (m, n) = (ld_.u64_of(L_ROWS) as usize, ld_.u64_of(L_COLS) as usize);
    let k = if ta { la.u64_of(L_ROWS) } else { la.u64_of(L_COLS) } as usize;
    let rd = |v: f32| if acc16 { round_f16(v) } else { v };
    for z in 0..ld_.i32_of(L_BATCH_COUNT) as usize {
        for j in 0..n {
            for i in 0..m {
                let mut acc = 0.0f32;
                for l in 0..k {
                    let ai = if ta { idx(la, l, i, z) } else { idx(la, i, l, z) };
                    let bi = if tb { idx(lb, j, l, z) } else { idx(lb, l, j, z) };
                    acc = rd(acc + decode(adt, a, ai) * decode(bdt, b, bi));
                }
                let mut v = rd(acc * alpha);
                if beta != 0.0 {
                    v = rd(v + beta * decode(cdt, c, idx(lc, i, j, z)));
                }
                let bv = bias.map(|(bytes, dt)| decode(dt, bytes, i)).unwrap_or(0.0);
                v = apply_epilogue(epi, v, bv);
                encode(ddt, dout, idx(ld_, i, j, z), v);
            }
        }
    }
}

/// Emulates what the launched kernel computes for a plan: operands exchanged
/// when `plan.swap`, strides from `GemmDesc`, epilogue after the beta term. `c`
/// is both input (beta != 0) and output, i.e. it is D's buffer.
pub fn emulate_plan(
    p: &LtPlan,
    alpha: f32,
    beta: f32,
    a: &[u8],
    b: &[u8],
    dbuf: &mut [u8],
    bias: Option<&[u8]>,
) {
    let (a, b) = if p.swap { (b, a) } else { (a, b) };
    let g = &p.gemm;
    let s = g.strides();
    let rd = |v: f32| if p.opts.acc16 { round_f16(v) } else { v };
    for z in 0..p.batch as i64 {
        for j in 0..g.n as usize {
            for i in 0..g.m as usize {
                let mut acc = 0.0f32;
                for l in 0..g.k as usize {
                    let ai = z * p.sa + (i * s.a_si as usize + l * s.a_sl as usize) as i64;
                    let bi = z * p.sb + (l * s.b_sl as usize + j * s.b_sj as usize) as i64;
                    acc = rd(acc + decode(g.a, a, ai as usize) * decode(g.b, b, bi as usize));
                }
                let ci = (z * p.sd + (i + j * g.ldc as usize) as i64) as usize;
                let mut v = rd(acc * alpha);
                if beta != 0.0 {
                    v = rd(v + beta * decode(g.c, dbuf, ci));
                }
                let bv = match (p.opts.epilogue.has_bias(), bias) {
                    (true, Some(bytes)) => decode(dtype_of(p.opts.bias), bytes, i),
                    _ => 0.0,
                };
                encode(g.c, dbuf, ci, apply_epilogue(p.opts.epilogue, v, bv));
            }
        }
    }
}

fn dtype_of(e: Elem) -> DType {
    match e {
        Elem::F32 => DType::F32,
        Elem::F16 => DType::F16,
        Elem::Bf16 => DType::Bf16,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((*seed >> 33) % 2001) as f32 - 1000.0) / 500.0
    }
    fn fill(dt: DType, n: usize, seed: &mut u64) -> Vec<u8> {
        let mut v = vec![0u8; n * dt.size()];
        for i in 0..n {
            let x = lcg(seed);
            encode(dt, &mut v, i, x);
        }
        v
    }
    fn set_i32(b: &mut AttrBag, id: c_int, v: i32) {
        assert_eq!(b.set(id, &v.to_le_bytes()), SUCCESS);
    }

    #[test]
    fn attribute_sizes_defaults_and_errors() {
        let mut d = new_desc(COMPUTE_32F, CUDA_R_32F);
        let mut out = [0u8; 8];
        let mut n = 0usize;
        // defaults
        assert_eq!(d.get(D_EPILOGUE, &mut out[..4], &mut n), SUCCESS);
        assert_eq!((n, u32::from_le_bytes(out[..4].try_into().unwrap())), (4, EPI_DEFAULT));
        assert_eq!(d.i32_of(D_TRANSA), OP_N);
        assert_eq!(d.i32_of(D_COMPUTE_TYPE), COMPUTE_32F);
        // wrong size / unknown / bad enum
        assert_eq!(d.set(D_TRANSA, &[0u8; 8]), INVALID_VALUE);
        assert_eq!(d.set(D_BIAS_POINTER, &[0u8; 4]), INVALID_VALUE);
        assert_eq!(d.set(D_TRANSA, &9i32.to_le_bytes()), INVALID_VALUE);
        assert_eq!(d.set(500, &[0u8; 4]), INVALID_VALUE);
        // opaque unknown id is accepted and read back
        assert_eq!(d.set(40, &[1, 2, 3]), SUCCESS);
        let mut o3 = [0u8; 3];
        assert_eq!(d.get(40, &mut o3, &mut n), SUCCESS);
        assert_eq!(o3, [1, 2, 3]);
        // pointer round trip, short buffer, size query
        assert_eq!(d.set(D_BIAS_POINTER, &0xdead_beef_u64.to_le_bytes()), SUCCESS);
        assert_eq!(d.u64_of(D_BIAS_POINTER), 0xdead_beef);
        assert_eq!(d.get(D_BIAS_POINTER, &mut out[..4], &mut n), INVALID_VALUE);
        assert_eq!(d.get(D_BIAS_POINTER, &mut [], &mut n), SUCCESS);
        assert_eq!(n, 8);

        let mut l = new_layout(CUDA_R_16F, 4, 3, 4);
        assert_eq!(l.i32_of(L_BATCH_COUNT), 1);
        assert_eq!(l.i32_of(L_ORDER), ORDER_COL);
        assert_eq!(l.set(L_ORDER, &7i32.to_le_bytes()), INVALID_VALUE);
        assert_eq!(l.set(L_ROWS, &[0u8; 4]), INVALID_VALUE);
        assert_eq!(l.set(L_BATCH_COUNT, &5i32.to_le_bytes()), SUCCESS);
        assert_eq!(l.i32_of(L_BATCH_COUNT), 5);

        let mut p = new_pref();
        assert_eq!(p.set(P_MAX_WORKSPACE_BYTES, &(1u64 << 20).to_le_bytes()), SUCCESS);
        assert_eq!(p.u64_of(P_MAX_WORKSPACE_BYTES), 1 << 20);
        assert_eq!(p.set(P_MAX_WORKSPACE_BYTES, &[0u8; 4]), INVALID_VALUE);
        assert_eq!(p.set(99, &[0u8; 4]), INVALID_VALUE);
        let mut dd = d.clone();
        set_i32(&mut dd, D_EPILOGUE, EPI_GELU_BIAS as i32);
        assert_eq!(decode_epilogue(dd.u32_of(D_EPILOGUE)), Some(Epilogue::GeluBias));
    }

    #[test]
    fn epilogue_enum_values() {
        assert_eq!(EPI_RELU_BIAS, 6);
        assert_eq!(EPI_GELU_BIAS, 36);
        assert_eq!(EPI_RELU_AUX, 130);
        assert_eq!(EPI_GELU_AUX_BIAS, 164);
        assert_eq!(EPI_DGELU_BGRAD, 208);
        assert_eq!(decode_epilogue(EPI_RELU_AUX), None);
        assert_eq!(decode_epilogue(EPI_BGRADA), None);
    }

    /// One case: build layouts + descriptor, compare plan emulation to the literal reference.
    #[allow(clippy::too_many_arguments)]
    fn check(
        dt: DType,
        out_dt: DType,
        compute: c_int,
        ta: bool,
        tb: bool,
        row_d: bool,
        epi: u32,
        beta: f32,
        batch: i32,
        mnk: (u64, u64, u64),
    ) {
        let (m, n, k) = mnk;
        let tag = format!("{dt:?}->{out_dt:?} ta={ta} tb={tb} rowD={row_d} epi={epi} beta={beta} batch={batch}");
        let scale = if compute == COMPUTE_16F { CUDA_R_16F } else { CUDA_R_32F };
        let mut desc = new_desc(compute, scale);
        set_i32(&mut desc, D_TRANSA, ta as i32);
        set_i32(&mut desc, D_TRANSB, tb as i32);
        set_i32(&mut desc, D_EPILOGUE, epi as i32);
        let code = |d: DType| match d {
            DType::F32 => CUDA_R_32F,
            DType::F16 => CUDA_R_16F,
            DType::Bf16 => CUDA_R_16BF,
        };
        // Matrix layout helper: stored dims (r,c); order; padding ld by 3.
        let mk = |dtype: DType, r: u64, c: u64, row: bool| {
            let ld = if row { c + 3 } else { r + 3 };
            let mut l = new_layout(code(dtype), r, c, ld as i64);
            if row {
                set_i32(&mut l, L_ORDER, ORDER_ROW);
            }
            if batch > 1 {
                set_i32(&mut l, L_BATCH_COUNT, batch);
                let per = if row { r * ld } else { c * ld };
                assert_eq!(l.set(L_STRIDED_BATCH_OFFSET, &(per as i64).to_le_bytes()), SUCCESS);
            }
            l
        };
        let (ar, ac) = if ta { (k, m) } else { (m, k) };
        let (br, bc) = if tb { (n, k) } else { (k, n) };
        // A is stored row-major when D is, to exercise mixed orders in the same case.
        let la = mk(dt, ar, ac, row_d && ta);
        let lb = mk(dt, br, bc, !row_d && tb);
        let ld_ = mk(out_dt, m, n, row_d);
        let lc = ld_.clone();
        let lens = |l: &AttrBag| {
            let (r, c, ld) = (l.u64_of(L_ROWS), l.u64_of(L_COLS), l.i64_of(L_LD) as u64);
            let per = if l.i32_of(L_ORDER) == ORDER_ROW { r * ld } else { c * ld };
            (per * batch as u64) as usize
        };
        let mut seed = 42u64;
        let a = fill(dt, lens(&la), &mut seed);
        let b = fill(dt, lens(&lb), &mut seed);
        let c0 = fill(out_dt, lens(&ld_), &mut seed);
        let bias_dt = out_dt;
        let bias = fill(bias_dt, m as usize, &mut seed);
        let epi_e = decode_epilogue(epi).unwrap();
        let bias_ref = epi_e.has_bias().then_some((&bias[..], bias_dt));
        let alpha = 0.75f32;
        let mut want = c0.clone();
        reference_matmul(&desc, &la, &lb, &lc, &ld_, alpha, beta, &a, &b, &c0, &mut want, bias_ref);
        let plan = plan_matmul(&desc, &la, &lb, &lc, &ld_);
        if row_d && epi_e.has_bias() {
            assert_eq!(plan.err(), Some(NOT_SUPPORTED), "{tag}");
            return;
        }
        let plan = plan.unwrap_or_else(|e| panic!("{tag}: plan error {e}"));
        assert_eq!(plan.swap, row_d, "{tag}");
        let mut got = c0.clone();
        emulate_plan(&plan, alpha, beta, &a, &b, &mut got, epi_e.has_bias().then_some(&bias[..]));
        let es = out_dt.size();
        for i in 0..want.len() / es {
            let (g, w) = (decode(out_dt, &got, i), decode(out_dt, &want, i));
            let tol = 2e-2 * (1.0 + w.abs());
            assert!((g - w).abs() <= tol, "{tag}: elem {i}: {g} vs {w}");
        }
    }

    #[test]
    fn plan_matches_literal_semantics() {
        for &(dt, out_dt, compute) in &[
            (DType::F32, DType::F32, COMPUTE_32F),
            (DType::F16, DType::F16, COMPUTE_32F),
            (DType::Bf16, DType::Bf16, COMPUTE_32F),
            (DType::F16, DType::F32, COMPUTE_32F),
            (DType::F16, DType::F16, COMPUTE_16F),
        ] {
            for ta in [false, true] {
                for tb in [false, true] {
                    for row_d in [false, true] {
                        for epi in [EPI_DEFAULT, EPI_BIAS, EPI_RELU_BIAS, EPI_GELU_BIAS, EPI_RELU, EPI_GELU] {
                            for (beta, batch) in [(0.0, 1), (0.5, 1), (0.0, 3)] {
                                check(dt, out_dt, compute, ta, tb, row_d, epi, beta, batch, (7, 5, 9));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn pytorch_linear_shape() {
        // nn.Linear(in=6, out=4) on x[3,6]: Lt computes out^T(4x3) = W^T-view(op T) * x^T-view.
        // cublasLt call from at::cuda::blas::gemm_and_bias: transa=T, m=out, n=batch, k=in,
        // lda=in, ldb=in, ldc=out; bias epilogue.
        let mut desc = new_desc(COMPUTE_32F, CUDA_R_32F);
        set_i32(&mut desc, D_TRANSA, 1);
        set_i32(&mut desc, D_EPILOGUE, EPI_BIAS as i32);
        let la = new_layout(CUDA_R_32F, 6, 4, 6); // stored in x out, transposed
        let lb = new_layout(CUDA_R_32F, 6, 3, 6);
        let ld_ = new_layout(CUDA_R_32F, 4, 3, 4);
        let p = plan_matmul(&desc, &la, &lb, &ld_, &ld_).unwrap();
        assert!(p.gemm.trans_a && !p.gemm.trans_b);
        assert_eq!((p.gemm.m, p.gemm.n, p.gemm.k), (4, 3, 6));
        assert_eq!((p.gemm.lda, p.gemm.ldb, p.gemm.ldc), (6, 6, 4));
        assert_eq!(p.opts.epilogue, Epilogue::Bias);
        assert_eq!(p.opts.bias, Elem::F32);
        assert!(p.c_matches_d && !p.swap);
    }

    #[test]
    fn plan_rejections() {
        let desc = new_desc(COMPUTE_32F, CUDA_R_32F);
        let l = |dt, r, c, ld| new_layout(dt, r, c, ld);
        // dimension mismatch -> INVALID_VALUE
        assert_eq!(
            plan_matmul(&desc, &l(CUDA_R_32F, 4, 5, 4), &l(CUDA_R_32F, 6, 3, 6), &l(CUDA_R_32F, 4, 3, 4), &l(CUDA_R_32F, 4, 3, 4)).err(),
            Some(INVALID_VALUE)
        );
        // short leading dimension
        assert_eq!(
            plan_matmul(&desc, &l(CUDA_R_32F, 4, 5, 3), &l(CUDA_R_32F, 5, 3, 5), &l(CUDA_R_32F, 4, 3, 4), &l(CUDA_R_32F, 4, 3, 4)).err(),
            Some(INVALID_VALUE)
        );
        // mixed A/B types, unsupported type
        assert_eq!(
            plan_matmul(&desc, &l(CUDA_R_16F, 4, 5, 4), &l(CUDA_R_32F, 5, 3, 5), &l(CUDA_R_32F, 4, 3, 4), &l(CUDA_R_32F, 4, 3, 4)).err(),
            Some(NOT_SUPPORTED)
        );
        assert_eq!(
            plan_matmul(&desc, &l(9, 4, 5, 4), &l(9, 5, 3, 5), &l(9, 4, 3, 4), &l(9, 4, 3, 4)).err(),
            Some(NOT_SUPPORTED)
        );
        // 16F compute with f32 data
        let d16 = new_desc(COMPUTE_16F, CUDA_R_16F);
        assert_eq!(
            plan_matmul(&d16, &l(CUDA_R_32F, 4, 5, 4), &l(CUDA_R_32F, 5, 3, 5), &l(CUDA_R_32F, 4, 3, 4), &l(CUDA_R_32F, 4, 3, 4)).err(),
            Some(NOT_SUPPORTED)
        );
        // unsupported epilogue
        let mut de = desc.clone();
        set_i32(&mut de, D_EPILOGUE, EPI_GELU_AUX as i32);
        assert_eq!(
            plan_matmul(&de, &l(CUDA_R_32F, 4, 5, 4), &l(CUDA_R_32F, 5, 3, 5), &l(CUDA_R_32F, 4, 3, 4), &l(CUDA_R_32F, 4, 3, 4)).err(),
            Some(NOT_SUPPORTED)
        );
    }

    #[test]
    fn bias_type_defaults_to_d_and_can_override() {
        let mut desc = new_desc(COMPUTE_32F, CUDA_R_32F);
        set_i32(&mut desc, D_EPILOGUE, EPI_BIAS as i32);
        let a = new_layout(CUDA_R_16F, 4, 5, 4);
        let b = new_layout(CUDA_R_16F, 5, 3, 5);
        let d = new_layout(CUDA_R_16F, 4, 3, 4);
        assert_eq!(plan_matmul(&desc, &a, &b, &d, &d).unwrap().opts.bias, Elem::F16);
        set_i32(&mut desc, D_BIAS_DATA_TYPE, CUDA_R_32F);
        assert_eq!(plan_matmul(&desc, &a, &b, &d, &d).unwrap().opts.bias, Elem::F32);
    }

    #[test]
    fn f16_compute_rounds_like_half() {
        // 2048 + 1 is not representable in f16: a true f16 accumulator absorbs the 1.
        assert_eq!(round_f16(2049.0), 2048.0);
        let mut desc = new_desc(COMPUTE_16F, CUDA_R_16F);
        set_i32(&mut desc, D_TRANSA, 1);
        let la = new_layout(CUDA_R_16F, 3, 1, 3);
        let lb = new_layout(CUDA_R_16F, 3, 1, 3);
        let ld_ = new_layout(CUDA_R_16F, 1, 1, 1);
        let plan = plan_matmul(&desc, &la, &lb, &ld_, &ld_).unwrap();
        assert!(plan.opts.acc16);
        let mut a = vec![0u8; 6];
        let mut b = vec![0u8; 6];
        for (i, (x, y)) in [(2048.0, 1.0), (1.0, 1.0), (1.0, 1.0)].iter().enumerate() {
            encode(DType::F16, &mut a, i, *x);
            encode(DType::F16, &mut b, i, *y);
        }
        let mut d16 = vec![0u8; 2];
        emulate_plan(&plan, 1.0, 0.0, &a, &b, &mut d16, None);
        assert_eq!(decode(DType::F16, &d16, 0), 2048.0);
        let mut d32 = vec![0u8; 2];
        let mut desc32 = desc.clone();
        desc32.set(D_COMPUTE_TYPE, &COMPUTE_32F.to_le_bytes());
        desc32.set(D_SCALE_TYPE, &CUDA_R_32F.to_le_bytes());
        let plan32 = plan_matmul(&desc32, &la, &lb, &ld_, &ld_).unwrap();
        emulate_plan(&plan32, 1.0, 0.0, &a, &b, &mut d32, None);
        assert_eq!(decode(DType::F16, &d32, 0), 2050.0);
    }

    #[test]
    fn gelu_matches_known_values() {
        assert!((gelu_tanh(0.0)).abs() < 1e-7);
        assert!((gelu_tanh(1.0) - 0.841192).abs() < 1e-4);
        assert!((gelu_tanh(-1.0) + 0.158808).abs() < 1e-4);
        assert!((gelu_tanh(3.0) - 2.99636).abs() < 1e-3);
    }
}
