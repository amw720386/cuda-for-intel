//! Pure (driver-free) GEMM parameter handling for the cuBLAS shim.
//!
//! Everything that decides *what* a cuBLAS call means lives here so it can be
//! unit-tested on any host: enum decoding, leading-dimension validation,
//! stride derivation for the op-agnostic kernel, batch chunking, and a CPU
//! reference implementation of the cuBLAS column-major semantics.

use std::ffi::c_int;

pub const SUCCESS: c_int = 0;
pub const NOT_INITIALIZED: c_int = 1;
pub const ALLOC_FAILED: c_int = 3;
pub const INVALID_VALUE: c_int = 7;
pub const MAPPING_ERROR: c_int = 11;
pub const EXECUTION_FAILED: c_int = 13;
pub const NOT_SUPPORTED: c_int = 15;

/// cublasOperation_t
pub const OP_N: c_int = 0;
pub const OP_T: c_int = 1;
pub const OP_C: c_int = 2;

/// cudaDataType_t values used by ggml.
pub const CUDA_R_32F: c_int = 0;
pub const CUDA_R_16F: c_int = 2;
pub const CUDA_R_16BF: c_int = 14;

/// cublasComputeType_t
pub const COMPUTE_16F: c_int = 64;
pub const COMPUTE_16F_PEDANTIC: c_int = 65;
pub const COMPUTE_32F: c_int = 68;
pub const COMPUTE_32F_PEDANTIC: c_int = 69;
pub const COMPUTE_32F_FAST_16F: c_int = 74;
pub const COMPUTE_32F_FAST_16BF: c_int = 75;
pub const COMPUTE_32F_FAST_TF32: c_int = 77;

/// cublasPointerMode_t
pub const POINTER_MODE_HOST: c_int = 0;
pub const POINTER_MODE_DEVICE: c_int = 1;

/// Largest gridDim.z the launcher uses per launch; larger batches are chunked.
pub const MAX_GRID_Z: u32 = 65535;
/// Largest gridDim.y (columns of C / 16).
pub const MAX_GRID_Y: u32 = 65535;
pub const TILE: u32 = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    F16,
    Bf16,
}

impl DType {
    pub fn from_cuda(code: c_int) -> Option<Self> {
        match code {
            CUDA_R_32F => Some(Self::F32),
            CUDA_R_16F => Some(Self::F16),
            CUDA_R_16BF => Some(Self::Bf16),
            _ => None,
        }
    }
    pub fn size(self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 | Self::Bf16 => 2,
        }
    }
    pub fn tag(self) -> &'static str {
        match self {
            Self::F32 => "f32",
            Self::F16 => "f16",
            Self::Bf16 => "bf16",
        }
    }
}

/// Scalar (alpha/beta) type: follows the compute type.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScalarKind {
    F16,
    F32,
}

pub fn scalar_kind(compute_type: c_int) -> Option<ScalarKind> {
    match compute_type {
        COMPUTE_16F | COMPUTE_16F_PEDANTIC => Some(ScalarKind::F16),
        COMPUTE_32F | COMPUTE_32F_PEDANTIC | COMPUTE_32F_FAST_16F | COMPUTE_32F_FAST_16BF
        | COMPUTE_32F_FAST_TF32 => Some(ScalarKind::F32),
        // Legacy callers pass a cudaDataType as the compute type.
        CUDA_R_32F => Some(ScalarKind::F32),
        CUDA_R_16F => Some(ScalarKind::F16),
        _ => None,
    }
}

/// True for ops that read the stored matrix transposed (T and C; C == T for reals).
pub fn op_is_transposed(op: c_int) -> Option<bool> {
    match op {
        OP_N => Some(false),
        OP_T | OP_C => Some(true),
        _ => None,
    }
}

/// One validated GEMM: C(m x n) = alpha * op(A)(m x k) * op(B)(k x n) + beta * C.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmDesc {
    pub trans_a: bool,
    pub trans_b: bool,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub lda: u32,
    pub ldb: u32,
    pub ldc: u32,
    pub a: DType,
    pub b: DType,
    pub c: DType,
    pub scalar: ScalarKind,
}

/// Element strides the op-agnostic kernel uses:
/// `A(i,l) = A[i*a_si + l*a_sl]`, `B(l,j) = B[l*b_sl + j*b_sj]`, `C(i,j) = C[i + j*ldc]`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Strides {
    pub a_si: u32,
    pub a_sl: u32,
    pub b_sl: u32,
    pub b_sj: u32,
}

impl GemmDesc {
    pub fn strides(&self) -> Strides {
        let (a_si, a_sl) = if self.trans_a { (self.lda, 1) } else { (1, self.lda) };
        let (b_sl, b_sj) = if self.trans_b { (self.ldb, 1) } else { (1, self.ldb) };
        Strides { a_si, a_sl, b_sl, b_sj }
    }

    /// Kernel variant name, e.g. `cfi_gemm_f16f16f32s` (`s` strided, `p` pointer arrays).
    pub fn kernel_name(&self, pointer_arrays: bool) -> String {
        kernel_name(self.a, self.b, self.c, pointer_arrays)
    }
}

pub fn kernel_name(a: DType, b: DType, c: DType, pointer_arrays: bool) -> String {
    format!(
        "cfi_gemm_{}{}{}{}",
        a.tag(),
        b.tag(),
        c.tag(),
        if pointer_arrays { "p" } else { "s" }
    )
}

/// Validate a cuBLAS GemmEx-style parameter set exactly as cuBLAS does
/// (INVALID_VALUE for negative sizes and short leading dimensions), and reject
/// type combinations cuBLAS itself lacks (NOT_SUPPORTED).
#[allow(clippy::too_many_arguments)]
pub fn validate(
    transa: c_int,
    transb: c_int,
    m: c_int,
    n: c_int,
    k: c_int,
    lda: c_int,
    ldb: c_int,
    ldc: c_int,
    a_type: c_int,
    b_type: c_int,
    c_type: c_int,
    compute_type: c_int,
) -> Result<GemmDesc, c_int> {
    let trans_a = op_is_transposed(transa).ok_or(INVALID_VALUE)?;
    let trans_b = op_is_transposed(transb).ok_or(INVALID_VALUE)?;
    if m < 0 || n < 0 || k < 0 {
        return Err(INVALID_VALUE);
    }
    let rows_a = if trans_a { k } else { m };
    let rows_b = if trans_b { n } else { k };
    if lda < rows_a.max(1) || ldb < rows_b.max(1) || ldc < m.max(1) {
        return Err(INVALID_VALUE);
    }
    let a = DType::from_cuda(a_type).ok_or(NOT_SUPPORTED)?;
    let b = DType::from_cuda(b_type).ok_or(NOT_SUPPORTED)?;
    let c = DType::from_cuda(c_type).ok_or(NOT_SUPPORTED)?;
    let scalar = scalar_kind(compute_type).ok_or(NOT_SUPPORTED)?;
    // cuBLAS requires matching A/B types; f32 inputs produce f32; 16F compute is f16-only.
    if a != b {
        return Err(NOT_SUPPORTED);
    }
    match (a, c) {
        (DType::F32, DType::F32) => {}
        (DType::F16, DType::F16 | DType::F32) => {}
        (DType::Bf16, DType::Bf16 | DType::F32) => {}
        _ => return Err(NOT_SUPPORTED),
    }
    if scalar == ScalarKind::F16 && (a != DType::F16 || c != DType::F16) {
        return Err(NOT_SUPPORTED);
    }
    Ok(GemmDesc {
        trans_a,
        trans_b,
        m: m as u32,
        n: n as u32,
        k: k as u32,
        lda: lda as u32,
        ldb: ldb as u32,
        ldc: ldc as u32,
        a,
        b,
        c,
        scalar,
    })
}

/// Split `batch` into `(start, count)` launches no taller than [`MAX_GRID_Z`].
pub fn batch_chunks(batch: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut start = 0;
    while start < batch {
        let count = (batch - start).min(MAX_GRID_Z);
        out.push((start, count));
        start += count;
    }
    out
}

/// Launch grid `(x, y)` for the 16x16 tiled kernel, or None if C is too wide.
pub fn grid_xy(m: u32, n: u32) -> Option<(u32, u32)> {
    let gy = n.div_ceil(TILE).max(1);
    if gy > MAX_GRID_Y {
        return None;
    }
    Some((m.div_ceil(TILE).max(1), gy))
}

// ---------------------------------------------------------------------------
// Number formats
// ---------------------------------------------------------------------------

pub fn f16_bits_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let mant = (bits & 0x3ff) as u32;
    let out = if exp == 0 {
        if mant == 0 {
            sign << 31
        } else {
            let mut m = mant;
            let mut e = 127 - 15 + 1;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            (sign << 31) | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        (sign << 31) | 0x7f80_0000 | (mant << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(out)
}

/// Round-to-nearest-even f32 -> binary16.
pub fn f32_to_f16_bits(v: f32) -> u16 {
    let x = v.to_bits();
    let sign = ((x >> 16) & 0x8000) as u16;
    let exp = ((x >> 23) & 0xff) as i32;
    let mant = x & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 31 {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut half = (m >> shift) as u16;
        let rem = m & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rem > halfway || (rem == halfway && half & 1 == 1) {
            half += 1;
        }
        return sign | half;
    }
    let mut half = sign | ((e as u16) << 10) | ((mant >> 13) as u16);
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && half & 1 == 1) {
        half += 1; // may carry into the exponent, which is correct
    }
    half
}

pub fn bf16_bits_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Round-to-nearest-even f32 -> bfloat16 (same integer recipe the PTX uses).
pub fn f32_to_bf16_bits(v: f32) -> u16 {
    if v.is_nan() {
        return 0x7fc0;
    }
    let x = v.to_bits();
    let r = x.wrapping_add(0x7fff + ((x >> 16) & 1));
    (r >> 16) as u16
}

pub fn decode(dtype: DType, bytes: &[u8], index: usize) -> f32 {
    match dtype {
        DType::F32 => f32::from_le_bytes(bytes[index * 4..index * 4 + 4].try_into().unwrap()),
        DType::F16 => f16_bits_to_f32(u16::from_le_bytes(
            bytes[index * 2..index * 2 + 2].try_into().unwrap(),
        )),
        DType::Bf16 => bf16_bits_to_f32(u16::from_le_bytes(
            bytes[index * 2..index * 2 + 2].try_into().unwrap(),
        )),
    }
}

pub fn encode(dtype: DType, bytes: &mut [u8], index: usize, v: f32) {
    match dtype {
        DType::F32 => bytes[index * 4..index * 4 + 4].copy_from_slice(&v.to_le_bytes()),
        DType::F16 => {
            bytes[index * 2..index * 2 + 2].copy_from_slice(&f32_to_f16_bits(v).to_le_bytes())
        }
        DType::Bf16 => {
            bytes[index * 2..index * 2 + 2].copy_from_slice(&f32_to_bf16_bits(v).to_le_bytes())
        }
    }
}

// ---------------------------------------------------------------------------
// CPU references
// ---------------------------------------------------------------------------

/// Literal cuBLAS semantics (column-major, `op()` applied to the stored matrix),
/// independent of the stride derivation. `a`, `b`, `c` are raw storage bytes.
pub fn reference_gemm(d: &GemmDesc, alpha: f32, beta: f32, a: &[u8], b: &[u8], c: &mut [u8]) {
    for j in 0..d.n as usize {
        for i in 0..d.m as usize {
            let mut acc = 0.0f32;
            for l in 0..d.k as usize {
                // op(A)(i,l): N reads A[i + l*lda]; T reads A[l + i*lda].
                let ai = if d.trans_a { l + i * d.lda as usize } else { i + l * d.lda as usize };
                // op(B)(l,j): N reads B[l + j*ldb]; T reads B[j + l*ldb].
                let bi = if d.trans_b { j + l * d.ldb as usize } else { l + j * d.ldb as usize };
                acc += decode(d.a, a, ai) * decode(d.b, b, bi);
            }
            let ci = i + j * d.ldc as usize;
            let prior = if beta == 0.0 { 0.0 } else { beta * decode(d.c, c, ci) };
            encode(d.c, c, ci, alpha * acc + prior);
        }
    }
}

/// Emulates exactly what the generated kernel computes from [`Strides`] (same
/// accumulation order, `beta == 0` never reads C).
pub fn emulate_kernel(d: &GemmDesc, alpha: f32, beta: f32, a: &[u8], b: &[u8], c: &mut [u8]) {
    let s = d.strides();
    for j in 0..d.n as usize {
        for i in 0..d.m as usize {
            let mut acc = 0.0f32;
            for l in 0..d.k as usize {
                let av = decode(d.a, a, i * s.a_si as usize + l * s.a_sl as usize);
                let bv = decode(d.b, b, l * s.b_sl as usize + j * s.b_sj as usize);
                acc += av * bv;
            }
            let ci = i + j * d.ldc as usize;
            let mut v = acc * alpha;
            if beta != 0.0 {
                v += beta * decode(d.c, c, ci);
            }
            encode(d.c, c, ci, v);
        }
    }
}


// ---------------------------------------------------------------------------
// Triangular solve (cublasStrsmBatched)
// ---------------------------------------------------------------------------

pub const SIDE_LEFT: c_int = 0;
pub const SIDE_RIGHT: c_int = 1;
pub const FILL_LOWER: c_int = 0;
pub const FILL_UPPER: c_int = 1;
pub const DIAG_NON_UNIT: c_int = 0;
pub const DIAG_UNIT: c_int = 1;

/// Kernel-level description of a batched TRSM (see `ptxgen::generate_trsm`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TrsmPlan {
    pub len: u32,
    pub nvec: u32,
    pub a_rs: u32,
    pub a_cs: u32,
    pub b_vs: u32,
    pub b_es: u32,
    pub lower: bool,
    pub unit: bool,
}

#[allow(clippy::too_many_arguments)]
pub fn trsm_plan(
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: c_int,
    n: c_int,
    lda: c_int,
    ldb: c_int,
) -> Result<TrsmPlan, c_int> {
    let right = match side {
        SIDE_LEFT => false,
        SIDE_RIGHT => true,
        _ => return Err(INVALID_VALUE),
    };
    let lower_stored = match uplo {
        FILL_LOWER => true,
        FILL_UPPER => false,
        _ => return Err(INVALID_VALUE),
    };
    let mut t = op_is_transposed(trans).ok_or(INVALID_VALUE)?;
    let unit = match diag {
        DIAG_NON_UNIT => false,
        DIAG_UNIT => true,
        _ => return Err(INVALID_VALUE),
    };
    if m < 0 || n < 0 {
        return Err(INVALID_VALUE);
    }
    let order = if right { n } else { m };
    if lda < order.max(1) || ldb < m.max(1) {
        return Err(INVALID_VALUE);
    }
    // X op(A) = B  <=>  op(A)^T x = b per row of B.
    if right {
        t = !t;
    }
    let (a_rs, a_cs) = if t { (lda as u32, 1) } else { (1, lda as u32) };
    let (b_vs, b_es) = if right { (1, ldb as u32) } else { (ldb as u32, 1) };
    Ok(TrsmPlan {
        len: order as u32,
        nvec: if right { m } else { n } as u32,
        a_rs,
        a_cs,
        b_vs,
        b_es,
        lower: lower_stored != t,
        unit,
    })
}

/// Emulates the TRSM kernel on one matrix pair (same order of operations).
pub fn emulate_trsm(p: &TrsmPlan, alpha: f32, a: &[f32], b: &mut [f32]) {
    for v in 0..p.nvec as usize {
        let base = v * p.b_vs as usize;
        for it in 0..p.len as usize {
            let r = if p.lower { it } else { p.len as usize - 1 - it };
            let ri = base + r * p.b_es as usize;
            let mut acc = b[ri] * alpha;
            for t in 0..it {
                let c = if p.lower { t } else { p.len as usize - 1 - t };
                let tv = a[r * p.a_rs as usize + c * p.a_cs as usize];
                acc = (-tv).mul_add(b[base + c * p.b_es as usize], acc);
            }
            if !p.unit {
                acc /= a[r * p.a_rs as usize + r * p.a_cs as usize];
            }
            b[ri] = acc;
        }
    }
}

/// Literal cuBLAS semantics (column-major, `op()` of the stored triangle,
/// `side`) solved by dense forward/back substitution, independent of
/// [`trsm_plan`]. Entries outside the stored triangle are ignored.
#[allow(clippy::too_many_arguments)]
pub fn reference_trsm(
    side: c_int,
    uplo: c_int,
    trans: c_int,
    diag: c_int,
    m: usize,
    n: usize,
    lda: usize,
    ldb: usize,
    alpha: f32,
    a: &[f32],
    b: &mut [f32],
) {
    let order = if side == SIDE_LEFT { m } else { n };
    // Dense op(A) with the unreferenced triangle zeroed.
    let mut t = vec![0.0f32; order * order];
    for r in 0..order {
        for c in 0..order {
            let (sr, sc) = if trans == OP_N { (r, c) } else { (c, r) };
            let stored = if uplo == FILL_LOWER { sr >= sc } else { sr <= sc };
            let mut v = if stored { a[sr + sc * lda] } else { 0.0 };
            if diag == DIAG_UNIT && sr == sc {
                v = 1.0;
            }
            t[r + c * order] = v;
        }
    }
    let lower = (0..order).all(|r| (r + 1..order).all(|c| t[r + c * order] == 0.0));
    let solve = |rhs: &mut Vec<f32>, tm: &dyn Fn(usize, usize) -> f32, lower: bool| {
        let idx: Vec<usize> = if lower { (0..order).collect() } else { (0..order).rev().collect() };
        for (pos, &r) in idx.iter().enumerate() {
            let mut acc = rhs[r];
            for &c in &idx[..pos] {
                acc -= tm(r, c) * rhs[c];
            }
            rhs[r] = acc / tm(r, r);
        }
    };
    if side == SIDE_LEFT {
        for j in 0..n {
            let mut x: Vec<f32> = (0..m).map(|i| alpha * b[i + j * ldb]).collect();
            solve(&mut x, &|r, c| t[r + c * order], lower);
            for i in 0..m {
                b[i + j * ldb] = x[i];
            }
        }
    } else {
        // x^T op(A) = alpha b^T  =>  op(A)^T x = alpha b
        for i in 0..m {
            let mut x: Vec<f32> = (0..n).map(|j| alpha * b[i + j * ldb]).collect();
            solve(&mut x, &|r, c| t[c + r * order], !lower);
            for j in 0..n {
                b[i + j * ldb] = x[j];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lcg(seed: &mut u64) -> f32 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (((*seed >> 33) % 2001) as f32 - 1000.0) / 250.0
    }

    fn fill(dtype: DType, elems: usize, seed: &mut u64) -> Vec<u8> {
        let mut bytes = vec![0u8; elems * dtype.size()];
        for i in 0..elems {
            let v = lcg(seed);
            encode(dtype, &mut bytes, i, v);
        }
        bytes
    }

    #[test]
    fn f16_roundtrip_and_rounding() {
        for bits in 0u16..=0xffff {
            let f = f16_bits_to_f32(bits);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16_bits(f), bits, "bits {bits:#06x}");
        }
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00); // rounds to inf
        assert_eq!(f32_to_f16_bits(1.0 + 1.0 / 2048.0), 0x3c00); // tie to even
        assert_eq!(f32_to_f16_bits(1.0 + 3.0 / 2048.0), 0x3c02);
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8000)), 0x3f80); // tie even
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f81_8000)), 0x3f82); // tie up
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x3f80_8001)), 0x3f81);
        assert_eq!(f32_to_bf16_bits(f32::NAN), 0x7fc0);
        assert_eq!(f32_to_bf16_bits(f32::INFINITY), 0x7f80);
        assert_eq!(f32_to_bf16_bits(f32::from_bits(0x7f7f_ffff)), 0x7f80); // overflow to inf
        assert_eq!(bf16_bits_to_f32(0xc000), -2.0);
    }

    #[test]
    fn compute_type_decoding() {
        assert_eq!(scalar_kind(COMPUTE_16F), Some(ScalarKind::F16));
        assert_eq!(scalar_kind(COMPUTE_32F), Some(ScalarKind::F32));
        assert_eq!(scalar_kind(COMPUTE_32F_FAST_16F), Some(ScalarKind::F32));
        assert_eq!(scalar_kind(COMPUTE_32F_FAST_TF32), Some(ScalarKind::F32));
        assert_eq!(scalar_kind(CUDA_R_32F), Some(ScalarKind::F32));
        assert_eq!(scalar_kind(CUDA_R_16F), Some(ScalarKind::F16));
        assert_eq!(scalar_kind(70), None); // 64F
        assert_eq!(scalar_kind(72), None); // 32I
    }

    #[test]
    fn validate_ggml_call_shapes() {
        // ggml fp16 path: A,B f16, C f32, compute 32F, T x N.
        let d = validate(OP_T, OP_N, 64, 8, 32, 32, 32, 64, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, COMPUTE_32F)
            .unwrap();
        assert!(d.trans_a && !d.trans_b);
        assert_eq!((d.a, d.c, d.scalar), (DType::F16, DType::F32, ScalarKind::F32));
        // bf16 in/out.
        assert!(validate(OP_T, OP_N, 4, 4, 4, 4, 4, 4, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, COMPUTE_32F).is_ok());
        // fp16 compute with f16 C.
        let d = validate(OP_T, OP_N, 4, 4, 4, 4, 4, 4, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, COMPUTE_16F).unwrap();
        assert_eq!(d.scalar, ScalarKind::F16);
        // 16F compute with f32 C is not a cuBLAS combination.
        assert_eq!(
            validate(OP_T, OP_N, 4, 4, 4, 4, 4, 4, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, COMPUTE_16F),
            Err(NOT_SUPPORTED)
        );
        // Mismatched A/B types.
        assert_eq!(
            validate(OP_N, OP_N, 4, 4, 4, 4, 4, 4, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F, COMPUTE_32F),
            Err(NOT_SUPPORTED)
        );
        // f32 A with f16 C.
        assert_eq!(
            validate(OP_N, OP_N, 4, 4, 4, 4, 4, 4, CUDA_R_32F, CUDA_R_32F, CUDA_R_16F, COMPUTE_32F),
            Err(NOT_SUPPORTED)
        );
    }

    #[test]
    fn validate_rejects_bad_dims() {
        let f = |ta, tb, m, n, k, lda, ldb, ldc| {
            validate(ta, tb, m, n, k, lda, ldb, ldc, 0, 0, 0, COMPUTE_32F)
        };
        assert_eq!(f(9, OP_N, 1, 1, 1, 1, 1, 1), Err(INVALID_VALUE));
        assert_eq!(f(OP_N, OP_N, -1, 1, 1, 1, 1, 1), Err(INVALID_VALUE));
        // N: lda >= m. T: lda >= k.
        assert_eq!(f(OP_N, OP_N, 8, 2, 4, 7, 4, 8), Err(INVALID_VALUE));
        assert!(f(OP_N, OP_N, 8, 2, 4, 8, 4, 8).is_ok());
        assert_eq!(f(OP_T, OP_N, 8, 2, 4, 3, 4, 8), Err(INVALID_VALUE));
        assert!(f(OP_T, OP_N, 8, 2, 4, 4, 4, 8).is_ok());
        // B: N: ldb >= k. T: ldb >= n.
        assert_eq!(f(OP_N, OP_N, 8, 2, 4, 8, 3, 8), Err(INVALID_VALUE));
        assert_eq!(f(OP_N, OP_T, 8, 5, 4, 8, 4, 8), Err(INVALID_VALUE));
        assert!(f(OP_N, OP_T, 8, 5, 4, 8, 5, 8).is_ok());
        assert_eq!(f(OP_N, OP_N, 8, 2, 4, 8, 4, 7), Err(INVALID_VALUE));
        // Empty problems still need ld >= 1.
        assert!(f(OP_N, OP_N, 0, 0, 0, 1, 1, 1).is_ok());
        assert_eq!(f(OP_N, OP_N, 0, 0, 0, 0, 1, 1), Err(INVALID_VALUE));
        // Conjugate transpose behaves as transpose for real types.
        assert!(f(OP_C, OP_C, 2, 3, 4, 4, 3, 2).unwrap().trans_a);
    }

    #[test]
    fn strides_match_op_semantics() {
        let d = validate(OP_T, OP_N, 5, 3, 7, 9, 11, 6, 0, 0, 0, COMPUTE_32F).unwrap();
        assert_eq!(d.strides(), Strides { a_si: 9, a_sl: 1, b_sl: 1, b_sj: 11 });
        let d = validate(OP_N, OP_T, 5, 3, 7, 9, 4, 6, 0, 0, 0, COMPUTE_32F).unwrap();
        assert_eq!(d.strides(), Strides { a_si: 1, a_sl: 9, b_sl: 4, b_sj: 1 });
    }

    #[test]
    fn kernel_matches_reference_all_ops_types_and_padding() {
        let type_sets = [
            (CUDA_R_32F, CUDA_R_32F, COMPUTE_32F),
            (CUDA_R_16F, CUDA_R_16F, COMPUTE_32F),
            (CUDA_R_16F, CUDA_R_32F, COMPUTE_32F),
            (CUDA_R_16F, CUDA_R_16F, COMPUTE_16F),
            (CUDA_R_16BF, CUDA_R_16BF, COMPUTE_32F),
            (CUDA_R_16BF, CUDA_R_32F, COMPUTE_32F_FAST_16BF),
        ];
        let mut seed = 7u64;
        for &(ab, cty, compute) in &type_sets {
            for ta in [OP_N, OP_T] {
                for tb in [OP_N, OP_T] {
                    for &(m, n, k) in &[(1, 1, 1), (5, 3, 7), (17, 16, 33), (4, 9, 0)] {
                        for pad in [0, 3] {
                            let rows_a = if ta == OP_N { m } else { k };
                            let cols_a = if ta == OP_N { k } else { m };
                            let rows_b = if tb == OP_N { k } else { n };
                            let cols_b = if tb == OP_N { n } else { k };
                            let (lda, ldb, ldc) = (rows_a.max(1) + pad, rows_b.max(1) + pad, m.max(1) + pad);
                            let d = validate(ta, tb, m, n, k, lda, ldb, ldc, ab, ab, cty, compute).unwrap();
                            let a = fill(d.a, (lda * cols_a.max(1)) as usize, &mut seed);
                            let b = fill(d.b, (ldb * cols_b.max(1)) as usize, &mut seed);
                            let c0 = fill(d.c, (ldc * n.max(1)) as usize, &mut seed);
                            for (alpha, beta) in [(1.0, 0.0), (0.5, 1.0), (-2.0, 0.25)] {
                                let mut c_ref = c0.clone();
                                let mut c_ker = c0.clone();
                                reference_gemm(&d, alpha, beta, &a, &b, &mut c_ref);
                                emulate_kernel(&d, alpha, beta, &a, &b, &mut c_ker);
                                assert_eq!(c_ref, c_ker, "ta={ta} tb={tb} m={m} n={n} k={k} pad={pad} ab={ab} c={cty}");
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn beta_zero_ignores_nan_in_c() {
        let d = validate(OP_N, OP_N, 2, 2, 2, 2, 2, 2, 0, 0, 0, COMPUTE_32F).unwrap();
        let a = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let b = [1.0f32, 0.0, 0.0, 1.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let mut c = vec![0xffu8; 16]; // NaN pattern
        emulate_kernel(&d, 1.0, 0.0, &a, &b, &mut c);
        assert_eq!(decode(DType::F32, &c, 0), 1.0);
        assert_eq!(decode(DType::F32, &c, 3), 4.0);
    }

    #[test]
    fn known_small_product() {
        // A = [[1,2],[3,4]] column-major [1,3,2,4]; B = identity-swap.
        let d = validate(OP_N, OP_N, 2, 2, 2, 2, 2, 2, 0, 0, 0, COMPUTE_32F).unwrap();
        let a = [1.0f32, 3.0, 2.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let b = [0.0f32, 1.0, 1.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>();
        let mut c = vec![0u8; 16];
        reference_gemm(&d, 1.0, 0.0, &a, &b, &mut c);
        let got: Vec<f32> = (0..4).map(|i| decode(DType::F32, &c, i)).collect();
        assert_eq!(got, vec![2.0, 4.0, 1.0, 3.0]); // columns swapped
    }

    #[test]
    fn batch_chunking_and_grid() {
        assert_eq!(batch_chunks(0), vec![]);
        assert_eq!(batch_chunks(5), vec![(0, 5)]);
        assert_eq!(batch_chunks(65535), vec![(0, 65535)]);
        assert_eq!(batch_chunks(65536), vec![(0, 65535), (65535, 1)]);
        assert_eq!(batch_chunks(131070 + 7), vec![(0, 65535), (65535, 65535), (131070, 7)]);
        assert_eq!(grid_xy(17, 16), Some((2, 1)));
        assert_eq!(grid_xy(0, 0), Some((1, 1)));
        assert_eq!(grid_xy(16, 65535 * 16 + 1), None);
    }

    #[test]
    fn kernel_names() {
        assert_eq!(kernel_name(DType::F16, DType::F16, DType::F32, false), "cfi_gemm_f16f16f32s");
        assert_eq!(kernel_name(DType::Bf16, DType::Bf16, DType::Bf16, true), "cfi_gemm_bf16bf16bf16p");
    }

    #[test]
    fn trsm_plan_matches_literal_semantics() {
        let mut seed = 7u64;
        for side in [SIDE_LEFT, SIDE_RIGHT] {
            for uplo in [FILL_LOWER, FILL_UPPER] {
                for trans in [OP_N, OP_T] {
                    for diag in [DIAG_NON_UNIT, DIAG_UNIT] {
                        let (m, n) = (5usize, 3usize);
                        let order = if side == SIDE_LEFT { m } else { n };
                        let (lda, ldb) = (order + 2, m + 1);
                        let mut a: Vec<f32> = (0..lda * order).map(|_| lcg(&mut seed)).collect();
                        for d in 0..order {
                            a[d + d * lda] = 4.0 + (d as f32); // well conditioned
                        }
                        let b0: Vec<f32> = (0..ldb * n).map(|_| lcg(&mut seed)).collect();
                        let mut want = b0.clone();
                        reference_trsm(side, uplo, trans, diag, m, n, lda, ldb, 1.5, &a, &mut want);
                        let p = trsm_plan(side, uplo, trans, diag, m as c_int, n as c_int, lda as c_int, ldb as c_int).unwrap();
                        let mut got = b0.clone();
                        emulate_trsm(&p, 1.5, &a, &mut got);
                        for (g, w) in got.iter().zip(&want) {
                            assert!((g - w).abs() <= 1e-4 * (1.0 + w.abs()), "{side}{uplo}{trans}{diag}: {g} vs {w}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn trsm_plan_validation() {
        assert_eq!(trsm_plan(2, 0, 0, 0, 2, 2, 2, 2), Err(INVALID_VALUE));
        assert_eq!(trsm_plan(0, 0, 0, 0, 4, 2, 3, 4), Err(INVALID_VALUE)); // lda < m
        assert_eq!(trsm_plan(1, 0, 0, 0, 4, 2, 2, 3), Err(INVALID_VALUE)); // ldb < m
        assert_eq!(trsm_plan(1, 0, 0, 0, 4, 2, 2, 4).unwrap().nvec, 4);
    }
}
