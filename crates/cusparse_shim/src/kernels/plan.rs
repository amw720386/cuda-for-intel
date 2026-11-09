//! Pure cuSPARSE logic for the shim: enum decoding, descriptors, and the translation of
//! SpMV / SpMM / SDDMM / BSR calls into kernel launches (`ptxgen`). No driver calls, so
//! `src/ptx/cusparse_tests.rs` runs the same launch lists in the SPIR-V simulator.
use super::ptxgen::{Kernel, VBLOCK, Variant};

pub const SUCCESS: i32 = 0;
pub const NOT_INITIALIZED: i32 = 1;
pub const ALLOC_FAILED: i32 = 2;
pub const INVALID_VALUE: i32 = 3;
pub const EXECUTION_FAILED: i32 = 6;
pub const INTERNAL_ERROR: i32 = 7;
pub const MATRIX_TYPE_NOT_SUPPORTED: i32 = 8;
pub const NOT_SUPPORTED: i32 = 10;

// cudaDataType
pub const R_32F: i32 = 0;
pub const R_64F: i32 = 1;
// cusparseIndexType_t
pub const INDEX_16U: i32 = 1;
pub const INDEX_32I: i32 = 2;
pub const INDEX_64I: i32 = 3;
// cusparseOrder_t
pub const ORDER_COL: i32 = 1;
pub const ORDER_ROW: i32 = 2;
// cusparseOperation_t
pub const OP_N: i32 = 0;
pub const OP_T: i32 = 1;
pub const OP_C: i32 = 2;
// cusparseFormat_t
pub const FORMAT_CSR: i32 = 1;
pub const FORMAT_CSC: i32 = 2;
pub const FORMAT_COO: i32 = 3;

/// Threads per block for the one-thread-per-output kernels.
pub const BLOCK: u32 = 256;

/// Rows at least this long on average use the vector (block-per-row) kernel.
pub const VECTOR_MIN_AVG: i64 = 32;

pub fn val_double(t: i32) -> Result<bool, i32> {
    match t {
        R_32F => Ok(false),
        R_64F => Ok(true),
        _ => Err(NOT_SUPPORTED),
    }
}

pub fn index_wide(t: i32) -> Result<bool, i32> {
    match t {
        INDEX_32I => Ok(false),
        INDEX_64I => Ok(true),
        _ => Err(NOT_SUPPORTED),
    }
}

pub fn valid_index_type(t: i32) -> bool {
    matches!(t, INDEX_16U | INDEX_32I | INDEX_64I)
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpMat {
    pub format: i32,
    pub rows: i64,
    pub cols: i64,
    pub nnz: i64,
    /// CSR row offsets, CSC column offsets, or COO row indices.
    pub ptr: u64,
    /// CSR column indices, CSC row indices, or COO column indices.
    pub idx: u64,
    pub val: u64,
    pub ptr_ty: i32,
    pub idx_ty: i32,
    pub base: i32,
    pub val_ty: i32,
    pub batch: i32,
    /// Batch stride of `ptr` for CSR/CSC (elements); COO uses `idx_bstride` for all three.
    pub ptr_bstride: i64,
    pub idx_bstride: i64,
    pub fill: i32,
    pub diag: i32,
}

impl SpMat {
    #[allow(clippy::too_many_arguments)]
    pub fn new(format: i32, rows: i64, cols: i64, nnz: i64, ptr: u64, idx: u64, val: u64, ptr_ty: i32, idx_ty: i32, base: i32, val_ty: i32) -> Result<SpMat, i32> {
        if rows < 0 || cols < 0 || nnz < 0 || !valid_index_type(ptr_ty) || !valid_index_type(idx_ty) || !matches!(base, 0 | 1) {
            return Err(INVALID_VALUE);
        }
        let ptr_bstride = match format {
            FORMAT_CSR => rows + 1,
            FORMAT_CSC => cols + 1,
            _ => nnz,
        };
        Ok(SpMat { format, rows, cols, nnz, ptr, idx, val, ptr_ty, idx_ty, base, val_ty, batch: 1, ptr_bstride, idx_bstride: nnz, fill: 0, diag: 0 })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DnMat {
    pub rows: i64,
    pub cols: i64,
    pub ld: i64,
    pub val: u64,
    pub val_ty: i32,
    pub order: i32,
    pub batch: i32,
    pub bstride: i64,
}

impl DnMat {
    pub fn new(rows: i64, cols: i64, ld: i64, val: u64, val_ty: i32, order: i32) -> Result<DnMat, i32> {
        if rows < 0 || cols < 0 || !matches!(order, ORDER_COL | ORDER_ROW) {
            return Err(INVALID_VALUE);
        }
        let lead = if order == ORDER_COL { rows } else { cols };
        if ld < lead.max(1) {
            return Err(INVALID_VALUE);
        }
        let other = if order == ORDER_COL { cols } else { rows };
        Ok(DnMat { rows, cols, ld, val, val_ty, order, batch: 1, bstride: ld * other })
    }

    /// A dense vector as an `n x 1` column-major matrix.
    pub fn vector(size: i64, val: u64, val_ty: i32) -> Result<DnMat, i32> {
        DnMat::new(size, 1, size.max(1), val, val_ty, ORDER_COL)
    }

    /// Element strides (row, column) of the stored matrix.
    pub fn strides(&self) -> (u64, u64) {
        if self.order == ORDER_COL { (1, self.ld as u64) } else { (self.ld as u64, 1) }
    }

    /// Strides of element (i, j) of `op(self)`.
    pub fn op_strides(&self, op: i32) -> (u64, u64) {
        let (r, c) = self.strides();
        if op == OP_N { (r, c) } else { (c, r) }
    }

    /// Shape of `op(self)`.
    pub fn op_shape(&self, op: i32) -> (i64, i64) {
        if op == OP_N { (self.rows, self.cols) } else { (self.cols, self.rows) }
    }
}

/// alpha/beta: bit patterns (host pointer mode) or device pointers (device pointer mode).
#[derive(Clone, Copy, Debug, Default)]
pub struct Scalars {
    pub alpha: u64,
    pub beta: u64,
    pub alpha_ptr: u64,
    pub beta_ptr: u64,
}

impl Scalars {
    pub fn host(alpha: f64, beta: f64, double: bool) -> Scalars {
        let bits = |v: f64| if double { v.to_bits() } else { u64::from((v as f32).to_bits()) };
        Scalars { alpha: bits(alpha), beta: bits(beta), alpha_ptr: 0, beta_ptr: 0 }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Choice {
    Auto,
    Scalar,
    Vector,
}

/// One kernel launch: `x` blocks of `block` threads, `batch` blocks in y.
#[derive(Clone, Debug, PartialEq)]
pub struct Launch {
    pub kernel: Kernel,
    pub variant: Variant,
    pub blocks: u64,
    pub block: u32,
    pub batch: u32,
    pub args: Vec<u64>,
}

fn threads(kernel: Kernel, variant: Variant, total: u64, batch: u32, args: Vec<u64>) -> Launch {
    Launch { kernel, variant, blocks: total.div_ceil(u64::from(BLOCK)), block: BLOCK, batch, args }
}

const MAX_BLOCKS: u64 = (1 << 31) - 1;

fn check_launches(v: Vec<Launch>) -> Result<Vec<Launch>, i32> {
    if v.iter().any(|l| l.blocks > MAX_BLOCKS || l.batch > 65535) { Err(NOT_SUPPORTED) } else { Ok(v) }
}

fn batch_of(n: i32, want: i32, stride: i64) -> Result<u64, i32> {
    if n == want {
        Ok(stride as u64)
    } else if n == 1 {
        Ok(0)
    } else {
        Err(INVALID_VALUE)
    }
}

/// `C = alpha * op(A) * op(B) + beta * C` (SpMV is the `n = 1` case).
#[allow(clippy::too_many_arguments)]
pub fn spmm(op_a: i32, op_b: i32, a: &SpMat, b: &DnMat, c: &DnMat, compute: i32, s: Scalars, choice: Choice) -> Result<Vec<Launch>, i32> {
    if !matches!(op_a, OP_N | OP_T | OP_C) || !matches!(op_b, OP_N | OP_T | OP_C) {
        return Err(INVALID_VALUE);
    }
    let double = val_double(a.val_ty)?;
    if b.val_ty != a.val_ty || c.val_ty != a.val_ty || compute != a.val_ty {
        return Err(NOT_SUPPORTED);
    }
    let (m, k) = if op_a == OP_N { (a.rows, a.cols) } else { (a.cols, a.rows) };
    let (bk, n) = b.op_shape(op_b);
    if bk != k || c.rows != m || c.cols != n {
        return Err(INVALID_VALUE);
    }
    let batch = c.batch.max(1);
    let a_scale = batch_of(a.batch, batch, 1)?;
    let b_bs = batch_of(b.batch, batch, b.bstride)?;
    let c_bs = c.bstride as u64;
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let (bsi, bsj) = b.op_strides(op_b);
    let (csi, csj) = c.strides();
    let base = a.base as u64;
    let transposed = op_a != OP_N;
    let (gather, storage_rows) = match a.format {
        FORMAT_CSR => (!transposed, a.rows),
        FORMAT_CSC => (transposed, a.cols),
        FORMAT_COO => (false, 0),
        _ => return Err(NOT_SUPPORTED),
    };
    let variant = if a.format == FORMAT_COO {
        let w = index_wide(a.idx_ty)?;
        Variant { double, ptr64: w, idx64: w }
    } else {
        Variant { double, ptr64: index_wide(a.ptr_ty)?, idx64: index_wide(a.idx_ty)? }
    };
    let batch_u = batch as u32;
    let (ptr_bs, idx_bs) = if a.format == FORMAT_COO {
        (a.idx_bstride as u64 * a_scale, a.idx_bstride as u64 * a_scale)
    } else {
        (a.ptr_bstride as u64 * a_scale, a.idx_bstride as u64 * a_scale)
    };
    let mn = (m as u64) * (n as u64);
    if gather {
        let vector = match choice {
            Choice::Scalar => false,
            Choice::Vector => true,
            Choice::Auto => storage_rows > 0 && a.nnz / storage_rows >= VECTOR_MIN_AVG,
        };
        let args = vec![a.ptr, a.idx, a.val, b.val, c.val, m as u64, n as u64, base, bsi, bsj, csi, csj, s.alpha, s.beta, s.alpha_ptr, s.beta_ptr, ptr_bs, idx_bs, b_bs, c_bs];
        let l = if vector {
            Launch { kernel: Kernel::VGather, variant, blocks: mn, block: VBLOCK, batch: batch_u, args }
        } else {
            threads(Kernel::Gather, variant, mn, batch_u, args)
        };
        return check_launches(vec![l]);
    }
    let mut v = vec![threads(Kernel::Scale, variant, mn, batch_u, vec![c.val, m as u64, n as u64, csi, csj, s.beta, s.beta_ptr, c_bs])];
    let (count, mode) = match a.format {
        FORMAT_COO => (a.nnz, if transposed { 2 } else { 1 }),
        _ => (storage_rows, 0),
    };
    if count > 0 && a.nnz > 0 {
        let args = vec![a.ptr, a.idx, a.val, b.val, c.val, count as u64, n as u64, base, mode, bsi, bsj, csi, csj, s.alpha, s.alpha_ptr, ptr_bs, idx_bs, b_bs, c_bs];
        v.push(threads(Kernel::Scatter, variant, count as u64 * n as u64, batch_u, args));
    }
    check_launches(v)
}

/// `y = alpha * op(A) * x + beta * y`.
pub fn spmv(op_a: i32, a: &SpMat, x: &DnMat, y: &DnMat, compute: i32, s: Scalars, choice: Choice) -> Result<Vec<Launch>, i32> {
    spmm(op_a, OP_N, a, x, y, compute, s, choice)
}

/// `C = alpha * (op(A) * op(B)) .* spy(C) + beta * C` with C in CSR.
pub fn sddmm(op_a: i32, op_b: i32, a: &DnMat, b: &DnMat, c: &SpMat, compute: i32, s: Scalars) -> Result<Vec<Launch>, i32> {
    if !matches!(op_a, OP_N | OP_T | OP_C) || !matches!(op_b, OP_N | OP_T | OP_C) {
        return Err(INVALID_VALUE);
    }
    if c.format != FORMAT_CSR {
        return Err(NOT_SUPPORTED);
    }
    let double = val_double(c.val_ty)?;
    if a.val_ty != c.val_ty || b.val_ty != c.val_ty || compute != c.val_ty {
        return Err(NOT_SUPPORTED);
    }
    let (m, k) = a.op_shape(op_a);
    let (bk, n) = b.op_shape(op_b);
    if bk != k || c.rows != m || c.cols != n {
        return Err(INVALID_VALUE);
    }
    let batch = c.batch.max(1);
    let a_bs = batch_of(a.batch, batch, a.bstride)?;
    let b_bs = batch_of(b.batch, batch, b.bstride)?;
    let variant = Variant { double, ptr64: index_wide(c.ptr_ty)?, idx64: index_wide(c.idx_ty)? };
    if m == 0 || c.nnz == 0 {
        return Ok(Vec::new());
    }
    let (asi, asj) = a.op_strides(op_a);
    let (bsi, bsj) = b.op_strides(op_b);
    let args = vec![c.ptr, c.idx, c.val, a.val, b.val, m as u64, k as u64, c.base as u64, asi, asj, bsi, bsj, s.alpha, s.beta, s.alpha_ptr, s.beta_ptr, c.ptr_bstride as u64, c.idx_bstride as u64, a_bs, b_bs];
    check_launches(vec![threads(Kernel::Sddmm, variant, m as u64, batch as u32, args)])
}

/// Legacy block-CSR product, 32-bit indices: `C = alpha * A * op(B) + beta * C` where A has
/// `mb x kb` blocks of `bd x bd`, B and C are column-major.
#[allow(clippy::too_many_arguments)]
pub fn bsrmm(double: bool, dir: i32, mb: i64, n: i64, kb: i64, nnzb: i64, bd: i64, base: i32, op_b: i32, ptr: u64, col: u64, val: u64, b: u64, ldb: i64, c: u64, ldc: i64, s: Scalars) -> Result<Vec<Launch>, i32> {
    if mb < 0 || n < 0 || kb < 0 || nnzb < 0 || bd < 1 || !matches!(dir, 0 | 1) || !matches!(base, 0 | 1) || !matches!(op_b, OP_N | OP_T) {
        return Err(INVALID_VALUE);
    }
    let k = kb * bd;
    let m = mb * bd;
    let (bsi, bsj, bld_min) = if op_b == OP_N { (1, ldb as u64, k) } else { (ldb as u64, 1, n) };
    if ldb < bld_min.max(1) || ldc < m.max(1) {
        return Err(INVALID_VALUE);
    }
    if m == 0 || n == 0 {
        return Ok(Vec::new());
    }
    let variant = Variant { double, ptr64: false, idx64: false };
    let args = vec![ptr, col, val, b, c, mb as u64, n as u64, base as u64, bd as u64, dir as u64, bsi, bsj, 1, ldc as u64, s.alpha, s.beta, s.alpha_ptr, s.beta_ptr];
    check_launches(vec![threads(Kernel::Bsr, variant, (m as u64) * (n as u64), 1, args)])
}

/// Dense f64 reference for `op(A) * op(B)` products with A sparse (tests).
pub fn dense_of(a: &SpMat, ptr: &[i64], idx: &[i64], val: &[f64]) -> Vec<f64> {
    let (r, c) = (a.rows as usize, a.cols as usize);
    let mut d = vec![0.0; r * c];
    let b = a.base as i64;
    match a.format {
        FORMAT_CSR => {
            for i in 0..r {
                for e in (ptr[i] - b)..(ptr[i + 1] - b) {
                    d[i * c + (idx[e as usize] - b) as usize] += val[e as usize];
                }
            }
        }
        FORMAT_CSC => {
            for j in 0..c {
                for e in (ptr[j] - b)..(ptr[j + 1] - b) {
                    d[(idx[e as usize] - b) as usize * c + j] += val[e as usize];
                }
            }
        }
        _ => {
            for e in 0..a.nnz as usize {
                d[(ptr[e] - b) as usize * c + (idx[e] - b) as usize] += val[e];
            }
        }
    }
    d
}
