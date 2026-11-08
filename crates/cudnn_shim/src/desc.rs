//! Descriptor state and the pure (driver-free) logic that maps cuDNN calls onto
//! kernel geometry. Everything here is unit-testable on any host.
use crate::abi::{self, Status};
use crate::ptxgen::Elem;
use crate::refmath::{ConvGeom, out_dim};

pub const MAX_DIMS: usize = 8;

#[derive(Clone, Debug, PartialEq)]
pub struct TensorDesc {
    pub dtype: i32,
    pub dims: Vec<i32>,
    pub strides: Vec<i32>,
    pub format: i32,
}

impl Default for TensorDesc {
    fn default() -> Self {
        // A freshly created cuDNN tensor descriptor is "unset".
        Self { dtype: abi::DT_FLOAT, dims: Vec::new(), strides: Vec::new(), format: abi::NCHW }
    }
}

/// Fully packed strides for `dims` in the given format (NHWC = channels last).
pub fn packed_strides(dims: &[i32], format: i32) -> Vec<i32> {
    let n = dims.len();
    let mut s = vec![1i32; n];
    if format == abi::NHWC && n >= 3 {
        // physical order: n, d1.., c  (channel is dims[1])
        let mut acc = 1i32;
        s[1] = acc;
        acc = acc.saturating_mul(dims[1]);
        for i in (2..n).rev() {
            s[i] = acc;
            acc = acc.saturating_mul(dims[i]);
        }
        s[0] = acc;
    } else {
        let mut acc = 1i32;
        for i in (0..n).rev() {
            s[i] = acc;
            acc = acc.saturating_mul(dims[i]);
        }
    }
    s
}

impl TensorDesc {
    pub fn set_nd(&mut self, dtype: i32, dims: &[i32], strides: &[i32]) -> Status {
        if dims.len() < 3 || dims.len() > MAX_DIMS || dims.len() != strides.len() {
            return abi::BAD_PARAM;
        }
        if dims.iter().any(|d| *d <= 0) || strides.iter().any(|s| *s <= 0) {
            return abi::BAD_PARAM;
        }
        if !valid_dtype(dtype) {
            return abi::BAD_PARAM;
        }
        // total element span must fit i32 like cuDNN's limit (2^31 elements)
        let span: i64 = dims.iter().zip(strides).map(|(d, s)| (*d as i64 - 1) * *s as i64).sum::<i64>() + 1;
        if span > i32::MAX as i64 * 2 {
            return abi::NOT_SUPPORTED;
        }
        *self = Self { dtype, dims: dims.to_vec(), strides: strides.to_vec(), format: abi::NCHW };
        abi::SUCCESS
    }

    pub fn set_4d(&mut self, format: i32, dtype: i32, n: i32, c: i32, h: i32, w: i32) -> Status {
        if format != abi::NCHW && format != abi::NHWC {
            return abi::BAD_PARAM;
        }
        let dims = [n, c, h, w];
        let st = packed_strides(&dims, format);
        let r = self.set_nd(dtype, &dims, &st);
        if r == abi::SUCCESS {
            self.format = format;
        }
        r
    }

    pub fn set_nd_ex(&mut self, format: i32, dtype: i32, dims: &[i32]) -> Status {
        if format != abi::NCHW && format != abi::NHWC {
            return abi::BAD_PARAM;
        }
        let st = packed_strides(dims, format);
        let r = self.set_nd(dtype, dims, &st);
        if r == abi::SUCCESS {
            self.format = format;
        }
        r
    }

    pub fn is_set(&self) -> bool {
        !self.dims.is_empty()
    }

    /// `cudnnGetTensorSizeInBytes`: (max reachable offset + 1) * element size.
    pub fn size_in_bytes(&self) -> usize {
        let span: usize = self.dims.iter().zip(&self.strides).map(|(d, s)| (*d as usize - 1) * *s as usize).sum::<usize>() + 1;
        span * dtype_bytes(self.dtype)
    }

    pub fn elems(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product()
    }
}

pub fn valid_dtype(d: i32) -> bool {
    matches!(d, 0..=11 | 12..=19)
}

pub fn dtype_bytes(d: i32) -> usize {
    match d {
        abi::DT_DOUBLE | abi::DT_INT64 => 8,
        abi::DT_HALF | abi::DT_BFLOAT16 => 2,
        abi::DT_INT8 | abi::DT_BOOLEAN | 6 => 1,
        _ => 4,
    }
}

/// Element type the generated kernels support for a tensor dtype.
pub fn elem_of(d: i32) -> Option<Elem> {
    match d {
        abi::DT_FLOAT => Some(Elem::F32),
        abi::DT_HALF => Some(Elem::F16),
        abi::DT_BFLOAT16 => Some(Elem::Bf16),
        _ => None,
    }
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct FilterDesc {
    pub dtype: i32,
    pub format: i32,
    pub dims: Vec<i32>,
}

impl FilterDesc {
    pub fn set_nd(&mut self, dtype: i32, format: i32, dims: &[i32]) -> Status {
        if dims.len() < 3 || dims.len() > MAX_DIMS || dims.iter().any(|d| *d <= 0) {
            return abi::BAD_PARAM;
        }
        if !valid_dtype(dtype) || (format != abi::NCHW && format != abi::NHWC) {
            return abi::BAD_PARAM;
        }
        *self = Self { dtype, format, dims: dims.to_vec() };
        abi::SUCCESS
    }
    pub fn strides(&self) -> Vec<i32> {
        packed_strides(&self.dims, self.format)
    }
    pub fn size_in_bytes(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product::<usize>() * dtype_bytes(self.dtype)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ConvDesc {
    pub pad: Vec<i32>,
    pub stride: Vec<i32>,
    pub dilation: Vec<i32>,
    pub mode: i32,
    pub compute_type: i32,
    pub math_type: i32,
    pub groups: i32,
    pub reorder: i32,
}

impl Default for ConvDesc {
    fn default() -> Self {
        Self {
            pad: vec![0, 0],
            stride: vec![1, 1],
            dilation: vec![1, 1],
            mode: abi::CROSS_CORRELATION,
            compute_type: abi::DT_FLOAT,
            math_type: abi::DEFAULT_MATH,
            groups: 1,
            reorder: 0,
        }
    }
}

impl ConvDesc {
    pub fn set_nd(&mut self, pad: &[i32], stride: &[i32], dil: &[i32], mode: i32, compute: i32) -> Status {
        let n = pad.len();
        if n == 0 || n > MAX_DIMS - 2 || stride.len() != n || dil.len() != n {
            return abi::BAD_PARAM;
        }
        if pad.iter().any(|p| *p < 0) || stride.iter().any(|s| *s <= 0) || dil.iter().any(|d| *d <= 0) {
            return abi::BAD_PARAM;
        }
        if mode != abi::CONVOLUTION && mode != abi::CROSS_CORRELATION {
            return abi::BAD_PARAM;
        }
        if !valid_dtype(compute) {
            return abi::BAD_PARAM;
        }
        self.pad = pad.to_vec();
        self.stride = stride.to_vec();
        self.dilation = dil.to_vec();
        self.mode = mode;
        self.compute_type = compute;
        abi::SUCCESS
    }
}

/// Which convolution direction a plan describes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dir {
    Fwd,
    BwdData,
    BwdFilter,
}

/// Result of validating a convolution call: geometry, element type, direction.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvPlan {
    pub geom: ConvGeom,
    pub elem: Elem,
    pub dir: Dir,
}

/// Output dimensions of the forward convolution for NCHW-logical tensors
/// (`cudnnGetConvolutionNdForwardOutputDim`): `[n, k, out...]`.
pub fn forward_output_dims(x: &TensorDesc, w: &FilterDesc, c: &ConvDesc) -> Result<Vec<i32>, Status> {
    let nd = x.dims.len();
    if nd != w.dims.len() || nd != c.pad.len() + 2 {
        return Err(abi::BAD_PARAM);
    }
    if c.groups <= 0 || x.dims[1] % c.groups != 0 || w.dims[0] % c.groups != 0 || w.dims[1] * c.groups != x.dims[1] {
        return Err(abi::BAD_PARAM);
    }
    let mut out = vec![x.dims[0], w.dims[0]];
    for i in 0..c.pad.len() {
        let o = out_dim(x.dims[i + 2] as u32, c.pad[i] as u32, w.dims[i + 2] as u32, c.dilation[i] as u32, c.stride[i] as u32)
            .ok_or(abi::BAD_PARAM)?;
        out.push(o as i32);
    }
    Ok(out)
}

fn st4(t: &[i32]) -> [u32; 4] {
    [t[0] as u32, t[1] as u32, t[2] as u32, t[3] as u32]
}

/// Validate a convolution call and build its kernel geometry. `x`/`y` are the
/// activation tensors in forward orientation (for backward-data `x` is dx and `y` is dy;
/// for backward-filter `w` is dw).
pub fn plan_conv(dir: Dir, x: &TensorDesc, w: &FilterDesc, c: &ConvDesc, y: &TensorDesc) -> Result<ConvPlan, Status> {
    if !x.is_set() || !y.is_set() || w.dims.is_empty() {
        return Err(abi::BAD_PARAM);
    }
    let expect = forward_output_dims(x, w, c)?;
    if y.dims != expect {
        return Err(abi::BAD_PARAM);
    }
    if x.dims.len() != 4 || c.pad.len() != 2 {
        return Err(abi::NOT_SUPPORTED); // only 2-D spatial convolution has kernels
    }
    if x.dtype != w.dtype || x.dtype != y.dtype {
        return Err(abi::NOT_SUPPORTED);
    }
    let elem = elem_of(x.dtype).ok_or(abi::NOT_SUPPORTED)?;
    if c.compute_type == abi::DT_DOUBLE {
        return Err(abi::NOT_SUPPORTED);
    }
    if x.format == abi::NCHW_VECT_C || w.format == abi::NCHW_VECT_C {
        return Err(abi::NOT_SUPPORTED);
    }
    let geom = ConvGeom {
        n: x.dims[0] as u32,
        c: x.dims[1] as u32,
        h: x.dims[2] as u32,
        w: x.dims[3] as u32,
        k: w.dims[0] as u32,
        r: w.dims[2] as u32,
        s: w.dims[3] as u32,
        p: y.dims[2] as u32,
        q: y.dims[3] as u32,
        pad_h: c.pad[0] as u32,
        pad_w: c.pad[1] as u32,
        str_h: c.stride[0] as u32,
        str_w: c.stride[1] as u32,
        dil_h: c.dilation[0] as u32,
        dil_w: c.dilation[1] as u32,
        groups: c.groups as u32,
        flip: c.mode == abi::CONVOLUTION,
        xs: st4(&x.strides),
        ws: st4(&w.strides()),
        ys: st4(&y.strides),
    };
    if !geom.validate() {
        return Err(abi::BAD_PARAM);
    }
    // element counts must fit the kernels' 32-bit thread index
    let total = [geom.x_len(), geom.w_len(), geom.y_len()].into_iter().max().unwrap();
    if total >= u32::MAX as usize {
        return Err(abi::NOT_SUPPORTED);
    }
    Ok(ConvPlan { geom, elem, dir })
}

/// One algorithm entry reported by the `*Algorithm_v7` / `Find*` calls.
pub fn perf_entry(algo: i32, time: f32) -> abi::AlgoPerf {
    abi::AlgoPerf {
        algo,
        status: abi::SUCCESS,
        time,
        memory: 0,
        determinism: abi::DETERMINISTIC,
        math_type: abi::DEFAULT_MATH,
        reserved: [0; 3],
    }
}

/// Spatial batch-norm tensor-descriptor check: x/y 4-D with identical layout;
/// scale/bias/mean/var descriptor is (1, C, 1, 1).
pub struct BnPlan {
    pub elem: Elem,
    pub n: u32,
    pub c: u32,
    pub hw: u32,
    pub w: u32,
    pub strides: [u32; 4],
}

pub fn plan_bn(mode: i32, x: &TensorDesc, y: &TensorDesc, p: &TensorDesc) -> Result<BnPlan, Status> {
    if mode != abi::BN_SPATIAL && mode != abi::BN_SPATIAL_PERSISTENT {
        return Err(abi::NOT_SUPPORTED); // per-activation mode has no kernel
    }
    if x.dims.len() != 4 || y.dims != x.dims || y.strides != x.strides || y.dtype != x.dtype {
        return Err(if x.dims.len() == 4 && y.dims == x.dims { abi::NOT_SUPPORTED } else { abi::BAD_PARAM });
    }
    let elem = elem_of(x.dtype).ok_or(abi::NOT_SUPPORTED)?;
    if p.dims.len() != 4 || p.dims[0] != 1 || p.dims[1] != x.dims[1] || p.dims[2] != 1 || p.dims[3] != 1 {
        return Err(abi::BAD_PARAM);
    }
    // statistics and parameters are float for float/half/bf16 data
    if p.dtype != abi::DT_FLOAT {
        return Err(abi::NOT_SUPPORTED);
    }
    if p.strides[1] != 1 {
        return Err(abi::NOT_SUPPORTED); // kernels index channel arrays densely
    }
    Ok(BnPlan {
        elem,
        n: x.dims[0] as u32,
        c: x.dims[1] as u32,
        hw: (x.dims[2] * x.dims[3]) as u32,
        w: x.dims[3] as u32,
        strides: st4(&x.strides),
    })
}

#[derive(Clone, Debug, PartialEq)]
pub struct ActivationDesc {
    pub mode: i32,
    pub nan_opt: i32,
    pub coef: f64,
    pub swish_beta: f64,
}
impl Default for ActivationDesc {
    fn default() -> Self {
        Self { mode: abi::ACT_RELU, nan_opt: 0, coef: 0.0, swish_beta: 1.0 }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PoolingDesc {
    pub mode: i32,
    pub nan_opt: i32,
    pub window: Vec<i32>,
    pub pad: Vec<i32>,
    pub stride: Vec<i32>,
}
impl Default for PoolingDesc {
    fn default() -> Self {
        Self { mode: abi::POOL_MAX, nan_opt: 0, window: vec![1, 1], pad: vec![0, 0], stride: vec![1, 1] }
    }
}

/// `cudnnGetPoolingNdForwardOutputDim`: `1 + (in + 2*pad - window) / stride`.
pub fn pooling_output_dims(p: &PoolingDesc, x: &TensorDesc) -> Result<Vec<i32>, Status> {
    let sp = p.window.len();
    if x.dims.len() != sp + 2 {
        return Err(abi::BAD_PARAM);
    }
    let mut out = vec![x.dims[0], x.dims[1]];
    for i in 0..sp {
        let padded = x.dims[i + 2] + 2 * p.pad[i];
        if p.stride[i] <= 0 || p.window[i] <= 0 || padded < p.window[i] {
            return Err(abi::BAD_PARAM);
        }
        out.push(1 + (padded - p.window[i]) / p.stride[i]);
    }
    Ok(out)
}

#[derive(Clone, Debug, PartialEq, Default)]
pub struct DropoutDesc {
    pub dropout: f32,
    pub states: usize,
    pub state_bytes: usize,
    pub seed: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t4(fmt: i32, n: i32, c: i32, h: i32, w: i32, dt: i32) -> TensorDesc {
        let mut t = TensorDesc::default();
        assert_eq!(t.set_4d(fmt, dt, n, c, h, w), abi::SUCCESS);
        t
    }

    #[test]
    fn packed_strides_nchw_and_nhwc() {
        assert_eq!(packed_strides(&[2, 3, 4, 5], abi::NCHW), vec![60, 20, 5, 1]);
        assert_eq!(packed_strides(&[2, 3, 4, 5], abi::NHWC), vec![60, 1, 15, 3]);
        assert_eq!(packed_strides(&[2, 3, 4, 5, 6], abi::NHWC), vec![360, 1, 90, 18, 3]);
        assert_eq!(packed_strides(&[2, 3, 7], abi::NCHW), vec![21, 7, 1]);
    }

    #[test]
    fn tensor_validation() {
        let mut t = TensorDesc::default();
        assert_eq!(t.set_nd(abi::DT_FLOAT, &[2, 3], &[3, 1]), abi::BAD_PARAM); // < 3 dims
        assert_eq!(t.set_nd(abi::DT_FLOAT, &[2, 0, 3], &[3, 1, 1]), abi::BAD_PARAM);
        assert_eq!(t.set_nd(99, &[2, 1, 3], &[3, 3, 1]), abi::BAD_PARAM);
        assert_eq!(t.set_4d(7, abi::DT_FLOAT, 1, 1, 1, 1), abi::BAD_PARAM);
        assert_eq!(t.set_nd(abi::DT_FLOAT, &[1; 9], &[1; 9]), abi::BAD_PARAM);
        let t = t4(abi::NHWC, 2, 3, 4, 5, abi::DT_HALF);
        assert_eq!(t.size_in_bytes(), 2 * 3 * 4 * 5 * 2);
        assert_eq!(t.format, abi::NHWC);
    }

    #[test]
    fn size_in_bytes_counts_strided_span() {
        let mut t = TensorDesc::default();
        assert_eq!(t.set_nd(abi::DT_FLOAT, &[1, 2, 3], &[100, 10, 1]), abi::SUCCESS);
        assert_eq!(t.size_in_bytes(), (10 + 2 + 1) * 4);
    }

    #[test]
    fn conv_desc_validation() {
        let mut c = ConvDesc::default();
        assert_eq!(c.set_nd(&[1, 1], &[2, 2], &[1, 1], abi::CROSS_CORRELATION, abi::DT_FLOAT), abi::SUCCESS);
        assert_eq!(c.set_nd(&[-1, 1], &[2, 2], &[1, 1], 1, 0), abi::BAD_PARAM);
        assert_eq!(c.set_nd(&[1, 1], &[0, 2], &[1, 1], 1, 0), abi::BAD_PARAM);
        assert_eq!(c.set_nd(&[1, 1], &[1, 1], &[1], 1, 0), abi::BAD_PARAM);
        assert_eq!(c.set_nd(&[1, 1], &[1, 1], &[1, 1], 5, 0), abi::BAD_PARAM);
        assert_eq!(c.set_nd(&[], &[], &[], 1, 0), abi::BAD_PARAM);
        assert_eq!(c.pad, vec![1, 1]); // failed sets leave the descriptor untouched
    }

    #[test]
    fn forward_dims_and_plan() {
        let x = t4(abi::NCHW, 2, 8, 32, 32, abi::DT_FLOAT);
        let mut w = FilterDesc::default();
        assert_eq!(w.set_nd(abi::DT_FLOAT, abi::NCHW, &[16, 4, 3, 3]), abi::SUCCESS);
        let mut c = ConvDesc::default();
        c.groups = 2;
        c.set_nd(&[1, 1], &[2, 2], &[1, 1], abi::CROSS_CORRELATION, abi::DT_FLOAT);
        let dims = forward_output_dims(&x, &w, &c).unwrap();
        assert_eq!(dims, vec![2, 16, 16, 16]);
        let y = t4(abi::NCHW, 2, 16, 16, 16, abi::DT_FLOAT);
        let p = plan_conv(Dir::Fwd, &x, &w, &c, &y).unwrap();
        assert_eq!((p.geom.groups, p.geom.p, p.geom.q, p.geom.flip), (2, 16, 16, false));
        assert_eq!(p.geom.ws, [36, 9, 3, 1]);
        // wrong y -> BAD_PARAM; mismatched filter channels -> BAD_PARAM
        let bad_y = t4(abi::NCHW, 2, 16, 15, 16, abi::DT_FLOAT);
        assert_eq!(plan_conv(Dir::Fwd, &x, &w, &c, &bad_y).unwrap_err(), abi::BAD_PARAM);
        c.groups = 1;
        assert_eq!(plan_conv(Dir::Fwd, &x, &w, &c, &y).unwrap_err(), abi::BAD_PARAM);
    }

    #[test]
    fn plan_rejects_unsupported_with_not_supported() {
        let c = ConvDesc::default();
        let mut w = FilterDesc::default();
        w.set_nd(abi::DT_FLOAT, abi::NCHW, &[4, 4, 1, 1]);
        let x = t4(abi::NCHW, 1, 4, 5, 5, abi::DT_FLOAT);
        let y = x.clone();
        assert!(plan_conv(Dir::Fwd, &x, &w, &c, &y).is_ok());
        // double data
        let xd = t4(abi::NCHW, 1, 4, 5, 5, abi::DT_DOUBLE);
        let mut wd = w.clone();
        wd.dtype = abi::DT_DOUBLE;
        assert_eq!(plan_conv(Dir::Fwd, &xd, &wd, &c, &xd).unwrap_err(), abi::NOT_SUPPORTED);
        // mixed dtypes
        let yh = t4(abi::NCHW, 1, 4, 5, 5, abi::DT_HALF);
        assert_eq!(plan_conv(Dir::Fwd, &x, &w, &c, &yh).unwrap_err(), abi::NOT_SUPPORTED);
        // 3-D convolution
        let mut x5 = TensorDesc::default();
        x5.set_nd_ex(abi::NCHW, abi::DT_FLOAT, &[1, 4, 5, 5, 5]);
        let mut w5 = FilterDesc::default();
        w5.set_nd(abi::DT_FLOAT, abi::NCHW, &[4, 4, 1, 1, 1]);
        let mut c3 = ConvDesc::default();
        c3.set_nd(&[0, 0, 0], &[1, 1, 1], &[1, 1, 1], 1, 0);
        assert_eq!(plan_conv(Dir::Fwd, &x5, &w5, &c3, &x5).unwrap_err(), abi::NOT_SUPPORTED);
    }

    #[test]
    fn nhwc_plan_strides() {
        let x = t4(abi::NHWC, 1, 4, 5, 5, abi::DT_HALF);
        let mut w = FilterDesc::default();
        w.set_nd(abi::DT_HALF, abi::NHWC, &[8, 4, 3, 3]);
        let mut c = ConvDesc::default();
        c.set_nd(&[1, 1], &[1, 1], &[1, 1], abi::CONVOLUTION, abi::DT_FLOAT);
        let y = t4(abi::NHWC, 1, 8, 5, 5, abi::DT_HALF);
        let p = plan_conv(Dir::Fwd, &x, &w, &c, &y).unwrap();
        assert_eq!(p.geom.xs, [100, 1, 20, 4]);
        assert_eq!(p.geom.ws, [36, 1, 12, 4]);
        assert!(p.geom.flip);
        assert_eq!(p.elem, Elem::F16);
    }

    #[test]
    fn pooling_dims() {
        let p = PoolingDesc { mode: 0, nan_opt: 0, window: vec![2, 2], pad: vec![0, 0], stride: vec![2, 2] };
        let x = t4(abi::NCHW, 1, 3, 7, 8, abi::DT_FLOAT);
        assert_eq!(pooling_output_dims(&p, &x).unwrap(), vec![1, 3, 3, 4]);
    }

    #[test]
    fn bn_plan_checks() {
        let x = t4(abi::NCHW, 2, 3, 4, 4, abi::DT_FLOAT);
        let p = t4(abi::NCHW, 1, 3, 1, 1, abi::DT_FLOAT);
        let plan = plan_bn(abi::BN_SPATIAL, &x, &x, &p).unwrap();
        assert_eq!((plan.n, plan.c, plan.hw, plan.w), (2, 3, 16, 4));
        assert_eq!(plan_bn(abi::BN_PER_ACTIVATION, &x, &x, &p).err(), Some(abi::NOT_SUPPORTED));
        let pb = t4(abi::NCHW, 1, 4, 1, 1, abi::DT_FLOAT);
        assert_eq!(plan_bn(abi::BN_SPATIAL, &x, &x, &pb).err(), Some(abi::BAD_PARAM));
    }

    #[test]
    fn perf_struct_layout() {
        assert_eq!(std::mem::size_of::<abi::AlgoPerf>(), 48);
        assert_eq!(std::mem::offset_of!(abi::AlgoPerf, memory), 16);
        assert_eq!(std::mem::offset_of!(abi::AlgoPerf, determinism), 24);
        assert_eq!(std::mem::offset_of!(abi::AlgoPerf, reserved), 32);
    }
}
