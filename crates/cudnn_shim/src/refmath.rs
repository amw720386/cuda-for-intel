//! Convolution geometry and naive CPU references (f32). Used by the shim to
//! build kernel arguments and by tests as the ground truth. Standalone so the
//! main crate's simulator tests can `#[path]`-include it.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct ConvGeom {
    pub n: u32,
    pub c: u32,
    pub h: u32,
    pub w: u32,
    pub k: u32,
    pub r: u32,
    pub s: u32,
    pub p: u32,
    pub q: u32,
    pub pad_h: u32,
    pub pad_w: u32,
    pub str_h: u32,
    pub str_w: u32,
    pub dil_h: u32,
    pub dil_w: u32,
    pub groups: u32,
    /// CUDNN_CONVOLUTION (true flips the filter) vs CUDNN_CROSS_CORRELATION.
    pub flip: bool,
    /// Element strides: x (n,c,h,w), filter (k,c,r,s), y (n,k,p,q).
    pub xs: [u32; 4],
    pub ws: [u32; 4],
    pub ys: [u32; 4],
}

/// cuDNN output extent: `1 + (in + 2*pad - (((k-1)*dil)+1)) / stride`.
pub fn out_dim(input: u32, pad: u32, k: u32, dil: u32, stride: u32) -> Option<u32> {
    let eff = (k.checked_sub(1)?).checked_mul(dil)?.checked_add(1)?;
    let padded = input.checked_add(pad.checked_mul(2)?)?;
    if stride == 0 || padded < eff {
        return None;
    }
    Some(1 + (padded - eff) / stride)
}

impl ConvGeom {
    pub fn cg(&self) -> u32 {
        self.c / self.groups
    }
    pub fn x_len(&self) -> usize {
        self.n as usize * self.c as usize * self.h as usize * self.w as usize
    }
    pub fn w_len(&self) -> usize {
        self.k as usize * self.cg() as usize * self.r as usize * self.s as usize
    }
    pub fn y_len(&self) -> usize {
        self.n as usize * self.k as usize * self.p as usize * self.q as usize
    }
    /// Number of output elements of each kernel kind (total thread count).
    pub fn total(&self, kind: Kind) -> u32 {
        match kind {
            Kind::Fwd => self.n * self.k * self.p * self.q,
            Kind::BwdData => self.n * self.c * self.h * self.w,
            Kind::BwdFilter => self.k * self.cg() * self.r * self.s,
        }
    }
    /// The 30 trailing u32 kernel arguments (params 5..=34).
    pub fn kernel_args(&self, kind: Kind) -> [u32; 30] {
        let mut a = [0u32; 30];
        let head = [
            self.n, self.c, self.h, self.w, self.k, self.r, self.s, self.p, self.q, self.pad_h,
            self.pad_w, self.str_h, self.str_w, self.dil_h, self.dil_w, self.groups, self.flip as u32,
        ];
        a[..17].copy_from_slice(&head);
        a[17..21].copy_from_slice(&self.xs);
        a[21..25].copy_from_slice(&self.ws);
        a[25..29].copy_from_slice(&self.ys);
        a[29] = self.total(kind);
        a
    }
    /// Packed NCHW strides helper.
    pub fn packed_nchw(c: u32, h: u32, w: u32) -> [u32; 4] {
        [c * h * w, h * w, w, 1]
    }
    /// Packed NHWC strides (logical n,c,h,w order).
    pub fn packed_nhwc(c: u32, h: u32, w: u32) -> [u32; 4] {
        [h * w * c, 1, w * c, c]
    }
    pub fn validate(&self) -> bool {
        self.groups > 0
            && self.c % self.groups == 0
            && self.k % self.groups == 0
            && out_dim(self.h, self.pad_h, self.r, self.dil_h, self.str_h) == Some(self.p)
            && out_dim(self.w, self.pad_w, self.s, self.dil_w, self.str_w) == Some(self.q)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Fwd,
    BwdData,
    BwdFilter,
}

fn xi(g: &ConvGeom, n: u32, c: u32, h: u32, w: u32) -> usize {
    (n * g.xs[0] + c * g.xs[1] + h * g.xs[2] + w * g.xs[3]) as usize
}
fn wi(g: &ConvGeom, k: u32, c: u32, r: u32, s: u32) -> usize {
    (k * g.ws[0] + c * g.ws[1] + r * g.ws[2] + s * g.ws[3]) as usize
}
fn yi(g: &ConvGeom, n: u32, k: u32, p: u32, q: u32) -> usize {
    (n * g.ys[0] + k * g.ys[1] + p * g.ys[2] + q * g.ys[3]) as usize
}

/// Visit every (n, k, c_local, p, q, r, s) contribution as
/// `(x_index, w_index, y_index)`; the three references are all reductions over it.
fn visit(g: &ConvGeom, mut f: impl FnMut(usize, usize, usize)) {
    let (cg, kg) = (g.cg(), g.k / g.groups);
    for n in 0..g.n {
        for k in 0..g.k {
            let c0 = (k / kg) * cg;
            for p in 0..g.p {
                for q in 0..g.q {
                    for cl in 0..cg {
                        for r in 0..g.r {
                            let ih = (p * g.str_h + r * g.dil_h) as i64 - g.pad_h as i64;
                            if ih < 0 || ih >= g.h as i64 {
                                continue;
                            }
                            for s in 0..g.s {
                                let iw = (q * g.str_w + s * g.dil_w) as i64 - g.pad_w as i64;
                                if iw < 0 || iw >= g.w as i64 {
                                    continue;
                                }
                                let (rw, sw) =
                                    if g.flip { (g.r - 1 - r, g.s - 1 - s) } else { (r, s) };
                                f(
                                    xi(g, n, c0 + cl, ih as u32, iw as u32),
                                    wi(g, k, cl, rw, sw),
                                    yi(g, n, k, p, q),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

/// y = alpha * conv(x, w) + beta * y.
pub fn conv_fwd(g: &ConvGeom, x: &[f32], w: &[f32], y: &mut [f32], alpha: f32, beta: f32) {
    let mut acc = vec![0f32; y.len()];
    visit(g, |xo, wo, yo| acc[yo] += x[xo] * w[wo]);
    scale_into(g, Kind::Fwd, &acc, y, alpha, beta);
}

/// dx = alpha * conv_bwd_data(w, dy) + beta * dx.
pub fn conv_bwd_data(g: &ConvGeom, w: &[f32], dy: &[f32], dx: &mut [f32], alpha: f32, beta: f32) {
    let mut acc = vec![0f32; dx.len()];
    visit(g, |xo, wo, yo| acc[xo] += dy[yo] * w[wo]);
    scale_into(g, Kind::BwdData, &acc, dx, alpha, beta);
}

/// dw = alpha * conv_bwd_filter(x, dy) + beta * dw.
pub fn conv_bwd_filter(g: &ConvGeom, x: &[f32], dy: &[f32], dw: &mut [f32], alpha: f32, beta: f32) {
    let mut acc = vec![0f32; dw.len()];
    visit(g, |xo, wo, yo| acc[wo] += x[xo] * dy[yo]);
    scale_into(g, Kind::BwdFilter, &acc, dw, alpha, beta);
}

fn scale_into(_g: &ConvGeom, _k: Kind, acc: &[f32], out: &mut [f32], alpha: f32, beta: f32) {
    for (o, a) in out.iter_mut().zip(acc) {
        *o = alpha * a + if beta == 0.0 { 0.0 } else { beta * *o };
    }
}

/// Spatial batch-norm training reference over packed NCHW data. Returns
/// (y, mean, invstd, unbiased_var) with y = scale*(x-mean)*invstd + bias.
pub fn bn_train(
    x: &[f32],
    n: usize,
    c: usize,
    hw: usize,
    scale: &[f32],
    bias: &[f32],
    eps: f32,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let m = (n * hw) as f32;
    let mut y = vec![0f32; x.len()];
    let (mut mean, mut inv, mut uvar) = (vec![0f32; c], vec![0f32; c], vec![0f32; c]);
    for ch in 0..c {
        let idx = |b: usize, i: usize| (b * c + ch) * hw + i;
        let mut s = 0f32;
        for b in 0..n {
            for i in 0..hw {
                s += x[idx(b, i)];
            }
        }
        let mu = s / m;
        let mut v = 0f32;
        for b in 0..n {
            for i in 0..hw {
                let d = x[idx(b, i)] - mu;
                v += d * d;
            }
        }
        let var = v / m;
        let is = 1.0 / (var + eps).sqrt();
        for b in 0..n {
            for i in 0..hw {
                y[idx(b, i)] = scale[ch] * (x[idx(b, i)] - mu) * is + bias[ch];
            }
        }
        mean[ch] = mu;
        inv[ch] = is;
        uvar[ch] = var * m / (m - 1.0).max(1.0);
    }
    (y, mean, inv, uvar)
}

/// Spatial batch-norm backward reference (packed NCHW). Returns (dx, dscale, dbias).
pub fn bn_bwd(
    x: &[f32],
    dy: &[f32],
    n: usize,
    c: usize,
    hw: usize,
    scale: &[f32],
    mean: &[f32],
    invstd: &[f32],
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let m = (n * hw) as f32;
    let mut dx = vec![0f32; x.len()];
    let (mut ds, mut db) = (vec![0f32; c], vec![0f32; c]);
    for ch in 0..c {
        let idx = |b: usize, i: usize| (b * c + ch) * hw + i;
        for b in 0..n {
            for i in 0..hw {
                let xh = (x[idx(b, i)] - mean[ch]) * invstd[ch];
                db[ch] += dy[idx(b, i)];
                ds[ch] += dy[idx(b, i)] * xh;
            }
        }
        for b in 0..n {
            for i in 0..hw {
                let xh = (x[idx(b, i)] - mean[ch]) * invstd[ch];
                dx[idx(b, i)] =
                    scale[ch] * invstd[ch] * (dy[idx(b, i)] - db[ch] / m - xh * ds[ch] / m);
            }
        }
    }
    (dx, ds, db)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geom(flip: bool, nhwc: bool) -> ConvGeom {
        let (n, c, h, w, k, r, s) = (2, 4, 7, 6, 6, 3, 2);
        let (pad_h, pad_w, str_h, str_w, dil_h, dil_w, groups) = (1, 0, 2, 1, 2, 1, 2);
        let p = out_dim(h, pad_h, r, dil_h, str_h).unwrap();
        let q = out_dim(w, pad_w, s, dil_w, str_w).unwrap();
        let cg = c / groups;
        let (xs, ys, ws) = if nhwc {
            (ConvGeom::packed_nhwc(c, h, w), ConvGeom::packed_nhwc(k, p, q), [r * s * cg, 1, s * cg, cg])
        } else {
            (ConvGeom::packed_nchw(c, h, w), ConvGeom::packed_nchw(k, p, q), [cg * r * s, r * s, s, 1])
        };
        ConvGeom { n, c, h, w, k, r, s, p, q, pad_h, pad_w, str_h, str_w, dil_h, dil_w, groups, flip, xs, ws, ys }
    }

    #[test]
    fn out_dim_matches_cudnn_formula() {
        assert_eq!(out_dim(32, 1, 3, 1, 1), Some(32));
        assert_eq!(out_dim(32, 3, 7, 1, 2), Some(16));
        assert_eq!(out_dim(7, 1, 3, 2, 2), Some(3));
        assert_eq!(out_dim(2, 0, 3, 1, 1), None);
    }

    #[test]
    fn single_known_value() {
        // 1x1x3x3 input of 1..9, 2x2 ones filter, no pad: sums of 2x2 windows.
        let g = ConvGeom {
            n: 1, c: 1, h: 3, w: 3, k: 1, r: 2, s: 2, p: 2, q: 2, pad_h: 0, pad_w: 0,
            str_h: 1, str_w: 1, dil_h: 1, dil_w: 1, groups: 1, flip: false,
            xs: [9, 9, 3, 1], ws: [4, 4, 2, 1], ys: [4, 4, 2, 1],
        };
        assert!(g.validate());
        let x: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let mut y = vec![0f32; 4];
        conv_fwd(&g, &x, &[1.0; 4], &mut y, 1.0, 0.0);
        assert_eq!(y, [12.0, 16.0, 24.0, 28.0]);
        conv_fwd(&g, &x, &[1.0; 4], &mut y, 2.0, 0.5);
        assert_eq!(y, [30.0, 40.0, 60.0, 70.0]);
    }

    #[test]
    fn flip_reverses_filter() {
        let mut g = ConvGeom {
            n: 1, c: 1, h: 1, w: 3, k: 1, r: 1, s: 3, p: 1, q: 1, pad_h: 0, pad_w: 0,
            str_h: 1, str_w: 1, dil_h: 1, dil_w: 1, groups: 1, flip: false,
            xs: [3, 3, 3, 1], ws: [3, 3, 3, 1], ys: [1, 1, 1, 1],
        };
        let (x, w) = ([1.0, 2.0, 3.0], [1.0, 10.0, 100.0]);
        let mut y = [0f32];
        conv_fwd(&g, &x, &w, &mut y, 1.0, 0.0);
        assert_eq!(y[0], 321.0);
        g.flip = true;
        conv_fwd(&g, &x, &w, &mut y, 1.0, 0.0);
        assert_eq!(y[0], 123.0);
    }

    /// <conv(x,w), dy> == <x, bwd_data(w,dy)> == <w, bwd_filter(x,dy)> (adjointness),
    /// for every layout/flip combination, with groups, dilation, stride and padding.
    #[test]
    fn backward_passes_are_adjoint() {
        for flip in [false, true] {
            for nhwc in [false, true] {
                let g = geom(flip, nhwc);
                assert!(g.validate());
                let f = |i: usize, m: usize| ((i * 7 + m) % 13) as f32 / 13.0 - 0.4;
                let x: Vec<f32> = (0..g.x_len()).map(|i| f(i, 1)).collect();
                let w: Vec<f32> = (0..g.w_len()).map(|i| f(i, 5)).collect();
                let dy: Vec<f32> = (0..g.y_len()).map(|i| f(i, 9)).collect();
                let mut y = vec![0f32; g.y_len()];
                conv_fwd(&g, &x, &w, &mut y, 1.0, 0.0);
                let mut dx = vec![0f32; g.x_len()];
                conv_bwd_data(&g, &w, &dy, &mut dx, 1.0, 0.0);
                let mut dw = vec![0f32; g.w_len()];
                conv_bwd_filter(&g, &x, &dy, &mut dw, 1.0, 0.0);
                let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(a, b)| a * b).sum::<f32>();
                let l = dot(&y, &dy);
                assert!((l - dot(&x, &dx)).abs() < 1e-3, "data flip={flip} nhwc={nhwc}");
                assert!((l - dot(&w, &dw)).abs() < 1e-3, "filter flip={flip} nhwc={nhwc}");
            }
        }
    }

    #[test]
    fn layouts_agree() {
        // NCHW and NHWC of the same logical tensors give the same logical output.
        let (a, b) = (geom(false, false), geom(false, true));
        let logical = |i: usize| (i % 11) as f32 / 11.0;
        let mut xa = vec![0f32; a.x_len()];
        let mut xb = vec![0f32; b.x_len()];
        let mut wa = vec![0f32; a.w_len()];
        let mut wb = vec![0f32; b.w_len()];
        let mut l = 0;
        for n in 0..a.n { for c in 0..a.c { for h in 0..a.h { for w in 0..a.w {
            let v = logical(l); l += 1;
            xa[xi(&a, n, c, h, w)] = v; xb[xi(&b, n, c, h, w)] = v;
        }}}}
        for k in 0..a.k { for c in 0..a.cg() { for r in 0..a.r { for s in 0..a.s {
            let v = logical(l); l += 1;
            wa[wi(&a, k, c, r, s)] = v; wb[wi(&b, k, c, r, s)] = v;
        }}}}
        let (mut ya, mut yb) = (vec![0f32; a.y_len()], vec![0f32; b.y_len()]);
        conv_fwd(&a, &xa, &wa, &mut ya, 1.0, 0.0);
        conv_fwd(&b, &xb, &wb, &mut yb, 1.0, 0.0);
        for n in 0..a.n { for k in 0..a.k { for p in 0..a.p { for q in 0..a.q {
            assert!((ya[yi(&a, n, k, p, q)] - yb[yi(&b, n, k, p, q)]).abs() < 1e-5);
        }}}}
    }

    #[test]
    fn bn_reference_zero_mean_unit_var() {
        let (n, c, hw) = (3, 2, 5);
        let x: Vec<f32> = (0..n * c * hw).map(|i| (i % 7) as f32 * 0.5 + (i / 9) as f32).collect();
        let (y, mean, inv, _) = bn_train(&x, n, c, hw, &[1.0; 2], &[0.0; 2], 1e-5);
        for ch in 0..c {
            let vals: Vec<f32> = (0..n).flat_map(|b| (0..hw).map(move |i| (b * c + ch) * hw + i)).map(|i| y[i]).collect();
            let mu: f32 = vals.iter().sum::<f32>() / vals.len() as f32;
            let var: f32 = vals.iter().map(|v| (v - mu).powi(2)).sum::<f32>() / vals.len() as f32;
            assert!(mu.abs() < 1e-5 && (var - 1.0).abs() < 1e-3);
        }
        // finite-difference check of the backward formula for one input
        let dy: Vec<f32> = (0..x.len()).map(|i| ((i * 3) % 5) as f32 - 2.0).collect();
        let (dx, _, _) = bn_bwd(&x, &dy, n, c, hw, &[1.0; 2], &mean, &inv);
        let loss = |x: &[f32]| {
            let (y, ..) = bn_train(x, n, c, hw, &[1.0; 2], &[0.0; 2], 1e-5);
            y.iter().zip(&dy).map(|(a, b)| a * b).sum::<f32>() as f64
        };
        let mut xp = x.clone();
        xp[4] += 1e-2;
        let mut xm = x.clone();
        xm[4] -= 1e-2;
        let fd = (loss(&xp) - loss(&xm)) / 2e-2;
        assert!((fd - dx[4] as f64).abs() < 5e-2, "fd {fd} vs {}", dx[4]);
    }
}
