//! Explicit-GEMM convolution planner (im2col / col2im around the shim's cuBLAS GEMM).
//!
//! The planner is pure: it turns a [`ConvGeom`] into a sequence of calls on an [`Exec`], which the
//! shim implements with generated PTX kernels plus `cublasGemmEx` and the unit tests implement with
//! the SPIR-V simulator plus a CPU matrix multiply. Everything is expressed in element offsets.
//!
//! Layout rules (checked by [`classify`]): the output tensor must be NCHW-like (`[n][k][p*q]`, "mode A")
//! or NHWC-like (`[n*p*q][k]`, "mode B"), and the filter packed `[k][c][r][s]` (order 0) or `[k][r][s][c]`
//! (order 1). The input tensor may have arbitrary strides. Anything else, `flip`, or many tiny groups
//! falls back to the direct kernels.
//!
//! This file is also `#[path]`-included by the main crate's tests (see `src/ptx.rs`).

use super::ptxgen::Elem;
use super::refmath::{ConvGeom, Kind};

/// Workspace budget for the column buffer (bytes); `CFI_CUDNN_WS_MB` overrides it.
pub const DEFAULT_WS_MB: usize = 256;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode {
    /// `[n][k][pq]` activations.
    A,
    /// `[n*pq][k]` activations.
    B,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Layout {
    pub mode: Mode,
    /// 0: filter `[k][c][r][s]`, 1: filter `[k][r][s][c]`.
    pub order: u32,
}

/// Which operand a call refers to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Buf {
    /// Kernel operand roles as in `ptxgen::generate_conv`: a, b, out.
    A,
    B,
    Out,
    /// Workspace, byte offset.
    Ws(usize),
}

#[derive(Clone, Copy, Debug)]
pub struct Im2col {
    pub n0: u32,
    pub nl: u32,
    pub order: u32,
    pub kc_fast: bool,
    pub col_strides: [u32; 4],
    /// Offset in elements of the column buffer.
    pub total: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Col2im {
    pub n0: u32,
    pub nl: u32,
    pub order: u32,
    pub c_fast: bool,
    pub col_strides: [u32; 4],
    pub total: u32,
    pub alpha: f32,
    pub beta: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gemm {
    pub ta: bool,
    pub tb: bool,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a: (Buf, usize),
    pub lda: u32,
    pub b: (Buf, usize),
    pub ldb: u32,
    pub c: (Buf, usize),
    pub ldc: u32,
    /// Batch count and element strides (batch == 1 means a plain GEMM).
    pub batch: u32,
    pub sa: usize,
    pub sb: usize,
    pub sc: usize,
    pub alpha: f32,
    pub beta: f32,
}

pub trait Exec {
    type Err;
    fn im2col(&mut self, src: Buf, dst: Buf, dst_off: usize, p: &Im2col) -> Result<(), Self::Err>;
    fn col2im(&mut self, src: Buf, src_off: usize, dst: Buf, p: &Col2im) -> Result<(), Self::Err>;
    /// `dst[n][k][p][q]` (strides `dst_strides`, logical dims `dims`) `= src` (strides `src_strides`) with offsets in elements.
    fn permute(&mut self, src: Buf, src_off: usize, src_strides: [u32; 4], dst: Buf, dst_off: usize, dst_strides: [u32; 4], dims: [u32; 4]) -> Result<(), Self::Err>;
    fn gemm(&mut self, g: &Gemm) -> Result<(), Self::Err>;
}

pub fn classify(g: &ConvGeom) -> Option<Layout> {
    let (cg, kk) = (g.cg(), g.k);
    let ys = g.ys;
    let pq = g.p * g.q;
    let mode = if ys[3] == 1 && ys[2] == g.q && ys[1] == pq && ys[0] >= kk * pq {
        Mode::A
    } else if ys[1] == 1 && ys[3] == kk && ys[2] == g.q * kk && ys[0] == pq * kk {
        Mode::B
    } else {
        return None;
    };
    let ws = g.ws;
    let order = if ws[3] == 1 && ws[2] == g.s && ws[1] == g.r * g.s && ws[0] == cg * g.r * g.s {
        0
    } else if ws[1] == 1 && ws[3] == cg && ws[2] == g.s * cg && ws[0] == g.r * g.s * cg {
        1
    } else {
        return None;
    };
    Some(Layout { mode, order })
}

/// Whether the explicit-GEMM path should be used at all (otherwise the direct kernels run).
pub fn eligible(g: &ConvGeom) -> Option<Layout> {
    if g.flip || g.n == 0 || g.p == 0 || g.q == 0 || g.k == 0 || g.c == 0 {
        return None;
    }
    if std::env::var_os("CFI_CUDNN_DIRECT").is_some_and(|v| !v.is_empty() && v != "0") {
        return None;
    }
    let (cg, kg) = (g.cg(), g.k / g.groups);
    // Per-group GEMM calls: only worthwhile when the groups are not tiny.
    if g.groups > 1 && (cg < 8 || kg < 8 || g.groups > 64) {
        return None;
    }
    let l = classify(g)?;
    let kc = cg as u64 * g.r as u64 * g.s as u64;
    // All indices are 32-bit in the kernels.
    let big = g.n as u64 * g.groups as u64 * kc * (g.p as u64 * g.q as u64);
    let xs_max = [g.n, g.c, g.h, g.w].iter().zip(g.xs).map(|(d, s)| *d as u64 * s as u64).max().unwrap_or(0);
    if big > u32::MAX as u64 || xs_max > u32::MAX as u64 {
        return None;
    }
    Some(l)
}

fn ws_cap_bytes() -> usize {
    std::env::var("CFI_CUDNN_WS_MB").ok().and_then(|v| v.parse::<usize>().ok()).unwrap_or(DEFAULT_WS_MB) << 20
}

/// Bytes of workspace per image of the chunk.
fn per_image_elems(g: &ConvGeom, kind: Kind, l: Layout) -> usize {
    let pq = g.p as usize * g.q as usize;
    let col = g.groups as usize * g.cg() as usize * g.r as usize * g.s as usize * pq;
    let dyt = if kind == Kind::BwdFilter && l.mode == Mode::A { g.k as usize * pq } else { 0 };
    col + dyt
}

/// Images per chunk allowed by `ws_bytes` (0: does not fit even one image).
fn chunk_for(g: &ConvGeom, kind: Kind, e: Elem, l: Layout, ws_bytes: usize) -> u32 {
    let per = per_image_elems(g, kind, l) << e.shift();
    if per == 0 {
        return 0;
    }
    ((ws_bytes / per).min(g.n as usize)) as u32
}

/// Workspace requirement (bytes) the GEMM path asks the caller for; 0 when the path is not used.
pub fn workspace_bytes(g: &ConvGeom, kind: Kind, e: Elem) -> usize {
    let Some(l) = eligible(g) else { return 0 };
    let per = per_image_elems(g, kind, l) << e.shift();
    let imgs = (ws_cap_bytes() / per.max(1)).clamp(1, g.n as usize);
    // single image bigger than 1 GiB: not worth it
    if per > (1 << 30) { 0 } else { imgs * per }
}

/// Plan and execute. `Ok(false)` means the path does not apply (nothing was executed): the caller
/// must use the direct kernel.
pub fn run<X: Exec>(x: &mut X, g: &ConvGeom, kind: Kind, e: Elem, alpha: f32, beta: f32, ws_bytes: usize) -> Result<bool, X::Err> {
    let Some(l) = eligible(g) else { return Ok(false) };
    let nl_max = chunk_for(g, kind, e, l, ws_bytes);
    if nl_max == 0 {
        return Ok(false);
    }
    let (cg, kg, gr) = (g.cg(), g.k / g.groups, g.groups);
    let kc = cg * g.r * g.s;
    let pq = g.p * g.q;
    let esz = 1usize << e.shift();
    let mut n0 = 0;
    let mut first = true;
    // Column buffer and (bwd filter, mode A) transposed dy live in the workspace.
    while n0 < g.n {
        let nl = nl_max.min(g.n - n0);
        let col_elems = (gr * nl * kc * pq) as usize;
        let dyt_off = (col_elems * esz).next_multiple_of(256);
        // column strides for this chunk: (g, n, kc, pq)
        let cs: [u32; 4] = match (l.mode, kind) {
            (Mode::A, Kind::BwdFilter) => [kc * nl * pq, pq, nl * pq, 1],
            (Mode::A, _) => [nl * kc * pq, kc * pq, pq, 1],
            (Mode::B, _) => [nl * pq * kc, pq * kc, 1, kc],
        };
        let col = Buf::Ws(0);
        let xs = g.xs;
        let ys = g.ys;
        let w_g = (kg * kc) as usize; // filter elements per group
        match kind {
            Kind::Fwd => {
                let ip = Im2col { n0, nl, order: l.order, kc_fast: l.mode == Mode::B, col_strides: cs, total: gr * nl * kc * pq };
                x.im2col(Buf::A, col, 0, &ip)?;
                for gi in 0..gr {
                    let col_g = (gi as usize) * cs[0] as usize;
                    match l.mode {
                        Mode::A => x.gemm(&Gemm {
                            ta: false, tb: false, m: pq, n: kg, k: kc,
                            a: (col, col_g), lda: pq,
                            b: (Buf::B, gi as usize * w_g), ldb: kc,
                            c: (Buf::Out, n0 as usize * ys[0] as usize + gi as usize * kg as usize * ys[1] as usize), ldc: ys[1],
                            batch: nl, sa: cs[1] as usize, sb: 0, sc: ys[0] as usize, alpha, beta,
                        })?,
                        Mode::B => x.gemm(&Gemm {
                            ta: true, tb: false, m: kg, n: nl * pq, k: kc,
                            a: (Buf::B, gi as usize * w_g), lda: kc,
                            b: (col, col_g), ldb: kc,
                            c: (Buf::Out, n0 as usize * ys[0] as usize + gi as usize * kg as usize), ldc: g.k,
                            batch: 1, sa: 0, sb: 0, sc: 0, alpha, beta,
                        })?,
                    }
                }
            }
            Kind::BwdData => {
                for gi in 0..gr {
                    let col_g = (gi as usize) * cs[0] as usize;
                    match l.mode {
                        Mode::A => x.gemm(&Gemm {
                            ta: false, tb: true, m: pq, n: kc, k: kg,
                            a: (Buf::A, n0 as usize * ys[0] as usize + gi as usize * kg as usize * ys[1] as usize), lda: ys[1],
                            b: (Buf::B, gi as usize * w_g), ldb: kc,
                            c: (col, col_g), ldc: pq,
                            batch: nl, sa: ys[0] as usize, sb: 0, sc: cs[1] as usize, alpha: 1.0, beta: 0.0,
                        })?,
                        Mode::B => x.gemm(&Gemm {
                            ta: false, tb: false, m: kc, n: nl * pq, k: kg,
                            a: (Buf::B, gi as usize * w_g), lda: kc,
                            b: (Buf::A, n0 as usize * ys[0] as usize + gi as usize * kg as usize), ldb: g.k,
                            c: (col, col_g), ldc: kc,
                            batch: 1, sa: 0, sb: 0, sc: 0, alpha: 1.0, beta: 0.0,
                        })?,
                    }
                }
                let cp = Col2im { n0, nl, order: l.order, c_fast: xs[1] == 1, col_strides: cs, total: nl * g.c * g.h * g.w, alpha, beta };
                x.col2im(col, 0, Buf::Out, &cp)?;
            }
            Kind::BwdFilter => {
                let ip = Im2col { n0, nl, order: l.order, kc_fast: l.mode == Mode::B, col_strides: cs, total: gr * nl * kc * pq };
                x.im2col(Buf::A, col, 0, &ip)?;
                let b_beta = if first { beta } else { 1.0 };
                if l.mode == Mode::A {
                    let dst_strides = [pq, nl * pq, g.q, 1];
                    x.permute(Buf::B, n0 as usize * ys[0] as usize, ys, Buf::Ws(dyt_off), 0, dst_strides, [nl, g.k, g.p, g.q])?;
                }
                for gi in 0..gr {
                    let col_g = (gi as usize) * cs[0] as usize;
                    let c_off = (Buf::Out, gi as usize * w_g);
                    match l.mode {
                        Mode::A => x.gemm(&Gemm {
                            ta: true, tb: false, m: kc, n: kg, k: nl * pq,
                            a: (col, col_g), lda: nl * pq,
                            b: (Buf::Ws(dyt_off), gi as usize * kg as usize * (nl * pq) as usize), ldb: nl * pq,
                            c: c_off, ldc: kc,
                            batch: 1, sa: 0, sb: 0, sc: 0, alpha, beta: b_beta,
                        })?,
                        Mode::B => x.gemm(&Gemm {
                            ta: false, tb: true, m: kc, n: kg, k: nl * pq,
                            a: (col, col_g), lda: kc,
                            b: (Buf::B, n0 as usize * ys[0] as usize + gi as usize * kg as usize), ldb: g.k,
                            c: c_off, ldc: kc,
                            batch: 1, sa: 0, sb: 0, sc: 0, alpha, beta: b_beta,
                        })?,
                    }
                }
            }
        }
        first = false;
        n0 += nl;
    }
    Ok(true)
}
