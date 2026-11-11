//! Self-check data for the native kernels: small deterministic problems with host-computed
//! reference results. Used by the unit tests (on the SPIR-V interpreter) and by the driver's
//! first-use validation (on the real device).
use super::refs::{self, Mmvq, Rng};
use super::{Kernel, Plan};

/// One kernel argument value, in launch order.
#[derive(Clone, Copy, Debug)]
pub(crate) enum PArg {
    Ptr(u64),
    U32(u32),
    U64(u64),
}

/// What a probe case needs from a device.
pub(crate) trait Device {
    fn alloc(&mut self, bytes: usize) -> Result<u64, String>;
    fn upload(&mut self, addr: u64, data: &[u8]) -> Result<(), String>;
    fn download(&mut self, addr: u64, len: usize) -> Result<Vec<u8>, String>;
    /// Launch the native kernel and wait for completion.
    fn run(&mut self, grid: [u32; 3], block: [u32; 3], args: &[PArg]) -> Result<(), String>;
}

/// Inputs of ggml's fused mmvq epilogue (`ggml_cuda_mm_fusion_args_device`): a gate matrix with
/// the main matrix's layout, and biases with the destination's layout.
#[derive(Clone, Default)]
pub(crate) struct Fusion {
    pub gate: Option<Vec<u8>>,
    pub x_bias: Option<Vec<f32>>,
    pub gate_bias: Option<Vec<f32>>,
}

pub(crate) struct MmvqCase {
    pub p: Mmvq,
    pub x: Vec<u8>,
    pub y: Vec<u8>,
    pub ids: Vec<i32>,
}

/// A general mmvq problem; `ratio` is the channel ratio, `ids` selects the MUL_MAT_ID path.
#[allow(clippy::too_many_arguments)]
pub(crate) fn mmvq_case(
    ty: u32,
    ncols_dst: usize,
    nrows: usize,
    ncols_x: usize,
    nch: usize,
    ratio: u32,
    nsamples: usize,
    use_ids: bool,
    seed: u64,
) -> MmvqCase {
    let t = refs::traits(ty).unwrap();
    let mut rng = Rng(seed);
    let bpr = ncols_x / t.qk;
    let nchannels_x = nch / ratio as usize;
    let ystride_col = (ncols_x / 32) as u32;
    let nch_y = if use_ids { 3 } else { nch };
    let nx_ch = nchannels_x.max(if use_ids { 4 } else { 1 });
    let p = Mmvq {
        ty,
        ncols_dst,
        nwarps: Mmvq::ggml_nwarps(ncols_dst),
        nrows_x: nrows,
        ncols_x,
        nchannels_dst: nch,
        nsamples_dst: nsamples,
        channel_ratio: ratio,
        sample_ratio: 1,
        nchannels_y: nch_y as u32,
        use_ids,
        stride_row_x: bpr as u32,
        stride_col_y: ystride_col,
        stride_col_dst: nrows as u32,
        stride_channel_x: (nrows * bpr) as u32,
        stride_channel_y: ystride_col * ncols_dst as u32,
        stride_channel_dst: (nrows * ncols_dst) as u32,
        stride_sample_x: (nrows * bpr * nx_ch) as u32,
        stride_sample_y: ystride_col * ncols_dst as u32 * nch_y as u32,
        stride_sample_dst: (nrows * ncols_dst * nch) as u32,
        rpb: if ncols_dst == 1 { 1 } else { 2 },
    };
    let x = refs::random_blocks(ty, nrows * bpr * nx_ch * nsamples, &mut rng);
    let acts: Vec<f32> = (0..ncols_x * ncols_dst * nch_y * nsamples).map(|_| rng.unit() * 3.0).collect();
    let y = refs::quantize_q8_1(&acts);
    let ids = (0..nch).map(|i| ((i * 3 + 1) % 4) as i32).collect();
    MmvqCase { p, x, y, ids }
}

impl MmvqCase {
    pub fn dst_len(&self) -> usize {
        self.p.nsamples_dst * self.p.stride_sample_dst as usize
    }

    /// Host reference result (ggml's order) for `nwarps` warps per block.
    pub fn reference(&self, nwarps: usize) -> Vec<f32> {
        let mut want = vec![0f32; self.dst_len()];
        let mut q = self.p.clone();
        q.nwarps = nwarps;
        refs::mmvq(&q, &self.x, &self.y, self.p.use_ids.then_some(&self.ids[..]), &mut want);
        want
    }

    /// Host reference of the decode-fast variant (`unroll` accumulators per lane).
    pub fn reference_fast(&self, f: super::mmvq::Fast) -> Vec<f32> {
        let mut want = vec![0f32; self.dst_len()];
        let ids = self.p.use_ids.then_some(&self.ids[..]);
        refs::mmvq_fast(&self.p, &self.x, &self.y, ids, &mut want, f.warps as usize, f.unroll as usize, f.wide);
        want
    }

    /// Uploads inputs, launches `nwarps` warps and returns the destination.
    pub fn run(&self, dev: &mut impl Device, nwarps: usize) -> Result<Vec<f32>, String> {
        self.run_with(dev, nwarps, None)
    }

    /// Like `run` for the decode-fast variant: `rows` subgroups per workgroup, `nrows` appended.
    pub fn run_fast(&self, dev: &mut impl Device, rows: u32, warps: u32) -> Result<Vec<f32>, String> {
        self.run_with(dev, 0, Some((rows, warps)))
    }

    /// Host reference of ggml's fused epilogue: `sum + x_bias`, and with a gate matrix
    /// `* silu(gate_sum + gate_bias)` (SwiGLU), each sum in ggml's order.
    pub fn reference_fusion(&self, nwarps: usize, f: &Fusion) -> Vec<f32> {
        let mut out = self.reference(nwarps);
        let gate_sum = f.gate.as_ref().map(|g| {
            let mut gc = MmvqCase { p: self.p.clone(), x: g.clone(), y: self.y.clone(), ids: self.ids.clone() };
            gc.p.nwarps = nwarps;
            gc.reference(nwarps)
        });
        for (i, v) in out.iter_mut().enumerate() {
            let mut r = *v + f.x_bias.as_ref().map_or(0.0, |b| b[i]);
            if let Some(gs) = &gate_sum {
                let g = gs[i] + f.gate_bias.as_ref().map_or(0.0, |b| b[i]);
                r *= g / (1.0 + (-g).exp());
            }
            *v = r;
        }
        out
    }

    /// Runs a native fused kernel (`FUSION_ARGS`): the legacy words, then `x_bias`, `gate`,
    /// `gate_bias` (0 when absent) and `glu_op` SwiGLU.
    pub fn run_fusion(&self, dev: &mut impl Device, nwarps: usize, f: &Fusion) -> Result<Vec<f32>, String> {
        let mut upload = |data: Option<Vec<u8>>| -> Result<u64, String> {
            match data {
                Some(b) => {
                    let a = dev.alloc(b.len())?;
                    dev.upload(a, &b)?;
                    Ok(a)
                }
                None => Ok(0),
            }
        };
        let f32s = |v: &Option<Vec<f32>>| v.as_ref().map(|v| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>());
        let extra = [
            PArg::Ptr(upload(f32s(&f.x_bias))?),
            PArg::Ptr(upload(f.gate.clone())?),
            PArg::Ptr(upload(f32s(&f.gate_bias))?),
            PArg::U32(super::mmvq::GLU_SWIGLU),
        ];
        self.run_with_extra(dev, nwarps, None, &extra)
    }

    fn run_with(&self, dev: &mut impl Device, nwarps: usize, fast_rows: Option<(u32, u32)>) -> Result<Vec<f32>, String> {
        self.run_with_extra(dev, nwarps, fast_rows, &[])
    }

    fn run_with_extra(
        &self,
        dev: &mut impl Device,
        nwarps: usize,
        fast_rows: Option<(u32, u32)>,
        extra: &[PArg],
    ) -> Result<Vec<f32>, String> {
        let p = &self.p;
        let xa = dev.alloc(self.x.len())?;
        dev.upload(xa, &self.x)?;
        let ya = dev.alloc(self.y.len())?;
        dev.upload(ya, &self.y)?;
        let ia = if p.use_ids {
            let bytes: Vec<u8> = self.ids.iter().flat_map(|v| v.to_le_bytes()).collect();
            let a = dev.alloc(bytes.len())?;
            dev.upload(a, &bytes)?;
            a
        } else {
            0
        };
        let n = self.dst_len();
        let da = dev.alloc(n * 4)?;
        dev.upload(da, &vec![0u8; n * 4])?;
        let chr = if p.use_ids { [0; 3] } else { refs::fastdiv_values(p.channel_ratio) };
        let nchy = if p.use_ids { refs::fastdiv_values(p.nchannels_y) } else { [0; 3] };
        let smr = refs::fastdiv_values(p.sample_ratio);
        let u = PArg::U32;
        let args = [
            PArg::Ptr(xa),
            PArg::Ptr(ya),
            PArg::Ptr(ia),
            PArg::Ptr(da),
            u(p.ncols_x as u32),
            u(nchy[0]),
            u(nchy[1]),
            u(nchy[2]),
            u(p.stride_row_x),
            u(p.stride_col_y),
            u(p.stride_col_dst),
            u(chr[0]),
            u(chr[1]),
            u(chr[2]),
            u(p.stride_channel_x),
            u(p.stride_channel_y),
            u(p.stride_channel_dst),
            u(smr[0]),
            u(smr[1]),
            u(smr[2]),
            u(p.stride_sample_x),
            u(p.stride_sample_y),
            u(p.stride_sample_dst),
        ];
        if let Some((rows, warps)) = fast_rows {
            let mut fa = args.to_vec();
            fa.push(u(p.nrows_x as u32));
            let g = p.grid();
            // the decode-fast kernel covers nrows_x rows whatever the stock rows per block
            dev.run([(p.nrows_x as u32).div_ceil(rows), g[1], g[2]], [32, rows * warps, 1], &fa)?;
        } else {
            let mut q = p.clone();
            q.nwarps = nwarps;
            let launch: Vec<PArg> = args.iter().chain(extra).copied().collect();
            dev.run(q.grid(), q.block(), &launch)?;
        }
        let bytes = dev.download(da, n * 4)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

pub(crate) struct QuantizeCase {
    pub ne00: usize,
    pub ne0: usize,
    pub ne1: usize,
    pub ne2: usize,
    pub ne3: usize,
    pub x: Vec<f32>,
}

pub(crate) fn quantize_case(ne00: usize, ne0: usize, ne1: usize, ne2: usize, ne3: usize, seed: u64) -> QuantizeCase {
    let mut rng = Rng(seed);
    let mut x: Vec<f32> = (0..ne00 * ne1 * ne2 * ne3).map(|_| rng.unit() * 5.0).collect();
    // an all-zero block and a tiny-magnitude block exercise the amax == 0 and denormal-d paths
    for v in x.iter_mut().take(32) {
        *v = 0.0;
    }
    for v in x.iter_mut().skip(32).take(32) {
        *v *= 1e-6;
    }
    QuantizeCase { ne00, ne0, ne1, ne2, ne3, x }
}

impl QuantizeCase {
    pub fn reference(&self) -> Vec<u8> {
        let mut want = Vec::new();
        for r in 0..self.ne1 * self.ne2 * self.ne3 {
            let mut row = self.x[r * self.ne00..(r + 1) * self.ne00].to_vec();
            row.resize(self.ne0, 0.0);
            want.extend(refs::quantize_q8_1(&row));
        }
        want
    }

    pub fn run(&self, dev: &mut impl Device) -> Result<Vec<u8>, String> {
        let xb: Vec<u8> = self.x.iter().flat_map(|v| v.to_le_bytes()).collect();
        let xa = dev.alloc(xb.len())?;
        dev.upload(xa, &xb)?;
        let out = self.ne1 * self.ne2 * self.ne3 * self.ne0 / 32 * refs::Q8_1_BYTES;
        let ya = dev.alloc(out)?;
        dev.upload(ya, &vec![0u8; out])?;
        let fd = refs::fastdiv_values(self.ne2 as u32);
        let (n00, n1, n2) = (self.ne00 as u64, self.ne1 as u64, self.ne2 as u64);
        let args = [
            PArg::Ptr(xa),
            PArg::Ptr(ya),
            PArg::U64(n00),
            PArg::U64(n00),
            PArg::U64(n00 * n1),
            PArg::U64(n00 * n1 * n2),
            PArg::U64(self.ne0 as u64),
            PArg::U32(self.ne1 as u32),
            PArg::U32(fd[0]),
            PArg::U32(fd[1]),
            PArg::U32(fd[2]),
        ];
        dev.run(
            [self.ne0.div_ceil(256) as u32, self.ne1 as u32, (self.ne2 * self.ne3) as u32],
            [256, 1, 1],
            &args,
        )?;
        dev.download(ya, out)
    }
}

pub(crate) struct MmqCase {
    pub ty: u32,
    pub p: refs::Mmq,
    pub x: Vec<u8>,
    pub y: Vec<u8>,
}

pub(crate) fn mmq_case(ncols_x: usize, nrows: usize, ncols: usize, nch: usize, ratio: usize, nsamples: usize, seed: u64) -> MmqCase {
    mmq_case_ty(refs::Q8_0, ncols_x, nrows, ncols, nch, ratio, nsamples, seed)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn mmq_case_ty(ty: u32, ncols_x: usize, nrows: usize, ncols: usize, nch: usize, ratio: usize, nsamples: usize, seed: u64) -> MmqCase {
    let mut rng = Rng(seed);
    let nb = ncols_x / 32 / refs::mmq_groups(ty);
    let nch_x = nch / ratio;
    let ychan = ncols_x / 128 * ncols * 36; // ints per (channel, sample) of y
    let p = refs::Mmq {
        ncols_x,
        nrows_x: nrows,
        ncols_dst: ncols,
        nchannels_y: nch,
        nsamples_y: nsamples,
        channel_ratio: ratio,
        sample_ratio: 1,
        stride_row_x: nb,
        ncols_y: ncols,
        stride_col_dst: nrows,
        stride_channel_x: nrows * nb,
        stride_channel_y: ychan,
        stride_channel_dst: nrows * ncols,
        stride_sample_x: nrows * nb * nch_x,
        stride_sample_y: ychan * nch,
        stride_sample_dst: nrows * ncols * nch,
    };
    let x = refs::random_blocks(ty, nrows * nb * nch_x * nsamples, &mut rng);
    let mut y = Vec::new();
    for _ in 0..nch * nsamples {
        let acts: Vec<f32> = (0..ncols_x * ncols).map(|_| rng.unit() * 2.0).collect();
        y.extend(refs::quantize_mmq(ty, &acts, ncols_x, ncols));
    }
    MmqCase { ty, p, x, y }
}

impl MmqCase {
    pub fn dst_len(&self) -> usize {
        self.p.nsamples_y * self.p.stride_sample_dst
    }

    pub fn reference(&self) -> Vec<f32> {
        let mut want = vec![0f32; self.dst_len()];
        refs::mmq_ref(self.ty, &self.p, &self.x, &self.y, &mut want);
        want
    }

    /// The launch words of `mul_mat_q` (pointers as addresses), in parameter order.
    pub fn args(&self, x: u64, y: u64, dst: u64) -> Vec<PArg> {
        let p = &self.p;
        let u = |v: usize| PArg::U32(v as u32);
        vec![
            PArg::Ptr(x),
            PArg::Ptr(y),
            PArg::Ptr(0),
            PArg::Ptr(0),
            PArg::Ptr(dst),
            PArg::Ptr(0),
            u(p.ncols_x),
            u(p.nrows_x),
            u(p.ncols_dst),
            u(p.stride_row_x),
            u(p.ncols_y),
            u(p.stride_col_dst),
            u(p.channel_ratio),
            u(p.nchannels_y),
            u(p.stride_channel_x),
            u(p.stride_channel_y),
            u(p.stride_channel_dst),
            u(p.sample_ratio),
            u(p.nsamples_y),
            u(p.stride_sample_x),
            u(p.stride_sample_y),
            u(p.stride_sample_dst),
            u(p.ncols_dst),
        ]
    }

    pub fn run(&self, dev: &mut impl Device, plan: &Plan) -> Result<Vec<f32>, String> {
        let xa = dev.alloc(self.x.len())?;
        dev.upload(xa, &self.x)?;
        let ya = dev.alloc(self.y.len())?;
        dev.upload(ya, &self.y)?;
        let n = self.dst_len();
        let da = dev.alloc(n * 4)?;
        dev.upload(da, &vec![0u8; n * 4])?;
        let args = self.args(xa, ya, da);
        let vals = launch_values(&args);
        let (grid, block) = plan.map_launch([1, 1, 1], [32, 8, 1], &vals).ok_or("mmq geometry rejected")?;
        dev.run(grid, block, &args)?;
        let bytes = dev.download(da, n * 4)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// Launch words as the integers `Plan::map_launch` expects.
pub(crate) fn launch_values(args: &[PArg]) -> Vec<u64> {
    args.iter()
        .map(|a| match *a {
            PArg::Ptr(v) | PArg::U64(v) => v,
            PArg::U32(v) => u64::from(v),
        })
        .collect()
}

/// Runs the plan's self-check on `dev`. `Ok` means the native kernel agrees with the host
/// reference (bitwise for quantize; NMSE below 1e-9 for the float dot products).
pub(crate) fn validate(plan: &Plan, dev: &mut impl Device) -> Result<(), String> {
    match plan.kernel {
        Kernel::Mmvq { ty, ncols, abi } => {
            // 11 rows: not a multiple of the decode-fast rows per workgroup (tail subgroups exit),
            // nor of the small_k rows per block
            let mut case = mmvq_case(ty, ncols as usize, if plan.fast.is_some() { 11 } else { 5 }, 512, 1, 1, 1, false, 0x5eed + u64::from(ty));
            case.p.nwarps = abi.stock_nwarps(ty, ncols) as usize;
            case.p.rpb = abi.rows_per_block(ty, ncols) as usize;
            let (want, got) = if let Some(f) = plan.fast {
                (case.reference_fast(f), case.run_fast(dev, f.rows, f.warps)?)
            } else {
                let nwarps = plan.nwarps_for(case.p.nwarps as u32) as usize;
                (case.reference(nwarps), case.run(dev, nwarps)?)
            };
            let nmse = refs::nmse(&got, &want);
            if !(nmse < 1e-9) || want.iter().all(|v| *v == 0.0) {
                return Err(format!("mmvq probe nmse {nmse}"));
            }
            Ok(())
        }
        Kernel::Mmq { ty, .. } => {
            let case = mmq_case_ty(ty, 256, 70, 40, 2, 1, 1, 0x3171);
            let want = case.reference();
            let got = case.run(dev, plan)?;
            let nmse = refs::nmse(&got, &want);
            if !(nmse < 1e-9) || want.iter().all(|v| *v == 0.0) {
                return Err(format!("mmq probe nmse {nmse}"));
            }
            Ok(())
        }
        Kernel::MmqFixup { .. } | Kernel::FattnFixup => Ok(()),
        Kernel::Fattn { d } => {
            // one tile and a partial one, 4 heads over 2 KV heads, 70 keys (partial chunk), causal mask
            let nq = super::fattn::qt() as usize + 3;
            let case = fattn_case(d as usize, nq, 4, 2, 70, 1, true, 0xfa77);
            let want = case.reference();
            let got = case.run(dev, plan)?;
            let nmse = refs::nmse(&got, &want);
            if !(nmse < 1e-8) || want.iter().all(|v| *v == 0.0) {
                return Err(format!("fattn probe nmse {nmse}"));
            }
            Ok(())
        }
        Kernel::QuantizeQ81 => {
            let case = quantize_case(96, 128, 3, 2, 1, 0x71);
            let want = case.reference();
            let got = case.run(dev)?;
            if got != want {
                let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
                return Err(format!("quantize probe: {bad} of {} bytes differ", want.len()));
            }
            Ok(())
        }
    }
}

/// A FlashAttention problem in ggml's contiguous layouts: Q `[d, nq, nh, ns]` f32, K/V
/// `[d, nkv, nh_kv, ns]` f16, mask `[nkv, nq_pad, 1, ns]` f16, dst `[d, nh, nq, ns]` f32.
pub(crate) struct FattnCase {
    pub d: usize,
    pub nq: usize,
    pub nh: usize,
    pub nh_kv: usize,
    pub nkv: usize,
    pub ns: usize,
    pub nq_pad: usize,
    pub scale: f32,
    pub q: Vec<f32>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    pub mask: Option<Vec<u16>>,
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn fattn_case(d: usize, nq: usize, nh: usize, nh_kv: usize, nkv: usize, ns: usize, causal: bool, seed: u64) -> FattnCase {
    let mut rng = Rng(seed);
    let q = (0..d * nq * nh * ns).map(|_| rng.unit()).collect();
    let k = (0..d * nkv * nh_kv * ns).map(|_| refs::f32_to_f16(rng.unit())).collect();
    let v = (0..d * nkv * nh_kv * ns).map(|_| refs::f32_to_f16(rng.unit())).collect();
    let nq_pad = nq.div_ceil(16) * 16;
    let mask = causal.then(|| {
        let mut m = vec![0u16; nkv * nq_pad * ns];
        for s in 0..ns {
            for i in 0..nq_pad {
                for j in 0..nkv {
                    // the last third of the keys is padding (-inf, as in ggml's padded KV cache);
                    // query i sees keys up to (nvis - nq + i); random biases on visible keys
                    let nvis = nkv - nkv / 3;
                    let visible = j < nvis && j + nq <= nvis + i;
                    m[(s * nq_pad + i) * nkv + j] = if visible { refs::f32_to_f16(rng.unit() * 0.5) } else { 0xfc00 };
                }
            }
        }
        m
    });
    FattnCase { d, nq, nh, nh_kv, nkv, ns, nq_pad, scale: 1.0 / (d as f32).sqrt(), q, k, v, mask }
}

impl FattnCase {
    pub fn dst_len(&self) -> usize {
        self.d * self.nh * self.nq * self.ns
    }

    /// Exact softmax attention in f64.
    pub fn reference(&self) -> Vec<f32> {
        let (d, nq, nh, nkv) = (self.d, self.nq, self.nh, self.nkv);
        let gqa = nh / self.nh_kv;
        let mut out = vec![0f32; self.dst_len()];
        for s in 0..self.ns {
            for h in 0..nh {
                let kh = h / gqa;
                for i in 0..nq {
                    let qrow = &self.q[((s * nh + h) * nq + i) * d..][..d];
                    let mut sc = vec![0f64; nkv];
                    for (j, scj) in sc.iter_mut().enumerate() {
                        let krow = &self.k[((s * self.nh_kv + kh) * nkv + j) * d..][..d];
                        let mut acc = 0f64;
                        for t in 0..d {
                            acc += f64::from(qrow[t] * self.scale) * f64::from(refs::f16_to_f32(krow[t]));
                        }
                        if let Some(m) = &self.mask {
                            acc += f64::from(refs::f16_to_f32(m[(s * self.nq_pad + i) * nkv + j]));
                        }
                        *scj = acc;
                    }
                    let mx = sc.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    let p: Vec<f64> = sc.iter().map(|x| (x - mx).exp()).collect();
                    let l: f64 = p.iter().sum();
                    for t in 0..d {
                        let mut o = 0f64;
                        for (j, pj) in p.iter().enumerate() {
                            o += pj * f64::from(refs::f16_to_f32(self.v[((s * self.nh_kv + kh) * nkv + j) * d + t]));
                        }
                        out[((s * nq + i) * nh + h) * d + t] = (o / l) as f32;
                    }
                }
            }
        }
        out
    }

    /// The 37 launch words of `flash_attn_ext_f16` for buffers at the given addresses.
    pub fn args(&self, q: u64, k: u64, v: u64, mask: u64, dst: u64) -> Vec<PArg> {
        let (d, nq, nh, nkv, ns) = (self.d as u32, self.nq as u32, self.nh as u32, self.nkv as u32, self.ns as u32);
        let nhk = self.nh_kv as u32;
        let u = PArg::U32;
        let f = |x: f32| PArg::U32(x.to_bits());
        vec![
            PArg::Ptr(q),
            PArg::Ptr(k),
            PArg::Ptr(v),
            PArg::Ptr(mask),
            PArg::Ptr(0),
            PArg::Ptr(0),
            PArg::Ptr(dst),
            PArg::Ptr(0),
            f(self.scale),
            f(0.0),
            f(1.0),
            f(1.0),
            u(0),
            f(0.0),
            u(d),
            u(nq),
            u(nh),
            u(ns),
            u(d * 4),
            u(d * 4 * nq),
            u(d * 4 * nq * nh),
            u(d),
            u(nkv),
            u(nhk),
            u(ns),
            u(d * 2),
            u(d * 2 * nkv),
            PArg::U64(u64::from(d * 2 * nkv * nhk)),
            u(d * 2),
            u(d * 2 * nkv),
            PArg::U64(u64::from(d * 2 * nkv * nhk)),
            u(nkv),
            u(self.nq_pad as u32),
            u(if self.mask.is_some() { ns } else { 0 }),
            u(nkv * 2),
            u(nkv * 2 * self.nq_pad as u32),
            PArg::U64(u64::from(nkv * 2 * self.nq_pad as u32)),
        ]
    }

    pub fn run(&self, dev: &mut impl Device, plan: &Plan) -> Result<Vec<f32>, String> {
        let bytes16 = |v: &[u16]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
        let qb: Vec<u8> = self.q.iter().flat_map(|x| x.to_le_bytes()).collect();
        let qa = dev.alloc(qb.len())?;
        dev.upload(qa, &qb)?;
        let ka = dev.alloc(self.k.len() * 2)?;
        dev.upload(ka, &bytes16(&self.k))?;
        let va = dev.alloc(self.v.len() * 2)?;
        dev.upload(va, &bytes16(&self.v))?;
        let ma = match &self.mask {
            Some(m) => {
                let a = dev.alloc(m.len() * 2)?;
                dev.upload(a, &bytes16(m))?;
                a
            }
            None => 0,
        };
        let n = self.dst_len();
        let da = dev.alloc(n * 4)?;
        dev.upload(da, &vec![0u8; n * 4])?;
        let args = self.args(qa, ka, va, ma, da);
        let vals = launch_values(&args);
        let (grid, block) = plan.map_launch([1, 1, 1], [32, 4, 1], &vals).ok_or("fattn geometry rejected")?;
        dev.run(grid, block, &args)?;
        let bytes = dev.download(da, n * 4)?;
        Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}
