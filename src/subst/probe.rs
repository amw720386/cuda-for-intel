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

    /// Uploads inputs, launches `nwarps` warps and returns the destination.
    pub fn run(&self, dev: &mut impl Device, nwarps: usize) -> Result<Vec<f32>, String> {
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
        let mut q = p.clone();
        q.nwarps = nwarps;
        dev.run(q.grid(), q.block(), &args)?;
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
        Kernel::Mmvq { ty, ncols } => {
            let case = mmvq_case(ty, ncols as usize, 5, 512, 1, 1, 1, false, 0x5eed + u64::from(ty));
            let nwarps = plan.nwarps_for(case.p.nwarps as u32) as usize;
            let want = case.reference(nwarps);
            let got = case.run(dev, nwarps)?;
            let nmse = refs::nmse(&got, &want);
            if !(nmse < 1e-9) || want.iter().all(|v| *v == 0.0) {
                return Err(format!("mmvq probe nmse {nmse}"));
            }
            Ok(())
        }
        Kernel::Mmq { ty } => {
            let case = mmq_case_ty(ty, 256, 70, 40, 2, 1, 1, 0x3171);
            let want = case.reference();
            let got = case.run(dev, plan)?;
            let nmse = refs::nmse(&got, &want);
            if !(nmse < 1e-9) || want.iter().all(|v| *v == 0.0) {
                return Err(format!("mmq probe nmse {nmse}"));
            }
            Ok(())
        }
        Kernel::MmqFixup { .. } => Ok(()),
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
