//! Host reference implementations of the ggml-cuda math the native kernels reproduce
//! (self-contained: also compiled into the hardware test). Float expressions follow the fused
//! multiply-add structure nvcc produced for the stock build (see the notes on each function).
#![allow(dead_code)]

pub const QK8_1: usize = 32;
pub const Q8_1_BYTES: usize = 36;

/// ggml_type ids the substitution layer knows.
pub const Q4_0: u32 = 2;
pub const Q5_0: u32 = 6;
pub const Q8_0: u32 = 8;
pub const Q4_K: u32 = 12;
pub const Q5_K: u32 = 13;
pub const Q6_K: u32 = 14;

#[derive(Clone, Copy, Debug)]
pub struct Traits {
    pub qk: usize,
    pub qi: usize,
    pub vdr: usize,
    pub block_bytes: usize,
}

pub fn traits(ty: u32) -> Option<Traits> {
    Some(match ty {
        Q4_0 => Traits { qk: 32, qi: 4, vdr: 2, block_bytes: 18 },
        Q8_0 => Traits { qk: 32, qi: 8, vdr: 2, block_bytes: 34 },
        Q4_K => Traits { qk: 256, qi: 32, vdr: 2, block_bytes: 144 },
        Q6_K => Traits { qk: 256, qi: 32, vdr: 1, block_bytes: 210 },
        _ => return None,
    })
}

pub fn f16_to_f32(h: u16) -> f32 {
    let (s, e, m) = ((h >> 15) as u32, ((h >> 10) & 0x1f) as i32, (h & 0x3ff) as u32);
    let mag = match e {
        0 => m as f32 * 2f32.powi(-24),
        0x1f => {
            if m == 0 { f32::INFINITY } else { f32::NAN }
        }
        _ => (1.0 + m as f32 / 1024.0) * 2f32.powi(e - 15),
    };
    if s == 1 { -mag } else { mag }
}

/// Round to nearest even.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x7f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let mant = man | 0x80_0000;
        let shift = (14 - e) as u32;
        let mut half = mant >> shift;
        let rem = mant & ((1 << shift) - 1);
        let halfway = 1 << (shift - 1);
        if rem > halfway || (rem == halfway && half & 1 == 1) {
            half += 1;
        }
        return sign | half as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1;
    }
    sign | h as u16
}

/// ggml `init_fastdiv_values`: `{mp, L, d}`.
pub fn fastdiv_values(d: u32) -> [u32; 3] {
    assert!(d > 0);
    let mut l = 0u32;
    while l < 32 && (1u64 << l) < u64::from(d) {
        l += 1;
    }
    let mp = (((1u64 << 32) * ((1u64 << l) - u64::from(d))) / u64::from(d) + 1) as u32;
    [mp, l, d]
}

pub fn fastdiv(n: u32, v: [u32; 3]) -> u32 {
    let hi = ((u64::from(n) * u64::from(v[0])) >> 32) as u32;
    hi.wrapping_add(n) >> v[1]
}

pub fn fastmodulo(n: u32, v: [u32; 3]) -> u32 {
    n.wrapping_sub(fastdiv(n, v).wrapping_mul(v[2]))
}

/// `warp_reduce_sum` / `warp_reduce_max`: butterfly over 32 lanes with offsets 16..1.
pub fn warp_sum(mut lanes: [f32; 32]) -> [f32; 32] {
    for off in [16, 8, 4, 2, 1] {
        let old = lanes;
        for l in 0..32 {
            lanes[l] = old[l] + old[l ^ off];
        }
    }
    lanes
}

fn fmax(a: f32, b: f32) -> f32 {
    if a.is_nan() { b } else if b.is_nan() { a } else { a.max(b) }
}

/// `quantize_q8_1` for one contiguous row group of `x.len()` (multiple of 32) values: ggml's
/// block_q8_1 (half2 ds = {d, sum}, int8 qs[32]).
pub fn quantize_q8_1(x: &[f32]) -> Vec<u8> {
    assert_eq!(x.len() % QK8_1, 0);
    let mut out = vec![0u8; x.len() / QK8_1 * Q8_1_BYTES];
    for (ib, chunk) in x.chunks(QK8_1).enumerate() {
        let mut amax = [0f32; 32];
        let mut sum = [0f32; 32];
        for l in 0..32 {
            amax[l] = chunk[l].abs();
            sum[l] = chunk[l];
        }
        for off in [16, 8, 4, 2, 1] {
            let old = amax;
            for l in 0..32 {
                amax[l] = fmax(old[l], old[l ^ off]);
            }
        }
        let sum = warp_sum(sum)[0];
        let amax = amax[0];
        let d = amax / 127.0;
        let o = ib * Q8_1_BYTES;
        out[o..o + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
        out[o + 2..o + 4].copy_from_slice(&f32_to_f16(sum).to_le_bytes());
        for l in 0..32 {
            let q = if amax == 0.0 { 0 } else { (chunk[l] / d).round() as i32 as i8 };
            out[o + 4 + l] = q as u8;
        }
    }
    out
}

pub fn dp4a(a: u32, b: u32, c: i32) -> i32 {
    let mut s = c;
    for k in 0..4 {
        s = s.wrapping_add(i32::from((a >> (8 * k)) as u8 as i8) * i32::from((b >> (8 * k)) as u8 as i8));
    }
    s
}

fn b2(bytes: &[u8], word: usize) -> u32 {
    let o = word * 4;
    u32::from(bytes[o]) | u32::from(bytes[o + 1]) << 8 | u32::from(bytes[o + 2]) << 16 | u32::from(bytes[o + 3]) << 24
}

fn fmaf(a: f32, b: f32, c: f32) -> f32 {
    a.mul_add(b, c)
}

/// The per-thread contribution `vec_dot_<type>_q8_1(vx, y, kbx, kqs)` for one x block `xb`
/// (`block_bytes` long) and the first of the q8_1 blocks `yb` that align with it.
/// `a + b*c` shapes below are the fused forms present in the stock PTX.
pub fn vec_dot(ty: u32, xb: &[u8], yb: &[u8], kqs: usize, acc: f32) -> f32 {
    let yq = |blk: usize, word: usize| -> u32 { b2(&yb[blk * Q8_1_BYTES + 4..], word) };
    let yd = |blk: usize| f16_to_f32(u16::from_le_bytes([yb[blk * Q8_1_BYTES], yb[blk * Q8_1_BYTES + 1]]));
    let ys = |blk: usize| f16_to_f32(u16::from_le_bytes([yb[blk * Q8_1_BYTES + 2], yb[blk * Q8_1_BYTES + 3]]));
    match ty {
        Q8_0 => {
            let d0 = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
            let mut sumi = 0i32;
            for i in 0..2 {
                sumi = dp4a(b2(&xb[2..], kqs + i), yq(0, kqs + i), sumi);
            }
            fmaf(d0 * yd(0), sumi as f32, acc)
        }
        Q4_0 => {
            let d4 = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
            let mut sumi = 0i32;
            for i in 0..2 {
                let v = b2(&xb[2..], kqs + i);
                sumi = dp4a(v & 0x0F0F0F0F, yq(0, kqs + i), sumi);
                sumi = dp4a((v >> 4) & 0x0F0F0F0F, yq(0, kqs + i + 4), sumi);
            }
            let inner = fmaf(yd(0), sumi as f32, ys(0) * -4.0);
            fmaf(d4, inner, acc)
        }
        Q4_K => {
            let dm = [f16_to_f32(u16::from_le_bytes([xb[0], xb[1]])), f16_to_f32(u16::from_le_bytes([xb[2], xb[3]]))];
            let scales = &xb[4..16];
            let qs = &xb[16..];
            let bq8_offset = 2 * ((kqs / 2) / 4);
            let q4 = 16 * bq8_offset + 4 * ((kqs / 2) % 4);
            let v = [b2(&qs[q4..], 0), b2(&qs[q4..], 4)];
            let sc16 = |i: usize| u16::from_le_bytes([scales[2 * i], scales[2 * i + 1]]);
            let j = bq8_offset / 2;
            let aux: [u16; 2] = if j < 2 {
                [sc16(j) & 0x3f3f, sc16(j + 2) & 0x3f3f]
            } else {
                [
                    (sc16(j + 2) & 0x0f0f) | ((sc16(j - 2) & 0xc0c0) >> 2),
                    ((sc16(j + 2) >> 4) & 0x0f0f) | ((sc16(j) & 0xc0c0) >> 2),
                ]
            };
            let scm = [aux[0] as u8, (aux[0] >> 8) as u8, aux[1] as u8, (aux[1] >> 8) as u8];
            let (sc, m) = ([scm[0], scm[1]], [scm[2], scm[3]]);
            let mut sumf_d = 0f32;
            let mut sumf_m = 0f32;
            for i in 0..2 {
                let blk = bq8_offset + i;
                let u0 = yq(blk, (kqs / 2) % 4);
                let u1 = yq(blk, (kqs / 2) % 4 + 4);
                let v0 = (v[0] >> (4 * i)) & 0x0F0F0F0F;
                let v1 = (v[1] >> (4 * i)) & 0x0F0F0F0F;
                let dot1 = dp4a(v1, u1, dp4a(v0, u0, 0));
                let dot2 = dp4a(0x01010101, u1, dp4a(0x01010101, u0, 0));
                let d8 = yd(blk);
                sumf_d = fmaf(d8, dot1.wrapping_mul(i32::from(sc[i])) as f32, sumf_d);
                sumf_m = fmaf(d8, dot2.wrapping_mul(i32::from(m[i])) as f32, sumf_m);
            }
            acc + (sumf_d * dm[0] - sumf_m * dm[1])
        }
        Q6_K => {
            // block_q6_K: ql[128] qh[64] scales[16] d(half)
            let ql = &xb[0..128];
            let qh = &xb[128..192];
            let scales = &xb[192..208];
            let d = f16_to_f32(u16::from_le_bytes([xb[208], xb[209]]));
            let bq8_offset = 2 * 2 * (kqs / 16) + (kqs % 16) / 8;
            let scale_offset = 8 * (kqs / 16) + (kqs % 16) / 4;
            let vh_shift = 2 * ((kqs % 16) / 8);
            let vl = b2(ql, kqs);
            let vh = b2(qh, 8 * (kqs / 16) + kqs % 8) >> vh_shift;
            let mut sumf = 0f32;
            for i in 0..2 {
                let sc = i32::from(scales[scale_offset + 4 * i] as i8);
                let vil = (vl >> (4 * i)) & 0x0F0F0F0F;
                let vih = ((vh >> (4 * i)) << 4) & 0x30303030;
                // per-byte (vil|vih) - 32 (values stay inside a signed byte)
                let comb = vil | vih;
                let mut sub = 0u32;
                for k in 0..4 {
                    let byte = ((comb >> (8 * k)) & 0xff) as i32 - 32;
                    sub |= ((byte as u32) & 0xff) << (8 * k);
                }
                let u = yq(bq8_offset + 2 * i, kqs % 8);
                let d8 = yd(bq8_offset + 2 * i);
                sumf = fmaf(d8, dp4a(sub, u, 0).wrapping_mul(sc) as f32, sumf);
            }
            fmaf(d, sumf, acc)
        }
        _ => panic!("unsupported ggml type {ty}"),
    }
}

#[derive(Clone, Debug)]
pub struct Mmvq {
    pub ty: u32,
    pub ncols_dst: usize,
    pub nwarps: usize,
    pub nrows_x: usize,
    pub ncols_x: usize,
    pub nchannels_dst: usize,
    pub nsamples_dst: usize,
    pub channel_ratio: u32,
    pub sample_ratio: u32,
    pub nchannels_y: u32,
    pub use_ids: bool,
    pub stride_row_x: u32,
    pub stride_col_y: u32,
    pub stride_col_dst: u32,
    pub stride_channel_x: u32,
    pub stride_channel_y: u32,
    pub stride_channel_dst: u32,
    pub stride_sample_x: u32,
    pub stride_sample_y: u32,
    pub stride_sample_dst: u32,
}

impl Mmvq {
    pub fn rows_per_block(&self) -> usize {
        if self.ncols_dst == 1 { 1 } else { 2 }
    }
    /// Stock (NVIDIA generic table) warps per block.
    pub fn ggml_nwarps(ncols_dst: usize) -> usize {
        if ncols_dst <= 4 { 4 } else { 2 }
    }
    pub fn grid(&self) -> [u32; 3] {
        let rpb = self.rows_per_block();
        [self.nrows_x.div_ceil(rpb) as u32, self.nchannels_dst as u32, self.nsamples_dst as u32]
    }
    pub fn block(&self) -> [u32; 3] {
        [32, self.nwarps as u32, 1]
    }
}

/// Simulates the CUDA kernel thread by thread (same accumulation and reduction order).
/// `x` holds quantized rows, `y` q8_1 blocks, `dst` is updated in place (only written entries).
pub fn mmvq(p: &Mmvq, x: &[u8], y: &[u8], ids: Option<&[i32]>, dst: &mut [f32]) {
    let t = traits(p.ty).unwrap();
    let (qk, qi, vdr) = (t.qk, t.qi, t.vdr);
    let rpb = p.rows_per_block();
    let nw = p.nwarps;
    let blocks_per_row_x = p.ncols_x / qk;
    let blocks_per_iter = vdr * nw * 32 / qi;
    let chr = fastdiv_values(p.channel_ratio.max(1));
    let smr = fastdiv_values(p.sample_ratio.max(1));
    let nchy = fastdiv_values(p.nchannels_y.max(1));
    let [gx, gy, gz] = p.grid();
    for bz in 0..gz {
        for by in 0..gy {
            for bx in 0..gx {
                let row0 = rpb * bx as usize;
                let channel_dst = by;
                let channel_x = if p.ncols_dst == 1 && p.use_ids {
                    ids.unwrap()[channel_dst as usize] as u32
                } else {
                    fastdiv(channel_dst, chr)
                };
                let channel_y = if p.ncols_dst == 1 && p.use_ids { fastmodulo(channel_dst, nchy) } else { channel_dst };
                let sample_dst = bz;
                let sample_x = fastdiv(sample_dst, smr);
                let sample_y = sample_dst;
                let y_base = sample_y as usize * p.stride_sample_y as usize + channel_y as usize * p.stride_channel_y as usize;
                let kbx_offset = sample_x as usize * p.stride_sample_x as usize
                    + channel_x as usize * p.stride_channel_x as usize
                    + row0 * p.stride_row_x as usize;
                // tmp[warp][lane][j][i]
                let mut tmp = vec![vec![vec![vec![0f32; rpb]; p.ncols_dst]; 32]; nw];
                for w in 0..nw {
                    for lane in 0..32 {
                        let tid = 32 * w + lane;
                        let mut kbx = tid / (qi / vdr);
                        while kbx < blocks_per_row_x {
                            let kby = kbx * (qk / QK8_1);
                            let kqs = vdr * (tid % (qi / vdr));
                            for j in 0..p.ncols_dst {
                                for i in 0..rpb {
                                    if i > 0 && row0 + i >= p.stride_col_dst as usize {
                                        continue; // rows past the matrix are not read
                                    }
                                    let xb_i = kbx_offset + i * p.stride_row_x as usize + kbx;
                                    let xb = &x[xb_i * t.block_bytes..(xb_i + 1) * t.block_bytes];
                                    let yb_i = y_base + j * p.stride_col_y as usize + kby;
                                    let yb = &y[yb_i * Q8_1_BYTES..];
                                    tmp[w][lane][j][i] = vec_dot(p.ty, xb, yb, kqs, tmp[w][lane][j][i]);
                                }
                            }
                            kbx += blocks_per_iter;
                        }
                    }
                }
                let dst_base = sample_dst as usize * p.stride_sample_dst as usize
                    + channel_dst as usize * p.stride_channel_dst as usize
                    + row0;
                for j in 0..p.ncols_dst {
                    let mut lanes = [[0f32; 32]; 2];
                    for i in 0..rpb {
                        for lane in 0..32 {
                            let mut v = tmp[0][lane][j][i];
                            for l in 1..nw {
                                v += tmp[l][lane][j][i];
                            }
                            lanes[i][lane] = v;
                        }
                        lanes[i] = warp_sum(lanes[i]);
                    }
                    for tx in 0..rpb {
                        if rpb == 1 || (row0 + tx) < p.stride_col_dst as usize {
                            dst[dst_base + j * p.stride_col_dst as usize + tx] = lanes[tx][0];
                        }
                    }
                }
            }
        }
    }
}

// ---- block builders (test data) ----

pub struct Rng(pub u64);
impl Rng {
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn u32(&mut self) -> u32 {
        (self.next() >> 32) as u32
    }
    pub fn unit(&mut self) -> f32 {
        (self.u32() >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0
    }
}

/// Random but valid quantized blocks: bytes random, half scales finite and moderate.
pub fn random_blocks(ty: u32, nblocks: usize, rng: &mut Rng) -> Vec<u8> {
    let bytes = mmq_block_bytes(ty);
    let mut out = vec![0u8; nblocks * bytes];
    for b in 0..nblocks {
        let blk = &mut out[b * bytes..(b + 1) * bytes];
        for byte in blk.iter_mut() {
            *byte = rng.u32() as u8;
        }
        let half = |rng: &mut Rng| f32_to_f16(rng.unit() * 0.05).to_le_bytes();
        match ty {
            Q4_0 | Q5_0 | Q8_0 => blk[0..2].copy_from_slice(&half(rng)),
            Q4_K | Q5_K => {
                blk[0..2].copy_from_slice(&half(rng));
                blk[2..4].copy_from_slice(&half(rng));
            }
            Q6_K => blk[208..210].copy_from_slice(&half(rng)),
            _ => {}
        }
    }
    out
}

pub fn nmse(got: &[f32], want: &[f32]) -> f64 {
    let (mut num, mut den) = (0f64, 0f64);
    for (g, w) in got.iter().zip(want) {
        num += f64::from(g - w).powi(2);
        den += f64::from(*w).powi(2);
    }
    if den == 0.0 { num } else { num / den }
}

// ---- mul_mat_q (Q8_0 x block_q8_1_mmq D4) ----

pub const MMQ_BLOCK_BYTES: usize = 144;

/// `quantize_mmq_q8_1<D4>` layout for `ncols` columns of `ne0` values (`ne0 % 128 == 0`):
/// block `(k/128) * ncols + col` holds `float d4[4]` then `int8 qs[128]`. Values are quantized
/// with `x * (127/amax)` as ggml does; an all-zero 32-group gets d = 0 and zero quants.
pub fn quantize_mmq_d4(x: &[f32], ne0: usize, ncols: usize) -> Vec<u8> {
    assert!(ne0 % 128 == 0 && x.len() == ne0 * ncols);
    let mut out = vec![0u8; ne0 / 128 * ncols * MMQ_BLOCK_BYTES];
    for col in 0..ncols {
        for g in 0..ne0 / 32 {
            let v = &x[col * ne0 + g * 32..col * ne0 + g * 32 + 32];
            let amax = v.iter().fold(0f32, |a, b| fmax(a, b.abs()));
            let d_inv = 127.0 / amax;
            let blk = (g / 4) * ncols + col;
            let o = blk * MMQ_BLOCK_BYTES;
            let d = if amax == 0.0 { 0.0 } else { 1.0 / d_inv };
            out[o + 4 * (g % 4)..o + 4 * (g % 4) + 4].copy_from_slice(&d.to_le_bytes());
            for (l, xv) in v.iter().enumerate() {
                let q = if amax == 0.0 { 0 } else { (xv * d_inv).round() as i32 as i8 };
                out[o + 16 + (g % 4) * 32 + l] = q as u8;
            }
        }
    }
    out
}

#[derive(Clone, Debug)]
pub struct Mmq {
    pub ncols_x: usize,
    pub nrows_x: usize,
    pub ncols_dst: usize,
    pub nchannels_y: usize,
    pub nsamples_y: usize,
    pub channel_ratio: usize,
    pub sample_ratio: usize,
    pub stride_row_x: usize,       // blocks
    pub ncols_y: usize,
    pub stride_col_dst: usize,
    pub stride_channel_x: usize,   // blocks
    pub stride_channel_y: usize,   // ints
    pub stride_channel_dst: usize,
    pub stride_sample_x: usize,
    pub stride_sample_y: usize,
    pub stride_sample_dst: usize,
}

/// Sequential-k reference of the Q8_0 `mul_mat_q` result for tiles that are not split by
/// stream-k: per k block `sum = fma(dA * (float)sumi, dB, sum)`.
pub fn mmq_q8_0(p: &Mmq, x: &[u8], y: &[u8], dst: &mut [f32]) {
    let nb = p.ncols_x / 32;
    for w in 0..p.nsamples_y {
        for z in 0..p.nchannels_y {
            let xo = (w / p.sample_ratio) * p.stride_sample_x + (z / p.channel_ratio) * p.stride_channel_x;
            let yo = w * p.stride_sample_y + z * p.stride_channel_y;
            for col in 0..p.ncols_dst {
                for row in 0..p.nrows_x {
                    let mut sum = 0f32;
                    for kb in 0..nb {
                        let xb = &x[(xo + row * p.stride_row_x + kb) * 34..];
                        let d_a = f16_to_f32(u16::from_le_bytes([xb[0], xb[1]]));
                        let yb_int = yo + (kb / 4) * p.ncols_y * 36 + col * 36;
                        let yb = &y[yb_int * 4..];
                        let d_b = f32::from_le_bytes([yb[4 * (kb % 4)], yb[4 * (kb % 4) + 1], yb[4 * (kb % 4) + 2], yb[4 * (kb % 4) + 3]]);
                        let mut sumi = 0i32;
                        for t in 0..8 {
                            sumi = dp4a(b2(&xb[2..], t), b2(&yb[16 + (kb % 4) * 32..], t), sumi);
                        }
                        sum = fmaf(d_a * sumi as f32, d_b, sum);
                    }
                    dst[w * p.stride_sample_dst + z * p.stride_channel_dst + col * p.stride_col_dst + row] = sum;
                }
            }
        }
    }
}

// ---- mul_mat_q for every type the native kernel covers ----

pub fn mmq_block_bytes(ty: u32) -> usize {
    match ty {
        Q4_0 => 18,
        Q5_0 => 22,
        Q8_0 => 34,
        Q4_K => 144,
        Q5_K => 176,
        Q6_K => 210,
        _ => panic!("unsupported ggml type {ty}"),
    }
}

/// 32-value groups per block (1 for the legacy types, 8 for the K-quants).
pub fn mmq_groups(ty: u32) -> usize {
    if matches!(ty, Q4_K | Q5_K | Q6_K) { 8 } else { 1 }
}

/// y layout ggml picks for the type: `half2 ds[4]` (true) or `float d[4]`.
pub fn mmq_ds4(ty: u32) -> bool {
    matches!(ty, Q4_0 | Q4_K | Q5_K)
}

fn h(b: &[u8], o: usize) -> f32 {
    f16_to_f32(u16::from_le_bytes([b[o], b[o + 1]]))
}

fn round_half(x: f32) -> f32 {
    f16_to_f32(f32_to_f16(x))
}

/// One 32-value group of an x block as the stock tile loader presents it.
#[derive(Clone, Debug)]
pub struct XGroup {
    pub q: [i8; 32],
    /// scale-per-32 types: d; K-quants: half-rounded `d*sc`; Q6_K: block d
    pub d: f32,
    /// K-quants: half-rounded `-dmin*m`
    pub m: f32,
    /// Q6_K: the two int8 scales of the group
    pub sc: [i32; 2],
}

/// ggml `get_scale_min_k4`.
fn scale_min_k4(j: usize, s: &[u8]) -> (i32, i32) {
    if j < 4 {
        (i32::from(s[j] & 63), i32::from(s[j + 4] & 63))
    } else {
        (
            i32::from((s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4)),
            i32::from((s[j + 4] >> 4) | ((s[j] >> 6) << 4)),
        )
    }
}

pub fn decode_group(ty: u32, blk: &[u8], g: usize) -> XGroup {
    let mut q = [0i8; 32];
    let (mut d, mut m, mut sc) = (0f32, 0f32, [0i32; 2]);
    match ty {
        Q8_0 => {
            d = h(blk, 0);
            for i in 0..32 {
                q[i] = blk[2 + i] as i8;
            }
        }
        Q4_0 => {
            d = h(blk, 0);
            for i in 0..32 {
                let v = blk[2 + i % 16];
                q[i] = (if i < 16 { v & 15 } else { v >> 4 }) as i8 - 8;
            }
        }
        Q5_0 => {
            d = h(blk, 0);
            let qh = u32::from_le_bytes([blk[2], blk[3], blk[4], blk[5]]);
            for i in 0..32 {
                let j = i % 16;
                let nib = if i < 16 { blk[6 + j] & 15 } else { blk[6 + j] >> 4 };
                q[i] = (nib | ((((qh >> i) & 1) as u8) << 4)) as i8 - 16;
            }
        }
        Q4_K | Q5_K => {
            let (sc_, m_) = scale_min_k4(g, &blk[4..16]);
            d = round_half(h(blk, 0) * sc_ as f32);
            m = round_half(-h(blk, 2) * m_ as f32);
            for i in 0..32 {
                let (nib, hi) = if ty == Q4_K {
                    let v = blk[16 + 32 * (g / 2) + i];
                    (if g % 2 == 0 { v & 15 } else { v >> 4 }, 0)
                } else {
                    let v = blk[48 + 32 * (g / 2) + i];
                    (if g % 2 == 0 { v & 15 } else { v >> 4 }, (blk[16 + i] >> g) & 1)
                };
                q[i] = (nib | (hi << 4)) as i8;
            }
        }
        Q6_K => {
            d = h(blk, 208);
            sc = [i32::from(blk[192 + 2 * g] as i8), i32::from(blk[192 + 2 * g + 1] as i8)];
            for i in 0..32 {
                let v = 32 * g + i;
                let (half, r) = (v / 128, v % 128);
                let (quad, l) = (r / 32, r % 32);
                let lo = blk[64 * half + 32 * (quad & 1) + l];
                let lo = if quad >> 1 == 0 { lo & 15 } else { lo >> 4 };
                let hi = (blk[128 + 32 * half + l] >> (2 * quad)) & 3;
                q[i] = (lo | (hi << 4)) as i8 - 32;
            }
        }
        _ => panic!("unsupported ggml type {ty}"),
    }
    XGroup { q, d, m, sc }
}

/// `block_q8_1_mmq` stream for `ncols` columns: 144-byte blocks, `half2 ds[4]` or `float d[4]`
/// (see `mmq_ds4`) then 128 int8 values. `ds = (d, sum of the float values)`.
pub fn quantize_mmq(ty: u32, x: &[f32], ne0: usize, ncols: usize) -> Vec<u8> {
    assert!(ne0 % 128 == 0 && x.len() == ne0 * ncols);
    let mut out = vec![0u8; ne0 / 128 * ncols * MMQ_BLOCK_BYTES];
    for col in 0..ncols {
        for g in 0..ne0 / 32 {
            let v = &x[col * ne0 + g * 32..col * ne0 + g * 32 + 32];
            let amax = v.iter().fold(0f32, |a, b| fmax(a, b.abs()));
            let d_inv = 127.0 / amax;
            let blk = (g / 4) * ncols + col;
            let o = blk * MMQ_BLOCK_BYTES;
            let d = if amax == 0.0 { 0.0 } else { 1.0 / d_inv };
            if mmq_ds4(ty) {
                let sum: f32 = v.iter().sum();
                out[o + 4 * (g % 4)..o + 4 * (g % 4) + 2].copy_from_slice(&f32_to_f16(d).to_le_bytes());
                out[o + 4 * (g % 4) + 2..o + 4 * (g % 4) + 4].copy_from_slice(&f32_to_f16(sum).to_le_bytes());
            } else {
                out[o + 4 * (g % 4)..o + 4 * (g % 4) + 4].copy_from_slice(&d.to_le_bytes());
            }
            for (l, xv) in v.iter().enumerate() {
                let q = if amax == 0.0 { 0 } else { (xv * d_inv).round() as i32 as i8 };
                out[o + 16 + (g % 4) * 32 + l] = q as u8;
            }
        }
    }
    out
}

/// Reference of the native `mul_mat_q` for tiles stream-k does not split: the float structure of
/// the stock MMA path per type (see the module doc of `subst/mmq.rs`).
pub fn mmq_ref(ty: u32, p: &Mmq, x: &[u8], y: &[u8], dst: &mut [f32]) {
    let nb = p.ncols_x / 32;
    let (bytes, groups) = (mmq_block_bytes(ty), mmq_groups(ty));
    let ds4 = mmq_ds4(ty);
    for w in 0..p.nsamples_y {
        for z in 0..p.nchannels_y {
            let xo = (w / p.sample_ratio) * p.stride_sample_x + (z / p.channel_ratio) * p.stride_channel_x;
            let yo = w * p.stride_sample_y + z * p.stride_channel_y;
            for col in 0..p.ncols_dst {
                for row in 0..p.nrows_x {
                    let mut sum = 0f32;
                    let mut tmp = 0f32;
                    let mut last_d = 0f32;
                    for kb in 0..nb {
                        let bi = xo + row * p.stride_row_x + kb / groups;
                        let xg = decode_group(ty, &x[bi * bytes..(bi + 1) * bytes], kb % groups);
                        let yb_int = yo + (kb / 4) * p.ncols_y * 36 + col * 36;
                        let yb = &y[yb_int * 4..];
                        let hb = 4 * (kb % 4);
                        let (d_b, s_b) = if ds4 {
                            (h(yb, hb), h(yb, hb + 2))
                        } else {
                            (f32::from_le_bytes([yb[hb], yb[hb + 1], yb[hb + 2], yb[hb + 3]]), 0.0)
                        };
                        let yq = &yb[16 + (kb % 4) * 32..];
                        let dot = |r: std::ops::Range<usize>| -> i32 {
                            let mut s = 0i32;
                            for t in r {
                                s = dp4a(b2i8(&xg.q, t), b2(yq, t), s);
                            }
                            s
                        };
                        match ty {
                            Q8_0 | Q4_0 | Q5_0 => {
                                sum = fmaf(xg.d * dot(0..8) as f32, d_b, sum);
                            }
                            Q4_K | Q5_K => {
                                let t = xg.d * d_b;
                                sum = fmaf(t, dot(0..8) as f32, sum);
                                sum = fmaf(xg.m, s_b, sum);
                            }
                            _ => {
                                let v = dot(0..4).wrapping_mul(xg.sc[0]).wrapping_add(dot(4..8).wrapping_mul(xg.sc[1]));
                                tmp = fmaf(v as f32, d_b, tmp);
                                last_d = xg.d;
                                if kb % 4 == 3 || kb == nb - 1 {
                                    // the stage's first group holds the block d
                                    let bi0 = xo + row * p.stride_row_x + (kb - kb % 4) / groups;
                                    let d0 = decode_group(ty, &x[bi0 * bytes..(bi0 + 1) * bytes], (kb - kb % 4) % groups).d;
                                    let _ = last_d;
                                    sum = fmaf(tmp, d0, sum);
                                    tmp = 0.0;
                                }
                            }
                        }
                    }
                    dst[w * p.stride_sample_dst + z * p.stride_channel_dst + col * p.stride_col_dst + row] = sum;
                }
            }
        }
    }
}

fn b2i8(q: &[i8; 32], word: usize) -> u32 {
    let o = word * 4;
    u32::from(q[o] as u8) | u32::from(q[o + 1] as u8) << 8 | u32::from(q[o + 2] as u8) << 16 | u32::from(q[o + 3] as u8) << 24
}
