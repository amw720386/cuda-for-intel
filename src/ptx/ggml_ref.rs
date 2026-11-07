//! Straightforward Rust reference of GGML's block-quantised math (block layouts from
//! `ggml-common.h`, dequantisation from `ggml-quants.c`, dot-product decomposition from
//! `vecdotq.cuh`). Used only by the offline conformance tests.

pub const KVALUES_IQ4NL: [i8; 16] = [-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113];

pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        // xorshift64*
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        (self.next() >> 11) % n
    }
    pub fn byte(&mut self) -> u8 {
        (self.next() >> 24) as u8
    }
    pub fn unit(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Q {
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    IQ4NL,
    IQ4XS,
}

pub const ALL: [Q; 12] = [Q::Q6K, Q::Q5K, Q::Q4K, Q::Q3K, Q::Q2K, Q::IQ4NL, Q::IQ4XS, Q::Q4_0, Q::Q5_0, Q::Q8_0, Q::Q4_1, Q::Q5_1];

impl Q {
    pub fn ggml_type(self) -> u32 {
        match self {
            Q::Q4_0 => 2,
            Q::Q4_1 => 3,
            Q::Q5_0 => 6,
            Q::Q5_1 => 7,
            Q::Q8_0 => 8,
            Q::Q2K => 10,
            Q::Q3K => 11,
            Q::Q4K => 12,
            Q::Q5K => 13,
            Q::Q6K => 14,
            Q::IQ4NL => 20,
            Q::IQ4XS => 23,
        }
    }
    pub fn qk(self) -> usize {
        match self {
            Q::Q2K | Q::Q3K | Q::Q4K | Q::Q5K | Q::Q6K | Q::IQ4XS => 256,
            _ => 32,
        }
    }
    pub fn block_bytes(self) -> usize {
        match self {
            Q::Q4_0 | Q::IQ4NL => 18,
            Q::Q4_1 => 20,
            Q::Q5_0 => 22,
            Q::Q5_1 => 24,
            Q::Q8_0 => 34,
            Q::Q2K => 84,
            Q::Q3K => 110,
            Q::Q4K => 144,
            Q::Q5K => 176,
            Q::Q6K => 210,
            Q::IQ4XS => 136,
        }
    }
    /// The dot product subtracts `min` through the q8_1 block sum (`ds.y`) rather than through
    /// the quantised activations.
    pub fn min_by_sum(self) -> bool {
        matches!(self, Q::Q4_0 | Q::Q4_1 | Q::Q5_0 | Q::Q5_1)
    }
    pub fn name(self) -> &'static str {
        match self {
            Q::Q4_0 => "q4_0",
            Q::Q4_1 => "q4_1",
            Q::Q5_0 => "q5_0",
            Q::Q5_1 => "q5_1",
            Q::Q8_0 => "q8_0",
            Q::Q2K => "q2_K",
            Q::Q3K => "q3_K",
            Q::Q4K => "q4_K",
            Q::Q5K => "q5_K",
            Q::Q6K => "q6_K",
            Q::IQ4NL => "iq4_nl",
            Q::IQ4XS => "iq4_xs",
        }
    }
}

pub fn f16_bits(x: f32) -> u16 {
    super::coverage_tests::interp::f32_to_f16_bits(x)
}
pub fn f16_val(h: u16) -> f32 {
    super::sim::f16_to_f32(u32::from(h))
}
fn rd16(b: &[u8], o: usize) -> u16 {
    u16::from_le_bytes([b[o], b[o + 1]])
}
fn half(b: &[u8], o: usize) -> f64 {
    f64::from(f16_val(rd16(b, o)))
}
fn put_half(b: &mut [u8], o: usize, v: f32) {
    b[o..o + 2].copy_from_slice(&f16_bits(v).to_le_bytes());
}

/// One decoded weight: value = `s * q - m`. `m` is constant within each 32-element group for
/// the types where `Q::min_by_sum` holds.
#[derive(Clone, Copy, Debug)]
pub struct E {
    pub s: f64,
    pub q: i32,
    pub m: f64,
}

/// Fills a block with random quant payload; scales are exact powers of two when `pow2`.
pub fn random_block(q: Q, rng: &mut Rng, pow2: bool) -> Vec<u8> {
    let mut b: Vec<u8> = (0..q.block_bytes()).map(|_| rng.byte()).collect();
    let scale = |rng: &mut Rng, _k: u32| -> f32 {
        if pow2 { 2f32.powi(rng.below(2) as i32) } else { (0.002 + rng.unit() * 0.05) as f32 }
    };
    match q {
        Q::Q4_0 | Q::Q5_0 | Q::Q8_0 | Q::IQ4NL => put_half(&mut b, 0, scale(rng, 2)),
        Q::Q4_1 | Q::Q5_1 => {
            put_half(&mut b, 0, scale(rng, 2));
            let m = scale(rng, 3) * if rng.below(2) == 0 { 1.0 } else { -1.0 };
            put_half(&mut b, 2, m);
        }
        Q::Q2K => {
            put_half(&mut b, 80, scale(rng, 6));
            put_half(&mut b, 82, scale(rng, 6));
        }
        Q::Q3K => put_half(&mut b, 108, scale(rng, 6)),
        Q::Q4K | Q::Q5K => {
            put_half(&mut b, 0, scale(rng, 8));
            put_half(&mut b, 2, scale(rng, 8));
        }
        Q::Q6K => put_half(&mut b, 208, scale(rng, 8)),
        Q::IQ4XS => put_half(&mut b, 0, scale(rng, 6)),
    }
    b
}

fn k4_scale_min(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        ((q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4))
    }
}

/// Decodes one block into `qk` elements in logical (dequantisation) order.
pub fn decode_block(q: Q, b: &[u8]) -> Vec<E> {
    let mut out = Vec::with_capacity(q.qk());
    match q {
        Q::Q4_0 | Q::Q4_1 => {
            let d = half(b, 0);
            let (qs, m) = if q == Q::Q4_0 { (&b[2..18], 8.0 * d) } else { (&b[4..20], -half(b, 2)) };
            for half_idx in 0..2 {
                for j in 0..16 {
                    let v = if half_idx == 0 { qs[j] & 0xF } else { qs[j] >> 4 };
                    out.push(E { s: d, q: i32::from(v), m });
                }
            }
        }
        Q::Q5_0 | Q::Q5_1 => {
            let d = half(b, 0);
            let (qh_o, qs_o, m) = if q == Q::Q5_0 { (2, 6, 16.0 * d) } else { (4, 8, -half(b, 2)) };
            let qh = u32::from_le_bytes(b[qh_o..qh_o + 4].try_into().unwrap());
            let qs = &b[qs_o..qs_o + 16];
            for half_idx in 0..2 {
                for j in 0..16 {
                    let xh = if half_idx == 0 { ((qh >> j) << 4) & 0x10 } else { (qh >> (j + 12)) & 0x10 };
                    let v = if half_idx == 0 { u32::from(qs[j] & 0xF) } else { u32::from(qs[j] >> 4) } | xh;
                    out.push(E { s: d, q: v as i32, m });
                }
            }
        }
        Q::Q8_0 => {
            let d = half(b, 0);
            for j in 0..32 {
                out.push(E { s: d, q: i32::from(b[2 + j] as i8), m: 0.0 });
            }
        }
        Q::Q2K => {
            let (d, dmin) = (half(b, 80), half(b, 82));
            let sc = &b[0..16];
            let qs = &b[16..80];
            let mut is = 0;
            for n in 0..2 {
                let qq = &qs[32 * n..];
                for j in 0..4 {
                    for half_idx in 0..2 {
                        let s = sc[is];
                        is += 1;
                        for l in 0..16 {
                            let v = (qq[16 * half_idx + l] >> (2 * j)) & 3;
                            out.push(E { s: d * f64::from(s & 0xF), q: i32::from(v), m: dmin * f64::from(s >> 4) });
                        }
                    }
                }
            }
        }
        Q::Q3K => {
            let d = half(b, 108);
            let hm = &b[0..32];
            let qs = &b[32..96];
            let raw = &b[96..108];
            // scale unpack as in ggml dequantize_row_q3_K
            let mut aux = [0u32; 4];
            for i in 0..3 {
                aux[i] = u32::from_le_bytes(raw[4 * i..4 * i + 4].try_into().unwrap());
            }
            let tmp = aux[2];
            let (k1, k2) = (0x0303_0303u32, 0x0f0f_0f0fu32);
            aux[2] = ((aux[0] >> 4) & k2) | (((tmp >> 4) & k1) << 4);
            aux[3] = ((aux[1] >> 4) & k2) | (((tmp >> 6) & k1) << 4);
            aux[0] = (aux[0] & k2) | ((tmp & k1) << 4);
            aux[1] = (aux[1] & k2) | (((tmp >> 2) & k1) << 4);
            let mut scales = [0i8; 16];
            for i in 0..4 {
                for k in 0..4 {
                    scales[4 * i + k] = (aux[i] >> (8 * k)) as u8 as i8;
                }
            }
            let mut is = 0;
            let mut m = 1u8;
            for n in 0..2 {
                let qq = &qs[32 * n..];
                let mut shift = 0;
                for _ in 0..4 {
                    for half_idx in 0..2 {
                        let dl = d * f64::from(i32::from(scales[is]) - 32);
                        is += 1;
                        for l in 0..16 {
                            let lo = (qq[16 * half_idx + l] >> shift) & 3;
                            let hi = if hm[16 * half_idx + l] & m != 0 { 0 } else { 4 };
                            out.push(E { s: dl, q: i32::from(lo) - hi, m: 0.0 });
                        }
                    }
                    shift += 2;
                    m <<= 1;
                }
            }
        }
        Q::Q4K | Q::Q5K => {
            let (d, dmin) = (half(b, 0), half(b, 2));
            let sc = &b[4..16];
            let (qh, ql) = if q == Q::Q4K { (None, &b[16..144]) } else { (Some(&b[16..48]), &b[48..176]) };
            let mut is = 0;
            let (mut u1, mut u2) = (1u8, 2u8);
            for jj in 0..4 {
                let qq = &ql[32 * jj..];
                let (s1, m1) = k4_scale_min(is, sc);
                let (s2, m2) = k4_scale_min(is + 1, sc);
                for l in 0..32 {
                    let hi = qh.map_or(0, |h| if h[l] & u1 != 0 { 16 } else { 0 });
                    out.push(E { s: d * f64::from(s1), q: i32::from(qq[l] & 0xF) + hi, m: dmin * f64::from(m1) });
                }
                for l in 0..32 {
                    let hi = qh.map_or(0, |h| if h[l] & u2 != 0 { 16 } else { 0 });
                    out.push(E { s: d * f64::from(s2), q: i32::from(qq[l] >> 4) + hi, m: dmin * f64::from(m2) });
                }
                is += 2;
                u1 <<= 2;
                u2 <<= 2;
            }
        }
        Q::Q6K => {
            let d = half(b, 208);
            let (ql, qh, sc) = (&b[0..128], &b[128..192], &b[192..208]);
            let mut o = vec![E { s: 0.0, q: 0, m: 0.0 }; 256];
            for n in 0..2 {
                let (ql, qh, sc) = (&ql[64 * n..], &qh[32 * n..], &sc[8 * n..]);
                for l in 0..32 {
                    let is = l / 16;
                    let q1 = i32::from((ql[l] & 0xF) | ((qh[l] & 3) << 4)) - 32;
                    let q2 = i32::from((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                    let q3 = i32::from((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                    let q4 = i32::from((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                    let base = 128 * n;
                    o[base + l] = E { s: d * f64::from(sc[is] as i8), q: q1, m: 0.0 };
                    o[base + l + 32] = E { s: d * f64::from(sc[is + 2] as i8), q: q2, m: 0.0 };
                    o[base + l + 64] = E { s: d * f64::from(sc[is + 4] as i8), q: q3, m: 0.0 };
                    o[base + l + 96] = E { s: d * f64::from(sc[is + 6] as i8), q: q4, m: 0.0 };
                }
            }
            out = o;
        }
        Q::IQ4NL => {
            let d = half(b, 0);
            for h in 0..2 {
                for j in 0..16 {
                    let v = if h == 0 { b[2 + j] & 0xF } else { b[2 + j] >> 4 };
                    out.push(E { s: d, q: i32::from(KVALUES_IQ4NL[v as usize]), m: 0.0 });
                }
            }
        }
        Q::IQ4XS => {
            let d = half(b, 0);
            let sh = u32::from(rd16(b, 2));
            let sl = &b[4..8];
            let qs = &b[8..136];
            for ib in 0..8 {
                let ls = i32::from(((sl[ib / 2] >> (4 * (ib % 2))) & 0xf) | ((((sh >> (2 * ib)) & 3) as u8) << 4));
                let dl = d * f64::from(ls - 32);
                for h in 0..2 {
                    for j in 0..16 {
                        let v = if h == 0 { qs[16 * ib + j] & 0xF } else { qs[16 * ib + j] >> 4 };
                        out.push(E { s: dl, q: i32::from(KVALUES_IQ4NL[v as usize]), m: 0.0 });
                    }
                }
            }
        }
    }
    assert_eq!(out.len(), q.qk());
    out
}

/// Dequantised values in f32 with the same operation order as the CUDA kernels (`s*q - m`,
/// each product rounded to f32).
pub fn dequant_row(q: Q, blocks: &[u8]) -> Vec<f32> {
    blocks
        .chunks(q.block_bytes())
        .flat_map(|b| decode_block(q, b).into_iter().map(|e| (e.s as f32) * (e.q as f32) - (e.m as f32)))
        .collect()
}

/// q8_1 activation block: half2 ds (d, s) then 32 int8 values.
pub const Q8_1_BYTES: usize = 36;

pub fn make_q8_1(d: f32, qs: &[i8; 32], sum: f32) -> Vec<u8> {
    let mut b = vec![0u8; Q8_1_BYTES];
    put_half(&mut b, 0, d);
    put_half(&mut b, 2, sum);
    for (i, v) in qs.iter().enumerate() {
        b[4 + i] = *v as u8;
    }
    b
}

pub fn random_q8_1(rng: &mut Rng, pow2: bool, range: i32) -> Vec<u8> {
    let mut qs = [0i8; 32];
    for v in &mut qs {
        *v = (rng.below(2 * range as u64 + 1) as i32 - range) as i8;
    }
    let d = if pow2 { 2f32.powi(-(rng.below(2) as i32)) } else { (0.001 + rng.unit() * 0.05) as f32 };
    let total: i32 = qs.iter().map(|&v| i32::from(v)).sum();
    make_q8_1(d, &qs, d * total as f32)
}

/// Reference mat-vec dot of one weight row against one activation column of q8_1 blocks, in
/// the decomposition `vecdotq.cuh` uses, evaluated in f64.
#[derive(Clone, Copy)]
pub struct DotOpts {
    /// Q4_1/Q5_1 mmvq multiplies (d, m) by (d8, s8) as half2.
    pub half_mul: bool,
    /// Use d8 * sum(q8) instead of the stored (half rounded) block sum.
    pub exact_sum: bool,
    /// mmq (DS4 layout) subtracts the K-quant mins through the stored block sum as well.
    pub k_min_by_sum: bool,
    /// mmq K-quant tiles keep (d*sc, dmin*m) as half2.
    pub half_scales: bool,
}

pub const MMVQ: DotOpts = DotOpts { half_mul: true, exact_sum: false, k_min_by_sum: false, half_scales: false };

pub fn dot_ref(q: Q, row: &[u8], y: &[u8]) -> (f64, f64) {
    dot_ref_opts(q, row, y, MMVQ)
}

pub fn dot_ref_opts(q: Q, row: &[u8], y: &[u8], opts: DotOpts) -> (f64, f64) {
    let mut total = 0.0f64;
    let mut abs = 0.0f64;
    let per = q.qk() / 32;
    let by_sum = q.min_by_sum() || (opts.k_min_by_sum && matches!(q, Q::Q4K | Q::Q5K));
    for (bi, wb) in row.chunks(q.block_bytes()).enumerate() {
        let mut elems = decode_block(q, wb);
        if opts.half_scales && matches!(q, Q::Q4K | Q::Q5K) {
            for e in &mut elems {
                e.s = f64::from(f16_val(f16_bits(e.s as f32)));
                e.m = f64::from(f16_val(f16_bits(e.m as f32)));
            }
        }
        for g in 0..per {
            let yb = &y[(bi * per + g) * Q8_1_BYTES..][..Q8_1_BYTES];
            let d8 = half(yb, 0);
            let s8 = if opts.exact_sum {
                d8 * (0..32).map(|i| f64::from(yb[4 + i] as i8)).sum::<f64>()
            } else {
                half(yb, 2)
            };
            let mut acc = 0.0f64;
            let mut macc = 0.0f64;
            for i in 0..32 {
                let e = elems[32 * g + i];
                let u = f64::from(yb[4 + i] as i8);
                let term = e.s * f64::from(e.q) * u * d8;
                acc += term;
                abs += term.abs();
                if !by_sum {
                    macc += e.m * u * d8;
                    abs += (e.m * u * d8).abs();
                }
            }
            if opts.half_mul && matches!(q, Q::Q4_1 | Q::Q5_1) {
                // vec_dot_q{4,5}_1_q8_1_impl multiply (d, m) by (d8, s8) as half2: both products
                // are rounded to binary16 before use.
                let (d4, m4) = (half(wb, 0) as f32, half(wb, 2) as f32);
                let (d8h, s8h) = (d8 as f32, s8 as f32);
                let sumi: f64 = (0..32).map(|i| f64::from(elems[32 * g + i].q) * f64::from(yb[4 + i] as i8)).sum();
                let d4d8 = f64::from(f16_val(f16_bits(d4 * d8h)));
                let m4s8 = f64::from(f16_val(f16_bits(m4 * s8h)));
                total += sumi * d4d8 + m4s8;
                continue;
            }
            total += acc - macc;
            if by_sum {
                let m = elems[32 * g].m;
                debug_assert!(elems[32 * g..32 * g + 32].iter().all(|e| e.m == m));
                total -= m * s8;
                abs += (m * s8).abs();
            }
        }
    }
    (total, abs)
}

/// `quantize_q8_1` of ggml-cuda for one group of 32 floats: returns the 36-byte block.
/// The warp sum is an xor butterfly in f32 (order matters for the bits of `ds.y`).
pub fn quantize_q8_1_group(x: &[f32; 32], int_sum: bool) -> Vec<u8> {
    let amax = x.iter().fold(0f32, |m, v| m.max(v.abs()));
    let mut sum = *x;
    for off in [16usize, 8, 4, 2, 1] {
        let prev = sum;
        for l in 0..32 {
            sum[l] = prev[l] + prev[l ^ off];
        }
    }
    let d = amax / 127.0;
    let mut qs = [0i8; 32];
    if amax != 0.0 {
        for (q, v) in qs.iter_mut().zip(x) {
            let t = v / d;
            // roundf via add.rz(t, copysign(0.5, t)) then cvt.rzi: truncation of t +- 0.5
            *q = (f64::from(t) + 0.5f64.copysign(f64::from(t))).trunc() as i32 as i8;
        }
    }
    // Newer ggml (b12190) stores d * sum(q) instead of the float sum.
    let s = if int_sum { d * qs.iter().map(|&q| i32::from(q)).sum::<i32>() as f32 } else { sum[0] };
    make_q8_1(d, &qs, s)
}
