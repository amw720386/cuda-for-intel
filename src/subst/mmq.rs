//! Native `mul_mat_q<type, mmq_x, need_check>` for the non-MoE case: a register-tiled SDot kernel
//! with its own grid (no stream-k, so the paired `mul_mat_q_stream_k_fixup` must be skipped).
//! Types: Q8_0, Q4_0, Q5_0 (one float scale per 32 values), Q4_K, Q5_K (half2 `(d*sc, -dmin*m)`
//! per 32 values), Q6_K (int8 scale per 16 values).
//!
//! Tile: 64 rows x 32 columns per 16x16 workgroup (each thread 4 rows x 2 columns), 4 groups of 32
//! values (128 values of k) staged through shared memory per step. The float accumulation follows
//! the stock MMA path of `mmq.cuh` per type (see `refs::mmq_ref`):
//! * scale-per-32 types: `sum = fma(dA * (float)sumi, dB, sum)` per group;
//! * K-quants: `sum = fma(dmA.x*dsB.x, (float)sumi, sum); sum = fma(dmA.y, dsB.y, sum)` per group,
//!   `dmA` being the half-rounded products ggml's tile loader stores;
//! * Q6_K: `tmp = fma((float)(c0*sc0 + c1*sc1), dB, tmp)` per group, `sum = fma(tmp, d, sum)` per
//!   128 values.
//! It differs from stock only on tiles stream-k splits between workgroups.
use super::refs;
use super::spv::{Arg, Kb, Ty, V};

pub(crate) const TILE_I: u32 = 64;
pub(crate) const TILE_J: u32 = 32;
const S: u32 = 4; // groups of 32 values per stage

pub(crate) const ARGS: [Arg; 23] = {
    let mut a = [Arg::U32; 23];
    let mut i = 0;
    while i < 6 {
        a[i] = Arg::Ptr;
        i += 1;
    }
    a
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Kind {
    /// one float scale per 32 values
    A,
    /// half2 (d*sc, -dmin*m) per 32 values (Q4_K, Q5_K)
    K,
    /// int8 scale per 16 values, float d per 256 (Q6_K)
    Q6,
}

#[derive(Clone, Copy, Debug)]
struct Spec {
    kind: Kind,
    bytes: u32,
    /// log2(groups of 32 per block)
    sb_shift: u32,
    /// y holds `half2 ds` per group (else a float d)
    ds4: bool,
    /// shared words of x scale data per (row, group)
    xs: u32,
    /// shared words of y scale data per (column, group)
    ys: u32,
}

fn spec(ty: u32) -> Spec {
    let (kind, bytes, sb_shift, ds4) = match ty {
        refs::Q4_0 => (Kind::A, 18, 0, true),
        refs::Q5_0 => (Kind::A, 22, 0, false),
        refs::Q8_0 => (Kind::A, 34, 0, false),
        refs::Q4_K => (Kind::K, 144, 3, true),
        refs::Q5_K => (Kind::K, 176, 3, true),
        refs::Q6_K => (Kind::Q6, 210, 3, false),
        _ => panic!("mmq: unsupported ggml type {ty}"),
    };
    let xs = match kind {
        Kind::A => 1,
        Kind::K => 2,
        Kind::Q6 => 3,
    };
    Spec { kind, bytes, sb_shift, ds4, xs, ys: if ds4 { 2 } else { 1 } }
}

pub(crate) fn supported(ty: u32) -> bool {
    matches!(ty, refs::Q4_0 | refs::Q5_0 | refs::Q8_0 | refs::Q4_K | refs::Q5_K | refs::Q6_K)
}

pub(crate) fn entry_name(ty: u32) -> String {
    if ty == refs::Q8_0 { "cfi_mmq_q8_0".to_string() } else { format!("cfi_mmq_t{ty}") }
}

/// Word `t` (0..8) of the 32 quantized values of group `g` of the block at byte address `blk`,
/// as the signed/unsigned bytes ggml's tile loader hands to the dot product.
fn x_word(kb: &mut Kb, ty: u32, blk: V, g: V, t: V) -> V {
    let mask = kb.c32(0x0F0F_0F0F);
    match ty {
        refs::Q8_0 => {
            let to = kb.mulc(t, 4);
            let a = kb.addr(blk, to, 1);
            let a = kb.addr_off(a, 2);
            kb.ld_b2(a)
        }
        refs::Q4_0 => {
            let t3 = kb.andc(t, 3);
            let to = kb.mulc(t3, 4);
            let a = kb.addr(blk, to, 1);
            let a = kb.addr_off(a, 2);
            let w = kb.ld_b2(a);
            let hi = kb.shrc(t, 2);
            let sh = kb.shlc(hi, 2);
            let n = kb.shr(w, sh);
            let n = kb.and(n, mask);
            let eight = kb.c32(0x0808_0808);
            kb.vsub4(n, eight)
        }
        refs::Q5_0 => {
            let t3 = kb.andc(t, 3);
            let to = kb.mulc(t3, 4);
            let a = kb.addr(blk, to, 1);
            let a = kb.addr_off(a, 6);
            let ql = kb.ld_b2(a);
            let hi = kb.shrc(t, 2);
            let sh = kb.shlc(hi, 2);
            let n = kb.shr(ql, sh);
            let n = kb.and(n, mask);
            let qa = kb.addr_off(blk, 2);
            let qh = kb.ld_b2(qa);
            let h16 = kb.shlc(hi, 4);
            let t4 = kb.shlc(t3, 2);
            let qsh = kb.add(h16, t4);
            let b = kb.shr(qh, qsh);
            let mut v = n;
            for (from, to) in [(0u32, 4u32), (1, 12), (2, 20), (3, 28)] {
                let s = kb.c32(to - from);
                let m = kb.shl(b, s);
                let m = kb.andc(m, 1 << to);
                v = kb.or(v, m);
            }
            let sixteen = kb.c32(0x1010_1010);
            kb.vsub4(v, sixteen)
        }
        refs::Q4_K | refs::Q5_K => {
            let c = kb.shrc(g, 1);
            let nib = kb.andc(g, 1);
            let nsh = kb.shlc(nib, 2);
            let qbase = if ty == refs::Q4_K { 16 } else { 48 };
            let c32 = kb.mulc(c, 32);
            let to = kb.mulc(t, 4);
            let off = kb.add(c32, to);
            let a = kb.addr(blk, off, 1);
            let a = kb.addr_off(a, qbase);
            let w = kb.ld_b2(a);
            let n = kb.shr(w, nsh);
            let n = kb.and(n, mask);
            if ty == refs::Q4_K {
                return n;
            }
            let to = kb.mulc(t, 4);
            let qa = kb.addr(blk, to, 1);
            let qa = kb.addr_off(qa, 16);
            let qh = kb.ld_b2(qa);
            let b = kb.shr(qh, g);
            let b = kb.andc(b, 0x0101_0101);
            let b = kb.shlc(b, 4);
            kb.or(n, b)
        }
        refs::Q6_K => {
            let h = kb.shrc(g, 2);
            let gg = kb.andc(g, 3);
            let lo = kb.andc(gg, 1);
            let h64 = kb.mulc(h, 64);
            let l32 = kb.mulc(lo, 32);
            let to = kb.mulc(t, 4);
            let off = kb.add(h64, l32);
            let off = kb.add(off, to);
            let a = kb.addr(blk, off, 1);
            let ql = kb.ld_b2(a);
            let hi = kb.shrc(gg, 1);
            let nsh = kb.shlc(hi, 2);
            let n = kb.shr(ql, nsh);
            let n = kb.and(n, mask);
            let h32 = kb.mulc(h, 32);
            let off = kb.add(h32, to);
            let qa = kb.addr(blk, off, 1);
            let qa = kb.addr_off(qa, 128);
            let qh = kb.ld_b2(qa);
            let gsh = kb.shlc(gg, 1);
            let b = kb.shr(qh, gsh);
            let b = kb.andc(b, 0x0303_0303);
            let b = kb.shlc(b, 4);
            let v = kb.or(n, b);
            let c32 = kb.c32(0x2020_2020);
            kb.vsub4(v, c32)
        }
        _ => unreachable!(),
    }
}

/// `(sc, m)` of sub-block `j` (0..8) of a Q4_K/Q5_K block (ggml `get_scale_min_k4`).
fn scale_min_k4(kb: &mut Kb, blk: V, j: V) -> (V, V) {
    let sa = kb.addr_off(blk, 4);
    let four = kb.c32(4);
    let ge = kb.ult(j, four);
    let jp4 = kb.addc(j, 4);
    let jm4 = kb.sub(j, four);
    let a_idx = j;
    let b_idx = jp4;
    let c_idx = kb.select(Ty::U32, ge, j, jm4);
    let ld = |kb: &mut Kb, i: V| {
        let a = kb.addr(sa, i, 1);
        kb.ld_u8(a)
    };
    let sj = ld(kb, a_idx);
    let sj4 = ld(kb, b_idx);
    let sc_ = ld(kb, c_idx);
    // j < 4
    let sc_lo = kb.andc(sj, 63);
    let m_lo = kb.andc(sj4, 63);
    // j >= 4: sc = (s[j+4] & 15) | ((s[j-4] >> 6) << 4); m = (s[j+4] >> 4) | ((s[j] >> 6) << 4)
    let a = kb.andc(sj4, 15);
    let b = kb.shrc(sc_, 6);
    let b = kb.shlc(b, 4);
    let sc_hi = kb.or(a, b);
    let a = kb.shrc(sj4, 4);
    let b = kb.shrc(sj, 6);
    let b = kb.shlc(b, 4);
    let m_hi = kb.or(a, b);
    let sc = kb.select(Ty::U32, ge, sc_lo, sc_hi);
    let m = kb.select(Ty::U32, ge, m_lo, m_hi);
    (sc, m)
}

pub(crate) fn build(ty: u32, dot: bool) -> Vec<u32> {
    let sp = spec(ty);
    let xd: u32 = TILE_I * S * 8;
    let yq: u32 = xd + TILE_I * S * sp.xs;
    let yd: u32 = yq + TILE_J * S * 8;
    let shared_words: u32 = yd + TILE_J * S * sp.ys;
    const XQ: u32 = 0;
    let mut kb = Kb::new(&entry_name(ty), &ARGS, shared_words, false, dot);
    let p = kb.params.clone();
    let x = kb.ptr_to_u64(p[0]);
    let y = kb.ptr_to_u64(p[1]);
    let dst = kb.ptr_to_u64(p[4]);
    let (ncols_x, nrows_x, ncols_dst, srx, ncols_y, scd) = (p[6], p[7], p[8], p[9], p[10], p[11]);
    let (chr, nch_y, scx, scy, scd_) = (p[12], p[13], p[14], p[15], p[16]);
    let (smr, ssx, ssy, ssd) = (p[17], p[19], p[20], p[21]);

    let tx = kb.local_id(0);
    let ty_ = kb.local_id(1);
    let tid = {
        let a = kb.shlc(ty_, 4);
        kb.add(a, tx)
    };
    let bx = kb.group_id(0);
    let by = kb.group_id(1);
    let bz = kb.group_id(2);
    let wt = kb.udiv(bz, nch_y);
    let zt = {
        let t = kb.mul(wt, nch_y);
        kb.sub(bz, t)
    };
    let row_base = kb.mulc(bx, TILE_I);
    let col_base = kb.mulc(by, TILE_J);
    let nb = kb.shrc(ncols_x, 5);
    let xoff = {
        let a = kb.udiv(wt, smr);
        let a = kb.mul(a, ssx);
        let b = kb.udiv(zt, chr);
        let b = kb.mul(b, scx);
        kb.add(a, b)
    };
    let yoff = {
        let a = kb.mul(wt, ssy);
        let b = kb.mul(zt, scy);
        kb.add(a, b)
    };
    let mut acc = Vec::new();
    let mut tmp = Vec::new();
    for _ in 0..8 {
        let z = kb.cf(0.0);
        acc.push(kb.var(Ty::F32, z));
        let z = kb.cf(0.0);
        tmp.push(kb.var(Ty::F32, z));
    }
    let zero = kb.c32(0);
    let step = kb.c32(S);
    kb.for_(zero, nb, step, |kb, kb0| {
        // ---- stage x quants: TILE_I rows x S groups x 8 ints
        for k in 0..(TILE_I * S * 8 / 256) {
            let e = kb.addc(tid, 256 * k);
            let r = kb.shrc(e, 5);
            let rem = kb.andc(e, 31);
            let s = kb.shrc(rem, 3);
            let t = kb.andc(rem, 7);
            let row = kb.add(row_base, r);
            let grp = kb.add(kb0, s);
            let v1 = kb.ult(row, nrows_x);
            let v2 = kb.ult(grp, nb);
            let valid = kb.land(v1, v2);
            let zero = kb.c32(0);
            let val = kb.var(Ty::U32, zero);
            kb.if_(valid, |kb| {
                let rs = kb.mul(row, srx);
                let idx = kb.add(xoff, rs);
                let bidx = kb.shrc(grp, sp.sb_shift);
                let idx = kb.add(idx, bidx);
                let blk = kb.addr(x, idx, sp.bytes);
                let g = kb.andc(grp, (1 << sp.sb_shift) - 1);
                let v = x_word(kb, ty, blk, g, t);
                kb.set(val, v);
            });
            let v = kb.get(val);
            let si = kb.addc(e, XQ);
            kb.sh_st(si, v);
        }
        // ---- stage x scales: one (row, group) per thread
        {
            let r = kb.shrc(tid, 2);
            let s = kb.andc(tid, 3);
            let row = kb.add(row_base, r);
            let grp = kb.add(kb0, s);
            let v1 = kb.ult(row, nrows_x);
            let v2 = kb.ult(grp, nb);
            let valid = kb.land(v1, v2);
            let mut vals = Vec::new();
            for _ in 0..sp.xs {
                let zf = kb.cf(0.0);
                vals.push(kb.var(Ty::F32, zf));
            }
            kb.if_(valid, |kb| {
                let rs = kb.mul(row, srx);
                let idx = kb.add(xoff, rs);
                let bidx = kb.shrc(grp, sp.sb_shift);
                let idx = kb.add(idx, bidx);
                let blk = kb.addr(x, idx, sp.bytes);
                let g = kb.andc(grp, (1 << sp.sb_shift) - 1);
                match sp.kind {
                    Kind::A => {
                        let d = kb.ld_f16(blk);
                        kb.set(vals[0], d);
                    }
                    Kind::K => {
                        let d = kb.ld_f16(blk);
                        let a2 = kb.addr_off(blk, 2);
                        let dmin = kb.ld_f16(a2);
                        let m1 = kb.cf(-1.0);
                        let ndmin = kb.fmul(dmin, m1);
                        let (sc, m) = scale_min_k4(kb, blk, g);
                        let scf = kb.s2f(sc);
                        let mf = kb.s2f(m);
                        let px = kb.fmul(d, scf);
                        let py = kb.fmul(ndmin, mf);
                        for (slot, pv) in [(0usize, px), (1, py)] {
                            let bits = kb.f2h_bits(pv);
                            let r = kb.h2f(bits);
                            kb.set(vals[slot], r);
                        }
                    }
                    Kind::Q6 => {
                        let da = kb.addr_off(blk, 208);
                        let d = kb.ld_f16(da);
                        kb.set(vals[0], d);
                        let g2 = kb.shlc(g, 1);
                        for (slot, add) in [(1usize, 0u32), (2, 1)] {
                            let i = kb.addc(g2, add);
                            let a = kb.addr(blk, i, 1);
                            let a = kb.addr_off(a, 192);
                            let b = kb.ld_u8(a);
                            let b = kb.shlc(b, 24);
                            let b = kb.sarc(b, 24);
                            let bf = kb.bitcast(Ty::F32, b);
                            kb.set(vals[slot], bf);
                        }
                    }
                }
            });
            for (w, v) in vals.iter().enumerate() {
                let value = kb.get(*v);
                let base = kb.mulc(tid, sp.xs);
                let si = kb.addc(base, xd + w as u32);
                kb.sh_st_f(si, value);
            }
        }
        // ---- stage y quants: TILE_J cols x S groups x 8 ints
        for k in 0..(TILE_J * S * 8 / 256) {
            let e = kb.addc(tid, 256 * k);
            let c = kb.shrc(e, 5);
            let rem = kb.andc(e, 31);
            let s = kb.shrc(rem, 3);
            let t = kb.andc(rem, 7);
            let col = kb.add(col_base, c);
            let blk = kb.add(kb0, s);
            let v1 = kb.ult(col, ncols_dst);
            let v2 = kb.ult(blk, nb);
            let valid = kb.land(v1, v2);
            let zero = kb.c32(0);
            let val = kb.var(Ty::U32, zero);
            kb.if_(valid, |kb| {
                let idx = y_index(kb, yoff, blk, ncols_y, col);
                let w = kb.add(idx, t);
                let b3 = kb.andc(blk, 3);
                let o = kb.mulc(b3, 8);
                let w = kb.add(w, o);
                let w = kb.addc(w, 4);
                let a = kb.addr(y, w, 4);
                let v = kb.ld_b4(a);
                kb.set(val, v);
            });
            let v = kb.get(val);
            let si = kb.addc(e, yq);
            kb.sh_st(si, v);
        }
        // ---- stage y scales (d, and the block sum for DS4)
        {
            let lt = kb.c32(TILE_J * S);
            let inb = kb.ult(tid, lt);
            kb.if_(inb, |kb| {
                let c = kb.shrc(tid, 2);
                let s = kb.andc(tid, 3);
                let col = kb.add(col_base, c);
                let blk = kb.add(kb0, s);
                let v1 = kb.ult(col, ncols_dst);
                let v2 = kb.ult(blk, nb);
                let valid = kb.land(v1, v2);
                let mut vals = Vec::new();
                for _ in 0..sp.ys {
                    let zf = kb.cf(0.0);
                    vals.push(kb.var(Ty::F32, zf));
                }
                kb.if_(valid, |kb| {
                    let idx = y_index(kb, yoff, blk, ncols_y, col);
                    let b3 = kb.andc(blk, 3);
                    let w = kb.add(idx, b3);
                    let a = kb.addr(y, w, 4);
                    if sp.ds4 {
                        let word = kb.ld_b4(a);
                        let lo = kb.andc(word, 0xffff);
                        let hi = kb.shrc(word, 16);
                        let lo = kb.uconv(Ty::U16, lo);
                        let hi = kb.uconv(Ty::U16, hi);
                        let d = kb.h2f(lo);
                        let sm = kb.h2f(hi);
                        kb.set(vals[0], d);
                        kb.set(vals[1], sm);
                    } else {
                        let d = kb.ld(Ty::F32, a, 4);
                        kb.set(vals[0], d);
                    }
                });
                for (w, v) in vals.iter().enumerate() {
                    let value = kb.get(*v);
                    let base = kb.mulc(tid, sp.ys);
                    let si = kb.addc(base, yd + w as u32);
                    kb.sh_st_f(si, value);
                }
            });
        }
        kb.barrier();
        // ---- compute
        for s in 0..S {
            let blk = kb.addc(kb0, s);
            let live = kb.ult(blk, nb);
            kb.if_(live, |kb| {
                let mut ys = Vec::new();
                for ci in 0..2u32 {
                    let c = {
                        let a = kb.shlc(tx, 1);
                        kb.addc(a, ci)
                    };
                    let base = {
                        let a = kb.mulc(c, S * 8);
                        kb.addc(a, yq + s * 8)
                    };
                    let mut w = Vec::new();
                    for t in 0..8u32 {
                        let i = kb.addc(base, t);
                        w.push(kb.sh_ld(i));
                    }
                    let di = {
                        let a = kb.mulc(c, S * sp.ys);
                        kb.addc(a, yd + s * sp.ys)
                    };
                    let d = kb.sh_ld_f(di);
                    let sm = if sp.ds4 {
                        let di1 = kb.addc(di, 1);
                        Some(kb.sh_ld_f(di1))
                    } else {
                        None
                    };
                    ys.push((w, d, sm));
                }
                for ri in 0..4u32 {
                    let r = {
                        let a = kb.shlc(ty_, 2);
                        kb.addc(a, ri)
                    };
                    let base = {
                        let a = kb.mulc(r, S * 8);
                        kb.addc(a, XQ + s * 8)
                    };
                    let mut xw = Vec::new();
                    for t in 0..8u32 {
                        let i = kb.addc(base, t);
                        xw.push(kb.sh_ld(i));
                    }
                    let di = {
                        let a = kb.mulc(r, S * sp.xs);
                        kb.addc(a, xd + s * sp.xs)
                    };
                    let xs: Vec<V> = (0..sp.xs)
                        .map(|w| {
                            let i = kb.addc(di, w);
                            if sp.kind == Kind::Q6 && w > 0 { kb.sh_ld(i) } else { kb.sh_ld_f(i) }
                        })
                        .collect();
                    for ci in 0..2usize {
                        let slot = (ri * 2) as usize + ci;
                        match sp.kind {
                            Kind::A => {
                                let mut sumi = kb.c32(0);
                                for t in 0..8 {
                                    sumi = kb.dp4a(xw[t], ys[ci].0[t], sumi);
                                }
                                let sf = kb.s2f(sumi);
                                let m = kb.fmul(xs[0], sf);
                                let cur = kb.get(acc[slot]);
                                let n = kb.fma(m, ys[ci].1, cur);
                                kb.set(acc[slot], n);
                            }
                            Kind::K => {
                                let mut sumi = kb.c32(0);
                                for t in 0..8 {
                                    sumi = kb.dp4a(xw[t], ys[ci].0[t], sumi);
                                }
                                let sf = kb.s2f(sumi);
                                let tt = kb.fmul(xs[0], ys[ci].1);
                                let cur = kb.get(acc[slot]);
                                let n = kb.fma(tt, sf, cur);
                                let n = kb.fma(xs[1], ys[ci].2.unwrap(), n);
                                kb.set(acc[slot], n);
                            }
                            Kind::Q6 => {
                                let mut c0 = kb.c32(0);
                                let mut c1 = kb.c32(0);
                                for t in 0..4 {
                                    c0 = kb.dp4a(xw[t], ys[ci].0[t], c0);
                                    c1 = kb.dp4a(xw[t + 4], ys[ci].0[t + 4], c1);
                                }
                                let a = kb.mul(c0, xs[1]);
                                let b = kb.mul(c1, xs[2]);
                                let v = kb.add(a, b);
                                let vf = kb.s2f(v);
                                let cur = kb.get(tmp[slot]);
                                let n = kb.fma(vf, ys[ci].1, cur);
                                kb.set(tmp[slot], n);
                            }
                        }
                    }
                }
            });
        }
        if sp.kind == Kind::Q6 {
            // sum += tmp * d per 128 values; the four groups of a stage share one block
            for ri in 0..4u32 {
                let r = {
                    let a = kb.shlc(ty_, 2);
                    kb.addc(a, ri)
                };
                let di = {
                    let a = kb.mulc(r, S * sp.xs);
                    kb.addc(a, xd)
                };
                let d = kb.sh_ld_f(di);
                for ci in 0..2usize {
                    let slot = (ri * 2) as usize + ci;
                    let t = kb.get(tmp[slot]);
                    let cur = kb.get(acc[slot]);
                    let n = kb.fma(t, d, cur);
                    kb.set(acc[slot], n);
                    let z = kb.cf(0.0);
                    kb.set(tmp[slot], z);
                }
            }
        }
        kb.barrier();
    });
    // ---- write back
    let out_base = {
        let a = kb.mul(wt, ssd);
        let b = kb.mul(zt, scd_);
        kb.add(a, b)
    };
    for ri in 0..4u32 {
        for ci in 0..2u32 {
            let row = {
                let a = kb.shlc(ty_, 2);
                let a = kb.addc(a, ri);
                kb.add(row_base, a)
            };
            let col = {
                let a = kb.shlc(tx, 1);
                let a = kb.addc(a, ci);
                kb.add(col_base, a)
            };
            let v1 = kb.ult(row, nrows_x);
            let v2 = kb.ult(col, ncols_dst);
            let ok = kb.land(v1, v2);
            let value = kb.get(acc[(ri * 2 + ci) as usize]);
            kb.if_(ok, |kb| {
                let cs = kb.mul(col, scd);
                let idx = kb.add(out_base, cs);
                let idx = kb.add(idx, row);
                let a = kb.addr(dst, idx, 4);
                kb.st(Ty::F32, a, value, 4);
            });
        }
    }
    kb.finish()
}

/// Int index of the `block_q8_1_mmq` holding k block `blk` of column `col`.
fn y_index(kb: &mut Kb, yoff: u32, blk: u32, ncols_y: u32, col: u32) -> u32 {
    let chunk = kb.shrc(blk, 2);
    let a = kb.mul(chunk, ncols_y);
    let a = kb.add(a, col);
    let a = kb.mulc(a, 36);
    kb.add(yoff, a)
}
