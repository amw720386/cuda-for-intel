//! Native `mul_mat_vec_q<type, ncols_dst>` (ggml-cuda mmvq.cu): same work decomposition,
//! accumulation and reduction order as the CUDA kernel (so the float result matches bit for bit
//! up to the fusion pattern documented in `refs.rs`), written against SPIR-V integer dot product.
//!
//! Launch mapping: grid/block are used unchanged (block = 32 x nwarps, 32-lane subgroups).
//! `nwarps` may be overridden for experiments; that changes only the cross-warp add order.
use super::refs::{self, Q4_0, Q4_K, Q6_K, Q8_0};
use super::spv::{Arg, Kb, Ty, V};

/// Kernel argument slots (the CUDA parameter list with `uint3` expanded to three words).
pub(crate) const ARGS: [Arg; 23] = {
    let mut a = [Arg::U32; 23];
    a[0] = Arg::Ptr;
    a[1] = Arg::Ptr;
    a[2] = Arg::Ptr;
    a[3] = Arg::Ptr;
    a
};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct Cfg {
    pub ty: u32,
    pub ncols: u32,
    pub nwarps: u32,
}

fn log2(v: u32) -> u32 {
    assert!(v.is_power_of_two());
    v.trailing_zeros()
}

/// `fastdiv(n, {mp, L, d})`.
fn fastdiv(kb: &mut Kb, n: V, mp: V, l: V) -> V {
    let n64 = kb.uconv(Ty::U64, n);
    let m64 = kb.uconv(Ty::U64, mp);
    let prod = kb.mul64(n64, m64);
    let hi = kb.shr64c(prod, 32);
    let hi = kb.uconv(Ty::U32, hi);
    let s = kb.add(hi, n);
    kb.shr(s, l)
}

struct Row {
    /// Runtime-opaque 0x01010101 and 0x20202020 (never constants in dot operands).
    ones: V,
    bias: V,
    /// u64 address of the x block, and of the first aligned q8_1 block.
    x: V,
    y: V,
}

fn fneg4(kb: &mut Kb, v: V) -> V {
    let k = kb.cf(-4.0);
    kb.fmul(v, k)
}

fn vec_dot(kb: &mut Kb, ty: u32, r: &Row, kqs: V, acc: V) -> V {
    let (x, y) = (r.x, r.y);
    let (ones_op, bias_op) = (r.ones, r.bias);
    match ty {
        Q8_0 => {
            let d0 = kb.ld_f16(x);
            let off = kb.mulc(kqs, 4);
            let xo = kb.addr(x, off, 1);
            let yo = kb.addr(y, off, 1);
            let mut sumi = kb.c32(0);
            for i in 0..2u32 {
                let xa = kb.addr_off(xo, 2 + 4 * i);
                let ya = kb.addr_off(yo, 4 + 4 * i);
                let v = kb.ld_b2(xa);
                let u = kb.ld_b4(ya);
                sumi = kb.dp4a(v, u, sumi);
            }
            let d1 = kb.ld_f16(y);
            let prod = kb.fmul(d0, d1);
            let sf = kb.s2f(sumi);
            kb.fma(prod, sf, acc)
        }
        Q4_0 => {
            let d4 = kb.ld_f16(x);
            let off = kb.mulc(kqs, 4);
            let xo = kb.addr(x, off, 1);
            let yo = kb.addr(y, off, 1);
            let mut sumi = kb.c32(0);
            for i in 0..2u32 {
                let xa = kb.addr_off(xo, 2 + 4 * i);
                let v = kb.ld_b2(xa);
                let u0 = {
                    let a = kb.addr_off(yo, 4 + 4 * i);
                    kb.ld_b4(a)
                };
                let u1 = {
                    let a = kb.addr_off(yo, 4 + 4 * i + 16);
                    kb.ld_b4(a)
                };
                let vi0 = kb.andc(v, 0x0F0F_0F0F);
                let sh = kb.shrc(v, 4);
                let vi1 = kb.andc(sh, 0x0F0F_0F0F);
                sumi = kb.dp4a(vi0, u0, sumi);
                sumi = kb.dp4a(vi1, u1, sumi);
            }
            let dsx = kb.ld_f16(y);
            let ys = kb.addr_off(y, 2);
            let dsy = kb.ld_f16(ys);
            let m4 = fneg4(kb, dsy);
            let sf = kb.s2f(sumi);
            let inner = kb.fma(dsx, sf, m4);
            kb.fma(d4, inner, acc)
        }
        Q4_K => {
            // kqs = 2 * (tid % 16); half = kqs / 2
            let half = kb.shrc(kqs, 1);
            let hw = kb.andc(half, 3);
            let g = kb.shrc(half, 2); // = j = bq8_offset / 2
            let bq8 = kb.shlc(g, 1);
            let qoff = {
                let a = kb.mulc(bq8, 16);
                let b = kb.mulc(hw, 4);
                let s = kb.add(a, b);
                kb.addc(s, 16)
            };
            let qa = kb.addr(x, qoff, 1);
            let v0 = kb.ld_b4(qa);
            let qa1 = kb.addr_off(qa, 16);
            let v1 = kb.ld_b4(qa1);
            // 6-bit scales/mins: aux0 (scales), aux1 (mins)
            let sc16 = |kb: &mut Kb, idx: V| -> V {
                let a = {
                    let o = kb.mulc(idx, 2);
                    let o = kb.addc(o, 4);
                    kb.addr(x, o, 1)
                };
                let w = kb.ld(Ty::U16, a, 2);
                kb.uconv(Ty::U32, w)
            };
            let two = kb.c32(2);
            let ge2 = {
                let c = kb.c32(2);
                let lt = kb.ult(g, c);
                let t = kb.t(Ty::Bool);
                kb.b.logical_not(t, None, lt).unwrap()
            };
            let gp2 = kb.add(g, two);
            let gm2 = {
                let d = kb.sub(g, two);
                let z = kb.c32(0);
                kb.select(Ty::U32, ge2, d, z)
            };
            let s_g = sc16(kb, g);
            let s_gp2 = sc16(kb, gp2);
            let s_gm2 = sc16(kb, gm2);
            let lo0 = kb.andc(s_g, 0x3f3f);
            let lo1 = kb.andc(s_gp2, 0x3f3f);
            let hi0 = {
                let a = kb.andc(s_gp2, 0x0f0f);
                let b = kb.andc(s_gm2, 0xc0c0);
                let b = kb.shrc(b, 2);
                kb.or(a, b)
            };
            let hi1 = {
                let a = kb.shrc(s_gp2, 4);
                let a = kb.andc(a, 0x0f0f);
                let b = kb.andc(s_g, 0xc0c0);
                let b = kb.shrc(b, 2);
                kb.or(a, b)
            };
            let aux0 = kb.select(Ty::U32, ge2, hi0, lo0);
            let aux1 = kb.select(Ty::U32, ge2, hi1, lo1);
            let sc = [kb.andc(aux0, 0xff), {
                let t = kb.shrc(aux0, 8);
                kb.andc(t, 0xff)
            }];
            let mn = [kb.andc(aux1, 0xff), {
                let t = kb.shrc(aux1, 8);
                kb.andc(t, 0xff)
            }];
            let zero_f = kb.cf(0.0);
            let mut sumf_d = zero_f;
            let mut sumf_m = zero_f;
            let ones = ones_op;
            for i in 0..2u32 {
                let blk = kb.addc(bq8, i);
                let yb = kb.addr(y, blk, 36);
                let d8 = kb.ld_f16(yb);
                let uo = kb.mulc(hw, 4);
                let ua = kb.addr(yb, uo, 1);
                let ua0 = kb.addr_off(ua, 4);
                let ua1 = kb.addr_off(ua, 4 + 16);
                let u0 = kb.ld_b4(ua0);
                let u1 = kb.ld_b4(ua1);
                let (v0i, v1i) = if i == 0 {
                    (kb.andc(v0, 0x0F0F_0F0F), kb.andc(v1, 0x0F0F_0F0F))
                } else {
                    let a = kb.shrc(v0, 4);
                    let b = kb.shrc(v1, 4);
                    (kb.andc(a, 0x0F0F_0F0F), kb.andc(b, 0x0F0F_0F0F))
                };
                let z = kb.c32(0);
                let t = kb.dp4a(v0i, u0, z);
                let dot1 = kb.dp4a(v1i, u1, t);
                let t = kb.dp4a(ones, u0, z);
                let dot2 = kb.dp4a(ones, u1, t);
                let a = kb.mul(dot1, sc[i as usize]);
                let a = kb.s2f(a);
                sumf_d = kb.fma(d8, a, sumf_d);
                let b = kb.mul(dot2, mn[i as usize]);
                let b = kb.s2f(b);
                sumf_m = kb.fma(d8, b, sumf_m);
            }
            let dm0 = kb.ld_f16(x);
            let xa2 = kb.addr_off(x, 2);
            let dm1 = kb.ld_f16(xa2);
            let p = kb.fmul(sumf_d, dm0);
            let q = kb.fmul(sumf_m, dm1);
            let diff = kb.fsub(p, q);
            kb.fadd(acc, diff)
        }
        Q6_K => {
            let hi = kb.shrc(kqs, 4);
            let lo15 = kb.andc(kqs, 15);
            let bq8 = {
                let a = kb.shlc(hi, 2);
                let b = kb.shrc(lo15, 3);
                kb.add(a, b)
            };
            let scale_off = {
                let a = kb.shlc(hi, 3);
                let b = kb.shrc(lo15, 2);
                kb.add(a, b)
            };
            let vh_shift = {
                let b = kb.shrc(lo15, 3);
                kb.shlc(b, 1)
            };
            let qo = kb.mulc(kqs, 4);
            let qa = kb.addr(x, qo, 1);
            let vl = kb.ld_b2(qa);
            let vh_idx = {
                let a = kb.shlc(hi, 3);
                let b = kb.andc(kqs, 7);
                kb.add(a, b)
            };
            let vho = kb.mulc(vh_idx, 4);
            let vha = kb.addr(x, vho, 1);
            let vha = kb.addr_off(vha, 128);
            let vh = kb.ld_b2(vha);
            let vh = kb.shr(vh, vh_shift);
            let d = {
                let a = kb.addr_off(x, 208);
                kb.ld_f16(a)
            };
            let mut sumf = kb.cf(0.0);
            let u_idx = kb.andc(kqs, 7);
            for i in 0..2u32 {
                let sa = {
                    let a = kb.addr(x, scale_off, 1);
                    kb.addr_off(a, 192 + 4 * i)
                };
                let s = kb.ld_u8(sa);
                let s = kb.shlc(s, 24);
                let sc = kb.sarc(s, 24);
                let vil = {
                    let t = if i == 0 { vl } else { kb.shrc(vl, 4) };
                    kb.andc(t, 0x0F0F_0F0F)
                };
                let vih = {
                    let t = if i == 0 { vh } else { kb.shrc(vh, 4) };
                    let t = kb.shlc(t, 4);
                    kb.andc(t, 0x3030_3030)
                };
                let comb = kb.or(vil, vih);
                let c20 = bias_op;
                let vi = kb.vsub4(comb, c20);
                let blk = {
                    let b = kb.addc(bq8, 2 * i);
                    b
                };
                let yb = kb.addr(y, blk, 36);
                let uo = kb.mulc(u_idx, 4);
                let ua = kb.addr(yb, uo, 1);
                let ua = kb.addr_off(ua, 4);
                let u = kb.ld_b4(ua);
                let d8 = kb.ld_f16(yb);
                let z = kb.c32(0);
                let dot = kb.dp4a(vi, u, z);
                let prod = kb.mul(dot, sc);
                let pf = kb.s2f(prod);
                sumf = kb.fma(d8, pf, sumf);
            }
            kb.fma(d, sumf, acc)
        }
        _ => unreachable!(),
    }
}

pub(crate) fn build(cfg: Cfg, dot: bool) -> Vec<u32> {
    let t = refs::traits(cfg.ty).expect("unsupported type");
    let (ncols, nw) = (cfg.ncols, cfg.nwarps);
    let rpb: u32 = if ncols == 1 { 1 } else { 2 };
    let shared_words = if nw > 1 { (nw - 1) * ncols * rpb * 32 } else { 0 };
    let name = format!("cfi_mmvq_t{}_c{}_w{}", cfg.ty, ncols, nw);
    let mut kb = Kb::new(&name, &ARGS, shared_words, true, dot);
    let p = kb.params.clone();
    let (vx, vy, ids, dst) = (p[0], p[1], p[2], p[3]);
    let (ncols_x, nch_y, srx, scy, scd) = (p[4], [p[5], p[6], p[7]], p[8], p[9], p[10]);
    let chr = [p[11], p[12], p[13]];
    let (scx_, scy_, scd_) = (p[14], p[15], p[16]);
    let smr = [p[17], p[18], p[19]];
    let (ssx, ssy, ssd) = (p[20], p[21], p[22]);
    // ncols_x is far below 2^31, so this is a zero only the device knows.
    let oz = kb.shrc(ncols_x, 31);
    kb.opaque_zero = Some(oz);
    let ones = kb.opaque_const(0x0101_0101);
    let bias = kb.opaque_const(0x2020_2020);
    let vx = kb.ptr_to_u64(vx);
    let vy = kb.ptr_to_u64(vy);
    let ids = kb.ptr_to_u64(ids);
    let dst = kb.ptr_to_u64(dst);

    let tidx = kb.local_id(0);
    let tidy = kb.local_id(1);
    let tid = {
        let a = kb.shlc(tidy, 5);
        kb.add(a, tidx)
    };
    let bx = kb.group_id(0);
    let by = kb.group_id(1);
    let bz = kb.group_id(2);
    let row0 = kb.mulc(bx, rpb);
    let blocks_per_row_x = kb.shrc(ncols_x, log2(t.qk as u32));
    let tpb = (t.qi / t.vdr) as u32; // threads per x block
    let blocks_per_iter = t.vdr as u32 * nw * 32 / t.qi as u32;

    // channel / sample selection
    let channel_x = kb.var(Ty::U32, by);
    let channel_y = kb.var(Ty::U32, by);
    let fd_x = fastdiv(&mut kb, by, chr[0], chr[1]);
    kb.set(channel_x, fd_x);
    if ncols == 1 {
        let zero = kb.c64(0);
        let has_ids = kb.ne(ids, zero);
        kb.if_(has_ids, |kb| {
            let a = kb.addr(ids, by, 4);
            let cx = kb.ld(Ty::U32, a, 4);
            kb.set(channel_x, cx);
            let q = fastdiv(kb, by, nch_y[0], nch_y[1]);
            let prod = kb.mul(q, nch_y[2]);
            let cy = kb.sub(by, prod);
            kb.set(channel_y, cy);
        });
    }
    let sample_x = fastdiv(&mut kb, bz, smr[0], smr[1]);

    let cy = kb.get(channel_y);
    let cx = kb.get(channel_x);
    // y base address: vy + (sample_y*stride_sample_y)*36 + (channel_y*stride_channel_y)*36
    let y_base = {
        let a = kb.mul(bz, ssy);
        let b = kb.mul(cy, scy_);
        let a = kb.addr(vy, a, 36);
        kb.addr(a, b, 36)
    };
    let kbx_offset = {
        let a = kb.mul(sample_x, ssx);
        let b = kb.mul(cx, scx_);
        let c = kb.mul(row0, srx);
        let ab = kb.add(a, b);
        kb.add(ab, c)
    };

    let mut tmp = Vec::new();
    for _ in 0..ncols * rpb {
        let z = kb.cf(0.0);
        tmp.push(kb.var(Ty::F32, z));
    }
    let kqs = {
        let m = kb.andc(tid, tpb - 1);
        kb.mulc(m, t.vdr as u32)
    };
    let start = kb.shrc(tid, log2(tpb));
    let kbx = kb.var(Ty::U32, start);
    kb.while_(
        |kb| {
            let v = kb.get(kbx);
            kb.ult(v, blocks_per_row_x)
        },
        |kb| {
            let k = kb.get(kbx);
            let kby = kb.mulc(k, (t.qk / 32) as u32);
            for j in 0..ncols {
                let jy = kb.mulc(scy, j);
                let yidx = kb.add(jy, kby);
                let ya = kb.addr(y_base, yidx, 36);
                for i in 0..rpb {
                    let ri = kb.mulc(srx, i);
                    let xi = kb.add(kbx_offset, ri);
                    let xi = kb.add(xi, k);
                    let xa = kb.addr(vx, xi, t.block_bytes as u32);
                    let slot = tmp[(j * rpb + i) as usize];
                    let body = |kb: &mut Kb| {
                        let acc = kb.get(slot);
                        let r = vec_dot(kb, cfg.ty, &Row { x: xa, y: ya, ones, bias }, kqs, acc);
                        kb.set(slot, r);
                    };
                    if i == 0 {
                        body(kb);
                    } else {
                        // The second row of a tile may lie past the matrix; never read it.
                        let r = kb.addc(row0, i);
                        let inb = kb.ult(r, scd);
                        kb.if_(inb, body);
                    }
                }
            }
            let n = kb.addc(k, blocks_per_iter);
            kb.set(kbx, n);
        },
    );

    // cross-warp reduction through shared memory
    if nw > 1 {
        let zero = kb.c32(0);
        let gt = kb.ugt(tidy, zero);
        kb.if_(gt, |kb| {
            let one = kb.c32(1);
            let w1 = kb.sub(tidy, one);
            for j in 0..ncols {
                for i in 0..rpb {
                    let v = kb.get(tmp[(j * rpb + i) as usize]);
                    let base = kb.mulc(w1, ncols * rpb * 32);
                    let off = kb.addc(base, (j * rpb + i) * 32);
                    let idx = kb.add(off, tidx);
                    kb.sh_st_f(idx, v);
                }
            }
        });
    }
    kb.barrier();
    let zero = kb.c32(0);
    let upper = kb.ugt(tidy, zero);
    kb.if_(upper, |kb| kb.ret());

    // dst += sample_dst*ssd + channel_dst*scd_ + row0
    let dst_base = {
        let a = kb.mul(bz, ssd);
        let b = kb.mul(by, scd_);
        let ab = kb.add(a, b);
        let ab = kb.add(ab, row0);
        kb.addr(dst, ab, 4)
    };
    for j in 0..ncols {
        let mut vals = Vec::new();
        for i in 0..rpb {
            let mut v = kb.get(tmp[(j * rpb + i) as usize]);
            for l in 0..nw.saturating_sub(1) {
                let off = kb.c32((((l * ncols + j) * rpb) + i) * 32);
                let idx = kb.add(off, tidx);
                let o = kb.sh_ld_f(idx);
                v = kb.fadd(v, o);
            }
            for mask in [16u32, 8, 4, 2, 1] {
                let o = kb.shfl_xor_f(v, mask);
                v = kb.fadd(v, o);
            }
            vals.push(v);
        }
        // if (tidx < rpb && (rpb == 1 || row0 + tidx < stride_col_dst)) dst[j*scd + tidx] = vals[tidx]
        let lim = kb.c32(rpb);
        let mut cond = kb.ult(tidx, lim);
        if rpb > 1 {
            let r = kb.add(row0, tidx);
            let inb = kb.ult(r, scd);
            cond = kb.land(cond, inb);
        }
        let value = if rpb == 1 {
            vals[0]
        } else {
            let z = kb.c32(0);
            let is0 = kb.eq(tidx, z);
            kb.select(Ty::F32, is0, vals[0], vals[1])
        };
        kb.if_(cond, |kb| {
            let jo = kb.mulc(scd, j);
            let idx = kb.add(jo, tidx);
            let a = kb.addr(dst_base, idx, 4);
            kb.st(Ty::F32, a, value, 4);
        });
    }
    kb.finish()
}
