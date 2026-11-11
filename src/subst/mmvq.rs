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
    /// Matrix rows per block (`MmvqAbi::rows_per_block`): 1 or 2, or `nwarps` for small_k.
    pub rpb: u32,
}

impl Cfg {
    /// The v0.12.6 decomposition: one row per block for one column, two otherwise.
    pub(crate) fn legacy(ty: u32, ncols: u32, nwarps: u32) -> Self {
        Self { ty, ncols, nwarps, rpb: if ncols == 1 { 1 } else { 2 } }
    }
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
    /// u64 address of the x block, and of the first aligned q8_1 block.
    x: V,
    y: V,
    /// Runtime-opaque 0x01010101 and 0x20202020 (never constants in dot operands).
    ones: V,
    bias: V,
}

/// Weight-side values of one block, loaded once per (thread, row, k step) and reused for every
/// activation column.
struct XData {
    v: Vec<V>,
}

fn fneg4(kb: &mut Kb, v: V) -> V {
    let k = kb.cf(-4.0);
    kb.fmul(v, k)
}

fn load_x(kb: &mut Kb, ty: u32, r: &Row, kqs: V) -> XData {
    let x = r.x;
    let mut v = Vec::new();
    match ty {
        Q8_0 => {
            v.push(kb.ld_f16(x));
            let off = kb.mulc(kqs, 4);
            let xo = kb.addr(x, off, 1);
            for i in 0..2u32 {
                let xa = kb.addr_off(xo, 2 + 4 * i);
                v.push(kb.ld_b2(xa));
            }
        }
        Q4_0 => {
            v.push(kb.ld_f16(x));
            let off = kb.mulc(kqs, 4);
            let xo = kb.addr(x, off, 1);
            for i in 0..2u32 {
                let xa = kb.addr_off(xo, 2 + 4 * i);
                let w = kb.ld_b2(xa);
                v.push(kb.andc(w, 0x0F0F_0F0F));
                let sh = kb.shrc(w, 4);
                v.push(kb.andc(sh, 0x0F0F_0F0F));
            }
        }
        Q4_K => {
            let half = kb.shrc(kqs, 1);
            let hw = kb.andc(half, 3);
            let g = kb.shrc(half, 2); // j = bq8_offset / 2
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
            // v = [v0 nib0, v1 nib0, v0 nib1, v1 nib1, sc0, sc1, m0, m1, dm0, dm1, bq8, hw]
            v.push(kb.andc(v0, 0x0F0F_0F0F));
            v.push(kb.andc(v1, 0x0F0F_0F0F));
            let a = kb.shrc(v0, 4);
            v.push(kb.andc(a, 0x0F0F_0F0F));
            let b = kb.shrc(v1, 4);
            v.push(kb.andc(b, 0x0F0F_0F0F));
            v.push(kb.andc(aux0, 0xff));
            let t = kb.shrc(aux0, 8);
            v.push(kb.andc(t, 0xff));
            v.push(kb.andc(aux1, 0xff));
            let t = kb.shrc(aux1, 8);
            v.push(kb.andc(t, 0xff));
            v.push(kb.ld_f16(x));
            let xa2 = kb.addr_off(x, 2);
            v.push(kb.ld_f16(xa2));
            v.push(bq8);
            v.push(hw);
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
            // v = [vi0, vi1, sc0, sc1, d, bq8, u_idx]: weights repacked once, bias is runtime-opaque
            let mut sc = Vec::new();
            for i in 0..2u32 {
                let sa = {
                    let a = kb.addr(x, scale_off, 1);
                    kb.addr_off(a, 192 + 4 * i)
                };
                let s = kb.ld_u8(sa);
                let s = kb.shlc(s, 24);
                sc.push(kb.sarc(s, 24));
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
                v.push(kb.vsub4(comb, r.bias));
            }
            v.extend(sc);
            let d = {
                let a = kb.addr_off(x, 208);
                kb.ld_f16(a)
            };
            v.push(d);
            v.push(bq8);
            v.push(kb.andc(kqs, 7));
        }
        _ => unreachable!(),
    }
    XData { v }
}

/// `acc + vec_dot(x block, y column block)` with the weight side already loaded.
fn dot_y(kb: &mut Kb, ty: u32, r: &Row, xd: &XData, kqs: V, acc: V) -> V {
    let (y, v) = (r.y, &xd.v);
    match ty {
        Q8_0 => {
            let off = kb.mulc(kqs, 4);
            let yo = kb.addr(y, off, 1);
            let mut sumi = kb.c32(0);
            for i in 0..2usize {
                let ya = kb.addr_off(yo, 4 + 4 * i as u32);
                let u = kb.ld_b4(ya);
                sumi = kb.dp4a(v[1 + i], u, sumi);
            }
            let d1 = kb.ld_f16(y);
            let prod = kb.fmul(v[0], d1);
            let sf = kb.s2f(sumi);
            kb.fma(prod, sf, acc)
        }
        Q4_0 => {
            let off = kb.mulc(kqs, 4);
            let yo = kb.addr(y, off, 1);
            let mut sumi = kb.c32(0);
            for i in 0..2usize {
                let u0 = {
                    let a = kb.addr_off(yo, 4 + 4 * i as u32);
                    kb.ld_b4(a)
                };
                let u1 = {
                    let a = kb.addr_off(yo, 4 + 4 * i as u32 + 16);
                    kb.ld_b4(a)
                };
                sumi = kb.dp4a(v[1 + 2 * i], u0, sumi);
                sumi = kb.dp4a(v[2 + 2 * i], u1, sumi);
            }
            let dsx = kb.ld_f16(y);
            let ys = kb.addr_off(y, 2);
            let dsy = kb.ld_f16(ys);
            let m4 = fneg4(kb, dsy);
            let sf = kb.s2f(sumi);
            let inner = kb.fma(dsx, sf, m4);
            kb.fma(v[0], inner, acc)
        }
        Q4_K => {
            let (bq8, hw) = (v[10], v[11]);
            let zero_f = kb.cf(0.0);
            let mut sumf_d = zero_f;
            let mut sumf_m = zero_f;
            let ones = r.ones;
            for i in 0..2usize {
                let blk = kb.addc(bq8, i as u32);
                let yb = kb.addr(y, blk, 36);
                let d8 = kb.ld_f16(yb);
                let uo = kb.mulc(hw, 4);
                let ua = kb.addr(yb, uo, 1);
                let ua0 = kb.addr_off(ua, 4);
                let ua1 = kb.addr_off(ua, 4 + 16);
                let u0 = kb.ld_b4(ua0);
                let u1 = kb.ld_b4(ua1);
                let z = kb.c32(0);
                let t = kb.dp4a(v[2 * i], u0, z);
                let dot1 = kb.dp4a(v[2 * i + 1], u1, t);
                let t = kb.dp4a(ones, u0, z);
                let dot2 = kb.dp4a(ones, u1, t);
                let a = kb.mul(dot1, v[4 + i]);
                let a = kb.s2f(a);
                sumf_d = kb.fma(d8, a, sumf_d);
                let b = kb.mul(dot2, v[6 + i]);
                let b = kb.s2f(b);
                sumf_m = kb.fma(d8, b, sumf_m);
            }
            let p = kb.fmul(sumf_d, v[8]);
            let q = kb.fmul(sumf_m, v[9]);
            let diff = kb.fsub(p, q);
            kb.fadd(acc, diff)
        }
        Q6_K => {
            let (bq8, u_idx) = (v[5], v[6]);
            let mut sumf = kb.cf(0.0);
            for i in 0..2usize {
                let blk = kb.addc(bq8, 2 * i as u32);
                let yb = kb.addr(y, blk, 36);
                let uo = kb.mulc(u_idx, 4);
                let ua = kb.addr(yb, uo, 1);
                let ua = kb.addr_off(ua, 4);
                let u = kb.ld_b4(ua);
                let d8 = kb.ld_f16(yb);
                let z = kb.c32(0);
                let dot = kb.dp4a(v[i], u, z);
                let prod = kb.mul(dot, v[2 + i]);
                let pf = kb.s2f(prod);
                sumf = kb.fma(d8, pf, sumf);
            }
            kb.fma(v[4], sumf, acc)
        }
        _ => unreachable!(),
    }
}

pub(crate) fn build(cfg: Cfg, dot: bool) -> Vec<u32> {
    let t = refs::traits(cfg.ty).expect("unsupported type");
    let (ncols, nw) = (cfg.ncols, cfg.nwarps);
    let rpb = cfg.rpb;
    assert!(rpb >= 1 && (ncols == 1 || rpb == 2), "rows per block {rpb} for ncols {ncols}");
    let shared_words = if nw > 1 { (nw - 1) * ncols * rpb * 32 } else { 0 };
    let name = format!("cfi_mmvq_t{}_c{}_w{}_r{}", cfg.ty, ncols, nw, rpb);
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
            let yas: Vec<V> = (0..ncols)
                .map(|j| {
                    let jy = kb.mulc(scy, j);
                    let yidx = kb.add(jy, kby);
                    kb.addr(y_base, yidx, 36)
                })
                .collect();
            for i in 0..rpb {
                let ri = kb.mulc(srx, i);
                let xi = kb.add(kbx_offset, ri);
                let xi = kb.add(xi, k);
                let xa = kb.addr(vx, xi, t.block_bytes as u32);
                let tmp = &tmp;
                let yas = &yas;
                let body = |kb: &mut Kb| {
                    // weights are loaded once and reused for every activation column
                    let xd = load_x(kb, cfg.ty, &Row { x: xa, y: xa, ones, bias }, kqs);
                    for j in 0..ncols {
                        let slot = tmp[(j * rpb + i) as usize];
                        let acc = kb.get(slot);
                        let r = dot_y(kb, cfg.ty, &Row { x: xa, y: yas[j as usize], ones, bias }, &xd, kqs, acc);
                        kb.set(slot, r);
                    }
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
        // lane i writes row i's sum (rows per block may exceed 2 for the 0.40 small_k mode)
        let mut value = vals[0];
        for (i, &v) in vals.iter().enumerate().skip(1) {
            let ci = kb.c32(i as u32);
            let is_i = kb.eq(tidx, ci);
            value = kb.select(Ty::F32, is_i, v, value);
        }
        kb.if_(cond, |kb| {
            let jo = kb.mulc(scd, j);
            let idx = kb.add(jo, tidx);
            let a = kb.addr(dst_base, idx, 4);
            kb.st(Ty::F32, a, value, 4);
        });
    }
    kb.finish()
}

/// Argument slots of the decode-fast variant: the stock 23 words plus `nrows` (appended by the
/// driver, see `subst_hook`).
pub(crate) const FAST_ARGS: [Arg; 24] = {
    let mut a = [Arg::U32; 24];
    a[0] = Arg::Ptr;
    a[1] = Arg::Ptr;
    a[2] = Arg::Ptr;
    a[3] = Arg::Ptr;
    a
};

/// Decode-fast `mul_mat_vec_q<ty, 1>` (`CUDA_FOR_INTEL_MMVQ_FAST`): `warps` 32-lane subgroups per
/// row, `rows` rows per workgroup. Each lane runs ggml's per-thread `vec_dot` over every
/// `warps * 32 / tpb`-th block with `unroll` independent accumulators (more loads in flight), each
/// subgroup reduces with one butterfly, and for `warps > 1` the per-warp sums are added in warp
/// order through shared memory. Same products as stock; only the add order differs.
///
/// Launch: block `[32, rows * warps, 1]`, grid `[ceil(nrows / rows), channels, samples]`; the
/// driver appends `nrows` after the 23 stock words.
pub(crate) fn build_fast(ty: u32, f: Fast, dot: bool) -> Vec<u32> {
    let Fast { rows, warps, unroll, wide } = f;
    let t = refs::traits(ty).expect("unsupported type");
    assert!(!wide || matches!(ty, Q8_0 | Q4_K | Q6_K), "no wide variant for type {ty}");
    assert!((1..=8).contains(&unroll) && warps.is_power_of_two() && warps <= 8);
    let name = fast_entry_name(ty, f);
    let shared = if warps > 1 { rows * warps } else { 0 };
    let mut kb = Kb::new(&name, &FAST_ARGS, shared, true, dot);
    let p = kb.params.clone();
    let (vx, vy, ids, dst) = (p[0], p[1], p[2], p[3]);
    let (ncols_x, nch_y) = (p[4], [p[5], p[6], p[7]]);
    let chr = [p[11], p[12], p[13]];
    let (scx_, scy_, scd_) = (p[14], p[15], p[16]);
    let smr = [p[17], p[18], p[19]];
    let (ssx, ssy, ssd) = (p[20], p[21], p[22]);
    let srx = p[8];
    let nrows = p[23];
    let oz = kb.shrc(ncols_x, 31);
    kb.opaque_zero = Some(oz);
    let ones = kb.opaque_const(0x0101_0101);
    let bias = kb.opaque_const(0x2020_2020);
    let vx = kb.ptr_to_u64(vx);
    let vy = kb.ptr_to_u64(vy);
    let ids = kb.ptr_to_u64(ids);
    let dst = kb.ptr_to_u64(dst);

    let lane = kb.local_id(0);
    let sg = kb.local_id(1);
    let bx = kb.group_id(0);
    let by = kb.group_id(1);
    let bz = kb.group_id(2);
    let (rsub, wi) = if warps > 1 { (kb.shrc(sg, log2(warps)), kb.andc(sg, warps - 1)) } else { (sg, kb.c32(0)) };
    let row0 = {
        let a = kb.mulc(bx, rows);
        kb.add(a, rsub)
    };
    let valid = kb.ult(row0, nrows);
    if warps == 1 {
        // whole subgroups leave together: row0 is uniform across the 32 lanes, and no barrier follows
        let past = kb.ule(nrows, row0);
        kb.if_(past, |kb| kb.ret());
    }

    let blocks_per_row_x = kb.shrc(ncols_x, log2(t.qk as u32));
    // lanes per x block: ggml's qi / vdr, or one lane per 32-value group for `wide`
    let tpb = if wide { (t.qk / 32) as u32 } else { (t.qi / t.vdr) as u32 };
    let step = warps * 32 / tpb;

    let channel_x = kb.var(Ty::U32, by);
    let channel_y = kb.var(Ty::U32, by);
    let fd_x = fastdiv(&mut kb, by, chr[0], chr[1]);
    kb.set(channel_x, fd_x);
    {
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
    let accs: Vec<_> = (0..unroll)
        .map(|_| {
            let z = kb.cf(0.0);
            kb.var(Ty::F32, z)
        })
        .collect();
    let tid = {
        let a = kb.shlc(wi, 5);
        kb.add(a, lane)
    };
    let kqs = {
        let m = kb.andc(tid, tpb - 1);
        if wide { m } else { kb.mulc(m, t.vdr as u32) }
    };
    let one_block = |kb: &mut Kb, k: V, slot: super::spv::Var| {
        let kby = kb.mulc(k, (t.qk / 32) as u32);
        let kby = if wide { kb.add(kby, kqs) } else { kby };
        let ya = kb.addr(y_base, kby, 36);
        let xi = kb.add(kbx_offset, k);
        let xa = kb.addr(vx, xi, t.block_bytes as u32);
        let acc = kb.get(slot);
        let r = if wide {
            wide_dot(kb, ty, xa, ya, kqs, ones, acc)
        } else {
            let xd = load_x(kb, ty, &Row { x: xa, y: xa, ones, bias }, kqs);
            dot_y(kb, ty, &Row { x: xa, y: ya, ones, bias }, &xd, kqs, acc)
        };
        kb.set(slot, r);
    };
    let body = |kb: &mut Kb| {
        let start = kb.shrc(tid, log2(tpb));
        let kbx = kb.var(Ty::U32, start);
        kb.while_(
            |kb| {
                let v = kb.get(kbx);
                kb.ult(v, blocks_per_row_x)
            },
            |kb| {
                let k = kb.get(kbx);
                one_block(kb, k, accs[0]);
                for u in 1..unroll {
                    let ku = kb.addc(k, u * step);
                    let inb = kb.ult(ku, blocks_per_row_x);
                    let slot = accs[u as usize];
                    kb.if_(inb, |kb| one_block(kb, ku, slot));
                }
                let n = kb.addc(k, unroll * step);
                kb.set(kbx, n);
            },
        );
    };
    if warps == 1 {
        body(&mut kb);
    } else {
        kb.if_(valid, body);
    }
    let mut v = kb.get(accs[0]);
    for u in 1..unroll {
        let o = kb.get(accs[u as usize]);
        v = kb.fadd(v, o);
    }
    for mask in [16u32, 8, 4, 2, 1] {
        let o = kb.shfl_xor_f(v, mask);
        v = kb.fadd(v, o);
    }
    let zero = kb.c32(0);
    let first = kb.eq(lane, zero);
    if warps > 1 {
        kb.if_(first, |kb| kb.sh_st_f(sg, v));
        kb.barrier();
    }
    let lead = if warps > 1 {
        let w0 = kb.eq(wi, zero);
        let a = kb.land(first, w0);
        kb.land(a, valid)
    } else {
        first
    };
    kb.if_(lead, |kb| {
        let mut total = v;
        for w in 1..warps {
            let idx = kb.addc(sg, w);
            let o = kb.sh_ld_f(idx);
            total = kb.fadd(total, o);
        }
        let a = kb.mul(bz, ssd);
        let b = kb.mul(by, scd_);
        let ab = kb.add(a, b);
        let ab = kb.add(ab, row0);
        let addr = kb.addr(dst, ab, 4);
        kb.st(Ty::F32, addr, total, 4);
    });
    kb.finish()
}

/// Shape of the decode-fast mmvq: rows per workgroup, warps per row, accumulators per lane.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) struct Fast {
    pub rows: u32,
    pub warps: u32,
    pub unroll: u32,
    /// Q8_0 / Q4_K / Q6_K: one lane per 32-value group, read as whole dwords (`wide_dot`).
    pub wide: bool,
}

pub(crate) fn fast_entry_name(ty: u32, f: Fast) -> String {
    format!("cfi_mmvq_fast_t{ty}_r{}_w{}_u{}{}", f.rows, f.warps, f.unroll, if f.wide { "_wide" } else { "" })
}

/// `n` little-endian dwords starting at the 2-byte-aligned address `a`, from `n` aligned dword
/// loads plus one halfword (never reading outside `[a, a + 4n)`) and a funnel shift.
fn ld_words2(kb: &mut Kb, a: V, n: u32) -> Vec<V> {
    let lo = kb.uconv(Ty::U32, a);
    let mis = kb.andc(lo, 3);
    let sh = kb.mulc(mis, 8);
    let m3 = {
        let c = kb.c64(!3u64);
        let t = kb.t(Ty::U64);
        kb.b.bitwise_and(t, None, a, c).unwrap()
    };
    let zero = kb.c32(0);
    let aligned = kb.eq(sh, zero);
    let mut w: Vec<V> = (0..n)
        .map(|i| {
            let p = kb.addr_off(m3, 4 * i);
            kb.ld_b4(p)
        })
        .collect();
    // the dword after the span is needed only when misaligned, and then only its low half: the
    // last two bytes of the span
    let tail = {
        let p = kb.addr_off(a, 4 * n - 2);
        let h = kb.ld(Ty::U16, p, 2);
        kb.uconv(Ty::U32, h)
    };
    w.push(tail);
    let c32 = kb.c32(32);
    let rsh = kb.sub(c32, sh);
    (0..n as usize)
        .map(|i| {
            let x = kb.shr(w[i], sh);
            let y = kb.shl(w[i + 1], rsh);
            let f = kb.or(x, y);
            kb.select(Ty::U32, aligned, w[i], f)
        })
        .collect()
}

/// The 8 activation dwords and the scale `d` of the q8_1 block at `ya`.
fn y_block(kb: &mut Kb, ya: V) -> (Vec<V>, V) {
    let u = (0..8u32)
        .map(|i| {
            let p = kb.addr_off(ya, 4 + 4 * i);
            kb.ld_b4(p)
        })
        .collect();
    (u, kb.ld_f16(ya))
}

/// Wide decode-fast per-lane dot products (`Fast::wide`): lane `g` of a block handles the 32
/// values of q8_1 block `g` (`ya` points at it) with whole-dword loads. Integer sums are exact;
/// the float shape is the one modelled by `refs::vec_dot_wide`.
fn wide_dot(kb: &mut Kb, ty: u32, xa: V, ya: V, g: V, ones: V, acc: V) -> V {
    let (u, d8) = y_block(kb, ya);
    match ty {
        Q8_0 => {
            // xa: d (half), qs[32]
            let dx = kb.ld_f16(xa);
            let qa = kb.addr_off(xa, 2);
            let q = ld_words2(kb, qa, 8);
            let mut sumi = kb.c32(0);
            for i in 0..8 {
                sumi = kb.dp4a(q[i], u[i], sumi);
            }
            let prod = kb.fmul(dx, d8);
            let sf = kb.s2f(sumi);
            kb.fma(prod, sf, acc)
        }
        Q4_K => {
            // d, dmin (half), scales[12], qs[128]: sub-block g = nibble (g & 1) of qs[32*(g/2)..][..32]
            let dx = kb.ld_f16(xa);
            let dm_a = kb.addr_off(xa, 2);
            let dmin = kb.ld_f16(dm_a);
            let sa = kb.addr_off(xa, 4);
            let sw = ld_words2(kb, sa, 3);
            let byte = |kb: &mut Kb, idx: V| -> V {
                // byte idx (0..12, lane-varying) of the scale words
                let wi = kb.shrc(idx, 2);
                let bi = kb.andc(idx, 3);
                let sh = kb.mulc(bi, 8);
                let one = kb.c32(1);
                let two = kb.c32(2);
                let is1 = kb.eq(wi, one);
                let is2 = kb.eq(wi, two);
                let w01 = kb.select(Ty::U32, is1, sw[1], sw[0]);
                let w = kb.select(Ty::U32, is2, sw[2], w01);
                let v = kb.shr(w, sh);
                kb.andc(v, 0xff)
            };
            // get_scale_min_k4(g)
            let four = kb.c32(4);
            let lt4 = kb.ult(g, four);
            let gp4 = kb.addc(g, 4);
            let gm4 = {
                let gm = kb.sub(g, four);
                let z = kb.c32(0);
                kb.select(Ty::U32, lt4, z, gm)
            };
            let bg = byte(kb, g);
            let bgp4 = byte(kb, gp4);
            let bgm4 = byte(kb, gm4);
            let sc_lo = kb.andc(bg, 63);
            let m_lo = kb.andc(bgp4, 63);
            let sc_hi = {
                let a = kb.andc(bgp4, 0xf);
                let b = kb.shrc(bgm4, 6);
                let b = kb.shlc(b, 4);
                kb.or(a, b)
            };
            let m_hi = {
                let a = kb.shrc(bgp4, 4);
                let b = kb.shrc(bg, 6);
                let b = kb.shlc(b, 4);
                kb.or(a, b)
            };
            let sc = kb.select(Ty::U32, lt4, sc_lo, sc_hi);
            let m = kb.select(Ty::U32, lt4, m_lo, m_hi);
            let pair = kb.shrc(g, 1);
            let qo = kb.mulc(pair, 32);
            let qo = kb.addc(qo, 16);
            let qa = kb.addr(xa, qo, 1);
            let q = ld_words2(kb, qa, 8);
            let nib = kb.andc(g, 1);
            let nsh = kb.mulc(nib, 4);
            let mut sumi = kb.c32(0);
            let mut sumy = kb.c32(0);
            for i in 0..8 {
                let v = kb.shr(q[i], nsh);
                let v = kb.andc(v, 0x0F0F_0F0F);
                sumi = kb.dp4a(v, u[i], sumi);
                sumy = kb.dp4a(ones, u[i], sumy);
            }
            let a = kb.mul(sumi, sc);
            let a = kb.s2f(a);
            let a = kb.fmul(d8, a);
            let b = kb.mul(sumy, m);
            let b = kb.s2f(b);
            let b = kb.fmul(d8, b);
            let p = kb.fmul(dx, a);
            let q = kb.fmul(dmin, b);
            let diff = kb.fsub(p, q);
            kb.fadd(acc, diff)
        }
        Q6_K => {
            // ql[128] qh[64] scales[16] d: group g -> half h = g/4, sub = g%4
            let h = kb.shrc(g, 2);
            let sub = kb.andc(g, 3);
            let qlo = {
                let a = kb.mulc(h, 64);
                let b = kb.andc(sub, 1);
                let b = kb.mulc(b, 32);
                kb.add(a, b)
            };
            let qla = kb.addr(xa, qlo, 1);
            let ql = ld_words2(kb, qla, 8);
            let qho = {
                let a = kb.mulc(h, 32);
                kb.addc(a, 128)
            };
            let qha = kb.addr(xa, qho, 1);
            let qh = ld_words2(kb, qha, 8);
            let nib = kb.shrc(sub, 1);
            let nsh = kb.mulc(nib, 4);
            let hsh = kb.mulc(sub, 2);
            // scales sc[8h + 2sub] (l < 16) and sc[8h + 2sub + 1] (l >= 16), int8
            let so = {
                let a = kb.mulc(h, 8);
                let b = kb.mulc(sub, 2);
                let s = kb.add(a, b);
                kb.addc(s, 192)
            };
            let sca = kb.addr(xa, so, 1);
            let sc2 = kb.ld(Ty::U16, sca, 2);
            let sc2 = kb.uconv(Ty::U32, sc2);
            let s0 = {
                let t = kb.shlc(sc2, 24);
                kb.sarc(t, 24)
            };
            let s1 = {
                let t = kb.shlc(sc2, 16);
                kb.sarc(t, 24)
            };
            let da = kb.addr_off(xa, 208);
            let dx = kb.ld_f16(da);
            let mut tot = kb.c32(0);
            for half in 0..2usize {
                let mut sumi = kb.c32(0);
                let mut sumu = kb.c32(0);
                for i in 4 * half..4 * half + 4 {
                    let lo = kb.shr(ql[i], nsh);
                    let lo = kb.andc(lo, 0x0F0F_0F0F);
                    let hi = kb.shr(qh[i], hsh);
                    let hi = kb.andc(hi, 0x0303_0303);
                    let hi = kb.shlc(hi, 4);
                    let v = kb.or(lo, hi);
                    sumi = kb.dp4a(v, u[i], sumi);
                    sumu = kb.dp4a(ones, u[i], sumu);
                }
                // sum((q - 32) * y) = sum(q * y) - 32 * sum(y), exact
                let s32 = kb.shlc(sumu, 5);
                let v = kb.sub(sumi, s32);
                let sc = if half == 0 { s0 } else { s1 };
                let v = kb.mul(v, sc);
                tot = kb.add(tot, v);
            }
            let prod = kb.fmul(dx, d8);
            let tf = kb.s2f(tot);
            kb.fma(prod, tf, acc)
        }
        _ => unreachable!("wide is Q8_0 / Q4_K / Q6_K"),
    }
}
