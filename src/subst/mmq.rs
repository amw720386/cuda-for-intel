//! Native `mul_mat_q<Q8_0, mmq_x, need_check>` for the non-MoE case: a register-tiled SDot kernel
//! with its own grid (no stream-k, so the paired `mul_mat_q_stream_k_fixup` must be skipped).
//!
//! Tile: 64 rows x 32 columns per 16x16 workgroup (each thread 4 rows x 2 columns), 4 Q8_0 blocks
//! (128 values of k) staged through shared memory per step. Each output accumulates its k blocks
//! sequentially with `sum = fma(dA * (float)sumi, dB, sum)`, the structure of the stock MMA path;
//! it differs from stock only on tiles stream-k splits between workgroups.
use super::spv::{Arg, Kb, Ty};

pub(crate) const TILE_I: u32 = 64;
pub(crate) const TILE_J: u32 = 32;
const S: u32 = 4; // k blocks per stage

pub(crate) const ARGS: [Arg; 23] = {
    let mut a = [Arg::U32; 23];
    let mut i = 0;
    while i < 6 {
        a[i] = Arg::Ptr;
        i += 1;
    }
    a
};

const XQ: u32 = 0;
const XD: u32 = TILE_I * S * 8;
const YQ: u32 = XD + TILE_I * S;
const YD: u32 = YQ + TILE_J * S * 8;
const SHARED_WORDS: u32 = YD + TILE_J * S;

pub(crate) fn build(dot: bool) -> Vec<u32> {
    let mut kb = Kb::new("cfi_mmq_q8_0", &ARGS, SHARED_WORDS, false, dot);
    let p = kb.params.clone();
    let x = kb.ptr_to_u64(p[0]);
    let y = kb.ptr_to_u64(p[1]);
    let dst = kb.ptr_to_u64(p[4]);
    let (ncols_x, nrows_x, ncols_dst, srx, ncols_y, scd) = (p[6], p[7], p[8], p[9], p[10], p[11]);
    let (chr, nch_y, scx, scy, scd_) = (p[12], p[13], p[14], p[15], p[16]);
    let (smr, ssx, ssy, ssd) = (p[17], p[19], p[20], p[21]);

    let tx = kb.local_id(0);
    let ty = kb.local_id(1);
    let tid = {
        let a = kb.shlc(ty, 4);
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
    for _ in 0..8 {
        let z = kb.cf(0.0);
        acc.push(kb.var(Ty::F32, z));
    }
    let zero = kb.c32(0);
    let step = kb.c32(S);
    kb.for_(zero, nb, step, |kb, kb0| {
        // ---- stage x quants: TILE_I rows x S blocks x 8 ints
        for k in 0..(TILE_I * S * 8 / 256) {
            let e = kb.addc(tid, 256 * k);
            let r = kb.shrc(e, 5);
            let rem = kb.andc(e, 31);
            let s = kb.shrc(rem, 3);
            let t = kb.andc(rem, 7);
            let row = kb.add(row_base, r);
            let blk = kb.add(kb0, s);
            let v1 = kb.ult(row, nrows_x);
            let v2 = kb.ult(blk, nb);
            let valid = kb.land(v1, v2);
            let zero = kb.c32(0);
            let val = kb.var(Ty::U32, zero);
            kb.if_(valid, |kb| {
                let rs = kb.mul(row, srx);
                let idx = kb.add(xoff, rs);
                let idx = kb.add(idx, blk);
                let a = kb.addr(x, idx, 34);
                let to = kb.mulc(t, 4);
                let a = kb.addr(a, to, 1);
                let a = kb.addr_off(a, 2);
                let v = kb.ld_b2(a);
                kb.set(val, v);
            });
            let v = kb.get(val);
            let si = kb.addc(e, XQ);
            kb.sh_st(si, v);
        }
        // ---- stage x scales: TILE_I * S floats (one per thread)
        {
            let r = kb.shrc(tid, 2);
            let s = kb.andc(tid, 3);
            let row = kb.add(row_base, r);
            let blk = kb.add(kb0, s);
            let v1 = kb.ult(row, nrows_x);
            let v2 = kb.ult(blk, nb);
            let valid = kb.land(v1, v2);
            let zf = kb.cf(0.0);
            let val = kb.var(Ty::F32, zf);
            kb.if_(valid, |kb| {
                let rs = kb.mul(row, srx);
                let idx = kb.add(xoff, rs);
                let idx = kb.add(idx, blk);
                let a = kb.addr(x, idx, 34);
                let d = kb.ld_f16(a);
                kb.set(val, d);
            });
            let v = kb.get(val);
            let si = kb.addc(tid, XD);
            kb.sh_st_f(si, v);
        }
        // ---- stage y quants: TILE_J cols x S blocks x 8 ints
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
            let si = kb.addc(e, YQ);
            kb.sh_st(si, v);
        }
        // ---- stage y scales
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
                let zf = kb.cf(0.0);
                let val = kb.var(Ty::F32, zf);
                kb.if_(valid, |kb| {
                    let idx = y_index(kb, yoff, blk, ncols_y, col);
                    let b3 = kb.andc(blk, 3);
                    let w = kb.add(idx, b3);
                    let a = kb.addr(y, w, 4);
                    let d = kb.ld(Ty::F32, a, 4);
                    kb.set(val, d);
                });
                let v = kb.get(val);
                let si = kb.addc(tid, YD);
                kb.sh_st_f(si, v);
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
                        kb.addc(a, YQ + s * 8)
                    };
                    let mut w = Vec::new();
                    for t in 0..8u32 {
                        let i = kb.addc(base, t);
                        w.push(kb.sh_ld(i));
                    }
                    let di = {
                        let a = kb.mulc(c, S);
                        kb.addc(a, YD + s)
                    };
                    ys.push((w, kb.sh_ld_f(di)));
                }
                for ri in 0..4u32 {
                    let r = {
                        let a = kb.shlc(ty, 2);
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
                        let a = kb.mulc(r, S);
                        kb.addc(a, XD + s)
                    };
                    let d_a = kb.sh_ld_f(di);
                    for ci in 0..2usize {
                        let mut sumi = kb.c32(0);
                        for t in 0..8 {
                            sumi = kb.dp4a(xw[t], ys[ci].0[t], sumi);
                        }
                        let sf = kb.s2f(sumi);
                        let m = kb.fmul(d_a, sf);
                        let slot = acc[(ri * 2) as usize + ci];
                        let cur = kb.get(slot);
                        let n = kb.fma(m, ys[ci].1, cur);
                        kb.set(slot, n);
                    }
                }
            });
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
                let a = kb.shlc(ty, 2);
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
