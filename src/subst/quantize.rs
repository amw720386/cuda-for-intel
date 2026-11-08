//! Native `quantize_q8_1(const float *x, void *vy, i64 ne00, s01, s02, s03, ne0, u32 ne1, uint3 ne2)`.
//! Grid `(ceil(ne0/256), ne1, ne2*ne3)`, block 256; lanes of a 32-wide subgroup reduce amax and sum
//! with the same xor butterfly as ggml's `warp_reduce_*`, so the output is bit-identical.
use super::spv::{Arg, Kb, Ty};

pub(crate) const ARGS: [Arg; 11] = [
    Arg::Ptr,
    Arg::Ptr,
    Arg::U64,
    Arg::U64,
    Arg::U64,
    Arg::U64,
    Arg::U64,
    Arg::U32,
    Arg::U32,
    Arg::U32,
    Arg::U32,
];

pub(crate) fn build(dot: bool) -> Vec<u32> {
    let mut kb = Kb::new("cfi_quantize_q8_1", &ARGS, 0, true, dot);
    let p = kb.params.clone();
    let x = kb.ptr_to_u64(p[0]);
    let y = kb.ptr_to_u64(p[1]);
    let (ne00, s01, s02, s03, ne0, ne1) = (p[2], p[3], p[4], p[5], p[6], p[7]);
    let (mp, l, ne2z) = (p[8], p[9], p[10]);

    let tidx = kb.local_id(0);
    let bx = kb.group_id(0);
    let by = kb.group_id(1);
    let bz = kb.group_id(2);
    let i0 = {
        let b = kb.shlc(bx, 8);
        let s = kb.add(b, tidx);
        kb.uconv(Ty::U64, s)
    };
    let oob = kb.ule(ne0, i0);
    kb.if_(oob, |kb| kb.ret());

    // i3 = fastdiv(bz, ne2); i2 = bz - i3 * ne2.z
    let bz64 = kb.uconv(Ty::U64, bz);
    let mp64 = kb.uconv(Ty::U64, mp);
    let prod = kb.mul64(bz64, mp64);
    let hi = kb.shr64c(prod, 32);
    let hi = kb.uconv(Ty::U32, hi);
    let sum = kb.add(hi, bz);
    let i3 = kb.shr(sum, l);
    let i2 = {
        let t = kb.mul(i3, ne2z);
        kb.sub(bz, t)
    };
    let i1 = kb.uconv(Ty::U64, by);
    let i3_64 = kb.uconv(Ty::U64, i3);
    let i2_64 = kb.uconv(Ty::U64, i2);
    let ne2z64 = kb.uconv(Ty::U64, ne2z);
    let ne1_64 = kb.uconv(Ty::U64, ne1);
    let i_cont = {
        let a = kb.mul64(i3_64, ne2z64);
        let a = kb.add64(a, i2_64);
        let a = kb.mul64(a, ne1_64);
        let a = kb.add64(a, i1);
        let a = kb.mul64(a, ne0);
        kb.add64(a, i0)
    };
    let ib = kb.shr64c(i_cont, 5);
    let iqs = {
        let c = kb.c64(31);
        let t = kb.t(Ty::U64);
        let m = kb.b.bitwise_and(t, None, i_cont, c).unwrap();
        kb.uconv(Ty::U32, m)
    };

    // xi = i0 < ne00 ? x[i3*s03 + i2*s02 + i1*s01 + i0] : 0
    let zf = kb.cf(0.0);
    let xi = kb.var(Ty::F32, zf);
    let inb = kb.ult64(i0, ne00);
    kb.if_(inb, |kb| {
        let a = kb.mul64(i3_64, s03);
        let b = kb.mul64(i2_64, s02);
        let c = kb.mul64(i1, s01);
        let idx = kb.add64(a, b);
        let idx = kb.add64(idx, c);
        let idx = kb.add64(idx, i0);
        let four = kb.c64(4);
        let off = kb.mul64(idx, four);
        let a = kb.add64(x, off);
        let v = kb.ld(Ty::F32, a, 4);
        kb.set(xi, v);
    });
    let xi = kb.get(xi);
    let mut amax = kb.fabs(xi);
    let mut sum = xi;
    for mask in [16u32, 8, 4, 2, 1] {
        let o = kb.shfl_xor_f(amax, mask);
        amax = kb.fmax(amax, o);
        let o = kb.shfl_xor_f(sum, mask);
        sum = kb.fadd(sum, o);
    }
    let c127 = kb.cf(127.0);
    let d = kb.fdiv(amax, c127);
    let zero = kb.cf(0.0);
    let is_zero = kb.foeq(amax, zero);
    // Avoid evaluating a division by zero lane differently: the select picks 0 for amax == 0.
    let ratio = kb.fdiv(xi, d);
    let r = kb.round(ratio);
    let qi = kb.f2s(r);
    let z = kb.c32(0);
    let qi = kb.select(Ty::U32, is_zero, z, qi);
    let q8 = kb.uconv(Ty::U8, qi);

    let blk = {
        let c = kb.c64(36);
        let o = kb.mul64(ib, c);
        kb.add64(y, o)
    };
    let qaddr = {
        let o = kb.uconv(Ty::U64, iqs);
        let a = kb.add64(blk, o);
        kb.addr_off(a, 4)
    };
    kb.st(Ty::U8, qaddr, q8, 1);
    let first = kb.eq(iqs, z);
    kb.if_(first, |kb| {
        let dh = kb.f2h_bits(d);
        let sh = kb.f2h_bits(sum);
        kb.st(Ty::U16, blk, dh, 2);
        let a = kb.addr_off(blk, 2);
        kb.st(Ty::U16, a, sh, 2);
    });
    kb.finish()
}
