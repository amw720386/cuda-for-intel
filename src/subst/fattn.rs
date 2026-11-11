//! Native `flash_attn_ext_f16<D, D, ncols1, ncols2, nwarps, ntiles, false, false>` (ggml-cuda
//! fattn-mma-f16.cuh): the tensor-core FlashAttention kernel ggml picks for prefill (and GQA decode)
//! on Turing-class devices. Translated, its `mma.sync` tiles become scalar subgroup code and the
//! kernel runs about 4x slower than unfused attention; this replacement is a plain SIMT tile kernel.
//!
//! Work split: one workgroup per (16 queries, head, sequence), 4 subgroups of 32 lanes, 4 queries
//! per subgroup. Keys stream through shared memory in chunks of 32, K and V kept as packed f16
//! pairs; in a chunk lane `j` scores key `j` against the subgroup's 4 queries, the online softmax
//! runs per query with subgroup max/sum butterflies, and the output accumulates with lane `l`
//! owning dimensions `64 i + 2 l` and `64 i + 2 l + 1`. Chunks whose mask is -inf for the whole
//! tile are skipped. Arithmetic is f32 (stock rounds P to f16 before the PV product), so results
//! agree with the stock kernel to about 1e-3 relative, not bit for bit.
//!
//! Supported: K/V f16 (ggml converts quantized caches before this kernel), mask f16 or none, GQA,
//! several sequences. Not supported (the launch keeps the translated kernel): sinks, ALiBi
//! (`max_bias != 0`), logit softcap, D other than 64 / 128, unaligned K/V rows. The paired
//! `flash_attn_stream_k_fixup` launch is skipped after a native launch (the native kernel writes
//! final results).
use super::spv::{Arg, Kb, Ty, V};

/// Subgroups per workgroup (`CUDA_FOR_INTEL_FATTN_NW` = 4 or 8, read once; default 4).
pub(crate) fn nw() -> u32 {
    static NW: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *NW.get_or_init(|| match std::env::var("CUDA_FOR_INTEL_FATTN_NW").ok().as_deref() {
        Some("8") => 8,
        _ => 4,
    })
}
/// Queries per subgroup: 4, so their Q values and probabilities load as one 16-byte vector.
pub(crate) const QW: u32 = 4;
/// Queries per workgroup.
pub(crate) fn qt() -> u32 {
    nw() * QW
}
/// Keys per lane in a chunk.
pub(crate) fn kpl(d: u32) -> u32 {
    // 2 keys per lane measured slower on the B390 (more shared memory per workgroup, lower occupancy)
    let _ = d;
    1
}

/// Keys per chunk.
pub(crate) fn kc(d: u32) -> u32 {
    32 * kpl(d)
}

/// The 37 CUDA parameters in order (pointers, 4-byte scalars, three int64 strides).
pub(crate) const ARGS: [Arg; 37] = {
    let mut a = [Arg::U32; 37];
    let mut i = 0;
    while i < 8 {
        a[i] = Arg::Ptr;
        i += 1;
    }
    a[8] = Arg::F32;
    a[9] = Arg::F32;
    a[10] = Arg::F32;
    a[11] = Arg::F32;
    a[13] = Arg::F32;
    a[27] = Arg::U64;
    a[30] = Arg::U64;
    a[36] = Arg::U64;
    a
};

/// Byte size of every launch word (for the layout check).
pub(crate) fn arg_sizes() -> [usize; 37] {
    ARGS.map(|a| match a {
        Arg::Ptr | Arg::U64 => 8,
        Arg::U32 | Arg::F32 => 4,
    })
}

pub(crate) fn entry_name(d: u32) -> String {
    format!("cfi_fattn_d{d}_w{}", nw())
}

pub(crate) fn supported_d(d: u32) -> bool {
    matches!(d, 64 | 128)
}

/// Shared-memory words: Q tile (transposed), K chunk (padded rows), V chunk, P per subgroup.
fn shared_words(d: u32) -> u32 {
    #[allow(non_snake_case)]
    let (NW, QT) = (nw(), qt());
    let kc = kc(d);
    d * QT + kc * (d / 2 + 1) + kc * (d / 2) + NW * kc * QW + 2
}

pub(crate) fn build(d: u32) -> Vec<u32> {
    assert!(supported_d(d));
    #[allow(non_snake_case)]
    let (NW, QT, NTH) = (nw(), qt(), 32 * nw());
    let name = entry_name(d);
    let mut kb = Kb::new(&name, &ARGS, shared_words(d), true, false);
    let p = kb.params.clone();
    let qp = kb.ptr_to_u64(p[0]);
    let kp = kb.ptr_to_u64(p[1]);
    let vp = kb.ptr_to_u64(p[2]);
    let mp = kb.ptr_to_u64(p[3]);
    let dp = kb.ptr_to_u64(p[6]);
    let scale = p[8];
    let (ne01, ne02) = (p[15], p[16]);
    let (nb01, nb02, nb03) = (p[18], p[19], p[20]);
    let (ne11, ne12) = (p[22], p[23]);
    let (nb11, nb12, nb13) = (p[25], p[26], p[27]);
    let (nb21, nb22, nb23) = (p[28], p[29], p[30]);
    let ne33 = p[33];
    let (nb31, nb33) = (p[34], p[36]);

    let kc = kc(d);
    let kpl = kpl(d);
    let qs_off = 0u32;
    let ks_off = d * QT;
    // K and V stay f16 in shared memory, two per word: K rows padded to d/2 + 1 words (odd stride,
    // conflict-free when lane j reads row j), V rows d/2 words (lane l reads word l: dims 2l, 2l+1)
    let vs_off = ks_off + kc * (d / 2 + 1);
    let ps_off = vs_off + kc * (d / 2);
    // two "chunk has a visible key" flags, alternating by chunk parity
    let flag_off = ps_off + NW * kc * QW;
    let ndw = d / 32;

    let lane = kb.local_id(0);
    let w = kb.local_id(1);
    let tid = {
        let a = kb.shlc(w, 5);
        kb.add(a, lane)
    };
    let tile = kb.group_id(0);
    let head = kb.group_id(1);
    let seq = kb.group_id(2);
    let q0 = kb.mulc(tile, QT);
    let gqa = kb.udiv(ne02, ne12);
    let kvh = kb.udiv(head, gqa);
    let u64of = |kb: &mut Kb, v: V| kb.uconv(Ty::U64, v);

    // per-sequence / head bases
    let q_base = {
        let s = u64of(&mut kb, seq);
        let n3 = u64of(&mut kb, nb03);
        let a = kb.mul64(s, n3);
        let h = u64of(&mut kb, head);
        let n2 = u64of(&mut kb, nb02);
        let b = kb.mul64(h, n2);
        let ab = kb.add64(a, b);
        kb.add64(qp, ab)
    };
    let kv_base = |kb: &mut Kb, base: V, nb2: V, nb3: V| {
        let s = u64of(kb, seq);
        let a = kb.mul64(s, nb3);
        let h = u64of(kb, kvh);
        let n2 = u64of(kb, nb2);
        let b = kb.mul64(h, n2);
        let ab = kb.add64(a, b);
        kb.add64(base, ab)
    };
    let k_base = kv_base(&mut kb, kp, nb12, nb13);
    let v_base = kv_base(&mut kb, vp, nb22, nb23);
    let m_base = {
        // ne33 is 0 when there is no mask
        let z = kb.c32(0);
        let one = kb.c32(1);
        let none = kb.eq(ne33, z);
        let ne33 = kb.select(Ty::U32, none, one, ne33);
        let sm = kb.umod(seq, ne33);
        let s = u64of(&mut kb, sm);
        let a = kb.mul64(s, nb33);
        kb.add64(mp, a)
    };
    let zero64 = kb.c64(0);
    let has_mask = kb.ne(mp, zero64);

    // Q tile -> shared, transposed [d][q], pre-scaled; rows past ne01 are zero
    for i in 0..(d * QT / NTH) {
        let e = kb.addc(tid, NTH * i);
        let qi = kb.shrc(e, d.trailing_zeros());
        let dd = kb.andc(e, d - 1);
        let q = kb.add(q0, qi);
        let valid = kb.ult(q, ne01);
        let z = kb.cf(0.0);
        let val = kb.var(Ty::F32, z);
        kb.if_(valid, |kb| {
            let qo = {
                let q64 = kb.uconv(Ty::U64, q);
                let n1 = kb.uconv(Ty::U64, nb01);
                kb.mul64(q64, n1)
            };
            let a = kb.add64(q_base, qo);
            let a = kb.addr(a, dd, 4);
            let x = kb.ld(Ty::F32, a, 4);
            let x = kb.fmul(x, scale);
            kb.set(val, x);
        });
        let idx = {
            let a = kb.mulc(dd, QT);
            let a = kb.add(a, qi);
            kb.addc(a, qs_off)
        };
        let v = kb.get(val);
        kb.sh_st_f(idx, v);
    }

    let ninf = kb.cf(f32::NEG_INFINITY);
    let zf = kb.cf(0.0);
    let m_run: Vec<_> = (0..QW).map(|_| kb.var(Ty::F32, ninf)).collect();
    let l_run: Vec<_> = (0..QW).map(|_| kb.var(Ty::F32, zf)).collect();
    let o_acc: Vec<Vec<_>> = (0..QW).map(|_| (0..ndw).map(|_| kb.var(Ty::F32, zf)).collect()).collect();
    let wq = kb.mulc(w, QW);
    let ne11m1 = {
        let one = kb.c32(1);
        kb.sub(ne11, one)
    };

    {
        let z = kb.c32(0);
        let first = kb.eq(tid, z);
        kb.if_(first, |kb| {
            for i in 0..2 {
                let idx = kb.c32(flag_off + i);
                let z = kb.c32(0);
                kb.sh_st(idx, z);
            }
        });
    }
    let kv0 = {
        let z = kb.c32(0);
        kb.var(Ty::U32, z)
    };
    kb.while_(
        |kb| {
            let v = kb.get(kv0);
            kb.ult(v, ne11)
        },
        |kb| {
            let k0 = kb.get(kv0);
            kb.barrier();
            // K / V chunk -> shared as packed f16 pairs (rows past ne11 read the last row; their scores are -inf)
            for i in 0..(kc * d / (2 * NTH)) {
                let e = kb.addc(tid, NTH * i);
                let j = kb.shrc(e, (d / 2).trailing_zeros());
                let pr = kb.andc(e, d / 2 - 1);
                let kv = kb.add(k0, j);
                let lt = kb.ult(kv, ne11);
                let kvc = kb.select(Ty::U32, lt, kv, ne11m1);
                let kv64 = kb.uconv(Ty::U64, kvc);
                for (base, nb, off, stride) in [(k_base, nb11, ks_off, d / 2 + 1), (v_base, nb21, vs_off, d / 2)] {
                    let n = kb.uconv(Ty::U64, nb);
                    let ro = kb.mul64(kv64, n);
                    let a = kb.add64(base, ro);
                    let a = kb.addr(a, pr, 4);
                    let word = kb.ld(Ty::U32, a, 4);
                    let idx = {
                        let a = kb.mulc(j, stride);
                        let s = kb.add(a, pr);
                        kb.addc(s, off)
                    };
                    kb.sh_st(idx, word);
                }
            }
            // Causal skip: a chunk whose keys are all masked (-inf) for every query of the tile adds
            // nothing; each thread checks 4 of the 16 x 32 mask entries and raises the chunk's flag.
            let parity = {
                let c = kb.shrc(k0, kc.trailing_zeros());
                kb.andc(c, 1)
            };
            let flag = kb.addc(parity, flag_off);
            {
                let one = kb.c32(1);
                let zero = kb.c32(0);
                let init = kb.select(Ty::U32, has_mask, zero, one);
                let any = kb.var(Ty::U32, init);
                kb.if_(has_mask, |kb| {
                    for i in 0..(QT * kc / NTH) {
                        let e = kb.addc(tid, NTH * i);
                        let qi = kb.shrc(e, kc.trailing_zeros());
                        let j = kb.andc(e, kc - 1);
                        let q = kb.add(q0, qi);
                        let kv = kb.add(k0, j);
                        let qv = kb.ult(q, ne01);
                        let kvv = kb.ult(kv, ne11);
                        let both = kb.land(qv, kvv);
                        kb.if_(both, |kb| {
                            let row = {
                                let q64 = kb.uconv(Ty::U64, q);
                                let n = kb.uconv(Ty::U64, nb31);
                                kb.mul64(q64, n)
                            };
                            let a = kb.add64(m_base, row);
                            let a = kb.addr(a, kv, 2);
                            let m = kb.ld(Ty::U16, a, 2);
                            let m = kb.uconv(Ty::U32, m);
                            let ninf_bits = kb.c32(0xfc00);
                            let vis = kb.ne(m, ninf_bits);
                            kb.if_(vis, |kb| {
                                let one = kb.c32(1);
                                kb.set(any, one);
                            });
                        });
                    }
                });
                let a = kb.get(any);
                let z = kb.c32(0);
                let a = kb.ne(a, z);
                kb.if_(a, |kb| {
                    let one = kb.c32(1);
                    kb.sh_st(flag, one);
                });
            }
            kb.barrier();
            let live = {
                let f = kb.sh_ld(flag);
                let z = kb.c32(0);
                kb.ne(f, z)
            };
            {
                // clear the other parity's flag for the next chunk (its readers passed this chunk's
                // first barrier; its writers start after the next one)
                let z = kb.c32(0);
                let first = kb.eq(tid, z);
                kb.if_(first, |kb| {
                    let other = kb.c32(1);
                    let op = kb.sub(other, parity);
                    let oi = kb.addc(op, flag_off);
                    let z = kb.c32(0);
                    kb.sh_st(oi, z);
                });
            }
            kb.if_(live, |kb| {
            // scores of keys `lane + 32 k` against the subgroup's queries
            let mut s: Vec<Vec<V>> = (0..kpl).map(|_| (0..QW).map(|_| kb.cf(0.0)).collect()).collect();
            let krows: Vec<V> = (0..kpl)
                .map(|k| {
                    let a = kb.addc(lane, 32 * k);
                    let a = kb.mulc(a, d / 2 + 1);
                    kb.addc(a, ks_off)
                })
                .collect();
            let qcol = kb.addc(wq, qs_off);
            let unpack = |kb: &mut Kb, w: V| -> [V; 2] {
                let lo = kb.uconv(Ty::U16, w);
                let hi = kb.shrc(w, 16);
                let hi = kb.uconv(Ty::U16, hi);
                [kb.h2f(lo), kb.h2f(hi)]
            };
            for dp in 0..d / 2 {
                // the subgroup's 4 queries are adjacent: one 16-byte broadcast load per dimension
                let qa = kb.addc(qcol, 2 * dp * QT);
                let qb = kb.addc(qcol, (2 * dp + 1) * QT);
                let qv = [kb.sh_ld_f4(qa), kb.sh_ld_f4(qb)];
                for (k, sk) in s.iter_mut().enumerate() {
                    let ki = kb.addc(krows[k], dp);
                    let kw = kb.sh_ld(ki);
                    let kd = unpack(kb, kw);
                    for h in 0..2 {
                        for (q, sq) in sk.iter_mut().enumerate() {
                            *sq = kb.fma(qv[h][q], kd[h], *sq);
                        }
                    }
                }
            }
            let kvs: Vec<(V, V)> = (0..kpl)
                .map(|k| {
                    let l = kb.addc(lane, 32 * k);
                    let kv = kb.add(k0, l);
                    let valid = kb.ult(kv, ne11);
                    (kv, valid)
                })
                .collect();
            for q in 0..QW as usize {
                let qq = kb.addc(wq, q as u32);
                let qq = kb.add(q0, qq);
                let qvalid = kb.ult(qq, ne01);
                let mut sc = Vec::new();
                for (k, &(kv, kvalid)) in kvs.iter().enumerate() {
                    let mval = kb.var(Ty::F32, zf);
                    let a = kb.land(has_mask, kvalid);
                    let cond = kb.land(a, qvalid);
                    kb.if_(cond, |kb| {
                        let row = {
                            let q64 = kb.uconv(Ty::U64, qq);
                            let n = kb.uconv(Ty::U64, nb31);
                            kb.mul64(q64, n)
                        };
                        let a = kb.add64(m_base, row);
                        let a = kb.addr(a, kv, 2);
                        let m = kb.ld_f16(a);
                        kb.set(mval, m);
                    });
                    let mv = kb.get(mval);
                    let sm = kb.fadd(s[k][q], mv);
                    sc.push(kb.select(Ty::F32, kvalid, sm, ninf));
                }
                // online softmax
                let mut mx = sc[0];
                for &x in &sc[1..] {
                    mx = kb.fmax(mx, x);
                }
                for mask in [16u32, 8, 4, 2, 1] {
                    let o = kb.shfl_xor_f(mx, mask);
                    mx = kb.fmax(mx, o);
                }
                let m_old = kb.get(m_run[q]);
                let m_new = kb.fmax(m_old, mx);
                let is_ninf = kb.foeq(m_new, ninf);
                let m_use = kb.select(Ty::F32, is_ninf, zf, m_new);
                let pexp: Vec<V> = sc
                    .iter()
                    .map(|&x| {
                        let ds = kb.fsub(x, m_use);
                        kb.exp(ds)
                    })
                    .collect();
                let da = kb.fsub(m_old, m_use);
                let alpha = kb.exp(da);
                let mut sum = pexp[0];
                for &x in &pexp[1..] {
                    sum = kb.fadd(sum, x);
                }
                for mask in [16u32, 8, 4, 2, 1] {
                    let o = kb.shfl_xor_f(sum, mask);
                    sum = kb.fadd(sum, o);
                }
                let l_old = kb.get(l_run[q]);
                let l_new = kb.fma(l_old, alpha, sum);
                kb.set(l_run[q], l_new);
                kb.set(m_run[q], m_new);
                for i in 0..ndw as usize {
                    let o = kb.get(o_acc[q][i]);
                    let o = kb.fmul(o, alpha);
                    kb.set(o_acc[q][i], o);
                }
                for (k, &p) in pexp.iter().enumerate() {
                    let pidx = {
                        let a = kb.mulc(w, kc);
                        let a = kb.add(a, lane);
                        let a = kb.addc(a, 32 * k as u32);
                        let a = kb.mulc(a, QW);
                        let a = kb.addc(a, q as u32);
                        kb.addc(a, ps_off)
                    };
                    kb.sh_st_f(pidx, p);
                }
            }
            kb.barrier();
            // O += P V
            let pw = {
                let a = kb.mulc(w, kc * QW);
                kb.addc(a, ps_off)
            };
            let vl = kb.addc(lane, vs_off);
            for jj in 0..kc {
                let pi = kb.addc(pw, jj * QW);
                let pq = kb.sh_ld_f4(pi);
                for i in 0..d / 64 {
                    let vi = kb.addc(vl, jj * (d / 2) + 32 * i);
                    let vw = kb.sh_ld(vi);
                    let v = unpack(kb, vw);
                    for (h, vh) in v.iter().enumerate() {
                        let slot = 2 * i as usize + h;
                        for q in 0..QW as usize {
                            let o = kb.get(o_acc[q][slot]);
                            let o = kb.fma(pq[q], *vh, o);
                            kb.set(o_acc[q][slot], o);
                        }
                    }
                }
            }
            });
            let n = kb.addc(k0, kc);
            kb.set(kv0, n);
        },
    );

    // dst[((seq*ne01 + q)*ne02 + head)*D + dim]
    for q in 0..QW as usize {
        let qq = kb.addc(wq, q as u32);
        let qq = kb.add(q0, qq);
        let qvalid = kb.ult(qq, ne01);
        let l = kb.get(l_run[q]);
        let pos = kb.fogt(l, zf);
        let one = kb.cf(1.0);
        let inv = kb.fdiv(one, l);
        let inv = kb.select(Ty::F32, pos, inv, zf);
        let o: Vec<V> = (0..ndw as usize).map(|i| kb.get(o_acc[q][i])).collect();
        kb.if_(qvalid, |kb| {
            let row = {
                let a = kb.mul(seq, ne01);
                let a = kb.add(a, qq);
                let a = kb.mul(a, ne02);
                kb.add(a, head)
            };
            let base = {
                let r = kb.uconv(Ty::U64, row);
                let c = kb.c64(u64::from(d) * 4);
                let off = kb.mul64(r, c);
                kb.add64(dp, off)
            };
            for (slot, oi) in o.iter().enumerate() {
                let v = kb.fmul(*oi, inv);
                // slot 2i + h holds dimension 64 i + 2 lane + h
                let l2 = kb.shlc(lane, 1);
                let di = kb.addc(l2, 64 * (slot as u32 / 2) + slot as u32 % 2);
                let a = kb.addr(base, di, 4);
                kb.st(Ty::F32, a, v, 4);
            }
        });
    }
    kb.finish()
}
