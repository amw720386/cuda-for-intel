//! Subgroup/warp behaviour, atomics, aligned loads, shared memory and barriers on hardware.
use super::hw_common::*;

const GUARD: &str = "\tmov.u32 %r0, %tid.x;\n\tmov.u32 %r1, %ctaid.x;\n\tmov.u32 %r2, %ntid.x;\n\tmad.lo.s32 %r3, %r1, %r2, %r0;\n";

// ---------------------------------------------------------------- subgroup size / vote

fn lane_ptx() -> String {
    let mut s = prologue("lane_k", 3, 0, "");
    s += GUARD;
    s += "\tmov.u32 %r4, %laneid;\n\tsetp.eq.u32 %p1, %r0, %r0;\n\tvote.sync.ballot.b32 %r5, %p1, 0xffffffff;\n\tactivemask.b32 %r6;\n";
    s += "\tmul.wide.u32 %rd20, %r3, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tst.global.u32 [%rd21], %r4;\n\tadd.s64 %rd21, %rd2, %rd20;\n\tst.global.u32 [%rd21], %r5;\n\tadd.s64 %rd21, %rd3, %rd20;\n\tst.global.u32 [%rd21], %r6;\n\tret;\n}\n";
    s
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_subgroup_is_32_lanes_laneid_ballot_activemask() {
    for block in [32u32, 64, 96, 128, 256, 48, 33, 17] {
        let n = block as usize;
        let mut bufs = vec![vec![0u8; n * 4], vec![0u8; n * 4], vec![0u8; n * 4]];
        launch(&lane_ptx(), "lane_k", 1, block, &mut bufs, &[]).unwrap();
        let (lane, ballot, active) = (u32s_of(&bufs[0]), u32s_of(&bufs[1]), u32s_of(&bufs[2]));
        for t in 0..n {
            let in_warp = (n - (t & !31)).min(32);
            let full = if in_warp == 32 { u32::MAX } else { (1u32 << in_warp) - 1 };
            assert_eq!(lane[t], (t % 32) as u32, "laneid block {block} tid {t}");
            assert_eq!(ballot[t], full, "ballot(true) block {block} tid {t}: subgroup is not 32 wide");
            assert_eq!(active[t], full, "activemask block {block} tid {t}");
        }
    }
}

fn vote_ptx() -> String {
    let mut s = prologue("vote_k", 3, 0, "");
    s += GUARD;
    s += "\tmul.wide.u32 %rd20, %r3, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tld.global.u32 %r4, [%rd21];\n\tsetp.ne.u32 %p1, %r4, 0;\n";
    s += "\tvote.sync.ballot.b32 %r5, %p1, 0xffffffff;\n\tvote.sync.all.pred %p2, %p1, 0xffffffff;\n\tvote.sync.any.pred %p3, %p1, 0xffffffff;\n";
    s += "\tselp.u32 %r6, 1, 0, %p2;\n\tselp.u32 %r7, 2, 0, %p3;\n\tor.b32 %r6, %r6, %r7;\n";
    s += "\tadd.s64 %rd22, %rd2, %rd20;\n\tst.global.u32 [%rd22], %r5;\n\tmul.wide.u32 %rd23, %r3, 8;\n";
    // out2[2*tid] = all|any bits ; out2[2*tid+1] = divergent ballot
    s += "\tadd.s64 %rd24, %rd3, %rd23;\n\tst.global.u32 [%rd24], %r6;\n";
    s += "\tand.b32 %r8, %r0, 1;\n\tsetp.ne.u32 %p4, %r8, 0;\n\t@%p4 bra $odd;\n";
    s += "\tactivemask.b32 %r9;\n\tvote.sync.ballot.b32 %r10, %p1, %r9;\n\txor.b32 %r9, %r9, %r10;\n\tst.global.u32 [%rd24+4], %r9;\n";
    s += "$odd:\n\tret;\n}\n";
    s
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_vote_ballot_all_any_and_divergent_activemask() {
    let mut rng = Rng(0x7007_7007_7007_7007);
    for block in [32u32, 128, 48] {
        for iter in 0..40 {
            let n = block as usize;
            let preds: Vec<u32> = (0..n)
                .map(|t| match (iter / 4 + t / 32) % 5 {
                    0 => 0,
                    1 => 1,
                    _ => (rng.u32() & 1) * (1 + (rng.u32() & 0xff)),
                })
                .collect();
            let mut bufs = vec![bytes_of_u32(&preds), vec![0u8; n * 4], vec![0xeeu8; n * 8]];
            launch(&vote_ptx(), "vote_k", 1, block, &mut bufs, &[]).unwrap();
            let (ballot, flags) = (u32s_of(&bufs[1]), u32s_of(&bufs[2]));
            for t in 0..n {
                let w = t & !31;
                let in_warp = (n - w).min(32);
                let bits = |f: &dyn Fn(usize) -> bool| (0..in_warp).filter(|&l| f(l)).fold(0u32, |m, l| m | 1 << l);
                let want_ballot = bits(&|l| preds[w + l] != 0);
                let all = want_ballot.count_ones() as usize == in_warp;
                let any = want_ballot != 0;
                assert_eq!(ballot[t], want_ballot, "ballot block {block} iter {iter} tid {t}");
                assert_eq!(flags[2 * t], all as u32 | (any as u32) << 1, "all/any block {block} iter {iter} tid {t}");
                if t % 2 == 0 {
                    // divergent branch: active = even lanes; xor(active, ballot over them) = active & !pred
                    let even = bits(&|l| l % 2 == 0);
                    let want = even ^ bits(&|l| l % 2 == 0 && preds[w + l] != 0);
                    assert_eq!(flags[2 * t + 1], want, "divergent activemask/ballot block {block} iter {iter} tid {t}");
                } else {
                    assert_eq!(flags[2 * t + 1], 0xeeee_eeee, "odd lane must not write");
                }
            }
        }
    }
}

// ---------------------------------------------------------------- shuffles

fn shfl_ptx(mode: &str, clamp: u32) -> String {
    let mut s = prologue("shfl_k", 4, 0, "");
    s += GUARD;
    s += "\tmul.wide.u32 %rd20, %r3, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tld.global.u32 %r4, [%rd21];\n\tadd.s64 %rd21, %rd2, %rd20;\n\tld.global.u32 %r5, [%rd21];\n";
    s += &format!("\tshfl.sync.{mode}.b32 %r6|%p1, %r4, %r5, {clamp:#x}, 0xffffffff;\n");
    s += "\tselp.u32 %r7, 1, 0, %p1;\n\tadd.s64 %rd22, %rd3, %rd20;\n\tst.global.u32 [%rd22], %r6;\n\tadd.s64 %rd22, %rd4, %rd20;\n\tst.global.u32 [%rd22], %r7;\n\tret;\n}\n";
    s
}

fn shfl_ref(mode: &str, lane: u32, b: u32, c: u32) -> (u32, bool) {
    let segmask = (c >> 8) & 0x1f;
    let min_lane = lane & segmask;
    let max_lane = min_lane | ((c & 0x1f) & !segmask);
    let (j, ok) = match mode {
        "up" => {
            let j = lane as i64 - (b & 0xffff) as i64; // PTX uses b[4:0] for bval except idx/bfly fix; handled below
            (j, j >= max_lane as i64)
        }
        "down" => {
            let j = lane as i64 + (b & 0x1f) as i64;
            (j, j <= max_lane as i64)
        }
        _ => {
            let j = (min_lane | ((b & 0x1f) & !segmask)) as i64;
            (j, j <= max_lane as i64)
        }
    };
    if ok { (j as u32, true) } else { (lane, false) }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_shfl_idx_up_down_with_clamp_and_segments() {
    let mut rng = Rng(0xc0ffee00_c0ffee11);
    // clamp operand: width 32/16/8/4 (segmask in bits 8..12), up uses cval 0.
    let widths = [32u32, 16, 8, 4, 2];
    for mode in ["idx", "up", "down"] {
        for &w in &widths {
            let segmask = 32 - w;
            let clamp = if mode == "up" { segmask << 8 } else { (segmask << 8) | 0x1f };
            let n = 128usize;
            for _ in 0..8 {
                let vals: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
                // deltas: in-range 0..31 (bval uses the low 5 bits per PTX)
                let deltas: Vec<u32> = (0..n).map(|_| rng.u32() % 32).collect();
                let mut bufs = vec![bytes_of_u32(&vals), bytes_of_u32(&deltas), vec![0u8; n * 4], vec![0u8; n * 4]];
                launch(&shfl_ptx(mode, clamp), "shfl_k", 1, n as u32, &mut bufs, &[]).unwrap();
                let (out, preds) = (u32s_of(&bufs[2]), u32s_of(&bufs[3]));
                for t in 0..n {
                    let lane = (t % 32) as u32;
                    let (j, ok) = shfl_ref(mode, lane, deltas[t], clamp);
                    let want = vals[(t & !31) + j as usize];
                    assert_eq!((out[t], preds[t]), (want, ok as u32), "shfl.{mode} width {w} tid {t} lane {lane} delta {}", deltas[t]);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- atomics

fn atomic_ptx(kind: &str) -> String {
    // kind: "atom_g", "red_g", "atom_s"
    let extra = if kind == "atom_s" { "\t.shared .align 4 .b8 sacc[64];\n" } else { "" };
    let mut s = prologue("atom_k", 3, 1, extra);
    s += GUARD;
    if kind != "atom_s" {
        s += "\tsetp.ge.u32 %p1, %r3, %r100;\n\t@%p1 bra $done;\n";
    }
    s += "\tand.b32 %r4, %r3, 15;\n\tmul.wide.u32 %rd20, %r4, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n";
    s += "\tmul.wide.u32 %rd22, %r3, 4;\n\tadd.s64 %rd23, %rd2, %rd22;\n\tld.global.f32 %f1, [%rd23];\n\tadd.s64 %rd24, %rd3, %rd22;\n";
    match kind {
        "atom_g" => s += "\tatom.global.add.f32 %f2, [%rd21], %f1;\n\tst.global.f32 [%rd24], %f2;\n",
        "red_g" => s += "\tred.global.add.f32 [%rd21], %f1;\n",
        _ => {
            // shared accumulation; thread 0..15 of the block zero it first. One block only.
            s += "\tmov.u32 %r9, sacc;\n\tmov.u32 %r10, 0;\n\tsetp.ge.u32 %p2, %r0, 16;\n\t@%p2 bra $zeroed;\n\tshl.b32 %r11, %r0, 2;\n\tadd.s32 %r11, %r11, %r9;\n\tst.shared.u32 [%r11], %r10;\n$zeroed:\n\tbar.sync 0;\n";
            s += "\tshl.b32 %r11, %r4, 2;\n\tadd.s32 %r11, %r11, %r9;\n\tatom.shared.add.f32 %f2, [%r11], %f1;\n\tst.global.f32 [%rd24], %f2;\n\tbar.sync 0;\n";
            s += "\tsetp.ge.u32 %p2, %r0, 16;\n\t@%p2 bra $done;\n\tshl.b32 %r11, %r0, 2;\n\tadd.s32 %r11, %r11, %r9;\n\tld.shared.f32 %f3, [%r11];\n\tst.global.f32 [%rd21], %f3;\n";
        }
    }
    s += "$done:\n\tret;\n}\n";
    s
}

fn run_atomic(kind: &str, n: usize, block: u32, grid: u32) {
    let mut rng = Rng(0xa70a_a70a_0000_0001 ^ n as u64);
    let vals: Vec<f32> = (0..n).map(|_| (1 + rng.u32() % 8) as f32 * 0.25).collect();
    let mut totals = [0f32; 16];
    for (i, v) in vals.iter().enumerate() {
        totals[i % 16] += v;
    }
    let mut bufs = vec![vec![0u8; 16 * 4], bytes_of_u32(&vals.iter().map(|v| v.to_bits()).collect::<Vec<_>>()), vec![0u8; n * 4]];
    launch(&atomic_ptx(kind), "atom_k", grid, block, &mut bufs, &[n as u32]).unwrap();
    let acc: Vec<f32> = u32s_of(&bufs[0]).iter().map(|b| f32::from_bits(*b)).collect();
    assert_eq!(acc.to_vec(), totals.to_vec(), "{kind}: final sums (lost updates?)");
    if kind != "red_g" {
        let olds: Vec<f32> = u32s_of(&bufs[2]).iter().map(|b| f32::from_bits(*b)).collect();
        for slot in 0..16 {
            let mut seq: Vec<(f32, f32)> = (0..n).filter(|i| i % 16 == slot).map(|i| (olds[i], vals[i])).collect();
            seq.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let mut expect = 0.0f32;
            for (old, v) in seq {
                assert_eq!(old, expect, "{kind}: slot {slot} returned old values are not a serial chain");
                expect += v;
            }
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_atomic_add_f32_global_return_value() {
    run_atomic("atom_g", 4096, 256, 16);
    run_atomic("atom_g", 100000, 256, 391);
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_red_add_f32_global() {
    run_atomic("red_g", 4096, 256, 16);
    run_atomic("red_g", 100000, 256, 391);
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_atom_add_f32_shared() {
    run_atomic("atom_s", 256, 256, 1);
}

// ---------------------------------------------------------------- aligned loads / stores

fn copy_ptx(ld: &str, st: &str, size: u32, regs: usize) -> String {
    let pfx = if ld.contains("f32") { "f" } else { "r" };
    let mut s = prologue("copy_k", 2, 1, "");
    s += GUARD;
    // scalar %r100 = byte offset applied to both buffers
    s += &format!("\tmul.wide.u32 %rd20, %r3, {size};\n\tcvt.u64.u32 %rd21, %r100;\n\tadd.s64 %rd20, %rd20, %rd21;\n\tadd.s64 %rd22, %rd1, %rd20;\n\tadd.s64 %rd23, %rd2, %rd20;\n");
    let list = |base: usize| (0..regs).map(|i| format!("%{pfx}{}", base + i)).collect::<Vec<_>>().join(",");
    if regs == 1 && size == 8 {
        s += &format!("\t{ld} %rd24, [%rd22];\n\t{st} [%rd23], %rd24;\n");
    } else if regs == 1 {
        s += &format!("\t{ld} %r10, [%rd22];\n\t{st} [%rd23], %r10;\n");
    } else {
        s += &format!("\t{ld} {{{}}}, [%rd22];\n\t{st} [%rd23], {{{}}};\n", list(10), list(10));
    }
    s += "\tret;\n}\n";
    s
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_aligned_global_loads_and_stores_all_widths() {
    let mut rng = Rng(0x10ad_10ad_10ad_10ad);
    let words = 4096usize;
    let cases: [(&str, &str, u32, usize); 8] = [
        ("ld.global.u32", "st.global.u32", 4, 1),
        ("ld.global.nc.u32", "st.global.u32", 4, 1),
        ("ld.global.u64", "st.global.u64", 8, 1),
        ("ld.global.v2.u32", "st.global.v2.u32", 8, 2),
        ("ld.global.v4.u32", "st.global.v4.u32", 16, 4),
        ("ld.global.nc.v4.u32", "st.global.v4.u32", 16, 4),
        ("ld.global.v4.f32", "st.global.v4.f32", 16, 4),
        ("ld.global.v2.f32", "st.global.v2.f32", 8, 2),
    ];
    for (ld, st, size, regs) in cases {
        let ptx = copy_ptx(ld, st, size, regs);
        // aligned offsets: every multiple of the access size up to 64 bytes
        let mut off = 0u32;
        while off <= 64 {
            let data: Vec<u32> = (0..words).map(|_| rng.u32()).collect();
            let n = 512u32;
            let mut bufs = vec![bytes_of_u32(&data), vec![0xa5u8; words * 4]];
            launch(&ptx, "copy_k", 2, 256, &mut bufs, &[off])
                .unwrap_or_else(|e| panic!("{ld}/{st} off {off}: {e}"));
            let out = u32s_of(&bufs[1]);
            let (lo, hi) = ((off / 4) as usize, (off / 4) as usize + (n * size / 4) as usize);
            for w in 0..words {
                let want = if (lo..hi).contains(&w) { data[w] } else { 0xa5a5_a5a5 };
                assert_eq!(out[w], want, "{ld}/{st} offset {off} word {w}");
            }
            off += size;
        }
    }
}

// ---------------------------------------------------------------- shared memory alignment + barrier

fn shared_align_ptx() -> String {
    let extra = "\t.shared .align 4 .b8 small[4];\n\t.shared .align 16 .b8 wide[1024];\n\t.shared .align 1 .b8 odd[3];\n\t.shared .align 8 .b8 qword[64];\n";
    let mut s = prologue("salign_k", 2, 0, extra);
    s += GUARD;
    s += "\tmov.u32 %r9, wide;\n\tmov.u32 %r8, qword;\n\tmov.u32 %r7, small;\n";
    // report the shared offsets (thread 0)
    s += "\tsetp.ne.u32 %p1, %r0, 0;\n\t@%p1 bra $skip;\n\tst.global.u32 [%rd2], %r9;\n\tst.global.u32 [%rd2+4], %r8;\n\tst.global.u32 [%rd2+8], %r7;\n$skip:\n";
    // each thread writes a v4 into wide[tid*16] and a u64 into qword[(tid%8)*8], then reads back its neighbour's v4
    s += "\tshl.b32 %r10, %r0, 4;\n\tadd.s32 %r10, %r10, %r9;\n\tmul.lo.s32 %r11, %r0, 4;\n\tadd.s32 %r12, %r11, 1;\n\tadd.s32 %r13, %r11, 2;\n\tadd.s32 %r14, %r11, 3;\n";
    s += "\tst.shared.v4.u32 [%r10], {%r11, %r12, %r13, %r14};\n\tbar.sync 0;\n";
    s += "\tadd.s32 %r15, %r0, 1;\n\tand.b32 %r15, %r15, 63;\n\tshl.b32 %r16, %r15, 4;\n\tadd.s32 %r16, %r16, %r9;\n\tld.shared.v4.u32 {%r20, %r21, %r22, %r23}, [%r16];\n";
    s += "\tmul.wide.u32 %rd20, %r0, 16;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tst.global.v4.u32 [%rd21], {%r20, %r21, %r22, %r23};\n\tret;\n}\n";
    s
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_shared_symbols_honor_declared_alignment_for_v4() {
    let mut bufs = vec![vec![0u8; 64 * 16], vec![0u8; 16]];
    launch(&shared_align_ptx(), "salign_k", 1, 64, &mut bufs, &[]).unwrap();
    let offs = u32s_of(&bufs[1]);
    assert_eq!(offs[0] % 16, 0, "shared symbol .align 16 starts at offset {}", offs[0]);
    assert_eq!(offs[1] % 8, 0, "shared symbol .align 8 starts at offset {}", offs[1]);
    assert_eq!(offs[2] % 4, 0);
    let out = u32s_of(&bufs[0]);
    for t in 0..64usize {
        let nb = (t + 1) & 63;
        for k in 0..4 {
            assert_eq!(out[4 * t + k], (4 * nb + k) as u32, "v4 shared round trip tid {t} comp {k}");
        }
    }
}

fn barrier_ptx(rounds: u32, block: u32) -> String {
    let extra = format!("\t.shared .align 4 .b8 buf[{}];\n", block * 4);
    let mut s = prologue("bar_k", 2, 0, &extra);
    s += GUARD;
    s += "\tmul.wide.u32 %rd20, %r3, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tld.global.u32 %r4, [%rd21];\n\tmov.u32 %r9, buf;\n\tshl.b32 %r10, %r0, 2;\n\tadd.s32 %r10, %r10, %r9;\n";
    for r in 0..rounds {
        s += "\tst.shared.u32 [%r10], %r4;\n\tbar.sync 0;\n";
        s += &format!(
            "\tmad.lo.s32 %r11, %r0, 37, {r};\n\trem.u32 %r11, %r11, {block};\n\tshl.b32 %r11, %r11, 2;\n\tadd.s32 %r11, %r11, %r9;\n\tld.shared.u32 %r12, [%r11];\n\tbar.sync 0;\n\tmad.lo.s32 %r4, %r4, 31, %r12;\n"
        );
    }
    s += "\tadd.s64 %rd22, %rd2, %rd20;\n\tst.global.u32 [%rd22], %r4;\n\tret;\n}\n";
    s
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_bar_sync_orders_shared_memory_across_warps() {
    let mut rng = Rng(0xbabe_babe_cafe_cafe);
    for block in [64u32, 128, 256, 512] {
        let rounds = 8;
        let x0: Vec<u32> = (0..block).map(|_| rng.u32()).collect();
        let mut x = x0.clone();
        for r in 0..rounds {
            let prev = x.clone();
            for t in 0..block {
                let src = (t * 37 + r) % block;
                x[t as usize] = prev[t as usize].wrapping_mul(31).wrapping_add(prev[src as usize]);
            }
        }
        let mut bufs = vec![bytes_of_u32(&x0), vec![0u8; block as usize * 4]];
        launch(&barrier_ptx(rounds, block), "bar_k", 1, block, &mut bufs, &[]).unwrap();
        assert_eq!(u32s_of(&bufs[1]), x, "bar.sync block {block}");
    }
}
