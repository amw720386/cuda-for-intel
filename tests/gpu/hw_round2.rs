//! Round 2: exited threads at barriers, special registers, shared alignment, atomics, subgroup sizes.
use super::hw_common::*;
use super::{Handle, library, ok, symbols};
use std::ffi::c_void;

const GUARD: &str = "\tmov.u32 %r0, %tid.x;\n\tmov.u32 %r1, %ctaid.x;\n\tmov.u32 %r2, %ntid.x;\n\tmad.lo.s32 %r3, %r1, %r2, %r0;\n";

// ------------------------------------------------ early exit before bar.sync

fn exit_barrier_ptx(block: u32) -> String {
    let extra = format!("\t.shared .align 4 .b8 buf[{}];\n", block * 4);
    let mut s = prologue("exitbar_k", 2, 1, &extra);
    s += GUARD;
    // Tail threads leave before the barriers (the usual nvcc guard).
    s += "\tsetp.ge.u32 %p1, %r0, %r100;\n\t@%p1 bra $done;\n";
    s += "\tmul.wide.u32 %rd20, %r0, 4;\n\tadd.s64 %rd21, %rd1, %rd20;\n\tld.global.u32 %r4, [%rd21];\n\tmov.u32 %r9, buf;\n\tshl.b32 %r10, %r0, 2;\n\tadd.s32 %r10, %r10, %r9;\n";
    for r in 0..3 {
        s += "\tst.shared.u32 [%r10], %r4;\n\tbar.sync 0;\n";
        s += &format!(
            "\tadd.s32 %r11, %r0, {};\n\trem.u32 %r11, %r11, %r100;\n\tshl.b32 %r11, %r11, 2;\n\tadd.s32 %r11, %r11, %r9;\n\tld.shared.u32 %r12, [%r11];\n\tbar.sync 0;\n\tmad.lo.s32 %r4, %r4, 7, %r12;\n",
            r + 1
        );
    }
    s += "\tadd.s64 %rd22, %rd2, %rd20;\n\tst.global.u32 [%rd22], %r4;\n$done:\n\tret;\n}\n";
    s
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_exited_threads_do_not_deadlock_bar_sync() {
    let mut rng = Rng(0xe417_e417_e417_e417);
    for (block, n) in [(64u32, 40u32), (64, 10), (128, 33), (256, 100), (256, 255), (256, 1), (128, 128)] {
        let x0: Vec<u32> = (0..n).map(|_| rng.u32()).collect();
        let mut x = x0.clone();
        for r in 0..3u32 {
            let prev = x.clone();
            for t in 0..n {
                x[t as usize] = prev[t as usize].wrapping_mul(7).wrapping_add(prev[((t + r + 1) % n) as usize]);
            }
        }
        let mut padded = x0.clone();
        padded.resize(block as usize, 0);
        let mut bufs = vec![bytes_of_u32(&padded), vec![0xeeu8; block as usize * 4]];
        launch(&exit_barrier_ptx(block), "exitbar_k", 1, block, &mut bufs, &[n]).unwrap();
        let out = u32s_of(&bufs[1]);
        assert_eq!(&out[..n as usize], &x[..], "block {block} n {n}");
        assert!(out[n as usize..].iter().all(|v| *v == 0xeeee_eeee), "exited threads must not store");
    }
}

// ------------------------------------------------ special registers

fn special_ptx() -> String {
    let regs = [
        "%warpsize", "%laneid", "%lanemask_eq", "%lanemask_lt", "%lanemask_le", "%lanemask_gt",
        "%lanemask_ge", "%warpid", "%nwarpid", "%smid", "%nsmid", "%clock", "%gridid",
        "%cluster_ctaid.x", "%cluster_nctaid.x", "%cluster_ctarank", "%cluster_nctarank",
    ];
    let mut s = prologue("spec_k", 1, 0, "");
    s += GUARD;
    s += &format!("\tmul.wide.u32 %rd20, %r3, {};\n\tadd.s64 %rd21, %rd1, %rd20;\n", regs.len() * 4 + 8);
    for (i, r) in regs.iter().enumerate() {
        s += &format!("\tmov.u32 %r{}, {r};\n\tst.global.u32 [%rd21+{}], %r{};\n", 20 + i, i * 4, 20 + i);
    }
    s += &format!("\tmov.u64 %rd25, %clock64;\n\tst.global.u64 [%rd21+{}], %rd25;\n", regs.len() * 4);
    s += "\tret;\n}\n";
    s
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_special_registers_have_sensible_values() {
    let n = 64usize;
    let stride = 17 * 4 + 8;
    let mut bufs = vec![vec![0xffu8; n * stride]];
    launch(&special_ptx(), "spec_k", 1, n as u32, &mut bufs, &[]).unwrap();
    let w = u32s_of(&bufs[0]);
    let per = stride / 4;
    for t in 0..n {
        let r = &w[t * per..t * per + 17];
        let lane = (t % 32) as u32;
        assert_eq!(r[0], 32, "%warpsize");
        assert_eq!(r[1], lane, "%laneid");
        assert_eq!(r[2], 1 << lane, "%lanemask_eq");
        assert_eq!(r[3], (1u32 << lane) - 1, "%lanemask_lt");
        assert_eq!(r[4], ((1u32 << lane) - 1) | (1 << lane), "%lanemask_le");
        assert_eq!(r[5], !(((1u32 << lane) - 1) | (1 << lane)), "%lanemask_gt");
        assert_eq!(r[6], !((1u32 << lane) - 1), "%lanemask_ge");
        assert_eq!(r[7], (t / 32) as u32, "%warpid");
        assert!(r[8] >= 2, "%nwarpid");
        assert!(r[10] >= 1, "%nsmid");
        assert_eq!((r[13], r[14], r[15], r[16]), (0, 1, 0, 1), "cluster registers");
    }
}

// ------------------------------------------------ shared alignment above 16

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_shared_alignment_32_64_128() {
    let extra = "\t.shared .align 4 .b8 a[4];\n\t.shared .align 128 .b8 b[256];\n\t.shared .align 32 .b8 c[40];\n\t.shared .align 64 .b8 d[64];\n";
    let mut s = prologue("salign2_k", 1, 0, extra);
    s += "\tmov.u32 %r5, b;\n\tmov.u32 %r6, c;\n\tmov.u32 %r7, d;\n\tmov.u32 %r8, a;\n\tst.global.u32 [%rd1], %r5;\n\tst.global.u32 [%rd1+4], %r6;\n\tst.global.u32 [%rd1+8], %r7;\n\tst.global.u32 [%rd1+12], %r8;\n\tret;\n}\n";
    let mut bufs = vec![vec![0u8; 16]];
    launch(&s, "salign2_k", 1, 1, &mut bufs, &[]).unwrap();
    let o = u32s_of(&bufs[0]);
    assert_eq!((o[0] % 128, o[1] % 32, o[2] % 64), (0, 0, 0), "offsets {o:?}");
    let mut ranges = [(o[3], 4u32), (o[0], 256), (o[1], 40), (o[2], 64)];
    ranges.sort();
    for pair in ranges.windows(2) {
        assert!(pair[0].0 + pair[0].1 <= pair[1].0, "overlapping shared symbols {o:?}");
    }
}

// ------------------------------------------------ atomics

fn atom_ptx(op: &str, ty: &str, shared: bool) -> String {
    let wide = ty.ends_with("64");
    let extra = if shared { "\t.shared .align 8 .b8 cell[8];\n" } else { "" };
    let mut s = prologue("atom2_k", 3, 1, extra);
    s += GUARD;
    s += "\tmul.wide.u32 %rd20, %r3, 8;\n\tadd.s64 %rd21, %rd2, %rd20;\n\tadd.s64 %rd22, %rd3, %rd20;\n";
    s += "\tld.global.u64 %rd30, [%rd21];\n\tcvt.u32.u64 %r10, %rd30;\n";
    let (val, old) = if wide { ("%rd30", "%rd31") } else { ("%r10", "%r11") };
    let addr;
    if shared {
        s += "\tmov.u32 %r9, cell;\n\tsetp.ne.u32 %p2, %r0, 0;\n\t@%p2 bra $init;\n\tld.global.u64 %rd26, [%rd1];\n";
        s += if wide { "\tst.shared.u64 [%r9], %rd26;\n" } else { "\tcvt.u32.u64 %r12, %rd26;\n\tst.shared.u32 [%r9], %r12;\n" };
        s += "$init:\n\tbar.sync 0;\n";
        addr = "%r9".to_string();
    } else {
        s += "\tsetp.ge.u32 %p1, %r3, %r100;\n\t@%p1 bra $done;\n";
        addr = "%rd1".to_string();
    }
    let sp = if shared { "shared" } else { "global" };
    if op == "cas" {
        let new = if wide { "%rd30" } else { "%r10" };
        s += &format!("\tatom.{sp}.cas.{ty} {old}, [{addr}], 0, {new};\n");
    } else {
        s += &format!("\tatom.{sp}.{op}.{ty} {old}, [{addr}], {val};\n");
    }
    s += &format!("\tst.global.{} [%rd22], {old};\n", if wide { "u64" } else { "u32" });
    if shared {
        s += "\tbar.sync 0;\n\tsetp.ne.u32 %p2, %r0, 0;\n\t@%p2 bra $done;\n";
        s += if wide { "\tld.shared.u64 %rd27, [%r9];\n" } else { "\tld.shared.u32 %r12, [%r9];\n\tcvt.u64.u32 %rd27, %r12;\n" };
        s += "\tst.global.u64 [%rd1], %rd27;\n";
    }
    s += "$done:\n\tret;\n}\n";
    s
}

fn apply(op: &str, ty: &str, old: u64, v: u64) -> u64 {
    let wide = ty.ends_with("64");
    let mask = if wide { u64::MAX } else { u32::MAX as u64 };
    let signed = ty.starts_with('s');
    let sx = |x: u64| if wide { x as i64 } else { x as u32 as i32 as i64 };
    (match op {
        "add" => old.wrapping_add(v),
        "and" => old & v,
        "or" => old | v,
        "xor" => old ^ v,
        "min" => if signed { if sx(v) < sx(old) { v } else { old } } else { old.min(v) },
        "max" => if signed { if sx(v) > sx(old) { v } else { old } } else { old.max(v) },
        "inc" => if old >= v { 0 } else { old + 1 },
        "dec" => if old == 0 || old > v { v } else { old - 1 },
        _ => unreachable!(),
    }) & mask
}

fn run_atom(op: &str, ty: &str, shared: bool, rng: &mut Rng) {
    let wide = ty.ends_with("64");
    let mask = if wide { u64::MAX } else { u32::MAX as u64 };
    let n: usize = if shared { 128 } else { 1500 };
    let block = if shared { 128 } else { 250 };
    let mut r64 = || ((rng.u32() as u64) << 32 | rng.u32() as u64) & mask;
    let (init, vals): (u64, Vec<u64>) = match op {
        "inc" | "dec" => {
            let bound = 1 + (rng.u32() % 20) as u64;
            (if op == "dec" { bound / 2 + 1 } else { 0 }, vec![bound; n])
        }
        "cas" => (0, (0..n as u64).map(|i| i + 1).collect()),
        "exch" | "add" | "min" | "max" | "and" | "or" | "xor" => {
            let v: Vec<u64> = (0..n).map(|i| match (op, i % 3) { ("and", _) => r64() | r64() | r64(), (_, 0) => r64(), (_, 1) => r64() >> 40, _ => r64() >> 3 }).collect();
            (if op == "and" { mask } else { r64() }, v)
        }
        _ => unreachable!(),
    };
    let mut target = init.to_le_bytes().to_vec();
    let mut vb: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut ob = vec![0u8; n * 8];
    let mut bufs = vec![std::mem::take(&mut target), std::mem::take(&mut vb), std::mem::take(&mut ob)];
    let grid = (n / block) as u32;
    launch(&atom_ptx(op, ty, shared), "atom2_k", grid, block as u32, &mut bufs, &[n as u32])
        .unwrap_or_else(|e| panic!("atom.{op}.{ty} shared={shared}: {e}"));
    let w = |b: &[u8], i: usize| u64::from_le_bytes(b[i * 8..i * 8 + 8].try_into().unwrap()) & mask;
    let fin = w(&bufs[0], 0);
    let olds: Vec<u64> = (0..n).map(|i| w(&bufs[2], i)).collect();
    let tag = format!("atom.{op}.{ty} shared={shared}");
    match op {
        "cas" => {
            let winners: Vec<usize> = (0..n).filter(|&i| olds[i] == 0).collect();
            assert_eq!(winners.len(), 1, "{tag}: exactly one thread swaps");
            assert_eq!(fin, vals[winners[0]], "{tag}: final value");
            assert!(olds.iter().enumerate().all(|(i, &o)| i == winners[0] || o == fin), "{tag}: losers see winner");
        }
        "exch" => {
            let mut have: Vec<u64> = olds.clone();
            have.push(fin);
            let mut want = vals.clone();
            want.push(init);
            have.sort_unstable();
            want.sort_unstable();
            assert_eq!(have, want, "{tag}: values conserved");
        }
        "inc" | "dec" => {
            let mut cur = init;
            let mut want = vec![];
            for _ in 0..n {
                want.push(cur);
                cur = apply(op, ty, cur, vals[0]);
            }
            let mut have = olds.clone();
            have.sort_unstable();
            want.sort_unstable();
            assert_eq!(have, want, "{tag}: returned olds");
            assert_eq!(fin, cur, "{tag}: final");
        }
        _ => {
            let want = vals.iter().fold(init, |a, &v| apply(op, ty, a, v));
            assert_eq!(fin, want, "{tag}: final value");
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_atomics_32bit_all_ops_global_and_shared() {
    let mut rng = Rng(0xa70a_0032_a70a_0032);
    for shared in [false, true] {
        for ty in ["u32", "s32", "b32"] {
            for op in ["add", "min", "max", "and", "or", "xor", "exch", "cas"] {
                if (op == "min" || op == "max") && ty == "b32" { continue; }
                if matches!(op, "add" | "exch" | "cas") && ty == "s32" && op != "add" { continue; }
                run_atom(op, ty, shared, &mut rng);
            }
        }
        for op in ["inc", "dec"] {
            run_atom(op, "u32", shared, &mut rng);
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_atomics_64bit_all_ops_global_and_shared() {
    let mut rng = Rng(0xa70a_0064_a70a_0064);
    for shared in [false, true] {
        for ty in ["u64", "s64", "b64"] {
            for op in ["add", "min", "max", "and", "or", "xor", "exch", "cas"] {
                if (op == "min" || op == "max") && ty == "b64" { continue; }
                if matches!(op, "exch" | "cas") && ty != "b64" { continue; }
                run_atom(op, ty, shared, &mut rng);
            }
        }
    }
}

// ------------------------------------------------ Level Zero subgroup sizes

#[repr(C)]
struct ComputeProps {
    stype: u32,
    next: *mut c_void,
    max_group_size: u32,
    group_size: [u32; 3],
    group_count: [u32; 3],
    shared_local_memory: u32,
    subgroup_count: u32,
    subgroup_sizes: [u32; 8],
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn hw_level_zero_device_supports_subgroup_size_32() {
    let lib = unsafe { libloading::Library::new("ze_loader.dll") }.expect("ze_loader.dll");
    type Init = unsafe extern "C" fn(u32) -> i32;
    type DriverGet = unsafe extern "C" fn(*mut u32, *mut Handle) -> i32;
    type DeviceGet = unsafe extern "C" fn(Handle, *mut u32, *mut Handle) -> i32;
    type Compute = unsafe extern "C" fn(Handle, *mut ComputeProps) -> i32;
    unsafe {
        let init: libloading::Symbol<Init> = lib.get(b"zeInit\0").unwrap();
        let driver_get: libloading::Symbol<DriverGet> = lib.get(b"zeDriverGet\0").unwrap();
        let device_get: libloading::Symbol<DeviceGet> = lib.get(b"zeDeviceGet\0").unwrap();
        let compute: libloading::Symbol<Compute> = lib.get(b"zeDeviceGetComputeProperties\0").unwrap();
        assert_eq!(init(0), 0);
        let mut drivers = 0u32;
        assert_eq!(driver_get(&mut drivers, std::ptr::null_mut()), 0);
        let mut handles = vec![std::ptr::null_mut(); drivers as usize];
        assert_eq!(driver_get(&mut drivers, handles.as_mut_ptr()), 0);
        let mut seen = 0;
        for driver in handles {
            let mut count = 0u32;
            assert_eq!(device_get(driver, &mut count, std::ptr::null_mut()), 0);
            let mut devices = vec![std::ptr::null_mut(); count as usize];
            assert_eq!(device_get(driver, &mut count, devices.as_mut_ptr()), 0);
            for device in devices {
                let mut props: ComputeProps = std::mem::zeroed();
                props.stype = 2; // ZE_STRUCTURE_TYPE_DEVICE_COMPUTE_PROPERTIES
                assert_eq!(compute(device, &mut props), 0);
                let sizes = &props.subgroup_sizes[..props.subgroup_count as usize];
                println!("subgroup sizes {sizes:?}, max group size {}", props.max_group_size);
                assert!(sizes.contains(&32), "device does not support required subgroup size 32: {sizes:?}");
                assert!(props.max_group_size >= 256 || props.max_group_size >= 32);
                seen += 1;
            }
        }
        assert!(seen > 0);
    }
    let _ = (library as fn() -> libloading::Library, ok_unused());
}

fn ok_unused() {}
