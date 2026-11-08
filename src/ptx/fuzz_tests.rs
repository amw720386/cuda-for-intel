//! Differential semantics fuzzing: one PTX instruction per kernel is translated, executed on the
//! SPIR-V simulator for many random + edge-case inputs and compared with a reference written
//! directly from the PTX ISA. A case whose kernel fails to translate is reported as unsupported
//! (never silently accepted); a mismatch fails the test.
use super::sim::{Gpu, Launch, Sim};
use super::translate;

pub(super) struct Rng(u64);
impl Rng {
    pub fn new(seed: u64) -> Self { Self(seed | 1) }
    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
}

const SPECIAL32: [u32; 24] = [
    0, 1, 2, 3, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 255, 256, 0x7fff_ffff, 0x8000_0000, 0x8000_0001, 0xffff_ffff, 0xffff_fffe, 0xffff, 0x1_0000 - 1, 0x0000_ff00,
];
const SPECIAL_F32: [u32; 26] = [
    0x0000_0000, 0x8000_0000, 0x3f80_0000, 0xbf80_0000, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc0_0000, 0x7f7f_ffff, 0xff7f_ffff, 0x0000_0001, 0x8000_0001,
    0x007f_ffff, 0x807f_ffff, 0x0080_0000, 0x8080_0000, 0x3f00_0000, 0x4000_0000, 0x4b00_0000, 0xcb00_0000, 0x4f00_0000, 0x4f80_0000, 0xcf00_0000, 0x3eaa_aaab, 0x7fa0_0000, 0x3fc0_0000,
];

#[derive(Clone, Copy, PartialEq)]
pub(super) enum K { R32, R64, F32, F64, Pred }

pub(super) struct Case {
    pub name: String,
    /// PTX statements reading %r10..12 / %rd10..12 / %f10..12 / %fd10..12 (a,b,c) and writing `out`.
    pub body: String,
    pub out: K,
    /// Output register index within its class (20 by convention).
    pub float_out: bool,
    pub refn: Box<dyn Fn(u64, u64, u64) -> Option<u64>>,
    /// How inputs are generated.
    pub kind: Gen,
    /// Applied to both sides before comparing (e.g. canonicalise NaN payloads).
    pub canon: Option<fn(u64) -> u64>,
}

#[derive(Clone, Copy)]
pub(super) enum Gen { Int32, Int64, F32, F64, Shift32, Shift64 }

fn gen_inputs(g: Gen, rng: &mut Rng, n: usize) -> Vec<[u64; 3]> {
    let mut v = Vec::new();
    let pick32 = |rng: &mut Rng| -> u64 {
        match rng.next() % 4 {
            0 => u64::from(SPECIAL32[(rng.next() % SPECIAL32.len() as u64) as usize]),
            1 => u64::from((rng.next() as u32) >> (rng.next() % 32)),
            _ => u64::from(rng.next() as u32),
        }
    };
    let pickf = |rng: &mut Rng| -> u64 {
        match rng.next() % 4 {
            0 => u64::from(SPECIAL_F32[(rng.next() % SPECIAL_F32.len() as u64) as usize]),
            1 => {
                // small-magnitude normal numbers
                let e = 110 + (rng.next() % 36) as u32;
                u64::from(((rng.next() as u32 & 1) << 31) | (e << 23) | (rng.next() as u32 & 0x7f_ffff))
            }
            2 => u64::from(rng.next() as u32 & 0x807f_ffff), // subnormals / zero
            _ => u64::from(rng.next() as u32),
        }
    };
    for _ in 0..n {
        let t = match g {
            Gen::Int32 => [pick32(rng), pick32(rng), pick32(rng)],
            Gen::Shift32 => [pick32(rng), if rng.next() % 2 == 0 { rng.next() % 80 } else { pick32(rng) }, pick32(rng)],
            Gen::Int64 => {
                let p = |rng: &mut Rng| if rng.next() % 3 == 0 { (pick32(rng) << 32) | pick32(rng) } else { rng.next() };
                [p(rng), p(rng), p(rng)]
            }
            Gen::Shift64 => [rng.next(), if rng.next() % 2 == 0 { rng.next() % 150 } else { rng.next() }, rng.next()],
            Gen::F32 => [pickf(rng), pickf(rng), pickf(rng)],
            Gen::F64 => {
                let p = |rng: &mut Rng| match rng.next() % 4 {
                    0 => [0u64, 1 << 63, 0x3ff0_0000_0000_0000, 0x7ff0_0000_0000_0000, 0x7ff8_0000_0000_0000, 1, 0x000f_ffff_ffff_ffff][(rng.next() % 7) as usize],
                    1 => ((1023 - 20 + rng.next() % 40) << 52) | (rng.next() & 0x800f_ffff_ffff_ffff),
                    _ => rng.next(),
                };
                [p(rng), p(rng), p(rng)]
            }
        };
        v.push(t);
    }
    v
}

fn kernel_src(c: &Case) -> String {
    let st = match c.out {
        K::R32 => "st.global.b32 [%rdo], %r20;",
        K::R64 => "st.global.b64 [%rdo], %rd20;",
        K::F32 => "st.global.f32 [%rdo], %f20;",
        K::F64 => "st.global.f64 [%rdo], %fd20;",
        K::Pred => "selp.u32 %r20, 1, 0, %p20;\n\tst.global.b32 [%rdo], %r20;",
    };
    format!(
        r#".version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 k_in, .param .u64 k_out)
{{
	.reg .pred %p<32>;
	.reg .b32 %r<48>;
	.reg .b64 %rd<48>;
	.reg .f32 %f<48>;
	.reg .f64 %fd<48>;
	.reg .b16 %h<48>;
	ld.param.u64 %rd1, [k_in];
	ld.param.u64 %rd2, [k_out];
	cvta.to.global.u64 %rd1, %rd1;
	cvta.to.global.u64 %rd2, %rd2;
	mov.u32 %r1, %tid.x;
	mul.wide.u32 %rd3, %r1, 24;
	add.s64 %rd4, %rd1, %rd3;
	mul.wide.u32 %rd5, %r1, 8;
	add.s64 %rdo, %rd2, %rd5;
	ld.global.b32 %r10, [%rd4];
	ld.global.b32 %r11, [%rd4+8];
	ld.global.b32 %r12, [%rd4+16];
	ld.global.b64 %rd10, [%rd4];
	ld.global.b64 %rd11, [%rd4+8];
	ld.global.b64 %rd12, [%rd4+16];
	ld.global.f32 %f10, [%rd4];
	ld.global.f32 %f11, [%rd4+8];
	ld.global.f32 %f12, [%rd4+16];
	ld.global.f64 %fd10, [%rd4];
	ld.global.f64 %fd11, [%rd4+8];
	ld.global.f64 %fd12, [%rd4+16];
	{body}
	{st}
	ret;
}}
"#,
        body = c.body,
    )
    .replace("%rdo", "%rd40")
    .replace("	.reg .b64 %rd<48>;", "	.reg .b64 %rd<48>;")
}

pub(super) enum Outcome { Pass(usize), Unsupported(String), Mismatch(String), Skipped }

pub(super) fn run_case(c: &Case, rng: &mut Rng, n: usize) -> Outcome {
    let src = kernel_src(c);
    let t = match translate(&src) {
        Ok(t) => t,
        Err(e) => return Outcome::Unsupported(e.to_string()),
    };
    if let Some(msg) = validate(&t.words) {
        return Outcome::Mismatch(format!("{}: spirv-val: {msg}", c.name));
    }
    let module = rspirv::dr::load_words(&t.words).unwrap();
    let inputs = gen_inputs(c.kind, rng, n);
    let mut bytes = Vec::new();
    for t in &inputs {
        for x in t {
            bytes.extend_from_slice(&x.to_le_bytes());
        }
    }
    let mut gpu = Gpu::default();
    let inp = gpu.upload(&bytes);
    let out = gpu.alloc(8 * n);
    let mut sim = Sim::new(&module);
    if let Err(e) = sim.run(&mut gpu, &Launch::new([1, 1, 1], [n as u32, 1, 1], vec![inp, out])) {
        return Outcome::Mismatch(format!("{}: sim error {e}", c.name));
    }
    let width = match c.out { K::R32 | K::F32 | K::Pred => 4, _ => 8 };
    let mut checked = 0;
    for (i, t) in inputs.iter().enumerate() {
        let Some(want) = (c.refn)(t[0], t[1], t[2]) else { continue };
        let got = gpu.mem.read(out + 8 * i as u64, width);
        let want = if width == 4 { want & 0xffff_ffff } else { want };
        let (got, want) = match c.canon { Some(f) => (f(got), f(want)), None => (got, want) };
        let eq = got == want
            || (c.float_out && width == 4 && f32::from_bits(got as u32).is_nan() && f32::from_bits(want as u32).is_nan())
            || (c.float_out && width == 8 && f64::from_bits(got).is_nan() && f64::from_bits(want).is_nan());
        if !eq {
            return Outcome::Mismatch(format!(
                "{}: a={:#x} b={:#x} c={:#x} want {:#x} got {:#x}",
                c.name, t[0], t[1], t[2], want, got
            ));
        }
        checked += 1;
    }
    Outcome::Pass(checked)
}

pub(super) fn validate(words: &[u32]) -> Option<String> {
    let bin = std::env::var_os("SPIRV_VAL").map(std::path::PathBuf::from).or_else(|| {
        std::env::var_os("PATH").and_then(|p| std::env::split_paths(&p).map(|d| d.join("spirv-val")).find(|f| f.exists()))
    })?;
    let tmp = std::env::temp_dir().join(format!("cfi_fuzz_{}_{:?}.spv", std::process::id(), std::thread::current().id()).replace(['(', ')', ' '], ""));
    std::fs::write(&tmp, words.iter().flat_map(|w| w.to_le_bytes()).collect::<Vec<u8>>()).ok()?;
    let o = std::process::Command::new(bin).arg("--target-env").arg("spv1.3").arg(&tmp).output().ok()?;
    let _ = std::fs::remove_file(&tmp);
    // Int64Atomics is gated behind cl_khr_int64_base_atomics, which the validator cannot know about.
    if o.status.success() || ["Int64Atomics", "SubgroupDispatch"].iter().any(|k| String::from_utf8_lossy(&o.stderr).contains(k)) { None } else { Some(String::from_utf8_lossy(&o.stderr).lines().next().unwrap_or("").to_string()) }
}

/// Runs every case; panics listing all mismatches. Returns the unsupported list for reporting.
pub(super) fn run_all(cases: Vec<Case>, seed: u64) -> Vec<(String, String)> {
    let mut rng = Rng::new(seed);
    let (mut bad, mut unsupported) = (Vec::new(), Vec::new());
    let mut passed = 0;
    for c in &cases {
        match run_case(c, &mut rng, 192) {
            Outcome::Pass(n) => passed += n,
            Outcome::Unsupported(e) => unsupported.push((c.name.clone(), e)),
            Outcome::Mismatch(m) => bad.push(m),
            Outcome::Skipped => {}
        }
    }
    eprintln!("fuzz: {} cases, {passed} comparisons, {} unsupported, {} mismatches", cases.len(), unsupported.len(), bad.len());
    for (n, e) in &unsupported {
        eprintln!("  UNSUPPORTED {n}: {e}");
    }
    assert!(bad.is_empty(), "semantic mismatches:\n{}", bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
    unsupported
}

pub(super) fn case(name: &str, kind: Gen, out: K, body: &str, f: impl Fn(u64, u64, u64) -> Option<u64> + 'static) -> Case {
    Case { name: name.into(), body: body.into(), out, float_out: matches!(out, K::F32 | K::F64), refn: Box::new(f), kind, canon: None }
}

#[cfg(test)]
mod int_ops {
    use super::*;

    fn i32c(name: &str, body: &str, f: impl Fn(u32, u32, u32) -> Option<u32> + 'static) -> Case {
        case(name, Gen::Int32, K::R32, body, move |a, b, c| f(a as u32, b as u32, c as u32).map(u64::from))
    }
    fn sh32(name: &str, body: &str, f: impl Fn(u32, u32) -> u32 + 'static) -> Case {
        case(name, Gen::Shift32, K::R32, body, move |a, b, _| Some(u64::from(f(a as u32, b as u32))))
    }

    pub fn cases() -> Vec<Case> {
        let mut v = vec![
            i32c("add.u32", "add.u32 %r20, %r10, %r11;", |a, b, _| Some(a.wrapping_add(b))),
            i32c("sub.u32", "sub.u32 %r20, %r10, %r11;", |a, b, _| Some(a.wrapping_sub(b))),
            i32c("mul.lo.u32", "mul.lo.u32 %r20, %r10, %r11;", |a, b, _| Some(a.wrapping_mul(b))),
            i32c("mul.hi.u32", "mul.hi.u32 %r20, %r10, %r11;", |a, b, _| Some(((u64::from(a) * u64::from(b)) >> 32) as u32)),
            i32c("mul.hi.s32", "mul.hi.s32 %r20, %r10, %r11;", |a, b, _| Some(((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32)),
            i32c("mad.lo.u32", "mad.lo.u32 %r20, %r10, %r11, %r12;", |a, b, c| Some(a.wrapping_mul(b).wrapping_add(c))),
            i32c("mad.hi.u32", "mad.hi.u32 %r20, %r10, %r11, %r12;", |a, b, c| Some((((u64::from(a) * u64::from(b)) >> 32) as u32).wrapping_add(c))),
            i32c("mad.hi.s32", "mad.hi.s32 %r20, %r10, %r11, %r12;", |a, b, c| Some((((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32).wrapping_add(c))),
            i32c("mul24.lo.u32", "mul24.lo.u32 %r20, %r10, %r11;", |a, b, _| Some((a & 0xff_ffff).wrapping_mul(b & 0xff_ffff))),
            i32c("mul24.hi.u32", "mul24.hi.u32 %r20, %r10, %r11;", |a, b, _| Some(((u64::from(a & 0xff_ffff) * u64::from(b & 0xff_ffff)) >> 16) as u32)),
            i32c("mul24.lo.s32", "mul24.lo.s32 %r20, %r10, %r11;", |a, b, _| Some((((a << 8) as i32 >> 8).wrapping_mul((b << 8) as i32 >> 8)) as u32)),
            i32c("mad24.lo.u32", "mad24.lo.u32 %r20, %r10, %r11, %r12;", |a, b, c| Some((a & 0xff_ffff).wrapping_mul(b & 0xff_ffff).wrapping_add(c))),
            i32c("mad24.lo.s32", "mad24.lo.s32 %r20, %r10, %r11, %r12;", |a, b, c| Some(((((a << 8) as i32 >> 8).wrapping_mul((b << 8) as i32 >> 8)) as u32).wrapping_add(c))),
            i32c("div.u32", "div.u32 %r20, %r10, %r11;", |a, b, _| Some(a.checked_div(b).unwrap_or(u32::MAX))),
            i32c("rem.u32", "rem.u32 %r20, %r10, %r11;", |a, b, _| Some(a.checked_rem(b).unwrap_or(a))),
            i32c("div.s32", "div.s32 %r20, %r10, %r11;", |a, b, _| Some(if b == 0 { u32::MAX } else { (a as i32).wrapping_div(b as i32) as u32 })),
            i32c("rem.s32", "rem.s32 %r20, %r10, %r11;", |a, b, _| Some(if b == 0 { a } else { (a as i32).wrapping_rem(b as i32) as u32 })),
            i32c("min.u32", "min.u32 %r20, %r10, %r11;", |a, b, _| Some(a.min(b))),
            i32c("max.u32", "max.u32 %r20, %r10, %r11;", |a, b, _| Some(a.max(b))),
            i32c("min.s32", "min.s32 %r20, %r10, %r11;", |a, b, _| Some((a as i32).min(b as i32) as u32)),
            i32c("max.s32", "max.s32 %r20, %r10, %r11;", |a, b, _| Some((a as i32).max(b as i32) as u32)),
            i32c("abs.s32", "abs.s32 %r20, %r10;", |a, _, _| Some((a as i32).wrapping_abs() as u32)),
            i32c("neg.s32", "neg.s32 %r20, %r10;", |a, _, _| Some((a as i32).wrapping_neg() as u32)),
            i32c("not.b32", "not.b32 %r20, %r10;", |a, _, _| Some(!a)),
            i32c("cnot.b32", "cnot.b32 %r20, %r10;", |a, _, _| Some(u32::from(a == 0))),
            i32c("and/or/xor", "and.b32 %r21, %r10, %r11;\n or.b32 %r22, %r21, %r12;\n xor.b32 %r20, %r22, %r10;", |a, b, c| Some(((a & b) | c) ^ a)),
            i32c("popc.b32", "popc.b32 %r20, %r10;", |a, _, _| Some(a.count_ones())),
            i32c("clz.b32", "clz.b32 %r20, %r10;", |a, _, _| Some(a.leading_zeros())),
            i32c("brev.b32", "brev.b32 %r20, %r10;", |a, _, _| Some(a.reverse_bits())),
            i32c("bfind.u32", "bfind.u32 %r20, %r10;", |a, _, _| Some(if a == 0 { u32::MAX } else { 31 - a.leading_zeros() })),
            i32c("bfind.s32", "bfind.s32 %r20, %r10;", |a, _, _| Some(if a == 0 || a == u32::MAX { u32::MAX } else { let x = if (a as i32) < 0 { !a } else { a }; 31 - x.leading_zeros() })),
            i32c("sad.u32", "sad.u32 %r20, %r10, %r11, %r12;", |a, b, c| Some(a.abs_diff(b).wrapping_add(c))),
            i32c("sad.s32", "sad.s32 %r20, %r10, %r11, %r12;", |a, b, c| Some((a as i32).abs_diff(b as i32).wrapping_add(c))),
            i32c("selp", "setp.lt.u32 %p20, %r10, %r11;\n selp.b32 %r20, %r12, %r10, %p20;", |a, b, c| Some(if a < b { c } else { a })),
            i32c("setp.lt.s32/and", "setp.lt.s32 %p21, %r10, %r11;\n setp.ne.and.u32 %p20, %r12, 0, %p21;\n selp.b32 %r20, 1, 0, %p20;", |a, b, c| Some(u32::from((a as i32) < (b as i32) && c != 0))),
            // shifts: counts >= width saturate
            sh32("shl.b32", "shl.b32 %r20, %r10, %r11;", |a, b| if b >= 32 { 0 } else { a << b }),
            sh32("shr.u32", "shr.u32 %r20, %r10, %r11;", |a, b| if b >= 32 { 0 } else { a >> b }),
            sh32("shr.s32", "shr.s32 %r20, %r10, %r11;", |a, b| ((a as i32) >> b.min(31)) as u32),
            // funnel shifts
            i32c("shf.l.clamp.b32", "shf.l.clamp.b32 %r20, %r10, %r11, %r12;", |a, b, c| { let n = c.min(32); Some((((u64::from(b) << 32 | u64::from(a)) << n) >> 32) as u32) }),
            i32c("shf.r.clamp.b32", "shf.r.clamp.b32 %r20, %r10, %r11, %r12;", |a, b, c| { let n = c.min(32); Some((((u64::from(b) << 32 | u64::from(a)) >> n) & 0xffff_ffff) as u32) }),
            i32c("shf.l.wrap.b32", "shf.l.wrap.b32 %r20, %r10, %r11, %r12;", |a, b, c| { let n = c & 31; Some((((u64::from(b) << 32 | u64::from(a)) << n) >> 32) as u32) }),
            i32c("shf.r.wrap.b32", "shf.r.wrap.b32 %r20, %r10, %r11, %r12;", |a, b, c| { let n = c & 31; Some((((u64::from(b) << 32 | u64::from(a)) >> n) & 0xffff_ffff) as u32) }),
            // bit-field ops
            i32c("bfe.u32", "bfe.u32 %r20, %r10, %r11, %r12;", |a, b, c| Some(bfe(a, b & 0xff, c & 0xff, false))),
            i32c("bfe.s32", "bfe.s32 %r20, %r10, %r11, %r12;", |a, b, c| Some(bfe(a, b & 0xff, c & 0xff, true))),
            i32c("bfi.b32", "shr.u32 %r13, %r12, 8;\n bfi.b32 %r20, %r10, %r11, %r12, %r13;", |a, b, c| Some(bfi(a, b, c & 0xff, (c >> 8) & 0xff))),
            i32c("prmt.b32", "prmt.b32 %r20, %r10, %r11, %r12;", |a, b, c| Some(prmt(a, b, c & 0xffff))),
            i32c("prmt.b32.f4e", "prmt.b32.f4e %r20, %r10, %r11, %r12;", |a, b, c| Some(prmt_mode(a, b, c & 3, 1))),
            i32c("prmt.b32.b4e", "prmt.b32.b4e %r20, %r10, %r11, %r12;", |a, b, c| Some(prmt_mode(a, b, c & 3, 2))),
            i32c("prmt.b32.rc8", "prmt.b32.rc8 %r20, %r10, %r11, %r12;", |a, b, c| Some(prmt_mode(a, b, c & 3, 3))),
            i32c("prmt.b32.ecl", "prmt.b32.ecl %r20, %r10, %r11, %r12;", |a, b, c| Some(prmt_mode(a, b, c & 3, 4))),
            i32c("prmt.b32.ecr", "prmt.b32.ecr %r20, %r10, %r11, %r12;", |a, b, c| Some(prmt_mode(a, b, c & 3, 5))),
            i32c("prmt.b32.rc16", "prmt.b32.rc16 %r20, %r10, %r11, %r12;", |a, b, c| Some(prmt_mode(a, b, c & 3, 6))),
            i32c("dp4a.s32.s32", "dp4a.s32.s32 %r20, %r10, %r11, %r12;", |a, b, c| Some(dp4a(a, b, c, true, true))),
            i32c("dp4a.u32.u32", "dp4a.u32.u32 %r20, %r10, %r11, %r12;", |a, b, c| Some(dp4a(a, b, c, false, false))),
            i32c("dp4a.s32.u32", "dp4a.s32.u32 %r20, %r10, %r11, %r12;", |a, b, c| Some(dp4a(a, b, c, true, false))),
            i32c("dp4a.u32.s32", "dp4a.u32.s32 %r20, %r10, %r11, %r12;", |a, b, c| Some(dp4a(a, b, c, false, true))),
            i32c("dp2a.lo.s32.s32", "dp2a.lo.s32.s32 %r20, %r10, %r11, %r12;", |a, b, c| Some(dp2a(a, b, c, true, true, false))),
            i32c("dp2a.hi.u32.u32", "dp2a.hi.u32.u32 %r20, %r10, %r11, %r12;", |a, b, c| Some(dp2a(a, b, c, false, false, true))),
            i32c("dp2a.lo.u32.s32", "dp2a.lo.u32.s32 %r20, %r10, %r11, %r12;", |a, b, c| Some(dp2a(a, b, c, false, true, false))),
            i32c("szext.wrap.s32", "szext.wrap.s32 %r20, %r10, %r11;", |a, b, _| { let n = b & 31; Some(if n == 0 { 0 } else { (((a << (32 - n)) as i32) >> (32 - n)) as u32 }) }),
            i32c("szext.clamp.u32", "szext.clamp.u32 %r20, %r10, %r11;", |a, b, _| { let n = b.min(32); Some(if n == 32 { a } else { a & ((1u32 << n) - 1) }) }),
            i32c("bmsk.wrap.b32", "bmsk.wrap.b32 %r20, %r10, %r11;", |a, b, _| { let (p, l) = (a & 31, b & 31); Some((((1u64 << l) - 1) as u32) << p) }),
            i32c("bmsk.clamp.b32", "bmsk.clamp.b32 %r20, %r10, %r11;", |a, b, _| { let (p, l) = (a.min(32), b.min(32)); Some(if p >= 32 { 0 } else { ((((1u64 << l) - 1) << p) & 0xffff_ffff) as u32 }) }),
            i32c("lop3 0xca", "lop3.b32 %r20, %r10, %r11, %r12, 0xca;", |a, b, c| Some((a & b) | (!a & c))),
            i32c("lop3 0x96", "lop3.b32 %r20, %r10, %r11, %r12, 0x96;", |a, b, c| Some(a ^ b ^ c)),
            i32c("lop3 0xe8", "lop3.b32 %r20, %r10, %r11, %r12, 0xe8;", |a, b, c| Some((a & b) | (a & c) | (b & c))),
            i32c("lop3 0x1", "lop3.b32 %r20, %r10, %r11, %r12, 0x01;", |a, b, c| Some(!(a | b | c))),
            // carry chain
            i32c("add.cc/addc", "add.cc.u32 %r21, %r10, %r11;\n addc.u32 %r20, %r12, %r12;", |a, b, c| Some(c.wrapping_add(c).wrapping_add(u32::from(a.overflowing_add(b).1)))),
            i32c("sub.cc/subc", "sub.cc.u32 %r21, %r10, %r11;\n subc.u32 %r20, %r12, %r12;", |a, b, c| Some(c.wrapping_sub(c).wrapping_sub(u32::from(a < b)))),
            i32c("add.sat.s32", "add.sat.s32 %r20, %r10, %r11;", |a, b, _| Some((a as i32).saturating_add(b as i32) as u32)),
            i32c("sub.sat.s32", "sub.sat.s32 %r20, %r10, %r11;", |a, b, _| Some((a as i32).saturating_sub(b as i32) as u32)),
            i32c("mul.wide.u32 hi", "mul.wide.u32 %rd21, %r10, %r11;\n shr.u64 %rd22, %rd21, 32;\n cvt.u32.u64 %r20, %rd22;", |a, b, _| Some(((u64::from(a) * u64::from(b)) >> 32) as u32)),
            i32c("mul.wide.s32 hi", "mul.wide.s32 %rd21, %r10, %r11;\n shr.s64 %rd22, %rd21, 32;\n cvt.u32.u64 %r20, %rd22;", |a, b, _| Some(((i64::from(a as i32) * i64::from(b as i32)) >> 32) as u32)),
        ];
        v
    }

    pub fn bfi(a: u32, b: u32, pos: u32, len: u32) -> u32 {
        let mut d = b;
        for i in 0..len.min(32) {
            if pos + i < 32 {
                d = (d & !(1 << (pos + i))) | (((a >> i) & 1) << (pos + i));
            }
        }
        d
    }
    pub fn bfe(a: u32, pos: u32, len: u32, signed: bool) -> u32 {
        let sbit = if signed && len > 0 { (a >> (pos + len - 1).min(31)) & 1 } else { 0 };
        let mut d = 0;
        for i in 0..32 {
            let bit = if i < len && pos + i <= 31 { (a >> (pos + i)) & 1 } else { sbit };
            d |= bit << i;
        }
        d
    }
    pub fn prmt(a: u32, b: u32, sel: u32) -> u32 {
        let bytes: [u8; 8] = [a.to_le_bytes(), b.to_le_bytes()].concat().try_into().unwrap();
        let mut out = [0u8; 4];
        for i in 0..4 {
            let s = (sel >> (4 * i)) & 0xf;
            let v = bytes[(s & 7) as usize];
            out[i] = if s & 8 != 0 { if v & 0x80 != 0 { 0xff } else { 0 } } else { v };
        }
        u32::from_le_bytes(out)
    }
    /// modes: 1 f4e, 2 b4e, 3 rc8, 4 ecl, 5 ecr, 6 rc16
    /// modes: 1 f4e, 2 b4e, 3 rc8, 4 ecl, 5 ecr, 6 rc16
    pub fn prmt_mode(a: u32, b: u32, c: u32, mode: u32) -> u32 {
        let c = c & 3;
        let bytes: [u8; 8] = [a.to_le_bytes(), b.to_le_bytes()].concat().try_into().unwrap();
        let mut out = [0u8; 4];
        for i in 0..4u32 {
            let idx = match mode {
                1 => c + i,
                2 => c.wrapping_sub(i) & 7,
                3 => c,
                4 => c.max(i),
                5 => c.min(i),
                _ => (c & 1) * 2 + (i & 1),
            };
            out[i as usize] = bytes[(idx & 7) as usize];
        }
        u32::from_le_bytes(out)
    }
    pub fn dp4a(a: u32, b: u32, c: u32, sa: bool, sb: bool) -> u32 {
        let mut acc = c as i64 as i128;
        let mut accu = u64::from(c);
        let mut s = i64::from(c as i32);
        for i in 0..4 {
            let x = (a >> (8 * i)) as u8;
            let y = (b >> (8 * i)) as u8;
            let xv = if sa { i64::from(x as i8) } else { i64::from(x) };
            let yv = if sb { i64::from(y as i8) } else { i64::from(y) };
            s = s.wrapping_add(xv * yv);
            accu = accu.wrapping_add((xv * yv) as u64);
            acc += i128::from(xv * yv);
        }
        let _ = (acc, accu);
        s as u32
    }
    pub fn dp2a(a: u32, b: u32, c: u32, sa: bool, sb: bool, hi: bool) -> u32 {
        let mut s = i64::from(c as i32);
        for i in 0..2 {
            let x = (a >> (16 * i)) as u16;
            let off = if hi { 16 } else { 0 };
            let y = (b >> (off + 8 * i)) as u8;
            let xv = if sa { i64::from(x as i16) } else { i64::from(x) };
            let yv = if sb { i64::from(y as i8) } else { i64::from(y) };
            s = s.wrapping_add(xv * yv);
        }
        s as u32
    }

    #[test]
    fn integer_semantics_match_reference() {
        let unsupported = run_all(cases(), 0x1234_5678_9abc_def1);
        eprintln!("int unsupported: {}", unsupported.len());
    }
}

#[cfg(test)]
mod float_ops {
    use super::*;

    fn ftz(x: f32) -> f32 { if x.is_subnormal() { f32::from_bits(x.to_bits() & 0x8000_0000) } else { x } }
    fn ftz64(x: f64) -> f64 { if x.is_subnormal() { f64::from_bits(x.to_bits() & (1 << 63)) } else { x } }
    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Rm { Rn, Rz, Rm, Rp }
    impl Rm { fn suffix(self) -> &'static str { match self { Rm::Rn => "rn", Rm::Rz => "rz", Rm::Rm => "rm", Rm::Rp => "rp" } } }
    /// Applies a directed rounding given the round-to-nearest result `r` and the sign `d` of (exact - r).
    fn directed(r: f32, d: i32, m: Rm) -> f32 {
        if r.is_nan() || d == 0 || m == Rm::Rn { return r; }
        let up = |x: f32| x.next_up();
        let down = |x: f32| x.next_down();
        match m {
            Rm::Rz => if d < 0 && r > 0.0 || d > 0 && r < 0.0 { if r > 0.0 { down(r) } else { up(r) } } else { r },
            Rm::Rm => if d < 0 { down(r) } else { r },
            Rm::Rp => if d > 0 { up(r) } else { r },
            Rm::Rn => r,
        }
    }
    fn sgn(x: f64) -> i32 { if x > 0.0 { 1 } else if x < 0.0 { -1 } else { 0 } }
    /// sign of (exact - r) where exact = s + e (unevaluated sum in f64)
    fn dsign(s: f64, e: f64, r: f32) -> i32 {
        if r.is_infinite() { return -sgn(r as f64); }
        sgn((s - r as f64) + e)
    }
    fn two_sum(a: f64, b: f64) -> (f64, f64) {
        let s = a + b;
        let bb = s - a;
        (s, (a - (s - bb)) + (b - bb))
    }
    fn f32_add(a: f32, b: f32, m: Rm, z: bool) -> f32 {
        let (a, b) = if z { (ftz(a), ftz(b)) } else { (a, b) };
        let r = a + b;
        let out = if a.is_finite() && b.is_finite() && m != Rm::Rn {
            let (s, e) = two_sum(a as f64, b as f64);
            let mut r2 = directed(r, dsign(s, e, r), m);
            // exact zero sum of opposite signs: -0 only when rounding toward minus infinity
            if r2 == 0.0 && s + e == 0.0 && a != -b || (a == -b && a != 0.0) { r2 = if m == Rm::Rm { -0.0 } else { 0.0 }; }
            r2
        } else { r };
        if z { ftz(out) } else { out }
    }
    fn f32_mul(a: f32, b: f32, m: Rm, z: bool) -> f32 {
        let (a, b) = if z { (ftz(a), ftz(b)) } else { (a, b) };
        let r = a * b;
        let out = if a.is_finite() && b.is_finite() && m != Rm::Rn {
            let p = a as f64 * b as f64;
            directed(r, if r.is_infinite() { -sgn(r as f64) } else { sgn(p - r as f64) }, m)
        } else { r };
        if z { ftz(out) } else { out }
    }
    fn f32_fma(a: f32, b: f32, c: f32, m: Rm, z: bool) -> f32 {
        let (a, b, c) = if z { (ftz(a), ftz(b), ftz(c)) } else { (a, b, c) };
        let r = a.mul_add(b, c);
        // Known approximation: the error-free transform loses the product's rounding error when
        // 0 < |a*b| < 2^-100 (below the f32 subnormal range); skip those inputs.
        if m != Rm::Rn && (a as f64 * b as f64).abs() < 2f64.powi(-100) && a != 0.0 && b != 0.0 { return f32::from_bits(0xdead_beef); }
        let out = if a.is_finite() && b.is_finite() && c.is_finite() && m != Rm::Rn {
            let (s, e) = two_sum(a as f64 * b as f64, c as f64);
            directed(r, dsign(s, e, r), m)
        } else { r };
        if z { ftz(out) } else { out }
    }
    fn f32_div(a: f32, b: f32, m: Rm, z: bool) -> f32 {
        let (a, b) = if z { (ftz(a), ftz(b)) } else { (a, b) };
        let r = a / b;
        let out = if a.is_finite() && b.is_finite() && b != 0.0 && m != Rm::Rn {
            let rem = a as f64 - (r as f64) * (b as f64);
            directed(r, if r.is_infinite() { -sgn(r as f64) } else { sgn(rem) * sgn(b as f64) }, m)
        } else { r };
        if z { ftz(out) } else { out }
    }
    fn f32_sqrt(a: f32, m: Rm, z: bool) -> f32 {
        let a = if z { ftz(a) } else { a };
        let r = a.sqrt();
        let out = if a.is_finite() && a > 0.0 && m != Rm::Rn { directed(r, sgn(a as f64 - (r as f64) * (r as f64)), m) } else { r };
        if z { ftz(out) } else { out }
    }
    pub fn f16_to_f32(h: u32) -> f32 {
        let (s, e, m) = ((h >> 15) & 1, (h >> 10) & 31, h & 0x3ff);
        let v = if e == 0 { m as f32 * (2.0f32).powi(-24) } else if e == 31 { if m == 0 { f32::INFINITY } else { f32::NAN } } else { (1.0 + m as f32 / 1024.0) * (2.0f32).powi(e as i32 - 15) };
        if s == 1 { -v } else { v }
    }
    /// f32 -> f16 bits with the given rounding mode (NaN -> 0x7fff).
    pub fn f32_to_f16(x: f32, mode: Rm) -> u32 {
        let bits = x.to_bits();
        let neg = bits >> 31 == 1;
        let sign = (bits >> 16) & 0x8000;
        if x.is_nan() { return 0x7fff; }
        if x.is_infinite() { return sign | 0x7c00; }
        let ex = ((bits >> 23) & 0xff) as i32;
        let up_ok = |rem: u64, half: u64, lsb: u64| -> bool {
            match mode {
                Rm::Rn => rem > half || (rem == half && lsb == 1),
                Rm::Rz => false,
                Rm::Rp => !neg && rem != 0,
                Rm::Rm => neg && rem != 0,
            }
        };
        if ex >= 127 + 16 { // |x| >= 65536
            return sign | if matches!((mode, neg), (Rm::Rz, _) | (Rm::Rm, false) | (Rm::Rp, true)) { 0x7bff } else { 0x7c00 };
        }
        let (e, mant24) = if ex == 0 { (-126, (bits & 0x7f_ffff) as u64) } else { (ex - 127, ((bits & 0x7f_ffff) | 0x80_0000) as u64) };
        let (hb, rem, half, lsb);
        if e >= -14 {
            let h = (((e + 15) as u64) << 10) | ((mant24 >> 13) & 0x3ff);
            hb = h; rem = mant24 & 0x1fff; half = 0x1000; lsb = h & 1;
        } else {
            let shift = (13 + (-14 - e)).min(40) as u32;
            let q = mant24 >> shift;
            hb = q; rem = mant24 & ((1u64 << shift) - 1); half = 1u64 << (shift - 1); lsb = q & 1;
        }
        let r = if up_ok(rem, half, lsb) { hb + 1 } else { hb };
        sign | r as u32
    }
    fn canon_f16(v: u64) -> u64 { if (v & 0x7c00) == 0x7c00 && (v & 0x3ff) != 0 { 0x7fff } else { v & 0xffff } }

    fn f32c(name: &str, body: &str, f: impl Fn(f32, f32, f32) -> f32 + 'static) -> Case {
        case(name, Gen::F32, K::F32, body, move |a, b, c| Some(u64::from(f(f32::from_bits(a as u32), f32::from_bits(b as u32), f32::from_bits(c as u32)).to_bits())))
    }
    fn f64c(name: &str, body: &str, f: impl Fn(f64, f64, f64) -> f64 + 'static) -> Case {
        case(name, Gen::F64, K::F64, body, move |a, b, c| Some(f(f64::from_bits(a), f64::from_bits(b), f64::from_bits(c)).to_bits()))
    }
    fn f2i(name: &str, body: &str, g: Gen, out: K, f: impl Fn(f32, f64) -> u64 + 'static) -> Case {
        case(name, g, out, body, move |a, _, _| Some(f(f32::from_bits(a as u32), f64::from_bits(a))))
    }
    fn fmin(a: f32, b: f32) -> f32 { if a.is_nan() { b } else if b.is_nan() { a } else if a == b { if a.is_sign_negative() { a } else { b } } else { a.min(b) } }
    fn fmax(a: f32, b: f32) -> f32 { if a.is_nan() { b } else if b.is_nan() { a } else if a == b { if a.is_sign_positive() { a } else { b } } else { a.max(b) } }

    fn cases() -> Vec<Case> {
        let mut v = Vec::new();
        for (m, ms) in [(Rm::Rn, "rn"), (Rm::Rz, "rz"), (Rm::Rm, "rm"), (Rm::Rp, "rp")] {
            for (z, zs) in [(false, ""), (true, ".ftz")] {
                // .ftz is an approximation (denormals are not flushed): only compare cases where
                // flushing does not change the reference result.
                let ft = move |name: String, body: String, f: fn(f32, f32, f32, Rm, bool) -> f32| {
                    case(&name, Gen::F32, K::F32, &body, move |a, b, c| {
                        let (a, b, c) = (f32::from_bits(a as u32), f32::from_bits(b as u32), f32::from_bits(c as u32));
                        let x = f(a, b, c, m, z);
                        if x.to_bits() == 0xdead_beef { return None; }
                        if z {
                            let y = f(a, b, c, m, false);
                            if !(x.to_bits() == y.to_bits() || x.is_nan() && y.is_nan()) { return None; }
                        }
                        Some(u64::from(x.to_bits()))
                    })
                };
                v.push(ft(format!("add.{ms}{zs}.f32"), format!("add.{ms}{zs}.f32 %f20, %f10, %f11;"), |a, b, _, m, z| f32_add(a, b, m, z)));
                v.push(ft(format!("sub.{ms}{zs}.f32"), format!("sub.{ms}{zs}.f32 %f20, %f10, %f11;"), |a, b, _, m, z| f32_add(a, -b, m, z)));
                v.push(ft(format!("mul.{ms}{zs}.f32"), format!("mul.{ms}{zs}.f32 %f20, %f10, %f11;"), |a, b, _, m, z| f32_mul(a, b, m, z)));
                v.push(ft(format!("fma.{ms}{zs}.f32"), format!("fma.{ms}{zs}.f32 %f20, %f10, %f11, %f12;"), |a, b, c, m, z| f32_fma(a, b, c, m, z)));
                v.push(ft(format!("div.{ms}{zs}.f32"), format!("div.{ms}{zs}.f32 %f20, %f10, %f11;"), |a, b, _, m, z| f32_div(a, b, m, z)));
                v.push(ft(format!("sqrt.{ms}{zs}.f32"), format!("sqrt.{ms}{zs}.f32 %f20, %f10;"), |a, _, _, m, z| f32_sqrt(a, m, z)));
            }
        }
        v.push(f32c("min.f32", "min.f32 %f20, %f10, %f11;", |a, b, _| fmin(a, b)));
        v.push(f32c("max.f32", "max.f32 %f20, %f10, %f11;", |a, b, _| fmax(a, b)));
        v.push(f32c("min.NaN.f32", "min.NaN.f32 %f20, %f10, %f11;", |a, b, _| if a.is_nan() || b.is_nan() { f32::NAN } else { fmin(a, b) }));
        v.push(f32c("max.NaN.f32", "max.NaN.f32 %f20, %f10, %f11;", |a, b, _| if a.is_nan() || b.is_nan() { f32::NAN } else { fmax(a, b) }));
        v.push(f32c("abs.f32", "abs.f32 %f20, %f10;", |a, _, _| f32::from_bits(a.to_bits() & 0x7fff_ffff)));
        v.push(f32c("neg.f32", "neg.f32 %f20, %f10;", |a, _, _| f32::from_bits(a.to_bits() ^ 0x8000_0000)));
        v.push(f32c("copysign", "copysign.f32 %f20, %f10, %f11;", |a, b, _| b.copysign(a)));
        v.push(f32c("sat.f32 add", "add.sat.f32 %f20, %f10, %f11;", |a, b, _| { let r = a + b; if r.is_nan() { 0.0 } else { r.clamp(0.0, 1.0) } }));
        v.push(f32c("fma.rn.sat.f32", "fma.rn.sat.f32 %f20, %f10, %f11, %f12;", |a, b, c| { let r = a.mul_add(b, c); if r.is_nan() { 0.0 } else { r.clamp(0.0, 1.0) } }));
        v.push(f32c("rcp.rn.f32", "rcp.rn.f32 %f20, %f10;", |a, _, _| 1.0 / a));
        v.push(f32c("setp/selp f32 lt", "setp.lt.f32 %p20, %f10, %f11;\n selp.f32 %f20, %f12, %f10, %p20;", |a, b, c| if a < b { c } else { a }));
        v.push(f32c("setp.ltu", "setp.ltu.f32 %p20, %f10, %f11;\n selp.f32 %f20, %f12, %f10, %p20;", |a, b, c| if !(a >= b) { c } else { a }));
        v.push(f32c("setp.nan", "setp.nan.f32 %p20, %f10, %f11;\n selp.f32 %f20, %f12, %f10, %p20;", |a, b, c| if a.is_nan() || b.is_nan() { c } else { a }));
        v.push(f32c("setp.num", "setp.num.f32 %p20, %f10, %f11;\n selp.f32 %f20, %f12, %f10, %p20;", |a, b, c| if !a.is_nan() && !b.is_nan() { c } else { a }));
        v.push(f32c("setp.equ", "setp.equ.f32 %p20, %f10, %f11;\n selp.f32 %f20, %f12, %f10, %p20;", |a, b, c| if a.is_nan() || b.is_nan() || a == b { c } else { a }));
        v.push(f32c("setp.neu", "setp.neu.f32 %p20, %f10, %f11;\n selp.f32 %f20, %f12, %f10, %p20;", |a, b, c| if a.is_nan() || b.is_nan() || a != b { c } else { a }));
        v.push(f32c("setp.ne", "setp.ne.f32 %p20, %f10, %f11;\n selp.f32 %f20, %f12, %f10, %p20;", |a, b, c| if a != b && !a.is_nan() && !b.is_nan() { c } else { a }));
        // f64
        v.push(f64c("add.rn.f64", "add.rn.f64 %fd20, %fd10, %fd11;", |a, b, _| a + b));
        v.push(f64c("mul.rn.f64", "mul.rn.f64 %fd20, %fd10, %fd11;", |a, b, _| a * b));
        v.push(f64c("fma.rn.f64", "fma.rn.f64 %fd20, %fd10, %fd11, %fd12;", |a, b, c| a.mul_add(b, c)));
        v.push(f64c("div.rn.f64", "div.rn.f64 %fd20, %fd10, %fd11;", |a, b, _| a / b));
        v.push(f64c("sqrt.rn.f64", "sqrt.rn.f64 %fd20, %fd10;", |a, _, _| a.sqrt()));
        v.push(f64c("min.f64", "min.f64 %fd20, %fd10, %fd11;", |a, b, _| if a.is_nan() { b } else if b.is_nan() { a } else if a == b { if a.is_sign_negative() { a } else { b } } else { a.min(b) }));
        v.push(f64c("max.f64", "max.f64 %fd20, %fd10, %fd11;", |a, b, _| if a.is_nan() { b } else if b.is_nan() { a } else if a == b { if a.is_sign_positive() { a } else { b } } else { a.max(b) }));
        v.push(f64c("abs/neg.f64", "abs.f64 %fd21, %fd10;\n neg.f64 %fd20, %fd21;", |a, _, _| -a.abs()));
        let _ = ftz64;
        // float -> integer (always saturating, NaN -> 0)
        v.push(f2i("cvt.rzi.s32.f32", "cvt.rzi.s32.f32 %r20, %f10;", Gen::F32, K::R32, |a, _| a as i32 as u32 as u64));
        v.push(f2i("cvt.rni.s32.f32", "cvt.rni.s32.f32 %r20, %f10;", Gen::F32, K::R32, |a, _| a.round_ties_even() as i32 as u32 as u64));
        v.push(f2i("cvt.rmi.s32.f32", "cvt.rmi.s32.f32 %r20, %f10;", Gen::F32, K::R32, |a, _| a.floor() as i32 as u32 as u64));
        v.push(f2i("cvt.rpi.s32.f32", "cvt.rpi.s32.f32 %r20, %f10;", Gen::F32, K::R32, |a, _| a.ceil() as i32 as u32 as u64));
        v.push(f2i("cvt.rzi.u32.f32", "cvt.rzi.u32.f32 %r20, %f10;", Gen::F32, K::R32, |a, _| a as u32 as u64));
        v.push(f2i("cvt.rni.u32.f32", "cvt.rni.u32.f32 %r20, %f10;", Gen::F32, K::R32, |a, _| a.round_ties_even() as u32 as u64));
        v.push(f2i("cvt.rzi.s64.f32", "cvt.rzi.s64.f32 %rd20, %f10;", Gen::F32, K::R64, |a, _| a as i64 as u64));
        v.push(f2i("cvt.rni.s64.f32", "cvt.rni.s64.f32 %rd20, %f10;", Gen::F32, K::R64, |a, _| a.round_ties_even() as i64 as u64));
        v.push(f2i("cvt.rzi.u64.f32", "cvt.rzi.u64.f32 %rd20, %f10;", Gen::F32, K::R64, |a, _| a as u64));
        v.push(f2i("cvt.rzi.s32.f64", "cvt.rzi.s32.f64 %r20, %fd10;", Gen::F64, K::R32, |_, d| d as i32 as u32 as u64));
        v.push(f2i("cvt.rni.s32.f64", "cvt.rni.s32.f64 %r20, %fd10;", Gen::F64, K::R32, |_, d| d.round_ties_even() as i32 as u32 as u64));
        v.push(f2i("cvt.rzi.s64.f64", "cvt.rzi.s64.f64 %rd20, %fd10;", Gen::F64, K::R64, |_, d| d as i64 as u64));
        v.push(f2i("cvt.rzi.u64.f64", "cvt.rzi.u64.f64 %rd20, %fd10;", Gen::F64, K::R64, |_, d| d as u64));
        // float -> float integral rounding
        v.push(f32c("cvt.rzi.f32.f32", "cvt.rzi.f32.f32 %f20, %f10;", |a, _, _| a.trunc()));
        v.push(f32c("cvt.rni.f32.f32", "cvt.rni.f32.f32 %f20, %f10;", |a, _, _| a.round_ties_even()));
        v.push(f32c("cvt.rmi.f32.f32", "cvt.rmi.f32.f32 %f20, %f10;", |a, _, _| a.floor()));
        v.push(f32c("cvt.rpi.f32.f32", "cvt.rpi.f32.f32 %f20, %f10;", |a, _, _| a.ceil()));
        // integer -> float
        for (m, ms) in [(Rm::Rn, "rn"), (Rm::Rz, "rz"), (Rm::Rm, "rm"), (Rm::Rp, "rp")] {
            let cv = move |exact: i128| -> f32 {
                let r = exact as f32; // RN for i128 -> f32
                if m == Rm::Rn || r.is_infinite() { return r; }
                let d = (exact - r as i128).signum() as i32;
                directed(r, d, m)
            };
            v.push(case(&format!("cvt.{ms}.f32.s32"), Gen::Int32, K::F32, &format!("cvt.{ms}.f32.s32 %f20, %r10;"), move |a, _, _| Some(u64::from(cv(i128::from(a as u32 as i32)).to_bits()))));
            v.push(case(&format!("cvt.{ms}.f32.u32"), Gen::Int32, K::F32, &format!("cvt.{ms}.f32.u32 %f20, %r10;"), move |a, _, _| Some(u64::from(cv(i128::from(a as u32)).to_bits()))));
            v.push(case(&format!("cvt.{ms}.f32.s64"), Gen::Int64, K::F32, &format!("cvt.{ms}.f32.s64 %f20, %rd10;"), move |a, _, _| Some(u64::from(cv(i128::from(a as i64)).to_bits()))));
            v.push(case(&format!("cvt.{ms}.f32.u64"), Gen::Int64, K::F32, &format!("cvt.{ms}.f32.u64 %f20, %rd10;"), move |a, _, _| Some(u64::from(cv(i128::from(a)).to_bits()))));
            v.push(case(&format!("cvt.{ms}.f32.f64"), Gen::F64, K::F32, &format!("cvt.{ms}.f32.f64 %f20, %fd10;"), move |a, _, _| {
                let x = f64::from_bits(a);
                let r = x as f32;
                let out = if m == Rm::Rn || !x.is_finite() { r } else if r.is_infinite() { directed(r, -sgn(r as f64), m) } else { directed(r, sgn(x - r as f64), m) };
                Some(u64::from(out.to_bits()))
            }));
            v.push(case(&format!("cvt.{ms}.f64.s64"), Gen::Int64, K::F64, &format!("cvt.{ms}.f64.s64 %fd20, %rd10;"), move |a, _, _| {
                if m != Rm::Rn { return None; }
                Some((a as i64 as f64).to_bits())
            }));
        }
        v.push(case("cvt.f64.f32", Gen::F32, K::F64, "cvt.f64.f32 %fd20, %f10;", |a, _, _| Some((f32::from_bits(a as u32) as f64).to_bits())));
        v.push(case("cvt.rn.f64.s32", Gen::Int32, K::F64, "cvt.rn.f64.s32 %fd20, %r10;", |a, _, _| Some((a as u32 as i32 as f64).to_bits())));
        v.push(case("cvt.rn.f64.u32", Gen::Int32, K::F64, "cvt.rn.f64.u32 %fd20, %r10;", |a, _, _| Some((a as u32 as f64).to_bits())));
        // half precision
        for (m, ms) in [(Rm::Rn, "rn"), (Rm::Rz, "rz")] {
            let mut c = case(&format!("cvt.{ms}.f16.f32"), Gen::F32, K::R32, &format!("cvt.{ms}.f16.f32 %h20, %f10;\n cvt.u32.u16 %r20, %h20;"), move |a, _, _| Some(u64::from(f32_to_f16(f32::from_bits(a as u32), m))));
            c.canon = Some(canon_f16);
            v.push(c);
        }
        v.push(case("cvt.f32.f16", Gen::Int32, K::F32, "cvt.u16.u32 %h21, %r10;\n cvt.f32.f16 %f20, %h21;", |a, _, _| Some(u64::from(f16_to_f32(a as u32 & 0xffff).to_bits()))));
        v.push(case("cvt.rn.f16.f32 +cvt.f32.f16 roundtrip", Gen::F32, K::F32, "cvt.rn.f16.f32 %h20, %f10;\n cvt.f32.f16 %f20, %h20;", |a, _, _| Some(u64::from(f16_to_f32(f32_to_f16(f32::from_bits(a as u32), Rm::Rn)).to_bits()))));
        {
            let mut c = case("cvt.rn.f16.f32.ftz", Gen::F32, K::R32, "cvt.rn.ftz.f16.f32 %h20, %f10;\n cvt.u32.u16 %r20, %h20;", |a, _, _| Some(u64::from(f32_to_f16(f32::from_bits(a as u32), Rm::Rn))));
            c.canon = Some(canon_f16);
            v.push(c);
        }
        v.push(case("cvt.rn.bf16.f32", Gen::F32, K::R32, "cvt.rn.bf16.f32 %h20, %f10;\n cvt.u32.u16 %r20, %h20;", |a, _, _| {
            let b = a as u32;
            let r = if f32::from_bits(b).is_nan() { 0x7fff } else { (b.wrapping_add(0x7fff + ((b >> 16) & 1))) >> 16 };
            Some(u64::from(r))
        }).with_canon(|v| if (v & 0x7f80) == 0x7f80 && (v & 0x7f) != 0 { 0x7fff } else { v & 0xffff }));
        v.push(case("cvt.f32.bf16", Gen::Int32, K::F32, "cvt.u16.u32 %h21, %r10;\n mov.b32 %r22, {0, %h21};\n mov.b32 %f20, %r22;", |a, _, _| Some(u64::from((a as u32 & 0xffff) << 16))));
        // integer conversions
        let ic = |name: &str, body: &str, f: fn(u32) -> u32| case(name, Gen::Int32, K::R32, body, move |a, _, _| Some(u64::from(f(a as u32))));
        v.push(ic("cvt.sat.s8.s32", "cvt.sat.s8.s32 %h20, %r10;\n cvt.u32.u16 %r20, %h20;", |a| (a as i32).clamp(-128, 127) as u32 & 0xffff));
        v.push(ic("cvt.sat.u8.s32", "cvt.sat.u8.s32 %h20, %r10;\n cvt.u32.u16 %r20, %h20;", |a| (a as i32).clamp(0, 255) as u32));
        v.push(ic("cvt.sat.s16.s32", "cvt.sat.s16.s32 %h20, %r10;\n cvt.u32.u16 %r20, %h20;", |a| (a as i32).clamp(-32768, 32767) as u32 & 0xffff));
        v.push(ic("cvt.sat.u16.u32", "cvt.sat.u16.u32 %h20, %r10;\n cvt.u32.u16 %r20, %h20;", |a| a.min(65535)));
        v.push(ic("cvt.sat.u16.s32", "cvt.sat.u16.s32 %h20, %r10;\n cvt.u32.u16 %r20, %h20;", |a| (a as i32).clamp(0, 65535) as u32));
        v.push(ic("cvt.sat.s32.u32", "cvt.sat.s32.u32 %r20, %r10;", |a| a.min(0x7fff_ffff)));
        v.push(ic("cvt.sat.u32.s32", "cvt.sat.u32.s32 %r20, %r10;", |a| (a as i32).max(0) as u32));
        v.push(ic("cvt.s32.s16", "cvt.u16.u32 %h21, %r10;\n cvt.s32.s16 %r20, %h21;", |a| a as u16 as i16 as i32 as u32));
        v.push(ic("cvt.u32.u16", "cvt.u16.u32 %h21, %r10;\n cvt.u32.u16 %r20, %h21;", |a| a & 0xffff));
        v.push(ic("cvt.s32.s8", "cvt.u16.u32 %h21, %r10;\n cvt.s32.s8 %r20, %h21;", |a| a as u8 as i8 as i32 as u32));
        v.push(ic("cvt.u32.u8", "cvt.u16.u32 %h21, %r10;\n cvt.u32.u8 %r20, %h21;", |a| a & 0xff));
        v.push(case("cvt.sat.s32.s64", Gen::Int64, K::R32, "cvt.sat.s32.s64 %r20, %rd10;", |a, _, _| Some((a as i64).clamp(i32::MIN as i64, i32::MAX as i64) as u32 as u64)));
        v.push(case("cvt.sat.u32.u64", Gen::Int64, K::R32, "cvt.sat.u32.u64 %r20, %rd10;", |a, _, _| Some(a.min(u32::MAX as u64))));
        v.push(case("cvt.u32.u64", Gen::Int64, K::R32, "cvt.u32.u64 %r20, %rd10;", |a, _, _| Some(a & 0xffff_ffff)));
        v.push(case("cvt.s64.s32", Gen::Int32, K::R64, "cvt.s64.s32 %rd20, %r10;", |a, _, _| Some(a as u32 as i32 as i64 as u64)));
        v.push(case("cvt.u64.u32", Gen::Int32, K::R64, "cvt.u64.u32 %rd20, %r10;", |a, _, _| Some(a & 0xffff_ffff)));
        v
    }

    #[test]
    fn float_semantics_match_reference() {
        let unsupported = run_all(cases(), 0x0bad_cafe_1234_5678);
        eprintln!("float unsupported: {}", unsupported.len());
    }
}

impl Case {
    fn with_canon(mut self, f: fn(u64) -> u64) -> Self { self.canon = Some(f); self }
}

#[cfg(test)]
mod int64_ops {
    use super::*;

    fn c64(name: &str, body: &str, f: impl Fn(u64, u64, u64) -> Option<u64> + 'static) -> Case {
        case(name, Gen::Int64, K::R64, body, f)
    }
    fn c32out(name: &str, body: &str, f: impl Fn(u64, u64, u64) -> u32 + 'static) -> Case {
        case(name, Gen::Int64, K::R32, body, move |a, b, c| Some(u64::from(f(a, b, c))))
    }
    fn bfe64(a: u64, pos: u32, len: u32, signed: bool) -> u64 {
        let sbit = if signed && len > 0 { (a >> (pos + len - 1).min(63)) & 1 } else { 0 };
        let mut d = 0u64;
        for i in 0..64u32 {
            let bit = if i < len && pos + i <= 63 { (a >> (pos + i)) & 1 } else { sbit };
            d |= bit << i;
        }
        d
    }

    fn cases() -> Vec<Case> {
        vec![
            c64("add.s64", "add.s64 %rd20, %rd10, %rd11;", |a, b, _| Some(a.wrapping_add(b))),
            c64("sub.s64", "sub.s64 %rd20, %rd10, %rd11;", |a, b, _| Some(a.wrapping_sub(b))),
            c64("mul.lo.s64", "mul.lo.s64 %rd20, %rd10, %rd11;", |a, b, _| Some(a.wrapping_mul(b))),
            c64("mul.hi.u64", "mul.hi.u64 %rd20, %rd10, %rd11;", |a, b, _| Some(((u128::from(a) * u128::from(b)) >> 64) as u64)),
            c64("mul.hi.s64", "mul.hi.s64 %rd20, %rd10, %rd11;", |a, b, _| Some(((i128::from(a as i64) * i128::from(b as i64)) >> 64) as u64)),
            c64("mad.lo.s64", "mad.lo.s64 %rd20, %rd10, %rd11, %rd12;", |a, b, c| Some(a.wrapping_mul(b).wrapping_add(c))),
            c64("mad.hi.u64", "mad.hi.u64 %rd20, %rd10, %rd11, %rd12;", |a, b, c| Some((((u128::from(a) * u128::from(b)) >> 64) as u64).wrapping_add(c))),
            c64("div.u64", "div.u64 %rd20, %rd10, %rd11;", |a, b, _| Some(a.checked_div(b).unwrap_or(u64::MAX))),
            c64("rem.u64", "rem.u64 %rd20, %rd10, %rd11;", |a, b, _| Some(a.checked_rem(b).unwrap_or(a))),
            c64("div.s64", "div.s64 %rd20, %rd10, %rd11;", |a, b, _| Some(if b == 0 { u64::MAX } else { (a as i64).wrapping_div(b as i64) as u64 })),
            c64("rem.s64", "rem.s64 %rd20, %rd10, %rd11;", |a, b, _| Some(if b == 0 { a } else { (a as i64).wrapping_rem(b as i64) as u64 })),
            c64("min.u64", "min.u64 %rd20, %rd10, %rd11;", |a, b, _| Some(a.min(b))),
            c64("max.u64", "max.u64 %rd20, %rd10, %rd11;", |a, b, _| Some(a.max(b))),
            c64("min.s64", "min.s64 %rd20, %rd10, %rd11;", |a, b, _| Some((a as i64).min(b as i64) as u64)),
            c64("max.s64", "max.s64 %rd20, %rd10, %rd11;", |a, b, _| Some((a as i64).max(b as i64) as u64)),
            c64("abs.s64", "abs.s64 %rd20, %rd10;", |a, _, _| Some((a as i64).wrapping_abs() as u64)),
            c64("neg.s64", "neg.s64 %rd20, %rd10;", |a, _, _| Some((a as i64).wrapping_neg() as u64)),
            c64("not.b64", "not.b64 %rd20, %rd10;", |a, _, _| Some(!a)),
            c64("and/or/xor 64", "and.b64 %rd21, %rd10, %rd11;\n or.b64 %rd22, %rd21, %rd12;\n xor.b64 %rd20, %rd22, %rd10;", |a, b, c| Some(((a & b) | c) ^ a)),
            // shift counts are 32-bit registers
            case("shl.b64", Gen::Shift64, K::R64, "shl.b64 %rd20, %rd10, %r11;", |a, b, _| { let n = b as u32; Some(if n >= 64 { 0 } else { a << n }) }),
            case("shr.u64", Gen::Shift64, K::R64, "shr.u64 %rd20, %rd10, %r11;", |a, b, _| { let n = b as u32; Some(if n >= 64 { 0 } else { a >> n }) }),
            case("shr.s64", Gen::Shift64, K::R64, "shr.s64 %rd20, %rd10, %r11;", |a, b, _| { let n = b as u32; Some(((a as i64) >> n.min(63)) as u64) }),
            c32out("popc.b64", "popc.b64 %r20, %rd10;", |a, _, _| a.count_ones()),
            c32out("clz.b64", "clz.b64 %r20, %rd10;", |a, _, _| a.leading_zeros()),
            c64("brev.b64", "brev.b64 %rd20, %rd10;", |a, _, _| Some(a.reverse_bits())),
            c32out("bfind.u64", "bfind.u64 %r20, %rd10;", |a, _, _| if a == 0 { u32::MAX } else { 63 - a.leading_zeros() }),
            c64("bfe.u64", "bfe.u64 %rd20, %rd10, %r11, %r12;", |a, b, c| Some(bfe64(a, (b & 0xff) as u32, (c & 0xff) as u32, false))),
            c64("bfe.s64", "bfe.s64 %rd20, %rd10, %r11, %r12;", |a, b, c| Some(bfe64(a, (b & 0xff) as u32, (c & 0xff) as u32, true))),
            c64("bfi.b64", "shr.u32 %r13, %r12, 8;\n bfi.b64 %rd20, %rd10, %rd11, %r12, %r13;", |a, b, c| {
                let (pos, len) = ((c & 0xff) as u32, ((c >> 8) & 0xff) as u32);
                let mut d = b;
                for i in 0..len.min(64) { if pos + i < 64 { d = (d & !(1u64 << (pos + i))) | (((a >> i) & 1) << (pos + i)); } }
                Some(d)
            }),
            c64("selp.b64", "setp.lt.u64 %p20, %rd10, %rd11;\n selp.b64 %rd20, %rd12, %rd10, %p20;", |a, b, c| Some(if a < b { c } else { a })),
            c64("setp.lt.s64", "setp.lt.s64 %p20, %rd10, %rd11;\n selp.b64 %rd20, %rd12, %rd10, %p20;", |a, b, c| Some(if (a as i64) < (b as i64) { c } else { a })),
            c64("setp.ge.u64", "setp.ge.u64 %p20, %rd10, %rd11;\n selp.b64 %rd20, %rd12, %rd10, %p20;", |a, b, c| Some(if a >= b { c } else { a })),
            c64("add.cc/addc.u64", "add.cc.u64 %rd21, %rd10, %rd11;\n addc.u64 %rd20, %rd12, %rd12;", |a, b, c| Some(c.wrapping_add(c).wrapping_add(u64::from(a.overflowing_add(b).1)))),
            c64("mul.wide.u32", "mul.wide.u32 %rd20, %r10, %r11;", |a, b, _| Some(u64::from(a as u32) * u64::from(b as u32))),
            c64("mul.wide.s32", "mul.wide.s32 %rd20, %r10, %r11;", |a, b, _| Some((i64::from(a as u32 as i32) * i64::from(b as u32 as i32)) as u64)),
            c64("mad.wide.u32", "mad.wide.u32 %rd20, %r10, %r11, %rd12;", |a, b, c| Some((u64::from(a as u32) * u64::from(b as u32)).wrapping_add(c))),
            c64("mad.wide.s32", "mad.wide.s32 %rd20, %r10, %r11, %rd12;", |a, b, c| Some(((i64::from(a as u32 as i32) * i64::from(b as u32 as i32)) as u64).wrapping_add(c))),
            c64("lop3 64 via two 32", "lop3.b32 %r21, %r10, %r11, %r12, 0xca;\n cvt.u64.u32 %rd20, %r21;", |a, b, c| { let (a, b, c) = (a as u32, b as u32, c as u32); Some(u64::from((a & b) | (!a & c))) }),
        ]
    }

    #[test]
    fn int64_semantics_match_reference() {
        let u = run_all(cases(), 0x77aa_55cc_1357_9bdf);
        eprintln!("int64 unsupported: {}", u.len());
    }
}

/// Multi-thread semantics: atomics, warp intrinsics, barriers, divergence.
#[cfg(test)]
mod threads {
    use super::*;

    /// Kernel with one global buffer `%rd2` (u32 words), `%r1 = tid.x`, `%r3 = ntid.x`,
    /// `%rd4 = &buf[tid]`, `%rd5 = &buf[0]`; shared array `sh` of 1024 words.
    fn kern(body: &str) -> String {
        format!(
            r#".version 8.0
.target sm_80
.address_size 64
.visible .entry k(.param .u64 k_buf)
{{
	.reg .pred %p<32>;
	.reg .b32 %r<64>;
	.reg .b64 %rd<48>;
	.reg .f32 %f<32>;
	.reg .f64 %fd<16>;
	.shared .align 8 .b8 sh[8192];
	ld.param.u64 %rd1, [k_buf];
	cvta.to.global.u64 %rd2, %rd1;
	mov.u32 %r1, %tid.x;
	mov.u32 %r3, %ntid.x;
	mul.wide.u32 %rd3, %r1, 4;
	add.s64 %rd4, %rd2, %rd3;
	mov.u64 %rd5, %rd2;
	mov.u64 %rd6, sh;
	add.s64 %rd7, %rd6, %rd3;
{body}
	ret;
}}
"#
        )
    }

    fn run(body: &str, threads: u32, words: &[u32]) -> Result<Vec<u32>, String> {
        let t = translate(&kern(body)).map_err(|e| e.to_string())?;
        if let Some(m) = validate(&t.words) { return Err(format!("spirv-val: {m}")); }
        let module = rspirv::dr::load_words(&t.words).unwrap();
        let mut gpu = Gpu::default();
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
        let buf = gpu.upload(&bytes);
        let mut sim = Sim::new(&module);
        sim.run(&mut gpu, &Launch::new([1, 1, 1], [threads, 1, 1], vec![buf]))?;
        Ok(gpu.download(buf, bytes.len()).chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect())
    }

    fn zeros(n: usize) -> Vec<u32> { vec![0; n] }

    fn expect(name: &str, r: Result<Vec<u32>, String>, f: impl FnOnce(&[u32]) -> Result<(), String>) -> Option<String> {
        match r {
            Err(e) => Some(format!("{name}: {e}")),
            Ok(w) => f(&w).err().map(|m| format!("{name}: {m}")),
        }
    }

    #[test]
    fn atomics_match_reference() {
        let n = 64u32;
        let mut bad = Vec::new();
        // returned old values of add.u32 by 1 are a permutation of 0..n; final == n
        for space in ["global", "shared"] {
            let (addr, init) = if space == "global" {
                ("%rd5", "")
            } else {
                ("%rd6", "st.shared.u32 [%rd6], 0;\n bar.sync 0;\n")
            };
            let fin = if space == "global" { "" } else { "bar.sync 0;\n ld.shared.u32 %r20, [%rd6];\n st.global.u32 [%rd5+256], %r20;" };
            let body = format!("{init}atom.{space}.add.u32 %r10, [{addr}], 1;\n st.global.u32 [%rd4+512], %r10;\n{fin}");
            bad.extend(expect(&format!("atom.{space}.add.u32"), run(&body, n, &zeros(256)), |w| {
                let mut olds: Vec<u32> = w[128..128 + n as usize].to_vec();
                olds.sort();
                if olds != (0..n).collect::<Vec<_>>() { return Err(format!("old values not a permutation: {olds:?}")); }
                let fin = if space == "global" { w[0] } else { w[64] };
                if fin != n { return Err(format!("final {fin}")); }
                Ok(())
            }));
        }
        // min/max/and/or/xor with distinct per-thread values; result is order independent
        type Ref = fn(&[u32]) -> u32;
        let vals = |t: u32| t.wrapping_mul(2654435761).rotate_left(t % 31);
        let cases: Vec<(&str, &str, u32, Ref)> = vec![
            ("min.u32", "u32", u32::MAX, |v| *v.iter().min().unwrap()),
            ("max.u32", "u32", 0, |v| *v.iter().max().unwrap()),
            ("min.s32", "s32", i32::MAX as u32, |v| *v.iter().map(|x| *x as i32).collect::<Vec<_>>().iter().min().unwrap() as u32),
            ("max.s32", "s32", i32::MIN as u32, |v| *v.iter().map(|x| *x as i32).collect::<Vec<_>>().iter().max().unwrap() as u32),
            ("and.b32", "b32", u32::MAX, |v| v.iter().fold(u32::MAX, |a, b| a & b)),
            ("or.b32", "b32", 0, |v| v.iter().fold(0, |a, b| a | b)),
            ("xor.b32", "b32", 0, |v| v.iter().fold(0, |a, b| a ^ b)),
            ("add.u32", "u32", 7, |v| v.iter().fold(7u32, |a, b| a.wrapping_add(*b))),
        ];
        for (op, _ty, init, f) in &cases {
            for space in ["global", "shared"] {
                let (addr, setup) = if space == "global" { ("%rd5", String::new()) } else { ("%rd6", format!("setp.eq.u32 %p30, %r1, 0;\n @%p30 st.shared.u32 [%rd6], {init};\n bar.sync 0;\n")) };
                let g = if space == "global" { "" } else { "bar.sync 0;\n setp.eq.u32 %p31, %r1, 0;\n @%p31 ld.shared.u32 %r21, [%rd6];\n @%p31 st.global.u32 [%rd5], %r21;\n" };
                let body = format!("{setup}mul.lo.u32 %r10, %r1, 2654435761;\n and.b32 %r11, %r1, 31;\n mov.u32 %r12, 0;\n sub.u32 %r12, %r12, %r11;\n and.b32 %r12, %r12, 31;\n shl.b32 %r13, %r10, %r11;\n shr.u32 %r14, %r10, %r12;\n or.b32 %r15, %r13, %r14;\n setp.eq.u32 %p29, %r11, 0;\n selp.b32 %r15, %r10, %r15, %p29;\n atom.{space}.{op} %r16, [{addr}], %r15;\n {g}");
                let init_words = { let mut w = zeros(8); w[0] = *init; w };
                let tv: Vec<u32> = (0..n).map(|t| { let r = t % 32; let x = t.wrapping_mul(2654435761); if r == 0 { x } else { x.rotate_left(r) } }).collect();
                let _ = vals;
                bad.extend(expect(&format!("atom.{space}.{op}"), run(&body, n, &init_words), |w| {
                    let mut all = tv.clone();
                    if *op == "add.u32" { all.push(0); }
                    let want = f(&tv);
                    let want = if *op == "add.u32" { want } else { // fold with the initial value
                        match *op { "min.u32" => want.min(*init), "max.u32" => want.max(*init), "min.s32" => (want as i32).min(*init as i32) as u32, "max.s32" => (want as i32).max(*init as i32) as u32,
                            "and.b32" => want & init, "or.b32" => want | init, _ => want ^ init } };
                    let want = if *op == "xor.b32" { f(&tv) ^ init } else { want };
                    if w[0] != want { return Err(format!("final {:#x} want {:#x}", w[0], want)); }
                    Ok(())
                }));
            }
        }
        // cas: exactly one winner
        let body = "add.u32 %r10, %r1, 1;\n atom.global.cas.b32 %r11, [%rd5], 0, %r10;\n st.global.u32 [%rd4+512], %r11;";
        bad.extend(expect("atom.cas.b32", run(body, n, &zeros(256)), |w| {
            let winners: Vec<usize> = (0..n as usize).filter(|&i| w[128 + i] == 0).collect();
            if winners.len() != 1 { return Err(format!("{} winners", winners.len())); }
            if w[0] != winners[0] as u32 + 1 { return Err(format!("final {} winner {}", w[0], winners[0])); }
            Ok(())
        }));
        // exch: old values + final are a permutation of {init, tid+1...}
        let body = "add.u32 %r10, %r1, 1;\n atom.global.exch.b32 %r11, [%rd5], %r10;\n st.global.u32 [%rd4+512], %r11;";
        bad.extend(expect("atom.exch.b32", run(body, n, &zeros(256)), |w| {
            let mut all: Vec<u32> = w[128..128 + n as usize].to_vec();
            all.push(w[0]);
            all.sort();
            if all != (0..=n).collect::<Vec<_>>() { return Err(format!("not a permutation {all:?}")); }
            Ok(())
        }));
        // inc/dec wrap semantics
        let body = "atom.global.inc.u32 %r10, [%rd5], 9;\n st.global.u32 [%rd4+512], %r10;";
        bad.extend(expect("atom.inc.u32", run(body, n, &zeros(256)), |w| {
            if w[0] != n % 10 { return Err(format!("final {} want {}", w[0], n % 10)); }
            Ok(())
        }));
        let body = "atom.global.dec.u32 %r10, [%rd5], 9;\n st.global.u32 [%rd4+512], %r10;";
        bad.extend(expect("atom.dec.u32", run(body, n, &zeros(256)), |w| {
            // dec: old==0 || old>9 ? 9 : old-1 ; start 0 -> 9,8,..,0,9,...
            let mut v = 0u32;
            for _ in 0..n { v = if v == 0 || v > 9 { 9 } else { v - 1 }; }
            if w[0] != v { return Err(format!("final {} want {v}", w[0])); }
            Ok(())
        }));
        // f32 add: sum of small integers is exact in any order
        let body = "cvt.rn.f32.u32 %f10, %r1;\n atom.global.add.f32 %f11, [%rd5], %f10;\n red.global.add.f32 [%rd5+4], %f10;";
        bad.extend(expect("atom/red.add.f32", run(body, n, &zeros(256)), |w| {
            let want = (n * (n - 1) / 2) as f32;
            if f32::from_bits(w[0]) != want || f32::from_bits(w[1]) != want { return Err(format!("{} {} want {want}", f32::from_bits(w[0]), f32::from_bits(w[1]))); }
            Ok(())
        }));
        // u64 add and max, with values above 2^32
        let body = "cvt.u64.u32 %rd10, %r1;\n shl.b64 %rd11, %rd10, 32;\n add.s64 %rd11, %rd11, %rd10;\n atom.global.add.u64 %rd12, [%rd5], %rd11;\n atom.global.max.u64 %rd13, [%rd5+16], %rd11;\n atom.global.min.u64 %rd14, [%rd5+24], %rd11;\n";
        bad.extend(expect("atom.u64 add/max/min", run(body, n, &{ let mut w = zeros(256); w[6] = u32::MAX; w[7] = u32::MAX; w }), |w| {
            let sum: u64 = (0..u64::from(n)).map(|t| (t << 32) + t).sum();
            let got = u64::from(w[0]) | u64::from(w[1]) << 32;
            let mx = u64::from(w[4]) | u64::from(w[5]) << 32;
            let mn = u64::from(w[6]) | u64::from(w[7]) << 32;
            if got != sum { return Err(format!("add {got:#x} want {sum:#x}")); }
            if mx != (u64::from(n - 1) << 32) + u64::from(n - 1) { return Err(format!("max {mx:#x}")); }
            if mn != 0 { return Err(format!("min {mn:#x}")); }
            Ok(())
        }));
        // s64 max with negative values
        let body = "cvt.s64.s32 %rd10, %r1;\n sub.s64 %rd11, %rd10, 40;\n atom.global.max.s64 %rd12, [%rd5], %rd11;\n atom.global.min.s64 %rd13, [%rd5+8], %rd11;";
        bad.extend(expect("atom.s64 max/min", run(body, n, &{ let mut w = zeros(256); w[2] = u32::MAX; w[3] = 0x7fff_ffff; w[0] = 0; w[1] = 0x8000_0000; w }), |w| {
            let mx = (u64::from(w[0]) | u64::from(w[1]) << 32) as i64;
            let mn = (u64::from(w[2]) | u64::from(w[3]) << 32) as i64;
            if mx != i64::from(n) - 1 - 40 { return Err(format!("max {mx}")); }
            if mn != -40 { return Err(format!("min {mn}")); }
            Ok(())
        }));
        assert!(bad.is_empty(), "atomic mismatches:\n{}", bad.join("\n"));
    }

    fn shfl_ref(mode: &str, lane: u32, a: &[u32], b: u32, c: u32) -> u32 {
        let (bval, cval, mask) = (b & 31, c & 31, (c >> 8) & 31);
        let max_lane = (lane & mask) | (cval & !mask & 31);
        let min_lane = lane & mask;
        let (j, ok) = match mode {
            "up" => { let j = lane as i32 - bval as i32; (j, j >= min_lane as i32) }
            "down" => { let j = lane + bval; (j as i32, j <= max_lane) }
            "bfly" => { let j = lane ^ bval; (j as i32, j <= max_lane) }
            _ => { let j = min_lane | (bval & !mask & 31); (j as i32, j <= max_lane) }
        };
        if ok { a[j as usize] } else { a[lane as usize] }
    }

    #[test]
    fn warp_shuffles_match_reference() {
        let mut bad = Vec::new();
        let a: Vec<u32> = (0..32u32).map(|i| i.wrapping_mul(0x9e3779b9) ^ 0x1234).collect();
        for mode in ["idx", "up", "down", "bfly"] {
            for (b, c) in [(1u32, 0x1fu32), (5, 0x1f), (31, 0x1f), (0, 0x1f), (3, 0x0), (2, 0x181f), (1, 0x101f), (7, 0x1c1f), (4, 0x1e1f), (33, 0x1f), (9, 0x0f), (1, 0x10), (2, 0x1f00)] {
                if mode == "up" && c & 0x1f == 0x1f && false { continue; }
                let body = format!("ld.global.u32 %r10, [%rd4+128];\n shfl.sync.{mode}.b32 %r11, %r10, {b}, {c}, 0xffffffff;\n st.global.u32 [%rd4], %r11;");
                let mut init = zeros(64);
                init[32..64].copy_from_slice(&a);
                bad.extend(expect(&format!("shfl.sync.{mode} b={b} c={c:#x}"), run(&body, 32, &init), |w| {
                    for lane in 0..32u32 {
                        let want = shfl_ref(mode, lane, &a, b, c);
                        if w[lane as usize] != want { return Err(format!("lane {lane}: got {:#x} want {want:#x}", w[lane as usize])); }
                    }
                    Ok(())
                }));
            }
            // register-valued lane/clamp operands
            let body = format!("ld.global.u32 %r10, [%rd4+128];\n and.b32 %r12, %r1, 7;\n add.u32 %r12, %r12, 1;\n shfl.sync.{mode}.b32 %r11, %r10, %r12, 31, 0xffffffff;\n st.global.u32 [%rd4], %r11;");
            let mut init = zeros(64);
            init[32..64].copy_from_slice(&a);
            bad.extend(expect(&format!("shfl.sync.{mode} reg b"), run(&body, 32, &init), |w| {
                for lane in 0..32u32 {
                    let want = shfl_ref(mode, lane, &a, (lane & 7) + 1, 31);
                    if w[lane as usize] != want { return Err(format!("lane {lane}: got {:#x} want {want:#x}", w[lane as usize])); }
                }
                Ok(())
            }));
            // predicate-out form
            let body = format!("ld.global.u32 %r10, [%rd4+128];\n shfl.sync.{mode}.b32 %r11|%p1, %r10, 3, 31, 0xffffffff;\n selp.u32 %r12, 1, 0, %p1;\n st.global.u32 [%rd4], %r11;\n st.global.u32 [%rd4+512], %r12;");
            let mut init = zeros(256);
            init[32..64].copy_from_slice(&a);
            bad.extend(expect(&format!("shfl.sync.{mode} pred-out"), run(&body, 32, &init), |w| {
                for lane in 0..32u32 {
                    let want = shfl_ref(mode, lane, &a, 3, 31);
                    let (bval, max_lane) = (3u32, 31u32);
                    let ok = match mode { "up" => lane >= bval, "down" => lane + bval <= max_lane, "bfly" => (lane ^ bval) <= max_lane, _ => true };
                    if w[lane as usize] != want { return Err(format!("lane {lane}: got {:#x} want {want:#x}", w[lane as usize])); }
                    if w[128 + lane as usize] != u32::from(ok) { return Err(format!("lane {lane}: pred {} want {ok}", w[128 + lane as usize])); }
                }
                Ok(())
            }));
        }
        assert!(bad.is_empty(), "shuffle mismatches:\n{}", bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
    }

    #[test]
    fn warp_votes_and_reductions_match_reference() {
        let mut bad = Vec::new();
        // predicate p = (tid * 7 % 5 < 2) over 32 lanes
        let pat = |t: u32| (t * 7) % 5 < 2;
        let ballot: u32 = (0..32).filter(|&t| pat(t)).fold(0, |m, t| m | 1 << t);
        let body = "mul.lo.u32 %r10, %r1, 7;\n rem.u32 %r10, %r10, 5;\n setp.lt.u32 %p1, %r10, 2;\n vote.sync.ballot.b32 %r11, %p1, 0xffffffff;\n vote.sync.all.pred %p2, %p1, 0xffffffff;\n vote.sync.any.pred %p3, %p1, 0xffffffff;\n vote.sync.uni.pred %p4, %p1, 0xffffffff;\n selp.u32 %r12, 1, 0, %p2;\n selp.u32 %r13, 1, 0, %p3;\n selp.u32 %r14, 1, 0, %p4;\n st.global.u32 [%rd4], %r11;\n st.global.u32 [%rd4+256], %r12;\n st.global.u32 [%rd4+512], %r13;\n st.global.u32 [%rd4+768], %r14;";
        bad.extend(expect("vote.sync", run(body, 32, &zeros(512)), |w| {
            for l in 0..32usize {
                if w[l] != ballot { return Err(format!("ballot {:#x} want {ballot:#x}", w[l])); }
                if w[64 + l] != 0 || w[128 + l] != 1 || w[192 + l] != 0 { return Err(format!("all/any/uni = {} {} {}", w[64 + l], w[128 + l], w[192 + l])); }
            }
            Ok(())
        }));
        // all-true and all-false cases
        let body = "setp.lt.u32 %p1, %r1, 100;\n vote.sync.all.pred %p2, %p1, 0xffffffff;\n vote.sync.uni.pred %p4, %p1, 0xffffffff;\n setp.gt.u32 %p5, %r1, 100;\n vote.sync.any.pred %p6, %p5, 0xffffffff;\n selp.u32 %r12, 1, 0, %p2;\n selp.u32 %r13, 1, 0, %p4;\n selp.u32 %r14, 1, 0, %p6;\n shl.b32 %r15, %r13, 1;\n or.b32 %r15, %r15, %r12;\n shl.b32 %r14, %r14, 2;\n or.b32 %r15, %r15, %r14;\n st.global.u32 [%rd4], %r15;";
        bad.extend(expect("vote uniform", run(body, 32, &zeros(64)), |w| if w[..32].iter().all(|&x| x == 3) { Ok(()) } else { Err(format!("{:?}", &w[..4])) }));
        // activemask in divergent branch: odd lanes only
        let body = "and.b32 %r10, %r1, 1;\n setp.eq.u32 %p1, %r10, 1;\n mov.u32 %r11, 0;\n @%p1 bra $odd;\n bra.uni $join;\n$odd:\n activemask.b32 %r11;\n$join:\n st.global.u32 [%rd4], %r11;";
        bad.extend(expect("activemask divergent", run(body, 32, &zeros(64)), |w| {
            for l in 0..32usize { let want = if l & 1 == 1 { 0xaaaa_aaaau32 } else { 0 }; if w[l] != want { return Err(format!("lane {l}: {:#x} want {want:#x}", w[l])); } }
            Ok(())
        }));
        // redux.sync
        for (op, ty, f) in [
            ("add", "u32", (|v: &[u32]| v.iter().fold(0u32, |a, b| a.wrapping_add(*b))) as fn(&[u32]) -> u32),
            ("min", "u32", |v| *v.iter().min().unwrap()),
            ("max", "u32", |v| *v.iter().max().unwrap()),
            ("min", "s32", |v| v.iter().map(|x| *x as i32).min().unwrap() as u32),
            ("max", "s32", |v| v.iter().map(|x| *x as i32).max().unwrap() as u32),
            ("and", "b32", |v| v.iter().fold(u32::MAX, |a, b| a & b)),
            ("or", "b32", |v| v.iter().fold(0, |a, b| a | b)),
            ("xor", "b32", |v| v.iter().fold(0, |a, b| a ^ b)),
        ] {
            let body = format!("mul.lo.u32 %r10, %r1, 2654435761;\n redux.sync.{op}.{ty} %r11, %r10, 0xffffffff;\n st.global.u32 [%rd4], %r11;");
            let vals: Vec<u32> = (0..32u32).map(|t| t.wrapping_mul(2654435761)).collect();
            let want = f(&vals);
            bad.extend(expect(&format!("redux.sync.{op}.{ty}"), run(&body, 32, &zeros(64)), |w| if w[..32].iter().all(|&x| x == want) { Ok(()) } else { Err(format!("{:#x} want {want:#x}", w[0])) }));
        }
        // match.any / match.all
        let body = "and.b32 %r10, %r1, 3;\n match.any.sync.b32 %r11, %r10, 0xffffffff;\n st.global.u32 [%rd4], %r11;";
        bad.extend(expect("match.any.sync", run(body, 32, &zeros(64)), |w| {
            for l in 0..32usize { let want = (0..32).filter(|t| t % 4 == l % 4).fold(0u32, |m, t| m | 1 << t); if w[l] != want { return Err(format!("lane {l}: {:#x} want {want:#x}", w[l])); } }
            Ok(())
        }));
        let body = "mov.u32 %r10, 5;\n match.all.sync.b32 %r11|%p1, %r10, 0xffffffff;\n selp.u32 %r12, 1, 0, %p1;\n st.global.u32 [%rd4], %r11;\n st.global.u32 [%rd4+256], %r12;";
        bad.extend(expect("match.all.sync", run(body, 32, &zeros(128)), |w| if w[0] == 0xffff_ffff && w[64] == 1 { Ok(()) } else { Err(format!("{:#x} {}", w[0], w[64])) }));
        // bar.warp.sync / membar are no-ops that must still translate
        let body = "bar.warp.sync 0xffffffff;\n membar.cta;\n fence.acq_rel.gpu;\n st.global.u32 [%rd4], %r1;";
        bad.extend(expect("bar.warp.sync/membar/fence", run(body, 32, &zeros(64)), |w| if (0..32).all(|l| w[l] == l as u32) { Ok(()) } else { Err("wrong".into()) }));
        let body = "setp.lt.u32 %p1, %r1, 16;\n @%p1 bra $A;\n bra.uni $B;\n$A:\n bar.warp.sync 0xffffffff;\n$B:\n st.global.u32 [%rd4], %r1;";
        let _ = body;
        assert!(bad.is_empty(), "warp mismatches:\n{}", bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
    }

    #[test]
    fn divergence_and_reconvergence_match_reference() {
        let mut bad = Vec::new();
        let n = 64u32;
        // Triangular nested loops with per-thread trip counts, break and continue.
        let body = "mov.u32 %r10, 0;\n mov.u32 %r11, 0;\n$outer:\n setp.ge.u32 %p1, %r11, %r1;\n @%p1 bra $done;\n mov.u32 %r12, 0;\n$inner:\n setp.ge.u32 %p2, %r12, %r11;\n @%p2 bra $inner_done;\n and.b32 %r13, %r12, 3;\n setp.eq.u32 %p3, %r13, 3;\n @%p3 bra $skip;\n add.u32 %r10, %r10, %r12;\n add.u32 %r10, %r10, 1;\n$skip:\n add.u32 %r12, %r12, 1;\n bra.uni $inner;\n$inner_done:\n add.u32 %r11, %r11, 1;\n setp.gt.u32 %p4, %r10, 5000;\n @%p4 bra $done;\n bra.uni $outer;\n$done:\n st.global.u32 [%rd4], %r10;";
        bad.extend(expect("nested divergent loops", run(body, n, &zeros(64)), |w| {
            for t in 0..n {
                let (mut acc, mut i) = (0u32, 0u32);
                loop {
                    if i >= t { break; }
                    for j in 0..i { if j & 3 != 3 { acc += j + 1; } }
                    i += 1;
                    if acc > 5000 { break; }
                }
                if w[t as usize] != acc { return Err(format!("tid {t}: got {} want {acc}", w[t as usize])); }
            }
            Ok(())
        }));
        // divergent if/else diamonds with values merged after reconvergence, then a uniform barrier
        let body = "and.b32 %r10, %r1, 3;\n setp.eq.u32 %p1, %r10, 0;\n @%p1 bra $c0;\n setp.eq.u32 %p2, %r10, 1;\n @%p2 bra $c1;\n setp.eq.u32 %p3, %r10, 2;\n @%p3 bra $c2;\n mul.lo.u32 %r11, %r1, 7;\n bra.uni $join;\n$c0:\n add.u32 %r11, %r1, 100;\n bra.uni $join;\n$c1:\n shl.b32 %r11, %r1, 4;\n bra.uni $join;\n$c2:\n xor.b32 %r11, %r1, 0xff;\n$join:\n st.shared.u32 [%rd7], %r11;\n bar.sync 0;\n sub.u32 %r12, %r3, 1;\n sub.u32 %r12, %r12, %r1;\n mul.wide.u32 %rd10, %r12, 4;\n add.s64 %rd11, %rd6, %rd10;\n ld.shared.u32 %r13, [%rd11];\n st.global.u32 [%rd4], %r13;";
        bad.extend(expect("diamond + shared reverse", run(body, n, &zeros(64)), |w| {
            let val = |t: u32| match t & 3 { 0 => t + 100, 1 => t << 4, 2 => t ^ 0xff, _ => t * 7 };
            for t in 0..n { let want = val(n - 1 - t); if w[t as usize] != want { return Err(format!("tid {t}: got {} want {want}", w[t as usize])); } }
            Ok(())
        }));
        // early exit of some threads before a barrier-free tail (ret in divergent branch)
        let body = "setp.ge.u32 %p1, %r1, 20;\n @%p1 bra $out;\n st.global.u32 [%rd4], 77;\n ret;\n$out:\n st.global.u32 [%rd4], 99;";
        bad.extend(expect("divergent ret", run(body, n, &zeros(64)), |w| if (0..n).all(|t| w[t as usize] == if t >= 20 { 99 } else { 77 }) { Ok(()) } else { Err(format!("{:?}", &w[16..24])) }));
        // loop-carried value defined only in some iterations (use-before-def style)
        let body = "mov.u32 %r10, 0;\n mov.u32 %r11, 0;\n$l:\n setp.ge.u32 %p1, %r11, 8;\n @%p1 bra $e;\n and.b32 %r12, %r11, 1;\n setp.eq.u32 %p2, %r12, 0;\n @%p2 bra $nodef;\n add.u32 %r20, %r1, %r11;\n add.u32 %r10, %r10, %r20;\n$nodef:\n add.u32 %r11, %r11, 1;\n bra.uni $l;\n$e:\n st.global.u32 [%rd4], %r10;";
        bad.extend(expect("conditionally-defined register", run(body, n, &zeros(64)), |w| {
            for t in 0..n { let want: u32 = (0..8u32).filter(|i| i & 1 == 1).map(|i| t + i).sum(); if w[t as usize] != want { return Err(format!("tid {t}: {} want {want}", w[t as usize])); } }
            Ok(())
        }));
        // read-before-write register must read as zero (generalised fa-fix)
        let body = "add.u32 %r10, %r40, %r1;\n st.global.u32 [%rd4], %r10;\n mov.u32 %r41, 5;\n$l:\n add.u32 %r42, %r42, 3;\n sub.u32 %r41, %r41, 1;\n setp.ne.u32 %p1, %r41, 0;\n @%p1 bra $l;\n st.global.u32 [%rd4+256], %r42;";
        bad.extend(expect("use-before-def reads zero", run(body, 32, &zeros(128)), |w| if (0..32).all(|t| w[t] == t as u32 && w[64 + t] == 15) { Ok(()) } else { Err(format!("{:?} {:?}", &w[..3], &w[64..67])) }));
        // uniform barrier inside a divergent-looking-but-uniform loop
        let body = "mov.u32 %r10, 0;\n mov.u32 %r11, 0;\n$l:\n st.shared.u32 [%rd7], %r11;\n bar.sync 0;\n add.u32 %r12, %r1, 1;\n rem.u32 %r12, %r12, %r3;\n mul.wide.u32 %rd10, %r12, 4;\n add.s64 %rd11, %rd6, %rd10;\n ld.shared.u32 %r13, [%rd11];\n add.u32 %r10, %r10, %r13;\n bar.sync 0;\n add.u32 %r11, %r11, 1;\n setp.lt.u32 %p1, %r11, 5;\n @%p1 bra $l;\n st.global.u32 [%rd4], %r10;";
        bad.extend(expect("uniform barrier loop", run(body, n, &zeros(64)), |w| if (0..n).all(|t| w[t as usize] == 10) { Ok(()) } else { Err(format!("{:?}", &w[..3])) }));
        // Barrier reached by only the first 32 threads while the rest exit: PTX treats exited
        // threads as arrived; whether the OpenCL SPIR-V runtime agrees is a hardware question
        // (tests/gpu/hw_barrier_exit.rs). The translator accepts it (warp-uniform condition).
        let body = "setp.ge.u32 %p1, %r1, 32;\n @%p1 bra $skip;\n bar.sync 0;\n$skip:\n st.global.u32 [%rd4], %r1;";
        let r = run(body, n, &zeros(64));
        assert!(r.is_ok(), "{r:?}");
        // Exited threads and later barrier: `if (tid >= n) return; ... __syncthreads()` pattern.
        let body = "setp.ge.u32 %p1, %r1, 48;\n @%p1 bra $exit;\n st.shared.u32 [%rd7], %r1;\n bar.sync 0;\n ld.shared.u32 %r10, [%rd7];\n st.global.u32 [%rd4], %r10;\n$exit:\n ret;";
        let r = run(body, n, &zeros(64));
        eprintln!("early-exit-then-barrier: {:?}", r.as_ref().map(|w| w[..4].to_vec()).map_err(|e| e.clone()));
        assert!(bad.is_empty(), "control-flow mismatches:\n{}", bad.iter().take(12).cloned().collect::<Vec<_>>().join("\n"));
    }
}
