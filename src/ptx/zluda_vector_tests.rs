//! ZLUDA's `test_ptx!` execution vectors (`tests/ptx_corpus/zluda/vectors.tsv`, extracted from
//! ZLUDA's `ptx/src/test/spirv_run/mod.rs` by `tests/ptx_corpus/import_zluda.py --vectors`), run
//! on the SPIR-V simulator and compared byte for byte.
//!
//! Each kernel is launched as ZLUDA's harness does: one block of `block` threads, arguments
//! (input, output) or (output, output) when the vector has no input, on a zero-filled output
//! buffer. The source goes through the same normalisation as the corpus harness
//! (`zluda_source::normalize`: `.version 8.8`, `%`-prefixed bare register names).
//!
//! Every vector must either match or appear in [`EXPECTED_FAILURES`] with its reason; a listed
//! vector that starts to pass also fails the test so the list stays exact.
use super::sim::{Gpu, Launch, Sim};
use super::{Param, prepare_module, translate, translate_prepared_entry, zluda_source};
use std::collections::BTreeMap;

const VECTORS: &str = include_str!("../../tests/ptx_corpus/zluda/vectors.tsv");

/// Vectors that do not match, with the reason. Keep sorted by name.
const EXPECTED_FAILURES: &[(&str, &str)] = &[
    ("const_ident", "module .const initialised with symbol addresses ({ x, y }) is unsupported"),
    ("cp_async", "entry-scope .global variable with an initialiser is unsupported (only .func-scope statics move to module scope)"),
    ("div_ftz", ".ftz is ignored, denormals are not flushed (docs/translator-coverage.md)"),
    ("mul_ftz", ".ftz is ignored, denormals are not flushed (docs/translator-coverage.md)"),
    ("reg_local", "non-standard `reg + imm` value operand (st.u64 [x], temp + 1)"),
    ("sqrt_rn_ftz", ".ftz is ignored, denormals are not flushed (docs/translator-coverage.md)"),
];

/// Kernels that access memory outside their buffers on 64 threads. ZLUDA's harness does not
/// notice because device allocations are rounded up; only the compared prefix is checked here.
/// - `ldmatrix`: threads 32-63 read past the 64-entry `values_g` and write past `values_s`.
/// - `mma_m16n8k16_f32_bf16_bf16_f32_2x`: 64 threads store 32 bytes each into a 1024-byte output.
const TOLERATED_OVERRUNS: &[&str] = &["ldmatrix", "mma_m16n8k16_f32_bf16_bf16_f32_2x"];

struct Vector<'a> {
    name: &'a str,
    ptx: &'a str,
    block: u32,
    input: Option<Vec<u8>>,
    output: Vec<u8>,
}

fn hex(s: &str) -> Vec<u8> {
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

fn vectors() -> Vec<Vector<'static>> {
    VECTORS
        .lines()
        .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split('\t').collect();
            assert_eq!(f.len(), 10, "malformed vectors.tsv row: {l}");
            Vector {
                name: f[0],
                ptx: f[2],
                block: f[3].parse().unwrap(),
                input: (f[6] != "-").then(|| hex(f[6])),
                output: hex(f[9]),
            }
        })
        .collect()
}

/// Runs one vector; `Ok(bytes)` is the output buffer after the launch.
fn run(v: &Vector) -> Result<Vec<u8>, String> {
    let path = format!("{}/tests/ptx_corpus/zluda/{}", env!("CARGO_MANIFEST_DIR"), v.ptx);
    let raw = std::fs::read_to_string(&path).map_err(|e| format!("read {path}: {e}"))?;
    let text = zluda_source::normalize(&raw);
    // cuModuleLoadData translates the whole module; the launch then uses one entry.
    translate(&text).map_err(|e| format!("translate: {e}"))?;
    let prepared = prepare_module(&text).map_err(|e| format!("prepare: {e}"))?;
    let mut t = translate_prepared_entry(&prepared, v.name).map_err(|e| format!("translate entry: {e}"))?;
    let meta = t.kernels.remove(0);
    let module = rspirv::dr::load_words(&t.words).map_err(|e| format!("load SPIR-V: {e:?}"))?;
    let mut gpu = Gpu::default();
    let out = gpu.alloc(v.output.len());
    let inp = v.input.as_ref().map_or(out, |i| gpu.upload(i));
    let ptrs = [inp, out];
    // ZLUDA always passes two pointers; kernels may declare fewer.
    if meta.params.len() > 2 {
        return Err(format!("expected at most 2 parameters, kernel has {:?}", meta.params));
    }
    let mut args = Vec::new();
    for (p, &ptr) in meta.params.iter().zip(&ptrs) {
        match p {
            Param::GlobalPtr | Param::U64 => args.push(ptr),
            other => return Err(format!("unexpected parameter kind {other:?}")),
        }
    }
    let mut sim = Sim::new(&module);
    match sim.run(&mut gpu, &Launch::new([1, 1, 1], [v.block, 1, 1], args)) {
        Err(e) if e.starts_with("memory violations") && TOLERATED_OVERRUNS.contains(&v.name) => {}
        r => r.map_err(|e| format!("simulate: {e}"))?,
    }
    Ok(gpu.download(out, v.output.len()))
}

fn outcome(v: &Vector) -> Result<(), String> {
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(v)));
    match r {
        Ok(Ok(got)) if got == v.output => Ok(()),
        Ok(Ok(got)) => Err(format!("wrong output: got {}, want {}", to_hex(&got), to_hex(&v.output))),
        Ok(Err(e)) => Err(e),
        Err(p) => Err(format!(
            "panic: {}",
            p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default()
        )),
    }
}

fn to_hex(b: &[u8]) -> String {
    let s: String = b.iter().map(|x| format!("{x:02x}")).collect();
    if s.len() > 96 { format!("{}...", &s[..96]) } else { s }
}

#[test]
fn at_least_180_vectors_parse() {
    let v = vectors();
    assert!(v.len() >= 180, "only {} vectors", v.len());
    for x in &v {
        assert!(!x.output.is_empty(), "{}", x.name);
    }
}

#[test]
fn expected_failures_list_is_sorted_and_known() {
    let names: Vec<&str> = vectors().iter().map(|v| v.name).collect();
    for w in EXPECTED_FAILURES.windows(2) {
        assert!(w[0].0 < w[1].0, "EXPECTED_FAILURES not sorted at {}", w[1].0);
    }
    for (n, _) in EXPECTED_FAILURES {
        assert!(names.contains(n), "EXPECTED_FAILURES names unknown vector {n}");
    }
}

#[test]
fn zluda_vectors_match_on_simulator() {
    let expected: BTreeMap<&str, &str> = EXPECTED_FAILURES.iter().copied().collect();
    let hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let all = vectors();
    let mut pass = 0;
    let mut problems = Vec::new();
    for v in &all {
        match (outcome(v), expected.get(v.name)) {
            (Ok(()), None) => pass += 1,
            (Ok(()), Some(_)) => problems.push(format!("{}: passes now; remove it from EXPECTED_FAILURES", v.name)),
            (Err(_), Some(_)) => {}
            (Err(e), None) => problems.push(format!("{}: {e}", v.name)),
        }
    }
    std::panic::set_hook(hook);
    eprintln!("zluda vectors: {pass} pass, {} expected failures, {} total", expected.len(), all.len());
    assert!(problems.is_empty(), "{} unexpected results:\n{}", problems.len(), problems.join("\n"));
}

/// `cargo test zluda_vector_report -- --ignored --nocapture` prints every outcome.
#[test]
#[ignore]
fn zluda_vector_report() {
    std::panic::set_hook(Box::new(|_| {}));
    for v in vectors() {
        match outcome(&v) {
            Ok(()) => println!("PASS\t{}", v.name),
            Err(e) => println!("FAIL\t{}\t{}", v.name, e.replace('\n', " ")),
        }
    }
}

/// Regression tests for translator bugs the vectors exposed, on minimal kernels of our own.
mod regressions {
    use super::super::sim::{Gpu, Launch, Sim};
    use super::super::translate;

    /// Runs entry `k(in, out)` on one thread; returns `out_len` bytes of `out`.
    fn run(body: &str, regs: &str, input: &[u8], out_len: usize) -> Vec<u8> {
        let src = format!(
            ".version 8.8\n.target sm_80\n.address_size 64\n\
             .visible .entry k(.param .u64 k_in, .param .u64 k_out)\n{{\n\
             .reg .b64 %in; .reg .b64 %out;\n{regs}\n\
             ld.param.u64 %in, [k_in];\nld.param.u64 %out, [k_out];\n{body}\nret;\n}}\n"
        );
        let t = translate(&src).unwrap_or_else(|e| panic!("{e}\n{src}"));
        let module = rspirv::dr::load_words(&t.words).unwrap();
        let mut gpu = Gpu::default();
        let inp = gpu.upload(input);
        let out = gpu.alloc(out_len);
        Sim::new(&module).run(&mut gpu, &Launch::new([1, 1, 1], [1, 1, 1], vec![inp, out])).unwrap();
        gpu.download(out, out_len)
    }

    fn words(b: &[u8]) -> Vec<u32> {
        b.chunks(4).map(|c| u32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn le(v: &[u32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    #[test]
    fn sub_cc_leaves_carry_not_borrow_in_the_flag() {
        // CC.CF after sub.cc is the carry of a + ~b + 1: 0 on borrow, 1 otherwise; subc
        // subtracts the complement, so 64-bit subtraction through 32-bit halves still works.
        let regs = ".reg .b32 %r<8>;";
        let body = "ld.global.u32 %r1, [%in]; ld.global.u32 %r2, [%in+4];\n\
                    sub.cc.u32 %r3, %r1, %r2; addc.u32 %r4, 0, 0;\n\
                    sub.cc.u32 %r5, %r2, %r1; addc.u32 %r6, 0, 0;\n\
                    sub.cc.u32 %r3, 0, 1; subc.u32 %r7, 1, 0;\n\
                    st.global.u32 [%out], %r4; st.global.u32 [%out+4], %r6; st.global.u32 [%out+8], %r7;";
        let got = words(&run(body, regs, &le(&[5, 9]), 12));
        assert_eq!(got, [0, 1, 0], "borrow -> CF 0, no borrow -> CF 1, 0x1_00000000 - 1 has hi 0");
    }

    #[test]
    fn cvt_nan_to_64_bit_integer_is_the_minimum_value() {
        // PTX: NaN -> 0 unless the source is .f64 or the destination .s64/.u64, then 1 << (bits - 1).
        let regs = ".reg .f32 %f1; .reg .b16 %h1; .reg .f64 %fd1; .reg .b64 %rd<4>; .reg .b32 %r<3>;";
        let body = "ld.global.f32 %f1, [%in]; ld.global.b16 %h1, [%in+4]; ld.global.f64 %fd1, [%in+8];\n\
                    cvt.rzi.u64.f32 %rd1, %f1; cvt.rzi.s64.f16 %rd2, %h1; cvt.rzi.u32.f32 %r1, %f1;\n\
                    cvt.rzi.s32.f64 %r2, %fd1;\n\
                    st.global.u64 [%out], %rd1; st.global.u64 [%out+8], %rd2; st.global.u32 [%out+16], %r1;\n\
                    st.global.u32 [%out+20], %r2;";
        let mut input = le(&[0x7FC0_0000]);
        input.extend_from_slice(&0x7E00u16.to_le_bytes());
        input.extend_from_slice(&[0, 0]);
        input.extend_from_slice(&f64::NAN.to_bits().to_le_bytes());
        let got = words(&run(body, regs, &input, 24));
        assert_eq!(got, [0, 0x8000_0000, 0, 0x8000_0000, 0, 0x8000_0000]);
    }

    #[test]
    fn set_u32_f16x2_writes_a_16_bit_mask_per_lane() {
        let regs = ".reg .b32 %r<4>;";
        let body = "ld.global.b32 %r1, [%in]; ld.global.b32 %r2, [%in+4];\n\
                    set.gtu.u32.f16x2 %r3, %r1, %r2; st.global.u32 [%out], %r3;";
        // lanes: (2.0 > 1.0, 1.0 > 2.0)
        let got = words(&run(body, regs, &le(&[0x3C00_4000, 0x4000_3C00]), 4));
        assert_eq!(got, [0x0000_FFFF]);
    }

    #[test]
    fn unreachable_blocks_after_bra_are_dropped() {
        let regs = ".reg .b64 %rd<3>;";
        let body = "ld.global.u64 %rd1, [%in]; bra $L_a;\n\
                    add.u64 %rd2, %rd1, 100;\n\
                    $L_dead: add.u64 %rd2, %rd1, 200; bra $L_b;\n\
                    $L_a: add.u64 %rd2, %rd1, 1;\n\
                    $L_b: st.global.u64 [%out], %rd2;";
        assert_eq!(run(body, regs, &7u64.to_le_bytes(), 8), 8u64.to_le_bytes());
    }

    #[test]
    fn state_space_glued_to_type_in_declarations() {
        let regs = ".reg.u64 %rd1;\n.reg.b32 %r1;";
        let body = "ld.global.u64 %rd1, [%in]; add.u64 %rd1, %rd1, 1; st.global.u64 [%out], %rd1;";
        assert_eq!(run(body, regs, &41u64.to_le_bytes(), 8), 42u64.to_le_bytes());
    }

    #[test]
    fn sad_64_bit() {
        let regs = ".reg .b64 %rd<8>;";
        let body = "ld.global.u64 %rd1, [%in]; ld.global.u64 %rd2, [%in+8]; ld.global.u64 %rd3, [%in+16];\n\
                    sad.s64 %rd4, %rd1, %rd2, %rd3; sad.u64 %rd5, %rd1, %rd2, %rd3;\n\
                    st.global.u64 [%out], %rd4; st.global.u64 [%out+8], %rd5;";
        let input: Vec<u8> = [12i64, -38, 915].iter().flat_map(|x| x.to_le_bytes()).collect();
        let got = run(body, regs, &input, 16);
        // signed |12 - -38| = 50; unsigned |12 - 0xFF..DA| = 2^64 - 50.
        let want: Vec<u8> = [965u64, 915u64.wrapping_sub(50)].iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn mov_b32_of_a_shared_symbol_is_its_address() {
        let regs = ".shared .align 4 .b32 tile[4];\n.reg .b32 %r<4>; .reg .f32 %f1;";
        let body = "ld.global.u32 %r1, [%in]; mov.b32 %r2, tile; st.shared.u32 [%r2+8], %r1;\n\
                    ld.shared.u32 %r3, [tile+8]; mov.b32 %f1, 0f3F800000;\n\
                    st.global.u32 [%out], %r3; st.global.f32 [%out+4], %f1;";
        assert_eq!(words(&run(body, regs, &le(&[77]), 8)), [77, 0x3F80_0000]);
    }
}
