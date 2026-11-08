//! Compile tests; they skip (with a message) when no clang with NVPTX is installed.
use nvrtc64_120_0 as nvrtc;
use nvrtc::compiler::{self, Failure, Request};
use std::ffi::{CStr, CString, c_char};

fn have_clang() -> bool {
    let ok = compiler::find_clang().is_ok();
    if !ok {
        eprintln!("SKIP: no clang with NVPTX (set NVRTC_SHIM_CLANG)");
    }
    ok
}

fn req(src: &str) -> Request {
    Request { source: src.into(), name: "t.cu".into(), ..Default::default() }
}

#[test]
fn vector_add_emits_entry_and_ptx_version_in_range() {
    if !have_clang() {
        return;
    }
    let out = compiler::compile(&req(include_str!("kernels/vadd.cu"))).unwrap();
    assert!(out.ptx.contains(".visible .entry vadd("));
    let ver = out.ptx.lines().find_map(|l| l.strip_prefix(".version ")).unwrap();
    let major: u32 = ver.split('.').next().unwrap().parse().unwrap();
    assert!((7..=9).contains(&major), "translator accepts PTX 7..9, got {ver}");
    assert!(out.ptx.contains(".address_size 64"));
}

#[test]
fn error_log_has_diagnostics_and_program_name() {
    if !have_clang() {
        return;
    }
    let e = compiler::compile(&req("extern \"C\" __global__ void k(int* p) { p[0] = undefined_symbol; }")).unwrap_err();
    let Failure::Compilation(log) = e else { panic!("expected compilation failure") };
    assert!(log.contains("t.cu:1") && log.contains("undefined_symbol"), "{log}");
    assert!(!log.contains("nvrtc_shim_"), "staging paths must be hidden: {log}");
}

#[test]
fn headers_and_defines_and_default_device() {
    if !have_clang() {
        return;
    }
    let mut r = req("#include \"my/h.cuh\"\nextern \"C\" __global__ void k(float* p) { p[0] = twice(VALUE); }");
    r.headers = vec![("my/h.cuh".into(), "__device__ float twice(float x) { return 2.f * x; }".into())];
    r.options = vec!["-DVALUE=3.0f".into(), "--gpu-architecture=compute_70".into()];
    let out = compiler::compile(&r).unwrap();
    assert!(out.ptx.contains(".target sm_70"));
    // -default-device: unannotated function usable from a kernel
    let mut r = req("float helper(float x) { return x + 1.f; }\nextern \"C\" __global__ void k(float* p) { p[0] = helper(p[1]); }");
    r.options = vec!["-default-device".into()];
    compiler::compile(&r).unwrap();
}

#[test]
fn arch_is_capped_for_the_translator() {
    if !have_clang() {
        return;
    }
    let mut r = req(include_str!("kernels/vadd.cu"));
    r.options = vec!["--gpu-architecture=sm_90".into()];
    assert_eq!(compiler::compile(&r).unwrap().arch, 75);
}

#[test]
fn name_expressions_are_lowered_via_clang_mangling() {
    if !have_clang() {
        return;
    }
    let mut r = req(include_str!("kernels/templ.cu"));
    r.name_expressions = vec!["axpy<float>".into(), "&axpy<int>".into(), "ns::scale<4>".into()];
    let out = compiler::compile(&r).unwrap();
    assert_eq!(out.lowered["axpy<float>"], "_Z4axpyIfEvT_PKS0_PS0_i");
    assert_eq!(out.lowered["&axpy<int>"], "_Z4axpyIiEvT_PKS0_PS0_i");
    assert_eq!(out.lowered["ns::scale<4>"], "_ZN2ns5scaleILi4EEEvPfi");
    for sym in out.lowered.values() {
        assert!(out.ptx.contains(&format!(".visible .entry {sym}(")), "{sym}");
    }
    assert!(!out.ptx.contains("__nvrtc_ne_"));
}

#[test]
fn ptx_passthrough() {
    let ptx = "// c\n.version 7.0\n.target sm_75\n.address_size 64\n";
    assert!(compiler::is_ptx(ptx));
    let out = compiler::compile(&req(ptx)).unwrap();
    assert_eq!(out.ptx, ptx);
    assert!(!compiler::is_ptx("__global__ void k() {}"));
}

#[test]
fn invalid_option_is_reported() {
    let mut r = req("x");
    r.options = vec!["--no-such-flag".into()];
    assert!(matches!(compiler::compile(&r), Err(Failure::InvalidOption(_))));
}

#[test]
fn cache_hits_on_identical_requests() {
    if !have_clang() {
        return;
    }
    let r = req("extern \"C\" __global__ void cache_probe_kernel(int* p) { p[0] = 42; }");
    let a = compiler::compile(&r).unwrap();
    let before = nvrtc::cache::hits();
    let b = compiler::compile(&r).unwrap();
    assert!(nvrtc::cache::hits() > before);
    assert_eq!(a.ptx, b.ptx);
    let mut r2 = r.clone();
    r2.options = vec!["-DX=1".into()];
    let before = nvrtc::cache::hits();
    compiler::compile(&r2).unwrap();
    assert_eq!(nvrtc::cache::hits(), before, "different options must miss");
}

#[test]
fn c_abi_roundtrip() {
    if !have_clang() {
        return;
    }
    unsafe {
        let src = CString::new(include_str!("kernels/templ.cu")).unwrap();
        let name = CString::new("t.cu").unwrap();
        let mut p = std::ptr::null_mut();
        assert_eq!(nvrtc::nvrtcCreateProgram(&mut p, src.as_ptr(), name.as_ptr(), 0, std::ptr::null(), std::ptr::null()), 0);
        let ne = CString::new("axpy<double>").unwrap();
        assert_eq!(nvrtc::nvrtcAddNameExpression(p, ne.as_ptr()), 0);
        let mut low: *const c_char = std::ptr::null();
        assert_eq!(nvrtc::nvrtcGetLoweredName(p, ne.as_ptr(), &mut low), nvrtc::NVRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION);
        let opt = CString::new("--std=c++17").unwrap();
        let opts = [opt.as_ptr()];
        assert_eq!(nvrtc::nvrtcCompileProgram(p, 1, opts.as_ptr()), 0);
        assert_eq!(nvrtc::nvrtcGetLoweredName(p, ne.as_ptr(), &mut low), 0);
        assert_eq!(CStr::from_ptr(low).to_str().unwrap(), "_Z4axpyIdEvT_PKS0_PS0_i");
        let (mut n, mut ln) = (0usize, 0usize);
        assert_eq!(nvrtc::nvrtcGetPTXSize(p, &mut n), 0);
        assert_eq!(nvrtc::nvrtcGetProgramLogSize(p, &mut ln), 0);
        assert!(ln >= 1);
        let mut buf = vec![0u8; n];
        assert_eq!(nvrtc::nvrtcGetPTX(p, buf.as_mut_ptr().cast()), 0);
        assert_eq!(*buf.last().unwrap(), 0);
        assert!(std::str::from_utf8(&buf[..n - 1]).unwrap().contains(".entry"));
        assert_eq!(nvrtc::nvrtcDestroyProgram(&mut p), 0);
        assert!(p.is_null());
        // compilation failure keeps the log and returns NVRTC_ERROR_COMPILATION
        let bad = CString::new("this is not CUDA").unwrap();
        assert_eq!(nvrtc::nvrtcCreateProgram(&mut p, bad.as_ptr(), std::ptr::null(), 0, std::ptr::null(), std::ptr::null()), 0);
        assert_eq!(nvrtc::nvrtcCompileProgram(p, 0, std::ptr::null()), nvrtc::NVRTC_ERROR_COMPILATION);
        assert_eq!(nvrtc::nvrtcGetProgramLogSize(p, &mut ln), 0);
        assert!(ln > 20);
        assert_eq!(nvrtc::nvrtcGetPTXSize(p, &mut n), nvrtc::NVRTC_ERROR_INVALID_INPUT);
        nvrtc::nvrtcDestroyProgram(&mut p);
        let (mut maj, mut min) = (0, 0);
        assert_eq!(nvrtc::nvrtcVersion(&mut maj, &mut min), 0);
        assert_eq!(maj, 12);
        assert_eq!(CStr::from_ptr(nvrtc::nvrtcGetErrorString(6)).to_str().unwrap(), "NVRTC_ERROR_COMPILATION");
    }
}

#[test]
fn large_switch_does_not_emit_jump_tables() {
    if !have_clang() {
        return;
    }
    // A 100-way switch makes NVPTX emit `brx.idx` unless jump tables are disabled; the translator
    // has no indirect branch (PyTorch's erfcx jiterator kernel hits this).
    let cases: String = (0..100).map(|k| format!("case {k}: r = x * {k}.5f + {}.f; break;\n", k * k + 1)).collect();
    let src = format!(
        "extern \"C\" __global__ void k(float* a, float* o) {{ int i = threadIdx.x; float x = a[i]; float r; switch ((int)x) {{ {cases} default: r = 0.f; }} o[i] = r; }}"
    );
    let out = compiler::compile(&req(&src)).unwrap();
    assert!(!out.ptx.contains("brx.idx"), "jump table emitted");
}
