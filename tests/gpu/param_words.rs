//! Diagnose aggregate `.b8` param word loads at offset 16 (soft_max ncols).
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::null_mut;

const SOURCE: &str = r#".version 8.8
.target sm_80
.address_size 64
.entry diag_param_words(
.param .align 8 .b8 diag_param_words_param_0[128],
.param .u64 diag_param_words_param_1
)
{
.reg .b64 %rd<6>;
.reg .b32 %r<4>;
ld.param.u64 %rd1, [diag_param_words_param_0+16];
ld.param.u32 %r1, [diag_param_words_param_0+8];
ld.param.u64 %rd2, [diag_param_words_param_0+0];
ld.param.f32 %f1, [diag_param_words_param_0+112];
ld.param.u64 %rd3, [diag_param_words_param_1];
cvta.to.global.u64 %rd4, %rd3;
st.global.u64 [%rd4+0], %rd1;
st.global.u32 [%rd4+8], %r1;
st.global.u64 [%rd4+16], %rd2;
st.global.f32 [%rd4+24], %f1;
ret;
}
"#;

const KERNEL: &std::ffi::CStr = c"diag_param_words";

#[repr(C)]
struct SoftMaxParams {
    nheads: i64,
    n_head_log2: u32,
    _pad: u32,
    ncols: i64,
    nrows_x: i64,
    nrows_y: i64,
    ne00: i64,
    ne01: i64,
    ne02: i64,
    ne03: i64,
    nb11: i64,
    nb12: i64,
    nb13: i64,
    ne12: i64,
    ne13: i64,
    scale: f32,
    max_bias: f32,
    m0: f32,
    m1: f32,
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn aggregate_param_offset16_roundtrip() {
    assert_eq!(std::mem::size_of::<SoftMaxParams>(), 128);
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
        cuCtxSynchronize(),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let source = std::ffi::CString::new(SOURCE.replace("%f1", "%f1")).unwrap();
    // Fix: need .reg .f32 for f1
    let source = std::ffi::CString::new(SOURCE.replace(
        ".reg .b32 %r<4>;",
        ".reg .b32 %r<4>;\n.reg .f32 %f<2>;",
    ))
    .unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, source.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, KERNEL.as_ptr()));

    let mut params = SoftMaxParams {
        nheads: 14,
        n_head_log2: 8,
        _pad: 0xDEAD_BEEF,
        ncols: 12,
        nrows_x: 14,
        nrows_y: 1,
        ne00: 12,
        ne01: 1,
        ne02: 14,
        ne03: 1,
        nb11: 48,
        nb12: 48,
        nb13: 48,
        ne12: 1,
        ne13: 1,
        scale: 0.125,
        max_bias: 0.0,
        m0: 1.0,
        m1: 1.0,
    };
    let mut dout = 0u64;
    ok!(cuMemAlloc_v2(&mut dout, 32));
    let zeros = [0u8; 32];
    ok!(cuMemcpyHtoD_v2(dout, zeros.as_ptr().cast(), 32));
    let mut out_ptr = dout;
    let mut args: [*mut c_void; 2] = [
        (&mut params as *mut SoftMaxParams).cast(),
        (&mut out_ptr as *mut u64).cast(),
    ];
    ok!(cuLaunchKernel(
        kernel, 1, 1, 1, 1, 1, 1, 0, null_mut(), args.as_mut_ptr(), null_mut()
    ));
    ok!(cuCtxSynchronize());
    let mut out = [0u8; 32];
    ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), dout, 32));
    let ncols = i64::from_ne_bytes(out[0..8].try_into().unwrap());
    let n_head_log2 = u32::from_ne_bytes(out[8..12].try_into().unwrap());
    let nheads = i64::from_ne_bytes(out[16..24].try_into().unwrap());
    let scale = f32::from_ne_bytes(out[24..28].try_into().unwrap());
    eprintln!("got ncols={ncols} n_head_log2={n_head_log2} nheads={nheads} scale={scale}");
    assert_eq!(ncols, 12, "ncols at +16");
    assert_eq!(n_head_log2, 8, "n_head_log2 at +8");
    assert_eq!(nheads, 14, "nheads at +0");
    assert!((scale - 0.125).abs() < 1e-6, "scale at +112");
    ok!(cuMemFree_v2(dout));
    ok!(cuCtxDestroy_v2(context));
}
