use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

/// dp4a in its four signedness forms and dp2a lo/hi in all four, over random inputs.
const FORMS: [(&str, &str); 12] = [
    ("dp4a", "s32.s32"),
    ("dp4a", "u32.u32"),
    ("dp4a", "s32.u32"),
    ("dp4a", "u32.s32"),
    ("dp2a.lo", "s32.s32"),
    ("dp2a.lo", "u32.u32"),
    ("dp2a.lo", "s32.u32"),
    ("dp2a.lo", "u32.s32"),
    ("dp2a.hi", "s32.s32"),
    ("dp2a.hi", "u32.u32"),
    ("dp2a.hi", "s32.u32"),
    ("dp2a.hi", "u32.s32"),
];

fn kernel_source() -> String {
    let mut body = String::new();
    for (index, (op, types)) in FORMS.iter().enumerate() {
        body += &format!("\t{op}.{types} %r{}, %r7, %r8, %r9;\n\tst.global.u32 [%rd10+{}], %r{};\n", 20 + index, index * 4, 20 + index);
    }
    format!(
        r#".version 7.0
.target sm_75
.address_size 64
.visible .entry dp_all(.param .u64 pa, .param .u64 pb, .param .u64 pc, .param .u64 pout, .param .u32 pn)
{{
	.reg .pred %p<2>;
	.reg .b32 %r<40>;
	.reg .b64 %rd<14>;
	ld.param.u64 %rd1, [pa];
	ld.param.u64 %rd2, [pb];
	ld.param.u64 %rd3, [pc];
	ld.param.u64 %rd4, [pout];
	ld.param.u32 %r2, [pn];
	mov.u32 %r3, %ctaid.x;
	mov.u32 %r4, %ntid.x;
	mov.u32 %r5, %tid.x;
	mad.lo.s32 %r1, %r3, %r4, %r5;
	setp.ge.s32 %p1, %r1, %r2;
	@%p1 bra DONE;
	cvta.to.global.u64 %rd5, %rd1;
	cvta.to.global.u64 %rd6, %rd2;
	cvta.to.global.u64 %rd7, %rd3;
	cvta.to.global.u64 %rd8, %rd4;
	mul.wide.s32 %rd9, %r1, 4;
	add.s64 %rd11, %rd5, %rd9;
	ld.global.u32 %r7, [%rd11];
	add.s64 %rd11, %rd6, %rd9;
	ld.global.u32 %r8, [%rd11];
	add.s64 %rd11, %rd7, %rd9;
	ld.global.u32 %r9, [%rd11];
	mul.wide.s32 %rd12, %r1, 48;
	add.s64 %rd10, %rd8, %rd12;
{body}DONE:
	ret;
}}
"#
    )
}

fn bytes(word: u32, signed: bool) -> [i64; 4] {
    std::array::from_fn(|i| {
        let b = (word >> (8 * i)) as u8;
        if signed { b as i8 as i64 } else { b as i64 }
    })
}

fn halves(word: u32, signed: bool) -> [i64; 2] {
    std::array::from_fn(|i| {
        let h = (word >> (16 * i)) as u16;
        if signed { h as i16 as i64 } else { h as i64 }
    })
}

fn reference(op: &str, types: &str, a: u32, b: u32, c: u32) -> u32 {
    let (sa, sb) = (types.starts_with('s'), types.split('.').nth(1).unwrap().starts_with('s'));
    let mut acc = c as i64;
    if op == "dp4a" {
        let (x, y) = (bytes(a, sa), bytes(b, sb));
        for i in 0..4 {
            acc += x[i] * y[i];
        }
    } else {
        let x = halves(a, sa);
        let y = bytes(b, sb);
        let base = if op.ends_with("hi") { 2 } else { 0 };
        for i in 0..2 {
            acc += x[i] * y[base + i];
        }
    }
    acc as u32
}

fn random_words(n: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    let mut next = move || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let special = [0u32, 0x8080_8080, 0xffff_ffff, 0x7f7f_7f7f, 0x0080_ff7f, 0x8000_8000, 0xffff_0000, 0x0000_ffff];
    (0..n).map(|i| if i < special.len() * special.len() { special[i / special.len()] ^ (special[i % special.len()] & 0x0101_0101) } else { next() as u32 }).collect()
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU; run with CUDA_FOR_INTEL_DP4A=0 and 1"]
#[allow(non_snake_case)]
fn dp4a_and_dp2a_match_the_reference_in_every_signedness() {
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
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let source = std::ffi::CString::new(kernel_source()).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, source.as_ptr().cast()));
    let mut kernel = null_mut();
    ok!(cuModuleGetFunction(&mut kernel, module, c"dp_all".as_ptr()));

    const N: usize = 8192;
    let (a, b, c) = (random_words(N, 0x1234_5678_9abc_def1), random_words(N, 0x0fed_cba9_8765_4321), random_words(N, 0xdead_beef_cafe_f00d));
    let (mut da, mut db, mut dc, mut dout) = (0u64, 0u64, 0u64, 0u64);
    let word_bytes = N * 4;
    ok!(cuMemAlloc_v2(&mut da, word_bytes));
    ok!(cuMemAlloc_v2(&mut db, word_bytes));
    ok!(cuMemAlloc_v2(&mut dc, word_bytes));
    ok!(cuMemAlloc_v2(&mut dout, N * 48));
    ok!(cuMemcpyHtoD_v2(da, a.as_ptr().cast(), word_bytes));
    ok!(cuMemcpyHtoD_v2(db, b.as_ptr().cast(), word_bytes));
    ok!(cuMemcpyHtoD_v2(dc, c.as_ptr().cast(), word_bytes));
    let mut n = N as i32;
    let mut params = [from_mut(&mut da).cast::<c_void>(), from_mut(&mut db).cast(), from_mut(&mut dc).cast(), from_mut(&mut dout).cast(), from_mut(&mut n).cast()];
    ok!(cuLaunchKernel(kernel, (N / 128) as u32, 1, 1, 128, 1, 1, 0, null_mut(), params.as_mut_ptr(), null_mut()));
    let mut out = vec![0u32; N * 12];
    ok!(cuMemcpyDtoH_v2(out.as_mut_ptr().cast(), dout, N * 48));
    let mut failures = Vec::new();
    for (form, (op, types)) in FORMS.iter().enumerate() {
        let bad = (0..N).filter(|&i| out[i * 12 + form] != reference(op, types, a[i], b[i], c[i])).count();
        if bad != 0 {
            let i = (0..N).find(|&i| out[i * 12 + form] != reference(op, types, a[i], b[i], c[i])).unwrap();
            failures.push(format!("{op}.{types}: {bad}/{N} wrong, first a={:#010x} b={:#010x} c={:#010x} got {:#010x} want {:#010x}", a[i], b[i], c[i], out[i * 12 + form], reference(op, types, a[i], b[i], c[i])));
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
    ok!(cuMemFree_v2(da));
    ok!(cuMemFree_v2(db));
    ok!(cuMemFree_v2(dc));
    ok!(cuMemFree_v2(dout));
    ok!(cuCtxDestroy_v2(context));
}
