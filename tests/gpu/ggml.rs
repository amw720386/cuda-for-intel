use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{CString, c_char, c_void};
use std::ptr::{from_mut, null_mut};

const SOURCE: &str = include_str!("../../examples/ggml_scale_clamp.ptx");
const SCALE: &std::ffi::CStr = c"_Z9scale_f32IfEvPKT_PS0_ffj5uint3S4_S4_jjjjjj";
const CLAMP: &std::ffi::CStr = c"_Z15op_clamp_kernelIfEvPKT_PS0_S0_S0_j5uint3S4_S4_jjjjjj";

fn fastdiv(d: u32) -> [u32; 3] {
    // GGML's host-side init_fastdiv_values. The reference uses ordinary division.
    let shift = d.next_power_of_two().trailing_zeros();
    let d64 = u64::from(d);
    let multiplier = (((1u64 << shift) - d64) << 32) / d64 + 1;
    [multiplier as u32, shift, d]
}

fn offset(mut i: u32, dims: [u32; 3], strides: [u32; 3]) -> usize {
    let mut element = i % dims[0];
    for axis in 0..3 {
        i /= dims[axis];
        let coordinate = if axis < 2 { i % dims[axis + 1] } else { i };
        element += coordinate * strides[axis];
    }
    element as usize
}

fn args(
    mut pointers: [u64; 2],
    mut floats: [f32; 2],
    mut count: u32,
    mut divisors: [[u32; 3]; 3],
    mut src: [u32; 3],
    mut dst: [u32; 3],
    launch: impl FnOnce(*mut Handle) -> i32,
) -> i32 {
    let mut params = Vec::new();
    params.extend(pointers.iter_mut().map(|v| from_mut(v).cast::<c_void>()));
    params.extend(floats.iter_mut().map(|v| from_mut(v).cast::<c_void>()));
    params.push(from_mut(&mut count).cast());
    params.extend(divisors.iter_mut().map(|v| from_mut(v).cast::<c_void>()));
    params.extend(
        src.iter_mut()
            .chain(&mut dst)
            .map(|v| from_mut(v).cast::<c_void>()),
    );
    launch(params.as_mut_ptr())
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn ggml_cuda_scale_clamp_chain() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxSetCurrent(Handle),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuModuleUnload(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuStreamCreate(*mut Handle, u32),
        cuStreamSynchronize(Handle),
        cuEventCreate(*mut Handle, u32),
        cuEventRecord(Handle, Handle),
        cuStreamWaitEvent(Handle, Handle, u32),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    let source = CString::new(SOURCE).unwrap();
    let (mut context, mut foreign, mut module, mut scale, mut clamp) =
        (null_mut(), null_mut(), null_mut(), null_mut(), null_mut());
    ok!(cuInit(0));
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    ok!(cuModuleLoadData(&mut module, source.as_ptr().cast()));
    ok!(cuModuleGetFunction(&mut scale, module, SCALE.as_ptr()));
    ok!(cuModuleGetFunction(&mut clamp, module, CLAMP.as_ptr()));

    let mut streams = [null_mut(); 2];
    for stream in &mut streams {
        ok!(cuStreamCreate(stream, 1));
    }
    let mut ready = null_mut();
    ok!(cuEventCreate(&mut ready, 2));

    let dims = [37, 5, 3];
    let count = dims.into_iter().product::<u32>() * 2;
    let divs = dims.map(fastdiv);
    let strides = [[41, 211, 647], [47, 251, 767], [43, 229, 691]];
    let bias = -0.107f32;
    let factor = 1.334f32;
    let limits = [-2.75f32, 1.25f32];
    let guard = 9999.0f32;
    let mut buffers = [0u64; 3];
    let mut initial = Vec::new();
    let mut expected = Vec::new();
    for (index, &stride) in strides.iter().enumerate() {
        // Include outer guards and check padding inside all strided views.
        let len = 2 * stride[2] as usize + 2;
        initial.push(vec![guard; len]);
        expected.push(vec![guard; len]);
        ok!(cuMemAlloc_v2(&mut buffers[index], len * 4));
    }
    for i in 0..count {
        let x = ((i * 131 % 383) as f32 - 171.0) / 63.0;
        let y = x.mul_add(factor, bias);
        initial[0][1 + offset(i, dims, strides[0])] = x;
        expected[1][1 + offset(i, dims, strides[1])] = y;
        expected[2][1 + offset(i, dims, strides[2])] = y.clamp(limits[0], limits[1]);
    }
    let upload = |index: usize, input: &[f32]| {
        ok!(cuMemcpyHtoD_v2(
            buffers[index],
            input.as_ptr().cast(),
            std::mem::size_of_val(input)
        ));
    };
    let read = |index: usize| {
        let mut actual = vec![0.0f32; initial[index].len()];
        let bytes = actual.len() * 4;
        ok!(cuMemcpyDtoH_v2(
            actual.as_mut_ptr().cast(),
            buffers[index],
            bytes
        ));
        actual
    };
    let launch_with = |kernel, src: usize, dst: usize, floats, stream, packed: bool| {
        args(
            [buffers[src] + 4, buffers[dst] + 4],
            floats,
            count,
            divs,
            strides[src],
            strides[dst],
            |params| {
                let launch = |params, extra| unsafe {
                    cuLaunchKernel(
                        kernel,
                        count.div_ceil(256),
                        1,
                        1,
                        256,
                        1,
                        1,
                        0,
                        stream,
                        params,
                        extra,
                    )
                };
                if !packed {
                    return launch(params, null_mut());
                }
                let mut buffer = [0u8; 88];
                let mut offset = 0;
                for (index, size) in [8, 8, 4, 4, 4, 12, 12, 12, 4, 4, 4, 4, 4, 4]
                    .into_iter()
                    .enumerate()
                {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            params.add(index).read().cast::<u8>(),
                            buffer.as_mut_ptr().add(offset),
                            size,
                        );
                    }
                    offset += size;
                }
                let mut size = buffer.len();
                let mut extra = [
                    std::ptr::without_provenance_mut(1),
                    buffer.as_mut_ptr().cast(),
                    std::ptr::without_provenance_mut(2),
                    from_mut(&mut size).cast(),
                    null_mut(),
                ];
                launch(null_mut(), extra.as_mut_ptr())
            },
        )
    };
    let launch =
        |kernel, src, dst, floats, stream| launch_with(kernel, src, dst, floats, stream, false);
    let assert_values = |actual: Vec<f32>, reference: &[f32]| {
        for (index, (&got, &want)) in actual.iter().zip(reference).enumerate() {
            if want.is_nan() {
                assert!(got.is_nan(), "word {index}: expected NaN, got {got}");
            } else {
                assert_eq!(
                    got.to_bits(),
                    want.to_bits(),
                    "word {index}: got {got}, expected {want}"
                );
            }
        }
    };
    for (i, values) in initial.iter().enumerate() {
        upload(i, values);
    }

    // Each implementation is first verified independently.
    assert_eq!(launch(scale, 0, 1, [factor, bias], streams[0]), 0);
    ok!(cuStreamSynchronize(streams[0]));
    assert_values(read(1), &expected[1]);
    let mut edge_input = expected[1].clone();
    let mut edge_output = expected[2].clone();
    let edge_values = [
        (f32::NAN, -0.0),
        (f32::NEG_INFINITY, -0.0),
        (-0.0, -0.0),
        (0.0, 0.0),
        (f32::from_bits(1), f32::from_bits(1)),
        (f32::INFINITY, limits[1]),
    ];
    for i in 0..count {
        let x = edge_input[1 + offset(i, dims, strides[1])];
        edge_output[1 + offset(i, dims, strides[2])] = x.clamp(-0.0, limits[1]);
    }
    for (i, (x, y)) in edge_values.into_iter().enumerate() {
        edge_input[1 + i] = x;
        edge_output[1 + i] = y;
    }
    upload(1, &edge_input);
    assert_eq!(launch(clamp, 1, 2, [-0.0, limits[1]], streams[1]), 0);
    ok!(cuStreamSynchronize(streams[1]));
    assert_values(read(2), &edge_output);

    // Then alternate a single stream with explicit cross-stream dependencies.
    for iteration in 0..8 {
        upload(1, &initial[1]);
        upload(2, &initial[2]);
        let packed = iteration >= 4;
        assert_eq!(
            launch_with(scale, 0, 1, [factor, bias], streams[0], packed),
            0
        );
        let consumer = if iteration % 2 == 0 {
            streams[0]
        } else {
            ok!(cuEventRecord(ready, streams[0]));
            ok!(cuStreamWaitEvent(streams[1], ready, 0));
            streams[1]
        };
        assert_eq!(launch_with(clamp, 1, 2, limits, consumer, packed), 0);
        ok!(cuStreamSynchronize(consumer));
        assert_values(read(1), &expected[1]);
        assert_values(read(2), &expected[2]);
    }
    assert_values(read(0), &initial[0]);

    ok!(cuCtxCreate_v2(&mut foreign, 0, test_device()));
    assert_eq!(launch(scale, 0, 1, [factor, bias], null_mut()), 400);
    let (mut foreign_module, mut foreign_kernel) = (null_mut(), null_mut());
    ok!(cuModuleLoadData(
        &mut foreign_module,
        source.as_ptr().cast()
    ));
    ok!(cuModuleGetFunction(
        &mut foreign_kernel,
        foreign_module,
        SCALE.as_ptr()
    ));
    // Inferred pointer parameters are validated against the owning context.
    assert_eq!(launch(foreign_kernel, 0, 1, [factor, bias], null_mut()), 1);
    ok!(cuCtxDestroy_v2(foreign));
    ok!(cuCtxSetCurrent(context));
    ok!(cuModuleUnload(module));
    assert_eq!(launch(scale, 0, 1, [factor, bias], streams[0]), 400);
    ok!(cuCtxDestroy_v2(context));
}

const INTEGER_EDGES: &str = r#"
.version 8.8
.target sm_80
.address_size 64
.entry integer_edges(.param .u64 out, .param .align 4 .b8 words[12]) {
    .reg .b64 %addr;
    .reg .b64 %base;
    .reg .b64 %wide;
    .reg .b64 %next;
    .reg .b32 %a;
    .reg .b32 %b;
    .reg .b32 %shift;
    .reg .b32 %high;
    .reg .b32 %low;
    ld.param.u64 %addr, [out];
    cvta.to.global.u64 %base, %addr;
    ld.param.u32 %a, [words];
    ld.param.u32 %b, [words+4];
    ld.param.u32 %shift, [words+8];
    mul.hi.u32 %high, %a, %b;
    shr.u32 %low, %a, %shift;
    st.global.b32 [%base], %high;
    cvt.u64.u32 %wide, 4;
    add.u64 %next, %base, %wide;
    st.global.b32 [%next], %low;
    ret;
}
"#;

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn modern_integer_edges() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuModuleLoadData(*mut Handle, *const c_void),
        cuModuleGetFunction(*mut Handle, Handle, *const c_char),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
    );
    let (mut context, mut module, mut kernel) = (null_mut(), null_mut(), null_mut());
    let mut out = 0u64;
    ok!(cuInit(0));
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let ptx = CString::new(INTEGER_EDGES).unwrap();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    ok!(cuModuleGetFunction(
        &mut kernel,
        module,
        c"integer_edges".as_ptr()
    ));
    ok!(cuMemAlloc_v2(&mut out, 8));
    for shift in [0, 1, 31, 32, 33, u32::MAX] {
        let mut words = [u32::MAX, 0xfedc_ba98, shift];
        let mut params = [from_mut(&mut out).cast(), from_mut(&mut words).cast()];
        ok!(cuLaunchKernel(
            kernel,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            null_mut(),
            params.as_mut_ptr(),
            null_mut()
        ));
        let mut output = [0u32; 2];
        ok!(cuMemcpyDtoH_v2(output.as_mut_ptr().cast(), out, 8));
        let product = u64::from(words[0]) * u64::from(words[1]);
        assert_eq!(
            output,
            [
                (product >> 32) as u32,
                words[0].checked_shr(shift).unwrap_or(0)
            ]
        );
    }
    ok!(cuCtxDestroy_v2(context));
}
