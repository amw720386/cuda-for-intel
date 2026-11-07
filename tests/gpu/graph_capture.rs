//! Stream capture and graph replay on a real Intel GPU.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::{c_char, c_void};
use std::ptr::{from_mut, null_mut};

const N: usize = 256;
const RELAXED: i32 = 2;

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn captured_copy_memset_kernel_graph_replays_and_updates() {
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
        cuMemcpyHtoDAsync_v2(u64, *const c_void, usize, Handle),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuMemcpyDtoHAsync_v2(*mut c_void, u64, usize, Handle),
        cuMemcpyDtoDAsync_v2(u64, u64, usize, Handle),
        cuMemsetD8Async(u64, u8, usize, Handle),
        cuLaunchKernel(Handle, u32, u32, u32, u32, u32, u32, u32, Handle, *mut Handle, *mut Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
        cuStreamSynchronize(Handle),
        cuStreamBeginCapture_v2(Handle, i32),
        cuStreamEndCapture(Handle, *mut Handle),
        cuStreamIsCapturing(Handle, *mut i32),
        cuEventCreate(*mut Handle, u32),
        cuEventDestroy_v2(Handle),
        cuEventRecord(Handle, Handle),
        cuStreamWaitEvent(Handle, Handle, u32),
        cuGraphInstantiateWithFlags(*mut Handle, Handle, u64),
        cuGraphLaunch(Handle, Handle),
        cuGraphDestroy(Handle),
        cuGraphExecDestroy(Handle),
    );
    // cuGraphExecUpdate_v2 takes a result structure rather than only integers.
    let update = unsafe {
        dll.get::<unsafe extern "system" fn(Handle, Handle, *mut [usize; 3]) -> i32>(
            b"cuGraphExecUpdate_v2\0",
        )
    }
    .expect("cuGraphExecUpdate_v2");

    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let ptx = std::ffi::CString::new(include_str!("../../examples/vector_add.ptx")).unwrap();
    let mut module = null_mut();
    ok!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
    let mut add = null_mut();
    ok!(cuModuleGetFunction(&mut add, module, c"vector_add".as_ptr()));
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 1));

    let bytes = N * 4;
    let (mut a, mut b, mut out, mut copy) = (0u64, 0u64, 0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut a, bytes));
    ok!(cuMemAlloc_v2(&mut b, bytes));
    ok!(cuMemAlloc_v2(&mut out, bytes));
    ok!(cuMemAlloc_v2(&mut copy, bytes));
    let poison = vec![0xffu8; bytes];
    for buffer in [a, b, out, copy] {
        ok!(cuMemcpyHtoD_v2(buffer, poison.as_ptr().cast(), bytes));
    }

    let mut input: Vec<f32> = (0..N).map(|i| i as f32).collect();
    let mut result = vec![0f32; N];
    let mut n = N as u32;
    // b is zeroed by a captured memset, so out == a after the add kernel.
    let mut params = [
        from_mut(&mut a).cast::<c_void>(),
        from_mut(&mut b).cast::<c_void>(),
        from_mut(&mut out).cast::<c_void>(),
        from_mut(&mut n).cast::<c_void>(),
    ];

    let mut capturing = -1;
    ok!(cuStreamBeginCapture_v2(stream, RELAXED));
    ok!(cuStreamIsCapturing(stream, &mut capturing));
    assert_eq!(capturing, 1);
    ok!(cuMemcpyHtoDAsync_v2(a, input.as_ptr().cast(), bytes, stream));
    ok!(cuMemsetD8Async(b, 0, bytes, stream));
    assert_eq!(
        unsafe {
            cuLaunchKernel(add, 1, 1, 1, N as u32, 1, 1, 0, stream, params.as_mut_ptr(), null_mut())
        },
        0
    );
    ok!(cuMemcpyDtoDAsync_v2(copy, out, bytes, stream));
    ok!(cuMemcpyDtoHAsync_v2(result.as_mut_ptr().cast(), out, bytes, stream));
    // Capture defers every effect: the poison is still in place.
    let mut seen = vec![0u8; bytes];
    ok!(cuMemcpyDtoH_v2(seen.as_mut_ptr().cast(), copy, bytes));
    assert!(seen.iter().all(|&byte| byte == 0xff), "capture executed work");
    let mut graph = null_mut();
    ok!(cuStreamEndCapture(stream, &mut graph));
    ok!(cuStreamIsCapturing(stream, &mut capturing));
    assert_eq!(capturing, 0);
    let mut exec = null_mut();
    ok!(cuGraphInstantiateWithFlags(&mut exec, graph, 0));
    ok!(cuGraphDestroy(graph));

    for round in 0..2 {
        result.fill(-1.0);
        if round == 1 {
            // The host source is read at launch time, not capture time.
            input.iter_mut().for_each(|value| *value += 1000.0);
        }
        ok!(cuGraphLaunch(exec, stream));
        ok!(cuStreamSynchronize(stream));
        let expected: Vec<f32> = input.clone();
        assert_eq!(result, expected, "round {round}");
        ok!(cuMemcpyDtoH_v2(seen.as_mut_ptr().cast(), copy, bytes));
        assert_eq!(seen, expected.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>());
    }

    // Re-capture with a different output buffer and update the executable.
    let mut second_out = 0u64;
    ok!(cuMemAlloc_v2(&mut second_out, bytes));
    let mut params2 = [
        from_mut(&mut a).cast::<c_void>(),
        from_mut(&mut b).cast::<c_void>(),
        from_mut(&mut second_out).cast::<c_void>(),
        from_mut(&mut n).cast::<c_void>(),
    ];
    ok!(cuStreamBeginCapture_v2(stream, RELAXED));
    ok!(cuMemcpyHtoDAsync_v2(a, input.as_ptr().cast(), bytes, stream));
    ok!(cuMemsetD8Async(b, 0, bytes, stream));
    assert_eq!(
        unsafe {
            cuLaunchKernel(add, 1, 1, 1, N as u32, 1, 1, 0, stream, params2.as_mut_ptr(), null_mut())
        },
        0
    );
    ok!(cuMemcpyDtoDAsync_v2(copy, second_out, bytes, stream));
    ok!(cuMemcpyDtoHAsync_v2(result.as_mut_ptr().cast(), second_out, bytes, stream));
    let mut graph2 = null_mut();
    ok!(cuStreamEndCapture(stream, &mut graph2));
    let mut info = [0usize; 3];
    assert_eq!(unsafe { update(exec, graph2, &mut info) }, 0, "update result {}", info[0] as i32);
    ok!(cuGraphDestroy(graph2));
    result.fill(-1.0);
    ok!(cuGraphLaunch(exec, stream));
    ok!(cuStreamSynchronize(stream));
    assert_eq!(result, input);
    ok!(cuMemcpyDtoH_v2(seen.as_mut_ptr().cast(), second_out, bytes));
    assert_eq!(seen, input.iter().flat_map(|v| v.to_ne_bytes()).collect::<Vec<_>>());

    ok!(cuGraphExecDestroy(exec));
    for buffer in [a, b, out, copy, second_out] {
        ok!(cuMemFree_v2(buffer));
    }
    ok!(cuStreamDestroy_v2(stream));
    ok!(cuCtxDestroy_v2(context));
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn captured_fork_join_preserves_dependencies_on_hardware() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuMemcpyDtoDAsync_v2(u64, u64, usize, Handle),
        cuMemsetD8Async(u64, u8, usize, Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
        cuStreamSynchronize(Handle),
        cuStreamBeginCapture_v2(Handle, i32),
        cuStreamEndCapture(Handle, *mut Handle),
        cuEventCreate(*mut Handle, u32),
        cuEventDestroy_v2(Handle),
        cuEventRecord(Handle, Handle),
        cuStreamWaitEvent(Handle, Handle, u32),
        cuGraphInstantiateWithFlags(*mut Handle, Handle, u64),
        cuGraphLaunch(Handle, Handle),
        cuGraphDestroy(Handle),
        cuGraphExecDestroy(Handle),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let (mut origin, mut worker, mut fork, mut join) = (null_mut(), null_mut(), null_mut(), null_mut());
    ok!(cuStreamCreate(&mut origin, 1));
    ok!(cuStreamCreate(&mut worker, 1));
    ok!(cuEventCreate(&mut fork, 2));
    ok!(cuEventCreate(&mut join, 2));
    let (mut x, mut y, mut z) = (0u64, 0u64, 0u64);
    for buffer in [&mut x, &mut y, &mut z] {
        ok!(cuMemAlloc_v2(buffer, 64));
    }
    ok!(cuMemcpyHtoD_v2(x, [0u8; 64].as_ptr().cast(), 64));
    ok!(cuMemcpyHtoD_v2(z, [0u8; 64].as_ptr().cast(), 64));

    // origin: x=0x11 ; fork ; worker: y=x (depends on the memset) ; join ;
    // origin: z=y (depends on the worker's copy).
    ok!(cuStreamBeginCapture_v2(origin, 2));
    ok!(cuMemsetD8Async(x, 0x11, 64, origin));
    ok!(cuEventRecord(fork, origin));
    ok!(cuStreamWaitEvent(worker, fork, 0));
    ok!(cuMemcpyDtoDAsync_v2(y, x, 64, worker));
    ok!(cuEventRecord(join, worker));
    ok!(cuStreamWaitEvent(origin, join, 0));
    ok!(cuMemcpyDtoDAsync_v2(z, y, 64, origin));
    let mut graph = null_mut();
    ok!(cuStreamEndCapture(origin, &mut graph));
    let mut exec = null_mut();
    ok!(cuGraphInstantiateWithFlags(&mut exec, graph, 0));
    ok!(cuGraphLaunch(exec, origin));
    ok!(cuStreamSynchronize(origin));
    let mut got = [0u8; 64];
    ok!(cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), z, 64));
    assert_eq!(got, [0x11u8; 64]);

    ok!(cuGraphExecDestroy(exec));
    ok!(cuGraphDestroy(graph));
    for buffer in [x, y, z] {
        ok!(cuMemFree_v2(buffer));
    }
    ok!(cuEventDestroy_v2(fork));
    ok!(cuEventDestroy_v2(join));
    ok!(cuStreamDestroy_v2(origin));
    ok!(cuStreamDestroy_v2(worker));
    ok!(cuCtxDestroy_v2(context));
}
