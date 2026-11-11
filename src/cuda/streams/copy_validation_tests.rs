//! Strided copies are validated as a whole before any row is issued: CUDA returns
//! CUDA_ERROR_INVALID_VALUE for a 2-D copy that leaves an allocation and copies nothing.
use super::fake_tests::{Fixture, assert_baseline, context, stream};
use super::{cuStreamDestroy_v2, cuStreamSynchronize};
use crate::cuda::gaps::{Memcpy2D, cuMemcpy2D_v2, cuMemcpy2DAsync_v2};
use crate::cuda::*;
use crate::level_zero::fake;
use std::ptr::{null, null_mut};

const DEVICE: u32 = 2; // CU_MEMORYTYPE_DEVICE
const HOST: u32 = 1; // CU_MEMORYTYPE_HOST

fn alloc(size: usize) -> u64 {
    let mut out = 0;
    assert_eq!(unsafe { cuMemAlloc_v2(&mut out, size) }, Success);
    out
}

fn copy(dst: u64, src: u64, src_pitch: usize, width: usize, rows: usize) -> Memcpy2D {
    Memcpy2D {
        src_x: 0,
        src_y: 0,
        src_type: DEVICE,
        src_host: null(),
        src_device: src,
        src_array: null_mut(),
        src_pitch,
        dst_x: 0,
        dst_y: 0,
        dst_type: DEVICE,
        dst_host: null_mut(),
        dst_device: dst,
        dst_array: null_mut(),
        dst_pitch: width,
        width,
        height: rows,
    }
}

#[test]
fn out_of_range_2d_copies_copy_nothing() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    let (src, dst) = (alloc(1024), alloc(1024));
    let host = vec![0u8; 4096];
    let before = fake::copy_log().len();

    // Source rows 32..63 (pitch 32) lie past the 1 KiB source allocation.
    let past_src = copy(dst, src, 32, 16, 64);
    assert_eq!(unsafe { cuMemcpy2DAsync_v2(&past_src, s) }, InvalidValue);
    assert_eq!(unsafe { cuMemcpy2D_v2(&past_src) }, InvalidValue);
    // Destination rows past the end (dst pitch 32 x 64 rows = 2 KiB).
    let mut past_dst = copy(dst, src, 16, 16, 64);
    past_dst.dst_pitch = 32;
    assert_eq!(unsafe { cuMemcpy2DAsync_v2(&past_dst, s) }, InvalidValue);
    // Host to device whose device side runs past the end.
    let mut from_host = copy(dst, 0, 64, 32, 64);
    from_host.src_type = HOST;
    from_host.src_host = host.as_ptr().cast();
    from_host.dst_pitch = 32;
    assert_eq!(unsafe { cuMemcpy2DAsync_v2(&from_host, s) }, InvalidValue);
    // A device pointer that belongs to no allocation at all.
    let stray = copy(dst, src + (1 << 30), 16, 16, 2);
    assert_eq!(unsafe { cuMemcpy2DAsync_v2(&stray, s) }, InvalidValue);

    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(
        fake::copy_log().len(),
        before,
        "rejected copies must not copy any row"
    );

    // An in-range copy of the same shape still succeeds (whatever command it becomes).
    let ok = copy(dst, src, 32, 16, 32);
    assert_eq!(unsafe { cuMemcpy2DAsync_v2(&ok, s) }, Success);
    assert_eq!(unsafe { cuMemcpy2D_v2(&ok) }, Success);
    assert_eq!(cuStreamSynchronize(s), Success);

    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuMemFree_v2(src), Success);
    assert_eq!(cuMemFree_v2(dst), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}
