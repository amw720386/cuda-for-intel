//! Pitched device-to-device copies (`cuMemcpy2DAsync_v2`) on the fake Level Zero backend.
//!
//! GGML's `ggml_cuda_cpy_tensor_2d` (stock Ollama v0.12.6, ggml-cuda.cu:1337) copies a padded
//! tensor slice with one `cudaMemcpy2DAsync(dst, row, src, nb1, row, rows, DeviceToDevice)`.
//! It must reach the GPU as one region copy, not one copy (plus barrier) per row.
use super::fake_tests::{Fixture, assert_baseline, context, stream};
use super::{cuStreamDestroy_v2, cuStreamSynchronize};
use crate::cuda::gaps::{Memcpy2D, cuMemcpy2DAsync_v2};
use crate::cuda::graphs::{cuGraphDestroy, cuStreamBeginCapture_v2, cuStreamEndCapture};
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

/// `rows` rows of `width` bytes from `src` (row pitch `src_pitch`, starting at byte `src_x` of
/// row `src_y`) to a packed `dst`.
fn d2d(
    dst: u64,
    src: u64,
    (src_x, src_y, src_pitch): (usize, usize, usize),
    width: usize,
    rows: usize,
) -> Memcpy2D {
    Memcpy2D {
        src_x,
        src_y,
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
fn padded_rows_are_one_region_copy() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    let (src, dst) = (alloc(2048), alloc(1024));
    let before = fake::copy_log().len();
    let copy = d2d(dst, src, (8, 1, 32), 16, 63);
    assert_eq!(unsafe { cuMemcpy2DAsync_v2(&copy, s) }, Success);
    let regions = fake::region_log();
    assert_eq!(fake::copy_log().len() - before, 0, "no per-row copies");
    assert_eq!(regions.len(), 1, "one region copy for 63 rows");
    let r = &regions[0];
    // The region starts at the requested origin; pitches and shape pass through unchanged.
    assert_eq!((r.dst, r.src), (dst as usize, src as usize + 32 + 8));
    assert_eq!(
        (r.width, r.height, r.dst_pitch, r.src_pitch),
        (16, 63, 16, 32)
    );
    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuMemFree_v2(src), Success);
    assert_eq!(cuMemFree_v2(dst), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}

#[test]
fn region_copy_validates_the_whole_span_and_keeps_fallbacks() {
    let _fixture = Fixture::new();
    let ctx = context(None);
    let s = stream();
    let (src, dst) = (alloc(2048), alloc(1024));
    // The last source row would end past the allocation: rejected, nothing appended.
    let past_end = d2d(dst, src, (24, 0, 32), 16, 64);
    assert_eq!(unsafe { cuMemcpy2DAsync_v2(&past_end, s) }, InvalidValue);
    assert!(fake::region_log().is_empty());

    // Host-to-device stays a per-row copy (pageable host memory needs no region support).
    let host = vec![0u8; 2048];
    let mut from_host = d2d(dst, 0, (0, 0, 32), 16, 4);
    from_host.src_type = HOST;
    from_host.src_host = host.as_ptr().cast();
    let before = fake::copy_log().len();
    assert_eq!(unsafe { cuMemcpy2DAsync_v2(&from_host, s) }, Success);
    assert_eq!(cuStreamSynchronize(s), Success);
    assert!(fake::copy_log().len() - before >= 1);
    assert!(fake::region_log().is_empty());

    // Under stream capture the copy becomes graph nodes (per row), never a region command.
    assert_eq!(cuStreamBeginCapture_v2(s, 2), Success); // relaxed
    let copy = d2d(dst, src, (0, 0, 32), 16, 8);
    assert_eq!(unsafe { cuMemcpy2DAsync_v2(&copy, s) }, Success);
    let mut graph = null_mut();
    assert_eq!(unsafe { cuStreamEndCapture(s, &mut graph) }, Success);
    assert!(fake::region_log().is_empty());
    assert_eq!(cuGraphDestroy(graph), Success);

    // One row is a plain copy.
    let one = d2d(dst, src, (0, 0, 32), 16, 1);
    let before = fake::copy_log().len();
    assert_eq!(unsafe { cuMemcpy2DAsync_v2(&one, s) }, Success);
    assert_eq!(fake::copy_log().len() - before, 1);
    assert!(fake::region_log().is_empty());

    assert_eq!(cuStreamSynchronize(s), Success);
    assert_eq!(cuStreamDestroy_v2(s), Success);
    assert_eq!(cuMemFree_v2(src), Success);
    assert_eq!(cuMemFree_v2(dst), Success);
    assert_eq!(cuCtxDestroy_v2(ctx), Success);
    assert_baseline();
}
