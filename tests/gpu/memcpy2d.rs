//! Pitched device-to-device cuMemcpy2DAsync on hardware: the shapes stock GGML issues from
//! `ggml_cuda_cpy_tensor_2d` (padded rows, and the one-element-wide strided gather), checked
//! byte for byte, plus the time per call. Run once as is (one region copy per call) and once
//! with CFI_COPY_REGION=0 (one copy per row) for the A/B; each run prints a `MEMCPY2D:` line.
use super::{Handle, library, ok, symbols, test_device};
use std::ffi::c_void;
use std::ptr::{null, null_mut};
use std::time::Instant;

#[repr(C)]
struct Memcpy2D {
    src_x: usize,
    src_y: usize,
    src_type: u32,
    src_host: *const c_void,
    src_device: u64,
    src_array: Handle,
    src_pitch: usize,
    dst_x: usize,
    dst_y: usize,
    dst_type: u32,
    dst_host: *mut c_void,
    dst_device: u64,
    dst_array: Handle,
    dst_pitch: usize,
    width: usize,
    height: usize,
}

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
        src_type: 2,
        src_host: null(),
        src_device: src,
        src_array: null_mut(),
        src_pitch,
        dst_x: 0,
        dst_y: 0,
        dst_type: 2,
        dst_host: null_mut(),
        dst_device: dst,
        dst_array: null_mut(),
        dst_pitch: width,
        width,
        height: rows,
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
#[allow(non_snake_case)]
fn memcpy2d_device_to_device_shapes_and_timing() {
    let dll = library();
    symbols!(dll;
        cuInit(u32),
        cuCtxCreate_v2(*mut Handle, u32, i32),
        cuCtxDestroy_v2(Handle),
        cuStreamCreate(*mut Handle, u32),
        cuStreamDestroy_v2(Handle),
        cuStreamSynchronize(Handle),
        cuMemAlloc_v2(*mut u64, usize),
        cuMemFree_v2(u64),
        cuMemcpyHtoD_v2(u64, *const c_void, usize),
        cuMemcpyDtoH_v2(*mut c_void, u64, usize),
        cuMemsetD8_v2(u64, u8, usize),
        cuMemcpy2DAsync_v2(*const Memcpy2D, Handle),
    );
    ok!(cuInit(0));
    let mut context = null_mut();
    ok!(cuCtxCreate_v2(&mut context, 0, test_device()));
    let mut stream = null_mut();
    ok!(cuStreamCreate(&mut stream, 1));

    const SRC: usize = 4 << 20;
    let pattern: Vec<u8> = (0..SRC).map(|i| (i * 131 + (i >> 9)) as u8).collect();
    let (mut src, mut dst) = (0u64, 0u64);
    ok!(cuMemAlloc_v2(&mut src, SRC));
    ok!(cuMemAlloc_v2(&mut dst, SRC));
    ok!(cuMemcpyHtoD_v2(src, pattern.as_ptr().cast(), SRC));

    // (src_x, src_y, src_pitch, width, rows): GGML padded rows (f32/f16 rows of a view with a
    // larger row stride), odd widths and origins, and the strided gather GGML falls back to
    // for a non-contiguous row (width = one element, pitch = element stride).
    let shapes: [(usize, usize, usize, usize, usize); 6] = [
        (0, 0, 4096, 3584, 64),
        (128, 3, 2048, 1792, 512),
        (5, 1, 1000, 997, 37),
        (0, 0, 256, 2, 896),
        (2, 0, 64, 4, 4864),
        (0, 0, 7, 7, 2),
    ];
    for &(x, y, pitch, width, rows) in &shapes {
        ok!(cuMemsetD8_v2(dst, 0xee, rows * width + 64));
        let copy = d2d(dst, src, (x, y, pitch), width, rows);
        ok!(cuMemcpy2DAsync_v2(&copy, stream));
        ok!(cuStreamSynchronize(stream));
        let mut got = vec![0u8; rows * width + 64];
        ok!(cuMemcpyDtoH_v2(got.as_mut_ptr().cast(), dst, got.len()));
        for row in 0..rows {
            let from = (y + row) * pitch + x;
            assert_eq!(
                &got[row * width..(row + 1) * width],
                &pattern[from..from + width],
                "shape x={x} y={y} pitch={pitch} width={width} rows={rows}: row {row}"
            );
        }
        assert!(
            got[rows * width..].iter().all(|&b| b == 0xee),
            "wrote past the region"
        );
    }

    // Time per call for GGML's two shapes, steady state (after one warm-up call each).
    let mut timing = Vec::new();
    for &(pitch, width, rows) in &[(4096usize, 3584usize, 512usize), (256, 2, 896)] {
        let copy = d2d(dst, src, (0, 0, pitch), width, rows);
        ok!(cuMemcpy2DAsync_v2(&copy, stream));
        ok!(cuStreamSynchronize(stream));
        const CALLS: u32 = 50;
        let started = Instant::now();
        for _ in 0..CALLS {
            ok!(cuMemcpy2DAsync_v2(&copy, stream));
        }
        let enqueue = started.elapsed().as_secs_f64() * 1e6 / f64::from(CALLS);
        ok!(cuStreamSynchronize(stream));
        let total = started.elapsed().as_secs_f64() * 1e6 / f64::from(CALLS);
        timing.push(format!(
            "{rows}x{width}B: enqueue_us={enqueue:.1} total_us={total:.1}"
        ));
    }

    ok!(cuStreamDestroy_v2(stream));
    ok!(cuMemFree_v2(src));
    ok!(cuMemFree_v2(dst));
    ok!(cuCtxDestroy_v2(context));
    let mode = match std::env::var("CFI_COPY_REGION").as_deref() {
        Ok("0") => "per-row",
        _ => "region",
    };
    println!("MEMCPY2D: mode={mode} shapes=PASS {}", timing.join(" "));
}
