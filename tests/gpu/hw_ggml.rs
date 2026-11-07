//! Randomized checks of authentic ggml CUDA kernels (PTX dumps in examples/) against CPU references.
use super::hw_common::*;

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_ggml_dequantize_block_q8_0_to_half() {
    let ptx = include_str!("../../examples/dequantize_block_dump.ptx");
    let entry = ptx
        .lines()
        .find_map(|l| l.strip_prefix(".entry ").map(|r| r.trim_end_matches('(').to_string()))
        .unwrap();
    let mut rng = Rng(0xdec0_dec0_dec0_dec0);
    let (ne00, ne01, ne02, ne03) = (96usize, 5usize, 2usize, 3usize);
    let row_blocks = ne00 / 32;
    let (s01, s02, s03) = (row_blocks + 1, ne01 * (row_blocks + 1), ne02 * ne01 * (row_blocks + 1)); // padded rows
    let total_blocks = ne03 * s03;
    let mut src = vec![0u8; total_blocks * 34];
    let mut scales = vec![0f32; total_blocks];
    for b in 0..total_blocks {
        let d = f32_to_f16(((rng.range(-64, 64) as f32) / 16.0) * 0.5);
        scales[b] = f16_to_f32(d);
        src[b * 34..b * 34 + 2].copy_from_slice(&d.to_le_bytes());
        for i in 0..32 {
            src[b * 34 + 2 + i] = rng.range(-128, 127) as i8 as u8;
        }
    }
    let n_out = ne00 * ne01 * ne02 * ne03;
    let mut bufs = vec![src.clone(), vec![0u8; n_out * 2]];
    let args = [
        Arg::Buf(0),
        Arg::Buf(1),
        i64_arg(ne00 as i64),
        i64_arg(ne01 as i64),
        i64_arg((ne02 * ne03) as i64),
        fastdiv_arg(ne02 as u32),
        i64_arg(s01 as i64),
        i64_arg(s02 as i64),
        i64_arg(s03 as i64),
    ];
    // Fewer blocks than elements so the grid-stride loops are exercised.
    launch_args(ptx, &entry, [1, 2, 4], [64, 1, 1], 0, &mut bufs, &args).unwrap();
    let out: Vec<u16> = bufs[1].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
    for i03 in 0..ne03 {
        for i02 in 0..ne02 {
            for i01 in 0..ne01 {
                for i00 in 0..ne00 {
                    let blk = i03 * s03 + i02 * s02 + i01 * s01 + i00 / 32;
                    let q = src[blk * 34 + 2 + i00 % 32] as i8 as f32;
                    let want = f32_to_f16(scales[blk] * q);
                    let idx = ((i03 * ne02 + i02) * ne01 + i01) * ne00 + i00;
                    assert_eq!(out[idx], want, "dequantize q8_0 at ({i00},{i01},{i02},{i03})");
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_ggml_bin_bcast_add_f32_broadcast() {
    let ptx = include_str!("../../examples/k_bin_bcast_add_dump.ptx");
    let entry = ptx
        .lines()
        .find_map(|l| l.strip_prefix(".entry ").map(|r| r.trim_end_matches('(').to_string()))
        .unwrap();
    let mut rng = Rng(0xb1ca_b1ca_b1ca_b1ca);
    let (ne0, ne1, ne2, ne3) = (70usize, 5usize, 3usize, 2usize);
    let (ne10, ne11, ne12, ne13) = (7usize, 1usize, 3usize, 1usize);
    let a: Vec<f32> = (0..ne0 * ne1 * ne2 * ne3).map(|_| rng.range(-1000, 1000) as f32 / 8.0).collect();
    let b: Vec<f32> = (0..ne10 * ne11 * ne12 * ne13).map(|_| rng.range(-1000, 1000) as f32 / 8.0).collect();
    let fb = |v: &[f32]| bytes_of_u32(&v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    let mut bufs = vec![fb(&a), fb(&b), vec![0u8; a.len() * 4]];
    let args = [
        Arg::Buf(0),
        Arg::Bytes(vec![0; 8]), // src1 travels in the variadic pack
        Arg::Buf(2),
        u32_arg(ne0 as u32),
        u32_arg(ne1 as u32),
        u32_arg(ne2 as u32),
        fastdiv_arg(ne3 as u32),
        fastdiv_arg(ne10 as u32),
        fastdiv_arg(ne11 as u32),
        fastdiv_arg(ne12 as u32),
        fastdiv_arg(ne13 as u32),
        u32_arg(ne0 as u32),               // s1
        u32_arg((ne0 * ne1) as u32),       // s2
        u32_arg((ne0 * ne1 * ne2) as u32), // s3
        u32_arg(1),                        // s00
        u32_arg(ne0 as u32),               // s01
        u32_arg((ne0 * ne1) as u32),       // s02
        u32_arg((ne0 * ne1 * ne2) as u32), // s03
        u32_arg(1),                        // s10
        u32_arg(ne10 as u32),              // s11
        u32_arg((ne10 * ne11) as u32),     // s12
        u32_arg((ne10 * ne11 * ne12) as u32), // s13
        Arg::Buf(1),
    ];
    // x covers ne0 in two passes; y splits ne1; z enumerates (ne2, ne3).
    launch_args(ptx, &entry, [1, 3, (ne2 * ne3) as u32], [32, 2, 1], 0, &mut bufs, &args).unwrap();
    let out = u32s_of(&bufs[2]);
    for i3 in 0..ne3 {
        for i2 in 0..ne2 {
            for i1 in 0..ne1 {
                for i0 in 0..ne0 {
                    let (i11, i12, i13, i10) = (i1 % ne11, i2 % ne12, i3 % ne13, i0 % ne10);
                    let bv = b[((i13 * ne12 + i12) * ne11 + i11) * ne10 + i10];
                    let idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;
                    assert_eq!(f32::from_bits(out[idx]), a[idx] + bv, "bin_bcast add at ({i0},{i1},{i2},{i3})");
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_ggml_get_rows_float_batched() {
    let ptx = include_str!("../../examples/k_get_rows_float_dump.ptx");
    let entry = ptx
        .lines()
        .find_map(|l| l.strip_prefix(".entry ").map(|r| r.trim_end_matches('(').to_string()))
        .unwrap();
    let mut rng = Rng(0x6e72_6f77_7367_6574);
    let (ne00, nrows, n_idx, ne11, ne12) = (48usize, 20usize, 6usize, 2usize, 3usize);
    let src: Vec<f32> = (0..ne12 * ne11 * nrows * ne00).map(|_| rng.range(-5000, 5000) as f32 / 16.0).collect();
    let idx: Vec<u32> = (0..ne12 * ne11 * n_idx).map(|_| rng.u32() % nrows as u32).collect();
    let (nb01, nb02, nb03) = (ne00 * 4, ne12 * nrows * ne00 * 4, nrows * ne00 * 4);
    let _ = nb03;
    let (s10, s11, s12) = (1usize, n_idx * ne12, n_idx);
    let (s1, s2, s3) = (ne00, n_idx * ne00 * ne12, n_idx * ne00);
    let fb = |v: &[f32]| bytes_of_u32(&v.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    let mut bufs = vec![fb(&src), bytes_of_u32(&idx), vec![0u8; ne12 * ne11 * n_idx * ne00 * 4]];
    // Parameter order recovered from the PTX: ne00, ne11, fastdiv(ne12), s1, s2, s3, nb01, nb02, nb03,
    // s10, s11, s12. The block's z index is i11 * ne12 + i12.
    let args = [
        Arg::Buf(0),
        Arg::Buf(1),
        Arg::Buf(2),
        i64_arg(ne00 as i64),
        i64_arg(ne11 as i64),
        fastdiv_arg(ne12 as u32),
        i64_arg(s1 as i64),
        i64_arg(s2 as i64),
        i64_arg(s3 as i64),
        i64_arg(nb01 as i64),
        i64_arg(nb02 as i64),
        i64_arg(nb03 as i64),
        i64_arg(s10 as i64),
        i64_arg(s11 as i64),
        i64_arg(s12 as i64),
    ];
    launch_args(ptx, &entry, [n_idx as u32, 1, (ne11 * ne12) as u32], [64, 1, 1], 0, &mut bufs, &args).unwrap();
    let out = u32s_of(&bufs[2]);
    for i12 in 0..ne12 {
        for i11 in 0..ne11 {
            for i10 in 0..n_idx {
                let row = idx[i10 * s10 + i11 * s11 + i12 * s12] as usize;
                for c in 0..ne00 {
                    let want = src[((i11 * ne12 + i12) * nrows + row) * ne00 + c];
                    let got = f32::from_bits(out[i10 * s1 + i11 * s2 + i12 * s3 + c]);
                    assert_eq!(got, want, "get_rows ({c},{i10},{i11},{i12})");
                }
            }
        }
    }
}

#[test]
#[ignore = "requires a built nvcuda.dll and an Intel Level Zero GPU"]
fn hw_ggml_mul_mat_vec_q8_0_q8_1() {
    let ptx = include_str!("../../examples/mul_mat_vec_q8_dump.ptx");
    let entry = ptx
        .lines()
        .find_map(|l| l.strip_prefix(".entry ").map(|r| r.trim_end_matches('(').to_string()))
        .unwrap();
    let mut rng = Rng(0x7671_7638_7671_7638);
    let (ncols, nrows, nchan) = (256usize, 8usize, 2usize);
    let nb = ncols / 32;
    // x: q8_0 weights per channel/row; y: q8_1 activations per channel.
    let mut x = vec![0u8; nchan * nrows * nb * 34];
    let (mut xd, mut xq) = (vec![0f32; nchan * nrows * nb], vec![0i32; nchan * nrows * nb * 32]);
    for b in 0..nchan * nrows * nb {
        let d = f32_to_f16(rng.range(-60, 60) as f32 / 64.0);
        xd[b] = f16_to_f32(d);
        x[b * 34..b * 34 + 2].copy_from_slice(&d.to_le_bytes());
        for i in 0..32 {
            let q = rng.range(-127, 127);
            x[b * 34 + 2 + i] = q as i8 as u8;
            xq[b * 32 + i] = q;
        }
    }
    let mut y = vec![0u8; nchan * nb * 36];
    let (mut yd, mut yq) = (vec![0f32; nchan * nb], vec![0i32; nchan * nb * 32]);
    for b in 0..nchan * nb {
        let d = f32_to_f16(rng.range(1, 60) as f32 / 256.0);
        yd[b] = f16_to_f32(d);
        y[b * 36..b * 36 + 2].copy_from_slice(&d.to_le_bytes());
        for i in 0..32 {
            let q = rng.range(-127, 127);
            y[b * 36 + 4 + i] = q as i8 as u8;
            yq[b * 32 + i] = q;
        }
    }
    let mut bufs = vec![x, y, vec![0u8; 16], vec![0u8; nchan * nrows * 4]];
    let args = [
        Arg::Buf(0),
        Arg::Buf(1),
        Arg::Bytes(vec![0; 8]),  // ids = null
        Arg::Bytes(vec![0; 80]), // fusion args = none
        Arg::Buf(3),
        u32_arg(ncols as u32),
        fastdiv_arg(nchan as u32),
        u32_arg(nb as u32),         // stride_row_x (blocks)
        u32_arg(nb as u32),         // stride_col_y (q8_1 blocks)
        u32_arg(nrows as u32),      // stride_col_dst
        fastdiv_arg(1),             // channel_ratio
        u32_arg((nrows * nb) as u32), // stride_channel_x
        u32_arg(nb as u32),         // stride_channel_y
        u32_arg(nrows as u32),      // stride_channel_dst
        fastdiv_arg(1),             // sample_ratio
        u32_arg((nchan * nrows * nb) as u32),
        u32_arg((nchan * nb) as u32),
        u32_arg((nchan * nrows) as u32),
        u32_arg(0),
    ];
    launch_args(ptx, &entry, [(nrows / 4) as u32, nchan as u32, 1], [32, 4, 1], 0, &mut bufs, &args).unwrap();
    let out = u32s_of(&bufs[3]);
    for c in 0..nchan {
        for r in 0..nrows {
            let (mut want, mut mag) = (0f64, 0f64);
            for b in 0..nb {
                let xb = (c * nrows + r) * nb + b;
                let yb = c * nb + b;
                let dot: i32 = (0..32).map(|i| xq[xb * 32 + i] * yq[yb * 32 + i]).sum();
                want += (xd[xb] * yd[yb]) as f64 * dot as f64;
                mag += ((xd[xb] * yd[yb]) as f64 * dot as f64).abs();
            }
            let got = f32::from_bits(out[c * nrows + r]) as f64;
            assert!((got - want).abs() <= 1e-4 * mag + 1e-6, "mmvq q8_0 channel {c} row {r}: got {got} want {want}");
        }
    }
}
