# cuFFT Bluestein and shared-memory passes on the Arc B390, 2026-11-09

Build: branch `libs-cufft` (`nvcuda.dll`, `cufft64_11.dll`, `cublas64_12.dll`, `cusparse64_12.dll`, `cfi_redirect.dll`, `cfi-run.exe`), release.
Baseline: `main` at the merge of PR #8 (one global Stockham launch per radix, a prime factor as one O(n p) direct-DFT stage).

## What changed

Each transformed dimension now picks one of (`crates/cufft_shim/src/plan.rs`, `Pass`):
- **local**: length <= 2048 (f32) / 1024 (f64) with factors <= 63: one launch, a 256-thread work-group transforms whole lines in shared memory (radix 8, 4, 2 then odd factors, a barrier between stages), several lines per group for short lengths.
- **four-step**: a longer last dimension `n = n1 n2` with both factors local-sized: n1-point local passes with the `W_n^(j2 k1)` twiddle fused into the store, then n2-point passes whose store transposes into place.
- **Bluestein**: a prime factor > 63: input chirp, forward FFT of the power of two m >= 2n - 1 (itself local or four-step), product with the host-computed FFT of the chirp, inverse FFT, output chirp with 1/m. Two extra work buffers of m elements per line; `cufftGetSize*` and the work-area size include them.
- otherwise the previous global Stockham stages.
Dense C2C plans read the input in the first pass and write the output in the last, without the gather and scatter copies.

## `tests/gpu/hw_cufft.rs`

1 passed, **56 transforms** (42 before), errors against a host f64 DFT:

| new case | f32 | f64 |
|---|---|---|
| Bluestein: 10007, 4099 batch 2 in place, 131 x 6 (prime along a non-last dimension) | 1.6e-7 to 2.3e-7 | - |
| Bluestein: 4099, 2062 (2 x 1031) | - | 3.2e-15 to 4.1e-15 |
| four-step: 16384, 3 x 6000 | 1.5e-7 to 2.0e-7 | - |
| four-step: 8192 in place | - | 5.2e-15 |

The earlier 42 transforms (local passes now) stay within 0.8e-7 to 4.7e-7 (f32) and 3e-16 to 6e-15 (f64).
Simulator (`cargo test --release --lib cufft`, macOS): 13 passed, including Bluestein 67 (f32), 67 x 3 (f64) and four-step 2048 (f64) through the translated kernels; the host model (`plan.rs` tests) covers 127, 262, 67 x 6, 5 x 71, 3000, 4096 and 4099 against a naive DFT and the chirp-z identity on the tables.

## torch.fft through `cfi-run` (PyTorch 2.2.2+cu118, untouched `torch\lib`)

`scripts/torch/fft_check.py`: **196 checks passed, 0 failed** (sizes 1-4096 incl. 97, 105, 840, 2D/3D, stft, f32 and f64).

## Throughput (`tests/gpu/hw_libs_perf.rs`, out-of-place forward, best of 3)

GB/s counts one read and one write of the data (the least an out-of-place FFT moves); the roofline is a device-to-device copy at 112-113 GB/s.

| case | before | after | % of copy | time | speedup |
|---|---|---|---|---|---|
| C2C n=1048576 batch=1 | 6.6 GB/s | 9.3 GB/s | 8% | 2.537 -> 1.81 ms | 1.40x |
| C2C n=4096 batch=256 | 10.9 GB/s | 15.4 GB/s | 14% | 1.545 -> 1.09 ms | 1.42x |
| C2C n=256 batch=4096 | 14.3 GB/s | 23.9 GB/s | 21% | 1.175 -> 0.701 ms | 1.68x |
| C2C n=1000 batch=1000 | 11.7 GB/s | 16.5 GB/s | 15% | 1.364 -> 0.969 ms | 1.41x |
| Z2Z n=262144 batch=1 | 11 GB/s | 11.8 GB/s | 11% | 0.761 -> 0.711 ms | 1.07x |
| Z2Z n=4096 batch=64 | 15.7 GB/s | 18.5 GB/s | 17% | 0.534 -> 0.454 ms | 1.18x |
| C2C n=4099 batch=64 | 0.1 GB/s | 1.7 GB/s | 2% | 46.984 -> 2.527 ms | 18.59x |
| C2C n=10007 batch=16 | 0 GB/s | 1.8 GB/s | 2% | 72.896 -> 1.391 ms | 52.41x |
| Z2Z n=10007 batch=16 | 0.1 GB/s | 3.2 GB/s | 3% | 50.336 -> 1.613 ms | 31.21x |
| C2C n=65537 batch=2 | 0 GB/s | 1.3 GB/s | 1% | 407.829 -> 1.672 ms | 243.92x |
| C2C n=4194304 batch=1 | 1.8 GB/s | 6.2 GB/s | 6% | 36.932 -> 10.861 ms | 3.40x |

Prime lengths gain 19-244x (no longer O(n^2)); power-of-two and smooth lengths 1.1-1.7x, 4M points 3.4x. Throughput is still 6-21% of the copy roofline for smooth sizes and 1-3% for Bluestein (five passes over 2-4x padded data).
Not done (likely next steps, not profiled): twiddles come from global memory on every stage and the index math uses 32-bit division; caching twiddles in shared memory, power-of-two specialised kernels and a fused gather for R2C/C2R would cut the remaining overhead.
