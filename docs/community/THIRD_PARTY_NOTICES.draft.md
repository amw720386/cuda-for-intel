# Third-party notices (DRAFT)

Generated from `cargo metadata --locked` at c065736 plus a manual audit of copied text. Regenerate with `cargo about` or `cargo license` before a release; verify full licence texts are shipped with binary releases (MIT/BSD/ISC/Apache require the notice).
Not legal advice.

## Material copied or adapted into this repository

| Material | Where | Upstream licence | Notice kept |
| --- | --- | --- | --- |
| ZLUDA (https://github.com/vosen/ZLUDA) test PTX kernels | `tests/ptx_corpus/zluda/` | MIT OR Apache-2.0 | per-file header + `LICENSE-ZLUDA-MIT`, `LICENSE-ZLUDA-APACHE` |
| ZLUDA dark_api digest and fatbin header layouts | `src/cuda/integrity.rs`, `crates/fatbin/src/lib.rs` | MIT OR Apache-2.0 | mention in `LICENSE`; exact copyright line to be added (open decision) |
| ggml / llama.cpp kernels (adapted source, compiled PTX) | `examples/ggml_*`, `tests/ptx_corpus/ggml/` | MIT, Copyright (c) 2023-2026 The ggml authors | `examples/ggml-LICENSE`, per-file header |
| Level Zero `ze_api.h` declarations (re-expressed in Rust) | `src/` Level Zero bindings | MIT, Copyright (C) 2019-2026 Intel Corporation | `LICENSE` |

Not included, by policy: NVIDIA CUDA headers, NVIDIA DLLs, any ZLUDA code beyond the above.

## Cargo dependencies (Cargo.lock)

| Crate | Version | Licence (as declared) |
| --- | --- | --- |
| adler2 | 2.0.1 | 0BSD OR MIT OR Apache-2.0 |
| bitflags | 2.13.2 | MIT OR Apache-2.0 |
| cc | 1.11.1 | MIT OR Apache-2.0 |
| cfg-if | 1.0.5 | MIT OR Apache-2.0 |
| crc32fast | 1.5.4 | MIT OR Apache-2.0 |
| find-msvc-tools | 0.1.15 | MIT OR Apache-2.0 |
| flate2 | 1.1.11 | MIT OR Apache-2.0 |
| getrandom | 0.4.3 | MIT OR Apache-2.0 |
| jobserver | 0.1.35 | MIT OR Apache-2.0 |
| libc | 0.2.191 | MIT OR Apache-2.0 |
| libloading | 0.8.9 | ISC |
| lz4_flex | 0.11.6 | MIT |
| miniz_oxide | 0.9.1 | MIT OR Zlib OR Apache-2.0 |
| pkg-config | 0.3.34 | MIT OR Apache-2.0 |
| r-efi | 6.0.0 | MIT OR Apache-2.0 OR LGPL-2.1-or-later (choose MIT/Apache) |
| rspirv | 0.13.0 | Apache-2.0 |
| rustc-hash | 1.1.0 | Apache-2.0 / MIT |
| shlex | 2.0.1 | MIT OR Apache-2.0 |
| simd-adler32 | 0.3.10 | MIT |
| spirv | 0.4.0 | Apache-2.0 |
| windows-link | 0.2.1 | MIT OR Apache-2.0 |
| zstd | 0.13.4 | BSD-3-Clause |
| zstd-safe | 7.4.0 | BSD-3-Clause |
| zstd-sys | 2.1.2 | BSD-3-Clause (bundles upstream zstd C library, BSD-3-Clause OR GPL-2.0; BSD chosen) |

Workspace crates (`cuda-for-intel`, `cfi-fatbin`, `cublas_shim`, `cudart_shim`, `nvrtc_shim`) are this project and carry the project licence.

## Runtime tools (not distributed)

clang with the NVPTX target (Apache-2.0 WITH LLVM-exception) is used by `nvrtc_shim` at run time if installed by the user. Khronos `spirv-val` is optional (Apache-2.0). Neither is bundled.
