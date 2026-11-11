# Precompiling kernels (`cfi-precompile`)

Roadmap item 6 of [zluda-parity.md](zluda-parity.md). Without it, the first run of a program pays for two things. Every kernel
it launches is translated from PTX to SPIR-V, and the Intel compiler then builds it. For stock Ollama with Qwen2.5-0.5B Q8_0,
most of that time goes to four `mul_mat_q` kernels of about 2 million SPIR-V words each, at 23 to 30 s apiece.
`cfi-precompile` does both steps ahead of time.

```
scripts\cfi.ps1 precompile -- --gguf "$env:USERPROFILE\.ollama\models\blobs" "<ollama>\lib\ollama\cuda_v13"
# or directly:
cfi-precompile.exe --install-dir <install>\bin [options] <file-or-dir>...
```

## What it does

1. Walks the given executables, DLLs, fatbins and `.ptx` files.
   - It scans every nvcc fatbin frame in a file and extracts the PTX record that the runtime would pick, using the same
     `cfi_fatbin::extract_ptx` call. Identical modules in several files are deduplicated.
   - NVIDIA libraries that `cfi-run` replaces with ours are skipped: `cublas64_*`, `cublasLt64_*`, `cufft64_*`, `cusparse64_*`
     and `nvrtc64_*` (`--all-files` scans them anyway). NVIDIA's cuSOLVER is scanned, because it runs on our cuBLAS.
2. Opens the device as an application does (primary context, device probe), so the translator switches that depend on the
   device, such as the dynamic shared-memory window and the dp4a lowering, are the same as at run time.
3. Translates every entry on all cores into the SPIR-V cache (`CFI_CACHE_DIR`), skipping architecture-guard stubs, which the
   runtime refuses to load anyway.
4. Builds the selected entries with the Intel compiler into the native-binary cache (`CUDA_FOR_INTEL_CACHE_DIR`). It also
   builds the native replacement kernels of `crate::subst` that the launch path would use for them, and, when the cuBLAS shim
   sits next to `nvcuda.dll`, the common f16 XMX GEMM kernels and their self-test probes.

The tool lives in `nvcuda.dll` as the export `cfi_precompile_main`. `cfi-precompile.exe` only loads the `nvcuda.dll` it is
given and calls that export, so the translator that fills the caches is exactly the one that will read them.

## Options

| Option | Meaning |
| --- | --- |
| `--gguf PATH` | Add the tensor types of a GGUF model, or of every GGUF file in a directory such as Ollama's `models\blobs`, to the native selection. f32 and f16 are always added |
| `--ggml-types LIST` | Native builds only for ggml kernels instantiated for these tensor types (`q4_k,q6_k,q8_0`, ...; `all`). Kernels without a `ggml_type` template argument are always built |
| `--native-skip TEXT` | No native build for entries whose name contains TEXT (repeatable). With Ollama's flash attention off, `--native-skip flash_attn` saves time |
| `--no-native` | Open the device but fill the SPIR-V cache only |
| `--translate-only` | No GPU at all: SPIR-V cache only, with `--dynamic-shared N` (default 65536, the B390's window) standing in for the device probe |
| `--threads N`, `--device N`, `--filter TEXT`, `--list`, `--report FILE`, `--quiet`, `--all-files` | Worker threads; device ordinal; only entries containing TEXT; list without translating; per-entry TSV; summary only; see above |

The `--report` TSV columns are `module, entry, result, translate_ms, compile_ms, prepare_ms, detail`. `prepare_ms` is the
module-wide PTX normalisation, charged once to the entry whose thread ran it (0 for the others of that module).
`translate_ms` is the entry's own parse, SPIR-V emission and cache write, without the prepare or the time spent waiting for
another thread's prepare of the same module.

By default (no type selection) every entry gets a native build. For a ggml library that is 3,307 entries, and the mmq kernels
of 21 tensor types alone take several GB. That is more than the native cache's default cap
(`CUDA_FOR_INTEL_CACHE_MAX_MB=4096`), so select the types your models use (`--gguf`).

## Cache correctness

The tool has no cache keys of its own.
- A SPIR-V entry is stored under the key the launch path computes: the hash of the raw PTX text, the entry name, the crate
  and translator versions (`TRANSLATOR_VERSION`), the device-probe switches and every `CFI_*` translator variable.
- A native binary is keyed by its SPIR-V words, the device and driver identity, and the build flags, exactly as
  `Context::load_module` keys it.

A mismatch is therefore a cache miss, never a wrong kernel. Precompile in the same environment as the application: the same
cache directories, and the same `CFI_*` and `CUDA_FOR_INTEL_*` switches.

Launcher, cache-location, dump and tracing variables are excluded from the SPIR-V key (`xcache::key_neutral`).
`CFI_REDIRECT_DIR` is one of them: `cfi-run` sets it in the application's environment but not in a precompile run, and hashing
it made every entry a miss. A unit test checks that a precompiled entry is found under the launch-path key, holds the same SPIR-V as a
fresh translation, and that a translator switch makes it a miss. A new translator release changes `TRANSLATOR_VERSION` or the
crate version, so the old entries are simply never hit.

## Measured (Arc B390, stock Ollama v0.12.6 under `cfi-run`, Qwen2.5-0.5B Q8_0)

| Run | First request: load + 307-token prefill + 1 token |
| --- | ---: |
| Cold, empty caches | 115.5 s |
| After `cfi-precompile --gguf <model> --native-skip flash_attn <ollama>\lib\ollama\cuda_v13` (211 s, 16 threads) | 2.36 s |
| Warm (a later run of either) | 1.64-1.89 s |

Details and findings are in [hw-logs/precompile-2026-11-09.md](hw-logs/precompile-2026-11-09.md). For this model the
precompiled caches hold 2.4 GB of SPIR-V and 319 MB of native binaries.

## Not covered

- Kernels with module variables (723 of ggml-cuda's 3,307 entries) are not stored in the SPIR-V cache, as before.
  Translating them again at run time is cheap, and their native binaries are precompiled.
- cuBLAS XMX kernels beyond the common f16 set (bf16, epilogues, bias, 2D block-load variants) are built on first use.
- PyTorch: the tool scans `torch\lib` DLLs the same way, but no torch cold-start number was measured here.
