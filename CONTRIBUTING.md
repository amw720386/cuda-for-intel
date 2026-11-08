# Contributing

## Branches and commits

- As seen in the history, day-to-day work is integrated on `integration` and `main` lags behind it; feature branches (`hw-torch`, `perf-decode`, `xmx-default`, ...) start from
  the current `origin/integration` and are merged back into it. Open pull requests against `integration` unless a maintainer says otherwise.
- Never force-push a shared branch and never push directly to `main`.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/): `feat(ptx): ...`, `fix(cuda): ...`, `test(gpu): ...`, `docs: ...`,
  `perf: ...`. Keep one logical change per commit and commit code together with its test.
- `docs/` is listed in `.gitignore`; add documentation files with `git add -f docs/<file>`.
- Never commit third-party binaries (`*.dll`, `*.exe` are ignored), NVIDIA's `nvcudart_hybrid64.dll`, model files, generated PTX dumps, host names, IP
  addresses, user paths or credentials. Scan your diff before pushing (`git diff origin/integration | grep -nEi 'password|secret|token|[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+|C:\\Users'`).
- Results and logs belong in the repository (`docs/hw-logs/`), with paths scrubbed, so a claim in the README can be traced to a log or a test.

## Toolchain

Rust stable with edition 2024 support (`cargo --version` >= 1.85). Windows needs the MSVC x64 toolchain; GPU runs need the Intel graphics driver
(it provides the Level Zero loader). Use `-j 2` on small machines: heavy parallel builds on a laptop iGPU host can starve the display.

## Unit tests (no GPU)

The default test set uses the fake Level Zero backend and runs everywhere.

| OS | Command | Notes |
| --- | --- | --- |
| macOS | `cargo test --lib` | 229 passed on the Windows box; the macOS count is UNVERIFIED. The Level Zero loader targets Windows and Linux, so use the library tests here rather than the GPU test target. |
| Linux | `cargo test --locked --workspace` | The Linux path is not hardware-verified; `--lib` needs no GPU. Untested here. |
| Windows | `cargo test --locked --workspace` | Same command, plus the opt-in GPU tests below. |

Per-crate: `cargo test -p cublas_shim` (generators, XMX SPIR-V interpreter against CPU references), `cargo test -p cfi-fatbin`.
Hardware-independent translator validation with Khronos `spirv-val`: set `SPIRV_VAL` or put it on `PATH` and run `cargo test --locked --lib validate_opencl_spirv -- --ignored`.

## PTX corpus harness (macOS, Linux, Windows)

`tests/ptx_corpus` holds 600+ PTX files (ZLUDA-derived kernels, handwritten probes, real ggml kernels). `tests/ptx_corpus/run.sh` builds a scratch harness
that includes `src/ptx.rs` directly, translates each file and compares the first failure with `expected_status.tsv`. It exits non-zero on any change.
Run it after every translator change. `run.sh --bless` rewrites the expected file: do that only when the change is intended, and commit the new
`expected_status.tsv` with the translator change. The harness also probes single modules:
`ptx-corpus-harness --probe module.ptx` (translate every entry, group failures) and `--prep` (print normalised PTX).
Coverage figures: `docs/ptx-gap-report.md`.

## Hardware tests (Intel GPU, Windows)

Hardware tests are `#[ignore]`d. Build the release DLLs first, then:

```powershell
cargo build --release --locked -p cuda-for-intel -p cublas_shim
cargo test --release --locked --test gpu -- --ignored --nocapture --test-threads=1
pwsh -File scripts\check-install.ps1          # quick smoke: device query + vector add
pwsh -File scripts\validate-gpu.ps1           # full validation of every detected GPU, log in docs\gpu-validation.log
```

They load the built DLL by absolute path. `CUDA_FOR_INTEL_DEVICE=<ordinal>` selects the Level Zero device for the tests. Always use `--test-threads=1`:
tests share one device and some deliberately stress it.

Some tests need extra inputs and skip silently when they are missing: `CFI_STOCK_GGML_DLL` (path to Ollama v0.12.6 `ggml-cuda.dll`, for the stock-kernel
compile and scan tests), `CFI_MIN_AVAILABLE_MB`, `CFI_STRESS_LAUNCHES`. Read the header comment of the test file for its variables.

### The GPU lock

One GPU is a shared resource. Two test runs at once give timing noise, can exhaust shared memory on integrated parts and, on Windows, can trigger a driver
reset (TDR) that kills both. When a GPU machine is used by several people or agents, take the lock before any hardware run and release it afterwards.
The convention in this repository (`tests/cuda_apps/run.bat`) is an atomic directory creation:

```bat
mkdir %USERPROFILE%\cfi-work\gpu.lock 2>nul || (echo GPU LOCK HELD & exit /b 2)
rem ... run hardware tests ...
rmdir %USERPROFILE%\cfi-work\gpu.lock
```

`mkdir` fails when the directory exists, so exactly one holder wins. Do not break someone else's lock; if a lock is stale, ask. Keep hardware sessions short
(slices of a few minutes) and release the lock in a `finally` block or trap so a failed test does not leave it behind.

## Pull request checklist

- `cargo test --lib` (or the workspace tests) passes; `tests/ptx_corpus/run.sh` is clean or deliberately blessed.
- New translator behaviour has a unit test; new hardware behaviour has an ignored GPU test with a CPU reference.
- Anything the README claims is backed by a committed log or a test name. Mark untested items as untested.
- The README states only current, correct results. Do not narrate failed runs there; put one line under "What we tried" at most.
- PowerShell scripts parse (`[System.Management.Automation.Language.Parser]::ParseFile`) and take parameters rather than hard-coded paths.

## Regenerating the PTX fixtures

The checked-in fixtures in `examples/*.ptx` come from Clang with the NVPTX backend (Clang 23.1.1 was used). Example:

```powershell
clang --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -O2 -S examples/vector_add.cu -o examples/vector_add.ptx
clang --cuda-device-only --cuda-gpu-arch=sm_52 -nocudainc -nocudalib -O2 -fno-unroll-loops -ffp-contract=off -S examples/rmsnorm.cu -o examples/rmsnorm.ptx
```

`block_reduce`, `forge_residual` and `forge_reduce` use `-fno-unroll-loops` (the last two also `-ffp-contract=off`); `atomic_tickets` uses the plain flags.
