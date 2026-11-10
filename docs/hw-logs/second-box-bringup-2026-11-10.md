# Second box bring-up: Lunar Lake / Arc 140V (Xe2), driver 32.0.101.5564 (2026-11-10)

Revision: main 8a0a5ed (TRANSLATOR_VERSION 7). Board: Intel Lunar Lake client reference platform (Core Ultra 7 268V,
Arc 140V, PCI 8086:64A0), BIOS 2024-11, 32 GB RAM, Windows 11 Enterprise 22631, Memory integrity (HVCI) on.

**Result: the GPU is not usable on this box, so it produced no GPU results.** The Intel kernel display driver `igfxn`
(`igdkmdn64.sys` 32.0.101.5564, driver date 2024-05-14) has failed to load at every boot since it was installed on
2026-10-15. Everything that reaches the GPU stops at `zeInit`.

## Toolchain installed (official installers, per user unless noted)

| Component | Version | Source |
| --- | --- | --- |
| Visual Studio Build Tools 2022 (machine-wide), workload VCTools + recommended | 17.14 (MSVC 14.44.35207, Windows SDK 10.0.26100) | `aka.ms/vs/17/release/vs_BuildTools.exe` |
| Rust, profile minimal, `stable-x86_64-pc-windows-msvc` | rustc/cargo 1.99.1 (the first box has the same) | `rustup-init.exe` from static.rust-lang.org (SHA-256 matches the published `.sha256`) |
| Python | 3.12.10 (the first box has the same) | python.org installer |
| Git for Windows (machine-wide) | 2.56.0.windows.4 | the git-for-windows GitHub release |
| LLVM, extracted under `cfi-work\llvm` (for `NVRTC_SHIM_CLANG`) | 20.1.8 (the first box has the same) | the llvm-project GitHub release archive |
| Stock Ollama, extracted under `cfi-work\stock\ollama-v0.12.6` (not installed, no service) | 0.12.6, `cuda_v13\ggml-cuda.dll` 4328F98A...38E1A | the official release zip |
| `nvcudart_hybrid64.dll` | BF430D65...22B3 | ZLUDA f4df215 (Git LFS) |
| Qwen2.5-0.5B-Instruct Q8_0 | CA59CA7F...6844E | Hugging Face |

Not installed: the Vulkan SDK (`spirv-val` is only used by ignored lib tests) and torch.
`gpu-queue.ps1` is a byte-identical copy from the first box. The machine execution policy is Restricted, so scripts run with
`powershell -ExecutionPolicy Bypass -File ...` (process scope, which child `powershell` processes inherit).
The GPU driver was not touched.

## Build and tests

| Item | Lunar Lake 140V (this box) | B390, driver 32.0.101.9033 (reference) |
| --- | --- | --- |
| `cargo build --release --workspace` | pass, 55 s | pass |
| `cargo test --release --no-run --test gpu` | pass | pass |
| `cargo test --release --lib -p cuda-for-intel` (no GPU needed) | 472 passed, 0 failed, 43 ignored | (not rerun on main) |
| `gpu-*.exe --ignored --test-threads=1` (116 tests) | 0 passed: every test fails at `cuInit` = 100; exit 99 | 112 passed; 4 failed for environment reasons only (`hw_nvrtc` without clang, 3 `hw_subst_*` without `CFI_STOCK_GGML_DLL`) |
| `scripts/hwvalidate/ptx_tests.ps1` (787 cases per mode) | ptx_tests built; 787/787 crash in both modes (`cuInit` = 100) | default 668 pass + 119 expected mismatch; exact-ftz 780 + 7 |
| Stock Ollama v0.12.6 under cfi-run | not run (no GPU) | coherent text, see `cfi-run-2026-11-09/` |

On this box the GPU suite set `NVRTC_SHIM_CLANG` and `CFI_STOCK_GGML_DLL`. The 4 environment-only failures on the B390
would therefore have been real comparisons, but nothing got past `cuInit`.

## Triage of the differences

All 116 GPU tests and all 787 ptx_tests cases have one cause, and it is not a translator or driver-API assumption:

- `CUDA_FOR_INTEL_TRACE=1`: `zeInit -> 0x78000001` (ZE_RESULT_ERROR_UNINITIALIZED: the loader found no driver), then `cuInit -> 100`.
- Device Manager: problem code 39, problem status 0xC00000E5. System log: Kernel-PnP event 219, "the driver
  \Driver\igfxn failed to load", at installation and at each of the two boots since.
- CodeIntegrity/Operational event 3004 for `igdkmdn64.sys`: "unable to verify the image integrity ... file hash could not be
  found". The file carries a valid Intel Authenticode signature (Sectigo CA), but kernel code integrity requires a
  Microsoft-signed (WHQL or attestation) catalog, so this looks like an engineering driver package. The same 3004 rejection
  hits the board's other Intel kernel drivers: the ME interface (`TeeDriverW10x64.sys`), the NPU (`npu_kmd.sys`) and the
  Smart Sound bus (`IntcAudioBus.sys`), all code 39.
- The first box also runs with HVCI on and loads its driver, so HVCI alone is not the cause. The unsigned-for-kernel
  package is.
- The Level Zero loader (`ze_loader.dll` 1.16.15) and the user-mode `ze_intel_gpu64.dll` (23.20.101.5564) are present, and
  the adapter's `LevelZeroDriverPath` is registered. Only the kernel driver is missing.

The only code issue the box exposed: `cfi-doctor.ps1` and `cfi.ps1 check` reported this adapter as a healthy Intel GPU with
driver 32.0.101.5564 and did not mention the problem code (fix in its own PR: `gpu-device-status` row, `check` fails).

## To make this box useful as an old-driver Xe2 target

This needs a decision by the box owner, because each option changes the box:
1. Install a Microsoft-signed build of an older Lunar Lake driver. This changes the driver, which was off-limits for this
   bring-up.
2. Keep 32.0.101.5564 and allow it to load: test signing (`bcdedit /set testsigning on`, usually with Secure Boot off),
   then a reboot. This weakens platform security.
Then rerun: `gpuhw.ps1 -Dir main -Tag hw` (in `cfi-work`) and `scripts\hwvalidate\ptx_tests.ps1`. The build, the ptx_tests
binary, Ollama, the model and LLVM are already staged.
