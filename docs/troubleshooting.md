# Troubleshooting and diagnostics

Start with the automatic checks, then the switches below.

```powershell
pwsh -File scripts\cfi.ps1 check      # prerequisites
pwsh -File scripts\cfi.ps1 selftest   # vector add through the shim: PASS or FAIL with a next step
pwsh -File scripts\cfi-doctor.ps1     # redacted report for bug reports
```

## Logs and switches

Set switches in the shell that starts the application. Boolean switches are on at `1` unless noted.
The complete table with defaults is in the README ("Environment variables").

| Switch | Use it to |
| --- | --- |
| `CUDA_FOR_INTEL_LOG=1` | Print driver diagnostics (translation and Level Zero errors, kernel sizes) to stderr and write `cuda-for-intel-export.log` in the working directory. First thing to try. |
| `CUDA_FOR_INTEL_TRACE=1` | Log every API result. Add `CUDA_FOR_INTEL_TRACE_ENTER=1` for entries too. Verbose and slow. |
| `CUDA_FOR_INTEL_SYNC_LAUNCH=1` | Wait after each launch so a device fault is attributed to the right kernel. Debugging only. |
| `CFI_LOG=1` | Log lines of the cudart and cuBLAS shim DLLs. |
| `CFI_CUBLAS_TRACE=1` | Trace cuBLASLt plan failures. |
| `CFI_PTX_DUMP_DIR=<dir>` | Write the PTX as supplied, the prepared PTX and failing modules. Attach the failing PTX to a translator bug (it is program code; check you may share it). |
| `CFI_DEBUG_COPYBACK=<selector>` | Wrong numbers? After selected launches (`N`, `A-B`, `n:<name>|<name>`, `all`) the shim copies every pointer argument back and logs non-finite counts, min/max and a checksum. `CFI_DEBUG_COPYBACK_MAX=<bytes>` caps the size, `CFI_DEBUG_COPYBACK_DUMP=<dir>` also writes the buffers. Run twice with different switches and compare the checksum lines to find the kernel that diverges. Only names, sizes and numbers are logged unless you set the dump directory. |
| `CFI_XMX=0` or `2` | `0`: never use the native XMX GEMM (compare results or isolate a GEMM bug). `2`: force XMX for every qualifying call. Unset: automatic for large fp16/bf16 GEMMs. |
| `CUDA_FOR_INTEL_SUBST=none` | Disable native kernel substitution and run the plain translated PTX. `auto` (default when unset) is the measured policy; `all` forces every native kernel; a comma list selects families, `-mmq` style entries remove one. If output is wrong with `auto` and right with `none`, report it with the model name. |
| `CFI_NO_CACHE=1`, `CUDA_FOR_INTEL_CACHE=0` | Disable the translation cache / the compiled-binary cache, to rule out a stale cache. |

## Caches and TRANSLATOR_VERSION

Two caches exist, both safe to delete:

- Translation cache (`CFI_CACHE_DIR`, default `%LOCALAPPDATA%\cuda-for-intel\spvcache`): PTX to SPIR-V results.
  Its key contains the PTX hash, entry name, the crate version, the internal `TRANSLATOR_VERSION`
  and the `CFI_*` translator switches. The version is bumped whenever the translator output for the same PTX can change,
  so an upgraded DLL never reads results of an older translator. The doctor prints it as `translator=N` for `nvcuda.dll`.
- Native cache (`CUDA_FOR_INTEL_CACHE_DIR`, default `%LOCALAPPDATA%\cuda-for-intel\cache\v1`, capped by
  `CUDA_FOR_INTEL_CACHE_MAX_MB`, default 4096): compiled GPU binaries.

If you suspect a stale cache: set `CFI_NO_CACHE=1 CUDA_FOR_INTEL_CACHE=0` for one run; if that fixes it, delete
both directories and report the doctor output. `cfi.ps1 install` puts the caches inside its install directory.

## Common failures

See the README Troubleshooting table for error codes (209, 218, 301, 3/100, 719, 2, 801). Short list:

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Self-test SKIPPED: Level Zero loader missing | No or old Intel graphics driver | Install the current driver from Intel, reboot |
| Self-test FAIL with 100 or 3 | No GPU visible (remote desktop, VM, hybrid laptop on the wrong GPU) | Run locally; check Device Manager |
| Application loads NVIDIA's `nvcuda.dll` instead | Environment not loaded | Dot-source `cfi-env.ps1` in the shell that starts the app; check `where.exe nvcuda.dll` |
| Error 719 or "device lost" | Windows GPU timeout (TDR) | Close other GPU programs; `CUDA_FOR_INTEL_SYNC_LAUNCH=1` to find the kernel |
| Very slow first request | Kernel compilation | Wait once; keep the caches |

## How to report a bug

1. Run `pwsh -File scripts\cfi-doctor.ps1` and paste the output into the issue template.
2. Say what you ran (application, exact version, model) and what you expected.
3. Include the first error from a run with `CUDA_FOR_INTEL_LOG=1` (the relevant lines, not the whole log).
4. Do not include host names, user names, IP addresses, keys or model files. The doctor redacts the common ones;
   logs you copy yourself are not redacted.
5. If the failure is wrong output, say whether `CUDA_FOR_INTEL_SUBST=none` and `CFI_XMX=0` change it.

## Uninstall

`pwsh -File scripts\cfi.ps1 uninstall` removes only the files the installer placed (matching hashes) and restores
backups. `-DryRun` shows what would happen.
