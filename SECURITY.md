# Security

## Reporting

Please report vulnerabilities privately through GitHub's "Report a vulnerability" (Security tab,
private advisories) rather than a public issue. Do not include exploit details in public issues.
This is an experimental, single-maintainer-scale project; expect a best-effort response.

## What you are installing

- `nvcuda.dll`, `cublas64_12.dll` (also copied as `cublasLt64_12.dll`) and optionally `cudart64_12.dll`
  replace NVIDIA's libraries **for the processes that load them from a directory you control**. They run
  with the privileges of the application. They are unsigned, experimental code that parses PTX and
  talks to the Intel driver; do not load untrusted PTX or run it on production machines.
- `scripts\cfi.ps1 install` writes only inside its install directory (default under `%LOCALAPPDATA%`),
  never into System32 or an existing Ollama/CUDA directory, never changes machine or user environment
  variables, backs up any file it would replace, and records SHA-256 hashes in `manifest.json`.
  `uninstall` reverses it. Read the script before running it: it is about 200 lines of PowerShell.
- Downloads: the installer only downloads when you pass `-ZipUrl` (https only) and it refuses the archive
  unless it matches the SHA-256 you pass with `-ZipSha256`.
- Third-party files (`nvcudart_hybrid64.dll`, Ollama) are never bundled; setup scripts verify pinned hashes.
- Verify any release you download against the SHA-256 list in its release notes.

## Privacy of diagnostics

`scripts\cfi-doctor.ps1` is read-only and redacts user name, computer name, profile path, IPv4 addresses
and e-mail-like strings. It lists GPU model, driver version, DLL hashes and the `CFI_*` switches. It
makes no network connection. Review it before posting.
