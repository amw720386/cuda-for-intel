# Release checklist

Release candidates are cut from a branch whose CI is green. Hardware steps need an Intel Arc machine and are done by hand.

## 1. Without a GPU (also runs in CI: `.github/workflows/ci.yml`)

- [ ] `cargo build --locked --workspace` (Windows and Linux)
- [ ] `cargo test --locked --lib --workspace` (includes the SPIR-V simulator and translator tests)
- [ ] `cargo test --locked -p cublas_shim` (XMX interpreter and generators)
- [ ] `tests/ptx_corpus/run.sh` reports no change versus `expected_status.tsv`
- [ ] `cargo clippy` and `cargo fmt --check` (advisory in CI until the backlog is cleared; do not add new findings)
- [ ] `bash scripts/scan-leaks.sh` clean, and `scripts/scan-leaks.sh staged` before each commit
- [ ] PowerShell scripts parse (CI job `powershell-syntax`); installer `-DryRun` leaves no files

## 2. On the Arc box (hand run, take the GPU lock)

- [ ] `cargo build --release --locked -p cuda-for-intel -p cublas_shim`
- [ ] `cargo test --release --locked --test gpu -- --ignored --nocapture --test-threads=1`
- [ ] Fresh directory: `pwsh -File scripts\cfi.ps1 install -InstallDir <scratch>`: self-test PASS
- [ ] `cfi.ps1 uninstall` restores a pre-existing file (place a dummy `nvcuda.dll` first, check the `.bak-*` round trip) and leaves the directory clean
- [ ] `scripts\cfi-doctor.ps1` output contains no user name, computer name or IP (grep it)
- [ ] Ollama v0.12.6 recipe from `docs/ollama.md`: `ollama-cfi.exe ps` shows 100% GPU and the model answers
- [ ] Record the logs under `docs/hw-logs/` (paths scrubbed)

## 3. Documents and artefacts

- [ ] README, `docs/compatibility.md` state only what the logs support; the tested-GPU list is correct
- [ ] Version in `Cargo.toml` and `TRANSLATOR_VERSION` (bump when translator output can change); the `build_info` unit test passes
- [ ] Release zip contains `nvcuda.dll`, `cublas64_12.dll`, `LICENSE`, no third-party binaries (no `nvcudart_hybrid64.dll`, no Ollama)
- [ ] Release notes list the SHA-256 of the zip and of each DLL (installer `-ZipSha256` uses it)
- [ ] `scan-leaks.sh` on the release tree; no IPs, hosts, user paths in logs or notes
- [ ] Tag and push; never force-push a shared branch
