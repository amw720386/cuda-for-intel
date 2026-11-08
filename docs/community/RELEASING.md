# Releasing (DRAFT checklist; do not tag or publish until the owner says so)

Roles: a maintainer with access to the hardware test device and the GPU lock.

1. **Branch**: cut from the integration branch at a green commit; create `release/vX.Y.Z`.
2. **Version**: bump the five `Cargo.toml` versions together (see VERSIONING.md), update `Cargo.lock` (`cargo build --locked` must still pass after `cargo update -w`). Confirm the translator/cache version stamp was bumped if translator output changed.
3. **CHANGELOG**: move "Unreleased" to the new version with the date; keep only user-visible changes.
4. **Gate: software tests (no GPU)**: `cargo test --locked --workspace`, `tests/ptx_corpus/run.sh` (no diff in expected_status.tsv unless intended).
5. **Gate: hardware suite** on the reference device: `cargo test --release --locked --test gpu -- --ignored --nocapture --test-threads=1` and `scripts\validate-gpu.ps1`, holding the GPU lock. Save logs under `docs/hw-logs/` (paths scrubbed). Every result quoted in release notes must come from a log committed at the release commit. A failing test blocks the release unless the owner lists it as a known issue.
6. **Gate: leak scan**: a full-history scan (grep for IPs, `user-at-host`, private keys, tokens, Windows user paths) on the tree, logs and release notes must report 0.
7. **Gate: dependency and licence check**: `cargo deny check` (draft config: `docs/community/deny.toml.draft`), refresh `THIRD_PARTY_NOTICES`.
8. **Build artifacts** on Windows MSVC x64: `cargo build --release --locked -p cuda-for-intel -p cublas_shim` (plus other shims). Stage with `scripts\setup-windows.ps1`. Zip the staged directory with the licence files and THIRD_PARTY_NOTICES. Verify the archive contains no NVIDIA files (`nvcudart*`, `cudart64*`, `cublas64*`, `*.cu` samples, CUDA headers) and no `.pdb` with user paths.
9. **Checksums**: `Get-FileHash -Algorithm SHA256` (or `sha256sum`) for each artifact into `SHA256SUMS`; sign if the owner provides a key (open decision: signing).
10. **Smoke test the zip** on a clean machine or user profile: unzip, `scripts\check-install.ps1`.
11. **Release notes** from `release-notes-template.md`; only verified results; no earlier or failed runs.
12. **Tag** (owner only): `git tag -s vX.Y.Z -m "vX.Y.Z"` on the release commit, `git push <explicit-remote-url> vX.Y.Z`.
13. **GitHub release** (owner only): attach zip and SHA256SUMS; mark pre-release while 0.x if desired.
14. Merge `release/vX.Y.Z` back; open "Unreleased" in CHANGELOG; update docs/compatibility.md from accepted compat reports.

Never: force-push a tag, push to `main` directly, attach a binary not produced by step 8.
