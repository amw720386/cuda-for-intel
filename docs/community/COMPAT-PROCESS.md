# Compatibility report process (DRAFT)

The matrix itself lives in [docs/compatibility.md](../compatibility.md), which links back here for how to report. This page describes how reports reach it.

## For reporters
1. Run the app per its guide (docs/ollama.md, docs/pytorch.md, ...) on a released version or a named commit.
2. Open an issue with label `kind:compat-report` (or a PR adding a file) using the YAML below. Include the output of `pwsh -File scripts\cfi-doctor.ps1` verbatim (it redacts user name, computer name, profile path and IPv4 addresses; read it once anyway). The issue template `compatibility_report.yml` asks for it.
3. Report failures too, with the first error. Do not attach NVIDIA files or SASS.

## Report file: `docs/compat/reports/<date>-<app>-<gpu>.yml`
```yaml
schema: 1
app: Ollama            # name
app_version: "0.12.6"
workload: "qwen2.5:0.5b q4_K_M, 32 tokens"   # model / test
project_version: "0.1.0"   # release or commit
gpu: "Intel Arc B390"
driver: "32.0.101.9033"
os: "Windows 11 24H2"
result: pass           # pass | partial | fail
detail: "100% GPU, correct output"
doctor: |
  <cfi-doctor.ps1 output, scrubbed>
log: docs/hw-logs/<file>   # optional, maintainers' own runs
reported_by: "<github handle>"
verified: false        # maintainers set true after a repro or a log they trust
```
Allowed `result` values: pass, partial, fail. No IPs, hostnames, user names or paths in any field.

## For maintainers
1. Check scrubbing and that the `cfi-doctor.ps1` output matches the claimed GPU/driver (leak scan).
2. Reproduce on the reference device or get a second independent report; then set `verified: true`.
3. Only `verified: true` and `result: pass` rows enter the public "works" list in docs/compatibility.md; `partial`/`fail` go in a "known issues" section. Unverified reports stay in the repo, shown as "community-reported".
4. Render: `python3 scripts/community/render_compat.py docs/compat/reports > matrix.md` (add `--verified-only` for the public list), paste into docs/compatibility.md in the same PR. Files starting with `_` (such as `_example.yml`) are templates and are skipped; to smoke-test the renderer copy the example to a scratch directory under a name without the underscore and point the script at that directory.
5. Old reports are never deleted; a newer report for the same app+gpu+version supersedes in the rendering.
