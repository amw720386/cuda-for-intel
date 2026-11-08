# Issue triage policy (DRAFT)

## What a bug report must contain
1. The output of `pwsh -File scripts\cfi-doctor.ps1` (the "doctor output"), pasted whole. It reports OS build, GPU, driver, Level Zero loader, shim DLL hashes and translator version, and the `CUDA_FOR_INTEL_*` / `CFI_*` switches in effect, with user name, computer name, profile path and IPv4 addresses redacted. Both issue templates require it. Add the project commit or release if the doctor output does not show it.
2. App and version (e.g. Ollama 0.12.6, PyTorch build), the model/workload, exact command.
3. Expected vs actual, with a log from `CUDA_FOR_INTEL_LOG=1` (trim to the failing part, and remove user names, paths and addresses).
4. For wrong-result bugs: a minimal PTX or CUDA C file if possible.

Missing items: label `status:needs-repro`, post the saved reply. No attachments of NVIDIA binaries, SASS disassembly or model weights.

## Targets (best effort, volunteer project)
| Event | Target |
| --- | --- |
| First maintainer response | 7 days |
| Triage (labels, status) | 14 days |
| `needs-repro` / `needs-info` with no answer | stale after 30 days, closed after 14 more |
| Security reports | acknowledge in 3 days, via private channel only |

## Flow
new -> label kind/area/hw -> repro complete? yes: `status:confirmed` (+ `good-first-issue` if bounded) / no: `status:needs-repro` -> fix PR references the issue -> closed on merge.
Out of scope (other GPUs, NVIDIA hardware issues, apps that need unimplemented APIs without a kernel trace) -> `status:wontfix` with a reason and a pointer to docs/compatibility.md (reporting recipe: [COMPAT-PROCESS.md](COMPAT-PROCESS.md)).
Hardware-only bugs: maintainers confirm on the test device or ask a second reporter; label `hw:` accordingly.
