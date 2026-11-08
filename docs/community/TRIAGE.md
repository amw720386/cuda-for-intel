# Issue triage policy (DRAFT)

## What a bug report must contain
1. Output of the environment check (the "doctor output"). Today that is: the result of `scripts\check-install.ps1`, plus GPU model, Intel driver version, Windows/Linux version, project commit or release, and the relevant `CUDA_FOR_INTEL_*` variables you set. If a `doctor` command is added later (open decision: none exists at c065736), it replaces this list; the template should ask for its output verbatim.
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
Out of scope (other GPUs, NVIDIA hardware issues, apps that need unimplemented APIs without a kernel trace) -> `status:wontfix` with a reason and a pointer to docs/compatibility.md.
Hardware-only bugs: maintainers confirm on the test device or ask a second reporter; label `hw:` accordingly.
