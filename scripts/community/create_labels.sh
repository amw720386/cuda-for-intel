#!/usr/bin/env bash
# Create the issue labels from docs/community/LABELS.md with gh.
# Outward-facing: DRY RUN by default (prints the commands). Maintainers run: create_labels.sh --apply <owner/repo>
set -euo pipefail
APPLY=0; [ "${1:-}" = "--apply" ] && { APPLY=1; shift; }
REPO="${1:-amw720386/cuda-for-intel}"
labels=(
"area:translator|1d76db|PTX to SPIR-V translator"
"area:subst|1d76db|Kernel substitution layer"
"area:cublas|1d76db|cuBLAS / cuBLASLt shim"
"area:cudnn|1d76db|cuDNN shim"
"area:ollama|1d76db|Ollama integration"
"area:pytorch|1d76db|PyTorch integration"
"area:nvrtc|1d76db|NVRTC shim"
"area:docs|1d76db|Documentation"
"area:build-ci|1d76db|Build, packaging, CI"
"hw:arc-b390|5319e7|Intel Arc B390"
"hw:other-arc|5319e7|Other Intel Arc device"
"hw:other-intel|5319e7|Other Intel GPU"
"kind:bug|d73a4a|Something is broken"
"kind:feature|a2eeef|New capability"
"kind:question|d876e3|Usage question"
"kind:compat-report|0e8a16|Tested-app report"
"status:needs-repro|fbca04|Needs reproduction details (doctor output, logs)"
"status:needs-info|fbca04|Waiting for the reporter"
"status:confirmed|0e8a16|Reproduced by a maintainer"
"status:wontfix|ffffff|Out of scope"
"status:duplicate|cfd3d7|Duplicate"
"status:stale|cfd3d7|No activity"
"good-first-issue|7057ff|Bounded, newcomer friendly"
"help-wanted|008672|Contributions welcome"
"breaking-change|b60205|Changes the public contract"
"security|b60205|Handle privately"
"release-blocker|b60205|Blocks the next release"
)
for l in "${labels[@]}"; do
  IFS='|' read -r name color desc <<<"$l"
  cmd=(gh label create "$name" --color "$color" --description "$desc" --repo "$REPO" --force)
  if [ "$APPLY" = 1 ]; then "${cmd[@]}"; else printf '%q ' "${cmd[@]}"; echo; fi
done
[ "$APPLY" = 1 ] || echo "(dry run: pass --apply to create)" >&2
