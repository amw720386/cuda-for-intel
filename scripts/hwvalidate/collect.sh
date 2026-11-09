#!/usr/bin/env bash
# Fetch the validation result with ONE scp: run_all.ps1 leaves results\collect.tgz (results.json, summary.md, stage*.json and
# logs under 256 KB) after every run, including failed and interrupted ones.
#   CFI_BOX=<ssh alias> scripts/hwvalidate/collect.sh [local_dir]
set -euo pipefail
BOX="${CFI_BOX:?set CFI_BOX to the ssh alias of the GPU box (see ~/.ssh/config)}"
REMOTE_ROOT="${CFI_REMOTE_ROOT:-cfi-work/hwval}"
dest="${1:-$HOME/cuda-intel-work/hwval-results/$(date +%Y%m%d-%H%M%S)}"
mkdir -p "$dest"
for i in 1 2 3 4 5 6; do
  if scp -q -o BatchMode=yes -o ConnectTimeout=25 "$BOX:$REMOTE_ROOT/results/collect.tgz" "$dest/collect.tgz" 2> "$dest/scp.err" && tar tzf "$dest/collect.tgz" > /dev/null 2>&1; then break; fi
  echo "attempt $i failed, waiting 15s" >&2
  [ "$i" = 6 ] && { echo "giving up: see $dest/scp.err (no collect.tgz yet? run_all.ps1 writes it at the end of every run)" >&2; exit 1; }
  sleep 15
done
tar xzf "$dest/collect.tgz" -C "$dest" && rm -f "$dest/collect.tgz" "$dest/scp.err"
echo "results in $dest"
[ -f "$dest/summary.md" ] && sed -n '1,45p' "$dest/summary.md"
exit 0
