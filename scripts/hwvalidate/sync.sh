#!/usr/bin/env bash
# Push the current tree (git HEAD, tracked files only) to the GPU box in ONE transfer, optionally start the validation.
#   CFI_BOX=<ssh alias from ~/.ssh/config> scripts/hwvalidate/sync.sh [--run [run_all.ps1 args...]]
# The alias comes from the environment only; nothing machine-specific is stored in the repo.
# Two ssh commands share ONE TCP connection (ControlMaster) so the flaky box sees a single login.
set -euo pipefail
BOX="${CFI_BOX:?set CFI_BOX to the ssh alias of the GPU box (see ~/.ssh/config)}"
REMOTE_ROOT="${CFI_REMOTE_ROOT:-cfi-work/hwval}"          # relative to the box user's home (ssh default dir)
repo="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$repo"
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then echo "refusing: tracked files have uncommitted changes (commit first, the box must run a pushed revision)" >&2; exit 2; fi
rev="$(git rev-parse --short HEAD)"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
git archive --format=tar HEAD > "$tmp/src.tar"
echo "$rev $(git rev-parse --abbrev-ref HEAD)" > "$tmp/REVISION"
(cd "$tmp" && tar rf src.tar REVISION)
gzip -1 "$tmp/src.tar"
cp="$tmp/cm"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=25 -o ServerAliveInterval=20 -o ControlMaster=auto -o ControlPath="$cp" -o ControlPersist=120 "$BOX")
retry() { local i out; for i in 1 2 3 4 5 6; do if "$@"; then return 0; fi; echo "ssh attempt $i failed, waiting 15s" >&2; sleep 15; done; return 1; }
win_root="${REMOTE_ROOT//\//\\}"
SCP=(scp -q -o BatchMode=yes -o ConnectTimeout=25 -o ControlMaster=auto -o ControlPath="$cp" -o ControlPersist=120)
# 1 file over scp (binary safe regardless of the box's default shell), 1 command to unpack; both ride the same TCP connection.
# The parentheses matter: in cmd, `if exist X a & b & c` makes b and c part of the if body, so a fresh box unpacked nothing.
retry "${SCP[@]}" "$tmp/src.tar.gz" "$BOX:$REMOTE_ROOT-src.tgz"
retry "${SSH[@]}" "cmd /c \"(if exist $win_root\\src rmdir /s /q $win_root\\src) & mkdir $win_root\\src & tar xzf $win_root-src.tgz -C $win_root\\src & del $win_root-src.tgz\""
echo "synced revision $rev to the box ($(du -h "$tmp/src.tar.gz" | cut -f1) transferred)"
if [ "${1:-}" = "--run" ]; then
  shift
  echo "starting run_all.ps1 (foreground; if the link drops, rerun: finished stages are skipped)"
  "${SSH[@]}" "powershell -NoProfile -ExecutionPolicy Bypass -File $win_root\\src\\scripts\\hwvalidate\\run_all.ps1 $*" || echo "run_all exited non-zero or the link dropped; use collect.sh to see what finished" >&2
fi
ssh -o ControlPath="$cp" -O exit "$BOX" 2>/dev/null || true
