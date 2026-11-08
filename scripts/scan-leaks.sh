#!/usr/bin/env bash
# Leak scan for files that would be published: IPv4 addresses, user@host strings, private keys,
# cloud/API token prefixes, private host names. Used by CI and by the release checklist.
# Usage: scripts/scan-leaks.sh            scan all tracked files (git ls-files)
#        scripts/scan-leaks.sh staged     scan only the staged diff additions
# Lines containing the marker `leak-ok` are ignored (use for documentation placeholders).
# Output is masked; exit 1 when anything is found.
set -u
PAT='\b([0-9]{1,3}\.){3}[0-9]{1,3}\b|[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}|BEGIN [A-Z ]*PRIVATE|AKIA[0-9A-Z]{8,}|ghp_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{16,}|USER|\bcsc-[a-z]|[A-Za-z]:\\Users\\[A-Za-z0-9._-]+'
mask() { sed -E 's/(.{0,6})[^ ]{0,40}/\1.../' ; }
if [ "${1:-}" = staged ]; then
  hits=$(git diff --cached --no-color -U0 --diff-filter=AMR | grep -E '^\+[^+]' | grep -v 'leak-ok' | grep -nE "$PAT" || true)
else
  # shellcheck disable=SC2046
  hits=$(git ls-files -z | xargs -0 grep -InE "$PAT" 2>/dev/null | grep -v 'leak-ok' | grep -vE '(^|/)Cargo\.lock:|^tests/ptx_corpus/' | sed -E 's/127\.0\.0\.1|0\.0\.0\.0//g;s/C:\\Users\\(USER|user|<user>|Public|Default)\b//g' | grep -E "$PAT" || true)
fi
if [ -n "$hits" ]; then
  echo "leak-scan: possible leaks found (masked):" >&2
  echo "$hits" | cut -c1-60 | sed -E 's/[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}/<ip>/g' >&2
  echo "leak-scan: FAIL ($(echo "$hits" | wc -l | tr -d ' ') lines)" >&2
  exit 1
fi
echo "leak-scan: clean"
