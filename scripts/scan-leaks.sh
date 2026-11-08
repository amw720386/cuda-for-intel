#!/usr/bin/env bash
# Leak scan for files that would be published: IPv4 addresses, user@host strings, private keys,
# cloud/API token prefixes, private host names, real Windows user names in C:\Users\<name>.
# Used by CI and by the release checklist. Works with GNU and BSD (macOS) grep/sed: no \b, no GNU-only flags.
# Usage: scripts/scan-leaks.sh            scan all tracked files (git ls-files)
#        scripts/scan-leaks.sh staged     scan only the staged diff additions
# Lines containing the marker `leak-ok` are ignored (use for documentation placeholders).
# Ignored without a marker: loopback/any-address, Intel driver versions (32.0.x.y), the placeholder
# profile names C:\Users\USER|user|<user>|Public|Default, semantic versions such as v0.1.0.
# Output is masked (file:line and pattern class only, never the matched text); exit 1 when anything is found.
set -u
B='(^|[^0-9A-Za-z_.])'   # left boundary (replaces \b)
E='($|[^0-9A-Za-z_])'    # right boundary
PAT_IP="${B}([0-9]{1,3}\.){3}[0-9]{1,3}${E}"
PAT_MAIL='[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}'
PAT_KEY='BEGIN [A-Z ]*PRIVATE|AKIA[0-9A-Z]{8,}|ghp_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9]{16,}'
PAT_HOST="dev[c]loud|pdx[8]8|${B}csc-[a-z]"
PAT_WINUSER='[A-Za-z]:\\{1,2}Users\\{1,2}[A-Za-z0-9._-]+'
PAT="${PAT_IP}|${PAT_MAIL}|${PAT_KEY}|${PAT_HOST}|${PAT_WINUSER}"

# Remove tokens that are known to be harmless before matching. A generic version rule is deliberately
# not used: it would hide real IPs. Semantic versions such as v0.1.0 never match the IPv4 pattern anyway.
strip_ok() {
  sed -E \
    -e 's/127\.0\.0\.1|0\.0\.0\.0|255\.255\.255\.[0-9]+//g' \
    -e 's/32\.0\.[0-9]+\.[0-9]+//g' \
    -e 's/C:\\{1,2}Users\\{1,2}(USER|user|<user>|%USERNAME%|Public|Default|All Users)([^A-Za-z0-9._-]|$)/PLACEHOLDER\2/g'
}

if [ "${1:-}" = staged ]; then
  raw=$(git diff --cached --no-color -U0 --diff-filter=AMR | awk '
    /^\+\+\+ b\// { f=substr($0,7); n=0; next }
    /^\+\+\+/ { next }
    /^@@/ { match($0,/\+[0-9]+/); n=substr($0,RSTART+1,RLENGTH-1)-1; next }
    /^\+/ { n++; print f ":" n ":" substr($0,2) }')
else
  raw=$(git ls-files -z | xargs -0 grep -InE "${PAT_IP}|${PAT_MAIL}|${PAT_KEY}|${PAT_HOST}|${PAT_WINUSER}" 2>/dev/null \
        | grep -vE '(^|/)Cargo\.lock:|^tests/ptx_corpus/' || true)
fi
hits=$(printf '%s\n' "$raw" | grep -v 'leak-ok' | strip_ok | grep -E "$PAT" || true)
# The scanner's own pattern definitions are not leaks.
hits=$(printf '%s\n' "$hits" | grep -vE '^scripts/scan-leaks\.sh:' | grep -v '^$' || true)

if [ -n "$hits" ]; then
  echo "leak-scan: possible leaks found (file:line only, text withheld):" >&2
  printf '%s\n' "$hits" | cut -d: -f1,2 | sed -E 's/^/  /' >&2
  echo "leak-scan: FAIL ($(printf '%s\n' "$hits" | wc -l | tr -d ' ') lines)" >&2
  exit 1
fi
echo "leak-scan: clean"
