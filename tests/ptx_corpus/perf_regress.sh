#!/bin/sh
# SPIR-V hash regression + timing over PTX dumps (not committed: third-party PTX stays local).
#   perf_regress.sh OUT.tsv [THREADS] FILE.ptx...   write "<file>\t<entry>\t<hash|ERR ...>" lines
#   cmp: diff two OUT.tsv files; any difference means the SPIR-V changed.
# MODE=lazy hashes through the runtime's per-entry path (translate_entry) instead of parse_module.
set -e
here=$(cd "$(dirname "$0")" && pwd)
cargo build --release --quiet --manifest-path "$here/harness/Cargo.toml"
bin="$here/harness/target/release/ptx-corpus-harness"
out=$1; threads=${2:-1}; shift 2
: > "$out.tmp"
for f in "$@"; do
  b=$(basename "$f")
  start=$(date +%s)
  "$bin" --time "$f" "$threads" 2>/dev/null | cut -f1,2 | awk -v b="$b" -F'\t' '{print b "\t" $1 "\t" $2}' >> "$out.tmp"
  echo "$b $(( $(date +%s) - start ))s" >&2
done
LC_ALL=C sort "$out.tmp" > "$out"; rm -f "$out.tmp"
echo "wrote $out ($(wc -l < "$out") entries)"
