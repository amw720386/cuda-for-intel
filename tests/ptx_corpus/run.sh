#!/bin/sh
# Run the corpus through the translator and diff against expected_status.tsv.
#   run.sh            compare (exit 1 on any change)
#   run.sh --bless    rewrite expected_status.tsv
# Builds a scratch harness that includes src/ptx.rs (the main crate does not build on macOS).
set -e
here=$(cd "$(dirname "$0")" && pwd)
cargo build --release --quiet --manifest-path "$here/harness/Cargo.toml"
tmp=$(mktemp)
"$here/harness/target/release/ptx-corpus-harness" "$here" | LC_ALL=C sort > "$tmp"
if [ "$1" = "--bless" ]; then cp "$tmp" "$here/expected_status.tsv"; echo blessed; fi
cut -f1,2 "$tmp" > "$tmp.cur"; cut -f1,2 "$here/expected_status.tsv" > "$tmp.exp"
awk -F'\t' '{c[$2]++} END{for(k in c) printf "%s=%d ", k, c[k]; print ""}' "$tmp"
diff "$tmp.exp" "$tmp.cur" && echo "corpus matches expected_status.tsv"
