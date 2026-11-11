#!/bin/sh
# Cold-cache PTX -> SPIR-V translation benchmark through cfi-precompile (no GPU needed).
#
#   scripts/profile/bench-translate.sh BIN_DIR INPUT OUT_DIR TAG [RUNS] [THREADS]
#
#   BIN_DIR  folder with nvcuda.dll and cfi-precompile.exe (e.g. target/release)
#   INPUT    file or folder to translate, e.g. <ollama-0.12.6>/lib/ollama/cuda_v13
#   OUT_DIR  receives TAG_rN.log (summary) and TAG_rN.tsv (--report)
#
# Every run starts from an empty SPIR-V cache in OUT_DIR/cache-TAG. After each run the script prints the wall
# time and a digest of the cache contents: equal digests mean byte-identical SPIR-V, so a performance change
# that alters the translator output is visible immediately. Use Git Bash on Windows. Run on an otherwise idle
# machine; on an iGPU box this also loads the memory bus that GPU benchmarks share.
set -u
bin=$1; input=$2; out=$3; tag=$4; runs=${5:-3}; threads=${6:-}
mkdir -p "$out"
cache="$out/cache-$tag"
targs=""; [ -n "$threads" ] && targs="--threads $threads"
w() { cygpath -w "$1" 2>/dev/null || printf '%s' "$1"; }
i=1
while [ "$i" -le "$runs" ]; do
  rm -rf "$cache"; mkdir -p "$cache"
  s=$(date +%s%N)
  CFI_CACHE_DIR="$(w "$cache")" "$bin/cfi-precompile.exe" --install-dir "$(w "$bin")" --translate-only --quiet $targs \
    --report "$(w "$out/${tag}_r$i.tsv")" "$(w "$input")" > "$out/${tag}_r$i.log" 2>&1
  rc=$?
  e=$(date +%s%N)
  digest=$(cd "$cache" && find . -type f -name '*.xt' | LC_ALL=C sort | xargs sha1sum | sha1sum | cut -c1-16)
  echo "$tag run $i rc=$rc wall_ms=$(( (e - s) / 1000000 )) cache_mb=$(du -sm "$cache" | cut -f1) digest=$digest"
  i=$((i + 1))
done
rm -rf "$cache"
