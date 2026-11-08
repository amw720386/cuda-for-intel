#!/bin/sh
# Translate every *.ptx below a corpus directory and print coverage statistics.
#   scripts/ptx-corpus-scan.sh <corpus-dir> [report.tsv]
# Extract PTX first (kept OUT of the repository; it is NVIDIA / third-party code):
#   CFI_STOCK_GGML_DLL=<ggml-cuda.dll> CFI_SCAN_DUMP_DIR=<dir> \
#     cargo test --release --lib scan_stock_dll_translation -- --ignored --nocapture
# torch/cuobjdump dumps and any other .ptx files can simply be placed (or symlinked) in <corpus-dir>.
# Set CFI_CORPUS_VAL=1 to also run spirv-val on every translated entry (slow).
set -e
dir=${1:?usage: ptx-corpus-scan.sh <corpus-dir> [report.tsv]}
out=${2:-ptx-corpus-report.tsv}
CFI_CORPUS_DIR="$dir" CFI_CORPUS_OUT="$out" \
  cargo test --release --lib corpus_scan -- --ignored
grep -a '^SUMMARY' "$out"
