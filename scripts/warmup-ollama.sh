#!/bin/sh
# Fills the CUDA for Intel kernel caches for a model: a few tokens at several prompt sizes
# against a running Ollama server (docs/first-run-latency.md).
# usage: warmup-ollama.sh MODEL [SERVER_URL]
set -e
model=${1:?usage: warmup-ollama.sh MODEL [SERVER_URL]}
server=${2:-http://127.0.0.1:11434}
for words in 4 64 512; do
  prompt=$(yes warm | head -n "$words" | tr '\n' ' ')
  start=$(date +%s)
  curl -sS --max-time 7200 "$server/api/generate" \
    -d "{\"model\":\"$model\",\"prompt\":\"$prompt\",\"stream\":false,\"options\":{\"num_predict\":8,\"temperature\":0}}" >/dev/null
  echo "$words prompt words: $(( $(date +%s) - start ))s"
done
