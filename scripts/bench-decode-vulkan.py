"""Warm prefill/decode tok/s against any running Ollama server, using exactly the request shape of
scripts/bench-decode.ps1, so stock Vulkan (or CPU) baselines compare directly with the CUDA-path numbers.

The prompt is calibrated to 631 tokens; one 8-token warm-up (variant 2) is followed by RUNS measured
requests on fresh variants (3, 4, ...) that the KV prefix cache has never seen; 48 greedy tokens each.
Verify placement yourself: the server log must show every layer offloaded, and the script prints
/api/ps size vs size_vram.

The run fails (exit 2) unless every measured request saw exactly 631 prompt tokens, so a tokenizer or
model change cannot silently alter the request shape the CUDA numbers are compared against.

usage: python -I scripts/bench-decode-vulkan.py HOST:PORT MODEL RUNS OUT.json
       python -I scripts/bench-decode-vulkan.py --pool LABEL=A.json,B.json,... [LABEL=...]
           pools the runs of several servers per label and prints the table row (n, prefill p50,
           decode p50 / p95 / best, per-server decode p50), e.g. from docs/hw-logs/vulkan-baseline-2026-11-10/
"""
import json
import statistics
import sys
import urllib.request

PROMPT_TOKENS = 631


def quantile(values, q):
    """Nearest-rank on the sorted values (index round(q * (n - 1))), as used in the hw-log tables."""
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(q * (len(ordered) - 1)))]


if len(sys.argv) > 1 and sys.argv[1] == "--pool":
    print("| label | n | prefill p50 | decode p50 | decode p95 | decode best | per-server decode p50 |")
    print("|---|---|---|---|---|---|---|")
    for spec in sys.argv[2:]:
        label, files = spec.split("=", 1)
        prefill, decode, servers = [], [], []
        for path in files.split(","):
            with open(path, encoding="utf-8") as f:
                runs = json.load(f)["runs"]
            prefill += [r["prefill"] for r in runs]
            decode += [r["decode"] for r in runs]
            servers.append(statistics.median(r["decode"] for r in runs))
        print(f"| {label} | {len(decode)} | {statistics.median(prefill):.0f} | {statistics.median(decode):.1f} | "
              f"{quantile(decode, 0.95):.1f} | {max(decode):.1f} | {' / '.join(f'{x:.1f}' for x in servers)} |")
    sys.exit(0)

host, model, runs, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]


def gen(prompt, n, num_gpu=99):
    body = {"model": model, "prompt": prompt, "stream": False, "raw": True,
            "options": {"num_predict": n, "temperature": 0, "seed": 0, "top_k": 1, "repeat_penalty": 1.0,
                        "num_gpu": num_gpu, "num_ctx": 2048, "num_batch": 512}}
    req = urllib.request.Request(f"http://{host}/api/generate", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=3600).read())


def long_prompt(words, variant):
    return f"{variant} " + " ".join(["lorem"] * words) + "\nCount from 1 to 300 separated by commas: 1, 2, 3,"


words = 600 + (PROMPT_TOKENS - gen(long_prompt(600, 1), 1)["prompt_eval_count"])
gen(long_prompt(words, 2), 8)
rec = {"model": model, "words": words, "runs": []}
for i in range(1, runs + 1):
    r = gen(long_prompt(words, 2 + i), 48)
    rec["runs"].append({"prompt_tokens": r["prompt_eval_count"], "gen_tokens": r["eval_count"],
                        "prefill": r["prompt_eval_count"] * 1e9 / r["prompt_eval_duration"],
                        "decode": r["eval_count"] * 1e9 / r["eval_duration"], "text": r["response"]})
ps = json.loads(urllib.request.urlopen(f"http://{host}/api/ps").read())["models"]
rec["ps"] = [{k: m[k] for k in ("name", "size", "size_vram")} for m in ps]
with open(out, "w") as f:
    json.dump(rec, f, indent=1)
prefill = [x["prefill"] for x in rec["runs"]]
decode = [x["decode"] for x in rec["runs"]]
print(f"prompt_tokens={rec['runs'][0]['prompt_tokens']} prefill p50 {statistics.median(prefill):.1f} "
      f"decode p50 {statistics.median(decode):.1f} p95 {quantile(decode, 0.95):.1f} min {min(decode):.1f} "
      f"max {max(decode):.1f} ps {rec['ps']}")
wrong = sorted({x["prompt_tokens"] for x in rec["runs"]} - {PROMPT_TOKENS})
if wrong:
    sys.exit(f"measured requests saw {wrong} prompt tokens, not {PROMPT_TOKENS}: the request shape changed")
