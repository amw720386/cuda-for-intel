"""Warm prefill/decode tok/s against any running Ollama server, using exactly the request shape of
scripts/bench-decode.ps1, so stock Vulkan (or CPU) baselines compare directly with the CUDA-path numbers.

The prompt is calibrated to 631 tokens; one 8-token warm-up (variant 2) is followed by RUNS measured
requests on fresh variants (3, 4, ...) that the KV prefix cache has never seen; 48 greedy tokens each.
Verify placement yourself: the server log must show every layer offloaded, and the script prints
/api/ps size vs size_vram.

usage: python -I scripts/bench-decode-vulkan.py HOST:PORT MODEL RUNS OUT.json
"""
import json
import statistics
import sys
import urllib.request

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


words = 600 + (631 - gen(long_prompt(600, 1), 1)["prompt_eval_count"])
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
      f"decode p50 {statistics.median(decode):.1f} min {min(decode):.1f} max {max(decode):.1f} ps {rec['ps']}")
