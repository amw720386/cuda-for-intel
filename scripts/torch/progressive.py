#!/usr/bin/env python3
"""Progressive PyTorch test ladder for the cuda-for-intel shim (torch 2.2.2+cu118 wheel, compute_37 PTX).

Every level runs in its own subprocess with a timeout, so a hang or a lost device in level N
does not hide levels N+1... The parent writes one JSON file per level plus results.json and a
markdown matrix that can be pasted into docs/pytorch-status.md.

  L0 import + device query        L4 conv2d fwd/bwd
  L1 create/copy/elementwise      L5 TransformerEncoderLayer fwd/bwd
  L2 matmul f32/f16 vs CPU        L6 small HF model greedy generation vs CPU
  L3 softmax/layernorm/reductions L7 SDPA variants (default, CFI_REPORT_CC=3.7, math only)

usage:  progressive.py [--levels L0,L1,...] [--timeout SEC] [--out DIR] [--stop-on-fail]
env:    CFI_FAKE_CPU=1    use device 'cpu' (exercises the harness logic without a GPU)
        CFI_NVCUDA        path of the shim's nvcuda library, loaded before `import torch`
        CFI_HF_MODEL      local directory of a HF causal LM (GPT-2 or Qwen2.5-0.5B) for L6
        CFI_L6_TOKENS     tokens to generate in L6 (default 32)
        CFI_L6_PROMPT     prompt for L6 (CFI_L6_IDS=1,2,3 gives raw token ids when no tokenizer files exist)
Exit code: 0 all requested levels pass or skip, 1 any fail/error/timeout.
Child protocol: the child prints one line `CFIRESULT {json}` as its last stdout line.
"""
import argparse
import json
import os
import subprocess
import sys
import time

LEVELS = ["L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7"]
TITLES = {
    "L0": "import + device query", "L1": "tensor create/copy/elementwise", "L2": "matmul f32/f16 vs CPU",
    "L3": "softmax/layernorm/reductions", "L4": "conv2d fwd/bwd", "L5": "TransformerEncoderLayer fwd/bwd",
    "L6": "HF greedy generation vs CPU", "L7": "SDPA variants",
}
SDPA_VARIANTS = ["default", "cc37", "math"]
RESULT_TAG = "CFIRESULT "


# ---------------------------------------------------------------- pure helpers (unit-testable)
def fake_cpu():
    return os.environ.get("CFI_FAKE_CPU") == "1"


def classify_error(text):
    """Map an exception text to a short error class used in the matrix."""
    t = text.lower()
    if "no kernel image" in t or "error 209" in t or "cuda_error_no_binary" in t:
        return "209"
    if "device lost" in t or "device_lost" in t:
        return "device-lost"
    if "out of memory" in t:
        return "oom"
    if "not implemented" in t or "not supported" in t:
        return "unsupported"
    return "error"


def close(a, b, rtol, atol):
    """(ok, max_abs_err, max_rel_err) for two same-shape tensors, NaN-aware."""
    import torch
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    if a.shape != b.shape:
        return False, float("inf"), float("inf")
    if torch.isnan(a).any() or torch.isnan(b).any():
        return bool(torch.equal(torch.isnan(a), torch.isnan(b))), float("nan"), float("nan")
    diff = (a - b).abs()
    mx = float(diff.max()) if diff.numel() else 0.0
    rel = float((diff / (b.abs() + 1e-12)).max()) if diff.numel() else 0.0
    ok = bool(torch.allclose(a, b, rtol=rtol, atol=atol))
    return ok, mx, rel


def first_divergence(a, b):
    """Index of the first differing element of two token lists, or None if equal (length differences count)."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def level_status(checks):
    if not checks:
        return "fail"
    return "pass" if all(c["ok"] for c in checks) else "fail"


def matrix_markdown(results, meta=None):
    """results: {level: {status, seconds, checks, note, ...}} -> markdown table."""
    meta = meta or {}
    lines = []
    if meta:
        lines.append("Host: " + ", ".join(f"{k}={v}" for k, v in meta.items()))
        lines.append("")
    lines += ["| Level | Workload | Result | Seconds | Notes |", "|---|---|---|---|---|"]
    for lv in LEVELS:
        r = results.get(lv)
        if not r:
            continue
        notes = r.get("note", "")
        if lv == "L7" and r.get("variants"):
            notes = "; ".join(f"{k}: {v['status']}" + (f" ({v['error_class']})" if v.get("error_class") else "")
                              for k, v in r["variants"].items())
        bad = [c for c in r.get("checks", []) if not c["ok"]]
        if bad and not notes:
            notes = "; ".join(f"{c['name']}: {c.get('note', 'mismatch')}" for c in bad[:3])
        lines.append(f"| {lv} | {TITLES[lv]} | {r['status']} | {r.get('seconds', 0):.1f} | {notes} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- child side
def setup_torch():
    """Load the shim before torch (as docs/pytorch.md prescribes) and return (torch, device)."""
    if not fake_cpu():
        path = os.environ.get("CFI_NVCUDA")
        if path:
            import ctypes
            ctypes.WinDLL(path) if hasattr(ctypes, "WinDLL") else ctypes.CDLL(path)
    import torch
    if not fake_cpu():
        torch.backends.cudnn.enabled = False  # no cuDNN backend: use ATen's native kernels
        return torch, "cuda"
    return torch, "cpu"


def check(checks, name, ok, **extra):
    checks.append({"name": name, "ok": bool(ok), **extra})


def cmp_check(checks, name, got, want, rtol=1e-4, atol=1e-5):
    ok, mx, rel = close(got, want, rtol, atol)
    check(checks, name, ok, max_abs=mx, max_rel=rel, note=None if ok else f"max_abs={mx:.3g}")


def sync(torch, dev):
    if dev == "cuda":
        torch.cuda.synchronize()


def level_L0(torch, dev, checks, info):
    info["torch"] = torch.__version__
    info["cuda_build"] = torch.version.cuda
    if dev == "cuda":
        check(checks, "is_available", torch.cuda.is_available())
        info["device_name"] = torch.cuda.get_device_name(0)
        info["capability"] = list(torch.cuda.get_device_capability(0))
        info["arch_list"] = torch.cuda.get_arch_list()
        info["device_count"] = torch.cuda.device_count()
        check(checks, "device_count>=1", torch.cuda.device_count() >= 1)
    else:
        check(checks, "cpu_fake", True, note="CFI_FAKE_CPU=1")


def level_L1(torch, dev, checks, info):
    g = torch.Generator().manual_seed(1)
    x = torch.randn(1024, generator=g)
    y = torch.randn(1024, generator=g)
    xd, yd = x.to(dev), y.to(dev)
    check(checks, "zeros", bool((torch.zeros(16, device=dev).cpu() == 0).all()))
    cmp_check(checks, "roundtrip", xd.cpu(), x, 0, 0)
    cmp_check(checks, "add", xd + yd, x + y)
    cmp_check(checks, "mul", xd * yd, x * y)
    cmp_check(checks, "relu", torch.relu(xd), torch.relu(x))
    cmp_check(checks, "gelu", torch.nn.functional.gelu(xd), torch.nn.functional.gelu(x), 1e-4, 1e-5)
    cmp_check(checks, "exp", torch.exp(xd), torch.exp(x), 1e-4, 1e-5)
    cmp_check(checks, "fp16_cast", xd.half().float(), x.half().float(), 0, 0)
    cmp_check(checks, "cat", torch.cat([xd, yd]), torch.cat([x, y]), 0, 0)
    idx = torch.tensor([3, 1, 4, 1, 5])
    cmp_check(checks, "index_select", xd.index_select(0, idx.to(dev)), x.index_select(0, idx), 0, 0)
    emb = torch.nn.Embedding(50, 8)
    cmp_check(checks, "embedding", emb.to(dev)(idx.to(dev)), emb(idx), 1e-6, 1e-6)
    cmp_check(checks, "pinned_nonblocking", x.pin_memory().to(dev, non_blocking=True).cpu() if dev == "cuda" else x, x, 0, 0)


def level_L2(torch, dev, checks, info):
    g = torch.Generator().manual_seed(2)
    for m, k, n in [(64, 64, 64), (130, 70, 90), (256, 512, 128)]:
        a = torch.randn(m, k, generator=g)
        b = torch.randn(k, n, generator=g)
        cmp_check(checks, f"f32_{m}x{k}x{n}", (a.to(dev) @ b.to(dev)), a @ b, 1e-4, 1e-3)
    a = torch.randn(256, 256, generator=g)
    b = torch.randn(256, 256, generator=g)
    ref = (a.half().float() @ b.half().float())
    cmp_check(checks, "f16_256", (a.half().to(dev) @ b.half().to(dev)).float(), ref, 2e-2, 2e-1)
    big = torch.randn(1024, 1024, generator=g)
    ref = (big.half().float() @ big.half().float())
    cmp_check(checks, "f16_1024_xmx_range", (big.half().to(dev) @ big.half().to(dev)).float(), ref, 3e-2, 1.0)
    for ta in (False, True):
        for tb in (False, True):
            p = torch.randn(48, 48, generator=g)
            q = torch.randn(48, 48, generator=g)
            f = lambda t, tr: t.t() if tr else t
            cmp_check(checks, f"f32_transpose_{int(ta)}{int(tb)}", f(p.to(dev), ta) @ f(q.to(dev), tb), f(p, ta) @ f(q, tb), 1e-4, 1e-3)
    bb = torch.randn(4, 32, 32, generator=g)
    cmp_check(checks, "bmm", torch.bmm(bb.to(dev), bb.to(dev)), torch.bmm(bb, bb), 1e-4, 1e-3)
    lin = torch.nn.Linear(64, 32)
    xx = torch.randn(8, 64, generator=g)
    cmp_check(checks, "linear_fwd", lin.to(dev)(xx.to(dev)), lin.cpu()(xx), 1e-4, 1e-4)


def level_L3(torch, dev, checks, info):
    F = torch.nn.functional
    g = torch.Generator().manual_seed(3)
    x = torch.randn(32, 257, generator=g)
    xd = x.to(dev)
    cmp_check(checks, "softmax", F.softmax(xd, -1), F.softmax(x, -1), 1e-4, 1e-6)
    cmp_check(checks, "log_softmax", F.log_softmax(xd, -1), F.log_softmax(x, -1), 1e-4, 1e-5)
    ln = torch.nn.LayerNorm(257)
    cmp_check(checks, "layer_norm", ln.to(dev)(xd), ln.cpu()(x), 1e-4, 1e-5)
    bn = torch.nn.BatchNorm1d(257)
    cmp_check(checks, "batch_norm", bn.to(dev)(xd), bn.cpu()(x), 1e-4, 1e-4)
    cmp_check(checks, "sum", xd.sum(), x.sum(), 1e-4, 1e-2)
    cmp_check(checks, "mean_dim", xd.mean(1), x.mean(1), 1e-4, 1e-5)
    cmp_check(checks, "max", xd.max(1).values, x.max(1).values, 0, 0)
    check(checks, "argmax", bool((xd.argmax(1).cpu() == x.argmax(1)).all()))
    cmp_check(checks, "cumsum", xd.cumsum(1), x.cumsum(1), 1e-4, 1e-3)
    cmp_check(checks, "var", xd.var(1), x.var(1), 1e-4, 1e-5)
    lbl = torch.randint(0, 257, (32,), generator=g)
    cmp_check(checks, "cross_entropy", F.cross_entropy(xd, lbl.to(dev)), F.cross_entropy(x, lbl), 1e-4, 1e-5)
    xh = x.half()
    cmp_check(checks, "softmax_f16", F.softmax(xh.to(dev), -1).float(), F.softmax(xh.float(), -1), 1e-2, 1e-3)


def level_L4(torch, dev, checks, info):
    g = torch.Generator().manual_seed(4)
    conv = torch.nn.Conv2d(3, 8, 3, padding=1)
    x = torch.randn(2, 3, 16, 16, generator=g)
    xc = x.clone().requires_grad_(True)
    yc = conv(xc)
    yc.square().sum().backward()
    gw, gb, gx = conv.weight.grad.clone(), conv.bias.grad.clone(), xc.grad.clone()
    conv.zero_grad()
    cd = torch.nn.Conv2d(3, 8, 3, padding=1).to(dev)
    cd.load_state_dict(conv.state_dict())
    xd = x.to(dev).requires_grad_(True)
    yd = cd(xd)
    yd.square().sum().backward()
    cmp_check(checks, "conv_fwd", yd, yc, 1e-4, 1e-4)
    cmp_check(checks, "conv_grad_w", cd.weight.grad, gw, 1e-3, 1e-3)
    cmp_check(checks, "conv_grad_b", cd.bias.grad, gb, 1e-3, 1e-2)
    cmp_check(checks, "conv_grad_x", xd.grad, gx, 1e-3, 1e-3)
    cmp_check(checks, "max_pool", torch.nn.functional.max_pool2d(x.to(dev), 2), torch.nn.functional.max_pool2d(x, 2), 0, 0)
    cmp_check(checks, "avg_pool", torch.nn.functional.avg_pool2d(x.to(dev), 2), torch.nn.functional.avg_pool2d(x, 2), 1e-5, 1e-6)


def level_L5(torch, dev, checks, info):
    torch.manual_seed(5)
    layer = torch.nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0, batch_first=True)
    layer.eval()
    x = torch.randn(2, 12, 64)
    xc = x.clone().requires_grad_(True)
    yc = layer(xc)
    yc.square().sum().backward()
    grads = {n: p.grad.clone() for n, p in layer.named_parameters()}
    layer.zero_grad()
    ld = torch.nn.TransformerEncoderLayer(d_model=64, nhead=4, dim_feedforward=128, dropout=0.0, batch_first=True).to(dev)
    ld.load_state_dict(layer.state_dict())
    ld.eval()
    xd = x.to(dev).requires_grad_(True)
    yd = ld(xd)
    yd.square().sum().backward()
    cmp_check(checks, "forward", yd, yc, 1e-3, 1e-4)
    bad = []
    worst = 0.0
    for n, p in ld.named_parameters():
        ok, mx, _ = close(p.grad, grads[n], 2e-3, 1e-3)
        worst = max(worst, mx)
        if not ok:
            bad.append(n)
    check(checks, f"grads({len(grads)})", not bad, max_abs=worst, note=",".join(bad[:4]) or None)
    cmp_check(checks, "grad_x", xd.grad, xc.grad, 2e-3, 1e-3)


def level_L6(torch, dev, checks, info):
    path = os.environ.get("CFI_HF_MODEL")
    if not path:
        info["skip"] = "CFI_HF_MODEL not set"
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer
    n = int(os.environ.get("CFI_L6_TOKENS", "32"))
    prompt = os.environ.get("CFI_L6_PROMPT", "The quick brown fox jumps over the lazy dog. The quick brown fox")
    try:
        tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
        ids = tok(prompt, return_tensors="pt").input_ids
        if ids.numel() == 0:
            raise ValueError("empty tokenization")
    except Exception:  # no tokenizer files: fall back to raw ids (CFI_L6_IDS="1,2,3")
        tok = None
        ids = torch.tensor([[int(t) for t in os.environ.get("CFI_L6_IDS", "1,2,3,4,5").split(",")]])
    dtype = torch.float32
    model = AutoModelForCausalLM.from_pretrained(path, local_files_only=True, torch_dtype=dtype).eval()
    kw = dict(max_new_tokens=n, do_sample=False, num_beams=1, pad_token_id=getattr(tok, 'eos_token_id', None) or 0)
    with torch.no_grad():
        ref = model.generate(ids, **kw)[0, ids.shape[1]:].tolist()
        ref_logits = model(ids).logits[0, -1].float()
        model = model.to(dev)
        got = model.generate(ids.to(dev), **kw)[0, ids.shape[1]:].tolist()
        got_logits = model(ids.to(dev)).logits[0, -1].float().cpu()
    pos = first_divergence(got, ref)
    info.update(prompt_tokens=ids.shape[1], generated=len(got), first_divergent_position=pos,
                cpu_ids=ref, dev_ids=got,
                cpu_text=tok.decode(ref) if tok else None, dev_text=tok.decode(got) if tok else None)
    top_ok = int(got_logits.argmax()) == int(ref_logits.argmax())
    check(checks, "prefill_logits_close", close(got_logits, ref_logits, 1e-2, 5e-2)[0],
          max_abs=close(got_logits, ref_logits, 1, 1)[1])
    check(checks, "prefill_top1_equal", top_ok)
    check(checks, "tokens_equal_cpu", pos is None, note=None if pos is None else f"first divergence at generated token {pos}")


def level_L7(torch, dev, checks, info):
    """One SDPA variant per child process; the variant name is in CFI_L7_VARIANT."""
    F = torch.nn.functional
    variant = os.environ.get("CFI_L7_VARIANT", "default")
    info["variant"] = variant
    if variant == "math" and dev == "cuda":
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_flash_sdp(False)
    g = torch.Generator().manual_seed(7)
    q, k, v = (torch.randn(2, 4, 16, 32, generator=g) for _ in range(3))
    for causal in (False, True):
        cmp_check(checks, f"sdpa_causal={causal}",
                  F.scaled_dot_product_attention(q.to(dev), k.to(dev), v.to(dev), is_causal=causal),
                  F.scaled_dot_product_attention(q, k, v, is_causal=causal), 1e-3, 1e-4)
    mask = torch.rand(16, 16, generator=g) > 0.2
    mask |= torch.eye(16, dtype=torch.bool)
    cmp_check(checks, "sdpa_mask", F.scaled_dot_product_attention(q.to(dev), k.to(dev), v.to(dev), attn_mask=mask.to(dev)),
              F.scaled_dot_product_attention(q, k, v, attn_mask=mask), 1e-3, 1e-4)
    qh, kh, vh = q.half(), k.half(), v.half()
    cmp_check(checks, "sdpa_f16", F.scaled_dot_product_attention(qh.to(dev), kh.to(dev), vh.to(dev)).float(),
              F.scaled_dot_product_attention(q, k, v), 2e-2, 2e-2)
    mha = torch.nn.MultiheadAttention(32, 4, batch_first=True).eval()
    x = torch.randn(2, 10, 32, generator=g)
    with torch.no_grad():
        want = mha(x, x, x, need_weights=False)[0]
        got = mha.to(dev)(x.to(dev), x.to(dev), x.to(dev), need_weights=False)[0]
    cmp_check(checks, "multihead_attention", got, want, 1e-3, 1e-4)


LEVEL_FUNCS = {f"L{i}": globals()[f"level_L{i}"] for i in range(8)}


def child(level):
    t0 = time.time()
    checks, info = [], {}
    out = {"level": level, "status": "error", "checks": checks, "info": info}
    try:
        torch, dev = setup_torch()
        info["device"] = dev
        LEVEL_FUNCS[level](torch, dev, checks, info)
        sync(torch, dev)
        if "skip" in info:
            out["status"], out["note"] = "skip", info["skip"]
        else:
            out["status"] = level_status(checks)
    except BaseException as e:  # noqa: BLE001 - report everything, including SystemExit from the runtime
        text = f"{type(e).__name__}: {e}"
        out["status"] = "error"
        out["error"] = text[:2000]
        out["error_class"] = classify_error(text)
    out["seconds"] = round(time.time() - t0, 2)
    print(RESULT_TAG + json.dumps(out, default=str), flush=True)
    return 0


# ---------------------------------------------------------------- parent side
def parse_child_output(stdout):
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_TAG):
            try:
                return json.loads(line[len(RESULT_TAG):])
            except ValueError:
                return None
    return None


def run_child(level, timeout, extra_env=None, runner=subprocess.run):
    env = dict(os.environ)
    env.update(extra_env or {})
    t0 = time.time()
    try:
        p = runner([sys.executable, os.path.abspath(__file__), "--child", level], capture_output=True, text=True,
                   timeout=timeout, env=env)
    except subprocess.TimeoutExpired as e:
        return {"level": level, "status": "timeout", "seconds": round(time.time() - t0, 1),
                "error": f"no result within {timeout}s", "error_class": "timeout", "checks": [],
                "stderr_tail": (e.stderr or "")[-1500:] if isinstance(e.stderr, str) else ""}
    res = parse_child_output(p.stdout)
    if res is None:
        res = {"level": level, "status": "error", "checks": [], "error": f"child exited rc={p.returncode} without a result",
               "error_class": classify_error(p.stderr or ""), "seconds": round(time.time() - t0, 1)}
    res["rc"] = p.returncode
    if res["status"] not in ("pass", "skip"):
        res["stderr_tail"] = (p.stderr or "")[-1500:]
    return res


def run_level(level, timeout, runner=subprocess.run):
    if level != "L7":
        return run_child(level, timeout, runner=runner)
    variants, checks, secs = {}, [], 0.0
    for v in SDPA_VARIANTS:
        extra = {"CFI_L7_VARIANT": v}
        if v == "cc37":
            extra["CFI_REPORT_CC"] = "3.7"
        r = run_child("L7", timeout, extra, runner=runner)
        variants[v] = {"status": r["status"], "error_class": r.get("error_class"), "error": r.get("error"),
                       "seconds": r.get("seconds", 0), "checks": r.get("checks", [])}
        secs += r.get("seconds", 0)
        checks.append({"name": f"variant_{v}", "ok": r["status"] == "pass"})
    # Expected on the cu118 wheel: default fails with 209, cc37 and math pass. Report, do not judge.
    status = "pass" if all(x["ok"] for x in checks) else ("partial" if any(x["ok"] for x in checks) else "fail")
    return {"level": "L7", "status": status, "seconds": round(secs, 1), "variants": variants, "checks": checks}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--timeout", type=int, default=900, help="seconds per level (per variant for L7); L6 gets 4x")
    ap.add_argument("--out", default="torch-progressive-out")
    ap.add_argument("--stop-on-fail", action="store_true")
    a = ap.parse_args(argv)
    if a.child:
        return child(a.child)
    levels = [x.strip().upper() for x in a.levels.split(",") if x.strip()]
    bad = [x for x in levels if x not in LEVELS]
    if bad:
        ap.error(f"unknown levels {bad}")
    os.makedirs(a.out, exist_ok=True)
    results = {}
    for lv in levels:
        timeout = a.timeout * (4 if lv == "L6" else 1)
        print(f"[{lv}] {TITLES[lv]} (timeout {timeout}s)", flush=True)
        r = run_level(lv, timeout)
        results[lv] = r
        with open(os.path.join(a.out, f"{lv}.json"), "w", encoding="utf-8") as f:
            json.dump(r, f, indent=1, default=str)
        print(f"[{lv}] {r['status']} in {r.get('seconds', 0)}s {r.get('error', '')[:200]}", flush=True)
        if a.stop_on_fail and r["status"] in ("fail", "error", "timeout"):
            break
    meta = {"fake_cpu": fake_cpu()}
    l0 = results.get("L0", {}).get("info", {})
    for key in ("torch", "cuda_build", "device_name"):
        if l0.get(key):
            meta[key] = l0[key]
    md = matrix_markdown(results, meta)
    with open(os.path.join(a.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "levels": results}, f, indent=1, default=str)
    with open(os.path.join(a.out, "matrix.md"), "w", encoding="utf-8") as f:
        f.write(md)
    print(md)
    return 0 if all(r["status"] in ("pass", "skip", "partial") for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
