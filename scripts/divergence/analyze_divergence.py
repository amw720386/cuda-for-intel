#!/usr/bin/env python3
"""Compare per-step logits of SUBST=auto and SUBST=none (cfi_logits.exe dumps) against an fp32 CPU reference.

The reference is built here: every tensor of the same GGUF is dequantised to fp32 (gguf-py) and loaded into a
transformers Qwen2 model that runs on the CPU in fp32 (or fp64 with --dtype float64).

usage: analyze_divergence.py --results DIR --gguf MODEL.gguf [--dtype float32] [--out report.md]
DIR holds what scripts/divergence/run_on_box.ps1 produced:
  llama_auto_gen.*, llama_none_gen.*            free runs  (<tag>.logits.bin float32 [steps][vocab], <tag>.ids.txt)
  llama_<mode>_force_<other>.*                  mode <mode> teacher-forced on the tokens that <other> generated
"""
import argparse, os, sys, json
import numpy as np


def read_ids(path):
    lines = open(path).read().split("\n")
    nv, npmt = map(int, lines[0].split())
    return nv, npmt, [int(x) for x in lines[1].split()], [int(x) for x in lines[2].split()]


def read_logits(path, nv):
    a = np.fromfile(path, dtype=np.float32)
    assert a.size % nv == 0, (path, a.size, nv)
    return a.reshape(-1, nv)


def build_reference(gguf_path, dtype):
    import torch, gguf
    from gguf.quants import dequantize
    from transformers import Qwen2Config, Qwen2ForCausalLM
    r = gguf.GGUFReader(gguf_path)

    def kv(name):
        f = r.fields[name]
        return f.contents() if hasattr(f, "contents") else f.parts[f.data[0]][0].item()

    tdt = {"float32": torch.float32, "float64": torch.float64}[dtype]
    nl = int(kv("qwen2.block_count"))
    cfg = Qwen2Config(
        vocab_size=int(next(t for t in r.tensors if t.name == "token_embd.weight").shape[1]),
        hidden_size=int(kv("qwen2.embedding_length")), intermediate_size=int(kv("qwen2.feed_forward_length")),
        num_hidden_layers=nl, num_attention_heads=int(kv("qwen2.attention.head_count")),
        num_key_value_heads=int(kv("qwen2.attention.head_count_kv")), rms_norm_eps=float(kv("qwen2.attention.layer_norm_rms_epsilon")),
        rope_theta=float(kv("qwen2.rope.freq_base")), max_position_embeddings=4096, tie_word_embeddings=False,
        use_sliding_window=False, attn_implementation="eager")
    cfg.torch_dtype = tdt
    sd = {}
    names = {"attn_q": "self_attn.q_proj", "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_output": "self_attn.o_proj",
             "ffn_gate": "mlp.gate_proj", "ffn_up": "mlp.up_proj", "ffn_down": "mlp.down_proj",
             "attn_norm": "input_layernorm", "ffn_norm": "post_attention_layernorm"}
    for t in r.tensors:
        w = torch.from_numpy(np.array(dequantize(t.data, t.tensor_type), dtype=np.float32).reshape(tuple(reversed(t.shape.tolist()))))
        n = t.name
        if n == "token_embd.weight": k = "model.embed_tokens.weight"
        elif n == "output_norm.weight": k = "model.norm.weight"
        elif n == "output.weight": k = "lm_head.weight"
        else:
            _, i, rest = n.split(".", 2)
            base, kind = rest.rsplit(".", 1)
            k = f"model.layers.{i}.{names[base]}.{kind}"
        sd[k] = w.to(tdt)
    if "lm_head.weight" not in sd: sd["lm_head.weight"] = sd["model.embed_tokens.weight"]
    torch.set_default_dtype(tdt)
    m = Qwen2ForCausalLM(cfg)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "rotary" not in k]
    assert not missing and not unexpected, (missing, unexpected)
    return m.eval()


def reference_logits(model, prompt_ids, gen_ids):
    import torch
    ids = torch.tensor([prompt_ids + gen_ids[:-1]])
    with torch.no_grad():
        lg = model(ids).logits[0]
    return lg[len(prompt_ids) - 1:].double().numpy()  # step i predicts gen[i]


def step_table(ref, a, n, gen):
    """per-step metrics for sequence `gen` (teacher forced), auto logits `a`, none logits `n`, reference `ref` (fp64 array)."""
    rows = []
    for i in range(len(gen)):
        r, x, y = ref[i], a[i].astype(np.float64), n[i].astype(np.float64)
        o = np.argsort(-r)[:2]
        top = np.argsort(-r)[:20]
        rows.append(dict(step=i, tok=gen[i], ref_top1=int(o[0]), ref_margin=float(r[o[0]] - r[o[1]]),
                         auto_top1=int(np.argmax(x)), none_top1=int(np.argmax(y)),
                         d_auto_ref=float(np.max(np.abs(x - r))), d_none_ref=float(np.max(np.abs(y - r))),
                         d_auto_ref_top20=float(np.max(np.abs(x[top] - r[top]))), d_none_ref_top20=float(np.max(np.abs(y[top] - r[top]))),
                         d_auto_none=float(np.max(np.abs(x - y))), d_auto_none_top20=float(np.max(np.abs(x[top] - y[top]))),
                         auto_margin=float(np.sort(x)[-1] - np.sort(x)[-2]), none_margin=float(np.sort(y)[-1] - np.sort(y)[-2])))
    return rows


def summarize(rows, label):
    L = []
    flips = [r for r in rows if r["auto_top1"] != r["none_top1"]]
    L.append(f"### {label}\n")
    L.append(f"steps: {len(rows)}; top-1 flips auto vs none: {len(flips)}; "
             f"auto top-1 != reference top-1: {sum(r['auto_top1'] != r['ref_top1'] for r in rows)}; "
             f"none top-1 != reference top-1: {sum(r['none_top1'] != r['ref_top1'] for r in rows)}\n")
    f = lambda k: float(np.max([r[k] for r in rows]))
    m = lambda k: float(np.mean([r[k] for r in rows]))
    L.append("| quantity (all steps) | max | mean |\n|---|---|---|")
    for k, nm in [("d_auto_ref", "max abs dlogit auto vs fp32 ref (full vocab)"), ("d_none_ref", "max abs dlogit none vs fp32 ref (full vocab)"),
                  ("d_auto_ref_top20", "auto vs ref, ref top-20 tokens"), ("d_none_ref_top20", "none vs ref, ref top-20 tokens"),
                  ("d_auto_none", "auto vs none (full vocab)"), ("d_auto_none_top20", "auto vs none, ref top-20 tokens")]:
        L.append(f"| {nm} | {f(k):.5f} | {m(k):.5f} |")
    L.append("")
    if flips:
        L.append("| step | token fed next | ref top-2 margin | auto margin | none margin | max abs dlogit auto-none (step) | auto top1 | none top1 | ref top1 |\n|---|---|---|---|---|---|---|---|---|")
        for r in flips:
            L.append(f"| {r['step']} | {r['tok']} | {r['ref_margin']:.5f} | {r['auto_margin']:.5f} | {r['none_margin']:.5f} | {r['d_auto_none']:.5f} | {r['auto_top1']} | {r['none_top1']} | {r['ref_top1']} |")
        L.append("")
    # smallest reference margins overall for context
    sm = sorted(rows, key=lambda r: r["ref_margin"])[:8]
    L.append("Eight smallest reference top-2 margins of the sequence:\n")
    L.append("| step | ref margin | auto==none top1 | max abs dlogit auto-none |\n|---|---|---|---|")
    for r in sm:
        L.append(f"| {r['step']} | {r['ref_margin']:.5f} | {'yes' if r['auto_top1'] == r['none_top1'] else 'NO'} | {r['d_auto_none']:.5f} |")
    L.append("")
    noise = f("d_auto_none")
    bad = [r for r in flips if r["ref_margin"] > 2 * noise]
    L.append(f"Check: largest auto-vs-none logit difference at any step = {noise:.5f}. "
             f"Flips with reference margin > 2x that = {len(bad)}.\n")
    return "\n".join(L), flips, noise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--gguf", required=True)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float64"]); ap.add_argument("--out", default="")
    a = ap.parse_args()
    R = a.results
    nv, npmt, prompt_ids, an_gen = read_ids(f"{R}/llama_auto_gen.ids.txt")
    _, _, _, no_gen = read_ids(f"{R}/llama_none_gen.ids.txt")
    first = next((i for i in range(min(len(an_gen), len(no_gen))) if an_gen[i] != no_gen[i]), None)
    out = [f"free-run token sequences: auto {len(an_gen)} tokens, none {len(no_gen)} tokens; first differing generated token index: {first}\n"]
    model = build_reference(a.gguf, a.dtype)
    cache = {}
    for seq_name, gen, la_tag, ln_tag in [("none-generated sequence", no_gen, "llama_auto_force_none", "llama_none_gen"),
                                          ("auto-generated sequence", an_gen, "llama_auto_gen", "llama_none_force_auto")]:
        pa, pn = f"{R}/{la_tag}.logits.bin", f"{R}/{ln_tag}.logits.bin"
        if not (os.path.exists(pa) and os.path.exists(pn)): continue
        la, ln = read_logits(pa, nv), read_logits(pn, nv)
        k = min(len(gen), len(la), len(ln))
        ref = reference_logits(model, prompt_ids, gen[:k])
        rows = step_table(ref, la[:k], ln[:k], gen[:k])
        txt, flips, noise = summarize(rows, f"Teacher-forced on the {seq_name} ({k} steps, reference dtype {a.dtype})")
        out.append(txt)
        json.dump(rows, open(f"{R}/steps_{la_tag}_vs_{ln_tag}.json", "w"))
    rep = "\n".join(out)
    print(rep)
    if a.out: open(a.out, "w").write(rep)


if __name__ == "__main__":
    main()
