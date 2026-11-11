#!/usr/bin/env python3
"""Summarise a CFI_GPU_PROFILE_OUT file (per-kernel GPU time from Level Zero kernel timestamps).

usage:
  gpu_profile.py PROFILE.tsv [--top N]                  totals of the last snapshot
  gpu_profile.py PROFILE.tsv --phase A_TOK B_TOK [--top N]
      decode-only: the run made two requests with the same prompt length after quiet pauses
      (>= 2.5 s, so each starts a new snapshot), A generating A_TOK tokens and B generating B_TOK;
      prints (B - A) per generated token, i.e. per-token decode with the prefill cancelled out.
      A third request after another pause marks the end of B.

Snapshot format (see src/cuda/gpuprof.rs):
  GPUPROF wall_ms=<ms> launches=<n> timed=<n> skipped=<n>
  K <count> <total_ns> <max_ns> <kernel name>
  END
"""
import argparse
import re
import sys


def parse(text):
    snaps = []
    for block in text.split("GPUPROF ")[1:]:
        head, *rows = block.splitlines()
        meta = dict(kv.split("=") for kv in head.split())
        kernels = {}
        for row in rows:
            if row == "END":
                break
            _, count, total, peak, name = row.split("\t", 4)
            kernels[name] = (int(count), int(total), int(peak))
        snaps.append(({k: int(v) for k, v in meta.items()}, kernels))
    return snaps


def short(name):
    """`_Z13mul_mat_vec_qIL9ggml_type8E...` -> `mul_mat_vec_q<ggml_type8,...>` (first template args only)."""
    m = re.match(r"_Z(\d+)", name)
    if not m:
        return name[:70]
    n = int(m[1])
    base = name[m.end():m.end() + n]
    rest = name[m.end() + n:]
    args = re.findall(r"L(?:\d+\w+?|i)(\d+|E)|Li(-?\d+)E|Lb(\d)E", rest[:60])
    nums = [a or b or c for a, b, c in args if (a or b or c) and (a or b or c) != "E"]
    return f"{base}<{','.join(nums[:4])}>" if nums else base


def diff(a, b):
    out = {}
    for k, (c, t, _) in b.items():
        c0, t0, _ = a.get(k, (0, 0, 0))
        if c - c0 or t - t0:
            out[k] = (c - c0, t - t0)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("profile")
    ap.add_argument("--phase", nargs=2, type=int, metavar=("A_TOK", "B_TOK"))
    ap.add_argument("--top", type=int, default=20)
    args = ap.parse_args()
    snaps = parse(open(args.profile, encoding="utf-8", errors="replace").read())
    if not snaps:
        sys.exit("no GPUPROF snapshots")
    if args.phase:
        starts = [i for i in range(1, len(snaps)) if snaps[i][0]["wall_ms"] - snaps[i - 1][0]["wall_ms"] > 2500]
        if len(starts) < 3:
            sys.exit(f"found {len(starts)} request boundaries in {len(snaps)} snapshots; need 3")
        sa, sb, sc = (snaps[i] for i in starts[-3:])
        da, db = diff(sa[1], sb[1]), diff(sb[1], sc[1])
        tokens = args.phase[1] - args.phase[0]
        rows = {k: (db.get(k, (0, 0))[0] - da.get(k, (0, 0))[0], db.get(k, (0, 0))[1] - da.get(k, (0, 0))[1]) for k in set(da) | set(db)}
        skipped = (sc[0]["skipped"] - sb[0]["skipped"]) - (sb[0]["skipped"] - sa[0]["skipped"])
        unit, scale = "per token", tokens
    else:
        rows = {k: (c, t) for k, (c, t, _) in snaps[-1][1].items()}
        skipped, unit, scale = snaps[-1][0]["skipped"], "total", 1
    total = sum(t for _, t in rows.values())
    calls = sum(c for c, _ in rows.values())
    print(f"GPU kernel time {unit}: {total / scale / 1e3:.1f} us over {calls / scale:.1f} launches ({skipped} untimed launches)")
    print(f"| # | kernel | launches {unit} | GPU us {unit} | % | avg us |")
    print("|---|---|---|---|---|---|")
    for i, (k, (c, t)) in enumerate(sorted(rows.items(), key=lambda kv: -kv[1][1])[: args.top], 1):
        print(f"| {i} | `{short(k)}` | {c / scale:.1f} | {t / scale / 1e3:.1f} | {100 * t / max(total, 1):.1f} | {t / max(c, 1) / 1e3:.2f} |")


if __name__ == "__main__":
    main()
