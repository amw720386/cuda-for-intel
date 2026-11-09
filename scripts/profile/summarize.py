#!/usr/bin/env python3
"""Turn a CFI_PROFILE_OUT dump into a ranked markdown table.

usage: summarize.py DUMP [--top N] [--out FILE] [--base EARLIER_DUMP]
--base subtracts an earlier snapshot of the same (cumulative) profile, so one phase of a long-running
process (e.g. one Ollama request) can be isolated.
Dump format (tab separated, see src/cuda/cfiprof.rs):
  CFIPROF 1
  K count total_ns launch_ns sync_ns name
  A count total_ns api
  M count bytes kind
"""
import argparse
import sys


def parse(text):
    kernels, apis, copies = [], [], []
    for line in text.splitlines():
        f = line.split("\t")
        try:
            if f[0] == "K" and len(f) >= 6:
                kernels.append((f[5], int(f[1]), int(f[2]), int(f[3]), int(f[4])))
            elif f[0] == "A" and len(f) >= 4:
                apis.append((f[3], int(f[1]), int(f[2])))
            elif f[0] == "M" and len(f) >= 4:
                copies.append((f[3], int(f[1]), int(f[2])))
        except ValueError:
            continue
    return kernels, apis, copies


def subtract(cur, base):
    """Per-name difference of two parsed dumps (cur - base); rows that did not change are dropped."""
    out = []
    for rows, brows in zip(cur, base):
        b = {r[0]: r[1:] for r in brows}
        diff = []
        for r in rows:
            prev = b.get(r[0], (0,) * (len(r) - 1))
            d = (r[0],) + tuple(x - y for x, y in zip(r[1:], prev))
            if d[1] > 0:
                diff.append(d)
        out.append(diff)
    return tuple(out)


def render(text, top=25, base=None):
    kernels, apis, copies = parse(text)
    if base is not None:
        kernels, apis, copies = subtract((kernels, apis, copies), parse(base))
    kernels.sort(key=lambda k: -k[2])
    apis.sort(key=lambda a: -a[2])
    total = sum(k[2] for k in kernels) or 1
    launches = sum(k[1] for k in kernels)
    launch = sum(k[3] for k in kernels)
    sync = sum(k[4] for k in kernels)
    out = ["## Kernels by time (serialised host-side timing; rank, do not read absolute speed)", "",
           f"{launches} launches, {total/1e6:.2f} ms total, submit {launch/1e6:.2f} ms, wait {sync/1e6:.2f} ms", "",
           "| # | kernel | launches | total ms | % | avg us | submit % |", "|---|---|---|---|---|---|---|"]
    for i, (name, n, t, l, s) in enumerate(kernels[:top], 1):
        short = name if len(name) <= 70 else name[:67] + "..."
        out.append(f"| {i} | `{short}` | {n} | {t/1e6:.2f} | {100*t/total:.1f} | {t/max(n,1)/1e3:.1f} | {100*l/max(t,1):.0f} |")
    if apis:
        out += ["", "## Driver API calls by host time", "", "| api | calls | total ms |", "|---|---|---|"]
        out += [f"| {a} | {n} | {t/1e6:.2f} |" for a, n, t in apis[:top]]
    if copies:
        out += ["", "## Memcpy", "", "| kind | calls | MiB |", "|---|---|---|"]
        out += [f"| {k} | {n} | {b/2**20:.2f} |" for k, n, b in sorted(copies, key=lambda c: -c[2])]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dump")
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--out")
    ap.add_argument("--base", help="earlier snapshot of the same profile to subtract")
    a = ap.parse_args()
    base = open(a.base, encoding="utf-8", errors="replace").read() if a.base else None
    md = render(open(a.dump, encoding="utf-8", errors="replace").read(), a.top, base)
    if a.out:
        open(a.out, "w", encoding="utf-8").write(md)
    else:
        sys.stdout.write(md)


if __name__ == "__main__":
    main()
