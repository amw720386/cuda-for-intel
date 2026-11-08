#!/usr/bin/env python3
"""Render docs/compat/reports/*.yml into a Markdown table.
usage: render_compat.py [reports_dir] [--verified-only]
Reads a flat YAML subset (key: value, and `key: |` blocks); no dependencies.
Files starting with '_' are templates and skipped."""
import sys, re
from pathlib import Path

def parse(text):
    d, key, block = {}, None, []
    for line in text.splitlines():
        if key and (line.startswith(" ") or not line.strip()):
            block.append(line.strip()); continue
        if key:
            d[key] = "\n".join(block).strip(); key = None
        m = re.match(r"^([A-Za-z_]+):\s*(.*)$", line)
        if not m: continue
        k, v = m.group(1), re.sub(r"\s+#.*$", "", m.group(2)).strip()
        if v == "|": key, block = k, []
        else: d[k] = v.strip('"')
    if key: d[key] = "\n".join(block).strip()
    return d

def main(argv):
    args = [a for a in argv if not a.startswith("--")]
    only = "--verified-only" in argv
    root = Path(args[0]) if args else Path(__file__).resolve().parents[2] / "docs/compat/reports"
    rows = {}
    for f in sorted(root.glob("*.yml")):
        if f.name.startswith("_"): continue
        r = parse(f.read_text())
        if r.get("schema") != "1": print(f"skip {f.name}: bad schema", file=sys.stderr); continue
        if r.get("result") not in ("pass", "partial", "fail"):
            print(f"skip {f.name}: bad result", file=sys.stderr); continue
        if only and r.get("verified") != "true": continue
        rows[(r.get("app"), r.get("gpu"), r.get("app_version"))] = (f.name, r)  # later file supersedes
    print("| App | Version | GPU | Driver | OS | Result | Status | Detail |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for _, (name, r) in sorted(rows.items(), key=lambda kv: tuple(map(str, kv[0]))):
        status = "verified" if r.get("verified") == "true" else "community-reported"
        print("| {app} | {app_version} | {gpu} | {driver} | {os} | {result} | {st} | {detail} |".format(
            st=status, **{k: r.get(k, "").replace("|", "/") for k in ("app","app_version","gpu","driver","os","result","detail")}))
main(sys.argv[1:])
