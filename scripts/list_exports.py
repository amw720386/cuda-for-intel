#!/usr/bin/env python3
"""List #[no_mangle] exported fn names per component (static scan)."""
import re, sys, pathlib
root = pathlib.Path(__file__).resolve().parent.parent
comps = {"driver": ["src"], "cudart": ["crates/cudart_shim/src"],
         "cublas": ["crates/cublas_shim/src"], "nvrtc": ["crates/nvrtc_shim/src"]}
pat = re.compile(r'#\[(?:unsafe\()?no_mangle\)?\]\s*(?:#\[[^\]]*\]\s*)*pub\s+(?:unsafe\s+)?extern\s+"[a-zA-Z-]+"\s+fn\s+(\w+)', re.S)
for c, dirs in comps.items():
    names = set()
    for d in dirs:
        for p in (root / d).rglob("*.rs"):
            names |= set(pat.findall(p.read_text()))
    print(f"== {c} {len(names)}")
    for n in sorted(names): print(n)
