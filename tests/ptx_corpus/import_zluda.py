"""Import ZLUDA PTX test kernels into tests/ptx_corpus/zluda/<family>/ with a license header.
usage: import_zluda.py <zluda checkout>   (read-only on the checkout; only reads text)"""
import re, sys
from pathlib import Path

Z = Path(sys.argv[1])
OUT = Path(__file__).parent / "zluda"
HEADER = ("// Derived from ZLUDA (https://github.com/vosen/ZLUDA), {src}\n"
          "// Licensed under MIT OR Apache-2.0. Copyright (c) ZLUDA contributors.\n"
          "// See LICENSE-ZLUDA-MIT and LICENSE-ZLUDA-APACHE in this directory.\n")
FAMILIES = [
 ("mma", r"^mma_"), ("ldmatrix", r"^ldmatrix"), ("cp_async", r"^cp_async|^createpolicy"),
 ("barrier", r"^bar|^membar"), ("shfl", r"^shfl"), ("vote_match_redux", r"^vote|^match|^redux|^activemask|^lanemask"),
 ("tex", r"^tex"), ("bitmanip", r"^(prmt|bfe|bfi|bfind|bmsk|brev|clz|popc|shf|dp[24]a|sad|lop3|fns|vshr)"),
 ("atomics", r"^atom"), ("cvt", r"^cvt|^uint_to"), ("float_math", r"^(sin|cos|ex2|lg2|rcp|rsqrt|sqrt|tanh|fma|fmax|copysign|div|mul_ftz|mul_non|add_ftz|min_|max|set_|neg|abs)"),
 ("integer", r"^(add|sub|mul|mad|min|max|rem|and|or|xor|not|shl|shr|sign|selp|setp|pred|neg|abs|addc|subc)"),
 ("memory", r"ld|st|mov|cvta|shared|global|local|param|const|stateful|reg_|vector|extern|non_scalar|implicit"),
 ("control", r".*"),
]
def fam(n):
    for f, r in FAMILIES:
        if re.search(r, n): return f
OUT.mkdir(exist_ok=True)
srcs = []
for sub in ["ptx/src/test/spirv_run", "ptx/src/test/spirv_build", "ptx/src/test", "zluda/src/impl/test_ptx", "ptx/src/pass/instruction_mode_to_global_mode"]:
    for p in sorted((Z / sub).glob("*.ptx")):
        srcs.append((p, sub))
for p, sub in srcs:
    f = "tex" if sub.endswith("test_ptx") else fam(p.stem)
    d = OUT / f; d.mkdir(exist_ok=True)
    dest = d / (p.name if sub.endswith("spirv_run") else f"{Path(sub).name}__{p.name}")
    dest.write_text(HEADER.format(src=f"{sub}/{p.name}") + p.read_text(errors="replace"))
(OUT / "LICENSE-ZLUDA-MIT").write_text((Z / "LICENSE-MIT").read_text())
(OUT / "LICENSE-ZLUDA-APACHE").write_text((Z / "LICENSE-APACHE").read_text())
print(len(srcs), "files")
