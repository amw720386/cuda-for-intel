#!/usr/bin/env python3
"""Opcodes ZLUDA's ptx_parser accepts (derive_parser! rule heads) vs opcode strings our src/ptx mentions,
plus our corpus status per opcode family. usage: ptx_ops.py ZLUDA OURS"""
import re, sys, pathlib
Z, O = map(pathlib.Path, sys.argv[1:3])
zl = (Z/"ptx_parser/src/lib.rs").read_text()
# rule lines inside derive_parser!: lines like "    add{.sat}.s32 d, a, b;" or "mma.sync.aligned..."
zops = set()
body = zl[zl.index("derive_parser!("):]
for m in re.finditer(r"^    ([a-z][a-z0-9_]*)[.{ ][^\n]*=>", body, re.M):
    zops.add(m.group(1))
zops |= {"bra", "call", "createpolicy"}  # parsed outside the rule table
ours_txt = "\n".join(p.read_text(errors="replace") for p in (O/"src/ptx").rglob("*.rs")) + (O/"src/ptx.rs").read_text()
def ours_has(op): return re.search(r'"%s[." ]|"%s"' % (re.escape(op), re.escape(op)), ours_txt) is not None
# PTX ISA 9.x opcode universe (hand list of common opcodes, for "neither" detection)
extra = "wmma stmatrix wgmma tcgen05 mbarrier cp.async.bulk elect redux testp slct fns szext isspacep istypep vadd vsub vabsdiff vmin vmax vset vote match activemask suld sust sured tex tld4 txq applypriority discard getctarank mapa multimem setmaxnreg red alloca stacksave stackrestore".split()
rows = []
for op in sorted(zops | {e.split(".")[0] for e in extra}):
    rows.append((op, op in zops, ours_has(op)))
both = [r[0] for r in rows if r[1] and r[2]]; zonly = [r[0] for r in rows if r[1] and not r[2]]; oonly = [r[0] for r in rows if r[2] and not r[1]]; neither=[r[0] for r in rows if not r[1] and not r[2]]
print(f"ZLUDA parser opcodes: {len(zops)}; both: {len(both)}; ZLUDA-only: {len(zonly)} {zonly}; ours-only (of checked list): {len(oonly)} {oonly}; neither: {neither}")
