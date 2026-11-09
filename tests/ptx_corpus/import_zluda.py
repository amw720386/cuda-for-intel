"""Import ZLUDA PTX test kernels into tests/ptx_corpus/zluda/<family>/ with a license header.
usage: import_zluda.py <zluda checkout>              import the .ptx files (and the vectors)
       import_zluda.py <zluda checkout> --vectors    only regenerate zluda/vectors.tsv
Read-only on the checkout; only reads text.

vectors.tsv holds the `test_ptx!(name, [input], [output])` execution vectors from
ptx/src/test/spirv_run/mod.rs as little-endian hex bytes. ZLUDA's harness launches entry
`name` on one block of `block` threads (1, or 64 for `test_ptx_warp!`) with (input, output)
pointers, or (output, output) when there is no input, on a zero-filled output buffer.
src/ptx/zluda_vector_tests.rs runs them on the SPIR-V simulator."""
import re, struct, subprocess, sys
from fractions import Fraction
from pathlib import Path

Z = Path(sys.argv[1])
VECTORS_ONLY = "--vectors" in sys.argv[2:]
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

# ---- test_ptx! vectors -------------------------------------------------------------------
TYPES = {"u8": ("<B", 1), "i8": ("<b", 1), "u16": ("<H", 2), "i16": ("<h", 2), "u32": ("<I", 4),
         "i32": ("<i", 4), "u64": ("<Q", 8), "i64": ("<q", 8), "f32": ("<f", 4), "f64": ("<d", 8),
         "f16": ("<H", 2)}
INT_LIMITS = {t: ((0, (1 << (8 * TYPES[t][1])) - 1) if t[0] == "u" else
                  (-(1 << (8 * TYPES[t][1] - 1)), (1 << (8 * TYPES[t][1] - 1)) - 1))
              for t in TYPES if t[0] in "ui"}
F16_SPECIAL = {"NAN": 0x7E00, "INFINITY": 0x7C00, "NEG_INFINITY": 0xFC00}
F32_SPECIAL = {"NAN": 0x7FC00000, "INFINITY": 0x7F800000, "NEG_INFINITY": 0xFF800000}
F64_SPECIAL = {"NAN": 0x7FF8000000000000, "INFINITY": 0x7FF0000000000000, "NEG_INFINITY": 0xFFF0000000000000}

def round_f32(x, neg=False):
    """Correctly rounded f32 bits of an exact rational (Rust's `f32` literal parsing)."""
    sign = 0x80000000 if x < 0 or neg else 0
    if x == 0: return sign
    a = abs(Fraction(x))
    guess = struct.unpack("<I", struct.pack("<f", float(a)))[0]
    cands = [c for c in (guess - 1, guess, guess + 1) if 0 <= c < 0x7F800000]
    val = lambda b: Fraction(struct.unpack("<f", struct.pack("<I", b))[0])
    best = min(cands, key=lambda b: (abs(val(b) - a), b & 1))
    return sign | best

def f16_of_f32_bits(b):
    return struct.unpack("<H", struct.pack("<e", struct.unpack("<f", struct.pack("<I", b))[0]))[0]

def split_top(s, sep=","):
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([": depth += 1
        if ch in ")]": depth -= 1
        if ch == sep and depth == 0:
            out.append(cur.strip()); cur = ""
        else:
            cur += ch
    if cur.strip(): out.append(cur.strip())
    return out

NUM = re.compile(r"^(-?)(0x[0-9a-fA-F_]+|0b[01_]+|[0-9][0-9_]*(?:\.[0-9_]*)?(?:[eE][+-]?[0-9]+)?)(u8|u16|u32|u64|i8|i16|i32|i64|f32|f64)?$")

def literal_type(e):
    e = e.strip()
    m = NUM.match(e)
    if m: return m.group(3)
    m = re.match(r"^(?:std::)?(u8|u16|u32|u64|i8|i16|i32|i64|f32|f64)::", e)
    if m: return m.group(1)
    if e.startswith("half::f16::"): return "u16" if e.endswith(".to_bits()") else "f16"
    return None

def value_bits(e, t):
    """Bits (as an unsigned integer of the type's width) of one Rust element expression."""
    e = e.strip()
    m = NUM.match(e)
    if m:
        neg, body = m.group(1) == "-", m.group(2).replace("_", "")
        if t in ("f32", "f64", "f16"):
            v = Fraction(int(body, 0)) if body.startswith(("0x", "0b")) else Fraction(body)
            if t == "f32": return round_f32(v, neg)
            if t == "f64": return struct.unpack("<Q", struct.pack("<d", -float(v) if neg else float(v)))[0]
        v = int(body, 0) * (-1 if neg else 1)
        lo, hi = INT_LIMITS[t]
        assert lo <= v <= hi, (e, t)
        return v & ((1 << (8 * TYPES[t][1])) - 1)
    m = re.match(r"^(u8|u16|u32|u64|i8|i16|i32|i64)::(MAX|MIN|max_value\(\)|min_value\(\))$", e)
    if m:
        lo, hi = INT_LIMITS[m.group(1)]
        v = hi if "max" in m.group(2).lower() else lo
        return v & ((1 << (8 * TYPES[m.group(1)][1])) - 1)
    m = re.match(r"^(f32|f64)::(NAN|INFINITY|NEG_INFINITY)$", e)
    if m: return (F32_SPECIAL if m.group(1) == "f32" else F64_SPECIAL)[m.group(2)]
    m = re.match(r"^f32::from_bits\((.*)\)$", e)
    if m: return value_bits(m.group(1), "u32")
    m = re.match(r"^std::f32::consts::PI(?:\s*/\s*2f32)?$", e)
    if m:
        pi = 0x40490FDB  # f32::consts::PI
        return pi - (1 << 23) if "/" in e else pi  # exact halving: exponent - 1
    m = re.match(r"^half::f16::(NAN|INFINITY|NEG_INFINITY)(\.to_bits\(\))?$", e)
    if m: return F16_SPECIAL[m.group(1)]
    m = re.match(r"^half::f16::from_bits\((.*)\)$", e)
    if m: return value_bits(m.group(1), "u16")
    m = re.match(r"^half::f16::from_f(32|64)\((.*)\)$", e)
    if m:
        inner = m.group(2)
        if m.group(1) == "32": return f16_of_f32_bits(value_bits(inner, "f32"))
        return struct.unpack("<H", struct.pack("<e", float(inner.replace("_", ""))))[0]
    raise ValueError(f"unhandled element expression {e!r}")

def encode(arr):
    """(type, hex) of a Rust array literal; untyped literals take the array's element type."""
    elems = split_top(arr.strip()[1:-1])
    types = [t for t in map(literal_type, elems) if t]
    if types: t = types[0]
    else: t = "f64" if any("." in x for x in elems) else "i32"
    size = TYPES[t][1]
    raw = b"".join(value_bits(x, t).to_bytes(size, "little") for x in elems)
    return t, len(elems), raw.hex()

def extract_vectors(src):
    code = "\n".join(l.split("//")[0] for l in src.splitlines())
    rows = []
    for m in re.finditer(r"^(test_ptx|test_ptx_with_32|test_ptx_warp)!\(", code, re.M):
        i, depth = m.end(), 1
        while depth:
            depth += {"(": 1, ")": -1}.get(code[i], 0); i += 1
        args = split_top(code[m.end():i - 1])
        name, kind = args[0], m.group(1)
        if kind == "test_ptx_warp":
            if len(args) != 2: continue
            inp, out, block = None, args[1], 64
        else:
            if len(args) != 3: continue
            inp, out, block = args[1], args[2], 1
        it, inn, ihex = encode(inp) if inp else ("-", 0, "-")
        ot, onn, ohex = encode(out)
        rows.append((name, kind, block, it, inn, ihex, ot, onn, ohex))
    return rows

def write_vectors():
    mod = Z / "ptx/src/test/spirv_run/mod.rs"
    rows = extract_vectors(mod.read_text())
    try:
        rev = subprocess.run(["git", "-C", str(Z), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        rev = "unknown"
    lines = ["# Derived from ZLUDA (https://github.com/vosen/ZLUDA) ptx/src/test/spirv_run/mod.rs at " + rev + ".",
             "# Licensed under MIT OR Apache-2.0. Copyright (c) ZLUDA contributors. See LICENSE-ZLUDA-MIT.",
             "# Regenerate: python3 -I tests/ptx_corpus/import_zluda.py <zluda> --vectors",
             "# name\tmacro\tptx\tblock\tin_type\tin_len\tin_hex\tout_type\tout_len\tout_hex"]
    for name, kind, block, it, inn, ihex, ot, onn, ohex in rows:
        ptx = f"{fam(name)}/{name}.ptx"
        assert (OUT / ptx).exists(), f"missing corpus file {ptx}"
        lines.append("\t".join(map(str, (name, kind, ptx, block, it, inn, ihex, ot, onn, ohex))))
    (OUT / "vectors.tsv").write_text("\n".join(lines) + "\n")
    print(len(rows), "vectors")

if VECTORS_ONLY:
    write_vectors(); sys.exit(0)

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
write_vectors()
