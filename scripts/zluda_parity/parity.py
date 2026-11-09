#!/usr/bin/env python3
"""Diff ZLUDA implemented entry points vs cuda-for-intel.
usage: parity.py ZLUDA_DIR OURS_DIR OUTDIR [--torch-gaps] [--write-doc | --check-doc]

--write-doc rewrites the section 1 table of OURS_DIR/docs/zluda-parity.md (between the
parity-table markers); --check-doc exits 1 when that table is stale. CI runs --check-doc
against the ZLUDA commit pinned in .github/workflows/ci.yml."""
import re, sys, pathlib, json, types
Z, O, OUT = map(pathlib.Path, sys.argv[1:4]); OUT.mkdir(exist_ok=True, parents=True)

# --- ours: reuse scan() from scripts/api_audit.py without running main()
src = (O/"scripts/api_audit.py").read_text().replace("\nmain()\n", "\n")
mod = types.ModuleType("aa"); mod.__file__ = str(O/"scripts/api_audit.py"); exec(compile(src, "aa", "exec"), mod.__dict__)
def ours(dirs):
    """name -> is_stub; a symbol defined only in a stubs.rs file counts as a stub."""
    out = mod.scan(dirs)
    real = set()
    for d in dirs:
        for f in (O/d).rglob("*.rs"):
            if f.name != "stubs.rs": real |= set(mod.FN.findall(f.read_text()))
    return {n: (s or n not in real) for n, s in out.items()}

# --- zluda: declared = fn names in cuda_macros/src/<x>.rs ; implemented = names in non-default lists
def declared(f): return set(re.findall(r"\bfn\s+(\w+)\s*\(", (Z/"cuda_macros/src"/f).read_text()))
def zimpl(crate, macro):
    t = (Z/crate/"src/lib.rs").read_text()
    names = set()
    for m in re.finditer(r"cuda_macros::%s!\s*[\(\{]" % macro, t):
        i = m.end(); d = 1
        while d: d += {"(":1,"{":1,")":-1,"}":-1}.get(t[i],0); i += 1
        body = t[m.end():i-1]
        for lst in re.findall(r"<=\s*\[([^\]]*)\]", body):
            names |= set(re.findall(r"\b([A-Za-z_]\w*)\b", lst))
    return names
def zdnn():
    t = (Z/"zluda_dnn/src/lib.rs").read_text()
    s = t[t.index("dual_function_declarations! {"):t.index("cuda_macros::miopen_function_declarations")]
    return set(n for lst in re.findall(r"<=\s*\[([^\]]*)\]", s) for n in re.findall(r"\b(cudnn\w*)\b", lst))

LIBS = [  # title, zluda declared file, zluda implemented, our dirs, name filter
 ("Driver API (nvcuda)", "cuda.rs", lambda: zimpl("zluda","cuda_function_declarations"), ["src"], lambda n: n.startswith("cu") and not n.startswith(("cublas","cudnn","cufft","cusparse","cusolver"))),
 ("cuBLAS", "cublas.rs", lambda: zimpl("zluda_blas","cublas_function_declarations"), ["crates/cublas_shim/src"], lambda n: n.startswith("cublas") and not n.startswith("cublasLt")),
 ("cuBLASLt", "cublaslt.rs", lambda: zimpl("zluda_blaslt","cublaslt_function_declarations"), ["crates/cublas_shim/src"], lambda n: n.startswith("cublasLt")),
 ("cuDNN 9", "cudnn9.rs", zdnn, ["crates/cudnn_shim/src"], lambda n: n.startswith("cudnn")),
 ("cuFFT", "cufft.rs", lambda: zimpl("zluda_fft","cufft_function_declarations"), ["crates/cufft_shim/src"], lambda n: n.startswith("cufft")),
 ("cuSPARSE", "cusparse.rs", lambda: zimpl("zluda_sparse","cusparse_function_declarations"), ["crates/cusparse_shim/src"], lambda n: n.startswith("cusparse")),
 ("NVML", "nvml.rs", lambda: zimpl("zluda_ml","nvml_function_declarations"), ["crates/nvml_shim/src"], lambda n: n.startswith("nvml")),
]
torch = set(l.split("\t")[1] for l in (O/"scripts/api_reference/torch_cuda_dll_imports.tsv").read_text().splitlines() if "\t" in l)
rows = []; detail = {}
for title, decl, zf, dirs, filt in LIBS:
    zd = {n for n in declared(decl) if filt(n)}; zi = {n for n in zf() if filt(n)} & (zd | zf())
    o = {n: s for n, s in ours(dirs).items() if filt(n)}
    oi = {n for n, s in o.items() if not s}; os_ = {n for n, s in o.items() if s}
    both = zi & oi; zonly = zi - oi; oonly = oi - zi
    tz = len(zi & torch); to = len(oi & torch); tref = len(zd & torch)
    rows.append((title, len(zd), len(zi), len(o), len(oi), len(both), len(zonly), len(oonly), f"{tref}: Z {tz} / ours {to}"))
    detail[title] = {"zluda_only": sorted(zonly), "zluda_only_we_stub": sorted(zonly & os_), "zluda_only_we_miss": sorted(zonly - set(o)), "ours_only": sorted(oonly)}
# extra libs: ours only
for title, dirs, pre in [("cuSOLVER", ["crates/cusolver_shim/src"], "cusolver"), ("NVRTC", ["crates/nvrtc_shim/src"], "nvrtc"), ("cudart (own)", ["crates/cudart_shim/src"], "cuda")]:
    o = {n: s for n, s in ours(dirs).items() if n.startswith(pre)}
    oi = [n for n, s in o.items() if not s]
    rows.append((title, "-", 0, len(o), len(oi), 0, 0, len(oi), f"{len(set(o)&torch)} imported; ours impl {len(set(oi)&torch)}"))
    detail[title] = {"ours_impl": sorted(oi)}
# --- docs/zluda-parity.md section 1 table
TORCH_NOTE = {"Driver API (nvcuda)": "n/a (torch links cudart)",
              "cuDNN 9": "not in the import list (torch loads cuDNN lazily)"}
def doc_rows():
    out = []
    for r in rows[:len(LIBS)]:
        title, zd, zi, _, oi, both, zonly, oonly, _t = r
        d = next(x for x in LIBS if x[0] == title)
        zimpl_set = {n for n in d[2]() if d[4](n)}
        o = {n: st for n, st in ours(d[3]).items() if d[4](n)}
        oimpl = {n for n, st in o.items() if not st}
        tref = len({n for n in declared(d[1]) if d[4](n)} & torch)
        tcol = TORCH_NOTE.get(title) or (f"{tref}: {len(zimpl_set & torch)} / {len(oimpl & torch)}" if tref else "0")
        out.append([title, zd, zi, oi, both, zonly, oonly, tcol])
    def ours_lib(dirs, pre):
        o = {n: st for n, st in ours(dirs).items() if n.startswith(pre)}
        oi = sorted(n for n, st in o.items() if not st)
        return o, oi
    has = lambda c: (Z/c).is_dir()
    o, oi = ours_lib(["crates/cusolver_shim/src"], "cusolver")
    note = " (version getters)" if oi and all("Version" in n or "Property" in n for n in oi) else ""
    t = {n for n in torch if n.startswith("cusolver")}
    out.append(["cuSOLVER", "no crate" if not has("zluda_solver") else "?", 0, f"{len(oi)}{note}", 0, 0, len(oi), f"{len(t)}: 0 / {len(set(oi) & t)}"])
    out.append(["cuRAND", "no crate", 0, "no crate" if not (O/"crates/curand_shim").is_dir() else "?", "-", "-", "-", "torch does not import it (it uses Philox in ATen)"])
    o, oi = ours_lib(["crates/nvrtc_shim/src"], "nvrtc")
    t = {n for n in torch if n.startswith("nvrtc")}
    out.append(["NVRTC", "no crate (uses NVIDIA's NVRTC, see 4)", 0, len(oi), "-", "-", len(oi), f"{len(t)}: - / {len(set(oi) & t)}"])
    out.append(["nvJitLink", "no crate", 0, "no crate" if not (O/"crates/nvjitlink_shim").is_dir() else "?", "-", "-", "-", "0"])
    o, oi = ours_lib(["crates/cudart_shim/src"], "cuda")
    t = {n for n in torch if n.startswith("cuda")}
    out.append(["cudart", "none (uses NVIDIA's cudart)", "-", f"{len(oi)} (own `cudart_shim`)", "-", "-", "-", f"{len(t)}: - / {len(set(oi) & t)}"])
    lines = ["| Library | ZLUDA declared | ZLUDA implemented | ours implemented | both | ZLUDA only | ours only | torch_cuda.dll imports: ZLUDA impl / ours impl |",
             "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in out]
    return "\n".join(lines)

BEGIN, END = "<!-- parity-table:begin (generated by scripts/zluda_parity/parity.py --write-doc) -->", "<!-- parity-table:end -->"
if "--write-doc" in sys.argv or "--check-doc" in sys.argv:
    doc = O/"docs/zluda-parity.md"
    text = doc.read_text()
    a, b = text.index(BEGIN) + len(BEGIN), text.index(END)
    new = text[:a] + "\n" + doc_rows() + "\n" + text[b:]
    if "--check-doc" in sys.argv:
        if new != text:
            sys.stderr.write("docs/zluda-parity.md section 1 is stale: run scripts/zluda_parity/parity.py <zluda> . <outdir> --write-doc\n")
            sys.exit(1)
        print("docs/zluda-parity.md section 1 is current")
        sys.exit(0)
    doc.write_text(new)

hdr = "| Library | ZLUDA declared | ZLUDA implemented | ours exported | ours implemented | both impl | ZLUDA-only impl | ours-only impl | torch_cuda imports (in ref): impl Z / ours |"
print(hdr); print("|" + "---|"*9)
for r in rows: print("| " + " | ".join(map(str, r)) + " |")
(OUT/"parity.json").write_text(json.dumps(detail, indent=1))

# torch_cuda imports that neither side implements, per library
if "--torch-gaps" in sys.argv:
    for title, decl, zf, dirs, filt in LIBS + [("cuSOLVER", None, lambda: set(), ["crates/cusolver_shim/src"], lambda n: n.startswith("cusolver")), ("cudart", None, lambda: set(), ["crates/cudart_shim/src"], lambda n: n.startswith("cuda"))]:
        o = ours(dirs); zi = zf()
        t = sorted(n for n in torch if filt(n))
        miss = [n for n in t if (n not in o or o[n]) and n not in zi]
        zbeats = [n for n in t if n in zi and (n not in o or o[n])]
        print(f"## {title}: torch imports {len(t)}; neither implements {len(miss)}; ZLUDA-only {zbeats}")
        print("   neither:", " ".join(miss[:60]))
