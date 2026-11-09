#!/usr/bin/env python3
"""Diff ZLUDA implemented entry points vs cuda-for-intel. usage: parity.py ZLUDA_DIR OURS_DIR OUTDIR"""
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
