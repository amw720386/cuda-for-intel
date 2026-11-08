#!/usr/bin/env python3
"""ptxscan: report whether a CUDA binary (exe/dll/so/fatbin/cubin) carries PTX.

Usage: ptxscan.py FILE [--json]
Exit code: 0 = every fatbin has PTX, 1 = some fatbins SASS-only, 2 = no PTX at all,
3 = no CUDA device code found.
Layout (nvcc fatbin, rev 0x101): frame magic 0xBA55ED50, 16-byte header
(ver u16, hdr u16, size u64); entries 64B header: kind u16 (1=PTX,2=SASS,
4=ELF/cubin-like), hdr u32@4, payload u64@8, arch u32@28, flags u64@40.
Read-only, nothing executed.
"""
import sys, struct, mmap, json, collections

def scan(path):
    with open(path, 'rb') as f:
        m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        magic = b'\x50\xed\x55\xba'
        fatbins = []  # list of {arch: set(kinds)}
        pos = 0
        while True:
            p = m.find(magic, pos)
            if p < 0: break
            pos = p + 4
            if p + 16 > len(m): break
            ver, hsz, fsz = struct.unpack_from('<HHQ', m, p + 4)
            if ver != 1 or hsz != 16 or fsz > 1 << 31 or p + 16 + fsz > len(m): continue
            q, end = p + 16, p + 16 + fsz
            pos = end
            ents = []
            while q + 64 <= end:
                kind, _rev, ehs, psz = struct.unpack_from('<HHIQ', m, q)
                arch = struct.unpack_from('<I', m, q + 28)[0]
                if ehs < 64 or ehs > 4096: break
                ents.append((kind, arch))
                q += ehs + psz
            fatbins.append(ents)
        # standalone cubin: ELF with e_machine 190 (EM_CUDA)
        cubin = len(m) > 20 and m[:4] == b'\x7fELF' and struct.unpack_from('<H', m, 18)[0] == 190
        m.close()
    return fatbins, cubin

def main():
    if len(sys.argv) < 2: sys.exit(__doc__)
    fb, cubin = scan(sys.argv[1])
    ptx_archs, sass_archs = collections.Counter(), collections.Counter()
    sass_only = 0
    for ents in fb:
        if not any(k == 1 for k, _ in ents): sass_only += 1
        for k, a in ents:
            (ptx_archs if k == 1 else sass_archs)[a] += 1
    rep = dict(file=sys.argv[1], fatbins=len(fb), standalone_cubin=cubin,
               fatbins_without_ptx=sass_only,
               ptx_archs={f'sm_{a}': n for a, n in sorted(ptx_archs.items())},
               sass_archs={f'sm_{a}': n for a, n in sorted(sass_archs.items())})
    if '--json' in sys.argv: print(json.dumps(rep, indent=1))
    else:
        print(f"{rep['file']}: {len(fb)} fatbin(s), {sass_only} without PTX" + (", standalone cubin ELF (SASS only)" if cubin else ""))
        print("  PTX :", rep['ptx_archs'] or 'none')
        print("  SASS:", rep['sass_archs'] or 'none')
        if (sass_only or cubin):
            print("  DIAGNOSTIC: this binary has device code the shim cannot run (SASS only, no PTX).\n"
                  "  Use a PTX-carrying build (see docs/sass-support.md), or set TORCH_CUDA_ARCH_LIST='...+PTX' when building.")
    if not fb and not cubin: sys.exit(3)
    sys.exit(2 if not ptx_archs and not (cubin and False) else (1 if sass_only or cubin else 0))
main()
