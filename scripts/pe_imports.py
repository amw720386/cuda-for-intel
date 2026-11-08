#!/usr/bin/env python3
"""Dump import (and delay-import) tables of a PE file without pefile.
usage: pe_imports.py file.dll  -> lines 'dll<TAB>symbol'"""
import mmap, struct, sys

def main(path):
    f = open(path, "rb"); m = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    pe = struct.unpack_from("<I", m, 0x3c)[0]
    assert m[pe:pe+4] == b"PE\0\0"
    nsec = struct.unpack_from("<H", m, pe+6)[0]
    optsz = struct.unpack_from("<H", m, pe+20)[0]
    opt = pe + 24
    magic = struct.unpack_from("<H", m, opt)[0]
    plus = magic == 0x20b
    dd = opt + (112 if plus else 96)
    secs = []
    so = opt + optsz
    for i in range(nsec):
        n, vs, va, rs, rp = struct.unpack_from("<8sIIII", m, so + 40*i)
        secs.append((va, max(vs, rs), rp))
    def rva(r):
        for va, sz, rp in secs:
            if va <= r < va + sz: return r - va + rp
        raise ValueError(hex(r))
    def cstr(o):
        e = m.find(b"\0", o); return m[o:e].decode("latin1")
    ptr = 8 if plus else 4
    def thunks(r):
        o = rva(r); out = []
        while True:
            v = struct.unpack_from("<Q" if plus else "<I", m, o)[0]
            if v == 0: break
            if v >> (63 if plus else 31) & 1: out.append("#%d" % (v & 0xffff))
            else: out.append(cstr(rva(v & 0x7fffffff) + 2))
            o += ptr
        return out
    imp_rva, imp_sz = struct.unpack_from("<II", m, dd + 8)
    if imp_rva:
        o = rva(imp_rva)
        while True:
            oft, _, _, name, ft = struct.unpack_from("<IIIII", m, o)
            if name == 0: break
            for s in thunks(oft or ft): print(cstr(rva(name)).lower(), s, sep="\t")
            o += 20
    d_rva, _ = struct.unpack_from("<II", m, dd + 13*8)
    if d_rva:
        o = rva(d_rva)
        while True:
            attrs, name, mod, iat, int_, *_ = struct.unpack_from("<IIIIIIII", m, o)
            if name == 0: break
            for s in thunks(int_): print(cstr(rva(name)).lower(), s, sep="\t")
            o += 32
main(sys.argv[1])
