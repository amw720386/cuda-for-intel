#!/usr/bin/env python3
"""Generates the SVG assets in this folder: python3 make_assets.py"""
import os
D = os.path.dirname(os.path.abspath(__file__))
HEAD = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-labelledby="t d">'
 '<title id="t">{t}</title><desc id="d">{d}</desc>'
 "<style>text{{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;fill:#e6edf3}} "
 ".m{{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}} .mut{{fill:#8b949e}}</style>"
 '<rect width="{w}" height="{h}" rx="16" fill="#0d1117" stroke="#30363d"/>')
def box(x,y,w,h,title,sub,accent,mono=False):
    c = ' class="m"' if mono else ''
    return (f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="#161b22" stroke="#30363d"/>'
            f'<rect x="{x}" y="{y}" width="7" height="{h}" rx="3.5" fill="{accent}"/>'
            f'<text x="{x+24}" y="{y+h/2-3}" font-size="17" font-weight="bold"{c}>{title}</text>'
            f'<text x="{x+24}" y="{y+h/2+17}" font-size="12.5" class="mut">{sub}</text>')
def arrow(x,y1,y2):
    return f'<path d="M{x} {y1} V{y2-6} m-6 -6 l6 6 l6 -6" stroke="#8b949e" stroke-width="2" fill="none"/>'

# ---------- architecture ----------
W,H=900,760
s=HEAD.format(w=W,h=H,t="CUDA for Intel architecture",d="CUDA app, genuine CUDA runtime, Rust nvcuda shim, PTX to SPIR-V translator with kernel substitution, Level Zero, Intel Arc GPU.")
s+='<text x="450" y="32" font-size="15" class="mut" text-anchor="middle">CUDA for Intel: the whole path from app to Arc</text>'
rows=[("CUDA app","Stock Ollama, llama.cpp, PyTorch, your own PTX-carrying program","#58a6ff",False),
("CUDA Runtime (genuine NVIDIA cudart)","unchanged; calls the Driver API","#8b949e",False),
("nvcuda.dll shim (Rust)","Driver API, contexts, streams, graphs; cuBLAS / cuBLASLt stand-ins","#f2c14e",False),
("PTX parser + normaliser","reads the PTX the CUDA program carries","#3fb950",False),
("PTX to SPIR-V translator","OpenCL-environment SPIR-V","#3fb950",False),
("Kernel substitution layer (opt-in switch)","recognised ggml kernels (mmvq, quantize_q8_1, Q8_0 mmq) swapped for hand-written Intel SPIR-V;  falls back to the translated kernel on any mismatch","#bc8cff",False),
("Level Zero","loads the module, launches the kernel","#58a6ff",False),
("Intel Arc GPU (Arc B390)","runs it","#1f6feb",False)]
y=56;bh=66
for i,(a,b,c,m) in enumerate(rows):
    h=bh+(14 if i==5 else 0)
    s+=box(110,y,680,h,a,b if i!=5 else b[:b.rfind(' ',0,80)]+"<tspan x='134' dy='15'>"+b[b.rfind(' ',0,80):]+"</tspan>",c)
    # glow pulse sequenced
    s+=(f'<rect x="110" y="{y}" width="680" height="{h}" rx="12" fill="{c}" opacity="0"><animate attributeName="opacity" '
        f'values="0;0;.28;0;0" keyTimes="0;{i/9:.3f};{(i+.5)/9:.3f};{(i+1)/9:.3f};1" dur="9s" repeatCount="indefinite"/></rect>')
    if i<len(rows)-1: s+=arrow(450,y+h+2,y+h+16)
    y+=h+18
s+='<text x="450" y="%d" font-size="12" class="mut" text-anchor="middle">PTX only (no SASS/cubin). Experimental. Details: docs/architecture.md</text></svg>'%(H-14)
open(f"{D}/architecture.svg","w").write(s)

# ---------- pipeline animation ----------
W,H=900,300
s=HEAD.format(w=W,h=H,t="A kernel launch travelling through the shim",d="A packet labelled PTX moves from the CUDA app through the shim and translator, becomes SPIR-V, goes through Level Zero and reaches the Arc GPU.")
s+='<text x="450" y="30" font-size="15" class="mut" text-anchor="middle">one kernel launch, end to end</text>'
st=[("","CUDA app"),("","nvcuda shim"),("","PTX → SPIR-V"),("","Level Zero"),("","Arc GPU")]
xs=[90,270,450,630,810]
s+='<line x1="90" y1="150" x2="810" y2="150" stroke="#30363d" stroke-width="3" stroke-dasharray="4 6"/>'
for (e,n),x in zip(st,xs):
    s+=f'<rect x="{x-62}" y="104" width="124" height="92" rx="14" fill="#161b22" stroke="#30363d"/><text x="{x}" y="132" font-size="22" text-anchor="middle" class="m" style="fill:#8b949e">{xs.index(x)+1}</text><text x="{x}" y="178" font-size="13.5" text-anchor="middle" font-weight="bold">{n}</text>'
s+='<rect x="388" y="214" width="124" height="34" rx="9" fill="#161b22" stroke="#bc8cff" stroke-dasharray="4 3"/><text x="450" y="236" font-size="12.5" text-anchor="middle" class="m">substitute?</text>'
s+='<path d="M450 198 V212" stroke="#bc8cff" stroke-width="2"/>'
dur=7
# packet: PTX (blue) then SPIR-V (green) after translator
s+=(f'<g><circle r="13" cx="90" cy="150" fill="#58a6ff"><animate attributeName="cx" values="90;450;450" keyTimes="0;.5;1" dur="{dur}s" repeatCount="indefinite"/>'
    f'<animate attributeName="cy" values="150;150;150" dur="{dur}s" repeatCount="indefinite"/>'
    f'<animate attributeName="opacity" values="1;1;0;0" keyTimes="0;.49;.5;1" dur="{dur}s" repeatCount="indefinite"/></circle>'
    f'<circle r="13" fill="#3fb950" opacity="0"><animate attributeName="cx" values="450;810;810" keyTimes="0;.5;1" dur="{dur}s" begin="{dur/2}s" repeatCount="indefinite"/>'
    f'<animate attributeName="cy" values="150;150;150" dur="{dur}s" repeatCount="indefinite"/></circle></g>')
# simpler second packet using one cycle
s=s.replace(s[s.index('<circle r="13" fill="#3fb950"'):s.index('</g>',s.index('<circle r="13" fill="#3fb950"'))],
 f'<circle r="13" cx="450" fill="#3fb950" cy="150"><animate attributeName="cx" values="450;450;810;810" keyTimes="0;.5;1;1" dur="{dur}s" repeatCount="indefinite"/>'
 f'<animate attributeName="opacity" values="0;0;1;1;0" keyTimes="0;.5;.51;.98;1" dur="{dur}s" repeatCount="indefinite"/></circle>')
s+='<text x="90" y="282" font-size="13" fill="#58a6ff" text-anchor="middle">PTX</text><text x="810" y="282" font-size="13" fill="#3fb950" text-anchor="middle">SPIR-V</text>'
s+='</svg>'
open(f"{D}/pipeline.svg","w").write(s)

# ---------- chart ----------
data=[("Qwen2.5-0.5B Q8_0",1818,6767,"3.7x"),("Llama-3.2-1B Q4_K_M",817,4877,"6.0x"),("Qwen2.5-1.5B Q4_K_M",627,3727,"5.9x")]
W,H=900,470
s=HEAD.format(w=W,h=H,t="Prefill speed with kernel substitution off and on",d="Grouped bars of prefill tokens per second on Arc B390, stock Ollama, 631-token prompt: 0.5B 1818 to 6767, Llama 1B 817 to 4877, 1.5B 627 to 3727.")
s+='<text x="40" y="38" font-size="19" font-weight="bold">Prefill speed, kernel substitution off vs on</text>'
s+='<text x="40" y="60" font-size="12.5" class="mut">Intel Arc B390, stock Ollama v0.12.6, 631-token prompt, flash attention off. Higher is better.</text>'
x0,base,top,mx=60,390,100,7000
sc=lambda v:(base-top)*v/mx
for g in (0,2000,4000,6000):
    yy=base-sc(g); s+=f'<line x1="{x0}" y1="{yy}" x2="860" y2="{yy}" stroke="#30363d" stroke-width="1"/><text x="{x0-8}" y="{yy+4}" font-size="11" class="mut" text-anchor="end">{g}</text>'
for i,(n,a,b,r) in enumerate(data):
    cx=x0+30+i*260
    for j,(v,c,lab) in enumerate(((a,"#8b949e","off"),(b,"#58a6ff","on"))):
        x=cx+j*88; hh=sc(v)
        s+=f'<rect x="{x}" y="{base-hh}" width="80" height="{hh}" rx="4" fill="{c}"><animate attributeName="height" from="0" to="{hh}" dur="1s" fill="freeze"/><animate attributeName="y" from="{base}" to="{base-hh}" dur="1s" fill="freeze"/></rect>'
        s+=f'<text x="{x+40}" y="{base-hh-8}" font-size="13" font-weight="bold" text-anchor="middle">{v}</text><text x="{x+40}" y="{base+16}" font-size="11.5" class="mut" text-anchor="middle">{lab}</text>'
    s+=f'<text x="{cx+84}" y="{base+40}" font-size="13.5" font-weight="bold" text-anchor="middle">{n}</text>'
    s+=f'<text x="{cx+84}" y="{base-sc(b)-28}" font-size="16" font-weight="bold" fill="#3fb950" text-anchor="middle" style="fill:#3fb950">{r}</text>'
s+='<rect x="620" y="76" width="12" height="12" rx="2" fill="#8b949e"/><text x="638" y="87" font-size="12" class="mut">substitution off</text><rect x="750" y="76" width="12" height="12" rx="2" fill="#58a6ff"/><text x="768" y="87" font-size="12" class="mut">on</text>'
s+='<text x="450" y="452" font-size="11.5" class="mut" text-anchor="middle">prefill tokens/s. Measured on release candidate 7fc40fb. Decode speed is not part of this chart.</text></svg>'
open(f"{D}/prefill-chart.svg","w").write(s)
