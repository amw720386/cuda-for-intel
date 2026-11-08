"""Caller-level checks of the cuDNN shim through stock PyTorch (cu118 wheel): what PyTorch/ONNX-style code actually does.

usage: python test_callers.py <helper|raw|off> <out.json>
  helper: cfi_torch.install() (shim on by default, unsupported ops routed to ATen)   <- the default-on configuration
  raw   : cudnn enabled with no wrapper (shows which op families the shim cannot run)
  off   : torch.backends.cudnn.enabled = False
Env: CFI_BENCH_ROOT (shim dir with nvcuda.dll, cudnn64_8.dll, hybrid/).  Result per case: ok/fail + NMSE vs CPU fp32/fp64.
"""
import os, sys, json, time, ctypes, traceback
R = os.environ['CFI_BENCH_ROOT']
mode, out = sys.argv[1], sys.argv[2]
os.environ.setdefault('CUDA_FOR_INTEL_HYBRID_DIR', R + r'\hybrid')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
ctypes.WinDLL(R + r'\nvcuda.dll')
if mode != 'off':
    ctypes.WinDLL(R + r'\cudnn64_8.dll')
import torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'torch'))
if mode == 'helper':
    import cfi_torch; cfi_torch.install()
else:
    torch.backends.cudnn.enabled = (mode == 'raw')
dev = 'cuda'
res = []
def nmse(a, b):
    a = a.double().cpu(); b = b.double().cpu()
    return float(((a - b) ** 2).sum() / (b ** 2).sum().clamp_min(1e-30))
TOL = {torch.float32: 1e-9, torch.float64: 1e-15, torch.float16: 1e-5, torch.bfloat16: 1e-3}
def case(name):
    def deco(fn):
        t = time.perf_counter()
        try:
            worst = fn()
            res.append(dict(name=name, ok=True, nmse=worst, s=round(time.perf_counter() - t, 2)))
        except Exception as e:
            res.append(dict(name=name, ok=False, err=(type(e).__name__ + ': ' + str(e))[:160], s=round(time.perf_counter() - t, 2)))
        print(res[-1], flush=True)
        return fn
    return deco
def conv_case(name, mod_fn, shape, dt, cl=False, tol=None, bench=False):
    import copy
    @case(name)
    def _():
        torch.manual_seed(1)
        m = mod_fn(); x = torch.randn(*shape)
        fmt = torch.channels_last if len(shape) == 4 else torch.channels_last_3d
        g = copy.deepcopy(m).to(dt).to(dev); xg = x.to(dt).to(dev)
        if cl and len(shape) >= 4: g = g.to(memory_format=fmt); xg = xg.to(memory_format=fmt)
        xg.requires_grad_(True)
        cpu = copy.deepcopy(g).cpu().double().to(memory_format=torch.contiguous_format)  # same rounded weights, fp64 math
        xc = xg.detach().cpu().double().contiguous().requires_grad_(True)
        cpu(xc).square().mean().backward()
        torch.backends.cudnn.benchmark = bench
        try:
            y = g(xg); y.float().square().mean().backward()
            torch.cuda.synchronize()
        finally:
            torch.backends.cudnn.benchmark = False
        yc = cpu(xc.detach())
        worst = max(nmse(y, yc), nmse(xg.grad, xc.grad), *[nmse(pg.grad, pc.grad) for pg, pc in zip(g.parameters(), cpu.parameters())])
        lim = tol or TOL[dt]
        assert worst < lim, f'nmse {worst:.3e} >= {lim:.1e}'
        return worst
C = nn.Conv2d
for dt in (torch.float32, torch.float16, torch.bfloat16):
    d = str(dt).split('.')[-1]
    for cl in (False, True):
        l = 'nhwc' if cl else 'nchw'
        conv_case(f'conv2d 3x3 s1 p1 {d} {l}', lambda: C(16, 32, 3, 1, 1), (4, 16, 20, 18), dt, cl)
        conv_case(f'conv2d 7x7 s2 p3 c3 {d} {l}', lambda: C(3, 16, 7, 2, 3, bias=False), (2, 3, 33, 31), dt, cl)
        conv_case(f'conv2d 1x1 s2 {d} {l}', lambda: C(16, 24, 1, 2), (3, 16, 15, 15), dt, cl)
        conv_case(f'conv2d dil2 s2 groups4 {d} {l}', lambda: C(16, 32, 3, 2, 2, dilation=2, groups=4), (3, 16, 21, 19), dt, cl)
        conv_case(f'conv2d depthwise {d} {l}', lambda: C(16, 16, 3, 1, 1, groups=16), (3, 16, 14, 14), dt, cl)
        conv_case(f'conv2d asym k(3,5) s(1,2) p(1,2) {d} {l}', lambda: C(8, 12, (3, 5), (1, 2), (1, 2)), (2, 8, 13, 17), dt, cl)
        conv_case(f'convT k4 s2 p1 {d} {l}', lambda: nn.ConvTranspose2d(16, 8, 4, 2, 1), (2, 16, 9, 9), dt, cl)
        conv_case(f'convT k3 s2 p1 op1 groups2 {d} {l}', lambda: nn.ConvTranspose2d(16, 8, 3, 2, 1, output_padding=1, groups=2), (2, 16, 9, 9), dt, cl)
    conv_case(f'conv1d k3 {d}', lambda: nn.Conv1d(8, 16, 3, 1, 1), (3, 8, 31), dt)
conv_case('conv2d padding=same k5 fp32', lambda: C(8, 8, 5, padding='same'), (2, 8, 16, 16), torch.float32)
conv_case('conv2d benchmark=True fp32', lambda: C(16, 32, 3, 1, 1), (4, 16, 20, 18), torch.float32, bench=True)
conv_case('conv2d benchmark=True fp16 nhwc', lambda: C(16, 32, 3, 1, 1), (4, 16, 20, 18), torch.float16, True, bench=True)
# families the shim has no kernel for: must work through the wrapper (helper/off), raise in raw
conv_case('conv3d k3 fp32', lambda: nn.Conv3d(4, 6, 3, 1, 1), (2, 4, 6, 7, 8), torch.float32)
conv_case('conv3d k3 fp16', lambda: nn.Conv3d(4, 6, 3, 1, 1), (2, 4, 6, 7, 8), torch.float16)
conv_case('convT3d fp32', lambda: nn.ConvTranspose3d(4, 6, 2, 2), (2, 4, 4, 4, 4), torch.float32)
conv_case('conv2d fp64', lambda: C(8, 8, 3, 1, 1), (2, 8, 12, 12), torch.float64)

@case('batchnorm2d train+eval fp32/fp16')
def _():
    worst = 0
    for dt in (torch.float32, torch.float16):
        torch.manual_seed(2)
        bn = nn.BatchNorm2d(8); x = torch.randn(4, 8, 9, 9)
        bn_c = bn.double(); xc = x.to(dt).float().double().requires_grad_(True)
        yc = bn_c(xc); yc.square().mean().backward()
        bg = nn.BatchNorm2d(8).to(dev).to(dt) if dt == torch.float16 else nn.BatchNorm2d(8).to(dev)
        xg = x.to(dt).to(dev).requires_grad_(True)
        y = bg(xg); y.float().square().mean().backward(); torch.cuda.synchronize()
        worst = max(worst, nmse(y, yc), nmse(xg.grad, xc.grad)); assert worst < (1e-9 if dt == torch.float32 else 5e-5), worst
        bg.eval(); bg(xg)
    return worst

@case('resnet-style block fwd/bwd + conv bias + residual (fp32)')
def _():
    torch.manual_seed(3)
    class B(nn.Module):
        def __init__(s):
            super().__init__(); s.a = C(16, 16, 3, 1, 1); s.b = nn.BatchNorm2d(16); s.c = C(16, 16, 3, 1, 1)
        def forward(s, x): return torch.relu(s.c(torch.relu(s.b(s.a(x)))) + x)
    m = B(); x = torch.randn(4, 16, 14, 14)
    mc = B(); mc.load_state_dict(m.state_dict()); mc = mc.double(); xc = x.double().requires_grad_(True)
    yc = mc(xc); yc.square().mean().backward()
    mg = B().to(dev); mg.load_state_dict(m.state_dict()); xg = x.to(dev).requires_grad_(True)
    y = mg(xg); y.square().mean().backward(); torch.cuda.synchronize()
    worst = max(nmse(y, yc), nmse(xg.grad, xc.grad)); assert worst < 1e-9, worst
    return worst

for name, mk in (('lstm', lambda: nn.LSTM(8, 12, 2, batch_first=True, bidirectional=True)), ('gru', lambda: nn.GRU(8, 12, 1)), ('rnn', lambda: nn.RNN(8, 12, 1))):
    @case(f'{name} fwd/bwd fp32')
    def _(mk=mk):
        torch.manual_seed(4)
        m = mk(); bf = getattr(m, 'batch_first', False)
        x = torch.randn(3, 5, 8) if bf else torch.randn(5, 3, 8)
        mc = mk(); mc.load_state_dict(m.state_dict()); mc = mc.double(); xc = x.double().requires_grad_(True)
        yc = mc(xc)[0]; yc.square().mean().backward()
        mg = mk().to(dev); mg.load_state_dict(m.state_dict()); xg = x.to(dev).requires_grad_(True)
        y = mg(xg)[0]; y.square().mean().backward(); torch.cuda.synchronize()
        worst = max(nmse(y, yc), nmse(xg.grad, xc.grad)); assert worst < 1e-8, worst
        return worst

@case('ctc_loss fp32')
def _():
    torch.manual_seed(5)
    lp = torch.randn(12, 3, 6).log_softmax(2); tg = torch.randint(1, 6, (3, 4), dtype=torch.int32)
    il = torch.full((3,), 12, dtype=torch.int32); tl = torch.full((3,), 4, dtype=torch.int32)
    lc = lp.double().requires_grad_(True); a = F.ctc_loss(lc, tg, il, tl)
    lg = lp.to(dev).requires_grad_(True); b = F.ctc_loss(lg, tg.to(dev), il.to(dev), tl.to(dev)); b.backward(); torch.cuda.synchronize()
    assert abs(float(a) - float(b)) < 1e-4 * abs(float(a)), (float(a), float(b)); return abs(float(a) - float(b))

@case('big batch + workspace pressure: conv 64x64x112x112 fp16, 3x3, fwd+bwd')
def _():
    c = C(64, 64, 3, 1, 1).to(dev).half(); x = torch.randn(32, 64, 112, 112, device=dev).half().requires_grad_(True)
    for _i in range(3):
        y = c(x); y.float().square().mean().backward()
    torch.cuda.synchronize(); assert torch.isfinite(y).all().item(); return 0.0

@case('repeat 200 small convs (no leak / handle reuse)')
def _():
    c = C(8, 8, 3, 1, 1).to(dev); x = torch.randn(2, 8, 10, 10, device=dev)
    for _i in range(200): y = c(x)
    torch.cuda.synchronize(); return 0.0

@case('zero-batch conv')
def _():
    c = C(8, 8, 3, 1, 1).to(dev); y = c(torch.randn(0, 8, 10, 10, device=dev)); assert y.shape == (0, 8, 10, 10); return 0.0

@case('non-contiguous (sliced / permuted) input')
def _():
    c = C(8, 8, 3, 1, 1); x = torch.randn(4, 16, 10, 10)[:, ::2]; xp = torch.randn(4, 10, 10, 8).permute(0, 3, 1, 2)
    import copy
    cg = copy.deepcopy(c).to(dev); cd = copy.deepcopy(c).double(); worst = 0
    for xx in (x, xp):
        y = cg(xx.to(dev)); yc = cd(xx.double())
        worst = max(worst, nmse(y, yc)); assert worst < 1e-9, worst
    return worst

json.dump(dict(mode=mode, results=res, torch=torch.__version__), open(out, 'w'), indent=1)
print('SUMMARY', mode, sum(r['ok'] for r in res), '/', len(res), 'passed', flush=True)
