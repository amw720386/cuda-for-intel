"""ResNet18 forward/backward on one batch, GPU (cuDNN shim or ATen) vs CPU fp32, NMSE per tensor.
usage: python test_resnet18.py <aten|cudnn> [out.json]   (env CFI_BENCH_ROOT like bench_conv.py)
"""
import os, sys, json, time, ctypes
R = os.environ['CFI_BENCH_ROOT']
mode = sys.argv[1]
out = sys.argv[2] if len(sys.argv) > 2 else None
os.environ.setdefault('CUDA_FOR_INTEL_HYBRID_DIR', R + r'\hybrid')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
ctypes.WinDLL(R + r'\nvcuda.dll')
if mode in ('cudnn', 'direct'):
    ctypes.WinDLL(R + r'\cudnn64_8.dll')
if mode == 'direct':
    os.environ['CFI_CUDNN_DIRECT'] = '1'; mode = 'cudnn_direct'
import torch, torch.nn as nn
torch.backends.cudnn.enabled = (mode != 'aten')
N = int(os.environ.get('CFI_BENCH_BATCH', '4')); S = int(os.environ.get('CFI_RESNET_SIZE', '112'))

class Block(nn.Module):
    def __init__(s, i, o, st):
        super().__init__()
        s.c1 = nn.Conv2d(i, o, 3, st, 1, bias=False); s.b1 = nn.BatchNorm2d(o)
        s.c2 = nn.Conv2d(o, o, 3, 1, 1, bias=False); s.b2 = nn.BatchNorm2d(o)
        s.d = None
        if st != 1 or i != o:
            s.d = nn.Sequential(nn.Conv2d(i, o, 1, st, bias=False), nn.BatchNorm2d(o))
    def forward(s, x):
        y = torch.relu(s.b1(s.c1(x))); y = s.b2(s.c2(y))
        return torch.relu(y + (x if s.d is None else s.d(x)))
class ResNet18(nn.Module):
    def __init__(s):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(3, 64, 7, 2, 3, bias=False), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(3, 2, 1))
        cfg = [(64, 64, 1), (64, 64, 1), (64, 128, 2), (128, 128, 1), (128, 256, 2), (256, 256, 1), (256, 512, 2), (512, 512, 1)]
        s.blocks = nn.Sequential(*[Block(*c) for c in cfg])
        s.fc = nn.Linear(512, 10)
    def forward(s, x):
        x = s.blocks(s.stem(x)); return s.fc(torch.flatten(nn.functional.adaptive_avg_pool2d(x, 1), 1))

def nmse(a, b):
    a = a.double().cpu(); b = b.double().cpu()
    return float(((a - b) ** 2).sum() / (b ** 2).sum().clamp_min(1e-30))
def run(model, x, dev, cl, half):
    model = model.to(dev)
    if cl: model = model.to(memory_format=torch.channels_last)
    x = x.to(dev)
    if cl: x = x.to(memory_format=torch.channels_last)
    model.zero_grad()
    t = time.perf_counter()
    with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=half and dev == 'cuda'):
        y = model(x)
    loss = y.float().square().mean(); loss.backward()
    if dev == 'cuda': torch.cuda.synchronize()
    return y.float().detach().cpu(), {n: p.grad.detach().float().cpu() for n, p in model.named_parameters()}, time.perf_counter() - t

res = {}
torch.manual_seed(0)
ref = ResNet18(); x = torch.randn(N, 3, S, S)
import copy
yc, gc_, _ = run(copy.deepcopy(ref), x, 'cpu', False, False)
for cl in (False, True):
    for half in (False, True):
        m = copy.deepcopy(ref)
        run(copy.deepcopy(ref), x, 'cuda', cl, half)  # warm-up (kernel translation)
        yg, gg, dt = run(m, x, 'cuda', cl, half)
        worst = max(nmse(gg[k], gc_[k]) for k in gc_)
        key = f'{"nhwc" if cl else "nchw"}_{"fp16" if half else "fp32"}'
        res[key] = dict(nmse_logits=nmse(yg, yc), nmse_grad_worst=worst, step_s=dt)
        print(key, res[key], flush=True)
res['mode'] = mode; res['N'] = N; res['size'] = S
if out: json.dump(res, open(out, 'w'))
