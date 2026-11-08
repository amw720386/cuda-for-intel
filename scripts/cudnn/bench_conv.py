"""Conv micro-benchmark through stock PyTorch on the shim (cu118 wheel).

usage: python bench_conv.py <aten|cudnn> <out.jsonl> [tag-filter]
  aten : torch.backends.cudnn.enabled=False (ATen im2col + cuBLAS path)
  cudnn: cudnn64_8.dll of this repo preloaded, torch.backends.cudnn.enabled=True
Env: CFI_BENCH_ROOT (shim dir with nvcuda.dll, cudnn64_8.dll, hybrid/), CFI_BENCH_BATCH
"""
import os, sys, json, time, ctypes
R = os.environ['CFI_BENCH_ROOT']
mode, out = sys.argv[1], sys.argv[2]
filt = sys.argv[3] if len(sys.argv) > 3 else ''
os.environ.setdefault('CUDA_FOR_INTEL_HYBRID_DIR', R + r'\hybrid')
os.environ.setdefault('CUDA_MODULE_LOADING', 'LAZY')
ctypes.WinDLL(R + r'\nvcuda.dll')
if mode == 'cudnn':
    ctypes.WinDLL(R + r'\cudnn64_8.dll')
if mode == 'cudnn' and os.path.exists(R + r'\direct.flag'):
    os.environ['CFI_CUDNN_DIRECT'] = '1'   # old direct-kernel path (pre-GEMM engine), for baselines
    mode = 'cudnn_direct'
import torch, torch.nn.functional as F
torch.backends.cudnn.enabled = (mode != 'aten')
torch.backends.cudnn.benchmark = False
B = int(os.environ.get('CFI_BENCH_BATCH', '8'))
# tag, C, K, H, kernel, stride, pad, groups, transposed
CFG = [
 ('r18_conv1_7x7s2', 3, 64, 224, 7, 2, 3, 1, 0),
 ('r18_3x3_64_56', 64, 64, 56, 3, 1, 1, 1, 0),
 ('r18_3x3s2_64to128', 64, 128, 56, 3, 2, 1, 1, 0),
 ('r18_3x3_128_28', 128, 128, 28, 3, 1, 1, 1, 0),
 ('r18_1x1s2_down', 64, 128, 56, 1, 2, 0, 1, 0),
 ('r18_3x3_256_14', 256, 256, 14, 3, 1, 1, 1, 0),
 ('r18_3x3_512_7', 512, 512, 7, 3, 1, 1, 1, 0),
 ('r50_1x1_256to64_56', 256, 64, 56, 1, 1, 0, 1, 0),
 ('r50_1x1_256to1024_14', 256, 1024, 14, 1, 1, 0, 1, 0),
 ('dw3x3_64_56', 64, 64, 56, 3, 1, 1, 64, 0),
 ('dw3x3s2_128_28', 128, 128, 28, 3, 2, 1, 128, 0),
 ('group4_3x3_128_28', 128, 128, 28, 3, 1, 1, 4, 0),
 ('convT_k4s2_64to32_32', 64, 32, 32, 4, 2, 1, 1, 1),
 ('convT_k2s2_128to64_28', 128, 64, 28, 2, 2, 0, 1, 1),
]
def sync(): torch.cuda.synchronize()
def timeit(fn, budget=3.0):
    fn(); sync()
    t = time.perf_counter(); fn(); sync(); one = time.perf_counter() - t
    n = max(1, min(20, int(budget / max(one, 1e-4))))
    t = time.perf_counter()
    for _ in range(n): fn()
    sync()
    return (time.perf_counter() - t) / n * 1e3
def nmse(a, b):
    a = a.double().cpu(); b = b.double().cpu()
    return float(((a - b) ** 2).sum() / (b ** 2).sum().clamp_min(1e-30))
res = []
for (tag, C, K, H, ks, st, pd, g, tr) in CFG:
  for dt in (torch.float32, torch.float16):
    for cl in (False, True):
      name = f'{tag}|{"fp16" if dt==torch.float16 else "fp32"}|{"nhwc" if cl else "nchw"}'
      if filt and filt not in name: continue
      rec = dict(name=name, mode=mode, N=B)
      try:
        mf = torch.channels_last if cl else torch.contiguous_format
        torch.manual_seed(1)
        x = torch.randn(B, C, H, H).to(dt)
        wshape = (C, K // g, ks, ks) if tr else (K, C // g, ks, ks)
        w = (torch.randn(*wshape) * 0.05).to(dt)
        f = (lambda x, w: F.conv_transpose2d(x, w, stride=st, padding=pd, groups=g)) if tr else \
            (lambda x, w: F.conv2d(x, w, stride=st, padding=pd, groups=g))
        # correctness on 2 images vs CPU fp32
        xs = x[:2].clone(); xc = xs.float().requires_grad_(); wc = w.float().requires_grad_()
        yc = f(xc, wc); gy = torch.randn_like(yc); yc.backward(gy)
        xg = xs.cuda().to(memory_format=mf).requires_grad_(); wg = w.cuda().to(memory_format=mf).requires_grad_()
        yg = f(xg, wg); yg.backward(gy.cuda().to(dt).to(memory_format=mf))
        rec.update(nmse_y=nmse(yg, yc), nmse_dx=nmse(xg.grad, xc.grad), nmse_dw=nmse(wg.grad, wc.grad))
        xg = x.cuda().to(memory_format=mf).requires_grad_(); wg = w.cuda().to(memory_format=mf).requires_grad_()
        with torch.no_grad(): rec['fwd_ms'] = timeit(lambda: f(xg, wg))
        yg = f(xg, wg); gyg = torch.randn_like(yg)
        def fb():
            xg.grad = None; wg.grad = None
            f(xg, wg).backward(gyg)
        rec['fwdbwd_ms'] = timeit(fb)
      except Exception as e:
        rec['error'] = repr(e)[:200]
      print(json.dumps(rec), flush=True)
      with open(out, 'a') as fh: fh.write(json.dumps(rec) + '\n')
