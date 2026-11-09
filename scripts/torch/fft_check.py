"""torch.fft on the GPU against torch.fft on the CPU (cuFFT shim acceptance, roadmap item 9).
Run under cfi-run: cfi-run.exe --install-dir <bin> -- python fft_check.py"""
import sys, time, torch
t0 = time.time()
assert torch.cuda.is_available(), "no cuda"
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
fails, n_ok = [], 0
def check(name, got, want, tol):
    global n_ok
    got = got.cpu()
    err = ((got - want).abs().max() / want.abs().max().clamp_min(1e-30)).item()
    if err <= tol: n_ok += 1
    else: fails.append(f"{name}: rel err {err:.2e}")
g = torch.Generator().manual_seed(0)
sizes = [1, 2, 3, 7, 8, 16, 64, 97, 100, 105, 128, 210, 256, 400, 512, 1000, 1024, 2048, 3 * 5 * 7 * 8, 4096]
for dt, ct, tol in [(torch.float32, torch.complex64, 1e-4), (torch.float64, torch.complex128, 1e-10)]:
    for n in sizes:
        x = torch.randn(3, n, dtype=dt, generator=g)
        z = torch.randn(3, n, dtype=ct, generator=g)
        xc, zc = x.cuda(), z.cuda()
        check(f"fft {n} {ct}", torch.fft.fft(zc), torch.fft.fft(z), tol)
        check(f"ifft {n} {ct}", torch.fft.ifft(zc), torch.fft.ifft(z), tol)
        check(f"rfft {n} {dt}", torch.fft.rfft(xc), torch.fft.rfft(x), tol)
        r = torch.fft.rfft(x)
        check(f"irfft {n} {dt}", torch.fft.irfft(r.cuda(), n=n), torch.fft.irfft(r, n=n), tol)
    for shape in [(16, 16), (30, 42), (64, 105), (3, 5, 7)]:
        x = torch.randn(2, *shape, dtype=dt, generator=g)
        z = torch.randn(2, *shape, dtype=ct, generator=g)
        dims = tuple(range(1, len(shape) + 1))
        check(f"fft2 {shape} {ct}", torch.fft.fft2(z.cuda()), torch.fft.fft2(z), tol)
        check(f"fftn {shape} {ct}", torch.fft.fftn(z.cuda(), dim=dims), torch.fft.fftn(z, dim=dims), tol)
        check(f"rfftn {shape} {dt}", torch.fft.rfftn(x.cuda(), dim=dims), torch.fft.rfftn(x, dim=dims), tol)
        check(f"irfftn {shape} {dt}", torch.fft.irfftn(torch.fft.rfftn(x, dim=dims).cuda(), s=shape, dim=dims), torch.fft.irfftn(torch.fft.rfftn(x, dim=dims), s=shape, dim=dims), tol)
    # non-last dim (strided layout) and STFT
    x = torch.randn(64, 12, dtype=dt, generator=g)
    check(f"fft dim0 {dt}", torch.fft.fft(x.cuda(), dim=0), torch.fft.fft(x, dim=0), tol)
    w = torch.randn(16000, dtype=dt, generator=g)
    win = torch.hann_window(400, dtype=dt)
    check(f"stft 400/160 {dt}", torch.stft(w.cuda(), 400, 160, window=win.cuda(), return_complex=True), torch.stft(w, 400, 160, window=win, return_complex=True), tol)
print(f"{n_ok} checks passed, {len(fails)} failed, {time.time() - t0:.1f} s")
for f in fails: print("FAIL", f)
sys.exit(1 if fails else 0)
