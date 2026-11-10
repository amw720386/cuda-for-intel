"""torch.linalg eigen and singular value routines on the GPU against the CPU. torch 2.2 runs
them through NVIDIA's own cuSOLVER (syevd/heevd, syevjBatched, gesvdj/gesvdjBatched, gesvd),
whose kernels go through the translator. Run under cfi-run with an untouched torch\\lib:
    cfi-run.exe --install-dir <bin> -- python solver_check.py
Values are compared directly. Vectors are checked by residuals (A V - V diag(w), U S Vh - A)
and orthonormality, because their signs and phases are not unique."""
import sys, time, torch
t0 = time.time()
assert torch.cuda.is_available(), "no cuda"
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
fails, n_ok = [], 0
g = torch.Generator().manual_seed(0)
TOL = {torch.float32: 2e-4, torch.float64: 1e-10, torch.complex64: 2e-4, torch.complex128: 1e-10}
VERBOSE = "-v" in sys.argv
DTYPES = [torch.float32, torch.float64, torch.complex64, torch.complex128]


def rel(got, want):
    if not want.numel():
        return 0.0
    return ((got - want).abs().max() / want.abs().max().clamp_min(1.0)).item()


def check(name, fn, tol):
    """fn() returns (error, ok_shape) or raises."""
    global n_ok
    if VERBOSE:
        print("..", name, flush=True)
    try:
        err = fn()
        torch.cuda.synchronize()
        if err <= tol:
            n_ok += 1
        else:
            fails.append(f"{name}: rel err {err:.2e}")
    except Exception as e:
        fails.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")


def rnd(*shape, dt):
    return torch.randn(*shape, dtype=dt, generator=g)


def herm(*shape, dt):
    a = rnd(*shape, dt=dt)
    return (a + a.mH) / 2


def eye_like(n, batch, dt):
    return torch.eye(n, dtype=dt).expand(*batch, n, n)


def eig_vals(a):
    return rel(torch.linalg.eigvalsh(a.cuda()).cpu(), torch.linalg.eigvalsh(a))


def eig_vecs(a, uplo):
    w, v = torch.linalg.eigh(a.cuda(), UPLO=uplo)
    w, v = w.cpu(), v.cpu()
    n = a.shape[-1]
    want = torch.linalg.eigvalsh(a, UPLO=uplo)
    resid = (a @ v - v * w.unsqueeze(-2).to(a.dtype)).abs().max().item() / max(1.0, a.abs().max().item())
    orth = (v.mH @ v - eye_like(n, a.shape[:-2], a.dtype)).abs().max().item()
    return max(rel(w, want), resid, orth)


def svd_vals(a):
    return rel(torch.linalg.svdvals(a.cuda()).cpu(), torch.linalg.svdvals(a))


def svd_full(a, full):
    u, s, vh = torch.linalg.svd(a.cuda(), full_matrices=full)
    u, s, vh = u.cpu(), s.cpu(), vh.cpu()
    m, n = a.shape[-2:]
    k = min(m, n)
    recon = (u[..., :, :k] * s.unsqueeze(-2).to(a.dtype)) @ vh[..., :k, :]
    err = rel(recon, a)
    err = max(err, (u.mH @ u - eye_like(u.shape[-1], a.shape[:-2], a.dtype)).abs().max().item())
    err = max(err, (vh @ vh.mH - eye_like(vh.shape[-2], a.shape[:-2], a.dtype)).abs().max().item())
    want_shape = (m, m) if full else (m, k)
    if tuple(u.shape[-2:]) != want_shape:
        return float("inf")
    return max(err, rel(s, torch.linalg.svdvals(a)))


for dt in DTYPES:
    tol = TOL[dt]
    tag = str(dt).replace("torch.", "")
    # syevd/heevd for single matrices of any size; syevjBatched for small batched ones.
    for batch, n in [((), 1), ((), 5), ((), 8), ((), 17), ((), 32), ((), 33), ((), 64), ((), 100), ((), 160),
                     ((4,), 16), ((3,), 32), ((2,), 40), ((2, 3), 9)]:
        a = herm(*batch, n, n, dt=dt)
        check(f"eigvalsh {tag} {batch} n={n}", lambda: eig_vals(a), tol * 10)
        check(f"eigh {tag} {batch} n={n} L", lambda: eig_vecs(a, "L"), tol * 10)
    a = herm(48, 48, dt=dt)
    check(f"eigh {tag} n=48 U", lambda: eig_vecs(a, "U"), tol * 10)
    # Clustered and repeated eigenvalues (deflation in the divide-and-conquer merge).
    q, _ = torch.linalg.qr(rnd(40, 40, dt=dt))
    w = torch.cat([torch.full((10,), 1.0), torch.full((10,), 2.0), torch.linspace(3, 4, 20)]).to(dt)
    clus = (q * w.unsqueeze(-2)) @ q.mH
    clus = (clus + clus.mH) / 2
    check(f"eigvalsh {tag} clustered n=40", lambda: eig_vals(clus), tol * 10)
    check(f"eigh {tag} clustered n=40", lambda: eig_vecs(clus, "L"), tol * 100)
    d = torch.diag(torch.arange(1, 25, dtype=dt))
    check(f"eigh {tag} diagonal n=24", lambda: eig_vecs(d, "L"), tol * 10)
    # gesvdj / gesvdjBatched (small), gesvd (large); tall, wide, square.
    for batch, (m, n) in [((), (1, 1)), ((), (8, 8)), ((), (32, 32)), ((), (40, 25)), ((), (25, 40)), ((), (64, 64)),
                          ((), (100, 37)), ((), (130, 130)), ((4,), (16, 16)), ((3,), (32, 20)), ((2,), (20, 32)),
                          ((2,), (48, 48))]:
        a = rnd(*batch, m, n, dt=dt)
        check(f"svdvals {tag} {batch} {m}x{n}", lambda: svd_vals(a), tol * 10)
        check(f"svd {tag} {batch} {m}x{n} reduced", lambda: svd_full(a, False), tol * 10)
        if not batch and max(m, n) <= 64:
            check(f"svd {tag} {m}x{n} full", lambda: svd_full(a, True), tol * 10)
    # pinv (svd or eigh based) and matrix_rank on rank-deficient input.
    for (m, n, r) in [(30, 20, 20), (20, 30, 12), (32, 32, 7)]:
        a = rnd(m, r, dt=dt) @ rnd(r, n, dt=dt)
        check(f"pinv {tag} {m}x{n} rank {r}", lambda: rel(torch.linalg.pinv(a.cuda()).cpu(), torch.linalg.pinv(a)), tol * 1000)
        check(f"matrix_rank {tag} {m}x{n} rank {r}",
              lambda: float(torch.linalg.matrix_rank(a.cuda()).item() != r), 0.0)
    h = herm(24, 24, dt=dt)
    check(f"pinv hermitian {tag} n=24", lambda: rel(torch.linalg.pinv(h.cuda(), hermitian=True).cpu(), torch.linalg.pinv(h, hermitian=True)), tol * 1000)
    check(f"matrix_rank hermitian {tag} n=24", lambda: float(torch.linalg.matrix_rank(h.cuda(), hermitian=True).item() != 24), 0.0)
    ab = rnd(3, 12, 4, dt=dt) @ rnd(3, 4, 10, dt=dt)
    check(f"matrix_rank {tag} batched rank 4", lambda: float((torch.linalg.matrix_rank(ab.cuda()).cpu() != 4).any().item()), 0.0)
    # backward through eigh and svd (uses the vectors).
    if dt in (torch.float64, torch.complex128):
        x = herm(12, 12, dt=dt)

        def grad_eigh(dev):
            y = x.to(dev).requires_grad_()
            w, v = torch.linalg.eigh(y)
            (w.sum() + (v.abs() ** 2).sum()).backward()
            return y.grad.cpu()
        check(f"eigh backward {tag}", lambda: rel(grad_eigh("cuda"), grad_eigh("cpu")), 1e-8)
        z = rnd(14, 9, dt=dt)

        def grad_svdvals(dev):
            y = z.to(dev).requires_grad_()
            torch.linalg.svdvals(y).sum().backward()
            return y.grad.cpu()
        check(f"svdvals backward {tag}", lambda: rel(grad_svdvals("cuda"), grad_svdvals("cpu")), 1e-8)

print(f"{n_ok} checks passed, {len(fails)} failed, {time.time() - t0:.1f} s")
for f_ in fails:
    print("FAIL", f_)
sys.exit(1 if fails else 0)
