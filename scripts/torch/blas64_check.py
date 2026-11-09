"""float64 / complex64 / complex128 dense linear algebra on the GPU against the same call on
the CPU (f64 and complex cuBLAS acceptance). Run under cfi-run with an untouched torch\\lib:
    cfi-run.exe --install-dir <bin> -- python blas64_check.py
Checked ops must match. The "gaps" section only records what happens for torch.linalg
operations that need cuSOLVER or MAGMA rather than cuBLAS (they must fail cleanly or match)."""
import sys, time, torch
t0 = time.time()
assert torch.cuda.is_available(), "no cuda"
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
fails, n_ok = [], 0
g = torch.Generator().manual_seed(0)
TOL = {torch.float32: 1e-4, torch.float64: 1e-10, torch.complex64: 1e-4, torch.complex128: 1e-10}


def check(name, fn_gpu, fn_cpu, tol):
    global n_ok
    try:
        got = fn_gpu()
        torch.cuda.synchronize()
        got = got.cpu()
        want = fn_cpu()
        err = ((got - want).abs().max() / want.abs().max().clamp_min(1.0)).item() if want.numel() else 0.0
        if got.shape != want.shape:
            fails.append(f"{name}: shape {tuple(got.shape)} != {tuple(want.shape)}")
        elif err <= tol:
            n_ok += 1
        else:
            fails.append(f"{name}: rel err {err:.2e}")
    except Exception as e:
        fails.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")


def rnd(*shape, dt):
    return torch.randn(*shape, dtype=dt, generator=g)


def tri(n, dt, upper, batch=()):
    """Well-conditioned triangular matrix."""
    a = rnd(*batch, n, n, dt=dt) * 0.3 + 2.0 * torch.eye(n, dtype=dt)
    return a.triu() if upper else a.tril()


for dt in [torch.float64, torch.complex64, torch.complex128]:
    tol = TOL[dt]
    cplx = dt.is_complex
    alpha = (0.75 - 0.5j) if cplx else 0.75
    beta = (-0.5 + 0.25j) if cplx else -0.5
    for (m, k, n) in [(1, 1, 1), (37, 41, 29), (64, 64, 64), (128, 96, 33), (257, 70, 130), (5, 300, 3)]:
        a, b, c = rnd(m, k, dt=dt), rnd(k, n, dt=dt), rnd(m, n, dt=dt)
        at, bt = a.t().contiguous(), b.t().contiguous()  # stored transposed
        A, B, C, AT, BT = a.cuda(), b.cuda(), c.cuda(), at.cuda(), bt.cuda()
        tag = f"{m}x{k}x{n} {dt}"
        check(f"matmul {tag}", lambda: A @ B, lambda: a @ b, tol)
        check(f"matmul A^T-stored {tag}", lambda: AT.t() @ B, lambda: a @ b, tol)
        check(f"matmul B^T-stored {tag}", lambda: A @ BT.t(), lambda: a @ b, tol)
        check(f"addmm alpha/beta {tag}", lambda: torch.addmm(C, A, B, beta=beta, alpha=alpha), lambda: torch.addmm(c, a, b, beta=beta, alpha=alpha), tol)
        check(f"addmm beta=0 {tag}", lambda: torch.addmm(C, A, B, beta=0), lambda: torch.addmm(c, a, b, beta=0), tol)
        check(f"F.linear {tag}", lambda: torch.nn.functional.linear(A, BT), lambda: torch.nn.functional.linear(a, bt), tol)
        if cplx:
            ah, bh = a.conj().t().contiguous(), b.conj().t().contiguous()
            AH, BH = ah.cuda(), bh.cuda()
            check(f"matmul A^H (conj-transpose view) {tag}", lambda: AH.mH @ B, lambda: a @ b, tol)
            check(f"matmul B^H {tag}", lambda: A @ BH.mH, lambda: a @ b, tol)
            check(f"matmul conj(A) {tag}", lambda: A.conj() @ B, lambda: a.conj() @ b, tol)
            check(f"matmul conj(A) @ conj(B) {tag}", lambda: A.conj() @ B.conj(), lambda: a.conj() @ b.conj(), tol)
        # matrix-vector, contiguous and strided vectors, transposed matrix
        x, x2, y = rnd(k, dt=dt), rnd(2 * k, dt=dt), rnd(m, dt=dt)
        X, X2, Y = x.cuda(), x2.cuda(), y.cuda()
        check(f"mv {tag}", lambda: torch.mv(A, X), lambda: torch.mv(a, x), tol)
        check(f"mv strided x {tag}", lambda: torch.mv(A, X2[::2]), lambda: torch.mv(a, x2[::2]), tol)
        check(f"mv A^T-stored {tag}", lambda: torch.mv(AT.t(), X), lambda: torch.mv(a, x), tol)
        check(f"addmv alpha/beta {tag}", lambda: torch.addmv(Y, A, X, beta=beta, alpha=alpha), lambda: torch.addmv(y, a, x, beta=beta, alpha=alpha), tol)
        check(f"vec @ mat {tag}", lambda: Y @ A, lambda: y @ a, tol)
        if cplx:
            check(f"mv A^H {tag}", lambda: torch.mv(A.mH, Y), lambda: torch.mv(a.mH, y), tol)
    # batched products
    for (bs, m, k, n) in [(3, 20, 15, 6), (17, 64, 32, 48), (2, 1, 300, 1)]:
        a, b, c = rnd(bs, m, k, dt=dt), rnd(bs, k, n, dt=dt), rnd(bs, m, n, dt=dt)
        A, B, C = a.cuda(), b.cuda(), c.cuda()
        tag = f"{bs}x{m}x{k}x{n} {dt}"
        check(f"bmm {tag}", lambda: torch.bmm(A, B), lambda: torch.bmm(a, b), tol)
        check(f"bmm transposed {tag}", lambda: torch.bmm(B.mT, A.mT), lambda: torch.bmm(b.mT, a.mT), tol)
        check(f"baddbmm {tag}", lambda: torch.baddbmm(C, A, B, beta=beta, alpha=alpha), lambda: torch.baddbmm(c, a, b, beta=beta, alpha=alpha), tol)
        check(f"matmul broadcast {tag}", lambda: A.unsqueeze(0).expand(2, -1, -1, -1) @ B, lambda: a.unsqueeze(0).expand(2, -1, -1, -1) @ b, tol)
        if cplx:
            check(f"bmm A^H {tag}", lambda: torch.bmm(A.mH, C), lambda: torch.bmm(a.mH, c), tol)
    # dot products (torch runs these in device pointer mode)
    for n in [1, 1000, 4097]:
        x, y = rnd(2 * n, dt=dt), rnd(n, dt=dt)
        X, Y = x.cuda(), y.cuda()
        check(f"dot n={n} {dt}", lambda: torch.dot(X[:n], Y), lambda: torch.dot(x[:n], y), tol)
        check(f"dot strided n={n} {dt}", lambda: torch.dot(X[::2], Y), lambda: torch.dot(x[::2], y), tol)
        check(f"inner n={n} {dt}", lambda: torch.inner(X[:n], Y), lambda: torch.inner(x[:n], y), tol)
        check(f"linalg.vecdot n={n} {dt}", lambda: torch.linalg.vecdot(X[:n], Y), lambda: torch.linalg.vecdot(x[:n], y), tol)
        if cplx:
            check(f"vdot n={n} {dt}", lambda: torch.vdot(X[:n], Y), lambda: torch.vdot(x[:n], y), tol)
    # linalg routines that reach cuBLAS (gemm, trsm, trsmBatched)
    for n, nrhs in [(1, 1), (16, 3), (64, 64), (200, 7)]:
        for upper in [False, True]:
            t, bm = tri(n, dt, upper), rnd(n, nrhs, dt=dt)
            T, BM = t.cuda(), bm.cuda()
            tag = f"n={n} nrhs={nrhs} upper={upper} {dt}"
            check(f"solve_triangular left {tag}", lambda: torch.linalg.solve_triangular(T, BM, upper=upper), lambda: torch.linalg.solve_triangular(t, bm, upper=upper), tol * 10)
            check(f"solve_triangular unit {tag}", lambda: torch.linalg.solve_triangular(T, BM, upper=upper, unitriangular=True), lambda: torch.linalg.solve_triangular(t, bm, upper=upper, unitriangular=True), tol * 10)
            br = rnd(nrhs, n, dt=dt)
            check(f"solve_triangular right {tag}", lambda: torch.linalg.solve_triangular(T, br.cuda(), upper=upper, left=False), lambda: torch.linalg.solve_triangular(t, br, upper=upper, left=False), tol * 10)
            check(f"triangular_solve transpose {tag}", lambda: torch.triangular_solve(BM, T, upper=upper, transpose=True)[0], lambda: torch.triangular_solve(bm, t, upper=upper, transpose=True)[0], tol * 10)
            if cplx:
                check(f"solve_triangular A^H {tag}", lambda: torch.linalg.solve_triangular(T.mH, BM, upper=not upper), lambda: torch.linalg.solve_triangular(t.mH, bm, upper=not upper), tol * 10)
        tb, bb = tri(n, dt, False, (4,)), rnd(4, n, nrhs, dt=dt)
        check(f"solve_triangular batched n={n} {dt}", lambda: torch.linalg.solve_triangular(tb.cuda(), bb.cuda(), upper=False), lambda: torch.linalg.solve_triangular(tb, bb, upper=False), tol * 10)
    # small batched LU (cublas getrfBatched / getrsBatched)
    for bs, n in [(6, 8), (32, 4), (3, 16)]:
        sq = rnd(bs, n, n, dt=dt) + n * torch.eye(n, dtype=dt) * 0.5
        rhs = rnd(bs, n, 2, dt=dt)
        tag = f"{bs}x{n}x{n} {dt}"
        check(f"linalg.solve batched {tag}", lambda: torch.linalg.solve(sq.cuda(), rhs.cuda()), lambda: torch.linalg.solve(sq, rhs), tol * 100)
        check(f"linalg.inv batched {tag}", lambda: torch.linalg.inv(sq.cuda()), lambda: torch.linalg.inv(sq), tol * 100)
        check(f"linalg.det batched {tag}", lambda: torch.linalg.det(sq.cuda()), lambda: torch.linalg.det(sq), tol * 100)
        check(f"linalg.lu_factor batched {tag}", lambda: torch.linalg.lu_factor(sq.cuda())[0], lambda: torch.linalg.lu_factor(sq)[0], tol * 100)
        check(f"lu_solve batched {tag}", lambda: torch.linalg.lu_solve(*torch.linalg.lu_factor(sq.cuda()), rhs.cuda()), lambda: torch.linalg.lu_solve(*torch.linalg.lu_factor(sq), rhs), tol * 100)
    mats = [rnd(30, 40, dt=dt), rnd(40, 25, dt=dt), rnd(25, 60, dt=dt), rnd(60, 5, dt=dt)]
    check(f"linalg.multi_dot {dt}", lambda: torch.linalg.multi_dot([m.cuda() for m in mats]), lambda: torch.linalg.multi_dot(mats), tol)
    sq = rnd(48, 48, dt=dt) / 7
    check(f"linalg.matrix_power 5 {dt}", lambda: torch.linalg.matrix_power(sq.cuda(), 5), lambda: torch.linalg.matrix_power(sq, 5), tol * 10)
    check(f"chain_matmul {dt}", lambda: torch.linalg.matmul(torch.linalg.matmul(mats[0].cuda(), mats[1].cuda()), mats[2].cuda()), lambda: mats[0] @ mats[1] @ mats[2], tol)
    # autograd through matmul (gemm with transposed operands)
    a, b = rnd(33, 21, dt=dt), rnd(21, 17, dt=dt)
    def grads(a, b):
        a, b = a.clone().requires_grad_(), b.clone().requires_grad_()
        # view_as_real keeps the loss real without torch's complex abs kernel, which the
        # cu118 wheel ships as SASS only
        out = a @ b
        (torch.view_as_real(out) if out.is_complex() else out).sum().backward()
        return torch.cat([a.grad.flatten(), b.grad.flatten()])
    check(f"matmul backward {dt}", lambda: grads(a.cuda(), b.cuda()), lambda: grads(a, b), tol * 10)

# float32 entry points routed through the new kernels: device-mode dot, strided mv, trsm
f = torch.float32
x, y = rnd(2000, dt=f), rnd(1000, dt=f)
check("dot f32 (device result)", lambda: torch.dot(x.cuda()[::2], y.cuda()), lambda: torch.dot(x[::2], y), 1e-4)
a = rnd(70, 1000, dt=f)
check("mv f32 strided x", lambda: torch.mv(a.cuda(), x.cuda()[::2]), lambda: torch.mv(a, x[::2]), 1e-4)
t, bm = tri(64, f, True), rnd(64, 5, dt=f)
check("solve_triangular f32", lambda: torch.linalg.solve_triangular(t.cuda(), bm.cuda(), upper=True), lambda: torch.linalg.solve_triangular(t, bm, upper=True), 1e-4)
tb = tri(16, f, False, (6,))
check("solve_triangular f32 batched", lambda: torch.linalg.solve_triangular(tb.cuda(), torch.ones(6, 16, 2).cuda(), upper=False), lambda: torch.linalg.solve_triangular(tb, torch.ones(6, 16, 2), upper=False), 1e-4)

print(f"{n_ok} checks passed, {len(fails)} failed, {time.time() - t0:.1f} s")
for f_ in fails: print("FAIL", f_)

# informational: torch.linalg routines torch runs through NVIDIA's cuSOLVER (loaded from
# torch\lib, its cuBLAS calls served by the shim) or through cuBLAS routines the shim lacks
s = (rnd(32, 32, dt=torch.float64) + 8 * torch.eye(32, dtype=torch.float64))
spd = s @ s.mT
gaps = {
    "linalg.solve": lambda: (torch.linalg.solve(s.cuda(), torch.ones(32, 1, dtype=torch.float64).cuda()).cpu(), torch.linalg.solve(s, torch.ones(32, 1, dtype=torch.float64))),
    "linalg.inv": lambda: (torch.linalg.inv(s.cuda()).cpu(), torch.linalg.inv(s)),
    "linalg.det": lambda: (torch.linalg.det(s.cuda()).cpu(), torch.linalg.det(s)),
    "linalg.cholesky": lambda: (torch.linalg.cholesky(spd.cuda()).cpu(), torch.linalg.cholesky(spd)),
    "linalg.qr": lambda: (torch.linalg.qr(s.cuda())[1].abs().cpu(), torch.linalg.qr(s)[1].abs()),
    "linalg.eigh": lambda: (torch.linalg.eigvalsh(spd.cuda()).cpu(), torch.linalg.eigvalsh(spd)),
    "linalg.svd": lambda: (torch.linalg.svdvals(s.cuda()).cpu(), torch.linalg.svdvals(s)),
    "linalg.lstsq": lambda: (torch.linalg.lstsq(s.cuda(), torch.ones(32, 2, dtype=torch.float64).cuda()).solution.cpu(), torch.linalg.lstsq(s, torch.ones(32, 2, dtype=torch.float64)).solution),
    "linalg.qr batched (geqrfBatched)": lambda: (torch.linalg.qr(s.expand(4, -1, -1).cuda())[1].abs().cpu(), torch.linalg.qr(s.expand(4, -1, -1))[1].abs()),
}
for name, fn in gaps.items():
    try:
        got, want = fn(); torch.cuda.synchronize()
        err = ((got - want).abs().max() / want.abs().max().clamp_min(1.0)).item()
        print("GAP", name, "-> ran,", "matches CPU" if err < 1e-8 else f"rel err {err:.1e}")
    except Exception as e:
        print("GAP", name, "->", type(e).__name__, str(e).splitlines()[0][:120])
sys.exit(1 if fails else 0)
