"""torch sparse matmul on the GPU against the same call on the CPU (cuSPARSE shim acceptance).
Run under cfi-run: cfi-run.exe --install-dir <bin> -- python sparse_check.py
Checked ops must match; the "gaps" section only records what happens for operations whose
cuSPARSE entry points the shim does not implement (they must fail cleanly, not crash)."""
import sys, time, torch
t0 = time.time()
assert torch.cuda.is_available(), "no cuda"
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
fails, n_ok = [], 0
g = torch.Generator().manual_seed(0)

def check(name, fn_gpu, fn_cpu, tol):
    global n_ok
    try:
        got = fn_gpu()
        torch.cuda.synchronize()
        got = got.to_dense().cpu() if got.layout != torch.strided else got.cpu()
        want = fn_cpu()
        want = want.to_dense() if want.layout != torch.strided else want
        err = ((got - want).abs().max() / want.abs().max().clamp_min(1.0)).item() if want.numel() else 0.0
        if err <= tol: n_ok += 1
        else: fails.append(f"{name}: rel err {err:.2e}")
    except Exception as e:
        fails.append(f"{name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")

def rand_sparse(m, n, density, dt):
    d = torch.randn(m, n, dtype=dt, generator=g)
    mask = torch.rand(m, n, generator=g) < density
    mask[::5] = False  # empty rows
    return d * mask

for dt, tol in [(torch.float32, 1e-4), (torch.float64, 1e-10)]:
    for (m, k, n, dens) in [(7, 5, 3, 0.4), (64, 48, 16, 0.1), (300, 200, 33, 0.05), (128, 1024, 8, 0.6), (1000, 1000, 1, 0.01)]:
        a = rand_sparse(m, k, dens, dt)
        b = torch.randn(k, n, dtype=dt, generator=g)
        c = torch.randn(m, n, dtype=dt, generator=g)
        v = torch.randn(k, dtype=dt, generator=g)
        csr, coo = a.to_sparse_csr(), a.to_sparse()
        tag = f"{m}x{k}x{n} {dt}"
        check(f"csr@dense {tag}", lambda: csr.cuda() @ b.cuda(), lambda: a @ b, tol)
        check(f"addmm csr {tag}", lambda: torch.addmm(c.cuda(), csr.cuda(), b.cuda(), beta=0.5, alpha=-2.0), lambda: torch.addmm(c, a, b, beta=0.5, alpha=-2.0), tol)
        check(f"csr@dense.t-contig {tag}", lambda: csr.cuda() @ b.t().contiguous().t().cuda(), lambda: a @ b, tol)
        check(f"mv csr {tag}", lambda: torch.mv(csr.cuda(), v.cuda()), lambda: torch.mv(a, v), tol)
        check(f"addmv csr {tag}", lambda: torch.addmv(c[:, 0].contiguous().cuda(), csr.cuda(), v.cuda(), beta=2.0, alpha=0.5), lambda: torch.addmv(c[:, 0], a, v, beta=2.0, alpha=0.5), tol)
        check(f"csr.t()@dense {tag}", lambda: csr.cuda().t() @ c.cuda(), lambda: a.t() @ c, tol)
        check(f"dense@csr {tag}", lambda: c.t().contiguous().cuda() @ csr.cuda(), lambda: c.t() @ a, tol)
        check(f"sparse.mm coo {tag}", lambda: torch.sparse.mm(coo.cuda(), b.cuda()), lambda: a @ b, tol)
        check(f"sparse.addmm coo {tag}", lambda: torch.sparse.addmm(c.cuda(), coo.cuda(), b.cuda(), beta=0.25, alpha=3.0), lambda: torch.addmm(c, a, b, beta=0.25, alpha=3.0), tol)
        check(f"coo@dense {tag}", lambda: coo.cuda() @ b.cuda(), lambda: a @ b, tol)
        if m % 4 == 0 and k % 4 == 0:
            bsr = a.to_sparse_bsr((4, 4))
            check(f"bsr@dense {tag}", lambda: bsr.cuda() @ b.cuda(), lambda: a @ b, tol)
            check(f"mv bsr {tag}", lambda: torch.mv(bsr.cuda(), v.cuda()), lambda: torch.mv(a, v), tol)
        # SDDMM: sampled_addmm on the sparsity of a (m x k) with dense (m x n) @ (n x k)
        p = torch.randn(m, n, dtype=dt, generator=g)
        q = torch.randn(n, k, dtype=dt, generator=g)
        check(f"sampled_addmm {tag}", lambda: torch.sparse.sampled_addmm(csr.cuda(), p.cuda(), q.cuda(), beta=0.5, alpha=2.0),
              lambda: (0.5 * a + 2.0 * (p @ q) * (a != 0)), tol)
    # sparse x sparse (cusparseSpGEMM) and sparse + sparse (csrgeam2)
    for (m, k, n, dens) in [(30, 30, 30, 0.2), (200, 150, 120, 0.05), (64, 300, 80, 0.1), (1000, 1000, 1000, 0.003)]:
        a, b = rand_sparse(m, k, dens, dt), rand_sparse(k, n, dens, dt)
        c = rand_sparse(m, k, dens, dt)
        tag = f"{m}x{k}x{n} {dt}"
        check(f"csr @ csr {tag}", lambda: (a.to_sparse_csr().cuda() @ b.to_sparse_csr().cuda()).to_dense(), lambda: a @ b, tol)
        check(f"sparse.mm csr csr {tag}", lambda: torch.sparse.mm(a.to_sparse_csr().cuda(), b.to_sparse_csr().cuda()).to_dense(), lambda: a @ b, tol)
        check(f"csr + csr {tag}", lambda: (a.to_sparse_csr().cuda() + c.to_sparse_csr().cuda()).to_dense(), lambda: a + c, tol)
        check(f"csr add alpha {tag}", lambda: torch.add(a.to_sparse_csr().cuda(), c.to_sparse_csr().cuda(), alpha=-0.5).to_dense(), lambda: a - 0.5 * c, tol)
    # batched: COO bmm and batched CSR @ dense
    a3 = torch.stack([rand_sparse(20, 15, 0.3, dt) for _ in range(3)])
    b3 = torch.randn(3, 15, 6, dtype=dt, generator=g)
    check(f"bmm coo {dt}", lambda: torch.bmm(a3.to_sparse().cuda(), b3.cuda()), lambda: torch.bmm(a3, b3), tol)
    # batched CSR @ dense is rejected by torch's cu118 build itself (CUDASparseDescriptors.cpp
    # asserts before calling cuSPARSE); the shim's strided-batch path is covered by tests/gpu.

# a simple GCN-style layer: relu(A @ (X W)) with A sparse
dt = torch.float32
A = rand_sparse(500, 500, 0.02, dt) + torch.eye(500)
X = torch.randn(500, 64, generator=g); W = torch.randn(64, 32, generator=g)
check("gcn layer", lambda: torch.relu(A.to_sparse_csr().cuda() @ (X.cuda() @ W.cuda())), lambda: torch.relu(A @ (X @ W)), 1e-4)

print(f"{n_ok} checks passed, {len(fails)} failed, {time.time() - t0:.1f} s")
for f in fails: print("FAIL", f)

# gaps: operations that need cuSPARSE entry points the shim does not implement
a = rand_sparse(30, 30, 0.2, torch.float32) + torch.eye(30)
csr = a.to_sparse_csr().cuda()
gaps = {
    # torch's cu118 COO x COO path checks for compute capability >= 5.3 before calling cuSPARSE
    "coo@coo (torch cc check)": lambda: torch.sparse.mm(csr.to_sparse_coo(), csr.to_sparse_coo()).to_dense(),
    "triangular_solve (SpSM)": lambda: torch.triangular_solve(torch.randn(30, 2).cuda(), csr, upper=False)[0],
    "complex64 csr@dense": lambda: a.to(torch.complex64).to_sparse_csr().cuda() @ torch.randn(30, 2, dtype=torch.complex64).cuda(),
    "float16 csr@dense": lambda: a.half().to_sparse_csr().cuda() @ torch.randn(30, 2).half().cuda(),
}
for name, fn in gaps.items():
    try:
        r = fn(); torch.cuda.synchronize()
        print("GAP", name, "-> ran (not checked)")
    except Exception as e:
        print("GAP", name, "->", type(e).__name__, str(e).splitlines()[0][:120])
sys.exit(1 if fails else 0)
