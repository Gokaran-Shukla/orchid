r"""Master-equation (ME) solvers for hopping transport with Fermi-Dirac
statistics (Yu et al. PRB 63, 085202; Pasveer et al. PRL 94, 206601).

Steady state of the mean-field master equation

.. math::
   \sum_j \big[W_{ij}\,p_i(1-p_j) - W_{ji}\,p_j(1-p_i)\big] = 0

Two solvers:

* :func:`me_linear_response` - exact linear response (Ohmic limit) as a
  Miller-Abrahams random resistor network: conductances
  :math:`G_{ij}=W^0_{ij}f_i(1-f_j)/k_BT` are symmetric by detailed balance,
  and Kirchhoff's law :math:`\sum_j G_{ij}(\eta_i-\eta_j)=-F\sum_j G_{ij}\Delta x_{ij}`
  is a sparse SPD (Laplacian) system solved with preconditioned CG.
* :func:`me_nonlinear` - full nonlinear field dependence via the
  self-consistent iteration
  :math:`p_i=\sum_j W_{ji}p_j/\sum_j[W_{ij}(1-p_j)+W_{ji}p_j]`
  with carrier-number conservation.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .constants import kB_eV
from .models import fermi_level_gaussian


def reverse_index(lat):
    """rev[i, m] such that nbr[nbr[i, m], rev[i, m]] == i."""
    off = lat.offsets
    lookup = {tuple(o): k for k, o in enumerate(off)}
    rmap = np.array([lookup[tuple(-o)] for o in off])
    return np.broadcast_to(rmap, lat.nbr.shape)


def fermi_level_sample(E, n_frac, T):
    """Fermi level such that sum f(E_i) = n_frac * N for the actual sample."""
    from scipy.optimize import brentq
    kT = kB_eV * T
    def g(EF):
        return np.mean(1.0 / (1.0 + np.exp(np.clip((E - EF) / kT, -700, 700)))) - n_frac
    return brentq(g, E.min() - 60 * kT, E.max() + 60 * kT)


def me_linear_response(lat, T, n_frac, rtol=1e-10, maxiter=20000,
                       precond="auto", backend="cpu", **rate_kw):
    """Ohmic mobility [m^2/Vs] at occupied fraction ``n_frac``.

    precond : "jacobi" | "amg" (pyamg smoothed aggregation) | "auto"
        AMG cuts CG iterations by 10-20x and is chosen automatically for
        N > 2e5 sites when pyamg is installed.
    backend : "cpu" (scipy) | "gpu" (CuPy CG, Jacobi preconditioner)
    Returns (mu, info) where info has Fermi level and solver iterations.
    """
    kT = kB_eV * T
    W0 = lat.rates(T, (0.0, 0.0, 0.0), **rate_kw)
    EF = fermi_level_sample(lat.energy, n_frac, T)
    f = 1.0 / (1.0 + np.exp((lat.energy - EF) / kT))
    G = W0 * f[:, None] * (1.0 - f[lat.nbr])          # [N, M], symmetric pairs
    N, M = G.shape
    dx = lat.disp[:, :, 0]
    rows = np.repeat(np.arange(N), M)
    A = sp.csr_matrix((-G.ravel(), (rows, lat.nbr.ravel())), shape=(N, N))
    A = A + sp.diags(G.sum(1))
    A = 0.5 * (A + A.T)                                 # remove round-off asymmetry
    F = 1.0                                             # unit reduced field (eV per a)
    b = -F * (G * dx).sum(1)
    # pin node 0 to remove the constant null space
    A = A.tolil()
    A[0, :] = 0
    A[:, 0] = 0
    A[0, 0] = 1.0
    A = A.tocsr()
    b[0] = 0.0
    # rescale for conditioning (solution of A eta = b is scale invariant)
    sc = 1.0 / A.diagonal().max()
    A = A * sc
    b = b * sc
    it = [0]

    def cb(_):
        it[0] += 1
    if backend == "gpu":
        from .gpu import cg_solve_gpu
        eta, info = cg_solve_gpu(A, b, rtol=rtol, maxiter=maxiter)
    else:
        use_amg = precond == "amg"
        if precond == "auto":
            try:
                import pyamg  # noqa: F401
                use_amg = N > 200000
            except Exception:
                use_amg = False
        if use_amg:
            import pyamg
            Pc = pyamg.smoothed_aggregation_solver(A, symmetry="symmetric").aspreconditioner()
        else:
            dinv = 1.0 / A.diagonal()
            Pc = spla.LinearOperator((N, N), matvec=lambda v: dinv * v)
        eta, info = spla.cg(A, b, rtol=rtol, maxiter=maxiter, M=Pc, callback=cb)
    I = (G / kT) * (eta[:, None] - eta[lat.nbr] + F * dx)  # particle current i->j
    v = 0.5 * (I * dx).sum() / f.sum()                       # a/s per carrier, per (eV/a)
    mu = v * lat.a ** 2                                      # m^2/(V s)
    return mu, dict(EF=EF, cg_info=info, iterations=it[0])


def me_nonlinear(lat, T, n_frac, F_red, tol=1e-9, maxiter=200000, mix=1.0,
                 p0=None, **rate_kw):
    """Nonlinear ME at finite field (eV per a along x). Returns (mu, p)."""
    kT = kB_eV * T
    W = lat.rates(T, (F_red, 0.0, 0.0), **rate_kw)
    rev = reverse_index(lat)
    Wrev = W[lat.nbr, rev]                 # W_{j->i} for each (i, m)
    if p0 is None:
        Eloc = lat.energy - F_red * lat.ijk[:, 0] * 0.0
        EF = fermi_level_sample(Eloc, n_frac, T)
        p = 1.0 / (1.0 + np.exp((Eloc - EF) / kT))
    else:
        p = p0.copy()
    Ntot = n_frac * lat.N
    for k in range(maxiter):
        pj = p[lat.nbr]
        num = (Wrev * pj).sum(1)
        den = (W * (1 - pj) + Wrev * pj).sum(1)
        pn = num / den
        pn *= Ntot / pn.sum()
        pn = np.clip(pn, 0.0, 1.0)
        err = np.max(np.abs(pn - p)) / max(p.max(), 1e-300)
        p = (1 - mix) * p + mix * pn
        if err < tol:
            break
    pj = p[lat.nbr]
    I = W * p[:, None] * (1 - pj) - Wrev * pj * (1 - p[:, None])
    dx = lat.disp[:, :, 0]
    v = 0.5 * (I * dx).sum() / p.sum()
    mu = v * lat.a ** 2 / F_red
    return mu, dict(p=p, iterations=k + 1, residual=err)


def finite_size_extrapolation(make_lattice, sizes, T, n_frac, seeds=(0, 1, 2),
                              backend="auto", workers=None, **rate_kw):
    """Mobility vs box size with disorder averaging, and an L -> infinity
    estimate from a linear fit of ln(mu) against 1/L.

    ``make_lattice(L, seed)`` must return a Lattice with energies set.
    Every (L, seed) pair is an independent task for :func:`orchid.parallel.parallel_map`
    (processes or MPI ranks). Returns dict(L, mu_mean, mu_se, mu_inf).
    """
    from .parallel import parallel_map
    tasks = [(L, s) for L in sizes for s in seeds]
    vals = parallel_map(_fs_task, [(make_lattice, L, s, T, n_frac, rate_kw) for L, s in tasks],
                        backend=backend, workers=workers)
    mu = np.array(vals).reshape(len(sizes), len(seeds))
    m = mu.mean(1)
    se = mu.std(1, ddof=1) / np.sqrt(len(seeds)) if len(seeds) > 1 else np.zeros(len(sizes))
    c = np.polyfit(1.0 / np.asarray(sizes, float), np.log(m), 1)
    return dict(L=np.asarray(sizes), mu_mean=m, mu_se=se, mu_inf=float(np.exp(c[1])))


def _fs_task(args):
    make_lattice, L, s, T, n_frac, kw = args
    return me_linear_response(make_lattice(L, s), T, n_frac, **kw)[0]
