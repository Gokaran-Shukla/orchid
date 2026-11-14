r"""Parallel-in-space interacting KMC: synchronous sublattice algorithm
(Shim & Amar, PRB 71, 125432 (2005)) - roadmap R23.

The periodic lattice is split into cubic blocks of side B >= 2 r_c, arranged
as 8 sublattices (2x2x2 checkerboard of blocks). In each cycle the 8
sublattices are visited in random order. All blocks of the active sublattice
run independent rejection-free KMC for a time window tau, *concurrently*
(Numba threads on CPU, one CUDA thread per block on GPU). Only carriers
inside an active block move. They may hop into the neighbouring inactive
block, which no other active block can reach, so there are no conflicts.
Events that would occur after tau are rejected. The algorithm becomes
exact as tau -> 0; tau ~ 1/(max rate per block) is accurate to a few %.

Interactions: site exclusion (Fermi-Dirac statistics). Field along +x.
"""
import math
import numpy as np
from ._jit import njit, prange


@njit(cache=True)
def _block_kmc(bx, by, bz, B, shape, nbr, offs, Wb, occ, who, pos, xdisp,
               tau, seed):
    np.random.seed(seed)
    Lx, Ly, Lz = shape[0], shape[1], shape[2]
    M = nbr.shape[1]
    # carriers currently inside this block
    cl = np.empty(4 * B * B * B, dtype=np.int64)
    nc = 0
    for ix in range(bx * B, bx * B + B):
        for iy in range(by * B, by * B + B):
            for iz in range(bz * B, bz * B + B):
                s = (ix * Ly + iy) * Lz + iz
                c = who[s]
                if c >= 0:
                    cl[nc] = c
                    nc += 1
    t = 0.0
    nev = 0
    rates = np.zeros((max(nc, 1), M))
    while True:
        R = 0.0
        for k in range(nc):
            c = cl[k]
            s = pos[c]
            ix = s // (Ly * Lz)
            iy = (s // Lz) % Ly
            iz = s % Lz
            inside = (ix >= bx * B) and (ix < bx * B + B) and (iy >= by * B) and \
                     (iy < by * B + B) and (iz >= bz * B) and (iz < bz * B + B)
            for m in range(M):
                w = 0.0
                if inside:
                    j = nbr[s, m]
                    if not occ[j]:
                        w = Wb[s, m]
                rates[k, m] = w
                R += w
        if R <= 0.0:
            break
        dt = -math.log(1.0 - np.random.random()) / R
        if t + dt > tau:
            break
        t += dt
        u = np.random.random() * R
        acc = 0.0
        kk = 0
        mm = 0
        found = False
        for k in range(nc):
            for m in range(M):
                acc += rates[k, m]
                if acc >= u:
                    kk = k
                    mm = m
                    found = True
                    break
            if found:
                break
        c = cl[kk]
        s = pos[c]
        j = nbr[s, mm]
        occ[s] = False
        who[s] = -1
        occ[j] = True
        who[j] = c
        pos[c] = j
        xdisp[c] += offs[mm, 0]
        nev += 1
    return nev


@njit(cache=True, parallel=True)
def _cycle(blocks, B, shape, nbr, offs, Wb, occ, who, pos, xdisp, tau, seed):
    nb = blocks.shape[0]
    nev = np.zeros(nb, dtype=np.int64)
    for k in prange(nb):
        nev[k] = _block_kmc(blocks[k, 0], blocks[k, 1], blocks[k, 2], B, shape, nbr, offs,
                            Wb, occ, who, pos, xdisp, tau, seed + 7919 * k)
    return nev.sum()


def mobility_sublattice(lat, T, F_red, density, B=4, tau=None, n_cycles=2000, n_eq=500,
                        seed=1, **rate_kw):
    """Finite-density mobility with the parallel sublattice algorithm.

    Lattice dimensions must be multiples of 2B. Returns (mu [m^2/Vs], info).
    """
    Lx, Ly, Lz = lat.shape
    if any(L % (2 * B) for L in lat.shape):
        raise ValueError("lattice sides must be multiples of 2B")
    if B < 2 * int(np.ceil(lat.rc)):
        raise ValueError("block side must be >= 2 r_c")
    W = lat.rates(T, (F_red, 0.0, 0.0), **rate_kw)
    rng = np.random.default_rng(seed)
    nc = max(1, int(round(density * lat.N)))
    pos = rng.choice(lat.N, nc, replace=False).astype(np.int64)
    occ = np.zeros(lat.N, dtype=np.bool_)
    occ[pos] = True
    who = -np.ones(lat.N, dtype=np.int64)
    who[pos] = np.arange(nc)
    xdisp = np.zeros(nc)
    if tau is None:
        tau = 1.0 / (W.sum(1).max())          # ~ one event per fastest site per window
    nbx, nby, nbz = Lx // B, Ly // B, Lz // B
    subl = []
    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                bl = np.array([(i, j, k) for i in range(sx, nbx, 2) for j in range(sy, nby, 2)
                               for k in range(sz, nbz, 2)], dtype=np.int64)
                subl.append(bl)
    shape = np.array(lat.shape, dtype=np.int64)
    offs = lat.offsets.astype(np.float64)
    t = 0.0
    x0 = 0.0
    nev = 0
    for cyc in range(n_eq + n_cycles):
        if cyc == n_eq:
            x0 = xdisp.sum()
            t = 0.0
        for s_ in rng.permutation(8):
            nev += _cycle(subl[s_], B, shape, lat.nbr, offs, W, occ, who, pos, xdisp, tau,
                          int(rng.integers(1 << 30)))
        t += tau                                   # each sublattice advanced by tau / 8 ... see doc
    # every carrier experiences one tau window per cycle (its own sublattice)
    dx = xdisp.sum() - x0
    F = F_red / lat.a
    mu = dx * lat.a / (F * t * nc)
    return mu, dict(tau=tau, events=nev, carriers=nc)


try:
    from numba import cuda as _cuda   # noqa: F401
    HAVE_CUDA_API = True
except Exception:  # pragma: no cover
    HAVE_CUDA_API = False


def mobility_sublattice_extrapolated(lat, T, F_red, density, factors=(4.0, 2.0, 1.0),
                                     tau0=None, order=None, sim_time_factor=40000.0, B=4,
                                     seeds=(1, 2), **rate_kw):
    """Remove the time-window bias of the synchronous-sublattice algorithm
    (roadmap R25, v0.4).

    The sublattice algorithm is exact only in the limit tau -> 0. In
    practice it is run at a finite window and so is biased (v0.3: about
    6-10 % at tau = 10 tau0). This function runs it at windows
    ``tau = f * tau0`` for every f in ``factors``, holding the simulated
    time fixed (``n_cycles`` proportional to 1/f). It then fits
    ``mu(tau) = mu_0 + c_1 tau (+ c_2 tau^2)`` by weighted least squares
    and extrapolates to tau -> 0. ``order`` defaults to 2 with three or
    more windows, else 1 (Richardson).

    tau0 defaults to 1 / max_i sum_j W_ij. Returns
    (mu_0, info), where info holds the per-window means and standard
    errors, the fitted coefficients and an extrapolation uncertainty.
    The uncertainty is the larger of the propagated statistical error and
    the |order-1 minus order-2| difference, when three or more windows
    are given.
    """
    W = lat.rates(T, (F_red, 0.0, 0.0), **rate_kw)
    if tau0 is None:
        tau0 = 1.0 / W.sum(1).max()
    taus, mus, errs = [], [], []
    for f in factors:
        n_cyc = max(200, int(sim_time_factor / f))
        vals = [mobility_sublattice(lat, T, F_red, density, B=B, tau=f * tau0,
                                    n_cycles=n_cyc, n_eq=max(50, n_cyc // 10), seed=s,
                                    **rate_kw)[0] for s in seeds]
        taus.append(f * tau0)
        mus.append(np.mean(vals))
        errs.append(np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
    taus, mus, errs = map(np.asarray, (taus, mus, errs))
    x = taus / tau0
    w = 1.0 / np.maximum(errs, 1e-3 * np.abs(mus))

    def fit(o):
        A = np.vander(x, o + 1, increasing=True)
        coef, *_ = np.linalg.lstsq(A * w[:, None], mus * w, rcond=None)
        cov = np.linalg.pinv((A * w[:, None]).T @ (A * w[:, None]))
        return coef, np.sqrt(max(cov[0, 0], 0.0))

    if order is None:
        order = 2 if len(x) >= 3 else 1
    o = min(order, len(x) - 1)
    coef, stat = fit(o)
    sys_err = 0.0
    if len(x) >= 3:
        c_alt, _ = fit(2 if o == 1 else 1)
        sys_err = abs(c_alt[0] - coef[0])
    return coef[0], dict(tau0=tau0, taus=taus, mu=mus, mu_err=errs, coef=coef, order=o,
                         err=max(stat, sys_err), stat_err=stat, sys_err=sys_err)
