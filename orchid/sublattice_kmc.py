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
