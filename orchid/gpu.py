r"""GPU back-ends.

* :func:`kmc_independent_gpu` - CUDA kernel (Numba) running one
  independent carrier per GPU thread with a per-thread xoroshiro128+ RNG.
  10^5-10^6 walkers run concurrently; rates are read from the same
  cumulative tables as the CPU engine.
* :func:`cg_solve_gpu` - CuPy conjugate-gradient solve of the master-equation
  Laplacian (``cupyx.scipy.sparse``), used by
  :func:`orchid.master_equation.me_linear_response` with ``backend="gpu"``.

Both are optional. :func:`available` reports what is usable. The CUDA kernel
can be tested without a GPU through Numba's simulator
(``NUMBA_ENABLE_CUDASIM=1``), which the test suite does.
"""
import os
import numpy as np


def available():
    """dict(cuda=bool, cupy=bool, simulator=bool)."""
    out = dict(cuda=False, cupy=False, simulator=os.environ.get("NUMBA_ENABLE_CUDASIM") == "1")
    try:
        from numba import cuda
        out["cuda"] = bool(cuda.is_available())
    except Exception:
        pass
    try:
        import cupy  # noqa: F401
        out["cupy"] = bool(cupy.cuda.runtime.getDeviceCount() > 0)
    except Exception:
        pass
    return out


import math

try:
    from numba import cuda
    from numba.cuda.random import xoroshiro128p_uniform_float64

    @cuda.jit
    def _kmc_kernel(nbr, dx, Wcum, Wtot, starts, n_eq, n_meas, rng, X, TT):
        c = cuda.grid(1)
        if c >= starts.shape[0]:
            return
        M = nbr.shape[1]
        s = starts[c]
        x = 0.0
        t = 0.0
        x0 = 0.0
        t0 = 0.0
        for hop in range(n_eq + n_meas):
            if hop == n_eq:
                x0 = x
                t0 = t
            u1 = xoroshiro128p_uniform_float64(rng, c)
            t += -math.log(1.0 - u1) / Wtot[s]
            u = xoroshiro128p_uniform_float64(rng, c) * Wtot[s]
            lo = 0
            hi = M - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if Wcum[s, mid] < u:
                    lo = mid + 1
                else:
                    hi = mid
            x += dx[s, lo]
            s = nbr[s, lo]
        X[c] = x - x0
        TT[c] = t - t0
    @cuda.jit
    def _sublattice_kernel(blocks, B, Lx, Ly, Lz, nbr, offs, Wb, occ, who, pos, xdisp,
                           tau, rng, nev_out):
        k = cuda.grid(1)
        if k >= blocks.shape[0]:
            return
        bx = blocks[k, 0]
        by = blocks[k, 1]
        bz = blocks[k, 2]
        M = nbr.shape[1]
        t = 0.0
        nev = 0
        while True:
            # pass 1: total rate of carriers inside the block
            R = 0.0
            for ix in range(bx * B, bx * B + B):
                for iy in range(by * B, by * B + B):
                    for iz in range(bz * B, bz * B + B):
                        s = (ix * Ly + iy) * Lz + iz
                        if who[s] >= 0:
                            for m in range(M):
                                if occ[nbr[s, m]] == 0:
                                    R += Wb[s, m]
            if R <= 0.0:
                break
            dt = -math.log(1.0 - xoroshiro128p_uniform_float64(rng, k)) / R
            if t + dt > tau:
                break
            t += dt
            u = xoroshiro128p_uniform_float64(rng, k) * R
            # pass 2: select the event
            acc = 0.0
            done = False
            for ix in range(bx * B, bx * B + B):
                if done:
                    break
                for iy in range(by * B, by * B + B):
                    if done:
                        break
                    for iz in range(bz * B, bz * B + B):
                        if done:
                            break
                        s = (ix * Ly + iy) * Lz + iz
                        c = who[s]
                        if c >= 0:
                            for m in range(M):
                                j = nbr[s, m]
                                if occ[j] == 0:
                                    acc += Wb[s, m]
                                    if acc >= u and not done:
                                        occ[s] = 0
                                        who[s] = -1
                                        occ[j] = 1
                                        who[j] = c
                                        pos[c] = j
                                        xdisp[c] += offs[m, 0]
                                        done = True
                                        break
            nev += 1
        nev_out[k] = nev
except Exception:  # pragma: no cover - numba without CUDA support
    cuda = None
    _kmc_kernel = None
    _sublattice_kernel = None


def kmc_independent_gpu(lat, W, n_carriers=65536, n_eq=100000, n_meas=200000,
                        seed=1, threads_per_block=128):
    """GPU version of :func:`orchid.kmc.kmc_independent` (dx, dt per walker)."""
    from numba.cuda.random import create_xoroshiro128p_states
    if _kmc_kernel is None:
        raise RuntimeError("numba.cuda not available")
    k = _kmc_kernel
    Wcum = np.cumsum(W, axis=1)
    Wtot = Wcum[:, -1].copy()
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, lat.N, n_carriers).astype(np.int64)
    d_nbr = cuda.to_device(lat.nbr)
    d_dx = cuda.to_device(np.ascontiguousarray(lat.disp[:, :, 0]))
    d_Wc = cuda.to_device(Wcum)
    d_Wt = cuda.to_device(Wtot)
    d_st = cuda.to_device(starts)
    d_X = cuda.device_array(n_carriers)
    d_T = cuda.device_array(n_carriers)
    states = create_xoroshiro128p_states(n_carriers, seed=seed)
    blocks = (n_carriers + threads_per_block - 1) // threads_per_block
    k[blocks, threads_per_block](d_nbr, d_dx, d_Wc, d_Wt, d_st, int(n_eq), int(n_meas),
                                 states, d_X, d_T)
    return dict(dx=d_X.copy_to_host(), dt=d_T.copy_to_host())


def mobility_independent_gpu(lat, T, F_red, n_carriers=65536, n_eq=100000,
                             n_meas=200000, seed=1, **rate_kw):
    W = lat.rates(T, (F_red, 0.0, 0.0), **rate_kw)
    out = kmc_independent_gpu(lat, W, n_carriers, n_eq, n_meas, seed)
    F = F_red / lat.a
    mu = out["dx"].sum() * lat.a / (F * out["dt"].sum())
    mu_each = out["dx"] * lat.a / (F * out["dt"])
    return mu, mu_each.std(ddof=1) / np.sqrt(n_carriers)


def cg_solve_gpu(A_csr, b, rtol=1e-10, maxiter=20000):
    """Solve SPD system on the GPU with Jacobi-preconditioned CG (CuPy)."""
    import cupy as cp
    import cupyx.scipy.sparse as csp
    import cupyx.scipy.sparse.linalg as cspla
    Ad = csp.csr_matrix(A_csr)
    bd = cp.asarray(b)
    dinv = 1.0 / Ad.diagonal()
    M = cspla.LinearOperator(Ad.shape, matvec=lambda v: dinv * v)
    x, info = cspla.cg(Ad, bd, tol=rtol, maxiter=maxiter, M=M)
    return cp.asnumpy(x), int(info)


def mobility_sublattice_gpu(lat, T, F_red, density, B=4, tau=None, n_cycles=2000, n_eq=500,
                            seed=1, threads_per_block=64, **rate_kw):
    """GPU synchronous-sublattice interacting KMC (one CUDA thread per block).

    Same algorithm and arguments as :func:`orchid.sublattice_kmc.mobility_sublattice`.
    Carriers that start a window outside their block do not move, exactly as
    on the CPU.
    """
    from numba.cuda.random import create_xoroshiro128p_states
    if _sublattice_kernel is None:
        raise RuntimeError("numba.cuda not available")
    Lx, Ly, Lz = lat.shape
    W = lat.rates(T, (F_red, 0.0, 0.0), **rate_kw)
    rng = np.random.default_rng(seed)
    nc = max(1, int(round(density * lat.N)))
    pos = rng.choice(lat.N, nc, replace=False).astype(np.int64)
    occ = np.zeros(lat.N, dtype=np.int8)
    occ[pos] = 1
    who = -np.ones(lat.N, dtype=np.int64)
    who[pos] = np.arange(nc)
    if tau is None:
        tau = 1.0 / W.sum(1).max()
    nbx, nby, nbz = Lx // B, Ly // B, Lz // B
    subl = []
    for sx in (0, 1):
        for sy in (0, 1):
            for sz in (0, 1):
                subl.append(cuda.to_device(np.array(
                    [(i, j, k) for i in range(sx, nbx, 2) for j in range(sy, nby, 2)
                     for k in range(sz, nbz, 2)], dtype=np.int64)))
    nmax = max(b.shape[0] for b in subl)
    d_nbr, d_offs, d_W = cuda.to_device(lat.nbr), cuda.to_device(lat.offsets.astype(np.float64)), cuda.to_device(W)
    d_occ, d_who, d_pos = cuda.to_device(occ), cuda.to_device(who), cuda.to_device(pos)
    d_x = cuda.to_device(np.zeros(nc))
    states = create_xoroshiro128p_states(nmax, seed=seed)
    d_nev = cuda.device_array(nmax, dtype=np.int64)
    x0 = 0.0
    t = 0.0
    for cyc in range(n_eq + n_cycles):
        if cyc == n_eq:
            x0 = d_x.copy_to_host().sum()
            t = 0.0
        for s_ in rng.permutation(8):
            bl = subl[s_]
            nb = bl.shape[0]
            g = (nb + threads_per_block - 1) // threads_per_block
            _sublattice_kernel[g, threads_per_block](bl, B, Lx, Ly, Lz, d_nbr, d_offs, d_W, d_occ,
                                                     d_who, d_pos, d_x, tau, states, d_nev)
        t += tau
    dx = d_x.copy_to_host().sum() - x0
    F = F_red / lat.a
    return dx * lat.a / (F * t * nc), dict(tau=tau)
