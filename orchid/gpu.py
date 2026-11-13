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
except Exception:  # pragma: no cover - numba without CUDA support
    cuda = None
    _kmc_kernel = None


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
