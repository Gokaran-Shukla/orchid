r"""Exhaustive (grid) search, zooming grid refinement and Latin-hypercube
screening for parameter extraction and materials / device design.

Typical uses
------------
* fit (sigma, mu_inf, C, Sigma) of the GDM to mu(T, F) data,
* fit EGDM (sigma, N_t, mu0) to temperature-dependent J-V curves,
* screen a design space (mobilities, energy offsets, thickness) for maximum
  PCE or balanced OLED charge transport.

All evaluations are independent, so they are distributed with
``concurrent.futures.ProcessPoolExecutor`` when ``workers > 1``.
"""
import itertools
import numpy as np
from concurrent.futures import ProcessPoolExecutor


def _eval(args):
    f, p = args
    try:
        return float(f(p))
    except Exception:
        return np.inf


def grid_search(objective, grid, workers=1, chunksize=8):
    """Evaluate ``objective(params_dict)`` on the full Cartesian ``grid``.

    grid : dict name -> 1D array of candidate values.
    Returns (best_params, best_value, table) where table is a structured
    record array of every evaluated point (exhaustive landscape).
    """
    names = list(grid)
    combos = list(itertools.product(*[np.asarray(grid[k]) for k in names]))
    pts = [dict(zip(names, c)) for c in combos]
    if workers > 1:
        with ProcessPoolExecutor(workers) as ex:
            vals = list(ex.map(_eval, [(objective, p) for p in pts], chunksize=chunksize))
    else:
        vals = [_eval((objective, p)) for p in pts]
    vals = np.array(vals)
    k = int(np.nanargmin(vals))
    dtype = [(n, float) for n in names] + [("objective", float)]
    table = np.array([tuple(c) + (v,) for c, v in zip(combos, vals)], dtype=dtype)
    return pts[k], vals[k], table


def zoom_search(objective, bounds, n=9, levels=4, shrink=0.35, log=(),
                workers=1):
    """Iterated exhaustive grid with successive zoom around the optimum.

    bounds : dict name -> (lo, hi); names listed in ``log`` are gridded in
    log10 space. Returns (best, value, history).
    """
    cur = {k: (np.log10(v[0]), np.log10(v[1])) if k in log else v
           for k, v in bounds.items()}
    hist = []
    best, bv = None, np.inf
    for lev in range(levels):
        grid = {k: np.linspace(lo, hi, n) for k, (lo, hi) in cur.items()}
        b, v, tab = grid_search(_Wrap(objective, tuple(log)), grid, workers)
        hist.append(tab)
        if v < bv:
            best, bv = b, v
        new = {}
        for k, (lo, hi) in cur.items():
            w = (hi - lo) * shrink / 2
            c = best[k]
            new[k] = (c - w, c + w)
        cur = new
    out = {k: (10 ** v if k in log else v) for k, v in best.items()}
    return out, bv, hist


class _Wrap:
    """Picklable log-space wrapper."""
    def __init__(self, f, log):
        self.f, self.log = f, log

    def __call__(self, p):
        return self.f({k: (10 ** v if k in self.log else v) for k, v in p.items()})


def latin_hypercube(bounds, n, seed=0, log=()):
    """Latin-hypercube sample of ``n`` points inside ``bounds``."""
    rng = np.random.default_rng(seed)
    names = list(bounds)
    d = len(names)
    u = (rng.permuted(np.tile(np.arange(n), (d, 1)), axis=1).T + rng.random((n, d))) / n
    pts = []
    for row in u:
        p = {}
        for k, ui in zip(names, row):
            lo, hi = bounds[k]
            if k in log:
                p[k] = 10 ** (np.log10(lo) + ui * (np.log10(hi) - np.log10(lo)))
            else:
                p[k] = lo + ui * (hi - lo)
        pts.append(p)
    return pts


def log_rmse(y_model, y_data):
    """Root-mean-square error in log10 space (standard for mobilities/J)."""
    y_model = np.asarray(y_model, float)
    y_data = np.asarray(y_data, float)
    m = (y_model > 0) & (y_data > 0)
    if not m.any():
        return np.inf
    return float(np.sqrt(np.mean((np.log10(y_model[m]) - np.log10(y_data[m])) ** 2)))
