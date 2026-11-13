r"""Machine-learning surrogates and Bayesian inversion (roadmap R22).

* :class:`GPSurrogate` - Gaussian-process regression (ARD squared-
  exponential kernel + noise), pure NumPy. Hyperparameters are selected by
  exhaustive search of the log marginal likelihood
  :math:`\log p(y|X)=-\tfrac12 y^TK^{-1}y-\tfrac12\log|K|-\tfrac n2\log2\pi`.
  Typical use: learn :math:`\log_{10}\mu(\hat\sigma,\log p,eFa/\sigma)` from
  ORCHID master-equation / KMC data, then evaluate it in microseconds
  inside drift-diffusion or a search loop.
* :func:`generate_mobility_dataset` - parallel ME data generation
  (:func:`orchid.parallel.parallel_map`).
* :func:`bayes_grid` - exhaustive-grid Bayesian inversion with a Gaussian
  log-likelihood. Returns the full posterior, marginals, MAP estimate and
  credible intervals: uncertainty quantification on top of the exhaustive
  search.
"""
import itertools
import numpy as np


class GPSurrogate:
    def __init__(self, normalize=True):
        self.normalize = normalize

    @staticmethod
    def _k(A, B, ell, sf):
        d = (A[:, None, :] - B[None, :, :]) / ell
        return sf ** 2 * np.exp(-0.5 * (d ** 2).sum(-1))

    def _nll(self, ell, sf, sn):
        K = self._k(self.X, self.X, ell, sf) + (sn ** 2 + 1e-10) * np.eye(len(self.X))
        try:
            Lc = np.linalg.cholesky(K)
        except np.linalg.LinAlgError:
            return np.inf
        a = np.linalg.solve(Lc.T, np.linalg.solve(Lc, self.y))
        return 0.5 * self.y @ a + np.log(np.diag(Lc)).sum() + 0.5 * len(self.y) * np.log(2 * np.pi)

    def fit(self, X, y, ell_grid=None, sf_grid=None, sn_grid=None):
        X = np.atleast_2d(np.asarray(X, float))
        y = np.asarray(y, float)
        self.xm, self.xs = (X.mean(0), X.std(0) + 1e-12) if self.normalize else (0.0, 1.0)
        self.ym, self.ys = (y.mean(), y.std() + 1e-12) if self.normalize else (0.0, 1.0)
        self.X = (X - self.xm) / self.xs
        self.y = (y - self.ym) / self.ys
        d = X.shape[1]
        ell_grid = np.logspace(-1, 1, 9) if ell_grid is None else ell_grid
        sf_grid = np.array([0.5, 1.0, 2.0]) if sf_grid is None else sf_grid
        sn_grid = np.array([1e-3, 1e-2, 1e-1]) if sn_grid is None else sn_grid
        best = (np.inf, None)
        for ell in itertools.product(*[ell_grid] * d):
            for sf in sf_grid:
                for sn in sn_grid:
                    v = self._nll(np.array(ell), sf, sn)
                    if v < best[0]:
                        best = (v, (np.array(ell), sf, sn))
        self.nll, (self.ell, self.sf, self.sn) = best
        K = self._k(self.X, self.X, self.ell, self.sf) + (self.sn ** 2 + 1e-10) * np.eye(len(self.X))
        self.Lc = np.linalg.cholesky(K)
        self.alpha = np.linalg.solve(self.Lc.T, np.linalg.solve(self.Lc, self.y))
        return self

    def predict(self, Xs, return_std=False):
        Xs = (np.atleast_2d(np.asarray(Xs, float)) - self.xm) / self.xs
        Ks = self._k(Xs, self.X, self.ell, self.sf)
        mu = Ks @ self.alpha * self.ys + self.ym
        if not return_std:
            return mu
        v = np.linalg.solve(self.Lc, Ks.T)
        var = self.sf ** 2 - (v ** 2).sum(0)
        return mu, np.sqrt(np.clip(var, 0, None)) * self.ys


def _me_point(args):
    shat, logp, L, seed, sigma, a = args
    from .lattice import Lattice
    from .master_equation import me_linear_response
    from .constants import kB_eV
    T = sigma / (kB_eV * shat)
    lat = Lattice(L, a=a, seed=seed).set_disorder(sigma)
    mu = me_linear_response(lat, T, 10 ** logp, alpha=10.0)[0]
    mu0 = a ** 2 * 1e12 / sigma
    return np.log10(mu / mu0)


def generate_mobility_dataset(shats, logps, L=24, seed=0, sigma=0.1, a=1.6e-9,
                              backend="auto", workers=None):
    """log10(mu/mu0) from the master equation on a (shat, log10 p) grid."""
    from .parallel import parallel_map
    pts = [(s, lp, L, seed, sigma, a) for s in shats for lp in logps]
    y = np.array(parallel_map(_me_point, pts, backend=backend, workers=workers))
    X = np.array([(s, lp) for s in shats for lp in logps])
    return X, y


def bayes_grid(model, data, grid, sigma_noise, prior=None, log_space=True):
    """Posterior on an exhaustive parameter grid.

    model(params_dict) -> predictions (same shape as data); Gaussian noise of
    std ``sigma_noise`` on log10(data) if ``log_space`` else on data.
    Returns dict(post (array over grid), axes, names, map, mean, ci95 per name).
    """
    names = list(grid)
    axes = [np.asarray(grid[k]) for k in names]
    shape = tuple(len(a) for a in axes)
    logL = np.empty(shape)
    d = np.log10(data) if log_space else np.asarray(data)
    for idx in itertools.product(*[range(s) for s in shape]):
        p = {k: axes[i][j] for i, (k, j) in enumerate(zip(names, idx))}
        m = model(p)
        m = np.log10(m) if log_space else m
        logL[idx] = -0.5 * np.sum((m - d) ** 2) / sigma_noise ** 2
    if prior is not None:
        logL = logL + np.log(prior)
    post = np.exp(logL - logL.max())
    post /= post.sum()
    out = dict(post=post, axes=axes, names=names)
    imax = np.unravel_index(np.argmax(post), shape)
    out["map"] = {k: axes[i][imax[i]] for i, k in enumerate(names)}
    out["mean"], out["ci95"], out["marginals"] = {}, {}, {}
    for i, k in enumerate(names):
        marg = post.sum(axis=tuple(j for j in range(len(names)) if j != i))
        cdf = np.cumsum(marg)
        out["marginals"][k] = marg
        out["mean"][k] = float((marg * axes[i]).sum())
        out["ci95"][k] = (float(np.interp(0.025, cdf, axes[i])), float(np.interp(0.975, cdf, axes[i])))
    return out
