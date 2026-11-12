r"""Exciton dynamics: kinetic Monte Carlo (Förster / Dexter hopping in a
disordered lattice) and continuum rate equations with bimolecular
annihilation for OLED emission zones and OPV exciton harvesting.

KMC
---
Each exciton hops with rates :math:`k_{ij}` (Förster or Dexter, Boltzmann
factor for uphill hops) and decays with :math:`k_d=1/\tau`. The diffusion
coefficient follows from :math:`\langle r^2\rangle = 6D\tau` (3D) and the
diffusion length is :math:`L_D=\sqrt{D\tau}`. Without disorder
:math:`D=\frac16\sum_j k_j r_j^2` exactly, which is used as a validation.

Continuum
---------
.. math::
   \partial_t S = D\,\partial_x^2 S + \eta_S R(x) - S/\tau - k_{SS} S^2
   - k_{SP}\,(n+p)\,S

solved in steady state on the device grid (Dirichlet quenching at metal
electrodes optional) -> emission profile and internal quantum efficiency.
"""
import numpy as np
from ._jit import njit, prange
from .constants import kB_eV


@njit(cache=True)
def _one_walk(nbr, disp, Wcum, Wtot, kd, s, seed):
    np.random.seed(seed)
    M = nbr.shape[1]
    x = 0.0
    y = 0.0
    z = 0.0
    t = 0.0
    nh = 0
    alive = True
    while alive:
        tot = Wtot[s] + kd
        t += -np.log(1.0 - np.random.random()) / tot
        u = np.random.random() * tot
        if u >= Wtot[s] or nh > 100000000:
            alive = False
        else:
            lo = 0
            hi = M - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if Wcum[s, mid] < u:
                    lo = mid + 1
                else:
                    hi = mid
            x += disp[s, lo, 0]
            y += disp[s, lo, 1]
            z += disp[s, lo, 2]
            s = nbr[s, lo]
            nh += 1
    return x * x + y * y + z * z, t, nh


@njit(cache=True, parallel=True)
def _exciton_walk(nbr, disp, Wcum, Wtot, kd, starts, seed):
    nc = starts.shape[0]
    R2 = np.zeros(nc)
    TT = np.zeros(nc)
    NH = np.zeros(nc, dtype=np.int64)
    for c in prange(nc):
        r2, t, nh = _one_walk(nbr, disp, Wcum, Wtot, kd, starts[c], seed + 104729 * c)
        R2[c] = r2
        TT[c] = t
        NH[c] = nh
    return R2, TT, NH


def exciton_kmc(lat, T, tau=1e-9, R0=2.0, mechanism="forster", kD=1e12,
                Ld=0.3, n_excitons=20000, seed=7):
    """Exciton KMC on ``lat`` (lengths in lattice units, energies eV).

    Returns dict with D [m^2/s], L_D [m], mean hops, and the analytic
    disorder-free D for comparison.
    """
    dE = lat.bond_dE()
    r = lat.dist
    up = np.exp(-np.clip(dE / (kB_eV * T), 0, None))
    if mechanism == "forster":
        W = (1 / tau) * (R0 / r) ** 6 * up
        W0 = (1 / tau) * (R0 / r[0]) ** 6
    else:
        W = kD * np.exp(-2 * r / Ld) * up
        W0 = kD * np.exp(-2 * r[0] / Ld)
    Wcum = np.cumsum(W, 1)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, lat.N, n_excitons).astype(np.int64)
    R2, TT, NH = _exciton_walk(lat.nbr, lat.disp, Wcum, Wcum[:, -1].copy(),
                               1 / tau, starts, seed)
    a = lat.a
    D = R2.mean() * a ** 2 / (6 * tau)
    D_err = R2.std() * a ** 2 / (6 * tau) / np.sqrt(len(R2))
    D_analytic = (W0 * (r[0] * a) ** 2).sum() / 6.0
    return dict(D=D, D_err=D_err, L_D=np.sqrt(D * tau), hops=NH.mean(),
                D_ordered=D_analytic, R2=R2 * a ** 2, t=TT)


def steady_state_excitons(x, R, D, tau, eta_S=0.25, kSS=0.0, kSP=0.0,
                          npol=None, quench_left=True, quench_right=True,
                          tol=1e-10, maxit=200):
    """Steady-state 1D exciton density with annihilation (Newton).

    x : grid [m]; R : exciton formation rate [m^-3 s^-1] (e.g. Langevin
    recombination profile from the drift-diffusion solver).
    Returns S(x) and the radiative yield fraction (decay/(formation)).
    """
    from scipy.sparse import diags
    from scipy.sparse.linalg import spsolve
    N = len(x)
    h = np.diff(x)
    dV = np.empty(N)
    dV[1:-1] = 0.5 * (h[1:] + h[:-1])
    dV[0], dV[-1] = 0.5 * h[0], 0.5 * h[-1]
    pol = np.zeros(N) if npol is None else npol
    src = eta_S * R
    S = src * tau
    lo = np.zeros(N)
    up_ = np.zeros(N)
    lo[1:] = D / h
    up_[:-1] = D / h
    diag0 = -(np.concatenate([[0], D / h]) + np.concatenate([D / h, [0]]))
    for it in range(maxit):
        F = (lo * np.roll(S, 1) + up_ * np.roll(S, -1) + diag0 * S) \
            + dV * (src - S / tau - kSS * S ** 2 - kSP * pol * S)
        Jd = diags([lo[1:], diag0 - dV * (1 / tau + 2 * kSS * S + kSP * pol), up_[:-1]],
                   [-1, 0, 1], format="lil")
        if quench_left:
            Jd[0, :] = 0
            Jd[0, 0] = 1
            F[0] = S[0]
        if quench_right:
            Jd[-1, :] = 0
            Jd[-1, -1] = 1
            F[-1] = S[-1]
        dS = spsolve(Jd.tocsr(), -F)
        S = np.maximum(S + dS, 0)
        if np.max(np.abs(dS)) < tol * max(S.max(), 1e-300):
            break
    gen = np.sum(src * dV)
    rad = np.sum(S / tau * dV)
    return S, (rad / gen if gen > 0 else np.nan)
