r"""Charge / energy transfer rate expressions for hopping transport.

All functions are vectorised over ``dE`` (energy of final minus initial site,
eV, *including* the electrostatic field term) and distance ``r``.

Implemented rates
-----------------
* Miller-Abrahams (MA):
  :math:`W = \nu_0 e^{-2\alpha r}\,\min\{1, e^{-\Delta E/k_BT}\}`
* Symmetric MA (detailed balance with factor 1/2 split):
  :math:`W = \nu_0 e^{-2\alpha r} e^{-(\Delta E+|\Delta E|)/2k_BT}`
* Marcus (semiclassical, high-T limit of polaron hopping):
  :math:`W = \frac{2\pi}{\hbar}|J|^2\frac{1}{\sqrt{4\pi\lambda k_BT}}
  \exp[-(\Delta E+\lambda)^2/4\lambda k_BT]`
* Marcus-Levich-Jortner (MLJ): one effective quantum mode
  :math:`\hbar\omega`, Huang-Rhys factor :math:`S`.
* Förster (dipole-dipole, excitons): :math:`k = \tau^{-1}(R_0/r)^6 b(\Delta E)`
* Dexter (exchange, triplets): :math:`k = k_D e^{-2r/L}b(\Delta E)`

where :math:`b(\Delta E)` is the MA-type Boltzmann factor for uphill hops.
"""
import numpy as np
from .constants import kB_eV, hbar_eVs

RATE_MA = 0
RATE_MARCUS = 1
RATE_SYM_MA = 2
RATE_MLJ = 3


def boltzmann_uphill(dE, T):
    """min(1, exp(-dE/kT)) computed without overflow."""
    x = np.asarray(dE, dtype=float) / (kB_eV * T)
    return np.exp(-np.clip(x, 0.0, None))


def miller_abrahams(dE, r, T, nu0=1e12, alpha=1.0):
    """Miller-Abrahams rate [1/s]. ``r`` and ``1/alpha`` share a length unit."""
    return nu0 * np.exp(-2.0 * alpha * np.asarray(r)) * boltzmann_uphill(dE, T)


def symmetric_ma(dE, r, T, nu0=1e12, alpha=1.0):
    """Symmetric variant exp(-dE/2kT); obeys detailed balance."""
    return nu0 * np.exp(-2.0 * alpha * np.asarray(r) - np.asarray(dE) / (2 * kB_eV * T))


def marcus(dE, J, T, lam):
    """Marcus rate [1/s]. dE, J (transfer integral), lam (reorganisation) in eV."""
    kT = kB_eV * T
    pref = (2 * np.pi / hbar_eVs) * np.asarray(J) ** 2 / np.sqrt(4 * np.pi * lam * kT)
    return pref * np.exp(-(np.asarray(dE) + lam) ** 2 / (4 * lam * kT))


def marcus_levich_jortner(dE, J, T, lam_s, hw, S, nmax=40):
    """MLJ rate: classical outer-sphere ``lam_s`` + one quantum mode ``hw``
    with Huang-Rhys factor ``S`` (intramolecular reorganisation S*hw).

    This is the standard low-temperature form (hw >> kT, only vibrational
    emission into the quantum mode); it therefore does *not* obey detailed
    balance exactly. Use Marcus or MA rates when strict detailed balance is
    required (e.g. equilibrium / master-equation calculations)."""
    from math import factorial
    kT = kB_eV * T
    dE = np.asarray(dE, dtype=float)
    pref = (2 * np.pi / hbar_eVs) * np.asarray(J) ** 2 / np.sqrt(4 * np.pi * lam_s * kT)
    out = np.zeros(np.broadcast(dE, np.asarray(J)).shape)
    for n in range(nmax):
        fc = np.exp(-S) * S ** n / factorial(n)
        out = out + fc * np.exp(-(dE + lam_s + n * hw) ** 2 / (4 * lam_s * kT))
    return pref * out


def forster(dE, r, T, tau, R0):
    """Förster exciton transfer rate [1/s]; r and R0 in the same unit."""
    return (1.0 / tau) * (R0 / np.asarray(r)) ** 6 * boltzmann_uphill(dE, T)


def dexter(dE, r, T, kD, L):
    """Dexter (exchange) transfer rate [1/s]."""
    return kD * np.exp(-2.0 * np.asarray(r) / L) * boltzmann_uphill(dE, T)


def rate_by_code(code, dE, r, T, nu0=1e12, alpha=10.0, J0=0.01, lam=0.2,
                 hw=0.17, S=1.0):
    """Dispatch helper used by lattice solvers.

    For Marcus/MLJ the coupling decays as J = J0 exp(-alpha (r-1)) with r in
    lattice units, so that the nearest-neighbour coupling equals J0.
    """
    if code == RATE_MA:
        return miller_abrahams(dE, r, T, nu0, alpha)
    if code == RATE_SYM_MA:
        return symmetric_ma(dE, r, T, nu0, alpha)
    J = J0 * np.exp(-alpha * (np.asarray(r) - 1.0))
    if code == RATE_MARCUS:
        return marcus(dE, J, T, lam)
    if code == RATE_MLJ:
        return marcus_levich_jortner(dE, J, T, lam, hw, S)
    raise ValueError(f"unknown rate code {code}")
