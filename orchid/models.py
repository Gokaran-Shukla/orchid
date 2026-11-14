r"""Analytical / parametrised mobility models for disordered organics.

* :func:`gdm_bassler` - Bässler Gaussian disorder model (Acc. Chem. Res. /
  phys. stat. sol. (b) 175, 15 (1993)):
  :math:`\mu=\mu_\infty\exp[-(2\hat\sigma/3)^2]\exp[C(\hat\sigma^2-\Sigma^2)\sqrt F]`
  (Sigma>=1.5; for Sigma<1.5 use Sigma^2 -> 2.25).
* :func:`cdm_novikov` - correlated disorder model (Novikov et al., PRL 81,
  4472 (1998)): :math:`\mu=\mu_\infty\exp[-(3\hat\sigma/5)^2+C_0(\hat\sigma^{3/2}-\Gamma)\sqrt{eaF/\sigma}]`.
* :func:`egdm_pasveer` - extended GDM (Pasveer et al., PRL 94, 206601 (2005)).
* :func:`poole_frenkel`, :func:`arrhenius` - empirical laws.
* Gaussian-DOS statistics: Fermi level, generalised Einstein relation,
  equilibrium energy, Vissenberg-Matters exponential-DOS mobility.
"""
import numpy as np
from scipy import integrate, optimize
from .constants import kB_eV, q


# ---------------------------------------------------------------- GDM / CDM
def gdm_bassler(T, F, sigma, mu_inf, C=2.9e-4, Sigma=0.0):
    """Bässler GDM.  F in V/cm (C in (cm/V)^1/2, default 2.9e-4)."""
    s = sigma / (kB_eV * T)
    S2 = np.where(np.asarray(Sigma) >= 1.5, np.asarray(Sigma) ** 2, 2.25)
    return mu_inf * np.exp(-(2 * s / 3) ** 2) * np.exp(C * (s ** 2 - S2) * np.sqrt(F))


def cdm_novikov(T, F, sigma, mu_inf, a=1e-9, C0=0.78, Gamma=2.0):
    """Novikov correlated disorder model. F in V/m, a in m."""
    s = sigma / (kB_eV * T)
    return mu_inf * np.exp(-(0.6 * s) ** 2 + C0 * (s ** 1.5 - Gamma)
                           * np.sqrt(a * F / sigma))


def poole_frenkel(F, mu0, beta):
    """mu = mu0 exp(beta sqrt(F))."""
    return mu0 * np.exp(beta * np.sqrt(F))


def arrhenius(T, mu_pref, Ea):
    return mu_pref * np.exp(-Ea / (kB_eV * T))


# ---------------------------------------------------------------- EGDM
EGDM_C1 = 1.8e-9
EGDM_C2 = 0.42


def egdm_mu0T(T, sigma, mu0, c1=EGDM_C1, c2=EGDM_C2):
    s = sigma / (kB_eV * T)
    return mu0 * c1 * np.exp(-c2 * s ** 2)


def egdm_g1(p_site, T, sigma):
    """Density enhancement; p_site = p a^3 (fraction of sites), valid <=0.1.

    Pasveer's parametrisation needs delta > 0, i.e. sigma/kT > ~1.82. Below
    that the density dependence is negligible and 1 is returned (the formula
    itself would diverge as p -> 0).
    """
    s = sigma / (kB_eV * T)
    pc = np.minimum(np.asarray(p_site, float), 0.1)
    if s ** 2 - s <= np.log(4.0):
        return np.ones_like(pc)
    delta = 2 * (np.log(s ** 2 - s) - np.log(np.log(4.0))) / s ** 2
    return np.exp(0.5 * (s ** 2 - s) * (2 * pc) ** delta)


def egdm_g2(F, T, sigma, a):
    """Field enhancement; F in V/m, a in m."""
    s = sigma / (kB_eV * T)
    Er = a * np.asarray(F, float) / sigma
    return np.exp(0.44 * (s ** 1.5 - 2.2) * (np.sqrt(1 + 0.8 * np.minimum(Er, 2.0) ** 2) - 1))


def egdm_pasveer(T, p, F, sigma, a, mu0=None, nu0=None):
    """Full EGDM mobility mu(T, p, F) [m^2/Vs].

    p : carrier density [m^-3]; F [V/m]; a = 1/N_t^{1/3} [m].
    Supply either mu0 (prefactor) or nu0 (attempt frequency, then
    mu0 = a^2 nu0 e / sigma).
    """
    if mu0 is None:
        mu0 = a ** 2 * nu0 / sigma      # e/sigma with sigma in eV -> 1/V
    return egdm_mu0T(T, sigma, mu0) * egdm_g1(p * a ** 3, T, sigma) * egdm_g2(F, T, sigma, a)


# ---------------------------------------------------------------- statistics
def _gauss(E, sigma):
    return np.exp(-E ** 2 / (2 * sigma ** 2)) / np.sqrt(2 * np.pi * sigma ** 2)


def gaussian_density(EF, T, sigma, Nt=1.0, Ec=0.0):
    """Carrier density for Fermi level EF in a Gaussian DOS centred at Ec."""
    kT = kB_eV * T
    f = lambda E: _gauss(E - Ec, sigma) / (1 + np.exp(np.clip((E - EF) / kT, -700, 700)))
    lo, hi = Ec - 10 * sigma, Ec + 10 * sigma
    val, _ = integrate.quad(f, lo, hi, points=[EF] if lo < EF < hi else None, limit=200)
    return Nt * val


def fermi_level_gaussian(n_frac, T, sigma):
    """Fermi level (eV, relative to DOS centre) for occupied fraction n_frac.

    Convention: carriers occupy the *low* energy side (holes measured in the
    hole-energy picture), i.e. f = 1/(1+exp((E-EF)/kT)).
    """
    g = lambda EF: np.log(gaussian_density(EF, T, sigma)) - np.log(n_frac)
    return optimize.brentq(g, -15 * sigma - 50 * kB_eV * T, 5 * sigma)


def einstein_ratio_gaussian(n_frac, T, sigma):
    r"""Generalised Einstein ratio  D/(mu kT/q) = (n/kT) / (dn/dE_F)."""
    EF = fermi_level_gaussian(n_frac, T, sigma)
    d = 1e-4 * sigma
    dn = (gaussian_density(EF + d, T, sigma) - gaussian_density(EF - d, T, sigma)) / (2 * d)
    return n_frac / (kB_eV * T * dn)


def equilibrium_energy(T, sigma):
    """Mean energy of a single carrier in thermal equilibrium: -sigma^2/kT."""
    return -sigma ** 2 / (kB_eV * T)


def vissenberg_matters(T, p, sigma0_cond, T0, alpha, Bc=2.8):
    r"""Vissenberg-Matters percolation conductivity (PRB 57, 12964 (1998))
    for an exponential DOS of width k_B T0 (valid for T < T0).

    sigma = sigma0 [ pi p (T0/T)^3 / ((2 alpha)^3 Bc Gamma(1-T/T0) Gamma(1+T/T0)) ]^(T0/T)

    p : carrier density [m^-3]; alpha : inverse localisation length [1/m].
    Returns conductivity [S/m].
    """
    from math import gamma as G
    if T >= T0:
        return np.nan
    x = T / T0
    return sigma0_cond * (np.pi * p * (1 / x) ** 3 /
                          ((2 * alpha) ** 3 * Bc * G(1 - x) * G(1 + x))) ** (1 / x)
