r"""1D drift-diffusion device solver for organic diodes (OLEDs, OPVs,
single-carrier devices) with a fully coupled Newton method.

Equations (SI units)
--------------------
.. math::
   \nabla\cdot(\varepsilon\nabla\psi) = -q\,(p - n + N_D^+ - N_A^- + \rho_t/q)

   \partial_x J_n = q\,(R - G),\qquad \partial_x J_p = -q\,(R - G)

Currents use the Scharfetter-Gummel scheme with the *generalised* Einstein
relation :math:`D = g_3\,\mu V_t` (Gaussian-DOS diffusion enhancement):

.. math::
   J_{n,i+\frac12} = \frac{q\mu_n g_3 V_t}{h}\left[n_{i+1}B(\Delta/g_3)-n_iB(-\Delta/g_3)\right],
   \quad \Delta=\frac{\psi_{i+1}-\psi_i}{V_t},\quad B(x)=\frac{x}{e^x-1}

Unknowns are :math:`\psi`, :math:`u=\ln(n/N_c)`, :math:`v=\ln(p/N_v)`
(positivity by construction). The Jacobian is built by 11-colour
finite differences exploiting the block-tridiagonal band (bandwidth 5),
which makes any mobility law :math:`\mu(n,F,T)` (EGDM, Poole-Frenkel,...)
plug-and-play. Physics options:

* mobility: constant, Poole-Frenkel, EGDM (Pasveer 2005) + g3 Einstein
* recombination: Langevin (reduction factor), SRH via single trap level
  (with trapped space charge), direct band-to-band
* generation: uniform, Beer-Lambert, or user profile (e.g. transfer-matrix)
* CT dissociation: Braun-Onsager field-dependent P(F) (Koster 2005 scheme)
* contacts: injection barriers (Ohmic -> Schottky), built-in voltage
"""
import numpy as np
from scipy.linalg import solve_banded
from .constants import q, eps0, kB_eV


def bernoulli(x):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    small = np.abs(x) < 1e-6
    out[small] = 1.0 - x[small] / 2.0 + x[small] ** 2 / 12.0
    xs = x[~small]
    # stable for both signs: B(x) = x e^{-x} / (1 - e^{-x}) for x > 0
    pos = xs > 0
    o = np.empty_like(xs)
    xp = xs[pos]
    o[pos] = xp * np.exp(-xp) / (-np.expm1(-xp))
    xn = xs[~pos]
    o[~pos] = xn / np.expm1(xn)
    out[~small] = o
    return out


def braun_dissociation(F, T, a_ct, kf, eps_r, mu_sum, n_terms=True):
    r"""Braun-Onsager CT dissociation probability P(F,T).

    P = k_d/(k_d + k_f),  k_d = (3 gamma / 4 pi a^3) e^{-E_b/kT} J1(2 sqrt(-2b))/sqrt(-2b),
    b = q^3 F / (8 pi eps kT^2),  gamma = q mu_sum / eps.
    """
    from scipy.special import iv
    eps = eps0 * eps_r
    kT = kB_eV * T * q
    Eb = q ** 2 / (4 * np.pi * eps * a_ct)
    gam = q * mu_sum / eps
    b = q ** 3 * np.abs(F) / (8 * np.pi * eps * kT ** 2)
    x = 2 * np.sqrt(2 * b)
    ratio = np.where(x < 1e-8, 1.0, 2 * iv(1, x) / np.maximum(x, 1e-300))
    kd = 3 * gam / (4 * np.pi * a_ct ** 3) * np.exp(-Eb / kT) * ratio
    return kd / (kd + kf)


class Device:
    """Single-layer organic diode description.

    Parameters (SI unless noted)
    ----------------------------
    L : thickness [m];  N : grid nodes
    eps_r : relative permittivity;  T : temperature [K]
    Eg : transport gap [eV] (LUMO-HOMO);  Nc, Nv : effective DOS [m^-3]
    phi_anode_p : hole barrier at anode (x=0) [eV]
    phi_cathode_n : electron barrier at cathode (x=L) [eV]
    mu_n, mu_p : mobility [m^2/Vs] or callables mu(dens, F, T)
    g3_n, g3_p : diffusion enhancement callables g3(dens) (default 1)
    langevin_prefactor : zeta (reduced Langevin)
    G : generation rate [m^-3 s^-1] scalar or array on nodes
    srh : dict(Nt, Et (eV below LUMO), Cn, Cp) single trap level or None
    braun : dict(a_ct, kf) to enable CT dissociation or None
    """

    def __init__(self, L=100e-9, N=201, eps_r=3.5, T=300.0, Eg=1.2, Nc=2.5e26,
                 Nv=2.5e26, phi_anode_p=0.1, phi_cathode_n=0.1, mu_n=1e-8,
                 mu_p=1e-8, g3_n=None, g3_p=None, langevin_prefactor=1.0, G=0.0,
                 srh=None, braun=None, Nd=0.0, Na=0.0, recombination=True,
                 grid_beta=3.0):
        self.__dict__.update(locals())
        del self.__dict__["self"]
        s = np.linspace(0.0, 1.0, N)
        if grid_beta > 0:   # tanh grid refined at both contacts
            s = 0.5 * (1 + np.tanh(grid_beta * (2 * s - 1)) / np.tanh(grid_beta))
        self.x = L * s
        self.h = np.diff(self.x)                       # edge lengths
        self.dV = 0.5 * (self.h[1:] + self.h[:-1])     # dual cell of interior nodes
        self.Vt = kB_eV * T
        self.eps = eps0 * eps_r
        self.Gx = np.broadcast_to(np.asarray(G, float), (N,)).copy()
        self.ni2 = Nc * Nv * np.exp(-Eg / self.Vt)

    # ---------------------------------------------------------------- helpers
    def _mu(self, mu, dens_edge, F_edge):
        if callable(mu):
            return mu(dens_edge, F_edge, self.T)
        return np.full_like(dens_edge, mu)

    def _g3(self, g3, dens_edge):
        if g3 is None:
            return np.ones_like(dens_edge)
        return g3(dens_edge)

    def boundary(self, V):
        d = self
        n0 = d.Nc * np.exp(-(d.Eg - d.phi_anode_p) / d.Vt)
        p0 = d.Nv * np.exp(-d.phi_anode_p / d.Vt)
        nL = d.Nc * np.exp(-d.phi_cathode_n / d.Vt)
        pL = d.Nv * np.exp(-(d.Eg - d.phi_cathode_n) / d.Vt)
        Vbi = d.Eg - d.phi_anode_p - d.phi_cathode_n
        return (0.0, np.log(n0 / d.Nc), np.log(p0 / d.Nv)), \
               (Vbi - V, np.log(nL / d.Nc), np.log(pL / d.Nv)), Vbi

    def unpack(self, z, V):
        (a0, u0, v0), (aL, uL, vL), _ = self.boundary(V)
        Z = z.reshape(-1, 3)
        psi = np.concatenate([[a0], Z[:, 0], [aL]])
        u = np.concatenate([[u0], Z[:, 1], [uL]])
        v = np.concatenate([[v0], Z[:, 2], [vL]])
        return psi, u, v

    def currents(self, psi, n, p):
        d = self
        F = -(psi[1:] - psi[:-1]) / d.h
        ne = 0.5 * (n[1:] + n[:-1])
        pe = 0.5 * (p[1:] + p[:-1])
        mun = d._mu(d.mu_n, ne, np.abs(F))
        mup = d._mu(d.mu_p, pe, np.abs(F))
        g3n = d._g3(d.g3_n, ne)
        g3p = d._g3(d.g3_p, pe)
        D = (psi[1:] - psi[:-1]) / d.Vt
        Jn = q * mun * g3n * d.Vt / d.h * (n[1:] * bernoulli(D / g3n) - n[:-1] * bernoulli(-D / g3n))
        Jp = q * mup * g3p * d.Vt / d.h * (p[:-1] * bernoulli(D / g3p) - p[1:] * bernoulli(-D / g3p))
        return Jn, Jp, F, mun, mup

    def recomb(self, psi, n, p, F_node):
        d = self
        R = np.zeros_like(n)
        rho_t = np.zeros_like(n)
        G = d.Gx.copy()
        if d.recombination:
            mun = d._mu(d.mu_n, n, np.abs(F_node))
            mup = d._mu(d.mu_p, p, np.abs(F_node))
            gam = d.langevin_prefactor * q * (mun + mup) / d.eps
            Rl = gam * (n * p - d.ni2)
            if d.braun is not None:
                P = braun_dissociation(F_node, d.T, d.braun["a_ct"], d.braun["kf"],
                                       d.eps_r, mun + mup)
                G = P * G
                Rl = (1 - P) * Rl
            R = R + Rl
        if d.srh is not None:
            s = d.srh
            n1 = d.Nc * np.exp(-s["Et"] / d.Vt)
            p1 = d.Nv * np.exp(-(d.Eg - s["Et"]) / d.Vt)
            Cn, Cp, Nt = s["Cn"], s["Cp"], s["Nt"]
            R = R + Cn * Cp * Nt * (n * p - d.ni2) / (Cn * (n + n1) + Cp * (p + p1))
            ft = (Cn * n + Cp * p1) / (Cn * (n + n1) + Cp * (p + p1))
            rho_t = -q * Nt * ft if s.get("acceptor", True) else q * Nt * (1 - ft)
        return R, G, rho_t

    def residual(self, z, V):
        d = self
        psi, u, v = d.unpack(z, V)
        n = d.Nc * np.exp(u)
        p = d.Nv * np.exp(v)
        Jn, Jp, F, _, _ = d.currents(psi, n, p)
        Fn = np.empty_like(psi)
        Fn[1:-1] = 0.5 * (F[1:] + F[:-1])
        Fn[0], Fn[-1] = F[0], F[-1]
        R, G, rho_t = d.recomb(psi, n, p, Fn)
        h = d.h
        dV = d.dV
        Eflux = (psi[1:] - psi[:-1]) / h
        lap = (Eflux[1:] - Eflux[:-1]) / dV
        rho = q * (p - n + d.Nd - d.Na) + rho_t
        r_psi = (lap + rho[1:-1] / d.eps) * dV ** 2 / d.Vt
        mid = slice(1, -1)
        # continuity rows scaled by the local density (relative residual,
        # same root, much better conditioned for densities spanning decades)
        r_n = ((Jn[1:] - Jn[:-1]) / dV - q * (R[mid] - G[mid])) / (q * (n[mid] + 1.0)) * 1e-6
        r_p = ((Jp[1:] - Jp[:-1]) / dV + q * (R[mid] - G[mid])) / (q * (p[mid] + 1.0)) * 1e-6
        return np.stack([r_psi, r_n, r_p], 1).ravel()

    # ---------------------------------------------------------------- Newton
    def jacobian_banded(self, z, V, r0):
        nz = len(z)
        bw = 5
        ab = np.zeros((2 * bw + 1, nz))
        for c in range(2 * bw + 1):
            cols = np.arange(c, nz, 2 * bw + 1)
            dz = 1e-7 * np.maximum(1.0, np.abs(z[cols]))
            zp = z.copy()
            zp[cols] += dz
            dr = (self.residual(zp, V) - r0)
            for j, dj in zip(cols, dz):
                lo, hi = max(0, j - bw), min(nz, j + bw + 1)
                rows = np.arange(lo, hi)
                ab[bw + rows - j, j] = dr[rows] / dj
        return ab

    def newton(self, z, V, tol=1e-9, maxit=60, verbose=False):
        for it in range(maxit):
            r = self.residual(z, V)
            ab = self.jacobian_banded(z, V, r)
            dz = solve_banded((5, 5), ab, -r)
            m = np.max(np.abs(dz.reshape(-1, 3)) / np.array([1.0, 1.0, 1.0]))
            damp = min(1.0, 2.0 / m) if m > 2.0 else 1.0
            z = z + damp * dz
            if verbose:
                print(it, m, np.abs(r).max())
            if m < tol:
                return z, True, it
        return z, False, it

    def initial_guess(self, V):
        (a0, u0, v0), (aL, uL, vL), _ = self.boundary(V)
        s = self.x[1:-1] / self.L
        return np.stack([a0 + (aL - a0) * s, u0 + (uL - u0) * s, v0 + (vL - v0) * s], 1).ravel()

    def solve(self, V, z0=None, **kw):
        z = self.initial_guess(V) if z0 is None else z0
        return self.newton(z, V, **kw)

    def jv(self, Vs, verbose=False):
        """Sweep voltages (continuation, adaptive sub-stepping).

        Returns dict(V, J, Jn_profile...) with J in A/m^2 (positive = forward).
        """
        Vs = np.asarray(Vs, float)
        _, _, Vbi = self.boundary(0.0)
        # equilibrium start
        z, ok, _ = self.solve(0.0)
        if not ok:
            # ramp generation / start from flat-band
            z, ok, _ = self.solve(0.0, z0=self.initial_guess(0.0), maxit=300)
        # sweep away from 0 in both directions separately
        res = {}
        for sign in (+1, -1):
            zc, Vc = z.copy(), 0.0
            targets = sorted([v for v in Vs if (v >= 0 if sign > 0 else v < 0)],
                             key=abs)
            for Vt_ in targets:
                step = Vt_ - Vc
                while abs(Vt_ - Vc) > 1e-12:
                    Vn = Vc + step if abs(step) < abs(Vt_ - Vc) else Vt_
                    zn, ok, _ = self.newton(zc, Vn)
                    if ok:
                        zc, Vc = zn, Vn
                        step = step * 1.5
                    else:
                        step = step / 3.0
                        if abs(step) < 1e-6:
                            raise RuntimeError(f"no convergence at V={Vn}")
                res[Vt_] = zc.copy()
        Vlist = list(Vs)
        J = []
        prof = []
        for Vv in Vlist:
            zc = res[Vv]
            psi, u, vv = self.unpack(zc, Vv)
            n = self.Nc * np.exp(u)
            p = self.Nv * np.exp(vv)
            Jn, Jp, F, _, _ = self.currents(psi, n, p)
            Jt = Jn + Jp
            J.append(np.median(Jt))
            prof.append(dict(psi=psi, n=n, p=p, Jn=Jn, Jp=Jp, F=F))
        return dict(V=Vs, J=np.array(J), profiles=prof, Vbi=Vbi)


# ---------------------------------------------------------------- PV metrics
def pv_metrics(V, J, P_in=1000.0):
    """Jsc, Voc, FF, PCE from a J-V curve (J>0 forward, A/m^2; P_in W/m^2).

    A monotone cubic (PCHIP) interpolant is used, so Voc and Pmax do not
    suffer from the linear-interpolation error of coarse voltage grids.
    """
    from scipy.interpolate import PchipInterpolator
    from scipy.optimize import brentq
    V = np.asarray(V, float)
    J = np.asarray(J, float)
    o = np.argsort(V)
    V, J = V[o], J[o]
    f = PchipInterpolator(V, J)
    Jsc = -float(f(0.0))
    s = np.where(np.diff(np.sign(J)) != 0)[0]
    if len(s) == 0:
        return dict(Jsc=Jsc, Voc=np.nan, FF=np.nan, PCE=np.nan, Pmax=np.nan)
    k = s[-1]
    Voc = brentq(lambda v: float(f(v)), V[k], V[k + 1], xtol=1e-9)
    Vf = np.linspace(0, Voc, 2001)
    P = -Vf * f(Vf)
    Pm = float(P.max())
    return dict(Jsc=Jsc, Voc=Voc, FF=Pm / (Jsc * Voc), PCE=Pm / P_in, Pmax=Pm)


def mott_gurney(V, L, eps_r, mu):
    """SCLC Mott-Gurney law J = 9/8 eps mu V^2/L^3 [A/m^2]."""
    return 9.0 / 8.0 * eps0 * eps_r * mu * np.asarray(V) ** 2 / L ** 3


def koster_voc(Eg, T, Nc, gamma, G):
    """Koster et al. APL 86, 123509 (2005) Voc for free-carrier generation:
    Voc = Eg/q - kT/q ln(gamma Nc^2 / G)."""
    return Eg - kB_eV * T * np.log(gamma * Nc ** 2 / G)
