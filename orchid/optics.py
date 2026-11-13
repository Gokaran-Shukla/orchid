r"""Transfer-matrix optics for thin-film stacks (Pettersson, Roman, Inganäs,
J. Appl. Phys. 86, 487 (1999)) -> photogeneration profile G(x) for OPVs and
an approximate outcoupling efficiency for OLEDs.

Normal incidence, coherent layers between a semi-infinite incidence medium
(glass, treated incoherently at the air/glass interface) and a semi-infinite
exit medium. Complex refractive index :math:`\tilde n=n+ik`.

Local absorbed power density in layer j:
:math:`Q_j(x)=\tfrac12 c\varepsilon_0\,\alpha_j n_j\,|E_j(x)|^2`,
:math:`\alpha_j=4\pi k_j/\lambda`. Generation
:math:`G(x)=\int Q(x,\lambda)\,\lambda/(hc)\,d\lambda`.
"""
import numpy as np
from .constants import c0, eps0, h


def interface_matrix(n1, n2):
    r = (n1 - n2) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    return np.array([[1, r], [r, 1]], dtype=complex) / t


def layer_matrix(n, d, lam):
    xi = 2 * np.pi * n / lam
    return np.array([[np.exp(-1j * xi * d), 0], [0, np.exp(1j * xi * d)]], dtype=complex)


def tmm_field(layers, lam, n_in=1.5 + 0j, n_out=1.0 + 0j, npts=200):
    """Field and absorption for one wavelength.

    layers : list of (name, n_complex, thickness[m]).
    Returns dict(R, T, A_layers, x_list, E2_list) with |E|^2 normalised to the
    incident field in the incidence medium.
    """
    ns = [n_in] + [l[1] for l in layers] + [n_out]
    ds = [l[2] for l in layers]
    # total system matrix
    S = interface_matrix(ns[0], ns[1])
    for j in range(len(layers)):
        S = S @ layer_matrix(ns[j + 1], ds[j], lam) @ interface_matrix(ns[j + 1], ns[j + 2])
    r = S[1, 0] / S[0, 0]
    t = 1 / S[0, 0]
    R = abs(r) ** 2
    T = abs(t) ** 2 * (ns[-1].real / ns[0].real)
    xs, E2s, A = [], [], []
    for j in range(len(layers)):
        # S' (left of layer j) and S'' (right of layer j)
        Sp = interface_matrix(ns[0], ns[1])
        for k in range(j):
            Sp = Sp @ layer_matrix(ns[k + 1], ds[k], lam) @ interface_matrix(ns[k + 1], ns[k + 2])
        Spp = interface_matrix(ns[j + 1], ns[j + 2])
        for k in range(j + 1, len(layers)):
            Spp = Spp @ layer_matrix(ns[k + 1], ds[k], lam) @ interface_matrix(ns[k + 1], ns[k + 2])
        nj = ns[j + 1]
        dj = ds[j]
        xi = 2 * np.pi * nj / lam
        rpp = Spp[1, 0] / Spp[0, 0]
        # Pettersson et al. (1999) internal transmission coefficient
        tj = 1.0 / (Sp[0, 0] + Sp[0, 1] * rpp * np.exp(2j * xi * dj))
        x = np.linspace(0, dj, npts)
        E = tj * (np.exp(1j * xi * x) + rpp * np.exp(1j * xi * (2 * dj - x)))
        E2 = np.abs(E) ** 2
        alpha = 4 * np.pi * nj.imag / lam
        # absorbed fraction (normalised to incident intensity in n_in)
        Aj = np.trapezoid(alpha * nj.real * E2, x) / ns[0].real
        xs.append(x)
        E2s.append(E2)
        A.append(Aj)
    return dict(R=R, T=T, A=np.array(A), x=xs, E2=E2s)


def am15_approx(lam):
    """Blackbody(5778 K) spectrum scaled to 1000 W/m^2 over 280-4000 nm.

    A *placeholder* for the tabulated ASTM G173 AM1.5G spectrum; load the
    real table with :func:`load_spectrum` for quantitative work.
    """
    lam = np.asarray(lam)
    from .constants import kB
    T = 5778.0
    B = 2 * h * c0 ** 2 / lam ** 5 / np.expm1(h * c0 / (lam * kB * T))
    lg = np.linspace(280e-9, 4000e-9, 4000)
    Bg = 2 * h * c0 ** 2 / lg ** 5 / np.expm1(h * c0 / (lg * kB * T))
    return B * 1000.0 / np.trapezoid(Bg, lg)           # W m^-2 m^-1


def load_spectrum(path):
    """Two-column file: wavelength [nm], irradiance [W m^-2 nm^-1]."""
    d = np.loadtxt(path, comments="#")
    return d[:, 0] * 1e-9, d[:, 1] * 1e9


def lorentz_nk(lam, n_inf=1.8, E0=2.0, gamma=0.3, f=0.6):
    """Single Lorentz oscillator complex index (energies eV) - a simple
    absorber model for organic blends."""
    E = 1239.84193e-9 / np.asarray(lam)
    eps = n_inf ** 2 + f * E0 ** 2 / (E0 ** 2 - E ** 2 - 1j * gamma * E)
    return np.sqrt(eps)


def generation_profile(layers, active_index, lams, spectrum=None, n_in=1.5,
                       n_out=1.0, npts=200):
    """Photogeneration G(x) [m^-3 s^-1] in layer ``active_index``.

    ``layers`` entries are (name, n(lam) callable or complex, thickness).
    Assumes every absorbed photon in the active layer creates one exciton
    (scale by IQE/dissociation separately).
    """
    lams = np.asarray(lams)
    Sp = am15_approx(lams) if spectrum is None else spectrum(lams)
    G = None
    Atot = np.zeros(len(lams))
    for k, lam in enumerate(lams):
        L = [(nm, (n(lam) if callable(n) else n), d) for nm, n, d in layers]
        r = tmm_field(L, lam, n_in=complex(n_in), n_out=complex(n_out), npts=npts)
        nj = L[active_index][1]
        alpha = 4 * np.pi * nj.imag / lam
        I0 = Sp[k]                                    # W m^-2 m^-1
        Q = alpha * nj.real / complex(n_in).real * r["E2"][active_index] * I0
        g = Q * lam / (h * c0)
        G = g * 0 if G is None else G
        G = G + g * (lams[1] - lams[0] if len(lams) > 1 else 1.0)
        Atot[k] = r["A"][active_index]
    return r["x"][active_index], G, Atot


def outcoupling_classical(n_org):
    """Ray-optics escape-cone estimate eta_out ~ 1/(2 n^2)."""
    return 1.0 / (2 * n_org ** 2)


# ============================================================================
# Dipole emission in planar multilayers (Chance-Prock-Silbey / Neyts /
# Furno et al., PRB 85, 115205 (2012)) -> Purcell factor and outcoupling
# ============================================================================
def _kz(n, kpar, k0):
    kz = np.sqrt((n * k0) ** 2 - kpar ** 2 + 0j)
    return np.where(kz.imag < 0, -kz, kz)


def _stack_rt(ns, ds, kpar, k0, pol):
    """Reflection and transmission seen from medium ns[0] into ns[-1]
    through finite layers ns[1:-1] (thicknesses ds). p-polarisation uses
    H-field amplitudes: r = (n2^2 kz1 - n1^2 kz2)/(n2^2 kz1 + n1^2 kz2)."""
    kz = [_kz(n, kpar, k0) for n in ns]

    def rt(i):
        n1, n2, k1, k2 = ns[i], ns[i + 1], kz[i], kz[i + 1]
        if pol == "s":
            return (k1 - k2) / (k1 + k2), 2 * k1 / (k1 + k2)
        return ((n2 ** 2 * k1 - n1 ** 2 * k2) / (n2 ** 2 * k1 + n1 ** 2 * k2),
                2 * n2 ** 2 * k1 / (n2 ** 2 * k1 + n1 ** 2 * k2))
    r, t = rt(len(ns) - 2)
    for i in range(len(ns) - 3, -1, -1):
        rij, tij = rt(i)
        ph = np.exp(1j * kz[i + 1] * ds[i])
        den = 1 + rij * r * ph ** 2
        t = tij * t * ph / den
        r = (rij + r * ph ** 2) / den
    with np.errstate(divide="ignore", invalid="ignore"):
        if pol == "s":
            T = np.abs(t) ** 2 * np.real(kz[-1]) / np.real(kz[0])
        else:
            T = np.abs(t) ** 2 * np.real(kz[-1] / ns[-1] ** 2) / np.real(kz[0] / ns[0] ** 2)
    T = np.nan_to_num(T, nan=0.0, posinf=0.0, neginf=0.0)
    return r, T


def dipole_emission(lam, below, emitter, above, z0, u_max=3.0, n_theta=3000, n_ev=6000,
                    theta=1/3, n_air=1.0):
    r"""Power dissipation spectrum of an oscillating dipole in a planar stack.

    below : list of (n, d) from the emitter layer downwards, the last entry
            the semi-infinite substrate (d ignored), e.g. ITO, glass.
    emitter : (n_e, d_e) emitting layer (lossless), dipole at z0 from its
              bottom interface.
    above : list of (n, d) upwards, last entry semi-infinite (e.g. Al).
    theta : fraction of vertical dipoles (1/3 isotropic, 0 horizontal).

    Returns dict(u, K_v, K_h, F (Purcell factor), eta_substrate (into the
    bottom medium), eta_air (into air through a thick incoherent substrate),
    fractions of guided / evanescent power).
    Normalised so that F = 1 in an infinite homogeneous emitter medium.
    Integration uses u = sin(t) for u < 1 and u = cosh(t) for u > 1, which
    removes the 1/sqrt(1-u^2) end-point singularities.
    """
    n_e, d_e = emitter
    k0 = 2 * np.pi / lam
    ke = n_e.real * k0 if np.iscomplexobj(n_e) else n_e * k0
    th = np.linspace(0.0, np.pi / 2, n_theta)
    tt = np.linspace(1e-9, np.arccosh(u_max), n_ev)
    u = np.concatenate([np.sin(th), np.cosh(tt)])
    jac = np.concatenate([np.cos(th), np.sinh(tt)])
    var = np.concatenate([th, tt])
    n1 = n_theta

    def integ(f):
        g = f * jac
        return np.trapezoid(g[:n1], var[:n1]) + np.trapezoid(g[n1:], var[n1:])
    u = np.where(np.abs(u - 1) < 1e-12, 1 - 1e-12, u)
    kpar = u * ke
    nb = [n_e] + [b[0] for b in below]
    db = [b[1] for b in below[:-1]]
    na = [n_e] + [a_[0] for a_ in above]
    da = [a_[1] for a_ in above[:-1]]
    kze = _kz(n_e, kpar, k0)
    zl, zu = z0, d_e - z0
    res = {}
    for pol in ("s", "p"):
        rm, Tm = _stack_rt(nb, db, kpar, k0, pol)
        rp, Tp = _stack_rt(na, da, kpar, k0, pol)
        am = rm * np.exp(2j * kze * zl)
        ap = rp * np.exp(2j * kze * zu)
        res[pol] = (am, ap, Tm, Tp)
    w = np.sqrt(1 - u ** 2 + 0j)
    am, ap, Tm, _ = res["p"]
    D = 1 - am * ap
    Kv = 1.5 * np.real(u ** 3 / w * (1 + am) * (1 + ap) / D)
    Kv_out = 0.75 * np.real(u ** 3 / w) * np.abs(1 + ap) ** 2 / np.abs(D) ** 2 * Tm
    Khp = 0.75 * np.real(u * w * (1 - am) * (1 - ap) / D)
    Khp_out = 0.375 * np.real(u * w) * np.abs(1 - ap) ** 2 / np.abs(D) ** 2 * Tm
    am, ap, Tm, _ = res["s"]
    D = 1 - am * ap
    Khs = 0.75 * np.real(u / w * (1 + am) * (1 + ap) / D)
    Khs_out = 0.375 * np.real(u / w) * np.abs(1 + ap) ** 2 / np.abs(D) ** 2 * Tm
    Kh = Khp + Khs
    K = theta * Kv + (1 - theta) * Kh
    Kout = theta * Kv_out + (1 - theta) * (Khp_out + Khs_out)
    F = integ(K)
    n_sub = np.real(below[-1][0])
    u_sub = n_sub / np.real(n_e)
    u_air = n_air / np.real(n_e)
    m_sub = u < min(u_sub, 1.0)
    eta_sub = integ(np.where(m_sub, Kout, 0)) / F
    # thick incoherent substrate -> air, Fresnel transmission (polarisation averaged)
    sin_s = u * np.real(n_e) / n_sub
    cos_s = np.sqrt(np.clip(1 - sin_s ** 2, 0, 1))
    sin_a = np.clip(sin_s * n_sub / n_air, 0, 1)
    cos_a = np.sqrt(1 - sin_a ** 2)
    ts = 1 - ((n_sub * cos_s - n_air * cos_a) / (n_sub * cos_s + n_air * cos_a + 1e-300)) ** 2
    tp = 1 - ((n_air * cos_s - n_sub * cos_a) / (n_air * cos_s + n_sub * cos_a + 1e-300)) ** 2
    m_air = u < u_air
    Kp_out = theta * Kv_out + (1 - theta) * Khp_out
    Ks_out = (1 - theta) * Khs_out
    eta_air = integ(np.where(m_air, Kp_out * tp + Ks_out * ts, 0)) / F
    evan = integ(np.where(u > 1, K, 0)) / F
    return dict(u=u, K=K, K_v=Kv, K_h=Kh, F=F, eta_substrate=eta_sub, eta_air=eta_air,
                evanescent=evan, guided=1 - eta_sub - evan)


def cps_perfect_mirror(x):
    """Analytic decay rate (normalised) of a dipole at distance d from a
    perfect mirror, x = 2 k d (Drexhage; Chance, Prock & Silbey 1978).
    Returns (vertical, horizontal)."""
    x = np.asarray(x, float)
    Fv = 1 - 3 * (np.cos(x) / x ** 2 - np.sin(x) / x ** 3)
    Fh = 1 + 1.5 * (np.sin(x) / x ** 3 - np.cos(x) / x ** 2 - np.sin(x) / x)
    return Fv, Fh
