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
