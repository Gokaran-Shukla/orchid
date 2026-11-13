r"""Electrostatics for 3D kinetic Monte Carlo.

* :func:`ewald_energy` - 3D Ewald summation for point charges in a periodic
  orthorhombic box (validated against the NaCl Madelung constant).
* :func:`ewald_lattice_green` - periodic Coulomb Green's function on a
  lattice (unit charge + neutralising background), used as a lookup table
  for bulk KMC with full long-range Coulomb interactions.
* :func:`image_self_energy` - image-charge self energy of a charge between
  two grounded metal electrodes (infinite image series, pairwise summed).
* :func:`sheet_green` - 1D Poisson Green's function (charge sheet between
  grounded electrodes); the long-range "mean-field" part of the device
  Coulomb interaction.

Energies are in eV when lengths are in metres and ``eps_r`` is given.
"""
import numpy as np
from scipy.special import erfc
from .constants import q, eps0


def _kcoul(eps_r):
    """e/(4 pi eps0 eps_r) in V m  (energy in eV for unit charges)."""
    return q / (4 * np.pi * eps0 * eps_r)


def ewald_energy(pos, charges, box, alpha=None, rcut=None, kmax=8, eps_r=1.0,
                 unit=False):
    """Total Coulomb energy of a neutral periodic system (Ewald).

    pos [n,3], charges [n] (units of e), box (Lx,Ly,Lz) in the same length
    unit as pos. If ``unit`` the energy is returned in units of
    e^2/(4 pi eps L_unit) (dimensionless), else in eV with lengths in m.
    """
    pos = np.asarray(pos, float)
    qv = np.asarray(charges, float)
    box = np.asarray(box, float)
    Lmin = box.min()
    if rcut is None:
        rcut = Lmin
    if alpha is None:
        alpha = 4.5 / rcut          # erfc(4.5) ~ 2e-10
    n = len(qv)
    # real space (minimum image, rcut <= L/2)
    E_r = 0.0
    nimg = int(np.ceil(rcut / Lmin))
    rng = range(-nimg, nimg + 1)
    for i in range(n):
        d = pos[i] - pos
        for ix in rng:
            for iy in rng:
                for iz in rng:
                    dd = d + np.array([ix, iy, iz]) * box
                    r = np.sqrt((dd ** 2).sum(1))
                    m = (r > 1e-12) & (r < rcut)
                    E_r += 0.5 * np.sum(qv[i] * qv[m] * erfc(alpha * r[m]) / r[m])
    # reciprocal space
    V = np.prod(box)
    ks = np.arange(-kmax, kmax + 1)
    KX, KY, KZ = np.meshgrid(ks, ks, ks, indexing="ij")
    K = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], 1) * (2 * np.pi / box)
    K = K[(K ** 2).sum(1) > 0]
    k2 = (K ** 2).sum(1)
    S = (qv[None, :] * np.exp(1j * K @ pos.T)).sum(1)
    E_k = (2 * np.pi / V) * np.sum(np.exp(-k2 / (4 * alpha ** 2)) / k2 * np.abs(S) ** 2)
    E_s = -alpha / np.sqrt(np.pi) * np.sum(qv ** 2)
    E = E_r + E_k + E_s
    return E if unit else E * _kcoul(eps_r)


def madelung_nacl(kmax=8):
    """Madelung constant of rock salt from :func:`ewald_energy` (1.747565)."""
    a = 1.0
    pos = []
    qv = []
    for x in range(2):
        for y in range(2):
            for z in range(2):
                pos.append([x * 0.5, y * 0.5, z * 0.5])
                qv.append((-1) ** (x + y + z))
    E = ewald_energy(np.array(pos), np.array(qv, float), (a, a, a), kmax=kmax, unit=True)
    r0 = 0.5
    return -E * r0 / 4.0


def ewald_lattice_green(shape, alpha=None, kmax=6, nimg=1):
    r"""Periodic Green's function G(d) on an integer lattice (units 1/a):
    potential at displacement d from a unit charge plus uniform neutralising
    background, Ewald-summed. Returns array of ``shape``; G[0,0,0] is the
    regular part (self term excluded)."""
    shape = tuple(shape)
    box = np.array(shape, float)
    if alpha is None:
        alpha = 5.0 / box.min()
    grids = np.meshgrid(*[np.arange(n) for n in shape], indexing="ij")
    P = np.stack([g.ravel() for g in grids], 1).astype(float)
    P = P - box * np.round(P / box)
    G = np.zeros(len(P))
    for ix in range(-nimg, nimg + 1):
        for iy in range(-nimg, nimg + 1):
            for iz in range(-nimg, nimg + 1):
                d = P + np.array([ix, iy, iz]) * box
                r = np.sqrt((d ** 2).sum(1))
                m = r > 1e-12
                G[m] += erfc(alpha * r[m]) / r[m]
    V = np.prod(box)
    ks = np.arange(-kmax, kmax + 1)
    KX, KY, KZ = np.meshgrid(ks, ks, ks, indexing="ij")
    K = np.stack([KX.ravel(), KY.ravel(), KZ.ravel()], 1) * (2 * np.pi / box)
    K = K[(K ** 2).sum(1) > 0]
    k2 = (K ** 2).sum(1)
    w = (4 * np.pi / V) * np.exp(-k2 / (4 * alpha ** 2)) / k2
    for start in range(0, len(K), 256):
        Kc = K[start:start + 256]
        G += (w[start:start + 256, None] * np.cos(Kc @ P.T)).sum(0)
    G -= np.pi / (alpha ** 2 * V)          # neutralising background
    G[0] -= 2 * alpha / np.sqrt(np.pi)     # remove self interaction
    return G.reshape(shape)


def image_self_energy(z, L, eps_r, nimg=2000):
    r"""Self energy (eV) of a point charge at distance z (m) from a grounded
    electrode at 0, with a second grounded electrode at L.

    Image series: -e at 2nL - z (all n), +e at z + 2nL (n != 0):
    U = (e/(8 pi eps)) sum_n [ 1/(2|n|L) (n!=0) - 1/|2(z - nL)| ].
    Close to one electrode it reduces to -e^2/(16 pi eps z).
    """
    z = np.atleast_1d(np.asarray(z, float))
    n = np.arange(-nimg, nimg + 1)
    term_img = np.where(n != 0, 1.0 / (2 * np.abs(n) * L + (n == 0)), 0.0)
    U = (term_img[None, :] - 1.0 / np.abs(2 * (z[:, None] - n[None, :] * L))).sum(1)
    return 0.5 * _kcoul(eps_r) * U


def sheet_green(z1, z2, L, area, eps_r):
    r"""Potential (V) at z1 from a uniform sheet of charge +e (area ``area``)
    at z2 between grounded electrodes at 0 and L:
    G = e z_< (L - z_>) / (eps A L)."""
    zl = np.minimum(z1, z2)
    zg = np.maximum(z1, z2)
    return q * zl * (L - zg) / (eps0 * eps_r * area * L)
