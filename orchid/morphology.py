r"""Molecular morphologies, quantum-chemical couplings and graph transport
(roadmap R21; VOTCA-style workflow).

1. Morphology: :func:`random_packing` (periodic random sequential addition
   with hard-core distance, an amorphous-film surrogate) or
   :func:`read_xyz_centres` (molecular centres from an MD snapshot).
2. Couplings: :func:`dipro` - projective dimer method (Valeev et al.,
   JACS 128, 9882 (2006); Baumeier et al., PCCP 12, 11103 (2010)):

   .. math::
      J_{eff}=\frac{J_{AB}-\tfrac12(e_A+e_B)S_{AB}}{1-S_{AB}^2}

   with :math:`J_{AB},e_A,e_B,S_{AB}` from projecting the dimer Fock matrix onto
   the monomer frontier orbitals. :func:`dipro_pyscf` runs it with PySCF
   (HF/DFT). :func:`coupling_model` gives the analytic decay
   :math:`J=J_0 e^{-\beta(r-r_0)}`, which can be fitted to DIPRO data.
3. Site energies: Gaussian + dipolar (random molecular dipoles, minimum
   image, cut-off) via :func:`dipolar_site_energies`.
4. :class:`MolecularGraph` exposes the same interface as
   :class:`orchid.lattice.Lattice` (``nbr``, ``disp``, ``dist``, ``energy``,
   ``rates``), so the master-equation and independent-carrier KMC solvers run
   unchanged on arbitrary morphologies.
"""
import numpy as np
from .constants import kB_eV, hbar_eVs, q, eps0


# ------------------------------------------------------------------ morphology
def random_packing(n, box, d_min, seed=0, max_tries=200000):
    """Periodic random sequential addition of n points with hard-core d_min."""
    rng = np.random.default_rng(seed)
    box = np.asarray(box, float)
    pts = np.empty((0, 3))
    tries = 0
    while len(pts) < n and tries < max_tries:
        tries += 1
        p = rng.random(3) * box
        if len(pts):
            d = pts - p
            d -= box * np.round(d / box)
            if np.min((d ** 2).sum(1)) < d_min ** 2:
                continue
        pts = np.vstack([pts, p])
    if len(pts) < n:
        raise RuntimeError("packing too dense for random sequential addition")
    return pts


def read_xyz_centres(path, atoms_per_molecule):
    """Centres of mass (equal weights) of consecutive molecules in an XYZ file."""
    lines = open(path).read().split("\n")
    nat = int(lines[0])
    xyz = np.array([[float(v) for v in l.split()[1:4]] for l in lines[2:2 + nat]])
    return xyz.reshape(-1, atoms_per_molecule, 3).mean(1) * 1e-10   # Angstrom -> m


# ------------------------------------------------------------------ couplings
def dipro(F_dimer, S_dimer, C_A, C_B, nA):
    """Projective dimer approach.

    F_dimer, S_dimer : dimer Fock and overlap matrices (AO basis, A atoms first).
    C_A, C_B : monomer MO coefficient vectors of the frontier orbital (AO basis
    of each monomer, nA = number of AO of A). Returns dict(J_eff, J, eA, eB, S).
    """
    nAO = F_dimer.shape[0]
    a = np.zeros(nAO)
    b = np.zeros(nAO)
    a[:nA] = C_A
    b[nA:] = C_B
    eA = a @ F_dimer @ a
    eB = b @ F_dimer @ b
    J = a @ F_dimer @ b
    S = a @ S_dimer @ b
    Jeff = (J - 0.5 * (eA + eB) * S) / (1 - S ** 2)
    return dict(J_eff=Jeff, J=J, eA=eA, eB=eB, S=S)


def dipro_pyscf(atoms_A, atoms_B, basis="sto-3g", orbital="HOMO", method="HF"):
    """Run DIPRO with PySCF. atoms_*: list of (symbol, (x,y,z) Angstrom).

    Returns dict with J_eff [eV], the dimer frontier splitting [eV], and the
    raw projection quantities.
    """
    from pyscf import gto, scf, dft
    HARTREE = 27.211386245988

    def run(atoms):
        mol = gto.M(atom=atoms, basis=basis, verbose=0)
        mf = scf.RHF(mol) if method == "HF" else dft.RKS(mol).set(xc=method)
        mf.kernel()
        return mol, mf
    molA, mfA = run(atoms_A)
    molB, mfB = run(atoms_B)
    molD, mfD = run(list(atoms_A) + list(atoms_B))

    def frontier(mf, mol):
        nocc = mol.nelectron // 2
        k = nocc - 1 if orbital == "HOMO" else nocc
        return mf.mo_coeff[:, k], k
    cA, _ = frontier(mfA, molA)
    cB, _ = frontier(mfB, molB)
    F = mfD.get_fock()
    S = mfD.get_ovlp()
    res = dipro(F, S, cA, cB, molA.nao)
    nocc = molD.nelectron // 2
    e = mfD.mo_energy
    split = (e[nocc - 1] - e[nocc - 2]) if orbital == "HOMO" else (e[nocc + 1] - e[nocc])
    return dict(J_eff=res["J_eff"] * HARTREE, splitting=split * HARTREE,
                eA=res["eA"] * HARTREE, eB=res["eB"] * HARTREE, S=res["S"])


def coupling_model(r, J0=0.05, beta=2.0e10, r0=0.5e-9):
    """J = J0 exp(-beta (r - r0)) [eV], r in m."""
    return J0 * np.exp(-beta * (np.asarray(r) - r0))


# ------------------------------------------------------------------ site energies
def dipolar_site_energies(pos, box, dipole_debye=3.0, eps_r=3.0, r_cut=None, seed=0):
    """Electrostatic energy (eV) of a +e charge at each site in the field of
    randomly oriented molecular dipoles on all other sites (minimum image)."""
    rng = np.random.default_rng(seed)
    n = len(pos)
    box = np.asarray(box, float)
    p = rng.normal(size=(n, 3))
    p /= np.linalg.norm(p, axis=1)[:, None]
    p *= dipole_debye * 3.33564e-30
    if r_cut is None:
        r_cut = 0.5 * box.min()
    E = np.zeros(n)
    for i in range(n):
        d = pos[i] - pos
        d -= box * np.round(d / box)
        r = np.linalg.norm(d, axis=1)
        m = (r > 0) & (r < r_cut)
        E[i] = np.sum((p[m] * d[m]).sum(1) / r[m] ** 3) / (4 * np.pi * eps0 * eps_r)
    return E          # J/C = V  -> eV for unit charge


# ------------------------------------------------------------------ graph
class MolecularGraph:
    """Hopping graph on arbitrary molecular positions (lattice-compatible).

    Lengths are stored in units of ``a`` = (V/N)^(1/3) so that the existing
    solvers (rates with alpha in 1/a, F_red = eFa) apply unchanged.
    couplings : None (Miller-Abrahams with exp(-2 alpha r)) or callable
    J(r [m]) -> eV used by Marcus rates (``kind="marcus_graph"``).
    """

    def __init__(self, pos, box, r_cut, energies=None, couplings=None, max_nbr=None):
        pos = np.asarray(pos, float)
        box = np.asarray(box, float)
        self.N = len(pos)
        self.a = (np.prod(box) / self.N) ** (1 / 3)
        self.box = box
        self.pos = pos
        lists = []
        for i in range(self.N):
            d = pos - pos[i]
            d -= box * np.round(d / box)
            r = np.linalg.norm(d, axis=1)
            idx = np.where((r > 0) & (r <= r_cut))[0]
            idx = idx[np.argsort(r[idx])]
            lists.append((idx, d[idx]))
        M = max(len(l[0]) for l in lists) if max_nbr is None else max_nbr
        self.M = M
        self.nbr = np.tile(np.arange(self.N)[:, None], (1, M)).astype(np.int64)
        self.disp = np.zeros((self.N, M, 3))
        self.mask = np.zeros((self.N, M), bool)
        for i, (idx, d) in enumerate(lists):
            k = min(len(idx), M)
            self.nbr[i, :k] = idx[:k]
            self.disp[i, :k] = d[:k] / self.a
            self.mask[i, :k] = True
        self.dist = np.where(self.mask, np.linalg.norm(self.disp, axis=2), 1.0)
        self.energy = np.zeros(self.N) if energies is None else np.asarray(energies, float)
        self.gamma = np.zeros(self.N)
        self.couplings = couplings
        self.shape = (self.N,)
        self.ijk = np.zeros((self.N, 3), dtype=int)

    def rates(self, T, F_red=(0.0, 0.0, 0.0), kind="ma", nu0=1e12, alpha=10.0,
              lam=0.2, J0=None, **kw):
        F = np.asarray(F_red, float)
        dE = self.energy[self.nbr] - self.energy[:, None] - self.disp @ F
        kT = kB_eV * T
        if kind == "ma":
            W = nu0 * np.exp(-2 * alpha * self.dist) * np.exp(-np.clip(dE / kT, 0, None))
        elif kind in ("marcus", "marcus_graph"):
            J = self.couplings(self.dist * self.a)
            W = (2 * np.pi / hbar_eVs) * J ** 2 / np.sqrt(4 * np.pi * lam * kT) * \
                np.exp(-(dE + lam) ** 2 / (4 * lam * kT))
        else:
            raise ValueError(kind)
        return np.where(self.mask, W, 0.0)
