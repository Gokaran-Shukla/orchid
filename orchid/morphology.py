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


# ------------------------------------------------- MD trajectory import (R26)
_MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "S": 32.06,
         "P": 30.974, "Cl": 35.45, "Br": 79.904, "I": 126.90, "Si": 28.085, "B": 10.81,
         "Se": 78.97, "Ir": 192.22, "Pt": 195.08, "Zn": 65.38, "Cu": 63.546, "Al": 26.982}


def element_mass(name):
    """Atomic mass from an atom or element name ('C', 'C12', 'CA', 'Cl1', 'Ir').

    A two-letter element is recognised only in mixed case ('Cl', 'Ir'),
    following the PDB/GROMACS convention that upper-case 'CA' is an alpha
    carbon. Otherwise the first letter decides. Pass explicit ``masses`` to
    :func:`molecule_centres` for anything unusual.
    """
    letters = "".join(ch for ch in name if ch.isalpha())
    if len(letters) >= 2 and letters[0].isupper() and letters[1].islower() \
            and letters[:2] in _MASS:
        return _MASS[letters[:2]]
    return _MASS.get(letters[:1].upper(), 12.0)


def read_gro(path):
    """Read a (multi-frame) GROMACS .gro file.

    Returns a list of frames. Each frame is a dict with keys
    ``resid`` (int array), ``resname``, ``atom`` (lists),
    ``xyz`` (natoms x 3, metres), ``box`` (3, metres; orthorhombic
    diagonal) and ``title``.
    """
    lines = open(path).read().splitlines()
    frames, i = [], 0
    while i < len(lines) and lines[i].strip():
        title = lines[i]
        n = int(lines[i + 1])
        resid, resname, atom, xyz = [], [], [], []
        for l in lines[i + 2:i + 2 + n]:
            resid.append(int(l[0:5]))
            resname.append(l[5:10].strip())
            atom.append(l[10:15].strip())
            xyz.append([float(l[20:28]), float(l[28:36]), float(l[36:44])])
        box = np.array([float(v) for v in lines[i + 2 + n].split()[:3]])
        frames.append(dict(title=title, resid=np.array(resid), resname=resname, atom=atom,
                           xyz=np.array(xyz) * 1e-9, box=box * 1e-9))
        i += n + 3
    return frames


def write_gro(path, frames):
    """Write frames (same dict layout as :func:`read_gro`) to a .gro file."""
    with open(path, "w") as f:
        for fr in frames:
            f.write((fr.get("title") or "ORCHID") + "\n")
            f.write(f"{len(fr['xyz']):5d}\n")
            for k, (r, rn, an, p) in enumerate(zip(fr["resid"], fr["resname"], fr["atom"],
                                                   fr["xyz"] * 1e9)):
                f.write(f"{r % 100000:5d}{rn:<5s}{an:>5s}{(k + 1) % 100000:5d}"
                        f"{p[0]:8.3f}{p[1]:8.3f}{p[2]:8.3f}\n")
            b = fr["box"] * 1e9
            f.write(f"{b[0]:10.5f}{b[1]:10.5f}{b[2]:10.5f}\n")


def read_xyz_trajectory(path, box=None):
    """Multi-frame (extended) XYZ. ``box`` [m] is taken from an extended-XYZ
    ``Lattice="ax 0 0 0 by 0 0 0 cz"`` comment if present (Angstrom), else
    from the argument. Returns frames as in :func:`read_gro`, with
    ``resid`` set to None (give molecule sizes to :func:`molecule_centres`)."""
    lines = open(path).read().splitlines()
    frames, i = [], 0
    while i < len(lines) and lines[i].strip():
        n = int(lines[i])
        com = lines[i + 1]
        b = None if box is None else np.asarray(box, float)
        if "Lattice=" in com:
            v = [float(x) for x in com.split("Lattice=")[1].split('"')[1].split()]
            b = np.array([v[0], v[4], v[8]]) * 1e-10
        at, xyz = [], []
        for l in lines[i + 2:i + 2 + n]:
            p = l.split()
            at.append(p[0])
            xyz.append([float(v) for v in p[1:4]])
        frames.append(dict(title=com, resid=None, resname=None, atom=at,
                           xyz=np.array(xyz) * 1e-10, box=b))
        i += n + 2
    return frames


def molecule_centres(frame, atoms_per_molecule=None, mass_weighted=True, masses=None):
    """Centres of mass of molecules in one frame, made whole across PBC.

    Molecules are defined by ``frame['resid']`` (GRO). If that is None,
    ``atoms_per_molecule`` consecutive atoms form one molecule. Each atom is
    unwrapped by the minimum image relative to the first atom of its
    molecule, so molecules split by the periodic boundary are rebuilt
    before averaging. The centres are wrapped back into [0, box).
    Returns (centres [m] (n_mol x 3), molecule index per atom).
    """
    xyz = np.asarray(frame["xyz"], float)
    box = frame["box"]
    if frame.get("resid") is not None:
        rid = np.asarray(frame["resid"])
        change = np.r_[True, rid[1:] != rid[:-1]]
        mol = np.cumsum(change) - 1
    else:
        if atoms_per_molecule is None:
            raise ValueError("atoms_per_molecule needed when resid is absent")
        mol = np.arange(len(xyz)) // atoms_per_molecule
    nm = mol.max() + 1
    first = np.searchsorted(mol, np.arange(nm))
    ref = xyz[first][mol]
    d = xyz - ref
    if box is not None:
        d -= box * np.round(d / box)
    if masses is not None:
        w = np.asarray(masses, float)
    elif mass_weighted:
        w = np.array([element_mass(a) for a in frame["atom"]])
    else:
        w = np.ones(len(xyz))
    W = np.bincount(mol, w, nm)
    com = np.stack([np.bincount(mol, w * d[:, k], nm) for k in range(3)], 1) / W[:, None]
    com += xyz[first]
    if box is not None:
        com %= box
    return com, mol


def trajectory_mobility(frames, T, n_frac=1e-6, r_cut=None, energies="dipolar",
                        atoms_per_molecule=None, alpha=10.0, dipole_debye=3.0, eps_r=3.0,
                        sigma=None, seed=0, **me_kw):
    """Charge mobility averaged over MD snapshots (static-disorder limit).

    For each frame: molecular centres -> :class:`MolecularGraph` ->
    linear-response master equation. Site energies are ``"dipolar"``
    (:func:`dipolar_site_energies`; correlated, they change with the
    morphology), ``"gaussian"`` (uncorrelated, width ``sigma``) or an
    array. Returns dict(mu_frames, mu_mean, mu_sem, n_molecules, a, sigma_frames).
    """
    from .master_equation import me_linear_response
    mus, sig = [], []
    rng = np.random.default_rng(seed)
    for k, fr in enumerate(frames):
        c, _ = molecule_centres(fr, atoms_per_molecule)
        box = fr["box"]
        a = (np.prod(box) / len(c)) ** (1 / 3)
        rc = 1.7 * a if r_cut is None else r_cut
        if isinstance(energies, str) and energies == "dipolar":
            E = dipolar_site_energies(c, box, dipole_debye=dipole_debye, eps_r=eps_r,
                                      seed=seed + k)
        elif isinstance(energies, str) and energies == "gaussian":
            E = rng.normal(0.0, sigma, len(c))
        else:
            E = np.asarray(energies, float)
        G = MolecularGraph(c, box, r_cut=rc, energies=E)
        mus.append(me_linear_response(G, T, n_frac, alpha=alpha, **me_kw)[0])
        sig.append(float(np.std(E)))
    mus = np.asarray(mus)
    sem = mus.std(ddof=1) / np.sqrt(len(mus)) if len(mus) > 1 else 0.0
    return dict(mu_frames=mus, mu_mean=mus.mean(), mu_sem=sem, n_molecules=len(c),
                a=a, sigma_frames=np.asarray(sig))
