r"""Exciton kinetic Monte Carlo inside a device (roadmap R16).

Excitons are generated where the 3D device KMC records recombination
events, at the rate it reports. They then hop (Förster or Dexter), decay
radiatively or non-radiatively, are quenched near the metal electrodes,
annihilate (triplet-triplet or singlet-singlet, TTA), and are quenched by
polarons (TPQ / SPQ) whose positions are drawn from the device KMC charge
profiles.

Events per exciton i (BKL selection, Fenwick tree, local updates):

* hop     :math:`k_{ij}=k_d(\Gx{R_0}/r_{ij})^6\,b(\Delta E)` (or Dexter)
* decay   :math:`k_r+k_{nr}+k_{m}(z)`, with :math:`k_m=k_{m0}(d_0/z)^3` (dipole to metal)
* TPQ     :math:`k_q(i)=\sum_{\rm polarons}k_{q0}(R_q/r)^6`
* TTA     :math:`k_{TT}(r)=k_{TT0}(R_{TT}/r)^6` for every exciton pair within
  the stencil; one partner is lost (T+T -> T + S0).

The radiative yield :math:`\eta=N_{rad}/N_{gen}` multiplies the charge
balance from the device KMC:
:math:`\mathrm{EQE}=\gamma_b\,\eta_{S/T}\,\eta\,\eta_{out}`.

Checks: without quenching or annihilation :math:`\eta=k_r/(k_r+k_{nr})`
exactly. In a homogeneous box the steady-state density follows the
bimolecular law :math:`G=n/\tau+\gamma_{TT}n^2`.
"""
import numpy as np
from ._jit import njit
from ._fenwick import fw_build, fw_add, fw_total, fw_find
from .lattice import neighbour_offsets
from .constants import kB_eV


@njit(cache=True)
def _sidx(ix, iy, iz, Lx, Ly, Lz, pz):
    if pz:
        iz = iz % Lz
    return ((ix % Lx) * Ly + (iy % Ly)) * Lz + iz


@njit(cache=True)
def _xrow(c, pos, occ, nbrE, hop, kdec, kq, st, kTT, shape, pz, rates):
    """rates[c, :M] hops, [M] decay, [M+1 ...] TTA with partners in stencil."""
    s = pos[c]
    M = nbrE.shape[1]
    Lx, Ly, Lz = shape[0], shape[1], shape[2]
    tot = 0.0
    for m in range(M):
        j = nbrE[s, m]
        if j < 0:
            rates[c, m] = 0.0
            continue
        w = hop[s, m]
        rates[c, m] = w
        tot += w
    rates[c, M] = kdec[s] + kq[s]
    tot += rates[c, M]
    ix = s // (Ly * Lz)
    iy = (s // Lz) % Ly
    iz = s % Lz
    for k in range(st.shape[0]):
        jz = iz + st[k, 2]
        if not pz and (jz < 0 or jz >= Lz):
            rates[c, M + 1 + k] = 0.0
            continue
        j = _sidx(ix + st[k, 0], iy + st[k, 1], jz, Lx, Ly, Lz, pz)
        if occ[j] >= 0:
            w = 0.5 * kTT[k]          # pair rate split between both partners
            rates[c, M + 1 + k] = w
            tot += w
        else:
            rates[c, M + 1 + k] = 0.0
    return tot


@njit(cache=True)
def _run_x(nbrE, hop, kdec, krad, kq, st, kTT, upd, shape, pz, gen_cdf, G_tot,
           n_events, n_eq, cmax, seed):
    np.random.seed(seed)
    N, M = nbrE.shape
    Lx, Ly, Lz = shape[0], shape[1], shape[2]
    K = M + 1 + st.shape[0]
    occ = -np.ones(N, dtype=np.int64)
    pos = -np.ones(cmax, dtype=np.int64)
    free = np.arange(cmax - 1, -1, -1).astype(np.int64)
    nfree = cmax
    rates = np.zeros((cmax, K))
    tot = np.zeros(cmax + 1)
    tree = np.zeros(cmax + 1)
    tot[cmax] = G_tot
    fw_build(tree, tot)
    t = 0.0
    t0 = 0.0
    ngen = 0
    nrad = 0
    nnr = 0
    nq = 0
    ntt = 0
    area_x = 0.0
    nx = 0
    meas = False
    for ev in range(n_eq + n_events):
        if ev == n_eq:
            meas = True
            t0 = t
            ngen = 0
            nrad = 0
            nnr = 0
            nq = 0
            ntt = 0
            area_x = 0.0
        if ev % 50000 == 0:
            fw_build(tree, tot)
        R = fw_total(tree)
        if R <= 0:
            break
        dt = -np.log(1.0 - np.random.random()) / R
        if meas:
            area_x += nx * dt
        t += dt
        e = fw_find(tree, np.random.random() * R)
        if tot[e] <= 0:
            continue
        touched0 = -1
        touched1 = -1
        if e == cmax:                                    # generation
            u = np.random.random()
            lo = 0
            hi = gen_cdf.shape[0] - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if gen_cdf[mid] < u:
                    lo = mid + 1
                else:
                    hi = mid
            iz = lo
            ix = np.random.randint(Lx)
            iy = np.random.randint(Ly)
            s = (ix * Ly + iy) * Lz + iz
            if occ[s] >= 0 or nfree == 0:
                # generation onto an occupied site = immediate annihilation
                if meas:
                    ngen += 1
                    ntt += 1
                continue
            nfree -= 1
            c = free[nfree]
            pos[c] = s
            occ[s] = c
            nx += 1
            touched0 = s
            if meas:
                ngen += 1
        else:
            c = e
            u2 = np.random.random() * tot[c]
            m = 0
            acc = rates[c, 0]
            while acc < u2 and m < K - 1:
                m += 1
                acc += rates[c, m]
            s = pos[c]
            touched0 = s
            if m < M:                                    # hop (onto exciton -> TTA)
                j = nbrE[s, m]
                if occ[j] >= 0:
                    occ[s] = -1
                    pos[c] = -1
                    fw_add(tree, c, -tot[c])
                    tot[c] = 0.0
                    free[nfree] = c
                    nfree += 1
                    nx -= 1
                    if meas:
                        ntt += 1
                else:
                    occ[s] = -1
                    occ[j] = c
                    pos[c] = j
                    touched1 = j
            elif m == M:                                 # decay or polaron quench
                kk = kdec[s] + kq[s]
                r_ = np.random.random() * kk
                if meas:
                    if r_ < krad[s]:
                        nrad += 1
                    elif r_ < kdec[s]:
                        nnr += 1
                    else:
                        nq += 1
                occ[s] = -1
                pos[c] = -1
                fw_add(tree, c, -tot[c])
                tot[c] = 0.0
                free[nfree] = c
                nfree += 1
                nx -= 1
            else:                                        # TTA with partner -> partner lost
                k = m - M - 1
                ix = s // (Ly * Lz)
                iy = (s // Lz) % Ly
                iz = s % Lz
                j = _sidx(ix + st[k, 0], iy + st[k, 1], iz + st[k, 2], Lx, Ly, Lz, pz)
                c2 = occ[j]
                if c2 >= 0:
                    occ[j] = -1
                    pos[c2] = -1
                    fw_add(tree, c2, -tot[c2])
                    tot[c2] = 0.0
                    free[nfree] = c2
                    nfree += 1
                    nx -= 1
                    touched1 = j
                    if meas:
                        ntt += 1
        for side in range(2):
            site = touched0 if side == 0 else touched1
            if site < 0:
                continue
            ix = site // (Ly * Lz)
            iy = (site // Lz) % Ly
            iz = site % Lz
            for kk in range(-1, upd.shape[0]):
                if kk < 0:
                    jx, jy, jz = ix, iy, iz
                else:
                    jx = ix + upd[kk, 0]
                    jy = iy + upd[kk, 1]
                    jz = iz + upd[kk, 2]
                if not pz and (jz < 0 or jz >= Lz):
                    continue
                j = _sidx(jx, jy, jz, Lx, Ly, Lz, pz)
                c3 = occ[j]
                if c3 >= 0:
                    new = _xrow(c3, pos, occ, nbrE, hop, kdec, kq, st, kTT, shape, pz, rates)
                    fw_add(tree, c3, new - tot[c3])
                    tot[c3] = new
    return t - t0, ngen, nrad, nnr, nq, ntt, area_x


class ExcitonDeviceKMC:
    """Exciton KMC on the lattice of a :class:`~orchid.device_kmc.DeviceKMC`
    (or a stand-alone periodic box if ``device`` is None).

    Per-layer parameters may be scalars or arrays of length Lz:
    tau_r, tau_nr [s], R0 (Förster hopping radius, sites), sigma_ex [eV].
    """

    def __init__(self, shape, a=1.6e-9, T=300.0, tau_r=1e-6, tau_nr=1e-5, R0=1.5,
                 sigma_ex=0.0, rc=2.0, k_TT0=0.0, R_TT=2.0, r_TT=3.0,
                 k_metal0=0.0, d_metal=1.0, periodic_z=False, seed=0):
        self.shape = tuple(int(s) for s in shape)
        Lx, Ly, Lz = self.shape
        self.N = Lx * Ly * Lz
        self.a, self.T, self.pz = a, T, bool(periodic_z)
        rng = np.random.default_rng(seed)
        z = np.arange(self.N) % Lz
        f = lambda v: np.broadcast_to(np.asarray(v, float), (Lz,))[z]
        self.krad = 1.0 / f(tau_r)
        knr = 1.0 / f(tau_nr)
        if k_metal0 > 0 and not periodic_z:
            zz = z + 1.0
            km = k_metal0 * ((d_metal / zz) ** 3 + (d_metal / (Lz - z)) ** 3)
        else:
            km = 0.0
        self.kdec = self.krad + knr + km
        self.E = f(sigma_ex) * rng.normal(size=self.N)
        off = neighbour_offsets(rc)
        ijk = np.stack(np.unravel_index(np.arange(self.N), self.shape), 1)
        nb = ijk[:, None, :] + off[None]
        valid = (nb[:, :, 2] >= 0) & (nb[:, :, 2] < Lz) | self.pz
        nb[:, :, 0] %= Lx
        nb[:, :, 1] %= Ly
        nb[:, :, 2] = nb[:, :, 2] % Lz if self.pz else np.clip(nb[:, :, 2], 0, Lz - 1)
        nbr = np.ravel_multi_index(nb.transpose(2, 0, 1), self.shape).astype(np.int64)
        nbr[~valid] = -1
        r = np.sqrt((off ** 2).sum(1))
        kd = 1.0 / f(tau_r) + knr                       # Förster uses donor lifetime
        R0a = f(R0)
        dE = self.E[np.clip(nbr, 0, None)] - self.E[:, None]
        up = np.exp(-np.clip(dE / (kB_eV * T), 0, None))
        self.hop = kd[:, None] * (R0a[:, None] / r[None]) ** 6 * up
        self.hop[~valid] = 0.0
        self.nbrE = nbr
        self.st = neighbour_offsets(r_TT).astype(np.int64)
        rs = np.sqrt((self.st ** 2).sum(1))
        self.kTT = k_TT0 * (R_TT / rs) ** 6
        self.upd = neighbour_offsets(max(rc, r_TT)).astype(np.int64)
        self.kq = np.zeros(self.N)

    def set_polarons(self, n_z, p_z, k_q0, R_q=1.5, r_q=3.0, seed=1):
        """Freeze a polaron configuration drawn from density profiles
        (occupation per site, length Lz) and build the TPQ rate field."""
        Lx, Ly, Lz = self.shape
        rng = np.random.default_rng(seed)
        occ = (rng.random(self.N) < np.asarray(n_z)[np.arange(self.N) % Lz] +
               np.asarray(p_z)[np.arange(self.N) % Lz])
        kq = np.zeros(self.N)
        st = neighbour_offsets(r_q)
        rs = np.sqrt((st ** 2).sum(1))
        w = k_q0 * (R_q / rs) ** 6
        ijk = np.stack(np.unravel_index(np.where(occ)[0], self.shape), 1)
        for o, ww in zip(st, w):
            t = ijk + o
            m = (t[:, 2] >= 0) & (t[:, 2] < Lz) if not self.pz else np.ones(len(t), bool)
            t = t[m]
            t[:, 0] %= Lx
            t[:, 1] %= Ly
            t[:, 2] %= Lz
            np.add.at(kq, np.ravel_multi_index(t.T, self.shape), ww)
        kq[occ] += k_q0 * 10.0                          # exciton on a polaron site
        self.kq = kq
        return self

    def run(self, G_volume, gen_profile=None, n_events=400000, n_eq=100000, seed=3):
        """Steady state at volumetric exciton generation rate G [m^-3 s^-1].

        gen_profile : length-Lz weights (e.g. DeviceKMC rec_profile).
        Returns dict(eta_rad, frac_nr, frac_quench, frac_TTA, n_ex [m^-3]).
        """
        Lx, Ly, Lz = self.shape
        prof = np.ones(Lz) if gen_profile is None else np.asarray(gen_profile, float)
        cdf = np.cumsum(prof) / prof.sum()
        Vbox = self.N * self.a ** 3
        G_tot = G_volume * Vbox
        cmax = max(1000, self.N // 2)
        t, ng, nr, nnr, nq, ntt, ax = _run_x(self.nbrE, self.hop, self.kdec, self.krad,
                                              self.kq, self.st, self.kTT, self.upd,
                                              np.array(self.shape, dtype=np.int64), self.pz,
                                              cdf, G_tot, int(n_events), int(n_eq), int(cmax),
                                              int(seed))
        ng = max(ng, 1)
        return dict(eta_rad=nr / ng, frac_nr=nnr / ng, frac_quench=nq / ng, frac_TTA=ntt / ng,
                    n_ex=ax / (t * Vbox), t=t, generated=ng)


def oled_eqe_from_device(dev_result, exciton_kmc, eta_ST=1.0, eta_out=0.2, **kw):
    """Couple a DeviceKMC.run() result to the exciton KMC."""
    G = dev_result["rec_rate"] / (exciton_kmc.shape[2] * exciton_kmc.a)    # m^-3 s^-1
    r = exciton_kmc.run(G, gen_profile=dev_result["rec_profile"], **kw)
    r["EQE"] = dev_result["gamma_b"] * eta_ST * r["eta_rad"] * eta_out
    r["J"] = dev_result["J"]
    return r
