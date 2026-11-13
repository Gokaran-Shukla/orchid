r"""Full 3D device kinetic Monte Carlo for organic diodes (OLEDs, single-carrier
devices, multilayer stacks), in the spirit of van der Holst *et al.*
(PRB 79, 085203 (2009); 83, 085206 (2011)) and Mesta *et al.*
(Nat. Mater. 12, 652 (2013)).

Model
-----
* Lattice ``Lx x Ly x Lz`` (constant a), periodic in x,y; electrodes are the
  planes z = -1 (anode) and z = Lz (cathode), i.e. separation (Lz+1) a.
* Multilayer stack: every z-plane belongs to a layer with its own HOMO /
  LUMO centres, Gaussian widths and attempt frequencies (hosts, transport
  and blocking layers, or emissive layers).
* Holes hop on HOMO and electrons on LUMO energies with Miller-Abrahams
  rates. A site holds at most one charge. When a carrier hops onto a site
  occupied by the opposite charge an exciton forms, and the pair is removed
  as a recombination event whose z-position is recorded.
* Electrodes inject and extract both species (MA rates, distance up to
  ``r_inj``), with barriers set by the work functions, the applied bias
  (vacuum-level alignment), the image-charge self energy and the Coulomb
  potential of all other charges.
* Electrostatics are split into long range and short range. The long-range
  part is the mean-field potential of charge sheets between grounded
  electrodes (1D Poisson Green's function), refreshed every
  ``mf_update`` events. The short-range part is the exact screened Coulomb
  interaction within ``r_coul``, minus the sheet part of the same pairs.
  Together they converge to the full Coulomb interaction with images as
  ``r_coul`` grows.
* Current from the Ramo-Shockley theorem: every hop contributes
  e q dz / (L A) (all hops, injections, extractions), which gives low-noise
  steady-state currents.
* Event selection: Fenwick tree over carriers and injection sources,
  incremental local updates -> O(log n) per event.

Energies in carrier picture: electron eps = E_L + U(z) - phi, hole
eps = -(E_H + U(z)) + phi, U(z) = (W_a - eV) + (W_c - W_a + eV) z / L the
vacuum level. Reference: cathode Fermi level = 0, anode Fermi level = -eV.
Hence the hole injection barrier is IP - W_a and the electron barrier
W_c - EA, and the built-in voltage is W_a - W_c.
"""
import numpy as np
from ._jit import njit
from ._fenwick import fw_build, fw_add, fw_total, fw_find
from .constants import q, eps0, kB_eV
from .lattice import neighbour_offsets
from .electrostatics import image_self_energy  # noqa


class Layer:
    """One organic layer of the stack (energies eV, thickness in sites)."""

    def __init__(self, n_sites, HOMO=-5.2, LUMO=-2.4, sigma_h=0.075,
                 sigma_e=0.075, nu0_h=1e12, nu0_e=1e12, name="layer"):
        self.__dict__.update(locals())
        del self.__dict__["self"]


class DeviceKMC:
    """3D device KMC. Build once, then call :meth:`run` for each voltage."""

    def __init__(self, layers, Lx=30, Ly=30, a=1.6e-9, T=300.0, W_anode=5.0,
                 W_cathode=2.6, eps_r=3.5, alpha=10.0, rc=np.sqrt(3.0),
                 r_inj=2.0, r_coul=4.0, coulomb="full", image=True, seed=0):
        self.layers = layers
        self.Lz = int(sum(l.n_sites for l in layers))
        self.Lx, self.Ly = int(Lx), int(Ly)
        self.shape = (self.Lx, self.Ly, self.Lz)
        self.N = self.Lx * self.Ly * self.Lz
        self.a, self.T, self.eps_r = a, T, eps_r
        self.Wa, self.Wc = W_anode, W_cathode
        self.alpha, self.rc, self.r_inj, self.r_coul = alpha, rc, r_inj, r_coul
        self.coulomb = coulomb
        rng = np.random.default_rng(seed)
        idx = np.arange(self.N)
        ijk = np.stack(np.unravel_index(idx, self.shape), 1)
        self.z = ijk[:, 2]
        off = neighbour_offsets(rc)
        nb = ijk[:, None, :] + off[None]
        valid = (nb[:, :, 2] >= 0) & (nb[:, :, 2] < self.Lz)
        nb[:, :, 0] %= self.Lx
        nb[:, :, 1] %= self.Ly
        nb[:, :, 2] = np.clip(nb[:, :, 2], 0, self.Lz - 1)
        self.nbr = np.ravel_multi_index(nb.transpose(2, 0, 1), self.shape).astype(np.int64)
        self.valid = valid
        self.off = off
        self.dist = np.sqrt((off ** 2).sum(1))
        # site energies per layer
        EH = np.empty(self.N)
        EL = np.empty(self.N)
        nuh = np.empty(self.N)
        nue = np.empty(self.N)
        z0 = 0
        self.layer_of_z = np.empty(self.Lz, dtype=np.int64)
        for li, l in enumerate(layers):
            m = (self.z >= z0) & (self.z < z0 + l.n_sites)
            EH[m] = l.HOMO + rng.normal(0, 1, m.sum()) * l.sigma_h
            EL[m] = l.LUMO + rng.normal(0, 1, m.sum()) * l.sigma_e
            nuh[m], nue[m] = l.nu0_h, l.nu0_e
            self.layer_of_z[z0:z0 + l.n_sites] = li
            z0 += l.n_sites
        self.EH, self.EL, self.nuh, self.nue = EH, EL, nuh, nue
        self.Ld = (self.Lz + 1) * a                    # electrode separation
        self.zm = (self.z + 1) * a                     # site distance to anode
        self.area = self.Lx * self.Ly * a ** 2
        zz = (np.arange(self.Lz) + 1) * a
        self.Uimg = image_self_energy(zz, self.Ld, eps_r) if image else np.zeros(self.Lz)
        # injection sources: (site, electrode, species)
        src = []
        for e in (0, 1):
            d = (self.z + 1) if e == 0 else (self.Lz - self.z)
            for s_ in np.where(d <= r_inj + 1e-9)[0]:
                for qq in (+1, -1):
                    src.append((s_, e, qq, d[s_]))
        self.src = np.array(src, dtype=float)
        # Coulomb stencil (first M entries = hopping offsets)
        if coulomb == "full":
            extra = [o for o in neighbour_offsets(r_coul) if not any((o == off).all(1))]
            st = np.vstack([off, np.array(extra).reshape(-1, 3)]).astype(np.int64)
        else:
            st = off.astype(np.int64)
        self.stencil = st
        self.Vst = q / (4 * np.pi * eps0 * eps_r * np.sqrt((st ** 2).sum(1)) * a)
        self.upd = neighbour_offsets((r_coul + rc) if coulomb == "full" else rc + r_inj).astype(np.int64)

    # ------------------------------------------------------------------
    def run(self, V, n_eq=200000, n_meas=1000000, mf_update=2000, max_carriers=None,
            seed=1):
        """Simulate at applied bias V (anode positive). Returns dict with
        J [A/m^2] (Ramo), recombination rate, emission and density profiles."""
        kT = kB_eV * self.T
        L = self.Ld
        U = (self.Wa - V) + (self.Wc - self.Wa + V) * self.zm / L
        if max_carriers is None:
            max_carriers = max(2000, self.N // 5)
        mode = {"none": 0, "meanfield": 1, "full": 2}[self.coulomb]
        out = _run_device(self.nbr, self.valid, self.off.astype(np.int64), self.dist,
                          self.z, self.EH, self.EL, self.nuh, self.nue, U,
                          self.Uimg, self.src, np.array(self.shape, dtype=np.int64),
                          self.stencil, self.Vst, self.upd, kT, self.alpha,
                          -V, 0.0, L, self.area,
                          self.eps_r, self.a, mode, int(n_eq), int(n_meas),
                          int(mf_update), int(max_carriers), int(seed),
                          float(self.r_inj))
        dzsum, tmeas, nrec, recz, pz, nz, nevents, ninj = out
        J = q * dzsum * self.a / (L * self.area * tmeas)
        Vcell = self.Lx * self.Ly * self.a ** 3
        return dict(V=V, J=J, rec_rate=nrec / (tmeas * self.area),
                    gamma_b=(q * nrec / (tmeas * self.area)) / J if J > 0 else np.nan,
                    rec_profile=recz / max(recz.sum(), 1), p_z=pz / (tmeas * Vcell),
                    n_z=nz / (tmeas * Vcell), z=(np.arange(self.Lz) + 1) * self.a,
                    events=nevents, t=tmeas, injections=ninj)

    def jv(self, Vs, **kw):
        return [self.run(V, **kw) for V in Vs]


@njit(cache=True)
def _sheet(z1, z2, L, area, eps):
    zl = min(z1, z2)
    zg = max(z1, z2)
    return 1.602176634e-19 * zl * (L - zg) / (eps * area * L)


@njit(cache=True)
def _sidx(ix, iy, iz, Lx, Ly, Lz):
    return ((ix % Lx) * Ly + (iy % Ly)) * Lz + iz


@njit(cache=True)
def _kill(c, tree, tot, cpos, cq, occ, who, free, nfree):
    """Remove carrier c; returns new nfree."""
    fw_add(tree, c, -tot[c])
    tot[c] = 0.0
    _remove(c, cpos, cq, occ, who, free, nfree)
    return nfree + 1


@njit(cache=True)
def _charge_change(site, dq, mode, phi_loc, phi_lay, qlay, cnt_h, cnt_e, sgn,
                   stencil, Vst, z, zmz, Lx, Ly, Lz, L, area, eps):
    """Add charge dq (+1/-1) at ``site``: update sheet and local potentials.
    ``sgn`` = +1 adds a carrier of charge dq, -1 removes it (bookkeeping)."""
    iz = z[site]
    qlay[iz] += dq
    if mode >= 1:
        for i in range(Lz):
            phi_lay[i] += dq * _sheet(zmz[i], zmz[iz], L, area, eps)
    if mode == 2:
        _stencil_add(site, dq, phi_loc, stencil, Vst, z, zmz, Lx, Ly, Lz, L, area, eps)
    qq = dq * sgn
    if qq > 0:
        cnt_h[iz] += sgn
    else:
        cnt_e[iz] += sgn


@njit(cache=True)
def _run_device(nbr, valid, off, dist, z, EH, EL, nuh, nue, U, Uimg, src, shape,
                stencil, Vst, upd, kT, alpha, EFa, EFc, L, area, eps_r, a, mode,
                n_eq, n_meas, mf_update, cmax, seed, r_inj):
    np.random.seed(seed)
    N, M = nbr.shape
    Lx, Ly, Lz = shape[0], shape[1], shape[2]
    eps = 8.8541878128e-12 * eps_r
    nsrc = src.shape[0]
    occ = np.zeros(N, dtype=np.int64)      # +1 hole, -1 electron, 0 empty
    who = -np.ones(N, dtype=np.int64)
    cpos = -np.ones(cmax, dtype=np.int64)
    cq = np.zeros(cmax, dtype=np.int64)
    free = np.arange(cmax - 1, -1, -1).astype(np.int64)
    nfree = cmax
    K = M + 2                               # hops + extraction to each electrode
    crate = np.zeros((cmax, K))
    tot = np.zeros(cmax + nsrc)
    tree = np.zeros(cmax + nsrc)
    phi_loc = np.zeros(N)
    phi_lay = np.zeros(Lz)
    qlay = np.zeros(Lz)
    cnt_h = np.zeros(Lz)
    cnt_e = np.zeros(Lz)
    zmz = (np.arange(Lz) + 1.0) * a
    src_of_site = -np.ones((N, 4), dtype=np.int64)
    for k in range(nsrc):
        s = int(src[k, 0])
        for slot in range(4):
            if src_of_site[s, slot] < 0:
                src_of_site[s, slot] = k
                break
    dzsum = 0.0
    t = 0.0
    tm0 = 0.0
    nrec = 0
    recz = np.zeros(Lz)
    pz = np.zeros(Lz)
    nzz = np.zeros(Lz)
    ninj = 0
    measuring = False
    for ev in range(n_eq + n_meas):
        if ev == n_eq:
            measuring = True
            tm0 = t
            dzsum = 0.0
            nrec = 0
            ninj = 0
            recz[:] = 0.0
            pz[:] = 0.0
            nzz[:] = 0.0
        if ev % mf_update == 0:
            for c in range(cmax):
                if cpos[c] >= 0:
                    tot[c] = _crow(c, cpos, cq, nbr, valid, dist, z, EH, EL, nuh, nue,
                                   U, Uimg, occ, phi_loc, phi_lay, Vst, kT, alpha,
                                   EFa, EFc, Lz, L, area, eps, a, mode, crate, r_inj)
                else:
                    tot[c] = 0.0
            for k in range(nsrc):
                tot[cmax + k] = _srate(k, src, occ, nuh, nue, EH, EL, U, Uimg, z,
                                       phi_loc, phi_lay, kT, alpha, EFa, EFc)
            fw_build(tree, tot)
        R = fw_total(tree)
        if R <= 0:
            break
        dt = -np.log(1.0 - np.random.random()) / R
        if measuring:
            for i in range(Lz):
                pz[i] += cnt_h[i] * dt
                nzz[i] += cnt_e[i] * dt
        t += dt
        e = fw_find(tree, np.random.random() * R)
        if tot[e] <= 0:
            continue
        s_old = -1
        s_new = -1
        if e < cmax:
            c = e
            u2 = np.random.random() * tot[c]
            m = 0
            acc = crate[c, 0]
            while acc < u2 and m < K - 1:
                m += 1
                acc += crate[c, m]
            s = cpos[c]
            qq = cq[c]
            s_old = s
            if m < M:
                j = nbr[s, m]
                dzsum += qq * (z[j] - z[s])
                if occ[j] == -qq:            # exciton formation -> recombination
                    c2 = who[j]
                    nfree = _kill(c, tree, tot, cpos, cq, occ, who, free, nfree)
                    nfree = _kill(c2, tree, tot, cpos, cq, occ, who, free, nfree)
                    _charge_change(s, -qq, mode, phi_loc, phi_lay, qlay, cnt_h, cnt_e,
                                   -1, stencil, Vst, z, zmz, Lx, Ly, Lz, L, area, eps)
                    _charge_change(j, qq, mode, phi_loc, phi_lay, qlay, cnt_h, cnt_e,
                                   -1, stencil, Vst, z, zmz, Lx, Ly, Lz, L, area, eps)
                    s_new = j
                    if measuring:
                        nrec += 1
                        recz[z[j]] += 1
                else:
                    occ[s] = 0
                    who[s] = -1
                    occ[j] = qq
                    who[j] = c
                    cpos[c] = j
                    _charge_change(s, -qq, mode, phi_loc, phi_lay, qlay, cnt_h, cnt_e,
                                   -1, stencil, Vst, z, zmz, Lx, Ly, Lz, L, area, eps)
                    _charge_change(j, qq, mode, phi_loc, phi_lay, qlay, cnt_h, cnt_e,
                                   1, stencil, Vst, z, zmz, Lx, Ly, Lz, L, area, eps)
                    s_new = j
            else:
                if m == M:
                    dzsum += qq * (-(z[s] + 1))
                else:
                    dzsum += qq * (Lz - z[s])
                nfree = _kill(c, tree, tot, cpos, cq, occ, who, free, nfree)
                _charge_change(s, -qq, mode, phi_loc, phi_lay, qlay, cnt_h, cnt_e,
                               -1, stencil, Vst, z, zmz, Lx, Ly, Lz, L, area, eps)
        else:
            k = e - cmax
            s = int(src[k, 0])
            el = int(src[k, 1])
            qq = int(src[k, 2])
            if occ[s] != 0 or nfree == 0:
                continue
            nfree -= 1
            c = free[nfree]
            cpos[c] = s
            cq[c] = qq
            occ[s] = qq
            who[s] = c
            if el == 0:
                dzsum += qq * (z[s] + 1)
            else:
                dzsum += qq * (-(Lz - z[s]))
            _charge_change(s, qq, mode, phi_loc, phi_lay, qlay, cnt_h, cnt_e,
                           1, stencil, Vst, z, zmz, Lx, Ly, Lz, L, area, eps)
            s_new = s
            if measuring:
                ninj += 1
        # incremental local updates around the touched sites
        for side in range(2):
            site = s_old if side == 0 else s_new
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
                if jz < 0 or jz >= Lz:
                    continue
                st_ = _sidx(jx, jy, jz, Lx, Ly, Lz)
                c3 = who[st_]
                if c3 >= 0:
                    new = _crow(c3, cpos, cq, nbr, valid, dist, z, EH, EL, nuh, nue,
                                U, Uimg, occ, phi_loc, phi_lay, Vst, kT, alpha,
                                EFa, EFc, Lz, L, area, eps, a, mode, crate, r_inj)
                    fw_add(tree, c3, new - tot[c3])
                    tot[c3] = new
                for slot in range(4):
                    k2 = src_of_site[st_, slot]
                    if k2 >= 0:
                        new = _srate(k2, src, occ, nuh, nue, EH, EL, U, Uimg, z,
                                     phi_loc, phi_lay, kT, alpha, EFa, EFc)
                        fw_add(tree, cmax + k2, new - tot[cmax + k2])
                        tot[cmax + k2] = new
    return dzsum, t - tm0, nrec, recz, pz, nzz, n_eq + n_meas, ninj


@njit(cache=True)
def _remove(c, cpos, cq, occ, who, free, nfree):
    s = cpos[c]
    occ[s] = 0
    who[s] = -1
    cpos[c] = -1
    cq[c] = 0
    free[nfree] = c


@njit(cache=True)
def _stencil_add(s, qq, phi_loc, stencil, Vst, z, zmz, Lx, Ly, Lz, L, area, eps):
    ix = s // (Ly * Lz)
    iy = (s // Lz) % Ly
    iz = s % Lz
    for k in range(stencil.shape[0]):
        jz = iz + stencil[k, 2]
        if jz < 0 or jz >= Lz:
            continue
        j = _sidx(ix + stencil[k, 0], iy + stencil[k, 1], jz, Lx, Ly, Lz)
        phi_loc[j] += qq * (Vst[k] - _sheet(zmz[jz], zmz[iz], L, area, eps))


@njit(cache=True)
def _crow(c, cpos, cq, nbr, valid, dist, z, EH, EL, nuh, nue, U, Uimg, occ,
          phi_loc, phi_lay, Vst, kT, alpha, EFa, EFc, Lz, L, area, eps, a, mode,
          crate, r_inj):
    s = cpos[c]
    qq = cq[c]
    M = nbr.shape[1]
    zs = (z[s] + 1.0) * a
    # own (self) contributions carry the carrier's own sign qq
    self_s = qq * _sheet(zs, zs, L, area, eps) if mode >= 1 else 0.0
    phis = phi_loc[s] + phi_lay[z[s]] - self_s
    if qq > 0:
        Es = -(EH[s] + U[s]) + phis + Uimg[z[s]]
        nu = nuh[s]
    else:
        Es = EL[s] + U[s] - phis + Uimg[z[s]]
        nu = nue[s]
    tot = 0.0
    for m in range(M):
        if not valid[s, m]:
            crate[c, m] = 0.0
            continue
        j = nbr[s, m]
        if occ[j] == qq:
            crate[c, m] = 0.0
            continue
        w = nu * np.exp(-2 * alpha * dist[m])
        if occ[j] == -qq:                 # exciton formation (strongly downhill)
            crate[c, m] = w
            tot += w
            continue
        # own contribution to the potential at j: exact Coulomb (full) or sheet
        if mode == 2:
            selfj = qq * Vst[m]
        elif mode == 1:
            selfj = qq * _sheet((z[j] + 1.0) * a, zs, L, area, eps)
        else:
            selfj = 0.0
        phij = phi_loc[j] + phi_lay[z[j]] - selfj
        if qq > 0:
            Ej = -(EH[j] + U[j]) + phij + Uimg[z[j]]
        else:
            Ej = EL[j] + U[j] - phij + Uimg[z[j]]
        dE = Ej - Es
        if dE > 0:
            w *= np.exp(-dE / kT)
        crate[c, m] = w
        tot += w
    for el in range(2):
        d = (z[s] + 1.0) if el == 0 else (Lz - z[s] * 1.0)
        if d > r_inj + 1e-9:
            crate[c, M + el] = 0.0
            continue
        EF = EFa if el == 0 else EFc
        mu = -EF if qq > 0 else EF
        dE = mu - Es
        w = nu * np.exp(-2 * alpha * d)
        if dE > 0:
            w *= np.exp(-dE / kT)
        crate[c, M + el] = w
        tot += w
    return tot


@njit(cache=True)
def _srate(k, src, occ, nuh, nue, EH, EL, U, Uimg, z, phi_loc, phi_lay, kT,
           alpha, EFa, EFc):
    s = int(src[k, 0])
    if occ[s] != 0:
        return 0.0
    el = int(src[k, 1])
    qq = int(src[k, 2])
    d = src[k, 3]
    phis = phi_loc[s] + phi_lay[z[s]]
    EF = EFa if el == 0 else EFc
    if qq > 0:
        E = -(EH[s] + U[s]) + phis + Uimg[z[s]]
        mu = -EF
        nu = nuh[s]
    else:
        E = EL[s] + U[s] - phis + Uimg[z[s]]
        mu = EF
        nu = nue[s]
    dE = E - mu
    w = nu * np.exp(-2 * alpha * d)
    if dE > 0:
        w *= np.exp(-dE / kT)
    return w
