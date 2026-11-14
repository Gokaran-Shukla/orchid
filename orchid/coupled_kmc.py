r"""Concurrent charge + exciton device KMC (roadmap R24, v0.4).

In v0.3 the OLED pipeline was sequential: :class:`~orchid.device_kmc.DeviceKMC`
produced recombination and polaron density profiles, then
:class:`~orchid.exciton_device.ExcitonDeviceKMC` ran on a *frozen* polaron
configuration drawn from those profiles. Spatial and temporal correlations
were lost. For example, excitons are born exactly where polarons meet, and
polarons drift past excitons while those excitons are alive.

This module puts holes, electrons and excitons in **one event loop** with
one clock:

* charges: hops, injection and extraction, Coulomb interaction, image
  charges and the Ramo current, exactly as in ``DeviceKMC``.
* recombination: when a carrier hops onto the opposite charge, both are
  removed and an exciton is created on that site.
* excitons: Förster hopping (with energetic disorder), radiative and
  non-radiative decay, metal quenching, and contact plus Förster-type
  TTA. **TPQ is evaluated against the live polaron positions**:
  rate k_q0 (R_q/r)^6 for each polaron inside the stencil, plus an onsite term.
* bookkeeping: two Fenwick trees (charges + sources, excitons) and a
  common total rate R = R_c + R_x. When a charge moves, the TPQ rates of
  nearby excitons are refreshed incrementally. When an exciton moves, the
  TTA rates of nearby excitons are refreshed.

Outputs: J, gamma_b, the exciton fate fractions (radiative, non-radiative,
TPQ, TTA), the emission profile, and EQE = eta_ST eta_out N_photon / N_electron.
N_electron is counted from the Ramo current, so EQE needs no separate gamma_b factor.
"""
import numpy as np
from ._jit import njit
from ._fenwick import fw_build, fw_add, fw_total, fw_find
from .constants import q, kB_eV
from .device_kmc import (DeviceKMC, _crow, _srate, _kill, _charge_change, _sidx)
from .exciton_device import ExcitonDeviceKMC


@njit(cache=True)
def _xrow_c(c, xpos, xocc, occ, nbrE, hop, kdec, st, kTT, kqw, kq_on, shape, rates):
    """Exciton rate row: [0,M) hops | M decay | M+1..M+S TTA | M+1+S..M+2S TPQ | M+1+2S onsite TPQ."""
    s = xpos[c]
    M = nbrE.shape[1]
    S = st.shape[0]
    Lx, Ly, Lz = shape[0], shape[1], shape[2]
    tot = 0.0
    for m in range(M):
        j = nbrE[s, m]
        if j < 0:
            rates[c, m] = 0.0
            continue
        rates[c, m] = hop[s, m]
        tot += hop[s, m]
    rates[c, M] = kdec[s]
    tot += kdec[s]
    ix = s // (Ly * Lz)
    iy = (s // Lz) % Ly
    iz = s % Lz
    for k in range(S):
        jz = iz + st[k, 2]
        if jz < 0 or jz >= Lz:
            rates[c, M + 1 + k] = 0.0
            rates[c, M + 1 + S + k] = 0.0
            continue
        j = _sidx(ix + st[k, 0], iy + st[k, 1], jz, Lx, Ly, Lz)
        w = 0.5 * kTT[k] if xocc[j] >= 0 else 0.0
        rates[c, M + 1 + k] = w
        tot += w
        w = kqw[k] if occ[j] != 0 else 0.0
        rates[c, M + 1 + S + k] = w
        tot += w
    w = kq_on if occ[s] != 0 else 0.0
    rates[c, M + 1 + 2 * S] = w
    tot += w
    return tot


@njit(cache=True)
def _xkill(c, xtree, xtot, xpos, xocc, xfree, nxfree):
    s = xpos[c]
    xocc[s] = -1
    xpos[c] = -1
    fw_add(xtree, c, -xtot[c])
    xtot[c] = 0.0
    xfree[nxfree] = c
    return nxfree + 1


@njit(cache=True)
def _refresh(site, updx, shape, xocc, xpos, occ, nbrE, hop, kdec, st, kTT, kqw, kq_on,
             xrates, xtree, xtot):
    Lx, Ly, Lz = shape[0], shape[1], shape[2]
    ix = site // (Ly * Lz)
    iy = (site // Lz) % Ly
    iz = site % Lz
    for kk in range(-1, updx.shape[0]):
        if kk < 0:
            jx, jy, jz = ix, iy, iz
        else:
            jx = ix + updx[kk, 0]
            jy = iy + updx[kk, 1]
            jz = iz + updx[kk, 2]
        if jz < 0 or jz >= Lz:
            continue
        j = _sidx(jx, jy, jz, Lx, Ly, Lz)
        c3 = xocc[j]
        if c3 >= 0:
            new = _xrow_c(c3, xpos, xocc, occ, nbrE, hop, kdec, st, kTT, kqw, kq_on,
                          shape, xrates)
            fw_add(xtree, c3, new - xtot[c3])
            xtot[c3] = new


@njit(cache=True)
def _run_coupled(nbr, valid, off, dist, z, EH, EL, nuh, nue, U, Uimg, src, shape,
                 stencil, Vst, upd, kT, alpha, EFa, EFc, L, area, eps_r, a, mode,
                 n_eq, n_meas, mf_update, cmax, seed, r_inj,
                 nbrE, hop, kdec, krad, st, kTT, kqw, kq_on, updx, xmax):
    np.random.seed(seed)
    N, M = nbr.shape
    Lx, Ly, Lz = shape[0], shape[1], shape[2]
    eps = 8.8541878128e-12 * eps_r
    nsrc = src.shape[0]
    occ = np.zeros(N, dtype=np.int64)
    who = -np.ones(N, dtype=np.int64)
    cpos = -np.ones(cmax, dtype=np.int64)
    cq = np.zeros(cmax, dtype=np.int64)
    free = np.arange(cmax - 1, -1, -1).astype(np.int64)
    nfree = cmax
    K = M + 2
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
    # excitons
    MX = nbrE.shape[1]
    S = st.shape[0]
    KX = MX + 2 + 2 * S
    xocc = -np.ones(N, dtype=np.int64)
    xpos = -np.ones(xmax, dtype=np.int64)
    xfree = np.arange(xmax - 1, -1, -1).astype(np.int64)
    nxfree = xmax
    xrates = np.zeros((xmax, KX))
    xtot = np.zeros(xmax)
    xtree = np.zeros(xmax)
    # counters: formed, rad, nr, tpq, tta, overflow
    cnt = np.zeros(6, dtype=np.int64)
    emis = np.zeros(Lz)
    nx_area = 0.0
    nx = 0
    dzsum = 0.0
    t = 0.0
    tm0 = 0.0
    nrec = 0
    recz = np.zeros(Lz)
    pz = np.zeros(Lz)
    nzz = np.zeros(Lz)
    ninj = 0
    measuring = False
    x_start = 0
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
            cnt[:] = 0
            emis[:] = 0.0
            nx_area = 0.0
            x_start = nx
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
            fw_build(xtree, xtot)
        Rc = fw_total(tree)
        Rx = fw_total(xtree)
        if Rx < 0:
            Rx = 0.0
        R = Rc + Rx
        if R <= 0:
            break
        dt = -np.log(1.0 - np.random.random()) / R
        if measuring:
            for i in range(Lz):
                pz[i] += cnt_h[i] * dt
                nzz[i] += cnt_e[i] * dt
            nx_area += nx * dt
        t += dt
        u = np.random.random() * R
        if u >= Rc and Rx > 0:
            # ------------------------------------------------ exciton event
            c = fw_find(xtree, min(u - Rc, Rx * (1 - 1e-15)))
            if xtot[c] <= 0 or xpos[c] < 0:
                continue
            u2 = np.random.random() * xtot[c]
            m = 0
            acc = xrates[c, 0]
            while acc < u2 and m < KX - 1:
                m += 1
                acc += xrates[c, m]
            s = xpos[c]
            t1 = -1
            if m < MX:                                   # Förster hop
                j = nbrE[s, m]
                if xocc[j] >= 0:                         # contact TTA: mover lost
                    nxfree = _xkill(c, xtree, xtot, xpos, xocc, xfree, nxfree)
                    nx -= 1
                    if measuring:
                        cnt[4] += 1
                else:
                    xocc[s] = -1
                    xocc[j] = c
                    xpos[c] = j
                    t1 = j
            elif m == MX:                                # intrinsic decay
                if measuring:
                    if np.random.random() * kdec[s] < krad[s]:
                        cnt[1] += 1
                        emis[s % Lz] += 1
                    else:
                        cnt[2] += 1
                nxfree = _xkill(c, xtree, xtot, xpos, xocc, xfree, nxfree)
                nx -= 1
            elif m <= MX + S:                            # Förster TTA, partner lost
                k = m - MX - 1
                ix = s // (Ly * Lz)
                iy = (s // Lz) % Ly
                j = _sidx(ix + st[k, 0], iy + st[k, 1], s % Lz + st[k, 2], Lx, Ly, Lz)
                c2 = xocc[j]
                if c2 >= 0:
                    nxfree = _xkill(c2, xtree, xtot, xpos, xocc, xfree, nxfree)
                    nx -= 1
                    t1 = j
                    if measuring:
                        cnt[4] += 1
            else:                                        # TPQ by a live polaron
                nxfree = _xkill(c, xtree, xtot, xpos, xocc, xfree, nxfree)
                nx -= 1
                if measuring:
                    cnt[3] += 1
            _refresh(s, updx, shape, xocc, xpos, occ, nbrE, hop, kdec, st, kTT, kqw,
                     kq_on, xrates, xtree, xtot)
            if t1 >= 0:
                _refresh(t1, updx, shape, xocc, xpos, occ, nbrE, hop, kdec, st, kTT, kqw,
                         kq_on, xrates, xtree, xtot)
            continue
        # ---------------------------------------------------- charge event
        e = fw_find(tree, min(u, Rc * (1 - 1e-15)))
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
                if occ[j] == -qq:            # recombination -> exciton on site j
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
                        cnt[0] += 1
                    if xocc[j] >= 0:
                        if measuring:
                            cnt[4] += 1                  # born onto an exciton
                    elif nxfree == 0:
                        if measuring:
                            cnt[5] += 1
                    else:
                        nxfree -= 1
                        cx = xfree[nxfree]
                        xpos[cx] = j
                        xocc[j] = cx
                        nx += 1
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
            # live TPQ: excitons near a moved polaron see the new configuration
            _refresh(site, updx, shape, xocc, xpos, occ, nbrE, hop, kdec, st, kTT, kqw,
                     kq_on, xrates, xtree, xtot)
    return (dzsum, t - tm0, nrec, recz, pz, nzz, ninj, cnt, emis, nx_area, x_start, nx)


class CoupledDeviceKMC(DeviceKMC):
    """3D OLED KMC with charges and excitons in one event loop.

    Takes all :class:`~orchid.device_kmc.DeviceKMC` arguments, plus these
    exciton parameters (scalars or per-z arrays):
    ``tau_r, tau_nr`` [s]; ``R0`` (Förster radius in sites); ``sigma_ex`` [eV];
    ``rc_x`` (Förster hop cutoff); ``k_TT0, R_TT`` (TTA); ``k_q0, R_q`` (TPQ);
    ``r_x`` (TTA/TPQ stencil); ``k_metal0, d_metal`` (electrode quenching);
    ``eta_ST`` (spin statistics); ``eta_out`` (outcoupling).
    """

    def __init__(self, layers, tau_r=1e-6, tau_nr=1e-5, R0=1.5, sigma_ex=0.0, rc_x=2.0,
                 k_TT0=1e8, R_TT=1.5, k_q0=1e8, R_q=1.5, r_x=3.0, k_metal0=0.0,
                 d_metal=1.0, eta_ST=1.0, eta_out=0.2, **kw):
        super().__init__(layers, **kw)
        seed = kw.get("seed", 0)
        self.x = ExcitonDeviceKMC(self.shape, a=self.a, T=self.T, tau_r=tau_r, tau_nr=tau_nr,
                                  R0=R0, sigma_ex=sigma_ex, rc=rc_x, k_TT0=k_TT0, R_TT=R_TT,
                                  r_TT=r_x, k_metal0=k_metal0, d_metal=d_metal,
                                  periodic_z=False, seed=seed + 101)
        rs = np.sqrt((self.x.st ** 2).sum(1))
        self.kqw = k_q0 * (R_q / rs) ** 6
        self.kq_on = 10.0 * k_q0
        self.eta_ST, self.eta_out = eta_ST, eta_out
        from .lattice import neighbour_offsets
        self.updx = neighbour_offsets(max(rc_x, r_x)).astype(np.int64)

    def run(self, V, n_eq=200000, n_meas=1000000, mf_update=2000, max_carriers=None,
            max_excitons=None, seed=1):
        kT = kB_eV * self.T
        L = self.Ld
        U = (self.Wa - V) + (self.Wc - self.Wa + V) * self.zm / L
        if max_carriers is None:
            max_carriers = max(2000, self.N // 5)
        if max_excitons is None:
            max_excitons = max(2000, self.N // 5)
        mode = {"none": 0, "meanfield": 1, "full": 2}[self.coulomb]
        x = self.x
        out = _run_coupled(self.nbr, self.valid, self.off.astype(np.int64), self.dist,
                           self.z, self.EH, self.EL, self.nuh, self.nue, U, self.Uimg,
                           self.src, np.array(self.shape, dtype=np.int64), self.stencil,
                           self.Vst, self.upd, kT, self.alpha, -V, 0.0, L, self.area,
                           self.eps_r, self.a, mode, int(n_eq), int(n_meas), int(mf_update),
                           int(max_carriers), int(seed), float(self.r_inj),
                           x.nbrE, x.hop, x.kdec, x.krad, x.st, x.kTT, self.kqw,
                           float(self.kq_on), self.updx, int(max_excitons))
        dzsum, tm, nrec, recz, pz, nz, ninj, cnt, emis, nxa, x0, x1 = out
        J = q * dzsum * self.a / (L * self.area * tm)
        Vcell = self.Lx * self.Ly * self.a ** 3
        formed, nrad, nnr, ntpq, ntta, novf = (int(v) for v in cnt)
        n_el = J * self.area * tm / q
        decided = nrad + nnr + ntpq + ntta
        fr = (lambda v: v / decided) if decided > 0 else (lambda v: np.nan)
        return dict(V=V, J=J, rec_rate=nrec / (tm * self.area),
                    gamma_b=(q * nrec / (tm * self.area)) / J if J > 0 else np.nan,
                    rec_profile=recz / max(recz.sum(), 1),
                    emission_profile=emis / max(emis.sum(), 1),
                    p_z=pz / (tm * Vcell), n_z=nz / (tm * Vcell),
                    z=(np.arange(self.Lz) + 1) * self.a, t=tm, injections=ninj,
                    excitons=dict(formed=formed, radiative=nrad, nonradiative=nnr, TPQ=ntpq,
                                  TTA=ntta, overflow=novf, alive_start=int(x0),
                                  alive_end=int(x1)),
                    eta_rad=fr(nrad), frac_nr=fr(nnr), frac_TPQ=fr(ntpq), frac_TTA=fr(ntta),
                    n_ex=nxa / (tm * self.N * self.a ** 3),
                    EQE=self.eta_ST * self.eta_out * nrad / n_el if n_el > 0 else np.nan)
