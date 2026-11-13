r"""Kinetic Monte Carlo (Gillespie / BKL, rejection-free) for hopping transport.

Two engines are provided:

1. :func:`kmc_independent` - non-interacting carriers (zero-density limit).
   Rates are static, so cumulative rate tables are pre-computed and every hop
   costs O(log M). Used for mobility vs. T and F, time-of-flight transients
   and energetic relaxation (approach to :math:`-\sigma^2/k_BT`).
2. :func:`kmc_interacting` - many carriers with site exclusion (Fermi-Dirac
   statistics emerge automatically) and optional screened Coulomb
   interaction within a cutoff (potential maintained incrementally).
   Used for the carrier-density dependence of the mobility.

Field convention: ``F_red = e F a`` in eV (energy drop per lattice constant)
along +x; positive carriers drift along +x. Mobility in SI follows from
:math:`\mu = v/F = \langle\Delta x\rangle a / (F\,\Delta t)`.
"""
import numpy as np
from ._jit import njit, prange
from .constants import q


# ----------------------------------------------------------------------------
# non-interacting engine
# ----------------------------------------------------------------------------
@njit(cache=True)
def _one_carrier(nbr, dx, Wcum, Wtot, energy, s, n_eq, n_meas, tgrid, x_stop,
                 seed, Xt_row, Et_row):
    np.random.seed(seed)
    M = nbr.shape[1]
    nt = tgrid.shape[0]
    t = 0.0
    x = 0.0
    k = 0
    xm0 = 0.0
    tm0 = 0.0
    ntot = n_eq + n_meas
    hop = 0
    running = True
    while running and hop < ntot:
        if hop == n_eq:
            xm0 = x
            tm0 = t
        dt = -np.log(1.0 - np.random.random()) / Wtot[s]
        while k < nt and tgrid[k] < t + dt:
            Xt_row[k] = x
            Et_row[k] = energy[s]
            k += 1
        t += dt
        u = np.random.random() * Wtot[s]
        lo = 0
        hi = M - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if Wcum[s, mid] < u:
                lo = mid + 1
            else:
                hi = mid
        x += dx[s, lo]
        s = nbr[s, lo]
        hop += 1
        if x >= x_stop:
            running = False
    fill = x if not running else np.nan
    while k < nt:
        Xt_row[k] = fill
        Et_row[k] = energy[s] if not running else np.nan
        k += 1
    return x - xm0, t - tm0


@njit(cache=True, parallel=True)
def _run_independent(nbr, dx, Wcum, Wtot, energy, starts, n_eq, n_meas,
                     tgrid, seed, x_stop):
    nc = starts.shape[0]
    nt = tgrid.shape[0]
    X = np.zeros(nc)          # displacement during measurement
    TT = np.zeros(nc)         # time during measurement
    Xt = np.zeros((nc, nt))   # x(t) from t=0 (TOF / transient)
    Et = np.zeros((nc, nt))   # energy(t) (relaxation)
    for c in prange(nc):
        a, b = _one_carrier(nbr, dx, Wcum, Wtot, energy, starts[c], n_eq, n_meas,
                            tgrid, x_stop, seed + 7919 * c, Xt[c], Et[c])
        X[c] = a
        TT[c] = b
    return X, TT, Xt, Et


def kmc_independent(lat, W, n_carriers=64, n_eq=200000, n_meas=200000,
                    tgrid=None, seed=1, start="random", x_stop=np.inf):
    """Run non-interacting KMC on a :class:`~orchid.lattice.Lattice`.

    Returns dict with total displacement / time in the measurement window
    (lattice units, s) and x(t), E(t) on ``tgrid``.
    """
    Wcum = np.cumsum(W, axis=1)
    Wtot = Wcum[:, -1].copy()
    rng = np.random.default_rng(seed)
    if start == "random":
        starts = rng.integers(0, lat.N, n_carriers)
    else:
        starts = np.asarray(start, dtype=np.int64)
    if tgrid is None:
        tgrid = np.array([0.0])
    X, TT, Xt, Et = _run_independent(lat.nbr, lat.disp[:, :, 0].copy(), Wcum,
                                     Wtot, lat.energy, starts.astype(np.int64),
                                     int(n_eq), int(n_meas),
                                     np.asarray(tgrid, float), int(seed),
                                     float(x_stop))
    return dict(dx=X, dt=TT, x_t=Xt, E_t=Et, tgrid=np.asarray(tgrid))


def mobility_independent(lat, T, F_red, n_carriers=64, n_eq=200000,
                         n_meas=400000, seed=1, **rate_kw):
    """Zero-density drift mobility (m^2/Vs) from steady-state velocity.

    The ratio estimator  mu = sum(dx) a / (F sum(dt))  is used, which is the
    correct time-average for an ergodic ensemble of independent carriers.
    Returns (mu, standard error).
    """
    W = lat.rates(T, (F_red, 0.0, 0.0), **rate_kw)
    out = kmc_independent(lat, W, n_carriers, n_eq, n_meas, seed=seed)
    F = F_red / lat.a                          # V/m  (F_red in eV -> volts)
    v = out["dx"] * lat.a / out["dt"]
    mu_each = v / F
    mu = out["dx"].sum() * lat.a / (F * out["dt"].sum())
    err = mu_each.std(ddof=1) / np.sqrt(len(mu_each))
    return mu, err


def time_of_flight(lat, T, F_red, L_sample, n_carriers=512, tgrid=None,
                   seed=3, **rate_kw):
    r"""Simulated time-of-flight transient.

    Carriers start at random sites (photogeneration sheet) and their mean
    displacement <x(t)> is recorded; the transient photocurrent is
    :math:`I(t)\propto d\langle x\rangle/dt` until the mean arrival at
    ``L_sample`` (lattice units). Returns t, <x>(t), I(t), t_transit.
    """
    W = lat.rates(T, (F_red, 0.0, 0.0), **rate_kw)
    if tgrid is None:
        tgrid = np.logspace(-14, -2, 241)
    out = kmc_independent(lat, W, n_carriers, n_eq=0, n_meas=int(2e8),
                          tgrid=tgrid, seed=seed, x_stop=float(L_sample))
    xt = out["x_t"]
    # carriers absorbed once they cross L_sample: count only those not arrived
    arrived = np.maximum.accumulate(np.nan_to_num(xt, nan=1e30) >= L_sample, axis=1)
    xclip = np.where(arrived, L_sample, np.nan_to_num(xt, nan=L_sample))
    xm = xclip.mean(0)
    I = np.gradient(xm, tgrid)
    frac = arrived.mean(0)
    t_tr = np.interp(0.5, frac, tgrid) if frac[-1] >= 0.5 else np.nan
    return dict(t=tgrid, x=xm, I=I, t_transit=t_tr, arrived=frac,
                E_mean=np.nanmean(out["E_t"], axis=0))


def energy_relaxation(lat, T, n_carriers=256, tgrid=None, seed=5, **rate_kw):
    r"""Mean carrier energy vs time at zero field (Bässler relaxation).

    For a Gaussian DOS the long-time limit is
    :math:`\langle E\rangle_\infty=-\sigma^2/k_BT`.
    """
    W = lat.rates(T, (0.0, 0.0, 0.0), **rate_kw)
    if tgrid is None:
        tgrid = np.logspace(-13, 0, 131)
    out = kmc_independent(lat, W, n_carriers, n_eq=0, n_meas=int(3e6),
                          tgrid=tgrid, seed=seed)
    return tgrid, np.nanmean(out["E_t"], axis=0)


# ----------------------------------------------------------------------------
# interacting engine (exclusion + screened Coulomb within cutoff)
# ----------------------------------------------------------------------------
@njit(cache=True)
def _run_interacting(nbr, dxyz, energy, Wbare_pref, dEb_field, occ, pos,
                     stencil, stencil_V, shape, n_eq, n_meas, kT, use_coul,
                     rate_kind, lam, seed):
    np.random.seed(seed)
    N, M = nbr.shape
    nc = pos.shape[0]
    Lx, Ly, Lz = shape[0], shape[1], shape[2]
    phi = np.zeros(N)
    ns = stencil.shape[0]
    if use_coul:
        for c in range(nc):
            s = pos[c]
            ix = s // (Ly * Lz)
            iy = (s // Lz) % Ly
            iz = s % Lz
            for k in range(ns):
                jx = (ix + stencil[k, 0]) % Lx
                jy = (iy + stencil[k, 1]) % Ly
                jz = (iz + stencil[k, 2]) % Lz
                phi[(jx * Ly + jy) * Lz + jz] += stencil_V[k]
    rates = np.zeros((nc, M))
    rtot = np.zeros(nc)
    xdisp = 0.0
    t = 0.0
    x0 = 0.0
    t0 = 0.0
    for step in range(n_eq + n_meas):
        if step == n_eq:
            x0 = xdisp
            t0 = t
        R = 0.0
        for c in range(nc):
            s = pos[c]
            rc_ = 0.0
            for m in range(M):
                j = nbr[s, m]
                if occ[j]:
                    rates[c, m] = 0.0
                    continue
                dE = dEb_field[s, m]
                if use_coul:
                    # remove self-interaction: carrier itself contributes
                    # V(r_sj) at j; nearest stencil entry index m matches
                    dE += phi[j] - phi[s] - stencil_V[m]
                if rate_kind == 0:
                    w = Wbare_pref[s, m]
                    if dE > 0:
                        w *= np.exp(-dE / kT)
                else:
                    w = Wbare_pref[s, m] * np.exp(-(dE + lam) ** 2 / (4 * lam * kT))
                rates[c, m] = w
                rc_ += w
            rtot[c] = rc_
            R += rc_
        if R <= 0:
            break
        t += -np.log(1.0 - np.random.random()) / R
        u = np.random.random() * R
        cc = 0
        acc = rtot[0]
        while acc < u and cc < nc - 1:
            cc += 1
            acc += rtot[cc]
        u2 = u - (acc - rtot[cc])
        mm = 0
        acc2 = rates[cc, 0]
        while acc2 < u2 and mm < M - 1:
            mm += 1
            acc2 += rates[cc, mm]
        s = pos[cc]
        j = nbr[s, mm]
        occ[s] = False
        occ[j] = True
        pos[cc] = j
        xdisp += dxyz[s, mm, 0]
        if use_coul:
            for sgn_site in range(2):
                site = s if sgn_site == 0 else j
                sgn = -1.0 if sgn_site == 0 else 1.0
                ix = site // (Ly * Lz)
                iy = (site // Lz) % Ly
                iz = site % Lz
                for k in range(ns):
                    jx = (ix + stencil[k, 0]) % Lx
                    jy = (iy + stencil[k, 1]) % Ly
                    jz = (iz + stencil[k, 2]) % Lz
                    phi[(jx * Ly + jy) * Lz + jz] += sgn * stencil_V[k]
    return xdisp - x0, t - t0


def mobility_interacting(lat, T, F_red, density, n_eq=100000, n_meas=300000,
                         coulomb=False, eps_r=3.5, r_coul=4.0, seed=11,
                         kind="ma", nu0=1e12, alpha=10.0, J0=0.01, lam=0.2,
                         engine="fast"):
    """Finite-density mobility (m^2/Vs) with exclusion and optional Coulomb.

    ``density`` is the fraction of occupied sites (p a^3 in EGDM notation).
    ``engine="fast"`` (default) uses incremental rate updates and a Fenwick
    tree: cost per hop O(log N_c + local carriers), independent of the total
    number of carriers. ``engine="dense"`` recomputes every rate each step
    (O(N_c M), reference implementation).
    """
    from .constants import kB_eV, eps0, hbar_eVs
    from .lattice import neighbour_offsets
    kT = kB_eV * T
    nc = max(1, int(round(density * lat.N)))
    rng = np.random.default_rng(seed)
    pos = rng.choice(lat.N, nc, replace=False).astype(np.int64)
    occ = np.zeros(lat.N, dtype=np.bool_)
    occ[pos] = True
    dEb = lat.energy[lat.nbr] - lat.energy[:, None] - lat.disp[:, :, 0] * F_red
    if kind == "ma":
        pref = nu0 * np.exp(-2 * alpha * lat.dist)
        rk = 0
    else:
        J = J0 * np.exp(-alpha * (lat.dist - 1.0))
        pref = (2 * np.pi / hbar_eVs) * J ** 2 / np.sqrt(4 * np.pi * lam * kT)
        rk = 1
    if np.any(lat.gamma):
        pref = pref * np.exp(lat.gamma[:, None] + lat.gamma[lat.nbr])
    # Coulomb stencil: first M entries must coincide with hopping offsets
    st = np.vstack([lat.offsets, [o for o in neighbour_offsets(r_coul)
                                  if not any((o == lat.offsets).all(1))]]) \
        if coulomb else lat.offsets.copy()
    r = np.sqrt((st ** 2).sum(1)) * lat.a
    V = q / (4 * np.pi * eps0 * eps_r * r) if coulomb else np.zeros(len(st))
    if engine == "fast":
        rc_upd = (r_coul + lat.rc) if coulomb else lat.rc
        upd = neighbour_offsets(rc_upd).astype(np.int64)
        dx, dt = _run_interacting_fast(lat.nbr, lat.disp, pref, dEb, occ, pos,
                                       st.astype(np.int64), V, upd,
                                       np.array(lat.shape, dtype=np.int64),
                                       int(n_eq), int(n_meas), kT, bool(coulomb),
                                       rk, lam, seed, 20000)
        F = F_red / lat.a
        return dx * lat.a / (F * dt * nc)
    dx, dt = _run_interacting(lat.nbr, lat.disp, lat.energy, pref, dEb, occ,
                              pos, st.astype(np.int64), V,
                              np.array(lat.shape, dtype=np.int64), int(n_eq),
                              int(n_meas), kT, bool(coulomb), rk, lam, seed)
    F = F_red / lat.a
    # total current = sum of displacements; mobility per carrier
    return dx * lat.a / (F * dt * nc)


# ----------------------------------------------------------------------------
# fast interacting engine: incremental rate updates + Fenwick-tree selection
# ----------------------------------------------------------------------------
from ._fenwick import fw_build, fw_add, fw_total, fw_find  # noqa: E402


@njit(cache=True)
def _site_index(ix, iy, iz, Lx, Ly, Lz):
    return ((ix % Lx) * Ly + (iy % Ly)) * Lz + (iz % Lz)


@njit(cache=True)
def _row(c, pos, nbr, occ, phi, dEb, pref, stencil_V, kT, use_coul, rate_kind,
         lam, rates):
    s = pos[c]
    M = nbr.shape[1]
    tot = 0.0
    for m in range(M):
        j = nbr[s, m]
        if occ[j]:
            rates[c, m] = 0.0
            continue
        dE = dEb[s, m]
        if use_coul:
            dE += phi[j] - phi[s] - stencil_V[m]
        if rate_kind == 0:
            w = pref[s, m]
            if dE > 0:
                w *= np.exp(-dE / kT)
        else:
            w = pref[s, m] * np.exp(-(dE + lam) ** 2 / (4 * lam * kT))
        rates[c, m] = w
        tot += w
    return tot


@njit(cache=True)
def _run_interacting_fast(nbr, dxyz, pref, dEb, occ, pos, stencil, stencil_V,
                          upd, shape, n_eq, n_meas, kT, use_coul, rate_kind,
                          lam, seed, rebuild_every):
    np.random.seed(seed)
    N, M = nbr.shape
    nc = pos.shape[0]
    Lx, Ly, Lz = shape[0], shape[1], shape[2]
    who = -np.ones(N, dtype=np.int64)
    for c in range(nc):
        who[pos[c]] = c
    phi = np.zeros(N)
    ns = stencil.shape[0]
    if use_coul:
        for c in range(nc):
            s = pos[c]
            ix = s // (Ly * Lz)
            iy = (s // Lz) % Ly
            iz = s % Lz
            for k in range(ns):
                phi[_site_index(ix + stencil[k, 0], iy + stencil[k, 1],
                                iz + stencil[k, 2], Lx, Ly, Lz)] += stencil_V[k]
    rates = np.zeros((nc, M))
    rtot = np.zeros(nc)
    for c in range(nc):
        rtot[c] = _row(c, pos, nbr, occ, phi, dEb, pref, stencil_V, kT, use_coul,
                       rate_kind, lam, rates)
    tree = np.zeros(nc)
    fw_build(tree, rtot)
    nu = upd.shape[0]
    t = 0.0
    xd = 0.0
    x0 = 0.0
    t0 = 0.0
    for step in range(n_eq + n_meas):
        if step == n_eq:
            x0 = xd
            t0 = t
        if step % rebuild_every == 0:
            fw_build(tree, rtot)
        R = fw_total(tree)
        if R <= 0:
            break
        t += -np.log(1.0 - np.random.random()) / R
        u = np.random.random() * R
        cc = fw_find(tree, u)
        if rtot[cc] <= 0:
            continue
        u2 = np.random.random() * rtot[cc]
        mm = 0
        acc = rates[cc, 0]
        while acc < u2 and mm < M - 1:
            mm += 1
            acc += rates[cc, mm]
        s = pos[cc]
        j = nbr[s, mm]
        occ[s] = False
        occ[j] = True
        who[s] = -1
        who[j] = cc
        pos[cc] = j
        xd += dxyz[s, mm, 0]
        if use_coul:
            for side in range(2):
                site = s if side == 0 else j
                sg = -1.0 if side == 0 else 1.0
                ix = site // (Ly * Lz)
                iy = (site // Lz) % Ly
                iz = site % Lz
                for k in range(ns):
                    phi[_site_index(ix + stencil[k, 0], iy + stencil[k, 1],
                                    iz + stencil[k, 2], Lx, Ly, Lz)] += sg * stencil_V[k]
        # refresh the moved carrier and every carrier whose rates changed
        new = _row(cc, pos, nbr, occ, phi, dEb, pref, stencil_V, kT, use_coul,
                   rate_kind, lam, rates)
        fw_add(tree, cc, new - rtot[cc])
        rtot[cc] = new
        for side in range(2):
            site = s if side == 0 else j
            ix = site // (Ly * Lz)
            iy = (site // Lz) % Ly
            iz = site % Lz
            for k in range(nu):
                kk = _site_index(ix + upd[k, 0], iy + upd[k, 1], iz + upd[k, 2],
                                 Lx, Ly, Lz)
                c2 = who[kk]
                if c2 >= 0 and c2 != cc:
                    new = _row(c2, pos, nbr, occ, phi, dEb, pref, stencil_V, kT,
                               use_coul, rate_kind, lam, rates)
                    fw_add(tree, c2, new - rtot[c2])
                    rtot[c2] = new
    return xd - x0, t - t0
