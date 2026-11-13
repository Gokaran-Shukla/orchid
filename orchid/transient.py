r"""Transient drift-diffusion: implicit (backward-Euler) time stepping of the
coupled Poisson / continuity system on top of :class:`~orchid.drift_diffusion.Device`.

Experiments implemented
-----------------------
* :func:`step_response` - generic V(t), G(t) protocol; total current
  includes the displacement current :math:`\varepsilon\,\partial_t F`.
* :func:`celiv` - charge extraction by linearly increasing voltage
  (Juska et al., PRL 84, 4946 (2000)); mobility from the peak time
  :math:`\mu = 2d^2/[3At_{max}^2(1+0.36\,\Delta j/j_0)]`.
* :func:`tpv` - small-perturbation transient photovoltage at open circuit
  (Shuttle et al., APL 92, 093311 (2008)); returns the decay time
  :math:`\tau_{\Delta n}`, which for Langevin recombination equals
  :math:`1/(2\gamma n)`.
* :func:`tpc` - transient photocurrent at short circuit (rise/fall).

Every step is solved with the fully coupled banded Newton method;
time steps are geometric (log-spaced), so microsecond to millisecond tails
are resolved.
"""
import numpy as np
from .constants import q


def _state(dev, z, V):
    psi, u, v = dev.unpack(z, V)
    return psi, dev.Nc_x * np.exp(u), dev.Nv_x * np.exp(v)


def _total_current(dev, z, V, psi_old, dt):
    psi, n, p = _state(dev, z, V)
    Jn, Jp, F, _, _ = dev.currents(psi, n, p)
    Jc = Jn + Jp
    if psi_old is None:
        return np.median(Jc), Jc
    F_old = -(psi_old[1:] - psi_old[:-1]) / dev.h
    Jd = dev.eps_e * (F - F_old) / dt
    Jt = Jc + Jd
    return np.median(Jt), Jt


def step_response(dev, tgrid, Vfun, Gfun=None, z0=None, V0=None, maxit=60):
    """Integrate from the steady state at (V0, G(0)) along ``tgrid``.

    Vfun(t) -> applied voltage, Gfun(t) -> generation scale factor (x dev.Gx).
    Returns dict(t, V, J) with the total (conduction + displacement) current.
    """
    G0 = dev.Gx.copy()
    if Gfun is not None:
        dev.Gx = G0 * Gfun(tgrid[0])
    V0 = Vfun(tgrid[0]) if V0 is None else V0
    if z0 is None:
        res = dev.jv(np.array([V0]))
        z0 = _z_from_profile(dev, res["profiles"][0])
    z = z0.copy()
    psi_old, n_old, p_old = _state(dev, z, V0)
    out_t, out_J, out_V = [tgrid[0]], [_total_current(dev, z, V0, None, 1)[0]], [V0]
    for k in range(1, len(tgrid)):
        t = tgrid[k]
        dt = t - tgrid[k - 1]
        Vt = Vfun(t)
        if Gfun is not None:
            dev.Gx = G0 * Gfun(t)
        dev._tstate = (n_old, p_old, dt)
        zn, ok, _ = dev.newton(z, Vt, maxit=maxit)
        if not ok:   # sub-step
            sub = np.linspace(tgrid[k - 1], t, 6)
            dts = sub[1] - sub[0]
            zc, nso, pso = z, n_old, p_old
            for ts in sub[1:]:
                dev._tstate = (nso, pso, dts)
                zc, ok, _ = dev.newton(zc, Vfun(ts), maxit=maxit)
                _, nso, pso = _state(dev, zc, Vfun(ts))
            zn = zc
        dev._tstate = None
        Jt, _ = _total_current(dev, zn, Vt, psi_old, dt)
        out_t.append(t)
        out_J.append(Jt)
        out_V.append(Vt)
        z = zn
        psi_old, n_old, p_old = _state(dev, z, Vt)
    dev.Gx = G0
    dev._tstate = None
    return dict(t=np.array(out_t), J=np.array(out_J), V=np.array(out_V), z=z)


def _z_from_profile(dev, prof):
    psi, n, p = prof["psi"], prof["n"], prof["p"]
    return np.stack([psi[1:-1], np.log(n[1:-1] / dev.Nc_x[1:-1]), np.log(p[1:-1] / dev.Nv_x[1:-1])], 1).ravel()


def celiv(dev, A, t_end, n_t=200, V_offset=0.0, G_pulse=0.0, t_pulse=0.0,
          t_delay=0.0):
    """Dark or photo-CELIV.

    Dark: ramp V(t) = V_offset - A t from the dark steady state.
    Photo: a generation pulse ``G_pulse`` [m^-3 s^-1] of length ``t_pulse``
    at V_offset, an optional delay, then the ramp. Times in the output are
    measured from the start of the ramp.

    Returns dict(t, J, t_max, j0, dj, mu_juska, mu_lorrmann), with
    mu_juska = 2 d^2 / (3 A t_max^2) (Juska et al. 2000) and the
    Lorrmann et al. (JAP 108, 113705 (2010)) correction 1/(1+0.36 dj/j0).
    """
    t0 = t_pulse + t_delay
    G_base = dev.Gx.copy()
    if G_pulse > 0:
        dev.Gx = np.full_like(G_base, G_pulse)
        gf = lambda t: 1.0 if (0 < t <= t_pulse) else 0.0
        pre = list(np.linspace(0, t_pulse, 11)) + (list(t_pulse + np.geomspace(
            t_delay * 1e-3, t_delay, 10)) if t_delay > 0 else [])
    else:
        gf = None
        pre = [0.0]
    ramp = list(t0 + np.geomspace(t_end * 1e-4, t_end, n_t))
    tg = np.array(pre + ramp)
    Vf = lambda t: V_offset - A * max(t - t0, 0.0)
    if gf is not None:
        # steady state in the dark, then pulse
        dev_G = dev.Gx.copy()
        dev.Gx = 0 * dev_G
        res = dev.jv(np.array([V_offset]))
        z0 = _z_from_profile(dev, res["profiles"][0])
        dev.Gx = dev_G
        r = step_response(dev, tg, Vf, Gfun=gf, z0=z0, V0=V_offset)
    else:
        r = step_response(dev, tg, Vf)
    dev.Gx = G_base
    m = r["t"] >= t0
    t = r["t"][m] - t0
    J = -r["J"][m]                               # extraction current positive
    j0 = dev.eps * A / dev.L                     # capacitive displacement step
    k = int(np.argmax(J[1:])) + 1
    t_max = t[k]
    if 1 < k < len(t) - 1:          # parabolic refinement of the peak in ln t
        x = np.log(t[k - 1:k + 2])
        c2, c1, c0 = np.polyfit(x, J[k - 1:k + 2], 2)
        if c2 < 0:
            t_max = float(np.exp(-c1 / (2 * c2)))
    dj = J[k] - j0
    d = dev.L
    mu_j = 2 * d ** 2 / (3 * A * t_max ** 2)
    mu_l = mu_j / (1 + 0.36 * dj / j0)
    return dict(t=t, J=J, t_max=t_max, j0=j0, dj=dj, mu_juska=mu_j, mu_lorrmann=mu_l)


def open_circuit_voltage(dev, V_guess=0.6, tol=1e-7):
    """Steady-state Voc by secant iteration on J(V)=0."""
    V0, V1 = V_guess - 0.05, V_guess + 0.05
    J0 = dev.jv(np.array([V0]))["J"][0]
    J1 = dev.jv(np.array([V1]))["J"][0]
    for _ in range(40):
        V2 = V1 - J1 * (V1 - V0) / (J1 - J0)
        J2 = dev.jv(np.array([V2]))["J"][0]
        V0, J0, V1, J1 = V1, J1, V2, J2
        if abs(V1 - V0) < tol:
            break
    return V1


def tpv(dev, pulse=0.05, t_pulse=None, t_end=None, n_t=120, Voc=None):
    """Small-perturbation transient photovoltage at open circuit.

    The device is held at open circuit (J=0 enforced at every step by a
    secant search on V). A generation pulse of relative size ``pulse`` with
    duration ``t_pulse`` is applied, and the voltage decay is fitted with a
    single exponential. Returns dict(t, V, dV, tau).
    """
    if Voc is None:
        Voc = open_circuit_voltage(dev)
    res = dev.jv(np.array([Voc]))
    z = _z_from_profile(dev, res["profiles"][0])
    pr = res["profiles"][0]
    nmid = np.sqrt(pr["n"][len(pr["n"]) // 2] * pr["p"][len(pr["p"]) // 2])
    gam = dev.langevin_prefactor * q * 2 * (dev.mu_n if not callable(dev.mu_n) else 0) / dev.eps
    tau_est = 1.0 / (2 * gam * nmid) if gam > 0 else 1e-6
    if t_pulse is None:
        t_pulse = 0.05 * tau_est
    if t_end is None:
        t_end = 8 * tau_est
    G0 = dev.Gx.copy()
    tg = np.concatenate([np.linspace(0, t_pulse, 6), t_pulse + np.geomspace(1e-3 * tau_est, t_end, n_t)])
    _, n_old, p_old = _state(dev, z, Voc)
    V = Voc
    Vs = [Voc]
    for k in range(1, len(tg)):
        dt = tg[k] - tg[k - 1]
        dev.Gx = G0 * (1 + pulse) if tg[k] <= t_pulse + 1e-30 else G0
        dev._tstate = (n_old, p_old, dt)

        def Jof(Vx, zz):
            zn, ok, _ = dev.newton(zz, Vx)
            psi, n, p = _state(dev, zn, Vx)
            Jn, Jp, F, _, _ = dev.currents(psi, n, p)
            return np.median(Jn + Jp), zn
        # open circuit: conduction current through the device must vanish
        Va, Vb = V, V + 1e-4
        Ja, za = Jof(Va, z)
        Jb, zb = Jof(Vb, za)
        for _ in range(30):
            Vc = Vb - Jb * (Vb - Va) / (Jb - Ja)
            Jc, zc = Jof(Vc, zb)
            Va, Ja, za, Vb, Jb, zb = Vb, Jb, zb, Vc, Jc, zc
            if abs(Vb - Va) < 1e-9:
                break
        V, z = Vb, zb
        dev._tstate = None
        _, n_old, p_old = _state(dev, z, V)
        Vs.append(V)
    dev.Gx = G0
    Vs = np.array(Vs)
    dV = Vs - Voc
    m = (tg > t_pulse) & (dV > 0.05 * dV.max())
    tau = -1.0 / np.polyfit(tg[m] - t_pulse, np.log(dV[m]), 1)[0]
    return dict(t=tg, V=Vs, dV=dV, tau=tau, tau_langevin=tau_est, n=nmid, Voc=Voc)


def tpc(dev, t_on, t_end, n_t=150, V=0.0):
    """Transient photocurrent: light switched on at t=0, off at t_on."""
    tg = np.concatenate([[0], np.geomspace(t_on * 1e-4, t_on, n_t // 2),
                         t_on + np.geomspace(t_on * 1e-4, t_end - t_on, n_t // 2)])
    r = step_response(dev, tg, lambda t: V, Gfun=lambda t: 1.0 if 0 < t <= t_on else 0.0)
    return r
