r"""Device-level workflows built on the drift-diffusion, exciton and optics
modules.

OLED external quantum efficiency (factorised form):

.. math::
   \mathrm{EQE} = \gamma_b\;\eta_{S/T}\;q_{\mathrm{eff}}\;\eta_{\mathrm{out}},
   \qquad \gamma_b = \frac{q\int R\,dx}{J}

where q_eff includes exciton quenching at the electrodes and
singlet-singlet / singlet-polaron annihilation from the continuum exciton
model (efficiency roll-off).

OPV: transfer-matrix generation profile -> drift-diffusion J-V under
illumination -> Jsc, Voc, FF, PCE.
"""
import numpy as np
from .constants import q
from .exciton import steady_state_excitons
from .drift_diffusion import pv_metrics


def recombination_profile(dev, prof):
    psi, n, p = prof["psi"], prof["n"], prof["p"]
    F = prof["F"]
    Fn = np.empty_like(psi)
    Fn[1:-1] = 0.5 * (F[1:] + F[:-1])
    Fn[0], Fn[-1] = F[0], F[-1]
    R, G, _ = dev.recomb(psi, n, p, Fn)
    return R


def simulate_oled(dev, V, D_ex=1e-8, tau_ex=1e-8, eta_ST=0.25, qPL=0.8,
                  n_org=1.75, kSS=0.0, kSP=0.0, eta_out=None):
    """Return dict(V, J, EQE, gamma_b, q_eff, emission_profiles)."""
    from .optics import outcoupling_classical
    if eta_out is None:
        eta_out = outcoupling_classical(n_org)
    res = dev.jv(V)
    out = dict(V=res["V"], J=res["J"], EQE=[], gamma_b=[], q_eff=[], S=[])
    for Vv, J, prof in zip(res["V"], res["J"], res["profiles"]):
        R = recombination_profile(dev, prof)
        x = dev.x
        Rint = np.trapezoid(R, x)
        gb = q * Rint / J if J > 0 else 0.0
        S, yld = steady_state_excitons(x, np.maximum(R, 0), D_ex, tau_ex,
                                       eta_S=1.0, kSS=kSS, kSP=kSP,
                                       npol=prof["n"] + prof["p"])
        qe = qPL * (yld if np.isfinite(yld) else 0.0)
        out["gamma_b"].append(gb)
        out["q_eff"].append(qe)
        out["EQE"].append(gb * eta_ST * qe * eta_out)
        out["S"].append(S)
    for k in ("EQE", "gamma_b", "q_eff"):
        out[k] = np.array(out[k])
    return out


def simulate_opv(dev, V, layers=None, active_index=None, lams=None,
                 eta_diss=1.0, P_in=1000.0):
    """Illuminated J-V. If ``layers`` given, G(x) from transfer matrix."""
    if layers is not None:
        from .optics import generation_profile
        if lams is None:
            lams = np.linspace(350e-9, 850e-9, 51)
        xg, G, _ = generation_profile(layers, active_index, lams)
        dev.Gx = eta_diss * np.interp(dev.x, xg, G)
    res = dev.jv(V)
    m = pv_metrics(res["V"], res["J"], P_in)
    return dict(V=res["V"], J=res["J"], metrics=m, G=dev.Gx.copy(),
                profiles=res["profiles"])
