"""V5 Mott-Gurney, V6 Koster Voc, V12 OLED / OPV device demonstrations,
plus EGDM-SCLC (van Mensfoort & Coehoorn 2008 trend)."""
import numpy as np
from common import plt, save, dump, C
from orchid.drift_diffusion import Device, mott_gurney, koster_voc, pv_metrics
from orchid.devices import simulate_oled, simulate_opv, recombination_profile
from orchid.models import egdm_pasveer, einstein_ratio_gaussian
from orchid.optics import lorentz_nk
from orchid.constants import q, eps0, kB_eV

import sys, os, json
from common import RES
SECT = sys.argv[1:] or ["v5", "v6", "opv", "oled"]
_p = os.path.join(RES, "devices.json")
out = json.load(open(_p)) if os.path.exists(_p) else {}
# ---------------------------------------------------------------- V5
if "v5" in SECT:
    V = np.logspace(np.log10(1.6), np.log10(40), 18)
    d = Device(L=100e-9, N=241, Eg=2.4, phi_anode_p=0.0, phi_cathode_n=1.0,
               mu_n=1e-9, mu_p=1e-8, recombination=False)
    r = d.jv(V)
    ratio = r["J"] / mott_gurney(V - r["Vbi"], d.L, d.eps_r, d.mu_p)
    # EGDM hole-only device (sigma = 0.1 eV, a = 1.6 nm) with g3 Einstein factor
    a, sig, nu0 = 1.6e-9, 0.1, 1.2e19
    T = 300.0
    mu_egdm = lambda p, F, T: egdm_pasveer(T, p, F, sig, a, nu0=nu0)
    pg = np.logspace(-8, -1, 60)
    g3tab = np.array([einstein_ratio_gaussian(x, T, sig) for x in pg])
    g3 = lambda dens: np.interp(np.log(np.clip(dens * a ** 3, 1e-8, 1e-1)), np.log(pg), g3tab)
    de = Device(L=100e-9, N=241, Eg=2.4, phi_anode_p=0.2, phi_cathode_n=1.0, Nv=1 / a ** 3,
                Nc=1 / a ** 3, mu_n=1e-12, mu_p=mu_egdm, g3_p=g3, recombination=False)
    Ve = np.linspace(1.4, 12, 14)
    re = de.jv(Ve)
    mu_low = egdm_pasveer(T, 1e10, 0.0, sig, a, nu0=nu0)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].semilogx(V - r["Vbi"], ratio, "o-", c=C[0])
    ax[0].axhline(1, c="k", lw=0.8)
    ax[0].set_xlabel(r"$V-V_{bi}$ (V)")
    ax[0].set_ylabel(r"$J_{DD}/J_{MG}$")
    ax[0].set_title("constant $\\mu$: DD $\\to$ Mott-Gurney", fontsize=10)
    ax[1].loglog(Ve - re["Vbi"], re["J"], "o-", c=C[1], label="EGDM + $g_3$ (DD)")
    ax[1].loglog(Ve - re["Vbi"], mott_gurney(Ve - re["Vbi"], de.L, de.eps_r, mu_low), "--",
                 c="k", label=r"MG with $\mu(p\to0,F\to0)$")
    ax[1].set_xlabel(r"$V-V_{bi}$ (V)")
    ax[1].set_ylabel(r"$J$ (A m$^{-2}$)")
    ax[1].legend(fontsize=8)
    save(fig, "v5_sclc")
    out["v5"] = dict(V=V, Vbi=r["Vbi"], ratio=ratio, egdm_V=Ve, egdm_J=re["J"])
    print("V5", ratio[-1], flush=True)



# ---------------------------------------------------------------- V6
if "v6" in SECT:
    Gs = np.logspace(25, 28.5, 8)
    voc, vk, Jsc, FF = [], [], [], []
    for G in Gs:
        dv = Device(L=100e-9, N=161, Eg=1.2, phi_anode_p=0.1, phi_cathode_n=0.1,
                    mu_n=1e-8, mu_p=1e-8, G=G)
        Vv = np.linspace(-0.3, 1.0, 27)
        rr = dv.jv(Vv)
        m = pv_metrics(Vv, rr["J"])
        voc.append(m["Voc"]); Jsc.append(m["Jsc"]); FF.append(m["FF"])
        vk.append(koster_voc(1.2, 300, 2.5e26, q * 2e-8 / (eps0 * 3.5), G))
    voc, vk = np.array(voc), np.array(vk)
    slope = np.polyfit(np.log(Gs), voc, 1)[0] / (kB_eV * 300)
    fig, ax = plt.subplots()
    ax.semilogx(Gs, voc, "o", c=C[0], label="ORCHID drift-diffusion")
    ax.semilogx(Gs, vk, "-", c=C[1], label="Koster et al. APL 86, 123509 (2005)")
    ax.set_xlabel(r"generation rate $G$ (m$^{-3}$s$^{-1}$)")
    ax.set_ylabel(r"$V_{oc}$ (V)")
    ax.text(0.05, 0.85, f"slope = {slope:.3f} $k_BT/q$", transform=ax.transAxes)
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "v6_voc")
    out["v6"] = dict(G=Gs, Voc=voc, Voc_koster=vk, slope_kT=slope, Jsc=Jsc, FF=FF,
                     max_abs_dev_mV=1e3 * np.max(np.abs(voc - vk)))
    print("V6", slope, out["v6"]["max_abs_dev_mV"], flush=True)



# ---------------------------------------------------------------- V12 OPV
if "opv" in SECT:
    dv = Device(L=100e-9, N=161, Eg=1.2, phi_anode_p=0.1, phi_cathode_n=0.1,
                mu_n=1e-8, mu_p=1e-8, langevin_prefactor=0.1)
    layers = [("ITO", 1.9 + 0.01j, 120e-9), ("PEDOT:PSS", 1.5 + 0.02j, 40e-9),
              ("blend", lorentz_nk, 100e-9), ("Al", 0.96 + 6.69j, 100e-9)]
    Vv = np.linspace(-0.4, 0.95, 28)
    o = simulate_opv(dv, Vv, layers=layers, active_index=2)
    dv2 = Device(L=100e-9, N=161, Eg=1.2, phi_anode_p=0.1, phi_cathode_n=0.1,
                 mu_n=1e-8, mu_p=1e-8, langevin_prefactor=0.1,
                 braun=dict(a_ct=1.5e-9, kf=1e6))
    dv2.Gx = o["G"].copy()
    r2 = dv2.jv(Vv)
    m2 = pv_metrics(Vv, r2["J"])
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].plot(dv.x * 1e9, o["G"] / 1e27, c=C[2])
    ax[0].set_xlabel("x (nm)")
    ax[0].set_ylabel(r"$G(x)$ ($10^{27}$ m$^{-3}$s$^{-1}$)")
    ax[0].set_title("transfer-matrix generation", fontsize=10)
    ax[1].plot(Vv, o["J"] / 10, c=C[0], label=f"free carriers, PCE={100*o['metrics']['PCE']:.1f}%")
    ax[1].plot(Vv, r2["J"] / 10, c=C[1], label=f"Braun-Onsager CT, PCE={100*m2['PCE']:.1f}%")
    ax[1].axhline(0, c="k", lw=0.6)
    ax[1].set_ylim(-20, 15)
    ax[1].set_xlabel("V (V)")
    ax[1].set_ylabel(r"$J$ (mA cm$^{-2}$)")
    ax[1].legend(fontsize=8)
    save(fig, "v12_opv")
    out["v12_opv"] = dict(free=o["metrics"], braun=m2)
    print("OPV", o["metrics"], m2, flush=True)



# ---------------------------------------------------------------- V12 OLED
if "oled" in SECT:
    Vo = np.linspace(2.6, 9, 12)
    res = {}
    for lab, kss in (("no annihilation", 0.0), (r"$k_{TT}=10^{-18}$ m$^3$/s", 1e-18),
                     (r"$k_{TT}=10^{-17}$ m$^3$/s", 1e-17)):
        dl = Device(L=100e-9, N=161, Eg=2.6, phi_anode_p=0.1, phi_cathode_n=0.1,
                    mu_n=1e-9, mu_p=1e-8)
        res[lab] = simulate_oled(dl, Vo, D_ex=4e-10, tau_ex=1e-6, kSS=kss, kSP=1e-19,
                                 eta_ST=1.0, qPL=0.9)
        print("OLED", lab, res[lab]["EQE"], flush=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    for k, (lab, rr) in enumerate(res.items()):
        ax[0].semilogx(rr["J"] / 10, 100 * rr["EQE"], "o-", c=C[k], label=lab)
    ax[0].set_xlabel(r"$J$ (mA cm$^{-2}$)")
    ax[0].set_ylabel("EQE (%)")
    ax[0].set_title(r"roll-off: triplets, $\tau$=1 $\mu$s, TTA", fontsize=10)
    ax[0].legend(fontsize=8)
    rr = res["no annihilation"]
    for k, i in enumerate((2, 6, 11)):
        prof = rr["S"][i]
        ax[1].plot(dl.x * 1e9, prof / prof.max(), c=C[k], label=f"V={Vo[i]:.1f} V")
    ax[1].set_xlabel("x (nm)  [anode $\\to$ cathode]")
    ax[1].set_ylabel("exciton density (norm.)")
    ax[1].set_title("emission zone ($\\mu_p=10\\mu_n$)", fontsize=10)
    ax[1].legend(fontsize=8)
    save(fig, "v12_oled")
    out["v12_oled"] = {k: dict(J=v["J"], EQE=v["EQE"], gamma_b=v["gamma_b"]) for k, v in res.items()}
dump("devices", out)

