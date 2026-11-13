"""V15 - code-to-code comparison with independent published software.

1. Drift-diffusion vs NIST **Sesame** (Gaudreau et al., J. Appl. Phys. 125,
   (2019); github.com/usnistgov/sesame). Same Poisson/continuity physics,
   SRH recombination, Schottky contacts; three device cases.
   Run:  python bench_published_software.py /path/to/sesame-checkout
2. Transfer-matrix optics vs **tmm** (S. J. Byrnes, arXiv:1603.02720;
   pypi.org/project/tmm): R, T, per-layer absorption and absorption
   profile of an ITO/PEDOT/blend/Al stack across the visible.
"""
import json
import os
import subprocess
import sys
import tempfile
import numpy as np
from common import plt, save, dump, C, HERE
from orchid.drift_diffusion import Device
from orchid.optics import tmm_field, lorentz_nk

sesame_path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nist_sesame"
V = np.linspace(0.0, 1.0, 41)
base = dict(L_m=100e-9, N=201, beta=3.0, T=300.0, Nc_m3=2.5e26, Nv_m3=2.5e26, Eg=1.2,
            chi=3.9, eps_r=3.5, tau=1e-6, V=V.tolist())
cases = {
    "dark, symmetric": dict(base, mu_n_m2=1e-8, mu_p_m2=1e-8, phi_a=0.1, phi_c=0.1, G_m3s=0.0),
    "light, symmetric": dict(base, mu_n_m2=1e-8, mu_p_m2=1e-8, phi_a=0.1, phi_c=0.1, G_m3s=1e27),
    "light, unbalanced, 0.3 eV barriers": dict(base, mu_n_m2=1e-9, mu_p_m2=1e-8, phi_a=0.3,
                                               phi_c=0.3, G_m3s=1e27),
}
orch = {}
for name, c in cases.items():
    Nt = 1e21
    Cc = 1.0 / (c["tau"] * Nt)
    d = Device(L=c["L_m"], N=c["N"], grid_beta=c["beta"], T=c["T"], Eg=c["Eg"], Nc=c["Nc_m3"],
               Nv=c["Nv_m3"], eps_r=c["eps_r"], mu_n=c["mu_n_m2"], mu_p=c["mu_p_m2"],
               phi_anode_p=c["phi_a"], phi_cathode_n=c["phi_c"], G=c["G_m3s"],
               recombination=False, srh=dict(Nt=Nt, Et=c["Eg"] / 2, Cn=Cc, Cp=Cc, charged=False))
    orch[name] = d.jv(V)["J"]
with tempfile.TemporaryDirectory() as td:
    cj = os.path.join(td, "cases.json")
    oj = os.path.join(td, "out.json")
    json.dump(cases, open(cj, "w"))
    subprocess.run([sys.executable, "-I", os.path.join(HERE, "_sesame_runner.py"), sesame_path,
                    cj, oj], check=True)
    ses = json.load(open(oj))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
dev = {}
for k, name in enumerate(cases):
    Js = np.array(ses[name]["J_A_m2"])
    Jo = orch[name]
    from orchid.drift_diffusion import pv_metrics
    if cases[name]["G_m3s"] > 0:
        # illuminated: error relative to Jsc, plus Voc difference
        mo, ms = pv_metrics(V, Jo), pv_metrics(V, Js)
        dev[name] = dict(max_dJ_over_Jsc=float(np.max(np.abs(Jo - Js)) / ms["Jsc"]),
                         dVoc_mV=1e3 * (mo["Voc"] - ms["Voc"]), Voc=ms["Voc"],
                         dFF=mo["FF"] - ms["FF"])
    else:
        m = np.abs(Js) > 1e-6 * np.abs(Js).max()
        dev[name] = dict(max_rel_dev=float(np.max(np.abs(Jo[m] - Js[m]) / np.abs(Js[m]))))
    ax[0].semilogy(V, np.abs(Js), "-", c=C[k], lw=3, alpha=0.35)
    ax[0].semilogy(V, np.abs(Jo), "o", ms=3, c=C[k], label=f"{name}")
ax[0].set_xlabel("V (V)")
ax[0].set_ylabel(r"$|J|$ (A m$^{-2}$)")
ax[0].set_title("lines: NIST Sesame, symbols: ORCHID", fontsize=10)
ax[0].legend(fontsize=7)
print("Sesame max rel dev", dev)

# ------------------------------------------------------------------ tmm
import tmm
lams = np.linspace(350e-9, 900e-9, 111)
errR, errA, errprof = [], [], []
Ar_o, Ar_t = [], []
for lam in lams:
    nb = lorentz_nk(lam)
    layers = [("ITO", 1.9 + 0.01j, 120e-9), ("PEDOT", 1.5 + 0.02j, 40e-9),
              ("blend", nb, 100e-9), ("Al", 0.96 + 6.69j, 100e-9)]
    ro = tmm_field(layers, lam, n_in=1.5 + 0j, n_out=1.0 + 0j, npts=401)
    n_list = [1.5] + [l[1] for l in layers] + [1.0]
    d_list = [np.inf] + [l[2] * 1e9 for l in layers] + [np.inf]
    rt = tmm.coh_tmm("s", n_list, d_list, 0.0, lam * 1e9)
    At = tmm.absorp_in_each_layer(rt)[1:-1]
    errR.append(abs(ro["R"] - rt["R"]))
    errA.append(np.max(np.abs(ro["A"] - At)))
    Ar_o.append(ro["A"][2])
    Ar_t.append(At[2])
    # absorption profile in the blend
    zs = ro["x"][2] * 1e9                                                    # nm in layer 3
    a_t = np.array([tmm.position_resolved(3, z, rt)["absor"] for z in zs])   # 1/nm
    alpha = 4 * np.pi * nb.imag / lam
    a_o = alpha * nb.real * ro["E2"][2] / 1.5 * 1e-9                          # 1/nm
    errprof.append(np.max(np.abs(a_o - a_t)) / np.max(a_t))
ax[1].plot(lams * 1e9, Ar_t, "-", c=C[1], lw=3, alpha=0.4, label="tmm (Byrnes)")
ax[1].plot(lams * 1e9, Ar_o, "o", ms=3, c=C[0], label="ORCHID")
ax[1].set_xlabel(r"$\lambda$ (nm)")
ax[1].set_ylabel("absorptance of 100 nm blend")
ax[1].set_title("transfer matrix vs tmm package", fontsize=10)
ax[1].legend(fontsize=8)
save(fig, "v15_published_software")
res = dict(sesame_max_rel_dev=dev, tmm_max_abs_err_R=float(max(errR)),
           tmm_max_abs_err_A=float(max(errA)), tmm_max_rel_err_profile=float(max(errprof)))
dump("published_software", res)
print(res)
