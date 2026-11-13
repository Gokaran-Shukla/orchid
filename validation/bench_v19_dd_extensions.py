"""V19 - multilayer DD vs NIST Sesame (R17) and small-signal impedance (R18)."""
import json, os, subprocess, sys, tempfile
import numpy as np
from common import plt, save, dump, C, HERE
from orchid.drift_diffusion import Device, MultilayerDevice
from orchid.smallsignal import admittance
from orchid.constants import eps0

ses = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nist_sesame"
V = np.linspace(0, 1.2, 25)
layers = [dict(d=50e-9, Eg=1.9, chi=3.0, eps_r=3.5, mu_n=1e-9, mu_p=1e-8, Nc=2.5e26, Nv=2.5e26),
          dict(d=50e-9, Eg=1.9, chi=3.8, eps_r=3.8, mu_n=1e-8, mu_p=1e-9, Nc=2.5e26, Nv=2.5e26)]
res = {}
for G in (0.0, 1e27):
    for l in layers:
        l["G"] = G
    m = MultilayerDevice(layers, W_anode=4.8, W_cathode=3.9, nodes_per_layer=80, recombination=False,
                         srh=dict(Nt=1e21, midgap=True, Cn=1e-15, Cp=1e-15, charged=False))
    Jo = m.jv(V)["J"]
    case = dict(layers=layers, x_m=m.x.tolist(), T=300.0, tau=1e-6, W_anode=4.8, W_cathode=3.9,
                G_m3s=G, V=V.tolist())
    with tempfile.TemporaryDirectory() as td:
        cj, oj = os.path.join(td, "c.json"), os.path.join(td, "o.json")
        json.dump({"h": case}, open(cj, "w"))
        subprocess.run([sys.executable, "-I", os.path.join(HERE, "_sesame_runner.py"), ses, cj, oj],
                       check=True, capture_output=True)
        Js = np.array(json.load(open(oj))["h"]["J_A_m2"])
    res[G] = (Jo, Js)
dev = {str(G): float(np.max(np.abs(a - b)) / np.max(np.abs(b))) for G, (a, b) in res.items()}
print("hetero vs Sesame", dev)
# impedance of trap-free SCLC
mu, L = 1e-8, 100e-9
spec = {}
for Vb in (20.0, 80.0):
    d = Device(L=L, N=201, Eg=2.4, phi_anode_p=0.0, phi_cathode_n=1.0, mu_n=1e-12, mu_p=mu,
               recombination=False)
    tdc = 4 / 3 * L ** 2 / (mu * (Vb - 1.4))
    f = np.logspace(-2, 1.3, 70) / tdc
    r = admittance(d, Vb, f)
    spec[Vb] = (f * tdc, r["C"] / (eps0 * 3.5 / L), -(2 * np.pi * f) * (r["C"] - eps0 * 3.5 / L))
ins = Device(L=L, N=121, Eg=3.0, phi_anode_p=1.5, phi_cathode_n=1.5, mu_n=1e-8, mu_p=1e-8,
             recombination=False)
Ci = admittance(ins, 0.0, [1e2, 1e4, 1e6])["C"] / (eps0 * 3.5 / L)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, (G, (a, b)) in enumerate(res.items()):
    ax[0].plot(V, b, "-", c=C[k], lw=3, alpha=0.35)
    ax[0].plot(V, a, "o", ms=3, c=C[k], label=("dark" if G == 0 else "illuminated"))
ax[0].set_xlabel("V (V)")
ax[0].set_ylabel(r"$J$ (A m$^{-2}$)")
ax[0].set_ylim(-15, 30)
ax[0].set_title("D/A heterojunction: lines Sesame, symbols ORCHID", fontsize=10)
ax[0].legend(fontsize=8)
for k, Vb in enumerate((20.0, 80.0)):
    x, c, b = spec[Vb]
    ax[1].semilogx(x, c, c=C[k], label=f"C/C$_{{geo}}$, {Vb:.0f} V")
ax[1].axhline(0.75, c="k", ls=":", lw=1)
ax[1].set_xlabel(r"$f\,\tau_{dc}$")
ax[1].set_ylabel(r"$C/C_{geo}$")
ax[1].set_title("SCLC impedance (dotted: 3/4)", fontsize=10)
ax[1].legend(fontsize=8)
save(fig, "v19_multilayer_impedance")
out = dict(hetero_rel_dev=dev, insulator_C=Ci.tolist())
for Vb, (x, c, b) in spec.items():
    k = int(np.argmax(b))
    cc = np.polyfit(np.log(x[k - 1:k + 2]), b[k - 1:k + 2], 2)
    out[f"V{int(Vb)}"] = dict(C0_over_Cgeo=float(c[0]), fmax_tdc=float(np.exp(-cc[1] / (2 * cc[0]))))
dump("v19", out)
print(out)
