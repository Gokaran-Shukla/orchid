"""V23 - 2D OFET vs gradual-channel model (R19) and parallel sublattice KMC
vs serial interacting KMC (R23)."""
import numpy as np
from common import plt, save, dump, C
from orchid.dd2d import ofet
from orchid.lattice import Lattice
from orchid.kmc import mobility_interacting
from orchid.sublattice_kmc import mobility_sublattice
from orchid.constants import kB_eV

mu_in = 1e-6
d = ofet(mu_p=mu_in)
VGs = np.array([-4.0, -6.0, -8.0, -10.0, -12.0])
VD_lin = -0.2
Ilin = []
z = None
for VG in VGs:
    z = d.solve({"gate": VG, "drain": VD_lin}, z0=z)
    Ilin.append(-d.contact_current(z, {"gate": VG, "drain": VD_lin}, "drain"))
Ilin = np.array(Ilin)
slope = np.polyfit(VGs, Ilin, 1)
mu_lin = abs(slope[0]) * d.L_ch / (d.C_i * abs(VD_lin))
VT = -slope[1] / slope[0]
VDs = -np.linspace(0.25, 14, 16)
out = {}
for VG in (-6.0, -10.0):
    z = None
    I = []
    for VD in VDs:
        z = d.solve({"gate": VG, "drain": VD}, z0=z)
        I.append(-d.contact_current(z, {"gate": VG, "drain": VD}, "drain"))
    out[VG] = np.array(I)
Vov = lambda VG: abs(VG - VT)
gca = lambda VG, VD: mu_in * d.C_i / d.L_ch * np.where(np.abs(VD) < Vov(VG),
                                                       Vov(VG) * np.abs(VD) - VD ** 2 / 2, Vov(VG) ** 2 / 2)
mu_sat = 2 * d.L_ch / d.C_i * out[-10.0][-1] / Vov(-10.0) ** 2
# sublattice KMC
s = 0.1
T = s / (kB_eV * 2.5)
lat = Lattice(16, a=1.6e-9, seed=3).set_disorder(s)
W = lat.rates(T, (0.005, 0, 0), alpha=10.0)
tau0 = 1 / W.sum(1).max()
dens = np.array([1e-2, 3e-2, 1e-1])
ser, sub, sse, sbe = [], [], [], []
for p in dens:
    a_ = [mobility_interacting(lat, T, 0.05 * s, p, n_eq=100000, n_meas=800000, seed=k) for k in (1, 2, 3)]
    b_ = [mobility_sublattice(lat, T, 0.05 * s, p, B=4, tau=10 * tau0, n_cycles=15000, n_eq=1500, seed=k)[0]
          for k in (1, 2, 3)]
    ser.append(np.mean(a_)); sse.append(np.std(a_) / np.sqrt(3))
    sub.append(np.mean(b_)); sbe.append(np.std(b_) / np.sqrt(3))
    print(p, ser[-1], sub[-1], flush=True)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, VG in enumerate((-6.0, -10.0)):
    ax[0].plot(-VDs, out[VG] * 1e3, "o", c=C[k], ms=4, label=f"2D DD, $V_G$={VG:.0f} V")
    ax[0].plot(-VDs, gca(VG, VDs) * 1e3, "-", c=C[k], lw=1.2)
ax[0].set_xlabel(r"$-V_D$ (V)")
ax[0].set_ylabel(r"$-I_D/W$ (mA/m)")
ax[0].set_title(f"OFET: lines gradual channel ($V_T$={VT:.2f} V)", fontsize=10)
ax[0].legend(fontsize=8)
ax[1].errorbar(dens, ser, yerr=sse, fmt="o-", c=C[0], label="serial Fenwick KMC")
ax[1].errorbar(dens, sub, yerr=sbe, fmt="s--", c=C[1], mfc="none", label="parallel sublattice KMC")
ax[1].set_xscale("log")
ax[1].set_xlabel(r"$p\,a^3$")
ax[1].set_ylabel(r"$\mu$ (m$^2$/Vs)")
ax[1].set_title(r"interacting KMC, $\hat\sigma$=2.5", fontsize=10)
ax[1].legend(fontsize=8)
save(fig, "v23_ofet_sublattice")
dump("v23", dict(mu_lin_over_input=mu_lin / mu_in, mu_sat_over_input=mu_sat / mu_in, VT=VT,
                 sub_over_serial=(np.array(sub) / np.array(ser)).tolist()))
print(mu_lin / mu_in, mu_sat / mu_in, VT, np.array(sub) / np.array(ser))
