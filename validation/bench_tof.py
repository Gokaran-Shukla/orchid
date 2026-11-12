"""V10 - simulated time-of-flight transients: non-dispersive -> dispersive
transport with increasing disorder (Scher-Montroll / Bässler phenomenology).

Plotted in the universal form I(t)/I(t_tr) vs t/t_tr. For non-dispersive
transport the current is flat with a sharp drop at t_tr; with disorder the
plateau tilts and the tail broadens.
"""
import numpy as np
from common import plt, save, dump, C
from orchid.lattice import Lattice
from orchid.kmc import time_of_flight
from orchid.constants import kB_eV

T = 300.0
Ls = 100
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
res = {}
for k, s in enumerate((0.0, 2.0, 3.0, 3.5)):
    sg = max(s * kB_eV * T, 1e-6)
    lat = Lattice(40, a=1.6e-9, seed=21).set_disorder(sg)
    F_red = 0.4 * kB_eV * T                 # eFa = 0.4 kT
    tg = np.logspace(-5, 6, 221)
    r = time_of_flight(lat, T, F_red, L_sample=Ls, n_carriers=1024, tgrid=tg, alpha=10.0)
    ttr = r["t_transit"]
    I = r["I"]
    Itr = np.interp(ttr, tg, I)
    m = I > 0
    ax[0].loglog(tg[m] / ttr, I[m] / Itr, c=C[k], label=rf"$\hat\sigma$={s:g}")
    ax[1].semilogx(tg / ttr, r["arrived"], c=C[k])
    res[s] = dict(t_transit=ttr, mu=Ls * lat.a / (ttr * F_red / lat.a))
    print(s, res[s], flush=True)
ax[0].set_xlim(1e-3, 30)
ax[0].set_ylim(1e-3, 1e2)
ax[0].set_xlabel(r"$t/t_{tr}$")
ax[0].set_ylabel(r"$I(t)/I(t_{tr})$")
ax[0].legend(fontsize=8)
ax[0].set_title(r"TOF photocurrent, $L$ = 100 $a$", fontsize=10)
ax[1].set_xlim(1e-2, 30)
ax[1].set_xlabel(r"$t/t_{tr}$")
ax[1].set_ylabel("collected fraction")
ax[1].set_title("arrival-time distribution broadens", fontsize=10)
save(fig, "v10_tof")
dump("tof", res)
