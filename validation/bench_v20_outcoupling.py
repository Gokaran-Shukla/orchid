"""V20 - dipole emission (R20): perfect-mirror CPS check and OLED mode analysis
with bundled refractiveindex.info data (ITO, Al)."""
import numpy as np
from common import plt, save, dump, C
from orchid.optics import dipole_emission, cps_perfect_mirror
from orchid.materials import nk, am15g

lam, n = 500e-9, 1.7
ds = np.linspace(10e-9, 400e-9, 40)
pv, ph = [], []
for d in ds:
    pv.append(dipole_emission(lam, [(n, 0)], (n, d), [(1e6j, 0)], 0, theta=1, u_max=60, n_ev=40000)["F"])
    ph.append(dipole_emission(lam, [(n, 0)], (n, d), [(1e6j, 0)], 0, theta=0, u_max=60, n_ev=40000)["F"])
x = 2 * 2 * np.pi * n / lam * ds
Fv, Fh = cps_perfect_mirror(x)
err = max(np.max(np.abs(np.array(pv) - Fv)), np.max(np.abs(np.array(ph) - Fh)))
lam2 = 520e-9
etl = np.linspace(20e-9, 200e-9, 37)
modes = []
for t in etl:
    r = dipole_emission(lam2, [(1.75, 30e-9), (nk("ITO")(lam2), 100e-9), (nk("BK7")(lam2).real, 0)],
                        (1.75, t), [(nk("Al")(lam2), 0)], 0.0, theta=1 / 3, u_max=4, n_ev=20000)
    modes.append([r["eta_air"], r["eta_substrate"] - r["eta_air"], r["guided"], r["evanescent"]])
modes = np.array(modes)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].plot(ds * 1e9, Fv, "-", c=C[0], lw=3, alpha=0.35)
ax[0].plot(ds * 1e9, pv, "o", ms=3, c=C[0], label="vertical")
ax[0].plot(ds * 1e9, Fh, "-", c=C[1], lw=3, alpha=0.35)
ax[0].plot(ds * 1e9, ph, "s", ms=3, c=C[1], label="horizontal")
ax[0].set_xlabel("distance to perfect mirror (nm)")
ax[0].set_ylabel("normalised decay rate")
ax[0].set_title("lines: Chance-Prock-Silbey, symbols: ORCHID", fontsize=10)
ax[0].legend(fontsize=8)
lab = ["air", "substrate", "waveguide", "evanescent/SPP"]
ax[1].stackplot(etl * 1e9, modes.T, labels=lab, colors=C[:4], alpha=0.8)
ax[1].set_xlabel("emitter-cathode distance (nm)")
ax[1].set_ylabel("fraction of dissipated power")
ax[1].set_title("bottom OLED: BK7/ITO/org/Al, 520 nm", fontsize=10)
ax[1].legend(fontsize=7, loc="upper right")
save(fig, "v20_outcoupling")
l, E = am15g()
k = int(np.argmax(modes[:, 0]))
dump("v20", dict(cps_max_abs_err=float(err), best_distance_nm=float(etl[k] * 1e9),
                 eta_air_max=float(modes[k, 0]), am15g_integral=float(np.trapezoid(E, l))))
print(err, etl[k], modes[k])
