"""V17 - transient drift-diffusion and electrostatics.

(a) photo-CELIV transients; (b) mobility from the Juska formula vs ramp
voltage at t_max: it converges to the input mobility once qV(t_max) >> kT,
where diffusion becomes negligible (Juska et al., PRL 84, 4946 (2000));
(c) small-perturbation TPV decay vs the bulk Langevin lifetime 1/(2 gamma n)
(differs in thin devices because of electrode/contact charge; Kiermasch et
al., Energy Environ. Sci. 2018); (d) Ewald Madelung constant and image-charge
self energy vs the single-plane limit.
"""
import numpy as np
from common import plt, save, dump, C
from orchid.drift_diffusion import Device
from orchid.transient import celiv, tpv
from orchid.electrostatics import madelung_nacl, image_self_energy
from orchid.constants import q, eps0

mu = 1e-8
As = np.array([1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8])
cel = []
for A in As:
    t_est = 1e-7 * np.sqrt(2 / (3 * mu * A))
    d = Device(L=100e-9, N=161, Eg=2.0, phi_anode_p=1.0, phi_cathode_n=1.0, mu_n=1e-14,
               mu_p=mu, recombination=False, grid_beta=2.0)
    r = celiv(d, A=A, t_end=12 * t_est, n_t=300, G_pulse=1e30, t_pulse=1e-10)
    cel.append(r)
    print(A, r["mu_juska"] / mu, flush=True)
dtpv = Device(L=100e-9, N=101, Eg=1.2, phi_anode_p=0.1, phi_cathode_n=0.1, mu_n=1e-8,
              mu_p=1e-8, G=1e27)
rt = tpv(dtpv)
dtk = Device(L=300e-9, N=121, Eg=1.2, phi_anode_p=0.1, phi_cathode_n=0.1, mu_n=1e-8,
             mu_p=1e-8, G=1e28)
rt3 = tpv(dtk)
print("TPV", rt["tau"], rt["tau_langevin"], rt3["tau"], rt3["tau_langevin"], flush=True)
fig, ax = plt.subplots(1, 3, figsize=(12, 3.6))
for k, i in enumerate((0, 3, 6)):
    r = cel[i]
    ax[0].plot(r["t"] / r["t_max"], (r["J"] - r["j0"]) / r["dj"], c=C[k], label=f"A={As[i]:.0e} V/s")
ax[0].set_xlim(0, 4)
ax[0].set_ylim(-0.1, 1.1)
ax[0].set_xlabel(r"$t/t_{max}$")
ax[0].set_ylabel(r"$(j-j_0)/\Delta j$")
ax[0].set_title("photo-CELIV transients", fontsize=10)
ax[0].legend(fontsize=7)
Vt = np.array([A * r["t_max"] for A, r in zip(As, cel)])
ax[1].semilogx(Vt / 0.02585, [r["mu_juska"] / mu for r in cel], "o-", c=C[0], label="Juska")
ax[1].semilogx(Vt / 0.02585, [r["mu_lorrmann"] / mu for r in cel], "s--", c=C[1], mfc="none",
               label="Lorrmann-corrected")
ax[1].axhline(1, c="k", lw=0.8)
ax[1].set_xlabel(r"$qV(t_{max})/k_BT$")
ax[1].set_ylabel(r"$\mu_{CELIV}/\mu_{input}$")
ax[1].set_title("CELIV mobility extraction", fontsize=10)
ax[1].legend(fontsize=8)
for k, (r, lab) in enumerate(((rt, "100 nm, G=1e27"), (rt3, "300 nm, G=1e28"))):
    m = r["t"] > r["t"][5]
    ax[2].semilogy((r["t"][m] - r["t"][5]) / r["tau_langevin"], r["dV"][m] / r["dV"].max(), c=C[k],
                   label=f"{lab}: $\\tau/\\tau_L$={r['tau'] / r['tau_langevin']:.2f}")
ax[2].set_xlabel(r"$t/\tau_L$,  $\tau_L=1/(2\gamma n)$")
ax[2].set_ylabel(r"$\Delta V/\Delta V_{max}$")
ax[2].set_title("small-perturbation TPV", fontsize=10)
ax[2].legend(fontsize=7)
ax[2].set_ylim(1e-2, 1.2)
save(fig, "v17_transient")
z = np.logspace(-10, -8, 30)
dump("transient", dict(A=As, Vt=Vt, mu_ratio=[r["mu_juska"] / mu for r in cel],
                       tpv_100nm=dict(tau=rt["tau"], tau_L=rt["tau_langevin"]),
                       tpv_300nm=dict(tau=rt3["tau"], tau_L=rt3["tau_langevin"]),
                       madelung=madelung_nacl(),
                       image_rel_err=float(np.max(np.abs(image_self_energy(z, 1e-5, 3.5) /
                                                         (-q / (16 * np.pi * eps0 * 3.5 * z)) - 1)))))
