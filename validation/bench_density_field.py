"""V2 + V3 - carrier-density and electric-field dependence.

V2: ME (linear response, Fermi-Dirac) and interacting KMC (site exclusion)
    mu(p)/mu(p->0) vs EGDM g1(p,T) (Pasveer 2005).
V3: independent-carrier KMC mu(F)/mu(F->0) vs EGDM g2(F,T).
"""
import time
import numpy as np
from common import plt, save, dump, C
from orchid.lattice import Lattice
from orchid.master_equation import me_linear_response
from orchid.kmc import mobility_interacting, mobility_independent
from orchid.models import egdm_g1, egdm_g2, egdm_mu0T
from orchid.constants import kB_eV

sigma, a = 0.1, 1.6e-9
t0 = time.time()
# ---------------------------------------------------------------- V2
dens = np.logspace(-6, -1, 11)
res2 = {}
absr = {}
for s in (3.0, 4.0, 5.0):
    T = sigma / (kB_eV * s)
    curves = []
    for seed in range(2):
        lat = Lattice(40, a=a, seed=200 + seed).set_disorder(sigma)
        curves.append([me_linear_response(lat, T, p, alpha=10.0)[0] for p in dens])
    c = np.mean(curves, 0)
    res2[s] = c / c[0]
    absr[s] = c / (egdm_mu0T(T, sigma, a ** 2 * 1e12 / sigma) * egdm_g1(dens, T, sigma))
    print("V2 ME", s, res2[s], time.time() - t0, flush=True)
# interacting KMC at shat=3
s = 3.0
T = sigma / (kB_eV * s)
lat = Lattice(24, a=a, seed=300).set_disorder(sigma)
mu_ref = me_linear_response(lat, T, 1e-7, alpha=10.0)[0]
kd = np.array([1e-3, 1e-2, 3e-2, 1e-1])
kmc2 = []
for p in kd:
    m = mobility_interacting(lat, T, 0.1 * sigma, p, n_eq=int(2e5), n_meas=int(6e5),
                             seed=5)
    kmc2.append(m / mu_ref)
    print("V2 KMC", p, kmc2[-1], me_linear_response(lat, T, p, alpha=10.0)[0] / mu_ref,
          time.time() - t0, flush=True)

fig, ax = plt.subplots()
pp = np.logspace(-6, -1, 100)
for k, s in enumerate((3.0, 4.0, 5.0)):
    T = sigma / (kB_eV * s)
    ax.loglog(pp, egdm_g1(pp, T, sigma), "-", c=C[k], label=rf"EGDM $g_1$, $\hat\sigma$={s:g}")
    ax.loglog(dens, res2[s], "o", c=C[k])
ax.loglog(kd, kmc2, "s", mfc="none", ms=9, c="k", label=r"KMC (exclusion), $\hat\sigma$=3")
ax.set_xlabel(r"carrier concentration $p\,a^3$")
ax.set_ylabel(r"$\mu(p)/\mu(p\to0)$")
ax.legend(fontsize=8, title="symbols: ORCHID ME", title_fontsize=8)
save(fig, "v2_density")

# ---------------------------------------------------------------- V3
res3 = {}
Fr = np.array([0.05, 0.2, 0.4, 0.6, 0.8, 1.0, 1.4])  # e F a / sigma
for s in (2.0, 3.0, 4.0):
    T = sigma / (kB_eV * s)
    lat = Lattice(30, a=a, seed=400).set_disorder(sigma)
    mus = []
    for f in Fr:
        m, e = mobility_independent(lat, T, f * sigma, n_carriers=48,
                                    n_eq=int(1e5 * s), n_meas=int(3e5 * s), alpha=10.0)
        mus.append(m)
    mus = np.array(mus)
    res3[s] = mus / mus[0]
    print("V3", s, res3[s], time.time() - t0, flush=True)
fig, ax = plt.subplots()
ff = np.linspace(0, 1.5, 100)
for k, s in enumerate((2.0, 3.0, 4.0)):
    T = sigma / (kB_eV * s)
    ax.semilogy(ff, egdm_g2(ff * sigma / a, T, sigma, a), "-", c=C[k],
                label=rf"EGDM $g_2$, $\hat\sigma$={s:g}")
    ax.semilogy(Fr, res3[s], "o", c=C[k])
ax.set_xlabel(r"reduced field $eFa/\sigma$")
ax.set_ylabel(r"$\mu(F)/\mu(F\to0)$")
ax.legend(fontsize=8, title="symbols: ORCHID KMC", title_fontsize=8)
save(fig, "v3_field")
dump("v2v3", dict(dens=dens, me_ratio={str(k): v for k, v in res2.items()},
                  abs_ratio_to_egdm={str(k): v for k, v in absr.items()},
                  kmc_dens=kd, kmc_ratio=kmc2, Fr=Fr,
                  field_ratio={str(k): v for k, v in res3.items()}))
