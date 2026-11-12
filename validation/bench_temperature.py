"""V1 - zero-density, zero-field mobility vs temperature.

Compares master equation (L=50, 2 disorder samples) and independent-carrier
KMC (L=30) with the EGDM parametrisation mu0(T) = mu0 c1 exp(-c2 sigma^2)
(Pasveer et al. PRL 94, 206601 (2005): c1 = 1.8e-9, c2 = 0.42 for alpha a = 10,
nearest-neighbour cutoff sqrt(3) a). c1, c2 are re-extracted from our data by
exhaustive grid search.
"""
import time
import numpy as np
from common import plt, save, dump, C
from orchid.lattice import Lattice
from orchid.master_equation import me_linear_response
from orchid.kmc import mobility_independent
from orchid.models import egdm_mu0T
from orchid.constants import kB_eV
from orchid.search import grid_search

sigma, a, nu0 = 0.1, 1.6e-9, 1e12
mu0 = a ** 2 * nu0 / sigma
shats = np.array([1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0])
me, me_sd = [], []
t0 = time.time()
for s in shats:
    T = sigma / (kB_eV * s)
    v = []
    for seed in range(3 if s < 4.6 else 2):
        lat = Lattice(50 if s < 4.6 else 40, a=a, seed=100 + seed).set_disorder(sigma)
        v.append(me_linear_response(lat, T, 1e-9, alpha=10.0)[0] / mu0)
    me.append(np.mean(v)); me_sd.append(np.std(v))
    print("ME", s, me[-1], time.time() - t0, flush=True)
kmc_s = np.array([2.0, 3.0, 4.0])
kmc, kmc_e = [], []
for s in kmc_s:
    T = sigma / (kB_eV * s)
    lat = Lattice(30, a=a, seed=7).set_disorder(sigma)
    m, e = mobility_independent(lat, T, 0.05 * sigma, n_carriers=48,
                                n_eq=int(1e5 * s), n_meas=int(4e5 * s), alpha=10.0)
    # same sample with ME for a paired comparison
    mme = me_linear_response(lat, T, 1e-9, alpha=10.0)[0]
    kmc.append(m / mu0); kmc_e.append(e / mu0)
    print("KMC", s, m / mu0, mme / mu0, flush=True)
me = np.array(me)


def obj(p):
    model = p["c1"] * np.exp(-p["c2"] * shats ** 2)
    return np.sqrt(np.mean((np.log(model) - np.log(me)) ** 2))


best, val, tab = grid_search(obj, {"c1": np.logspace(-9.5, -8.0, 151),
                                   "c2": np.linspace(0.30, 0.50, 201)})
print(best, val)
fig, ax = plt.subplots()
ss = np.linspace(1.2, 5.3, 100)
ax.semilogy(ss ** 2, egdm_mu0T(sigma / (kB_eV * ss), sigma, 1.0), "-", c=C[1],
            label="EGDM (Pasveer 2005)")
ax.semilogy(ss ** 2, best["c1"] * np.exp(-best["c2"] * ss ** 2), "--", c=C[0],
            label=f"fit: $c_1$={best['c1']:.2e}, $c_2$={best['c2']:.3f}")
ax.errorbar(shats ** 2, me, yerr=me_sd, fmt="o", c=C[0], label="ORCHID master eq.")
ax.errorbar(kmc_s ** 2, kmc, yerr=kmc_e, fmt="s", mfc="none", c=C[2], ms=8,
            label="ORCHID KMC")
ax.set_xlabel(r"$\hat\sigma^2=(\sigma/k_BT)^2$")
ax.set_ylabel(r"$\mu/\mu_0$,  $\mu_0=a^2\nu_0 e/\sigma$")
ax.legend(fontsize=8)
save(fig, "v1_mobility_temperature")
dump("v1", dict(shat=shats, me=me, me_sd=me_sd, kmc_shat=kmc_s, kmc=kmc,
                kmc_err=kmc_e, fit=best, fit_rmse_ln=val,
                pasveer=dict(c1=1.8e-9, c2=0.42),
                ratio_to_egdm=me / egdm_mu0T(sigma / (kB_eV * shats), sigma, 1.0)))
