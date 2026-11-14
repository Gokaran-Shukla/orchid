"""V25 - removing the time-window bias of parallel sublattice KMC (R25)."""
import numpy as np
from common import plt, save, dump, C
from orchid.lattice import Lattice
from orchid.kmc import mobility_interacting
from orchid.sublattice_kmc import mobility_sublattice_extrapolated
from orchid.constants import kB_eV
from orchid.parallel import parallel_map

T = 300.0
s = 2.5 * kB_eV * T
lat = Lattice(16, a=1.6e-9, seed=3).set_disorder(s)
p = 0.03


def serial(seed):
    return mobility_interacting(lat, T, 0.05 * s, p, n_eq=100000, n_meas=800000, seed=seed)


def extrap(factors):
    return mobility_sublattice_extrapolated(lat, T, 0.05 * s, p, factors=factors,
                                            sim_time_factor=150000, seeds=(1, 2, 3, 4))


if __name__ == "__main__":
    ser = np.array(parallel_map(serial, [1, 2, 3, 4, 5, 6], workers=2))
    mu_s, mu_se = ser.mean(), ser.std(ddof=1) / np.sqrt(len(ser))
    res = parallel_map(extrap, [(20.0, 10.0, 5.0), (10.0, 5.0, 2.5)], workers=2)
    big, small = res
    single10 = big[1]["mu"][1] / mu_s
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for k, (mu0, inf), lab in ((0, big, r"windows 20,10,5 $\tau_0$"),
                               (1, small, r"windows 10,5,2.5 $\tau_0$")):
        x = inf["taus"] / inf["tau0"]
        ax.errorbar(x, inf["mu"] / mu_s, yerr=inf["mu_err"] / mu_s, fmt="os"[k], c=C[k],
                    label=lab)
        xx = np.linspace(0, x.max(), 50)
        ax.plot(xx, np.polyval(inf["coef"][::-1], xx) / mu_s, "-", c=C[k], lw=1)
        ax.errorbar([0], [mu0 / mu_s], yerr=[inf["err"] / mu_s], fmt="*", ms=11, c=C[k])
    ax.axhspan(1 - mu_se / mu_s, 1 + mu_se / mu_s, color="grey", alpha=0.3,
               label="serial Fenwick KMC (1 s.e.)")
    ax.set_xlabel(r"time window $\tau/\tau_0$")
    ax.set_ylabel(r"$\mu_{sub}/\mu_{serial}$")
    ax.set_title(r"sublattice KMC, $\hat\sigma$=2.5, $p a^3$=0.03", fontsize=10)
    ax.legend(fontsize=7)
    save(fig, "v25_sublattice_extrap")
    out = dict(mu_serial=mu_s, mu_serial_sem=mu_se, single_tau10_ratio=single10,
               extrap_big_ratio=big[0] / mu_s, extrap_big_err=big[1]["err"] / mu_s,
               extrap_small_ratio=small[0] / mu_s, extrap_small_err=small[1]["err"] / mu_s,
               windows_big=big[1]["mu"] / mu_s, windows_small=small[1]["mu"] / mu_s)
    dump("v25", out)
    print(out)
