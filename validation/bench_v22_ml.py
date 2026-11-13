"""V22 - ML surrogate + Bayesian inversion (R22)."""
import numpy as np
from common import plt, save, dump, C
from orchid.ml import GPSurrogate, generate_mobility_dataset, bayes_grid
from orchid.models import gdm_bassler

sh = np.linspace(1.5, 4.5, 9)
lp = np.linspace(-6, -1.3, 9)
X, y = generate_mobility_dataset(sh, lp, L=24, workers=2)
rng = np.random.default_rng(0)
idx = rng.permutation(len(y))
tr, te = idx[:55], idx[55:]
gp = GPSurrogate().fit(X[tr], y[tr])
pr, sd = gp.predict(X[te], True)
rmse = float(np.sqrt(np.mean((pr - y[te]) ** 2)))
TT, FF = np.meshgrid([200, 240, 280, 320.0], [1e4, 1e5, 4e5, 1e6], indexing="ij")
truth = dict(sigma=0.095, lmu=np.log10(3e-2), Sigma=2.5)
data = gdm_bassler(TT, FF, 0.095, 3e-2, Sigma=2.5) * np.exp(rng.normal(0, 0.05, TT.shape))
r = bayes_grid(lambda p: gdm_bassler(TT, FF, p["sigma"], 10 ** p["lmu"], Sigma=p["Sigma"]), data,
               dict(sigma=np.linspace(0.085, 0.105, 61), lmu=np.linspace(-2, -1, 61),
                    Sigma=np.linspace(1.6, 3.4, 46)), 0.05 / np.log(10))
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].errorbar(y[te], pr, yerr=2 * sd, fmt="o", c=C[0], ms=4)
lim = [y.min() - 0.2, y.max() + 0.2]
ax[0].plot(lim, lim, "k-", lw=0.8)
ax[0].set_xlabel(r"master equation $\log_{10}(\mu/\mu_0)$")
ax[0].set_ylabel("GP surrogate")
ax[0].set_title(f"held-out RMSE = {rmse:.3f} decades", fontsize=10)
ax[1].plot(r["axes"][0] * 1e3, r["marginals"]["sigma"], c=C[0])
ax[1].axvline(95, c="k", ls="--", lw=1, label="truth")
lo, hi = r["ci95"]["sigma"]
ax[1].axvspan(lo * 1e3, hi * 1e3, color=C[0], alpha=0.15, label="95 % credible")
ax[1].set_xlabel(r"$\sigma$ (meV)")
ax[1].set_ylabel("posterior")
ax[1].set_title("Bayesian GDM inversion (exhaustive grid)", fontsize=10)
ax[1].legend(fontsize=8)
save(fig, "v22_ml")
cover = {k: bool(r["ci95"][k][0] <= truth[k] <= r["ci95"][k][1]) for k in truth}
dump("v22", dict(rmse=rmse, mean_pred_std=float(sd.mean()), map=r["map"], ci95=r["ci95"], covered=cover))
print(rmse, r["ci95"], cover)
