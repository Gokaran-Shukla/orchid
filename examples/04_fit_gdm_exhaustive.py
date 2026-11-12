"""Exhaustive (zooming grid) extraction of GDM parameters from mu(T,F)."""
import numpy as np
from orchid.models import gdm_bassler
from orchid.search import zoom_search, log_rmse

T, F = np.meshgrid([200, 240, 280, 320.0], [1e4, 1e5, 4e5, 1e6], indexing="ij")
data = gdm_bassler(T, F, 0.09, 1e-2, Sigma=2.0)          # replace with measurements
best, err, _ = zoom_search(lambda p: log_rmse(gdm_bassler(T, F, p["sigma"], p["mu_inf"],
                                                          Sigma=p["Sigma"]), data),
                           {"sigma": (0.05, 0.15), "mu_inf": (1e-5, 1.0), "Sigma": (1.5, 4)},
                           n=15, levels=5, log=("mu_inf",))
print(best, err)
