"""Zero-density mobility of a Gaussian-disordered lattice: ME vs KMC."""
import numpy as np
from orchid.lattice import Lattice
from orchid.master_equation import me_linear_response
from orchid.kmc import mobility_independent
from orchid.constants import kB_eV

sigma = 0.1                       # eV
T = sigma / (3.0 * kB_eV)         # sigma/kT = 3
lat = Lattice(30, a=1.6e-9, seed=1).set_disorder(sigma)
mu_me, info = me_linear_response(lat, T, 1e-7, alpha=10.0)
mu_kmc, err = mobility_independent(lat, T, 0.05 * sigma, alpha=10.0)
print(f"ME  : {mu_me * 1e4:.3e} cm2/Vs")
print(f"KMC : {mu_kmc * 1e4:.3e} +- {err * 1e4:.1e} cm2/Vs")
