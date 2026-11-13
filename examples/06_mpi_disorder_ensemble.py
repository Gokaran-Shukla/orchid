"""Disorder-averaged master-equation mobility on many MPI ranks.

    mpiexec -n 8 python 06_mpi_disorder_ensemble.py
"""
from orchid.lattice import Lattice
from orchid.master_equation import me_linear_response
from orchid.parallel import ensemble, root_print, size
from orchid.constants import kB_eV

SIGMA, T = 0.1, 0.1 / (3 * kB_eV)


def sample(seed):
    lat = Lattice(40, a=1.6e-9, seed=seed).set_disorder(SIGMA)
    return me_linear_response(lat, T, 1e-4, alpha=10.0)[0]


if __name__ == "__main__":
    mu, se, _ = ensemble(sample, range(16))
    root_print(f"{size()} rank(s): mu = {mu*1e4:.3e} +- {se*1e4:.1e} cm2/Vs")
