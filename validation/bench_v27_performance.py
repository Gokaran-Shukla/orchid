"""V27 - performance benchmark of every engine (events or solves per second).

Records throughput on the current machine so regressions show up in
BENCHMARKS.md. Absolute numbers depend on the hardware; the file records
CPU count and versions."""
import os
import platform
import time
import numpy as np
import numba
from common import dump
import orchid
from orchid.lattice import Lattice
from orchid.kmc import mobility_independent, mobility_interacting
from orchid.master_equation import me_linear_response
from orchid.drift_diffusion import Device
from orchid.device_kmc import DeviceKMC, Layer
from orchid.coupled_kmc import CoupledDeviceKMC
from orchid.sublattice_kmc import mobility_sublattice
from orchid.constants import kB_eV


def timed(f, *a, reps=1, **k):
    f(*a, **k)                                     # JIT warm-up
    t0 = time.perf_counter()
    for _ in range(reps):
        f(*a, **k)
    return (time.perf_counter() - t0) / reps


T = 300.0
s = 3 * kB_eV * T
lat = Lattice(30, seed=1).set_disorder(s)
res = {}
nc, ne = 64, 200000
dt = timed(mobility_independent, lat, T, 0.05 * s, n_carriers=nc, n_eq=ne, n_meas=ne)
res["kmc_independent_hops_per_s"] = nc * 2 * ne / dt
dt = timed(mobility_interacting, lat, T, 0.05 * s, 0.01, n_eq=50000, n_meas=300000)
res["kmc_interacting_events_per_s"] = 350000 / dt
lat16 = Lattice(16, seed=1).set_disorder(s)
dt = timed(mobility_sublattice, lat16, T, 0.05 * s, 0.01, B=4, n_cycles=2000, n_eq=100)
res["sublattice_cycles_per_s"] = 2100 / dt
lat40 = Lattice(40, seed=1).set_disorder(s)
res["me_linear_response_64k_sites_s"] = timed(me_linear_response, lat40, T, 1e-6)
dev = Device(L=100e-9, N=201, T=300, Eg=1.2, mu_n=1e-7, mu_p=1e-7, G=1e27)
res["dd_jv_21pts_s"] = timed(dev.jv, np.linspace(0, 0.8, 21))
STACK = [Layer(8, HOMO=-5.2, LUMO=-2.0), Layer(10, HOMO=-5.4, LUMO=-2.7),
         Layer(8, HOMO=-6.0, LUMO=-2.8)]
d = DeviceKMC(STACK, Lx=16, Ly=16, W_anode=5.1, W_cathode=2.9, coulomb="full", seed=2)
dt = timed(d.run, 5.0, n_eq=20000, n_meas=200000, mf_update=300)
res["device_kmc_events_per_s"] = 220000 / dt
d = CoupledDeviceKMC(STACK, Lx=16, Ly=16, W_anode=5.1, W_cathode=2.9, coulomb="full", seed=2)
dt = timed(d.run, 5.0, n_eq=20000, n_meas=200000, mf_update=300)
res["coupled_kmc_events_per_s"] = 220000 / dt
res["machine"] = dict(cpus=os.cpu_count(), python=platform.python_version(),
                      numpy=np.__version__, numba=numba.__version__, orchid=orchid.__version__,
                      processor=platform.processor() or platform.machine())
dump("v27", res)
for k, v in res.items():
    print(f"{k:38s} {v}")
