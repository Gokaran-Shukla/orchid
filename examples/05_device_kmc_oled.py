"""3D device KMC of a three-layer OLED; voltages run in parallel
(processes here, MPI ranks under `mpiexec -n R python 05_device_kmc_oled.py`)."""
import numpy as np
from orchid.device_kmc import DeviceKMC, Layer
from orchid.parallel import parallel_map, root_print

STACK = [Layer(10, HOMO=-5.2, LUMO=-2.0, name="HTL"),
         Layer(10, HOMO=-5.4, LUMO=-2.7, name="EML"),
         Layer(10, HOMO=-6.0, LUMO=-2.8, name="ETL")]


def point(V):
    dev = DeviceKMC(STACK, Lx=20, Ly=20, W_anode=5.1, W_cathode=2.9, coulomb="full", seed=1)
    r = dev.run(V, n_eq=100000, n_meas=300000)
    return V, r["J"], r["gamma_b"], int(np.argmax(r["rec_profile"]))


if __name__ == "__main__":
    for V, J, gb, zpk in parallel_map(point, [3.0, 4.0, 5.0], workers=2):
        root_print(f"V={V:.1f} V  J={J:.3e} A/m2  gamma_b={gb:.2f}  recombination peak at z={zpk}")
