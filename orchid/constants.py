"""Physical constants (SI) and unit helpers used throughout ORCHID.

Conventions
-----------
* Energies passed to user-facing functions are in **eV**.
* Lengths are in **metres** unless a function says otherwise (lattice codes
  work in units of the lattice constant ``a``).
* Mobilities are returned in **m^2 / (V s)**; use :data:`M2VS_TO_CM2VS` to
  convert to the customary cm^2/(V s).
"""
import numpy as np

q = 1.602176634e-19          # elementary charge [C]
kB = 1.380649e-23            # Boltzmann constant [J/K]
kB_eV = 8.617333262e-5       # Boltzmann constant [eV/K]
eps0 = 8.8541878128e-12      # vacuum permittivity [F/m]
h = 6.62607015e-34           # Planck constant [J s]
hbar = h / (2 * np.pi)       # reduced Planck constant [J s]
hbar_eVs = 6.582119569e-16   # reduced Planck constant [eV s]
c0 = 299792458.0             # speed of light [m/s]

M2VS_TO_CM2VS = 1.0e4        # m^2/Vs -> cm^2/Vs


def thermal_voltage(T):
    """k_B T / q in volts."""
    return kB_eV * T
