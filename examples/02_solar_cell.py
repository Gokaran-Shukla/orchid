"""Organic solar cell: transfer-matrix generation + drift-diffusion J-V."""
import numpy as np
from orchid.drift_diffusion import Device
from orchid.devices import simulate_opv
from orchid.optics import lorentz_nk

dev = Device(L=100e-9, Eg=1.2, mu_n=1e-8, mu_p=1e-8, phi_anode_p=0.1,
             phi_cathode_n=0.1, langevin_prefactor=0.1,
             braun=dict(a_ct=1.5e-9, kf=1e6))          # Braun-Onsager CT states
stack = [("ITO", 1.9 + 0.01j, 120e-9), ("PEDOT:PSS", 1.5 + 0.02j, 40e-9),
         ("blend", lorentz_nk, 100e-9), ("Al", 0.96 + 6.69j, 100e-9)]
res = simulate_opv(dev, np.linspace(-0.4, 0.95, 28), layers=stack, active_index=2)
print({k: round(float(v), 4) for k, v in res["metrics"].items()})
