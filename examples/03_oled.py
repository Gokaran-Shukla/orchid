"""Single-layer OLED: J-V, emission zone, EQE roll-off."""
import numpy as np
from orchid.drift_diffusion import Device
from orchid.devices import simulate_oled

dev = Device(L=100e-9, Eg=2.6, mu_n=1e-9, mu_p=1e-8, phi_anode_p=0.1, phi_cathode_n=0.1)
out = simulate_oled(dev, np.linspace(2.6, 9, 12), eta_ST=1.0, qPL=0.9, kSS=1e-17, kSP=1e-18)
for V, J, e in zip(out["V"], out["J"], out["EQE"]):
    print(f"V={V:5.2f} V  J={J/10:10.3e} mA/cm2  EQE={100*e:5.2f} %")
