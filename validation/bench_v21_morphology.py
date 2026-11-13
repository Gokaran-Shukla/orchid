"""V21 - morphology + DFT couplings (R21): DIPRO vs dimer splitting and
mobility on an amorphous packing vs the lattice GDM."""
import numpy as np
from common import plt, save, dump, C
from orchid.morphology import dipro_pyscf, random_packing, dipolar_site_energies, MolecularGraph
from orchid.lattice import Lattice
from orchid.master_equation import me_linear_response
from orchid.constants import kB_eV

eth = [("C", (0, 0, 0.667)), ("C", (0, 0, -0.667)), ("H", (0, 0.923, 1.232)),
       ("H", (0, -0.923, 1.232)), ("H", (0, 0.923, -1.232)), ("H", (0, -0.923, -1.232))]
dd = np.linspace(3.4, 6.0, 14)
J, S = [], []
for d in dd:
    r = dipro_pyscf(eth, [(s, (x + d, y, z)) for s, (x, y, z) in eth])
    J.append(abs(r["J_eff"]))
    S.append(abs(r["splitting"]) / 2)
J, S = np.array(J), np.array(S)
beta = -np.polyfit(dd, np.log(J), 1)[0]
# amorphous morphology with dipolar disorder vs lattice with the same sigma
box = np.array([20e-9] * 3)
pos = random_packing(2000, box, 1.0e-9, seed=2)
E = dipolar_site_energies(pos, box, dipole_debye=4.0)
sig = E.std()
G = MolecularGraph(pos, box, r_cut=2.6e-9, energies=E)
Ts = np.array([200, 250, 300, 350, 400.0])
mu_g = [me_linear_response(G, T, 1e-6, alpha=10 * 1.0)[0] for T in Ts]
lat = Lattice(13, a=G.a, seed=2).set_disorder(sig)
mu_l = [me_linear_response(lat, T, 1e-6, alpha=10.0)[0] for T in Ts]
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].semilogy(dd, S * 1e3, "-", c=C[1], lw=3, alpha=0.4, label="half dimer splitting")
ax[0].semilogy(dd, J * 1e3, "o", c=C[0], label=r"DIPRO $J_{eff}$ (HF/STO-3G, PySCF)")
ax[0].set_xlabel(r"cofacial ethylene distance ($\AA$)")
ax[0].set_ylabel(r"$|J|$ (meV)")
ax[0].set_title(rf"$\beta$ = {beta:.2f} $\AA^{{-1}}$", fontsize=10)
ax[0].legend(fontsize=8)
ax[1].semilogy(1000 / Ts, mu_g, "o-", c=C[0], label=f"amorphous packing, dipolar ($\\sigma$={1e3*sig:.0f} meV)")
ax[1].semilogy(1000 / Ts, mu_l, "s--", c=C[1], label="cubic lattice, Gaussian, same $\\sigma$")
ax[1].set_xlabel("1000/T (1/K)")
ax[1].set_ylabel(r"$\mu$ (m$^2$/Vs)")
ax[1].set_title("morphology matters: correlated vs uncorrelated", fontsize=10)
ax[1].legend(fontsize=7)
save(fig, "v21_morphology")
dump("v21", dict(d=dd, J=J, half_split=S, max_rel=float(np.max(np.abs(J - S) / S)), beta=beta,
                 sigma=sig, mu_graph=mu_g, mu_lattice=mu_l))
print(np.max(np.abs(J - S) / S), beta, sig)
