"""V26 - MD trajectory import (R26): PBC-whole molecular centres, GRO
round trip, graph ME = lattice ME on a lattice "trajectory", and mobility
vs thermal displacement amplitude."""
import os
import tempfile
import numpy as np
from common import plt, save, dump, C
from orchid.lattice import Lattice
from orchid.master_equation import me_linear_response
from orchid.morphology import (write_gro, read_gro, molecule_centres, trajectory_mobility)

rng = np.random.default_rng(0)
RING = np.array([[np.cos(t), np.sin(t), 0.0] for t in np.arange(6) * np.pi / 3]) * 0.14e-9
NAMES = ["C", "C", "C", "C", "C", "C"]


def make_frame(centres, box):
    xyz, atom, resid = [], [], []
    for i, c in enumerate(centres):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        for p in RING @ Q.T:
            xyz.append((c + p) % box)
        atom += NAMES
        resid += [i + 1] * 6
    return dict(title="ORCHID synthetic", resid=np.array(resid), resname=["MOL"] * len(atom),
                atom=atom, xyz=np.array(xyz), box=np.asarray(box, float))


n, a = 12, 1.6e-9
lat = Lattice(n, a=a, seed=4).set_disorder(0.075)
box = np.array([n * a] * 3)
cent = lat.ijk * a          # molecules at i=0 straddle the periodic boundary
fr = make_frame(cent, box)
c, _ = molecule_centres(fr)
d = c - cent
d -= box * np.round(d / box)
exact_err = np.abs(d).max()
broken = int(np.sum(np.ptp(fr["xyz"].reshape(-1, 6, 3), axis=1).max(1) > box[0] / 2))
# GRO stores 0.001 nm: use off-grid centres so that rounding errors do not cancel
cent_g = (cent + rng.uniform(0, 1e-12, cent.shape)) % box
tmp = tempfile.mkdtemp()
write_gro(os.path.join(tmp, "t.gro"), [make_frame(cent_g, box)])
c2, _ = molecule_centres(read_gro(os.path.join(tmp, "t.gro"))[0])
d = c2 - cent_g
d -= box * np.round(d / box)
gro_err = np.abs(d).max()
Ts = [250.0, 300.0, 350.0]
mu_lat = np.array([me_linear_response(lat, T, 1e-6, alpha=10.0)[0] for T in Ts])
mu_trj = np.array([trajectory_mobility([fr], T, energies=lat.energy, r_cut=np.sqrt(3) * a * 1.0001,
                                       alpha=10.0)["mu_mean"] for T in Ts])
rel = np.abs(mu_trj / mu_lat - 1).max()
amps = np.array([0.0, 0.05, 0.1, 0.15, 0.2])
mus, sems = [], []
for A in amps:
    frames = [make_frame((cent + rng.normal(0, A * a, cent.shape)) % box, box) for _ in range(4)]
    r = trajectory_mobility(frames, 300.0, energies="gaussian", sigma=0.075, alpha=10.0,
                            r_cut=1.8 * a, seed=1)
    mus.append(r["mu_mean"])
    sems.append(r["mu_sem"])
mus, sems = np.array(mus), np.array(sems)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].semilogy(1000 / np.array(Ts), mu_lat, "-", c=C[0], lw=3, alpha=0.4, label="Lattice ME")
ax[0].semilogy(1000 / np.array(Ts), mu_trj, "o", c=C[1], label="GRO-style frame -> graph ME")
ax[0].set_xlabel("1000/T (1/K)")
ax[0].set_ylabel(r"$\mu$ (m$^2$/Vs)")
ax[0].set_title(f"lattice trajectory, max rel. dev. {rel:.1e}", fontsize=10)
ax[0].legend(fontsize=8)
ax[1].errorbar(amps, mus / mus[0], yerr=sems / mus[0], fmt="o-", c=C[2])
ax[1].set_xlabel(r"thermal displacement $\sigma_r/a$")
ax[1].set_ylabel(r"$\langle\mu\rangle_{frames}/\mu(\sigma_r=0)$")
ax[1].set_title("4 snapshots per amplitude, Gaussian DOS", fontsize=10)
save(fig, "v26_md_import")
out = dict(exact_centre_err_m=exact_err, gro_centre_err_m=gro_err, molecules_split_by_pbc=broken,
           mu_rel_dev=rel, amps=amps, mu_ratio=mus / mus[0], mu_sem_ratio=sems / mus[0])
dump("v26", out)
print(out)
