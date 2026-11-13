"""V18 - exciton KMC in devices (R16): bimolecular TTA law and OLED roll-off
from coupled device KMC + exciton KMC."""
import numpy as np
from common import plt, save, dump, C
from orchid.exciton_device import ExcitonDeviceKMC, oled_eqe_from_device
from orchid.device_kmc import DeviceKMC, Layer
from orchid.parallel import parallel_map

tau = 0.8e-6
Gs = np.logspace(28, 30.3, 8)


def tta_point(G):
    x = ExcitonDeviceKMC((20, 20, 20), tau_r=1e-6, tau_nr=4e-6, R0=1.5, periodic_z=True,
                         k_TT0=1e8, R_TT=1.5)
    r = x.run(G, n_events=400000, n_eq=150000)
    return r["n_ex"], r["frac_TTA"], r["eta_rad"]


STACK = [Layer(8, HOMO=-5.2, LUMO=-2.0, name="HTL"), Layer(10, HOMO=-5.4, LUMO=-2.7, name="EML"),
         Layer(8, HOMO=-6.0, LUMO=-2.8, name="ETL")]


def oled_point(V):
    d = DeviceKMC(STACK, Lx=16, Ly=16, W_anode=5.1, W_cathode=2.9, coulomb="full", seed=2)
    r = d.run(V, n_eq=150000, n_meas=500000, mf_update=300, seed=5)
    out = {}
    for k, kq in (("TTA", 0.0), ("TTA+TPQ", 1e8)):
        x = ExcitonDeviceKMC((16, 16, 26), tau_r=1e-6, tau_nr=1e-5, R0=1.5, k_TT0=1e8,
                             R_TT=1.5, k_metal0=1e9, d_metal=1.0)
        if kq > 0:
            x.set_polarons(r["n_z"] * (1.6e-9) ** 3, r["p_z"] * (1.6e-9) ** 3, kq)
        out[k] = oled_eqe_from_device(r, x, eta_ST=1.0, eta_out=0.2, n_events=300000, n_eq=100000)
    return r["J"], r["gamma_b"], {k: v["EQE"] for k, v in out.items()}


if __name__ == "__main__":
    res = parallel_map(tta_point, Gs, workers=2)
    n = np.array([r[0] for r in res])
    fT = np.array([r[1] for r in res])
    gam = fT * Gs / n ** 2
    oled = parallel_map(oled_point, [3.0, 4.0, 5.0, 6.0], workers=2)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].semilogx(n, gam, "o-", c=C[0], label=r"KMC: TTA loss rate / $n^2$")
    ax[0].axhline(np.mean(gam[2:]), c="k", ls="--", lw=1, label=r"constant $\gamma_{TT}$")
    ax[0].set_ylim(0, 2 * gam.max())
    ax[0].set_xlabel(r"exciton density $n$ (m$^{-3}$)")
    ax[0].set_ylabel(r"$\gamma_{TT}$ (m$^3$/s)")
    ax[0].set_title("bimolecular annihilation law", fontsize=10)
    ax[0].legend(fontsize=8)
    J = np.array([o[0] for o in oled])
    for k, lab in enumerate(("TTA", "TTA+TPQ")):
        ax[1].semilogx(J / 10, [100 * o[2][lab] for o in oled], "os"[k] + "-", c=C[k + 1], label=lab)
    ax[1].set_xlabel(r"$J$ (mA cm$^{-2}$)")
    ax[1].set_ylabel("EQE (%)")
    ax[1].set_title("3D device KMC + exciton KMC (HTL/EML/ETL)", fontsize=10)
    ax[1].legend(fontsize=8)
    save(fig, "v18_exciton_device")
    dump("v18", dict(n=n, gamma=gam, gamma_spread=float(np.std(gam[2:]) / np.mean(gam[2:])),
                     oled=[dict(J=o[0], gamma_b=o[1], EQE=o[2]) for o in oled]))
    print("done", gam, J, [o[2] for o in oled])
