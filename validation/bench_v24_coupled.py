"""V24 - concurrent charge + exciton device KMC (R24).

(a) exact exciton bookkeeping in every run;
(b) quenching off: eta_rad -> k_r/(k_r+k_nr) (exact limit);
(c) excitons do not perturb charge transport: J(coupled) = J(DeviceKMC);
(d) EQE roll-off, concurrent vs v0.3 sequential pipeline (frozen polarons)."""
import numpy as np
from common import plt, save, dump, C
from orchid.coupled_kmc import CoupledDeviceKMC
from orchid.device_kmc import DeviceKMC, Layer
from orchid.exciton_device import ExcitonDeviceKMC, oled_eqe_from_device
from orchid.parallel import parallel_map

A = 1.6e-9
STACK = [Layer(8, HOMO=-5.2, LUMO=-2.0, name="HTL"), Layer(10, HOMO=-5.4, LUMO=-2.7, name="EML"),
         Layer(8, HOMO=-6.0, LUMO=-2.8, name="ETL")]
GEO = dict(Lx=16, Ly=16, W_anode=5.1, W_cathode=2.9, coulomb="full", seed=2)
XP = dict(tau_r=1e-6, tau_nr=1e-5, R0=1.5, k_TT0=1e8, R_TT=1.5)
Vs = [4.0, 5.0, 6.0, 7.0]
RUN = dict(n_eq=150000, n_meas=600000, mf_update=300)


def coupled(args):
    V, seed, kq, ktt = args
    d = CoupledDeviceKMC(STACK, tau_r=1e-6, tau_nr=1e-5, R0=1.5, k_TT0=ktt, R_TT=1.5,
                         k_q0=kq, R_q=1.5, eta_out=0.2, **GEO)
    r = d.run(V, seed=seed, **RUN)
    x = r["excitons"]
    bal = x["formed"] + x["alive_start"] - (x["radiative"] + x["nonradiative"] + x["TPQ"]
                                            + x["TTA"] + x["overflow"] + x["alive_end"])
    return dict(V=V, J=r["J"], EQE=r["EQE"], eta=r["eta_rad"], tpq=r["frac_TPQ"],
                tta=r["frac_TTA"], n=x["radiative"] + x["nonradiative"] + x["TPQ"] + x["TTA"],
                balance=bal, emis=r["emission_profile"], rec=r["rec_profile"])


def sequential(args):
    V, seed = args
    d = DeviceKMC(STACK, **GEO)
    r = d.run(V, seed=seed, **RUN)
    x = ExcitonDeviceKMC(d.shape, **XP)
    x.set_polarons(r["n_z"] * A ** 3, r["p_z"] * A ** 3, 1e8)
    o = oled_eqe_from_device(r, x, eta_ST=1.0, eta_out=0.2, n_events=300000, n_eq=100000)
    return dict(V=V, J=r["J"], EQE=o["EQE"], eta=o["eta_rad"], tpq=o["frac_quench"])


if __name__ == "__main__":
    seeds = (5, 6)
    co = parallel_map(coupled, [(V, s, 1e8, 1e8) for V in Vs for s in seeds], workers=2)
    sq = parallel_map(sequential, [(V, s) for V in Vs for s in seeds], workers=2)
    nq = parallel_map(coupled, [(7.0, s, 0.0, 0.0) for s in (11, 12, 13, 14)], workers=2)

    def agg(res, key):
        return np.array([[r[key] for r in res if r["V"] == V] for V in Vs])
    Jc, Js = agg(co, "J"), agg(sq, "J")
    Ec, Es = agg(co, "EQE"), agg(sq, "EQE")
    ec, es = agg(co, "eta"), agg(sq, "eta")
    nrad = sum(r["eta"] * r["n"] for r in nq)
    ntot = sum(r["n"] for r in nq)
    eta0 = nrad / ntot
    eta_exact = 1e-6 ** -1 / (1e-6 ** -1 + 1e-5 ** -1)
    z_eta = (eta0 - eta_exact) / np.sqrt(eta_exact * (1 - eta_exact) / ntot)
    J_ratio = Jc.mean(1) / Js.mean(1)
    J_rel_spread = np.abs(np.diff(np.concatenate([Jc, Js], 1), axis=1)).max() / Jc.mean()
    balance = max(abs(r["balance"]) for r in co + nq)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    ax[0].errorbar(Jc.mean(1) / 10, 100 * Ec.mean(1), yerr=100 * Ec.std(1), fmt="o-", c=C[0],
                   label="concurrent (v0.4, one event loop)")
    ax[0].errorbar(Js.mean(1) / 10, 100 * Es.mean(1), yerr=100 * Es.std(1), fmt="s--", c=C[1],
                   mfc="none", label="sequential (v0.3, frozen polarons)")
    ax[0].axhline(100 * 0.2 * eta_exact, c="k", ls=":", lw=1, label="no quenching (exact)")
    ax[0].set_xscale("log")
    from matplotlib.ticker import NullFormatter
    ax[0].xaxis.set_minor_formatter(NullFormatter())
    ax[0].set_xlabel(r"$J$ (mA cm$^{-2}$)")
    ax[0].set_ylabel("EQE (%)")
    ax[0].set_title("TTA + TPQ roll-off, HTL/EML/ETL", fontsize=10)
    ax[0].legend(fontsize=7)
    r7 = [r for r in co if r["V"] == 7.0][0]
    zz = np.arange(len(r7["rec"])) + 1
    ax[1].plot(zz, r7["rec"], "-", c=C[2], label="recombination profile")
    ax[1].plot(zz, r7["emis"], "o", c=C[3], ms=4, label="emission profile (live TPQ)")
    for zb in (8.5, 18.5):
        ax[1].axvline(zb, c="grey", ls=":", lw=1)
    ax[1].set_xlabel("z (sites)")
    ax[1].set_ylabel("normalised profile")
    ax[1].set_title("V = 7 V, concurrent KMC", fontsize=10)
    ax[1].legend(fontsize=7)
    save(fig, "v24_coupled")
    out = dict(rec_profile_7V=r7["rec"], emission_profile_7V=r7["emis"], V=Vs, J_coupled=Jc, J_sequential=Js, EQE_coupled=Ec, EQE_sequential=Es,
               eta_coupled=ec, eta_sequential=es, eta_noquench=eta0, eta_exact=eta_exact,
               z_eta=z_eta, n_noquench=ntot, J_ratio=J_ratio, balance_max=balance,
               EQE_ratio=Ec.mean(1) / Es.mean(1))
    dump("v24", out)
    print({k: out[k] for k in ("eta_noquench", "eta_exact", "z_eta", "J_ratio", "balance_max",
                               "EQE_ratio")})
