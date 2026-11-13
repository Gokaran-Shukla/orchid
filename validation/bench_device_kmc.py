"""V13 - 3D device KMC vs 1D drift-diffusion (EGDM + g3) for a hole-only
diode, the comparison of van der Holst et al., PRB 79, 085203 (2009).
V14 - three-layer OLED (HTL / EML / ETL) with the 3D device KMC:
recombination profile and charge-balance factor.
Runs in parallel through orchid.parallel (processes or MPI ranks).
"""
import numpy as np
from common import plt, save, dump, C
from orchid.device_kmc import DeviceKMC, Layer
from orchid.drift_diffusion import Device
from orchid.models import egdm_pasveer, einstein_ratio_gaussian
from orchid.parallel import parallel_map, is_root
from orchid.constants import kB_eV

T, sig, a, nu0, Lz = 290.0, 0.075, 1.6e-9, 1e12, 30
Wa, Wc, IP = 4.9, 4.0, 5.2          # hole barrier 0.3 eV, Vbi = 0.9 V
Vs = np.array([1.5, 2.0, 3.0, 4.0, 5.0, 6.0])


def kmc_job(args):
    mode, V, seed = args
    d = DeviceKMC([Layer(Lz, HOMO=-IP, LUMO=-2.0, sigma_h=sig, sigma_e=sig)], Lx=30, Ly=30,
                  a=a, T=T, W_anode=Wa, W_cathode=Wc, coulomb=mode, image=False, seed=seed)
    r = d.run(V, n_eq=300000, n_meas=900000, mf_update=300, seed=seed + 7)
    return r["J"], r["p_z"]


def oled_job(V):
    lay = [Layer(10, HOMO=-5.2, LUMO=-2.0, sigma_h=0.07, sigma_e=0.07, name="HTL"),
           Layer(10, HOMO=-5.4, LUMO=-2.7, sigma_h=0.07, sigma_e=0.07, name="EML"),
           Layer(10, HOMO=-6.0, LUMO=-2.8, sigma_h=0.07, sigma_e=0.07, name="ETL")]
    d = DeviceKMC(lay, Lx=30, Ly=30, a=a, T=T, W_anode=5.1, W_cathode=2.9,
                  coulomb="full", image=True, seed=3)
    return d.run(V, n_eq=300000, n_meas=900000, mf_update=300, seed=11)


if __name__ == "__main__":
    import sys, json, os
    from common import RES
    if "--oled-only" in sys.argv:
        old = json.load(open(os.path.join(RES, "device_kmc.json")))
        J = {m: {float(V): v for V, v in d_.items()} for m, d_ in old["J"].items()}
    else:
        tasks = [(m, V, s) for m in ("meanfield", "full") for V in Vs for s in (0, 1)]
        res = parallel_map(kmc_job, tasks, workers=2)
        J = {}
        for (m, V, s), (j, pz) in zip(tasks, res):
            J.setdefault(m, {}).setdefault(V, []).append(j)
    # 1D drift diffusion with EGDM mobility and g3, same parameters
    Lsite = (Lz + 1) * a
    shat = sig / (kB_eV * T)
    Nt = 1 / a ** 3
    Nv_eff = Nt * np.exp(0.5 * shat ** 2)      # non-degenerate Gaussian DOS
    mu = lambda p, F, T_: egdm_pasveer(T_, p, F, sig, a, nu0=nu0)
    pg = np.logspace(-8, -1, 60)
    g3t = np.array([einstein_ratio_gaussian(x, T, sig) for x in pg])
    g3 = lambda dens: np.interp(np.log(np.clip(dens / Nt, 1e-8, 1e-1)), np.log(pg), g3t)
    dd = Device(L=Lsite, N=201, T=T, Eg=3.2, Nc=Nv_eff, Nv=Nv_eff, eps_r=3.5,
                phi_anode_p=IP - Wa, phi_cathode_n=Wc - 2.0, mu_n=1e-20, mu_p=mu,
                g3_p=g3, recombination=False)
    Vd = np.linspace(1.0, 6.0, 21)
    Jdd = dd.jv(Vd)["J"]
    # OLED
    oled = parallel_map(oled_job, [3.0, 4.0, 5.0], workers=2)
    if is_root():
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
        ax[0].semilogy(Vd, Jdd, "-", c="k", label="1D DD, EGDM + $g_3$")
        for k, m in enumerate(("meanfield", "full")):
            Jm = np.array([np.mean(J[m][V]) for V in Vs])
            Je = np.array([np.std(J[m][V]) for V in Vs])
            ax[0].errorbar(Vs, Jm, yerr=Je, fmt="os"[k], c=C[k], mfc="none" if k else C[k],
                           label=f"3D device KMC ({m} Coulomb)")
        ax[0].set_xlabel("V (V)")
        ax[0].set_ylabel(r"$J$ (A m$^{-2}$)")
        ax[0].set_title(r"hole-only, $\hat\sigma$=3, L=50 nm", fontsize=10)
        ax[0].legend(fontsize=7)
        zz = np.arange(30) + 1
        for k, r in enumerate(oled):
            ax[1].plot(zz, r["rec_profile"], c=C[k],
                       label=f"V={r['V']:.0f} V, $\\gamma_b$={r['gamma_b']:.2f}")
        for zb in (10.5, 20.5):
            ax[1].axvline(zb, c="grey", ls=":", lw=1)
        ax[1].text(3, ax[1].get_ylim()[1] * 0.9, "HTL", fontsize=8)
        ax[1].text(13, ax[1].get_ylim()[1] * 0.9, "EML", fontsize=8)
        ax[1].text(23, ax[1].get_ylim()[1] * 0.9, "ETL", fontsize=8)
        ax[1].set_xlabel("z (sites, anode $\\to$ cathode)")
        ax[1].set_ylabel("recombination profile")
        ax[1].set_title("3-layer OLED, 3D device KMC", fontsize=10)
        ax[1].legend(fontsize=7)
        save(fig, "v13_device_kmc")
        dump("device_kmc", dict(V=Vs, J={m: {str(V): v for V, v in d_.items()} for m, d_ in J.items()},
                                Vdd=Vd, Jdd=Jdd,
                                oled=[dict(V=r["V"], J=r["J"], gamma_b=r["gamma_b"],
                                           rec=r["rec_profile"]) for r in oled]))
        print("done", {m: [np.mean(J[m][V]) for V in Vs] for m in J}, np.interp(Vs, Vd, Jdd))
