"""V4 energetic relaxation, V7 exciton diffusion, V8 transfer matrix,
V9 exhaustive-search parameter recovery, V11 Marcus."""
import time
import numpy as np
from common import plt, save, dump, C
from orchid.lattice import Lattice
from orchid.kmc import energy_relaxation, mobility_independent
from orchid.exciton import exciton_kmc
from orchid.optics import tmm_field
from orchid.models import gdm_bassler
from orchid.search import grid_search, zoom_search, log_rmse
from orchid.rates import marcus, miller_abrahams
from orchid.drift_diffusion import Device
from orchid.constants import kB_eV

out = {}
t0 = time.time()
# ---------------------------------------------------------------- V4
sigma = 0.1
fig, ax = plt.subplots()
v4 = {}
for k, s in enumerate((2.0, 3.0)):
    T = sigma / (kB_eV * s)
    lat = Lattice(40, a=1.6e-9, seed=11).set_disorder(sigma)
    t, E = energy_relaxation(lat, T, n_carriers=512, tgrid=np.logspace(-13, 1, 141),
                             alpha=10.0)
    Einf_sample = np.sum(lat.energy * np.exp(-lat.energy / (kB_eV * T))) / \
        np.sum(np.exp(-lat.energy / (kB_eV * T)))
    ax.semilogx(t, E / sigma, c=C[k], label=rf"KMC $\hat\sigma$={s:g}")
    ax.axhline(-s, c=C[k], ls="--", lw=1)
    v4[s] = dict(E_last=np.nanmean(E[-10:]), Einf=-sigma * s, Einf_sample=Einf_sample)
ax.set_xlabel("t (s)")
ax.set_ylabel(r"$\langle E\rangle/\sigma$")
ax.set_title(r"dashed: $-\sigma^2/k_BT$ (Bässler 1993)", fontsize=10)
ax.legend(fontsize=8)
save(fig, "v4_relaxation")
out["v4"] = v4
print("V4", v4, time.time() - t0, flush=True)

# ---------------------------------------------------------------- V7
lat = Lattice(30, a=1e-9, rc=3.0, seed=0).set_disorder(0.0)
r0 = exciton_kmc(lat, 300, tau=1e-9, R0=2.0, n_excitons=100000)
sig_ex = np.array([0.0, 0.025, 0.05, 0.075, 0.1, 0.125])
LD = []
for s in sig_ex:
    lat.set_disorder(s) if s > 0 else lat.set_disorder(0.0)
    LD.append(exciton_kmc(lat, 300, tau=1e-9, R0=2.0, n_excitons=30000)["L_D"])
fig, ax = plt.subplots()
ax.plot(sig_ex * 1e3, np.array(LD) * 1e9, "o-", c=C[0], label="Förster KMC")
ax.axhline(np.sqrt(r0["D_ordered"] * 1e-9) * 1e9, c=C[1], ls="--",
           label=r"analytic $\sqrt{D\tau}$, $D=\frac{1}{6}\sum_j k_j r_j^2$")
ax.set_xlabel(r"exciton energetic disorder $\sigma_{ex}$ (meV)")
ax.set_ylabel(r"$L_D$ (nm)")
ax.legend(fontsize=8)
save(fig, "v7_exciton")
out["v7"] = dict(D_kmc=r0["D"], D_err=r0["D_err"], D_analytic=r0["D_ordered"],
                 sigma=sig_ex, LD=LD)
print("V7", r0["D"], r0["D_err"], r0["D_ordered"], time.time() - t0, flush=True)

# ---------------------------------------------------------------- V8
lams = np.linspace(350e-9, 900e-9, 56)
from orchid.optics import lorentz_nk
cons, airy = [], []
for lam in lams:
    L = [("ITO", 1.9 + 0.01j, 120e-9), ("PEDOT", 1.5 + 0.02j, 40e-9),
         ("blend", lorentz_nk(lam), 100e-9), ("Al", 0.96 + 6.69j, 100e-9)]
    r = tmm_field(L, lam, npts=801)
    cons.append(r["R"] + r["T"] + r["A"].sum())
    n, d = 2.0, 100e-9
    rr = tmm_field([("s", n + 0j, d)], lam, n_in=1 + 0j, n_out=1 + 0j)
    r1 = (1 - n) / (1 + n)
    dl = 2 * np.pi * n * d / lam
    Ra = abs((r1 - r1 * np.exp(2j * dl)) / (1 - r1 ** 2 * np.exp(2j * dl))) ** 2
    airy.append((rr["R"], Ra))
airy = np.array(airy)
fig, ax = plt.subplots()
ax.plot(lams * 1e9, airy[:, 1], "-", c=C[1], label="Airy formula (exact)")
ax.plot(lams * 1e9, airy[:, 0], "o", ms=3, c=C[0], label="ORCHID TMM")
ax.plot(lams * 1e9, np.array(cons) - 1 + 0.5, c=C[2], label="R+T+A $-$ 0.5 (stack)")
ax.set_xlabel(r"$\lambda$ (nm)")
ax.set_ylabel("reflectance")
ax.legend(fontsize=8)
save(fig, "v8_tmm")
out["v8"] = dict(max_airy_err=np.max(np.abs(airy[:, 0] - airy[:, 1])),
                 max_energy_err=np.max(np.abs(np.array(cons) - 1)))
print("V8", out["v8"], flush=True)

# ---------------------------------------------------------------- V9
rng = np.random.default_rng(1)
Ts = np.array([200, 230, 260, 290, 320.0])
Fs = np.array([1e4, 1e5, 2.5e5, 5e5, 1e6])          # V/cm
TT, FF = np.meshgrid(Ts, Fs, indexing="ij")
true = dict(sigma=0.095, mu_inf=3e-2, Sigma=2.5)
data = gdm_bassler(TT, FF, true["sigma"], true["mu_inf"], Sigma=true["Sigma"])
data *= np.exp(rng.normal(0, 0.05, data.shape))


def obj(p):
    return log_rmse(gdm_bassler(TT, FF, p["sigma"], p["mu_inf"], Sigma=p["Sigma"]), data)


best, val, hist = zoom_search(obj, {"sigma": (0.05, 0.15), "mu_inf": (1e-4, 1.0),
                                    "Sigma": (1.5, 4.0)}, n=15, levels=5, log=("mu_inf",))
out["v9_gdm"] = dict(true=true, fit=best, rmse=val)
print("V9 GDM", best, val, flush=True)
# DD: extract mobility & barrier from synthetic hole-only J-V
dtrue = dict(mu=3e-8, phi=0.15)
Vd = np.linspace(2, 8, 7)


def jv(mu, phi):
    d = Device(L=100e-9, N=121, Eg=2.4, phi_anode_p=phi, phi_cathode_n=1.0,
               mu_n=1e-10, mu_p=mu, recombination=False)
    return d.jv(Vd)["J"]


Jd = jv(**dtrue) * np.exp(rng.normal(0, 0.03, len(Vd)))
def dd_obj(p):
    return log_rmse(jv(p["mu"], p["phi"]), Jd)


b2, v2, tab2 = grid_search(dd_obj,
                           {"mu": np.logspace(-8.5, -7, 13), "phi": np.linspace(0.0, 0.3, 13)},
                           workers=2)
out["v9_dd"] = dict(true=dtrue, fit=b2, rmse=v2)
print("V9 DD", b2, v2, time.time() - t0, flush=True)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
for k, T in enumerate(Ts):
    ax[0].semilogy(np.sqrt(Fs), data[k], "o", c=C[k])
    ax[0].semilogy(np.sqrt(Fs), gdm_bassler(T, Fs, best["sigma"], best["mu_inf"],
                                             Sigma=best["Sigma"]), "-", c=C[k], label=f"{T:.0f} K")
ax[0].set_xlabel(r"$\sqrt{F}$ (V/cm)$^{1/2}$")
ax[0].set_ylabel(r"$\mu$ (cm$^2$/Vs)")
ax[0].legend(fontsize=7, ncol=2)
ax[0].set_title("GDM fit (zoom grid search)", fontsize=10)
Z = tab2["objective"].reshape(13, 13)
im = ax[1].contourf(np.linspace(0.0, 0.3, 13), np.logspace(-8.5, -7, 13) * 1e4, Z, 20,
                    cmap="viridis")
ax[1].set_yscale("log")
ax[1].plot(dtrue["phi"], dtrue["mu"] * 1e4, "r*", ms=14, label="truth")
ax[1].plot(b2["phi"], b2["mu"] * 1e4, "wx", ms=10, mew=2, label="best")
ax[1].set_xlabel(r"injection barrier $\varphi_p$ (eV)")
ax[1].set_ylabel(r"$\mu_p$ (cm$^2$/Vs)")
ax[1].set_title("exhaustive DD landscape (log-RMSE)", fontsize=10)
ax[1].legend(fontsize=8)
fig.colorbar(im, ax=ax[1])
save(fig, "v9_search")

# V10 (time of flight) lives in bench_tof.py

# ---------------------------------------------------------------- V11
dE = np.linspace(-0.6, 0.6, 241)
lam_, Tm = 0.2, 300.0
km = marcus(dE, 0.01, Tm, lam_)
db = np.max(np.abs(np.log(km / marcus(-dE, 0.01, Tm, lam_)) + dE / (kB_eV * Tm)))
mus_ma, mus_m = [], []
Ts11 = np.array([200, 250, 300, 350, 400.0])
lat = Lattice(30, a=1.6e-9, seed=31).set_disorder(0.07)
for T in Ts11:
    mus_ma.append(mobility_independent(lat, T, 0.005, n_carriers=32, n_eq=100000,
                                       n_meas=300000, alpha=10.0)[0])
    mus_m.append(mobility_independent(lat, T, 0.005, n_carriers=32, n_eq=100000,
                                      n_meas=300000, kind="marcus", J0=0.01,
                                      lam=lam_, alpha=2.0)[0])
mus_ma, mus_m = np.array(mus_ma), np.array(mus_m)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
ax[0].semilogy(dE, km, c=C[0], label="Marcus")
ax[0].semilogy(dE, miller_abrahams(dE, 1.0, Tm, nu0=km.max(), alpha=0.0), c=C[1],
               label="Miller-Abrahams")
ax[0].axvline(-lam_, c="k", ls=":", lw=1)
ax[0].set_ylim(km.max() * 1e-8, km.max() * 3)
ax[0].set_xlabel(r"$\Delta E$ (eV)")
ax[0].set_ylabel("rate (1/s)")
ax[0].legend(fontsize=8)
ax[0].set_title("inverted region at $\\Delta E<-\\lambda$", fontsize=10)
x = 1e3 / Ts11
ax[1].semilogy(x, mus_ma * 1e4, "o-", c=C[1], label="MA")
ax[1].semilogy(x, mus_m * 1e4, "s-", c=C[0], label=rf"Marcus, $\lambda$={lam_} eV")
ax[1].set_xlabel("1000/T (1/K)")
ax[1].set_ylabel(r"$\mu$ (cm$^2$/Vs)")
ax[1].legend(fontsize=8)
ax[1].set_title(r"KMC, $\sigma$=70 meV", fontsize=10)
save(fig, "v11_marcus")
Ea = -np.polyfit(1 / (kB_eV * Ts11), np.log(mus_m), 1)[0]
out["v11"] = dict(detailed_balance_err=db, mu_ma=mus_ma, mu_marcus=mus_m,
                  Ea_marcus_eff=Ea)
print("V11", db, Ea, time.time() - t0, flush=True)
dump("misc", out)
