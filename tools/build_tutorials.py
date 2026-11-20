"""Build the ORCHID tutorial notebooks (docs/tutorials/*.ipynb).

    python tools/build_tutorials.py            # write notebooks
    python tools/build_tutorials.py --execute  # write and execute (outputs stored)

Each notebook has: learning goals, theory with equations, runnable examples,
exercises with hidden solutions, and a five-question quiz (orchid.tutorials).
"""
import json
import os
import sys

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "tutorials")
sys.path.insert(0, ROOT)
QUIZ = json.load(open(os.path.join(ROOT, "orchid", "data", "quizzes.json"), encoding="utf-8"))

HEAD = """import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.figsize": (5.4, 3.6), "axes.grid": True, "grid.alpha": 0.3})
import orchid
print("ORCHID", orchid.__version__)"""


def md(s):
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s):
    return nbf.v4.new_code_cell(s.strip("\n"))


def quiz_cells(k):
    key = "".join(q["answer"] for q in QUIZ[k]["questions"])
    return [
        md(f"""## Quiz

Five questions on this section. Run the cell to see them, then grade yourself with
`quiz("{k}", answers="abcda")` (one letter per question). The same quiz runs in a terminal
with `orchid quiz {k}`."""),
        code(f'from orchid.tutorials import quiz\nquiz("{k}")'),
        code(f'# replace with your answers, e.g. "abcda"\n# quiz("{k}", answers=".....")'),
        md(f"""<details><summary>Answer key</summary>

`{key}`. Run `quiz("{k}", answers="{key}")` to see the explanation for every question.
</details>"""),
    ]


def exercise(text, solution):
    return md(f"""**Exercise.** {text}

<details><summary>Solution</summary>

{solution}
</details>""")


def notebook(k, cells):
    title = QUIZ[k]["title"]
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
    nb.cells = [md(f"# ORCHID tutorial {k}: {title}"), *cells, *quiz_cells(k),
                md("---\n[Tutorial index](README.md) · ORCHID " "documentation · `orchid tutorials`")]
    return QUIZ[k]["notebook"], nb


NB = []

# ----------------------------------------------------------------------------- 01
NB.append(notebook("01", [
    md(r"""**Goals.** Evaluate the hopping-rate models in `orchid.rates`, check detailed balance, and
build disordered lattices (`orchid.lattice`) with Gaussian, correlated (dipolar) and off-diagonal disorder.

### Theory
Charge transport in amorphous organic semiconductors is incoherent hopping between localised
sites $i$ with energies $\varepsilon_i$. ORCHID implements:

* **Miller–Abrahams (MA):** $W_{ij} = \nu_0\, e^{-2\alpha r_{ij}}\,\min\!\left(1, e^{-\Delta\varepsilon_{ij}/k_BT}\right)$
* **Marcus:** $W_{ij} = \dfrac{2\pi}{\hbar}\dfrac{J_{ij}^2}{\sqrt{4\pi\lambda k_BT}}\exp\!\left[-\dfrac{(\Delta\varepsilon_{ij}+\lambda)^2}{4\lambda k_BT}\right]$
* **MLJ** (quantum high-frequency mode $\hbar\omega$, Huang–Rhys $S$), **Förster** $k=\tau^{-1}(R_0/r)^6$, **Dexter** $k\propto e^{-2r/L}$.

All of them obey **detailed balance**, $W_{ij}/W_{ji} = e^{-(\varepsilon_j-\varepsilon_i)/k_BT}$.
Disorder is described by the Gaussian DOS $g(\varepsilon)=\frac{N_t}{\sqrt{2\pi}\sigma}e^{-\varepsilon^2/2\sigma^2}$,
with dimensionless width $\hat\sigma=\sigma/k_BT$."""),
    code(HEAD),
    code(r"""from orchid.rates import miller_abrahams, marcus, symmetric_ma
from orchid.constants import kB_eV
T = 300.0
dE = np.linspace(-0.6, 0.6, 400)
fig, ax = plt.subplots()
ax.semilogy(dE, miller_abrahams(dE, 1.0, T, alpha=0.0), label="Miller-Abrahams")
ax.semilogy(dE, symmetric_ma(dE, 1.0, T, alpha=0.0), label="symmetric MA")
for lam in (0.1, 0.3):
    k = marcus(dE, 0.01, T, lam)
    ax.semilogy(dE, k / k.max() * 1e12, label=f"Marcus, $\\lambda$={lam} eV (scaled)")
ax.set_xlabel(r"$\Delta\varepsilon$ (eV)"); ax.set_ylabel("rate (1/s)"); ax.legend(fontsize=8)
ax.set_ylim(1e3, 3e12);"""),
    md(r"""Marcus rates peak at $-\Delta\varepsilon=\lambda$ and fall again further out (the *inverted region*).
MA rates stay flat for every downhill hop. Now check detailed balance numerically:"""),
    code(r"""for name, f in [("MA", lambda d: miller_abrahams(d, 1.0, T, alpha=0.0)),
                ("Marcus", lambda d: marcus(d, 0.01, T, 0.2))]:
    d = np.linspace(-0.4, 0.4, 41)
    err = np.max(np.abs(np.log(f(d) / f(-d)) + d / (kB_eV * T)))
    print(f"{name:7s} max |ln(W_ij/W_ji) + dE/kT| = {err:.1e}")"""),
    md(r"""### Disordered lattices
`Lattice(L, a, rc)` is a periodic cubic lattice with hops up to `rc` lattice constants.
`set_disorder(sigma, model)` draws the site energies: `"gaussian"` (uncorrelated), `"dipolar"`
(correlated, CDM-like) or `"exponential"`."""),
    code(r"""from orchid.lattice import Lattice
lat_g = Lattice(24, a=1.6e-9, seed=1).set_disorder(0.1)
lat_d = Lattice(24, a=1.6e-9, seed=1).set_disorder(0.1, "dipolar")
def corr(lat, shift):
    E = lat.energy.reshape(lat.shape)
    return np.mean(E * np.roll(E, shift, 0)) / np.var(E)
print("std:", lat_g.energy.std(), lat_d.energy.std())
print("nearest-neighbour correlation  gaussian %.3f   dipolar %.3f" % (corr(lat_g, 1), corr(lat_d, 1)))
fig, ax = plt.subplots(1, 2, figsize=(8, 3.4))
for a_, l, t in zip(ax, (lat_g, lat_d), ("Gaussian", "dipolar")):
    a_.imshow(l.energy.reshape(l.shape)[:, :, 0], cmap="RdBu_r"); a_.set_title(t); a_.grid(False)"""),
    md(r"""The dipolar landscape is smooth, so neighbouring sites have similar energies. This correlation
produces the extended Poole–Frenkel $\ln\mu\propto\sqrt F$ regime of the correlated disorder model."""),
    exercise(r"""Plot the Förster rate `orchid.rates.forster(0, r, T, tau, R0)` for $r=0.5\ldots5\,R_0$ and
confirm the $r^{-6}$ slope on a log–log plot.""",
             r"""```python
from orchid.rates import forster
r = np.logspace(np.log10(0.5), np.log10(5), 30)
k = forster(0.0, r, 300.0, 1e-9, 1.0)
print(np.polyfit(np.log(r), np.log(k), 1)[0])   # -> -6
```"""),
    exercise(r"""At what $\hat\sigma$ is a 0.1 eV Gaussian DOS at 300 K? At 200 K?""",
             r"""$\hat\sigma = 0.1/(8.617\times10^{-5}\cdot300) = 3.87$; at 200 K, 5.80."""),
]))

# ----------------------------------------------------------------------------- 02
NB.append(notebook("02", [
    md(r"""**Goals.** Run kinetic Monte Carlo for independent carriers (mobility, energy relaxation,
time of flight), and for interacting carriers with the $O(\log n)$ Fenwick-tree engine.

### Theory
KMC follows the BKL/Gillespie scheme. With total rate $R=\sum_k W_k$, the next event $k$ is chosen with
probability $W_k/R$ and the clock advances by $\Delta t=-\ln u/R$, where $u\in(0,1]$.
The mobility is $\mu = \langle x(t)\rangle/(F t)$. In a Gaussian DOS, a carrier at low density relaxes
to the equilibrium energy $\varepsilon_\infty=-\sigma^2/k_BT$ (Bässler 1993)."""),
    code(HEAD),
    code(r"""from orchid.lattice import Lattice
from orchid.kmc import mobility_independent, mobility_interacting, energy_relaxation, time_of_flight
from orchid.constants import kB_eV
T = 300.0
for sh in (1.0, 2.0, 3.0):
    s = sh * kB_eV * T
    lat = Lattice(24, a=1.6e-9, seed=2).set_disorder(s)
    mu, err = mobility_independent(lat, T, 0.05 * s, n_carriers=64, n_eq=50000, n_meas=150000)
    print(f"sigma-hat={sh}: mu = {mu*1e4:.3e} +- {err*1e4:.1e} cm2/Vs")"""),
    md("Mobility falls steeply with disorder, roughly as $\\ln\\mu\\approx -c\\,\\hat\\sigma^2$. Next comes energy relaxation:"),
    code(r"""sh = 2.0; s = sh * kB_eV * T
lat = Lattice(30, a=1.6e-9, seed=3).set_disorder(s)
tg, E = energy_relaxation(lat, T, n_carriers=128, tgrid=np.logspace(-12, 2, 60))
plt.semilogx(tg, E / s, label="KMC")
plt.axhline(-sh, c="k", ls="--", label=r"$-\sigma^2/kT$")
plt.xlabel("t (s)"); plt.ylabel(r"$\langle\varepsilon\rangle/\sigma$"); plt.legend();"""),
    md(r"""### Time of flight
Carriers start in a sheet and drift a distance $L$. The photocurrent $I(t)\propto d\langle x\rangle/dt$
is flat (non-dispersive) at low $\hat\sigma$. At high $\hat\sigma$ the plateau tilts and the tail broadens."""),
    code(r"""fig, ax = plt.subplots()
for sh in (0.0, 2.0, 3.0):
    s = max(sh * kB_eV * T, 1e-6)
    lat = Lattice(30, a=1.6e-9, seed=21).set_disorder(s)
    tg = np.logspace(-5, 6, 160)
    r = time_of_flight(lat, T, 0.4 * kB_eV * T, L_sample=60, n_carriers=256, tgrid=tg)
    I = r["I"]; m = I > 0
    ax.loglog(tg[m] / r["t_transit"], I[m] / np.interp(r["t_transit"], tg, I), label=rf"$\hat\sigma$={sh}")
ax.set_xlim(1e-3, 30); ax.set_ylim(1e-3, 1e2)
ax.set_xlabel(r"$t/t_{tr}$"); ax.set_ylabel(r"$I/I(t_{tr})$"); ax.legend();"""),
    md(r"""### Interacting carriers
At finite density, site exclusion (and optionally Coulomb interaction) couples the carriers.
`mobility_interacting` uses a Fenwick tree for event selection and local rate updates, at $O(\log n)$ per event."""),
    code(r"""s = 3 * kB_eV * T
lat = Lattice(20, a=1.6e-9, seed=4).set_disorder(s)
for p in (1e-3, 1e-2, 3e-2):
    mu = mobility_interacting(lat, T, 0.05 * s, p, n_eq=50000, n_meas=200000)
    print(f"p = {p:.0e}: mu = {mu*1e4:.3e} cm2/Vs")"""),
    exercise("Why does mobility increase with carrier density in the cell above?",
             "Deep tail states fill first, so the remaining carriers need less activation to reach the transport level. This is the EGDM $g_1(p)$ enhancement (tutorial 03)."),
    exercise("Repeat the energy relaxation at $\\hat\\sigma=3$. Why does the final value miss $-3\\sigma$ on a 30³ lattice?",
             "With $\\hat\\sigma=3$, the equilibrium energy lies $3\\sigma$ deep, where a 27 000-site sample has very few states. The finite sample cannot represent the equilibrium tail (V4 calls this the finite-sample limit)."),
]))

# ----------------------------------------------------------------------------- 03
NB.append(notebook("03", [
    md(r"""**Goals.** Solve the master equation (ME) for the mobility, compare it with KMC, and reproduce the
extended Gaussian disorder model (EGDM) of Pasveer *et al.* (PRL 94, 206601, 2005).

### Theory
In steady state, $\sum_j [W_{ij}p_i(1-p_j) - W_{ji}p_j(1-p_i)]=0$. Linearising around Fermi–Dirac
occupations $f_i$ gives a resistor network with conductances

$$G_{ij} = \frac{e^2}{k_BT}\,W_{ij}\,f_i(1-f_j),$$

solved by preconditioned CG. The EGDM factorises the mobility as
$\mu(T,p,F)=\mu_0(T)\,g_1(p,T)\,g_2(F,T)$ with $\mu_0(T)=c_1\frac{a^2\nu_0 e}{\sigma}e^{-c_2\hat\sigma^2}$,
$c_1=1.8\times10^{-9}$, $c_2=0.42$."""),
    code(HEAD),
    code(r"""from orchid.lattice import Lattice
from orchid.master_equation import me_linear_response
from orchid.kmc import mobility_independent
from orchid.models import egdm_pasveer
from orchid.constants import kB_eV
sigma, a = 0.1, 1.6e-9
shats = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
mu_me, mu_eg = [], []
for sh in shats:
    T = sigma / (sh * kB_eV)
    lat = Lattice(24, a=a, seed=1).set_disorder(sigma)
    mu_me.append(me_linear_response(lat, T, 1e-7, alpha=10.0)[0])
    mu_eg.append(egdm_pasveer(T, 1e-7, 0.0, sigma, a, nu0=1e12))
plt.semilogy(shats**2, mu_me, "o", label="ORCHID master equation")
plt.semilogy(shats**2, mu_eg, "-", label="EGDM (Pasveer 2005)")
plt.xlabel(r"$\hat\sigma^2$"); plt.ylabel(r"$\mu$ (m$^2$/Vs)"); plt.legend();"""),
    md("Both curves are close to straight lines in $\\hat\\sigma^2$. Now cross-check the ME against KMC at one temperature:"),
    code(r"""T = sigma / (2.5 * kB_eV)
lat = Lattice(24, a=a, seed=1).set_disorder(sigma)
me = me_linear_response(lat, T, 1e-7, alpha=10.0)[0]
kmc, err = mobility_independent(lat, T, 0.05 * sigma, alpha=10.0, n_carriers=64, n_eq=100000, n_meas=300000)
print(f"ME {me:.3e}   KMC {kmc:.3e} +- {err:.1e}   ratio {kmc/me:.3f}")"""),
    md(r"""### Density dependence and the Einstein relation"""),
    code(r"""from orchid.models import egdm_g1, einstein_ratio_gaussian
p = np.logspace(-6, -1, 30)
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
for sh in (2, 3, 4):
    T = sigma / (sh * kB_eV)
    ax[0].loglog(p, egdm_g1(p, T, sigma), label=rf"$\hat\sigma$={sh}")
    ax[1].semilogx(p, [einstein_ratio_gaussian(x, T, sigma) for x in p], label=rf"$\hat\sigma$={sh}")
ax[0].set_xlabel("p (per site)"); ax[0].set_ylabel("$g_1$"); ax[0].legend()
ax[1].set_xlabel("p (per site)"); ax[1].set_ylabel(r"$g_3=eD/\mu kT$");"""),
    exercise("Use `orchid.master_equation.me_linear_response` at three densities $p=10^{-5},10^{-3},10^{-2}$ for $\\hat\\sigma=3$ and compare with `egdm_g1`.",
             "```python\nT = sigma/(3*kB_eV); lat = Lattice(24, a=a, seed=1).set_disorder(sigma)\nm0 = me_linear_response(lat, T, 1e-7)[0]\nfor p in (1e-5, 1e-3, 1e-2):\n    print(p, me_linear_response(lat, T, p)[0]/m0, egdm_g1(p, T, sigma))\n```\nThe two agree to within tens of percent (V2)."),
]))

# ----------------------------------------------------------------------------- 04
NB.append(notebook("04", [
    md(r"""**Goals.** Use the coupled-Newton drift–diffusion (DD) solver for single-carrier diodes (SCLC), solar
cells and OLEDs, and for multilayer heterostructures.

### Theory
In 1D, ORCHID solves

$$\frac{d}{dx}\!\left(\varepsilon\frac{d\psi}{dx}\right) = -q(p-n+N_D-N_A),\qquad
\frac{1}{q}\frac{dJ_n}{dx}=R-G,\qquad -\frac{1}{q}\frac{dJ_p}{dx}=R-G,$$

with $J_n=q\mu_n nF+qD_n n'$. Here $D=g_3\mu k_BT/q$, the recombination is Langevin,
$R=\gamma(np-n_i^2)$ with $\gamma=q(\mu_n+\mu_p)/\varepsilon$, and fluxes use Scharfetter–Gummel discretisation.

The Mott–Gurney limit is $J=\tfrac98\varepsilon\mu V^2/L^3$."""),
    code(HEAD),
    code(r"""from orchid.drift_diffusion import Device, mott_gurney
mu, L = 1e-8, 100e-9
d = Device(L=L, N=201, Eg=2.4, phi_anode_p=0.0, phi_cathode_n=1.0, mu_n=1e-14, mu_p=mu,
           recombination=False)
V = np.logspace(0.3, 1.5, 12)
J = d.jv(V)["J"]
Vbi = 1.4
plt.loglog(V, J, "o", label="ORCHID DD")
plt.loglog(V, mott_gurney(V - Vbi, L, 3.5, mu), "-", label=r"Mott-Gurney, $V-V_{bi}$")
plt.xlabel("V (V)"); plt.ylabel(r"J (A/m$^2$)"); plt.legend();"""),
    md(r"""### Organic solar cell
The open-circuit voltage follows Koster *et al.* (APL 86, 123509): $V_{oc}=\frac{E_g}{q}-\frac{kT}{q}\ln\frac{(1-P)\gamma N_c^2}{PG}$."""),
    code(r"""from orchid.drift_diffusion import pv_metrics, koster_voc
Gs = np.logspace(25, 28, 6)
Voc, Vk = [], []
for G in Gs:
    dev = Device(L=100e-9, N=161, Eg=1.2, mu_n=1e-8, mu_p=1e-8, phi_anode_p=0.1, phi_cathode_n=0.1, G=G)
    r = dev.jv(np.linspace(0, 0.9, 37))
    Voc.append(pv_metrics(r["V"], r["J"])["Voc"])
    gam = 1.602e-19 * 2e-8 / (3.5 * 8.854e-12)
    Vk.append(koster_voc(1.2, 300.0, dev.Nc, gam, G))
plt.semilogx(Gs, Voc, "o", label="DD"); plt.semilogx(Gs, Vk, "-", label="Koster formula")
plt.xlabel(r"G (m$^{-3}$s$^{-1}$)"); plt.ylabel(r"$V_{oc}$ (V)"); plt.legend()
print("slope / (kT/e):", np.polyfit(np.log(Gs), Voc, 1)[0] / 0.025852)"""),
    md("### OLED with emission zone and roll-off (continuum excitons)"),
    code(r"""from orchid.devices import simulate_oled
dev = Device(L=100e-9, Eg=2.6, mu_n=1e-9, mu_p=1e-8, phi_anode_p=0.1, phi_cathode_n=0.1)
out = simulate_oled(dev, np.linspace(2.8, 9, 8), eta_ST=1.0, qPL=0.9, kSS=1e-17, kSP=1e-18)
plt.semilogx(np.array(out["J"]) / 10, 100 * np.array(out["EQE"]), "o-")
plt.xlabel(r"J (mA/cm$^2$)"); plt.ylabel("EQE (%)");"""),
    md(r"""### Multilayer heterostructures
`MultilayerDevice` takes a list of layers, each with its own $E_g$, $\chi$, $\varepsilon_r$ and mobilities.
Band offsets appear at the interfaces. V19 validates it against NIST Sesame to $3\times10^{-6}$."""),
    code(r"""from orchid.drift_diffusion import MultilayerDevice
layers = [dict(d=50e-9, Eg=1.9, chi=3.0, eps_r=3.5, mu_n=1e-9, mu_p=1e-8, Nc=2.5e26, Nv=2.5e26, G=1e27),
          dict(d=50e-9, Eg=1.9, chi=3.8, eps_r=3.8, mu_n=1e-8, mu_p=1e-9, Nc=2.5e26, Nv=2.5e26, G=1e27)]
m = MultilayerDevice(layers, W_anode=4.8, W_cathode=3.9, nodes_per_layer=60, recombination=False,
                     srh=dict(Nt=1e21, midgap=True, Cn=1e-15, Cp=1e-15, charged=False))
r = m.jv(np.linspace(0, 1.0, 21))
plt.plot(r["V"], r["J"]); plt.xlabel("V (V)"); plt.ylabel(r"J (A/m$^2$)"); plt.title("D/A bilayer under illumination");"""),
    exercise("Halve the Langevin prefactor (`langevin_prefactor=0.5`) in the solar-cell example. How does $V_{oc}$ change?",
             "$V_{oc}$ rises by $(kT/q)\\ln 2\\approx 18$ mV, because $V_{oc}$ depends on $\\ln\\gamma$."),
]))

# ----------------------------------------------------------------------------- 05
NB.append(notebook("05", [
    md(r"""**Goals.** Simulate transient experiments (photo-CELIV, TPV, TPC) and small-signal spectroscopy
(admittance, IMPS, IMVS) with the transient DD solver.

### Theory
In CELIV, a voltage ramp $V=At$ extracts the carriers. Juška's formula gives
$\mu=\dfrac{2d^2}{3At_{max}^2}$ when the extracted charge is small. Small-signal analysis linearises the DD
equations about a dc operating point: $Y(\omega)=G+i\omega C$. For trap-free SCLC,
$C(0)=\tfrac34C_{geo}$ and $\tau_{dc}=\tfrac43L^2/(\mu V)$."""),
    code(HEAD),
    code(r"""from orchid.drift_diffusion import Device
from orchid.transient import celiv
mu = 1e-8
fig, ax = plt.subplots()
for A in (1e5, 1e6, 1e7):
    t_est = 1e-7 * np.sqrt(2 / (3 * mu * A))
    d = Device(L=100e-9, N=121, Eg=2.0, phi_anode_p=1.0, phi_cathode_n=1.0, mu_n=1e-14, mu_p=mu,
               recombination=False, grid_beta=2.0)
    r = celiv(d, A=A, t_end=12 * t_est, n_t=200, G_pulse=1e30, t_pulse=1e-10)
    ax.plot(r["t"] / r["t_max"], (r["J"] - r["j0"]) / r["dj"], label=f"A={A:.0e} V/s: mu/mu_in={r['mu_juska']/mu:.2f}")
ax.set_xlim(0, 4); ax.set_ylim(-0.1, 1.1); ax.set_xlabel(r"$t/t_{max}$"); ax.set_ylabel(r"$(J-j_0)/\Delta j$"); ax.legend(fontsize=8);"""),
    md("The extracted mobility converges to the input mobility at fast ramps, once $qAt_{max}\\gg kT$."),
    code(r"""from orchid.smallsignal import admittance
from orchid.constants import eps0
L, Vb = 100e-9, 20.0
d = Device(L=L, N=161, Eg=2.4, phi_anode_p=0.0, phi_cathode_n=1.0, mu_n=1e-12, mu_p=mu, recombination=False)
tdc = 4 / 3 * L**2 / (mu * (Vb - 1.4))
f = np.logspace(-2, 1.3, 40) / tdc
r = admittance(d, Vb, f)
Cg = eps0 * 3.5 / L
plt.semilogx(f * tdc, r["C"] / Cg); plt.axhline(0.75, c="k", ls="--", label="3/4")
plt.xlabel(r"$f\,\tau_{dc}$"); plt.ylabel(r"$C/C_{geo}$"); plt.legend();"""),
    code(r"""from orchid.transient import tpv
d = Device(L=100e-9, N=101, Eg=1.2, phi_anode_p=0.1, phi_cathode_n=0.1, mu_n=1e-8, mu_p=1e-8, G=1e27)
r = tpv(d)
print(f"Voc = {r['Voc']:.3f} V, TPV lifetime {r['tau']*1e6:.2f} us, bulk 1/(2 gamma n) = {r['tau_langevin']*1e6:.2f} us")
plt.plot(r["t"] * 1e6, r["dV"] * 1e3); plt.xlabel("t (us)"); plt.ylabel("dV (mV)");"""),
    exercise("Compute an IMPS spectrum with `orchid.smallsignal.imps(dev, 0.0, freqs)` for the illuminated device above and find the frequency of the minimum of Im H.",
             "```python\nfrom orchid.smallsignal import imps\nfreqs = np.logspace(3, 9, 40)\nh = imps(d, 0.0, freqs)['H']\nprint(freqs[np.argmin(h.imag)])\n```\nThe characteristic frequency is about $1/(2\\pi\\tau)$ for the slowest transport or recombination process."),
]))

# ----------------------------------------------------------------------------- 06
NB.append(notebook("06", [
    md(r"""**Goals.** Use the 2D drift–diffusion solver (`orchid.dd2d`) to simulate an organic field-effect
transistor and compare it with the gradual-channel approximation.

### Theory
Gradual channel: $I_D/W=\dfrac{\mu C_i}{L}\left[(V_G-V_T)V_D-\tfrac12V_D^2\right]$ below pinch-off, and
$I_{sat}/W=\dfrac{\mu C_i}{2L}(V_G-V_T)^2$ above it. The 2D solver works on a non-uniform $(x,y)$ grid.
It uses Newton's method with a sparse finite-difference Jacobian built with 15-colour graph colouring."""),
    code(HEAD),
    code(r"""from orchid.dd2d import ofet
mu_in = 1e-6
d = ofet(mu_p=mu_in, Nx=31, ny_s=20)
VGs = np.array([-4.0, -6.0, -8.0, -10.0])
I, z = [], None
for VG in VGs:
    z = d.solve({"gate": VG, "drain": -0.2}, z0=z)
    I.append(-d.contact_current(z, {"gate": VG, "drain": -0.2}, "drain"))
slope = np.polyfit(VGs, I, 1)
mu_lin = abs(slope[0]) * d.L_ch / (d.C_i * 0.2)
VT = -slope[1] / slope[0]
print(f"linear mobility {mu_lin:.3e} (input {mu_in:.0e}), V_T = {VT:.2f} V")"""),
    code(r"""VDs = -np.linspace(0.5, 12, 10)
fig, ax = plt.subplots()
for VG in (-6.0, -10.0):
    z, Id = None, []
    for VD in VDs:
        z = d.solve({"gate": VG, "drain": VD}, z0=z)
        Id.append(-d.contact_current(z, {"gate": VG, "drain": VD}, "drain"))
    Vov = abs(VG - VT)
    gca = mu_in * d.C_i / d.L_ch * np.where(np.abs(VDs) < Vov, Vov * np.abs(VDs) - VDs**2 / 2, Vov**2 / 2)
    ax.plot(-VDs, np.array(Id) * 1e3, "o", label=f"2D DD, VG={VG:.0f}")
    ax.plot(-VDs, gca * 1e3, "-", c=ax.lines[-1].get_color())
ax.set_xlabel(r"$-V_D$ (V)"); ax.set_ylabel(r"$-I_D/W$ (mA/m)"); ax.legend();"""),
    exercise("Double the oxide thickness `t_ox` in `ofet(...)`. What happens to $C_i$ and to the drain current?",
             "$C_i=\\varepsilon_0\\varepsilon_{ox}/t_{ox}$ halves, so at the same overdrive the current also halves."),
]))

# ----------------------------------------------------------------------------- 07
NB.append(notebook("07", [
    md(r"""**Goals.** Compute fields in thin-film stacks with the transfer-matrix method (TMM), turn them into
generation profiles under AM1.5G, and analyse OLED outcoupling with the dipole model.

### Theory
Each interface contributes $I_{jk}=\frac{1}{t_{jk}}\begin{pmatrix}1&r_{jk}\\r_{jk}&1\end{pmatrix}$ and each layer
$L_j=\mathrm{diag}(e^{-i\xi_jd_j},e^{i\xi_jd_j})$ with $\xi_j=2\pi\tilde n_j/\lambda$.
The absorbed photon flux density is $\propto \frac{4\pi n k}{\lambda}|E|^2$.
The dipole-emission model integrates the power spectrum $K(u)$ over in-plane wavevectors
$u=k_\parallel/k$ (CPS theory)."""),
    code(HEAD),
    code(r"""from orchid.optics import tmm_field, generation_profile, lorentz_nk
from orchid.materials import nk, am15g, available
print("bundled n,k data:", available())
stack = [("ITO", nk("ITO"), 120e-9), ("PEDOT:PSS", nk("PEDOT-PSS"), 40e-9),
         ("P3HT:PCBM", nk("P3HT-PCBM"), 100e-9), ("Al", nk("Al"), 100e-9)]
lams = np.linspace(350e-9, 800e-9, 46)
x, G, A = generation_profile(stack, 2, lams)
plt.plot(x * 1e9, G); plt.xlabel("depth in active layer (nm)"); plt.ylabel(r"G (m$^{-3}$s$^{-1}$)");
print(f"mean G = {G.mean():.2e} m^-3 s^-1  ->  J_max = {1.602e-19*G.mean()*100e-9:.1f} A/m^2")"""),
    code(r"""lam = np.linspace(300e-9, 1200e-9, 300)
plt.plot(lam * 1e9, am15g(lam)); plt.xlabel("wavelength (nm)"); plt.ylabel("AM1.5G (W m$^{-2}$ m$^{-1}$)");"""),
    md("### Dipole emission and outcoupling"),
    code(r"""from orchid.optics import dipole_emission, outcoupling_classical
lam2 = 520e-9
etl = np.linspace(20e-9, 200e-9, 19)
eta = [dipole_emission(lam2, [(1.75, 30e-9), (nk("ITO")(lam2), 100e-9), (nk("BK7")(lam2).real, 0)],
                       (1.75, t), [(nk("Al")(lam2), 0)], 0.0, u_max=4, n_ev=8000)["eta_air"] for t in etl]
plt.plot(etl * 1e9, eta, "o-", label="dipole model (isotropic)")
plt.axhline(outcoupling_classical(1.75), c="k", ls="--", label=r"$1/2n^2$")
plt.xlabel("emitter-cathode distance (nm)"); plt.ylabel(r"$\eta_{air}$"); plt.legend();"""),
    exercise("Change the emitter to horizontal dipoles (`theta=0`). Why does the outcoupling rise?",
             "Horizontal dipoles emit mostly near the surface normal, inside the escape cone, and couple less to surface plasmons. That is why oriented emitters reach EQE above 30 %."),
]))

# ----------------------------------------------------------------------------- 08
NB.append(notebook("08", [
    md(r"""**Goals.** Measure exciton diffusion with KMC, solve the steady-state exciton continuity equation with
annihilation, and run exciton KMC inside a device box.

### Theory
$D=\frac16\sum_jk_jr_j^2$ (3D) and $L_D=\sqrt{D\tau}$. Continuum steady state:
$D n''-n/\tau-\gamma_{TT}n^2-k_{SP}n\,p+G=0$. In KMC, TTA appears as a pair rate $k_{TT}(R_{TT}/r)^6$
and TPQ as a quenching rate set by nearby polarons."""),
    code(HEAD),
    code(r"""from orchid.lattice import Lattice
from orchid.exciton import exciton_kmc
for s in (0.0, 0.05, 0.1):
    lat = Lattice(30, a=1.6e-9, seed=1).set_disorder(max(s, 1e-6))
    r = exciton_kmc(lat, 300.0, tau=1e-9, R0=2.0, n_excitons=4000)
    print(f"sigma={s:.2f} eV: D = {r['D']:.2e} m^2/s, L_D = {r['L_D']*1e9:.1f} nm")"""),
    md("Energetic disorder traps excitons in low-energy sites and shortens $L_D$ (V7)."),
    code(r"""from orchid.exciton_device import ExcitonDeviceKMC
Gs = np.logspace(28, 30, 5)
gam = []
for G in Gs:
    x = ExcitonDeviceKMC((16, 16, 16), tau_r=1e-6, tau_nr=4e-6, R0=1.5, periodic_z=True, k_TT0=1e8, R_TT=1.5)
    r = x.run(G, n_events=150000, n_eq=50000)
    gam.append(r["frac_TTA"] * G / r["n_ex"]**2)
plt.semilogx(Gs, gam, "o-"); plt.xlabel(r"G (m$^{-3}$s$^{-1}$)"); plt.ylabel(r"$\gamma_{TT}$ (m$^3$/s)")
plt.ylim(0, 2 * max(gam)); plt.title("TTA loss / n$^2$ is constant: bimolecular law");"""),
    exercise("Set `tau_nr` very large and all annihilation to zero. What should `eta_rad` be?",
             "1, because every exciton then decays radiatively. In general $\\eta_{rad}=k_r/(k_r+k_{nr})$, and V24 checks this limit in the full device."),
]))

# ----------------------------------------------------------------------------- 09
NB.append(notebook("09", [
    md(r"""**Goals.** Run the full 3D device KMC (electrodes, injection, Coulomb and image charges, multilayer
stack), then the v0.4 concurrent charge + exciton engine.

### Theory
Holes and electrons hop with MA rates on the HOMO/LUMO energies of each layer. The electrodes inject and
extract over image-lowered barriers. The current follows the Ramo–Shockley theorem:
$J=\frac{e}{LA\,t}\sum_{\text{hops}}q\,\Delta z$. Electrostatics = mean-field sheets (1D Poisson Green's function)
+ exact screened Coulomb within $r_{coul}$ minus its sheet part.

**New in v0.4 (`CoupledDeviceKMC`).** When a hole and an electron meet, an exciton is created on that site
inside the same event loop. Charges and excitons share one clock: $R=R_c+R_x$, with two Fenwick trees.
Exciton–polaron quenching sees the *live* polaron positions. The v0.3 pipeline used a polaron
snapshot frozen from the mean density profile."""),
    code(HEAD),
    code(r"""from orchid.device_kmc import DeviceKMC, Layer
STACK = [Layer(6, HOMO=-5.2, LUMO=-2.0, name="HTL"), Layer(8, HOMO=-5.4, LUMO=-2.7, name="EML"),
         Layer(6, HOMO=-6.0, LUMO=-2.8, name="ETL")]
d = DeviceKMC(STACK, Lx=12, Ly=12, W_anode=5.1, W_cathode=2.9, coulomb="full", seed=2)
r = d.run(6.0, n_eq=50000, n_meas=200000, mf_update=300)
print(f"J = {r['J']:.3e} A/m^2, gamma_b = {r['gamma_b']:.3f}")
fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
ax[0].semilogy(r["z"] * 1e9, r["p_z"] + 1, label="holes"); ax[0].semilogy(r["z"] * 1e9, r["n_z"] + 1, label="electrons")
ax[0].set_xlabel("z (nm)"); ax[0].set_ylabel(r"density (m$^{-3}$)"); ax[0].legend()
ax[1].plot(r["z"] * 1e9, r["rec_profile"]); ax[1].set_xlabel("z (nm)"); ax[1].set_ylabel("recombination profile");"""),
    md("Charges pile up at the HTL/EML and EML/ETL energy steps, and recombination is concentrated there."),
    code(r"""from orchid.coupled_kmc import CoupledDeviceKMC
c = CoupledDeviceKMC(STACK, Lx=12, Ly=12, W_anode=5.1, W_cathode=2.9, coulomb="full", seed=2,
                     tau_r=1e-6, tau_nr=1e-5, k_TT0=1e8, k_q0=1e8, eta_out=0.2)
rc = c.run(6.0, n_eq=50000, n_meas=200000, mf_update=300)
x = rc["excitons"]
print(x)
print(f"EQE = {100*rc['EQE']:.2f} %  (radiative {rc['eta_rad']:.2f}, TPQ {rc['frac_TPQ']:.2f}, TTA {rc['frac_TTA']:.2f})")
print("bookkeeping:", x["formed"] + x["alive_start"] - (x["radiative"] + x["nonradiative"] + x["TPQ"]
                                                    + x["TTA"] + x["overflow"] + x["alive_end"]))"""),
    exercise("Turn quenching off (`k_q0=0, k_TT0=0`) and verify that `eta_rad` approaches $k_r/(k_r+k_{nr})=0.909$.",
             "With $\\tau_r=1\\,\\mu s$ and $\\tau_{nr}=10\\,\\mu s$, $\\eta=1/(1+0.1)=0.909$. Expect agreement within $\\sqrt{\\eta(1-\\eta)/N}$ (V24: within 3σ)."),
]))

# ----------------------------------------------------------------------------- 10
NB.append(notebook("10", [
    md(r"""**Goals.** Go from molecules to hopping graphs: DIPRO transfer integrals (PySCF), dipolar site energies,
amorphous packings, and **MD trajectory import** (GROMACS `.gro`, multi-frame XYZ) with molecules made
whole across periodic boundaries (new in v0.4).

### Theory
DIPRO projects the dimer Fock matrix onto the monomer frontier orbitals: $J_{eff}=\frac{J-\frac12(e_A+e_B)S}{1-S^2}$.
For a symmetric dimer, $|J_{eff}|$ is half the orbital splitting. Transfer integrals decay as $J\propto e^{-\beta r}$."""),
    code(HEAD),
    code(r"""from orchid.morphology import dipro_pyscf
eth = [("C", (0, 0, 0.667)), ("C", (0, 0, -0.667)), ("H", (0, 0.923, 1.232)),
       ("H", (0, -0.923, 1.232)), ("H", (0, 0.923, -1.232)), ("H", (0, -0.923, -1.232))]
for d in (3.6, 4.2, 5.0):
    r = dipro_pyscf(eth, [(s, (x + d, y, z)) for s, (x, y, z) in eth])
    print(f"d = {d} A: J_eff = {1e3*abs(r['J_eff']):.2f} meV, half splitting = {1e3*abs(r['splitting'])/2:.2f} meV")"""),
    md("### Amorphous packing, dipolar energies, hopping graph"),
    code(r"""from orchid.morphology import random_packing, dipolar_site_energies, MolecularGraph
from orchid.master_equation import me_linear_response
box = np.array([14e-9] * 3)
pos = random_packing(700, box, 1.0e-9, seed=2)
E = dipolar_site_energies(pos, box, dipole_debye=4.0)
G = MolecularGraph(pos, box, r_cut=2.6e-9, energies=E)
print(f"{G.N} molecules, a = {G.a*1e9:.2f} nm, sigma = {1e3*E.std():.0f} meV")
print("mu(300 K) =", me_linear_response(G, 300.0, 1e-6, alpha=10.0)[0], "m^2/Vs")"""),
    md(r"""### MD trajectory import (v0.4)
`read_gro`/`read_xyz_trajectory` read the snapshots. `molecule_centres` unwraps every atom by the minimum
image relative to the first atom of its molecule, then takes the mass-weighted centre.
`trajectory_mobility` builds one graph per frame and averages the ME mobility over the frames."""),
    code(r"""import tempfile, os
from orchid.morphology import write_gro, read_gro, molecule_centres, trajectory_mobility
rng = np.random.default_rng(0)
ring = np.array([[np.cos(t), np.sin(t), 0] for t in np.arange(6) * np.pi / 3]) * 0.14e-9
box = np.array([10e-9] * 3)
centres = random_packing(500, box, 0.9e-9, seed=1)
frames = []
for f in range(3):                                   # three 'MD snapshots' with thermal jitter
    c = (centres + rng.normal(0, 0.05e-9, centres.shape)) % box
    xyz, atom, resid = [], [], []
    for i, ci in enumerate(c):
        Q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        xyz += [((ci + p) % box) for p in ring @ Q.T]; atom += ["C"] * 6; resid += [i + 1] * 6
    frames.append(dict(title="demo", resid=np.array(resid), resname=["BEN"] * len(atom), atom=atom,
                       xyz=np.array(xyz), box=box))
path = os.path.join(tempfile.mkdtemp(), "traj.gro")
write_gro(path, frames)
fr = read_gro(path)
cen, mol = molecule_centres(fr[0])
print(len(fr), "frames,", len(cen), "molecules")
r = trajectory_mobility(fr, 300.0, energies="gaussian", sigma=0.075, r_cut=2.0e-9)
print("mobility per frame:", r["mu_frames"], " mean", r["mu_mean"], "+-", r["mu_sem"])"""),
    md(r"""Each snapshot is a separate static-disorder sample (new positions and new site energies), so in a
small box the per-frame mobilities scatter strongly. Production runs use larger boxes and more frames,
and report the frame average with its standard error, as here."""),
    exercise("Why would the naive average of atom positions fail for a molecule that straddles the box edge?",
             "Atoms near $x=0$ and near $x=L$ would average to about $L/2$, far from either copy of the molecule. `molecule_centres` unwraps first, and V26 recovers 397 split molecules exactly."),
]))

# ----------------------------------------------------------------------------- 11
NB.append(notebook("11", [
    md(r"""**Goals.** Extract model parameters from data by exhaustive (zooming) grid search, train Gaussian-process
surrogates of ORCHID simulations, and run Bayesian inversion with credible intervals.

### Theory
Fits minimise the log-RMSE $\sqrt{\langle(\ln y_{model}-\ln y_{data})^2\rangle}$. A GP gives a posterior mean
$\bar f_*=k_*^\top(K+\sigma_n^2I)^{-1}y$ and a variance. Bayesian inversion evaluates
$p(\theta|d)\propto e^{-\chi^2/2}p(\theta)$ on a grid and reads off marginals and 95 % intervals."""),
    code(HEAD),
    code(r"""from orchid.models import gdm_bassler
from orchid.search import zoom_search, log_rmse
TT, FF = np.meshgrid([200, 240, 280, 320.0], [1e4, 1e5, 4e5, 1e6], indexing="ij")
data = gdm_bassler(TT, FF, 0.09, 1e-2, Sigma=2.0)
best, err, _ = zoom_search(lambda p: log_rmse(gdm_bassler(TT, FF, p["sigma"], p["mu_inf"], Sigma=p["Sigma"]), data),
                           {"sigma": (0.05, 0.15), "mu_inf": (1e-5, 1.0), "Sigma": (1.5, 4)},
                           n=11, levels=4, log=("mu_inf",))
print(best, err)"""),
    code(r"""from orchid.ml import GPSurrogate
rng = np.random.default_rng(1)
X = rng.uniform(0, 1, (40, 2)); y = np.sin(6 * X[:, 0]) + X[:, 1] ** 2
gp = GPSurrogate().fit(X[:30], y[:30])
pr, sd = gp.predict(X[30:], return_std=True)
plt.errorbar(y[30:], pr, yerr=2 * sd, fmt="o"); plt.plot([-1, 2], [-1, 2], "k-", lw=0.8)
plt.xlabel("truth"); plt.ylabel("GP prediction (2 sd)");"""),
    code(r"""from orchid.ml import bayes_grid
noisy = data * np.exp(rng.normal(0, 0.05, data.shape))
r = bayes_grid(lambda p: gdm_bassler(TT, FF, p["sigma"], 10 ** p["lmu"], Sigma=p["Sigma"]), noisy,
               dict(sigma=np.linspace(0.08, 0.10, 31), lmu=np.linspace(-2.5, -1.5, 31), Sigma=np.linspace(1.5, 2.5, 21)),
               0.05 / np.log(10))
print("MAP", r["map"]); print("95 % CI", r["ci95"])"""),
    exercise("How many model evaluations would a full 31×31×21 grid need, and how many did `zoom_search` use with n=11, levels=4?",
             "20 181 for the full grid. Zoom search uses $4\\times11^3=5324$ and reaches finer resolution near the optimum."),
]))

# ----------------------------------------------------------------------------- 12
NB.append(notebook("12", [
    md(r"""**Goals.** Use the parallel layers of ORCHID: Numba threads, process pools and MPI task farms; the CUDA
kernels; and the synchronous-sublattice interacting KMC with the v0.4 $\tau\to0$ extrapolation.

### Theory
Amdahl's law: $S(P)\le 1/(s+(1-s)/P)$. The synchronous-sublattice KMC (Shim & Amar 2005) divides the lattice
into blocks of side $B\ge2r_c$ coloured like an 8-colour checkerboard. Blocks of one colour evolve
independently for a window $\tau$. The result is exact only as $\tau\to0$, so v0.4 fits
$\mu(\tau)=\mu_0+c_1\tau+c_2\tau^2$ and reports $\mu_0$."""),
    code(HEAD),
    code(r"""from orchid.parallel import describe, parallel_map
print(describe())
def work(seed):
    rng = np.random.default_rng(seed)
    return rng.normal(size=200000).std()
print(parallel_map(work, range(4), workers=2))"""),
    md("Under MPI (`mpiexec -n 64 python script.py`), the same `parallel_map` becomes a block-cyclic task farm. Seeds belong to tasks, so the results do not depend on the number of ranks."),
    code(r"""from orchid.lattice import Lattice
from orchid.sublattice_kmc import mobility_sublattice_extrapolated
from orchid.kmc import mobility_interacting
from orchid.constants import kB_eV
T = 300.0; s = 2.5 * kB_eV * T
lat = Lattice(16, a=1.6e-9, seed=3).set_disorder(s)
ref = np.mean([mobility_interacting(lat, T, 0.05 * s, 0.03, n_eq=100000, n_meas=600000, seed=k) for k in (1, 2, 3)])
mu0, info = mobility_sublattice_extrapolated(lat, T, 0.05 * s, 0.03, factors=(10, 5, 2.5),
                                             sim_time_factor=120000, seeds=(1, 2, 3))
x = info["taus"] / info["tau0"]
plt.errorbar(x, info["mu"] / ref, yerr=info["mu_err"] / ref, fmt="o", label="single windows")
xx = np.linspace(0, x.max(), 50); plt.plot(xx, np.polyval(info["coef"][::-1], xx) / ref, "-")
plt.errorbar([0], [mu0 / ref], yerr=[info["err"] / ref], fmt="*", ms=12, label=r"extrapolated $\tau\to0$")
plt.axhline(1, c="k", ls="--", lw=1, label="serial KMC")
plt.xlabel(r"$\tau/\tau_0$"); plt.ylabel(r"$\mu/\mu_{serial}$"); plt.legend();"""),
    code(r"""from orchid import gpu
print("CUDA GPU available:", gpu.available())
# with a GPU: gpu.mobility_independent_gpu(lat, T, 0.05*s, n_carriers=65536)"""),
    exercise("With $s=5\\%$ serial work, what is the maximum speed-up on 1000 cores?",
             "$1/(0.05+0.95/1000)=19.6$. This is why ORCHID parallelises the outer ensemble loops as well."),
]))

# ----------------------------------------------------------------------------- 13
NB.append(notebook("13", [
    md(r"""**Goals.** Understand how ORCHID is validated (V1–V27), read the benchmark report, and re-run any
benchmark yourself.

The suite has three kinds of checks:
* **Exact limits:** detailed balance, Madelung constant, Airy formula, $\eta=k_r/(k_r+k_{nr})$, exciton bookkeeping.
* **Published results:** Pasveer EGDM, Koster $V_{oc}$, Mott–Gurney, Juška CELIV, Martens $\tfrac34C_{geo}$, the gradual channel.
* **Other codes:** NIST Sesame (drift–diffusion) and `tmm` (optics), compared on identical inputs.

The tolerances in `validation/criteria.py` are fixed in advance."""),
    code(HEAD),
    code(r"""import os, sys
root = os.path.abspath(os.path.join(os.getcwd(), "..", ".."))
sys.path.insert(0, os.path.join(root, "validation"))
from criteria import evaluate
rows = evaluate()
for r in rows:
    st = {True: "PASS", False: "FAIL", None: "INFO"}[r["passed"]]
    print(f"{r['id']:>4} {st}  {r['what']:<32} {r['value'][:70]}")
print(sum(r["passed"] is True for r in rows), "/", sum(r["passed"] is not None for r in rows), "pass")"""),
    md(r"""Re-run from a terminal:

```bash
python validation/run_all.py --run          # everything (about an hour on 2 cores)
python validation/run_all.py --run V24 V25  # selected benchmarks
orchid bench                                # evaluate existing results
```"""),
    code(r"""from IPython.display import Image, display
fig = os.path.join(root, "validation", "figures", "v24_coupled.png")
if os.path.exists(fig):
    display(Image(fig, width=700))"""),
    exercise("A KMC estimate gives $\\eta=0.897\\pm0.008$ against an exact 0.909. Is this a pass under a 3σ criterion?",
             "$z=(0.897-0.909)/0.008=-1.5$, so $|z|<3$ and it passes."),
]))


def main():
    os.makedirs(OUT, exist_ok=True)
    sel = [a for a in sys.argv[1:] if a[:2].isdigit()]
    for name, nb in NB:
        if not sel or name[:2] in sel:
            if not any('gs-watermark' in c.source for c in nb.cells):
                nb.cells.insert(0, nbf.v4.new_markdown_cell("<!-- gs-watermark -->\n> **© 2026 Gokaran Shukla** · part of [orchid](https://github.com/Gokaran-Shukla/orchid) · MIT License — any reuse must keep this copyright notice and credit the author."))
            nbf.write(nb, os.path.join(OUT, name))
    if "--execute" in sys.argv:
        from nbclient import NotebookClient
        only = [a for a in sys.argv[1:] if a[:2].isdigit()]
        failed = []
        for name, _ in NB:
            if only and name[:2] not in only:
                continue
            path = os.path.join(OUT, name)
            nb = nbf.read(path, as_version=4)
            import time
            t0 = time.time()
            try:
                NotebookClient(nb, timeout=1800, kernel_name="python3",
                               resources={"metadata": {"path": OUT}}).execute()
            except Exception as e:  # report and continue with the next notebook
                failed.append(name)
                print(f"FAILED {name}: {str(e)[-1500:]}", flush=True)
                continue
            nbf.write(nb, path)
            print(f"executed {name} in {time.time()-t0:.0f} s", flush=True)
    print("wrote", len(NB), "notebooks to", OUT)
    if "--execute" in sys.argv and failed:
        sys.exit(f"failed: {failed}")


if __name__ == "__main__":
    main()
