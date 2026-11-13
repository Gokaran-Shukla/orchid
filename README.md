# ORCHID

**O**rganic **R**ate-based **CH**arge-hopping & **I**ntegrated **D**evice simulator

ORCHID simulates charge and exciton transport in disordered organic semiconductors
at several scales: hopping rates → kinetic Monte Carlo / master equation → mobility
laws μ(T, p, F) → drift-diffusion device models for **OLEDs**, **organic solar
cells** and single-carrier diodes. Exhaustive search ties the scales together for
parameter extraction and design screening. Every module is checked against an
analytical limit or a published result (see `validation/`).

| Module | Physics | Method |
|---|---|---|
| `rates` | Miller–Abrahams, symmetric MA, Marcus, Marcus–Levich–Jortner, Förster, Dexter | closed form, vectorised |
| `lattice` | Gaussian (GDM), dipolar-correlated (CDM), smoothed, exponential, host–guest/trap DOS; positional and off-diagonal disorder | FFT lattice sums |
| `kmc` | non-interacting & interacting carriers (site exclusion + screened Coulomb), mobility, time-of-flight, energetic relaxation | rejection-free BKL/Gillespie, Numba-parallel |
| `master_equation` | Fermi–Dirac master equation: exact linear response (Miller–Abrahams resistor network) and nonlinear field dependence | sparse PCG / self-consistent iteration |
| `models` | Bässler GDM, Novikov CDM, Pasveer EGDM, Poole–Frenkel, Vissenberg–Matters, Gaussian-DOS statistics & generalised Einstein relation | analytic |
| `drift_diffusion` | Poisson + continuity, Scharfetter–Gummel with generalised Einstein, Langevin / SRH (trapped charge) / Braun–Onsager, Schottky/Ohmic contacts | fully coupled Newton, banded colour-FD Jacobian, tanh grid |
| `optics` | coherent transfer matrix (Pettersson 1999), generation profile, escape-cone outcoupling | 2×2 matrices |
| `exciton` | Förster/Dexter exciton KMC; continuum emission zone with S–S and S–P annihilation | KMC + Newton |
| `devices` | OLED EQE = γ_b·η_S/T·q_eff·η_out with roll-off; OPV J–V, Jsc, Voc, FF, PCE | workflow |
| `search` | exhaustive grid, zooming grid, Latin hypercube, log-RMSE | MPI / process parallel |
| `device_kmc` | **full 3D device KMC**: electrodes, injection/extraction, image charges, short- + long-range Coulomb, bipolar transport, exciton formation, multilayer OLED stacks | Fenwick tree, incremental updates, Ramo current |
| `electrostatics` | Ewald sums, periodic lattice Green's function, image-charge series, sheet Green's function | Ewald |
| `transient` | time-domain DD: CELIV, TPV, TPC, arbitrary V(t), G(t), displacement current | implicit Euler + banded Newton |
| `parallel` | hybrid MPI × processes × Numba threads; `parallel_map`, `ensemble` | mpi4py task farm |
| `gpu` | CUDA KMC kernel (one walker per thread), CuPy master-equation CG | Numba CUDA, CuPy |
| `exciton_device` | exciton KMC coupled to device KMC: Förster hopping, TTA, TPQ, metal quenching → EQE roll-off | Fenwick KMC |
| `drift_diffusion.MultilayerDevice` | heterostructures (band offsets, per-layer materials), Gaussian-DOS contacts | coupled Newton |
| `smallsignal` | impedance / admittance, IMPS, IMVS | linearised frequency domain |
| `dd2d` | 2D drift-diffusion: OFETs, bulk-heterojunction morphologies | sparse Newton, 15-colour Jacobian |
| `optics.dipole_emission` | dipole emitters in multilayers: Purcell factor, air/substrate/waveguide/SPP modes | plane-wave expansion (CPS) |
| `materials` | refractiveindex.info n,k data (Al, Ag, Au, ITO, PEDOT:PSS, P3HT:PCBM, SiO2, BK7), ASTM G173 | — |
| `morphology` | amorphous packings, DIPRO transfer integrals (PySCF), dipolar site energies, hopping graphs | VOTCA-style |
| `ml` | Gaussian-process surrogates, exhaustive-grid Bayesian inversion | NumPy |
| `sublattice_kmc` | parallel-in-space interacting KMC (Shim–Amar), CPU threads and CUDA | synchronous sublattice |
| `roadmap` | machine-readable roadmap (`orchid roadmap`), see ROADMAP.md: **23/23 done** | — |

## Install

```bash
pip install -e ".[dev]"     # numpy, scipy, numba (+ pytest, matplotlib, pyamg, tmm)
pip install -e ".[mpi,gpu,qc]" # optional: mpi4py, cupy, pyscf
pytest -q                   # 49 tests, about 3 min
```

## Quick start

```python
from orchid.lattice import Lattice
from orchid.master_equation import me_linear_response
from orchid.kmc import mobility_independent
from orchid.constants import kB_eV

sigma = 0.1; T = sigma / (3 * kB_eV)
lat = Lattice(30, a=1.6e-9, seed=1).set_disorder(sigma)
mu_me, _ = me_linear_response(lat, T, 1e-7, alpha=10.0)        # m^2/Vs
mu_kmc, err = mobility_independent(lat, T, 0.05 * sigma, alpha=10.0)
```

```bash
orchid mobility --method me --sigma 0.1 --T 300 --L 30
orchid jv --Eg 1.2 --G 1e27 --Vmin -0.3 --Vmax 0.9     # illuminated J-V + PV metrics
orchid egdm --T 300 --p 1e23 --F 1e7
```

More in `examples/` (ME vs KMC, solar cell with optics, OLED roll-off, GDM fit).

## Validation (`validation/`, run `python bench_*.py`)

| # | Benchmark | Reference | Result |
|---|---|---|---|
| V1 | μ(T), p→0 | Pasveer *et al.*, PRL 94, 206601 (2005) | ME = KMC within 3–10 %; within 1.2× of EGDM for σ̂ ≤ 3, finite-size deviation above |
| V2 | μ(p) | EGDM g₁ | ME and KMC with exclusion agree within 1–7 %; EGDM matched for σ̂ = 3–4 |
| V3 | μ(F) | EGDM g₂ | KMC matches σ̂ = 3; within about 30 % for σ̂ = 2, 4 |
| V4 | ⟨E⟩(t→∞) | −σ²/k_BT (Bässler 1993) | −1.97σ vs −2σ at σ̂=2; finite-sample limit at σ̂=3 |
| V5 | SCLC | Mott–Gurney | J/J_MG → 1.06 at 40 V (diffusion-corrected approach) |
| V6 | V_oc(G) | Koster *et al.*, APL 86, 123509 (2005) | ≤ 1.3 mV deviation; slope 1.004 k_BT/q |
| V7 | exciton D, σ = 0 | D = Σk r²/6 | within statistical error |
| V8 | optics | Airy formula, energy conservation | 3e-16 / 7e-6 |
| V9 | inverse problems | known synthetic truth | GDM σ, μ∞, Σ within 1 %; DD barrier and mobility recovered |
| V10–V12 | TOF, Marcus vs MA, OLED / OPV demos | qualitative literature trends | — |
| V13 | 3D device KMC | 1D EGDM DD (van der Holst 2009); zero-bias equilibrium | ≤ 1.6× up to 2 V; J(0)=0; e/h mirror symmetry |
| V14 | Ewald / images | NaCl Madelung constant; single-plane image limit | 6e-10; 1e-3 |
| V15 | **code-to-code** | **NIST Sesame** (DD) and **tmm** (optics) | ΔJ/Jsc ≤ 3e-6, ΔVoc 4 nV; 1e-15 |
| V16 | parallel efficiency | ideal | 2.0× on 2 cores for threads, processes and MPI; MPI bit-identical |
| V17 | CELIV, TPV | Juška 2000; 1/(2γn) | μ_CELIV/μ → 1.05; 1.14× (300 nm) |

## Parallel runs

```bash
export NUMBA_NUM_THREADS=8
mpiexec -n 64 python my_study.py     # every parallel_map(...) call spreads over 64 ranks
orchid parallel                      # show ranks, threads, GPU
```

## Units

SI throughout (mobility in m²/Vs; multiply by 1e4 for cm²/Vs). Energies are in eV.
Lattice codes work in units of the lattice constant `a`; `F_red = eFa` (eV).

## Citing

If you use ORCHID, please cite this repository and the original method papers
listed in `docs/slides/orchid_slides.pdf` (References).

MIT licence © 2026 Gokaran Shukla
