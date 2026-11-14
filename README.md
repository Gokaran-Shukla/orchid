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
| `coupled_kmc` | **v0.4:** charges and excitons in **one** device-KMC event loop: live exciton–polaron quenching, TTA, emission profile, EQE | two Fenwick trees, common clock |
| `sublattice_kmc.mobility_sublattice_extrapolated` | **v0.4:** removes the time-window bias of parallel sublattice KMC (τ → 0) | polynomial extrapolation in τ |
| `morphology` (MD import) | **v0.4:** GROMACS `.gro` / extended-XYZ trajectories, PBC-whole molecular centres, frame-averaged mobility | minimum-image unwrapping |
| `tutorials` | **v0.4:** 13 tutorial notebooks, 65-question quiz bank (`orchid quiz`) | Jupyter |
| `roadmap` | machine-readable roadmap (`orchid roadmap`), see ROADMAP.md | — |

## Install

```bash
pip install -e ".[dev]"     # numpy, scipy, numba (+ pytest, matplotlib, pyamg, tmm)
pip install -e ".[mpi,gpu,qc]" # optional: mpi4py, cupy, pyscf
pytest -q                   # 57 tests, about 4 min
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

## Tutorials and quizzes (v0.4)

Thirteen notebooks in [`docs/tutorials/`](docs/tutorials/README.md) cover every section of the code.
Each has theory with equations, runnable examples, exercises with hidden solutions and a five-question quiz:

```bash
orchid tutorials            # list them
orchid quiz 09              # interactive quiz in the terminal
```

```python
from orchid.tutorials import quiz
quiz("09")                  # show the questions
quiz("09", answers="....")  # grade, with an explanation for every answer
```

## Validation and benchmarks

All 27 benchmarks (V1–V27) are re-run for every release. They are graded against tolerances fixed in
advance in `validation/criteria.py`, and the full report with values and runtimes is in
[BENCHMARKS.md](BENCHMARKS.md).

```bash
python validation/run_all.py --run        # re-run everything (about an hour on 2 cores)
python validation/run_all.py --run V24    # one benchmark
orchid bench                              # re-evaluate stored results
```

| Group | Benchmarks | References |
|---|---|---|
| Transport physics | V1–V4, V10, V11 | Pasveer 2005 (EGDM), Bässler 1993, Scher–Montroll, detailed balance |
| Devices | V5, V6, V12, V13, V17–V19, V23 | Mott–Gurney, Koster 2005, van der Holst 2009, Juška 2000, Martens (¾C_geo), gradual channel |
| Code-to-code | V15, V19 | **NIST Sesame** (ΔJ/Jsc ≤ 3e-6), **tmm** (1e-15) |
| Exact limits | V7, V8, V14, V20, V21, V24, V26 | Σkr²/6, Airy, Madelung, CPS mirror, DIPRO splitting, η = k_r/(k_r+k_nr), exact PBC centres |
| Numerics / parallel | V9, V16, V22, V25, V27 | parameter recovery, MPI = serial, credible-interval coverage, τ→0 extrapolation, throughput |

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
