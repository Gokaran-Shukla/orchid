# ORCHID roadmap

This file is generated from `orchid/roadmap.py`; `orchid roadmap` prints it. Every item has an importable module, a validation benchmark (`validation/`) and tests.

Status: **all 23 items done** (v0.1 to v0.4).

## v0.1: multiscale core

| ID | Status | Item | Module | Validation |
|---|---|---|---|---|
| R01 | done | Hopping rates: MA, symmetric MA, Marcus, MLJ, Förster, Dexter | `orchid.rates` | detailed balance, Marcus inverted region |
| R02 | done | GDM / dipolar CDM / exponential / host-guest disorder; positional and off-diagonal | `orchid.lattice` | disorder statistics, spatial correlation |
| R03 | done | Bulk KMC: independent carriers, TOF, energy relaxation | `orchid.kmc` | Pasveer 2005, Bässler 1993 -sigma^2/kT |
| R04 | done | Master equation: linear-response resistor network + nonlinear | `orchid.master_equation` | ME = KMC; EGDM g1 |
| R05 | done | 1D coupled-Newton drift-diffusion, EGDM + g3, Langevin/SRH/Braun | `orchid.drift_diffusion` | Mott-Gurney, Koster 2005 Voc, NIST Sesame |
| R06 | done | Transfer-matrix optics and generation profiles | `orchid.optics` | Airy formula, tmm (Byrnes) package |
| R07 | done | Exciton KMC (Förster/Dexter) and emission-zone continuum with annihilation | `orchid.exciton` | D = sum k r^2 / 6 |
| R08 | done | Exhaustive grid / zoom / Latin-hypercube search | `orchid.search` | parameter recovery |

## v0.2: 3D device KMC, transients, massive parallelism

| ID | Status | Item | Module | Validation |
|---|---|---|---|---|
| R09 | done | Hybrid parallelism: MPI task farm x processes x Numba threads | `orchid.parallel` | MPI result identical to serial; scaling benchmark |
| R10 | done | CUDA KMC kernel (one walker per thread) and CuPy master-equation CG | `orchid.gpu` | CUDA-simulator test vs CPU engine |
| R11 | done | O(log N) interacting KMC: Fenwick tree + incremental local updates | `orchid.kmc` | identical to dense reference engine; 27-135x faster |
| R12 | done | Ewald summation, periodic lattice Green's function, image charges, sheet Green's function | `orchid.electrostatics` | NaCl Madelung constant 1.747565 (1e-9) |
| R13 | done | Full 3D device KMC: electrodes, injection/extraction, image charge, Coulomb, excitons, multilayer stacks | `orchid.device_kmc` | zero-bias equilibrium (J=0); vs 1D DD (van der Holst 2009); 3-layer OLED |
| R14 | done | Transient DD: CELIV, TPV, TPC, arbitrary V(t)/G(t) | `orchid.transient` | Juska CELIV formula; TPV small perturbation |
| R15 | done | AMG-preconditioned ME and finite-size extrapolation | `orchid.master_equation` | 10-20x fewer CG iterations |

## v0.3: device-physics completeness

| ID | Status | Item | Module | Validation |
|---|---|---|---|---|
| R16 | done | Exciton KMC in devices: generation from device KMC, Förster hopping, TTA, TPQ, metal quenching -> EQE roll-off | `orchid.exciton_device` | eta = kr/(kr+knr) exactly; TTA loss = gamma n^2 (constant gamma) |
| R17 | done | Multilayer heterostructure DD (band offsets, per-layer materials) with Gaussian-DOS contact statistics | `orchid.drift_diffusion` | D/A heterojunction vs NIST Sesame |
| R18 | done | Small-signal impedance / admittance spectroscopy, IMPS, IMVS | `orchid.smallsignal` | insulator C=eps/L; SCLC C(0)=3/4 C_geo, f_max tau_dc = 0.72 |
| R19 | done | 2D drift-diffusion for OFETs and bulk-heterojunction morphologies (sparse Newton, 15-colour Jacobian) | `orchid.dd2d` | OFET vs gradual-channel model (linear + saturation mobility) |
| R20 | done | Dipole-emitter outcoupling (CPS / plane-wave expansion), ASTM G173 spectrum, refractiveindex.info n,k database | `orchid.optics` | CPS perfect-mirror formulas; BK7 n_d = 1.5168; AM1.5G = 1000 W/m^2 |

## v0.4: first principles, machine learning, parallel-in-space KMC

| ID | Status | Item | Module | Validation |
|---|---|---|---|---|
| R21 | done | Molecular morphologies + DFT transfer integrals (DIPRO/PySCF) + dipolar site energies on hopping graphs | `orchid.morphology` | graph = lattice ME exactly; DIPRO J_eff = half dimer splitting |
| R22 | done | Gaussian-process surrogates of mu(T,p) trained on ORCHID data; exhaustive-grid Bayesian inversion | `orchid.ml` | held-out RMSE; truth inside 95 % credible intervals |
| R23 | done | Parallel-in-space interacting KMC (synchronous sublattice), CPU threads + CUDA kernel | `orchid.sublattice_kmc` | agrees with serial Fenwick KMC; CUDA simulator test |

## Parallel strategy

| Level | Mechanism | Scales to | Used by |
|---|---|---|---|
| GPU | CUDA kernels: independent walkers, synchronous-sublattice interacting KMC; CuPy sparse CG | 10^5 to 10^6 walkers or blocks per GPU | `orchid.gpu` |
| Threads | Numba `prange` (no GIL) | all cores of a node | KMC, exciton KMC, sublattice KMC |
| Processes | `ProcessPoolExecutor` | one node | `parallel_map`, `search`, `ml` |
| MPI | mpi4py block-cyclic task farm | thousands of ranks | disorder ensembles, T/F/p sweeps, J-V sets, search grids |

## Next steps (not yet scheduled)

- Hardware scaling study on a cluster and GPUs (the code paths exist; only 2 CPU cores were available during development).
- Exciton and charge events in a single concurrent device-KMC event loop.
- DFT-level (larger basis) coupling database and automated morphology import from MD trajectories.
