# ORCHID roadmap

This file is generated from `orchid/roadmap.py` (`orchid roadmap` prints it). Every item marked done has an importable module and a validation target. The test suite checks both.

Status: **15 of 23 items done** (v0.1 and v0.2). v0.3 and v0.4 are planned.

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

## v0.2: 3D device KMC, transients, massive parallelism (this release)

| ID | Status | Item | Module | Validation |
|---|---|---|---|---|
| R09 | done | Hybrid parallelism: MPI task farm x processes x Numba threads | `orchid.parallel` | MPI result identical to serial; scaling benchmark |
| R10 | done | CUDA KMC kernel (one walker per thread) and CuPy master-equation CG | `orchid.gpu` | CUDA-simulator test vs CPU engine |
| R11 | done | O(log N) interacting KMC: Fenwick tree + incremental local updates | `orchid.kmc` | identical to dense reference engine; 27-135x faster |
| R12 | done | Ewald summation, periodic lattice Green's function, image charges, sheet Green's function | `orchid.electrostatics` | NaCl Madelung constant 1.747565 (1e-9) |
| R13 | done | Full 3D device KMC: electrodes, injection/extraction, image charge, Coulomb, excitons, multilayer stacks | `orchid.device_kmc` | zero-bias equilibrium (J=0); vs 1D DD (van der Holst 2009); 3-layer OLED |
| R14 | done | Transient DD: CELIV, TPV, TPC, arbitrary V(t)/G(t) | `orchid.transient` | Juska CELIV formula; TPV small perturbation |
| R15 | done | AMG-preconditioned ME and finite-size extrapolation | `orchid.master_equation` | 10-20x fewer CG iterations |

## v0.3: device physics completeness (planned)

| ID | Status | Item | Module | Validation |
|---|---|---|---|---|
| R16 | planned | Exciton dynamics inside device KMC: TTA, TPQ, SPA, Förster to guests, phosphorescent emitters | `orchid.device_kmc` | Mesta et al. Nat. Mater. 12, 652 (2013) roll-off |
| R17 | planned | Multilayer heterojunction DD with Gaussian-DOS (Paasch-Scheinert) contact statistics | `orchid.drift_diffusion` | van Mensfoort 2010 multilayer OLED |
| R18 | planned | Small-signal impedance spectroscopy and IMPS/IMVS | `orchid.transient` | analytic RC / transit-time spectra |
| R19 | planned | 2D/3D drift-diffusion for OFETs and bulk-heterojunction morphologies | `orchid.dd2d` | Sesame 2D benchmarks |
| R20 | planned | Dipole-emitter microcavity outcoupling (Chance-Prock-Silbey), ASTM G173 spectrum, n,k database | `orchid.optics` | published OLED outcoupling spectra |

## v0.4: first principles and machine learning (planned)

| ID | Status | Item | Module | Validation |
|---|---|---|---|---|
| R21 | planned | MD morphologies + DFT transfer integrals/site energies (VOTCA-like) for parameter-free mobilities | `orchid.morphology` | VOTCA-XTP benchmark systems |
| R22 | planned | ML surrogates of mu(T,p,F) and J-V trained on ORCHID data; Bayesian inversion | `orchid.ml` | held-out ME/KMC data |
| R23 | planned | GPU interacting and device KMC (parallel-in-space sublattice KMC) | `orchid.gpu` | bitwise statistics vs CPU engine |

## Parallel strategy

| Level | Mechanism | Scales to | Used by |
|---|---|---|---|
| GPU | CUDA kernel, one walker per thread; CuPy sparse CG | 10^5 to 10^6 walkers per GPU | `orchid.gpu` |
| Threads | Numba `prange` (no GIL) | all cores of a node | KMC, exciton KMC |
| Processes | `ProcessPoolExecutor` | one node | `parallel_map`, `search` |
| MPI | mpi4py block-cyclic task farm | thousands of ranks | disorder ensembles, T/F/p sweeps, J-V sets, search grids |

Launch on a cluster: `export NUMBA_NUM_THREADS=t; mpiexec -n R python study.py`. Any `parallel_map` call then spreads its tasks over the R ranks.
