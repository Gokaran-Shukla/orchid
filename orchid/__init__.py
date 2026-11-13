"""ORCHID - Organic Rate-based CHarge-hopping & Integrated Device simulator.

Multiscale simulation of charge and exciton transport in organic
semiconductors and devices (OLEDs, organic solar cells, single-carrier
diodes):

    rates            Miller-Abrahams, Marcus, Marcus-Levich-Jortner, Förster, Dexter
    lattice          cubic hopping lattices; Gaussian / dipolar-correlated /
                     exponential / host-guest disorder; positional & off-diagonal disorder
    kmc              kinetic Monte Carlo: independent & interacting carriers
                     (exclusion + Coulomb), time-of-flight, energetic relaxation
    master_equation  linear-response resistor-network ME and nonlinear ME
    models           GDM, CDM, EGDM, Poole-Frenkel, Vissenberg-Matters, Gaussian statistics
    drift_diffusion  coupled-Newton 1D device solver (SG, generalised Einstein,
                     Langevin / SRH / Braun-Onsager)
    exciton          exciton KMC and continuum emission-zone model with annihilation
    optics           transfer-matrix optics, generation profiles, outcoupling
    devices          OLED and OPV workflows (EQE, roll-off, J-V, PCE)
    search           exhaustive grid / zoom search, Latin hypercube, log-RMSE
    device_kmc       full 3D device KMC (electrodes, injection, images, Coulomb,
                     excitons, multilayer OLED stacks)
    electrostatics   Ewald sums, periodic lattice Green's function, image charges
    transient        transient DD: CELIV, TPV, TPC, arbitrary V(t), G(t)
    parallel         MPI x processes x Numba-thread hybrid parallel layer
    gpu              CUDA KMC kernel, CuPy master-equation solver
    roadmap          machine-readable development roadmap
    exciton_device   exciton KMC coupled to the device KMC (TTA, TPQ, roll-off)
    smallsignal      impedance / admittance, IMPS, IMVS
    dd2d             2D drift-diffusion (OFETs, bulk heterojunctions)
    materials        refractiveindex.info n,k database, ASTM G173 spectrum
    morphology       molecular packings, DIPRO couplings (PySCF), hopping graphs
    ml               Gaussian-process surrogates, Bayesian inversion
    sublattice_kmc   parallel-in-space interacting KMC (CPU / CUDA)
"""
__version__ = "0.3.0"
__all__ = ["rates", "lattice", "kmc", "master_equation", "models",
           "drift_diffusion", "exciton", "optics", "devices", "search", "device_kmc",
           "electrostatics", "transient", "parallel", "gpu", "roadmap", "exciton_device",
           "smallsignal", "dd2d", "materials", "morphology", "ml", "sublattice_kmc"]
