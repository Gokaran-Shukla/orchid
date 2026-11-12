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
"""
__version__ = "0.1.0"
__all__ = ["rates", "lattice", "kmc", "master_equation", "models",
           "drift_diffusion", "exciton", "optics", "devices", "search"]
