"""Machine-readable ORCHID development roadmap (``orchid roadmap``).

Each item has: id, title, version, status ("done" | "in progress" |
"planned"), module, validation target. ROADMAP.md is the prose version;
tests check that every "done" item names an importable module.
"""

ROADMAP = [
    # ---------------------------------------------------------------- v0.1
    dict(id="R01", version="0.1", status="done", module="orchid.rates",
         title="Hopping rates: MA, symmetric MA, Marcus, MLJ, Förster, Dexter",
         validation="detailed balance, Marcus inverted region"),
    dict(id="R02", version="0.1", status="done", module="orchid.lattice",
         title="GDM / dipolar CDM / exponential / host-guest disorder; positional and off-diagonal",
         validation="disorder statistics, spatial correlation"),
    dict(id="R03", version="0.1", status="done", module="orchid.kmc",
         title="Bulk KMC: independent carriers, TOF, energy relaxation",
         validation="Pasveer 2005, Bässler 1993 -sigma^2/kT"),
    dict(id="R04", version="0.1", status="done", module="orchid.master_equation",
         title="Master equation: linear-response resistor network + nonlinear",
         validation="ME = KMC; EGDM g1"),
    dict(id="R05", version="0.1", status="done", module="orchid.drift_diffusion",
         title="1D coupled-Newton drift-diffusion, EGDM + g3, Langevin/SRH/Braun",
         validation="Mott-Gurney, Koster 2005 Voc, NIST Sesame"),
    dict(id="R06", version="0.1", status="done", module="orchid.optics",
         title="Transfer-matrix optics and generation profiles",
         validation="Airy formula, tmm (Byrnes) package"),
    dict(id="R07", version="0.1", status="done", module="orchid.exciton",
         title="Exciton KMC (Förster/Dexter) and emission-zone continuum with annihilation",
         validation="D = sum k r^2 / 6"),
    dict(id="R08", version="0.1", status="done", module="orchid.search",
         title="Exhaustive grid / zoom / Latin-hypercube search",
         validation="parameter recovery"),
    # ---------------------------------------------------------------- v0.2
    dict(id="R09", version="0.2", status="done", module="orchid.parallel",
         title="Hybrid parallelism: MPI task farm x processes x Numba threads",
         validation="MPI result identical to serial; scaling benchmark"),
    dict(id="R10", version="0.2", status="done", module="orchid.gpu",
         title="CUDA KMC kernel (one walker per thread) and CuPy master-equation CG",
         validation="CUDA-simulator test vs CPU engine"),
    dict(id="R11", version="0.2", status="done", module="orchid.kmc",
         title="O(log N) interacting KMC: Fenwick tree + incremental local updates",
         validation="identical to dense reference engine; 27-135x faster"),
    dict(id="R12", version="0.2", status="done", module="orchid.electrostatics",
         title="Ewald summation, periodic lattice Green's function, image charges, sheet Green's function",
         validation="NaCl Madelung constant 1.747565 (1e-9)"),
    dict(id="R13", version="0.2", status="done", module="orchid.device_kmc",
         title="Full 3D device KMC: electrodes, injection/extraction, image charge, Coulomb, excitons, multilayer stacks",
         validation="zero-bias equilibrium (J=0); vs 1D DD (van der Holst 2009); 3-layer OLED"),
    dict(id="R14", version="0.2", status="done", module="orchid.transient",
         title="Transient DD: CELIV, TPV, TPC, arbitrary V(t)/G(t)",
         validation="Juska CELIV formula; TPV small perturbation"),
    dict(id="R15", version="0.2", status="done", module="orchid.master_equation",
         title="AMG-preconditioned ME and finite-size extrapolation",
         validation="10-20x fewer CG iterations"),
    # ---------------------------------------------------------------- v0.3
    dict(id="R16", version="0.3", status="done", module="orchid.exciton_device",
         title="Exciton KMC in devices: generation from device KMC, Förster hopping, TTA, TPQ, metal quenching -> EQE roll-off",
         validation="eta = kr/(kr+knr) exactly; TTA loss = gamma n^2 (constant gamma)"),
    dict(id="R17", version="0.3", status="done", module="orchid.drift_diffusion",
         title="Multilayer heterostructure DD (band offsets, per-layer materials) with Gaussian-DOS contact statistics",
         validation="D/A heterojunction vs NIST Sesame"),
    dict(id="R18", version="0.3", status="done", module="orchid.smallsignal",
         title="Small-signal impedance / admittance spectroscopy, IMPS, IMVS",
         validation="insulator C=eps/L; SCLC C(0)=3/4 C_geo, f_max tau_dc = 0.72"),
    dict(id="R19", version="0.3", status="done", module="orchid.dd2d",
         title="2D drift-diffusion for OFETs and bulk-heterojunction morphologies (sparse Newton, 15-colour Jacobian)",
         validation="OFET vs gradual-channel model (linear + saturation mobility)"),
    dict(id="R20", version="0.3", status="done", module="orchid.optics",
         title="Dipole-emitter outcoupling (CPS / plane-wave expansion), ASTM G173 spectrum, refractiveindex.info n,k database",
         validation="CPS perfect-mirror formulas; BK7 n_d = 1.5168; AM1.5G = 1000 W/m^2"),
    # ------------------------------------------- v0.3 (continued, shipped in 0.3.0)
    dict(id="R21", version="0.3", status="done", module="orchid.morphology",
         title="Molecular morphologies + DFT transfer integrals (DIPRO/PySCF) + dipolar site energies on hopping graphs",
         validation="graph = lattice ME exactly; DIPRO J_eff = half dimer splitting"),
    dict(id="R22", version="0.3", status="done", module="orchid.ml",
         title="Gaussian-process surrogates of mu(T,p) trained on ORCHID data; exhaustive-grid Bayesian inversion",
         validation="held-out RMSE; truth inside 95 % credible intervals"),
    dict(id="R23", version="0.3", status="done", module="orchid.sublattice_kmc",
         title="Parallel-in-space interacting KMC (synchronous sublattice), CPU threads + CUDA kernel",
         validation="agrees with serial Fenwick KMC; CUDA simulator test"),
    # ---------------------------------------------------------------- v0.4
    dict(id="R24", version="0.4", status="done", module="orchid.coupled_kmc",
         title="Concurrent charge + exciton device KMC: one event loop, live TPQ/TTA, emission profile, EQE",
         validation="exact bookkeeping; eta = kr/(kr+knr) within 3 sigma; J unchanged; vs v0.3 pipeline"),
    dict(id="R25", version="0.4", status="done", module="orchid.sublattice_kmc",
         title="Time-window extrapolation (tau -> 0) for synchronous-sublattice KMC",
         validation="extrapolated mobility vs serial Fenwick KMC"),
    dict(id="R26", version="0.4", status="done", module="orchid.morphology",
         title="MD trajectory import (GROMACS .gro, extended XYZ), PBC-whole molecular centres, frame-averaged mobility",
         validation="exact centres for PBC-split molecules; graph ME = lattice ME (3e-14)"),
    dict(id="R27", version="0.4", status="done", module="orchid.cli",
         title="Benchmark harness: criteria for V1-V27, run_all, BENCHMARKS.md, throughput record, orchid bench",
         validation="27 benchmarks evaluated against fixed tolerances"),
    dict(id="R28", version="0.4", status="done", module="orchid.tutorials",
         title="13 executed tutorial notebooks with exercises and 65-question quiz bank (orchid quiz)",
         validation="notebooks execute in CI-equivalent run; quiz answer keys tested"),
]


def summary(version=None):
    rows = [r for r in ROADMAP if version is None or r["version"] == version]
    w = max(len(r["title"]) for r in rows)
    lines = []
    for r in rows:
        mark = {"done": "[x]", "in progress": "[~]", "planned": "[ ]"}[r["status"]]
        lines.append(f"{mark} {r['id']} v{r['version']}  {r['title']:<{w}}  ({r['module']})")
    return "\n".join(lines)


VERSION_TITLES = {
    "0.1": "multiscale core",
    "0.2": "3D device KMC, transients, massive parallelism",
    "0.3": "device-physics completeness, first principles, ML, parallel-in-space KMC",
    "0.4": "concurrent excitons, unbiased parallel KMC, MD import, benchmarks, tutorials",
}


def markdown(extra=""):
    """Render the roadmap tables (used to regenerate ROADMAP.md)."""
    done, n = progress()
    out = ["# ORCHID roadmap", "",
           "This file is generated from `orchid/roadmap.py`; `orchid roadmap` prints it. Every item has "
           "an importable module, a validation benchmark (`validation/`) and tests.", ""]
    for v, title in VERSION_TITLES.items():
        out += [f"## v{v}: {title}", "", "| ID | Status | Item | Module | Validation |",
                "|---|---|---|---|---|"]
        out += [f"| {r['id']} | {r['status']} | {r['title']} | `{r['module']}` | {r['validation']} |"
                for r in ROADMAP if r["version"] == v]
        out.append("")
    return "\n".join(out) + extra


def progress():
    done = sum(r["status"] == "done" for r in ROADMAP)
    return done, len(ROADMAP)
