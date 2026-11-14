"""Source of orchid/data/quizzes.json (edit here, then run this script)."""
import json
import os

Q = {}


def sec(k, title, nb, qs):
    Q[k] = dict(title=title, notebook=nb,
                questions=[dict(q=q, options=o, answer=a, why=w) for q, o, a, w in qs])


sec("01", "Hopping rates and energetic disorder", "01_rates_and_disorder.ipynb", [
    ("A Miller-Abrahams hop goes uphill by dE > 0 over distance r. Its rate is",
     ["nu0 exp(-2 alpha r)", "nu0 exp(-2 alpha r) exp(-dE/kT)",
      "nu0 exp(-2 alpha r) exp(-dE/2kT)", "nu0 exp(-(dE + lambda)^2 / 4 lambda kT)"], "b",
     "MA rates are asymmetric: downhill hops pay only the tunnelling factor exp(-2 alpha r), uphill "
     "hops also pay the full Boltzmann factor exp(-dE/kT). The exp(-dE/2kT) form is the symmetric "
     "variant and the last option is Marcus theory."),
    ("Which relation must every rate model in orchid.rates satisfy so that equilibrium is the Fermi-Dirac/Boltzmann state?",
     ["W_ij = W_ji", "W_ij / W_ji = exp(-(E_j - E_i)/kT)", "sum_j W_ij = nu0", "W_ij proportional to r^-6"], "b",
     "Detailed balance, W_ij/W_ji = exp(-(E_j-E_i)/kT), guarantees that the equilibrium occupation is "
     "Boltzmann (or Fermi-Dirac with exclusion). V11 checks it to 7e-15 for MA, Marcus and MLJ."),
    ("In Marcus theory, what happens to the rate as the driving force -dE grows beyond the reorganisation energy lambda?",
     ["It keeps increasing", "It saturates at 2 pi J^2 / hbar", "It decreases (inverted region)", "It becomes temperature independent"], "c",
     "The Marcus rate is proportional to exp(-(dE+lambda)^2/(4 lambda kT)). It peaks at -dE = lambda and "
     "falls again beyond that: this is the inverted region. Miller-Abrahams has no such regime."),
    ("The dimensionless disorder sigma-hat used throughout ORCHID is defined as",
     ["sigma / kT", "sigma / eFa", "kT / sigma", "sigma^2 / kT"], "a",
     "sigma-hat = sigma/kT. Most GDM/EGDM results (for example ln mu ~ -c sigma-hat^2) are written in terms of it."),
    ("How do the Förster and Dexter exciton transfer rates fall off with distance r?",
     ["Both as r^-6", "Förster as exp(-2r/L), Dexter as r^-6", "Förster as r^-6, Dexter as exp(-2r/L)", "Both exponentially"], "c",
     "Förster transfer is dipole-dipole coupling, so k = (1/tau)(R0/r)^6. Dexter transfer is an "
     "electron-exchange process that needs orbital overlap, so it decays exponentially with distance."),
])
sec("02", "Kinetic Monte Carlo: mobility, TOF and relaxation", "02_kinetic_monte_carlo.ipynb", [
    ("In the BKL/Gillespie algorithm, the time step after an event is drawn as",
     ["dt = 1/R_tot", "dt = -ln(u)/R_tot, u uniform in (0,1]", "dt = u/R_tot", "dt = -ln(u)/W_chosen"], "b",
     "The waiting time of a Poisson process with total rate R_tot is exponentially distributed. Sampling "
     "it by inversion gives dt = -ln(u)/R_tot. The event itself is chosen with probability W_i/R_tot."),
    ("What does the Fenwick tree in orchid._fenwick buy for interacting KMC with n events?",
     ["O(1) selection, O(n) update", "O(log n) selection and O(log n) update", "O(n) selection, O(1) update", "Exact parallelism"], "b",
     "A binary indexed tree keeps prefix sums, so event selection and the local rate updates both cost "
     "O(log n). V16/R11 measure a 27-135x speed-up over the dense O(n) engine."),
    ("At low density and long times, the mean energy of carriers in a Gaussian DOS of width sigma relaxes to",
     ["0", "-sigma", "-sigma^2/kT", "-sigma^2/(2kT)"], "c",
     "The occupied DOS is the Gaussian multiplied by the Boltzmann factor, which is again a Gaussian, "
     "centred at -sigma^2/kT (Bässler 1993). V4 recovers this to 1.7 % at sigma-hat = 2."),
    ("A time-of-flight transient becomes dispersive when disorder increases. What do you see?",
     ["A flat plateau and a sharp drop", "The plateau tilts and the post-transit tail broadens", "The current increases after t_tr", "No transit time can be defined"], "b",
     "Dispersive (Scher-Montroll-like) transport shows a current that already decays before t_tr and a "
     "long tail after it. The transit time is then taken from the intersection of the log-log asymptotes (V10)."),
    ("How is the drift mobility obtained from an independent-carrier KMC run?",
     ["mu = D e/kT", "mu = <x(t)>/(F t), the mean drift displacement per unit field and time", "mu = nu0 a^2/kT", "mu = 1/(n e rho)"], "b",
     "The carriers are driven by a field F along x. The mean drift velocity is <x>/t, and mu = v/F. "
     "Reduced fields eFa much smaller than sigma keep the run in the linear-response regime."),
])
sec("03", "Master equation and the EGDM", "03_master_equation_egdm.ipynb", [
    ("In linear response, the master equation maps onto which problem?",
     ["A random walk with absorbing boundaries", "A resistor network with conductances G_ij proportional to W_ij f_i (1 - f_j)", "A Poisson equation", "An eigenvalue problem"], "b",
     "Linearising the steady-state ME around equilibrium gives Kirchhoff's laws on a network with "
     "conductances (e^2/kT) W_ij f_i (1-f_j). ORCHID solves it with CG, preconditioned by Jacobi or AMG."),
    ("What causes the density enhancement g1(p) of the EGDM?",
     ["Coulomb repulsion", "Filling of deep tail states, which raises the effective transport level", "Higher temperature", "Field-assisted hopping"], "b",
     "Extra carriers first fill the deepest states. The remaining carriers sit higher in the DOS and need "
     "less activation, so mu rises steeply once p is above about 1e-4 per site (V2)."),
    ("In Pasveer's parametrisation mu0(T) = mu_00 exp(-c2 sigma-hat^2), c2 is approximately",
     ["0.42", "4/9 = 0.44 exactly", "1", "0.11"], "a",
     "Pasveer et al. (PRL 2005) fitted c1 = 1.8e-9 and c2 = 0.42 for the 3D lattice with alpha = 10/a. "
     "ORCHID's own fit gives c2 = 0.36 (V1). The difference comes from the finite lattice and the hopping range."),
    ("Why is the ME much cheaper than KMC for the zero-field mobility?",
     ["It ignores disorder", "It solves one sparse linear system and has no stochastic noise", "It runs on the GPU only", "It uses a smaller lattice"], "b",
     "One CG solve gives the exact ensemble mean for that disorder realisation, whereas KMC must average "
     "very long trajectories. V1 shows that the two agree to 3 % where both apply."),
    ("For a Gaussian DOS at high density, the generalised Einstein ratio g3 = eD/(mu kT) is",
     ["exactly 1", "less than 1", "greater than 1", "zero"], "c",
     "g3 = (n/kT)(dE_F/dn). For a degenerate (state-filling) Gaussian DOS this exceeds 1, and the "
     "drift-diffusion code must include it (R05; see van Mensfoort and Coehoorn)."),
])
sec("04", "Drift-diffusion: diodes, solar cells and OLEDs", "04_drift_diffusion.ipynb", [
    ("Why is the Scharfetter-Gummel discretisation used for the current between two nodes?",
     ["It is second-order accurate in h", "It uses the exact exponential solution of J between nodes, so it stays stable for large potential steps", "It avoids solving Poisson's equation", "It conserves energy"], "b",
     "SG integrates J = q mu n F + q D n' exactly for a constant field between the nodes. The result "
     "contains Bernoulli functions B(dpsi/kT) and avoids the oscillations of central differences."),
    ("The trap-free space-charge-limited current (Mott-Gurney) is",
     ["J = 9/8 eps mu V^2 / L^3", "J = eps mu V / L^2", "J = 9/8 eps mu V / L^3", "J = q n mu V / L"], "a",
     "Mott-Gurney: J = 9 eps mu V^2/(8 L^3). V5 approaches it (ratio 1.06 at 40 V) once diffusion and the "
     "built-in voltage become negligible."),
    ("Koster et al. (2005) predict that Voc depends on the light intensity (generation G) as",
     ["dVoc/dln G = 2kT/e", "dVoc/dln G = kT/e for bimolecular (Langevin) recombination", "Voc is independent of G", "dVoc/dln G = Eg/e"], "b",
     "With bimolecular recombination, Voc = Eg/e - (kT/e) ln((1-P)gamma Nc^2/(P G)). The slope is kT/e. "
     "ORCHID reproduces it with a slope of 1.004 kT/e and deviations of at most 1.3 mV (V6)."),
    ("The Langevin recombination coefficient is",
     ["gamma = q (mu_n + mu_p) / eps", "gamma = q mu_n mu_p / eps", "gamma = eps / (q mu)", "gamma = kT mu / q"], "a",
     "Langevin: gamma_L = q(mu_n+mu_p)/(eps0 eps_r). ``langevin_prefactor`` scales it for reduced recombination."),
    ("For the metal-insulator-metal picture used in Device, with injection barriers phi_a and phi_c, the built-in voltage is",
     ["Eg/e", "(Eg - phi_a - phi_c)/e", "(phi_a + phi_c)/e", "0"], "b",
     "The difference between the electrode work functions equals the gap minus the two injection "
     "barriers. Below Vbi the dark current is diffusion-limited, above it the current becomes drift/SCLC-limited."),
])
sec("05", "Transients and small-signal analysis", "05_transients_smallsignal.ipynb", [
    ("Photo-CELIV extracts the mobility (Juska et al. 2000, small-charge limit) from",
     ["mu = 2 d^2 / (3 A t_max^2)", "mu = d^2 / (V t_tr)", "mu = A t_max / d", "mu = j0 / dj"], "a",
     "With a ramp of slope A, the extraction peak t_max gives mu = 2d^2/(3A t_max^2) (plus a correction "
     "for dj/j0). V17 shows that the extracted mu converges to the input mu once qA t_max >> kT."),
    ("The low-frequency capacitance of a trap-free single-carrier SCLC diode is",
     ["C_geo", "3/4 C_geo", "1/2 C_geo", "2 C_geo"], "b",
     "Space charge makes the small-signal response lag the transit, so C(0) = 3/4 eps/L (Martens). "
     "V19 obtains 0.76-0.77 and a peak at f tau_dc = 0.72."),
    ("What does a small-perturbation transient photovoltage (TPV) measure?",
     ["The transit time", "The lifetime of the excess carriers at open circuit", "The series resistance", "The exciton diffusion length"], "b",
     "A weak light pulse at Voc raises the carrier density a little. The decay of Voc gives the "
     "small-signal recombination lifetime. In thin devices this differs from the bulk 1/(2 gamma n) because of electrode charge (V17)."),
    ("IMPS modulates ____ and measures ____.",
     ["voltage; current", "light intensity; photocurrent at short circuit", "light; photovoltage at open circuit", "temperature; current"], "b",
     "IMPS is the photocurrent transfer function at short circuit. IMVS is the photovoltage version at open circuit."),
    ("The dc transit time of a trap-free SCLC diode is",
     ["L^2 / (mu V)", "4/3 L^2/(mu V)", "3/4 L^2 / (mu V)", "L / (mu V)"], "b",
     "The field grows across an SCLC device, and integrating dx/(mu F) gives tau_dc = (4/3) L^2/(mu V). "
     "This is the time scale for the admittance analysis."),
])
sec("06", "2D drift-diffusion and organic field-effect transistors", "06_dd2d_ofet.ipynb", [
    ("In the gradual-channel model, the linear-regime drain current per unit width is",
     ["mu C_i (V_G - V_T)^2 / (2L)", "mu C_i (V_G - V_T) V_D / L", "mu C_i V_D^2 / L", "q n mu V_D / L"], "b",
     "For |V_D| << |V_G - V_T| the channel is a uniform sheet of charge C_i(V_G-V_T), so "
     "I/W = mu C_i (V_G-V_T) V_D / L."),
    ("In saturation, the gradual-channel current is",
     ["mu C_i (V_G - V_T)^2 / (2L)", "mu C_i (V_G - V_T) V_D / L", "Independent of V_G", "Proportional to V_D^2"], "a",
     "Beyond pinch-off, I_sat/W = mu C_i (V_G-V_T)^2/(2L). The saturation mobility is extracted from "
     "the slope of sqrt(I_D) against V_G."),
    ("Why does an OFET need a 2D solver?",
     ["Because transport is 3D", "The field has strong lateral and vertical components, and the accumulation layer is a few nm thick", "1D codes cannot handle holes", "To include optics"], "b",
     "The gate field confines charge vertically while the drain field drives it laterally. The contacts "
     "are 2D boundaries. Dd2d solves the coupled Poisson/continuity equations on an (x, y) grid."),
    ("What does the 15-colour Jacobian in dd2d do?",
     ["It uses 15 Newton steps", "It perturbs structurally independent unknowns together, so building the finite-difference Jacobian costs 15 residual evaluations instead of N", "It uses 15 threads", "It solves 15 voltages at once"], "b",
     "Unknowns whose stencils do not overlap can be perturbed at the same time. Graph colouring reduces "
     "the cost of the sparse Jacobian to one residual evaluation per colour."),
    ("V23 compares the 2D DD OFET with",
     ["NIST Sesame", "The gradual-channel model with the input mobility", "Experimental pentacene data", "The 3D device KMC"], "b",
     "The mobilities extracted from the simulated transfer curves reproduce the input mobility: 0.97x in "
     "the linear regime and 1.06x in saturation."),
])
sec("07", "Optics: transfer matrices, generation and outcoupling", "07_optics.ipynb", [
    ("The transfer-matrix method assumes",
     ["Incoherent light", "Coherent plane waves in planar layers, with interface and propagation matrices", "Geometric optics", "Scattering by particles"], "b",
     "Each interface (Fresnel coefficients) and each layer (phase exp(i k d)) is a 2x2 matrix. The "
     "product gives the field everywhere. ORCHID agrees with the independent tmm package to 1e-15 (V15)."),
    ("The local absorption that sets the exciton generation rate is proportional to",
     ["|E|", "n k |E|^2 (i.e. Im(eps) |E|^2)", "R + T", "n^2"], "b",
     "Absorbed power density is (1/2) omega eps0 Im(eps) |E|^2 = omega eps0 n k |E|^2. Dividing by the "
     "photon energy and integrating over the AM1.5G spectrum gives G(x)."),
    ("For any lossless check of an optics solver, which identity must hold?",
     ["R = T", "R + T + A = 1", "A = 0 always", "R + T = 2"], "b",
     "Energy conservation. V8 checks the Airy formula to 1e-16 and R+T+A=1 to 1e-5 for absorbing stacks."),
    ("The classical ray-optics estimate of OLED outcoupling into air is",
     ["1/(2 n^2)", "1/n", "n^2", "0.5"], "a",
     "With isotropic emission inside a medium of index n, only the escape cone sin(theta) < 1/n gets "
     "out: eta ~ 1/(2n^2), about 20 % for n = 1.7. The dipole model (R20) refines this."),
    ("Near a metal mirror, a dipole's decay rate oscillates with distance and is strongly enhanced at very short distances. Why?",
     ["Thermal effects", "Interference with its reflected field (CPS); near-field coupling to surface plasmons and lossy waves", "Exciton diffusion", "Spin statistics"], "b",
     "The Chance-Prock-Silbey theory gives the oscillation from interference and the short-distance "
     "quenching from evanescent coupling into the metal. ORCHID reproduces the perfect-mirror formulas to 8e-4 (V20)."),
])
sec("08", "Excitons: diffusion, annihilation and quenching", "08_excitons.ipynb", [
    ("In 3D, the exciton diffusion coefficient from KMC rates k_j to neighbours at r_j is",
     ["D = sum k_j r_j^2", "D = sum k_j r_j^2 / 6", "D = sum k_j r_j / 3", "D = 1 / (6 tau)"], "b",
     "<r^2> = 6 D t in 3D, and each hop adds r_j^2 at rate k_j. V7 confirms this within statistical error."),
    ("The exciton diffusion length is",
     ["L_D = D tau", "L_D = sqrt(D tau)", "L_D = sqrt(6 D tau)", "L_D = D / tau"], "b",
     "L_D = sqrt(D tau) by convention, which is the 1D RMS displacement. Some papers quote sqrt(6 D tau) "
     "for the 3D displacement, so always check the definition."),
    ("At high exciton density, triplet-triplet annihilation adds which loss term to the continuity equation?",
     ["-k n", "-gamma_TT n^2", "-gamma n p", "-n / tau"], "b",
     "TTA is bimolecular. V18 checks that the KMC loss rate divided by n^2 is constant (5 % spread), so "
     "the continuum gamma_TT is well defined."),
    ("For a purely fluorescent emitter, spin statistics limit the singlet fraction to",
     ["100 %", "75 %", "25 %", "50 %"], "c",
     "Electron-hole recombination produces singlets and triplets in a 1:3 ratio. Phosphorescent or TADF "
     "emitters can harvest the triplets as well (eta_ST = 1)."),
    ("Why does EQE roll off at high current in OLEDs?",
     ["Outcoupling drops", "Bimolecular quenching (TTA, exciton-polaron quenching TPQ) grows faster than the generation", "Charge balance always improves", "The emission wavelength shifts"], "b",
     "Loss channels proportional to n^2 (TTA) or n*p (TPQ) grow faster than the linear radiative "
     "channel, so the quantum efficiency falls as J rises."),
])
sec("09", "3D device KMC with concurrent excitons", "09_device_kmc_coupled.ipynb", [
    ("How does the device KMC compute the current with low noise?",
     ["By counting carriers that reach the electrodes", "Ramo-Shockley: every hop contributes e q dz/(L A)", "From the mean-field potential", "From the Einstein relation"], "b",
     "Every displacement induces charge on the electrodes, so all hops contribute. This converges much "
     "faster than counting only the extracted carriers."),
    ("The image charge in the electrode makes a carrier near the contact",
     ["Higher in energy", "Lower in energy, which lowers the injection barrier", "Unchanged", "Leave the device"], "b",
     "The image attraction -e^2/(16 pi eps z) lowers the barrier near the electrode. This is Schottky "
     "barrier lowering, and it strongly affects injection-limited currents."),
    ("Why are electrostatics split into a mean-field sheet part and an exact short-range part?",
     ["The sheet part is more accurate", "Exact long-range Coulomb sums are too expensive. Sheets give the long-range part cheaply, and close pairs are treated exactly.", "To remove image charges", "To use the GPU"], "b",
     "The 1D sheet Green's function updates in O(Lz) per event. The exact screened Coulomb within r_coul "
     "minus its sheet part makes the sum converge to full Coulomb as r_coul grows."),
    ("What changed for exciton-polaron quenching in v0.4 (CoupledDeviceKMC)?",
     ["TPQ was removed", "TPQ is evaluated against the live polaron positions in the same event loop, not against a frozen configuration", "TPQ became a continuum term", "Polarons became immobile"], "b",
     "In v0.3, excitons saw a polaron snapshot drawn from the mean density profile. Now charges and "
     "excitons share one clock and two Fenwick trees, so all correlations are kept (R24)."),
    ("Which exact identity does V24 check in every coupled run?",
     ["J = 0 at V = 0", "formed + alive_start = radiative + non-radiative + TPQ + TTA + overflow + alive_end", "eta_rad = 1", "gamma_b = 1"], "b",
     "Every exciton that exists during the measurement must end in exactly one channel or still be "
     "alive at the end. V24 finds an imbalance of exactly 0 in all runs."),
])
sec("10", "Morphology, DFT couplings and MD trajectories", "10_morphology_md.ipynb", [
    ("For a symmetric dimer, DIPRO's effective transfer integral equals",
     ["The full dimer splitting", "Half the dimer orbital splitting", "The HOMO energy", "Zero"], "b",
     "For identical monomers with zero site-energy difference, the bonding/antibonding splitting is 2|J|. "
     "V21 checks this to 0.24 % with HF/STO-3G in PySCF."),
    ("Why must molecules be made whole across periodic boundaries before computing centres of mass?",
     ["To save memory", "A molecule split by the boundary would otherwise get its centre somewhere in the middle of the box", "GROMACS requires it", "It changes the energies"], "b",
     "molecule_centres unwraps every atom by the minimum image relative to the first atom of its molecule. "
     "V26 places 397 boundary-straddling molecules and recovers the centres exactly."),
    ("Dipolar disorder produces site energies that are",
     ["Uncorrelated", "Spatially correlated, which extends the sqrt(F) Poole-Frenkel-like regime", "Always zero on average with zero width", "Temperature dependent"], "b",
     "Each site energy is a sum over many nearby dipoles, so neighbouring sites share most of the sum. "
     "This is the correlated disorder model (CDM, Novikov)."),
    ("The transfer integral between molecules typically decays with distance as",
     ["r^-6", "exp(-beta r)", "r^-1", "constant"], "b",
     "Orbital overlap decays exponentially. V21 extracts beta = 2.9 per Angstrom for cofacial ethylene."),
    ("What does V26 prove about the graph solver?",
     ["Graphs are faster", "A lattice-shaped 'trajectory' imported through molecule_centres gives the same ME mobility as the Lattice solver to 3e-14", "MD is always needed", "Mobility does not depend on morphology"], "b",
     "This exact equivalence shows that the import -> MolecularGraph -> master equation pipeline adds no "
     "error, so differences on real morphologies are physical."),
])
sec("11", "Exhaustive search, ML surrogates and Bayesian inversion", "11_search_ml_bayes.ipynb", [
    ("A full grid with n points per parameter in d dimensions needs",
     ["n d evaluations", "n^d evaluations", "d^n evaluations", "log n evaluations"], "b",
     "This exponential cost is why zoom_search refines the grid around the best point level by level, "
     "and why surrogates help for expensive models."),
    ("Why does ORCHID fit mobility data with log-RMSE?",
     ["It is faster", "Mobility spans many decades, so relative (log) errors weight all points fairly", "To avoid negative mobilities only", "Because data are linear"], "b",
     "On a linear scale the largest-mobility points would dominate the fit. The log metric treats a "
     "factor of 2 equally everywhere."),
    ("A Gaussian-process surrogate provides",
     ["Only a point prediction", "A predictive mean and an uncertainty; hyperparameters set by maximising the marginal likelihood", "An exact solution", "A neural network"], "b",
     "The GP posterior gives mean and variance. GPSurrogate.fit grid-searches length scale, signal and "
     "noise using the log marginal likelihood."),
    ("V22 reports '3/3 covered'. What does that mean?",
     ["Three CPUs were used", "The true parameters lie inside the 95 % credible intervals for all three parameters", "Three folds of cross-validation", "Three seeds agree"], "b",
     "This is a calibration check of the Bayesian inversion: the synthetic ground truth must fall inside "
     "the reported intervals."),
    ("Latin hypercube sampling ensures that",
     ["All points lie on a grid", "Each parameter's range is stratified, one sample per stratum", "Samples are correlated", "Only corners are sampled"], "b",
     "LHS spreads n samples so that every one-dimensional projection is evenly covered. It is a good "
     "design for training surrogates."),
])
sec("12", "Massive parallelism: MPI, threads, GPU and sublattice KMC", "12_parallel_gpu.ipynb", [
    ("Why is independent-carrier KMC ideal for GPUs?",
     ["It needs double precision", "Walkers do not interact, so each GPU thread can propagate one walker", "It uses the Fenwick tree", "Because of Coulomb interactions"], "b",
     "Embarrassingly parallel. orchid.gpu.kmc_independent_gpu launches one CUDA thread per walker."),
    ("In the synchronous-sublattice algorithm, blocks of the same sublattice can be updated in parallel because",
     ["They are random", "They are separated by at least the interaction range, so their events are independent within a window", "They use different seeds", "They contain no carriers"], "b",
     "Blocks of side B >= 2 r_c in an 8-colour checkerboard never share hops or neighbours within one window "
     "(Shim and Amar 2005)."),
    ("How does v0.4 remove the time-window bias of sublattice KMC?",
     ["By using one block", "By running several windows tau and extrapolating mu(tau) to tau -> 0", "By using the GPU", "By increasing the lattice size"], "b",
     "mobility_sublattice_extrapolated fits mu(tau) with a polynomial in tau. V25 reduces the error from "
     "about 5 % (single window) to about 1 %."),
    ("ORCHID's MPI task farm reproduces the serial result bit-for-bit because",
     ["MPI is deterministic", "Random seeds are attached to tasks, not to ranks", "Only rank 0 computes", "Results are averaged"], "b",
     "Each task carries its own seed, so the decomposition over ranks cannot change any random stream. V16 checks this."),
    ("Amdahl's law: if a fraction s of the runtime is serial, the speed-up on P cores is at most",
     ["P", "1 / (s + (1 - s)/P)", "P s", "s P / (1 - s)"], "b",
     "The serial part does not shrink. This is why ORCHID parallelises the outermost loop (ensembles, "
     "sweeps) as well as the inner kernels."),
])
sec("13", "Benchmarking and validation", "13_benchmarking.ipynb", [
    ("Why does V15 compare against NIST Sesame on identical grids?",
     ["To compare speed", "Identical inputs isolate the equations and discretisation, so agreement to 1e-6 shows both codes solve the same problem", "Sesame is faster", "To test optics"], "b",
     "With the same mesh and parameters, any remaining difference comes from the model or the "
     "discretisation, not from resolution."),
    ("When a benchmark fails, what does validation/criteria.py say to do?",
     ["Loosen the tolerance", "Fix the code; tolerances are set in advance", "Delete the benchmark", "Change the reference"], "b",
     "Tolerances come from the published references and are fixed before running. Loosening them after "
     "the fact would make the benchmark meaningless."),
    ("How should a stochastic KMC result be compared to an exact value?",
     ["It must be exactly equal", "By a z-score: |x - x_exact| / sigma_stat <= 3", "By visual inspection", "Only the sign matters"], "b",
     "KMC estimates have statistical error. V7 and V24 use z-scores: 1.4 sigma for D and well within 3 sigma for eta_rad."),
    ("V27 records events per second and solve times. What is it for?",
     ["Comparing with other codes' papers", "Detecting performance regressions on the same machine across versions", "Pass/fail validation", "Choosing the GPU"], "b",
     "Throughput depends on the hardware, so V27 is recorded rather than graded, together with CPU "
     "count and library versions."),
    ("Which command re-runs every benchmark and rewrites BENCHMARKS.md?",
     ["pytest", "python validation/run_all.py --run", "orchid roadmap", "make bench"], "b",
     "Without --run it only re-evaluates existing results. With IDs (for example --run V24 V25) it re-runs only those benchmarks."),
])

# vary the position of the correct option so the answer key is not all 'b'
import random  # noqa: E402
rng = random.Random(2026)
for k, s in Q.items():
    for q in s["questions"]:
        correct = q["options"]["abcd".index(q["answer"])]
        opts = q["options"][:]
        rng.shuffle(opts)
        q["options"] = opts
        q["answer"] = "abcd"[opts.index(correct)]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "orchid", "data", "quizzes.json")
with open(out, "w", encoding="utf-8") as f:
    json.dump(Q, f, indent=1, ensure_ascii=False)
print("wrote", out, sum(len(s["questions"]) for s in Q.values()), "questions")
