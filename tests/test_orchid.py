"""Fast regression / physics tests for ORCHID (run: pytest -q)."""
import numpy as np
import pytest

from orchid import rates, models
from orchid.constants import kB_eV, q, eps0
from orchid.lattice import Lattice, neighbour_offsets
from orchid.master_equation import me_linear_response, me_nonlinear, reverse_index
from orchid.kmc import mobility_independent
from orchid.drift_diffusion import (Device, bernoulli, mott_gurney, koster_voc,
                                    pv_metrics, braun_dissociation)
from orchid.optics import tmm_field
from orchid.exciton import exciton_kmc, steady_state_excitons
from orchid.search import grid_search, zoom_search, latin_hypercube, log_rmse

SIG = 0.1


# ----------------------------------------------------------------- rates
@pytest.mark.parametrize("T", [150.0, 300.0])
def test_detailed_balance(T):
    dE = np.linspace(-0.5, 0.5, 51)
    for f in (lambda d: rates.miller_abrahams(d, 1.0, T),
              lambda d: rates.symmetric_ma(d, 1.0, T),
              lambda d: rates.marcus(d, 0.01, T, 0.2)):
        ratio = np.log(f(dE) / f(-dE))
        assert np.allclose(ratio, -dE / (kB_eV * T), atol=1e-6)


def test_mlj_reduces_to_marcus():
    dE = np.linspace(-0.5, 0.5, 21)
    assert np.allclose(rates.marcus_levich_jortner(dE, 0.01, 300, 0.2, 0.17, 0.0),
                       rates.marcus(dE, 0.01, 300, 0.2))


def test_marcus_maximum_at_minus_lambda():
    dE = np.linspace(-1, 0.5, 3001)
    k = rates.marcus(dE, 0.01, 300, 0.3)
    assert abs(dE[np.argmax(k)] + 0.3) < 1e-3


# ----------------------------------------------------------------- lattice
def test_neighbours_and_reverse():
    assert len(neighbour_offsets(np.sqrt(3))) == 26
    lat = Lattice(6, seed=0)
    rev = reverse_index(lat)
    i = np.arange(lat.N)[:, None]
    assert np.all(lat.nbr[lat.nbr, rev] == i)


@pytest.mark.parametrize("model", ["gaussian", "dipolar", "smoothed"])
def test_disorder_width(model):
    lat = Lattice(16, seed=1).set_disorder(SIG, model=model)
    assert abs(lat.energy.std() - SIG) < 1e-9
    assert abs(lat.energy.mean()) < 1e-9


def test_dipolar_is_correlated():
    lat = Lattice(20, seed=2).set_disorder(SIG, model="dipolar")
    e = lat.energy
    corr = np.mean(e * e[lat.nbr[:, 0]]) / SIG ** 2
    lat2 = Lattice(20, seed=2).set_disorder(SIG)
    corr2 = np.mean(lat2.energy * lat2.energy[lat2.nbr[:, 0]]) / SIG ** 2
    assert corr > 0.3 and abs(corr2) < 0.05


# ----------------------------------------------------------------- ME vs KMC
def test_me_kmc_agree():
    T = SIG / (kB_eV * 2.5)
    lat = Lattice(14, a=1.6e-9, seed=3).set_disorder(SIG)
    mu_me = me_linear_response(lat, T, 1e-7, alpha=10.0)[0]
    mu_k, err = mobility_independent(lat, T, 0.03 * SIG, n_carriers=16, n_eq=20000,
                                     n_meas=150000, alpha=10.0)
    assert abs(mu_k / mu_me - 1) < 0.15


def test_me_nonlinear_matches_linear():
    T = SIG / (kB_eV * 2.0)
    lat = Lattice(8, a=1.6e-9, seed=4).set_disorder(SIG)
    ml = me_linear_response(lat, T, 1e-2, alpha=10.0)[0]
    mn = me_nonlinear(lat, T, 1e-2, 0.002, alpha=10.0, tol=1e-10)[0]
    assert abs(mn / ml - 1) < 0.02


def test_ordered_lattice_mobility_einstein():
    # sigma -> 0 : mu = D/(kT/q), D = nu0 e^{-2 alpha} a^2 (6 nn, rc=1)
    T = 300.0
    lat = Lattice(8, a=1e-9, rc=1.0, seed=0).set_disorder(0.0)
    lat.energy[:] = 0.0
    mu = me_linear_response(lat, T, 1e-6, alpha=1.0, nu0=1e12)[0]
    D = 1e12 * np.exp(-2.0) * lat.a ** 2
    assert abs(mu / (D / (kB_eV * T)) - 1) < 1e-5


# ----------------------------------------------------------------- models
def test_egdm_limits():
    T = 300.0
    assert abs(models.egdm_g1(1e-12, T, SIG) - 1) < 1e-2
    assert abs(models.egdm_g2(0.0, T, SIG, 1e-9) - 1) < 1e-12
    assert models.egdm_g1(1e-2, T, SIG) > models.egdm_g1(1e-4, T, SIG)


def test_gaussian_statistics():
    T = 300.0
    EF = models.fermi_level_gaussian(1e-3, T, SIG)
    assert abs(models.gaussian_density(EF, T, SIG) / 1e-3 - 1) < 1e-6
    assert abs(models.einstein_ratio_gaussian(1e-12, T, SIG) - 1) < 1e-2
    assert models.einstein_ratio_gaussian(1e-2, T, SIG) > 1.5


# ----------------------------------------------------------------- DD
def test_bernoulli():
    x = np.array([-50, -1e-8, 0.0, 1e-8, 1.0, 800.0])
    ref = np.array([-50 / np.expm1(-50), 1, 1, 1,
                    1 / np.expm1(1), 0.0])
    assert np.allclose(bernoulli(x), ref, rtol=1e-6, atol=1e-300)


def test_koster_voc():
    G = 1e27
    d = Device(L=100e-9, N=121, Eg=1.2, phi_anode_p=0.1, phi_cathode_n=0.1,
               mu_n=1e-8, mu_p=1e-8, G=G)
    V = np.linspace(-0.2, 0.9, 23)
    m = pv_metrics(V, d.jv(V)["J"])
    vk = koster_voc(1.2, 300, 2.5e26, q * 2e-8 / (eps0 * 3.5), G)
    assert abs(m["Voc"] - vk) < 0.01
    assert abs(m["Jsc"] / (q * G * d.L) - 1) < 0.2


def test_current_continuity_and_mg():
    d = Device(L=100e-9, N=161, Eg=2.4, phi_anode_p=0.0, phi_cathode_n=1.0,
               mu_n=1e-9, mu_p=1e-8, recombination=False)
    r = d.jv(np.array([20.0]))
    pr = r["profiles"][0]
    Jt = pr["Jn"] + pr["Jp"]
    assert np.std(Jt) / np.mean(Jt) < 1e-6
    ratio = r["J"][0] / mott_gurney(20.0 - r["Vbi"], d.L, d.eps_r, d.mu_p)
    assert 1.0 < ratio < 1.2


def test_braun_limits():
    P0 = braun_dissociation(0.0, 300, 1.5e-9, 1e6, 3.5, 2e-8)
    P1 = braun_dissociation(1e8, 300, 1.5e-9, 1e6, 3.5, 2e-8)
    assert 0 < P0 < P1 <= 1


# ----------------------------------------------------------------- optics
def test_tmm_energy_and_airy():
    L = [("ITO", 1.9 + 0.01j, 120e-9), ("blend", 1.8 + 0.3j, 100e-9),
         ("Al", 0.96 + 6.69j, 100e-9)]
    r = tmm_field(L, 550e-9, npts=2001)
    assert abs(r["R"] + r["T"] + r["A"].sum() - 1) < 1e-4
    n, d, lam = 2.0, 100e-9, 500e-9
    rr = tmm_field([("s", n + 0j, d)], lam, n_in=1 + 0j, n_out=1 + 0j)
    r1 = (1 - n) / (1 + n)
    dl = 2 * np.pi * n * d / lam
    Ra = abs((r1 - r1 * np.exp(2j * dl)) / (1 - r1 ** 2 * np.exp(2j * dl))) ** 2
    assert abs(rr["R"] - Ra) < 1e-12


# ----------------------------------------------------------------- excitons
def test_exciton_diffusion_ordered():
    lat = Lattice(20, a=1e-9, rc=2.0, seed=0).set_disorder(0.0)
    r = exciton_kmc(lat, 300, tau=1e-9, R0=1.5, n_excitons=40000)
    assert abs(r["D"] / r["D_ordered"] - 1) < 4 * r["D_err"] / r["D_ordered"] + 0.02


def test_exciton_continuum_no_diffusion():
    x = np.linspace(0, 100e-9, 101)
    R = np.full_like(x, 1e26)
    S, y = steady_state_excitons(x, R, 1e-15, 1e-9, eta_S=1.0, quench_left=False,
                                 quench_right=False)
    assert abs(y - 1) < 1e-3


# ----------------------------------------------------------------- search
def _quad(p):
    return (p["a"] - 0.3) ** 2 + (np.log10(p["b"]) + 5) ** 2


def test_grid_and_zoom_search():
    b, v, tab = grid_search(_quad, {"a": np.linspace(0, 1, 11), "b": np.logspace(-7, -3, 9)})
    assert abs(b["a"] - 0.3) < 1e-12 and abs(np.log10(b["b"]) + 5) < 1e-12
    assert len(tab) == 99
    b2, v2, _ = zoom_search(_quad, {"a": (0, 1), "b": (1e-8, 1e-2)}, n=7, levels=5,
                            log=("b",))
    assert v2 < 1e-4
    pts = latin_hypercube({"a": (0, 1)}, 10)
    assert sorted(int(p["a"] * 10) for p in pts) == list(range(10))
    assert log_rmse([1, 10], [1, 10]) == 0


# ================================================================= v0.2 modules
from orchid.electrostatics import madelung_nacl, image_self_energy, ewald_lattice_green
from orchid.kmc import mobility_interacting
from orchid.device_kmc import DeviceKMC, Layer
from orchid.parallel import parallel_map, ensemble
from orchid._fenwick import fw_build, fw_add, fw_find, fw_total
from orchid import roadmap


def test_madelung_constant():
    assert abs(madelung_nacl() - 1.747564594633) < 1e-7


def test_image_charge_single_plane_limit():
    z = np.array([0.5e-9, 1e-9])
    U = image_self_energy(z, 1e-6, 3.5)
    ref = -q / (16 * np.pi * eps0 * 3.5 * z)
    assert np.allclose(U, ref, rtol=1e-3)


def test_lattice_green_symmetry():
    G = ewald_lattice_green((8, 8, 8))
    assert abs(G[1, 0, 0] - G[0, 1, 0]) < 1e-10 and abs(G[1, 0, 0] - G[7, 0, 0]) < 1e-10
    assert G[1, 0, 0] > G[2, 0, 0] > G[3, 0, 0]


def test_fenwick_tree():
    rng = np.random.default_rng(0)
    v = rng.random(37)
    t = np.zeros(37)
    fw_build(t, v)
    assert abs(fw_total(t) - v.sum()) < 1e-12
    fw_add(t, 5, 2.0)
    v[5] += 2.0
    c = np.cumsum(v)
    for u in rng.random(50) * v.sum():
        assert fw_find(t, u) == np.searchsorted(c, u, side="right")


def test_fast_interacting_matches_dense():
    T = SIG / (kB_eV * 2.5)
    lat = Lattice(12, a=1.6e-9, seed=8).set_disorder(SIG)
    f = [mobility_interacting(lat, T, 0.1 * SIG, 0.02, n_eq=20000, n_meas=150000,
                              engine="fast", seed=s) for s in (1, 2, 3)]
    d = [mobility_interacting(lat, T, 0.1 * SIG, 0.02, n_eq=20000, n_meas=150000,
                              engine="dense", seed=s) for s in (1, 2, 3)]
    assert abs(np.mean(f) / np.mean(d) - 1) < 0.25


def test_device_kmc_zero_bias_equilibrium():
    d = DeviceKMC([Layer(8, HOMO=-5.2, LUMO=-2.0)], Lx=10, Ly=10, W_anode=5.2,
                  W_cathode=4.0, coulomb="full", seed=0)
    J0 = np.mean([d.run(0.0, n_eq=20000, n_meas=100000, seed=s)["J"] for s in range(3)])
    J1 = d.run(2.0, n_eq=20000, n_meas=100000)["J"]
    assert J1 > 0 and abs(J0) < 0.05 * J1


def test_device_kmc_bipolar_recombines():
    lay = [Layer(6, HOMO=-5.2, LUMO=-2.6)]
    d = DeviceKMC(lay, Lx=8, Ly=8, W_anode=5.1, W_cathode=2.7, coulomb="full", seed=1)
    r = d.run(3.5, n_eq=20000, n_meas=80000)
    assert r["J"] > 0 and r["rec_rate"] > 0 and 0 < r["gamma_b"] <= 1.05


def test_transient_celiv_high_ramp():
    from orchid.transient import celiv
    mu = 1e-8
    d = Device(L=100e-9, N=101, Eg=2.0, phi_anode_p=1.0, phi_cathode_n=1.0, mu_n=1e-14,
               mu_p=mu, recombination=False, grid_beta=2.0)
    A = 1e8
    t_est = 1e-7 * np.sqrt(2 / (3 * mu * A))
    r = celiv(d, A=A, t_end=12 * t_est, n_t=200, G_pulse=1e30, t_pulse=1e-10)
    assert abs(r["mu_juska"] / mu - 1) < 0.12


def test_transient_relaxes_to_steady_state():
    from orchid.transient import step_response
    d = Device(L=100e-9, N=81, Eg=1.2, mu_n=1e-8, mu_p=1e-8, G=1e27)
    ss = d.jv(np.array([0.3]))["J"][0]
    r = step_response(d, np.concatenate([[0], np.geomspace(1e-9, 1e-3, 60)]),
                      lambda t: 0.0 if t <= 0 else 0.3, V0=0.0)
    assert abs(r["J"][-1] / ss - 1) < 1e-3


def _sq(x):
    return x * x


def test_parallel_map_and_ensemble():
    assert parallel_map(_sq, range(6)) == [0, 1, 4, 9, 16, 25]
    assert parallel_map(_sq, range(6), backend="process", workers=2) == [0, 1, 4, 9, 16, 25]
    m, se, v = ensemble(float, [1, 2, 3])
    assert m == 2.0


def test_me_amg_matches_jacobi():
    T = SIG / (kB_eV * 3.0)
    lat = Lattice(16, a=1.6e-9, seed=9).set_disorder(SIG)
    a_ = me_linear_response(lat, T, 1e-4, alpha=10.0, precond="amg")[0]
    b_ = me_linear_response(lat, T, 1e-4, alpha=10.0, precond="jacobi")[0]
    assert abs(a_ / b_ - 1) < 1e-6


def test_gpu_kernel_in_cuda_simulator():
    import subprocess, sys, os
    code = ("import numpy as np\n"
            "from orchid.lattice import Lattice\n"
            "from orchid.gpu import kmc_independent_gpu\n"
            "from orchid.kmc import kmc_independent\n"
            "lat=Lattice(6,a=1.6e-9,seed=1).set_disorder(0.03)\n"
            "W=lat.rates(300,(0.01,0,0),alpha=10.)\n"
            "g=kmc_independent_gpu(lat,W,n_carriers=16,n_eq=50,n_meas=800,threads_per_block=16)\n"
            "c=kmc_independent(lat,W,64,50,3000)\n"
            "vg=g['dx'].sum()/g['dt'].sum(); vc=c['dx'].sum()/c['dt'].sum()\n"
            "print(vg/vc)\n")
    env = dict(os.environ, NUMBA_ENABLE_CUDASIM="1",
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = subprocess.run([sys.executable, "-W", "ignore", "-c", code], env=env,
                         capture_output=True, text=True, timeout=600)
    ratio = float(out.stdout.strip().splitlines()[-1])
    assert 0.6 < ratio < 1.6


def test_roadmap_done_items_importable():
    import importlib
    for r in roadmap.ROADMAP:
        if r["status"] == "done":
            importlib.import_module(r["module"])
    d, n = roadmap.progress()
    assert 0 < d < n


def test_device_kmc_electron_hole_symmetry():
    """Mirror-image hole-only and electron-only devices carry equal current."""
    common = dict(Lx=10, Ly=10, coulomb="full")
    dh = DeviceKMC([Layer(8, HOMO=-5.2, LUMO=-2.0)], W_anode=5.2, W_cathode=4.0, seed=4, **common)
    de = DeviceKMC([Layer(8, HOMO=-6.0, LUMO=-2.8)], W_anode=4.0, W_cathode=2.8, seed=4, **common)
    Jh = np.mean([dh.run(2.5, n_eq=20000, n_meas=80000, seed=s)["J"] for s in (1, 2)])
    Je = np.mean([de.run(2.5, n_eq=20000, n_meas=80000, seed=s)["J"] for s in (1, 2)])
    assert abs(Jh / Je - 1) < 0.35
