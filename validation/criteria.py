"""Pass/fail criteria for every ORCHID validation benchmark (V1-V27).

Each entry has: the benchmark id, what is compared, the published
reference, the results file and a function returning (value_text, passed).
Tolerances are the ones quoted in the slides and README. They are fixed
in advance. Do not loosen them to make a run pass; fix the code instead.
"""
import json
import os
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def load(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


def _v1():
    r = load("v1")
    sh, me = np.array(r["shat"]), np.array(r["me"])
    me_at = np.interp(r["kmc_shat"], sh, np.log(me))
    dev = np.abs(np.array(r["kmc"]) / np.exp(me_at) - 1).max()
    egdm = np.array(r["ratio_to_egdm"])[sh <= 3].max()
    return f"ME vs KMC {100*dev:.1f} %; ME/EGDM <= {egdm:.2f} (s<=3)", dev <= 0.10 and egdm <= 1.35


def _v2():
    r = load("v2v3")
    mono = all(np.all(np.diff(v) >= -1e-9) for v in r["me_ratio"].values())
    worst = max(np.abs(np.log(np.array(r["abs_ratio_to_egdm"][k]))).max() for k in ("3.0", "4.0"))
    return f"g1 monotone: {mono}; EGDM within x{np.exp(worst):.2f} (s=3,4)", mono and np.exp(worst) <= 1.6


def _v3():
    r = load("v2v3")
    mono = all(np.all(np.diff(v) >= -1e-9) for v in r["field_ratio"].values())
    return f"field enhancement monotone in F: {mono}", mono


def _v4():
    r = load("misc")["v4"]["2.0"]
    d = abs(r["E_last"] - r["Einf"]) / abs(r["Einf"])
    return f"{100*d:.1f} % from -sigma^2/kT at s=2", d <= 0.03


def _v5():
    r = np.array(load("devices")["v5"]["ratio"])
    return f"J/J_MG -> {r[-1]:.3f}", abs(r[-1] - 1) <= 0.10 and np.all(np.diff(r) < 0)


def _v6():
    r = load("devices")["v6"]
    return (f"max |dVoc| {r['max_abs_dev_mV']:.2f} mV; slope {r['slope_kT']:.4f} kT/e",
            r["max_abs_dev_mV"] <= 2.0 and abs(r["slope_kT"] - 1) <= 0.01)


def _v7():
    r = load("misc")["v7"]
    z = abs(r["D_kmc"] - r["D_analytic"]) / r["D_err"]
    return f"|D_KMC - D_exact| = {z:.2f} sigma", z <= 3


def _v8():
    r = load("misc")["v8"]
    return (f"Airy {r['max_airy_err']:.1e}; R+T+A {r['max_energy_err']:.1e}",
            r["max_airy_err"] <= 1e-12 and r["max_energy_err"] <= 1e-4)


def _v9():
    r = load("misc")
    g = r["v9_gdm"]
    dg = max(abs(g["fit"][k] / g["true"][k] - 1) for k in g["true"])
    d = r["v9_dd"]
    dd = abs(np.log10(d["fit"]["mu"] / d["true"]["mu"]))
    return (f"GDM params within {100*dg:.2f} %; DD mu within {dd:.3f} dec",
            dg <= 0.015 and dd <= 0.05 and d["fit"]["phi"] == d["true"]["phi"])


def _v10():
    r = load("tof")
    k = sorted(r, key=float)
    t = [r[x]["t_transit"] for x in k]
    return "t_transit grows with sigma-hat: " + str(bool(np.all(np.diff(t) > 0))), bool(np.all(np.diff(t) > 0))


def _v11():
    r = load("misc")["v11"]
    return f"detailed balance {r['detailed_balance_err']:.1e}", r["detailed_balance_err"] <= 1e-12


def _v12():
    r = load("devices")
    opv = r["v12_opv"]["free"]["PCE"] > r["v12_opv"]["braun"]["PCE"]
    o = r["v12_oled"]
    e0 = np.array(o["no annihilation"]["EQE"])
    e2 = np.array(o[[k for k in o if "17" in k][0]]["EQE"])
    roll = np.all(np.diff(e0) < 0) and np.all(e2 <= e0 + 1e-12)
    return f"roll-off and TTA ordering: {roll}; Braun PCE < free: {opv}", bool(roll and opv)


def _v13():
    r = load("device_kmc")
    Vd, Jd = np.array(r["Vdd"]), np.array(r["Jdd"])
    rat = [np.mean(r["J"]["full"][str(V)]) / np.interp(V, Vd, Jd) for V in r["V"] if V <= 2.0]
    return f"KMC/DD <= {max(rat):.2f} up to 2 V", max(rat) <= 1.6 and min(rat) >= 1 / 1.6


def _v14():
    r = load("transient")
    d = abs(r["madelung"] - 1.747564594633182)
    return f"Madelung {d:.1e}; image {r['image_rel_err']:.1e}", d <= 1e-8 and r["image_rel_err"] <= 1e-6


def _v15():
    r = load("published_software")
    s = r["sesame_max_rel_dev"]
    dark = s["dark, symmetric"]["max_rel_dev"]
    light = max(v["max_dJ_over_Jsc"] for k, v in s.items() if "light" in k)
    return (f"Sesame dark {dark:.1e}, light dJ/Jsc {light:.1e}; tmm R {r['tmm_max_abs_err_R']:.0e}",
            dark <= 1e-6 and light <= 1e-5 and r["tmm_max_abs_err_R"] <= 1e-12
            and r["tmm_max_rel_err_profile"] <= 1e-12 and r["tmm_max_abs_err_A"] <= 1e-4)


def _v16():
    r = load("parallel")
    sp = r["threads_hops_per_s"]["2"] / r["threads_hops_per_s"]["1"]
    ok = r["mpi_equals_serial"] and r["process_equals_serial"] and sp >= 1.8
    return f"thread speed-up {sp:.2f}x on 2 cores; MPI = serial", ok


def _v17():
    r = load("transient")
    m = r["mu_ratio"][-1]
    tp = r["tpv_300nm"]["tau"] / r["tpv_300nm"]["tau_L"]
    return f"CELIV mu/mu_true -> {m:.3f}; TPV tau/tau_L {tp:.2f}", abs(m - 1) <= 0.1 and tp <= 1.2


def _v18():
    r = load("v18")
    ok = all(o["EQE"]["TTA+TPQ"] < o["EQE"]["TTA"] for o in r["oled"])
    return f"gamma_TT spread {100*r['gamma_spread']:.1f} %; TPQ lowers EQE: {ok}", r["gamma_spread"] <= 0.1 and ok


def _v19():
    r = load("v19")
    h = r["hetero_rel_dev"]["1e+27"]
    c = max(abs(x - 1) for x in r["insulator_C"])
    c0 = max(abs(r[k]["C0_over_Cgeo"] / 0.75 - 1) for k in ("V20", "V80"))
    f = max(abs(r[k]["fmax_tdc"] / 0.72 - 1) for k in ("V20", "V80"))
    return (f"hetero vs Sesame {h:.1e}; C0/Cgeo within {100*c0:.1f} % of 3/4; f tau within {100*f:.1f} %",
            h <= 1e-5 and c <= 1e-9 and c0 <= 0.05 and f <= 0.05)


def _v20():
    r = load("v20")
    return (f"CPS {r['cps_max_abs_err']:.1e}; AM1.5G {r['am15g_integral']:.1f} W/m2",
            r["cps_max_abs_err"] <= 1e-3 and abs(r["am15g_integral"] / 1000 - 1) <= 1e-3)


def _v21():
    r = load("v21")
    return f"DIPRO vs half splitting {100*r['max_rel']:.2f} %", r["max_rel"] <= 0.005


def _v22():
    r = load("v22")
    ok = all(r["covered"].values())
    return f"held-out RMSE {r['rmse']:.4f} dec; coverage {sum(r['covered'].values())}/3", ok and r["rmse"] <= 0.02


def _v23():
    r = load("v23")
    s = max(abs(x - 1) for x in r["sub_over_serial"])
    return (f"mu_lin {r['mu_lin_over_input']:.3f}, mu_sat {r['mu_sat_over_input']:.3f}; sublattice {100*s:.0f} %",
            abs(r["mu_lin_over_input"] - 1) <= 0.05 and abs(r["mu_sat_over_input"] - 1) <= 0.10 and s <= 0.12)


def _v24():
    r = load("v24")
    jr = max(abs(x - 1) for x in r["J_ratio"])
    return (f"bookkeeping {r['balance_max']}; eta {r['eta_noquench']:.3f} vs {r['eta_exact']:.3f} "
            f"({r['z_eta']:+.1f} sigma); J coupled/charge-only within {100*jr:.0f} %",
            r["balance_max"] == 0 and abs(r["z_eta"]) <= 3 and jr <= 0.15)


def _v25():
    r = load("v25")
    e = abs(r["extrap_small_ratio"] - 1)
    s = abs(r["single_tau10_ratio"] - 1)
    tol = max(2 * r["extrap_small_err"], 0.03)
    return (f"single window {100*s:.1f} % -> extrapolated {100*e:.1f} % (+-{100*r['extrap_small_err']:.1f})",
            e <= tol and e < s)


def _v26():
    r = load("v26")
    return (f"centres {r['exact_centre_err_m']:.0e} m (GRO {r['gro_centre_err_m']:.0e} m), "
            f"{r['molecules_split_by_pbc']} PBC-split molecules; graph=lattice {r['mu_rel_dev']:.0e}",
            r["exact_centre_err_m"] <= 1e-20 and r["gro_centre_err_m"] <= 5e-13
            and r["molecules_split_by_pbc"] > 0 and r["mu_rel_dev"] <= 1e-10)


def _v27():
    r = load("v27")
    return (f"device KMC {r['device_kmc_events_per_s']:.2e} ev/s, coupled {r['coupled_kmc_events_per_s']:.2e} ev/s, "
            f"ME 64k sites {r['me_linear_response_64k_sites_s']:.2f} s", None)


CRITERIA = [
    ("V1", "mu(T), p->0", "Pasveer PRL 94, 206601 (2005)", "bench_temperature", _v1),
    ("V2", "mu(p)", "EGDM g1", "bench_density_field", _v2),
    ("V3", "mu(F)", "EGDM g2", "bench_density_field", _v3),
    ("V4", "<E>_inf", "Baessler 1993", "bench_misc", _v4),
    ("V5", "SCLC", "Mott-Gurney", "bench_devices", _v5),
    ("V6", "Voc(G)", "Koster APL 86, 123509 (2005)", "bench_devices", _v6),
    ("V7", "exciton D", "sum k r^2 / 6 (exact)", "bench_misc", _v7),
    ("V8", "optics", "Airy formula; R+T+A=1", "bench_misc", _v8),
    ("V9", "inverse problems", "synthetic ground truth", "bench_misc", _v9),
    ("V10", "TOF transients", "Scher-Montroll dispersive transport", "bench_tof", _v10),
    ("V11", "Marcus vs MA", "detailed balance", "bench_misc", _v11),
    ("V12", "OLED / OPV", "device-physics trends", "bench_devices", _v12),
    ("V13", "3D device KMC", "1D DD (van der Holst 2009)", "bench_device_kmc", _v13),
    ("V14", "Madelung, images", "exact constants", "bench_transient", _v14),
    ("V15", "DD and optics", "NIST Sesame; tmm", "bench_published_software", _v15),
    ("V16", "parallel", "ideal scaling; MPI = serial", "bench_parallel", _v16),
    ("V17", "CELIV, TPV", "Juska 2000; 1/(2 gamma n)", "bench_transient", _v17),
    ("V18", "excitons in device", "bimolecular law", "bench_v18_excitons", _v18),
    ("V19", "hetero DD, impedance", "Sesame; C0 = 3/4 Cgeo (Martens)", "bench_v19_dd_extensions", _v19),
    ("V20", "dipole outcoupling", "CPS mirror formulas; ASTM G173", "bench_v20_outcoupling", _v20),
    ("V21", "DIPRO, graphs", "dimer splitting", "bench_v21_morphology", _v21),
    ("V22", "ML / Bayes", "held-out data; CI coverage", "bench_v22_ml", _v22),
    ("V23", "OFET, sublattice KMC", "gradual channel; serial KMC", "bench_v23_ofet_sublattice", _v23),
    ("V24", "concurrent exciton+charge KMC", "exact yield; bookkeeping; v0.3 pipeline", "bench_v24_coupled", _v24),
    ("V25", "sublattice tau->0 extrapolation", "serial Fenwick KMC", "bench_v25_sublattice_extrap", _v25),
    ("V26", "MD trajectory import", "exact centres; lattice ME", "bench_v26_md_import", _v26),
    ("V27", "performance", "throughput record", "bench_v27_performance", _v27),
]


def evaluate():
    rows = []
    for vid, what, ref, script, fn in CRITERIA:
        try:
            val, ok = fn()
        except FileNotFoundError:
            val, ok = "no results (run the benchmark)", False
        except Exception as e:  # pragma: no cover
            val, ok = f"error: {e}", False
        rows.append(dict(id=vid, what=what, reference=ref, script=script, value=val,
                         passed=None if ok is None else bool(ok)))
    return rows
