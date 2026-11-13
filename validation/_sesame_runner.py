"""Run NIST Sesame (https://github.com/usnistgov/sesame) on the device cases
listed in a JSON file written by bench_published_software.py.

Usage:  python -I _sesame_runner.py <path-to-sesame-checkout> <cases.json> <out.json>

Geometry: the device is mirrored (Sesame-left = ORCHID cathode, Sesame-right
= ORCHID anode). With Sesame's sign convention ORCHID's forward bias V is
Sesame bias -V, and the current sign is flipped on output. The mapping was
checked on the short-circuit photocurrent and the dark diode current.
"""
import json
import logging
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, sys.argv[1])
import sesame  # noqa: E402

logging.disable(logging.CRITICAL)
cases = json.load(open(sys.argv[2]))
results = {}
for name, c in cases.items():
    if "layers" in c:                     # heterostructure: explicit grid
        x_m = np.asarray(c["x_m"])
        L = x_m[-1] * 100.0
        xs = L - x_m[::-1] * 100.0          # mirrored (cathode on the left)
        sysm = sesame.Builder(xs, input_length="cm", T=c["T"])
        bounds = np.cumsum([l["d"] for l in c["layers"]]) * 100.0
        for k, l in enumerate(c["layers"]):
            lo = 0.0 if k == 0 else bounds[k - 1]
            hi = bounds[k]
            # ORCHID assigns an interface node to the left (anode-side) layer
            tol = 1e-9 * L
            lo_eff = lo - tol if k == 0 else lo + tol

            def make_loc(lo_eff, hi, L, tol):
                def loc(xx):
                    X = L - xx
                    return (X > lo_eff) & (X <= hi + tol)
                return loc
            loc = make_loc(lo_eff, hi, L, tol)
            sysm.add_material(dict(Nc=l["Nc"] * 1e-6, Nv=l["Nv"] * 1e-6, Eg=l["Eg"],
                                   affinity=l["chi"], epsilon=l["eps_r"], mu_e=l["mu_n"] * 1e4,
                                   mu_h=l["mu_p"] * 1e4, tau_e=c["tau"], tau_h=c["tau"],
                                   Et=0.0, B=0.0), loc)
        wf_anode, wf_cathode = c["W_anode"], c["W_cathode"]
    else:
        L = c["L_m"] * 100.0                                    # cm
        s = np.linspace(0, 1, c["N"])
        x = L * 0.5 * (1 + np.tanh(c["beta"] * (2 * s - 1)) / np.tanh(c["beta"]))
        sysm = sesame.Builder(x, input_length="cm", T=c["T"])
        tau = c["tau"]
        sysm.add_material(dict(Nc=c["Nc_m3"] * 1e-6, Nv=c["Nv_m3"] * 1e-6, Eg=c["Eg"],
                               affinity=c["chi"], epsilon=c["eps_r"], mu_e=c["mu_n_m2"] * 1e4,
                               mu_h=c["mu_p_m2"] * 1e4, tau_e=tau, tau_h=tau, Et=0.0, B=0.0))
        wf_anode = c["chi"] + c["Eg"] - c["phi_a"]
        wf_cathode = c["chi"] + c["phi_c"]
    # empirically verified orientation: cathode on the left, anode on the
    # right, ORCHID forward bias V <-> Sesame bias -V
    sysm.contact_type("Schottky", "Schottky", left_wf=wf_cathode, right_wf=wf_anode)
    S = 1e7
    sysm.contact_S(S, S, S, S)
    G = c["G_m3s"] * 1e-6
    if G > 0:
        def make_g(g):
            def gfun(xx):              # Sesame inspects the argument count
                return g + 0 * xx
            return gfun
        sysm.generation(make_g(G))
    V = np.asarray(c["V"])
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as td:
        os.chdir(td)
        j = sesame.IVcurve(sysm, -V, "iv", verbose=False)
        os.chdir(cwd)
    results[name] = dict(V=V.tolist(), J_A_m2=(-np.asarray(j) * sysm.scaling.current * 1e4).tolist())
json.dump(results, open(sys.argv[3], "w"))
print("sesame ok")
