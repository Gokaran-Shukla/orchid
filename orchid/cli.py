"""Command-line interface:  ``orchid <command> [options]``.

Commands
--------
mobility   master-equation / KMC mobility of a Gaussian-disordered lattice
jv         drift-diffusion J-V of a single-layer diode (dark or illuminated)
egdm       evaluate the EGDM / GDM analytical models
roadmap    print the development roadmap (done / planned)
parallel   show MPI ranks, threads and GPU availability
info       print module overview
"""
import argparse
import json
import numpy as np


def _mobility(a):
    from .lattice import Lattice
    from .constants import M2VS_TO_CM2VS
    lat = Lattice(a.L, a=a.a * 1e-9, rc=a.rc, seed=a.seed).set_disorder(a.sigma, a.disorder)
    if a.method == "me":
        from .master_equation import me_linear_response
        mu, info = me_linear_response(lat, a.T, a.density, alpha=a.alpha, nu0=a.nu0)
        res = dict(method="ME", mu_cm2Vs=mu * M2VS_TO_CM2VS, EF=info["EF"])
    else:
        from .kmc import mobility_independent, mobility_interacting
        if a.density <= 0:
            mu, err = mobility_independent(lat, a.T, a.field * a.a * 1e-9, alpha=a.alpha,
                                           nu0=a.nu0, kind=a.rate)
            res = dict(method="KMC", mu_cm2Vs=mu * M2VS_TO_CM2VS, err=err * M2VS_TO_CM2VS)
        else:
            mu = mobility_interacting(lat, a.T, a.field * a.a * 1e-9, a.density,
                                      coulomb=a.coulomb, alpha=a.alpha, nu0=a.nu0)
            res = dict(method="KMC-interacting", mu_cm2Vs=mu * M2VS_TO_CM2VS)
    print(json.dumps(res, indent=1))


def _jv(a):
    from .drift_diffusion import Device, pv_metrics
    d = Device(L=a.thickness * 1e-9, N=a.nodes, Eg=a.Eg, mu_n=a.mu_n * 1e-4,
               mu_p=a.mu_p * 1e-4, phi_anode_p=a.phi_a, phi_cathode_n=a.phi_c,
               G=a.G, langevin_prefactor=a.zeta, T=a.T, eps_r=a.eps_r)
    V = np.linspace(a.Vmin, a.Vmax, a.nV)
    r = d.jv(V)
    for v, j in zip(V, r["J"]):
        print(f"{v:10.4f} {j / 10:14.6e}")    # mA/cm^2
    if a.G > 0:
        print(json.dumps({k: float(v) for k, v in pv_metrics(V, r["J"]).items()}))


def _egdm(a):
    from .models import egdm_pasveer
    mu = egdm_pasveer(a.T, a.p, a.F, a.sigma, a.a * 1e-9, nu0=a.nu0)
    print(f"mu = {mu * 1e4:.4e} cm^2/Vs")


def main(argv=None):
    p = argparse.ArgumentParser(prog="orchid", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("mobility")
    m.add_argument("--method", choices=["me", "kmc"], default="me")
    m.add_argument("--sigma", type=float, default=0.1, help="eV")
    m.add_argument("--T", type=float, default=300.0)
    m.add_argument("--L", type=int, default=30)
    m.add_argument("--a", type=float, default=1.6, help="nm")
    m.add_argument("--rc", type=float, default=3 ** 0.5)
    m.add_argument("--alpha", type=float, default=10.0, help="1/a")
    m.add_argument("--nu0", type=float, default=1e12)
    m.add_argument("--density", type=float, default=1e-6, help="fraction of sites")
    m.add_argument("--field", type=float, default=1e7, help="V/m (KMC)")
    m.add_argument("--rate", default="ma", choices=["ma", "marcus", "sym_ma"])
    m.add_argument("--disorder", default="gaussian",
                   choices=["gaussian", "dipolar", "smoothed", "exponential"])
    m.add_argument("--coulomb", action="store_true")
    m.add_argument("--seed", type=int, default=0)
    m.set_defaults(f=_mobility)
    j = sub.add_parser("jv")
    j.add_argument("--thickness", type=float, default=100.0, help="nm")
    j.add_argument("--nodes", type=int, default=161)
    j.add_argument("--Eg", type=float, default=1.2)
    j.add_argument("--mu_n", type=float, default=1e-4, help="cm^2/Vs")
    j.add_argument("--mu_p", type=float, default=1e-4, help="cm^2/Vs")
    j.add_argument("--phi_a", type=float, default=0.1)
    j.add_argument("--phi_c", type=float, default=0.1)
    j.add_argument("--G", type=float, default=0.0, help="m^-3 s^-1")
    j.add_argument("--zeta", type=float, default=1.0)
    j.add_argument("--T", type=float, default=300.0)
    j.add_argument("--eps_r", type=float, default=3.5)
    j.add_argument("--Vmin", type=float, default=-0.5)
    j.add_argument("--Vmax", type=float, default=1.0)
    j.add_argument("--nV", type=int, default=31)
    j.set_defaults(f=_jv)
    e = sub.add_parser("egdm")
    e.add_argument("--T", type=float, default=300.0)
    e.add_argument("--p", type=float, default=1e22, help="m^-3")
    e.add_argument("--F", type=float, default=1e7, help="V/m")
    e.add_argument("--sigma", type=float, default=0.1)
    e.add_argument("--a", type=float, default=1.6, help="nm")
    e.add_argument("--nu0", type=float, default=1e12)
    e.set_defaults(f=_egdm)
    r = sub.add_parser("roadmap", help="print the development roadmap")
    r.add_argument("--version", default=None)
    r.set_defaults(f=lambda a: (print(__import__("orchid.roadmap", fromlist=["x"]).summary(a.version)),
                                print("progress: %d/%d done" % __import__("orchid.roadmap", fromlist=["x"]).progress())))
    pz = sub.add_parser("parallel", help="show available parallel resources")
    pz.set_defaults(f=lambda a: print(json.dumps(__import__("orchid.parallel", fromlist=["x"]).describe(), indent=1)))
    i = sub.add_parser("info")
    i.set_defaults(f=lambda a: print(__import__("orchid").__doc__))
    a = p.parse_args(argv)
    a.f(a)


if __name__ == "__main__":
    main()
