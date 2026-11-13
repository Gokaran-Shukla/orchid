"""Optical-constant database and reference solar spectrum.

* :func:`nk(name)` returns a callable lam[m] -> complex n + ik from files in
  ``orchid/data/nk`` (refractiveindex.info YAML format, public domain / CC0).
  Bundled: Al (Rakic LD), Ag, Au (Johnson & Christy), ITO (König),
  PEDOT:PSS (Chen), P3HT:PC61BM (Stelling), SiO2 (Malitson), N-BK7 (Schott).
  Any other refractiveindex.info YAML file can be loaded with
  :func:`load_refractiveindex`.
* :func:`am15g` - ASTM G173-03 global tilt spectrum (NREL; file as
  distributed with pvlib), W m^-2 per m of wavelength; integral 1000.4 W/m^2.
"""
import os
import numpy as np

DATA = os.path.join(os.path.dirname(__file__), "data")


def _parse(path):
    import yaml
    d = yaml.safe_load(open(path, encoding="utf-8"))
    n_part, k_part = None, None
    for blk in d["DATA"]:
        t = blk["type"]
        if t.startswith("tabulated"):
            arr = np.array([[float(v) for v in line.split()] for line in blk["data"].strip().splitlines()])
            if t == "tabulated nk":
                n_part = ("tab", arr[:, 0], arr[:, 1])
                k_part = ("tab", arr[:, 0], arr[:, 2])
            elif t == "tabulated n":
                n_part = ("tab", arr[:, 0], arr[:, 1])
            elif t == "tabulated k":
                k_part = ("tab", arr[:, 0], arr[:, 1])
        elif t.startswith("formula"):
            c = np.array([float(v) for v in str(blk["coefficients"]).split()])
            n_part = ("formula", int(t.split()[1]), c)
    return n_part, k_part


def _eval_n(part, lam_um):
    if part[0] == "tab":
        return np.interp(lam_um, part[1], part[2])
    kind, c = part[1], part[2]
    l2 = lam_um ** 2
    if kind == 1:        # Sellmeier: n^2-1 = C0 + sum Ci l^2/(l^2 - C_{i+1}^2)
        s = c[0] + sum(c[i] * l2 / (l2 - c[i + 1] ** 2) for i in range(1, len(c) - 1, 2))
        return np.sqrt(1 + s)
    if kind == 2:        # Sellmeier-2: n^2-1 = C0 + sum Ci l^2/(l^2 - C_{i+1})
        s = c[0] + sum(c[i] * l2 / (l2 - c[i + 1]) for i in range(1, len(c) - 1, 2))
        return np.sqrt(1 + s)
    raise NotImplementedError(f"formula {kind}")


def load_refractiveindex(path):
    """Callable lam[m] -> complex index from a refractiveindex.info YAML file."""
    n_part, k_part = _parse(path)

    def f(lam):
        lu = np.asarray(lam, float) * 1e6
        n = _eval_n(n_part, lu)
        k = np.interp(lu, k_part[1], k_part[2]) if k_part is not None else 0.0
        return n + 1j * k
    f.__name__ = os.path.basename(path)
    return f


def available():
    return sorted(f[:-4] for f in os.listdir(os.path.join(DATA, "nk")) if f.endswith(".yml"))


def nk(name):
    """Bundled material by name (see :func:`available`)."""
    return load_refractiveindex(os.path.join(DATA, "nk", name + ".yml"))


def am15g(lam=None):
    """ASTM G173 global tilt irradiance [W m^-2 m^-1] (at ``lam`` [m] if given)."""
    d = np.loadtxt(os.path.join(DATA, "am15g.csv"), delimiter=",", skiprows=1)
    lam0, E = d[:, 0] * 1e-9, d[:, 1] * 1e9
    if lam is None:
        return lam0, E
    return np.interp(lam, lam0, E)
