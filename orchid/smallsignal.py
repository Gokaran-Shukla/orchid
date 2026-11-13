r"""Small-signal frequency-domain analysis: impedance / admittance
spectroscopy, IMPS and IMVS, by linearising the coupled drift-diffusion
system about a steady state.

.. math::
   (\mathbf J + i\omega\mathbf M)\,\delta\mathbf z = -\frac{\partial\mathbf R}{\partial V}\delta V
   - \frac{\partial\mathbf R}{\partial G}\delta G

:math:`\mathbf J` is the steady-state Jacobian (banded, colour finite differences) and
:math:`\mathbf M` the diagonal mass matrix of the continuity equations. The
terminal current includes the displacement current
:math:`i\omega\varepsilon\,\delta F`, averaged over all edges (it is
divergence-free).

* :func:`admittance` - Y(omega) = dJ/dV, capacitance C = Im Y / omega,
  conductance G = Re Y.
* :func:`imps` - dJ/dG at fixed V (intensity-modulated photocurrent).
* :func:`imvs` - dV/dG at open circuit: dV/dG = -(dJ/dG)/(dJ/dV).

Validation: an insulator gives C = eps/L at all frequencies. A trap-free
SCLC diode gives C(0)/C_geo = 3/4 and the transit-time feature in the
negative differential susceptance (Martens et al., PRB 60, R8489 (1999)).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def _steady(dev, V):
    res = dev.jv(np.array([V]))
    pr = res["profiles"][0]
    z = np.stack([pr["psi"][1:-1], np.log(pr["n"][1:-1] / dev.Nc_x[1:-1]),
                  np.log(pr["p"][1:-1] / dev.Nv_x[1:-1])], 1).ravel()
    return z


def _banded_to_csr(ab, bw=5):
    nz = ab.shape[1]
    rows, cols, vals = [], [], []
    for k in range(2 * bw + 1):
        off = bw - k                      # ab[bw + i - j, j] = A[i, j]
        j = np.arange(nz)
        i = j - off
        m = (i >= 0) & (i < nz)
        rows.append(i[m])
        cols.append(j[m])
        vals.append(ab[k, m])
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(nz, nz))


def _current_lin(dev, z, V, dz, dV, omega, eps_fd=1e-6):
    """Linearised total current for complex perturbation (dz, dV)."""
    def Jc(zz, VV):
        psi, u, v = dev.unpack(zz, VV)
        Jn, Jp, F, _, _ = dev.currents(psi, dev.Nc_x * np.exp(u), dev.Nv_x * np.exp(v))
        return Jn + Jp, psi
    J0, psi0 = Jc(z, V)
    out = np.zeros_like(J0, dtype=complex)
    for part, fac in ((np.real, 1.0), (np.imag, 1j)):
        d = part(dz)
        dv = part(np.asarray(dV, dtype=complex))
        scale = eps_fd / max(np.max(np.abs(d)), abs(dv), 1e-30)
        J1, psi1 = Jc(z + scale * d, V + scale * dv)
        dJc = (J1 - J0) / scale
        dpsi = (psi1 - psi0) / scale
        dF = -(dpsi[1:] - dpsi[:-1]) / dev.h
        out += fac * (dJc + 1j * omega * dev.eps_e * dF)
    return out.mean()


def _system(dev, z, V):
    r0 = dev.residual(z, V)
    J = _banded_to_csr(dev.jacobian_banded(z, V, r0))
    psi, u, v = dev.unpack(z, V)
    n = dev.Nc_x[1:-1] * np.exp(u[1:-1])
    p = dev.Nv_x[1:-1] * np.exp(v[1:-1])
    m = np.zeros((len(n), 3))
    m[:, 1] = -n / (n + 1.0) * 1e-6
    m[:, 2] = p / (p + 1.0) * 1e-6
    M = sp.diags(m.ravel())
    hV = 1e-6
    dRdV = (dev.residual(z, V + hV) - r0) / hV
    G0 = dev.Gx.copy()
    dev.Gx = G0 * (1 + 1e-6)
    dRdG = (dev.residual(z, V) - r0) / 1e-6          # per relative change of G
    dev.Gx = G0
    return J, M, dRdV, dRdG


def admittance(dev, V, freqs):
    """Admittance spectrum at bias V. Returns dict(f, Y, C, G)."""
    z = _steady(dev, V)
    J, M, dRdV, _ = _system(dev, z, V)
    Y = []
    for f in freqs:
        w = 2 * np.pi * f
        A = (J + 1j * w * M).tocsc()
        dz = spla.spsolve(A, -dRdV.astype(complex))
        Y.append(_current_lin(dev, z, V, dz, 1.0, w))
    Y = np.array(Y)
    w = 2 * np.pi * np.asarray(freqs)
    return dict(f=np.asarray(freqs), Y=Y, C=Y.imag / w, G=Y.real)


def imps(dev, V, freqs):
    """IMPS transfer function dJ/d(ln G) at fixed bias V [A/m^2]."""
    z = _steady(dev, V)
    J, M, _, dRdG = _system(dev, z, V)
    H = []
    for f in freqs:
        w = 2 * np.pi * f
        dz = spla.spsolve((J + 1j * w * M).tocsc(), -dRdG.astype(complex))
        H.append(_current_lin(dev, z, V, dz, 0.0, w))
    return dict(f=np.asarray(freqs), H=np.array(H))


def imvs(dev, Voc, freqs):
    """IMVS dV/d(ln G) at open circuit = -IMPS / Y."""
    y = admittance(dev, Voc, freqs)["Y"]
    h = imps(dev, Voc, freqs)["H"]
    return dict(f=np.asarray(freqs), H=-h / y)
