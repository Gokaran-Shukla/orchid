r"""2D drift-diffusion (roadmap R19): organic field-effect transistors and
bulk-heterojunction morphologies.

Rectangular tensor-product grid (x uniform or user, y user, e.g. refined at
the gate interface). Every node is either *insulator* (Poisson only) or
*semiconductor* with its own chi, Eg, Nc, Nv, mu_n, mu_p, eps_r and G, so a
donor/acceptor morphology is a node map. Dirichlet contacts are node sets:
gate (potential only), source/drain or anode/cathode (potential and carrier
densities from injection barriers). All other boundaries have zero normal
flux.

Discretisation: box integration, 2D Scharfetter-Gummel fluxes with band
offsets along every edge, unknowns (psi, ln n/Nc, ln p/Nv). Newton's method
with a sparse Jacobian from 15-colour finite differences (distance-2
colouring (i + 2j) mod 5 of the 5-point stencil times 3 variables), sparse
LU solve and voltage continuation.

Validation: OFET linear and saturation currents vs the gradual-channel
approximation :math:`I_D=\frac{W}{L}\mu C_i[(V_G-V_T)V_D-V_D^2/2]`
(Shockley; Horowitz, Adv. Mater. 10, 365 (1998)).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .constants import q, eps0, kB_eV
from .drift_diffusion import bernoulli


class Device2D:
    def __init__(self, x, y, T=300.0):
        self.x, self.y = np.asarray(x, float), np.asarray(y, float)
        self.Nx, self.Ny = len(self.x), len(self.y)
        self.N = self.Nx * self.Ny
        self.T = T
        self.Vt = kB_eV * T
        N = self.N
        self.semi = np.zeros(N, bool)
        self.chi = np.zeros(N)
        self.Eg = np.full(N, 2.0)
        self.Nc = np.full(N, 1e26)
        self.Nv = np.full(N, 1e26)
        self.mun = np.full(N, 1e-8)
        self.mup = np.full(N, 1e-8)
        self.epsr = np.full(N, 3.5)
        self.G = np.zeros(N)
        self.langevin = 0.0
        self.contacts = {}          # name -> dict(nodes, W, kind)
        self.carriers = ("n", "p")  # set to ("p",) for unipolar hole devices
        hx = np.diff(self.x)
        hy = np.diff(self.y)
        wx = np.zeros(self.Nx)
        wy = np.zeros(self.Ny)
        wx[:-1] += hx / 2
        wx[1:] += hx / 2
        wy[:-1] += hy / 2
        wy[1:] += hy / 2
        self.hx, self.hy, self.wx, self.wy = hx, hy, wx, wy

    def idx(self, i, j):
        return i * self.Ny + j

    def set_region(self, mask_fn, semiconductor=True, **params):
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        m = mask_fn(X, Y).ravel()
        if semiconductor:
            self.semi[m] = True
        else:
            self.semi[m] = False
        for k, v in params.items():
            getattr(self, k)[m] = v
        return m

    def add_contact(self, name, mask_fn, W, kind="ohmic"):
        """kind: "ohmic" (psi + carriers, needs semiconductor nodes) or "gate"."""
        X, Y = np.meshgrid(self.x, self.y, indexing="ij")
        nodes = np.where(mask_fn(X, Y).ravel())[0]
        self.contacts[name] = dict(nodes=nodes, W=W, kind=kind)

    # ------------------------------------------------------------------
    def _prepare(self):
        N, Ny = self.N, self.Ny
        I, J = np.divmod(np.arange(N), Ny)
        ex = np.where(I < self.Nx - 1)[0]
        ey = np.where(J < Ny - 1)[0]
        self.e_a = np.concatenate([ex, ey])
        self.e_b = np.concatenate([ex + Ny, ey + 1])
        self.e_h = np.concatenate([self.hx[I[ex]], self.hy[J[ey]]])
        self.e_w = np.concatenate([self.wy[J[ex]], self.wx[I[ey]]])   # face width
        a, b = self.e_a, self.e_b
        self.e_eps = eps0 * 2 * self.epsr[a] * self.epsr[b] / (self.epsr[a] + self.epsr[b])
        self.e_semi = self.semi[a] & self.semi[b]
        hm = lambda m: 2 * m[a] * m[b] / (m[a] + m[b])
        self.e_mun, self.e_mup = hm(self.mun), hm(self.mup)
        Vt = self.Vt
        self.off_n = (self.chi[b] - self.chi[a]) / Vt + np.log(self.Nc[b] / self.Nc[a])
        self.off_p = ((self.chi + self.Eg)[b] - (self.chi + self.Eg)[a]) / Vt - np.log(self.Nv[b] / self.Nv[a])
        self.vol = np.outer(self.wx, self.wy).ravel()
        self.ni2 = self.Nc * self.Nv * np.exp(-self.Eg / Vt)
        # Dirichlet bookkeeping
        self.fix_psi = np.zeros(N, bool)
        self.fix_c = ~self.semi.copy()          # insulator: carriers fixed (inert)
        for c in self.contacts.values():
            self.fix_psi[c["nodes"]] = True
            if c["kind"] == "ohmic":
                self.fix_c[c["nodes"]] = True
        I, J = np.divmod(np.arange(N), Ny)
        col = (I + 2 * J) % 5
        self.colors = col
        # neighbour lists for Jacobian assembly
        self.nb = [np.arange(N)]
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ii, jj = I + di, J + dj
            ok = (ii >= 0) & (ii < self.Nx) & (jj >= 0) & (jj < Ny)
            self.nb.append(np.where(ok, ii * Ny + jj, -1))

    def _bc(self, Vs):
        """Dirichlet values: dict contact -> applied voltage."""
        N = self.N
        psi_b = np.zeros(N)
        u_b = np.full(N, -40.0)
        v_b = np.full(N, -40.0)
        for name, c in self.contacts.items():
            V = Vs.get(name, 0.0)
            nodes = c["nodes"]
            # vacuum level at a metal: psi = -(W) shifted by applied V (electron
            # energy E_vac = -q psi): psi = V - W  (relative reference)
            psi_b[nodes] = V - c["W"]
            if c["kind"] == "ohmic":
                W = c["W"]
                bn = W - self.chi[nodes]
                bp = self.chi[nodes] + self.Eg[nodes] - W
                u_b[nodes] = -bn / self.Vt
                v_b[nodes] = -bp / self.Vt
        return psi_b, u_b, v_b

    def residual(self, z, Vs):
        N = self.N
        Z = z.reshape(N, 3)
        psi, u, v = Z[:, 0].copy(), Z[:, 1].copy(), Z[:, 2].copy()
        pb, ub, vb = self._bc(Vs)
        n = self.Nc * np.exp(u)
        p = self.Nv * np.exp(v)
        a, b, h, w = self.e_a, self.e_b, self.e_h, self.e_w
        Vt = self.Vt
        # Poisson
        flux = self.e_eps * (psi[b] - psi[a]) / h * w
        div = np.zeros(N)
        np.add.at(div, a, flux)
        np.add.at(div, b, -flux)
        rho = np.where(self.semi, q * (p - n), 0.0)
        r_psi = (div + rho * self.vol) / (eps0 * Vt)
        # currents (semiconductor edges only)
        D = (psi[b] - psi[a]) / Vt
        Dn, Dp = D + self.off_n, D + self.off_p
        s = self.e_semi
        Jn = np.where(s, q * self.e_mun * Vt / h * (n[b] * bernoulli(Dn) - n[a] * bernoulli(-Dn)), 0.0) * w
        Jp = np.where(s, q * self.e_mup * Vt / h * (p[a] * bernoulli(Dp) - p[b] * bernoulli(-Dp)), 0.0) * w
        divn = np.zeros(N)
        divp = np.zeros(N)
        np.add.at(divn, a, Jn)
        np.add.at(divn, b, -Jn)
        np.add.at(divp, a, Jp)
        np.add.at(divp, b, -Jp)
        R = self.langevin * q * (self.mun + self.mup) / (eps0 * self.epsr) * (n * p - self.ni2)
        RG = np.where(self.semi, (R - self.G) * self.vol, 0.0)
        # scale continuity rows by q (n+1) mu Vt (dimensionless, O(1) entries
        # even for an immobile / absent carrier species)
        r_n = (divn - q * RG) / (q * n * self.mun * Vt)
        r_p = (-divp - q * RG) / (q * p * self.mup * Vt)
        # Dirichlet rows
        r_psi = np.where(self.fix_psi, psi - pb, r_psi)
        r_n = np.where(self.fix_c, u - ub, r_n)
        r_p = np.where(self.fix_c, v - vb, r_p)
        if "n" not in self.carriers:       # frozen species: keep equilibrium values
            r_n = u - self._u_eq
        if "p" not in self.carriers:
            r_p = v - self._v_eq
        return np.stack([r_psi, r_n, r_p], 1).ravel()

    def jacobian(self, z, Vs, r0):
        N = self.N
        rows, cols, vals = [], [], []
        for c in range(5):
            nodes = np.where(self.colors == c)[0]
            for var in range(3):
                colidx = 3 * nodes + var
                dz = 1e-7 * np.maximum(1.0, np.abs(z[colidx]))
                zp = z.copy()
                zp[colidx] += dz
                dr = (self.residual(zp, Vs) - r0).reshape(N, 3)
                for k in range(5):
                    nbn = self.nb[k][nodes]
                    ok = nbn >= 0
                    for rv in range(3):
                        rows.append(3 * nbn[ok] + rv)
                        cols.append(colidx[ok])
                        vals.append(dr[nbn[ok], rv] / dz[ok])
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(3 * N, 3 * N))

    def newton(self, z, Vs, tol=1e-7, maxit=60):
        for it in range(maxit):
            r = self.residual(z, Vs)
            Jm = self.jacobian(z, Vs, r)
            try:
                dz = spla.spsolve(Jm.tocsc(), -r)
            except Exception:
                return z, False
            if not np.all(np.isfinite(dz)):
                return z, False
            m = np.max(np.abs(dz))
            # component-wise limiting: psi steps <= 1 V, ln-density steps <= 3
            D = dz.reshape(-1, 3)
            D[:, 0] = np.clip(D[:, 0], -1.0, 1.0)
            D[:, 1:] = np.clip(D[:, 1:], -3.0, 3.0)
            D = D.ravel()
            # backtracking line search on the residual norm
            r0n = np.linalg.norm(r)
            lam = 1.0
            for _ in range(12):
                zt = z + lam * D
                rt = self.residual(zt, Vs)
                if np.all(np.isfinite(rt)) and np.linalg.norm(rt) < (1 - 1e-4 * lam) * r0n:
                    break
                lam *= 0.5
            z = zt
            if m < tol:
                return z, True
        return z, False

    def initial(self, Vs):
        self._prepare()
        pb, ub, vb = self._bc(Vs)
        N = self.N
        psi = np.full(N, np.mean(pb[self.fix_psi]) if self.fix_psi.any() else 0.0)
        psi[self.fix_psi] = pb[self.fix_psi]
        # flat quasi-Fermi levels (E_F = 0 at zero bias): u = (psi+chi)/Vt etc.
        u = np.clip((psi + self.chi) / self.Vt, -700, 30)
        v = np.clip((-psi - self.chi - self.Eg) / self.Vt, -700, 30)
        u[self.fix_c] = ub[self.fix_c]
        v[self.fix_c] = vb[self.fix_c]
        return np.stack([psi, u, v], 1).ravel()

    def equilibrium(self, maxit=300, tol=1e-10):
        """Zero-bias state: nonlinear Poisson with flat Fermi level E_F = 0
        (carriers u = (psi+chi)/Vt, v = -(psi+chi+Eg)/Vt in semiconductors)."""
        self._prepare()
        N = self.N
        pb, ub, vb = self._bc({})
        z = self.initial({})
        psi = z.reshape(N, 3)[:, 0].copy()
        a, b, h, w = self.e_a, self.e_b, self.e_h, self.e_w
        Vt = self.Vt
        Lap = sp.coo_matrix((np.concatenate([-self.e_eps * w / h, -self.e_eps * w / h,
                                             self.e_eps * w / h, self.e_eps * w / h]),
                             (np.concatenate([a, b, a, b]), np.concatenate([b, a, a, b]))),
                            shape=(N, N)).tocsr()
        for it in range(maxit):
            u = np.clip((psi + self.chi) / Vt, -700, 30)
            v = np.clip((-psi - self.chi - self.Eg) / Vt, -700, 30)
            n = self.Nc * np.exp(u)
            p = self.Nv * np.exp(v)
            rho = np.where(self.semi, q * (p - n), 0.0) * self.vol
            F = -(Lap @ psi) + rho
            dF = np.where(self.semi, -q * (p + n) / Vt, 0.0) * self.vol
            A = (-Lap + sp.diags(dF)).tolil()
            fix = np.where(self.fix_psi)[0]
            F[fix] = psi[fix] - pb[fix]
            for k in fix:
                A.rows[k] = [k]
                A.data[k] = [1.0]
            dpsi = spla.spsolve(A.tocsr(), -F)
            dpsi = np.clip(dpsi, -Vt, Vt)
            psi += dpsi
            if np.max(np.abs(dpsi)) < tol:
                break
        u = np.clip((psi + self.chi) / Vt, -700, 30)
        v = np.clip((-psi - self.chi - self.Eg) / Vt, -700, 30)
        u[self.fix_c] = ub[self.fix_c]
        v[self.fix_c] = vb[self.fix_c]
        self._u_eq, self._v_eq = u.copy(), v.copy()
        return np.stack([psi, u, v], 1).ravel()

    def solve(self, Vs, z0=None, steps=None):
        """Solve for contact voltages Vs (dict); ramps from 0 if needed."""
        if not hasattr(self, "e_a"):
            self._prepare()
        z = self.equilibrium() if z0 is None else z0
        if z0 is None:
            z, ok = self.newton(z, {}, maxit=200)
            if not ok:
                raise RuntimeError("equilibrium did not converge")
        start = {k: 0.0 for k in Vs} if z0 is None else getattr(self, "_lastV", {k: 0.0 for k in Vs})
        frac, step = 0.0, 0.25 if steps is None else 1.0 / steps
        zc = z
        while frac < 1.0 - 1e-12:
            f2 = min(1.0, frac + step)
            Vt = {k: start.get(k, 0.0) + f2 * (Vs[k] - start.get(k, 0.0)) for k in Vs}
            zn, ok = self.newton(zc, Vt)
            if ok:
                zc, frac = zn, f2
                step = min(step * 1.5, 0.5)
            else:
                step /= 3
                if step < 1e-4:
                    raise RuntimeError(f"no convergence near {Vt}")
        self._lastV = dict(Vs)
        return zc

    def contact_current(self, z, Vs, name):
        """Total (n + p) current into the device through a contact [A/m]."""
        N = self.N
        Z = z.reshape(N, 3)
        psi, u, v = Z[:, 0], Z[:, 1], Z[:, 2]
        n = self.Nc * np.exp(u)
        p = self.Nv * np.exp(v)
        a, b, h, w = self.e_a, self.e_b, self.e_h, self.e_w
        Vt = self.Vt
        D = (psi[b] - psi[a]) / Vt
        Dn, Dp = D + self.off_n, D + self.off_p
        s = self.e_semi
        Jn = np.where(s, q * self.e_mun * Vt / h * (n[b] * bernoulli(Dn) - n[a] * bernoulli(-Dn)), 0.0) * w
        Jp = np.where(s, q * self.e_mup * Vt / h * (p[a] * bernoulli(Dp) - p[b] * bernoulli(-Dp)), 0.0) * w
        Jt = Jn + Jp                       # current from a to b
        nodes = set(self.contacts[name]["nodes"].tolist())
        ina = np.array([x in nodes for x in a])
        inb = np.array([x in nodes for x in b])
        m1 = ina & ~inb
        m2 = inb & ~ina
        return Jt[m1].sum() - Jt[m2].sum()

    def fields(self, z):
        Z = z.reshape(self.N, 3)
        sh = (self.Nx, self.Ny)
        return dict(psi=Z[:, 0].reshape(sh), n=(self.Nc * np.exp(Z[:, 1])).reshape(sh),
                    p=(self.Nv * np.exp(Z[:, 2])).reshape(sh))


def ofet(L=4e-6, t_ox=50e-9, t_s=40e-9, eps_ox=3.9, mu_p=1e-6, Nx=41, ny_ox=8, ny_s=30,
         W_contact=5.0, HOMO=5.1, T=300.0, eps_s=3.5, Nv=1e26):
    """Bottom-gate, edge-contact p-type OFET (holes)."""
    sx = np.linspace(0, 1, Nx)
    x = L * 0.5 * (1 + np.tanh(2.5 * (2 * sx - 1)) / np.tanh(2.5))   # refined at the contacts
    yo = np.linspace(0, t_ox, ny_ox + 1)
    # geometric grid in the semiconductor, first cell ~0.5 nm at the interface
    h0 = min(0.5e-9, t_s / ny_s)
    from scipy.optimize import brentq
    g = brentq(lambda r: h0 * (r ** ny_s - 1) / (r - 1) - t_s, 1.0 + 1e-9, 2.0) if h0 * ny_s < t_s else 1.0
    ys = t_ox + np.cumsum(h0 * g ** np.arange(ny_s))
    y = np.concatenate([yo, ys])
    d = Device2D(x, y, T=T)
    d.set_region(lambda X, Y: Y <= t_ox + 1e-15, semiconductor=False, epsr=eps_ox)
    d.set_region(lambda X, Y: Y > t_ox + 1e-15, semiconductor=True, epsr=eps_s,
                 chi=HOMO - 2.5, Eg=2.5, Nv=Nv, Nc=Nv, mup=mu_p, mun=1e-12)
    d.add_contact("gate", lambda X, Y: Y < 1e-15, W=W_contact, kind="gate")
    d.add_contact("source", lambda X, Y: (X < 1e-15) & (Y > t_ox + 1e-15), W=W_contact)
    d.add_contact("drain", lambda X, Y: (X > L - 1e-15) & (Y > t_ox + 1e-15), W=W_contact)
    d.carriers = ("p",)
    d.C_i = eps0 * eps_ox / t_ox
    d.L_ch = L
    return d
