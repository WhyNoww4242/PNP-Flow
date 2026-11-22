"""Post-processing: fields, fluxes, forces, profiles, checks, sensitivities.

Boundary fluxes use the "reaction" method: the element residual summed
over the nodes of a boundary equals the boundary integral of the flux.
This is conservative and more accurate than differentiating fields.

Axisymmetric models (model.coord == "axisym"): every integral here
(fluxes, charges, forces, integrate) is per radian, i.e. uses
r dr dz and r ds.  Multiply by 2 pi for the full 3D body.
"""
import numpy as np
import jax
import jax.numpy as jnp
from matplotlib.tri import Triangulation

from . import fem
from .solver import linear_solve


class Analysis:
    def __init__(self, model, assembler, U, params):
        self.m = model
        self.asm = assembler
        self.U = jnp.asarray(U)
        self.params = {k: jnp.asarray(v) for k, v in params.items()}
        self.fields, self.p_vert = model.split(np.asarray(U))
        self._boundary_weights()
        self._compile()
        self._recover_gradients()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _boundary_weights(self):
        """Boundary node lists, and which boundaries are Dirichlet per field."""
        m = self.m
        mesh = m.mesh
        self.bnodes = {name: mesh.boundary_nodes(name) for name in mesh.boundaries}
        self.dir_weight = {}
        for f in range(m.F):
            count = np.zeros(mesh.n_nodes)
            dirichlet = []
            for name, b in mesh.boundaries.items():
                mids = b["edges"][:, 2]
                if np.all(m.is_dirichlet[m.dof(f, mids)]):
                    dirichlet.append(name)
                    count[self.bnodes[name]] += 1
            for name in dirichlet:
                self.dir_weight[(f, name)] = 1.0 / count[self.bnodes[name]]
        # used by drag sensitivity
        self.bweight = {name: self.dir_weight.get((self.m.i_uz, name),
                                                  np.ones(len(nodes)))
                        for name, nodes in self.bnodes.items()}

    def _compile(self):
        m = self.m
        kernels = {}
        for name in m.edge:
            fn = lambda u, wd, n, xq, p, name=name: m.edge_flux_totals(name, u, wd, n, xq, p)
            kernels[name] = jax.jit(jax.vmap(fn, in_axes=(0, 0, 0, 0, None)))
        self._edge_totals = kernels
        self._raw = self.asm.raw_residual

    # ------------------------------------------------------------------
    # Boundary integrals: one function for species, phi and momentum
    # ------------------------------------------------------------------
    def boundary_integrals(self):
        """table[field][boundary] = outward boundary flux of that field.

        species : total N.n            phi : delta^2 dphi/dn
        uy, uz  : traction (grad u - p I) n exerted ON the fluid
        Natural boundaries: the prescribed flux, integrated exactly.
        Dirichlet boundaries: the reaction of the full residual (volume +
        natural edges) at the boundary's nodes.  The species weak form is
        written with -grad(w).N, so its reaction has the opposite sign.
        """
        if hasattr(self, "_table"):
            return self._table
        m = self.m
        R = np.asarray(self._raw(self.U, self.params))
        table = {name: {} for name in m.field_names}
        for b in self.bnodes:
            e = m.edge[b]
            loc = self.U[jnp.asarray(e["emap"])]
            totals = np.asarray(self._edge_totals[b](
                loc, e["wd"], e["n"], e["xq"], self.params)).sum(axis=0)
            for f, fname in enumerate(m.field_names):
                w = self.dir_weight.get((f, b))
                if w is None:
                    table[fname][b] = float(totals[f])
                    continue
                ids = m.dof(f, self.bnodes[b])
                reaction = float(np.sum(R[ids] * w))
                if f < m.S:
                    reaction = -reaction
                    if m.advection != "conservative":
                        reaction += float(totals[m.F + f])
                table[fname][b] = reaction
        self._table = table
        return table

    def _recover_gradients(self):
        """Smooth nodal gradients: average of element gradients at nodes."""
        mesh = self.m.mesh
        G = np.asarray(fem.nodal_gradients(mesh.X, mesh.tri))  # (E, 6, 6, 2)
        count = np.bincount(mesh.tri.ravel(), minlength=mesh.n_nodes)
        self.grad = {}
        for f, name in enumerate(self.m.field_names):
            loc = self.fields[f][mesh.tri]                         # (E, 6)
            g = np.einsum("enak,ea->enk", G, loc)                  # (E, 6, 2)
            total = np.zeros((mesh.n_nodes, 2))
            np.add.at(total, mesh.tri.ravel(), g.reshape(-1, 2))
            self.grad[name] = total / count[:, None]

    # ------------------------------------------------------------------
    # Nodal fields
    # ------------------------------------------------------------------
    def pressure_nodal(self):
        """P1 pressure interpolated to all P2 nodes."""
        mesh = self.m.mesh
        p = np.zeros(mesh.n_nodes)
        pv = self.p_vert[mesh.tri_p1]                  # (E, 3)
        p[mesh.tri[:, :3]] = pv
        p[mesh.tri[:, 3]] = 0.5 * (pv[:, 0] + pv[:, 1])
        p[mesh.tri[:, 4]] = 0.5 * (pv[:, 1] + pv[:, 2])
        p[mesh.tri[:, 5]] = 0.5 * (pv[:, 2] + pv[:, 0])
        return p

    def nodal_fields(self):
        m = self.m
        f = self.fields
        out = {}
        for i, sp in enumerate(m.species):
            out[sp.name] = f[i]
        rho = sum(sp.z * f[i] for i, sp in enumerate(m.species)) if m.S else 0 * f[0]
        out["rho"] = rho
        out["phi"] = f[m.i_phi]
        gphi = self.grad["phi"]
        out["Ey"] = -gphi[:, 0]
        out["Ez"] = -gphi[:, 1]
        out["|E|"] = np.hypot(gphi[:, 0], gphi[:, 1])
        if m.flow != "none":
            uy, uz = f[m.i_uy], f[m.i_uz]
            out["uy"], out["uz"] = uy, uz
            out["|u|"] = np.hypot(uy, uz)
            out["vorticity"] = self.grad["uz"][:, 0] - self.grad["uy"][:, 1]
        if m.flow in ("stokes", "navier_stokes"):
            out["p"] = self.pressure_nodal()
            out["div_u"] = self.grad["uy"][:, 0] + self.grad["uz"][:, 1]
        current = np.zeros((m.n, 2))
        for i, sp in enumerate(m.species):
            N = self.species_flux_nodal(i)
            out[f"N_{sp.name}_y"] = N[:, 0]
            out[f"N_{sp.name}_z"] = N[:, 1]
            current += sp.z * N
        if m.S:
            out["j_y"], out["j_z"] = current[:, 0], current[:, 1]
        if m.extra_fields is not None:
            for k, v in m.extra_fields(self).items():
                out[k] = np.asarray(v, dtype=float)
        return out

    def source_nodal(self, key):
        """Nodal values of a volume source (e.g. "phi") for plotting.

        The source is evaluated per element (at the element's 6 nodes, so a
        source that tests the element centroid sees the same element), then
        averaged over the elements that share a node.  Nodes on a material
        interface therefore get the mean of both sides.
        """
        mesh = self.m.mesh
        fn = self.m.sources.get(key)
        if fn is None:
            return np.zeros(mesh.n_nodes)
        Xe = jnp.asarray(mesh.X[mesh.tri])                       # (E, 6, 2)
        vals = np.broadcast_to(np.asarray(fn(Xe, self.params), dtype=float),
                               mesh.tri.shape)
        total = np.zeros(mesh.n_nodes)
        np.add.at(total, mesh.tri.ravel(), vals.ravel())
        count = np.bincount(mesh.tri.ravel(), minlength=mesh.n_nodes)
        return total / count

    def species_flux_nodal(self, i):
        m = self.m
        sp = m.species[i]
        c = self.fields[i]
        u = np.stack([self.fields[m.i_uy], self.fields[m.i_uz]], 1)
        D = sp.d / float(self.params["Pe"])
        return c[:, None] * u - D * (self.grad[sp.name] + sp.z * c[:, None] * self.grad["phi"])

    # ------------------------------------------------------------------
    # Integrals
    # ------------------------------------------------------------------
    def boundary_sum(self, values, name):
        nodes = self.bnodes[name]
        return float(np.sum(np.asarray(values)[nodes]))

    def species_fluxes(self):
        """Outward total flux of every species through every boundary."""
        table = self.boundary_integrals()
        out = {}
        for sp in self.m.species:
            out[sp.name] = dict(table[sp.name])
            out[sp.name]["net"] = sum(table[sp.name].values())
            src = float(self.integrate_source(sp.name))
            if src:
                out[sp.name]["volume_source"] = src
        return out

    def integrate_source(self, key):
        fn = self.m.sources.get(key)
        if fn is None:
            return 0.0
        geo = self.m.geo
        return jnp.sum(fn(geo["xq"], self.params) * geo["wd"])

    def electric_flux(self):
        """delta^2 * dphi/dn (outward normal) on each boundary.

        On a wall this is minus the induced surface charge (per rho_n).
        Gauss: sum over boundaries + diffuse charge = 0.
        """
        return dict(self.boundary_integrals()["phi"])

    def diffuse_charge(self):
        """Total mobile charge  int rho / rho_n dA."""
        m = self.m
        if not m.S:
            return 0.0
        rho = sum(sp.z * self.fields[i] for i, sp in enumerate(m.species))
        return float(self.integrate(rho)) / m.rho_n

    def integrate(self, nodal):
        geo = self.m.geo
        vals = jnp.einsum("qa,ea->eq", self.m.N, jnp.asarray(nodal)[self.m.mesh.tri])
        return jnp.sum(vals * geo["wd"])

    def body_force(self):
        """Total electric force on the fluid: int -K rho grad(phi) dA."""
        m = self.m
        if not m.S:
            return np.zeros(2)
        geo = m.geo
        tri = m.mesh.tri
        rho = sum(sp.z * self.fields[i] for i, sp in enumerate(m.species))
        rho_q = jnp.einsum("qa,ea->eq", m.N, jnp.asarray(rho)[tri])
        gphi = jnp.einsum("eqak,ea->eqk", geo["G"], jnp.asarray(self.fields[m.i_phi])[tri])
        f = -self.params["K"] * rho_q[..., None] * gphi
        return np.asarray(jnp.sum(f * geo["wd"][..., None], axis=(0, 1)))

    def wall_force(self, name):
        """Forces on a no-slip body boundary, per unit length.

        hydro    : -(traction on the fluid), from the momentum reaction
        electric : force on the body's surface charge, -int f dA (Maxwell
                   stress at the outer boundary neglected)
        """
        m = self.m
        table = self.boundary_integrals()
        hydro = -np.array([table["uy"][name], table["uz"][name]])
        electric = -self.body_force() if m.S else np.zeros(2)
        total = hydro + electric
        return {"hydro": hydro.tolist(), "electric": electric.tolist(),
                "total": total.tolist()}

    def div_u_l2(self):
        m = self.m
        geo = m.geo
        tri = m.mesh.tri
        gy = jnp.einsum("eqak,ea->eqk", geo["G"], jnp.asarray(self.fields[m.i_uy])[tri])
        gz = jnp.einsum("eqak,ea->eqk", geo["G"], jnp.asarray(self.fields[m.i_uz])[tri])
        div = gy[..., 0] + gz[..., 1]
        return float(jnp.sqrt(jnp.sum(div ** 2 * geo["wd"])))

    # ------------------------------------------------------------------
    # Point evaluation and profiles
    # ------------------------------------------------------------------
    def _finder(self):
        if not hasattr(self, "_tri"):
            mesh = self.m.mesh
            self._tri = Triangulation(mesh.X[:, 0], mesh.X[:, 1], mesh.tri[:, :3])
            self._find = self._tri.get_trifinder()
        return self._find

    def evaluate(self, points, values):
        """P2 interpolation of nodal `values` at points (m, 2). NaN outside."""
        mesh = self.m.mesh
        points = np.asarray(points, dtype=float)
        e = self._finder()(points[:, 0], points[:, 1])
        out = np.full(len(points), np.nan)
        ok = e >= 0
        corners = mesh.X[mesh.tri[e[ok], :3]]
        ref = _reference_coords(corners, points[ok])
        shapes = np.asarray(jax.vmap(fem.p2_shape)(jnp.asarray(ref)))
        out[ok] = np.sum(shapes * np.asarray(values)[mesh.tri[e[ok]]], axis=1)
        return out

    def line_profile(self, start, end, n=400, names=None, spacing="linear",
                     label="", coord_name="s"):
        start, end = np.asarray(start, float), np.asarray(end, float)
        t = np.linspace(0, 1, n)
        if spacing == "geometric":
            t = np.expm1(np.linspace(0, np.log(1 + 200.0), n)) / 200.0
        pts = start[None] + t[:, None] * (end - start)[None]
        s = t * np.linalg.norm(end - start)
        fields = self.nodal_fields()
        names = names or list(fields)
        values = {k: _clean(self.evaluate(pts, fields[k])) for k in names if k in fields}
        return {"label": label, "coord_name": coord_name, "coord": s.tolist(),
                "points": pts.tolist(), "values": values}

    def node_normals(self, name):
        """Outward (from fluid) unit normal at each node of a boundary."""
        mesh = self.m.mesh
        b = mesh.boundaries[name]
        edges = b["edges"]
        t = mesh.X[edges[:, 1]] - mesh.X[edges[:, 0]]
        n = np.stack([t[:, 1], -t[:, 0]], 1) * b["sign"][:, None]
        acc = np.zeros((mesh.n_nodes, 2))
        for k in range(3):
            np.add.at(acc, edges[:, k], n)
        nodes = self.bnodes[name]
        acc = acc[nodes]
        return nodes, acc / np.linalg.norm(acc, axis=1, keepdims=True)

    def wall_profile(self, name, center=(0.0, 0.0)):
        """Values along a closed wall, ordered by angle from +z (upstream)."""
        nodes, n = self.node_normals(name)
        x = self.m.mesh.X[nodes] - np.asarray(center)
        theta = np.degrees(np.arctan2(x[:, 0], x[:, 1])) % 360.0
        order = np.argsort(theta)
        nodes, n, theta = nodes[order], n[order], theta[order]
        fields = self.nodal_fields()
        d2 = float(self.params["delta"]) ** 2
        values = {}
        for k in ["phi", "rho", "p", "vorticity"] + [s.name for s in self.m.species]:
            if k in fields:
                values[k] = _clean(fields[k][nodes])
        values["sigma"] = _clean(d2 * np.sum(self.grad["phi"][nodes] * n, axis=1))
        if "vorticity" in fields:
            values["shear"] = values.pop("vorticity")
        return {"label": f"along {name} (angle from upstream +z)",
                "coord_name": "theta_deg", "coord": theta.tolist(), "values": values}

    # ------------------------------------------------------------------
    # Sensitivities with the adjoint method
    # ------------------------------------------------------------------
    def sensitivities(self, quantities, names):
        """d q / d param for each q(U, params) -> scalar (jax function).

        Uses one transposed linear solve per quantity:
          K^T lam = dq/dU,   dq/dp = partial q/partial p - lam . dR/dp
        """
        U, params = self.U, self.params
        A = self.asm.jacobian(U, params)
        At = A.T.tocsr()
        dR_dp = {}
        for name in names:
            tangent = {k: jnp.zeros_like(v) for k, v in params.items()}
            tangent[name] = jnp.ones_like(params[name])
            _, d = jax.jvp(lambda p: self.asm.residual(U, p), (params,), (tangent,))
            dR_dp[name] = np.asarray(d)
        out = {}
        for qname, q in quantities.items():
            dq_dU, dq_dp = jax.grad(q, argnums=(0, 1))(U, params)
            lam = linear_solve(At, np.asarray(dq_dU))
            out[qname] = {n: float(dq_dp[n] - lam @ dR_dp[n]) for n in names}
        return out


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _split_jax(m, U):
    return U[:m.p_offset].reshape(m.F, m.n), U[m.p_offset:]


def _reference_coords(corners, points):
    """Reference (xi, eta) of points in straight triangles."""
    a = corners[:, 0]
    e1 = corners[:, 1] - a
    e2 = corners[:, 2] - a
    d = points - a
    det = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]
    xi = (d[:, 0] * e2[:, 1] - d[:, 1] * e2[:, 0]) / det
    eta = (e1[:, 0] * d[:, 1] - e1[:, 1] * d[:, 0]) / det
    return np.stack([xi, eta], 1)


def _clean(a):
    """List with NaN -> None (valid JSON)."""
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else float(v) for v in a]
