"""The PNP + flow model: unknowns, weak form, boundary conditions.

Nondimensional equations (see docs/THEORY.md):

  species i :  div N_i = s_i
               N_i = c_i u - (d_i/Pe) (grad c_i + z_i c_i grad phi)
  Poisson   :  -delta^2 lap phi = rho / rho_n + s_phi
               rho = sum_i z_i c_i,   rho_n = sum_i z_i^2 c_i,bulk
  momentum  :  Re (u.grad) u = -grad p + lap u - K rho grad phi + s_u
  mass      :  div u = 0

Unknown vector layout (field major):
  [c_1 ... c_S, phi, uy, uz]  on all P2 nodes, then p on P1 vertices.

Coordinates: coord="planar" uses (y, z) per unit depth.  coord="axisym"
uses (r, z) stored as mesh (x, y), r >= 0, and weights every volume and
edge integral by r (per radian).  That is exact for the Poisson and
Nernst-Planck weak forms; Stokes / Navier-Stokes would need hoop terms,
so they are rejected with axisym.
"""
from dataclasses import dataclass

import numpy as np
import jax
import jax.numpy as jnp

from . import bc as B
from . import fem


@dataclass
class Species:
    name: str
    z: float
    d: float = 1.0
    bulk: float = 1.0


FLOW_TYPES = ("none", "prescribed", "stokes", "navier_stokes")
COORD_TYPES = fem.COORDS
SOLVED_FLOW = ("stokes", "navier_stokes")


class Model:
    """Holds the mesh, physics choices and boundary conditions.

    bcs maps a boundary name to a dict with optional keys:
      "c"   : one BC for every species, or {species_name: BC}
      "phi" : BC for the potential
      "u"   : BC for the velocity (B.NOSLIP, B.FREE, B.Dirichlet, B.Component)
    Later boundaries override earlier ones at shared corner nodes.

    coord        : "planar" or "axisym" (see module docstring)
    species      : may be empty (Poisson only; use flow="none")
    extra_fields : optional f(analysis) -> {name: nodal array}, added to
                   Analysis.nodal_fields (exact solutions, fixed charge...)
    """

    def __init__(self, mesh, species, params, bcs, flow="navier_stokes",
                 advection="conservative", supg=False, sources=None,
                 prescribed_u=None, pressure_pin=None, coord="planar",
                 extra_fields=None):
        assert flow in FLOW_TYPES
        if coord not in COORD_TYPES:
            raise ValueError(f"coord must be one of {COORD_TYPES}, got {coord!r}")
        if coord == "axisym":
            if flow in SOLVED_FLOW:
                raise ValueError(
                    f"coord='axisym' does not support flow='{flow}': the "
                    "axisymmetric Stokes/Navier-Stokes hoop terms are not "
                    "implemented. Use flow='none' or 'prescribed'.")
            r_min = float(np.min(mesh.X[:, 0]))
            scale = float(np.ptp(mesh.X)) or 1.0
            if r_min < -1e-9 * scale:
                raise ValueError(f"coord='axisym' needs r = x >= 0 on the mesh "
                                 f"(min x = {r_min:.3g})")
        self.coord = coord
        mesh.coord = coord
        self.extra_fields = extra_fields
        self.mesh = mesh
        self.species = list(species)
        self.params = {k: float(v) for k, v in params.items()}
        self.bcs = bcs
        self.flow = flow
        self.advection = advection
        self.supg = supg
        self.sources = sources or {}
        self.prescribed_u = prescribed_u
        self.pressure_pin = pressure_pin
        self.rho_n = sum(s.z ** 2 * s.bulk for s in self.species) or 1.0
        self._layout()
        self._geometry()
        self._dirichlet_structure()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _layout(self):
        m = self.mesh
        self.S = len(self.species)
        self.field_names = [s.name for s in self.species] + ["phi", "uy", "uz"]
        self.F = self.S + 3
        self.i_phi = self.S
        self.i_uy = self.S + 1
        self.i_uz = self.S + 2
        self.n = m.n_nodes
        self.p_offset = self.F * self.n
        self.ndof = self.p_offset + m.n_vertices

        blocks = [f * self.n + m.tri for f in range(self.F)]
        blocks.append(self.p_offset + m.tri_p1)
        self.emap = np.concatenate(blocks, axis=1)          # (E, 6F+3)

    def dof(self, field, nodes):
        """Global dof ids of a P2 field at given nodes."""
        f = self.field_names.index(field) if isinstance(field, str) else field
        return f * self.n + np.asarray(nodes)

    def _geometry(self):
        m = self.mesh
        self.geo, self.N, self.L = fem.element_geometry(m.X, m.tri, self.coord)
        self.edge = {}
        for name, b in m.boundaries.items():
            data, M = fem.edge_geometry(m.X, b["edges"], jnp.asarray(b["sign"]),
                                        self.coord)
            data["M"] = M
            data["emap"] = np.concatenate(
                [f * self.n + b["edges"] for f in range(self.F)], axis=1)
            self.edge[name] = data

    # ------------------------------------------------------------------
    # Small helpers
    # ------------------------------------------------------------------
    def split(self, U):
        """Return (P2 fields (F, n), pressure (nv,))."""
        return U[:self.p_offset].reshape(self.F, self.n), U[self.p_offset:]

    def initial_guess(self, params=None):
        params = params or self.params
        U = np.zeros(self.ndof)
        for i, s in enumerate(self.species):
            U[i * self.n:(i + 1) * self.n] = s.bulk
        if self.prescribed_u is not None:
            vel = np.asarray(B.resolve_vector(self.prescribed_u, self.mesh.X, params))
            U[self.dof("uy", np.arange(self.n))] = vel[:, 0]
            U[self.dof("uz", np.arange(self.n))] = vel[:, 1]
        return self.apply_dirichlet(jnp.asarray(U), params)

    def species_bc(self, spec, species):
        """The BC of one species on one boundary."""
        if isinstance(spec, dict):
            return spec.get(species.name, B.BLOCKING)
        return B.BLOCKING if spec is None else spec

    # ------------------------------------------------------------------
    # Dirichlet conditions
    # ------------------------------------------------------------------
    def _dirichlet_structure(self):
        """List of (dof ids, node coords, value function) in apply order."""
        X = self.mesh.X
        items = []

        def add(field, nodes, getter):
            items.append((self.dof(field, nodes), X[nodes], getter))

        # Flow that is not solved: fix all velocity and pressure dofs.
        if self.flow in ("none", "prescribed"):
            spec = self.prescribed_u or (0.0, 0.0)
            all_nodes = np.arange(self.n)
            add("uy", all_nodes, _vector_part(spec, 0))
            add("uz", all_nodes, _vector_part(spec, 1))
            vids = self.p_offset + np.arange(self.mesh.n_vertices)
            items.append((vids, X[self.mesh.vertices], lambda x, p: 0.0 * x[:, 0]))

        for name, spec in self.bcs.items():
            nodes = self.mesh.boundary_nodes(name)
            for s in self.species:
                c_bc = self.species_bc(spec.get("c"), s)
                if isinstance(c_bc, B.Dirichlet):
                    value = s.bulk if c_bc.value == "bulk" else c_bc.value
                    add(s.name, nodes, _scalar(value))
            phi_bc = spec.get("phi")
            if isinstance(phi_bc, B.Dirichlet):
                add("phi", nodes, _scalar(phi_bc.value))
            if self.flow in ("stokes", "navier_stokes"):
                self._velocity_dirichlet(spec.get("u", B.NOSLIP), nodes, add)

        if self.pressure_pin is not None and self.flow in ("stokes", "navier_stokes"):
            point, value = self.pressure_pin
            dist = np.linalg.norm(X[self.mesh.vertices] - np.asarray(point), axis=1)
            v = int(np.argmin(dist))
            items.append((np.array([self.p_offset + v]),
                          X[self.mesh.vertices[[v]]], _scalar(value)))
        self.dirichlet_items = items
        ids = np.concatenate([it[0] for it in items]) if items else np.zeros(0, int)
        self.dirichlet_ids = np.unique(ids)
        self.is_dirichlet = np.zeros(self.ndof, dtype=bool)
        self.is_dirichlet[self.dirichlet_ids] = True

    def _velocity_dirichlet(self, u_bc, nodes, add):
        if u_bc == B.FREE or isinstance(u_bc, B.Pressure):
            return
        if u_bc == B.NOSLIP:
            u_bc = B.Dirichlet((0.0, 0.0))
        if isinstance(u_bc, B.Component):
            field = "uy" if u_bc.index == 0 else "uz"
            add(field, nodes, _scalar(u_bc.value))
            return
        add("uy", nodes, _vector_part(u_bc.value, 0))
        add("uz", nodes, _vector_part(u_bc.value, 1))

    def dirichlet_values(self, params):
        """(ids, values) in apply order; values are jax arrays."""
        return [(ids, getter(jnp.asarray(x), params))
                for ids, x, getter in self.dirichlet_items]

    def apply_dirichlet(self, U, params):
        for ids, vals in self.dirichlet_values(params):
            U = U.at[ids].set(vals)
        return U

    # ------------------------------------------------------------------
    # Weak form on one element
    # ------------------------------------------------------------------
    def element_residual(self, u_loc, G, wd, xq, h, params):
        """Residual of one element.

        u_loc : (6F + 3,) local unknowns
        G     : (Q, 6, 2) shape gradients, wd : (Q,) weights * |det J|
        Returns (6F + 3,).
        """
        F, S = self.F, self.S
        N, L = self.N, self.L
        Ue = u_loc[:6 * F].reshape(F, 6)
        pe = u_loc[6 * F:]

        val = Ue @ N.T                                # (F, Q)
        grad = jnp.einsum("qak,fa->fqk", G, Ue)       # (F, Q, 2)
        p = L @ pe                                    # (Q,)

        phi_g = grad[self.i_phi]
        u = jnp.stack([val[self.i_uy], val[self.i_uz]], axis=-1)   # (Q, 2)
        rho = sum(self.species[i].z * val[i] for i in range(S)) if S else 0.0 * p

        prm = params
        res = []
        for i in range(S):
            res.append(self._species_terms(i, val[i], grad[i], phi_g, u, rho,
                                           G, wd, xq, h, prm))
        res.append(self._poisson_terms(grad[self.i_phi], rho, G, wd, xq, prm))
        r_u, r_p = self._flow_terms(u, grad, p, rho, phi_g, G, wd, xq, prm)
        res.extend(r_u)
        return jnp.concatenate(res + [r_p])

    def _source(self, key, xq, params, shape=()):
        fn = self.sources.get(key)
        if fn is None:
            return jnp.zeros(xq.shape[:-1] + shape)
        return fn(xq, params)

    def _species_terms(self, i, c, gc, gphi, u, rho, G, wd, xq, h, prm):
        sp = self.species[i]
        N = self.N
        D = sp.d / prm["Pe"]
        s = self._source(sp.name, xq, prm)

        J = -D * (gc + sp.z * c[:, None] * gphi)                # (Q, 2)
        if self.advection == "conservative":
            flux = c[:, None] * u + J
            r = -jnp.einsum("qak,qk->qa", G, flux)
        else:
            r = N * jnp.sum(u * gc, axis=1)[:, None]
            r = r - jnp.einsum("qak,qk->qa", G, J)
        r = r - N * s[:, None]

        if self.supg:
            r = r + self._supg(sp, c, gc, gphi, u, rho, s, G, xq, h, prm)
        return jnp.sum(r * wd[:, None], axis=0)

    def _supg(self, sp, c, gc, gphi, u, rho, s, G, xq, h, prm):
        """Streamline upwind term using an approximate strong residual."""
        D = sp.d / prm["Pe"]
        v = u - D * sp.z * gphi                        # transport velocity
        speed = jnp.sqrt(jnp.sum(v * v, axis=1) + 1e-30)
        lap_phi = -(rho / self.rho_n + self._source("phi", xq, prm)) / prm["delta"] ** 2
        strong = jnp.sum(u * gc, axis=1)
        strong = strong - D * sp.z * (jnp.sum(gc * gphi, axis=1) + c * lap_phi)
        strong = strong - s
        h_eff = 0.5 * h                               # P2 elements
        pe_cell = speed * h_eff / (2.0 * D)
        xi = (pe_cell / 3.0) / jnp.sqrt(1.0 + (pe_cell / 3.0) ** 2)
        tau = h_eff / (2.0 * speed) * xi
        test = jnp.einsum("qak,qk->qa", G, v)
        return (tau * strong)[:, None] * test

    def _poisson_terms(self, gphi, rho, G, wd, xq, prm):
        delta2 = prm["delta"] ** 2
        s = self._source("phi", xq, prm)
        r = delta2 * jnp.einsum("qak,qk->qa", G, gphi)
        r = r - self.N * (rho / self.rho_n + s)[:, None]
        return jnp.sum(r * wd[:, None], axis=0)

    def _flow_terms(self, u, grad, p, rho, gphi, G, wd, xq, prm):
        """Momentum in Laplacian form:  (grad u : grad v) - p div v.

        Its natural condition is (grad u) n - p n = t ("do nothing"), which
        lets fully developed flow leave the domain.  On a no-slip wall with
        div u = 0 the traction equals the symmetric-stress traction.
        """
        N, L = self.N, self.L
        if self.flow not in SOLVED_FLOW:
            # Velocity and pressure are fixed (Dirichlet rows): nothing to
            # assemble, and Re / K need not be parameters.
            zero = 0.0 * jnp.sum(wd)
            return [zero * N[0], zero * N[0]], zero * L[0]
        Gu = jnp.stack([grad[self.i_uy], grad[self.i_uz]])      # (2, Q, 2): du_i/dx_j
        conv = jnp.einsum("qj,iqj->qi", u, Gu)                   # (u.grad) u
        force = -prm["K"] * rho[:, None] * gphi                  # electric body force
        s_u = self._source("u", xq, prm, shape=(2,))
        if self.flow != "navier_stokes":
            conv = 0.0 * conv
        div_u = Gu[0, :, 0] + Gu[1, :, 1]

        r_u = []
        for i in range(2):
            body = prm["Re"] * conv[:, i] - force[:, i] - s_u[:, i]
            r = N * body[:, None]
            r = r + jnp.einsum("qaj,qj->qa", G, Gu[i])
            r = r - G[:, :, i] * p[:, None]
            r_u.append(jnp.sum(r * wd[:, None], axis=0))
        s_p = self._source("p", xq, prm)
        r_p = -jnp.sum(L * ((div_u - s_p) * wd)[:, None], axis=0)
        return r_u, r_p

    # ------------------------------------------------------------------
    # Weak form on boundary edges (natural conditions)
    # ------------------------------------------------------------------
    def edge_residual(self, name, u_loc, wd, n, xq, params):
        """Boundary terms of one edge.  u_loc is (3F,)."""
        spec = self.bcs.get(name, {})
        M = self.edge[name]["M"]                      # (Qb, 3)
        Ub = u_loc.reshape(self.F, 3)
        val = Ub @ M.T                                # (F, Qb)
        u = jnp.stack([val[self.i_uy], val[self.i_uz]], axis=-1)
        un = jnp.sum(u * n, axis=1)

        res = []
        for i, sp in enumerate(self.species):
            total = self._species_total(spec, sp, val[i], un, xq, params)
            if self.advection != "conservative":
                total = total - val[i] * un
            res.append(total)
        res.append(self._poisson_edge(spec, val[self.i_phi], xq, params))
        traction = self._traction_edge(spec, n, xq, params)
        res.append(traction[:, 0])
        res.append(traction[:, 1])
        out = jnp.stack(res)                          # (F, Qb)
        return jnp.einsum("fq,qa,q->fa", out, M, wd).reshape(-1)

    def edge_flux_totals(self, name, u_loc, wd, n, xq, params):
        """Integrals over one edge of the natural-boundary fluxes.

        Returns (F,): species outward N.n, then delta^2 dphi/dn, then the
        traction t = (grad u - p I) n on the fluid.
        Only meaningful where that field is not a Dirichlet field.
        """
        spec = self.bcs.get(name, {})
        M = self.edge[name]["M"]
        val = u_loc.reshape(self.F, 3) @ M.T
        u = jnp.stack([val[self.i_uy], val[self.i_uz]], axis=-1)
        un = jnp.sum(u * n, axis=1)
        out = [self._species_total(spec, sp, val[i], un, xq, params)
               for i, sp in enumerate(self.species)]
        out.append(-self._poisson_edge(spec, val[self.i_phi], xq, params))
        t = -self._traction_edge(spec, n, xq, params)
        out += [t[:, 0], t[:, 1]]
        out += [val[i] * un for i in range(self.S)]   # convective c u.n
        return jnp.stack(out) @ wd

    def _species_total(self, spec, sp, c, un, xq, prm):
        """Outward total flux N.n prescribed by a natural species BC."""
        bc = self.species_bc(spec.get("c"), sp)
        if bc == B.OUTFLOW:
            return c * un                             # diffusive part is zero
        if isinstance(bc, B.Flux):
            return B.resolve(bc.value, xq, prm)
        return 0.0 * c                                # blocking (or Dirichlet)

    def _traction_edge(self, spec, n, xq, prm):
        """Integrand of -w . t, with t = -p0 n on a Pressure boundary."""
        bc = spec.get("u")
        if isinstance(bc, B.Pressure):
            p0 = B.resolve(bc.p0, xq, prm)
            return p0[:, None] * n
        return 0.0 * n

    def _poisson_edge(self, spec, phi, xq, prm):
        """Integrand of -w * delta^2 dphi/dn."""
        bc = spec.get("phi")
        if isinstance(bc, B.Robin):
            coef = B.resolve(bc.coef, xq, prm)
            ref = B.resolve(bc.ref, xq, prm)
            return -coef * (ref - phi)
        if isinstance(bc, B.Charge):
            return -B.resolve(bc.sigma, xq, prm)
        return 0.0 * phi


def _scalar(spec):
    return lambda x, p: B.resolve(spec, x, p)


def _vector_part(spec, k):
    if callable(spec):
        return lambda x, p: jnp.asarray(spec(x, p))[:, k]
    return _scalar(spec[k])
