"""Case library.

A case is a function  build(params) -> Setup.  Register it with @case.
To make a new problem: copy one of these, change geometry, species, bcs.
"""
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_bvp

from . import bc as B
from . import geometry, mms
from .model import Model, Species

V_T_PER_K = 8.617333262e-5          # k_B / e  [V/K]

CASES = {}


@dataclass
class Setup:
    model: Model
    path: list                                   # continuation targets
    base: dict                                   # starting parameters
    focus_bbox: list = None
    profiles: Callable = None                    # analysis -> list of lines
    extra: Callable = None                       # analysis -> (checks, integrals)
    quantities: Callable = None                  # analysis -> {name: q(U, p)}
    sens_params: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    labels: dict = field(default_factory=dict)


def case(name, defaults, info):
    def register(fn):
        CASES[name] = {"build": fn, "params": defaults, "info": info,
                       "doc": (fn.__doc__ or "").strip()}
        return fn
    return register


def thermal_potential(mV, T):
    """Potential in mV -> units of V_T = k_B T / e."""
    return 1e-3 * mV / (V_T_PER_K * T)


LABELS = {"c+": "cation c+/c0", "c-": "anion c-/c0", "phi": "potential phi/V_T",
          "rho": "charge (c+ - c-)/c0", "|u|": "speed |u|/U", "p": "pressure p a/(eta U)",
          "vorticity": "vorticity", "|E|": "|E| a/V_T"}


# =====================================================================
# 1. Charged cylinder in a cross flow  (the target problem)
# =====================================================================

CYL_DEFAULTS = {
    "delta": 0.1, "Pe": 1.0, "Re": 0.5, "K": 50.0,
    "zeta_mV": 25.0, "T": 298.15, "d_plus": 1.0, "d_minus": 1.0,
    "stern_C": 0.0, "side_slip": 1.0,
    "half_width": 40.0, "upstream": 40.0, "downstream": 80.0,
    "h_wall": 0.0, "h_far": 4.0, "sens": 1.0,
}
CYL_INFO = {
    "delta": "Debye length / cylinder radius",
    "Pe": "Peclet number U a / D0",
    "Re": "Reynolds number rho U a / eta (must be > 0: Stokes paradox)",
    "K": "electric coupling F c0 V_T a / (eta U); EO speed ~ K delta^2",
    "zeta_mV": "cylinder potential vs far field [mV]",
    "T": "temperature [K] (sets V_T)",
    "d_plus": "D+ / D0", "d_minus": "D- / D0",
    "stern_C": "Stern capacitance (0 = potential applied directly at r=1)",
    "side_slip": "1: slip side walls, 0: side walls move with the far-field flow",
    "half_width": "box half width / a", "upstream": "inlet distance / a",
    "downstream": "outlet distance / a",
    "h_wall": "element size at the wall (0 = delta/8)",
    "h_far": "element size far away", "sens": "1: compute adjoint sensitivities",
}


@case("cylinder", CYL_DEFAULTS, CYL_INFO)
def build_cylinder(p):
    """Electrolyte cross-flow past a charged cylinder with resolved EDL.

    Flow comes in at the top (z = +upstream) with u = (0, -1).
    """
    phi_w = thermal_potential(p["zeta_mV"], p["T"])
    h_wall = p["h_wall"] or p["delta"] / 8
    mesh = geometry.cylinder_in_box(
        half_width=p["half_width"], upstream=p["upstream"],
        downstream=p["downstream"], h_wall=h_wall, h_far=p["h_far"],
        layer=3 * p["delta"], grow=20.0, wake_size=0.25)

    params = {k: p[k] for k in ("delta", "Pe", "Re", "K")}
    params["phi_wall"] = phi_w
    params["stern_C"] = p["stern_C"]

    far_u = B.Dirichlet((0.0, -1.0))
    if p["stern_C"] > 0:
        wall_phi = B.Robin("stern_C", "phi_wall")
    else:
        wall_phi = B.Dirichlet("phi_wall")
    side_u = B.Component(0, 0.0) if p["side_slip"] else far_u
    bcs = {
        "inlet": {"u": far_u, "c": B.Dirichlet("bulk"), "phi": B.Dirichlet(0.0)},
        "side": {"u": side_u},
        "outlet": {"u": B.Pressure(0.0), "c": B.OUTFLOW},
        "wall": {"u": B.NOSLIP, "c": B.BLOCKING, "phi": wall_phi},
    }
    species = [Species("c+", 1, p["d_plus"]), Species("c-", -1, p["d_minus"])]
    model = Model(mesh, species, params, bcs, flow="navier_stokes",
                  advection="conservative")

    base = dict(params, phi_wall=0.0, K=0.0)
    path = [{"phi_wall": phi_w}, {"K": params["K"]}]
    notes = [
        f"phi_wall = {p['zeta_mV']} mV / V_T = {phi_w:.4f} at T = {p['T']} K",
        "No-slip and the wall potential are both applied at r = 1.",
        "Finite Re and a finite box replace the unbounded Stokes problem.",
        "Electric wall force = minus the force on the diffuse charge "
        "(outer Maxwell stress neglected).",
    ]
    d = p["delta"]
    return Setup(model, path, base, focus_bbox=[-3, 3, -3, 3],
                 profiles=_cylinder_profiles, extra=_cylinder_extra,
                 quantities=_cylinder_quantities if p["sens"] else None,
                 sens_params=["delta", "K", "phi_wall", "Pe", "Re"],
                 notes=notes, labels=LABELS)


def _cylinder_profiles(an):
    d = float(an.params["delta"])
    lines = []
    for name, ang in [("upstream", 0), ("side", 90), ("downstream", 180)]:
        t = np.radians(ang)
        e = np.array([np.sin(t), np.cos(t)])
        for tag, r1, sp in [("edl", 1 + 12 * d, "geometric"), ("far", 8.0, "geometric")]:
            line = an.line_profile(e * 1.0, e * r1, n=300, spacing=sp,
                                   label=f"radial, {name} (theta={ang} deg), r to {r1:g}",
                                   coord_name="r - 1")
            line["name"] = f"{tag}_{name}"
            lines.append(line)
    w = an.wall_profile("wall")
    w["name"] = "wall"
    lines.append(w)
    axis = an.line_profile([0, 1.0], [0, 30.0], n=300, spacing="geometric",
                           label="upstream axis z>1", coord_name="z - 1")
    axis["name"] = "axis_up"
    lines.append(axis)
    wake = an.line_profile([0, -1.0], [0, -30.0], n=300, spacing="geometric",
                           label="wake axis z<-1", coord_name="-z - 1")
    wake["name"] = "axis_wake"
    lines.append(wake)
    return lines


def _cylinder_extra(an):
    force = an.wall_force("wall")
    force["drag"] = -force["total"][1]          # flow is -z, drag pushes -z
    force["lift"] = force["total"][0]
    ef = an.electric_flux()
    q = an.diffuse_charge()
    charge = {"wall_induced": ef["wall"], "diffuse": q,
              "outer_boundaries": sum(v for k, v in ef.items() if k != "wall"),
              "sum": sum(ef.values()) + q}
    integrals = {"force": force, "charge": charge}
    checks = [
        _check("wall ion flux c+", abs(an.flux["c+"]["wall"]), 1e-8,
               "blocking wall: should be ~0"),
        _check("Gauss balance", abs(charge["sum"]), 1e-6 * max(1, abs(q)),
               "sum of electric fluxes + diffuse charge"),
        _check("lift (symmetry)", abs(force["lift"]), 1e-6 * abs(force["drag"]),
               "symmetric problem: lift should be ~0", warn_only=True),
    ]
    return checks, integrals


def _cylinder_quantities(an):
    m = an.m
    asm = an.asm
    nodes = an.bnodes["wall"]
    w = jnp.asarray(an.bweight["wall"])
    uz = jnp.asarray(m.dof("uz", nodes))
    ph = jnp.asarray(m.dof("phi", nodes))

    def drag(U, p):
        R = asm.raw_residual(U, p)
        hydro_z = -jnp.sum(R[uz] * w)
        return -(hydro_z - _jax_body_force(m, U, p)[1])

    def wall_charge(U, p):
        R = asm.raw_residual(U, p)
        return jnp.sum(R[ph] * w)

    return {"drag": drag, "wall_charge": wall_charge}


def _jax_body_force(m, U, p):
    geo = m.geo
    tri = m.mesh.tri
    f = U[:m.p_offset].reshape(m.F, m.n)
    rho = sum(s.z * f[i] for i, s in enumerate(m.species))
    rho_q = jnp.einsum("qa,ea->eq", m.N, rho[tri])
    gphi = jnp.einsum("eqak,ea->eqk", geo["G"], f[m.i_phi][tri])
    force = -p["K"] * rho_q[..., None] * gphi
    return jnp.sum(force * geo["wd"][..., None], axis=(0, 1))


# =====================================================================
# 2. Gouy-Chapman double layer (validation, analytic)
# =====================================================================

@case("gouy_chapman",
      {"delta": 0.1, "zeta_mV": 25.0, "T": 298.15, "length": 2.0, "h_wall": 0.0,
       "stern_C": 0.0},
      {"delta": "Debye length / length scale", "zeta_mV": "wall potential [mV]",
       "T": "temperature [K]", "length": "strip length", "h_wall": "wall size (0 = delta/20)",
       "stern_C": "Stern capacitance (0 = off)"})
def build_gouy_chapman(p):
    """Planar equilibrium double layer; compared to the exact solution."""
    phi_w = thermal_potential(p["zeta_mV"], p["T"])
    d = p["delta"]
    h_wall = p["h_wall"] or d / 20
    mesh = geometry.rectangle(0, p["length"], 0, 0.2, h=0.05, wall_names=["left"],
                              h_wall=h_wall, layer=0.2 * d, grow=5 * d)
    params = {"delta": d, "Pe": 1.0, "Re": 0.0, "K": 0.0, "phi_wall": phi_w,
              "stern_C": p["stern_C"]}
    wall = B.Robin("stern_C", "phi_wall") if p["stern_C"] > 0 else B.Dirichlet("phi_wall")
    bcs = {"left": {"phi": wall},
           "right": {"phi": B.Dirichlet(0.0), "c": B.Dirichlet("bulk")}}
    model = Model(mesh, [Species("c+", 1), Species("c-", -1)], params, bcs, flow="none")

    def extra(an):
        x = an.m.mesh.X[:, 0]
        phi0 = _stern_phi0(phi_w, d, p["stern_C"])
        exact = 4 * np.arctanh(np.tanh(phi0 / 4) * np.exp(-x / d))
        f = an.fields
        e_phi = float(np.abs(f[2] - exact).max())
        e_c = float(np.abs(f[0] - np.exp(-exact)).max())
        sigma_exact = 2 * d * np.sinh(phi0 / 2) * 0.2
        sigma = an.electric_flux()["left"]
        checks = [_check("max |phi - exact|", e_phi, 1e-4, "Gouy-Chapman"),
                  _check("max |c+ - exact|", e_c, 1e-4, "Boltzmann"),
                  _check("surface charge rel. err", abs(sigma / sigma_exact - 1), 1e-4,
                         "Grahame equation")]
        return checks, {"charge": {"wall_induced": sigma, "exact": sigma_exact,
                                   "diffuse": an.diffuse_charge()}}

    def profiles(an):
        line = an.line_profile([0, 0.1], [p["length"], 0.1], n=400, spacing="geometric",
                               label="across the double layer", coord_name="x")
        line["name"] = "x"
        x = np.asarray(line["coord"])
        phi0 = _stern_phi0(phi_w, d, p["stern_C"])
        line["values"]["phi_exact"] = (4 * np.arctanh(np.tanh(phi0 / 4) * np.exp(-x / d))).tolist()
        return [line]

    return Setup(model, [{"phi_wall": phi_w}], dict(params, phi_wall=0.0),
                 focus_bbox=[0, 6 * d, 0, 0.2], profiles=profiles, extra=extra,
                 notes=[f"phi_wall = {phi_w:.4f}"], labels=LABELS)


def _stern_phi0(phi_m, d, C):
    """Diffuse-layer potential with a Stern layer (solves the Robin law)."""
    if C <= 0:
        return phi_m
    lo, hi = 0.0, phi_m
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        # delta^2 dphi/dn = 2 delta sinh(phi0/2)  must equal  C (phi_m - phi0)
        if 2 * d * np.sinh(mid / 2) > C * (phi_m - mid):
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# =====================================================================
# 3. Electroosmotic slit (validation, nonlinear Poisson-Boltzmann)
# =====================================================================

@case("eo_slit",
      {"delta": 0.1, "zeta_mV": 25.0, "T": 298.15, "Ex": 0.5, "K": 2.0, "h_wall": 0.0},
      {"delta": "Debye length / half width", "zeta_mV": "wall potential [mV]",
       "T": "temperature [K]", "Ex": "applied field along the slit",
       "K": "electric coupling", "h_wall": "wall size (0 = delta/10)"})
def build_eo_slit(p):
    """Fully developed electroosmotic flow between walls y = +-1."""
    d, Ex = p["delta"], p["Ex"]
    zeta = thermal_potential(p["zeta_mV"], p["T"])
    yt, pt = _slit_pb(d, zeta)
    psi = lambda y: jnp.interp(y, yt, pt)
    mesh = geometry.rectangle(0, 0.5, -1, 1, h=0.1, wall_names=["bottom", "top"],
                              h_wall=p["h_wall"] or d / 10, layer=0.5 * d, grow=0.6)
    params = {"delta": d, "Pe": 1.0, "Re": 0.0, "K": p["K"], "Ex": Ex}
    phi_bc = lambda x, q: psi(x[:, 1]) - q["Ex"] * x[:, 0]
    end = {"phi": B.Dirichlet(phi_bc),
           "u": B.Pressure(lambda x, q: 2 * q["K"] * (jnp.cosh(psi(x[:, 1])) - 1)),
           "c": {"c+": B.Dirichlet(lambda x, q: jnp.exp(-psi(x[:, 1]))),
                 "c-": B.Dirichlet(lambda x, q: jnp.exp(psi(x[:, 1])))}}
    wall = {"u": B.NOSLIP, "phi": B.Dirichlet(phi_bc)}
    bcs = {"left": end, "right": end, "bottom": wall, "top": wall}
    model = Model(mesh, [Species("c+", 1), Species("c-", -1)], params, bcs, flow="stokes")

    def exact_u(y):
        return 2 * p["K"] * Ex * d ** 2 * (np.interp(y, yt, pt) - zeta)

    def extra(an):
        y = an.m.mesh.X[:, 1]
        e = float(np.abs(an.fields[an.m.i_uy] - exact_u(y)).max())
        scale = float(np.abs(exact_u(np.array([0.0]))).max())
        return [_check("max |u_x - exact| / u_max", e / scale, 1e-3,
                       "nonlinear PB electroosmosis")], {}

    def profiles(an):
        line = an.line_profile([0.25, -1], [0.25, 1], n=400, label="across slit",
                               coord_name="y")
        line["name"] = "y"
        line["values"]["ux_exact"] = exact_u(np.asarray(line["coord"]) - 1).tolist()
        return [line]

    return Setup(model, [{"Ex": Ex}], dict(params, Ex=0.0),
                 profiles=profiles, extra=extra, labels=LABELS,
                 notes=["Helmholtz-Smoluchowski limit: u -> -2 K delta^2 zeta Ex"])


def _slit_pb(d, zeta):
    y = np.linspace(-1, 1, 4001)
    guess = np.vstack([zeta * np.cosh(y / d) / np.cosh(1 / d), 0 * y])
    sol = solve_bvp(lambda x, s: np.vstack([s[1], np.sinh(s[0]) / d ** 2]),
                    lambda a, b: np.array([a[0] - zeta, b[0] - zeta]),
                    y, guess, tol=1e-12, max_nodes=10 ** 6)
    yt = np.linspace(-1, 1, 20001)
    return yt, sol.sol(yt)[0]


# =====================================================================
# 4. Manufactured solution (all terms, all BC types)
# =====================================================================

@case("mms", {"h": 0.1, "delta": 0.3, "Pe": 2.0, "Re": 1.0, "K": 1.0, "stern": 2.0,
              "advection": 0.0, "supg": 0.0},
      {"h": "element size", "delta": "Debye length", "Pe": "Peclet", "Re": "Reynolds",
       "K": "coupling", "stern": "Robin coefficient on the top side",
       "advection": "0 conservative, 1 non-conservative", "supg": "1: SUPG on"})
def build_mms(p):
    """Exact smooth fields; sources from jax autodiff; mixed BC types."""
    mesh = geometry.rectangle(0, 1, 0, 1, h=p["h"])
    params = {k: p[k] for k in ("delta", "Pe", "Re", "K")}
    ex = {k: B.Dirichlet(mms.field_fn(k)) for k in ("c+", "c-")}
    vel = B.Dirichlet(mms.velocity_fn())
    phi = B.Dirichlet(mms.field_fn("phi"))
    flux_right = {n: B.Flux(mms.flux_normal_fn(n, z, d, [1, 0])) for n, z, d in mms.SPECIES}
    bcs = {
        "left": {"c": ex, "phi": phi, "u": vel},
        "bottom": {"c": ex, "phi": phi, "u": vel},
        "right": {"c": flux_right, "phi": phi, "u": vel},
        "top": {"c": ex, "u": vel, "phi": B.Robin(p["stern"], mms.robin_ref_fn(p["stern"], [0, 1]))},
    }
    species = [Species(n, z, d) for n, z, d in mms.SPECIES]
    model = Model(mesh, species, params, bcs, flow="navier_stokes",
                  advection="nonconservative" if p["advection"] else "conservative",
                  supg=bool(p["supg"]), sources=mms.sources(),
                  pressure_pin=((0.0, 0.0), float(mms.exact(jnp.zeros(2))["p"])))

    def extra(an):
        exact = mms.nodal_exact(an.m.mesh.X)
        names = ["c+", "c-", "phi", "uy", "uz"]
        integ = {}
        checks = []
        for k in names:
            i = an.m.field_names.index(k)
            e = float(np.abs(an.fields[i] - exact[k]).max())
            integ[k] = e
            checks.append(_check(f"max err {k}", e, 1e-3 if p["h"] <= 0.1 else 1e-2,
                                 "vs manufactured solution", warn_only=True))
        pe = float(np.abs(an.p_vert - np.asarray(mms.field_fn("p")(
            jnp.asarray(an.m.mesh.X[an.m.mesh.vertices]), {}))).max())
        integ["p"] = pe
        return checks, {"errors": integ}

    return Setup(model, [], params, extra=extra, labels=LABELS,
                 notes=["Sources were derived with jax.jacfwd / jax.hessian."])


# =====================================================================
# 5. Uniformly charged sphere, Poisson only (axisymmetric validation)
# =====================================================================

SPHERE_LABELS = {
    "phi": "potential phi", "phi_exact": "exact potential phi_exact",
    "phi_err": "error phi - phi_exact", "|E|": "field magnitude |E|",
    "Ey": "radial field E_r", "Ez": "axial field E_z",
    "rho_f": "fixed charge density q (source)", "rho": "mobile ion charge (none)",
}


@case("charged_sphere",
      {"R": 1.0, "R_out": 10.0, "q": 1.0, "h_sphere": 0.0, "h_far": 0.5,
       "outer_bc": 1.0},
      {"R": "sphere radius", "R_out": "domain radius (far boundary)",
       "q": "uniform charge density: -lap phi = q inside, 0 outside",
       "h_sphere": "element size at the sphere surface (0 = R/20)",
       "h_far": "element size far away",
       "outer_bc": "1: Robin far field dphi/dn = -phi/R_out (exact for a monopole), "
                   "0: Dirichlet with the exact value"})
def build_charged_sphere(p):
    """Uniformly charged sphere in an unbounded uniform dielectric (Poisson only).

    Axisymmetric (r, z) model, no ions and no flow:
      -lap phi = q for rho < R, 0 outside, with rho = sqrt(r^2 + z^2), delta = 1.
    Exact solution (phi -> 0 at infinity):
      phi = q R^2/2 - q rho^2/6 inside,  phi = q R^3 / (3 rho) outside.
    The mesh follows the sphere surface, so the source jumps exactly on
    element edges.  The axis r = 0 is a symmetry line (natural BC).
    """
    R, R_out, q = p["R"], p["R_out"], p["q"]
    if not 0 < R < R_out:
        raise ValueError("charged_sphere needs 0 < R < R_out")
    h_sphere = p["h_sphere"] or R / 20
    mesh = geometry.sphere_axisym(R=R, R_out=R_out, h_sphere=h_sphere,
                                  h_far=max(p["h_far"], h_sphere))

    def charge(xq, prm):
        """q inside the sphere, tested on the element centroid.

        xq is (..., Q, 2): the points of one element (or of all elements).
        The mesh conforms to rho = R, so every element is wholly inside or
        outside and the source is exact up to the interface."""
        centre = jnp.mean(xq, axis=-2, keepdims=True)
        inside = jnp.sum(centre ** 2, axis=-1) < R ** 2
        return jnp.where(inside, 1.0, 0.0) * prm["q"] * jnp.ones(xq.shape[:-1])

    # the centroid test must agree with gmsh's regions on every element
    centre = mesh.X[mesh.tri].mean(axis=1)
    inside = np.sum(centre ** 2, axis=1) < R ** 2
    if not np.array_equal(np.flatnonzero(inside), np.sort(mesh.regions["sphere"])):
        raise RuntimeError("charged_sphere: centroid test disagrees with the mesh regions")

    def exact_phi(X, qq=q):
        rho = np.hypot(X[:, 0], X[:, 1])
        far = np.maximum(rho, R)                    # only used where rho >= R
        return np.where(rho < R, qq * R ** 2 / 2 - qq * rho ** 2 / 6, qq * R ** 3 / (3 * far))

    def exact_E(X, qq=q):
        rho = np.hypot(X[:, 0], X[:, 1])
        far = np.maximum(rho, R)
        return np.where(rho < R, qq * rho / 3, qq * R ** 3 / (3 * far ** 2))

    if p["outer_bc"]:
        outer = B.Robin(1.0 / R_out, 0.0)          # delta^2 dphi/dn = -phi / R_out
    else:
        outer = B.Dirichlet(lambda x, prm: prm["q"] * R ** 3 / (3 * jnp.hypot(x[:, 0], x[:, 1])))
    bcs = {"outer": {"phi": outer}}                 # axis: natural = symmetry
    params = {"delta": 1.0, "q": q}

    def extra_fields(an):
        X = an.m.mesh.X
        phi = an.fields[an.m.i_phi]
        ex = exact_phi(X, float(an.params["q"]))
        return {"rho_f": an.m.rho_n * an.source_nodal("phi"),
                "phi_exact": ex, "phi_err": phi - ex}

    model = Model(mesh, [], params, bcs, flow="none", coord="axisym",
                  sources={"phi": charge}, extra_fields=extra_fields)

    def extra(an):
        qq = float(an.params["q"])
        X = an.m.mesh.X
        phi = an.fields[an.m.i_phi]
        err = float(np.abs(phi - exact_phi(X, qq)).max())
        rel = err / float(np.abs(phi).max())
        two_pi = 2 * np.pi
        flux_out = -two_pi * an.electric_flux()["outer"]      # outward flux of E (delta = 1)
        flux_axis = -two_pi * an.electric_flux()["axis"]
        q_enc = two_pi * float(an.integrate_source("phi"))
        q_exact = 4 * np.pi * qq * R ** 3 / 3
        phi_min, phi_max = float(phi.min()), float(phi.max())
        checks = [
            _check("max |phi - exact| / max phi", rel, 1e-4,
                   "vs exact charged-sphere potential (all nodes)"),
            _check("Gauss: outer flux vs enclosed charge", abs(flux_out / q_enc - 1), 1e-8,
                   "3D totals: 2 pi * per-radian integrals; flux of E through 'outer' "
                   "= 2 pi int -dphi/dn r ds, charge = 2 pi int q r dr dz"),
            _check("Gauss: outer flux vs 4/3 pi R^3 q", abs(flux_out / q_exact - 1), 1e-4,
                   "includes the P2 approximation of the sphere volume"),
            {"name": "min phi", "value": phi_min, "limit": q * R ** 3 / (3 * R_out),
             "status": "info", "note": "limit column = exact q R^3/(3 R_out) at the outer boundary"},
            {"name": "max phi", "value": phi_max, "limit": q * R ** 2 / 2,
             "status": "info", "note": "limit column = exact q R^2/2 at the centre"},
        ]
        integrals = {
            "charge": {"enclosed_3d": q_enc, "enclosed_exact_3d": q_exact,
                       "outer_flux_3d": flux_out, "axis_flux_3d": flux_axis,
                       "enclosed_per_radian": q_enc / two_pi,
                       "outer_flux_per_radian": flux_out / two_pi},
            "errors": {"phi_max_abs": err, "phi_max_rel": rel,
                       "phi_min": phi_min, "phi_max": phi_max},
        }
        return checks, integrals

    def profiles(an):
        qq = float(an.params["q"])
        f = an.nodal_fields()
        names = ["phi", "|E|", "phi_err"]
        lines = []
        for tag, end, coord, label in [
                ("axis", (0.0, R_out), "z", "along the axis r = 0, out to the domain edge"),
                ("radial", (R_out, 0.0), "r", "along r at z = 0, out to the domain edge")]:
            line = an.line_profile([0.0, 0.0], end, n=400, names=names,
                                   label=label, coord_name=coord)
            line["name"] = tag
            pts = np.asarray(line["points"])
            line["values"]["phi_exact"] = exact_phi(pts, qq).tolist()
            line["values"]["|E|_exact"] = exact_E(pts, qq).tolist()
            lines.append(line)
        return lines

    notes = [
        "Axisymmetric (r, z): mesh x = r >= 0, y = z; the UI may mirror about r = 0.",
        "Integrals in summary.json are per radian (r dr dz, r ds) unless named *_3d; "
        "*_3d = 2 pi * per radian.",
        f"outer BC: {'Robin dphi/dn = -phi/R_out' if p['outer_bc'] else 'Dirichlet exact'}; "
        f"h_sphere = {h_sphere:g}, h_far = {max(p['h_far'], h_sphere):g}.",
        "Ey / Ez hold E_r / E_z.",
    ]
    return Setup(model, [], params, focus_bbox=[0, 3 * R, -3 * R, 3 * R],
                 profiles=profiles, extra=extra, notes=notes, labels=SPHERE_LABELS)


# =====================================================================
# helpers
# =====================================================================

def _check(name, value, limit, note, warn_only=False):
    ok = value <= limit
    status = "pass" if ok else ("warn" if warn_only else "fail")
    return {"name": name, "value": float(value), "limit": float(limit),
            "status": status, "note": note}
