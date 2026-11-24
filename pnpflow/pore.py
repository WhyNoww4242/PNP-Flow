"""Charging of an axisymmetric pore fed by a cylindrical reservoir (SDL).

Direct numerical simulation of the full nonlinear PNP equations for the pore
charging problem: at t = 0 the pore wall is stepped from 0 to phi_w, ions
enter from the SDL, and the electric double layers form along the pore.

Paper nondimensionalisation (reported in the outputs):
  rho = (c+ - c-)/c0, S = (c+ + c-)/c0, Phi = e phi/(k_B T),
  tau = D t / l_p^2, R = r / a_p, Z = z / l_p, alpha(Z) = alpha*/a_p,
  kappa = a_p / lambda_D,  mu = rho + 2 Phi  (electrochemical potential of charge).
Solver nondimensionalisation (isotropic, needed by the FE mesh): lengths / a_p,
time t~ = D t / a_p^2 (Pe = 1), so  tau = t~ (a_p / l_p)^2 = t~ / L^2  with
L = l_p / a_p, and delta = lambda_D / a_p = 1 / kappa.  Poisson
-delta^2 lap phi = rho / 2 is the paper's Eq. (4c) in these units.
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from . import bc as B
from . import geometry
from .cases import Setup, _check, case, thermal_potential
from .model import Model, Species

_trapz = getattr(np, "trapezoid", None) or np.trapz

K_B = 1.380649e-23        # J/K
E_CHARGE = 1.602176634e-19
N_A = 6.02214076e23

# output times (tau = D t / l_p^2); those <= tau_end are written
TAU_OUT = (1e-4, 1e-3, 3e-3, 0.01, 0.03, 0.05, 0.08, 0.13, 0.2, 0.3, 0.5, 0.75,
           1.0, 1.5, 2.0, 3.0, 5.0, 10.0)
TAU_EARLY = 0.13          # the paper's early time

DEFAULTS = {
    "alpha_in": 1.0, "alpha_out": 1.0, "L_nm": 1000.0, "a_ref_nm": 10.0,
    "phi_w_mV": 10.0, "c0_mM": 0.94, "eps": 7.1e-10, "T": 298.15, "D": 1.34e-9,
    "phi_end_mV": 10.0, "sdl_radius": 0.0, "sdl_length": 0.0,
    "tau_end": 1.0, "tau_mid": 0.5,
    "h_wall": 0.0, "h_entr": 0.0, "h_far": 1.0, "h_radius_n": 0.0,
    "dt0": 0.01, "rtol": 2e-3, "equilibrium_check": 1.0,
    "dns_times": 0.0, "dns_init": 0.0, "err_mixed": 0.0,
}
INFO = {
    "alpha_in": "pore radius at the mouth Z = 0, in units of a_ref (alpha(0))",
    "alpha_out": "pore radius at the end Z = 1, in units of a_ref (alpha(1)); "
                 "alpha is linear in Z (narrow 1/1, wide 2/2, converging 2/1, diverging 1/2)",
    "L_nm": "pore length l_p [nm]",
    "a_ref_nm": "reference pore radius a_p [nm] (fixed between pores)",
    "phi_w_mV": "wall potential step vs the reservoir [mV] (no Stern layer)",
    "c0_mM": "reservoir concentration c0 [mM]",
    "eps": "permittivity of water [F/m]",
    "T": "temperature [K] (sets V_T and lambda_D)",
    "D": "ion diffusivity [m^2/s] (dimensional time only)",
    "phi_end_mV": "potential of the closed pore end Z = 1 [mV]. Set it equal to phi_w_mV "
                  "for an end that is part of the electrode (default 10 mV, same as the wall). "
                  "Ions cannot pass the end",
    "sdl_radius": "SDL radius / a_ref (0 = 2 max alpha)",
    "sdl_length": "SDL length / a_ref (0 = L/2 + 1.5 max alpha; the 1.5 max alpha "
                  "is extra room for the access region at the mouth)",
    "tau_end": "final time tau = D t / l_p^2",
    "tau_mid": "the 'intermediate' time shown next to tau = 0.13 in the summary figure",
    "h_wall": "element size in the pore / at the pore wall (0 = min(delta, alpha)/10, so >= 8 elements cross the EDL)",
    "h_entr": "element size at the mouth corner (0 = h_wall/2)",
    "h_far": "element size far into the SDL",
    "h_radius_n": "0 = off. n > 0: element size in the pore at most alpha(Z)/n (n elements across "
                  "the local radius), mouth sizes at most alpha_in/n; for tips thinner than the EDL",
    "dt0": "first time step in units of a_ref^2 / D (~ Debye time when delta ~ 1)",
    "rtol": "time-step control: max change vs linear extrapolation per step",
    "equilibrium_check": "1: also solve the steady (equilibrium) state and compare the "
                         "radial profile at Z = 0.5 with the 1D cylindrical PB solution",
    "dns_times": "1: save only at the DNS folder times t = 0.005 k s, k = 0..15 "
                 "(tau = D t / l_p^2), plus tau_end",
    "dns_init": "1: start from the DNS initial field (c = c0, phi = phi_w in the pore, "
                "0 in the SDL) as given, without the consistent-phi solve; phi follows "
                "from Poisson from the first time step on",
    "err_mixed": "1: time-step error measured relative to 1 + |U| (needed at high wall "
                 "potentials, where c- ~ exp(Phi_w) >> 1 at the wall); 0: absolute (default)",
}

DNS_DT_S = 0.005          # DNS output interval [s]
DNS_N_OUT = 15            # DNS folders 0.005 ... 0.075 s
# rounded constants used by pawel.py (reported only, never used by the solver)
PAWEL_CONST = {"kB": 1.38e-23, "e": 1.6022e-19, "NA": 6.022e23}


def debye_length(c0_mM, eps, T):
    """lambda_D = sqrt(eps k_B T / (2 e^2 c0)) for a 1:1 salt [m]; c0 in mM = mol/m^3."""
    return float(np.sqrt(eps * K_B * T / (2 * E_CHARGE ** 2 * c0_mM * N_A)))


def cylinder_pb(alpha, delta, phi_w, r):
    """Nonlinear PB in an infinite cylinder: delta^2 (1/r)(r phi')' = sinh phi,
    phi'(0) = 0, phi(alpha) = phi_w.  Returns phi(r) by shooting on phi(0)."""
    def shoot(phi0, r_eval=None):
        r0 = 1e-6 * alpha
        y0 = [phi0 + np.sinh(phi0) * r0 ** 2 / (4 * delta ** 2),
              np.sinh(phi0) * r0 / (2 * delta ** 2)]

        def rhs(rr, y):
            return [y[1], np.sinh(y[0]) / delta ** 2 - y[1] / rr]
        return solve_ivp(rhs, (r0, alpha), y0, t_eval=r_eval, rtol=1e-12, atol=1e-14)

    phi0 = brentq(lambda f0: shoot(f0).y[0, -1] - phi_w, 0.0 if phi_w > 0 else phi_w,
                  phi_w if phi_w > 0 else 0.0, xtol=1e-14)
    rr = np.clip(np.asarray(r, float), 1e-6 * alpha, alpha)
    order = np.argsort(rr)
    out = np.empty_like(rr)
    out[order] = shoot(phi0, rr[order]).y[0]
    return out


@case("pore_charging", DEFAULTS, INFO)
def build_pore_charging(p):
    """Charging of an axisymmetric pore from a cylindrical reservoir (time-dependent PNP).

    Geometry (r, z), lengths / a_ref: SDL cylinder r <= a_s, -l_s <= z <= 0, then
    the pore 0 <= z <= L = l_p / a_ref with wall r = alpha(z / L), alpha linear.
    BCs: SDL far end z = -l_s: c+- = c0, phi = 0.  SDL side and face: uncharged
    (dphi/dn = 0), no ion flux.  Pore wall: phi = phi_w (step at t = 0, no Stern),
    no ion flux.  Pore end: no ion flux, phi = phi_end (phi_end_mV; the wall value for an electrode end).
    Initial state: c+- = c0 everywhere; phi follows from Poisson at the first step.
    Outputs at each tau: cross-section-averaged mu = rho + 2 Phi, Phi, rho along Z
    (SDL included at Z < 0) and radial Phi, rho, mu at Z = 0.5 vs R / alpha(0.5).
    """
    lam = debye_length(p["c0_mM"], p["eps"], p["T"])
    a_ref = p["a_ref_nm"] * 1e-9
    delta = lam / a_ref
    L = p["L_nm"] / p["a_ref_nm"]
    ai, ao = p["alpha_in"], p["alpha_out"]
    amax, amin = max(ai, ao), min(ai, ao)
    a_s = p["sdl_radius"] or 2 * amax
    l_s = p["sdl_length"] or 0.5 * L + 1.5 * amax
    phi_w = thermal_potential(p["phi_w_mV"], p["T"])
    h_wall = p["h_wall"] or min(delta, amin) / 10
    h_entr = p["h_entr"] or h_wall / 2
    h_far = max(p["h_far"], h_wall)
    mesh = geometry.pore_axisym(L=L, alpha_in=ai, alpha_out=ao, sdl_radius=a_s,
                                sdl_length=l_s, h_wall=h_wall, h_entr=h_entr,
                                h_far=h_far, wall_layer=min(amax, 3 * delta),
                                n_radius=p.get("h_radius_n", 0.0))

    species = [Species("c+", 1.0), Species("c-", -1.0)]
    params = {"delta": delta, "Pe": 1.0, "phi_w": phi_w}
    params["phi_end"] = thermal_potential(p["phi_end_mV"], p["T"])
    end = {"phi": B.Dirichlet("phi_end")}
    bcs = {
        "axis": {},
        "sdl_side": {},
        "sdl_face": {},
        "sdl_far": {"c": B.Dirichlet("bulk"), "phi": B.Dirichlet(0.0)},
        "pore_end": end,
        "pore_wall": {"phi": B.Dirichlet("phi_w")},     # last: owns the mouth corner
    }
    model = Model(mesh, species, params, bcs, flow="none", coord="axisym")

    def alpha(Z):
        return ai + (ao - ai) * np.asarray(Z, float)

    # charge in the pore at every accepted time step: Q = int_pore rho r dr dz (per
    # radian) = w . rho with w_i = int_pore N_i r dA.  dQ/dtau gives the charge flux
    # through the mouth (paper Jright), exactly conservative (walls and end are blocking).
    w_pore = np.zeros(model.n)
    pore_el = np.asarray(model.mesh.regions.get("pore", []), dtype=np.int64)
    if pore_el.size:
        contrib = np.asarray(model.geo["wd"])[pore_el] @ np.asarray(model.N)   # (Ep, 6)
        np.add.at(w_pore, np.asarray(model.mesh.tri)[pore_el], contrib)
    n_nodes = model.n
    charge_steps = {"t": [], "Q": []}

    def on_accept(t, U):
        U = np.asarray(U)
        rho = U[:n_nodes] - U[n_nodes:2 * n_nodes]
        charge_steps["t"].append(float(t))
        charge_steps["Q"].append(float(w_pore @ rho))

    t_per_tau = L ** 2                                  # t~ = tau L^2
    tau_per_s = p["D"] / (p["L_nm"] * 1e-9) ** 2
    if p["dns_times"]:
        taus = sorted({k * DNS_DT_S * tau_per_s for k in range(1, DNS_N_OUT + 1)}
                      | {p["tau_end"]})
    else:
        taus = sorted({t for t in TAU_OUT if t <= p["tau_end"] * (1 + 1e-9)}
                      | {TAU_EARLY, p["tau_mid"], p["tau_end"]})
    taus = [t for t in taus if 0 < t <= p["tau_end"] * (1 + 1e-9)]
    Z_mid = 0.5
    xr = np.linspace(0.0, 0.999, 160)                   # R / alpha at Z = 0.5
    gl_x, gl_w = np.polynomial.legendre.leggauss(16)
    gl_x, gl_w = 0.5 * (gl_x + 1), 0.5 * gl_w           # on [0, 1]
    Z_axial = np.unique(np.concatenate([
        np.linspace(-l_s / L, 0.0, 121), np.linspace(0.0, 0.03, 31),
        np.linspace(0.03, 1.0, 292)]))

    def fields_of(an, U):
        f, _ = model.split(np.asarray(U))
        rho = f[0] - f[1]
        phi = f[model.i_phi]
        return {"phi": phi, "rho": rho, "mu": rho + 2 * phi, "S": f[0] + f[1]}

    def axial(an, fl):
        z = Z_axial * L
        R_loc = np.where(Z_axial < 0, a_s, alpha(np.maximum(Z_axial, 0)))
        r = R_loc[:, None] * gl_x[None, :]
        pts = np.stack([r.ravel(), np.repeat(z, len(gl_x))], axis=1)
        w = gl_w[None, :] * r                            # r dr weight
        vals = {}
        for k in ("mu", "phi", "rho"):
            v = an.evaluate(pts, fl[k]).reshape(r.shape)
            vals[k] = (np.sum(w * v, axis=1) / np.sum(w, axis=1)).tolist()
        axis_pts = np.stack([0 * z, z], 1)
        for k in ("mu", "phi", "rho"):                    # centreline (paper Psi_m, rho_m)
            vals[f"{k}_axis"] = an.evaluate(axis_pts, fl[k]).tolist()
        return vals

    def radial(an, fl):
        a_mid = float(alpha(Z_mid))
        pts = np.stack([xr * a_mid, np.full_like(xr, Z_mid * L)], axis=1)
        return {k: an.evaluate(pts, fl[k]).tolist() for k in ("phi", "rho", "mu")}

    def time_profiles(an, snapshots):
        """Lines at every output time + time series of pore-averaged quantities."""
        lines, series = [], {"tau": [], "mu_pore_avg": [], "Q_pore_per_rad": [],
                             "mu_Z05": []}
        inside = Z_axial >= 0
        Zp = Z_axial[inside]
        area = 0.5 * alpha(Zp) ** 2                      # int_0^alpha r dr
        for t, U in snapshots:
            tau = t / t_per_tau
            fl = fields_of(an, U)
            ax = axial(an, fl)
            rd = radial(an, fl)
            tag = f"{tau:.4g}"
            meta = {"tau": tau, "t_solver": t,
                    "t_s": tau * (p["L_nm"] * 1e-9) ** 2 / p["D"]}
            lines.append({"name": f"axial_tau_{tag}", "coord_name": "Z",
                          "label": f"cross-section average along the pore, tau = {tag}",
                          "coord": Z_axial.tolist(), "values": ax, "time": meta})
            lines.append({"name": f"radial_Z0.5_tau_{tag}", "coord_name": "R/alpha",
                          "label": f"radial profile at Z = 0.5, tau = {tag}",
                          "coord": xr.tolist(), "values": rd, "time": meta})
            mu = np.asarray(ax["mu"])[inside]
            rho = np.asarray(ax["rho"])[inside]
            series["tau"].append(tau)
            series["mu_pore_avg"].append(float(_trapz(mu * area, Zp)
                                               / _trapz(area, Zp)))
            series["Q_pore_per_rad"].append(float(_trapz(rho * area, Zp) * L))
            series["mu_Z05"].append(float(np.interp(0.5, Zp, mu)))
        out = {"charging": series}
        if len(charge_steps["t"]) > 2:
            tau_s = np.asarray(charge_steps["t"]) / t_per_tau
            Q_s = np.asarray(charge_steps["Q"])
            keep = np.concatenate([[True], np.diff(tau_s) > 0])
            tau_s, Q_s = tau_s[keep], Q_s[keep]
            # J = (2 / (L alpha_in^2)) dQ/dtau: mouth charge flux in D c0 / l_p, averaged
            # over the mouth area (Q per radian in a_ref units; t~ = tau L^2)
            J = 2.0 / (L * ai ** 2) * np.gradient(Q_s, tau_s)
            out["mouth"] = {"tau": tau_s.tolist(), "Q_pore_per_rad": Q_s.tolist(),
                            "J_right": J.tolist(),
                            "note": "every accepted step; J_right = 2/(L alpha_in^2) dQ/dtau, "
                                    "units D c0 / l_p, positive into the pore"}
        return lines, out

    def equilibrium_lines(an):
        """Steady state: radial profile at Z = 0.5 vs the cylinder PB solution."""
        fl = fields_of(an, an.U)
        rd = radial(an, fl)
        a_mid = float(alpha(Z_mid))
        ex = cylinder_pb(a_mid, delta, phi_w, xr * a_mid)
        return [{"name": "radial_Z0.5_equilibrium", "coord_name": "R/alpha",
                 "label": "equilibrium radial profile at Z = 0.5",
                 "coord": xr.tolist(),
                 "values": {"phi": rd["phi"], "phi_exact": ex.tolist(),
                            "rho": rd["rho"], "rho_exact": (-2 * np.sinh(ex)).tolist()}}]

    def extra(an):
        lines = equilibrium_lines(an)
        v = lines[0]["values"]
        phi, ex = np.asarray(v["phi"]), np.asarray(v["phi_exact"])
        err = float(np.nanmax(np.abs(phi - ex)) / abs(phi_w))
        fl_eq = fields_of(an, an.U)
        mu_eq, phi_eq = np.asarray(fl_eq["mu"]), np.asarray(fl_eq["phi"])
        slope = abs(ao - ai) / L
        # Boltzmann (zero flux): c+- = exp(-+Phi), so rho = -2 sinh Phi and
        # mu = 2 Phi - 2 sinh Phi at every point, in any geometry.  The
        # linearised (Debye-Hueckel, I0 Bessel) limit has rho = -2 Phi, mu = 0.
        mu_boltz = 2 * phi_eq - 2 * np.sinh(phi_eq)
        mu_wall = float(abs(2 * phi_w - 2 * np.sinh(phi_w)))
        mu_max = float(np.max(np.abs(mu_eq)))
        scale = 2 * abs(phi_w)                           # charge scale |rho| ~ 2 Phi_w
        checks = [
            _check("equilibrium Phi(R, Z=0.5) vs cylinder PB", err, 1e-3,
                   f"max |Phi - Phi_PB| / Phi_w; exact only for a straight pore "
                   f"(wall slope here {slope:.2g})", warn_only=slope > 0),
            _check("Boltzmann comparison: max |mu| vs wall value (equilibrium)",
                   abs(mu_max - mu_wall) / scale, 1e-3,
                   f"value = |max|mu| - |2 Phi_w - 2 sinh Phi_w|| / (2 Phi_w); "
                   f"max|mu| = {mu_max:.4e}, Boltzmann (nonlinear) = {mu_wall:.4e}, "
                   f"linearised limit (Debye-Hueckel) = 0. "
                   f"Boltzmann holds in any geometry; Bessel I0 enters only Phi(r)"),
            _check("Boltzmann comparison: mu = 2 Phi - 2 sinh Phi at all nodes (equilibrium)",
                   float(np.max(np.abs(mu_eq - mu_boltz))) / scale, 1e-3,
                   "value = max |mu - (2 Phi - 2 sinh Phi)| / (2 Phi_w) over all nodes"),
        ]
        return checks, {"equilibrium": {"phi_axis_Z05": float(phi[0]),
                                        "phi_axis_Z05_PB": float(ex[0]),
                                        "rel_err": err}}

    def make_figures(folder, lines, summary, log):
        from . import figures_pore
        return figures_pore.write(folder, lines, summary, TAU_EARLY, p["tau_mid"], log)

    notes = [
        f"lambda_D = {lam * 1e9:.4g} nm, delta = lambda_D/a_p = {delta:.4g} "
        f"(kappa = {1 / delta:.4g}); Phi_w = {phi_w:.4g}; L = l_p/a_p = {L:g}.",
        f"SDL: radius {a_s:g} a_p, length {l_s:g} a_p (= L/2 + 1.5 max alpha by default).",
        f"tau = D t / l_p^2 = t_solver / {t_per_tau:g}; 1 tau = "
        f"{(p['L_nm'] * 1e-9) ** 2 / p['D']:.4g} s.",
        "mu = rho + 2 Phi (cross-section average with weight r dr in axial lines).",
        f"mesh: h_wall = {h_wall:.3g}, h_entr = {h_entr:.3g} (mouth corner), h_far = {h_far:g}"
        + (f", size <= alpha(Z)/{p['h_radius_n']:g} in the pore." if p.get("h_radius_n") else "."),
        "Axisymmetric (r, z): mesh x = r, y = z; integrals per radian.",
    ]
    kP, eP, NP = PAWEL_CONST["kB"], PAWEL_CONST["e"], PAWEL_CONST["NA"]
    lamP = np.sqrt(p["eps"] * kP * p["T"] / (2 * eP ** 2 * p["c0_mM"] * NP))
    notes += [
        f"constants: CODATA (used) V_T = {K_B * p['T'] / E_CHARGE * 1e3:.5g} mV, "
        f"lambda_D = {lam * 1e9:.5g} nm; pawel.py rounded constants (report only) "
        f"V_T = {kP * p['T'] / eP * 1e3:.5g} mV, lambda_D = {lamP * 1e9:.5g} nm.",
        "DNS differences not matched (report only): DNS mouth is rounded, here a sharp "
        "corner refined with h_entr; DNS pore-end line has non-uniform phi, here the end "
        "face is phi = phi_end (phi_end_mV); the DNS Biot number has no "
        "equivalent here.",
    ]
    if p["dns_times"]:
        notes.append(f"output times: DNS folder times t = {DNS_DT_S:g} k s, k = 1..{DNS_N_OUT} "
                     f"(tau = {DNS_DT_S * tau_per_s:.5g} k), plus tau_end.")
    U0_dns = None
    if p["dns_init"]:
        Xn = np.asarray(model.mesh.X)
        U0_dns = np.array(model.initial_guess(params))
        U0_dns[model.dof("phi", np.arange(model.n))] = np.where(Xn[:, 1] >= 0, phi_w, 0.0)
        notes.append("initial state: DNS field (c = c0, phi = phi_w in the pore z >= 0, "
                     "0 in the SDL), saved as given at tau = 0 without the consistent-phi "
                     "solve; phi follows from Poisson from the first step on.")
    labels = {"phi": "potential Phi", "rho": "charge rho = (c+ - c-)/c0",
              "c+": "cation c+/c0", "c-": "anion c-/c0", "|E|": "|E| a_p / V_T"}
    transient = {
        "t_end": p["tau_end"] * t_per_tau,
        "outputs": [t * t_per_tau for t in taus],
        # always saved, whatever the case's save mode (the summary figure shows them)
        "required_outputs": [t * t_per_tau for t in (TAU_EARLY, p["tau_mid"])
                             if 0 < t <= p["tau_end"] * (1 + 1e-9)],
        "dt0": p["dt0"], "rtol": p["rtol"], "dt_max": 0.02 * t_per_tau,
        "t_per_tau": t_per_tau,
        "profiles": time_profiles,
        "figures": make_figures,
        "equilibrium": bool(p["equilibrium_check"]),
        "params": params,
        "U0": U0_dns,
        "on_accept": on_accept,
        "err_scale": "mixed" if p.get("err_mixed") else "abs",
    }
    # base/path describe the steady equilibrium solve: phi_w ramped up from 0
    return Setup(model, [{"phi_w": phi_w}], dict(params, phi_w=0.0),
                 focus_bbox=[0.0, a_s, -2.0 * a_s, 6.0 * amax],
                 profiles=equilibrium_lines if p["equilibrium_check"] else None,
                 extra=extra if p["equilibrium_check"] else None,
                 notes=notes, labels=labels, transient=transient)
