#!/usr/bin/env python3
"""Generate a synthetic pnpflow result folder for HUD development.

Writes ``results/_sample/`` (relative to the pnpflow root) following
docs/DATA_FORMAT.md: a graded cylinder mesh (~30k triangles), plausible
EDL-like fields (phi, c+, c-, rho, E), a potential-flow velocity with a
boundary-layer no-slip factor, 1D profiles, a summary with solver history,
checks, integrals and sensitivities, and a few matplotlib plots.

The data is *fake* — it only has to look realistic for UI work.

Usage:  python hud/make_sample.py [--out results/_sample]
"""
import argparse
import json
import math
import os
import time

import numpy as np
from scipy.spatial import Delaunay

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Domain (nondimensional, cylinder radius 1 at origin, flow in -z)
YMIN, YMAX, ZMIN, ZMAX = -10.0, 10.0, -20.0, 10.0
DELTA = 0.1          # Debye length / radius
PHI_WALL = 0.973     # 25 mV / thermal voltage
U_INF = 1.0
BL = 0.3             # fake hydrodynamic boundary-layer thickness


# --------------------------------------------------------------------------
# Mesh
# --------------------------------------------------------------------------
def ring_points():
    """Concentric rings around the cylinder, graded towards the wall."""
    pts, radii = [], []
    r, h0, g = 1.0, 0.004, 1.09
    k = 0
    while r < 3.0:                       # Debye-layer resolving rings
        radii.append((r, 220))
        r = 1.0 + h0 * (g ** (k + 1) - 1) / (g - 1)
        k += 1
    while r < 24.0:                      # outer rings with aspect ~1
        n = 220 if r < 6 else (110 if r < 12 else 56)
        radii.append((r, n))
        r += 2 * math.pi * r / n
    for i, (r, n) in enumerate(radii):
        off = 0.5 * (i % 2) * 2 * math.pi / n
        th = np.arange(n) * 2 * math.pi / n + off
        pts.append(np.c_[r * np.sin(th), r * np.cos(th)])
    return np.vstack(pts)


def local_h(y, z):
    r = math.hypot(y, z)
    n = 220 if r < 6 else (110 if r < 12 else 56)
    return 2 * math.pi * max(r, 1.0) / n


def rect_boundary_points():
    """Points along the outer rectangle, spaced like the nearby rings."""
    corners = [(YMIN, ZMIN), (YMAX, ZMIN), (YMAX, ZMAX), (YMIN, ZMAX)]
    pts = []
    for a, b in zip(corners, corners[1:] + corners[:1]):
        a, b = np.array(a), np.array(b)
        L = np.linalg.norm(b - a)
        s = 0.0
        while s < L - 1e-9:
            p = a + (b - a) * s / L
            pts.append(p)
            s += local_h(*p)
    return np.array(pts)


def build_mesh():
    ring = ring_points()
    inside = (ring[:, 0] > YMIN) & (ring[:, 0] < YMAX) & (ring[:, 1] > ZMIN) & (ring[:, 1] < ZMAX)
    ring = ring[inside]
    dist = np.minimum.reduce([ring[:, 0] - YMIN, YMAX - ring[:, 0], ring[:, 1] - ZMIN, ZMAX - ring[:, 1]])
    hloc = np.array([local_h(*p) for p in ring])
    ring = ring[dist > 0.6 * hloc]
    nodes = np.vstack([ring, rect_boundary_points()])
    tri = Delaunay(nodes).simplices
    cen = nodes[tri].mean(axis=1)
    tri = tri[np.hypot(cen[:, 0], cen[:, 1]) > 1.0]
    # orient CCW
    a, b, c = nodes[tri[:, 0]], nodes[tri[:, 1]], nodes[tri[:, 2]]
    area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    tri[area2 < 0] = tri[area2 < 0][:, [0, 2, 1]]
    # drop unused nodes, renumber
    used = np.unique(tri)
    remap = -np.ones(len(nodes), int)
    remap[used] = np.arange(len(used))
    return nodes[used], remap[tri]


def boundary_edges(nodes, tri):
    e = np.sort(np.vstack([tri[:, [0, 1]], tri[:, [1, 2]], tri[:, [2, 0]]]), axis=1)
    uniq, cnt = np.unique(e, axis=0, return_counts=True)
    bnd = uniq[cnt == 1]
    mid = nodes[bnd].mean(axis=1)
    groups = {"wall": [], "inlet": [], "outlet": [], "side": []}
    for (i, j), (y, z) in zip(bnd.tolist(), mid):
        if math.hypot(y, z) < 1.5:
            groups["wall"].append([i, j])
        elif abs(z - ZMAX) < 1e-6:
            groups["inlet"].append([i, j])
        elif abs(z - ZMIN) < 1e-6:
            groups["outlet"].append([i, j])
        else:
            groups["side"].append([i, j])
    return groups


def tri_quality(nodes, tri):
    p = nodes[tri]
    l2 = sum(np.sum((p[:, (k + 1) % 3] - p[:, k]) ** 2, axis=1) for k in range(3))
    a, b, c = p[:, 0], p[:, 1], p[:, 2]
    area = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
    return 4 * math.sqrt(3) * area / l2, np.sqrt(np.max(
        [np.sum((p[:, (k + 1) % 3] - p[:, k]) ** 2, axis=1) for k in range(3)], axis=0))


def nodal_gradient(nodes, tri, f):
    """Area-weighted average of P1 element gradients onto nodes."""
    p = nodes[tri]
    x, y = p[..., 0], p[..., 1]
    det = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (y[:, 1] - y[:, 0])
    fv = f[tri]
    gx = ((fv[:, 1] - fv[:, 0]) * (y[:, 2] - y[:, 0]) - (fv[:, 2] - fv[:, 0]) * (y[:, 1] - y[:, 0])) / det
    gy = ((fv[:, 2] - fv[:, 0]) * (x[:, 1] - x[:, 0]) - (fv[:, 1] - fv[:, 0]) * (x[:, 2] - x[:, 0])) / det
    w = np.abs(det) / 2
    acc = np.zeros((len(nodes), 3))
    for k in range(3):
        np.add.at(acc, tri[:, k], np.c_[gx * w, gy * w, w])
    return acc[:, 0] / acc[:, 2], acc[:, 1] / acc[:, 2]


# --------------------------------------------------------------------------
# Fields
# --------------------------------------------------------------------------
def compute_fields(nodes, tri):
    y, z = nodes[:, 0], nodes[:, 1]
    r = np.maximum(np.hypot(y, z), 1.0)
    phi = PHI_WALL * np.exp(-(r - 1) / DELTA)
    # slight fore/aft asymmetry from advection of the diffuse layer
    th = np.arctan2(y, z)                       # 0 = upstream (+z)
    phi = phi * (1 + 0.08 * np.cos(th) * (r - 1) / (r - 1 + DELTA))
    cp, cm = np.exp(-phi), np.exp(phi)
    rho = (cp - cm) / (2 * DELTA ** 2)
    # potential flow past cylinder for uniform flow U in -z, plus no-slip factor
    # local frame x' = -z (stream direction), y' = y
    xp, yp = -z, y
    rr2 = np.maximum(xp ** 2 + yp ** 2, 1.0)
    ux_p = U_INF * (1 - (xp ** 2 - yp ** 2) / rr2 ** 2)
    uy_p = U_INF * (-2 * xp * yp / rr2 ** 2)
    f = 1 - np.exp(-(r - 1) / BL)
    # mild wake deficit downstream (z < 0)
    wake = 1 - 0.35 * np.exp(-(y ** 2) / (0.6 + 0.08 * np.maximum(-z, 0))) * (z < 0) * np.exp(z / 12)
    uz = -ux_p * f * wake
    uy = uy_p * f
    umag = np.hypot(uy, uz)
    p = 0.5 * (U_INF ** 2 - umag ** 2) + 0.02 * np.cos(th) / r
    dphidy, dphidz = nodal_gradient(nodes, tri, phi)
    Ey, Ez = -dphidy, -dphidz
    duzdy, _ = nodal_gradient(nodes, tri, uz)
    _, duydz = nodal_gradient(nodes, tri, uy)
    vort = duzdy - duydz
    dcy, dcz = nodal_gradient(nodes, tri, cp)
    Npy = cp * uy - dcy + cp * Ey
    Npz = cp * uz - dcz + cp * Ez
    return {
        "c+": cp, "c-": cm, "phi": phi, "rho": rho,
        "uy": uy, "uz": uz, "|u|": umag, "p": p,
        "Ey": Ey, "Ez": Ez, "|E|": np.hypot(Ey, Ez), "vorticity": vort,
        "N_c+_y": Npy, "N_c+_z": Npz,
    }


def rounded(a, sig=6):
    return [float(f"{v:.{sig}g}") for v in np.asarray(a, float)]


# --------------------------------------------------------------------------
# Profiles
# --------------------------------------------------------------------------
def make_profiles(nodes, tri, fields):
    import matplotlib.tri as mtri
    T = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tri)
    finder = T.get_trifinder()
    interp = {k: mtri.LinearTriInterpolator(T, v, trifinder=finder) for k, v in fields.items()}

    def sample(ys, zs, names):
        out = {}
        for n in names:
            v = interp[n](ys, zs)
            out[n] = rounded(np.ma.filled(v, np.nan))
        return out

    lines = []
    rr = 1 + 1e-4 + np.concatenate([np.geomspace(1e-3, 0.5, 80), np.linspace(0.52, 6, 60)])
    rr = np.concatenate([[1.0 + 1e-5], rr])
    for deg, lab in [(0, "upstream"), (90, "side"), (180, "downstream")]:
        t = math.radians(deg)
        ys, zs = rr * math.sin(t), rr * math.cos(t)
        vals = sample(ys, zs, ["c+", "c-", "phi", "rho", "|u|", "uz", "p", "|E|"])
        lines.append({"name": f"radial_{deg}", "label": f"radial cut, theta = {deg} deg ({lab})",
                      "coord_name": "r", "coord": rounded(rr), "values": vals})
    th = np.arange(0, 362, 2.0)
    rw = 1.0 + 2e-4
    ys, zs = rw * np.sin(np.radians(th)), rw * np.cos(np.radians(th))
    vals = sample(ys, zs, ["phi", "c+", "c-", "p", "vorticity"])
    vals["sigma"] = rounded(np.array(vals["phi"]) / DELTA * DELTA ** 2 * 10)
    vals["shear"] = vals.pop("vorticity")
    lines.append({"name": "wall", "label": "on cylinder wall", "coord_name": "theta_deg",
                  "coord": rounded(th), "values": vals})
    zz = np.linspace(ZMIN, ZMAX, 200)
    zz = zz[np.abs(zz) > 1.0005]
    vals = sample(np.zeros_like(zz) + 1e-6, zz, ["uz", "|u|", "p", "phi", "c+", "c-"])
    lines.append({"name": "axis", "label": "symmetry axis y = 0", "coord_name": "z",
                  "coord": rounded(zz), "values": vals})
    return {"lines": lines}


# --------------------------------------------------------------------------
# Summary / progress
# --------------------------------------------------------------------------
def newton_history(rng):
    stages, hist = [], []
    for s, pw in enumerate([0.3, 0.65, PHI_WALL], start=1):
        res, its = 0.4 / s, []
        for it in range(8):
            alpha = 1.0 if it > 0 or s == 1 else 0.5
            its.append({"it": it, "res_norm": float(f"{res:.4g}"),
                        "step_norm": float(f"{min(1.0, res * 3):.4g}"), "alpha": alpha})
            hist.append({"stage": s, **its[-1]})
            if res < 1e-10:
                break
            res = res ** (1.6 if it > 1 else 1.2) * (0.5 + rng.random()) if res < 0.5 else res * 0.3
        stages.append({"params": {"phi_wall": pw}, "converged": True, "iterations": its})
    return stages, hist


def make_summary(nodes, tri, bedges, fields, stages):
    q, hmax_el = tri_quality(nodes, tri)
    counts, edges = np.histogram(q, bins=20, range=(0, 1))
    wall_len = sum(np.linalg.norm(nodes[i] - nodes[j]) for i, j in bedges["wall"])
    return {
        "run_name": "_sample", "case": "cylinder",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": {"species": [{"name": "c+", "z": 1, "d": 1.0}, {"name": "c-", "z": -1, "d": 1.0}],
                  "flow": "navier_stokes", "advection": "conservative", "supg": False},
        "params": {"delta": DELTA, "Pe": 1.0, "Re": 0.1, "K": 2.0, "phi_wall": PHI_WALL,
                   "U": U_INF, "L_up": ZMAX, "L_down": -ZMIN, "W": YMAX, "h_wall": 0.004},
        "param_info": {
            "delta": "Debye length / cylinder radius (lambda_D / a)",
            "Pe": "ion Peclet number U a / D",
            "Re": "Reynolds number rho U a / mu",
            "K": "electro-viscous coupling eps V_T^2 / (mu U a)",
            "phi_wall": "wall potential / thermal voltage (25 mV -> 0.973)",
            "U": "far-field velocity magnitude (flow in -z)",
            "L_up": "upstream domain length", "L_down": "downstream domain length",
            "W": "half width of the domain", "h_wall": "first-cell size at the wall",
        },
        "mesh": {"n_elements": int(len(tri)), "n_nodes": int(len(nodes)),
                 "n_dofs": int(len(nodes) * 5 + len(nodes) * 3),
                 "h_min": float(f"{hmax_el.min():.4g}"), "h_max": float(f"{hmax_el.max():.4g}"),
                 "quality_min": float(f"{q.min():.4g}"),
                 "quality_hist": {"edges": rounded(edges, 4), "counts": counts.tolist()}},
        "solver": {"converged": True, "time_s": 41.7,
                   "n_newton_total": sum(len(s["iterations"]) for s in stages), "stages": stages},
        "checks": [
            {"name": "min concentration", "value": float(f"{min(fields['c+'].min(), fields['c-'].min()):.4g}"),
             "limit": 0.0, "status": "pass", "note": "must stay > 0"},
            {"name": "Newton converged", "value": 1.0, "limit": 1.0, "status": "pass", "note": "all stages"},
            {"name": "wall flux c+ (rel)", "value": 3.2e-9, "limit": 1e-6, "status": "pass",
             "note": "blocking wall: no normal flux"},
            {"name": "charge balance", "value": 4.1e-4, "limit": 1e-3, "status": "warn",
             "note": "|diffuse + wall| / |wall|, close to limit"},
            {"name": "Debye resolution", "value": round(DELTA / 0.004, 2), "limit": 10.0, "status": "pass",
             "note": "cells per Debye length at wall"},
            {"name": "outlet reflux", "value": 0.012, "limit": 0.0, "status": "fail",
             "note": "inflow detected at outlet (domain too short?)"},
            {"name": "wall length", "value": round(wall_len, 5), "limit": round(2 * math.pi, 5),
             "status": "info", "note": "discrete perimeter vs 2 pi"},
        ],
        "integrals": {
            "flux": {
                "c+": {"wall": 3.1e-12, "inlet": -20.04, "outlet": 20.03, "side": 0.011},
                "c-": {"wall": -2.7e-12, "inlet": -19.98, "outlet": 19.99, "side": -0.009},
            },
            "force": {"hydro": [0.0013, -11.84], "electric": [-0.0002, -0.412],
                      "total": [0.0011, -12.252], "drag": 12.252, "lift": 0.0011},
            "charge": {"diffuse": -0.3093, "wall_induced": 0.3094, "sum": 1.27e-4},
            "other": {"div_u_L2": 2.3e-7, "mass_inlet": -20.0, "mass_outlet": 20.0},
        },
        "sensitivities": {
            "drag": {"K": 0.121, "delta": -3.04, "Re": 1.87, "phi_wall": 0.442},
            "lift": {"K": 1.2e-5, "delta": -2.0e-4, "Re": 3.1e-4, "phi_wall": 4.0e-6},
            "charge_diffuse": {"K": 0.0, "delta": 2.71, "Re": 0.0, "phi_wall": -0.318},
        },
        "notes": [
            "SYNTHETIC SAMPLE generated by hud/make_sample.py - not a real solution.",
            "Continuation in phi_wall over 3 stages: 0.3 -> 0.65 -> 0.973.",
            "Electroviscous drag increment ~ 3.4% relative to uncharged cylinder.",
        ],
    }


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def make_plots(out, nodes, tri, bedges, fields, profiles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    os.makedirs(os.path.join(out, "plots"), exist_ok=True)
    T = mtri.Triangulation(nodes[:, 0], nodes[:, 1], tri)

    def save(fig, name):
        fig.savefig(os.path.join(out, "plots", name), dpi=110, bbox_inches="tight")
        plt.close(fig)

    for name, lim in [("mesh.png", None), ("mesh_zoom.png", 1.6)]:
        fig, ax = plt.subplots(figsize=(6, 7 if lim is None else 6))
        ax.triplot(T, lw=0.15 if lim is None else 0.3, color="k")
        ax.set_aspect("equal")
        if lim:
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
        ax.set_xlabel("y"); ax.set_ylabel("z"); ax.set_title(name[:-4].replace("_", " "))
        save(fig, name)
    import cmasher  # noqa: F401  registers the cmr.* colormaps (never matplotlib defaults)
    for key, cmap, fn in [("phi", "cmr.cosmic", "phi.png"), ("|u|", "cmr.ember", "u_mag.png"),
                          ("c-", "cmr.bubblegum", "c_minus.png")]:
        fig, ax = plt.subplots(figsize=(6, 5))
        tc = ax.tripcolor(T, fields[key], shading="gouraud", cmap=cmap)
        fig.colorbar(tc, ax=ax, label=key)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
        ax.set_xlabel("y"); ax.set_ylabel("z"); ax.set_title(key)
        save(fig, fn)
    fig, ax = plt.subplots(figsize=(6, 4))
    ln = profiles["lines"][0]
    r = np.array(ln["coord"])
    for k in ["c+", "c-", "phi"]:
        ax.semilogx(r - 1, ln["values"][k], label=k)
    ax.set_xlabel("r - 1"); ax.legend(); ax.set_title(ln["label"]); ax.grid(alpha=0.3)
    save(fig, "profiles_radial.png")


# --------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "_sample"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    rng = np.random.default_rng(1)

    nodes, tri = build_mesh()
    bedges = boundary_edges(nodes, tri)
    fields = compute_fields(nodes, tri)
    profiles = make_profiles(nodes, tri, fields)
    stages, hist = newton_history(rng)
    summary = make_summary(nodes, tri, bedges, fields, stages)

    fj = {
        "nodes": [[round(float(a), 6), round(float(b), 6)] for a, b in nodes],
        "triangles": tri.tolist(),
        "boundary_edges": bedges,
        "fields": {k: rounded(v, 5) for k, v in fields.items()},
        "vectors": {"u": ["uy", "uz"], "E": ["Ey", "Ez"], "N_c+": ["N_c+_y", "N_c+_z"]},
        "labels": {"c+": "cation concentration c+/c0", "c-": "anion concentration c-/c0",
                   "phi": "potential phi/V_T", "rho": "charge density rho", "|u|": "speed |u|/U",
                   "uy": "velocity u_y", "uz": "velocity u_z", "p": "pressure p",
                   "vorticity": "vorticity omega_x", "|E|": "field strength |E|",
                   "Ey": "electric field E_y", "Ez": "electric field E_z"},
        "bbox": [YMIN, YMAX, ZMIN, ZMAX],
        "focus_bbox": [-3, 3, -3, 3],
    }
    with open(os.path.join(out, "fields.json"), "w") as f:
        json.dump(fj, f, separators=(",", ":"))
    with open(os.path.join(out, "profiles.json"), "w") as f:
        # points outside the mesh interpolate to NaN -> write as JSON null
        f.write(json.dumps(profiles, separators=(",", ":")).replace("NaN", "null"))
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    with open(os.path.join(out, "progress.json"), "w") as f:
        json.dump({"state": "done", "stage": 3, "n_stages": 3, "iteration": len(stages[-1]["iterations"]) - 1,
                   "res_norm": hist[-1]["res_norm"], "message": "finished", "time_s": 41.7,
                   "history": hist}, f)
    with open(os.path.join(out, "stdout.log"), "w") as f:
        f.write("[sample] synthetic run generated by hud/make_sample.py\n")
        for h in hist:
            f.write(f"stage {h['stage']} it {h['it']:2d}  |R| = {h['res_norm']:.3e}  alpha = {h['alpha']}\n")
        f.write("done.\n")
    if not args.no_plots:
        make_plots(out, nodes, tri, bedges, fields, profiles)
    print(f"wrote {out}: {len(nodes)} nodes, {len(tri)} triangles, "
          f"boundaries {', '.join(f'{k}={len(v)}' for k, v in bedges.items())}")


if __name__ == "__main__":
    main()
