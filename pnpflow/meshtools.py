"""Mesh rendering (two views) and mesh diagnosis (docs/MCP.md §4–5).

render_two_views(mesh, folder) -> mesh/view_a.png, mesh/view_b.png
diagnose(mesh, bcs, delta, charged) -> checks, written to mesh/diagnose.json

numpy + scipy + matplotlib only (no jax).
"""
import json
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                       # noqa: E402
from matplotlib.collections import LineCollection, PolyCollection  # noqa: E402
from matplotlib.lines import Line2D                    # noqa: E402
from matplotlib.tri import Triangulation               # noqa: E402

from . import figures                                  # noqa: E402

DPI = 300


# =====================================================================
# Geometry helpers
# =====================================================================

def _coord(mesh):
    return getattr(mesh, "coord", "planar")


def _edge_curve(X, edges, n=5):
    """Points along P2 edges (B, 3) [end, end, mid] -> (B, n, 2)."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    a, b, m = X[edges[:, 0]], X[edges[:, 1]], X[edges[:, 2]]
    return (a[:, None] * ((1 - t) * (1 - 2 * t)) + b[:, None] * (t * (2 * t - 1))
            + m[:, None] * (4 * t * (1 - t)))


def _element_outline(X, tri, n=4):
    """Curved outline of P2 triangles (E, 3n, 2)."""
    sides = [(0, 1, 3), (1, 2, 4), (2, 0, 5)]
    parts = [_edge_curve(X, tri[:, list(s)], n + 1)[:, :-1] for s in sides]
    return np.concatenate(parts, axis=1)


def _areas(X, tri):
    P = X[tri[:, :3]]
    e1, e2 = P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]
    return 0.5 * (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0])   # signed


def element_size(mesh):
    """Side of the equilateral triangle with the same area."""
    return np.sqrt(4.0 * np.abs(_areas(mesh.X, mesh.tri)) / np.sqrt(3.0))


def _neighbours(mesh):
    """Pairs of triangles sharing an edge, from the shared P2 mid node."""
    mids = mesh.tri[:, 3:6].ravel()
    owner = np.repeat(np.arange(mesh.n_elements), 3)
    order = np.argsort(mids, kind="stable")
    m, o = mids[order], owner[order]
    same = m[1:] == m[:-1]
    pairs = np.stack([o[:-1][same], o[1:][same]], axis=1)
    counts = np.bincount(mids, minlength=mesh.n_nodes)
    return pairs, counts


def _boundary_h(mesh):
    """Mean boundary-edge length per named boundary."""
    out = {}
    for name, b in mesh.boundaries.items():
        e = b["edges"]
        if len(e):
            out[name] = float(np.mean(np.linalg.norm(mesh.X[e[:, 1]] - mesh.X[e[:, 0]], axis=1)))
    return out


def _group_colors(names):
    # distinct colours sampled from the three house colormaps (cosmic, ember, bubblegum)
    picks = [("cosmic", 0.55), ("ember", 0.6), ("bubblegum", 0.5), ("cosmic", 0.85),
             ("ember", 0.35), ("bubblegum", 0.8), ("cosmic", 0.35), ("ember", 0.85)]
    return {k: figures.cmap(picks[i % len(picks)][0])(picks[i % len(picks)][1])
            for i, k in enumerate(names)}


# =====================================================================
# Two views
# =====================================================================

def render_two_views(mesh, folder, focus=None, symmetry=None, log=print):
    """Write mesh/view_a.png (whole domain by group) and mesh/view_b.png
    (zoom on the finest region by quality).  Returns [{file, title, what_to_check}]."""
    out = os.path.join(folder, "mesh")
    os.makedirs(out, exist_ok=True)
    coord = symmetry or _coord(mesh)
    views = []
    for fn in (_view_a, _view_b):
        views.append(figures.render(fn, mesh, out, focus, coord, log=log))
    return views


def _labels(ax, coord):
    xn, yn = figures.coord_names(coord)
    ax.set_xlabel(f"${xn}$")
    ax.set_ylabel(f"${yn}$")
    ax.tick_params(axis="x", pad=10)
    ax.set_aspect("equal")
    figures.arrow_axes(ax)


def _figsize(w_data, h_data, width=7.0, max_h=7.0):
    a = h_data / max(w_data, 1e-300)
    w, h = width, width * a
    if h > max_h:
        h, w = max_h, max_h / a
    return w, max(h, 0.6)


def _view_a(mesh, out, focus, coord):
    X = mesh.X
    x0, x1, y0, y1 = X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()
    w, h = _figsize(x1 - x0, y1 - y0,
                    width=10.0 if (y1 - y0) < 0.25 * (x1 - x0) else 7.0)
    fig = plt.figure(figsize=(w + 3.2, h + 1.6))
    ax = fig.add_axes([0.9 / (w + 3.2), 1.0 / (h + 1.6), w / (w + 3.2), h / (h + 1.6)])
    t = mesh.tri
    segs = np.concatenate([X[t[:, [0, 1]]], X[t[:, [1, 2]]], X[t[:, [2, 0]]]])
    ax.add_collection(LineCollection(segs, colors="0.72", linewidths=0.15,
                                     rasterized=True))
    names = [k for k, b in mesh.boundaries.items() if len(b["edges"])]
    colors = _group_colors(names)
    handles = []
    for k in names:
        curves = _edge_curve(X, mesh.boundaries[k]["edges"])
        ax.add_collection(LineCollection(curves, colors=[colors[k]], linewidths=3.0))
        label = k.replace("_", r"\_") if plt.rcParams["text.usetex"] else k
        handles.append(Line2D([], [], color=colors[k], lw=3.0, label=label))
    _unnamed_overlay(ax, mesh, handles)
    pad = 0.02 * max(x1 - x0, y1 - y0)
    ax.set_xlim(x0 - pad, x1 + pad)
    ax.set_ylim(y0 - pad, y1 + pad)
    _labels(ax, coord)
    ax.set_title(f"Mesh, {mesh.n_elements} P2 triangles", pad=20)
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.02, 0.5),
              frameon=False, fontsize=20, handlelength=1.5)
    figures.save_to(fig, os.path.join(out, "view_a.png"))
    return {"file": "mesh/view_a.png",
            "title": "Whole domain, boundaries coloured by physical group",
            "what_to_check": "every boundary has the expected colour and name (red dots "
                             "= unnamed edges); the domain has the intended shape and size; the "
                             "element density is highest at charged/reacting walls"}


def _unnamed_overlay(ax, mesh, handles):
    named = set()
    for b in mesh.boundaries.values():
        named.update(b["edges"][:, 2].tolist())
    _, counts = _neighbours(mesh)
    mids = np.unique(mesh.tri[:, 3:6])
    bmid = mids[counts[mids] == 1]
    missing = np.array([m for m in bmid if m not in named], dtype=np.int64)
    if len(missing):
        pts = mesh.X[missing]
        ax.plot(pts[:, 0], pts[:, 1], "o", color="red", ms=3)
        handles.append(Line2D([], [], color="red", marker="o", ls="", label="unnamed"))


def _zoom_window(mesh, focus):
    """[x0, x1, y0, y1] around the boundary with the smallest h (or focus)."""
    if focus:
        return list(map(float, focus)), None
    hb = _boundary_h(mesh)
    if not hb:
        X = mesh.X
        return [X[:, 0].min(), X[:, 0].max(), X[:, 1].min(), X[:, 1].max()], None
    name = min(hb, key=hb.get)
    e = mesh.boundaries[name]["edges"]
    X = mesh.X
    lengths = np.linalg.norm(X[e[:, 1]] - X[e[:, 0]], axis=1)
    k = int(np.argmin(lengths + 1e-12 * np.arange(len(lengths))))
    # the smallest edge of that boundary, nearest its middle when sizes tie
    near = np.where(lengths <= lengths.min() * 1.05)[0]
    if len(near) > 1:
        centre = X[e[near, 2]].mean(axis=0)
        k = near[np.argmin(np.linalg.norm(X[e[near, 2]] - centre, axis=1))]
    p = X[e[k, 2]]
    t = X[e[k, 1]] - X[e[k, 0]]
    n_out = np.array([t[1], -t[0]]) * mesh.boundaries[name].get("sign", np.ones(len(e)))[k]
    n_out /= np.linalg.norm(n_out) or 1.0
    L = 40.0 * hb[name]
    span = max(X[:, 0].max() - X[:, 0].min(), X[:, 1].max() - X[:, 1].min())
    L = min(L, span)
    c = p - n_out * 0.3 * L
    box = []
    for d in range(2):                     # keep the window inside the domain bbox
        lo, hi = X[:, d].min(), X[:, d].max()
        a = c[d] - L / 2
        if L >= hi - lo:
            a = 0.5 * (lo + hi) - L / 2
        else:
            a = min(max(a, lo), hi - L)
        box += [a, a + L]
    return box, name


def _view_b(mesh, out, focus, coord):
    X = mesh.X
    box, bname = _zoom_window(mesh, focus)
    q = mesh.quality()
    inside = ((X[:, 0] >= box[0]) & (X[:, 0] <= box[1])
              & (X[:, 1] >= box[2]) & (X[:, 1] <= box[3]))
    sel = inside[mesh.tri].any(axis=1)
    w, h = _figsize(box[1] - box[0], box[3] - box[2], width=6.5)
    W = w + 2.65                               # room for the x arrow before the colour bar
    fig = plt.figure(figsize=(W, h + 1.6))
    ax = fig.add_axes([0.9 / W, 1.0 / (h + 1.6), w / W, h / (h + 1.6)])
    polys = _element_outline(X, mesh.tri[sel])
    pc = PolyCollection(polys, array=q[sel], cmap=figures.cmap("cosmic"),
                        edgecolors="0.15", linewidths=0.3, rasterized=True)
    pc.set_clim(0.0, 1.0)
    ax.add_collection(pc)
    colors = _group_colors([k for k, b in mesh.boundaries.items() if len(b["edges"])])
    for k, b in mesh.boundaries.items():
        if len(b["edges"]):
            ax.add_collection(LineCollection(_edge_curve(X, b["edges"]),
                                             colors=[colors[k]], linewidths=3.0))
    ax.set_xlim(box[0], box[1])
    ax.set_ylim(box[2], box[3])
    _labels(ax, coord)
    qmin = float(q[sel].min()) if sel.any() else float("nan")
    ax.set_title(figures.tex(f"Finest region, $q_{{\\min}} = {qmin:.2f}$"), pad=20)
    cax = fig.add_axes([(0.9 + w + 0.5) / W, 1.0 / (h + 1.6), 0.22 / W, h / (h + 1.6)])
    figures.colorbar(fig, pc, cax, "$q$")
    figures.save_to(fig, os.path.join(out, "view_b.png"))
    where = f"around boundary '{bname}' (smallest h)" if bname else "focus window"
    return {"file": "mesh/view_b.png",
            "title": f"Zoom on the finest region, {where}, elements coloured by shape "
                     f"quality q (1 = equilateral)",
            "what_to_check": "the layer of small elements hugs the wall; size grows "
                             "smoothly away from it with no sudden jumps; no dark "
                             "(low-q) slivers; curved P2 edges follow the wall arc"}


# =====================================================================
# Diagnose
# =====================================================================

def _chk(name, value, limit, status, note, fix=""):
    return {"name": name, "value": None if value is None else float(value),
            "limit": None if limit is None else float(limit),
            "status": status, "note": note, "fix": fix}


def _bc_kind(v):
    return type(v).__name__ if not isinstance(v, (str, dict)) else v


def charged_boundaries(bcs):
    """Boundaries that carry an EDL: a potential, Stern or charge BC on phi,
    or a species flux (reacting), except bulk boundaries (Dirichlet c)."""
    out = []
    for name, spec in (bcs or {}).items():
        c = spec.get("c")
        bulk = c is not None and (_bc_kind(c) == "Dirichlet"
                                  or (isinstance(c, dict) and c and all(
                                      _bc_kind(x) == "Dirichlet" for x in c.values())))
        phi = spec.get("phi")
        k = _bc_kind(phi) if phi is not None else None
        charged = k in ("Robin", "Charge")
        if k == "Dirichlet":
            val = getattr(phi, "value", 0.0)
            charged = not (isinstance(val, (int, float)) and val == 0.0)
        reacting = c is not None and ("Flux" in {_bc_kind(c)} or (
            isinstance(c, dict) and any(_bc_kind(x) == "Flux" for x in c.values())))
        if (charged and not bulk) or reacting:
            out.append(name)
    return out


def far_boundaries(bcs):
    """Bulk boundaries: c = Dirichlet('bulk')."""
    out = []
    for name, spec in (bcs or {}).items():
        c = spec.get("c")
        if _bc_kind(c) == "Dirichlet" and getattr(c, "value", None) == "bulk":
            out.append(name)
    return out


# fem.tri_quadrature() points and the P2 node positions (kept here: no jax import)
_a1, _b1, _a2, _b2 = 0.059715871789770, 0.470142064105115, 0.797426985353087, 0.101286507323456
_QUAD7 = np.array([(1 / 3, 1 / 3), (_b1, _b1), (_a1, _b1), (_b1, _a1),
                   (_b2, _b2), (_a2, _b2), (_b2, _a2)])
_P2_NODES = np.array([[0, 0], [1, 0], [0, 1], [0.5, 0], [0.5, 0.5], [0, 0.5]], float)


def p2_jacobians(X, tri):
    """det J of the P2 map at the 7 quadrature points + 6 nodes, (E, 13),
    divided by the det of the straight (corner) triangle."""
    pts = np.vstack([_QUAD7, _P2_NODES])
    xi, eta = pts[:, 0], pts[:, 1]
    l0 = 1.0 - xi - eta
    dxi = np.stack([-(4 * l0 - 1), 4 * xi - 1, 0 * xi, 4 * (l0 - xi), 4 * eta, -4 * eta], 1)
    deta = np.stack([-(4 * l0 - 1), 0 * xi, 4 * eta - 1, -4 * xi, 4 * xi, 4 * (l0 - eta)], 1)
    Xe = X[tri]                                           # (E, 6, 2)
    Jx = np.einsum("qa,eak->eqk", dxi, Xe)               # d x / d xi
    Je = np.einsum("qa,eak->eqk", deta, Xe)              # d x / d eta
    det = Jx[..., 0] * Je[..., 1] - Jx[..., 1] * Je[..., 0]
    lin = 2.0 * _areas(X, tri)
    return det / np.where(lin == 0, 1e-300, lin)[:, None]


def edl_resolution(mesh, names, delta, n_lines=400, n_samples=200):
    """Elements across 3 delta along inward normals: min over sampled lines of
    int_0^{3 delta} ds / h(x(s)).  Returns {boundary: (min, median)}."""
    X = mesh.X
    tr = Triangulation(X[:, 0], X[:, 1], mesh.tri[:, :3])
    finder = tr.get_trifinder()
    h = element_size(mesh)
    L = 3.0 * delta
    s = (np.arange(n_samples) + 0.5) / n_samples * L
    ds = L / n_samples
    out = {}
    for name in names:
        b = mesh.boundaries.get(name)
        if b is None or not len(b["edges"]):
            continue
        e = b["edges"]
        pick = np.unique(np.linspace(0, len(e) - 1, min(n_lines, len(e))).astype(int))
        t = X[e[pick, 1]] - X[e[pick, 0]]
        n_in = -np.stack([t[:, 1], -t[:, 0]], 1) * b["sign"][pick, None]
        n_in /= np.linalg.norm(n_in, axis=1, keepdims=True)
        p0 = X[e[pick, 2]]
        pts = p0[:, None, :] + n_in[:, None, :] * s[None, :, None]   # (P, S, 2)
        el = finder(pts[..., 0].ravel(), pts[..., 1].ravel()).reshape(len(pick), -1)
        # stop each line where it first leaves the domain
        outside = el < 0
        first_out = np.where(outside.any(1), outside.argmax(1), n_samples)
        valid = np.arange(n_samples)[None, :] < first_out[:, None]
        counts = np.where(valid, ds / h[np.maximum(el, 0)], 0.0).sum(1)
        out[name] = (float(counts.min()), float(np.median(counts)))
    return out


def diagnose(mesh, bcs=None, delta=None, charged=None, ndof=None, folder=None,
             n_species=None):
    """Mesh checks (docs/MCP.md §5) -> list of check dicts with a 'fix'.

    bcs     : {boundary: {...}} as given to Model (None: coverage not checked)
    delta   : Debye length / a (None: EDL checks skipped)
    charged : boundary names with an EDL (default: inferred from bcs)
    ndof    : dof count for the size budget (default: estimate, 2 species + flow)
    folder  : if given, writes <folder>/mesh/diagnose.json
    """
    X, tri = mesh.X, mesh.tri
    checks = []
    pairs, counts = _neighbours(mesh)
    names = [k for k, b in mesh.boundaries.items() if len(b["edges"])]

    # ---- unnamed / duplicate boundary edges --------------------------------
    mids = np.unique(tri[:, 3:6])
    bmid = mids[counts[mids] == 1]
    named = np.concatenate([b["edges"][:, 2] for b in mesh.boundaries.values()
                            if len(b["edges"])] or [np.zeros(0, np.int64)])
    unnamed = np.setdiff1d(bmid, named)
    checks.append(_chk("unnamed boundary edges", len(unnamed), 0,
                       "fail" if len(unnamed) else "pass",
                       f"{len(unnamed)} of {len(bmid)} outer edges are in no physical group"
                       " (the solver treats them as blocking/insulating)",
                       "add those curves to a dim-1 physical group" if len(unnamed) else ""))
    u, c = np.unique(named, return_counts=True)
    dup = int((c > 1).sum())
    checks.append(_chk("duplicate boundary naming", dup, 0, "fail" if dup else "pass",
                       f"{dup} edges are in more than one physical group",
                       "keep each curve in exactly one group" if dup else ""))

    # ---- BC coverage ------------------------------------------------------
    if bcs is not None:
        missing = sorted(set(bcs) - set(mesh.boundaries))
        natural = sorted(set(names) - set(bcs))
        checks.append(_chk("BC names a missing boundary", len(missing), 0,
                           "fail" if missing else "pass",
                           ("missing: " + ", ".join(missing)) if missing
                           else "every BC names a mesh boundary",
                           "rename the BC or the physical group" if missing else ""))
        checks.append(_chk("boundaries without BC", len(natural), None, "info",
                           ("natural BC used on: " + ", ".join(natural)
                            + " (blocking, no surface charge, traction free)")
                           if natural else "every boundary has a BC entry",
                           "add a BC if natural is not intended" if natural else ""))

    # ---- quality ----------------------------------------------------------
    q = mesh.quality()
    qmin = float(q.min())
    st = "fail" if qmin < 0.2 else ("warn" if qmin < 0.3 else "pass")
    checks.append(_chk("min quality", qmin, 0.2 if st == "fail" else 0.3, st,
                       f"shape quality, 1 = equilateral; {int((q < 0.3).sum())} elements"
                       f" below 0.3, mean {q.mean():.2f}",
                       "lower grow, add a size box, or remesh" if st != "pass" else ""))

    # ---- size jump --------------------------------------------------------
    h = element_size(mesh)
    if len(pairs):
        r = h[pairs].max(1) / np.maximum(h[pairs].min(1), 1e-300)
        jmax = float(r.max())
        n_bad = int((r > 2).sum())
    else:
        jmax, n_bad = 1.0, 0
    checks.append(_chk("size jump", jmax, 2.0, "warn" if jmax > 2 else "pass",
                       f"max size ratio of neighbouring elements; {n_bad} pairs > 2",
                       "smoother grow (larger growth distance)" if jmax > 2 else ""))

    # ---- inverted / bad P2 ------------------------------------------------
    J = p2_jacobians(X, tri)
    jmin = float(J.min())
    bad = int((J.min(1) <= 0).sum())
    flipped = int((_areas(X, tri) < 0).sum())
    checks.append(_chk("inverted / bad P2", jmin, 0.0, "fail" if bad else
                       ("warn" if jmin < 0.2 else "pass"),
                       f"min of det J / det J_straight over quadrature points and nodes;"
                       f" {bad} elements non-positive"
                       + (f"; {flipped} elements have clockwise corners" if flipped else ""),
                       "optimize('HighOrder'), smaller h on curved boundaries" if bad else ""))

    # ---- connectivity -----------------------------------------------------
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    E = mesh.n_elements
    A = coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(E, E)) \
        if len(pairs) else coo_matrix((E, E))
    ncomp, _ = connected_components(A, directed=False)
    checks.append(_chk("connected regions", ncomp, 1, "warn" if ncomp > 1 else "pass",
                       "connected components of the triangle adjacency",
                       "fix the booleans (use fragment)" if ncomp > 1 else ""))

    # ---- EDL resolution + far field ---------------------------------------
    if charged is None:
        charged = charged_boundaries(bcs) if bcs else []
    charged = [k for k in charged if k in mesh.boundaries]
    if delta and charged and n_species != 0:
        res = edl_resolution(mesh, charged, float(delta))
        worst = min(res, key=lambda k: res[k][0])
        nmin = res[worst][0]
        detail = "; ".join(f"{k}: min {a:.1f}, median {b:.1f}" for k, (a, b) in res.items())
        checks.append(_chk("EDL resolution", nmin, 8.0, "fail" if nmin < 8 else "pass",
                           f"elements across 3 delta (delta = {float(delta):g}) along wall "
                           f"normals, int ds/h; {detail}",
                           f"lower h_wall to <= delta/8 = {float(delta) / 8:.3g} or widen "
                           f"the refined layer at '{worst}'" if nmin < 8 else ""))
        far = [k for k in far_boundaries(bcs) if k in mesh.boundaries and k not in charged]
        if far:
            from scipy.spatial import cKDTree
            tree = cKDTree(np.vstack([X[mesh.boundary_nodes(k)] for k in charged]))
            dist = {k: float(np.median(tree.query(X[mesh.boundary_nodes(k)])[0]))
                    for k in far}
            k = min(dist, key=dist.get)
            dmin = dist[k] / float(delta)
            checks.append(_chk("far field", dmin, 20.0, "warn" if dmin < 20 else "pass",
                               f"median distance of bulk boundary '{k}' from the charged "
                               f"walls, in units of delta",
                               "enlarge the box" if dmin < 20 else ""))
    elif charged and n_species == 0:
        checks.append(_chk("EDL resolution", None, None, "info",
                           "no mobile species: no diffuse layer to resolve"))
    elif charged:
        checks.append(_chk("EDL resolution", None, None, "info",
                           "delta unknown: EDL resolution not checked"))

    # ---- size budget ------------------------------------------------------
    est = ndof is None
    if est:
        ndof = 5 * mesh.n_nodes + mesh.n_vertices
    checks.append(_chk("size budget (dofs)", ndof, 2e6, "warn" if ndof > 2e6 else "pass",
                       "number of unknowns" + (" (estimate: 2 species + flow)" if est else ""),
                       "coarsen the far field" if ndof > 2e6 else ""))

    if folder:
        os.makedirs(os.path.join(folder, "mesh"), exist_ok=True)
        with open(os.path.join(folder, "mesh", "diagnose.json"), "w") as f:
            json.dump({"checks": checks}, f, indent=1)
    return checks
