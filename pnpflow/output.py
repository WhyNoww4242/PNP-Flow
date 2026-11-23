"""Writing results: JSON for the HUD, .vtu / .msh for ParaView / gmsh, PNG plots."""
import json
import os
import time

import numpy as np


def _round(a, digits=6):
    """Round to `digits` significant figures; NaN -> None."""
    a = np.asarray(a, dtype=float)
    out = []
    for v in a.ravel():
        out.append(None if not np.isfinite(v) else float(f"{v:.{digits}g}"))
    return out


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, separators=(",", ":"), default=_default)
    os.replace(tmp, path)


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return float(o)


class Progress:
    """Rewrites progress.json as the run goes.

    Live-status keys (docs/DATA_FORMAT.md, docs/MCP.md section 6):
      fraction : 0-1 through the continuation targets, from completed targets
                 plus the log-residual progress of the current Newton solve
                 toward its tolerance; None when the solver gave no status
      health   : converging | struggling | stalled | failed | done
      tells    : {res_norm, ratio, alpha, min_c, halvings}
    """

    STALL_WINDOW = 3        # iterations
    STALL_DROP = 0.9        # stalled if |R_k| > 0.9 |R_{k-3}|
    LOW_ALPHA = 0.25

    def __init__(self, folder):
        self.path = os.path.join(folder, "progress.json")
        self.t0 = time.time()
        self.failed = False
        self.data = {"state": "meshing", "stage": 0, "n_stages": 0,
                     "iteration": 0, "res_norm": None, "message": "",
                     "fraction": None, "health": None,
                     "tells": {"res_norm": None, "ratio": None, "alpha": None,
                               "min_c": None, "halvings": 0},
                     "history": []}
        self.write()

    def set(self, **kw):
        state = kw.get("state")
        if state == "failed":
            self.failed = True
            kw.setdefault("health", "failed")
        elif state == "done":
            if self.failed:
                kw.setdefault("health", "failed")
            else:
                kw.setdefault("health", "done")
                kw.setdefault("fraction", 1.0)
        self.data.update(kw)
        self.write()

    def iteration(self, stage, history, status=None):
        last = history[-1]
        if status is not None and status.get("outcome"):
            # end-of-solve event: history already holds this iteration
            self._status(history, status)
            if status["outcome"] == "failed":
                self.failed = True
                msg = f"stage {stage}: Newton failed, giving up"
            else:
                msg = f"stage {stage}: Newton failed, halving the step"
            self.set(state="solving", message=msg)
            return
        self.data["history"].append({"stage": stage, **last})
        if status is not None:
            self._status(history, status)
        self.set(state="solving", stage=stage, iteration=last["it"],
                 res_norm=last["res_norm"],
                 message=f"stage {stage}, Newton it {last['it']}")

    def _status(self, history, st):
        res = [h["res_norm"] for h in history]
        last = history[-1]
        ratio = res[-1] / res[-2] if len(res) > 1 and res[-2] > 0 else None
        alpha = float(last["alpha"]) if last["it"] > 0 else None

        # fraction: completed targets + progress inside the current one
        r0, tol = st["r0"], st["tol_eff"]
        if res[-1] <= tol or r0 <= tol:
            newton = 1.0
        else:
            newton = np.log(r0 / res[-1]) / np.log(r0 / tol)
            newton = float(min(1.0, max(0.0, newton)))
        inner = st["frac_done"] + (st["trial"] - st["frac_done"]) * newton
        fraction = (st["target"] + inner) / max(1, st["n_targets"])

        k = self.STALL_WINDOW
        if st["outcome"] == "failed":
            health = "failed"
        elif len(res) > k and res[-1] > self.STALL_DROP * res[-1 - k]:
            health = "stalled"
        elif (st["outcome"] == "attempt_failed" or st["halvings"] > 0
              or (alpha is not None and alpha < self.LOW_ALPHA)):
            health = "struggling"
        else:
            health = "converging"
        self.data["fraction"] = float(min(1.0, max(0.0, fraction)))
        self.data["health"] = health
        self.data["tells"] = {"res_norm": float(res[-1]), "ratio": ratio,
                              "alpha": alpha, "min_c": st["min_c"],
                              "halvings": int(st["halvings"])}

    def write(self):
        self.data["time_s"] = time.time() - self.t0
        write_json(self.path, self.data)


# ----------------------------------------------------------------------
# Fields for the HUD
# ----------------------------------------------------------------------

COORD_NAMES = {"planar": ("y", "z"), "axisym": ("r", "z")}


def fields_json(mesh, fields, focus_bbox=None, labels=None):
    X = mesh.X
    edges = {}
    for name, b in mesh.boundaries.items():
        e = b["edges"]
        edges[name] = np.concatenate([e[:, [0, 2]], e[:, [2, 1]]]).tolist()
    vectors = {"u": ["uy", "uz"], "E": ["Ey", "Ez"], "j": ["j_y", "j_z"]}
    for k in list(fields):
        if k.startswith("N_") and k.endswith("_y"):
            vectors[k[:-2]] = [k, k[:-2] + "_z"]
    vectors = {k: v for k, v in vectors.items() if v[0] in fields}
    bbox = [float(X[:, 0].min()), float(X[:, 0].max()),
            float(X[:, 1].min()), float(X[:, 1].max())]
    coord = getattr(mesh, "coord", "planar")
    return {
        "symmetry": coord,
        "coord_names": list(COORD_NAMES[coord]),
        "nodes": np.asarray(_round(X, 7)).reshape(-1, 2).tolist(),
        "triangles": mesh.linear_triangles().tolist(),
        "boundary_edges": edges,
        "fields": {k: _round(v) for k, v in fields.items()},
        "vectors": vectors,
        "labels": labels or {},
        "bbox": bbox,
        "focus_bbox": focus_bbox or bbox,
    }


def write_fields(folder, mesh, fields, focus_bbox=None, labels=None):
    data = fields_json(mesh, fields, focus_bbox, labels)
    write_json(os.path.join(folder, "fields.json"), data)


def write_fields_time(folder, model, snapshots, t_per_tau=1.0, digits=5):
    """fields_time.json: the primary fields (each species, rho, phi) at every output
    time, one value per P2 node like fields.json, for the app's time slider.
    snapshots = [(t, U)]; the unknown layout is [c_1..c_S, phi, ...] in blocks of n."""
    n, S = model.n, len(model.species)
    fields = {}
    for i, sp in enumerate(model.species):
        fields[sp.name] = [_round(U[i * n:(i + 1) * n], digits) for _, U in snapshots]
    if S:
        fields["rho"] = [_round(sum(sp.z * np.asarray(U[i * n:(i + 1) * n])
                                    for i, sp in enumerate(model.species)), digits)
                         for _, U in snapshots]
    fields["phi"] = [_round(U[S * n:(S + 1) * n], digits) for _, U in snapshots]
    t = [float(tt) for tt, _ in snapshots]
    write_json(os.path.join(folder, "fields_time.json"),
               {"t": t, "tau": [x / t_per_tau for x in t], "t_per_tau": t_per_tau,
                "n_nodes": int(n), "fields": fields})


# ----------------------------------------------------------------------
# ParaView / gmsh
# ----------------------------------------------------------------------

def write_meshfiles(folder, mesh, fields):
    import meshio
    points = np.column_stack([mesh.X, np.zeros(mesh.n_nodes)])
    cells = [("triangle6", mesh.tri)]
    data = {k: np.asarray(v, dtype=float) for k, v in fields.items()}
    meshio.write(os.path.join(folder, "solution.vtu"),
                 meshio.Mesh(points, cells, point_data=data))
    try:
        cell_data = {"gmsh:physical": [np.ones(mesh.n_elements, dtype=int)],
                     "gmsh:geometrical": [np.ones(mesh.n_elements, dtype=int)]}
        m = meshio.Mesh(points, cells, point_data=data, cell_data=cell_data)
        meshio.write(os.path.join(folder, "solution.msh"), m,
                     file_format="gmsh22", binary=False)
    except Exception as exc:          # gmsh output is a convenience only
        print("  (solution.msh skipped:", exc, ")")


# ----------------------------------------------------------------------
# Static plots
# ----------------------------------------------------------------------


def write_plots(folder, mesh, fields, profiles, focus_bbox=None):
    """Quick plots for the old HUD (plots/*.png), in the house look: cmasher maps
    (cosmic / ember / bubblegum, see figures.field_cmap), arrow axes, no boxes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation
    from . import figures as F

    out = os.path.join(folder, "plots")
    os.makedirs(out, exist_ok=True)
    tri = Triangulation(mesh.X[:, 0], mesh.X[:, 1], mesh.linear_triangles())
    tri_coarse = Triangulation(mesh.X[:, 0], mesh.X[:, 1], mesh.tri[:, :3])

    xname, yname = COORD_NAMES[getattr(mesh, "coord", "planar")]

    def frame(ax, bbox):
        if bbox:
            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
        ax.set_aspect("equal")
        ax.set_xlabel(xname)
        ax.set_ylabel(yname)
        F.arrow_axes(ax)

    with F.house_style(usetex=False):
        # mesh, full and zoom
        for tag, bbox in [("mesh", None), ("mesh_zoom", focus_bbox)]:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.triplot(tri_coarse, lw=0.2, color="0.35")
            frame(ax, bbox)
            ax.set_title(f"{mesh.n_elements} P2 triangles")
            fig.savefig(os.path.join(out, f"{tag}.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)

        for name in ["phi", "rho", "c+", "c-", "|u|", "p", "vorticity", "|E|",
                     "rho_f", "phi_err"]:
            if name not in fields:
                continue
            v = np.asarray(fields[name], dtype=float)
            fig, ax = plt.subplots(figsize=(7, 6))
            lim = float(np.nanmax(np.abs(v))) or 1.0
            lo, hi = float(np.nanmin(v)), float(np.nanmax(v))
            base = F.field_cmap(name)
            if hi <= 0.05 * lim and lo < 0:                 # all negative: magnitude brightest
                cs = ax.tripcolor(tri, v, shading="gouraud", cmap=F.cmap(base + "_r"),
                                  vmin=lo, vmax=0.0)
            else:
                cs = ax.tripcolor(tri, v, shading="gouraud", cmap=F.cmap(base))
            cb = fig.colorbar(cs, ax=ax, label=name)
            cb.outline.set_visible(False)
            if name == "|u|" and "uy" in fields:
                _streamlines(ax, mesh, fields, focus_bbox)
            frame(ax, focus_bbox)
            ax.set_title(name)
            safe = name.replace("|", "abs_").replace("+", "plus").replace("-", "minus")
            fig.savefig(os.path.join(out, f"field_{safe}.png"), dpi=130, bbox_inches="tight")
            plt.close(fig)

        for line in profiles.get("lines", []):
            keys = [k for k in line["values"] if k in ("c+", "c-", "phi", "phi_exact", "rho",
                                                       "|u|", "sigma", "shear", "p")]
            if not keys:
                continue
            fig, ax = plt.subplots(figsize=(7, 4))
            s = np.asarray(line["coord"])
            for k in keys:
                ax.plot(s, np.asarray(line["values"][k], dtype=float), label=k,
                        ls="--" if k.endswith("_exact") else "-")
            ax.set_xlabel(line["coord_name"])
            ax.set_title(line["label"])
            ax.grid(False)
            ax.legend(frameon=False)
            F.arrow_axes(ax)
            fig.savefig(os.path.join(out, f"profile_{line['name']}.png"), dpi=130,
                        bbox_inches="tight")
            plt.close(fig)


def _streamlines(ax, mesh, fields, bbox):
    from matplotlib.tri import LinearTriInterpolator, Triangulation
    if bbox is None:
        return
    tri = Triangulation(mesh.X[:, 0], mesh.X[:, 1], mesh.linear_triangles())
    gy, gz = np.meshgrid(np.linspace(bbox[0], bbox[1], 160),
                         np.linspace(bbox[2], bbox[3], 160))
    uy = LinearTriInterpolator(tri, fields["uy"])(gy, gz)
    uz = LinearTriInterpolator(tri, fields["uz"])(gy, gz)
    ax.streamplot(gy, gz, uy.filled(np.nan), uz.filled(np.nan), color="w",
                  linewidth=0.6, density=1.4, arrowsize=0.6)
