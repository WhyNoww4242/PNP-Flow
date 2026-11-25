"""Automatic house-style figures for a finished run (AGENTS.md §9).

write_run_figures() writes field maps, one figure per profile line and the
Newton convergence plot to  <run>/figures/<name>.png (300 dpi)  and
<run>/svg/<name>.svg.  numpy + matplotlib only (no jax).  A figure that
fails is skipped and reported; it never fails the run.
"""
import contextlib
import io
import os
import re

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.tri import Triangulation, LinearTriInterpolator  # noqa: E402
from matplotlib.lines import Line2D                              # noqa: E402

try:
    import cmasher as cmr
except ImportError:                                  # pragma: no cover
    cmr = None

DPI = 300
PAD_IN = 0.08            # saved margin (inches) around the tight bbox


# =====================================================================
# Style
# =====================================================================

def apply_custom_style(num_colors: int = 4, cmap: str = "cmr.cosmic_r", cmap_range: tuple = (0.2, 0.75)):
    cm = cmr.get_sub_cmap(cmap, *cmap_range)
    colors = [cm(i / (num_colors - 1)) for i in range(num_colors)]
    plt.rcParams.update({
        "axes.prop_cycle": plt.cycler(color=colors),
        "font.size": 24, "axes.labelsize": 24, "axes.titlesize": 24, "figure.titlesize": 24,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "times-roman", "text.usetex": True,
    })
    return colors


_LATEX_OK = None


def latex_ok():
    """True when text.usetex works here (tested once with a stacked fraction)."""
    global _LATEX_OK
    if _LATEX_OK is None:
        try:
            with plt.rc_context({"text.usetex": True, "font.family": "serif",
                                 "font.serif": ["Times"]}):
                fig = plt.figure(figsize=(1, 1))
                fig.text(0.1, 0.5, r"$\displaystyle\frac{c_+}{c_{+,\max}}$ ok")
                fig.savefig(io.BytesIO(), format="png", dpi=50)
                plt.close(fig)
            _LATEX_OK = True
        except Exception:
            plt.close("all")
            _LATEX_OK = False
    return _LATEX_OK


@contextlib.contextmanager
def house_style(usetex=None):
    """rc_context with the house style; global rcParams are restored on exit.

    The family name "times-roman" is not one matplotlib's TeX backend knows
    (it would silently use Computer Modern), so Times is selected explicitly
    as serif -> \\usepackage{mathptmx}.  Without LaTeX: mathtext + STIX.
    """
    if usetex is None:
        usetex = latex_ok()
    with plt.rc_context():
        colors = apply_custom_style() if cmr else None
        plt.rcParams.update({
            "font.family": "serif", "font.serif": ["Times", "Times New Roman",
                                                   "Nimbus Roman", "STIXGeneral"],
            "text.usetex": bool(usetex),
            "lines.linewidth": 2.5, "legend.frameon": False, "axes.grid": False,
            "xtick.labelsize": 20, "ytick.labelsize": 20, "legend.fontsize": 20,
            "axes.titlepad": 20, "savefig.dpi": DPI, "svg.fonttype": "path",
        })
        if not usetex:
            plt.rcParams.update({"mathtext.fontset": "stix",
                                 "font.serif": ["STIXGeneral", "Times New Roman",
                                                "Nimbus Roman", "DejaVu Serif"]})
        yield colors


def tex(s):
    """Label text for the active backend (mathtext has no \\displaystyle)."""
    if plt.rcParams["text.usetex"]:
        return s
    return s.replace(r"\displaystyle", "").replace(r"\quad", " ")


# Only three colormaps are used, all cmasher (never viridis or other matplotlib
# defaults, and no diverging maps):
#   cosmic    potential, concentrations, mesh quality (default)
#   ember     magnitudes |E|, |u|
#   bubblegum charge density rho, rho_f
CMAPS = ("cosmic", "ember", "bubblegum")
MAGNITUDES = ("|u|", "|E|", "vorticity")
CHARGES = ("rho", "rho_f")


def cmap(name):
    """A cmasher colormap by short name ('cosmic', 'ember', 'bubblegum', '<name>_r')."""
    assert name.removesuffix("_r") in CMAPS, f"colormap {name!r} is not one of {CMAPS}"
    return plt.get_cmap("cmr." + name) if cmr else plt.get_cmap("gray")


def field_cmap(name):
    """The colormap name for a field."""
    if name in MAGNITUDES:
        return "ember"
    if name in CHARGES:
        return "bubblegum"
    return "cosmic"


class _AxisArrow(Line2D):
    """Extension of the bottom (x) or left (y) axis line by `extend_pt` past the axes,
    with an arrowhead at the tip.  The tip is recomputed from the axes size on every
    draw, so it stays `extend_pt` long after layout; being a Line2D with clip_on=False,
    tight bboxes and constrained layout include it."""

    def __init__(self, ax, horizontal, extend_pt, size):
        super().__init__([0, 0], [0, 0], color="k", lw=1.2, clip_on=False, zorder=10,
                         marker=">" if horizontal else "^", markevery=[1], ms=size,
                         solid_capstyle="butt", transform=ax.transAxes)
        self._horizontal, self._extend_pt = horizontal, extend_pt

    def _update(self):
        bb = self.axes.get_window_extent()
        ext = self._extend_pt * self.figure.dpi / 72
        if self._horizontal:
            self.set_data([1, 1 + ext / max(bb.width, 1)], [0, 0])
        else:
            self.set_data([0, 0], [1, 1 + ext / max(bb.height, 1)])

    def draw(self, renderer):
        self._update()
        super().draw(renderer)

    def get_window_extent(self, renderer=None):
        self._update()
        return super().get_window_extent(renderer)


def arrow_axes(ax, extend_pt=14, size=8):
    """Only the left and bottom axis lines, each running `extend_pt` past the plot and
    ending in an arrowhead, so the arrow never sits on the last tick label (no box)."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(1.2)
    for horizontal in (True, False):
        ax.add_artist(_AxisArrow(ax, horizontal, extend_pt, size))


def add_subplot_labels(axes, fontsize=24, xytext=(-35, 10), va="baseline"):
    """Add labels a), b), ... to the upper left corner of each subplot."""
    axes = np.asarray(axes).ravel()
    for i, ax in enumerate(axes):
        ax.annotate(chr(97 + i) + ")", xy=(0, 1), xycoords="axes fraction",
                    xytext=xytext, textcoords="offset points", va=va, fontsize=fontsize)


def colorbar(fig, mappable, cax, label=None):
    """Colour bar without an outline box."""
    cb = fig.colorbar(mappable, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=20)
    if label is not None:
        cb.set_label(label, rotation=0, ha="left", va="center", fontsize=24, labelpad=10)
    return cb


def save(fig, folder, name):
    """figures/<name>.png (300 dpi) + svg/<name>.svg; returns name."""
    for sub, ext in (("figures", "png"), ("svg", "svg")):
        os.makedirs(os.path.join(folder, sub), exist_ok=True)
        fig.savefig(os.path.join(folder, sub, f"{name}.{ext}"), dpi=DPI,
                    bbox_inches="tight", pad_inches=PAD_IN)
    plt.close(fig)
    return name


def save_to(fig, path):
    """Save one PNG at 300 dpi with the house margins."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches="tight", pad_inches=PAD_IN)
    plt.close(fig)


def render(fn, *args, log=print, **kw):
    """Run a figure function in house style; retry without LaTeX on error."""
    for usetex in ([True, False] if latex_ok() else [False]):
        try:
            with house_style(usetex):
                return fn(*args, **kw)
        except Exception as exc:
            plt.close("all")
            err = exc
            if usetex:
                log(f"  figure {getattr(fn, '__name__', fn)}: LaTeX failed "
                    f"({type(exc).__name__}), retrying with mathtext")
    raise err


# =====================================================================
# Symbols and number formatting
# =====================================================================

def num(v, digits=3):
    """Number as TeX math (no $): 2.68, 1.2\\times10^{-3}."""
    v = float(v)
    if v == 0 or not np.isfinite(v):
        return "0" if v == 0 else r"\mathrm{nan}"
    e = int(np.floor(np.log10(abs(v))))
    if -2 <= e <= 3:
        return f"{v:.{digits}g}"
    m = v / 10 ** e
    return f"{m:.{max(digits - 1, 1)}f}" + r"\times10^{" + str(e) + "}"


def species_symbol(name):
    """(base, subscript) in TeX for a species name: 'c+' -> ('c', '+')."""
    m = re.fullmatch(r"c([+-]+)", name)
    if m:
        return "c", m.group(1)
    m = re.fullmatch(r"c_?(\w+)", name)
    if m:
        return "c", r"\mathrm{" + m.group(1).replace("_", r"\_") + "}"
    return r"\mathrm{" + name.replace("_", r"\_") + "}", ""


def _sub(base, sub, extra=""):
    parts = ",".join(p for p in (sub, extra) if p)
    return f"{base}_{{{parts}}}" if parts else base


FIELD_SYMBOL = {"phi": (r"\phi", ""), "rho": (r"\rho", ""),
                "|u|": (r"|\mathbf{u}|", ""), "|E|": (r"|\mathbf{E}|", ""),
                "p": ("p", ""), "uy": ("u", "y"), "uz": ("u", "z")}


def symbol(name, coord="planar"):
    if name in FIELD_SYMBOL:
        b, s = FIELD_SYMBOL[name]
        if coord == "axisym" and name == "uy":
            s = "r"
        return b, s
    return species_symbol(name)


def sym_tex(name, coord="planar"):
    return _sub(*symbol(name, coord))


def frac_label(name, signed=False, coord="planar", bulk=False):
    """Stacked-fraction colour-bar label, e.g. c_+ / c_{+,max} or c_+ / c_{+,bulk}."""
    b, s = symbol(name, coord)
    top = _sub(b, s)
    if bulk:
        den = r"c_{\mathrm{bulk}}" if name == "rho" else _sub("c", s, r"\mathrm{bulk}")
    elif signed or b.startswith("|"):
        den = (f"|{b}|" if not b.startswith("|") else b)
        den = f"{den}_{{{s + ',' if s else ''}\\max}}"
    else:
        den = _sub(b, s, r"\max")
    return rf"$\displaystyle\frac{{{top}}}{{{den}}}$"


def coord_names(coord):
    return ("r", "z") if coord == "axisym" else ("y", "z")


# =====================================================================
# Field maps
# =====================================================================

TITLES = {"phi": "Potential", "rho": "Charge density", "|u|": "Speed",
          "|E|": "Field strength"}


def _mirror(X, tris, values):
    """Mirror a triangulation about x = 0 (axisymmetric -> full section)."""
    n = len(X)
    Xm = np.vstack([X, X * [-1.0, 1.0]])
    tm = np.vstack([tris, tris[:, ::-1] + n])
    vm = [np.concatenate([v, v]) for v in values]
    return Xm, tm, vm


def _window(X, tris, bbox, margin=0.05):
    """Triangles with at least one node inside the (slightly grown) bbox."""
    x0, x1, y0, y1 = bbox
    dx, dy = margin * (x1 - x0), margin * (y1 - y0)
    inside = ((X[:, 0] >= x0 - dx) & (X[:, 0] <= x1 + dx)
              & (X[:, 1] >= y0 - dy) & (X[:, 1] <= y1 + dy))
    keep = inside[tris].any(axis=1)
    return tris[keep]


def _axes_size(bbox, width=6.0, max_h=7.0):
    """Axes (w, h) in inches for an equal-aspect map; aspect=False if extreme."""
    w_data, h_data = bbox[1] - bbox[0], bbox[3] - bbox[2]
    a = h_data / w_data
    if a > 8 or a < 1 / 8:
        return width, 3.0, False
    w, h = width, width * a
    if h > max_h:
        h, w = max_h, max_h / a
    return w, h, True


def field_map(folder, name, X, tris, v, bbox, coord="planar", vectors=None, bulk=None):
    """One field map, normalised colour bar with a stacked-fraction label.

    bulk: reference concentration for species and rho (values / bulk, labelled
    c / c_bulk); None normalises by the field's largest magnitude.
    """
    v = np.asarray(v, dtype=float)
    vmax_abs = float(np.nanmax(np.abs(v)))
    # A field gets a diverging map only when both signs are substantial (> 5 % of
    # the max); one-signed fields (e.g. rho in a single EDL) use a sequential map.
    signed = (np.nanmin(v) < -0.05 * vmax_abs and np.nanmax(v) > 0.05 * vmax_abs)
    vecs = list(vectors) if vectors is not None else None
    if coord == "axisym":
        X, tris, vals = _mirror(X, tris, [v] + (vecs or []))
        v = vals[0]
        if vecs:
            vals[1][len(vals[1]) // 2:] *= -1.0          # u_r(-r) = -u_r(r)
            vecs = vals[1:]
        r = max(abs(bbox[0]), abs(bbox[1]))
        bbox = [-r, r, bbox[2], bbox[3]]
    tris = _window(X, tris, bbox)
    tri = Triangulation(X[:, 0], X[:, 1], tris)
    ref = bulk if bulk else vmax_abs
    norm_v = v / ref if ref > 0 else v
    base = field_cmap(name)
    vmin, vmax = float(np.nanmin(norm_v)), float(np.nanmax(norm_v))
    if signed:
        cm = cmap(base)                  # sequential over the real range, no diverging map
    elif vmax <= 0.05:
        cm = cmap(base + "_r")           # all negative: largest magnitude brightest
        vmax = 0.0
    else:
        cm = cmap(base)
        vmin = 0.0 if vmin >= 0 else vmin
    if vmax - vmin < 1e-12:
        vmin, vmax = vmax - 0.5, vmax + 0.5

    w, h, equal = _axes_size(bbox)
    W = w + 2.85                               # room for the x arrow before the colour bar
    fig = plt.figure(figsize=(W, h + 1.4))
    ax = fig.add_axes([0.6 / W, 0.9 / (h + 1.4), w / W, h / (h + 1.4)])
    cs = ax.tripcolor(tri, norm_v, shading="gouraud", cmap=cm, vmin=vmin, vmax=vmax,
                      rasterized=True)
    if vecs is not None and name == "|u|":
        _streamlines(ax, tri, vecs, bbox)
    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.tick_params(axis="x", pad=10)          # keep the two corner zeros apart
    ax.tick_params(axis="y", pad=6)
    arrow_axes(ax)
    if equal:
        ax.set_aspect("equal")
    xn, yn = coord_names(coord)
    ax.set_xlabel(f"${xn}$")
    ax.set_ylabel(f"${yn}$")
    sym = sym_tex(name, coord)
    has_neg = bool(np.nanmin(v) < 0)             # signed or all-negative: label |x|_max
    maxname = (f"|{sym}|" if has_neg and not sym.startswith("|") else sym)
    if "_{" in maxname and not maxname.startswith("|"):
        maxname = maxname[:-1] + r",\max}"
    else:
        maxname = maxname + r"_{\max}"
    title = TITLES.get(name, "Concentration")
    if bulk:
        ax.set_title(tex(title), pad=20)
    else:
        ax.set_title(tex(f"{title}, ${maxname} = {num(vmax_abs)}$"), pad=20)
    cax = fig.add_axes([(0.6 + w + 0.5) / W, 0.9 / (h + 1.4), 0.22 / W, h / (h + 1.4)])
    colorbar(fig, cs, cax, tex(frac_label(name, has_neg, coord, bulk=bool(bulk))))
    return save(fig, folder, "field_" + safe_name(name))


def _streamlines(ax, tri, vecs, bbox):
    gy, gz = np.meshgrid(np.linspace(bbox[0], bbox[1], 160),
                         np.linspace(bbox[2], bbox[3], 160))
    uy = LinearTriInterpolator(tri, vecs[0])(gy, gz).filled(np.nan)
    uz = LinearTriInterpolator(tri, vecs[1])(gy, gz).filled(np.nan)
    if np.all(np.isnan(uy)) or np.nanmax(np.hypot(uy, uz)) == 0:
        return
    ax.streamplot(gy, gz, uy, uz, color="w", linewidth=0.6, density=1.2,
                  arrowsize=0.6)


def safe_name(name):
    if name.startswith("|") and name.endswith("|"):
        name = "abs_" + name[1:-1]
    return (name.replace("|", "abs_").replace("+", "plus").replace("-", "minus")
            .replace(" ", "_").replace("/", "_"))


# =====================================================================
# Profiles
# =====================================================================

AXIS_LABEL = {"theta_deg": r"$\theta$ (deg)", "r - 1": r"$r - 1$", "z - 1": r"$z - 1$",
              "-z - 1": r"$-z - 1$"}


def _panel_groups(values, species):
    """Split a profile's values into panels: concentrations, phi, flow, rest."""
    def present(keys):
        out = []
        for k in keys:
            if k in values and np.any(np.isfinite(np.asarray(values[k], dtype=float))):
                out.append(k)
        return out
    conc = present([s for s in species] + [s + "_exact" for s in species])
    pot = present(["phi", "phi_exact"])
    flow = []
    if "|u|" in values or any(k.startswith("u") and k.endswith("_exact") for k in values):
        u = np.asarray(values.get("|u|", [0]), dtype=float)
        if np.nanmax(np.abs(u), initial=0) > 1e-12 or any(k.endswith("_exact") and k.startswith("u")
                                                          for k in values):
            flow = present(["uy", "uz"] + [k for k in values
                                           if k.startswith("u") and k.endswith("_exact")])
    groups = [("conc", conc), ("phi", pot), ("flow", flow)]
    for k in ("sigma", "shear", "p"):
        if k in values:
            groups.append((k, present([k])))
    return [(g, ks) for g, ks in groups if ks]


YLABEL = {"conc": r"$c_i$", "phi": r"$\phi$", "flow": r"$u$", "sigma": r"$\sigma$",
          "shear": r"$\omega_w$", "p": r"$p$"}


def _key_label(k, coord):
    exact = k.endswith("_exact")
    base = k[:-6] if exact else k
    if base == "ux":
        base = "uy"
    s = sym_tex(base, coord)
    return f"${s}$" + (" exact" if exact else "")


def profile_figure(folder, line, species, coord="planar"):
    s = np.asarray(line["coord"], dtype=float)
    groups = _panel_groups(line["values"], species)
    if not groups or len(s) < 2:
        return None
    ds = np.diff(s)
    logx = s[0] >= 0 and ds[0] > 0 and ds[-1] / ds[0] > 20 and line["coord_name"] != "theta_deg"
    keep = s > 0 if logx else np.ones(len(s), bool)
    n = len(groups)
    fig, axes = plt.subplots(n, 1, figsize=(8.0, 2.9 * n + 0.9), sharex=True,
                             squeeze=False, layout="constrained")
    fig.get_layout_engine().set(hspace=0.06)
    axes = axes[:, 0]
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for i, (ax, (g, keys)) in enumerate(zip(axes, groups)):
        solid = [k for k in keys if not k.endswith("_exact")]
        idx = np.round(np.linspace(0, len(cycle) - 1, max(len(solid), 1))).astype(int)
        if len(solid) == 1:
            idx = [1]
        color = {k: cycle[j] for k, j in zip(solid, idx)}
        for k in keys:
            y = np.asarray(line["values"][k], dtype=float)
            exact = k.endswith("_exact")
            ax.plot(s[keep], y[keep], ls="--" if exact else "-",
                    color="k" if exact else color[k], lw=1.8 if exact else 2.5,
                    label=_key_label(k, coord), zorder=3 if exact else 2)
        ax.set_ylabel(YLABEL[g], labelpad=12)
        if logx:
            ax.set_xscale("log")
        if len(keys) > 1:
            ax.legend(frameon=False, fontsize=20, loc="best")
        ax.grid(False)
        arrow_axes(ax)
    xl = AXIS_LABEL.get(line["coord_name"], f"${line['coord_name']}$")
    axes[-1].set_xlabel(tex(xl))
    fig.align_ylabels(axes)
    if n > 1:
        add_subplot_labels(axes, xytext=(-60, 0), va="top")
    axes[0].set_title(tex(_title_text(line.get("label") or line["name"])), pad=20)
    return save(fig, folder, "profile_" + safe_name(line["name"]))


def _title_text(label):
    """Profile labels are plain text; protect TeX specials, keep it short."""
    label = label[0].upper() + label[1:] if label else label
    for a, b in (("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")):
        label = label.replace(a, b)
    label = label.replace("theta", r"$\theta$").replace(" deg", r"$^\circ$")
    label = re.sub(r"z ?([<>]) ?(-?\d+)", r"$z \1 \2$", label)
    return label


# =====================================================================
# Newton convergence
# =====================================================================

def convergence_figure(folder, stages):
    its = []
    for k, st in enumerate(stages):
        for rec in st.get("iterations", []):
            r = rec.get("res_norm")
            if r is not None and np.isfinite(r) and r > 0:
                its.append((k, r))
    if not its:
        return None
    fig, ax = plt.subplots(figsize=(8.0, 5.2), layout="constrained")
    x = 0
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for k in range(len(stages)):
        r = [v for kk, v in its if kk == k]
        if not r:
            continue
        xs = np.arange(x, x + len(r))
        ax.semilogy(xs, r, "-o", ms=6, color=colors[k % len(colors)])
        if k > 0:
            ax.axvline(x - 0.5, color="0.6", ls="--", lw=1.2)
        if len(stages) > 1:
            ax.text(x + 0.5 * (len(r) - 1), 1.0, f"{k + 1}",
                    transform=ax.get_xaxis_transform(), ha="center", va="bottom",
                    fontsize=18, color="0.35")
        x += len(r)
    multi = len(stages) > 1
    ax.set_xlabel("Newton iteration" + (" (all stages)" if multi else ""))
    ax.set_ylabel(r"$\|R\|$")
    ax.set_title("Newton convergence" + (", by continuation stage" if multi else ""),
                 pad=34 if multi else 20)
    ax.xaxis.get_major_locator().set_params(integer=True)
    ax.grid(False)
    arrow_axes(ax)
    return save(fig, folder, "convergence")


# =====================================================================
# Entry point
# =====================================================================

FIELD_ORDER = ["phi", "rho", "|u|", "|E|"]


def write_run_figures(folder, mesh, fields, profiles, summary, focus=None,
                      symmetry=None, log=print):
    """Write the standard figures of a run; returns the figure names written.

    fields   : nodal dict (as analysis.nodal_fields)
    profiles : {"lines": [...]} (as profiles.json)
    summary  : the run summary (model.species, model.coord, solver.stages)
    focus    : [x0, x1, y0, y1] window for field maps (default: whole mesh)
    symmetry : "axisym" mirrors field maps about r = 0
    """
    model = summary.get("model", {}) or {}
    coord = symmetry or model.get("coord") or getattr(mesh, "coord", "planar")
    species = [s["name"] for s in model.get("species", [])]
    # concentrations (and rho) are shown relative to the bulk; nondimensional
    # concentrations are already in units of c0, so rho's reference is 1
    bulk = {s["name"]: float(s.get("bulk", 1.0) or 1.0) for s in model.get("species", [])}
    bulk["rho"] = 1.0
    X = np.asarray(mesh.X, dtype=float)
    tris = mesh.linear_triangles()
    bbox = list(focus) if focus else [float(X[:, 0].min()), float(X[:, 0].max()),
                                      float(X[:, 1].min()), float(X[:, 1].max())]
    names = []

    def attempt(label, fn, *a, **kw):
        try:
            out = render(fn, *a, log=log, **kw)
            if out:
                names.append(out)
        except Exception as exc:
            log(f"  figure {label} skipped: {type(exc).__name__}: {exc}")

    for name in species + FIELD_ORDER:
        if name not in fields:
            continue
        v = np.asarray(fields[name], dtype=float)
        if not np.any(np.isfinite(v)) or np.nanmax(np.abs(v)) < 1e-14:
            continue                                # e.g. rho with no species
        vec = None
        if name == "|u|" and "uy" in fields and "uz" in fields:
            vec = [np.asarray(fields["uy"], float), np.asarray(fields["uz"], float)]
        attempt(name, field_map, folder, name, X, tris, v, bbox, coord, vec,
                bulk=bulk.get(name))

    for line in (profiles or {}).get("lines", []):
        attempt(line.get("name"), profile_figure, folder, line, species, coord)

    attempt("convergence", convergence_figure, folder,
            (summary.get("solver") or {}).get("stages", []))
    return names
