"""House-style figures for the pore_charging case (AGENTS.md §9).

pore_profiles   mu(Z), Phi(R/alpha), rho(R/alpha) at the early and intermediate
                times (layout of the paper's Fig. 3, one geometry)
pore_evolution  the same three profiles at every output time
pore_charging   pore-averaged mu and mu(Z = 0.5) against tau
time_steps      BDF2 step size and Newton iterations against tau
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LogNorm

from .figures import add_subplot_labels, arrow_axes, cmap, render, save, tex

YLAB = {"mu": r"$\hat\mu$", "phi": r"$\hat\Phi$", "rho": r"$\hat\rho$"}
XLAB = {"Z": r"$Z$", "R/alpha": r"$R/\alpha(Z)$"}


def _by_tau(lines, kind):
    out = {}
    for ln in lines:
        if ln["name"].startswith(kind) and ln.get("time", {}).get("tau", 0) > 0:
            out[ln["time"]["tau"]] = ln
    return dict(sorted(out.items()))


def _nearest(d, tau):
    keys = np.array(list(d))
    return float(keys[np.argmin(np.abs(np.log(keys / tau)))])


def _tau_text(tau):
    return rf"$\tau = {tau:.3g}$"


def profiles_figure(folder, lines, tau_early, tau_mid):
    ax_l, rad_l = _by_tau(lines, "axial"), _by_tau(lines, "radial")
    if not ax_l or not rad_l:
        return None
    cols = [_nearest(ax_l, tau_early), _nearest(ax_l, tau_mid)]
    color = plt.rcParams["axes.prop_cycle"].by_key()["color"][1]
    fig, axes = plt.subplots(3, 2, figsize=(13.0, 12.5), layout="constrained")
    fig.get_layout_engine().set(hspace=0.08, wspace=0.08)
    rows = [("axial", "mu", ax_l), ("radial", "phi", rad_l), ("radial", "rho", rad_l)]
    for j, tau in enumerate(cols):
        for i, (kind, key, src) in enumerate(rows):
            ax = axes[i, j]
            ln = src[tau]
            x = np.asarray(ln["coord"], float)
            y = np.asarray(ln["values"][key], float)
            ax.plot(x, y, color=color)
            if kind == "axial":
                ax.axvline(0.0, color="0.6", ls="--", lw=1.2)
            ax.set_xlabel(tex(XLAB[ln["coord_name"]]))
            if j == 0:
                ax.set_ylabel(tex(YLAB[key]), rotation=0, labelpad=22)
            arrow_axes(ax)
        axes[0, j].set_title(tex(_tau_text(tau)), pad=20)
    # share y within a row so the two times compare directly
    for i in range(3):
        lo = min(axes[i, j].get_ylim()[0] for j in range(2))
        hi = max(axes[i, j].get_ylim()[1] for j in range(2))
        for j in range(2):
            axes[i, j].set_ylim(lo, hi)
    add_subplot_labels(axes.T, xytext=(-50, 6))
    return save(fig, folder, "pore_profiles")


def evolution_figure(folder, lines):
    ax_l, rad_l = _by_tau(lines, "axial"), _by_tau(lines, "radial")
    taus = [t for t in ax_l if t > 0]
    if len(taus) < 2:
        return None
    norm = LogNorm(vmin=min(taus), vmax=max(taus))
    cm = cmap("cosmic")
    fig, axes = plt.subplots(1, 3, figsize=(19.0, 5.8), layout="constrained")
    fig.get_layout_engine().set(wspace=0.06)
    for ax, (key, src) in zip(axes, [("mu", ax_l), ("phi", rad_l), ("rho", rad_l)]):
        for tau in taus:
            ln = src[tau]
            ax.plot(np.asarray(ln["coord"], float), np.asarray(ln["values"][key], float),
                    color=cm(0.1 + 0.8 * norm(tau)), lw=2.0)
        ax.set_xlabel(tex(XLAB[src[taus[0]]["coord_name"]]))
        ax.set_ylabel(tex(YLAB[key]), rotation=0, labelpad=22)
        if key == "mu":
            ax.axvline(0.0, color="0.6", ls="--", lw=1.2)
        arrow_axes(ax)
    axes[0].set_title("Along the pore", pad=20)
    axes[1].set_title(tex(r"At $Z = 0.5$"), pad=20)
    axes[2].set_title(tex(r"At $Z = 0.5$"), pad=20)
    sm = ScalarMappable(norm=LogNorm(vmin=min(taus), vmax=max(taus)),
                        cmap=cmap("cosmic").resampled(256))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, location="right", shrink=0.85, pad=0.02, aspect=25)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=18)
    cb.ax.set_title(tex(r"$\tau$"), fontsize=24, pad=12)
    add_subplot_labels(axes, xytext=(-50, 6))
    return save(fig, folder, "pore_evolution")


def charging_figure(folder, series):
    tau = np.asarray(series.get("tau", []), float)
    keep = tau > 0
    if keep.sum() < 2:
        return None
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, ax = plt.subplots(figsize=(8.5, 5.6), layout="constrained")
    ax.semilogx(tau[keep], np.asarray(series["mu_pore_avg"])[keep], "-o", ms=5,
                color=colors[0], label=tex(r"pore average"))
    ax.semilogx(tau[keep], np.asarray(series["mu_Z05"])[keep], "-s", ms=5,
                color=colors[2], label=tex(r"$Z = 0.5$"))
    ax.set_xlabel(tex(r"$\tau$"))
    ax.set_ylabel(tex(r"$\hat\mu$"), rotation=0, labelpad=22)
    ax.set_title("Charging of the pore", pad=20)
    ax.legend(frameon=False)
    arrow_axes(ax)
    return save(fig, folder, "pore_charging")


def steps_figure(folder, steps, t_per_tau):
    if len(steps) < 2:
        return None
    t = np.array([s["t"] for s in steps]) / t_per_tau
    dt = np.array([s["dt"] for s in steps]) / t_per_tau
    its = np.array([s["newton_its"] for s in steps])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 8.0), sharex=True, layout="constrained")
    axes[0].loglog(t, dt, "-", color=colors[0])
    axes[0].set_ylabel(tex(r"$\Delta\tau$"), rotation=0, labelpad=26)
    axes[0].set_title("BDF2 time steps", pad=20)
    axes[1].semilogx(t, its, "o", ms=4, color=colors[2])
    axes[1].set_ylabel("Newton its")
    axes[1].set_xlabel(tex(r"$\tau$"))
    axes[1].yaxis.get_major_locator().set_params(integer=True)
    for ax in axes:
        arrow_axes(ax)
    fig.align_ylabels(axes)
    add_subplot_labels(axes, xytext=(-60, 0), va="top")
    return save(fig, folder, "time_steps")


def write(folder, lines, summary, tau_early, tau_mid, log=print):
    names = []
    tr = summary.get("transient", {})
    jobs = [("pore_profiles", profiles_figure, (folder, lines, tau_early, tau_mid)),
            ("pore_evolution", evolution_figure, (folder, lines)),
            ("pore_charging", charging_figure,
             (folder, summary.get("integrals", {}).get("charging", {}))),
            ("time_steps", steps_figure, (folder, tr.get("steps", []), tr.get("t_per_tau", 1.0)))]
    for label, fn, args in jobs:
        try:
            out = render(fn, *args, log=log)
            if out:
                names.append(out)
        except Exception as exc:
            log(f"  figure {label} skipped: {type(exc).__name__}: {exc}")
    return names
