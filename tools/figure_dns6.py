"""pnpflow vs Pawel's DNS, 3 x 2 figure (figureDNS6) for the tau053 runs.

Columns: tau = 0.13 and 0.53 (DNS t = 0.01 s, 0.04 s; tau = 0.1337, 0.5348).
Rows: a), b) centreline mu = rho + 2 Phi vs Z (SDL included, Z from -0.5 to 1);
      c), d) Phi at Z = 0.5 vs R/alpha;   e), f) rho at Z = 0.5 vs R/alpha.
pnpflow = lines, DNS = squares (every 50th point). One cmr.cosmic colour per geometry
(diverging lightest -> converging darkest); legend on the right with a pore schematic.

Runs are picked explicitly: from the queue status (tools/run_dns_batch.py) or --run.
Writes results/pore_dns_compare/figures/figureDNS6.png, svg/figureDNS6.svg and
csv/dns_vs_pnpflow_tau053.csv, csv/dns_vs_pnpflow_tau053_error.csv (mu, phi, rho).
Extreme cases (r20_200, r200_10, r200_1; DNS to t = 0.035 s only):
    python tools/figure_dns6.py --status results/dns_extreme_status.json --times 0.01 0.035 \
        --name figureDNS6_extreme        # -> csv/dns_vs_pnpflow_tau053_extreme*.csv

    python tools/figure_dns6.py --status results/dns_tau053_status.json --dns DNS_compare/transportFolder
    python tools/figure_dns6.py --run converging=cases/pore_dns_converging_coarse/results/<run> ...
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Polygon  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import cmasher as cmr  # noqa: E402

import figure_dns4 as D4  # noqa: E402
from pnpflow import figures as F  # noqa: E402

GEOM = {"narrow": ("Narrow", 1, 1), "converging": ("Converging", 2, 1),
        "diverging": ("Diverging", 1, 2), "wide": ("Wide", 2, 2),
        "r20_200": ("Diverging 0.2 to 2", 0.2, 2), "r200_10": ("Converging 2 to 0.1", 2, 0.1),
        "r200_1": ("Converging 2 to 0.01", 2, 0.01)}
SHADE = ["diverging", "wide", "narrow", "converging"]     # lightest -> darkest
SHADE_EXTREME = ["r20_200", "r200_10", "r200_1"]
ORDER = ["narrow", "converging", "diverging", "wide", "r20_200", "r200_10", "r200_1"]
OUT = D4.OUT


def colours(cases=None):
    cm = cmr.get_sub_cmap("cmr.cosmic", 0.2, 0.9)
    col = {}
    for shade in (SHADE, SHADE_EXTREME):
        n = len(shade)
        col.update({c: cm(1 - i / (n - 1)) for i, c in enumerate(shade)})
    return col


def pick_runs(status_path, run_args):
    runs = {}
    if status_path:
        st = json.load(open(status_path))
        for case, cs in st.get("cases", {}).items():
            if cs.get("result") == "ok" and cs.get("run"):
                runs[case] = os.path.join(ROOT, cs["run"])
    for item in run_args or []:
        case, path = item.split("=", 1)
        runs[case] = os.path.abspath(path)
    return runs


def schematic(ax, x0, y0, w, h, a_in, a_out, colour="0.6"):
    """Small grey pore outline: mouth (a_in) on the left, closed end (a_out) on the right."""
    s = h / 2 / 2.0                 # radius 2 fills the half height
    pts = [(x0, y0 + a_in * s), (x0 + w, y0 + a_out * s),
           (x0 + w, y0 - a_out * s), (x0, y0 - a_in * s)]
    ax.add_patch(Polygon(pts, closed=True, facecolor=colour, edgecolor="none"))


def figure(data, times_s, name="figureDNS6"):
    col = colours()
    order = [c for c in ORDER if c in data]
    with F.house_style():
        fig = plt.figure(figsize=(14, 15))
        gs = fig.add_gridspec(3, len(times_s), left=0.1, right=0.68, top=0.94, bottom=0.06,
                              wspace=0.42, hspace=0.6)
        axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(len(times_s))]
                         for i in range(3)])
        for j, t_req in enumerate(times_s):
            ax_mu, ax_phi, ax_rho = axes[:, j]
            tau_title = None
            for case in order:
                pn = data[case]["pn"]
                keys = [k for k in pn if "axial" in pn[k] and abs(k - t_req) < 1e-6]
                if not keys:
                    print(f"  {case}: no pnpflow output at t = {t_req} s, skipped")
                    continue
                t_s, c = keys[0], col[case]
                rec = pn[t_s]
                tau_title = t_req * 1.337e-9 / (10e-6) ** 2
                Z = np.asarray(rec["axial"]["coord"])
                keep = Z >= -0.5
                ax_mu.plot(Z[keep], np.asarray(rec["axial"]["values"]["mu_axis"])[keep],
                           color=c, lw=2.5)
                R = np.asarray(rec["radial"]["coord"])
                ax_phi.plot(R, rec["radial"]["values"]["phi"], color=c, lw=2.5)
                ax_rho.plot(R, rec["radial"]["values"]["rho"], color=c, lw=2.5)
                if t_s in data[case]["dns"]:
                    dc, dr = data[case]["dns"][t_s]
                    m = dc["Z"] >= -0.5
                    kw = dict(s=28, color=c, marker="s", zorder=3, linewidths=0)
                    ax_mu.scatter(dc["Z"][m][::50], dc["mu"][m][::50], **kw)
                    ax_phi.scatter(dr["R"][::50], dr["phi"][::50], **kw)
                    ax_rho.scatter(dr["R"][::50], dr["rho"][::50], **kw)
            ax_mu.set_xlim(-0.5, 1.0)
            ax_mu.set_xlabel(r"$Z$")
            ax_mu.set_ylabel(r"$\hat{\mu}$", rotation=0, labelpad=22)
            if tau_title is not None:
                ax_mu.set_title(rf"$\tau = {tau_title:.2f}$", pad=22)
            for a, lab in ((ax_phi, r"$\hat{\Phi}$"), (ax_rho, r"$\hat{\rho}$")):
                a.set_xlim(0, 1)
                a.set_xlabel(r"$R/\alpha(Z)$")
                a.set_ylabel(lab, rotation=0, labelpad=22)
            for a in (ax_mu, ax_phi, ax_rho):
                a.yaxis.set_major_locator(MaxNLocator(nbins=5))
                a.xaxis.set_major_locator(MaxNLocator(nbins=4))
                a.tick_params(labelsize=18)
                F.arrow_axes(a)
        F.add_subplot_labels(axes)
        # legend: per geometry "<G> pnpflow" (line) and "<G> DNS" (square) + schematic
        lax = fig.add_axes([0.725, 0.25, 0.27, 0.5])
        lax.set_xlim(0, 1)
        lax.set_ylim(0, 1)
        lax.axis("off")
        n = len(order)
        for i, case in enumerate(order):
            label, a_in, a_out = GEOM[case]
            y = 1 - (i + 0.5) / max(n, 1)
            c = col[case]
            schematic(lax, 0.0, y, 0.14, 0.12, a_in, a_out)
            lax.add_line(Line2D([0.2, 0.32], [y + 0.04] * 2, color=c, lw=2.5))
            lax.scatter([0.26], [y - 0.04], s=40, color=c, marker="s", linewidths=0)
            lax.text(0.36, y + 0.04, f"{label} pnpflow", va="center", fontsize=17)
            lax.text(0.36, y - 0.04, f"{label} DNS", va="center", fontsize=17)
        F.save(fig, OUT, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", help="results/dns_tau053_status.json from run_dns_batch.py")
    ap.add_argument("--run", action="append", metavar="CASE=RUN_DIR",
                    help="explicit run folder per case (overrides --status)")
    ap.add_argument("--dns", default=os.path.join(ROOT, "DNS_compare", "transportFolder"))
    ap.add_argument("--times", type=float, nargs="+", default=[0.01, 0.04],
                    help="DNS times in seconds (default 0.01 0.04 = tau 0.13, 0.53)")
    ap.add_argument("--name", default="figureDNS6")
    a = ap.parse_args()
    runs = pick_runs(a.status, a.run)
    if not runs:
        sys.exit("no runs: pass --status or --run CASE=RUN_DIR")
    for k, v in runs.items():
        print(f"  {k}: {os.path.relpath(v, ROOT)}")
    D4.run_folder = lambda case: runs.get(case)        # explicit runs, not "newest"
    rows, errs, data = D4.compare(a.dns)
    errs = [e for e in errs if any(abs(e["t_s"] - t) < 1e-6 for t in a.times)] or errs
    tag = a.name.replace("figureDNS6", "")          # "" for the main figure, e.g. "_extreme"
    D4.write_csv(os.path.join(OUT, "csv", f"dns_vs_pnpflow_tau053{tag}.csv"), rows,
                 ["case", "dns_folder", "t_s", "tau", "line", "coord_name", "coord", "source",
                  "value", "pnpflow_interp"])
    D4.write_csv(os.path.join(OUT, "csv", f"dns_vs_pnpflow_tau053{tag}_error.csv"), errs,
                 ["case", "dns_folder", "t_s", "tau", "line", "max_abs_diff", "where",
                  "max_abs_diff_incl_sdl", "n_points"])
    figure(data, a.times, a.name)
    print(f"wrote {OUT}/figures/{a.name}.png, svg/{a.name}.svg, csv/dns_vs_pnpflow_tau053{tag}*.csv")
    for e in errs:
        print(f"  {e['case']:10s} t = {e['t_s']:.3f} s  {e['line']:14s} "
              f"max |pnpflow - DNS| = {e['max_abs_diff']:.3g} (at {e['where']:.3g})")


if __name__ == "__main__":
    main()
