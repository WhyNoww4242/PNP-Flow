"""pnpflow vs Pawel's DNS for the four pore geometries (figureDNS4 layout).

Reads results/pore_dns_<case>/profiles.json (or the _coarse fallback) and the DNS line
files (transportFolder/poreRadius<in>_<out>/<t>/line*.xy, as read by pawel.py) and writes
  results/pore_dns_compare/figures/figureDNS4.png, svg/figureDNS4.svg
  results/pore_dns_compare/csv/dns_vs_pnpflow.csv   (every saved time, every line)
  results/pore_dns_compare/csv/dns_vs_pnpflow_error.csv   (max |pnpflow - DNS|)

Layout (user's figureDNS4): row 1 centreline mu(Z) at t = 0.01 s and 0.025 s (SDL at Z < 0),
row 2 Phi vs R/alpha at Z = 0.5; pnpflow = lines, DNS = squares (every 50th point).
DNS units follow pawel.py: Phi = potential / (kB T / e) with its rounded constants,
rho = n+ - n- (files are c / c0), mu = rho + 2 Phi, Z = x * 1e5 + 1, R/alpha from y
(wall at y.min, axis at y.max).  pnpflow values are in its own (CODATA) units.

Usage: python tools/figure_dns4.py [--dns <transportFolder>] [--times 0.01 0.025]
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
import cmasher as cmr  # noqa: E402

from pnpflow import figures as F  # noqa: E402

DNS_DEFAULT = "/mnt/c/Users/Gnome/Documents/Pore Charging Comparison/transportFolder"
CASES = [("narrow", "100_100", "Narrow (10 nm)"),
         ("converging", "200_100", "Converging (20 to 10 nm)"),
         ("diverging", "100_200", "Diverging (10 to 20 nm)"),
         ("wide", "200_200", "Wide (20 nm)")]
KB, E, T = 1.38e-23, 1.6022e-19, 298.15          # pawel.py constants
VT_PAWEL = KB * T / E
OUT = os.path.join(ROOT, "results", "pore_dns_compare")


def run_folder(case):
    """results/pore_dns_<case>[_coarse] (queue script), else the newest finished run of
    the case cases/pore_dns_<case>[_coarse] (runs started from the app)."""
    for name in (f"pore_dns_{case}", f"pore_dns_{case}_coarse"):
        f = os.path.join(ROOT, "results", name)
        if os.path.exists(os.path.join(f, "profiles.json")):
            return f
    for name in (f"pore_dns_{case}", f"pore_dns_{case}_coarse"):
        rdir = os.path.join(ROOT, "cases", name, "results")
        if not os.path.isdir(rdir):
            continue
        runs = [os.path.join(rdir, r) for r in os.listdir(rdir)
                if os.path.exists(os.path.join(rdir, r, "profiles.json"))]
        if runs:
            return max(runs, key=lambda f: os.path.getmtime(os.path.join(f, "profiles.json")))
    return None


def load_pnpflow(folder):
    """{t_s (rounded): {"axial": line, "radial": line}}."""
    lines = json.load(open(os.path.join(folder, "profiles.json")))["lines"]
    out = {}
    for ln in lines:
        meta = ln.get("time")
        if not meta:
            continue
        key = round(meta["t_s"], 6)
        kind = "axial" if ln["name"].startswith("axial") else "radial"
        out.setdefault(key, {"tau": meta["tau"]})[kind] = ln
    return out


def load_dns_line(path):
    a = np.loadtxt(path)
    x, y, npl, nmi, pot = a[:, 0], a[:, 1], a[:, 3], a[:, 4], a[:, 5]
    phi = pot / VT_PAWEL
    rho = npl - nmi
    return {"Z": x * 1e5 + 1, "R": (y.max() - y) / (y.max() - y.min()) if np.ptp(y) else y,
            "phi": phi, "rho": rho, "mu": rho + 2 * phi}


def dns_time_dir(folder, t_s):
    for d in os.listdir(folder):
        try:
            if abs(float(d) - t_s) < 1e-9:
                return os.path.join(folder, d)
        except ValueError:
            pass
    return None


def compare(dns_root):
    """Rows for the combined CSV and the error table, over every common time."""
    rows, errs, data = [], [], {}
    for case, folder_tag, label in CASES:
        run = run_folder(case)
        if run is None:
            print(f"  {case}: no pnpflow run yet, skipped")
            continue
        pn = load_pnpflow(run)
        dns_dir = os.path.join(dns_root, f"poreRadius{folder_tag}")
        data[case] = {"run": run, "pn": pn, "dns": {}}
        for t_s in sorted(pn):
            rec = pn[t_s]
            if "axial" not in rec or "radial" not in rec:
                continue
            ax, rd = rec["axial"], rec["radial"]
            Zp, Rp = np.asarray(ax["coord"]), np.asarray(rd["coord"])
            td = dns_time_dir(dns_dir, t_s) if os.path.isdir(dns_dir) else None
            dc = dr = None
            if td:
                dc = load_dns_line(os.path.join(td, "lineYe_nPlus_nMinus_ePhi.xy"))
                dr = load_dns_line(os.path.join(td, "lineXm5_nPlus_nMinus_ePhi.xy"))
                data[case]["dns"][t_s] = (dc, dr)
            base = {"case": case, "dns_folder": f"poreRadius{folder_tag}",
                    "t_s": t_s, "tau": rec["tau"]}
            # pnpflow lines on their own grid
            for k, v in (("mu_centreline", ax["values"]["mu_axis"]),
                         ("mu_cross_section_avg", ax["values"]["mu"])):
                for z, val in zip(Zp, v):
                    rows.append(dict(base, line=k, coord_name="Z", coord=z,
                                     source="pnpflow", value=val))
            for k in ("phi", "rho", "mu"):
                for r, val in zip(Rp, rd["values"][k]):
                    rows.append(dict(base, line=f"{k}_Z0.5", coord_name="R/alpha",
                                     coord=r, source="pnpflow", value=val))
            if dc is None:
                continue
            # DNS points + pnpflow interpolated there
            pairs = [("mu_centreline", "Z", dc["Z"], dc["mu"], Zp,
                      np.asarray(ax["values"]["mu_axis"]))]
            order = np.argsort(dr["R"])
            for k in ("phi", "rho", "mu"):
                pairs.append((f"{k}_Z0.5", "R/alpha", dr["R"][order], dr[k][order], Rp,
                              np.asarray(rd["values"][k])))
            for name, cn, xd, vd, xp, vp in pairs:
                inside = (xd >= xp.min()) & (xd <= xp.max())
                vi = np.interp(xd[inside], xp, vp)
                for x_, a_, b_ in zip(xd[inside], vd[inside], vi):
                    rows.append(dict(base, line=name, coord_name=cn, coord=x_,
                                     source="dns", value=a_, pnpflow_interp=b_))
                d = np.abs(vi - vd[inside])
                pore = xd[inside] >= 0 if cn == "Z" else np.ones_like(d, bool)
                errs.append(dict(base, line=name,
                                 max_abs_diff=float(d[pore].max()) if pore.any() else np.nan,
                                 where=float(xd[inside][pore][np.argmax(d[pore])])
                                 if pore.any() else np.nan,
                                 max_abs_diff_incl_sdl=float(d.max()),
                                 n_points=int(inside.sum())))
    return rows, errs, data


def write_csv(path, rows, fields):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def nearest(keys, t):
    keys = np.array(sorted(keys))
    return float(keys[np.argmin(np.abs(keys - t))])


def figure(data, times_s):
    cm = cmr.get_sub_cmap("cmr.bubblegum_r", 0.1, 0.9)
    colors = [cm(i / (len(CASES) - 1)) for i in range(len(CASES))]
    with F.house_style():
        fig = plt.figure(figsize=(13, 9))
        gs = fig.add_gridspec(2, len(times_s), left=0.1, right=0.74, wspace=0.42,
                              hspace=0.55)
        axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(len(times_s))]
                         for i in range(2)])
        for j, t_req in enumerate(times_s):
            ax_mu, ax_phi = axes[0, j], axes[1, j]
            tau_shown = None
            for (case, _, _), col in zip(CASES, colors):
                if case not in data:
                    continue
                pn = data[case]["pn"]
                t_s = nearest([k for k in pn if "axial" in pn[k]], t_req)
                rec = pn[t_s]
                tau_shown = rec["tau"]
                Z = np.asarray(rec["axial"]["coord"])
                ax_mu.plot(Z, rec["axial"]["values"]["mu_axis"], color=col)
                R = np.asarray(rec["radial"]["coord"])
                ax_phi.plot(R, rec["radial"]["values"]["phi"], color=col)
                if t_s in data[case]["dns"]:
                    dc, dr = data[case]["dns"][t_s]
                    ax_mu.scatter(dc["Z"][::50], dc["mu"][::50], s=25, color=col,
                                  marker="s", zorder=3)
                    ax_phi.scatter(dr["R"][::50], dr["phi"][::50], s=25, color=col,
                                   marker="s", zorder=3)
            ax_mu.set_xlabel(r"$Z$")
            ax_mu.set_ylabel(r"$\hat{\mu}$", rotation=0, labelpad=20)
            ax_mu.yaxis.set_major_locator(MaxNLocator(nbins=5))
            if tau_shown is not None:
                ax_mu.set_title(rf"$\tau = {tau_shown:.2g}$", pad=20)
            ax_phi.set_xlabel(r"$R/\alpha(Z)$")
            ax_phi.set_ylabel(r"$\hat{\Phi}$", rotation=0, labelpad=20)
            ax_phi.yaxis.set_major_locator(MaxNLocator(nbins=5))
            for a in (ax_mu, ax_phi):
                a.tick_params(labelsize=18)
                F.arrow_axes(a)
        handles = [Line2D([], [], color=c, lw=2.5, label=lab)
                   for (case, _, lab), c in zip(CASES, colors) if case in data]
        handles += [Line2D([], [], color="0.35", lw=2.5, label="pnpflow"),
                    Line2D([], [], color="0.35", lw=0, marker="s", markersize=6,
                           label="DNS")]
        fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.76, 0.5),
                   frameon=False, fontsize=18)
        F.add_subplot_labels(axes)
        F.save(fig, OUT, "figureDNS4")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dns", default=DNS_DEFAULT)
    ap.add_argument("--times", type=float, nargs="+", default=[0.01, 0.025],
                    help="DNS times in seconds (figureDNS4 default 0.01 0.025)")
    a = ap.parse_args()
    rows, errs, data = compare(a.dns)
    if not data:
        sys.exit("no pore_dns_* runs found")
    write_csv(os.path.join(OUT, "csv", "dns_vs_pnpflow.csv"), rows,
              ["case", "dns_folder", "t_s", "tau", "line", "coord_name", "coord", "source",
               "value", "pnpflow_interp"])
    write_csv(os.path.join(OUT, "csv", "dns_vs_pnpflow_error.csv"), errs,
              ["case", "dns_folder", "t_s", "tau", "line", "max_abs_diff", "where",
               "max_abs_diff_incl_sdl", "n_points"])
    figure(data, a.times)
    print(f"wrote {OUT}/figures/figureDNS4.png, csv/dns_vs_pnpflow*.csv")
    for e in errs:
        print(f"  {e['case']:10s} t = {e['t_s']:.3f} s  {e['line']:14s} "
              f"max |pnpflow - DNS| = {e['max_abs_diff']:.3g} (at {e['where']:.3g})")


if __name__ == "__main__":
    main()
