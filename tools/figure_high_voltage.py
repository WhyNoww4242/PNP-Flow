#!/usr/bin/env python
"""High_Voltage_Filipe: pnpflow version of Fig. 8 of Henrique, Zuk & Gupta, Soft Matter 18,
198 (2022), and an HTML page that puts it next to the paper's figure.

Panels (same layout as the paper):
  a) Psi_m / Psi_D vs Z at tau = 0.07      b) -rho_m / Psi_D vs Z at tau = 0.07
  c) Psi_m / Psi_D vs Z at tau = 0.15      d) -rho_m / Psi_D vs Z at tau = 0.15
  e) -J_right / Psi_D vs tau (charge flux into the pore at the mouth, D c0 / l_p units)
Psi_m, rho_m are on the centreline R = 0 (profiles.json `phi_axis`, `rho_axis`), Z from -0.5
(SDL far end) to 1 (pore end). J_right = 2 / (L alpha^2) dQ/dtau from the pore charge at
every accepted time step (summary.json integrals.mouth). pnpflow is drawn dashed, as the
paper's DNS.

Runs: for each potential the newest run in cases/High_Voltage_Filipe_<mV>mV/results/ that
finished (stdout.log ends with "converged: True" and t_reached = t_end), or --run.

    python tools/figure_high_voltage.py
    python tools/figure_high_voltage.py --run 200=cases/High_Voltage_Filipe_200mV/results/hv200_run1
Writes results/high_voltage_filipe/figures/figure8_pnpflow.png (+ svg/, panels), csv/, and
results/high_voltage_filipe/High_Voltage_Filipe_comparison.html (self-contained).
"""
import argparse
import base64
import csv
import datetime
import glob
import html
import io
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pnpflow import figures as F  # noqa: E402

NAME = "High_Voltage_Filipe"
POTENTIALS_MV = (10, 50, 100, 200)
TAUS = (0.07, 0.15)
OUT = os.path.join(ROOT, "results", "high_voltage_filipe")
PAPER = os.path.join(ROOT, "docs", "high_voltage_filipe", "paper_fig8.png")
# panels of the paper figure (pixels in paper_fig8.png, 932 x 1446): left, top, right, bottom
PAPER_CROPS = {"a": (85, 120, 490, 465), "b": (485, 120, 900, 465),
               "c": (85, 462, 490, 805), "d": (485, 462, 900, 805),
               "e": (230, 805, 700, 1145)}
PANELS = {
    "a": ("Psi", 0.07, r"Centreline potential $\Psi_m/\Psi_D$ along the pore and SDL, $\tau = 0.07$"),
    "b": ("rho", 0.07, r"Centreline charge $-\rho_m/\Psi_D$, $\tau = 0.07$"),
    "c": ("Psi", 0.15, r"Centreline potential $\Psi_m/\Psi_D$, $\tau = 0.15$"),
    "d": ("rho", 0.15, r"Centreline charge $-\rho_m/\Psi_D$, $\tau = 0.15$"),
    "e": ("J", None, r"Charge flux into the pore at the mouth, $-J_\mathrm{right}/\Psi_D$ vs $\tau$"),
}
PANEL_TEXT = {
    "a": ("Psi_m / Psi_D vs Z at tau = 0.07",
          "phi_axis / Phi_w on R = 0 (profiles.json, axial_tau_0.07). Z < 0 is the SDL."),
    "b": ("-rho_m / Psi_D vs Z at tau = 0.07",
          "-rho_axis / Phi_w on R = 0, rho = (c+ - c-)/c0. Zero in the electroneutral SDL."),
    "c": ("Psi_m / Psi_D vs Z at tau = 0.15", "as a), at tau = 0.15."),
    "d": ("-rho_m / Psi_D vs Z at tau = 0.15", "as b), at tau = 0.15."),
    "e": ("-J_right / Psi_D vs tau",
          "J_right = 2/(L alpha^2) dQ/dtau, Q = pore charge at every accepted BDF2 step; "
          "cross-section average over the mouth, units D c0 / l_p (summary.json integrals.mouth)."),
}
Z_POINTS = (-0.25, -0.02, 0.02, 0.25, 0.5, 0.75, 1.0)
TAU_POINTS = (0.03, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------
def run_ok(folder):
    """(ok, reason): finished and converged (AGENTS.md section 4)."""
    log = os.path.join(folder, "stdout.log")
    summ = os.path.join(folder, "summary.json")
    if not os.path.isfile(summ):
        return False, "no summary.json"
    if os.path.isfile(log):
        with open(log, errors="replace") as f:
            tail = f.read()[-4000:]
        if "converged: True" not in tail:
            return False, "stdout.log has no 'converged: True'"
    with open(summ) as f:
        s = json.load(f)
    tr = s.get("transient") or {}
    if tr and tr.get("t_reached", 0) < tr.get("t_end", 0) * (1 - 1e-9):
        return False, f"stopped at tau = {tr['t_reached'] / tr['t_per_tau']:.4g}"
    return True, ""


def pick_runs(run_args, cases_dir):
    runs, notes = {}, {}
    for item in run_args:
        mv, path = item.split("=", 1)
        runs[int(mv)] = os.path.abspath(path)
    for mv in POTENTIALS_MV:
        if mv in runs:
            continue
        cands = sorted(glob.glob(os.path.join(cases_dir, f"{NAME}_{mv}mV", "results", "*")),
                       key=os.path.getmtime, reverse=True)
        why = "no runs yet"
        for c in cands:
            ok, why = run_ok(c)
            if ok:
                runs[mv] = c
                break
        if mv not in runs:
            notes[mv] = why
    return runs, notes


def load_run(folder):
    with open(os.path.join(folder, "summary.json")) as f:
        summ = json.load(f)
    with open(os.path.join(folder, "profiles.json")) as f:
        lines = json.load(f)["lines"]
    psi = float(summ["params"]["phi_w"])
    axial = {}
    for ln in lines:
        if ln["name"].startswith("axial_tau_"):
            axial[float(ln["time"]["tau"])] = ln
    prof = {}
    for tau in TAUS:
        if not axial:
            break
        t_near = min(axial, key=lambda t: abs(t - tau))
        if abs(t_near - tau) > 1e-3 * max(tau, 1e-3):
            continue
        ln = axial[t_near]
        v = ln["values"]
        if "phi_axis" not in v:
            raise SystemExit(f"{folder}: profiles.json has no phi_axis (run made before the "
                             "centreline lines were added; rerun)")
        prof[tau] = {"Z": np.asarray(ln["coord"], float),
                     "Psi": np.asarray(v["phi_axis"], float) / psi,
                     "rho": -np.asarray(v["rho_axis"], float) / psi}
    mouth = (summ.get("integrals") or {}).get("mouth")
    flux = None
    if mouth:
        flux = {"tau": np.asarray(mouth["tau"], float),
                "J": -np.asarray(mouth["J_right"], float) / psi}
    tr = summ.get("transient") or {}
    info = {
        "folder": os.path.relpath(folder, ROOT), "psi": psi,
        "phi_w_mV": summ["input"].get("phi_w_mV"),
        "triangles": summ["mesh"].get("n_elements") or summ["mesh"].get("n_triangles"),
        "dofs": summ["mesh"].get("n_dofs"),
        "steps": tr.get("n_steps"),
        "rejects": sum(s.get("rejects", 0) for s in tr.get("steps", [])),
        "time_min": (summ["solver"].get("time_s") or 0) / 60,
        "checks_fail": [c["name"] for c in summ.get("checks", []) if c.get("status") == "fail"],
        "checks_warn": [c["name"] for c in summ.get("checks", []) if c.get("status") == "warn"],
    }
    return {"prof": prof, "flux": flux, "info": info}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def colours():
    import cmasher as cmr
    cm = cmr.get_sub_cmap("cmr.cosmic_r", 0.2, 0.75)
    return {mv: cm(i / (len(POTENTIALS_MV) - 1)) for i, mv in enumerate(POTENTIALS_MV)}


def draw_panel(ax, key, data, col, legend=False):
    kind, tau, _ = PANELS[key]
    drawn = False
    for mv in POTENTIALS_MV:
        d = data.get(mv)
        if not d:
            continue
        lab = f"{d['info']['psi']:.1f}"
        if kind == "J":
            if d["flux"] is None:
                continue
            ax.plot(d["flux"]["tau"], d["flux"]["J"], "--", color=col[mv], label=lab)
        else:
            if tau not in d["prof"]:
                continue
            p = d["prof"][tau]
            ax.plot(p["Z"], p[kind], "--", color=col[mv], label=lab)
        drawn = True
    if kind == "J":
        ax.set_xlim(0.02, 0.5)
        ax.set_ylim(0, 6)
        ax.set_yticks([0, 1.5, 3, 4.5, 6])
        ax.set_xticks([0.1, 0.2, 0.3, 0.4, 0.5])
        ax.set_xlabel(r"$\tau$")
        ax.set_ylabel(r"$-J_\mathrm{right}/\Psi_D$")
    else:
        ax.set_xlim(-0.5, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([-0.5, 0, 0.5, 1])
        ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.set_xlabel(r"$Z$")
        ax.set_ylabel(r"$\Psi_m/\Psi_D$" if kind == "Psi" else r"$-\rho_m/\Psi_D$")
    F.arrow_axes(ax)
    if legend and drawn:
        ax.legend(title=r"pnpflow: $\Psi_D$", loc="upper left", fontsize=18,
                  title_fontsize=18, handlelength=1.6, borderaxespad=0.2)
    if not drawn:
        ax.text(0.5, 0.5, "no finished runs", transform=ax.transAxes, ha="center",
                va="center", fontsize=18, color="0.4")


def make_figures(data):
    col = colours()
    names = {}
    with F.house_style():
        fig = plt.figure(figsize=(12.5, 15.5))
        gs = fig.add_gridspec(3, 4, hspace=0.55, wspace=0.9)
        axes = [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]),
                fig.add_subplot(gs[1, 0:2]), fig.add_subplot(gs[1, 2:4]),
                fig.add_subplot(gs[2, 1:3])]
        for key, ax in zip("abcde", axes):
            draw_panel(ax, key, data, col, legend=(key == "a"))
        F.add_subplot_labels(axes)
        names["full"] = F.save(fig, OUT, "figure8_pnpflow")
        for key in "abcde":
            fig, ax = plt.subplots(figsize=(6.2, 5.0))
            draw_panel(ax, key, data, col, legend=(key in "ae"))
            names[key] = F.save(fig, OUT, f"figure8_pnpflow_{key}")
    return names


def write_csv(data):
    os.makedirs(os.path.join(OUT, "csv"), exist_ok=True)
    p1 = os.path.join(OUT, "csv", "figure8_profiles.csv")
    with open(p1, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi_w_mV", "Psi_D", "tau", "Z", "Psi_m_over_Psi_D", "minus_rho_m_over_Psi_D"])
        for mv, d in sorted(data.items()):
            for tau, p in sorted(d["prof"].items()):
                for z, a, b in zip(p["Z"], p["Psi"], p["rho"]):
                    w.writerow([mv, f"{d['info']['psi']:.6g}", tau, f"{z:.6g}", f"{a:.6g}", f"{b:.6g}"])
    p2 = os.path.join(OUT, "csv", "figure8_flux.csv")
    with open(p2, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phi_w_mV", "Psi_D", "tau", "minus_J_right_over_Psi_D"])
        for mv, d in sorted(data.items()):
            if d["flux"] is None:
                continue
            for t, j in zip(d["flux"]["tau"], d["flux"]["J"]):
                w.writerow([mv, f"{d['info']['psi']:.6g}", f"{t:.6g}", f"{j:.6g}"])
    return [p1, p2]


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def b64_png(path=None, image=None):
    if image is not None:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        raw = buf.getvalue()
    else:
        with open(path, "rb") as f:
            raw = f.read()
    return "data:image/png;base64," + base64.b64encode(raw).decode()


def values_table(key, data):
    kind, tau, _ = PANELS[key]
    pts = TAU_POINTS if kind == "J" else Z_POINTS
    head = "".join(f"<th>{'&tau;' if kind == 'J' else 'Z'} = {x:g}</th>" for x in pts)
    rows = []
    for mv in POTENTIALS_MV:
        d = data.get(mv)
        cells = []
        for x in pts:
            v = None
            if d and kind == "J" and d["flux"] is not None:
                t, j = d["flux"]["tau"], d["flux"]["J"]
                if t[0] <= x <= t[-1]:
                    v = float(np.interp(x, t, j))
            elif d and kind != "J" and tau in d["prof"]:
                p = d["prof"][tau]
                v = float(np.interp(x, p["Z"], p[kind]))
            cells.append(f"<td>{v:.3f}</td>" if v is not None else "<td class=na>&ndash;</td>")
        psi = f"{d['info']['psi']:.2f}" if d else f"{mv / 25.693:.2f}"
        rows.append(f"<tr><th>{mv} mV (&Psi;<sub>D</sub> = {psi})</th>{''.join(cells)}</tr>")
    return (f"<table class=vals><thead><tr><th>pnpflow</th>{head}</tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>")


CSS = """
:root{--ink:#1d2330;--muted:#5b6475;--line:#d9dde5;--bg:#fbfbfc;--panel:#fff;--accent:#3b4cca}
*{box-sizing:border-box}body{margin:0;font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:var(--ink);background:var(--bg)}
main{max-width:1280px;margin:0 auto;padding:32px 28px 64px}
h1{font-size:26px;margin:0 0 6px}h2{font-size:19px;margin:40px 0 12px}h3{font-size:16px;margin:0 0 4px}
p.lead{color:var(--muted);margin:0 0 20px;max-width:900px}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:14px}
.card img{width:100%;height:auto;display:block}
.cap{font-size:13px;color:var(--muted);margin-top:8px}
.panel{margin-top:22px}
.panel .def{color:var(--muted);font-size:14px;margin:2px 0 10px}
table{border-collapse:collapse;font-size:13px;margin-top:10px;width:100%}
th,td{border-bottom:1px solid var(--line);padding:5px 8px;text-align:right;white-space:nowrap}
thead th{color:var(--muted);font-weight:600}tbody th{text-align:left;font-weight:600}
td.na{color:#aab}
dl.setup{display:grid;grid-template-columns:max-content 1fr;gap:4px 18px;margin:0}
dl.setup dt{color:var(--muted)}dl.setup dd{margin:0}
ul.notes{margin:6px 0 0 18px;padding:0;color:var(--ink)}
.missing{color:#a3361f}
@media (max-width:860px){.pair{grid-template-columns:1fr}}
"""


def write_html(data, notes_missing, names, paper):
    from PIL import Image
    im = Image.open(paper).convert("RGB") if os.path.isfile(paper) else None
    fig_dir = os.path.join(OUT, "figures")
    full_paper = b64_png(paper) if im else ""
    full_ours = b64_png(os.path.join(fig_dir, f"{names['full']}.png"))
    parts = []
    parts.append(f"<h1>{NAME}: pnpflow vs Fig. 8 of Henrique, Zuk &amp; Gupta (2022)</h1>")
    parts.append("<p class=lead>Pore charging at high wall potentials. Left: the paper's figure "
                 "(solid = first-order perturbation model, dashed = OpenFOAM DNS). Right: pnpflow, "
                 "the full nonlinear PNP equations, drawn dashed like the DNS. Same axes and "
                 "quantities in both.</p>")
    setup = [
        ("pore", "straight, radius a<sub>p</sub> = 20 nm (&alpha; = 2, a<sub>ref</sub> = 10 nm), "
                 "length l<sub>p</sub> = 1 &micro;m (a<sub>p</sub>/l<sub>p</sub> = 0.02)"),
        ("electrolyte", "1:1, c<sub>0</sub> = 0.948 mM, &epsilon; = 7.0832e-10 F/m, "
                        "D = 1.337e-9 m&sup2;/s, T = 298.15 K &rarr; &lambda;<sub>D</sub> = 9.97 nm, "
                        "a<sub>p</sub>/&lambda;<sub>D</sub> = 2.005"),
        ("SDL", "radius 2 a<sub>p</sub>, length l<sub>p</sub>/2 &rarr; Bi = 8; far end c = c<sub>0</sub>, "
                "&phi; = 0; side and face uncharged, no flux"),
        ("wall", "step to &Psi;<sub>D</sub> at &tau; = 0, blocking, no Stern layer; pore end at "
                 "&Psi;<sub>D</sub>, blocking"),
        ("potentials", "10, 50, 100, 200 mV &rarr; &Psi;<sub>D</sub> = 0.389, 1.946, 3.892, 7.784 "
                       "(paper legend 0.4, 1.9, 3.9, 7.8)"),
        ("time", "&tau; = D t / l<sub>p</sub>&sup2; (1 &tau; = 0.748 ms); profiles at &tau; = 0.07, 0.15; "
                 "J<sub>right</sub> at every step to &tau; = 0.5"),
    ]
    parts.append("<div class=card><dl class=setup>" + "".join(
        f"<dt>{k}</dt><dd>{v}</dd>" for k, v in setup) + "</dl></div>")
    if notes_missing:
        parts.append("<p class=missing>Not shown yet: " + "; ".join(
            f"{mv} mV ({html.escape(why)})" for mv, why in sorted(notes_missing.items())) + "</p>")

    parts.append("<h2>Whole figure</h2><div class=pair>")
    parts.append(f"<div class=card><img alt='Paper Fig. 8' src='{full_paper}'>"
                 "<div class=cap>Henrique, Zuk &amp; Gupta, Soft Matter 18, 198 (2022), Fig. 8 "
                 "(CC BY 4.0)</div></div>" if im else "<div class=card>paper image missing</div>")
    parts.append(f"<div class=card><img alt='pnpflow Fig. 8' src='{full_ours}'>"
                 "<div class=cap>pnpflow, cases/High_Voltage_Filipe_*mV</div></div></div>")

    parts.append("<h2>Panel by panel</h2>")
    for key in "abcde":
        title, definition = PANEL_TEXT[key]
        crop = b64_png(image=im.crop(PAPER_CROPS[key])) if im else ""
        ours = b64_png(os.path.join(fig_dir, f"{names[key]}.png"))
        parts.append(f"<section class='panel card'><h3>{key}) {html.escape(title)}</h3>"
                     f"<div class=def>pnpflow: {html.escape(definition)}</div><div class=pair>"
                     f"<div><img alt='paper panel {key}' src='{crop}'><div class=cap>paper</div></div>"
                     f"<div><img alt='pnpflow panel {key}' src='{ours}'><div class=cap>pnpflow</div></div>"
                     f"</div>{values_table(key, data)}</section>")

    parts.append("<h2>Runs</h2><table><thead><tr><th>potential</th><th>run</th><th>triangles</th>"
                 "<th>dofs</th><th>BDF2 steps</th><th>rejected</th><th>solve [min]</th>"
                 "<th>failed checks</th></tr></thead><tbody>")
    for mv in POTENTIALS_MV:
        d = data.get(mv)
        if not d:
            parts.append(f"<tr><th>{mv} mV</th><td class=na colspan=7>"
                         f"{html.escape(notes_missing.get(mv, 'missing'))}</td></tr>")
            continue
        i = d["info"]
        fails = ", ".join(i["checks_fail"]) or "none"
        parts.append(f"<tr><th>{mv} mV</th><td>{html.escape(i['folder'])}</td><td>{i['triangles']}</td>"
                     f"<td>{i['dofs']}</td><td>{i['steps']}</td><td>{i['rejects']}</td>"
                     f"<td>{i['time_min']:.1f}</td><td>{html.escape(fails)}</td></tr>")
    parts.append("</tbody></table>")
    parts.append("<h2>Differences from the paper's DNS set-up</h2><ul class=notes>"
                 "<li>Pore end: the paper uses zero gradients at Z = 1 (long-pore limit); pnpflow "
                 "closes the pore with a blocking end at &Psi;<sub>D</sub>, which adds an end EDL "
                 "within about a<sub>p</sub> of Z = 1 (&Delta;Z &asymp; 0.02).</li>"
                 "<li>Mouth: sharp corner refined locally; the OpenFOAM mesh had a rounded mouth.</li>"
                 "<li>Initial state: c = c<sub>0</sub> everywhere, &phi; from Poisson at &tau; = 0 "
                 "(consistent start).</li>"
                 "<li>Wall mesh: element size min(0.1, &ell;/4) a<sub>ref</sub> at the wall, with "
                 "&ell; = &lambda;<sub>D</sub>/(2 sinh(&Psi;<sub>D</sub>/2)) the counter-ion "
                 "e-folding length (0.005 a<sub>ref</sub> at 200 mV).</li>"
                 "<li>Time-step control relative to 1 + |U| (err_mixed = 1), since c<sub>&minus;</sub> "
                 "reaches e<sup>7.8</sup> &asymp; 2400 c<sub>0</sub> at the wall at 200 mV.</li></ul>")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    parts.append(f"<p class=cap>Made by tools/figure_high_voltage.py, {stamp}. Data: "
                 "results/high_voltage_filipe/csv/.</p>")
    doc = ("<!DOCTYPE html><html lang=en><head><meta charset=utf-8>"
           "<meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>{NAME}: pnpflow vs paper Fig. 8</title><style>{CSS}</style></head>"
           f"<body><main>{''.join(parts)}</main></body></html>")
    path = os.path.join(OUT, f"{NAME}_comparison.html")
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", action="append", default=[], metavar="MV=RUN_DIR",
                    help="use this run for a potential, e.g. 200=cases/.../results/hv200_run1")
    ap.add_argument("--cases", default=os.path.join(ROOT, "cases"))
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--paper", default=PAPER, help="paper Fig. 8 image (PNG)")
    args = ap.parse_args()
    OUT = os.path.abspath(args.out)
    runs, missing = pick_runs(args.run, args.cases)
    data = {}
    for mv, folder in sorted(runs.items()):
        data[mv] = load_run(folder)
        print(f"{mv:4d} mV: {os.path.relpath(folder, ROOT)}")
    for mv, why in sorted(missing.items()):
        print(f"{mv:4d} mV: not shown ({why})")
    names = make_figures(data)
    files = write_csv(data)
    page = write_html(data, missing, names, args.paper)
    print("figure:", os.path.join(OUT, "figures", names["full"] + ".png"))
    print("csv   :", *files)
    print("html  :", page)


if __name__ == "__main__":
    main()
