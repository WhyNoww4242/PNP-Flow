#!/usr/bin/env python
"""Interactive HTML report: pnpflow vs DNS for the DNS pore-charging runs.

Reads the queue status files of tools/run_dns_batch.py (the runs picked there), the DNS
lines (transportFolder/poreRadius<in>_<out>/<t>/lineYe*, lineXm5*) and each run's
profiles.json / summary.json, and writes one self-contained HTML page (Plotly inlined):
run table, interactive profiles (pnpflow lines vs DNS points, any time, any case),
pnpflow - DNS differences, error vs time, time steps, failures and queue events, plus an
optional notes fragment (analysis, improvements) written by hand.

    python tools/report_dns.py --status results/dns_tau053_status.json \\
        --status results/dns_extreme_status.json \\
        --notes results/pore_dns_compare/report_notes.html \\
        --out results/pore_dns_compare/report.html
"""
import argparse
import datetime
import html
import json
import os
import re
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tools"))
import figure_dns4 as D4  # noqa: E402
import figure_dns6 as D6  # noqa: E402

PLOTLY = os.path.join(ROOT, "app", "ui", "vendor", "plotly.min.js")
DNS_TAG = {c: tag for c, tag, _ in D4.CASES}
DNS_LABEL = {c: lab for c, _, lab in D4.CASES}
T_PER_TAU_S = (10e-6) ** 2 / 1.337e-9          # tau = D t / l_p^2 (DNS set-up)


def thin(x, n):
    """Every k-th index so that at most n points remain (keeps the last one)."""
    m = len(x)
    if m <= n:
        return np.arange(m)
    idx = np.unique(np.linspace(0, m - 1, n).round().astype(int))
    return idx


def r4(a, nd=6):
    return [float(f"{v:.{nd}g}") for v in np.asarray(a, float)]


def line_error(xd, vd, xp, vp, pore_only=False):
    inside = (xd >= xp.min()) & (xd <= xp.max())
    if pore_only:
        inside &= xd >= 0
    if not inside.any():
        return None, None
    o = np.argsort(xp)
    vi = np.interp(xd[inside], xp[o], vp[o])
    d = vi - vd[inside]
    scale = float(np.max(np.abs(vd[inside]))) or 1.0
    k = int(np.argmax(np.abs(d)))
    err = {"max": float(np.abs(d).max()), "rms": float(np.sqrt(np.mean(d ** 2))),
           "rel": float(np.abs(d).max() / scale), "where": float(xd[inside][k]),
           "scale": scale}
    return err, (xd[inside], d)


def case_data(case, run_dir, dns_root):
    pn = D4.load_pnpflow(run_dir)
    dns_dir = os.path.join(dns_root, f"poreRadius{DNS_TAG[case]}")
    times = []
    for t_s in sorted(pn):
        rec = pn[t_s]
        if "axial" not in rec or "radial" not in rec:
            continue
        td = D4.dns_time_dir(dns_dir, t_s) if os.path.isdir(dns_dir) else None
        ax, rd = rec["axial"], rec["radial"]
        Zp = np.asarray(ax["coord"], float)
        mup = np.asarray(ax["values"]["mu_axis"], float)
        Rp = np.asarray(rd["coord"], float)
        keep = Zp >= -0.5
        iz = thin(Zp[keep], 500)
        ir = thin(Rp, 200)
        item = {"t_s": t_s, "tau": float(rec["tau"]),
                "pn": {"mu": [r4(Zp[keep][iz]), r4(mup[keep][iz])],
                       "phi": [r4(Rp[ir]), r4(np.asarray(rd["values"]["phi"])[ir])],
                       "rho": [r4(Rp[ir]), r4(np.asarray(rd["values"]["rho"])[ir])],
                       "mur": [r4(Rp[ir]), r4(np.asarray(rd["values"]["mu"])[ir])]},
                "dns": None, "err": {}, "diff": {}}
        if td:
            dc = D4.load_dns_line(os.path.join(td, "lineYe_nPlus_nMinus_ePhi.xy"))
            dr = D4.load_dns_line(os.path.join(td, "lineXm5_nPlus_nMinus_ePhi.xy"))
            m = dc["Z"] >= -0.5
            Zd, mud = dc["Z"][m], dc["mu"][m]
            o = np.argsort(Zd)
            Zd, mud = Zd[o], mud[o]
            o = np.argsort(dr["R"])
            Rd = dr["R"][o]
            jz, jr = thin(Zd, 160), thin(Rd, 80)
            item["dns"] = {"mu": [r4(Zd[jz]), r4(mud[jz])],
                           "phi": [r4(Rd[jr]), r4(dr["phi"][o][jr])],
                           "rho": [r4(Rd[jr]), r4(dr["rho"][o][jr])],
                           "mur": [r4(Rd[jr]), r4(dr["mu"][o][jr])]}
            for key, xd, vd, xp, vp, pore in (
                    ("mu", Zd, mud, Zp, mup, True),
                    ("phi", Rd, dr["phi"][o], Rp, np.asarray(rd["values"]["phi"]), False),
                    ("rho", Rd, dr["rho"][o], Rp, np.asarray(rd["values"]["rho"]), False),
                    ("mur", Rd, dr["mu"][o], Rp, np.asarray(rd["values"]["mu"]), False)):
                err, dd = line_error(xd, vd, xp, vp, pore_only=pore)
                if err:
                    item["err"][key] = err
                    k = thin(dd[0], 300)
                    item["diff"][key] = [r4(dd[0][k]), r4(dd[1][k], 4)]
        times.append(item)
    return times


def run_meta(run_dir, attempt):
    s = D4.json.load(open(os.path.join(run_dir, "summary.json"))) if os.path.isfile(
        os.path.join(run_dir, "summary.json")) else {}
    tr = s.get("transient") or {}
    mesh = s.get("mesh") or {}
    steps = tr.get("steps") or []
    tpt = tr.get("t_per_tau") or 1.0
    checks = [{k: c.get(k) for k in ("name", "value", "status", "note")}
              for c in s.get("checks", [])]
    mchk = [{k: c.get(k) for k in ("name", "value", "limit", "status", "note")}
            for c in (s.get("mesh_checks") or []) if c.get("status") != "pass"]
    return {
        "run": os.path.relpath(run_dir, ROOT),
        "density": attempt.get("density"),
        "triangles": mesh.get("n_elements") or attempt.get("triangles"),
        "dofs": attempt.get("dofs"),
        "h_min": mesh.get("h_min"), "q_min": mesh.get("quality_min"),
        "wall_min": round((attempt.get("wall_time_s") or 0) / 60, 1),
        "n_steps": tr.get("n_steps"),
        "rejects": int(sum(st.get("rejects", 0) for st in steps)),
        "newton": (s.get("solver") or {}).get("n_newton_total"),
        "tau_reached": (tr.get("t_reached") or 0) / tpt,
        "device": (s.get("device") or {}).get("platform"),
        "checks": checks, "mesh_checks": mchk,
        "steps": {"tau": r4([st["t"] / tpt for st in steps]),
                  "dtau": r4([st["dt"] / tpt for st in steps]),
                  "its": [st.get("newton_its") for st in steps],
                  "rej": [st.get("rejects", 0) for st in steps]},
        "params": {k: (s.get("params") or {}).get(k) for k in
                   ("alpha_in", "alpha_out", "L_nm", "h_wall", "h_radius_n", "tau_end",
                    "dns_init")},
    }


def failures(status):
    out = []
    for case, cs in status.get("cases", {}).items():
        for a in cs.get("attempts", []) + cs.get("invalid_runs", []):
            if a.get("class") == "ok":
                continue
            out.append({"case": case, "run": a.get("run"), "class": a.get("class"),
                        "density": a.get("density"), "reasons": a.get("reasons") or [],
                        "started": a.get("started"), "finished": a.get("finished")})
        if cs.get("result") not in (None, "ok"):
            out.append({"case": case, "run": cs.get("run"), "class": "case " + str(cs["result"]),
                        "density": cs.get("density"), "reasons": cs.get("bad_params") or [],
                        "started": None, "finished": None})
    return out


def queue_events(log_paths):
    ev = []
    for p in log_paths:
        if not os.path.isfile(p):
            continue
        for ln in open(p, encoding="utf-8", errors="replace"):
            m = re.match(r"(\S+)\s+(.*)", ln.rstrip())
            if m:
                ev.append({"time": m.group(1), "text": m.group(2), "log": os.path.basename(p)})
    return ev


def colours_hex():
    out = {}
    for c, rgba in D6.colours().items():
        out[c] = "#%02x%02x%02x" % tuple(int(round(255 * v)) for v in rgba[:3])
    return out


def build(statuses, dns_root, logs):
    cases, fails, order = {}, [], []
    for sp in statuses:
        st = json.load(open(sp))
        fails += failures(st)
        for case, cs in st.get("cases", {}).items():
            if cs.get("result") != "ok" or not cs.get("run"):
                continue
            run_dir = os.path.join(ROOT, cs["run"])
            att = next((a for a in cs.get("attempts", []) if a.get("run") == cs["run"]), {})
            g = D6.GEOM.get(case, (case, None, None))
            cases[case] = {"label": g[0], "alpha_in": g[1], "alpha_out": g[2],
                           "dns": f"poreRadius{DNS_TAG.get(case, '?')}",
                           "dns_label": DNS_LABEL.get(case, case),
                           "set": "extreme" if case in D6.SHADE_EXTREME else "main",
                           "meta": run_meta(run_dir, att),
                           "times": case_data(case, run_dir, dns_root)}
    order = [c for c in D6.ORDER if c in cases]
    return {"generated": datetime.datetime.now().isoformat(timespec="minutes"),
            "order": order, "cases": cases, "colours": colours_hex(),
            "failures": fails, "events": queue_events(logs),
            "t_per_tau_s": T_PER_TAU_S}


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pore DNS Comparison</title>
<style>
:root {
  --bg:#0d1016; --surface:#141922; --raised:#1a202b; --hover:#212836; --border:#252c39;
  --border-strong:#343d4e; --text:#dfe5ee; --dim:#8b96a8; --faint:#5d6778; --accent:#5cb8ff;
  --accent-ink:#06121f; --ok:#51cf66; --warn:#fcc419; --bad:#ff6b6b; --grid:#252c39;
  color-scheme: dark;
}
@media (prefers-color-scheme: light) {
  :root:not([data-theme="dark"]) {
    --bg:#f3f5f8; --surface:#ffffff; --raised:#f7f8fa; --hover:#edf0f5; --border:#dde2ea;
    --border-strong:#c7cfdb; --text:#1c2330; --dim:#5d6879; --faint:#8a94a4; --accent:#0b6fd0;
    --accent-ink:#ffffff; --ok:#2b8a3e; --warn:#c48a00; --bad:#d6336c; --grid:#e3e7ee;
    color-scheme: light;
  }
}
:root[data-theme="light"] {
  --bg:#f3f5f8; --surface:#ffffff; --raised:#f7f8fa; --hover:#edf0f5; --border:#dde2ea;
  --border-strong:#c7cfdb; --text:#1c2330; --dim:#5d6879; --faint:#8a94a4; --accent:#0b6fd0;
  --accent-ink:#ffffff; --ok:#2b8a3e; --warn:#c48a00; --bad:#d6336c; --grid:#e3e7ee;
  color-scheme: light;
}
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--text);
  font: 14px/1.5 Inter, "Segoe UI", system-ui, -apple-system, sans-serif; }
main { max-width: 1180px; margin: 0 auto; padding-block: 28px 64px;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
h1 { font-size: 22px; font-weight: 600; margin: 0 0 4px; }
h2 { font-size: 17px; font-weight: 600; margin: 40px 0 10px; }
h3 { font-size: 14.5px; font-weight: 600; margin: 20px 0 6px; }
p, li { color: var(--text); max-width: 78ch; }
.dim { color: var(--dim); }
.num { font-family: "JetBrains Mono", "Cascadia Mono", "SF Mono", Consolas, monospace;
  font-size: 12.5px; font-variant-numeric: tabular-nums; }
section.panel { background: var(--surface); border: 1px solid var(--border); border-radius: 8px;
  padding: 16px; margin-top: 12px; }
.table-wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--border);
  vertical-align: top; }
th { color: var(--dim); font-weight: 500; white-space: nowrap; }
td.r, th.r { text-align: right; }
.tag { display:inline-block; padding: 1px 8px; border-radius: 10px; font-size: 12px;
  border: 1px solid var(--border-strong); white-space: nowrap; }
.tag.ok { color: var(--ok); border-color: currentColor; }
.tag.warn { color: var(--warn); border-color: currentColor; }
.tag.bad { color: var(--bad); border-color: currentColor; }
.controls { display: flex; flex-wrap: wrap; gap: 14px 22px; align-items: center; margin-bottom: 10px; }
.seg { display: inline-flex; border: 1px solid var(--border-strong); border-radius: 6px; overflow: hidden; }
.seg button { background: var(--raised); color: var(--text); border: 0; padding: 0 12px; height: 30px;
  font: inherit; font-size: 13px; cursor: pointer; transition: background .15s; }
.seg button + button { border-left: 1px solid var(--border-strong); }
.seg button:hover { background: var(--hover); }
.seg button[aria-pressed="true"] { background: var(--accent); color: var(--accent-ink); }
.seg button:focus-visible, .chk input:focus-visible, input[type=range]:focus-visible {
  outline: 2px solid var(--accent); outline-offset: 2px; }
.chk { display: inline-flex; align-items: center; gap: 6px; cursor: pointer; font-size: 13px; }
.chk .sw { width: 14px; height: 3px; border-radius: 2px; display: inline-block; }
label.lbl { color: var(--dim); font-size: 12.5px; margin-right: 6px; }
input[type=range] { width: 260px; accent-color: var(--accent); vertical-align: middle; }
.plot { width: 100%; height: 440px; }
.plot.small { height: 340px; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
@media (max-width: 860px) { .grid2 { grid-template-columns: 1fr; } input[type=range] { width: 100%; } }
figure { margin: 12px 0 0; }
figure img { width: 100%; max-width: 900px; background: #fff; border-radius: 6px; }
figcaption { color: var(--dim); font-size: 12.5px; margin-top: 6px; }
details { margin-top: 8px; }
summary { cursor: pointer; color: var(--dim); }
.notes h2:first-child { margin-top: 0; }
.theme { float: right; }
</style>
</head>
<body>
<main>
<div class="theme seg" role="group" aria-label="Theme">
  <button data-theme="auto" aria-pressed="true">Auto</button><button data-theme="light" aria-pressed="false">Light</button><button data-theme="dark" aria-pressed="false">Dark</button>
</div>
<h1>Pore charging: pnpflow vs DNS</h1>
<p class="dim">Generated __GENERATED__ by <span class="num">tools/report_dns.py</span>. Axisymmetric PNP charging of a closed pore
(L = 10 µm, a<sub>ref</sub> = 10 nm, c<sub>0</sub> = 0.948 mM, λ<sub>D</sub> ≈ 10 nm, Φ<sub>w</sub> = 10 mV) from the reservoir,
compared with the DNS at its saved times τ = D t / l<sub>p</sub><sup>2</sup> = 0.06685 k.</p>

<div id="headline"></div>

<h2>Runs</h2>
<section class="panel"><div class="table-wrap"><table id="runs"></table></div>
<p class="dim" style="margin-bottom:0">Max |Δμ| = largest |pnpflow − DNS| of the centreline μ = ρ + 2Φ inside the pore (Z ≥ 0), at the DNS points.
Relative values are to the largest |DNS value| on that line. Mesh diagnose: every run fails or warns only on the EDL-resolution check
along the pore wall (≈ 6 elements across 3λ<sub>D</sub> instead of 8; the coarse h<sub>wall</sub> = 0.15 the plan allows so the meshes fit in memory).</p></section>

<h2>Profiles</h2>
<section class="panel">
  <div class="controls">
    <div><label class="lbl">Quantity</label><span class="seg" id="qty" role="group" aria-label="Quantity"></span></div>
    <div><label class="lbl">Show</label><span class="seg" id="mode" role="group" aria-label="Show"></span></div>
  </div>
  <div class="controls">
    <div><label class="lbl" for="tslider">Time</label><input type="range" id="tslider" min="0" max="0" step="1" value="0">
      <span class="num" id="tlabel"></span></div>
  </div>
  <div class="controls" id="casebox"></div>
  <div id="pplot" class="plot"></div>
  <p class="dim" id="pnote" style="margin:4px 0 0"></p>
</section>

<h2>Difference over time</h2>
<section class="panel">
  <p class="dim" style="margin-top:0">Max |pnpflow − DNS| of the quantity chosen above, at every saved DNS time. The case selection above applies.</p>
  <div class="grid2"><div id="eplot" class="plot small"></div><div id="rplot" class="plot small"></div></div>
</section>

<h2>Time stepping</h2>
<section class="panel">
  <p class="dim" style="margin-top:0">Variable-step BDF2. Left: step size Δτ against τ. Right: Newton iterations per step (rejected steps are counted in the run table).</p>
  <div class="grid2"><div id="dtplot" class="plot small"></div><div id="itplot" class="plot small"></div></div>
</section>

<h2>Failures and interruptions</h2>
<section class="panel"><div class="table-wrap"><table id="fails"></table></div>
<details><summary>Queue events (every line of the queue logs)</summary><div class="table-wrap"><table id="events"></table></div></details>
</section>

__FIGURES__

<section class="notes">__NOTES__</section>
</main>
<script>__PLOTLY__</script>
<script>
const DATA = __DATA__;
const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const fmt = (v, d = 3) => v == null || Number.isNaN(v) ? "–" : (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v).toPrecision(d));
const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const QTY = { mu: { name: "μ centreline vs Z", x: "Z", y: "μ̂ = ρ̂ + 2Φ̂ (centreline)" },
              phi: { name: "Φ at Z = 0.5", x: "R / α(Z)", y: "Φ̂ at Z = 0.5" },
              rho: { name: "ρ at Z = 0.5", x: "R / α(Z)", y: "ρ̂ at Z = 0.5" },
              mur: { name: "μ at Z = 0.5", x: "R / α(Z)", y: "μ̂ at Z = 0.5" } };
const S = { qty: "mu", mode: "lines", t: 0, on: new Set(DATA.order.filter((c) => DATA.cases[c].set === "main")) };
if (!S.on.size) DATA.order.forEach((c) => S.on.add(c));

// all saved DNS times (seconds) over the cases
const TIMES = [...new Set(DATA.order.flatMap((c) => DATA.cases[c].times.filter((t) => t.dns).map((t) => t.t_s)))].sort((a, b) => a - b);
S.t = Math.max(0, TIMES.findIndex((t) => Math.abs(t - 0.04) < 1e-9));
if (S.t < 0) S.t = TIMES.length - 1;

function layout(xt, yt, extra = {}) {
  const fg = css("--text"), dim = css("--dim"), grid = css("--grid");
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: "Inter, Segoe UI, system-ui, sans-serif", size: 13, color: fg },
    margin: { l: 64, r: 16, t: 12, b: 52 },
    xaxis: { title: { text: xt }, gridcolor: grid, zerolinecolor: grid, linecolor: dim, ticks: "outside", tickcolor: dim },
    yaxis: { title: { text: yt }, gridcolor: grid, zerolinecolor: grid, linecolor: dim, ticks: "outside", tickcolor: dim },
    legend: { orientation: "h", y: -0.2, font: { size: 12 } }, hovermode: "closest",
  }, extra);
}
const CFG = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d"] };

function seg(el, items, key, onchange) {
  el.innerHTML = items.map(([v, l]) => `<button data-v="${v}" aria-pressed="${S[key] === v}">${l}</button>`).join("");
  el.addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b) return;
    S[key] = b.dataset.v;
    el.querySelectorAll("button").forEach((x) => x.setAttribute("aria-pressed", x === b));
    onchange();
  });
}

function headline() {
  let worst = null;
  DATA.order.forEach((c) => DATA.cases[c].times.forEach((t) => ["mu", "phi", "rho"].forEach((q) => {
    const e = t.err[q]; if (e && (!worst || e.rel > worst.rel)) worst = { ...e, c, q, tau: t.tau };
  })));
  const n = DATA.order.length;
  const nf = DATA.failures.length;
  $("#headline").innerHTML = worst ? `<section class="panel"><p style="margin:0">${n} runs compared over ${TIMES.length} DNS times.
    Largest difference anywhere: <span class="num">${fmt(worst.max)}</span> (${(100 * worst.rel).toFixed(2)} % of the line's largest value),
    ${QTY[worst.q].name}, ${esc(DATA.cases[worst.c].label)}, τ = ${worst.tau.toFixed(3)}, at ${QTY[worst.q].x} = ${fmt(worst.where)}.
    ${nf} failed or interrupted attempt${nf === 1 ? "" : "s"} along the way (table below).</p></section>` : "";
}

function runsTable() {
  const rows = DATA.order.map((c) => {
    const d = DATA.cases[c], m = d.meta;
    const errAt = (ts) => { const t = d.times.find((x) => Math.abs(x.t_s - ts) < 1e-9); return t && t.err.mu ? t.err.mu : null; };
    const e1 = errAt(0.01), eL = errAt(TIMES[TIMES.length - 1]) || errAt(0.035);
    const diag = m.mesh_checks.length ? m.mesh_checks.map((k) => k.status).includes("fail") ? "bad" : "warn" : "ok";
    const mc = m.mesh_checks.map((k) => `${k.name}: ${k.status}`).join("; ");
    return `<tr><td><span class="sw" style="display:inline-block;width:12px;height:3px;background:${DATA.colours[c]};vertical-align:middle;margin-right:6px"></span>${esc(d.label)}<div class="dim">${esc(d.dns)}</div></td>
      <td class="num r">${fmt(d.alpha_in)} → ${fmt(d.alpha_out)}</td>
      <td class="num r">${m.triangles?.toLocaleString() ?? "–"}<div class="dim">${m.dofs ? (m.dofs / 1e6).toFixed(2) + " M dofs" : ""}</div></td>
      <td class="num r">${fmt(m.h_min)}<div class="dim">q ${fmt(m.q_min, 2)}</div></td>
      <td class="num r">${m.n_steps ?? "–"}<div class="dim">${m.rejects} rej · ${m.newton ?? "–"} Newton</div></td>
      <td class="num r">${m.wall_min}</td>
      <td class="num r">${e1 ? fmt(e1.max) : "–"}<div class="dim">${e1 ? (100 * e1.rel).toFixed(2) + " %" : ""}</div></td>
      <td class="num r">${eL ? fmt(eL.max) : "–"}<div class="dim">${eL ? (100 * eL.rel).toFixed(2) + " %" : ""}</div></td>
      <td><span class="tag ok">τ ${m.tau_reached.toFixed(3)}</span> <span class="tag ${diag}" title="${esc(mc)}">mesh ${diag === "ok" ? "pass" : diag === "bad" ? "fail" : "warn"}</span>
        <div class="dim num" style="font-size:11.5px">${esc(m.run)}</div></td></tr>`;
  }).join("");
  $("#runs").innerHTML = `<thead><tr><th>Case</th><th class="r">α in → out</th><th class="r">Triangles</th><th class="r">h<sub>min</sub></th>
    <th class="r">Steps</th><th class="r">Wall [min]</th><th class="r">Max |Δμ| τ 0.13</th><th class="r">Max |Δμ| last τ</th><th>Status · run folder</th></tr></thead><tbody>${rows}</tbody>`;
}

function caseBoxes() {
  const groups = [["main", "DNS shapes"], ["extreme", "Extreme cases"]];
  $("#casebox").innerHTML = groups.filter(([g]) => DATA.order.some((c) => DATA.cases[c].set === g)).map(([g, gl]) =>
    `<div><label class="lbl">${gl}</label>` + DATA.order.filter((c) => DATA.cases[c].set === g).map((c) =>
      `<label class="chk" style="margin-right:12px"><input type="checkbox" data-c="${c}" ${S.on.has(c) ? "checked" : ""}>
       <span class="sw" style="background:${DATA.colours[c]}"></span>${esc(DATA.cases[c].label)}</label>`).join("") + "</div>").join("");
  $("#casebox").addEventListener("change", (e) => {
    const c = e.target.dataset.c; if (!c) return;
    e.target.checked ? S.on.add(c) : S.on.delete(c); drawAll();
  });
}

function drawProfiles() {
  const ts = TIMES[S.t], tau = ts / DATA.t_per_tau_s;
  $("#tlabel").textContent = `t = ${ts.toFixed(3)} s,  τ = ${tau.toFixed(4)}`;
  const tr = [], missing = [];
  DATA.order.filter((c) => S.on.has(c)).forEach((c) => {
    const d = DATA.cases[c], col = DATA.colours[c];
    const t = d.times.find((x) => Math.abs(x.t_s - ts) < 1e-9);
    if (!t) { missing.push(d.label); return; }
    if (S.mode === "lines") {
      const p = t.pn[S.qty];
      tr.push({ x: p[0], y: p[1], mode: "lines", line: { color: col, width: 2.5 }, name: `${d.label} pnpflow`, legendgroup: c,
        hovertemplate: `${d.label} pnpflow<br>%{x:.4f}, %{y:.5f}<extra></extra>` });
      if (t.dns) tr.push({ x: t.dns[S.qty][0], y: t.dns[S.qty][1], mode: "markers", marker: { color: col, symbol: "square", size: 6 },
        name: `${d.label} DNS`, legendgroup: c, hovertemplate: `${d.label} DNS<br>%{x:.4f}, %{y:.5f}<extra></extra>` });
      else missing.push(d.label + " (no DNS at this time)");
    } else if (t.diff[S.qty]) {
      tr.push({ x: t.diff[S.qty][0], y: t.diff[S.qty][1], mode: "lines", line: { color: col, width: 2 }, name: d.label,
        hovertemplate: `${d.label}<br>%{x:.4f}: Δ = %{y:.2e}<extra></extra>` });
    } else missing.push(d.label + " (no DNS at this time)");
  });
  const q = QTY[S.qty];
  const ylab = S.mode === "lines" ? q.y : `pnpflow − DNS (${q.name})`;
  const ex = S.qty === "mu" && S.mode === "lines" ? { shapes: [{ type: "line", x0: 0, x1: 0, yref: "paper", y0: 0, y1: 1, line: { color: css("--faint"), dash: "dot", width: 1 } }],
    annotations: [{ x: 0, yref: "paper", y: 1, text: "pore mouth", showarrow: false, xanchor: "left", font: { size: 11, color: css("--dim") } }] } : {};
  Plotly.react("pplot", tr, layout(q.x, ylab, ex), CFG);
  $("#pnote").textContent = missing.length ? "Not shown: " + missing.join(", ") + "." : (S.qty === "mu" ? "Z < 0 is the reservoir (SDL); the pore runs from Z = 0 (mouth) to Z = 1 (closed end)." : "Radial line at the pore middle, R scaled by the local radius α(0.5).");
}

function drawErrors() {
  const tr = [], trr = [];
  DATA.order.filter((c) => S.on.has(c)).forEach((c) => {
    const d = DATA.cases[c], col = DATA.colours[c];
    const pts = d.times.filter((t) => t.err[S.qty]);
    tr.push({ x: pts.map((t) => t.tau), y: pts.map((t) => t.err[S.qty].max), mode: "lines+markers", name: d.label,
      line: { color: col, width: 2.5 }, marker: { color: col, size: 6 }, hovertemplate: `${d.label}<br>τ %{x:.3f}: %{y:.2e}<extra></extra>` });
    trr.push({ x: pts.map((t) => t.tau), y: pts.map((t) => 100 * t.err[S.qty].rel), mode: "lines+markers", name: d.label, showlegend: false,
      line: { color: col, width: 2.5 }, marker: { color: col, size: 6 }, hovertemplate: `${d.label}<br>τ %{x:.3f}: %{y:.3f} %<extra></extra>` });
  });
  Plotly.react("eplot", tr, layout("τ", `max |Δ| (${QTY[S.qty].name})`), CFG);
  Plotly.react("rplot", trr, layout("τ", "max |Δ| / max |DNS|  [%]"), CFG);
}

function drawSteps() {
  const a = [], b = [];
  DATA.order.filter((c) => S.on.has(c)).forEach((c) => {
    const d = DATA.cases[c], s = d.meta.steps, col = DATA.colours[c];
    a.push({ x: s.tau, y: s.dtau, mode: "lines", name: d.label, line: { color: col, width: 2.5 }, hovertemplate: `${d.label}<br>τ %{x:.2e}: Δτ %{y:.2e}<extra></extra>` });
    b.push({ x: s.tau, y: s.its, mode: "markers", name: d.label, showlegend: false, marker: { color: col, size: 5 },
      hovertemplate: `${d.label}<br>τ %{x:.2e}: %{y} its<extra></extra>` });
  });
  const lx = { type: "log", exponentformat: "power" };
  const L1 = layout("τ", "Δτ"); Object.assign(L1.xaxis, lx); Object.assign(L1.yaxis, lx);
  const L2 = layout("τ", "Newton iterations"); Object.assign(L2.xaxis, lx); L2.yaxis.dtick = 1;
  Plotly.react("dtplot", a, L1, CFG);
  Plotly.react("itplot", b, L2, CFG);
}

function failTables() {
  const cls = (k) => /ok/.test(k) ? "ok" : /interrupted|stopped/.test(k) ? "warn" : "bad";
  $("#fails").innerHTML = `<thead><tr><th>Case</th><th>Class</th><th class="r">Density</th><th>Run</th><th>Why</th></tr></thead><tbody>` +
    (DATA.failures.length ? DATA.failures.map((f) => `<tr><td>${esc(f.case)}</td><td><span class="tag ${cls(f.class)}">${esc(f.class)}</span></td>
      <td class="num r">${f.density ?? "–"}</td><td class="num" style="font-size:12px">${esc(f.run)}</td><td>${f.reasons.map(esc).join("<br>")}</td></tr>`).join("")
      : `<tr><td colspan="5" class="dim">No failed attempts.</td></tr>`) + "</tbody>";
  $("#events").innerHTML = `<thead><tr><th>Time</th><th>Log</th><th>Event</th></tr></thead><tbody>` +
    DATA.events.map((e) => `<tr><td class="num" style="white-space:nowrap">${esc(e.time)}</td><td class="dim">${esc(e.log)}</td><td>${esc(e.text)}</td></tr>`).join("") + "</tbody>";
}

function drawAll() { drawProfiles(); drawErrors(); drawSteps(); }

seg($("#qty"), Object.entries(QTY).map(([k, v]) => [k, v.name]), "qty", () => { drawProfiles(); drawErrors(); });
seg($("#mode"), [["lines", "pnpflow and DNS"], ["diff", "pnpflow − DNS"]], "mode", drawProfiles);
const sl = $("#tslider"); sl.max = TIMES.length - 1; sl.value = S.t;
sl.addEventListener("input", () => { S.t = +sl.value; drawProfiles(); });
document.querySelectorAll(".theme button").forEach((b) => b.addEventListener("click", () => {
  const v = b.dataset.theme;
  if (v === "auto") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", v);
  document.querySelectorAll(".theme button").forEach((x) => x.setAttribute("aria-pressed", x === b));
  drawAll();
}));
matchMedia("(prefers-color-scheme: light)").addEventListener("change", drawAll);
headline(); runsTable(); caseBoxes(); failTables(); drawAll();
</script>
</body>
</html>
"""


PLOTLY_CDN = "https://cdnjs.cloudflare.com/ajax/libs/plotly.js/2.35.2/plotly.min.js"


def write_artifact(out_dir, data, notes, figs, figure_args, base_dir):
    """Page content without the document shell (the host adds it), Plotly from cdnjs,
    data fetched from report_data.json beside the page, figures copied."""
    import shutil
    os.makedirs(out_dir, exist_ok=True)
    page = PAGE.replace("__GENERATED__", data["generated"].replace("T", " ")).replace(
        "__FIGURES__", figs).replace("__NOTES__", notes)
    head_end = page.index("<style>")
    page = '<meta name="claude:viewport-fit-sides" content="cover">\n<title>Pore DNS Comparison</title>\n' \
        + page[head_end:]
    page = page.replace("</head>\n<body>\n", "").replace("</body>\n</html>\n", "")
    i0 = page.index('<div class="theme seg"')        # the viewer sets the theme
    page = page[:i0] + page[page.index("</div>", i0) + len("</div>\n"):]
    page = page.replace("<script>__PLOTLY__</script>", f'<script src="{PLOTLY_CDN}"></script>')
    page = page.replace("const DATA = __DATA__;",
                        '(async () => {\nconst DATA = await (await fetch("report_data.json")).json();')
    k = page.rindex("</script>")
    page = page[:k] + "})();\n" + page[k:]
    with open(os.path.join(out_dir, "index.html"), "w", encoding="utf-8") as f:
        f.write(page)
    with open(os.path.join(out_dir, "report_data.json"), "w") as f:
        json.dump(data, f, separators=(",", ":"))
    for item in figure_args:
        rel = item.split("=", 1)[0]
        dst = os.path.join(out_dir, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(os.path.join(base_dir, rel), dst)
    print(f"artifact build in {out_dir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--status", action="append", required=True, help="queue status json (repeatable)")
    ap.add_argument("--log", action="append", help="queue log(s); default: next to each status file")
    ap.add_argument("--dns", default=os.path.join(ROOT, "DNS_compare", "transportFolder"))
    ap.add_argument("--notes", help="HTML fragment with the analysis / improvements")
    ap.add_argument("--figure", action="append", default=[],
                    help="PATH=CAPTION of a figure to show (relative to the output folder)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "pore_dns_compare", "report.html"))
    ap.add_argument("--data-out", help="also write the report data as JSON")
    ap.add_argument("--artifact", metavar="DIR",
                    help="also write a claude.ai Artifact build to DIR: index.html (page content only, "
                         "Plotly from cdnjs), report_data.json, figures/")
    a = ap.parse_args()
    logs = a.log or [s.replace("_status.json", "_queue.log") for s in a.status]
    data = build(a.status, a.dns, logs)
    notes = open(a.notes, encoding="utf-8").read() if a.notes else ""
    figs = ""
    if a.figure:
        figs = "<h2>Figures</h2><section class=\"panel\">" + "".join(
            f'<figure><img src="{html.escape(p)}" alt="{html.escape(c)}" loading="lazy">'
            f"<figcaption>{html.escape(c)}</figcaption></figure>"
            for p, c in (f.split("=", 1) for f in a.figure)) + "</section>"
    # Plotly last, so nothing inside it is taken for a placeholder
    page = (PAGE.replace("__GENERATED__", data["generated"].replace("T", " "))
            .replace("__FIGURES__", figs)
            .replace("__DATA__", json.dumps(data, separators=(",", ":")).replace("</", "<\\/"))
            .replace("__NOTES__", notes)
            .replace("__PLOTLY__", open(PLOTLY, encoding="utf-8").read().replace("</script", "<\\/script")))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(page)
    if a.data_out:
        json.dump(data, open(a.data_out, "w"), indent=1)
    if a.artifact:
        write_artifact(a.artifact, data, notes, figs, a.figure, os.path.dirname(os.path.abspath(a.out)))
    print(f"wrote {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB): "
          f"{len(data['order'])} runs, {len(data['failures'])} failures")
    for c in data["order"]:
        d = data["cases"][c]
        for t in d["times"]:
            if t["err"]:
                print(f"  {c:11s} tau {t['tau']:.4f}  " + "  ".join(
                    f"{k} {v['max']:.2e} ({100 * v['rel']:.2f}%)" for k, v in t["err"].items()))


if __name__ == "__main__":
    main()
