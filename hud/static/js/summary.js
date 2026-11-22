// Summary panel: overview, parameters, mesh, convergence, checks,
// integrals (fluxes, forces, charge), sensitivities and notes.

import { h, setChildren, placeholder, fmt, fmtValue, fmtDuration, isNum, prettyName } from "./util.js";
import { baseLayout, plot, SERIES_COLORS } from "./plots.js";

const STATUS_ORDER = { fail: 0, warn: 1, pass: 2, info: 3 };

/** A titled card; `wide` spans two grid columns. */
function card(title, body, { wide = false, extra = null, cls = "" } = {}) {
  return h(`section.card${wide ? ".wide" : ""}`, { class: cls },
    h("header.card-head", {}, h("h3", { text: title }), extra),
    h("div.card-body", {}, body));
}

function kvTable(rows) {
  return h("table.kv", {}, h("tbody", {}, rows.filter(Boolean).map(([k, v, title]) =>
    h("tr", { title }, h("th", { text: k }),
      h(typeof v === "number" ? "td.mono.num" : "td.mono", {}, v instanceof Node ? v : fmtValue(v))))));
}

function badge(status, text) {
  return h(`span.badge.b-${status || "info"}`, { text: text || status || "?" });
}

export class SummaryView {
  constructor(root) {
    this.root = root;
  }

  showMessage(kind, title, detail) {
    setChildren(this.root, placeholder(kind, title, detail));
  }

  /** Render summary (may be null) + progress (may be null). */
  render(summary, progress) {
    if (!summary && !progress) {
      return this.showMessage("empty", "No summary yet", "summary.json is written when the run finishes.");
    }
    const s = summary || {};
    const cards = [
      this.errorCard(s, progress),
      this.overviewCard(s, progress),
      this.paramsCard(s),
      this.meshCard(s),
      this.convergenceCard(s, progress),
      this.forcesCard(s),
      this.checksCard(s),
      this.chargeCard(s),
      this.fluxCard(s),
      this.sensitivityCard(s),
      this.notesCard(s),
      this.modelCard(s),
    ].filter(Boolean);
    setChildren(this.root, h("div.summary-grid", {}, cards));
    this.drawDeferred();
  }

  // -- cards ------------------------------------------------------------------
  overviewCard(s, progress) {
    const solver = s.solver || {};
    const model = s.model || {};
    const species = Array.isArray(model.species) ? model.species : [];
    const conv = solver.converged;
    const rows = [
      ["case", s.case],
      ["created", s.created],
      ["converged", conv === undefined ? "—" : badge(conv ? "pass" : "fail", conv ? "yes" : "no")],
      ["solve time", fmtDuration(solver.time_s ?? (progress && progress.time_s))],
      ["wall time", isNum(solver.wall_time_s) ? fmtDuration(solver.wall_time_s) : undefined],
      ["Newton iterations", solver.n_newton_total],
      ["stages", Array.isArray(solver.stages) ? solver.stages.length : undefined],
      ["flow", model.flow],
      ["advection", model.advection],
      ["SUPG", model.supg],
    ].filter(([, v]) => v !== undefined && v !== null);
    const sp = species.length ? h("div.species", {}, species.map((x) => h("span.species-chip", {},
      h("b", { text: prettyName(x.name) }),
      h("span.mono", { text: ` z=${x.z > 0 ? "+" : ""}${fmt(x.z)}  D=${fmt(x.d)}` })))) : null;
    const doc = model.doc ? h("p.doc", { text: model.doc }) : null;
    return card("Overview", [doc, kvTable(rows), sp]);
  }

  /** Red banner card for failed runs: message + collapsible traceback. */
  errorCard(s, progress) {
    const err = s.error || (progress && progress.state === "failed" ? progress.message : null);
    if (!err) return null;
    const text = String(err);
    const lines = text.trim().split(/\r?\n/);
    const last = lines[lines.length - 1] || text;
    const body = [h("div.err-msg.mono", { text: last })];
    if (lines.length > 1) {
      body.push(h("details.err-trace", {}, h("summary", { text: `traceback (${lines.length} lines)` }),
        h("pre.mono", { text: text })));
    }
    return card("Run failed", body, { wide: true, cls: "card-error" });
  }

  /** Boundary conditions per boundary, if the solver reports them. */
  modelCard(s) {
    const b = s.model && s.model.boundaries;
    if (!b || typeof b !== "object" || !Object.keys(b).length) return null;
    const fmtBC = (v) => (v && typeof v === "object"
      ? Object.entries(v).map(([k, x]) => `${prettyName(k)}: ${x}`).join(", ") : String(v));
    const keys = [...new Set(Object.values(b).flatMap((v) => Object.keys(v || {})))];
    return card("Boundary conditions", h("table.grid-table.bc-table", {},
      h("thead", {}, h("tr", {}, h("th", { text: "boundary" }), keys.map((k) => h("th", { text: prettyName(k) })))),
      h("tbody", {}, Object.entries(b).map(([name, v]) => h("tr", {}, h("th", { text: name }),
        keys.map((k) => h("td.mono.small", { text: v && v[k] !== undefined ? fmtBC(v[k]) : "—" })))))), { wide: true });
  }

  paramsCard(s) {
    // `params` = model parameters; `input` = raw case inputs (may overlap)
    const p = Object.assign({}, s.input || {}, s.params || {});
    const info = s.param_info || {};
    const keys = Object.keys(p);
    if (!keys.length) return null;
    const rows = keys.map((k) => h("tr", {},
      h("th", {}, h("span.param", { text: k }, info[k] ? h("span.info-dot", { text: "i", title: info[k] }) : null)),
      h("td.mono.num", { text: fmtValue(p[k]) }),
      h("td.dim.small.desc", { text: info[k] || "", title: info[k] || "" })));
    return card("Parameters", h("table.kv.params", {}, h("tbody", {}, rows)));
  }

  meshCard(s) {
    const m = s.mesh;
    if (!m) return null;
    const rows = [
      ["elements", m.n_elements], ["nodes", m.n_nodes], ["DOFs", m.n_dofs],
      ["vertices", m.n_vertices], ["h min", m.h_min], ["h max", m.h_max],
      ["quality min", m.quality_min], ["quality mean", m.quality_mean],
    ].filter(([, v]) => v !== undefined);
    const hist = m.quality_hist;
    this.meshHistEl = hist && Array.isArray(hist.counts) ? h("div.chart.chart-sm") : null;
    this.meshHist = hist;
    const bnd = m.boundaries && typeof m.boundaries === "object" ? h("div.species", {},
      Object.entries(m.boundaries).map(([k, v]) => h("span.species-chip", {}, h("b", { text: k }), h("span.mono.dim", { text: ` ${fmtValue(v)} edges` })))) : null;
    return card("Mesh", [kvTable(rows), bnd, this.meshHistEl ? h("div.chart-caption", { text: "element quality histogram" }) : null, this.meshHistEl]);
  }

  convergenceCard(s, progress) {
    const series = convergenceSeries(s, progress);
    this.convSeries = series;
    if (!series.length) return card("Convergence", placeholder("empty", "No iteration history"), { wide: true });
    this.convEl = h("div.chart.chart-md");
    const stages = (s.solver && s.solver.stages) || [];
    const stageInfo = stages.length ? h("div.stage-chips", {}, stages.map((st, i) => h("span.stage-chip", {},
      h("span.swatch", { style: { background: SERIES_COLORS[i % SERIES_COLORS.length] } }),
      `stage ${i + 1}`, st.params ? h("span.mono.dim", { text: " " + Object.entries(st.params).map(([k, v]) => `${k}=${fmt(v)}`).join(" ") }) : null,
      st.converged === false ? badge("fail", "not conv.") : null))) : null;
    return card("Solver convergence", [this.convEl, stageInfo], { wide: true });
  }

  checksCard(s) {
    const checks = Array.isArray(s.checks) ? s.checks.slice() : [];
    if (!checks.length) return null;
    checks.sort((a, b) => (STATUS_ORDER[a.status] ?? 9) - (STATUS_ORDER[b.status] ?? 9));
    const counts = {};
    for (const c of checks) counts[c.status] = (counts[c.status] || 0) + 1;
    const summary = h("div.check-counts", {}, ["fail", "warn", "pass", "info"].filter((k) => counts[k])
      .map((k) => badge(k, `${counts[k]} ${k}`)));
    const rows = checks.map((c) => h(`tr.st-${c.status || "info"}`, {},
      h("td", {}, badge(c.status)),
      h("td", { text: c.name || "" }),
      h("td.mono.num", { text: fmtValue(c.value) }),
      h("td.mono.num.dim", { text: c.limit === undefined ? "" : fmtValue(c.limit) }),
      h("td.dim.small", { text: c.note || "" })));
    return card("Checks", h("table.grid-table.checks", {},
      h("thead", {}, h("tr", {}, ["", "check", "value", "limit", "note"].map((t) => h("th", { text: t })))),
      h("tbody", {}, rows)), { wide: true, extra: summary });
  }

  forcesCard(s) {
    const f = (s.integrals || {}).force;
    if (!f) return null;
    const vecRows = ["hydro", "electric", "total"].filter((k) => Array.isArray(f[k]));
    const total = Array.isArray(f.total) ? f.total : null;
    const table = vecRows.length ? h("table.grid-table", {},
      h("thead", {}, h("tr", {}, h("th", { text: "" }), h("th.num", { text: "F_y" }), h("th.num", { text: "F_z" }), h("th.num", { text: "share of |F_z|" }))),
      h("tbody", {}, vecRows.map((k) => h(`tr${k === "total" ? ".total" : ""}`, {},
        h("th", { text: k }),
        h("td.mono.num", { text: fmt(f[k][0]) }),
        h("td.mono.num", { text: fmt(f[k][1]) }),
        h("td.mono.num.dim", { text: total && k !== "total" && total[1] ? `${(100 * f[k][1] / total[1]).toFixed(1)} %` : "" })))))
      : null;
    const big = h("div.big-numbers", {},
      isNum(f.drag) ? h("div.bignum", {}, h("span.bn-label", { text: "drag" }), h("span.bn-value.mono", { text: fmt(f.drag, 5) })) : null,
      isNum(f.lift) ? h("div.bignum", {}, h("span.bn-label", { text: "lift" }), h("span.bn-value.mono", { text: fmt(f.lift, 4) })) : null);
    this.forceEl = vecRows.length >= 2 ? h("div.chart.chart-xs") : null;
    this.forceData = f;
    return card("Forces", [big, table, this.forceEl]);
  }

  fluxCard(s) {
    const flux = (s.integrals || {}).flux;
    if (!flux || typeof flux !== "object") return null;
    const species = Object.keys(flux);
    const NET = new Set(["net", "sum", "total"]);
    const bnds = [...new Set(species.flatMap((sp) => Object.keys(flux[sp] || {})))].filter((b) => !NET.has(b));
    if (!bnds.length) return null;
    // prefer the solver's own net value; otherwise sum the boundary fluxes
    const sum = (sp) => {
      for (const k of NET) if (isNum(flux[sp][k])) return flux[sp][k];
      return bnds.reduce((a, b) => a + (isNum(flux[sp][b]) ? flux[sp][b] : 0), 0);
    };
    return card("Species fluxes", [
      h("table.grid-table", {},
        h("thead", {}, h("tr", {}, h("th", { text: "species" }), bnds.map((b) => h("th.num", { text: b })), h("th.num", { text: "Σ (net)" }))),
        h("tbody", {}, species.map((sp) => h("tr", {},
          h("th", { text: prettyName(sp) }),
          bnds.map((b) => h("td.mono.num", { text: fmt(flux[sp][b]) })),
          h("td.mono.num.sum", { text: fmt(sum(sp)) }))))),
      h("div.chart-caption", { text: "outward normal flux ∫ N·n ds per boundary; Σ ≈ 0 for a conservative steady solution" }),
    ], { wide: bnds.length > 3 });
  }

  chargeCard(s) {
    const ig = s.integrals || {};
    const rows = [];
    for (const [k, v] of Object.entries(ig.charge || {})) rows.push([`charge · ${k}`, v]);
    for (const [k, v] of Object.entries(ig.other || {})) rows.push([k, v]);
    for (const [k, v] of Object.entries(ig)) {
      if (!["flux", "force", "charge", "other"].includes(k)) rows.push([k, v]);
    }
    if (!rows.length) return null;
    return card("Charge balance & other integrals", kvTable(rows));
  }

  sensitivityCard(s) {
    const sens = s.sensitivities;
    if (!sens || typeof sens !== "object" || !Object.keys(sens).length) return null;
    const qs = Object.keys(sens);
    const ps = [...new Set(qs.flatMap((q) => Object.keys(sens[q] || {})))];
    const params = s.params || {};
    // per-quantity max for the inline magnitude bars
    const rowMax = (q) => Math.max(...ps.map((p) => Math.abs(sens[q][p] || 0)), 1e-300);
    const cell = (q, p) => {
      const v = sens[q][p];
      if (!isNum(v)) return h("td.dim.num", { text: "—" });
      const frac = Math.abs(v) / rowMax(q);
      const Q = lookupQuantity(s, q);
      const el = isNum(params[p]) && isNum(Q) && Q !== 0 ? (v * params[p]) / Q : null;
      // diverging bar: zero in the middle of the track, negative grows left
      const bar = h("span.sens-track", {}, h("span.sens-bar", {
        class: v < 0 ? "neg" : "pos",
        style: { width: `${(frac * 50).toFixed(1)}%`, [v < 0 ? "right" : "left"]: "50%" },
      }));
      return h("td.sens", { title: el === null ? "" : `elasticity (p/Q)·∂Q/∂p = ${fmt(el, 3)}` },
        h("div.sens-cell", {}, bar, h("span.mono.sens-val", { text: fmt(v, 3) })));
    };
    return card("Sensitivities  ∂Q/∂p", [
      h("table.grid-table.sens-table", {},
        h("thead", {}, h("tr", {}, h("th", { text: "quantity Q" }), h("th.num", { text: "max |∂Q/∂p|" }),
          ps.map((p) => h("th.num", { text: `∂/∂${p}`, title: (s.param_info || {})[p] || "" })))),
        h("tbody", {}, qs.map((q) => h("tr", {}, h("th", { text: q }),
          h("td.mono.num.dim", { text: fmt(rowMax(q), 2) }), ps.map((p) => cell(q, p)))))),
      h("div.chart-caption", { text: "derivatives from automatic differentiation of the converged solution; bars: sign and magnitude relative to the row maximum; hover a cell for the elasticity (p/Q)·∂Q/∂p" }),
    ], { wide: true });
  }

  notesCard(s) {
    const notes = Array.isArray(s.notes) ? s.notes : s.notes ? [String(s.notes)] : [];
    if (!notes.length) return null;
    return card("Notes", h("ul.notes", {}, notes.map((n) => h("li", { text: n }))), { wide: true });
  }

  // -- charts (drawn after the cards are in the DOM so Plotly can size them) ---
  drawDeferred() {
    if (this.convEl) this.drawConvergence();
    if (this.meshHistEl) this.drawMeshHist();
    if (this.forceEl) this.drawForces();
  }

  drawConvergence() {
    const traces = [];
    const shapes = [];
    let x0 = 0;
    this.convSeries.forEach((st, i) => {
      const color = SERIES_COLORS[i % SERIES_COLORS.length];
      const xs = st.its.map((_, k) => x0 + k);
      traces.push({
        x: xs, y: st.its.map((it) => it.res_norm), name: `stage ${st.stage} ‖R‖`, type: "scatter", mode: "lines+markers",
        line: { color, width: 2 }, marker: { size: 6 },
        customdata: st.its.map((it) => [it.it, it.alpha, it.step_norm]),
        hovertemplate: `stage ${st.stage}, it %{customdata[0]}<br>‖R‖ = %{y:.3e}<br>α = %{customdata[1]}<br>‖δ‖ = %{customdata[2]:.3e}<extra></extra>`,
      });
      if (st.its.some((it) => isNum(it.step_norm))) {
        traces.push({
          x: xs, y: st.its.map((it) => it.step_norm), name: `stage ${st.stage} ‖δ‖`, type: "scatter", mode: "lines",
          line: { color, width: 1, dash: "dot" }, hoverinfo: "skip", showlegend: false, legendgroup: "step",
        });
      }
      if (i > 0) shapes.push({ type: "line", x0: x0 - 0.5, x1: x0 - 0.5, yref: "paper", y0: 0, y1: 1, line: { color: "rgba(128,128,128,0.5)", width: 1, dash: "dash" } });
      x0 += st.its.length;
    });
    // one neutral legend entry explaining the dotted step-norm lines
    if (traces.some((t) => t.legendgroup === "step")) {
      traces.push({ x: [null], y: [null], type: "scatter", mode: "lines", name: "dotted: step norm ‖δ‖",
        line: { color: "#868e96", width: 1, dash: "dot" }, hoverinfo: "skip" });
    }
    plot(this.convEl, traces, baseLayout({
      xaxis: { title: { text: "Newton iteration (cumulative)" }, dtick: x0 > 14 ? undefined : 1 },
      yaxis: { title: { text: "residual norm" }, type: "log" },
      shapes, legend: { orientation: "h", x: 1, xanchor: "right", y: 1.02, yanchor: "bottom" },
      margin: { l: 60, r: 16, t: 30, b: 44 }, hovermode: "closest",
    }));
  }

  drawMeshHist() {
    const { edges, counts } = this.meshHist;
    const centers = counts.map((_, i) => (isNum(edges[i]) && isNum(edges[i + 1]) ? (edges[i] + edges[i + 1]) / 2 : i));
    const widths = counts.map((_, i) => (isNum(edges[i]) && isNum(edges[i + 1]) ? (edges[i + 1] - edges[i]) * 0.92 : 0.9));
    plot(this.meshHistEl, [{
      type: "bar", x: centers, y: counts, width: widths, marker: { color: SERIES_COLORS[0] },
      hovertemplate: "q ∈ %{x:.2f}: %{y} elements<extra></extra>",
    }], baseLayout({
      margin: { l: 48, r: 8, t: 4, b: 32 }, xaxis: { title: { text: "quality" } },
      yaxis: { type: Math.max(...counts) > 50 * Math.max(1, Math.min(...counts.filter((c) => c > 0))) ? "log" : "linear" },
      showlegend: false, bargap: 0,
    }), { displayModeBar: false });
  }

  drawForces() {
    const f = this.forceData;
    const parts = ["hydro", "electric"].filter((k) => Array.isArray(f[k]));
    plot(this.forceEl, parts.map((k, i) => ({
      type: "bar", orientation: "h", y: ["F_z "], x: [f[k][1]], name: k,
      marker: { color: SERIES_COLORS[i === 0 ? 0 : 3] }, hovertemplate: `${k}: %{x:.4g}<extra></extra>`,
    })), baseLayout({
      barmode: "relative", margin: { l: 36, r: 12, t: 26, b: 22 },
      legend: { orientation: "h", x: 0, y: 1.0, yanchor: "bottom" },
      xaxis: { zeroline: true, title: { text: "" } }, height: 104,
    }), { displayModeBar: false });
  }
}

/** Iteration history per stage from summary.solver.stages, else progress.history. */
export function convergenceSeries(summary, progress) {
  const stages = summary && summary.solver && Array.isArray(summary.solver.stages) ? summary.solver.stages : [];
  const out = [];
  stages.forEach((st, i) => {
    const its = (st.iterations || []).filter((it) => isNum(it.res_norm));
    if (its.length) out.push({ stage: i + 1, its });
  });
  if (out.length) return out;
  const hist = progress && Array.isArray(progress.history) ? progress.history : [];
  const by = new Map();
  for (const it of hist) {
    if (!isNum(it.res_norm)) continue;
    const s = it.stage ?? 1;
    if (!by.has(s)) by.set(s, []);
    by.get(s).push(it);
  }
  return [...by.entries()].sort((a, b) => a[0] - b[0]).map(([stage, its]) => ({ stage, its }));
}

/** Find a scalar quantity named `q` in summary.integrals (for elasticities). */
function lookupQuantity(s, q) {
  const ig = s.integrals || {};
  if (ig.force && isNum(ig.force[q])) return ig.force[q];
  for (const grp of ["charge", "other"]) {
    if (ig[grp] && isNum(ig[grp][q])) return ig[grp][q];
    const short = q.replace(/^charge_/, "");
    if (grp === "charge" && ig[grp] && isNum(ig[grp][short])) return ig[grp][short];
  }
  return null;
}
