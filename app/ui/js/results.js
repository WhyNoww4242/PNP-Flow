// Results workbench: field map, profiles, checks (with convergence and integrals),
// mesh views + diagnose, figure gallery and log.

import { h, setChildren, fmt, fmtValue, fmtDuration, isNum, prettyName, placeholder, isSignedFieldName } from "./util.js";
import { icon } from "./icons.js";
import { bridge } from "./bridge.js";
import { plot, baseLayout, seriesColors, residualPlot } from "./plots.js";
import { colormapSource } from "./colormaps.js";
import { convergenceSeries } from "./progress.js";
import { checksList, checkCounts, renderMesh } from "./pages.js";
import { caseTitle } from "./start.js";

export const RESULT_VIEWS = [
  { id: "field", label: "Field map", icon: "field-map" },
  { id: "profiles", label: "Profiles", icon: "profiles" },
  { id: "checks", label: "Checks and convergence", icon: "checks" },
  { id: "mesh", label: "Mesh views", icon: "mesh-views" },
  { id: "figures", label: "Figures", icon: "figures" },
  { id: "log", label: "Log", icon: "terminal" },
];

function panel(title, body, extra) {
  return h("section.panel", {}, h("header.panel-head", {}, h("h2", { text: title }), h("span.spacer"), extra || null), h("div.panel-body", {}, body));
}

/** Render the Results workbench for `doc` into `root`. */
export function renderResults(app, doc, root) {
  const view = RESULT_VIEWS.find((v) => v.id === doc.rview) || RESULT_VIEWS[0];
  const s = doc.summary || {};
  const conv = s.solver ? s.solver.converged : undefined;
  const head = h("div.results-head", {},
    icon(view.icon, 16), h("h2", { text: view.label }),
    h("span.rh-meta", { text: [doc.name, s.case ? caseTitle(s.case) : null, s.created ? String(s.created).replace("T", " ").slice(0, 16) : null].filter(Boolean).join("  ·  ") }),
    h("span.spacer"),
    conv === true ? h("span.badge.b-ok", {}, icon("check", 12), "converged") : conv === false ? h("span.badge.b-bad", {}, icon("x", 12), "not converged") : null,
    Array.isArray(s.checks) && s.checks.length ? h("button.btn-link", { type: "button", onclick: () => app.setResultsView("checks"), "aria-label": "Show checks" }, checkCounts(s.checks)) : null);
  const body = h("div.results-body");
  setChildren(root, h("div.results", {}, head, body));
  if (!doc.name) {
    setChildren(body, h("div.page", {}, placeholder("empty", "No results yet", "Solve this project in the Study workbench; the field map, profiles, checks and figures appear here when the run finishes.",
      h("button.btn.primary", { type: "button", onclick: () => app.setWorkbench("study") }, icon("play", 12), "Go to Study"))));
    return;
  }
  ({ field: showField, profiles: showProfiles, checks: showChecks, mesh: showMesh, figures: showFigures, log: showLog })[view.id](app, doc, body);
}

// ---------------------------------------------------------------------------
async function showField(app, doc, body) {
  const fv = app.field;
  const host = app.fieldHost;
  body.appendChild(host);
  const src = colormapSource();
  if (src.fallback) {
    body.appendChild(h("div.cm-notice", { role: "note", title: src.reason }, icon("info", 14), "Built-in colormaps (data/colormaps.json not found)"));
  }
  if (!doc.listing || !doc.listing.files || !doc.listing.files["fields.json"]) {
    fv.mesh = null;
    fv.showMessage("empty", "No field data", doc.listing ? "This run has no fields.json (yet). It is written when the run finishes." : "Loading the run…");
    return;
  }
  if (fv.docId === doc.id && fv.mesh && fv.meshKey === app.meshKey(doc)) { fv.resize(); fv.redraw(); return; }
  const size = doc.listing.files["fields.json"].size;
  const bar = h("span.stage-bar", { style: { width: "220px", display: "block", margin: "8px auto 0" } }, h("span.fill", { style: { width: "0%" } }));
  fv.showMessage("loading", "Loading the mesh and fields", h("div", {}, h("span.mono", { text: `fields.json, ${(size / 1048576).toFixed(1)} MB` }), bar));
  try {
    const mesh = await app.loadMesh(doc, (frac, got) => { bar.firstChild.style.width = `${(frac ?? got / size) * 100}%`; });
    if (app.activeDoc() !== doc || doc.wb !== "results" || doc.rview !== "field") return;
    if (!mesh) return fv.showMessage("empty", "No field data", "fields.json was not found.");
    fv.docId = doc.id;
    fv.meshKey = app.meshKey(doc);
    fv.setMesh(mesh);
    app.loadTimeSeries(doc, mesh).then((ts) => { if (ts && fv.mesh === mesh) fv.setTimeSeries(ts); });
    if (mesh.bad) app.toast(`fields.json: ${mesh.bad} triangle indices were out of range and were clamped`, "warn");
  } catch (e) {
    fv.showMessage("error", "Could not read fields.json", e.message);
  }
}

// ---------------------------------------------------------------------------
function showProfiles(app, doc, body) {
  const data = doc.profiles;
  const lines = data && Array.isArray(data.lines) ? data.lines.filter((l) => l && Array.isArray(l.coord)) : [];
  if (!lines.length) {
    setChildren(body, h("div.page", {}, placeholder("empty", "No profiles", "This run has no profiles.json, or it contains no lines.")));
    return;
  }
  const ps = (doc.pstate = doc.pstate || { line: null, vars: {}, logy: false, split: true, normalize: false });
  const line = lines.find((l) => l.name === ps.line) || lines[0];
  ps.line = line.name;
  const avail = Object.keys(line.values || {});
  let chosen = (ps.vars[line.name] || []).filter((v) => avail.includes(v));
  if (!chosen.length) chosen = avail.slice(0, Math.min(3, avail.length));
  const rerender = () => showProfiles(app, doc, body);
  const opt = (key, label, title) => h("label.chk", { title }, h("input", { type: "checkbox", checked: !!ps[key], onchange: (e) => { ps[key] = e.target.checked; draw(); } }), h("span", { text: label }));
  const plotEl = h("div.prof-plot");
  setChildren(body, h("div.profiles", {},
    h("aside.prof-side", { "aria-label": "Profile lines" }, h("div.pgroup", { text: "Lines" }),
      lines.map((l) => h("button.prof-line", { type: "button", "aria-pressed": String(l.name === line.name), onclick: () => { ps.line = l.name; rerender(); } },
        h("span.pl-name", { text: l.name }), l.label ? h("span.pl-label", { text: l.label }) : null))),
    h("div.prof-main", {},
      h("div.prof-bar", {},
        h("span.bar-label", { text: "Variables" }),
        h("div.chips", {}, avail.map((v) => h("button.chip", { type: "button", "aria-pressed": String(chosen.includes(v)),
          onclick: () => { const next = chosen.includes(v) ? chosen.filter((x) => x !== v) : [...chosen, v]; ps.vars[line.name] = next.length ? next : chosen; rerender(); }, text: prettyName(v) }))),
        h("span", { style: { flex: "1" } }),
        opt("logy", "Log y", "Logarithmic y axis (absolute values)"), opt("split", "One plot per variable", "Stack one subplot per variable"), opt("normalize", "Normalise", "Divide each curve by its largest absolute value")),
      plotEl)));
  function draw() {
    const colors = seriesColors(Math.max(chosen.length, 2));
    const split = ps.split && chosen.length > 1;
    const traces = chosen.map((v, i) => {
      let y = (line.values[v] || []).map((a) => (a === null || a === undefined ? NaN : +a));
      if (ps.normalize) { const m = Math.max(...y.filter(Number.isFinite).map(Math.abs)) || 1; y = y.map((a) => a / m); }
      if (ps.logy) y = y.map((a) => (Math.abs(a) > 0 ? Math.abs(a) : NaN));
      return { x: line.coord, y, type: "scatter", mode: line.coord.length < 60 ? "lines+markers" : "lines", name: prettyName(v),
        line: { color: colors[i % colors.length], width: 2.5 }, marker: { size: 5 }, yaxis: split && i ? `y${i + 1}` : "y",
        hovertemplate: `${prettyName(v)}: %{y:.5g}<br>${prettyName(line.coord_name || "s")}: %{x:.5g}<extra></extra>` };
    });
    const lay = baseLayout({ xaxis: { title: { text: prettyName(line.coord_name || "s") } }, yaxis: { type: ps.logy ? "log" : "linear" }, hovermode: "x unified", showlegend: !split, legend: { orientation: "h", y: 1.06, x: 1, xanchor: "right" }, margin: { l: 70, r: 16, t: 26, b: 44 } });
    if (split) {
      const n = chosen.length, gap = 0.05, hgt = (1 - gap * (n - 1)) / n;
      lay.xaxis = Object.assign({}, lay.xaxis, { anchor: `y${n}` });
      chosen.forEach((v, i) => {
        const key = i === 0 ? "yaxis" : `yaxis${i + 1}`;
        const top = 1 - i * (hgt + gap);
        lay[key] = Object.assign({}, lay.yaxis, { domain: [top - hgt, top], anchor: "x", type: ps.logy ? "log" : "linear",
          title: { text: prettyName(v), font: { size: 12, color: colors[i % colors.length] } }, zeroline: !ps.logy && isSignedFieldName(v) });
      });
    }
    plot(plotEl, traces, lay);
  }
  requestAnimationFrame(draw);
}

// ---------------------------------------------------------------------------
function showChecks(app, doc, body) {
  const s = doc.summary;
  if (!s) {
    setChildren(body, h("div.page", {}, placeholder("empty", "No summary yet", "summary.json is written when the run finishes; checks and convergence appear then.")));
    return;
  }
  const checks = Array.isArray(s.checks) ? s.checks : [];
  const convEl = h("div.chart.chart-md");
  const solver = s.solver || {};
  const ig = s.integrals || {};
  const igRows = [];
  for (const [grp, vals] of Object.entries(ig)) {
    if (vals && typeof vals === "object" && !Array.isArray(vals)) {
      for (const [k, v] of Object.entries(vals)) {
        if (v && typeof v === "object" && !Array.isArray(v)) for (const [k2, v2] of Object.entries(v)) igRows.push([`${grp} · ${k} · ${k2}`, v2]);
        else igRows.push([`${grp} · ${k}`, v]);
      }
    } else igRows.push([grp, vals]);
  }
  const notes = Array.isArray(s.notes) ? s.notes : s.notes ? [String(s.notes)] : [];
  const err = s.error || (doc.progress && doc.progress.state === "failed" ? doc.progress.message : null);
  setChildren(body, h("div.page", {}, h("div.page-pad", {}, h("div.stack", { style: { maxWidth: "1240px" } },
    err ? h("div.msg.msg-error", { role: "alert" }, icon("x-circle", 16), h("div", {}, h("b", { text: "The run failed. " }), h("span.mono", { text: String(err).trim().split(/\n/).pop() }))) : null,
    panel("Checks", checks.length ? checksList(checks) : h("div.dim", { text: "This run reports no checks." }), checks.length ? checkCounts(checks) : null),
    h("div.grid2", {},
      panel("Newton convergence", [convEl, h("div.small.dim", { text: [
        isNum(solver.n_newton_total) ? `${solver.n_newton_total} Newton iterations` : null,
        Array.isArray(solver.stages) ? `${solver.stages.length} continuation stage${solver.stages.length === 1 ? "" : "s"}` : null,
        isNum(solver.time_s) ? `solve time ${fmtDuration(solver.time_s)}` : null].filter(Boolean).join(", ") })]),
      igRows.length ? panel("Integrals", h("table.kv", {}, h("tbody", {}, igRows.slice(0, 40).map(([k, v]) => h("tr", {}, h("th", { text: k }), h("td.mono.num", { text: fmtValue(v) })))))) : null),
    notes.length ? panel("Notes", h("ul", { style: { margin: 0, paddingLeft: "18px" } }, notes.map((n) => h("li", { text: n })))) : null))));
  const series = convergenceSeries(s, doc.progress);
  requestAnimationFrame(() => {
    if (!series.length) return setChildren(convEl, placeholder("empty", "No iteration history"));
    residualPlot(convEl, series, { label: doc.progress && doc.progress.time ? "time step" : "stage" });
  });
}

function showMesh(app, doc, body) {
  renderMesh(app, doc, body, { resultOnly: true });
}

// ---------------------------------------------------------------------------
function showFigures(app, doc, body) {
  const l = doc.listing || {};
  const groups = [
    { title: "Figures", note: "house style, 300 dpi (figures/)", files: (l.figures || []).map((f) => `figures/${f}`) },
    { title: "Quick plots", note: "plots/", files: (l.plots || []).filter((f) => /\.png$/i.test(f)).map((f) => `plots/${f}`) },
  ].filter((g) => g.files.length);
  if (!groups.length) {
    setChildren(body, h("div.page", {}, placeholder("empty", "No figures", "This run has no figures/*.png or plots/*.png. Figures are written automatically when a run finishes.")));
    return;
  }
  const all = groups.flatMap((g) => g.files);
  const svgs = new Set((l.svgs || []).map((f) => f.replace(/\.svg$/i, "")));
  setChildren(body, h("div.page", {}, h("div.page-pad", {}, groups.map((g) => h("section.gallery-group", {},
    h("h3", {}, g.title, h("span", { text: `${g.files.length}, ${g.note}` })),
    h("div.gallery", {}, g.files.map((f) => {
      const img = h("img", { alt: f, loading: "lazy" });
      bridge.imageURL(doc.name, f).then((u) => { img.src = u; }).catch(() => {});
      const base = f.split("/").pop().replace(/\.png$/i, "");
      return h("button.thumb", { type: "button", onclick: () => app.lightbox(all.map((x) => ({ name: doc.name, file: x, caption: x })), all.indexOf(f)), "aria-label": `Open ${f}` },
        h("div.thumb-img", {}, img), h("span.cap", { text: base.replace(/_/g, " ") + (g.title === "Figures" && svgs.has(base) ? "  (SVG too)" : "") }));
    })))))));
}

// ---------------------------------------------------------------------------
async function showLog(app, doc, body) {
  const pre = h("pre.log.mono", { tabindex: "0", "aria-label": "Solver log", style: { position: "absolute", inset: "0" } }, "Loading stdout.log…");
  setChildren(body, pre);
  try {
    const r = await bridge.log(doc.name, 512 * 1024);
    pre.textContent = r.size ? r.text : "No stdout.log for this run. Runs started from the command line write their log to the terminal instead.";
    pre.scrollTop = pre.scrollHeight;
  } catch (e) {
    pre.textContent = `Could not read the log: ${e.message}`;
  }
}

