// Workbench pages for the construction steps: Geometry, Mesh, Materials,
// Boundaries, Equations and Study (solve + live status). Results live in results.js.
//
// Editing geometry, ions and boundary conditions needs the project file
// (docs/PLAN.md Phase 1); until then these pages show what the template or run
// defines, and parameters are edited in the Data tab.

import { h, setChildren, fmt, fmtValue, isNum, prettyName, placeholder, stripUnit } from "./util.js";
import { icon } from "./icons.js";
import { bridge } from "./bridge.js";
import { plot, baseLayout, seriesColors } from "./plots.js";
import { boundaryColor } from "./fieldview.js";
import { runStatus, healthBadge, stageBars, tellsGrid, convergenceSeries } from "./progress.js";
import { caseTitle, caseDoc } from "./start.js";

const GEOM_RE = /^(R|R_out|radius|length|width|half_width|upstream|downstream|L|H|W|r_?\w*|h_\w+|.*_size|domain\w*|far\w*)$/;

function panel(title, body, extra) {
  return h("section.panel", {}, h("header.panel-head", {}, h("h2", { text: title }), h("span.spacer"), extra || null), h("div.panel-body", {}, body));
}

function kv(rows) {
  return h("table.kv", {}, h("tbody", {}, rows.filter(Boolean).map(([k, v, tip]) =>
    h("tr", { title: tip || null }, h("th", { text: k }), h("td", {}, v instanceof Node ? v : typeof v === "number" ? h("span.mono", { text: fmt(v) }) : h("span", { text: fmtValue(v) }))))));
}

function stub(text) {
  return h("div.stub", {}, icon("info", 16), h("span", {}, text));
}

function pageShell(wb, title, lead, ...content) {
  return h("div.page", {}, h("div.page-pad", {}, h("div.wb-page", {},
    h("h1", {}, icon(wb, 20), title), lead ? h("p.lead", {}, lead) : null, ...content)));
}

function imageFigure(app, doc, file, caption, check) {
  const img = h("img", { alt: caption, loading: "lazy" });
  bridge.imageURL(doc.name, file).then((url) => { img.src = url; }).catch(() => {});
  img.addEventListener("click", () => app.lightbox([{ name: doc.name, file, caption }], 0));
  return h("figure.img-frame", {}, h("div.img-box", {}, img),
    h("figcaption", {}, h("div", { text: caption }), check ? h("div.small", { text: `Check: ${check}` }) : null));
}

export function checkStatusBadge(status) {
  const map = { pass: ["b-ok", "check", "pass"], warn: ["b-warn", "alert", "warning"], fail: ["b-bad", "x", "fail"], info: ["b-info", "info", "info"] };
  const [cls, ic, word] = map[status] || map.info;
  return h("span.badge", { class: cls }, icon(ic, 12), word);
}

export function checksList(checks, { withFix = false } = {}) {
  const order = { fail: 0, warn: 1, pass: 2, info: 3 };
  const sorted = (checks || []).slice().sort((a, b) => (order[a.status] ?? 9) - (order[b.status] ?? 9));
  return h("div.checks-list", { role: "table" },
    h("div.check.check-head", { role: "row" }, ["status", "check", "value", "limit", "note"].map((t, i) => h("span", { role: "columnheader", text: t, style: i === 2 || i === 3 ? { textAlign: "right" } : null }))),
    sorted.map((c) => h("div.check", { role: "row" },
      h("span", {}, checkStatusBadge(c.status)),
      h("span.c-name", { text: c.name || "" }),
      h("span.c-val", { text: c.value === undefined || c.value === null ? "–" : fmtValue(c.value) }),
      h("span.c-lim", { text: c.limit === undefined || c.limit === null ? "" : fmtValue(c.limit) }),
      h("span.c-note", {}, c.note || "", withFix && c.fix && c.status !== "pass" ? h("div", { text: `Fix: ${c.fix}` }) : null))));
}

export function checkCounts(checks) {
  const counts = {};
  for (const c of checks || []) counts[c.status] = (counts[c.status] || 0) + 1;
  return h("span.check-sum", {}, ["fail", "warn", "pass", "info"].filter((k) => counts[k]).map((k) => {
    const b = checkStatusBadge(k);
    b.lastChild.textContent = `${counts[k]} ${k === "warn" ? "warning" : k}`;
    return b;
  }));
}

// ---------------------------------------------------------------------------
export function renderGeometry(app, doc, root) {
  const c = app.caseOf(doc);
  const rows = app.paramRows(doc).filter((r) => GEOM_RE.test(r.key));
  const bnames = app.boundaryNames(doc);
  const sym = app.symmetryOf(doc);
  const left = panel("Template", kv([
    ["template", h("span", {}, caseTitle(doc.case), " ", h("code", { text: doc.case }))],
    ["coordinates", sym === "axisym" ? "axisymmetric (r, z), r ≥ 0" : sym === "planar" ? "planar 2D (y, z)" : "known after the first solve"],
    ["geometry kernel", "gmsh (OCC), built from the template parameters"],
  ]));
  const params = panel("Geometry parameters", rows.length ? h("table.tbl", {}, h("tbody", {}, rows.map((r) =>
    h("tr", {}, h("th", { class: "mono", text: r.key }), h("td.num", { text: fmt(r.value) }), h("td", { class: "dim", text: r.unit || "" }), h("td.dim.small", { text: stripUnit(r.info) }))))
  ) : h("div.dim", { text: "This template has no separate geometry parameters." }), h("button.btn-link", { type: "button", onclick: () => app.setPropsTab("data") }, "Edit in the Data tab"));
  const bnd = panel("Named boundaries", bnames.length ? h("table.tbl", {}, h("tbody", {}, bnames.map((b, i) =>
    h("tr", {}, h("th", {}, h("span.swatch", { class: /axis|symmetry/.test(b) ? "dashed" : "", style: /axis|symmetry/.test(b) ? { borderColor: boundaryColor(b, i), width: "14px" } : { background: boundaryColor(b, i) } })),
      h("td.mono", { text: b }), h("td.dim.small", { text: app.bcSummary(doc, b) })))))
    : h("div.dim", { text: "Boundary names are known after the mesh is built (solve once)." }));
  const views = (doc.listing && doc.listing.mesh_files || []).filter((f) => /view_a/.test(f));
  setChildren(root, pageShell("geometry", "Geometry",
    caseDoc(c) || "The geometry comes from the template.",
    h("div.grid2", {}, h("div.stack", {}, left, params), h("div.stack", {}, bnd,
      views.length ? imageFigure(app, doc, views[0], "Whole domain, boundaries coloured by group") : null)),
    stub("Geometry editing (template preview, CAD and mesh import, gmsh scripts) arrives with the project file. For now the template parameters in the Data tab define the geometry.")));
}

// ---------------------------------------------------------------------------
export function renderMesh(app, doc, root) {
  const s = doc.summary || {};
  const m = s.mesh;
  const diag = (doc.diagnose && doc.diagnose.checks) || s.mesh_checks || null;
  const viewsMeta = Array.isArray(s.mesh_views) ? s.mesh_views : [];
  const files = (doc.listing && doc.listing.mesh_files) || [];
  const content = [];
  if (!m && !files.length) {
    content.push(h("div.panel", {}, placeholder("empty", doc.kind === "draft" ? "The mesh is built when you solve" : "No mesh information in this run",
      doc.kind === "draft" ? "gmsh meshes the template with curved P2 triangles, refined near charged walls (h ≈ λ_D / 8 within 3 λ_D). The two mesh views and the diagnose checks appear here after the run." : "summary.json has no mesh section.")));
  } else {
    const histEl = m && m.quality_hist && Array.isArray(m.quality_hist.counts) ? h("div.chart.chart-sm") : null;
    const stats = m ? panel("Mesh statistics", [kv([
      ["elements", m.n_elements], ["nodes (P2)", m.n_nodes], ["vertices", m.n_vertices], ["unknowns (DOFs)", m.n_dofs],
      ["smallest element h", m.h_min], ["largest element h", m.h_max], ["min quality", m.quality_min], ["mean quality", m.quality_mean],
    ].filter(([, v]) => v !== undefined)), histEl ? h("div.bar-label", { text: "Element quality (1 = equilateral)", style: { marginTop: "10px" } }) : null, histEl]) : null;
    const diagPanel = diag ? panel("Mesh diagnose", checksList(diag, { withFix: true }), checkCounts(diag)) : panel("Mesh diagnose", h("div.dim", { text: "This run has no mesh/diagnose.json (older runs were not diagnosed)." }));
    const views = files.filter((f) => /\.png$/.test(f)).map((f) => {
      const meta = viewsMeta.find((v) => v.file === f) || {};
      return imageFigure(app, doc, f, meta.title || (/view_a/.test(f) ? "View A: whole domain by boundary group" : "View B: finest region by quality"), meta.what_to_check);
    });
    content.push(views.length ? h("div.mesh-views", {}, views) : null,
      h("div.grid-1-2", { style: { marginTop: "12px" } }, stats, diagPanel));
    if (histEl) requestAnimationFrame(() => {
      const { edges, counts } = m.quality_hist;
      const centers = counts.map((_, i) => (isNum(edges[i]) && isNum(edges[i + 1]) ? (edges[i] + edges[i + 1]) / 2 : i));
      plot(histEl, [{ type: "bar", x: centers, y: counts, marker: { color: seriesColors(3)[1] }, hovertemplate: "quality %{x:.2f}: %{y} elements<extra></extra>" }],
        baseLayout({ margin: { l: 48, r: 8, t: 4, b: 30 }, xaxis: { title: { text: "quality" }, range: [0, 1] }, yaxis: { type: "log", title: { text: "elements" } }, showlegend: false, bargap: 0.05 }),
        { displayModeBar: false });
    });
  }
  setChildren(root, pageShell("mesh", "Mesh", "Every mesh is rendered from two views and diagnosed before it is trusted.", ...content,
    stub("Mesh controls (sizes, refinement layers, size boxes) arrive with the project file; the template sets them from the Debye length today.")));
}

// ---------------------------------------------------------------------------
export function renderMaterials(app, doc, root) {
  const s = doc.summary || {};
  const species = (s.model && Array.isArray(s.model.species)) ? s.model.species : null;
  const rows = app.paramRows(doc).filter((r) => /^(T|delta|d_\w+|zeta\w*|stern\w*|q|eps\w*|c0|bulk\w*|z_\w+)$/.test(r.key));
  let spBody;
  if (species === null) spBody = h("div.dim", { text: doc.kind === "draft" ? "The species list is written when the template is solved." : "This run does not list its species." });
  else if (!species.length) spBody = h("div", {}, h("p", { style: { margin: "0 0 6px" } }, "No mobile ions: this model solves the Poisson equation only (fixed charge in a uniform dielectric)."));
  else spBody = h("table.tbl", {}, h("thead", {}, h("tr", {}, ["species", "valence z", "diffusivity D / D0", "bulk c / c0"].map((t, i) => h("th", { class: i ? "num" : "", text: t })))),
    h("tbody", {}, species.map((sp) => h("tr", {}, h("th", { text: prettyName(sp.name) }), h("td.num", { text: `${sp.z > 0 ? "+" : ""}${fmt(sp.z)}` }), h("td.num", { text: fmt(sp.d) }), h("td.num", { text: fmt(sp.bulk) })))));
  setChildren(root, pageShell("materials", "Materials", "Solvent, temperature and the ions in the electrolyte.",
    h("div.grid2", {}, panel("Species", spBody),
      panel("Material parameters", rows.length ? h("table.tbl", {}, h("tbody", {}, rows.map((r) => h("tr", {}, h("th.mono", { text: r.key }), h("td.num", { text: fmt(r.value) }), h("td.dim", { text: r.unit }), h("td.dim.small", { text: stripUnit(r.info) })))))
        : h("div.dim", { text: "No material parameters in this template." }))),
    stub("An ion library (valence and diffusivity at 25 °C), custom ions and lab units arrive with the project file.")));
}

// ---------------------------------------------------------------------------
export function renderBoundaries(app, doc, root) {
  const s = doc.summary || {};
  const bcs = (s.model && s.model.boundaries) || {};
  const names = app.boundaryNames(doc);
  const diag = (doc.diagnose && doc.diagnose.checks) || s.mesh_checks || [];
  const coverage = diag.filter((c) => /BC|boundar/i.test(c.name));
  let body;
  if (!names.length && !Object.keys(bcs).length) {
    body = placeholder("empty", "Boundaries are named when the mesh is built", "Solve once to see each named boundary and the condition on it.");
  } else {
    const all = [...new Set([...names, ...Object.keys(bcs)])];
    const keys = [...new Set(Object.values(bcs).flatMap((v) => Object.keys(v || {})))];
    body = h("table.tbl", {}, h("thead", {}, h("tr", {}, h("th", { text: "boundary" }), keys.map((k) => h("th", { text: prettyName(k) })), keys.length ? null : h("th", { text: "condition" }))),
      h("tbody", {}, all.map((b, i) => h("tr", {},
        h("th", {}, h("span", { style: { display: "inline-flex", alignItems: "center", gap: "8px" } },
          h("span.swatch", { class: /axis|symmetry/.test(b) ? "dashed" : "", style: /axis|symmetry/.test(b) ? { borderColor: boundaryColor(b, i), width: "14px" } : { background: boundaryColor(b, i) } }), h("span.mono", { text: b }))),
        keys.length ? keys.map((k) => h("td.mono", { text: bcs[b] && bcs[b][k] !== undefined ? String(bcs[b][k]) : "natural (default)" }))
          : h("td.dim", { text: "natural (default)" })))));
  }
  setChildren(root, pageShell("boundary", "Boundaries", "The condition on every named boundary. Boundaries without an entry use the natural condition: blocking for ions, no surface charge, no-slip.",
    h("div.stack", {}, panel("Conditions per boundary", body), coverage.length ? panel("Coverage check", checksList(coverage, { withFix: true })) : null),
    stub("Choosing conditions per boundary (Dirichlet, flux, Stern, surface charge, inlet/outlet) arrives with the project file.")));
}

// ---------------------------------------------------------------------------
export function renderEquations(app, doc, root) {
  const s = doc.summary || {};
  const model = s.model || {};
  const known = !!s.model;
  const species = Array.isArray(model.species) ? model.species : [];
  const flow = model.flow || (known ? "none" : null);
  const axisym = model.coord === "axisym" || app.symmetryOf(doc) === "axisym";
  const eq = (on, name, math, note) => h("div.eq", { class: on ? "" : "off" },
    h("span.eq-state", {}, on === null ? h("span.sdot.s-none", { title: "not known yet" }) : on ? h("span.sdot.s-done", { title: "active" }) : h("span.sdot.s-todo", { title: "not used" })),
    h("span.eq-name", {}, name, h("div.small.dim", { text: on === null ? "known after solving" : on ? "active" : "not used" })),
    h("span", {}, h("span.eq-math", { html: math }), note ? h("div.small.dim", { text: note }) : null));
  const list = h("div.eq-list", {},
    eq(true, "Poisson", "&minus;&delta;<sup>2</sup> &nabla;<sup>2</sup>&phi; = &rho; / &rho;<sub>n</sub> + s<sub>&phi;</sub>", axisym ? "axisymmetric: &nabla;² in (r, z) with the r dr dz measure".replace("&nabla;", "∇") : null),
    eq(known ? species.length > 0 : null, "Nernst–Planck", "&nabla;&middot;N<sub>i</sub> = s<sub>i</sub>, &nbsp; N<sub>i</sub> = c<sub>i</sub>u &minus; (d<sub>i</sub>/Pe)(&nabla;c<sub>i</sub> + z<sub>i</sub>c<sub>i</sub>&nabla;&phi;)",
      species.length ? `${species.length} species: ${species.map((x) => prettyName(x.name)).join(", ")}` : null),
    eq(flow === null ? null : flow === "stokes" || flow === "navier_stokes", flow === "navier_stokes" ? "Navier–Stokes" : "Stokes flow",
      "Re (u&middot;&nabla;)u = &minus;&nabla;p + &nabla;<sup>2</sup>u &minus; K &rho; &nabla;&phi;, &nbsp; &nabla;&middot;u = 0", flow ? `flow model: ${flow}${model.advection ? `, ${model.advection} advection` : ""}${model.supg ? ", SUPG" : ""}` : null));
  setChildren(root, pageShell("equations", "Governing equations", "Steady, nondimensional. Jacobians come from JAX automatic differentiation, not hand-written derivatives.",
    panel("Physics", list),
    stub("Switching physics on and off (ions, flow type, advection form) arrives with the project file; the template fixes it today.")));
}

// ---------------------------------------------------------------------------
// Study: solve, live progress, health, residual plot and log tail
// ---------------------------------------------------------------------------
export function renderStudy(app, doc, root) {
  const rows = app.paramRows(doc);
  const changed = rows.filter((r) => r.changed);
  const nameIn = h("input.inp#run-name", { type: "text", spellcheck: "false", value: doc.nextName || app.suggestName(doc), "aria-label": "Run name",
    oninput: (e) => { doc.nextName = e.target.value; updateCmd(); } });
  const overwrite = h("input", { type: "checkbox", checked: !!doc.overwrite, onchange: (e) => { doc.overwrite = e.target.checked; } });
  const cmd = h("code.cmd");
  const msg = h("div", { "aria-live": "polite" });
  const updateCmd = () => { cmd.textContent = app.commandLine(doc, nameIn.value.trim()); };
  updateCmd();
  const st = app.statusOf(doc);
  const solveBtn = h("button.btn.primary", { type: "button", onclick: () => app.solve(doc, { name: nameIn.value.trim(), overwrite: overwrite.checked, msgEl: msg }) },
    icon("play", 12), doc.kind === "run" ? "Solve again" : "Solve");
  const stopBtn = h("button.btn.danger", { type: "button", onclick: () => app.stop(doc), disabled: !(st && st.running && app.canStop(doc)) || null, title: "Stop this run (Shift+F8)" }, icon("stop", 12), "Stop");
  if (st && st.running) { solveBtn.disabled = true; solveBtn.title = "A run of this document is in progress"; }
  const form = panel("Solve", h("div.solve-form", {},
    changed.length ? h("div.small", {}, `${changed.length} parameter${changed.length > 1 ? "s" : ""} changed from the template: `,
      h("span.mono", { text: changed.map((r) => `${r.key} = ${fmt(r.value)}`).join(", ") })) : h("div.small.dim", { text: rows.length ? "All parameters at their template values (edit them in the Data tab)." : "Template parameters are not loaded yet." }),
    h("div.form-row", {}, h("label.lbl", { for: "run-name", text: "Run name" }), nameIn),
    h("div.form-row", {}, h("span.lbl"), h("label.chk", {}, overwrite, h("span", { text: "Replace an existing run with this name" }))),
    app.S.expert ? cmd : null,
    h("div.form-row", {}, solveBtn, stopBtn, h("span.small.dim", { text: "F8 solve · Shift+F8 stop" })),
    msg));
  const live = h("div");
  const plotEl = h("div.chart.chart-live");
  const logPre = h("pre.log.mono", { tabindex: "0", "aria-label": "Solver log" });
  const follow = h("input", { type: "checkbox", checked: true });
  setChildren(root, h("div.page", {}, h("div.page-pad", { style: { height: "100%" } },
    h("div.solve-grid", {},
      h("div.stack", {}, form, panel("Run status", live)),
      h("div.stack", { style: { minHeight: 0 } },
        panel("Residual per Newton iteration", plotEl),
        h("section.panel.logpanel", { style: { flex: "1", minHeight: "220px" } },
          h("header.panel-head", {}, h("h2", { text: "Log" }), h("span.spacer"), h("label.chk", {}, follow, h("span", { text: "Follow" }))),
          logPre))))));
  doc._study = { live, plotEl, logPre, follow, root };
  updateStudy(app, doc);
}

/** Refresh the dynamic parts of the Study page (called on every poll). */
export async function updateStudy(app, doc) {
  const els = doc._study;
  if (!els || !els.root.isConnected) return;
  const st = app.statusOf(doc);
  if (!doc.name || !st) {
    setChildren(els.live, h("div.dim", { text: "Not solved yet. Press Solve to start the run; its progress, health and log appear here." }));
    setChildren(els.plotEl, placeholder("empty", "No iterations yet"));
    setChildren(els.logPre, "");
    return;
  }
  setChildren(els.live, h("div.stack", {},
    h("div.health-row", {}, healthBadge(st), h("span.mono", { text: doc.name }), h("span.msgline", { text: st.message || "" })),
    stageBars(st, { wide: true }),
    h("div.small.dim", { text: st.fraction === null && st.running ? "The solver does not report how far the current stage is, so its bar is indeterminate." : st.nStages ? `Stage ${Math.min(st.stage || 1, st.nStages)} of ${st.nStages}${st.fraction !== null ? `, ${(st.fraction * 100).toFixed(0)} % done` : ""}` : "" }),
    tellsGrid(st)));
  const series = convergenceSeries(st.running ? null : doc.summary, doc.progress);
  if (series.length) {
    const colors = seriesColors(Math.max(2, series.length));
    let x0 = 0;
    plot(els.plotEl, series.map((s, i) => {
      const xs = s.its.map((_, k) => x0 + k);
      x0 += s.its.length;
      return { x: xs, y: s.its.map((it) => it.res_norm), type: "scatter", mode: "lines+markers", name: `stage ${s.stage}`,
        line: { color: colors[i % colors.length], width: 2.5 }, marker: { size: 6 },
        customdata: s.its.map((it) => [it.it, it.alpha]), hovertemplate: `stage ${s.stage}, iteration %{customdata[0]}<br>‖R‖ = %{y:.3e}<br>α = %{customdata[1]}<extra></extra>` };
    }), baseLayout({ margin: { l: 58, r: 10, t: 8, b: 40 }, yaxis: { type: "log", title: { text: "residual ‖R‖" } },
      xaxis: { title: { text: "Newton iteration (all stages)" }, dtick: x0 < 16 ? 1 : undefined }, showlegend: series.length > 1, legend: { orientation: "h", y: 1.12, x: 1, xanchor: "right" } }),
    { displayModeBar: false });
  } else setChildren(els.plotEl, placeholder(st.running ? "loading" : "empty", st.running ? "Waiting for the first Newton iteration" : "No iteration history"));
  try {
    const r = await bridge.log(doc.name, 128 * 1024);
    if (!els.root.isConnected) return;
    if (r.size !== els.lastSize) {
      els.logPre.textContent = r.size ? r.text : "No stdout.log for this run. Runs started from the command line write their log to the terminal instead.";
      els.lastSize = r.size;
      if (els.follow.checked) els.logPre.scrollTop = els.logPre.scrollHeight;
    }
  } catch (e) {
    els.logPre.textContent = `No log: ${e.message}`;
  }
}

