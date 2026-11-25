// Workbench pages for the construction steps: Geometry, Mesh, Materials,
// Boundaries, Equations and Study (solve + live status). Results live in results.js.
//
// Each page edits its step's parameters in the project (case.json); the Boundaries page
// also sets the condition per boundary (case.json "bcs", bceditor.js). Results are read-only.

import { h, setChildren, fmt, fmtValue, isNum, prettyName, placeholder } from "./util.js";
import { icon } from "./icons.js";
import { bridge } from "./bridge.js";
import { plot, baseLayout, seriesColors, residualPlot } from "./plots.js";
import { boundaryColor } from "./fieldview.js";
import { boundaryDiagram, highlightBoundary, bcWords, bcShort, effectiveBCs, BC_QTY } from "./bdiagram.js";
import { boundaryEditor } from "./bceditor.js";
import { runStatus, healthBadge, stageBars, timeBar, tellsGrid, convergenceSeries } from "./progress.js";
import { caseTitle, caseDoc } from "./start.js";
import { paramForm, rowsOf, setParam, shownUnit } from "./params.js";
import { validationBar, renderCaseMesh, meshStrip } from "./meshtools.js";
import { convert, scalesFor } from "./units.js";

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
    h("h1", {}, icon(wb, 32), title,
      lead ? h("span.info-dot", { tabindex: "0", role: "img", "aria-label": `About ${title}: ${lead}`, "data-tip": String(lead) }, icon("info", 18)) : null),
    ...content)));
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

// Checks whose value / limit are flags or counts get words; `optional` checks do not
// decide whether the run can be trusted (the steady pre-solve after a transient run,
// a failed figure). Runs may also carry value_text / limit_text / optional themselves.
const CHECK_WORDS = [
  [/^steady equilibrium solve$/, () => ({ value: "not converged", limit: "should converge", optional: true,
    note: "Optional steady solve after the time run. The time-dependent result is unaffected." })],
  [/^time integration reached t_end$/, (c) => ({ value: `${fmt(100 * c.value, 3)} % of t_end`, limit: "must reach 100 %" })],
  [/^mesh diagnose$/, (c) => ({ value: `${fmt(c.value)} failed`, limit: "none may fail" })],
  [/^post-processing: /, () => ({ value: "failed", limit: "", optional: true })],
];

export function checkWords(c) {
  const hit = CHECK_WORDS.find(([re]) => re.test(c.name || ""));
  const w = hit ? hit[1](c) : {};
  const num = (v) => (v === undefined || v === null ? "" : fmtValue(v));
  return {
    value: c.value_text ?? w.value ?? (c.value === undefined || c.value === null ? "–" : num(c.value)),
    limit: c.limit_text ?? w.limit ?? num(c.limit),
    optional: c.optional ?? w.optional ?? false,
    note: w.note && c.status !== "pass" ? `${c.note ? `${c.note}. ` : ""}${w.note}` : c.note || "",
  };
}

export function checksList(checks, { withFix = false } = {}) {
  const order = { fail: 0, warn: 1, pass: 2, info: 3 };
  const rows = (checks || []).map((c) => ({ c, w: checkWords(c) }))
    .sort((a, b) => (a.w.optional - b.w.optional) || (order[a.c.status] ?? 9) - (order[b.c.status] ?? 9));
  return h("div.checks-list", { role: "table" },
    h("div.check.check-head", { role: "row" }, ["status", "check", "value", "limit", "note"].map((t, i) => h("span", { role: "columnheader", text: t, style: i === 2 || i === 3 ? { textAlign: "right" } : null }))),
    rows.map(({ c, w }) => h("div.check", { role: "row" },
      h("span", {}, checkStatusBadge(c.status)),
      h("span.c-name", {}, c.name || "", w.optional ? h("span.c-opt", { text: "optional" }) : null),
      h("span.c-val", { text: w.value }),
      h("span.c-lim", { text: w.limit }),
      h("span.c-note", {}, w.note, withFix && c.fix && c.status !== "pass" ? h("div", { text: `Fix: ${c.fix}` }) : null))));
}

/** Count of checks by status as plain text, e.g. "4 pass, 1 optional warning". */
export function checkCountText(checks) {
  const counts = {};
  for (const c of checks || []) {
    const k = checkWords(c).optional && c.status !== "pass" ? `opt-${c.status}` : c.status;
    counts[k] = (counts[k] || 0) + 1;
  }
  const word = { fail: "failed", warn: "warning", pass: "pass", info: "info", "opt-fail": "optional failed", "opt-warn": "optional warning", "opt-info": "optional info" };
  return ["fail", "warn", "pass", "info", "opt-fail", "opt-warn", "opt-info"].filter((k) => counts[k])
    .map((k) => `${counts[k]} ${word[k]}${/warn$/.test(k) && counts[k] > 1 ? "s" : ""}`).join(", ");
}

/**
 * One verdict for a run: solver convergence plus the checks, e.g. "Converged · 1 warning".
 * Optional checks never change the colour; when they are all that is left they are named.
 */
export function runVerdict(s) {
  const conv = s.solver ? s.solver.converged : undefined;
  const all = Array.isArray(s.checks) ? s.checks : [];
  const req = all.filter((c) => !checkWords(c).optional);
  const n = (st) => req.filter((c) => c.status === st).length;
  const fails = n("fail"), warns = n("warn");
  const optBad = all.length - req.length - all.filter((c) => checkWords(c).optional && c.status === "pass").length;
  if (conv === false) return { cls: "b-bad", ic: "x", text: "Not converged" };
  const head = conv === true ? "Converged" : "Finished";
  if (fails) return { cls: "b-bad", ic: "x", text: `${head} · ${fails} check${fails > 1 ? "s" : ""} failed` };
  if (warns) return { cls: "b-warn", ic: "alert", text: `${head} · ${warns} warning${warns > 1 ? "s" : ""}` };
  if (conv !== true) return null;
  if (optBad) return { cls: "b-ok", ic: "check", text: `${head} · ${optBad} optional warning${optBad > 1 ? "s" : ""}` };
  return { cls: "b-ok", ic: "check", text: req.length ? `${head} · all checks pass` : head };
}

export function verdictBadge(v) {
  return v ? h("span.badge", { class: v.cls }, icon(v.ic, 12), v.text) : null;
}

// ---------------------------------------------------------------------------
/** Panel with the editable parameters of one construction step. */
export function stepParams(app, doc, step, title, empty, { layout = "page" } = {}) {
  const rows = rowsOf(app, doc, step);
  const n = rows.filter((r) => r.changed).length;
  const readonly = doc.kind === "run";
  return h("section.panel#step-params", {}, h("header.panel-head", {}, h("h2", { text: title }), h("span.spacer"),
      h("span.badge.b-neutral.step-count", { dataset: { step }, hidden: !n || null }, `${n} ${readonly ? "differ" : "changed"} from template`)),
    h("div.panel-body.flush", {}, paramForm(app, doc, rows, { layout, empty, readonly, expert: app.S.expert })),
    readonly ? resultFoot(app, doc)
      : h("div.panel-foot", { text: `Saved in cases/${doc.caseName}/case.json. The next solve uses it.` }));
}

/** Footer of a result's (read-only) inputs: go to the case to change them. */
function resultFoot(app, doc) {
  const proj = (doc.listing && doc.listing.project) || null;
  return h("div.panel-foot.row", {},
    h("span", { text: "The inputs of this result are read-only. " }),
    h("button.btn-link", { type: "button", onclick: () => app.caseOfRun(doc) },
      proj ? `Open its project (${proj}) to change them` : "New project from these inputs"));
}

// ---------------------------------------------------------------------------
export function renderGeometry(app, doc, root) {
  const c = app.caseOf(doc);
  const bnames = app.boundaryNames(doc);
  const sym = app.symmetryOf(doc);
  const params = stepParams(app, doc, "geometry", "Geometry parameters", "This template has no geometry parameters. Its shape is fixed.");
  const tpl = panel("Template", kv([
    ["template", h("span", {}, caseTitle(doc.case), " ", h("code", { text: doc.case }))],
    ["coordinates", sym === "axisym" ? "axisymmetric (r, z), r ≥ 0" : sym === "planar" ? "planar 2D (y, z)" : "known after the first solve"],
    ["geometry kernel", "gmsh (OCC), built from these parameters"],
  ]));
  const bnd = panel("Named boundaries", bnames.length ? h("table.tbl", {}, h("tbody", {}, bnames.map((b, i) =>
    h("tr", {}, h("th", {}, h("span.swatch", { class: /axis|symmetry/.test(b) ? "dashed" : "", style: /axis|symmetry/.test(b) ? { borderColor: boundaryColor(b, i), width: "14px" } : { background: boundaryColor(b, i) } })),
      h("td.mono", { text: b }), h("td.dim.small", { text: /axis|symmetry/.test(b) ? "symmetry axis" : bcShort(effectiveBCs(doc).bcs[b], bcContext(app, doc)) })))))
    : h("div.dim", { text: "Boundary names are known after the mesh is built (solve once)." }));
  const views = (doc.listing && doc.listing.mesh_files || []).filter((f) => /view_a/.test(f));
  setChildren(root, pageShell("geometry", "Geometry",
    caseDoc(c) || "The geometry comes from the template.",
    doc.kind === "case" && doc.caseDetail ? meshStrip(app, doc) : null,
    h("div.grid2", {}, h("div.stack", {}, params, tpl), h("div.stack", {},
      views.length ? imageFigure(app, doc, views[0], "Domain of the last solve, boundaries coloured by group") : null, bnd)),
    stub("Importing CAD or meshes and editing gmsh scripts need the project file, which is not built yet. Today the template builds the shape from the parameters above.")));
}

export function renderMesh(app, doc, root, { resultOnly = false } = {}) {
  if (doc.kind === "case" && doc.caseData && !resultOnly) {
    renderCaseMesh(app, doc, root, { pageShell, panel, imageFigure, stepParams });
    return;
  }
  const s = doc.summary || {};
  const m = s.mesh;
  const diag = (doc.diagnose && doc.diagnose.checks) || s.mesh_checks || null;
  const viewsMeta = Array.isArray(s.mesh_views) ? s.mesh_views : [];
  const files = (doc.listing && doc.listing.mesh_files) || [];
  const content = [];

  // Mesh Validation first: one bar with the counts, click for every check
  if (diag && diag.length) content.push(validationBar(doc, diag, { source: "mesh of this result" }));
  else if (m || files.length) content.push(validationBar(doc, null, { source: "This run has no mesh/diagnose.json (older runs were not validated)." }));

  const sizes = stepParams(app, doc, "mesh", "Mesh sizes", "This template sets the mesh sizes from the Debye length; it has no size parameters.");

  if (!m && !files.length) {
    content.push(h("div.grid2", {}, sizes, h("div.panel", {}, placeholder("empty", doc.kind === "draft" ? "The mesh is built when you solve" : "No mesh information in this run",
      doc.kind === "draft" ? "gmsh meshes the geometry with curved P2 triangles, refined near charged walls (h ≈ λ_D / 8 within 3 λ_D). The checks, two mesh views and statistics appear here after the run." : "summary.json has no mesh section."))));
  } else {
    const histEl = m && m.quality_hist && Array.isArray(m.quality_hist.counts) ? h("div.chart.chart-sm") : null;
    const stats = m ? panel("Mesh statistics", [kv([
      ["elements", m.n_elements], ["nodes (P2)", m.n_nodes], ["vertices", m.n_vertices], ["unknowns (DOFs)", m.n_dofs],
      ["smallest element h", m.h_min], ["largest element h", m.h_max], ["min quality", m.quality_min], ["mean quality", m.quality_mean],
    ].filter(([, v]) => v !== undefined)), histEl ? h("div.bar-label", { text: "Element quality (1 = equilateral)", style: { marginTop: "10px" } }) : null, histEl]) : null;
    const views = files.filter((f) => /\.png$/.test(f)).map((f) => {
      const meta = viewsMeta.find((v) => v.file === f) || {};
      return imageFigure(app, doc, f, meta.title || (/view_a/.test(f) ? "View A: whole domain by boundary group" : "View B: finest region by quality"), meta.what_to_check);
    });
    content.push(views.length ? h("div.mesh-views", {}, views) : null,
      h("div.grid2", { style: { marginTop: "12px" } }, sizes, stats));
    if (histEl) requestAnimationFrame(() => {
      const { edges, counts } = m.quality_hist;
      const centers = counts.map((_, i) => (isNum(edges[i]) && isNum(edges[i + 1]) ? (edges[i] + edges[i + 1]) / 2 : i));
      plot(histEl, [{ type: "bar", x: centers, y: counts, marker: { color: seriesColors(3)[1] }, hovertemplate: "quality %{x:.2f}: %{y} elements<extra></extra>" }],
        baseLayout({ margin: { l: 48, r: 8, t: 4, b: 30 }, xaxis: { title: { text: "quality" }, range: [0, 1] }, yaxis: { type: "log", title: { text: "elements" } }, showlegend: false, bargap: 0.05 }),
        { displayModeBar: false });
    });
  }
  setChildren(root, pageShell("mesh", "Mesh", "Every mesh is rendered from two views and validated before a solve.", ...content));
}

// ---------------------------------------------------------------------------
export function renderMaterials(app, doc, root) {
  const s = doc.summary || {};
  const species = (s.model && Array.isArray(s.model.species)) ? s.model.species : null;
  let spBody;
  if (species === null) spBody = h("div.dim", { text: doc.kind === "draft" ? "The species list is written when the template is solved." : "This run does not list its species." });
  else if (!species.length) spBody = h("div", {}, h("p", { style: { margin: "0 0 6px" } }, "No mobile ions. This model solves the Poisson equation only, for a fixed charge in a uniform dielectric."));
  else spBody = h("table.tbl", {}, h("thead", {}, h("tr", {}, ["species", "valence z", "diffusivity D / D0", "bulk c / c0"].map((t, i) => h("th", { class: i ? "num" : "", text: t })))),
    h("tbody", {}, species.map((sp) => h("tr", {}, h("th", { text: prettyName(sp.name) }), h("td.num", { text: `${sp.z > 0 ? "+" : ""}${fmt(sp.z)}` }), h("td.num", { text: fmt(sp.d) }), h("td.num", { text: fmt(sp.bulk) })))));
  setChildren(root, pageShell("materials", "Materials", "Solvent, temperature and the ions in the electrolyte.",
    h("div.grid2", {}, stepParams(app, doc, "materials", "Material parameters", "No material parameters in this template."), panel("Species", spBody)),
    stub("An ion library (valence and diffusivity at 25 °C), custom ions and lab units need the project file, which is not built yet.")));
}

// ---------------------------------------------------------------------------
/** What bcWords / bcShort need to show a condition's values as entered or scaled. */
function bcContext(app, doc) {
  const rows = app.paramRows(doc);
  return { params: (doc.summary || {}).params, si: app.S.units === "SI", inputs: Object.fromEntries(rows.map((r) => [r.key, r.value])), scales: scalesFor(rows) };
}

export function renderBoundaries(app, doc, root) {
  const s = doc.summary || {};
  const { bcs, base, changed } = effectiveBCs(doc);
  const names = app.boundaryNames(doc);
  const diag = (doc.diagnose && doc.diagnose.checks) || s.mesh_checks || [];
  const coverage = diag.filter((c) => /BC|boundar/i.test(c.name));
  const flow = s.model && s.model.flow && s.model.flow !== "none";
  const expert = app.S.expert;
  const ctx = bcContext(app, doc);
  const editable = doc.kind === "case" && !!doc.caseData;
  const all = [...new Set([...names, ...Object.keys(bcs)])];
  if (doc.bSel && !all.includes(doc.bSel)) doc.bSel = null;
  root.dataset.bsel = doc.bSel || "";

  // re-render in place: keep the scroll position and the focused control
  const refresh = () => {
    const page = root.querySelector(".page");
    const top = page ? page.scrollTop : 0;
    const focus = document.activeElement && root.contains(document.activeElement) ? document.activeElement.id : null;
    renderBoundaries(app, doc, root);
    const p2 = root.querySelector(".page");
    if (p2) p2.scrollTop = top;
    if (focus) { const el = document.getElementById(focus); if (el) el.focus(); }
  };
  const select = (b) => { doc.bSel = doc.bSel === b ? null : b; refresh(); app.syncURL(); };
  const hover = (b) => highlightBoundary(root, b);

  let body;
  if (!all.length) {
    body = placeholder("empty", "Boundaries are named when the mesh is built", editable ? "Create the mesh to see each named boundary and set its condition." : "Solve once to see each named boundary and the condition on it.");
  } else {
    const keys = ["c", "phi", ...(flow ? ["u"] : [])];
    const cell = (b, k) => {
      const spec = bcs[b] ? bcs[b][k] : undefined;
      if (/axis|symmetry/.test(b) && spec === undefined) return h("td.dim", { text: "symmetry" });
      const raw = typeof spec === "object" && spec !== null && typeof spec.type === "string" ? null : spec;
      return h("td", { class: spec === undefined ? "dim" : "" }, bcWords(k, spec, ctx),
        expert && raw !== undefined && raw !== null ? h("div.mono.small.dim", { text: typeof raw === "object" ? JSON.stringify(raw) : String(raw) }) : null);
    };
    body = h("table.tbl.bc-table", {}, h("thead", {}, h("tr", {}, h("th", { text: "boundary" }), keys.map((k) => h("th", { text: BC_QTY[k] })))),
      h("tbody", {}, all.map((b, i) => h("tr", { "data-b": b, tabindex: "0", "aria-label": `${b}: ${editable ? "change its condition" : "show its condition"}`,
        onpointerenter: () => hover(b), onpointerleave: () => hover(null), onclick: () => select(b),
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); select(b); } } },
        h("th", {}, h("span", { style: { display: "inline-flex", alignItems: "center", gap: "8px" } },
          h("span.swatch", { class: /axis|symmetry/.test(b) ? "dashed" : "", style: /axis|symmetry/.test(b) ? { borderColor: boundaryColor(b, i), width: "14px" } : { background: boundaryColor(b, i) } }),
          h("span.mono", { text: b }),
          changed.has(b) ? h("span.badge.b-neutral", { text: editable ? "changed" : "differs" }) : null)),
        keys.map((k) => cell(b, k))))));
  }
  const nChanged = changed.size;
  const changedBadge = nChanged ? h("span.badge.b-neutral", { text: `${nChanged} ${editable ? "changed" : "differ"} from template` }) : null;
  const sel = doc.bSel;
  const editor = sel ? boundaryEditor(app, doc, sel, ctx, { base, index: Math.max(0, all.indexOf(sel)), onChange: refresh, onClose: () => select(sel) }) : null;
  const lead = editable ? "Where each named boundary is and the condition on it. Click a boundary in the drawing or the table to change its condition."
    : "Where each named boundary is and the condition on it. Grey entries are the defaults: no flux, no charge, no-slip.";
  setChildren(root, pageShell("boundary", "Boundaries", lead,
    h("div.stack", { onkeydown: (e) => { if (e.key === "Escape" && doc.bSel) select(doc.bSel); } },
      names.length ? panel("Where each boundary is", boundaryDiagram(app, doc, names, bcs, ctx, { onHover: hover, onSelect: select, editable })) : null,
      editor,
      panel("Conditions per boundary", body, changedBadge),
      stepParams(app, doc, "boundaries", "Boundary parameters", "No boundary parameters in this template."),
      coverage.length ? panel("Coverage check", checksList(coverage, { withFix: true })) : null)));
  highlightBoundary(root, null);
}

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
    h("div.stack", {}, panel("Physics", list), stepParams(app, doc, "equations", "Physics parameters", "No dimensionless groups to set in this template.")),
    stub("Switching physics on and off (ions, flow type, advection form) needs the project file. Today the template fixes it.")));
}

// ---------------------------------------------------------------------------
// Study: solve, live progress, health, residual plot and log tail
// ---------------------------------------------------------------------------
export function renderStudy(app, doc, root) {
  const rows = app.paramRows(doc);
  const nameIn = h("input.inp#run-name", { type: "text", spellcheck: "false", value: app.nextNameOf(doc) || app.suggestName(doc), "aria-label": "Run name",
    oninput: (e) => { app.setNextName(doc, e.target.value); updateCmd(); } });
  const overwrite = h("input", { type: "checkbox", checked: !!doc.overwrite, onchange: (e) => { doc.overwrite = e.target.checked; } });
  const cmd = h("code.cmd");
  const msg = h("div", { "aria-live": "polite" });
  const devNote = h("span.small.dim", { text: "checking devices…" });
  const devDot = h("span.dev-dot", { role: "img", "aria-label": "checking devices" });
  let gpuFound = null;
  const setDot = () => {
    // a finished or running run reports the platform it used; otherwise show what the selection will pick
    const used = doc.progress && doc.progress.device && doc.progress.device.platform;
    const gpu = used ? used !== "cpu" : gpuFound === null ? null : devSel.value !== "cpu" && gpuFound;
    const word = gpu === null ? "checking devices" : gpu ? (used ? "This run used the GPU" : "The GPU will be used") : (used ? "This run used the CPU" : "The CPU will be used");
    devDot.className = `dev-dot ${gpu === null ? "" : gpu ? "on" : "off"}`;
    devDot.setAttribute("aria-label", word); devDot.dataset.tip = word; devDot.title = word;
  };
  const devSel = h("select.sel#run-device", { "aria-label": "Compute device", title: "Where JAX assembles the residual and Jacobian (the sparse solve runs on the CPU)",
    onchange: (e) => { app.setDevice(e.target.value); updateCmd(); setDot(); } },
    h("option", { value: "cpu", text: "CPU" }),
    h("option", { value: "auto", text: "Auto (GPU when present)" }),
    h("option", { value: "gpu", text: "GPU" }));
  devSel.value = app.device();
  app.devices().then((dv) => {
    const gpuOpt = devSel.querySelector('option[value="gpu"]');
    gpuOpt.disabled = !dv.gpu;
    if (!dv.gpu && devSel.value === "gpu") { devSel.value = "cpu"; app.setDevice("cpu"); updateCmd(); }
    gpuFound = !!dv.gpu; setDot();
    devNote.textContent = dv.gpu ? `GPU found: ${(dv.devices || []).join(", ")}` : (dv.error ? `No GPU (${dv.error.slice(0, 120)})` : "No GPU found. Runs use the CPU");
  });
  const changedLine = h("div.small");
  const updateCmd = () => {
    cmd.textContent = app.commandLine(doc, nameIn.value.trim());
    const ch = app.paramRows(doc).filter((r) => r.changed);
    setChildren(changedLine, ch.length ? [`${ch.length} parameter${ch.length > 1 ? "s" : ""} changed from the template: `, h("span.mono", { text: ch.map((r) => `${r.key} = ${fmt(r.value)}`).join(", ") })]
      : h("span.dim", { text: rows.length ? "All parameters at their template values." : "Template parameters are not loaded yet." }));
  };
  doc._onParams = updateCmd;
  updateCmd();
  const st = app.statusOf(doc);
  const solveBtn = h("button.btn.primary", { type: "button", onclick: () => app.solve(doc, { name: nameIn.value.trim(), overwrite: overwrite.checked, msgEl: msg }) },
    icon("play", 12), doc.kind === "run" ? "Solve in its case" : "Solve");
  const stopBtn = h("button.btn.danger", { type: "button", onclick: () => app.stop(doc), disabled: !(st && st.running && app.canStop(doc)) || null, title: "Stop this run (Shift+F8)" }, icon("stop", 12), "Stop");
  if (st && st.running) { solveBtn.disabled = true; solveBtn.title = "A run of this document is in progress"; }
  const form = panel("Solve", h("div.solve-form", {},
    changedLine,
    h("div.form-row", {}, h("label.lbl", { for: "run-name", text: "Run name" }), nameIn),
    h("div.form-row", {}, h("span.lbl"), h("label.chk", {}, overwrite, h("span", { text: "Replace an existing run with this name" }))),
    h("div.form-row", {}, h("label.lbl", { for: "run-device", text: "Device" }), devSel),
    h("div.form-row", {}, h("span.lbl"), devDot, devNote),
    app.S.expert ? cmd : null,
    h("div.form-row", {}, solveBtn, stopBtn, h("span.small.dim", { text: "F8 solve · Shift+F8 stop" })),
    msg));
  const live = h("div");
  const plotEl = h("div.chart.chart-live");
  const logPre = h("pre.log.mono", { tabindex: "0", "aria-label": "Solver log" });
  const follow = h("input", { type: "checkbox", checked: true });
  setChildren(root, h("div.page", {}, h("div.page-pad", { style: { height: "100%" } },
    h("div.solve-grid", {},
      h("div.stack", {}, form, studySettings(app, doc), panel("Run status", live)),
      h("div.stack", { style: { minHeight: 0 } },
        panel("Residual per Newton iteration", plotEl),
        h("section.panel.logpanel", { style: { flex: "1", minHeight: "220px" } },
          h("header.panel-head", {}, h("h2", { text: "Log" }), h("span.spacer"), h("label.chk", {}, follow, h("span", { text: "Follow" }))),
          logPre))))));
  doc._study = { live, plotEl, logPre, follow, root };
  updateStudy(app, doc);
}

// ---------------------------------------------------------------------------
// Time stepping (time-dependent templates): compact, everything on a few lines
// ---------------------------------------------------------------------------
const TIME_KEYS = ["tau_end", "dt0", "rtol"];

function studySettings(app, doc) {
  const c = app.caseOf(doc);
  const rows = rowsOf(app, doc, "study");
  if (!c.transient) {
    return h("section.panel#step-params", {}, h("header.panel-head", {}, h("h2", { text: "Solver settings" })),
      h("div.panel-body", {}, paramForm(app, doc, rows, { layout: "grid", expert: app.S.expert, readonly: doc.kind === "run", empty: "No solver settings in this template." })),
      doc.kind === "run" ? resultFoot(app, doc) : null);
  }
  const readonly = doc.kind !== "case";
  // time settings: the case's (editable) or the result's frozen case.json (read-only).
  // A save can replace doc.caseData with a fresh object from the server (saveCaseNow),
  // so every write goes through curTime()/curSave() to fetch it at that moment rather
  // than closing over doc.caseData.time once: writing into a stale copy would otherwise
  // be silently lost (the next save posts the real, unchanged doc.caseData).
  const tm = doc.kind === "case" ? doc.caseData.time : app.normalizeCase(doc.runCase || {}).time;
  const curTime = () => (doc.kind === "case" && doc.caseData ? doc.caseData.time : tm);
  const curSave = () => curTime().save;
  const save = () => { if (!readonly) app.caseChanged(doc); };
  const byKey = Object.fromEntries(rows.map((r) => [r.key, r]));

  const paramCell = (label, key, tip) => {
    const r = byKey[key];
    if (!r) return null;
    const unit = shownUnit(doc, r);
    const inp = h("input.inp", { type: "text", inputmode: "decimal", spellcheck: "false", readonly: readonly || null, id: `ts-${key}`,
      value: String(Number.parseFloat(convert(r.value, r.unit, unit).toPrecision(10))), class: r.changed ? "changed" : "",
      onchange: (e) => {
        const ok = setParam(app, doc, key, e.target.value, true, unit);
        e.target.classList.toggle("invalid", !ok);
        e.target.setAttribute("aria-invalid", ok ? "false" : "true");
        if (ok) app.paramsChanged(doc);
      } });
    return h("label.ts-cell", { "data-tip-title": label, "data-tip": tip || r.info }, h("span.ts-lbl", {}, label, h("code", { text: key })), inp);
  };
  const timeCell = (label, key, tip, { min = 0, integer = false, blank = "" } = {}) => {
    const inp = h("input.inp", { type: "text", inputmode: "decimal", spellcheck: "false", readonly: readonly || null, placeholder: blank,
      value: tm[key] === null || tm[key] === undefined ? "" : String(tm[key]),
      onchange: (e) => {
        const t = e.target.value.trim();
        const v = Number(t);
        const ok = (t === "" && blank) || (Number.isFinite(v) && v > min && (!integer || Number.isInteger(v)));
        e.target.classList.toggle("invalid", !ok);
        if (ok) { curTime()[key] = t === "" ? null : v; save(); }
      } });
    return h("label.ts-cell", { "data-tip-title": label, "data-tip": tip }, h("span.ts-lbl", { text: label }), inp);
  };

  // saved times: mode picker
  const sv = curSave();
  const modes = [["template", "Template", "The template's own list of saved times. The pore template saves a log-spaced set, or the DNS folder times when dns_times = 1"],
    ["uniform", "Every Δτ", "Evenly spaced from 0 to τ_end, as N points (default 100) or one save every Δτ"], ["log", "Log-spaced", "N points evenly spaced in log τ between τ_min and τ_end"], ["list", "List", "Exactly these τ values"]];
  const modeSeg = h("span.seg.ts-seg", { role: "radiogroup", "aria-label": "Saved times" }, modes.map(([m, t, tip]) => h("button", {
    type: "button", role: "radio", "aria-checked": String(sv.mode === m), "aria-pressed": String(sv.mode === m), disabled: readonly || null, "data-tip": tip,
    onclick: () => { curSave().mode = m; save(); drawSave(); modeSeg.querySelectorAll("button").forEach((b, i) => { const on = modes[i][0] === m; b.setAttribute("aria-pressed", String(on)); b.setAttribute("aria-checked", String(on)); }); } }, t)));
  const saveBody = h("div.ts-row");
  const preview = h("span.small.dim.ts-preview");
  const tauEnd = () => (byKey.tau_end ? Number(byKey.tau_end.value) : null);
  const saveTimes = () => {
    const te = tauEnd();
    const cs = curSave();
    if (!te) return null;
    if (cs.mode === "uniform" && cs.by === "count") return cs.n_lin > 0 ? cs.n_lin : null;
    if (cs.mode === "uniform") return cs.dtau > 0 ? Math.floor(te / cs.dtau + 1e-9) + (Math.abs(Math.round(te / cs.dtau) * cs.dtau - te) > 1e-9 * te ? 1 : 0) : null;
    if (cs.mode === "log") return cs.n || null;
    if (cs.mode === "list") return new Set([...(cs.list || []).filter((t) => t > 0 && t <= te), te]).size;
    return null;
  };
  const drawPreview = () => {
    const n = saveTimes();
    preview.textContent = curSave().mode === "template" ? "the template decides (τ = 0.13 and τ_mid are always saved)" : n ? `${n} saved time${n === 1 ? "" : "s"} plus τ = 0.13 and τ_mid when inside the run. τ_end is always saved` : "";
  };
  const svCell = (label, key, tip, opts = {}) => {
    const inp = h("input.inp", { type: "text", inputmode: "decimal", readonly: readonly || null, value: String(curSave()[key] ?? ""),
      onchange: (e) => {
        const v = Number(e.target.value);
        const ok = Number.isFinite(v) && v > 0 && (!opts.integer || Number.isInteger(v));
        e.target.classList.toggle("invalid", !ok);
        if (ok) { curSave()[key] = v; save(); drawPreview(); }
      } });
    return h("label.ts-cell", { "data-tip": tip }, h("span.ts-lbl", { text: label }), inp);
  };
  function drawSave() {
    const cs = curSave();
    let cells = [];
    if (cs.mode === "uniform") {
      const bys = [["count", "Linear spacing", "N points evenly spaced in τ: τ_end/N, 2 τ_end/N, …, τ_end. Play in Results steps through these times"],
        ["dtau", "Fixed Δτ", "Save once every Δτ, a step you set"]];
      const bySeg = h("span.seg.ts-seg", { role: "radiogroup", "aria-label": "Spacing" }, bys.map(([b, t, tip]) => h("button", {
        type: "button", role: "radio", "aria-checked": String(cs.by === b), "aria-pressed": String(cs.by === b), disabled: readonly || null, "data-tip": tip,
        onclick: () => { curSave().by = b; save(); drawSave(); } }, t)));
      cells = [h("div.ts-cell.auto", {}, h("span.ts-lbl", { text: "Spacing" }), bySeg), cs.by === "count"
        ? svCell("N points", "n_lin", "Number of linearly spaced saved times (default 100). Each one is a frame when you play the field", { integer: true })
        : svCell("Δτ", "dtau", "Time between two saved times")];
    }
    else if (cs.mode === "log") cells = [svCell("N points", "n", "Number of saved times", { integer: true }), svCell("from τ_min", "tau_min", "First saved time")];
    else if (cs.mode === "list") {
      const inp = h("input.inp.ts-list", { type: "text", readonly: readonly || null, value: (cs.list || []).join(", "), placeholder: "0.01, 0.13, 0.5, 1",
        onchange: (e) => {
          const parts = e.target.value.split(/[,\s;]+/).filter(Boolean).map(Number);
          const ok = parts.every((v) => Number.isFinite(v) && v > 0);
          e.target.classList.toggle("invalid", !ok);
          if (ok) { curSave().list = parts.sort((a, b) => a - b); save(); drawPreview(); }
        } });
      cells = [h("label.ts-cell.wide", { "data-tip": "Comma-separated τ values" }, h("span.ts-lbl", { text: "τ values" }), inp)];
    }
    setChildren(saveBody, cells);
    drawPreview();
  }
  drawSave();

  const others = rows.filter((r) => !TIME_KEYS.includes(r.key));
  return h("section.panel#step-params.ts-panel", {},
    h("header.panel-head", {}, h("h2", { text: "Time stepping" }), h("span.spacer"),
      h("span.small.dim", { text: "τ = D t / l_p²  ·  BDF2, variable step" })),
    h("div.panel-body", {},
      h("div.ts-row", {},
        paramCell("End time", "tau_end", "Final time τ_end"),
        paramCell("First step", "dt0", "First time step (in a_ref² / D)"),
        timeCell("Max step Δτ", "dt_max", "Largest step in τ units (blank uses the template default, 0.02)", { blank: "auto" }),
        paramCell("Step tolerance", "rtol", "Largest change per step against the linear extrapolation. A smaller value gives more, smaller steps"),
        timeCell("Max rejects", "max_rejects", "Rejected steps in a row before the run gives up", { integer: true })),
      h("div.ts-sub", {}, h("span.ts-head", { text: "Saved times" }), modeSeg, preview),
      saveBody,
      others.length ? [h("div.ts-sub", {}, h("span.ts-head", { text: "Other study settings" })),
        paramForm(app, doc, others, { layout: "grid", expert: app.S.expert, readonly })] : null),
    readonly ? resultFoot(app, doc) : h("div.panel-foot", { text: `Saved in cases/${doc.caseName}/case.json. The next solve uses it.` }));
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
    st.time ? timeBar(st) : null,
    st.time ? null : stageBars(st, { wide: true }),
    st.time ? null : h("div.small.dim", { text: st.fraction === null && st.running ? "The solver does not report how far the current stage is, so its bar is indeterminate." : st.nStages ? `Stage ${Math.min(st.stage || 1, st.nStages)} of ${st.nStages}${st.fraction !== null ? `, ${(st.fraction * 100).toFixed(0)} % done` : ""}` : "" }),
    tellsGrid(st)));
  const series = convergenceSeries(st.running ? null : doc.summary, doc.progress);
  if (series.length) {
    residualPlot(els.plotEl, series, { label: st.time ? "time step" : "stage" });
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

