// Mesh step: Mesh Validation bar (shared with results) and, for a case, the mesh
// tools: build now, global density, boundary refinement, refinement regions (box,
// disk), sharp-corner refinement and a spray-paint brush. Every tool edits the case's
// "mesh" block in case.json (pnpflow/geometry.py apply_mesh_options turns it into gmsh
// size fields); the canvas draws cases/<case>/mesh/mesh.json with the refinements on top.

import { h, setChildren, fmt, fmtValue, isNum, placeholder } from "./util.js";
import { icon } from "./icons.js";
import { bridge, optional } from "./bridge.js";
import { boundaryColor } from "./fieldview.js";
import { repairWorkspace } from "./repair.js";

const ORDER = { fail: 0, warn: 1, pass: 2, info: 3 };
const WORD = { pass: "pass", warn: "warning", fail: "fail", info: "info" };
const IC = { pass: "check", warn: "alert", fail: "x", info: "info" };

function statusBadge(status, text) {
  const cls = { pass: "b-ok", warn: "b-warn", fail: "b-bad", info: "b-info" }[status] || "b-info";
  return h("span.badge", { class: cls }, icon(IC[status] || "info", 12), text || WORD[status] || status);
}

/**
 * Mesh Validation: one full-width bar with the counts; click opens every check with
 * its value, limit, note and fix. Opens by itself when a check warns or fails.
 */
export function validationBar(doc, checks, { source = "" } = {}) {
  const wrap = h("section.vbar-wrap");
  if (!checks) {
    setChildren(wrap, h("div.vbar.empty", {}, icon("diagnose", 16), h("b", { text: "Mesh Validation" }),
      h("span.dim", { text: source || "No validation yet: create the mesh to check it." })));
    return wrap;
  }
  const counts = {};
  for (const c of checks) counts[c.status] = (counts[c.status] || 0) + 1;
  const worst = ["fail", "warn"].find((k) => counts[k]);
  if (doc._vbarOpen === undefined) doc._vbarOpen = !!worst;
  const sorted = checks.map((c, i) => ({ c, i })).sort((a, b) => (ORDER[a.c.status] ?? 9) - (ORDER[b.c.status] ?? 9) || a.i - b.i).map((x) => x.c);
  const list = h("div.vbar-list", { id: "vbar-list" },
    sorted.map((c) => h("div.vcheck", { class: `st-${c.status}` },
      h("span.vc-status", {}, statusBadge(c.status)),
      h("span.vc-name", { text: c.name }),
      h("span.vc-num", {}, h("span.dim", { text: "value " }), h("span.mono", { text: isNum(c.value) || c.value ? fmtValue(c.value) : "–" }),
        c.limit !== null && c.limit !== undefined ? [h("span.dim", { text: "  limit " }), h("span.mono", { text: fmtValue(c.limit) })] : null),
      h("span.vc-note", {}, c.note || "", c.fix && c.status !== "pass" ? h("div.vc-fix", {}, h("b", { text: "Fix: " }), c.fix) : null))),
    h("p.vbar-note", {}, icon("info", 14), h("span", { text: "A warning or failure does not stop the run, but it can slow or break the convergence and bias the results (fluxes, charge, EDL profiles). Fix failures before trusting a run." })));
  const bar = h("button.vbar#mesh-validation-bar", {
    type: "button", class: worst ? `st-${worst}` : "st-pass", "aria-expanded": String(!!doc._vbarOpen), "aria-controls": "vbar-list",
    onclick: () => { doc._vbarOpen = !doc._vbarOpen; draw(); },
  }, icon("chevron-right", 14), h("b", { text: "Mesh Validation" }),
    h("span.vbar-counts", {}, ["fail", "warn", "pass", "info"].filter((k) => counts[k]).map((k) => statusBadge(k, `${counts[k]} ${k === "warn" ? (counts[k] > 1 ? "warnings" : "warning") : WORD[k]}`))),
    h("span.spacer"), source ? h("span.dim.small", { text: source }) : null);
  const draw = () => {
    bar.setAttribute("aria-expanded", String(!!doc._vbarOpen));
    list.hidden = !doc._vbarOpen;
  };
  draw();
  setChildren(wrap, bar, list);
  return wrap;
}

/** The Create mesh button (gmsh on the CPU: mesh, validate, two views; no solve). */
export function createMeshButton(app, doc, busy) {
  return h("button.btn.primary", { type: "button", disabled: busy || null, "data-tip-title": "Create mesh",
    "data-tip": "Runs on the CPU: gmsh meshes the case, then it is validated and drawn from two views. No solve.",
    onclick: () => app.buildMesh(doc) }, busy ? h("span.spinner.sm") : icon("mesh", 14), busy ? "Creating…" : "Create mesh");
}

/** One-line mesh state of a case with the Create mesh button (Geometry page). */
export function meshStrip(app, doc) {
  const st = (doc.caseDetail && doc.caseDetail.mesh_status) || { state: "none" };
  const n = (st.stats || {}).n_elements;
  const word = st.state === "building" ? "creating the mesh on the CPU…" : st.state === "failed" ? `mesh failed: ${st.error || "see mesh.log"}`
    : st.state === "done" ? `${fmt(n)} elements, ${st.up_to_date ? "up to date" : "out of date (geometry changed)"}` : "no mesh yet";
  const badge = st.state === "done" ? statusBadge(st.up_to_date ? "pass" : "warn", st.up_to_date ? "mesh up to date" : "mesh out of date")
    : st.state === "failed" ? statusBadge("fail", "mesh failed") : st.state === "building" ? h("span.badge.b-run", {}, h("span.spinner.sm"), "creating") : statusBadge("info", "no mesh");
  return h("div.panel.mesh-state-panel", {}, h("div.mesh-state", {}, badge, h("span.mesh-state-txt", { text: word }), h("span.spacer"),
    h("button.btn", { type: "button", onclick: () => app.setWorkbench("mesh") }, icon("mesh", 14), "Mesh tools"),
    createMeshButton(app, doc, st.state === "building")));
}

// ---------------------------------------------------------------------------
// Case mesh page
// ---------------------------------------------------------------------------
function numInput(value, onchange, { label, min = null, width = null, placeholderText = "" } = {}) {
  const el = h("input.inp.num-inp", { type: "text", inputmode: "decimal", spellcheck: "false", "aria-label": label, placeholder: placeholderText,
    value: value === null || value === undefined || value === "" ? "" : String(value), style: width ? { width } : null });
  el.addEventListener("change", () => {
    const t = el.value.trim();
    const v = t === "" ? null : Number(t);
    const ok = t === "" ? placeholderText !== "" : Number.isFinite(v) && (min === null || v > min);
    el.classList.toggle("invalid", !ok);
    el.setAttribute("aria-invalid", ok ? "false" : "true");
    if (ok) onchange(v);
  });
  el.addEventListener("keydown", (e) => { if (e.key === "Enter") el.blur(); });
  return el;
}

function field(label, control, tip) {
  return h("label.mt-field", { "data-tip": tip || null }, h("span.mt-lbl", { text: label }), control);
}

export function renderCaseMesh(app, doc, root, { pageShell, panel, imageFigure, stepParams }) {
  const det = doc.caseDetail || {};
  const ms = det.mesh_status || { state: "none" };
  const mesh = doc.caseData.mesh;
  const save = () => { app.caseChanged(doc); };

  // ---- validation (case mesh/validation.json) ----
  const vSlot = h("div");
  const drawValidation = () => setChildren(vSlot, validationBar(doc, doc.caseValidation ? doc.caseValidation.checks : null,
    { source: ms.state === "done" ? (ms.up_to_date ? "case mesh, up to date" : "case mesh, out of date") : "" }));
  if (doc.caseValidation === undefined && (det.mesh_files || []).includes("mesh/validation.json")) {
    doc.caseValidation = null;
    optional(bridge.caseJSON(doc.caseName, "mesh/validation.json")).then((v) => { doc.caseValidation = v; drawValidation(); }).catch(() => {});
  }
  drawValidation();

  // ---- Repair mesh: a tree of the checks below the state row (FreeCAD model-tree look) ----
  const rSlot = h("div");
  let grid = null;
  const repairButton = (busy) => h("button.btn", { type: "button", disabled: busy || null, "aria-expanded": String(!!doc._repairOpen),
    "data-tip-title": "Repair mesh", "data-tip": "Lists the failed and warned mesh checks as a tree, with the fix for each. Where a fix can be applied, one click edits the mesh settings; then create the mesh again.",
    onclick: () => { doc._repairOpen = !doc._repairOpen; drawRepair(); drawState(); } }, icon("wrench", 14), "Repair mesh");
  const deleteButton = (busy) => {
    const b = h("button.btn", { type: "button", disabled: busy || null, "data-tip-title": "Delete mesh",
      "data-tip": "Removes the generated mesh files of this case (mesh, views, validation). case.json and earlier runs stay. Click twice to confirm.",
      onclick: () => {
        if (b.dataset.armed) { clearTimeout(b._t); app.deleteMesh(doc); return; }
        b.dataset.armed = "1"; b.classList.add("danger"); setChildren(b, icon("trash", 14), "Click again to delete");
        b._t = setTimeout(() => { delete b.dataset.armed; b.classList.remove("danger"); setChildren(b, icon("trash", 14), "Delete mesh"); }, 4000);
      } }, icon("trash", 14), "Delete mesh");
    return b;
  };
  const wsHost = h("div");
  let ws = null;
  const autoFix = (c) => {
    // EDL resolution: refine the named wall to delta/8 inside 3 delta
    if (c.name !== "EDL resolution") return null;
    const b = /'([^']+)'/.exec(c.fix || ""), d = /delta = ([0-9.eE+-]+)/.exec(c.note || "");
    const delta = d ? Number(d[1]) : NaN;
    if (!b || !Number.isFinite(delta) || delta <= 0) return null;
    return { label: `Refine ${b[1]} to delta/8`, run: () => {
      const size = Number((delta / 8 * 0.85).toPrecision(3)), layer = Number((3 * delta).toPrecision(3));
      const row = mesh.boundaries.find((x) => x.boundary === b[1]);
      if (row) { row.size = Math.min(row.size || size, size); row.layer = Math.max(row.layer || 0, layer); }
      else mesh.boundaries.push({ boundary: b[1], size, layer, growth: 1.2 });
      save(); drawBoundaries(); draw(); drawRepair();
      app.toast(`Boundary refinement on ${b[1]}: size ${size}, layer ${layer}. Create the mesh to apply it.`, "ok");
    } };
  };
  const drawRepair = () => {
    if (grid) grid.hidden = !!doc._repairOpen;
    wsHost.hidden = !doc._repairOpen;
    if (doc._repairOpen && !ws) {
      ws = repairWorkspace(app, doc, { mesh, save, defaultSize, numInput, field, leftSlot: rSlot,
        touched: () => { drawBoundaries(); drawRegions(); drawPaintCount(); draw(); } });
      setChildren(wsHost, ws);
    }
    if (doc._repairOpen && ws) requestAnimationFrame(() => ws._draw());
    if (!doc._repairOpen) return;
    const checks = (doc.caseValidation && doc.caseValidation.checks) || null;
    const bad = checks ? checks.filter((c) => c.status === "fail" || c.status === "warn")
      .sort((a, b) => (ORDER[a.status] ?? 9) - (ORDER[b.status] ?? 9)) : [];
    const open = (doc._repairOpenRows ||= {});
    const rows = [];
    if (!checks) rows.push(h("div.tree-note", { text: "No validation yet. Create the mesh first; its checks appear here." }));
    else if (!bad.length) rows.push(h("div.tn.muted", { style: { cursor: "default" } }, h("span.tn-tw.leaf"), icon("check-circle", 15),
      h("span.tn-label", { text: "Nothing to repair: every check passes" })));
    for (const c of bad) {
      const isOpen = open[c.name] !== false;
      const fix = autoFix(c);
      rows.push(h("div.tn", { role: "treeitem", tabindex: "0", "aria-expanded": String(isOpen), style: { "--depth": 0 },
        onclick: () => { open[c.name] = !isOpen; drawRepair(); },
        onkeydown: (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open[c.name] = !isOpen; drawRepair(); } } },
        h("span.tn-tw", {}, icon("chevron-right", 12)), icon(c.status === "fail" ? "x-circle" : "alert", 15),
        h("span.tn-label", { text: c.name }), h("span.tn-meta", { text: `${fmtValue(c.value)} / ${fmtValue(c.limit)}` })));
      if (isOpen) rows.push(h("div.rp-detail", {},
        h("p", { text: c.note || "" }),
        c.fix ? h("p", {}, h("b", { text: "Fix: " }), c.fix) : null,
        fix ? h("button.btn.sm.primary", { type: "button", onclick: fix.run }, icon("wrench", 14), fix.label) : null));
    }
    setChildren(rSlot, h("section.panel.repair-panel", {},
      h("div.combo-head", {}, icon("wrench", 15), h("span.panel-title", { text: "Repair mesh" }),
        h("span.props-sub", { text: checks ? `${bad.length} to repair` : "" }), h("span.spacer", { style: { flex: "1" } }),
        h("button.icon-btn", { type: "button", "aria-label": "Close repair", onclick: () => { doc._repairOpen = false; drawRepair(); drawState(); } }, icon("x", 14))),
      h("div.tree", { role: "tree", "aria-label": "Mesh checks to repair" }, rows),
      h("p.tree-note", { text: "Applying a fix edits the mesh settings in case.json only. Create the mesh again to see the result." })));
  };

  // ---- state line + build ----
  const stateLine = h("div.mesh-state");
  const drawState = () => {
    const st = (doc.caseDetail && doc.caseDetail.mesh_status) || { state: "none" };
    const n = (st.stats || {}).n_elements;
    let badge, text;
    if (st.state === "building") { badge = h("span.badge.b-run", {}, h("span.spinner.sm"), "creating"); text = "gmsh is meshing the case on the CPU; validation and views follow."; }
    else if (st.state === "failed") { badge = statusBadge("fail", "mesh failed"); text = st.error || "see mesh/mesh.log"; }
    else if (st.state === "done") {
      badge = st.up_to_date ? statusBadge("pass", "up to date") : statusBadge("warn", "out of date");
      text = `${fmt(n)} elements, ${fmt(st.n_dofs)} unknowns, built ${String(st.finished || "").replace("T", " ")}${st.up_to_date ? "" : ". Settings changed since: create it again, or Solve does."}`;
    } else { badge = statusBadge("info", "no mesh yet"); text = "Create the mesh to check it before solving (Solve also creates it)."; }
    const busy = st.state === "building";
    setChildren(stateLine, badge, h("span.mesh-state-txt", { text }), h("span.spacer"),
      st.state === "none" ? null : deleteButton(busy),
      repairButton(busy),
      createMeshButton(app, doc, busy),
      st.state === "failed" ? h("a.btn", { href: "#", onclick: async (e) => { e.preventDefault(); const u = await bridge.caseFileURL(doc.caseName, "mesh/mesh.log"); window.open(u, "_blank", "noopener"); } }, icon("terminal", 14), "mesh.log") : null);
  };
  drawState();
  doc._onMeshPoll = drawState;
  doc._onCaseSaved = () => { drawState(); drawValidation(); drawRepair(); updateEstimate(); };

  // ---- density ----
  const built = (ms.info && ms.info.density) || 1;
  const nBuilt = (ms.stats || {}).n_elements;
  const est = h("span.small.dim");
  const updateEstimate = () => {
    const dd = Number(mesh.density) || 1;
    est.textContent = nBuilt ? `≈ ${fmt(Math.round(nBuilt * (dd / built) ** 2))} elements (built: ${fmt(nBuilt)} at ×${fmt(built)})` : "the element count scales with density²";
  };
  const densIn = numInput(mesh.density, (v) => { mesh.density = v; slider.value = String(Math.log2(v)); updateEstimate(); save(); }, { label: "Element density factor", min: 0 });
  densIn.id = "mt-density";
  const slider = h("input.range", { type: "range", min: "-2", max: "2", step: "0.25", value: String(Math.log2(Number(mesh.density) || 1)), "aria-label": "Element density (log scale)",
    oninput: (e) => { const v = Number((2 ** Number(e.target.value)).toPrecision(3)); mesh.density = v; densIn.value = String(v); updateEstimate(); },
    onchange: () => save() });
  updateEstimate();
  const densityPanel = panel("Global density", h("div.mt-stack", {},
    h("div.mt-row", {}, field("Density ×", densIn, "1 = template sizes; 2 = every element half as big (≈ 4× elements)"), h("span.range-wrap", {}, h("span.small.dim", { text: "×¼" }), slider, h("span.small.dim", { text: "×4" }))),
    est));

  // ---- boundary refinement ----
  const names = (doc.caseMesh && Object.keys(doc.caseMesh.boundary_edges || {})) || app.boundaryNames(doc);
  const bBody = h("div.mt-stack");
  const drawBoundaries = () => {
    const rows = mesh.boundaries.map((b, i) => h("div.mt-row.mt-item", {},
      h("span.swatch", { style: { background: boundaryColor(b.boundary, Math.max(0, names.indexOf(b.boundary))) } }),
      field("Boundary", h("select.sel", { "aria-label": "Boundary", onchange: (e) => { b.boundary = e.target.value; save(); draw(); } },
        (names.includes(b.boundary) ? names : [b.boundary, ...names]).map((n) => h("option", { value: n, text: n, selected: n === b.boundary || null })))),
      field("Size", numInput(b.size, (v) => { b.size = v; save(); }, { label: "Element size", min: 0 }), "Element size on the boundary (model length units)"),
      field("Layer", numInput(b.layer, (v) => { b.layer = v; save(); }, { label: "Layer thickness", min: -1e-300 }), "Keep that size up to this distance from the boundary"),
      field("Growth", numInput(b.growth, (v) => { b.growth = v; save(); }, { label: "Growth rate", min: 1 }), "Size ratio of neighbouring elements beyond the layer (1.1–1.5)"),
      h("button.icon-btn", { type: "button", "aria-label": `Remove refinement on ${b.boundary}`, "data-tip": "Remove", onclick: () => { mesh.boundaries.splice(i, 1); save(); drawBoundaries(); draw(); } }, icon("x", 14))));
    setChildren(bBody, rows.length ? rows : h("div.small.dim", { text: "None. Refine along a named boundary, for example a charged wall." }),
      h("button.btn.sm#mt-boundary-add", { type: "button", disabled: !names.length || null, "data-tip": names.length ? null : "Boundary names are known after the mesh is built",
        onclick: () => { mesh.boundaries.push({ boundary: names[0], size: defaultSize(), layer: 0, growth: 1.2 }); save(); drawBoundaries(); draw(); } }, icon("file-plus", 14), "Add boundary refinement"));
  };
  drawBoundaries();

  // ---- regions ----
  const rBody = h("div.mt-stack");
  const drawRegions = () => {
    const rows = mesh.regions.map((r, i) => {
      const n = (k, label, tip, min) => field(label, numInput(r[k], (v) => { r[k] = v; save(); draw(); }, { label, min }), tip);
      return h("div.mt-row.mt-item", {},
        h("span.kind", { text: r.kind === "box" ? "Box" : "Disk" }),
        r.kind === "box" ? [n("x0", "x₀"), n("x1", "x₁"), n("y0", "y₀"), n("y1", "y₁")] : [n("x", "x"), n("y", "y"), n("r", "radius", "Size held inside this radius", -1e-300)],
        n("size", "Size", "Element size inside", 0), n("growth", "Growth", "Size ratio of neighbours outside", 1),
        h("button.icon-btn", { type: "button", "aria-label": "Remove region", "data-tip": "Remove", onclick: () => { mesh.regions.splice(i, 1); save(); drawRegions(); draw(); } }, icon("x", 14)));
    });
    setChildren(rBody, rows.length ? rows : h("div.small.dim", { text: "None. Draw a box or a disk on the mesh with the Box and Disk tools." }));
  };
  drawRegions();

  // ---- sharp corners ----
  const c = mesh.corners;
  const cornersFound = (ms.info && ms.info.corners) || (doc.caseMesh && doc.caseMesh.corners) || null;
  const cornerPanel = panel("Sharp corners", h("div.mt-stack", {},
    h("label.chk", {}, h("input#mt-corners", { type: "checkbox", checked: !!c.enabled || null, onchange: (e) => { c.enabled = e.target.checked; save(); app.renderToolbarNow(); } }),
      h("span", { text: "Refine re-entrant and acute corners (found automatically)" })),
    h("div.mt-row", {},
      field("Size", numInput(c.size || "", (v) => { c.size = v || 0; save(); }, { label: "Corner element size", placeholderText: "auto" }), "Element size at the corner (blank = radius / 8)"),
      field("Radius", numInput(c.radius || "", (v) => { c.radius = v || 0; save(); }, { label: "Corner radius", placeholderText: "auto" }), "Size held within this distance (blank = 2 % of the domain)"),
      field("Acute below", numInput(c.max_angle, (v) => { c.max_angle = v; save(); }, { label: "Acute angle limit", min: 0 }), "Interior angles below this (degrees) count as sharp; re-entrant (> 185°) always do")),
    h("div.small.dim", { text: cornersFound ? `${cornersFound.length} corner${cornersFound.length === 1 ? "" : "s"} found at the last build${cornersFound.length ? `: ${cornersFound.map((k) => `(${fmt(k.x, 3)}, ${fmt(k.y, 3)}) ${Math.round(k.angle)}°`).join(", ")}` : ""}` : "Corners are listed after the next build." })));

  // ---- paint ----
  doc.paintUI = doc.paintUI || { r: null, size: null };
  const paintCount = h("span.small.dim");
  const drawPaintCount = () => { paintCount.textContent = `${mesh.paint.length} painted spot${mesh.paint.length === 1 ? "" : "s"}`; };
  drawPaintCount();
  const paintPanel = panel("Spray paint", h("div.mt-stack", {},
    h("div.small.dim", { text: "Pick the Paint tool and drag over the mesh where you want smaller elements; the Erase tool removes spots." }),
    h("div.mt-row", {},
      field("Brush radius", numInput(doc.paintUI.r ?? "", (v) => { doc.paintUI.r = v; draw(); }, { label: "Brush radius", placeholderText: "auto" }), "In model length units (blank = 3 % of the view)"),
      field("Spot size", numInput(doc.paintUI.size ?? "", (v) => { doc.paintUI.size = v; }, { label: "Element size in painted spots", placeholderText: "auto" }), "Element size inside painted spots (blank = half the smallest element of the last build)"),
      field("Growth", numInput(mesh.paint_growth, (v) => { mesh.paint_growth = v; save(); }, { label: "Paint growth rate", min: 1 }))),
    h("div.mt-row", {}, paintCount, h("span.spacer"),
      h("button.btn.sm", { type: "button", onclick: () => { mesh.paint = []; save(); drawPaintCount(); draw(); } }, icon("reset", 14), "Clear paint"))));

  function defaultSize() {
    const hmin = (ms.stats || {}).h_min;
    if (isNum(hmin)) return Number((hmin * 0.5).toPrecision(2));
    const bb = doc.caseMesh && doc.caseMesh.bbox;
    return bb ? Number((Math.max(bb[1] - bb[0], bb[3] - bb[2]) / 200).toPrecision(2)) : 0.05;
  }

  // ---- canvas ----
  const canvas = h("canvas.mt-canvas", { tabindex: "0", "aria-label": "Mesh with refinements. Tools: pan, box, disk, paint, erase." });
  const hint = h("div.mt-hint");
  const canvasBox = h("div.mt-canvas-box", {}, canvas, hint);
  const view = doc.meshView || (doc.meshView = { cx: 0, cy: 0, k: 0 });
  let geom = null;
  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const toScreen = (x, y) => [(x - view.cx) * view.k + canvas.width / 2, canvas.height / 2 - (y - view.cy) * view.k];
  const toWorld = (sx, sy) => [(sx - canvas.width / 2) / view.k + view.cx, view.cy - (sy - canvas.height / 2) / view.k];
  const fit = () => {
    const bb = geom.bbox;
    const W = canvas.width, H = canvas.height;
    view.cx = (bb[0] + bb[1]) / 2; view.cy = (bb[2] + bb[3]) / 2;
    view.k = 0.92 * Math.min(W / Math.max(1e-12, bb[1] - bb[0]), H / Math.max(1e-12, bb[3] - bb[2]));
  };
  const brushR = () => doc.paintUI.r || 0.03 * Math.min(canvas.width, canvas.height) / view.k;
  let pending = false;
  function draw() {
    if (pending) return;
    pending = true;
    requestAnimationFrame(() => { pending = false; paint(); });
  }
  let drag = null;
  let hover = null;
  function paint() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
      const oldW = canvas.width;
      canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
      if (view.k && oldW) view.k *= canvas.width / oldW;
    }
    const g = canvas.getContext("2d");
    g.fillStyle = css("--plot-bg"); g.fillRect(0, 0, canvas.width, canvas.height);
    if (!geom) return;
    if (!view.k) fit();
    // mesh edges. Large meshes (10^5–10^6 triangles): skip triangles off screen, and
    // draw triangles under ~2 px as one shaded pixel instead of three lines, so a frame
    // stays a single cheap pass however fine the mesh is.
    const P = geom.nodes;
    const W = canvas.width, H = canvas.height;
    const k = view.k, ox = W / 2 - view.cx * k, oy = H / 2 + view.cy * k;
    const tiny = 2 * dpr;
    g.lineWidth = 1;
    g.strokeStyle = css("--border-strong");
    g.fillStyle = css("--border-strong");
    g.beginPath();
    const T = geom.triangles;
    for (let i = 0; i < T.length; i++) {
      const t = T[i];
      const a = P[t[0]], b = P[t[1]], c = P[t[2]];
      const ax = a[0] * k + ox, ay = oy - a[1] * k;
      const bx = b[0] * k + ox, by = oy - b[1] * k;
      const cx = c[0] * k + ox, cy = oy - c[1] * k;
      const x0 = Math.min(ax, bx, cx), x1 = Math.max(ax, bx, cx);
      const y0 = Math.min(ay, by, cy), y1 = Math.max(ay, by, cy);
      if (x1 < 0 || y1 < 0 || x0 > W || y0 > H) continue;
      if (x1 - x0 < tiny && y1 - y0 < tiny) { g.fillRect(x0 | 0, y0 | 0, 1, 1); continue; }
      g.moveTo(ax, ay); g.lineTo(bx, by); g.lineTo(cx, cy); g.closePath();
    }
    g.stroke();
    // boundaries
    g.lineWidth = 2.5 * dpr;
    Object.entries(geom.boundary_edges || {}).forEach(([name, edges], i) => {
      g.strokeStyle = boundaryColor(name, i);
      g.setLineDash(/axis|symmetry/.test(name) ? [6 * dpr, 4 * dpr] : []);
      g.beginPath();
      for (const [a, b] of edges) { const [x0, y0] = toScreen(P[a][0], P[a][1]); const [x1, y1] = toScreen(P[b][0], P[b][1]); g.moveTo(x0, y0); g.lineTo(x1, y1); }
      g.stroke();
    });
    g.setLineDash([]);
    // refinements
    const acc = css("--accent");
    g.lineWidth = 2 * dpr;
    g.strokeStyle = acc;
    g.fillStyle = css("--accent-soft");
    for (const r of mesh.regions) {
      g.beginPath();
      if (r.kind === "box") {
        const [x0, y0] = toScreen(Math.min(r.x0, r.x1), Math.max(r.y0, r.y1));
        const [x1, y1] = toScreen(Math.max(r.x0, r.x1), Math.min(r.y0, r.y1));
        g.rect(x0, y0, x1 - x0, y1 - y0);
      } else {
        const [x, y] = toScreen(r.x, r.y);
        g.arc(x, y, Math.max(2, (r.r || 0) * view.k), 0, 2 * Math.PI);
      }
      g.fill(); g.stroke();
    }
    for (const s of mesh.paint) {
      const [x, y] = toScreen(s.x, s.y);
      g.beginPath(); g.arc(x, y, Math.max(1.5, s.r * view.k), 0, 2 * Math.PI); g.fill();
    }
    // corners found at the last build
    g.fillStyle = css("--warn");
    for (const k of geom.corners || []) {
      const [x, y] = toScreen(k.x, k.y);
      const a = 6 * dpr;
      g.beginPath(); g.moveTo(x, y - a); g.lineTo(x + a, y); g.lineTo(x, y + a); g.lineTo(x - a, y); g.closePath(); g.fill();
    }
    // drag preview / brush
    g.strokeStyle = acc; g.setLineDash([5 * dpr, 4 * dpr]);
    if (drag && drag.tool === "box") { g.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0); }
    if (drag && drag.tool === "disk") { g.beginPath(); g.arc(drag.x0, drag.y0, Math.hypot(drag.x1 - drag.x0, drag.y1 - drag.y0), 0, 2 * Math.PI); g.stroke(); }
    g.setLineDash([]);
    const tool = doc.meshTool || "pan";
    if (hover && (tool === "paint" || tool === "erase")) {
      g.strokeStyle = tool === "erase" ? css("--bad") : acc;
      g.beginPath(); g.arc(hover[0], hover[1], brushR() * view.k, 0, 2 * Math.PI); g.stroke();
    }
  }
  const HINTS = {
    pan: "Pan: drag to move, wheel to zoom, double-click to fit.",
    box: "Box: drag a rectangle; elements inside get the region size.",
    disk: "Disk: press at the centre and drag out the radius.",
    paint: "Paint: drag to spray finer elements (brush and spot size in Spray paint).",
    erase: "Erase: drag over painted spots or regions to remove them.",
  };
  doc._onTool = () => { hint.textContent = HINTS[doc.meshTool || "pan"]; canvas.dataset.tool = doc.meshTool || "pan"; draw(); };
  doc._onTool();
  const pos = (e) => { const r = canvas.getBoundingClientRect(); const dpr = canvas.width / r.width; return [(e.clientX - r.left) * dpr, (e.clientY - r.top) * dpr]; };
  const sprayAt = (sx, sy) => {
    const [x, y] = toWorld(sx, sy);
    const r = brushR();
    const last = mesh.paint[mesh.paint.length - 1];
    if (last && Math.hypot(last.x - x, last.y - y) < 0.5 * r) return;
    mesh.paint.push({ x: Number(x.toPrecision(6)), y: Number(y.toPrecision(6)), r: Number(r.toPrecision(4)), size: doc.paintUI.size || defaultSize() });
    drawPaintCount();
  };
  const eraseAt = (sx, sy) => {
    const [x, y] = toWorld(sx, sy);
    const r = brushR();
    const n0 = mesh.paint.length + mesh.regions.length;
    mesh.paint = mesh.paint.filter((s) => Math.hypot(s.x - x, s.y - y) > r + s.r);
    mesh.regions = mesh.regions.filter((g) => {
      const cx = g.kind === "box" ? (g.x0 + g.x1) / 2 : g.x, cy = g.kind === "box" ? (g.y0 + g.y1) / 2 : g.y;
      return Math.hypot(cx - x, cy - y) > r;
    });
    if (mesh.paint.length + mesh.regions.length !== n0) { drawPaintCount(); drawRegions(); }
  };
  canvas.addEventListener("pointerdown", (e) => {
    if (!geom) return;
    canvas.setPointerCapture(e.pointerId);
    const [sx, sy] = pos(e);
    const tool = e.button === 1 ? "pan" : doc.meshTool || "pan";
    drag = { tool, x0: sx, y0: sy, x1: sx, y1: sy, cx: view.cx, cy: view.cy };
    if (tool === "paint") sprayAt(sx, sy);
    if (tool === "erase") eraseAt(sx, sy);
    draw();
  });
  canvas.addEventListener("pointermove", (e) => {
    const [sx, sy] = pos(e);
    hover = [sx, sy];
    if (drag) {
      drag.x1 = sx; drag.y1 = sy;
      if (drag.tool === "pan") { view.cx = drag.cx - (sx - drag.x0) / view.k; view.cy = drag.cy + (sy - drag.y0) / view.k; }
      if (drag.tool === "paint") sprayAt(sx, sy);
      if (drag.tool === "erase") eraseAt(sx, sy);
    }
    draw();
  });
  canvas.addEventListener("pointerleave", () => { hover = null; draw(); });
  canvas.addEventListener("pointerup", () => {
    if (!drag) return;
    const d = drag;
    drag = null;
    const [x0, y0] = toWorld(d.x0, d.y0), [x1, y1] = toWorld(d.x1, d.y1);
    const sig = (v) => Number(v.toPrecision(6));
    if (d.tool === "box" && Math.abs(d.x1 - d.x0) > 4 && Math.abs(d.y1 - d.y0) > 4) {
      mesh.regions.push({ kind: "box", x0: sig(Math.min(x0, x1)), x1: sig(Math.max(x0, x1)), y0: sig(Math.min(y0, y1)), y1: sig(Math.max(y0, y1)), size: defaultSize(), growth: 1.2 });
      drawRegions(); save();
    } else if (d.tool === "disk" && Math.hypot(d.x1 - d.x0, d.y1 - d.y0) > 4) {
      mesh.regions.push({ kind: "disk", x: sig(x0), y: sig(y0), r: sig(Math.hypot(x1 - x0, y1 - y0)), size: defaultSize(), growth: 1.2 });
      drawRegions(); save();
    } else if (d.tool === "paint" || d.tool === "erase") save();
    draw();
  });
  canvas.addEventListener("dblclick", () => { if (geom) { fit(); draw(); } });
  canvas.addEventListener("wheel", (e) => {
    if (!geom) return;
    e.preventDefault();
    const [sx, sy] = pos(e);
    const [wx, wy] = toWorld(sx, sy);
    view.k *= Math.exp(-e.deltaY * 0.0015);
    const [nx, ny] = toWorld(sx, sy);
    view.cx += wx - nx; view.cy += wy - ny;
    draw();
  }, { passive: false });
  new ResizeObserver(() => draw()).observe(canvas);

  const canvasPanel = h("section.panel.mt-canvas-panel", {},
    h("header.panel-head", {}, h("h2", { text: "Mesh and refinements" }), h("span.spacer"),
      h("span.small.dim", { text: "Toolbar: Pan · Box · Disk · Paint · Erase" })),
    canvasBox);
  if (doc.caseMesh) { geom = doc.caseMesh; draw(); }
  else if ((det.mesh_files || []).includes("mesh/mesh.json")) {
    setChildren(hint, h("span.spinner.sm"), " Loading the mesh…");
    bridge.caseJSON(doc.caseName, "mesh/mesh.json").then((m) => { doc.caseMesh = m; geom = m; doc._onTool(); draw(); drawBoundaries(); })
      .catch((e) => { hint.textContent = `Could not load mesh.json: ${e.message}`; });
  } else hint.textContent = "Create the mesh once to draw on it (Create mesh).";

  // ---- images + template sizes ----
  const views = (det.mesh_files || []).filter((f) => /view_[ab]\.png$/.test(f)).map((f) => {
    const img = h("img", { alt: f, loading: "lazy" });
    bridge.caseFileURL(doc.caseName, f).then((u) => { img.src = `${u}?t=${encodeURIComponent(ms.finished || "")}`; });
    return h("figure.img-frame", {}, h("div.img-box", {}, img), h("figcaption", { text: /view_a/.test(f) ? "View A: whole domain by boundary group" : "View B: finest region by element quality" }));
  });

  setChildren(root, pageShell("mesh", "Mesh", "The case's mesh lives in cases/" + doc.caseName + "/mesh/. Tools here change case.json; build to check the result before solving.",
    h("div.panel.mesh-state-panel", {}, stateLine), wsHost, vSlot,
    grid = h("div.mt-grid", {},
      h("div.stack", {}, densityPanel,
        panel("Boundary refinement", bBody),
        panel("Refinement regions", rBody),
        cornerPanel, paintPanel,
        stepParams(app, doc, "mesh", "Template mesh sizes", "This template sets its sizes from the Debye length; it has no size parameters.")),
      h("div.stack", {}, canvasPanel, views.length ? h("div.mesh-views", {}, views) : null))));
  drawRepair();
  requestAnimationFrame(draw);
}
