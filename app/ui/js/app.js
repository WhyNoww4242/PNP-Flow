// pnpflow desktop app: the shell (FreeCAD-style layout, docs/PLAN.md §1).
// Menu bar · workbench tabs (construction steps) · one toolbar row · combo panel
// (model tree + View/Data property editor) · canvas · document tabs · status bar.

import { bridge, optional, inTauri } from "./bridge.js";
import { h, $, $$, setChildren, fmt, store, debounce, placeholder, unitOf, stripUnit, sameValue, prettyName, isNum } from "./util.js";
import { icon } from "./icons.js";
import { initColormaps, colormapNames, colormapType, colormapSource } from "./colormaps.js";
import { prepareMesh, prepareTimeFields } from "./glfield.js";
import { FieldView, boundaryColor } from "./fieldview.js";
import { resizeAll } from "./plots.js";
import { runStatus, healthBadge, stageBars, RUNNING, STATE_WORD, isStale } from "./progress.js";
import { renderStart, caseTitle, runStatusWord } from "./start.js";
import { renderGeometry, renderMesh, renderMaterials, renderBoundaries, renderEquations, renderStudy, updateStudy } from "./pages.js";
import { renderResults, RESULT_VIEWS } from "./results.js";

const POLL_MS = 1500;
const WB = [
  { id: "geometry", label: "Geometry", icon: "geometry", key: "1" },
  { id: "mesh", label: "Mesh", icon: "mesh", key: "2" },
  { id: "materials", label: "Materials", icon: "materials", key: "3" },
  { id: "boundaries", label: "Boundaries", icon: "boundary", key: "4" },
  { id: "equations", label: "Equations", icon: "equations", key: "5" },
  { id: "study", label: "Study", icon: "study", key: "6" },
  { id: "results", label: "Results", icon: "results", key: "7" },
];
const LATER = "Arrives with the project file (build plan, Phase 1 and 5)";

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const S = {
  runs: [], runsLoaded: false, resultsDir: "",
  cases: {}, casesLoaded: false, casesError: null,
  docs: [{ id: "start", kind: "start" }], active: "start",
  ptab: store.get("ptab", "data"),
  units: store.get("units", "SI"),
  expert: store.get("expert", false),
  themePref: store.get("theme", "system"),
  connected: null, startSig: "", startFilter: "",
  cmSource: null, docSeq: 0,
};

const app = {
  S, inTauri,
  activeDoc: () => S.docs.find((d) => d.id === S.active) || S.docs[0],
};

// ---------------------------------------------------------------------------
// Model helpers used by the pages
// ---------------------------------------------------------------------------
app.caseOf = (doc) => S.cases[doc.case] || { params: {}, info: {} };

/** Parameter rows: template defaults, the run's inputs and the user's edits. */
app.paramRows = (doc) => {
  const c = app.caseOf(doc);
  const dflt = c.params || {};
  const s = doc.summary || {};
  // a finished run reports its inputs; a running one is known from its launch command
  const base = s.input || (doc.listing && doc.listing.launch_params ? Object.assign({}, dflt, doc.listing.launch_params) : {});
  const info = Object.assign({}, s.param_info || {}, c.info || {});
  const keys = [...new Set([...Object.keys(dflt), ...Object.keys(base)])];
  return keys.map((k) => {
    const has = Object.prototype.hasOwnProperty;
    const value = has.call(doc.edits, k) ? doc.edits[k] : has.call(base, k) ? base[k] : dflt[k];
    const d = dflt[k];
    return { key: k, value, dflt: d, info: typeof info[k] === "string" ? info[k] : "", unit: unitOf(info[k]),
      changed: d !== undefined && !sameValue(value, d), edited: has.call(doc.edits, k), invalid: doc.invalid && doc.invalid[k] };
  });
};

app.boundaryNames = (doc) => {
  if (doc.mesh) return Object.keys(doc.mesh.boundaries);
  const s = doc.summary || {};
  if (s.mesh && s.mesh.boundaries && typeof s.mesh.boundaries === "object") return Object.keys(s.mesh.boundaries);
  return s.model && s.model.boundaries ? Object.keys(s.model.boundaries) : [];
};

app.symmetryOf = (doc) => {
  if (doc.fieldsJSON && doc.fieldsJSON.symmetry) return doc.fieldsJSON.symmetry;
  const s = doc.summary;
  if (s && s.model) return s.model.coord === "axisym" ? "axisym" : "planar";
  return null;
};

app.bcSummary = (doc, b) => {
  const bc = doc.summary && doc.summary.model && doc.summary.model.boundaries && doc.summary.model.boundaries[b];
  if (!bc) return /axis|symmetry/.test(b) ? "symmetry axis (natural condition)" : "natural condition (default)";
  return Object.entries(bc).map(([k, v]) => `${prettyName(k)}: ${v}`).join(", ");
};

app.entryOf = (name) => S.runs.find((r) => r.name === name) || null;
app.statusOf = (doc) => (doc && doc.name ? runStatus(doc.progress, app.entryOf(doc.name)) : null);
app.canStop = (doc) => { const e = doc && app.entryOf(doc.name); return !!(e && e.proc && e.proc.alive); };

app.suggestName = (doc) => {
  const d = new Date();
  const p = (n) => String(n).padStart(2, "0");
  return `${doc.case || "run"}_${String(d.getFullYear()).slice(2)}${p(d.getMonth() + 1)}${p(d.getDate())}_${p(d.getHours())}${p(d.getMinutes())}`;
};

/** Values that differ from the template defaults (what gets passed as --set). */
function changedParams(doc) {
  const out = {};
  for (const r of app.paramRows(doc)) if (r.changed || (r.dflt === undefined && r.edited)) out[r.key] = r.value;
  return out;
}

app.commandLine = (doc, name) => {
  const sets = Object.entries(changedParams(doc)).map(([k, v]) => ` --set ${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join("");
  const env = { cpu: "JAX_PLATFORMS=cpu ", gpu: "JAX_PLATFORMS=cuda " }[app.device()] || "";
  return `${env}python run.py ${doc.case} --name ${name || "<name>"}${sets}`;
};

// Compute device for new runs: "auto" | "cpu" | "gpu" (remembered per browser)
app.device = () => { try { return localStorage.getItem("pnpflow.device") || "auto"; } catch { return "auto"; } };
app.setDevice = (v) => { try { localStorage.setItem("pnpflow.device", v); } catch { /* storage blocked */ } };
let devicesPromise = null;
app.devices = () => (devicesPromise ||= bridge.devices().catch((e) => ({ gpu: false, devices: [], error: e.message })));

// ---------------------------------------------------------------------------
// Toasts, tooltip, lightbox, dialogs
// ---------------------------------------------------------------------------
app.toast = (text, kind = "info") => {
  const ic = { ok: "check-circle", error: "x-circle", warn: "alert", info: "info" }[kind] || "info";
  const t = h(`div.toast.toast-${kind}`, { role: kind === "error" ? "alert" : "status" }, icon(ic, 16), h("span", { text }));
  $("#toasts").appendChild(t);
  setTimeout(() => t.classList.add("out"), 4200);
  setTimeout(() => t.remove(), 4500);
};

function status(text) {
  $("#sb-msg").textContent = text;
}

const tip = { el: null, timer: 0, target: null };
function tipContent(target) {
  const title = target.dataset.tipTitle || "";
  const body = target.dataset.tip || "";
  const key = target.dataset.tipKey || "";
  const why = target.dataset.tipWhy || "";
  const nodes = [];
  if (title) nodes.push(h("b", { text: title, style: { fontWeight: "600" } }));
  if (body && body !== title) nodes.push(h("span", { text: body, style: { whiteSpace: "pre-line" } }));
  if (key) nodes.push(h("kbd", { text: key }));
  return why ? [h("span", {}, nodes, h("span.tt-why", { text: why }))] : nodes;
}
function showTip(target) {
  const el = $("#tooltip");
  setChildren(el, tipContent(target));
  el.hidden = false;
  const r = target.getBoundingClientRect();
  const tr = el.getBoundingClientRect();
  let x = r.left + r.width / 2 - tr.width / 2;
  let y = r.bottom + 6;
  if (y + tr.height > innerHeight - 4) y = r.top - tr.height - 6;
  x = Math.max(4, Math.min(innerWidth - tr.width - 4, x));
  el.style.left = `${x}px`;
  el.style.top = `${y}px`;
}
function hideTip() {
  clearTimeout(tip.timer);
  tip.target = null;
  $("#tooltip").hidden = true;
}
function initTooltips() {
  const over = (e) => {
    const t = e.target.closest && e.target.closest("[data-tip],[data-tip-title]");
    if (t === tip.target) return;
    hideTip();
    if (!t || document.querySelector(".menu.open")) return;   // no tooltips over an open menu
    tip.target = t;
    tip.timer = setTimeout(() => tip.target === t && t.isConnected && showTip(t), e.type === "focusin" ? 0 : 380);
  };
  document.addEventListener("mouseover", over);
  document.addEventListener("focusin", over);
  document.addEventListener("mousedown", hideTip);
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); });
}

app.lightbox = (items, idx) => {
  const lb = $("#lightbox");
  const show = async (i) => {
    idx = (i + items.length) % items.length;
    const it = items[idx];
    const url = await bridge.imageURL(it.name, it.file).catch(() => "");
    setChildren(lb, h("div.lb-frame", { role: "dialog", "aria-modal": "true", "aria-label": it.caption },
      h("div.lb-head", {}, h("span.mono", { text: it.caption }),
        h("span.dim.small", { text: items.length > 1 ? `${idx + 1} of ${items.length}  ·  arrow keys browse, Esc closes` : "Esc closes" }),
        url && !url.startsWith("data:") ? h("a.btn.sm", { href: url, target: "_blank", rel: "noopener" }, icon("external", 14), "Open") : null,
        h("button.icon-btn#lb-close", { type: "button", "aria-label": "Close", onclick: close }, icon("x", 16))),
      h("div.lb-body", {}, h("img", { src: url, alt: it.caption }))));
    $("#lb-close").focus();
  };
  const close = () => { lb.classList.add("hidden"); lb.onkeydown = null; document.removeEventListener("keydown", key, true); };
  const key = (e) => {
    if (e.key === "Escape") { close(); e.stopPropagation(); e.preventDefault(); }
    else if (e.key === "ArrowRight") { show(idx + 1); e.preventDefault(); e.stopPropagation(); }
    else if (e.key === "ArrowLeft") { show(idx - 1); e.preventDefault(); e.stopPropagation(); }
  };
  lb.classList.remove("hidden");
  lb.onclick = (e) => { if (e.target === lb) close(); };
  document.addEventListener("keydown", key, true);
  show(idx);
};

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
const mqLight = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;
function resolvedTheme() {
  const q = new URLSearchParams(location.search).get("theme");
  const pref = q === "light" || q === "dark" ? q : S.themePref;
  if (pref === "light" || pref === "dark") return pref;
  return mqLight && mqLight.matches ? "light" : "dark";
}
function applyTheme() {
  const t = resolvedTheme();
  if (document.documentElement.dataset.theme !== t) {
    document.documentElement.dataset.theme = t;
    themeChanged();
  }
}
function setThemePref(p) {
  S.themePref = p;
  store.set("theme", p);
  const u = new URL(location.href);
  if (u.searchParams.has("theme")) { u.searchParams.delete("theme"); history.replaceState(null, "", u); }
  applyTheme();
  renderToolbar();
  status(`Theme: ${p === "system" ? "follows the system" : p}`);
}
function themeChanged() {
  app.field && app.field.themeChanged();
  // Plotly bakes colours in: re-render the current page
  const d = app.activeDoc();
  if (d.kind !== "start" && !(d.wb === "results" && d.rview === "field")) renderCanvas();
}

// ---------------------------------------------------------------------------
// Commands (menus, toolbar, keyboard and tooltips all read from here)
// ---------------------------------------------------------------------------
const fieldActive = () => { const d = app.activeDoc(); return d.kind !== "start" && d.wb === "results" && d.rview === "field" && app.field.mesh && app.field.docId === d.id; };
const needField = () => (fieldActive() ? true : "Open Results, Field map first");
const docOpen = () => (app.activeDoc().kind !== "start" ? true : "Open a template or a run first");

const CMD = {
  new: { label: "New project from template", icon: "file-plus", key: "N", run: () => { activate("start"); requestAnimationFrame(() => { const t = $(".tpl"); if (t) t.focus(); }); } },
  open: { label: "Open run", icon: "folder-open", key: "O", run: () => { activate("start"); requestAnimationFrame(() => { const f = $("#run-filter"); if (f) f.focus(); }); } },
  close: { label: "Close document", icon: "x", enabled: docOpen, run: () => closeDoc(S.active) },
  refresh: { label: "Refresh runs", icon: "refresh", key: "R", run: () => refreshAll(true) },
  savePNG: { label: "Save the field view as PNG", icon: "image-down", key: "P", enabled: needField, run: () => app.field.savePNG(app.activeDoc().name) },
  copyCmd: { label: "Copy the run command", icon: "script", enabled: docOpen, run: async () => {
    const d = app.activeDoc();
    const text = app.commandLine(d, d.nextName || app.suggestName(d));
    try { await navigator.clipboard.writeText(text); app.toast("Run command copied", "ok"); } catch { app.toast(text, "info"); }
  } },
  resetParams: { label: "Reset parameters to the template values", icon: "reset",
    enabled: () => { const d = app.activeDoc(); return d.kind === "start" ? "Open a template or a run first" : Object.keys(d.edits || {}).length ? true : "No parameter has been changed"; },
    run: () => { const d = app.activeDoc(); d.edits = {}; d.invalid = {}; renderProps(); renderCanvasIf(["study"]); status("Parameters reset to the run's values"); } },
  themeToggle: { label: "Switch dark / light theme", icon: "sun", key: "T", run: () => setThemePref(resolvedTheme() === "dark" ? "light" : "dark") },
  themeSystem: { label: "Follow the system theme", radio: () => S.themePref === "system", run: () => setThemePref("system") },
  themeDark: { label: "Dark", radio: () => S.themePref === "dark", run: () => setThemePref("dark") },
  themeLight: { label: "Light", radio: () => S.themePref === "light", run: () => setThemePref("light") },
  fit: { label: "Fit all", icon: "fit", key: "F", enabled: needField, run: () => app.field.zoomAll() },
  focus: { label: "Zoom to the focus region", icon: "focus", key: "G", enabled: needField, run: () => app.field.zoomFocus() },
  edges: { label: "Mesh edges", icon: "edges", key: "E", enabled: needField, pressed: () => !!app.field.opts.edges, run: () => app.field.toggleOpt("edges") },
  bnd: { label: "Boundaries", icon: "boundary", key: "B", enabled: needField, pressed: () => !!app.field.opts.bnd, run: () => app.field.toggleOpt("bnd") },
  contours: { label: "Contour lines", icon: "contours", key: "C", enabled: needField, pressed: () => !!app.field.opts.contours, run: () => app.field.toggleOpt("contours") },
  stream: { label: "Streamlines of u", icon: "flow", key: "S",
    enabled: () => (fieldActive() ? (app.field.mesh.vectors.u ? true : "This run has no velocity field") : "Open Results, Field map first"),
    pressed: () => !!app.field.opts.stream, run: () => app.field.toggleOpt("stream") },
  mirror: { label: "Mirror about the axis (axisymmetric)", icon: "mirror", key: "M",
    enabled: () => (fieldActive() ? (app.field.mesh.axisym ? true : "Only for axisymmetric runs") : "Open Results, Field map first"),
    pressed: () => !!app.field.opts.mirror, run: () => setMirror(!app.field.opts.mirror) },
  combo: { label: "Show the model panel", icon: "list", key: "[", pressed: () => !$("#workspace").classList.contains("combo-hidden"),
    run: () => { $("#workspace").classList.toggle("combo-hidden"); setTimeout(() => { app.field.resize(); resizeAll(document); }, 30); } },
  solve: { label: "Solve", icon: "play", key: "F8", primary: true,
    enabled: () => { const d = app.activeDoc(); if (d.kind === "start") return "Open a template or a run first"; if (!S.casesLoaded) return "Loading the templates from run.py"; if (!S.cases[d.case]) return `Template '${d.case}' is not available from run.py`; const st = app.statusOf(d); return st && st.running ? "This run is already in progress" : true; },
    run: () => { const d = app.activeDoc(); if (d.wb !== "study") setWorkbench("study"); app.solve(d, {}); } },
  stop: { label: "Stop the run", icon: "stop", key: "Shift+F8",
    enabled: () => { const d = app.activeDoc(); const st = app.statusOf(d); return st && st.running ? (app.canStop(d) ? true : "Only runs started from this app can be stopped") : "No run in progress"; },
    run: () => app.stop(app.activeDoc()) },
  units: { label: "Units: SI / dimensionless", icon: "units", key: "U", run: () => setUnits(S.units === "SI" ? "dimensionless" : "SI") },
  expert: { label: "Expert mode", icon: "sliders", key: "X", pressed: () => S.expert, run: () => setExpert(!S.expert) },
  help: { label: "Keyboard shortcuts", icon: "keyboard", key: "?", run: () => openHelp() },
  about: { label: "About pnpflow", icon: "info", run: () => openAbout() },
  editData: { label: "Edit the template parameters", icon: "sliders", enabled: docOpen, run: () => app.setPropsTab("data") },
  importGeo: { label: "Import CAD or mesh", icon: "import", enabled: () => LATER, run: () => {} },
  gmshScript: { label: "Edit gmsh script", icon: "script", enabled: () => LATER, run: () => {} },
  refine: { label: "Mesh sizes and refinement", icon: "sliders", enabled: () => LATER, run: () => {} },
  meshOnField: { label: "Show the mesh on the field map", icon: "edges", enabled: docOpen, run: () => { app.field.setOpt("edges", true); app.setResultsView("field"); } },
  addIon: { label: "Add an ion from the library", icon: "ion", enabled: () => LATER, run: () => {} },
  editBC: { label: "Set a boundary condition", icon: "electrode", enabled: () => LATER, run: () => {} },
  editPhysics: { label: "Choose physics (ions, flow)", icon: "equations", enabled: () => LATER, run: () => {} },
  diagnose: { label: "Mesh views and diagnose", icon: "diagnose", enabled: docOpen, run: () => setWorkbench("mesh") },
};
for (const w of WB) CMD[`wb-${w.id}`] = { label: w.label, icon: w.icon, key: w.key, enabled: docOpen, run: () => setWorkbench(w.id) };
for (const v of RESULT_VIEWS) {
  CMD[`rv-${v.id}`] = { label: v.label, icon: v.icon, enabled: docOpen,
    pressed: () => { const d = app.activeDoc(); return d.wb === "results" && d.rview === v.id; }, run: () => app.setResultsView(v.id) };
}

function enabledOf(c) {
  const e = c.enabled ? c.enabled() : true;
  return e === true ? { ok: true, why: "" } : { ok: false, why: typeof e === "string" ? e : "Not available" };
}
function runCmd(id) {
  const c = CMD[id];
  if (!c) return;
  const e = enabledOf(c);
  if (!e.ok) { status(`${c.label}: ${e.why}`); return; }
  c.run();
  renderToolbar();
}

// ---------------------------------------------------------------------------
// Menu bar
// ---------------------------------------------------------------------------
const MENUS = [
  { label: "File", items: ["new", "open", "-", "refresh", "savePNG", "-", "close"] },
  { label: "Edit", items: ["resetParams", "copyCmd"] },
  { label: "View", items: [{ head: "Theme" }, "themeSystem", "themeDark", "themeLight", "-", "fit", "focus", "-", "edges", "bnd", "contours", "stream", "mirror", "-", "combo"] },
  { label: "Model", items: WB.map((w) => `wb-${w.id}`) },
  { label: "Study", items: ["solve", "stop", "-", "editData", "copyCmd"] },
  { label: "Results", items: RESULT_VIEWS.map((v) => `rv-${v.id}`).concat(["-", "savePNG"]) },
  { label: "Tools", items: ["units", "expert", "-", "diagnose"] },
  { label: "Help", items: ["help", "about"] },
];

function buildMenus() {
  const nav = $("#menus");
  setChildren(nav, MENUS.map((m, mi) => {
    const pop = h("div.menu-pop", { role: "menu", "aria-label": m.label });
    const btn = h("button.menu-btn", { type: "button", role: "menuitem", "aria-haspopup": "true", "aria-expanded": "false", id: `menu-${mi}`, text: m.label });
    const wrap = h("div.menu", { dataset: { idx: String(mi) } }, btn, pop);
    btn.addEventListener("click", () => (wrap.classList.contains("open") ? closeMenus() : openMenu(mi, false)));
    btn.addEventListener("mouseenter", () => { if ($(".menu.open") && !wrap.classList.contains("open")) openMenu(mi, false); });
    btn.addEventListener("keydown", (e) => {
      if (["ArrowDown", "Enter", " "].includes(e.key)) { e.preventDefault(); openMenu(mi, true); }
      else if (e.key === "ArrowRight") { e.preventDefault(); focusMenu(mi + 1); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); focusMenu(mi - 1); }
      else if (e.key === "Escape") { closeMenus(); btn.blur(); }
    });
    pop.addEventListener("keydown", (e) => {
      const items = $$(".mi", pop);
      const i = items.indexOf(document.activeElement);
      if (e.key === "ArrowDown") { e.preventDefault(); items[(i + 1) % items.length].focus(); }
      else if (e.key === "ArrowUp") { e.preventDefault(); items[(i - 1 + items.length) % items.length].focus(); }
      else if (e.key === "Home") { e.preventDefault(); items[0].focus(); }
      else if (e.key === "End") { e.preventDefault(); items[items.length - 1].focus(); }
      else if (e.key === "Escape") { e.preventDefault(); closeMenus(); btn.focus(); }
      else if (e.key === "ArrowRight") { e.preventDefault(); openMenu((mi + 1) % MENUS.length, true); }
      else if (e.key === "ArrowLeft") { e.preventDefault(); openMenu((mi - 1 + MENUS.length) % MENUS.length, true); }
      else if (e.key === "Tab") closeMenus();
    });
    return wrap;
  }));
  document.addEventListener("mousedown", (e) => { if (!e.target.closest(".menu")) closeMenus(); });
}

function fillMenu(mi) {
  const m = MENUS[mi];
  const pop = $(`.menu[data-idx="${mi}"] .menu-pop`);
  setChildren(pop, m.items.map((it) => {
    if (it === "-") return h("div.mi-sep", { role: "separator" });
    if (it.head) return h("div.mi-head", { text: it.head });
    const c = CMD[it];
    const e = enabledOf(c);
    const checked = c.radio ? c.radio() : c.pressed ? c.pressed() : null;
    const lead = checked !== null ? h("span.mi-check", {}, checked ? icon("check", 14) : null) : c.icon ? icon(c.icon, 15) : h("span");
    const el = h("button.mi", {
      type: "button", role: c.radio ? "menuitemradio" : checked !== null ? "menuitemcheckbox" : "menuitem",
      "aria-checked": checked !== null ? String(!!checked) : null, "aria-disabled": e.ok ? null : "true",
      "data-tip-title": e.ok ? null : c.label, "data-tip-why": e.ok ? null : e.why, tabindex: "-1",
      onclick: () => { if (!e.ok) { status(`${c.label}: ${e.why}`); return; } closeMenus(); runCmd(it); },
    }, lead, h("span", { text: c.label }), h("span.mi-key", { text: c.key || "" }));
    return el;
  }));
}

function openMenu(mi, focusFirst) {
  closeMenus();
  hideTip();
  const wrap = $(`.menu[data-idx="${mi}"]`);
  fillMenu(mi);
  wrap.classList.add("open");
  wrap.querySelector(".menu-btn").setAttribute("aria-expanded", "true");
  if (focusFirst) {
    const first = wrap.querySelector(".mi:not([aria-disabled])") || wrap.querySelector(".mi");
    if (first) first.focus();
  }
}
function focusMenu(mi) {
  const n = MENUS.length;
  const b = $(`#menu-${(mi + n) % n}`);
  if (b) b.focus();
}
function closeMenus() {
  for (const w of $$(".menu.open")) {
    w.classList.remove("open");
    w.querySelector(".menu-btn").setAttribute("aria-expanded", "false");
  }
}

// ---------------------------------------------------------------------------
// Workbench tabs and toolbar
// ---------------------------------------------------------------------------
/** Status of one construction step for a document: {s, why}. */
function stepStatus(doc, wb) {
  if (doc.kind === "start") return { s: "none", why: "No document open" };
  const s = doc.summary;
  const st = app.statusOf(doc);
  const draft = !doc.name;
  const diag = (doc.diagnose && doc.diagnose.checks) || (s && s.mesh_checks) || [];
  const worst = (list) => (list.some((c) => c.status === "fail") ? "fail" : list.some((c) => c.status === "warn") ? "warn" : null);
  switch (wb) {
    case "geometry": return { s: "done", why: `Template: ${caseTitle(doc.case)}` };
    case "mesh":
      if (draft || !s) return { s: "none", why: "Built by gmsh when you solve" };
      if (worst(diag) === "fail") return { s: "error", why: "Mesh diagnose has failures" };
      if (worst(diag) === "warn") return { s: "warn", why: "Mesh diagnose has warnings" };
      return { s: "done", why: s.mesh ? `${fmt(s.mesh.n_elements)} elements, min quality ${fmt(s.mesh.quality_min, 2)}` : "Mesh built" };
    case "materials": return { s: "done", why: "Set by the template" };
    case "boundaries": return { s: "done", why: "Set by the template; unlisted boundaries use the natural condition" };
    case "equations": return { s: "done", why: "Set by the template" };
    case "study": {
      if (doc.invalid && Object.keys(doc.invalid).length) return { s: "error", why: "Some parameters are invalid" };
      if (!st) return { s: "todo", why: "Ready to solve" };
      if (st.state === "stale") return { s: "warn", why: "The run stopped updating; it was probably interrupted" };
      if (st.running) return { s: "running", why: `Running: ${STATE_WORD[st.state] || st.state}` };
      if (st.state === "failed") return { s: "error", why: "The run failed" };
      if (s && s.solver && s.solver.converged === false) return { s: "warn", why: "The solver did not converge" };
      return { s: "done", why: "Solved" };
    }
    case "results": {
      if (!st || st.running || st.state === "stale") return { s: "none", why: st && st.running ? "Waiting for the run to finish" : "No results yet" };
      if (st.state === "failed") return { s: "error", why: "No results: the run failed" };
      const w = worst((s && s.checks) || []);
      if (w === "fail") return { s: "error", why: "At least one check failed" };
      if (w === "warn") return { s: "warn", why: "At least one check has a warning" };
      return { s: "done", why: s ? "Results and checks available" : "Results available" };
    }
    default: return { s: "none", why: "" };
  }
}
const STEP_WORD = { done: "complete", todo: "needs input", warn: "warning", error: "error", running: "running", none: "not yet" };

function renderWbTabs() {
  const d = app.activeDoc();
  const nav = $("#wb-tabs");
  setChildren(nav, WB.map((w, i) => {
    const st = stepStatus(d, w.id);
    const sel = d.kind !== "start" && d.wb === w.id;
    return [i ? h("span.wb-arrow", { "aria-hidden": "true" }, icon("chevron-right", 12)) : null,
      h("button.wb-tab", {
        type: "button", role: "tab", "aria-selected": String(sel), "aria-disabled": d.kind === "start" ? "true" : null,
        "data-tip-title": w.label, "data-tip": d.kind === "start" ? "Open a template or a run to use this step" : `${STEP_WORD[st.s]}: ${st.why}`, "data-tip-key": w.key,
        "aria-label": `${w.label}, step ${i + 1}, ${STEP_WORD[st.s]}`,
        onclick: () => runCmd(`wb-${w.id}`),
      }, icon(w.icon, 15), w.label, h("span.sdot", { class: `s-${st.s}`, "aria-hidden": "true" }))];
  }));
}

const TOOLBARS = {
  start: ["new", "open", "refresh", "|", "themeToggle", "help"],
  geometry: ["editData", "importGeo", "gmshScript", "|", "wb-boundaries", "diagnose"],
  mesh: ["diagnose", "meshOnField", "refine"],
  materials: ["addIon", "units", "editData"],
  boundaries: ["editBC", "diagnose"],
  equations: ["editPhysics", "editData"],
  study: ["solve", "stop", "|", "editData", "resetParams", "copyCmd", "|", "rv-log"],
  results: ["rv-field", "rv-profiles", "rv-checks", "rv-mesh", "rv-figures", "rv-log", "|", "fit", "focus", "edges", "bnd", "contours"],
};

function renderToolbar() {
  const d = app.activeDoc();
  const ids = d.kind === "start" ? TOOLBARS.start : TOOLBARS[d.wb] || [];
  const tools = ids.map((id) => {
    if (id === "|") return h("span.tool-sep", { "aria-hidden": "true" });
    const c = CMD[id];
    const e = enabledOf(c);
    const pressed = c.pressed ? c.pressed() : null;
    const isTheme = id === "themeToggle";
    return h("button.tool", {
      type: "button", class: c.primary ? "primary" : "", "aria-label": c.label,
      "aria-pressed": pressed === null ? null : String(!!pressed), "aria-disabled": e.ok ? null : "true",
      "data-tip-title": c.label, "data-tip-key": c.key || null, "data-tip-why": e.ok ? null : e.why,
      onclick: () => runCmd(id),
    }, icon(isTheme ? (resolvedTheme() === "dark" ? "sun" : "moon") : c.icon, 18), c.primary ? c.label : null);
  });
  // Solve is the primary action of every document; keep it at the right end of the row.
  const right = [];
  if (d.kind !== "start" && !ids.includes("solve")) {
    right.push(h("span", { style: { flex: "1" } }));
    const c = CMD.solve;
    const e = enabledOf(c);
    right.push(h("button.tool.primary", { type: "button", "aria-label": c.label, "aria-disabled": e.ok ? null : "true",
      "data-tip-title": c.label, "data-tip-key": c.key, "data-tip-why": e.ok ? null : e.why, onclick: () => runCmd("solve") }, icon("play", 14), "Solve"));
  } else right.push(h("span", { style: { flex: "1" } }));
  const label = d.kind === "start" ? "Start" : `${WB.find((w) => w.id === d.wb).label} tools`;
  setChildren($("#toolbar"), tools, right, h("span.tool-label", { text: label }));
}

// ---------------------------------------------------------------------------
// Model tree
// ---------------------------------------------------------------------------
function treeData(d) {
  if (d.kind === "start") {
    return WB.map((w) => ({ id: w.id, label: w.label, icon: w.icon, status: "none", muted: true }));
  }
  const s = d.summary || {};
  const files = (d.listing && d.listing.files) || {};
  const kids = {};
  const sym = app.symmetryOf(d);
  kids.geometry = [{ id: "g-tpl", label: caseTitle(d.case), icon: "geometry", meta: d.case, wb: "geometry" },
    sym ? { id: "g-sym", label: sym === "axisym" ? "Axisymmetric (r, z)" : "Planar 2D", icon: sym === "axisym" ? "axis" : "geometry", wb: "geometry" } : null].filter(Boolean);
  kids.mesh = s.mesh ? [{ id: "m-el", label: "Elements", icon: "mesh", meta: fmt(s.mesh.n_elements), wb: "mesh" },
    { id: "m-q", label: "Min quality", icon: "quality", meta: fmt(s.mesh.quality_min, 2), wb: "mesh" }] : [];
  const species = s.model && Array.isArray(s.model.species) ? s.model.species : null;
  kids.materials = species === null ? [] : species.length ? species.map((sp, i) => ({ id: `sp-${i}`, label: prettyName(sp.name), icon: "ion", meta: `z ${sp.z > 0 ? "+" : ""}${sp.z}`, wb: "materials" }))
    : [{ id: "sp-none", label: "No mobile ions", icon: "ion", wb: "materials", muted: true }];
  kids.boundaries = app.boundaryNames(d).map((b, i) => ({ id: `b-${b}`, label: b, swatch: boundaryColor(b, i), dashed: /axis|symmetry/.test(b), wb: "boundaries", meta: s.model && s.model.boundaries && s.model.boundaries[b] ? Object.keys(s.model.boundaries[b]).map(prettyName).join(", ") : "natural" }));
  const flow = s.model ? s.model.flow : null;
  kids.equations = s.model ? [{ id: "e-p", label: "Poisson", icon: "equations", wb: "equations" },
    species && species.length ? { id: "e-np", label: "Nernst–Planck", icon: "equations", wb: "equations" } : null,
    flow && flow !== "none" ? { id: "e-f", label: flow === "navier_stokes" ? "Navier–Stokes" : "Stokes", icon: "flow", wb: "equations" } : null].filter(Boolean) : [];
  const st = app.statusOf(d);
  kids.study = [{ id: "st-run", label: d.name ? d.name : "Not solved yet", icon: d.name ? "play" : "study", wb: "study", meta: st ? (st.running ? STATE_WORD[st.state] : runStatusWord(app.entryOf(d.name) || { state: st.state }).text) : "", muted: !d.name }];
  const nChecks = Array.isArray(s.checks) ? s.checks.length : 0;
  kids.results = d.name ? RESULT_VIEWS.map((v) => ({ id: `rv-${v.id}`, label: v.label, icon: v.icon, wb: "results", rview: v.id,
    meta: v.id === "checks" && nChecks ? String(nChecks) : v.id === "figures" && d.listing ? String((d.listing.figures || []).length + (d.listing.plots || []).length) : "",
    muted: (v.id === "field" && !files["fields.json"]) || (v.id === "profiles" && !files["profiles.json"]) })) : [];
  return WB.map((w) => {
    const ss = stepStatus(d, w.id);
    return { id: w.id, label: w.label, icon: w.icon, status: ss.s, why: ss.why, wb: w.id, children: kids[w.id] || [] };
  });
}

function renderTree() {
  const d = app.activeDoc();
  const root = $("#tree");
  $("#combo-doc").textContent = d.kind === "start" ? "" : d.name || `new ${d.case}`;
  const exp = d.expanded || (d.expanded = new Set(["boundaries", "results", "materials"]));
  const sel = d.kind === "start" ? null : d.treeSel || d.wb;
  const nodes = [];
  for (const n of treeData(d)) {
    const hasKids = n.children && n.children.length;
    const open = hasKids && exp.has(n.id);
    nodes.push(treeNode(d, n, 0, hasKids, open, sel));
    if (open) for (const k of n.children) nodes.push(treeNode(d, k, 1, false, false, sel));
  }
  const note = d.kind === "start" ? h("div.tree-note", { text: "These are the construction steps. Open a template or a run from the Start page to fill them in." }) : null;
  setChildren(root, nodes, note);
  const items = $$(".tn", root);
  const cur = items.find((x) => x.getAttribute("aria-selected") === "true") || items[0];
  if (cur) cur.tabIndex = 0;
}

function treeNode(d, n, depth, hasKids, open, sel) {
  const isSel = sel === n.id || (depth === 0 && sel === n.wb && !d.treeSel);
  const el = h("div.tn", {
    role: "treeitem", tabindex: "-1", "aria-level": String(depth + 1), "aria-selected": String(!!isSel),
    "aria-expanded": hasKids ? String(open) : null, class: n.muted ? "muted" : "", style: { "--depth": depth },
    "data-id": n.id, "data-tip": n.why || null, "data-tip-title": depth === 0 ? n.label : null,
  },
  h("span.tn-tw", { class: hasKids ? "" : "leaf", onclick: (e) => { e.stopPropagation(); toggleNode(d, n.id); } }, icon("chevron-right", 12)),
  n.swatch ? h("span.swatch", { class: n.dashed ? "dashed" : "", style: n.dashed ? { borderColor: n.swatch, width: "12px" } : { background: n.swatch, width: "12px" } }) : icon(n.icon || "list", 15),
  h("span.tn-label", { text: n.label }),
  depth === 0 && n.status ? h("span.sdot", { class: `s-${n.status}`, style: { marginLeft: n.meta ? "6px" : "auto" }, "aria-label": STEP_WORD[n.status] }) : null,
  n.meta ? h("span.tn-meta", { text: n.meta }) : null);
  el.addEventListener("click", () => activateNode(d, n, depth));
  el.addEventListener("keydown", (e) => treeKey(e, d, n, hasKids, open));
  return el;
}

function toggleNode(d, id) {
  if (d.kind === "start") return;
  if (d.expanded.has(id)) d.expanded.delete(id); else d.expanded.add(id);
  renderTree();
  const el = $(`.tn[data-id="${CSS.escape(id)}"]`);
  if (el) el.focus();
}

function activateNode(d, n, depth) {
  if (d.kind === "start") { status("Open a template or a run first"); return; }
  d.treeSel = depth ? n.id : null;
  if (n.rview) app.setResultsView(n.rview);
  else if (n.wb) setWorkbench(n.wb, { keepTree: true });
  renderTree();
  const el = $(`.tn[data-id="${CSS.escape(n.id)}"]`);
  if (el) el.focus();
}

function treeKey(e, d, n, hasKids, open) {
  const items = $$(".tn");
  const i = items.indexOf(e.currentTarget);
  const go = (j) => { const t = items[Math.max(0, Math.min(items.length - 1, j))]; if (t) { items.forEach((x) => { x.tabIndex = -1; }); t.tabIndex = 0; t.focus(); } };
  if (e.key === "ArrowDown") { e.preventDefault(); go(i + 1); }
  else if (e.key === "ArrowUp") { e.preventDefault(); go(i - 1); }
  else if (e.key === "Home") { e.preventDefault(); go(0); }
  else if (e.key === "End") { e.preventDefault(); go(items.length - 1); }
  else if (e.key === "ArrowRight" && hasKids && !open) { e.preventDefault(); toggleNode(d, n.id); }
  else if (e.key === "ArrowLeft" && hasKids && open) { e.preventDefault(); toggleNode(d, n.id); }
  else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activateNode(d, n, Number(e.currentTarget.getAttribute("aria-level")) - 1); }
}

// ---------------------------------------------------------------------------
// Property editor (View / Data)
// ---------------------------------------------------------------------------
app.setPropsTab = (tab) => {
  S.ptab = tab;
  store.set("ptab", tab);
  renderProps();
  syncURL();
};

function prow(name, control, { changed = false, info = "", plain = true, err = "" } = {}) {
  return h("div.prow", { class: changed ? "changed" : "" },
    h("label.pname", { class: plain ? "plain" : "" }, changed ? h("span.chg-dot", { title: "changed from the template value" }) : null, h("span", { text: name })),
    h("div.pval", {}, control),
    info ? h("div.pinfo", { text: info }) : null,
    err ? h("div.perr", { role: "alert", text: err }) : null);
}

function renderProps() {
  for (const b of $$(".ptab")) b.setAttribute("aria-selected", String(b.dataset.tab === S.ptab));
  const d = app.activeDoc();
  const body = $("#props-body");
  $("#props-title").textContent = S.ptab === "data" ? "Parameters" : "Display";
  $("#props-sub").textContent = d.kind === "start" ? "" : caseTitle(d.case);
  if (d.kind === "start") {
    setChildren(body, h("div.pempty", { text: S.ptab === "data" ? "Open a template or a run to see and edit its parameters here." : "Display settings for the field map appear here when a run's results are open." }));
    return;
  }
  if (S.ptab === "data") renderDataTab(d, body);
  else renderViewTab(d, body);
}

function parseParam(raw, dflt) {
  if (typeof dflt === "boolean") return { ok: true, v: raw };
  if (typeof dflt === "string") return { ok: true, v: raw };
  const t = String(raw).trim();
  let v;
  try { v = JSON.parse(t); } catch { v = NaN; }
  if (typeof dflt === "number" || dflt === undefined) {
    if (typeof v === "number" && Number.isFinite(v)) return { ok: true, v };
    return { ok: false, err: "Enter a number, for example 0.1 or 2.5e-3" };
  }
  return t ? { ok: true, v: Number.isNaN(v) ? t : v } : { ok: false, err: "Enter a value" };
}

function renderDataTab(d, body) {
  const rows = app.paramRows(d);
  if (!S.casesLoaded && !rows.length) { setChildren(body, h("div.pempty", {}, h("div.spinner"), " Loading the template parameters…")); return; }
  d.invalid = d.invalid || {};
  const nChanged = rows.filter((r) => r.changed).length;
  const items = rows.map((r) => {
    let control;
    if (typeof r.dflt === "boolean" || typeof r.value === "boolean") {
      control = h("label.chk", {}, h("input", { type: "checkbox", checked: !!r.value, "aria-label": r.key, onchange: (e) => setParam(d, r.key, e.target.checked) }), h("span", { text: r.value ? "on" : "off" }));
    } else {
      const inp = h("input.inp", { type: "text", spellcheck: "false", inputmode: typeof r.dflt === "number" ? "decimal" : null, id: `p-${r.key}`,
        value: d.invalid[r.key] !== undefined ? d.invalid[r.key].raw : typeof r.value === "string" ? r.value : JSON.stringify(r.value),
        class: [r.unit ? "has-unit" : "", r.changed ? "changed" : "", d.invalid[r.key] ? "invalid" : ""].join(" "),
        "aria-invalid": d.invalid[r.key] ? "true" : null, "aria-describedby": d.invalid[r.key] ? `perr-${r.key}` : null,
        onchange: (e) => setParam(d, r.key, e.target.value, true),
        onkeydown: (e) => { if (e.key === "Enter") e.target.blur(); if (e.key === "Escape") { e.target.value = JSON.stringify(r.value); e.target.blur(); } } });
      control = h("span.field-wrap", {}, inp, r.unit ? h("span.unit", { text: r.unit }) : null);
    }
    const reset = r.changed || r.edited ? h("button.icon-btn", { type: "button", "aria-label": `Reset ${r.key} to ${fmt(r.dflt)}`, "data-tip-title": `Reset to the template value ${fmt(r.dflt)}`, onclick: () => resetParam(d, r.key) }, icon("reset", 14)) : null;
    const row = prow(r.key, [control, reset], { changed: r.changed, plain: false, info: S.expert ? "" : stripUnit(r.info), err: d.invalid[r.key] ? d.invalid[r.key].err : "" });
    row.querySelector(".pname").setAttribute("for", `p-${r.key}`);
    if (S.expert && r.info) row.querySelector(".pname").dataset.tip = r.info;
    const perr = row.querySelector(".perr");
    if (perr) perr.id = `perr-${r.key}`;
    return row;
  });
  const derived = d.summary && d.summary.params ? Object.entries(d.summary.params) : [];
  setChildren(body,
    h("div.pgroup", {}, h("span", { text: d.name ? `Inputs of ${d.name}` : "Template parameters" }),
      nChanged ? h("span.badge.b-warn", { style: { height: "18px" } }, `${nChanged} changed`) : null,
      h("button.btn-link", { type: "button", disabled: !Object.keys(d.edits).length || null, onclick: () => runCmd("resetParams") }, "Reset all")),
    rows.length ? items : h("div.pempty", { text: S.casesError ? `Template parameters unavailable: ${S.casesError}` : "This template has no parameters." }),
    S.expert && derived.length ? [h("div.pgroup", { text: "Model parameters (dimensionless, from the run)" }),
      derived.map(([k, v]) => prow(k, h("input.inp", { readonly: true, value: fmt(v, 6), "aria-label": k }), { plain: false }))] : null,
    h("div.pnote", {}, S.units === "SI" ? "Values are shown as the template defines them; units in the fields come from the template's help text." : "Dimensionless display is a preview: conversion arrives with the project file. Values are shown as the template defines them."));
}

function setParam(d, k, raw, fromText = false) {
  const r = app.paramRows(d).find((x) => x.key === k);
  const p = fromText ? parseParam(raw, r ? r.dflt : undefined) : { ok: true, v: raw };
  if (!p.ok) {
    d.invalid[k] = { raw, err: p.err };
  } else {
    delete d.invalid[k];
    const base = d.summary && d.summary.input && k in d.summary.input ? d.summary.input[k] : r ? r.dflt : undefined;
    if (sameValue(p.v, base)) delete d.edits[k]; else d.edits[k] = p.v;
  }
  renderProps();
  renderWbTabs();
  renderCanvasIf(["study"]);
  renderToolbar();
  const el = $(`#p-${CSS.escape(k)}`);
  if (el && !p.ok) el.focus();
}

function resetParam(d, k) {
  const r = app.paramRows(d).find((x) => x.key === k);
  delete d.invalid[k];
  if (r && r.dflt !== undefined) {
    const base = d.summary && d.summary.input && k in d.summary.input ? d.summary.input[k] : r.dflt;
    if (sameValue(base, r.dflt)) delete d.edits[k]; else d.edits[k] = r.dflt;
  } else delete d.edits[k];
  renderProps();
  renderCanvasIf(["study"]);
  renderToolbar();
}

function renderViewTab(d, body) {
  const fv = app.field;
  if (!(fv.mesh && fv.docId === d.id)) {
    setChildren(body, h("div.pempty", {}, d.name ? "These settings control the field map. " : "Solve this project to get a field map. ",
      d.name ? h("button.btn-link", { type: "button", onclick: () => app.setResultsView("field") }, "Open Results, Field map") : null));
    return;
  }
  const o = fv.opts;
  const name = o.field;
  const sel = (items, value, onchange, label) => h("select.sel", { onchange: (e) => onchange(e.target.value), "aria-label": label },
    items.map(([v, t]) => h("option", { value: v, text: t, selected: v === value || null })));
  const chk = (checked, onchange, label, disabled, why) => h("label.chk", { class: disabled ? "disabled" : "", "data-tip": disabled ? why : null },
    h("input", { type: "checkbox", checked: !!checked || null, disabled: disabled || null, onchange: (e) => onchange(e.target.checked) }), h("span", { text: label }));
  const fields = fv.fieldNames.map((n) => [n, `${prettyName(n)}${fv.mesh.labels[n] ? ` · ${fv.mesh.labels[n]}` : ""}`]);
  const maps = colormapNames().map((n) => [n, `${n} (${colormapType(n)})`]);
  const dr = fv.displayRange;
  const lo = h("input.inp", { type: "text", value: fmt(dr.lo, 4), "aria-label": "Range minimum", inputmode: "decimal" });
  const hi = h("input.inp", { type: "text", value: fmt(dr.hi, 4), "aria-label": "Range maximum", inputmode: "decimal" });
  const applyRange = () => { if (!fv.setManualRange(parseFloat(lo.value), parseFloat(hi.value))) { lo.classList.add("invalid"); hi.classList.add("invalid"); status("Range: enter two numbers (positive for log scale)"); } };
  for (const x of [lo, hi]) { x.addEventListener("change", applyRange); x.addEventListener("keydown", (e) => e.key === "Enter" && applyRange()); }
  const vecs = [["none", "none"], ...Object.keys(fv.mesh.vectors).map((v) => [v, `arrows of ${prettyName(v)}`])];
  const src = colormapSource();
  setChildren(body,
    h("div.pgroup", { text: "Field" }),
    prow("Field", sel(fields, name, (v) => fv.selectField(v), "Field")),
    prow("Colormap", sel(maps, fv.cmapName, (v) => fv.setColormap(v), "Colormap")),
    prow("", chk(fv.reversed, (v) => fv.setReversed(v), "Reverse colormap")),
    src.fallback ? h("div.pnote", { text: `Using built-in colormaps: ${src.reason}.` }) : null,
    h("div.pgroup", { text: "Range" }),
    prow("Mode", sel([["auto", "automatic, all nodes"], ["view", "automatic, visible area"], ["manual", "manual"]], fv.range.mode, (v) => fv.setRangeMode(v), "Range mode")),
    prow("Minimum", lo), prow("Maximum", hi),
    prow("", chk(fv.isLog, (v) => fv.setLog(v), "Logarithmic scale", !fv.canLog, "Only for fields with positive values")),
    h("div.pgroup", { text: "Overlays" }),
    prow("", chk(o.edges, (v) => fv.setOpt("edges", v), "Mesh edges  (E)")),
    prow("", chk(o.bnd, (v) => fv.setOpt("bnd", v), "Boundaries  (B)")),
    prow("Contours", sel([["0", "off"], ["5", "5 levels"], ["10", "10 levels"], ["20", "20 levels"]], String(o.contours || 0), (v) => fv.setOpt("contours", Number(v)), "Contours")),
    prow("", chk(o.axes, (v) => fv.setOpt("axes", v), "Axis ticks")),
    prow("Vectors", sel(vecs, o.vectors, (v) => fv.setOpt("vectors", v), "Vector arrows")),
    prow("", chk(o.stream, (v) => fv.setOpt("stream", v), "Streamlines of u  (S)", !fv.mesh.vectors.u, "This run has no velocity field")),
    fv.mesh.axisym ? [h("div.pgroup", { text: "Symmetry" }), prow("", chk(o.mirror, (v) => setMirror(v), "Mirror about r = 0  (M)"))] : null);
}

function setMirror(on) {
  app.field.opts.mirror = !!on;
  app.field.persist();
  const d = app.activeDoc();
  if (d.fieldsJSON) {
    d.mesh = null;
    renderCanvas();
  }
  renderToolbar();
  status(on ? "Showing the full cross-section (mirrored about the axis)" : "Showing the computed half (r ≥ 0)");
}

// ---------------------------------------------------------------------------
// Documents
// ---------------------------------------------------------------------------
function newDoc(fields) {
  return Object.assign({ id: `doc${++S.docSeq}`, kind: "run", name: null, case: null, wb: "results", rview: "field", edits: {}, invalid: {} }, fields);
}

app.openRun = async (name, { wb = null, rview = null } = {}) => {
  let d = S.docs.find((x) => x.name === name);
  if (!d) {
    const e = app.entryOf(name);
    d = newDoc({ kind: "run", name, case: e && e.case, wb: wb || (e && RUNNING.has(e.state) ? "study" : "results"), rview: rview || "field" });
    S.docs.push(d);
    activate(d.id);
    await loadRunData(d);
  } else {
    if (wb) d.wb = wb;
    if (rview) d.rview = rview;
    activate(d.id);
  }
};

app.openTemplate = (caseName) => {
  const d = newDoc({ kind: "draft", case: caseName, wb: "study" });
  d.nextName = app.suggestName(d);
  S.docs.push(d);
  activate(d.id);
  status(`New project from the template ${caseTitle(caseName)}. Check the parameters in the Data tab, then press Solve (F8).`);
  if (S.ptab !== "data") app.setPropsTab("data");
};

function activate(id) {
  S.active = id;
  hideTip();
  renderAll();
  syncURL();
}

function closeDoc(id) {
  if (id === "start") return;
  const i = S.docs.findIndex((d) => d.id === id);
  if (i < 0) return;
  S.docs.splice(i, 1);
  if (S.active === id) S.active = (S.docs[i] || S.docs[i - 1] || S.docs[0]).id;
  if (app.field.docId === id) { app.field.docId = null; app.field.mesh = null; }
  renderAll();
  syncURL();
}

async function loadRunData(d, { quiet = false } = {}) {
  if (!d.name) return;
  if (!quiet) { d.loading = true; if (S.active === d.id) renderCanvas(); }
  try {
    const listing = await bridge.listing(d.name);
    const has = (f) => listing.files && listing.files[f];
    const [summary, progress, profiles, diagnose] = await Promise.all([
      has("summary.json") ? optional(bridge.json(d.name, "summary.json")) : null,
      has("progress.json") ? optional(bridge.json(d.name, "progress.json")) : null,
      has("profiles.json") ? optional(bridge.json(d.name, "profiles.json")) : null,
      (listing.mesh_files || []).includes("mesh/diagnose.json") ? optional(bridge.json(d.name, "mesh/diagnose.json")) : null,
    ]);
    const fStamp = has("fields.json") ? `${listing.files["fields.json"].mtime}:${listing.files["fields.json"].size}` : null;
    if (fStamp !== d.fieldsStamp) { d.timeJSON = null; d.timePromise = null; d.fieldsJSON = null; d.mesh = null; d.fieldsStamp = fStamp; if (app.field.docId === d.id) app.field.docId = null; }
    Object.assign(d, { listing, summary, progress, profiles, diagnose, error: null });
    if (!d.case && summary && summary.case) d.case = summary.case;
    if (!d.case && listing.entry && listing.entry.case) d.case = listing.entry.case;
    if (!d.case) { const e = app.entryOf(d.name); if (e && e.case) d.case = e.case; }
  } catch (e) {
    d.error = e.status === 404 ? `The run '${d.name}' was not found in the results folder.` : e.message;
  }
  d.loading = false;
  if (S.active === d.id) renderAll();
}

app.meshKey = (d) => `${d.fieldsStamp}|${app.field.opts.mirror ? 1 : 0}`;

app.loadMesh = async (d, onProgress) => {
  if (d.mesh && d.meshKey === app.meshKey(d)) return d.mesh;
  if (!d.fieldsJSON) {
    if (!d.fieldsPromise) {
      d.fieldsPromise = bridge.jsonProgress(d.name, "fields.json", onProgress).finally(() => { d.fieldsPromise = null; });
    }
    try {
      d.fieldsJSON = await d.fieldsPromise;
    } catch (e) {
      if (e.status === 404) return null;
      throw e;
    }
  }
  d.mesh = prepareMesh(d.fieldsJSON, { mirror: app.field.opts.mirror });
  d.meshKey = app.meshKey(d);
  renderTree();
  return d.mesh;
};

/** Time series of the primary fields for the time slider, or null (steady run / old run without the file). */
app.loadTimeSeries = async (d, mesh, onProgress) => {
  if (!d.listing || !d.listing.files || !d.listing.files["fields_time.json"]) return null;
  if (!d.timeJSON) {
    d.timePromise ||= bridge.jsonProgress(d.name, "fields_time.json", onProgress).finally(() => { d.timePromise = null; });
    try { d.timeJSON = await d.timePromise; } catch { return null; }
  }
  return prepareTimeFields(d.timeJSON, mesh);
};

// ---------------------------------------------------------------------------
// Workbench switching and canvas
// ---------------------------------------------------------------------------
function setWorkbench(wb, { keepTree = false } = {}) {
  const d = app.activeDoc();
  if (d.kind === "start") return;
  d.wb = wb;
  if (!keepTree) d.treeSel = null;
  renderAll();
  syncURL();
}
app.setWorkbench = setWorkbench;

app.setResultsView = (v) => {
  const d = app.activeDoc();
  if (d.kind === "start") return;
  d.wb = "results";
  d.rview = v;
  d.treeSel = `rv-${v}`;
  if (d.expanded) d.expanded.add("results");
  renderAll();
  syncURL();
};

function renderCanvasIf(wbs) {
  const d = app.activeDoc();
  if (d.kind !== "start" && wbs.includes(d.wb)) renderCanvas();
}

function renderCanvas() {
  const d = app.activeDoc();
  const root = $("#canvas");
  if (app.fieldHost.parentNode) app.fieldHost.remove();
  if (d.kind === "start") { S.startSig = startSignature(); renderStart(app, root); return; }
  if (d.loading && !d.listing) { setChildren(root, h("div.page", {}, placeholder("loading", `Opening ${d.name}…`))); return; }
  if (d.error) {
    setChildren(root, h("div.page", {}, placeholder("error", "Could not open the run", d.error,
      h("button.btn.sm", { type: "button", onclick: () => loadRunData(d) }, icon("refresh", 14), "Try again"))));
    return;
  }
  const r = { geometry: renderGeometry, mesh: renderMesh, materials: renderMaterials, boundaries: renderBoundaries, equations: renderEquations, study: renderStudy, results: renderResults }[d.wb];
  r(app, d, root);
  requestAnimationFrame(() => resizeAll(root));
}
app.renderCanvas = renderCanvas;

function renderDocTabs() {
  const bar = $("#doc-tabs");
  setChildren(bar, S.docs.map((d) => {
    const sel = d.id === S.active;
    const label = d.kind === "start" ? "Start" : d.name || `new ${d.case}`;
    const st = d.kind === "start" ? null : app.statusOf(d);
    const tab = h("div.dtab", {
      role: "tab", tabindex: sel ? "0" : "-1", "aria-selected": String(sel), class: d.kind === "start" ? "start" : "",
      "data-tip-title": d.kind === "start" ? "Start page" : label, "data-tip": d.kind === "start" ? "Templates, recent runs and examples" : d.name ? `Run of ${caseTitle(d.case)}` : `Unsaved project from ${caseTitle(d.case)}`,
      onclick: (e) => { if (!e.target.closest(".icon-btn")) activate(d.id); },
      onkeydown: (e) => {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); activate(d.id); }
        else if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
          const i = S.docs.indexOf(d) + (e.key === "ArrowRight" ? 1 : -1);
          const n = S.docs[(i + S.docs.length) % S.docs.length];
          activate(n.id);
          requestAnimationFrame(() => { const t = $('.dtab[aria-selected="true"]'); if (t) t.focus(); });
        } else if (e.key === "Delete" && d.kind !== "start") closeDoc(d.id);
      },
    }, d.kind === "start" ? icon("inbox", 14) : st && st.running ? h("span.sdot.s-running") : icon(d.name ? "results" : "file-plus", 14),
    h("span.dt-name", { text: label }),
    d.kind === "start" ? null : h("button.icon-btn", { type: "button", "aria-label": `Close ${label}`, tabindex: "-1", onclick: () => closeDoc(d.id) }, icon("x", 12)));
    return tab;
  }));
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------
function setUnits(u) {
  S.units = u;
  store.set("units", u);
  renderStatusbar();
  renderProps();
  status(u === "SI" ? "Units: SI (lab units)" : "Units: dimensionless. This is a preview; conversion arrives with the project file.");
}
function setExpert(on) {
  S.expert = !!on;
  store.set("expert", S.expert);
  renderStatusbar();
  renderProps();
  renderCanvasIf(["study"]);
  status(on ? "Expert mode: every setting, raw names and the command line are shown" : "Beginner mode: explanations are shown next to the settings");
}

function renderStatusbar() {
  const d = app.activeDoc();
  let doc = d.kind !== "start" && d.name ? d : null;
  let st = doc ? app.statusOf(doc) : null;
  let name = doc ? doc.name : null;
  if (!doc) {
    const r = S.runs.find((x) => RUNNING.has(x.state) && !isStale(x));
    if (r) { name = r.name; st = runStatus(r.progress, r); }
  }
  const run = $("#sb-run");
  if (!st) setChildren(run, h("span.dim", { text: d.kind === "start" ? "No run in progress" : "Not solved yet" }));
  else {
    setChildren(run,
      h("span.sb-name", { text: name, title: name }),
      stageBars(st),
      healthBadge(st),
      h("span.sb-res", { "data-tip-title": "Last residual", "data-tip": "Norm of the nonlinear residual after the last Newton iteration" }, "‖R‖ ", h("b", { text: st.resNorm !== null ? fmt(st.resNorm, 2) : "–" })));
  }
  const seg = (items, value, onPick, label) => h("span.seg", { role: "group", "aria-label": label },
    items.map(([v, t, tipText]) => h("button", { type: "button", "aria-pressed": String(v === value), "data-tip": tipText, onclick: () => onPick(v) }, t)));
  setChildren($("#sb-controls"),
    seg([["SI", "SI", "Lab units [U]"], ["dimensionless", "dimensionless", "Scaled units [U]"]], S.units, setUnits, "Units"),
    h("span.stub-tag", { "data-tip": "Visual stub: the unit toggle does not convert values yet. Conversion arrives with the project file." }, "preview"),
    h("span.sb-sep"),
    seg([[false, "Beginner", "Explanations next to every setting [X]"], [true, "Expert", "Every setting, raw names, command line [X]"]], S.expert, setExpert, "Experience level"));
}

// ---------------------------------------------------------------------------
// Help / about
// ---------------------------------------------------------------------------
function dialogShell(dlg, id, title, body) {
  setChildren(dlg, h("div.dlg-head", {}, h("h2", { id, text: title }),
    h("button.icon-btn", { type: "button", "aria-label": "Close", onclick: () => dlg.close() }, icon("x", 16))), h("div.dlg-body", {}, body));
  dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
}
function openHelp() {
  const dlg = $("#help-dialog");
  const row = (label, ...keys) => h("div.keyrow", {}, h("span", { text: label }), h("span", {}, keys.map((k) => h("kbd", { text: k }))));
  dialogShell(dlg, "help-title", "Keyboard shortcuts", h("div.keys", {},
    h("h3", { text: "Application" }),
    row("New project from template", "N"), row("Open run", "O"), row("Refresh runs", "R"), row("Filter runs (Start page)", "/"),
    row("Next / previous document", "J", "K"), row("Switch dark / light theme", "T"), row("Units SI / dimensionless", "U"), row("Beginner / expert", "X"),
    row("Show / hide the model panel", "["), row("Menu bar", "F10"), row("This help", "?"), row("Close menus and dialogs", "Esc"),
    h("h3", { text: "Construction steps" }),
    ...WB.map((w) => row(w.label, w.key)),
    row("Solve", "F8"), row("Stop the run", "Shift+F8"),
    h("h3", { text: "Field map" }),
    row("Fit all / zoom to focus", "F", "G"), row("Mesh edges / boundaries", "E", "B"), row("Contours / streamlines", "C", "S"),
    row("Mirror axisymmetric runs", "M"), row("Save view as PNG", "P"), row("Pan / zoom (canvas focused)", "Arrows", "+", "-"),
    row("Probe all fields at a point", "Click"), row("Zoom to focus", "Double-click")));
  dlg.showModal();
}
function openAbout() {
  const dlg = $("#about-dialog");
  dialogShell(dlg, "about-title", "About pnpflow", h("div", {},
    h("p", { text: "pnpflow: finite elements for Poisson–Nernst–Planck ion transport with optional Stokes / Navier–Stokes flow. Jacobians and sensitivities come from JAX automatic differentiation." }),
    h("p.dim", { text: `Front end v0.1 (first version). ${inTauri ? "Desktop shell (Tauri 2)." : "Running in a browser against the development server."} Results folder: ${S.resultsDir || "unknown"}.` }),
    h("p.dim", { text: "Geometry, materials, boundary and physics editing arrive with the project file; today a project is a template plus parameters." })));
  dlg.showModal();
}

// ---------------------------------------------------------------------------
// Solve / stop
// ---------------------------------------------------------------------------
app.solve = async (d, { name, overwrite, msgEl } = {}) => {
  const say = (kind, text) => {
    if (msgEl && msgEl.isConnected) setChildren(msgEl, h(`div.msg.msg-${kind}`, {}, icon(kind === "error" ? "x-circle" : kind === "ok" ? "check-circle" : "info", 14), h("span", { text })));
    else app.toast(text, kind === "error" ? "error" : kind);
  };
  if (d.invalid && Object.keys(d.invalid).length) return say("error", `Fix the highlighted parameters first: ${Object.keys(d.invalid).join(", ")}.`);
  name = (name || d.nextName || app.suggestName(d)).trim();
  if (!/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/.test(name)) return say("error", "Run names use letters, digits, '_', '.' and '-' only, and start with a letter, digit or '_'.");
  say("info", `Starting ${name}…`);
  try {
    await bridge.launch({ case: d.case, name, params: changedParams(d), overwrite: !!(overwrite ?? d.overwrite), device: app.device() });
  } catch (e) {
    const exists = /exists/.test(e.message);
    return say("error", exists ? `A run called '${name}' already exists. Choose another name or tick "Replace an existing run".` : `Could not start the run: ${e.message}`);
  }
  // the document now follows the new run: a draft becomes the run; a run document
  // that is solved again opens the new run in its own tab
  const target = d.name ? newDoc({ kind: "run", case: d.case }) : d;
  if (target !== d) S.docs.push(target);
  for (const other of S.docs.slice()) if (other !== target && other.name === name) closeDoc(other.id);
  Object.assign(target, { kind: "run", name, wb: "study", rview: "field", summary: null, progress: null, profiles: null, diagnose: null,
    listing: null, fieldsJSON: null, mesh: null, fieldsStamp: null, nextName: null, edits: {}, invalid: {}, wasRunning: true, error: null });
  if (target !== d) { d.edits = {}; d.invalid = {}; }
  activate(target.id);
  status(`Started ${name}`);
  await refreshRuns();
  await loadRunData(target, { quiet: true });
};

app.stop = async (d) => {
  try {
    await bridge.stop(d.name);
    app.toast(`Stop requested for ${d.name}`, "warn");
  } catch (e) {
    app.toast(`Could not stop the run: ${e.message}`, "error");
  }
};

// ---------------------------------------------------------------------------
// Polling
// ---------------------------------------------------------------------------
function startSignature() {
  return S.runs.map((r) => `${r.name}:${r.state}:${r.converged}:${r.thumb}:${r.checks}`).join("|") + `#${Object.keys(S.cases).length}`;
}

async function refreshRuns() {
  try {
    const r = await bridge.runs();
    S.runs = r.runs || [];
    S.resultsDir = r.results || S.resultsDir;
    S.runsLoaded = true;
    setConn(true);
  } catch (e) {
    setConn(false, e.message);
  }
}

function setConn(ok, why) {
  if (S.connected === ok) return;
  S.connected = ok;
  const c = $("#conn");
  c.className = `conn ${ok ? "ok" : "off"}`;
  $(".conn-text", c).textContent = ok ? (inTauri ? "Local" : "Server connected") : "Server unreachable";
  c.dataset.tip = ok ? `Results: ${S.resultsDir}` : `The app cannot reach its backend (${why || "no answer"}). Start it with: python app/server.py`;
}

async function poll() {
  if (document.hidden) return;
  await refreshRuns();
  const finished = [];
  for (const d of S.docs) {
    if (!d.name) continue;
    const e = app.entryOf(d.name);
    const running = e && RUNNING.has(e.state) && !isStale(e);
    if (running) {
      d.progress = await optional(bridge.json(d.name, "progress.json")).catch(() => d.progress);
      d.wasRunning = true;
    } else if (d.wasRunning && e) {
      d.wasRunning = false;
      finished.push(d);
    }
  }
  for (const d of finished) {
    await loadRunData(d, { quiet: true });
    const e = app.entryOf(d.name);
    const ok = e && e.state === "done";
    app.toast(ok ? `${d.name} finished${d.summary && d.summary.solver && d.summary.solver.converged === false ? " but did not converge" : ""}` : `${d.name} failed; see the log`, ok ? "ok" : "error");
    if (ok && S.active === d.id && d.wb === "study") { d.wb = "results"; d.rview = "field"; }
    if (S.active === d.id) { renderAll(); syncURL(); }
  }
  // live updates without rebuilding the canvas
  const a = app.activeDoc();
  if (a.kind === "start") { if (startSignature() !== S.startSig) renderCanvas(); }
  else if (a.wb === "study") updateStudy(app, a);
  renderStatusbar();
  renderWbTabs();
  renderDocTabs();
  if (a.kind !== "start" && app.statusOf(a) && app.statusOf(a).running) { renderTree(); renderToolbar(); }
}

async function refreshAll(user = false) {
  await refreshRuns();
  const d = app.activeDoc();
  if (d.name) await loadRunData(d, { quiet: true });
  if (user) { await app.loadCases(true); status("Runs refreshed"); }
  renderAll();
}

app.loadCases = async (force = false) => {
  if (S.casesLoaded && !force) return;
  const res = await bridge.cases().catch((e) => ({ cases: {}, error: e.message }));
  S.cases = res.cases || {};
  S.casesError = res.error;
  S.casesLoaded = true;
  renderAll();
};

// ---------------------------------------------------------------------------
// URL state: ?run=<name>&wb=<wb>&view=<results view>&new=<case>&ptab=view|data&theme=light
// ---------------------------------------------------------------------------
function syncURL() {
  const d = app.activeDoc();
  const u = new URL(location.href);
  for (const k of ["run", "wb", "view", "new", "doc"]) u.searchParams.delete(k);
  if (d.kind !== "start") {
    if (d.name) u.searchParams.set("run", d.name); else u.searchParams.set("new", d.case);
    u.searchParams.set("wb", d.wb);
    if (d.wb === "results") u.searchParams.set("view", d.rview);
  }
  u.searchParams.set("ptab", S.ptab);
  if (u.href !== location.href) history.replaceState(null, "", u);
}

async function applyURL() {
  const q = new URLSearchParams(location.search);
  if (q.get("ptab") === "view" || q.get("ptab") === "data") S.ptab = q.get("ptab");
  const wb = WB.some((w) => w.id === q.get("wb")) ? q.get("wb") : null;
  const view = RESULT_VIEWS.some((v) => v.id === q.get("view")) ? q.get("view") : null;
  if (q.get("run")) await app.openRun(q.get("run"), { wb: wb || (view ? "results" : null), rview: view });
  else if (q.get("new")) {
    app.openTemplate(q.get("new"));
    if (wb) setWorkbench(wb);
  }
}

// ---------------------------------------------------------------------------
// Keyboard
// ---------------------------------------------------------------------------
const KEYMAP = {
  n: "new", o: "open", r: "refresh", p: "savePNG", t: "themeToggle", u: "units", x: "expert", "[": "combo", "?": "help",
  f: "fit", g: "focus", e: "edges", b: "bnd", c: "contours", s: "stream", m: "mirror",
  1: "wb-geometry", 2: "wb-mesh", 3: "wb-materials", 4: "wb-boundaries", 5: "wb-equations", 6: "wb-study", 7: "wb-results",
};
function onKey(e) {
  if (e.key === "F8") { e.preventDefault(); runCmd(e.shiftKey ? "stop" : "solve"); return; }
  if (e.key === "F10") { e.preventDefault(); focusMenu(0); return; }
  if (document.querySelector("dialog[open]") || !$("#lightbox").classList.contains("hidden")) return;
  const tag = (e.target.tagName || "").toLowerCase();
  if (["input", "select", "textarea"].includes(tag) || e.target.isContentEditable || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "Escape") { closeMenus(); return; }
  if (e.target.closest && e.target.closest(".menu")) return;
  if (e.key === "/") { e.preventDefault(); runCmd("open"); return; }
  if (e.key === "j" || e.key === "k") {
    const i = S.docs.findIndex((d) => d.id === S.active) + (e.key === "j" ? 1 : -1);
    activate(S.docs[(i + S.docs.length) % S.docs.length].id);
    return;
  }
  const id = KEYMAP[e.key.length === 1 ? e.key.toLowerCase() : e.key] || KEYMAP[e.key];
  if (id) { e.preventDefault(); runCmd(id); }
}

// ---------------------------------------------------------------------------
// Splitters
// ---------------------------------------------------------------------------
function initSplitters() {
  const rootStyle = document.documentElement.style;
  const ws = $("#workspace");
  const combo = $("#combo");
  const sizes = store.get("layout", {});
  if (sizes.combo) rootStyle.setProperty("--combo-w", `${sizes.combo}px`);
  if (sizes.props) rootStyle.setProperty("--props-h", `${sizes.props}px`);
  const save = debounce(() => store.set("layout", sizes), 200);
  const after = debounce(() => { app.field.resize(); resizeAll(document); }, 60);
  const drag = (el, onMove) => {
    el.addEventListener("pointerdown", (e) => {
      e.preventDefault();
      el.setPointerCapture(e.pointerId);
      const move = (ev) => { onMove(ev); after(); };
      const up = () => { el.removeEventListener("pointermove", move); el.removeEventListener("pointerup", up); save(); };
      el.addEventListener("pointermove", move);
      el.addEventListener("pointerup", up);
    });
  };
  const setCombo = (w) => { sizes.combo = Math.round(Math.max(200, Math.min(560, w))); rootStyle.setProperty("--combo-w", `${sizes.combo}px`); };
  const setProps = (hgt) => { const max = combo.getBoundingClientRect().height - 160; sizes.props = Math.round(Math.max(120, Math.min(max, hgt))); rootStyle.setProperty("--props-h", `${sizes.props}px`); };
  drag($("#vsplit"), (ev) => setCombo(ev.clientX - ws.getBoundingClientRect().left));
  drag($("#hsplit"), (ev) => setProps(combo.getBoundingClientRect().bottom - 34 - ev.clientY));
  $("#vsplit").addEventListener("keydown", (e) => {
    const cur = $("#combo").getBoundingClientRect().width;
    if (e.key === "ArrowLeft") { setCombo(cur - 16); save(); after(); }
    if (e.key === "ArrowRight") { setCombo(cur + 16); save(); after(); }
  });
  $("#hsplit").addEventListener("keydown", (e) => {
    const cur = $("#props").getBoundingClientRect().height;
    if (e.key === "ArrowUp") { setProps(cur + 16); save(); }
    if (e.key === "ArrowDown") { setProps(cur - 16); save(); }
  });
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function renderAll() {
  renderWbTabs();
  renderToolbar();
  renderTree();
  renderProps();
  renderCanvas();
  renderDocTabs();
  renderStatusbar();
}

async function init() {
  buildMenus();
  initTooltips();
  initSplitters();
  for (const b of $$(".ptab")) b.addEventListener("click", () => app.setPropsTab(b.dataset.tab));
  $(".props-tabs").addEventListener("keydown", (e) => {
    if (e.key === "ArrowLeft" || e.key === "ArrowRight") { app.setPropsTab(S.ptab === "view" ? "data" : "view"); $(`#ptab-${S.ptab}`).focus(); }
  });
  app.fieldHost = h("div.fv-host");
  app.field = new FieldView(app.fieldHost, { onChange: () => { if (S.ptab === "view") renderProps(); renderToolbar(); } });
  document.addEventListener("keydown", onKey);
  if (mqLight) mqLight.addEventListener("change", applyTheme);
  window.addEventListener("resize", debounce(() => resizeAll(document), 150));
  S.cmSource = await initColormaps();
  renderAll();
  status("Ready");
  await refreshRuns();
  renderAll();
  await applyURL();
  app.loadCases();
  setInterval(poll, POLL_MS);
  // ?menu=<label> and ?dialog=help|about open them on load (for screenshots and links)
  const q = new URLSearchParams(location.search);
  const mi = MENUS.findIndex((m) => m.label.toLowerCase() === (q.get("menu") || "").toLowerCase());
  if (mi >= 0) setTimeout(() => openMenu(mi, true), 300);
  if (q.get("dialog") === "help") openHelp();
  if (q.get("dialog") === "about") openAbout();
  if (!window.Plotly) app.toast("Charts are unavailable: vendor/plotly.min.js did not load", "error");
  if (S.cmSource.fallback) status(`Using built-in colormaps (${S.cmSource.reason})`);
}

init();
