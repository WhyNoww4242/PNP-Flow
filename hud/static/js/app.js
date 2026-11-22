// pnpflow HUD — application shell: run list, tabs, routing, polling.

import { api, optional } from "./api.js";
import { h, $, $$, setChildren, placeholder, fmt, fmtAgo, fmtDuration, store, debounce } from "./util.js";
import { prepareMesh } from "./glfield.js";
import { FieldView } from "./fieldview.js";
import { ProfilesView } from "./profiles.js";
import { SummaryView } from "./summary.js";
import { GalleryView, LogView, LaunchView, renderProgress, RUNNING_STATES, STATE_LABEL } from "./panels.js";
import { resizeAll } from "./plots.js";

const TABS = [
  { id: "field", label: "Field", key: "1" },
  { id: "profiles", label: "Profiles", key: "2" },
  { id: "summary", label: "Summary", key: "3" },
  { id: "gallery", label: "Gallery", key: "4" },
  { id: "progress", label: "Progress & log", key: "5" },
];
const POLL_MS = 2000;

const app = {
  runs: [],
  current: null,        // name of the selected run
  tab: "summary",
  data: {},             // per-run cache: {listing, summary, progress, profiles, mesh, meshPromise}
  launching: false,     // launch view visible
  filter: "",
};

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------
function applyTheme(theme) {
  document.documentElement.dataset.theme = theme;
  store.set("theme", theme);
  $("#theme-btn").textContent = theme === "light" ? "☾" : "☀";
  $("#theme-btn").title = `Switch to ${theme === "light" ? "dark" : "light"} theme  [T]`;
  if (app.field) app.field.themeChanged();
  rerenderCharts();
}

function toggleTheme() {
  applyTheme(document.documentElement.dataset.theme === "light" ? "dark" : "light");
}

/** Plotly charts bake colors in; re-render panels that contain charts. */
function rerenderCharts() {
  const d = app.current && app.data[app.current];
  if (!d) return;
  if (d.summary !== undefined && app.tab === "summary") app.summary.render(d.summary, d.progress);
  if (app.tab === "profiles" && app.profiles.data) app.profiles.drawPlot();
  if (app.tab === "progress") renderProgress($("#progress-box"), d.progress, entryOf(app.current));
}

// ---------------------------------------------------------------------------
// Run list (sidebar)
// ---------------------------------------------------------------------------
function entryOf(name) {
  return app.runs.find((r) => r.name === name) || null;
}

function stateBadge(entry) {
  const st = entry.state || "unknown";
  const running = RUNNING_STATES.has(st);
  const label = STATE_LABEL[st] || st;
  return h(`span.badge.s-${st}`, { title: entry.progress && entry.progress.message ? entry.progress.message : label },
    running ? h("span.pulse") : null, label);
}

function runItem(r) {
  const p = r.progress || {};
  const running = RUNNING_STATES.has(r.state);
  const meta = [];
  if (r.case) meta.push(r.case);
  if (r.n_elements) meta.push(`${(r.n_elements / 1000).toFixed(r.n_elements < 1e4 ? 1 : 0)}k el`);
  if (r.time_s) meta.push(fmtDuration(r.time_s));
  const progressLine = running ? h("div.ri-progress", {},
    h("div.ri-bar", {}, h("div.ri-fill", {
      style: { width: p.n_stages ? `${Math.min(100, ((p.stage || 0) - 0.5) / p.n_stages * 100)}%` : "100%" },
      class: p.n_stages ? "" : "anim",
    })),
    h("span.mono", { text: `${p.n_stages ? `st ${p.stage}/${p.n_stages} · ` : ""}it ${p.iteration ?? "–"} · ‖R‖ ${p.res_norm !== undefined ? fmt(p.res_norm, 2) : "–"}` }))
    : null;
  return h("li.run-item", {
    class: (r.name === app.current ? "active " : "") + (running ? "running" : ""),
    dataset: { name: r.name }, tabindex: "0", title: r.name,
    onclick: () => selectRun(r.name),
    onkeydown: (e) => e.key === "Enter" && selectRun(r.name),
  },
  h("div.ri-top", {}, h("span.ri-name", { text: r.name }), stateBadge(r)),
  h("div.ri-meta", {},
    h("span", { text: meta.join(" · ") }),
    r.checks === "fail" || r.checks === "warn"
      ? h(`span.ri-check.c-${r.checks}`, { title: `at least one check is '${r.checks}'`, text: r.checks === "fail" ? "✕ check" : "! check" }) : null,
    h("span.ri-time", { text: fmtAgo(r.mtime) })),
  progressLine);
}

function renderRunList() {
  const list = $("#run-list");
  const f = app.filter.toLowerCase();
  const runs = app.runs.filter((r) => !f || r.name.toLowerCase().includes(f) || (r.case || "").toLowerCase().includes(f));
  $("#run-count").textContent = app.runs.length ? `${app.runs.length}` : "";
  if (!app.runs.length) {
    return setChildren(list, h("li.run-empty", {}, placeholder("empty", "No runs yet",
      h("span", {}, "Results appear here from ", h("code", { text: "results/" }), ". Start one with ", h("b", { text: "New run" }), " or generate a sample with ", h("code", { text: "python hud/make_sample.py" }), "."))));
  }
  if (!runs.length) return setChildren(list, h("li.run-empty.dim.small", { text: `No runs match “${app.filter}”.` }));
  setChildren(list, runs.map(runItem));
}

async function refreshRuns() {
  try {
    const r = await api.runs();
    const before = new Map(app.runs.map((x) => [x.name, x.state]));
    app.runs = r.runs || [];
    $("#conn").classList.remove("offline");
    $("#conn").title = `connected · results: ${r.results}`;
    renderRunList();
    // auto-select the newest run on first load
    if (!app.current && !app.launching && app.runs.length) selectRun(app.runs[0].name, { replace: true });
    if (app.current) onCurrentRunPolled(before.get(app.current));
    updateHeader();
  } catch (e) {
    $("#conn").classList.add("offline");
    $("#conn").title = `server unreachable: ${e.message}`;
  }
}

/** Called after each poll; refreshes live data of the selected run. */
async function onCurrentRunPolled(prevState) {
  const entry = entryOf(app.current);
  if (!entry) return;
  const d = app.data[app.current];
  if (!d) return;
  const running = RUNNING_STATES.has(entry.state);
  const finished = prevState && RUNNING_STATES.has(prevState) && !running;
  if (running || finished) {
    d.progress = await optional(api.file(app.current, "progress.json")).catch(() => d.progress);
    if (app.tab === "progress") {
      renderProgress($("#progress-box"), d.progress, entry);
      app.log.refresh();
    }
  }
  if (finished) {
    toast(`Run '${app.current}' ${entry.state}`, entry.state === "done" ? "ok" : "error");
    delete app.data[app.current];
    await loadRun(app.current);
  }
}

// ---------------------------------------------------------------------------
// Selecting and loading a run
// ---------------------------------------------------------------------------
function selectRun(name, { replace = false, tab = null } = {}) {
  app.launching = false;
  const changed = name !== app.current;
  app.current = name;
  if (tab) app.tab = tab;
  writeHash(replace);
  $$(".run-item").forEach((el) => el.classList.toggle("active", el.dataset.name === name));
  showMain();
  if (changed || !app.data[name]) loadRun(name);
  else showTab(app.tab);
}

async function loadRun(name) {
  const d = (app.data[name] = app.data[name] || {});
  updateHeader();
  showTab(app.tab, { loading: true });
  try {
    const [listing, summary, progress, profiles] = await Promise.all([
      api.listing(name),
      optional(api.file(name, "summary.json")),
      optional(api.file(name, "progress.json")),
      optional(api.file(name, "profiles.json")),
    ]);
    if (app.current !== name) return;
    Object.assign(d, { listing, summary, progress, profiles, error: null });
  } catch (e) {
    d.error = e.message;
  }
  if (app.current !== name) return;
  // a running job without a summary opens on the progress tab
  const entry = entryOf(name);
  if (entry && RUNNING_STATES.has(entry.state) && !d.summary && app.tab !== "progress" && !d.visited) {
    app.tab = "progress";
    writeHash(true);
  }
  d.visited = true;
  updateHeader();
  showTab(app.tab);
}

/** Load (once) and cache the mesh for a run; returns the prepared mesh or null. */
function loadMesh(name, onProgress) {
  const d = app.data[name];
  if (d.mesh !== undefined) return Promise.resolve(d.mesh);
  if (!d.meshPromise) {
    d.meshPromise = api.fileProgress(name, "fields.json", onProgress)
      .then((json) => (d.mesh = prepareMesh(json)))
      .catch((e) => {
        d.meshPromise = null;
        if (e.status === 404) return (d.mesh = null);
        throw e;
      });
  }
  return d.meshPromise;
}

// ---------------------------------------------------------------------------
// Header + tabs
// ---------------------------------------------------------------------------
function updateHeader() {
  const title = $("#run-title");
  if (app.launching) {
    setChildren(title, h("span.rt-name", { text: "New run" }));
    return;
  }
  const entry = entryOf(app.current);
  if (!app.current) return setChildren(title, h("span.dim", { text: "no run selected" }));
  const d = app.data[app.current] || {};
  const s = d.summary || {};
  const bits = [];
  if (s.case || (entry && entry.case)) bits.push(s.case || entry.case);
  if (s.created) bits.push(String(s.created).replace("T", " ").slice(0, 16));
  const f = s.integrals && s.integrals.force;
  setChildren(title,
    h("span.rt-name", { text: app.current }),
    entry ? stateBadge(entry) : null,
    h("span.rt-meta", { text: bits.join(" · ") }),
    f && Number.isFinite(f.drag) ? h("span.rt-kpi", {}, h("span.dim", { text: "drag " }), h("b.mono", { text: fmt(f.drag, 5) })) : null);
  // tab availability hints
  const files = (d.listing && d.listing.files) || {};
  const has = {
    field: "fields.json" in files, profiles: "profiles.json" in files,
    summary: "summary.json" in files || "progress.json" in files,
    gallery: d.listing && d.listing.plots && d.listing.plots.length > 0, progress: true,
  };
  for (const t of TABS) {
    const btn = $(`#tab-${t.id}`);
    btn.classList.toggle("unavailable", d.listing ? !has[t.id] : false);
    const cnt = t.id === "gallery" && d.listing ? d.listing.plots.length : null;
    btn.querySelector(".tab-count").textContent = cnt ? String(cnt) : "";
  }
}

function showMain() {
  $("#launch-view").classList.add("hidden");
  $("#tabs").classList.remove("hidden");
  $("#new-run-btn").classList.remove("active");
}

function showTab(tab, { loading = false } = {}) {
  app.tab = tab;
  writeHash(true);
  for (const t of TABS) {
    $(`#tab-${t.id}`).classList.toggle("active", t.id === tab);
    $(`#tab-${t.id}`).setAttribute("aria-selected", String(t.id === tab));
    $(`#pane-${t.id}`).classList.toggle("hidden", t.id !== tab);
  }
  const name = app.current;
  const d = name && app.data[name];
  const pane = $(`#pane-${tab}`);
  if (!name) return;
  if (loading || !d || (!d.listing && !d.error)) {
    if (tab !== "field") setChildren(pane, placeholder("loading", "Loading run…"));
    return;
  }
  if (d.error) {
    if (tab === "field") app.field.showMessage("error", "Could not load run", d.error);
    else setChildren(pane, placeholder("error", "Could not load run", d.error));
    return;
  }
  ({ field: showField, profiles: showProfiles, summary: showSummary, gallery: showGallery, progress: showProgress })[tab](name, d);
  requestAnimationFrame(() => resizeAll(pane));
}

async function showField(name, d) {
  const fv = app.field;
  if (fv.runName === name && fv.mesh) return fv.resize();
  if (!d.listing.files["fields.json"]) {
    fv.runName = null;
    return fv.showMessage("empty", "No field data", "This run has no fields.json (yet).");
  }
  const size = d.listing.files["fields.json"].size;
  const bar = h("div.load-bar", {}, h("div.load-fill"));
  const label = h("span.mono", { text: `fields.json · ${(size / 1048576).toFixed(1)} MB` });
  fv.showMessage("loading", "Loading mesh & fields…", h("div", {}, label, bar));
  try {
    const mesh = await loadMesh(name, (frac, got) => {
      bar.firstChild.style.width = `${(frac ?? got / size) * 100}%`;
    });
    if (app.current !== name || app.tab !== "field") return;
    if (!mesh) return fv.showMessage("empty", "No field data", "fields.json not found.");
    fv.runName = name;
    fv.setMesh(mesh);
    if (mesh.bad) toast(`fields.json: ${mesh.bad} triangle indices out of range (clamped)`, "warn");
  } catch (e) {
    if (app.current === name) fv.showMessage("error", "Failed to load fields.json", e.message);
  }
}

function showProfiles(name, d) {
  if (app.profiles.runName !== name || app.profiles.data !== d.profiles) app.profiles.setData(name, d.profiles);
  else app.profiles.drawPlot();
}

function showSummary(name, d) {
  app.summary.render(d.summary, d.progress);
}

function showGallery(name, d) {
  app.gallery.render(name, d.listing.plots);
}

function showProgress(name, d) {
  renderProgress($("#progress-box"), d.progress, entryOf(name));
  const entry = entryOf(name);
  const stopBtn = $("#stop-btn");
  stopBtn.classList.toggle("hidden", !(entry && entry.proc && entry.proc.alive));
  if (app.log.runName !== name) app.log.load(name);
  else app.log.refresh();
}

// ---------------------------------------------------------------------------
// Launch view
// ---------------------------------------------------------------------------
function showLaunch() {
  app.launching = true;
  writeHash(false);
  $("#tabs").classList.add("hidden");
  for (const t of TABS) $(`#pane-${t.id}`).classList.add("hidden");
  $("#launch-view").classList.remove("hidden");
  $("#new-run-btn").classList.add("active");
  $$(".run-item").forEach((el) => el.classList.remove("active"));
  updateHeader();
  app.launch.show();
}

async function onLaunched(name) {
  await refreshRuns();
  delete app.data[name];
  app.current = null;
  selectRun(name, { tab: "progress" });
}

// ---------------------------------------------------------------------------
// Hash routing:  #run=<name>&tab=<tab>   or  #new
// ---------------------------------------------------------------------------
function writeHash(replace) {
  const hash = app.launching ? "#new"
    : app.current ? `#run=${encodeURIComponent(app.current)}&tab=${app.tab}` : "";
  if (location.hash === hash) return;
  if (replace) history.replaceState(null, "", hash || location.pathname);
  else history.pushState(null, "", hash || location.pathname);
}

function readHash() {
  const hs = location.hash.slice(1);
  if (hs === "new") return { launch: true };
  const p = new URLSearchParams(hs);
  const tab = TABS.some((t) => t.id === p.get("tab")) ? p.get("tab") : null;
  return { run: p.get("run"), tab };
}

function applyHash() {
  const r = readHash();
  if (r.launch) return showLaunch();
  if (r.tab) app.tab = r.tab;
  if (r.run) selectRun(r.run, { replace: true });
}

// ---------------------------------------------------------------------------
// Toasts, keyboard, help
// ---------------------------------------------------------------------------
function toast(text, kind = "info") {
  const t = h(`div.toast.toast-${kind}`, { text });
  $("#toasts").appendChild(t);
  setTimeout(() => t.classList.add("out"), 3800);
  setTimeout(() => t.remove(), 4300);
}

function moveSelection(delta) {
  const items = $$(".run-item");
  if (!items.length) return;
  const i = items.findIndex((el) => el.dataset.name === app.current);
  const next = items[Math.min(items.length - 1, Math.max(0, (i < 0 ? 0 : i + delta)))];
  next.scrollIntoView({ block: "nearest" });
  selectRun(next.dataset.name);
}

function onKey(e) {
  if (app.gallery.key(e)) return e.preventDefault();
  const tag = (e.target.tagName || "").toLowerCase();
  if (["input", "select", "textarea"].includes(tag) || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.key === "?") return toggleHelp();
  if (e.key === "Escape") return toggleHelp(false);
  const tab = TABS.find((t) => t.key === e.key);
  if (tab && app.current) { showMain(); app.launching = false; return showTab(tab.id); }
  const fieldTab = app.tab === "field" && !app.launching;
  const actions = {
    j: () => moveSelection(1), k: () => moveSelection(-1),
    t: toggleTheme, n: showLaunch, r: () => refreshRuns(), "/": () => $("#run-filter").focus(),
    f: () => fieldTab && app.field.zoomAll(), g: () => fieldTab && app.field.zoomFocus(),
    e: () => fieldTab && app.field.toggleOpt("edges"), b: () => fieldTab && app.field.toggleOpt("bnd"),
    s: () => fieldTab && app.field.toggleOpt("stream"),
    "[": () => $("#app").classList.toggle("sidebar-collapsed"),
  };
  const fn = actions[e.key.toLowerCase()];
  if (fn) {
    e.preventDefault();
    fn();
  }
}

function toggleHelp(force) {
  const el = $("#help");
  const show = force === undefined ? el.classList.contains("hidden") : force;
  el.classList.toggle("hidden", !show);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function buildTabs() {
  setChildren($("#tabs"), TABS.map((t) => h("button.tab", {
    id: `tab-${t.id}`, role: "tab", title: `${t.label}  [${t.key}]`,
    onclick: () => showTab(t.id),
  }, t.label, h("span.tab-count"))));
}

async function loadCompareProfiles(name) {
  const d = (app.data[name] = app.data[name] || {});
  if (d.profiles === undefined) d.profiles = await optional(api.file(name, "profiles.json"));
  return d.profiles;
}

function init() {
  applyTheme(store.get("theme", "dark"));
  buildTabs();
  app.field = new FieldView($("#pane-field"));
  app.profiles = new ProfilesView($("#pane-profiles"), {
    loadCompare: loadCompareProfiles, runNames: () => app.runs.filter((r) => r.files.includes("profiles.json")).map((r) => r.name),
  });
  app.summary = new SummaryView($("#pane-summary"));
  app.gallery = new GalleryView($("#pane-gallery"));
  app.log = new LogView($("#log-box"));
  app.launch = new LaunchView($("#launch-view"), { onLaunched });
  $("#theme-btn").addEventListener("click", toggleTheme);
  $("#new-run-btn").addEventListener("click", showLaunch);
  $("#refresh-btn").addEventListener("click", () => {
    if (app.current) delete app.data[app.current];
    app.field.runName = null;
    refreshRuns().then(() => app.current && !app.launching && loadRun(app.current));
  });
  $("#help-btn").addEventListener("click", () => toggleHelp());
  $("#help").addEventListener("click", (e) => e.target.id === "help" && toggleHelp(false));
  $("#collapse-btn").addEventListener("click", () => {
    $("#app").classList.toggle("sidebar-collapsed");
    setTimeout(() => { app.field.resize(); resizeAll(document); }, 220);
  });
  $("#stop-btn").addEventListener("click", async () => {
    try {
      await api.stop(app.current);
      toast(`Stop requested for '${app.current}'`, "warn");
    } catch (e) { toast(e.message, "error"); }
  });
  $("#run-filter").addEventListener("input", debounce((e) => { app.filter = e.target.value; renderRunList(); }, 80));
  document.addEventListener("keydown", onKey);
  window.addEventListener("popstate", applyHash);
  window.addEventListener("resize", debounce(() => resizeAll(document), 150));
  applyHash();
  refreshRuns();
  setInterval(() => { if (!document.hidden) refreshRuns(); }, POLL_MS);
  if (!window.Plotly) toast("Plotly not loaded — charts disabled (static/vendor/plotly.min.js)", "error");
}

init();
