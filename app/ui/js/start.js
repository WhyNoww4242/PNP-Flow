// Start page: cases (cases/<case>/), new case from a template, recent runs, and a
// short ranked list of examples (the pore runs and the key validation runs).

import { h, setChildren, fmt, fmtDate, unitOf, placeholder, store } from "./util.js";
import { icon } from "./icons.js";
import { bridge } from "./bridge.js";
import { RUNNING, STATE_WORD, isStale } from "./progress.js";

// Plain-language titles for known templates (the case name stays visible as code).
const TITLES = {
  charged_sphere: "Charged sphere (Poisson)",
  cylinder: "Charged cylinder in cross flow",
  gouy_chapman: "Planar double layer (Gouy–Chapman)",
  eo_slit: "Electro-osmotic slit flow",
  mms: "Manufactured solution (all terms)",
};
// Examples: at most five, ranked by importance (5 = most important).
const EXAMPLES = [
  { name: "pore_narrow", rank: 5, why: "Pore charging in time (paper DNS set-up, 1 µm pore): the reference pore case" },
  { name: "pore_dns_narrow_coarse", rank: 4, why: "Pore DNS comparison run (10 µm pore, 111k elements); one check fails, see Checks" },
  { name: "val_gouy_chapman", rank: 4, why: "Planar double layer against the exact Gouy–Chapman solution" },
  { name: "val_mms", rank: 3, why: "Manufactured solution: every equation term and boundary-condition type" },
  { name: "val_eo_slit", rank: 3, why: "Electro-osmotic slit flow against nonlinear Poisson–Boltzmann" },
];
const EXAMPLE_NAMES = new Set(EXAMPLES.map((e) => e.name));

function importance(rank) {
  return h("span.importance", { role: "img", "aria-label": `importance ${rank} of 5`, "data-tip": `Importance ${rank} of 5` },
    [1, 2, 3, 4, 5].map((i) => h("i", { class: i <= rank ? "on" : "" })));
}

// Parameters worth showing first on a template row.
const KEY_PARAMS = ["delta", "zeta_mV", "R", "radius", "rho", "q", "charge", "Pe", "Re", "K", "Ex", "length", "T", "h"];

export function caseTitle(name) {
  if (!name) return "Unknown case";
  return TITLES[name] || String(name).replace(/_/g, " ").replace(/^./, (c) => c.toUpperCase());
}

export function caseDoc(c) {
  if (!c) return "";
  if (typeof c.doc === "string") return c.doc.trim().split(/\n\s*\n/)[0].replace(/\s+/g, " ").trim();
  if (typeof c.description === "string") return c.description;
  return "";
}

function keyParams(c, n = 4) {
  const params = c.params || {};
  const names = Object.keys(params);
  const pick = KEY_PARAMS.filter((k) => names.includes(k));
  for (const k of names) if (pick.length < n && !pick.includes(k)) pick.push(k);
  return pick.slice(0, n).map((k) => {
    const u = unitOf((c.info || {})[k]);
    return h("span", {}, `${k} `, h("b", { text: fmt(params[k]) }), u ? ` ${u}` : "");
  });
}

/** Status word + class for a run-list entry. */
export function runStatusWord(r) {
  if (isStale(r)) return { text: "stopped updating", cls: "st-warn" };
  if (RUNNING.has(r.state)) return { text: STATE_WORD[r.state] || "running", cls: "st-run" };
  if (r.state === "failed") return { text: "failed", cls: "st-bad" };
  if (r.converged === true) {
    if (r.checks === "fail") return { text: "converged, check failed", cls: "st-bad" };
    if (r.checks === "warn") return { text: "converged, check warning", cls: "st-warn" };
    return { text: "converged", cls: "st-ok" };
  }
  if (r.converged === false) return { text: "not converged", cls: "st-bad" };
  if (r.state === "done") return { text: "done", cls: "st-ok" };
  return { text: "no result", cls: "st-neutral" };
}

function thumbImage(r) {
  const box = h("div.rc-thumb", {}, icon("field-map", 28));
  if (r.thumb) {
    bridge.imageURL(r.name, r.thumb).then((url) => {
      const img = h("img", { src: url, alt: "", loading: "lazy" });
      img.addEventListener("load", () => setChildren(box, img));
    }).catch(() => {});
  }
  return box;
}

function runCard(app, r, extra) {
  const st = runStatusWord(r);
  return h("button.rcard", {
    type: "button", "data-run": r.name, "data-case": r.case || "", onclick: () => app.openRun(r.name),
    "aria-label": `Open run ${r.name}, ${st.text}`,
  },
  thumbImage(r),
  h("div.rc-body", {},
    h("span.rc-name", { text: r.name, title: r.name }),
    h("span.rc-meta", {}, h("span.rc-status", { class: st.cls, text: st.text })),
    h("span.rc-meta", {}, h("span.rc-case", { text: r.case ? caseTitle(r.case) : "case unknown" }), h("span.rc-date", { text: fmtDate(r.mtime) })),
    extra ? h("span.rc-extra", { text: extra }) : null));
}

/** For validation runs: the largest error against an exact solution, if reported. */
async function exactError(name) {
  try {
    const s = await bridge.json(name, "summary.json");
    const errs = (s.checks || []).filter((c) => /exact|err|manufactured|mms/i.test(`${c.name} ${c.note || ""}`) && typeof c.value === "number");
    if (!errs.length) return null;
    const worst = errs.reduce((a, b) => (Math.abs(b.value) > Math.abs(a.value) ? b : a));
    return `${worst.name}: ${fmt(worst.value, 2)}`;
  } catch {
    return null;
  }
}

export function renderStart(app, root) {
  const S = app.S;
  const caseNames = Object.keys(S.cases);
  caseNames.sort((a, b) => (a === "charged_sphere" ? -1 : b === "charged_sphere" ? 1 : 0));
  const runsByCase = {};
  for (const r of S.runs) if (r.case && r.thumb && !runsByCase[r.case]) runsByCase[r.case] = r;

  // --- templates
  let tplBody;
  if (!S.casesLoaded) {
    tplBody = h("div.tpl-list", {}, [0, 1, 2, 3].map(() => h("div.tpl", { "aria-hidden": "true" },
      h("div.tpl-thumb.skel"), h("div.tpl-body", {}, h("div.skel", { style: { height: "14px", width: "40%" } }),
        h("div.skel", { style: { height: "12px", width: "85%", marginTop: "6px" } })))));
  } else if (!caseNames.length) {
    tplBody = h("div.empty-row", {}, placeholder("error", "No templates available",
      h("span", {}, S.casesError || "run.py --list returned no cases.", " The solver must be reachable to create projects; recent runs below still open."),
      h("button.btn.sm", { type: "button", onclick: () => app.loadCases(true) }, icon("refresh", 14), "Try again")));
  } else {
    tplBody = h("div.tpl-list", {}, caseNames.map((n) => {
      const c = S.cases[n];
      const sample = runsByCase[n];
      const th = h("div.tpl-thumb", {}, icon(n === "charged_sphere" ? "ion" : "geometry", 26));
      if (sample) bridge.imageURL(sample.name, sample.thumb).then((url) => setChildren(th, h("img", { src: url, alt: "" }))).catch(() => {});
      return h("button.tpl", {
        type: "button", class: n === "charged_sphere" ? "featured" : "", onclick: () => app.openTemplate(n),
        "aria-label": `New case from template ${caseTitle(n)}`,
      }, th, h("div.tpl-body", {},
        h("span.tpl-title", {}, h("b", { text: caseTitle(n) }), h("code", { text: n }),
          n === "charged_sphere" ? h("span.tag", { text: "start here" }) : null),
        h("span.tpl-doc", { text: caseDoc(c) || `${Object.keys(c.params || {}).length} parameters` }),
        h("span.tpl-params", {}, keyParams(c))));
    }));
  }

  // --- cases
  const projects = (S.projects || []).slice().sort((a, b) => (b.mtime || 0) - (a.mtime || 0));
  let casesBody;
  if (!S.projectsLoaded) casesBody = h("div.empty-row", { text: "Loading cases…" });
  else if (!projects.length) casesBody = h("div.empty-row", { text: `No cases yet. Pick a template below to make one (saved in ${S.casesDir || "cases/"}).` });
  else casesBody = h("div.cards", {}, projects.map((c) => {
    const m = c.mesh || {};
    const meshTxt = m.state === "done" ? `${fmt(m.n_elements)} elements${m.up_to_date ? "" : ", out of date"}` : m.state === "building" ? "mesh building" : m.state === "failed" ? "mesh failed" : "no mesh yet";
    return h("button.rcard.ccard", { type: "button", "data-run": c.name, "data-case": c.template || "", onclick: () => app.openCase(c.name),
      "aria-label": `Open case ${c.name}` },
    h("div.rc-thumb", {}, icon("folder-open", 26)),
    h("div.rc-body", {},
      h("span.rc-name", { text: c.name, title: c.name }),
      h("span.rc-meta", {}, h("span.rc-case", { text: caseTitle(c.template) })),
      h("span.rc-meta", {}, h("span", { class: m.state === "failed" ? "rc-status st-bad" : m.state === "done" && m.up_to_date ? "rc-status st-ok" : "rc-status st-neutral", text: meshTxt }),
        h("span.rc-date", { text: `${c.runs.length} run${c.runs.length === 1 ? "" : "s"}` }))));
  }));

  // --- recent runs and examples
  // Clear recents hides the runs present now (remembered per browser); nothing is deleted
  const clearedAt = Number(store.get("recentClearedAt", 0)) || 0;
  const notExample = S.runs.filter((r) => !EXAMPLE_NAMES.has(r.name));
  const recent = notExample.filter((r) => !(r.mtime <= clearedAt) || (RUNNING.has(r.state) && !isStale(r)));
  const hiddenN = notExample.length - recent.length;
  const clearBtn = h("button.btn.sm", { type: "button", disabled: !recent.length || null,
    "data-tip-title": "Clear recents", "data-tip": "Hide these runs from Recent runs. Nothing is deleted; they stay in their folders and in Open.",
    onclick: () => { store.set("recentClearedAt", Date.now() / 1000); app.rerenderStart(); } }, icon("x", 14), "Clear recents");
  const restore = hiddenN ? h("button.btn-link", { type: "button", onclick: () => { store.set("recentClearedAt", 0); app.rerenderStart(); } },
    `Show ${hiddenN} cleared`) : null;
  const examples = EXAMPLES.map((e) => ({ e, r: S.runs.find((x) => x.name === e.name) })).filter((x) => x.r);
  const filterInput = h("input.inp#run-filter", {
    type: "search", placeholder: "Filter runs  ( / )", value: S.startFilter || "", "aria-label": "Filter runs",
    oninput: (e) => { S.startFilter = e.target.value; applyFilter(root, S.startFilter); },
  });

  let recentBody;
  if (!S.runsLoaded) recentBody = h("div.cards", {}, [0, 1, 2].map(() => h("div.rcard", { "aria-hidden": "true" }, h("div.rc-thumb.skel"), h("div.rc-body", {}, h("div.skel", { style: { height: "12px" } })))));
  else if (!S.runs.length) recentBody = h("div.empty-row", { text: "No runs yet. Pick a template above and press Solve; results appear here." });
  else if (!recent.length) recentBody = h("div.empty-row", { text: hiddenN ? "Recents cleared. New runs appear here." : "No runs besides the examples yet." });
  else recentBody = h("div.cards", {}, recent.slice(0, 18).map((r) => runCard(app, r)));

  const exampleCards = examples.map(({ e, r }) => {
    const card = runCard(app, r, e.why);
    card.querySelector(".rc-body").prepend(importance(e.rank));
    if (/^val_/.test(r.name)) exactError(r.name).then((t) => { const el = card.querySelector(".rc-extra"); if (el && t) el.textContent = `${e.why}. Error vs exact, ${t}`; });
    return card;
  });
  const exBody = examples.length ? h("div.cards", {}, exampleCards)
    : h("div.empty-row", { text: "No example runs found. Run bash tools/validate.sh to create the validation runs." });

  // --- environment line (replaces FreeCAD's first-start setup for now)
  const env = h("div.start-foot", {},
    h("span", {}, "Solver: ", S.casesLoaded ? (caseNames.length ? h("span.ok", { text: `run.py found, ${caseNames.length} templates` }) : h("span.bad", { text: "run.py --list failed" })) : "checking…"),
    h("span", { text: `Results folder: ${S.resultsDir || "–"}` }),
    h("span", {}, "Colormaps: ", S.cmSource && S.cmSource.fallback ? h("span.bad", { text: "built-in fallback (data/colormaps.json missing)" }) : h("span.ok", { text: "cmasher tables loaded" })),
    h("span", { text: app.inTauri ? "Running in the desktop shell" : "Running in a browser (dev server)" }));

  setChildren(root, h("div.page", {}, h("div.page-pad", {}, h("div.start-page", {},
    h("section.section", {},
      h("div.section-head", {}, h("h2", { text: "Cases" }), h("span.count", { text: S.projectsLoaded ? `${projects.length} in ${S.casesDir || "cases/"}` : "" })),
      casesBody),
    h("section.section", {},
      h("div.section-head", {}, h("h2", { text: "New case" }), h("span.count", { text: "from a template; saved as cases/<name>/case.json" })),
      tplBody),
    h("section.section", {},
      h("div.section-head", {}, h("h2", { text: "Recent runs" }), h("span.count", { text: S.runsLoaded ? String(recent.length) : "" }), restore, h("span.spacer"), clearBtn, filterInput),
      recentBody),
    h("section.section", {},
      h("div.section-head", {}, h("h2", { text: "Examples" }), h("span.count", { text: "ranked by importance: the pore runs and the key validation runs" })),
      exBody),
    env))));
  applyFilter(root, S.startFilter);
}

/** Hide run cards that do not match the filter (no re-render, so thumbnails stay). */
function applyFilter(root, text) {
  const f = (text || "").toLowerCase().trim();
  for (const grid of root.querySelectorAll(".cards")) {
    let shown = 0;
    for (const card of grid.querySelectorAll(".rcard[data-run]")) {
      const hay = `${card.dataset.run} ${card.dataset.case || ""}`.toLowerCase();
      const ok = !f || hay.includes(f);
      card.classList.toggle("hidden", !ok);
      if (ok) shown++;
    }
    let note = grid.nextElementSibling && grid.nextElementSibling.classList.contains("filter-note") ? grid.nextElementSibling : null;
    if (!shown && f && grid.querySelector(".rcard[data-run]")) {
      if (!note) { note = h("div.empty-row.filter-note"); grid.after(note); }
      note.textContent = `No runs match "${text}".`;
    } else if (note) note.remove();
  }
}
