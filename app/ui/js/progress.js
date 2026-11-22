// Run state, progress and health.
//
// progress.json may carry `fraction`, `health` and `tells` {res_norm, ratio, alpha,
// min_c, halvings} (docs/MCP.md §6). When they are present they are used as is.
// When they are absent, health is derived from `history` and marked as derived;
// the fraction is never invented: the current stage bar is then indeterminate.

import { h, fmt, fmtDuration, isNum } from "./util.js";
import { icon } from "./icons.js";

export const RUNNING = new Set(["meshing", "assembling", "solving", "post", "launched"]);
export const STATE_WORD = {
  meshing: "meshing", assembling: "assembling", solving: "solving", post: "post-processing",
  done: "done", failed: "failed", launched: "starting", unknown: "unknown", stale: "stopped updating",
};
const HEALTH = {
  converging: { cls: "b-ok", icon: "trend-down", text: "converging" },
  struggling: { cls: "b-warn", icon: "alert", text: "struggling" },
  stalled: { cls: "b-warn", icon: "alert", text: "stalled" },
  failed: { cls: "b-bad", icon: "x-circle", text: "failed" },
  done: { cls: "b-ok", icon: "check-circle", text: "done" },
};

// A run that says it is running but was not launched by this server and has not
// written anything for this long is reported as "stopped updating", not as live.
const STALE_S = 600;

/** True when progress.json claims a running state but nothing is alive behind it. */
export function isStale(entry) {
  if (!entry || !RUNNING.has(entry.state)) return false;
  if (entry.proc && entry.proc.alive) return false;
  return typeof entry.mtime === "number" && Date.now() / 1000 - entry.mtime > STALE_S;
}

/** History of the current (last) stage. */
function stageHistory(p) {
  const hist = Array.isArray(p.history) ? p.history.filter((it) => isNum(it.res_norm)) : [];
  if (!hist.length) return [];
  const st = isNum(p.stage) ? p.stage : hist[hist.length - 1].stage;
  const cur = hist.filter((it) => (it.stage ?? st) === st);
  return cur.length ? cur : hist;
}

/** Derive tells and a health word from the iteration history. */
function deriveHealth(p, state) {
  const its = stageHistory(p);
  const last = its[its.length - 1];
  const tells = {};
  if (isNum(p.res_norm)) tells.res_norm = p.res_norm;
  else if (last) tells.res_norm = last.res_norm;
  if (its.length >= 2) {
    const a = its[its.length - 2].res_norm, b = last.res_norm;
    if (a > 0) tells.ratio = b / a;
  }
  if (last && isNum(last.alpha) && last.it > 0) tells.alpha = last.alpha;
  if (state === "failed") return { health: "failed", tells };
  if (state === "done") return { health: "done", tells };
  if (!RUNNING.has(state) || its.length < 2) return { health: null, tells };
  const ratios = [];
  for (let k = Math.max(1, its.length - 3); k < its.length; k++) {
    if (its[k - 1].res_norm > 0) ratios.push(its[k].res_norm / its[k - 1].res_norm);
  }
  let health = "converging";
  if (ratios.length >= 3 && ratios.every((r) => r > 0.95)) health = "stalled";
  else if (tells.ratio > 0.7 || (isNum(tells.alpha) && tells.alpha < 0.5)) health = "struggling";
  return { health, tells };
}

/**
 * Combine progress.json and the run-list entry into one status object.
 * fraction: overall 0..1 or null (unknown -> indeterminate).
 */
export function runStatus(progress, entry) {
  const p = progress || {};
  const stale = isStale(entry);
  const state = stale ? "stale" : (entry && entry.state) || p.state || "unknown";
  const running = RUNNING.has(state);
  const fromSolver = typeof p.health === "string" && p.health in HEALTH;
  const derived = deriveHealth(p, state);
  const health = state === "done" ? "done" : state === "failed" ? "failed" : fromSolver ? p.health : derived.health;
  const tells = Object.assign({}, derived.tells, p.tells && typeof p.tells === "object" ? p.tells : {});
  let fraction = isNum(p.fraction) ? Math.min(1, Math.max(0, p.fraction)) : null;
  if (state === "done") fraction = 1;
  return {
    state, running, health, healthDerived: !fromSolver && !!health && running,
    tells, fraction,
    stage: isNum(p.stage) ? p.stage : null,
    nStages: isNum(p.n_stages) ? p.n_stages : null,
    iteration: isNum(p.iteration) ? p.iteration : null,
    resNorm: isNum(tells.res_norm) ? tells.res_norm : isNum(p.res_norm) ? p.res_norm : null,
    message: p.message || "",
    timeS: isNum(p.time_s) ? p.time_s : null,
    history: Array.isArray(p.history) ? p.history : [],
  };
}

/** Text for the health tooltip: the numbers behind the word. */
export function tellsText(st) {
  const t = st.tells || {};
  const lines = [];
  if (isNum(t.res_norm)) lines.push(`residual ‖R‖ = ${fmt(t.res_norm, 3)}`);
  if (isNum(t.ratio)) lines.push(`contraction ratio = ${fmt(t.ratio, 3)}`);
  if (isNum(t.alpha)) lines.push(`line-search step α = ${fmt(t.alpha, 3)}`);
  if (isNum(t.min_c)) lines.push(`min concentration = ${fmt(t.min_c, 3)}`);
  if (isNum(t.halvings)) lines.push(`continuation halvings = ${fmt(t.halvings)}`);
  if (st.state === "stale") lines.push("progress.json says the run is in progress, but it has not changed for a long time and no process launched by this app is behind it. The run was probably interrupted.");
  if (!lines.length) lines.push("no iteration data yet");
  if (st.healthDerived) lines.push("(derived from the iteration history; the solver does not report health yet)");
  return lines.join("\n");
}

/** Health badge (word + icon + colour). Returns null when there is nothing to say. */
export function healthBadge(st, { asButton = true } = {}) {
  let spec = st.health ? HEALTH[st.health] : null;
  if (!spec) {
    if (!st.running && st.state === "unknown") return null;
    spec = st.state === "stale" ? { cls: "b-warn", icon: "alert", text: "stopped updating" }
      : { cls: st.running ? "b-run" : "b-neutral", icon: st.running ? "refresh" : "info", text: STATE_WORD[st.state] || st.state };
  }
  const tag = asButton ? "button.badge" : "span.badge";
  return h(tag, { class: spec.cls, type: asButton ? "button" : null, "data-tip": tellsText(st), "data-tip-title": `Health: ${spec.text}`, "aria-label": `Run health: ${spec.text}. ${tellsText(st)}` },
    icon(spec.icon, 12), spec.text);
}

/** One bar per stage. Completed stages are full; the current one shows the reported
 *  fraction, or is indeterminate when the solver does not report one. */
export function stageBars(st, { wide = false } = {}) {
  const n = st.nStages || 1;
  const cur = st.stage || (st.state === "done" ? n : 1);
  const bars = [];
  for (let i = 1; i <= n; i++) {
    let cls = "", width = "0%";
    if (st.state === "done" || i < cur) { cls = "done"; width = "100%"; }
    else if (i === cur) {
      if (st.state === "failed") { cls = "failed"; }
      else if (st.fraction !== null) {
        const within = Math.min(1, Math.max(0, st.fraction * n - (i - 1)));
        width = `${(within * 100).toFixed(1)}%`;
      } else if (st.running) cls = "indet";
    }
    bars.push(h("span.stage-bar", { class: cls, title: `stage ${i} of ${n}` }, h("span.fill", { style: { width } })));
  }
  const label = st.nStages ? `stage ${Math.min(cur, n)} of ${n}` : "stage count not reported";
  const fr = st.fraction !== null ? `, ${(st.fraction * 100).toFixed(0)} % done` : st.running ? ", progress not reported" : "";
  return h("span.stages", { class: wide ? "wide" : "", role: "progressbar", "aria-label": label + fr,
    "aria-valuemin": "0", "aria-valuemax": "100", "aria-valuenow": st.fraction !== null ? String(Math.round(st.fraction * 100)) : null,
    "data-tip": label + fr }, bars);
}

/** Small key-value grid of the numbers behind the health verdict. */
export function tellsGrid(st) {
  const t = st.tells || {};
  const items = [
    ["residual ‖R‖", isNum(t.res_norm) ? fmt(t.res_norm, 3) : "–"],
    ["contraction ratio", isNum(t.ratio) ? fmt(t.ratio, 3) : "–"],
    ["line-search α", isNum(t.alpha) ? fmt(t.alpha, 3) : "–"],
    ["min concentration", isNum(t.min_c) ? fmt(t.min_c, 3) : "not reported"],
    ["halvings", isNum(t.halvings) ? fmt(t.halvings) : "not reported"],
    ["iteration", st.iteration ?? "–"],
    ["elapsed", fmtDuration(st.timeS)],
  ];
  return h("dl.tells", {}, items.map(([k, v]) => h("div.tell", {}, h("dt", { text: k }), h("dd", { text: String(v) }))));
}

/** Iteration history grouped by stage: [{stage, its}] (summary first, then progress). */
export function convergenceSeries(summary, progress) {
  const stages = summary && summary.solver && Array.isArray(summary.solver.stages) ? summary.solver.stages : [];
  const out = [];
  stages.forEach((st, i) => {
    const its = (st.iterations || []).filter((it) => isNum(it.res_norm));
    if (its.length) out.push({ stage: i + 1, its, params: st.params, converged: st.converged });
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
