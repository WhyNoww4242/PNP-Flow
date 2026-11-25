// Plotly theming from the DESIGN.md tokens (read from CSS variables at draw time).
// Line colours follow the house style: cmasher cosmic sampled over a sub-range
// (cosmic_r over 0.2–0.75 on light backgrounds; the lighter end on dark ones).

import { cssVar } from "./util.js";
import { sampleColors } from "./colormaps.js";
import { isNum } from "./util.js";

export function seriesColors(n = 4) {
  const dark = document.documentElement.dataset.theme !== "light";
  const k = Math.max(n, 2);
  return dark ? sampleColors("cosmic", k, 0.55, 0.95) : sampleColors("cosmic", k, 0.2, 0.75, true);
}

export function baseLayout(extra = {}) {
  const fg = cssVar("--text") || "#dfe5ee";
  const dim = cssVar("--text-dim") || "#8b96a8";
  const grid = cssVar("--grid") || "#262d3a";
  const line = cssVar("--border-strong") || "#343d4e";
  const font = cssVar("--font-sans") || "sans-serif";
  const mono = cssVar("--font-mono") || "monospace";
  const axis = (a = {}) => Object.assign({
    gridcolor: grid, zerolinecolor: line, linecolor: line, tickcolor: line, showgrid: false,
    tickfont: { color: dim, size: 11, family: mono },
    title: { font: { color: dim, size: 12, family: font } }, automargin: true, exponentformat: "e", showexponent: "all",
  }, a);
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: fg, family: font, size: 12 },
    margin: { l: 56, r: 16, t: 12, b: 44 },
    legend: { bgcolor: "rgba(0,0,0,0)", font: { size: 11.5, color: fg }, orientation: "h", y: -0.2 },
    hoverlabel: { font: { family: mono, size: 11, color: fg }, bgcolor: cssVar("--surface-raised") || "#1a202b", bordercolor: line },
    colorway: seriesColors(6),
  }, extra, {
    xaxis: axis(extra.xaxis), yaxis: axis(extra.yaxis),
    ...(extra.yaxis2 ? { yaxis2: axis(extra.yaxis2) } : {}),
  });
}

export const PLOT_CONFIG = {
  displaylogo: false,
  responsive: true,
  modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d", "toggleSpikelines"],
  toImageButtonOptions: { format: "png", scale: 2 },
};

/** Plotly.react if available; returns false (with a note) if Plotly failed to load. */
export function plot(el, traces, layout, config = {}) {
  if (!window.Plotly) {
    el.textContent = "Charts are unavailable because vendor/plotly.min.js did not load.";
    return false;
  }
  // drop a placeholder (loading / empty) that was shown before the first draw
  if (!el.classList.contains("js-plotly-plot")) el.textContent = "";
  window.Plotly.react(el, traces, layout, Object.assign({}, PLOT_CONFIG, config));
  return true;
}

export function resizeAll(root) {
  if (!window.Plotly) return;
  for (const el of root.querySelectorAll(".js-plotly-plot")) {
    if (el.offsetParent !== null) window.Plotly.Plots.resize(el);
  }
}

/**
 * Residual history as one trace: line through every Newton iteration, markers
 * coloured by stage (colour bar instead of a legend entry per stage).
 * series: [{stage, its:[{it, res_norm, alpha}]}]; label: colour-bar title.
 */
export function residualPlot(el, series, { label = "stage" } = {}) {
  const dark = document.documentElement.dataset.theme !== "light";
  // same direction in both themes (early = dark end); light only drops the faintest part
  const ramp = dark ? sampleColors("cosmic", 9, 0.45, 0.95) : sampleColors("cosmic", 9, 0.3, 0.85);
  const colorscale = ramp.map((c, i) => [i / (ramp.length - 1), c]);
  const x = [], y = [], stage = [], custom = [];
  let k = 0;
  for (const s of series) for (const it of s.its) {
    x.push(k++); y.push(it.res_norm); stage.push(s.stage);
    custom.push([s.stage, it.it, isNum(it.alpha) ? Number(it.alpha).toPrecision(3) : "–"]);
  }
  const lo = series[0].stage, hi = series[series.length - 1].stage;
  const dim = cssVar("--text-dim") || "#8b96a8";
  const traces = [{
    x, y, customdata: custom, type: "scatter", mode: "lines+markers", name: "residual", showlegend: false,
    line: { color: dark ? "rgba(139,150,168,0.45)" : "rgba(90,100,120,0.45)", width: 1.5 },
    marker: { size: 6, color: stage, colorscale, cmin: lo, cmax: Math.max(hi, lo + 1), showscale: true,
      colorbar: { title: { text: label, side: "right", font: { size: 12, color: dim } }, thickness: 10, len: 0.9, outlinewidth: 0,
        tickfont: { size: 11, color: dim }, nticks: 5 } },
    hovertemplate: `${label} %{customdata[0]}, iteration %{customdata[1]}<br>‖R‖ = %{y:.3e}<br>α = %{customdata[2]}<extra></extra>`,
  }];
  return plot(el, traces, baseLayout({ margin: { l: 58, r: 10, t: 8, b: 40 }, yaxis: { type: "log", title: { text: "residual ‖R‖" } },
    xaxis: { title: { text: "Newton iteration (all stages)" }, dtick: x.length < 16 ? 1 : undefined }, showlegend: false }),
    { displayModeBar: false });
}

/**
 * Time-dependent run, one point per accepted time step: final Newton residual (dots), the
 * Newton tolerance (dashed), Newton iterations per step (faint bars, right axis) and a ring on
 * steps that needed rejected attempts first. `steps` = summary.transient.steps.
 */
export function stepResidualPlot(el, steps, { tol = 1e-9, tPerTau = null } = {}) {
  const dark = document.documentElement.dataset.theme !== "light";
  const dot = sampleColors("cosmic", 3, dark ? 0.7 : 0.45, dark ? 0.9 : 0.65)[1];
  const warn = cssVar("--warn") || "#f2b134";
  const dim = cssVar("--text-dim") || "#8b96a8";
  const n = steps.map((_, i) => i + 1);
  const when = steps.map((st) => (tPerTau ? `τ = ${(st.t / tPerTau).toPrecision(3)}` : `t = ${Number(st.t).toPrecision(3)}`));
  const rej = steps.map((st, i) => [i, st]).filter(([, st]) => st.rejects > 0);
  const traces = [
    { x: n, y: steps.map((st) => st.newton_its), type: "bar", yaxis: "y2", name: "Newton iterations", marker: { color: dark ? "rgba(139,150,168,0.18)" : "rgba(90,100,120,0.16)" },
      customdata: when, hovertemplate: "step %{x} (%{customdata}): %{y} Newton iterations<extra></extra>" },
    { x: [1, steps.length], y: [tol, tol], type: "scatter", mode: "lines", name: "tolerance", line: { color: dim, width: 1.2, dash: "dash" },
      hovertemplate: `Newton tolerance ${tol.toExponential(0)}<extra></extra>` },
    { x: n, y: steps.map((st) => st.res_final), type: "scatter", mode: "markers", name: "final ‖R‖", marker: { size: 6, color: dot },
      customdata: steps.map((st, i) => [when[i], st.newton_its]), hovertemplate: "step %{x} (%{customdata[0]})<br>final ‖R‖ = %{y:.2e} after %{customdata[1]} iterations<extra></extra>" },
  ];
  if (rej.length) traces.push({ x: rej.map(([i]) => i + 1), y: rej.map(([, st]) => st.res_final), type: "scatter", mode: "markers", name: "had rejected attempts",
    marker: { size: 13, color: "rgba(0,0,0,0)", line: { color: warn, width: 2 } },
    customdata: rej.map(([, st]) => st.rejects), hovertemplate: "step %{x}: %{customdata} rejected attempt(s) before it was accepted<extra></extra>" });
  return plot(el, traces, baseLayout({ margin: { l: 58, r: 46, t: 8, b: 84 }, bargap: 0.25,
    yaxis: { type: "log", title: { text: "final residual ‖R‖" }, exponentformat: "power" },
    yaxis2: { overlaying: "y", side: "right", showgrid: false, rangemode: "tozero", dtick: 1, tickformat: "d", range: [0, Math.max(...steps.map((st) => st.newton_its)) * 2.2], title: { text: "iterations", font: { size: 11, color: dim } }, tickfont: { size: 10, color: dim } },
    xaxis: { title: { text: "accepted time step" } },
    showlegend: true, legend: { orientation: "h", y: -0.36, yanchor: "top", x: 0, font: { size: 11, color: dim }, bgcolor: "rgba(0,0,0,0)" } }), { displayModeBar: false });
}
