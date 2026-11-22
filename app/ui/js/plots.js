// Plotly theming from the DESIGN.md tokens (read from CSS variables at draw time).
// Line colours follow the house style: cmasher cosmic sampled over a sub-range
// (cosmic_r over 0.2–0.75 on light backgrounds; the lighter end on dark ones).

import { cssVar } from "./util.js";
import { sampleColors } from "./colormaps.js";

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
    el.textContent = "Charts are unavailable: vendor/plotly.min.js did not load.";
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
