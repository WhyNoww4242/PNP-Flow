// Plotly theming shared by all charts (reads the CSS theme variables).

import { cssVar } from "./util.js";

// Ordered so neighbours stay distinguishable for common colour-vision deficiencies.
export const SERIES_COLORS = ["#4dabf7", "#ffa94d", "#b197fc", "#ffd43b", "#3bc9db", "#f783ac", "#69db7c", "#ff6b6b", "#a9e34b", "#868e96"];

/** Base layout with theme colors; merge `extra` (shallow per top-level key). */
export function baseLayout(extra = {}) {
  const fg = cssVar("--fg") || "#dde";
  const dim = cssVar("--fg-dim") || "#99a";
  const grid = cssVar("--grid") || "#2a2f3a";
  const font = cssVar("--font-sans") || "sans-serif";
  const axis = (a = {}) => Object.assign({
    gridcolor: grid, zerolinecolor: grid, linecolor: grid, tickcolor: grid,
    tickfont: { color: dim, size: 11, family: cssVar("--font-mono") || "monospace" },
    title: { font: { color: dim, size: 12 } }, automargin: true, exponentformat: "e", showexponent: "all",
  }, a);
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    font: { color: fg, family: font, size: 12 },
    margin: { l: 56, r: 16, t: 12, b: 44 },
    legend: { bgcolor: "rgba(0,0,0,0)", font: { size: 11, color: fg }, orientation: "h", y: -0.2 },
    hoverlabel: { font: { family: cssVar("--font-mono") || "monospace", size: 11 } },
    colorway: SERIES_COLORS,
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

/** Plotly.react if available; returns false (and shows nothing) if Plotly failed to load. */
export function plot(el, traces, layout, config = {}) {
  if (!window.Plotly) {
    el.textContent = "Plotly failed to load (static/vendor/plotly.min.js missing).";
    return false;
  }
  window.Plotly.react(el, traces, layout, Object.assign({}, PLOT_CONFIG, config));
  return true;
}

export function purge(el) {
  if (window.Plotly && el) window.Plotly.purge(el);
}

/** Resize all Plotly charts inside `root` (after tab switches / layout changes). */
export function resizeAll(root) {
  if (!window.Plotly) return;
  for (const el of root.querySelectorAll(".js-plotly-plot")) {
    if (el.offsetParent !== null) window.Plotly.Plots.resize(el);
  }
}
