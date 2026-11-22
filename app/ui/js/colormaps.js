// Colormaps for the field viewer and the colour bar.
//
// Source of truth: app/ui/data/colormaps.json, written by the post-processing step
// from cmasher:  {name: {"type": "sequential"|"diverging", "colors": [256 "#rrggbb"]}}.
// If that file is missing or malformed, a small embedded cmasher sample is used and
// `colormapSource().fallback` is true so the UI can say so.

import { bridge } from "./bridge.js";
import { FALLBACK } from "./colormaps_fallback.js";

const MAPS = new Map(); // name -> {type, lut: Uint8Array(1024)}
let SOURCE = { fallback: true, reason: "not loaded yet" };

function hexToBytes(hex) {
  const s = String(hex).replace("#", "");
  return [0, 2, 4].map((o) => parseInt(s.substr(o, 2), 16) || 0);
}

/** Build a 256-entry RGBA LUT from any number of colour stops (linear interpolation). */
function lutFromStops(stops) {
  const n = stops.length;
  const lut = new Uint8Array(256 * 4);
  for (let i = 0; i < 256; i++) {
    const t = n === 1 ? 0 : (i / 255) * (n - 1);
    const a = Math.min(Math.floor(t), Math.max(n - 2, 0));
    const f = n === 1 ? 0 : t - a;
    const b = Math.min(a + 1, n - 1);
    for (let c = 0; c < 3; c++) lut[i * 4 + c] = Math.round(stops[a][c] * (1 - f) + stops[b][c] * f);
    lut[i * 4 + 3] = 255;
  }
  return lut;
}

function loadFallback(reason) {
  MAPS.clear();
  for (const [name, m] of Object.entries(FALLBACK)) {
    const stops = [];
    for (let i = 0; i < m.stops.length / 6; i++) stops.push(hexToBytes(m.stops.substr(i * 6, 6)));
    MAPS.set(name, { type: m.type, lut: lutFromStops(stops) });
  }
  SOURCE = { fallback: true, reason };
}

/** Load colormaps.json once at start-up. Never throws. */
export async function initColormaps() {
  try {
    const data = await bridge.colormaps();
    const entries = Object.entries(data || {}).filter(([, m]) =>
      m && Array.isArray(m.colors) && m.colors.length >= 2 && (m.type === "sequential" || m.type === "diverging"));
    if (!entries.length) throw new Error("colormaps.json has no valid maps");
    MAPS.clear();
    for (const [name, m] of entries) {
      MAPS.set(name.replace(/^cmr\./, ""), { type: m.type, lut: lutFromStops(m.colors.map(hexToBytes)) });
    }
    SOURCE = { fallback: false, reason: "" };
  } catch (e) {
    loadFallback(e.message || "colormaps.json unavailable");
  }
  return SOURCE;
}

export function colormapSource() {
  return SOURCE;
}

export function colormapNames() {
  return [...MAPS.keys()];
}

export function colormapType(name) {
  const m = MAPS.get(name);
  return m ? m.type : "sequential";
}

/** Default map by field: ember for magnitudes, bubblegum for charge, cosmic otherwise. */
export function defaultColormap(signed, field = "") {
  // Only cosmic, ember and bubblegum (cmasher); never viridis, no diverging maps.
  const names = colormapNames();
  const byField = /^\|(u|E)\|$|vorticity/.test(field) ? "ember"
    : /^rho/.test(field) ? "bubblegum" : "cosmic";
  if (MAPS.has(byField)) return byField;
  if (MAPS.has("cosmic")) return "cosmic";
  return names.find((n) => colormapType(n) === "sequential") || names[0];
}

/** RGBA LUT (Uint8Array of 256*4), optionally reversed. */
export function colormapLUT(name, reversed = false) {
  const m = MAPS.get(name) || MAPS.get(defaultColormap(false));
  if (!reversed) return m.lut;
  const out = new Uint8Array(1024);
  for (let i = 0; i < 256; i++) out.set(m.lut.subarray((255 - i) * 4, (255 - i) * 4 + 4), i * 4);
  return out;
}

/** CSS linear-gradient for colour bars and previews. */
export function colormapCSS(name, direction = "to right", reversed = false) {
  const lut = colormapLUT(name, reversed);
  const parts = [];
  for (let i = 0; i <= 24; i++) {
    const j = Math.round((i / 24) * 255) * 4;
    parts.push(`rgb(${lut[j]},${lut[j + 1]},${lut[j + 2]}) ${((i / 24) * 100).toFixed(1)}%`);
  }
  return `linear-gradient(${direction}, ${parts.join(", ")})`;
}

/** n colours sampled from a map over [a, b] (for line plots in house style). */
export function sampleColors(name, n, a = 0.2, b = 0.75, reversed = false) {
  const lut = colormapLUT(name, reversed);
  const out = [];
  for (let i = 0; i < n; i++) {
    const t = n === 1 ? (a + b) / 2 : a + ((b - a) * i) / (n - 1);
    const j = Math.round(t * 255) * 4;
    out.push(`#${[lut[j], lut[j + 1], lut[j + 2]].map((v) => v.toString(16).padStart(2, "0")).join("")}`);
  }
  return out;
}
