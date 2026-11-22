// Shared helpers: DOM construction, number formatting, ticks, storage.
// Adapted from hud/static/js/util.js.

import { icon } from "./icons.js";

/** Create an element: h("div.card#id", {attrs}, ...children). */
export function h(spec, attrs = {}, ...children) {
  const m = spec.match(/^([a-z0-9]+)?((?:[.#][\w-]+)*)$/i);
  const el = document.createElement((m && m[1]) || "div");
  const rest = (m && m[2]) || "";
  for (const part of rest.match(/[.#][\w-]+/g) || []) {
    if (part[0] === ".") el.classList.add(part.slice(1));
    else el.id = part.slice(1);
  }
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") { if (v) el.className += " " + v; }
    else if (k === "text") el.textContent = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") {
      for (const [sk, sv] of Object.entries(v)) {
        if (sv === null || sv === undefined) continue;
        if (sk.startsWith("--")) el.style.setProperty(sk, String(sv));
        else el.style[sk] = sv;
      }
    }
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.appendChild(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

/** Replace all children of `el`. */
export function setChildren(el, ...children) {
  el.textContent = "";
  append(el, children);
  return el;
}

export const isNum = (v) => typeof v === "number" && Number.isFinite(v);

/** Compact number formatting for display (4 significant digits). */
export function fmt(v, digits = 4) {
  if (v === null || v === undefined) return "–";
  if (typeof v === "boolean") return v ? "true" : "false";
  if (typeof v !== "number") return String(v);
  if (!Number.isFinite(v)) return String(v);
  if (v === 0) return "0";
  const a = Math.abs(v);
  if (Number.isInteger(v) && a < 1e12) return v.toLocaleString("en-US").replace(/,/g, " ");
  if (a < 1e-3 || a >= 1e5) {
    const [m, e] = v.toExponential(digits - 1).split("e");
    return `${trimZeros(m)}e${e.replace("+", "")}`;
  }
  return trimZeros(v.toPrecision(digits));
}

function trimZeros(s) {
  return s.includes(".") ? s.replace(/\.?0+$/, "") : s;
}

export function fmtValue(v) {
  if (Array.isArray(v)) return "[" + v.map((x) => fmtValue(x)).join(", ") + "]";
  if (v && typeof v === "object") return JSON.stringify(v);
  return fmt(v);
}

export function fmtDuration(s) {
  if (!isNum(s)) return "–";
  if (s < 60) return `${s.toFixed(s < 10 ? 1 : 0)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
  return `${Math.floor(s / 3600)} h ${Math.round((s % 3600) / 60)} min`;
}

/** Calendar date + time for lists ("21 Nov 2026, 23:40"). */
export function fmtDate(epochSec) {
  if (!isNum(epochSec)) return "";
  const d = new Date(epochSec * 1000);
  return d.toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" }) +
    ", " + d.toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit" });
}

export function fmtBytes(n) {
  if (!isNum(n)) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1 << 20) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1 << 20)).toFixed(1)} MB`;
}

/** "Nice" linear ticks covering [a, b] with roughly n intervals. */
export function niceTicks(a, b, n = 5) {
  if (!(b > a)) return [a];
  const raw = (b - a) / n;
  const p = Math.pow(10, Math.floor(Math.log10(raw)));
  const m = raw / p;
  const step = (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
  const ticks = [];
  for (let t = Math.ceil(a / step - 1e-9) * step; t <= b + step * 1e-9; t += step) {
    ticks.push(Math.abs(t) < step * 1e-9 ? 0 : t);
  }
  return ticks;
}

/** Decade ticks for a log range given in log10 units [la, lb]. */
export function logTicks(la, lb, maxTicks = 7) {
  const lo = Math.ceil(la - 1e-9), hi = Math.floor(lb + 1e-9);
  const span = hi - lo + 1;
  const every = Math.max(1, Math.ceil(span / maxTicks));
  const out = [];
  for (let e = lo; e <= hi; e += every) out.push(e);
  if (out.length < 2) return niceTicks(la, lb, 3);
  return out;
}

export function debounce(fn, ms) {
  let t = 0;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

export const store = {
  get(key, dflt) {
    try {
      const v = localStorage.getItem("pnpflow." + key);
      return v === null ? dflt : JSON.parse(v);
    } catch {
      return dflt;
    }
  },
  set(key, v) {
    try {
      localStorage.setItem("pnpflow." + key, JSON.stringify(v));
    } catch { /* storage may be disabled */ }
  },
};

/** Read a CSS custom property of the document (theme colours for canvas / Plotly). */
export function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Empty / error / loading placeholder block. `action` is an optional button. */
export function placeholder(kind, title, detail, action) {
  const ic = { empty: "inbox", error: "alert", loading: null }[kind] || "inbox";
  return h(`div.placeholder.ph-${kind}`, { role: kind === "error" ? "alert" : null },
    kind === "loading" ? h("div.spinner", { "aria-hidden": "true" }) : h("div.ph-icon", {}, icon(ic, 20)),
    h("div.ph-title", { text: title }),
    detail ? h("div.ph-detail", {}, detail) : null,
    action || null);
}

/** Pretty label for a field / variable name: c+ -> c⁺ etc. */
export function prettyName(name) {
  return String(name)
    .replace(/^c\+$/, "c⁺").replace(/^c-$/, "c⁻")
    .replace(/_c\+/, " c⁺").replace(/_c-/, " c⁻")
    .replace(/^phi$/, "φ").replace(/^rho$/, "ρ")
    .replace(/^vorticity$/, "ω").replace(/^theta_deg$/, "θ (deg)");
}

/** Field names that are signed by nature (used together with the data range). */
export function isSignedFieldName(name) {
  return /^(phi|rho|vorticity|omega|uy|uz|ux|ur|Ey|Ez|Ex|Er|p|sigma|shear|N_.*_[xyzr])$/.test(name);
}

/** Unit in square brackets at the end of an info string: "wall potential [mV]" -> "mV". */
export function unitOf(info) {
  if (typeof info !== "string") return "";
  const m = info.match(/\[([^\]]{1,16})\]\s*$/) || info.match(/\[([^\]]{1,16})\]/);
  return m ? m[1] : "";
}

/** Info text without its trailing unit. */
export function stripUnit(info) {
  return typeof info === "string" ? info.replace(/\s*\[[^\]]{1,16}\]\s*$/, "") : "";
}

export function sameValue(a, b) {
  return JSON.stringify(a) === JSON.stringify(b);
}
