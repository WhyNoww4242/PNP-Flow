// Boundary diagram for the Boundaries page: the named boundary curves of the mesh as SVG,
// coloured like the field viewer, each labelled. Hovering a curve or a table row
// highlights both; clicking selects the boundary for the condition editor (bceditor.js).
// Long thin domains are drawn stretched (said in the caption).

import { h, setChildren, placeholder, fmt } from "./util.js";
import { boundaryColor } from "./fieldview.js";
import { convert, scaleOf, thermalVoltage } from "./units.js";
import { bridge } from "./bridge.js";

const SVG = "http://www.w3.org/2000/svg";
const AXIS = /axis|symmetry/;
const PAD = 24, MIN_H = 150, MAX_H = 420;

function svgEl(tag, attrs) {
  const e = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  return e;
}

// {xy(i) -> [x, y], edges: {name: [[a, b], ...]}, axisym} from either source
function fromCaseMesh(m) {
  return { xy: (i) => m.nodes[i], edges: m.boundary_edges || {}, axisym: m.symmetry === "axisym" };
}
function fromFieldMesh(m) {
  const edges = {};
  for (const [name, flat] of Object.entries(m.boundaries || {})) {
    const pairs = [];
    for (let k = 0; k < flat.length; k += 2) if (flat[k] < m.n0 && flat[k + 1] < m.n0) pairs.push([flat[k], flat[k + 1]]);
    edges[name] = pairs;
  }
  return { xy: (i) => [m.xy[2 * i], m.xy[2 * i + 1]], edges, axisym: !!m.axisym };
}

async function loadGeometry(app, doc) {
  if (doc.kind === "case") {
    if (!doc.caseMesh) {
      if (!((doc.caseDetail || {}).mesh_files || []).includes("mesh/mesh.json")) return null;
      doc.caseMesh = await bridge.caseJSON(doc.caseName, "mesh/mesh.json");
    }
    return { g: fromCaseMesh(doc.caseMesh), source: "case mesh" };
  }
  const mesh = await app.loadMesh(doc);
  return mesh ? { g: fromFieldMesh(mesh), source: "mesh of this run" } : null;
}

// --- conditions set in the project (case.json "bcs", see runner.apply_case_bcs) ----------
// {boundary: {c: {type, value}, phi: {type, value, unit, coef, ref, sigma}}}. "blocking" and
// "natural" are the defaults, so they read as no condition.
const isCaseSpec = (spec) => !!spec && typeof spec === "object" && typeof spec.type === "string";

/** The conditions set in the project (a case) or frozen with the run. */
export function caseBCs(doc) {
  const c = doc.kind === "case" ? doc.caseData : doc.runCase;
  return (c && c.bcs) || {};
}

/** The template's conditions (as the solver wrote them) with the project's changes on top,
 *  and the names of the boundaries the project changed. */
export function effectiveBCs(doc) {
  const s = doc.summary || {};
  const base = s.template_boundaries || (s.model && s.model.boundaries) || {};
  const set = caseBCs(doc);
  const bcs = {};
  for (const [b, spec] of Object.entries(base)) bcs[b] = Object.assign({}, spec);
  for (const [b, fields] of Object.entries(set)) {
    const spec = (bcs[b] = Object.assign({}, bcs[b]));
    for (const [k, o] of Object.entries(fields || {})) {
      if (o.type === "blocking" || o.type === "natural") delete spec[k];
      else spec[k] = o;
    }
  }
  return { bcs, base, changed: new Set(Object.keys(set).filter((b) => Object.keys(set[b] || {}).length)) };
}

// a potential of a case spec: {lab: "75 mV" | null, dimless: "Φ = 2.919"}
function casePotential(o, key, ctx = {}) {
  const v = Number(o[key]);
  if (o.unit === "mV") {
    const phi = v / (1000 * thermalVoltage((ctx.inputs || {}).T));
    return { lab: `${fmt(v)} mV`, dimless: `Φ = ${fmt(phi)}` };
  }
  return { lab: null, dimless: `Φ = ${fmt(v)}` };
}
const potText = (o, key, ctx) => { const p = casePotential(o, key, ctx); return ctx && ctx.si !== false && p.lab ? `${p.lab} (${p.dimless})` : p.dimless; };
const potShort = (o, key, ctx) => { const p = casePotential(o, key, ctx); return ctx && ctx.si !== false && p.lab ? p.lab : p.dimless.replace(/^Φ = /, ""); };

function caseWords(key, o, ctx) {
  switch (`${key}:${o.type}`) {
    case "c:bulk": return "fixed at bulk concentration";
    case "c:fixed": return `fixed at ${fmt(Number(o.value))} × bulk concentration`;
    case "c:flux": return `set flux, N·n = ${fmt(Number(o.value))}`;
    case "phi:fixed": return Number(o.value) === 0 ? "fixed at 0 (grounded)" : `fixed at ${potText(o, "value", ctx)}`;
    case "phi:stern": return `Stern layer (C = ${fmt(Number(o.coef))}, reference ${potText(o, "ref", ctx)})`;
    case "phi:charge": return `surface charge σ = ${fmt(Number(o.sigma))}`;
    default: return NATURAL[key] || "natural";
  }
}

function caseShort(key, o, ctx) {
  const sym = key === "phi" ? "φ" : key;
  switch (`${key}:${o.type}`) {
    case "c:bulk": return "c = bulk";
    case "c:fixed": return `c = ${fmt(Number(o.value))} × bulk`;
    case "c:flux": return "c flux set";
    case "phi:fixed": return `${sym} = ${potShort(o, "value", ctx)}`;
    case "phi:stern": return "Stern layer";
    case "phi:charge": return "σ set";
    default: return NATURAL[key];
  }
}

// --- conditions in words. The solver writes them as "Dirichlet(value=bulk)", "Robin(coef=.., ref=..)",
// "blocking", ... (runner._bc_text); a species dict gives one entry per ion. A value that names a
// parameter is shown as the user entered it when an input `<name>_<unit>` exists (phi_w -> phi_w_mV),
// with the solver's dimensionless value beside it; in dimensionless display only the latter.
// ctx = { params: summary.params (dimensionless), inputs: {key: value}, si: bool, scales: units.scalesFor }
export const BC_QTY = { c: "Ions", phi: "Potential", u: "Flow" };
const NATURAL = { c: "no flux", phi: "no charge", u: "no-slip wall" };
const KEYWORD = { blocking: "no flux", outflow: "outflow (ions leave with the flow)", noslip: "no-slip wall", free: "traction free" };
const UNIT_SUFFIX = { mV: "mV", V: "V", nm: "nm", um: "µm", mM: "mM", M: "M", K: "K", C_m2: "C/m²" };
const SYM = { phi: "Φ" };

export function parseBC(text) {
  const m = /^(\w+)\((.*)\)$/.exec(String(text).trim());
  if (!m) return { type: String(text).trim().toLowerCase(), args: {} };
  const args = {};
  for (const part of m[2].split(/,\s*(?=\w+=)/)) { const [k, ...v] = part.split("="); if (k) args[k.trim()] = v.join("=").trim(); }
  return { type: m[1], args };
}

// {lab: "75 mV" | null, dimless: "Φ = 2.919" | null} for one value spec
export function valueParts(v, key, ctx = {}) {
  if (v === "bulk") return { lab: "bulk", dimless: null };
  const x = Number(v);
  if (v !== "" && v !== undefined && Number.isFinite(x)) return { lab: fmt(x), dimless: null };
  const inputs = ctx.inputs || {};
  const inKey = Object.keys(inputs).find((k) => k.startsWith(`${v}_`) && UNIT_SUFFIX[k.slice(v.length + 1)]);
  const lab = inKey !== undefined && Number.isFinite(Number(inputs[inKey])) ? `${fmt(Number(inputs[inKey]))} ${UNIT_SUFFIX[inKey.slice(v.length + 1)]}` : null;
  let d = ctx.params && typeof ctx.params[v] === "number" ? `${SYM[key] || v} = ${fmt(ctx.params[v])}` : null;
  // no run yet: scale the input itself (75 mV -> 75 mV / V_T)
  const sc = !d && inKey !== undefined && scaleOf(inKey, inKey.slice(v.length + 1), ctx.scales);
  if (sc) d = `${SYM[key] || v} = ${fmt(convert(Number(inputs[inKey]), inKey.slice(v.length + 1), sc.sym, ctx.scales), 4)}`;
  return { lab, dimless: d, name: v };
}

function valueText(v, key, ctx = {}) {
  const p = valueParts(v, key, ctx);
  if (p.name === undefined) return p.lab;
  if (ctx.si !== false && p.lab) return p.dimless ? `${p.lab} (${p.dimless})` : p.lab;
  return p.dimless || p.lab || p.name;
}

function valueShort(v, key, ctx = {}) {
  const p = valueParts(v, key, ctx);
  if (p.name === undefined) return p.lab;
  if (ctx.si !== false && p.lab) return p.lab;
  return p.dimless ? p.dimless.replace(/^.* = /, "") : p.lab || p.name;
}

/** One condition as words: key "c" | "phi" | "u", spec the solver text (or a per-ion object). */
export function bcWords(key, spec, ctx) {
  if (spec === undefined || spec === null) return NATURAL[key] || "natural";
  if (isCaseSpec(spec)) return caseWords(key, spec, ctx);
  if (typeof spec === "object") return Object.entries(spec).map(([ion, t]) => `${ion}: ${bcWords(key, t, ctx)}`).join("; ");
  const { type, args } = parseBC(spec);
  if (KEYWORD[type]) return KEYWORD[type];
  const v = (x) => valueText(x, key, ctx);
  switch (type) {
    case "Dirichlet": return key === "c" && args.value === "bulk" ? "fixed at bulk concentration"
      : key === "phi" && Number(args.value) === 0 ? "fixed at 0 (grounded)" : `fixed at ${v(args.value)}`;
    case "Flux": return `set flux, N·n = ${v(args.value)}`;
    case "Robin": return `Stern layer (C = ${valueText(args.coef, "", ctx)}, reference ${v(args.ref)})`;
    case "Charge": return `surface charge σ = ${valueText(args.sigma, "", ctx)}`;
    case "Component": return `${Number(args.index) === 0 ? "u_r" : "u_z"} = ${valueText(args.value ?? 0, "", ctx)}`;
    case "Pressure": return Number(args.p0 || 0) === 0 ? "open (p = 0)" : `pressure p = ${valueText(args.p0, "", ctx)}`;
    default: return String(spec);
  }
}

/** Short label for the diagram, ions then potential: "no flux, no charge", "no flux · φ = 75 mV", "c = bulk · φ = 0". */
export function bcShort(bc, ctx) {
  const part = (k) => {
    const spec = (bc || {})[k];
    if (spec === undefined || spec === null) return NATURAL[k];
    if (isCaseSpec(spec)) return caseShort(k, spec, ctx);
    if (typeof spec === "object") return `${k === "phi" ? "φ" : k} per ion`;
    const { type, args } = parseBC(spec);
    if (KEYWORD[type]) return type === "outflow" ? "outflow" : KEYWORD[type];
    const sym = k === "phi" ? "φ" : k;
    if (type === "Dirichlet") return `${sym} = ${valueShort(args.value, k, ctx)}`;
    if (type === "Flux") return `${sym} flux set`;
    if (type === "Robin") return "Stern layer";
    if (type === "Charge") return "σ set";
    return type;
  };
  const c = part("c"), phi = part("phi");
  return c === NATURAL.c && phi === NATURAL.phi ? "no flux, no charge" : `${c} · ${phi}`;
}

// Edges [[a, b], ...] -> node chains, so a dashed stroke runs continuously along a curve.
function chainsOf(edges) {
  const adj = new Map();
  const link = (a, b) => { if (!adj.has(a)) adj.set(a, []); adj.get(a).push(b); };
  for (const [a, b] of edges) { link(a, b); link(b, a); }
  const used = new Set(), key = (a, b) => (a < b ? `${a},${b}` : `${b},${a}`);
  const walk = (from) => {
    const ch = [from];
    for (let cur = from; ;) {
      const next = adj.get(cur).find((n) => !used.has(key(cur, n)));
      if (next === undefined) return ch;
      used.add(key(cur, next)); ch.push(next); cur = next;
    }
  };
  const chains = [];
  // start at curve ends (degree 1) first, then whatever is left (closed loops)
  for (const [n, nb] of adj) if (nb.length === 1 && nb.some((m) => !used.has(key(n, m)))) chains.push(walk(n));
  for (const [n, nb] of adj) if (nb.some((m) => !used.has(key(n, m)))) chains.push(walk(n));
  return chains;
}

function draw(g, names, bcs, W, ctx, editable) {
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  for (const name of names) for (const [a, b] of g.edges[name]) for (const i of [a, b]) {
    const [x, y] = g.xy(i);
    if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
  }
  // put the long side horizontal; stretch the short side to at least 1/4 of the box
  const swap = y1 - y0 > x1 - x0;
  const P = (i) => { const [x, y] = g.xy(i); return swap ? [y, x] : [x, y]; };
  const [u0, u1, v0, v1] = swap ? [y0, y1, x0, x1] : [x0, x1, y0, y1];
  const du = u1 - u0 || 1, dv = v1 - v0 || 1;
  const su = Math.min((W - 2 * PAD) / du, MAX_H / dv);
  const sv = Math.max(su, MIN_H / dv);
  const stretch = sv / su;
  const H = dv * sv + 2 * PAD + 16;
  const ou = (W - du * su) / 2, ov = PAD + 8 + dv * sv;
  const T = (i) => { const [u, v] = P(i); return [ou + (u - u0) * su, ov - (v - v0) * sv]; };
  const mid = ov - (dv * sv) / 2;

  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, class: "bd-svg", role: "img", "aria-label": `Named boundaries: ${names.join(", ")}` });
  const labels = [];
  names.forEach((name, i) => {
    const chains = chainsOf(g.edges[name]);
    let d = "", bx0 = Infinity, bx1 = -Infinity, by0 = Infinity, by1 = -Infinity;
    for (const ch of chains) ch.forEach((n, k) => {
      const [x, y] = T(n);
      d += `${k ? "L" : "M"}${x.toFixed(1)} ${y.toFixed(1)}`;
      if (x < bx0) bx0 = x; if (x > bx1) bx1 = x; if (y < by0) by0 = y; if (y > by1) by1 = y;
    });
    // label at the middle node of the longest chain, so it stays on the curve for L shapes
    const longest = chains.reduce((a, b) => (b.length > a.length ? b : a));
    const at = T(longest[longest.length >> 1]);
    const g1 = svgEl("g", { class: "bd-curve", "data-b": name, style: `--bc:${boundaryColor(name, i)}`,
      tabindex: "0", role: "button", "aria-label": `${name}: ${editable ? "change its condition" : "show its condition"}` });
    g1.appendChild(svgEl("path", { d, class: "bd-hit" }));
    g1.appendChild(svgEl("path", { d, class: AXIS.test(name) ? "bd-line dashed" : "bd-line" }));
    const title = svgEl("title", {}); title.textContent = name; g1.appendChild(title);
    svg.appendChild(g1);
    labels.push({ name, at, vertical: by1 - by0 > bx1 - bx0 });
  });
  // labels drawn last, above the curves: beside vertical curves (towards the middle),
  // otherwise above in the top half and below in the bottom half; nudged apart on collision
  // two lines per label: the name, then its conditions in short (bcShort)
  const placed = [];
  for (const L of labels) {
    let [x, y] = L.at, anchor = "middle";
    const up = !L.vertical && y <= mid;
    if (L.vertical) { anchor = x < W / 2 ? "start" : "end"; x += x < W / 2 ? 9 : -9; y -= 4; }
    else { y += up ? -24 : 18; x = Math.min(Math.max(x, PAD + 60), W - PAD - 60); }
    while (placed.some((p) => Math.abs(p[0] - x) < 110 && Math.abs(p[1] - y) < 30)) y += up ? -30 : 30;
    placed.push([x, y]);
    const t = svgEl("text", { x: x.toFixed(1), y: y.toFixed(1), class: "bd-label", "data-b": L.name, "text-anchor": anchor });
    t.textContent = L.name;
    const sub = svgEl("tspan", { x: x.toFixed(1), dy: "15", class: "bd-cond" });
    sub.textContent = AXIS.test(L.name) ? "symmetry axis" : bcShort(bcs[L.name], ctx);
    t.appendChild(sub);
    svg.appendChild(t);
  }
  const ys = placed.map((p) => p[1]);
  const top = Math.min(0, ...ys.map((y) => y - 14)), bot = Math.max(H, ...ys.map((y) => y + 21));
  svg.setAttribute("viewBox", `0 ${top.toFixed(0)} ${W} ${(bot - top).toFixed(0)}`);
  return { svg, stretch, swap };
}

/** Diagram element; fills itself in when the geometry arrives. `bcs`: effectiveBCs(doc).bcs; `ctx`: values for bcShort;
 *  `onHover(name|null)` links the table; `onSelect(name)` opens the condition editor; `editable` only changes the wording. */
export function boundaryDiagram(app, doc, names, bcs, ctx, { onHover, onSelect, editable = false }) {
  const box = h("div.bd", {}, placeholder("loading", "Loading the boundaries"));
  loadGeometry(app, doc).then((res) => {
    if (!res) return setChildren(box, placeholder("empty", "No mesh yet", doc.kind === "case" ? "Create the mesh to see where each boundary is." : "This run has no fields.json, so there is no geometry to draw."));
    const shown = names.filter((n) => res.g.edges[n] && res.g.edges[n].length);
    if (!shown.length) return setChildren(box, placeholder("empty", "No named boundary curves in the mesh"));
    const { svg, stretch, swap } = draw(res.g, shown, bcs, Math.max(560, Math.round(box.clientWidth || 1000)), ctx, editable);
    const coords = res.g.axisym ? ["r", "z"] : ["x", "y"];
    const across = swap ? coords[1] : coords[0];
    svg.addEventListener("pointerover", (e) => { const c = e.target.closest("[data-b]"); onHover(c ? c.dataset.b : null); });
    svg.addEventListener("pointerleave", () => onHover(null));
    const pick = (e) => { const c = e.target.closest("[data-b]"); if (c) onSelect(c.dataset.b); };
    svg.addEventListener("click", pick);
    svg.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(e); } });
    setChildren(box, svg, h("div.small.dim", { text: [
      `From the ${res.source}. ${across} runs left to right${res.g.axisym ? "; the dashed line is the symmetry axis" : ""}.`,
      stretch > 1.05 ? ` The short direction is stretched ${stretch < 10 ? stretch.toFixed(1) : Math.round(stretch)}× so thin parts stay visible.` : "",
      editable ? " Click a boundary to change its condition." : " Click a boundary to see its condition.",
    ].join("") }));
    onHover(null);                     // shows the selected boundary, if any
  }).catch((e) => setChildren(box, placeholder("error", "Could not load the mesh", e.message)));
  return box;
}

/** Highlight one boundary in the diagram and the table under `root`; with no name, the selected one (if any). */
export function highlightBoundary(root, name) {
  const sel = root.dataset.bsel || null;
  const on = name || sel;
  root.classList.toggle("bd-focus", !!on);
  for (const el of root.querySelectorAll("[data-b]")) {
    el.classList.toggle("hi", el.dataset.b === on);
    el.classList.toggle("sel", el.dataset.b === sel);
  }
}
