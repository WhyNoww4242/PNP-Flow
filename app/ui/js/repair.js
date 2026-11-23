// Mesh repair workspace (Mesh page, "Repair mesh"): a full-width, Field-map style editor.
// Left: the failed checks as a tree and the options of the active tool. Centre: toolbar,
// the mesh canvas and a status footer. Right (inside the canvas): the curvature colour bar.
//
// Tools: pan, box, disk, pen (freehand line | polygon | pick edges), spray, erase.
// Spray can put down three things: a fixed element size, a local density factor, or a size
// taken from the boundary curvature. Everything edits the case's `mesh` block
// (paint spots, regions, boundary refinements); geometry.apply_mesh_options turns it into
// gmsh size fields when the mesh is created again.
//
// Curvature: from the vertex chains of the named boundaries in mesh.json (turning angle over
// arc length). Turns sharper than CORNER_ANGLE are corners, not curvature: they are drawn as
// markers and left out of the colour range.

import { h, setChildren, fmt } from "./util.js";
import { icon } from "./icons.js";
import { bridge } from "./bridge.js";
import { boundaryColor } from "./fieldview.js";
import { colormapCSS, colormapLUT } from "./colormaps.js";

const CORNER_ANGLE = 0.5;          // rad (about 29 degrees)
const MAX_SPOTS = 6000;
const PEN_MODES = { line: "Freehand line", polygon: "Polygon", edges: "Pick edges" };
const SPRAY_WHAT = { size: "Element size", density: "Local density", curvature: "From curvature" };
const sig = (v, n = 6) => Number(v.toPrecision(n));

/** Boundary vertex curvature: { kappa: Map(node -> k), corners: [node], max, edges: [[a, b, k]] }. */
export function boundaryCurvature(geom) {
  const P = geom.nodes;
  const kappa = new Map();
  const corners = [];
  const segs = [];
  for (const edges of Object.values(geom.boundary_edges || {})) {
    const adj = new Map();
    for (const [a, b] of edges) {
      (adj.get(a) || adj.set(a, []).get(a)).push(b);
      (adj.get(b) || adj.set(b, []).get(b)).push(a);
    }
    for (const [v, nb] of adj) {
      if (nb.length !== 2) continue;
      const p = P[v], a = P[nb[0]], b = P[nb[1]];
      const ux = p[0] - a[0], uy = p[1] - a[1], wx = b[0] - p[0], wy = b[1] - p[1];
      const l1 = Math.hypot(ux, uy), l2 = Math.hypot(wx, wy);
      if (!l1 || !l2) continue;
      const ang = Math.abs(Math.atan2(ux * wy - uy * wx, ux * wx + uy * wy));
      if (ang > CORNER_ANGLE) { corners.push(v); continue; }
      if (ang < 1e-3) { kappa.set(v, 0); continue; }   // float noise on straight edges
      kappa.set(v, ang / (0.5 * (l1 + l2)));
    }
    for (const [a, b] of edges) segs.push([a, b]);
  }
  const vals = [...kappa.values()].filter((k) => k > 0).sort((x, y) => x - y);
  const max = vals.length ? vals[Math.min(vals.length - 1, Math.floor(0.98 * vals.length))] : 0;
  const edgesK = segs.map(([a, b]) => {
    const ka = kappa.get(a), kb = kappa.get(b);
    return [a, b, ka === undefined ? (kb === undefined ? null : kb) : kb === undefined ? ka : 0.5 * (ka + kb)];
  });
  return { kappa, corners, max, edges: edgesK };
}

/** A fixed-size hash of points for nearest lookups. */
function hashGrid(cell) {
  const m = new Map();
  const key = (i, j) => i * 73856093 ^ j * 19349663;
  return {
    add(x, y, v) { const k = key(Math.floor(x / cell), Math.floor(y / cell)); (m.get(k) || m.set(k, []).get(k)).push([x, y, v]); },
    nearest(x, y, rings = 2) {
      const i0 = Math.floor(x / cell), j0 = Math.floor(y / cell);
      let best = null, bd = Infinity;
      for (let i = i0 - rings; i <= i0 + rings; i++) for (let j = j0 - rings; j <= j0 + rings; j++) {
        for (const q of m.get(key(i, j)) || []) {
          const d = Math.hypot(q[0] - x, q[1] - y);
          if (d < bd) { bd = d; best = q; }
        }
      }
      return best ? { x: best[0], y: best[1], v: best[2], d: bd } : null;
    },
  };
}

function segDist(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / (dx * dx + dy * dy || 1)));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function inPolygon(x, y, pts) {
  let c = false;
  for (let i = 0, j = pts.length - 1; i < pts.length; j = i++) {
    const [xi, yi] = pts[i], [xj, yj] = pts[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) c = !c;
  }
  return c;
}

/**
 * ctx: { mesh, save, defaultSize, numInput, field, touched, leftSlot, status }
 *   touched() re-draws the panels of the ordinary mesh page after an edit here.
 */
export function repairWorkspace(app, doc, ctx) {
  const { mesh, save, defaultSize, numInput, field, touched, leftSlot } = ctx;
  const rp = (doc._rp ||= { tool: "pan", pen: "line", spray: "size", heat: false, r: null, size: null, density: 2, penWidth: null, nPerRadius: 8, minSize: null });
  const view = (doc.rpView ||= { cx: 0, cy: 0, k: 0 });
  let geom = null, curv = null, cells = null, bgrid = null;
  let drag = null, hover = null, poly = [], line = [], hoverEdge = null;

  const canvas = h("canvas.mt-canvas", { tabindex: "0", "aria-label": "Mesh repair canvas" });
  const hint = h("div.mt-hint");
  const readout = h("span.fv-readout.dim", { text: "Move over the mesh" });
  const legend = h("div.rp-cbar", { hidden: true });
  const stage = h("div.rp-stage", {}, canvas, legend, hint);

  const css = (n) => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const toScreen = (x, y) => [(x - view.cx) * view.k + canvas.width / 2, canvas.height / 2 - (y - view.cy) * view.k];
  const toWorld = (sx, sy) => [(sx - canvas.width / 2) / view.k + view.cx, view.cy - (sy - canvas.height / 2) / view.k];
  const pos = (e) => { const r = canvas.getBoundingClientRect(); const d = canvas.width / r.width; return [(e.clientX - r.left) * d, (e.clientY - r.top) * d]; };
  const fit = () => {
    const bb = geom.bbox, W = canvas.width, H = canvas.height;
    view.cx = (bb[0] + bb[1]) / 2; view.cy = (bb[2] + bb[3]) / 2;
    view.k = 0.92 * Math.min(W / Math.max(1e-12, bb[1] - bb[0]), H / Math.max(1e-12, bb[3] - bb[2]));
  };
  const brushR = () => rp.r || 0.03 * Math.min(canvas.width, canvas.height) / view.k;
  const spotSize = () => rp.size || defaultSize();
  const changed = () => { save(); touched(); };

  // ---- geometry derived data ----
  const prepare = () => {
    const P = geom.nodes, bb = geom.bbox;
    const diag = Math.hypot(bb[1] - bb[0], bb[3] - bb[2]) || 1;
    // mean element edge length per cell (local h)
    cells = hashGrid(diag / 120);
    const acc = new Map();
    const cell = diag / 120, key = (i, j) => i * 73856093 ^ j * 19349663;
    for (const t of geom.triangles) {
      const a = P[t[0]], b = P[t[1]], c = P[t[2]];
      const hh = (Math.hypot(a[0] - b[0], a[1] - b[1]) + Math.hypot(b[0] - c[0], b[1] - c[1]) + Math.hypot(c[0] - a[0], c[1] - a[1])) / 3;
      const k = key(Math.floor(((a[0] + b[0] + c[0]) / 3) / cell), Math.floor(((a[1] + b[1] + c[1]) / 3) / cell));
      const s = acc.get(k) || acc.set(k, { x: (a[0] + b[0] + c[0]) / 3, y: (a[1] + b[1] + c[1]) / 3, s: 0, n: 0 }).get(k);
      s.s += hh; s.n++;
    }
    for (const s of acc.values()) cells.add(s.x, s.y, s.s / s.n);
    curv = boundaryCurvature(geom);
    bgrid = hashGrid(diag / 120);
    for (const [v, k] of curv.kappa) bgrid.add(P[v][0], P[v][1], k);
    legendUpdate();
  };
  const localH = (x, y) => { const q = cells && cells.nearest(x, y, 3); return q ? q.v : defaultSize(); };
  const curvatureAt = (x, y, within) => { const q = bgrid && bgrid.nearest(x, y, 3); return q && q.d <= within ? q : null; };

  // ---- legend (curvature colour bar) ----
  function legendUpdate() {
    legend.hidden = !rp.heat || !curv;
    if (legend.hidden) return;
    const flat = !(curv.max > 0);
    setChildren(legend,
      h("div.cbar-name", { text: "κ" }),
      h("div.cbar-label", { text: "boundary curvature (1 / length)" }),
      h("div.cbar-end", { text: flat ? "max 0" : `max ${curv.max.toPrecision(3)}` }),
      h("div.rp-cbar-grad", { style: { background: colormapCSS("ember", "to top") } }),
      h("div.cbar-end", { text: "min 0" }),
      h("div.cbar-label", { text: flat ? "Every boundary is straight here." : `Smallest radius ${(1 / curv.max).toPrecision(3)}. ${curv.corners.length} corner vertices are left out (diamonds).` }));
  }

  // ---- drawing ----
  let pending = false;
  const draw = () => { if (pending) return; pending = true; requestAnimationFrame(() => { pending = false; paint(); }); };
  function paint() {
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    if (!rect.width) return;
    if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
      const oldW = canvas.width;
      canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
      if (view.k && oldW) view.k *= canvas.width / oldW;
    }
    const g = canvas.getContext("2d");
    g.fillStyle = css("--plot-bg"); g.fillRect(0, 0, canvas.width, canvas.height);
    if (!geom) return;
    if (!view.k) fit();
    const P = geom.nodes, W = canvas.width, H = canvas.height, k = view.k;
    const ox = W / 2 - view.cx * k, oy = H / 2 + view.cy * k, tiny = 2 * dpr;
    g.lineWidth = 1;
    g.strokeStyle = rp.heat ? css("--border") : css("--border-strong");
    g.fillStyle = g.strokeStyle;
    g.beginPath();
    for (const t of geom.triangles) {
      const a = P[t[0]], b = P[t[1]], c = P[t[2]];
      const ax = a[0] * k + ox, ay = oy - a[1] * k, bx = b[0] * k + ox, by = oy - b[1] * k, cx = c[0] * k + ox, cy = oy - c[1] * k;
      const x0 = Math.min(ax, bx, cx), x1 = Math.max(ax, bx, cx), y0 = Math.min(ay, by, cy), y1 = Math.max(ay, by, cy);
      if (x1 < 0 || y1 < 0 || x0 > W || y0 > H) continue;
      if (x1 - x0 < tiny && y1 - y0 < tiny) { g.fillRect(x0 | 0, y0 | 0, 1, 1); continue; }
      g.moveTo(ax, ay); g.lineTo(bx, by); g.lineTo(cx, cy); g.closePath();
    }
    g.stroke();
    // boundaries: by group colour, or by curvature (heat map)
    const selected = new Set(mesh.boundaries.map((b) => b.boundary));
    if (rp.heat && curv) {
      const lut = colormapLUT("ember");
      g.lineWidth = 4 * dpr; g.lineCap = "round";
      const bins = Array.from({ length: 48 }, () => []);
      for (const e of curv.edges) {
        const b = e[2] === null || !(curv.max > 0) ? 0 : Math.min(47, Math.floor(47 * Math.min(1, e[2] / curv.max)));
        bins[b].push(e);
      }
      bins.forEach((list, bi) => {
        if (!list.length) return;
        const j = Math.round((bi / 47) * 255) * 4;
        g.strokeStyle = `rgb(${lut[j]},${lut[j + 1]},${lut[j + 2]})`;
        g.beginPath();
        for (const [a, b] of list) { const [x0, y0] = toScreen(P[a][0], P[a][1]); const [x1, y1] = toScreen(P[b][0], P[b][1]); g.moveTo(x0, y0); g.lineTo(x1, y1); }
        g.stroke();
      });
      g.lineCap = "butt";
      g.fillStyle = css("--warn");
      for (const v of curv.corners) {
        const [x, y] = toScreen(P[v][0], P[v][1]), a = 5 * dpr;
        g.beginPath(); g.moveTo(x, y - a); g.lineTo(x + a, y); g.lineTo(x, y + a); g.lineTo(x - a, y); g.closePath(); g.fill();
      }
    } else {
      Object.entries(geom.boundary_edges || {}).forEach(([name, edges], i) => {
        g.lineWidth = (selected.has(name) ? 4.5 : 2.5) * dpr;
        g.strokeStyle = boundaryColor(name, i);
        g.setLineDash(/axis|symmetry/.test(name) ? [6 * dpr, 4 * dpr] : []);
        g.beginPath();
        for (const [a, b] of edges) { const [x0, y0] = toScreen(P[a][0], P[a][1]); const [x1, y1] = toScreen(P[b][0], P[b][1]); g.moveTo(x0, y0); g.lineTo(x1, y1); }
        g.stroke();
      });
      g.setLineDash([]);
    }
    // refinements
    const acc = css("--accent");
    g.lineWidth = 2 * dpr; g.strokeStyle = acc; g.fillStyle = css("--accent-soft");
    for (const r of mesh.regions) {
      g.beginPath();
      if (r.kind === "box") {
        const [x0, y0] = toScreen(Math.min(r.x0, r.x1), Math.max(r.y0, r.y1));
        const [x1, y1] = toScreen(Math.max(r.x0, r.x1), Math.min(r.y0, r.y1));
        g.rect(x0, y0, x1 - x0, y1 - y0);
      } else { const [x, y] = toScreen(r.x, r.y); g.arc(x, y, Math.max(2, (r.r || 0) * k), 0, 2 * Math.PI); }
      g.fill(); g.stroke();
    }
    for (const s of mesh.paint) { const [x, y] = toScreen(s.x, s.y); g.beginPath(); g.arc(x, y, Math.max(1.5, s.r * k), 0, 2 * Math.PI); g.fill(); }
    // hovered boundary group (pick edges)
    if (rp.tool === "pen" && rp.pen === "edges" && hoverEdge) {
      g.strokeStyle = acc; g.lineWidth = 6 * dpr; g.globalAlpha = 0.6;
      g.beginPath();
      for (const [a, b] of geom.boundary_edges[hoverEdge] || []) { const [x0, y0] = toScreen(P[a][0], P[a][1]); const [x1, y1] = toScreen(P[b][0], P[b][1]); g.moveTo(x0, y0); g.lineTo(x1, y1); }
      g.stroke(); g.globalAlpha = 1;
    }
    // previews
    g.strokeStyle = acc; g.setLineDash([5 * dpr, 4 * dpr]); g.lineWidth = 2 * dpr;
    if (drag && drag.tool === "box") g.strokeRect(drag.x0, drag.y0, drag.x1 - drag.x0, drag.y1 - drag.y0);
    if (drag && drag.tool === "disk") { g.beginPath(); g.arc(drag.x0, drag.y0, Math.hypot(drag.x1 - drag.x0, drag.y1 - drag.y0), 0, 2 * Math.PI); g.stroke(); }
    const trace = rp.pen === "polygon" ? poly : line;
    if (rp.tool === "pen" && trace.length) {
      g.setLineDash([]); g.beginPath();
      trace.forEach(([x, y], i) => { const [sx, sy] = toScreen(x, y); i ? g.lineTo(sx, sy) : g.moveTo(sx, sy); });
      if (rp.pen === "polygon" && hover) g.lineTo(hover[0], hover[1]);
      g.stroke();
      if (rp.pen === "polygon") for (const [x, y] of poly) { const [sx, sy] = toScreen(x, y); g.beginPath(); g.arc(sx, sy, 4 * dpr, 0, 2 * Math.PI); g.fillStyle = acc; g.fill(); }
    }
    g.setLineDash([]);
    const brushy = rp.tool === "spray" || rp.tool === "erase" || (rp.tool === "pen" && rp.pen !== "edges");
    if (hover && brushy) {
      g.strokeStyle = rp.tool === "erase" ? css("--bad") : acc;
      g.beginPath(); g.arc(hover[0], hover[1], (rp.tool === "pen" ? penR() : brushR()) * k, 0, 2 * Math.PI); g.stroke();
    }
  }
  const penR = () => rp.penWidth || brushR();

  // ---- editing helpers ----
  const addSpot = (x, y, r, size, src) => {
    if (mesh.paint.length >= MAX_SPOTS) return false;
    mesh.paint.push({ x: sig(x), y: sig(y), r: sig(r, 4), size: sig(size, 4), src });
    return true;
  };
  const sprayAt = (sx, sy) => {
    const [x, y] = toWorld(sx, sy), r = brushR();
    const last = mesh.paint[mesh.paint.length - 1];
    if (last && last.src === "spray:" + rp.spray && Math.hypot(last.x - x, last.y - y) < 0.5 * r) return;
    let size;
    if (rp.spray === "size") size = spotSize();
    else if (rp.spray === "density") size = localH(x, y) / Math.max(1.01, rp.density);
    else {
      const q = curvatureAt(x, y, 3 * r);
      if (!q || !(q.v > 1e-12)) { status("No curved boundary within reach of the brush: nothing sprayed."); return; }
      size = Math.max(rp.minSize || 0, 1 / (q.v * Math.max(2, rp.nPerRadius)));
    }
    if (!addSpot(x, y, r, size, "spray:" + rp.spray)) status(`Limit of ${MAX_SPOTS} spots reached: erase some or use a box/disk region.`);
    drawCount();
  };
  const eraseAt = (sx, sy) => {
    const [x, y] = toWorld(sx, sy), r = brushR();
    const n0 = mesh.paint.length + mesh.regions.length;
    mesh.paint = mesh.paint.filter((s) => Math.hypot(s.x - x, s.y - y) > r + s.r);
    mesh.regions = mesh.regions.filter((q) => {
      const cx = q.kind === "box" ? (q.x0 + q.x1) / 2 : q.x, cy = q.kind === "box" ? (q.y0 + q.y1) / 2 : q.y;
      return Math.hypot(cx - x, cy - y) > r;
    });
    if (mesh.paint.length + mesh.regions.length !== n0) drawCount();
  };
  const commitLine = () => {
    if (line.length < 2) { line = []; return; }
    const r = penR(), step = Math.max(r * 0.6, 1e-12), size = spotSize();
    let n = 0, ok = true;
    for (let i = 1; i < line.length && ok; i++) {
      const [x0, y0] = line[i - 1], [x1, y1] = line[i], L = Math.hypot(x1 - x0, y1 - y0);
      const m = Math.max(1, Math.ceil(L / step));
      for (let j = 0; j < m && ok; j++) { ok = addSpot(x0 + ((x1 - x0) * j) / m, y0 + ((y1 - y0) * j) / m, r, size, "pen:line"); n++; }
    }
    addSpot(line[line.length - 1][0], line[line.length - 1][1], r, size, "pen:line");
    line = [];
    if (!ok) status(`Limit of ${MAX_SPOTS} spots reached: the line was cut short.`);
    changed(); drawCount();
  };
  const commitPolygon = () => {
    if (poly.length < 3) { poly = []; draw(); return; }
    const xs = poly.map((p) => p[0]), ys = poly.map((p) => p[1]);
    const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
    const r = penR(), step = r * 1.2, size = spotSize();
    let ok = true;
    for (let y = y0; y <= y1 + step && ok; y += step) for (let x = x0; x <= x1 + step && ok; x += step) {
      if (inPolygon(x, y, poly)) ok = addSpot(x, y, r, size, "pen:polygon");
    }
    for (let i = 0; i < poly.length && ok; i++) {
      const [ax, ay] = poly[i], [bx, by] = poly[(i + 1) % poly.length], L = Math.hypot(bx - ax, by - ay), m = Math.max(1, Math.ceil(L / step));
      for (let j = 0; j < m && ok; j++) ok = addSpot(ax + ((bx - ax) * j) / m, ay + ((by - ay) * j) / m, r, size, "pen:polygon");
    }
    poly = [];
    if (!ok) status(`Limit of ${MAX_SPOTS} spots reached: use a larger pen width or a box/disk region.`);
    changed(); drawCount(); draw();
  };
  const edgeAt = (sx, sy) => {
    const [x, y] = toWorld(sx, sy), P = geom.nodes;
    let best = null, bd = Infinity;
    for (const [name, edges] of Object.entries(geom.boundary_edges || {})) {
      for (const [a, b] of edges) {
        const d = segDist(x, y, P[a][0], P[a][1], P[b][0], P[b][1]);
        if (d < bd) { bd = d; best = name; }
      }
    }
    return best && bd * view.k < 14 * (window.devicePixelRatio || 1) ? best : null;
  };
  const toggleEdge = (name) => {
    const i = mesh.boundaries.findIndex((b) => b.boundary === name);
    if (i >= 0) { mesh.boundaries.splice(i, 1); status(`Refinement on ${name} removed.`); }
    else {
      mesh.boundaries.push({ boundary: name, size: spotSize(), layer: sig(Math.max(spotSize() * 6, 0), 3), growth: 1.2 });
      status(`${name} refined to size ${spotSize()} (edit the layer on the Mesh page).`);
    }
    changed(); drawCount(); drawOptions(); draw();
  };
  const curvatureAll = () => {
    if (!curv || !(curv.max > 0)) { app.toast("No curved boundaries to refine.", "info"); return; }
    const P = geom.nodes, N = Math.max(2, rp.nPerRadius), r0 = brushR();
    const thr = 0.25 * curv.max, taken = hashGrid(r0);
    let n = 0;
    for (const [v, kv] of curv.kappa) {
      if (kv < thr) continue;
      const x = P[v][0], y = P[v][1];
      if (taken.nearest(x, y, 1) && taken.nearest(x, y, 1).d < r0) continue;
      taken.add(x, y, 1);
      if (!addSpot(x, y, r0, Math.max(rp.minSize || 0, 1 / (kv * N)), "curvature")) break;
      n++;
    }
    changed(); drawCount();
    app.toast(n ? `${n} curvature spots added (size = radius / ${N}). Create the mesh to apply them.` : "Nothing above the curvature threshold.", n ? "ok" : "info");
    draw();
  };

  // ---- pointer ----
  const HINTS = {
    pan: "Pan: drag to move, wheel to zoom, double-click to fit.",
    box: "Box: drag a rectangle; elements inside get the size set in Options.",
    disk: "Disk: press at the centre and drag out the radius.",
    "pen:line": "Pen, freehand line: drag a line; it refines a strip of the pen width on release.",
    "pen:polygon": "Pen, polygon: click the corners; double-click or Enter closes and fills it. Esc cancels.",
    "pen:edges": "Pen, pick edges: click a boundary to refine it along its whole length; click again to remove.",
    spray: "Spray: drag to paint. What it paints is the Spray menu: size, local density or curvature.",
    erase: "Erase: drag over painted spots or regions to remove them.",
  };
  const setHint = () => { hint.textContent = HINTS[rp.tool === "pen" ? `pen:${rp.pen}` : rp.tool]; canvas.dataset.tool = rp.tool === "pen" && rp.pen === "edges" ? "box" : rp.tool; };
  function status(t) { hint.textContent = t; }
  canvas.addEventListener("pointerdown", (e) => {
    if (!geom) return;
    canvas.focus();
    canvas.setPointerCapture(e.pointerId);
    const [sx, sy] = pos(e);
    const tool = e.button === 1 ? "pan" : rp.tool;
    drag = { tool, x0: sx, y0: sy, x1: sx, y1: sy, cx: view.cx, cy: view.cy, moved: false };
    if (tool === "spray") sprayAt(sx, sy);
    if (tool === "erase") eraseAt(sx, sy);
    if (tool === "pen" && rp.pen === "line") line = [toWorld(sx, sy)];
    draw();
  });
  canvas.addEventListener("pointermove", (e) => {
    const [sx, sy] = pos(e);
    hover = [sx, sy];
    const [wx, wy] = toWorld(sx, sy);
    updateReadout(wx, wy);
    if (rp.tool === "pen" && rp.pen === "edges" && geom) hoverEdge = edgeAt(sx, sy);
    if (drag) {
      drag.x1 = sx; drag.y1 = sy;
      if (Math.hypot(sx - drag.x0, sy - drag.y0) > 3) drag.moved = true;
      if (drag.tool === "pan") { view.cx = drag.cx - (sx - drag.x0) / view.k; view.cy = drag.cy + (sy - drag.y0) / view.k; }
      if (drag.tool === "spray") sprayAt(sx, sy);
      if (drag.tool === "erase") eraseAt(sx, sy);
      if (drag.tool === "pen" && rp.pen === "line") {
        const l = line[line.length - 1];
        if (!l || Math.hypot(l[0] - wx, l[1] - wy) * view.k > 4) line.push([wx, wy]);
      }
    }
    draw();
  });
  canvas.addEventListener("pointerleave", () => { hover = null; hoverEdge = null; draw(); });
  canvas.addEventListener("pointerup", () => {
    if (!drag) return;
    const d = drag; drag = null;
    const [x0, y0] = toWorld(d.x0, d.y0), [x1, y1] = toWorld(d.x1, d.y1);
    if (d.tool === "box" && Math.abs(d.x1 - d.x0) > 4 && Math.abs(d.y1 - d.y0) > 4) {
      mesh.regions.push({ kind: "box", x0: sig(Math.min(x0, x1)), x1: sig(Math.max(x0, x1)), y0: sig(Math.min(y0, y1)), y1: sig(Math.max(y0, y1)), size: spotSize(), growth: 1.2 });
      changed(); drawCount();
    } else if (d.tool === "disk" && Math.hypot(d.x1 - d.x0, d.y1 - d.y0) > 4) {
      mesh.regions.push({ kind: "disk", x: sig(x0), y: sig(y0), r: sig(Math.hypot(x1 - x0, y1 - y0)), size: spotSize(), growth: 1.2 });
      changed(); drawCount();
    } else if (d.tool === "spray" || d.tool === "erase") changed();
    else if (d.tool === "pen") {
      if (rp.pen === "line") commitLine();
      else if (rp.pen === "polygon" && !d.moved) { const [sx, sy] = [d.x1, d.y1]; poly.push(toWorld(sx, sy)); }
      else if (rp.pen === "edges" && !d.moved) { const n = edgeAt(d.x1, d.y1); if (n) toggleEdge(n); }
    }
    draw();
  });
  canvas.addEventListener("dblclick", () => {
    if (!geom) return;
    if (rp.tool === "pen" && rp.pen === "polygon") { poly.pop(); commitPolygon(); return; }
    if (rp.tool === "pan") { fit(); draw(); }
  });
  canvas.addEventListener("keydown", (e) => {
    if (e.key === "Escape") { poly = []; line = []; draw(); }
    if (e.key === "Enter" && rp.tool === "pen" && rp.pen === "polygon") commitPolygon();
  });
  canvas.addEventListener("wheel", (e) => {
    if (!geom) return;
    e.preventDefault();
    const [sx, sy] = pos(e), [wx, wy] = toWorld(sx, sy);
    view.k *= Math.exp(-e.deltaY * 0.0015);
    const [nx, ny] = toWorld(sx, sy);
    view.cx += wx - nx; view.cy += wy - ny;
    draw();
  }, { passive: false });
  new ResizeObserver(() => draw()).observe(canvas);

  function updateReadout(x, y) {
    if (!geom) return;
    const q = bgrid && bgrid.nearest(x, y, 2);
    setChildren(readout,
      h("span.k", { text: "x " }), h("span.v", { text: fmt(sig(x, 4)) }), h("span.k", { text: "  y " }), h("span.v", { text: fmt(sig(y, 4)) }),
      h("span.k", { text: "   element size " }), h("span.v", { text: fmt(sig(localH(x, y), 3)) }),
      q ? [h("span.k", { text: "   κ nearby " }), h("span.v", { text: q.v.toPrecision(3) })] : null);
    readout.classList.remove("dim");
  }

  // ---- toolbar ----
  const toolBtn = (id, label, ic, extra = {}) => h("button.tool", Object.assign({ type: "button", "data-tip-title": label, "aria-label": label,
    "aria-pressed": String(rp.tool === id), "data-rp-tool": id,
    onclick: () => { rp.tool = id; poly = []; line = []; syncBar(); setHint(); drawOptions(); draw(); } }, extra), icon(ic, 18));
  const penSel = h("select.sel.rp-sel", { "aria-label": "Pen mode", "data-tip": "What the pen does", onchange: (e) => { rp.pen = e.target.value; rp.tool = "pen"; poly = []; line = []; syncBar(); setHint(); drawOptions(); draw(); } },
    Object.entries(PEN_MODES).map(([v, t]) => h("option", { value: v, text: t, selected: v === rp.pen || null })));
  const spraySel = h("select.sel.rp-sel", { "aria-label": "What the spray puts down", "data-tip": "What the spray paints: a fixed size, a local density factor, or a size from the boundary curvature",
    onchange: (e) => { rp.spray = e.target.value; rp.tool = "spray"; syncBar(); setHint(); drawOptions(); draw(); } },
    Object.entries(SPRAY_WHAT).map(([v, t]) => h("option", { value: v, text: t, selected: v === rp.spray || null })));
  const heatBtn = h("button.tool", { type: "button", "aria-label": "Curvature heat map", "data-tip-title": "Curvature heat map", "data-tip": "Colour the boundaries by curvature (ember map). Straight edges are dark.",
    "aria-pressed": String(rp.heat), onclick: () => { rp.heat = !rp.heat; syncBar(); legendUpdate(); drawOptions(); draw(); } }, icon("curvature", 18));
  const count = h("span.small.dim");
  const drawCount = () => { count.textContent = `${mesh.paint.length} spots, ${mesh.regions.length} regions, ${mesh.boundaries.length} boundaries`; touched(); };
  const bar = h("div.fv-bar", {},
    toolBtn("pan", "Pan and zoom", "hand"), h("span.tool-sep"),
    toolBtn("box", "Refinement box", "box-region"), toolBtn("disk", "Refinement disk", "disk-region"), h("span.tool-sep"),
    toolBtn("pen", "Pen", "pen"), penSel, h("span.tool-sep"),
    toolBtn("spray", "Spray", "spray"), spraySel, toolBtn("erase", "Erase", "eraser"), h("span.tool-sep"),
    heatBtn, h("button.tool", { type: "button", "aria-label": "Fit to window", "data-tip-title": "Fit to window", onclick: () => { if (geom) { fit(); draw(); } } }, icon("fit", 18)),
    h("span.spacer"), count);
  function syncBar() {
    for (const b of bar.querySelectorAll("[data-rp-tool]")) b.setAttribute("aria-pressed", String(b.dataset.rpTool === rp.tool));
    heatBtn.setAttribute("aria-pressed", String(rp.heat));
    penSel.value = rp.pen; spraySel.value = rp.spray;
  }

  // ---- options (left column, under the checks) ----
  const optSlot = h("div");
  function drawOptions() {
    const dens = numInput(mesh.density, (v) => { mesh.density = v; changed(); }, { label: "Global density", min: 0 });
    const rows = [];
    rows.push(h("div.mt-row", {}, field("Global density ×", dens, "1 = template sizes; 2 = every element half as big (≈ 4× elements)")));
    rows.push(h("div.mt-row", {},
      field("Brush / pen width", numInput(rp.r ?? "", (v) => { rp.r = v; rp.penWidth = v; draw(); }, { label: "Brush radius", placeholderText: "auto" }), "Model length units (blank = 3 % of the view)"),
      field("Element size", numInput(rp.size ?? "", (v) => { rp.size = v; }, { label: "Element size", placeholderText: "auto" }), "Size inside sprayed spots, boxes, disks, the pen and picked edges (blank = half the smallest element)")));
    if (rp.tool === "spray" && rp.spray === "density") rows.push(h("div.mt-row", {}, field("Local density ×", numInput(rp.density, (v) => { rp.density = v; }, { label: "Local density factor", min: 1 }), "Sprayed spots get the local element size divided by this factor")));
    if (rp.spray === "curvature" || rp.heat || rp.tool === "spray") {
      rows.push(h("div.mt-row", {},
        field("Elements per radius", numInput(rp.nPerRadius, (v) => { rp.nPerRadius = v; }, { label: "Elements per radius of curvature", min: 1 }), "Size = radius of curvature / this number"),
        field("Smallest size", numInput(rp.minSize ?? "", (v) => { rp.minSize = v; }, { label: "Smallest curvature size", placeholderText: "none" }), "Floor for curvature-based sizes")));
      rows.push(h("div.mt-row", {}, h("button.btn.sm", { type: "button", disabled: !curv || !(curv.max > 0) || null, onclick: curvatureAll, "data-tip": "Add spots along every boundary where curvature is above a quarter of the maximum" },
        icon("curvature", 14), "Refine along curved boundaries")));
    }
    rows.push(h("div.mt-row", {}, h("button.btn.sm", { type: "button", onclick: () => {
      mesh.paint = []; mesh.regions = []; mesh.boundaries = []; changed(); drawCount(); drawOptions(); draw(); }, "data-tip": "Remove every refinement of this case" }, icon("reset", 14), "Clear all refinements")));
    setChildren(optSlot, h("section.panel.rp-opts", {},
      h("div.combo-head", {}, icon("sliders", 15), h("span.panel-title", { text: "Options" })),
      h("div.mt-stack.rp-opts-body", {}, rows)));
  }

  // ---- assemble ----
  drawCount(); drawOptions(); syncBar(); setHint();
  const mainCol = h("div.rp-main", {}, bar, stage,
    h("div.fv-footer", {}, readout, h("span.fv-hint", { text: "Wheel: zoom · drag: pan or paint · Esc cancels a pen stroke" })));
  const el = h("section.rp-ws", { "aria-label": "Mesh repair workspace" }, h("div.rp-left", {}, leftSlot, optSlot), mainCol);

  const load = () => {
    if (doc.caseMesh) { geom = doc.caseMesh; prepare(); draw(); return; }
    const det = doc.caseDetail || {};
    if ((det.mesh_files || []).includes("mesh/mesh.json")) {
      hint.textContent = "Loading the mesh…";
      bridge.caseJSON(doc.caseName, "mesh/mesh.json").then((m) => { doc.caseMesh = m; geom = m; prepare(); setHint(); drawOptions(); draw(); })
        .catch((e) => { hint.textContent = `Could not load mesh.json: ${e.message}`; });
    } else hint.textContent = "Create the mesh once to repair it (Create mesh).";
  };
  load();
  el._refresh = () => { geom = null; load(); };
  el._draw = draw;
  return el;
}
