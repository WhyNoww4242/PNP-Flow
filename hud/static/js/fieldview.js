// Field viewer: WebGL triangulated field + 2D overlay (axes, boundaries,
// arrows, streamlines), colorbar, pan/zoom, hover readout and point probe.

import { GLRenderer, MeshIndex, fieldStats, uniqueEdges } from "./glfield.js";
import { COLORMAP_NAMES, DIVERGING, colormapCSS } from "./colormaps.js";
import {
  h, $, fmt, niceTicks, logTicks, store, cssVar, placeholder, setChildren,
  prettyName, isSignedFieldName, debounce,
} from "./util.js";

// Boundary colors (Okabe–Ito-ish, readable on dark and light).
const BOUNDARY_PALETTE = ["#ff6b6b", "#4dabf7", "#51cf66", "#fcc419", "#cc5de8", "#ff922b", "#22b8cf", "#f06595"];
const KNOWN_BOUNDARY = { wall: 0, inlet: 1, outlet: 2, side: 3, top: 3, bottom: 4, symmetry: 4, axis: 4 };

function boundaryColor(name, i) {
  const k = name in KNOWN_BOUNDARY ? KNOWN_BOUNDARY[name] : i;
  return BOUNDARY_PALETTE[k % BOUNDARY_PALETTE.length];
}

function hexToRGB(hex) {
  const m = hex.replace("#", "").match(/.{2}/g) || ["00", "00", "00"];
  return m.slice(0, 3).map((x) => parseInt(x, 16) / 255);
}

export class FieldView {
  constructor(root) {
    this.root = root;
    this.mesh = null;
    this.gl = null;
    this.opts = Object.assign({
      field: null, cmap: {}, log: {}, edges: false, bnd: true, axes: true,
      vectors: "none", arrowMode: "log", stream: false,
    }, store.get("fieldview", {}));
    this.range = { mode: "auto", lo: 0, hi: 1 }; // mode: auto | view | manual
    this.view = { cx: 0, cy: 0, ppu: 1 };
    this.size = { w: 1, h: 1, dpr: 1 };
    this.streamCache = null;
    this.hover = null;
    this.probe = null;
    this.buildDOM();
    this.scheduleStream = debounce(() => this.computeStreamlines(), 120);
  }

  // -------------------------------------------------------------------------
  // DOM
  // -------------------------------------------------------------------------
  buildDOM() {
    const sel = (cls, onchange) => h(`select.${cls}`, { onchange });
    this.el = {};
    const e = this.el;
    e.field = sel("fv-field", () => this.selectField(e.field.value));
    e.cmap = sel("fv-cmap", () => this.setColormap(e.cmap.value));
    for (const n of COLORMAP_NAMES) e.cmap.appendChild(h("option", { value: n, text: n + (DIVERGING.has(n) ? " ◐" : "") }));
    e.rangeMode = sel("fv-range-mode", () => this.setRangeMode(e.rangeMode.value));
    for (const [v, t] of [["auto", "auto · all"], ["view", "auto · view"], ["manual", "manual"]]) {
      e.rangeMode.appendChild(h("option", { value: v, text: t }));
    }
    const num = (cls) => h(`input.${cls}.num`, {
      type: "text", spellcheck: "false", inputmode: "decimal",
      onchange: () => this.manualRange(), onkeydown: (ev) => ev.key === "Enter" && this.manualRange(),
    });
    e.lo = num("fv-lo");
    e.hi = num("fv-hi");
    e.log = this.toggle("log", "Logarithmic color scale (positive values only)", () => this.setLog(e.log.checked));
    e.edges = this.toggle("mesh", "Show mesh edges  [E]", () => this.setOpt("edges", e.edges.checked));
    e.bnd = this.toggle("boundaries", "Show boundary edges by name  [B]", () => this.setOpt("bnd", e.bnd.checked));
    e.axes = this.toggle("axes", "Show axis ticks", () => this.setOpt("axes", e.axes.checked));
    e.stream = this.toggle("streamlines", "Streamlines of u  [S]", () => this.setOpt("stream", e.stream.checked));
    e.vectors = sel("fv-vectors", () => this.setOpt("vectors", e.vectors.value));
    e.arrowMode = sel("fv-arrowmode", () => this.setOpt("arrowMode", e.arrowMode.value));
    for (const [v, t] of [["log", "len ∝ log|v|"], ["linear", "len ∝ |v|"], ["uniform", "uniform len"]]) {
      e.arrowMode.appendChild(h("option", { value: v, text: t }));
    }
    const btn = (text, title, onclick) => h("button.btn.btn-sm", { title, onclick, text });
    e.focusBtn = btn("Focus", "Zoom to focus region  [G]", () => this.zoomFocus());
    e.fitBtn = btn("Fit all", "Show the whole domain  [F]", () => this.zoomAll());
    e.pngBtn = btn("PNG", "Save the current view as PNG", () => this.savePNG());

    const group = (label, ...items) => h("div.tb-group", {}, h("span.tb-label", { text: label }), ...items);
    e.toolbar = h("div.toolbar.fv-toolbar", {},
      group("Field", e.field),
      group("Colormap", e.cmap, e.log.wrap),
      group("Range", e.rangeMode, e.lo, h("span.tb-sep", { text: "…" }), e.hi),
      group("Overlay", e.edges.wrap, e.bnd.wrap, e.axes.wrap),
      group("Vectors", e.vectors, e.arrowMode, e.stream.wrap),
      h("div.tb-spacer"),
      h("div.tb-group", {}, e.focusBtn, e.fitBtn, e.pngBtn));

    e.glCanvas = h("canvas.fv-gl");
    e.ovCanvas = h("canvas.fv-overlay");
    e.legend = h("div.fv-legend");
    e.readout = h("div.fv-readout.mono", { text: "hover the field to probe values" });
    e.probe = h("div.fv-probe.hidden");
    e.status = h("div.fv-stage-status");
    e.stage = h("div.fv-stage", { tabindex: "0" }, e.glCanvas, e.ovCanvas, e.legend, e.probe, e.status);
    e.cbarTitle = h("div.cbar-title");
    e.cbarGrad = h("div.cbar-grad");
    e.cbarTicks = h("div.cbar-ticks.mono");
    e.cbarMax = h("div.cbar-end.mono");
    e.cbarMin = h("div.cbar-end.mono");
    e.cbar = h("div.colorbar", {}, e.cbarTitle, e.cbarMax, h("div.cbar-body", {}, e.cbarGrad, e.cbarTicks), e.cbarMin);
    e.main = h("div.fv-main", {}, e.stage, e.cbar);
    e.footer = h("div.fv-footer", {}, e.readout,
      h("div.fv-hint", { text: "wheel: zoom · drag: pan · click: pin probe · dbl-click: focus" }));
    setChildren(this.root, h("div.fieldview", {}, e.toolbar, e.main, e.footer));
    this.attachInteraction();
    new ResizeObserver(() => this.resize()).observe(e.stage);
  }

  /** Small labelled checkbox; returns the input with a .wrap label element. */
  toggle(label, title, onchange) {
    const input = h("input", { type: "checkbox", onchange });
    input.wrap = h("label.chk", { title }, input, h("span", { text: label }));
    return input;
  }

  // -------------------------------------------------------------------------
  // Data
  // -------------------------------------------------------------------------
  showMessage(kind, title, detail) {
    this.el.status.classList.remove("hidden");
    setChildren(this.el.status, placeholder(kind, title, detail));
  }

  hideMessage() {
    this.el.status.classList.add("hidden");
  }

  setMesh(mesh) {
    this.mesh = mesh;
    this.index = null;
    this.stats = {};
    this.probe = null;
    this.el.probe.classList.add("hidden");
    try {
      if (!this.gl) this.gl = new GLRenderer(this.el.glCanvas);
    } catch (err) {
      this.showMessage("error", "Cannot render field", err.message);
      return;
    }
    if (!mesh.nTri) {
      this.showMessage("empty", "Empty mesh", "fields.json contains no triangles.");
      return;
    }
    this.hideMessage();
    this.gl.setMesh(mesh);
    if (this.opts.edges) this.gl.setEdges(uniqueEdges(mesh));
    this.index = new MeshIndex(mesh);
    this.fillFieldSelect();
    this.fillVectorSelect();
    this.buildLegend();
    this.syncControls();
    this.resize(true);
    this.zoomFocus();
    const names = Object.keys(mesh.fields);
    const want = names.includes(this.opts.field) ? this.opts.field : this.defaultField(names);
    if (want) this.selectField(want);
    else this.showMessage("empty", "No nodal fields", "fields.json has a mesh but no fields with one value per node.");
  }

  defaultField(names) {
    for (const n of ["phi", "|u|", "c+", "c-", "rho", "p"]) if (names.includes(n)) return n;
    return names[0] || null;
  }

  fillFieldSelect() {
    const sel = this.el.field;
    sel.textContent = "";
    const vecComps = new Set(Object.values(this.mesh.vectors).flat().filter((c) => /_[yz]$/.test(c) && c.startsWith("N_")));
    for (const name of Object.keys(this.mesh.fields)) {
      const label = this.mesh.labels[name];
      const opt = h("option", { value: name, text: prettyName(name) + (label ? ` — ${label}` : "") });
      if (vecComps.has(name)) opt.dataset.minor = "1";
      sel.appendChild(opt);
    }
  }

  fillVectorSelect() {
    const sel = this.el.vectors;
    sel.textContent = "";
    sel.appendChild(h("option", { value: "none", text: "no arrows" }));
    for (const name of Object.keys(this.mesh.vectors)) {
      sel.appendChild(h("option", { value: name, text: `arrows: ${prettyName(name)}` }));
    }
    if (!(this.opts.vectors in this.mesh.vectors)) this.opts.vectors = "none";
    sel.disabled = Object.keys(this.mesh.vectors).length === 0;
    this.el.stream.disabled = !this.mesh.vectors.u;
    this.el.stream.wrap.classList.toggle("disabled", !this.mesh.vectors.u);
    this.el.stream.wrap.title = this.mesh.vectors.u ? "Streamlines of u  [S]" : "No velocity vector 'u' in fields.json";
  }

  buildLegend() {
    const names = Object.keys(this.mesh.boundaries);
    setChildren(this.el.legend, names.map((n, i) => h("span.legend-item", {},
      h("span.swatch", { style: { background: boundaryColor(n, i) } }), n)));
    this.el.legend.classList.toggle("hidden", !names.length || !this.opts.bnd);
  }

  syncControls() {
    const e = this.el, o = this.opts;
    e.edges.checked = o.edges;
    e.bnd.checked = o.bnd;
    e.axes.checked = o.axes;
    e.stream.checked = o.stream && !!(this.mesh && this.mesh.vectors.u);
    e.vectors.value = o.vectors;
    e.arrowMode.value = o.arrowMode;
    e.arrowMode.disabled = o.vectors === "none";
    e.rangeMode.value = this.range.mode;
    e.lo.disabled = e.hi.disabled = false;
  }

  persist() {
    store.set("fieldview", this.opts);
  }

  // -------------------------------------------------------------------------
  // Field / colormap / range
  // -------------------------------------------------------------------------
  statsOf(name) {
    if (!this.stats[name]) this.stats[name] = fieldStats(this.mesh.fields[name]);
    return this.stats[name];
  }

  selectField(name) {
    if (!this.mesh || !this.mesh.fields[name]) return;
    this.opts.field = name;
    this.el.field.value = name;
    const st = this.statsOf(name);
    const big = Math.max(Math.abs(st.min), Math.abs(st.max)) || 1;
    const signed = st.min < -0.05 * big && st.max > 0.05 * big;   // both signs substantial
    this.signed = signed;
    const cmap = this.opts.cmap[name] || (/^\|(u|E)\|$|vorticity/.test(name) ? "ember" : /^rho/.test(name) ? "bubblegum" : "cosmic");
    this.el.cmap.value = cmap;
    this.gl.setColormap(cmap);
    this.cmapName = cmap;
    const canLog = st.minPos !== null && st.max > 0;
    this.el.log.disabled = !canLog;
    this.el.log.wrap.classList.toggle("disabled", !canLog);
    this.el.log.checked = canLog && !!this.opts.log[name];
    this.gl.setValues(this.mesh.fields[name]);
    if (this.range.mode === "manual") this.range.mode = "auto";
    this.el.rangeMode.value = this.range.mode;
    this.autoRange();
    this.persist();
    this.updateProbe();
  }

  get isLog() {
    return this.el.log.checked && !this.el.log.disabled;
  }

  setColormap(name) {
    this.opts.cmap[this.opts.field] = name;
    this.cmapName = name;
    this.gl.setColormap(name);
    this.persist();
    this.autoRange();
  }

  setLog(on) {
    this.opts.log[this.opts.field] = on;
    if (this.range.mode === "manual") this.range.mode = "auto";
    this.el.rangeMode.value = this.range.mode;
    this.persist();
    this.autoRange();
  }

  setRangeMode(mode) {
    this.range.mode = mode;
    if (mode !== "manual") this.autoRange();
  }

  /** Compute the color range from all nodes or from nodes in view. */
  autoRange() {
    if (!this.mesh || !this.opts.field) return;
    const vals = this.mesh.fields[this.opts.field];
    let lo = Infinity, hi = -Infinity, loPos = Infinity;
    const box = this.range.mode === "view" ? this.visibleBox() : null;
    const xy = this.mesh.xy;
    for (let i = 0; i < vals.length; i++) {
      const v = vals[i];
      if (!(v === v)) continue;
      if (box) {
        const x = xy[2 * i], y = xy[2 * i + 1];
        if (x < box[0] || x > box[1] || y < box[2] || y > box[3]) continue;
      }
      if (v < lo) lo = v;
      if (v > hi) hi = v;
      if (v > 0 && v < loPos) loPos = v;
    }
    if (lo === Infinity) { lo = 0; hi = 1; }
    if (this.isLog) {
      if (loPos === Infinity) loPos = hi > 0 ? hi / 10 : 1;
      lo = Math.log10(loPos);
      hi = Math.log10(Math.max(hi, loPos));
      if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }
    } else if (DIVERGING.has(this.cmapName) && this.signed) {
      const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
      lo = -m; hi = m;
    } else if (hi - lo < 1e-30) {
      const d = Math.abs(hi) * 1e-3 || 1;
      lo -= d; hi += d;
    }
    this.range.lo = lo;
    this.range.hi = hi;
    this.showRangeInputs();
    this.redraw();
  }

  showRangeInputs() {
    const conv = (v) => (this.isLog ? Math.pow(10, v) : v);
    this.el.lo.value = fmt(conv(this.range.lo), 4);
    this.el.hi.value = fmt(conv(this.range.hi), 4);
  }

  manualRange() {
    let lo = parseFloat(this.el.lo.value), hi = parseFloat(this.el.hi.value);
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return this.showRangeInputs();
    if (this.isLog) {
      if (lo <= 0 || hi <= 0) return this.showRangeInputs();
      lo = Math.log10(lo);
      hi = Math.log10(hi);
    }
    if (hi < lo) [lo, hi] = [hi, lo];
    if (hi === lo) hi = lo + Math.abs(lo) * 1e-6 + 1e-30;
    Object.assign(this.range, { mode: "manual", lo, hi });
    this.el.rangeMode.value = "manual";
    this.showRangeInputs();
    this.redraw();
  }

  setOpt(key, value) {
    this.opts[key] = value;
    this.persist();
    if (key === "edges" && value && this.mesh) this.gl.setEdges(uniqueEdges(this.mesh));
    if (key === "bnd") this.el.legend.classList.toggle("hidden", !value || !Object.keys(this.mesh.boundaries).length);
    if (key === "stream") this.streamCache = null;
    this.el.arrowMode.disabled = this.opts.vectors === "none";
    this.redraw();
  }

  /** Toggle a boolean option from a keyboard shortcut. */
  toggleOpt(key) {
    const input = this.el[key];
    if (!input || input.disabled) return;
    input.checked = !input.checked;
    this.setOpt(key, input.checked);
  }

  // -------------------------------------------------------------------------
  // View transform
  // -------------------------------------------------------------------------
  resize(force) {
    const r = this.el.stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), hgt = Math.max(1, Math.round(r.height * dpr));
    if (!force && w === this.size.w && hgt === this.size.h) return;
    const oldW = this.size.w, oldH = this.size.h;
    this.size = { w, h: hgt, dpr };
    if (this.gl) this.gl.resize(w, hgt);
    this.el.ovCanvas.width = w;
    this.el.ovCanvas.height = hgt;
    // keep the same visible extent when the stage resizes
    if (oldW > 1 && !force) this.view.ppu *= Math.min(w / oldW, hgt / oldH);
    this.redraw();
  }

  fitBox(b, margin = 0.06) {
    const { w, h: hh } = this.size;
    const bw = Math.max(b[1] - b[0], 1e-12), bh = Math.max(b[3] - b[2], 1e-12);
    this.view.cx = (b[0] + b[1]) / 2;
    this.view.cy = (b[2] + b[3]) / 2;
    this.view.ppu = Math.min(w / bw, hh / bh) * (1 - 2 * margin);
    this.onViewChange();
  }

  zoomFocus() {
    if (this.mesh) this.fitBox(this.mesh.focus);
  }

  zoomAll() {
    if (this.mesh) this.fitBox(this.mesh.dataBox, 0.04);
  }

  toWorld(px, py) {
    const { w, h: hh } = this.size;
    return [this.view.cx + (px - w / 2) / this.view.ppu, this.view.cy - (py - hh / 2) / this.view.ppu];
  }

  toScreen(x, y) {
    const { w, h: hh } = this.size;
    return [(x - this.view.cx) * this.view.ppu + w / 2, hh / 2 - (y - this.view.cy) * this.view.ppu];
  }

  visibleBox() {
    const [x0, y1] = this.toWorld(0, 0);
    const [x1, y0] = this.toWorld(this.size.w, this.size.h);
    return [x0, x1, y0, y1];
  }

  onViewChange() {
    if (this.range.mode === "view") this.autoRange();
    else this.redraw();
    if (this.opts.stream) this.scheduleStream();
  }

  attachInteraction() {
    const stage = this.el.stage;
    let drag = null;
    const devXY = (ev) => {
      const r = stage.getBoundingClientRect();
      return [(ev.clientX - r.left) * this.size.dpr, (ev.clientY - r.top) * this.size.dpr];
    };
    stage.addEventListener("wheel", (ev) => {
      if (!this.mesh) return;
      ev.preventDefault();
      const [px, py] = devXY(ev);
      const [wx, wy] = this.toWorld(px, py);
      const dy = ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaY;
      const f = Math.exp(-Math.max(-300, Math.min(300, dy)) * 0.0022);
      this.view.ppu = Math.min(Math.max(this.view.ppu * f, 1e-6), 1e9);
      const [nx, ny] = this.toWorld(px, py);
      this.view.cx += wx - nx;
      this.view.cy += wy - ny;
      this.onViewChange();
      this.updateHover(px, py);
    }, { passive: false });
    stage.addEventListener("pointerdown", (ev) => {
      if (ev.button !== 0) return;
      try { stage.setPointerCapture(ev.pointerId); } catch { /* synthetic events */ }
      const [px, py] = devXY(ev);
      drag = { px, py, cx: this.view.cx, cy: this.view.cy, moved: false };
      stage.classList.add("dragging");
    });
    stage.addEventListener("pointermove", (ev) => {
      const [px, py] = devXY(ev);
      if (drag) {
        const dx = px - drag.px, dy = py - drag.py;
        if (Math.abs(dx) + Math.abs(dy) > 3 * this.size.dpr) drag.moved = true;
        this.view.cx = drag.cx - dx / this.view.ppu;
        this.view.cy = drag.cy + dy / this.view.ppu;
        this.onViewChange();
      }
      this.updateHover(px, py);
    });
    const end = (ev) => {
      if (!drag) return;
      stage.classList.remove("dragging");
      if (!drag.moved && ev.type === "pointerup") this.pinProbe(...devXY(ev));
      drag = null;
    };
    stage.addEventListener("pointerup", end);
    stage.addEventListener("pointercancel", end);
    stage.addEventListener("pointerleave", () => {
      this.hover = null;
      this.showHover();
    });
    stage.addEventListener("dblclick", () => this.zoomFocus());
  }

  // -------------------------------------------------------------------------
  // Hover & probe
  // -------------------------------------------------------------------------
  sample(x, y) {
    if (!this.index) return null;
    return this.index.locate(x, y);
  }

  updateHover(px, py) {
    if (!this.mesh) return;
    const [x, y] = this.toWorld(px, py);
    this.hover = { x, y, loc: this.sample(x, y) };
    this.showHover();
  }

  showHover() {
    const r = this.el.readout;
    if (!this.hover) {
      r.textContent = "hover the field to probe values";
      r.classList.add("dim");
      return;
    }
    r.classList.remove("dim");
    const { x, y, loc } = this.hover;
    const parts = [h("span.k", { text: "y " }), h("span.v", { text: fmt(x, 5) }),
      h("span.k", { text: "  z " }), h("span.v", { text: fmt(y, 5) }),
      h("span.k", { text: "  r " }), h("span.v", { text: fmt(Math.hypot(x, y), 5) })];
    if (!loc) parts.push(h("span.k.out", { text: "   outside mesh" }));
    else {
      const name = this.opts.field;
      const v = MeshIndex.interp(this.mesh.fields[name], loc);
      parts.push(h("span.k", { text: `   ${prettyName(name)} = ` }), h("span.v.hl", { text: fmt(v, 5) }));
      const vec = this.opts.vectors !== "none" ? this.opts.vectors : this.opts.stream ? "u" : null;
      if (vec && this.mesh.vectors[vec]) {
        const [a, b] = this.mesh.vectors[vec].map((c) => MeshIndex.interp(this.mesh.fields[c], loc));
        parts.push(h("span.k", { text: `   |${prettyName(vec)}| = ` }), h("span.v", { text: fmt(Math.hypot(a, b), 4) }));
      }
      parts.push(h("span.k", { text: `   △${loc[6]}` }));
    }
    setChildren(r, parts);
  }

  pinProbe(px, py) {
    if (!this.mesh) return;
    const [x, y] = this.toWorld(px, py);
    const loc = this.sample(x, y);
    this.probe = loc ? { x, y, loc } : null;
    this.updateProbe();
    this.redraw();
  }

  updateProbe() {
    const box = this.el.probe;
    if (!this.probe || !this.mesh) {
      box.classList.add("hidden");
      return;
    }
    const { x, y } = this.probe;
    const loc = this.sample(x, y);
    if (!loc) { box.classList.add("hidden"); return; }
    const rows = Object.keys(this.mesh.fields).map((n) => h("tr", { class: n === this.opts.field ? "cur" : "" },
      h("td", { text: prettyName(n) }), h("td.mono.num", { text: fmt(MeshIndex.interp(this.mesh.fields[n], loc), 5) })));
    setChildren(box,
      h("div.probe-head", {},
        h("span", { text: "Probe" }),
        h("span.mono.dim", { text: `y ${fmt(x, 4)}, z ${fmt(y, 4)}` }),
        h("button.icon-btn", { title: "Close probe", text: "×", onclick: (ev) => { ev.stopPropagation(); this.probe = null; this.updateProbe(); this.redraw(); } })),
      h("div.probe-body", {}, h("table.probe-table", {}, h("tbody", {}, rows))));
    box.classList.remove("hidden");
  }

  // -------------------------------------------------------------------------
  // Drawing
  // -------------------------------------------------------------------------
  redraw() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => {
      this.raf = 0;
      this.drawNow();
    });
  }

  drawNow() {
    if (!this.gl || !this.mesh || !this.opts.field) return;
    const bg = hexToRGB(cssVar("--plot-bg") || "#0b0e13");
    const nan = hexToRGB(cssVar("--plot-nan") || "#555555");
    const dark = document.documentElement.dataset.theme !== "light";
    this.gl.draw(this.view, {
      lo: this.range.lo, hi: this.range.hi, log: this.isLog, bg, nan,
      edges: this.opts.edges, edgeColor: dark ? [0, 0, 0, 0.35] : [0, 0, 0, 0.3],
    });
    this.drawOverlay();
    this.drawColorbar();
  }

  drawOverlay() {
    const c = this.el.ovCanvas.getContext("2d");
    const { w, h: hh, dpr } = this.size;
    c.clearRect(0, 0, w, hh);
    if (this.opts.stream && this.mesh.vectors.u) this.drawStreamlines(c);
    if (this.opts.vectors !== "none") this.drawArrows(c);
    if (this.opts.bnd) this.drawBoundaries(c);
    if (this.probe) this.drawProbeMarker(c);
    if (this.opts.axes) this.drawAxes(c);
    c.lineWidth = dpr;
  }

  drawBoundaries(c) {
    const { xy } = this.mesh;
    const names = Object.keys(this.mesh.boundaries);
    c.lineCap = "round";
    names.forEach((name, i) => {
      const e = this.mesh.boundaries[name];
      c.strokeStyle = boundaryColor(name, i);
      c.lineWidth = 2.2 * this.size.dpr;
      c.beginPath();
      for (let k = 0; k < e.length; k += 2) {
        const [ax, ay] = this.toScreen(xy[2 * e[k]], xy[2 * e[k] + 1]);
        const [bx, by] = this.toScreen(xy[2 * e[k + 1]], xy[2 * e[k + 1] + 1]);
        c.moveTo(ax, ay);
        c.lineTo(bx, by);
      }
      c.stroke();
    });
  }

  drawProbeMarker(c) {
    const [sx, sy] = this.toScreen(this.probe.x, this.probe.y);
    const d = this.size.dpr;
    c.strokeStyle = "#fff";
    c.lineWidth = 3 * d;
    c.beginPath();
    c.arc(sx, sy, 5 * d, 0, 2 * Math.PI);
    c.stroke();
    c.strokeStyle = "#000";
    c.lineWidth = 1.2 * d;
    c.stroke();
  }

  /** Arrow glyphs on a screen-space grid. */
  drawArrows(c) {
    const comps = this.mesh.vectors[this.opts.vectors];
    if (!comps) return;
    const [fx, fy] = comps.map((n) => this.mesh.fields[n]);
    const d = this.size.dpr, step = 30 * d;
    const pts = [];
    let vmax = 0;
    for (let py = step / 2; py < this.size.h; py += step) {
      for (let px = step / 2; px < this.size.w; px += step) {
        const [x, y] = this.toWorld(px, py);
        const loc = this.sample(x, y);
        if (!loc) continue;
        const vx = MeshIndex.interp(fx, loc), vy = MeshIndex.interp(fy, loc);
        const m = Math.hypot(vx, vy);
        if (!(m > 0)) continue;
        vmax = Math.max(vmax, m);
        pts.push([px, py, vx / m, vy / m, m]);
      }
    }
    if (!vmax) return;
    const mode = this.opts.arrowMode;
    const decades = 3; // log mode spans 3 decades below the max in view
    c.lineCap = "round";
    for (const [px, py, ux, uy, m] of pts) {
      let s;
      if (mode === "uniform") s = 0.7;
      else if (mode === "linear") s = m / vmax;
      else s = Math.max(0, 1 + Math.log10(m / vmax) / decades);
      if (s < 0.04) continue;
      const L = s * step * 0.9;
      const x0 = px - (ux * L) / 2, y0 = py + (uy * L) / 2;
      const x1 = px + (ux * L) / 2, y1 = py - (uy * L) / 2;
      this.arrow(c, x0, y0, x1, y1, Math.min(6 * d, L * 0.45));
    }
  }

  arrow(c, x0, y0, x1, y1, head) {
    const ang = Math.atan2(y1 - y0, x1 - x0);
    const path = () => {
      c.beginPath();
      c.moveTo(x0, y0);
      c.lineTo(x1, y1);
      c.moveTo(x1 - head * Math.cos(ang - 0.45), y1 - head * Math.sin(ang - 0.45));
      c.lineTo(x1, y1);
      c.lineTo(x1 - head * Math.cos(ang + 0.45), y1 - head * Math.sin(ang + 0.45));
    };
    const d = this.size.dpr;
    path();
    c.strokeStyle = "rgba(0,0,0,0.55)";
    c.lineWidth = 3 * d;
    c.stroke();
    c.strokeStyle = "rgba(255,255,255,0.92)";
    c.lineWidth = 1.3 * d;
    c.stroke();
  }

  /** Evenly-spaced streamlines of u in the visible region (cached in world coords). */
  computeStreamlines() {
    if (!this.mesh || !this.mesh.vectors.u || !this.opts.stream) return;
    const [fx, fy] = this.mesh.vectors.u.map((n) => this.mesh.fields[n]);
    const vb = this.visibleBox(), db = this.mesh.dataBox;
    const box = [Math.max(vb[0], db[0]), Math.min(vb[1], db[1]), Math.max(vb[2], db[2]), Math.min(vb[3], db[3])];
    if (box[1] <= box[0] || box[3] <= box[2]) { this.streamCache = []; this.redraw(); return; }
    const dsep = (26 * this.size.dpr) / this.view.ppu; // world spacing between lines
    const cell = dsep / 2;
    const nx = Math.ceil((box[1] - box[0]) / cell), ny = Math.ceil((box[3] - box[2]) / cell);
    if (nx * ny > 4e6) return;
    const occ = new Int32Array(nx * ny).fill(-1);
    const cellOf = (x, y) => {
      const i = Math.floor((x - box[0]) / cell), j = Math.floor((y - box[2]) / cell);
      return i < 0 || j < 0 || i >= nx || j >= ny ? -1 : j * nx + i;
    };
    const free = (x, y, id) => {
      const i = Math.floor((x - box[0]) / cell), j = Math.floor((y - box[2]) / cell);
      for (let dj = -1; dj <= 1; dj++) for (let di = -1; di <= 1; di++) {
        const ii = i + di, jj = j + dj;
        if (ii < 0 || jj < 0 || ii >= nx || jj >= ny) continue;
        const o = occ[jj * nx + ii];
        if (o >= 0 && o !== id) return false;
      }
      return true;
    };
    const vel = (x, y) => {
      const loc = this.sample(x, y);
      if (!loc) return null;
      const vx = MeshIndex.interp(fx, loc), vy = MeshIndex.interp(fy, loc);
      const m = Math.hypot(vx, vy);
      return m > 1e-12 ? [vx / m, vy / m, m] : null;
    };
    const hstep = dsep / 5;
    const trace = (x, y, dir, id) => {
      const pts = [];
      for (let k = 0; k < 400; k++) {
        const v1 = vel(x, y);
        if (!v1) break;
        const xm = x + dir * v1[0] * hstep * 0.5, ym = y + dir * v1[1] * hstep * 0.5;
        const v2 = vel(xm, ym);
        if (!v2) break;
        x += dir * v2[0] * hstep;
        y += dir * v2[1] * hstep;
        if (cellOf(x, y) < 0 || !free(x, y, id)) break;
        pts.push(x, y);
        budget--;
        if (budget <= 0) break;
      }
      return pts;
    };
    let budget = 80000;
    const lines = [];
    // seeds: jittered grid, visited in a fixed pseudo-random order
    const seeds = [];
    for (let j = 0; j < ny; j += 2) for (let i = 0; i < nx; i += 2) seeds.push([i, j]);
    let rnd = 12345;
    for (let k = seeds.length - 1; k > 0; k--) {
      rnd = (rnd * 1103515245 + 12345) & 0x7fffffff;
      const r = rnd % (k + 1);
      [seeds[k], seeds[r]] = [seeds[r], seeds[k]];
    }
    for (const [i, j] of seeds) {
      if (budget <= 0) break;
      const x = box[0] + (i + 0.5) * cell, y = box[2] + (j + 0.5) * cell;
      if (!free(x, y, -2) || !vel(x, y)) continue;
      const id = lines.length;
      const fwd = trace(x, y, 1, id), bwd = trace(x, y, -1, id);
      const pts = [];
      for (let k = bwd.length - 2; k >= 0; k -= 2) pts.push(bwd[k], bwd[k + 1]);
      pts.push(x, y, ...fwd);
      if (pts.length < 8) continue;
      for (let k = 0; k < pts.length; k += 2) {
        const c = cellOf(pts[k], pts[k + 1]);
        if (c >= 0) occ[c] = id;
      }
      lines.push({ pts: new Float32Array(pts), mid: bwd.length });
    }
    this.streamCache = lines;
    this.redraw();
  }

  drawStreamlines(c) {
    if (!this.streamCache) { this.scheduleStream(); return; }
    const d = this.size.dpr;
    const dark = document.documentElement.dataset.theme !== "light";
    c.lineJoin = "round";
    const pass = (color, width) => {
      c.strokeStyle = color;
      c.lineWidth = width;
      c.beginPath();
      for (const ln of this.streamCache) {
        const p = ln.pts;
        let [sx, sy] = this.toScreen(p[0], p[1]);
        c.moveTo(sx, sy);
        for (let k = 2; k < p.length; k += 2) {
          [sx, sy] = this.toScreen(p[k], p[k + 1]);
          c.lineTo(sx, sy);
        }
      }
      c.stroke();
    };
    pass("rgba(0,0,0,0.35)", 2.6 * d);
    pass(dark ? "rgba(255,255,255,0.85)" : "rgba(255,255,255,0.9)", 1.1 * d);
    // direction chevrons at the seed point of each line
    for (const ln of this.streamCache) {
      const p = ln.pts, k = Math.min(ln.mid, p.length - 4);
      const [ax, ay] = this.toScreen(p[k], p[k + 1]);
      const [bx, by] = this.toScreen(p[k + 2], p[k + 3]);
      if (Math.hypot(bx - ax, by - ay) < 1e-3) continue;
      this.arrow(c, ax, ay, bx, by, 5 * d);
    }
  }

  /** Tick marks + labels along the left (z) and bottom (y) edges of the stage. */
  drawAxes(c) {
    const { w, h: hh, dpr } = this.size;
    const [x0, x1, y0, y1] = this.visibleBox();
    const nT = Math.max(2, Math.round(w / (110 * dpr)));
    const xt = niceTicks(x0, x1, nT), yt = niceTicks(y0, y1, Math.max(2, Math.round(hh / (90 * dpr))));
    const fg = cssVar("--axis-fg") || "#ccd";
    c.font = `${11 * dpr}px ${cssVar("--font-mono") || "monospace"}`;
    c.fillStyle = cssVar("--axis-bg") || "rgba(0,0,0,0.5)";
    const band = 18 * dpr;
    c.fillRect(0, hh - band, w, band);
    c.fillRect(0, 0, 46 * dpr, hh - band);
    c.strokeStyle = fg;
    c.fillStyle = fg;
    c.lineWidth = dpr;
    c.textBaseline = "middle";
    c.textAlign = "center";
    c.beginPath();
    for (const t of xt) {
      const [sx] = this.toScreen(t, 0);
      if (sx < 56 * dpr || sx > w - 48 * dpr) continue;
      c.moveTo(sx, hh - band);
      c.lineTo(sx, hh - band + 4 * dpr);
      c.fillText(fmt(t, 3), sx, hh - band / 2 + dpr);
    }
    c.textAlign = "right";
    for (const t of yt) {
      const [, sy] = this.toScreen(0, t);
      if (sy < 8 * dpr || sy > hh - band - 6 * dpr) continue;
      c.moveTo(46 * dpr, sy);
      c.lineTo(42 * dpr, sy);
      c.fillText(fmt(t, 3), 39 * dpr, sy);
    }
    c.stroke();
    c.textAlign = "left";
    c.fillStyle = cssVar("--accent") || "#7cf";
    c.fillText("y →", w - 32 * dpr, hh - band / 2 + dpr);
    c.save();
    c.translate(10 * dpr, 10 * dpr);
    c.fillText("z ↑", 0, 4 * dpr);
    c.restore();
  }

  drawColorbar() {
    const e = this.el;
    const name = this.opts.field;
    e.cbarGrad.style.background = colormapCSS(this.cmapName, "to top");
    const label = this.mesh.labels[name];
    setChildren(e.cbarTitle, h("span.cbar-name", { text: prettyName(name) }),
      this.isLog ? h("span.cbar-tag", { text: "log" }) : null,
      label ? h("span.cbar-label", { text: label }) : null);
    const { lo, hi } = this.range;
    const ticks = this.isLog ? logTicks(lo, hi, 9) : niceTicks(lo, hi, 8);
    const items = [];
    for (const t of ticks) {
      const f = (t - lo) / (hi - lo);
      if (f < -1e-6 || f > 1 + 1e-6) continue;
      const text = this.isLog ? (Number.isInteger(t) ? `1e${t}` : fmt(Math.pow(10, t), 3)) : fmt(t, 4);
      items.push(h("div.cbar-tick", { style: { bottom: `${f * 100}%` } }, h("i"), h("span", { text })));
    }
    setChildren(e.cbarTicks, items);
    const end = (v) => (this.isLog ? fmt(Math.pow(10, v), 3) : fmt(v, 4));
    e.cbarMax.textContent = `max ${end(hi)}`;
    e.cbarMin.textContent = `min ${end(lo)}`;
  }

  savePNG() {
    if (!this.mesh) return;
    this.drawNow(); // ensure the GL buffer is current (preserveDrawingBuffer is off)
    const out = document.createElement("canvas");
    out.width = this.size.w;
    out.height = this.size.h;
    const c = out.getContext("2d");
    c.fillStyle = cssVar("--plot-bg") || "#0a0d12";
    c.fillRect(0, 0, out.width, out.height);
    c.drawImage(this.el.glCanvas, 0, 0);
    c.drawImage(this.el.ovCanvas, 0, 0);
    const a = h("a", { href: out.toDataURL("image/png"), download: `${this.runName || "field"}_${this.opts.field}.png` });
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  /** Re-read theme colors (called after theme switch). */
  themeChanged() {
    this.redraw();
  }
}
