// Field viewer: WebGL triangulated field + 2D overlay (axes, boundaries, contours,
// arrows, streamlines), colour bar, pan/zoom, hover readout and point probe.
// Ported from hud/static/js/fieldview.js. Its controls now live outside the canvas
// (workbench toolbar + View tab of the property editor); this class exposes setters
// and calls `onChange()` whenever its display state changes.

import { GLRenderer, MeshIndex, fieldStats, uniqueEdges, contourSegments } from "./glfield.js";
import { colormapCSS, colormapType, defaultColormap, colormapNames } from "./colormaps.js";
import { h, fmt, niceTicks, logTicks, store, cssVar, placeholder, setChildren, prettyName, isSignedFieldName, debounce } from "./util.js";
import { icon } from "./icons.js";

// Fixed categorical boundary colours, so a name keeps its colour in every view.
const BOUNDARY_PALETTE = ["#ff6b6b", "#4dabf7", "#51cf66", "#fcc419", "#cc5de8", "#ff922b", "#22b8cf", "#f06595"];
const KNOWN_BOUNDARY = { wall: 0, electrode: 0, sphere: 0, surface: 0, inlet: 1, outlet: 2, side: 3, top: 3, outer: 3, far: 3, bottom: 4, symmetry: 5, axis: 5 };

export function boundaryColor(name, i) {
  const k = name in KNOWN_BOUNDARY ? KNOWN_BOUNDARY[name] : i;
  return BOUNDARY_PALETTE[k % BOUNDARY_PALETTE.length];
}

function hexToRGB(hex) {
  const m = hex.replace("#", "").match(/.{2}/g) || ["00", "00", "00"];
  return m.slice(0, 3).map((x) => parseInt(x, 16) / 255);
}

export class FieldView {
  constructor(root, { onChange = () => {} } = {}) {
    this.root = root;
    this.onChange = onChange;
    this.mesh = null;
    this.gl = null;
    this.opts = Object.assign({
      field: null, cmap: {}, rev: {}, log: {}, edges: false, bnd: true, axes: true,
      contours: 0, vectors: "none", arrowMode: "log", stream: false, mirror: true,
    }, store.get("fieldview", {}));
    this.range = { mode: "auto", lo: 0, hi: 1 }; // mode: auto | view | manual
    this.view = { cx: 0, cy: 0, ppu: 1, ky: 1 }; // ppu: device px per unit along x; ky: y scale relative to x (1 = true aspect)
    this.size = { w: 1, h: 1, dpr: 1 };
    this.streamCache = null;
    this.contourCache = null;
    this.hover = null;
    this.probe = null;
    this.buildDOM();
    this.scheduleStream = debounce(() => this.computeStreamlines(), 120);
  }

  // -------------------------------------------------------------------------
  // DOM
  // -------------------------------------------------------------------------
  buildDOM() {
    const e = (this.el = {});
    e.glCanvas = h("canvas.fv-gl");
    e.ovCanvas = h("canvas.fv-overlay");
    e.legend = h("div.fv-legend", { "aria-label": "Boundary legend" });
    e.note = h("div.fv-note.hidden");
    e.readout = h("div.fv-readout", { text: "Point at the field to read values" });
    e.probe = h("div.fv-probe.hidden");
    e.status = h("div.fv-stage-status.hidden");
    e.stage = h("div.fv-stage", { tabindex: "0", "aria-label": "Field map. Wheel to zoom, drag to pan, click to probe." },
      e.glCanvas, e.ovCanvas, e.legend, e.note, e.probe, e.status);
    e.cbarTitle = h("div.cbar-title");
    e.cbarGrad = h("div.cbar-grad");
    e.cbarTicks = h("div.cbar-ticks.mono");
    e.cbarMax = h("div.cbar-end.mono");
    e.cbarMin = h("div.cbar-end.mono");
    e.cbar = h("div.colorbar", {}, e.cbarTitle, e.cbarMax, h("div.cbar-body", {}, e.cbarGrad, e.cbarTicks), e.cbarMin);
    e.main = h("div.fv-main", {}, e.stage, e.cbar);
    e.field = h("select.sel.fv-field", { "aria-label": "Field shown", "data-tip": "Field shown on the map", onchange: (ev) => this.selectField(ev.target.value) });
    const btn = (ic, label, run, extra = {}) => h("button.tool", Object.assign({ type: "button", "aria-label": label, "data-tip-title": label, onclick: run }, extra), icon(ic, 18));
    e.bar = h("div.fv-bar", {}, e.field, h("span.tool-sep", { "aria-hidden": "true" }),
      btn("home", "Home: zoom to the focus region", () => this.home(), { "data-tip-key": "G" }),
      btn("zoom-in", "Zoom in", () => this.zoomBy(1.4)),
      btn("zoom-out", "Zoom out", () => this.zoomBy(1 / 1.4)),
      h("span.fv-axgrp", { role: "group", "aria-label": "Scale the x axis" }, h("span.fv-axl", { text: "x" }),
        btn("zoom-out", "Shrink the x axis (Shift + wheel)", () => this.zoomAxis("x", 1 / 1.4)),
        btn("zoom-in", "Stretch the x axis (Shift + wheel)", () => this.zoomAxis("x", 1.4))),
      h("span.fv-axgrp", { role: "group", "aria-label": "Scale the y axis" }, h("span.fv-axl", { text: "y" }),
        btn("zoom-out", "Shrink the y axis (Alt + wheel)", () => this.zoomAxis("y", 1 / 1.4)),
        btn("zoom-in", "Stretch the y axis (Alt + wheel)", () => this.zoomAxis("y", 1.4))),
      btn("fit", "Fit the whole mesh to the window (true aspect ratio)", () => this.fitTrue(), { "data-tip-key": "F" }),
      btn("rescale", "Rescale: fill the window, each axis with its own scale", () => this.rescale()));
    e.tPlay = h("button.tool", { type: "button", "aria-label": "Play the time steps", "data-tip-title": "Play / pause", onclick: () => this.togglePlay() }, icon("play", 16));
    e.tRange = h("input.fv-trange", { type: "range", min: "0", max: "0", step: "1", value: "0", "aria-label": "Time step",
      oninput: (ev) => { this.stopPlay(); this.setTimeIndex(+ev.target.value); } });
    e.tRead = h("div.fv-tread.mono");
    e.time = h("div.fv-time.hidden", {}, e.tPlay, e.tRange, e.tRead);
    e.scaleNote = h("span.fv-scale-note.hidden");
    e.bar.appendChild(h("span.spacer"));
    e.bar.appendChild(e.scaleNote);
    e.footer = h("div.fv-footer", {}, e.readout,
      h("div.fv-hint", { text: "Wheel: zoom · drag: pan · click: probe · double-click: focus" }));
    setChildren(this.root, h("div.fieldview", {}, e.bar, e.main, e.time, e.footer));
    this.attachInteraction();
    new ResizeObserver(() => this.resize()).observe(e.stage);
  }

  showMessage(kind, title, detail) {
    this.el.status.classList.remove("hidden");
    setChildren(this.el.status, placeholder(kind, title, detail));
  }

  hideMessage() {
    this.el.status.classList.add("hidden");
  }

  // -------------------------------------------------------------------------
  // Data
  // -------------------------------------------------------------------------
  setMesh(mesh) {
    this.setTimeSeries(null);      // restores the previous mesh's final fields before it is replaced
    this.mesh = mesh;
    this.index = null;
    this.stats = {};
    this.probe = null;
    this.contourCache = null;
    this.streamCache = null;
    this.el.probe.classList.add("hidden");
    try {
      if (!this.gl) this.gl = new GLRenderer(this.el.glCanvas);
    } catch (err) {
      this.showMessage("error", "This browser cannot draw the field map", `WebGL is unavailable (${err.message}). The profiles, checks and figures still work.`);
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
    if (!(this.opts.vectors in mesh.vectors)) this.opts.vectors = "none";
    this.buildLegend();
    this.el.note.classList.toggle("hidden", !mesh.mirrored);
    setChildren(this.el.note, icon("mirror", 14), h("span", { text: `Mirrored about ${mesh.coordNames[0]} = 0 (axisymmetric)` }));
    this.resize(true);
    this.zoomFocus();
    const names = Object.keys(mesh.fields);
    setChildren(this.el.field, names.map((n) => h("option", { value: n, text: `${prettyName(n)}${mesh.labels && mesh.labels[n] ? ` · ${mesh.labels[n]}` : ""}` })));
    const want = names.includes(this.opts.field) ? this.opts.field : this.defaultField(names);
    if (want) this.selectField(want);
    else this.showMessage("empty", "No nodal fields", "fields.json has a mesh but no field with one value per node.");
  }

  defaultField(names) {
    for (const n of ["phi", "|u|", "c+", "c-", "rho", "p"]) if (names.includes(n)) return n;
    return names[0] || null;
  }

  get fieldNames() {
    return this.mesh ? Object.keys(this.mesh.fields) : [];
  }

  buildLegend() {
    const names = Object.keys(this.mesh.boundaries);
    setChildren(this.el.legend, names.map((n, i) => h("span.legend-item", {},
      h("span.swatch", { class: this.mesh.axisNames.has(n) ? "dashed" : "", style: { background: this.mesh.axisNames.has(n) ? "transparent" : boundaryColor(n, i), borderColor: boundaryColor(n, i) } }), n)));
    this.el.legend.classList.toggle("hidden", !names.length || !this.opts.bnd);
  }

  persist() {
    store.set("fieldview", this.opts);
  }

  statsOf(name) {
    if (!this.stats[name]) {
      const frames = this.ts && this.ts.fields[name];   // time-resolved: statistics over all times
      if (frames) {
        const all = frames.map(fieldStats);
        const pos = all.map((a) => a.minPos).filter((v) => v !== null);
        this.stats[name] = { min: Math.min(...all.map((a) => a.min)), max: Math.max(...all.map((a) => a.max)),
          minPos: pos.length ? Math.min(...pos) : null, nan: all[0].nan };
      } else this.stats[name] = fieldStats(this.mesh.fields[name]);
    }
    return this.stats[name];
  }

  /** A field is shown as signed when its data actually cross zero. */
  isSigned(name) {
    const st = this.statsOf(name);
    const m = Math.max(Math.abs(st.min), Math.abs(st.max)) || 1;
    // diverging only when the data really take both signs (a numerical undershoot
    // of a positive field does not count); signed-by-name fields need 1 %, others 2 %
    const frac = isSignedFieldName(name) ? 0.01 : 0.02;
    return st.min < -frac * m && st.max > frac * m;
  }

  // -------------------------------------------------------------------------
  // Time slider (fields_time.json)
  // -------------------------------------------------------------------------
  /** ts = {t, tau, fields:{name: Float32Array[]}} on the prepared mesh, or null. */
  setTimeSeries(ts) {
    this.stopPlay();
    if (this.finalFields && this.mesh) for (const [k, v] of Object.entries(this.finalFields)) this.mesh.fields[k] = v;
    this.finalFields = null;
    this.ts = null;
    this.stats = {};
    this.el.time.classList.add("hidden");
    if (!ts || !this.mesh) return;
    const names = Object.keys(ts.fields).filter((k) => this.mesh.fields[k]);
    if (!names.length || ts.t.length < 2) return;
    this.finalFields = Object.fromEntries(names.map((k) => [k, this.mesh.fields[k]]));
    this.ts = { t: ts.t, tau: ts.tau, fields: Object.fromEntries(names.map((k) => [k, ts.fields[k]])) };
    this.tIdx = ts.t.length - 1;
    this.el.tRange.max = String(this.tIdx);
    this.el.tRange.value = String(this.tIdx);
    this.el.time.classList.remove("hidden");
    this.setTimeIndex(this.tIdx);
    if (this.opts.field) this.selectField(this.opts.field);
  }

  setTimeIndex(i) {
    if (!this.ts) return;
    i = Math.max(0, Math.min(this.ts.t.length - 1, Math.round(i)));
    this.tIdx = i;
    for (const [k, frames] of Object.entries(this.ts.fields)) this.mesh.fields[k] = frames[i];
    this.el.tRange.value = String(i);
    const f = this.opts.field;
    if (f && this.ts.fields[f] && this.gl) {
      this.gl.setValues(this.mesh.fields[f]);
      this.contourCache = null;
      this.streamCache = null;
      if (this.range.mode === "view") this.autoRange(); else this.redraw();
      this.updateProbe();
    }
    this.updateTimeReadout();
  }

  updateTimeReadout() {
    if (!this.ts) return;
    const i = this.tIdx, n = this.ts.t.length;
    const tau = this.ts.tau ? `\u03c4 = ${fmt(this.ts.tau[i], 3)}   ` : "";
    const f = this.opts.field;
    const stale = f && !this.ts.fields[f] && i !== n - 1;
    this.el.tRead.textContent = `${tau}t = ${fmt(this.ts.t[i], 4)}   step ${i + 1} of ${n}${stale ? `   (${prettyName(f)} is only available at the final time)` : ""}`;
    this.el.tRead.classList.toggle("warn", !!stale);
  }

  togglePlay() {
    if (this.playTimer) return this.stopPlay();
    if (!this.ts) return;
    if (this.tIdx >= this.ts.t.length - 1) this.setTimeIndex(0);
    setChildren(this.el.tPlay, icon("stop", 16));
    this.el.tPlay.setAttribute("aria-label", "Pause");
    this.playTimer = setInterval(() => {
      if (this.tIdx >= this.ts.t.length - 1) return this.stopPlay();
      this.setTimeIndex(this.tIdx + 1);
    }, 500);
  }

  stopPlay() {
    if (this.playTimer) { clearInterval(this.playTimer); this.playTimer = null; }
    setChildren(this.el.tPlay, icon("play", 16));
    this.el.tPlay.setAttribute("aria-label", "Play the time steps");
  }

  selectField(name) {
    if (!this.mesh || !this.mesh.fields[name] || !this.gl) return;
    this.opts.field = name;
    this.el.field.value = name;
    this.signed = this.isSigned(name);
    let cmap = this.opts.cmap[name];
    if (!cmap || !colormapNames().includes(cmap)) cmap = defaultColormap(this.signed, name);
    this.cmapName = cmap;
    this.gl.setColormap(cmap, !!this.opts.rev[name]);
    this.gl.setValues(this.mesh.fields[name]);
    if (this.range.mode === "manual") this.range.mode = "auto";
    this.contourCache = null;
    this.autoRange();
    this.persist();
    this.updateProbe();
    this.updateTimeReadout();
    this.onChange();
  }

  get canLog() {
    const st = this.opts.field ? this.statsOf(this.opts.field) : null;
    return !!st && st.minPos !== null && st.max > 0;
  }

  get isLog() {
    return this.canLog && !!this.opts.log[this.opts.field];
  }

  get reversed() {
    return !!this.opts.rev[this.opts.field];
  }

  setColormap(name) {
    this.opts.cmap[this.opts.field] = name;
    this.cmapName = name;
    this.gl.setColormap(name, this.reversed);
    this.persist();
    this.autoRange();
    this.onChange();
  }

  setReversed(on) {
    this.opts.rev[this.opts.field] = !!on;
    this.gl.setColormap(this.cmapName, !!on);
    this.persist();
    this.redraw();
    this.onChange();
  }

  setLog(on) {
    this.opts.log[this.opts.field] = !!on;
    if (this.range.mode === "manual") this.range.mode = "auto";
    this.contourCache = null;
    this.persist();
    this.autoRange();
    this.onChange();
  }

  setRangeMode(mode) {
    this.range.mode = mode;
    if (mode !== "manual") this.autoRange();
    this.onChange();
  }

  /** Colour range from all nodes or from the nodes in view. */
  autoRange() {
    if (!this.mesh || !this.opts.field) return;
    // a time-resolved field keeps one colour range over all times, so frames are comparable
    const frames = this.ts && this.ts.fields[this.opts.field];
    const arrs = frames && this.range.mode !== "view" ? frames : [this.mesh.fields[this.opts.field]];
    let lo = Infinity, hi = -Infinity, loPos = Infinity;
    const box = this.range.mode === "view" ? this.visibleBox() : null;
    const xy = this.mesh.xy;
    for (const vals of arrs) for (let i = 0; i < vals.length; i++) {
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
    } else if (colormapType(this.cmapName) === "diverging" && this.signed) {
      const m = Math.max(Math.abs(lo), Math.abs(hi)) || 1;
      lo = -m; hi = m;
    } else if (hi - lo < 1e-30) {
      const d = Math.abs(hi) * 1e-3 || 1;
      lo -= d; hi += d;
    }
    this.range.lo = lo;
    this.range.hi = hi;
    this.contourCache = null;
    this.redraw();
  }

  /** Range shown to the user (in data units, also for log scale). */
  get displayRange() {
    const conv = (v) => (this.isLog ? Math.pow(10, v) : v);
    return { mode: this.range.mode, lo: conv(this.range.lo), hi: conv(this.range.hi) };
  }

  setManualRange(lo, hi) {
    if (!Number.isFinite(lo) || !Number.isFinite(hi)) return false;
    if (this.isLog) {
      if (lo <= 0 || hi <= 0) return false;
      lo = Math.log10(lo);
      hi = Math.log10(hi);
    }
    if (hi < lo) [lo, hi] = [hi, lo];
    if (hi === lo) hi = lo + Math.abs(lo) * 1e-6 + 1e-30;
    Object.assign(this.range, { mode: "manual", lo, hi });
    this.contourCache = null;
    this.redraw();
    this.onChange();
    return true;
  }

  setOpt(key, value) {
    this.opts[key] = value;
    this.persist();
    if (key === "edges" && value && this.mesh && this.gl) this.gl.setEdges(uniqueEdges(this.mesh));
    if (key === "bnd" && this.mesh) this.el.legend.classList.toggle("hidden", !value || !Object.keys(this.mesh.boundaries).length);
    if (key === "stream") this.streamCache = null;
    if (key === "contours") this.contourCache = null;
    this.redraw();
    this.onChange();
  }

  /** Toggle a boolean option (keyboard shortcut or toolbar). */
  toggleOpt(key) {
    if (key === "stream" && !(this.mesh && this.mesh.vectors.u)) return false;
    if (key === "contours") this.setOpt("contours", this.opts.contours ? 0 : 10);
    else this.setOpt(key, !this.opts[key]);
    return true;
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
    if (oldW > 1 && !force) this.view.ppu *= Math.min(w / oldW, hgt / oldH);
    this.redraw();
  }

  /**
   * Fit a box into the area not covered by the axis bands. stretch: "auto" keeps the
   * true aspect unless the box is so slender that it would be unreadable (then each
   * axis gets its own scale), true always uses independent scales, false never does.
   */
  fitBox(b, margin = 0.06, stretch = "auto") {
    const { w, h: hh, dpr } = this.size;
    const left = this.opts.axes ? 48 * dpr : 0, band = this.opts.axes ? 20 * dpr : 0;
    const bw = Math.max(b[1] - b[0], 1e-12), bh = Math.max(b[3] - b[2], 1e-12);
    const k = 1 - 2 * margin;
    const px = ((w - left) / bw) * k, py = ((hh - band) / bh) * k;
    const iso = Math.min(px, py);
    // how much of the stage the true-aspect box would leave empty
    const fill = iso / Math.max(px, py);
    const st = stretch === "auto" ? fill < 0.34 : !!stretch;
    const ppx = Math.max(1e-9, st ? px : iso), ppy = Math.max(1e-9, st ? py : iso);
    this.view.ppu = ppx;
    this.view.ky = ppy / ppx;
    this.view.cx = (b[0] + b[1]) / 2 - left / (2 * ppx);
    this.view.cy = (b[2] + b[3]) / 2 - band / (2 * ppy);
    this.onViewChange();
  }

  get stretched() {
    return Math.abs(this.view.ky - 1) > 0.02;
  }

  /** Home: the focus region, readable even for slender domains. */
  home() { if (this.mesh) this.fitBox(this.mesh.focus); }
  /** Fit the whole mesh at its true aspect ratio. */
  fitTrue() { if (this.mesh) this.fitBox(this.mesh.dataBox, 0.04, false); }
  /** Rescale: fit the whole mesh with independent axis scales (tick labels show the real values). */
  rescale() { if (this.mesh) this.fitBox(this.mesh.dataBox, 0.04, true); }

  /** Scale x by fx and y by fy about a screen point (device px); default the stage centre. */
  scaleAt(fx, fy, px = this.size.w / 2, py = this.size.h / 2) {
    if (!this.mesh) return;
    const [wx, wy] = this.toWorld(px, py);
    const ppu = Math.min(Math.max(this.view.ppu * fx, 1e-6), 1e9);
    this.view.ky = Math.min(Math.max(this.view.ky * fy / fx, 1e-4), 1e4);
    this.view.ppu = ppu;
    const [nx, ny] = this.toWorld(px, py);
    this.view.cx += wx - nx;
    this.view.cy += wy - ny;
    this.onViewChange();
  }

  zoomBy(f) { this.scaleAt(f, f); }
  /** Scale one axis only ("x" or "y"). */
  zoomAxis(axis, f) { if (axis === "x") this.scaleAt(f, 1); else this.scaleAt(1, f); }

  zoomFocus() {
    this.home();
  }

  zoomAll() {
    if (this.mesh) this.fitBox(this.mesh.dataBox, 0.04);
  }

  toWorld(px, py) {
    const { w, h: hh } = this.size;
    return [this.view.cx + (px - w / 2) / this.view.ppu, this.view.cy - (py - hh / 2) / (this.view.ppu * this.view.ky)];
  }

  toScreen(x, y) {
    const { w, h: hh } = this.size;
    return [(x - this.view.cx) * this.view.ppu + w / 2, hh / 2 - (y - this.view.cy) * this.view.ppu * this.view.ky];
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
      const dy = ev.deltaMode === 1 ? ev.deltaY * 16 : ev.deltaY;
      const f = Math.exp(-Math.max(-300, Math.min(300, dy)) * 0.0022);
      // shift: scale x only; alt: scale y only
      this.scaleAt(ev.altKey ? 1 : f, ev.shiftKey ? 1 : f, px, py);
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
        this.view.cy = drag.cy + dy / (this.view.ppu * this.view.ky);
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
    stage.addEventListener("pointerleave", () => { this.hover = null; this.showHover(); });
    stage.addEventListener("dblclick", () => this.zoomFocus());
    // keyboard panning / zoom when the canvas has focus
    stage.addEventListener("keydown", (ev) => {
      if (!this.mesh) return;
      const step = 60 * this.size.dpr / this.view.ppu;
      const k = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, step], ArrowDown: [0, -step] }[ev.key];
      if (k) { this.view.cx += k[0]; this.view.cy += k[1]; this.onViewChange(); ev.preventDefault(); }
      else if (ev.key === "+" || ev.key === "=") { this.view.ppu *= 1.25; this.onViewChange(); ev.preventDefault(); }
      else if (ev.key === "-") { this.view.ppu /= 1.25; this.onViewChange(); ev.preventDefault(); }
    });
  }

  // -------------------------------------------------------------------------
  // Hover & probe
  // -------------------------------------------------------------------------
  sample(x, y) {
    return this.index ? this.index.locate(x, y) : null;
  }

  /** Display coordinate (|x| for the mirrored half of axisymmetric data). */
  coordX(x) {
    return this.mesh && this.mesh.mirrored ? Math.abs(x) : x;
  }

  updateHover(px, py) {
    if (!this.mesh) return;
    const [x, y] = this.toWorld(px, py);
    this.hover = { x, y, loc: this.sample(x, y) };
    this.showHover();
  }

  showHover() {
    const r = this.el.readout;
    if (!this.hover || !this.mesh) {
      r.textContent = "Point at the field to read values";
      r.classList.add("dim");
      return;
    }
    r.classList.remove("dim");
    const { x, y, loc } = this.hover;
    const [cx, cy] = this.mesh.coordNames;
    const parts = [h("span.k", { text: `${cx} ` }), h("span.v", { text: fmt(this.coordX(x), 5) }),
      h("span.k", { text: `   ${cy} ` }), h("span.v", { text: fmt(y, 5) })];
    if (!loc) parts.push(h("span.k.out", { text: "   outside the mesh" }));
    else {
      const name = this.opts.field;
      const v = MeshIndex.interp(this.mesh.fields[name], loc);
      parts.push(h("span.k", { text: `   ${prettyName(name)} = ` }), h("span.v.hl", { text: fmt(v, 5) }));
      const vec = this.opts.vectors !== "none" ? this.opts.vectors : this.opts.stream ? "u" : null;
      if (vec && this.mesh.vectors[vec]) {
        const [a, b] = this.mesh.vectors[vec].map((c) => MeshIndex.interp(this.mesh.fields[c], loc));
        parts.push(h("span.k", { text: `   |${prettyName(vec)}| = ` }), h("span.v", { text: fmt(Math.hypot(a, b), 4) }));
      }
    }
    setChildren(r, parts);
  }

  pinProbe(px, py) {
    if (!this.mesh) return;
    const [x, y] = this.toWorld(px, py);
    const loc = this.sample(x, y);
    this.probe = loc ? { x, y } : null;
    this.updateProbe();
    this.redraw();
  }

  updateProbe() {
    const box = this.el.probe;
    if (!this.probe || !this.mesh) { box.classList.add("hidden"); return; }
    const { x, y } = this.probe;
    const loc = this.sample(x, y);
    if (!loc) { box.classList.add("hidden"); return; }
    const [cx, cy] = this.mesh.coordNames;
    const rows = Object.keys(this.mesh.fields).map((n) => h("tr", { class: n === this.opts.field ? "cur" : "" },
      h("td", { text: prettyName(n) }), h("td.mono.num", { text: fmt(MeshIndex.interp(this.mesh.fields[n], loc), 5) })));
    const close = h("button.icon-btn", { title: "Close probe", "aria-label": "Close probe",
      onclick: (ev) => { ev.stopPropagation(); this.probe = null; this.updateProbe(); this.redraw(); } }, icon("x", 14));
    setChildren(box,
      h("div.probe-head", {}, h("span", { text: "Probe" }),
        h("span.mono.dim", { text: `${cx} ${fmt(this.coordX(x), 4)}, ${cy} ${fmt(y, 4)}` }), close),
      h("div.probe-body", {}, h("table.probe-table", {}, h("tbody", {}, rows))));
    box.classList.remove("hidden");
  }

  // -------------------------------------------------------------------------
  // Drawing
  // -------------------------------------------------------------------------
  redraw() {
    if (this.raf) return;
    this.raf = requestAnimationFrame(() => { this.raf = 0; this.drawNow(); });
  }

  drawNow() {
    if (!this.gl || !this.mesh || !this.opts.field) return;
    const nan = hexToRGB(cssVar("--plot-nan") || "#555555");
    this.gl.draw(this.view, {
      lo: this.range.lo, hi: this.range.hi, log: this.isLog, nan,
      edges: this.opts.edges, edgeColor: [0, 0, 0, 0.32],
    });
    this.drawOverlay();
    this.drawColorbar();
    this.updateScaleNote();
  }

  updateScaleNote() {
    const n = this.el.scaleNote, on = this.stretched && !!this.mesh;
    n.classList.toggle("hidden", !on);
    if (on) {
      const [a, b] = this.mesh.coordNames;
      const k = 1 / this.view.ky;
      n.textContent = `Axes not to scale: ${b} is ${k >= 1 ? `compressed ${k.toFixed(k < 10 ? 1 : 0)}x` : `stretched ${(1 / k).toFixed(1 / k < 10 ? 1 : 0)}x`} relative to ${a}`;
    }
  }

  drawOverlay() {
    const c = this.el.ovCanvas.getContext("2d");
    const { w, h: hh } = this.size;
    c.clearRect(0, 0, w, hh);
    if (this.opts.contours) this.drawContours(c);
    if (this.opts.stream && this.mesh.vectors.u) this.drawStreamlines(c);
    if (this.opts.vectors !== "none") this.drawArrows(c);
    if (this.opts.bnd) this.drawBoundaries(c);
    if (this.probe) this.drawProbeMarker(c);
    if (this.opts.axes) this.drawAxes(c);
  }

  contourLevels() {
    const n = this.opts.contours;
    const { lo, hi } = this.range;
    let lv = this.isLog ? logTicks(lo, hi, n) : niceTicks(lo, hi, n);
    if (this.isLog && lv.length < 4) lv = niceTicks(lo, hi, n);
    return lv.filter((t) => t > lo && t < hi);
  }

  drawContours(c) {
    const key = `${this.opts.field}|${this.range.lo}|${this.range.hi}|${this.isLog}|${this.opts.contours}`;
    if (!this.contourCache || this.contourCache.key !== key) {
      const tf = this.isLog ? (v) => (v > 0 ? Math.log10(v) : NaN) : (v) => v;
      this.contourCache = { key, segs: contourSegments(this.mesh, this.mesh.fields[this.opts.field], this.contourLevels(), tf) };
    }
    const s = this.contourCache.segs;
    const d = this.size.dpr;
    const path = () => {
      c.beginPath();
      for (let k = 0; k < s.length; k += 4) {
        const [ax, ay] = this.toScreen(s[k], s[k + 1]);
        const [bx, by] = this.toScreen(s[k + 2], s[k + 3]);
        c.moveTo(ax, ay);
        c.lineTo(bx, by);
      }
    };
    c.lineCap = "round";
    path();
    c.strokeStyle = "rgba(0,0,0,0.45)";
    c.lineWidth = 2.2 * d;
    c.stroke();
    c.strokeStyle = "rgba(255,255,255,0.8)";
    c.lineWidth = 1 * d;
    c.stroke();
  }

  drawBoundaries(c) {
    const { xy } = this.mesh;
    const names = Object.keys(this.mesh.boundaries);
    const d = this.size.dpr;
    c.lineCap = "round";
    names.forEach((name, i) => {
      const e = this.mesh.boundaries[name];
      const axis = this.mesh.axisNames.has(name);
      c.strokeStyle = boundaryColor(name, i);
      c.lineWidth = (axis ? 1.4 : 2.2) * d;
      c.setLineDash(axis ? [6 * d, 5 * d] : []);
      c.beginPath();
      for (let k = 0; k < e.length; k += 2) {
        const [ax, ay] = this.toScreen(xy[2 * e[k]], xy[2 * e[k] + 1]);
        const [bx, by] = this.toScreen(xy[2 * e[k + 1]], xy[2 * e[k + 1] + 1]);
        c.moveTo(ax, ay);
        c.lineTo(bx, by);
      }
      c.stroke();
    });
    c.setLineDash([]);
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
    c.lineCap = "round";
    for (const [px, py, ux, uy, m] of pts) {
      const s = mode === "uniform" ? 0.7 : mode === "linear" ? m / vmax : Math.max(0, 1 + Math.log10(m / vmax) / 3);
      if (s < 0.04) continue;
      const L = s * step * 0.9;
      this.arrow(c, px - (ux * L) / 2, py + (uy * L) / 2, px + (ux * L) / 2, py - (uy * L) / 2, Math.min(6 * d, L * 0.45));
    }
  }

  arrow(c, x0, y0, x1, y1, head) {
    const ang = Math.atan2(y1 - y0, x1 - x0);
    c.beginPath();
    c.moveTo(x0, y0);
    c.lineTo(x1, y1);
    c.moveTo(x1 - head * Math.cos(ang - 0.45), y1 - head * Math.sin(ang - 0.45));
    c.lineTo(x1, y1);
    c.lineTo(x1 - head * Math.cos(ang + 0.45), y1 - head * Math.sin(ang + 0.45));
    const d = this.size.dpr;
    c.strokeStyle = "rgba(0,0,0,0.55)";
    c.lineWidth = 3 * d;
    c.stroke();
    c.strokeStyle = "rgba(255,255,255,0.92)";
    c.lineWidth = 1.3 * d;
    c.stroke();
  }

  /** Evenly spaced streamlines of u in the visible region (cached in world coords). */
  computeStreamlines() {
    if (!this.mesh || !this.mesh.vectors.u || !this.opts.stream) return;
    const [fx, fy] = this.mesh.vectors.u.map((n) => this.mesh.fields[n]);
    const vb = this.visibleBox(), db = this.mesh.dataBox;
    const box = [Math.max(vb[0], db[0]), Math.min(vb[1], db[1]), Math.max(vb[2], db[2]), Math.min(vb[3], db[3])];
    if (box[1] <= box[0] || box[3] <= box[2]) { this.streamCache = []; this.redraw(); return; }
    const dsep = (26 * this.size.dpr) / this.view.ppu;
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
    let budget = 80000;
    const trace = (x, y, dir, id) => {
      const pts = [];
      for (let k = 0; k < 400; k++) {
        const v1 = vel(x, y);
        if (!v1) break;
        const v2 = vel(x + dir * v1[0] * hstep * 0.5, y + dir * v1[1] * hstep * 0.5);
        if (!v2) break;
        x += dir * v2[0] * hstep;
        y += dir * v2[1] * hstep;
        if (cellOf(x, y) < 0 || !free(x, y, id)) break;
        pts.push(x, y);
        if (--budget <= 0) break;
      }
      return pts;
    };
    const lines = [];
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
        const cc = cellOf(pts[k], pts[k + 1]);
        if (cc >= 0) occ[cc] = id;
      }
      lines.push({ pts: new Float32Array(pts), mid: bwd.length });
    }
    this.streamCache = lines;
    this.redraw();
  }

  drawStreamlines(c) {
    if (!this.streamCache) { this.scheduleStream(); return; }
    const d = this.size.dpr;
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
    pass("rgba(255,255,255,0.88)", 1.1 * d);
    for (const ln of this.streamCache) {
      const p = ln.pts, k = Math.min(ln.mid, p.length - 4);
      const [ax, ay] = this.toScreen(p[k], p[k + 1]);
      const [bx, by] = this.toScreen(p[k + 2], p[k + 3]);
      if (Math.hypot(bx - ax, by - ay) < 1e-3) continue;
      this.arrow(c, ax, ay, bx, by, 5 * d);
    }
  }

  /** Ticks and labels along the left and bottom edges, named from coord_names. */
  drawAxes(c) {
    const { w, h: hh, dpr } = this.size;
    const [x0, x1, y0, y1] = this.visibleBox();
    const xt = niceTicks(x0, x1, Math.max(2, Math.round(w / (110 * dpr))));
    const yt = niceTicks(y0, y1, Math.max(2, Math.round(hh / (90 * dpr))));
    const fg = cssVar("--axis-fg") || "#ccd";
    const [nx, ny] = this.mesh.coordNames;
    c.font = `${11 * dpr}px ${cssVar("--font-mono") || "monospace"}`;
    c.fillStyle = cssVar("--axis-bg") || "rgba(0,0,0,0.5)";
    const band = 20 * dpr, left = 48 * dpr;
    c.fillRect(0, hh - band, w, band);
    c.fillRect(0, 0, left, hh - band);
    c.strokeStyle = fg;
    c.fillStyle = fg;
    c.lineWidth = dpr;
    c.textBaseline = "middle";
    c.textAlign = "center";
    c.beginPath();
    for (const t of xt) {
      const [sx] = this.toScreen(t, 0);
      if (sx < left + 10 * dpr || sx > w - 40 * dpr) continue;
      c.moveTo(sx, hh - band);
      c.lineTo(sx, hh - band + 4 * dpr);
      c.fillText(fmt(this.coordX(t) === 0 ? 0 : this.coordX(t), 3), sx, hh - band / 2 + 1.5 * dpr);
    }
    c.textAlign = "right";
    for (const t of yt) {
      const [, sy] = this.toScreen(0, t);
      if (sy < 22 * dpr || sy > hh - band - 6 * dpr) continue;
      c.moveTo(left, sy);
      c.lineTo(left - 4 * dpr, sy);
      c.fillText(fmt(t, 3), left - 7 * dpr, sy);
    }
    c.stroke();
    // axis names in the UI font, bold, in the corners of the tick bands
    c.font = `600 ${12 * dpr}px ${cssVar("--font-sans") || "sans-serif"}`;
    c.textAlign = "right";
    c.fillText(nx, w - 12 * dpr, hh - band / 2 + 1.5 * dpr);
    c.textAlign = "center";
    c.fillText(ny, left / 2, 11 * dpr);
  }

  drawColorbar() {
    const e = this.el;
    const name = this.opts.field;
    e.cbarGrad.style.background = colormapCSS(this.cmapName, "to top", this.reversed);
    const label = this.mesh.labels[name];
    setChildren(e.cbarTitle, h("span.cbar-name", { text: prettyName(name) }),
      this.isLog ? h("span.cbar-tag", { text: "log scale" }) : null,
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

  savePNG(fileBase) {
    if (!this.mesh || !this.gl) return;
    this.drawNow();
    const out = document.createElement("canvas");
    out.width = this.size.w;
    out.height = this.size.h;
    const c = out.getContext("2d");
    c.fillStyle = cssVar("--plot-bg") || "#0a0d12";
    c.fillRect(0, 0, out.width, out.height);
    c.drawImage(this.el.glCanvas, 0, 0);
    c.drawImage(this.el.ovCanvas, 0, 0);
    const a = h("a", { href: out.toDataURL("image/png"), download: `${fileBase || "field"}_${this.opts.field}.png` });
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  themeChanged() {
    this.redraw();
  }
}
