// Mesh data preparation, point location and the WebGL triangle renderer.
//
// The field is drawn as one triangle mesh with a per-vertex scalar attribute;
// the fragment shader maps the *interpolated value* through a 256-entry
// colormap texture, so colors are exact P1 interpolation at any zoom level.

import { colormapLUT } from "./colormaps.js";

const NAN_SENTINEL = 3.0e38; // stands in for NaN/null in GPU buffers

// ---------------------------------------------------------------------------
// Mesh preparation
// ---------------------------------------------------------------------------

/** Convert fields.json into typed arrays + derived topology. */
export function prepareMesh(json) {
  const nodes = json.nodes || [];
  const tris = json.triangles || [];
  const n = nodes.length;
  const xy = new Float32Array(n * 2);
  let xmin = Infinity, xmax = -Infinity, ymin = Infinity, ymax = -Infinity;
  for (let i = 0; i < n; i++) {
    const x = +nodes[i][0], y = +nodes[i][1];
    xy[2 * i] = x;
    xy[2 * i + 1] = y;
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  const tri = new Uint32Array(tris.length * 3);
  let bad = 0;
  for (let t = 0; t < tris.length; t++) {
    for (let k = 0; k < 3; k++) {
      const v = tris[t][k] | 0;
      if (v < 0 || v >= n) bad++;
      tri[3 * t + k] = Math.min(Math.max(v, 0), Math.max(n - 1, 0));
    }
  }
  const fields = {};
  for (const [name, arr] of Object.entries(json.fields || {})) {
    if (!Array.isArray(arr) || arr.length !== n) continue; // one value per node only
    fields[name] = toFloat(arr);
  }
  const bbox = validBox(json.bbox) || [xmin, xmax, ymin, ymax];
  const focus = validBox(json.focus_bbox) || bbox;
  return {
    n, nTri: tris.length, xy, tri, fields, bad,
    bbox, focus, dataBox: [xmin, xmax, ymin, ymax],
    labels: json.labels || {},
    vectors: filterVectors(json.vectors, fields),
    boundaries: prepareBoundaries(json.boundary_edges, n),
    edges: null, // unique interior+boundary edges, built lazily
  };
}

function toFloat(arr) {
  const out = new Float32Array(arr.length);
  for (let i = 0; i < arr.length; i++) {
    const v = arr[i];
    out[i] = v === null || v === undefined ? NaN : +v;
  }
  return out;
}

function validBox(b) {
  return Array.isArray(b) && b.length === 4 && b.every(Number.isFinite) && b[1] > b[0] && b[3] > b[2]
    ? b.map(Number) : null;
}

function filterVectors(vectors, fields) {
  const out = {};
  for (const [name, comps] of Object.entries(vectors || {})) {
    if (Array.isArray(comps) && comps.length === 2 && fields[comps[0]] && fields[comps[1]]) {
      out[name] = comps;
    }
  }
  return out;
}

function prepareBoundaries(be, n) {
  const out = {};
  for (const [name, edges] of Object.entries(be || {})) {
    if (!Array.isArray(edges)) continue;
    const a = new Uint32Array(edges.length * 2);
    let m = 0;
    for (const e of edges) {
      if (e && e[0] < n && e[1] < n && e[0] >= 0 && e[1] >= 0) {
        a[m++] = e[0];
        a[m++] = e[1];
      }
    }
    out[name] = a.subarray(0, m);
  }
  return out;
}

/** Unique edge list (pairs of node indices) for the wireframe overlay. */
export function uniqueEdges(mesh) {
  if (mesh.edges) return mesh.edges;
  const seen = new Set();
  const out = [];
  const t = mesh.tri, n = mesh.n;
  for (let i = 0; i < t.length; i += 3) {
    for (let k = 0; k < 3; k++) {
      let a = t[i + k], b = t[i + ((k + 1) % 3)];
      if (a > b) [a, b] = [b, a];
      const key = a * n + b;
      if (!seen.has(key)) {
        seen.add(key);
        out.push(a, b);
      }
    }
  }
  mesh.edges = new Uint32Array(out);
  return mesh.edges;
}

/** Per-field statistics ignoring NaNs. */
export function fieldStats(values) {
  let min = Infinity, max = -Infinity, minPos = Infinity, nan = 0;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!(v === v)) { nan++; continue; }
    if (v < min) min = v;
    if (v > max) max = v;
    if (v > 0 && v < minPos) minPos = v;
  }
  if (min === Infinity) min = max = 0;
  return { min, max, minPos: minPos === Infinity ? null : minPos, nan };
}

// ---------------------------------------------------------------------------
// Point location: uniform bucket grid over triangle bounding boxes
// ---------------------------------------------------------------------------

export class MeshIndex {
  constructor(mesh) {
    this.mesh = mesh;
    const [x0, x1, y0, y1] = mesh.dataBox;
    const nTri = mesh.nTri;
    const aspect = (x1 - x0) / Math.max(y1 - y0, 1e-12);
    const cells = Math.max(16, Math.min(nTri, 1 << 20));
    this.nx = Math.max(1, Math.round(Math.sqrt(cells * aspect)));
    this.ny = Math.max(1, Math.round(cells / this.nx));
    this.x0 = x0;
    this.y0 = y0;
    this.dx = (x1 - x0) / this.nx || 1;
    this.dy = (y1 - y0) / this.ny || 1;
    this.build();
    this.last = 0; // last hit triangle (fast path for coherent queries)
  }

  cellRange(t) {
    const { xy, tri } = this.mesh;
    let ax = Infinity, bx = -Infinity, ay = Infinity, by = -Infinity;
    for (let k = 0; k < 3; k++) {
      const v = tri[3 * t + k];
      const x = xy[2 * v], y = xy[2 * v + 1];
      ax = Math.min(ax, x); bx = Math.max(bx, x);
      ay = Math.min(ay, y); by = Math.max(by, y);
    }
    const cx = (v) => Math.min(this.nx - 1, Math.max(0, Math.floor((v - this.x0) / this.dx)));
    const cy = (v) => Math.min(this.ny - 1, Math.max(0, Math.floor((v - this.y0) / this.dy)));
    return [cx(ax), cx(bx), cy(ay), cy(by)];
  }

  build() {
    const nCells = this.nx * this.ny;
    const count = new Uint32Array(nCells + 1);
    const nTri = this.mesh.nTri;
    for (let t = 0; t < nTri; t++) {
      const [i0, i1, j0, j1] = this.cellRange(t);
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) count[j * this.nx + i + 1]++;
    }
    for (let c = 0; c < nCells; c++) count[c + 1] += count[c];
    const items = new Uint32Array(count[nCells]);
    const fill = count.slice(0, nCells);
    for (let t = 0; t < nTri; t++) {
      const [i0, i1, j0, j1] = this.cellRange(t);
      for (let j = j0; j <= j1; j++) for (let i = i0; i <= i1; i++) items[fill[j * this.nx + i]++] = t;
    }
    this.start = count;
    this.items = items;
  }

  /** Barycentric coordinates of (x,y) in triangle t, or null if outside. */
  bary(t, x, y) {
    const { xy, tri } = this.mesh;
    const a = tri[3 * t], b = tri[3 * t + 1], c = tri[3 * t + 2];
    const ax = xy[2 * a], ay = xy[2 * a + 1];
    const v0x = xy[2 * b] - ax, v0y = xy[2 * b + 1] - ay;
    const v1x = xy[2 * c] - ax, v1y = xy[2 * c + 1] - ay;
    const det = v0x * v1y - v1x * v0y;
    if (det === 0) return null;
    const px = x - ax, py = y - ay;
    const l1 = (px * v1y - v1x * py) / det;
    const l2 = (v0x * py - px * v0y) / det;
    const l0 = 1 - l1 - l2;
    const eps = -1e-7;
    if (l0 < eps || l1 < eps || l2 < eps) return null;
    return [a, b, c, l0, l1, l2];
  }

  /** Locate (x,y): returns [a,b,c,l0,l1,l2,tri] or null outside the mesh. */
  locate(x, y) {
    const hit = this.bary(this.last, x, y);
    if (hit) { hit.push(this.last); return hit; }
    const i = Math.floor((x - this.x0) / this.dx);
    const j = Math.floor((y - this.y0) / this.dy);
    if (i < 0 || j < 0 || i >= this.nx || j >= this.ny) return null;
    const c = j * this.nx + i;
    for (let s = this.start[c]; s < this.start[c + 1]; s++) {
      const t = this.items[s];
      const r = this.bary(t, x, y);
      if (r) {
        this.last = t;
        r.push(t);
        return r;
      }
    }
    return null;
  }

  /** Interpolate nodal array `f` at a located point. */
  static interp(f, loc) {
    return f[loc[0]] * loc[3] + f[loc[1]] * loc[4] + f[loc[2]] * loc[5];
  }
}

// ---------------------------------------------------------------------------
// WebGL renderer
// ---------------------------------------------------------------------------

const VS_FIELD = `
attribute vec2 a_pos;
attribute float a_val;
uniform vec2 u_center;
uniform vec2 u_scale;
varying float v_val;
void main() {
  v_val = a_val;
  gl_Position = vec4((a_pos - u_center) * u_scale, 0.0, 1.0);
}`;

const FS_FIELD = `
precision highp float;
varying float v_val;
uniform sampler2D u_lut;
uniform float u_lo;
uniform float u_hi;
uniform float u_log;
uniform vec3 u_nan;
void main() {
  float v = v_val;
  if (v > 1.0e37 || v < -1.0e37) { gl_FragColor = vec4(u_nan, 1.0); return; }
  if (u_log > 0.5) v = log(max(v, 1.0e-37)) * 0.4342944819;
  float t = clamp((v - u_lo) / (u_hi - u_lo), 0.0, 1.0);
  gl_FragColor = texture2D(u_lut, vec2(t * (255.0 / 256.0) + 0.5 / 256.0, 0.5));
}`;

const VS_LINE = `
attribute vec2 a_pos;
uniform vec2 u_center;
uniform vec2 u_scale;
void main() { gl_Position = vec4((a_pos - u_center) * u_scale, 0.0, 1.0); }`;

const FS_LINE = `
precision mediump float;
uniform vec4 u_color;
void main() { gl_FragColor = u_color; }`;

export class GLRenderer {
  constructor(canvas) {
    const opts = { antialias: true, alpha: true, preserveDrawingBuffer: false, premultipliedAlpha: false };
    this.canvas = canvas;
    this.gl = canvas.getContext("webgl2", opts) || canvas.getContext("webgl", opts);
    if (!this.gl) throw new Error("WebGL is not available in this browser");
    const gl = this.gl;
    this.isGL2 = typeof WebGL2RenderingContext !== "undefined" && gl instanceof WebGL2RenderingContext;
    if (!this.isGL2 && !gl.getExtension("OES_element_index_uint")) {
      throw new Error("WebGL lacks 32-bit index support (OES_element_index_uint)");
    }
    this.fieldProg = this.program(VS_FIELD, FS_FIELD);
    this.lineProg = this.program(VS_LINE, FS_LINE);
    this.buf = { pos: gl.createBuffer(), val: gl.createBuffer(), tri: gl.createBuffer(), edge: gl.createBuffer() };
    this.lut = gl.createTexture();
    this.nTriIdx = 0;
    this.nEdgeIdx = 0;
  }

  program(vsSrc, fsSrc) {
    const gl = this.gl;
    const compile = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
      return s;
    };
    const p = gl.createProgram();
    gl.attachShader(p, compile(gl.VERTEX_SHADER, vsSrc));
    gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fsSrc));
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(p));
    const loc = (name) => gl.getUniformLocation(p, name);
    return { p, loc, aPos: gl.getAttribLocation(p, "a_pos"), aVal: gl.getAttribLocation(p, "a_val") };
  }

  setMesh(mesh) {
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.pos);
    gl.bufferData(gl.ARRAY_BUFFER, mesh.xy, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buf.tri);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.tri, gl.STATIC_DRAW);
    this.nTriIdx = mesh.tri.length;
    this.nEdgeIdx = 0;
    this.mesh = mesh;
  }

  setEdges(edges) {
    const gl = this.gl;
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buf.edge);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, edges, gl.STATIC_DRAW);
    this.nEdgeIdx = edges.length;
  }

  /** Upload the scalar values of the shown field (NaN -> sentinel). */
  setValues(values) {
    const gl = this.gl;
    const v = new Float32Array(values.length);
    for (let i = 0; i < values.length; i++) v[i] = values[i] === values[i] ? values[i] : NAN_SENTINEL;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.val);
    gl.bufferData(gl.ARRAY_BUFFER, v, gl.DYNAMIC_DRAW);
  }

  setColormap(name) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, this.lut);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, colormapLUT(name));
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }

  resize(w, h) {
    if (this.canvas.width !== w || this.canvas.height !== h) {
      this.canvas.width = w;
      this.canvas.height = h;
    }
  }

  /**
   * Draw. view = {cx, cy, ppu} (world center, device pixels per unit).
   * style = {lo, hi, log, bg:[r,g,b], nan:[r,g,b], edges:bool, edgeColor:[r,g,b,a]}
   */
  draw(view, style) {
    const gl = this.gl;
    const W = this.canvas.width, H = this.canvas.height;
    gl.viewport(0, 0, W, H);
    gl.clearColor(0, 0, 0, 0); // transparent: the CSS hatch shows "outside the mesh"
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (!this.mesh || !this.nTriIdx) return;
    const sx = (2 * view.ppu) / W, sy = (2 * view.ppu) / H;
    const idxType = gl.UNSIGNED_INT;

    // field
    const fp = this.fieldProg;
    gl.useProgram(fp.p);
    gl.uniform2f(fp.loc("u_center"), view.cx, view.cy);
    gl.uniform2f(fp.loc("u_scale"), sx, sy);
    gl.uniform1f(fp.loc("u_lo"), style.lo);
    gl.uniform1f(fp.loc("u_hi"), style.hi > style.lo ? style.hi : style.lo + 1e-30);
    gl.uniform1f(fp.loc("u_log"), style.log ? 1 : 0);
    gl.uniform3f(fp.loc("u_nan"), ...style.nan);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.lut);
    gl.uniform1i(fp.loc("u_lut"), 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.pos);
    gl.enableVertexAttribArray(fp.aPos);
    gl.vertexAttribPointer(fp.aPos, 2, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.val);
    gl.enableVertexAttribArray(fp.aVal);
    gl.vertexAttribPointer(fp.aVal, 1, gl.FLOAT, false, 0, 0);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buf.tri);
    gl.drawElements(gl.TRIANGLES, this.nTriIdx, idxType, 0);
    gl.disableVertexAttribArray(fp.aVal);

    // wireframe
    if (style.edges && this.nEdgeIdx) {
      const lp = this.lineProg;
      gl.useProgram(lp.p);
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
      gl.uniform2f(lp.loc("u_center"), view.cx, view.cy);
      gl.uniform2f(lp.loc("u_scale"), sx, sy);
      gl.uniform4f(lp.loc("u_color"), ...style.edgeColor);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buf.pos);
      gl.enableVertexAttribArray(lp.aPos);
      gl.vertexAttribPointer(lp.aPos, 2, gl.FLOAT, false, 0, 0);
      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, this.buf.edge);
      gl.drawElements(gl.LINES, this.nEdgeIdx, idxType, 0);
      gl.disable(gl.BLEND);
    }
  }

  dispose() {
    const gl = this.gl;
    for (const b of Object.values(this.buf)) gl.deleteBuffer(b);
    gl.deleteTexture(this.lut);
    const lose = gl.getExtension("WEBGL_lose_context");
    if (lose) lose.loseContext();
  }
}
