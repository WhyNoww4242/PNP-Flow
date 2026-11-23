// Gallery, log viewer, live-progress strip and the launch panel.

import { api } from "./api.js";
import { h, $, setChildren, placeholder, fmt, fmtDuration, fmtBytes, store } from "./util.js";
import { baseLayout, plot, SERIES_COLORS } from "./plots.js";
import { convergenceSeries } from "./summary.js";

// ---------------------------------------------------------------------------
// Gallery
// ---------------------------------------------------------------------------
export class GalleryView {
  constructor(root) {
    this.root = root;
    this.lightbox = null;
  }

  render(runName, plots) {
    this.runName = runName;
    this.plots = plots || [];
    if (!this.plots.length) {
      return setChildren(this.root, placeholder("empty", "No plots", "This run has no plots/*.png images."));
    }
    setChildren(this.root, h("div.gallery", {}, this.plots.map((p, i) => h("figure.thumb", {
      tabindex: "0", onclick: () => this.open(i), onkeydown: (e) => e.key === "Enter" && this.open(i),
    }, h("div.thumb-img", {}, h("img", { src: api.plotURL(runName, p), alt: p, loading: "lazy" })),
    h("figcaption.mono", { text: p.replace(/\.(png|jpe?g|svg)$/i, "").replace(/_/g, " ") })))));
  }

  open(i) {
    this.close();
    this.idx = (i + this.plots.length) % this.plots.length;
    const p = this.plots[this.idx];
    const url = api.plotURL(this.runName, p);
    this.lightbox = h("div.lightbox", { onclick: (e) => e.target === this.lightbox && this.close() },
      h("div.lb-frame", {},
        h("div.lb-head", {},
          h("span.mono", { text: p }),
          h("span.dim.small", { text: `${this.idx + 1} / ${this.plots.length}  ·  ← → to browse, Esc to close` }),
          h("a.btn.btn-sm", { href: url, target: "_blank", text: "open" }),
          h("button.icon-btn", { text: "×", title: "Close", onclick: () => this.close() })),
        h("div.lb-body", {}, h("img", { src: url, alt: p }))));
    document.body.appendChild(this.lightbox);
  }

  close() {
    if (this.lightbox) this.lightbox.remove();
    this.lightbox = null;
  }

  /** Keyboard handler while the lightbox is open; returns true if consumed. */
  key(e) {
    if (!this.lightbox) return false;
    if (e.key === "Escape") this.close();
    else if (e.key === "ArrowRight") this.open(this.idx + 1);
    else if (e.key === "ArrowLeft") this.open(this.idx - 1);
    else return false;
    return true;
  }
}

// ---------------------------------------------------------------------------
// Log viewer (tail of stdout.log, auto-follow)
// ---------------------------------------------------------------------------
export class LogView {
  constructor(root) {
    this.root = root;
    this.follow = true;
    this.pre = h("pre.log.mono");
    this.meta = h("span.dim.small");
    this.followChk = h("input", { type: "checkbox", checked: true, onchange: (e) => { this.follow = e.target.checked; } });
    setChildren(root, h("div.logview", {},
      h("div.toolbar", {}, h("div.tb-group", {}, h("span.tb-label", { text: "stdout.log" }), this.meta),
        h("div.tb-spacer"),
        h("label.chk", {}, this.followChk, h("span", { text: "follow" })),
        h("button.btn.btn-sm", { text: "Refresh", onclick: () => this.refresh() })),
      this.pre));
  }

  async load(runName) {
    this.runName = runName;
    this.lastSize = -1;
    this.pre.textContent = "";
    await this.refresh();
  }

  async refresh() {
    if (!this.runName) return;
    const name = this.runName;
    try {
      const r = await api.log(name, 256 * 1024);
      if (name !== this.runName) return;
      if (!r.size) {
        this.pre.textContent = "";
        setChildren(this.meta, "no output yet");
        this.pre.appendChild(placeholder("empty", "No log", "stdout.log does not exist or is empty."));
        return;
      }
      if (r.size !== this.lastSize) {
        this.pre.textContent = r.text;
        this.lastSize = r.size;
        if (this.follow) this.pre.scrollTop = this.pre.scrollHeight;
      }
      this.meta.textContent = `${fmtBytes(r.size)}${r.size > 256 * 1024 ? " (showing tail)" : ""}`;
    } catch (e) {
      this.meta.textContent = `error: ${e.message}`;
    }
  }
}

// ---------------------------------------------------------------------------
// Live progress strip for running jobs
// ---------------------------------------------------------------------------
export const STATE_LABEL = {
  meshing: "meshing", assembling: "assembling", solving: "solving", post: "post-processing",
  done: "done", failed: "failed", launched: "starting", unknown: "unknown",
};
export const RUNNING_STATES = new Set(["meshing", "assembling", "solving", "post", "launched"]);

/** Render a progress panel (stage bar, numbers, mini convergence) into `root`. */
export function renderProgress(root, progress, entry) {
  const p = progress || {};
  const state = (entry && entry.state) || p.state || "unknown";
  const nSt = p.n_stages || 0, st = p.stage || 0;
  const frac = nSt ? Math.min(1, Math.max(0, (st - (state === "done" ? 0 : 0.5)) / nSt)) : null;
  const stats = [
    ["state", h(`span.badge.s-${state}`, { text: STATE_LABEL[state] || state })],
    ["stage", nSt ? `${st} / ${nSt}` : st || "—"],
    ["iteration", p.iteration ?? "—"],
    ["‖R‖", p.res_norm !== undefined ? fmt(p.res_norm, 3) : "—"],
    ["elapsed", fmtDuration(p.time_s)],
  ];
  const chart = h("div.chart.chart-live");
  setChildren(root, h("div.progress-panel", {},
    h("div.pp-stats", {}, stats.map(([k, v]) => h("div.pp-stat", {}, h("span.pp-k", { text: k }), h("span.pp-v.mono", {}, v)))),
    h("div.pp-bar", { title: nSt ? `stage ${st} of ${nSt}` : "" },
      h("div.pp-fill", { class: RUNNING_STATES.has(state) ? "anim" : "", style: { width: frac === null ? (RUNNING_STATES.has(state) ? "100%" : "0") : `${frac * 100}%` } })),
    h("div.pp-msg.mono", { text: p.message || "" }),
    chart));
  const series = convergenceSeries(null, p);
  if (series.length) {
    let x0 = 0;
    plot(chart, series.map((s, i) => {
      const xs = s.its.map((_, k) => x0 + k);
      x0 += s.its.length;
      return {
        x: xs, y: s.its.map((it) => it.res_norm), type: "scatter", mode: "lines+markers", name: `stage ${s.stage}`,
        line: { color: SERIES_COLORS[i % SERIES_COLORS.length], width: 2 }, marker: { size: 5 },
      };
    }), baseLayout({
      margin: { l: 52, r: 8, t: 6, b: 30 }, yaxis: { type: "log", title: { text: "‖R‖" } },
      xaxis: { title: { text: "Newton iteration" } }, legend: { orientation: "h", y: -0.35 },
    }), { displayModeBar: false });
  } else {
    chart.appendChild(placeholder("empty", "No iterations yet"));
  }
}

// ---------------------------------------------------------------------------
// Launch panel
// ---------------------------------------------------------------------------
export class LaunchView {
  constructor(root, { onLaunched }) {
    this.root = root;
    this.onLaunched = onLaunched;
    this.cases = null;
    this.state = store.get("launch", { case: null });
  }

  async show() {
    setChildren(this.root, placeholder("loading", "Loading cases…", "running python run.py --list"));
    let res;
    try {
      res = await api.cases();
    } catch (e) {
      res = { cases: {}, error: e.message };
    }
    this.cases = res.cases || {};
    this.error = res.error;
    this.render();
  }

  render() {
    const names = Object.keys(this.cases);
    if (!names.length) {
      const detail = h("span", {}, this.error || "run.py --list returned no cases.", h("br"),
        "Launching needs ", h("code", { text: "run.py" }), " in the pnpflow root (",
        h("code", { text: "python run.py --list" }), ").");
      const retry = h("button.btn.btn-sm", { text: "Retry", onclick: () => this.show() });
      return setChildren(this.root, h("div.launch.launch-empty", {},
        h("section.card", {}, placeholder("empty", "No cases available", detail), h("div.center", {}, retry))));
    }
    if (!names.includes(this.state.case)) this.state.case = names[0];
    const caseSel = h("div.case-list", {}, names.map((n) => h("button.case-btn", {
      class: n === this.state.case ? "active" : "",
      onclick: () => { this.state.case = n; store.set("launch", this.state); this.render(); },
    }, h("span.mono", { text: n }), h("span.dim.small", { text: this.caseDescription(n) }))));
    const c = this.cases[this.state.case] || {};
    const params = c.params || {};
    const info = c.info || {};
    this.inputs = {};
    const rows = Object.entries(params).map(([k, v]) => {
      const input = this.paramInput(k, v);
      this.inputs[k] = { input, dflt: v };
      return h("tr", {},
        h("th", {}, h("label.param", { text: k, for: `p-${k}` })),
        h("td", {}, input),
        h("td.dim.small", { text: typeof info[k] === "string" ? info[k] : "" }),
        h("td", {}, h("button.icon-btn", { title: `reset to default (${JSON.stringify(v)})`, text: "↺", onclick: () => this.resetParam(k) })));
    });
    const stamp = new Date().toISOString().slice(2, 16).replace(/[-:]/g, "").replace("T", "_");
    this.nameInput = h("input#run-name.mono", { type: "text", value: `${this.state.case}_${stamp}`, spellcheck: "false" });
    this.overwrite = h("input", { type: "checkbox" });
    this.msg = h("div.launch-msg");
    this.cmdPreview = h("code.cmd-preview");
    const doc = typeof c.doc === "string" ? h("p.doc", { text: c.doc.replace(/\s+/g, " ").trim() }) : null;
    setChildren(this.root, h("div.launch", {},
      h("section.card.launch-cases", {}, h("header.card-head", {}, h("h3", { text: "Case" })), h("div.card-body", {}, caseSel)),
      h("section.card.launch-form", {},
        h("header.card-head", {}, h("h3", { text: `Parameters · ${this.state.case}` }),
          h("button.btn.btn-sm", { text: "Reset all", onclick: () => Object.keys(params).forEach((k) => this.resetParam(k)) })),
        h("div.card-body", {}, doc,
          rows.length ? h("table.kv.param-form", {}, h("tbody", {}, rows)) : h("div.dim", { text: "This case has no parameters." }),
          h("div.launch-row", {},
            h("label.field", {}, h("span.tb-label", { text: "Run name" }), this.nameInput),
            h("label.chk", { title: "Allow re-using an existing results folder" }, this.overwrite, h("span", { text: "overwrite" })),
            h("button.btn.btn-primary", { text: "Launch run  ▶", onclick: () => this.launch() })),
          this.cmdPreview,
          this.msg))));
    this.root.querySelectorAll("input").forEach((el) => el.addEventListener("input", () => this.updatePreview()));
    this.updatePreview();
  }

  caseDescription(n) {
    const c = this.cases[n] || {};
    if (typeof c.description === "string") return c.description;
    if (typeof c.doc === "string") return c.doc.trim().split(/\r?\n/)[0];
    if (c.info && typeof c.info._case === "string") return c.info._case;
    return `${Object.keys(c.params || {}).length} parameters`;
  }

  paramInput(k, v) {
    const id = `p-${k}`;
    if (typeof v === "boolean") {
      const el = h("input", { id, type: "checkbox", checked: v });
      el.dataset.kind = "bool";
      return el;
    }
    const el = h("input.mono", {
      id, type: "text", spellcheck: "false",
      value: typeof v === "string" ? v : JSON.stringify(v),
    });
    el.dataset.kind = typeof v === "number" ? "num" : typeof v === "string" ? "str" : "json";
    return el;
  }

  resetParam(k) {
    const { input, dflt } = this.inputs[k];
    if (input.dataset.kind === "bool") input.checked = dflt;
    else input.value = typeof dflt === "string" ? dflt : JSON.stringify(dflt);
    input.classList.remove("changed", "invalid");
    this.updatePreview();
  }

  /** Collect params that differ from the defaults; mark invalid inputs. */
  collect() {
    const out = {};
    let ok = true;
    for (const [k, { input, dflt }] of Object.entries(this.inputs)) {
      let v;
      const kind = input.dataset.kind;
      if (kind === "bool") v = input.checked;
      else if (kind === "str") v = input.value;
      else {
        const raw = input.value.trim();
        try {
          v = JSON.parse(raw);
        } catch {
          v = kind === "num" ? NaN : raw;
        }
        if (kind === "num" && !(typeof v === "number" && Number.isFinite(v))) {
          ok = false;
          input.classList.add("invalid");
          continue;
        }
      }
      input.classList.remove("invalid");
      const changed = JSON.stringify(v) !== JSON.stringify(dflt);
      input.classList.toggle("changed", changed);
      if (changed) out[k] = v;
    }
    return { params: out, ok };
  }

  updatePreview() {
    const { params } = this.collect();
    const sets = Object.entries(params).map(([k, v]) => ` --set ${k}=${typeof v === "string" ? v : JSON.stringify(v)}`).join("");
    this.cmdPreview.textContent = `$ python run.py ${this.state.case} --name ${this.nameInput.value.trim()}${sets}`;
  }

  async launch() {
    const { params, ok } = this.collect();
    const name = this.nameInput.value.trim();
    if (!ok) return this.message("error", "Fix the highlighted parameters (numbers expected).");
    if (!/^[A-Za-z0-9_][A-Za-z0-9_.-]{0,127}$/.test(name)) {
      return this.message("error", "Run name: letters, digits, '_', '.', '-' only.");
    }
    if (!this.state.case || !this.cases[this.state.case]) return this.message("error", "No case selected: the case list did not load.");
    this.message("info", "Launching…");
    try {
      const r = await api.launch({ case: this.state.case, name, params, overwrite: this.overwrite.checked });
      this.message("ok", `Started '${name}' (pid ${r.pid}).`);
      store.set("launch", this.state);
      this.onLaunched(name);
    } catch (e) {
      this.message("error", e.message + (/exists/.test(e.message) ? " — tick 'overwrite' or choose another name." : ""));
    }
  }

  message(kind, text) {
    setChildren(this.msg, h(`div.msg.msg-${kind}`, { text }));
  }
}
