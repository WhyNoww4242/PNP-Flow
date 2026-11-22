// Profiles panel: Plotly line plots of profiles.json, with an optional
// overlay of the same line from a second run (compare mode).

import { h, setChildren, placeholder, prettyName, store, isSignedFieldName } from "./util.js";
import { baseLayout, plot, SERIES_COLORS } from "./plots.js";

export class ProfilesView {
  constructor(root, { loadCompare, runNames }) {
    this.root = root;
    this.loadCompare = loadCompare; // async (name) => profiles json | null
    this.runNames = runNames;       // () => [names]
    this.data = null;
    this.cmp = null;
    this.state = Object.assign({ line: null, vars: {}, logy: false, logx: false, normalize: false, split: true, compare: "" },
      store.get("profiles", {}));
  }

  persist() {
    store.set("profiles", { ...this.state, compare: "" });
  }

  setData(runName, data) {
    this.runName = runName;
    this.data = data && Array.isArray(data.lines) ? data : null;
    this.cmp = null;
    this.state.compare = "";
    this.render();
  }

  showMessage(kind, title, detail) {
    setChildren(this.root, placeholder(kind, title, detail));
  }

  get lines() {
    return this.data ? this.data.lines.filter((l) => l && Array.isArray(l.coord)) : [];
  }

  currentLine() {
    const lines = this.lines;
    return lines.find((l) => l.name === this.state.line) || lines[0];
  }

  /** Variables chosen for the current line (default: first three). */
  chosenVars(line) {
    const avail = Object.keys(line.values || {});
    let chosen = (this.state.vars[line.name] || []).filter((v) => avail.includes(v));
    if (!chosen.length) chosen = avail.slice(0, Math.min(3, avail.length));
    return chosen;
  }

  render() {
    if (!this.data) return this.showMessage("empty", "No profiles", "This run has no profiles.json.");
    const lines = this.lines;
    if (!lines.length) return this.showMessage("empty", "No profile lines", "profiles.json contains no lines.");
    const line = this.currentLine();
    this.state.line = line.name;
    const chosen = this.chosenVars(line);

    const lineList = h("div.prof-lines", {}, lines.map((l) => h("button.prof-line", {
      class: l.name === line.name ? "active" : "",
      title: l.label || l.name,
      onclick: () => { this.state.line = l.name; this.persist(); this.render(); },
    }, h("span.pl-name.mono", { text: l.name }), h("span.pl-label", { text: l.label || "" }),
    h("span.pl-meta.mono", { text: `${prettyName(l.coord_name || "s")} · ${l.coord.length} pts` }))));

    const varChips = h("div.chips", {}, Object.keys(line.values || {}).map((v) => h("button.chip", {
      class: chosen.includes(v) ? "on" : "",
      onclick: () => this.toggleVar(line, v),
      text: prettyName(v),
    })));

    const opt = (key, label, title) => h("label.chk", { title },
      h("input", { type: "checkbox", checked: this.state[key], onchange: (e) => { this.state[key] = e.target.checked; this.persist(); this.drawPlot(); } }),
      h("span", { text: label }));

    const others = this.runNames().filter((n) => n !== this.runName);
    const cmpSel = h("select.prof-compare", {
      onchange: (e) => this.setCompare(e.target.value), title: "Overlay the same line from another run",
    }, h("option", { value: "", text: "— none —" }), others.map((n) => h("option", { value: n, text: n, selected: n === this.state.compare })));

    this.plotEl = h("div.prof-plot");
    this.cmpNote = h("span.dim.small");
    setChildren(this.root, h("div.profiles", {},
      h("aside.prof-side", {},
        h("div.section-label", { text: "Lines" }), lineList),
      h("div.prof-main", {},
        h("div.toolbar", {},
          h("div.tb-group.grow", {}, h("span.tb-label", { text: "Variables" }), varChips),
          h("div.tb-group", {}, opt("logy", "log y", "Logarithmic y axis (|value|)"), opt("logx", "log x", "Logarithmic x axis"),
            opt("split", "split", "One subplot per variable (shared x)"), opt("normalize", "normalize", "Scale each curve by its max |value|"))),
        this.plotEl,
        h("div.toolbar.prof-foot", {},
          h("div.tb-group", {}, h("span.tb-label", { text: "Compare with" }), cmpSel, this.cmpNote)))));
    this.drawPlot();
  }

  toggleVar(line, v) {
    const cur = this.chosenVars(line);
    const next = cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v];
    this.state.vars[line.name] = next.length ? next : cur;
    this.persist();
    this.render();
  }

  async setCompare(name) {
    this.state.compare = name;
    this.cmp = null;
    if (name) {
      this.cmpNote.textContent = "loading…";
      try {
        this.cmp = await this.loadCompare(name);
        this.cmpNote.textContent = this.cmp ? "" : "that run has no profiles.json";
      } catch (e) {
        this.cmpNote.textContent = `failed: ${e.message}`;
      }
    } else this.cmpNote.textContent = "";
    this.drawPlot();
  }

  traceFor(line, v, color, dash, suffix, axisIdx) {
    let y = (line.values[v] || []).map((a) => (a === null || a === undefined ? NaN : +a));
    if (this.state.normalize) {
      const m = Math.max(...y.filter(Number.isFinite).map(Math.abs)) || 1;
      y = y.map((a) => a / m);
    }
    if (this.state.logy) y = y.map((a) => (Math.abs(a) > 0 ? Math.abs(a) : NaN));
    const ax = axisIdx > 1 ? String(axisIdx) : "";
    return {
      x: line.coord, y, type: "scatter", mode: line.coord.length < 60 ? "lines+markers" : "lines",
      name: prettyName(v) + suffix, line: { color, width: 1.8, dash }, marker: { size: 4 },
      xaxis: "x", yaxis: "y" + ax, connectgaps: false,
      hovertemplate: `${prettyName(v)}${suffix}: %{y:.5g}<br>${prettyName(line.coord_name || "s")}: %{x:.5g}<extra></extra>`,
    };
  }

  drawPlot() {
    if (!this.plotEl) return;
    const line = this.currentLine();
    const vars = this.chosenVars(line);
    const cmpLine = this.cmp && Array.isArray(this.cmp.lines) ? this.cmp.lines.find((l) => l.name === line.name) : null;
    if (this.cmp && !cmpLine) this.cmpNote.textContent = `no line '${line.name}' in ${this.state.compare}`;
    else if (cmpLine) this.cmpNote.textContent = "dashed = comparison run";
    const split = this.state.split && vars.length > 1;
    const traces = [];
    vars.forEach((v, i) => {
      const color = SERIES_COLORS[i % SERIES_COLORS.length];
      const axis = split ? i + 1 : 1;
      traces.push(this.traceFor(line, v, color, "solid", "", axis));
      if (cmpLine && cmpLine.values && cmpLine.values[v]) {
        traces.push(this.traceFor(cmpLine, v, color, "dash", ` · ${this.state.compare}`, axis));
      }
    });
    const xTitle = prettyName(line.coord_name || "s");
    const extra = {
      xaxis: { title: { text: xTitle }, type: this.state.logx ? "log" : "linear" },
      yaxis: { type: this.state.logy ? "log" : "linear" },
      hovermode: "x unified",
      showlegend: true,
    };
    if (split) {
      const n = vars.length, gap = 0.04;
      const hgt = (1 - gap * (n - 1)) / n;
      extra.grid = { rows: n, columns: 1, pattern: "coupled", roworder: "top to bottom" };
      const lay = baseLayout(extra);
      vars.forEach((v, i) => {
        const key = i === 0 ? "yaxis" : `yaxis${i + 1}`;
        const top = 1 - i * (hgt + gap);
        lay[key] = Object.assign({}, lay.yaxis, {
          domain: [top - hgt, top], title: { text: prettyName(v), font: { size: 12, color: SERIES_COLORS[i % SERIES_COLORS.length] } },
          type: this.state.logy ? "log" : "linear",
          zeroline: !this.state.logy && isSignedFieldName(v),
        });
      });
      lay.margin = Object.assign({}, lay.margin, { l: 70 });
      lay.legend = Object.assign({}, lay.legend, { y: -0.12 });
      plot(this.plotEl, traces, lay);
    } else {
      plot(this.plotEl, traces, baseLayout(extra));
    }
  }
}
