// Condition editor of one boundary (Boundaries page): what the ions and the potential do there.
// It writes case.json "bcs" (applied by runner.apply_case_bcs). A field set back to the
// template's condition is removed again, so "bcs" only holds what differs from the template.

import { h, fmt } from "./util.js";
import { icon } from "./icons.js";
import { boundaryColor } from "./fieldview.js";
import { bcWords, parseBC } from "./bdiagram.js";
import { thermalVoltage } from "./units.js";

const AXIS = /axis|symmetry/;

// [type, label, help]
const TYPES = {
  c: [
    ["blocking", "No flux", "Ions cannot cross this boundary (the default)"],
    ["bulk", "Bulk concentration", "Every ion fixed at its bulk concentration, as at a reservoir"],
    ["fixed", "Fixed concentration", "Every ion fixed at a multiple of its bulk concentration"],
    ["flux", "Set flux", "Outward flux N·n of every ion, dimensionless (D c0 / a)"],
  ],
  phi: [
    ["natural", "No charge", "Zero normal field: an uncharged, insulating boundary (the default)"],
    ["fixed", "Fixed potential", "Potential fixed against the reservoir, as on an electrode"],
    ["stern", "Stern layer", "δ² ∂φ/∂n = C (φ_ref − φ): a compact layer between a metal at φ_ref and the diffuse layer"],
    ["charge", "Surface charge", "δ² ∂φ/∂n = σ, a fixed charge per area (dimensionless)"],
  ],
};
const FIELD_LABEL = { c: "Ions", phi: "Potential" };
const UNITS = [["mV", "mV"], ["V_T", "Φ (V_T)"]];

const num = (v, ctx) => {
  const x = Number(v);
  if (v !== "" && v !== undefined && Number.isFinite(x)) return x;
  const p = ctx.params && ctx.params[v];
  return typeof p === "number" ? p : 0;
};

// a template potential value (number or parameter name) as {[key]: v, unit}
function potOf(v, key, ctx) {
  const inputs = ctx.inputs || {};
  if (typeof v === "string" && Number.isFinite(Number(inputs[`${v}_mV`]))) return { [key]: Number(inputs[`${v}_mV`]), unit: "mV" };
  return { [key]: num(v, ctx), unit: "V_T" };
}

/** The template's condition (solver text) as a case spec, or null when the editor cannot show it (per-ion, flow). */
function fromTemplate(key, text, ctx) {
  if (text === undefined || text === null) return { type: key === "c" ? "blocking" : "natural" };
  if (typeof text === "object") return null;
  const { type, args } = parseBC(text);
  if (type === "blocking") return key === "c" ? { type: "blocking" } : null;
  if (key === "c") {
    if (type === "Dirichlet") return args.value === "bulk" ? { type: "bulk" } : { type: "fixed", value: num(args.value, ctx) };
    if (type === "Flux") return { type: "flux", value: num(args.value, ctx) };
    return null;
  }
  if (type === "Dirichlet") return { type: "fixed", ...potOf(args.value, "value", ctx) };
  if (type === "Robin") return { type: "stern", coef: num(args.coef, ctx), ...potOf(args.ref, "ref", ctx) };
  if (type === "Charge") return { type: "charge", sigma: num(args.sigma, ctx) };
  return null;
}

// the parameter a template condition reads, as the user knows it (phi_w -> phi_w_mV)
function templateParam(text, ctx) {
  if (typeof text !== "string") return null;
  const { args } = parseBC(text);
  const v = args.value ?? args.ref ?? args.sigma;
  if (!v || v === "bulk" || Number.isFinite(Number(v))) return null;
  const inputs = ctx.inputs || {};
  return `${v}_mV` in inputs ? `${v}_mV` : v in inputs ? v : null;
}

const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
const vt = (ctx) => 1000 * thermalVoltage((ctx.inputs || {}).T);         // mV per Φ

function numInput(id, value, label, onValue) {
  const el = h("input.inp.num-inp", { id, type: "text", inputmode: "decimal", spellcheck: "false", "aria-label": label, value: fmt(value, 6) });
  el.addEventListener("change", () => {
    const v = Number(el.value.trim());
    const ok = el.value.trim() !== "" && Number.isFinite(v);
    el.classList.toggle("invalid", !ok);
    el.setAttribute("aria-invalid", ok ? "false" : "true");
    if (ok) onValue(v);
  });
  el.addEventListener("keydown", (e) => { if (e.key === "Enter") el.blur(); });
  return el;
}

const field = (label, control, tip) => h("label.mt-field", { "data-tip": tip || null }, h("span.mt-lbl", { text: label }), control);

/**
 * Editor panel for boundary `name`.
 *   base     : the template's conditions (effectiveBCs(doc).base), solver text per boundary
 *   ctx      : bcContext (params, inputs, si)
 *   onChange : called after case.json "bcs" changed (re-render the page)
 *   onClose  : closes the editor
 */
export function boundaryEditor(app, doc, name, ctx, { base, index, onChange, onClose }) {
  const editable = doc.kind === "case" && !!doc.caseData;
  // looked up on every write: a save replaces doc.caseData with the server's copy without a redraw
  const liveBCs = () => (doc.caseData.bcs = doc.caseData.bcs || {});
  const own = ((editable ? doc.caseData.bcs : doc.runCase && doc.runCase.bcs) || {})[name] || {};
  const tmplText = (k) => (base[name] || {})[k];
  const tmpl = (k) => fromTemplate(k, tmplText(k), ctx);
  const current = (k) => own[k] || tmpl(k);
  const changedKeys = Object.keys(own);
  const axis = AXIS.test(name);
  const flow = doc.summary && doc.summary.model && doc.summary.model.flow && !["none", "prescribed"].includes(doc.summary.model.flow);

  const write = (k, spec) => {
    const set = liveBCs();
    const mine = (set[name] = Object.assign({}, set[name]));
    if (spec === null || same(spec, tmpl(k))) delete mine[k]; else mine[k] = spec;
    if (!Object.keys(mine).length) delete set[name];
    app.caseChanged(doc);
    onChange();
  };

  // a new condition type: the template's own values when it is the template's type, else
  // values carried over from the current condition (a potential stays the same potential)
  const switchTo = (k, type) => {
    const t = tmpl(k);
    if (t && t.type === type) return write(k, null);
    const cur = current(k) || {};
    const unit = cur.unit || ("T" in (ctx.inputs || {}) ? "mV" : "V_T");
    const pot = cur.value ?? cur.ref ?? 0;
    const spec = {
      blocking: { type }, bulk: { type }, natural: { type },
      fixed: k === "c" ? { type, value: 1 } : { type, value: pot, unit },
      flux: { type, value: 0 },
      stern: { type, coef: 1, ref: pot, unit },
      charge: { type, sigma: 0 },
    }[type];
    write(k, spec);
  };

  const withUnit = (k, spec, key, label) => {
    const id = `bce-${k}-${key}`;
    const unitSel = h("select.sel.bce-unit", { id: `${id}-unit`, "aria-label": `Unit of the ${label.toLowerCase()}`,
      onchange: (e) => {
        const u = e.target.value;
        if (u === spec.unit) return;
        const v = u === "mV" ? spec[key] * vt(ctx) : spec[key] / vt(ctx);   // same potential, other unit
        write(k, { ...spec, [key]: Number(v.toPrecision(10)), unit: u });
      } }, UNITS.map(([u, l]) => h("option", { value: u, text: l, selected: u === (spec.unit || "V_T") || null })));
    return field(label, h("span.bce-pair", {}, numInput(id, spec[key], label, (v) => write(k, { ...spec, [key]: v })), unitSel));
  };

  const valueFields = (k, spec) => {
    if (!spec) return [];
    switch (`${k}:${spec.type}`) {
      case "c:fixed": return [field("c / c bulk", numInput("bce-c-value", spec.value, "Concentration over bulk", (v) => write(k, { ...spec, value: v })), "1 = bulk concentration")];
      case "c:flux": return [field("N·n", numInput("bce-c-value", spec.value, "Outward flux", (v) => write(k, { ...spec, value: v })), "Outward flux of every ion, dimensionless; negative flows in")];
      case "phi:fixed": return [withUnit(k, spec, "value", "Potential")];
      case "phi:stern": return [field("C", numInput("bce-phi-coef", spec.coef, "Stern capacitance", (v) => write(k, { ...spec, coef: v })), "Stern capacitance, dimensionless (C_S λ_D / ε)"), withUnit(k, spec, "ref", "Metal potential")];
      case "phi:charge": return [field("σ", numInput("bce-phi-sigma", spec.sigma, "Surface charge", (v) => write(k, { ...spec, sigma: v })), "Surface charge, dimensionless (σ λ_D / (ε V_T))")];
      default: return [];
    }
  };

  const row = (k) => {
    const spec = current(k);
    const changed = k in own;
    const words = bcWords(k, k in own ? own[k] : tmplText(k), ctx);
    if (!editable || !spec) {
      return h("div.bce-row", {}, h("span.bce-name", { text: FIELD_LABEL[k] }), h("span.bce-words", { text: words }),
        !spec && editable ? h("span.small.dim", { text: "This template's condition cannot be changed here." }) : null);
    }
    const opts = TYPES[k];
    const sel = h("select.sel.bce-type", { id: `bce-${k}-type`, "aria-label": `${FIELD_LABEL[k]} condition on ${name}`,
      onchange: (e) => switchTo(k, e.target.value) },
      opts.map(([t, label, help]) => h("option", { value: t, text: label, title: help, selected: t === spec.type || null })));
    const param = changed ? templateParam(tmplText(k), ctx) : null;
    return h("div.bce-row", {},
      h("span.bce-name", { text: FIELD_LABEL[k] }),
      h("div.bce-controls", {},
        field("Condition", sel, (opts.find(([t]) => t === spec.type) || [])[2]),
        ...valueFields(k, spec),
        changed ? h("button.icon-btn.bce-reset", { type: "button", "aria-label": `Reset ${FIELD_LABEL[k].toLowerCase()} on ${name} to the template`,
          "data-tip-title": "Reset to template", "data-tip": bcWords(k, tmplText(k), ctx), onclick: () => write(k, null) }, icon("reset", 14)) : null),
      changed ? h("div.bce-words", {}, h("span", { text: words }),
        h("span.dim", { text: ` · template: ${bcWords(k, tmplText(k), ctx)}${param ? `. ${param} no longer sets this boundary` : ""}` })) : null);
  };

  let body;
  if (axis) body = h("div.bce-words", { text: "Symmetry axis. There is no condition to set: the solution is mirrored here." });
  else body = h("div.bce", {}, row("c"), row("phi"),
    flow ? h("div.small.dim", { text: "Flow conditions stay as the template sets them." }) : null);

  const head = h("header.panel-head", {},
    h("h2", {}, h("span.swatch", { class: axis ? "dashed" : "", style: axis ? { borderColor: boundaryColor(name, index), width: "14px" } : { background: boundaryColor(name, index) } }),
      h("span", { text: `Condition on ${name}` })),
    changedKeys.length ? h("span.badge.b-neutral", { text: editable ? "changed from template" : "differs from template" }) : null,
    h("span.spacer"),
    editable && changedKeys.length ? h("button.btn.sm", { type: "button", onclick: () => { delete liveBCs()[name]; app.caseChanged(doc); onChange(); } }, icon("reset", 14), "Reset to template") : null,
    h("button.icon-btn", { type: "button", "aria-label": "Close the condition editor", "data-tip": "Close (Esc)", onclick: onClose }, icon("x", 14)));
  const foot = editable
    ? h("div.panel-foot", { text: `Saved in cases/${doc.caseName}/case.json. The next solve uses it.` })
    : h("div.panel-foot.row", {}, h("span", { text: "The conditions of this result are read-only. " }),
      h("button.btn-link", { type: "button", onclick: () => app.caseOfRun(doc) }, "Open its project to change them"));
  return h("section.panel#bc-editor", {}, head, h("div.panel-body", {}, body), foot);
}
