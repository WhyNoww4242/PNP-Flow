// Template parameters: which construction step owns each one, and the editor used
// on the step pages (Geometry, Mesh, Materials, Boundaries, Equations, Study) and in
// the Data tab of the property panel (Results), where all of them are listed by step.
//
// Step ownership comes from the solver (`run.py --list` -> "steps", from
// pnpflow/project.py STEP_RULES); the regex list below is only a fallback for an older
// solver. Editing updates doc.edits / doc.invalid and redraws only the edited row, so
// focus and scroll position survive; app.paramsChanged(doc) does the rest (case save,
// step tabs, toolbar, page hooks).
//
// Units: every value is stored in its template unit; the unit menu next to a field
// changes the unit it is shown and typed in (units.js). Case documents remember the
// choice in case.json "units"; other documents only for this session.

import { h, fmt, stripUnit, sameValue } from "./util.js";
import { icon } from "./icons.js";
import { unitChoices, convert, canonUnit } from "./units.js";

const STEP_RULES = [
  ["mesh", /^(h|h_\w+|\w+_size|n_elem\w*|refine\w*)$/],
  ["geometry", /^(R|R_out|radius|length|width|half_width|upstream|downstream|L|H|W|L_nm|a_ref\w*|alpha_\w+|sdl_\w+|domain\w*|far_\w+)$/],
  ["materials", /^(T|delta|d_\w+|D|eps\w*|c0\w*|bulk\w*|z_\w+|q)$/],
  ["boundaries", /^(zeta\w*|phi_w\w*|stern\w*|side_slip|phi_end\w*|outer_bc|sigma\w*)$/],
  ["equations", /^(Pe|Re|K|Ex|advection|supg)$/],
];

export const STEP_LABEL = { geometry: "Geometry", mesh: "Mesh", materials: "Materials", boundaries: "Boundaries", equations: "Equations", study: "Study" };

export function stepOf(key, steps) {
  if (steps && steps[key]) return steps[key];
  for (const [step, re] of STEP_RULES) if (re.test(key)) return step;
  return "study";
}

/** Parameter rows of one step (app.paramRows filtered). */
export function rowsOf(app, doc, step) {
  const steps = app.caseOf(doc).steps;
  return app.paramRows(doc).filter((r) => stepOf(r.key, steps) === step);
}

function parseParam(raw, dflt) {
  if (typeof dflt === "boolean" || typeof dflt === "string") return { ok: true, v: raw };
  const t = String(raw).trim();
  let v;
  try { v = JSON.parse(t); } catch { v = NaN; }
  if (typeof dflt === "number" || dflt === undefined) {
    if (typeof v === "number" && Number.isFinite(v)) return { ok: true, v };
    return { ok: false, err: "Enter a number, for example 0.1 or 2.5e-3" };
  }
  return t ? { ok: true, v: Number.isNaN(v) ? t : v } : { ok: false, err: "Enter a value" };
}

/** The value the document started from (case or run input), else the template default. */
function baseValue(doc, k, dflt) {
  const inp = doc.kind === "case" && doc.caseData ? doc.caseData.params : doc.summary && doc.summary.input;
  return inp && k in inp ? inp[k] : dflt;
}

export function setParam(app, doc, k, raw, fromText, unit) {
  const r = app.paramRows(doc).find((x) => x.key === k);
  const p = fromText ? parseParam(raw, r ? r.dflt : undefined) : { ok: true, v: raw };
  doc.invalid = doc.invalid || {};
  if (!p.ok) { doc.invalid[k] = { raw, err: p.err }; return false; }
  delete doc.invalid[k];
  // typed in a display unit: store in the template unit
  const v = r && unit && typeof p.v === "number" ? convert(p.v, unit, r.unit) : p.v;
  if (sameValue(v, baseValue(doc, k, r ? r.dflt : undefined))) delete doc.edits[k]; else doc.edits[k] = v;
  return true;
}

export function resetParam(app, doc, k) {
  const r = app.paramRows(doc).find((x) => x.key === k);
  if (doc.invalid) delete doc.invalid[k];
  if (r && r.dflt !== undefined) {
    if (sameValue(baseValue(doc, k, r.dflt), r.dflt)) delete doc.edits[k]; else doc.edits[k] = r.dflt;
  } else delete doc.edits[k];
}

/** Unit a parameter is shown in for this document. */
export function shownUnit(doc, r) {
  const units = (doc.kind === "case" && doc.caseData && doc.caseData.units) || doc.units || {};
  const u = units[r.key];
  return u && unitChoices(r.unit).includes(canonUnit(u)) ? canonUnit(u) : canonUnit(r.unit);
}

function setShownUnit(app, doc, key, unit) {
  if (doc.kind === "case" && doc.caseData) {
    doc.caseData.units = doc.caseData.units || {};
    doc.caseData.units[key] = unit;
    app.caseChanged(doc);
  } else {
    doc.units = doc.units || {};
    doc.units[key] = unit;
  }
}

const fmtNum = (v) => (typeof v === "number" ? String(Number.parseFloat(v.toPrecision(10))) : JSON.stringify(v));

/**
 * Editable parameter list.
 *   layout "page"  : name | value | help text in one line (step pages)
 *   layout "panel" : name | value, help below (property panel)
 *   layout "grid"  : compact cells, label above the field, help in a tooltip (Study)
 *   readonly       : values shown but not editable (results: their inputs are fixed)
 *   expert         : help text as a tooltip instead of a line (panel only)
 */
export function paramForm(app, doc, rows, { layout = "page", expert = false, readonly = false, empty = "No parameters for this step." } = {}) {
  const form = h("div.pform", { class: `pform-${layout}${readonly ? " readonly" : ""}` });
  if (!rows.length) { form.appendChild(h("div.pempty", { text: empty })); return form; }

  const build = (r) => {
    doc.invalid = doc.invalid || {};
    const bad = doc.invalid[r.key];
    const id = `p-${layout}-${r.key}`;
    const unit = shownUnit(doc, r);
    const choices = unitChoices(r.unit);
    let control;
    if (typeof r.dflt === "boolean" || typeof r.value === "boolean") {
      control = h("label.chk", {}, h("input", { type: "checkbox", id, checked: !!r.value, disabled: readonly || null, onchange: (e) => commit(r.key, e.target.checked, false) }),
        h("span", { text: r.value ? "on" : "off" }));
    } else {
      const shown = typeof r.value === "number" ? convert(r.value, r.unit, unit) : r.value;
      const inp = h("input.inp", {
        type: "text", spellcheck: "false", id, inputmode: typeof r.dflt === "number" ? "decimal" : null, readonly: readonly || null,
        value: bad ? bad.raw : typeof shown === "string" ? shown : fmtNum(shown),
        class: [r.changed ? "changed" : "", bad ? "invalid" : ""].join(" "),
        "aria-invalid": bad ? "true" : null, "aria-describedby": bad ? `${id}-err` : null,
        onchange: (e) => commit(r.key, e.target.value, true, unit),
        onkeydown: (e) => {
          if (e.key === "Enter") e.target.blur();
          if (e.key === "Escape") { e.target.value = fmtNum(shown); e.target.blur(); }
        },
      });
      // unit: a menu when the unit has alternatives, plain text otherwise
      const unitEl = choices.length > 1
        ? h("select.unit-sel", { "aria-label": `Unit of ${r.key}`, "data-tip": `Show ${r.key} in another unit (stored in ${r.unit})`,
          onchange: (e) => { setShownUnit(app, doc, r.key, e.target.value); redraw(r.key); } },
          choices.map((u) => h("option", { value: u, text: u, selected: u === unit || null })))
        : r.unit ? h("span.unit-txt", { text: r.unit }) : null;
      control = h("span.field-wrap", {}, inp, unitEl);
    }
    const reset = !readonly && (r.changed || r.edited)
      ? h("button.icon-btn", { type: "button", "aria-label": `Reset ${r.key} to ${fmt(r.dflt)}`, "data-tip-title": `Reset to the template value ${fmt(r.dflt)}`,
        onclick: () => { resetParam(app, doc, r.key); redraw(r.key); app.paramsChanged(doc); } }, icon("reset", 14))
      : h("span.reset-slot");
    const help = stripUnit(r.info);
    const tipHelp = (expert && layout === "panel") || layout === "grid";
    const label = h("label.pname", { for: id }, r.changed ? h("span.chg-dot", { title: "changed from the template value" }) : null, h("span", { text: r.key }));
    if (tipHelp && r.info) { label.dataset.tipTitle = r.key; label.dataset.tip = help || r.info; }
    return h("div.prow", { class: r.changed ? "changed" : "", dataset: { key: r.key } },
      label,
      h("div.pval", {}, control, reset),
      help && !tipHelp ? h("div.pinfo", { text: help }) : null,
      bad ? h("div.perr", { role: "alert", id: `${id}-err`, text: bad.err }) : null);
  };

  const redraw = (k) => {
    const old = form.querySelector(`.prow[data-key="${CSS.escape(k)}"]`);
    const fresh = app.paramRows(doc).find((x) => x.key === k);
    if (old && fresh) old.replaceWith(build(fresh));
  };
  const commit = (k, raw, fromText, unit) => {
    if (readonly) return;
    const ok = setParam(app, doc, k, raw, fromText, unit);
    redraw(k);
    app.paramsChanged(doc);
    if (!ok) { const el = form.querySelector(`#${CSS.escape(`p-${layout}-${k}`)}`); if (el) el.focus(); }
  };

  for (const r of rows) form.appendChild(build(r));
  return form;
}
