// Display units for template parameters. A parameter is stored in its template unit
// (the "[unit]" at the end of its help text); the unit menu only changes the unit
// the value is shown and typed in. Each family lists factors to its first (SI) unit.

const FAMILIES = [
  { name: "length", units: { m: 1, mm: 1e-3, "µm": 1e-6, nm: 1e-9, "Å": 1e-10 } },
  { name: "potential", units: { V: 1, mV: 1e-3, "µV": 1e-6 } },
  { name: "concentration", units: { "mol/m³": 1, mM: 1, M: 1e3, "µM": 1e-3 } },
  { name: "diffusivity", units: { "m²/s": 1, "cm²/s": 1e-4, "µm²/s": 1e-12, "nm²/ns": 1e-9 } },
  { name: "time", units: { s: 1, ms: 1e-3, "µs": 1e-6, ns: 1e-9 } },
  { name: "permittivity", units: { "F/m": 1, "ε_r": 8.8541878128e-12 } },
  // temperature has an offset: handled in toBase / fromBase
  { name: "temperature", units: { K: 1, "°C": 1 } },
];

// spellings the templates use -> canonical unit
const ALIAS = { um: "µm", "m^2/s": "m²/s", "cm^2/s": "cm²/s", "um^2/s": "µm²/s", "mol/m^3": "mol/m³", mol_m3: "mol/m³", C: "°C", degC: "°C", A: "Å", uM: "µM", us: "µs", eps_r: "ε_r" };

export function canonUnit(u) {
  if (!u) return "";
  const t = String(u).trim();
  return ALIAS[t] || t;
}

export function familyOf(unit) {
  const u = canonUnit(unit);
  return FAMILIES.find((f) => Object.prototype.hasOwnProperty.call(f.units, u)) || null;
}

// --- dimensionless display. The reference scales come from the document's own parameters:
//   potential     V_T = k_B T / e          (T, default 298.15 K)
//   length        a_ref                    (a_ref_* parameter)
//   concentration c0                       (c0_* parameter)
//   diffusivity   D                        (D parameter)
//   time          l_p^2 / D  (tau)         (L_* and D; a_ref^2 / D without L)
//   permittivity  eps0 (eps_r)
// Each becomes one extra unit of its family, so convert() handles both directions. The
// parameters that define a scale stay in their own unit (a_ref in a_ref units is always 1).
const K_B = 1.380649e-23, E_CH = 1.602176634e-19, EPS0 = 8.8541878128e-12;

/** k_B T / e in volts (T in K, default 298.15). */
export const thermalVoltage = (T) => (K_B * (Number(T) > 0 ? Number(T) : 298.15)) / E_CH;

const siOf = (r) => { const f = familyOf(r.unit); return f ? toBase(r.value, canonUnit(r.unit), f) : null; };

/** Scales for the dimensionless display from parameter rows {key, value, unit}. */
export function scalesFor(rows) {
  const num = rows.filter((r) => typeof r.value === "number" && Number.isFinite(r.value));
  const find = (re, fam) => num.find((r) => re.test(r.key) && (familyOf(r.unit) || {}).name === fam);
  const out = { defs: new Set() };
  const T = num.find((r) => r.key === "T" && (familyOf(r.unit) || { name: "temperature" }).name === "temperature");
  const Tk = T ? (canonUnit(T.unit) === "°C" ? T.value + 273.15 : T.value) : 298.15;
  if (T) out.defs.add(T.key);
  out.potential = { sym: "V_T", factor: (K_B * Tk) / E_CH, what: `thermal voltage k_B T / e at ${Number.parseFloat(Tk.toPrecision(5))} K` };
  const a = find(/^a_ref/, "length");
  if (a) { out.length = { sym: "a_ref", factor: siOf(a), what: "reference radius a_ref" }; out.defs.add(a.key); }
  const c = find(/^c0/, "concentration");
  if (c) { out.concentration = { sym: "c0", factor: siOf(c), what: "bulk concentration c0" }; out.defs.add(c.key); }
  const D = find(/^D(_|$)/, "diffusivity");
  if (D) { out.diffusivity = { sym: "D", factor: siOf(D), what: "ion diffusivity D" }; out.defs.add(D.key); }
  const L = find(/^L(_|$)/, "length");
  if (D && (L || a)) out.time = { sym: L ? "l_p²/D" : "a_ref²/D", factor: siOf(L || a) ** 2 / siOf(D), what: L ? "diffusion time over the pore length, tau = D t / l_p²" : "diffusion time over a_ref" };
  out.permittivity = { sym: "ε_r", factor: EPS0, what: "relative permittivity ε / ε0" };
  return out;
}

/** The dimensionless unit a parameter is shown in, or null (no scale, or it defines one). */
export function scaleOf(key, unit, scales) {
  if (!scales || scales.defs.has(key)) return null;
  const f = familyOf(unit);
  return f && scales[f.name] ? scales[f.name] : null;
}

function unitsOf(f, scales) {
  const sc = scales && scales[f.name];
  return sc ? Object.assign({}, f.units, { [sc.sym]: sc.factor }) : f.units;
}

/** Units the value can be shown in (empty when the unit is unknown). */
export function unitChoices(unit, scales) {
  const f = familyOf(unit);
  return f ? Object.keys(unitsOf(f, scales)) : [];
}

function toBase(v, u, f, scales) {
  if (f.name === "temperature") return u === "°C" ? v + 273.15 : v;
  return v * unitsOf(f, scales)[u];
}
function fromBase(v, u, f, scales) {
  if (f.name === "temperature") return u === "°C" ? v - 273.15 : v;
  return v / unitsOf(f, scales)[u];
}

/** Convert v from unit `from` to unit `to` (same family; `scales` adds the dimensionless units). */
export function convert(v, from, to, scales) {
  from = canonUnit(from); to = canonUnit(to);
  if (from === to || typeof v !== "number" || !Number.isFinite(v)) return v;
  // a dimensionless unit (V_T, a_ref, ...) is only known through `scales`: take the other side's family
  const f = familyOf(from) || familyOf(to);
  if (!f) return v;
  const units = unitsOf(f, scales);
  if (!(to in units) || !(from in units)) return v;
  const r = fromBase(toBase(v, from, f, scales), to, f, scales);
  // trim float noise (1000 nm -> 1 µm, not 0.9999999999)
  return Number.parseFloat(r.toPrecision(12));
}
