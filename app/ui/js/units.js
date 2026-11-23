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

/** Units the value can be shown in (empty when the unit is unknown). */
export function unitChoices(unit) {
  const f = familyOf(unit);
  return f ? Object.keys(f.units) : [];
}

function toBase(v, u, f) {
  if (f.name === "temperature") return u === "°C" ? v + 273.15 : v;
  return v * f.units[u];
}
function fromBase(v, u, f) {
  if (f.name === "temperature") return u === "°C" ? v - 273.15 : v;
  return v / f.units[u];
}

/** Convert v from unit `from` to unit `to` (same family). */
export function convert(v, from, to) {
  from = canonUnit(from); to = canonUnit(to);
  if (from === to || typeof v !== "number" || !Number.isFinite(v)) return v;
  const f = familyOf(from);
  if (!f || !(to in f.units)) return v;
  const r = fromBase(toBase(v, from, f), to, f);
  // trim float noise (1000 nm -> 1 µm, not 0.9999999999)
  return Number.parseFloat(r.toPrecision(12));
}
