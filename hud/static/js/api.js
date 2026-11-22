// Thin wrappers around the HUD server API (see hud/server.py).

const enc = encodeURIComponent;

async function getJSON(url, opts = {}) {
  const r = await fetch(url, { cache: "no-store", ...opts });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j && j.error) msg = j.error;
    } catch { /* not JSON */ }
    const err = new Error(msg);
    err.status = r.status;
    throw err;
  }
  return r.json();
}

/** Fetch JSON with a progress callback onProgress(fraction|null, bytes). */
async function getJSONProgress(url, onProgress) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) {
    const err = new Error(r.status === 404 ? "not found" : `${r.status} ${r.statusText}`);
    err.status = r.status;
    throw err;
  }
  const total = Number(r.headers.get("X-Uncompressed-Length") ||
    (r.headers.get("Content-Encoding") ? 0 : r.headers.get("Content-Length")) || 0);
  if (!r.body || !r.body.getReader) return r.json();
  const reader = r.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (onProgress) onProgress(total ? Math.min(1, got / total) : null, got);
  }
  const buf = new Uint8Array(got);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  const text = new TextDecoder().decode(buf);
  // Python's json may write NaN/Infinity, which JSON.parse rejects.
  try {
    return JSON.parse(text);
  } catch {
    return JSON.parse(text.replace(/\bNaN\b|-?\bInfinity\b/g, "null"));
  }
}

export const api = {
  info: () => getJSON("/api/info"),
  runs: () => getJSON("/api/runs"),
  listing: (name) => getJSON(`/api/run/${enc(name)}`),
  file: (name, file) => getJSON(`/api/run/${enc(name)}/${file}`),
  fileProgress: (name, file, cb) => getJSONProgress(`/api/run/${enc(name)}/${file}`, cb),
  log: (name, tail = 65536) => getJSON(`/api/run/${enc(name)}/log?tail=${tail}`),
  plotURL: (name, png) => `/api/run/${enc(name)}/plots/${enc(png)}`,
  async cases() {
    const r = await fetch("/api/cases", { cache: "no-store" });
    const data = r.ok ? await r.json() : {};
    return { cases: data || {}, error: r.headers.get("X-Cases-Error") || (r.ok ? null : r.statusText) };
  },
  launch: (body) => getJSON("/api/launch", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }),
  stop: (name) => getJSON("/api/stop", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
  }),
};

/** Optional file: resolve to null on 404 instead of throwing. */
export async function optional(promise) {
  try {
    return await promise;
  } catch (e) {
    if (e.status === 404) return null;
    throw e;
  }
}
