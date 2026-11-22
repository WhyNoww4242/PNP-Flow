// The one module that talks to the backend.
//
// Inside Tauri 2 (window.__TAURI__ present, withGlobalTauri = true) every call goes
// through `invoke` to the Rust commands in app/src-tauri/src/main.rs. In a plain
// browser it uses fetch() against the dev server (app/server.py), which serves the
// same JSON API as hud/server.py. The rest of the UI imports only `bridge`.

const enc = encodeURIComponent;
const tauri = typeof window !== "undefined" && window.__TAURI__ && window.__TAURI__.core;

export const inTauri = !!tauri;

function httpError(status, msg) {
  const err = new Error(msg);
  err.status = status;
  return err;
}

/** Parse JSON written by Python, which may contain NaN / Infinity. */
function parseLoose(text) {
  try {
    return JSON.parse(text);
  } catch {
    return JSON.parse(text.replace(/\bNaN\b|-?\bInfinity\b/g, "null"));
  }
}

async function getJSON(url, opts = {}) {
  const r = await fetch(url, { cache: "no-store", ...opts });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      if (j && j.error) msg = j.error;
    } catch { /* not JSON */ }
    throw httpError(r.status, msg);
  }
  return parseLoose(await r.text());
}

/** Fetch JSON with a progress callback onProgress(fraction|null, bytes). */
async function getJSONProgress(url, onProgress) {
  const r = await fetch(url, { cache: "no-store" });
  if (!r.ok) throw httpError(r.status, r.status === 404 ? "not found" : `${r.status} ${r.statusText}`);
  const total = Number(r.headers.get("X-Uncompressed-Length") ||
    (r.headers.get("Content-Encoding") ? 0 : r.headers.get("Content-Length")) || 0);
  if (!r.body || !r.body.getReader) return parseLoose(await r.text());
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
  for (const c of chunks) { buf.set(c, off); off += c.length; }
  return parseLoose(new TextDecoder().decode(buf));
}

/** Tauri invoke with errors mapped to the same shape as HTTP errors. */
async function call(cmd, args) {
  try {
    return await tauri.invoke(cmd, args);
  } catch (e) {
    const msg = typeof e === "string" ? e : (e && e.message) || String(e);
    throw httpError(/not found|no such/i.test(msg) ? 404 : 500, msg);
  }
}

const MIME = { png: "image/png", jpg: "image/jpeg", jpeg: "image/jpeg", svg: "image/svg+xml" };

const httpBridge = {
  runs: () => getJSON("/api/runs"),
  listing: (name) => getJSON(`/api/run/${enc(name)}`),
  json: (name, file) => getJSON(`/api/run/${enc(name)}/${file}`),
  jsonProgress: (name, file, cb) => getJSONProgress(`/api/run/${enc(name)}/${file}`, cb),
  log: (name, tail = 65536) => getJSON(`/api/run/${enc(name)}/log?tail=${tail}`),
  imageURL: async (name, path) => `/api/run/${enc(name)}/${path.split("/").map(enc).join("/")}`,
  async cases() {
    const r = await fetch("/api/cases", { cache: "no-store" });
    const data = r.ok ? parseLoose(await r.text()) : {};
    return { cases: data || {}, error: r.headers.get("X-Cases-Error") || (r.ok ? null : r.statusText) };
  },
  launch: (body) => getJSON("/api/launch", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  }),
  stop: (name) => getJSON("/api/stop", {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ name }),
  }),
  async colormaps() {
    const r = await fetch("data/colormaps.json", { cache: "no-store" });
    if (!r.ok) throw httpError(r.status, "colormaps.json not found");
    return parseLoose(await r.text());
  },
};

const tauriBridge = {
  runs: () => call("list_runs"),
  listing: (name) => call("run_listing", { name }),
  json: async (name, file) => parseLoose(await call("read_run_file", { name, file })),
  jsonProgress: async (name, file, cb) => {
    if (cb) cb(null, 0);
    return parseLoose(await call("read_run_file", { name, file }));
  },
  log: (name, tail = 65536) => call("read_run_log", { name, tail }),
  imageURL: async (name, path) => {
    const b64 = await call("read_run_file_base64", { name, file: path });
    const ext = path.split(".").pop().toLowerCase();
    return `data:${MIME[ext] || "application/octet-stream"};base64,${b64}`;
  },
  async cases() {
    try {
      return { cases: await call("list_cases"), error: null };
    } catch (e) {
      return { cases: {}, error: e.message };
    }
  },
  launch: (body) => call("launch_run", { request: body }),
  stop: (name) => call("stop_run", { name }),
  async colormaps() {
    // frontendDist is bundled, so the static file is reachable through fetch here too
    const r = await fetch("data/colormaps.json");
    if (!r.ok) throw httpError(r.status, "colormaps.json not found");
    return parseLoose(await r.text());
  },
};

export const bridge = inTauri ? tauriBridge : httpBridge;

/** Optional file: resolve to null on 404 instead of throwing. */
export async function optional(promise) {
  try {
    return await promise;
  } catch (e) {
    if (e.status === 404) return null;
    throw e;
  }
}
