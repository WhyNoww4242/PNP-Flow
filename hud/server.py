#!/usr/bin/env python3
"""pnpflow HUD server — stdlib only.

Serves the static dashboard in hud/static and a small JSON API over the
result folders in <pnpflow root>/results (format: docs/DATA_FORMAT.md).

    python hud/server.py [--port 8765] [--host 127.0.0.1] [--results DIR]

API
---
GET  /api/info                      root paths, python, whether run.py exists
GET  /api/runs                      run folders, newest first, with summary basics + state
GET  /api/run/<name>                file listing of one run (which files / plots exist)
GET  /api/run/<name>/<file>         summary.json | fields.json | profiles.json |
                                    progress.json | stdout.log | plots/<img>.png
GET  /api/run/<name>/log?tail=N     last N bytes of stdout.log (default 64 KiB)
GET  /api/cases                     output of `python run.py --list` ({} if unavailable)
POST /api/launch                    {case, name, params:{k:v}, overwrite?, device?} -> spawn run.py
                                    device: "auto" (default) | "cpu" | "gpu" (sets JAX_PLATFORMS)
GET  /api/devices                   {"gpu": bool, "devices": [...], "error"?} (probed once, cached)
POST /api/stop                      {name} -> terminate a run launched by this server
"""
import argparse
import gzip
import json
import mimetypes
import os
import re
import subprocess
import sys
import threading
import time
from email.utils import formatdate, parsedate_to_datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, unquote, urlsplit

HUD_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(HUD_DIR, "static")
ROOT = os.path.dirname(HUD_DIR)

# Set from the command line in main()
CONFIG = {"results": os.path.join(ROOT, "results"),
          "python": os.environ.get("PNPFLOW_PYTHON", sys.executable)}

NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127}$")
CASE_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$")
KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-+]{0,63}$")
PLOT_RE = re.compile(r"^plots/[A-Za-z0-9_.\-+]{1,128}\.(png|jpg|jpeg|svg)$", re.I)
RUN_FILES = {"summary.json", "fields.json", "fields_time.json", "profiles.json", "progress.json", "stdout.log"}
FINAL_STATES = {"done", "failed"}

# Processes spawned through /api/launch: name -> {"proc", "started", "cmd"}
PROCS = {}
PROCS_LOCK = threading.Lock()

# gzip cache for large JSON files: path -> (mtime, size, bytes)
GZ_CACHE = {}
GZ_LOCK = threading.Lock()

CASES_CACHE = {"key": None, "time": 0.0, "data": None, "error": None}

# JAX platform per launch device; "auto" leaves JAX to choose (GPU when present)
DEVICE_ENV = {"cpu": "cpu", "gpu": "cuda"}
DEVICES_CACHE = {"data": None}
DEVICES_LOCK = threading.Lock()


def get_devices():
    """Which JAX devices the solver's Python sees (probed once, then cached)."""
    with DEVICES_LOCK:
        if DEVICES_CACHE["data"] is not None:
            return DEVICES_CACHE["data"]
        code = ("import json, jax\n"
                "try:\n    g = [str(d) for d in jax.devices('gpu')]\n"
                "except Exception:\n    g = []\n"
                "print(json.dumps({'gpu': bool(g), 'devices': [str(d) for d in jax.devices()]}))")
        try:
            out = subprocess.run([CONFIG["python"], "-c", code], capture_output=True,
                                 text=True, timeout=120)
            line = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "{}"
            data = json.loads(line)
            if out.returncode != 0:
                data["error"] = out.stderr.strip()[-300:]
        except (OSError, ValueError, IndexError, subprocess.TimeoutExpired) as e:
            data = {"gpu": False, "devices": [], "error": "device probe failed: %s" % e}
        data.setdefault("gpu", False)
        DEVICES_CACHE["data"] = data
        return data


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def results_dir():
    return CONFIG["results"]


def run_path(name):
    """Return the absolute folder of run `name`, or None if the name is unsafe."""
    if not name or not NAME_RE.match(name) or ".." in name:
        return None
    base = os.path.realpath(results_dir())
    path = os.path.realpath(os.path.join(base, name))
    if os.path.dirname(path) != base:
        return None
    return path


def read_json(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def proc_status(name):
    """Status of a process this server launched for run `name` (or None)."""
    with PROCS_LOCK:
        entry = PROCS.get(name)
    if not entry:
        return None
    code = entry["proc"].poll()
    return {"pid": entry["proc"].pid, "alive": code is None, "returncode": code,
            "started": entry["started"]}


def run_state(progress, proc, has_summary):
    """Combine progress.json and our own process bookkeeping into one state."""
    state = (progress or {}).get("state")
    if proc:
        if proc["alive"]:
            return state if state and state not in FINAL_STATES else (state or "launched")
        if state not in FINAL_STATES:
            return "done" if proc["returncode"] == 0 and has_summary else "failed"
    if state:
        return state
    return "done" if has_summary else "unknown"


def latest_mtime(path):
    best = os.path.getmtime(path)
    for fn in ("progress.json", "summary.json", "fields.json", "stdout.log"):
        p = os.path.join(path, fn)
        if os.path.exists(p):
            best = max(best, os.path.getmtime(p))
    return best


def run_entry(name, path):
    """Compact description of one run for the sidebar."""
    summary = read_json(os.path.join(path, "summary.json"))
    progress = read_json(os.path.join(path, "progress.json"))
    proc = proc_status(name)
    has_summary = isinstance(summary, dict)
    summary = summary if has_summary else {}
    progress = progress if isinstance(progress, dict) else {}
    solver = summary.get("solver") or {}
    mesh = summary.get("mesh") or {}
    force = ((summary.get("integrals") or {}).get("force") or {})
    checks = summary.get("checks") or []
    worst = None
    order = {"fail": 3, "warn": 2, "pass": 1, "info": 0}
    for c in checks if isinstance(checks, list) else []:
        s = c.get("status") if isinstance(c, dict) else None
        if s in order and (worst is None or order[s] > order[worst]):
            worst = s
    entry = {
        "name": name,
        "case": summary.get("case"),
        "created": summary.get("created"),
        "mtime": latest_mtime(path),
        "state": run_state(progress, proc, has_summary),
        "converged": solver.get("converged"),
        "time_s": solver.get("time_s", progress.get("time_s")),
        "n_nodes": mesh.get("n_nodes"),
        "n_elements": mesh.get("n_elements"),
        "drag": force.get("drag"),
        "checks": worst,
        "files": sorted(f for f in RUN_FILES if os.path.exists(os.path.join(path, f))),
        "progress": {k: progress.get(k) for k in
                     ("state", "stage", "n_stages", "iteration", "res_norm", "message", "time_s")
                     if k in progress},
        "proc": proc,
    }
    return entry


def list_runs():
    base = results_dir()
    if not os.path.isdir(base):
        return []
    runs = []
    for name in os.listdir(base):
        path = run_path(name)
        if path and os.path.isdir(path):
            try:
                runs.append(run_entry(name, path))
            except OSError:
                continue
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


def run_listing(name, path):
    plots_dir = os.path.join(path, "plots")
    plots = []
    if os.path.isdir(plots_dir):
        plots = sorted(f for f in os.listdir(plots_dir) if PLOT_RE.match("plots/" + f))
    files = {}
    for fn in sorted(RUN_FILES):
        p = os.path.join(path, fn)
        if os.path.exists(p):
            files[fn] = {"size": os.path.getsize(p), "mtime": os.path.getmtime(p)}
    return {"name": name, "files": files, "plots": plots, "entry": run_entry(name, path)}


def tail_bytes(path, n):
    try:
        size = os.path.getsize(path)
        with open(path, "rb") as f:
            if size > n:
                f.seek(size - n)
            data = f.read()
        return data.decode("utf-8", "replace"), size
    except OSError:
        return "", 0


def get_cases():
    """Run `python run.py --list` (cached for 20 s or until run.py changes)."""
    run_py = os.path.join(ROOT, "run.py")
    if not os.path.isfile(run_py):
        return {}, "run.py not found in %s" % ROOT
    key = os.path.getmtime(run_py)
    now = time.time()
    c = CASES_CACHE
    if c["key"] == key and now - c["time"] < 20 and c["data"] is not None:
        return c["data"], c["error"]
    data, err = {}, None
    try:
        out = subprocess.run([CONFIG["python"], run_py, "--list"], cwd=ROOT,
                             capture_output=True, text=True, timeout=120)
        text = out.stdout.strip()
        try:
            data = json.loads(text)
        except ValueError:
            # tolerate log noise before/after the JSON object
            i, j = text.find("{"), text.rfind("}")
            data = json.loads(text[i:j + 1]) if i >= 0 and j > i else {}
        if not isinstance(data, dict):
            data, err = {}, "run.py --list did not print a JSON object"
        if out.returncode != 0 and not data:
            err = "run.py --list exited with %d: %s" % (out.returncode, out.stderr.strip()[-500:])
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        data, err = {}, "run.py --list failed: %s" % e
    c.update(key=key, time=now, data=data, error=err)
    return data, err


def launch(body):
    case = body.get("case")
    name = str(body.get("name", ""))
    params = body.get("params") or {}
    if not isinstance(case, str) or not CASE_RE.match(case):
        return 400, {"error": "no case selected (the case list did not load: check /api/cases)"}
    path = run_path(name)
    if not path:
        return 400, {"error": "invalid run name (letters, digits, _ . - only)"}
    if not isinstance(params, dict):
        return 400, {"error": "params must be an object"}
    device = str(body.get("device") or "auto")
    if device not in ("auto", "cpu", "gpu"):
        return 400, {"error": "device must be auto, cpu or gpu"}
    run_py = os.path.join(ROOT, "run.py")
    if not os.path.isfile(run_py):
        return 409, {"error": "run.py not found in %s" % ROOT}
    st = proc_status(name)
    if st and st["alive"]:
        return 409, {"error": "run '%s' is already running (pid %d)" % (name, st["pid"])}
    if os.path.exists(path) and os.listdir(path) and not body.get("overwrite"):
        return 409, {"error": "results/%s already exists" % name, "exists": True}
    cmd = [CONFIG["python"], "-u", run_py, case, "--name", name]
    for k, v in params.items():
        k = str(k)
        v = v if isinstance(v, str) else json.dumps(v)
        if not KEY_RE.match(k) or any(ch in v for ch in "\r\n\0"):
            return 400, {"error": "invalid parameter %r" % k}
        cmd += ["--set", "%s=%s" % (k, v)]
    os.makedirs(path, exist_ok=True)
    # stale progress from a previous run with the same name would confuse the UI
    old = os.path.join(path, "progress.json")
    if body.get("overwrite") and os.path.exists(old):
        os.remove(old)
    log = open(os.path.join(path, "stdout.log"), "w", encoding="utf-8")
    log.write("# launched by HUD %s\n# %s\n# device: %s\n" % (
        time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd), device))
    log.flush()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    if device in DEVICE_ENV:
        env["JAX_PLATFORMS"] = DEVICE_ENV[device]
    else:
        env.pop("JAX_PLATFORMS", None)
    try:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, env=env)
    except OSError as e:
        log.close()
        return 500, {"error": "spawn failed: %s" % e}
    with PROCS_LOCK:
        PROCS[name] = {"proc": proc, "started": time.time(), "cmd": cmd, "log": log,
                       "device": device}
    threading.Thread(target=_reap, args=(name, proc, log), daemon=True).start()
    return 200, {"ok": True, "name": name, "pid": proc.pid, "cmd": cmd, "device": device}


def _reap(name, proc, log):
    proc.wait()
    try:
        log.write("\n# process exited with code %d\n" % proc.returncode)
        log.close()
    except (OSError, ValueError):
        pass


def stop(body):
    name = str(body.get("name", ""))
    with PROCS_LOCK:
        entry = PROCS.get(name)
    if not entry or entry["proc"].poll() is not None:
        return 404, {"error": "no running process for '%s' launched by this server" % name}
    entry["proc"].terminate()
    return 200, {"ok": True}


def gzipped(path):
    """gzip bytes of a file, cached by (mtime, size)."""
    st = os.stat(path)
    with GZ_LOCK:
        hit = GZ_CACHE.get(path)
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            return hit[2]
    with open(path, "rb") as f:
        data = gzip.compress(f.read(), compresslevel=5)
    with GZ_LOCK:
        if len(GZ_CACHE) > 16:
            GZ_CACHE.clear()
        GZ_CACHE[path] = (st.st_mtime, st.st_size, data)
    return data


# ---------------------------------------------------------------------------
# request handler
# ---------------------------------------------------------------------------
class Handler(BaseHTTPRequestHandler):
    server_version = "pnpflow-hud/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quieter log: skip polling noise
        msg = fmt % args
        if "/api/runs" in msg or "progress.json" in msg or "/log" in msg:
            return
        sys.stderr.write("[hud] %s\n" % msg)

    # -- responses ---------------------------------------------------------
    def send_json(self, obj, status=200):
        data = json.dumps(_clean(obj), default=str).encode("utf-8")
        self.send_bytes(data, "application/json; charset=utf-8", status, cache=False)

    def send_error_json(self, status, message):
        self.send_json({"error": message}, status)

    def send_bytes(self, data, ctype, status=200, cache=False, extra=None):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache" if not cache else "max-age=3600")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def send_file(self, path, ctype=None, allow_gzip=True):
        if not os.path.isfile(path):
            return self.send_error_json(404, "not found")
        ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"
        if ctype.startswith("text/") or ctype in ("application/json", "application/javascript"):
            ctype += "; charset=utf-8"
        st = os.stat(path)
        last_mod = formatdate(st.st_mtime, usegmt=True)
        ims = self.headers.get("If-Modified-Since")
        if ims:
            try:
                if int(st.st_mtime) <= parsedate_to_datetime(ims).timestamp():
                    self.send_response(304)
                    self.send_header("Last-Modified", last_mod)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
            except (TypeError, ValueError):
                pass
        extra = {"Last-Modified": last_mod}
        use_gz = (allow_gzip and st.st_size > 64 * 1024
                  and "gzip" in self.headers.get("Accept-Encoding", "")
                  and not ctype.startswith("image/"))
        if use_gz:
            data = gzipped(path)
            extra["Content-Encoding"] = "gzip"
            extra["X-Uncompressed-Length"] = str(st.st_size)
        else:
            with open(path, "rb") as f:
                data = f.read()
        self.send_bytes(data, ctype, 200, cache=False, extra=extra)

    def read_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 1 << 20:
            raise ValueError("body too large")
        raw = self.rfile.read(n) if n else b"{}"
        body = json.loads(raw.decode("utf-8") or "{}")
        if not isinstance(body, dict):
            raise ValueError("JSON object expected")
        return body

    # -- routing -----------------------------------------------------------
    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        url = urlsplit(self.path)
        path = unquote(url.path)
        query = parse_qs(url.query)
        try:
            if path.startswith("/api/"):
                return self.api_get(path[5:], query)
            return self.static(path)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as e:  # never kill the server on a bad request
            self.send_error_json(500, "%s: %s" % (type(e).__name__, e))

    def do_POST(self):
        path = urlsplit(self.path).path
        try:
            body = self.read_body()
        except (ValueError, UnicodeDecodeError) as e:
            return self.send_error_json(400, "bad request body: %s" % e)
        if path == "/api/launch":
            status, obj = launch(body)
        elif path == "/api/stop":
            status, obj = stop(body)
        else:
            status, obj = 404, {"error": "unknown endpoint"}
        self.send_json(obj, status)

    def api_get(self, rest, query):
        if rest == "info":
            return self.send_json({
                "root": ROOT, "results": results_dir(), "python": CONFIG["python"],
                "run_py": os.path.isfile(os.path.join(ROOT, "run.py")),
                "server_time": time.time()})
        if rest == "runs":
            return self.send_json({"runs": list_runs(), "results": results_dir()})
        if rest == "devices":
            return self.send_json(get_devices())
        if rest == "cases":
            data, err = get_cases()
            extra = {"X-Cases-Error": err.replace("\n", " ")[:300]} if err else None
            body = json.dumps(data, default=str).encode("utf-8")
            return self.send_bytes(body, "application/json; charset=utf-8", 200, extra=extra)
        if rest.startswith("run/"):
            parts = rest[4:].split("/", 1)
            name = parts[0]
            run_dir = run_path(name)
            if not run_dir or not os.path.isdir(run_dir):
                return self.send_error_json(404, "no such run")
            if len(parts) == 1 or parts[1] == "":
                return self.send_json(run_listing(name, run_dir))
            fname = parts[1]
            if fname == "log":
                n = int((query.get("tail") or ["65536"])[0])
                text, size = tail_bytes(os.path.join(run_dir, "stdout.log"), max(1024, min(n, 4 << 20)))
                return self.send_json({"text": text, "size": size,
                                       "proc": proc_status(name)})
            if fname in RUN_FILES or PLOT_RE.match(fname):
                target = os.path.realpath(os.path.join(run_dir, fname))
                if not target.startswith(os.path.realpath(run_dir) + os.sep):
                    return self.send_error_json(403, "forbidden")
                ctype = "text/plain" if fname.endswith(".log") else None
                return self.send_file(target, ctype)
            return self.send_error_json(404, "unknown run file")
        return self.send_error_json(404, "unknown endpoint")

    def static(self, path):
        if path in ("", "/"):
            path = "/index.html"
        target = os.path.realpath(os.path.join(STATIC_DIR, path.lstrip("/")))
        if not target.startswith(os.path.realpath(STATIC_DIR) + os.sep):
            return self.send_error_json(403, "forbidden")
        return self.send_file(target)


def _clean(obj):
    """Replace NaN/inf (invalid JSON) by None, recursively."""
    if isinstance(obj, float):
        return obj if obj == obj and obj not in (float("inf"), float("-inf")) else None
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    return obj


def main():
    ap = argparse.ArgumentParser(description="pnpflow HUD server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("HUD_PORT", 8765)))
    ap.add_argument("--host", default=os.environ.get("HUD_HOST", "127.0.0.1"))
    ap.add_argument("--results", default=CONFIG["results"], help="results directory")
    ap.add_argument("--python", default=CONFIG["python"], help="interpreter used for run.py")
    args = ap.parse_args()
    CONFIG["results"] = os.path.abspath(args.results)
    CONFIG["python"] = args.python
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    srv.daemon_threads = True
    host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1", "") else args.host
    print("pnpflow HUD: http://%s:%d/   (results: %s)" % (host, args.port, CONFIG["results"]),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
