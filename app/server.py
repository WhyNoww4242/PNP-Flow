#!/usr/bin/env python3
"""pnpflow desktop app — development server (Python stdlib only).

Serves the app front end in app/ui and the same JSON API as hud/server.py, by
importing that module and subclassing its request handler (hud/ is not
modified, and the old HUD keeps working on its own port).

    python app/server.py [--port 8766] [--host 127.0.0.1] [--results DIR] [--python EXE]

Extra over the HUD API
----------------------
GET /api/run/<name>                 listing also has "figures", "svgs", "mesh_files"
GET /api/runs                       each entry also has "thumb" (relative file path or null)
GET /api/run/<name>/figures/<f>.png house-style figures
GET /api/run/<name>/svg/<f>.svg     vector versions (served with a strict CSP)
GET /api/run/<name>/mesh/view_a.png | mesh/view_b.png | mesh/diagnose.json
GET /api/run/<name>/convergence.txt | case.json

Cases (docs/case_answers.md): cases/<case>/case.json + mesh/, runs in cases/<case>/results/
GET  /api/caselist                   every case with its mesh state and runs
GET  /api/case/<case>                case.json, mesh status, mesh files, runs
GET  /api/case/<case>/file/mesh/<f>  view_a.png view_b.png validation.json mesh.json mesh.log
POST /api/case/<case>/save           {"case": {...}, "create": bool}  writes case.json
POST /api/case/<case>/mesh           {"device"}  builds the mesh only (run.py --mesh-only)
POST /api/case/<case>/delete-mesh    removes the generated mesh files of the case (case.json and runs stay)
POST /api/case/<case>/run            {"name", "device", "overwrite"}  starts a run of the case
Run names stay unique across results/ and every case, so /api/run/<name> keeps working.
"""
import argparse
import importlib.util
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer
from urllib.parse import urlsplit

APP_DIR = os.path.dirname(os.path.abspath(__file__))
UI_DIR = os.path.join(APP_DIR, "ui")
ROOT = os.path.dirname(APP_DIR)


def _load_hud():
    """Import hud/server.py under a private module name (avoids clashing with this file)."""
    path = os.path.join(ROOT, "hud", "server.py")
    spec = importlib.util.spec_from_file_location("pnpflow_hud_server", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hud = _load_hud()


def _load_project():
    """pnpflow/project.py by path: stdlib only, so the server never imports jax."""
    spec = importlib.util.spec_from_file_location("pnpflow_project", os.path.join(ROOT, "pnpflow", "project.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


project = _load_project()

# New run files (written by the post-processing step). Every pattern is anchored and
# allows a single directory level with a restricted character set, so no "..".
FIG_RE = re.compile(r"^figures/[A-Za-z0-9_.\-+]{1,128}\.png$", re.I)
SVG_RE = re.compile(r"^svg/[A-Za-z0-9_.\-+]{1,128}\.svg$", re.I)
MESH_RE = re.compile(r"^mesh/(view_a\.png|view_b\.png|diagnose\.json)$")
MESH_FILES = ("mesh/view_a.png", "mesh/view_b.png", "mesh/diagnose.json")


hud.RUN_FILES.update({"convergence.txt", "case.json"})

# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------
CONF = {"cases": os.path.join(ROOT, "cases")}
CASE_NAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.\-]{0,63}$")
CASE_FILE_RE = re.compile(r"^mesh/(view_a\.png|view_b\.png|validation\.json|mesh\.json|status\.json|mesh\.log)$")
MESH_PROCS = {}          # case name -> {"proc", "started", "log"}
MESH_LOCK = threading.Lock()


def cases_dir():
    return CONF["cases"]


def case_path(name):
    """Absolute folder of case `name` (may not exist), or None if the name is unsafe."""
    if not name or not CASE_NAME_RE.match(name) or ".." in name:
        return None
    base = os.path.realpath(cases_dir())
    path = os.path.realpath(os.path.join(base, name))
    return path if os.path.dirname(path) == base else None


def case_names():
    base = cases_dir()
    if not os.path.isdir(base):
        return []
    return sorted(n for n in os.listdir(base)
                  if case_path(n) and os.path.isfile(os.path.join(base, n, "case.json")))


def case_run_dirs():
    """(run name, folder, case name) of every run inside a case."""
    out = []
    for c in case_names():
        rdir = os.path.join(cases_dir(), c, "results")
        if os.path.isdir(rdir):
            for r in os.listdir(rdir):
                if hud.NAME_RE.match(r) and os.path.isdir(os.path.join(rdir, r)):
                    out.append((r, os.path.join(rdir, r), c))
    return out


_legacy_run_path = hud.run_path
_legacy_list_runs = hud.list_runs


def run_path(name):
    """Legacy results/<name>, else the case run of that name; for a new name the
    legacy path (it does not exist yet)."""
    p = _legacy_run_path(name)
    if p is None or os.path.isdir(p):
        return p
    for r, path, _ in case_run_dirs():
        if r == name:
            return path
    return p


def list_runs():
    runs = _legacy_list_runs()
    for r, path, c in case_run_dirs():
        try:
            e = hud.run_entry(r, path)
        except OSError:
            continue
        e["project"] = c
        runs.append(e)
    runs.sort(key=lambda r: r["mtime"], reverse=True)
    return runs


hud.run_path = run_path
hud.list_runs = list_runs


def run_name_taken(name):
    p = run_path(name)
    return bool(p and os.path.isdir(p) and os.listdir(p))


def _log_failure(path):
    """If the last mesh build died before writing status.json, say why (last error line of
    mesh.log). None when there is no log or its build did not fail."""
    try:
        with open(os.path.join(path, "mesh", "mesh.log"), encoding="utf-8", errors="replace") as f:
            lines = [ln.rstrip() for ln in f.read().splitlines() if ln.strip()]
    except OSError:
        return None
    if not lines or not re.match(r"# process exited with code (?!0\b)\d+", lines[-1]):
        return None
    body = [ln for ln in lines[:-1] if not ln.startswith("#")]
    return (body[-1] if body else lines[-1]).strip()


def mesh_status(name, path):
    st = hud.read_json(os.path.join(path, "mesh", "status.json")) or {"state": "none"}
    if st.get("state") == "none":
        why = _log_failure(path)
        if why:
            st = {"state": "failed", "error": why + " (see mesh.log)"}
    with MESH_LOCK:
        entry = MESH_PROCS.get(name)
    alive = bool(entry and entry["proc"].poll() is None)
    if st.get("state") == "building" and not alive:
        # the build process died without writing its result (e.g. a gmsh crash)
        st = dict(st, state="failed", error=st.get("error") or "the mesh build stopped without a result; see mesh.log")
    if alive:
        st = dict(st, state="building")
    return st


def case_summary(name):
    path = case_path(name)
    case = hud.read_json(os.path.join(path, "case.json")) or {}
    st = mesh_status(name, path)
    runs = [r for r, _, c in case_run_dirs() if c == name]
    return {"name": name, "template": case.get("template"), "description": case.get("description", ""),
            "created": case.get("created"), "modified": case.get("modified"),
            "mesh": {k: st.get(k) for k in ("state", "key", "finished", "error")}
                    | {"n_elements": (st.get("stats") or {}).get("n_elements"),
                       "up_to_date": st.get("state") == "done" and st.get("key") == _safe_key(case)},
            "runs": runs, "mtime": hud.latest_mtime(path)}


def _safe_key(case):
    try:
        return project.mesh_key(case)
    except Exception:
        return None


def case_detail(name):
    path = case_path(name)
    out = case_summary(name)
    out["case"] = hud.read_json(os.path.join(path, "case.json"))
    st = mesh_status(name, path)
    try:
        st["current_key"] = project.mesh_key(out["case"] or {})
    except Exception:
        st["current_key"] = None
    st["up_to_date"] = st.get("state") == "done" and st.get("key") == st["current_key"]
    out["mesh_status"] = st
    out["mesh"]["up_to_date"] = st["up_to_date"]
    out["mesh_files"] = sorted("mesh/" + f for f in os.listdir(os.path.join(path, "mesh"))
                               if CASE_FILE_RE.match("mesh/" + f)) if os.path.isdir(os.path.join(path, "mesh")) else []
    out["run_entries"] = [dict(hud.run_entry(r, p), project=name) for r, p, c in case_run_dirs() if c == name]
    return out


def save_case(name, body):
    path = case_path(name)
    if not path:
        return 400, {"error": "invalid case name (letters, digits, _ . - only)"}
    case = body.get("case")
    if not isinstance(case, dict) or not isinstance(case.get("template"), str) or not hud.CASE_RE.match(case["template"]):
        return 400, {"error": "case must be an object with a template name"}
    target = os.path.join(path, "case.json")
    if body.get("create") and os.path.exists(target):
        return 409, {"error": "a case called '%s' already exists" % name, "exists": True}
    os.makedirs(path, exist_ok=True)
    case = dict(case, name=name, modified=time.strftime("%Y-%m-%dT%H:%M:%S"))
    case.setdefault("created", case["modified"])
    tmp = target + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=1)
    os.replace(tmp, target)
    return 200, {"ok": True, "case": case}


def _spawn(cmd, log_path, device, header):
    log = open(log_path, "w", encoding="utf-8")
    log.write("# %s %s\n# %s\n# device: %s\n" % (header, time.strftime("%Y-%m-%d %H:%M:%S"), " ".join(cmd), device))
    log.flush()
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    # Allocate GPU memory on demand: the default BFC pool was capped at 3.7 GiB on WSL2.
    env.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    env.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
    if device in hud.DEVICE_ENV:
        env["JAX_PLATFORMS"] = hud.DEVICE_ENV[device]
    else:
        env.pop("JAX_PLATFORMS", None)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, env=env)
    return proc, log


def build_mesh(name, body):
    path = case_path(name)
    if not path or not os.path.isfile(os.path.join(path, "case.json")):
        return 404, {"error": "no such case"}
    # meshing is gmsh (CPU) plus building the model once; keep JAX off the GPU so a
    # running solve keeps it, and start-up is faster
    device = str(body.get("device") or "cpu")
    if device not in ("auto", "cpu", "gpu"):
        return 400, {"error": "device must be auto, cpu or gpu"}
    with MESH_LOCK:
        e = MESH_PROCS.get(name)
        if e and e["proc"].poll() is None:
            return 409, {"error": "the mesh of '%s' is already being built" % name}
    os.makedirs(os.path.join(path, "mesh"), exist_ok=True)
    cmd = [hud.CONFIG["python"], "-u", os.path.join(ROOT, "run.py"), "--case", path, "--mesh-only"]
    try:
        proc, log = _spawn(cmd, os.path.join(path, "mesh", "mesh.log"), device, "mesh build")
    except OSError as e:
        return 500, {"error": "spawn failed: %s" % e}
    with MESH_LOCK:
        MESH_PROCS[name] = {"proc": proc, "started": time.time(), "log": log}
    threading.Thread(target=hud._reap, args=("mesh:" + name, proc, log), daemon=True).start()
    return 200, {"ok": True, "pid": proc.pid}


MESH_FILES = ("case.msh", "mesh.json", "status.json", "validation.json", "view_a.png", "view_b.png", "mesh.log")


def delete_mesh(name, body):
    """Remove the generated mesh of a case. Only the known generated files go; case.json and
    the runs (each keeps its own frozen mesh copy) are untouched."""
    path = case_path(name)
    if not path or not os.path.isfile(os.path.join(path, "case.json")):
        return 404, {"error": "no such case"}
    with MESH_LOCK:
        e = MESH_PROCS.get(name)
        if e and e["proc"].poll() is None:
            return 409, {"error": "the mesh of '%s' is being built; wait for it to finish" % name}
    mdir = os.path.join(path, "mesh")
    removed = []
    for f in MESH_FILES:
        fp = os.path.join(mdir, f)
        if os.path.isfile(fp):
            os.remove(fp)
            removed.append(f)
    try:
        os.rmdir(mdir)
    except OSError:
        pass                                        # other files in there: leave them
    return 200, {"ok": True, "removed": removed}


def run_case_api(name, body):
    path = case_path(name)
    if not path or not os.path.isfile(os.path.join(path, "case.json")):
        return 404, {"error": "no such case"}
    run = str(body.get("name", ""))
    if not hud.NAME_RE.match(run):
        return 400, {"error": "invalid run name (letters, digits, _ . - only)"}
    device = str(body.get("device") or "auto")
    if device not in ("auto", "cpu", "gpu"):
        return 400, {"error": "device must be auto, cpu or gpu"}
    st = hud.proc_status(run)
    if st and st["alive"]:
        return 409, {"error": "run '%s' is already running" % run}
    folder = os.path.join(path, "results", run)
    if run_name_taken(run):
        if not body.get("overwrite") or os.path.realpath(run_path(run)) != os.path.realpath(folder):
            return 409, {"error": "a run called '%s' already exists" % run, "exists": True}
        shutil.rmtree(folder)               # replace this case's own run of that name
    with MESH_LOCK:
        e = MESH_PROCS.get(name)
        if e and e["proc"].poll() is None:
            return 409, {"error": "the mesh is still being built; wait for it to finish"}
    os.makedirs(folder, exist_ok=True)
    cmd = [hud.CONFIG["python"], "-u", os.path.join(ROOT, "run.py"), "--case", path, "--name", run]
    try:
        proc, log = _spawn(cmd, os.path.join(folder, "stdout.log"), device, "launched by app")
    except OSError as e:
        return 500, {"error": "spawn failed: %s" % e}
    with hud.PROCS_LOCK:
        hud.PROCS[run] = {"proc": proc, "started": time.time(), "cmd": cmd, "log": log, "device": device}
    threading.Thread(target=hud._reap, args=(run, proc, log), daemon=True).start()
    return 200, {"ok": True, "name": run, "pid": proc.pid, "device": device}


def extra_file_ok(fname):
    return bool(FIG_RE.match(fname) or SVG_RE.match(fname) or MESH_RE.match(fname))


def list_dir(path, sub, regex):
    d = os.path.join(path, sub)
    if not os.path.isdir(d):
        return []
    return sorted(f for f in os.listdir(d) if regex.match(sub + "/" + f))


def thumb_for(path):
    """Best preview image of a run: a figure, else a field plot, else any plot."""
    figs = list_dir(path, "figures", FIG_RE)
    if figs:
        for want in ("field_phi.png", "field_cplus.png"):
            if want in figs:
                return "figures/" + want
        pref = [f for f in figs if f.startswith("field")]
        return "figures/" + (pref or figs)[0]
    plots = list_dir(path, "plots", hud.PLOT_RE)
    plots = [p for p in plots if p.lower().endswith(".png")]
    if plots:
        for want in ("field_phi.png", "field_cplus.png"):
            if want in plots:
                return "plots/" + want
        pref = [p for p in plots if p.startswith("field_")]
        return "plots/" + (pref or plots)[0]
    return None


CASE_IN_CMD = re.compile(r"run\.py\s+([A-Za-z0-9_][A-Za-z0-9_.\-]{0,63})\s")
SET_IN_CMD = re.compile(r"--set\s+([A-Za-z_][A-Za-z0-9_.\-+]{0,63})=(\S+)")


def launch_info(name, path):
    """Case and --set values of a run that has no summary.json yet: from the launch
    command this server holds, else from the command line the launcher writes at the
    top of stdout.log. Returns (case or None, {key: value})."""
    with hud.PROCS_LOCK:
        entry = hud.PROCS.get(name)
    if entry and entry.get("cmd"):
        text = " ".join(entry["cmd"]) + " "
    else:
        try:
            with open(os.path.join(path, "stdout.log"), "r", encoding="utf-8", errors="replace") as f:
                text = f.read(2048).split("\n# process exited")[0]
        except OSError:
            return None, {}
    m = CASE_IN_CMD.search(text)
    params = {}
    for k, v in SET_IN_CMD.findall(text):
        try:
            params[k] = json.loads(v)
        except ValueError:
            params[k] = v
    return (m.group(1) if m else None), params


class AppHandler(hud.Handler):
    server_version = "pnpflow-app/0.1"

    def log_message(self, fmt, *args):
        msg = fmt % args
        if "/api/runs" in msg or "progress.json" in msg or "/log" in msg:
            return
        sys.stderr.write("[app] %s\n" % msg)

    def do_POST(self):
        path = urlsplit(self.path).path
        m = re.match(r"^/api/case/([^/]+)/(save|mesh|run|delete-mesh)$", path)
        if not m:
            return super().do_POST()
        try:
            body = self.read_body()
        except (ValueError, UnicodeDecodeError) as e:
            return self.send_error_json(400, "bad request body: %s" % e)
        name, action = m.group(1), m.group(2)
        if not case_path(name):
            return self.send_error_json(400, "invalid case name")
        try:
            status, obj = {"save": save_case, "mesh": build_mesh, "run": run_case_api, "delete-mesh": delete_mesh}[action](name, body)
        except Exception as e:                       # never kill the server on a bad request
            status, obj = 500, {"error": "%s: %s" % (type(e).__name__, e)}
        self.send_json(obj, status)

    def api_get(self, rest, query):
        if rest == "caselist":
            return self.send_json({"cases": [case_summary(n) for n in case_names()],
                                   "dir": cases_dir()})
        if rest.startswith("case/"):
            parts = rest[5:].split("/", 2)
            name = parts[0]
            path = case_path(name)
            if not path or not os.path.isfile(os.path.join(path, "case.json")):
                return self.send_error_json(404, "no such case")
            if len(parts) == 1 or parts[1] == "":
                return self.send_json(case_detail(name))
            if parts[1] == "file" and len(parts) == 3 and CASE_FILE_RE.match(parts[2]):
                target = os.path.realpath(os.path.join(path, parts[2]))
                if not target.startswith(path + os.sep):
                    return self.send_error_json(403, "forbidden")
                return self.send_file(target, "text/plain" if target.endswith(".log") else None)
            return self.send_error_json(404, "unknown case file")
        if rest == "runs":
            runs = hud.list_runs()
            for r in runs:
                p = hud.run_path(r["name"])
                r["thumb"] = thumb_for(p) if p else None
                if not r.get("case") and p:
                    r["case"] = launch_info(r["name"], p)[0]
            return self.send_json({"runs": runs, "results": hud.results_dir()})
        if rest.startswith("run/"):
            parts = rest[4:].split("/", 1)
            name = parts[0]
            run_dir = hud.run_path(name)
            if not run_dir or not os.path.isdir(run_dir):
                return self.send_error_json(404, "no such run")
            if len(parts) == 1 or parts[1] == "":
                listing = hud.run_listing(name, run_dir)
                listing["figures"] = list_dir(run_dir, "figures", FIG_RE)
                listing["svgs"] = list_dir(run_dir, "svg", SVG_RE)
                listing["mesh_files"] = [f for f in MESH_FILES
                                         if os.path.isfile(os.path.join(run_dir, f))]
                listing["thumb"] = thumb_for(run_dir)
                listing["project"] = (os.path.basename(os.path.dirname(os.path.dirname(run_dir)))
                                      if os.path.basename(os.path.dirname(run_dir)) == "results"
                                      and os.path.dirname(os.path.dirname(os.path.dirname(run_dir))) == os.path.realpath(cases_dir())
                                      else None)
                case, params = launch_info(name, run_dir)
                if not listing["entry"].get("case"):
                    listing["entry"]["case"] = case
                listing["launch_params"] = params
                return self.send_json(listing)
            fname = parts[1]
            if extra_file_ok(fname):
                target = os.path.realpath(os.path.join(run_dir, fname))
                if not target.startswith(os.path.realpath(run_dir) + os.sep):
                    return self.send_error_json(403, "forbidden")
                if fname.lower().endswith(".svg"):
                    return self.send_svg(target)
                return self.send_file(target)
        return super().api_get(rest, query)

    def send_svg(self, path):
        """SVG can carry script: serve it with a CSP that blocks scripts and remote loads."""
        if not os.path.isfile(path):
            return self.send_error_json(404, "not found")
        with open(path, "rb") as f:
            data = f.read()
        self.send_bytes(data, "image/svg+xml", 200, extra={
            "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'; img-src data:",
            "X-Content-Type-Options": "nosniff"})

    def static(self, path):
        if path in ("", "/"):
            path = "/index.html"
        base = os.path.realpath(UI_DIR)
        target = os.path.realpath(os.path.join(base, path.lstrip("/")))
        if not target.startswith(base + os.sep):
            return self.send_error_json(403, "forbidden")
        return self.send_file(target)


def main():
    ap = argparse.ArgumentParser(description="pnpflow app dev server")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PNPFLOW_APP_PORT", 8766)))
    ap.add_argument("--host", default=os.environ.get("PNPFLOW_APP_HOST", "127.0.0.1"))
    ap.add_argument("--results", default=hud.CONFIG["results"], help="results directory")
    ap.add_argument("--python", default=hud.CONFIG["python"], help="interpreter used for run.py")
    ap.add_argument("--cases", default=CONF["cases"], help="cases directory")
    args = ap.parse_args()
    CONF["cases"] = os.path.abspath(args.cases)
    hud.CONFIG["results"] = os.path.abspath(args.results)
    hud.CONFIG["python"] = args.python
    mimetypes.add_type("application/javascript", ".js")
    mimetypes.add_type("text/css", ".css")
    mimetypes.add_type("image/svg+xml", ".svg")
    srv = ThreadingHTTPServer((args.host, args.port), AppHandler)
    srv.daemon_threads = True
    host = "localhost" if args.host in ("0.0.0.0", "127.0.0.1", "") else args.host
    print("pnpflow app: http://%s:%d/   (results: %s)" % (host, args.port, hud.CONFIG["results"]),
          flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down", flush=True)
    finally:
        srv.server_close()


if __name__ == "__main__":
    main()
