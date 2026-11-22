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
"""
import argparse
import importlib.util
import json
import mimetypes
import os
import re
import sys
from http.server import ThreadingHTTPServer

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

# New run files (written by the post-processing step). Every pattern is anchored and
# allows a single directory level with a restricted character set, so no "..".
FIG_RE = re.compile(r"^figures/[A-Za-z0-9_.\-+]{1,128}\.png$", re.I)
SVG_RE = re.compile(r"^svg/[A-Za-z0-9_.\-+]{1,128}\.svg$", re.I)
MESH_RE = re.compile(r"^mesh/(view_a\.png|view_b\.png|diagnose\.json)$")
MESH_FILES = ("mesh/view_a.png", "mesh/view_b.png", "mesh/diagnose.json")


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

    def api_get(self, rest, query):
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
    args = ap.parse_args()
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
