#!/usr/bin/env python
"""Queue the four DNS pore-charging cases to tau_end, remeshing on failure
(docs/run_dns_tau053.md). Standard library only; runs without an agent attached.

    nohup python -u tools/run_dns_batch.py > results/dns_tau053_nohup.log 2>&1 &
    python tools/run_dns_batch.py --classify cases/<case>/results/<run> [...]   # test classifier
    python tools/run_dns_batch.py --dry-run                                     # print the plan
    nohup python -u tools/run_dns_batch.py --set extreme > results/dns_extreme_nohup.log 2>&1 &

Per attempt: set mesh.density in case.json, build + diagnose the mesh on the CPU
(run.py --mesh-only), solve on the GPU (run.py --case ... --name <shape>_tau053_d<100*density>),
classify the run as ok | oom | solver | stopped and pick the next density:
    oom    -> coarser: next untried of 0.90, 0.85, 0.80
    solver -> finer:   next untried of 1.00, 1.05, 1.10
At most 5 attempts per case (an attempt cut off by a crash or reboot is retried at the same
density and not counted); a direction that is used up, or a change of failure class
(solver then oom, or oom then solver), stops that case. Exit -15 (stopped by someone)
stops the whole queue. An attempt cut off by a crash or reboot is retried once at the same
density; cut off twice there counts as oom. Solves run without JAX VRAM preallocation, at nice 10.
--resume (used at boot by tools/pnpflow-dns-queue.service) restarts an unfinished queue; a lock
file keeps one queue at a time. Events go to results/dns_tau053_queue.log, state to
results/dns_tau053_status.json. At the end the figure script tools/figure_dns6.py runs.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
RESULTS = os.path.join(ROOT, "results")
QUEUE_LOG = os.path.join(RESULTS, "dns_tau053_queue.log")
STATUS = os.path.join(RESULTS, "dns_tau053_status.json")

CASES = [("narrow", "cases/pore_dns_narrow_coarse"),
         ("converging", "cases/pore_dns_converging_coarse"),
         ("diverging", "cases/pore_dns_diverging_coarse"),
         ("wide", "cases/pore_dns_wide_coarse")]
# what every case.json must hold before a solve (the app once saved template defaults over one)
EXPECT_COMMON = {"L_nm": 10000, "tau_end": 0.535, "dns_init": 1, "dns_times": 1, "c0_mM": 0.948,
                 "phi_w_mV": 10, "D": 1.337e-09, "eps": 7.0832e-10, "h_wall": 0.15}
EXPECT_ALPHA = {"narrow": (1, 1), "converging": (2, 1), "diverging": (1, 2), "wide": (2, 2),
                "r20_200": (0.2, 2), "r200_10": (2, 0.1), "r200_1": (2, 0.01)}
EXPECT_EXTRA = {"r20_200": {"h_radius_n": 6}, "r200_10": {"h_radius_n": 6},
                "r200_1": {"h_radius_n": 6}}
# --set extreme: DNS folders poreRadius20_200, 200_10, 200_1 (DNS data to t = 0.035 s only),
# own status / log files and figure (figureDNS6_extreme at t = 0.01, 0.035 s)
SETS = {
    "main": {"cases": None, "log": "dns_tau053_queue.log", "status": "dns_tau053_status.json",
             "fig_args": []},
    "extreme": {"cases": [("r20_200", "cases/pore_dns_r20_200_coarse"),
                          ("r200_10", "cases/pore_dns_r200_10_coarse"),
                          ("r200_1", "cases/pore_dns_r200_1_coarse")],
                "log": "dns_extreme_queue.log", "status": "dns_extreme_status.json",
                "fig_args": ["--times", "0.01", "0.035", "--name", "figureDNS6_extreme"]},
}
FIG_ARGS = []
START = 0.95
COARSER = [0.90, 0.85, 0.80]
FINER = [1.00, 1.05, 1.10]
MAX_ATTEMPTS = 5

OOM_PATTERNS = re.compile(r"MemoryError|RESOURCE_EXHAUSTED|out of memory|Out of memory|"
                          r"\bOOM\b|pardiso.*memory|memory.*pardiso|std::bad_alloc|Killed",
                          re.IGNORECASE)
NAN_PATTERNS = re.compile(r"\bnan\b|\binf\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def now():
    return datetime.datetime.now().isoformat(timespec="seconds")


DRY = False


def event(msg):
    line = f"{now()}  {msg}"
    print(line, flush=True)
    if DRY:
        return
    os.makedirs(RESULTS, exist_ok=True)
    with open(QUEUE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_json(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def write_json(path, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def load_case(case_dir):
    return read_json(os.path.join(ROOT, case_dir, "case.json"))


def save_case(case_dir, case):
    case = dict(case, modified=now())
    write_json(os.path.join(ROOT, case_dir, "case.json"), case)


def set_density(case_dir, density):
    case = load_case(case_dir)
    case["mesh"]["density"] = density
    save_case(case_dir, case)


def run_name(shape, density):
    return f"{shape}_tau053_d{round(density * 100):03d}"


def unique_folder(case_dir, name):
    """Never reuse an existing run folder: add _2, _3, ... if needed."""
    base = os.path.join(ROOT, case_dir, "results")
    cand, k = name, 1
    while os.path.exists(os.path.join(base, cand)):
        k += 1
        cand = f"{name}_{k}"
    return cand, os.path.join(base, cand)


# ---------------------------------------------------------------------------
# classifier
# ---------------------------------------------------------------------------
def _exit_code_from_log(text):
    m = re.findall(r"# process exited with code (-?\d+)", text)
    return int(m[-1]) if m else None


def _kernel_oom_since(t0):
    """True if the kernel log has an OOM kill since epoch t0 (best effort)."""
    for cmd in (["journalctl", "-k", "--no-pager", "--since", f"@{int(t0)}"], ["dmesg"]):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout
        except Exception:
            continue
        if re.search(r"Out of memory|oom-kill|Killed process", out):
            return True
        if out:
            return False
    return False


def classify(run_dir, returncode=None, started=None):
    """-> (class, reasons) with class in ok | oom | solver | stopped."""
    log_path = os.path.join(run_dir, "stdout.log")
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            text = f.read()
    except OSError:
        text = ""
    if returncode is None:
        returncode = _exit_code_from_log(text)
    summary = read_json(os.path.join(run_dir, "summary.json"), {}) or {}
    reasons = []
    has_conv_line = re.search(r"^converged:", text, re.MULTILINE) is not None

    if returncode in (-15, 143):
        return "stopped", [f"exit code {returncode} (SIGTERM: stopped by someone)"]

    # --- ok ---------------------------------------------------------------
    tr = summary.get("transient") or {}
    t_end, t_reached = tr.get("t_end"), tr.get("t_reached")
    ok_checks = {
        "converged: True in stdout": re.search(r"^converged: True", text, re.MULTILINE) is not None,
        "summary.solver.converged": bool((summary.get("solver") or {}).get("converged")),
        "t_reached >= t_end": (t_end is not None and t_reached is not None
                               and t_reached >= t_end * (1 - 1e-9)),
        "no NaN/inf in log": not NAN_PATTERNS.search(text),
        "exit code 0": returncode in (0, None),
    }
    if all(ok_checks.values()):
        return "ok", [f"t_reached {t_reached:g} = t_end {t_end:g}"]
    reasons += [f"not {k}" for k, v in ok_checks.items() if not v]

    # --- oom --------------------------------------------------------------
    if returncode in (-9, 137):
        return "oom", reasons + [f"exit code {returncode} (SIGKILL)"]
    err = str(summary.get("error") or "")
    if OOM_PATTERNS.search(err) or OOM_PATTERNS.search(text):
        return "oom", reasons + ["out-of-memory message in log/summary"]
    if not has_conv_line and returncode not in (0, None):
        if started is not None and _kernel_oom_since(started):
            return "oom", reasons + ["kernel OOM kill"]
        return "oom", reasons + [f"killed (exit {returncode}) with no converged: line"]

    # --- solver -----------------------------------------------------------
    if err:
        reasons.append("traceback in summary.error: " + err.strip().splitlines()[-1][:200])
    return "solver", reasons


# ---------------------------------------------------------------------------
# one attempt
# ---------------------------------------------------------------------------
def mesh_check(shape, case_dir, density):
    """Build + diagnose the mesh on the CPU. -> (ok, info dict)."""
    env = dict(os.environ, JAX_PLATFORMS="cpu")
    logf = os.path.join(RESULTS, "dns_tau053_mesh", f"{run_name(shape, density)}.log")
    os.makedirs(os.path.dirname(logf), exist_ok=True)
    t0 = time.time()
    with open(logf, "w", encoding="utf-8") as f:
        rc = subprocess.run([PY, "-u", "run.py", "--case", case_dir, "--mesh-only"],
                            cwd=ROOT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    st = read_json(os.path.join(ROOT, case_dir, "mesh", "status.json"), {}) or {}
    stats = st.get("stats") or {}
    chk = st.get("summary_check") or {}
    info = {"mesh_log": os.path.relpath(logf, ROOT), "mesh_rc": rc,
            "mesh_time_s": round(time.time() - t0, 1),
            "triangles": stats.get("n_elements"), "dofs": st.get("n_dofs"),
            "h_min": stats.get("h_min"), "quality_min": stats.get("quality_min"),
            "diagnose": chk.get("status"), "diagnose_note": chk.get("note"),
            "views": [os.path.join(case_dir, "mesh", v) for v in ("view_a.png", "view_b.png")]}
    ok = rc == 0 and st.get("state") == "done"
    if not ok:
        info["mesh_error"] = st.get("error")
    return ok, info


def solve(case_dir, name, folder):
    """Launch run.py on the GPU with stdout.log in the run folder (same header as the app)."""
    os.makedirs(folder, exist_ok=True)
    cmd = [PY, "-u", "run.py", "--case", os.path.join(ROOT, case_dir), "--name", name]
    # no 75 % VRAM preallocation (the desktop shares the GPU) and lower CPU priority, so the
    # machine stays usable during a run
    env = dict(os.environ, JAX_PLATFORMS="cuda", XLA_PYTHON_CLIENT_PREALLOCATE="false")
    started = time.time()
    with open(os.path.join(folder, "stdout.log"), "w", encoding="utf-8") as log:
        log.write(f"# launched by tools/run_dns_batch.py {now()}\n# {' '.join(cmd)}\n"
                  f"# device: gpu (JAX_PLATFORMS=cuda, no VRAM preallocation, nice 10)\n")
        log.flush()
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env,
                                preexec_fn=lambda: os.nice(10))
        rc = proc.wait()
        log.write(f"\n# process exited with code {rc}\n")
    return rc, started, time.time() - started


def check_params(shape, case_dir):
    """-> list of mismatches between case.json params and the DNS set-up."""
    p = (load_case(case_dir).get("params") or {})
    exp = dict(EXPECT_COMMON, alpha_in=EXPECT_ALPHA[shape][0], alpha_out=EXPECT_ALPHA[shape][1],
               **EXPECT_EXTRA.get(shape, {}))
    bad = []
    for k, v in exp.items():
        try:
            ok = abs(float(p.get(k)) - v) <= 1e-9 * max(1.0, abs(v))
        except (TypeError, ValueError):
            ok = False
        if not ok:
            bad.append(f"{k} = {p.get(k)!r} (expected {v})")
    return bad


def next_density(tried, cls):
    seq = COARSER if cls == "oom" else FINER
    for d in seq:
        if d not in tried:
            return d
    return None


# ---------------------------------------------------------------------------
# queue
# ---------------------------------------------------------------------------
def run_queue(shapes, dry_run=False):
    status = read_json(STATUS, {}) or {}
    if dry_run:
        status = json.loads(json.dumps(status))
    status.setdefault("started", now())
    status.setdefault("cases", {})
    event(f"queue start: {', '.join(shapes)} (python {PY})")
    for shape, case_dir in CASES:
        if shape not in shapes:
            continue
        cs = status["cases"].setdefault(shape, {"case": case_dir, "attempts": []})
        if cs.get("result") == "ok":
            event(f"{shape}: already ok ({cs.get('run')}), skipped")
            continue
        cs["attempts"] = cs.get("attempts") or []
        tried = [a["density"] for a in cs["attempts"]]
        density = START if not tried else None
        if tried:   # resume after an interrupted queue
            last = cs["attempts"][-1]
            if last["class"] is None:   # cut off (crash / reboot / freeze)
                n_cut = sum(a["class"] == "interrupted" and a["density"] == last["density"]
                            for a in cs["attempts"][:-1])
                last["finished"] = last.get("finished") or now()
                if n_cut == 0:          # first time: retry the same density
                    last["class"] = "interrupted"
                    event(f"{shape}: attempt at density {last['density']} was interrupted "
                          f"({last.get('run')}); retrying the same density under a new name")
                    density = last["density"]
                else:                   # second time: the run likely froze the machine
                    last["class"] = "oom"
                    last["reasons"] = ["interrupted twice at this density (machine froze or "
                                       "rebooted): treated as oom"]
                    event(f"{shape}: density {last['density']} interrupted twice -> treated as "
                          f"oom, going coarser")
                    density = next_density(tried, "oom")
            elif last["class"] == "interrupted":   # queue died before the retry started
                density = last["density"]
            else:
                density = (next_density(tried, last["class"])
                           if last["class"] in ("oom", "solver") else None)
        classes = [a["class"] for a in cs["attempts"] if a["class"] in ("oom", "solver")]
        def n_real():   # interrupted attempts (crash / reboot) do not count
            return sum(a["class"] != "interrupted" for a in cs["attempts"])
        bad = check_params(shape, case_dir) if density is not None else []
        if bad:
            cs.update(result="bad case.json", bad_params=bad)
            event(f"{shape}: case.json does not match the DNS set-up, case skipped: " + "; ".join(bad))
            density = None
        while density is not None and n_real() < MAX_ATTEMPTS:
            name, folder = unique_folder(case_dir, run_name(shape, density))
            if dry_run:
                event(f"{shape}: would run density {density} as {name}")
                break
            event(f"{shape}: attempt {n_real() + 1}, density {density}: mesh check")
            set_density(case_dir, density)
            att = {"density": density, "run": None, "class": None, "started": now()}
            cs["attempts"].append(att)
            mok, minfo = mesh_check(shape, case_dir, density)
            att.update(minfo)
            write_json(STATUS, status)
            if not mok:
                att["class"] = "solver"
                att["reasons"] = ["mesh build failed: " + str(minfo.get("mesh_error"))]
                event(f"{shape}: mesh build FAILED at density {density} -> class solver")
            else:
                event(f"{shape}: mesh {minfo['triangles']} triangles, {minfo['dofs']} dofs, "
                      f"h_min {minfo['h_min']:.4g}, q_min {minfo['quality_min']:.3f}, "
                      f"diagnose {minfo['diagnose']} ({minfo['diagnose_note']})")
                event(f"{shape}: solving {name} on the GPU")
                att["run"] = os.path.relpath(folder, ROOT)
                write_json(STATUS, status)
                rc, started, wall = solve(case_dir, name, folder)
                cls, reasons = classify(folder, rc, started)
                summ = read_json(os.path.join(folder, "summary.json"), {}) or {}
                tr = summ.get("transient") or {}
                att.update({"class": cls, "reasons": reasons, "exit_code": rc,
                            "wall_time_s": round(wall, 1), "t_reached": tr.get("t_reached"),
                            "t_end": tr.get("t_end"), "n_steps": tr.get("n_steps"),
                            "checks": [{k: c.get(k) for k in ("name", "value", "status")}
                                       for c in summ.get("checks", [])]})
                event(f"{shape}: {name} -> {cls} (exit {rc}, {wall / 60:.1f} min): "
                      + "; ".join(reasons))
            att["finished"] = now()
            write_json(STATUS, status)
            cls = att["class"]
            if cls == "ok":
                cs.update(result="ok", run=att["run"], density=density)
                break
            if cls == "stopped":
                cs.update(result="stopped")
                write_json(STATUS, status)
                event(f"{shape}: run was stopped (SIGTERM); queue ends, ask the user")
                return status
            if classes and classes[-1] != cls:
                cs.update(result=f"conflict ({classes[-1]} then {cls})")
                event(f"{shape}: failure classes conflict ({classes[-1]} then {cls}); case stopped")
                break
            classes.append(cls)
            density = next_density([a["density"] for a in cs["attempts"]], cls)
            if density is None:
                cs.update(result=f"exhausted ({cls})")
                event(f"{shape}: no untried density left for {cls}; case stopped")
            elif n_real() >= MAX_ATTEMPTS:
                cs.update(result="max attempts")
                event(f"{shape}: {MAX_ATTEMPTS} attempts used; case stopped")
        if dry_run:
            continue
        # leave case.json at the density of the successful attempt (else the last tried)
        if cs.get("result") == "ok":
            set_density(case_dir, cs["density"])
        write_json(STATUS, status)
    if dry_run:
        return status
    status["finished"] = now()
    write_json(STATUS, status)
    event("queue done: " + ", ".join(f"{s}={c.get('result')}" for s, c in status["cases"].items()))
    fig = os.path.join(ROOT, "tools", "figure_dns6.py")
    if os.path.isfile(fig) and any(c.get("result") == "ok" for c in status["cases"].values()):
        event(f"figure: tools/figure_dns6.py --status {os.path.relpath(STATUS, ROOT)} "
              + " ".join(FIG_ARGS))
        rc = subprocess.run([PY, fig, "--status", STATUS, "--dns",
                             os.path.join(ROOT, "DNS_compare", "transportFolder")] + FIG_ARGS,
                            cwd=ROOT).returncode
        event(f"figure: exit {rc}")
    return status


def main():
    global DRY, CASES, QUEUE_LOG, STATUS, FIG_ARGS
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--set", choices=list(SETS), default="main",
                    help="main: the four DNS shapes; extreme: r20_200, r200_10, r200_1 "
                         "(own status/log files, figureDNS6_extreme)")
    ap.add_argument("shapes", nargs="*", metavar="SHAPE",
                    help="subset of: " + " ".join(s for s, _ in CASES) + " (default: all)")
    ap.add_argument("--classify", nargs="+", metavar="RUN_DIR",
                    help="only classify existing run folders and print the result")
    ap.add_argument("--dry-run", action="store_true", help="print the next attempt per case")
    ap.add_argument("--resume", action="store_true",
                    help="do nothing if the queue already finished (used at boot by "
                         "tools/pnpflow-dns-queue.service)")
    args = ap.parse_args()
    if args.classify:
        for d in args.classify:
            cls, reasons = classify(d)
            print(f"{cls:8s} {d}\n         " + "\n         ".join(reasons))
        return
    DRY = args.dry_run
    cfg = SETS[args.set]
    CASES = cfg["cases"] or CASES
    QUEUE_LOG = os.path.join(RESULTS, cfg["log"])
    STATUS = os.path.join(RESULTS, cfg["status"])
    FIG_ARGS = cfg["fig_args"]
    shapes = args.shapes or [s for s, _ in CASES]
    bad = [s for s in shapes if s not in dict(CASES)]
    if bad:
        ap.error(f"unknown shape(s): {bad}")
    if args.resume:
        st = read_json(STATUS, {}) or {}
        if st.get("finished"):
            print(f"queue already finished ({STATUS}); nothing to resume")
            return
        if any(c.get("result") == "stopped" for c in (st.get("cases") or {}).values()):
            print("queue was stopped by someone (exit -15); not resuming")
            return
    if not args.dry_run:   # one queue at a time
        import fcntl
        os.makedirs(RESULTS, exist_ok=True)
        lock = open(os.path.join(RESULTS, "dns_tau053.lock"), "w")
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("another run_dns_batch.py queue is running; exiting")
            return
    run_queue(shapes, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
