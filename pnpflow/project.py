"""Cases (everything before a run) and their results.

Layout (docs/case_answers.md):

    cases/<case>/case.json            template, parameters, mesh options, time settings,
                                      figures wanted, display units
    cases/<case>/mesh/case.msh        the mesh (P2, named physical groups), reused by runs
    cases/<case>/mesh/status.json     state of the last mesh build + the key it was built for
    cases/<case>/mesh/view_a.png, view_b.png, validation.json, mesh.json
    cases/<case>/results/<run>/       one folder per run: frozen case.json, mesh/, plots,
                                      figures, csv, convergence.txt, summary.json, ...

Plain Python on purpose (no jax import): the app server may read these helpers.
"""
import copy
import datetime
import hashlib
import json
import math
import os

FORMAT = 1

DEFAULT_MESH = {
    "density": 1.0,          # element density factor: 2 = every size halved
    "boundaries": [],        # [{"boundary", "size", "layer", "growth"}]
    "regions": [],           # [{"kind": "box", "x0", "x1", "y0", "y1", "size", "growth"}
                             #  | {"kind": "disk", "x", "y", "r", "size", "growth"}]
    "corners": {"enabled": False, "size": 0.0, "radius": 0.0, "max_angle": 80.0},
    "paint": [],             # [{"x", "y", "r", "size"}] spray-paint spots
    "paint_growth": 1.2,
}
DEFAULT_TIME = {
    # which times are saved: "template" (the case's own list), "uniform", "log", "list"
    "save": {"mode": "template", "dtau": 0.05, "n": 20, "tau_min": 1e-4, "list": []},
    "dt_max": None,          # largest step in tau units (None = template default)
    "max_rejects": 12,       # rejected steps in a row before giving up
}


def default_case(template, params, name):
    now = datetime.datetime.now().isoformat(timespec="seconds")
    return {
        "format": FORMAT, "name": name, "template": template, "description": "",
        "created": now, "modified": now,
        "params": dict(params),
        "units": {},                                   # display unit per parameter (UI)
        "figures": {"standard": True},                 # which figures to make (to be extended)
        "mesh": copy.deepcopy(DEFAULT_MESH),
        "time": copy.deepcopy(DEFAULT_TIME),
    }


def normalize(case):
    """Fill missing blocks with defaults (tolerates hand-written or older files)."""
    c = dict(case)
    c.setdefault("format", FORMAT)
    c.setdefault("params", {})
    c.setdefault("units", {})
    c.setdefault("figures", {"standard": True})
    mesh = copy.deepcopy(DEFAULT_MESH)
    mesh.update(c.get("mesh") or {})
    corners = copy.deepcopy(DEFAULT_MESH["corners"])
    corners.update((c.get("mesh") or {}).get("corners") or {})
    mesh["corners"] = corners
    c["mesh"] = mesh
    time = copy.deepcopy(DEFAULT_TIME)
    time.update(c.get("time") or {})
    save = copy.deepcopy(DEFAULT_TIME["save"])
    save.update((c.get("time") or {}).get("save") or {})
    time["save"] = save
    c["time"] = time
    return c


def load(case_dir):
    with open(os.path.join(case_dir, "case.json"), encoding="utf-8") as f:
        return normalize(json.load(f))


def save(case_dir, case):
    os.makedirs(case_dir, exist_ok=True)
    case = dict(case, modified=datetime.datetime.now().isoformat(timespec="seconds"))
    tmp = os.path.join(case_dir, "case.json.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(case, f, indent=1)
    os.replace(tmp, os.path.join(case_dir, "case.json"))
    return case


# Which construction step owns a template parameter, from its name. First match wins;
# anything else belongs to Study. The app reads this through `run.py --list` ("steps").
STEP_RULES = [
    ("mesh", r"^(h|h_\w+|\w+_size|n_elem\w*|refine\w*)$"),
    ("geometry", r"^(R|R_out|radius|length|width|half_width|upstream|downstream|L|H|W|L_nm|"
                 r"a_ref\w*|alpha_\w+|sdl_\w+|domain\w*|far_\w+)$"),
    ("materials", r"^(T|delta|d_\w+|D|eps\w*|c0\w*|bulk\w*|z_\w+|q)$"),
    ("boundaries", r"^(zeta\w*|phi_w\w*|stern\w*|side_slip|end_charged|outer_bc|sigma\w*)$"),
    ("equations", r"^(Pe|Re|K|Ex|advection|supg)$"),
]
MESH_STEPS = ("geometry", "mesh", "materials")   # materials: default sizes follow delta


def step_of(key):
    import re
    for step, rx in STEP_RULES:
        if re.match(rx, key):
            return step
    return "study"


def mesh_key(case):
    """Hash of everything that can change the mesh: the template, the mesh block and
    the geometry, mesh and materials parameters (default sizes follow delta). Time,
    boundary and physics settings do not trigger a new mesh."""
    c = normalize(case)
    prm = {k: v for k, v in c["params"].items() if step_of(k) in MESH_STEPS}
    blob = json.dumps({"template": c.get("template"), "params": prm,
                       "mesh": c["mesh"]}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# saved times
# ---------------------------------------------------------------------------
def save_times(save, tau_end):
    """Output times (tau units) for a save block; None for mode "template".
    tau_end is always included; times outside (0, tau_end] are dropped."""
    mode = (save or {}).get("mode", "template")
    if mode == "template":
        return None
    if mode == "uniform":
        d = float(save.get("dtau") or 0)
        if not d > 0:
            raise ValueError("save every dtau: dtau must be > 0")
        n = int(math.floor(tau_end / d + 1e-9))
        if n > 5000:
            raise ValueError(f"save every dtau = {d:g} gives {n} outputs (limit 5000)")
        taus = [d * k for k in range(1, n + 1)]
    elif mode == "log":
        n = int(save.get("n") or 0)
        t0 = float(save.get("tau_min") or 0)
        if n < 1 or not 0 < t0 < tau_end:
            raise ValueError("log-spaced saves need n >= 1 and 0 < tau_min < tau_end")
        if n == 1:
            taus = [tau_end]
        else:
            r = math.log(tau_end / t0) / (n - 1)
            taus = [t0 * math.exp(r * k) for k in range(n)]
    elif mode == "list":
        taus = [float(t) for t in (save.get("list") or [])]
    else:
        raise ValueError(f"unknown save mode '{mode}'")
    taus = sorted({round(t, 12) for t in taus if 0 < t <= tau_end * (1 + 1e-9)} | {tau_end})
    return taus


# ---------------------------------------------------------------------------
# convergence.txt
# ---------------------------------------------------------------------------
def _f(v, w=11, p=4):
    if v is None:
        return "-".rjust(w)
    try:
        return f"{float(v):{w}.{p}e}"
    except (TypeError, ValueError):
        return str(v).rjust(w)


def convergence_text(summary, progress=None):
    """Plain-text account of how the solve converged (steady stages, time steps,
    equilibrium check), from summary.json (or progress.json history on failure)."""
    s = summary or {}
    solver = s.get("solver") or {}
    out = ["pnpflow convergence log",
           f"run      : {s.get('run_name', '?')}",
           f"template : {s.get('case', '?')}",
           f"created  : {s.get('created', '?')}"]
    dev = s.get("device") or {}
    if dev:
        out.append(f"device   : {dev.get('platform', '?')} ({', '.join(dev.get('devices', []))})")
    conv = solver.get("converged")
    out.append(f"result   : {'converged' if conv else 'NOT converged' if conv is False else 'unknown'}"
               + (f", solve {solver['time_s']:.1f} s" if isinstance(solver.get("time_s"), (int, float)) else "")
               + (f", wall {solver['wall_time_s']:.1f} s" if isinstance(solver.get("wall_time_s"), (int, float)) else ""))
    if s.get("error"):
        out += ["", "error:", s["error"].rstrip()]

    stages = solver.get("stages") or []
    if stages:
        out += ["", "steady Newton solve (continuation stages)",
                "  newton tolerance 1e-9 on |R|; alpha = line-search step"]
        for i, st in enumerate(stages, 1):
            prm = ", ".join(f"{k} = {v:.6g}" if isinstance(v, (int, float)) else f"{k} = {v}"
                            for k, v in (st.get("params") or {}).items())
            out.append(f"  stage {i}: {'converged' if st.get('converged') else 'failed'}"
                       + (f"  [{prm}]" if prm else ""))
            out.append(f"    {'it':>4} {'|R|':>11} {'|dU|':>11} {'alpha':>7}")
            for it in st.get("iterations") or []:
                out.append(f"    {it.get('it', ''):>4} {_f(it.get('res_norm'))} {_f(it.get('step_norm'))} "
                           f"{(it.get('alpha') if it.get('alpha') is not None else float('nan')):7.3f}")

    tr = s.get("transient") or {}
    steps = tr.get("steps") or []
    if tr:
        t_per = tr.get("t_per_tau") or 1.0
        n_rej = sum(int(x.get("rejects") or 0) for x in steps)
        out += ["", f"time stepping: {tr.get('scheme', 'BDF2')}, {len(steps)} accepted steps, "
                    f"{n_rej} rejected, reached tau = {(tr.get('t_reached') or 0) / t_per:.6g} "
                    f"of {(tr.get('t_end') or 0) / t_per:.6g}",
                f"  {'step':>5} {'tau':>11} {'dtau':>11} {'newton':>6} {'err':>11} {'rejects':>7} {'|R| final':>11}"]
        for k, x in enumerate(steps, 1):
            out.append(f"  {k:>5} {_f((x.get('t') or 0) / t_per)} {_f((x.get('dt') or 0) / t_per)} "
                       f"{x.get('newton_its', ''):>6} {_f(x.get('err'))} {x.get('rejects', 0):>7} "
                       f"{_f(x.get('res_final'))}")
        eq = solver.get("equilibrium")
        if eq:
            out.append(f"  steady equilibrium check: {'converged' if eq.get('converged') else 'failed'}, "
                       f"{eq.get('newton_its', '?')} Newton iterations")

    if not stages and not steps and progress and progress.get("history"):
        out += ["", "Newton history from progress.json (the run did not finish)",
                f"  {'stage':>5} {'it':>4} {'|R|':>11} {'|dU|':>11} {'alpha':>7}"]
        for it in progress["history"]:
            a = it.get("alpha")
            out.append(f"  {it.get('stage', ''):>5} {it.get('it', ''):>4} {_f(it.get('res_norm'))} "
                       f"{_f(it.get('step_norm'))} {a if a is not None else float('nan'):7.3f}")

    checks = s.get("checks") or []
    if checks:
        out += ["", "checks"]
        for c in checks:
            v = c.get("value")
            out.append(f"  [{c.get('status', '?'):4s}] {c.get('name', '')}: "
                       f"{_f(v, 1, 3).strip() if isinstance(v, (int, float)) else v}  ({c.get('note', '')})")
    return "\n".join(out) + "\n"


def write_convergence(folder, summary, progress=None):
    with open(os.path.join(folder, "convergence.txt"), "w", encoding="utf-8") as f:
        f.write(convergence_text(summary, progress))
