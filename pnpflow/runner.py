"""Run a case end to end: mesh -> solve -> analyse -> write results."""
import datetime
import json
import os
import time
import traceback

import jax.numpy as jnp
import numpy as np

from . import output
from .analysis import Analysis
from .cases import CASES
from .solver import Assembler, Newton, continuation


def run_case(name, overrides=None, run_name=None, root="results", plots=True,
             log=print):
    spec = CASES[name]
    params = dict(spec["params"])
    for k, v in (overrides or {}).items():
        if k not in params:
            raise KeyError(f"unknown parameter '{k}' for case '{name}'")
        params[k] = float(v)
    run_name = run_name or f"{name}_{datetime.datetime.now():%Y%m%d_%H%M%S}"
    folder = os.path.join(root, run_name)
    os.makedirs(folder, exist_ok=True)
    progress = output.Progress(folder)
    t_start = time.time()
    post = {}                     # mesh views / checks, kept if the solve fails
    try:
        summary = _run(spec, name, params, run_name, folder, progress, plots, log, post)
    except Exception as exc:
        progress.set(state="failed", message=f"{type(exc).__name__}: {exc}")
        output.write_json(os.path.join(folder, "summary.json"), {
            "run_name": run_name, "case": name, "input": params,
            "error": traceback.format_exc(), "solver": {"converged": False}, **post})
        raise
    summary["solver"]["wall_time_s"] = time.time() - t_start
    output.write_json(os.path.join(folder, "summary.json"), summary)
    try:
        from .export import export_csv
        n = len(export_csv(folder))
        log(f"csv: {n} files written to csv/")
    except Exception as exc:                        # never fail a run on export
        log(f"  csv export failed: {type(exc).__name__}: {exc}")
    progress.set(state="done", message="finished")
    return summary


def _device_info():
    try:
        import jax
        d = jax.devices()
        return {"platform": d[0].platform, "devices": [str(x) for x in d]}
    except Exception as exc:                        # pragma: no cover
        return {"platform": "unknown", "error": str(exc)}


def _run(spec, name, params, run_name, folder, progress, plots, log, post):
    # ---- build ----------------------------------------------------------
    t0 = time.time()
    post["device"] = _device_info()
    progress.set(device=post["device"])
    log(f"jax device: {post['device'].get('platform')} "
        f"({', '.join(post['device'].get('devices', []))})")
    setup = spec["build"](params)
    model = setup.model
    mesh_stats = model.mesh.stats()
    log(f"mesh: {mesh_stats['n_elements']} triangles, {model.ndof} dofs "
        f"({time.time() - t0:.1f}s)")
    pp_checks = _mesh_postprocess(model, setup, folder, post, log)
    if setup.transient:
        return _run_transient(spec, name, params, run_name, folder, progress, plots,
                              log, post, setup, mesh_stats, pp_checks)

    # ---- solve ----------------------------------------------------------
    progress.set(state="assembling", message="compiling jax kernels",
                 n_stages=len(setup.path) or 1)
    t1 = time.time()
    asm = Assembler(model)
    newton = Newton(asm, tol=1e-9, log=log, on_iteration=progress.iteration)
    U = model.initial_guess(setup.base)
    if setup.path:
        U, final, ok, stages = continuation(newton, U, setup.base, setup.path, log=log)
    else:
        U, ok, hist = newton.solve(U, setup.base, stage=1)
        final, stages = setup.base, [{"params": {}, "converged": ok, "iterations": hist}]
    solve_time = time.time() - t1
    log(f"solve: converged={ok} in {solve_time:.1f}s")

    # ---- analyse --------------------------------------------------------
    progress.set(state="post", message="analysis")
    an = Analysis(model, asm, U, final)
    an.flux = an.species_fluxes()
    fields = an.nodal_fields()
    checks = _common_checks(an, fields)
    integrals = {"flux": an.flux, "other": {}}
    if model.flow in ("stokes", "navier_stokes"):
        integrals["other"]["div_u_L2"] = an.div_u_l2()
    if setup.extra:
        more_checks, more = setup.extra(an)
        checks += more_checks
        integrals.update(more)

    sens = {}
    if setup.quantities and ok:
        log("adjoint sensitivities ...")
        names = [k for k in setup.sens_params if k in final]
        sens = an.sensitivities(setup.quantities(an), names)

    profiles = {"lines": setup.profiles(an) if setup.profiles else []}

    # ---- write ----------------------------------------------------------
    output.write_fields(folder, model.mesh, fields, setup.focus_bbox, setup.labels)
    output.write_json(os.path.join(folder, "profiles.json"), _strip(profiles))
    output.write_meshfiles(folder, model.mesh, fields)
    np.savez_compressed(os.path.join(folder, "solution.npz"), U=np.asarray(U),
                        X=model.mesh.X, tri=model.mesh.tri)
    if plots:
        try:
            output.write_plots(folder, model.mesh, fields, profiles, setup.focus_bbox)
        except Exception as exc:
            pp_checks.append(_pp_warn("plots/ (legacy)", exc, log))

    summary = {
        "run_name": run_name, "case": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "input": params,
        "model": {
            "species": [{"name": s.name, "z": s.z, "d": s.d, "bulk": s.bulk}
                        for s in model.species],
            "flow": model.flow, "advection": model.advection, "supg": model.supg,
            "coord": getattr(model, "coord", "planar"),
            "boundaries": _describe_bcs(model),
            "doc": spec["doc"],
        },
        "params": {k: float(v) for k, v in final.items()},
        "param_info": spec["info"],
        "mesh": mesh_stats | {"n_dofs": int(model.ndof)},
        "solver": {"converged": bool(ok), "time_s": solve_time,
                   "n_newton_total": sum(len(s["iterations"]) - 1 for s in stages),
                   "stages": stages},
        "checks": checks,
        "integrals": integrals,
        "sensitivities": sens,
        "notes": setup.notes,
        **post,
    }
    summary["figures"] = []
    if plots:
        progress.set(state="post", message="figures")
        try:
            from . import figures
            summary["figures"] = figures.write_run_figures(
                folder, model.mesh, fields, profiles, summary,
                focus=setup.focus_bbox, symmetry=summary["model"]["coord"], log=log)
            log(f"figures: {len(summary['figures'])} written to figures/ and svg/")
        except Exception as exc:
            pp_checks.append(_pp_warn("figures", exc, log))
    summary["checks"] = checks + pp_checks
    return summary


def _run_transient(spec, name, params, run_name, folder, progress, plots, log, post,
                   setup, mesh_stats, pp_checks):
    """Time-dependent run (setup.transient): adaptive BDF2, profiles at output
    times, then (optionally) the steady equilibrium for the setup's checks."""
    from .transient import TransientAssembler, integrate
    model, tr = setup.model, setup.transient
    t_end = float(tr["t_end"])
    t_per = float(tr.get("t_per_tau", 1.0))
    progress.set(state="assembling", message="compiling jax kernels", n_stages=1)
    t1 = time.time()
    asm = Assembler(model)
    tasm = TransientAssembler(asm)
    clock = {"t": 0.0, "dt": None, "step": 0, "rejects": 0}

    def time_fraction():
        return float(min(1.0, clock["t"] / t_end))

    def on_newton(stage, history, status=None):
        progress.iteration(stage, history, status=status)
        progress.set(fraction=time_fraction(),
                     message=f"step {stage}: tau = {clock['t'] / t_per:.4g}, "
                             f"Newton it {history[-1]['it']}")

    def on_step(info):
        if info["accepted"]:
            clock.update(t=info["t"] + info["dt"], dt=info["dt"], step=info["n_steps"] + 1)
        clock["rejects"] = info["rejects"] + (0 if info["accepted"] else 1)
        health = "converging" if info["accepted"] else "struggling"
        tells = dict(progress.data.get("tells") or {})
        tells.update(dt=info["dt"], newton_its=info["newton_its"], err=info["err"],
                     rejects=clock["rejects"])
        progress.set(state="solving", fraction=time_fraction(), health=health, tells=tells,
                     time={"t": clock["t"], "t_end": t_end, "tau": clock["t"] / t_per,
                           "tau_end": t_end / t_per, "dt": info["dt"],
                           "step": clock["step"], "rejects": clock["rejects"]})

    newton = Newton(tasm, tol=1e-9, log=lambda *a: None, on_iteration=on_newton)
    snapshots = []
    prm = tr.get("params", setup.base)
    U0_given = tr.get("U0")
    U0 = model.initial_guess(prm) if U0_given is None else jnp.asarray(U0_given)
    progress.set(state="solving", message="time stepping", fraction=0.0, health="converging")
    U, t, steps, ok = integrate(
        newton, tasm, U0, prm, t_end, outputs=tr.get("outputs", ()),
        consistent_init=U0_given is None,
        dt0=tr.get("dt0", 1e-2), dt_max=tr.get("dt_max"), rtol=tr.get("rtol", 1e-3),
        on_output=lambda tt, UU: snapshots.append((float(tt), np.asarray(UU))),
        on_step=on_step, log=log)
    solve_time = time.time() - t1
    log(f"time stepping: ok={ok}, {len(steps)} steps to t = {t:.6g} "
        f"(tau = {t / t_per:.4g}) in {solve_time:.1f}s")
    if not ok:
        progress.set(health="failed", message=f"time step failed at tau = {t / t_per:.4g}")

    # ---- analyse --------------------------------------------------------
    progress.set(state="post", message="analysis")
    an = Analysis(model, asm, U, prm)
    fields = an.nodal_fields()
    checks = [{"name": f"min {s.name} (all times)",
               "value": float(min(np.min(UU[i * model.n:(i + 1) * model.n])
                                  for _, UU in snapshots)),
               "limit": 0.0, "status": "pass", "note": "concentration must stay positive"}
              for i, s in enumerate(model.species)]
    for c in checks:
        c["status"] = "pass" if c["value"] > 0 else "fail"
    checks.append({"name": "time integration reached t_end", "value": float(t / t_end),
                   "limit": 1.0, "status": "pass" if ok else "fail",
                   "note": f"{len(steps)} BDF2 steps, "
                           f"{sum(s['rejects'] for s in steps)} rejected"})
    lines, integrals = tr["profiles"](an, snapshots)

    eq_lines, eq_info = [], {}
    if tr.get("equilibrium") and ok:
        progress.set(state="solving", message="steady equilibrium check")
        newton_s = Newton(asm, tol=1e-9, log=log)
        U_eq = model.initial_guess(setup.base)
        if setup.path:
            U_eq, p_eq, ok_eq, st_eq = continuation(newton_s, U_eq, setup.base,
                                                    setup.path, log=log)
        else:
            U_eq, ok_eq, hist = newton_s.solve(U_eq, setup.base)
            p_eq, st_eq = setup.base, [{"iterations": hist}]
        eq_info = {"converged": bool(ok_eq),
                   "newton_its": sum(len(s["iterations"]) - 1 for s in st_eq)}
        if ok_eq:
            an_eq = Analysis(model, asm, U_eq, p_eq)
            an_eq.flux = an_eq.species_fluxes()
            checks += [dict(c, name=c["name"] + " (equilibrium)")
                       for c in _common_checks(an_eq, an_eq.nodal_fields())]
            if setup.extra:
                more_checks, more = setup.extra(an_eq)
                checks += more_checks
                integrals.update(more)
            if setup.profiles:
                eq_lines = setup.profiles(an_eq)
        else:
            checks.append({"name": "steady equilibrium solve", "value": 1.0, "limit": 0.0,
                           "status": "warn", "note": "Newton did not converge"})

    profiles = {"lines": lines + eq_lines}
    output.write_fields(folder, model.mesh, fields, setup.focus_bbox, setup.labels)
    output.write_json(os.path.join(folder, "profiles.json"), _strip(profiles))
    output.write_fields_time(folder, model, snapshots, t_per)
    output.write_meshfiles(folder, model.mesh, fields)
    np.savez_compressed(os.path.join(folder, "solution.npz"), U=np.asarray(U),
                        X=model.mesh.X, tri=model.mesh.tri,
                        t=np.array([s[0] for s in snapshots]),
                        U_t=np.stack([s[1] for s in snapshots]))

    summary = {
        "run_name": run_name, "case": name,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "input": params,
        "model": {
            "species": [{"name": s.name, "z": s.z, "d": s.d, "bulk": s.bulk}
                        for s in model.species],
            "flow": model.flow, "advection": model.advection, "supg": model.supg,
            "coord": getattr(model, "coord", "planar"),
            "boundaries": _describe_bcs(model),
            "doc": spec["doc"],
        },
        "params": {k: float(v) for k, v in prm.items()},
        "param_info": spec["info"],
        "mesh": mesh_stats | {"n_dofs": int(model.ndof)},
        "solver": {"converged": bool(ok), "time_s": solve_time,
                   "n_newton_total": sum(s["newton_its"] for s in steps),
                   "stages": [], "equilibrium": eq_info},
        "transient": {"scheme": "BDF2 (variable step, first step backward Euler)",
                      "t_end": t_end, "t_reached": t, "t_per_tau": t_per,
                      "n_steps": len(steps), "steps": steps,
                      "output_times": [s[0] for s in snapshots]},
        "checks": checks,
        "integrals": integrals,
        "sensitivities": {},
        "notes": setup.notes,
        **post,
    }
    summary["figures"] = []
    if plots:
        progress.set(state="post", message="figures")
        from . import figures
        try:
            summary["figures"] = figures.write_run_figures(
                folder, model.mesh, fields, {"lines": eq_lines}, summary,
                focus=setup.focus_bbox, symmetry=summary["model"]["coord"], log=log)
        except Exception as exc:
            pp_checks.append(_pp_warn("figures", exc, log))
        if tr.get("figures"):
            try:
                summary["figures"] += tr["figures"](folder, lines, summary, log)
            except Exception as exc:
                pp_checks.append(_pp_warn("case figures", exc, log))
        log(f"figures: {len(summary['figures'])} written to figures/ and svg/")
    summary["checks"] = checks + pp_checks
    return summary


def _pp_warn(what, exc, log):
    """A post-processing failure: logged and reported, never fatal."""
    log(f"  post-processing '{what}' failed: {type(exc).__name__}: {exc}")
    return {"name": f"post-processing: {what}", "value": 1.0, "limit": 0.0,
            "status": "warn", "note": f"{type(exc).__name__}: {exc}"[:300],
            "group": "postprocess"}


def _mesh_postprocess(model, setup, folder, post, log):
    """Two-view mesh render + mesh diagnose (docs/MCP.md §4-5), before the solve.

    Fills post["mesh_views"] and post["mesh_checks"]; returns checks for
    summary["checks"]: one aggregate "mesh diagnose" entry (group "mesh"), or
    a warn if this step itself failed.
    """
    out = []
    mesh = model.mesh
    coord = getattr(model, "coord", "planar")
    try:
        from . import meshtools
        post["mesh_views"] = meshtools.render_two_views(mesh, folder, symmetry=coord,
                                                        log=log)
    except Exception as exc:
        post["mesh_views"] = []
        out.append(_pp_warn("mesh views", exc, log))
    try:
        from . import meshtools
        mc = meshtools.diagnose(mesh, model.bcs, model.params.get("delta"),
                                ndof=int(model.ndof), folder=folder,
                                n_species=len(model.species))
        post["mesh_checks"] = mc
        fails = [c["name"] for c in mc if c["status"] == "fail"]
        warns = [c["name"] for c in mc if c["status"] == "warn"]
        status = "fail" if fails else ("warn" if warns else "pass")
        note = ("; ".join(f"{c['name']}: {c['note']}" for c in mc
                          if c["status"] in ("fail", "warn"))
                or "all mesh checks pass") + " (details: mesh_checks, mesh/diagnose.json)"
        out.append({"name": "mesh diagnose", "value": float(len(fails)), "limit": 0.0,
                    "status": status, "note": note, "group": "mesh"})
        for c in mc:
            if c["status"] in ("fail", "warn"):
                log(f"  mesh [{c['status']}] {c['name']}: {c['note']}")
    except Exception as exc:
        post["mesh_checks"] = []
        out.append(_pp_warn("mesh diagnose", exc, log))
    return out


def _common_checks(an, fields):
    from .cases import _check
    checks = []
    m = an.m
    for s in m.species:
        cmin = float(np.min(fields[s.name]))
        checks.append({"name": f"min {s.name}", "value": cmin, "limit": 0.0,
                       "status": "pass" if cmin > 0 else "fail",
                       "note": "concentration must stay positive"})
        net = an.flux[s.name]["net"] - an.flux[s.name].get("volume_source", 0.0)
        scale = max(1e-6, max(abs(v) for k, v in an.flux[s.name].items() if k != "net"))
        checks.append(_check(f"{s.name} balance", abs(net) / scale, 1e-6,
                             "|outflow - sources| / largest boundary flux"))
    if "div_u" in fields:
        checks.append({"name": "max |div u| (nodal)", "value": float(np.abs(fields["div_u"]).max()),
                       "limit": 0.0, "status": "info",
                       "note": "Taylor-Hood: weakly divergence free"})
    return checks


def _describe_bcs(model):
    out = {}
    for name, spec in model.bcs.items():
        out[name] = {k: _bc_text(v) for k, v in spec.items()}
    return out


def _bc_text(v):
    if isinstance(v, dict):
        return {k: _bc_text(x) for k, x in v.items()}
    if isinstance(v, str):
        return v
    t = type(v).__name__
    fields = getattr(v, "__dict__", {})
    args = ", ".join(f"{k}={x if not callable(x) else 'f(x)'}" for k, x in fields.items())
    return f"{t}({args})"


def _strip(profiles):
    for line in profiles["lines"]:
        line.pop("points", None)
    return profiles


def list_cases():
    return {k: {"params": v["params"], "info": v["info"], "doc": v["doc"]}
            for k, v in CASES.items()}
