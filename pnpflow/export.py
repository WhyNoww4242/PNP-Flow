"""CSV export of a finished run (results/<run>/csv/), for comparing with other codes.

Reads profiles.json and summary.json only, so it works on any run, old or new:

    python tools/export_csv.py results/<run> [results/<run2> ...]

Files written (plain comma-separated, one header row, '#' comment lines on top
with the run name, case and units):
  time-dependent lines (profiles.json lines with a "time" key), grouped by the
  line name without its "_tau_<value>" suffix, one file per group and quantity:
      <group>__<quantity>.csv      coord, <quantity>_tau=<t1>, <quantity>_tau=<t2>, ...
      e.g. axial__mu.csv (Z, mu at each tau), radial_Z0.5__phi.csv (R/alpha, phi ...)
  other lines (steady profiles):
      <line>.csv                   coord, every quantity of that line
  time series in summary.integrals that are dicts of equal-length lists
  (e.g. pore_charging's "charging"):
      timeseries_<name>.csv
  profiles_long.csv                every value in long form:
                                   line, tau, coord_name, coord, quantity, value
NaN is written as an empty cell.
"""
import csv
import json
import os
import re

TAU_SUFFIX = re.compile(r"_tau_([0-9.eE+-]+)$")


def _num(v):
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "" if f != f else repr(f)


def _safe(name):
    return re.sub(r"[^A-Za-z0-9_.+-]", "_", name)


def _write(path, header, rows, comments):
    with open(path, "w", newline="", encoding="utf-8") as f:
        for c in comments:
            f.write(f"# {c}\n")
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def export_csv(folder):
    """Write <folder>/csv/*.csv; returns the list of file names written."""
    with open(os.path.join(folder, "profiles.json"), encoding="utf-8") as f:
        lines = json.load(f).get("lines", [])
    summary = {}
    sp = os.path.join(folder, "summary.json")
    if os.path.isfile(sp):
        with open(sp, encoding="utf-8") as f:
            summary = json.load(f)
    out_dir = os.path.join(folder, "csv")
    os.makedirs(out_dir, exist_ok=True)
    head = [f"run: {summary.get('run_name', os.path.basename(folder.rstrip('/')))}",
            f"case: {summary.get('case', '?')}",
            "units: nondimensional (see summary.json notes); "
            "pore_charging: rho=(c+-c-)/c0, phi=e phi/kT, mu=rho+2phi, tau=D t/l_p^2, "
            "Z=z/l_p, R/alpha = r / local wall radius"]
    written = []

    groups, steady = {}, []
    for ln in lines:
        m = TAU_SUFFIX.search(ln.get("name", ""))
        if "time" in ln and m:
            groups.setdefault(ln["name"][:m.start()], []).append(ln)
        else:
            steady.append(ln)

    for group, lns in groups.items():
        lns = sorted((l for l in lns if l["time"].get("tau", 0) > 0),
                     key=lambda l: l["time"]["tau"])
        if not lns:
            continue
        coord = lns[0]["coord"]
        cname = lns[0].get("coord_name", "s")
        for q in lns[0]["values"]:
            cols = [l for l in lns if q in l["values"] and len(l["coord"]) == len(coord)]
            header = [cname] + [f"{q}_tau={l['time']['tau']:.6g}" for l in cols]
            rows = [[_num(coord[i])] + [_num(l["values"][q][i]) for l in cols]
                    for i in range(len(coord))]
            name = f"{_safe(group)}__{_safe(q)}.csv"
            _write(os.path.join(out_dir, name), header, rows,
                   head + [f"{group}: {q} vs {cname}, one column per tau"])
            written.append(name)

    for ln in steady:
        qs = list(ln["values"])
        header = [ln.get("coord_name", "s")] + qs
        rows = [[_num(c)] + [_num(ln["values"][q][i]) for q in qs]
                for i, c in enumerate(ln["coord"])]
        name = f"{_safe(ln['name'])}.csv"
        _write(os.path.join(out_dir, name), header, rows,
               head + [ln.get("label", ln["name"])])
        written.append(name)

    for key, block in (summary.get("integrals") or {}).items():
        if not isinstance(block, dict):
            continue
        cols = {k: v for k, v in block.items() if isinstance(v, list)}
        n = {len(v) for v in cols.values()}
        if len(cols) < 2 or len(n) != 1:
            continue
        qs = list(cols)
        rows = [[_num(cols[q][i]) for q in qs] for i in range(n.pop())]
        name = f"timeseries_{_safe(key)}.csv"
        _write(os.path.join(out_dir, name), qs, rows, head + [f"integrals.{key}"])
        written.append(name)

    rows = []
    for ln in lines:
        tau = _num(ln.get("time", {}).get("tau")) if "time" in ln else ""
        for q, vals in ln["values"].items():
            for c, v in zip(ln["coord"], vals):
                rows.append([ln["name"], tau, ln.get("coord_name", "s"), _num(c), q, _num(v)])
    _write(os.path.join(out_dir, "profiles_long.csv"),
           ["line", "tau", "coord_name", "coord", "quantity", "value"], rows, head)
    written.append("profiles_long.csv")
    return written
