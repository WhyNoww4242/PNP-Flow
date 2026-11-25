#!/usr/bin/env python
"""Command line entry point.

  python run.py --list
  python run.py cylinder --name demo --set delta=0.1 --set K=50
  python run.py --case cases/my_pore --name run1      # run a case (cases/<case>/case.json)
  python run.py --case cases/my_pore --mesh-only      # build + validate its mesh only
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def use_cpus():
    """Run on all logical CPUs but one (PNPFLOW_THREADS=n overrides), so the desktop stays
    responsive. Must run before jax or MKL load. The affinity caps XLA's thread pool;
    MKL_DYNAMIC=FALSE stops MKL dropping to the physical-core count (8 of 16 here), which
    made PARDISO 15 % faster at 632k dofs. Returns (used, total)."""
    total = os.cpu_count() or 1
    want = int(os.environ.get("PNPFLOW_THREADS") or max(1, total - 1))
    if hasattr(os, "sched_setaffinity"):            # Linux; macOS only gets the thread counts
        cpus = sorted(os.sched_getaffinity(0))
        if want < len(cpus):
            os.sched_setaffinity(0, cpus[-want:])   # leave CPU 0 (interrupts) free
        want = len(os.sched_getaffinity(0))
    for var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(want))
    os.environ.setdefault("MKL_DYNAMIC", "FALSE")
    return want, total


def main():
    used, total = use_cpus()
    ap = argparse.ArgumentParser(description="steady PNP + flow solver")
    ap.add_argument("case", nargs="?", help="case name (see --list)")
    ap.add_argument("--name", help="run folder name")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--list", action="store_true", help="print cases as JSON")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    ap.add_argument("--case", dest="case_dir", metavar="DIR",
                    help="case folder with case.json (results go to DIR/results/<name>)")
    ap.add_argument("--mesh-only", action="store_true",
                    help="with --case: build, validate and render the mesh, no solve")
    args = ap.parse_args()

    from pnpflow.runner import list_cases, run_case, run_from_case, build_case_mesh
    log = lambda *a: print(*a, flush=True)
    if args.case_dir or (args.case and not args.list):     # --list must print JSON only
        log(f"cpu: {used} of {total} logical CPUs")
    if args.case_dir:
        if args.mesh_only:
            st = build_case_mesh(args.case_dir, log=log)
            print("mesh:", st["stats"]["n_elements"], "triangles;", "validation in",
                  os.path.join(args.case_dir, "mesh", "validation.json"))
            return
        summary = run_from_case(args.case_dir, args.name, plots=not args.no_plots, log=log)
        print("converged:", summary["solver"]["converged"])
        return
    if args.list or not args.case:
        print(json.dumps(list_cases(), indent=1))
        return
    overrides = {}
    for item in args.set:
        key, value = item.split("=", 1)
        overrides[key.strip()] = float(value)
    summary = run_case(args.case, overrides, args.name, args.results,
                       plots=not args.no_plots, log=lambda *a: print(*a, flush=True))
    print("converged:", summary["solver"]["converged"])
    for c in summary["checks"]:
        print(f"  [{c['status']:4s}] {c['name']}: {c['value']:.3e}  ({c['note']})")


if __name__ == "__main__":
    main()
