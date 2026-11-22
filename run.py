#!/usr/bin/env python
"""Command line entry point.

  python run.py --list
  python run.py cylinder --name demo --set delta=0.1 --set K=50
"""
import argparse
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)


def main():
    ap = argparse.ArgumentParser(description="steady PNP + flow solver")
    ap.add_argument("case", nargs="?", help="case name (see --list)")
    ap.add_argument("--name", help="run folder name")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    ap.add_argument("--list", action="store_true", help="print cases as JSON")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--results", default=os.path.join(ROOT, "results"))
    args = ap.parse_args()

    from pnpflow.runner import list_cases, run_case
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
