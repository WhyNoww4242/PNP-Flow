#!/usr/bin/env python
"""Create the pore DNS comparison runs as cases (cases/<name>/case.json).

Same settings as tools/run_pore_dns.sh (docs/pore_dns_answers.md): L = 10 µm, a_ref = 10 nm,
DNS constants, DNS start and output times. One case per pore shape, optionally on the coarse
mesh (h_wall = 0.15) that fits in memory.

    python tools/make_dns_cases.py                      # narrow converging diverging wide
    python tools/make_dns_cases.py --coarse converging  # cases/pore_dns_converging_coarse
    python tools/make_dns_cases.py --out /tmp/cases     # another cases folder

Then, per case:
    JAX_PLATFORMS=cpu python run.py --case cases/pore_dns_narrow --mesh-only   # check size first
    JAX_PLATFORMS=cuda python run.py --case cases/pore_dns_narrow --name run1  # solve
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import importlib.util  # noqa: E402  (project.py only: no jax import needed here)

spec = importlib.util.spec_from_file_location("project", os.path.join(ROOT, "pnpflow", "project.py"))
project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project)

SHAPES = {"narrow": (1, 1), "converging": (2, 1), "diverging": (1, 2), "wide": (2, 2)}
COMMON = {"L_nm": 10000.0, "a_ref_nm": 10.0, "phi_w_mV": 10.0, "c0_mM": 0.948,
          "eps": 7.0832e-10, "T": 298.15, "D": 1.337e-9, "phi_end_mV": 10.0,
          "sdl_radius": 0.0, "sdl_length": 0.0, "tau_end": 1.01, "dns_times": 1.0,
          "dns_init": 1.0}
COARSE_H = 0.15


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("shapes", nargs="*", default=list(SHAPES), choices=list(SHAPES))
    ap.add_argument("--coarse", action="store_true", help=f"h_wall = {COARSE_H} (fits in memory)")
    ap.add_argument("--out", default=os.path.join(ROOT, "cases"), help="cases folder")
    ap.add_argument("--force", action="store_true", help="overwrite an existing case.json")
    args = ap.parse_args()
    for shape in args.shapes:
        a_in, a_out = SHAPES[shape]
        name = f"pore_dns_{shape}" + ("_coarse" if args.coarse else "")
        folder = os.path.join(args.out, name)
        if os.path.isfile(os.path.join(folder, "case.json")) and not args.force:
            print(f"exists, kept: {folder} (use --force to overwrite)")
            continue
        params = dict(COMMON, alpha_in=float(a_in), alpha_out=float(a_out))
        if args.coarse:
            params["h_wall"] = COARSE_H
        case = project.default_case("pore_charging", params, name)
        case["description"] = (f"DNS comparison, {shape} pore (alpha {a_in} -> {a_out}), "
                               f"L = 10 µm{', coarse mesh' if args.coarse else ''}")
        case["time"]["save"]["mode"] = "template"   # dns_times = 1: the DNS folder times
        project.save(folder, case)
        print(f"created {folder}/case.json")


if __name__ == "__main__":
    main()
