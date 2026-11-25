"""Check that boundary conditions set in a case (case.json "bcs", the app's boundary editor)
reach the solve.

Gouy-Chapman template (wall potential 25 mV) with the wall condition replaced by the case:
  1. fixed potential 50 mV         -> exact Gouy-Chapman profile for 50 mV
  2. Stern layer C = 0.5, ref 50 mV -> exact profile for the Stern-reduced wall potential
Both differ from the template's own 25 mV profile, so passing means the override was used.

    python tools/check_case_bcs.py      (CPU, about a minute; work folder in a temp dir)
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from pnpflow import project, runner                            # noqa: E402
from pnpflow.cases import CASES, _stern_phi0, thermal_potential  # noqa: E402

TOL = 1e-4
CHECKS = [
    ("fixed 50 mV", {"type": "fixed", "value": 50.0, "unit": "mV"}, 0.0),
    ("Stern C = 0.5, ref 50 mV", {"type": "stern", "coef": 0.5, "ref": 50.0, "unit": "mV"}, 0.5),
]


def main():
    params = dict(CASES["gouy_chapman"]["params"])
    d, T = params["delta"], params["T"]
    phi_ref = thermal_potential(50.0, T)
    work = tempfile.mkdtemp(prefix="check_case_bcs_")
    ok = True
    for i, (label, spec, stern_C) in enumerate(CHECKS):
        case_dir = os.path.join(work, f"gc_{i}")
        case = project.default_case("gouy_chapman", params, f"gc_{i}")
        case["bcs"] = {"left": {"phi": spec}}
        project.save(case_dir, case)
        summary = runner.run_from_case(case_dir, run_name=f"gc_{i}", plots=False, log=lambda *a: None)
        folder = os.path.join(case_dir, "results", f"gc_{i}")
        with open(os.path.join(folder, "profiles.json")) as f:
            line = json.load(f)["lines"][0]
        x = np.asarray(line["coord"])
        phi = np.asarray(line["values"]["phi"], float)
        phi0 = _stern_phi0(phi_ref, d, stern_C)
        exact = 4 * np.arctanh(np.tanh(phi0 / 4) * np.exp(-x / d))
        err = float(np.nanmax(np.abs(phi - exact)))
        wall = summary["model"]["boundaries"]["left"]["phi"]
        template = summary["template_boundaries"]["left"]["phi"]
        passed = summary["solver"]["converged"] and err < TOL and wall != template
        ok &= passed
        print(f"{'PASS' if passed else 'FAIL'}  {label}: max |phi - exact| = {err:.2e} (limit {TOL:g}); "
              f"wall {template} -> {wall}")
    print(f"work folder: {work}")
    print("ALL PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
