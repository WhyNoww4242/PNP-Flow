#!/usr/bin/env python
"""Create the High_Voltage_Filipe cases: Fig. 8 of Henrique, Zuk & Gupta, Soft Matter 18,
198 (2022), "Validity of analytical results" -- pore charging at high wall potentials.

Set-up (docs/high_voltage_filipe.md):
  a_p / lambda_D = 2   : straight pore alpha = 2 with a_ref = 10 nm (radius 20 nm) and the
                         DNS-comparison constants (lambda_D = 9.97 nm, a_p / lambda_D = 2.005)
  Bi = 8               : SDL radius a_s = 2 a_p (4 a_ref), SDL length l_s = l_p / 2 exactly
                         (Bi = (a_s/a_p)^2 l_p/l_s), so Z runs from -0.5 as in the figure
  l_p = 1 um           : L = l_p / a_ref = 100, a_p / l_p = 0.02
  Phi_w = 10, 50, 100, 200 mV -> Psi_D = 0.389, 1.946, 3.892, 7.784 (paper: 0.4 1.9 3.9 7.8)
  pore end             : blocking, phi_end = phi_w (electrode end)
  times                : profiles at tau = 0.07, 0.15 (panels a-d); mouth flux Jright at every
                         accepted step to tau = 0.5 (panel e)

Wall mesh: at high Phi_w the counter-ion layer at the wall is much thinner than lambda_D.
The c- e-folding length at the wall is l = delta / (2 sinh(Phi_w / 2)) (Gouy-Chapman field),
so the pore wall and end get a refinement of size min(0.1, l / 4) growing by 1.2, and the
mouth corner a disk of half that size.

    python tools/make_high_voltage_cases.py                 # all four
    python tools/make_high_voltage_cases.py 200 --force     # one, overwrite case.json
Then per case (check the mesh size first):
    JAX_PLATFORMS=cpu python run.py --case cases/High_Voltage_Filipe_200mV --mesh-only
    JAX_PLATFORMS=cuda python run.py --case cases/High_Voltage_Filipe_200mV --name hv200_run1
and the figure + side-by-side page:
    python tools/figure_high_voltage.py
"""
import argparse
import importlib.util
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

spec = importlib.util.spec_from_file_location("project", os.path.join(ROOT, "pnpflow", "project.py"))
project = importlib.util.module_from_spec(spec)
spec.loader.exec_module(project)

NAME = "High_Voltage_Filipe"
POTENTIALS_MV = (10, 50, 100, 200)
K_B, E_CHARGE, N_A = 1.380649e-23, 1.602176634e-19, 6.02214076e23

L_NM, A_REF_NM, ALPHA = 1000.0, 10.0, 2.0
COMMON = {
    "alpha_in": ALPHA, "alpha_out": ALPHA, "L_nm": L_NM, "a_ref_nm": A_REF_NM,
    "c0_mM": 0.948, "eps": 7.0832e-10, "T": 298.15, "D": 1.337e-9,
    "sdl_radius": 2 * ALPHA,                     # a_s / a_p = 2
    "sdl_length": 0.5 * L_NM / A_REF_NM,         # l_s / l_p = 0.5  -> Bi = 8
    "tau_end": 0.5, "tau_mid": 0.15,
    "dns_times": 0.0, "dns_init": 0.0, "equilibrium_check": 1.0,
    "rtol": 2e-3, "err_mixed": 1.0,
}
SAVE_TAUS = [0.01, 0.03, 0.07, 0.15, 0.3, 0.5]  # 0.13 is always added by the template


def case_name(mv):
    return f"{NAME}_{mv}mV"


def wall_size(mv, delta):
    """Element size at the pore wall for wall potential mv (a_ref units)."""
    psi = mv * 1e-3 * E_CHARGE / (K_B * COMMON["T"])
    ell = delta / (2 * math.sinh(psi / 2))      # c- e-folding length at the wall
    return min(0.1, ell / 4), psi, ell


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("potentials", nargs="*", type=int, default=list(POTENTIALS_MV),
                    help="wall potentials in mV (default 10 50 100 200)")
    ap.add_argument("--out", default=os.path.join(ROOT, "cases"), help="cases folder")
    ap.add_argument("--force", action="store_true", help="overwrite an existing case.json")
    args = ap.parse_args()
    lam = math.sqrt(COMMON["eps"] * K_B * COMMON["T"]
                    / (2 * E_CHARGE ** 2 * COMMON["c0_mM"] * N_A))
    delta = lam / (A_REF_NM * 1e-9)
    print(f"lambda_D = {lam * 1e9:.4g} nm, a_p/lambda_D = {ALPHA / delta:.4g}, "
          f"Bi = {(COMMON['sdl_radius'] / ALPHA) ** 2 * (L_NM / A_REF_NM) / COMMON['sdl_length']:.4g}")
    for mv in args.potentials:
        name = case_name(mv)
        folder = os.path.join(args.out, name)
        if os.path.isfile(os.path.join(folder, "case.json")) and not args.force:
            print(f"exists, kept: {folder} (use --force to overwrite)")
            continue
        h, psi, ell = wall_size(mv, delta)
        params = dict(COMMON, phi_w_mV=float(mv), phi_end_mV=float(mv),
                      dt0=0.01 if mv <= 50 else 1e-3)
        case = project.default_case("pore_charging", params, name)
        case["description"] = (f"Fig. 8 of Henrique, Zuk & Gupta 2022: wall {mv} mV "
                               f"(Psi_D = {psi:.3f}), a_p/lambda_D = 2, Bi = 8, l_p = 1 um")
        case["time"]["save"]["mode"] = "list"
        case["time"]["save"]["list"] = list(SAVE_TAUS)
        if h < 0.1:                              # the template's own h_wall is 0.1 here
            case["mesh"]["boundaries"] = [
                {"boundary": "pore_wall", "size": round(h, 5), "layer": 0.0, "growth": 1.2},
                {"boundary": "pore_end", "size": round(h, 5), "layer": 0.0, "growth": 1.2},
            ]
            case["mesh"]["regions"] = [
                {"kind": "disk", "x": ALPHA, "y": 0.0, "r": max(4 * h, 0.02),
                 "size": round(h / 2, 5), "growth": 1.2},
            ]
        project.save(folder, case)
        print(f"created {folder}/case.json  (Psi_D = {psi:.3f}, wall e-fold {ell:.3g}, "
              f"wall size {h:.3g})")


if __name__ == "__main__":
    main()
