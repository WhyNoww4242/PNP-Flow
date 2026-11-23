# Pore DNS comparison runs: the user's answers (2026-11-22)

Goal: run the four DNS geometries (narrow, converging, diverging, wide) one at a time on this
PC, matched to Pawel's DNS (`C:\Users\Gnome\Documents\Pore Charging Comparison\transportFolder`,
read by `pawel.py`), and compare in a figureDNS4-style plot. Supersedes the length and
constants in `docs/pore_answers.md` for these runs.

| question | answer |
|---|---|
| geometries | Narrow 100_100 (alpha 1 -> 1), Converging 200_100 (2 -> 1), Diverging 100_200 (1 -> 2), Wide 200_200 (2 -> 2); names from `comparison_results/scripts/compare.py` |
| pore length | **10 um** (`L_nm=10000`, a_ref = 10 nm): follows the DNS files and pawel.py (tau = 13.37 t[s] only holds for 10 um). If the schematic (1 um) is right, the pawel.py time conversion is wrong too |
| constants | `c0_mM=0.948 eps=7.0832e-10 T=298.15 D=1.337e-9 phi_w_mV=10`; pnpflow keeps CODATA constants (V_T 25.693 mV, lambda_D 9.9740 nm) vs pawel.py (25.680 mV, 9.9716 nm): report both, do not change |
| SDL | `sdl_radius=2 sdl_length=501.5` (pnpflow defaults; equal to the DNS 5.015 um) |
| end | `end_charged=1`, `tau_end=1.01` |
| output times | tau_k = 0.06685 k, k = 0..15 (t = 0, 0.005, ..., 0.075 s, the DNS folder times); these are the saved times in `solution.npz` (`U_t`) |
| initial state | DNS state (phi = phi_w in the pore, c = c0) via a **new option**; the default stays as it is |
| mesh | dry run first (build + diagnose all four meshes, estimate memory) before any solve; if a mesh does not fit, a slightly coarser `h_wall` is allowed |
| launch | queue script, one case at a time, GPU; the agent monitors and reports |
| after runs | comparison figures (figureDNS4 layout, the user's code) + one combined CSV |
| mu line | centreline (`mu_axis`), to match the DNS `lineYe` file |
| report only | DNS mouth is rounded, pnpflow's is a sharp corner; DNS pore-end line has non-uniform phi (0.376-0.389), pnpflow's end face is phi = phi_w; the Biot number has no pnpflow equivalent |
