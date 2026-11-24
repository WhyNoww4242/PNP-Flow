# High_Voltage_Filipe: Fig. 8 of Henrique, Zuk & Gupta (2022)

Paper: F. Henrique, P. J. Zuk, A. Gupta, "Charging dynamics of electrical double layers inside a
cylindrical pore: predicting the effects of arbitrary pore size", Soft Matter 18, 198 (2022),
doi:10.1039/d1sm01239h (CC BY 4.0). Fig. 8 compares the first-order perturbation model with
OpenFOAM DNS at wall potentials up to 200 mV. These cases redo the DNS with pnpflow.
The figure is in `docs/high_voltage_filipe/paper_fig8.png`.

## The user's answers (2026-11-23)

| question | answer |
|---|---|
| pore length | 1 µm (`L_nm = 1000`), a_p / l_p = 0.02 |
| a_p / lambda_D = 2 | straight wide pore alpha = 2 with a_ref = 10 nm (radius 20 nm), DNS-comparison constants |
| layout | four case folders `cases/High_Voltage_Filipe_{10,50,100,200}mV`, one figure script |
| overlays | pnpflow only (no perturbation model, no OpenFOAM data); an HTML page with the paper figure and the pnpflow figure side by side, with the quantities |

## Set-up (from the paper)

| item | paper | pnpflow |
|---|---|---|
| a_p / lambda_D | 2 | 2.005 (c0 = 0.948 mM, eps = 7.0832e-10 F/m, T = 298.15 K, lambda_D = 9.97 nm) |
| Bi = (a_s/a_p)^2 (l_p/l_s) (D_s/D_p) | 8 (Fig. 4, 8) | `sdl_radius = 4` (= 2 a_p), `sdl_length = 50` (= l_p / 2) |
| Psi_D | 0.4, 1.9, 3.9, 7.8 (10, 50, 100, 200 mV) | `phi_w_mV` = 10, 50, 100, 200 -> 0.389, 1.946, 3.892, 7.784 |
| pore end | zero gradients (long-pore limit) | blocking, `phi_end_mV = phi_w_mV` (end EDL within ~a_p of Z = 1) |
| times | profiles at tau = 0.07, 0.15; J_right to tau = 0.5 | save list 0.01 0.03 0.07 0.15 0.3 0.5 (+0.13); `tau_end = 0.5`, `tau_mid = 0.15` |
| Psi_m, rho_m | centreline (Eq. 10d, 10e) | profiles.json `phi_axis`, `rho_axis` (new) |
| J_right | J_Z at Z = 0+ | `summary.json integrals.mouth.J_right` = 2/(L alpha^2) dQ/dtau, Q = pore charge at every accepted step (new) |

## Mesh and time stepping at high potential

The counter-ion layer at the wall thins with the potential: from the Gouy–Chapman field,
c- e-folds over l = lambda_D / (2 sinh(Psi_D / 2)): 2.55, 0.44, 0.146, 0.020 a_ref for 10–200 mV.
`tools/make_high_voltage_cases.py` adds a case mesh refinement of size min(0.1, l/4) on
`pore_wall` and `pore_end` (growth 1.2) and a disk at the mouth corner (half that size) when
l/4 < 0.1. Mesh sizes (`--mesh-only`, density 1):

| case | triangles | dofs |
|---|---|---|
| 10 mV | 55 330 | 593 632 |
| 100 mV | 68 648 | 740 610 |
| 200 mV | 182 542 | 1 973 864 |

`err_mixed = 1` (new template option): the BDF2 step control measures |U - U_pred| / (1 + |U|),
because c- reaches e^7.8 ≈ 2400 c0 at the wall at 200 mV and the absolute measure would force
tiny steps. `dt0 = 1e-3` for 100 and 200 mV.

## Commands

```bash
python tools/make_high_voltage_cases.py                       # the four case.json
JAX_PLATFORMS=cpu python run.py --case cases/High_Voltage_Filipe_200mV --mesh-only
nohup env JAX_PLATFORMS=cuda python -u run.py --case cases/High_Voltage_Filipe_10mV --name hv10_run1 \
    > results/hv10_run1.log 2>&1 &                            # then 50, 100, 200
python tools/figure_high_voltage.py                           # figure + CSV + HTML
```

Outputs: `results/high_voltage_filipe/figures/figure8_pnpflow.png` (+ `svg/`, one PNG per panel),
`csv/figure8_profiles.csv`, `csv/figure8_flux.csv`, and
`results/high_voltage_filipe/High_Voltage_Filipe_comparison.html` (self-contained: paper figure
and pnpflow figure side by side, panel by panel with values at fixed Z and tau, run table).
