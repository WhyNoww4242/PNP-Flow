# Pore-charging DNS: the user's answers (2026-11-22)

Goal: reproduce the paper's DNS of pore charging (time-dependent nonlinear PNP in an
axisymmetric pore fed by a cylindrical reservoir, "SDL") with pnpflow, starting with one
test geometry, and report axial and radial profiles.

| question | answer |
|---|---|
| scalings | rho = (c+ - c-)/c0, S = (c+ + c-)/c0, Phi = e phi / k_B T, tau = D t / l_p^2, R = r / a_p, Z = z / l_p, alpha(Z) = alpha*/a_p, kappa = a_p / lambda_D (paper Eq. 4) |
| mu | mu = rho + 2 Phi (electrochemical potential of charge, paper Eq. 7) |
| wall | Phi_w applied directly (no Stern layer); 10 mV -> Phi_w = 0.389 |
| SDL walls | uncharged; no extra condition needed (code: no ion flux, dphi/dn = 0) |
| pore end | blocking |
| SDL far end | phi = 0, c = c0 at z = -(l_p/2 + 3 max(alpha)/2) |
| Biot number | not needed. The extra 3 max(alpha)/2 of SDL length is room for meshing the access region; the pore entrance needs extra mesh refinement |
| SDL radius | 2 max(alpha): 20 nm for the narrow pore, 40 nm otherwise (OpenFOAM set-up) |
| physical values | eps = 7.1e-10 F/m, D = 1.34e-9 m^2/s, c0 = 0.94 mM (lambda_D = 10.0 nm), radii 10 nm (narrow) / 20 nm (wide), a_p = 10 nm |
| times | early tau = 0.13; the "intermediate" time is relative to the nondimensional time; find a sensible one |
| comparison data | none yet: show the results, compare later |
| where to run | from the app server (`app/server.py`), with a choice of GPU or CPU |
| first case | one test geometry (narrow pore alpha = 1 chosen: straight, has an exact equilibrium check) |

## Interpretations (to confirm)

- Pore length: the DNS section of the paper says the pores are 1 um long (a_p / l_p = 0.01);
  the OpenFOAM methods paragraph says 10 um. Default `L_nm = 1000`; set `L_nm=10000` for the other.
- Pore end potential: "blocking" is applied to the ions. The potential on the end cap is taken
  as the electrode potential Phi_w (`end_charged = 1`); `end_charged = 0` makes it uncharged.
- No arc at the mouth (the OpenFOAM mesh had a smoothing arc); the sharp corner is refined
  instead (`h_entr`).
- Axial mu(Z) is the cross-section average (weight r dr); the line on the axis is also saved
  (`mu_axis`).
- Wall potential is a step at t = 0 from a uniform reservoir state.
