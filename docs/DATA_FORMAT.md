# pnpflow result format (contract between solver and HUD)

Every run writes one folder: `results/<run_name>/`.
All JSON is plain UTF-8. Numbers are floats. Missing items may be absent — the HUD must tolerate that.

## progress.json  (rewritten during the run)
```json
{"state": "meshing|assembling|solving|post|done|failed",
 "stage": 1, "n_stages": 3, "iteration": 4,
 "res_norm": 1.2e-6, "message": "Newton it 4", "time_s": 12.3,
 "fraction": 0.42, "health": "converging",
 "tells": {"res_norm": 1.2e-6, "ratio": 0.03, "alpha": 1.0, "min_c": 0.37, "halvings": 0},
 "history": [{"stage": 1, "it": 0, "res_norm": 0.3, "step_norm": 1.0, "alpha": 1.0}]}
```
`stage` counts Newton solves (continuation attempts, including halved ones); `n_stages` is
the number of continuation targets.

Live-status keys (optional; `null` until the first Newton iteration):

| key | meaning |
|---|---|
| `fraction` | 0–1 progress through the continuation targets: `(target + done + (trial − done) · p) / n_targets`, where `done`/`trial` are the parameter fractions of the current target already reached / being attempted, and `p = log(r0/|R|) / log(r0/tol)` clipped to [0, 1] is the log-residual progress of the current Newton solve. `null` if the solver gives no status. `1.0` at `state = done` (unless failed). It can step back when a continuation step is halved. |
| `health` | `converging`; `struggling` (last line-search `alpha < 0.25`, a failed Newton attempt, or ≥ 1 step halving in the current target); `stalled` (`|R_k| > 0.9 |R_{k−3}|`: no real decrease over 3 iterations); `failed` (continuation or Newton gave up, or the run raised); `done` (finished, not failed). Precedence: failed > stalled > struggling > converging. |
| `tells.res_norm` | last residual norm `|R_k|` |
| `tells.ratio` | contraction `|R_k| / |R_{k−1}|` (`null` at iteration 0) |
| `tells.alpha` | last line-search step length (`null` at iteration 0) |
| `tells.min_c` | minimum concentration over all species nodes (`null` with no species) |
| `tells.halvings` | continuation step halvings so far in the current target |

Time-dependent runs (optional keys, only when the case sets `Setup.transient`):

| key | meaning |
|---|---|
| `fraction` | simulated time `t / t_end` (not wall time; steps grow as the run goes) |
| `health` | `converging` after an accepted step, `struggling` after a rejected one, `failed` |
| `time` | `{t, t_end, tau, tau_end, dt, step, rejects}`: solver time, the case's time unit (`tau = t / t_per_tau`), last step, accepted steps, rejected steps |
| `tells.dt`, `tells.newton_its`, `tells.err`, `tells.rejects` | last step size, its Newton iterations, its change-vs-extrapolation estimate, total rejections |
| `history` | Newton iterations of every step (`stage` = step number) |

## summary.json
```json
{
 "run_name": "cyl_demo", "case": "cylinder", "created": "2026-11-15T20:00:00",
 "model": {"species": [{"name": "c+", "z": 1, "d": 1.0}, {"name": "c-", "z": -1, "d": 1.0}],
           "flow": "navier_stokes", "advection": "conservative", "supg": false},
 "params": {"delta": 0.1, "Pe": 1.0, "Re": 0.1, "K": 2.0, "phi_wall": 0.973, "U": 1.0},
 "param_info": {"delta": "Debye length / radius", "...": "..."},
 "mesh": {"n_elements": 25000, "n_nodes": 50000, "n_dofs": 260000,
          "h_min": 0.016, "h_max": 4.0, "quality_min": 0.4,
          "quality_hist": {"edges": [0, 0.1, "..."], "counts": [0, 3, "..."]}},
 "solver": {"converged": true, "time_s": 40.2, "n_newton_total": 12,
            "stages": [{"params": {"phi_wall": 0.5}, "converged": true,
                        "iterations": [{"it": 0, "res_norm": 0.3, "step_norm": 1.0, "alpha": 1.0}]}]},
 "checks": [{"name": "min concentration", "value": 0.31, "limit": 0.0,
             "status": "pass", "note": "must stay > 0"}],
 "integrals": {
    "flux": {"c+": {"wall": 1e-12, "inlet": -0.3, "outlet": 0.3, "side": 0.0}},
    "force": {"hydro": [0.0, -12.1], "electric": [0.0, -0.4], "total": [0.0, -12.5],
              "drag": 12.5, "lift": 0.0},
    "charge": {"diffuse": -0.31, "wall_induced": 0.31, "sum": 1e-6},
    "other": {"div_u_L2": 1e-6}},
 "sensitivities": {"drag": {"K": 0.12, "delta": -3.0}},
 "notes": ["free text lines shown to the user"]
}
```
`status` is one of `pass`, `warn`, `fail`, `info`.

Post-processing keys (optional, added 2026-11-21):

| key | meaning |
|---|---|
| `model.coord` | `"planar"` or `"axisym"` |
| `figures` | names of the house-style figures in `figures/` and `svg/` (no extension) |
| `mesh_views` | `[{"file": "mesh/view_a.png", "title": "...", "what_to_check": "..."}, ...]` — always two views |
| `mesh_checks` | the full mesh diagnose list (same check shape, plus `fix`) |

`checks` gains one roll-up entry `"mesh diagnose"` (`value` = number of failures, `status` =
worst status, `"group": "mesh"`). A broken post-processing step adds a `warn` check with
`"group": "postprocess"`; it never fails a converged run. Mesh views and diagnose are written
before the solve, so failed runs have them too.

## fields.json  (nodal data on a linear triangulation)
```json
{"nodes": [[y, z], ...],
 "triangles": [[i, j, k], ...],
 "boundary_edges": {"wall": [[i, j], ...], "inlet": [[i, j], ...]},
 "fields": {"c+": [...], "c-": [...], "phi": [...], "rho": [...],
            "uy": [...], "uz": [...], "|u|": [...], "p": [...],
            "Ey": [...], "Ez": [...], "|E|": [...], "vorticity": [...]},
 "vectors": {"u": ["uy", "uz"], "E": ["Ey", "Ez"], "N_c+": ["N_c+_y", "N_c+_z"]},
 "labels": {"c+": "cation concentration c+/c0", "phi": "potential phi/V_T"},
 "bbox": [ymin, ymax, zmin, zmax],
 "focus_bbox": [-3, 3, -3, 3]}
```
One value per node. Fields are optional: show whatever exists. `focus_bbox` = region of interest (default zoom).

Optional coordinate keys (written by every run since 2026-11-21; absent in older runs = planar):

| key | planar | axisymmetric |
|---|---|---|
| `symmetry` | `"planar"` | `"axisym"` |
| `coord_names` | `["y", "z"]` | `["r", "z"]` |

For `"axisym"` the first node coordinate is the radius `r ≥ 0` and the axis is `r = 0`; mirror
the mesh and fields about `r = 0` to draw the whole body (scalar fields are even in `r`; the
radial vector component `Ey` = `E_r` changes sign under the mirror, `Ez` does not). Vector
names stay `Ey`/`Ez`, `uy`/`uz`; their labels say r and z. In summary.json, integrals of an
axisymmetric run are per radian (`r dr dz`, `r ds`) unless the key ends in `_3d`
(= 2π × per radian).

Fields of the `charged_sphere` case: `phi`, `Ey`(=E_r), `Ez`, `|E|`, `rho` (zero, no ions),
`rho_f` (fixed charge density, the Poisson source, element-averaged at nodes), `phi_exact`,
`phi_err` (= phi − phi_exact).

## profiles.json  (1D cuts)
```json
{"lines": [
   {"name": "radial_0", "label": "radial cut, theta = 0 deg (upstream)",
    "coord_name": "r", "coord": [1.0, 1.01, ...],
    "values": {"c+": [...], "c-": [...], "phi": [...], "rho": [...], "|u|": [...]}},
   {"name": "wall", "label": "on cylinder wall", "coord_name": "theta_deg",
    "coord": [0, 2, ...],
    "values": {"phi": [...], "c+": [...], "sigma": [...], "shear": [...], "p": [...]}}]}
```

Optional per line: `"time": {"tau": 0.13, "t_solver": 1300.0, "t_s": 9.7e-5}` (time-dependent
runs write one line per output time, e.g. `axial_tau_0.13`, `radial_Z0.5_tau_0.13`).

Time-dependent runs also add to `summary.json` (optional): `transient` =
`{scheme, t_end, t_reached, t_per_tau, n_steps, steps: [{t, dt, newton_its, err, rejects,
res_final}], output_times}`, `solver.equilibrium` = `{converged, newton_its}`,
`integrals.charging` (pore case: `tau`, `mu_pore_avg`, `mu_Z05`, `Q_pore_per_rad`), and
`fields_time.json` = `{t, tau, t_per_tau, n_nodes, fields: {c+, c-, rho, phi: [one array per output time]}}` (one value per P2 node like `fields.json`; drives the app's time slider; `tools/export_fields_time.py` writes it for older runs). `solution.npz` holds `t` and `U_t` (snapshot per output time). Every run records `device`
= `{platform, devices}` (the JAX backend it ran on); `progress.json` carries the same `device` as soon as the run starts (the app shows a green/red dot from it).

## csv/*.csv  (every run, from profiles.json + summary.json)
Written at the end of each run by `pnpflow/export.py`, or later with
`python tools/export_csv.py results/<run>`. `#` comment lines first (run, case, units), then
one header row. Time-dependent lines: `<group>__<quantity>.csv` with the coordinate and one
column per tau (e.g. `axial__mu.csv`, `radial_Z0.5__phi.csv`); steady lines: `<line>.csv`;
list-valued `integrals` blocks: `timeseries_<name>.csv`; everything in long form:
`profiles_long.csv` (`line, tau, coord_name, coord, quantity, value`). NaN = empty cell.

## plots/*.png
Static matplotlib images (mesh, mesh zoom, field maps, profiles). The HUD can show them as a gallery.

## mesh/  (every run, written right after meshing)
- `view_a.png`: whole domain, boundaries coloured by physical group (unnamed edges as red dots).
- `view_b.png`: zoom on the finest boundary, elements coloured by shape quality.
- `diagnose.json`: `{"checks": [...]}` — unnamed/duplicate boundaries, BC coverage, quality,
  size jumps, inverted P2 elements, connectivity, EDL resolution, far field, size budget.
  Each check has a `fix`.

## figures/*.png and svg/*.svg  (house style, `AGENTS.md` §9)
300 dpi PNG and SVG pairs: `field_<name>`, `profile_<line>`, `convergence`. Axisymmetric fields
are mirrored about r = 0. Skipped with `--no-plots`.

## solution.msh / solution.vtu
For gmsh / ParaView. Not read by the HUD.

## Launching runs
`python run.py <case> --name <run_name> --set key=value --set key2=value2`
(`JAX_PLATFORMS=cuda|cpu` picks the device; the servers' `/api/launch` takes
`device: auto|gpu|cpu` and sets it.)
`python run.py --list` prints the cases and their default params as JSON:
`{"cylinder": {"params": {...}, "info": {...}}, ...}`
