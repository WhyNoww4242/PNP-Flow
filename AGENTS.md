# AGENTS.md — pnpflow

Guide for AI coding agents (and new humans) working in this repo. Read this first, then:
`docs/THEORY.md` (equations), `docs/DATA_FORMAT.md` (solver ↔ GUI files), `docs/MCP.md`
(agent protocol and gmsh instructions), `PRODUCT.md` / `DESIGN.md` (product and visual
design), `docs/ui_answers.md` (the user's design decisions).

## 1. Goal

pnpflow is a finite-element tool for **electrochemical systems**: Poisson–Nernst–Planck (PNP)
ion transport, optionally coupled to fluid flow (Stokes / Navier–Stokes). It lets
electrochemists model *their own* cell, electrode, channel or pore without writing FEM code.

Goals:
1. **Physics with exact derivatives.** Every Jacobian, linearisation and sensitivity comes
   from JAX autodiff (`jacfwd`, `jvp`, `grad`). No hand-written derivatives.
2. **Model building without code.** A desktop GUI (Rust, Tauri 2) with a model tree and
   step-by-step construction: Geometry → Mesh → Materials → Boundary Conditions → Governing
   Equations → Study → Results. Lab (SI) units or dimensionless, with a toggle.
3. **Live status.** Every run reports its state while it runs: progress bars per stage, a live
   residual plot, and a health verdict with the numbers behind it (`docs/MCP.md` §6).
4. **Automatic graphs.** Every finished run writes its standard figures in house style
   (§9) without being asked: fields, EDL profiles, fluxes, convergence, swept curves.
5. **Agent-operable.** An MCP server exposes the same operations as the GUI. Geometry goes
   through gmsh; meshes are always rendered from two views and diagnosed before solving
   (`docs/MCP.md`).
6. **Trustworthy results.** Every run reports checks (balances, positivity, convergence).

| part | language | status |
|---|---|---|
| Solver (`pnpflow/`) | Python + **JAX**, gmsh, PARDISO | working, validated |
| Desktop app (`app/`) | **Rust + Tauri 2**, web front end; stdlib Python dev server | first version runs in a browser (`python app/server.py`, port 8766; JSON API in `app/api.py`); Tauri shell written, not compiled. The old HUD dashboard (`hud/`) was retired 2026-11-23: its API moved to `app/api.py`, its look and WebGL viewer live on in `app/ui` |
| MCP server (`pnpflow/mcp`) | Python `mcp` SDK | specified (`docs/MCP.md`), not started |
| Time stepping (`pnpflow/transient.py`) | Python + JAX | variable-step BDF2, checked (order 2.1, `tools/check_transient.py`) |
| Cases (`pnpflow/project.py`) | JSON | `cases/<case>/case.json` = template + parameters + mesh options + time settings; mesh built and reused per case; runs in `cases/<case>/results/` (2026-11-22). Generic (template-free) case not started |
| Mesh tools (`pnpflow/meshtools.py`, `geometry.apply_mesh_options`, `app/ui/js/meshtools.js`, `repair.js`) | Python + JS | two-view render + diagnose before every solve; case `mesh` block (density, boundary / box / disk refinement, sharp corners, spray paint) → gmsh size fields; Mesh page with Validation bar and a "Repair mesh" workspace (2026-11-23) |
| Pore DNS comparison (`tools/make_dns_cases.py`, `run_pore_dns.sh`, `figure_dns4.py`) | Python + bash | coarse cases (h_wall 0.15) for converging / diverging / wide; converging run to τ = 0.14 done; full runs to τ ≈ 0.53 planned in `docs/run_dns_tau053.md` (§6) |

## 2. Physics (nondimensional)

Scales: length `a`, speed `U`, concentration `c0`, potential `V_T = k_B T / e`.

```
species i :  dc_i/dt + div N_i = s_i       (dc_i/dt only in time-dependent runs, §3 item 7)
             N_i = c_i u - (d_i/Pe) (grad c_i + z_i c_i grad phi)      Nernst–Planck flux
Poisson   :  -delta^2 lap phi = rho / rho_n + s_phi,   rho = sum z_i c_i
momentum  :  Re (u.grad) u = -grad p + lap u - K rho grad phi + s_u    (optional)
mass      :  div u = 0                                                   (optional)
```
Groups: `Pe = U a / D0`, `Re = rho U a / eta`, `delta = lambda_D / a`,
`K = F c0 V_T a / (eta U)`. `flow` is one of `none | prescribed | stokes | navier_stokes`.
Details, discretisation and BC table: `docs/THEORY.md`.

## 3. How the solver works

1. **Geometry + mesh** (`geometry.py`, `mesh.py`): gmsh (OCC kernel) builds curved P2
   triangles with named physical boundaries and a distance-based size field
   (`h ≈ delta/8` near charged walls so the EDL is resolved). Builders: `cylinder_in_box`,
   `rectangle`, `sphere_axisym`, `pore_axisym`; `load_msh` reads external meshes. Every
   builder runs inside `geometry.mesh_options(opts, load=, save=)`, which applies the case
   `mesh` block and can load / save `case.msh` instead of re-meshing.
2. **Elements** (`fem.py`): P2 for `c_i, phi, u`; P1 for `p` (Taylor–Hood). 7-point triangle
   quadrature (degree 5), 3-point edge quadrature.
3. **Weak form** (`model.py`): `Model.element_residual` is a plain JAX function of one
   element's unknowns; `edge_residual` holds natural BCs (flux, Robin/Stern, surface charge,
   pressure). Conservative or non-conservative advection; optional SUPG.
   Unknown layout: `[c_1..c_S, phi, uy, uz]` on P2 nodes, then `p` on P1 vertices.
4. **Assembly** (`solver.Assembler`): `vmap` over elements, `jit` for the whole residual.
   Element Jacobian = `jax.jacfwd(element_residual)` (exact). Sparsity computed once;
   values scattered with `segment_sum` into CSR. Dirichlet rows → `U_i − g_i`.
5. **Newton** (`solver.Newton`): PARDISO (fallback SuperLU), backtracking that keeps all
   concentrations positive. **Continuation** walks parameters, halving the step on failure.
6. **Analysis** (`analysis.py`): reaction-method boundary fluxes, induced wall charge, forces,
   recovered gradients (`E`, vorticity, `N_i`, current), line/wall profiles, and **adjoint
   sensitivities** `dQ/dp` (one extra solve per quantity; `dR/dp` via `jax.jvp`).
7. **Time-dependent runs** (`transient.py`): `M dU/dt + R(U) = 0`, where `M` is the species mass
   matrix (from `jacfwd` of the element mass term; phi, u, p have none, so it is an index-1
   DAE). Variable-step BDF2 (first step backward Euler) wraps the steady assembler, so
   `Newton` is reused. A template asks for it with `Setup.transient` (t_end, output times,
   profile/figure callbacks); `runner._run_transient` writes profiles at each output time,
   `summary.transient`, `solution.npz` (`t`, `U_t`) and live `progress.time`.
8. **Output** (`output.py`, `figures.py`, `export.py`, `runner.py`): the run folder with
   `progress.json` (live: `fraction`, `health` = converging | struggling | stalled | failed |
   done, `tells`), `summary.json`, `fields.json`, `profiles.json`, `fields_time.json`
   (transient: primary fields at every saved time, drives the app's time slider),
   `figures/*.png` + `svg/*.svg` (house style, §9), `csv/*.csv`, `plots/*.png` (quick plots, app figure gallery),
   `solution.{vtu,msh,npz}`, `mesh/` (copy of the mesh + views + `diagnose.json`) and
   `convergence.txt` (every Newton stage / time step, then the checks;
   `project.convergence_text`). Contract: `docs/DATA_FORMAT.md`.
9. **Cases** (`project.py`, `runner.run_from_case`, `runner.build_case_mesh`): a case is
   `cases/<case>/case.json` (template, `params`, display `units`, `figures`, `mesh` block,
   `time` block). Its mesh lives in `cases/<case>/mesh/` (`case.msh` with named groups,
   `view_a/b.png`, `validation.json`, `mesh.json` for the app, `status.json` with the mesh key).
   A run reuses `case.msh` when the mesh key (template + geometry/mesh/materials parameters +
   mesh block) is unchanged, else rebuilds and stores it back. Runs go to
   `cases/<case>/results/<run>/` with a frozen `case.json` and `mesh/case.msh`. The `mesh` block
   (density, boundary refinement, box/disk regions, sharp corners, paint spots) becomes gmsh size
   fields in `geometry.apply_mesh_options` (`_finish` applies it to every builder); the `time`
   block (saved-time mode, `dt_max`, `max_rejects`) overrides the template's transient settings in
   `runner._run_transient`. Legacy `results/<run>/` runs still open read-only.
   - `mesh.density` sets gmsh `Mesh.MeshSizeFactor = 1/density`: **higher = finer**
     (2 halves every size, ≈ 4× elements). Example: converging DNS pore, density 1.2 →
     241k triangles / 2.63M dofs; 0.95 → 152k / 1.67M. Changing it changes the mesh key, so
     the next run or `--mesh-only` rebuilds `case.msh`.
   - Time save modes (`project.save_times`): `template` (the template's own list, e.g. the
     DNS times with `dns_times = 1`), `uniform` (every `dtau`), `log` (`n` from `tau_min`),
     `list`. τ_end and the template's `required_outputs` (pore: 0.13 and τ_mid) are always
     saved. Many saved times make `fields_time.json` large (≈ 14 MB per snapshot at 241k
     triangles).
   - Run names: `run.py --case` without `--name` uses `<case>_<yymmdd>_<HHMMSS>`; the app uses
     `<case>_<yymmdd>_<HHMM>`. Run names are unique across `results/` and all cases.

### Boundary conditions (`bc.py`)
Values are numbers, parameter names (strings) or JAX functions `f(x, params)`.

| key | options | default if unset |
|---|---|---|
| `c` | `Dirichlet(v or "bulk")`, `Flux(N·n)`, `BLOCKING`, `OUTFLOW` (per species via dict) | blocking |
| `phi` | `Dirichlet`, `Robin(C, ref)` (Stern), `Charge(sigma)` | `dphi/dn = 0` |
| `u` | `NOSLIP`, `Dirichlet((uy, uz))`, `Component(i, v)`, `Pressure(p0)`, `FREE` | no-slip |

### Mesh diagnose (`meshtools.diagnose`, written to `mesh/diagnose.json` / `validation.json`)
| check | limit | status if over |
|---|---|---|
| unnamed boundary edges, duplicate naming, BC names a missing boundary | 0 | fail |
| min quality | 0.3 (fail below 0.2) | warn / fail |
| size jump between neighbours | 2 | warn |
| inverted / bad P2 (min det J / det J_straight at quadrature points + nodes) | > 0 (warn below 0.2) | fail / warn |
| connected regions | 1 | warn |
| EDL resolution: elements across 3δ along wall normals | ≥ 8 | fail |
| far field: bulk boundary distance | ≥ 20 δ | warn |
| size budget | 2e6 dofs | warn |

A diagnose `fail` is reported (Mesh Validation bar, `stdout.log`) but **does not stop a run**.
The coarse DNS meshes (h_wall 0.15) fail EDL resolution on `pore_wall` (min ≈ 6–8): the user
approved this fallback (`docs/pore_dns_plan.md`) so the meshes fit in memory.

### Templates (`cases.py`, `pore.py`) — `build(params) -> Setup`, registered with `@case`

Terminology: a **template** is a registered `@case` function (`run.py <template>`); a **case** is a
folder `cases/<case>/` whose `case.json` names a template plus parameters, mesh and time
settings (`run.py --case`); a **run** is one solve in `results/<run>/` or
`cases/<case>/results/<run>/`.

| template | purpose | validated against |
|---|---|---|
| `cylinder` | charged cylinder in cross flow, EDL resolved | symmetry, Gauss balance, zero wall flux |
| `gouy_chapman` | planar EDL, optional Stern layer | exact GC: max err ~3e-6 |
| `eo_slit` | electro-osmotic slit flow | nonlinear PB: rel. err ~2e-4 |
| `mms` | manufactured solution, all terms + BC types | max err ~2e-4 |
| `charged_sphere` | axisymmetric Poisson | exact potential, Gauss |
| `pore_charging` | **time-dependent** pore charging from an SDL reservoir (axisym, paper DNS; `pnpflow/pore.py`) | equilibrium vs cylinder PB at Z = 0.5; BDF2 vs exact decay |

`pore_charging` options beyond the paper set-up: `phi_end_mV` (closed pore end, default =
wall; replaced `end_charged`), `dns_times` (save at the DNS folder times τ_k = 0.06685 k for
L = 10 µm, k = 1..15, plus τ_end), `dns_init` (start from the DNS field: c = c0, φ = φ_w in the
pore), `equilibrium_check`, `dt0`, `rtol`. `python run.py --list` prints every default with
its help text.

`bash tools/validate.sh` runs `tools/check_transient.py` and the validation cases
gouy_chapman, eo_slit, mms, charged_sphere as runs `val_<case>` (all pass as of 2026-11-15).
`scratch_eo.py` (repo root) is an older stand-alone smoke test of the EO slit.

## 4. Running

```bash
# One conda env for Linux and macOS
conda env create -f environment.yml && conda activate pnpflow
python run.py --list
python run.py cylinder --name demo --set delta=0.05
python app/server.py                       # app at http://localhost:8766 (Study tab: Device = Auto/GPU/CPU)
python run.py --case cases/my_pore --mesh-only   # build + validate a case's mesh (no solve)
python run.py --case cases/my_pore --name run1   # run a case -> cases/my_pore/results/run1/
python tools/check_transient.py            # time-stepper check (BDF2 order 2)

# pore DNS comparison (§6)
python tools/make_dns_cases.py --coarse narrow          # cases/pore_dns_narrow_coarse/case.json
bash tools/run_pore_dns.sh [narrow converging ...]      # template runs -> results/pore_dns_<case>/
python tools/figure_dns4.py --dns DNS_compare/transportFolder   # figureDNS4 + combined CSV

# re-export older runs
python tools/export_csv.py results/<run>               # csv/
python tools/export_fields_time.py results/<run>       # fields_time.json from solution.npz
python tools/export_colormaps.py                       # app/ui/data/colormaps.json (cmasher LUTs)
```

**Failure detection:** `run.py` exits 0 even when the solve fails. A run succeeded only if
`stdout.log` ends with `converged: True` (and, for transient runs, `summary.transient.t_reached`
equals `t_end`). Exit code −15 means the run was stopped (app Stop / SIGTERM); a kill with no
`converged:` line is usually out of memory. Runs started by the app write
`# launched by app <date>`, the command and the device at the top of `stdout.log`.

Device: the server sets `JAX_PLATFORMS` per run (`/api/launch` `device: auto|cpu|gpu`,
`/api/devices` probes the GPU). JAX assembly runs on the GPU; the sparse solve (PARDISO) stays on
the CPU. From the shell: `JAX_PLATFORMS=cuda python run.py ...`.
`run.py --list` prints every template with default params + help text as JSON.
`environment.yml`: gmsh from conda-forge (no system libGLU); `pypardiso` only on x86_64
(Apple Silicon uses SuperLU); Linux installs `jax[cuda13]` (GPU) automatically.

## 5. Repository map

```
run.py               CLI: run a template (--set overrides, --list) or a case (--case DIR [--mesh-only])
environment.yml      conda env (solver, gmsh, cmasher, mcp)

pnpflow/__init__.py  turns on jax_enable_x64; exports Model, Species, Mesh
pnpflow/fem.py       shape functions, quadrature, element/edge geometry
pnpflow/mesh.py      Mesh class (P2 nodes, P1 vertices, named boundaries, quality)
pnpflow/geometry.py  gmsh builders (cylinder_in_box, rectangle, sphere_axisym, pore_axisym, load_msh),
                     mesh_options context, apply_mesh_options (case mesh block -> size fields), sharp_corners
pnpflow/meshtools.py two-view mesh render (view_a/b.png) and mesh diagnose (numpy/scipy/matplotlib, no jax)
pnpflow/bc.py        BC dataclasses + value resolution
pnpflow/model.py     Species, Model: dof layout, Dirichlet data, weak form
pnpflow/solver.py    Assembler (jit/vmap/jacfwd), linear_solve (PARDISO / SuperLU), Newton, continuation
pnpflow/transient.py BDF2 time stepping (TransientAssembler, integrate)
pnpflow/analysis.py  fluxes, forces, profiles, adjoint sensitivities
pnpflow/cases.py     case library (@case): cylinder, gouy_chapman, eo_slit, mms, charged_sphere
pnpflow/pore.py      pore_charging case (axisym pore + SDL, paper DNS; dns_times / dns_init)
pnpflow/mms.py       manufactured solution helpers
pnpflow/project.py   case.json schema/defaults, STEP_RULES (step of each parameter), mesh key,
                     save_times, convergence.txt (plain Python, no jax: the app server imports it)
pnpflow/runner.py    end-to-end run -> results/<name>/ or cases/<case>/results/<name>/; build_case_mesh
pnpflow/output.py    JSON/VTU/MSH writers, fields_time.json, quick plots/, Progress (progress.json)
pnpflow/figures.py   house-style helpers (§9: house_style, arrow_axes, colorbar, save) + standard run figures
pnpflow/figures_pore.py  pore figures (Fig. 3 layout, evolution, charging, time steps)
pnpflow/export.py    CSV export of profiles (results/<run>/csv/)

tools/validate.sh            validation runs (check_transient + 4 analytic cases)
tools/check_transient.py     BDF2 exact-solution check
tools/export_csv.py          CSV for old runs          tools/export_fields_time.py  fields_time.json for old runs
tools/export_colormaps.py    cmasher LUTs -> app/ui/data/colormaps.json
tools/make_dns_cases.py      create cases/pore_dns_<shape>[_coarse]/case.json
tools/run_pore_dns.sh        queue the 4 DNS template runs (retry once with h_wall 0.15)
tools/figure_dns4.py         pnpflow vs DNS figure (figureDNS4 layout) + combined CSV / error table
scratch_eo.py                old stand-alone EO-slit smoke test

cases/               cases (case.json is source; mesh/ and results/ are generated, git-ignored)
results/             template runs and comparison output (not source, git-ignored)
DNS_compare/         the user's DNS data + scripts (transportFolder/, pawel.py, compare_mu.py,
                     extract_dns.py, graphStyle.py); local only, git-ignored

app/api.py           JSON API over run folders: runs, run files, logs, launch / stop, devices, template list
app/server.py        app dev server (port 8766): imports app/api.py, serves app/ui, adds figures/svg/mesh files and the case API
app/ui/js/app.js     shell: state, menus, toolbar, tree, documents, polling, keyboard, URL state
app/ui/js/bridge.js  the only backend module (Tauri invoke or fetch)
app/ui/js/start.js   Start page: cases, new case, recent runs, ranked examples
app/ui/js/pages.js   Geometry, Mesh, Materials, Boundaries, Equations, Study pages
app/ui/js/results.js Results views          app/ui/js/progress.js  run state, health, stage bars
app/ui/js/params.js  parameter editor (steps from run.py --list "steps"; unit menu; read-only for results)
app/ui/js/meshtools.js  Mesh Validation bar + case mesh tools (density, boundary/region refinement, corners, spray paint)
app/ui/js/repair.js  "Repair mesh" workspace: failed checks + pan/box/disk/pen/spray/erase on the mesh canvas;
                     spray sets a size, a local density or a curvature-based size; edits case.json `mesh`
app/ui/js/units.js   display units (length, potential, concentration, diffusivity, time, permittivity, temperature)
app/ui/js/fieldview.js, glfield.js, colormaps.js, plots.js, icons.js, util.js   viewer, WebGL, colours, Plotly, icons, helpers
app/src-tauri/       Tauri 2 shell (never compiled; see app/README.md)

docs/THEORY.md       equations, discretisation, BCs
docs/DATA_FORMAT.md  results folder contract
docs/MCP.md          agent protocol: tools, gmsh how-to, 2-view render, mesh diagnose
docs/PLAN.md         build plan in phases + FreeCAD layout mapping (screenshot in docs/)
docs/ui_design.html  GUI workbook (questions)      docs/ui_answers.md  the answers
docs/case_answers.md the user's answers on cases, mesh tools, saved times, units
docs/pore_answers.md the user's answers for the pore DNS (1 µm paper set-up)
docs/pore_dns_answers.md, pore_dns_plan.md   DNS comparison runs (10 µm): answers and plan
docs/run_dns_tau053.md   instructions: the four DNS runs to tau = 0.535 with auto-remesh + 6-panel figure
docs/summary.html    illustrated progress summary (docs/SUMMARY.md = text version)
docs/how_it_works.html  interactive map: pipeline, run-folder files, where the mesh and data are saved
PRODUCT.md DESIGN.md impeccable design context (product truth, design tokens)
.claude/skills/impeccable/   impeccable design skill (project install)
```

## 6. GUI decisions (from `docs/ui_answers.md`)

| topic | decision |
|---|---|
| users | experimental + computational, beginner / expert switch |
| platforms | Windows, Linux, macOS |
| stack | **Tauri 2**: Rust backend + web front end; reuse the (former) HUD's WebGL field viewer |
| GUI ↔ solver | start `run.py` as a process; exchange files (project JSON in, `results/` out) |
| Python install | `environment.yml`; the app creates the env on first run if missing |
| project file | one human-readable JSON file |
| model tree | Geometry, Mesh, Materials, Boundary Conditions, Governing Equations, Study, Results; step-by-step construction with status dots |
| units | SI and dimensionless, toggle; ions from a library, custom, or dimensionless |
| geometry | parametric templates, CAD/mesh import, gmsh scripts |
| continuation | automatic, shown on failure; always visual + numeric convergence tells |
| results | fields, EDL profiles, I–V, fluxes, forces, sensitivities, convergence, run comparison |
| exports | CSV, figures (PNG/PDF/SVG), reproducing Python script, interactive HTML report |
| look | the former HUD's colours and type, left menu with SVG icons + shortcut tooltips, standard menu bar (File, Edit, View…), dark default / follows OS, balanced density |
| layout reference | FreeCAD 1.1 (user's modified dark theme, `docs/Screenshot 2026-11-21 233015.png`): workbench tabs = construction steps, combo panel (tree + properties), Start page with templates / recent / examples, document tabs, status bar. Mapping and changes: `docs/PLAN.md` §1 |
| parameters | edited on the page of the step that owns them (Geometry, Mesh, Materials, Boundaries, Equations, Study; unknown names → Study); the panel under the tree only shows on Results |
| mesh validation | **Mesh Validation**: one full-width bar with the counts at the top of the Mesh page; click expands every check (value, limit, note, fix) and a note that problems can slow or break convergence and bias results; opens by itself on a warning/failure |
| case vs result (`docs/case_answers.md`) | nested: `cases/<case>/case.json` + `mesh/`, runs in `cases/<case>/results/<run>/` (frozen case.json, mesh copy, plots, figures, csv, `convergence.txt`); result inputs are read-only in the app ("Open its case") |
| mesh tools | toolbar: Create mesh (runs on the CPU; also on the Geometry page), Pan, Box, Disk, Paint (spray), Erase, Density, Boundary refinement, Sharp corners, Validation; "Repair mesh" workspace (failed checks, pen, spray by size / density / curvature); all edit case.json `mesh` |
| saved times | mode picker: template, every Δτ, N log-spaced, list (τ_end, 0.13 and τ_mid always saved); plus `dt_max`, `max_rejects`; compact Time stepping panel |
| units | unit menu next to every value with a known unit; stored value stays in the template unit; choice saved in case.json `units` |
| sizes | clickable elements 10 % bigger than the first version (heights, hit areas, their text) |
| examples | Start page shows at most 5, ranked by importance: pore_narrow 5, pore_dns_narrow_coarse 4, val_gouy_chapman 4, val_mms 3, val_eo_slit 3 |
| recent runs | Clear recents (Start page) hides the runs listed now, per browser; nothing is deleted; "Show N cleared" restores them |

The build plan, in phases with a check for each: `docs/PLAN.md`.

Design rules come from impeccable (project skill in `.claude/skills/impeccable`; use
`/impeccable shape|critique|audit|polish …`). The app is **Operate** mode: one UI type
family, fixed type scale, accent only for actions/selection/state, motion 150–250 ms for
state only, all states on every control. Banned: numbered section labels, uppercase kicker
labels, side-stripe borders, nested cards, gradient text, glassmorphism, decorative
monospace, Unicode glyphs as icons. Its `detect` linter reads HTML/CSS/JS, so it works on the
Tauri front end. Tokens and components: `DESIGN.md`.

### Pore charging, paper set-up (from `docs/pore_answers.md`)

| topic | decision |
|---|---|
| equations | paper Eq. (4): rho, S, Phi, tau = D t / l_p^2, R = r/a_p, Z = z/l_p; solver uses lengths / a_p, t~ = D t / a_p^2, so tau = t~ / L^2 |
| mu | mu = rho + 2 Phi; axial line `mu` is the cross-section average (weight r dr), `mu_axis` the centreline |
| BCs | pore wall phi = Phi_w (step, no Stern), blocking; pore end blocking (phi = `phi_end_mV`, default = Phi_w; replaced the old `end_charged` switch); SDL walls uncharged, no flux; SDL far end c = c0, phi = 0 |
| SDL | radius 2 max α, length l_p/2 + 1.5 max α (extra for meshing); Biot number not used |
| values | ε = 7.1e-10 F/m, D = 1.34e-9 m²/s, c0 = 0.94 mM (λ_D = 10.0 nm), a_p = 10 nm, l_p = 1 µm |
| times | early τ = 0.13; intermediate `tau_mid` (default 0.5); profiles saved at every τ in `pore.TAU_OUT` |
| runs | launched from the app server with a GPU/CPU choice |

### Pore DNS comparison runs (from `docs/pore_dns_answers.md`, plan `docs/pore_dns_plan.md`)

| topic | decision |
|---|---|
| cases | narrow (1→1), converging (2→1), diverging (1→2), wide (2→2) = DNS folders 100_100, 200_100, 100_200, 200_200 |
| length | **L = 10 µm** (DNS files + pawel.py), a_ref = 10 nm |
| values | c0 = 0.948 mM, ε = 7.0832e-10 F/m, D = 1.337e-9 m²/s, T = 298.15 K, Φ_w = 10 mV; CODATA constants kept |
| times | DNS folder times t = 0.005 k s, k = 0..15 (τ = 0.06685 k), τ_end = 1.01 |
| start | DNS initial state via new `dns_init` option; default unchanged |
| mesh | dry run + diagnose first; slightly coarser h_wall allowed only if a mesh does not fit in RAM |
| output | figureDNS4 layout (centreline μ, Φ at Z = 0.5, DNS squares) + combined CSV |
| μ line | centreline `mu_axis` (matches the DNS `lineYe` file); the cross-section average `mu` is also in profiles.json |
| DNS data | `DNS_compare/transportFolder/poreRadius<in>_<out>/<t>/lineYe_*.xy` (centreline) and `lineXm5_*.xy` (Z = 0.5). `figure_dns4.py`'s default `--dns` is an old WSL path: pass `--dns DNS_compare/transportFolder` |

Status (2026-11-23): coarse cases `cases/pore_dns_{converging,diverging,wide}_coarse`
(h_wall 0.15; mesh density 0.95 / 1.0 / 0.91); the narrow case is not created yet. One
converging run reached τ = 0.14 (`..._261123_1018`, density 1.2, 241k triangles, 2.63M dofs,
76 BDF2 steps in 2622 s on the GPU, then the equilibrium solve). Next: all four to τ_end = 0.535
(DNS k = 8 is τ = 0.5348) at density 0.95, remeshing on failure, and a 3 × 2 figure
(μ̂(Z), Φ̂ and ρ̂ at Z = 0.5; τ = 0.13, 0.53): `docs/run_dns_tau053.md`. The memory numbers in
`docs/pore_dns_plan.md` were for WSL (15 GB); the repo now runs on native Linux (pop-os,
`~/Code/PNP-Flow`), so check `free -g` before sizing meshes.

## 7. Agent protocol (MCP) — summary of `docs/MCP.md`

- The MCP server exposes project, geometry, mesh, run and results tools. Same Python
  functions as the GUI; state lives in files.
- **Geometry is built in gmsh** (OCC kernel): build shapes, `fragment` multi-region models,
  name every boundary curve and region with physical groups, size the mesh from the Debye
  length (`h_wall ≤ λ_D/8` within `3 λ_D` of charged/reacting walls), generate P2.
- **Entrances need extra mesh.** Where a charged pore wall meets an uncharged reservoir
  face (pore mouths, re-entrant corners) refine beyond `λ_D/8`: the corner is singular and
  the access region carries the entrance resistance. `geometry.pore_axisym` uses
  `h_entr = h_wall/2` at the corner and `h_wall` within 1.5 max(α) of the mouth; leave
  extra reservoir length (here `+1.5 max α`) so the refined region fits.
- **Always render the mesh from two views** (whole domain by boundary group; zoom on the finest
  region by quality; two camera angles in 3D) and look at both.
- **Always run `mesh_diagnose`** (unnamed boundaries, BC coverage, EDL resolution, quality,
  size jumps, bad P2 elements, connectivity, far field, size budget) and fix failures before
  solving.
- Runs report live status (`fraction`, `health`, `tells`) and finish with automatic figures.

## 8. Physics roadmap (user priority)

1. Project file + generic case (needed by the GUI and MCP for everything below). *Cases done
   (2026-11-22: case.json per template, mesh tools, time settings); still to do: a generic
   case that is not tied to a template (geometry, ions, BCs chosen freely), figure selection.*
2. Applied potential / current (potentiostat, galvanostat) and **Butler–Volmer** kinetics
   (with Frumkin correction) as new `edge_residual` terms.
3. Time-dependent: chronoamperometry, CV. *BDF2 time stepping done (2026-11-22, used by
   `pore_charging`); still to do: time-varying BCs (CV sweeps), current output vs time.*
4. **EIS** by small-signal linearisation: `(iω M + J) δU = −(∂R/∂p) δp`, reusing the JAX
   Jacobian.
5. Homogeneous reactions (reaction network → sources).
6. Axisymmetric (r–z). *Done for PNP + Poisson (`Model(coord="axisym")`, used by
   `charged_sphere` and `pore_charging`); flow in axisym still to do.*
7. Porous electrodes.
8. 3D.

Known limits today: time-dependent BCs not yet supported (steps only), 2D (planar or
axisymmetric, no flow in axisym), dimensionless input except in `pore_charging`, and the electric
force on bodies neglects Maxwell stress on the outer boundary.

## 9. Figure house style

All automatic and exported figures (matplotlib) follow this style:

```python
import cmasher as cmr
import matplotlib.pyplot as plt

def apply_custom_style(num_colors: int = 4, cmap: str = "cmr.cosmic_r", cmap_range: tuple = (0.2, 0.75)):
    cm = cmr.get_sub_cmap(cmap, *cmap_range)
    colors = [cm(i / (num_colors - 1)) for i in range(num_colors)]
    plt.rcParams.update({
        "axes.prop_cycle": plt.cycler(color=colors),
        "font.size": 24, "axes.labelsize": 24, "axes.titlesize": 24, "figure.titlesize": 24,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.family": "times-roman", "text.usetex": True,
    })
    return colors
```
- Note: under `text.usetex`, matplotlib does not recognise `"times-roman"` and falls back to
  Computer Modern. `pnpflow/figures.py` keeps the function as written and then sets
  `font.family="serif"`, `font.serif=["Times"]` so figures really are Times.
- Colormaps: **cmasher, three maps only**: `cmr.cosmic` (potential, concentrations, mesh
  quality), `cmr.ember` (magnitudes |E|, |u|), `cmr.bubblegum` (charge density). No diverging
  maps (a two-signed field uses its map over the real range; all-negative uses `_r`).
  **Never viridis** or any other matplotlib default anywhere: figures, `plots/`, the app, the
  thumbnails. Check with `grep -rniE "viridis|RdBu|magma|plasma|inferno|jet|turbo" pnpflow app tools`.
- Axes: no boxes. Only the left and bottom axis lines, each ending in an arrowhead
  (`figures.arrow_axes`); colour bars without outline (`figures.colorbar`); no grid.
- Panel labels a), b), … with `figures.add_subplot_labels` (same as the TLsolver style).
- Line width 2.5, `frameon=False` legends.
- Normalised colour bars: horizontal stacked-fraction label with a subscript max, e.g.
  `r'$\displaystyle\frac{c_+}{c_{+,\max}}$'`, never "/max" and never vertical.
- One title per graph, directly above it, with clear space (pad ≈ 18–24 pt); ticks ≥ 18 pt.
- Minimal whitespace (≈ 0.3 in margins, never 0 = clipped). ≥ 300 dpi.
- Save to `results/<run>/figures/<name>.png` (300 dpi) and `results/<run>/svg/<name>.svg`.
- Without a TeX install, fall back to mathtext with a Times-like font; never fail the run.
- Before showing or batch-rendering a figure, render one and inspect it (margins, clipping,
  overlaps).

## 10. Rules for agents

- Use JAX for anything that needs derivatives. Never hand-code a Jacobian; add terms to
  `element_residual` / `edge_residual` and let `jacfwd` do the rest.
- Keep `jax_enable_x64` on (set in `pnpflow/__init__.py`); Newton on PNP needs float64.
- Every new term or BC needs a check: extend `mms` or add an analytic validation case, then
  run `tools/validate.sh`.
- New outputs go into `summary.json` / `fields.json` as *optional* keys documented in
  `docs/DATA_FORMAT.md`; the GUI must tolerate missing keys. Keep the file contract stable.
- Meshes: follow `docs/MCP.md` §2–5 (gmsh, named groups, two-view render, diagnose).
- UI work: read `PRODUCT.md` and `DESIGN.md` first; use the impeccable skill.
- Plots: §9. numpy (not jnp) in plotting code.
- Run solver code inside the `pnpflow` conda env (set `JAX_PLATFORMS=cpu` to force CPU).
  Mesh builds (`--mesh-only`) run on the CPU; solves use the GPU for assembly, but PARDISO
  factorises in system RAM, so RAM (not GPU memory) limits mesh size.
- Long runs (tens of minutes to hours): launch detached (`nohup … &`) and poll
  `progress.json` / `stdout.log`; never block on a run. Judge success as in §4 (exit code 0 is
  not enough).
- Never overwrite or delete an existing run folder; give new runs new names.
- Do not commit `results/`, `cases/*/mesh/`, `cases/*/results/`, `DNS_compare/`,
  `__pycache__/` or Windows `*:Zone.Identifier` files (all git-ignored). Some `.pyc` and
  `Zone.Identifier` files were committed before the ignore rules existed; remove them from the
  index with `git rm --cached`, not from disk.
- When a change adds a module, tool, option, output file or workflow, update this file
  (§1 status, §5 map, and §4 commands if it is run by hand).
