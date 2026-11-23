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
| Current dashboard (`hud/`) | stdlib server + vanilla JS/WebGL + Plotly | working; visual reference for the app |
| Desktop app (`app/`) | **Rust + Tauri 2**, web front end | first version runs in a browser (`python app/server.py`, port 8766); Tauri shell written, not compiled |
| MCP server (`pnpflow/mcp`) | Python `mcp` SDK | specified (`docs/MCP.md`), not started |
| Time stepping (`pnpflow/transient.py`) | Python + JAX | variable-step BDF2, checked (order 2.1, `tools/check_transient.py`) |
| Project file + generic case | JSON | not started — next step |

## 2. Physics (nondimensional, steady)

Scales: length `a`, speed `U`, concentration `c0`, potential `V_T = k_B T / e`.

```
species i :  div N_i = s_i
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
   (`h ≈ delta/8` near charged walls so the EDL is resolved). `load_msh` reads external meshes.
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
   `Newton` is reused. A case asks for it with `Setup.transient` (t_end, output times,
   profile/figure callbacks); `runner._run_transient` writes profiles at each output time,
   `summary.transient`, `solution.npz` (`t`, `U_t`) and live `progress.time`.
8. **Output** (`output.py`, `runner.py`): `results/<run>/` with `progress.json` (live),
   `summary.json`, `fields.json`, `profiles.json`, `plots/*.png`, `solution.{vtu,msh,npz}`.

### Boundary conditions (`bc.py`)
Values are numbers, parameter names (strings) or JAX functions `f(x, params)`.

| key | options | default if unset |
|---|---|---|
| `c` | `Dirichlet(v or "bulk")`, `Flux(N·n)`, `BLOCKING`, `OUTFLOW` (per species via dict) | blocking |
| `phi` | `Dirichlet`, `Robin(C, ref)` (Stern), `Charge(sigma)` | `dphi/dn = 0` |
| `u` | `NOSLIP`, `Dirichlet((uy, uz))`, `Component(i, v)`, `Pressure(p0)`, `FREE` | no-slip |

### Cases (`cases.py`) — `build(params) -> Setup`, registered with `@case`
| case | purpose | validated against |
|---|---|---|
| `cylinder` | charged cylinder in cross flow, EDL resolved | symmetry, Gauss balance, zero wall flux |
| `gouy_chapman` | planar EDL, optional Stern layer | exact GC: max err ~3e-6 |
| `eo_slit` | electro-osmotic slit flow | nonlinear PB: rel. err ~2e-4 |
| `mms` | manufactured solution, all terms + BC types | max err ~2e-4 |
| `charged_sphere` | axisymmetric Poisson | exact potential, Gauss |
| `pore_charging` | **time-dependent** pore charging from an SDL reservoir (axisym, paper DNS) | equilibrium vs cylinder PB at Z = 0.5; BDF2 vs exact decay |

`bash tools/validate.sh` runs the validation cases (all pass as of 2026-11-15).

## 4. Running

```bash
# One conda env for Linux and macOS
conda env create -f environment.yml && conda activate pnpflow
python run.py --list
python run.py cylinder --name demo --set delta=0.05
python hud/server.py                       # current dashboard at http://localhost:8765
python app/server.py                       # app at http://localhost:8766 (Study tab: Device = Auto/GPU/CPU)
python tools/check_transient.py            # time-stepper check (BDF2 order 2)
```
Device: the server sets `JAX_PLATFORMS` per run (`/api/launch` `device: auto|cpu|gpu`,
`/api/devices` probes the GPU). JAX assembly runs on the GPU; the sparse solve (PARDISO) stays on
the CPU. From the shell: `JAX_PLATFORMS=cuda python run.py ...`.
`run.py --list` prints every case with default params + help text as JSON.
`environment.yml`: gmsh from conda-forge (no system libGLU); `pypardiso` only on x86_64
(Apple Silicon uses SuperLU); Linux installs `jax[cuda13]` (GPU) automatically.

## 5. Repository map

```
run.py               CLI: run a case, --set overrides, --list
environment.yml      conda env (solver, gmsh, cmasher, mcp)
pnpflow/fem.py       shape functions, quadrature, element/edge geometry
pnpflow/mesh.py      Mesh class (P2 nodes, P1 vertices, named boundaries, quality)
pnpflow/geometry.py  gmsh builders: cylinder_in_box, rectangle, load_msh
pnpflow/bc.py        BC dataclasses + value resolution
pnpflow/model.py     Species, Model: dof layout, Dirichlet data, weak form
pnpflow/solver.py    Assembler (jit/vmap/jacfwd), Newton, continuation
pnpflow/analysis.py  fluxes, forces, profiles, adjoint sensitivities
pnpflow/cases.py     case library (@case)
pnpflow/mms.py       manufactured solution helpers
pnpflow/output.py    JSON/VTU/PNG writers, Progress
pnpflow/transient.py BDF2 time stepping (TransientAssembler, integrate)
pnpflow/pore.py      pore_charging case (axisym pore + SDL, paper DNS)
pnpflow/figures_pore.py  pore figures (Fig. 3 layout, evolution, charging, time steps)
tools/check_transient.py  BDF2 exact-solution check
pnpflow/export.py    CSV export of profiles (results/<run>/csv/); tools/export_csv.py re-exports old runs
docs/pore_answers.md the user's answers for the pore DNS
pnpflow/runner.py    end-to-end run → results/<name>/
hud/                 current web dashboard (visual reference for the app)
docs/THEORY.md       equations, discretisation, BCs
docs/DATA_FORMAT.md  results folder contract
docs/MCP.md          agent protocol: tools, gmsh how-to, 2-view render, mesh diagnose
docs/ui_design.html  GUI workbook (questions)      docs/ui_answers.md  the answers
docs/PLAN.md         build plan in phases + FreeCAD layout mapping (screenshot in docs/)
docs/summary.html    illustrated progress summary (docs/SUMMARY.md = text version)
app/                 desktop app: server.py (dev server), ui/ (front end), src-tauri/ (unbuilt)
PRODUCT.md DESIGN.md impeccable design context (product truth, design tokens)
.claude/skills/impeccable/   impeccable design skill (project install)
results/             run outputs (not source)
```

## 6. GUI decisions (from `docs/ui_answers.md`)

| topic | decision |
|---|---|
| users | experimental + computational, beginner / expert switch |
| platforms | Windows, Linux, macOS |
| stack | **Tauri 2**: Rust backend + web front end; reuse the HUD's WebGL field viewer |
| GUI ↔ solver | start `run.py` as a process; exchange files (project JSON in, `results/` out) |
| Python install | `environment.yml`; the app creates the env on first run if missing |
| project file | one human-readable JSON file |
| model tree | Geometry, Mesh, Materials, Boundary Conditions, Governing Equations, Study, Results; step-by-step construction with status dots |
| units | SI and dimensionless, toggle; ions from a library, custom, or dimensionless |
| geometry | parametric templates, CAD/mesh import, gmsh scripts |
| continuation | automatic, shown on failure; always visual + numeric convergence tells |
| results | fields, EDL profiles, I–V, fluxes, forces, sensitivities, convergence, run comparison |
| exports | CSV, figures (PNG/PDF/SVG), reproducing Python script, interactive HTML report |
| look | HUD colours and type, left menu with SVG icons + shortcut tooltips, standard menu bar (File, Edit, View…), dark default / follows OS, balanced density |
| layout reference | FreeCAD 1.1 (user's modified dark theme, `docs/Screenshot 2026-11-21 233015.png`): workbench tabs = construction steps, combo panel (tree + properties), Start page with templates / recent / examples, document tabs, status bar. Mapping and changes: `docs/PLAN.md` §1 |

### Pore-charging DNS decisions (from `docs/pore_answers.md`)

| topic | decision |
|---|---|
| equations | paper Eq. (4): rho, S, Phi, tau = D t / l_p^2, R = r/a_p, Z = z/l_p; solver uses lengths / a_p, t~ = D t / a_p^2, so tau = t~ / L^2 |
| mu | mu = rho + 2 Phi; axial lines are cross-section averages (weight r dr) |
| BCs | pore wall phi = Phi_w (step, no Stern), blocking; pore end blocking (phi = Phi_w by default, `end_charged`); SDL walls uncharged, no flux; SDL far end c = c0, phi = 0 |
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

The build plan, in phases with a check for each: `docs/PLAN.md`.

Design rules come from impeccable (project skill in `.claude/skills/impeccable`; use
`/impeccable shape|critique|audit|polish …`). The app is **Operate** mode: one UI type
family, fixed type scale, accent only for actions/selection/state, motion 150–250 ms for
state only, all states on every control. Banned: numbered section labels, uppercase kicker
labels, side-stripe borders, nested cards, gradient text, glassmorphism, decorative
monospace, Unicode glyphs as icons. Its `detect` linter reads HTML/CSS/JS, so it works on the
Tauri front end. Tokens and components: `DESIGN.md`.

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

1. Project file + generic case (needed by the GUI and MCP for everything below).
2. Applied potential / current (potentiostat, galvanostat) and **Butler–Volmer** kinetics
   (with Frumkin correction) as new `edge_residual` terms.
3. Time-dependent: chronoamperometry, CV. *BDF2 time stepping done (2026-11-22, used by
   `pore_charging`); still to do: time-varying BCs (CV sweeps), current output vs time.*
4. **EIS** by small-signal linearisation: `(iω M + J) δU = −(∂R/∂p) δp`, reusing the JAX
   Jacobian.
5. Homogeneous reactions (reaction network → sources).
6. Axisymmetric (r–z).
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
  HUD. Check with `grep -rniE "viridis|RdBu|magma|plasma|inferno|jet|turbo" pnpflow app hud`.
- Axes: no boxes. Only the left and bottom axis lines, each ending in an arrowhead
  (`figures.arrow_axes`); colour bars without outline (`figures.colorbar`); no grid.
- Panel labels a), b), … with `figures.add_subplot_labels` (same as the TLsolver style).
- Line width 2.5, `frameon=False` legends, no grid, panels labelled a), b), ….
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
- Do not commit `results/` contents or `__pycache__/`.
