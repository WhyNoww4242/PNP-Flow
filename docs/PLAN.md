# pnpflow build plan

Written 2026-11-21. Inputs: `docs/ui_answers.md`, `PRODUCT.md`, `DESIGN.md`, `docs/MCP.md`,
and the layout reference `docs/Screenshot 2026-11-21 233015.png` (FreeCAD 1.1, dark theme,
modified by the user, which the user rates as "pretty good").

## 1. Layout reference: what we take from FreeCAD and what we change

The screenshot shows the FreeCAD Start page. Its shell maps closely onto pnpflow's
step-by-step model building.

| FreeCAD element | pnpflow version | change from FreeCAD |
|---|---|---|
| Menu bar: File, Edit, View, Tools, Macro, Sketch, Part Design, Windows, Help | File, Edit, View, Model, Study, Results, Tools, Help | fewer menus; workbench-specific commands live in the workbench menu, not new top-level menus |
| Workbench tabs (Sketcher, Part Design, Part, Surface, Assembly, TechDraw, Draft, +) | Workbench tabs = the construction steps: **Geometry, Mesh, Materials, Boundaries, Equations, Study, Results** | tabs follow the model-tree order and show a status dot (done / needs input / error); a tab is never hidden, just marked |
| Icon toolbar (three rows, 40+ multicoloured raster icons) | One row of ≤ 12 tools for the active workbench, grouped, with an overflow menu | single monochrome SVG icon set (accent only for state); every icon has a tooltip with its name and shortcut |
| Left "Model" panel with **View / Data** tabs (tree above, properties below) | Combo panel on the left: model tree on top, property editor below (Data), display settings (View) | keep FreeCAD's combo layout, which leaves the canvas wide; the tree doubles as the step-by-step guide |
| Start page: **New File** tiles (Parametric Body, Assembly, 2D Draft, BIM, Empty, Open) | **New project** from templates: Electrode pair in a channel, Charged cylinder in flow, Planar EDL (Gouy–Chapman), Electro-osmotic slit, Nanopore, Empty project, Open project | templates show a real geometry preview and their key parameters instead of generic icons; a compact list, not a wall of identical tiles |
| **Recent Files** with thumbnails and file size | **Recent projects** with a thumbnail of the last result (field map in cmasher), status (converged / failed) and date | thumbnails are real renders; status is a word plus colour |
| **Examples** gallery (incl. a FEM example) | **Examples** = the validation cases with their error vs the exact solution | doubles as a credibility check for new users |
| Document tabs at the bottom (Start ×) | Tabs for open items: Start, the project, each opened run, run comparisons | same |
| Status bar dropdowns (colour, navigation style "CAD", units "Dimension") | Status bar: units toggle (SI / dimensionless), beginner / expert, run progress bar, health badge, last residual | live run state always visible |
| "Open First Start Setup", "Do not show this Start page again" | First-run setup: create/find the conda env, check gmsh and JAX, pick units; same opt-out | the env check replaces FreeCAD's preferences wizard |

Things in the screenshot we avoid:
- grey-on-grey tiles with low text contrast;
- mixed icon styles and colours, and toolbars too crowded to scan;
- an empty Model panel on the Start page (we show the construction steps there instead).

## 2. Phases

Each phase ends with a check that proves it works; nothing counts as done on code reading alone.

### Phase 0 — groundwork (done 2026-11-21)
`AGENTS.md`, `PRODUCT.md`, `DESIGN.md`, `docs/MCP.md`, `docs/ui_answers.md`,
`environment.yml` (conda solve checked on linux-64), impeccable skill in `.claude/skills/`.

### Phase 1 — project file and generic case
- JSON schema for a project: geometry (template, script or file), mesh sizes, materials
  (solvent ε, T, ions from a library / custom / dimensionless), boundary conditions per
  named boundary, governing equations (PNP, flow type, advection form), study
  (continuation, tolerances, sensitivities), outputs.
- SI ↔ dimensionless conversion (`delta`, `Pe`, `Re`, `K` derived and reported).
- Ion library (z, D at 25 °C) with custom entries.
- `run.py --project file.json` through one generic `@case`.
- **Check:** the four existing cases rewritten as project files give the same results as now
  (`tools/validate.sh`).

### Phase 2 — mesh tools, live status, figures
- `mesh_render` (two views) and `mesh_diagnose` (`docs/MCP.md` §4–5) as Python functions.
- `progress.json` gains `fraction`, `health`, `tells` (`docs/MCP.md` §6).
- `pnpflow/figures.py`: house-style figures written automatically after every run
  (`AGENTS.md` §9), cmasher look-up tables exported for the GUI.
- **Check:** diagnose catches a deliberately unnamed boundary and an under-resolved EDL;
  figures pass the graphInspect review (margins, clipping, 300 dpi).

### Phase 3 — MCP server
- `python -m pnpflow.mcp` with the tools in `docs/MCP.md` §1, images returned inline.
- **Check:** an agent builds the electrode-pair geometry from scratch through MCP, renders
  both views, fixes a diagnose failure, solves and gets the report.

### Phase 4 — Tauri shell (FreeCAD-style layout from §1)
- Tauri 2 project; Rust commands for launching runs, watching `progress.json`, reading
  results, managing the conda env.
- Shell: menu bar, workbench tabs, toolbar, combo panel (tree + properties), canvas,
  document tabs, status bar. Start page with templates, recent projects, examples.
- Port the HUD's WebGL field and mesh viewer; replace viridis/RdBu with cmasher LUTs.
- SVG icon set (Lucide + custom electrode / ion / mesh icons), tooltips with shortcuts.
- **Check:** impeccable `detect` passes on the front end; `/impeccable critique` and `audit`
  have no P0/P1 findings; screenshots in light and dark on Windows and Linux.

### Phase 5 — workbenches
Geometry (templates, import, script editor, gmsh preview), Mesh (sizes, two-view render,
diagnose panel), Materials (ion table, units), Boundaries (per-boundary conditions with
the BC coverage check), Equations, Study (continuation hidden unless needed, convergence
tells), Results (fields, profiles, I–V, fluxes, forces, sensitivities, compare runs).
- **Check:** a user who has not seen the code builds the electrode-pair model end to end.

### Phase 6 — exports
CSV, PNG/PDF/SVG figures, a Python script that reproduces the run, and a one-file interactive
HTML report per run.

### Phase 7 — physics roadmap
In order (`AGENTS.md` §8): applied potential/current + Butler–Volmer → time-dependent
(chronoamperometry, CV) → EIS (small signal) → homogeneous reactions → axisymmetric →
porous electrodes → 3D. Each item gets a manufactured-solution or analytic check.
