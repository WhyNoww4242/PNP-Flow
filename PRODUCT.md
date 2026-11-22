# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack
Desktop app built with Tauri 2: a Rust backend (process control, file watching, fast mesh and
field decoding) and a web front end in the system webview. Runs on Windows, Linux and macOS.
The solver is a separate Python/JAX program (`run.py`) that the app starts as a process; the two
exchange files only (`docs/DATA_FORMAT.md`). The Python side is installed from
`environment.yml`. An MCP server (`docs/MCP.md`) gives AI agents the same operations as the GUI.

## Users
- **Experimental electrochemists** who know the chemistry but not finite elements. They want
  to model their own cell, electrode, channel or pore: ions in lab units, a geometry from a
  template or CAD, sane defaults, and a clear signal when something is wrong.
- **Computational researchers** who want every setting visible, dimensionless input, raw
  output and scripts that reproduce a run.
A beginner / expert switch serves both from one interface.

## Product Purpose
Steady (and later transient and frequency-domain) Poisson–Nernst–Planck transport, optionally
coupled to Stokes or Navier–Stokes flow, with exact JAX derivatives. Users set up a model,
solve it, and read fields, EDL profiles, currents, fluxes, forces and sensitivities.
Success: a scientist goes from a blank project to a converged, checked result without writing
code, and trusts the result because the app shows its checks.

## Positioning
Exact Jacobians and adjoint sensitivities from JAX for every model the user builds, with
validation checks (balances, positivity, convergence health) reported for every run. The
model, mesh and results are plain files that an AI agent can drive through the same protocol
as the GUI.

## Operating Context
- Desktop or laptop at a desk, often next to lab data and papers; sessions last hours.
- Workflow: Geometry → Mesh → Materials → Boundary Conditions → Governing Equations → Study →
  Results, shown as a model tree with step-by-step construction.
- Runs take seconds to tens of minutes; users watch live progress and convergence.
- Outputs go into papers and talks: publication figures (PNG/PDF/SVG), CSV, and a quick
  interactive HTML report.
- Familiar software: COMSOL (the user finds its look dated), ParaView, matplotlib.

## Capabilities and Constraints
- Working today: 2D planar PNP + flow, Dirichlet / flux / blocking / outflow / Stern / surface
  charge / pressure BCs, continuation, adjoint sensitivities, four validated cases.
- Units: SI and dimensionless, with a toggle. Ions from a built-in library, custom ions, or
  fully dimensionless species.
- Continuation and sweeps run automatically and surface only on trouble, but convergence
  always has visible and numeric tells.
- One human-readable project file (JSON).
- Planned physics, in priority order: applied potential/current, Butler–Volmer kinetics,
  time-dependent (chronoamperometry, CV), EIS, homogeneous reactions, axisymmetric, porous
  electrodes, 3D.
- Geometry: parametric templates, CAD/mesh import, gmsh scripts. gmsh is the only geometry kernel.

## Brand Commitments
- Name: pnpflow.
- Keep the current HUD's colours and type, its left menu, overall layout and mesh view.
- Standard desktop menu bar on top (File, Edit, View, …).
- Scientific figures use cmasher colormaps and the `apply_custom_style` house style
  (LaTeX, Times, 24 pt, no top/right spines). See `AGENTS.md` §9.

## Evidence on Hand
- Validation results in `results/val_*` (Gouy–Chapman, EO slit, manufactured solution) and
  `results/cyl_quick`.
- The current HUD (`hud/`) as the visual reference.
- No user testimonials or external case studies exist; do not invent any.

## Product Principles
1. **Show the checks.** Every result comes with its balances, positivity and convergence
   health; a wrong answer should look wrong.
2. **Lab units in, physics out.** Users enter what they measure; dimensionless groups are
   derived and shown, never required (unless expert mode is on).
3. **Files are the interface.** Project, mesh and results are plain files shared by the GUI,
   the CLI and AI agents.
4. **Hide the solver until it matters.** Continuation, tolerances and stabilisation have good
   defaults and appear on failure or in expert mode.
5. **Figures ready for a paper.** Every plot can be exported in house style without rework.

## Accessibility & Inclusion
Text contrast at least 4.5:1 in both themes; full keyboard operation with shortcuts shown on
hover; colormaps that remain readable in greyscale and for colour-vision deficiencies
(cmasher maps are perceptually uniform).
