# pnpflow — progress summary (2026-11-22)

The illustrated version with panels is `docs/summary.html`; open it in a browser.

## At a glance

pnpflow is a finite-element program for electrochemical systems: Poisson–Nernst–Planck ion
transport, optionally coupled to fluid flow. Its goal is to let electrochemists model their own
systems without writing code. The solver is Python + JAX. The desktop app is a Rust (Tauri 2) shell
around a web front end, which runs in a browser for now.

| Part | State |
|---|---|
| Solver: PNP + Stokes/Navier–Stokes, exact JAX Jacobians, adjoint sensitivities | working |
| Axisymmetric (r–z) for Poisson and ion transport | new, validated |
| `charged_sphere` example (Poisson only) | new, validated |
| Two-view mesh render + mesh diagnose on every run | new |
| House-style figures (cmasher, LaTeX, 300 dpi) on every run | new |
| Live status (`fraction`, `health`, `tells` in progress.json) | new |
| Desktop app front end, FreeCAD-style (`app/`) | first version, in a browser |
| Tauri native window (`app/src-tauri/`) | written, not compiled (no Rust/MSVC on Windows; no WebKitGTK in WSL) |
| MCP agent protocol | specified (`docs/MCP.md`) |
| Project file for no-code models | planned (`docs/PLAN.md` Phase 1) |

## How it works

- **Desktop app (Rust + JavaScript):** window, menus, model tree and field viewer. It starts runs and
  watches their files. `.js` files are JavaScript, the language of the web view, not Java.
- **`run.py` (Python):** builds the mesh, solves, analyses, and writes `results/<run>/`.
- **JAX core:** the weak form per element; exact Jacobians with `jax.jacfwd`; Newton with line search and
  continuation; adjoint sensitivities.

One run goes: gmsh geometry → P2 mesh → two-view mesh check + diagnose → weak form → `jacfwd`
Jacobian → Newton (live `progress.json`) → analysis + checks → JSON, figures, ParaView files.

## Example: uniformly charged sphere

The case solves `−lap φ = q` inside a sphere of radius R = 1, and `0` outside, in a domain of radius 10.
It is axisymmetric, so the solver works on r ≥ 0 and the figures mirror the result. The exact solution is
`φ = qR²/2 − qρ²/6` inside and `qR³/(3ρ)` outside.

| Check | Value |
|---|---|
| max \|φ − exact\| / max φ, default mesh (4,637 elements) | 3.7e-6 |
| same, mesh size halved (18,058 elements) | 5.3e-7 (≈ h³) |
| Gauss: outer flux vs enclosed charge | 4e-15 |
| Mesh diagnose | all pass (min quality 0.78) |

Images:
- `summary_assets/sphere_phi.jpg`, `sphere_E.jpg`, `sphere_profile_axis.jpg`
- `sphere_mesh_a.jpg`, `sphere_mesh_b.jpg`

## Validation

| Case | Compared with | Error |
|---|---|---|
| gouy_chapman | exact planar double layer | 3.5e-6 |
| eo_slit | nonlinear Poisson–Boltzmann flow | 2.3e-4 |
| mms | manufactured solution | 2.1e-4 |
| cylinder | Gauss balance, symmetry | 1.8e-15 |
| charged_sphere | exact sphere potential | 3.7e-6 |

## Desktop app

The layout follows the FreeCAD reference:
- menu bar;
- workbench tabs in build order (Geometry → Mesh → Materials → Boundaries → Equations → Study → Results);
- model tree, with the parameter editor below it;
- document tabs;
- a status bar showing the run's progress and health.

Screenshots: `summary_assets/ui_start.jpg`, `ui_sphere.jpg`, `ui_mesh.jpg`, `ui_sphere_light.jpg`.

**Stubs:** editing in the build-step workbenches, and the SI / dimensionless toggle. Both wait for the
project file.

## How to run

```
cd C:\Users\Gnome\Desktop\VMBox\pnpflow
wsl -d Debian -- bash tools/wsl_run.sh python run.py charged_sphere --name sphere_demo
wsl -d Debian -- bash tools/wsl_run.sh python app/server.py
  → http://localhost:8766/?run=sphere_demo&wb=results&view=field
```

## Next

1. **Project file:** models defined in JSON instead of Python.
2. **MCP server:** lets an AI agent build, check, run and report models.
3. **Tauri build:** needs Rust and the MSVC Build Tools on Windows.
4. **Physics, in your order:** Butler–Volmer and applied potential/current; CA/CV; EIS; homogeneous
   reactions; porous electrodes; 3D.
