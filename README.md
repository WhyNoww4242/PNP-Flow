# pnpflow

A steady Poisson–Nernst–Planck + Navier–Stokes finite-element solver. Meshes come from gmsh. Assembly, Jacobians and sensitivities use JAX (`jit` + `vmap` + `jacfwd`). Linear solves use PARDISO.
Equations and design choices are in `docs/THEORY.md`. The result format the app reads is in `docs/DATA_FORMAT.md`. Guide for agents and contributors: `AGENTS.md`.

## Setup
```bash
conda env create -f environment.yml && conda activate pnpflow
# Linux installs jax[cuda13] automatically; check: python -c "import jax; print(jax.devices())"
```

## Run
```bash
python run.py --list
python run.py cylinder --name my_run --set delta=0.05 --set K=20
bash tools/validate.sh     # validation cases
```
Each run writes its outputs to `results/<name>/`:
- `summary.json`: checks, fluxes, forces, charge balance, sensitivities
- `fields.json` and `profiles.json`: data for the app
- `plots/*.png`
- `solution.vtu` (open in ParaView) and `solution.msh` (open in gmsh)
- `solution.npz`

## Cases (`pnpflow/cases.py`)
| case | what it is |
|---|---|
| `cylinder` | target problem: charged cylinder (+25 mV) in cross flow, EDL resolved |
| `gouy_chapman` | planar EDL compared to the exact solution; optional Stern layer |
| `eo_slit` | electroosmotic slit compared to the nonlinear Poisson–Boltzmann solution |
| `mms` | manufactured solution, all terms and all BC types; sources come from JAX autodiff |
| `charged_sphere` | axisymmetric Poisson, compared to the exact potential |
| `pore_charging` | time-dependent pore charging from a reservoir (`pnpflow/pore.py`) |

To add a problem, copy a `@case` function and change four things: the geometry builder, the species, the `bcs` dict and the continuation path.

## Layout
`fem.py` holds the shapes and quadrature. `mesh.py` reads gmsh output. `geometry.py` builds geometries in gmsh. `bc.py` defines the BC types. `model.py` holds the weak form. `solver.py` does assembly, Newton and continuation. `analysis.py` computes fluxes, forces, profiles and adjoint sensitivities. `output.py` writes results. `runner.py` runs a case from start to finish. `app/` is the desktop app: `python app/server.py` serves it at http://localhost:8766 (`app/api.py` is its JSON API over the run folders).
