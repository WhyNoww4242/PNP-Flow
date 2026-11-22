# pnpflow

A steady Poisson–Nernst–Planck + Navier–Stokes finite-element solver. Meshes come from gmsh. Assembly, Jacobians and sensitivities use JAX (`jit` + `vmap` + `jacfwd`). Linear solves use PARDISO.
Equations and design choices are in `docs/THEORY.md`. The result format the HUD reads is in `docs/DATA_FORMAT.md`.

## Setup (WSL Debian, already done on this machine)
```bash
uv venv -p 3.12 ~/venvs/pnp && source ~/venvs/pnp/bin/activate
uv pip install "jax[cuda12]" gmsh scipy numpy matplotlib meshio pypardiso
# gmsh needs libGLU (no sudo): it is unpacked to ~/venvs/pnp/syslib
# and that folder is added to LD_LIBRARY_PATH by the activate script.
```

## Run
```bash
wsl -d Debian -- bash /mnt/c/Users/Gnome/Desktop/VMBox/pnpflow/tools/wsl_run.sh python run.py --list
wsl -d Debian -- bash .../tools/wsl_run.sh python run.py cylinder --name my_run --set delta=0.05 --set K=20
wsl -d Debian -- bash .../tools/wsl_run.sh bash tools/validate.sh     # validation cases
```
Each run writes its outputs to `results/<name>/`:
- `summary.json`: checks, fluxes, forces, charge balance, sensitivities
- `fields.json` and `profiles.json`: data for the HUD
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

To add a problem, copy a `@case` function and change four things: the geometry builder, the species, the `bcs` dict and the continuation path.

## Layout
`fem.py` holds the shapes and quadrature. `mesh.py` reads gmsh output. `geometry.py` builds geometries in gmsh. `bc.py` defines the BC types. `model.py` holds the weak form. `solver.py` does assembly, Newton and continuation. `analysis.py` computes fluxes, forces, profiles and adjoint sensitivities. `output.py` writes results. `runner.py` runs a case from start to finish. `hud/` is the web dashboard (`python hud/server.py` starts it at http://localhost:8765).
