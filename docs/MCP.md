# pnpflow agent protocol (MCP)

Status: **specification, not built yet.** This file defines how an AI agent (Claude Code or
any MCP client) builds, checks, runs and reports a pnpflow model. The desktop app and the MCP
server share the same Python functions, so anything the agent can do, the GUI can do, and
the other way round.

- Server: `python -m pnpflow.mcp` (stdio transport, Python `mcp` SDK, installed by
  `environment.yml`).
- Geometry and meshing go through **gmsh** (its Python API or `.geo` scripts). gmsh is the
  geometry interface for agents; there is no other geometry kernel.
- Every tool returns JSON. Tools that make pictures also return the PNGs as MCP image content,
  so the agent can look at them.
- State lives in files, the same files the GUI reads: the project file, the `.msh` mesh, and
  `results/<run>/` (`docs/DATA_FORMAT.md`). Nothing is held only in server memory.

## 1. Tools

### Project
| tool | input | returns |
|---|---|---|
| `project_new` | `name`, optional `template` | path of a new project file |
| `project_read` | `path` | the project JSON |
| `project_update` | `path`, `patch` (JSON merge patch) | the updated JSON + validation messages |
| `project_validate` | `path` | `checks[]` with `pass/warn/fail/info` (same shape as `summary.json`) |

### Geometry (gmsh)
| tool | input | returns |
|---|---|---|
| `geometry_templates` | none | the templates and their parameters (channel, cylinder in box, electrode pair, nanopore, …) |
| `geometry_from_template` | `template`, `params` | entity list, physical groups, bounding box |
| `geometry_from_script` | `script`, `language: "python" \| "geo"`, `params` | same as above, plus the gmsh log on error |
| `geometry_import` | `path` (`.step .stp .brep .iges .geo .msh`) | same as above |
| `geometry_info` | none | entities, physical groups, areas/lengths, unnamed entities |

### Mesh
| tool | input | returns |
|---|---|---|
| `mesh_generate` | `sizes` (see §3), `order` (default 2) | mesh stats (`Mesh.stats()`), path of the `.msh` |
| `mesh_render` | `views` (default: the two in §4), `color_by: "group" \| "quality" \| "size"` | two PNGs + what each shows |
| `mesh_diagnose` | optional `delta` | `checks[]` (§5) and a suggested fix for every failure |

### Runs
| tool | input | returns |
|---|---|---|
| `run_start` | `project`, `name` | run name; starts `run.py` as a process |
| `run_status` | `name` | `progress.json` plus derived fields (§6) |
| `run_log` | `name`, `tail` | last lines of `stdout.log` |
| `run_stop` | `name` | final state |

### Results
| tool | input | returns |
|---|---|---|
| `results_summary` | `name` | `summary.json`, trimmed to checks, integrals and sensitivities |
| `results_plot` | `name`, `kind` (`field \| profile \| convergence \| iv \| compare`), options | PNG + SVG paths, images inline |
| `results_report` | `name` | path of an interactive single-file HTML report |
| `results_export` | `name`, `what` (`csv \| vtu \| script`) | file paths; `script` reproduces the run in Python |

## 2. How an agent builds a geometry in gmsh

Follow these steps in order. Do not skip §4 and §5.

1. **Pick the frame.** 2D planar uses `(y, z)` stored as gmsh `(x, y)`. Axisymmetric (planned)
   uses `(r, z)` with the symmetry axis on `x = 0` and `r ≥ 0`. Lengths are in units of the
   project's length scale `a` (or in SI, if the project's unit mode is SI; the server rescales).
2. **Build with the OCC kernel** (`gmsh.model.occ`): `addRectangle`, `addDisk`, `addPolygon`
   (via `addCurveLoop` + `addPlaneSurface`), then booleans `cut`, `fuse`, `fragment`.
   Use `fragment` when there are several regions (electrolyte + porous electrode) so they
   share nodes on their interface. Call `occ.synchronize()` before naming anything.
3. **Name every boundary.** Each boundary curve goes in exactly one dim-1 physical group.
   Each region goes in a dim-2 physical group. Names are lower_snake_case and describe the
   physics: `electrode`, `counter`, `wall`, `inlet`, `outlet`, `symmetry`, `axis`, `bulk`;
   regions `electrolyte`, `porous`. The solver ignores unnamed curves, which silently
   makes them blocking/insulating — `mesh_diagnose` fails on them.
   Find curves by position, as `geometry.py` does: `getBoundary` then `getCenterOfMass`.
4. **Size the mesh from the physics.** Near every charged or reacting boundary the element
   size must resolve the Debye length: `h_wall ≤ λ_D / 8` (in units of `a`: `delta / 8`),
   kept for `layer ≈ 3 λ_D` from the wall, then growing to `h_far` by `grow`. Use a
   `Distance` → `Threshold` field combined with `Min`, and turn off
   `MeshSizeFromPoints`, `MeshSizeFromCurvature` and `MeshSizeExtendFromBoundary`
   (see `geometry._wall_size_field`).
5. **Generate** with `Mesh.Algorithm = 6`, `generate(2)`, `setOrder(2)`,
   `optimize("HighOrder")` (see `geometry._finish`).
6. **Render (§4) and look at both images.** Then **diagnose (§5)**. Fix and repeat until
   there are no failures. Only then save the `.msh` (format 4.1) next to the project file and
   reference it from the project.

A geometry script gets `gmsh` (already initialised) and `params`. It builds the shapes and
the physical groups, and leaves meshing to the server. Example:

```python
# electrode pair in a channel, 2D, lengths in units of a
occ = gmsh.model.occ
L, H, w = params["length"], params["height"], params["electrode_width"]
chan = occ.addRectangle(0, 0, 0, L, H)
occ.synchronize()
# split the bottom wall so each electrode is its own curve
p1 = occ.addPoint(L/2 - 1.5*w, 0, 0); p2 = occ.addPoint(L/2 - 0.5*w, 0, 0)
p3 = occ.addPoint(L/2 + 0.5*w, 0, 0); p4 = occ.addPoint(L/2 + 1.5*w, 0, 0)
occ.fragment([(2, chan)], [(0, p) for p in (p1, p2, p3, p4)])
occ.synchronize()
groups = {"anode": [], "cathode": [], "wall": [], "inlet": [], "outlet": []}
for _, c in gmsh.model.getBoundary([(2, chan)], oriented=False):
    x, y, _ = occ.getCenterOfMass(1, c)
    if abs(x) < 1e-9: groups["inlet"].append(c)
    elif abs(x - L) < 1e-9: groups["outlet"].append(c)
    elif abs(y) < 1e-9 and abs(x - (L/2 - w)) < 0.5*w: groups["anode"].append(c)
    elif abs(y) < 1e-9 and abs(x - (L/2 + w)) < 0.5*w: groups["cathode"].append(c)
    else: groups["wall"].append(c)
for name, tags in groups.items():
    gmsh.model.setPhysicalName(1, gmsh.model.addPhysicalGroup(1, tags), name)
gmsh.model.setPhysicalName(2, gmsh.model.addPhysicalGroup(2, [chan]), "electrolyte")
```

## 3. Mesh size input

```json
{"h_far": 4.0,
 "refine": [{"boundaries": ["electrode"], "h": "auto", "layer": "3*delta", "grow": 20.0}],
 "boxes": [{"y": [-2.5, 2.5], "z": [-12, 2.5], "h": 0.25}]}
```
`"auto"` means `delta / 8`. Expressions may use project parameters.

## 4. Rendering: always two views

One picture of a mesh hides problems. `mesh_render` always makes two images, and the agent
must look at both before going on.

| | 2D mesh | 3D mesh (planned) |
|---|---|---|
| view A | the whole domain, boundaries coloured by physical group with a legend | isometric camera, surface mesh coloured by group |
| view B | zoom on the finest region (by default the smallest-`h` boundary), elements coloured by quality | a second camera at 90° to the first, plus a cut plane through the finest region |

What to check in the images: every boundary has the expected colour and name; the layer of
small elements hugs charged and reacting walls; the size grows smoothly with no sudden jumps;
no slivers; curved P2 edges follow arcs.

## 5. `mesh_diagnose` checks

Same `checks[]` format as `summary.json` (`name, value, limit, status, note`) plus `fix`.

| check | fails when | typical fix |
|---|---|---|
| unnamed boundary | a boundary curve is in no physical group | add it to a group |
| duplicate naming | a curve is in two groups | keep one |
| BC coverage | a group has no BC in the project, or a BC names a missing group | add BC / rename |
| EDL resolution | elements across `3 λ_D` at a charged wall `< 8` | lower `h_wall` or widen `layer` |
| min quality | min shape quality `< 0.2` (warn `< 0.3`) | lower `grow`, add a size box |
| size jump | neighbouring element size ratio `> 2` | smoother `grow` |
| inverted / bad P2 | any curved element with a non-positive Jacobian | `optimize("HighOrder")`, smaller `h` on that curve |
| connectivity | more than one disconnected fluid region | fix the booleans (use `fragment`) |
| far field | domain edge closer than `20 λ_D` to a charged wall (warn) | enlarge the box |
| size budget | dofs `> 2e6` (warn) | coarsen far field |

## 6. Live status (`run_status`)

The run writes `progress.json` (`docs/DATA_FORMAT.md`). `run_status` adds:
- `fraction`: progress through the continuation stages, `0–1`, from completed stages and the
  log of the residual drop in the current stage. Never invented: if the solver cannot tell, it is `null`.
- `health`: `converging` (residual falls each step, `alpha ≈ 1`), `struggling` (line search
  cutting `alpha < 0.25` or stage halvings), `stalled` (no decrease in 3 steps), `failed`.
- `tells`: the numbers behind `health` — last residual, contraction ratio `|R_k|/|R_{k-1}|`,
  `alpha`, minimum concentration, number of stage halvings.
The GUI shows the same thing: a progress bar per stage, a live log-residual plot and a
coloured health badge. Continuation is hidden while `health` is `converging` and opens by
itself otherwise.

## 7. Automatic graphs

Every finished run makes its standard figures without being asked: field maps of each
species, `phi`, `rho`, `|u|`; EDL profiles; boundary fluxes; Newton convergence; and, when
the study sweeps a parameter, the swept curve (e.g. I–V). They are written to
`results/<run>/figures/<name>.png` (300 dpi) and `results/<run>/svg/<name>.svg` in the house
style (`AGENTS.md` §9).
