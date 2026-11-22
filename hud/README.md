# pnpflow HUD

Offline web dashboard for pnpflow result folders (`results/<run>/`, format in
`docs/DATA_FORMAT.md`). Python stdlib server + vanilla JS. No build step, no CDN.
Plotly ships in `static/vendor/`.

## Launch

```bash
# from the pnpflow root (WSL: source ~/venvs/pnp/bin/activate first)
python hud/server.py              # -> http://localhost:8765/
python hud/server.py --port 9000 --host 0.0.0.0 --results /path/to/results --python /path/to/python
```

`--python` (or `PNPFLOW_PYTHON`) sets the interpreter for `run.py --list` and for launched runs.
If no results exist yet, `python hud/make_sample.py` writes a synthetic run to `results/_sample/`.

## API

| method | path | |
|---|---|---|
| GET | `/api/runs` | runs, newest first: state (from progress.json and the process), case, mesh size, drag, worst check |
| GET | `/api/run/<name>` | which files and plots the run has |
| GET | `/api/run/<name>/<file>` | `summary.json`, `fields.json`, `profiles.json`, `progress.json`, `stdout.log`, `plots/*.png` (large files are gzipped) |
| GET | `/api/run/<name>/log?tail=N` | tail of `stdout.log` |
| GET | `/api/cases` | `python run.py --list`, or `{}` if run.py is missing or fails (the reason is in the `X-Cases-Error` header) |
| POST | `/api/launch` | `{case, name, params:{k:v}, overwrite?}` runs `python run.py <case> --name <name> --set k=v ...`, with output to `results/<name>/stdout.log` |
| POST | `/api/stop` | `{name}` stops a run that this server launched |

Run names, case names and keys are checked against regexes and resolved paths
must stay inside `results/` or `static/`, so path traversal is rejected.

## Features

- **Sidebar**: runs with done / failed / running badges, live stage / iteration / ‖R‖ from polling (every 2 s), filter, newest first.
- **Field**: WebGL rendering of the P1 triangulation. Colors are interpolated per fragment, so it stays smooth at 50k+ nodes.
  - Colormaps (viridis default; diverging RdBu centered at 0 for signed fields), colorbar with ticks and min/max.
  - Range: auto over all nodes, auto over the visible area, or manual. Log scale for positive fields.
  - Wheel zoom about the cursor, drag to pan. Opens zoomed to `focus_bbox`; *Fit all* shows the whole domain.
  - Hover readout of (y, z, r, value). Click pins a probe that lists every field at that point.
  - Overlays: mesh edges, boundary edges colored by name with a legend, axis ticks, vector arrows (any entry in `vectors`), evenly spaced streamlines of `u`.
  - The region outside the mesh is hatched. PNG export.
- **Profiles**: Plotly plots of `profiles.json`. Pick a line and several variables; split or overlaid subplots, log y/x, normalize. *Compare with* overlays the same line from another run (dashed).
- **Summary**: cards for
  - failure banner with the traceback
  - overview with the case doc and species
  - parameters with `param_info` tooltips
  - mesh stats with a quality histogram
  - Newton convergence (log residual per stage, step norm dotted, stage separators)
  - checks colored pass / warn / fail / info
  - forces split into hydro and electric (drag and lift)
  - species fluxes per boundary with the net value
  - charge balance and other integrals
  - sensitivity matrix with diverging bars (hover a cell for the elasticity)
  - boundary conditions and notes
- **Gallery**: thumbnails of `plots/*`, with a lightbox (← → Esc).
- **Progress & log**: live stage bar, iteration stats, live convergence plot, `stdout.log` tail with follow, Stop button.
- **New run**: case list from `run.py --list`. Parameters are prefilled with defaults and info text; changed values are highlighted and invalid ones flagged. Shows a preview of the command line. After launch it switches to live progress.
- Dark theme by default, light theme with `t`. Keyboard: `1`–`5` tabs, `j`/`k` runs, `/` filter, `n` new run, `f`/`g` fit/focus, `e`/`b`/`s` overlays, `?` help.
  View state is kept in the URL hash and localStorage.

## Files

`server.py` · `make_sample.py` · `static/index.html` · `static/css/hud.css` ·
`static/js/{app,api,util,colormaps,glfield,fieldview,profiles,summary,panels,plots}.js` ·
`static/vendor/plotly.min.js`
