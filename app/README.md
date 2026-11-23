# pnpflow app (first version)

The desktop front end: a FreeCAD-style shell (docs/PLAN.md §1, DESIGN.md) for building,
solving and reading pnpflow models. It runs in a browser today through a small dev server;
the same `ui/` folder becomes the Tauri 2 window later.

## Run it

```bash
# from the repo root, inside the pnpflow conda env
python app/server.py            # http://localhost:8766
# options: --port 8766 --host 127.0.0.1 --results DIR --python EXE
```

Open **http://localhost:8766** in a browser. Useful links:

| link | shows |
|---|---|
| `http://localhost:8766/` | Start page: templates, recent runs, examples |
| `http://localhost:8766/?run=sphere_demo&wb=results&view=field` | the charged sphere, mirrored about r = 0 |
| `http://localhost:8766/?new=charged_sphere` | a new project from the charged-sphere template (press F8 to solve) |
| `...&theme=light`, `...&ptab=view`, `...&menu=view`, `...&dialog=help` | light theme, View tab, an open menu, the shortcut list |

URL parameters: `run`, `new`, `wb` (geometry, mesh, materials, boundaries, equations, study,
results), `view` (field, profiles, checks, mesh, figures, log), `ptab` (view, data), `theme`.

The old HUD (`hud/server.py`, port 8765) is untouched and keeps working.

## What works

- **Shell**: menu bar (File, Edit, View, Model, Study, Results, Tools, Help) with keyboard
  navigation (F10, arrows, Esc); workbench tabs in construction order with status dots
  (shape and colour: filled = complete, ring = needs input, diamond = warning, square = error);
  one toolbar row per workbench (SVG icons, tooltips with name and shortcut, disabled tools say
  why); combo panel with the model tree and a View / Data property editor; document tabs;
  status bar.
- **Start page**: templates from `run.py --list` (charged sphere first), recent runs with
  thumbnails, status word + colour and date, examples (`val_*`) with their error against the
  exact solution, a filter (`/`).
- **Data tab**: case parameters with defaults and help text from `/api/cases`, units parsed
  from the help text (`[mV]`), changed values marked, invalid values explained, reset.
- **View tab**: field, colormap (+ reverse), range (auto all / auto visible / manual), log
  scale, mesh edges, boundaries, contours, axis ticks, vector arrows, streamlines, mirror.
- **Field map** (WebGL, ported from the HUD): mirrors `"symmetry": "axisym"` data about r = 0
  (radial vector components change sign), axis names from `coord_names`, cmasher colormaps
  from `ui/data/colormaps.json` (cosmic for sequential data, a diverging map only when the
  data really take both signs), contours, probe, pan/zoom, PNG export.
- **Results**: field map, profiles (Plotly in the DESIGN tokens, house-style cosmic lines),
  checks with pass / warning / fail / info as words, Newton convergence, integrals, notes,
  mesh views A/B with what to check, mesh diagnose with fixes, figure gallery (figures/ and
  plots/) with a lightbox, log.
- **Solve**: F8 or the Solve button launches the template via `POST /api/launch` with the
  changed parameters, then polls every 1.5 s: per-stage progress bars, health badge with the
  numbers on hover, live log-residual plot, log tail, Stop. It uses `fraction`, `health` and
  `tells` from progress.json when present; otherwise health is derived from `history` (and
  says so) and the current stage bar is indeterminate. A run whose progress.json says
  "solving" but has not changed for 10 minutes, with no process behind it, is shown as
  "stopped updating".
- **Theme**: dark by default, light when the OS prefers light, manual override in View.
- **Keyboard**: `?` lists everything. 1–7 workbenches, F8 / Shift+F8 solve / stop, N new,
  O open, R refresh, T theme, U units, X expert, [ panel, J/K documents, F G E B C S M P on the
  field map.

## Stubs and limits

- **Units toggle** (status bar) is a visual stub labelled "preview": it switches the label,
  not the values. Conversion arrives with the project file (PLAN Phase 1).
- **Geometry, Mesh, Materials, Boundaries, Equations** show what the template or run defines;
  editing them needs the project file. Their editing tools are present but disabled, and the
  tooltip says why.
- **Beginner / expert** changes how much is shown (help lines vs. raw names, derived
  dimensionless groups, the command line); it does not hide physics yet.
- A project is a template plus parameters; nothing is saved besides the run folder.

## Files

```
app/server.py                  dev server: imports hud/server.py, subclasses its handler,
                               serves ui/ and adds figures/*.png, svg/*.svg (strict CSP),
                               mesh/view_a.png, mesh/view_b.png, mesh/diagnose.json
app/ui/index.html              shell markup
app/ui/css/app.css             DESIGN.md tokens (dark + light) and all components
app/ui/js/bridge.js            the only backend module: Tauri invoke or fetch()
app/ui/js/app.js               shell: state, commands, menus, toolbar, tree, properties,
                               documents, status bar, polling, keyboard, URL state
app/ui/js/start.js             Start page
app/ui/js/pages.js             Geometry, Mesh, Materials, Boundaries, Equations, Study pages
app/ui/js/results.js           Results workbench views
app/ui/js/progress.js          run state, health, per-stage bars (never invents progress)
app/ui/js/fieldview.js         field viewer (from the HUD) + mirror, contours, coord names
app/ui/js/glfield.js           mesh prep, point location, contours, WebGL renderer
app/ui/js/colormaps.js         loads data/colormaps.json; fallback in colormaps_fallback.js
app/ui/js/plots.js             Plotly theme from the tokens, cosmic line colours
app/ui/js/icons.js             hand-written SVG icon set (one stroke style)
app/ui/js/util.js              DOM and number helpers
app/ui/data/colormaps.json     cmasher tables (written by the post-processing step)
app/ui/vendor/plotly.min.js    copy of hud/static/vendor/plotly.min.js
app/src-tauri/                 Tauri 2 shell (UNBUILT, see below)
```

## Tauri 2 shell: unbuilt

`app/src-tauri/` holds a small Tauri 2 project: `tauri.conf.json` (frontendDist `../ui`,
`withGlobalTauri` so `bridge.js` finds `window.__TAURI__.core.invoke`), `src/main.rs` with
commands that mirror the HTTP API (`list_runs`, `run_listing`, `read_run_file`,
`read_run_file_base64`, `read_run_log`, `list_cases`, `launch_run`, `stop_run`) and the same
name / path checks as the server.

**It has never been compiled** (needs the Rust toolchain and, on Linux, the webkit2gtk dev
libraries). Expect small fixes on the first build.

Prerequisites: Rust via rustup, the Tauri CLI (`cargo install tauri-cli --version "^2"`), and
on Linux the webkit2gtk-4.1 dev packages (see the Tauri 2 prerequisites page).

Then, from `app/src-tauri`:
```bash
export PNPFLOW_ROOT="$(git rev-parse --show-toplevel)"
export PNPFLOW_PYTHON="python"     # the pnpflow conda env must be active
cargo tauri dev
```
First-build checklist: generate icons (`cargo tauri icon some.png`) before bundling; check
that Plotly runs under the CSP (it is allowed `unsafe-eval`).
