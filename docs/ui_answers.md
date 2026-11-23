# pnpflow GUI — answers (2026-11-21)

Answers to the questions in `docs/ui_design.html`. Decisions drawn from them are in
`AGENTS.md` §6.

## Who and where
1. **Main user:** both experimental and computational users, with a beginner / expert switch.
2. **Operating systems:** Windows, Linux, macOS.
3. **Python/JAX install:** "can we use a yml?" → `environment.yml` (conda / mamba / micromamba).

## Physics input
4. **Units:** both SI and dimensionless, with a toggle.
5. **Features wanted next:** Butler–Volmer kinetics; applied potential or current;
   time-dependent (chronoamperometry, CV); EIS by small-signal linearisation;
   homogeneous reactions; axisymmetric (r–z); porous electrodes; 3D.
6. **Geometry:** parametric templates; CAD / mesh import; gmsh / Python scripts.
   > Use gmsh as the interface for the MCP. Leave instructions for how to create geometry,
   > instruct the agent how to construct the gmsh model, make sure it renders from 2
   > different angles so the mesh can be checked, and add a diagnose feature.
7. **Ions:** built-in library + custom entries, and allow custom / nondimensional ions.

## Workflow
8. **Model set-up:** model tree + settings panel, with step-by-step construction in the order
   Geometry, Mesh, Materials, Boundary Conditions, Governing Equations, …
9. **Continuation and sweeps:** automatic, shown only on failure, but with visual and numerical
   tells for convergence.
10. **Saved project:** one human-readable file (JSON or TOML).

## Results
11. **Results looked at first:** all of them: field maps, EDL profiles, current / I–V,
    boundary fluxes, forces, sensitivities, convergence, run-to-run comparison.
12. **Exports:** CSV; publication figures (PNG / PDF / SVG); a Python script that reproduces
    the run; a run report as a quick interactive HTML.

## Technology
13. **Rust GUI:** Tauri 2.
14. **GUI ↔ solver:** start `run.py` as a process; exchange files.
15. **impeccable:** install for this project; write PRODUCT.md + DESIGN.md.

## Look and feel
16. **Theme:** follow the OS, default dark.
17. **Density:** balanced.
18. **Carry over from the HUD:** its colours and type, the left menu (with SVG icons and the
    keyboard shortcut shown on hover), the overall view and the mesh view.
    Graph style: cmasher colormaps via `apply_custom_style` (below). Live status, progress bars
    and automatic graph generation are goals for AGENTS.md and the MCP protocol.

```python
def apply_custom_style(num_colors: int=4, cmap: str='cmr.cosmic_r', cmap_range: tuple=(0.2, 0.75)):
    cm = cmr.get_sub_cmap(cmap, *cmap_range)
    colors = [cm(i / (num_colors - 1)) for i in range(num_colors)]
    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=colors),
        'font.size': 24, 'axes.labelsize': 24, 'axes.titlesize': 24, 'figure.titlesize': 24,
        'axes.spines.top': False, 'axes.spines.right': False,
        'font.family': 'times-roman', 'text.usetex': True})
    return colors
```

19. **Other software:** COMSOL looks old and ugly. Want a standard File / Edit / View … menu bar
    across the top.

## Follow-up feedback (2026-11-22)

20. **Geometry tab:** it should hold all the geometry parameters itself; having the parameters
    in the panel below the tree is weird. → Each construction step page now edits its own
    parameters (`app/ui/js/params.js` decides the step from the parameter name); the panel under
    the tree is hidden on those pages and only shows on Results (View settings + run inputs
    grouped by step).
21. **Mesh tab:** every check should be its own bubble at the top, clickable, showing when it
    warns. → One bubble per diagnose check, coloured and iconed by status (fail, warning, pass,
    info; worst first). Clicking opens value, limit, note and fix; a warning or failure opens
    automatically.
22. **Project explainer:** wanted an interactive HTML of how the project works, especially
    where meshes and data are saved. → `docs/how_it_works.html`.
    *Superseded 2026-11-22: the bubbles became the expandable Mesh Validation bar
    (`docs/case_answers.md`).*
