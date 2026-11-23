# Case / result split, mesh tools, time stepping — the user's answers (2026-11-22)

Request (paraphrased): separate *case* and *result*. The case holds everything before the
run: the project, which figures to make (to be discussed further), the mesh, the parameters
(geometry, species, constants) and the simulation details. The result holds its own mesh copy,
plots, CSV and a text file of how the convergence went. Add control over how often time points
are saved and over the time stepping, and rework the time-stepping layout (typing values is
fine, but it is too spaced out). Rename the mesh checks to **Mesh Validation**, make it a
clickable bar, and note that problems can affect convergence. Add mesh toolbars with the
ability to add mesh refinement and increase element density. Make every clickable element
10 % bigger.

| question | answer |
|---|---|
| case layout on disk | **Nested**: `cases/<case>/case.json` + `cases/<case>/mesh/` (`case.msh` with named boundaries, two views, `validation.json`); runs in `cases/<case>/results/<run>/` with a frozen `case.json`, `mesh/`, plots, figures, csv, `convergence.txt`, summary, fields. One case, many runs. |
| Mesh Validation bar | **Summary bar that expands**: one full-width bar with the counts (warning / pass / info); click opens the list of checks with numbers and fixes; a warning or failure opens it automatically; a note says problems can slow or break convergence and bias results. |
| mesh tools | **All of**: Build mesh now (mesh + validate + views, no solve); Global density (one factor scaling all sizes); Boundary refinement (named boundary, size, layer, growth); Refinement regions (box, disk). **Plus**: more density around sharp areas, e.g. with a **spray-paint tool**. |
| saved time points | **Mode picker**: every Δτ (uniform), N log-spaced between τ_min and τ_end, or an explicit list; τ_end always saved. |
| figures in the case | to be discussed later; the case keeps a `figures` block (standard set for now). |
| units (follow-up) | a clickable unit menu on every parameter that has a unit (length nm/µm/mm/m and other units) to change the unit it is typed in; the value is converted, the template unit stays the stored one. |
| clear recents (follow-up) | a Clear recents button on the Start page hides the runs listed now (per browser); nothing is deleted; "Show N cleared" brings them back |
| create mesh (follow-up) | "Build mesh now" is now **Create mesh**, also on the Geometry page and toolbar; it always runs on the CPU (JAX_PLATFORMS=cpu), keeping the GPU for solves |
