# UI critique answers (2026-11-24)

From `/impeccable critique` of Home, Boundaries and Results > Checks (25/40; report in
`.impeccable/critique/2026-11-25T02-24-36Z__app-ui-index-html.md`).

| question | answer |
|---|---|
| area first | plain language (conditions in words, value/limit as words, labels on the diagram) |
| scope | all 5 priority issues (Home loading, stale spinner, Checks verdict, colour discipline, plain language) |
| Checks verdict | one line + warning count, e.g. "Converged · 1 warning"; the optional steady pre-solve row is marked optional with "transient run unaffected" |

## Re-run (2026-11-24, 25/40, `.impeccable/critique/2026-11-25T02-47-09Z__app-ui-index-html.md`)

| question | answer |
|---|---|
| area first | units + wording: show "75 mV (Φ = 2.92)" following the unit toggle; one vocabulary for conditions in diagram and table |
| wording | a boundary with no ion flux and no surface charge reads **"no flux, no charge"** (diagram); table: "no flux" / "no charge" |
| scope | top 3: units, wording, Home layout (run thumbnails on project cards, recent first, compact list past ~6) |

## Third run (2026-11-24, 25/40, `.impeccable/critique/2026-11-25T03-00-04Z__app-ui-index-html.md`)

| question | answer |
|---|---|
| SI / dimensionless toggle | **build the conversion**: dimensionless mode shows every value scaled (Φ = eφ/kT, lengths / a_ref, τ, c / c0) from the run's derived parameters |
| other issues | all four: notation (symbols in Notes, display names with raw key on hover in Beginner), Home onboarding (Create-mesh per empty project row, Run-validation button, Parameters collapsed on Home), Newton plot (fade connectors, same colour direction in both themes, bar scaled to the real step count, rejected steps marked), navigation cleanup ("Re-run in project" on runs, text on Results icons, drop stray "… tools" labels) |

## Fourth run (2026-11-24, 27/40, `.impeccable/critique/2026-11-25T03-19-56Z__app-ui-index-html.md`)

The user stopped here. Open issues, for a later pass: unit-aware help text and readable integral names
(P1), folding the 11 no-mesh projects on Home into one row (P1), Checks layout (P2), field-tool labels and
full timestamps in run names (P2), body font for worded limits (P3).
