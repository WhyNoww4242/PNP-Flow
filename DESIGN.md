---
name: pnpflow
description: Desktop model builder and results viewer for PNP + flow electrochemistry
colors:
  bg: "#0d1016"
  surface: "#141922"
  surface-raised: "#1a202b"
  surface-hover: "#212836"
  input: "#0f131a"
  border: "#252c39"
  border-strong: "#343d4e"
  text: "#dfe5ee"
  text-dim: "#8b96a8"
  text-faint: "#5d6778"
  accent: "#5cb8ff"
  accent-ink: "#06121f"
  ok: "#51cf66"
  warn: "#fcc419"
  bad: "#ff6b6b"
  info: "#74c0fc"
  light-bg: "#f3f5f8"
  light-surface: "#ffffff"
  light-surface-raised: "#f7f8fa"
  light-surface-hover: "#edf0f5"
  light-border: "#dde2ea"
  light-border-strong: "#c7cfdb"
  light-text: "#1c2330"
  light-text-dim: "#5d6879"
  light-accent: "#0b6fd0"
  light-accent-ink: "#ffffff"
  light-ok: "#2b8a3e"
  light-warn: "#c48a00"
  light-bad: "#d6336c"
typography:
  title:
    fontFamily: "Inter, Segoe UI, system-ui, -apple-system, sans-serif"
    fontSize: "15px"
    fontWeight: 600
    lineHeight: 1.3
  body:
    fontFamily: "Inter, Segoe UI, system-ui, -apple-system, sans-serif"
    fontSize: "13px"
    fontWeight: 400
    lineHeight: 1.45
  label:
    fontFamily: "Inter, Segoe UI, system-ui, -apple-system, sans-serif"
    fontSize: "11.5px"
    fontWeight: 500
    lineHeight: 1.3
  data:
    fontFamily: "JetBrains Mono, Cascadia Mono, SF Mono, Consolas, monospace"
    fontSize: "12px"
    fontWeight: 400
    lineHeight: 1.4
    fontFeature: "tnum"
rounded:
  sm: "5px"
  md: "8px"
spacing:
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "24px"
components:
  button:
    backgroundColor: "{colors.surface-raised}"
    textColor: "{colors.text}"
    rounded: "{rounded.sm}"
    height: "28px"
    padding: "0 12px"
  button-hover:
    backgroundColor: "{colors.surface-hover}"
  button-primary:
    backgroundColor: "{colors.accent}"
    textColor: "{colors.accent-ink}"
    rounded: "{rounded.sm}"
    height: "28px"
    padding: "0 12px"
  input:
    backgroundColor: "{colors.input}"
    textColor: "{colors.text}"
    typography: "{typography.data}"
    rounded: "{rounded.sm}"
    height: "26px"
  panel:
    backgroundColor: "{colors.surface}"
    rounded: "{rounded.md}"
  rail-item-selected:
    backgroundColor: "{colors.surface-hover}"
    textColor: "{colors.text}"
---

# Design System: pnpflow

## Overview

**Creative North Star: "The Bench Instrument"**

pnpflow should feel like a well-made piece of lab equipment: calm, exact, and built for long
sessions. The screen belongs to the model and the data; the interface around them is quiet,
dense enough to keep everything in view, and always honest about state (is the mesh good, is
the solver converging, did the checks pass).

It follows impeccable's **Operate** mode: scanability, consistency and native desktop
behaviour come before expression. The look carries over from the existing HUD (`hud/`),
which the user likes: dark blue-grey neutrals, one sky-blue accent, Inter for text and a
monospace face for numbers. The rejected reference is COMSOL's dated, crowded chrome.

**Key Characteristics:**
- Standard desktop shell: menu bar (File, Edit, View, Model, Study, Results, Help), left icon
  rail, model tree, central canvas, right settings panel, status bar with live progress.
- Balanced density: compact panels and tables; roomier step-by-step set-up screens.
- Dark by default, follows the OS when it says light; both themes designed, not inverted.
- Data colours (cmasher colormaps) are separate from UI colours.

## Colors

Restrained: tinted blue-grey neutrals and one accent. State colours appear only where they
carry meaning.

### Primary
- **Sky Signal** (#5cb8ff dark / #0b6fd0 light): primary actions (Solve), selection, focus
  rings, the active tree item, progress bars. Nothing decorative.

### Neutral
- **Night Bench** (#0d1016): app background. **Panel** (#141922) and **Raised Panel**
  (#1a202b): the second neutral layer for rail, tree, inspector and status bar.
- **Hairline** (#252c39) and **Strong Hairline** (#343d4e): dividers and input borders.
- **Ink** (#dfe5ee): main text. **Dim Ink** (#8b96a8): secondary text (≈6:1 on the
  background). **Faint Ink** (#5d6778): disabled and decorative only (below 4.5:1, never
  for information).

### State
- **ok** (#51cf66), **warn** (#fcc419), **bad** (#ff6b6b), **info** (#74c0fc): check results,
  run health badges and validation messages. Always paired with a word or icon, never colour
  alone.

### Data colours (not UI colours)
- Fields use **three cmasher maps only**: `cmr.cosmic` (potential, concentrations, mesh
  quality), `cmr.ember` (magnitudes |E|, |u|), `cmr.bubblegum` (charge density ρ, ρ_f).
  No diverging maps; a field that takes both signs uses its map over the real range, an
  all-negative field the reversed map. **Never viridis** or any other matplotlib default
  (RdBu, magma, plasma, jet, …) — in figures, legacy plots, the app, the HUD or thumbnails.
- Plot frame: only the left and bottom axis lines, each ending in an arrowhead; no box, no
  colour-bar outline, no grid, frameless legends.
- Line colours come from `apply_custom_style`: `cmr.cosmic_r` sampled over (0.2, 0.75).
- Boundary groups in the mesh view use a fixed categorical set so `electrode` has the same
  colour in every view.

### Named Rules
**The One Signal Rule.** The accent marks only what the user can act on or what is selected.
If two things on one screen are accent-coloured, one of them is wrong.

## Typography

**UI font:** Inter (fallback Segoe UI, system-ui). **Numbers:** JetBrains Mono with tabular
figures, used for values, units, coordinates and logs only, never for decoration.

### Hierarchy
- **title** (600, 15px): panel and dialog titles.
- **body** (400, 13px, line-height 1.45): everything else in panels.
- **label** (500, 11.5px): field labels, tree group headers, axis captions. Sentence case,
  no letter-spacing tricks, no uppercase kickers.
- **data** (400, 12px mono, tabular): values in inputs, tables, logs, the status bar.
Fixed px scale (≈1.15 ratio); no fluid type in the app.

## Layout

- Shell, modelled on the user's FreeCAD reference (`docs/Screenshot 2026-11-21 233015.png`,
  mapping in `docs/PLAN.md` §1): menu bar (28px); workbench tabs, one per construction step;
  one toolbar row for the active workbench; left combo panel (~280px, resizable) with the
  model tree on top and the property editor below; canvas (flexible); document tabs at the
  bottom; status bar (24px) with the units toggle, beginner/expert, stage progress, health
  badge and residual.
- Start page: new project from templates (with real geometry previews), recent projects with
  result thumbnails and status, examples (the validation cases).
- The rail holds SVG icons (one consistent icon set, e.g. Lucide, plus custom icons for
  electrode, ion and mesh). Hover shows a tooltip with the name and its keyboard shortcut.
- Model tree order: Geometry, Mesh, Materials, Boundary Conditions, Governing Equations,
  Study, Results. Each node shows a status dot (complete / needs input / error) and the tree
  doubles as the step-by-step guide.
- Spacing steps 4/8/12/16/24px. Data tables may run wide; prose stays under 75ch.

## Elevation & Depth

Flat, tonal layering: depth comes from the two neutral surface levels and hairlines. Shadows
only on floating layers (menus, tooltips, dialogs): `0 8px 30px rgba(0,0,0,.45)` dark,
`0 8px 30px rgba(30,40,60,.15)` light.

## Shapes

Small radii: 5px for controls, 8px for panels and dialogs. 1px hairline borders. No pills
except status badges.

## Components

### Buttons
Default: raised panel with a strong hairline, 28px high. Primary (Solve, Run): accent fill.
Every button has hover, focus-visible (2px accent ring), active, disabled and loading states.

### Inputs / Fields
Mono values with the unit shown inside the field on the right, in Dim Ink. Changed-from-default
values are marked; invalid values show the reason under the field. Unit toggle (SI ↔
dimensionless) in the panel header.

### Navigation
Menu bar + icon rail + model tree. Keyboard: the HUD's shortcuts carry over (`1–5` views,
`j/k`, `/` filter, `n` new, `f/g` fit/focus, `e/b/s` overlays, `?` help).

### Convergence and progress (signature component)
A per-stage progress bar, a live log-residual plot, and a health badge
(converging / struggling / stalled / failed) with the numbers behind it on hover: residual,
contraction ratio, line-search step, minimum concentration. Progress is never invented: when
the solver cannot tell, the bar is indeterminate.

### Mesh view
The HUD's WebGL triangulation view carries over: boundaries coloured by group with a legend,
quality colouring, wheel zoom, pan, probe on click.

## Do's and Don'ts

### Do:
- Show checks and convergence next to every result.
- Keep the accent for actions, selection and progress.
- Give each control all its states (hover, focus, active, disabled, loading, error).
- Keep motion to 150–250 ms, and only to show state changes.

### Don't:
- Use viridis or other matplotlib default colormaps; use cmasher.
- Use monospace, uppercase kickers, numbered section labels or side-stripe borders as decoration.
- Use Unicode glyphs or emoji as icons; use the SVG icon set.
- Nest cards inside cards or build dashboards out of identical metric tiles.
- Copy COMSOL's crowded toolbars; prefer the menu bar and the context panel.
