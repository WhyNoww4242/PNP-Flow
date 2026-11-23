# Task: four pore-charging DNS runs to τ ≈ 0.53, auto-remesh on failure, 6-panel figure

For Claude Code, run from the repo root (`~/Code/PNP-Flow`). Read `AGENTS.md` first
(especially §3 Cases, §9 figure style, §10 rules), then `docs/pore_dns_plan.md` and
`docs/pore_dns_answers.md`. Load the `/pore-charging-mu-compare` skill before the figure step
and follow it where it is more specific than this file.

Work in the `pnpflow` conda env. Report to the user after each case finishes or fails.

## 1. Goal

Run the four DNS-matched pore cases with pnpflow from τ = 0 to τ_end = 0.535, mesh
density 0.95, one at a time on the GPU. If a run fails, remesh (rules in §4) and run it
again automatically. Then make the 6-panel comparison figure (§6) against Pawel's DNS.

| order | case folder | DNS folder | alpha_in → alpha_out |
|---|---|---|---|
| 1 | `cases/pore_dns_narrow_coarse` (create it, §2) | `poreRadius100_100` | 1 → 1 |
| 2 | `cases/pore_dns_converging_coarse` | `poreRadius200_100` | 2 → 1 |
| 3 | `cases/pore_dns_diverging_coarse` | `poreRadius100_200` | 1 → 2 |
| 4 | `cases/pore_dns_wide_coarse` | `poreRadius200_200` | 2 → 2 |

## 2. Case setup (before any solve)

1. Create the narrow case: `python tools/make_dns_cases.py --coarse narrow`
   (gives `h_wall = 0.15` like the other three).
2. In all four `case.json` files set exactly these, change nothing else:
   - `mesh.density = 0.95`
   - `params.tau_end = 0.535`
   - `params.dns_times = 1`, `params.dns_init = 1`
   - `time.save.mode = "template"` (so the saved times are the DNS folder times
     τ_k = 0.06685 k, k = 1..8, plus τ_end; this puts snapshots exactly on τ = 0.1337 and
     0.5348, the DNS times t = 0.01 s and 0.04 s). The converging case currently has
     `"uniform"`, dtau 0.01: change it and tell the user (uniform 0.01 would save ~54
     snapshots, miss the DNS times, and make `fields_time.json` several hundred MB).
3. Keep the other parameters as they are (L = 10 µm, c0 = 0.948 mM, ε = 7.0832e-10,
   D = 1.337e-9, Φ_w = 10 mV, phi_end_mV = 10, sdl_radius = sdl_length = 0 → defaults).
4. Do not delete or overwrite existing runs (e.g. converging `..._261123_1018`, τ = 0.14 at
   density 1.2). New runs get new names (§5).

Why 0.535: the figure's τ = 0.13 and 0.53 are the DNS folders k = 2 and k = 8
(τ = 0.1337, 0.5348). τ_end slightly above 0.5348 avoids a rounding miss on the last DNS time.

## 3. Mesh check per attempt

Before each solve: `JAX_PLATFORMS=cpu python run.py --case cases/<case> --mesh-only`.
Record triangles, dofs, h_min, quality_min and the diagnose result from
`cases/<case>/mesh/status.json`; look at `mesh/view_a.png` and `view_b.png` (AGENTS.md §7).
A diagnose `fail` (today: EDL resolution on `pore_wall`, min ≈ 6 elements across 3δ) is
logged and reported but does **not** stop the run. A crash of the mesh build counts as a
failure (treat as `solver`, §4).

## 4. Failure detection and remesh rule ("smart by failure type")

`run.py` exits 0 even when the solve fails, so classify each attempt from the exit code,
`stdout.log` and `summary.json` of its run folder:

| class | detected by | next density |
|---|---|---|
| `ok` | `converged: True` in stdout, `summary.solver.converged` true, `summary.transient.t_reached ≥ t_end·(1 − 1e-9)`, no NaN/inf in the log | done |
| `oom` | exit code −9/137 or killed with no `converged:` line, `MemoryError`, `RESOURCE_EXHAUSTED`, `out of memory`, PARDISO memory error, or an OOM-kill entry in `journalctl -k`/`dmesg` | coarser: next untried of 0.90, 0.85, 0.80 |
| `solver` | `time stepping: ok=False`, `converged: False`, max_rejects reached, NaN/inf, `t_reached < t_end`, a Python traceback in `summary.error` | finer: next untried of 1.00, 1.05, 1.10 |

- Start every case at 0.95. At most 5 attempts per case.
- If the needed direction is exhausted, or the classes conflict (e.g. `solver` at 0.95 then
  `oom` at 1.00), stop that case, report it with both logs, and go on to the next case.
- A failed steady equilibrium solve after the time stepping is only a warning, not a failure.
- Exit code −15 (SIGTERM) means someone stopped it: do not retry; ask the user.
- Changing `mesh.density` changes the mesh key, so `run.py --case` rebuilds the case mesh by
  itself. Restore `mesh.density` in `case.json` to the density of the successful attempt.

## 5. Launch and monitor

Write `tools/run_dns_batch.py` (Python, stdlib only) implementing §2–§4 for the four cases
in order, so the whole queue runs without an agent attached:

- per attempt: edit `case.json` density, run the mesh check, then
  `JAX_PLATFORMS=cuda python -u run.py --case cases/<case> --name tau053_d<density×100>`
  (e.g. `tau053_d095`), stdout to that run's `stdout.log`;
- classify (§4), pick the next density or move on;
- append one line per event to `results/dns_tau053_queue.log` and keep a machine-readable
  `results/dns_tau053_status.json` (per case: attempts, density, class, run folder,
  wall time, dofs, diagnose status);
- at the end, run the figure script (§6).

Test the classifier on the existing run folders first (`..._1018` must classify `ok`
for its own t_end; the two killed runs `pore_dns_converging` and `..._1017` must classify
as exit −15 → stop). Then launch detached so it survives this session:

```bash
nohup python -u tools/run_dns_batch.py > results/dns_tau053_nohup.log 2>&1 &
```

Monitor by polling (every ~5–10 min, never a blocking command longer than the tool
timeout): the queue log, the current run's `progress.json` (stage, τ, dt, Newton residual,
health) and `nvidia-smi` / `free -g`. Expected time: the converging run took 2622 s for
76 BDF2 steps to τ = 0.14 at 241k triangles; at density 0.95 (~152k triangles) and
τ_end = 0.535 expect roughly 40–90 min per case, wide the longest.

## 6. Comparison figure (like the user's reference figure)

Extend `tools/figure_dns4.py` (or add `tools/figure_dns6.py`, same helpers) to make a
**3 × 2** figure, columns τ = 0.1337 and 0.5348 (DNS t = 0.01 s, 0.04 s; titles
`τ = 0.13`, `τ = 0.53`):

| row | y | x | pnpflow source |
|---|---|---|---|
| a), b) | μ̂ = ρ + 2Φ on the centreline | Z from −0.5 to 1 (SDL included) | axial line `mu_axis` |
| c), d) | Φ̂ at Z = 0.5 | R/α(Z) from 0 to 1 | radial line `phi` |
| e), f) | ρ̂ at Z = 0.5 | R/α(Z) from 0 to 1 | radial line `rho` |

- pnpflow = lines, DNS = squares (every ~50th point, as now). DNS files per time folder:
  `lineYe_nPlus_nMinus_ePhi.xy` (centreline) and `lineXm5_nPlus_nMinus_ePhi.xy` (Z = 0.5),
  converted as in `figure_dns4.load_dns_line` (pawel.py constants).
- DNS data: `--dns DNS_compare/transportFolder` (in the repo root, git-ignored; the script's
  default is an old WSL path). Also use
  `DNS_compare/compare_mu.py`, `extract_dns.py` and `pawel.py` as references for the DNS
  units and line extraction.
- Pick the runs explicitly (the new `tau053_*` folders), not "newest profiles.json".
- Colours and legend as in the reference: one colour per geometry from
  `cmr.bubblegum_r` 0.1–0.9 (diverging lightest → converging darkest), legend on the right
  with "<Geometry> pnpflow" (line) and "<Geometry> DNS" (square) per case and a small grey
  pore schematic next to each pair. If the user's skill says the lines should be the ROM
  (pawel.py) instead, follow the skill and ask before dropping pnpflow lines.
- House style (AGENTS.md §9): Times/usetex, arrow axes, no grid, panel labels a)–f),
  ticks ≥ 18 pt, 300 dpi PNG + SVG to `results/pore_dns_compare/figures/figureDNS6.png`
  and `svg/`. Render once, inspect it (clipping, overlaps, legend), then finalise.
- Also write the combined CSV and the error table
  (`results/pore_dns_compare/csv/dns_vs_pnpflow*.csv`) with ρ added, and print
  max |pnpflow − DNS| per case, time and line.

## 7. Final report to the user

Per case: final density and number of attempts (with each failure class), triangles / dofs,
mesh diagnose status, wall time, t_reached, the checks from `summary.json` (positivity,
time integration), and the max |pnpflow − DNS| for μ̂, Φ̂, ρ̂ at both times. Then the
figure path. List every change made to case files and code. Keep the open items from
`docs/pore_dns_plan.md` (sharp vs rounded mouth, pore-end φ, CODATA vs pawel.py constants)
as notes; do not change them.
