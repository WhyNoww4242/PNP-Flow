# Plan: four pore-charging runs matched to the DNS

Status: **planned, not started** (2026-11-22). Answers: `docs/pore_dns_answers.md`.
Rule from the user: match the DNS, change nothing else, report every change and its check.

## Common parameters (every case)

```bash
JAX_PLATFORMS=cuda python run.py pore_charging --name pore_dns_<case> \
  --set alpha_in=<a0> --set alpha_out=<a1> \
  --set L_nm=10000 --set a_ref_nm=10 \
  --set phi_w_mV=10 --set c0_mM=0.948 --set eps=7.0832e-10 --set T=298.15 --set D=1.337e-9 \
  --set phi_end_mV=10 --set sdl_radius=2 --set sdl_length=501.5 \
  --set tau_end=1.01 --set dns_times=1 --set dns_init=1
```

| order | case | DNS folder | alpha_in | alpha_out | why this order |
|---|---|---|---|---|---|
| 1 | narrow | poreRadius100_100 | 1 | 1 | smallest mesh; exact equilibrium check (cylinder PB) |
| 2 | converging | poreRadius200_100 | 2 | 1 | |
| 3 | diverging | poreRadius100_200 | 1 | 2 | |
| 4 | wide | poreRadius200_200 | 2 | 2 | largest mesh, last |

## Phase 0: code changes (each with its check)

| # | change | check |
|---|---|---|
| 0.1 | `dns_times` option (default 0): output times t_k = 0.005 k s, k = 0..15, converted with the run's D and L (tau_k = 0.06685 k here). With it on, `U_t`/profiles hold exactly these times (+ tau = 0) | `summary.transient` times equal the list to 1e-12; `U_t.shape[0] == 16` |
| 0.2 | `dns_init` option (default 0): t = 0 state c = c0 everywhere, phi = phi_w in the pore and 0 in the SDL (DNS "0" folder). Note: phi is algebraic in the BDF2 DAE, so after the first step phi follows from Poisson either way; the option sets the saved tau = 0 state and the first Newton guess. Today's default saves phi = 0 inside the pore at tau = 0 (hence pore-avg mu = 0.021) | tau = 0 snapshot: pore-avg mu = 2 Phi_w = 0.779; first-step solution vs default run reported (expected: equal to Newton tolerance) |
| 0.3 | constants report: summary note with V_T and lambda_D from CODATA and from pawel.py's rounded constants. No change to the solver | numbers match 25.693 / 25.680 mV, 9.9740 / 9.9716 nm |
| 0.4 | geometry report: summary notes for sharp vs rounded mouth, pore end phi, no Biot number | present in `summary.json` |
| 0.5 | mesh dry run: `run.py --mesh-only` builds the mesh, renders the two views, runs diagnose, prints elements / nodes / unknowns and a memory estimate (matrix nnz, PARDISO factor), then exits | narrow at L = 1 um reproduces 26,154 elements / 286,734 unknowns |
| 0.6 | `tools/figure_dns4.py`: the user's figureDNS4 layout (2 x 2, `cmr.bubblegum_r` 0.1-0.9, 4 colours). Row 1: centreline mu(Z) at t = 0.01 s and 0.025 s (tau 0.134, 0.334), SDL included. Row 2: Phi vs R/alpha at Z = 0.5. pnpflow = lines, DNS = squares every 50th point (same files as pawel.py: `lineYe_*`, `lineXm5_*`). Legend right, a)-d) labels, house style (arrow axes, Times, 300 dpi + svg) | render one, inspect with graphInspect; grep for banned colormaps |
| 0.7 | combined CSV `results/pore_dns_compare/csv/`: per case and saved time, centreline mu(Z), cross-section-average mu(Z), radial Phi/rho/mu at Z = 0.5, with DNS values interpolated at the same points | row counts; spot check against `profiles.json` |

After phase 0: `bash tools/validate.sh` (defaults unchanged, so all cases must still pass).

## Phase 1: mesh dry run (no solve), then stop for go-ahead

Run `--mesh-only` for all four cases. Report: elements, unknowns, h_wall / h_entr, mesh
diagnose result, two-view renders, memory estimate vs **15 GB WSL RAM**.

Rough estimate (scaled from the 1 um narrow run, h_wall = 0.1 through the pore):
narrow ~2.3M unknowns, converging / diverging ~3.5M, wide ~4.5M.

Where memory goes: the **RTX 4090 (24 GB)** does the JAX assembly (element Jacobians), which
fits easily. The sparse direct solve (PARDISO) runs on the **CPU in system RAM**, and its
factorization is the limit; the GPU does not help there in the current code.
Fallback if a case does not fit (approved by the user): slightly coarser `h_wall`
(e.g. 0.12-0.15, still >= 6-8 elements across lambda_D), reported per case. Other options,
only if asked: raise the WSL limit in `.wslconfig` (PC has 31 GB), or leave out the unused
u, p unknowns when flow = none (~40% fewer).

## Phase 2: the runs (after go-ahead)

- `tools/run_pore_dns.sh` runs the four cases in the order above, one at a time, in the
  background; each writes `results/pore_dns_<case>/` (stdout.log, live progress.json).
- I monitor progress.json (stage, tau, step size, Newton residual, health) and report each
  finish with its checks: positivity, time integration reached t_end, equilibrium solve
  (narrow: Phi vs cylinder PB), the two Boltzmann comparisons, mesh diagnose.
- On a failure: log it, report it, and go on to the next case.
- Time: narrow at 1 um took 375 s (108 BDF2 steps, 287k unknowns). At ~10x the unknowns,
  expect roughly 1-1.5 h for narrow and 2-3 h for wide; total ~6-8 h. Firm numbers after
  phase 1. Disk: ~1 GB per run.

## Phase 3: comparison output

`python tools/figure_dns4.py` -> `results/pore_dns_compare/figures/figureDNS4.png` + `svg/`,
plus the combined CSV. Report max |pnpflow - DNS| for centreline mu and radial Phi per case
and time, in pawel.py's units.

## Open items (report only, not changed)

- Mouth: DNS rounded, pnpflow sharp corner (refined with h_entr).
- Pore end: DNS phi non-uniform on x = 0 (0.376-0.389), pnpflow phi = phi_w.
- Constants: CODATA vs pawel.py, ~0.05%.
- Biot number: no pnpflow equivalent.
