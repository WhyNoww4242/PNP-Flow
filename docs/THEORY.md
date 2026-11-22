# pnpflow — theory and design notes

## 1. Equations (nondimensional, steady)

Scales: length `a` (cylinder radius), speed `U`, concentration `c0` (bulk of each ion),
potential `V_T = k_B T / e`, pressure `eta U / a`.

| symbol | meaning |
|---|---|
| `Pe = U a / D0` | Péclet number |
| `d_i = D_i / D0` | relative diffusivity |
| `Re = rho U a / eta` | Reynolds number |
| `delta = lambda_D / a`, `lambda_D^2 = eps V_T / (2 F c0)` | Debye length |
| `K = F c0 V_T a / (eta U)` | electric-to-viscous coupling |

```
species i :  div N_i = s_i
             N_i = c_i u - (d_i/Pe) (grad c_i + z_i c_i grad phi)
Poisson   :  -delta^2 lap phi = rho / rho_n + s_phi    rho = sum z_i c_i  (s_phi: fixed charge)
                                                         rho_n = sum z_i^2 c_i,bulk  (= 2 for 1:1)
momentum  :  Re (u.grad) u = -grad p + lap u - K rho grad phi
mass      :  div u = 0
```
For a 1:1 electrolyte this is exactly the system in `problem.txt`
(`-delta^2 lap phi = (c+ - c-)/2`, body force `-K (c+ - c-) grad phi`).

A useful number: the electro-osmotic (Helmholtz–Smoluchowski) speed in these units is
`u_eo = 2 K delta^2 phi_wall E`. So the flow feels the EDL strongly only when `K delta^2` is O(1) or larger.

## 2. Discretisation

* **Elements**: P2 triangles for `c_i`, `phi`, `u` (curved, from gmsh `setOrder(2)`), P1 for `p`
  (Taylor–Hood, inf–sup stable). 7-point quadrature, exact to degree 5.
* **Weak form** (`model.py`): written once per element as a plain jax function.
  * species, conservative: `-int grad w . N_i`  + boundary term `w N_i.n`
  * species, non-conservative: `int w u.grad c - grad w . J_i` (only equal if div u = 0)
  * Poisson: `int delta^2 grad v . grad phi - v rho/rho_n` (+ Robin / surface charge terms)
  * momentum (Laplacian form): `int grad v : grad u - p div v + v.(Re u.grad u + K rho grad phi)`
  * optional SUPG on species (approximate strong residual, transport velocity `u - D z grad phi`).
* **Assembly** (`solver.py`): `jax.vmap` over elements, `jax.jit` for the whole residual.
  The element Jacobian is `jax.jacfwd(element_residual)` — exact, no hand derivatives.
  The sparse pattern is computed once; values are scattered with `segment_sum`.
* **Dirichlet**: rows replaced by `U_i - g_i`. Later boundaries win at shared corners.
* **Newton**: PARDISO direct solve, backtracking until the residual decreases and all
  concentrations stay positive. **Continuation** walks parameters (e.g. wall potential,
  then `K`), halving the step on failure.

## 3. Boundary conditions (`bc.py`)

| key | options | natural (unset) |
|---|---|---|
| `c`   | `Dirichlet(value or "bulk")`, `Flux(N.n)`, `BLOCKING`, `OUTFLOW` | blocking `N.n = 0` |
| `phi` | `Dirichlet`, `Robin(C, ref)` (Stern: `delta^2 dphi/dn = C (ref - phi)`), `Charge(sigma)` | `dphi/dn = 0` |
| `u`   | `NOSLIP`, `Dirichlet((uy, uz))`, `Component(i, v)` (symmetry/slip), `Pressure(p0)`, `FREE` | no-slip |

Values may be numbers, parameter names, or `f(x, params)` jax functions.

## 4. Post-processing (`analysis.py`)

* **Boundary fluxes** by the reaction method (residual of the volume terms summed on the boundary
  nodes). This gives conservative total species flux, induced wall charge (`delta^2 dphi/dn`)
  and hydrodynamic wall force without differentiating the solution.
* **Electric force on the cylinder** = minus the total body force on the diffuse charge
  (momentum balance of the charged fluid; Maxwell stress on the outer boundary neglected —
  small when the box is many Debye lengths).
* Smooth nodal `E`, vorticity, fluxes `N_i`, current `j` from averaged element gradients.
* P2-accurate point evaluation for line profiles; wall profiles vs angle.
* **Adjoint sensitivities** `dQ/dp` for any jax scalar `Q(U, p)`:
  `K^T lambda = dQ/dU`,  `dQ/dp = dQ/dp|explicit - lambda . dR/dp` (`dR/dp` by `jax.jvp`).
  One extra linear solve per quantity, for all parameters at once.

## 5. Cylinder case choices (defaults — change them in `cases.py` / `--set`)

* Box `[-40, 40] x [-80, 40]`; inlet at top `u = (0,-1)`, `c = 1`, `phi = 0`;
  outlet bottom traction free with ions convected out; sides slip.
* `Re = 0.5` (> 0) is the answer to the **Stokes paradox**: with finite inertia an Oseen-like
  far wake exists and the box is far enough away for drag to be box-insensitive (check it!).
* Wall: blocking for ions, no slip, `phi = 25 mV / V_T = 0.973` at `r = 1`.
  `stern_C > 0` switches to a Stern (Robin) condition instead.
* Mesh: `h = delta/8` within `3 delta` of the wall, growing to `h_far`.

## 6. Axisymmetric (r–z) coordinates

`Model(..., coord="axisym")` (default `"planar"`). The mesh lives in the half plane
`r ≥ 0`, stored as gmsh `(x, y) = (r, z)`; the axis is `r = 0`.

In cylindrical coordinates, for fields independent of the angle θ,
`div N = (1/r) ∂(r N_r)/∂r + ∂N_z/∂z`, `grad c = (∂c/∂r, ∂c/∂z)`, and the volume element is
`r dr dθ dz`. Multiplying the strong form by a test function and integrating over θ gives the
planar weak form with every integral weighted by `r`. So `fem.element_geometry` and
`fem.edge_geometry` multiply the quadrature weights `wd` by the `r` of each quadrature point;
nothing else changes. This is exact for
* Poisson: `int delta^2 grad v . grad phi r dA − int v (rho/rho_n + s_phi) r dA` + edge terms `r ds`,
* Nernst–Planck (conservative and non-conservative forms, SUPG included),
because they only contain gradients, divergences of vectors in the (r, z) plane, and scalars.
The axis needs no condition: its edges have `r = 0` weight, and the natural condition there
is the symmetry condition `∂/∂r = 0`.

Not valid for Stokes / Navier–Stokes: the vector Laplacian and `div u` gain hoop terms
(`−u_r/r^2`, `u_r/r`) that the planar momentum form lacks, so `coord="axisym"` with
`flow="stokes" | "navier_stokes"` raises an error. `flow="prescribed"` is allowed.

Convention: every integral reported by `analysis.py` (fluxes, charges, `integrate`) for an
axisymmetric model is **per radian**. Multiply by 2π for the full 3D body (case outputs ending
in `_3d` already include it). The quadrature is one degree less exact than in planar
(`r` raises the integrand degree by one); with the degree-5 rule this is still exact for the
P2 stiffness on straight elements.

## 7. Charged sphere (`charged_sphere`, Poisson only, axisymmetric)

A sphere of radius `R` with uniform charge density `q` in an unbounded uniform dielectric,
no ions, no flow (`species = []`, `flow = "none"`, `delta = 1`):
```
-lap phi = q  (rho < R),   0  (rho > R),     rho = sqrt(r^2 + z^2)
exact (phi -> 0 at infinity):
  phi = q R^2/2 - q rho^2/6     (rho <= R)        |E| = q rho/3
  phi = q R^3 / (3 rho)         (rho >= R)        |E| = q R^3 / (3 rho^2)
```
* Domain: half disk `rho ≤ R_out`, `r ≥ 0` (`geometry.sphere_axisym`), OCC-fragmented with the
  half disk of radius `R` so the mesh conforms to the sphere surface. Groups: curves `axis`,
  `outer`; surfaces `sphere`, `medium`. The interior arc has no physical group (an edge in a
  boundary group must belong to one triangle); it drives a Distance size field instead
  (`h_sphere` at the arc, growing to `h_far`).
* Charge: the Poisson source `s_phi = q` on elements whose centroid lies inside `rho < R`
  (checked against the gmsh `sphere` region at build time), so the jump sits exactly on
  element edges.
* Far field: `outer_bc = 1` (default) uses `B.Robin(1/R_out, 0)`, i.e. `dphi/dn = −phi/R_out`,
  exact for a monopole; `outer_bc = 0` uses Dirichlet with the exact value.
* Checks: max `|phi − exact| / max phi` over nodes; Gauss law `2π ∫_outer −dphi/dn r ds =
  2π ∫ q r dA` (discrete, to round-off) and against `4π q R^3/3` (geometric error of the
  curved P2 sphere only); min / max phi vs `q R^3/(3 R_out)` and `q R^2/2`.
* Measured (R = 1, R_out = 10, q = 1, Robin): max rel. error 3.7e-6 at the default mesh
  (h_sphere = 0.05, h_far = 0.5, 4637 elements) and 5.3e-7 at h_sphere = 0.025,
  h_far = 0.25 (18058 elements), about `h^3` as expected for P2. Gauss balance ~1e-14.
