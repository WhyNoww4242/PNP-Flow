"""Time-dependent solves: variable-step BDF2 on top of the steady Assembler.

The semi-discrete system is

    M dU/dt + R(U) = 0

where R is the steady residual (Model.element_residual / edge_residual) and M
is the consistent P2 mass matrix of the species fields only (phi, u and p have
no time derivative, so the system is an index-1 DAE that BDF handles).  M comes
from jax.jacfwd of the element mass residual, like every other Jacobian here.

One BDF step with step dt_n and ratio w = dt_n / dt_{n-1}:

    a0 M (U - U*) + R(U) = 0,
    a0 = (1 + 2w) / ((1 + w) dt_n),
    U* = [(1 + w) U^n - w^2/(1 + w) U^{n-1}] (1 + w) / (1 + 2w)

(w = 0 gives backward Euler, used for the first step).  Dirichlet rows keep
U_i - g_i (their mass rows are zeroed).

TransientAssembler has the same interface as Assembler (residual, jacobian,
model), so solver.Newton is reused unchanged.
"""
import time

import numpy as np
import jax
import jax.numpy as jnp


class TransientAssembler:
    """R_t(U) = R(U) + a0 M (U - U*) for a fixed (a0, U*) set by set_step()."""

    def __init__(self, asm):
        self.asm = asm
        self.model = m = asm.model
        F, S, N = m.F, m.S, m.N
        n_loc = m.emap.shape[1]

        def elem_mass(u_loc, wd):
            Ue = u_loc[:6 * F].reshape(F, 6)
            val = Ue[:S] @ N.T                                   # (S, Q)
            r = jnp.einsum("sq,qa,q->sa", val, N, wd)            # (S, 6)
            out = jnp.zeros((F, 6)).at[:S].set(r).reshape(-1)
            return jnp.concatenate([out, jnp.zeros(n_loc - 6 * F)])

        mass_jac = jax.jacfwd(elem_mass)                         # exact (linear)
        zero = jnp.zeros(n_loc)
        k = jax.vmap(mass_jac, in_axes=(None, 0))(zero, m.geo["wd"])
        vals = k.reshape(-1)
        n_edge = int(asm.slot.shape[0]) - vals.shape[0]
        vals = jnp.concatenate([vals, jnp.zeros(n_edge)])
        vals = jnp.where(asm.coo_is_dirichlet_row, 0.0, vals)
        self.mass_data = jax.ops.segment_sum(vals, asm.slot, num_segments=asm.nnz)
        rows = jnp.asarray(asm.csr_rows)
        cols = jnp.asarray(asm.csr_cols)
        ndof = m.ndof

        def mass_apply(x):
            return jax.ops.segment_sum(self.mass_data * x[cols], rows,
                                       num_segments=ndof)

        def residual(U, params, Ustar, a0):
            return asm.residual(U, params) + a0 * mass_apply(U - Ustar)

        self.mass_apply = jax.jit(mass_apply)
        self._residual = jax.jit(residual)
        self.a0 = 0.0
        self.Ustar = jnp.zeros(ndof)

    def set_step(self, a0, Ustar):
        self.a0 = float(a0)
        self.Ustar = jnp.asarray(Ustar)

    def residual(self, U, params):
        return self._residual(U, params, self.Ustar, self.a0)

    def jacobian(self, U, params):
        return self.asm.csr(self.asm.jacobian_values(U, params) + self.a0 * self.mass_data)

    def mass_matrix(self):
        return self.asm.csr(self.mass_data)


def bdf_coefficients(dt, dt_prev):
    """(a0, c_n, c_nm1) with U* = c_n U^n + c_nm1 U^{n-1}; dt_prev=None -> Euler."""
    if dt_prev is None:
        return 1.0 / dt, 1.0, 0.0
    w = dt / dt_prev
    a0 = (1 + 2 * w) / ((1 + w) * dt)
    scale = (1 + w) / (1 + 2 * w)
    return a0, scale * (1 + w), -scale * w * w / (1 + w)


def integrate(newton, tasm, U0, params, t_end, outputs=(), dt0=1e-2, dt_max=None,
              rtol=1e-3, w_max=2.0, max_rejects=12, on_output=None, on_step=None,
              log=print, consistent_init=True, on_accept=None, err_scale="abs",
              clock_text=None):
    """Adaptive BDF2 from t = 0 to t_end.

    clock_text: optional f(t, dt) -> str for the log lines (e.g. real time and tau beside
                the solver time); default "t <t>  dt <dt>" in solver units

    outputs   : times at which on_output(t, U) is called (hit exactly)
    rtol      : target for |U - U_pred|_inf over the species and phi dofs, with
                U_pred the linear extrapolation of the last two steps.  That
                difference ~ dt^2 U'' (a curvature measure, not the BDF2 error
                itself), so dt is scaled by sqrt(rtol / err); the time-
                discretisation error is checked separately (tools/check_transient.py)
    w_max     : largest step ratio (BDF2 is zero-stable for w < 1 + sqrt 2)
    on_step(info) gets a dict per accepted/rejected step (for live status).
    on_accept(t, U) is called after every accepted step (and at t = 0) for cheap
                per-step quantities (e.g. the charge in a region).
    err_scale : "abs" (default): err = max |U - U_pred|.  "mixed": err = max
                |U - U_pred| / (1 + |U|), for fields that grow far above 1 (EDL
                concentrations at high potentials, c ~ exp(Phi)).
    Returns (U, t, steps, ok).
    """
    clock = clock_text or (lambda t, dt: f"t {t:.6g}  dt {dt:.3g}")
    m = tasm.model
    watch = np.arange((m.S + 1) * m.n)              # species + phi dofs
    dt_max = dt_max or t_end / 20
    outputs = sorted(t for t in outputs if 0 < t <= t_end * (1 + 1e-12))
    if not outputs or abs(outputs[-1] - t_end) > 1e-12 * t_end:
        outputs.append(t_end)
    # consistent initial state: phi (no time derivative) from Poisson with the
    # initial concentrations, i.e. a backward-Euler step with dt -> 0
    # (consistent_init=False: U0 is a prescribed field, saved as given)
    if consistent_init:
        tasm.set_step(1e12, U0)
        U0c, ok0, _ = newton.solve(jnp.asarray(U0), params, stage=0)
        if ok0:
            U0 = U0c
        else:
            log("  consistent initial phi: Newton did not converge, using U0 as given")
    if on_output is not None:
        on_output(0.0, U0)
    if on_accept is not None:
        on_accept(0.0, U0)

    t, U, U_prev, dt_prev = 0.0, jnp.asarray(U0), None, None
    dt = min(dt0, dt_max)
    steps, rejects_total, k_out = [], 0, 0
    while k_out < len(outputs):
        target = outputs[k_out]
        remaining = target - t
        if remaining <= 1e-12 * max(1.0, t_end):
            k_out += 1
            continue
        # land on the output time without a tiny final step
        dt_try = dt
        if dt_try >= remaining:
            dt_try = remaining
        elif dt_try > 0.5 * remaining:
            dt_try = 0.5 * remaining
        if dt_prev is not None:
            dt_try = min(dt_try, w_max * dt_prev)

        rejects = 0
        while True:
            a0, cn, cm = bdf_coefficients(dt_try, dt_prev)
            Ustar = cn * U + (cm * U_prev if U_prev is not None else 0.0)
            tasm.set_step(a0, Ustar)
            if U_prev is not None:
                w = dt_try / dt_prev
                U_pred = U + w * (U - U_prev)
            else:
                U_pred = U
            t0 = time.time()
            U_new, ok, hist = newton.solve(U_pred, params, stage=len(steps) + 1)
            err = None
            if ok:
                diff = np.asarray(U_new - U_pred)[watch]
                if err_scale == "mixed":
                    diff = diff / (1.0 + np.abs(np.asarray(U_new)[watch]))
                err = float(np.max(np.abs(diff)))
            accept = ok and (err is None or U_prev is None or err <= 4 * rtol)
            info = {"t": t, "dt": dt_try, "newton_its": len(hist) - 1,
                    "converged": bool(ok), "err": err, "accepted": bool(accept),
                    "time_s": time.time() - t0}
            if on_step is not None:
                on_step(dict(info, t_end=t_end, n_steps=len(steps),
                             rejects=rejects_total))
            if accept:
                break
            rejects += 1
            rejects_total += 1
            if rejects > max_rejects:
                log(f"time step failed at {clock(t, dt_try)}")
                return U, t, steps, False
            dt_try *= 0.25 if not ok else 0.5
            log(f"  reject step: converged={ok} err={err} -> {clock(t, dt_try)}")

        U_prev, U = U, U_new
        dt_prev = dt_try
        t = t + dt_try
        steps.append({"t": t, "dt": dt_try, "newton_its": len(hist) - 1,
                      "err": err, "rejects": rejects,
                      "res_final": hist[-1]["res_norm"]})
        log(f"step {len(steps):4d}  {clock(t, dt_try)}  "
            f"its {len(hist) - 1}  err {err if err is None else f'{err:.2e}'}")
        if on_accept is not None:
            on_accept(t, U)
        # next step size from the error estimate and Newton effort
        if err is None or err <= 0:
            fac = 1.5
        else:
            fac = 0.9 * np.sqrt(rtol / err)
        if len(hist) - 1 > 6:
            fac = min(fac, 1.0)
        dt = float(np.clip(fac, 0.5, w_max) * dt_try)
        dt = min(dt, dt_max)
        if abs(t - target) <= 1e-12 * max(1.0, t_end):
            t = target
            if on_output is not None:
                on_output(t, U)
            k_out += 1
    return U, t, steps, True
