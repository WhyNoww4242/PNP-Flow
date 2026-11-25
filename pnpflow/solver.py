"""Assembly and Newton solver.

The residual is assembled with jax (jit + vmap).  The Jacobian of every
element comes from jax.jacfwd, so no hand-written derivatives are needed.
Sparse linear systems are solved on the CPU with PARDISO (or SuperLU).
"""
import ctypes
import inspect
import time

import numpy as np
import scipy.sparse as sp
import jax
import jax.numpy as jnp

try:
    import pypardiso
except ImportError:          # pragma: no cover
    pypardiso = None


# Elements per batch in the Jacobian kernel: bounds the jacfwd transient memory.
ELEMENT_CHUNK = 20000


class Assembler:
    """Builds R(U) and the sparse Jacobian dR/dU for a Model."""

    def __init__(self, model):
        self.model = model
        self._sparsity()
        self._compile()

    # ------------------------------------------------------------------
    # Sparsity pattern: computed once with numpy
    # ------------------------------------------------------------------
    def _sparsity(self):
        m = self.model
        blocks_r, blocks_c = [], []
        r, c = _pairs(m.emap)
        blocks_r.append(r)
        blocks_c.append(c)
        for name in m.edge:
            r, c = _pairs(m.edge[name]["emap"])
            blocks_r.append(r)
            blocks_c.append(c)
        rows = np.concatenate(blocks_r)
        cols = np.concatenate(blocks_c)

        key = rows * m.ndof + cols
        unique, inverse = np.unique(key, return_inverse=True)
        self.nnz = len(unique)
        self.csr_rows = unique // m.ndof
        self.csr_cols = unique % m.ndof
        self.slot = jnp.asarray(inverse.astype(np.int32))              # coo entry -> csr slot
        self.coo_is_dirichlet_row = jnp.asarray(m.is_dirichlet[rows])
        # csr slots of the diagonal of Dirichlet rows
        diag = (self.csr_rows == self.csr_cols) & m.is_dirichlet[self.csr_rows]
        self.dirichlet_diag = jnp.asarray(diag.astype(float))
        # csr indptr (unique keys are sorted by row, then column)
        counts = np.bincount(self.csr_rows, minlength=m.ndof)
        self.indptr = np.concatenate([[0], np.cumsum(counts)])
        # int32 copies shared by every Jacobian, so the linear solver sees one pattern
        self._csr_indices = self.csr_cols.astype(np.int32)
        self._csr_indptr = self.indptr.astype(np.int32)

    # ------------------------------------------------------------------
    # Jitted kernels
    # ------------------------------------------------------------------
    def _compile(self):
        m = self.model
        emap = jnp.asarray(m.emap)
        geo = m.geo
        edge_maps = {k: jnp.asarray(v["emap"]) for k, v in m.edge.items()}

        def element_res(u_loc, G, wd, xq, h, params):
            return m.element_residual(u_loc, G, wd, xq, h, params)

        element_jac = jax.jacfwd(element_res)

        def edge_funcs(name):
            def res(u_loc, wd, n, xq, params):
                return m.edge_residual(name, u_loc, wd, n, xq, params)
            return res, jax.jacfwd(res)

        edge_kernels = {k: edge_funcs(k) for k in m.edge}

        def raw_residual(U, params):
            """Residual with no Dirichlet rows replaced."""
            loc = U[emap]
            r = jax.vmap(element_res, in_axes=(0, 0, 0, 0, 0, None))(
                loc, geo["G"], geo["wd"], geo["xq"], geo["h"], params)
            R = jnp.zeros(m.ndof).at[emap].add(r)
            for name, (res, _) in edge_kernels.items():
                e = m.edge[name]
                loc = U[edge_maps[name]]
                rb = jax.vmap(res, in_axes=(0, 0, 0, 0, None))(
                    loc, e["wd"], e["n"], e["xq"], params)
                R = R.at[edge_maps[name]].add(rb)
            return R

        def volume_residual(U, params):
            """Element terms only (used for reaction fluxes and forces)."""
            loc = U[emap]
            r = jax.vmap(element_res, in_axes=(0, 0, 0, 0, 0, None))(
                loc, geo["G"], geo["wd"], geo["xq"], geo["h"], params)
            return jnp.zeros(m.ndof).at[emap].add(r)

        def residual(U, params):
            R = raw_residual(U, params)
            for ids, vals in m.dirichlet_values(params):
                R = R.at[ids].set(U[ids] - vals)
            return R

        def jacobian_values(U, params):
            loc = U[emap]
            k = jax.lax.map(
                lambda a: element_jac(a[0], a[1], a[2], a[3], a[4], params),
                (loc, geo["G"], geo["wd"], geo["xq"], geo["h"]),
                batch_size=ELEMENT_CHUNK)
            parts = [k.reshape(-1)]
            for name, (_, jac) in edge_kernels.items():
                e = m.edge[name]
                loc = U[edge_maps[name]]
                kb = jax.vmap(jac, in_axes=(0, 0, 0, 0, None))(
                    loc, e["wd"], e["n"], e["xq"], params)
                parts.append(kb.reshape(-1))
            vals = jnp.concatenate(parts)
            vals = jnp.where(self.coo_is_dirichlet_row, 0.0, vals)
            data = jax.ops.segment_sum(vals, self.slot, num_segments=self.nnz)
            return data + self.dirichlet_diag

        self.raw_residual = jax.jit(raw_residual)
        self.volume_residual = jax.jit(volume_residual)
        self.residual = jax.jit(residual)
        self.jacobian_values = jax.jit(jacobian_values)

    # ------------------------------------------------------------------
    def jacobian(self, U, params):
        return self.csr(self.jacobian_values(U, params))

    def csr(self, data):
        """CSR matrix with this assembler's sparsity pattern and the given values."""
        n = self.model.ndof
        return sp.csr_matrix((np.asarray(data), self._csr_indices, self._csr_indptr),
                             shape=(n, n), copy=False)


def _pairs(emap):
    """All (row, col) pairs of the local blocks, flattened like jacfwd."""
    n_loc = emap.shape[1]
    rows = np.repeat(emap, n_loc, axis=1).reshape(-1)
    cols = np.tile(emap, (1, n_loc)).reshape(-1)
    return rows, cols


# ----------------------------------------------------------------------
# Linear solve
# ----------------------------------------------------------------------

class Pardiso:
    """MKL PARDISO (real unsymmetric) that keeps its reordering between solves.

    Phase 11 (fill-reducing reordering and symbolic factorisation) depends on the
    sparsity pattern and is mostly serial: at 1.1M dofs it took 3.5 s of a 5.3 s
    solve on about 3 of 16 cores. It runs once per pattern; each solve then runs phase 23
    (numeric factorisation on all cores, then the solve). Phase 11 also picks a pivot
    matching from the values it saw, so a solve whose relative residual is above
    `rtol` redoes phase 11 with the current values and solves again.
    MKL settings are its defaults (iparm[0] = 0), as pypardiso.spsolve used them.
    """

    MTYPE = 11

    def __init__(self, rtol=1e-8):
        self.rtol = rtol
        self._mkl = pypardiso.PyPardisoSolver()      # finds and loads mkl_rt
        self._pattern = None                         # (indptr, indices) last analysed
        self._ia = self._ja = None                   # the same, 1-based for MKL
        self.n_analyses = 0

    def solve(self, A, b):
        A = A.tocsr()
        if not A.has_sorted_indices:
            A = A.sorted_indices()
        data = np.ascontiguousarray(A.data, dtype=np.float64)
        b = np.asfortranarray(b, dtype=np.float64)
        if not self._same_pattern(A):
            self._analyse(A, data)
        x = self._call(23, data, b)
        if self._rel_residual(A, x, b) > self.rtol:
            self._analyse(A, data)
            x = self._call(23, data, b)
        return x

    def _same_pattern(self, A):
        if self._pattern is None:
            return False
        indptr, indices = self._pattern
        return (A.shape[0] + 1 == len(indptr) and np.array_equal(A.indptr, indptr)
                and np.array_equal(A.indices, indices))

    def _analyse(self, A, data):
        self._pattern = (A.indptr.copy(), A.indices.copy())
        self._ia = A.indptr.astype(np.int32) + 1
        self._ja = A.indices.astype(np.int32) + 1
        self._call(11, data, np.zeros((A.shape[0], 1), order="F"))
        self.n_analyses += 1

    @staticmethod
    def _rel_residual(A, x, b):
        bn = np.linalg.norm(b)
        return np.linalg.norm(A @ x - b) / bn if bn > 0 else 0.0

    def _call(self, phase, data, b):
        mkl = self._mkl
        x = np.zeros_like(b)
        err = ctypes.c_int32(0)
        i32 = ctypes.POINTER(ctypes.c_int32)
        f64 = ctypes.POINTER(ctypes.c_double)
        one = ctypes.byref(ctypes.c_int32(1))
        mkl._mkl_pardiso(
            mkl.pt.ctypes.data_as(ctypes.POINTER(mkl._pt_type[0])),
            one, one,                                        # maxfct, mnum
            ctypes.byref(ctypes.c_int32(self.MTYPE)),
            ctypes.byref(ctypes.c_int32(phase)),
            ctypes.byref(ctypes.c_int32(len(self._ia) - 1)),
            data.ctypes.data_as(f64), self._ia.ctypes.data_as(i32),
            self._ja.ctypes.data_as(i32), mkl.perm.ctypes.data_as(i32),
            ctypes.byref(ctypes.c_int32(1 if b.ndim == 1 else b.shape[1])),  # nrhs
            mkl.iparm.ctypes.data_as(i32),
            ctypes.byref(ctypes.c_int32(0)),                 # msglvl
            b.ctypes.data_as(f64), x.ctypes.data_as(f64), ctypes.byref(err))
        if err.value != 0:
            raise RuntimeError(f"PARDISO phase {phase} failed with error {err.value}")
        return x


_pardiso = None


def linear_solve(A, b):
    """Solve A x = b: PARDISO when installed (reordering reused while the pattern
    stays the same), else SuperLU."""
    global _pardiso
    if pypardiso is None:
        return sp.linalg.spsolve(A.tocsc(), b)
    if _pardiso is None:
        _pardiso = Pardiso()
    return _pardiso.solve(A, b)


# ----------------------------------------------------------------------
# Newton with backtracking and parameter continuation
# ----------------------------------------------------------------------

class Newton:
    """Damped Newton.

    Accepts a step when the residual norm decreases and all concentrations
    stay positive.  Otherwise the step is halved.

    on_iteration(stage, history[, status=...]) is called after every
    iteration.  If it accepts a `status` keyword it also gets a dict for
    live progress (see status()), and one extra call with status["outcome"]
    set when a solve or a continuation run fails (history is not extended
    by that call).
    """

    def __init__(self, assembler, tol=1e-9, max_iter=30, min_alpha=1e-3,
                 log=print, on_iteration=None):
        self.asm = assembler
        self.tol = tol
        self.max_iter = max_iter
        self.min_alpha = min_alpha
        self.log = log
        self.on_iteration = on_iteration
        self._wants_status = _accepts_keyword(on_iteration, "status")
        self.context = None             # set by continuation()
        m = assembler.model
        self.conc_ids = np.arange(m.S * m.n)

    def norm(self, U, params):
        return float(jnp.linalg.norm(self.asm.residual(U, params)))

    def positive(self, U):
        if len(self.conc_ids) == 0:
            return True
        return bool(np.all(np.asarray(U)[self.conc_ids] > 0.0))

    def solve(self, U, params, stage=0):
        params = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in params.items()}
        model = self.asm.model
        U = model.apply_dirichlet(jnp.asarray(U), params)
        r0 = self.norm(U, params)
        history = [{"it": 0, "res_norm": r0, "step_norm": 0.0, "alpha": 0.0}]
        self._U = U
        self._report(stage, history)
        if r0 < self.tol:
            return U, True, history
        final = "attempt_failed" if self.context else "failed"

        res = r0
        for it in range(1, self.max_iter + 1):
            t0 = time.time()
            R = np.asarray(self.asm.residual(U, params))
            A = self.asm.jacobian(U, params)
            dU = jnp.asarray(linear_solve(A, -R))
            alpha, U_new, res_new = self._line_search(U, dU, res, params)
            if U_new is None:
                self.log(f"  line search failed at it {it}")
                self._report(stage, history, outcome=final)
                return U, False, history
            U, res = U_new, res_new
            self._U = U
            step = float(jnp.linalg.norm(alpha * dU))
            history.append({"it": it, "res_norm": res, "step_norm": step,
                            "alpha": alpha, "time_s": time.time() - t0})
            self.log(f"  it {it:2d}  |R| {res:.3e}  |dU| {step:.3e}  "
                     f"alpha {alpha:.3f}  ({time.time() - t0:.2f}s)")
            self._report(stage, history)
            if res < self.tol_eff(r0):
                return U, True, history
        self._report(stage, history, outcome=final)
        return U, False, history

    def tol_eff(self, r0):
        """Residual norm at which a solve that started at r0 has converged."""
        return self.tol * max(1.0, 1e-3 * r0)

    def _line_search(self, U, dU, res, params):
        alpha = 1.0
        while alpha >= self.min_alpha:
            U_try = U + alpha * dU
            if self.positive(U_try):
                res_try = self.norm(U_try, params)
                if np.isfinite(res_try) and res_try < (1 - 1e-4 * alpha) * res:
                    return alpha, U_try, res_try
            alpha *= 0.5
        return alpha, None, res

    def _report(self, stage, history, outcome=None):
        if self.on_iteration is None:
            return
        if not self._wants_status:
            if outcome is None:
                self.on_iteration(stage, history)
            return
        self.on_iteration(stage, history, status=self.status(history, outcome))

    def status(self, history, outcome=None):
        """Numbers for live progress (all plain floats / ints / None).

        n_targets, target : continuation targets in total, current index
        frac_done, trial  : fraction of the current target already reached,
                            and the fraction this Newton solve aims for
        halvings          : step halvings so far in the current target
        r0, tol_eff       : residual at it 0 and the convergence threshold
        min_c             : minimum concentration (None without species)
        outcome           : None, "attempt_failed" (continuation will halve)
                            or "failed" (the solve is given up)
        """
        ctx = self.context or {"n_targets": 1, "target": 0, "frac_done": 0.0,
                               "trial": 1.0, "halvings": 0}
        r0 = history[0]["res_norm"]
        min_c = None
        if len(self.conc_ids) and getattr(self, "_U", None) is not None:
            min_c = float(np.min(np.asarray(self._U)[self.conc_ids]))
        return dict(ctx, r0=float(r0), tol_eff=float(self.tol_eff(r0)),
                    min_c=min_c, outcome=outcome)


def _accepts_keyword(fn, name):
    if fn is None:
        return False
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(p.name == name or p.kind == p.VAR_KEYWORD for p in params)


def continuation(newton, U, base, path, max_halvings=6, log=print):
    """Walk parameters from base towards the targets in `path`.

    path : list of dicts of target values.  Each target is reached in one
           step, and the step is cut in half when Newton fails.
    Returns (U, params, converged, stages).
    """
    params = dict(base)
    stages = []
    try:
        for t, target in enumerate(path):
            start = {k: params[k] for k in target}
            frac, step, halvings = 0.0, 1.0, 0
            while frac < 1.0:
                trial = min(1.0, frac + step)
                p = dict(params)
                for k in target:
                    p[k] = start[k] + trial * (target[k] - start[k])
                log(f"stage {len(stages) + 1}: " +
                    ", ".join(f"{k}={p[k]:.4g}" for k in target))
                newton.context = {"n_targets": len(path), "target": t,
                                  "frac_done": frac, "trial": trial,
                                  "halvings": halvings}
                U_new, ok, hist = newton.solve(U, p, stage=len(stages) + 1)
                stages.append({"params": {k: float(p[k]) for k in target},
                               "converged": ok, "iterations": hist})
                if ok:
                    U, params, frac = U_new, p, trial
                    step = min(1.0 - frac, step * 1.5) if frac < 1 else step
                else:
                    halvings += 1
                    step *= 0.5
                    if halvings > max_halvings:
                        newton.context["halvings"] = halvings
                        newton._report(len(stages), hist, outcome="failed")
                        return U, params, False, stages
    finally:
        newton.context = None
    return U, params, True, stages
