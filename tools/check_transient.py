"""Check of the time stepper (pnpflow/transient.py) against an exact solution.

Axisymmetric box r in [0, 0.5], z in [0, 1], two ions (z = +-1, d = 1, Pe = 1),
c+- = 1 and phi = 0 on z = 0 and z = 1, blocking side wall and axis.
Initial state c+- = 1 + A sin(pi z).  Then rho = 0, phi = 0 for all time and

    c+-(z, t) = 1 + A sin(pi z) exp(-pi^2 t)

exactly, which exercises the r-weighted mass matrix, the BDF2 coefficients
and the DAE treatment of phi.  Fixed steps dt, dt/2, dt/4 must show order 2.

    python tools/check_transient.py
"""
import os
import sys

import numpy as np
import jax.numpy as jnp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pnpflow import bc as B, geometry                         # noqa: E402
from pnpflow.model import Model, Species                      # noqa: E402
from pnpflow.solver import Assembler, Newton                  # noqa: E402
from pnpflow.transient import TransientAssembler, integrate   # noqa: E402

A, T_END = 0.1, 0.1


def main():
    mesh = geometry.rectangle(0.0, 0.5, 0.0, 1.0, h=0.05)
    sp = [Species("c+", 1.0), Species("c-", -1.0)]
    ends = {"c": B.Dirichlet("bulk"), "phi": B.Dirichlet(0.0)}
    bcs = {"bottom": ends, "top": ends}
    params = {"delta": 0.5, "Pe": 1.0}
    model = Model(mesh, sp, params, bcs, flow="none", coord="axisym")
    asm = Assembler(model)
    tasm = TransientAssembler(asm)
    newton = Newton(tasm, tol=1e-11, log=lambda *a: None)

    z = mesh.X[:, 1]
    U0 = np.array(model.initial_guess(params))
    for i in range(2):
        U0[i * model.n:(i + 1) * model.n] = 1 + A * np.sin(np.pi * z)
    exact = 1 + A * np.sin(np.pi * z) * np.exp(-np.pi ** 2 * T_END)

    errs = []
    for dt in (0.01, 0.005, 0.0025):
        U, t, steps, ok = integrate(newton, tasm, jnp.asarray(U0), params, T_END,
                                    dt0=dt, dt_max=dt, rtol=1e9, log=lambda *a: None)
        f = np.asarray(U)[:2 * model.n].reshape(2, -1)
        phi = np.asarray(U)[2 * model.n:3 * model.n]
        e = float(np.max(np.abs(f - exact)))
        errs.append(e)
        print(f"dt {dt:.4f}  steps {len(steps):3d}  ok {ok}  max|c - exact| {e:.3e}  "
              f"max|phi| {np.abs(phi).max():.1e}")
    orders = [np.log2(errs[i] / errs[i + 1]) for i in range(len(errs) - 1)]
    print("observed order:", " ".join(f"{o:.2f}" for o in orders))
    passed = all(o > 1.8 for o in orders) and errs[-1] < 1e-5
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
