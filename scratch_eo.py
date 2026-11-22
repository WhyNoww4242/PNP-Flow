"""Smoke test 3: electroosmotic slit, nonlinear Poisson-Boltzmann.

Exact solution (x along slit, walls y = +-1, applied field Ex):
  phi = psi(y) - Ex x,  c+- = exp(-+psi),
  u_x = 2 K Ex delta^2 (psi - zeta),  p = 2 K (cosh psi - 1).
"""
import numpy as np
import jax.numpy as jnp
from scipy.integrate import solve_bvp
from pnpflow import Model, bc as B, Species
from pnpflow.geometry import rectangle
from pnpflow.solver import Assembler, Newton

delta, zeta, Ex, K = 0.1, 1.0, 0.5, 2.0
yy = np.linspace(-1, 1, 4001)
guess = np.vstack([zeta * np.cosh(yy / delta) / np.cosh(1 / delta), 0 * yy])
sol = solve_bvp(lambda y, s: np.vstack([s[1], np.sinh(s[0]) / delta ** 2]),
                lambda a, b: np.array([a[0] - zeta, b[0] - zeta]),
                yy, guess, tol=1e-12, max_nodes=10 ** 6)
yt = np.linspace(-1, 1, 20001)
pt = sol.sol(yt)[0]
psi = lambda y: jnp.interp(y, yt, pt)

params = {"delta": delta, "Pe": 1.0, "Re": 0.0, "K": K, "Ex": Ex}
phi_bc = lambda x, p: psi(x[:, 1]) - p["Ex"] * x[:, 0]
cp_bc = lambda x, p: jnp.exp(-psi(x[:, 1]))
cm_bc = lambda x, p: jnp.exp(psi(x[:, 1]))
p_bc = lambda x, p: 2 * p["K"] * (jnp.cosh(psi(x[:, 1])) - 1)
end = {"phi": B.Dirichlet(phi_bc), "u": B.Pressure(p_bc),
       "c": {"c+": B.Dirichlet(cp_bc), "c-": B.Dirichlet(cm_bc)}}
bcs = {"left": end, "right": end,
       "bottom": {"u": B.NOSLIP, "phi": B.Dirichlet(phi_bc)},
       "top": {"u": B.NOSLIP, "phi": B.Dirichlet(phi_bc)}}

for hw in [0.02, 0.01, 0.005]:
    mesh = rectangle(0, 0.5, -1, 1, h=0.1, wall_names=["bottom", "top"],
                     h_wall=hw, layer=0.05, grow=0.6)
    model = Model(mesh, [Species("c+", 1), Species("c-", -1)], params, bcs,
                  flow="stokes")
    U, ok, hist = Newton(Assembler(model), tol=1e-10, log=lambda *a: None).solve(
        model.initial_guess(), params)
    f, p = model.split(np.asarray(U))
    y = mesh.X[:, 1]
    ps = np.asarray(psi(y))
    ux = 2 * K * Ex * delta ** 2 * (ps - zeta)
    print(f"hw={hw} ok={ok} its={len(hist)-1} ndof={model.ndof} "
          f"phi err {np.abs(f[2] - (ps - Ex * mesh.X[:, 0])).max():.2e} "
          f"c+ err {np.abs(f[0] - np.exp(-ps)).max():.2e} "
          f"ux err {np.abs(f[3] - ux).max():.2e} (|ux|max {np.abs(ux).max():.3f}) "
          f"uy max {np.abs(f[4]).max():.2e}")
