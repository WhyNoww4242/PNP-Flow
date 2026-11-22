"""Manufactured solution: every term of every equation is exercised.

The exact fields are smooth jax functions.  The source terms that make
them exact are derived automatically with jax.jacfwd / jax.hessian.
"""
import numpy as np
import jax
import jax.numpy as jnp

PI = jnp.pi


def exact(x):
    """Exact fields at one point x = (x0, x1)."""
    a, b = x[0], x[1]
    phi = 0.5 * jnp.sin(PI * a) * jnp.cos(PI * b) + 0.3 * a
    cp = 1.0 + 0.3 * jnp.sin(PI * a) * jnp.sin(PI * b)
    cm = 1.0 + 0.2 * jnp.cos(PI * a * b)
    p = jnp.cos(PI * a) * jnp.sin(PI * b)
    return {"c+": cp, "c-": cm, "phi": phi, "p": p}


def stream(x):
    """Stream function -> divergence free velocity."""
    a, b = x[0], x[1]
    return 0.1 * jnp.sin(PI * a) ** 2 * jnp.sin(PI * b) ** 2 + 0.5 * b + 0.3 * a


def velocity(x):
    g = jax.grad(stream)(x)
    return jnp.array([g[1], -g[0]])


SPECIES = (("c+", 1.0, 1.0), ("c-", -1.0, 0.7))   # name, z, d


def _scalar(name):
    return lambda x: exact(x)[name]


def _rho(x):
    e = exact(x)
    return sum(z * e[n] for n, z, _ in SPECIES)


def species_flux(x, name, z, d, params):
    c = _scalar(name)
    gc = jax.grad(c)(x)
    gphi = jax.grad(_scalar("phi"))(x)
    return c(x) * velocity(x) - (d / params["Pe"]) * (gc + z * c(x) * gphi)


def source_species(name, z, d):
    def s(x, params):
        J = jax.jacfwd(lambda y: species_flux(y, name, z, d, params))(x)
        return jnp.trace(J)
    return s


def source_phi(x, params):
    lap = jnp.trace(jax.hessian(_scalar("phi"))(x))
    rho_n = sum(z * z for _, z, _ in SPECIES)
    return -params["delta"] ** 2 * lap - _rho(x) / rho_n


def source_u(x, params):
    u = velocity(x)
    Gu = jax.jacfwd(velocity)(x)                     # du_i/dx_j
    lap = jnp.array([jnp.trace(jax.hessian(lambda y: velocity(y)[i])(x))
                     for i in range(2)])
    gp = jax.grad(_scalar("p"))(x)
    gphi = jax.grad(_scalar("phi"))(x)
    return params["Re"] * Gu @ u + gp - lap + params["K"] * _rho(x) * gphi


def pointwise(fn):
    """Make fn(x(2,), params) work on x of shape (..., 2)."""
    def wrapped(x, params):
        flat = x.reshape(-1, 2)
        out = jax.vmap(lambda y: fn(y, params))(flat)
        return out.reshape(x.shape[:-1] + out.shape[1:])
    return wrapped


def field_fn(name):
    return pointwise(lambda y, p: exact(y)[name])


def velocity_fn():
    return pointwise(lambda y, p: velocity(y))


def sources():
    out = {"phi": pointwise(source_phi), "u": pointwise(source_u)}
    for name, z, d in SPECIES:
        out[name] = pointwise(source_species(name, z, d))
    return out


def flux_normal_fn(name, z, d, normal):
    n = jnp.asarray(normal, dtype=float)
    return pointwise(lambda y, p: species_flux(y, name, z, d, p) @ n)


def robin_ref_fn(coef, normal):
    """ref such that delta^2 dphi/dn = coef (ref - phi) holds exactly."""
    n = jnp.asarray(normal, dtype=float)

    def ref(y, p):
        dn = jax.grad(_scalar("phi"))(y) @ n
        return exact(y)["phi"] + p["delta"] ** 2 * dn / coef
    return pointwise(ref)


def nodal_exact(X):
    X = jnp.asarray(X)
    out = {k: np.asarray(field_fn(k)(X, {})) for k in ("c+", "c-", "phi", "p")}
    u = np.asarray(velocity_fn()(X, {}))
    out["uy"], out["uz"] = u[:, 0], u[:, 1]
    return out
