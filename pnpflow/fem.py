"""Finite element basics.

Elements:
  P2 triangle (6 nodes): 3 corners, then midpoints of edges 01, 12, 20.
  P1 triangle (3 nodes): corners only, used for pressure.
  P2 line (3 nodes): 2 ends, then midpoint, used on boundaries.

Each shape function is written once.  Its gradient comes from jax.
"""
import numpy as np
import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Quadrature
# ---------------------------------------------------------------------------

def tri_quadrature():
    """7-point rule, exact to degree 5 (Dunavant).

    Returns reference points (Q, 2) and weights (Q,).
    The weights sum to 1/2, the reference triangle area.
    """
    a1, b1 = 0.059715871789770, 0.470142064105115
    a2, b2 = 0.797426985353087, 0.101286507323456
    w0, w1, w2 = 0.225, 0.132394152788506, 0.125939180544827
    third = 1.0 / 3.0
    points = [
        (third, third),
        (b1, b1), (a1, b1), (b1, a1),
        (b2, b2), (a2, b2), (b2, a2),
    ]
    weights = [w0, w1, w1, w1, w2, w2, w2]
    return np.array(points), 0.5 * np.array(weights)


def line_quadrature():
    """3-point Gauss rule on [0, 1], exact to degree 5."""
    s = 0.5 * np.sqrt(3.0 / 5.0)
    points = np.array([0.5 - s, 0.5, 0.5 + s])
    weights = np.array([5.0, 8.0, 5.0]) / 18.0
    return points, weights


# ---------------------------------------------------------------------------
# Shape functions
# ---------------------------------------------------------------------------

P2_NODES = np.array([
    [0.0, 0.0], [1.0, 0.0], [0.0, 1.0],
    [0.5, 0.0], [0.5, 0.5], [0.0, 0.5],
])


def p2_shape(p):
    """Six quadratic shape functions at reference point p = (xi, eta)."""
    l1 = p[0]
    l2 = p[1]
    l0 = 1.0 - l1 - l2
    return jnp.array([
        l0 * (2 * l0 - 1),
        l1 * (2 * l1 - 1),
        l2 * (2 * l2 - 1),
        4 * l0 * l1,
        4 * l1 * l2,
        4 * l2 * l0,
    ])


def p1_shape(p):
    """Three linear shape functions."""
    return jnp.array([1.0 - p[0] - p[1], p[0], p[1]])


def line_shape(t):
    """Three quadratic shape functions on [0, 1]: end, end, middle."""
    return jnp.array([(1 - t) * (1 - 2 * t), t * (2 * t - 1), 4 * t * (1 - t)])


def p2_tables(points):
    """Values (Q, 6) and reference gradients (Q, 6, 2) of P2 shapes."""
    points = jnp.asarray(points)
    values = jax.vmap(p2_shape)(points)
    grads = jax.vmap(jax.jacfwd(p2_shape))(points)
    return values, grads


# ---------------------------------------------------------------------------
# Geometry on the actual (possibly curved) elements
# ---------------------------------------------------------------------------

def physical_gradients(Xe, dN):
    """Map reference gradients to physical ones.

    Xe : (E, 6, 2) element node coordinates
    dN : (Q, 6, 2) reference gradients
    Returns gradients (E, Q, 6, 2) and |det J| (E, Q).
    """
    # J[e, q, k, l] = d x_k / d xi_l
    J = jnp.einsum("eak,qal->eqkl", Xe, dN)
    det = J[..., 0, 0] * J[..., 1, 1] - J[..., 0, 1] * J[..., 1, 0]
    inv = jnp.linalg.inv(J)
    # dN/dx_k = sum_l dN/dxi_l * dxi_l/dx_k
    G = jnp.einsum("qal,eqlk->eqak", dN, inv)
    return G, jnp.abs(det)


COORDS = ("planar", "axisym")


def _measure(coord, wd, xq):
    """Integration weights of the chosen coordinate system.

    planar : dA = dy dz            (per unit depth)
    axisym : dA = r dr dz          (per radian; x[..., 0] = r >= 0)
    """
    if coord == "planar":
        return wd
    if coord == "axisym":
        return wd * xq[..., 0]
    raise ValueError(f"coord must be one of {COORDS}, got {coord!r}")


def element_geometry(X, tri, coord="planar"):
    """Everything the element loop needs, at quadrature points.

    For coord="axisym" the weights wd include the factor r of the
    axisymmetric measure; h stays the plain element size.
    """
    points, weights = tri_quadrature()
    N, dN = p2_tables(points)
    L = jax.vmap(p1_shape)(jnp.asarray(points))
    Xe = jnp.asarray(X)[tri]
    G, det = physical_gradients(Xe, dN)
    xq = jnp.einsum("qa,eak->eqk", N, Xe)
    geo = {
        "G": G,                                       # (E, Q, 6, 2)
        "wd": _measure(coord, weights[None, :] * det, xq),   # (E, Q)
        "xq": xq,                                     # (E, Q, 2)
        "h": jnp.sqrt(2.0 * jnp.sum(weights[None, :] * det, axis=1)),
    }
    return geo, N, L


def nodal_gradients(X, tri):
    """Physical gradients of the 6 shapes at the 6 element nodes.

    Returns (E, 6, 6, 2): [element, at node, of shape, direction].
    Used to recover smooth nodal gradients (E field, vorticity...).
    """
    _, dN = p2_tables(P2_NODES)
    Xe = jnp.asarray(X)[tri]
    G, _ = physical_gradients(Xe, dN)
    return G


def edge_geometry(X, edges, sign, coord="planar"):
    """Boundary edge data at line quadrature points.

    edges : (B, 3) node ids
    sign  : (B,) +1 or -1 so that the normal points out of the domain
    coord : "axisym" multiplies the weights by r (edges on r = 0 get 0)
    """
    t, w = line_quadrature()
    t = jnp.asarray(t)
    M = jax.vmap(line_shape)(t)                      # (Qb, 3)
    dM = jax.vmap(jax.jacfwd(line_shape))(t)         # (Qb, 3)
    Xb = jnp.asarray(X)[edges]                        # (B, 3, 2)
    tangent = jnp.einsum("qa,bak->bqk", dM, Xb)       # (B, Qb, 2)
    length = jnp.linalg.norm(tangent, axis=-1)        # (B, Qb)
    nx = tangent[..., 1] / length
    ny = -tangent[..., 0] / length
    normal = jnp.stack([nx, ny], axis=-1) * sign[:, None, None]
    xq = jnp.einsum("qa,bak->bqk", M, Xb)
    data = {
        "wd": _measure(coord, w[None, :] * length, xq),
        "n": normal,
        "xq": xq,
    }
    return data, M
