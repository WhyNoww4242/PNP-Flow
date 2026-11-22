"""Boundary condition types.

A value can be:
  a number            -> constant
  a string            -> name of a model parameter
  a function(x, p)    -> x is (m, 2) node coordinates, p the parameter dict

Boundaries that are not mentioned are natural:
  species  -> zero total flux (blocking)
  phi      -> zero normal field (no surface charge)
  velocity -> zero traction ("do nothing")
"""
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass
class Dirichlet:
    """Fixed value.  For velocity use a pair (vy, vz) or a function."""
    value: Any


@dataclass
class Flux:
    """Prescribed outward total flux N.n (species only)."""
    value: Any


@dataclass
class Robin:
    """Stern layer: delta^2 dphi/dn = coef * (ref - phi)."""
    coef: Any
    ref: Any


@dataclass
class Charge:
    """Surface charge: delta^2 dphi/dn = sigma."""
    sigma: Any


@dataclass
class Component:
    """Fix one velocity component (0 = y, 1 = z).  Used for symmetry."""
    index: int
    value: Any = 0.0


@dataclass
class Pressure:
    """Normal traction: sigma.n = -p0 n  (p0 = 0 is traction free)."""
    p0: Any = 0.0


OUTFLOW = "outflow"     # species: diffusive flux zero, convection leaves
BLOCKING = "blocking"   # species: zero total flux
NOSLIP = "noslip"       # velocity zero
FREE = "free"           # velocity: traction free


def resolve(spec, x, params):
    """Turn a value spec into an array at points x (m, 2)."""
    m = x.shape[0]
    if callable(spec):
        return jnp.asarray(spec(x, params))
    if isinstance(spec, str):
        return jnp.full(m, 1.0) * params[spec]
    return jnp.full(m, float(spec))


def resolve_vector(spec, x, params):
    """Velocity value: pair of scalar specs, or a function -> (m, 2)."""
    if callable(spec):
        return jnp.asarray(spec(x, params))
    vy = resolve(spec[0], x, params)
    vz = resolve(spec[1], x, params)
    return jnp.stack([vy, vz], axis=1)
