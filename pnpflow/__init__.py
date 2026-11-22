"""pnpflow: steady Poisson-Nernst-Planck + Navier-Stokes finite elements.

Built on gmsh (meshing) and JAX (assembly, Jacobians, sensitivities).
"""
import jax

# Double precision is required for Newton on stiff PNP systems.
jax.config.update("jax_enable_x64", True)

from .model import Model, Species  # noqa: E402,F401
from .mesh import Mesh  # noqa: E402,F401
