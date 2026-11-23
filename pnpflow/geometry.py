"""gmsh geometries.  Every builder returns a P2 Mesh.

Mesh size is controlled with a distance field:
  h = h_wall inside `layer` of the walls, growing to h_far at `grow`.
"""
import numpy as np
import gmsh

from .mesh import gmsh_session, mesh_from_gmsh


def _wall_size_field(curves, h_wall, h_far, layer, grow, extra=None):
    """Distance-based size field around the given curves."""
    f = gmsh.model.mesh.field
    dist = f.add("Distance")
    f.setNumbers(dist, "CurvesList", curves)
    f.setNumber(dist, "Sampling", 400)
    thr = f.add("Threshold")
    f.setNumber(thr, "InField", dist)
    f.setNumber(thr, "SizeMin", h_wall)
    f.setNumber(thr, "SizeMax", h_far)
    f.setNumber(thr, "DistMin", layer)
    f.setNumber(thr, "DistMax", grow)
    fields = [thr] + (extra or [])
    final = f.add("Min")
    f.setNumbers(final, "FieldsList", fields)
    f.setAsBackgroundMesh(final)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)


def _finish(save=None):
    gmsh.option.setNumber("Mesh.Algorithm", 6)        # Frontal-Delaunay
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)                        # curved P2 elements
    gmsh.model.mesh.optimize("HighOrder")
    if save:
        gmsh.write(save)
    return mesh_from_gmsh()


def _name(dim, tags, name):
    tag = gmsh.model.addPhysicalGroup(dim, list(tags))
    gmsh.model.setPhysicalName(dim, tag, name)


# ---------------------------------------------------------------------------
# Cylinder in a box
# ---------------------------------------------------------------------------

def cylinder_in_box(radius=1.0, half_width=30.0, upstream=30.0, downstream=50.0,
                    h_wall=0.02, h_far=3.0, layer=0.3, grow=12.0,
                    wake_size=None, save=None):
    """Circle of `radius` at the origin; flow comes from +z (top).

    Boundaries: wall (circle), inlet (top), outlet (bottom), side (left+right).
    """
    with gmsh_session("cylinder"):
        occ = gmsh.model.occ
        box = occ.addRectangle(-half_width, -downstream, 0,
                               2 * half_width, upstream + downstream)
        disk = occ.addDisk(0, 0, 0, radius, radius)
        surf, _ = occ.cut([(2, box)], [(2, disk)])
        occ.synchronize()

        groups = {"wall": [], "inlet": [], "outlet": [], "side": []}
        for dim, tag in gmsh.model.getBoundary(surf, oriented=False):
            x, y, _ = gmsh.model.occ.getCenterOfMass(dim, tag)
            if np.hypot(x, y) < 2 * radius:
                groups["wall"].append(tag)
            elif y > upstream - 1e-6:
                groups["inlet"].append(tag)
            elif y < -downstream + 1e-6:
                groups["outlet"].append(tag)
            else:
                groups["side"].append(tag)
        for name, tags in groups.items():
            _name(1, tags, name)
        _name(2, [s[1] for s in surf], "fluid")

        extra = []
        if wake_size:
            extra.append(_wake_box(radius, downstream, wake_size, h_far))
        _wall_size_field(groups["wall"], h_wall, h_far, layer, grow, extra)
        return _finish(save)


def _wake_box(radius, downstream, size, h_far):
    f = gmsh.model.mesh.field
    box = f.add("Box")
    f.setNumber(box, "VIn", size)
    f.setNumber(box, "VOut", h_far)
    f.setNumber(box, "XMin", -2.5 * radius)
    f.setNumber(box, "XMax", 2.5 * radius)
    f.setNumber(box, "YMin", -min(downstream, 12 * radius))
    f.setNumber(box, "YMax", 2.5 * radius)
    f.setNumber(box, "Thickness", 4 * radius)
    return box


# ---------------------------------------------------------------------------
# Rectangle (channels, 1D-like strips, manufactured tests)
# ---------------------------------------------------------------------------

def rectangle(x0, x1, y0, y1, h=0.1, wall_names=None, h_wall=None,
              layer=0.0, grow=1.0, save=None):
    """Rectangle with named sides "left", "right", "bottom", "top".

    wall_names: sides to refine towards (size h_wall).
    """
    with gmsh_session("rectangle"):
        occ = gmsh.model.occ
        surf = occ.addRectangle(x0, y0, 0, x1 - x0, y1 - y0)
        occ.synchronize()
        sides = {}
        for dim, tag in gmsh.model.getBoundary([(2, surf)], oriented=False):
            x, y, _ = occ.getCenterOfMass(dim, tag)
            if abs(x - x0) < 1e-9:
                sides["left"] = tag
            elif abs(x - x1) < 1e-9:
                sides["right"] = tag
            elif abs(y - y0) < 1e-9:
                sides["bottom"] = tag
            else:
                sides["top"] = tag
        for name, tag in sides.items():
            _name(1, [tag], name)
        _name(2, [surf], "domain")

        if wall_names and h_wall:
            curves = [sides[n] for n in wall_names]
            _wall_size_field(curves, h_wall, h, layer, grow)
        else:
            gmsh.option.setNumber("Mesh.MeshSizeMin", h)
            gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        return _finish(save)


# ---------------------------------------------------------------------------
# Axisymmetric sphere (r-z half plane)
# ---------------------------------------------------------------------------

def _half_disk(occ, radius, big):
    """Half disk r >= 0 of the given radius, centred at the origin."""
    disk = occ.addDisk(0, 0, 0, radius, radius)
    left = occ.addRectangle(-2 * big, -2 * big, 0, 2 * big, 4 * big)
    out, _ = occ.cut([(2, disk)], [(2, left)])
    return out


def sphere_axisym(R=1.0, R_out=10.0, h_sphere=0.05, h_far=0.5, grow=None,
                  save=None):
    """Sphere of radius R inside a sphere of radius R_out, in the r-z plane.

    Mesh coordinates: x = r >= 0, y = z.  The domain is the half disk
    r^2 + z^2 <= R_out^2, r >= 0, fragmented with the half disk of radius R
    so element edges follow the sphere surface (curved P2 edges).

    Physical groups:
      curves   : axis  (r = 0, three segments; natural BC = symmetry)
                 outer (the arc rho = R_out, split at z = 0 by OCC)
      surfaces : sphere (rho < R), medium (R < rho < R_out)
    The interior arc rho = R gets no physical group (boundary normals are
    taken from a single owning triangle, see mesh.Mesh); it only drives the
    size field: h = h_sphere at the arc, growing to h_far over `grow`.
    """
    grow = grow or 0.5 * R_out
    with gmsh_session("sphere_axisym"):
        occ = gmsh.model.occ
        outer = _half_disk(occ, R_out, R_out)
        inner = _half_disk(occ, R, R_out)
        surf, _ = occ.fragment(outer, inner)
        occ.synchronize()

        tol = 1e-6 * R_out
        regions = {"sphere": [], "medium": []}
        for dim, tag in surf:
            xmax = occ.getBoundingBox(dim, tag)[3]
            regions["sphere" if xmax < R + tol else "medium"].append(tag)

        curves = {"axis": [], "outer": [], "interface": []}
        for dim, tag in gmsh.model.getBoundary(surf, combined=False, oriented=False):
            x0, y0, _, x1, y1, _ = occ.getBoundingBox(dim, tag)
            if x1 < tol:
                key = "axis"
            elif x1 > R_out - tol:
                key = "outer"
            else:
                key = "interface"
            if tag not in curves[key]:
                curves[key].append(tag)
        if not (curves["axis"] and curves["outer"] and curves["interface"]
                and regions["sphere"] and regions["medium"]):
            raise RuntimeError(f"sphere_axisym: unexpected topology {curves} {regions}")

        _name(1, curves["axis"], "axis")
        _name(1, curves["outer"], "outer")
        _name(2, regions["sphere"], "sphere")
        _name(2, regions["medium"], "medium")
        _wall_size_field(curves["interface"], h_sphere, h_far, 0.0, grow)
        mesh = _finish(save)
    # gmsh leaves ~1e-13 round-off on the axis; put those nodes exactly at r = 0
    on_axis = np.abs(mesh.X[:, 0]) < 1e-9 * R_out
    mesh.X[on_axis, 0] = 0.0
    return mesh


def load_msh(path):
    """Load any order-2 gmsh .msh file with physical groups."""
    with gmsh_session("loaded"):
        gmsh.open(path)
        if gmsh.model.mesh.getElementTypes(2).tolist() != [9]:
            gmsh.model.mesh.setOrder(2)
        return mesh_from_gmsh()


# ---------------------------------------------------------------------------
# Axisymmetric pore with a cylindrical reservoir (SDL) in front of it
# ---------------------------------------------------------------------------

def pore_axisym(L=100.0, alpha_in=1.0, alpha_out=1.0, sdl_radius=2.0,
                sdl_length=51.5, h_wall=0.125, h_entr=0.0625, h_far=1.0,
                wall_layer=None, entr_radius=None, save=None):
    """Pore 0 <= z <= L with wall r = alpha(z), alpha linear from alpha_in (mouth)
    to alpha_out (end), fed by a cylinder r <= sdl_radius, -sdl_length <= z <= 0.

    Mesh coordinates: x = r >= 0, y = z.  Lengths in units of the reference pore
    radius a_p, so alpha is the paper's alpha(Z) and Z = z / L.

    Physical groups:
      curves   : axis (r = 0), sdl_far (z = -sdl_length, bulk), sdl_side
                 (r = sdl_radius), sdl_face (z = 0, alpha_in < r < sdl_radius),
                 pore_wall (r = alpha(z)), pore_end (z = L)
      surfaces : sdl (z < 0), pore (z > 0)
    Sizes: h_wall within `wall_layer` of the pore wall and end (the whole pore
    when the pore is no wider than the layer), h_entr inside `entr_radius` of
    the mouth corner (the re-entrant corner where the charged wall meets the
    uncharged SDL face, plus the access region), growing to h_far in the SDL.
    """
    a_max = max(alpha_in, alpha_out)
    if not (0 < alpha_in < sdl_radius and alpha_out > 0 and L > 0 and sdl_length > 0):
        raise ValueError("pore_axisym needs 0 < alpha_in < sdl_radius, alpha_out > 0")
    wall_layer = wall_layer if wall_layer is not None else a_max
    entr_radius = entr_radius if entr_radius is not None else 1.5 * a_max
    with gmsh_session("pore_axisym"):
        occ = gmsh.model.occ
        P = [occ.addPoint(*xy, 0) for xy in [
            (0.0, -sdl_length), (sdl_radius, -sdl_length), (sdl_radius, 0.0),
            (alpha_in, 0.0), (alpha_out, L), (0.0, L), (0.0, 0.0)]]
        lines = {
            "sdl_far": occ.addLine(P[0], P[1]),
            "sdl_side": occ.addLine(P[1], P[2]),
            "sdl_face": occ.addLine(P[2], P[3]),
            "pore_wall": occ.addLine(P[3], P[4]),
            "pore_end": occ.addLine(P[4], P[5]),
            "axis_pore": occ.addLine(P[5], P[6]),
            "axis_sdl": occ.addLine(P[6], P[0]),
            "mouth": occ.addLine(P[6], P[3]),        # interior line z = 0
        }
        l = lines
        sdl_loop = occ.addCurveLoop([l["sdl_far"], l["sdl_side"], l["sdl_face"],
                                     -l["mouth"], l["axis_sdl"]])
        pore_loop = occ.addCurveLoop([l["mouth"], l["pore_wall"], l["pore_end"],
                                      l["axis_pore"]])
        s_sdl = occ.addPlaneSurface([sdl_loop])
        s_pore = occ.addPlaneSurface([pore_loop])
        occ.synchronize()

        for name in ("sdl_far", "sdl_side", "sdl_face", "pore_wall", "pore_end"):
            _name(1, [l[name]], name)
        _name(1, [l["axis_pore"], l["axis_sdl"]], "axis")
        _name(2, [s_sdl], "sdl")
        _name(2, [s_pore], "pore")

        f = gmsh.model.mesh.field
        # pore wall + end: EDL resolution
        d_wall = f.add("Distance")
        f.setNumbers(d_wall, "CurvesList", [l["pore_wall"], l["pore_end"]])
        f.setNumber(d_wall, "Sampling", int(max(400, 4 * L / h_wall)))
        t_wall = f.add("Threshold")
        f.setNumber(t_wall, "InField", d_wall)
        f.setNumber(t_wall, "SizeMin", h_wall)
        f.setNumber(t_wall, "SizeMax", h_far)
        f.setNumber(t_wall, "DistMin", wall_layer)
        f.setNumber(t_wall, "DistMax", wall_layer + 8 * a_max)
        # mouth corner: extra refinement (re-entrant corner, access resistance)
        d_corner = f.add("Distance")
        f.setNumbers(d_corner, "PointsList", [P[3]])
        t_corner = f.add("Threshold")
        f.setNumber(t_corner, "InField", d_corner)
        f.setNumber(t_corner, "SizeMin", h_entr)
        f.setNumber(t_corner, "SizeMax", h_far)
        f.setNumber(t_corner, "DistMin", 0.25 * entr_radius)
        f.setNumber(t_corner, "DistMax", 0.25 * entr_radius + 12 * a_max)
        # access region in front of the mouth: at least h_wall inside entr_radius
        d_mouth = f.add("Distance")
        f.setNumbers(d_mouth, "CurvesList", [l["mouth"]])
        t_mouth = f.add("Threshold")
        f.setNumber(t_mouth, "InField", d_mouth)
        f.setNumber(t_mouth, "SizeMin", h_wall)
        f.setNumber(t_mouth, "SizeMax", h_far)
        f.setNumber(t_mouth, "DistMin", entr_radius)
        f.setNumber(t_mouth, "DistMax", entr_radius + 12 * a_max)
        final = f.add("Min")
        f.setNumbers(final, "FieldsList", [t_wall, t_corner, t_mouth])
        f.setAsBackgroundMesh(final)
        gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
        gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
        mesh = _finish(save)
    on_axis = np.abs(mesh.X[:, 0]) < 1e-9 * L
    mesh.X[on_axis, 0] = 0.0
    return mesh
