"""gmsh geometries.  Every builder returns a P2 Mesh.

Mesh size is controlled with a distance field:
  h = h_wall inside `layer` of the walls, growing to h_far at `grow`.
"""
import contextlib
import os

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
    ctx = MESH_CTX
    if ctx["load"] and os.path.isfile(ctx["load"]):
        # reuse a mesh built earlier for the same case (same mesh key)
        gmsh.clear()
        gmsh.open(ctx["load"])
        ctx["info"]["loaded"] = ctx["load"]
        return mesh_from_gmsh()
    if ctx["opts"]:
        apply_mesh_options(ctx["opts"], ctx["info"])
    gmsh.option.setNumber("Mesh.Algorithm", 6)        # Frontal-Delaunay
    gmsh.model.mesh.generate(2)
    gmsh.model.mesh.setOrder(2)                        # curved P2 elements
    gmsh.model.mesh.optimize("HighOrder")
    for path in (save, ctx["save"]):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            gmsh.option.setNumber("Mesh.SaveAll", 1)   # every element, groups kept
            gmsh.write(path)
            ctx["info"]["saved"] = path
    return mesh_from_gmsh()


# ---------------------------------------------------------------------------
# Case mesh options: density, boundary / region refinement, sharp corners, paint
# ---------------------------------------------------------------------------
MESH_CTX = {"opts": None, "load": None, "save": None, "info": {}}


@contextlib.contextmanager
def mesh_options(opts=None, load=None, save=None):
    """Apply a case's mesh block to every gmsh builder called inside the block.

    load : path of a saved mesh to read instead of meshing (the builder still draws
           its geometry, which is cheap; only the meshing is skipped)
    save : path to write the generated mesh to (msh4, named physical groups)
    Yields an info dict: corners found, fields added, density, loaded / saved path.
    """
    old = dict(MESH_CTX)
    info = {}
    MESH_CTX.update(opts=opts, load=load, save=save, info=info)
    try:
        yield info
    finally:
        MESH_CTX.clear()
        MESH_CTX.update(old)


def _ramp(in_field, size, dist_min, growth):
    """Size `size` up to distance dist_min of in_field, then growing linearly with
    slope (growth - 1) so neighbouring elements differ by about `growth`; no
    influence once it exceeds 1000 size (StopAtDistMax)."""
    f = gmsh.model.mesh.field
    g = max(1.01, float(growth or 1.2))
    size = float(size)
    size_max = 1000.0 * size
    t = f.add("Threshold")
    f.setNumber(t, "InField", in_field)
    f.setNumber(t, "SizeMin", size)
    f.setNumber(t, "SizeMax", size_max)
    f.setNumber(t, "DistMin", max(0.0, float(dist_min)))
    f.setNumber(t, "DistMax", max(0.0, float(dist_min)) + (size_max - size) / (g - 1.0))
    f.setNumber(t, "StopAtDistMax", 1)
    return t


def _n(v):
    """A number for a gmsh MathEval expression: always parenthesised (no '--')."""
    v = float(v)
    if not np.isfinite(v):
        raise ValueError("mesh options: coordinates must be finite numbers")
    return f"({v!r})"


def _curves_of_group(name):
    tags = []
    for dim, tag in gmsh.model.getPhysicalGroups(1):
        if gmsh.model.getPhysicalName(dim, tag) == name:
            tags += [int(c) for c in gmsh.model.getEntitiesForPhysicalGroup(dim, tag)]
    return tags


def _model_bbox():
    xs, ys = [], []
    for _, s in gmsh.model.getEntities(2):
        b = gmsh.model.getBoundingBox(2, s)
        xs += [b[0], b[3]]
        ys += [b[1], b[4]]
    return min(xs), max(xs), min(ys), max(ys)


def sharp_corners(max_angle=80.0, reentrant=185.0):
    """Geometry corners where two boundary curves meet at an interior angle below
    `max_angle` (acute) or above `reentrant` (re-entrant, singular): [(tag, x, y, angle)]."""
    x0, x1, y0, y1 = _model_bbox()
    eps = 1e-4 * max(x1 - x0, y1 - y0)
    surfaces = [s for _, s in gmsh.model.getEntities(2)]
    out = []
    for _, p in gmsh.model.getEntities(0):
        P = np.array(gmsh.model.getValue(0, p, []))[:2]
        up, _ = gmsh.model.getAdjacencies(0, p)
        bnd = [int(c) for c in up if len(gmsh.model.getAdjacencies(1, int(c))[0]) == 1]
        if len(bnd) != 2:
            continue
        dirs = []
        for c in bnd:
            lo, hi = gmsh.model.getParametrizationBounds(1, c)
            u0, u1 = float(lo[0]), float(hi[0])
            a = np.array(gmsh.model.getValue(1, c, [u0]))[:2]
            b = np.array(gmsh.model.getValue(1, c, [u1]))[:2]
            at_start = np.linalg.norm(a - P) <= np.linalg.norm(b - P)
            d = np.array(gmsh.model.getDerivative(1, c, [u0 if at_start else u1]))[:2]
            d = d if at_start else -d                  # pointing away from the corner
            n = np.linalg.norm(d)
            if n == 0:
                break
            dirs.append(d / n)
        if len(dirs) != 2:
            continue
        ang = float(np.degrees(np.arccos(np.clip(dirs[0] @ dirs[1], -1, 1))))
        if ang > 179.0:                                # straight continuation
            continue
        bis = dirs[0] + dirs[1]
        bis = bis / np.linalg.norm(bis)
        q = P + eps * bis
        inside = any(gmsh.model.isInside(2, s, [q[0], q[1], 0.0]) for s in surfaces)
        interior = ang if inside else 360.0 - ang
        if interior < max_angle or interior > reentrant:
            out.append((int(p), float(P[0]), float(P[1]), interior))
    return out


def apply_mesh_options(opts, info):
    """Add the case's refinement fields to the builder's background field."""
    f = gmsh.model.mesh.field
    density = float(opts.get("density") or 1.0)
    if density <= 0:
        raise ValueError("mesh density must be > 0")
    gmsh.option.setNumber("Mesh.MeshSizeFactor", 1.0 / density)
    info["density"] = density
    added = []
    for b in opts.get("boundaries") or []:
        curves = _curves_of_group(b.get("boundary"))
        if not curves:
            raise ValueError(f"mesh refinement: no boundary named '{b.get('boundary')}'")
        if not float(b.get("size") or 0) > 0:
            raise ValueError(f"mesh refinement on '{b.get('boundary')}': size must be > 0")
        d = f.add("Distance")
        f.setNumbers(d, "CurvesList", curves)
        f.setNumber(d, "Sampling", 600)
        added.append(_ramp(d, b["size"], b.get("layer") or 0.0, b.get("growth")))
    for r in opts.get("regions") or []:
        if not float(r.get("size") or 0) > 0:
            raise ValueError("mesh region: size must be > 0")
        m = f.add("MathEval")
        if r.get("kind") == "box":
            xa, xb = sorted((float(r["x0"]), float(r["x1"])))
            ya, yb = sorted((float(r["y0"]), float(r["y1"])))
            f.setString(m, "F", f"Sqrt(Max(Max({_n(xa)}-x,x-{_n(xb)}),0)^2+Max(Max({_n(ya)}-y,y-{_n(yb)}),0)^2)")
            added.append(_ramp(m, r["size"], 0.0, r.get("growth")))
        else:
            f.setString(m, "F", f"Sqrt((x-{_n(r['x'])})^2+(y-{_n(r['y'])})^2)")
            added.append(_ramp(m, r["size"], float(r.get("r") or 0.0), r.get("growth")))
    corners = opts.get("corners") or {}
    if corners.get("enabled"):
        found = sharp_corners(float(corners.get("max_angle") or 80.0))
        x0, x1, y0, y1 = _model_bbox()
        radius = float(corners.get("radius") or 0) or 0.02 * np.hypot(x1 - x0, y1 - y0)
        size = float(corners.get("size") or 0) or radius / 8
        info["corners"] = [{"x": x, "y": y, "angle": a} for _, x, y, a in found]
        if found:
            d = f.add("Distance")
            f.setNumbers(d, "PointsList", [t for t, *_ in found])
            added.append(_ramp(d, size, radius, 1.2))
        info["corner_size"], info["corner_radius"] = size, radius
    paint = opts.get("paint") or []
    if paint:
        by_size = {}
        for s in paint:
            by_size.setdefault(float(s["size"]), []).append(s)
        for size, spots in by_size.items():
            if not size > 0:
                raise ValueError("paint spots need a size > 0")
            terms = [f"Sqrt((x-{_n(s['x'])})^2+(y-{_n(s['y'])})^2)-{_n(s.get('r') or 0)}"
                     for s in spots]
            # chunks of 60 spots: a long nested Min(...) overflows gmsh's expression parser
            for i in range(0, len(terms), 60):
                chunk = terms[i:i + 60]
                expr = chunk[0]
                for t in chunk[1:]:
                    expr = f"Min({expr},{t})"
                m = f.add("MathEval")
                f.setString(m, "F", f"Max({expr},0)")
                added.append(_ramp(m, size, 0.0, opts.get("paint_growth") or 1.2))
        info["paint_spots"] = len(paint)
    if added:
        # combine with the builder's own background field (the Min not used by another Min)
        mins = [t for t in f.list() if f.getType(t) == "Min" and t not in added]
        used = set()
        for t in mins:
            used.update(int(x) for x in f.getNumbers(t, "FieldsList"))
        base = [t for t in mins if t not in used]
        top = f.add("Min")
        f.setNumbers(top, "FieldsList", added + base[-1:])
        f.setAsBackgroundMesh(top)
    info["fields_added"] = len(added)


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
