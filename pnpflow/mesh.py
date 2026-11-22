"""Meshes from gmsh.

A Mesh holds a P2 (6-node) triangle mesh and its named boundaries.
Names come from gmsh physical groups.
"""
import contextlib
import numpy as np
import gmsh

TRI6 = 9    # gmsh element type: 6-node triangle
LINE3 = 8   # gmsh element type: 3-node line


@contextlib.contextmanager
def gmsh_session(name="model", verbose=False):
    """Open gmsh, yield, always close."""
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", 1 if verbose else 0)
    gmsh.model.add(name)
    try:
        yield gmsh
    finally:
        gmsh.finalize()


class Mesh:
    """P2 triangle mesh.

    X          : (n, 2) node coordinates
    tri        : (E, 6) node ids, gmsh order (corners, then mid 01, 12, 20)
    boundaries : {name: {"edges": (B, 3) ids, "sign": (B,) +-1}}
    regions    : {name: (k,) element ids} from physical surfaces (optional)
    coord      : "planar" (x = y, z) or "axisym" (x = r, z); set by Model

    Physical curves must lie on the outer boundary: an edge's outward
    normal is taken from the single triangle that owns it.
    """

    def __init__(self, X, tri, boundaries, regions=None):
        self.X = np.asarray(X, dtype=float)
        self.tri = np.asarray(tri, dtype=np.int64)
        self.boundaries = boundaries
        self.regions = regions or {}
        self.coord = "planar"
        self._number_vertices()
        self._orient_boundaries()

    # -- sizes --------------------------------------------------------------
    @property
    def n_nodes(self):
        return len(self.X)

    @property
    def n_elements(self):
        return len(self.tri)

    @property
    def n_vertices(self):
        return len(self.vertices)

    # -- setup --------------------------------------------------------------
    def _number_vertices(self):
        """Give each corner node a P1 index (used by pressure)."""
        self.vertices = np.unique(self.tri[:, :3])
        self.p1_index = -np.ones(self.n_nodes, dtype=np.int64)
        self.p1_index[self.vertices] = np.arange(len(self.vertices))
        self.tri_p1 = self.p1_index[self.tri[:, :3]]

    def _orient_boundaries(self):
        """Choose each edge sign so its normal points out of the domain.

        Every P2 edge has its own mid node, so the mid node tells us
        which triangle owns a boundary edge.
        """
        owner = -np.ones(self.n_nodes, dtype=np.int64)
        for k in range(3, 6):
            owner[self.tri[:, k]] = np.arange(self.n_elements)
        for name, b in self.boundaries.items():
            edges = b["edges"]
            a = self.X[edges[:, 0]]
            c = self.X[edges[:, 1]]
            tangent = c - a
            normal = np.stack([tangent[:, 1], -tangent[:, 0]], axis=1)
            centroid = self.X[self.tri[owner[edges[:, 2]], :3]].mean(axis=1)
            outward = self.X[edges[:, 2]] - centroid
            sign = np.where(np.sum(normal * outward, axis=1) > 0, 1.0, -1.0)
            b["sign"] = sign

    # -- queries ------------------------------------------------------------
    def boundary_nodes(self, name):
        return np.unique(self.boundaries[name]["edges"])

    def quality(self):
        """Shape quality of each triangle: 1 = equilateral, 0 = flat."""
        P = self.X[self.tri[:, :3]]
        e0 = P[:, 1] - P[:, 0]
        e1 = P[:, 2] - P[:, 1]
        e2 = P[:, 0] - P[:, 2]
        area = 0.5 * np.abs(e0[:, 0] * (-e2[:, 1]) - e0[:, 1] * (-e2[:, 0]))
        sum_sq = (e0 ** 2).sum(1) + (e1 ** 2).sum(1) + (e2 ** 2).sum(1)
        return 4.0 * np.sqrt(3.0) * area / sum_sq

    def edge_lengths(self):
        P = self.X[self.tri[:, :3]]
        lengths = [np.linalg.norm(P[:, i] - P[:, (i + 1) % 3], axis=1)
                   for i in range(3)]
        return np.concatenate(lengths)

    def linear_triangles(self):
        """Split each P2 triangle into 4 linear ones (for plotting)."""
        t = self.tri
        parts = [
            t[:, [0, 3, 5]], t[:, [3, 1, 4]],
            t[:, [5, 4, 2]], t[:, [3, 4, 5]],
        ]
        return np.concatenate(parts)

    def stats(self):
        q = self.quality()
        h = self.edge_lengths()
        counts, edges = np.histogram(q, bins=20, range=(0, 1))
        return {
            "n_elements": int(self.n_elements),
            "n_nodes": int(self.n_nodes),
            "n_vertices": int(self.n_vertices),
            "h_min": float(h.min()),
            "h_max": float(h.max()),
            "quality_min": float(q.min()),
            "quality_mean": float(q.mean()),
            "quality_hist": {"edges": edges.tolist(), "counts": counts.tolist()},
            "boundaries": {k: int(len(v["edges"])) for k, v in self.boundaries.items()},
        }


def mesh_from_gmsh():
    """Read the current gmsh model (already meshed, order 2) into a Mesh."""
    tags, coords, _ = gmsh.model.mesh.getNodes()
    tags = tags.astype(np.int64)
    coords = coords.reshape(-1, 3)
    lookup = np.zeros(tags.max() + 1, dtype=np.int64)
    lookup[tags] = np.arange(len(tags))

    tri = _elements_of_dim(2, TRI6, 6, lookup)
    regions, start = {}, 0
    for dim, tag in gmsh.model.getPhysicalGroups(2):
        k = len(_elements_in_group(dim, tag, TRI6, 6, lookup))
        name = gmsh.model.getPhysicalName(dim, tag) or f"surface_{tag}"
        regions[name] = np.arange(start, start + k)
        start += k
    boundaries = {}
    for dim, tag in gmsh.model.getPhysicalGroups(1):
        name = gmsh.model.getPhysicalName(dim, tag)
        edges = _elements_in_group(dim, tag, LINE3, 3, lookup)
        boundaries[name] = {"edges": edges}

    # Keep only nodes used by triangles, numbered 0..n-1.
    used = np.unique(tri)
    renumber = -np.ones(len(tags), dtype=np.int64)
    renumber[used] = np.arange(len(used))
    X = coords[used, :2]
    tri = renumber[tri]
    for b in boundaries.values():
        b["edges"] = renumber[b["edges"]]
    return Mesh(X, tri, boundaries, regions)


def _elements_of_dim(dim, etype, n_per, lookup):
    blocks = []
    for d, tag in gmsh.model.getPhysicalGroups(dim):
        blocks.append(_elements_in_group(d, tag, etype, n_per, lookup))
    return np.concatenate(blocks)


def _elements_in_group(dim, tag, etype, n_per, lookup):
    blocks = []
    for entity in gmsh.model.getEntitiesForPhysicalGroup(dim, tag):
        types, _, nodes = gmsh.model.mesh.getElements(dim, entity)
        for t, n in zip(types, nodes):
            if t == etype:
                blocks.append(lookup[n.astype(np.int64)].reshape(-1, n_per))
    return np.concatenate(blocks)
