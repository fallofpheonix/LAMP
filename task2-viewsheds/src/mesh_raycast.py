"""
Optional mesh-based ray tracing with BVH acceleration for Task-2 viewsheds.

Provides an alternative to voxel-based LOS queries when triangle mesh geometry
is available.  Falls back gracefully to voxel methods when not used.

Classes
-------
Triangle
    Stores the three vertices of a triangle and provides AABB/centroid helpers.
BVHNode
    Bounding-volume-hierarchy node wrapping a list of triangles.
MeshScene
    Convenience container holding a list of triangles with a BVH root.

Functions
---------
moller_trumbore(ray_origin, ray_dir, triangle)
    Möller–Trumbore ray–triangle intersection; returns (hit, t) where *t* is
    the parametric distance along the ray.
build_bvh(triangles, max_leaf_size)
    Build a BVH from a flat list of triangles.
mesh_is_visible(mesh_scene, start_xyz, end_xyz)
    Test whether the segment from *start_xyz* to *end_xyz* is unoccluded by
    any triangle in the mesh.
dem_to_mesh(terrain, geotransform)
    Tessellate a DEM grid into two triangles per cell.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Primitive geometry
# ---------------------------------------------------------------------------

@dataclass
class Triangle:
    """An axis-aligned-free triangle defined by three 3-D vertices (float64)."""

    v0: np.ndarray  # shape (3,)
    v1: np.ndarray  # shape (3,)
    v2: np.ndarray  # shape (3,)

    def __post_init__(self) -> None:
        self.v0 = np.asarray(self.v0, dtype=np.float64)
        self.v1 = np.asarray(self.v1, dtype=np.float64)
        self.v2 = np.asarray(self.v2, dtype=np.float64)

    def aabb_min(self) -> np.ndarray:
        """Axis-aligned bounding box minimum corner."""
        return np.minimum(self.v0, np.minimum(self.v1, self.v2))

    def aabb_max(self) -> np.ndarray:
        """Axis-aligned bounding box maximum corner."""
        return np.maximum(self.v0, np.maximum(self.v1, self.v2))

    def centroid(self) -> np.ndarray:
        """Triangle centroid."""
        return (self.v0 + self.v1 + self.v2) / 3.0


# ---------------------------------------------------------------------------
# Möller–Trumbore intersection
# ---------------------------------------------------------------------------

_EPS = 1e-10


def moller_trumbore(
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    tri: Triangle,
) -> tuple[bool, float]:
    """Möller–Trumbore ray–triangle intersection test.

    Parameters
    ----------
    ray_origin:
        3-D origin of the ray, shape (3,).
    ray_dir:
        3-D direction vector (need not be normalised), shape (3,).
    tri:
        Triangle to test against.

    Returns
    -------
    (hit, t)
        *hit* is ``True`` when the ray intersects *tri* at parametric
        distance ``t > 0`` (either face, depending on ray direction and
        triangle winding).  *t* is ``float('inf')`` when there is no
        intersection.
    """
    edge1 = tri.v1 - tri.v0
    edge2 = tri.v2 - tri.v0
    h = np.cross(ray_dir, edge2)
    a = float(np.dot(edge1, h))

    if abs(a) < _EPS:
        return False, float("inf")  # Ray is parallel to triangle.

    inv_a = 1.0 / a
    s = ray_origin - tri.v0
    u = inv_a * float(np.dot(s, h))

    if u < 0.0 or u > 1.0:
        return False, float("inf")

    q = np.cross(s, edge1)
    v = inv_a * float(np.dot(ray_dir, q))

    if v < 0.0 or (u + v) > 1.0:
        return False, float("inf")

    t = inv_a * float(np.dot(edge2, q))
    if t < _EPS:
        return False, float("inf")

    return True, t


# ---------------------------------------------------------------------------
# Bounding-volume hierarchy
# ---------------------------------------------------------------------------

@dataclass
class BVHNode:
    """Binary BVH node.

    Leaf nodes have ``left is None`` and ``right is None`` and store a
    non-empty list of triangles in ``triangles``.  Internal nodes have
    ``left`` and ``right`` children and an empty ``triangles`` list.
    """

    bbox_min: np.ndarray  # (3,) float64
    bbox_max: np.ndarray  # (3,) float64
    triangles: list[Triangle] = field(default_factory=list)
    left: Optional["BVHNode"] = None
    right: Optional["BVHNode"] = None

    def is_leaf(self) -> bool:
        """Return ``True`` when this node is a leaf."""
        return self.left is None and self.right is None


def _compute_aabb(triangles: Sequence[Triangle]) -> tuple[np.ndarray, np.ndarray]:
    mins = np.stack([t.aabb_min() for t in triangles])
    maxs = np.stack([t.aabb_max() for t in triangles])
    return mins.min(axis=0), maxs.max(axis=0)


def build_bvh(triangles: list[Triangle], max_leaf_size: int = 4) -> BVHNode:
    """Recursively build a BVH from *triangles*.

    Parameters
    ----------
    triangles:
        Non-empty list of :class:`Triangle` objects.
    max_leaf_size:
        Maximum number of triangles per leaf node.

    Returns
    -------
    BVHNode
        Root of the BVH tree.
    """
    if not triangles:
        raise ValueError("Cannot build BVH from empty triangle list")

    bbox_min, bbox_max = _compute_aabb(triangles)

    if len(triangles) <= max_leaf_size:
        return BVHNode(bbox_min=bbox_min, bbox_max=bbox_max, triangles=list(triangles))

    # Split along the longest axis at the centroid median.
    extent = bbox_max - bbox_min
    axis = int(np.argmax(extent))
    centroids = np.array([t.centroid()[axis] for t in triangles])
    median = float(np.median(centroids))

    left_tris = [t for t, c in zip(triangles, centroids) if c <= median]
    right_tris = [t for t, c in zip(triangles, centroids) if c > median]

    # Degenerate split fallback: divide in half.
    if not left_tris or not right_tris:
        mid = len(triangles) // 2
        left_tris = triangles[:mid]
        right_tris = triangles[mid:]

    left_node = build_bvh(left_tris, max_leaf_size)
    right_node = build_bvh(right_tris, max_leaf_size)

    return BVHNode(bbox_min=bbox_min, bbox_max=bbox_max, left=left_node, right=right_node)


def _ray_aabb_intersect(
    ray_origin: np.ndarray,
    inv_dir: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    t_max: float,
) -> bool:
    """Slab test for ray–AABB intersection.

    This implementation is robust to zero ray-direction components
    (corresponding to infinite values in ``inv_dir``): for axes where the
    ray is effectively parallel, the ray intersects the box only if the
    origin lies within the box slab on that axis, and such axes do not
    constrain the global ``t`` interval.
    """
    tmin = -np.inf
    tmax_box = np.inf

    for i in range(3):
        inv_d = inv_dir[i]
        o = ray_origin[i]
        bmin = bbox_min[i]
        bmax = bbox_max[i]

        if not np.isfinite(inv_d):
            # Ray is parallel to this axis. It can only intersect the box if
            # the origin lies within the slab [bmin, bmax] along this axis.
            if o < bmin or o > bmax:
                return False
            axis_tmin = -np.inf
            axis_tmax = np.inf
        else:
            t1 = (bmin - o) * inv_d
            t2 = (bmax - o) * inv_d
            axis_tmin = t1 if t1 < t2 else t2
            axis_tmax = t2 if t2 > t1 else t1

        if axis_tmin > tmin:
            tmin = axis_tmin
        if axis_tmax < tmax_box:
            tmax_box = axis_tmax

    return tmax_box >= max(tmin, 0.0) and tmin < t_max


def _traverse_bvh(
    node: BVHNode,
    ray_origin: np.ndarray,
    ray_dir: np.ndarray,
    inv_dir: np.ndarray,
    t_max: float,
) -> bool:
    """Traverse the BVH; return ``True`` if any triangle is hit within [0, t_max]."""
    if not _ray_aabb_intersect(ray_origin, inv_dir, node.bbox_min, node.bbox_max, t_max):
        return False

    if node.is_leaf():
        for tri in node.triangles:
            hit, t = moller_trumbore(ray_origin, ray_dir, tri)
            if hit and t <= t_max:
                return True
        return False

    return (
        _traverse_bvh(node.left, ray_origin, ray_dir, inv_dir, t_max)
        or _traverse_bvh(node.right, ray_origin, ray_dir, inv_dir, t_max)
    )


# ---------------------------------------------------------------------------
# Mesh scene
# ---------------------------------------------------------------------------

@dataclass
class MeshScene:
    """Container for a triangle mesh with a pre-built BVH.

    Attributes
    ----------
    triangles:
        All triangles in the scene.
    bvh_root:
        Root node of the BVH built from *triangles*.
    """

    triangles: list[Triangle]
    bvh_root: BVHNode


def build_mesh_scene(triangles: list[Triangle], max_leaf_size: int = 4) -> MeshScene:
    """Build a :class:`MeshScene` from a list of triangles.

    Parameters
    ----------
    triangles:
        Non-empty list of triangles.
    max_leaf_size:
        BVH leaf size limit.

    Returns
    -------
    MeshScene
        Ready for LOS queries via :func:`mesh_is_visible`.
    """
    if not triangles:
        raise ValueError("Cannot build MeshScene from empty triangle list")
    bvh = build_bvh(triangles, max_leaf_size=max_leaf_size)
    return MeshScene(triangles=triangles, bvh_root=bvh)


def mesh_is_visible(
    scene: MeshScene,
    start_xyz: tuple[float, float, float],
    end_xyz: tuple[float, float, float],
    aperture_m: float = 0.0,
    n_samples: int = 1,
) -> float:
    """Test LOS with optional aperture sampling.

    Returns the visibility fraction [0, 1] based on *n_samples* rays
    jittered within *aperture_m* radius around the observer.
    """
    o_base = np.asarray(start_xyz, dtype=np.float64)
    target = np.asarray(end_xyz, dtype=np.float64)
    
    visibility_sum = 0.0
    
    for i in range(n_samples):
        if i == 0 or aperture_m <= 0:
            o = o_base
        else:
            # Simple disk sampling in horizontal plane for aperture
            angle = np.random.uniform(0, 2 * np.pi)
            dist = np.random.uniform(0, aperture_m)
            o = o_base + np.array([dist * np.cos(angle), dist * np.sin(angle), 0.0])
            
        d = target - o
        length = float(np.linalg.norm(d))
        if length < 1e-9:
            visibility_sum += 1.0
            continue

        with np.errstate(divide="ignore", invalid="ignore"):
            inv_dir = 1.0 / d
            
        if not _traverse_bvh(scene.bvh_root, o, d, inv_dir, t_max=1.0):
            visibility_sum += 1.0
            
    return visibility_sum / n_samples


# ---------------------------------------------------------------------------
# DEM tessellation helper
# ---------------------------------------------------------------------------

def dem_to_mesh(
    terrain: np.ndarray,
    geotransform: tuple,
) -> list[Triangle]:
    """Tessellate a 2-D DEM into two triangles per cell.

    The DEM is split at the cell *centre* elevations.  Adjacent cell centres
    form the triangle vertices.  Each grid quad (r, c) → (r+1, c+1) gives two
    triangles.

    Parameters
    ----------
    terrain:
        2-D array of elevation values (metres).  NaN cells are skipped.
    geotransform:
        GDAL-style 6-tuple ``(origin_x, px_w, rot_x, origin_y, rot_y, px_h)``.

    Returns
    -------
    list[Triangle]
        Triangles covering the valid DEM cells.
    """
    ox, px_w, _, oy, _, px_h = geotransform
    rows, cols = terrain.shape
    triangles: list[Triangle] = []

    def _xyz(r: int, c: int) -> np.ndarray:
        x = ox + (c + 0.5) * px_w
        y = oy + (r + 0.5) * px_h
        z = terrain[r, c]
        return np.array([x, y, float(z)], dtype=np.float64)

    for r in range(rows - 1):
        for c in range(cols - 1):
            p00 = _xyz(r, c)
            p01 = _xyz(r, c + 1)
            p10 = _xyz(r + 1, c)
            p11 = _xyz(r + 1, c + 1)

            # Skip quads that contain NaN.
            if any(not np.isfinite(p[2]) for p in (p00, p01, p10, p11)):
                continue

            triangles.append(Triangle(v0=p00, v1=p01, v2=p10))
            triangles.append(Triangle(v0=p01, v1=p11, v2=p10))

    return triangles
