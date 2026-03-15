"""Unit tests for mesh-based BVH ray tracing (mesh_raycast module)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mesh_raycast import (
    BVHNode,
    MeshScene,
    Triangle,
    build_bvh,
    build_mesh_scene,
    dem_to_mesh,
    mesh_is_visible,
    moller_trumbore,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unit_tri() -> Triangle:
    """Unit triangle in the XY plane at z = 0."""
    return Triangle(
        v0=np.array([0.0, 0.0, 0.0]),
        v1=np.array([1.0, 0.0, 0.0]),
        v2=np.array([0.0, 1.0, 0.0]),
    )


def _wall_scene() -> MeshScene:
    """A vertical wall at x = 5, spanning y=0..10 and z=0..5."""
    tris = [
        Triangle(
            v0=np.array([5.0, 0.0, 0.0]),
            v1=np.array([5.0, 10.0, 0.0]),
            v2=np.array([5.0, 0.0, 5.0]),
        ),
        Triangle(
            v0=np.array([5.0, 10.0, 0.0]),
            v1=np.array([5.0, 10.0, 5.0]),
            v2=np.array([5.0, 0.0, 5.0]),
        ),
    ]
    return build_mesh_scene(tris)


# ---------------------------------------------------------------------------
# Triangle
# ---------------------------------------------------------------------------

class TestTriangle:
    def test_aabb_min_max(self) -> None:
        tri = _unit_tri()
        np.testing.assert_allclose(tri.aabb_min(), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(tri.aabb_max(), [1.0, 1.0, 0.0])

    def test_centroid(self) -> None:
        tri = _unit_tri()
        np.testing.assert_allclose(tri.centroid(), [1 / 3, 1 / 3, 0.0], atol=1e-10)


# ---------------------------------------------------------------------------
# Möller–Trumbore
# ---------------------------------------------------------------------------

class TestMollerTrumbore:
    def test_direct_hit(self) -> None:
        tri = _unit_tri()
        origin = np.array([0.25, 0.25, 1.0])
        direction = np.array([0.0, 0.0, -1.0])
        hit, t = moller_trumbore(origin, direction, tri)
        assert hit
        assert t == pytest.approx(1.0, abs=1e-8)

    def test_miss_outside_triangle(self) -> None:
        tri = _unit_tri()
        origin = np.array([2.0, 2.0, 1.0])
        direction = np.array([0.0, 0.0, -1.0])
        hit, t = moller_trumbore(origin, direction, tri)
        assert not hit

    def test_parallel_ray_no_hit(self) -> None:
        tri = _unit_tri()
        origin = np.array([0.0, 0.0, 1.0])
        direction = np.array([1.0, 0.0, 0.0])  # Parallel to plane.
        hit, _ = moller_trumbore(origin, direction, tri)
        assert not hit

    def test_ray_behind_origin_no_hit(self) -> None:
        tri = _unit_tri()
        origin = np.array([0.25, 0.25, -1.0])
        direction = np.array([0.0, 0.0, -1.0])  # Away from triangle.
        hit, _ = moller_trumbore(origin, direction, tri)
        assert not hit


# ---------------------------------------------------------------------------
# BVH build
# ---------------------------------------------------------------------------

class TestBuildBVH:
    def test_single_triangle_leaf(self) -> None:
        tris = [_unit_tri()]
        node = build_bvh(tris)
        assert node.is_leaf()
        assert len(node.triangles) == 1

    def test_many_triangles_builds_tree(self) -> None:
        tris = [
            Triangle(
                v0=np.array([float(i), 0.0, 0.0]),
                v1=np.array([float(i) + 1.0, 0.0, 0.0]),
                v2=np.array([float(i) + 0.5, 1.0, 0.0]),
            )
            for i in range(20)
        ]
        root = build_bvh(tris, max_leaf_size=4)
        assert not root.is_leaf()

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            build_bvh([])

    def test_bbox_encloses_all(self) -> None:
        tris = [
            Triangle(
                v0=np.array([0.0, 0.0, 0.0]),
                v1=np.array([1.0, 0.0, 0.0]),
                v2=np.array([0.5, 1.0, 0.0]),
            ),
            Triangle(
                v0=np.array([5.0, 5.0, 0.0]),
                v1=np.array([6.0, 5.0, 0.0]),
                v2=np.array([5.5, 6.0, 0.0]),
            ),
        ]
        node = build_bvh(tris)
        assert float(node.bbox_min[0]) <= 0.0
        assert float(node.bbox_max[0]) >= 6.0


# ---------------------------------------------------------------------------
# mesh_is_visible
# ---------------------------------------------------------------------------

class TestMeshIsVisible:
    def test_clear_path_is_visible(self) -> None:
        scene = _wall_scene()
        # Both points on the same side of the wall → visible.
        assert mesh_is_visible(scene, (1.0, 5.0, 1.5), (4.0, 5.0, 1.5))

    def test_wall_blocks_los(self) -> None:
        scene = _wall_scene()
        # Segment crosses the wall.
        assert not mesh_is_visible(scene, (1.0, 5.0, 1.5), (9.0, 5.0, 1.5))

    def test_identical_points_visible(self) -> None:
        scene = _wall_scene()
        assert mesh_is_visible(scene, (2.0, 5.0, 1.0), (2.0, 5.0, 1.0))

    def test_ray_over_wall_is_visible(self) -> None:
        scene = _wall_scene()
        # Ray goes above the wall (z > 5).
        assert mesh_is_visible(scene, (1.0, 5.0, 6.0), (9.0, 5.0, 6.0))


# ---------------------------------------------------------------------------
# dem_to_mesh
# ---------------------------------------------------------------------------

class TestDemToMesh:
    def test_flat_dem_gives_triangles(self) -> None:
        dem = np.ones((5, 5), dtype=np.float64)
        gt = (0.0, 1.0, 0.0, 5.0, 0.0, -1.0)
        tris = dem_to_mesh(dem, gt)
        # 4x4 quads × 2 triangles each.
        assert len(tris) == 4 * 4 * 2

    def test_nan_cells_skipped(self) -> None:
        dem = np.ones((4, 4), dtype=np.float64)
        dem[1, 1] = np.nan
        gt = (0.0, 1.0, 0.0, 4.0, 0.0, -1.0)
        tris = dem_to_mesh(dem, gt)
        # Some quads will be skipped due to NaN.
        assert len(tris) < 3 * 3 * 2

    def test_elevations_match_dem(self) -> None:
        dem = np.zeros((3, 3), dtype=np.float64)
        dem[1, 1] = 10.0
        gt = (0.0, 1.0, 0.0, 3.0, 0.0, -1.0)
        tris = dem_to_mesh(dem, gt)
        z_vals = set()
        for t in tris:
            for v in (t.v0, t.v1, t.v2):
                z_vals.add(round(float(v[2]), 6))
        assert 10.0 in z_vals and 0.0 in z_vals
