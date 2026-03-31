from __future__ import annotations

import numpy as np
import pytest

from lamp.tasks.viewsheds.mesh_raycast import Triangle, build_bvh, build_mesh_scene, dem_to_mesh, mesh_is_visible, moller_trumbore


def _unit_triangle() -> Triangle:
    return Triangle(
        v0=np.array([0.0, 0.0, 0.0]),
        v1=np.array([1.0, 0.0, 0.0]),
        v2=np.array([0.0, 1.0, 0.0]),
    )


def _wall_scene():
    triangles = [
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
    return build_mesh_scene(triangles)


class TestTriangle:
    def test_aabb_min_max(self) -> None:
        triangle = _unit_triangle()
        np.testing.assert_allclose(triangle.aabb_min(), [0.0, 0.0, 0.0])
        np.testing.assert_allclose(triangle.aabb_max(), [1.0, 1.0, 0.0])

    def test_centroid(self) -> None:
        triangle = _unit_triangle()
        np.testing.assert_allclose(triangle.centroid(), [1 / 3, 1 / 3, 0.0], atol=1e-10)


class TestMollerTrumbore:
    def test_direct_hit(self) -> None:
        triangle = _unit_triangle()
        hit, distance = moller_trumbore(np.array([0.25, 0.25, 1.0]), np.array([0.0, 0.0, -1.0]), triangle)
        assert hit
        assert distance == pytest.approx(1.0, abs=1e-8)

    def test_miss_outside_triangle(self) -> None:
        hit, _ = moller_trumbore(np.array([2.0, 2.0, 1.0]), np.array([0.0, 0.0, -1.0]), _unit_triangle())
        assert not hit

    def test_parallel_ray_no_hit(self) -> None:
        hit, _ = moller_trumbore(np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0]), _unit_triangle())
        assert not hit

    def test_ray_behind_origin_no_hit(self) -> None:
        hit, _ = moller_trumbore(np.array([0.25, 0.25, -1.0]), np.array([0.0, 0.0, -1.0]), _unit_triangle())
        assert not hit


class TestBuildBVH:
    def test_single_triangle_leaf(self) -> None:
        root = build_bvh([_unit_triangle()])
        assert root.is_leaf()
        assert len(root.triangles) == 1

    def test_many_triangles_builds_tree(self) -> None:
        triangles = [
            Triangle(
                v0=np.array([float(i), 0.0, 0.0]),
                v1=np.array([float(i) + 1.0, 0.0, 0.0]),
                v2=np.array([float(i) + 0.5, 1.0, 0.0]),
            )
            for i in range(20)
        ]
        root = build_bvh(triangles, max_leaf_size=4)
        assert not root.is_leaf()

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError):
            build_bvh([])

    def test_bbox_encloses_all(self) -> None:
        triangles = [
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
        root = build_bvh(triangles)
        assert float(root.bbox_min[0]) <= 0.0
        assert float(root.bbox_max[0]) >= 6.0


class TestMeshIsVisible:
    def test_clear_path_is_visible(self) -> None:
        visibility = mesh_is_visible(_wall_scene(), (1.0, 5.0, 1.5), (4.0, 5.0, 1.5))
        assert visibility == pytest.approx(1.0)

    def test_wall_blocks_los(self) -> None:
        visibility = mesh_is_visible(_wall_scene(), (1.0, 5.0, 1.5), (9.0, 5.0, 1.5))
        assert visibility == pytest.approx(0.0)

    def test_identical_points_visible(self) -> None:
        visibility = mesh_is_visible(_wall_scene(), (2.0, 5.0, 1.0), (2.0, 5.0, 1.0))
        assert visibility == pytest.approx(1.0)

    def test_ray_over_wall_is_visible(self) -> None:
        visibility = mesh_is_visible(_wall_scene(), (1.0, 5.0, 6.0), (9.0, 5.0, 6.0))
        assert visibility == pytest.approx(1.0)


class TestDemToMesh:
    def test_flat_dem_gives_triangles(self) -> None:
        dem = np.ones((5, 5), dtype=np.float64)
        triangles = dem_to_mesh(dem, (0.0, 1.0, 0.0, 5.0, 0.0, -1.0))
        assert len(triangles) == 4 * 4 * 2

    def test_nan_cells_skipped(self) -> None:
        dem = np.ones((4, 4), dtype=np.float64)
        dem[1, 1] = np.nan
        triangles = dem_to_mesh(dem, (0.0, 1.0, 0.0, 4.0, 0.0, -1.0))
        assert len(triangles) < 3 * 3 * 2

    def test_elevations_match_dem(self) -> None:
        dem = np.zeros((3, 3), dtype=np.float64)
        dem[1, 1] = 10.0
        triangles = dem_to_mesh(dem, (0.0, 1.0, 0.0, 3.0, 0.0, -1.0))
        z_values = set()
        for triangle in triangles:
            for vertex in (triangle.v0, triangle.v1, triangle.v2):
                z_values.add(round(float(vertex[2]), 6))
        assert 10.0 in z_values and 0.0 in z_values
