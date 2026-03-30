"""Unit tests for LOS ray casting (2D visibility module)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raycast import compute_viewshed
from src.visibility import compute_multi_observer_visibility


def _flat_surface(rows: int = 10, cols: int = 10, elev: float = 100.0) -> np.ndarray:
    return np.full((rows, cols), elev, dtype=np.float64)


def _wall_surface(rows: int = 20, cols: int = 20, wall_col: int = 10, wall_height: float = 20.0) -> np.ndarray:
    """Flat surface with a tall wall at *wall_col*."""
    surf = np.full((rows, cols), 100.0, dtype=np.float64)
    surf[:, wall_col] = 100.0 + wall_height
    return surf


class TestComputeViewshed:
    def test_flat_terrain_fully_visible(self) -> None:
        """On a perfectly flat surface every cell should be visible."""
        surf = _flat_surface(10, 10)
        vis, stats = compute_viewshed(surf, nodata=None, obs_row=5, obs_col=5)
        assert stats["visible_cells"] > 0
        assert float(stats["visible_ratio"]) == pytest.approx(1.0, abs=0.01)

    def test_observer_out_of_bounds_raises(self) -> None:
        surf = _flat_surface(5, 5)
        with pytest.raises(ValueError):
            compute_viewshed(surf, nodata=None, obs_row=10, obs_col=10)

    def test_wall_blocks_far_side(self) -> None:
        """Cells behind a tall wall should be invisible."""
        surf = _wall_surface(rows=20, cols=30, wall_col=10, wall_height=15.0)
        vis, stats = compute_viewshed(
            surf,
            nodata=None,
            obs_row=10,
            obs_col=2,
            pixel_size_m=1.0,
        )
        # At least some cells beyond wall_col should be invisible.
        beyond = vis[:, 12:]
        assert int(beyond.sum()) < int(beyond.size)

    def test_output_is_binary(self) -> None:
        surf = _flat_surface(8, 8)
        vis, _ = compute_viewshed(surf, nodata=None, obs_row=4, obs_col=4)
        unique_vals = set(np.unique(vis).tolist())
        assert unique_vals <= {0, 1}

    def test_stats_keys_present(self) -> None:
        surf = _flat_surface(6, 6)
        _, stats = compute_viewshed(surf, nodata=None, obs_row=3, obs_col=3)
        for key in ("checked_cells", "visible_cells", "visible_ratio", "observer_elevation_m"):
            assert key in stats

    def test_observer_cell_is_visible(self) -> None:
        surf = _flat_surface(7, 7)
        vis, _ = compute_viewshed(surf, nodata=None, obs_row=3, obs_col=3)
        assert vis[3, 3] == 1

    def test_nodata_cells_excluded(self) -> None:
        surf = _flat_surface(5, 5)
        surf[0, 0] = -9999.0
        vis, stats = compute_viewshed(surf, nodata=-9999.0, obs_row=2, obs_col=2)
        # nodata cell should not be counted as visible.
        assert vis[0, 0] == 0

    def test_max_distance_limits_reach(self) -> None:
        surf = _flat_surface(20, 20)
        vis_full, stats_full = compute_viewshed(surf, nodata=None, obs_row=10, obs_col=10)
        vis_ltd, stats_ltd = compute_viewshed(
            surf, nodata=None, obs_row=10, obs_col=10, max_distance_m=3.0, pixel_size_m=1.0
        )
        assert stats_ltd["checked_cells"] <= stats_full["checked_cells"]


class TestMultiObserverVisibility:
    def test_returns_probability_in_range(self) -> None:
        surf = _flat_surface(10, 10)
        observers = [
            {"row": 2, "col": 2},
            {"row": 7, "col": 7},
        ]
        result = compute_multi_observer_visibility(
            surface=surf,
            nodata=None,
            observers_rc=observers,
            observer_height=1.6,
            target_height=0.0,
            pixel_size_m=1.0,
            max_distance_m=None,
        )
        prob = result["viewshed_probability"]
        assert float(prob.min()) >= 0.0
        assert float(prob.max()) <= 1.0

    def test_any_visible_is_union(self) -> None:
        surf = _flat_surface(8, 8)
        observers = [{"row": 1, "col": 1}, {"row": 6, "col": 6}]
        result = compute_multi_observer_visibility(
            surf, None, observers, 1.6, 0.0, 1.0, None
        )
        any_vis = result["viewshed_any"]
        per = result["per_observer"]
        expected_union = np.maximum(per[0]["viewshed"], per[1]["viewshed"])
        np.testing.assert_array_equal(any_vis, expected_union)

    def test_per_observer_stats(self) -> None:
        surf = _flat_surface(6, 6)
        observers = [{"row": 3, "col": 3}]
        result = compute_multi_observer_visibility(surf, None, observers, 1.6, 0.0, 1.0, None)
        assert len(result["per_observer"]) == 1
        assert "stats" in result["per_observer"][0]
        assert "viewshed" in result["per_observer"][0]
