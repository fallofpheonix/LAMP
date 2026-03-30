from __future__ import annotations

import numpy as np
import pytest

from lamp.tasks.viewsheds.raycast import compute_viewshed
from lamp.tasks.viewsheds.visibility import compute_multi_observer_visibility


def _flat_surface(rows: int = 10, cols: int = 10, elevation: float = 100.0) -> np.ndarray:
    return np.full((rows, cols), elevation, dtype=np.float64)


def _wall_surface(rows: int = 20, cols: int = 20, wall_col: int = 10, wall_height: float = 20.0) -> np.ndarray:
    surface = np.full((rows, cols), 100.0, dtype=np.float64)
    surface[:, wall_col] = 100.0 + wall_height
    return surface


class TestComputeViewshed:
    def test_flat_terrain_fully_visible(self) -> None:
        surface = _flat_surface(10, 10)
        _, stats = compute_viewshed(surface, nodata=None, obs_row=5, obs_col=5)
        assert stats["visible_cells"] > 0
        assert float(stats["visible_ratio"]) == pytest.approx(1.0, abs=0.01)

    def test_observer_out_of_bounds_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_viewshed(_flat_surface(5, 5), nodata=None, obs_row=10, obs_col=10)

    def test_wall_blocks_far_side(self) -> None:
        surface = _wall_surface(rows=20, cols=30, wall_col=10, wall_height=15.0)
        viewshed, _ = compute_viewshed(surface, nodata=None, obs_row=10, obs_col=2, pixel_size_m=1.0)
        beyond = viewshed[:, 12:]
        assert int(beyond.sum()) < int(beyond.size)

    def test_output_is_binary(self) -> None:
        viewshed, _ = compute_viewshed(_flat_surface(8, 8), nodata=None, obs_row=4, obs_col=4)
        assert set(np.unique(viewshed).tolist()) <= {0, 1}

    def test_stats_keys_present(self) -> None:
        _, stats = compute_viewshed(_flat_surface(6, 6), nodata=None, obs_row=3, obs_col=3)
        for key in ("checked_cells", "visible_cells", "visible_ratio", "observer_elevation_m"):
            assert key in stats

    def test_observer_cell_is_visible(self) -> None:
        viewshed, _ = compute_viewshed(_flat_surface(7, 7), nodata=None, obs_row=3, obs_col=3)
        assert viewshed[3, 3] == 1

    def test_nodata_cells_excluded(self) -> None:
        surface = _flat_surface(5, 5)
        surface[0, 0] = -9999.0
        viewshed, _ = compute_viewshed(surface, nodata=-9999.0, obs_row=2, obs_col=2)
        assert viewshed[0, 0] == 0

    def test_max_distance_limits_reach(self) -> None:
        surface = _flat_surface(20, 20)
        _, stats_full = compute_viewshed(surface, nodata=None, obs_row=10, obs_col=10)
        _, stats_limited = compute_viewshed(surface, nodata=None, obs_row=10, obs_col=10, max_distance_m=3.0, pixel_size_m=1.0)
        assert stats_limited["checked_cells"] <= stats_full["checked_cells"]


class TestMultiObserverVisibility:
    def test_returns_probability_in_range(self) -> None:
        result = compute_multi_observer_visibility(
            surface=_flat_surface(10, 10),
            nodata=None,
            observers_rc=[{"row": 2, "col": 2}, {"row": 7, "col": 7}],
            observer_height=1.6,
            target_height=0.0,
            pixel_size_m=1.0,
            max_distance_m=None,
        )
        probability = result["viewshed_probability"]
        assert float(probability.min()) >= 0.0
        assert float(probability.max()) <= 1.0

    def test_any_visible_is_union(self) -> None:
        result = compute_multi_observer_visibility(
            _flat_surface(8, 8),
            None,
            [{"row": 1, "col": 1}, {"row": 6, "col": 6}],
            1.6,
            0.0,
            1.0,
            None,
        )
        expected_union = np.maximum(result["per_observer"][0]["viewshed"], result["per_observer"][1]["viewshed"])
        np.testing.assert_array_equal(result["viewshed_any"], expected_union)

    def test_per_observer_stats(self) -> None:
        result = compute_multi_observer_visibility(_flat_surface(6, 6), None, [{"row": 3, "col": 3}], 1.6, 0.0, 1.0, None)
        assert len(result["per_observer"]) == 1
        assert "stats" in result["per_observer"][0]
        assert "viewshed" in result["per_observer"][0]
