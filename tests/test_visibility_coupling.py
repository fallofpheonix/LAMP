from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from lamp.tasks.path_tracing.simulation.calibration import default_weight_grid
from lamp.tasks.path_tracing.simulation.cost_surface import compute_cost_surface
from scripts.run_path_tracing import load_visibility_probability


def test_cost_surface_applies_visibility_term_and_obstacle_precedence() -> None:
    slope = np.array([[0.2, 0.4], [0.6, np.nan]], dtype=np.float32)
    roughness = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
    surface = np.array([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32)
    path_prior = np.array([[0.8, 0.25], [1.4, -0.5]], dtype=np.float32)
    visibility = np.array([[0.9, 0.1], [2.0, -1.0]], dtype=np.float32)
    obstacle_mask = np.array([[False, False], [True, False]])
    weights = (0.4, 0.2, 0.1, 0.2, 0.1)

    cost = compute_cost_surface(
        slope,
        roughness,
        surface,
        path_prior,
        visibility_probability=visibility,
        obstacle_mask=obstacle_mask,
        weights=weights,
    )

    expected_00 = 0.4 * 0.2 + 0.2 * 0.1 + 0.1 * 0.3 + 0.2 * (1.0 - 0.8) + 0.1 * (1.0 - 0.9)
    expected_01 = 0.4 * 0.4 + 0.2 * 0.2 + 0.1 * 0.4 + 0.2 * (1.0 - 0.25) + 0.1 * (1.0 - 0.1)

    assert np.isclose(cost[0, 0], expected_00)
    assert np.isclose(cost[0, 1], expected_01)
    assert np.isinf(cost[1, 0])
    assert np.isinf(cost[1, 1])


def test_default_weight_grid_varies_visibility_weight_only() -> None:
    base = (0.55, 0.30, 0.10, 0.05, 0.0)
    grid = default_weight_grid(base, enable_visibility_search=True)

    assert any(np.isclose(candidate[4], 0.0) for candidate in grid)
    assert any(np.isclose(candidate[4], 0.5) for candidate in grid)

    base_core = np.asarray(base[:4], dtype=np.float32)
    base_core /= float(base_core.sum())

    for candidate in grid:
        assert np.isclose(sum(candidate), 1.0)
        visibility_weight = candidate[4]
        core = np.asarray(candidate[:4], dtype=np.float32)
        if visibility_weight < 1.0:
            renormalized_core = core / max(float(core.sum()), 1e-8)
            assert np.allclose(renormalized_core, base_core)


def test_load_visibility_probability_resamples_to_reference_grid(tmp_path: Path) -> None:
    src_path = tmp_path / "visibility.tif"
    data = np.array([[0.2, 0.7], [1.2, -0.4]], dtype=np.float32)
    transform = from_origin(100.0, 200.0, 2.0, 2.0)

    with rasterio.open(
        src_path,
        "w",
        driver="GTiff",
        height=2,
        width=2,
        count=1,
        dtype="float32",
        crs="EPSG:32636",
        transform=transform,
        nodata=-9999.0,
    ) as dst:
        dst.write(data, 1)

    aligned, meta = load_visibility_probability(
        src_path,
        {
            "height": 4,
            "width": 4,
            "transform": from_origin(100.0, 200.0, 1.0, 1.0),
            "crs": rasterio.crs.CRS.from_epsg(32636),
        },
    )

    assert aligned is not None
    assert meta is not None
    assert aligned.shape == (4, 4)
    assert meta["resampled"] is True
    assert float(np.nanmin(aligned)) >= 0.0
    assert float(np.nanmax(aligned)) <= 1.0
