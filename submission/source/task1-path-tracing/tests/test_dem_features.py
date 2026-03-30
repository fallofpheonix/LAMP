"""Unit tests for DEM feature extraction (slope, roughness, surface penalty)."""
from __future__ import annotations

import numpy as np
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import rasterio
from rasterio.transform import from_bounds

from preprocessing.dem_processing import compute_slope_norm, read_raster
from preprocessing.terrain_features import compute_roughness, derive_surface_penalty, robust_normalize


def _flat_transform(rows: int = 10, cols: int = 10, res: float = 1.0) -> rasterio.Affine:
    return from_bounds(0.0, 0.0, cols * res, rows * res, cols, rows)


class TestRobustNormalize:
    def test_uniform_array_returns_zeros(self) -> None:
        arr = np.full((5, 5), 7.0, dtype=np.float32)
        out = robust_normalize(arr)
        assert np.all(out[np.isfinite(out)] == 0.0)

    def test_range_clipped_to_0_1(self) -> None:
        rng = np.random.default_rng(0)
        arr = rng.uniform(-100.0, 200.0, size=(20, 20)).astype(np.float32)
        out = robust_normalize(arr)
        assert float(out[np.isfinite(out)].min()) >= 0.0
        assert float(out[np.isfinite(out)].max()) <= 1.0

    def test_nan_propagated(self) -> None:
        arr = np.array([[1.0, 2.0, np.nan], [3.0, 4.0, 5.0]], dtype=np.float32)
        out = robust_normalize(arr)
        assert np.isnan(out[0, 2])

    def test_all_nan_returns_zeros(self) -> None:
        arr = np.full((4, 4), np.nan, dtype=np.float32)
        out = robust_normalize(arr)
        assert np.all(out == 0.0)


class TestComputeSlopeNorm:
    def test_flat_dem_gives_zero_slope(self) -> None:
        dem = np.ones((10, 10), dtype=np.float32)
        transform = _flat_transform()
        slope = compute_slope_norm(dem, transform)
        assert float(np.nanmax(slope)) == pytest.approx(0.0, abs=1e-4)

    def test_slope_in_valid_range(self) -> None:
        rng = np.random.default_rng(42)
        dem = rng.uniform(0.0, 50.0, (20, 20)).astype(np.float32)
        transform = _flat_transform(20, 20, res=1.0)
        slope = compute_slope_norm(dem, transform)
        finite = slope[np.isfinite(slope)]
        assert float(finite.min()) >= 0.0
        assert float(finite.max()) <= 1.0 + 1e-5

    def test_inclined_plane_gives_consistent_slope(self) -> None:
        rows, cols = 10, 10
        c_idx = np.arange(cols, dtype=np.float32)
        dem = np.tile(c_idx * 1.0, (rows, 1))  # 1 m/m slope along x
        transform = _flat_transform(rows, cols, res=1.0)
        slope = compute_slope_norm(dem, transform)
        # Interior cells should have a consistently positive slope.
        assert float(np.nanmean(slope[1:-1, 1:-1])) > 0.0

    def test_nan_dem_cells_produce_nan_slope(self) -> None:
        dem = np.ones((5, 5), dtype=np.float32)
        dem[2, 2] = np.nan
        transform = _flat_transform(5, 5)
        slope = compute_slope_norm(dem, transform)
        assert np.isnan(slope[2, 2])


class TestComputeRoughness:
    def test_flat_dem_gives_low_roughness(self) -> None:
        dem = np.ones((10, 10), dtype=np.float32)
        rough = compute_roughness(dem, sigma=1.0)
        # Uniform DEM → roughness ≈ 0 everywhere (before normalization may be all-zero).
        finite = rough[np.isfinite(rough)]
        assert float(finite.max()) < 0.1

    def test_rough_dem_gives_non_trivial_values(self) -> None:
        rng = np.random.default_rng(7)
        dem = (rng.random((20, 20)) * 100).astype(np.float32)
        rough = compute_roughness(dem)
        finite = rough[np.isfinite(rough)]
        assert float(finite.max()) > 0.0
        assert float(finite.min()) >= 0.0

    def test_output_shape_matches_input(self) -> None:
        dem = np.ones((15, 25), dtype=np.float32)
        rough = compute_roughness(dem)
        assert rough.shape == dem.shape


class TestDeriveSurfacePenalty:
    def test_output_in_valid_range(self) -> None:
        rng = np.random.default_rng(1)
        sar = rng.uniform(0.1, 1.0, (10, 10)).astype(np.float32)
        slope = rng.uniform(0.0, 0.5, (10, 10)).astype(np.float32)
        penalty = derive_surface_penalty(sar, slope)
        finite = penalty[np.isfinite(penalty)]
        assert float(finite.min()) >= 0.0
        assert float(finite.max()) <= 1.0

    def test_output_shape_matches_input(self) -> None:
        sar = np.ones((8, 12), dtype=np.float32)
        slope = np.zeros((8, 12), dtype=np.float32)
        penalty = derive_surface_penalty(sar, slope)
        assert penalty.shape == (8, 12)
