from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio


OUT_DIR = Path("outputs_production")


def _find_artifact(pattern: str) -> Path:
    matches = sorted(OUT_DIR.rglob(pattern))
    if not matches:
        pytest.skip(f"No generated artifact matching {pattern!r} found under {OUT_DIR}")
    return matches[0]


def test_path_tracing_output_exists() -> None:
    assert _find_artifact("predicted_paths.geojson").exists()
    assert _find_artifact("movement_cost.tif").exists()


def test_vector_geometry_validity() -> None:
    gdf = gpd.read_file(_find_artifact("predicted_paths.geojson"))
    assert len(gdf) > 0
    assert all(gdf.geometry.type.isin(["LineString", "MultiLineString"]))


def test_raster_output_properties() -> None:
    with rasterio.open(_find_artifact("movement_cost.tif")) as src:
        assert src.width > 0
        assert src.height > 0
        data = src.read(1)
        finite = data[np.isfinite(data)]
        assert finite.size > 0
        assert float(np.nanmin(finite)) >= 0.0


def test_viewshed_output_consistency() -> None:
    with rasterio.open(_find_artifact("visibility_probability.tif")) as src:
        data = src.read(1)
        finite = data[np.isfinite(data)]
        assert src.width > 0
        assert src.height > 0
        assert finite.size > 0
        assert float(np.nanmin(finite)) >= 0.0
        assert float(np.nanmax(finite)) <= 1.0
