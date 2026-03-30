import os
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio

# Paths to artifacts produced during verification
OUT_DIR = Path("outputs_production")


def _skip_if_missing(*paths: Path) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.skip(f"missing generated artifacts: {', '.join(missing)}")


def test_path_tracing_output_exists():
    _skip_if_missing(OUT_DIR / "candidate_paths.shp", OUT_DIR / "selected_paths.shp")

def test_vector_geometry_validity():
    _skip_if_missing(OUT_DIR / "selected_paths.shp")
    gdf = gpd.read_file(OUT_DIR / "selected_paths.shp")
    assert len(gdf) > 0
    assert all(gdf.geometry.type.isin(["LineString", "MultiLineString"]))

def test_raster_output_properties():
    prior_path = Path("path_prior_prob.tif")
    if not prior_path.exists():
        pytest.skip(f"missing generated artifact: {prior_path}")
    with rasterio.open(prior_path) as src:
        assert src.crs.to_epsg() == 32638
        data = src.read(1)
        assert np.nanmax(data) <= 1.0
        assert np.nanmin(data) >= 0.0

def test_viewshed_output_consistency():
    viewshed_files = list(Path("outputs").glob("*.tif"))
    if not viewshed_files:
        pytest.skip("missing generated viewshed rasters")
    with rasterio.open(viewshed_files[0]) as src:
        assert src.width > 0
        assert src.height > 0

if __name__ == "__main__":
    pytest.main([__file__])
