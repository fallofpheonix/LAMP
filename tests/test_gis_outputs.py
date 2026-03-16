import os
import pytest
import geopandas as gpd
import rasterio
import numpy as np
from pathlib import Path

# Paths to artifacts produced during verification
OUT_DIR = Path("outputs_production")

def test_path_tracing_output_exists():
    assert (OUT_DIR / "candidate_paths.shp").exists()
    assert (OUT_DIR / "selected_paths.shp").exists()

def test_vector_geometry_validity():
    gdf = gpd.read_file(OUT_DIR / "selected_paths.shp")
    assert len(gdf) > 0
    assert all(gdf.geometry.type.isin(['LineString', 'MultiLineString']))

def test_raster_output_properties():
    prior_path = Path("path_prior_prob.tif")
    if prior_path.exists():
        with rasterio.open(prior_path) as src:
            assert src.crs.to_epsg() == 32638
            data = src.read(1)
            assert np.nanmax(data) <= 1.0
            assert np.nanmin(data) >= 0.0

def test_viewshed_output_consistency():
    # Check if any viewshed probability raster exists in outputs
    viewshed_files = list(Path("outputs").glob("*.tif"))
    if viewshed_files:
        with rasterio.open(viewshed_files[0]) as src:
            assert src.width > 0
            assert src.height > 0

if __name__ == "__main__":
    pytest.main([__file__])
