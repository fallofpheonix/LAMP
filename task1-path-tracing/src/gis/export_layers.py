from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio


def write_vector(gdf: gpd.GeoDataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".geojson" or suffix == ".json":
        gdf.to_file(path, driver="GeoJSON")
    elif suffix == ".gpkg":
        gdf.to_file(path, driver="GPKG")
    elif suffix == ".shp":
        gdf.to_file(path, driver="ESRI Shapefile")
    else:
        raise ValueError(f"Unsupported vector format: {path}")


def write_raster_float32(path: Path, arr: np.ndarray, profile: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(np.float32), 1)
