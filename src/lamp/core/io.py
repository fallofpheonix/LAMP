from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio


@dataclass
class RasterBundle:
    data: np.ndarray
    transform: rasterio.Affine
    crs: object
    profile: dict


def read_raster(path: Path | str, band: int = 1) -> RasterBundle:
    with rasterio.open(path) as src:
        array = src.read(band).astype(np.float32)
        if src.nodata is not None:
            array = np.where(array == src.nodata, np.nan, array)
        return RasterBundle(array, src.transform, src.crs, src.profile.copy())


def write_raster(path: Path | str, array: np.ndarray, profile: dict) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
    with rasterio.open(destination, "w", **out_profile) as dst:
        dst.write(array.astype(np.float32), 1)


def write_vector(gdf: gpd.GeoDataFrame, path: Path | str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in (".geojson", ".json"):
        gdf.to_file(destination, driver="GeoJSON")
        return
    if suffix == ".gpkg":
        gdf.to_file(destination, driver="GPKG")
        return
    if suffix == ".shp":
        gdf.to_file(destination, driver="ESRI Shapefile")
        return
    raise ValueError(f"Unsupported vector format: {destination}")


__all__ = ["RasterBundle", "read_raster", "write_raster", "write_vector"]
