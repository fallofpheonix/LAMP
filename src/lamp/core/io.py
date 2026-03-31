"""GIS raster and vector I/O primitives built on rasterio and geopandas.

Provides a lightweight ``RasterBundle`` container together with
``read_raster``, ``write_raster``, and ``write_vector`` helpers that
are used throughout the LAMP pipeline to load and persist geospatial
data with consistent nodata handling and compression settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio


@dataclass
class RasterBundle:
    """Container for a raster band together with its spatial metadata."""

    data: np.ndarray
    transform: rasterio.Affine
    crs: object
    profile: dict


def read_raster(path: Path | str, band: int = 1) -> RasterBundle:
    """Read a single band from a raster file and return a :class:`RasterBundle`.

    Nodata pixels are replaced with ``np.nan``.  The returned array is
    always ``float32``.
    """
    with rasterio.open(path) as src:
        array = src.read(band).astype(np.float32)
        if src.nodata is not None:
            array = np.where(array == src.nodata, np.nan, array)
        return RasterBundle(array, src.transform, src.crs, src.profile.copy())


def write_raster(path: Path | str, array: np.ndarray, profile: dict) -> None:
    """Write *array* to a GeoTIFF at *path* using LZW compression.

    Parent directories are created automatically.  The output dtype is
    always ``float32`` and nodata is set to ``np.nan``.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
    if not out_profile.get("tiled", False):
        out_profile.pop("blockxsize", None)
        out_profile.pop("blockysize", None)
    with rasterio.open(destination, "w", **out_profile) as dst:
        dst.write(array.astype(np.float32), 1)


def write_vector(gdf: gpd.GeoDataFrame, path: Path | str) -> None:
    """Write *gdf* to a vector file, selecting the driver from the file extension.

    Supported extensions: ``.geojson`` / ``.json``, ``.gpkg``, ``.shp``.
    Parent directories are created automatically.
    """
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
