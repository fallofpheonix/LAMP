from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
import rasterio
import geopandas as gpd

@dataclass
class RasterBundle:
    data: np.ndarray
    transform: rasterio.Affine
    crs: object
    profile: dict

def read_raster(path: Path | str, band: int = 1) -> RasterBundle:
    with rasterio.open(path) as src:
        arr = src.read(band).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        profile = src.profile.copy()
        return RasterBundle(data=arr, transform=src.transform, crs=src.crs, profile=profile)

def write_raster(path: Path | str, arr: np.ndarray, profile: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out_profile = profile.copy()
    out_profile.update(dtype="float32", count=1, compress="lzw", nodata=np.nan)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(arr.astype(np.float32), 1)

def write_vector(gdf: gpd.GeoDataFrame, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix in (".geojson", ".json"):
        gdf.to_file(path, driver="GeoJSON")
    elif suffix == ".gpkg":
        gdf.to_file(path, driver="GPKG")
    elif suffix == ".shp":
        gdf.to_file(path, driver="ESRI Shapefile")
    else:
        raise ValueError(f"Unsupported vector format: {path}")
