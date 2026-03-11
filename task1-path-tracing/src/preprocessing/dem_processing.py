from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

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
        arr = src.read(band).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        profile = src.profile.copy()
        return RasterBundle(data=arr, transform=src.transform, crs=src.crs, profile=profile)


def compute_slope_norm(dem: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    xres = abs(float(transform.a)) or 1.0
    yres = abs(float(transform.e)) or 1.0

    valid = np.nan_to_num(dem, nan=np.nanmedian(dem))
    gy, gx = np.gradient(valid, yres, xres)
    slope_rad = np.arctan(np.sqrt(gx * gx + gy * gy))
    slope_deg = np.degrees(slope_rad)
    slope_norm = np.clip(slope_deg / 90.0, 0.0, 1.0)
    slope_norm[np.isnan(dem)] = np.nan
    return slope_norm.astype(np.float32)
