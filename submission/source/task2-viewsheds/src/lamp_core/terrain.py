from __future__ import annotations

from typing import Tuple

import numpy as np
import rasterio


def world_to_pixel(transform: rasterio.Affine, x: float, y: float) -> Tuple[int, int]:
    from rasterio.transform import rowcol

    r, c = rowcol(transform, x, y)
    return int(r), int(c)


def pixel_to_world(transform: rasterio.Affine, row: int, col: int) -> Tuple[float, float]:
    from rasterio.transform import xy

    x, y = xy(transform, row, col, offset="center")
    return float(x), float(y)


def inside(shape: tuple[int, int], row: int, col: int) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def bilinear_sample(array: np.ndarray, row: float, col: float) -> float:
    h, w = array.shape
    if row < 0 or col < 0 or row > h - 1 or col > w - 1:
        return float("nan")

    r0 = int(np.floor(row))
    c0 = int(np.floor(col))
    r1 = min(r0 + 1, h - 1)
    c1 = min(c0 + 1, w - 1)

    dr = row - r0
    dc = col - c0

    v00 = array[r0, c0]
    v01 = array[r0, c1]
    v10 = array[r1, c0]
    v11 = array[r1, c1]

    return (
        v00 * (1 - dr) * (1 - dc)
        + v01 * (1 - dr) * dc
        + v10 * dr * (1 - dc)
        + v11 * dr * dc
    )


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
