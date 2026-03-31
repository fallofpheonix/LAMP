"""Terrain utility functions for raster coordinate conversions and sampling.

Provides thin wrappers around rasterio coordinate transforms and a
pure-NumPy bilinear interpolation helper, plus a slope normalisation
utility used by both the path-tracing and viewshed pipelines.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import rasterio


def world_to_pixel(transform: rasterio.Affine, x: float, y: float) -> Tuple[int, int]:
    """Convert world coordinates *(x, y)* to integer (row, col) pixel indices."""
    from rasterio.transform import rowcol

    row, col = rowcol(transform, x, y)
    return int(row), int(col)


def pixel_to_world(transform: rasterio.Affine, row: int, col: int) -> Tuple[float, float]:
    """Convert integer (row, col) pixel indices to world (x, y) centre coordinates."""
    from rasterio.transform import xy

    x, y = xy(transform, row, col, offset="center")
    return float(x), float(y)


def inside(shape: tuple[int, int], row: int, col: int) -> bool:
    """Return ``True`` if *(row, col)* lies within an array of the given *shape*."""
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def bilinear_sample(array: np.ndarray, row: float, col: float) -> float:
    """Sample *array* at a sub-pixel position using bilinear interpolation.

    Returns ``nan`` for positions outside the array bounds.
    """
    height, width = array.shape
    if row < 0 or col < 0 or row > height - 1 or col > width - 1:
        return float("nan")

    row0 = int(np.floor(row))
    col0 = int(np.floor(col))
    row1 = min(row0 + 1, height - 1)
    col1 = min(col0 + 1, width - 1)

    delta_row = row - row0
    delta_col = col - col0

    v00 = array[row0, col0]
    v01 = array[row0, col1]
    v10 = array[row1, col0]
    v11 = array[row1, col1]

    return (
        v00 * (1 - delta_row) * (1 - delta_col)
        + v01 * (1 - delta_row) * delta_col
        + v10 * delta_row * (1 - delta_col)
        + v11 * delta_row * delta_col
    )


def compute_slope_norm(dem: np.ndarray, transform: rasterio.Affine) -> np.ndarray:
    """Compute slope from *dem* and normalise to the [0, 1] range.

    The slope is derived as the arctangent of the gradient magnitude
    (in degrees / 90°).  NoData cells in the input are preserved as
    ``nan`` in the output.
    """
    x_resolution = abs(float(transform.a)) or 1.0
    y_resolution = abs(float(transform.e)) or 1.0

    filled = np.nan_to_num(dem, nan=np.nanmedian(dem))
    grad_y, grad_x = np.gradient(filled, y_resolution, x_resolution)
    slope_rad = np.arctan(np.sqrt(grad_x * grad_x + grad_y * grad_y))
    slope_deg = np.degrees(slope_rad)
    slope_norm = np.clip(slope_deg / 90.0, 0.0, 1.0)
    slope_norm[np.isnan(dem)] = np.nan
    return slope_norm.astype(np.float32)


__all__ = ["bilinear_sample", "compute_slope_norm", "inside", "pixel_to_world", "world_to_pixel"]
