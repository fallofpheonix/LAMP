from __future__ import annotations

from typing import Tuple

import numpy as np
import rasterio


def world_to_pixel(transform: rasterio.Affine, x: float, y: float) -> Tuple[int, int]:
    from rasterio.transform import rowcol

    row, col = rowcol(transform, x, y)
    return int(row), int(col)


def pixel_to_world(transform: rasterio.Affine, row: int, col: int) -> Tuple[float, float]:
    from rasterio.transform import xy

    x, y = xy(transform, row, col, offset="center")
    return float(x), float(y)


def inside(shape: tuple[int, int], row: int, col: int) -> bool:
    return 0 <= row < shape[0] and 0 <= col < shape[1]


def bilinear_sample(array: np.ndarray, row: float, col: float) -> float:
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
