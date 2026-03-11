from __future__ import annotations

from typing import Tuple

import numpy as np


def world_to_pixel(geotransform: tuple, x: float, y: float) -> Tuple[int, int]:
    """Convert world coordinates to nearest pixel indices (row, col)."""
    origin_x, px_w, rot_x, origin_y, rot_y, px_h = geotransform
    if rot_x != 0.0 or rot_y != 0.0:
        raise ValueError("Rotated geotransforms are not supported in this pipeline")

    col_f = (x - origin_x) / px_w
    row_f = (y - origin_y) / px_h
    return int(round(row_f)), int(round(col_f))


def pixel_to_world(geotransform: tuple, row: int, col: int) -> Tuple[float, float]:
    """Convert pixel indices to world coordinates at cell center."""
    origin_x, px_w, _, origin_y, _, px_h = geotransform
    x = origin_x + (col + 0.5) * px_w
    y = origin_y + (row + 0.5) * px_h
    return x, y


def inside(array: np.ndarray, row: int, col: int) -> bool:
    return 0 <= row < array.shape[0] and 0 <= col < array.shape[1]


def bilinear_sample(array: np.ndarray, row: float, col: float) -> float:
    """Bilinear sample from a 2D array using floating-point pixel indices."""
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
