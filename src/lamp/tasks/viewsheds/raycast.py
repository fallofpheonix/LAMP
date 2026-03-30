from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .terrain import bilinear_sample


def _build_valid_mask(surface: np.ndarray, nodata: Optional[float]) -> np.ndarray:
    valid = np.isfinite(surface)
    if nodata is not None:
        valid &= surface != nodata
    return valid


def _is_visible(
    surface: np.ndarray,
    valid: np.ndarray,
    obs_row: int,
    obs_col: int,
    tgt_row: int,
    tgt_col: int,
    obs_z: float,
    tgt_z: float,
) -> bool:
    dx = tgt_col - obs_col
    dy = tgt_row - obs_row
    steps = int(max(abs(dx), abs(dy)))
    if steps <= 1:
        return True

    for i in range(1, steps):
        t = i / steps
        r = obs_row + dy * t
        c = obs_col + dx * t

        z_occ = bilinear_sample(surface, r, c)
        if not np.isfinite(z_occ):
            return False

        z_los = obs_z + t * (tgt_z - obs_z)
        if z_occ > z_los + 1e-6:
            return False

    return True


def compute_viewshed(
    surface: np.ndarray,
    nodata: Optional[float],
    obs_row: int,
    obs_col: int,
    observer_height: float = 1.6,
    target_height: float = 0.0,
    pixel_size_m: float = 1.0,
    max_distance_m: Optional[float] = None,
) -> Tuple[np.ndarray, dict]:
    """
    Deterministic line-of-sight viewshed over a 2.5D occlusion surface.
    Uses terrain/building elevations in meters for both observer and targets.
    """
    h, w = surface.shape
    valid = _build_valid_mask(surface, nodata)

    if not (0 <= obs_row < h and 0 <= obs_col < w and valid[obs_row, obs_col]):
        raise ValueError("Observer is outside valid raster region")

    obs_z = float(surface[obs_row, obs_col] + observer_height)
    out = np.zeros((h, w), dtype=np.uint8)

    checked = 0
    visible = 0

    for r in range(h):
        for c in range(w):
            if not valid[r, c]:
                continue

            if max_distance_m is not None:
                dist = np.hypot(r - obs_row, c - obs_col) * pixel_size_m
                if dist > max_distance_m:
                    continue

            tgt_z = float(surface[r, c] + target_height)
            checked += 1

            if _is_visible(surface, valid, obs_row, obs_col, r, c, obs_z, tgt_z):
                out[r, c] = 1
                visible += 1

    ratio = (visible / checked) if checked else 0.0
    stats = {
        "checked_cells": checked,
        "visible_cells": visible,
        "visible_ratio": ratio,
        "observer_elevation_m": obs_z,
    }
    return out, stats
