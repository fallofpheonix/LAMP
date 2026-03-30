from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def gradient_magnitude(surface: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(surface)
    return np.sqrt(gx * gx + gy * gy)


def observer_feature_matrix(
    surface: np.ndarray,
    obs_row: int,
    obs_col: int,
    observer_height: float,
    target_height: float,
    nodata: Optional[float],
    max_distance_m: Optional[float],
    pixel_size_m: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build per-cell features for one observer.

    Returns:
      X: shape (N, F)
      valid_mask: shape (H, W) boolean mask of samples included in X
    """
    h, w = surface.shape
    rows, cols = np.indices((h, w), dtype=np.float64)

    drow = rows - float(obs_row)
    dcol = cols - float(obs_col)
    dist_cells = np.sqrt(drow * drow + dcol * dcol)
    dist_m = dist_cells * pixel_size_m

    valid = np.isfinite(surface)
    if nodata is not None:
        valid &= surface != nodata
    if max_distance_m is not None:
        valid &= dist_m <= max_distance_m

    obs_z = float(surface[obs_row, obs_col] + observer_height)
    tgt_z = surface + float(target_height)
    dz = tgt_z - obs_z

    slope_mag = gradient_magnitude(surface)

    # Avoid division by zero for the observer cell.
    dist_safe = np.maximum(dist_m, 1e-6)
    elev_angle = dz / dist_safe

    # Normalize by raster dimensions for better conditioning.
    h_norm = max(h - 1, 1)
    w_norm = max(w - 1, 1)

    inv_dist = 1.0 / dist_safe
    dist_sq = dist_m * dist_m
    abs_dz = np.abs(dz)
    slope_dz = slope_mag * dz

    feats = [
        rows / h_norm,
        cols / w_norm,
        drow / h_norm,
        dcol / w_norm,
        dist_m,
        dist_sq,
        inv_dist,
        dz,
        abs_dz,
        elev_angle,
        slope_mag,
        slope_dz,
        tgt_z,
        np.full_like(rows, obs_z),
    ]

    X_all = np.stack(feats, axis=-1).reshape(-1, len(feats))
    valid_flat = valid.reshape(-1)
    X = X_all[valid_flat]

    return X, valid


def labels_to_flat(labels: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    return labels.reshape(-1)[valid_mask.reshape(-1)].astype(np.float64)


def flat_to_raster(values: np.ndarray, valid_mask: np.ndarray, fill_value: float = 0.0) -> np.ndarray:
    out = np.full(valid_mask.shape, fill_value, dtype=np.float32)
    out.reshape(-1)[valid_mask.reshape(-1)] = values.astype(np.float32)
    return out
