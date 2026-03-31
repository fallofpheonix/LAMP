"""Terrain feature extraction for the Task 1 preprocessing stage.

Derives normalised roughness and surface-penalty rasters from a DEM
and co-registered SAR image.  All outputs are ``float32`` arrays in
the [0, 1] range with ``nan`` where the source DEM has nodata.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def robust_normalize(arr: np.ndarray, low_q: float = 2.0, high_q: float = 98.0) -> np.ndarray:
    """Clip *arr* to the [*low_q*, *high_q*] percentile range and scale to [0, 1]."""
    out = arr.astype(np.float32).copy()
    finite = np.isfinite(out)
    if not finite.any():
        return np.zeros_like(out, dtype=np.float32)
    lo, hi = np.percentile(out[finite], [low_q, high_q])
    if hi <= lo:
        out[finite] = 0.0
        return out
    out[finite] = (out[finite] - lo) / (hi - lo)
    out = np.clip(out, 0.0, 1.0)
    out[~finite] = np.nan
    return out


def compute_roughness(dem: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Return normalised surface roughness as the Gaussian-residual magnitude."""
    filled = np.nan_to_num(dem, nan=np.nanmedian(dem))
    smooth = ndimage.gaussian_filter(filled, sigma=sigma)
    rough = np.abs(filled - smooth)
    rough[np.isnan(dem)] = np.nan
    return robust_normalize(rough)


def derive_surface_penalty(sar: np.ndarray, slope_norm: np.ndarray) -> np.ndarray:
    """Derive a movement-difficulty surface penalty from *sar* and *slope_norm*.

    Combines the SAR gradient magnitude (70 %) with the slope term (30 %)
    to produce a [0, 1] penalty raster where higher values deter movement.
    """
    sar_n = robust_normalize(sar)
    gy, gx = np.gradient(np.nan_to_num(sar_n, nan=np.nanmedian(sar_n)))
    grad = np.sqrt(gx * gx + gy * gy)
    grad_n = robust_normalize(grad)

    # Higher penalty for rough/high-gradient regions and steep local terrain.
    penalty = 0.7 * grad_n + 0.3 * np.nan_to_num(slope_norm, nan=1.0)
    penalty[np.isnan(slope_norm)] = np.nan
    return np.clip(penalty, 0.0, 1.0).astype(np.float32)
