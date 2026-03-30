from __future__ import annotations

import numpy as np


def compute_cost_surface(
    slope_norm: np.ndarray,
    roughness: np.ndarray,
    surface_penalty: np.ndarray,
    path_prior: np.ndarray,
    obstacle_mask: np.ndarray | None = None,
    weights: tuple[float, float, float, float] = (0.50, 0.25, 0.15, 0.10),
) -> np.ndarray:
    """
    Lower is easier movement cost.
    path_prior is inverted into a penalty so existing/likely paths are preferred.
    """
    s = np.nan_to_num(slope_norm, nan=1.0)
    r = np.nan_to_num(roughness, nan=1.0)
    p = np.nan_to_num(surface_penalty, nan=1.0)
    prior_penalty = 1.0 - np.clip(np.nan_to_num(path_prior, nan=0.0), 0.0, 1.0)

    w = np.asarray(weights, dtype=np.float32)
    if w.shape != (4,) or not np.isfinite(w).all() or float(w.sum()) <= 0.0:
        raise ValueError("weights must be finite 4-tuple with positive sum")
    w /= float(w.sum())

    cost = w[0] * s + w[1] * r + w[2] * p + w[3] * prior_penalty
    cost = np.clip(cost, 1e-4, None)

    nodata = np.isnan(slope_norm)
    if obstacle_mask is not None:
        nodata |= obstacle_mask.astype(bool)
    cost[nodata] = np.inf
    return cost.astype(np.float32)
