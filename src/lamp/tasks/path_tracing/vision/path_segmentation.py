from __future__ import annotations

import numpy as np
from scipy import ndimage

from lamp.tasks.path_tracing.preprocessing.terrain_features import robust_normalize


def detect_visible_path_prior(sar: np.ndarray, slope_norm: np.ndarray) -> np.ndarray:
    """
    Deterministic proxy for visible path likelihood from SAR + local terrain.
    Returns [0,1] where higher means more likely visible path.
    """
    sar_n = robust_normalize(sar)
    filled = np.nan_to_num(sar_n, nan=np.nanmedian(sar_n))

    local_mean = ndimage.uniform_filter(filled, size=5)
    local_var = ndimage.uniform_filter((filled - local_mean) ** 2, size=5)
    texture = robust_normalize(local_var)

    gy, gx = np.gradient(filled)
    grad = np.sqrt(gx * gx + gy * gy)
    grad_n = robust_normalize(grad)

    # Paths are typically smoother/less textured and avoid steepest local cells.
    prior = 1.0 - (0.5 * texture + 0.3 * grad_n + 0.2 * np.nan_to_num(slope_norm, nan=1.0))
    prior = np.clip(prior, 0.0, 1.0).astype(np.float32)
    prior[np.isnan(slope_norm)] = 0.0
    return prior
