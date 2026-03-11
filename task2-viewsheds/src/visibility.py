from __future__ import annotations

from typing import List, Optional

import numpy as np

from .raycast import compute_viewshed


def compute_multi_observer_visibility(
    surface: np.ndarray,
    nodata: Optional[float],
    observers_rc: List[dict],
    observer_height: float,
    target_height: float,
    pixel_size_m: float,
    max_distance_m: Optional[float],
) -> dict:
    """Compute per-observer binary viewsheds and aggregate probability layers."""
    per_observer = []
    arrays = []

    for obs in observers_rc:
        arr, stats = compute_viewshed(
            surface=surface,
            nodata=nodata,
            obs_row=obs["row"],
            obs_col=obs["col"],
            observer_height=observer_height,
            target_height=target_height,
            pixel_size_m=pixel_size_m,
            max_distance_m=max_distance_m,
        )
        per_observer.append({"observer": obs, "stats": stats, "viewshed": arr})
        arrays.append(arr.astype(np.float32))

    stack = np.stack(arrays, axis=0)
    prob = np.mean(stack, axis=0).astype(np.float32)
    any_visible = (np.max(stack, axis=0) > 0).astype(np.uint8)

    return {
        "per_observer": per_observer,
        "viewshed_probability": prob,
        "viewshed_any": any_visible,
    }
