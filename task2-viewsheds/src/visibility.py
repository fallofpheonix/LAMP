from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Tuple

import numpy as np

from .raycast import compute_viewshed


def _compute_one_observer(
    args: Tuple[np.ndarray, Optional[float], dict, float, float, float, Optional[float]]
) -> Tuple[dict, dict, np.ndarray]:
    """Worker helper: compute viewshed for a single observer.

    Packaged as a tuple so it can be submitted to ``ProcessPoolExecutor``.
    """
    surface, nodata, obs, observer_height, target_height, pixel_size_m, max_distance_m = args
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
    return obs, stats, arr


def compute_multi_observer_visibility(
    surface: np.ndarray,
    nodata: Optional[float],
    observers_rc: List[dict],
    observer_height: float,
    target_height: float,
    pixel_size_m: float,
    max_distance_m: Optional[float],
    n_workers: int | None = None,
) -> dict:
    """Compute per-observer binary viewsheds and aggregate probability layers.

    Parameters
    ----------
    surface:
        2-D occlusion surface in metres.
    nodata:
        Nodata value to exclude from checks (``None`` → no exclusion).
    observers_rc:
        List of ``{"row": int, "col": int, ...}`` dicts.
    observer_height:
        Observer eye height above the surface (metres).
    target_height:
        Target height above the surface (metres).
    pixel_size_m:
        Ground-sample distance (metres per pixel).
    max_distance_m:
        Maximum LOS range (``None`` → unlimited).
    n_workers:
        Parallel worker processes.  ``None`` → auto (up to 4 for ≥ 2 observers).

    Returns
    -------
    dict with keys ``per_observer``, ``viewshed_probability``, ``viewshed_any``.
    """
    if not observers_rc:
        h, w = surface.shape
        return {
            "per_observer": [],
            "viewshed_probability": np.zeros((h, w), dtype=np.float32),
            "viewshed_any": np.zeros((h, w), dtype=np.uint8),
        }

    # Decide whether to parallelise.
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 4) if len(observers_rc) >= 2 else 1

    task_args = [
        (surface, nodata, obs, observer_height, target_height, pixel_size_m, max_distance_m)
        for obs in observers_rc
    ]

    if n_workers <= 1:
        results = [_compute_one_observer(a) for a in task_args]
    else:
        results = [None] * len(task_args)  # type: ignore[list-item]
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_idx = {
                executor.submit(_compute_one_observer, a): i
                for i, a in enumerate(task_args)
            }
            for fut in as_completed(future_to_idx):
                idx = future_to_idx[fut]
                results[idx] = fut.result()

    per_observer = []
    arrays = []
    for obs, stats, arr in results:
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

