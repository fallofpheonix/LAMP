from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import Resampling, reproject


def load_learned_path_prior(
    prior_raster: Path | str,
    ref_transform: rasterio.Affine,
    ref_crs: object,
    ref_shape: tuple[int, int],
) -> np.ndarray:
    """
    Load a model-produced path probability/logit raster and align it to DEM grid.
    Returns probability in [0,1].
    """
    with rasterio.open(prior_raster) as src:
        src_arr = src.read(1).astype(np.float32)
        dst = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
        nodata = src.nodata
        if nodata is not None:
            dst = np.where(np.isclose(dst, nodata), np.nan, dst)

    finite = np.isfinite(dst)
    if not finite.any():
        return np.zeros(ref_shape, dtype=np.float32)

    # Accept either probability [0,1] or logits/score ranges.
    lo = float(np.nanmin(dst[finite]))
    hi = float(np.nanmax(dst[finite]))
    if lo < 0.0 or hi > 1.0:
        clipped = np.clip(dst, -20.0, 20.0)
        prob = 1.0 / (1.0 + np.exp(-clipped))
    else:
        prob = np.clip(dst, 0.0, 1.0)

    prob[~finite] = 0.0
    return prob.astype(np.float32)
