from __future__ import annotations

import numpy as np


def build_occlusion_surface(
    dem_original: np.ndarray,
    dem_with_buildings: np.ndarray,
    building_heights: np.ndarray,
    mode: str = "fused",
) -> np.ndarray:
    """
    Build occlusion surface for visibility checks.

    mode:
      - provided: use DEM with buildings as-is
      - synthetic: original DEM + rasterized footprint heights
      - fused: max(provided, synthetic)
    """
    if dem_original.shape != dem_with_buildings.shape or dem_original.shape != building_heights.shape:
        raise ValueError("Input rasters must be aligned and have identical dimensions")

    synthetic = dem_original + np.maximum(building_heights, 0.0)

    if mode == "provided":
        return dem_with_buildings.copy()
    if mode == "synthetic":
        return synthetic
    if mode == "fused":
        return np.maximum(dem_with_buildings, synthetic)

    raise ValueError(f"Unknown scene mode: {mode}")
