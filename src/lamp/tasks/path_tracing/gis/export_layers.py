from __future__ import annotations
from pathlib import Path
import numpy as np
from lamp.core.io import write_raster, write_vector

def write_raster_float32(path: Path, arr: np.ndarray, profile: dict) -> None:
    write_raster(path, arr, profile)
