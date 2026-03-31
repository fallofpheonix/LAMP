"""GIS export helpers for Task 1 path-tracing outputs.

Thin wrapper around :mod:`lamp.core.io` kept for backward compatibility
with legacy pipeline scripts that import from the task-specific namespace.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
from lamp.core.io import write_raster, write_vector

def write_raster_float32(path: Path, arr: np.ndarray, profile: dict) -> None:
    """Write a float32 raster to *path*.  Delegates to :func:`lamp.core.io.write_raster`."""
    write_raster(path, arr, profile)
