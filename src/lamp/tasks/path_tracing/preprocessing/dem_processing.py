"""DEM processing helpers re-exported for the Task 1 preprocessing stage.

Re-exports :func:`~lamp.core.io.read_raster` and
:func:`~lamp.core.terrain.compute_slope_norm` under the preprocessing
namespace so pipeline code has a single consistent import path.
"""

from __future__ import annotations
from lamp.core.io import RasterBundle, read_raster
from lamp.core.terrain import compute_slope_norm
