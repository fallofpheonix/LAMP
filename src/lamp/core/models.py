"""Immutable result data-classes shared across LAMP services.

These plain dataclasses carry structured results produced by the
validation, security-audit, and benchmark service layer so that
consumers can work with typed objects rather than raw dicts.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VectorValidation:
    """Validation result for a single vector (OGR/shapefile) layer."""

    path: Path
    crs: str
    total_features: int
    invalid_geometries: int
    empty_geometries: int
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class RasterValidation:
    """Validation result for a single raster (GeoTIFF) layer."""

    path: Path
    crs: str
    resolution: tuple[float, float]
    shape: tuple[int, int]
    nodata_value: float | int | None
    nodata_percentage: float
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class SecurityRisk:
    """A single potential path-traversal risk detected during a security scan."""

    path: Path
    pattern: str
    line_number: int


@dataclass(frozen=True)
class BenchmarkResult:
    """Timing measurements from a single raycasting benchmark run."""

    mesh_setup_seconds: float
    mesh_los_seconds: float
    mesh_aperture_seconds: float
    voxel_viewshed_seconds: float
