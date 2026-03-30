from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VectorValidation:
    path: Path
    crs: str
    total_features: int
    invalid_geometries: int
    empty_geometries: int
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class RasterValidation:
    path: Path
    crs: str
    resolution: tuple[float, float]
    shape: tuple[int, int]
    nodata_value: float | int | None
    nodata_percentage: float
    bounds: tuple[float, float, float, float]


@dataclass(frozen=True)
class SecurityRisk:
    path: Path
    pattern: str
    line_number: int


@dataclass(frozen=True)
class BenchmarkResult:
    mesh_setup_seconds: float
    mesh_los_seconds: float
    mesh_aperture_seconds: float
    voxel_viewshed_seconds: float
