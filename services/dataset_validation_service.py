from __future__ import annotations

from pathlib import Path

from core.exceptions import DataValidationError
from core.models import RasterValidation, VectorValidation


def validate_vector_layer(path: Path) -> VectorValidation:
    import geopandas as gpd

    frame = gpd.read_file(path)
    bounds = tuple(float(value) for value in frame.total_bounds)
    return VectorValidation(
        path=path,
        crs=str(frame.crs),
        total_features=len(frame),
        invalid_geometries=int((~frame.geometry.is_valid).sum()),
        empty_geometries=int(frame.geometry.is_empty.sum()),
        bounds=bounds,
    )


def validate_raster_layer(path: Path) -> RasterValidation:
    import numpy as np
    import rasterio

    with rasterio.open(path) as src:
        band = src.read(1)
        if band.size == 0:
            raise DataValidationError(f"Raster contains no cells: {path}")

        if src.nodata is None:
            nodata_mask = np.isnan(band)
        else:
            nodata_mask = np.isnan(band) | np.isclose(band, src.nodata)

        bounds = tuple(float(value) for value in src.bounds)
        return RasterValidation(
            path=path,
            crs=str(src.crs),
            resolution=(float(src.res[0]), float(src.res[1])),
            shape=(int(src.shape[0]), int(src.shape[1])),
            nodata_value=src.nodata,
            nodata_percentage=float(nodata_mask.sum() / band.size),
            bounds=bounds,
        )


def find_crs_mismatches(reference_crs: str, rasters: list[RasterValidation], vectors: list[VectorValidation]) -> list[Path]:
    mismatches: list[Path] = []
    for raster in rasters:
        if raster.crs != reference_crs:
            mismatches.append(raster.path)
    for vector in vectors:
        if vector.crs != reference_crs:
            mismatches.append(vector.path)
    return mismatches


def render_dataset_markdown(rasters: list[RasterValidation], vectors: list[VectorValidation], crs_mismatches: list[Path]) -> str:
    lines = ["# Dataset Integrity Report", "", "## 1. CRS Consistency"]
    if crs_mismatches:
        mismatch_list = ", ".join(path.name for path in crs_mismatches)
        lines.append(f"- **Status**: WARNING ({mismatch_list})")
    else:
        lines.append("- **Status**: PASS (All layers align to reference CRS)")

    lines.append("")
    lines.append("## 2. Vector Validation")
    for vector in vectors:
        lines.append(f"### {vector.path.name}")
        lines.append(f"- Total Features: {vector.total_features}")
        lines.append(f"- Invalid Geometries: {vector.invalid_geometries}")
        lines.append(f"- Empty Geometries: {vector.empty_geometries}")
        lines.append("")

    lines.append("## 3. Raster Validation")
    for raster in rasters:
        lines.append(f"### {raster.path.name}")
        lines.append(f"- Shape: {raster.shape}")
        lines.append(f"- Resolution: {raster.resolution}")
        lines.append(f"- NoData %: {raster.nodata_percentage:.2%}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"
