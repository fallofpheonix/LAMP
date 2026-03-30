from __future__ import annotations

import argparse
from pathlib import Path

from lamp.services.dataset_validation_service import (
    find_crs_mismatches,
    render_dataset_markdown,
    validate_raster_layer,
    validate_vector_layer,
)


def run(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate dataset integrity for LAMP assets")
    parser.add_argument("--dem", default="task1-path-tracing/Task_1/DEM_Subset-Original.tif")
    parser.add_argument("--sar", default="task1-path-tracing/Task_1/SAR-MS.tif")
    parser.add_argument("--marks", default="task1-path-tracing/Task_1/Marks_Brief1.shp")
    parser.add_argument("--buildings", default="task1-path-tracing/Task_1/BuildingFootprints.shp")
    parser.add_argument("--out-report", default="DATASET_INTEGRITY_REPORT.md")
    args = parser.parse_args(argv)

    rasters = [validate_raster_layer(Path(args.dem)), validate_raster_layer(Path(args.sar))]
    vectors = [validate_vector_layer(Path(args.marks)), validate_vector_layer(Path(args.buildings))]

    crs_mismatches = find_crs_mismatches(rasters[0].crs, rasters[1:], vectors)
    report = render_dataset_markdown(rasters, vectors, crs_mismatches)
    Path(args.out_report).write_text(report, encoding="utf-8")
    print(f"Report written to {args.out_report}")
    return 0
