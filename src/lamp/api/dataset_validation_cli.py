from __future__ import annotations

import argparse
from pathlib import Path

from lamp.config import load_defaults
from lamp.services.dataset_validation_service import (
    find_crs_mismatches,
    render_dataset_markdown,
    validate_raster_layer,
    validate_vector_layer,
)


def run(argv: list[str] | None = None) -> int:
    defaults = load_defaults()
    parser = argparse.ArgumentParser(description="Validate dataset integrity for LAMP assets")
    parser.add_argument("--dem", default=str(defaults.dem_path))
    parser.add_argument("--sar", default=str(defaults.sar_path))
    parser.add_argument("--marks", default=str(defaults.marks_path))
    parser.add_argument("--buildings", default=str(defaults.buildings_path))
    parser.add_argument("--out-report", default="outputs/dataset_integrity_latest.md")
    args = parser.parse_args(argv)

    rasters = [validate_raster_layer(Path(args.dem)), validate_raster_layer(Path(args.sar))]
    vectors = [validate_vector_layer(Path(args.marks)), validate_vector_layer(Path(args.buildings))]

    crs_mismatches = find_crs_mismatches(rasters[0].crs, rasters[1:], vectors)
    report = render_dataset_markdown(rasters, vectors, crs_mismatches)
    output_path = Path(args.out_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    print(f"Report written to {args.out_report}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(argv)
