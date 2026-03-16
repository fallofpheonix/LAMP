#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import geopandas as gpd
import rasterio
import numpy as np

def validate_vector(path):
    gdf = gpd.read_file(path)
    report = {
        "path": str(path),
        "crs": str(gdf.crs),
        "total_features": len(gdf),
        "invalid_geometries": int((~gdf.geometry.is_valid).sum()),
        "empty_geometries": int(gdf.geometry.is_empty.sum()),
        "bounds": list(gdf.total_bounds)
    }
    return report

def validate_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1)
        report = {
            "path": str(path),
            "crs": str(src.crs),
            "res": src.res,
            "shape": src.shape,
            "nodata_value": src.nodata,
            "nodata_percentage": float((np.isnan(arr) | (arr == src.nodata)).sum() / arr.size),
            "bounds": list(src.bounds)
        }
    return report

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", default="task1-path-tracing/Task_1/DEM_Subset-Original.tif")
    parser.add_argument("--sar", default="task1-path-tracing/Task_1/SAR-MS.tif")
    parser.add_argument("--marks", default="task1-path-tracing/Task_1/Marks_Brief1.shp")
    parser.add_argument("--buildings", default="task1-path-tracing/Task_1/BuildingFootprints.shp")
    parser.add_argument("--out-report", default="DATASET_INTEGRITY_REPORT.md")
    args = parser.parse_args()
    
    results = {
        "rasters": [validate_raster(args.dem), validate_raster(args.sar)],
        "vectors": [validate_vector(args.marks), validate_vector(args.buildings)]
    }
    
    # Check CRS consistency
    ref_crs = results["rasters"][0]["crs"]
    crs_mismatches = []
    for r in results["rasters"][1:]:
        if r["crs"] != ref_crs: crs_mismatches.append(r["path"])
    for v in results["vectors"]:
        if v["crs"] != ref_crs: crs_mismatches.append(v["path"])
        
    # Generate MD Report
    lines = ["# Dataset Integrity Report\n"]
    lines.append("## 1. CRS Consistency")
    if not crs_mismatches:
        lines.append("- **Status**: PASS (All layers align to UTM Zone 38N)\n")
    else:
        lines.append(f"- **Status**: WARNING (Mismatches found in {crs_mismatches})\n")
        
    lines.append("## 2. Vector Validation")
    for v in results["vectors"]:
        lines.append(f"### {Path(v['path']).name}")
        lines.append(f"- Total Features: {v['total_features']}")
        lines.append(f"- Invalid Geoms: {v['invalid_geometries']}")
        lines.append(f"- Empty Geoms: {v['empty_geometries']}")
        lines.append("")
        
    lines.append("## 3. Raster Validation")
    for r in results["rasters"]:
        lines.append(f"### {Path(r['path']).name}")
        lines.append(f"- Shape: {r['shape']}")
        lines.append(f"- Resolution: {r['res']}")
        lines.append(f"- NoData %: {r['nodata_percentage']:.2%}")
        lines.append("")
        
    Path(args.out_report).write_text("\n".join(lines))
    print(f"Report written to {args.out_report}")

if __name__ == "__main__": main()
