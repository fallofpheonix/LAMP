#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
# # sys.path.insert(0, str(repo_root))

import numpy as np
from osgeo import gdal

from lamp.tasks.viewsheds.buildings import rasterize_building_heights
from lamp.tasks.viewsheds.export_gis import polygonize_raster, write_raster
from lamp.tasks.viewsheds.load_data import load_dem, load_observers
from lamp.tasks.viewsheds.ml_features import flat_to_raster, observer_feature_matrix
from lamp.tasks.viewsheds.ml_model import LogisticVisibilityModel
from lamp.tasks.viewsheds.scene import build_occlusion_surface
from lamp.tasks.viewsheds.terrain import inside, world_to_pixel


gdal.UseExceptions()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Predict viewsheds with trained model and export GIS layers")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--scene-mode", default="fused", choices=["provided", "synthetic", "fused"])
    p.add_argument("--observer-height", type=float, default=1.6)
    p.add_argument("--target-height", type=float, default=0.0)
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument("--model-path", default="outputs/viewshed_model.npz")
    p.add_argument("--metrics-path", default="outputs/viewshed_model_metrics.json")
    p.add_argument("--threshold", type=float, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model = LogisticVisibilityModel.load(str(model_path))

    threshold = args.threshold
    if threshold is None:
        metrics_path = Path(args.metrics_path)
        if not metrics_path.is_absolute():
            metrics_path = Path.cwd() / metrics_path
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            threshold = float(metrics.get("validation_best_threshold", 0.5))
        else:
            threshold = 0.5

    dem_original = load_dem(str(data_dir / "dem_original.tif"))
    dem_with_buildings = load_dem(str(data_dir / "dem_with_buildings.tif"))

    building_heights, _, _ = rasterize_building_heights(
        building_shp_path=str(data_dir / "BuildingFootprints.shp"),
        reference_dem_path=str(data_dir / "dem_original.tif"),
    )

    surface = build_occlusion_surface(
        dem_original=dem_original.array,
        dem_with_buildings=dem_with_buildings.array,
        building_heights=building_heights,
        mode=args.scene_mode,
    )

    pixel_size_m = float(abs(dem_original.geotransform[1]))
    observers = load_observers(str(data_dir / "Marks_Brief2.shp"), dem_original.projection_wkt)

    pred_rasters = []
    for obs in observers:
        row, col = world_to_pixel(dem_original.geotransform, obs["x"], obs["y"])
        if not inside(surface, row, col):
            continue

        X, valid_mask = observer_feature_matrix(
            surface=surface,
            obs_row=row,
            obs_col=col,
            observer_height=args.observer_height,
            target_height=args.target_height,
            nodata=dem_original.nodata,
            max_distance_m=args.max_distance,
            pixel_size_m=pixel_size_m,
        )

        prob_flat = model.predict_proba(X)
        prob_raster = flat_to_raster(prob_flat, valid_mask, fill_value=0.0)
        bin_raster = (prob_raster >= threshold).astype(np.uint8)

        rid = int(obs["id"])
        prob_path = out_dir / f"viewshed_model_prob_observer_{rid}.tif"
        bin_path = out_dir / f"viewshed_model_bin_observer_{rid}.tif"
        vec_path = out_dir / f"viewshed_model_bin_observer_{rid}.gpkg"

        write_raster(
            path=str(prob_path),
            array=prob_raster,
            geotransform=dem_original.geotransform,
            projection_wkt=dem_original.projection_wkt,
            gdal_dtype=gdal.GDT_Float32,
            nodata=-1.0,
        )
        write_raster(
            path=str(bin_path),
            array=bin_raster,
            geotransform=dem_original.geotransform,
            projection_wkt=dem_original.projection_wkt,
            gdal_dtype=gdal.GDT_Byte,
            nodata=0,
        )
        polygonize_raster(
            raster_path=str(bin_path),
            vector_path=str(vec_path),
            layer_name=f"viewshed_model_obs_{rid}",
            field_name="visible",
        )

        pred_rasters.append(prob_raster)
        print(f"observer={rid} model outputs: {prob_path.name}, {bin_path.name}, {vec_path.name}")

    if pred_rasters:
        stacked = np.stack(pred_rasters, axis=0)
        mean_prob = np.mean(stacked, axis=0).astype(np.float32)
        union_bin = (np.max(stacked, axis=0) >= threshold).astype(np.uint8)

        write_raster(
            path=str(out_dir / "viewshed_model_probability.tif"),
            array=mean_prob,
            geotransform=dem_original.geotransform,
            projection_wkt=dem_original.projection_wkt,
            gdal_dtype=gdal.GDT_Float32,
            nodata=-1.0,
        )
        write_raster(
            path=str(out_dir / "viewshed_model_union.tif"),
            array=union_bin,
            geotransform=dem_original.geotransform,
            projection_wkt=dem_original.projection_wkt,
            gdal_dtype=gdal.GDT_Byte,
            nodata=0,
        )
        polygonize_raster(
            raster_path=str(out_dir / "viewshed_model_union.tif"),
            vector_path=str(out_dir / "viewshed_model_union.shp"),
            layer_name="viewshed_model_union",
            field_name="visible",
        )
        print(f"Aggregate model outputs written to: {out_dir.resolve()} (threshold={threshold:.3f})")


if __name__ == "__main__":
    main()
