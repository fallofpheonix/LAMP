#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from osgeo import gdal

gdal.UseExceptions()

from lamp.tasks.viewsheds.buildings import rasterize_building_heights
from lamp.tasks.viewsheds.export_gis import polygonize_raster, write_raster
from lamp.tasks.viewsheds.load_data import load_dem, load_observers
from lamp.tasks.viewsheds.scene import build_occlusion_surface
from lamp.tasks.viewsheds.terrain import inside, world_to_pixel
from lamp.tasks.viewsheds.visibility import compute_multi_observer_visibility
from lamp.tasks.viewsheds.visualize import export_hillshade


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic 3D(terrain+buildings) viewshed pipeline")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "task2"), help="Input directory")
    p.add_argument("--output-dir", default="outputs", help="Output directory")
    p.add_argument(
        "--scene-mode",
        default="fused",
        choices=["provided", "synthetic", "fused"],
        help="Occlusion surface construction strategy",
    )
    p.add_argument("--observer-height", type=float, default=1.6, help="Observer eye height in meters")
    p.add_argument("--target-height", type=float, default=0.0, help="Target height above surface in meters")
    p.add_argument("--max-distance", type=float, default=None, help="Max LOS distance in meters")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_original_path = str(data_dir / "dem_original.tif")
    dem_with_bld_path = str(data_dir / "dem_with_buildings.tif")
    bld_shp_path = str(data_dir / "BuildingFootprints.shp")
    marks_shp_path = str(data_dir / "Marks_Brief2.shp")

    dem_original = load_dem(dem_original_path)
    dem_with_buildings = load_dem(dem_with_bld_path)

    building_heights, _, _ = rasterize_building_heights(
        building_shp_path=bld_shp_path,
        reference_dem_path=dem_original_path,
        out_raster_path=str(out_dir / "building_height_raster.tif"),
    )

    occlusion_surface = build_occlusion_surface(
        dem_original=dem_original.array,
        dem_with_buildings=dem_with_buildings.array,
        building_heights=building_heights,
        mode=args.scene_mode,
    )

    write_raster(
        path=str(out_dir / "occlusion_surface.tif"),
        array=occlusion_surface.astype(np.float32),
        geotransform=dem_original.geotransform,
        projection_wkt=dem_original.projection_wkt,
        gdal_dtype=gdal.GDT_Float32,
        nodata=dem_original.nodata,
    )

    observers_xy = load_observers(marks_shp_path, dem_original.projection_wkt)
    observers_rc = []
    for obs in observers_xy:
        row, col = world_to_pixel(dem_original.geotransform, obs["x"], obs["y"])
        if inside(occlusion_surface, row, col):
            observers_rc.append({**obs, "row": row, "col": col})

    if not observers_rc:
        raise RuntimeError("No valid observers fall inside raster bounds")

    pixel_size_m = float(abs(dem_original.geotransform[1]))
    result = compute_multi_observer_visibility(
        surface=occlusion_surface,
        nodata=dem_original.nodata,
        observers_rc=observers_rc,
        observer_height=args.observer_height,
        target_height=args.target_height,
        pixel_size_m=pixel_size_m,
        max_distance_m=args.max_distance,
    )

    # Per observer outputs
    first_viewshed = None
    for entry in result["per_observer"]:
        obs_id = entry["observer"]["id"]
        arr = entry["viewshed"].astype(np.uint8)
        base = out_dir / f"viewshed_observer_{obs_id}"

        write_raster(
            path=str(base.with_suffix(".tif")),
            array=arr,
            geotransform=dem_original.geotransform,
            projection_wkt=dem_original.projection_wkt,
            gdal_dtype=gdal.GDT_Byte,
            nodata=0,
        )
        polygonize_raster(
            raster_path=str(base.with_suffix(".tif")),
            vector_path=str(base.with_suffix(".gpkg")),
            layer_name=f"viewshed_obs_{obs_id}",
            field_name="visible",
        )

        if first_viewshed is None:
            first_viewshed = arr

        stats = entry["stats"]
        print(
            f"observer={obs_id} row={entry['observer']['row']} col={entry['observer']['col']} "
            f"visible_ratio={stats['visible_ratio']:.4f} "
            f"visible_cells={stats['visible_cells']} checked={stats['checked_cells']}"
        )

    # Aggregate products
    write_raster(
        path=str(out_dir / "viewshed_probability.tif"),
        array=result["viewshed_probability"],
        geotransform=dem_original.geotransform,
        projection_wkt=dem_original.projection_wkt,
        gdal_dtype=gdal.GDT_Float32,
        nodata=-1.0,
    )
    write_raster(
        path=str(out_dir / "viewshed_all_observers.tif"),
        array=result["viewshed_any"].astype(np.uint8),
        geotransform=dem_original.geotransform,
        projection_wkt=dem_original.projection_wkt,
        gdal_dtype=gdal.GDT_Byte,
        nodata=0,
    )

    # Required task aliases: viewshed.tif and viewshed.shp
    if first_viewshed is not None:
        write_raster(
            path=str(out_dir / "viewshed.tif"),
            array=first_viewshed.astype(np.uint8),
            geotransform=dem_original.geotransform,
            projection_wkt=dem_original.projection_wkt,
            gdal_dtype=gdal.GDT_Byte,
            nodata=0,
        )
        polygonize_raster(
            raster_path=str(out_dir / "viewshed.tif"),
            vector_path=str(out_dir / "viewshed.shp"),
            layer_name="viewshed",
            field_name="visible",
        )

    export_hillshade(str(out_dir / "occlusion_surface.tif"), str(out_dir / "scene_hillshade.tif"))

    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
