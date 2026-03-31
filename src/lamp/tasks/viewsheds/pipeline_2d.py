from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import warnings

try:
    from osgeo import gdal
    gdal.UseExceptions()
except ImportError:
    gdal = None

from lamp.tasks.viewsheds.buildings import rasterize_building_heights
from lamp.tasks.viewsheds.export_gis import polygonize_raster, write_raster
from lamp.tasks.viewsheds.load_data import load_dem, load_observers
from lamp.tasks.viewsheds.scene import build_occlusion_surface
from lamp.tasks.viewsheds.terrain import inside, world_to_pixel
from lamp.tasks.viewsheds.visibility import compute_multi_observer_visibility
from lamp.tasks.viewsheds.visualize import export_hillshade
from lamp.services.dataset_validation_service import validate_raster_layer, validate_vector_layer


def preflight_check(data_dir: Path) -> None:
    """Validate existence and format of required Task 2 inputs."""
    required_rasters = [
        data_dir / "DEM_Subset-Original.tif",
        data_dir / "DEM_Subset-WithBuildings.tif",
    ]
    required_vectors = [
        data_dir / "BuildingFootprints.shp",
        data_dir / "Marks_Brief2.shp",
    ]

    for r in required_rasters:
        if not r.exists():
            raise FileNotFoundError(f"Missing required raster: {r}")
        validate_raster_layer(r)

    for v in required_vectors:
        if not v.exists():
            raise FileNotFoundError(f"Missing required vector: {v}")
        validate_vector_layer(v)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Deterministic 2.5D(terrain+buildings) viewshed pipeline")
    p.add_argument("--data-dir", default="data/task2", help="Input directory")
    p.add_argument("--output-dir", default="outputs/viewsheds_2d", help="Output directory")
    p.add_argument(
        "--scene-mode",
        default="fused",
        choices=["provided", "synthetic", "fused"],
        help="Occlusion surface construction strategy",
    )
    p.add_argument("--observer-height", type=float, default=1.6, help="Observer eye height in meters")
    p.add_argument("--target-height", type=float, default=0.0, help="Target height above surface in meters")
    p.add_argument("--max-distance", type=float, default=None, help="Max LOS distance in meters")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if gdal is None:
        print("Error: GDAL Python bindings (osgeo) are required for Task 2.", file=sys.stderr)
        return 2

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    
    # Preflight Check
    try:
        preflight_check(data_dir)
    except Exception as e:
        print(f"Preflight validation failed: {e}", file=sys.stderr)
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)

    dem_original_path = str(data_dir / "DEM_Subset-Original.tif")
    dem_with_bld_path = str(data_dir / "DEM_Subset-WithBuildings.tif")
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
        print("Error: No valid observers fall inside raster bounds", file=sys.stderr)
        return 1

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

    print(f"2D Viewshed outputs written to: {out_dir.resolve()}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
