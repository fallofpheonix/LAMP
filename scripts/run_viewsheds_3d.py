#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from osgeo import gdal

from lamp.tasks.viewsheds.buildings import rasterize_building_heights
from lamp.tasks.viewsheds.export_gis import polygonize_raster, write_raster, write_structured_points_vtk
from lamp.tasks.viewsheds.load_data import load_dem, load_observers
from lamp.tasks.viewsheds.scene import build_occlusion_surface
from lamp.tasks.viewsheds.terrain import inside, world_to_pixel
from lamp.tasks.viewsheds.voxel_scene import (
    build_voxel_scene,
    carve_openings_from_vector,
    compute_ground_viewshed_3d,
    compute_visibility_volume_from_observer,
)


gdal.UseExceptions()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D voxel viewshed + volume pipeline")
    p.add_argument("--data-dir", default=str(ROOT / "data" / "task2"))
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--scene-mode", default="fused", choices=["provided", "synthetic", "fused"])
    p.add_argument("--observer-height", type=float, default=1.6)
    p.add_argument("--target-height", type=float, default=0.0)
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument("--z-res", type=float, default=0.5)
    p.add_argument("--openings-path", default=str(ROOT / "data" / "task2" / "openings.geojson"))
    p.add_argument("--volume-observer-id", type=int, default=1)
    p.add_argument("--volume-azimuth-steps", type=int, default=360)
    p.add_argument("--volume-elevation-steps", type=int, default=91)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dem_original = load_dem(str(data_dir / "dem_original.tif"))
    dem_with_buildings = load_dem(str(data_dir / "dem_with_buildings.tif"))

    building_heights, _, _ = rasterize_building_heights(
        building_shp_path=str(data_dir / "BuildingFootprints.shp"),
        reference_dem_path=str(data_dir / "dem_original.tif"),
    )

    # Build 2.5D occlusion surface first, then derive 3D voxel solid volume.
    occlusion_surface = build_occlusion_surface(
        dem_original=dem_original.array,
        dem_with_buildings=dem_with_buildings.array,
        building_heights=building_heights,
        mode=args.scene_mode,
    )

    scene3d = build_voxel_scene(
        terrain=dem_original.array,
        building_heights=building_heights,
        geotransform=dem_original.geotransform,
        projection_wkt=dem_original.projection_wkt,
        nodata=dem_original.nodata,
        z_res=args.z_res,
    )

    openings_used = 0
    if args.openings_path and Path(args.openings_path).exists():
        openings_used = carve_openings_from_vector(scene3d, args.openings_path)

    observers_xy = load_observers(str(data_dir / "Marks_Brief2.shp"), dem_original.projection_wkt)
    observers = []
    for obs in observers_xy:
        row, col = world_to_pixel(dem_original.geotransform, obs["x"], obs["y"])
        if inside(occlusion_surface, row, col):
            observers.append({**obs, "row": row, "col": col})

    if not observers:
        raise RuntimeError("No valid observers in terrain bounds")

    per = []
    stack = []

    for obs in observers:
        arr, stats = compute_ground_viewshed_3d(
            scene=scene3d,
            obs_row=obs["row"],
            obs_col=obs["col"],
            observer_height=args.observer_height,
            target_height=args.target_height,
            max_distance_m=args.max_distance,
        )

        rid = int(obs["id"])
        base = out_dir / f"viewshed3d_observer_{rid}"

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
            layer_name=f"viewshed3d_obs_{rid}",
            field_name="visible",
        )

        per.append({"observer": obs, "stats": stats, "viewshed": arr})
        stack.append(arr.astype(np.float32))

        print(
            f"3d observer={rid} row={obs['row']} col={obs['col']} "
            f"visible_ratio={stats['visible_ratio']:.4f} "
            f"visible_cells={stats['visible_cells']} checked={stats['checked_cells']}"
        )

    s = np.stack(stack, axis=0)
    prob = np.mean(s, axis=0).astype(np.float32)
    union = (np.max(s, axis=0) > 0).astype(np.uint8)

    write_raster(
        path=str(out_dir / "viewshed3d_probability.tif"),
        array=prob,
        geotransform=dem_original.geotransform,
        projection_wkt=dem_original.projection_wkt,
        gdal_dtype=gdal.GDT_Float32,
        nodata=-1.0,
    )
    write_raster(
        path=str(out_dir / "viewshed3d_union.tif"),
        array=union,
        geotransform=dem_original.geotransform,
        projection_wkt=dem_original.projection_wkt,
        gdal_dtype=gdal.GDT_Byte,
        nodata=0,
    )
    polygonize_raster(
        raster_path=str(out_dir / "viewshed3d_union.tif"),
        vector_path=str(out_dir / "viewshed3d_union.shp"),
        layer_name="viewshed3d_union",
        field_name="visible",
    )

    # Required aliases for easy comparison.
    first = per[0]["viewshed"].astype(np.uint8)
    write_raster(
        path=str(out_dir / "viewshed3d.tif"),
        array=first,
        geotransform=dem_original.geotransform,
        projection_wkt=dem_original.projection_wkt,
        gdal_dtype=gdal.GDT_Byte,
        nodata=0,
    )
    polygonize_raster(
        raster_path=str(out_dir / "viewshed3d.tif"),
        vector_path=str(out_dir / "viewshed3d.shp"),
        layer_name="viewshed3d",
        field_name="visible",
    )

    # Optional 3D volume from selected observer.
    vol_obs = None
    for obs in observers:
        if int(obs["id"]) == int(args.volume_observer_id):
            vol_obs = obs
            break
    if vol_obs is None:
        vol_obs = observers[0]

    vol = compute_visibility_volume_from_observer(
        scene=scene3d,
        obs_row=vol_obs["row"],
        obs_col=vol_obs["col"],
        observer_height=args.observer_height,
        azimuth_steps=args.volume_azimuth_steps,
        elevation_steps=args.volume_elevation_steps,
        max_distance_m=args.max_distance,
    )

    origin_x = dem_original.geotransform[0]
    origin_y = dem_original.geotransform[3] + dem_original.geotransform[5] * dem_original.array.shape[0]
    spacing_x = abs(dem_original.geotransform[1])
    spacing_y = abs(dem_original.geotransform[5])
    origin_z = scene3d.z_min
    spacing_z = scene3d.z_res

    write_structured_points_vtk(
        path=str(out_dir / "viewshed3d_volume.vtk"),
        volume_zyx=vol,
        origin_xyz=(origin_x, origin_y, origin_z),
        spacing_xyz=(spacing_x, spacing_y, spacing_z),
        scalar_name="visibility",
    )

    print(f"3d openings applied: {openings_used}")
    print(f"3d outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
