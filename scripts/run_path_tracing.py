#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.transform import rowcol
from rasterio.warp import Resampling, reproject

try:
    from skimage.morphology import skeletonize as sk_skeletonize
except ImportError:  # Optional dependency, fallback handled below.
    sk_skeletonize = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lamp.config import DEFAULT_CONFIG, PipelineConfig
from lamp.tasks.path_tracing.gis.export_layers import write_raster_float32, write_vector
from lamp.tasks.path_tracing.gis.raster_to_vector import mask_to_polygon_gdf, path_records_to_gdf, skeleton_to_centerline_gdf
from lamp.tasks.path_tracing.preprocessing.dem_processing import compute_slope_norm, read_raster
from lamp.tasks.path_tracing.preprocessing.terrain_features import compute_roughness, derive_surface_penalty
from lamp.tasks.path_tracing.simulation.calibration import (
    calibrate_weights,
    default_weight_grid,
    evaluate_topk_metrics,
    rasterize_known_paths,
)
from lamp.tasks.path_tracing.simulation.cost_surface import compute_cost_surface
from lamp.tasks.path_tracing.simulation.probabilistic_paths import sample_probabilistic_paths
from lamp.tasks.path_tracing.vision.learned_prior import load_learned_path_prior
from lamp.tasks.path_tracing.vision.path_segmentation import detect_visible_path_prior


def align_band_to_reference(src_path: Path, ref_profile: dict) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        dst = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_profile["transform"],
            dst_crs=ref_profile["crs"],
            resampling=Resampling.bilinear,
        )
        nodata = src.nodata
        if nodata is not None:
            dst = np.where(np.isclose(dst, nodata), np.nan, dst)
        return dst


def load_terminals(marks_path: Path, crs: object, transform: rasterio.Affine, shape: tuple[int, int]) -> list[tuple[int, int]]:
    gdf = gpd.read_file(marks_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)

    pts: list[tuple[int, int]] = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        if geom.geom_type == "Point":
            geoms = [geom]
        elif geom.geom_type in ("MultiPoint", "GeometryCollection"):
            geoms = [g for g in geom.geoms if g.geom_type == "Point"]
        else:
            continue

        for p in geoms:
            r, c = rowcol(transform, float(p.x), float(p.y))
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                pts.append((int(r), int(c)))

    seen = set()
    uniq = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def build_obstacle_mask(buildings_path: Path, crs: object, profile: dict) -> np.ndarray:
    gdf = gpd.read_file(buildings_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    return rasterize(
        shapes,
        out_shape=(profile["height"], profile["width"]),
        transform=profile["transform"],
        fill=0,
        default_value=1,
        dtype="uint8",
    ).astype(bool)


def run(config: PipelineConfig) -> dict:
    config.out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(config.dem_path) as dem_src:
        dem_meta = {
            "crs": str(dem_src.crs),
            "shape": [int(dem_src.height), int(dem_src.width)],
            "resolution": [float(dem_src.res[0]), float(dem_src.res[1])],
            "bounds": [float(v) for v in dem_src.bounds],
        }
    with rasterio.open(config.sar_path) as sar_src:
        sar_meta = {
            "crs": str(sar_src.crs),
            "shape": [int(sar_src.height), int(sar_src.width)],
            "resolution": [float(sar_src.res[0]), float(sar_src.res[1])],
            "bounds": [float(v) for v in sar_src.bounds],
        }

    dem_bundle = read_raster(config.dem_path)
    dem = dem_bundle.data

    sar = align_band_to_reference(
        config.sar_path,
        {
            "height": dem_bundle.profile["height"],
            "width": dem_bundle.profile["width"],
            "transform": dem_bundle.transform,
            "crs": dem_bundle.crs,
        },
    )

    slope_norm = compute_slope_norm(dem, dem_bundle.transform)
    roughness = compute_roughness(dem)
    surface_penalty = derive_surface_penalty(sar, slope_norm)

    if config.path_prior_mode == "learned":
        if config.path_prior_raster is None:
            # Fallback keeps CLI usable when model output is not yet generated.
            path_prior = detect_visible_path_prior(sar, slope_norm)
            path_prior_source = "deterministic-fallback"
        else:
            path_prior = load_learned_path_prior(
                prior_raster=config.path_prior_raster,
                ref_transform=dem_bundle.transform,
                ref_crs=dem_bundle.crs,
                ref_shape=(dem_bundle.profile["height"], dem_bundle.profile["width"]),
            )
            path_prior_source = "learned"
    else:
        path_prior = detect_visible_path_prior(sar, slope_norm)
        path_prior_source = "deterministic"

    obstacles = build_obstacle_mask(
        config.buildings_path,
        dem_bundle.crs,
        {
            "height": dem_bundle.profile["height"],
            "width": dem_bundle.profile["width"],
            "transform": dem_bundle.transform,
        },
    )

    terminals = load_terminals(
        config.marks_path,
        dem_bundle.crs,
        dem_bundle.transform,
        (dem_bundle.profile["height"], dem_bundle.profile["width"]),
    )
    if len(terminals) < 2:
        raise RuntimeError("Need at least 2 valid terminal points from marks shapefile")

    for r, c in terminals:
        obstacles[r, c] = False

    selected_weights = (
        config.cost_w_slope,
        config.cost_w_roughness,
        config.cost_w_surface,
        config.cost_w_path_prior,
    )

    known_path_mask = None
    calibration_report = None

    if config.known_paths_path is not None:
        known_path_mask = rasterize_known_paths(
            config.known_paths_path,
            out_shape=(dem_bundle.profile["height"], dem_bundle.profile["width"]),
            transform=dem_bundle.transform,
            crs=dem_bundle.crs,
        )

    if config.calibrate_weights:
        if known_path_mask is None:
            raise RuntimeError("--calibrate-weights requires --known-paths")
        best_w, calib_results = calibrate_weights(
            slope_norm=slope_norm,
            roughness=roughness,
            surface_penalty=surface_penalty,
            path_prior=path_prior,
            obstacle_mask=obstacles,
            terminals=terminals,
            known_path_mask=known_path_mask,
            samples_per_pair=config.calibration_samples,
            top_k=config.top_k_paths,
            rng_seed=config.rng_seed,
            temperature=config.noise_temperature,
            weight_candidates=default_weight_grid(selected_weights),
        )
        selected_weights = best_w
        calibration_report = {
            "best_weights": {
                "slope": best_w[0],
                "roughness": best_w[1],
                "surface": best_w[2],
                "path_prior": best_w[3],
            },
            "results": [
                {
                    "weights": {
                        "slope": r.weights[0],
                        "roughness": r.weights[1],
                        "surface": r.weights[2],
                        "path_prior": r.weights[3],
                    },
                    "topk_recall": r.topk_recall,
                    "iou": r.iou,
                    "precision": r.precision,
                    "f1": r.f1,
                }
                for r in calib_results
            ],
        }
        (config.out_dir / "calibration_report.json").write_text(json.dumps(calibration_report, indent=2), encoding="utf-8")

    cost = compute_cost_surface(
        slope_norm,
        roughness,
        surface_penalty,
        path_prior,
        obstacle_mask=obstacles,
        weights=selected_weights,
    )

    all_pairs = list(combinations(range(len(terminals)), 2))
    if config.max_pairs <= 0 or config.max_pairs >= len(all_pairs):
        pairs = all_pairs
    else:
        pairs = all_pairs[: config.max_pairs]

    all_records = []
    density_acc = np.zeros_like(cost, dtype=np.float32)
    processed_pairs = 0
    successful_pairs = 0

    for idx, (src_idx, dst_idx) in enumerate(pairs):
        recs, density, successful = sample_probabilistic_paths(
            base_cost=cost,
            start=terminals[src_idx],
            goal=terminals[dst_idx],
            samples=config.samples_per_pair,
            temperature=config.noise_temperature,
            top_k=config.top_k_paths,
            seed=config.rng_seed + idx,
        )
        processed_pairs += 1
        if successful > 0:
            successful_pairs += 1
        density_acc += density
        for rec in recs:
            all_records.append((src_idx, dst_idx, rec))

    if processed_pairs > 0:
        density_acc /= float(processed_pairs)

    finite = np.isfinite(density_acc)
    threshold = float(np.nanpercentile(density_acc[finite], 92)) if finite.any() else 1.0
    dense_mask = density_acc >= threshold
    if sk_skeletonize is not None:
        skeleton_mask = sk_skeletonize(dense_mask).astype(np.uint8)
    else:
        skeleton_mask = dense_mask.astype(np.uint8)
    lost_mask = (dense_mask & (path_prior < 0.35)).astype(np.uint8)
    lost_skeleton_mask = ((skeleton_mask > 0) & (path_prior < 0.35)).astype(np.uint8)

    paths_gdf = path_records_to_gdf(all_records, dem_bundle.transform, dem_bundle.crs)
    centerline_gdf = skeleton_to_centerline_gdf(skeleton_mask, dem_bundle.transform, dem_bundle.crs)
    lost_centerline_gdf = skeleton_to_centerline_gdf(lost_skeleton_mask, dem_bundle.transform, dem_bundle.crs)
    cell_area = abs(float(dem_bundle.transform.a) * float(dem_bundle.transform.e))
    lost_gdf = mask_to_polygon_gdf(lost_mask, dem_bundle.transform, dem_bundle.crs, min_area=cell_area * 4.0)

    write_vector(paths_gdf, config.out_dir / "predicted_paths.geojson")
    write_vector(paths_gdf, config.out_dir / "predicted_paths.gpkg")
    write_vector(centerline_gdf, config.out_dir / "predicted_centerlines.geojson")
    write_vector(lost_centerline_gdf, config.out_dir / "lost_path_centerlines.geojson")
    write_vector(lost_gdf, config.out_dir / "lost_path_candidates.geojson")

    raster_profile = dem_bundle.profile.copy()
    raster_profile.update(
        height=dem_bundle.profile["height"],
        width=dem_bundle.profile["width"],
        transform=dem_bundle.transform,
        crs=dem_bundle.crs,
    )
    write_raster_float32(config.out_dir / "probability_heatmap.tif", density_acc, raster_profile)
    write_raster_float32(config.out_dir / "detected_paths.tif", path_prior.astype(np.float32), raster_profile)
    write_raster_float32(config.out_dir / "movement_cost.tif", np.where(np.isfinite(cost), cost, np.nan), raster_profile)
    write_raster_float32(config.out_dir / "movement_dense_mask.tif", dense_mask.astype(np.float32), raster_profile)
    write_raster_float32(config.out_dir / "skeleton_mask.tif", skeleton_mask.astype(np.float32), raster_profile)
    write_raster_float32(config.out_dir / "lost_paths_mask.tif", lost_mask.astype(np.float32), raster_profile)

    metrics = None
    if known_path_mask is not None:
        pred_topk_mask, metrics = evaluate_topk_metrics(
            all_records,
            shape=cost.shape,
            gt_mask=known_path_mask,
            top_k=config.top_k_paths,
        )
        write_raster_float32(config.out_dir / "known_paths_mask.tif", known_path_mask.astype(np.float32), raster_profile)
        write_raster_float32(config.out_dir / "predicted_topk_mask.tif", pred_topk_mask.astype(np.float32), raster_profile)

    preprocess_report = {
        "dem": dem_meta,
        "sar": sar_meta,
        "alignment": {
            "target_grid": {
                "crs": str(dem_bundle.crs),
                "shape": [int(dem_bundle.profile["height"]), int(dem_bundle.profile["width"])],
                "resolution": [float(dem_bundle.transform.a), float(abs(dem_bundle.transform.e))],
            },
            "crs_equal_before_reproject": dem_meta["crs"] == sar_meta["crs"],
            "resolution_equal_before_reproject": np.allclose(dem_meta["resolution"], sar_meta["resolution"]),
        },
        "path_prior": {
            "mode": path_prior_source,
            "source_raster": str(config.path_prior_raster) if config.path_prior_raster else None,
        },
    }
    (config.out_dir / "preprocess_report.json").write_text(json.dumps(preprocess_report, indent=2), encoding="utf-8")

    summary = {
        "terminals": len(terminals),
        "candidate_pairs": len(all_pairs),
        "processed_pairs": processed_pairs,
        "successful_pairs": successful_pairs,
        "path_features": int(len(paths_gdf)),
        "centerline_features": int(len(centerline_gdf)),
        "lost_centerline_features": int(len(lost_centerline_gdf)),
        "lost_path_polygons": int(len(lost_gdf)),
        "samples_per_pair": config.samples_per_pair,
        "top_k_paths": config.top_k_paths,
        "skeletonize_backend": "skimage" if sk_skeletonize is not None else "fallback-binary",
        "selected_weights": {
            "slope": selected_weights[0],
            "roughness": selected_weights[1],
            "surface": selected_weights[2],
            "path_prior": selected_weights[3],
        },
    }
    if metrics is not None:
        summary.update(metrics)
    if calibration_report is not None:
        summary["calibration_candidates"] = len(calibration_report["results"])

    (config.out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> PipelineConfig:
    parser = argparse.ArgumentParser(description="LAMP Task 1 path inference pipeline")
    parser.add_argument("--dem", type=Path, default=DEFAULT_CONFIG.dem_path)
    parser.add_argument("--sar", type=Path, default=DEFAULT_CONFIG.sar_path)
    parser.add_argument("--marks", type=Path, default=DEFAULT_CONFIG.marks_path)
    parser.add_argument("--buildings", type=Path, default=DEFAULT_CONFIG.buildings_path)
    parser.add_argument("--known-paths", type=Path, default=DEFAULT_CONFIG.known_paths_path)
    parser.add_argument("--path-prior-raster", type=Path, default=DEFAULT_CONFIG.path_prior_raster)
    parser.add_argument("--path-prior-mode", choices=["learned", "deterministic"], default=DEFAULT_CONFIG.path_prior_mode)
    parser.add_argument("--out", type=Path, default=DEFAULT_CONFIG.out_dir)
    parser.add_argument("--samples", type=int, default=DEFAULT_CONFIG.samples_per_pair)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_CONFIG.max_pairs, help="0 means all terminal pairs")
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k_paths)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.noise_temperature)
    parser.add_argument("--w-slope", type=float, default=DEFAULT_CONFIG.cost_w_slope)
    parser.add_argument("--w-roughness", type=float, default=DEFAULT_CONFIG.cost_w_roughness)
    parser.add_argument("--w-surface", type=float, default=DEFAULT_CONFIG.cost_w_surface)
    parser.add_argument("--w-path-prior", type=float, default=DEFAULT_CONFIG.cost_w_path_prior)
    parser.add_argument("--calibrate-weights", action="store_true", default=DEFAULT_CONFIG.calibrate_weights)
    parser.add_argument("--calibration-samples", type=int, default=DEFAULT_CONFIG.calibration_samples)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.rng_seed)
    args = parser.parse_args()

    return PipelineConfig(
        dem_path=args.dem,
        sar_path=args.sar,
        marks_path=args.marks,
        buildings_path=args.buildings,
        known_paths_path=args.known_paths,
        path_prior_raster=args.path_prior_raster,
        path_prior_mode=args.path_prior_mode,
        out_dir=args.out,
        samples_per_pair=args.samples,
        max_pairs=args.max_pairs,
        top_k_paths=args.top_k,
        noise_temperature=args.temperature,
        cost_w_slope=args.w_slope,
        cost_w_roughness=args.w_roughness,
        cost_w_surface=args.w_surface,
        cost_w_path_prior=args.w_path_prior,
        calibrate_weights=args.calibrate_weights,
        calibration_samples=args.calibration_samples,
        rng_seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    summary = run(cfg)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
