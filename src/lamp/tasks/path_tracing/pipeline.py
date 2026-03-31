from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, replace
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

from lamp.tasks.path_tracing.config import DEFAULT_CONFIG, PipelineConfig
from lamp.tasks.path_tracing.gis.export_layers import write_raster_float32, write_vector
from lamp.tasks.path_tracing.gis.raster_to_vector import (
    mask_to_polygon_gdf,
    path_records_to_gdf,
    skeleton_to_centerline_gdf,
)
from lamp.tasks.path_tracing.preprocessing.dem_processing import compute_slope_norm, read_raster
from lamp.tasks.path_tracing.preprocessing.terrain_features import (
    compute_roughness,
    derive_surface_penalty,
)
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
from lamp.services.dataset_validation_service import validate_raster_layer, validate_vector_layer


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


def load_terminals(
    marks_path: Path,
    crs: object,
    transform: rasterio.Affine,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
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

        for point in geoms:
            row, col = rowcol(transform, float(point.x), float(point.y))
            if 0 <= row < shape[0] and 0 <= col < shape[1]:
                pts.append((int(row), int(col)))

    seen: set[tuple[int, int]] = set()
    uniq: list[tuple[int, int]] = []
    for point in pts:
        if point not in seen:
            seen.add(point)
            uniq.append(point)
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


@dataclass
class ScenarioArtifacts:
    summary: dict
    density: np.ndarray
    raster_profile: dict
    output_dir: Path


def _transform_components(transform: rasterio.Affine) -> tuple[float, float, float, float, float, float]:
    return (
        float(transform.a),
        float(transform.b),
        float(transform.c),
        float(transform.d),
        float(transform.e),
        float(transform.f),
    )


def load_visibility_probability(src_path: Path | None, ref_profile: dict) -> tuple[np.ndarray | None, dict | None]:
    if src_path is None:
        return None, None

    with rasterio.open(src_path) as src:
        source_meta = {
            "source_path": str(src_path),
            "source_crs": str(src.crs),
            "source_shape": [int(src.height), int(src.width)],
            "source_transform": list(_transform_components(src.transform)),
            "source_semantics": "mean_visibility_probability",
        }
        same_crs = str(src.crs) == str(ref_profile["crs"])
        same_shape = src.height == ref_profile["height"] and src.width == ref_profile["width"]
        same_transform = np.allclose(
            _transform_components(src.transform),
            _transform_components(ref_profile["transform"]),
        )

        if same_crs and same_shape and same_transform:
            dst = src.read(1).astype(np.float32)
        else:
            dst = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float32)
            reproject(
                source=src.read(1),
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
        dst = np.clip(dst, 0.0, 1.0).astype(np.float32)

    alignment = {
        **source_meta,
        "reference_crs": str(ref_profile["crs"]),
        "reference_shape": [int(ref_profile["height"]), int(ref_profile["width"])],
        "reference_transform": list(_transform_components(ref_profile["transform"])),
        "crs_equal_before_reproject": same_crs,
        "shape_equal_before_reproject": same_shape,
        "transform_equal_before_reproject": same_transform,
        "resampled": not (same_crs and same_shape and same_transform),
    }
    return dst, alignment


def preflight_check(config: PipelineConfig) -> None:
    """Validate existence and format of required Task 1 inputs."""
    required_rasters = [config.dem_path, config.sar_path]
    required_vectors = [config.marks_path, config.buildings_path]

    for r in required_rasters:
        if not r.exists():
            raise FileNotFoundError(f"Missing required raster: {r}")
        validate_raster_layer(r)

    for v in required_vectors:
        if not v.exists():
            raise FileNotFoundError(f"Missing required vector: {v}")
        validate_vector_layer(v)

    if config.known_paths_path and config.known_paths_path.exists():
        validate_vector_layer(config.known_paths_path)

    if config.visibility_raster and config.visibility_raster.exists():
        validate_raster_layer(config.visibility_raster)


def write_comparison_figure(
    path: Path,
    baseline_density: np.ndarray,
    coupled_density: np.ndarray,
    delta_density: np.ndarray,
) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    panels = [
        (baseline_density, "Baseline Density", "viridis"),
        (coupled_density, "Coupled Density", "viridis"),
        (delta_density, "Coupled - Baseline", "coolwarm"),
    ]
    for ax, (array, title, cmap) in zip(axes, panels):
        image = ax.imshow(array, cmap=cmap)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(image, ax=ax, shrink=0.8)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _load_path_prior(
    config: PipelineConfig,
    sar: np.ndarray,
    slope_norm: np.ndarray,
    dem_transform: rasterio.Affine,
    dem_crs: object,
    dem_shape: tuple[int, int],
) -> tuple[np.ndarray, str]:
    if config.path_prior_mode != "learned":
        return detect_visible_path_prior(sar, slope_norm), "deterministic"

    prior_path = config.path_prior_raster
    if prior_path is None or not prior_path.exists():
        if config.path_prior_mode == "learned":
            warnings.warn(
                f"Learned prior raster missing at {prior_path}. Falling back to deterministic path prior.",
                RuntimeWarning,
            )
        return detect_visible_path_prior(sar, slope_norm), "deterministic-fallback"

    return (
        load_learned_path_prior(
            prior_raster=prior_path,
            ref_transform=dem_transform,
            ref_crs=dem_crs,
            ref_shape=dem_shape,
        ),
        "learned",
    )


def _run_single(
    config: PipelineConfig,
    *,
    scenario_name: str,
    scenario_out_dir: Path,
    visibility_probability: np.ndarray | None = None,
    visibility_alignment: dict | None = None,
) -> ScenarioArtifacts:
    preflight_check(config)
    output_dir = scenario_out_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(config.dem_path) as dem_src:
        dem_meta = {
            "crs": str(dem_src.crs),
            "shape": [int(dem_src.height), int(dem_src.width)],
            "resolution": [float(dem_src.res[0]), float(dem_src.res[1])],
            "bounds": [float(value) for value in dem_src.bounds],
        }
    with rasterio.open(config.sar_path) as sar_src:
        sar_meta = {
            "crs": str(sar_src.crs),
            "shape": [int(sar_src.height), int(sar_src.width)],
            "resolution": [float(sar_src.res[0]), float(sar_src.res[1])],
            "bounds": [float(value) for value in sar_src.bounds],
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
    path_prior, path_prior_source = _load_path_prior(
        config,
        sar,
        slope_norm,
        dem_bundle.transform,
        dem_bundle.crs,
        (dem_bundle.profile["height"], dem_bundle.profile["width"]),
    )

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

    for row, col in terminals:
        obstacles[row, col] = False

    selected_weights = (
        config.cost_w_slope,
        config.cost_w_roughness,
        config.cost_w_surface,
        config.cost_w_path_prior,
        config.cost_w_visibility if visibility_probability is not None else 0.0,
    )

    known_path_mask = None
    calibration_report = None

    if config.known_paths_path is not None:
        if not config.known_paths_path.exists():
            raise FileNotFoundError(f"Known paths layer not found: {config.known_paths_path}")
        known_path_mask = rasterize_known_paths(
            config.known_paths_path,
            out_shape=(dem_bundle.profile["height"], dem_bundle.profile["width"]),
            transform=dem_bundle.transform,
            crs=dem_bundle.crs,
        )

    if config.calibrate_weights:
        if known_path_mask is None:
            raise RuntimeError("--calibrate-weights requires an existing --known-paths layer")
        best_weights, calibration_results = calibrate_weights(
            slope_norm=slope_norm,
            roughness=roughness,
            surface_penalty=surface_penalty,
            path_prior=path_prior,
            visibility_probability=visibility_probability,
            obstacle_mask=obstacles,
            terminals=terminals,
            known_path_mask=known_path_mask,
            samples_per_pair=config.calibration_samples,
            top_k=config.top_k_paths,
            rng_seed=config.rng_seed,
            temperature=config.noise_temperature,
            weight_candidates=default_weight_grid(
                selected_weights,
                enable_visibility_search=visibility_probability is not None,
            ),
        )
        selected_weights = best_weights
        calibration_report = {
            "best_weights": {
                "slope": best_weights[0],
                "roughness": best_weights[1],
                "surface": best_weights[2],
                "path_prior": best_weights[3],
                "visibility": best_weights[4],
            },
            "results": [
                {
                    "weights": {
                        "slope": result.weights[0],
                        "roughness": result.weights[1],
                        "surface": result.weights[2],
                        "path_prior": result.weights[3],
                        "visibility": result.weights[4],
                    },
                    "topk_recall": result.topk_recall,
                    "iou": result.iou,
                    "precision": result.precision,
                    "f1": result.f1,
                }
                for result in calibration_results
            ],
        }
        (output_dir / "calibration_report.json").write_text(
            json.dumps(calibration_report, indent=2),
            encoding="utf-8",
        )

    cost = compute_cost_surface(
        slope_norm,
        roughness,
        surface_penalty,
        path_prior,
        visibility_probability=visibility_probability,
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

    for index, (src_idx, dst_idx) in enumerate(pairs):
        records, density, successful = sample_probabilistic_paths(
            base_cost=cost,
            start=terminals[src_idx],
            goal=terminals[dst_idx],
            samples=config.samples_per_pair,
            temperature=config.noise_temperature,
            top_k=config.top_k_paths,
            seed=config.rng_seed + index,
        )
        processed_pairs += 1
        if successful > 0:
            successful_pairs += 1
        density_acc += density
        for record in records:
            all_records.append((src_idx, dst_idx, record))

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
    lost_gdf = mask_to_polygon_gdf(
        lost_mask,
        dem_bundle.transform,
        dem_bundle.crs,
        min_area=cell_area * 4.0,
    )

    write_vector(paths_gdf, output_dir / "predicted_paths.geojson")
    write_vector(paths_gdf, output_dir / "predicted_paths.gpkg")
    write_vector(centerline_gdf, output_dir / "predicted_centerlines.geojson")
    write_vector(lost_centerline_gdf, output_dir / "lost_path_centerlines.geojson")
    write_vector(lost_gdf, output_dir / "lost_path_candidates.geojson")

    raster_profile = dem_bundle.profile.copy()
    raster_profile.update(
        height=dem_bundle.profile["height"],
        width=dem_bundle.profile["width"],
        transform=dem_bundle.transform,
        crs=dem_bundle.crs,
    )
    write_raster_float32(output_dir / "probability_heatmap.tif", density_acc, raster_profile)
    write_raster_float32(output_dir / "detected_paths.tif", path_prior.astype(np.float32), raster_profile)
    write_raster_float32(
        output_dir / "movement_cost.tif",
        np.where(np.isfinite(cost), cost, np.nan),
        raster_profile,
    )
    write_raster_float32(
        output_dir / "movement_dense_mask.tif",
        dense_mask.astype(np.float32),
        raster_profile,
    )
    write_raster_float32(output_dir / "skeleton_mask.tif", skeleton_mask.astype(np.float32), raster_profile)
    write_raster_float32(output_dir / "lost_paths_mask.tif", lost_mask.astype(np.float32), raster_profile)
    if visibility_probability is not None:
        write_raster_float32(
            output_dir / "visibility_probability.tif",
            visibility_probability.astype(np.float32),
            raster_profile,
        )

    metrics = None
    if known_path_mask is not None:
        pred_topk_mask, metrics = evaluate_topk_metrics(
            all_records,
            shape=cost.shape,
            gt_mask=known_path_mask,
            top_k=config.top_k_paths,
        )
        write_raster_float32(
            output_dir / "known_paths_mask.tif",
            known_path_mask.astype(np.float32),
            raster_profile,
        )
        write_raster_float32(
            output_dir / "predicted_topk_mask.tif",
            pred_topk_mask.astype(np.float32),
            raster_profile,
        )

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
            "resolution_equal_before_reproject": np.allclose(
                dem_meta["resolution"],
                sar_meta["resolution"],
            ),
        },
        "path_prior": {
            "mode": path_prior_source,
            "source_raster": str(config.path_prior_raster) if config.path_prior_raster else None,
        },
        "scenario": scenario_name,
        "visibility": {
            "source": config.visibility_source if visibility_probability is not None else None,
            "enabled": visibility_probability is not None,
            "alignment": visibility_alignment,
        },
    }
    (output_dir / "preprocess_report.json").write_text(
        json.dumps(preprocess_report, indent=2),
        encoding="utf-8",
    )

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
            "visibility": selected_weights[4],
        },
        "scenario": scenario_name,
    }
    if metrics is not None:
        summary.update(metrics)
    if calibration_report is not None:
        summary["calibration_candidates"] = len(calibration_report["results"])

    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return ScenarioArtifacts(summary=summary, density=density_acc, raster_profile=raster_profile, output_dir=output_dir)


def run(config: PipelineConfig) -> dict:
    if not config.compare_visibility_coupling:
        visibility_probability = None
        visibility_alignment = None
        if config.visibility_raster is not None and config.cost_w_visibility > 0.0:
            dem_bundle = read_raster(config.dem_path)
            visibility_probability, visibility_alignment = load_visibility_probability(
                config.visibility_raster,
                {
                    "height": dem_bundle.profile["height"],
                    "width": dem_bundle.profile["width"],
                    "transform": dem_bundle.transform,
                    "crs": dem_bundle.crs,
                },
            )
        return _run_single(
            config,
            scenario_name="visibility_coupled" if visibility_probability is not None else "baseline",
            scenario_out_dir=config.out_dir,
            visibility_probability=visibility_probability,
            visibility_alignment=visibility_alignment,
        ).summary

    if config.visibility_raster is None:
        raise RuntimeError("--compare-visibility-coupling requires --visibility-raster")

    dem_bundle = read_raster(config.dem_path)
    visibility_probability, visibility_alignment = load_visibility_probability(
        config.visibility_raster,
        {
            "height": dem_bundle.profile["height"],
            "width": dem_bundle.profile["width"],
            "transform": dem_bundle.transform,
            "crs": dem_bundle.crs,
        },
    )

    baseline_config = replace(config, out_dir=config.out_dir / "baseline", cost_w_visibility=0.0)
    coupled_config = replace(config, out_dir=config.out_dir / "visibility_coupled")

    baseline = _run_single(
        baseline_config,
        scenario_name="baseline",
        scenario_out_dir=baseline_config.out_dir,
        visibility_probability=None,
        visibility_alignment=None,
    )
    coupled = _run_single(
        coupled_config,
        scenario_name="visibility_coupled",
        scenario_out_dir=coupled_config.out_dir,
        visibility_probability=visibility_probability,
        visibility_alignment=visibility_alignment,
    )

    delta_density = coupled.density - baseline.density
    write_raster_float32(
        config.out_dir / "comparison_density_delta.tif",
        delta_density,
        baseline.raster_profile,
    )
    figure_written = write_comparison_figure(
        config.out_dir / "comparison_visibility_coupling.png",
        baseline.density,
        coupled.density,
        delta_density,
    )

    metrics_delta: dict[str, float] = {}
    for metric in ("topk_recall", "iou", "precision", "f1"):
        if metric in baseline.summary and metric in coupled.summary:
            metrics_delta[f"{metric}_delta"] = float(coupled.summary[metric]) - float(
                baseline.summary[metric]
            )

    comparison_summary = {
        "comparison_mode": True,
        "visibility_source": config.visibility_source,
        "visibility_raster": str(config.visibility_raster),
        "figure_written": figure_written,
        "baseline_dir": str(baseline.output_dir),
        "visibility_coupled_dir": str(coupled.output_dir),
        "baseline": baseline.summary,
        "visibility_coupled": coupled.summary,
        "delta_metrics": metrics_delta,
        "delta_density": {
            "min": float(np.nanmin(delta_density)),
            "max": float(np.nanmax(delta_density)),
            "mean": float(np.nanmean(delta_density)),
        },
    }
    (config.out_dir / "comparison_summary.json").write_text(
        json.dumps(comparison_summary, indent=2),
        encoding="utf-8",
    )
    return comparison_summary


def parse_args(argv: list[str] | None = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="LAMP Task 1 path inference pipeline")
    parser.add_argument("--dem", type=Path, default=DEFAULT_CONFIG.dem_path)
    parser.add_argument("--sar", type=Path, default=DEFAULT_CONFIG.sar_path)
    parser.add_argument("--marks", type=Path, default=DEFAULT_CONFIG.marks_path)
    parser.add_argument("--buildings", type=Path, default=DEFAULT_CONFIG.buildings_path)
    parser.add_argument("--known-paths", type=Path, default=DEFAULT_CONFIG.known_paths_path)
    parser.add_argument("--path-prior-raster", type=Path, default=DEFAULT_CONFIG.path_prior_raster)
    parser.add_argument(
        "--path-prior-mode",
        choices=["learned", "deterministic"],
        default=DEFAULT_CONFIG.path_prior_mode,
    )
    parser.add_argument("--visibility-raster", type=Path, default=DEFAULT_CONFIG.visibility_raster)
    parser.add_argument(
        "--visibility-source",
        choices=["deterministic", "model"],
        default=DEFAULT_CONFIG.visibility_source,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_CONFIG.out_dir)
    parser.add_argument("--samples", type=int, default=DEFAULT_CONFIG.samples_per_pair)
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=DEFAULT_CONFIG.max_pairs,
        help="0 means all terminal pairs",
    )
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k_paths)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.noise_temperature)
    parser.add_argument("--w-slope", type=float, default=DEFAULT_CONFIG.cost_w_slope)
    parser.add_argument("--w-roughness", type=float, default=DEFAULT_CONFIG.cost_w_roughness)
    parser.add_argument("--w-surface", type=float, default=DEFAULT_CONFIG.cost_w_surface)
    parser.add_argument("--w-path-prior", type=float, default=DEFAULT_CONFIG.cost_w_path_prior)
    parser.add_argument("--w-visibility", type=float, default=DEFAULT_CONFIG.cost_w_visibility)
    parser.add_argument(
        "--calibrate-weights",
        action="store_true",
        default=DEFAULT_CONFIG.calibrate_weights,
    )
    parser.add_argument(
        "--compare-visibility-coupling",
        action="store_true",
        default=DEFAULT_CONFIG.compare_visibility_coupling,
        help="Emit baseline and visibility-coupled outputs in a single run",
    )
    parser.add_argument("--calibration-samples", type=int, default=DEFAULT_CONFIG.calibration_samples)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.rng_seed)
    args = parser.parse_args(argv)

    return PipelineConfig(
        dem_path=args.dem,
        sar_path=args.sar,
        marks_path=args.marks,
        buildings_path=args.buildings,
        known_paths_path=args.known_paths,
        path_prior_raster=args.path_prior_raster,
        path_prior_mode=args.path_prior_mode,
        visibility_raster=args.visibility_raster,
        visibility_source=args.visibility_source,
        out_dir=args.out,
        samples_per_pair=args.samples,
        max_pairs=args.max_pairs,
        top_k_paths=args.top_k,
        noise_temperature=args.temperature,
        cost_w_slope=args.w_slope,
        cost_w_roughness=args.w_roughness,
        cost_w_surface=args.w_surface,
        cost_w_path_prior=args.w_path_prior,
        cost_w_visibility=args.w_visibility,
        calibrate_weights=args.calibrate_weights,
        compare_visibility_coupling=args.compare_visibility_coupling,
        calibration_samples=args.calibration_samples,
        rng_seed=args.seed,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_args(argv)
    summary = run(config)
    print(json.dumps(summary, indent=2))
    return 0


__all__ = [
    "DEFAULT_CONFIG",
    "PipelineConfig",
    "ScenarioArtifacts",
    "_run_single",
    "align_band_to_reference",
    "build_obstacle_mask",
    "load_terminals",
    "load_visibility_probability",
    "main",
    "parse_args",
    "run",
    "write_comparison_figure",
]
