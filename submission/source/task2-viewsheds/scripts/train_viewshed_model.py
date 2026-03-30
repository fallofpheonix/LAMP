#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
from osgeo import gdal

from src.buildings import rasterize_building_heights
from src.load_data import load_dem, load_observers
from src.ml_features import labels_to_flat, observer_feature_matrix
from src.ml_model import best_threshold, binary_metrics, train_logistic_model
from src.raycast import compute_viewshed
from src.scene import build_occlusion_surface
from src.terrain import inside, world_to_pixel
from src.voxel_scene import build_voxel_scene, compute_ground_viewshed_3d


gdal.UseExceptions()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train NumPy logistic model for viewshed projection")
    p.add_argument("--data-dir", default="data", help="Input directory")
    p.add_argument("--output-dir", default="outputs", help="Output directory")
    p.add_argument("--scene-mode", default="fused", choices=["provided", "synthetic", "fused"])
    p.add_argument("--observer-height", type=float, default=1.6)
    p.add_argument("--target-height", type=float, default=0.0)
    p.add_argument("--max-distance", type=float, default=None)
    p.add_argument("--label-mode", default="3d", choices=["2d", "3d"], help="Label generator mode")
    p.add_argument("--z-res", type=float, default=0.5, help="Voxel Z resolution when label-mode=3d")
    p.add_argument("--synthetic-observers", type=int, default=60, help="Random observers for training")
    p.add_argument("--samples-per-observer", type=int, default=2200, help="Max sampled cells per observer")
    p.add_argument("--epochs", type=int, default=350)
    p.add_argument("--learning-rate", type=float, default=0.04)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--model-path", default="outputs/viewshed_model.npz")
    p.add_argument("--metrics-path", default="outputs/viewshed_model_metrics.json")
    return p.parse_args()


def _sample_rows_cols(valid_mask: np.ndarray, k: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    rows, cols = np.where(valid_mask)
    n = len(rows)
    if n == 0:
        return []
    if k >= n:
        return [(int(r), int(c)) for r, c in zip(rows, cols)]
    idx = rng.choice(n, size=k, replace=False)
    return [(int(rows[i]), int(cols[i])) for i in idx]


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

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
    scene3d = None
    if args.label_mode == "3d":
        scene3d = build_voxel_scene(
            terrain=dem_original.array,
            building_heights=building_heights,
            geotransform=dem_original.geotransform,
            projection_wkt=dem_original.projection_wkt,
            nodata=dem_original.nodata,
            z_res=args.z_res,
        )

    pixel_size_m = float(abs(dem_original.geotransform[1]))
    valid = np.isfinite(surface)

    # Validation observers: provided project marks.
    val_observers = []
    marks = load_observers(str(data_dir / "Marks_Brief2.shp"), dem_original.projection_wkt)
    for obs in marks:
        row, col = world_to_pixel(dem_original.geotransform, obs["x"], obs["y"])
        if inside(surface, row, col):
            val_observers.append({**obs, "row": row, "col": col})

    if not val_observers:
        raise RuntimeError("No valid project observer marks found for validation")

    # Training observers: random valid cells excluding validation coordinates.
    forbidden = {(o["row"], o["col"]) for o in val_observers}
    candidates = [(r, c) for r, c in _sample_rows_cols(valid, 10_000, rng) if (r, c) not in forbidden]
    if len(candidates) < args.synthetic_observers:
        raise RuntimeError("Not enough candidate observer cells for training")

    train_observers = []
    picked = rng.choice(len(candidates), size=args.synthetic_observers, replace=False)
    for i, idx in enumerate(picked, start=1):
        r, c = candidates[int(idx)]
        train_observers.append({"id": 10_000 + i, "row": int(r), "col": int(c), "x": 0.0, "y": 0.0})

    X_train_parts = []
    y_train_parts = []

    for obs in train_observers:
        if args.label_mode == "3d":
            y_raster, _ = compute_ground_viewshed_3d(
                scene=scene3d,
                obs_row=obs["row"],
                obs_col=obs["col"],
                observer_height=args.observer_height,
                target_height=args.target_height,
                max_distance_m=args.max_distance,
            )
        else:
            y_raster, _ = compute_viewshed(
                surface=surface,
                nodata=dem_original.nodata,
                obs_row=obs["row"],
                obs_col=obs["col"],
                observer_height=args.observer_height,
                target_height=args.target_height,
                pixel_size_m=pixel_size_m,
                max_distance_m=args.max_distance,
            )
        X, valid_mask = observer_feature_matrix(
            surface=surface,
            obs_row=obs["row"],
            obs_col=obs["col"],
            observer_height=args.observer_height,
            target_height=args.target_height,
            nodata=dem_original.nodata,
            max_distance_m=args.max_distance,
            pixel_size_m=pixel_size_m,
        )
        y = labels_to_flat(y_raster, valid_mask)

        if args.samples_per_observer > 0 and len(y) > args.samples_per_observer:
            idx = rng.choice(len(y), size=args.samples_per_observer, replace=False)
            X = X[idx]
            y = y[idx]

        X_train_parts.append(X)
        y_train_parts.append(y)

    X_train = np.vstack(X_train_parts)
    y_train = np.concatenate(y_train_parts)

    # Validation set from the real observer marks (all cells).
    X_val_parts = []
    y_val_parts = []
    for obs in val_observers:
        if args.label_mode == "3d":
            y_raster, _ = compute_ground_viewshed_3d(
                scene=scene3d,
                obs_row=obs["row"],
                obs_col=obs["col"],
                observer_height=args.observer_height,
                target_height=args.target_height,
                max_distance_m=args.max_distance,
            )
        else:
            y_raster, _ = compute_viewshed(
                surface=surface,
                nodata=dem_original.nodata,
                obs_row=obs["row"],
                obs_col=obs["col"],
                observer_height=args.observer_height,
                target_height=args.target_height,
                pixel_size_m=pixel_size_m,
                max_distance_m=args.max_distance,
            )
        X, valid_mask = observer_feature_matrix(
            surface=surface,
            obs_row=obs["row"],
            obs_col=obs["col"],
            observer_height=args.observer_height,
            target_height=args.target_height,
            nodata=dem_original.nodata,
            max_distance_m=args.max_distance,
            pixel_size_m=pixel_size_m,
        )
        y = labels_to_flat(y_raster, valid_mask)
        X_val_parts.append(X)
        y_val_parts.append(y)

    X_val = np.vstack(X_val_parts)
    y_val = np.concatenate(y_val_parts)

    model, train_metrics, val_metrics = train_logistic_model(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
    )

    model_path = Path(args.model_path)
    if not model_path.is_absolute():
        model_path = Path.cwd() / model_path
    model.save(str(model_path))

    y_val_prob = model.predict_proba(X_val)
    opt_threshold, opt_val_metrics = best_threshold(y_val, y_val_prob)
    opt_train_metrics = binary_metrics(y_train, model.predict(X_train, threshold=opt_threshold))

    metrics = {
        "train": train_metrics,
        "validation": val_metrics,
        "validation_best_threshold": opt_threshold,
        "validation_best_threshold_metrics": opt_val_metrics,
        "train_at_best_threshold": opt_train_metrics,
        "train_observers": len(train_observers),
        "validation_observers": len(val_observers),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_val)),
        "scene_mode": args.scene_mode,
        "observer_height": args.observer_height,
        "target_height": args.target_height,
        "max_distance": args.max_distance,
        "seed": args.seed,
        "label_mode": args.label_mode,
        "z_res": args.z_res,
    }

    metrics_path = Path(args.metrics_path)
    if not metrics_path.is_absolute():
        metrics_path = Path.cwd() / metrics_path
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Model saved: {model_path}")
    print(f"Metrics saved: {metrics_path}")
    print(
        "Validation metrics: "
        f"acc={val_metrics['accuracy']:.4f} "
        f"precision={val_metrics['precision']:.4f} "
        f"recall={val_metrics['recall']:.4f} "
        f"f1={val_metrics['f1']:.4f} "
        f"iou={val_metrics['iou']:.4f}"
    )
    print(
        "Validation best threshold: "
        f"t={opt_threshold:.3f} "
        f"f1={opt_val_metrics['f1']:.4f} "
        f"iou={opt_val_metrics['iou']:.4f}"
    )


if __name__ == "__main__":
    main()
