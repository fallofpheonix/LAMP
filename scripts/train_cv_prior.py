#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize
from rasterio.warp import Resampling, reproject
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lamp.tasks.path_tracing.preprocessing.dem_processing import compute_slope_norm, read_raster
from lamp.tasks.path_tracing.preprocessing.terrain_features import compute_roughness, robust_normalize


def align_band_to_reference(src_path: Path, ref_transform, ref_crs, ref_shape: tuple[int, int]) -> np.ndarray:
    with rasterio.open(src_path) as src:
        src_arr = src.read(1).astype(np.float32)
        dst = np.empty(ref_shape, dtype=np.float32)
        reproject(
            source=src_arr,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=ref_transform,
            dst_crs=ref_crs,
            resampling=Resampling.bilinear,
        )
        if src.nodata is not None:
            dst = np.where(np.isclose(dst, src.nodata), np.nan, dst)
    return dst


def rasterize_paths(path_file: Path, crs, shape: tuple[int, int], transform, buffer_m: float) -> np.ndarray:
    gdf = gpd.read_file(path_file)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    geoms = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if buffer_m > 0:
        geoms = [geom.buffer(buffer_m) for geom in geoms]
    if not geoms:
        return np.zeros(shape, dtype=np.uint8)
    return rasterize(
        [(geom, 1) for geom in geoms],
        out_shape=shape,
        transform=transform,
        fill=0,
        default_value=1,
        dtype="uint8",
    )


def build_features(dem: np.ndarray, sar: np.ndarray, transform) -> np.ndarray:
    slope = compute_slope_norm(dem, transform)
    rough = compute_roughness(dem)
    sar_n = robust_normalize(sar)
    dem_n = robust_normalize(dem)

    sar_fill = np.nan_to_num(sar_n, nan=np.nanmedian(sar_n))
    dem_fill = np.nan_to_num(dem_n, nan=np.nanmedian(dem_n))

    gy_s, gx_s = np.gradient(sar_fill)
    grad_s = robust_normalize(np.sqrt(gx_s * gx_s + gy_s * gy_s))

    gy_d, gx_d = np.gradient(dem_fill)
    grad_d = robust_normalize(np.sqrt(gx_d * gx_d + gy_d * gy_d))

    sar_mean = ndimage.uniform_filter(sar_fill, size=5)
    sar_var = ndimage.uniform_filter((sar_fill - sar_mean) ** 2, size=5)
    sar_std = robust_normalize(np.sqrt(np.maximum(sar_var, 0.0)))

    lap = robust_normalize(np.abs(ndimage.laplace(sar_fill)))

    feats = np.stack(
        [
            np.nan_to_num(sar_n, nan=0.0),
            np.nan_to_num(dem_n, nan=0.0),
            np.nan_to_num(slope, nan=0.0),
            np.nan_to_num(rough, nan=0.0),
            np.nan_to_num(grad_s, nan=0.0),
            np.nan_to_num(grad_d, nan=0.0),
            np.nan_to_num(sar_std, nan=0.0),
            np.nan_to_num(lap, nan=0.0),
        ],
        axis=-1,
    ).astype(np.float32)
    return feats


def metrics_from_masks(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    tp = int(np.logical_and(pred_b, gt_b).sum())
    fp = int(np.logical_and(pred_b, ~gt_b).sum())
    fn = int(np.logical_and(~pred_b, gt_b).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "iou": iou, "f1": f1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train a classical CV prior model and export path_prior_prob.tif")
    p.add_argument("--dem", type=Path, default=ROOT / "data" / "task1" / "DEM_Subset-Original.tif")
    p.add_argument("--sar", type=Path, default=ROOT / "data" / "task1" / "SAR-MS.tif")
    p.add_argument("--train-paths", type=Path, default=ROOT / "data" / "task1" / "known_paths_train.shp")
    p.add_argument("--eval-paths", type=Path, default=ROOT / "data" / "task1" / "known_paths_eval.shp")
    p.add_argument("--out-prior", type=Path, default=ROOT / "data" / "task1" / "path_prior_prob.tif")
    p.add_argument("--out-report", type=Path, default=Path("outputs/prior_training_report.json"))
    p.add_argument("--out-eval-mask", type=Path, default=Path("outputs/prior_eval_mask.tif"))
    p.add_argument("--buffer-m", type=float, default=2.25)
    p.add_argument("--neg-pos-ratio", type=float, default=4.0)
    p.add_argument("--n-estimators", type=int, default=500)
    p.add_argument("--seed", type=int, default=11)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)

    dem_bundle = read_raster(args.dem)
    dem = dem_bundle.data
    sar = align_band_to_reference(
        args.sar,
        ref_transform=dem_bundle.transform,
        ref_crs=dem_bundle.crs,
        ref_shape=(dem_bundle.profile["height"], dem_bundle.profile["width"]),
    )
    feats = build_features(dem, sar, dem_bundle.transform)
    valid = np.isfinite(dem)

    train_mask = rasterize_paths(
        args.train_paths,
        crs=dem_bundle.crs,
        shape=(dem_bundle.profile["height"], dem_bundle.profile["width"]),
        transform=dem_bundle.transform,
        buffer_m=args.buffer_m,
    )
    eval_mask = rasterize_paths(
        args.eval_paths,
        crs=dem_bundle.crs,
        shape=(dem_bundle.profile["height"], dem_bundle.profile["width"]),
        transform=dem_bundle.transform,
        buffer_m=args.buffer_m,
    )

    pos_idx = np.argwhere((train_mask == 1) & valid)
    neg_idx = np.argwhere((train_mask == 0) & valid)
    if len(pos_idx) == 0:
        raise RuntimeError("No positive pixels in train mask. Increase --buffer-m or check train paths.")

    n_neg = min(len(neg_idx), int(max(1, round(len(pos_idx) * args.neg_pos_ratio))))
    sel_neg = neg_idx[np.random.choice(len(neg_idx), size=n_neg, replace=False)]
    train_idx = np.vstack([pos_idx, sel_neg])
    y = np.hstack([np.ones(len(pos_idx), dtype=np.uint8), np.zeros(len(sel_neg), dtype=np.uint8)])

    np.random.shuffle(train_idx)
    X = feats[train_idx[:, 0], train_idx[:, 1], :]

    clf = RandomForestClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
        min_samples_leaf=2,
    )
    clf.fit(X, y)

    prior = np.zeros(valid.shape, dtype=np.float32)
    all_X = feats[valid]
    prior[valid] = clf.predict_proba(all_X)[:, 1].astype(np.float32)

    thresholds = np.linspace(0.1, 0.9, 17)
    best_thr = 0.5
    best = {"iou": -1.0}
    for thr in thresholds:
        m = metrics_from_masks((prior >= thr).astype(np.uint8), eval_mask)
        if m["iou"] > best["iou"]:
            best = m
            best_thr = float(thr)

    args.out_prior.parent.mkdir(parents=True, exist_ok=True)
    out_profile = dem_bundle.profile.copy()
    out_profile.update(dtype="float32", count=1, compress="lzw", nodata=0.0)
    with rasterio.open(args.out_prior, "w", **out_profile) as dst:
        dst.write(prior, 1)

    args.out_eval_mask.parent.mkdir(parents=True, exist_ok=True)
    eval_profile = dem_bundle.profile.copy()
    eval_profile.update(dtype="float32", count=1, compress="lzw", nodata=0.0)
    with rasterio.open(args.out_eval_mask, "w", **eval_profile) as dst:
        dst.write(eval_mask.astype(np.float32), 1)

    report = {
        "train_path_file": str(args.train_paths),
        "eval_path_file": str(args.eval_paths),
        "buffer_m": args.buffer_m,
        "neg_pos_ratio": args.neg_pos_ratio,
        "n_estimators": args.n_estimators,
        "seed": args.seed,
        "train_positive_pixels": int(len(pos_idx)),
        "train_negative_pixels_sampled": int(len(sel_neg)),
        "best_threshold_eval": best_thr,
        "eval_metrics_at_best_threshold": best,
    }
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
