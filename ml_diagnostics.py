#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, confusion_matrix, auc
import geopandas as gpd
from rasterio.features import rasterize

# Ensure lamp_core is available
sys.path.insert(0, str(Path.cwd() / "shared_utils/src"))
sys.path.insert(0, str(Path.cwd() / "task1-path-tracing/src"))

from lamp_core.io import read_raster

def build_features(dem, sar, transform):
    from preprocessing.dem_processing import compute_slope_norm
    from preprocessing.terrain_features import compute_roughness, robust_normalize
    from scipy import ndimage
    
    slope = compute_slope_norm(dem, transform)
    rough = compute_roughness(dem)
    sar_n = robust_normalize(sar)
    dem_n = robust_normalize(dem)
    sar_fill = np.nan_to_num(sar_n, nan=np.nanmedian(sar_n))
    dem_fill = np.nan_to_num(dem_n, nan=np.nanmedian(dem_n))
    gy_s, gx_s = np.gradient(sar_fill)
    grad_s = robust_normalize(np.sqrt(gx_s**2 + gy_s**2))
    gy_d, gx_d = np.gradient(dem_fill)
    grad_d = robust_normalize(np.sqrt(gx_d**2 + gy_d**2))
    sar_mean = ndimage.uniform_filter(sar_fill, size=5)
    sar_std = robust_normalize(np.sqrt(np.maximum(ndimage.uniform_filter((sar_fill - sar_mean)**2, size=5), 0.0)))
    lap = robust_normalize(np.abs(ndimage.laplace(sar_fill)))
    
    return np.stack([sar_n, dem_n, slope, rough, grad_s, grad_d, sar_std, lap], axis=-1)

def align_band_to_reference(src_path: Path, ref_transform, ref_crs, ref_shape: tuple[int, int]) -> np.ndarray:
    from rasterio.warp import Resampling, reproject
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dem", default="task1-path-tracing/Task_1/DEM_Subset-Original.tif")
    parser.add_argument("--sar", default="task1-path-tracing/Task_1/SAR-MS.tif")
    parser.add_argument("--paths", default="task1-path-tracing/known_paths_train.shp")
    parser.add_argument("--eval-paths", default="task1-path-tracing/known_paths_eval.shp")
    parser.add_argument("--out-dir", default="outputs/diagnostics")
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    dem_b = read_raster(args.dem)
    sar = align_band_to_reference(
        args.sar,
        ref_transform=dem_b.transform,
        ref_crs=dem_b.crs,
        ref_shape=dem_b.data.shape,
    )
    feats = build_features(dem_b.data, sar, dem_b.transform)
    
    # Rasterize paths
    def get_mask(path_file):
        gdf = gpd.read_file(path_file)
        if gdf.crs != dem_b.crs: gdf = gdf.to_crs(dem_b.crs)
        geoms = [g.buffer(2.25) for g in gdf.geometry if g is not None]
        return rasterize([(g, 1) for g in geoms], out_shape=dem_b.data.shape, transform=dem_b.transform)

    train_mask = get_mask(args.paths)
    eval_mask = get_mask(args.eval_paths)
    valid = np.isfinite(dem_b.data)
    
    # Training RF
    pos_idx = np.argwhere((train_mask == 1) & valid)
    neg_idx = np.argwhere((train_mask == 0) & valid)
    sel_neg = neg_idx[np.random.choice(len(neg_idx), size=len(pos_idx)*4, replace=False)]
    train_idx = np.vstack([pos_idx, sel_neg])
    X = feats[train_idx[:,0], train_idx[:,1]]
    y = np.hstack([np.ones(len(pos_idx)), np.zeros(len(sel_neg))])
    
    clf = RandomForestClassifier(n_estimators=100, random_state=11, n_jobs=-1)
    clf.fit(np.nan_to_num(X), y)
    
    # Feature Importance
    f_names = ["SAR", "DEM", "Slope", "Roughness", "Grad_SAR", "Grad_DEM", "SAR_STD", "Laplace"]
    plt.figure(figsize=(10,6))
    plt.barh(f_names, clf.feature_importances_)
    plt.title("Feature Importance")
    plt.savefig(out_dir / "feature_importance.png")
    
    # Precision-Recall
    eval_idx = np.argwhere(valid)
    X_eval = feats[eval_idx[:,0], eval_idx[:,1]]
    y_eval = eval_mask[eval_idx[:,0], eval_idx[:,1]]
    y_scores = clf.predict_proba(np.nan_to_num(X_eval))[:, 1]
    
    precision, recall, _ = precision_recall_curve(y_eval, y_scores)
    plt.figure(figsize=(10,6))
    plt.plot(recall, precision, label=f"AUC={auc(recall, precision):.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(out_dir / "pr_curve.png")
    
    # Spatial Cross-Validation (Split top/bottom)
    mid_row = dem_b.data.shape[0] // 2
    X_top = feats[:mid_row][valid[:mid_row]]
    y_top = eval_mask[:mid_row][valid[:mid_row]]
    X_bot = feats[mid_row:][valid[mid_row:]]
    y_bot = eval_mask[mid_row:][valid[mid_row:]]
    
    clf_top = RandomForestClassifier(n_estimators=50, random_state=11).fit(np.nan_to_num(X_top[::100]), y_top[::100])
    score_bot = clf_top.score(np.nan_to_num(X_bot[::100]), y_bot[::100])
    
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "feature_importance": dict(zip(f_names, clf.feature_importances_.tolist())),
            "pr_auc": auc(recall, precision),
            "spatial_cv_score": score_bot
        }, f, indent=2)

if __name__ == "__main__": main()
