from __future__ import annotations

import json
from pathlib import Path


def build_features(dem, sar, transform):
    import numpy as np
    from preprocessing.dem_processing import compute_slope_norm
    from preprocessing.terrain_features import compute_roughness, robust_normalize
    from scipy import ndimage

    slope = compute_slope_norm(dem, transform)
    roughness = compute_roughness(dem)
    sar_normalized = robust_normalize(sar)
    dem_normalized = robust_normalize(dem)

    sar_fill = np.nan_to_num(sar_normalized, nan=np.nanmedian(sar_normalized))
    dem_fill = np.nan_to_num(dem_normalized, nan=np.nanmedian(dem_normalized))

    grad_sar_y, grad_sar_x = np.gradient(sar_fill)
    grad_dem_y, grad_dem_x = np.gradient(dem_fill)

    grad_sar = robust_normalize(np.sqrt(grad_sar_x**2 + grad_sar_y**2))
    grad_dem = robust_normalize(np.sqrt(grad_dem_x**2 + grad_dem_y**2))

    sar_mean = ndimage.uniform_filter(sar_fill, size=5)
    sar_std = robust_normalize(
        np.sqrt(np.maximum(ndimage.uniform_filter((sar_fill - sar_mean) ** 2, size=5), 0.0))
    )
    laplace = robust_normalize(np.abs(ndimage.laplace(sar_fill)))

    return np.stack(
        [sar_normalized, dem_normalized, slope, roughness, grad_sar, grad_dem, sar_std, laplace],
        axis=-1,
    )


def align_band_to_reference(src_path: Path, ref_transform, ref_crs, ref_shape):
    import numpy as np
    import rasterio
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


def run_diagnostics(
    dem_path: Path,
    sar_path: Path,
    train_paths_path: Path,
    eval_paths_path: Path,
    out_dir: Path,
) -> None:
    import sys

    import geopandas as gpd
    import matplotlib.pyplot as plt
    import numpy as np
    from rasterio.features import rasterize
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import auc, precision_recall_curve

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "shared_utils/src"))
    sys.path.insert(0, str(root / "task1-path-tracing/src"))

    from lamp_core.io import read_raster

    out_dir.mkdir(parents=True, exist_ok=True)

    dem_band = read_raster(str(dem_path))
    sar = align_band_to_reference(
        sar_path,
        ref_transform=dem_band.transform,
        ref_crs=dem_band.crs,
        ref_shape=dem_band.data.shape,
    )
    features = build_features(dem_band.data, sar, dem_band.transform)

    def get_mask(path_file: Path):
        frame = gpd.read_file(path_file)
        if frame.crs != dem_band.crs:
            frame = frame.to_crs(dem_band.crs)
        geoms = [geometry.buffer(2.25) for geometry in frame.geometry if geometry is not None]
        return rasterize([(geometry, 1) for geometry in geoms], out_shape=dem_band.data.shape, transform=dem_band.transform)

    train_mask = get_mask(train_paths_path)
    eval_mask = get_mask(eval_paths_path)
    valid = np.isfinite(dem_band.data)

    pos_idx = np.argwhere((train_mask == 1) & valid)
    neg_idx = np.argwhere((train_mask == 0) & valid)
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        raise ValueError("Training masks produced empty positive or negative samples")

    neg_sample_size = min(len(neg_idx), len(pos_idx) * 4)
    selected_neg_idx = neg_idx[np.random.choice(len(neg_idx), size=neg_sample_size, replace=False)]
    train_idx = np.vstack([pos_idx, selected_neg_idx])

    x_train = features[train_idx[:, 0], train_idx[:, 1]]
    y_train = np.hstack([np.ones(len(pos_idx)), np.zeros(len(selected_neg_idx))])

    classifier = RandomForestClassifier(n_estimators=100, random_state=11, n_jobs=-1)
    classifier.fit(np.nan_to_num(x_train), y_train)

    feature_names = ["SAR", "DEM", "Slope", "Roughness", "Grad_SAR", "Grad_DEM", "SAR_STD", "Laplace"]
    plt.figure(figsize=(10, 6))
    plt.barh(feature_names, classifier.feature_importances_)
    plt.title("Feature Importance")
    plt.savefig(out_dir / "feature_importance.png")
    plt.close()

    eval_idx = np.argwhere(valid)
    x_eval = features[eval_idx[:, 0], eval_idx[:, 1]]
    y_eval = eval_mask[eval_idx[:, 0], eval_idx[:, 1]]
    y_scores = classifier.predict_proba(np.nan_to_num(x_eval))[:, 1]

    precision, recall, _ = precision_recall_curve(y_eval, y_scores)
    pr_auc = auc(recall, precision)

    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label=f"AUC={pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.savefig(out_dir / "pr_curve.png")
    plt.close()

    midpoint = dem_band.data.shape[0] // 2
    x_top = features[:midpoint][valid[:midpoint]]
    y_top = eval_mask[:midpoint][valid[:midpoint]]
    x_bottom = features[midpoint:][valid[midpoint:]]
    y_bottom = eval_mask[midpoint:][valid[midpoint:]]

    classifier_top = RandomForestClassifier(n_estimators=50, random_state=11)
    classifier_top.fit(np.nan_to_num(x_top[::100]), y_top[::100])
    score_bottom = classifier_top.score(np.nan_to_num(x_bottom[::100]), y_bottom[::100])

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as file_handle:
        json.dump(
            {
                "feature_importance": dict(zip(feature_names, classifier.feature_importances_.tolist())),
                "pr_auc": pr_auc,
                "spatial_cv_score": score_bottom,
            },
            file_handle,
            indent=2,
        )
