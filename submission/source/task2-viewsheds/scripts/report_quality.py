#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

import numpy as np
from osgeo import gdal


gdal.UseExceptions()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate run quality report")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--report-path", default="outputs/run_quality_report.md")
    p.add_argument("--metrics-path", default="outputs/viewshed_model_metrics.json")
    return p.parse_args()


def _load(path: Path) -> np.ndarray:
    ds = gdal.Open(str(path), gdal.GA_ReadOnly)
    if ds is None:
        raise FileNotFoundError(path)
    return ds.GetRasterBand(1).ReadAsArray()


def _ratio_from_binary(arr: np.ndarray) -> float:
    v = (arr == 1).sum()
    n = arr.size
    return float(v / n) if n else 0.0


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    required = [
        "viewshed.tif",
        "viewshed.shp",
        "viewshed_probability.tif",
        "viewshed3d.tif",
        "viewshed3d.shp",
        "viewshed3d_probability.tif",
        "viewshed3d_volume.vtk",
        "viewshed_model.npz",
        "viewshed_model_metrics.json",
        "viewshed_model_union.shp",
    ]

    missing = [f for f in required if not (out_dir / f).exists()]

    ratios = {}
    for obs in [1, 2, 3]:
        p = out_dir / f"viewshed_observer_{obs}.tif"
        if p.exists():
            ratios[f"observer_{obs}"] = _ratio_from_binary(_load(p))

    prob_path = out_dir / "viewshed_probability.tif"
    prob_stats = None
    if prob_path.exists():
        prob = _load(prob_path).astype(np.float64)
        prob_stats = {
            "min": float(np.nanmin(prob)),
            "max": float(np.nanmax(prob)),
            "mean": float(np.nanmean(prob)),
        }

    metrics_path = Path(args.metrics_path)
    if not metrics_path.is_absolute():
        metrics_path = Path.cwd() / metrics_path
    model_metrics = None
    if metrics_path.exists():
        model_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

    lines = []
    lines.append("# Run Quality Report")
    lines.append("")
    lines.append(f"Generated: {ts}")
    lines.append("")
    lines.append("## Artifact Check")
    if missing:
        lines.append(f"Status: FAIL ({len(missing)} missing)")
        for m in missing:
            lines.append(f"- missing: `{m}`")
    else:
        lines.append("Status: PASS")

    lines.append("")
    lines.append("## Deterministic Viewshed Ratios")
    if ratios:
        for k, v in ratios.items():
            lines.append(f"- {k}: `{v:.4f}`")
    else:
        lines.append("- unavailable")

    lines.append("")
    lines.append("## Probability Raster Stats")
    if prob_stats is not None:
        lines.append(f"- min: `{prob_stats['min']:.4f}`")
        lines.append(f"- max: `{prob_stats['max']:.4f}`")
        lines.append(f"- mean: `{prob_stats['mean']:.4f}`")
    else:
        lines.append("- unavailable")

    lines.append("")
    lines.append("## Model Metrics")
    if model_metrics is not None:
        val = model_metrics.get("validation", {})
        label_mode = model_metrics.get("label_mode", "unknown")
        best_t = model_metrics.get("validation_best_threshold", None)
        best = model_metrics.get("validation_best_threshold_metrics", {})
        lines.append(f"- label mode: `{label_mode}`")
        lines.append(f"- validation accuracy: `{val.get('accuracy', 0.0):.4f}`")
        lines.append(f"- validation F1 @0.5: `{val.get('f1', 0.0):.4f}`")
        lines.append(f"- validation IoU @0.5: `{val.get('iou', 0.0):.4f}`")
        if best_t is not None:
            lines.append(f"- best threshold: `{best_t:.3f}`")
            lines.append(f"- validation F1 @best: `{best.get('f1', 0.0):.4f}`")
            lines.append(f"- validation IoU @best: `{best.get('iou', 0.0):.4f}`")
    else:
        lines.append("- unavailable")

    lines.append("")
    lines.append("## Readiness")
    if missing:
        lines.append("- Runtime readiness: FAIL")
    else:
        lines.append("- Runtime readiness: PASS")
    lines.append("- Technical scope: baseline complete (2.5D + voxel-3D LOS + trained surrogate)")
    lines.append("- Remaining for full architectural 3D claim: aperture-rich mesh raycasting and calibrated opening dataset")

    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Quality report written: {report_path}")


if __name__ == "__main__":
    main()
