#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 2 acceptance check")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--report-path", default="outputs/task2_completion.md")
    p.add_argument("--metrics-path", default="outputs/viewshed_model_metrics.json")
    p.add_argument("--min-f1-best", type=float, default=0.50)
    p.add_argument("--min-iou-best", type=float, default=0.33)
    return p.parse_args()


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)

    required = [
        "viewshed.tif",
        "viewshed.shp",
        "viewshed3d.tif",
        "viewshed3d.shp",
        "viewshed_probability.tif",
        "viewshed3d_probability.tif",
        "viewshed_model.npz",
        "viewshed_model_metrics.json",
        "viewshed_model_union.shp",
    ]

    optional = [
        "viewshed3d_volume.vtk",
    ]

    missing_req = [f for f in required if not (out / f).exists()]
    missing_opt = [f for f in optional if not (out / f).exists()]

    metrics_path = Path(args.metrics_path)
    if not metrics_path.is_absolute():
        metrics_path = Path.cwd() / metrics_path
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    best = metrics.get("validation_best_threshold_metrics", {})
    best_f1 = float(best.get("f1", 0.0))
    best_iou = float(best.get("iou", 0.0))
    label_mode = str(metrics.get("label_mode", "unknown"))

    checks = []
    checks.append(("Required artifacts", len(missing_req) == 0, f"missing={missing_req}"))
    checks.append(("3D label mode", label_mode == "3d", f"label_mode={label_mode}"))
    checks.append(("Model F1(best)", best_f1 >= args.min_f1_best, f"value={best_f1:.4f}, min={args.min_f1_best:.4f}"))
    checks.append(("Model IoU(best)", best_iou >= args.min_iou_best, f"value={best_iou:.4f}, min={args.min_iou_best:.4f}"))
    checks.append(("Optional volume artifact", len(missing_opt) == 0, f"missing={missing_opt}"))

    required_pass = all(ok for name, ok, _ in checks if name != "Optional volume artifact")
    overall = required_pass and (len(missing_opt) == 0)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    lines.append("# Task 2 Completion Check")
    lines.append("")
    lines.append(f"Generated: {ts}")
    lines.append("")
    lines.append("## Checks")
    for name, ok, msg in checks:
        lines.append(f"- {name}: **{status(ok)}** ({msg})")

    lines.append("")
    lines.append("## Result")
    lines.append(f"- Required scope: **{status(required_pass)}**")
    lines.append(f"- Required + optional scope: **{status(overall)}**")

    report_path = Path(args.report_path)
    if not report_path.is_absolute():
        report_path = Path.cwd() / report_path
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Completion report: {report_path}")
    print(f"Required scope: {status(required_pass)}")
    print(f"Required + optional scope: {status(overall)}")


if __name__ == "__main__":
    main()
