#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Task 1 acceptance check")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--report-path", default="outputs/task1_completion.md")
    p.add_argument("--run-summary", default="outputs/run_summary.json")
    p.add_argument("--preprocess-report", default="outputs/preprocess_report.json")
    p.add_argument("--prior-report", default="outputs/prior_training_report.json")
    p.add_argument("--min-topk-recall", type=float, default=0.50)
    p.add_argument("--min-f1", type=float, default=0.30)
    p.add_argument("--min-prior-iou", type=float, default=0.10)
    p.add_argument("--min-path-features", type=int, default=1)
    p.add_argument("--min-lost-polygons", type=int, default=1)
    return p.parse_args()


def status(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def load_json(path_s: str) -> dict:
    p = Path(path_s)
    if not p.is_absolute():
        p = Path.cwd() / p
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    out = Path(args.output_dir)

    required = [
        "predicted_paths.gpkg",
        "predicted_paths.geojson",
        "predicted_centerlines.geojson",
        "lost_path_candidates.geojson",
        "probability_heatmap.tif",
        "movement_cost.tif",
        "run_summary.json",
        "preprocess_report.json",
        "prior_training_report.json",
    ]
    optional = [
        "calibration_report.json",
        "predicted_topk_mask.tif",
    ]

    missing_req = [f for f in required if not (out / f).exists()]
    missing_opt = [f for f in optional if not (out / f).exists()]

    summary = load_json(args.run_summary)
    preprocess = load_json(args.preprocess_report)
    prior = load_json(args.prior_report)

    processed_pairs = int(summary.get("processed_pairs", 0))
    successful_pairs = int(summary.get("successful_pairs", 0))
    path_features = int(summary.get("path_features", 0))
    lost_polygons = int(summary.get("lost_path_polygons", 0))

    topk_recall = float(summary.get("topk_recall", 0.0))
    f1 = float(summary.get("f1", 0.0))

    prior_mode = str(preprocess.get("path_prior", {}).get("mode", "unknown"))
    prior_iou = float(prior.get("eval_metrics_at_best_threshold", {}).get("iou", 0.0))

    checks = []
    checks.append(("Required artifacts", len(missing_req) == 0, f"missing={missing_req}"))
    checks.append(("Learned-prior mode", prior_mode == "learned", f"mode={prior_mode}"))
    checks.append(
        (
            "Terminal-pair solve coverage",
            processed_pairs > 0 and successful_pairs == processed_pairs,
            f"processed={processed_pairs}, successful={successful_pairs}",
        )
    )
    checks.append(
        (
            "Vector output non-empty",
            path_features >= args.min_path_features,
            f"path_features={path_features}, min={args.min_path_features}",
        )
    )
    checks.append(
        (
            "Lost-path candidates non-empty",
            lost_polygons >= args.min_lost_polygons,
            f"lost_polygons={lost_polygons}, min={args.min_lost_polygons}",
        )
    )
    checks.append(
        ("Top-k recall", topk_recall >= args.min_topk_recall, f"value={topk_recall:.4f}, min={args.min_topk_recall:.4f}")
    )
    checks.append(("Path F1", f1 >= args.min_f1, f"value={f1:.4f}, min={args.min_f1:.4f}"))
    checks.append(
        (
            "Prior IoU",
            prior_iou >= args.min_prior_iou,
            f"value={prior_iou:.4f}, min={args.min_prior_iou:.4f}",
        )
    )
    checks.append(("Optional artifacts", len(missing_opt) == 0, f"missing={missing_opt}"))

    required_pass = all(ok for name, ok, _ in checks if name != "Optional artifacts")
    overall = required_pass and (len(missing_opt) == 0)

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = []
    lines.append("# Task 1 Completion Check")
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
