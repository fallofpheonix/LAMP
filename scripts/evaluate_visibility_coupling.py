#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lamp.config import DEFAULT_CONFIG, PipelineConfig
from lamp.tasks.path_tracing.preprocessing.dem_processing import read_raster
from run_path_tracing import _run_single, load_visibility_probability


def parse_visibility_values(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not (0.0 <= value <= 0.95):
            raise ValueError(f"visibility weight must be in [0.0, 0.95], got {value}")
        values.append(value)
    if not values:
        raise ValueError("at least one visibility weight is required")
    uniq = sorted({round(v, 4) for v in values})
    return uniq


def scale_weights_for_visibility(
    base_core_weights: tuple[float, float, float, float],
    visibility_weight: float,
) -> tuple[float, float, float, float, float]:
    core = np.asarray(base_core_weights, dtype=np.float32)
    if core.shape != (4,):
        raise ValueError("base_core_weights must be a 4-tuple")
    core = np.clip(core, 0.0, None)
    core_sum = float(core.sum())
    if core_sum <= 0.0:
        raise ValueError("base core weights must have positive sum")
    visibility_weight = float(np.clip(visibility_weight, 0.0, 0.95))
    scaled_core = core / core_sum * (1.0 - visibility_weight)
    return (
        float(scaled_core[0]),
        float(scaled_core[1]),
        float(scaled_core[2]),
        float(scaled_core[3]),
        visibility_weight,
    )


def slugify_visibility_weight(value: float) -> str:
    return f"w_visibility_{value:.2f}".replace(".", "p")


def write_sweep_figure(path: Path, results: list[dict]) -> bool:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    weights = [float(r["visibility_weight"]) for r in results]
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    if all("topk_recall" in r for r in results):
        ax.plot(weights, [float(r["topk_recall"]) for r in results], marker="o", label="top-k recall")
    if all("iou" in r for r in results):
        ax.plot(weights, [float(r["iou"]) for r in results], marker="o", label="IoU")
    if all("f1" in r for r in results):
        ax.plot(weights, [float(r["f1"]) for r in results], marker="o", label="F1")
    if not ax.lines:
        ax.plot(weights, [float(r["mean_abs_density_delta"]) for r in results], marker="o", label="mean |density delta|")

    ax.set_xlabel("visibility weight")
    ax.set_ylabel("metric")
    ax.set_title("Visibility Coupling Sweep")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _score_result(result: dict) -> float | None:
    if "iou" not in result or "topk_recall" not in result:
        return None
    return 0.65 * float(result["iou"]) + 0.35 * float(result["topk_recall"])


def _write_markdown_report(path: Path, summary: dict) -> None:
    lines: list[str] = []
    lines.append("# Visibility Coupling Evaluation")
    lines.append("")
    lines.append("## Baseline")
    baseline = summary["baseline"]
    lines.append(f"- output_dir: `{baseline['output_dir']}`")
    lines.append(f"- selected_visibility_weight: `{baseline['selected_weights']['visibility']:.4f}`")
    if "topk_recall" in baseline:
        lines.append(f"- topk_recall: `{baseline['topk_recall']:.4f}`")
        lines.append(f"- iou: `{baseline['iou']:.4f}`")
        lines.append(f"- f1: `{baseline['f1']:.4f}`")
    lines.append("")
    lines.append("## Sweep")
    lines.append("")
    lines.append("| visibility_weight | topk_recall | iou | f1 | mean_abs_density_delta | output_dir |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for result in summary["runs"]:
        lines.append(
            "| "
            f"{result['visibility_weight']:.2f} | "
            f"{result.get('topk_recall', float('nan')):.4f} | "
            f"{result.get('iou', float('nan')):.4f} | "
            f"{result.get('f1', float('nan')):.4f} | "
            f"{result['mean_abs_density_delta']:.6f} | "
            f"`{result['output_dir']}` |"
        )

    best = summary.get("best_run")
    if best is not None:
        lines.append("")
        lines.append("## Best Run")
        lines.append(f"- visibility_weight: `{best['visibility_weight']:.2f}`")
        lines.append(f"- score: `{best['score']:.6f}`")
        lines.append(f"- output_dir: `{best['output_dir']}`")

    lines.append("")
    lines.append("## Artifacts")
    lines.append(f"- summary_json: `{summary['summary_json']}`")
    lines.append(f"- table_csv: `{summary['table_csv']}`")
    lines.append(f"- figure_written: `{summary['figure_written']}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_visibility_coupling(
    config: PipelineConfig,
    *,
    visibility_weights: list[float],
    output_dir: Path,
) -> dict:
    if config.visibility_raster is None:
        raise RuntimeError("visibility evaluation requires --visibility-raster")

    output_dir.mkdir(parents=True, exist_ok=True)

    dem_bundle = read_raster(config.dem_path)
    reference_profile = {
        "height": dem_bundle.profile["height"],
        "width": dem_bundle.profile["width"],
        "transform": dem_bundle.transform,
        "crs": dem_bundle.crs,
    }
    visibility_probability, visibility_alignment = load_visibility_probability(config.visibility_raster, reference_profile)
    if visibility_probability is None:
        raise RuntimeError("failed to load visibility raster")

    base_core = (
        config.cost_w_slope,
        config.cost_w_roughness,
        config.cost_w_surface,
        config.cost_w_path_prior,
    )
    baseline_weights = scale_weights_for_visibility(base_core, 0.0)
    baseline_config = replace(
        config,
        out_dir=output_dir / "baseline",
        cost_w_slope=baseline_weights[0],
        cost_w_roughness=baseline_weights[1],
        cost_w_surface=baseline_weights[2],
        cost_w_path_prior=baseline_weights[3],
        cost_w_visibility=0.0,
    )
    baseline = _run_single(
        baseline_config,
        scenario_name="baseline",
        scenario_out_dir=baseline_config.out_dir,
        visibility_probability=None,
        visibility_alignment=None,
    )

    rows: list[dict] = []
    for visibility_weight in visibility_weights:
        scenario_weights = scale_weights_for_visibility(base_core, visibility_weight)
        scenario_dir = output_dir / slugify_visibility_weight(visibility_weight)
        scenario_config = replace(
            config,
            out_dir=scenario_dir,
            cost_w_slope=scenario_weights[0],
            cost_w_roughness=scenario_weights[1],
            cost_w_surface=scenario_weights[2],
            cost_w_path_prior=scenario_weights[3],
            cost_w_visibility=scenario_weights[4],
        )
        scenario = _run_single(
            scenario_config,
            scenario_name=f"visibility_weight_{visibility_weight:.2f}",
            scenario_out_dir=scenario_dir,
            visibility_probability=visibility_probability if visibility_weight > 0.0 else None,
            visibility_alignment=visibility_alignment if visibility_weight > 0.0 else None,
        )

        density_delta = scenario.density - baseline.density
        row = {
            "visibility_weight": float(visibility_weight),
            "output_dir": str(scenario.output_dir),
            "mean_abs_density_delta": float(np.nanmean(np.abs(density_delta))),
            "rmse_density_delta": float(np.sqrt(np.nanmean(np.square(density_delta)))),
            "selected_weights": scenario.summary["selected_weights"],
        }
        for metric in ("topk_recall", "iou", "precision", "f1"):
            if metric in scenario.summary:
                row[metric] = float(scenario.summary[metric])
                row[f"{metric}_delta_vs_baseline"] = float(scenario.summary[metric]) - float(baseline.summary.get(metric, 0.0))
        score = _score_result(row)
        if score is not None:
            row["score"] = float(score)
        rows.append(row)

    best_run = None
    scored_rows = [row for row in rows if "score" in row]
    if scored_rows:
        best_run = max(scored_rows, key=lambda row: float(row["score"]))

    summary_json = output_dir / "visibility_sweep_summary.json"
    table_csv = output_dir / "visibility_sweep_table.csv"
    report_md = output_dir / "visibility_sweep_report.md"
    figure_png = output_dir / "visibility_sweep_metrics.png"

    with table_csv.open("w", encoding="utf-8", newline="") as fh:
        fieldnames = [
            "visibility_weight",
            "topk_recall",
            "iou",
            "precision",
            "f1",
            "topk_recall_delta_vs_baseline",
            "iou_delta_vs_baseline",
            "precision_delta_vs_baseline",
            "f1_delta_vs_baseline",
            "mean_abs_density_delta",
            "rmse_density_delta",
            "score",
            "output_dir",
        ]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})

    figure_written = write_sweep_figure(figure_png, rows)
    summary = {
        "visibility_raster": str(config.visibility_raster),
        "visibility_source": config.visibility_source,
        "baseline": {
            **baseline.summary,
            "output_dir": str(baseline.output_dir),
        },
        "runs": rows,
        "best_run": best_run,
        "summary_json": str(summary_json),
        "table_csv": str(table_csv),
        "report_md": str(report_md),
        "figure_png": str(figure_png),
        "figure_written": figure_written,
    }
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown_report(report_md, summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep visibility weights for Task 1 path inference")
    parser.add_argument("--dem", type=Path, default=DEFAULT_CONFIG.dem_path)
    parser.add_argument("--sar", type=Path, default=DEFAULT_CONFIG.sar_path)
    parser.add_argument("--marks", type=Path, default=DEFAULT_CONFIG.marks_path)
    parser.add_argument("--buildings", type=Path, default=DEFAULT_CONFIG.buildings_path)
    parser.add_argument("--known-paths", type=Path, default=DEFAULT_CONFIG.known_paths_path)
    parser.add_argument("--path-prior-raster", type=Path, default=DEFAULT_CONFIG.path_prior_raster)
    parser.add_argument("--path-prior-mode", choices=["learned", "deterministic"], default=DEFAULT_CONFIG.path_prior_mode)
    parser.add_argument("--visibility-raster", type=Path, default=DEFAULT_CONFIG.visibility_raster)
    parser.add_argument("--visibility-source", choices=["deterministic", "model"], default=DEFAULT_CONFIG.visibility_source)
    parser.add_argument("--out", type=Path, default=Path("outputs_production/visibility_sweep"))
    parser.add_argument("--samples", type=int, default=DEFAULT_CONFIG.samples_per_pair)
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_CONFIG.max_pairs)
    parser.add_argument("--top-k", type=int, default=DEFAULT_CONFIG.top_k_paths)
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG.noise_temperature)
    parser.add_argument("--w-slope", type=float, default=DEFAULT_CONFIG.cost_w_slope)
    parser.add_argument("--w-roughness", type=float, default=DEFAULT_CONFIG.cost_w_roughness)
    parser.add_argument("--w-surface", type=float, default=DEFAULT_CONFIG.cost_w_surface)
    parser.add_argument("--w-path-prior", type=float, default=DEFAULT_CONFIG.cost_w_path_prior)
    parser.add_argument("--w-visibility-values", default="0.0,0.1,0.2,0.3,0.4,0.5")
    parser.add_argument("--calibrate-weights", action="store_true", default=DEFAULT_CONFIG.calibrate_weights)
    parser.add_argument("--calibration-samples", type=int, default=DEFAULT_CONFIG.calibration_samples)
    parser.add_argument("--seed", type=int, default=DEFAULT_CONFIG.rng_seed)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visibility_weights = parse_visibility_values(args.w_visibility_values)
    config = PipelineConfig(
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
        cost_w_visibility=0.0,
        calibrate_weights=args.calibrate_weights,
        compare_visibility_coupling=False,
        calibration_samples=args.calibration_samples,
        rng_seed=args.seed,
    )
    summary = evaluate_visibility_coupling(config, visibility_weights=visibility_weights, output_dir=args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
