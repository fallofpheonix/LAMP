from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import rasterize

from lamp.tasks.path_tracing.simulation.cost_surface import compute_cost_surface
from lamp.tasks.path_tracing.simulation.probabilistic_paths import PathRecord, sample_probabilistic_paths


@dataclass
class CalibrationResult:
    weights: tuple[float, float, float, float, float]
    topk_recall: float
    iou: float
    precision: float
    f1: float


def rasterize_known_paths(
    known_paths_path: Path | str,
    out_shape: tuple[int, int],
    transform: rasterio.Affine,
    crs: object,
) -> np.ndarray:
    gdf = gpd.read_file(known_paths_path)
    if gdf.crs != crs:
        gdf = gdf.to_crs(crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None and not geom.is_empty]
    if not shapes:
        return np.zeros(out_shape, dtype=np.uint8)
    mask = rasterize(shapes, out_shape=out_shape, transform=transform, fill=0, default_value=1, dtype="uint8")
    return mask


def _records_topk_mask(records: list[tuple[int, int, PathRecord]], shape: tuple[int, int], top_k: int) -> np.ndarray:
    grouped: dict[tuple[int, int], list[PathRecord]] = {}
    for src_idx, dst_idx, rec in records:
        grouped.setdefault((src_idx, dst_idx), []).append(rec)

    mask = np.zeros(shape, dtype=np.uint8)
    for _, recs in grouped.items():
        ranked = sorted(recs, key=lambda r: r.probability, reverse=True)[:top_k]
        for rec in ranked:
            for r, c in rec.path:
                mask[r, c] = 1
    return mask


def _metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> tuple[float, float, float, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(recall), float(iou), float(precision), float(f1)


def evaluate_topk_metrics(
    records: list[tuple[int, int, PathRecord]],
    shape: tuple[int, int],
    gt_mask: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, dict]:
    pred_mask = _records_topk_mask(records, shape, top_k=top_k)
    recall, iou, precision, f1 = _metrics(pred_mask, gt_mask)
    return pred_mask, {
        "topk_recall": recall,
        "iou": iou,
        "precision": precision,
        "f1": f1,
    }


def _normalize_weights(
    weights: tuple[float, float, float, float] | tuple[float, float, float, float, float]
) -> tuple[float, float, float, float, float]:
    arr = np.asarray(weights, dtype=np.float32)
    if arr.shape == (4,):
        arr = np.concatenate([arr, np.array([0.0], dtype=np.float32)])
    if arr.shape != (5,):
        raise ValueError("weights must be a 4- or 5-tuple")
    arr = np.clip(arr, 0.0, None)
    if float(arr.sum()) <= 0.0:
        raise ValueError("weights must have positive sum")
    arr /= float(arr.sum())
    return tuple(float(v) for v in arr)


def default_weight_grid(
    base_weights: tuple[float, float, float, float] | tuple[float, float, float, float, float],
    *,
    enable_visibility_search: bool,
) -> list[tuple[float, float, float, float, float]]:
    base = _normalize_weights(base_weights)
    if not enable_visibility_search:
        return [base]

    core = np.asarray(base[:4], dtype=np.float32)
    core_sum = float(core.sum())
    if core_sum <= 0.0:
        core = np.asarray([0.55, 0.30, 0.10, 0.05], dtype=np.float32)
        core_sum = float(core.sum())

    candidates: dict[tuple[float, float, float, float, float], None] = {}
    coarse_values = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5, round(base[4], 2)}
    for visibility_weight in sorted(coarse_values):
        visibility_weight = float(np.clip(visibility_weight, 0.0, 0.95))
        scaled_core = core / core_sum * (1.0 - visibility_weight)
        candidate = tuple(float(v) for v in scaled_core) + (visibility_weight,)
        candidates[_normalize_weights(candidate)] = None
    return list(candidates.keys())


def calibrate_weights(
    slope_norm: np.ndarray,
    roughness: np.ndarray,
    surface_penalty: np.ndarray,
    path_prior: np.ndarray,
    visibility_probability: np.ndarray | None,
    obstacle_mask: np.ndarray,
    terminals: list[tuple[int, int]],
    known_path_mask: np.ndarray,
    samples_per_pair: int,
    top_k: int,
    rng_seed: int,
    temperature: float,
    weight_candidates: list[tuple[float, float, float, float, float]],
) -> tuple[tuple[float, float, float, float, float], list[CalibrationResult]]:
    all_pairs = list(combinations(range(len(terminals)), 2))
    if not all_pairs:
        raise RuntimeError("Need at least one terminal pair for calibration")

    results: list[CalibrationResult] = []
    best_w = weight_candidates[0]
    best_score = -1.0

    for w_idx, weights in enumerate(weight_candidates):
        cost = compute_cost_surface(
            slope_norm,
            roughness,
            surface_penalty,
            path_prior,
            visibility_probability=visibility_probability,
            obstacle_mask=obstacle_mask,
            weights=weights,
        )

        all_records: list[tuple[int, int, PathRecord]] = []
        for p_idx, (src_idx, dst_idx) in enumerate(all_pairs):
            recs, _, _ = sample_probabilistic_paths(
                base_cost=cost,
                start=terminals[src_idx],
                goal=terminals[dst_idx],
                samples=samples_per_pair,
                temperature=temperature,
                top_k=top_k,
                seed=rng_seed + 1000 * w_idx + p_idx,
            )
            for rec in recs:
                all_records.append((src_idx, dst_idx, rec))

        pred_mask = _records_topk_mask(all_records, cost.shape, top_k=top_k)
        recall, iou, precision, f1 = _metrics(pred_mask, known_path_mask)
        results.append(CalibrationResult(weights=weights, topk_recall=recall, iou=iou, precision=precision, f1=f1))

        score = 0.65 * iou + 0.35 * recall
        if score > best_score:
            best_score = score
            best_w = weights

    # Optional local refinement around the strongest visibility weight.
    best_visibility_weight = best_w[4]
    if visibility_probability is not None and best_visibility_weight > 0.0:
        local_candidates: list[tuple[float, float, float, float, float]] = []
        core = np.asarray(best_w[:4], dtype=np.float32)
        core_sum = float(core.sum()) or 1.0
        for visibility_weight in (best_visibility_weight - 0.05, best_visibility_weight + 0.05):
            if not (0.0 <= visibility_weight <= 0.95):
                continue
            scaled_core = core / core_sum * (1.0 - visibility_weight)
            candidate = _normalize_weights(tuple(float(v) for v in scaled_core) + (float(visibility_weight),))
            if candidate in weight_candidates or candidate in local_candidates:
                continue
            local_candidates.append(candidate)

        for local_idx, weights in enumerate(local_candidates, start=len(weight_candidates)):
            cost = compute_cost_surface(
                slope_norm,
                roughness,
                surface_penalty,
                path_prior,
                visibility_probability=visibility_probability,
                obstacle_mask=obstacle_mask,
                weights=weights,
            )

            all_records: list[tuple[int, int, PathRecord]] = []
            for p_idx, (src_idx, dst_idx) in enumerate(all_pairs):
                recs, _, _ = sample_probabilistic_paths(
                    base_cost=cost,
                    start=terminals[src_idx],
                    goal=terminals[dst_idx],
                    samples=samples_per_pair,
                    temperature=temperature,
                    top_k=top_k,
                    seed=rng_seed + 1000 * local_idx + p_idx,
                )
                for rec in recs:
                    all_records.append((src_idx, dst_idx, rec))

            pred_mask = _records_topk_mask(all_records, cost.shape, top_k=top_k)
            recall, iou, precision, f1 = _metrics(pred_mask, known_path_mask)
            results.append(CalibrationResult(weights=weights, topk_recall=recall, iou=iou, precision=precision, f1=f1))

            score = 0.65 * iou + 0.35 * recall
            if score > best_score:
                best_score = score
                best_w = weights

    results.sort(key=lambda r: (r.iou, r.topk_recall), reverse=True)
    return best_w, results
