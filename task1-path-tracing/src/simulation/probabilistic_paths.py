from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

import numpy as np

from simulation.path_finder import astar_path


@dataclass
class PathRecord:
    path: list[tuple[int, int]]
    probability: float
    count: int
    base_cost: float


def _path_cost(cost: np.ndarray, path: list[tuple[int, int]]) -> float:
    """Compute the total traversal cost of *path* on *cost* surface."""
    if not path:
        return float("inf")
    total = 0.0
    for idx in range(1, len(path)):
        r0, c0 = path[idx - 1]
        r1, c1 = path[idx]
        dist = 1.41421356 if (r0 != r1 and c0 != c1) else 1.0
        total += 0.5 * (float(cost[r0, c0]) + float(cost[r1, c1])) * dist
    return total


def _worker_sample_batch(
    base_cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    n: int,
    noise_scale: float,
    seed: int,
) -> list[list[tuple[int, int]]]:
    """Worker function: draw *n* noisy A* paths and return the successful ones."""
    rng = np.random.default_rng(seed)
    finite = np.isfinite(base_cost)
    results: list[list[tuple[int, int]]] = []
    for _ in range(n):
        noise = rng.gumbel(0.0, noise_scale, size=base_cost.shape).astype(np.float32)
        noisy = np.where(finite, base_cost + noise, np.inf)
        path = astar_path(noisy, start, goal)
        if path:
            results.append(path)
    return results


def sample_probabilistic_paths(
    base_cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    samples: int,
    temperature: float,
    top_k: int,
    seed: int,
    n_workers: int | None = None,
) -> tuple[list[PathRecord], np.ndarray, int]:
    """Monte Carlo A* path sampling with optional multi-process parallelism.

    Parameters
    ----------
    base_cost:
        2-D cost surface (float32, ``np.inf`` at obstacles).
    start, goal:
        (row, col) integer grid coordinates.
    samples:
        Total number of Monte Carlo draws.
    temperature:
        Noise magnitude as a fraction of the cost surface standard deviation.
    top_k:
        Number of highest-frequency paths to keep in the output.
    seed:
        Base random seed (each worker gets a deterministic derived seed).
    n_workers:
        Number of parallel worker processes.  ``None`` → use
        ``min(os.cpu_count(), 4)`` when samples ≥ 64, else 1.

    Returns
    -------
    (records, density, successful_count)
    """
    finite = np.isfinite(base_cost)
    scale = float(np.nanstd(base_cost[finite])) if finite.any() else 0.0
    noise_scale = max(temperature * scale, 1e-6)

    # Decide worker count.
    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 4) if samples >= 64 else 1

    density = np.zeros_like(base_cost, dtype=np.float32)
    counts: dict[tuple[tuple[int, int], ...], int] = {}

    if n_workers <= 1 or samples < 4:
        # Single-process path.
        rng = np.random.default_rng(seed)
        successful = 0
        for _ in range(samples):
            noise = rng.gumbel(0.0, noise_scale, size=base_cost.shape).astype(np.float32)
            noisy = np.where(finite, base_cost + noise, np.inf)
            path = astar_path(noisy, start, goal)
            if not path:
                continue
            successful += 1
            key = tuple(path)
            counts[key] = counts.get(key, 0) + 1
            for r, c in path:
                density[r, c] += 1.0
    else:
        # Multi-process path: split samples across workers.
        # Clamp the number of workers so we never schedule more workers than samples.
        effective_workers = min(n_workers, samples)

        # Distribute samples across workers so that the total equals `samples` exactly.
        q, r = divmod(samples, effective_workers)
        batches: list[tuple[int, int]] = []
        for i in range(effective_workers):
            n = q + (1 if i < r else 0)
            if n > 0:
                batches.append((n, seed + i * 100_000))

        successful = 0
        with ProcessPoolExecutor(max_workers=effective_workers) as executor:
            futures = [
                executor.submit(
                    _worker_sample_batch,
                    base_cost,
                    start,
                    goal,
                    n,
                    noise_scale,
                    s,
                )
                for n, s in batches
            ]
            for fut in as_completed(futures):
                for path in fut.result():
                    successful += 1
                    key = tuple(path)
                    counts[key] = counts.get(key, 0) + 1
                    for r, c in path:
                        density[r, c] += 1.0

    if successful == 0:
        return [], density, 0

    ranked = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    records: list[PathRecord] = []
    for key, cnt in ranked:
        p = list(key)
        records.append(
            PathRecord(path=p, probability=cnt / successful, count=cnt, base_cost=_path_cost(base_cost, p))
        )

    density /= float(successful)
    return records, density, successful
