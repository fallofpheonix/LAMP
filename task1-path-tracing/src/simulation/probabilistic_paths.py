from __future__ import annotations

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
    if not path:
        return float("inf")
    total = 0.0
    for idx in range(1, len(path)):
        r0, c0 = path[idx - 1]
        r1, c1 = path[idx]
        dist = 1.41421356 if (r0 != r1 and c0 != c1) else 1.0
        total += 0.5 * (float(cost[r0, c0]) + float(cost[r1, c1])) * dist
    return total


def sample_probabilistic_paths(
    base_cost: np.ndarray,
    start: tuple[int, int],
    goal: tuple[int, int],
    samples: int,
    temperature: float,
    top_k: int,
    seed: int,
) -> tuple[list[PathRecord], np.ndarray, int]:
    rng = np.random.default_rng(seed)

    counts: dict[tuple[tuple[int, int], ...], int] = {}
    density = np.zeros_like(base_cost, dtype=np.float32)
    successful = 0

    finite = np.isfinite(base_cost)
    scale = float(np.nanstd(base_cost[finite])) if finite.any() else 0.0
    noise_scale = max(temperature * scale, 1e-6)

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
