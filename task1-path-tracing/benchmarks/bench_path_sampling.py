#!/usr/bin/env python3
"""
Benchmark: Monte Carlo path sampling runtime vs raster size.

Usage
-----
    PYTHONPATH=src python benchmarks/bench_path_sampling.py

Measures wall-clock time and peak memory for increasing DEM grid sizes and
a fixed number of MC samples, then prints a summary table.
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from simulation.cost_surface import compute_cost_surface
from simulation.probabilistic_paths import sample_probabilistic_paths


def _synthetic_cost(rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    slope = rng.uniform(0.0, 0.8, (rows, cols)).astype(np.float32)
    rough = rng.uniform(0.0, 0.5, (rows, cols)).astype(np.float32)
    surf = rng.uniform(0.0, 0.3, (rows, cols)).astype(np.float32)
    prior = rng.uniform(0.0, 1.0, (rows, cols)).astype(np.float32)
    return compute_cost_surface(slope, rough, surf, prior)


def benchmark_path_sampling(
    grid_sizes: list[int],
    samples: int = 64,
    top_k: int = 4,
    temperature: float = 0.1,
    seed: int = 0,
) -> list[dict]:
    rng = np.random.default_rng(seed)
    results = []

    for n in grid_sizes:
        cost = _synthetic_cost(n, n, rng)
        start_pt = (2, 2)
        goal_pt = (n - 3, n - 3)

        tracemalloc.start()
        t0 = time.perf_counter()
        recs, density, succ = sample_probabilistic_paths(
            base_cost=cost,
            start=start_pt,
            goal=goal_pt,
            samples=samples,
            temperature=temperature,
            top_k=top_k,
            seed=seed,
            n_workers=1,  # Single-process for reproducible benchmark.
        )
        elapsed = time.perf_counter() - t0
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append(
            {
                "grid": f"{n}×{n}",
                "cells": n * n,
                "samples": samples,
                "successful": succ,
                "elapsed_s": round(elapsed, 4),
                "peak_kb": round(peak_kb / 1024, 1),
            }
        )
        print(
            f"  {n:4d}×{n:<4d}  elapsed={elapsed:.3f}s  "
            f"peak={peak_kb/1024:.0f} KiB  succ={succ}/{samples}"
        )

    return results


def main() -> None:
    print("=" * 60)
    print("Task-1 Path Sampling Benchmark")
    print("=" * 60)
    grid_sizes = [32, 64, 128, 256]
    benchmark_path_sampling(grid_sizes=grid_sizes, samples=32)
    print("Done.")


if __name__ == "__main__":
    main()
