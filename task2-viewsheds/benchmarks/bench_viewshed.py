#!/usr/bin/env python3
"""
Benchmark: viewshed LOS computation runtime vs raster size.

Usage
-----
    PYTHONPATH=src python benchmarks/bench_viewshed.py

Measures wall-clock time for increasing surface sizes (single observer,
no multi-process) and prints a summary table.
"""
from __future__ import annotations

import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.raycast import compute_viewshed


def _flat_surface(n: int) -> np.ndarray:
    rng = np.random.default_rng(42)
    return 100.0 + rng.uniform(0.0, 5.0, (n, n))


def benchmark_viewshed(
    grid_sizes: list[int],
    observer_height: float = 1.6,
) -> list[dict]:
    results = []

    for n in grid_sizes:
        surf = _flat_surface(n)
        obs_r, obs_c = n // 2, n // 2

        tracemalloc.start()
        t0 = time.perf_counter()
        vis, stats = compute_viewshed(
            surface=surf,
            nodata=None,
            obs_row=obs_r,
            obs_col=obs_c,
            observer_height=observer_height,
            pixel_size_m=1.0,
        )
        elapsed = time.perf_counter() - t0
        _, peak_kb = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        results.append(
            {
                "grid": f"{n}×{n}",
                "cells": n * n,
                "visible": int(stats["visible_cells"]),
                "elapsed_s": round(elapsed, 4),
                "peak_kb": round(peak_kb / 1024, 1),
            }
        )
        print(
            f"  {n:4d}×{n:<4d}  elapsed={elapsed:.3f}s  "
            f"peak={peak_kb/1024:.0f} KiB  visible={stats['visible_cells']}"
        )

    return results


def main() -> None:
    print("=" * 60)
    print("Task-2 Viewshed LOS Benchmark")
    print("=" * 60)
    grid_sizes = [32, 64, 128]
    benchmark_viewshed(grid_sizes=grid_sizes)
    print("Done.")


if __name__ == "__main__":
    main()
