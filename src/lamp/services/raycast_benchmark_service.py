from __future__ import annotations

import time
from pathlib import Path

from lamp.core.exceptions import DependencyUnavailableError
from lamp.core.models import BenchmarkResult


def run_raycast_benchmark(samples: int = 100) -> BenchmarkResult:
    try:
        import sys

        import numpy as np

        root = Path(__file__).resolve().parents[1]
        sys.path.insert(0, str(root / "shared_utils/src"))
        sys.path.insert(0, str(root / "task2-viewsheds"))
        from src.mesh_raycast import build_mesh_scene, dem_to_mesh, mesh_is_visible
        from src.raycast import compute_viewshed
    except Exception as exc:
        raise DependencyUnavailableError("task2-viewsheds raycast modules unavailable") from exc

    dem = np.random.uniform(0, 100, (100, 100)).astype(np.float32)
    geotransform = (0, 1, 0, 0, 0, -1)

    started = time.time()
    triangles = dem_to_mesh(dem, geotransform)
    scene = build_mesh_scene(triangles)
    mesh_setup_seconds = time.time() - started

    observer = (50.5, -50.5, 10.0)
    target = (10.5, -10.5, 5.0)

    started = time.time()
    for _ in range(samples):
        mesh_is_visible(scene, observer, target)
    mesh_los_seconds = (time.time() - started) / samples

    started = time.time()
    for _ in range(samples):
        mesh_is_visible(scene, observer, target, aperture_m=1.0, n_samples=8)
    mesh_aperture_seconds = (time.time() - started) / samples

    started = time.time()
    compute_viewshed(dem, None, 50, 50, 2.0, 0.0, 1.0, 100.0)
    voxel_viewshed_seconds = time.time() - started

    return BenchmarkResult(
        mesh_setup_seconds=mesh_setup_seconds,
        mesh_los_seconds=mesh_los_seconds,
        mesh_aperture_seconds=mesh_aperture_seconds,
        voxel_viewshed_seconds=voxel_viewshed_seconds,
    )


def render_benchmark_report(result: BenchmarkResult) -> str:
    return (
        "# Raycasting Benchmark\n\n"
        "## Setup Time\n"
        f"- Mesh generation and BVH build: {result.mesh_setup_seconds:.4f}s\n\n"
        "## Performance (per query)\n"
        f"- Baseline Mesh LOS: {result.mesh_los_seconds:.6f}s\n"
        f"- Aperture Mesh LOS (8 samples): {result.mesh_aperture_seconds:.6f}s\n\n"
        "## Scaling\n"
        f"- Voxel-based (100x100 full viewshed): {result.voxel_viewshed_seconds:.4f}s\n"
        "- Mesh-based is O(log N) per ray and still better for sparse long-range queries.\n"
    )
