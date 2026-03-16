#!/usr/bin/env python3
import time
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd() / "task2-viewsheds"))
sys.path.insert(0, str(Path.cwd() / "shared_utils/src"))

from src.mesh_raycast import dem_to_mesh, build_mesh_scene, mesh_is_visible
from src.raycast import compute_viewshed

def benchmark():
    # Mock data
    h, w = 100, 100
    dem = np.random.uniform(0, 100, (h, w)).astype(np.float32)
    gt = (0, 1, 0, 0, 0, -1)
    
    # Mesh setup
    t0 = time.time()
    tris = dem_to_mesh(dem, gt)
    scene = build_mesh_scene(tris)
    t_mesh_setup = time.time() - t0
    
    # Raycast benchmark
    obs = (50.5, -50.5, 10.0)
    target = (10.5, -10.5, 5.0)
    
    t0 = time.time()
    for _ in range(100):
        mesh_is_visible(scene, obs, target)
    t_mesh = (time.time() - t0) / 100
    
    # Aperture raycast benchmark
    t0 = time.time()
    for _ in range(100):
        mesh_is_visible(scene, obs, target, aperture_m=1.0, n_samples=8)
    t_mesh_aperture = (time.time() - t0) / 100
    
    # Voxel benchmark (100x100 viewshed)
    t0 = time.time()
    compute_viewshed(dem, None, 50, 50, 2.0, 0.0, 1.0, 100.0)
    t_voxel = time.time() - t0
    
    report = f"""# Raycasting Benchmark

## Setup Time
- Mesh generation and BVH build: {t_mesh_setup:.4f}s

## Performance (per query)
- Baseline Mesh LOS: {t_mesh:.6f}s
- Aperture Mesh LOS (8 samples): {t_mesh_aperture:.6f}s

## Scaling
- Voxel-based (100x100 full viewshed): {t_voxel:.4f}s
- Mesh-based is O(log N) per ray, making it superior for sparse long-range queries.
"""
    Path("RAYCASTING_BENCHMARK.md").write_text(report)
    print(report)

if __name__ == "__main__": benchmark()
