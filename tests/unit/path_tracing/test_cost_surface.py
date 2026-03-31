from __future__ import annotations

import numpy as np

from lamp.tasks.path_tracing.simulation.cost_surface import compute_cost_surface


def test_cost_surface_blocks_obstacles() -> None:
    slope = np.zeros((3, 3), dtype=np.float32)
    roughness = np.zeros((3, 3), dtype=np.float32)
    surface = np.zeros((3, 3), dtype=np.float32)
    prior = np.ones((3, 3), dtype=np.float32)
    obstacle_mask = np.zeros((3, 3), dtype=bool)
    obstacle_mask[1, 1] = True

    cost = compute_cost_surface(
        slope,
        roughness,
        surface,
        prior,
        obstacle_mask=obstacle_mask,
    )

    assert np.isinf(cost[1, 1])
    assert np.isfinite(cost[0, 0])
