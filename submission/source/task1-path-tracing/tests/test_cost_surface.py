import numpy as np

from simulation.cost_surface import compute_cost_surface


def test_cost_surface_blocks_obstacles() -> None:
    slope = np.zeros((3, 3), dtype=np.float32)
    rough = np.zeros((3, 3), dtype=np.float32)
    surf = np.zeros((3, 3), dtype=np.float32)
    prior = np.ones((3, 3), dtype=np.float32)
    obs = np.zeros((3, 3), dtype=bool)
    obs[1, 1] = True

    cost = compute_cost_surface(slope, rough, surf, prior, obs)
    assert np.isinf(cost[1, 1])
    assert np.isfinite(cost[0, 0])
