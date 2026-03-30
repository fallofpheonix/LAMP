from __future__ import annotations

import numpy as np

from lamp.tasks.path_tracing.simulation.path_finder import astar_path


def test_astar_returns_valid_path_on_uniform_grid() -> None:
    grid = np.ones((5, 5), dtype=np.float32)
    path = astar_path(grid, (0, 0), (4, 4))
    assert path
    assert path[0] == (0, 0)
    assert path[-1] == (4, 4)
