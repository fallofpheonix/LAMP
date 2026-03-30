from __future__ import annotations

import heapq
from typing import Iterable

import numpy as np


def _neighbors(r: int, c: int, rows: int, cols: int, diag: bool = True) -> Iterable[tuple[int, int, float]]:
    if diag:
        dirs = [
            (-1, 0, 1.0),
            (1, 0, 1.0),
            (0, -1, 1.0),
            (0, 1, 1.0),
            (-1, -1, 1.41421356),
            (-1, 1, 1.41421356),
            (1, -1, 1.41421356),
            (1, 1, 1.41421356),
        ]
    else:
        dirs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0)]

    for dr, dc, d in dirs:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < rows and 0 <= cc < cols:
            yield rr, cc, d


def _heuristic(a: tuple[int, int], b: tuple[int, int], min_cost: float) -> float:
    return (((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5) * min_cost


def astar_path(cost: np.ndarray, start: tuple[int, int], goal: tuple[int, int], max_expansions: int = 1_000_000) -> list[tuple[int, int]]:
    rows, cols = cost.shape
    sr, sc = start
    gr, gc = goal
    if not (0 <= sr < rows and 0 <= sc < cols and 0 <= gr < rows and 0 <= gc < cols):
        return []
    if not np.isfinite(cost[sr, sc]) or not np.isfinite(cost[gr, gc]):
        return []

    finite = np.isfinite(cost)
    if not finite.any():
        return []
    min_cost = float(np.nanmin(cost[finite]))

    open_heap: list[tuple[float, float, tuple[int, int]]] = []
    heapq.heappush(open_heap, (0.0, 0.0, start))

    g_score: dict[tuple[int, int], float] = {start: 0.0}
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    expanded = 0

    while open_heap:
        _, g_curr, curr = heapq.heappop(open_heap)
        if curr == goal:
            break
        expanded += 1
        if expanded > max_expansions:
            return []

        r, c = curr
        for nr, nc, dist in _neighbors(r, c, rows, cols, diag=True):
            if not np.isfinite(cost[nr, nc]):
                continue
            step = 0.5 * (float(cost[r, c]) + float(cost[nr, nc])) * dist
            tentative = g_curr + step
            nxt = (nr, nc)
            if tentative < g_score.get(nxt, float("inf")):
                g_score[nxt] = tentative
                came_from[nxt] = curr
                f = tentative + _heuristic(nxt, goal, min_cost)
                heapq.heappush(open_heap, (f, tentative, nxt))

    if goal not in came_from and goal != start:
        return []

    path = [goal]
    cur = goal
    while cur != start:
        cur = came_from[cur]
        path.append(cur)
    path.reverse()
    return path
