"""Optimal assignment: which agent takes which target.

Which agent goes where is not cosmetic. Assigning by index sends agents
across each other to targets someone else is standing next to, which is both
slower and the main source of conflicts on the way. The Hungarian algorithm
(Kuhn 1955) finds the assignment minimising the total cost exactly, in
``O(n^3)`` -- nothing at fleet sizes.

Pure standard library, like the solvers, so the planners can use it without
pulling numpy into the core.
"""

from __future__ import annotations

import math
from typing import List, Sequence

__all__ = ["hungarian", "assignment_cost"]


def hungarian(cost: Sequence[Sequence[float]]) -> List[int]:
    """Minimum-cost assignment of rows to columns.

    Args:
        cost: an ``n x m`` matrix with ``n <= m``; ``cost[i][j]`` is what it
            costs to give row ``i`` column ``j``. ``inf`` forbids a pair.

    Returns:
        ``columns`` with ``columns[i]`` the column assigned to row ``i``; every
        row gets a distinct column. When ``m > n`` the surplus columns stay
        unassigned. Raises ``ValueError`` when the only complete assignments
        use a forbidden pair, or when there are more rows than columns.

    The Jonker-Volgenant formulation of the Hungarian method, rows added one
    at a time along shortest augmenting paths.
    """
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if any(len(row) != m for row in cost):
        raise ValueError("cost matrix rows must have the same length")
    if n > m:
        raise ValueError("more rows than columns: %d agents for %d targets" % (n, m))

    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)  # p[j]: row assigned to column j (1-based), 0 if none
    for i in range(1, n + 1):
        _augment(cost, i, u, v, p)

    columns = [0] * n
    for j in range(1, m + 1):
        if p[j]:
            columns[p[j] - 1] = j - 1
    return columns


def _augment(cost, i: int, u: List[float], v: List[float], p: List[int]) -> None:
    """Assign row ``i`` along a shortest augmenting path, updating the duals
    ``u``/``v`` and the column-to-row map ``p`` in place."""
    m = len(v) - 1
    INF = math.inf
    p[0] = i
    j0 = 0
    minv = [INF] * (m + 1)
    way = [0] * (m + 1)
    used = [False] * (m + 1)
    while True:
        used[j0] = True
        i0 = p[j0]
        delta, j1 = _scan(cost[i0 - 1], u[i0], v, minv, way, used, j0)
        if delta == INF:
            raise ValueError("no complete assignment avoids a forbidden pair")
        for j in range(m + 1):
            if used[j]:
                u[p[j]] += delta
                v[j] -= delta
            else:
                minv[j] -= delta
        j0 = j1
        if p[j0] == 0:
            break
    while j0:
        j1 = way[j0]
        p[j0] = p[j1]
        j0 = j1


def _scan(row, u_i, v, minv, way, used, j0):
    """Relax the reduced costs of the unused columns from column ``j0`` and
    return the tightest one as ``(delta, column)``."""
    delta, j1 = math.inf, 0
    for j in range(1, len(v)):
        if used[j]:
            continue
        current = row[j - 1] - u_i - v[j]
        if current < minv[j]:
            minv[j] = current
            way[j] = j0
        if minv[j] < delta:
            delta, j1 = minv[j], j
    return delta, j1


def assignment_cost(cost: Sequence[Sequence[float]], columns: Sequence[int]) -> float:
    """Total cost of an assignment returned by :func:`hungarian`."""
    return sum(cost[i][j] for i, j in enumerate(columns))
