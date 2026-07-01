"""Pluggable, admissible grid heuristics.

All heuristics operate on ``(row, col)`` integer cells and return a float
estimate of the cost to travel between them. They are registered by name so
that solvers can be configured with a string (e.g. ``"manhattan"``) instead of
hard-coding a single global heuristic as the legacy ``common.HEURISTIC`` flag
did.

Coordinate convention across the framework: a cell is ``(row, col)`` which maps
to ``grid[row][col]`` (i.e. ``(y, x)``).
"""

from __future__ import annotations

from math import sqrt
from typing import Callable, Dict, Tuple

Cell = Tuple[int, int]
Heuristic = Callable[[Cell, Cell], float]

_SQRT2 = sqrt(2)


def manhattan(a: Cell, b: Cell) -> float:
    """L1 distance. Admissible for 4-connected grids with unit moves."""
    return float(abs(a[0] - b[0]) + abs(a[1] - b[1]))


def euclidean(a: Cell, b: Cell) -> float:
    """Straight-line distance. Admissible for any move set."""
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def chebyshev(a: Cell, b: Cell) -> float:
    """L-infinity distance. Admissible for 8-connected grids with unit moves."""
    return float(max(abs(a[0] - b[0]), abs(a[1] - b[1])))


def octile(a: Cell, b: Cell) -> float:
    """Octile distance. Admissible for 8-connected grids where diagonal moves
    cost ``sqrt(2)`` and orthogonal moves cost ``1``."""
    dr = abs(a[0] - b[0])
    dc = abs(a[1] - b[1])
    return (dr + dc) + (_SQRT2 - 2) * min(dr, dc)


HEURISTICS: Dict[str, Heuristic] = {
    "manhattan": manhattan,
    "euclidean": euclidean,
    "chebyshev": chebyshev,
    "octile": octile,
}


def get_heuristic(heuristic) -> Heuristic:
    """Resolve a heuristic from a name or pass through a callable.

    Accepts either a registered name (``str``) or a callable with the
    ``(Cell, Cell) -> float`` signature, so callers can supply custom
    heuristics without registering them.
    """
    if callable(heuristic):
        return heuristic
    try:
        return HEURISTICS[heuristic]
    except KeyError:
        raise ValueError(
            "Unknown heuristic %r. Available: %s"
            % (heuristic, ", ".join(sorted(HEURISTICS)))
        )
