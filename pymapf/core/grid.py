"""Deterministic grid map used by the MAPF solvers.

Unlike :class:`pymapf.centralized.world.World` -- which randomly scatters walls
on construction and is therefore unsuitable for reproducible planning or tests
-- :class:`GridMap` is built from an explicit occupancy grid. This makes it
possible to describe a specific scenario, which is a prerequisite for a usable
planning framework.
"""

from __future__ import annotations

from typing import Iterable, List, Tuple

Cell = Tuple[int, int]

# 4-connected orthogonal moves, then the 4 diagonals.
_ORTHOGONAL = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIAGONAL = [(-1, -1), (-1, 1), (1, -1), (1, 1)]


class GridMap:
    """An immutable, explicit occupancy grid.

    Args:
        grid: a 2D structure indexable as ``grid[row][col]`` where a *truthy*
            value marks a blocked cell and a *falsy* value marks a free cell.
            Works with nested lists as well as numpy arrays.
    """

    def __init__(self, grid: Iterable[Iterable]):
        rows = [list(row) for row in grid]
        if not rows or not rows[0]:
            raise ValueError("grid must be a non-empty 2D structure")
        width = len(rows[0])
        if any(len(row) != width for row in rows):
            raise ValueError("all grid rows must have the same length")

        self.height = len(rows)
        self.width = width
        # Store only the blocked cells; grids are usually mostly free so a set
        # of obstacles is compact and gives O(1) membership tests.
        self._blocked = frozenset(
            (r, c)
            for r, row in enumerate(rows)
            for c, value in enumerate(row)
            if value
        )

    @classmethod
    def from_world(cls, world) -> "GridMap":
        """Build a :class:`GridMap` from a legacy ``World`` (walls are cells
        whose grid value equals ``1``)."""
        grid = [
            [1 if world.grid[r][c] == 1 else 0 for c in range(world.length)]
            for r in range(world.height)
        ]
        return cls(grid)

    def in_bounds(self, cell: Cell) -> bool:
        r, c = cell
        return 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, cell: Cell) -> bool:
        return self.in_bounds(cell) and cell not in self._blocked

    @property
    def free_cells(self) -> int:
        return self.height * self.width - len(self._blocked)

    def neighbors(self, cell: Cell, allow_diagonals: bool = False) -> List[Cell]:
        """Return the free, in-bounds spatial neighbors of ``cell``.

        Diagonal moves are only allowed when both orthogonal cells they "cut
        the corner" of are also free, which prevents agents from squeezing
        between two walls.
        """
        r, c = cell
        result = []
        for dr, dc in _ORTHOGONAL:
            n = (r + dr, c + dc)
            if self.is_free(n):
                result.append(n)
        if allow_diagonals:
            for dr, dc in _DIAGONAL:
                n = (r + dr, c + dc)
                if self.is_free(n) and self.is_free((r + dr, c)) and self.is_free((r, c + dc)):
                    result.append(n)
        return result

    def __repr__(self) -> str:
        return "GridMap(height=%d, width=%d, obstacles=%d)" % (
            self.height,
            self.width,
            len(self._blocked),
        )
