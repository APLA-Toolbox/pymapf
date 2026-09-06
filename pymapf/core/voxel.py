"""Three-dimensional occupancy grids, so the solvers plan for drones too.

The MAPF solvers in this library never look at a map's shape. They ask two
questions -- *is this vertex free?* and *what is adjacent to it?* -- and
:class:`VoxelGrid` answers them for a stack of layers exactly as
:class:`~pymapf.core.grid.GridMap` does for one. Every solver, the space-time
A* low level, the true-distance heuristic and the benchmark harness therefore
run on a voxel grid unchanged; a cell is ``(layer, row, col)`` and indexes
``grid[layer][row][col]``.

    grid = VoxelGrid([
        [[0, 0], [0, 0]],   # layer 0
        [[0, 1], [0, 0]],   # layer 1: one blocked voxel
    ])
    problem = pymapf.MAPFProblem(grid, [pymapf.Agent("a", (0, 0, 0), (1, 1, 1))])
    pymapf.solve(problem, "cbs")

Connectivity is 6-connected by default (the three axes, both ways). With
``allow_diagonals`` it is 26-connected, under the same corner-cutting rule the
planar grid uses, generalised: a move that changes several coordinates at once
is allowed only if every axis-aligned *part* of it is free -- the agent must be
able to slide along each face and edge of the box it is cutting through. That
rules out squeezing between two obstacles that touch only at a corner or an
edge, in any orientation.
"""

from __future__ import annotations

from itertools import product
from typing import Iterable, List, Tuple

Voxel = Tuple[int, int, int]

# The 6 face-adjacent moves, then the 20 that change two or three axes.
_FACES: List[Voxel] = [
    (-1, 0, 0),
    (1, 0, 0),
    (0, -1, 0),
    (0, 1, 0),
    (0, 0, -1),
    (0, 0, 1),
]
_DIAGONALS: List[Voxel] = [
    delta for delta in product((-1, 0, 1), repeat=3) if sum(1 for d in delta if d) >= 2
]


class VoxelGrid:
    """An immutable, explicit 3D occupancy grid.

    Args:
        voxels: a 3D structure indexable as ``voxels[layer][row][col]`` where a
            *truthy* value marks a blocked voxel. Nested lists and numpy
            arrays both work.
    """

    def __init__(self, voxels: Iterable[Iterable[Iterable]]):
        layers = [[list(row) for row in layer] for layer in voxels]
        if not layers or not layers[0] or not layers[0][0]:
            raise ValueError("voxels must be a non-empty 3D structure")
        height, width = len(layers[0]), len(layers[0][0])
        for layer in layers:
            if len(layer) != height or any(len(row) != width for row in layer):
                raise ValueError("every layer must have the same height and width")

        self.depth = len(layers)
        self.height = height
        self.width = width
        self._blocked = frozenset(
            (z, r, c)
            for z, layer in enumerate(layers)
            for r, row in enumerate(layer)
            for c, value in enumerate(row)
            if value
        )

    # -- shape ----------------------------------------------------------
    @property
    def shape(self) -> Voxel:
        return (self.depth, self.height, self.width)

    @property
    def dimension(self) -> int:
        return 3

    @property
    def free_cells(self) -> int:
        return self.depth * self.height * self.width - len(self._blocked)

    def cells(self) -> Iterable[Voxel]:
        """Every voxel, blocked or not, in ``(layer, row, col)`` order."""
        return product(range(self.depth), range(self.height), range(self.width))

    # -- queries the solvers make ---------------------------------------
    def in_bounds(self, cell: Voxel) -> bool:
        z, r, c = cell
        return 0 <= z < self.depth and 0 <= r < self.height and 0 <= c < self.width

    def is_free(self, cell: Voxel) -> bool:
        return self.in_bounds(cell) and cell not in self._blocked

    def neighbors(self, cell: Voxel, allow_diagonals: bool = False) -> List[Voxel]:
        """Free, in-bounds neighbours: the 6 faces, plus 20 diagonals if allowed.

        A diagonal is allowed only when every axis-aligned sub-move of it is
        free -- the 3D reading of "no cutting corners".
        """
        z, r, c = cell
        result = [
            (z + dz, r + dr, c + dc)
            for dz, dr, dc in _FACES
            if self.is_free((z + dz, r + dr, c + dc))
        ]
        if not allow_diagonals:
            return result
        for dz, dr, dc in _DIAGONALS:
            target = (z + dz, r + dr, c + dc)
            if not self.is_free(target):
                continue
            # Every proper, non-empty subset of the moving axes must also
            # lead to a free voxel: the faces and edges of the cut box.
            axes = [d for d in ((dz, 0, 0), (0, dr, 0), (0, 0, dc)) if any(d)]
            clear = True
            for mask in product((0, 1), repeat=len(axes)):
                if not any(mask) or all(mask):
                    continue
                probe = (z, r, c)
                for on, axis in zip(mask, axes):
                    if on:
                        probe = tuple(p + a for p, a in zip(probe, axis))
                if not self.is_free(probe):
                    clear = False
                    break
            if clear:
                result.append(target)
        return result

    def layer(self, z: int) -> List[List[int]]:
        """One layer as an occupancy grid, for drawing or ASCII rendering."""
        return [
            [0 if self.is_free((z, r, c)) else 1 for c in range(self.width)]
            for r in range(self.height)
        ]

    def __repr__(self) -> str:
        return "VoxelGrid(depth=%d, height=%d, width=%d, obstacles=%d)" % (
            self.depth,
            self.height,
            self.width,
            len(self._blocked),
        )
